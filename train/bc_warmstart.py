#!/usr/bin/env python3
"""行为克隆预热：先把"经典门限策略"蒸馏进 MLP，再用 PPO 微调。

为什么需要（DESIGN.md §3.0.2 的风险预案，现在被实测逼出来了）：
从随机权重开始跑 PPO，策略会在"打/不打"之间反复横跳 —— 实测平均回报从 22.5
一路掉到 19.2，checkpoint 命中率 36% → 33%。原因是奖励极稀疏（每步只有 2.5% 是"开火"），
冷启动要先学会"什么时候该开火"这件本来就有人会做的事。

做法：
  1. 用镜像的经典门限策略（env/policies.py，等价于 sp_vision 现役 shooter）跑若干回合，
     记录 (观测, 动作)；
  2. 只用 actor 头做**加权交叉熵**（正样本只占 ~2.5%，不加权会直接学成"永不开火"）；
  3. 存成与 ppo.py 完全相同的 policy.pt 格式 → ppo.py --init-from 接着练。

用法:
    python auto_aim_rl/train/bc_warmstart.py --episodes 8 --episode-s 40 \
        --out auto_aim_rl/runs/stage1_a/bc.pt
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_ENV = os.path.join(os.path.dirname(_HERE), "env")
sys.path.insert(0, _ENV)

from policies import ThresholdPolicy  # noqa: E402
from sim_env import OBS_DIM, FireEnv  # noqa: E402
from ppo import Policy  # noqa: E402


def collect(episodes: int, episode_s: float, seed0: int, randomize: bool):
    obs_buf, act_buf = [], []
    teacher = ThresholdPolicy()
    for i in range(episodes):
        env = FireEnv(seed=seed0 + i, episode_s=episode_s, randomize=randomize)
        obs, _ = env.reset()
        while True:
            a = teacher(obs)
            obs_buf.append(obs)
            act_buf.append(a)
            obs, _r, term, trunc, _info = env.step(a)
            if term or trunc:
                break
        print("  回合 %d：%d 步，开火 %d 次（%.1f%%）" % (
            i, len(act_buf), sum(act_buf), 100.0 * sum(act_buf) / max(len(act_buf), 1)),
            flush=True)
    X = torch.as_tensor(np.asarray(obs_buf), dtype=torch.float32)
    y = torch.as_tensor(np.asarray(act_buf), dtype=torch.long)
    return X, y


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--episode-s", type=float, default=40.0)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--randomize", action="store_true", default=True)
    # 类别权重直接决定"学生开火比老师多还是少"：均衡权重（balanced）会让学生开火率
    # 翻倍（实测 15% → 31%），虽然"与老师一致率"更高，但多开的那部分基本都是错的；
    # 只有真正决定好坏的是**闭环命中率**，所以三种权重都留着手动对比。
    ap.add_argument("--class-weight", choices=["none", "sqrt", "balanced"], default="sqrt")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    print("收集示范数据（经典门限策略）…")
    X, y = collect(args.episodes, args.episode_s, 1000 + args.seed, args.randomize)
    n_pos = int(y.sum()); n_neg = int(len(y) - n_pos)
    print("样本 %d：开火 %d（%.1f%%）/ 不开火 %d" % (len(y), n_pos, 100.0 * n_pos / len(y), n_neg))
    if n_pos == 0:
        sys.exit("示范里一次都没开火 —— 门限策略或观测有问题，先查这个")

    # 类别加权：不加权的话正样本（~15%）容易被淹没（极端情况学成"永不开火"）；
    # 加权过头又会让学生"见谁打谁"。sqrt 取中间。
    if args.class_weight == "none":
        w = torch.tensor([1.0, 1.0], dtype=torch.float32)
    else:
        w = torch.tensor([1.0 / max(n_neg, 1), 1.0 / max(n_pos, 1)], dtype=torch.float32)
        if args.class_weight == "sqrt":
            w = w.sqrt()
        w = w / w.sum() * 2.0
    print("类别权重：%s -> %s" % (args.class_weight, [round(float(x), 4) for x in w]))

    net = Policy(obs_dim=OBS_DIM)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    lossf = nn.CrossEntropyLoss(weight=w)
    for ep in range(args.epochs):
        logits, _ = net(X)
        loss = lossf(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 50 == 0:
            with torch.no_grad():
                pred = torch.argmax(logits, dim=-1)
                acc = float((pred == y).float().mean())
                fire_rate = float(pred.float().mean())
            print("  epoch %3d  loss %.4f  与老师一致率 %.1f%%  开火率 %.1f%%（老师 %.1f%%）"
                  % (ep + 1, float(loss), 100 * acc, 100 * fire_rate, 100.0 * n_pos / len(y)),
                  flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({"model": net.state_dict(),
                "args": {"from": "bc_warmstart", "episodes": args.episodes,
                         "episode_s": args.episode_s, "epochs": args.epochs}},
               args.out)
    print("已保存：%s（接 ppo.py --init-from 继续练）" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
