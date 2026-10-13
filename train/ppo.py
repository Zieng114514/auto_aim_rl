#!/usr/bin/env python3
"""PPO 训练阶段 1（只学开火）—— CleanRL 风格的单文件实现。

为什么自己写而不是直接上 rsl_rl：环境是**单个离散动作 + 22 维标量观测**的小问题，
单文件 PPO 更容易看清每一行在干什么、也便于按需改动（比如加行为克隆预热）。
跑通、确认能学之后再考虑换 rsl_rl（见 auto_aim_rl/DESIGN.md §4）。

并行：用 gymnasium 的 AsyncVectorEnv（**多进程**）。实测多线程几乎不扩展
（mj_step 不放 GIL）：1 线程 93 Hz、4 线程合计只有 120 Hz；多进程才能吃满 12 核。

用法：
    python auto_aim_rl/train/ppo.py --steps 200000 --envs 6 --out runs/stage1_a
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_ENV = os.path.join(os.path.dirname(_HERE), "env")
if _ENV not in sys.path:
    sys.path.insert(0, _ENV)

from sim_env import OBS_DIM, FireEnv  # noqa: E402


def make_env(seed: int, episode_s: float, randomize: bool, c_fire: float = 0.10,
             c_block: float = 0.0, c_waste: float = 0.0):
    def _f():
        return FireEnv(seed=seed, episode_s=episode_s, randomize=randomize,
                       c_fire=c_fire, c_block=c_block, c_waste=c_waste)
    return _f


class Policy(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, hidden: int = 64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh())
        self.actor = nn.Linear(hidden, 2)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.body(x)
        return self.actor(h), self.critic(h).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200_000, help="总环境步数")
    ap.add_argument("--envs", type=int, default=6, help="并行环境数（进程）")
    ap.add_argument("--rollout", type=int, default=1024, help="每个环境每次采多少步")
    ap.add_argument("--episode-s", type=float, default=90.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--target-kl", type=float, default=0.02, help="KL 早停阈值（超过 1.5x 就停这一轮）")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    # 每发弹的代价。**这是决定策略性格的关键旋钮**：
    #   * 0.10（默认）：只要 P(命中) > 10% 就值得打 → 实测学成"热量一好就开火"（spam），
    #     开火率 74%，把热量全花在低概率时机上；
    #   * 0.30：只有 P(命中) > 30% 才值得 → 逼策略在"移动靶"这类低概率段**攒热量**，
    #     留给"原地甩头"这类高概率段爆发（热量上限 100、回热 10/s，攒得住）。
    #   因为靶型每 10s 一换，攒热量对**命中率和击杀时间同时有利**。
    ap.add_argument("--c-fire", type=float, default=0.10, help="每发弹的奖励代价")
    # 被锁枪挡下的请求的代价（shaping）：不加这一条，策略最优解就是"一直请求"，
    # 学不会挑时机。0.005 ≈ 锁枪期间 0.15 分/s，相对"一发命中 = 1.0"是合适的量级。
    ap.add_argument("--c-block", type=float, default=0.0, help="被锁枪挡下的开火请求的代价")
    # 热量顶到上限（回热被浪费）每步的代价 —— **"择时"这个行为唯一的经济来源**。
    # 没有它，最优解就是"有热量就打"（实测开火请求占 74% 决策步），
    # 策略永远学不会在每秒 30 个决策格里挑最值得打的那一格。
    ap.add_argument("--c-waste", type=float, default=0.0, help="热量堵在上限每步的代价")
    ap.add_argument("--randomize", action="store_true", help="域随机化（推荐）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(_HERE), "runs", "stage1"))
    # 断点续训（只接权重，不接优化器状态）：换环境代码/加长训练时不用从零开始。
    # 例：环境热路径优化后重跑 —— 物理逐位一致，接着练就行。
    ap.add_argument("--init-from", default=None, help="从 policy.pt 初始化权重（续训）")
    args = ap.parse_args()

    import gymnasium as gym
    envs = gym.vector.AsyncVectorEnv(
        [make_env(args.seed + i, args.episode_s, args.randomize, args.c_fire, args.c_block,
                  args.c_waste)
         for i in range(args.envs)])
    obs, _ = envs.reset(seed=[args.seed + i for i in range(args.envs)])

    torch.manual_seed(args.seed)
    net = Policy()
    if args.init_from:
        ck = torch.load(args.init_from, map_location="cpu")
        net.load_state_dict(ck["model"] if "model" in ck else ck)
        print("从 %s 续训" % args.init_from, flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)

    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "log.jsonl")
    log_f = open(log_path, "a")
    n_updates = max(1, args.steps // (args.envs * args.rollout))
    ep_returns = np.zeros(args.envs)
    ep_hits = np.zeros(args.envs); ep_shots = np.zeros(args.envs)
    finished = []
    t0 = time.perf_counter()
    total_steps = 0

    for upd in range(n_updates):
        ob = torch.as_tensor(obs, dtype=torch.float32)
        b_obs, b_act, b_logp, b_rew, b_val, b_done = [], [], [], [], [], []
        for _ in range(args.rollout):
            with torch.no_grad():
                logits, value = net(ob)
                dist = torch.distributions.Categorical(logits=logits)
                act = dist.sample()
                logp = dist.log_prob(act)
            nobs, rew, term, trunc, info = envs.step(act.numpy())
            done = np.logical_or(term, trunc)
            b_obs.append(ob); b_act.append(act); b_logp.append(logp)
            b_rew.append(torch.as_tensor(rew, dtype=torch.float32))
            b_val.append(value); b_done.append(torch.as_tensor(done, dtype=torch.float32))
            ep_returns += rew
            if "hits" in info:
                ep_hits += np.asarray(info["hits"], dtype=float) - 0  # 累计值由 env 内部维护
            ob = torch.as_tensor(nobs, dtype=torch.float32)
            total_steps += args.envs
            for i in np.nonzero(done)[0]:
                finished.append({"return": float(ep_returns[i])})
                ep_returns[i] = 0.0
        # 最后一步的自举
        with torch.no_grad():
            _, last_val = net(ob)
        b_obs = torch.stack(b_obs); b_act = torch.stack(b_act); b_logp = torch.stack(b_logp)
        b_rew = torch.stack(b_rew); b_val = torch.stack(b_val); b_done = torch.stack(b_done)
        adv = torch.zeros_like(b_rew); last_gae = 0.0
        for t in reversed(range(args.rollout)):
            next_val = last_val if t == args.rollout - 1 else b_val[t + 1]
            next_nonterm = 1.0 - (b_done[t] if t == args.rollout - 1 else b_done[t + 1])
            delta = b_rew[t] + args.gamma * next_val * next_nonterm - b_val[t]
            last_gae = delta + args.gamma * args.lam * next_nonterm * last_gae
            adv[t] = last_gae
        ret = adv + b_val
        f_obs = b_obs.reshape(-1, OBS_DIM); f_act = b_act.reshape(-1)
        f_logp = b_logp.reshape(-1); f_adv = adv.reshape(-1); f_ret = ret.reshape(-1)
        f_adv = (f_adv - f_adv.mean()) / (f_adv.std() + 1e-8)
        n = f_obs.shape[0]; mb = n // args.minibatches
        idx = np.arange(n)
        # ---- PPO 更新：带 **KL 早停** ----------------------------------------
        # 不早停的话，一轮 rollout 会被反复过拟合（6 epoch × 4 minibatch = 24 步），
        # 实测策略会"越练越差"：平均回报从 22.5 一路掉到 19.2，checkpoint 的
        # 命中率也从 36% 掉到 33%。加上 KL 早停 + 降学习率后曲线才稳。
        stop_epoch = False
        for ep in range(args.epochs):
            np.random.shuffle(idx)
            for s in range(0, n, mb):
                j = torch.as_tensor(idx[s:s + mb])
                logits, val = net(f_obs[j])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(f_act[j])
                with torch.no_grad():
                    approx_kl = float((f_logp[j] - logp).mean())
                if approx_kl > 1.5 * args.target_kl:
                    stop_epoch = True
                    break
                ratio = torch.exp(logp - f_logp[j])
                pg = torch.min(ratio * f_adv[j],
                               torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * f_adv[j])
                ent = dist.entropy().mean()
                loss = -pg.mean() + 0.5 * ((val - f_ret[j]) ** 2).mean() - args.ent_coef * ent
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
            if stop_epoch:
                break
        if (upd + 1) % 5 == 0 or upd == n_updates - 1:
            recent = [f["return"] for f in finished[-20:]]
            msg = {"upd": upd + 1, "steps": total_steps,
                   "wall_s": round(time.perf_counter() - t0, 1),
                   "mean_return": round(float(np.mean(recent)), 2) if recent else None,
                   "episodes": len(finished)}
            log_f.write(json.dumps(msg) + "\n"); log_f.flush()
            # 周期性存盘：一是防中途崩掉白跑，二是**联调 C++ 的 ShooterRL 时不用等
            # 训练跑完**（ONNX 导出脚本读的就是这个文件）。
            torch.save({"model": net.state_dict(), "args": vars(args)},
                       os.path.join(args.out, "policy.pt"))
            print("  upd %3d/%d  步数 %7d  %.0fs  平均回报 %s（%d 回合）"
                  % (msg["upd"], n_updates, msg["steps"], msg["wall_s"],
                     msg["mean_return"], msg["episodes"]), flush=True)

    ckpt = os.path.join(args.out, "policy.pt")
    torch.save({"model": net.state_dict(), "args": vars(args)}, ckpt)
    print("已保存：%s" % ckpt)
    envs.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
