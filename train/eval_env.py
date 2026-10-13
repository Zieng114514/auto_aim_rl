#!/usr/bin/env python3
"""在 **Python 镜像环境**里快速评测一个策略（秒级），用于训练期间迭代。

为什么需要它：C++ 闭环评测一次 3 种子 × 300 s 要 ~20 分钟；而镜像与 C++ 的**排序**
一致（镜像偏乐观 ~10 个点，见 agnet_experience.md §1.16），所以调参阶段先用它筛，
**最终数字仍然以 C++ 闭环为准**。

用法:
    python auto_aim_rl/train/eval_env.py --ckpt auto_aim_rl/runs/stage5/policy.pt
    python auto_aim_rl/train/eval_env.py --policy threshold     # 规则门限（对照组）
    python auto_aim_rl/train/eval_env.py --policy spam          # 每帧开火（下界参考）
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ENV = os.path.join(os.path.dirname(_HERE), "env")
sys.path.insert(0, _ENV)
sys.path.insert(0, _HERE)

from policies import ThresholdPolicy  # noqa: E402
from sim_env import OBS_DIM, FireEnv  # noqa: E402
from ppo import Policy  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--policy", choices=["net", "threshold", "spam"], default="net")
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--episode-s", type=float, default=90.0)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--c-fire", type=float, default=0.10)
    ap.add_argument("--c-waste", type=float, default=0.0)
    args = ap.parse_args()

    if args.policy == "net":
        if not args.ckpt:
            sys.exit("--policy net 需要 --ckpt")
        ck = torch.load(args.ckpt, map_location="cpu")
        net = Policy(obs_dim=OBS_DIM)
        net.load_state_dict(ck["model"] if "model" in ck else ck)
        net.eval()
        act = (lambda o: int(torch.distributions.Categorical(
            logits=net(torch.as_tensor(o[None], dtype=torch.float32))[0]).sample().item())) \
            if args.stochastic else \
            (lambda o: int(torch.argmax(
                net(torch.as_tensor(o[None], dtype=torch.float32))[0], dim=-1).item()))
    elif args.policy == "threshold":
        act = ThresholdPolicy()
    else:
        act = lambda o: 1

    seeds = args.seeds or [100 + i for i in range(args.episodes)]
    tot_s = tot_h = tot_k = tot_block = 0
    tot_steps = tot_waste = 0
    rows = []
    for sd in seeds:
        env = FireEnv(seed=sd, episode_s=args.episode_s, randomize=False,
                      c_fire=args.c_fire, c_waste=args.c_waste)
        obs, _ = env.reset()
        waste = 0
        while True:
            a = act(obs)
            obs, r, term, trunc, info = env.step(a)
            g = env.world.observe()["gun"]
            if g.get("heat_frac", 0.0) >= 0.995:
                waste += 1
            tot_steps += 1
            if term or trunc:
                break
        tot_waste += waste
        g = env.world.observe()["gun"]
        rows.append((sd, info["shots"], info["hits"], info["deaths"], g["shots_blocked"]))
        tot_s += info["shots"]; tot_h += info["hits"]; tot_k += info["deaths"]
        tot_block += g["shots_blocked"]

    print("%-10s %8s %8s %8s %8s %10s" % ("种子", "发射", "命中", "命中率", "击杀", "被挡"))
    for sd, s, h, k, b in rows:
        print("%-10d %8d %8d %7.1f%% %8d %10d" % (sd, s, h, 100.0 * h / max(s, 1), k, b))
    print("合计：%d 发 / %d 中 = **%.1f%%**，击杀 %d 台，每击杀 %.0f 发"
          " | 被挡 %d | 热量堵上限 %.0f%% 的步数" % (
              tot_s, tot_h, 100.0 * tot_h / max(tot_s, 1), tot_k,
              tot_s / max(tot_k, 1), tot_block, 100.0 * tot_waste / max(tot_steps, 1)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
