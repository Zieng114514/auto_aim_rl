#!/usr/bin/env python3
"""同一批 C++ 实数上，比较"规则门限"与"逻辑回归门限"的 (射频, 命中率) 前沿。

关键问题：在热量约束（可持续 ~1 发/s）下，**命中率 × 射频**才是输出能力。
规则门限靠"卡 0.5°"只打 0.65 发/s；逻辑回归能不能在同样的射频下打得更好，
或者用更高的射频维持同样的命中率？这张表就是答案。

用法: python auto_aim_rl/train/frontier.py /tmp/shooter_rl_log.csv
"""
from __future__ import annotations

import math
import sys

import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__file__))
from fit_trigger import FEATURES, auc, sigmoid  # noqa: E402


def gate_mask(obs, tol_deg=0.5, rate_deg=1.0):
    """规则门限（shooter.cpp）：|cmd-gimbal| < tol 且 |cmd-last_cmd| < 2*tol 且瞄准有效。"""
    tol = math.radians(tol_deg)
    return (np.abs(obs[:, 10]) < tol) & (np.abs(obs[:, 14]) / 10.0 < 2 * tol) & (obs[:, 20] > 0.5)


def main():
    raw = np.loadtxt(sys.argv[1], delimiter=",")
    y = raw[:, 0].astype(int)
    obs = raw[:, 1:]
    X = np.array([[f(obs[i]) for _, f in FEATURES] for i in range(len(obs))])
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
    Z = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    w = np.zeros(Z.shape[1])
    for _ in range(4000):
        w -= 0.05 * (Z.T @ (sigmoid(Z @ w) - y) / len(y) + 1e-3 * w)
    p = sigmoid(Z @ w)

    print("样本 %d，总命中率 %.1f%%，AUC %.3f" % (len(y), 100 * y.mean(), auc(y, p)))
    print("\n【规则门限】")
    m = gate_mask(obs)
    if m.sum():
        print("  放行 %d/%d = %.0f%% 的弹 | 这些弹命中率 %.1f%%" % (
            m.sum(), len(y), 100 * m.mean(), 100 * y[m].mean()))
    else:
        print("  在这批数据上没有放行任何一发（视角/门限口径不同）")

    print("\n【逻辑回归：按预测概率取前 k%%】")
    order = np.argsort(-p)
    print("  %6s %8s %10s %12s" % ("取前", "弹数", "命中率", "折合射频(发/s)"))
    # 300 s 里能打多少发：把"取前 k%"当作门限，落到 3 种子 × 300 s 上
    for frac in (0.10, 0.25, 0.40, 0.55, 0.70, 1.00):
        k = max(1, int(len(y) * frac))
        sel = order[:k]
        rate = k / 900.0        # 数据来自 3 × 300 s
        print("  %5.0f%% %8d %9.1f%% %12.2f" % (frac * 100, k, 100 * y[sel].mean(), rate))

    print("\n【如果按'每 1 秒只打最好的一发'调度（热量约束的真实最优）】")
    # 以 1 s 为一格，每格取概率最高的那一发 —— 模拟"热量只能给 1 发/s"时的最优选择
    bins = len(y) // 20        # 数据是 20 FPS 采的，1 s ≈ 20 个决策；这里按样本序号分组近似
    if bins > 0:
        best = []
        for i in range(0, len(y) - 20, 20):
            seg = order[(order >= i) & (order < i + 20)]
            if len(seg):
                best.append(seg[0])
        best = np.array(best)
        print("  每 20 个样本取 1 发：%d 发，命中率 %.1f%%（全打是 %.1f%%）" % (
            len(best), 100 * y[best].mean(), 100 * y.mean()))


if __name__ == "__main__":
    main()
