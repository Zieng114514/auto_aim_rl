#!/usr/bin/env python3
"""用 **C++ 闭环里采到的真实数据** 拟合"这一发会不会中"的门限（stage 2 监督学习）。

为什么走这条路：镜像环境再怎么标定，跟踪/指令的统计与 C++ 仍差一个量级
（|ey| 中位 0.9° vs 3.1°），策略学到的阈值迁移不过去（见 agnet_experience.md §1.16）。
那就让射手在 C++ 里**放开打**（rl_shooter_mode: log），把"开火那一刻的 21 维观测 +
这一发中没中"记下来，直接在这里拟合 —— **没有任何迁移损失**。

输出：
  * 特征标准化参数 + 逻辑回归权重（可直接写进 sim.yaml 的 rl_shooter_clf_w）
  * AUC / 准确率 / 各特征权重（看它到底学到了什么）

用法:
    python auto_aim_rl/train/fit_trigger.py /tmp/shooter_rl_log.csv
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

# 特征选择：**物理上说得通**的量（而不是把 21 维一股脑丢进去 —— 几百个样本下
# 维度越高越容易过拟合，而且没法解释）
FEATURES = [
    ("abs_ey", lambda o: abs(o[10])),
    ("abs_ep", lambda o: abs(o[11])),
    ("abs_dey", lambda o: abs(o[12])),
    ("abs_dep", lambda o: abs(o[13])),
    ("abs_cmd_rate", lambda o: abs(o[14])),
    ("dist", lambda o: o[3]),
    ("vyaw", lambda o: o[7]),
    ("t_since_shot", lambda o: o[15]),
    ("t_since_hit", lambda o: o[16]),
    ("recent_hit", lambda o: o[17]),
    ("heat", lambda o: o[18]),
    ("visible", lambda o: o[8]),
    ("valid", lambda o: o[20]),
    # 交互项：|ey| 与距离的乘积 ≈ 弹丸在板上的横向误差（mm 量级），是真正决定命中的量
    ("ey_x_dist", lambda o: abs(o[10]) * o[3]),
    ("cmdrate_x_vyaw", lambda o: abs(o[14]) * abs(o[7])),
]


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def auc(y, p):
    """ROC-AUC（Mann-Whitney U 的等价写法）。"""
    order = np.argsort(p)
    ranks = np.empty(len(p), dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return (ranks[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--holdout", type=float, default=0.25)
    ap.add_argument("--out", default="/tmp/trigger_params.sh", help="导出给 C++ 的参数文件")
    args = ap.parse_args()

    raw = np.loadtxt(args.csv, delimiter=",")
    if raw.ndim == 1:
        raw = raw[None, :]
    y = raw[:, 0].astype(int)
    obs = raw[:, 1:]
    X = np.array([[f(obs[i]) for _, f in FEATURES] for i in range(len(obs))])
    print("样本 %d：命中 %d（%.1f%%）" % (len(y), y.sum(), 100.0 * y.mean()))

    # 按时间切分（前 75% 训练 / 后 25% 留出）：随机切分会让"同一段时间的状态"泄漏到两边
    n_tr = int(len(y) * (1 - args.holdout))
    Xtr, ytr, Xte, yte = X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:]

    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-9
    Ztr = (Xtr - mu) / sd
    Zte = (Xte - mu) / sd
    Ztr = np.hstack([Ztr, np.ones((len(Ztr), 1))])
    Zte = np.hstack([Zte, np.ones((len(Zte), 1))])

    w = np.zeros(Ztr.shape[1])
    for _ in range(args.epochs):
        g = Ztr.T @ (sigmoid(Ztr @ w) - ytr) / len(ytr) + args.l2 * w
        w -= args.lr * g

    for name, Z, yy in (("训练", Ztr, ytr), ("留出", Zte, yte)):
        p = sigmoid(Z @ w)
        acc = ((p > 0.5).astype(int) == yy).mean() if len(yy) else float("nan")
        print("%s：n=%4d 命中率 %5.1f%% | AUC %.3f | 准确率 %.3f | 预测命中率 %.1f%%" % (
            name, len(yy), 100.0 * yy.mean() if len(yy) else float("nan"), auc(yy, p), acc,
            100.0 * (p > 0.5).mean()))

    print("\n特征权重（标准化后，正 = 更该开火）：")
    for (name, _), wi in sorted(zip(FEATURES, w[:-1]), key=lambda t: -abs(t[1])):
        print("  %-16s %+7.3f" % (name, wi))
    print("  %-16s %+7.3f" % ("bias", w[-1]))

    # 导出给 C++：**按派生特征版导出**（z = Σ w_i (f_i-μ_i)/σ_i + b，原始尺度传 μ/σ/w）。
    # C++ 侧 shooter_rl.cpp 按 FEATURES 的同一顺序算 15 个派生特征再套这个公式。
    out = args.out or "/tmp/trigger_params.sh"
    with open(out, "w") as f:
        f.write("# 由 auto_aim_rl/train/fit_trigger.py 生成（%s）\n" % args.csv)
        f.write('CLF_W="%s"\n' % ",".join("%.6f" % v for v in w))
        f.write('CLF_MU="%s"\n' % ",".join("%.6f" % v for v in mu))
        f.write('CLF_SD="%s"\n' % ",".join("%.6f" % v for v in sd))
    print("\n已写出 %s（source 之后用 CLF_W/CLF_MU/CLF_SD 传给 eval_policy.sh）" % out)
    print("偏置项（w 最后一项）= %.6f" % w[-1])
    return 0


if __name__ == "__main__":
    sys.exit(main())
