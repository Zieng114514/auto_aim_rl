"""算出让 classifier 达到指定射频的 P 门限（用已采集的数据的分位数）。"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from fit_trigger import FEATURES, sigmoid

raw = np.loadtxt(sys.argv[1], delimiter=",")
y = raw[:, 0].astype(int); obs = raw[:, 1:]
X = np.array([[f(obs[i]) for _, f in FEATURES] for i in range(len(obs))])
mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
Z = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
w = np.zeros(Z.shape[1])
for _ in range(4000):
    w -= 0.05 * (Z.T @ (sigmoid(Z @ w) - y) / len(y) + 1e-3 * w)
p = sigmoid(Z @ w)
print("数据里 P 的分布：min %.3f p25 %.3f 中位 %.3f p75 %.3f max %.3f" % (
    p.min(), np.percentile(p, 25), np.median(p), np.percentile(p, 75), p.max()))
for frac in (0.40, 0.55, 0.70, 0.85, 1.00):
    thr = np.quantile(p, 1 - frac)
    sel = p >= thr
    print("  目标射频 %.2f 发/s -> P 门限 %.3f（数据里放行 %.0f%%，命中率 %.1f%%）" % (
        frac, thr, 100 * sel.mean(), 100 * y[sel].mean() if sel.any() else float("nan")))
