#!/usr/bin/env python3
"""用**闭环真实数据**训练"这一发会不会中"的质量模型（小型 MLP），导出 ONNX 给 C++ 用。

为什么不是逻辑回归：规则门限的本质是两个条件的**合取**
（|cmd-gimbal| < tol 且 |cmd-last_cmd| < 2·tol），线性模型天然表达不了 AND；
实测逻辑回归在同等射频下（37.1% @0.70 发/s）打不过手工门限（38.3% @0.61 发/s）。
MLP 能学出这种合取结构，还能顺带利用"距离/自转/最近命中率"等门限没用到的信息。

特征子集（17 维，屏蔽掉与"这一发准不准"无关的量）：
  0..14 瞄准/目标状态 | 17 最近 10 发命中率（当前这轮机动到底好不好打）| 20 瞄准点有效
  **不用** 15/16（距上次开火/命中，是时序不是质量）、18/19（热量/锁枪 —— 部署时的
  "热量快满就放掉"规则单独处理，不混进质量估计）。

用法:
    python auto_aim_rl/train/fit_quality_mlp.py /tmp/rl_log_train.csv /tmp/rl_log3.csv \
        --out auto_aim_rl/runs/quality_q.onnx
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

# **不能包含 obs[17]（最近 10 发命中率）** —— 它是策略自己行为的函数：
# 一旦门限收紧、开了几发没中，"最近命中率"就掉到 0，模型据此判断"现在不好打"，
# 于是更不开火 —— 自锁死循环（实测 λ=0.69 时放行率只有 0.4%，远低于离线预期的 10%）。
# 质量模型只能看**与开火历史无关**的量：目标状态 + 瞄准误差 + 距离 + 自转 + 瞄准有效性。
DIMS = list(range(15)) + [20]


class QualityNet(nn.Module):
    def __init__(self, n_in: int = len(DIMS), hidden: int = 32):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(n_in, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1))

    def forward(self, x):
        return self.body(x).squeeze(-1)


def auc(y, p):
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
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--out", default="auto_aim_rl/runs/quality_q.onnx")
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--hidden", type=int, default=32)
    args = ap.parse_args()

    raws = [np.loadtxt(f, delimiter=",") for f in args.csv]
    raws = [r if r.ndim > 1 else r[None, :] for r in raws]
    # **按文件切分**：每个 CSV 是一次独立运行（不同种子/不同时间），
    # 训练用前面几次、留出最后一次。随机切分或按行切分都会让"同一段时间的状态"
    # 泄漏到两边 —— 实测按行切分时训练 AUC 直接 1.000、留出 0.67（纯记忆）。
    te_raw = raws[-1]
    tr_raw = np.vstack(raws[:-1]) if len(raws) > 1 else raws[0]
    y = tr_raw[:, 0].astype(np.float32)
    X = tr_raw[:, 1:][:, DIMS].astype(np.float32)
    yte = te_raw[:, 0].astype(np.float32)
    Xte = te_raw[:, 1:][:, DIMS].astype(np.float32)
    print("训练 %d 条（命中 %.1f%%）/ 留出 %d 条（命中 %.1f%%），特征 %d 维" % (
        len(y), 100 * y.mean(), len(yte), 100 * yte.mean(), X.shape[1]))
    Xtr, ytr = X, y
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Ztr = torch.as_tensor((Xtr - mu) / sd)
    Zte = torch.as_tensor((Xte - mu) / sd)
    ttr, tte = torch.as_tensor(ytr), torch.as_tensor(yte)

    net = QualityNet(X.shape[1], args.hidden)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-3)
    # 类别不平衡（命中 ~28%）：给正样本加权，但不能过头（会牺牲排序质量）
    w_pos = torch.tensor(float((ytr == 0).sum()) / max(float((ytr == 1).sum()), 1.0)).sqrt()
    lossf = nn.BCEWithLogitsLoss(pos_weight=w_pos)

    # **留出集早停**：这个模型很小但样本更少，跑满 3000 轮必然记忆训练集
    # （实测训练 AUC 1.000、留出 0.67，而且 P 输出退化成一堆 0/1、完全没校准）。
    best_auc, best_state, bad = -1.0, None, 0
    for ep in range(args.epochs):
        net.train()
        loss = lossf(net(Ztr), ttr)
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 25 == 0:
            net.eval()
            with torch.no_grad():
                a_te = auc(yte, torch.sigmoid(net(Zte)).numpy())
            if a_te > best_auc + 1e-4:
                best_auc, bad = a_te, 0
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
            if (ep + 1) % 250 == 0:
                print("  ep %4d loss %.4f | 留出 AUC %.3f（历史最好 %.3f）"
                      % (ep + 1, loss, a_te, best_auc), flush=True)
            if bad >= 12:
                print("  早停于 ep %d" % (ep + 1))
                break
    if best_state is not None:
        net.load_state_dict(best_state)

    net.eval()
    with torch.no_grad():
        p_te = torch.sigmoid(net(Zte)).numpy()
    print("留出集 AUC %.3f | 命中率 %.1f%% | 预测均值 %.3f（越接近越校准）" % (
        auc(yte, p_te), 100 * yte.mean(), p_te.mean()))
    print("按预测分位放行的命中率（部署时选门限用）：")
    for frac in (0.10, 0.20, 0.30, 0.50, 0.70, 1.00):
        thr = float(np.quantile(p_te, 1 - frac))
        sel = p_te >= thr
        print("  取前 %3.0f%% -> P>=%.3f 命中率 %.1f%%" % (
            frac * 100, thr, 100 * yte[sel].mean() if sel.any() else float("nan")))

    class Wrap(nn.Module):
        """标准化 + 网络 + sigmoid 打包成一个图，部署时只喂原始 17 维观测。

        mu/sd **必须注册成 buffer（tensor）**：存成普通属性（尤其是 numpy 数组）时，
        老导出器会把输入整个常量折叠掉 —— 导出的模型 inputs: []，
        到 C++ 侧一 run 就抛 Ort::Exception "out of index"（踩过，排查了半天）。
        """

        def __init__(self, net, mu, sd):
            super().__init__()
            self.net = net
            self.register_buffer("mu", torch.as_tensor(mu, dtype=torch.float32))
            self.register_buffer("sd", torch.as_tensor(sd, dtype=torch.float32))

        def forward(self, x):
            return torch.sigmoid(self.net((x - self.mu) / self.sd))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    dummy = torch.zeros(1, X.shape[1])
    # **用老的 TorchScript 导出器（dynamo=False）**：torch 2.14 的新导出器在本机
    # 会在 onnxscript 的 optimize pass 上抛 PassError（InlinePass 之后），
    # 而这条链路只是个三层 MLP，老导出器完全够用且稳。
    torch.onnx.export(Wrap(net, mu, sd), dummy, args.out,
                      input_names=["obs"], output_names=["p"], opset_version=17,
                      dynamo=False)

    # ---- 导出后**自检**（必需，不是可选）----------------------------------------
    import onnxruntime as ort
    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    ins = [(i.name, i.shape) for i in sess.get_inputs()]
    assert len(ins) == 1, "导出的图没有输入（输入被常量折叠了）：%s" % ins
    probe = np.asarray(X[:1], dtype=np.float32)   # 固定 batch=1（部署就是每帧一帧）
    got = sess.run(None, {ins[0][0]: probe})[0].reshape(-1)
    with torch.no_grad():
        ref = torch.sigmoid(net(torch.as_tensor((probe - mu) / sd))).numpy().reshape(-1)
    err = float(np.abs(got - ref).max())
    print("ONNX 自检：输入 %s | 与 torch 最大偏差 %.2e" % (ins, err))
    assert err < 1e-4, "ONNX 与 torch 不一致（%.2e）" % err
    print("已保存 %s（输入 %d 维，输出 P(命中)）" % (args.out, X.shape[1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
