#!/usr/bin/env python3
"""阶段 2（预测）：用闭环数据拟合"经典瞄准点的残差修正量"，导出 ONNX 给 C++ 用。

* 标签 `y = (Δ_lat, Δ_vert)`（米）来自数据集（见 build_pred_dataset.py）；
* **验收指标直接按闭环口径算**：修正后 `off_h` 的 1σ 相对修正前降低多少
  （DoD：水平散布压掉 ≥ 30%）；
* **必须按"运行"留出验证**：同一次运行的相邻发高度相关，按样本随机划分会严重高估泛化。

用法：
    python auto_aim_rl/train/fit_pred_residual.py --ds /tmp/pred_ds.npz --model ridge \
        --out-onnx auto_aim_rl/runs/pred_ridge.onnx
    python auto_aim_rl/train/fit_pred_residual.py --ds /tmp/pred_ds.npz --holdout-run 3 \
        --model mlp --epochs 400 --out-onnx auto_aim_rl/runs/pred_mlp.onnx
"""
from __future__ import annotations

import argparse
import os

import numpy as np

# 训练时挡掉的列（恒 0 / 与"这一发准不准"无关），但**喂给 C++ 的特征向量长度不变**
DROP_FEATURES = {"err_yaw_rate"}


def load_ds(path: str):
    d = np.load(path, allow_pickle=True)
    feats = [str(f) for f in d["features"]]
    keep = [i for i, f in enumerate(feats) if f not in DROP_FEATURES]
    X, y, raw, motion, run = d["X"], d["y"], d["raw"], d["motion"], d["run"]
    # 丢掉"历史不足"的样本（win_ok=0）：那时窗口特征全是 0，会教坏模型
    if "win_ok" in feats:
        good = X[:, feats.index("win_ok")] > 0.5
        if not good.all():
            print(f"丢弃历史不足的样本 {int((~good).sum())} / {X.shape[0]} 发（win_ok=0）")
        X, y, raw, motion, run = X[good], y[good], raw[good], motion[good], run[good]
    return X[:, keep], y, raw, motion, run, [feats[i] for i in keep]


def report(tag: str, raw_te: np.ndarray, pred: np.ndarray, idx=None):
    """闭环口径指标：修正前后 off_h/off_v 的 1σ，以及"板内"代理命中率。"""
    if idx is not None:
        raw_te, pred = raw_te[idx], pred[idx]
    if raw_te.shape[0] == 0:
        print(f"  {tag}: 没有样本")
        return None
    off_h, off_v = raw_te[:, 0], raw_te[:, 1]
    h0, v0 = float(np.std(off_h)), float(np.std(off_v))
    h1 = float(np.std(off_h + pred[:, 0]))
    v1 = float(np.std(off_v + pred[:, 1]))
    # 命中率代理：小装甲板半宽 70mm / 半高 28mm —— 只作趋势参考（真实判定在仿真侧）
    hit0 = np.mean((np.abs(off_h) < 0.070) & (np.abs(off_v) < 0.028))
    hit1 = np.mean((np.abs(off_h + pred[:, 0]) < 0.070) & (np.abs(off_v + pred[:, 1]) < 0.028))
    print(f"  {tag}: off_h 1σ {h0 * 1000:6.1f} → {h1 * 1000:6.1f} mm"
          f"（{100 * (h1 / h0 - 1):+5.1f}%）｜ off_v 1σ {v0 * 1000:6.1f} → {v1 * 1000:6.1f} mm"
          f"（{100 * (v1 / v0 - 1):+5.1f}%）｜ 板内代理命中 {hit0 * 100:4.1f}% → {hit1 * 100:4.1f}%")
    return h1 / h0 - 1.0


def fit_ridge(Xtr: np.ndarray, ytr: np.ndarray, lam: float = 1e-2):
    """带标准化的岭回归（闭式解）：返回 (mu, sd, W, b)，y ≈ ((x-mu)/sd) @ W + b。"""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Z = np.hstack([(Xtr - mu) / sd, np.ones((Xtr.shape[0], 1))])
    W = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ ytr)
    return mu, sd, W[:-1], W[-1]


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean(0)) ** 2)) + 1e-12
    return 1 - ss_res / ss_tot


def _make_mlp(n_in: int, hidden: int, n_out: int):
    """小 MLP（float32，标准化烘进图里，方便 C++ 端只喂原始特征）。"""
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("mu", torch.zeros(n_in))
            self.register_buffer("sd", torch.ones(n_in))
            self.net = nn.Sequential(
                nn.Linear(n_in, hidden), nn.Tanh(),
                nn.Linear(hidden, hidden), nn.Tanh(),
                nn.Linear(hidden, n_out))

        def set_norm(self, mu: np.ndarray, sd: np.ndarray):
            self.mu.copy_(torch.tensor(mu, dtype=torch.float32))
            self.sd.copy_(torch.tensor(sd, dtype=torch.float32))

        def forward(self, x):   # x: (batch, n_in) 原始特征
            return self.net((x - self.mu) / self.sd)

    return Net()


def _onnx_export(module, dummy, path: str):
    """导出 ONNX：优先走**旧版（TorchScript）导出器**——它只依赖 `onnx`，
    而新版默认的 dynamo 导出器还要求 `onnxscript`（踩过：torch 2.14 上
    `ModuleNotFoundError: No module named 'onnxscript'`，训练白跑一轮）。
    旧接口不可用时再回落默认导出器（那时才需要 onnxscript）。"""
    import torch

    try:
        torch.onnx.export(
            module, dummy, path, input_names=["features"], output_names=["delta"],
            dynamic_axes={"features": {0: "batch"}, "delta": {0: "batch"}},
            opset_version=16, dynamo=False)
    except TypeError:
        torch.onnx.export(
            module, dummy, path, input_names=["features"], output_names=["delta"],
            dynamic_axes={"features": {0: "batch"}, "delta": {0: "batch"}}, opset_version=16)


def _export_onnx(net, n_in: int, path: str):
    """导出 (batch, n_in) → (batch, 2) 的 ONNX（float32）。"""
    import torch

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    net.eval()
    _onnx_export(net, torch.zeros(1, n_in, dtype=torch.float32), path)


def _export_linear_onnx(mu, sd, W, b, n_in: int, path: str):
    """线性模型的 ONNX（标准化 + 仿射合并成一个 Linear）。"""
    import torch
    import torch.nn as nn

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    Wt = (W / sd[:, None])                                   # y = x_scaled @ W + b，x_scaled=(x-mu)/sd
    lin = nn.Linear(n_in, W.shape[1])
    with torch.no_grad():
        lin.weight.copy_(torch.tensor(Wt.T, dtype=torch.float32))
        lin.bias.copy_(torch.tensor(b - (mu / sd) @ W, dtype=torch.float32))
    lin.eval()
    _onnx_export(lin, torch.zeros(1, n_in, dtype=torch.float32), path)


def main() -> int:
    ap = argparse.ArgumentParser(description="阶段 2 残差模型训练（ridge / MLP）")
    ap.add_argument("--ds", required=True, help="build_pred_dataset.py 产出的 npz")
    ap.add_argument("--holdout-run", type=int, default=None,
                    help="留出的运行编号；默认留出样本最多的那次")
    ap.add_argument("--model", choices=("ridge", "linear", "mlp"), default="ridge")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-onnx", default="")
    a = ap.parse_args()

    X, y, raw, motion, run, feats = load_ds(a.ds)
    print(f"样本 {X.shape[0]} 发 × {X.shape[1]} 特征")
    print(f"特征列：{feats}")

    if a.holdout_run is None:
        uniq, cnt = np.unique(run, return_counts=True)
        a.holdout_run = int(uniq[int(np.argmax(cnt))])
    te = run == a.holdout_run
    tr = ~te
    print(f"按运行留出：验证 = run{a.holdout_run}（{int(te.sum())} 发），训练 = {int(tr.sum())} 发")
    if te.sum() < 10 or tr.sum() < 10:
        print("⚠ 样本太少，结论不可信（先多采几次运行）")

    print("\n[基线] 不修正（Δ=0）：")
    report("验证集", raw[te], np.zeros((int(te.sum()), 2)))

    if a.model in ("ridge", "linear"):
        lam = 1e-2 if a.model == "ridge" else 1e-6
        mu, sd, W, b = fit_ridge(X[tr], y[tr], lam=lam)

        def forward(x):
            return ((x - mu) / sd) @ W + b
        print(f"\n[{a.model}] 参数量 {W.size + b.size}（特征 {W.shape[0]} → 输出 2）")
        print("  y 的 R²：训练 %.3f　验证 %.3f" % (
            _r2(y[tr], forward(X[tr])), _r2(y[te], forward(X[te]))))
        for j, name in enumerate(("Δ_lat", "Δ_vert")):
            mae = np.mean(np.abs(forward(X[te])[:, j] - y[te][:, j])) * 1000
            print(f"  {name}：验证 MAE {mae:5.1f} mm（该分量自身 1σ "
                  f"{np.std(y[te][:, j]) * 1000:5.1f} mm）")
        print("  标准化后的系数（|w| 前 6）：")
        for j, name in enumerate(("Δ_lat", "Δ_vert")):
            order = np.argsort(-np.abs(W[:, j]))[:6]
            print(f"    {name}: " + "  ".join(f"{feats[k]}={W[k, j]:+.3f}" for k in order))
        if a.out_onnx:
            _export_linear_onnx(mu, sd, W, b, X.shape[1], a.out_onnx)
            print(f"  已导出 ONNX → {a.out_onnx}")
        pred = forward(X)
    else:
        import torch
        torch.manual_seed(a.seed)
        net = _make_mlp(X.shape[1], a.hidden, y.shape[1])
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        net.set_norm(mu, sd)
        opt = torch.optim.Adam(net.parameters(), lr=a.lr)
        Xtr = torch.tensor(X[tr], dtype=torch.float32)
        Ytr = torch.tensor(y[tr], dtype=torch.float32)
        for ep in range(a.epochs):
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(net(Xtr), Ytr)
            loss.backward()
            opt.step()
            if (ep + 1) % max(1, a.epochs // 8) == 0:
                print(f"  epoch {ep + 1:4d}　train MSE {loss.item() * 1e6:8.2f} mm²")
        net.eval()

        def forward(x):
            with torch.no_grad():
                return net(torch.tensor(x, dtype=torch.float32)).numpy()
        print(f"\n[MLP] 隐藏层 {a.hidden}，参数量 {sum(p.numel() for p in net.parameters())}")
        print("  y 的 R²：训练 %.3f　验证 %.3f" % (
            _r2(y[tr], forward(X[tr])), _r2(y[te], forward(X[te]))))
        pred = forward(X)
        if a.out_onnx:
            _export_onnx(net, X.shape[1], a.out_onnx)
            print(f"  已导出 ONNX → {a.out_onnx}")

    print("\n[验收] 按运行留出的闭环口径（DoD：水平 1σ 降 ≥30%）：")
    report("全部验证发", raw[te], pred[te])
    for m in np.unique(motion[te]):
        report(f"靶型段 {m}", raw[te], pred[te], idx=(motion[te] == m))
    print("\n提示：以这个数字为准之前，务必在闭环里做 residual off/on 的 A/B（4 种子 × 300 s）——"
          "离线指标只是筛模型用的。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

