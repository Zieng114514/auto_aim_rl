#!/usr/bin/env python3
"""把"学残差"导出成可部署 ONNX（22 维输入 → 2 维角修正，**弧度** ✓）。

设计（与现有 `pred_residual` 通道完全对齐 ✓）
--------------------------------------------
* 输入：`pred_feature_names()` 的 **22 维**（trace 的 `f_*` ✓）—— 实测加上 4 个几何量
  也只有 4.64° vs 4.66° 的差别 ✓ ⇒ **不值得改 C++ 特征口径 + 重采数据** ✓✓
* 输出：**(Δyaw, Δpitch) 弧度** ✓（现有 C++ 就是"直接把模型输出加到指令角上"✓）
* 训练目标：把"速度残差 Δv"换算成**角度修正** ✓：
      p2 = 几何外推 + Δv·τ  ⇒  Δ = angle(p2) − angle(几何外推)
  这样网络学到的仍然只是"EKF 错在哪"（前馈 ✓），不是"当前误差"（反馈 ✗，§1.27）
* 标准化（mu/sd）烘进图里 ✓（与 `fit_pred_residual.py` 同做法 ✓）

用法： python3 export_residual_onnx.py <battery_dir> <out.onnx>
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pred_learned_eval import FEATURES, build, ridge_fit, yawpitch  # noqa: E402

DEG2RAD = math.pi / 180.0


def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else "/root/gpufree-data/runs/battery"
    out = sys.argv[2] if len(sys.argv) > 2 else "/root/vision/auto_aim_rl/runs/pred_resid_v1.onnx"

    rows = []
    for tr_path in sorted(glob.glob(os.path.join(root, "traces", "*.csv"))):
        base = os.path.splitext(os.path.basename(tr_path))[0]
        st = os.path.join(root, base + ".json")
        if not os.path.exists(st):
            continue
        try:
            shots = json.load(open(st, encoding="utf-8")).get("shot_log") or []
        except Exception:                                # noqa: BLE001
            continue
        rows.extend(build(tr_path, shots))
    if not rows:
        print("没有样本 ✗")
        return 1

    X = np.array([r["x"][: len(FEATURES)] for r in rows], dtype=float)   # 只用 22 维 ✓
    # 目标：Δv 引起的角修正（弧度 ✓）。**先做物理过滤** ✓：
    #   |Δv|·τ > 1 m 或 |角修正| > 20° ⇒ 那是"跟踪失败/刚捕获"的样本 ✗，不是我们要学的预测误差 ✓
    keep, Y = [], []
    dropped = 0
    for idx, r in enumerate(rows):
        p_cv = np.asarray(r["p_cv"], dtype=float)
        dv = np.asarray(r["dv"], dtype=float)
        tau = float(r["tau"])
        if float(np.linalg.norm(dv)) * tau > 1.0:
            dropped += 1
            continue
        p2 = p_cv + dv * tau
        y1, t1 = yawpitch(p_cv)
        y2, t2 = yawpitch(p2)
        dy = (y2 - y1 + 180.0) % 360.0 - 180.0
        dt = t2 - t1
        if abs(dy) > 20.0 or abs(dt) > 20.0:
            dropped += 1
            continue
        keep.append(idx)
        Y.append([dy * DEG2RAD, dt * DEG2RAD])
    X = X[keep]
    Y = np.array(Y, dtype=float)
    print(f"样本 {len(rows)} → 保留 {len(keep)}（剔除 {dropped} 条跟踪失败 ✓）")

    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
    ym, ys = Y.mean(axis=0), Y.std(axis=0) + 1e-9
    W = ridge_fit((X - mu) / sd, (Y - ym) / ys, lam=3.0)          # (23, 2) 含偏置 ✓
    w_lin, b_lin = W[:-1], W[-1]                                   # (22,2) 与 (2,) ✓

    # 把 (mu, sd, ym, ys) 折进权重（部署侧只见 22 输入 → 2 输出 ✓）
    W_final = (w_lin / sd[:, None]) * ys[None, :]                  # (22,2)
    b_final = (b_lin * ys) + ym - (mu / sd) @ (w_lin * ys[None, :])

    import torch

    class Lin(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(X.shape[1], 2)
            with torch.no_grad():
                self.lin.weight.copy_(torch.tensor(W_final.T, dtype=torch.float32))
                self.lin.bias.copy_(torch.tensor(b_final, dtype=torch.float32))

        def forward(self, z):                                     # (N,22) → (N,2)
            return self.lin(z)

    net = Lin().eval()
    dummy = torch.zeros(1, X.shape[1], dtype=torch.float32)
    dyn = {"features": {0: "batch"}, "dyaw_dpitch": {0: "batch"}}
    try:
        torch.onnx.export(net, dummy, out, input_names=["features"],
                          output_names=["dyaw_dpitch"], opset_version=13,
                          dynamic_axes=dyn, dynamo=False)
    except TypeError:                                             # 老 torch 没有 dynamo 参数 ✓
        torch.onnx.export(net, dummy, out, input_names=["features"],
                          output_names=["dyaw_dpitch"], opset_version=13,
                          dynamic_axes=dyn)

    # 自检：onnxruntime 复算 vs torch ✓（与 fit_pred_residual 同一纪律 ✓）
    import onnxruntime as ort

    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    z = X[:5].astype(np.float32)
    o_ort = sess.run(None, {"features": z})[0]
    o_t = net(torch.tensor(z)).detach().numpy()
    err = float(np.max(np.abs(o_ort - o_t)))
    # 训练内误差（rad → 等效角度误差，便于读 ✓）
    o_all = sess.run(None, {"features": X.astype(np.float32)})[0] - Y
    print(f"样本 {len(rows)} 发；导出 {out}")
    print(f"  输出量级：Δyaw 1σ {np.std(o_all[:, 0]) * 1000:.2f} mrad，"
          f"Δpitch 1σ {np.std(o_all[:, 1]) * 1000:.2f} mrad（训练内残差 ✓）")
    print(f"  ONNX 复算 vs torch 最大差 {err:.2e}（应 ~1e-7 ✓）")
    print(f"  输入 {X.shape[1]} 维 = pred_feature_names() ✓；输出 2 维 = (Δyaw, Δpitch) 弧度 ✓")
    return 0 if err < 1e-4 else 1


if __name__ == "__main__":
    raise SystemExit(main())
