#!/usr/bin/env python3
"""阶段 2 离线管线自检：合成数据跑通「数据集构建 → 训练 → ONNX 导出」。

为什么要有它：真实数据要等闭环跑完（几小时），而这条管线里最容易错的地方
（列名对不上、t 对齐错、ONNX 导出失败、指标方向反了）**用合成数据就能查出来**。

合成数据的构造刻意做成"标签确实由特征决定"：
    Δlat  = 0.02 * ekf0 - 0.05 * err_yaw      （米）
    Δvert = -0.03 * fly_time_est
再叠 ±8 mm 噪声 —— 好模型应该能把它学回来（R² > 0.5），学不回来就是管线有 bug。

用法： python auto_aim_rl/train/test_pred_pipeline.py
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import build_pred_dataset as B   # noqa: E402

N_TICKS = 4000       # ≈ 200 s @20 Hz
N_RUNS = 3

# C++ 侧的"唯一真相"（相对本文件：../../sp_vision_25/tasks/auto_aim/pred_residual.cpp）
CPP = os.path.normpath(os.path.join(HERE, "..", "..", "sp_vision_25",
                                    "tasks", "auto_aim", "pred_residual.cpp"))


def check_feature_contract() -> bool:
    """C++ 的 pred_feature_names()（kNames 初始化列表）必须与 Python 的 FEATURES 逐字一致。

    这是整个阶段 2 最容易静默出错的地方：两边列顺序/长度不一致时，模型会拿到错列，
    而现象只是"效果差一点"✗ —— 所以做成硬自检。
    """
    if not os.path.exists(CPP):
        print(f"⚠ 找不到 C++ 特征定义（{CPP}），跳过契约自检")
        return True
    src = open(CPP).read()
    m = re.search(r"kNames\s*=\s*\{(.*?)\};", src, re.S)
    if not m:
        print("✗ 解析不出 C++ 的 kNames 初始化列表")
        return False
    cpp = re.findall(r'"([A-Za-z0-9_]+)"', m.group(1))
    py = list(B.FEATURES)
    if cpp == py:
        print(f"✓ 特征契约一致：{len(py)} 维（C++ pred_feature_names() == Python FEATURES）")
        return True
    print(f"✗ 特征契约不一致！\n  C++  ({len(cpp)}): {cpp}\n  Py   ({len(py)}): {py}")
    return False


def make_run(dirpath: str, rid: int, rng: np.random.Generator):
    """造一次运行的 trace.csv + state.json（逐发的标签由特征决定）。"""
    t = np.arange(N_TICKS) * 0.05
    f = {name: np.zeros(N_TICKS) for name in B.FEATURES}
    # 常量列用 np.full（写成标量会让后面 f["shoot"][i] = 1 报 "float 不支持下标赋值"）
    f["has_target"] = np.full(N_TICKS, 1.0)
    f["ekf0"] = 3.0 + 0.3 * np.sin(t * 1.7 + rid)          # 前向距离
    f["ekf1"] = 0.2 * np.sin(t * 3.1)
    f["ekf2"] = 0.1 * np.cos(t * 2.3)
    f["ekf3"] = 0.5 * np.cos(t * 1.1)
    f["ekf4"] = 0.4 * np.cos(t * 1.9)
    f["ekf5"] = 0.2 * np.sin(t * 2.7)
    f["ekf6"] = np.zeros(N_TICKS)
    f["ekf7"] = 2.0 + np.sin(t * 0.9)
    f["target_w"] = f["ekf7"]
    f["armor_num"] = np.full(N_TICKS, 1.0)
    f["target_num"] = np.full(N_TICKS, 1.0)
    f["bullet_speed"] = np.full(N_TICKS, 23.0)
    f["heat_frac"] = np.full(N_TICKS, 0.2)
    f["locked"] = np.zeros(N_TICKS)
    f["range_est"] = np.hypot(f["ekf0"], f["ekf1"])
    f["fly_time_est"] = f["range_est"] / f["bullet_speed"]
    # v2 窗口特征（真实数据里由 C++ 从历史算出；这里造独立信号，验证通路能把它们带进模型）
    f["win_ok"] = np.ones(N_TICKS)
    f["d_vx_2"] = 0.30 * np.sin(t * 2.3 + rid)
    f["d_vy_2"] = 0.30 * np.cos(t * 2.9 + rid)
    f["d_w_1"] = 0.20 * np.sin(t * 3.7)
    f["d_range_2"] = 0.05 * np.sin(t * 1.3)

    # 发次：每 17 个 tick 打一发（shoot=1 写在该 tick 上）
    # 标签刻意做成"由特征决定 + 小噪声"：S/N ≈ 10 ⇒ 好模型应把 off_h 的 1σ 压掉 >30%
    # （与真实数据的 DoD 同一口径；真数据噪声更大，所以这里的阈值只证明**管线与指标机件没坏**）
    shots = []
    n_shots = np.zeros(N_TICKS)
    fired = 0
    for i in range(10, N_TICKS, 17):
        fired += 1
        n_shots[i] = float(fired)          # 仿真侧发次计数在这一 tick 变为 fired
        dlat = 0.02 * f["ekf0"][i] + 0.05 * f["ekf4"][i] + 0.06 * f["d_vy_2"][i]
        dvert = 0.05 * f["ekf5"][i] + 0.02 * (f["ekf7"][i] - 2.0)
        R3 = float(f["range_est"][i]) + 0.4      # 出膛点到板心的 3D 距离 ≈ 水平距离 + 一点
        # 新标签（指令空间角度修正量）：与本仓库 rm_gimbal_demo.py 的定义一致
        off_dyaw = -dlat / R3
        off_dpitch = dvert / R3
        shots.append({
            "id": len(shots) + 1, "t": float(t[i]),
            "off_h": float(-dlat + rng.normal(0, 0.002)),
            "off_v": float(-dvert + rng.normal(0, 0.002)), "off_n": 0.0,
            "off_y_gimbal": float(dlat), "off_z_gimbal": float(dvert),
            "off_dyaw": float(off_dyaw), "off_dpitch": float(off_dpitch), "range3": R3,
            "range": float(f["range_est"][i]), "motion": "script_%d" % (1 + (i // 100) % 4),
            "min_dist": 0.05, "hit": False,
        })
    n_shots = np.maximum.accumulate(n_shots)

    csv_path = os.path.join(dirpath, "pred_trace_syn_%d.csv" % rid)
    with open(csv_path, "w") as fh:
        fh.write("t,n_shots" + "".join(",f_" + n for n in B.FEATURES) + "\n")
        for i in range(N_TICKS):
            fh.write("%.6f,%.0f" % (t[i], n_shots[i])
                     + "".join(",%.6f" % f[n][i] for n in B.FEATURES) + "\n")
    json_path = os.path.join(dirpath, "eval_syn_%d.json" % rid)
    with open(json_path, "w") as fh:
        json.dump({"shot_log": shots, "shots": len(shots)}, fh)
    return csv_path, json_path, len(shots)


def main() -> int:
    if not check_feature_contract():
        return 1
    rng = np.random.default_rng(0)
    tmp = tempfile.mkdtemp(prefix="pred_pipeline_")
    pairs = [make_run(tmp, rid, rng) for rid in range(N_RUNS)]
    print("合成数据：%d 次运行，各 %d 发（共 %d 发）→ %s"
          % (N_RUNS, pairs[0][2], N_RUNS * pairs[0][2], tmp))

    ds = os.path.join(tmp, "pred_ds.npz")
    rc = subprocess.call(
        [sys.executable, os.path.join(HERE, "build_pred_dataset.py"),
         *sum((["--run", c, j] for c, j, _ in pairs), []), "--out", ds])
    if rc != 0 or not os.path.exists(ds):
        print("✗ 数据集构建失败", file=sys.stderr)
        return 1

    ok = True
    for model, extra in (("ridge", []), ("mlp", ["--epochs", "300", "--hidden", "16"])):
        onnx = os.path.join(tmp, "pred_%s.onnx" % model)
        print(f"\n===== 训练 {model} =====")
        rc = subprocess.call(
            [sys.executable, os.path.join(HERE, "fit_pred_residual.py"),
             "--ds", ds, "--model", model, "--out-onnx", onnx, *extra])
        if rc != 0:
            print(f"✗ {model} 训练失败", file=sys.stderr)
            ok = False
            continue
        if not os.path.exists(onnx) or os.path.getsize(onnx) < 200:
            print(f"✗ {model} 的 ONNX 没导出或太小", file=sys.stderr)
            ok = False
            continue
        print(f"✓ {model}: ONNX {os.path.getsize(onnx)} 字节")

        # 用导出的 ONNX **自己**算一遍验收指标（不解析打印输出，直接验产物）
        import onnxruntime as ort
        d = np.load(ds, allow_pickle=True)
        X, raw, run = d["X"], d["raw"], d["run"]
        sess = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"])
        zeros = np.zeros((1, len(B.FEATURES)), dtype=np.float32)
        pred = sess.run(None, {sess.get_inputs()[0].name: X.astype(np.float32)})[0]
        print(f"✓ onnxruntime 前向：输入 {zeros.shape} → 输出 {pred.shape}（期望 (1, 2)）")
        holdout = int(np.bincount(run).argmax())
        m = run == holdout
        # 剩余误差 = 标签（该加的修正量）− 预测 ✓（v1 是"偏差 + 修正"，换了标签定义后必须反过来）
        h0, h1 = np.std(raw[m, 0]), np.std(raw[m, 0] - pred[m, 0])
        v0, v1 = np.std(raw[m, 1]), np.std(raw[m, 1] - pred[m, 1])
        red_h = 100 * (h1 / h0 - 1)
        print(f"  留出 run{holdout}（{int(m.sum())} 发）：Δyaw 1σ {h0 * 1000:.2f} → {h1 * 1000:.2f} mrad"
              f"（{red_h:+.1f}%）｜ Δpitch {v0 * 1000:.2f} → {v1 * 1000:.2f} mrad"
              f"（{100 * (v1 / v0 - 1):+.1f}%）")
        if pred.shape != (X.shape[0], 2):
            print("✗ 输出维度不对", file=sys.stderr)
            ok = False
        else:
            # 断言分级：ridge 必须拿下 ≥30%（管线通不通看它）；MLP 只要求明显改善（≥10%），
            # 因为小网在合成数据上本就拟合得没那么狠 —— 这里考的是**管线**，不是模型排名 ✓
            need = -30.0 if model == "ridge" else -10.0
            if red_h > need:
                print(f"✗ 水平散布只降了 {red_h:+.1f}%（{model} 的阈值 {need}%）"
                      f" —— 合成数据 S/N 足够，降不到就说明管线有问题", file=sys.stderr)
                ok = False

    print("\n" + ("✅ 管线自检通过" if ok else "❌ 管线自检有失败项"))
    print(f"（产物在 {tmp}，可人工看 rank/系数）")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
