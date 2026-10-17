#!/usr/bin/env python3
"""学预测：在同一批样本上对比 `now` / `cv_ekf`（因果）/ `cv_oracle`（上界）/ ridge / MLP。

为什么要这个脚本
----------------
`eval_pred_ceiling.py` 给出了**上界**：完美匀速外推能把水平误差降到 **3.14°（RMS）** ✓。
但它用的速度是**真值**（oracle）✗ —— 真实系统只有 **EKF 速度** ✓。
所以"学预测值不值得"必须这样比 ✓：

| 基线 | 用什么信息 | 含义 |
|---|---|---|
| `now` | Aimer 自己的瞄准点 | 现状 ✓ |
| **`cv_ekf`** | EKF 位置+速度+角速度（**因果 ✓**） | **真正的对手**：经典匀速外推 ✓ |
| `cv_oracle` | 真值位置+速度 | 上界（不可能超过 ✓） |
| `ridge` / `mlp` | 22 维前馈特征（不含 aim/cmd/err ✗） | 学习模型 ✓ |

验收线（A1 口径）：**学习模型必须在"按机动留出"下 RMS 低于 `cv_ekf`** ✓，
且越接近 `cv_oracle` 越好 ✓；只在随机划分下赢不算 ✓（那是背答案 ✗）。

特征纪律（见 `pred_residual.cpp` 注释 / agnet_experience §1.27 ✓）
--------------------------------------------------------------
只用 trace 里的 22 个 `f_*`（目标状态 + 感知/弹药 + 窗口历史 ✓），
**绝不用** `g_aim`（瞄准点 = 控制器输出 ✗）—— 否则模型会学成"误差反馈项"✗，
闭环里与 Aimer 抢活 ✓（上一版就是这么翻车的 ✓）。

用法
----
    python3 pred_learned_eval.py /root/gpufree-data/runs/battery
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_pred_ceiling import (  # noqa: E402
    fit_circle, gimbal_frame, load_trace, plates_of, wrap,
)

FEATURES = [
    "has_target", "ekf0", "ekf1", "ekf2", "ekf3", "ekf4", "ekf5", "ekf6", "ekf7",
    "target_w", "armor_num", "target_num", "bullet_speed", "heat_frac", "locked",
    "range_est", "fly_time_est", "win_ok", "d_vx_2", "d_vy_2", "d_w_1", "d_range_2",
]  # 与 sp_vision 的 pred_feature_names() **逐字一致** ✓


def rotz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def yawpitch(v: np.ndarray) -> tuple:
    """云台系向量 → (yaw, pitch) 度 ✓"""
    return (math.degrees(math.atan2(v[1], v[0])),
            math.degrees(math.atan2(v[2], math.hypot(v[0], v[1]))))


def build(trace_path: str, shots: list):
    """一场 → 样本列表。每条含：特征 / 真值方向 / 三档基线预测方向 / motion / run ✓"""
    tr = load_trace(trace_path)
    if not tr or "t" not in tr:
        return []
    R = gimbal_frame(tr)
    pos_w, present = plates_of(tr)
    t = tr["t"]
    aim_w = np.stack([tr["g_aim_x"], tr["g_aim_y"], tr["g_aim_z"]], axis=1)
    aim_valid = tr.get("g_aim_valid", np.ones_like(t))
    f = {k: tr.get(k, np.zeros_like(t)) for k in FEATURES}

    out = []
    for s in shots:
        ts = float(s.get("t", -1.0))
        if ts < 0:
            continue
        i = int(np.searchsorted(t, ts, side="right") - 1)
        if i < 20 or i + 4 >= t.size or not aim_valid[i]:
            continue
        tau = float(f["fly_time_est"][i])
        if not (0.02 <= tau <= 0.8):
            continue
        # 瞄准的板（世界系最近可见板 ✓）
        d = np.linalg.norm(pos_w[i] - aim_w[i], axis=1)
        d[~present[i]] = np.inf
        pid = int(np.argmin(d))
        if not np.isfinite(d[pid]) or d[pid] > 0.6:
            continue
        # 真值：同 id 在 t+τ（±5 帧内插值 ✓）
        target_t = ts + tau
        jj = int(np.searchsorted(t, target_t, side="right") - 1)
        lo, hi = max(0, jj - 5), min(t.size - 1, jj + 5)
        cand = [m for m in range(lo, hi + 1) if present[m, pid]]
        if len(cand) < 2:
            continue
        m1 = m2 = cand[0]
        for m in cand:
            if t[m] <= target_t:
                m1 = m
            if t[m] >= target_t:
                m2 = m
                break
        span = t[m2] - t[m1]
        alpha = 0.0 if span <= 0 else min(max((target_t - t[m1]) / span, 0.0), 1.0)
        truth_w = pos_w[m1, pid] * (1 - alpha) + pos_w[m2, pid] * alpha
        if not np.all(np.isfinite(truth_w)):
            continue
        # 统一到**开火时刻云台系** ✓
        Rg = R[i].T
        truth_g = Rg @ truth_w
        aim_g = Rg @ aim_w[i]
        pos_g = Rg @ pos_w[i, pid]

        # oracle 匀速：真值位置 + 真值速度（同 id 最近两帧 ✓）
        prev = None
        for m in range(i - 1, max(0, i - 10), -1):
            if present[m, pid]:
                prev = m
                break
        if prev is None:
            continue
        vel_w = (pos_w[i, pid] - pos_w[prev, pid]) / max(t[i] - t[prev], 1e-6)
        p_oracle = pos_g + (Rg @ vel_w) * tau

        # 因果匀速：EKF 位置/速度/角速度（云台系 ✓ 直接用 ✓）+ 板相对偏移
        xe = np.array([f["ekf0"][i], 0.0, f["ekf2"][i]])       # x, y, z（云台系 ✓）
        ve = np.array([f["ekf1"][i], 0.0, f["ekf3"][i]])
        r_vec = aim_g - xe
        p_ekf = xe + ve * tau + rotz(float(f["ekf7"][i]) * tau) @ r_vec

        # ① 补"板几何"特征（诊断：缺 r/相位 ⇒ 模型推不出"那块板"在哪 ✓）
        #   r / φ 从"EKF 中心 → 瞄准点"反算（因果 ✓，不用真值 ✓）；
        #   φ_相对 = φ − EKF yaw ⇒ 对固定安装的装甲板近似常数 ✓（可当"哪块板"的身份 ✓）
        r_xy = float(math.hypot(r_vec[0], r_vec[1]))
        phi = float(math.atan2(r_vec[1], r_vec[0]))
        phi_rel = math.degrees(wrap(np.radians(phi - float(f["ekf6"][i]))))
        v_tan = abs(float(f["ekf7"][i])) * r_xy
        extra = [r_xy, phi_rel, float(r_vec[2]), v_tan]

        # ② 学"状态残差"：真值板速度 − EKF 隐含的板速度 ✓（有物理量纲 ✓）
        #   EKF 板速度 = 中心速度 + ω×r（w = ekf7 ✓）
        w = float(f["ekf7"][i])
        v_plate_ekf = ve + np.array([-w * r_vec[1], w * r_vec[0], 0.0])
        v_plate_true = Rg @ vel_w
        dv = v_plate_true - v_plate_ekf

        out.append(dict(
            x=[float(f[k][i]) for k in FEATURES] + extra,
            truth=truth_g.tolist(),
            now=aim_g.tolist(),
            p_cv=p_ekf.tolist(),
            p_oracle=p_oracle.tolist(),
            dv=dv.tolist(),
            motion=str(s.get("motion", "-")), t=ts, tau=tau,
        ))
    return out


# ---------------------------------------------------------------- 模型
def ridge_fit(X: np.ndarray, Y: np.ndarray, lam: float = 1.0):
    X1 = np.hstack([X, np.ones((X.shape[0], 1))])
    I = np.eye(X1.shape[1])
    I[-1, -1] = 0.0
    return np.linalg.solve(X1.T @ X1 + lam * I, X1.T @ Y)


def ridge_pred(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.hstack([X, np.ones((X.shape[0], 1))]) @ W


def mlp_fit(X: np.ndarray, Y: np.ndarray, seed: int = 0):
    """小 MLP（22-64-64-2）⇒ 返回 predict ✓（torch 已在 venv 里 ✓，几千样本秒级 ✓）"""
    import torch

    torch.manual_seed(seed)
    net = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], 64), torch.nn.ReLU(),
        torch.nn.Linear(64, 64), torch.nn.ReLU(),
        torch.nn.Linear(64, Y.shape[1]),
    )
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
    lossf = torch.nn.MSELoss()
    xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(Y, dtype=torch.float32)
    for _ in range(400):
        opt.zero_grad()
        lossf(net(xt), yt).backward()
        opt.step()

    def predict(Z):
        with torch.no_grad():
            return net(torch.tensor(Z, dtype=torch.float32)).numpy()

    return predict


def ang_err(pred_deg: np.ndarray, truth_deg: np.ndarray) -> np.ndarray:
    return np.degrees(wrap(np.radians(pred_deg - truth_deg)))


def rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(a * a))) if a.size else float("nan")


def evaluate(rows: list, split: str) -> dict:
    """按 split 做留出，返回各方法的 (yaw RMS, pitch RMS, yaw 中位, pitch 中位) ✓"""
    runs = np.array([r["run"] for r in rows])
    motions = np.array([r["motion"] for r in rows])
    X = np.array([r["x"] for r in rows], dtype=float)
    TAU = np.array([r["tau"] for r in rows], dtype=float)
    truth = np.array([r["truth"] for r in rows], dtype=float)
    P = {k: np.array([r[k] for r in rows], dtype=float)
         for k in ("now", "p_cv", "p_oracle")}
    Ydir = np.array([yawpitch(v) for v in truth])            # (N,2) 目标方向 ✓
    Ydv = np.array([r["dv"] for r in rows], dtype=float)     # (N,3) 速度残差 ✓

    dir_pred = {k: np.zeros_like(Ydir) for k in ("ridge", "mlp")}
    pos_pred = {k: np.zeros_like(P["p_cv"]) for k in ("ridge_r", "mlp_r")}

    if split == "motion":
        folds = [(motions != g, motions == g) for g in sorted(set(motions))]
    elif split == "run":
        uniq = sorted(set(runs))
        folds = []
        for k in range(0, len(uniq), 4):
            hold = list(uniq[k:k + 4])
            folds.append((~np.isin(runs, hold), np.isin(runs, hold)))
    else:
        idx = np.random.default_rng(0).permutation(len(rows))
        tr = np.zeros(len(rows), bool)
        tr[idx[: int(0.7 * len(rows))]] = True
        folds = [(tr, ~tr)]

    for tr_m, te_m in folds:
        if te_m.sum() == 0 or tr_m.sum() < 50:
            continue
        mu, sd = X[tr_m].mean(axis=0), X[tr_m].std(axis=0) + 1e-9
        Xtr, Xte = (X[tr_m] - mu) / sd, (X[te_m] - mu) / sd
        # ① 直接给方向
        ym, ys = Ydir[tr_m].mean(axis=0), Ydir[tr_m].std(axis=0) + 1e-9
        Ytr = (Ydir[tr_m] - ym) / ys
        dir_pred["ridge"][te_m] = ridge_pred(ridge_fit(Xtr, Ytr, 3.0), Xte) * ys + ym
        dir_pred["mlp"][te_m] = mlp_fit(Xtr, Ytr)(Xte) * ys + ym
        # ② 给速度残差 ⇒ 加到几何外推上（× τ ✓）
        ym2, ys2 = Ydv[tr_m].mean(axis=0), Ydv[tr_m].std(axis=0) + 1e-9
        Ytr2 = (Ydv[tr_m] - ym2) / ys2
        dvh = ridge_pred(ridge_fit(Xtr, Ytr2, 3.0), Xte) * ys2 + ym2
        pos_pred["ridge_r"][te_m] = P["p_cv"][te_m] + dvh * TAU[te_m][:, None]
        dvh2 = mlp_fit(Xtr, Ytr2)(Xte) * ys2 + ym2
        pos_pred["mlp_r"][te_m] = P["p_cv"][te_m] + dvh2 * TAU[te_m][:, None]

    out = {}
    for name, Q in list(P.items()) + list(pos_pred.items()):
        py = np.array([yawpitch(v) for v in Q])
        e = ang_err(py, Ydir)
        out[name] = (rms(e[:, 0]), rms(e[:, 1]),
                     float(np.median(np.abs(e[:, 0]))), float(np.median(np.abs(e[:, 1]))))
    for name, D in dir_pred.items():
        e = ang_err(D, Ydir)
        out[name] = (rms(e[:, 0]), rms(e[:, 1]),
                     float(np.median(np.abs(e[:, 0]))), float(np.median(np.abs(e[:, 1]))))
    return out


def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else "/root/gpufree-data/runs/battery"
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
        for r in build(tr_path, shots):
            r["run"] = base
            rows.append(r)
    if not rows:
        print("没有样本 ✗")
        return 1

    print("# 学预测评估（特征 = trace 的 22 维前馈量 ✓；标签 = 开火时刻云台系里的未来方向 ✓）\n")
    print("样本 **%d 发**，%d 场，%d 种机动 ✓\n"
          % (len(rows), len({r["run"] for r in rows}), len({r["motion"] for r in rows})))
    print("| 划分 | 方法 | yaw RMS | pitch RMS | yaw 中位 | pitch 中位 |")
    print("|---|---|---|---|---|---|")
    for split, label in (("motion", "按机动留出（A1 口径 ✓）"),
                         ("run", "按场留出 ✓")):
        res = evaluate(rows, split)
        for name in ("now", "p_cv", "p_oracle", "ridge", "mlp", "ridge_r", "mlp_r"):
            a = res[name]
            tag = {"p_cv": " ← 几何外推（对手）", "p_oracle": " ← 上界",
                   "ridge": " ← 学方向", "mlp": " ← 学方向",
                   "ridge_r": " ← **学残差**", "mlp_r": " ← **学残差**"}.get(name, "")
            print("| %s | `%s`%s | %.2f° | %.2f° | %.2f° | %.2f° |"
                  % (label, name, tag, a[0], a[1], a[2], a[3]))
        print()

    print("> **验收线**：`ridge_r`/`mlp_r` 在**按机动留出**下 yaw RMS 低于 `p_cv`（4.75°）才算有价值 ✓")
    print("> 上界 `p_oracle` = 3.14° ✓（`SUMMARY_CEILING.md` ✓）")
    return 0

    print("> **验收线**：`ridge`/`mlp` 在**按机动留出**下 yaw RMS 低于 `cv_ekf` 才算有价值 ✓")
    print("> 参考：oracle 上界 yaw ≈ 3.1° ✓（`SUMMARY_CEILING.md` ✓）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


