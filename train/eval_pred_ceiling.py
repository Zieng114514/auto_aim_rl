#!/usr/bin/env python3
"""预测天花板评估（A1 方法）：在**角空间**量化"预测还有多少可学空间"。

问题
----
闭环里每发的命中由"开火时刻的瞄准点误差"决定（水平 yaw / 竖直 pitch 两轴）。
经典 Aimer 已经在做"匀速 + 小陀螺"外推 ⇒ 想再提升，只能靠**更好的预测**。
本脚本用**真值**构造理想预测的**上界（oracle）**，回答三件事：

1. `now`   ：现在 Aimer 的预测误差有多大（对照基准 ✓）
2. `cv`    ：**完美匀速外推**能到多少（用真值速度 ⇒ oracle ✓）
3. `circ`  ：**完美圆周外推**能到多少（水平面拟合圆 + 匀速角速度 ⇒ oracle ✓）

⇒ `now` 与 `cv/circ` 之间的差距 = **任何预测器（含学习模型）最多能拿到的收益** ✓
   （这就是 A1 说的"天花板 ≈ −44%（水平）"的复现口径 ✓）

数据
----
`runs/battery/traces/*.csv`（54 列）：22 个特征 + `g_gt{0..3}_{x,y,z,id}` 四块板真值 +
`g_R00..22` 云台→世界旋转 + `g_aim_{x,y,z}` 瞄准点 + `g_aim_valid` ✓

口径（重要 ✓）
--------------
* 全部换算到**云台系**再算角：`p_gim = Rᵀ · p_world`（R = 云台→世界 ✓）
* τ = `f_fly_time_est`（飞行时间 ✓），只取 0.02 ~ 0.8 s
* "瞄准的板" = 时刻 t 与瞄准点最近的**真值板**；真值 = **同一块板**在 `t+τ` 的位置 ✓
* 误差按 yaw（水平）/ pitch（竖直）两轴分别统计 ✓（与命中直接相关 ✓）
* 机动分类用**物理量**（不依赖预设名 ✓）：水平速度 |v| 与板绕车的角速度 |ψ̇| ⇒
  still / rotate / translate / both 四类 ✓ —— 正好对应"纯转 / 纯平移 / 平移+转"

用法
----
    python3 eval_pred_ceiling.py /root/gpufree-data/runs/battery/traces
"""
from __future__ import annotations

import csv
import glob
import math
import os
import sys

import numpy as np

R_EARTH = 6.371e6  # 仅用于避免除零的尺度参考，不参与物理


# ---------------------------------------------------------------- 读取
def load_trace(path: str) -> dict:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    out = {}
    for k in rows[0].keys():
        name = k[2:] if k.startswith("f_") else k
        try:
            out[name] = np.array([float(r[k]) for r in rows])
        except (TypeError, ValueError):
            continue
    return out


def gimbal_frame(tr: dict) -> np.ndarray:
    """返回 (N, 3, 3) 的 云台→世界 旋转矩阵序列 ✓（没有就退化为单位阵）。"""
    keys = ["g_R00", "g_R01", "g_R02", "g_R10", "g_R11", "g_R12", "g_R20", "g_R21", "g_R22"]
    if not all(k in tr for k in keys):
        n = len(tr.get("t", []))
        return np.tile(np.eye(3), (n, 1, 1))
    m = np.stack([tr[k] for k in keys], axis=1)          # (N, 9)
    return m.reshape(-1, 3, 3)


def to_gimbal(R: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """世界 → 云台系：p_g = Rᵀ · p_w ✓（pts 可为 (N,3) 或 (N,K,3) ✓）。"""
    return np.einsum("nji,n...j->n...i", R, pts)


def angles(v: np.ndarray) -> tuple:
    """云台系向量 → (yaw, pitch) 角（弧度）。pitch 抬头为正（与射击仰角同向 ✓）。"""
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    yaw = np.arctan2(y, x)
    pitch = np.arctan2(z, np.hypot(x, y))
    return yaw, pitch


def wrap(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------- 圆周拟合
def fit_circle(xy: np.ndarray):
    """代数最小二乘拟合圆：返回 (cx, cy, r)；退化时返回 None ✓。"""
    if xy.shape[0] < 8:
        return None
    x, y = xy[:, 0], xy[:, 1]
    A = np.stack([x, y, np.ones_like(x)], axis=1)
    b = -(x * x + y * y)
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cy, c = -sol[0] / 2, -sol[1] / 2, sol[2]
    r2 = cx * cx + cy * cy - c
    if not np.isfinite(r2) or r2 <= 0:
        return None
    r = math.sqrt(r2)
    # 半径离谱（比窗口内点间距大太多）⇒ 视为"基本是直线" ⇒ 交给 cv 更合适 ✓
    span = float(np.max(np.hypot(x - x.mean(), y - y.mean())))
    if r > 8.0 * max(span, 0.05):
        return None
    return float(cx), float(cy), float(r)


def predict_circle(xy_hist: np.ndarray, tau: float, dt_s: float):
    """用历史水平轨迹做圆周外推（匀速角速度 ✓），退化时返回 None。

    ⚠ 相位必须从**窗口末点**外推 ✗→✓：之前用 `a0 + w·τ`（a0 = 窗口起点角 ✗）
    ⇒ 等于把 τ 换成"从 2 s 前开始"✗，圆外推当然比现状还差 ✗（实测 +260% ✗）。
    """
    fit = fit_circle(xy_hist)
    if fit is None:
        return None
    cx, cy, r = fit
    ang = np.unwrap(np.arctan2(xy_hist[:, 1] - cy, xy_hist[:, 0] - cx))
    n = ang.size
    if n < 4:
        return None
    tt = (np.arange(n) - (n - 1)) * max(dt_s, 1e-6)       # 末点 = 0 ✓
    A = np.stack([tt, np.ones_like(tt)], axis=1)
    (w, a_last), *_ = np.linalg.lstsq(A, ang, rcond=None)
    a = a_last + w * tau                                  # 从末点外推 ✓
    return float(cx + r * math.cos(a)), float(cy + r * math.sin(a))


# ---------------------------------------------------------------- 样本
def plates_of(tr: dict, n_ids: int = 4):
    """按 **id** 建序列：返回 (N,4,3) 位置（缺 = NaN）与 (N,4) 存在位 ✓。

    ⚠ 不能按"槽位"用：仿真每帧只写可见的 1~2 块板 ✗，槽位会被不同 id 复用 ✗
    ⇒ 直接按槽位差分会把"两块不同的板"当成"一块板在飞"✗（实测导致速度离谱、oracle 反而更差 ✓）。
    """
    n = len(tr["t"])
    pos = np.full((n, n_ids, 3), np.nan)
    ok = np.zeros((n, n_ids), dtype=bool)
    for slot in range(4):
        if ("g_gt%d_x" % slot) not in tr:
            continue
        x, y, z = tr["g_gt%d_x" % slot], tr["g_gt%d_y" % slot], tr["g_gt%d_z" % slot]
        gid = tr["g_gt%d_id" % slot]
        for i in range(n):
            k = int(gid[i])
            if 0 <= k < n_ids:
                pos[i, k] = (x[i], y[i], z[i])
                ok[i, k] = True
    return pos, ok


def build_samples(trace_path: str, shots: list):
    """把一场的 (trace, shot_log) 变成样本：每个开火时刻一条 ✓。

    每条样本用**开火前最近的一行 trace**（与 build_pred_dataset 同口径 ✓），
    真值 = 瞄准的那块板在 t+τ 的位置 ✓，predictor 只用 ≤ t 的信息（oracle 除外 ✓）。
    """
    tr = load_trace(trace_path)
    if not tr or "t" not in tr:
        return []
    R = gimbal_frame(tr)
    pos_w, present = plates_of(tr)
    t = tr["t"]
    aim_w = np.stack([tr["g_aim_x"], tr["g_aim_y"], tr["g_aim_z"]], axis=1)
    aim_valid = tr.get("g_aim_valid", np.ones_like(t))
    fly = tr.get("fly_time_est", np.full_like(t, 0.15))

    out = []
    for s in shots:
        ts = float(s.get("t", -1.0))
        tau = float(s.get("fly_time_est", 0.0)) or 0.0
        if ts < 0:
            continue
        i = int(np.searchsorted(t, ts, side="right") - 1)
        if i < 20 or i + 4 >= t.size:
            continue
        tau = float(fly[i]) if not tau else tau
        if not (0.02 <= tau <= 0.8):
            continue
        if not aim_valid[i]:
            continue
        # 瞄准的板：t 时刻离瞄准点最近的**可见**真值板 ✓（世界系 ✓）
        d = np.linalg.norm(pos_w[i] - aim_w[i], axis=1)
        d[~present[i]] = np.inf
        pid = int(np.argmin(d))
        if not np.isfinite(d[pid]) or d[pid] > 0.6:
            continue
        # 真值：同一块板在 t+τ 的位置（±5 帧内找可见帧并线性插值 ✓）
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
        f = 0.0 if span <= 0 else min(max((target_t - t[m1]) / span, 0.0), 1.0)
        truth = pos_w[m1, pid] * (1 - f) + pos_w[m2, pid] * f
        if not np.all(np.isfinite(truth)):
            continue
        # 历史（仅 ≤ t，同一 id ✓；**世界系** ✓）：板的位置
        hist = pos_w[max(0, i - 40):i + 1, pid]
        hist = hist[np.all(np.isfinite(hist), axis=1)]
        pos_now = pos_w[i, pid]
        # 速度：最近两帧**同一 id** 的可见样本（世界系 ✓ 真值差分 ⇒ oracle ✓）
        prev = None
        for m in range(i - 1, max(0, i - 10), -1):
            if present[m, pid]:
                prev = m
                break
        if prev is None or not np.all(np.isfinite(pos_now)):
            continue
        vel = (pos_now - pos_w[prev, pid]) / max(t[i] - t[prev], 1e-6)
        if not np.all(np.isfinite(vel)):
            continue
        # 预测器：now = Aimer 的瞄准点（现状 ✓）；cv/circ = oracle（用真值 ✓，**世界系** ✓）
        preds = {"now": aim_w[i], "cv": pos_now + vel * tau}
        dt_s = max(t[i] - t[i - 1], 1e-6)
        pc = predict_circle(hist[:, :2], tau, dt_s) if hist.shape[0] >= 8 else None
        if pc is not None:
            preds["circ"] = np.array([pc[0], pc[1], pos_now[2] + vel[2] * tau])

        # 角度：统一用**开火时刻**的云台旋转 R(i) 换算 ✓
        # （固定参考系 ✓ —— 之前按"逐帧云台系"算，云台跟着目标转 ⇒ 把转动当成了速度 ✗✗）
        Rg = R[i].T

        def ang(v_w):
            g = Rg @ np.asarray(v_w, dtype=float)
            return (math.degrees(math.atan2(g[1], g[0])),
                    math.degrees(math.atan2(g[2], math.hypot(g[0], g[1]))))

        ty, tp = ang(truth)
        row = dict(motion=str(s.get("motion", "-")), t=ts, tau=tau, hit=bool(s.get("hit")))
        for name, p in preds.items():
            if not np.all(np.isfinite(p)):
                continue
            y, pt = ang(p)
            row[name + "_yaw"] = math.degrees(wrap(np.radians(y - ty)))
            row[name + "_pitch"] = math.degrees(wrap(np.radians(pt - tp)))
        if "now_yaw" in row:
            out.append(row)
    return out


# ---------------------------------------------------------------- 统计
def rms(vals):
    a = np.asarray([v for v in vals if v is not None and np.isfinite(v)], dtype=float)
    return float(np.sqrt(np.mean(a * a))) if a.size else float("nan")


def med(vals):
    a = np.asarray([v for v in vals if v is not None and np.isfinite(v)], dtype=float)
    return float(np.median(np.abs(a))) if a.size else float("nan")


def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else "/root/gpufree-data/runs/battery"
    traces = sorted(glob.glob(os.path.join(root, "traces", "*.csv")))
    if not traces:
        print("没找到 trace ✗")
        return 1

    rows = []
    n_run = 0
    for tr_path in traces:
        base = os.path.splitext(os.path.basename(tr_path))[0]
        st_path = os.path.join(root, base + ".json")
        if not os.path.exists(st_path):
            print("  ! 缺 state：%s（跳过）" % st_path)
            continue
        try:
            import json
            shots = json.load(open(st_path, encoding="utf-8")).get("shot_log") or []
        except Exception:                                  # noqa: BLE001
            print("  ! state 解析失败：%s" % st_path)
            continue
        got = build_samples(tr_path, shots)
        if got:
            n_run += 1
            rows.extend(got)

    if not rows:
        print("没有可用样本 ✗")
        return 1

    print("# 预测天花板（角空间，RMS，单位 deg ✓）\n")
    print("样本：**%d 发**（来自 %d 场 trace=on 的场次 ✓）\n" % (len(rows), n_run))
    print("口径：`now` = Aimer 自己的瞄准点（现状）；`cv`/`circ` = **oracle**"
          "（用真值做匀速/圆周外推 ⇒ 任何预测器的上界 ✓）\n")

    motions = sorted({r["motion"] for r in rows}, key=lambda m: -sum(1 for r in rows if r["motion"] == m))
    print("| 机动 | 样本 | now 水平 | **cv 水平** | **circ 水平** | now 竖直 | cv 竖直 | circ 竖直 |")
    print("|---|---|---|---|---|---|---|---|")
    for mo in motions:
        g = [r for r in rows if r["motion"] == mo]
        print("| %s | %d | %.2f° | **%.2f°** | **%.2f°** | %.2f° | %.2f° | %.2f° |"
              % (mo, len(g),
                 rms(r.get("now_yaw") for r in g),
                 rms(r.get("cv_yaw") for r in g),
                 rms(r.get("circ_yaw") for r in g),
                 rms(r.get("now_pitch") for r in g),
                 rms(r.get("cv_pitch") for r in g),
                 rms(r.get("circ_pitch") for r in g)))
    print()

    for axis in ("yaw", "pitch"):
        now = rms(r.get("now_" + axis) for r in rows)
        cv = rms(r.get("cv_" + axis) for r in rows)
        ci = rms(r.get("circ_" + axis) for r in rows)
        print("* **%s**（RMS）：now %.2f° → cv **%.2f°**（%+.1f%%）→ circ **%.2f°**（%+.1f%%）"
              % (axis, now, cv, 100 * (cv - now) / now, ci, 100 * (ci - now) / now))
        print("  （中位|误差|：now %.2f° / cv %.2f° / circ %.2f° ✓ 中位数更能反应典型情况 ✓）"
              % (med(r.get("now_" + axis) for r in rows),
                 med(r.get("cv_" + axis) for r in rows),
                 med(r.get("circ_" + axis) for r in rows)))
    print()
    print("> 读法：`cv`/`circ` 比 `now` 低多少，就是**更好的预测**最多能把该轴误差压到多少 ✓")
    print("> 若某类机动上 cv/circ 与 now 几乎一样 ⇒ 这类靶**已经预测到位**，别在上面投 ML ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


