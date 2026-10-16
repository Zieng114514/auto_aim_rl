#!/usr/bin/env python3
"""A1 误差预算：把"每发的瞄准误差"拆开，判断该修哪一份。

**思路**：任何"学预测"再好也修不了"目标真实轨迹本身偏离可外推模型"的那一部分 ✓，
所以先把这个**天花板**量出来，再决定值不值得学 ✓。

对每一发（用 shot_log 的 off_dyaw/off_dpitch = 仿真自己测的"该修多少"当作总误差）：
  · `e_total`    : |(off_dyaw, off_dpitch)| × 1000 → mrad（含一切原因 ✓）
  · `e_maneuver` : 用**目标真值轨迹**在 [t−0.3, t] 上做局部二次外推，外推到 t+fly_time，
                   再与**真值未来位置**比 ⇒ "状态估计完美 + 最优短时外推"仍会偏的部分 ✓
                   **这是"学预测"的理论天花板**（学不出比它更低 ✗）
  · `e_est`      : e_total 与 e_maneuver 的合成（估计误差 + 弹道几何 + 选板 ✓）

判据（按靶型看）：
  · e_maneuver ≪ e_total ⇒ 学预测没用 ✗（去攻估计/跟踪/标定 ✓）
  · e_maneuver ≈ e_total ⇒ 有空间 ✓，而且天花板就是 e_maneuver ✓

用法：
    python auto_aim_rl/train/analyze_aim_budget.py --glob '/tmp/pred_trace_budget_*'
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys

import numpy as np

PAIR_WINDOW_S = 0.35        # 与数据集构建器一致：靠 n_shots 事件计数配对（这里只做兜底）
FIT_WIN_S = 0.30            # 外推用的历史窗口
PEAK_N = 3                  # 二次外推至少要几个点


def read_trace(path: str) -> dict:
    """trace.csv → {列名: float 数组}（f_/g_ 前缀剥掉；t 保留）。"""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    out = {}
    for k in rows[0].keys():
        name = k[2:] if k[:2] in ("f_", "g_") else k
        out[name] = np.array([float(r[k]) for r in rows], dtype=float)
    return out


def plate_series(tr: dict, pid: int):
    """取出某块板（g_gtN_id == pid）在每一 tick 的真值位置：返回 (t, xyz) 只含有效样本。"""
    for k in range(4):
        if f"gt{k}_id" not in tr:
            continue
        m = tr[f"gt{k}_id"] == float(pid)
        if m.any():
            t = tr["t"][m]
            xyz = np.stack([tr[f"gt{k}_x"][m], tr[f"gt{k}_y"][m], tr[f"gt{k}_z"][m]], axis=1)
            order = np.argsort(t)
            return t[order], xyz[order]
    return None, None


def ct_extrapolate(t_hist: np.ndarray, xyz_hist: np.ndarray, t_eval: float):
    """**匀速转动外推**（对旋转靶才是物理正确的"最优外推"）。

    为什么不用二次拟合：靶车甩头时 ω 可达 12 rad/s，板子在 0.3 s 内能转过 200° ✗，
    局部二次外推会飞出几米（实测"机动"一项被算成 17~80° ✗✗，比总误差大 10~100 倍）。

    做法：对窗口内的点做代数**圆拟合**（靶车自转 ⇒ 装甲板中心走圆 ✓），得到圆心/半径，
    再用相邻点定出角速度符号 ⇒ 沿圆外推到 t_eval ✓。
    拟合病态（几乎直线运动）时退回**线性外推** ✓。
    """
    n = t_hist.size
    if n < 4:
        return None
    x, y, z = xyz_hist[:, 0], xyz_hist[:, 1], xyz_hist[:, 2]
    A = np.stack([x, y, np.ones_like(x)], axis=1)
    b = x**2 + y**2
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        sol = None
    if sol is not None:
        cx, cy = sol[0] / 2.0, sol[1] / 2.0
        r2 = sol[2] + cx * cx + cy * cy
        if r2 > 1e-4:                       # 半径 > 1 cm 才算真的在转
            r = math.sqrt(r2)
            th = np.unwrap(np.arctan2(y - cy, x - cx))
            # 角速度：对 (t, θ) 做线性拟合
            dt = t_hist - t_hist[-1]
            om, th0 = np.polyfit(dt, th, 1) if n >= 2 else (0.0, th[-1])
            if abs(om) > 1e-3:
                th_eval = th0 + om * (t_eval - t_hist[-1])
                return np.array([cx + r * math.cos(th_eval),
                                 cy + r * math.sin(th_eval),
                                 float(np.interp(t_eval, t_hist, z))])
    # 退化：线性外推（用最后两个点定速度）
    v = (xyz_hist[-1] - xyz_hist[-2]) / max(t_hist[-1] - t_hist[-2], 1e-6)
    return xyz_hist[-1] + v * (t_eval - t_hist[-1])


def interp_xyz(t: np.ndarray, xyz: np.ndarray, t_eval: float):
    """线性插值到 t_eval；超出范围返回 None。"""
    if t_eval < t[0] or t_eval > t[-1]:
        return None
    return np.array([np.interp(t_eval, t, xyz[:, i]) for i in range(3)])


def transverse_mrad(v_world: np.ndarray, aim_gimbal: np.ndarray, R: np.ndarray, rng: float):
    """把世界系位移投到"瞄准方向的横向(水平)/竖直"上 → (yaw_mrad, pitch_mrad)。"""
    if rng <= 0.5 or np.linalg.norm(aim_gimbal) < 1e-6:
        return None
    u = R @ (aim_gimbal / np.linalg.norm(aim_gimbal))
    yh = np.cross(np.array([0.0, 0.0, 1.0]), u)
    ny = np.linalg.norm(yh)
    if ny < 1e-6:
        return None
    yh = yh / ny
    return float(v_world @ yh / rng * 1000.0), float(v_world[2] / rng * 1000.0)


def plate_id_from_name(name) -> int | None:
    """shot_log 的 plate 形如 "板3" → 3（与 trace 的 g_gtN_id 对应）。"""
    if not name:
        return None
    digits = "".join(ch for ch in str(name) if ch.isdigit())
    return int(digits) if digits else None


def analyze_run(trace_path: str, state_path: str, acc: dict) -> int:
    tr = read_trace(trace_path)
    if not tr:
        print(f"  ! trace 为空：{trace_path}", file=sys.stderr)
        return 0
    try:
        with open(state_path) as f:
            st = json.load(f)
    except Exception as e:                      # noqa: BLE001
        print(f"  ! state.json 读不了：{state_path}（{e}）", file=sys.stderr)
        return 0
    if "n_shots" not in tr:
        print(f"  ! trace 缺 n_shots 列（旧格式）：{trace_path}", file=sys.stderr)
        return 0

    shots = st.get("shot_log", [])
    order = np.argsort(tr["t"])
    t_tr, n_shots = tr["t"][order], tr["n_shots"][order]
    n_used = 0
    n_fallback = 0
    n_reject = 0
    for rec in shots:
        if "off_dyaw" not in rec or "off_dpitch" not in rec:
            continue
        pid = plate_id_from_name(rec.get("plate"))
        if pid is None:
            continue
        k = int(rec.get("id", 0))
        j = int(np.searchsorted(n_shots, k, side="left"))
        if j <= 0 or j >= n_shots.size:
            continue
        i = j - 1                                   # 指令那一 tick
        t0 = float(t_tr[i])

        aim = np.array([tr["aim_x"][i], tr["aim_y"][i], tr["aim_z"][i]])
        if tr["aim_valid"][i] < 0.5 or np.linalg.norm(aim) <= 0.5:
            continue
        R = np.array([[tr[f"R{r}{c}"][i] for c in range(3)] for r in range(3)])
        rng = float(np.linalg.norm(aim))
        fly = float(tr["fly_time_est"][i])

        tp, xp = plate_series(tr, pid)
        if tp is None or tp.size < PEAK_N + 1:
            # 兜底：万一 "板N" ↔ "armorN" 的映射不一致，就取"外推位置离瞄准点最近"的那块板
            # （仍要求它有足够历史）——避免因为命名差异静默丢掉全部样本 ✗
            best = None
            for cand in range(4):
                tc, xc = plate_series(tr, cand)
                if tc is None or tc.size < PEAK_N + 1:
                    continue
                mc = (tc >= t0 - FIT_WIN_S) & (tc <= t0 + 1e-6)
                pc = ct_extrapolate(tc[mc], xc[mc], t0 + fly)
                if pc is None:
                    continue
                # 在云台系里比（外推点 → 云台系）
                pc_g = R.T @ (pc - R @ np.zeros(3))     # 仅旋转对齐（原点近似为云台原点）
                d = np.linalg.norm(pc_g - aim)
                if best is None or d < best[0]:
                    best = (d, cand, tc, xc)
            if best is None:
                continue
            n_fallback += 1
            pid, tp, xp = best[1], best[2], best[3]
        mh = (tp >= t0 - FIT_WIN_S) & (tp <= t0 + 1e-6)
        p_ex = ct_extrapolate(tp[mh], xp[mh], t0 + fly)
        p_fu = interp_xyz(tp, xp, t0 + fly)
        if p_ex is None or p_fu is None:
            continue
        man = transverse_mrad(p_ex - p_fu, aim, R, rng)
        if man is None:
            continue
        # 防呆：外推断言（>1 m 就是跑飞了）⇒ 剔除并计数，避免垃圾数字静默进表 ✗
        if float(np.linalg.norm(p_ex - p_fu)) > 1.0:
            n_reject += 1
            continue

        key = str(rec.get("motion", "-"))
        a = acc.setdefault(key, {"n": 0, "et": [], "em": [], "etp": [], "emp": [], "gz": []})
        a["n"] += 1
        a["et"].append(abs(float(rec["off_dyaw"])) * 1000.0)
        a["etp"].append(abs(float(rec["off_dpitch"])) * 1000.0)
        a["em"].append(abs(man[0]))
        a["emp"].append(abs(man[1]))
        # **竖直模型缺口**：tracker 的预测里没有竖直运动学（z 恒定 ✗）⇒
        # |z_true(t+Δ) − z_true(t)| / R 就是它漏掉的那一份 ✓（升降靶上会很大 ✓）
        i0 = int(np.searchsorted(tp, t0, side="right")) - 1
        if 0 <= i0 < tp.size:
            gz = abs(float(xp[i0, 2] - p_fu[2])) / max(rng, 0.5) * 1000.0
            a["gz"].append(gz)
        n_used += 1
    if n_fallback or n_reject:
        print(f"  （其中 {n_fallback} 发用「最近板」兜底，{n_reject} 发因外推跑飞被剔除）", file=sys.stderr)
    return n_used


def main() -> int:
    ap = argparse.ArgumentParser(description="A1 误差预算")
    ap.add_argument("--glob", required=True, help="trace 通配，例如 '/tmp/pred_trace_budget_*'")
    ap.add_argument("--state-dir", default="/tmp")
    a = ap.parse_args()

    acc: dict = {}
    pairs = []
    for tr_path in sorted(glob.glob(a.glob)):
        base = os.path.basename(tr_path)
        if not (base.startswith("pred_trace_") and base.endswith(".csv")):
            continue
        tag = base[len("pred_trace_"):-len(".csv")]
        st_path = os.path.join(a.state_dir, f"eval_{tag}.json")
        if not os.path.exists(st_path):
            print(f"  ! 缺 state json：{st_path}（跳过）", file=sys.stderr)
            continue
        n = analyze_run(tr_path, st_path, acc)
        pairs.append((tag, n))
        print(f"  {tag}: 用了 {n} 发")

    if not acc:
        print("没有可用样本（是不是 trace 里没有 g_ 诊断列？需要重跑带新列的采集 ✓）",
              file=sys.stderr)
        return 1

    print("\n=== A1 误差预算（单位 mrad；1 mrad ≈ 每米 1 mm）===")
    print("%-18s %4s %9s %9s %6s %9s %9s %6s %9s"
          % ("靶型", "N", "总Δyaw", "机动Δyaw", "占比", "总Δpitch", "机动Δpitch", "占比", "竖直缺口"))
    tot = {"et": [], "em": [], "etp": [], "emp": [], "gz": []}
    for key in sorted(acc, key=lambda k: -acc[k]["n"]):
        d = acc[key]
        if d["n"] < 3:
            continue
        et = float(np.median(d["et"])); em = float(np.median(d["em"]))
        etp = float(np.median(d["etp"])); emp = float(np.median(d["emp"]))
        gz = float(np.median(d["gz"])) if d["gz"] else 0.0
        print("%-18s %4d %9.2f %9.2f %5.0f%% %9.2f %9.2f %5.0f%% %9.2f"
              % (key, d["n"], et, em, 100 * em / max(et, 1e-9),
                 etp, emp, 100 * emp / max(etp, 1e-9), gz))
        for s in ("et", "em", "etp", "emp", "gz"):
            tot[s].extend(d[s])

    if tot["et"]:
        et = float(np.median(tot["et"])); em = float(np.median(tot["em"]))
        etp = float(np.median(tot["etp"])); emp = float(np.median(tot["emp"]))
        print("\n总计（中位）：水平 总 %.2f mrad，其中「完美外推仍会偏」%.2f mrad（%.0f%%）"
              % (et, em, 100 * em / max(et, 1e-9)))
        print("              竖直 总 %.2f mrad，其中 %.2f mrad（%.0f%%）"
              % (etp, emp, 100 * emp / max(etp, 1e-9)))
        r = em / max(et, 1e-9)
        print("\n判据：机动占比 %.0f%% ⇒ %s"
              % (100 * r,
                 "学预测有空间（天花板就是它 ✓）" if r > 0.6 else
                 "学预测收益有限，优先攻估计/跟踪/标定 ✓" if r < 0.3 else
                 "两者都要看，建议先做估计质量 ✓"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

