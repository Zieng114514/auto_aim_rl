#!/usr/bin/env python3
"""阶段 2（预测）数据集构建：把闭环运行的 (trace.csv, state.json) 拼成 (特征, 标签)。

设计与口径见 `agnet_logs/20261015_1700_stage2_prediction_design.md`：
* **标签**来自 rm_sim 的 `shot_log`（逐发 off_h/off_v/off_n、range、motion）—— 已现成；
* **特征**来自 `sim_main --dump_trace` 的逐控制周期 CSV（只含真机拿得到的量）；
* 两者用 **sim 时间 t** 对齐：取"造成这一发的那个开火指令 tick"= `shoot==1 且 t <= t_shot` 的最后一行。

用法：
    # 单次运行（一个 trace + 一个 state json）
    python auto_aim_rl/train/build_pred_dataset.py \\
        --run /tmp/pred_trace_pred_aero_20251012.csv /tmp/eval_pred_aero_20251012.json \\
        --out /tmp/pred_ds.npz
    # 批量：按 trace 前缀自动找同名 state json
    python auto_aim_rl/train/build_pred_dataset.py --glob '/tmp/pred_trace_pred_aero_*' \\
        --out /tmp/pred_ds.npz

输出的 npz：
    X        (N, F) 特征（列名见 features / FEATURES）
    y        (N, 2) 标签：Δ_lat/Δ_vert（米，装甲板局部系 ≈ 世界水平/竖直）
    raw      (N, 4) off_h/off_v/off_n（米）+ range（米）
    motion   (N,)   靶型段编号（0=非脚本，其余=段序号）
    run      (N,)   运行编号（**按运行留出验证集时用它**，别按样本随机划分）
    shot_id  (N,)   该运行内的发次序号
    t        (N,)   发次时刻（sim 秒）
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

# 特征列 = sim_main 的 kPredFeatureNames（C++ 侧是**唯一真相**，这里必须逐字一致）。
# 顺序/含义改动 ⇒ 数据集与已训模型全部作废，必须重采重训。
FEATURES = [
    "has_target",
    "ekf0", "ekf1", "ekf2", "ekf3", "ekf4", "ekf5", "ekf6", "ekf7",
    "target_w",
    "gimbal_yaw", "gimbal_pitch", "cmd_yaw", "cmd_pitch",
    "err_yaw", "err_pitch",
    "control", "shoot", "aim_valid",
    "armor_num", "target_num",
    "bullet_speed", "heat_frac", "locked",
    "range_est", "fly_time_est",
    # ---- v2 窗口历史特征（与 C++ pred_feature_names() 逐字一致）----
    "win_ok", "d_vx_2", "d_vy_2", "d_w_1", "d_erry_1", "d_errp_1",
    "d_cmd_yaw_1", "d_range_2", "mean_erry_2", "std_erry_2",
]

# 训练时丢弃的行：启动阶段历史不足（win_ok=0）——那时窗口特征全是 0 ✗ 会教坏模型
DROP_IF_ZERO = ["win_ok"]

# 开火指令 → "仿真真的打出这一发" 之间允许的时差；超过就认为没配上（宁缺勿错）
PAIR_WINDOW_S = 0.35


def read_trace(path: str) -> dict:
    """读逐控制周期 CSV → {列名: 数组}。

    表头两段式：第 1 列 `t`（sim 时间，用来和 shot_log 对齐），其后是 `f_<特征名>`。
    这样"时间"和"特征"永远不会混（踩过的坑：表头一改，对齐列就没了）。
    """
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    out = {}
    for k in rows[0].keys():
        name = k[2:] if k.startswith("f_") else k
        out[name] = np.array([float(r[k]) for r in rows])
    return out


def motion_code(motion: str) -> int:
    """靶型段名 → 编号（`-` = 非脚本模式；其余取名字里的数字，没有则 0）。"""
    if not motion or motion == "-":
        return 0
    digits = "".join(ch for ch in motion if ch.isdigit())
    return int(digits) if digits else 0


def pair_run(trace_path: str, state_path: str, run_id: int):
    """一次运行 → (X, y, raw, motion, run, shot_id, t)；失败返回 None。"""
    tr = read_trace(trace_path)
    if not tr:
        print(f"  ! trace 为空：{trace_path}", file=sys.stderr)
        return None
    try:
        with open(state_path) as f:
            st = json.load(f)
    except Exception as e:                      # noqa: BLE001
        # 闭环偶尔会落下一个截断的 state.json（评测被中断/写盘竞争）⇒ 跳过这一次运行，
        # 不要让一条坏 json 把整个数据集构建搞崩（踩过：JSONDecodeError 直接终止）
        print(f"  ! state json 解析失败，跳过：{state_path}（{e}）", file=sys.stderr)
        return None
    shots = st.get("shot_log", [])
    if not shots:
        print(f"  ! shot_log 为空：{state_path}", file=sys.stderr)
        return None

    t_tr = np.asarray(tr["t"], float)
    shoot = np.asarray(tr["shoot"], float)
    # n_shots（仿真侧发次计数）用于**精确配对**；旧 trace 没有这一列时退回时间窗
    shots_cnt = np.asarray(tr.get("n_shots", np.zeros_like(t_tr)), float)
    order = np.argsort(t_tr)          # trace 本来就是时序，保险起见排一下
    t_tr, shoot, shots_cnt = t_tr[order], shoot[order], shots_cnt[order]
    have_cnt = bool(shots_cnt.size and shots_cnt.max() > 0)

    X, y, raw, motion, run, shot_id, ts = [], [], [], [], [], [], []
    missed = 0
    paired_by_cnt = 0
    for rec in shots:
        t_shot = float(rec["t"])
        k = int(rec.get("id", 0))
        i = None
        if have_cnt and k > 0:
            # n_shots 首次 ≥ k 的那一行 = 这一发在仿真里出现的那 tick ⇒ 取它**前一行**
            # （= 下发指令的那 tick）。这样绕开 rm_link 的 pending_fire 延迟 ✗→✓。
            j = int(np.searchsorted(shots_cnt, k, side="left"))
            if j < shots_cnt.size and shots_cnt[j] >= k:
                i = max(j - 1, 0)
                paired_by_cnt += 1
        if i is None:
            # 旧 trace（没有 n_shots 列）：退回时间窗（会漏掉一批，见下面的统计输出）
            cand = np.where((shoot > 0.5) & (t_tr <= t_shot) & (t_tr >= t_shot - PAIR_WINDOW_S))[0]
            if cand.size == 0:
                missed += 1
                continue
            i = int(cand[-1])

        X.append([float(tr[k][i]) for k in FEATURES])     # 列 = C++ 的 kPredFeatureNames
        # 标签：落点偏差 → "瞄准点该往哪挪"（板局部系 ≈ 世界水平/竖直，米）
        y.append([-float(rec["off_h"]), -float(rec["off_v"])])
        raw.append([float(rec["off_h"]), float(rec["off_v"]),
                    float(rec.get("off_n", 0.0)), float(rec.get("range", 0.0) or 0.0)])
        motion.append(motion_code(rec.get("motion", "-")))
        run.append(run_id)
        shot_id.append(int(rec.get("id", len(shot_id) + 1)))
        ts.append(t_shot)

    if missed or paired_by_cnt:
        print(f"  · 按发次计数精确配对 {paired_by_cnt} 发；时间窗兜底/漏掉 {missed} 发", file=sys.stderr)
    if not X:
        return None
    return (np.asarray(X, float), np.asarray(y, float), np.asarray(raw, float),
            np.asarray(motion, int), np.asarray(run, int),
            np.asarray(shot_id, int), np.asarray(ts, float))


def main() -> int:
    ap = argparse.ArgumentParser(description="阶段 2 预测数据集构建")
    ap.add_argument("--run", nargs=2, action="append", metavar=("TRACE", "STATE"),
                    help="一次运行的 (trace.csv, state.json)")
    ap.add_argument("--glob", help="按 trace 通配批量；同名 state json 去 --state-dir 找")
    ap.add_argument("--state-dir", default="/tmp", help="批量模式里 state json 所在目录")
    ap.add_argument("--out", required=True, help="输出 npz 路径")
    a = ap.parse_args()

    pairs = list(a.run or [])
    if a.glob:
        for tr in sorted(glob.glob(a.glob)):
            base = os.path.basename(tr)
            # pred_trace_<label>_<seed>.csv → <state-dir>/eval_<label>_<seed>.json
            if base.startswith("pred_trace_") and base.endswith(".csv"):
                tag = base[len("pred_trace_"):-len(".csv")]
                pairs.append((tr, os.path.join(a.state_dir, f"eval_{tag}.json")))
    if not pairs:
        print("没有输入（用 --run 或 --glob）", file=sys.stderr)
        return 2

    parts = []
    for rid, (tr, st) in enumerate(pairs):
        if not os.path.exists(st):
            print(f"  ! 缺 state json：{st}（跳过）", file=sys.stderr)
            continue
        r = pair_run(tr, st, rid)
        if r is None:
            continue
        parts.append(r)
        print(f"  run{rid}: {r[0].shape[0]:4d} 发  ← {os.path.basename(tr)}")
    if not parts:
        print("没有可用样本", file=sys.stderr)
        return 1

    cat = lambda i: np.concatenate([p[i] for p in parts])     # noqa: E731
    X, y, raw, motion, run, shot_id, t = (cat(i) for i in range(7))

    np.savez(a.out, X=X, y=y, raw=raw, motion=motion, run=run, shot_id=shot_id, t=t,
             features=np.array(FEATURES, dtype=object))
    print(f"\n合计 {X.shape[0]} 发 → {a.out}")
    print(f"  水平 off_h：1σ {np.std(raw[:, 0]) * 1000:6.1f} mm"
          f"  均值 {np.mean(raw[:, 0]) * 1000:+7.1f} mm")
    print(f"  竖直 off_v：1σ {np.std(raw[:, 1]) * 1000:6.1f} mm"
          f"  均值 {np.mean(raw[:, 1]) * 1000:+7.1f} mm")
    uniq, cnt = np.unique(motion, return_counts=True)
    print("  按靶型段：" + "  ".join(f"{u}:{c}" for u, c in zip(uniq, cnt)))
    uniq, cnt = np.unique(run, return_counts=True)
    print("  按运行：" + "  ".join(f"{u}:{c}" for u, c in zip(uniq, cnt)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


