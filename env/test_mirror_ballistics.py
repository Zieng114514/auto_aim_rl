#!/usr/bin/env python3
"""镜像弹道的自检：默认（k=0）与闭式解一致；打开阻力后与独立参考解一致，且与 C++ 同表。

跑法：  python3 auto_aim_rl/env/test_mirror_ballistics.py
（不需要 MuJoCo，纯算术，秒级）

为什么要跟 C++ 对表：项目规矩是"C++ 与 Python 必须用同一套算法，改一处必须改两处"
（见 agnet.md §4 决策表）。本脚本打印的表格与
`sp_vision_25/sim/test_trajectory_drag.cpp` 的第 3 节**同一组工况、同一列格式**，
两边输出直接 diff 就能证明同源。
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mirror import G, trajectory  # noqa: E402

K = 0.02042          # = ½·1.225·0.47·π·0.0085²/0.0032（rm_sim/configs/ballistics.yaml）
CASES = [(24, 3, 0, K), (24, 5, 0, K), (24, 8, 0, K), (24, 10, 0, K),
         (22, 5, 0.2, K), (25, 10, -0.2, K), (24, 5, 0.2, 0.0)]
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print("  %s %s%s" % ("✓" if ok else "✗", name, ("  " + detail) if detail else ""))
    if not ok:
        FAILED.append(name)


def ref_z(v0: float, pitch: float, d: float, k: float, dt: float = 1e-4):
    """独立参考：细步 RK4 推进到 d，返回 (z, t)。比生产用的 2 ms 细 20 倍。"""
    vx, vz = v0 * math.cos(pitch), v0 * math.sin(pitch)
    x = z = t = 0.0
    for _ in range(2_000_000):
        s1 = math.hypot(vx, vz)
        a1x, a1z = -k * s1 * vx, -G - k * s1 * vz
        s2 = math.hypot(vx + 0.5 * dt * a1x, vz + 0.5 * dt * a1z)
        a2x, a2z = -k * s2 * (vx + 0.5 * dt * a1x), -G - k * s2 * (vz + 0.5 * dt * a1z)
        s3 = math.hypot(vx + 0.5 * dt * a2x, vz + 0.5 * dt * a2z)
        a3x, a3z = -k * s3 * (vx + 0.5 * dt * a2x), -G - k * s3 * (vz + 0.5 * dt * a2z)
        s4 = math.hypot(vx + dt * a3x, vz + dt * a3z)
        a4x, a4z = -k * s4 * (vx + dt * a3x), -G - k * s4 * (vz + dt * a3z)
        x0, z0, t0 = x, z, t
        vx += dt * (a1x + 2 * a2x + 2 * a3x + a4x) / 6.0
        vz += dt * (a1z + 2 * a2z + 2 * a3z + a4z) / 6.0
        x += vx * dt
        z += vz * dt
        t += dt
        if x >= d:
            frac = (d - x0) / (x - x0) if (x - x0) > 1e-12 else 0.0
            return z0 + (z - z0) * frac, t0 + (t - t0) * frac
    return z, t


def ref_pitch(v0: float, d: float, h: float, k: float, dt: float = 1e-4) -> float:
    lo, hi = -0.6, 0.7854
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if ref_z(v0, mid, d, k, dt)[0] < h:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main() -> int:
    print("== 1. k = 0：与闭式解一致（默认路径，历史行为）==")
    all_ok = True
    for (v0, d, h, _k) in CASES:
        unsolvable, _t, pitch = trajectory(v0, d, h)          # 不传 k ⇒ 闭式
        unsolvable2, _t2, pitch2 = trajectory(v0, d, h, 0.0)  # 显式 0 ⇒ 也走闭式
        same = (unsolvable == unsolvable2) and (pitch == pitch2)
        all_ok = all_ok and same
        print("    v0=%2.0f d=%4.1f h=%+5.2f | pitch=%.9f  %s" %
              (v0, d, h, pitch, "同式" if same else "**不一致**"))
    check("k=0 与 k 缺省行为完全一致（闭式解）", all_ok)

    print()
    print("== 2. k > 0：与独立参考解一致 + 方向性 ==")
    print("    %-5s %-5s %-6s %-11s %-10s %-11s %-8s %s" %
          ("v0", "d", "h", "pitch(镜像)", "同dt差(°)", "细dt差(°)", "t(ms)", "结论"))
    worse = 0
    for (v0, d, h, k) in CASES:
        if k <= 0.0:
            continue
        ok = trajectory(v0, d, h, k)
        if ok[0]:
            print("    %-5.0f %-5.1f %-6.2f unsolvable" % (v0, d, h))
            continue
        _, t_drag, p_drag = ok
        _, t_g, p_g = trajectory(v0, d, h)                    # 无阻力对照
        # 两把尺子：① 同 dt（2 ms）⇒ 检验"实现是否照算法做了"；② 细 dt（0.1 ms）⇒ 截断误差
        err_same = abs(p_drag - ref_pitch(v0, d, h, k, 2e-3)) * 180.0 / math.pi
        err_fine = abs(p_drag - ref_pitch(v0, d, h, k, 1e-4)) * 180.0 / math.pi
        higher, longer = p_drag > p_g, t_drag > t_g
        if not (err_same < 2e-3 and higher and longer):
            worse += 1
        print("    %-5.0f %-5.1f %-6.2f %-11.6f %-10.5f %-11.5f %-8.2f %s%s%s" %
              (v0, d, h, p_drag, err_same, err_fine, t_drag * 1e3,
               "同算法 " if err_same < 2e-3 else "**实现不符** ",
               "更高 " if higher else "**未抬高** ", "更久" if longer else "**未变久**"))
    print("    （列：pitch(镜像) | 与同 dt 参考的差 ° | 与细 dt 参考的差 ° = 截断误差）")
    check("开阻力后与同 dt 参考解一致（<0.002°）且抬得更高、飞得更久", worse == 0)

    print()
    print("== 3. 对表（应与 C++ test_trajectory_drag 的第 3 节逐行相同）==")
    print("    %-6s %-6s %-6s %-9s %-13s %s" % ("v0", "d", "h", "k", "pitch_deg",
                                                "fly_time_ms"))
    for (v0, d, h, k) in CASES:
        unsolvable, t, pitch = trajectory(v0, d, h, k)
        if unsolvable:
            print("    %-6.0f %-6.1f %-6.2f %-9.5f unsolvable" % (v0, d, h, k))
            continue
        print("    %-6.0f %-6.1f %-6.2f %-9.5f %-13.6f %.2f" %
              (v0, d, h, k, math.degrees(pitch), t * 1e3))

    print()
    if FAILED:
        print("❌ 失败 %d 项：%s" % (len(FAILED), "；".join(FAILED)))
        return 1
    print("✅ 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
