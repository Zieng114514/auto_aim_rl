"""sp_vision 的 Python 镜像：跟踪 + 弹道解算 + 提前量，给 RL 训练环境提供**同口径**特征。

为什么需要镜像
--------------
RL 训练不可能直接跑 sp_vision 的 C++ 闭环（那条链路带渲染，只有 ~23 fps），
所以感知这一层必须在 Python 侧重建。重建的原则是：

* **弹道解算逐行照搬** \`sp_vision_25/tools/trajectory.cpp\`（闭式解，g=9.7833），
  同一个模型、同一个提前量迭代逻辑 —— 这部分是确定性的，没有"近似"的余地；
* **跟踪器不假装复刻** EKF：真值 + 延迟 + 平滑噪声（OU），参数可随机化。
  理由是 sp_vision 的 tracker 是调过的，逐行复刻成本极高且未必更真；
  而"估计误差有多大、滞后多少"才是策略真正需要扛的东西。
  **代价**：训练/部署仍存在分布差，所以最终一切数字都要在 C++ 闭环里复测
  （见 auto_aim_rl/DESIGN.md §1）——这也是为什么验收只认 eval_policy.sh 的结果。

所有参数集中在 MirrorParams，训练时随机化（域随机化）即可覆盖"不同车/不同跟踪质量"。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# 与 sp_vision_25/tools/trajectory.cpp 保持一致
G = 9.7833


def trajectory(v0: float, d: float, h: float):
    """闭式弹道解（照搬 trajectory.cpp）。返回 (unsolvable, fly_time, pitch)，pitch 抬头为正。"""
    a = G * d * d / (2.0 * v0 * v0)
    b = -d
    c = a + h
    delta = b * b - 4.0 * a * c
    if delta < 0.0:
        return True, 0.0, 0.0
    sq = math.sqrt(delta)
    tan1 = (-b + sq) / (2.0 * a)
    tan2 = (-b - sq) / (2.0 * a)
    p1, p2 = math.atan(tan1), math.atan(tan2)
    t1 = d / (v0 * max(math.cos(p1), 1e-6))
    t2 = d / (v0 * max(math.cos(p2), 1e-6))
    if t1 < t2:
        return False, t1, p1
    return False, t2, p2


@dataclass
class MirrorParams:
    """镜像跟踪器的"有多不准"。默认值由实测量级推出，训练时随机化。"""

    latency_s: float = 0.030        # 估计滞后（图像+链路），默认约一帧
    pos_noise_deg: float = 0.45     # 位置估计噪声（在垂直于视线的平面上）1σ，等效角度
    vel_noise: float = 0.25         # 速度估计噪声 1σ（m/s）
    omega_noise: float = 0.60       # 角速度估计噪声 1σ（rad/s）
    yaw_noise: float = 0.05         # 车体朝向估计噪声 1σ（rad）
    # **指令抖动**（2026-10-13 加，本轮最关键的保真度修复之一）：
    # C++ 实测 |ey| 中位 3.13°、p95 12.1°，但同样条件下命中率仍有 32.5% ——
    # 说明那 3° 是**瞬态滞后**（跟踪器的逐帧估计抖动 → 指令跳 → 云台追不上），
    # 而不是"瞄准点本身不准"。这两件事对策略的意义完全不同：
    #   * 瞬态滞后：云台稳下来的时刻，打出去就是准的（规则门限正是靠"卡 0.5°"吃到这一点）；
    #   * 瞄准点不准：怎么等都打不准（把估计噪声调大就是这种情况，命中率会崩到 14%）。
    # 所以这里对**指令**加一个短相关时间的抖动，让云台物理去追 ——
    # 既有真实的 |ey| 分布，又不破坏"稳了就准"这个关键性质。
    cmd_jitter_deg: float = 2.0     # 指令抖动幅度 1σ（度）
    cmd_jitter_tau: float = 0.05    # 抖动相关时间（秒，越短越"跳"）
    tau: float = 0.20               # 噪声相关时间 s（跟踪器输出是平滑的，不是白噪声）
    dropout_prob: float = 0.0       # 单帧丢失概率（检测抖动/遮挡）


class MirrorTracker:
    """把世界真值变成"带延迟与平滑误差"的目标估计。

    状态量：车体位置、水平速度、yaw 角速度。时间对齐用一个小环形缓冲实现延迟。
    """

    def __init__(self, p: MirrorParams, fps: float, rng):
        self.p = p
        self.fps = fps
        self.rng = rng
        self._hist: list[dict] = []
        self._delay_n = max(0, int(round(p.latency_s * fps)))
        # OU 噪声状态（位置用视线垂直平面上的两个分量）
        self._e_pos = np.zeros(3)
        self._e_vel = np.zeros(2)
        self._e_omega = 0.0
        self._e_yaw = 0.0
        self.visible = True
        self.t = 0.0

    # ------------------------------------------------------------------ 内部
    def _ou(self, cur, sigma, dt):
        k = math.sqrt(2.0 * dt / max(self.p.tau, 1e-3))
        return cur + (-cur / max(self.p.tau, 1e-3)) * dt + sigma * k * self.rng.normal(size=np.shape(cur))

    # ------------------------------------------------------------------ 每帧
    def update(self, gt: dict, dt: float) -> None:
        self.t += dt
        self._hist.append({
            "t": gt["t"], "pos": gt["car_pos"].copy(), "vel": gt["car_vel"].copy(),
            "omega": gt["car_omega"], "alive": gt["car_alive"],
            # 车体 yaw 与装甲板半径：瞄准器要知道"板转到哪了"（真 EKF 的 x[6]/x[8]）
            "yaw": float(gt.get("car_yaw", 0.0)), "radius": float(gt.get("car_radius", 0.25)),
        })
        if len(self._hist) > 200:
            del self._hist[:100]
        self.visible = bool(gt["car_alive"]) and (self.rng.random() >= self.p.dropout_prob)
        # OU 噪声演化
        self._e_pos = self._ou(self._e_pos, 0.0, dt)         # 位置噪声在 estimate 里按视线施加
        self._e_vel = self._ou(self._e_vel, self.p.vel_noise, dt)
        self._e_omega = float(self._ou(self._e_omega, self.p.omega_noise, dt))
        self._e_yaw = float(self._ou(self._e_yaw, self.p.yaw_noise, dt))

    def estimate(self) -> dict | None:
        """当前的目标估计（含延迟与噪声）；不可见时返回 None。"""
        if not self.visible or not self._hist:
            return None
        i = max(0, len(self._hist) - 1 - self._delay_n)
        h = self._hist[i]
        pos = h["pos"].copy()
        # 位置噪声施加在**垂直于视线**的平面上（跟踪器的角度误差就是这个量级）
        r = pos - np.array([0.0, 0.0, 0.0])
        n = np.linalg.norm(r)
        if n > 1e-6:
            u = r / n
            tmp = np.array([0.0, 0.0, 1.0]) if abs(u[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
            a = np.cross(u, tmp); a /= (np.linalg.norm(a) + 1e-12)
            b = np.cross(u, a)
            ang = math.radians(self.p.pos_noise_deg)
            pos = pos + (self.rng.normal(0.0, ang) * a + self.rng.normal(0.0, ang) * b) * n
        return {
            "pos": pos,
            "vel": h["vel"][:2] + self._e_vel,
            "omega": float(h["omega"]) + self._e_omega,
            "yaw": float(h["yaw"]) + self._e_yaw,
            "radius": float(h["radius"]),
            "t": h["t"],
        }


@dataclass
class AimerParams:
    pitch_offset_deg: float = 1.8      # 与 configs/sim.yaml 一致（标定值）
    yaw_offset_deg: float = 0.0
    lead_iter: int = 3                 # 提前量迭代次数（sp_vision 用 10 次收敛判据）
    # **竖直偏差标定**（2026-10-13 加）：C++ 闭环逐发实测"离板心竖直偏移"均值 +35 mm
    # （弹速标称 22 / 实际 ~24 m/s 导致的系统性偏高），而镜像原来只有 +8 mm。
    # 这个 bias 让镜像的落点分布与 C++ 对齐 —— 否则策略学到的"这一发值不值得打"
    # 建立在过于乐观的落点分布上（实测会让选择性策略在 C++ 里崩掉）。
    pitch_bias_deg: float = 0.68      # 实测标定值（见 scratch/calib_bias.py）


class MirrorAimer:
    """镜像瞄准：**瞄转动中的装甲板**（真 Aimer 的行为）+ 弹道解算 + 提前量迭代。

    为什么必须瞄板而不是瞄车心（2026-10-13 修，本次关键的一处保真度修复）：
    真 Aimer 的 choose_aim_point() 选的是"当前最正对射手的那块板"，瞄的是**板心**。
    车辆自转 ω=7 rad/s 时，板心在 0.25 m 半径上以 1.75 m/s 扫过，视线角速度 ≈ 25°/s，
    云台必然滞后 —— C++ 闭环实测 |ey| 中位 **3.13°**、p95 **12.1°**。
    而镜像原来瞄"车心"（自转时几乎不动），|ey| 只有 0.70°/p95 2.07°：
    策略于是学到"什么时候都能打"（择时训练失去意义），
    并且把"跟稳了"的阈值定在一个 C++ 里几乎不出现的水平上。
    """

    def __init__(self, p: AimerParams, rng=None, jitter: tuple[float, float] | None = None):
        self.p = p
        self.rng = rng or np.random.default_rng(0)
        self.lock_k = -1          # 上一块锁定的板（模拟 choose_aim_point 的滞回）
        # 指令抖动：两轴 OU 过程（短相关时间），σ/τ 由 MirrorParams 给
        self.jit_sigma = math.radians(jitter[0]) if jitter else 0.0
        self.jit_tau = jitter[1] if jitter else 0.05
        self._j = np.zeros(2)
        self._t_j = None

    def _update_jitter(self, t: float):
        if self.jit_sigma <= 0.0:
            return 0.0, 0.0
        if self._t_j is None:
            self._t_j = t
        dt = max(min(t - self._t_j, 0.1), 1e-4)
        self._t_j = t
        k = math.sqrt(2.0 * dt / self.jit_tau)
        self._j = (self._j * (1.0 - dt / self.jit_tau)
                   + self.jit_sigma * k * self.rng.normal(size=2))
        return float(self._j[0]), float(self._j[1])

    def _pick_plate(self, center, yaw, radius, n_armor, muzzle):
        """选"最正对射手"的板；已有锁定且仍可接受时保持锁定（滞回，避免来回跳）。"""
        best_k, best_cos = -1, -2.0
        for k in range(n_armor):
            ang = yaw + k * 2.0 * math.pi / n_armor
            nvec = np.array([math.cos(ang), math.sin(ang), 0.0])
            p = center + radius * nvec
            los = p - muzzle
            nl = float(np.linalg.norm(los))
            if nl < 1e-6:
                continue
            c = float(np.dot(-nvec, los / nl))     # 板面朝射手的程度 ∈ [-1,1]
            if c > best_cos:
                best_cos, best_k = c, k
        # 滞回：锁定的板只要还在 60° 内就继续用（真 aimer 有 lock_id）
        if self.lock_k >= 0:
            ang = yaw + self.lock_k * 2.0 * math.pi / n_armor
            nvec = np.array([math.cos(ang), math.sin(ang), 0.0])
            los = center + radius * nvec - muzzle
            nl = float(np.linalg.norm(los))
            if nl > 1e-6 and float(np.dot(-nvec, los / nl)) > 0.5:
                return self.lock_k
        self.lock_k = best_k
        return best_k

    def aim(self, est: dict, bullet_speed: float, muzzle=None):
        """返回 (yaw, pitch, valid, debug)。

        yaw/pitch 是**世界系**指令，语义与 io::Command 一致（pitch 抬头为负）。
        muzzle: 炮口世界坐标（不给就用原点近似，仅用于测试）。
        """
        p = self.p
        m = np.zeros(3) if muzzle is None else np.asarray(muzzle, dtype=float)
        v = float(bullet_speed)
        if v < 14.0:
            v = 23.0                                   # 与 Aimer::aim 里的兜底一致

        n_armor = 4
        radius = float(est.get("radius", 0.25))
        yaw_t = float(est.get("yaw", 0.0))
        omega = float(est.get("omega", 0.0))
        vel_xy = np.array([est["vel"][0], est["vel"][1], 0.0])

        fly = 0.0
        aim_pt = est["pos"].copy()
        for _ in range(max(1, p.lead_iter)):
            # 把"车心 + 板绕心旋转"一起预测到弹丸飞到的时刻
            center = est["pos"] + vel_xy * fly
            k = self._pick_plate(center, yaw_t + omega * fly, radius, n_armor, m)
            ang = yaw_t + omega * fly + k * 2.0 * math.pi / n_armor
            aim_pt = center + radius * np.array([math.cos(ang), math.sin(ang), 0.0])
            rel = aim_pt - m
            d = float(np.hypot(rel[0], rel[1]))
            traj = trajectory(v, d, float(rel[2]))
            if traj[0]:
                return 0.0, 0.0, False, {}
            fly = traj[1]
        rel = aim_pt - m
        d = float(np.hypot(rel[0], rel[1]))
        ok, fly, tp = trajectory(v, d, float(rel[2]))
        if ok:
            return 0.0, 0.0, False, {}
        yaw = math.atan2(rel[1], rel[0]) + math.radians(p.yaw_offset_deg)
        pitch = -(tp + math.radians(p.pitch_offset_deg) + math.radians(p.pitch_bias_deg))
        # 指令抖动：云台得靠伺服去追，追不上的那部分就是 C++ 里看到的 |ey|
        jy, jp = self._update_jitter(float(est.get("t", 0.0)))
        yaw += jy
        pitch -= jp
        return yaw, pitch, True, {"fly_time": fly, "aim_point": aim_pt, "plate": self.lock_k}
