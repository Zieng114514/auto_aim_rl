"""阶段 1 的 gym 环境：只学「什么时候开火」。

分工（与 agnet.md §3.0-E 一致）：
    世界/物理/靶车     -> rm_sim（HeadlessWorld，与联调同一套配置）
    跟踪 + 瞄准 + 弹道 -> mirror.py（与 sp_vision 同口径）
    策略               -> 只输出 {不开火, 开火}

为什么只让它学开火：瞄准/弹道是确定性几何（已标定验证过），让 RL 去学只会更差更慢；
真正的不确定性在"**现在这一发值不值得打**"——目标在动、弹速在漂、热量有预算，
固定阈值门限（sp_vision 现在用的）没法表达这种状态依赖。

奖励：命中 +1；每发开火 −c_fire（模拟热量/弹药预算）。
**验收不看累计奖励**，只看 eval_policy.sh 里的命中率与 TTK（见 DESIGN.md §1）。
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:                                   # 允许无 gym 依赖下当普通类用
    gym, spaces = None, None

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from mirror import AimerParams, MirrorAimer, MirrorParams, MirrorTracker  # noqa: E402
from world import HeadlessWorld  # noqa: E402

# 观测规格**冻结**在这里，C++ 侧 ShooterRL 必须逐项对齐（见 env/OBS_SPEC.md）。
# 原则：**只放真机上拿得到的量**。曾经放过"回合进度"，但真机没有回合概念，
# 会造成训练/部署分布不一致 —— 已经删掉（宁可少一维，不要一维对不上）。
OBS_DIM = 21


class FireEnv(gym.Env if gym else object):
    """靶车在动、弹速在漂、热量有预算 -> 决定这一发出不出膛。"""

    metadata = {"render_modes": []}

    def __init__(self, seed: int = 0, episode_s: float = 90.0, c_fire: float = 0.10,
                 c_block: float = 0.0, c_waste: float = 0.0,
                 disturb: bool = True, car_motion: str = "script",
                 mirror: MirrorParams | None = None, aimer: AimerParams | None = None,
                 randomize: bool = False):
        super().__init__()
        self.episode_s = float(episode_s)
        self.c_fire = float(c_fire)
        # 被锁枪挡下的开火请求的代价。**这是"选择性开火"的关键 shaping**：
        # 热量上限 100、每发 10、回热 10/s ⇒ 可持续射频只有 ~1 发/s，而决策有 30 次/s。
        # 若请求被挡是免费的（真机确实免费），最优解就是"一直请求"（实测开火请求占 74%），
        # 策略永远学不会**挑时机**：热量一好就自动打出去，跟"这发值不值得打"无关。
        # 加一点代价后，策略必须自己决定"这一格热量现在花掉，还是再等 0.2s 打更有把握的一发"
        # —— 这正是相比规则门限唯一可能赢的地方。
        self.c_block = float(c_block)
        # **热量浪费感知**（本次的主 shaping）：热量顶到上限时，10/s 的回热全被丢掉。
        # 只惩罚"被挡的请求"太粗 —— 实测会把策略训成"干脆不开火"（回报 +23 → -18）。
        # 惩罚"热量堵在上限"则语义清晰：策略**必须**把回热用掉（≈1 发/s），
        # 但用掉的方式由它自己挑 —— 每秒 30 个决策格里选最值得打的那一格。
        # 这样"择时"才第一次成为真正的取舍，而不是"有热量就打"。
        self.c_waste = float(c_waste)
        self.waste_thresh = 0.995     # heat_frac 超过这个值算"堵住回热"
        self.disturb = bool(disturb)
        self.car_motion = car_motion
        self.mirror_params = mirror or MirrorParams()
        self.aimer_params = aimer or AimerParams()
        self.randomize = bool(randomize)      # 域随机化（每回合重抽镜像/扰动参数）
        self._seed = int(seed)
        self.world = None
        self._rng = np.random.default_rng(seed)
        if spaces is not None:
            self.action_space = spaces.Discrete(2)
            self.observation_space = spaces.Box(-5.0, 5.0, shape=(OBS_DIM,), dtype=np.float32)

    # ------------------------------------------------------------------ 构建
    def _build(self, seed: int) -> None:
        """第一次构造世界；之后每回合只 reset 状态（重建模型很贵且会并发写盘）。"""
        rng = np.random.default_rng(seed)
        if self.world is not None:
            self.world.reset(seed=seed)
            self._reset_episode_state(seed)
            return
        mp = MirrorParams(**vars(self.mirror_params))
        if self.randomize:
            # 域随机化：不同车/不同跟踪质量 —— 策略不能绑死在某一套估计误差上
            mp.latency_s = float(rng.uniform(0.01, 0.06))
            mp.pos_noise_deg = float(rng.uniform(0.15, 0.9))
            mp.vel_noise = float(rng.uniform(0.1, 0.5))
            mp.omega_noise = float(rng.uniform(0.3, 1.2))
            mp.dropout_prob = float(rng.uniform(0.0, 0.05))
        ap = AimerParams(**vars(self.aimer_params))
        if self.randomize:
            # 竖直偏差也一起随机化：这个 bias 是"弹速标称/实际不一致"造成的，
            # 真机上每把枪、每个热量状态都不一样 —— 策略不能假设它固定是 0.68°。
            ap.pitch_bias_deg = float(rng.uniform(0.45, 0.95))
            ap.yaw_offset_deg = float(rng.uniform(-0.15, 0.15))
        self.mirror_params = mp
        self.aimer_params = ap
        self.world = HeadlessWorld(seed=seed, car_motion=self.car_motion,
                                   chassis_disturb="on" if self.disturb else "off")
        self._reset_episode_state(seed)

    def _reset_episode_state(self, seed: int) -> None:
        self.tracker = MirrorTracker(self.mirror_params, self.world.fps,
                                     np.random.default_rng(seed + 7))
        self.aimer = MirrorAimer(
            self.aimer_params, np.random.default_rng(seed + 13),
            jitter=(self.mirror_params.cmd_jitter_deg, self.mirror_params.cmd_jitter_tau))
        self._last_cmd = np.zeros(2)
        self._last_aim_err = np.zeros(2)
        self._t_last_shot = -1e9
        self._t_last_hit = -1e9
        self._hist: list[int] = []            # 最近 10 发的命中标记
        self._prev_shots = 0
        self._prev_hits = 0
        self._prev_blocked = 0
        self._target_seen_t = 0.0
        self._yaw, self._pitch, self._valid = 0.0, 0.1, False

    # ------------------------------------------------------------------ 观测
    def _obs(self, gt: dict, est: dict | None, gun: dict) -> np.ndarray:
        o = np.zeros(OBS_DIM, dtype=np.float32)
        gy, gp = gt["gimbal_yaw"], gt["gimbal_pitch"]
        cy, cs = math.cos(gy), math.sin(gy)
        mp = gt["muzzle"]
        if est is not None:
            rel = est["pos"] - mp
            # 转到云台 yaw 系：前向/侧向/高度（yaw 无关，策略不必学"目标在哪个方位"）
            fwd = rel[0] * cy + rel[1] * cs
            lat = -rel[0] * cs + rel[1] * cy
            o[0], o[1], o[2] = fwd / 5.0, lat / 5.0, rel[2]
            o[3] = math.hypot(rel[0], rel[1]) / 5.0
            vf = est["vel"][0] * cy + est["vel"][1] * cs
            vl = -est["vel"][0] * cs + est["vel"][1] * cy
            o[4], o[5] = vf / 3.0, vl / 3.0
            o[6] = float(np.hypot(est["vel"][0], est["vel"][1])) / 3.0
            o[7] = est["omega"] / 8.0
            o[8] = 1.0                              # 目标可见
            o[9] = min(self.world.t - self._target_seen_t, 2.0) / 2.0
        # 瞄准误差（指令 - 实际），弧度。
        # yaw **必须绕回到 ±π**：两边各自都在 [-π,π] 内，直接相减会在跨 ±π 边界时
        # 蹦出 ±2π 尖峰（C++ 侧实测出现过 ey=-6.10）。这里统一绕回，两边口径一致。
        ey, ep = (self._yaw - gy + math.pi) % (2 * math.pi) - math.pi, self._pitch - gp
        o[10], o[11] = ey, ep
        o[12], o[13] = (ey - self._last_aim_err[0]) * 10.0, (ep - self._last_aim_err[1]) * 10.0
        o[14] = abs(self._yaw - self._last_cmd[0]) * 10.0     # 指令变化率（抖不抖）
        o[15] = min(self.world.t - self._t_last_shot, 5.0) / 5.0
        o[16] = min(self.world.t - self._t_last_hit, 10.0) / 10.0
        o[17] = (sum(self._hist[-10:]) / 10.0) if self._hist else 0.0
        o[18] = float(gun.get("heat_frac", 0.0))
        o[19] = 1.0 if gun.get("locked", False) else 0.0
        o[20] = 1.0 if self._valid else 0.0
        return o

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._seed = int(seed)
        self._build(self._seed)
        gt = self.world.observe()
        self.tracker.update(gt, self.world.frame_dt)
        est = self.tracker.estimate()
        gun = gt["gun"] or {}
        # 首帧也必须是"真值"，不能让变化率项凭空跳一下
        self._last_cmd = np.array([self._yaw, self._pitch])
        self._last_aim_err = np.array([self._yaw - gt["gimbal_yaw"],
                                       self._pitch - gt["gimbal_pitch"]])
        return self._obs(gt, est, gun), {}

    def step(self, action):
        w = self.world
        gt = w.observe()
        est = self.tracker.estimate()
        gun = gt["gun"] or {}
        # 镜像瞄准：用估计状态算出这一周期的指令
        self._yaw, self._pitch, self._valid, _dbg = self.aimer.aim(
            est, w.demo.cmd.ball_speed, muzzle=gt["muzzle"]) if est is not None \
            else (self._yaw, self._pitch, False, {})
        shoot = bool(int(action) == 1) and self._valid
        w.set_command(self._yaw, self._pitch, shoot=shoot)
        w.step()

        gt2 = w.observe()
        self.tracker.update(gt2, w.frame_dt)
        est2 = self.tracker.estimate()
        gun2 = gt2["gun"] or {}

        d_hits = gt2["hits"] - self._prev_hits
        d_shots = gt2["shots"] - self._prev_shots
        d_block = int(gun2.get("shots_blocked", 0)) - self._prev_blocked
        self._prev_hits, self._prev_shots = gt2["hits"], gt2["shots"]
        self._prev_blocked = int(gun2.get("shots_blocked", 0))
        if d_shots > 0:
            self._t_last_shot = gt2["t"]
            self._hist.append(1 if d_hits > 0 else 0)
            if len(self._hist) > 50:
                del self._hist[:-50]
        if d_hits > 0:
            self._t_last_hit = gt2["t"]
        if est2 is not None:
            self._target_seen_t = gt2["t"]

        d_waste = 1 if float(gun2.get("heat_frac", 0.0)) >= self.waste_thresh else 0
        reward = (float(d_hits) - self.c_fire * float(d_shots)
                  - self.c_block * float(d_block) - self.c_waste * float(d_waste))
        # 先算 obs 再更新"上一周期"的量：否则 obs[12..14]（误差/指令变化率）拿到的
        # 是**本周期**的值，恒等于 0 —— 这一条以前写反了，三个维度全是死特征。
        obs = self._obs(gt2, est2, gun2)
        self._last_cmd = np.array([self._yaw, self._pitch])
        self._last_aim_err = np.array([self._yaw - gt2["gimbal_yaw"],
                                       self._pitch - gt2["gimbal_pitch"]])
        truncated = bool(gt2["t"] >= self.episode_s)
        info = {"hits": gt2["hits"], "shots": gt2["shots"], "deaths": gt2["deaths"],
                "t": gt2["t"], "car_hp": gt2["car_hp"]}
        return obs, reward, False, truncated, info

    # ------------------------------------------------------------------ 评测
    def episode_stats(self) -> dict:
        gt = self.world.observe()
        return {"hits": gt["hits"], "shots": gt["shots"], "deaths": gt["deaths"],
                "t": gt["t"]}
