"""无渲染仿真世界：给 RL 训练用的最小世界（物理 + 靶车 + 枪 + 扰动），不带 HTTP/渲染。

为什么不让 env 直接连现成的 HTTP 闭环：那条链路带 MuJoCo 渲染（每次 ~30 ms 的 GL
上下文开销），只有 ~23 fps —— 一个 300 s 回合要 5 分钟，PPO 要几千个回合，根本跑不动。
这里直接用 rm_gimbal_demo.Demo 的**纯物理**步进 + rm_cars.CarField，能跑到几百 Hz。

**世界配置必须和联调一致**：所以枪/靶车机动/基准脚本的配置统一走
rm_link.apply_cmd_args（同一份代码），而不是在这里抄一遍 —— 训练与评测的世界
只要有一项对不上，练出来的策略在评测里就是另一回事。
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_WS = os.path.dirname(os.path.dirname(_HERE))
_SIM = os.path.join(_WS, "rm_sim")
if _SIM not in sys.path:
    sys.path.insert(0, _SIM)

import mujoco  # noqa: E402
import rm_gimbal_demo as R  # noqa: E402
import rm_link  # noqa: E402
from rm_cars import CarField  # noqa: E402
from rm_chassis import ChassisDisturb, ChassisParams  # noqa: E402


class HeadlessWorld:
    """一个不带渲染/HTTP 的仿真世界。

    用法：
        w = HeadlessWorld(seed=0, **overrides)
        w.set_command(yaw, pitch, shoot=True)
        w.step()                      # 推进一个控制周期（1/fps 秒）
        obs = w.observe()             # 世界真值（供镜像跟踪器/奖励使用）
    """

    def __init__(self, seed: int = 0, car_motion: str = "script",
                 chassis_disturb: str = "on", fps: float | None = None, **overrides):
        # 用 rm_link 的默认参数当基底，再按需覆盖 —— 保证与联调同一套默认值
        argv = ["--model", "real", "--seed", str(seed),
                "--car-motion", car_motion, "--chassis-disturb", chassis_disturb]
        for k, v in overrides.items():
            flag = "--" + k.replace("_", "-")
            if isinstance(v, bool):
                if v:
                    argv.append(flag)
            else:
                argv += [flag, str(v)]
        self.args = rm_link.build_parser().parse_args(argv)

        elev_deg = rm_link.read_barrel_elev(self.args.assets_dir) \
            if self.args.model == "real" else 0.0
        self.barrel_elev = np.radians(elev_deg)
        self.demo = R.Demo(
            dt=self.args.dt,
            real_assets=(self.args.assets_dir if self.args.model == "real" else None),
            xml_hook=rm_link._build_xml_hook(self.args, elev_deg),
            gun_params=rm_link.gun_params_from_args(self.args),
            # 弹道气动也走同一份配置（configs/ballistics.yaml）——训练世界必须与评测一致，
            # 默认 enabled=false ⇒ 只有重力（历史上一直如此）
            ballistics_params=rm_link.ballistics_params_from_config(
                rm_link.load_config(self.args), self.args),
        )
        rm_link.apply_cmd_args(self.demo, self.args)      # 与联调同一份配置
        self.demo._rng = np.random.default_rng(int(seed))

        self.model, self.data = self.demo.model, self.demo.data
        # 弹道阻力系数 k（1/m）：**镜像解算器必须用与仿真同一个值**（默认 0 = 无阻力）。
        # 来源 = configs/ballistics.yaml（经 demo.ballistics 带过来），不给两处漂移的机会。
        _b = getattr(self.demo, "ballistics", None)
        self.ball_k = float(_b.k) if (_b is not None and _b.enabled) else 0.0
        # 炮口所在 body（= pitch 体）：取世界指向要用它的姿态矩阵
        self.pitch_bid = int(self.model.site_bodyid[self.demo.muzzle_sid])
        self.cars = CarField(
            self.demo, self.model, self.data, mujoco,
            np.random.default_rng(int(seed)), self.demo.cmd,
            spawn_count=self.args.cars, random_spawn=self.args.cars_random)
        self.demo._deaths_fn = lambda: self.cars.deaths
        self.disturb = ChassisDisturb(
            ChassisParams.from_args(self.args),
            rng=np.random.default_rng(int(seed) + 991))

        self.fps = float(fps if fps is not None else self.args.fps)
        self.frame_dt = max(1, int(round(1.0 / (self.fps * self.args.dt)))) * self.args.dt
        self.cmd_yaw = 0.0
        self.cmd_pitch = 0.10
        self.pending_fire = 0
        self.t = 0.0

    # ------------------------------------------------------------------ 重置
    def reset(self, seed: int | None = None) -> None:
        """重置世界状态但**不重建模型**（重建要重新解析/编译 MJCF，很贵，
        而且多进程同时写合并 XML 会互相踩）。换种子只影响随机源。"""
        if seed is not None:
            self.demo._rng = np.random.default_rng(int(seed))
            self.cars.rng = np.random.default_rng(int(seed))
            self.disturb.rng = np.random.default_rng(int(seed) + 991)
        self.demo.reset()
        self.cars.park_all()
        self.cars.respawns = 0
        self.cars.deaths = 0
        self.cars.hits = 0
        self.cars.kill_log = []
        self.cars.respawn_round(first=True)
        self.demo._script = None          # 运动基准脚本重新开始（顺序按新种子重抽）
        self.demo._gyro = None
        self.cmd_yaw, self.cmd_pitch = 0.0, 0.10
        self.pending_fire = 0
        self.t = 0.0

    # ------------------------------------------------------------------ 控制
    def set_command(self, yaw: float, pitch: float, shoot: bool = False) -> None:
        self.cmd_yaw, self.cmd_pitch = float(yaw), float(pitch)
        if shoot:
            self.pending_fire += 1

    # ------------------------------------------------------------------ 步进
    def step(self) -> None:
        """推进一个控制周期：底盘扰动 -> 关节目标 -> 物理 -> 开火。"""
        d, m = self.demo, self.model
        self.disturb.step(self.frame_dt)
        n_fire = self.pending_fire
        self.pending_fire = 0

        joint_yaw = self.cmd_yaw + self.disturb.yaw_jitter + self.disturb.yaw_drift
        joint_pitch = float(np.clip(
            self.barrel_elev + self.cmd_pitch + self.disturb.pitch_jitter
            + self.disturb.pitch_drift,
            self.demo.pitch_lo, self.demo.pitch_hi))
        self.demo.cmd.aim_point = None
        self.demo.cmd.yaw, self.demo.cmd.pitch = joint_yaw, joint_pitch
        for _ in range(n_fire):
            self.demo.fire()
        for _ in range(max(1, int(round(1.0 / (self.fps * m.opt.timestep))))):
            self.cars.pre_step(m.opt.timestep)
            self.demo.step(m, self.data)
            self.cars.post_step(m.opt.timestep)
        self.t += self.frame_dt

    # ------------------------------------------------------------------ 真值
    def observe(self) -> dict:
        """世界真值（不是视觉输出）：镜像跟踪器与奖励函数用这一份。"""
        d = self.cars.cars[0]
        pos = np.array(self.data.xpos[d.bid], dtype=float)
        # 云台实际指向（世界系 yaw/pitch）
        Rm = np.array(self.data.xmat[self.pitch_bid]).reshape(3, 3)
        yaw = float(np.arctan2(Rm[1, 0], Rm[0, 0]))
        pitch = float(np.arcsin(np.clip(-Rm[2, 0], -1.0, 1.0)))
        plates = []
        for car, gid, name in self.cars.armor_geoms():
            plates.append({
                "car": int(car.idx), "gid": int(gid), "name": name,
                "pos": np.array(self.data.geom_xpos[gid], dtype=float).copy(),
                "mat": np.array(self.data.geom_xmat[gid], dtype=float).reshape(3, 3).copy(),
            })
        # 车体朝向（自由关节四元数 → yaw）与装甲板半径：镜像瞄准器要**瞄转动中的板**
        # （真 Aimer 就是瞄板心的：车辆自转 7 rad/s 时板心在 0.25 m 半径上以 1.75 m/s
        #  扫过，云台必然滞后 —— 这正是 C++ 实测 |ey| 中位 3.1°、p95 12° 的来源；
        #  镜像原来瞄"车心不动点"，误差只有 0.7°，于是策略学到的择时阈值完全对不上）
        car_q = np.array(self.data.qpos[self.demo.car_qadr + 3:
                                       self.demo.car_qadr + 7], dtype=float)
        qw, qx, qy, qz = (float(car_q[0]), float(car_q[1]), float(car_q[2]), float(car_q[3]))
        car_yaw = math.atan2(2.0 * (qw * qz + qx * qy),
                             1.0 - 2.0 * (qy * qy + qz * qz))
        radius = 0.25
        if plates:
            radius = float(np.median([float(np.hypot(pl["pos"][0] - pos[0],
                                                     pl["pos"][1] - pos[1]))
                                      for pl in plates]))
        return {
            "t": self.t,
            "car_alive": bool(d.alive), "car_hp": int(d.hp),
            "car_pos": pos, "car_vel": np.array(d.vel, dtype=float).copy(),
            "car_yaw": car_yaw, "car_radius": radius,
            "car_omega": float(getattr(self.demo, "_car_omega", 0.0)),
            "gimbal_yaw": yaw, "gimbal_pitch": pitch,
            "cmd_yaw": self.cmd_yaw, "cmd_pitch": self.cmd_pitch,
            "plates": plates,
            "shots": int(self.demo.shots), "hits": int(self.demo.hits),
            "deaths": int(self.cars.deaths),
            "gun": None if self.demo.gun is None else self.demo.gun.status(),
            "muzzle": self.demo.muzzle()[0].copy(),
        }
