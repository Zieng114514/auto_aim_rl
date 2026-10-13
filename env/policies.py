"""对照策略：把 sp_vision 现在的固定阈值门限**镜像**进来。

作用有两个：
1. **验证环境**：如果这套门限在本环境里的命中率明显偏离 C++ 闭环的基准（34.4%），
   说明镜像（跟踪/瞄准/延迟）跟真实链路差得太多，训练出来的策略也就不可信；
2. **行为克隆的示范来源**：RL 冷启动先用它跑出的轨迹做监督预热（见 agnet.md §3.0.2 风险预案）。

门限逻辑照搬 \`tasks/auto_aim/shooter.cpp\`：
    tolerance = 距离 > judge_distance ? second : first
    |last_cmd.yaw - cmd.yaw| < tolerance*2      （命令突变不射）
    且 |gimbal_yaw - last_cmd.yaw| < tolerance  （云台得跟上）
    且 瞄准点有效
"""
from __future__ import annotations

import math


class ThresholdPolicy:
    """固定阈值门限（sp_vision 现役逻辑的镜像）。"""

    def __init__(self, first_deg: float = 0.5, second_deg: float | None = None,
                 judge_distance: float = 2.0):
        self.tol = math.radians(first_deg)
        self.tol2 = math.radians(second_deg if second_deg is not None else first_deg)
        self.judge = judge_distance

    def __call__(self, obs) -> int:
        # obs 布局见 sim_env.OBS_DIM 注释（**0 基下标**）：
        #   3 = 水平距离/5，10/11 = 瞄准误差(rad)，14 = 指令 yaw 变化率×10，20 = 瞄准点有效
        # 注意：删掉"回合进度"那一维之后瞄准点有效从 obs[21] 变成 obs[20]，
        # 这里原来写的是 obs[21] —— 在 21 维观测下会直接 IndexError（已修）。
        ey, ep = float(obs[10]), float(obs[11])
        cmd_rate = float(obs[14]) / 10.0
        dist = float(obs[3]) * 5.0
        valid = float(obs[20]) > 0.5
        tol = self.tol2 if dist > self.judge else self.tol
        if not valid:
            return 0
        if abs(cmd_rate) >= tol * 2.0:
            return 0
        if abs(ey) >= tol:
            return 0
        return 1
