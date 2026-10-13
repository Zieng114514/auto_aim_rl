# auto_aim_rl 实施设计（阶段 E：RL 阶段 1 —— 发射策略）

> 章程见 ../agnet.md §3.0-E；评测口径见 ../agnet_experience.md §4。
> 本文件只讲"怎么落地"，不做实验记录的重复。

## 1. 目标与验收（唯一标准）

在 **同一套评测**（\`sp_vision_25/sim/eval_policy.sh\`，4 种子 × 300 s、运动基准脚本、
底盘扰动 on）上**超过 sp_vision 基准**：

| 指标 | 看什么 |
|---|---|
| 命中率 | 总体 + **分靶型**（尤其"固定周期旋转移动靶"这一栏） |
| 击杀时间 | TTK(首中→death)、击杀频率 |
| 弹量 | 每击杀弹量（不能靠乱射刷命中率） |

基准数字存在 \`sp_vision_25/sim/baselines/sp_vision.json\`，RL 的结果存成
\`baselines/rl_stage1.json\`，两个文件直接对比。

## 2. 环境（env/）：为什么会是"Python 侧镜像"

训练不能直接用现在那条 HTTP 闭环：它带渲染、23 fps，一个 300 s 回合要 5 分钟，
RL 需要几千个回合 —— 必须走**无渲染物理**。

但也不能只喂真值：策略部署后读的是 sp_vision 的 tracker/aimer 输出，
喂真值会造成训练/部署分布不一致。

所以 env 的结构是：

    rm_gimbal_demo.Demo（纯物理，无渲染）        ← 世界真值
      + CarField（靶车机动/血量/命中判定）        ← 与联调完全同一套
      + 轻量 Python 跟踪器/瞄准器（**sp_vision 的镜像**）← 产出与 C++ 侧同口径的特征
      + 感知噪声（检测抖动/丢帧/延迟）            ← 检测器已单独验证 99.4%，不必陪渲染

* 控制频率与 sp_vision 一致（30 Hz）；
* 命令延迟按实测（图像+链路）注入，作为 domain randomization 的一项；
* 镜像跟踪器的参数可随机化（域随机化覆盖 tracker 估计质量）——
  这样策略不会绑死在 Python 跟踪器的具体实现上。

## 2b. 已实测的关键事实（2026-10-13，都影响设计）

| 事实 | 数值 | 结论 |
|---|---|---|
| 无渲染步进速度 | dt=0.001 → **51 Hz**；dt=0.002 → 100 Hz；dt=0.004 → 210 Hz | 物理是瓶颈，线性于子步数 |
| 缩小步长的代价 | 同样打法命中率 100% → 89.7%（dt=0.002）→ 79.3%（dt=0.003） | **必须保持 dt=0.001**，否则命中判定本身失真（接触检测变粗） |
| 多线程并行 | 1 线程 93 Hz；4 线程合计仅 120 Hz | mj_step 不释放 GIL → **必须多进程** |
| 多进程并行 | 6 个 worker ≈ 121 步/s 合计 | 可用；再往上受核心数限制 |
| 并发写合并 XML | 4 个 worker 必崩（"substring not found"） | 已给 merge_real_assets 加**文件锁 + 原子替换** |
| 每回合重建世界 | 很贵且加剧上面的并发问题 | 已改成**模型只建一次**，每回合只 reset 状态 |

### 2c. 环境可信度验证（**最重要的一条**）

把 sp_vision 的固定阈值门限镜像进 env（`env/policies.py`），4 个种子各 100 s：

| | 发射 | 命中 | 命中率 |
|---|---|---|---|
| **C++ 闭环 sp_vision 基准**（4 种子 × 300 s） | 885 | 304 | **34.4%** |
| **Python env + 镜像门限**（4 种子 × 100 s） | 433 | 144 | **33.3%** |

**差 1 个百分点** —— 说明镜像（跟踪噪声 + 弹道解算 + 门限）是个可信的代理，
训练出来的策略值得拿去 C++ 闭环里复测。

## 3. 观测 / 动作 / 奖励（阶段 1）

* **观测（标量，先不做历史）**：
  目标（距离、方位角、俯仰、水平速度、角速度、facing、板号）、
  瞄准（yaw/pitch 指令 − 云台实际角、指令变化率）、
  枪（热量占比、是否锁枪、弹速偏置估计）、
  时间（距上次开火、目标连续可见时长）、
  近期战绩（最近 N 发的命中/脱靶计数）。
* **动作**：离散 {不开火, 开火}（真机就是离散触发 + 射速上限）。
* **奖励**：命中 +1；开火代价 −c_fire（模拟热量/弹药，量级决定策略落点）；
  可选 shaping：−(离板心最近距离)/板半高。
  **验收只看 TTK 与命中率，不看累计奖励**。
* **回合**：固定 300 s 或固定击杀数，与评测口径一致。

## 4. 算法与训练

* **rsl_rl**（PPO，ETH 那套，见 agnet.md §3）为主；先用单文件 CleanRL 版跑通再换。
* **行为克隆 warm-start**（风险预案，见 agnet.md §3.0.2）：
  用 sp_vision 的固定阈值策略跑若干回合产生 (观测, 动作) 对，先监督学一遍，
  再 PPO 微调 —— 避免从随机权重瞎转。
* **Ablation（负责人定的）**：标量观测 vs 带目标轨迹历史（LSTM），哪个好上哪个。

## 5. 部署（deploy/）

* 策略导出 ONNX（\`policy.onnx\`）；
* C++ 侧加 \`ShooterRL\` 类，**照抄 sim_ort 的 ORT 加载/推理模式**
  （含"实际 provider"日志，别被静默回落 CPU 骗）；
* 只替换 \`Shooter::shoot\` 的门限逻辑，**最小侵入**；
* 通过 \`configs/sim.yaml\` 的一个开关切换（\`shooter: threshold | rl\`）。

## 6. 评测流程（与 sp_vision 完全相同）

    bash sp_vision_25/sim/eval_policy.sh rl_stage1 300 20251012 777 42 12345

然后对比：

    sp_vision_25/sim/baselines/sp_vision.json
    sp_vision_25/sim/baselines/rl_stage1.json

## 7. 目录结构

    auto_aim_rl/
      env/      gymnasium 封装（无渲染物理 + 镜像跟踪器 + 噪声）
      train/    rsl_rl / PPO 训练脚本与配置
      eval/     调用 sp_vision_25/sim/eval_policy.sh 并对比 baselines
      deploy/   ShooterRL（C++ 补丁）+ ONNX 导出
      scratch/  临时脚本（规矩见 agnet.md §5.2）
