# auto_aim_rl

**学习型开火决策**：把"什么时候开火"交给学出来的策略，瞄准解算仍是原来那套。
两个层次都用到了：

* **门限/质量模型**（已上闭环）：学 `P(命中 | 观测)`，用 `p >= λ` 当开火门限 ——
  比规则门限（卡 yaw 误差）在命中率、TTK、每击杀弹量上全面更好；
* **RL 射手（PPO）**：直接学"开不开火"这一位，观测 21 维（瞄准误差、目标机动、热量…）。

项目章程见 `../agnet.md`，开发日志见 `../agnet_logs/`，
指标口径与踩坑见 `../agnet_experience.md`。

## 目录

| 路径 | 内容 |
|---|---|
| `env/` | gym 封装：`world.py`（无渲染物理世界）、`sim_env.py`（FireEnv）、`mirror.py`（纯 Python 弹道/跟踪镜像）、`policies.py` |
| `train/` | `ppo.py`（PPO）、`bc_warmstart.py`、`fit_quality_mlp.py` / `fit_trigger.py`（离线拟合）、`export_onnx.py`、`eval_env.py` |
| `env/test_*.py` | 自检：env 冒烟、镜像弹道对表、门限逻辑 |
| `scratch/` | 临时脚本（含 `golden_trace.py` —— 物理指纹，改仿真前后必须比） |
| `DESIGN.md`、`env/OBS_SPEC.md` | 设计说明与观测规格（维度与含义写死在那） |

## 部署

**前置**：`rm_sim` 必须与本仓库**并列在同一个父目录**（`env` 直接 import 它）：

```
ws/
├── vision/           # = rm-vision 仓库（含 rm_sim）
└── auto_aim_rl/      # ← 本仓库
```

```bash
cd ~/ws
bash vision/deploy/deploy_rm_sim.sh      # 先装仿真环境（并确保真模型资产在位）
bash auto_aim_rl/deploy_env.sh           # 再装本仓库的训练/评测环境（含自检）
```

`deploy_env.sh` 自动：建 venv（默认 `~/mjlab-env/venv`，`VENV=` 可换）→ 装依赖
（x86 走 **CPU torch** 索引；Jetson 走 jetson-ai-lab 的 CUDA 轮子并把 `numpy` 钉 `<2`）
→ 跑 `env/test_*.py` 与**黄金轨迹**（必须打印
`qpos_hash=c76a72ab6149211b800259e871dad902`）。`--no-torch` 可先只装纯算术依赖。

## 训练 / 评测

```bash
V=~/mjlab-env/venv/bin

# 冒烟（2 万步，几分钟）
$V/python train/ppo.py --steps 20000 --envs 4 --rollout 512 --out runs/smoke

# 正式训练（吞吐参考：x86 16 线程 ≈509 步/s @16 envs；Jetson ≈154 步/s @8 envs）
$V/python train/ppo.py --steps 200000 --envs 12 --randomize --seed 0 --out runs/stage1_a

# 评测 / 导出 ONNX
$V/python train/eval_env.py --help
$V/python train/export_onnx.py --help
```

⚠ **训练吃满 CPU**：别和"实时闭环评测"同时跑（要同时就把 `--envs` 压到 4~6）。

## 为什么要"镜像环境"

C++ 闭环单帧要 ~30 ms 的 MuJoCo 渲染，一个 300 s 回合要 5 分钟，PPO 根本跑不动。
所以这里用 `world.py` 的**纯物理**世界（不带 HTTP/渲染，几百 Hz），
并且**世界配置与联调共用同一份代码**（`rm_link.apply_cmd_args`）
—— 训练与评测只要有一项对不上，练出来的策略在评测里就是另一回事。
已知的"镜像 vs C++"系统性偏差与对齐方法见 `../agnet_experience.md` §1.16。


