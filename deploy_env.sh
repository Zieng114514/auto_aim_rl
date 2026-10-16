#!/usr/bin/env bash
# =============================================================================
#  一键部署 auto_aim_rl（学习型开火决策）的训练/评测环境
#
#  用法：
#      bash deploy_env.sh                       # 装到 ~/mjlab-env/venv
#      VENV=~/venvs/rl bash deploy_env.sh       # 换 venv
#      bash deploy_env.sh --no-torch            # 先不装 torch（只跑纯算术自检）
#      TORCH_INDEX=<url> bash deploy_env.sh     # 指定 torch 源（x86 上要 CUDA 版时）
#      TORCH_FIND_LINKS=<dir-url> TORCH_PIN='torch==2.9.1+cpu' bash deploy_env.sh   # 国内镜像走这条
#
#  ---- 前置（很硬）----
#  1) **`rm_sim` 必须与 `auto_aim_rl` 并列在同一个父目录**（env 直接 import 它）：
#         <父目录>/rm_sim/       ← 本工作区的仿真包
#         <父目录>/auto_aim_rl/  ← 本仓库（脚本就在这一层）
#  2) **真模型资产要在位**（`rm_sim/assets/obj/`）：env 走 `--model real`，
#     缺了它连第一个 env 都建不起来。缺的话先跑： bash ../deploy/deploy_rm_sim.sh
#
#  ---- 平台差异（脚本自动处理）----
#      x86_64 : torch 用 **CPU 轮子索引**（PyPI 默认 torch 会拖 2.5 GB 的 CUDA 依赖）
#      aarch64: torch 用 jetson-ai-lab 的 CUDA 轮子（jp6/cu126），并把 numpy 钉 <2
# =============================================================================
set -uo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # .../auto_aim_rl
PARENT="$(dirname "$WS")"
SIM="${SIM:-$PARENT/rm_sim}"
VENV="${VENV:-$HOME/mjlab-env/venv}"
PIP_MIRROR="${PIP_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"
WITH_TORCH=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-torch) WITH_TORCH=0; shift ;;
    -h|--help)  sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "未知参数: $1（--help 看用法）" >&2; exit 2 ;;
  esac
done

ok()  { echo "    ✓ $*"; }
bad() { echo "    ✗ $*" >&2; }
say() { echo "==> $*"; }

say "[1/5] 前置检查"
[[ -f "$SIM/rm_link.py" ]] || {
  bad "找不到 $SIM/rm_link.py"
  bad "env 直接 import rm_sim，两者必须并列在同一父目录；可用 SIM=<路径> 覆盖"
  exit 1
}
ok "仿真包：$SIM"
[[ -f "$SIM/assets/obj/chassis.obj" ]] || {
  bad "$SIM 里没有真模型资产（assets/obj/）—— env 走 --model real，必须要有"
  bad "先跑： bash $SIM/../deploy/deploy_rm_sim.sh"
  exit 1
}
ok "真模型资产在位"
command -v python3 >/dev/null || { bad "没有 python3"; exit 1; }
PY3="$(command -v python3)"
ARCH="$(uname -m)"
ok "python3 $("$PY3" -c 'import sys;print("%d.%d"%sys.version_info[:2])') ($ARCH)，venv 目标 $VENV"

say "[2/5] 建 venv"
if [[ -x "$VENV/bin/python" ]]; then
  ok "venv 已存在：$VENV"
else
  "$PY3" -m venv "$VENV" || { bad "建 venv 失败（缺 python3-venv 包？）"; exit 1; }
  ok "已创建 $VENV"
fi
V="$VENV/bin"
"$V/pip" install -q -U pip setuptools wheel || { bad "升级 pip 失败（网络？）"; exit 1; }

say "[3/5] 安装依赖"
NUMPY_PIN="numpy"
[[ "$ARCH" == "aarch64" ]] && NUMPY_PIN="numpy<2"
# mujoco 是 env 的硬依赖（world.py 直接 import）；两个脚本默认同一个 venv 时这里是幂等的，
# 单独给本仓库建 venv 时也能自洽。**必须钉 3.11.0**，理由同 deploy_rm_sim.sh。
MUJOCO_PIN="${MUJOCO_PIN:-mujoco==3.11.0}"
# 其余清单与 tools/dual_machine/*_setup.sh 同一套：训练/评测/导出 ONNX 都用得到
# onnxscript：torch>=2.6 的**默认** ONNX 导出器要它（我们优先走旧导出器，装上更保险）
if ! "$V/pip" install -q -i "$PIP_MIRROR" "$MUJOCO_PIN" \
      "$NUMPY_PIN" PyYAML gymnasium matplotlib onnx onnxscript onnxruntime scikit-learn; then
  bad "镜像源装失败，回落 PyPI 再试一次"
  "$V/pip" install -q "$MUJOCO_PIN" \
      "$NUMPY_PIN" PyYAML gymnasium matplotlib onnx onnxscript onnxruntime scikit-learn \
      || { bad "依赖装不上，检查网络/代理"; exit 1; }
fi
ok "基础依赖装好（含 $MUJOCO_PIN）"

if [[ "$WITH_TORCH" == 0 ]]; then
  ok "跳过 torch（--no-torch）"
else
  if [[ "$ARCH" == "aarch64" ]]; then
    # Jetson：CUDA 轮子来自 jetson-ai-lab（jp6/cu126 = JetPack 6.2 / CUDA 12.6）
    "$V/pip" install -q --index-url "${TORCH_INDEX:-https://pypi.jetson-ai-lab.io/jp6/cu126}" \
        --extra-index-url https://pypi.org/simple "torch==2.8.0" \
      || { bad "torch（Jetson CUDA 轮子）装失败；确认 JetPack 版本与索引匹配"; exit 1; }
  else
    # x86：默认 CPU 轮子（要 CUDA 就 TORCH_INDEX=https://download.pytorch.org/whl/cu121）
    # ⚠ 国内网络：download.pytorch.org 常不通/极慢（2026-10-16 云端 T4 实测卡死 ✗，
    #   而且改从 PyPI 装会连带 ~5GB 的 cu1xx nvidia 依赖 ⇒ 8GB 小根盘直接爆 ✗）
    #   ⇒ 正解：用镜像的**轮子目录**（find-links）+ 版本钉死，例：
    #      TORCH_FIND_LINKS=https://mirrors.aliyun.com/pytorch-wheels/cpu \
    #      TORCH_PIN='torch==2.9.1+cpu' bash deploy_env.sh
    if [[ -n "${TORCH_FIND_LINKS:-}" ]]; then
      "$V/pip" install -q \
          -i "${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}" \
          -f "$TORCH_FIND_LINKS" "${TORCH_PIN:-torch}" \
        || { bad "torch 装失败（find-links=$TORCH_FIND_LINKS pin=${TORCH_PIN:-torch}）"; exit 1; }
    else
      "$V/pip" install -q --index-url "${TORCH_INDEX:-https://download.pytorch.org/whl/cpu}" torch \
        || { bad "torch 装失败"; exit 1; }
    fi
  fi
  ok "torch 装好"
fi

say "[4/5] 自检（env 三个 + 黄金轨迹）"
FAILED=""
for t in env/test_env.py env/test_mirror_ballistics.py env/test_threshold.py; do
  printf "    %-32s " "$t"
  if (cd "$WS" && timeout 900 "$V/python" "$t" > /tmp/deploy_rl_test.log 2>&1); then
    echo "PASS"
  else
    echo "FAIL"; tail -6 /tmp/deploy_rl_test.log | sed 's/^/        /'; FAILED="$FAILED $t"
  fi
done
[[ -z "$FAILED" ]] || { bad "有失败：$FAILED"; exit 1; }
ok "env 自检通过"

if [[ "$WITH_TORCH" == 1 ]]; then
  "$V/python" - <<'PY'
import torch
print("    torch      ", torch.__version__, "cuda =", torch.cuda.is_available())
PY
fi

# 黄金轨迹：整条物理链路的指纹（600 步 + 枪/命中/重生）。它同时验证了
# rm_sim 的模型/参数与 env 的连接是否与文档基线一致。
printf "    %-32s " "scratch/golden_trace.py"
HASH="$( (cd "$WS" && timeout 900 "$V/python" scratch/golden_trace.py deploy 2>/dev/null) \
         | grep -oE "qpos_hash=[0-9a-f]+" | cut -d= -f2)"
EXPECT="c76a72ab6149211b800259e871dad902"
if [[ "$HASH" == "$EXPECT" ]]; then
  echo "PASS（$HASH）"
elif [[ -n "$HASH" ]]; then
  echo "⚠ 与文档基线不一致：$HASH"
  echo "       文档基线是 $EXPECT。可能原因：mujoco 没钉 3.11.0、或真模型资产被重建过。"
  echo "       影响：物理与历史数字**不可直接对比**（要么装回 3.11.0，要么重新标定基线）。"
  echo "       这不算部署失败 —— 其余自检都过了；只是别拿旧基线当参照。"
else
  echo "FAIL（没打出 hash）"
  bad "黄金轨迹跑不起来，见上面输出"; exit 1
fi

say "[5/5] 完成"
cat <<TIP

==> 装好了。接下来：

  1) 训练（PPO，先小步数冒烟）：
       $V/python train/ppo.py --steps 20000 --envs 4 --out runs/smoke
     吞吐参考：x86 16 线程 ≈ 509 步/s @16 envs；Jetson ≈ 154 步/s @8 envs（饱和）
     ⚠ 训练吃满 CPU：别和"实时闭环评测"同时跑（要同时就把 --envs 压到 4~6）

  2) 评测 / 对照（不训练，只跑策略）：
       $V/python train/eval_env.py --help

  3) 导出 ONNX 给 C++ 闭环用：
       $V/python train/export_onnx.py --help

  4) 想跑"仿真 + 视觉"整条闭环：见工作区 README 的「与 sp_vision 闭环」
     （需要另外的 C++ 框架，不在本仓库里）
TIP

