#!/usr/bin/env python3
"""把 PPO 策略导出成 ONNX，给 C++ 的 ShooterRL 用。

    python auto_aim_rl/train/export_onnx.py auto_aim_rl/runs/stage1_a/policy.pt \
                                          auto_aim_rl/runs/stage1_a/policy.onnx

导出两个输出，C++ 侧两个都认（优先 action）：
    obs    float32[1,21]   与 auto_aim_rl/env/OBS_SPEC.md 逐项一致
    action int64  [1]      argmax logits —— C++ 直接用这一个，省得再取 argmax
    logits float32[1,2]    原始 logits（调试/分析用）

**必须核对 obs 维度**：导出前拿真实 env 跑一步，把 obs 和 C++ 的观测规格对齐，
维度或归一化写错的话，策略在仿真里会表现成"完全不响应"，很难查。
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "env"))
from sim_env import OBS_DIM  # noqa: E402


class Policy(nn.Module):
    """和 train/ppo.py 里的网络**结构必须一致**（否则 state_dict 加载不上）。"""

    def __init__(self, obs_dim: int = OBS_DIM, hidden: int = 64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh())
        self.actor = nn.Linear(hidden, 2)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.body(x)
        return self.actor(h), self.critic(h)


class ExportWrapper(nn.Module):
    """只导出 actor：C++ 不需要 value。"""

    def __init__(self, net: Policy):
        super().__init__()
        self.net = net

    def forward(self, obs: torch.Tensor):
        logits, _ = self.net(obs)
        return torch.argmax(logits, dim=-1).to(torch.int64), logits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", help="policy.pt（ppo.py 的产物）")
    ap.add_argument("out", nargs="?", default=None, help="输出 .onnx，默认与 ckpt 同目录 policy.onnx")
    args = ap.parse_args()
    out = args.out or os.path.join(os.path.dirname(args.ckpt), "policy.onnx")

    ck = torch.load(args.ckpt, map_location="cpu")
    net = Policy(obs_dim=OBS_DIM)
    net.load_state_dict(ck["model"])
    net.eval()

    total = sum(p.numel() for p in net.parameters())
    print("载入 %s | 参数 %d" % (args.ckpt, total))

    # 导出前的形状检查：喂一个全零观测和几个随机观测
    wrapper = ExportWrapper(net).eval()
    dummy = torch.zeros(1, OBS_DIM, dtype=torch.float32)
    a, lg = wrapper(dummy)
    assert a.shape == (1,) and lg.shape == (1, 2), (a.shape, lg.shape)

    torch.onnx.export(
        wrapper, dummy, out,
        input_names=["obs"], output_names=["action", "logits"],
        dynamic_axes=None,          # batch 固定 1：C++ 每帧一次
        opset_version=17,
        do_constant_folding=True)

    # 导出后**自检**：ONNX Runtime 跑一遍，和 torch 的输出逐位对比。
    # （torch.onnx.export 经常"成功"但图是错的，尤其是 argmax/int64 这类算子。）
    try:
        import onnxruntime as ort
        s = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
        rng = torch.Generator().manual_seed(0)
        bad = 0
        for _ in range(8):
            o = torch.randn(1, OBS_DIM, generator=rng)
            ta = int(torch.argmax(net(o)[0], dim=-1).item())
            ra, rl = s.run(None, {"obs": o.numpy().astype("float32")})
            if int(ra[0]) != ta or abs(float(rl[0, ta]) - float(net(o)[0][0, ta])) > 1e-4:
                bad += 1
        print("ONNX 自检：8 个随机观测，动作不一致 %d 个" % bad)
        assert bad == 0, "ONNX 与 PyTorch 输出不一致"
    except ImportError:
        print("（没装 onnxruntime，跳过自检）")

    print("已保存：%s（%.1f KB）" % (out, os.path.getsize(out) / 1024))
    return 0


if __name__ == "__main__":
    sys.exit(main())
