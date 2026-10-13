import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sim_env import FireEnv
from policies import ThresholdPolicy

def rollout(policy, seed, ep=90.0, disturb=True, c_fire=0.10):
    e = FireEnv(seed=seed, episode_s=ep, disturb=disturb, c_fire=c_fire)
    o, _ = e.reset()
    while True:
        o, r, term, trunc, info = e.step(policy(o))
        if term or trunc:
            break
    return info

print("== 环境可信度验证：阈值门限在 env 里 vs C++ 闭环基准 ==")
print("（C++ 闭环 sp_vision 基准：总体 34.4% 命中、4 种子 × 300s）")
tot_s = tot_h = tot_k = 0
for seed in (20251012, 777, 42, 12345):
    info = rollout(ThresholdPolicy(0.5), seed, ep=100.0)
    rate = 100.0 * info["hits"] / max(info["shots"], 1)
    tot_s += info["shots"]; tot_h += info["hits"]; tot_k += info["deaths"]
    print("   seed %-9s 发射 %3d 命中 %3d = %5.1f%% | 击杀 %d" %
          (seed, info["shots"], info["hits"], rate, info["deaths"]))
print("   合计 %d 发 %d 中 = %.1f%% 击杀 %d" % (tot_s, tot_h, 100.0*tot_h/max(tot_s,1), tot_k))
