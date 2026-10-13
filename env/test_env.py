import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sim_env import FireEnv, OBS_DIM

env = FireEnv(seed=1, episode_s=20.0)
obs, _ = env.reset()
assert obs.shape == (OBS_DIM,), obs.shape
print("obs 维度 OK:", obs.shape, "前 6 维:", np.round(obs[:6], 3))

# 1) 随机策略跑一个回合
rng = np.random.default_rng(0)
tot = 0.0
t0 = time.perf_counter(); n = 0
while True:
    obs, r, term, trunc, info = env.step(rng.integers(0, 2))
    tot += r; n += 1
    if term or trunc:
        break
dt = time.perf_counter() - t0
print("随机策略：%d 步 / %.1fs（%.0f Hz）累计奖励 %.2f | 发射 %d 命中 %d 击杀 %d"
      % (n, dt, n/dt, tot, info["shots"], info["hits"], info["deaths"]))
print("obs 统计：min %.2f max %.2f 有 nan? %s" % (obs.min(), obs.max(), bool(np.isnan(obs).any())))

# 1b) 变化率三维（12/13/14）必须是**活**特征：以前先更新 last_* 再算 obs，恒等于 0
e2 = FireEnv(seed=2, episode_s=20.0)
o, _ = e2.reset()
hist = []
for _ in range(60):
    o, r, term, trunc, info = e2.step(1)
    hist.append(o[12:15].copy())
    if term or trunc:
        break
hist = np.array(hist)
spread = hist.std(axis=0)
print("变化率三维 std：yaw_err %.4f  pitch_err %.4f  cmd_yaw %.4f" % tuple(spread))
assert spread.max() > 1e-6, "obs[12..14] 是死特征（恒为 0）——检查 _last_* 的更新顺序！"

# 2) "每帧都开火" vs "只在瞄准误差很小时开火"，看奖励与命中率的差别
def run(policy, seed=1, ep=60.0):
    e = FireEnv(seed=seed, episode_s=ep)
    o, _ = e.reset()
    tot = 0.0
    while True:
        o, r, term, trunc, info = e.step(policy(o))
        tot += r
        if term or trunc:
            break
    return tot, info

r_always, i_always = run(lambda o: 1)
r_careful, i_careful = run(lambda o: 1 if (abs(o[10]) < 0.004 and abs(o[11]) < 0.004 and o[9] < 0.3) else 0)
print("每帧开火  ：奖励 %6.2f | %d 发 %d 中 = %.1f%%" %
      (r_always, i_always["shots"], i_always["hits"], 100.0*i_always["hits"]/max(i_always["shots"],1)))
print("误差小才开：奖励 %6.2f | %d 发 %d 中 = %.1f%%" %
      (r_careful, i_careful["shots"], i_careful["hits"], 100.0*i_careful["hits"]/max(i_careful["shots"],1)))
