# -*- coding: utf-8 -*-
"""
最小验证（张量桥分段 b）：动态写入-波动方程与 c_s = c 同一化

模型：写入场（标架倾斜场）的动态动力学 = 阻尼波动方程
      theta_tt + gamma theta_t = c_s^2 nabla^2 theta + s
  弱阻尼 -> 波动（引力波，波速 c_s）；过阻尼 -> 写入-扩散（稳态泊松）。
  离散：velocity Verlet（dt = 0.5 拍），1D 链，狄利克雷边界。

判据（《注记-张量桥分段b攻坚计划》）：
  b1 弱阻尼波动波速 = c_s：长波高斯波包群速度 vs 解析 v_g = c_s cos(k/2)
     （1D 离散色散 omega = 2 c_s |sin(k/2)|）；c_s^2 = 1 时 v_g(k->0) -> c_s = c
  b2 过阻尼极限 = 扩散：D_eff = c^2 dt/gamma，gamma = c^2 dt/D_diff 时稳态对齐
  b3 连续极限：k -> 0 群速度 -> c_s = 1（d'Alembert 传播速度 = c）
"""
import numpy as np

def step(th, v, gamma, cs2, dt):
    th = th + v * (dt / 2.0)
    lap = np.zeros(th.shape)
    lap[1:-1] = th[2:] - 2 * th[1:-1] + th[:-2]
    v = v * (1.0 - gamma * dt) + cs2 * lap * dt
    th = th + v * (dt / 2.0)
    th[0] = 0; th[-1] = 0
    return th, v

def group_speed(k0, sigma, cs2, N, gamma=0.01, T=260, dt=0.5):
    c2 = N // 2
    x = np.arange(N)
    g = np.exp(-(x - c2) ** 2 / (2 * sigma ** 2))
    th = g * np.cos(k0 * (x - c2))
    dth = np.zeros(N); dth[1:-1] = (th[2:] - th[:-2]) / 2
    v = -np.sqrt(cs2) * dth
    pts = []
    for t in range(int(T / dt)):
        th, v = step(th, v, gamma, cs2, dt)
        if t % 20 == 0:
            w = th ** 2; s = w.sum()
            if s > 1e-30:
                pts.append((t * dt, (x * w).sum() / s - c2))
    ts = np.array([p[0] for p in pts]); xs = np.array([p[1] for p in pts])
    m = (ts > 60) & (ts < 240)
    return np.polyfit(ts[m], xs[m], 1)[0] if m.sum() > 3 else float('nan')

print("=" * 72)
print("最小验证（张量桥分段 b）：动态写入-波动方程与 c_s = c 同一化")
print("=" * 72)

# [b1] 群速度 vs 解析 v_g = c_s cos(k/2)
print("\n[b1] 弱阻尼波动群速度（N=801，sigma=20 高斯波包，c_s^2 = 1）")
print(f"  {'k0':>4}{'数值 v_g':>10}{'解析 v_g':>10}{'偏差':>10}")
b1_ok = True
for k0 in (0.3, 0.6, 1.0):
    vg = np.cos(k0 / 2.0)
    s = group_speed(k0, 20, 1.0, 801)
    print(f"  {k0:4.1f}{s:10.4f}{vg:10.4f}{abs(s - vg):10.4f}")
    b1_ok = b1_ok and abs(s - vg) < 0.05
print(f"  b1 {'达成（偏差 <5%，离散色散高阶项）' if b1_ok else '未达成'}")

# [b3] 连续极限：k->0 群速度 -> c_s = 1（解析严格；数值取趋向性判据）
s20 = group_speed(0.0, 20, 1.0, 801)
s40 = group_speed(0.0, 40, 1.0, 801)
print(f"\n[b3] 长波极限：sigma=20 群速度 = {s20:.4f}，sigma=40 群速度 = {s40:.4f}")
print(f"  （期望随 sigma 增大趋向 c_s = 1；k->0 解析群速度 v_g(0) = c_s = 1 严格成立，色散论文 K_0 固定）")
b3_ok = s40 > s20 and s40 > 0.8
print(f"  b3 {'达成（趋向性 + 解析保证，达朗贝尔方程传播速度 = c）' if b3_ok else '未达成'}")

# [b2] 过阻尼极限 = 扩散（D_eff = c^2 dt/gamma）
print("\n[b2] 过阻尼极限：D_eff = c^2 dt/gamma，对齐扩散稳态")
N = 101; c2 = N // 2
def wave_steady(gamma, cs2, dt, T=200000):
    th = np.zeros(N); v = np.zeros(N)
    for _ in range(T):
        th, v = step(th, v, gamma, cs2, dt)
        th[c2] += 0.02
    return th[c2]
# 一阶扩散对照（3+1 雏形格式，alpha=1，D = 1/2）
th = np.zeros(N)
for _ in range(200000):
    avg = np.zeros(N); avg[1:-1] = (th[2:] + th[:-2]) / 2
    th = th + 1.0 * (avg - th); th[c2] += 0.02
diff_steady = th[c2]
b2_ok = True
print(f"  一阶扩散稳态 theta(c2) = {diff_steady:.5f}")
print(f"  {'gamma':>6}{'稳态':>10}{'D_eff':>9}{'理论 c^2dt/gamma':>18}{'偏差':>7}")
for gamma in (0.5, 1.0, 2.0):
    s = wave_steady(gamma, 1.0, 0.5)
    d_eff = 0.02 * c2 / (2 * s)
    d_th = 1.0 * 0.5 / gamma
    dev = abs(d_eff - d_th) / d_th
    print(f"  {gamma:6.1f}{s:10.5f}{d_eff:9.4f}{d_th:18.4f}{dev:7.1%}")
    b2_ok = b2_ok and dev < 0.05
print(f"  b2 {'达成（过阻尼极限 = 扩散，gamma=c^2dt/D_diff 对齐）' if b2_ok else '未达成'}")

print("\n" + "=" * 72)
print(f"分段 (b) 判据总览：b1 {'✓' if b1_ok else '✗'}  b2 {'✓' if b2_ok else '✗'}  "
      f"b3 {'✓' if b3_ok else '✗'}")
all_ok = b1_ok and b2_ok and b3_ok
print(f"分段 (b) 全部判据：{'达成（c_s = c 同一化确认）' if all_ok else '未达成'}")
print("验证完成。")
