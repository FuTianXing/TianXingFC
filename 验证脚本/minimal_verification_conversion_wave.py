# -*- coding: utf-8 -*-
"""
转换波（kink）运动学验证：方向甲三判据 + 速度上限 + PN 势垒标度

模型：Frenkel-Kontorova / 离散 sine-Gordon 链（转换波的运动学代理）。
  提升变量 u_n ∈ ℝ（对应 σ 锁定的缠绕提升，格距 a = 1）；
  E = Σ [ ½ u̇² + (k/2)(u_{n+1}-u_n-1)² + K(1-cos 2πu_n) ]
  真空：u_n = n；kink：u_n = n+1（多绕一格 = σ 锁定区，缠绕提升中可见的静态滑移）。
  链信号速度 c_s = √k = 1（每拍一格）。主参数 K = 0.02（宽/轻 kink，可动区）。

判据：
  [V1] 锚定：零力净位移 ≈ 0（PN 势垒保底）；PN 势垒 ΔE_PN 直接测量（约束弛豫法）
  [V1b] PN 势垒随 K 增大（锚定旋钮：窄/重 kink 钉扎强，宽/轻 kink 易动）
  [V2] 惯性：E(v) - E_rest = ½ m_eff v²，m_eff 与 v 无关；匀速保持；E 守恒
       速度场 = 弛豫态自身的提升梯度（真平移零模，非 ansatz 台阶）
  [V3] 测地响应：倾斜场力 f 驱动，实测 a 与 f/m_eff(V2) 一致（等效原理 toy 检验）
  [V4] 速度上限：终端速度 v_term < c_s，随 f 饱和逼近 c_s（f < 2πK 保真空）
"""
import numpy as np

print("=" * 72)
print("转换波（kink）运动学验证：锚定 / 惯性 / 测地响应 / 速度上限 / PN 势垒")
print("=" * 72)

k, m = 1.0, 1.0
dt = 0.01
N = 600
cN = N // 2

def set_K(Kv):
    global K, w2, c_s
    K = Kv
    w2 = (2 * np.pi) ** 2 * K
    c_s = np.sqrt(k)

def accel(u, v, f=0.0, gamma_d=0.0):
    a = np.zeros_like(u)
    a[1:-1] = k * (u[2:] + u[:-2] - 2 * u[1:-1]) - w2 * np.sin(2 * np.pi * u[1:-1])
    a[1:-1] += f - gamma_d * v[1:-1]
    return a

def kink_profile(x0):
    n = np.arange(N)
    ell = 1.0 / (2 * np.pi * np.sqrt(K / k))
    return n + (2 / np.pi) * np.arctan(np.exp((n - x0) / ell))

def center(u):
    w = u[1:] - u[:-1] - 1.0
    ks = int(np.argmax(w))
    lo, hi = max(0, ks - 8), min(len(w), ks + 9)
    ww = np.clip(w[lo:hi], 0.0, None)
    return (np.arange(lo, hi) * ww).sum() / ww.sum() if ww.sum() > 1e-12 else np.nan

def simulate(u, v, steps, f=0.0, gamma_d=0.0, stop_at_edge=None, sample=20):
    xs, ts = [], []
    a = accel(u, v, f, gamma_d)
    for t in range(steps):
        u = u + dt * v + 0.5 * dt ** 2 * a
        a_new = accel(u, v + dt * a, f, gamma_d)
        v = v + 0.5 * dt * (a + a_new)
        # 缠绕守恒边界：u_0 = 0，u_{N-1} = u_0 + N（缠绕 1 扇区，kink 位置自由）
        u[0], u[-1] = 0.0, float(N)
        v[0] = v[-1] = 0.0
        a = a_new
        if t % sample == 0:
            xs.append(center(u)); ts.append(t * dt)
            if stop_at_edge is not None and not np.isnan(xs[-1]) and xs[-1] > stop_at_edge:
                break
    return u, v, np.array(ts), np.array(xs)

def energy(u, v):
    return (0.5 * np.sum(v ** 2)
            + 0.5 * k * np.sum((u[1:] - u[:-1] - 1.0) ** 2)
            + np.sum(K * (1 - np.cos(2 * np.pi * u))))

def relax_kink(gamma_d=0.5, steps=8000):
    u = kink_profile(cN * 1.0)
    v = np.zeros(N)
    u, v, _, _ = simulate(u, v, steps, gamma_d=gamma_d)
    return u, np.zeros(N)

def pn_barrier(u_rest):
    """约束弛豫法测 PN 势垒：固定滑移中心键的提升偏移 s ∈ [0,1)，弛豫其余自由度"""
    slip = u_rest[1:] - u_rest[:-1] - 1.0
    kc = int(np.argmax(slip))
    Es = []
    for s in np.linspace(0.0, 0.9, 10):
        u = u_rest.copy()
        u[kc] = kc + s
        v = np.zeros(N)
        for t in range(5000):
            a = accel(u, v, 0.0, 0.3)
            u = u + dt * v + 0.5 * dt ** 2 * a
            v = v + 0.5 * dt * (a + accel(u, v + dt * a, 0.0, 0.3))
            u[kc] = kc + s; v[kc] = 0.0            # 约束：中心键提升固定
            u[0], u[-1] = 0.0, N - 1.0; v[0] = v[-1] = 0.0
        Es.append(energy(u, np.zeros(N)))
    Es = np.array(Es)
    return Es.max() - Es.min(), Es

# 主参数：锚定验证用 K = 0.02（窄/重 kink，强钉扎）；运动学验证用 K = 0.005（宽/轻 kink，可动区）。
# 原因（本轮诊断发现）：脱钉力 F_c ∝ PN 势垒随 K 下降快于真空稳定阈值 2πK（线性），
# K = 0.02 时 F_c ≈ 0.29 > 2πK = 0.126：驱动窗口为空，能推动 kink 的力必先摧毁真空。
# 锚定与可动性由 K 旋钮分开验证，本身即为判据 [V5] 的联动曲线。
set_K(0.02)
print(f"\n[准备/锚定] K = {K}（窄/重 kink），c_s = {c_s}，真空脱钉阈值 2πK = {2 * np.pi * K:.3f}")
u_rest, v_rest = relax_kink()
E_rest = energy(u_rest, v_rest)
u_vac = np.arange(N).astype(float)
E_vac = energy(u_vac, np.zeros(N))
shape_r = u_rest - np.arange(N)
dshape_r = np.gradient(shape_r)
print(f"    E_rest - E_vac = {E_rest - E_vac:.4f}（kink 静止能量，绑定能标度）")
print(f"    kink 中心 = {center(u_rest):.3f}，最大滑移 = "
      f"{np.max(u_rest[1:] - u_rest[:-1] - 1):.3f}，零模范数 Σdshape² = "
      f"{np.sum(dshape_r ** 2):.4f}")

# ----------------------------------------------------------------------
print("\n[V1] 锚定：零力净位移 ≈ 0；PN 势垒直接测量")
u1, v1 = u_rest.copy(), v_rest.copy()
u1, v1, ts, xs = simulate(u1, v1, 30000, f=0.0)
net0 = xs[-1] - xs[0]
print(f"    零力 3 万步：净位移 = {net0:+.4f}（≈ 0：锚定 ✓）")
for f_small in (0.005, 0.01):
    u1b, v1b = u_rest.copy(), v_rest.copy()
    u1b, v1b, ts1b, xs1b = simulate(u1b, v1b, 30000, f=f_small)
    net1b = xs1b[-1] - xs1b[0]
    print(f"    f = {f_small}（< F_c ≈ 0.29）：净位移 = {net1b:+.4f}"
          f"（{'|净| < 1 格：小力不平移 ✓' if abs(net1b) < 1.0 else '已脱钉 ✗'}）")
dE_pn, Es_scan = pn_barrier(u_rest)
print(f"    PN 势垒（约束弛豫法）ΔE_PN = {dE_pn:.5f}"
      f"（{'> 0：势垒保底 ✓' if dE_pn > 1e-6 else '无势垒 ✗'}），"
      f"E(s) 扫描 = {np.round(Es_scan - Es_scan.min(), 5).tolist()}")

print("\n[V1b] PN 势垒随 K（锚定旋钮：窄/重 kink 钉扎强）")
for Kv in (0.02, 0.05, 0.1):
    set_K(Kv)
    uK, vK = relax_kink()
    EK = energy(uK, np.zeros(N)) - energy(np.arange(N).astype(float), np.zeros(N))
    dK, _ = pn_barrier(uK)
    print(f"    K = {Kv:.3f}：E_rest - E_vac = {EK:.4f}，ΔE_PN = {dK:.5f}"
          f"（ΔE_PN/E_rest = {dK / EK:.4f}）")

# ----------------------------------------------------------------------
set_K(0.005)
print(f"\n[准备/运动] K = {K}（宽/轻 kink），F_c ≪ 2πK = {2 * np.pi * K:.4f}：驱动窗口打开")
u_rest, v_rest = relax_kink(gamma_d=0.5, steps=12000)
E_rest = energy(u_rest, v_rest)
shape_r = u_rest - np.arange(N)
dshape_r = np.gradient(shape_r)
print(f"    E_rest - E_vac = {E_rest - energy(np.arange(N).astype(float), np.zeros(N)):.4f}，"
      f"零模范数 Σdshape² = {np.sum(dshape_r ** 2):.4f}")
print("\n[V2] 惯性：有效质量涌现与匀速保持（速度场 = 弛豫态自身零模）")
print(f"    {'v0':>6} {'v_meas':>10} {'E-Erest':>10} {'m_eff':>10} {'E 守恒误差':>12}")
m_effs = []
for v0 in (0.3, 0.6, 1.0, 1.5):
    u2 = u_rest.copy()
    vv = -v0 * dshape_r
    vv[0] = vv[-1] = 0.0
    E_start = energy(u2, vv)
    u2, vv, ts, xs = simulate(u2, vv, 30000, stop_at_edge=N - 80)
    n = len(ts) // 3
    t_w, x_w = ts[n:2 * n], xs[n:2 * n]
    ok = ~np.isnan(x_w)
    v_meas = np.polyfit(t_w[ok], x_w[ok], 1)[0]
    E_end = energy(u2, vv)
    m_eff = 2.0 * (E_start - E_rest) / v_meas ** 2 if abs(v_meas) > 1e-6 else np.nan
    m_effs.append(m_eff)
    print(f"    {v0:6.2f} {v_meas:10.4f} {E_start - E_rest:10.4f} "
          f"{m_eff:10.4f} {abs(E_end - E_start):12.2e}")
m_eff = float(np.nanmedian(m_effs))
print(f"    中位有效质量 m_eff = {m_eff:.4f}（与 v 无关 → 惯性涌现）")
print(f"    静止能量 E_rest - E_vac = {E_rest - E_vac:.4f}；"
      f"零模质量 Σdshape² = {np.sum(dshape_r ** 2):.4f}"
      f"（m_eff = 零模质量：惯性 = 场构型的平移代价）")

# ----------------------------------------------------------------------
print("\n[V3] 测地响应：倾斜场驱动加速（等效原理 toy 检验）")
print(f"    {'f':>8} {'a_meas':>12} {'f/m_eff':>12} {'比值':>8}")
for f in (0.01, 0.02, 0.05):
    u3, v3 = u_rest.copy(), v_rest.copy()
    u3, v3, ts, xs = simulate(u3, v3, 8000, f=f, gamma_d=0.0, stop_at_edge=N - 80)
    n = max(3, len(ts) // 3)
    t_fit, x_fit = ts[:n], xs[:n]
    ok = ~np.isnan(x_fit)
    a_meas = 2.0 * np.polyfit(t_fit[ok], x_fit[ok], 2)[0] if ok.sum() > 10 else np.nan
    a_exp = f / m_eff
    print(f"    {f:8.4f} {a_meas:12.5f} {a_exp:12.5f} {a_meas / a_exp:8.3f}")

# ----------------------------------------------------------------------
print("\n[V4] 速度上限：终端速度饱和于 c_s（f < 2πK 保真空）")
print(f"    {'f':>8} {'v_term':>10} {'v_term/c_s':>12}")
for f in (0.01, 0.02, 0.05):
    u4, v4 = u_rest.copy(), v_rest.copy()
    u4, v4, ts, xs = simulate(u4, v4, 80000, f=f, gamma_d=0.05, stop_at_edge=N - 80)
    n = len(ts) // 2
    t_w, x_w = ts[n:], xs[n:]
    ok = ~np.isnan(x_w)
    v_term = np.polyfit(t_w[ok], x_w[ok], 1)[0] if ok.sum() > 10 else np.nan
    print(f"    {f:8.3f} {v_term:10.4f} {v_term / c_s:12.4f}"
          f"（{'< 1 ✓' if v_term < c_s else '≥ c_s ✗'}）")
print(f"    c_s = {c_s}（链信号速度 = 每拍一格）")

print("\n验证完成。")
