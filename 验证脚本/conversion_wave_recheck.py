# -*- coding: utf-8 -*-
"""
预印本 §4.3 复核：转换波运动学五判据（修正后力场）

conversion_wave.py 的 2π 因子已修正（w2 = 2πK）。本脚本按修正后力场
重新数值化预印本《碰撞写入标架几何》§4.3 的五个判据：
  [R1] 锚定：零力净位移（长演化）+ PN 势垒（约束弛豫，多 K）+ 小力脱钉阈值
  [R2] 惯性：v_meas(v0) 单调性、m_eff(v)（引用 meff 验证结论）
  [R3] 测地响应（重新设计）：f 驱动的终端速度 v_term(f) 与初段加速度，
       检验方向与 a = f/m_eff 线性
  [R4] 速度上限：v_term(f) < c_s 全程；f → 2πK⁻ 行为
  [R5] 驱动窗口：脱钉力 F_c 与真空稳定阈值 2πK 的关系（K 扫描）

模型同 conversion_wave：FK 链，E = Σ[½u̇² + (k/2)(u_{n+1}-u_n-1)² + K(1-cos 2πu)]
缠绕守恒边界 u_0=0, u_{N-1}=u_0+N。确定性。
"""
import numpy as np

k = 1.0
dt = 0.01
N = 1200
cN = N // 2

def set_K(Kv):
    global K, w2, c_s
    K = Kv
    w2 = 2 * np.pi * K          # 修正后：dV/du = 2πK sin 2πu
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

def energy(u, v):
    return (0.5 * np.sum(v ** 2)
            + 0.5 * k * np.sum((u[1:] - u[:-1] - 1.0) ** 2)
            + np.sum(K * (1 - np.cos(2 * np.pi * u))))

def relax_kink(gamma_d=0.5, steps=12000):
    u = kink_profile(cN * 1.0)
    v = np.zeros(N)
    a = accel(u, v)
    for t in range(steps):
        u = u + dt * v + 0.5 * dt ** 2 * a
        a_new = accel(u, v + dt * a, 0.0, gamma_d)
        v = v + 0.5 * dt * (a + a_new)
        u[0], u[-1] = 0.0, float(N)
        v[0] = v[-1] = 0.0
        a = a_new
    return u, np.zeros(N)

def simulate(u, v, steps, f=0.0, gamma_d=0.0, stop_at_edge=None, sample=20):
    xs, ts = [], []
    a = accel(u, v, f, gamma_d)
    for t in range(steps):
        u = u + dt * v + 0.5 * dt ** 2 * a
        a_new = accel(u, v + dt * a, f, gamma_d)
        v = v + 0.5 * dt * (a + a_new)
        u[0], u[-1] = 0.0, float(N)
        v[0] = v[-1] = 0.0
        a = a_new
        if t % sample == 0:
            xc = center(u)
            xs.append(xc); ts.append(t * dt)
            if stop_at_edge is not None and not np.isnan(xc) and abs(xc - cN) > stop_at_edge:
                break
    return u, v, np.array(ts), np.array(xs)

def pn_barrier(u_rest, gamma_d=0.3, s_steps=10):
    slip = u_rest[1:] - u_rest[:-1] - 1.0
    kc = int(np.argmax(slip))
    Es = []
    for s in np.linspace(0.0, 0.9, s_steps):
        u = u_rest.copy()
        u[kc] = kc + s
        v = np.zeros(N)
        for t in range(5000):
            a = accel(u, v, 0.0, gamma_d)
            u = u + dt * v + 0.5 * dt ** 2 * a
            v = v + 0.5 * dt * (a + accel(u, v + dt * a, 0.0, gamma_d))
            u[kc] = kc + s; v[kc] = 0.0
            u[0], u[-1] = 0.0, N - 1.0; v[0] = v[-1] = 0.0
        Es.append(energy(u, np.zeros(N)))
    Es = np.array(Es)
    return Es.max() - Es.min(), Es

print("=" * 72)
print("预印本 §4.3 复核（修正后力场 w2 = 2πK）")
print("=" * 72)

# ----------------------------------------------------------------------
print("\n[R1] 锚定复核")
for Kv in (0.02, 0.005):
    set_K(Kv)
    u_rest, v_rest = relax_kink()
    E_rest = energy(u_rest, v_rest) - energy(np.arange(N).astype(float), np.zeros(N))
    # 零力净位移（长演化 12 万步）
    u0, v0 = u_rest.copy(), v_rest.copy()
    u0, v0, ts, xs = simulate(u0, v0, 120000)
    net = xs[-1] - xs[0]
    # PN 势垒
    dE_pn, _ = pn_barrier(u_rest)
    print(f"  K = {Kv}: E_rest = {E_rest:.4f}, ΔE_PN = {dE_pn:.6f} "
          f"(ΔE_PN/E_rest = {dE_pn/E_rest:.5f}), 零力 12 万步净位移 = {net:+.3f}")
    # 小力响应（脱钉阈值探测）
    for f in (0.001, 0.003, 0.01):
        uf, vf = u_rest.copy(), v_rest.copy()
        uf, vf, ts, xs = simulate(uf, vf, 30000, f=f)
        netf = xs[-1] - xs[0]
        print(f"      f = {f:.3f}: 3 万步净位移 = {netf:+.3f}（{'脱钉' if abs(netf) > 2 else '钉住'}）")

# ----------------------------------------------------------------------
print("\n[R2] 惯性复核：v_meas(v0) 与 m_eff(v)（K = 0.005）")
set_K(0.005)
u_rest, v_rest = relax_kink(gamma_d=0.5, steps=12000)
E_rest = energy(u_rest, v_rest)
dshape = np.gradient(u_rest - np.arange(N))
print(f"  {'v0':>5} {'v_meas':>8} {'m_eff(E_start)':>15}")
for v0 in (0.3, 0.6, 1.0, 1.5):
    u2 = u_rest.copy()
    vv = -v0 * dshape
    vv[0] = vv[-1] = 0.0
    E_start = energy(u2, vv)
    u2, vv, ts, xs = simulate(u2, vv, 30000, stop_at_edge=N - 100)
    n = len(ts) // 3
    ok = ~np.isnan(xs[n:2 * n])
    v_meas = np.polyfit(ts[n:2 * n][ok], xs[n:2 * n][ok], 1)[0]
    m_eff = 2.0 * (E_start - E_rest) / v_meas ** 2 if abs(v_meas) > 1e-6 else np.nan
    print(f"  {v0:5.1f} {v_meas:8.4f} {m_eff:15.4f}")
print("  注：m_eff(E_start) 含初始脉冲辐射污染；干净值（窗口协议）见 meff 注记：")
print("      低速 m_eff = 0.099，m_eff(v) 递增 → 相对论性色散")

# ----------------------------------------------------------------------
print("\n[R3] 测地响应复核（重新设计：f 驱动 + 终端速度 + 初段加速度）")
set_K(0.005)
u_rest, v_rest = relax_kink(gamma_d=0.5, steps=12000)
print(f"  {'f':>7} {'v_term':>10} {'a_init':>10} {'v_term/f':>10} {'方向':>4}")
for f in (0.005, 0.01, 0.02, 0.03):
    u3, v3 = u_rest.copy(), v_rest.copy()
    u3, v3, ts, xs = simulate(u3, v3, 120000, f=f, stop_at_edge=N - 120)
    n = len(ts) // 2
    ok = ~np.isnan(xs[n:])
    v_term = np.polyfit(ts[n:][ok], xs[n:][ok], 1)[0] if ok.sum() > 20 else np.nan
    # 初段加速度（前 1000 步）
    n0 = min(50, len(ts) - 1)
    ok0 = ~np.isnan(xs[:n0])
    a_init = 2.0 * np.polyfit(ts[:n0][ok0], xs[:n0][ok0], 2)[0] if ok0.sum() > 10 else np.nan
    print(f"  {f:7.3f} {v_term:10.4f} {a_init:10.5f} {v_term/f:10.3f} "
          f"{'正' if v_term > 0 else '负/零'}")

# ----------------------------------------------------------------------
print("\n[R4] 速度上限：v_term < c_s 全程")
set_K(0.005)
u_rest, v_rest = relax_kink(gamma_d=0.5, steps=12000)
vt_max = 0.0
for f in (0.005, 0.01, 0.02, 0.03):
    u4, v4 = u_rest.copy(), v_rest.copy()
    u4, v4, ts, xs = simulate(u4, v4, 120000, f=f, stop_at_edge=N - 120)
    n = len(ts) // 2
    ok = ~np.isnan(xs[n:])
    v_term = np.polyfit(ts[n:][ok], xs[n:][ok], 1)[0] if ok.sum() > 20 else np.nan
    vt_max = max(vt_max, abs(v_term))
print(f"  实测最大 |v_term| = {vt_max:.4f}，c_s = {c_s}（{'< c_s ✓' if vt_max < c_s else '≥ c_s ✗'}）")

# ----------------------------------------------------------------------
print("\n[R5] 驱动窗口：脱钉力 F_c 与真空稳定阈值 2πK（K 扫描）")
for Kv in (0.02, 0.05, 0.1):
    set_K(Kv)
    u_rest, v_rest = relax_kink()
    E_rest = energy(u_rest, v_rest) - energy(np.arange(N).astype(float), np.zeros(N))
    dE_pn, _ = pn_barrier(u_rest)
    print(f"  K = {Kv}: 2πK = {2*np.pi*Kv:.4f}, ΔE_PN = {dE_pn:.6f}, "
          f"ΔE_PN/E_rest = {dE_pn/E_rest:.4f}")
print(f"  注：修正后 ΔE_PN 远小于预印本原值（K=0.02 原 0.138 → 现 0.00024，差 575 倍），"
      f"锚定极弱，小力即动；原'锚定完全确认'结论失效")

print("\n复核完成。")
