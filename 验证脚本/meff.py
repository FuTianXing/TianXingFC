# -*- coding: utf-8 -*-
"""
m_eff 定量：转换波（kink）有效质量的阻尼协议测量与色散形状检验

背景：conversion_wave.py 的 [V2] 用 E_start - E_rest 测
m_eff，被初始脉冲的辐射污染（能量散入呼吸与辐射模）。本脚本用窗口协议：
辐射以链信号速度 c_s 离开 kink（kink 速度 < c_s），kink 中心窗口内的能量在
长时间后 = 干净的 kink 能量（本征 + 平移）。据此测 E(v) 并检验色散形状：
  牛顿：    ΔE = ½ m v²（m 与 v 无关）
  相对论：  ΔE = M c_s² (1/√(1-v²/c_s²) - 1) ≈ ½M v² + (3/8)M v⁴/c_s²（v⁴ 项）
检验：m_eff(v) 是否与 v 无关；v⁴ 系数；两种拟合的残差；m_eff 与 K 的依赖
（连续 sine-Gordon：M ∝ 1/ell ∝ √K）。

模型：FK 链（同 conversion_wave）：E = Σ[½u̇² + (k/2)(u_{n+1}-u_n-1)² + K(1-cos 2πu_n)]
缠绕守恒边界 u_0=0, u_{N-1}=u_0+N。确定性（无随机）。
"""
import numpy as np

k = 1.0
dt = 0.01
N = 1400
cN = N // 2
W = 30          # kink 能量窗口半径（kink 半宽 ~ 2-3 格距，W 富余）

def set_K(Kv):
    global K, w2, c_s
    K = Kv
    w2 = 2 * np.pi * K
    c_s = np.sqrt(k)

def accel(u, v, f=0.0):
    a = np.zeros_like(u)
    a[1:-1] = k * (u[2:] + u[:-2] - 2 * u[1:-1]) - w2 * np.sin(2 * np.pi * u[1:-1])
    a[1:-1] += f
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

def window_energy(u, v, xc):
    n = np.arange(N)
    ms = np.abs(n - xc) <= W
    b = np.arange(N - 1)
    mb = np.abs(b + 0.5 - xc) <= W
    E = 0.5 * np.sum(v[ms] ** 2) + np.sum(K * (1 - np.cos(2 * np.pi * u[ms])))
    E += 0.5 * k * np.sum((u[b[mb] + 1] - u[b[mb]] - 1.0) ** 2)
    return E

def total_energy(u, v):
    return (0.5 * np.sum(v ** 2)
            + 0.5 * k * np.sum((u[1:] - u[:-1] - 1.0) ** 2)
            + np.sum(K * (1 - np.cos(2 * np.pi * u))))

def relax_kink(gamma_d=0.5, steps=12000):
    u = kink_profile(cN * 1.0)
    v = np.zeros(N)
    a = accel(u, v)
    for t in range(steps):
        u = u + dt * v + 0.5 * dt ** 2 * a
        a_new = accel(u, v + dt * a)
        v = v + 0.5 * dt * (a + a_new)
        u[0], u[-1] = 0.0, float(N)
        v[0] = v[-1] = 0.0
        a = a_new
    return u, np.zeros(N)

def run_boost(v0, steps, sample=20):
    """初速度场 = -v0 * 零模（kink 形状导数），演化并采样窗口能量。"""
    u = u_rest.copy()
    v = -v0 * np.gradient(u_rest - np.arange(N))
    v[0] = v[-1] = 0.0
    E0 = total_energy(u, v)
    a = accel(u, v)
    ts, xs, Ews = [], [], []
    for t in range(steps):
        u = u + dt * v + 0.5 * dt ** 2 * a
        a_new = accel(u, v + dt * a)
        v = v + 0.5 * dt * (a + a_new)
        u[0], u[-1] = 0.0, float(N)
        v[0] = v[-1] = 0.0
        a = a_new
        if t % sample == 0:
            xc = center(u)
            ts.append(t * dt); xs.append(xc)
            Ews.append(window_energy(u, v, xc))
            if not np.isnan(xc) and xc > N - 100:
                break
    E1 = total_energy(u, v)
    return np.array(ts), np.array(xs), np.array(Ews), E0, E1

print("=" * 72)
print("m_eff 定量：阻尼协议（窗口能量法）+ 色散形状检验")
print("=" * 72)

results = {}
for K in (0.005, 0.01):
    set_K(K)
    print(f"\n--- K = {K}（c_s = {c_s}，kink 半宽 ~ {1/(2*np.pi*np.sqrt(K/k)):.2f} 格距）---")
    u_rest, v_rest = relax_kink()
    x0 = center(u_rest)
    E_rest_w = window_energy(u_rest, v_rest, x0)
    E_vac_w = window_energy(np.arange(N).astype(float), np.zeros(N), x0)
    print(f"  静止 kink：中心 {x0:.2f}，窗口能量 E_rest_W = {E_rest_w:.5f}，"
          f"E_rest_W - E_vac_W = {E_rest_w - E_vac_w:.5f}")
    rows = []
    for v0 in (0.20, 0.35, 0.50, 0.65, 0.80, 0.90):
        ts, xs, Ews, E0, E1 = run_boost(v0, steps=60000)
        # 匀速窗口：中段
        n1, n2 = len(ts) // 3, 2 * len(ts) // 3
        ok = ~np.isnan(xs[n1:n2])
        v_meas = np.polyfit(ts[n1:n2][ok], xs[n1:n2][ok], 1)[0]
        # 窗口能量取尾段平均（辐射已离开）
        Ew_avg = np.mean(Ews[n2:]) if n2 < len(Ews) else np.mean(Ews[n1:])
        dE = Ew_avg - E_rest_w
        m_eff_v = 2.0 * dE / v_meas ** 2 if abs(v_meas) > 1e-6 else np.nan
        rows.append((v0, v_meas, Ew_avg, dE, m_eff_v, abs(E1 - E0)))
        print(f"  v0 = {v0:4.2f} | v_meas = {v_meas:7.4f} | E_W = {Ew_avg:9.5f} | "
              f"ΔE = {dE:8.5f} | m_eff(v) = {m_eff_v:8.4f} | 能量误差 {abs(E1-E0):.1e}")
    rows = np.array([r for r in rows if np.isfinite(r[1]) and abs(r[1]) > 1e-4])
    v_m, dE_m = rows[:, 1], rows[:, 3]
    # 牛顿拟合（低速点 v < 0.5）
    lo = v_m < 0.5
    m_newt = np.sum(v_m[lo] ** 2 * dE_m[lo]) / np.sum(v_m[lo] ** 4) if lo.sum() else np.nan
    dE_newt = 0.5 * m_newt * v_m ** 2
    res_newt = np.max(np.abs(dE_newt - dE_m)) / np.max(np.abs(dE_m)) if dE_m.size else np.nan
    # 相对论拟合：ΔE = M c_s²(1/√(1-v²/c_s²) - 1)
    g1 = c_s ** 2 * (1.0 / np.sqrt(1.0 - (v_m / c_s) ** 2) - 1.0)
    M_rel = np.sum(g1 * dE_m) / np.sum(g1 ** 2)
    dE_rel = M_rel * g1
    res_rel = np.max(np.abs(dE_rel - dE_m)) / np.max(np.abs(dE_m)) if dE_m.size else np.nan
    # v⁴ 系数：拟合 ΔE = a v² + b v⁴
    A = np.stack([v_m ** 2, v_m ** 4], axis=1)
    (a4, b4), *_ = np.linalg.lstsq(A, dE_m, rcond=None)
    m_eff_low = float(np.median(rows[:, 4][rows[:, 1] < 0.5]))
    results[K] = (m_eff_low, M_rel, res_newt, res_rel, a4, b4)
    print(f"  牛顿拟合 m = {m_newt:.4f}，相对残差 {res_newt:.2e}")
    print(f"  相对论拟合 M = {M_rel:.4f}，相对残差 {res_rel:.2e}")
    print(f"  v⁴ 系数 b = {b4:.5f}（{'>0：相对论性修正出现' if b4 > 0 and b4 > a4 * 1e-4 else '≈0：低速区纯牛顿'}），"
          f"相对论预期 b = 3M/(8c_s²) = {3 * M_rel / 8:.5f}")
    print(f"  低速 m_eff = {m_eff_low:.4f}；M_rel/m_eff_low = {M_rel / m_eff_low:.4f}（≈1 则色散为相对论性）")

print("\n--- m_eff 与 K 的依赖（连续 sine-Gordon 预期 M ∝ √K）---")
Ks, Ms = [], []
for K in (0.005, 0.01, 0.02):
    set_K(K)
    u_rest, v_rest = relax_kink()
    x0 = center(u_rest)
    E_rest_w = window_energy(u_rest, v_rest, x0)
    v0 = 0.35
    ts, xs, Ews, E0, E1 = run_boost(v0, steps=60000)
    n1, n2 = len(ts) // 3, 2 * len(ts) // 3
    ok = ~np.isnan(xs[n1:n2])
    v_meas = np.polyfit(ts[n1:n2][ok], xs[n1:n2][ok], 1)[0]
    Ew_avg = np.mean(Ews[n2:])
    m = 2.0 * (Ew_avg - E_rest_w) / v_meas ** 2
    Ks.append(K); Ms.append(m)
    print(f"  K = {K:.3f}：m_eff = {m:.4f}")
Ms = np.array(Ms); Ks = np.array(Ks)
if len(Ks) > 1:
    slope = np.polyfit(np.log(Ks), np.log(Ms), 1)[0]
    print(f"  m_eff ∝ K^{slope:.2f}（连续 sine-Gordon 预期 0.5）")

print("\n验证完成。")
