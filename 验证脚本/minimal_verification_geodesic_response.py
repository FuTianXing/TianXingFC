# -*- coding: utf-8 -*-
"""
测地响应线性验证与质量对齐 v2：统一 K，演化中采样窗口能量

修正 v1 的两处问题：窗口能量在演化循环中采样（跟随当前 kink 中心）；
统一 K = 0.02（并行会话惯性-响应合并实验的 K，真空阈值 2πK = 0.126）。
同一 kink 上对齐四质量：M_zero、M_shape、m_eff（窗口）、m_resp（阻尼迁移率）。
"""
import numpy as np

k = 1.0
dt = 0.01
N = 1400
cN = N // 2
W = 30

def set_K(Kv):
    global K, w2, c_s
    K = Kv
    w2 = 2 * np.pi * K
    c_s = 1.0

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

def window_energy(u, v, xc):
    n = np.arange(N)
    ms = np.abs(n - xc) <= W
    b = np.arange(N - 1)
    mb = np.abs(b + 0.5 - xc) <= W
    E = 0.5 * np.sum(v[ms] ** 2) + np.sum(K * (1 - np.cos(2 * np.pi * u[ms])))
    E += 0.5 * k * np.sum((u[b[mb] + 1] - u[b[mb]] - 1.0) ** 2)
    return E

def relax_kink(gamma_d=0.5, steps=12000):
    u = kink_profile(cN * 1.0)
    v = np.zeros(N)
    a = accel(u, v, 0.0, gamma_d)
    for t in range(steps):
        u = u + dt * v + 0.5 * dt ** 2 * a
        a_new = accel(u, v + dt * a, 0.0, gamma_d)
        v = v + 0.5 * dt * (a + a_new)
        u[0], u[-1] = 0.0, float(N)
        v[0] = v[-1] = 0.0
        a = a_new
    return u, np.zeros(N)

def run(f=0.0, gamma_d=0.0, v0=None, steps=150000, sample=20):
    """通用演化：可选初速度 boost，演化中采样中心与窗口能量。"""
    u = u_rest.copy()
    if v0 is not None:
        v = -v0 * np.gradient(u_rest - np.arange(N))
    else:
        v = np.zeros(N)
    v[0] = v[-1] = 0.0
    a = accel(u, v, f, gamma_d)
    ts, xs, Ews = [], [], []
    for t in range(steps):
        u = u + dt * v + 0.5 * dt ** 2 * a
        a_new = accel(u, v + dt * a, f, gamma_d)
        v = v + 0.5 * dt * (a + a_new)
        u[0], u[-1] = 0.0, float(N)
        v[0] = v[-1] = 0.0
        a = a_new
        if t % sample == 0:
            xc = center(u)
            ts.append(t * dt); xs.append(xc)
            Ews.append(window_energy(u, v, xc))
            if not np.isnan(xc) and abs(xc - cN) > N - 150:
                break
    return np.array(ts), np.array(xs), np.array(Ews)

print("=" * 72)
print("测地响应线性验证与质量对齐 v2（统一 K = 0.02）")
print("=" * 72)

for Kv in (0.02, 0.005):
    set_K(Kv)
    print(f"\n--- K = {Kv}（真空阈值 2πK = {2*np.pi*Kv:.3f}）---")
    u_rest, v_rest = relax_kink()
    x0 = center(u_rest)
    w = u_rest[1:] - u_rest[:-1] - 1.0
    M_zero = np.sum(np.gradient(u_rest - np.arange(N)) ** 2)
    M_shape = np.sum(w ** 2)
    S_tot = np.sum(w)
    E_rest_w = window_energy(u_rest, v_rest, x0)
    print(f"  M_zero = {M_zero:.4f}, M_shape = {M_shape:.4f}, S_tot = {S_tot:.4f}, "
          f"E_rest_W = {E_rest_w:.5f}, 中心 = {x0:.2f}")

    # 窗口协议 m_eff（演化中采样）
    m_effs = []
    for v0 in (0.30, 0.50, 0.70):
        ts, xs, Ews = run(v0=v0, steps=90000)
        n1, n2 = len(ts) // 3, 2 * len(ts) // 3
        ok = ~np.isnan(xs[n1:n2])
        v_meas = np.polyfit(ts[n1:n2][ok], xs[n1:n2][ok], 1)[0]
        Ew_avg = np.mean(Ews[n2:]) if n2 < len(Ews) else np.mean(Ews[n1:])
        dE = Ew_avg - E_rest_w
        m = 2.0 * dE / v_meas ** 2 if abs(v_meas) > 1e-4 else np.nan
        m_effs.append(m)
        print(f"  窗口 v0={v0:.2f}: v_meas = {v_meas:.4f}, ΔE = {dE:.5f}, m_eff = {m:.4f}")
    m_eff = float(np.nanmedian(m_effs)) if m_effs else np.nan
    print(f"  → 窗口 m_eff = {m_eff:.4f}；M_zero = {M_zero:.4f}（比值 {m_eff/M_zero:.3f}）")

    # 阻尼协议 m_resp（γ_d = 0.2，f < 2πK）
    gd = 0.2
    print(f"  [阻尼] γ_d = {gd}")
    for f in (0.005, 0.01, 0.02, 0.04):
        ts, xs, Ews = run(f=f, gamma_d=gd, steps=150000)
        n = len(ts) // 2
        ok = ~np.isnan(xs[n:])
        v_term = np.polyfit(ts[n:][ok], xs[n:][ok], 1)[0] if ok.sum() > 20 else np.nan
        mu = v_term / f if f > 0 else np.nan
        m_resp = f / (gd * abs(v_term)) if abs(v_term) > 1e-6 else np.nan
        print(f"    f = {f:.3f}: v_term = {v_term:+.4f}, μ = {mu:+.2f}, m_resp = {m_resp:.4f}")
    print(f"  μ_theory = 1/(γ_d S_tot) = {1/(gd*S_tot):.2f}")

print("\n验证完成。")
