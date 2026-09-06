# -*- coding: utf-8 -*-
"""
μ 差异调查：阻尼平衡理论与实测 5 倍差异的诊断

理论：kink 匀速移动时 Σu̇ = Σφ'·v = S_tot·v（S_tot = 总缠绕 = 1），
阻尼力 = γ_d·Σu̇ = γ_d·v，平衡 f = γ_d·v → μ_theory = v/f = 1/γ_d = 5。
实测 μ ≈ 25（K=0.005，γ_d=0.2）。差异意味着 Σu̇ 实际 ≈ v/5。

诊断协议：稳态驱动（f, γ_d）下测量：
  1. v_term（kink 终端速度）
  2. Σu̇ = Σv_n（全场速度总和）
  3. Σ|v_n|（速度场绝对总量）
  4. kink 窗口内 Σv（平移模贡献）与窗口外 Σv（声子流贡献）
  5. 速度场形状 v_n 对 φ'_n 的拟合比例（kink 局域）
对照：Σu̇ vs v_term·S_tot；f vs γ_d·Σu̇。
"""
import numpy as np

k, K, dt, N = 1.0, 0.005, 0.01, 1200
cN = N // 2
c_s = np.sqrt(k)
w2 = 2 * np.pi * K          # 修正后力场
gd = 0.2
W = 30

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

print("=" * 72)
print("μ 差异调查：阻尼平衡理论 vs 实测")
print("=" * 72)

u_rest, v_rest = relax_kink()
w = u_rest[1:] - u_rest[:-1] - 1.0
S_tot = np.sum(w)
dphi = np.gradient(u_rest - np.arange(N))
M_zero = np.sum(dphi ** 2)
print(f"S_tot = {S_tot:.4f}（总缠绕），M_zero = {M_zero:.4f}，最大滑移 = {np.max(w):.4f}")
print(f"μ_theory = 1/(γ_d·S_tot) = {1/(gd*S_tot):.2f}")

for f in (0.005, 0.01):
    u = u_rest.copy()
    v = np.zeros(N)
    a = accel(u, v, f, gd)
    # 长演化到稳态，采样末期全场量
    steps = 150000
    rec = []
    for t in range(steps):
        u = u + dt * v + 0.5 * dt ** 2 * a
        a_new = accel(u, v + dt * a, f, gd)
        v = v + 0.5 * dt * (a + a_new)
        u[0], u[-1] = 0.0, float(N)
        v[0] = v[-1] = 0.0
        a = a_new
        if t > steps - 20000 and t % 100 == 0:
            xc = center(u)
            rec.append((t, xc, v.copy()))
    # 稳态统计
    xs = np.array([r[1] for r in rec])
    vk = np.polyfit(np.arange(len(rec)) * 100 * dt, xs, 1)[0]
    # 用最后一个时刻的场做分布诊断
    t_last, xc_last, v_last = rec[-1]
    S_v = np.sum(v_last)
    S_abs = np.sum(np.abs(v_last))
    n = np.arange(N)
    mask_in = np.abs(n - xc_last) <= W
    mask_out = ~mask_in
    S_in = np.sum(v_last[mask_in])
    S_out = np.sum(v_last[mask_out])
    S_abs_in = np.sum(np.abs(v_last[mask_in]))
    S_abs_out = np.sum(np.abs(v_last[mask_out]))
    # kink 速度场对 φ' 的拟合（窗口内）
    vi = v_last[mask_in]
    di = dphi[mask_in]
    A = np.stack([di, np.ones(vi.size)], axis=1)
    (coef, c0), *_ = np.linalg.lstsq(A, vi, rcond=None)
    # 阻尼平衡检查
    damp = gd * S_v
    print(f"\nf = {f}: v_term = {vk:+.4f}, Σu̇ = {S_v:+.4f}, Σ|v| = {S_abs:.4f}")
    print(f"  Σu̇/(v·S_tot) = {S_v/(vk*S_tot):.3f}（理论应为 1）")
    print(f"  μ 实测 = v/f = {vk/f:.2f}；阻尼力 γ_d·Σu̇ = {damp:+.4f} vs f = {f}")
    print(f"  窗口内 Σv = {S_in:+.4f}（平移模），窗口外 Σv = {S_out:+.4f}（声子流）")
    print(f"  Σ|v| 窗口内/外 = {S_abs_in:.3f} / {S_abs_out:.3f}")
    print(f"  v_n vs φ' 拟合（窗口内）：斜率 = {coef:+.4f}（理论 -v = {-vk:.4f}），截距 = {c0:+.5f}")
    ratio = abs(coef + vk) / (abs(vk) + 1e-12)
    tag = "近似平移模 -v·phi'" if ratio < 0.2 else "偏离平移模"
    print(f"  → v_n 形状拟合斜率 = {coef:+.4f}（理论 -v = {-vk:.4f}，偏离 {ratio:.2f}）：{tag}")

print("\n调查完成。")
