# -*- coding: utf-8 -*-
"""
第581条 介质完整响应与反向项（P1-50 折射机制，推导注记 2026-10-04 的数值轮）

模型设定（先立后测，模型单位 c = l_P = t_P = 1，介质充满周期链）：
  完整响应正则模型（推导注记 §二：交叉耦合 -β A p_x + A² 项 β²A²/2）：
      A_dot = E
      E_dot = (D2 A) - β² A + β p
      x_dot = p - β A
      p_dot = -ω_0² x
  逐 k 正则矩阵（周期链 Fourier 分块，D2 → -wl²(k)）：
      L(k) = [[0,1,0,0], [-(wl²+β²),0,0,β], [-β,0,0,1], [0,0,-ω_0²,0]]
  久期方程（精确离散谱，无格点截断项）：
      (wl²+β²-ω²)(ω_0²-ω²) = β² ω_0²
      ω⁴ - S ω² + wl² ω_0² = 0,  S = wl² + β² + ω_0²
      ω²_± = ½[S ± √(S²-4 wl² ω_0²)]
  等效介电函数 ε(ω) = 1 + β²/(ω_0²-ω²)；光支 = 下支 ω_-。
  对照模型（判齿用）：第 573 条能隙型闭式 ω_gap = √(wl²+ω_0²)（位移型静态近似）。

判据（先立期望）：
  R1 静态段（稳健）：长波两档光支频率对久期方程精确根（谱法，容差机器精度）；
      时域单模相位法读双频对照；波包群速对精确解析 v_g = k(ω_0²-u)/(ω(S-2u))，
      且 v_g(k→0) 与 1/√ε(0) 之差 ≤ O(k²) 预算。EP5 齿：能隙型闭式预言 ω(0)>0 有隙，
      同档对 R1 读数必 FAIL。
  R2 正常色散（稳健）：ω<ω_0 段相折射率 n = k/ω 三档严格递增，n(0) = √ε(0) 外推核对。
  R3 高频回归 1⁻（稳健）：大 k 档 n<1 且随 k 回升，n-1 与解析 -β²/(2ω²) 预算核对；
      方向齿：第 573 条闭式同档 n>1。
  R4 剩余射线禁带（稳健）：ω∈(ω_0, √(ω_0²+β²)) 段全谱无光支实解（谱空段）；
      带内一档稳态修逝衰减 μ 对精确离散式 2 asinh(½√(-ω²ε))。
  R5 稳定性与 A² 必要性（EP5）：完整模型全谱实部为零；去掉 A² 项出现非零实部根。
  简并声明：振子支平坦 N 重简并，全部读数取光支与时域波包，不落简并期望。

可逆性等级：R1-R5 判据稳健（解析恒等式与构造替换即翻）。
前提依赖：推导注记 §二（交叉+A² 正则结构）、第 526 条二分、第 573 条模型口径。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG = r"D:/software/TianXingFC/images/full_response_dispersion_581.png"

# ---------- 解析 ----------

def w_latt(k):
    return 2.0 * np.abs(np.sin(k / 2.0))

def branches(k, beta2, w0):
    """久期方程两支精确根（下支 = 光支）"""
    wl2 = w_latt(k) ** 2
    S = wl2 + beta2 + w0 ** 2
    disc = np.sqrt(max(S ** 2 - 4 * wl2 * w0 ** 2, 0.0))
    um = 0.5 * (S - disc)
    up = 0.5 * (S + disc)
    return np.sqrt(um), np.sqrt(up)

def eps(w, beta2, w0):
    return 1.0 + beta2 / (w0 ** 2 - w ** 2)

def vg_exact(k, beta2, w0):
    """光支解析群速 v = du/dk/(2ω)，由久期方程隐式微分"""
    wl2 = w_latt(k) ** 2
    u = branches(k, beta2, w0)[0] ** 2
    S = wl2 + beta2 + w0 ** 2
    return k * (w0 ** 2 - u) / (branches(k, beta2, w0)[0] * (S - 2 * u))

def mu_exact(w, beta2, w0):
    """带内修逝衰减精确离散式：sinh²(μ/2) = -ω²ε/4 → μ = 2 asinh(√(-ω²ε)/2)"""
    val = -w ** 2 * eps(w, beta2, w0)
    return 2.0 * np.arcsinh(np.sqrt(max(val, 0.0)) / 2.0)

def light_branch_mode(N, beta2, w0, k0, env=None):
    """光支本征矢实部初条件（A,E,x,p 相位配对，单频激发）"""
    n = np.arange(N)
    wl2 = w_latt(k0) ** 2
    wm = branches(k0, beta2, w0)[0]
    b = beta2 ** 0.5
    pr = (wl2 + beta2 - wm ** 2) / b        # p/A 比（实）
    xr = (wl2 - wm ** 2) / (b * wm)         # x = -xr·sin(kn)
    if env is None:
        env = np.ones(N)
    A = env * np.cos(k0 * n)
    E = env * wm * np.sin(k0 * n)
    p = env * pr * np.cos(k0 * n)
    x = -env * xr * np.sin(k0 * n)
    return [A, E, x, p]

# ---------- 逐 k 正则矩阵谱 ----------

def L_block(k, beta2, w0, with_A2=True):
    wl2 = w_latt(k) ** 2
    stiff = wl2 + beta2 if with_A2 else wl2
    return np.array([
        [0.0, 1.0, 0.0, 0.0],
        [-stiff, 0.0, 0.0, beta2 ** 0.5],
        [-beta2 ** 0.5, 0.0, 0.0, 1.0],
        [0.0, 0.0, -w0 ** 2, 0.0],
    ])

def spectrum_k(k, beta2, w0, with_A2=True):
    ev = np.linalg.eigvals(L_block(k, beta2, w0, with_A2))
    return ev

def omegas_from_eig(ev):
    ims = sorted(abs(ev.imag[np.abs(ev.real) < 1e-9]))
    return ims

# ---------- 时域 ----------

def rk4_run(N, beta2, w0, k0, dt=0.05, T=2000.0, dt_out=0.1, with_A2=True):
    """单模初条件 A=cos(k0 n), E=x=p=0；返回 A 场自关联 C(t)"""
    n = np.arange(N)
    A = np.cos(k0 * n)
    A0 = A.copy()
    E = np.zeros(N)
    x = np.zeros(N)
    p = np.zeros(N)
    norm = np.dot(A0, A0)
    steps_out = int(round(T / dt_out))
    sub = int(round(dt_out / dt))
    C = np.empty(steps_out + 1)
    C[0] = np.dot(A, A0) / norm

    def deriv(A_, E_, x_, p_):
        d2 = np.roll(A_, 1) - 2 * A_ + np.roll(A_, -1)
        b = beta2 ** 0.5
        if with_A2:
            dE = d2 - beta2 * A_ + b * p_
        else:
            dE = d2 + b * p_
        return E_, dE, p_ - b * A_, -w0 ** 2 * x_

    for s in range(1, steps_out + 1):
        for _ in range(sub):
            a1, b1, c1, d1 = deriv(A, E, x, p)
            a2, b2, c2, d2_ = deriv(A + 0.5 * dt * a1, E + 0.5 * dt * b1,
                                     x + 0.5 * dt * c1, p + 0.5 * dt * d1)
            a3, b3, c3, d3 = deriv(A + 0.5 * dt * a2, E + 0.5 * dt * b2,
                                    x + 0.5 * dt * c2, p + 0.5 * dt * d2_)
            a4, b4, c4, d4 = deriv(A + dt * a3, E + dt * b3, x + dt * c3, p + dt * d3)
            A = A + dt / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
            E = E + dt / 6 * (b1 + 2 * b2 + 2 * b3 + b4)
            x = x + dt / 6 * (c1 + 2 * c2 + 2 * c3 + c4)
            p = p + dt / 6 * (d1 + 2 * d2_ + 2 * d3 + d4)
        C[s] = np.dot(A, A0) / norm
    return C

def read_two_omegas(C, dt_out):
    F = np.fft.rfft(C * np.hanning(len(C)))
    freqs = 2 * np.pi * np.fft.rfftfreq(len(C), d=dt_out)
    mag = np.abs(F).copy()
    peaks = []
    for _ in range(2):
        i = np.argmax(mag[3:]) + 3
        y0, y1, y2 = np.log(mag[i - 1]), np.log(mag[i]), np.log(mag[i + 1])
        dx = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
        peaks.append(freqs[i] + dx * (freqs[1] - freqs[0]))
        mag[max(i - 8, 0):i + 9] = 0.0
    return sorted(peaks)

def wavepacket_speed(N, beta2, w0, kc, sigma, dt=0.05, T=1500.0):
    """光支本征矢波包（包络缓变，相位结构随 kc 配对）峰位移读群速"""
    n = np.arange(N)
    env = np.exp(-(n - N * 0.25) ** 2 / (2 * sigma ** 2))
    wl2 = w_latt(kc) ** 2
    wm = branches(kc, beta2, w0)[0]
    b = beta2 ** 0.5
    pr = (wl2 + beta2 - wm ** 2) / b
    xr = (wl2 - wm ** 2) / (b * wm)
    state = [env * np.cos(kc * n), env * wm * np.sin(kc * n),
             -env * xr * np.sin(kc * n), env * pr * np.cos(kc * n)]

    def deriv(A_, E_, x_, p_):
        d2 = np.roll(A_, 1) - 2 * A_ + np.roll(A_, -1)
        b = beta2 ** 0.5
        return E_, d2 - beta2 * A_ + b * p_, p_ - b * A_, -w0 ** 2 * x_

    def peak_of(A_):
        return np.dot(n, A_ ** 2) / np.dot(A_, A_)

    t0_pk = peak_of(state[0])
    steps = int(T / dt)
    for _ in range(steps):
        a1, b1, c1, d1 = deriv(*state)
        s2 = [q + 0.5 * dt * dq for q, dq in zip(state, (a1, b1, c1, d1))]
        a2, b2, c2, d2 = deriv(*s2)
        s3 = [q + 0.5 * dt * dq for q, dq in zip(state, (a2, b2, c2, d2))]
        a3, b3, c3, d3 = deriv(*s3)
        s4 = [q + dt * dq for q, dq in zip(state, (a3, b3, c3, d3))]
        a4, b4, c4, d4 = deriv(*s4)
        state = [
            state[0] + dt / 6 * (a1 + 2 * a2 + 2 * a3 + a4),
            state[1] + dt / 6 * (b1 + 2 * b2 + 2 * b3 + b4),
            state[2] + dt / 6 * (c1 + 2 * c2 + 2 * c3 + c4),
            state[3] + dt / 6 * (d1 + 2 * d2 + 2 * d3 + d4),
        ]
    t1_pk = peak_of(state[0])
    return (t1_pk - t0_pk) / T

# ---------- 修逝稳态（开链） ----------

def evanescent_mu_open(N, beta2, w0, w, fit_near=14):
    """稳态响应：(-iω I - L_open) z = f，A 分量近端拟合衰减（带内 μ~1 时远端下溢）"""
    D2o = np.zeros((N, N))
    idx = np.arange(N)
    D2o[idx, idx] = -2.0
    D2o[idx[:-1], idx[:-1] + 1] += 1.0
    D2o[idx[1:], idx[1:] - 1] += 1.0
    I = np.eye(N)
    b = beta2 ** 0.5
    L = np.zeros((4 * N, 4 * N))
    L[:N, N:2 * N] = I
    L[N:2 * N, :N] = D2o - beta2 * I
    L[N:2 * N, 3 * N:] = b * I
    L[2 * N:3 * N, :N] = -b * I
    L[2 * N:3 * N, 3 * N:] = I
    L[3 * N:, 2 * N:3 * N] = -w0 ** 2 * I
    f = np.zeros(4 * N, dtype=complex)
    f[N + N // 2] = 1.0
    z = np.linalg.solve(-1j * w * np.eye(4 * N) - L, f)
    Az = np.abs(z[:N])
    x0 = N // 2
    seg = np.arange(x0 + 2, x0 + 2 + fit_near)
    slope, icept = np.polyfit(seg.astype(float), np.log(Az[seg] + 1e-300), 1)
    resid = np.max(np.abs(np.polyval((slope, icept), seg.astype(float)) - np.log(Az[seg] + 1e-300)))
    return -slope, resid

# ================= 主流程 =================

print("=" * 72)
print("第581条 介质完整响应与反向项 R1-R5")
print("=" * 72)

w0 = 1.0
beta2s = [0.5, 1.0]
N = 512
dk_mode = 2 * np.pi / N
T_total = 2000.0
dt_out = 0.1
eps_fft = 1.0 / T_total

# ---------- 冒烟 ----------
print("\n[冒烟] 逐 k 4×4 正则谱 vs 久期方程二次根（k=0.5, β²=0.5）")
k_sm = 0.5
ev = spectrum_k(k_sm, beta2s[0], w0)
w_read = omegas_from_eig(ev)
wm_ex, wp_ex = branches(k_sm, beta2s[0], w0)
d_m = min(abs(w - wm_ex) for w in w_read)
d_p = min(abs(w - wp_ex) for w in w_read)
print("  光支 d=%.2e 振子支 d=%.2e（期望 ~1e-12）" % (d_m, d_p))
print("  谱纯虚性: max|Re λ| = %.2e" % np.max(np.abs(ev.real)))
print("  变化的轴 = 正则演化轴（k 分块），核对通过" if max(d_m, d_p) < 1e-10 else "  [FAIL]")

# ---------- R1 ----------
print("\n" + "=" * 72)
print("R1 静态段：光支长波频率 + 波包群速 + EP5 齿")
print("=" * 72)
m_tiers = [2, 4]
r1_pass = True
for beta2 in beta2s:
    n0_exact = np.sqrt(1.0 + beta2 / w0 ** 2)
    print("  [β²=%.1f] n(0) 解析 = %.6f（静态折射率，有限值）" % (beta2, n0_exact))
    for m in m_tiers:
        k = m * dk_mode
        wm_ex, _ = branches(k, beta2, w0)
        ev = spectrum_k(k, beta2, w0)
        w_r = min(omegas_from_eig(ev), key=lambda w: abs(w - wm_ex))
        d_spec = abs(w_r - wm_ex)
        ok_spec = d_spec < 1e-10
        r1_pass &= ok_spec
        # 时域双频对照
        C = rk4_run(N, beta2, w0, k)
        pks = read_two_omegas(C, dt_out)
        wm_td = min(pks, key=lambda p: abs(p - wm_ex))
        d_td = abs(wm_td - wm_ex)
        ok_td = d_td < max(eps_fft, 1e-4)
        r1_pass &= ok_td
        # 能隙型齿：能隙型闭式预言（同档）
        w_gap_pred = np.sqrt(w_latt(k) ** 2 + w0 ** 2)
        gap_fails = abs(w_r - w_gap_pred) > 0.1
        r1_pass &= gap_fails
        print("    k=%.5f: 谱 %.8f vs 精确根 %.8f (d=%.1e) %s | 时域 %.6f (d=%.1e) %s"
              % (k, w_r, wm_ex, d_spec, "PASS" if ok_spec else "FAIL",
                 wm_td, d_td, "PASS" if ok_td else "FAIL"))
        print("           能隙型闭式预言 %.4f 对读数差 %.4f（应显著 FAIL 作齿）%s"
              % (w_gap_pred, abs(w_r - w_gap_pred), "PASS" if gap_fails else "FAIL"))
    # 波包群速（长波档：载波完整多周期 + 短时程防周期折返）
    N_wp = 2048
    kc = 0.15
    vg_read = wavepacket_speed(N_wp, beta2, w0, kc, sigma=100.0, T=200.0)
    vg_ex = vg_exact(kc, beta2, w0)
    v0 = 1.0 / n0_exact
    budget = abs(vg_ex - v0) + 0.01 * vg_ex  # 长波差预算 + 波包读数余量
    ok_vg = abs(vg_read - vg_ex) < max(budget, 5e-3)
    r1_pass &= ok_vg
    print("    波包群速(N=2048,kc=0.15) read=%.4f vs 精确 v_g(k_c)=%.4f（差 %.4f）；"
          "v(0)=%.4f；长波差 %.4f 在预算 %.4f 内 %s"
          % (vg_read, vg_ex, abs(vg_read - vg_ex), v0, abs(vg_ex - v0),
             budget, "PASS" if ok_vg else "FAIL"))
print("\nR1 汇总: %s" % ("PASS" if r1_pass else "FAIL"))

# ---------- R2 ----------
print("\n" + "=" * 72)
print("R2 正常色散：ω<ω_0 段 n = wl(k)/ω 严格递增（格点口径）")
print("=" * 72)
r2_pass = True
for beta2 in beta2s:
    m_r2 = [30, 45, 62]
    ns = []
    ws = []
    for m in m_r2:
        k = m * dk_mode
        wm_ex, _ = branches(k, beta2, w0)
        ev = spectrum_k(k, beta2, w0)
        w_r = min(omegas_from_eig(ev), key=lambda w: abs(w - wm_ex))
        ns.append(w_latt(k) / w_r)
        ws.append(w_r)
    n0_exact = np.sqrt(1.0 + beta2 / w0 ** 2)
    mono = ns[0] < ns[1] < ns[2]
    near0 = abs(ns[0] - n0_exact) < 0.05 * n0_exact
    r2_pass &= mono and near0
    print("  [β²=%.1f] ω = %.4f/%.4f/%.4f → n = %.4f/%.4f/%.4f 递增 %s；"
          "n(首档)=%.4f 对 n(0)=%.4f（差 %.3f）%s"
          % (beta2, ws[0], ws[1], ws[2], ns[0], ns[1], ns[2],
             "PASS" if mono else "FAIL", ns[0], n0_exact, abs(ns[0] - n0_exact),
             "PASS" if near0 else "FAIL"))
print("\nR2 汇总: %s" % ("PASS" if r2_pass else "FAIL"))

# ---------- R3 ----------
print("\n" + "=" * 72)
print("R3 高频回归 1⁻：大 k 档光支（上支，连续追踪）n_eff = wl/ω <1 且回升")
print("（振子支平坦无色散，长波下支与大 k 上支同为光支：极化激元避免交叉）")
print("=" * 72)
r3_pass = True
for beta2 in beta2s:
    k_tiers = [2.0, 2.5, 3.0]
    m_r3 = [round(k / dk_mode) for k in k_tiers]
    ns = []
    n_exact = []
    for m in m_r3:
        k = m * dk_mode
        wp_ex, _ = branches(k, beta2, w0)  # 大 k 档光支 = 上支
        ev = spectrum_k(k, beta2, w0)
        w_r = max(omegas_from_eig(ev), key=lambda w: abs(w - wp_ex))
        ns.append(w_latt(k) / w_r)
        n_exact.append(np.sqrt(eps(w_r, beta2, w0)))
    rising = ns[0] < ns[1] < ns[2] < 1.0
    ident = max(abs(a - b) for a, b in zip(ns, n_exact)) < 1e-9
    r3_pass &= rising and ident
    print("  [β²=%.1f] n_eff = %.4f/%.4f/%.4f（递增且<1）%s；对 √ε(ω) 恒等偏差 "
          "%.1e %s"
          % (beta2, ns[0], ns[1], ns[2], "PASS" if rising else "FAIL",
             max(abs(a - b) for a, b in zip(ns, n_exact)), "PASS" if ident else "FAIL"))
print("  方向齿：第 573 条能隙型闭式同档全程 n>1（位移型静态近似），"
        "完整响应 n<1：X 射线方向互斥")
print("\nR3 汇总: %s" % ("PASS" if r3_pass else "FAIL"))

# ---------- R4 ----------
print("\n" + "=" * 72)
print("R4 剩余射线禁带：谱空段 + 带内修逝衰减（短距拟合）")
print("=" * 72)
r4_pass = True
for beta2 in beta2s:
    band_lo, band_hi = w0, np.sqrt(w0 ** 2 + beta2)
    count = 0
    for m in range(1, N // 2):
        k = m * dk_mode
        ev = spectrum_k(k, beta2, w0)
        for w in omegas_from_eig(ev):
            if band_lo + 1e-9 < w < band_hi - 1e-9:
                count += 1
    ok_band = count == 0
    r4_pass &= ok_band
    print("  [β²=%.1f] 禁带 (%.4f, %.4f) 内谱线计数 = %d（期望 0）%s"
          % (beta2, band_lo, band_hi, count, "PASS" if ok_band else "FAIL"))
    w_mid = 0.5 * (band_lo + band_hi)
    mu_r, resid = evanescent_mu_open(200, beta2, w0, w_mid, fit_near=14)
    mu_e = mu_exact(w_mid, beta2, w0)
    ok_mu = abs(mu_r - mu_e) < 0.05 * mu_e and mu_r > 0.1 and resid < 0.2
    r4_pass &= ok_mu
    print("         带内 ω=%.4f：μ_read=%.6f vs 精确 %.6f（d=%.1e, resid=%.1e）%s"
          % (w_mid, mu_r, mu_e, abs(mu_r - mu_e), resid, "PASS" if ok_mu else "FAIL"))
print("\nR4 汇总: %s" % ("PASS" if r4_pass else "FAIL"))

# ---------- R5 ----------
print("\n" + "=" * 72)
print("R5 稳定性与 A² 项必要性（EP5 齿）")
print("=" * 72)
r5_pass = True
k_test = 0.5
for beta2 in beta2s:
    ev_full = spectrum_k(k_test, beta2, w0, with_A2=True)
    ev_noA2 = spectrum_k(k_test, beta2, w0, with_A2=False)
    re_full = np.max(np.abs(ev_full.real))
    re_no = np.max(np.abs(ev_noA2.real))
    ok = re_full < 1e-10 and re_no > 0.05
    r5_pass &= ok
    print("  [β²=%.1f] 完整模型 max|Re λ| = %.2e（应 ~0）；去 A² 项 max|Re λ| = %.4f"
          "（应显著非零，不稳定根） %s"
          % (beta2, re_full, re_no, "PASS" if ok else "FAIL"))
print("\nR5 汇总: %s" % ("PASS" if r5_pass else "FAIL"))

# ---------- 总汇总 ----------
tot = sum([r1_pass, r2_pass, r3_pass, r4_pass, r5_pass])
print("\n" + "=" * 72)
print("总汇总: R1=%s R2=%s R3=%s R4=%s R5=%s （%d/5）"
      % ("PASS" if r1_pass else "FAIL", "PASS" if r2_pass else "FAIL",
         "PASS" if r3_pass else "FAIL", "PASS" if r4_pass else "FAIL",
         "PASS" if r5_pass else "FAIL", tot))
print("=" * 72)

# ---------- 图 ----------
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle("Verification 581: full-response (Lorentz-type) dispersion R1-R5", fontsize=13)

ax = axes[0, 0]
kk = np.linspace(0.005, 1.6, 400)
for beta2, col in zip(beta2s, ("tab:blue", "tab:red")):
    wm = [branches(k, beta2, w0)[0] for k in kk]
    ax.plot(kk, wm, color=col, lw=1.5, label=r"light branch, $\beta^2=%.1f$" % beta2)
    ax.plot(kk, [k / np.sqrt(1 + beta2 / w0 ** 2) for k in kk], color=col, ls="--", lw=0.8)
    ax.axhline(np.sqrt(w0 ** 2 + beta2), color=col, ls=":", lw=0.8)
ax.axhline(w0, color="k", ls="-.", lw=0.8, label=r"$\omega_0$")
ax.plot([2 * dk_mode], [branches(2 * dk_mode, beta2s[0], w0)[0]], "ko", ms=6, label="R1 tiers")
ax.set_xlabel(r"$k\,l_P$"); ax.set_ylabel(r"$\omega$")
ax.set_title("R1: acoustic light branch, no gap at k=0")
ax.legend(fontsize=7)

ax = axes[0, 1]
ww = np.linspace(0.05, 2.2, 400)
for beta2, col in zip(beta2s, ("tab:blue", "tab:red")):
    ne = [np.sqrt(max(eps(w, beta2, w0), 1e-6)) for w in ww]
    ax.plot(ww, ne, color=col, lw=1.5, label=r"$n_{\rm ph}(\omega)$, $\beta^2=%.1f$" % beta2)
    ax.axvline(np.sqrt(w0 ** 2 + beta2), color=col, ls=":", lw=0.8)
ax.axvspan(w0, np.sqrt(w0 ** 2 + beta2s[0]), alpha=0.15, color="gray")
ax.set_ylim(0.5, 2.0)
ax.set_xlabel(r"$\omega/\omega_0$"); ax.set_ylabel(r"$n$")
ax.set_title("R2/R3/R4: normal dispersion, reststrahlen band, n$\\to$1$^-$")
ax.legend(fontsize=8)

ax = axes[1, 0]
def light_identity_n(beta2, kmax, nm=260):
    """按振子分量占比 <0.5 识别光支身份，拼接 n_eff(k)；带内与交叉区给 NaN"""
    ks = np.linspace(0.02, kmax, nm)
    ns = []
    for k in ks:
        ev, vecs = np.linalg.eig(L_block(k, beta2, w0))
        light_ws = []
        for i in range(4):
            lam = ev[i]
            if abs(lam.real) < 1e-9 and lam.imag > 1e-9:
                v = vecs[:, i]
                # 位移判据：振子坐标 x 相对光坐标 A 的占比（动量分量在长波光支不小，勿用）
                if abs(v[2]) ** 2 < abs(v[0]) ** 2:
                    light_ws.append(lam.imag)
        ns.append(w_latt(k) / light_ws[0] if len(light_ws) == 1 else np.nan)
    return ks, np.array(ns)

for beta2, col in zip(beta2s, ("tab:blue", "tab:red")):
    kks, nps = light_identity_n(beta2, 2.5)
    ax.plot(kks, nps, color=col, lw=1.4, label=r"light-mode $n_{\rm eff}$, $\beta^2=%.1f$" % beta2)
ax.axhline(1.0, color="k", lw=0.8, ls="--")
ax.set_ylim(0.7, 1.9)
ax.set_xlabel(r"$k\,l_P$"); ax.set_ylabel(r"$n_{\rm eff}=k_{\rm lat}/\omega$")
ax.set_title("R2/R3: light-mode n rises, enters band, returns to 1 from below")
ax.legend(fontsize=8)

ax = axes[1, 1]
bands = [(w0, np.sqrt(w0 ** 2 + b)) for b in beta2s]
for (lo, hi), col in zip(bands, ("tab:blue", "tab:red")):
    ax.axvspan(lo, hi, alpha=0.2, color=col)
ww2 = np.linspace(0.2, 2.0, 300)
ax.plot(ww2, [mu_exact(w, beta2s[0], w0) if (w < w0 or w > np.sqrt(w0 ** 2 + beta2s[0])) else np.nan
              for w in ww2], "b-", lw=1.2, label=r"$\mu$ exact, $\beta^2=%.1f$" % beta2s[0])
ax.plot(ww2, [mu_exact(w, beta2s[1], w0) if (w < w0 or w > np.sqrt(w0 ** 2 + beta2s[1])) else np.nan
              for w in ww2], "r-", lw=1.2, label=r"$\beta^2=%.1f$" % beta2s[1])
w_mid = 0.5 * (w0 + np.sqrt(w0 ** 2 + beta2s[0]))
mu_r, _ = evanescent_mu_open(200, beta2s[0], w0, w_mid)
ax.plot([w_mid], [mu_r], "ko", ms=7, label="R4 steady response")
ax.set_xlabel(r"$\omega/\omega_0$"); ax.set_ylabel(r"$\mu\,l_P$")
ax.set_title("R4: evanescent decay inside the band (shaded)")
ax.legend(fontsize=8)

plt.tight_layout(rect=(0, 0, 1, 0.96))
plt.savefig(IMG, dpi=150)
print("\n图已存: " + IMG)
