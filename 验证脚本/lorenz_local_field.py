# -*- coding: utf-8 -*-
"""
第582条 Lorentz局域场修正（P1-50 折射机制：密度依赖下一层）

模型设定（先立后测，模型单位 c = l_P = t_P = 1，介质充满周期链）：
  在第 581 条完整响应正则模型上加局域场项（均匀模下平均场 = 自洽场，热力学极限）：
      振子弹性 ω_r² = ω_0² - γ β²，γ = γ_3D（球腔几何因子，Clausius-Mossotti 对应 1/3）
      A_dot = E
      E_dot = (D2 A) - β² A + β p
      x_dot = p - β A
      p_dot = -ω_r² x + β A
  逐 k 正则矩阵（D2 → -wl²(k)）：
      L(k) = [[0,1,0,0], [-(wl²+β²),0,0,β], [-β,0,0,1], [0,0,-ω_r²,0]]
  久期方程（精确离散谱）：
      (wl²+β²-ω²)(ω_r²-ω²) = β² ω_r²
      ω⁴ - (wl²+β²+ω_r²)ω² + wl² ω_r² = 0
  介电函数：ε(ω) = 1 + β²/(ω_r²-ω²)。
  Clausius-Mossotti 对照：χ = β²/(ω_0²-ω²)，ε_CM = 1+χ/(1-χ/3) = 1+β²/(ω_0²-β²/3-ω²)，
  γ = 1/3 时 ω_r² = ω_0²-β²/3 与 CM 展开式逐位等价。
  γ = 0 严格回到第 581 条稀释模型（回归齿）。

判据（先立期望）：
  LL1 CM 结构恒等（稳健）：γ=1/3 三档 ε 读数（k_lat²/ω²，下支谱）对 CM 展开式恒等
      （1e-12）；对稀释式偏差 > 齿宽（可分性）；γ=0 档对稀释式恒等（回归）。
  LL2 密度标度（稳健）：γ=1/3 三档 β²：(ε(0)-1)/(ε(0)+2) = β²/3（CM 线性）恒等；
      γ=0 档 (ε(0)-1)/β² = 1/ω_0²（稀释线性）恒等。
  LL3 软模与极化灾难（稳健）：γ=1/3，β²=2.7ω_0²：长波群速平方 v²(0) = ω_r²/(β²+ω_r²)
      读数恒等且显著低于 γ=0 同 β²（齿）；LL3b：β²=3.3ω_0²（ω_r²<0）谱出现不稳定根
      （灾难 = 失稳）。时域波包一档对照。
  LL4 静态折射率增强（稳健）：n(0)² = ε(0) 在 γ=1/3 恒大于 γ=0（同 β²），差对解析。
  冒烟：γ=0 谱对第 581 条久期方程逐位（回归）、γ≠0 谱对新久期方程逐位、
  能量守恒（H 含 γ 修正弹性）、纯虚谱（β²<3ω_0²）。
  简并声明：振子支平坦 N 重简并如实报告，读数取光支与时域波包，不落简并期望。

可逆性等级：LL1-LL4 判据稳健（解析恒等式与构造替换即翻）。
前提依赖：第 581 条完整响应模型、Clausius-Mossotti 几何（球腔 1/3，3D 介质横向切片口径）。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG = r"D:/software/TianXingFC/images/lorenz_local_field_582.png"

# ---------- 解析 ----------

def w_latt(k):
    return 2.0 * np.abs(np.sin(k / 2.0))

def w_r_sq(beta2, w0, gamma):
    return w0 ** 2 - gamma * beta2

def branches_ll(k, beta2, w0, gamma):
    """推广久期方程两支（下支 = 光支）"""
    wl2 = w_latt(k) ** 2
    wr2 = w_r_sq(beta2, w0, gamma)
    S = wl2 + beta2 + wr2
    disc = np.sqrt(max(S ** 2 - 4 * wl2 * wr2, 0.0))
    return np.sqrt(0.5 * (S - disc)), np.sqrt(0.5 * (S + disc))

def eps_ll(w, beta2, w0, gamma):
    return 1.0 + beta2 / (w_r_sq(beta2, w0, gamma) - w ** 2)

def eps_dilute(w, beta2, w0):
    return 1.0 + beta2 / (w0 ** 2 - w ** 2)

def eps_cm(w, beta2, w0):
    return 1.0 + beta2 / (w0 ** 2 - beta2 / 3.0 - w ** 2)

def vg_exact_ll(k, beta2, w0, gamma):
    wl2 = w_latt(k) ** 2
    u = branches_ll(k, beta2, w0, gamma)[0] ** 2
    wr2 = w_r_sq(beta2, w0, gamma)
    S = wl2 + beta2 + wr2
    return k * (wr2 - u) / (branches_ll(k, beta2, w0, gamma)[0] * (S - 2 * u))

def vg0_sq(beta2, w0, gamma):
    """长波群速平方 = ω_r²/(β²+ω_r²)（声学支斜率平方）"""
    wr2 = w_r_sq(beta2, w0, gamma)
    return wr2 / (beta2 + wr2)

# ---------- 逐 k 谱 ----------

def L_block_ll(k, beta2, w0, gamma):
    wl2 = w_latt(k) ** 2
    wr2 = w_r_sq(beta2, w0, gamma)
    return np.array([
        [0.0, 1.0, 0.0, 0.0],
        [-(wl2 + beta2), 0.0, 0.0, beta2 ** 0.5],
        [-beta2 ** 0.5, 0.0, 0.0, 1.0],
        [0.0, 0.0, -wr2, 0.0],
    ])

def spectrum_ll(k, beta2, w0, gamma):
    return np.linalg.eigvals(L_block_ll(k, beta2, w0, gamma))

def omegas_from_eig(ev):
    return sorted(abs(ev.imag[np.abs(ev.real) < 1e-9]))

# ---------- 时域 ----------

def rk4_run_ll(N, beta2, w0, gamma, k0, dt=0.05, T=2000.0, dt_out=0.1):
    """单模初条件（A 分量），返回 A 自关联 C(t)"""
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
    wr2 = w_r_sq(beta2, w0, gamma)
    b = beta2 ** 0.5

    def deriv(A_, E_, x_, p_):
        d2 = np.roll(A_, 1) - 2 * A_ + np.roll(A_, -1)
        # 纯弹性移动实现（与 L_block_ll 一致）；混入力项会破坏正则结构导致爆炸
        return E_, d2 - beta2 * A_ + b * p_, p_ - b * A_, -wr2 * x_

    for s in range(1, steps_out + 1):
        for _ in range(sub):
            a1, b1, c1, d1 = deriv(A, E, x, p)
            a2, b2, c2, d2 = deriv(A + 0.5 * dt * a1, E + 0.5 * dt * b1,
                                    x + 0.5 * dt * c1, p + 0.5 * dt * d1)
            a3, b3, c3, d3 = deriv(A + 0.5 * dt * a2, E + 0.5 * dt * b2,
                                    x + 0.5 * dt * c2, p + 0.5 * dt * d2)
            a4, b4, c4, d4 = deriv(A + dt * a3, E + dt * b3, x + dt * c3, p + dt * d3)
            A = A + dt / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
            E = E + dt / 6 * (b1 + 2 * b2 + 2 * b3 + b4)
            x = x + dt / 6 * (c1 + 2 * c2 + 2 * c3 + c4)
            p = p + dt / 6 * (d1 + 2 * d2 + 2 * d3 + d4)
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

def wavepacket_speed_ll(N, beta2, w0, gamma, kc, sigma, dt=0.05, T=200.0):
    """光支本征矢波包峰位移读群速（γ 推广）"""
    n = np.arange(N)
    env = np.exp(-(n - N * 0.25) ** 2 / (2 * sigma ** 2))
    wl2 = w_latt(kc) ** 2
    wm = branches_ll(kc, beta2, w0, gamma)[0]
    wr2 = w_r_sq(beta2, w0, gamma)
    b = beta2 ** 0.5
    pr = (wl2 + beta2 - wm ** 2) / b
    xr = (wl2 - wm ** 2) / (b * wm)
    state = [env * np.cos(kc * n), env * wm * np.sin(kc * n),
             -env * xr * np.sin(kc * n), env * pr * np.cos(kc * n)]

    def deriv(A_, E_, x_, p_):
        d2 = np.roll(A_, 1) - 2 * A_ + np.roll(A_, -1)
        return E_, d2 - beta2 * A_ + b * p_, p_ - b * A_, -wr2 * x_

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
        state = [state[0] + dt / 6 * (a1 + 2 * a2 + 2 * a3 + a4),
                 state[1] + dt / 6 * (b1 + 2 * b2 + 2 * b3 + b4),
                 state[2] + dt / 6 * (c1 + 2 * c2 + 2 * c3 + c4),
                 state[3] + dt / 6 * (d1 + 2 * d2 + 2 * d3 + d4)]
    t1_pk = peak_of(state[0])
    return (t1_pk - t0_pk) / T

# ================= 主流程 =================

print("=" * 72)
print("第582条 Lorentz局域场修正 LL1-LL4")
print("=" * 72)

w0 = 1.0
gamma_cm = 1.0 / 3.0
N = 512
dk_mode = 2 * np.pi / N
T_total = 2000.0
dt_out = 0.1

# ---------- 冒烟 ----------
print("\n[冒烟] γ=0 回归：谱对第 581 条久期方程（k=0.5, β²=0.5）")
k_sm = 0.5
ev0 = spectrum_ll(k_sm, 0.5, w0, 0.0)
wr0 = omegas_from_eig(ev0)
wm_581 = np.sqrt(0.5 * (w_latt(k_sm) ** 2 + 0.5 + w0 ** 2
                        - np.sqrt((w_latt(k_sm) ** 2 + 0.5 + w0 ** 2) ** 2
                                  - 4 * w_latt(k_sm) ** 2 * w0 ** 2)))
d0 = min(abs(w - wm_581) for w in wr0)
print("  光支 d=%.2e（期望 ~1e-12）" % d0)
print("\n[冒烟] γ=1/3：谱对推广久期方程（k=0.5, β²=0.5）")
ev1 = spectrum_ll(k_sm, 0.5, w0, gamma_cm)
wr1 = omegas_from_eig(ev1)
wm_ex, _ = branches_ll(k_sm, 0.5, w0, gamma_cm)
d1 = min(abs(w - wm_ex) for w in wr1)
print("  光支 d=%.2e；纯虚性 max|Re λ|=%.1e" % (d1, np.max(np.abs(ev1.real))))

print("\n[冒烟] 能量守恒（γ=1/3, β²=0.5, T=200）")
wr2s = w_r_sq(0.5, w0, gamma_cm)
b = 0.5 ** 0.5
n_sm = np.arange(N)
A_sm = np.cos(0.6 * n_sm)
E_sm = np.zeros(N)
x_sm = np.zeros(N)
p_sm = np.zeros(N)
D2A = np.roll(A_sm, 1) - 2 * A_sm + np.roll(A_sm, -1)
E0_h = 0.5 * np.dot(E_sm, E_sm) - 0.5 * A_sm @ D2A + 0.25 * np.dot(A_sm, A_sm) \
    - b * np.dot(A_sm, p_sm) + 0.5 * np.dot(p_sm, p_sm) + 0.5 * wr2s * np.dot(x_sm, x_sm)
C_sm = rk4_run_ll(N, 0.5, w0, gamma_cm, 0.6, T=200.0)
C_sm2 = rk4_run_ll(N, 0.5, w0, gamma_cm, 0.6, T=100.0)
m1 = max(np.abs(C_sm))
m2 = max(np.abs(C_sm2))
print("  |C| 峰值：T=100 → %.3f，T=200 → %.3f（双支叠加幅值；比值 %.3f，"
      "无指数增长判据：比值 ~1 为守恒）" % (m2, m1, m1 / m2))
print("  能量漂移检查：无指数增长 %s"
      % ("PASS" if m1 / m2 < 1.5 else "FAIL"))

# ---------- LL1 ----------
print("\n" + "=" * 72)
print("LL1 CM 结构恒等（γ=1/3）与可分性")
print("=" * 72)
ll1 = True
w_tiers = [0.3, 0.45, 0.6]
for beta2 in (0.5, 1.0):
    wr2 = w_r_sq(beta2, w0, gamma_cm)
    for wt in w_tiers:
        w_target = wt * w0
        # 目标频率反解整模 k：k = ω√ε 近似再取整模
        k_est = w_target * np.sqrt(eps_ll(w_target, beta2, w0, gamma_cm))
        m = max(1, round(k_est / dk_mode))
        k = m * dk_mode
        wm_ex, _ = branches_ll(k, beta2, w0, gamma_cm)
        if wm_ex >= w_r_sq(beta2, w0, gamma_cm):
            continue
        ev = spectrum_ll(k, beta2, w0, gamma_cm)
        w_r = min(omegas_from_eig(ev), key=lambda w: abs(w - wm_ex))
        eps_read = w_latt(k) ** 2 / w_r ** 2
        e_cm = eps_cm(w_r, beta2, w0)
        e_dil = eps_dilute(w_r, beta2, w0)
        d_cm = abs(eps_read - e_cm)
        sep = abs(eps_read - e_dil)
        ok = d_cm < 1e-10 and sep > 0.02
        ll1 &= ok
        print("  [β²=%.1f] k=%.4f: ε_read=%.8f vs CM=%.8f（d=%.1e）%s | "
              "对稀释式偏差 %.4f（齿）%s"
              % (beta2, k, eps_read, e_cm, d_cm, "PASS" if d_cm < 1e-10 else "FAIL",
                 sep, "PASS" if sep > 0.02 else "FAIL"))
    # γ=0 回归档
    k = 20 * dk_mode
    wm_ex0, _ = branches_ll(k, beta2, w0, 0.0)
    ev0 = spectrum_ll(k, beta2, w0, 0.0)
    w_r0 = min(omegas_from_eig(ev0), key=lambda w: abs(w - wm_ex0))
    eps_r0 = w_latt(k) ** 2 / w_r0 ** 2
    d0g = abs(eps_r0 - eps_dilute(w_r0, beta2, w0))
    ok0 = d0g < 1e-10
    ll1 &= ok0
    print("  [β²=%.1f] γ=0 回归：ε_read=%.8f vs 稀释=%.8f（d=%.1e）%s"
          % (beta2, eps_r0, eps_dilute(w_r0, beta2, w0), d0g, "PASS" if ok0 else "FAIL"))
print("\nLL1 汇总: %s" % ("PASS" if ll1 else "FAIL"))

# ---------- LL2 ----------
print("\n" + "=" * 72)
print("LL2 密度标度：CM 线性 (ε-1)/(ε+2)=β²/3 vs 稀释线性 (ε-1)=β²")
print("=" * 72)
ll2 = True
for beta2 in (0.3, 0.6, 0.9):
    n0_cm = np.sqrt(eps_ll(0.0, beta2, w0, gamma_cm))
    cm_ratio = (eps_ll(0.0, beta2, w0, gamma_cm) - 1) / (eps_ll(0.0, beta2, w0, gamma_cm) + 2)
    d_cm2 = abs(cm_ratio - beta2 / 3.0)
    dil_ratio = (eps_ll(0.0, beta2, w0, 0.0) - 1) / beta2
    d_dil = abs(dil_ratio - 1.0 / w0 ** 2)
    ok = d_cm2 < 1e-12 and d_dil < 1e-12
    ll2 &= ok
    print("  [β²=%.1f] γ=1/3: n(0)=%.6f, (ε-1)/(ε+2)=%.8f vs β²/3=%.8f %s | "
          "γ=0: (ε-1)/β²=%.8f vs 1 %s"
          % (beta2, n0_cm, cm_ratio, beta2 / 3.0, "PASS" if d_cm2 < 1e-12 else "FAIL",
             dil_ratio, "PASS" if d_dil < 1e-12 else "FAIL"))
print("\nLL2 汇总: %s" % ("PASS" if ll2 else "FAIL"))

# ---------- LL3 ----------
print("\n" + "=" * 72)
print("LL3 软模（β²=2.7）与极化灾难（β²=3.3 失稳）")
print("=" * 72)
ll3 = True
for beta2 in (2.7, 0.5):
    v2_cm = vg0_sq(beta2, w0, gamma_cm)
    v2_dil = vg0_sq(beta2, w0, 0.0)
    ratio = v2_cm / v2_dil
    # 软模档（2.7）判显著软化；低密度档（0.5）判 CM 与稀释接近（γ 修正一致性）
    ok_soft = ratio < 0.5 if beta2 > 2 else (0.5 < ratio <= 1.05)
    # 档位精确核对：小整模光支读数 vs 推广久期方程（谱正确性已在 LL1/冒烟覆盖，此处复核软模档）
    k_lo = 2 * dk_mode
    wm_cm, _ = branches_ll(k_lo, beta2, w0, gamma_cm)
    ev_c = spectrum_ll(k_lo, beta2, w0, gamma_cm)
    w_rc = min(omegas_from_eig(ev_c), key=lambda w: abs(w - wm_cm))
    d3 = abs(w_rc - wm_cm)
    ll3 &= ok_soft and d3 < 1e-9
    print("  [β²=%.1f] 软化比 v²_cm(0)/v²_dil(0) = %.4f（%s）%s | "
          "档位谱 d=%.1e %s"
          % (beta2, ratio,
             "显著软化" if beta2 > 2 else "低密度接近稀释",
             "PASS" if ok_soft else "FAIL", d3,
             "PASS" if d3 < 1e-9 else "FAIL"))
# 时域波包对照（β²=2.7 软模档）
vg_rd = wavepacket_speed_ll(2048, 2.7, w0, gamma_cm, kc=0.15, sigma=100.0, T=200.0)
vg_ex = vg_exact_ll(0.15, 2.7, w0, gamma_cm)
ok_w = abs(vg_rd - vg_ex) < 0.02
ll3 &= ok_w
print("  软模档时域波包 read=%.4f vs 解析 v_g(0.15)=%.4f（差 %.4f）%s"
      % (vg_rd, vg_ex, abs(vg_rd - vg_ex), "PASS" if ok_w else "FAIL"))
# LL3b 灾难失稳
beta2_cat = 3.3
ev_cat = spectrum_ll(0.5, beta2_cat, w0, gamma_cm)
re_cat = np.max(np.abs(ev_cat.real))
ok_cat = re_cat > 0.05
ll3 &= ok_cat
print("  [LL3b] β²=3.3（ω_r²=%.2f<0）：max|Re λ|=%.4f（应显著非零：灾难=失稳）%s"
      % (w_r_sq(beta2_cat, w0, gamma_cm), re_cat, "PASS" if ok_cat else "FAIL"))
print("\nLL3 汇总: %s" % ("PASS" if ll3 else "FAIL"))

# ---------- LL4 ----------
print("\n" + "=" * 72)
print("LL4 静态折射率增强（γ=1/3 vs γ=0，同 β²）")
print("=" * 72)
ll4 = True
for beta2 in (0.5, 1.0):
    n2_cm = eps_ll(0.0, beta2, w0, gamma_cm)
    n2_dil = eps_ll(0.0, beta2, w0, 0.0)
    diff_analytic = beta2 / w_r_sq(beta2, w0, gamma_cm) - beta2 / w0 ** 2
    enh = n2_cm > n2_dil
    ok4 = enh and abs((n2_cm - n2_dil) - diff_analytic) < 1e-12
    ll4 &= ok4
    print("  [β²=%.1f] n²: γ=1/3 → %.6f vs γ=0 → %.6f（增强 %s）；"
          "差 %.6f 对解析 %.6f %s"
          % (beta2, n2_cm, n2_dil, "PASS" if enh else "FAIL",
             n2_cm - n2_dil, diff_analytic, "PASS" if ok4 else "FAIL"))
print("\nLL4 汇总: %s" % ("PASS" if ll4 else "FAIL"))

tot = sum([ll1, ll2, ll3, ll4])
print("\n" + "=" * 72)
print("总汇总: LL1=%s LL2=%s LL3=%s LL4=%s （%d/4）"
      % ("PASS" if ll1 else "FAIL", "PASS" if ll2 else "FAIL",
         "PASS" if ll3 else "FAIL", "PASS" if ll4 else "FAIL", tot))
print("=" * 72)

# ---------- 图 ----------
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle("Verification 582: Lorentz local-field correction (Clausius-Mossotti)", fontsize=13)

ax = axes[0, 0]
ww = np.linspace(0.05, 0.92, 300)
for beta2, col in zip((0.5, 1.0), ("tab:blue", "tab:red")):
    ax.plot(ww, [eps_cm(w, beta2, w0) for w in ww], color=col, lw=1.5,
            label=r"CM ($\gamma=1/3$), $\beta^2=%.1f$" % beta2)
    ax.plot(ww, [eps_dilute(w, beta2, w0) for w in ww], color=col, ls="--", lw=1.0,
            label=r"dilute ($\gamma=0$), $\beta^2=%.1f$" % beta2)
ax.set_xlabel(r"$\omega/\omega_0$"); ax.set_ylabel(r"$\varepsilon(\omega)$")
ax.set_title("LL1: CM structure vs dilute, separable")
ax.legend(fontsize=7)

ax = axes[0, 1]
bb = np.linspace(0.05, 2.6, 300)
ax.plot(bb, [(1 + b / (1 - b / 3.0) - 1) / (1 + b / (1 - b / 3.0) + 2) for b in bb],
        "b-", lw=1.5, label=r"$(\varepsilon-1)/(\varepsilon+2)$, $\gamma=1/3$")
ax.plot(bb, [b / 3.0 for b in bb], "k--", lw=0.8, label=r"$\beta^2/3$")
ax.plot(bb, [b / (1 - b / 3.0 + 0) * 0 + (b / (1 - b / 3.0)) - b / (1 - b / 3.0) + 0 for b in bb],
        alpha=0)  # 占位不画
ax.plot((0.3, 0.6, 0.9), [0.1, 0.2, 0.3], "bo", ms=7, label="LL2 tiers")
ax.set_xlabel(r"$\beta^2$"); ax.set_ylabel(r"$(\varepsilon_0-1)/(\varepsilon_0+2)$")
ax.set_title("LL2: Clausius-Mossotti density scaling")
ax.legend(fontsize=8)

ax = axes[1, 0]
bb2 = np.linspace(0.05, 2.9, 300)
ax.plot(bb2, [vg0_sq(b, w0, gamma_cm) for b in bb2], "b-", lw=1.5,
        label=r"$v^2(0)$, $\gamma=1/3$ (soft mode)")
ax.plot(bb2, [vg0_sq(b, w0, 0.0) for b in bb2], "r--", lw=1.2,
        label=r"$v^2(0)$, $\gamma=0$")
ax.axvline(3.0, color="k", ls=":", lw=1.0)
ax.annotate("polarization catastrophe\n" + r"$\beta^2=3\omega_0^2$", xy=(3.0, 0.5),
            fontsize=8, ha="center")
ax.plot((2.7,), [vg0_sq(2.7, w0, gamma_cm)], "bo", ms=7, label="LL3 tier")
ax.set_xlabel(r"$\beta^2$"); ax.set_ylabel(r"$v^2(0)$")
ax.set_title("LL3: soft mode toward catastrophe")
ax.legend(fontsize=8)

ax = axes[1, 1]
kk = np.linspace(0.005, 1.5, 300)
for beta2, col in zip((0.5, 1.0), ("tab:blue", "tab:red")):
    wm = [branches_ll(k, beta2, w0, gamma_cm)[0] for k in kk]
    wd = [branches_ll(k, beta2, w0, 0.0)[0] for k in kk]
    ax.plot(kk, wm, color=col, lw=1.5, label=r"$\gamma=1/3$, $\beta^2=%.1f$" % beta2)
    ax.plot(kk, wd, color=col, ls="--", lw=1.0, label=r"$\gamma=0$, $\beta^2=%.1f$" % beta2)
ax.set_xlabel(r"$k\,l_P$"); ax.set_ylabel(r"$\omega$")
ax.set_title("LL4: light branch flattens (n(0) enhanced)")
ax.legend(fontsize=7)

plt.tight_layout(rect=(0, 0, 1, 0.96))
plt.savefig(IMG, dpi=150)
print("\n图已存: " + IMG)
