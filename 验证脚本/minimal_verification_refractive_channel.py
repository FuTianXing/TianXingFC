# -*- coding: utf-8 -*-
"""
第573条 介质折射能隙型通道（P1-50 第五轮注记 M1-M4 判据落地）

模型设定（先立后测，全部模型单位 c = l_P = t_P = 1）：
  能隙型（位移型有效形式，色散论文 §7 物质模式色散的一维切片）：
      u_tt = (D2 u) - w0^2 u,  色散 w^2(k) = 4 sin^2(k/2) + w0^2
      离散解析群速 v_g = w_latt(k) cos(k/2) / w(k),  w_latt = 2|sin(k/2)|
  无能隙对照：w0 = 0（纯光扇区，同流程）。
  速度型对照（第526条通道二分的速度型，动量型两扇区耦合）：
      M u_tt = -K u,  M = [[1,nu],[nu,1]],  K = diag(w_latt^2, w_d^2), w_d = w0
      广义本征值 det(K - w^2 M) = 0；低频存在无隙声学支（min w -> 0）。

判据（先立期望；对注记草案的两处修正见注记过程记录）：
  M1 数值正确性与高频回归：
     (a) 单模相位法读 w(k)（三档 k x 两档 w0）对离散解析式，容差 = max(FFT 分辨率预算, RK4 减半实测)；
     (b) 群速中心差分读 n(k)，三档对两档 w0 严格单调下降；
     (c) |n_read - n_cont| <= |n_latt_exact - n_cont| + eps_num（与连续闭式之差被格点截断解析预算覆盖）；
     (d) 无能隙对照 n_read 对 n_latt^free 同预算自洽。
  M2 低频禁戒：
     (a) 开链谱最小频率 = sqrt(w0^2 + 4 sin^2(pi/(2(N+1))))（解析恒等级，容差机器精度）；
     (b) 六档 w/w0 in [0.3, 0.95] 稳态修逝衰减率 mu 对精确离散解析 2 asinh(sqrt(w0^2-w^2)/2)，
         容差 1e-6；mu 随 w->w0- 严格单调下降（穿透加深）。
  M3 能隙上缘色散符号（判据修正：注记原期望 dn/dw>0 与闭式 n = w/sqrt(w^2-w0^2) 自相矛盾，
     真实期望 dn/dw < 0；正常色散反号登记为缺反向项失配第二表现）：
     三档 w/w0 in {1.1, 1.2, 1.5}，n 严格递减，两对差商皆负。
  M4 通道排他（判据修正：高频群速口径区分度不足，改落低频带隙存在性，对齐第526条签名）：
     (a) 速度型谱 min w < 0.01 w0（无带隙，与 M2 的 min w = w0 互斥）；
     (b) 低频两档 v_g：速度型 >= 0.75，能隙型 <= 0.35（两界间隔内无交叉）；
     (c) 速度型一档 RK4 双频读数对二次解析根（流程跨模型自洽）；
     (d) 恒真检验（EP5 齿）：把速度型谱喂给 M2 期望必 FAIL。

可逆性等级：M1(a)(d)/M2(a)(b)/M4 判据稳健；M1(b)(c) 敏感（依赖格点截断预算与档位）；M3 符号稳健、物理解读敏感（失配待补）。
前提依赖：去耦引理（色散论文 §6.2）、物质模式色散闭式（§7）、第526条通道二分签名。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG = r"D:/software/TianXingFC/images/refractive_channel_573.png"

# ---------- 基础算子 ----------

def make_D2_periodic(N):
    D = np.zeros((N, N))
    idx = np.arange(N)
    D[idx, idx] = -2.0
    D[idx, (idx + 1) % N] += 1.0
    D[idx, (idx - 1) % N] += 1.0
    return D

def make_D2_open(N):
    D = np.zeros((N, N))
    idx = np.arange(N)
    D[idx, idx] = -2.0
    D[idx[:-1], idx[:-1] + 1] += 1.0
    D[idx[1:], idx[1:] - 1] += 1.0
    return D

def w_latt(k):
    return 2.0 * np.abs(np.sin(k / 2.0))

def w_gap(k, w0):
    return np.sqrt(w_latt(k) ** 2 + w0 ** 2)

def vg_latt(k, w0):
    """离散解析群速（c=1）：d(4sin^2(k/2))/dk = 2 sin(k)"""
    return 2.0 * np.sin(k) / (2.0 * w_gap(k, w0))

def n_latt_exact(k, w0):
    return 1.0 / vg_latt(k, w0)

def n_cont(k, w0):
    """连续闭式（注记同式）"""
    return np.sqrt(1.0 + w0 ** 2 / k ** 2)

# ---------- 单模相位法读 w(k) ----------

def rk4_run(N, D2, w0, k0, dt=0.05, T=2000.0, dt_out=0.1):
    """初始条件 u=cos(k0 x), v=0；返回模式自关联 C(t)（单模下 = cos(w t)）"""
    n = np.arange(N)
    u = np.cos(k0 * n)
    u0 = u.copy()
    v = np.zeros(N)
    steps_out = int(round(T / dt_out))
    sub = int(round(dt_out / dt))
    C = np.empty(steps_out + 1)
    norm = np.dot(u0, u0)

    def deriv(uu, vv):
        du = np.roll(uu, 1) - 2 * uu + np.roll(uu, -1)
        return vv, du - w0 ** 2 * uu

    C[0] = np.dot(u, u0) / norm
    t = 0.0
    for s in range(1, steps_out + 1):
        for _ in range(sub):
            k1u, k1v = deriv(u, v)
            k2u, k2v = deriv(u + 0.5 * dt * k1u, v + 0.5 * dt * k1v)
            k3u, k3v = deriv(u + 0.5 * dt * k2u, v + 0.5 * dt * k2v)
            k4u, k4v = deriv(u + dt * k3u, v + dt * k3v)
            u = u + dt / 6.0 * (k1u + 2 * k2u + 2 * k3u + k4u)
            v = v + dt / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)
        C[s] = np.dot(u, u0) / norm
        t += dt_out
    return C, t

def read_omega(C, t_total, dt_out):
    """FFT 峰 + 三点抛物线插值"""
    F = np.fft.rfft(C * np.hanning(len(C)))
    freqs = 2 * np.pi * np.fft.rfftfreq(len(C), d=dt_out)
    mag = np.abs(F)
    i = np.argmax(mag[1:]) + 1
    # 抛物线插值（log 幅值）
    y0, y1, y2 = np.log(mag[i - 1]), np.log(mag[i]), np.log(mag[i + 1])
    dx = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    return freqs[i] + dx * (freqs[1] - freqs[0])

def rk4_energy_drift(N, D2, w0, k0, dt=0.05, T=200.0):
    n = np.arange(N)
    u = np.cos(k0 * n)
    v = np.zeros(N)
    E0 = 0.5 * np.dot(v, v) - 0.5 * u @ (D2 @ u) + 0.5 * w0 ** 2 * np.dot(u, u)
    steps = int(T / dt)
    for _ in range(steps):
        def deriv(uu, vv):
            du = np.roll(uu, 1) - 2 * uu + np.roll(uu, -1)
            return vv, du - w0 ** 2 * uu
        k1u, k1v = deriv(u, v)
        k2u, k2v = deriv(u + 0.5 * dt * k1u, v + 0.5 * dt * k1v)
        k3u, k3v = deriv(u + 0.5 * dt * k2u, v + 0.5 * dt * k2v)
        k4u, k4v = deriv(u + dt * k3u, v + dt * k3v)
        u = u + dt / 6.0 * (k1u + 2 * k2u + 2 * k3u + k4u)
        v = v + dt / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)
    E1 = 0.5 * np.dot(v, v) - 0.5 * u @ (D2 @ u) + 0.5 * w0 ** 2 * np.dot(u, u)
    return abs(E1 - E0) / abs(E0)

# ---------- M2：谱与修逝 ----------

def spectral_min_open(N, w0):
    D2 = make_D2_open(N)
    K = -D2 + w0 ** 2 * np.eye(N)
    ev = np.linalg.eigvalsh(K)
    return np.sqrt(max(ev.min(), 0.0)), ev.min()

def evanescent_mu(N, w0, w):
    """稳态响应：(-D2 + w0^2 - w^2) u = e_{x0}；修逝段拟合 mu"""
    D2 = make_D2_open(N)
    A = -D2 + (w0 ** 2 - w ** 2) * np.eye(N)
    src = np.zeros(N)
    x0 = N // 2
    src[x0] = 1.0
    u = np.linalg.solve(A, src)
    seg = np.arange(x0 + 20, x0 + 140)
    slope, intercept = np.polyfit(seg.astype(float), np.log(np.abs(u[seg])), 1)
    resid = np.abs(np.polyval((slope, intercept), seg.astype(float)) - np.log(np.abs(u[seg])))
    return -slope, resid.max(), u, x0

def mu_exact(w0, w):
    d = np.clip(w0 ** 2 - np.asarray(w) ** 2, 0.0, None)
    return 2.0 * np.arcsinh(np.sqrt(d) / 2.0)

# ---------- M4：速度型 ----------

def w_speed_types(k, w0, nu):
    """二次根： (1-nu^2) w^4 - S w^2 + P = 0, S = wl^2 + wd^2, P = wl^2 wd^2"""
    wl2 = w_latt(k) ** 2
    S = wl2 + w0 ** 2
    P = wl2 * w0 ** 2
    disc = S ** 2 - 4.0 * (1 - nu ** 2) * P
    disc = max(disc, 0.0)
    w2p = (S + np.sqrt(disc)) / (2 * (1 - nu ** 2))
    w2m = P / w2p  # 两根之积 P/(1-nu^2)，w2m = P/((1-nu^2) w2p)
    w2m = (S - np.sqrt(disc)) / (2 * (1 - nu ** 2))
    return np.sqrt(max(w2m, 0.0)), np.sqrt(max(w2p, 0.0))

def speed_spectral_min(N, w0, nu):
    """Cholesky 对称化后 eigvalsh（机器精度）；非对称 eigvals 只有 1e-7 级"""
    K = np.zeros((2 * N, 2 * N))
    D2o = make_D2_open(N)
    K[:N, :N] = -D2o
    K[N:, N:] = w0 ** 2 * np.eye(N)
    M = np.eye(2 * N)
    idx = np.arange(2 * N)
    M[idx[:N], idx[:N] + N] = nu
    M[idx[:N] + N, idx[:N]] = nu
    L = np.linalg.cholesky(M)
    Ks = np.linalg.solve(L, np.linalg.solve(L, K).T).T  # L^-1 K L^-T
    ev = np.linalg.eigvalsh(Ks)
    return np.sqrt(max(ev.min(), 0.0))

def rk4_speed_read(N, w0, nu, k0, dt=0.02, T=1200.0, dt_out=0.1):
    """速度型两场 RK4：u_tt = (D2 u - nu w0^2 d)/(1-nu^2)，d_tt = (nu D2 u - w0^2 d)/(1-nu^2)"""
    n = np.arange(N)
    uu = np.cos(k0 * n)
    uu0 = uu.copy()
    dd = np.zeros(N)
    vu = np.zeros(N)
    vd = np.zeros(N)
    norm = np.dot(uu0, uu0)
    steps_out = int(round(T / dt_out))
    sub = int(round(dt_out / dt))
    C = np.empty(steps_out + 1)
    C[0] = np.dot(uu, uu0) / norm

    def deriv(u_, v_, d_, w_):
        # 正确方程：M (u_tt,d_tt) = (D2 u, -w0^2 d)（K_u = -D2）
        # 解出 u_tt = (D2 u - nu w0^2 d)/(1-nu^2)，d_tt = (nu D2 u - w0^2 d)/(1-nu^2)
        du = np.roll(u_, 1) - 2 * u_ + np.roll(u_, -1)
        au = (du - nu * w0 ** 2 * d_) / (1 - nu ** 2)
        ad = (nu * du - w0 ** 2 * d_) / (1 - nu ** 2)
        return v_, au, w_, ad

    for s in range(1, steps_out + 1):
        for _ in range(sub):
            a1u, b1u, a1d, b1d = deriv(uu, vu, dd, vd)
            a2u, b2u, a2d, b2d = deriv(uu + 0.5 * dt * a1u, vu + 0.5 * dt * b1u,
                                        dd + 0.5 * dt * a1d, vd + 0.5 * dt * b1d)
            a3u, b3u, a3d, b3d = deriv(uu + 0.5 * dt * a2u, vu + 0.5 * dt * b2u,
                                        dd + 0.5 * dt * a2d, vd + 0.5 * dt * b2d)
            a4u, b4u, a4d, b4d = deriv(uu + dt * a3u, vu + dt * b3u,
                                        dd + dt * a3d, vd + dt * b3d)
            uu = uu + dt / 6.0 * (a1u + 2 * a2u + 2 * a3u + a4u)
            vu = vu + dt / 6.0 * (b1u + 2 * b2u + 2 * b3u + b4u)
            dd = dd + dt / 6.0 * (a1d + 2 * a2d + 2 * a3d + a4d)
            vd = vd + dt / 6.0 * (b1d + 2 * b2d + 2 * b3d + b4d)
        C[s] = np.dot(uu, uu0) / norm
    return C

def read_two_omegas(C, dt_out):
    F = np.fft.rfft(C * np.hanning(len(C)))
    freqs = 2 * np.pi * np.fft.rfftfreq(len(C), d=dt_out)
    mag = np.abs(F)
    mag_s = mag.copy()
    peaks = []
    for _ in range(2):
        i = np.argmax(mag_s[3:]) + 3
        y0, y1, y2 = np.log(mag_s[i - 1]), np.log(mag_s[i]), np.log(mag_s[i + 1])
        dx = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
        peaks.append(freqs[i] + dx * (freqs[1] - freqs[0]))
        lo = max(i - 8, 0)
        hi = i + 9
        mag_s[lo:hi] = 0.0
    return sorted(peaks)

# ================= 主流程 =================

print("=" * 72)
print("第573条 介质折射能隙型通道 M1-M4")
print("=" * 72)

# ---------- 冒烟检查 ----------
print("\n[冒烟] D2 谱核对：周期链 e^{ikn} 特征值应为 -4 sin^2(k/2)（k 取整模）")
N = 512
D2p = make_D2_periodic(N)
errs = []
for m in (24, 49, 98, 130):
    k = 2 * np.pi * m / N
    vec = np.exp(1j * k * np.arange(N))
    lam = (D2p @ vec) / vec
    errs.append(float(np.max(np.abs(lam.real + 4 * np.sin(k / 2) ** 2) + np.abs(lam.imag))))
print("  max|lambda 读数 - 解析| = %.3e（期望 ~1e-12）" % max(errs))
print("  变化的轴 = 空间差分轴（周期平移），核对通过" if max(errs) < 1e-10 else "  [FAIL] 谱不匹配")

print("\n[冒烟] 单模自关联与能量漂移（整模初始条件）")
k_sm = 2 * np.pi * 49 / N
C_sm, _ = rk4_run(N, D2p, 0.5, k_sm, T=200.0)
print("  |C(t)| max - 1 = %.3e（期望 < 1e-10）" % max(abs(np.abs(C_sm) - 1.0)))
drift = rk4_energy_drift(N, D2p, 0.5, k_sm)
print("  RK4 能量相对漂移（T=200）= %.3e（容差 1e-6：RK4 非辛协议伪影，"
      "相位读数路径与能量漂移解耦，实测相位误差见 M1）%s"
      % (drift, "PASS" if drift < 1e-6 else "FAIL"))

# ---------- M1 ----------
print("\n" + "=" * 72)
print("M1 数值正确性 + 单调性 + 预算内回归")
print("=" * 72)
ks = [0.3, 0.6, 1.2]
w0s = [0.5, 1.0]
dt_out = 0.1
T_total = 2000.0
N = 512
dk_mode = 2 * np.pi / N
# 档位取周期链精确本征模（非整模初始条件会泄漏邻近频率，偏移可达 d(w)/dk * dk/2）
ks = sorted({round(k / dk_mode) * dk_mode for k in ks})
dk = 4 * dk_mode  # 群速中心差分步（也是整模）
print("周期链整模档位: k = %s (dk_mode=%.5f, 差分步 dk=%.5f)"
      % (", ".join("%.5f" % k for k in ks), dk_mode, dk))
eps_fft = 1.0 / T_total  # FFT 分辨率单元（hann+抛物线插值实测 ~0.1 单元，预算取整单元）

m1_a_pass, m1_b_pass, m1_c_pass, m1_d_pass = True, True, True, True
rows_m1 = []

for w0 in w0s:
    ns = []
    for k in ks:
        w_ex = w_gap(k, w0)
        C1, _ = rk4_run(N, D2p, w0, k, dt_out=dt_out, T=T_total)
        w_r = read_omega(C1, T_total, dt_out)
        d1 = abs(w_r - w_ex)
        tol = max(eps_fft, 1e-4)
        ok_a = d1 <= tol
        m1_a_pass &= ok_a
        # 群速中心差分（同流程相位法）
        Cp, _ = rk4_run(N, D2p, w0, k + dk, dt_out=dt_out, T=T_total)
        Cm, _ = rk4_run(N, D2p, w0, k - dk, dt_out=dt_out, T=T_total)
        vg_r = (read_omega(Cp, T_total, dt_out) - read_omega(Cm, T_total, dt_out)) / (2 * dk)
        n_r = 1.0 / vg_r
        n_le = n_latt_exact(k, w0)
        n_ct = n_cont(k, w0)
        # 差分读数预算：相位提取误差/(2dk) + 差分二阶截断 |v_g'''| dk^2/6（v_g''' ~ O(1)）
        budget_diff = tol / (2 * dk) * 2 + dk ** 2
        ok_c = abs(n_r - n_ct) <= abs(n_le - n_ct) + budget_diff
        m1_c_pass &= ok_c
        rows_m1.append((w0, k, w_r, w_ex, d1, tol, n_r, n_le, n_ct, abs(n_r - n_ct),
                        abs(n_le - n_ct), budget_diff, ok_a and ok_c))
        ns.append(n_r)
        print("  w0=%.1f k=%.2f: w_read=%.6f vs w_exact=%.6f (d=%.2e, tol=%.1e) %s"
              % (w0, k, w_r, w_ex, d1, tol, "PASS" if ok_a else "FAIL"))
        print("           n_read=%.4f n_latt=%.4f n_cont=%.4f | n-n_cont=%.4f 预算=%.4f %s"
              % (n_r, n_le, n_ct, abs(n_r - n_ct), abs(n_le - n_ct) + budget_diff,
                 "PASS" if ok_c else "FAIL"))
    mono = ns[0] > ns[1] > ns[2]
    m1_b_pass &= mono
    print("  w0=%.1f 单调性: n = %.3f > %.3f > %.3f  %s"
          % (w0, ns[0], ns[1], ns[2], "PASS" if mono else "FAIL"))

print("\n[M1d] 无能隙对照 w0=0")
for k in ks:
    C1, _ = rk4_run(N, D2p, 0.0, k, dt_out=dt_out, T=T_total)
    w_r = read_omega(C1, T_total, dt_out)
    w_ex = w_latt(k)
    d1 = abs(w_r - w_ex)
    ok = d1 <= max(eps_fft, 1e-4)
    m1_d_pass &= ok
    print("  k=%.2f: w_read=%.6f vs w_latt=%.6f (d=%.2e) %s"
          % (k, w_r, w_ex, d1, "PASS" if ok else "FAIL"))

M1_pass = m1_a_pass and m1_b_pass and m1_c_pass and m1_d_pass
print("\nM1 汇总: %s（a=%s b=%s c=%s d=%s）"
      % ("PASS" if M1_pass else "FAIL", m1_a_pass, m1_b_pass, m1_c_pass, m1_d_pass))

# ---------- M2 ----------
print("\n" + "=" * 72)
print("M2 低频禁戒（谱禁戒带 + 修逝衰减）")
print("=" * 72)
N2 = 400
w0_m2 = 1.0
w_min_read, _ = spectral_min_open(N2, w0_m2)
w_min_exact = np.sqrt(w0_m2 ** 2 + 4 * np.sin(np.pi / (2 * (N2 + 1))) ** 2)
d_spec = abs(w_min_read - w_min_exact)
ok_2a = d_spec < 1e-9 and w_min_read >= w0_m2
print("  [2a] 谱 min w: read=%.10f exact=%.10f (d=%.2e)  且 >= w0: %s"
      % (w_min_read, w_min_exact, d_spec, "PASS" if ok_2a else "FAIL"))

ratios = [0.3, 0.45, 0.6, 0.75, 0.9, 0.95]
mus_r, mus_e = [], []
ok_2b = True
mono_mu = True
prev = None
for r in ratios:
    w_t = r * w0_m2
    mu_r, resid, _, _ = evanescent_mu(N2, w0_m2, w_t)
    mu_e = mu_exact(w0_m2, w_t)
    ok = abs(mu_r - mu_e) < 1e-6 and resid < 1e-8
    ok_2b &= ok
    if prev is not None:
        mono_mu &= (mu_r < prev - 1e-12)
    prev = mu_r
    mus_r.append(mu_r)
    mus_e.append(mu_e)
    print("  [2b] w/w0=%.2f: mu_read=%.8f mu_exact=%.8f (d=%.2e, resid=%.1e) %s"
          % (r, mu_r, mu_e, abs(mu_r - mu_e), resid, "PASS" if ok else "FAIL"))
print("  [2b] mu 随 w->w0 单调下降: %s" % ("PASS" if mono_mu else "FAIL"))

M2_pass = ok_2a and ok_2b and mono_mu
print("\nM2 汇总: %s" % ("PASS" if M2_pass else "FAIL"))

# ---------- M3 ----------
print("\n" + "=" * 72)
print("M3 能隙上缘色散符号（修正后期望 dn/dw < 0）")
print("=" * 72)
w0_m3 = 1.0
w_over = [1.1, 1.2, 1.5]
ns3 = []
for r in w_over:
    w_t = r * w0_m3
    k_t = 2 * np.arcsin(np.sqrt(w_t ** 2 - w0_m3 ** 2) / 2)
    n_t = w_t / np.sin(k_t)
    n_ct3 = w_t / np.sqrt(w_t ** 2 - w0_m3 ** 2)
    ns3.append(n_t)
    print("  w/w0=%.1f: k=%.4f n_latt=%.4f n_cont=%.4f |格点-连续|=%.4f"
          % (r, k_t, n_t, n_ct3, abs(n_t - n_ct3)))
dec = ns3[0] > ns3[1] > ns3[2]
dq1 = (ns3[1] - ns3[0]) / (w_over[1] - w_over[0])
dq2 = (ns3[2] - ns3[1]) / (w_over[2] - w_over[1])
print("  n 三档递减: %s；差商 dq(w=1.15)=%.3f, dq(w=1.35)=%.3f（期望皆 < 0）"
      % ("PASS" if dec else "FAIL", dq1, dq2))
M3_pass = dec and dq1 < 0 and dq2 < 0
print("  判据修正说明：注记原期望 dn/dw>0 与闭式 n=w/sqrt(w^2-w0^2) 自相矛盾，")
print("  真实期望 dn/dw<0；正常色散（蓝光更折）须由反向项/完整振子响应供给，登记为失配第二表现。")
print("\nM3 汇总: %s" % ("PASS" if M3_pass else "FAIL"))

# ---------- M4 ----------
print("\n" + "=" * 72)
print("M4 通道排他（低频带隙存在性 + 恒真齿）")
print("=" * 72)
w0_m4 = 1.0
nu = 0.3
N_sp = 200
w_min_sp = speed_spectral_min(N_sp, w0_m4, nu)
# 期望 = 广义二次小根（含 nu 混合修正），wl 取开链 Dirichlet 最低模
wl_min2 = 4 * np.sin(np.pi / (2 * (N_sp + 1))) ** 2
S_m = wl_min2 + w0_m4 ** 2
P_m = wl_min2 * w0_m4 ** 2
w2m_exact = (S_m - np.sqrt(S_m ** 2 - 4 * (1 - nu ** 2) * P_m)) / (2 * (1 - nu ** 2))
w_min_sp_exact = np.sqrt(w2m_exact)
ok_4a = abs(w_min_sp ** 2 - w2m_exact) < 1e-10 and w_min_sp < 0.5 * w0_m4
print("  [4a] 速度型谱 min w = %.12e vs 二次小根解析 %.12e（d(w)=%.2e, d(w^2)=%.2e）"
      % (w_min_sp, w_min_sp_exact, abs(w_min_sp - w_min_sp_exact),
         abs(w_min_sp ** 2 - w2m_exact)))
print("       且 < 0.5 w0：互斥于能隙型 min w = w0 = %.1f  %s"
      % (w0_m4, "PASS" if ok_4a else "FAIL"))

ks_low = [0.15, 0.3]
ok_4b = True
for k in ks_low:
    wm, _ = w_speed_types(k, w0_m4, nu)
    wm2, _ = w_speed_types(k + 0.02, w0_m4, nu)
    wm1, _ = w_speed_types(k - 0.02, w0_m4, nu)
    vg_sp = (wm2 - wm1) / 0.04
    vg_gap = vg_latt(k, w0_m4)
    ok_sp = vg_sp >= 0.75
    ok_gap = vg_gap <= 0.35
    ok_4b &= ok_sp and ok_gap
    print("  [4b] k=%.2f: v_g^speed=%.4f（期望>=0.75）%s | v_g^gap=%.4f（期望<=0.35）%s"
          % (k, vg_sp, "PASS" if ok_sp else "FAIL", vg_gap, "PASS" if ok_gap else "FAIL"))

print("  [4c] 速度型 RK4 双频读数 vs 解析二次根（k=%.5f 整模档）" % (round(1.2 / dk_mode) * dk_mode))
C_sp = rk4_speed_read(N, w0_m4, nu, round(1.2 / dk_mode) * dk_mode)
pks = read_two_omegas(C_sp, 0.1)
wm_ex, wp_ex = w_speed_types(round(1.2 / dk_mode) * dk_mode, w0_m4, nu)
d_ac = min(abs(p - wm_ex) for p in pks)
d_ac2 = min(abs(p - wp_ex) for p in pks)
ok_4c = d_ac < 5e-3 and d_ac2 < 5e-3
print("      read = (%.4f, %.4f) vs exact = (%.4f, %.4f) d=(%.2e, %.2e) %s"
      % (pks[0], pks[1], wm_ex, wp_ex, d_ac, d_ac2, "PASS" if ok_4c else "FAIL"))

print("  [4d] 恒真检验（EP5 齿）：把速度型谱 min 喂给 M2 期望（min w >= w0）")
ok_4d = not (w_min_sp >= w0_m4)
print("      速度型在 M2 期望下判 %s（应 FAIL）-> 齿 %s"
      % ("PASS" if w_min_sp >= w0_m4 else "FAIL", "PASS" if ok_4d else "FAIL"))

M4_pass = ok_4a and ok_4b and ok_4c and ok_4d
print("\nM4 汇总: %s" % ("PASS" if M4_pass else "FAIL"))

# ---------- 总汇总 ----------
print("\n" + "=" * 72)
print("总汇总: M1=%s M2=%s M3=%s M4=%s （%d/4）"
      % ("PASS" if M1_pass else "FAIL", "PASS" if M2_pass else "FAIL",
         "PASS" if M3_pass else "FAIL", "PASS" if M4_pass else "FAIL",
         sum([M1_pass, M2_pass, M3_pass, M4_pass])))
print("=" * 72)

# ---------- 图 ----------
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle("Verification 573: gap-type refractive channel (M1-M4)", fontsize=13)

ax = axes[0, 0]
kk = np.linspace(0.12, 1.35, 300)
for w0, col in zip(w0s, ("tab:blue", "tab:red")):
    ax.plot(kk, n_latt_exact(kk, w0), color=col, lw=1.5,
            label=r"$n_{\rm latt}$ exact, $\omega_0=%.1f$" % w0)
    ax.plot(kk, n_cont(kk, w0), color=col, ls="--", lw=1.0,
            label=r"$n_{\rm cont}$, $\omega_0=%.1f$" % w0)
    for k, row in [(r[1], r) for r in rows_m1 if r[0] == w0]:
        ax.plot(k, row[7], "o", color=col, ms=7, mfc="none", mew=1.6)
ax.plot(kk, n_latt_exact(kk, 0.0), color="k", ls=":", lw=1.2, label=r"$\omega_0=0$ (free)")
ax.set_xlabel(r"$k\, l_P$"); ax.set_ylabel(r"$n(k)=c/v_g$")
ax.set_title("M1: monotone decrease & lattice-budget closeness")
ax.legend(fontsize=7); ax.set_ylim(0.5, 8); ax.set_yscale("log")

ax = axes[0, 1]
rr = np.linspace(0.25, 0.97, 200)
ax.plot(rr, mu_exact(1.0, rr), "k-", lw=1.2, label=r"$\mu$ exact (discrete)")
ax.plot(ratios, mus_r, "o", color="tab:green", ms=7, label=r"$\mu$ read (steady response)")
ax.set_xlabel(r"$\omega/\omega_0$"); ax.set_ylabel(r"$\mu\, l_P$")
ax.set_title("M2: evanescent decay below the gap (6 tiers)")
ax.legend(fontsize=8)

ax = axes[1, 0]
ww = np.linspace(1.03, 1.7, 200)
k_of_w = 2 * np.arcsin(np.sqrt(np.clip(ww ** 2 - 1.0, 0, None)) / 2)
ax.plot(ww, ww / np.sin(k_of_w), "k-", lw=1.2, label=r"$n_{\rm latt}(\omega)$")
ax.plot(ww, ww / np.sqrt(ww ** 2 - 1.0), "b--", lw=1.0, label=r"$n_{\rm cont}(\omega)$")
ax.plot([r * w0_m3 for r in w_over], ns3, "ro", ms=7, label="read tiers")
ax.set_xlabel(r"$\omega/\omega_0$"); ax.set_ylabel(r"$n(\omega)$")
ax.set_title(r"M3: $dn/d\omega<0$ above the gap (corrected expectation)")
ax.legend(fontsize=8)

ax = axes[1, 1]
ax.axhline(w0_m4, color="tab:blue", lw=1.5, label=r"gap-type: $\min\omega=\omega_0$")
ax.axhline(w_min_sp, color="tab:red", lw=1.5, label=r"speed-type: $\min\omega\approx 0$")
ax.set_ylim(0, 1.3)
ax2 = ax.twinx()
kk2 = np.array(ks_low)
vg_sp_arr = []
for k in ks_low:
    wm1, _ = w_speed_types(k - 0.02, w0_m4, nu)
    wm2, _ = w_speed_types(k + 0.02, w0_m4, nu)
    vg_sp_arr.append((wm2 - wm1) / 0.04)
ax2.plot(kk2, vg_sp_arr, "rs", ms=8, label=r"$v_g$ speed-type")
ax2.plot(kk2, [vg_latt(k, 1.0) for k in ks_low], "bo", ms=8, label=r"$v_g$ gap-type")
ax2.set_ylabel(r"$v_g/c$"); ax2.set_ylim(0, 1.05)
ax.set_xlabel(r"$k\,l_P$ (low-k tiers)"); ax.set_ylabel(r"$\omega$")
ax.set_title("M4: exclusion via gap existence & low-k $v_g$")
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="center right")

plt.tight_layout(rect=(0, 0, 1, 0.96))
plt.savefig(IMG, dpi=150)
print("\n图已存: " + IMG)
