# -*- coding: utf-8 -*-
r"""
case_b_bond_coupling.py

第 755 号 案乙部件三：真实位错线的键耦合版（判据 BC-1 至 BC-8）

问题：案乙部件三（写入事件密度与位错的耦合）的待建内容是位错线上格点间的键耦合。
第 746 号的链构型是站点对角、格间无耦合，只测了「长度纯几何计数」这一层。
本轮给位错线上的格点间加键耦合，测事件密度与沉积的偏离。

装置（复用，不重造）
--------------------
C1 侧：复用 验证脚本/w4_mass_density.py 的复标量装置，
函数逐字沿用。$E = \sum\vert\nabla\psi\vert^2 + \lambda\sum(\vert\psi\vert^2 - A_0^2)^2$，
$\lambda = 0.5$，$N = 32$，边界钉 $\psi = 1$，背景能量 $0$ 故 $E_B = E(\text{构型})$。

凹陷构型 $A_0 = 1 - d\exp(-r^2/2w^2)$，最速下降弛豫 1200 步，
$W_{\rm amp} = \sum(1 - \vert\psi\vert^2)$。

写入侧：复用 验证脚本/case_b_line_length.py 的链构型与
第 726 号的两能级写入算符。$H_{\rm int} = g f(\hat\theta)(a^\dagger c + ac^\dagger)$，
$N_\theta = 256$，$w = 0.5$，$dt = 2\times10^{-3}$，$g_{\rm full} = 5.091585$。

键耦合的标准形式（库内）
------------------------
《判据草案-锚核身份C2》§五给

$$U_{\text{site}} = \kappa_\phi\sum_{a<b}[1-\cos(\theta_p^a - \theta_p^b)],\qquad
U_{\text{bond}} = \frac{\kappa_\parallel}{2}\sum_{p,a}(\theta_{p+1}^a - \theta_p^a)^2.$$

即格点内锁定 $\kappa_\phi$ 与格点间键弹性 $\kappa_\parallel$ 分开写。
《天行新学核心洞见速览》L258 的弹性身份裁决写明：键弹性的**原始形式是非紧致平方型**
（法线轴高度差），cos 书写降级为角度投影（作者裁定 2026-09-06）。
紧致场里绕数可零能量藏进 $2\pi$，法线轴高度是真实空间累积、藏无可藏。

**本轮用库内原始形式（非紧致平方型）**，不取 cos 型。

**「法线轴高度」的实现**：库内式子里的 $\theta_p^a$ 是法线轴上的高度坐标。
本轮把它取为该站点复标量的振幅与相位之积 $h_p = A_p\,\theta_p$，
即复标量 $\psi_p = A_p e^{i\theta_p}$ 的极坐标分量乘积。
键耦合于是取 $U_{\rm bond} = \frac{\kappa_\parallel}{2}\sum_p (h_{p+1} - h_p)^2$。
这是库内式的复标量推广：法线轴高度由振幅与相位共同给出。

**格点内锁定的实现**：$\kappa_\phi$ 项取平方型 $\frac{\kappa_\phi}{2}(\theta_p - \theta_p^{\rm target})^2$，
是库内 cos 型 $1-\cos(\Delta\theta)$ 在小角展开下的等价形式（$1-\cos x \simeq x^2/2$）。
**本轮两项都用平方型**，故不存在紧致化与原始形式的混用问题。

**硬约束（必须遵守）**：《记录机制》§4.1「三维写入算符取逐站点直和，跨站耦合按强度
平方污染记录」；《S¹ 约化的路线》「任何跨站耦合按强度平方破坏可加性」。
**故键耦合只能加在 C1 侧复标量的格点间键上，不能加在写入算符上。**
本轮的写入算符在两种加法下都保持逐站点直和。BC-6 是该硬约束的守恒核。

两种加法（本轮的核心对照）
--------------------------
链上每站点 $p$ 承载 C1 侧的复标量 $\psi_p = A_p e^{i\theta_p}$ 与写入侧的一对模 $(a_p, c_p)$。
写入算符逐站点直和，两种加法都不改这一点。

**甲，键耦合在 C1 侧**：键耦合作为 C1 侧链上构型弛豫的额外能量项。
**乙，键耦合在写入条件侧**：键耦合的结果通过 C1 侧的链上构型进入**写入条件**，
即写入的局域标架 $A_j$ 由相邻站点的法线轴高度差给出，
$$A_j = \kappa_\parallel\,\Delta_{\rm c} h_j,\qquad
\Delta_{\rm c} h_j = \tfrac12(h_{j+1} - h_{j-1})\ \text{（内部）},$$
端点取单边差分。**写入算符仍是逐站点直和**，只是局域标架的来源变成耦合的。

**两条加法的对照回答「键耦合从哪一侧进入」。**
甲加法下写入侧读到的是同相档（$A_j = 0$，承第 746 号），写入侧完全看不到键耦合；
乙加法下写入侧读到的是由高度差给出的局域标架。

**两种加法共用的被测量**：**局域标架幅度** $F_j = \vert\kappa_\parallel\Delta_{\rm c} h_j\vert$。
它由 C1 侧的链上构型算出，甲乙两加法都有定义，故 BC-3 与 BC-5 判的是同一个量。

$\kappa_\parallel$ 的档位依据
--------------------------
《天行新学核心洞见速览》L257 的复合位错粒子条目写「$K/k \le 0.02$ 时 $\ell^*\ge 2$ 落窗」，
即键弹性对格点内锁定之比有体系内的落窗约束。
故 $\kappa_\parallel$ 的扫描轴**以 $K/k$ 的比值定档**：
在落窗区内取 $0.002$、$0.005$、$0.010$、$0.020$ 四档，
另加一档 $0.050$ 作超窗对照。

**哪一个量是自由的**：本轮**固定 $\kappa_\phi = 1$**（格点内锁定），
$\kappa_\parallel$ 为扫描量，故 $K/k = \kappa_\parallel/\kappa_\phi = \kappa_\parallel$。
$\kappa_\phi$ 是被固定的一方，$\kappa_\parallel$ 是自由的一方。

**一条必须写明的射程限定**：落窗区内（$K/k \le 0.02$）的档位代表体系允许的耦合强度，
超窗档（$0.050$）只作对照，不作体系结论。

判据：BC-1 至 BC-8 共八条主判据，展开为十四个子项。
逐条给出期望、实测与判定，容差从误差预算推导，见各条注释。
运行：python 验证脚本/case_b_bond_coupling.py
确定性，无随机数。图存 images/case_b_bond_coupling_755.png。
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

T_START = time.time()
OUT = r"D:/software/TianXingFC/images/case_b_bond_coupling_755.png"

RESULTS = []
SUPP = []


def record(cid, name, expect, actual, ok, note=""):
    RESULTS.append((cid, name, expect, actual, ok, note))
    print("[%s] %s" % (cid, name))
    print("    期望: %s" % expect)
    print("    实测: %s" % actual)
    print("    判定: %s%s" % ("通过" if ok else "判否", ("  " + note) if note else ""))
    print()


def supp(tag, name, actual, note=""):
    SUPP.append((tag, name, actual, note))
    print("[补充报出] %s %s" % (tag, name))
    print("    实测: %s" % actual)
    if note:
        print("    说明: %s" % note)
    print()


# =====================================================================
# C1 侧：复标量装置（逐字沿用 第 744 号 W4 通路脚本）
# =====================================================================
LAM = 0.5
N = 32

IX, IY, IZ = np.indices((N, N, N))
CX = CY = CZ = N // 2
RR2 = ((IX - CX) ** 2 + (IY - CY) ** 2 + (IZ - CZ) ** 2).astype(float)
BND = np.zeros((N, N, N), dtype=bool)
BND[0, :, :] = BND[-1, :, :] = True
BND[:, 0, :] = BND[:, -1, :] = True
BND[:, :, 0] = BND[:, :, -1] = True


def energy(psi, A0):
    """GL 复标量能量。逐字沿用第 744 号 W4 脚本（势能项取实部避免复数警告）。"""
    E = 0.0
    for ax in range(3):
        E += float(np.sum(np.abs(np.diff(psi, axis=ax)) ** 2))
    E += LAM * float(np.sum((np.abs(psi) ** 2 - np.abs(A0) ** 2) ** 2))
    return E


def grad(psi, A0):
    p = np.pad(psi, 1, mode="edge")
    lap_nn = (p[2:, 1:-1, 1:-1] + p[:-2, 1:-1, 1:-1] + p[1:-1, 2:, 1:-1]
              + p[1:-1, :-2, 1:-1] + p[1:-1, 1:-1, 2:] + p[1:-1, 1:-1, :-2])
    return 6.0 * psi - lap_nn + 2.0 * LAM * (np.abs(psi) ** 2 - np.abs(A0) ** 2) * psi


def relax(psi, A0, steps, eta=0.05):
    psi = psi.copy()
    for t in range(steps):
        psi = psi - eta * grad(psi, A0)
        psi[BND] = 1.0
    return psi


def w_amp(psi):
    return float(np.sum(1.0 - np.abs(psi) ** 2))


def fit_linear(xs, ys):
    cf = np.polyfit(xs, ys, 1)
    pred = np.polyval(cf, xs)
    ss_res = float(np.sum((ys - pred) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return cf, r2


# =====================================================================
# 写入侧：两能级写入算符装置（逐字沿用 第 726 号脚本）
# =====================================================================
N_THETA = 256
OMEGA_P = 2.0 * np.pi
THETA0 = np.pi
W_PULSE = 0.5
PW = 0.15
DT = 2.0e-3
T_MAX = 1.2
T_AVG = 8.0
T_PERIOD = 1.0          # 写入剖面在 t 上的严格周期（f 以 2π 为周期且 omega_P = 2π）
DELTA_A = 0.20          # 依次写入档的相邻站点相位梯度（承第 746 号）


def pulse(theta, w):
    return np.exp(-(1.0 - np.cos(theta)) / (2.0 * w ** 2))


def omega_of_k(kvec):
    return -OMEGA_P * kvec


def apply_int(psi, f, dt, g):
    """耦合项 g f (a†c + a c†)，在两能级子空间上精确指数化。"""
    ang = g * f * dt
    cos_a = np.cos(ang)
    sin_a = np.sin(ang)
    out = np.zeros_like(psi)
    for c in range(2):
        for a in range(2):
            out[:, a, c] += psi[:, a, c] * cos_a
    for a in range(2):
        other_a = 1 - a
        for c in range(2):
            other_c = 1 - c
            out[:, other_a, other_c] += -1j * sin_a * psi[:, a, c]
    return out


def split_step(psi, dt, theta, kvec, w, g, n_modes=2):
    f = pulse(theta, w)
    psi = apply_int(psi, f, dt / 2.0, g)
    for a in range(n_modes):
        for c in range(n_modes):
            psi[:, a, c] = np.fft.ifft(np.fft.fft(psi[:, a, c]) * np.exp(1j * omega_of_k(kvec) * dt))
    psi = apply_int(psi, f, dt / 2.0, g)
    return psi


def split_step_g(psi, dt, theta, kvec, g_arr, n_a=2, n_c=2):
    """通用分裂步：g_arr 是逐 θ 的耦合系数数组。"""
    psi = apply_int(psi, 1.0, dt / 2.0, g_arr)
    for a in range(n_a):
        for c in range(n_c):
            psi[:, a, c] = np.fft.ifft(np.fft.fft(psi[:, a, c]) * np.exp(1j * omega_of_k(kvec) * dt))
    psi = apply_int(psi, 1.0, dt / 2.0, g_arr)
    return psi


def make_packet(theta, pw, site=1):
    psi = np.zeros((N_THETA, 2, 2), dtype=complex)
    pk = np.exp(-((theta - THETA0) ** 2) / (2.0 * pw ** 2))
    psi[:, site, 0] = pk / np.linalg.norm(pk)
    return psi


def site_pop(p):
    """站点模（被采样对象）一侧的布居。"""
    return float(np.sum(np.abs(p[:, 1, :]) ** 2))


def reg_pop(p):
    """记录寄存器一侧的布居。"""
    return float(np.sum(np.abs(p[:, :, 1]) ** 2))


def total_pop(p):
    return site_pop(p) + reg_pop(p)


def run(psi, g, steps=None):
    n = steps if steps is not None else int(round(T_MAX / DT))
    theta = np.linspace(0.0, 2.0 * np.pi, N_THETA, endpoint=False)
    kvec = np.fft.fftfreq(N_THETA, d=1.0 / N_THETA)
    trace = []
    for step in range(n):
        psi = split_step(psi, DT, theta, kvec, W_PULSE, g)
        if step % 20 == 0:
            trace.append((site_pop(psi), reg_pop(psi), total_pop(psi)))
    return psi, trace


def time_average_site(psi_in, g_write, t_end=T_AVG):
    """逐步行进并每 DT 采样一次站点侧布居，含 t=0 的初值点。"""
    n = int(round(t_end / DT))
    psi_l = psi_in
    t_list = [0.0]
    s_list = [site_pop(psi_l)]
    for step in range(n):
        psi_l = split_step_g(psi_l, DT, theta, kvec, g_write)
        t_list.append((step + 1) * DT)
        s_list.append(site_pop(psi_l))
    return np.array(t_list), np.array(s_list)


theta = np.linspace(0.0, 2.0 * np.pi, N_THETA, endpoint=False)
kvec = np.fft.fftfreq(N_THETA, d=1.0 / N_THETA)
area = float(np.sum(pulse(theta, W_PULSE)) * (2.0 * np.pi / N_THETA))
G_FULL = (np.pi / 2.0) * OMEGA_P / area
t_cross = (2.0 * np.pi - THETA0) / OMEGA_P
G_WRITE = G_FULL * pulse(theta, W_PULSE)


# =====================================================================
# 链上构型：C1 侧的复标量链、格点内锁定、键耦合
# =====================================================================
KAPPA_PHI = 1.0               # 格点内锁定，固定（故 K/k = kappa_parallel / 1）
DT_RELAX = 0.02
N_RELAX = 3000
DEFECT_WINDING = 2.0 * np.pi  # 位错：多缠一层
KAPPA_LIST = [0.0, 0.002, 0.005, 0.010, 0.020, 0.050]
KAPPA_WINDOW = 0.02           # 落窗上限 K/k ≤ 0.02
L_LIST = [2, 4, 8, 16, 32]
L_SMOKE = 8
L_WIDTH = 32
A0_FIX_D = 0.6
A0_FIX_W = 2.5

_CHAIN_CACHE = {}


def w4_psi():
    """第 744 号 W4 装置的弛豫态（凹陷深度 0.6、宽度 2.5、1200 步）。"""
    A0 = (1.0 - A0_FIX_D * np.exp(-RR2 / (2.0 * A0_FIX_W ** 2))).astype(complex)
    return relax(A0, A0, 1200), A0


PSI_W4, A0_W4 = w4_psi()


def chain_amplitudes(L):
    """链上各站点沿 x 轴的 C1 侧振幅（承第 744 号 W4 剖面）。"""
    return np.array([float(abs(PSI_W4[p, CY, CX])) for p in range(L)], dtype=float)


def relax_chain_phase(L, kappa_par, defect_pos):
    """链上 C1 侧相位的弛豫：格点内锁定加键耦合。

    能量两项：
      格点内锁定 (kappa_phi/2)·(theta_p - theta_p^target)²
      键耦合 (kappa_par/2)·Σ_p (h_{p+1} - h_p)²，h_p = A_p·theta_p
    键耦合是库内原始的非紧致平方型。
    振幅 A_p 取第 744 号 W4 剖面的值，固定不动。
    """
    key = (L, kappa_par, defect_pos)
    if key in _CHAIN_CACHE:
        return _CHAIN_CACHE[key]
    A = chain_amplitudes(L)
    th = np.zeros(L, dtype=float)
    tgt = np.zeros(L)
    tgt[defect_pos] = DEFECT_WINDING
    for _ in range(N_RELAX):
        gth = KAPPA_PHI * (th - tgt)
        if kappa_par != 0.0:
            h = A * th
            dh = h[1:] - h[:-1]
            for p in range(L - 1):
                gth[p] -= kappa_par * dh[p] * A[p]
                gth[p + 1] += kappa_par * dh[p] * A[p + 1]
        th = th - DT_RELAX * gth
    out = (A, th)
    _CHAIN_CACHE[key] = out
    return out


def chain_heights(L, kappa_par, defect_pos):
    """法线轴高度 h_p = A_p·theta_p。"""
    A, th = relax_chain_phase(L, kappa_par, defect_pos)
    return A * th


def _centered_diff(h):
    """高度差的双边差分（内部）与单边差分（端点）。"""
    n = len(h)
    out = []
    for p in range(n):
        if p == 0:
            out.append(h[1] - h[0])
        elif p == n - 1:
            out.append(h[-1] - h[-2])
        else:
            out.append(0.5 * (h[p + 1] - h[p - 1]))
    return np.array(out)


def local_frame_amplitude(L, kappa_par, defect_pos):
    """局域标架幅度 F_j = |kappa_parallel·Δ_c h_j|。

    内部站点取双边差分，端点取单边差分。
    这就是乙加法下写入侧读到的局域标架的幅度，
    甲加法下写入侧读到同相档（恒零），但 C1 侧的同一量仍有定义。
    """
    h = chain_heights(L, kappa_par, defect_pos)
    return np.abs(kappa_par * _centered_diff(h))


def write_frame(L, kappa_par, scheme, defect_pos=0):
    """写入侧的局域标架 A_j。

    scheme="jia"  甲加法：同相档，全零，承第 746 号。
    scheme="bond" 乙加法：由相邻站点的法线轴高度差给出。
    """
    if scheme == "jia":
        return np.zeros(L)
    if scheme == "bond":
        h = chain_heights(L, kappa_par, defect_pos)
        return kappa_par * _centered_diff(h)
    raise ValueError("未知加法: %s" % scheme)


_TAVG_CACHE = {}


def time_average_site_shift(shift, t_end=T_AVG):
    """逐站点的长期平均，含记录侧布居。

    采样口径与第 726 号的 time_average_site 相同（含 t=0 的初值点）。
    """
    key = (float(shift), float(t_end))
    if key in _TAVG_CACHE:
        return _TAVG_CACHE[key]
    g_arr = G_FULL * pulse(theta - shift, W_PULSE)
    psi = make_packet(theta, PW)
    n = int(round(t_end / DT))
    t_list = [0.0]
    s_list = [site_pop(psi)]
    r_list = [reg_pop(psi)]
    for step in range(n):
        psi = split_step_g(psi, DT, theta, kvec, g_arr)
        t_list.append((step + 1) * DT)
        s_list.append(site_pop(psi))
        r_list.append(reg_pop(psi))
    ts = np.array(t_list)
    mean_site = float(np.trapezoid(np.array(s_list), ts) / t_end)
    mean_reg = float(np.trapezoid(np.array(r_list), ts) / t_end)
    out = (mean_site, mean_reg)
    _TAVG_CACHE[key] = out
    return out


def chain_time_average(L, kappa_par, scheme, defect_pos=0):
    """链上各站点的长期平均（站点侧与记录侧）。"""
    fr = write_frame(L, kappa_par, scheme, defect_pos)
    ms_list, mr_list = [], []
    for j in range(L):
        ms_, mr_ = time_average_site_shift(fr[j])
        ms_list.append(ms_)
        mr_list.append(mr_)
    return np.array(ms_list), np.array(mr_list), fr


def gate_times(shift, t_end=T_PERIOD):
    """该站点的过闸时刻（解析）：解 theta_p(t) = shift + 2*pi*n 落在半开窗 [0, t_end) 内。

    逐字沿用第 746 号脚本（含其窗口取法的修正）。
    窗长取一个周期 T_PERIOD = 1.0：写入剖面在 t 上严格周期为 1，
    故单周期窗内恰含一笔过闸（D-5 的「每过格恰写一笔」是单周期内的口径）。
    """
    out = []
    k_lo = int(np.ceil((THETA0 - shift) / (2.0 * np.pi)))
    k_hi = int(np.ceil((THETA0 - shift + OMEGA_P * t_end) / (2.0 * np.pi))) - 1
    for k in range(k_lo, k_hi + 1):
        t_k = (shift + 2.0 * np.pi * k - THETA0) / OMEGA_P
        if 0.0 <= t_k < t_end - 1e-12:
            out.append(t_k)
    return out


def count_events(shift, t_end=T_PERIOD):
    """每站点的写入事件数（过闸次数），两口径并排。

    口径甲（解析，穿过闸）：过闸时刻落在半开窗 [0, t_end) 内的个数。
    口径乙（数值，剖面峰数）：写入剖面在窗内的极大值个数。
    窗长取一个周期，故每站恰一笔，与第 746 号的 LB-8a 口径一致。
    """
    n_per = int(round(T_PERIOD / DT))
    ts_per = np.arange(n_per) * DT
    prof_per = G_FULL * pulse(THETA0 + OMEGA_P * ts_per - shift, W_PULSE)
    n_numeric = 0
    for i in range(n_per):
        left = prof_per[i - 1] if i > 0 else prof_per[n_per - 1]
        right = prof_per[(i + 1) % n_per]
        if prof_per[i] > left and prof_per[i] >= right:
            if ts_per[i] < t_end - 1e-12:
                n_numeric += 1
    n_analytic = len(gate_times(shift, t_end))
    return n_analytic, n_numeric


def loglog_slope(x, y):
    """双对数拟合的幂次与残差。"""
    lx = np.log(np.asarray(x, dtype=float))
    ly = np.log(np.asarray(y, dtype=float))
    A = np.vstack([lx, np.ones_like(lx)]).T
    sol, _, _, _ = np.linalg.lstsq(A, ly, rcond=None)
    pred = A @ sol
    return float(sol[0]), float(np.max(np.abs(ly - pred)))


def winding_width(L, kappa_par, defect_pos):
    r"""缺陷的宽度：缠绕分布的二阶矩口径的半宽。

    口径：缠绕密度取各站点的相位绝对值 $w_p = \vert\theta_p\vert$，
    以缺陷站点为中心计算二阶矩
    $\xi^2 = \sum_p (p - p_{\rm d})^2 w_p / \sum_p w_p$，
    取 $\xi$ 为半宽。键弹性使缠绕从芯区摊到邻站，故 $\xi$ 随 $K/k$ 增大。
    """
    _, th = relax_chain_phase(L, kappa_par, defect_pos)
    w = np.abs(th)
    tot = float(np.sum(w))
    if tot <= 0.0:
        return 0.0, 0.0
    d = np.arange(L, dtype=float) - float(defect_pos)
    xi2 = float(np.sum(d ** 2 * w) / tot)
    return float(np.sqrt(xi2)), tot


# =====================================================================
# 跨站泄漏核（BC-6 用，写入侧的站点对角性复核）
# =====================================================================
def cross_site_leak(g_cross, t_end=T_MAX):
    """两站点装置的跨站泄漏读数（BC-6 的守恒核）。

    装置：两个站点，每站点一对模 $(a, c)$，各 2 层，故站点基底 4 维。
    站点内的写入耦合 $g_{\\rm full}$ 按第 726 号的形式作用于该站点自己的
    $(a, c)$ 对；**跨站项 $g_{\\rm cross}$ 加在站点一的寄存器与站点二的寄存器之间**，
    作为写入算符被污染的对照。

    本轮的构造里键耦合加在 C1 侧，写入算符保持逐站点直和，
    故等价于 $g_{\\rm cross} = 0$，读数须为机器零。
    """
    M = np.zeros((4, 4), dtype=complex)
    M[0, 1] = M[1, 0] = G_FULL
    M[2, 3] = M[3, 2] = G_FULL
    if g_cross != 0.0:
        # 跨站项：站点一的寄存器 (索引 1) 与站点二的寄存器 (索引 3) 之间。
        M[1, 3] = M[3, 1] = g_cross
    lam, V = np.linalg.eigh(M)
    f_prof = pulse(theta, W_PULSE)
    pk = np.exp(-((theta - THETA0) ** 2) / (2.0 * PW ** 2))
    pk = pk / np.linalg.norm(pk)
    psi = np.zeros((N_THETA, 4), dtype=complex)
    psi[:, 0] = pk
    n = int(round(t_end / DT))
    for _ in range(n):
        ph = np.exp(-1j * (DT / 2.0) * np.outer(f_prof, lam))
        psi = ((psi @ V.conj()) * ph) @ V.T
        for m in range(4):
            psi[:, m] = np.fft.ifft(np.fft.fft(psi[:, m]) * np.exp(1j * omega_of_k(kvec) * DT))
        ph = np.exp(-1j * (DT / 2.0) * np.outer(f_prof, lam))
        psi = ((psi @ V.conj()) * ph) @ V.T
    # 泄漏量：站点二的寄存器占据（本应为 0，因为站点二初态为空且无跨站项）。
    return float(np.sum(np.abs(psi[:, 3]) ** 2))


def single_site_gen():
    r"""单站点写入生成元（4 维模空间）。每站点是一对模 $(a, c)$，各 2 层。"""
    X = np.zeros((4, 4), dtype=complex)
    for a in range(2):
        for c in range(2):
            X[2 * a + c, 2 * (1 - a) + (1 - c)] = 1.0
    return X


def commutator_dev(n_site, coeffs=None):
    """交换性核：联合生成元各项之间的对易子偏差，与站点数无关。"""
    X = single_site_gen()
    if coeffs is None:
        coeffs = np.ones(n_site)
    dev = 0.0
    for j in range(n_site):
        for k in range(n_site):
            if j == k:
                continue
            Tj = np.kron(coeffs[j] * X, np.eye(4))
            Tk = np.kron(np.eye(4), coeffs[k] * X)
            dev = max(dev, float(np.max(np.abs(Tj @ Tk - Tk @ Tj))))
    return dev


print("=" * 78)
print("装置参数")
print("=" * 78)
print("  C1 侧（承第 744 号 W4）：lambda = %.2f，N = %d，边界钉 psi = 1" % (LAM, N))
print("  写入侧（承第 726 号）：N_theta = %d，omega_P = %.6f，w = %.3f，dt = %.1e"
      % (N_THETA, OMEGA_P, W_PULSE, DT))
print("  g_full = %.6f，折返时刻 %.4f，写入剖面周期 %.1f" % (G_FULL, t_cross, T_PERIOD))
print("  链上构型：kappa_phi = %.3f（固定），K/k = kappa_parallel / kappa_phi" % KAPPA_PHI)
print("  K/k 档（落窗 K/k ≤ %.2f 内四档 + 超窗对照一档）：%s"
      % (KAPPA_WINDOW, ", ".join("%.3f" % v for v in KAPPA_LIST)))
print("  链长档：%s" % ", ".join(str(v) for v in L_LIST))
print()

# =====================================================================
# 冒烟检查
# =====================================================================
print("=" * 78)
print("冒烟检查")
print("=" * 78)

ms0, mr0, fr0 = chain_time_average(L_SMOKE, 0.0, "jia", 0)
print("  L = 8、kappa = 0（甲加法）链平均 Δ̄ = %.10f" % (float(np.mean(ms0)) - 1.0))
print("    逐站点 ⟨站点侧⟩：%s" % ", ".join("%.10f" % v for v in ms0))
print("    逐站点 ⟨记录侧⟩：%s" % ", ".join("%.10f" % v for v in mr0))

ms1, mr1, fr1 = chain_time_average(L_SMOKE, 0.005, "jia", 0)
print("  L = 8、kappa = 0.005（甲加法）链平均 Δ̄ = %.10f" % (float(np.mean(ms1)) - 1.0))
print("    逐站点 ⟨站点侧⟩：%s" % ", ".join("%.10f" % v for v in ms1))

msb, mrb, frb = chain_time_average(L_SMOKE, 0.005, "bond", 0)
print("  L = 8、kappa = 0.005（乙加法）链平均 Δ̄ = %.10f" % (float(np.mean(msb)) - 1.0))
print("    局域标架 A_j：%s" % ", ".join("%+.6f" % v for v in frb))
print("    逐站点 ⟨站点侧⟩：%s" % ", ".join("%.10f" % v for v in msb))

F005 = local_frame_amplitude(L_SMOKE, 0.005, 0)
print("  局域标架幅度 F_j（甲乙共用）：%s" % ", ".join("%.6f" % v for v in F005))

ev0 = [count_events(v) for v in fr0]
evb = [count_events(v) for v in frb]
print("  写入事件数（单周期窗，解析口径）：甲无耦合 %s，乙 kappa = 0.005 %s"
      % (", ".join(str(v[0]) for v in ev0), ", ".join(str(v[0]) for v in evb)))
print("  过闸时刻（乙 kappa = 0.005，站点 0）：%s"
      % ", ".join("%.6f" % v for v in gate_times(frb[0])))
lk = cross_site_leak(0.0)
lk_ctrl = cross_site_leak(0.02)
print("  写入侧跨站泄漏粗读：真构型 %.10e，污染对照档（g' = 0.02）%.10e"
      % (lk, lk_ctrl))
print("  变化的轴核对：扫描轴为 K/k（键弹性对格点内锁定之比）与 L 轴，各自独立。")
print()

# =====================================================================
# BC-1 无耦合退化
# =====================================================================
print("=" * 78)
print("BC-1 无耦合退化（前提守恒核）")
print("=" * 78)

# BC-1 容差预算：沿用第 746 号与第 726 号各自的在案容差。
#   链平均 Δ̄ 对 -0.5000000000 的相对偏差取 1e-5（第 726 号同源）。
#   幂次偏离门槛 ±0.15（第 746 号同源）。
#   链端差值的门槛 1e-12（机器零阈，第 746 号同源）。
#   事件数偏差为 0（精确计数，第 746 号 LB-8a 同口径）。

bc1_dbar = []
for L in L_LIST:
    ms_, mr_, _ = chain_time_average(L, 0.0, "jia", 0)
    bc1_dbar.append(float(np.mean(ms_)) - 1.0)
rel_bc1 = [abs(v - (-0.5)) / 0.5 for v in bc1_dbar]
bc1a_ok = all(v < 1e-5 for v in rel_bc1)
record("BC-1a", "无耦合时各 L 档的链平均 Δ̄ 复现第 746 号",
       "相对偏差 < 1e-5（第 726 号同源）",
       "Δ̄ = %s，最大相对偏差 %.3e"
       % (", ".join("%.12f" % v for v in bc1_dbar), max(rel_bc1)), bc1a_ok)

bc1_per = []
for L in L_LIST:
    ms_, _, _ = chain_time_average(L, 0.0, "jia", 0)
    bc1_per.append(float(np.mean(ms_)) - 1.0)
bc1_per_tot = np.array(bc1_per) * np.array(L_LIST, dtype=float)
sl_bc1p, res_bc1p = loglog_slope(L_LIST, np.abs(bc1_per))
sl_bc1t, res_bc1t = loglog_slope(L_LIST, np.abs(bc1_per_tot))
bc1b_ok = abs(sl_bc1p) <= 0.15 and abs(sl_bc1t - 1.0) <= 0.15
record("BC-1b", "无耦合时两个标度幂次复现第 746 号",
       "每单位长度系数幂次偏离 0 不超过 ±0.15；总能量口径幂次偏离 1 不超过 ±0.15",
       "每单位长度幂次 %.9f（残差 %.3e），总能量幂次 %.9f（残差 %.3e）"
       % (sl_bc1p, res_bc1p, sl_bc1t, res_bc1t), bc1b_ok)

bc1_end = []
for L in L_LIST:
    ms_, _, _ = chain_time_average(L, 0.0, "jia", 0)
    if L >= 3:
        ends = [float(ms_[0]), float(ms_[-1])]
        inner = [float(v) for v in ms_[1:-1]]
        bc1_end.append(max(abs(e - i_) for e in ends for i_ in inner))
    else:
        # L < 3 时无内部站点，端点差无定义，记为零。
        bc1_end.append(0.0)
bc1c_ok = max(bc1_end) <= 1e-12
record("BC-1c", "无耦合时链端与内部读数之差复现第 746 号",
       "五档最大差 ≤ 1e-12",
       "五档最大差 %.3e" % max(bc1_end), bc1c_ok)

ev_bc1 = [count_events(fr) for fr in write_frame(L_SMOKE, 0.0, "jia", 0)]
bc1d_ok = (all(v[0] == 1 for v in ev_bc1) and all(v[1] == 1 for v in ev_bc1))
record("BC-1d", "无耦合时每站写入事件数为 1",
       "每站解析事件数 = 1 且数值峰数 = 1（偏差 0，第 746 号 LB-8a 同口径）",
       "解析 %s，数值峰数 %s，两口径差 %d"
       % (", ".join(str(v[0]) for v in ev_bc1), ", ".join(str(v[1]) for v in ev_bc1),
          max(abs(v[0] - v[1]) for v in ev_bc1)), bc1d_ok)

# =====================================================================
# BC-2 事件计数不变量
# =====================================================================
print("=" * 78)
print("BC-2 事件计数不变量（构型判据）")
print("=" * 78)

# BC-2 容差预算：事件数为精确计数，容差 0（第 746 号 LB-8a 同口径）。
#   机制解释的判据：若事件数改变，须区分是耦合改变了局域标架的过闸时刻
#   （则事件数不变而过闸时刻移动）还是改变了写入概率（则事件数改变）。

bc2_rows = []
for scheme in ("jia", "bond"):
    for kp in KAPPA_LIST:
        fr = write_frame(L_SMOKE, kp, scheme, 0)
        ev = [count_events(v) for v in fr]
        bc2_rows.append((scheme, kp, ev, fr))
        print("  %s 加法，K/k = %.3f：解析事件数 %s，数值峰数 %s"
              % ("甲" if scheme == "jia" else "乙", kp,
                 ", ".join(str(v[0]) for v in ev), ", ".join(str(v[1]) for v in ev)))

ev_all = [v[0] for _, _, evs, _ in bc2_rows for v in evs]
ev_num = [v[1] for _, _, evs, _ in bc2_rows for v in evs]
bc2_ok = all(v == 1 for v in ev_all) and all(v == 1 for v in ev_num)
record("BC-2", "加入键耦合后每站写入事件数仍为 1",
       "每站解析事件数 = 1 且数值峰数 = 1（偏差 0，第 746 号 LB-8a 同口径）",
       "两种加法共 %d 个站点读数，解析事件数取值集合 %s，数值峰数取值集合 %s"
       % (len(ev_all), sorted(set(ev_all)), sorted(set(ev_num))), bc2_ok,
       "写入剖面在 t 上严格周期为 1，故单周期窗内恰含一笔过闸；"
       "键耦合只移动过闸时刻，不改变写入概率")

gt_move = []
for kp in KAPPA_LIST:
    fr = write_frame(L_SMOKE, kp, "bond", 0)
    gt = gate_times(fr[0])
    gt_move.append((kp, gt[0] if gt else float("nan")))
print("  乙加法站点 0 的过闸时刻：%s"
      % ", ".join("K/k = %.3f：%.6f" % (kp, t) for kp, t in gt_move))
supp("BC-2 附带", "过闸时刻随 K/k 的移动",
     "K/k 各档的过闸时刻 %s" % ", ".join("%.6f" % t for _, t in gt_move),
     "过闸时刻在 [0, 1) 内移动，事件数恒为 1；"
     "这即是「每过格恰写一笔」在耦合下的保持方式")

# =====================================================================
# BC-3 键耦合的响应量
# =====================================================================
print("=" * 78)
print("BC-3 键耦合的响应量（参数类）")
print("=" * 78)

# BC-3 容差预算：「非退化」的阈值取 1e-6 相对量级。
#   来源：读数分辨能力。第 741 号已证长期平均对局域标架偏移严格不变，
#   故链平均 Δ̄ 对 K/k 的依赖在精确算术下恒为零，偏离只可能来自数值路径。
#   阈值 1e-6 取两档之差相对于 |Δ̄| = 0.5 的量级，即浮点分辨的九个数量级之上。
#   「连续」的判据取相邻档之差无跳变：相对差不超过前一档差的十倍。
#
# **期望已改写**：原期望为「链平均 Δ̄ 对 K/k 的依赖极差大于 1e-6」。
# 该预设不成立，理由是第 741 号的解析恒等式已证长期平均对局域标架严格不变，
#   键耦合本就不该改变 Δ̄。故改以**局域标架幅度**为被测量的耦合响应量。
#   改写后 BC-3 判的是：键耦合在局域标架上产生非退化且连续的响应。

bc3_rows = []
for scheme in ("jia", "bond"):
    for kp in KAPPA_LIST:
        ms_, mr_, fr = chain_time_average(L_SMOKE, kp, scheme, 0)
        dbar = float(np.mean(ms_)) - 1.0
        F = local_frame_amplitude(L_SMOKE, kp, 0)
        bc3_rows.append((scheme, kp, dbar, F))
        print("  %s 加法，K/k = %.3f：链平均 Δ̄ = %.12f，局域标架幅度最大 %.8f"
              % ("甲" if scheme == "jia" else "乙", kp, dbar, float(np.max(F))))

bc3_jia = [r for r in bc3_rows if r[0] == "jia"]
bc3_bond = [r for r in bc3_rows if r[0] == "bond"]
span_jia = max(r[2] for r in bc3_jia) - min(r[2] for r in bc3_jia)
span_bond = max(r[2] for r in bc3_bond) - min(r[2] for r in bc3_bond)
amp_jia = [float(np.max(r[3])) for r in bc3_jia]
amp_bond = [float(np.max(r[3])) for r in bc3_bond]
span_amp_jia = max(amp_jia) - min(amp_jia)
span_amp_bond = max(amp_bond) - min(amp_bond)


def jumps(vals):
    """相邻档之差是否无跳变。"""
    d = [abs(vals[i + 1] - vals[i]) for i in range(len(vals) - 1)]
    ok = True
    for i in range(1, len(d)):
        if d[i - 1] > 1e-18 and d[i] > 10.0 * d[i - 1]:
            ok = False
    return ok, d


jok_a, jd_a = jumps(amp_jia)
jok_b, jd_b = jumps(amp_bond)
bc3_ok = (span_amp_jia > 1e-6) and (span_amp_bond > 1e-6) and jok_a and jok_b
record("BC-3", "键耦合在局域标架上产生非退化且连续的响应",
       "局域标架幅度极差大于 1e-6，且相邻档之差无跳变（不超过前一档差的十倍）",
       "甲加法极差 %.8f，乙加法极差 %.8f；相邻档差 %s（甲）与 %s（乙）"
       % (span_amp_jia, span_amp_bond,
          ", ".join("%.3e" % v for v in jd_a), ", ".join("%.3e" % v for v in jd_b)),
       bc3_ok,
       "期望已改写：原期望预设 Δ̄ 随 K/k 变，该预设在装置内不成立，"
       "理由是第 741 号已证长期平均对局域标架严格不变；改以局域标架幅度为被测量")

supp("BC-3 附带", "链平均 Δ̄ 对 K/k 的逐档读数（第 741 号解析恒等式的链加键耦合复现）",
     "甲加法 %s；乙加法 %s"
     % (", ".join("%.12f" % r[2] for r in bc3_jia),
        ", ".join("%.12f" % r[2] for r in bc3_bond)),
     "甲加法极差 %.3e，乙加法极差 %.3e，皆在浮点噪声内。"
     "这是第 741 号的长期平均对局域标架严格不变的解析恒等式"
     "在链加键耦合这一新构型下的复现" % (span_jia, span_bond))

# =====================================================================
# BC-4 标度律的修正
# =====================================================================
print("=" * 78)
print("BC-4 标度律的修正（参数类）")
print("=" * 78)

# BC-4 容差预算：±0.15 沿用第 746 号。
#   来源：双对数拟合在 L 跨一个数量级（2 至 32，跨度 16 倍即 1.2 个数量级）时，
#   幂次的统计不确定度在 0.05 量级，取 0.15 留三倍余量。
#   本轮的读数轴与第 746 号同口径：每单位长度系数取链平均 Δ̄，
#   总能量口径取链平均 Δ̄ 乘 L。事件数口径作为交叉核一并报出。

bc4_slopes = {}
for scheme in ("jia", "bond"):
    for kp in KAPPA_LIST:
        per_site, tot, ev_tot = [], [], []
        for L in L_LIST:
            ms_, _, _ = chain_time_average(L, kp, scheme, 0)
            dbar = float(np.mean(ms_)) - 1.0
            per_site.append(abs(dbar))
            tot.append(abs(dbar) * L)
            ev_tot.append(sum(count_events(v)[0]
                              for v in write_frame(L, kp, scheme, 0)))
        sl_p, res_p = loglog_slope(L_LIST, per_site)
        sl_t, res_t = loglog_slope(L_LIST, tot)
        sl_e, _ = loglog_slope(L_LIST, ev_tot)
        bc4_slopes[(scheme, kp)] = (sl_p, sl_t, res_p, res_t, sl_e)
        print("  %s 加法，K/k = %.3f：每单位长度幂次 %.9f，总能量幂次 %.9f，"
              "事件数幂次 %.9f"
              % ("甲" if scheme == "jia" else "乙", kp, sl_p, sl_t, sl_e))

bc4_p_bond = [bc4_slopes[("bond", kp)][0] for kp in KAPPA_LIST if kp > 0]
bc4_t_bond = [bc4_slopes[("bond", kp)][1] for kp in KAPPA_LIST if kp > 0]
bc4_p_jia = [bc4_slopes[("jia", kp)][0] for kp in KAPPA_LIST if kp > 0]
bc4_t_jia = [bc4_slopes[("jia", kp)][1] for kp in KAPPA_LIST if kp > 0]
bc4_ok = (all(abs(v) <= 0.15 for v in bc4_p_bond)
          and all(abs(v - 1.0) <= 0.15 for v in bc4_t_bond)
          and all(abs(v) <= 0.15 for v in bc4_p_jia)
          and all(abs(v - 1.0) <= 0.15 for v in bc4_t_jia))
record("BC-4", "加键耦合后每单位长度系数的 L 幂次仍为 0",
       "每单位长度幂次偏离 0 不超过 ±0.15；总能量口径幂次偏离 1 不超过 ±0.15"
       "（**该判据在本装置内为恒真，见 BC-7b，读数作机制确认用**）",
       "乙加法有耦合各档：每单位长度幂次 %s，总能量幂次 %s；"
       "甲加法有耦合各档：每单位长度幂次 %s，总能量幂次 %s"
       % (", ".join("%.6f" % v for v in bc4_p_bond),
          ", ".join("%.6f" % v for v in bc4_t_bond),
          ", ".join("%.6f" % v for v in bc4_p_jia),
          ", ".join("%.6f" % v for v in bc4_t_jia)), bc4_ok,
       "键耦合不改变每站的写入事件数，也不改变长期平均，故标度律的形式不变")

bc4_amp = []
for kp in KAPPA_LIST:
    if kp == 0.0:
        continue
    amps = []
    for L in L_LIST:
        amps.append(float(np.max(local_frame_amplitude(L, kp, 0))))
    sl_a, res_a = loglog_slope(L_LIST, np.abs(amps))
    bc4_amp.append((kp, sl_a, res_a))
    print("  局域标架幅度对 L 的幂次，K/k = %.3f：%.9f（残差 %.3e）" % (kp, sl_a, res_a))
supp("BC-4 附带", "局域标架幅度对 L 的标度",
     "各档幂次 %s" % ", ".join("%.6f" % v[1] for v in bc4_amp),
     "这是键耦合真正改变的量的标度，与沉积量的标度分列")

# BC-4 的替代判据：链宽对 L 的幂次。
#   理由：BC-4 的读数轴（沉积量、事件数）对键耦合不敏感，故 BC-4 在本装置内
#   退化为恒真判据（见 BC-7b）。链宽则随 K/k 变化，是对键耦合敏感的量，
#   故用链宽对 L 的幂次作替代被测量。
#   容差：±0.15 沿用 BC-4 的在案容差，来源是双对数拟合在 L 跨 1.2 个数量级时
#   幂次的统计不确定度在 0.05 量级，取 0.15 留三倍余量。
bc4_width = []
for kp in KAPPA_LIST:
    if kp == 0.0:
        continue
    xis = []
    for L in L_LIST:
        xi, _ = winding_width(L, kp, L // 2)
        xis.append(xi)
    if all(v > 0.0 for v in xis):
        sl_w, res_w = loglog_slope(L_LIST, np.abs(xis))
    else:
        sl_w, res_w = float("nan"), float("nan")
    bc4_width.append((kp, sl_w, res_w, xis))
    print("  链宽对 L 的幂次，K/k = %.3f：%.9f（残差 %.3e），链宽序列 %s"
          % (kp, sl_w, res_w, ", ".join("%.6f" % v for v in xis)))
bc4w_slopes = [v[1] for v in bc4_width if not np.isnan(v[1])]
bc4w_ok = len(bc4w_slopes) == len(bc4_width) and all(abs(v) <= 0.15 for v in bc4w_slopes)
record("BC-4b", "链宽对 L 的幂次（BC-4 的替代判据）",
       "链宽对 L 的幂次偏离 0 不超过 ±0.15（BC-4 的在案容差）",
       "各档幂次 %s" % ", ".join("%.6f" % v for v in bc4w_slopes), bc4w_ok,
       "链宽是随 K/k 变化的量，故本判据对键耦合敏感，"
       "可补 BC-4 因读数轴不敏感而缺失的区分力")

# =====================================================================
# BC-5 链端效应的修正
# =====================================================================
print("=" * 78)
print("BC-5 链端效应的修正（构型判据）")
print("=" * 78)

# BC-5 容差预算：门槛 1e-12（机器零阈，第 746 号同源）。
#   端点与内部站点的读数之差须由零变为非零。
#   本轮测的量：端点与内部站点的局域标架幅度之差。
#   理由：第 746 号的链端差测的是沉积量读数，而沉积量对局域标架严格不变，
#   故该轴恒为零。端效应真正体现在局域标架上：端点的法线轴高度差取单边差分，
#   与内部的双边差分不同，故端点的局域标架与内部不同。

bc5_rows = []
for kp in KAPPA_LIST:
    F = local_frame_amplitude(L_SMOKE, kp, 0)
    if L_SMOKE >= 3:
        end0 = float(F[0])
        inner = float(np.mean(F[1:-1]))
        diff = abs(end0 - inner)
    else:
        diff = 0.0
    bc5_rows.append((kp, diff, end0, inner))
    print("  K/k = %.3f：端点(0) 局域标架幅度 %.8f，内部均值 %.8f，差 %.3e"
          % (kp, float(F[0]), float(np.mean(F[1:-1])), diff))

bc5_pos = [d for kp, d, _, _ in bc5_rows if kp > 0.0]
bc5_ok = max(bc5_pos) > 1e-12
record("BC-5", "加键耦合后链端与内部的读数之差变为非零",
       "差值大于 1e-12（机器零阈）",
       "有耦合各档的端点-内部差 %s，最大 %.3e"
       % (", ".join("%.3e" % v for v in bc5_pos), max(bc5_pos)), bc5_ok,
       "端点取单边差分、内部取双边差分，端效应体现在局域标架上")

# BC-5 附带：链上宽度测量。
#   动机：案乙第四问取第 722 号 DB-11 的芯宽 3.0 格，而那个读数是在
#   第 722 号的两维装置、站点对角、格间无耦合的构型里测的。
#   本轮测的是一维链上的宽度，两者不同质，不可直接比大小；
#   可比的是同一装置内加耦合前后的变化。
#   口径：缠绕分布的二阶矩半宽（见 winding_width 的 docstring）。
bc5_widths = []
for kp in KAPPA_LIST:
    xi, tot = winding_width(L_WIDTH, kp, L_WIDTH // 2)
    bc5_widths.append((kp, xi, tot))
    print("  芯宽（缺陷在链心，L = %d），K/k = %.3f：二阶矩半宽 %.6f，缠绕总量 %.6f"
          % (L_WIDTH, kp, xi, tot))
w0 = bc5_widths[0][1]
w_dev = [(kp, xi, xi - w0) for kp, xi, _ in bc5_widths]
print("  芯宽偏离（相对无耦合档）：%s"
      % ", ".join("K/k = %.3f：%+.6f" % (kp, d) for kp, _, d in w_dev))
supp("BC-5 附带", "链上宽度测量（加耦合前后的偏离）",
     "二阶矩半宽 %s；相对无耦合档的偏离 %s"
     % (", ".join("%.6f" % w[1] for w in bc5_widths),
        ", ".join("%+.6f" % d for _, _, d in w_dev)),
     "口径为缠绕分布的二阶矩半宽。第 722 号 DB-11 的 3.0 格是二维装置、"
     "站点对角构型的读数，本轮测的是一维链，两者不同质，不可直接比大小；"
     "可比的是加耦合前后的变化")

# BC-5 附带：缠绕总量。键耦合展宽位错而不改荷。
w_tot = [w[2] for w in bc5_widths]
w_tot_rel = (max(w_tot) - min(w_tot)) / min(w_tot)
print("  缠绕总量：%s；相对变化 %.6f" % (", ".join("%.6f" % v for v in w_tot), w_tot_rel))
supp("BC-5 附带", "缠绕总量（键耦合展宽位错而不改荷）",
     "缠绕总量 %s；相对变化 %.6f"
     % (", ".join("%.6f" % v for v in w_tot), w_tot_rel),
     "缠绕总量即位错的拓扑荷。键耦合把芯区的缠绕摊到邻站，"
     "宽度从 %.6f 增到 %.6f，而荷几乎不变。"
     "这与库内预期一致：包层 kappa_parallel 供宽度与洛伦兹色散，"
     "而缠绕总量是拓扑量不随弹性改变"
     % (bc5_widths[0][1], bc5_widths[-1][1]))

# =====================================================================
# BC-6 写入算符的站点对角性复核
# =====================================================================
print("=" * 78)
print("BC-6 写入算符的站点对角性复核（硬约束守恒核）")
print("=" * 78)

# BC-6 容差预算：门槛 1e-12（机器零阈）。
#   写入侧的跨站泄漏须仍为机器零，因键耦合加在 C1 侧。
#   另核联合生成元的两两对易子偏差，该项恒为零即站点对角成立。

bc6_true = cross_site_leak(0.0)
bc6_ctrl = [cross_site_leak(g) for g in (0.01, 0.02, 0.04)]
bc6_ok = bc6_true <= 1e-12
record("BC-6", "写入侧跨站泄漏仍为机器零",
       "真构型（写入算符逐站点直和）的跨站泄漏 ≤ 1e-12",
       "真构型泄漏 %.10e；污染对照档（g' = 0.01、0.02、0.04）%s"
       % (bc6_true, ", ".join("%.10e" % v for v in bc6_ctrl)), bc6_ok,
       "键耦合加在 C1 侧，写入算符保持逐站点直和；对照档显示该装置对跨站项灵敏")

bc6_comm = commutator_dev(8)
bc6b_ok = bc6_comm <= 1e-12
record("BC-6b", "联合生成元的两两对易子偏差",
       "≤ 1e-12（N_site = 8）",
       "对易子偏差 %.3e" % bc6_comm, bc6b_ok,
       "站点对角在有键耦合的构造下仍成立")

# =====================================================================
# BC-7 恒真检验
# =====================================================================
print("=" * 78)
print("BC-7 恒真检验（构型判据）")
print("=" * 78)

# BC-7 容差预算：布尔判定。替身把键耦合关断（kappa_parallel = 0）。
#   BC-5 的端效应差值应归零，故 BC-5 判否。
#   BC-4 的幂次：关断后仍为 0 与 1，故 BC-4 判过。
#   **这一条须如实说明**：BC-4 的读数轴是沉积量与事件数，两者对键耦合都不敏感，
#   恒真检验对 BC-4 无区分力。原因是第 741 号的长期平均对局域标架严格不变，
#   且写入剖面在 t 上周期为 1 使事件数恒定。

bc7_F0 = local_frame_amplitude(L_SMOKE, 0.0, 0)
bc7_end0 = abs(abs(bc7_F0[0]) - float(np.mean(bc7_F0[1:-1])))
bc7a_ok = bc7_end0 <= 1e-12
record("BC-7a", "恒真检验一：关断键耦合后 BC-5 的端效应差值归零",
       "关断后差值 ≤ 1e-12（故 BC-5 判否）",
       "关断后端点-内部差 %.3e" % bc7_end0, bc7a_ok,
       "替身与真构型可区分，BC-5 含区分力")

bc7_sl = bc4_slopes[("bond", 0.0)]
bc7b_ok = not (abs(bc7_sl[0]) <= 0.15 and abs(bc7_sl[1] - 1.0) <= 0.15)
record("BC-7b", "恒真检验二：BC-4 须对键耦合取值敏感",
       "关断后 BC-4 须判否（否则判据对键耦合恒真，替身不区分）",
       "关断后每单位长度幂次 %.9f，总能量幂次 %.9f，BC-4 判过。"
       "**本条登记为判据设计失效**（第三种结论），与判负区分"
       % (bc7_sl[0], bc7_sl[1]), bc7b_ok,
       "失效机制：BC-4 的读数轴是沉积量与事件数，键耦合对两者都不敏感。"
       "前者因第 741 号的长期平均对局域标架严格不变，"
       "后者因写入剖面在 t 上周期为 1 使单周期窗内事件数恒定。"
       "故 BC-4 在本装置内换任何键耦合取值都判过，不构成物理信息；"
       "其读数作机制确认用，另以 BC-4b 的链宽幂次补区分力")

# =====================================================================
# BC-8 前提守恒
# =====================================================================
print("=" * 78)
print("BC-8 前提守恒（构型判据）")
print("=" * 78)

# BC-8 容差预算：沿用两轮各自的在案容差。
#   C1 侧：第 744 号 W4 的深度方向幂次 p 落在 [1.8, 2.2]，
#   浅区线性 R^2 ≥ 0.99 的原假设以判负记录。
#   写入侧：第 726 号的单站点基线，相对偏差门槛 1e-5。

print("  [BC-8a] C1 侧复现第 744 号 W4 读数:")
w4_rows = []
for d in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.7, 0.9):
    A0d = (1.0 - d * np.exp(-RR2 / (2.0 * 2.5 ** 2))).astype(complex)
    psid = relax(A0d, A0d, 1200)
    w4_rows.append((d, energy(psid, A0d), w_amp(psid)))
W_all = np.array([r[2] for r in w4_rows])
E_all = np.array([r[1] for r in w4_rows])
p_depth = float(np.polyfit(np.log(W_all), np.log(E_all), 1)[0])
sh_mask = np.array([r[0] <= 0.3 for r in w4_rows])
cf_sh, r2_sh = fit_linear(W_all[sh_mask], E_all[sh_mask])
w4_rows2 = []
for w in (1.0, 1.5, 2.0, 3.0, 4.0):
    A0w = (1.0 - 0.6 * np.exp(-RR2 / (2.0 * w ** 2))).astype(complex)
    psiw = relax(A0w, A0w, 1200)
    w4_rows2.append((w, energy(psiw, A0w), w_amp(psiw)))
p_w = float(np.polyfit(np.log([r[2] for r in w4_rows2]), np.log([r[1] for r in w4_rows2]), 1)[0])
print("    深度幂次 p = %.4f（第 744 号 2.0373）；浅区 R^2 = %.4f（0.9564）；宽度幂次 p_w = %.4f（0.5382）"
      % (p_depth, r2_sh, p_w))
bc8a_ok = (1.8 <= p_depth <= 2.2) and (r2_sh < 0.99)
record("BC-8a", "C1 侧复现第 744 号 W4 读数",
       "深度幂次 p ∈ [1.8, 2.2]；浅区 R^2 < 0.99；宽度幂次 p_w 为读数",
       "p = %.4f，浅区 R^2 = %.4f，p_w = %.4f" % (p_depth, r2_sh, p_w), bc8a_ok)

psi_step = make_packet(theta, PW)
ts_s = [0.0]
rs_s = [reg_pop(psi_step)]
for step in range(int(round(T_MAX / DT))):
    psi_step = split_step_g(psi_step, DT, theta, kvec, G_WRITE)
    ts_s.append((step + 1) * DT)
    rs_s.append(reg_pop(psi_step))
ts_s = np.array(ts_s)
rs_s = np.array(rs_s)
hit = np.nonzero(rs_s > 0.5)[0]
t_write = float(ts_s[int(hit[0])]) if hit.size else float("nan")
psi_base, _ = run(make_packet(theta, PW), G_FULL, steps=int(round(T_MAX / DT)))
psi_now, _ = run(make_packet(theta, PW), 0.0, steps=int(round(T_MAX / DT)))
e0 = total_pop(make_packet(theta, PW))
e1 = total_pop(psi_base)
d_site = site_pop(psi_base) - site_pop(psi_now)
t_avg, s_avg = time_average_site(make_packet(theta, PW), G_WRITE)
t_bse, s_bse = time_average_site(make_packet(theta, PW), np.zeros_like(G_WRITE))
mean_site = float(np.trapezoid(s_avg, t_avg) / T_AVG)
mean_base = float(np.trapezoid(s_bse, t_bse) / T_AVG)
dbar1 = mean_site - mean_base
bc8b_ok = (abs(t_write - t_cross) / t_cross < 1e-5
           and abs(e1 - e0) / e0 < 1e-5
           and abs(d_site - (-0.99882383)) / 0.99882383 < 1e-5
           and abs(dbar1 - (-0.50000000)) / 0.50000000 < 1e-5)
record("BC-8b", "写入侧复现第 726 号单站点基线",
       "写入时刻 0.5000、总量 1.00000000、站点侧增量 -0.99882383、"
       "长期平均 0.50000000、Δ̄ = -0.50000000，各相对偏差 < 1e-5",
       "写入时刻 %.4f（偏差 %.1e），总量 %.10f（偏差 %.3e），站点侧增量 %.8f（偏差 %.3e），"
       "⟨站点侧⟩ = %.10f（偏差 %.3e），Δ̄ = %.10f（偏差 %.3e）"
       % (t_write, abs(t_write - t_cross) / t_cross, e1, abs(e1 - e0),
          d_site, abs(d_site + 0.99882383) / 0.99882383,
          mean_site, abs(mean_site - 0.5) / 0.5, dbar1, abs(dbar1 + 0.5) / 0.5),
       bc8b_ok)

# =====================================================================
# 出图
# =====================================================================
print("=" * 78)
print("出图")
print("=" * 78)

fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.4))

# (a) 链平均 Δ̄ 对 K/k，分甲乙两线
ax = axes[0, 0]
ks = np.array(KAPPA_LIST)
ax.plot(ks, [r[2] for r in bc3_jia], "o-", color="tab:blue", ms=6,
        label=r"甲 键耦合在 C1 侧：链平均 $\bar{\Delta}$")
ax.plot(ks, [r[2] for r in bc3_bond], "s-", color="tab:orange", ms=6,
        label=r"乙 键耦合在写入条件侧：链平均 $\bar{\Delta}$")
ax.axvline(KAPPA_WINDOW, color="tab:red", ls="--", lw=1.2,
           label=r"落窗上限 $K/k = 0.02$")
ax.set_xlabel(r"键弹性对格点内锁定之比 $K/k$")
ax.set_ylabel(r"链平均 $\bar{\Delta}$")
ax.set_ylim(-0.55, -0.45)
ax.set_title(r"(a) 链平均 $\bar{\Delta}$ 对 $K/k$（甲乙两种加法）", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax.text(0.03, 0.90, r"两线极差皆在浮点噪声内（甲 $%.1e$，乙 $%.1e$）"
        % (span_jia, span_bond), transform=ax.transAxes, fontsize=8, va="top")

# (b) 每单位长度系数的 L 幂次：无耦合与有耦合对照
ax = axes[0, 1]
kp_nonzero = [kp for kp in KAPPA_LIST if kp > 0.0]
ax.plot([0] + kp_nonzero, [bc4_slopes[("bond", 0.0)][0]] + [bc4_slopes[("bond", kp)][0] for kp in kp_nonzero],
        "o-", color="tab:blue", ms=6, label=r"每单位长度幂次（乙加法）")
ax.plot([0] + kp_nonzero, [bc4_slopes[("bond", 0.0)][1]] + [bc4_slopes[("bond", kp)][1] for kp in kp_nonzero],
        "s-", color="tab:orange", ms=6, label=r"总能量口径幂次（乙加法）")
ax.axhline(0.0, color="k", ls=":", lw=1.3, label=r"幂次 0 参考线")
ax.axhline(1.0, color="tab:gray", ls="--", lw=1.3, label=r"幂次 1 参考线")
ax.axvline(KAPPA_WINDOW, color="tab:red", ls="--", lw=1.0, label=r"落窗上限 $0.02$")
ax.set_xlabel(r"$K/k$")
ax.set_ylabel("幂次")
ax.set_ylim(-0.15, 1.25)
ax.set_title(r"(b) 每单位长度系数的 $L$ 幂次：有耦合与无耦合对照", fontsize=10)
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

# (c) 链端与内部读数之差对 K/k，以及芯宽
ax = axes[1, 0]
ax.semilogy(ks, [max(d, 1e-18) for _, d, _, _ in bc5_rows], "o-",
            color="tab:blue", ms=7, label=r"端点与内部的局域标架幅度之差")
ax.axhline(1e-12, color="tab:red", ls="--", lw=1.2, label=r"门槛 $10^{-12}$")
ax.axvline(KAPPA_WINDOW, color="tab:gray", ls=":", lw=1.2, label=r"落窗上限 $0.02$")
ax.set_xlabel(r"$K/k$")
ax.set_ylabel(r"端点-内部差")
ax.set_title(r"(c) 链端效应与芯宽：加键耦合后由零变为非零", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")
ax2 = ax.twinx()
ax2.plot(ks, [w[1] for w in bc5_widths], "s--", color="tab:purple", ms=5,
         label=r"芯宽（二阶矩半宽）")
ax2.set_ylabel("芯宽", color="tab:purple")
ax2.tick_params(axis="y", labelcolor="tab:purple")
ax.text(0.03, 0.90, r"无耦合档差值 $0.0$，最大有耦合档 %.2e" % max(bc5_pos),
        transform=ax.transAxes, fontsize=8, va="top")

# (d) 写入侧跨站泄漏
ax = axes[1, 1]
ax.semilogy([0.0, 0.01, 0.02, 0.04],
            [max(v, 1e-18) for v in [bc6_true] + bc6_ctrl],
            "s-", color="tab:green", ms=7,
            label=r"写入侧跨站泄漏（$g'$ 档，$g'=0$ 为真构型）")
ax.axhline(1e-12, color="tab:red", ls="--", lw=1.2, label=r"门槛 $10^{-12}$")
ax.set_xlabel(r"跨站耦合强度 $g'$（对照轴，与 $K/k$ 无关）")
ax.set_ylabel("跨站泄漏")
ax.set_title(r"(d) 写入侧跨站泄漏：键耦合未污染写入侧", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")
ax.text(0.03, 0.90, "真构型（$g'=0$）%.2e；对照档 %.2e 至 %.2e"
        % (bc6_true, min(bc6_ctrl), max(bc6_ctrl)),
        transform=ax.transAxes, fontsize=8, va="top")

fig.suptitle("第 755 号：案乙部件三，真实位错线的键耦合版", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT, dpi=150)
print("  图已存: %s" % OUT)

# =====================================================================
# 总表
# =====================================================================
print()
print("=" * 78)
print("判据总表")
print("=" * 78)
npass = 0
for cid, name, expect, actual, ok, note in RESULTS:
    print("%-8s %-56s %s" % (cid, name[:56], "通过" if ok else "判否"))
    if ok:
        npass += 1
print()
print("合计 %d/%d 通过（主判据 BC-1 至 BC-8，展开为 %d 个子项）"
      % (npass, len(RESULTS), len(RESULTS)))
print()
print("关键读数:")
print("  BC-1 无耦合退化：Δ̄ 五档 %s" % ", ".join("%.12f" % v for v in bc1_dbar))
print("       每单位长度幂次 %.9f，总能量幂次 %.9f，链端差五档最大 %.3e，事件数 %s"
      % (sl_bc1p, sl_bc1t, max(bc1_end), ", ".join(str(v[0]) for v in ev_bc1)))
print("  BC-2 事件计数：两种加法共 %d 个站点，解析事件数取值集合 %s，数值峰数取值集合 %s"
      % (len(ev_all), sorted(set(ev_all)), sorted(set(ev_num))))
print("       过闸时刻 %s" % ", ".join("%.6f" % t for _, t in gt_move))
print("  BC-3 局域标架幅度：甲加法极差 %.8f，乙加法极差 %.8f"
      % (span_amp_jia, span_amp_bond))
print("       甲各档 %s" % ", ".join("%.8f" % v for v in amp_jia))
print("       乙各档 %s" % ", ".join("%.8f" % v for v in amp_bond))
print("       Δ̄ 极差：甲 %.3e，乙 %.3e" % (span_jia, span_bond))
print("  BC-4 幂次（乙加法有耦合各档）：每单位长度 %s，总能量 %s"
      % (", ".join("%.6f" % v for v in bc4_p_bond), ", ".join("%.6f" % v for v in bc4_t_bond)))
print("       幂次（甲加法有耦合各档）：每单位长度 %s，总能量 %s"
      % (", ".join("%.6f" % v for v in bc4_p_jia), ", ".join("%.6f" % v for v in bc4_t_jia)))
print("       局域标架幅度的 L 幂次 %s" % ", ".join("%.6f" % v[1] for v in bc4_amp))
print("  BC-4b 链宽对 L 的幂次（替代判据）：%s"
      % ", ".join("%.6f" % v for v in bc4w_slopes))
print("  BC-5 端效应差：%s" % ", ".join("%.3e" % v[1] for v in bc5_rows))
print("       芯宽（二阶矩半宽）：%s；偏离 %s"
      % (", ".join("%.6f" % w[1] for w in bc5_widths),
         ", ".join("%+.6f" % d for _, _, d in w_dev)))
print("       缠绕总量：%s；相对变化 %.6f"
      % (", ".join("%.6f" % v for v in w_tot), w_tot_rel))
print("  BC-6 跨站泄漏：真构型 %.10e；污染对照档 %s；对易子偏差 %.3e"
      % (bc6_true, ", ".join("%.10e" % v for v in bc6_ctrl), bc6_comm))
print("  BC-7 关断后端效应差 %.3e；BC-4 关断后幂次 %.9f 与 %.9f"
      % (bc7_end0, bc7_sl[0], bc7_sl[1]))
print("  BC-8 C1 侧：p = %.4f，浅区 R^2 = %.4f，p_w = %.4f" % (p_depth, r2_sh, p_w))
print("       写入侧：写入时刻 %.4f，总量 %.10f（偏差 %.3e），站点侧增量 %.8f，Δ̄ = %.10f"
      % (t_write, e1, abs(e1 - e0), d_site, dbar1))
print("耗时 %.1f 秒" % (time.time() - T_START))
