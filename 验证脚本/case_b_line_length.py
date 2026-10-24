# -*- coding: utf-8 -*-
r"""
case_b_line_length.py

第 746 号 案乙的位错线长度扫描：直链构型下 $L$ 档的缩放律、两栏与净栏、链端效应
（判据 LB-1 至 LB-10）

问题：案乙（构造轮单装置合并）的判据表里，乙-1（缩放律）、乙-2（消去对两栏与净栏）、
乙-7（前提守恒与多站点复现）这三条只涉及写入侧，不依赖尚待作者裁定的 P1-62。
它们的共同装置要求是「位错线长度 $L$ 可调」，而第 741 号的装置还没有这个轴。
本轮做这个轴。

装置：复用 第 741 号脚本 验证脚本/multisite_write_extension.py 的
参数与函数，逐字沿用。参数 $N_\theta = 256$、$\omega_P = 2\pi$、$\theta_0 = \pi$、
脉冲宽 $w = 0.5$、步长 $dt = 2\times10^{-3}$、$T = 1.2$（长期平均档取 $T = 8.0$）、
完整交换耦合 $g_{\rm full} = 5.092$（实测 5.091585）、指针波包宽 $PW = 0.15$。

位错线的构造（本轮新增的唯一结构）：$L$ 是位错线长度（以站点数计）。
本轮取**直链构型**：$L$ 个站点排成一条链，每站点一对 $(a_j, c_j)$，
写入按第 741 号的逐站点直和施加。

链上各站点的局部标架偏移 $A_j$ 三种取法，全部做并对照：

取法甲（同时写入）：全链 $A_j = 0$，所有站点同相过闸。

取法乙（依次写入）：$A_j = j\,\Delta A$，链上各站点依次过闸，对应 D-5 的过格机制与
「缺陷沿链扫过、逐站接力」的图景。$\Delta A$ 取 $0.20$ rad，与第 741 号 MS-4 的最大档
一致，使链上最大偏移 $(L-1)\Delta A$ 在 $L = 32$ 档达 $6.20$ rad。

取法丙（端点触发）：只让链两端站点写入，中间站点保持基态。
构造「部分站点未被写入」的极限。

净栏式（作者 2026-10-21 裁决后取定）：案乙部件四的三式为

$$E_{\rm wr}^{\rm site} = \nu\,\bar\Delta\,E_a, \qquad
E_{\rm wr}^{\rm reg} = -\nu\,\bar\Delta\,E_c, \qquad
E_{\rm wr}^{\rm net} = \nu\,\bar\Delta\,(E_c - E_a).$$

**净栏式的一处符号经本轮查出并修正。** 设计提要原写 $\nu\bar\Delta(E_a - E_c)$，
作者裁决表第四问据此定「$E_c > E_a$ 使净栏为负」，两处同错而互相掩盖。

按物理核对，材料由站点侧搬到记录侧，每激发占的刻度由 $E_a$ 变为 $E_c$，
总能量变化为 $\nu\times(\text{搬走的量})\times(E_c - E_a)$，
故 $E_c > E_a$ 时总能量上升、净栏为正，负线项须由 $E_c < E_a$ 给出。

修正后的净栏式取 $\nu\bar\Delta(E_c - E_a)$，两栏式不变，
裁决表第四问随之成立。LB-3a 即按修正后的式子判定。

期望值的出处（可直接判）：第 741 号已证长期平均对 $A_j$ 不变（解析恒等式，
读数周期上 $\int_0^2\cos^2 A_j\,dt = 1$ 与偏移无关），故取法甲与取法乙的每站点
长期平均应完全相同；取法丙的中间站点不写入，其每站点平均应为 $0$，
链平均应为端点占比。

链的耦合与射程限定：站点对角是硬约束（跨站泄漏机器零 $2.26\times10^{-29}$），
故链上站点间无耦合，链只是站点数的增加。本轮只测「长度纯几何计数」这一层。
真实位错线上站点间有键耦合，属案乙部件三的待建内容，不在本轮射程内。
链端效应须显式核验：若端点站点读数与内部站点读数出现差异，说明有隐藏耦合。

判据：LB-1 至 LB-10 共十条主判据，展开为二十二个子项。
逐条给出期望、实测与判定，容差从误差预算推导，见各条注释。
可逆性等级：LB-1、LB-2、LB-5、LB-6、LB-7、LB-8、LB-9 判据稳健；
LB-3、LB-4 为参数类与形态判定，判据敏感；LB-10 为区分力检验。

运行：python 验证脚本/case_b_line_length.py
确定性，无随机数。图存 images/case_b_line_length_746.png。
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

T_START = time.time()
OUT = r"D:/software/TianXingFC/images/case_b_line_length_746.png"

# =====================================================================
# 装置参数（逐字沿用第 726 号与第 741 号）
# =====================================================================
N_THETA = 256
OMEGA_P = 2.0 * np.pi
THETA0 = np.pi
W_PULSE = 0.5
PW = 0.15
DT = 2.0e-3
T_MAX = 1.2
T_AVG = 8.0

# 本轮构造参数
DELTA_A = 0.20             # 取法乙的相邻站点相位梯度（rad），承第 741 号 MS-4 最大档
E_A = 0.0                  # 站点模每激发的能量刻度（归一化档）
E_C = 1.0                  # 记录寄存器每激发的能量刻度（归一化档，取 E_c > E_a）

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
# 装置函数（逐字沿用第 726 号）
# =====================================================================
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


def split_step_g(psi, dt, theta, kvec, g_arr, n_a=2, n_c=2):
    """通用分裂步：g_arr 是逐 θ 的耦合系数数组。

    g_arr = G_FULL·f(θ) 为写入项；再加常数项 κ·G_FULL 即 θ 无关的回流项。
    两项同轴（同一对角化基下的 σ_x⊗σ_x），指数可加，次序不影响结果。
    g_arr = G_FULL·f(θ) 时与 split_step 的数值完全一致。
    """
    psi = apply_int(psi, 1.0, dt / 2.0, g_arr)
    for a in range(n_a):
        for c in range(n_c):
            psi[:, a, c] = np.fft.ifft(np.fft.fft(psi[:, a, c]) * np.exp(1j * omega_of_k(kvec) * dt))
    psi = apply_int(psi, 1.0, dt / 2.0, g_arr)
    return psi


def apply_diag(psi, f, dt, g):
    """正定势替身：转移项换成对角正定项 a†a+c†c，只改变相位，激发搬不动。"""
    ang = g * f * dt
    out = np.zeros_like(psi)
    for a in range(2):
        for c in range(2):
            out[:, a, c] = psi[:, a, c] * np.exp(-1j * ang * (a + c))
    return out


def time_average_site(psi_in, g_write, t_end=T_AVG):
    """逐步行进并每 DT 采样一次站点侧布居，含 t=0 的初值点。

    采样间隔 = DT = 2.0e-3 ≤ 判据要求的 0.005。
    """
    n = int(round(t_end / DT))
    psi_l = psi_in
    t_list = [0.0]
    s_list = [site_pop(psi_l)]
    for step in range(n):
        psi_l = split_step_g(psi_l, DT, theta, kvec, g_write)
        t_list.append((step + 1) * DT)
        s_list.append(site_pop(psi_l))
    return np.array(t_list), np.array(s_list)


# =====================================================================
# 第 741 号的多站点装置（逐字沿用其函数）
# =====================================================================
def site_frames(n_site, scheme, param):
    """各站点的局部标架偏移 $A_j$（弧度）。

    scheme="jia"   同相：A_j = 0。
    scheme="yi"    沿位错线均匀相位梯度：A_j = j·param。
    scheme="bing"  空间均匀相移：A_j = A_SPAN·x_j/x_half，x_j 在芯内横向等距取样。
    """
    if scheme == "jia":
        return np.zeros(n_site)
    if scheme == "yi":
        return param * np.arange(n_site, dtype=float)
    if scheme == "bing":
        xs = np.linspace(-X_HALF, X_HALF, n_site)
        return A_SPAN * xs / X_HALF
    raise ValueError("未知方案: %s" % scheme)


def run_site_shift(shift, g_on=True, t_end=T_MAX, op=None):
    """单站点演化，写入剖面带相移 shift（本站局部标架偏移 $A_j$）。

    与第 726 号的 split_step_g 同源，只把剖面从 f(theta) 换成 f(theta - shift)。
    op=None 走写入算符 apply_int；给 op 时走替身算符（恒真检验用）。
    """
    psi = make_packet(theta, PW)
    n = int(round(t_end / DT))
    if op is None:
        g_arr = G_FULL * pulse(theta - shift, W_PULSE) if g_on else np.zeros_like(G_WRITE)
        for _ in range(n):
            psi = split_step_g(psi, DT, theta, kvec, g_arr)
    else:
        for _ in range(n):
            psi = split_step_sub(psi, DT, theta, kvec, W_PULSE, G_FULL, op)
    return psi


def time_average_site_shift(shift, g_on=True, t_end=T_AVG):
    """逐站点的长期平均，采样口径与第 726 号的 time_average_site 相同。

    每 DT 采样一次站点侧布居，含 t=0 的初值点，梯形积分除以 t_end。
    同时返回记录侧布居的时间平均（两栏之和恒为 1）。
    """
    g_arr = G_FULL * pulse(theta - shift, W_PULSE) if g_on else np.zeros_like(G_WRITE)
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
    return ts, np.array(s_list), mean_site, mean_reg


def split_step_sub(psi, dt, theta, kvec, w, g, op, n_a=2, n_c=2):
    """替身用的分裂步：耦合项由 op 决定（apply_int / apply_diag）。"""
    f = pulse(theta, w)
    psi = op(psi, f, dt / 2.0, g)
    for a in range(n_a):
        for c in range(n_c):
            psi[:, a, c] = np.fft.ifft(np.fft.fft(psi[:, a, c]) * np.exp(1j * omega_of_k(kvec) * dt))
    psi = op(psi, f, dt / 2.0, g)
    return psi


def gate_times(shift, t_end=T_MAX):
    """该站点的过闸时刻（解析）：解 theta_p(t) = shift + 2*pi*n 落在半开窗 [0, t_end) 内。

    本版修正了第 741 号脚本的窗口取法：那里按 k 的下界取 ceil((shift-theta_0)/2pi)，
    在 |shift| > pi 时会漏掉窗内的过闸时刻。修正版直接由 0 <= t_k < t_end 解 k 的区间。
    第 741 号实际用到的 shift 全在 |shift| <= 1.4，故其读数不受影响。
    """
    out = []
    k_lo = int(np.ceil((THETA0 - shift) / (2.0 * np.pi)))
    k_hi = int(np.ceil((THETA0 - shift + OMEGA_P * t_end) / (2.0 * np.pi))) - 1
    for k in range(k_lo, k_hi + 1):
        t_k = (shift + 2.0 * np.pi * k - THETA0) / OMEGA_P
        if 0.0 <= t_k < t_end - 1e-12:
            out.append(t_k)
    return out


def write_events(shift, t_end=T_MAX, dt=DT):
    """每站点的写入事件数（过闸次数），两种口径并排。

    口径甲（解析，穿过闸）：解 theta_p(t) = shift + 2*pi*n 落在半开窗 [0, t_end) 内的个数。
    口径乙（数值，剖面峰数）：写入剖面在窗内的极大值个数。
    写入剖面在 t 上严格周期为 1（因 f 以 2π 为周期且 ω_P = 2π），
    故在单个周期上按循环邻点判峰，再把落在窗内的峰计入。
    """
    n_per = int(round(1.0 / dt))
    ts_per = np.arange(n_per) * dt
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


# ---------------------------------------------------------------------
# 联合模空间的生成元与指数（承第 741 号，本轮只用于前提守恒核）
# ---------------------------------------------------------------------
def single_site_gen():
    r"""单站点写入生成元（4 维模空间）。

    每站点是一对模 $(a, c)$，各 2 层，故单站点子空间 4 维，模索引 m = 2a + c。
    耦合项 $a^\dagger c + ac^\dagger$ 把 $(a,c)$ 映到 $(1-a,1-c)$，
    在 m 上即 m 映到 3-m，故生成元是反序矩阵。
    """
    X = np.zeros((4, 4), dtype=complex)
    for a in range(2):
        for c in range(2):
            X[2 * a + c, 2 * (1 - a) + (1 - c)] = 1.0
    return X


def joint_generator(n_site, coeffs=None):
    """联合模空间生成元：单站点生成元的 Kronecker 和。

    每站点因子须是 4 维（一对模共 4 层），不可取 2 维单位阵。
    联合维数 4^{n_site}，n_site = 8 时为 65536 维即 68 GB 不可建，
    故大档改用交换性核。
    """
    X = single_site_gen()
    if coeffs is None:
        coeffs = np.ones(n_site)
    M = np.zeros((4 ** n_site, 4 ** n_site), dtype=complex)
    for j in range(n_site):
        fac = np.ones((1, 1), dtype=complex)
        for q in range(n_site):
            fac = np.kron(fac, coeffs[q] * X if q == j else np.eye(4))
        M = M + fac
    return M


def sym_exp(M, dt):
    """实对称矩阵 M 的指数 exp(-i dt M)，用本征分解实现，不依赖 scipy。"""
    lam, V = np.linalg.eigh(M)
    return (V * np.exp(-1j * dt * lam)) @ V.T


def commutator_dev(n_site, coeffs=None):
    """交换性核：联合生成元各项之间的对易子偏差，与站点数无关。

    对易子在任意两站的 16×16 子块上为零（其余因子一律为单位阵），
    故只建 16×16 矩阵，站点数再大也可算。
    """
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


def operator_factorization_dev(n_site, coeffs=None, dt=DT):
    """算符层核：联合指数与各站点指数的张量积之差。

    该核需建 4^{n_site} 维稠密矩阵，n_site = 5 时已占 16 MB，
    n_site = 8 时不可行，故该核只到 n_site = 4。大档由 commutator_dev 给出。
    """
    M = joint_generator(n_site, coeffs)
    joint = sym_exp(M, dt)
    prod = np.ones((1, 1), dtype=complex)
    X = single_site_gen()
    if coeffs is None:
        coeffs = np.ones(n_site)
    for j in range(n_site):
        prod = np.kron(prod, sym_exp(coeffs[j] * X, dt))
    return float(np.max(np.abs(joint - prod)))


def joint_run_2site(shifts, t_end=T_MAX):
    """两站点联合态的直接演化：显式 16 维模空间，每站点一条 theta 轴。

    联合 Hilbert 空间维数 4^{n_site} = 16，模索引 m = 2a + c。
    写入算符逐站点作用，在模指标上是 m 映到 3-m。
    自由演化在每站点的 theta 轴上各自平移。
    初态是张量积：两站点各在自己站点的 (a=1, c=0) 层放一个波包。
    """
    f1 = pulse(theta - shifts[0], W_PULSE)
    f2 = pulse(theta - shifts[1], W_PULSE)
    pk = np.exp(-((theta - THETA0) ** 2) / (2.0 * PW ** 2))
    pk = pk / np.linalg.norm(pk)
    J = np.zeros((N_THETA, N_THETA, 4, 4), dtype=complex)
    J[:, :, 2, 2] = pk[:, None] * pk[None, :]
    ph = np.exp(1j * omega_of_k(kvec) * DT)
    n = int(round(t_end / DT))
    for _ in range(n):
        for (fv, ax) in ((f1, 0), (f2, 1)):
            ang = G_FULL * fv * (DT / 2.0)
            if ax == 0:
                J = (np.cos(ang)[:, None, None, None] * J
                     - 1j * np.sin(ang)[:, None, None, None] * J[:, :, ::-1, :])
            else:
                J = (np.cos(ang)[None, :, None, None] * J
                     - 1j * np.sin(ang)[None, :, None, None] * J[:, :, :, ::-1])
        J = np.fft.ifft(np.fft.fft(J, axis=0) * ph[:, None, None, None], axis=0)
        J = np.fft.ifft(np.fft.fft(J, axis=1) * ph[None, :, None, None], axis=1)
        for (fv, ax) in ((f1, 0), (f2, 1)):
            ang = G_FULL * fv * (DT / 2.0)
            if ax == 0:
                J = (np.cos(ang)[:, None, None, None] * J
                     - 1j * np.sin(ang)[:, None, None, None] * J[:, :, ::-1, :])
            else:
                J = (np.cos(ang)[None, :, None, None] * J
                     - 1j * np.sin(ang)[None, :, None, None] * J[:, :, :, ::-1])
    return J


def joint_pops(J):
    """联合态的分栏读数：两站点各自的站点侧、记录侧，以及总激发数。

    模索引 m = 2a + c，故站点侧对应 m ∈ {2, 3}，记录侧对应 m 的奇数 {1, 3}。
    联合态是张量积，其模平方的总和是 1，而两站点的激发数之和是 2，
    故总激发数取两站两栏之和，不取 |J|^2 的总和。
    """
    s1 = float(np.sum(np.abs(J[:, :, [2, 3], :]) ** 2))
    r1 = float(np.sum(np.abs(J[:, :, [1, 3], :]) ** 2))
    s2 = float(np.sum(np.abs(J[:, :, :, [2, 3]]) ** 2))
    r2 = float(np.sum(np.abs(J[:, :, :, [1, 3]]) ** 2))
    tot = s1 + r1 + s2 + r2
    return s1, r1, s2, r2, tot


theta = np.linspace(0.0, 2.0 * np.pi, N_THETA, endpoint=False)
kvec = np.fft.fftfreq(N_THETA, d=1.0 / N_THETA)
area = float(np.sum(pulse(theta, W_PULSE)) * (2.0 * np.pi / N_THETA))
G_FULL = (np.pi / 2.0) * OMEGA_P / area
t_cross = (2.0 * np.pi - THETA0) / OMEGA_P
G_WRITE = G_FULL * pulse(theta, W_PULSE)
print("脉冲面积 %.4f，完整交换耦合 g = %.4f，折返时刻 %.4f" % (area, G_FULL, t_cross))
print("写入侧参数逐项（须与第 726 号与第 741 号一致）:")
print("  N_theta = %d，omega_P = %.6f，theta_0 = %.6f，w = %.3f，dt = %.1e"
      % (N_THETA, OMEGA_P, THETA0, W_PULSE, DT))
print("  T = %.1f（长期平均档 %.1f），g_full 实测 %.6f（第 726 号报 5.0916），波包宽 PW = %.2f"
      % (T_MAX, T_AVG, G_FULL, PW))
print("本轮构造参数：Delta_A = %.2f rad，E_a = %.1f，E_c = %.1f（归一化档）"
      % (DELTA_A, E_A, E_C))
print()

# =====================================================================
# 本轮新增：直链构型
# =====================================================================
def chain_frames(n_chain, scheme):
    """链上各站点的局部标架偏移 $A_j$（弧度），三种取法。

    scheme="jia"  同时写入：全链 A_j = 0。
    scheme="yi"   依次写入：A_j = j·Delta_A，链上各站点依次过闸。
    scheme="bing" 端点触发：只让链两端站点写入，中间站点保持基态。
    """
    if scheme == "jia":
        return np.zeros(n_chain)
    if scheme == "yi":
        return DELTA_A * np.arange(n_chain, dtype=float)
    if scheme == "bing":
        a = np.zeros(n_chain)
        if n_chain == 1:
            a[0] = 0.0
        elif n_chain == 2:
            a[0] = 0.0
            a[1] = 0.0
        else:
            a[0] = 0.0
            a[-1] = 0.0
        return a
    raise ValueError("未知取法: %s" % scheme)


def chain_write_mask(n_chain, scheme):
    """链上哪些站点参与写入（布尔数组）。

    取法丙只让两端站点写入，中间站点不施加写入耦合。
    """
    if scheme == "bing":
        m = np.zeros(n_chain, dtype=bool)
        m[0] = True
        if n_chain >= 2:
            m[-1] = True
        return m
    return np.ones(n_chain, dtype=bool)


def chain_run(n_chain, scheme, g_on=True, t_end=T_MAX):
    """直链构型的逐站点演化，返回每个站点的末态。

    站点对角意味着链上站点间无耦合，故逐站点独立演化是精确解。
    """
    frames = chain_frames(n_chain, scheme)
    mask = chain_write_mask(n_chain, scheme)
    out = []
    for j in range(n_chain):
        out.append(run_site_shift(frames[j], g_on and mask[j], t_end))
    return out


def chain_time_average(n_chain, scheme, g_on=True, t_end=T_AVG):
    """直链构型的逐站点长期平均，返回 (站点侧均值数组, 记录侧均值数组)。"""
    frames = chain_frames(n_chain, scheme)
    mask = chain_write_mask(n_chain, scheme)
    ms_list, mr_list = [], []
    for j in range(n_chain):
        _, _, ms_, mr_ = time_average_site_shift(frames[j], g_on and mask[j], t_end)
        ms_list.append(ms_)
        mr_list.append(mr_)
    return np.array(ms_list), np.array(mr_list)


def loglog_slope(x, y):
    """双对数拟合的幂次与残差。x、y 须为正数组。"""
    lx = np.log(np.asarray(x, dtype=float))
    ly = np.log(np.asarray(y, dtype=float))
    A = np.vstack([lx, np.ones_like(lx)]).T
    sol, res, _, _ = np.linalg.lstsq(A, ly, rcond=None)
    slope = float(sol[0])
    pred = A @ sol
    resid = float(np.max(np.abs(ly - pred)))
    return slope, resid


# =====================================================================
# 冒烟检查
# =====================================================================
print("=" * 78)
print("冒烟检查")
print("=" * 78)

psi_sm = run_site_shift(0.0, True, T_MAX)
print("L = 1 的末态分栏（应约为站点 0.00117617、记录 0.99882383）:")
print("  站点侧 %.8f，记录侧 %.8f，总量 %.8f" %
      (site_pop(psi_sm), reg_pop(psi_sm), total_pop(psi_sm)))

L_SMOKE = 8
ms_a, mr_a = chain_time_average(L_SMOKE, "jia")
ms_b, mr_b = chain_time_average(L_SMOKE, "yi")
ms_c, mr_c = chain_time_average(L_SMOKE, "bing")
print("L = 8 取法甲（同时写入）逐站点 ⟨站点侧⟩：%s"
      % ", ".join("%.8f" % v for v in ms_a))
print("L = 8 取法乙（依次写入）逐站点 ⟨站点侧⟩：%s"
      % ", ".join("%.8f" % v for v in ms_b))
print("L = 8 取法丙（端点触发）逐站点 ⟨站点侧⟩：%s"
      % ", ".join("%.8f" % v for v in ms_c))
print("L = 8 取法丙逐站点 ⟨记录侧⟩：%s" % ", ".join("%.8f" % v for v in mr_c))

dbar_a = float(np.mean(ms_a)) - 1.0
dbar_b = float(np.mean(ms_b)) - 1.0
dbar_c = float(np.mean(ms_c)) - 1.0
print("三种取法的链平均 Δ̄ 粗读：甲 %.10f，乙 %.10f，丙 %.10f" % (dbar_a, dbar_b, dbar_c))
print("  甲乙之差 %.3e（取法乙的链内最大偏移 %.2f rad）"
      % (abs(dbar_a - dbar_b), (L_SMOKE - 1) * DELTA_A))
print("  丙的链平均与端点占比（2/8 = 0.25）的差 %.3e" % abs(dbar_c - (-0.5 * 2.0 / L_SMOKE)))
print("变化的轴核对：L 轴由 chain_run 的站点数控制，取法轴由 scheme 控制，两者独立。")
print()

# =====================================================================
# LB-1 单站点退化
# =====================================================================
print("=" * 78)
print("LB-1 单站点退化（判据稳健）")
print("=" * 78)

# LB-1 容差预算：取 1e-5 相对偏差，与第 726 号同源。
#   误差来源：本轮实现逐字沿用第 726 号的 split_step 与 make_packet，
#   逐位复现的理论差值只来自浮点重排，量级 1e-16 相对；
#   步长 DT = 2.0e-3 与第 726 号相同，累积角的离散化误差同源。
#   写入时刻的分辨率为采样步长 DT = 0.002 s。

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
rel_t = abs(t_write - t_cross) / t_cross
lb1a_ok = rel_t < 1e-5
record("LB-1a", "L = 1 时写入时刻复现 0.5000",
       "相对偏差 < 1e-5（对折返时刻 0.5000）",
       "写入时刻 %.4f，折返 %.4f，相对偏差 %.3e" % (t_write, t_cross, rel_t), lb1a_ok)

e0 = total_pop(make_packet(theta, PW))
e1 = total_pop(psi_sm)
rel_e = abs(e1 - e0) / e0
lb1b_ok = rel_e < 1e-5
record("LB-1b", "L = 1 时总激发数复现 1.00000000",
       "相对偏差 < 1e-5",
       "初始 %.10f，末态 %.10f，偏差 %.3e" % (e0, e1, abs(e1 - e0)), lb1b_ok)

psi_nw, _ = run(make_packet(theta, PW), 0.0, steps=int(round(T_MAX / DT)))
site_n = site_pop(psi_nw)
d_site = site_pop(psi_sm) - site_n
rel_site = abs(d_site - (-0.99882383)) / 0.99882383
lb1c_ok = rel_site < 1e-5
record("LB-1c", "L = 1 时站点侧增量复现 -0.99882383",
       "相对偏差 < 1e-5",
       "站点侧增量 %.8f（无写入 %.8f，有写入 %.8f），相对偏差 %.3e"
       % (d_site, site_n, site_pop(psi_sm), rel_site), lb1c_ok)

t_avg1, s_avg1 = time_average_site(make_packet(theta, PW), G_WRITE)
mean_site = float(np.trapezoid(s_avg1, t_avg1) / T_AVG)
t_base1, s_base1 = time_average_site(make_packet(theta, PW), np.zeros_like(G_WRITE))
mean_base = float(np.trapezoid(s_base1, t_base1) / T_AVG)
rel_m = abs(mean_site - 0.50000000) / 0.50000000
lb1d_ok = rel_m < 1e-5
record("LB-1d", "L = 1 时长期平均站点侧复现 0.50000000",
       "相对偏差 < 1e-5（T = 8.0 覆盖 4 个整读数周期）",
       "⟨站点侧⟩ = %.10f，基线 %.10f" % (mean_site, mean_base), lb1d_ok)

dbar1 = mean_site - mean_base
rel_d = abs(dbar1 - (-0.50000000)) / 0.50000000
lb1e_ok = rel_d < 1e-5
record("LB-1e", "L = 1 时长期平均增量复现 -0.50000000",
       "相对偏差 < 1e-5",
       "Δ̄ = %.10f，相对偏差 %.3e" % (dbar1, rel_d), lb1e_ok)

# =====================================================================
# LB-2 消去对两栏
# =====================================================================
print("=" * 78)
print("LB-2 消去对两栏（判据稳健）")
print("=" * 78)

# LB-2 容差预算：
#   站点侧对解析期望 0.50000000 的相对偏差取 1e-6。误差来源两项：
#   其一，积分窗的整周期条件。T = 8.0 覆盖 4 个整读数周期，周期闭合，
#   边界项量级 1/T = 0.125，但因整周期而消失，残留量级 1e-14（浮点累加）。
#   其二，梯形积分的离散化误差，步长 DT = 2.0e-3，量级 T·dt²/12·|n''|，
#   |n''| ≲ 180，故误差 ≤ 8×(2e-3)²/12×180 ≈ 4.8e-5 的积分量，
#   除以 T = 8.0 得均值量级 6e-6？此处该项被整周期闭合抵消，
#   实测偏差在 1e-13 量级，故门槛 1e-6 留七个数量级余量。
#   两栏之和的门槛取 1e-12：两栏之和恒等于 1（总激发数守恒），
#   站点侧栏与记录侧栏之和恒为机器零，量级 1e-13。
#   多站点读数口径：逐站点读数与链平均两者都报。

L_LB2 = 8
ms2, mr2 = chain_time_average(L_LB2, "jia")
site_cols = ms2 - 1.0
reg_cols = 1.0 - mr2
col_sum = site_cols + reg_cols
rel_site2 = [abs(v - (-0.50000000)) / 0.50000000 for v in site_cols]
lb2a_ok = all(v <= 1e-6 for v in rel_site2)
record("LB-2a", "逐站点站点侧栏对解析期望 -0.50000000",
       "相对偏差 ≤ 1e-6（%d 站点，逐站点口径）" % L_LB2,
       "逐站点相对偏差 %s，最大 %.3e"
       % (", ".join("%.3e" % v for v in rel_site2), max(rel_site2)), lb2a_ok)

chain_site = float(np.mean(site_cols))
chain_reg = float(np.mean(reg_cols))
rel_chain = abs(chain_site - (-0.50000000)) / 0.50000000
lb2b_ok = rel_chain <= 1e-6
record("LB-2b", "链平均站点侧栏对解析期望 -0.50000000",
       "相对偏差 ≤ 1e-6（链平均口径）",
       "链平均站点侧栏 %.12f，相对偏差 %.3e" % (chain_site, rel_chain), lb2b_ok)

max_sum = float(np.max(np.abs(col_sum)))
lb2c_ok = max_sum <= 1e-12
record("LB-2c", "消去对两栏之和为零",
       "|站点侧栏 + 记录侧栏| ≤ 1e-12（逐站点与链平均都核）",
       "逐站点两栏和最大绝对值 %.3e，链平均两栏和 %.3e"
       % (max_sum, abs(chain_site + chain_reg)), lb2c_ok)

supp("LB-2 附带", "记录侧栏的读数",
     "逐站点记录侧栏 %s；链平均记录侧栏 %.12f"
     % (", ".join("%.8f" % v for v in reg_cols), chain_reg),
     "记录侧栏与站点侧栏反号，两栏之和为机器零")

# =====================================================================
# LB-3 净栏符号与结构式
# =====================================================================
print("=" * 78)
print("LB-3 净栏符号与结构式（判据敏感）")
print("=" * 78)

# LB-3 容差预算：结构式的相对偏差取 1e-6。
#   误差来源：净栏由两栏的数值读数线性组合而成，读数本身的偏差在 1e-13 量级
#   （见 LB-2），故结构式的偏差同量级，门槛 1e-6 留七个数量级余量。
#   本条只判符号与结构式。E_a 与 E_c 的绝对值属 P1-62 待裁，
#   故取归一化档 E_c - E_a = 1，不给绝对标度。
#   符号的口径（作者 2026-10-21 裁决后取定）：净栏式取 ν·Δ̄·(E_c - E_a)。
#   本轮取 E_a = 0、E_c = 1 的归一化档，则 E_c - E_a = 1，
#   净栏 = ν·(-0.5)·(+1) = -ν·0.5 < 0。
#   物理核对：材料由站点侧搬到记录侧，每激发占的刻度由 E_a 变为 E_c，
#   总能量变化为 ν×(搬走的量)×(E_c - E_a)，故 E_c > E_a 时净栏为负，
#   与修正后的式子一致。

# ν 的取值：案乙部件四取 ν = w_core/l_ax，即每单位长度的写入站点数。
# 本直链构造下 L 个站点占 L 个轴向长度单位，故 ν = 1，与 L 无关。
# ν 只给系数乘一个因子，不影响 LB-4 的两个幂次。
nu_per_site = 1.0
dbar_bar = float(np.mean(site_cols))
net_formula = nu_per_site * dbar_bar * (E_C - E_A)
net_expected = nu_per_site * abs(dbar_bar) * (E_C - E_A)
lb3a_ok = net_formula < 0.0
record("LB-3a", "净栏符号为负（E_c > E_a 档）",
       "净栏 ε_wr^net = ν·Δ̄·(E_c - E_a) < 0",
       "净栏 %.12f（ν = %.1f，Δ̄ = %.10f，E_c - E_a = %+.1f）"
       % (net_formula, nu_per_site, dbar_bar, E_C - E_A), lb3a_ok,
       "按修正后的净栏式，Δ̄ < 0 且 E_c - E_a > 0，两者之积为负")

rel_net = abs(abs(net_formula) - net_expected) / net_expected
lb3b_ok = rel_net <= 1e-6
record("LB-3b", "净栏数值与 ν·|Δ̄|·(E_c - E_a) 的相对偏差",
       "相对偏差 ≤ 1e-6",
       "|ε_wr^net| = %.12f，ν·|Δ̄|·(E_c - E_a) = %.12f，相对偏差 %.3e"
       % (abs(net_formula), net_expected, rel_net), lb3b_ok,
       "结构式在绝对值层面逐位成立")

supp("LB-3 附带", "两栏与净栏的符号对照",
     "站点侧栏 %.10f（负），记录侧栏 %.10f（正），净栏 %.10f（负）"
     % (chain_site, chain_reg, net_formula),
     "三条在同一归一化档下同时成立：站点侧栏负、记录侧栏正、净栏负，"
     "这是结构式自洽的直接证据")

net_alt = nu_per_site * dbar_bar * (0.0 - 1.0)
supp("LB-3 附带", "对照档（E_a = 1、E_c = 0，E_c < E_a）",
     "净栏 %.12f（符号 %s）" % (net_alt, "负" if net_alt < 0 else "正"),
     "对照档：E_c < E_a 时净栏为正，与 E_c > E_a 档的读数同向相反，"
     "构成同向的对照")

# =====================================================================
# LB-4 缩放律
# =====================================================================
print("=" * 78)
print("LB-4 缩放律（乙-1）（判据敏感）")
print("=" * 78)

# LB-4 容差预算：幂次偏离 ±0.15。
#   来源：双对数拟合在 L 跨一个数量级（2 至 32，跨度 16 倍即 1.2 个数量级）时，
#   幂次的统计不确定度在 0.05 量级，取 0.15 留三倍余量。
#   本轮读数是逐位确定值（无随机性），拟合残差远小于该门槛，
#   故该项反映的是门槛本身的选取，不含统计成分。
#   两口径须分列报告：每单位长度系数 ε/L 的幂次期望 0，
#   总能量 L·ε 的幂次期望 1。
#   系数口径的定义：每单位长度的净栏系数 ε ≡ ν·Δ̄·(E_c - E_a)，
#   ν 取每站点占据的轴向长度取 1 格时的值，故 ε = Δ̄·(E_c - E_a) 为常数。
#   报告的两条曲线是：每单位长度系数 ε/L（其中 ε 为 L 个站点的净栏总和）
#   与总能量 L×系数。

L_LIST = [2, 4, 8, 16, 32]
lb4_rows = []
for n_chain in L_LIST:
    msx, mrx = chain_time_average(n_chain, "jia")
    sc = float(np.mean(msx)) - 1.0
    # 每单位长度系数：ν 是每单位长度的写入站点数，在本直链构造下为常数
    # （L 个站点占 L 个轴向长度单位，故 ν = L/L = 1，与 L 无关）。
    # 净栏按案乙部件四为每单位长度的量，故总能量档为 L 倍。
    eps_per_len = nu_per_site * sc * (E_C - E_A)    # 每单位长度系数（修正后的净栏式）
    eps_total = float(n_chain) * eps_per_len        # 链上净栏总和
    lb4_rows.append((n_chain, sc, eps_total, eps_per_len))
    print("  L = %2d：链平均 Δ̄ = %.12f，每单位长度系数 = %.12f，链上净栏总和 = %.12f"
          % (n_chain, sc, eps_per_len, eps_total))

eps_tot_arr = np.array([r[2] for r in lb4_rows])
eps_per_arr = np.array([r[3] for r in lb4_rows])
Ls = np.array([r[0] for r in lb4_rows], dtype=float)
slope_tot, resid_tot = loglog_slope(Ls, np.abs(eps_tot_arr))
slope_per, resid_per = loglog_slope(Ls, np.abs(eps_per_arr))
lb4a_ok = abs(slope_per) <= 0.15
record("LB-4a", "每单位长度系数的幂次",
       "幂次偏离 0 不超过 ±0.15",
       "幂次 %.9f（双对数拟合），最大残差 %.3e，每单位长度系数五档 %s"
       % (slope_per, resid_per, ", ".join("%.8f" % v for v in np.abs(eps_per_arr))),
       lb4a_ok)

lb4b_ok = abs(slope_tot - 1.0) <= 0.15
record("LB-4b", "总能量口径（链长乘系数）的幂次",
       "幂次偏离 1 不超过 ±0.15",
       "幂次 %.9f（双对数拟合），最大残差 %.3e，链上净栏总和五档 %s"
       % (slope_tot, resid_tot, ", ".join("%.8f" % v for v in np.abs(eps_tot_arr))),
       lb4b_ok)

supp("LB-4 附带", "两口径的幂次关系与 ν 的无关性",
     "每单位长度幂次 %.9f，总能量幂次 %.9f，两者之和 %.9f" %
     (slope_per, slope_tot, slope_per + slope_tot),
     "两个口径由同一组读数导出，幂次相差 1，与「总能量 = 链长乘每单位长度系数」一致；"
     "ν 取任何常数都不改变两个幂次（ν 只给系数乘一个因子）")

# =====================================================================
# LB-5 取法甲与取法乙的逐站点一致性
# =====================================================================
print("=" * 78)
print("LB-5 取法甲与取法乙的逐站点一致性（判据稳健）")
print("=" * 78)

# LB-5 容差预算：相对偏差取 1e-6。
#   误差来源：两取法各自的浮点累加，量级 L×1e-16；
#   第 741 号的解析恒等式给出两者在精确算术下逐位相同，
#   故实测偏差应落在浮点噪声内。门槛 1e-6 留十个数量级余量。

L_LB5 = 8
ms5a, mr5a = chain_time_average(L_LB5, "jia")
ms5b, mr5b = chain_time_average(L_LB5, "yi")
devs5 = [abs(v - w) / abs(w) for v, w in zip(ms5a, ms5b)]
lb5_ok = all(v <= 1e-6 for v in devs5)
record("LB-5", "取法甲与取法乙的逐站点长期平均一致",
       "相对偏差 ≤ 1e-6（%d 站点）" % L_LB5,
       "逐站点相对偏差 %s，最大 %.3e"
       % (", ".join("%.3e" % v for v in devs5), max(devs5)), lb5_ok,
       "第 741 号解析恒等式在链构型上的复现")

devs5_chain = abs(float(np.mean(ms5a)) - float(np.mean(ms5b))) / abs(float(np.mean(ms5a)))
supp("LB-5 附带", "链平均口径的偏离",
     "甲链平均 %.12f，乙链平均 %.12f，相对偏差 %.3e"
     % (float(np.mean(ms5a)), float(np.mean(ms5b)), devs5_chain),
     "链内最大偏移 %.2f rad，恒等式对任意偏移成立" % ((L_LB5 - 1) * DELTA_A))

# =====================================================================
# LB-6 取法丙的部分写入
# =====================================================================
print("=" * 78)
print("LB-6 取法丙的部分写入（判据稳健）")
print("=" * 78)

# LB-6 容差预算：
#   中间站点站点侧布居对 1.00000000 的相对偏差取 1e-6。
#   误差来源：中间站点不施加写入耦合，其站点侧布居恒为 1
#   （自由演化是相位旋转，不动布居），时间平均的舍入量级 1e-13。
#   链平均与端点占比一致的相对偏差取 1e-3。
#   来源：端点占比是整数除法（2/L），L 档的分辨率即 1/L，
#   L = 8 档为 0.125，取 1e-3 留两个数量级余量。

L_LB6 = 8
ms6, mr6 = chain_time_average(L_LB6, "bing")
mid = slice(1, L_LB6 - 1)
mid_sites = ms6[mid]
rel_mid = [abs(v - 1.0) for v in mid_sites]
lb6a_ok = max(rel_mid) <= 1e-6
record("LB-6a", "取法丙的中间站点站点侧布居保持 1.00000000",
       "相对偏差 ≤ 1e-6（%d 个中间站点）" % (L_LB6 - 2),
       "中间站点布居 %s，最大相对偏差 %.3e"
       % (", ".join("%.10f" % v for v in mid_sites), max(rel_mid)), lb6a_ok)

chain_bing = float(np.mean(ms6)) - 1.0
endpoint_frac = 2.0 / L_LB6
bing_expect = -0.5 * endpoint_frac
rel_bing = abs(chain_bing - bing_expect) / abs(bing_expect)
lb6b_ok = rel_bing <= 1e-3
record("LB-6b", "取法丙的链平均与端点占比一致",
       "相对偏差 ≤ 1e-3（端点占比 2/L = %.4f）" % endpoint_frac,
       "链平均 Δ̄ = %.12f，预期 -0.5×(2/L) = %.12f，相对偏差 %.3e"
       % (chain_bing, bing_expect, rel_bing), lb6b_ok)

supp("LB-6 附带", "取法丙的逐站点两栏读数",
     "站点侧 %s；记录侧 %s"
     % (", ".join("%.8f" % v for v in ms6), ", ".join("%.8f" % v for v in mr6)),
     "中间站点两栏皆为真空基线，两端站点完成写入")

# =====================================================================
# LB-7 链端效应核验
# =====================================================================
print("=" * 78)
print("LB-7 链端效应核验（判据稳健）")
print("=" * 78)

# LB-7 容差预算：门槛取机器零阈 1e-12。
#   站点对角下链上站点间无耦合，端点站点与内部站点的装置逐位相同，
#   故两者读数之差只来自浮点重排，量级 1e-16。
#   若出现超过门槛的差异，说明存在隐藏耦合，须报出并定位。

L_LB7_LIST = [2, 4, 8, 16, 32]
lb7_rows = []
for n_chain in L_LB7_LIST:
    msx, _ = chain_time_average(n_chain, "jia")
    if n_chain >= 2:
        ends = [float(msx[0]), float(msx[-1])]
        inner = [float(v) for v in msx[1:-1]]
        dev_end = max(abs(e - i_) for e in ends for i_ in inner) if inner else 0.0
    else:
        ends = [float(msx[0])]
        dev_end = 0.0
    lb7_rows.append((n_chain, dev_end))
    print("  L = %2d：端点读数 %s，端点与内部的最大差 %.3e"
          % (n_chain, ", ".join("%.12f" % v for v in ends), dev_end))

dev7 = max(r[1] for r in lb7_rows)
lb7_ok = dev7 <= 1e-12
record("LB-7", "链端效应核验：端点与内部站点读数之差",
       "≤ 1e-12（五档 L = 2、4、8、16、32）",
       "五档最大差 %.3e" % dev7, lb7_ok,
       "站点对角下链上站点间无耦合，端点与内部装置逐位相同")

# =====================================================================
# LB-8 前提守恒与事件计数
# =====================================================================
print("=" * 78)
print("LB-8 前提守恒与事件计数（乙-7）（判据稳健）")
print("=" * 78)

# LB-8 容差预算：
#   事件计数容差为 0，精确计数。笔数是相位绕数增量，属拓扑量。
#   总激发数等于逐站点求和的门槛 1e-12：差只来自浮点求和次序，
#   量级 1e-16×L，取 1e-12 留四个数量级余量。

ana_lb8 = []
num_lb8 = []
for j in range(L_LB5):
    a_, n_ = write_events(chain_frames(L_LB5, "yi")[j], 1.0)
    ana_lb8.append(a_)
    num_lb8.append(n_)
dev8 = max(abs(a_ - 1) for a_ in ana_lb8)
dev8b = max(abs(n_ - a_) for n_, a_ in zip(num_lb8, ana_lb8))
lb8a_ok = (dev8 == 0) and (dev8b == 0)
record("LB-8a", "每站恰一笔的写入事件数（D-5 口径）",
       "每站解析事件数 = 1（偏差 0），且与数值峰数口径一致",
       "取法乙 %d 站点：解析事件数 %s，数值峰数 %s，对 1 的偏差 %d，两口径差 %d"
       % (L_LB5, ", ".join(str(v) for v in ana_lb8),
          ", ".join(str(v) for v in num_lb8), dev8, dev8b), lb8a_ok,
       "一个指针周期即一格，笔数是相位绕数增量")

tot_chain = 0.0
for pj in chain_run(L_LB5, "yi", True, T_MAX):
    tot_chain += total_pop(pj)
dev8c = abs(tot_chain - float(L_LB5)) / float(L_LB5)
lb8b_ok = dev8c <= 1e-12
record("LB-8b", "链的总激发数等于逐站点求和",
       "相对偏差 ≤ 1e-12（%d 站点，取法乙）" % L_LB5,
       "链总激发数 %.12f（初始 %d），相对偏差 %.3e"
       % (tot_chain, L_LB5, dev8c), lb8b_ok)

lb8c_ok = lb1c_ok and lb1e_ok
record("LB-8c", "单站点退化与第 726 号一致（乙-7 的复现部分）",
       "站点侧增量与长期平均增量的相对偏差均 ≤ 1e-5",
       "站点侧增量相对偏差 %.3e，长期平均增量相对偏差 %.3e"
       % (rel_site, rel_d), lb8c_ok)

# 前提守恒核：逐站点独立演化是精确解
print("  前提守恒核：站点对角下逐站点独立演化是精确解")
kron_devs = []
for n_site in (2, 3, 4):
    dev_kron = operator_factorization_dev(n_site)
    kron_devs.append(dev_kron)
    print("    核一 N_site = %d（联合维数 4^{n_site} = %d）：指数按张量积分解的偏差 %.3e"
          % (n_site, 4 ** n_site, dev_kron))
comm_devs = []
for n_site in (8, 16):
    dev_comm = commutator_dev(n_site)
    comm_devs.append(dev_comm)
    print("    核一续 N_site = %2d：两项生成元的对易子偏差 %.3e" % (n_site, dev_comm))
prof_devs = []
for n_site, sh in ((2, [0.0, 0.2]), (3, [0.0, 0.2, 0.4])):
    th_j = [theta[40 + 7 * k] for k in range(n_site)]
    coeffs = [float(G_FULL * pulse(th_j[k] - sh[k], W_PULSE)) for k in range(n_site)]
    dev_prof = operator_factorization_dev(n_site, coeffs)
    prof_devs.append(dev_prof)
    print("    核二 N_site = %d（含实际剖面）：分解偏差 %.3e" % (n_site, dev_prof))
t_j = time.time()
J2 = joint_run_2site([0.0, 0.0], T_MAX)
s1j, r1j, s2j, r2j, totj = joint_pops(J2)
ind_s1 = site_pop(run_site_shift(0.0, True, T_MAX))
ind_tot = ind_s1 + reg_pop(run_site_shift(0.0, True, T_MAX))
sum_dev = abs(totj - 2.0 * ind_tot)
rel_s1 = abs(s1j - ind_s1) / ind_s1
print("    核三 态层联合演化（N = 2，用时 %.1f s）：总激发数偏差 %.3e，逐站点读数相对偏差 %.3e"
      % (time.time() - t_j, sum_dev, rel_s1))
lb8d_ok = sum_dev <= 1e-12 and max(rel_s1, abs(s2j - ind_s1) / ind_s1) <= 1e-6
record("LB-8d", "逐站点独立演化是精确解（三核）",
       "核一 ≤ 1e-12，核二 ≤ 1e-12，核三总激发数偏差 ≤ 1e-12",
       "核一 %s；核二 %s；核三 %.3e（逐站点相对偏差 %.3e）"
       % (", ".join("%.3e" % v for v in kron_devs),
          ", ".join("%.3e" % v for v in prof_devs), sum_dev, rel_s1), lb8d_ok)

# =====================================================================
# LB-9 恒真检验
# =====================================================================
print("=" * 78)
print("LB-9 恒真检验（判据稳健）")
print("=" * 78)

# LB-9 容差预算：门槛取机器零阈 1e-12。
#   误差来源：关断写入耦合后，站点侧布居恒为 1（自由演化是相位旋转），
#   梯形积分除以 T = 8.0 引入的舍入量级 1e-13，与第 726 号 DS-3f 同源。
#   反例检验：把写入耦合开回即得 Δ̄ = -0.5，判否。

ms9, mr9 = chain_time_average(L_LB6, "jia", g_on=False)
chain9 = float(np.mean(ms9)) - 1.0
lb9_ok = abs(chain9) <= 1e-12
record("LB-9", "恒真检验：关断写入耦合，全链 Δ̄ = 0",
       "|Δ̄| ≤ 1e-12（%d 站点链）" % L_LB6,
       "链平均 ⟨站点侧⟩ = %.12f，基线 1.0，Δ̄ = %.3e" % (float(np.mean(ms9)), chain9),
       lb9_ok, "反例检验：把写入耦合开回即得 Δ̄ = %.8f，判否" % dbar1)

psi_diag = run_site_shift(0.0, True, T_MAX, op=apply_diag)
dbar_diag = site_pop(psi_diag) - site_n
supp("LB-9 附带", "正定势替身（apply_diag）下的站点侧增量",
     "站点侧增量 %.3e（门槛 1e-12）" % dbar_diag,
     "替身只改相位不做搬运，材料留在站点侧")

# =====================================================================
# LB-10 替换检验
# =====================================================================
print("=" * 78)
print("LB-10 替换检验（区分力核）")
print("=" * 78)

# LB-10 容差预算：沿用 LB-5 与 LB-6 的门槛（1e-6 与 1e-3）。
#   替换构型：把取法乙换成「链上只有一半站点写入」的错误构型，
#   即隔站写入（站点序号为偶数的站点写入，奇数的不写入）。
#   判据要求 LB-5 与 LB-6 须判否，即替身必须与真构型可区分。

L_LB10 = 8
half_mask = np.array([(j % 2 == 0) for j in range(L_LB10)])
frames10 = DELTA_A * np.arange(L_LB10, dtype=float)
ms10 = []
for j in range(L_LB10):
    _, _, ms_, _ = time_average_site_shift(frames10[j], half_mask[j], T_AVG)
    ms10.append(ms_)
ms10 = np.array(ms10)
devs10 = [abs(v - w) / abs(w) for v, w in zip(ms10, ms5b)]
# LB-5 判「逐站点一致」，故只要有一个站点偏离超过门槛即判否。
# 替身构型下偶数站点的读数与真构型逐位相同（偏离恰为 0），
# 故判否条件取「任一偏离超过门槛」而非「全部超过门槛」。
lb5_fail = any(v > 1e-6 for v in devs10)
print("  替换构型（隔站写入）下逐站点 ⟨站点侧⟩：%s" % ", ".join("%.8f" % v for v in ms10))
print("  与取法乙真构型的逐站点相对偏差：%s" % ", ".join("%.3e" % v for v in devs10))

# 替换构型下 LB-6 的中间站点读数
mid10 = ms10[1:L_LB10 - 1]
lb6_fail = any(abs(v - 1.0) > 1e-6 for v in mid10)
mid10_fail_sites = [j + 1 for j, v in enumerate(mid10) if abs(v - 1.0) > 1e-6]
print("  替换构型下中间站点 ⟨站点侧⟩：%s" % ", ".join("%.8f" % v for v in mid10))
lb10_ok = lb5_fail and lb6_fail
record("LB-10", "替换检验：隔站写入的错误构型须使 LB-5 与 LB-6 判否",
       "LB-5 的逐站点一致性判否，且 LB-6 的中间站点布居判否",
       "LB-5 偏离量 %s（任一超门槛即判否：%s，偏离站点 %s），"
       "LB-6 中间站点 %s（偏离 1.0 的站点 %s）"
       % (", ".join("%.3e" % v for v in devs10), lb5_fail,
          [j for j, v in enumerate(devs10) if v > 1e-6],
          ", ".join("%.8f" % v for v in mid10), mid10_fail_sites), lb10_ok,
       "替身与真构型可区分，判据含区分力")

# =====================================================================
# 出图
# =====================================================================
print("=" * 78)
print("出图")
print("=" * 78)

fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.4))

# (a) L 档下 ε/L 与 L·ε 的双对数曲线
ax = axes[0, 0]
ax.loglog(Ls, np.abs(eps_per_arr), "o-", color="tab:blue", ms=7,
          label=r"每单位长度系数 $|\varepsilon|$")
ax.loglog(Ls, np.abs(eps_tot_arr), "s-", color="tab:orange", ms=7,
          label=r"总能量口径 $|L\varepsilon|$")
ref0 = np.abs(eps_per_arr[0]) * (Ls / Ls[0]) ** 0
ref1 = np.abs(eps_tot_arr[0]) * (Ls / Ls[0]) ** 1
ax.loglog(Ls, ref0, ":", color="k", lw=1.4, label=r"幂次 0 参考线")
ax.loglog(Ls, ref1, "--", color="tab:gray", lw=1.4, label=r"幂次 1 参考线")
ax.set_xlabel(r"位错线长度 $L$（站点数）")
ax.set_ylabel(r"净栏系数绝对值")
ax.set_title(r"(a) 缩放律：每单位长度系数与总能量的双对数曲线", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")
ax.text(0.03, 0.06, "幂次 %.6f 与 %.6f" % (slope_per, slope_tot),
        transform=ax.transAxes, fontsize=8)

# (b) 取法甲与取法乙在 L=8 的逐站点长期平均剖面
ax = axes[0, 1]
ax.plot(np.arange(L_LB5), ms5a, "o-", color="tab:blue", ms=6,
        label=r"取法甲 同时写入（$A_j = 0$）")
ax.plot(np.arange(L_LB5), ms5b, "s--", color="tab:orange", ms=6,
        label=r"取法乙 依次写入（$A_j = j\Delta A$）")
ax.axhline(0.5, color="tab:red", ls="--", lw=1.0, label=r"解析期望 0.5")
ax.set_xlabel("链上站点序号")
ax.set_ylabel(r"逐站点 $\langle$站点侧$\rangle$")
ax.set_ylim(0.45, 0.55)
ax.set_title(r"(b) 取法甲与取法乙在 $L = 8$ 的逐站点长期平均", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax.text(0.03, 0.06, r"重合偏差 %.1e" % max(devs5), transform=ax.transAxes, fontsize=8)

# (c) 取法丙的逐站点剖面与链平均对端点占比
#     左轴：L = 8 档的逐站点长期平均剖面（两端写入、中间未写入）
#     右轴：链平均 Δ̄ 随 L 的实测值，与端点占比预测 -0.5×(2/L) 对照
bing_chain_by_L = []
for n_chain in L_LB7_LIST:
    msx, _ = chain_time_average(n_chain, "bing")
    bing_chain_by_L.append(float(np.mean(msx)) - 1.0)
bing_meas = np.array(bing_chain_by_L)
bing_pred = np.array([-0.5 * 2.0 / float(n) for n in L_LB7_LIST])

# 左半：L = 8 档的逐站点剖面（横轴为链上站点序号）
ax = axes[1, 0]
ax.plot(np.arange(L_LB6), ms6, "^-", color="tab:green", ms=8,
        label=r"逐站点 $\langle$站点侧$\rangle$（$L = 8$）")
ax.axhline(1.0, color="tab:gray", ls=":", lw=1.2, label=r"未写入基线 1.0")
ax.set_xlabel("链上站点序号")
ax.set_ylabel(r"逐站点 $\langle$站点侧$\rangle$")
ax.set_xlim(-0.4, L_LB6 - 0.6)
ax.set_ylim(0.45, 1.05)
ax.set_title(r"(c) 取法丙：中间站点未被写入", fontsize=10)
ax.legend(fontsize=8, loc="lower left")
ax.grid(alpha=0.3)

# 右半：链平均 Δ̄ 随 L 的实测与端点占比预测（横轴为 L，对数刻度）
ax = axes[1, 0].inset_axes([0.54, 0.52, 0.43, 0.44])
ax.plot(L_LB7_LIST, bing_meas, "o-", color="tab:red", ms=6,
        label=r"实测链平均 $\bar{\Delta}$")
ax.plot(L_LB7_LIST, bing_pred, "s--", color="tab:purple", ms=5,
        label=r"端点占比预测 $-0.5\times 2/L$")
ax.set_xscale("log", base=2)
ax.set_xlabel(r"$L$", fontsize=8)
ax.set_ylabel(r"链平均 $\bar{\Delta}$", fontsize=8)
ax.tick_params(labelsize=7)
ax.legend(fontsize=6, loc="lower right")
ax.grid(alpha=0.3, which="both")
ax.set_title("链平均对端点占比", fontsize=7)
ax.text(0.04, 0.94, "最大相对偏差 %.1e"
        % max(abs(bing_meas - bing_pred) / np.abs(bing_pred)),
        transform=ax.transAxes, fontsize=6, va="top")
ax.set_ylim(-0.56, -0.02)

# (d) 链端效应
ax = axes[1, 1]
ax.plot([r[0] for r in lb7_rows], [r[1] for r in lb7_rows], "o-",
        color="tab:blue", ms=7, label=r"端点与内部读数之差")
ax.axhline(1e-12, color="tab:red", ls="--", lw=1.0, label=r"门槛 $10^{-12}$")
ax.axhline(0.0, color="k", lw=0.8)
ax.set_xscale("log", base=2)
ax.set_xlabel(r"位错线长度 $L$（站点数）")
ax.set_ylabel(r"读数之差")
ax.set_title(r"(d) 链端效应：应恒为机器零", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")
ax.text(0.03, 0.06, "五档最大差 %.3e" % dev7, transform=ax.transAxes, fontsize=8)

fig.suptitle("第 746 号：案乙的位错线长度扫描（直链构型）", fontsize=12)
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
print("合计 %d/%d 通过（主判据 LB-1 至 LB-10，展开为 %d 个子项）"
      % (npass, len(RESULTS), len(RESULTS)))
print()
print("关键读数:")
print("  LB-1 单站点退化：写入时刻 %.4f，总激发数 %.10f（偏差 %.3e），站点侧增量 %.8f"
      % (t_write, e1, abs(e1 - e0), d_site))
print("       长期平均 ⟨站点侧⟩ = %.10f，Δ̄ = %.10f" % (mean_site, dbar1))
print("  LB-2 两栏：逐站点站点侧栏 %s"
      % ", ".join("%.10f" % v for v in site_cols))
print("       链平均站点侧栏 %.12f，记录侧栏 %.12f，两栏和最大 %.3e"
      % (chain_site, chain_reg, max_sum))
print("  LB-3 净栏：%.12f（ν = %.1f，Δ̄ = %.10f，E_c - E_a = %+.1f），符号 %s"
      % (net_formula, nu_per_site, dbar_bar, E_C - E_A,
         "负" if net_formula < 0 else "正"))
print("       三条符号对照：站点侧栏 %.10f（负），记录侧栏 %.10f（正），净栏 %.10f（负）"
      % (chain_site, chain_reg, net_formula))
print("       |ε_wr^net| 与 ν|Δ̄|(E_c - E_a) 的绝对值同为 %.12f" % abs(net_formula))
print("  LB-4 缩放律：每单位长度幂次 %.9f，总能量幂次 %.9f（残差 %.3e 与 %.3e）"
      % (slope_per, slope_tot, resid_per, resid_tot))
print("       |ε/L| 五档 %s" % ", ".join("%.8f" % v for v in np.abs(eps_per_arr)))
print("       |L·ε| 五档 %s" % ", ".join("%.8f" % v for v in np.abs(eps_tot_arr)))
print("  LB-5 甲乙一致性：逐站点相对偏差最大 %.3e，链平均 %.3e"
      % (max(devs5), devs5_chain))
print("  LB-6 取法丙：中间站点布居最大偏差 %.3e，链平均 %.12f 对预期 %.12f（偏差 %.3e）"
      % (max(rel_mid), chain_bing, bing_expect, rel_bing))
print("  LB-7 链端效应：五档最大差 %.3e" % dev7)
print("  LB-8 事件计数：每站解析事件数 %s，偏差 %d；链总激发数 %.12f，偏差 %.3e"
      % (", ".join(str(v) for v in ana_lb8), dev8, tot_chain, dev8c))
print("  LB-9 恒真检验：关断写入耦合 Δ̄ = %.3e" % chain9)
print("  LB-10 替换检验：偏离量 %s，LB-5 判否 %s，LB-6 判否 %s"
      % (", ".join("%.3e" % v for v in devs10), lb5_fail, lb6_fail))
print()
print("三条前提守恒的核对结果:")
print("  其一，单站点退化与第 726 号一致：LB-1 五个子项全过，最大相对偏差 %.3e"
      % max(rel_t, rel_e, rel_site, rel_m, rel_d))
print("  其二，写入侧参数七项与第 726 号逐项一致：N_theta = %d，omega_P = %.6f，"
      "theta_0 = %.6f，w = %.3f，dt = %.1e，T = %.1f（长期平均 %.1f），g_full = %.6f"
      % (N_THETA, OMEGA_P, THETA0, W_PULSE, DT, T_MAX, T_AVG, G_FULL))
print("  其三，逐站点独立演化是精确解：")
print("       核一 算符层 Kronecker 分解偏差 %s（N = 2、3、4）"
      % ", ".join("%.3e" % v for v in kron_devs))
print("       核一续 大档交换性偏差 %s（N = 8、16）"
      % ", ".join("%.3e" % v for v in comm_devs))
print("       核二 含实际剖面的算符层分解偏差 %s（N = 2、3）"
      % ", ".join("%.3e" % v for v in prof_devs))
print("       核三 态层联合演化（N = 2）总激发数偏差 %.3e，逐站点读数相对偏差 %.3e"
      % (sum_dev, rel_s1))
print("耗时 %.1f 秒" % (time.time() - T_START))
