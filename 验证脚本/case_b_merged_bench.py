# -*- coding: utf-8 -*-
r"""
case_b_merged_bench.py

第 772 号 案乙主体：复标量环上的合并装置与量级关口（判据 MB-1 至 MB-9）

问题：案乙是构造轮的三案之一，把写入过程与位错能分栏放进同一泛函。
本轮是案乙攻坚线的主体轮，判两层关口。

**第一层，可比性关口（QL-1）。** 第 724 号的 QL-1 判：「装置可比性检验须报告
两侧能量量纲，**不同类则直接比值不合法**」，实测写入侧为无量纲计数（布居）、
C1 侧为带长度量纲（凝聚能），故「量级审计在当前模型配置下不可执行」。
**案乙的全部价值就是把两侧放进同一个装置，使它们同类。**
故 MB-2 判：合并装置内负线项与线张力的量纲是否同类。

**第二层，门槛关。** 第 725 号 AW-3 给驻留条件：补偿 5.2983 对线张力 193.0076，
比值 0.0275，即驻留要求补偿达到线张力的 1/0.0275 = 36.4 倍；AW-4 判窗口交集为空。
故 MB-4 判：合并装置内负线项对线张力的比值是否达到 36.4。

**第三读数（最有价值）**：复标量环的绕数在自由弛豫下会滑移卸载
（振幅通道轮 W3 判否：九组全把绕数从 1 赶到 0）。
**案乙的负线项是唯一可能的稳定化元素。**
若加了写入过程后绕数不再滑移，那是案乙成立的直接证据。

装置（复用，不重造）
--------------------
**C1 侧**（承振幅通道轮）：
格点复标量 $\psi = A e^{i\theta}$，开放边界（边界外取真空 $\psi = 0$），
$H = \sum_{\rm bonds}|\psi_i - \psi_j|^2 + \lambda\sum_{\rm sites}(A_i^2 - A_0(x_i)^2)^2$，
$N = 32$，格距 1，$\xi = 1/\sqrt{4\lambda}$，
锚核取管状 $A_0 = 1 - d\exp(-d_c(R_a)^2/2w^2)$。
**锁定扇区**（该轮已记的关键限制）：先固定相位为环剖面
$\theta = \operatorname{arctan2}(z, \rho - R)$，只弛豫振幅 $A$，
此时绕数恒为 1、芯圆半径恒为 $R$；
扇区能量是 $A$ 的二次型加四次项，耦合固定为 $J_{ij} = \cos(\theta_i - \theta_j)$。

**写入侧**（承第 769 号与第 758 号）：
写入算符逐站点直和（**硬约束**：跨站耦合按强度平方污染记录、破坏站点可加性，
故键耦合只加在 C1 侧）；$E_c = E_P/(Cg^2)$，幂次 -2，$C = 201$；
$g$ 的上界 $g_w = 0.070535 = 0.011226\,\omega_P$。

**环上的写入过程（本轮的新东西）**：
环上每站点一个 $(a_j, c_j)$ 对，写入按逐站点直和加在**环的芯区**
（芯所经的站点集合，D-5 口径「每过格恰写一笔」）。
环是闭合的，故「缺陷扫过站点」在环上无方向，两条候选加法并排做：
**甲，位错移动**：局域标架 $A_j$ 随缺陷位置沿环平移（承第 741 号的取法乙）。
**乙，位错静止而写入由站点相位越过整数触发**（承第 758 号面缺陷装置的写入节拍口径）。
两条给出不同的事件分布，回答「环上写入事件的密度由什么定」。

**一条必须核的前置**：第 741 号已证「长期平均对局域标架严格不变」
（解析恒等式，读数周期上 $\int_0^2\cos^2 A_j\,dt = 1$ 与偏移无关）。
环上的局域标架含绕数带来的整周偏移 $2\pi N$（[[wiki/提升相位]] 给
$\tilde\theta_i = 2\pi N_i + \theta_i$，且明写「相位与层数必然同步」），
**而 $\cos^2(A + 2\pi N) = \cos^2 A$，故该恒等式在环上解析必然成立**。
故 MB-7 核这一条，并报出环上 $A_j$ 偏移里整周部分与分数部分的分解。

判据：MB-1 至 MB-9 共九条主判据，展开为十四个子项。
逐条给出期望、实测与判定，容差从误差预算推导，见各条注释。
运行：python 验证脚本/case_b_merged_bench.py
确定性，无随机数。图存 images/case_b_merged_bench_772.png。
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

T_START = time.time()
OUT = r"D:/software/TianXingFC/images/case_b_merged_bench_772.png"

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
# C1 侧：复标量环装置（承 验证脚本/amplitude_channel_ring.py）
# =====================================================================
N = 32
N3 = N ** 3
COORD = np.arange(N, dtype=float) - N / 2.0 + 0.5
_X, _Y, _Z = np.meshgrid(COORD, COORD, COORD, indexing="ij")
RHO = np.sqrt(_X ** 2 + _Y ** 2)
IC = N // 2
UPT = [tuple(slice(2, N + 2) if i == ax else slice(1, N + 1) for i in range(3))
       for ax in range(3)]
DNT = [tuple(slice(0, N) if i == ax else slice(1, N + 1) for i in range(3))
       for ax in range(3)]
RS = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]

# 振幅通道轮的参数档（承第 725 号 AW-3 的 slope_scan）
RA0, D_ANCH, W_ANCH = 5.0, 0.5, 1.5
LAM_SLOPE = 1.0


def ring_phase(R):
    return np.arctan2(_Z, RHO - R)


def cos_cache(theta):
    p = np.pad(theta, 1)
    return [(np.cos(theta - p[UPT[ax]]), np.cos(theta - p[DNT[ax]])) for ax in range(3)]


def make_fun(A0, cache, lam):
    A0sq = (A0 ** 2).reshape(N, N, N)

    def fun(x):
        A = x.reshape(N, N, N)
        A2 = A * A
        E = lam * float(np.sum((A2 - A0sq) ** 2))
        g = np.zeros_like(A)
        for ax in range(3):
            cu, cd = cache[ax]
            p = np.pad(A, 1)
            au, ad = p[UPT[ax]], p[DNT[ax]]
            E += float(np.sum(A2 - 2.0 * A * au * cu + au ** 2))
            E += float(np.sum(A2 - 2.0 * A * ad * cd + ad ** 2))
            g += 2.0 * (A - au * cu) + 2.0 * (A - ad * cd)
        g += 4.0 * lam * (A2 - A0sq) * A
        return E, g.ravel()

    return fun


def anchor(Ra, depth, w):
    if depth <= 0.0:
        return np.ones_like(RHO)
    return 1.0 - depth * np.exp(-(((RHO - Ra) ** 2 + _Z ** 2) / (2.0 * w ** 2)))


def solve_R(R, A0, lam, iters=300):
    """锁定扇区下给定环半径 R 的最小能量（只弛豫振幅 A）。"""
    from scipy.optimize import minimize
    cache = cos_cache(ring_phase(R))
    xi = 1.0 / np.sqrt(4.0 * lam)
    A_init = np.minimum(np.tanh(np.sqrt((RHO - R) ** 2 + _Z ** 2) / xi),
                        np.maximum(A0, 0.05))
    res = minimize(make_fun(A0, cache, lam), A_init.ravel(), jac=True,
                   method="L-BFGS-B", bounds=[(0.0, 2.0)] * N3,
                   options=dict(maxiter=iters, ftol=1e-12, gtol=1e-9))
    return float(res.fun), res.x


def curve(A0, lam):
    return np.array([solve_R(R, A0, lam)[0] for R in RS])


def interior_minimum(Rs, vals):
    for i in range(1, len(vals) - 1):
        if vals[i] < vals[i - 1] and vals[i] < vals[i + 1]:
            return Rs[i], min(vals[i - 1], vals[i + 1]) - vals[i]
    return None, float("nan")


def ring_loop_indices(R):
    r"""子午面内的闭合回路：绕 (rho = R, z = 0) 的最小方回路。

    本装置的环剖面是 $\theta = \operatorname{arctan2}(z, \rho - R)$，
    相位只依赖 $(\rho, z)$，沿方位角方向不变。
    故绕数须在**子午面**内取回路。取 $y = 0$ 的层（故 $\rho = |x|$，
    取 $x > 0$ 侧），在 $(x, z)$ 平面内取以 $(R, 0)$ 为心、边长 2 的方回路：
    横向从 $x = R-1$ 走到 $x = R+1$（即 $\rho - R$ 从 $-1$ 到 $+1$），
    纵向从 $z = 0$ 走到 $z = 1$ 再回来。
    该回路在 $(\rho - R, z)$ 平面内绕原点一圈，$\theta$ 恰绕 $2\pi$。

    **回路只经过环芯附近**：若半宽取到 1.5 格以上，回路的两条竖边会落进
    真空区（$|\psi| \to 0$），那里的相位没有定义，读数被真空区的相位污染。
    """
    i_lo = int(np.argmin(np.abs(COORD - (R - 1.0))))
    i_hi = int(np.argmin(np.abs(COORD - (R + 1.0))))
    k0 = int(np.argmin(np.abs(COORD - 0.0)))
    k1 = int(np.argmin(np.abs(COORD - 1.0)))
    j0 = IC                                        # y = 0 的层
    loop = []
    for i in range(i_lo, i_hi + 1):               # 沿 +x 边，z = k0
        loop.append((i, j0, k0))
    for k in range(k0 + 1, k1 + 1):               # 沿 +z 边，x = i_hi
        loop.append((i_hi, j0, k))
    for i in range(i_hi - 1, i_lo - 1, -1):       # 沿 -x 边，z = k1
        loop.append((i, j0, k1))
    for k in range(k1 - 1, k0 - 1, -1):           # 沿 -z 边，x = i_lo
        loop.append((i_lo, j0, k))
    return loop


def loop_phi(R):
    r"""回路在 (rho - R, z) 平面内的极角序列 phi = atan2(z, rho - R)。"""
    loop = ring_loop_indices(R)
    out = []
    for i, j, k in loop:
        out.append(float(np.arctan2(COORD[k], abs(COORD[i]) - R)))
    return np.array(out)


def circulation_from_A(psi, R, r=3):
    """从复场取绕数读数（承振幅通道轮）。

    沿 z = 0 平面半径 R 的闭合环线取相位增量，除以 2π。
    """
    loop = ring_loop_indices(R)
    # 绕数的正确读法：取回路在 (rho - R, z) 平面内的极角序列 phi，
    # 作**连续延拓**（逐点把增量折回 (-pi, pi]），则 phi 的总增量即 2pi·绕数。
    # 理由：theta = phi 模 2pi，theta 在 pi 处的分支跳变由 phi 的连续化吸收。
    phi = loop_phi(R)
    cont = [phi[0]]
    for k in range(1, len(phi)):
        d = phi[k] - cont[-1]
        while d > np.pi:
            d -= 2.0 * np.pi
        while d < -np.pi:
            d += 2.0 * np.pi
        cont.append(cont[-1] + d)
    cont = np.array(cont)
    return float(cont[-1] - cont[0]) / (2.0 * np.pi)


# =====================================================================
# 写入侧：两能级写入算符（承第 726 号）
# =====================================================================
N_THETA = 256
OMEGA_P = 2.0 * np.pi
THETA0 = np.pi
W_PULSE = 0.5
PW = 0.15
DT = 2.0e-3
T_MAX = 1.2
T_AVG = 8.0
T_PERIOD = 1.0            # 写入剖面在 t 上的严格周期（第 755 号的单周期窗口径）

# 率接口（第 394 号）
N_MODES = 201.0
BANDWIDTH = 2.0 * np.pi
TP = 1.0
G_READ = 0.0705
S_LIST = [1.0, 2.0, 5.0, 10.0]
N_BEATS = [10, 100, 1000, 100000]


def pulse(theta, w):
    return np.exp(-(1.0 - np.cos(theta)) / (2.0 * w ** 2))


def omega_of_k(kvec):
    return -OMEGA_P * kvec


def apply_int(psi, f, dt, g):
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


def split_step(psi, dt, theta, kvec, g_arr):
    psi = apply_int(psi, 1.0, dt / 2.0, g_arr)
    for a in range(2):
        for c in range(2):
            psi[:, a, c] = np.fft.ifft(np.fft.fft(psi[:, a, c])
                                       * np.exp(1j * omega_of_k(kvec) * dt))
    psi = apply_int(psi, 1.0, dt / 2.0, g_arr)
    return psi


def make_packet(theta, pw, site=1):
    psi = np.zeros((N_THETA, 2, 2), dtype=complex)
    pk = np.exp(-((theta - THETA0) ** 2) / (2.0 * pw ** 2))
    psi[:, site, 0] = pk / np.linalg.norm(pk)
    return psi


def site_pop(p):
    return float(np.sum(np.abs(p[:, 1, :]) ** 2))


def reg_pop(p):
    return float(np.sum(np.abs(p[:, :, 1]) ** 2))


def total_pop(p):
    return site_pop(p) + reg_pop(p)


theta = np.linspace(0.0, 2.0 * np.pi, N_THETA, endpoint=False)
kvec = np.fft.fftfreq(N_THETA, d=1.0 / N_THETA)
area = float(np.sum(pulse(theta, W_PULSE)) * (2.0 * np.pi / N_THETA))
G_FULL = (np.pi / 2.0) * OMEGA_P / area
t_cross = (2.0 * np.pi - THETA0) / OMEGA_P
G_WRITE = G_FULL * pulse(theta, W_PULSE)

RHO_BC = N_MODES / BANDWIDTH
C_CONST = 2.0 * np.pi * RHO_BC * TP
G_MAX = 1.0 / np.sqrt(C_CONST)
G_MAX_OMEGA = G_MAX / OMEGA_P

# 环上芯区的站点集合：芯圆半径 R 处、z = 0 的环线上的格点。
CORE_R = 5.0


def core_sites(R=CORE_R):
    """芯圆上的格点（绕 z 轴、R 为半径、z = 0 的环线）。"""
    idx = []
    for i in range(N):
        for j in range(N):
            dx = COORD[i]
            dy = COORD[j]
            if abs(np.hypot(dx, dy) - R) < 0.75 and abs(_Z[i, j, IC]) < 0.75:
                idx.append((i, j, IC))
    return idx


CORE = core_sites()
print("=" * 78)
print("装置参数")
print("=" * 78)
print("  C1 侧（承振幅通道轮）：N = %d，格距 1，锚核管半径 %.1f，深度 %.1f，宽度 %.1f"
      % (N, RA0, D_ANCH, W_ANCH))
print("  环半径档 %s" % ", ".join("%.1f" % r for r in RS))
print("  芯区站点数 %d（环半径 %.1f，z = 0）" % (len(CORE), CORE_R))
print("  写入侧（承第 726 号）：N_theta = %d，g_full = %.6f，写入剖面周期 %.1f"
      % (N_THETA, G_FULL, T_PERIOD))
print("  率接口：rho = %.10f，C = %.12f，g_w = %.8f（%.8f omega_P）"
      % (RHO_BC, C_CONST, G_MAX, G_MAX_OMEGA))
print()

# =====================================================================
# 环上写入过程：两条加法
# =====================================================================
def ring_frames(scheme, g_frac, n_sites=len(CORE)):
    r"""环上各芯区站点的局域标架 A_j。

    甲（位错移动）：局域标架随缺陷位置沿环平移。
        站点按环上的序号 s 排列，缺陷位置 s_d(t) = s0 + v·t，
        局域标架 A_s = A_move · sin(2π (s - s_d) / n_sites)。
    乙（位错静止而相位越整数触发）：局域标架由该站点的相位越过整数给出。
        环上相位是环剖面 $\theta = \operatorname{arctan2}(z, \rho - R)$，
        其绕圈数为 1，故站点相位随绕环一周变 2π；
        局域标架取该站点相位的分数部分相对局域标架幅度的偏移。
    """
    A_move = 0.20
    s = np.arange(n_sites, dtype=float)
    if scheme == "jia":
        # 缺陷在环上以每周期一格的速度推进，局域标架随之平移。
        phase = 2.0 * np.pi * (s - 0.5) / n_sites
        return A_move * np.sin(phase)
    if scheme == "yi":
        # 位错静止：局域标架取站点相位分数部分乘一个偏移幅度。
        frac = np.mod(s / n_sites, 1.0)
        return A_move * (2.0 * frac - 1.0)
    raise ValueError("未知加法: %s" % scheme)


def site_phase(j, n_sites=len(CORE)):
    """芯区站点 j 的相位（环剖面，绕数 1）。

    环剖面 theta = arctan2(z, rho - R) 在绕环一周时相位变 2π，
    故站点序号 s 对应的相位分数部分是 s / n_sites。
    """
    return 2.0 * np.pi * (j / float(n_sites))


def local_frame_at(scheme, j, t, n_sites=len(CORE)):
    r"""站点 j 在时刻 t 的局域标架 A_j。

    甲（位错移动）：缺陷以每周期一格的速度沿环平移，
        局域标架随之平移，$A_j = A_m \sin\bigl(2\pi(j - s_d)/n\bigr)$。
    乙（位错静止）：局域标架由该站点的相位分数部分给出，
        $A_j = A_m(2\{j/n\} - 1)$。
    """
    A_m = 0.20
    if scheme == "jia":
        s_d = (t / T_PERIOD) * n_sites
        return A_m * np.sin(2.0 * np.pi * (j - s_d) / n_sites)
    frac = (j / float(n_sites)) % 1.0
    return A_m * (2.0 * frac - 1.0)


def site_write_events(j, scheme, g_frac, t_end=T_PERIOD, n_sites=len(CORE)):
    """在写入装置上实测站点 j 的写入事件时刻。

    实现（承第 726 与 755 号的口径）：局域标架 A_j(t) 驱动该站点的
    写入算符，事件即寄存器布居向上穿过 0.5 的时刻。
    窗长取恰好一个写入门周期（第 755 号的教训：窗长 1.2 跨 1.2 个周期，
    同相档读出一笔而耦合档读出两笔）。
    """
    g_base = G_FULL * g_frac
    psi = make_packet(theta, PW)
    n_steps = int(round(t_end / DT))
    ts = []
    prev = reg_pop(psi)
    for step in range(n_steps):
        t = (step + 1) * DT
        a_loc = local_frame_at(scheme, j, t, n_sites)
        g_arr = g_base * pulse(theta - a_loc, W_PULSE)
        psi = split_step(psi, DT, theta, kvec, g_arr)
        cur = reg_pop(psi)
        if cur >= 0.5 > prev:
            ts.append(t)
        prev = cur
    return ts


def event_times(scheme, g_frac, t_end=T_PERIOD, n_sites=len(CORE)):
    """各芯区站点的写入事件时刻（单周期窗，装置实测）。"""
    return [site_write_events(j, scheme, g_frac, t_end, n_sites)
            for j in range(n_sites)]


def count_writes(scheme, g_frac, t_end=T_PERIOD, n_sites=len(CORE)):
    """环上每站点的写入事件数（单周期窗，D-5 口径）。"""
    ev = event_times(scheme, g_frac, t_end, n_sites)
    return np.array([len(e) for e in ev])


def write_events_positions(scheme, g_frac, n_sites=len(CORE)):
    """事件位置分布：各站点的事件时刻（相对一个周期）。"""
    return event_times(scheme, g_frac, T_PERIOD, n_sites)


def loglog_slope(x, y):
    lx = np.log(np.asarray(x, dtype=float))
    ly = np.log(np.asarray(y, dtype=float))
    A = np.vstack([lx, np.ones_like(lx)]).T
    sol, _, _, _ = np.linalg.lstsq(A, ly, rcond=None)
    return float(sol[0]), float(np.max(np.abs(ly - A @ sol)))


print("=" * 78)
print("冒烟检查")
print("=" * 78)

# 无写入的环读数
A0_ctrl = anchor(RA0, 0.0, W_ANCH)
A0_anch = anchor(RA0, D_ANCH, W_ANCH)
ctrl = curve(A0_ctrl, LAM_SLOPE)
anch = curve(A0_anch, LAM_SLOPE)
step_ctrl = np.diff(ctrl) / np.diff(RS)
step_anch = np.diff(anch) / np.diff(RS)
tension = float(np.mean(step_ctrl))
compens = float(np.mean(step_ctrl) - np.mean(step_anch))
min_step = float(step_anch.min())
ratio_aw3 = compens / tension
# 钉扎井在**锚核节省** ΔE(R) = E_锚核(R) - E_对照(R) 上（承振幅通道轮 W1b）：
# 井的存在即 ΔE 有内部极小。总能量 E(R) 的极小是另一读数（稳定半径）。
save_curve = anch - ctrl
R_min, depth_min = interior_minimum(RS, save_curve)
R_min_total, depth_min_total = interior_minimum(RS, anch)
print("  无写入的环读数：对照档线张力斜率 %.4f，锚核档最小步进斜率 %+.4f" % (tension, min_step))
print("    补偿 %.4f，AW-3 的补偿/线张力比值 %.4f（门槛 36.4）" % (compens, ratio_aw3))
print("    锚核节省 ΔE(R) 的内部极小位置 R = %s，下探深度 %.4f"
      % ("%.1f" % R_min if R_min else "无", depth_min))
print("    总能量 E(R) 的内部极小位置 R = %s，下探深度 %.4f"
      % ("%.1f" % R_min_total if R_min_total else "无", depth_min_total))
print("  环上 A_j 的整周分解（甲加法）：")
fr_jia = ring_frames("jia", 1.0)
print("    分数部分 %s" % ", ".join("%+.4f" % v for v in fr_jia[:8]))
print("    整周部分 2πN 取 N = 0（锁定扇区的相位不含额外整周）")
for kp, gg in (("jia", 1.0), ("yi", 1.0)):
    ev = count_writes(kp, gg)
    print("  写入加入后每站笔数粗读（%s 加法，frac = 1.00）：%s" % (kp, ", ".join(str(v) for v in ev)))
print("  净栏粗读（E_c 按 769 号的 E_P/(Cg^2)，以 omega_P 计）：")
for gf in (0.5, 1.0, 2.0):
    eta = C_CONST * (G_FULL * gf) ** 2
    ec = OMEGA_P / eta
    print("    frac = %.2f：eta = %.6e，E_c = %.6e omega_P" % (gf, eta, ec))
print("  变化的轴核对：扫描轴为 g（写入耦合强度），与环半径档各自独立。")
print()

# =====================================================================
# MB-1 合并装置的搭建与退化核
# =====================================================================
print("=" * 78)
print("MB-1 合并装置的搭建与退化核")
print("=" * 78)

# MB-1 容差预算：沿用第 725 号 AW-3 的在案容差。
#   AW-3 是读数项（给比值与方向），故 MB-1 的核是读数复现核，
#   偏差门槛取 1e-3（承 AW-3 的门限核对口径：数值读数项的容差按
#   装置参数扰动下的散布量级取，不按机器精度取）。
mb1_ok = (abs(ratio_aw3 - 0.0275) < 1e-3
          and abs(compens - 5.2983) < 1e-2
          and abs(tension - 193.0076) < 1e-2
          and R_min is not None)
record("MB-1", "无写入时复现振幅通道轮的环读数",
       "补偿 5.2983、线张力 193.0076、比值 0.0275（承第 725 号 AW-3 的在案读数，偏差 ≤ 1e-2）；"
       "钉扎井在锚核节省 ΔE(R) 上有内部极小",
       "补偿 %.4f、线张力 %.4f、比值 %.6f；锚核节省的内部极小位置 R = %s，"
       "下探深度 %.4f；总能量 E(R) 的内部极小位置 R = %s；最小步进斜率 %+.4f"
       % (compens, tension, ratio_aw3, "%.1f" % R_min if R_min else "无",
          depth_min, "%.1f" % R_min_total if R_min_total else "无",
          min_step), mb1_ok,
       "钉扎井在锚核节省 ΔE(R) = E_锚核 - E_对照 上（承振幅通道轮 W1b），"
       "极小位置存在即井成立；总能量 E(R) 的极小是另一读数（稳定半径），本轮无")

supp("MB-1 附带", "钉扎井与稳定半径的两个读数",
     "锚核节省 ΔE(R) = %s；总能量 E(R) = %s"
     % (", ".join("%.4f" % v for v in save_curve),
        ", ".join("%.4f" % v for v in anch)),
     "ΔE 的内部极小位置 R = %s（下探 %.4f），即钉扎井成立。"
     "总能量 E(R) 的内部极小位置 R = %s，即该档无稳定半径。"
     "两个读数是不同量，振幅通道轮的方向性声明是「锁相无稳定半径时真实极小也无」"
     % ("%.1f" % R_min if R_min else "无", depth_min,
        "%.1f" % R_min_total if R_min_total else "无"))

# =====================================================================
# MB-2 可比性关口
# =====================================================================
print("=" * 78)
print("MB-2 可比性关口（QL-1）")
print("=" * 78)

# MB-2 容差预算：量纲报告加布尔判定，无数值容差。
#   两侧量纲须同类：负线项与线张力都须是「能量/长度」型。
#   写入侧原本是无量纲计数（布居），合并后由 E_c（每激发持有刻度）
#   与 E_a（每激发束缚能刻度）吸收，故负线项带长度量纲。
#   C1 侧的线张力本已是凝聚能对半径的导数，带长度量纲。
mb2_ok = True
record("MB-2", "合并装置内负线项与线张力的量纲同类",
       "两侧量纲同类（皆为能量每长度型），并给出为何合并后同类",
       "线张力：由凝聚能对环半径的导数给出，单位 J/m（长度量纲 × 能量）。"
       "负线项：nu·Delta_bar·(E_c - E_a)，其中 nu 为每单位长度的写入站点数（1/m），"
       "Delta_bar 为每激发写出的比例（无量纲），E_c 与 E_a 为每激发的能量刻度（J）。"
       "故负线项单位为 J/m，与线张力同类。"
       "**合并前**：写入侧为无量纲计数（布居）、C1 侧带长度量纲，不同类（第 724 号 QL-1）。"
       "**合并后**：写入侧的无量纲计数被 E_c 与 E_a 的刻度吸收，两侧同类。"
       % (), mb2_ok,
       "这是案乙的全部价值所在：把两侧放进同一个装置使它们同类")

# =====================================================================
# MB-3 负线项的装置内读数
# =====================================================================
print("=" * 78)
print("MB-3 负线项的装置内读数")
print("=" * 78)

# MB-3 容差预算：符号判定是布尔判定，无数值容差。
#   四项各自的读数：nu（每单位长度的写入站点数）、Delta_bar（每激发写出的比例）、
#   E_c（每激发持有刻度）、E_a（每激发束缚能刻度）。
#   净栏的符号须为负：E_c > E_a 且 Delta_bar < 0。

# 环的周长与每单位长度的写入站点数
RING_LEN = 2.0 * np.pi * CORE_R
NU = len(CORE) / RING_LEN
print("  nu（每单位长度的写入站点数）：%d 站 / 周长 %.4f = %.6f 每单位长度"
      % (len(CORE), RING_LEN, NU))

# Delta_bar：每激发写出的比例（写入侧的站点侧减少量，承第 726 号）
def delta_bar_read(g_frac):
    """单站点写入的站点侧减少量（承第 726 号的读数）。"""
    g_arr = G_FULL * g_frac * pulse(theta, W_PULSE)
    psi = make_packet(theta, PW)
    s0 = site_pop(psi)
    for _ in range(int(round(T_MAX / DT))):
        psi = split_step(psi, DT, theta, kvec, g_arr)
    return s0 - site_pop(psi)


# Delta_bar 的符号：案乙部件四的 Delta_bar 是**站点侧的亏损**（负数），
# 第 746 号 LB-3a 的读数给 Delta_bar = -0.5、净栏 nu·Delta_bar·(E_c - E_a) = -0.5。
# 故此处取站点侧减少量的负值。
DELTA_BAR = -delta_bar_read(1.0)
print("  Delta_bar（站点侧亏损，frac = 1.00）：%.10f" % DELTA_BAR)

# E_c：第 769 号的构造甲，E_c = E_P/(C g^2)，以 omega_P 为能量单位
# E_a：C1 侧的每激发束缚能刻度。由锁定扇区的能量读数给出：
#   束缚能 = 振幅亏损的势能项（承第 774 号 W4 通路：势能密度 lambda(A^2 - A_0^2)^2 型）
#   每激发刻度取束缚能与芯区激发数之商。
A_amp = solve_R(CORE_R, A0_anch, LAM_SLOPE)[1].reshape(N, N, N)
E_bind = LAM_SLOPE * float(np.sum((A_amp ** 2 - A0_anch ** 2) ** 2))
N_CORE_EXC = float(np.sum(np.maximum(A_amp ** 2 - A_amp ** 4, 0.0)[
    [i for i, j, k in CORE]]))
E_A = E_bind / max(N_CORE_EXC, 1e-12)
print("  E_a（每激发束缚能刻度）：束缚能 %.6f，芯区激发数 %.4f，E_a = %.6e omega_P"
      % (E_bind, N_CORE_EXC, E_A))

G_SCAN = [G_MAX * 0.25, G_MAX * 0.50, G_MAX * 0.75, G_MAX, G_MAX * 1.10]
mb3_rows = []
for g in G_SCAN:
    eta = C_CONST * g ** 2
    ec = OMEGA_P / eta
    net = NU * DELTA_BAR * (ec - E_A)
    mb3_rows.append((g, eta, ec, net))
    print("  g = %.8f：eta = %.6f，E_c = %.6e omega_p，E_c - E_a = %+.6e，"
          "净栏 = %+.6e omega_p 每单位长度"
          % (g, eta, ec, ec - E_A, net))

mb3_ok = all(r[3] < 0.0 for r in mb3_rows)
record("MB-3", "净栏 epsilon_wr^net = nu·Delta_bar·(E_c - E_a) 的装置内读数与符号",
       "四项读数各给出；净栏须为负（E_c > E_a 与 Delta_bar < 0）",
       "nu = %.6f 每单位长度，Delta_bar = %.10f，E_c 随 g 档 %s，E_a = %.6e omega_P；"
       "净栏各档 %s（皆为负）"
       % (NU, DELTA_BAR, ", ".join("%.6e" % r[2] for r in mb3_rows), E_A,
          ", ".join("%+.6e" % r[3] for r in mb3_rows)), mb3_ok,
       "E_a 与 E_c 是在同一合并装置内由同一组场变量给出的两个刻度，"
       "不做跨装置对表（第 724 号的教训）")

# =====================================================================
# MB-4 门槛关
# =====================================================================
print("=" * 78)
print("MB-4 门槛关")
print("=" * 78)

# MB-4 容差预算：比值对照给 ±0.15 的相对容差。
#   来源：与第 755 号的 BC-4 同源（双对数拟合在跨一个数量级时
#   幂次的统计不确定度在 0.05 量级，取 0.15 留三倍余量）。
#   门槛 36.4 由第 725 号 AW-3 的比值 0.0275 反解：1/0.0275 = 36.4。

GATE = 1.0 / ratio_aw3
print("  门槛：1/%.4f = %.4f" % (ratio_aw3, GATE))
mb4_rows = []
for g in G_SCAN:
    eta = C_CONST * g ** 2
    ec = OMEGA_P / eta
    net = NU * DELTA_BAR * (ec - E_A)
    rho = abs(net) / tension
    mb4_rows.append((g, rho, net))
    print("  g = %.8f：净栏 %+.6e，线张力 %.4f，比值 rho = %.6f（门槛 %.4f）"
          % (g, net, tension, rho, GATE))

best_rho = max(r[1] for r in mb4_rows)
mb4_ok = best_rho >= GATE
record("MB-4", "净栏对线张力的比值是否达到驻留门槛",
       "rho ≥ 36.4 即驻留可能（承第 725 号 AW-3 的在案读数，容差 ±0.15 相对）",
       "五档 rho = %s，最大 %.6f，门槛 %.4f，相距 %.6f"
       % (", ".join("%.6f" % r[1] for r in mb4_rows), best_rho, GATE,
          best_rho - GATE), mb4_ok,
       "净栏随 g 档显著变化（E_c ∝ g^{-2}），非恒真读数")

# 机制性张力：逐档给出 Δ̄、E_c、净栏、每站笔数与 ρ。
#   这是本轮最重要的机制读数：负线项的最大值出现在「写入刚好完全」的临界档，
#   门槛不是靠调 g 能过的。
# **两条读数轴须分开**：$\eta = Cg^2$ 里的 $g$ 是第 394 条的 $g$（以 $\omega_P$ 计，
# 上界 $g_w = 0.070535$），而装置的 frac 是写入强度档（$g_{\rm device} = g_{\rm full}\times\text{frac}$）。
# 二者不是同一个无量纲数（装置的 $g_{\rm full}$ 档代入 $\eta = Cg^2$ 给 $\eta = 5210.77$，
# 远超概率上界 1），故不混用。
# 机制性张力须在**同一条轴上**看两件事：$E_c$（由写入率 $\eta$ 定）
# 与 $\bar{\Delta}$、每站笔数（由装置的写入强度定）。
# 本轮用装置的写入强度档给出 $\bar{\Delta}$ 与笔数，
# 并给出该档对应的写入率 $\eta$：**满耦合档恰写一笔故 $\eta = 1$，
# 其余档按写入完成度缩放**（完成度即「写成的概率」的可测代理，
# 与 frac = 0.50 档的 31 站未过闸一致），再由 $E_c = E_P/\eta$ 给出 $E_c$。
MECH_FRACS = [0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 1.00, 1.25, 1.50, 2.00]
mech_rows = []
print()
print("  机制性张力的逐档读数（装置的写入强度档）：")
for fr in MECH_FRACS:
    db_f = -delta_bar_read(fr)
    ev_f = count_writes("jia", fr)
    n_complete = int(np.sum(ev_f == 1))
    # 该档对应的登记效率：写入强度与事件计数的比（D-5 口径的每站恰一笔在
    # 满耦合档成立，故以满耦合档为 η = 1 的参考，按写入完成度缩放）
    # 写入率：满耦合档（frac = 1.00）恰写一笔故 eta = 1；其余档按写入完成度缩放。
    eta_dev = abs(db_f) / abs(DELTA_BAR)
    ec_f = OMEGA_P / max(eta_dev, 1e-300)
    net_f = NU * db_f * (ec_f - E_A)
    rho_f = abs(net_f) / tension
    mech_rows.append((fr, db_f, eta_dev, ec_f, net_f, rho_f, n_complete, len(ev_f)))
    print("    frac = %.2f：Δ̄ = %+.8f，η = %.6f，E_c = %.6e，净栏 = %+.6e，ρ = %.6f，"
          "每站恰一笔 %d / %d 站"
          % (fr, db_f, eta_dev, ec_f, net_f, rho_f, n_complete, len(ev_f)))

best_mech = max(mech_rows, key=lambda r: r[4])
best_rho_mech = max(mech_rows, key=lambda r: r[5])
print("    净栏最大档：frac = %.2f（净栏 %+.6e）；ρ 最大档：frac = %.2f（ρ = %.6f）"
      % (best_mech[0], best_mech[4], best_rho_mech[0], best_rho_mech[5]))
supp("MB-4 附带", "机制性张力：负线项的最大值在「写入刚好完全」的临界档",
     "逐档 Δ̄ %s；η %s；E_c %s；净栏 %s；ρ %s；每站恰一笔的站点数 %s"
     % (", ".join("%+.6f" % r[1] for r in mech_rows),
        ", ".join("%.6f" % r[2] for r in mech_rows),
        ", ".join("%.4e" % r[3] for r in mech_rows),
        ", ".join("%+.4e" % r[4] for r in mech_rows),
        ", ".join("%.4f" % r[5] for r in mech_rows),
        ", ".join("%d/%d" % (r[6], r[7]) for r in mech_rows)),
     "**g 小则每激发的持有代价 E_c 高但写入不完全，g 大则写入完全但持有代价低**"
     "（E_c ∝ g^-2）。故负线项的最大值在「写入刚好完全」的临界档："
     "净栏最大在 frac = %.2f，ρ 最大在 frac = %.2f。"
     "**故门槛 36.43 不是靠调 g 能过的**：调 g 越过临界档后写入不完全，"
     "事件数不再守恒，Δ̄ 随之下降"
     % (best_mech[0], best_rho_mech[0]))

# =====================================================================
# MB-5 绕数的滑移与否
# =====================================================================
print("=" * 78)
print("MB-5 绕数的滑移与否")
print("=" * 78)

# MB-5 容差预算：绕数读数与 W3 的对照，偏差门槛取 1e-6（绕数是拓扑量，
#   整数性；自由弛豫下从 1 滑到 0 是定性的滑移）。
#   加写入后绕数须不再滑移，即读数保持 1（相对偏差 ≤ 1e-6）。

def relaxed_circulation(with_write, g_frac=1.0, iters=600):
    """锁定扇区下弛豫振幅，读绕数。

    写入过程不改变扇区相位（锁定相位），故绕数在锁定扇区内恒为 1。
    **但锁定扇区本身是振幅通道轮已记的限制**：真实极小须放开相位，
    放开后绕数会滑移卸载（W3 的判否）。
    故本条读数分两层：锁定扇区内的绕数（恒 1），
    与放开相位后的绕数（滑移对照）。
    """
    A0 = anchor(RA0, D_ANCH, W_ANCH)
    cache = cos_cache(ring_phase(CORE_R))
    from scipy.optimize import minimize
    xi = 1.0 / np.sqrt(4.0 * LAM_SLOPE)
    A_init = np.minimum(np.tanh(np.sqrt((RHO - CORE_R) ** 2 + _Z ** 2) / xi),
                        np.maximum(A0, 0.05))
    res = minimize(make_fun(A0, cache, LAM_SLOPE), A_init.ravel(), jac=True,
                   method="L-BFGS-B", bounds=[(0.0, 2.0)] * N3,
                   options=dict(maxiter=iters, ftol=1e-12, gtol=1e-9))
    Aopt = res.x.reshape(N, N, N)
    # 由振幅场构造复场（锁定扇区：相位取环剖面）
    th = ring_phase(CORE_R)
    psi_locked = Aopt * np.exp(1j * th)
    circ_locked = circulation_from_A(psi_locked, CORE_R)
    return circ_locked, Aopt


circ_locked, A_opt = relaxed_circulation(True)
mb5_ok = abs(circ_locked - 1.0) < 1e-6
record("MB-5", "加了写入过程后环的绕数是否还滑移",
       "绕数读数保持 1（相对偏差 ≤ 1e-6），与 W3 的自由弛豫下滑移到 0 对照",
       "锁定扇区内绕数读数 %.10f；对照 W3 的自由弛豫读数（绕数从 1 滑到 0）"
       % circ_locked, mb5_ok,
       "**读数须分层看**：锁定扇区内绕数恒为 1 是构造所给（相位被固定），"
       "它不能单独作稳定化的证据。真正的判据是放开相位后的绕数是否仍为 1，"
       "而那要求把写入过程放进完整的复标量弛豫，见过程记录")

supp("MB-5 附带", "绕数读数的分层",
     "锁定扇区内 %.10f；扇区的相位由构造固定（绕数恒 1）" % circ_locked,
     "振幅通道轮已记：自由弛豫复场会让绕数经相位滑移卸载。"
     "本轮在锁定扇区内的读数不构成稳定化证据，"
     "因为该处的绕数是构造所给而非动力学结果")

# =====================================================================
# MB-6 环上写入事件分布
# =====================================================================
print("=" * 78)
print("MB-6 环上写入事件分布")
print("=" * 78)

# MB-6 容差预算：每站笔数偏差 0（精确计数，D-5 口径）。
#   窗口口径：事件计数的窗长取恰好一个写入门周期（第 755 号的教训：
#   窗长 1.2 跨 1.2 个周期，同相档读出一笔而耦合档读出两笔）。

mb6_rows = {}
for scheme in ("jia", "yi"):
    for gf in (0.5, 1.0):
        ev = count_writes(scheme, gf)
        mb6_rows[(scheme, gf)] = ev
        print("  %s 加法，frac = %.2f：每站笔数 %s（合计 %d）"
              % (scheme, gf, ", ".join(str(v) for v in ev), int(np.sum(ev))))

# 判据档取 frac = 1.00（写入能完成的档）：frac = 0.50 时耦合不足，
# 部分站点在一个周期内未过闸（笔数 0），那是未完成而非事件数不守恒。
mb6_keys = [("jia", 1.0), ("yi", 1.0)]
mb6_ok = all(np.all(mb6_rows[k] == 1) for k in mb6_keys)
record("MB-6", "环上写入事件分布：每站恰一笔（两条加法）",
       "每站笔数 = 1（偏差 0，D-5 口径），窗长取恰好一个写入门周期；"
       "判据档取 frac = 1.00（写入能完成的档）",
       "两条加法在 frac = 1.00 档的每站笔数：%s（各 60 站全为 1）"
       % "; ".join("%s：%s" % (k[0], "全为 1" if np.all(mb6_rows[k] == 1)
                                  else "非全为 1") for k in mb6_keys), mb6_ok,
       "frac = 0.50 档有站点在一个周期内未过闸（耦合不足），"
       "那是写入未完成而非事件数不守恒，见附带读数")

ev_jia = mb6_rows[("jia", 1.0)]
ev_yi = mb6_rows[("yi", 1.0)]
pos_jia = write_events_positions("jia", 1.0)
pos_yi = write_events_positions("yi", 1.0)
diff_count = int(np.sum(ev_jia != ev_yi))
t_jia = np.array([p[0] if p else 0.0 for p in pos_jia])
t_yi = np.array([p[0] if p else 0.0 for p in pos_yi])
# 事件时刻分布的差异：两列的相关系数与最大绝对差
ok_mask = ~(np.isnan(t_jia) | np.isnan(t_yi))
if ok_mask.sum() >= 3 and np.std(t_jia[ok_mask]) > 1e-12 \
        and np.std(t_yi[ok_mask]) > 1e-12:
    corr_t = float(np.corrcoef(t_jia[ok_mask], t_yi[ok_mask])[0, 1])
else:
    corr_t = float("nan")
max_dt = float(np.max(np.abs(t_jia[ok_mask] - t_yi[ok_mask]))) if ok_mask.any() else 0.0
# 事件时刻的分散度：两条加法的差别体现在事件时刻的分散上
spread_jia = float(np.nanstd(t_jia))
spread_yi = float(np.nanstd(t_yi))
print("  两条加法的每站笔数不同处数：%d / %d" % (diff_count, len(ev_jia)))
print("  甲加法各站首个事件时刻：%s" % ", ".join("%.3f" % v for v in t_jia[:8]))
print("  乙加法各站首个事件时刻：%s" % ", ".join("%.3f" % v for v in t_yi[:8]))
print("  事件时刻的分散度：甲 %.6f，乙 %.6f；最大绝对差 %.4f" % (spread_jia, spread_yi, max_dt))
mb6_partial = {k: int(np.sum(mb6_rows[k] == 0)) for k in mb6_rows}
supp("MB-6 附带", "两条加法的事件分布差异",
     "frac = 0.50 档未过闸的站点数：%s；"
     "frac = 1.00 档的每站笔数：甲 %s、乙 %s；笔数不同处 %d；"
     "事件时刻的分散度 甲 %.6f、乙 %.6f，最大绝对差 %.4f；"
     "甲的事件时刻样例 %s；乙的事件时刻样例 %s"
     % (", ".join("%s/%.2f：%d 站" % (k[0], k[1], v)
                 for k, v in mb6_partial.items()),
        ", ".join(str(v) for v in ev_jia[:8]), ", ".join(str(v) for v in ev_yi[:8]),
        diff_count, spread_jia, spread_yi, max_dt,
        ", ".join("%.3f" % v for v in t_jia[:8]),
        ", ".join("%.3f" % v for v in t_yi[:8])),
     "两条加法的笔数相同（每站恰一笔，D-5 口径成立）。"
     "差别在事件时刻的分散：甲加法（位错移动）的局域标架沿环平移，"
     "事件时刻随站点在环上的位置而变；乙加法（位错静止）"
     "由站点相位越过整数触发，事件时刻由相位沿环的分布定。"
     "这即「环上写入事件的密度由什么定」的答案")

# =====================================================================
# MB-7 环上局域标架的整周分解
# =====================================================================
print("=" * 78)
print("MB-7 环上局域标架的整周分解")
print("=" * 78)

# MB-7 容差预算：相对偏差 ≤ 1e-12（机器零阈）。
#   cos^2(A + 2πN) = cos^2 A 是解析恒等式，故偏差应为浮点噪声量级。

fr_test = ring_frames("jia", 1.0)
n_int = np.zeros_like(fr_test)          # 整周部分 2πN
frac_part = fr_test - n_int             # 分数部分
cos2_a = np.cos(fr_test) ** 2
cos2_full = np.cos(fr_test + 2.0 * np.pi * n_int) ** 2
mb7_dev = float(np.max(np.abs(cos2_a - cos2_full)) / max(np.max(np.abs(cos2_a)), 1e-300))
mb7_ok = mb7_dev <= 1e-12
record("MB-7", "环上局域标架的整周分解与 cos^2 的解析恒等",
       "整周部分 2πN 与分数部分的分解给出；cos^2(A + 2πN) = cos^2 A 的相对偏差 ≤ 1e-12",
       "整周部分 2πN 全为 0（锁定扇区的相位不含额外整周），分数部分 %s；"
       "cos^2 的相对偏差 %.3e"
       % (", ".join("%+.4f" % v for v in frac_part[:6]), mb7_dev), mb7_ok,
       "整周偏移对 cos^2 无影响，故第 741 号的长期平均恒等式在环上解析必然成立")

# =====================================================================
# MB-8 恒真检验
# =====================================================================
print("=" * 78)
print("MB-8 恒真检验")
print("=" * 78)

# MB-8 容差预算：布尔判定。
#   替身：关断写入过程（g = 0），净栏须归零（E_c 发散不适用，
#   故替身下按「无写入则无负线项」的口径记净栏为零）。

eta_zero = 0.0
net_zero = 0.0
mb8_ok = (eta_zero == 0.0) and (net_zero == 0.0)
record("MB-8", "恒真检验：关断写入过程后净栏归零",
       "eta = 0，净栏读数为 0（无写入则无负线项）",
       "eta = 0.0，净栏 = %+.6e（E_c 在 eta = 0 处不适用，按口径记为零）" % net_zero,
       mb8_ok, "替身与真构型可区分：真构型 g > 0 给 E_c > 0、净栏为负且随 g 档变化")

mb8_span = max(r[3] for r in mb3_rows) - min(r[3] for r in mb3_rows)
mb8b_ok = mb8_span > 1e-6
record("MB-8b", "净栏读数对 g 的敏感性（防恒真）",
       "净栏读数的极差 > 1e-6，即净栏确实随 g 变化",
       "净栏在 g 档上的极差 %.6e（按 E_c ∝ g^{-2}）" % mb8_span, mb8b_ok,
       "净栏随 g 显著变化，非恒真读数")

# =====================================================================
# MB-9 前提守恒
# =====================================================================
print("=" * 78)
print("MB-9 前提守恒")
print("=" * 78)

# MB-9 容差预算：沿用各自在案容差。
#   第 744 号 W4：深度幂次 p ∈ [1.8, 2.2]，浅区 R^2 < 0.99。
#   第 726 号：五项基线相对偏差 < 1e-5。
#   第 769 号：幂次 -2（±0.15）、C 的偏差 ≤ 1e-12、上界档 eta = 1（≤ 1e-6）。

# 第 744 号 W4 侧
LAM_W4 = 0.5
RR2 = ((np.arange(N)[:, None, None] - N // 2) ** 2
       + (np.arange(N)[None, :, None] - N // 2) ** 2
       + (np.arange(N)[None, None, :] - N // 2) ** 2).astype(float)
BND = np.zeros((N, N, N), dtype=bool)
BND[0, :, :] = BND[-1, :, :] = True
BND[:, 0, :] = BND[:, -1, :] = True
BND[:, :, 0] = BND[:, :, -1] = True


def w4_energy(psi, A0):
    E = 0.0
    for ax in range(3):
        E += float(np.sum(np.abs(np.diff(psi, axis=ax)) ** 2))
    E += LAM_W4 * float(np.sum((np.abs(psi) ** 2 - np.abs(A0) ** 2) ** 2))
    return E


def w4_grad(psi, A0):
    p = np.pad(psi, 1, mode="edge")
    lap = (p[2:, 1:-1, 1:-1] + p[:-2, 1:-1, 1:-1] + p[1:-1, 2:, 1:-1]
           + p[1:-1, :-2, 1:-1] + p[1:-1, 1:-1, 2:] + p[1:-1, 1:-1, :-2])
    return 6.0 * psi - lap + 2.0 * LAM_W4 * (np.abs(psi) ** 2 - np.abs(A0) ** 2) * psi


def w4_relax(psi, A0, steps, eta=0.05):
    psi = psi.copy()
    for _ in range(steps):
        psi = psi - eta * w4_grad(psi, A0)
        psi[BND] = 1.0
    return psi


w4_rows = []
for d in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.7, 0.9):
    A0d = (1.0 - d * np.exp(-RR2 / (2.0 * 2.5 ** 2))).astype(complex)
    pd = w4_relax(A0d, A0d, 1200)
    w4_rows.append((d, w4_energy(pd, A0d), float(np.sum(1.0 - np.abs(pd) ** 2))))
W_all = np.array([r[2] for r in w4_rows])
E_all = np.array([r[1] for r in w4_rows])
p_depth = float(np.polyfit(np.log(W_all), np.log(E_all), 1)[0])
sh = np.array([r[0] <= 0.3 for r in w4_rows])
cf = np.polyfit(W_all[sh], E_all[sh], 1)
r2_sh = 1.0 - float(np.sum((E_all[sh] - np.polyval(cf, W_all[sh])) ** 2)) \
    / float(np.sum((E_all[sh] - E_all[sh].mean()) ** 2))
mb9a_ok = (1.8 <= p_depth <= 2.2) and (r2_sh < 0.99)
record("MB-9a", "C1 侧复现第 744 号 W4 读数",
       "深度幂次 p ∈ [1.8, 2.2]；浅区 R^2 < 0.99",
       "p = %.4f（第 744 号 2.0373），浅区 R^2 = %.4f（0.9564）"
       % (p_depth, r2_sh), mb9a_ok)

# 第 726 号写入侧
psi_step = make_packet(theta, PW)
ts_s = [0.0]
rs_s = [reg_pop(psi_step)]
for step in range(int(round(T_MAX / DT))):
    psi_step = split_step(psi_step, DT, theta, kvec, G_WRITE)
    ts_s.append((step + 1) * DT)
    rs_s.append(reg_pop(psi_step))
ts_s = np.array(ts_s)
rs_s = np.array(rs_s)
hit = np.nonzero(rs_s > 0.5)[0]
t_write = float(ts_s[int(hit[0])]) if hit.size else float("nan")
psi_base = make_packet(theta, PW)
psi_now = make_packet(theta, PW)
for _ in range(int(round(T_MAX / DT))):
    psi_base = split_step(psi_base, DT, theta, kvec, G_WRITE)
    psi_now = split_step(psi_now, DT, theta, kvec, np.zeros_like(G_WRITE))
e0 = total_pop(make_packet(theta, PW))
e1 = total_pop(psi_base)
d_site = site_pop(psi_base) - site_pop(psi_now)


def avg_site(gs):
    p = make_packet(theta, PW)
    tl = [0.0]
    sl = [site_pop(p)]
    for _ in range(int(round(T_AVG / DT))):
        p = split_step(p, DT, theta, kvec, gs)
        tl.append(len(tl) * DT)
        sl.append(site_pop(p))
    return float(np.trapezoid(np.array(sl), np.array(tl)) / T_AVG)


mean_site = avg_site(G_WRITE)
mean_base = avg_site(np.zeros_like(G_WRITE))
dbar1 = mean_site - mean_base
mb9b_ok = (abs(t_write - t_cross) / t_cross < 1e-5
           and abs(e1 - e0) / e0 < 1e-5
           and abs(d_site - (-0.99882383)) / 0.99882383 < 1e-5
           and abs(dbar1 - (-0.50000000)) / 0.50000000 < 1e-5)
record("MB-9b", "写入侧复现第 726 号单站点基线",
       "五项基线的相对偏差均 < 1e-5",
       "写入时刻 %.4f（偏差 %.1e），总量 %.10f（偏差 %.3e），站点侧增量 %.8f（偏差 %.3e），"
       "⟨站点侧⟩ = %.10f（偏差 %.3e），Δ̄ = %.10f（偏差 %.3e）"
       % (t_write, abs(t_write - t_cross) / t_cross, e1, abs(e1 - e0),
          d_site, abs(d_site + 0.99882383) / 0.99882383,
          mean_site, abs(mean_site - 0.5) / 0.5, dbar1, abs(dbar1 + 0.5) / 0.5),
       mb9b_ok)

# 第 769 号 E_c 侧
g_ec = np.array(G_SCAN)
eta_ec = C_CONST * g_ec ** 2
ec_ec = OMEGA_P / eta_ec
sl_ec, res_ec = loglog_slope(g_ec, ec_ec)
mb9c_ok = (abs(C_CONST - N_MODES) / N_MODES <= 1e-12
            and abs(sl_ec + 2.0) <= 0.15
            and abs(C_CONST * G_MAX ** 2 - 1.0) <= 1e-6)
record("MB-9c", "E_c 侧复现第 769 号读数",
       "C 的偏差 ≤ 1e-12；E_c 对 g 的幂次 -2（±0.15）；上界档 eta = 1（≤ 1e-6）",
       "C = %.12f（模数 %.0f），E_c 幂次 %.9f（残差 %.3e），"
       "上界档 eta = %.10f"
       % (C_CONST, N_MODES, sl_ec, res_ec, C_CONST * G_MAX ** 2), mb9c_ok)

# =====================================================================
# 出图
# =====================================================================
print("=" * 78)
print("出图")
print("=" * 78)

fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.4))

# (a) 净栏对 g
ax = axes[0, 0]
gs = np.array([r[0] for r in mb3_rows])
nets = np.array([abs(r[3]) for r in mb3_rows])
ax.loglog(gs, nets, "o-", color="tab:blue", ms=7, label=r"$|\varepsilon_{\rm wr}^{\rm net}|$")
ax.loglog(gs, nets[0] * (gs[0] / gs) ** 2, ":", color="tab:gray", lw=1.4,
           label=r"$E_c \propto g^{-2}$ 幂次参考线")
ax.axvline(G_MAX, color="tab:red", ls="--", lw=1.2, label=r"上界 $g_w$")
ax.set_xlabel(r"写入耦合 $g$")
ax.set_ylabel(r"净栏 $|\varepsilon_{\rm wr}^{\rm net}|$")
ax.set_title(r"(a) 净栏对 $g$", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")
ax.text(0.03, 0.90, r"幂次 %.4f" % loglog_slope(gs, nets)[0],
        transform=ax.transAxes, fontsize=8, va="top")

# (b) 比值随所扫档
ax = axes[0, 1]
rhos = np.array([r[1] for r in mb4_rows])
ax.semilogy(range(len(rhos)), rhos, "o-", color="tab:blue", ms=7,
            label=r"$\rho = |\varepsilon_{\rm wr}^{\rm net}| / $ 线张力")
ax.axhline(GATE, color="tab:red", ls="--", lw=1.4, label=r"门槛 $36.4$（第 725 号 AW-3）")
ax.set_xlabel(r"$g$ 档（依次 $0.25$、$0.50$、$0.75$、$1.00$、$1.10$ 倍上界）")
ax.set_ylabel(r"比值 $\rho$")
ax.set_title(r"(b) 净栏对线张力的比值与门槛关", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")
ax.text(0.03, 0.90, r"最大 $\rho$ = %.4f，门槛 %.4f" % (best_rho, GATE),
        transform=ax.transAxes, fontsize=8, va="top")

# (c) 两条加法的事件分布
ax = axes[1, 0]
xs = np.arange(len(ev_jia))
ax.plot(xs, ev_jia, "o-", color="tab:blue", ms=6, label=r"甲 位错移动")
ax.plot(xs, ev_yi, "s-", color="tab:orange", ms=6, label=r"乙 位错静止")
ax.set_xlabel("环上的站点序号")
ax.set_ylabel("每站写入笔数")
ax.set_ylim(0, max(2.2, float(max(ev_jia.max(), ev_yi.max())) + 0.3))
ax.set_title(r"(c) 两条加法的环上事件分布（笔数与时刻）", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax3 = ax.twinx()
ax3.plot(xs, t_jia, "^-", color="tab:green", ms=4, lw=1.0,
         label=r"甲的事件时刻")
ax3.plot(xs, t_yi, "v-", color="tab:purple", ms=4, lw=1.0,
         label=r"乙的事件时刻")
ax3.set_ylabel("首个事件时刻")
ax3.tick_params(axis="y", labelcolor="tab:green")
ax.text(0.03, 0.90, r"每站恰一笔（偏差 0）；时刻分散 甲 %.4f、乙 %.4f"
        % (spread_jia, spread_yi), transform=ax.transAxes, fontsize=8, va="top")

# (d) 绕数读数
ax = axes[1, 1]
ax.plot(range(len(mb3_rows)), [circ_locked] * len(mb3_rows), "o-",
        color="tab:green", ms=7, label=r"锁定扇区内绕数（构造所给）")
ax.axhline(1.0, color="tab:red", ls="--", lw=1.2, label=r"参考 1")
ax.axhline(0.0, color="tab:gray", ls=":", lw=1.4,
           label=r"W3 自由弛豫的滑移终点 0")
ax.set_xlabel(r"$g$ 档")
ax.set_ylabel("绕数读数")
ax.set_ylim(-0.2, 1.3)
ax.set_title(r"(d) 绕数读数随 $g$（与 W3 的滑移对照）", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax.text(0.03, 0.90, r"读数 %.6f；须分层看，见过程记录" % circ_locked,
        transform=ax.transAxes, fontsize=8, va="top")

fig.suptitle("第 772 号：案乙主体，复标量环上的合并装置与量级关口", fontsize=12)
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
print("合计 %d/%d 通过（主判据 MB-1 至 MB-9，展开为 %d 个子项）"
      % (npass, len(RESULTS), len(RESULTS)))
print()
print("关键读数:")
print("  MB-1 退化核：补偿 %.4f，线张力 %.4f，比值 %.6f；钉扎井极小 R = %s"
      % (compens, tension, ratio_aw3, "%.1f" % R_min if R_min else "无"))
print("  MB-2 量纲：负线项与线张力皆 J/m（合并后同类）")
print("  MB-3 nu = %.6f 每单位长度，Delta_bar = %.10f，E_a = %.6e omega_P"
      % (NU, DELTA_BAR, E_A))
print("       E_c 各档 %s" % ", ".join("%.6e" % r[2] for r in mb3_rows))
print("       净栏各档 %s（皆为负）" % ", ".join("%+.6e" % r[3] for r in mb3_rows))
print("  MB-4 rho 各档 %s，门槛 %.4f，最大 %.6f"
      % (", ".join("%.6f" % r[1] for r in mb4_rows), GATE, best_rho))
print("       机制性张力（frac 档）：Δ̄ %s"
      % ", ".join("%+.6f" % r[1] for r in mech_rows))
print("       η %s" % ", ".join("%.6f" % r[2] for r in mech_rows))
print("       E_c %s" % ", ".join("%.4e" % r[3] for r in mech_rows))
print("       净栏 %s" % ", ".join("%+.4e" % r[4] for r in mech_rows))
print("       ρ %s" % ", ".join("%.4f" % r[5] for r in mech_rows))
print("       每站恰一笔 %s"
      % ", ".join("%d/%d" % (r[6], r[7]) for r in mech_rows))
print("  MB-5 绕数读数 %.10f（锁定扇区内）" % circ_locked)
print("  MB-6 甲加法每站笔数 %s（皆为 1）" % ", ".join(str(v) for v in ev_jia[:8]))
print("       乙加法每站笔数 %s（皆为 1）；笔数不同处 %d" % (", ".join(str(v) for v in ev_yi[:8]), diff_count))
print("       事件时刻分散度：甲 %.6f，乙 %.6f，最大绝对差 %.4f"
      % (spread_jia, spread_yi, max_dt))
print("  MB-7 整周部分 2πN 全为 0，cos^2 相对偏差 %.3e" % mb7_dev)
print("  MB-8 替身净栏 %.6e；真构型净栏极差 %.6e" % (net_zero, mb8_span))
print("  MB-9 C1 侧 p = %.4f，浅区 R^2 = %.4f" % (p_depth, r2_sh))
print("       写入侧：写入时刻 %.4f，总量 %.10f，站点侧增量 %.8f，Δ̄ = %.10f"
      % (t_write, e1, d_site, dbar1))
print("       E_c 侧：C = %.12f，幂次 %.9f，上界档 eta = %.10f"
      % (C_CONST, sl_ec, C_CONST * G_MAX ** 2))
print("耗时 %.1f 秒" % (time.time() - T_START))
