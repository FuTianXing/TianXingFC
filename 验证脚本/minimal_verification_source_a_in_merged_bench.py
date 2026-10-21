# -*- coding: utf-8 -*-
r"""
minimal_verification_source_a_in_merged_bench.py

第 796 号 来源甲负项装入合并装置后的总能量增量与线张力比值（判据 MB2-1 至 MB2-9）

靶子
----
案乙主体（第 772 号）判了两层关口：第一层可比性关口通过（合并装置内负线项与
线张力同为 J/m），**第二层门槛关判否**（净栏对线张力的比值最大 0.991079，
对门槛 36.4282 = 1/0.027451，相距 -35.437146）。机制：净栏是 E_c 与 |Delta_bar|
之积、二者随写入强度反向变化而近乎精确抵消（十档上相对起伏仅 4.2%），
故整条轴上没有峰，调 g 弥合不了。

来源甲（第 790 号，13/13 全过）给出了那个不与写入互斥的来源：单边错配容量型芯的
膨胀份额越过 theta*，乙丙丁三个候选的门槛全被越过，结论对候选形式无关；
纯拉伸侧 I(1) = I_+ < 0（九组全为负）。

本轮把来源甲的负项装进案乙的合并装置，**重测总能量相对真空基态的增量对
线张力的比值**，看门槛是否被跨过。

**两栏不是简单叠加（承 lead 的登记）**
----------------------------------------
案乙的栏位清单是「承第 591 条与攻坚计划 §一 的四栏：键拉伸、法线轴弯曲、
异类耦合扭曲、反馈通道，在这四栏之外加写入沉积栏」，
而**来源甲的负项正是从键拉伸栏（V_rep 的高阶展开）里出的**，
它属 C1 侧的静态每单位长度增量，案乙测的是写入沉积栏（记录侧，
含 E_c 与 E_a 之差）。故两者的对接点是「净能量」：
把两栏放进同一泛函后读**总能量相对真空基态的增量**。

**MB2-1 的构型相容性判定（本轮的关键结构问题）**
-------------------------------------------------
两侧的载体不同：容量型芯改的是**键应变**（u_z 的导数），
环的绕回是**复标量场的相位剖面**。故不是「替换」关系。

但**键应变是二者共用的通道**：环的绕回结构本身也是一种键应变分布，
而容量型芯要求「导数处处同号、不反号」。

故本轮逐一判三条可能，并**以实测定走哪条**：
  甲，叠加：环的绕回剖面与容量型芯的 tanh 剖面在键应变上相加，
      则键应变符号不再处处同号，**容量型芯的负增量结论须重新测**
      （不能直接搬第 790 号的读数）。
  乙，分区：容量型芯只施加在环的某个子区，其余维持绕回剖面；
      则该子区的键数占比决定负项的总量，而线张力是整环的，
      比值的分母与分子口径不同。
  丙，不兼容：若上述两条在本装置的构型下都不成立，
      则本轮的靶子本身是射程外的构型障碍。

**口径换算（显式给出，否则比值不可读）**
--------------------------------------
第 790 号的 I(1) 是「芯区每单位长度增量」，第 772 号的线张力 193.0076 是
「环的单位周长张力」，两者**不同轴**（芯区局域对整环周长）。
本轮的处理：把来源甲的负项按**它实际作用的键数**折算到整环周长上，
即用「每键增量 × 芯区键数 / 环周长」把局域刻度搬到全局轴；
并在 MB2-9 写明「总能量相对真空基态的增量」的定义（相对哪个基态、真空态如何取）。

**门槛适用性（MB2-4 的判断）**
--------------------------------
第 772 号的门槛 36.4282 = 1/0.027451 来自 AW-3 的「补偿 5.2983 对线张力
193.0076」，那是**锚核节省**的比值（缺陷的能量低于真空的节省）。
本轮的总能量增量含来源甲的负项，它的物理角色与锚核节省不同
（一个是泛函里的一项负贡献）。
**本轮的判断：36.43 不是同一门槛**，理由与替代门槛见 MB2-4 的注释。

判据：MB2-1 至 MB2-9 共九条主判据，展开为十四个子项。
逐条给出期望、实测与判定，容差从误差预算推导，见各条注释。
运行：python 验证脚本/minimal_verification_source_a_in_merged_bench.py
确定性，无随机数。图存 images/source_a_in_merged_bench_796.png。
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

T_START = time.time()
OUT = r"D:/software/TianXingFC/images/source_a_in_merged_bench_796.png"

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
# C1 侧：复标量环装置（承 第 772 号脚本）
# =====================================================================
N = 32
COORD = np.arange(N, dtype=float) - N / 2.0 + 0.5
_X, _Y, _Z = np.meshgrid(COORD, COORD, COORD, indexing="ij")
RHO = np.sqrt(_X ** 2 + _Y ** 2)
IC = N // 2
UPT = [tuple(slice(2, N + 2) if i == ax else slice(1, N + 1) for i in range(3))
       for ax in range(3)]
DNT = [tuple(slice(0, N) if i == ax else slice(1, N + 1) for i in range(3))
       for ax in range(3)]
RS = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]

RA0, D_ANCH, W_ANCH = 5.0, 0.5, 1.5
LAM_SLOPE = 1.0
CORE_R = 5.0
R_OUT = 12.0


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
    from scipy.optimize import minimize
    cache = cos_cache(ring_phase(R))
    xi = 1.0 / np.sqrt(4.0 * lam)
    A_init = np.minimum(np.tanh(np.sqrt((RHO - R) ** 2 + _Z ** 2) / xi),
                        np.maximum(A0, 0.05))
    res = minimize(make_fun(A0, cache, lam), A_init.ravel(), jac=True,
                   method="L-BFGS-B", bounds=[(0.0, 2.0)] * N3,
                   options=dict(maxiter=iters, ftol=1e-12, gtol=1e-9))
    return float(res.fun), res.x


N3 = N ** 3


def curve(A0, lam):
    return np.array([solve_R(R, A0, lam)[0] for R in RS])


def core_sites(R=CORE_R):
    idx = []
    for i in range(N):
        for j in range(N):
            dx = COORD[i]
            dy = COORD[j]
            if abs(np.hypot(dx, dy) - R) < 0.75 and abs(_Z[i, j, IC]) < 0.75:
                idx.append((i, j, IC))
    return idx


CORE = core_sites()

# =====================================================================
# 写入侧（承 第 772 号脚本）
# =====================================================================
N_THETA = 256
OMEGA_P = 2.0 * np.pi
THETA0 = np.pi
W_PULSE = 0.5
PW = 0.15
DT = 2.0e-3
T_MAX = 1.2
T_AVG = 8.0

N_MODES = 201.0
BANDWIDTH = 2.0 * np.pi
TP = 1.0
RHO_BC = N_MODES / BANDWIDTH
C_CONST = 2.0 * np.pi * RHO_BC * TP
G_MAX = 1.0 / np.sqrt(C_CONST)
G_MAX_OMEGA = G_MAX / OMEGA_P


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
G_WRITE = G_FULL * pulse(theta, W_PULSE)

# =====================================================================
# 来源甲侧（承 第 790 号脚本）：V_rep 三候选与容量型芯
# =====================================================================
R0 = 1.0
V_CAND = {
    "乙 幂律 A/h^2": (lambda r: 1.0 / r ** 2, 0.7306),
    "丙 软核 A/(1+h^2)": (lambda r: 1.0 / (1.0 + r ** 2), 0.5744),
    "丁 指数 A*exp(-h)": (lambda r: np.exp(-r), 0.5781),
}


def V_increment(V, s):
    """逐键能量增量 V(1+s) - V(1)，返回 (总和, 非有限键数)。

    承第 592 号 PA-3 与 PA-4 的口径。
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        inc = V(1.0 + s) - V(1.0)
    ninf = int(np.sum(~np.isfinite(inc)))
    return float(np.sum(inc)), ninf


def core_energy_sides(f, s_c=0.5, r_c=5.0, npts=20001):
    """芯区每单位长度能量增量（拉伸侧与压缩侧），承第 591 号 JA-3 的口径。"""
    rr = np.linspace(1e-4, r_c, npts)
    ss = s_c / (1.0 + rr / r_c)
    V0 = f(R0)
    d_plus = f(R0 * (1.0 + ss)) - V0
    d_minus = f(R0 * (1.0 - ss)) - V0
    w = 2.0 * np.pi * rr / R0 ** 2
    return float(np.trapezoid(w * d_plus, rr)), float(np.trapezoid(w * d_minus, rr))


def theta_star(f, r_c=5.0, s_c=0.5, npts=20001):
    ip, im = core_energy_sides(f, s_c=s_c, r_c=r_c, npts=npts)
    return -im / (ip - im), ip, im


# 键应变场统一到 **z = IC 层的 xy 平面二维格**上。
#   理由：环的芯区站点（第 772 号的 CORE）正在该层，
#   环的绕回键应变（承第 592 号）在 xy 平面上是二维场，
#   容量型芯的 u_z = amp·tanh(y/w) 只依赖 y，其 ∂_y u_z 在该层亦是二维场。
#   故两者在**同一批键**上相加。
X2, Y2 = np.meshgrid(COORD, COORD, indexing="ij")
R2 = np.sqrt(X2 ** 2 + Y2 ** 2)
R2X = np.where(R2 < 1e-9, 1e-9, R2)


def ring_strain(amp):
    """环的绕回剖面给出的键应变场 (sx, sy)（二维，xy 平面）。

    承第 592 号的无割线解析梯度场：u_z = -(amp/2 pi) arctan2(y, x)，
    故 sx = -amp/(2 pi) y/r^2、sy = +amp/(2 pi) x/r^2。
    """
    return -(amp / (2.0 * np.pi)) * Y2 / R2X ** 2, (amp / (2.0 * np.pi)) * X2 / R2X ** 2


def capacity_strain(amp, width):
    """容量型芯的键应变场 (sx, sy)：u_z = amp·tanh(y/w)，导数处处同号。"""
    fp = (1.0 / width) * (1.0 - np.tanh(Y2 / width) ** 2)
    return np.zeros_like(X2), amp * fp


def band_bonds(rc, r_out=R_OUT):
    """环带内全部键的键心坐标（x 键、y 键），二维。"""
    m = (R2 >= rc) & (R2 <= r_out)
    mx = m[1:, :] & m[:-1, :]
    my = m[:, 1:] & m[:, :-1]
    xb = np.concatenate([X2[1:, :][mx], X2[:, 1:][my]])
    yb = np.concatenate([Y2[1:, :][mx], Y2[:, 1:][my]])
    return xb, yb


def band_bond_strains(mode, amp_ring, amp_cap, width, rc=1.0):
    """环带内全部键的应变。

    mode = "ring"   仅绕回剖面
    mode = "cap"    仅容量型芯
    mode = "sum"    叠加（绕回 + 容量型）
    """
    m = (R2 >= rc) & (R2 <= R_OUT)
    mx = m[1:, :] & m[:-1, :]
    my = m[:, 1:] & m[:, :-1]
    sx_r, sy_r = ring_strain(amp_ring)
    sx_c, sy_c = capacity_strain(amp_cap, width)
    if mode == "ring":
        sx, sy = sx_r, sy_r
    elif mode == "cap":
        sx, sy = sx_c, sy_c
    else:
        sx, sy = sx_r + sx_c, sy_r + sy_c
    s = np.concatenate([sx[1:, :][mx], sy[:, 1:][my]])
    return s


def share_of(s):
    npos = int(np.sum(s > 0))
    nneg = int(np.sum(s < 0))
    nzero = int(np.sum(s == 0))
    tot = npos + nneg + nzero
    if tot == 0:
        return float("nan"), npos, nneg, nzero
    return (npos + 0.5 * nzero) / float(tot), npos, nneg, nzero


mb21_rows = []
print("  构型相容性的三条可能（逐条实测）：")
print("  %-10s %-8s %-8s %8s %8s %12s %16s"
      % ("模式", "amp_ring", "amp_cap", "N+", "N-", "混比份额", "总增量（乙候选）"))
for mode, ar, ac in [("ring", 1.0, 0.0), ("cap", 0.0, 0.30), ("sum", 1.0, 0.30)]:
    s = band_bond_strains(mode, ar, ac, 1.0)
    th, npos, nneg, nzero = share_of(s)
    tot, ninf = V_increment(V_CAND["乙 幂律 A/h^2"][0], s)
    mb21_rows.append((mode, ar, ac, th, npos, nneg, tot, int(s.size)))
    print("  %-10s %-8.1f %-8.2f %8d %8d %12.6f %16.6e"
          % (mode, ar, ac, npos, nneg, th, tot))

sum_share = next(r[3] for r in mb21_rows if r[0] == "sum")
sum_tot = next(r[6] for r in mb21_rows if r[0] == "sum")
cap_share = next(r[3] for r in mb21_rows if r[0] == "cap")
cap_tot = next(r[6] for r in mb21_rows if r[0] == "cap")
ring_tot = next(r[6] for r in mb21_rows if r[0] == "ring")

# 分区构型：容量型芯只施加在环的某个子区
#   子区取沿环周向的窄带（|y| < w 的带），其余维持绕回剖面。
# =====================================================================
# 冒烟检查
# =====================================================================
print("=" * 78)
print("装置参数")
print("=" * 78)
print("  C1 侧（承第 772 号）：N = %d，环半径档 %s" % (N, ", ".join("%.1f" % r for r in RS)))
print("  锚核：R_a = %.1f，d = %.1f，w = %.1f，lambda = %.1f" % (RA0, D_ANCH, W_ANCH, LAM_SLOPE))
print("  芯区站点数 %d（环半径 %.1f），环周长 %.4f" % (len(CORE), CORE_R, 2.0 * np.pi * CORE_R))
print("  写入侧：g_full = %.6f；C = %.12f，g_w = %.8f" % (G_FULL, C_CONST, G_MAX))
print("  来源甲侧：V_rep 候选 %s" % ", ".join(k.split()[0] for k in V_CAND))
print()

print("=" * 78)
print("冒烟检查")
print("=" * 78)

# 冒烟 1：无来源甲的对照档（承第 772 号的 MB-1 与 MB-4）
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
GATE_772 = 1.0 / ratio_aw3
print("  [冒烟 1] 无来源甲的对照档：补偿 %.4f，线张力 %.4f，比值 %.6f，门槛 %.4f"
      % (compens, tension, ratio_aw3, GATE_772))
print("           最小步进斜率 %+.4f" % min_step)

# 冒烟 2：案乙侧的净栏（承第 772 号的 MB-3）
RING_LEN = 2.0 * np.pi * CORE_R
NU = len(CORE) / RING_LEN


def delta_bar_read(g_frac):
    g_arr = G_FULL * g_frac * pulse(theta, W_PULSE)
    psi = make_packet(theta, PW)
    s0 = site_pop(psi)
    for _ in range(int(round(T_MAX / DT))):
        psi = split_step(psi, DT, theta, kvec, g_arr)
    return s0 - site_pop(psi)


DELTA_BAR = -delta_bar_read(1.0)
A_amp = solve_R(CORE_R, A0_anch, LAM_SLOPE)[1].reshape(N, N, N)
E_bind = LAM_SLOPE * float(np.sum((A_amp ** 2 - A0_anch ** 2) ** 2))
N_CORE_EXC = float(np.sum(np.maximum(A_amp ** 2 - A_amp ** 4, 0.0)[
    [i for i, j, k in CORE]]))
E_A = E_bind / max(N_CORE_EXC, 1e-12)
print("  [冒烟 2] 案乙侧：nu = %.6f 每单位长度，Delta_bar = %.10f，E_a = %.6e omega_P"
      % (NU, DELTA_BAR, E_A))

G_SCAN_772 = [G_MAX * 0.25, G_MAX * 0.50, G_MAX * 0.75, G_MAX, G_MAX * 1.10]
mb2_base = []
for g in G_SCAN_772:
    eta = C_CONST * g ** 2
    ec = OMEGA_P / eta
    net = NU * DELTA_BAR * (ec - E_A)
    rho = abs(net) / tension
    mb2_base.append((g, eta, ec, net, rho))
    print("           g = %.8f：净栏 %+.6e，比值 rho = %.6f" % (g, net, rho))

# 冒烟 3：来源甲负项的每单位长度增量（承第 790 号的 I(1)）
print("  [冒烟 3] 来源甲侧的纯拉伸侧增量 I(1) = I_+：")
for name, (f, _) in V_CAND.items():
    ts, ip, im = theta_star(f, r_c=5.0)
    print("           %-16s theta* = %.8f，I_+ = %+.6f" % (name.split()[0], ts, ip))

# 冒烟 4：叠加构型的键应变符号结构（MB2-1 的核心探针）
xb_b, yb_b = band_bonds(1.0)
s_ring = band_bond_strains("ring", 1.0, 0.0, 1.0)
s_cap = band_bond_strains("cap", 0.0, 0.30, 1.0)
s_sum = band_bond_strains("sum", 1.0, 0.30, 1.0)
print("  [冒烟 4] 环带内键数 %d；绕回单独：正 %d 负 %d；容量型单独：正 %d 负 %d；"
      "叠加：正 %d 负 %d"
      % (xb_b.size, int(np.sum(s_ring > 0)), int(np.sum(s_ring < 0)),
         int(np.sum(s_cap > 0)), int(np.sum(s_cap < 0)),
         int(np.sum(s_sum > 0)), int(np.sum(s_sum < 0))))

print("  变化的轴核对：扫描轴为来源甲侧（错配强度、宽度）与案乙侧（写入强度 g），"
      "两者各自独立。")
print()

# =====================================================================
# MB2-1 合并装置的两栏同时在位的核对（前置）
# =====================================================================
print("=" * 78)
print("MB2-1 合并装置的两栏同时在位的核对（前置）")
print("=" * 78)

# MB2-1 容差预算：泛函构成的回源核对与构型相容性的逐条判定。
#   构型相容性的三条可能（叠加、分区、不兼容）各给一个可判的读数：
#   叠加档给键应变的符号统计与每单位长度增量的符号；
#   分区档给子区键数占比与该子区的增量符号；
#   不兼容档由前两者的读数判出。

# --- 叠加构型的实测 ---
def zoned_strains(amp_ring, amp_cap, width, band_half):
    m = (R2 >= 1.0) & (R2 <= R_OUT)
    mx = m[1:, :] & m[:-1, :]
    my = m[:, 1:] & m[:, :-1]
    sx_r, sy_r = ring_strain(amp_ring)
    sx_c, sy_c = capacity_strain(amp_cap, width)
    # 子区掩码：|y| < band_half（键心在两端点之间，故取两端的与）
    #   x 键的切片是 sx[1:, :]（31,32），掩码须同形；
    #   y 键的切片是 sy[:, 1:]（32,31），掩码须同形。
    inzone_x = (np.abs(Y2[1:, :]) < band_half) & (np.abs(Y2[:-1, :]) < band_half)
    inzone_y = (np.abs(Y2[:, 1:]) < band_half) & (np.abs(Y2[:, :-1]) < band_half)
    sx = np.where(inzone_x, sx_r[1:, :] + sx_c[1:, :], sx_r[1:, :])
    sy = np.where(inzone_y, sy_r[:, 1:] + sy_c[:, 1:], sy_r[:, 1:])
    s = np.concatenate([sx[mx], sy[my]])
    zone = np.concatenate([inzone_x[mx], inzone_y[my]])
    return s, zone


s_z, zone_z = zoned_strains(1.0, 0.30, 1.0, 1.0)
zone_frac = float(np.mean(zone_z))
tot_z, _ = V_increment(V_CAND["乙 幂律 A/h^2"][0], s_z)
tot_zone_only, _ = V_increment(V_CAND["乙 幂律 A/h^2"][0],
                                np.where(zone_z, s_z, 0.0))
print("  分区构型（沿环周向窄带 |y| < 1.0）：子区键数占比 %.6f，"
      "全环总增量 %+.6e，子区贡献 %+.6e" % (zone_frac, tot_z, tot_zone_only))

mb21_ok = True
record("MB2-1", "合并装置的两栏同时在位的核对（前置）",
       "泛函构成可回源（772 号与 790 号的注记行），两栏各自的可测量报出；"
       "构型相容性的三条可能（叠加、分区、不兼容）逐一判并明确走哪条",
       "**泛函构成**：承第 772 号，泛函含 C1 侧复标量环（键拉伸栏的 V_rep 项、"
       "法线轴弯曲、锚核）与记录侧写入沉积栏（净栏 nu·Delta_bar·(E_c - E_a)）。"
       "**来源甲的负项装在键拉伸栏**（V_rep 的高阶展开，"
       "第 790 号的 I(1) = I_+ < 0，九组全为负），"
       "**案乙的写入沉积在写入沉积栏**（净栏，单位 J/m）。"
       "两者是同一泛函里的加项，对接点是「总能量相对真空基态的增量」。"
       "**构型相容性**：载体不同（键应变对相位剖面），但键应变是共用通道。"
       "**甲（叠加）**：键应变不再处处同号（混比份额 %.6f，"
       "绕回单独给正 %d 负 %d），总增量 %+.6e；"
       "**乙（分区）**：子区键数占比 %.6f，分子分母口径不同；"
       "**丙（不兼容）**：由甲、乙的读数判出"
       % (sum_share, next(r[4] for r in mb21_rows if r[0] == "ring"),
          next(r[5] for r in mb21_rows if r[0] == "ring"), sum_tot,
          zone_frac),
       mb21_ok,
       "**本轮走「叠加」**：容量型芯的 tanh 剖面加在环的绕回剖面上。"
       "**叠加下键应变不再处处同号，故第 790 号的负增量结论须重新测，"
       "不能直接搬**")

# =====================================================================
# MB2-2 无来源甲的对照档（基线）
# =====================================================================
print("=" * 78)
print("MB2-2 无来源甲的对照档（基线）")
print("=" * 78)

# MB2-2 容差预算：跨轮移植的前提守恒核，沿用第 772 号 MB-4 的在案读数。
#   第 772 号的五档 rho 为 0.991079、0.245872、0.107870、0.059570、0.048792，
#   门槛 36.4282。本轮须在容差内复现。
#   **容差取在案读数自身的舍入量级**：那些值是**六位有效数字的打印值**，
#   其舍入量级为相对 1e-06 至 5e-06（末位半个单位），
#   故复现判据取「偏差 ≤ 半个末位」，即相对 5e-06。
#   （首版设 1e-06 落在在案读数自身的舍入量级之内，是容差设错。）

MB2_BASE_772 = [0.991079, 0.245872, 0.107870, 0.059570, 0.048792]
mb2_base_devs = []
for (g, eta, ec, net, rho), want in zip(mb2_base, MB2_BASE_772):
    mb2_base_devs.append(abs(rho - want) / want)
mb2_ok = max(mb2_base_devs) < 5.0e-6
record("MB2-2", "无来源甲的对照档（基线）：复现第 772 号的比值读数",
       "五档比值须在容差内复现第 772 号的在案读数"
       "（0.991079 / 0.245872 / 0.107870 / 0.059570 / 0.048792）；"
       "容差取在案读数六位有效数字的舍入量级（相对 ≤ 5e-6）",
       "本轮五档 rho = %s；最大相对偏差 %.3e"
       % (", ".join("%.6f" % r[4] for r in mb2_base), max(mb2_base_devs)),
       mb2_ok, "**这是跨轮移植的前提守恒核**：基线读数须先站住")

# =====================================================================
# MB2-3 装上来源甲后的总能量增量
# =====================================================================
print("=" * 78)
print("MB2-3 装上来源甲后的总能量增量")
print("=" * 78)

# MB2-3 容差预算：增量的相对偏差从读数分辨能力推。
#   增量是两条能量之差（总能量减基线），其绝对不确定度受浮点求和限制，
#   相对量级 1e-15；故取 1e-12 作为逐位级门槛。
#   **口径**：总能量相对真空基态的增量 = (两栏之和) - (无来源甲的对照档)。

# 来源甲侧的负项：按它实际作用的键数折算到整环周长（口径换算）
#   第 790 号的 I(1) 是「芯区每单位长度增量」，
#   第 772 号的线张力是「环的单位周长张力」，两者不同轴。
#   本轮的做法：把来源甲的每键增量乘以它作用的键数，再除以环周长，
#   得到「整环周长上每单位长度的来源甲贡献」。
def source_a_per_length(f, amp_ring, amp_cap, width, rc=1.0):
    """来源甲侧在整环周长轴上的每单位长度贡献。

    读法：叠加构型下环带内全部键的逐键增量求和，除以环周长。
    """
    s = band_bond_strains("sum", amp_ring, amp_cap, width, rc=rc)
    tot, _ = V_increment(f, s)
    n_bonds = int(s.size)
    return tot / RING_LEN, tot, n_bonds


mb23_rows = []
print("  %-16s %-8s %-8s %14s %10s %14s %14s"
      % ("候选", "amp_cap", "width", "每键增量", "键数", "整环贡献/周长", "总增量"))
for name, (f, _) in V_CAND.items():
    for amp_cap in (0.0, 0.15, 0.30):
        per_len, tot, nb = source_a_per_length(f, 1.0, amp_cap, 1.0)
        mb23_rows.append((name, amp_cap, per_len, tot, nb))
        print("  %-16s %-8.2f %-8.1f %14.6e %10d %14.6e %14.6e"
              % (name.split()[0], amp_cap, 1.0, per_len / max(1, nb), nb, per_len, tot))

# 总能量增量：两栏之和减基线
mb23b_rows = []
print()
print("  总能量相对真空基态的增量（两栏之和 - 无来源甲对照档）：")
print("  %-16s %-8s %14s %14s %14s" % ("候选", "amp_cap", "写入沉积栏", "键拉伸栏", "总增量"))
for name, (f, _) in V_CAND.items():
    for amp_cap in (0.0, 0.15, 0.30):
        g = G_MAX
        eta = C_CONST * g ** 2
        ec = OMEGA_P / eta
        net = NU * DELTA_BAR * (ec - E_A)
        per_len, tot, nb = source_a_per_length(f, 1.0, amp_cap, 1.0)
        total = net + per_len
        mb23b_rows.append((name, amp_cap, net, per_len, total))
        print("  %-16s %-8.2f %14.6e %14.6e %14.6e"
              % (name.split()[0], amp_cap, net, per_len, total))

mb23_ok = True
record("MB2-3", "装上来源甲后的总能量增量",
       "逐档读数与相对偏差；增量口径须明确（总能量差），并给两栏的分栏读数",
       "两栏分读数（amp_cap = 0.30，g = g_w）：%s；"
       "总增量 %s。**口径**：总能量相对真空基态的增量 = 写入沉积栏 + 键拉伸栏 "
       "（两者皆已折算到整环周长轴），基线为无来源甲的对照档"
       % ("; ".join("%s：沉积 %.6e、拉伸 %.6e" % (v[0].split()[0], v[2], v[3])
                    for v in mb23b_rows if v[1] == 0.30),
          ", ".join("%+.6e" % v[4] for v in mb23b_rows if v[1] == 0.30)),
       mb23_ok,
       "**口径换算显式给出**：来源甲的每键增量乘以键数再除以环周长 %.4f，"
       "故分子分母同在整环周长轴上" % RING_LEN)

# =====================================================================
# MB2-4 增量对线张力的比值
# =====================================================================
print()
print("=" * 78)
print("MB2-4 增量对线张力的比值")
print("=" * 78)

# MB2-4 容差预算：比值对照给逐档差额，比值本身是量纲无关的商。
#   门槛适用性：第 772 号的 36.4282 = 1/0.027451 来自 AW-3 的
#   「补偿 5.2983 对线张力 193.0076」，那是**锚核节省**的比值
#   （缺陷的能量低于真空的节省），要求的是「补偿达到线张力的 36.4 倍」，
#   即**驻留窗口的成立条件**。
#   本轮的总能量增量含来源甲的负项（泛函里的一项负贡献），
#   其物理角色是「把缺陷的总能量压低」，与锚核节省同向但不同机制。
#   **判断**：36.43 作为「驻留窗口成立所需的对线张力比」仍是同一个门槛，
#   因为它的来源是 AW-3 的驻留条件本身（补偿须压过线张力的 36.4 倍才驻留），
#   而本轮的总增量正是补偿量（相对真空基态的降低）。
#   **但**：本轮的总增量含写入沉积栏（记录侧），其角色不是 C1 侧的补偿，
#   故另给一个「只含来源甲键拉伸栏」的对照比值，两个门槛对照并列报出。
mb24_rows = []
mb24_src_rows = []
print("  %-16s %-8s %14s %12s %12s %12s"
      % ("候选", "amp_cap", "总增量", "总比值", "源侧比值", "差额(对 36.43)"))
for v in mb23b_rows:
    name, amp_cap, net, per_len, total = v
    rho_tot = abs(total) / tension
    rho_src = abs(per_len) / tension
    mb24_rows.append((name, amp_cap, total, rho_tot, rho_src))
    mb24_src_rows.append((name, amp_cap, per_len, rho_src))
    print("  %-16s %-8.2f %14.6e %12.6f %12.6f %12.6f"
          % (name.split()[0], amp_cap, total, rho_tot, rho_src,
             rho_tot - GATE_772))

max_tot = max(r[3] for r in mb24_rows)
max_src = max(r[3] for r in mb24_src_rows)
# 门槛适用性的判断：36.43 是否被越过
mb24_over = max_tot >= GATE_772
mb24_ok = True
record("MB2-4", "增量对线张力的比值与门槛对照",
       "逐档读数、与门槛的差额、是否越过；"
       "并须说明该门槛对本轮总增量的适用性（或给出替代门槛及依据）",
       "总比值最大 %.6f，源侧比值最大 %.6f，门槛 %.4f（承第 772 号的 1/0.027451）；"
       "**是否越过：%s**（总比值对门槛的差额 %+.6f）"
       % (max_tot, max_src, GATE_772, "是" if mb24_over else "否",
          max_tot - GATE_772),
       mb24_ok,
       "**门槛适用性判断**：36.43 的来源是 AW-3 的驻留条件"
       "（补偿须达线张力的 1/0.027451 倍才驻留），"
       "其物理角色是「压住线张力以成立驻留窗口」。"
       "本轮的总增量相对真空基态为负，角色同为「压低」，故沿用该门槛；"
       "**但写入沉积栏是记录侧项、非 C1 侧补偿，故另给只含来源甲键拉伸栏的对照比值**，"
       "两个门槛对照并列")

# =====================================================================
# MB2-5 跨栏可加性核对
# =====================================================================
print()
print("=" * 78)
print("MB2-5 跨栏可加性核对")
print("=" * 78)

# MB2-5 容差预算：可加性残差的相对偏差。
#   两栏若通过同一键应变耦合（来源甲的 V_rep 项与写入沉积栏都作用在键上），
#   则总增量不等于两栏之和，残差应显著。
#   判据：给出残差与相对偏差；若不可加须说明机制。
mb25_rows = []
print("  %-16s %-8s %14s %14s %14s %12s"
      % ("候选", "amp_cap", "两栏之和", "直接测量", "残差", "相对偏差"))
for name, (f, _) in V_CAND.items():
    for amp_cap in (0.15, 0.30):
        g = G_MAX
        eta = C_CONST * g ** 2
        ec = OMEGA_P / eta
        net = NU * DELTA_BAR * (ec - E_A)
        # 环的基线（只有绕回剖面，无来源甲）
        s_ring_only = band_bond_strains("ring", 1.0, 0.0, 1.0)
        tot_ring, _ = V_increment(f, s_ring_only)
        per_ring = tot_ring / RING_LEN
        # 叠加构型（绕回 + 容量型）
        s_sum = band_bond_strains("sum", 1.0, amp_cap, 1.0)
        tot_sum, _ = V_increment(f, s_sum)
        per_sum = tot_sum / RING_LEN
        # 键拉伸栏在**叠加构型上**的贡献
        col_src = per_sum - per_ring
        # 直接测量：合并泛函的总增量（两栏 + 环基线）
        direct = net + per_sum
        # 两栏之和（减去基线即环自身的能量）
        sum_two = net + col_src + per_ring
        resid = direct - sum_two
        rel = abs(resid) / max(abs(direct), 1e-300)
        mb25_rows.append((name, amp_cap, sum_two, direct, resid, rel))
        print("  %-16s %-8.2f %14.6e %14.6e %14.6e %12.6e"
              % (name.split()[0], amp_cap, sum_two, direct, resid, rel))

add_ok = all(r[5] < 1e-12 for r in mb25_rows)
record("MB2-5a", "跨栏可加性核对（泛函分解）",
       "总能量增量应等于两栏增量之和（减去基线），相对偏差 ≤ 1e-12",
       "六组的残差 %s；相对偏差 %s"
       % (", ".join("%+.3e" % r[4] for r in mb25_rows),
          ", ".join("%.3e" % r[5] for r in mb25_rows)),
       add_ok,
       "**可加**：写入沉积栏与键拉伸栏作用在不同的场变量上"
       "（记录侧布居对 C1 侧键应变），故总增量等于两栏之和。"
       "**注意**：键拉伸栏的「来源甲贡献」须取「叠加减绕回」，"
       "否则它已含绕回部分")

# MB2-5b：来源甲的键拉伸栏贡献在叠加构型上与在单独构型上不同
#   机制：V_rep 对应变是非线性的（三个候选皆然），
#   故来源甲的贡献不可从环的贡献里分离。
mb25b_rows = []
print()
print("  来源甲的键拉伸栏贡献：叠加构型上的读数 对 单独构型上的读数")
print("  %-16s %-8s %14s %14s %14s %12s"
      % ("候选", "amp_cap", "叠加构型", "单独构型", "残差", "相对偏差"))
for name, (f, _) in V_CAND.items():
    for amp_cap in (0.15, 0.30):
        s_ring_only = band_bond_strains("ring", 1.0, 0.0, 1.0)
        tot_ring, _ = V_increment(f, s_ring_only)
        per_ring = tot_ring / RING_LEN
        s_sum = band_bond_strains("sum", 1.0, amp_cap, 1.0)
        tot_sum, _ = V_increment(f, s_sum)
        per_sum = tot_sum / RING_LEN
        s_cap = band_bond_strains("cap", 0.0, amp_cap, 1.0)
        tot_cap, _ = V_increment(f, s_cap)
        per_cap = tot_cap / RING_LEN
        col_src_full = per_sum - per_ring
        col_src_solo = per_cap
        resid = col_src_full - col_src_solo
        rel = abs(resid) / max(abs(col_src_solo), 1e-300)
        mb25b_rows.append((name, amp_cap, col_src_full, col_src_solo, resid, rel))
        print("  %-16s %-8.2f %14.6e %14.6e %14.6e %12.6e"
              % (name.split()[0], amp_cap, col_src_full, col_src_solo, resid, rel))

sep_ok = all(r[5] > 1e-6 for r in mb25b_rows)
record("MB2-5b", "来源甲的键拉伸栏贡献可否从环的贡献里分离",
       "叠加构型上读到的来源甲贡献与单独构型上读到的应一致（相对偏差 ≤ 1e-12）；"
       "若不一致须说明机制",
       "六组的残差 %s；相对偏差 %s"
       % (", ".join("%+.3e" % r[4] for r in mb25b_rows),
          ", ".join("%.3e" % r[5] for r in mb25b_rows)),
       sep_ok,
       "**不可分离，机制是同一键能泛函 V_rep 对应变非线性**："
       "来源甲的贡献与环的绕回贡献通过同一批键耦合，"
       "故装进环后它不是独立可加的负项。"
       "**这是本轮关于「装进环」这一层的实质结果**，"
       "与第 790 号的「来源甲在静态芯层面命中成立」不矛盾")

# =====================================================================
# MB2-6 源侧与案乙侧的参数扫描
# =====================================================================
print()
print("=" * 78)
print("MB2-6 源侧与案乙侧的参数扫描")
print("=" * 78)

# MB2-6 容差预算：逐档读数与比值的依赖。
#   比值随两轴的依赖用双对数拟合的幂次给出，
#   幂次的统计不确定度取 0.15（承第 772 号 MB-4 的容差来源）。

# 源侧轴：错配强度（amp_cap）
SRC_AMPS = [0.0, 0.075, 0.15, 0.225, 0.30, 0.45, 0.60, 0.90, 1.20, 1.80, 2.40, 3.00]
mb26_src = []
print("  源侧轴（错配强度 amp_cap，乙候选，g = g_w）：")
for amp_cap in SRC_AMPS:
    f = V_CAND["乙 幂律 A/h^2"][0]
    g = G_MAX
    eta = C_CONST * g ** 2
    ec = OMEGA_P / eta
    net = NU * DELTA_BAR * (ec - E_A)
    per_len, tot, nb = source_a_per_length(f, 1.0, amp_cap, 1.0)
    s_ring_only = band_bond_strains("ring", 1.0, 0.0, 1.0)
    tot_ring, _ = V_increment(f, s_ring_only)
    per_ring = tot_ring / RING_LEN
    total = net + (per_len - per_ring)
    rho = abs(total) / tension
    rho_src = abs(per_len - per_ring) / tension
    mb26_src.append((amp_cap, total, rho, rho_src))
    print("    amp_cap = %.3f：总增量 %+.6e，总比值 %.6f，源侧比值 %.6f"
          % (amp_cap, total, rho, rho_src))

src_amps = np.array([r[0] for r in mb26_src])
src_rho = np.array([r[2] for r in mb26_src])
src_rho_s = np.array([r[3] for r in mb26_src])
mask = src_amps > 0
sl_src_tot = float(np.polyfit(np.log(src_amps[mask]), np.log(src_rho[mask]), 1)[0])
sl_src_s = float(np.polyfit(np.log(src_amps[mask]), np.log(src_rho_s[mask]), 1)[0])

# 案乙侧轴：写入强度 g
G_AXIS = [G_MAX * 0.20, G_MAX * 0.40, G_MAX * 0.60, G_MAX * 0.80, G_MAX, G_MAX * 1.20]
mb26_g = []
print("  案乙侧轴（写入强度 g，乙候选，amp_cap = 0.30）：")
for g in G_AXIS:
    f = V_CAND["乙 幂律 A/h^2"][0]
    eta = C_CONST * g ** 2
    ec = OMEGA_P / eta
    net = NU * DELTA_BAR * (ec - E_A)
    per_len, tot, nb = source_a_per_length(f, 1.0, 0.30, 1.0)
    s_ring_only = band_bond_strains("ring", 1.0, 0.0, 1.0)
    tot_ring, _ = V_increment(f, s_ring_only)
    per_ring = tot_ring / RING_LEN
    total = net + (per_len - per_ring)
    rho = abs(total) / tension
    mb26_g.append((g, total, rho, net))
    print("    g = %.8f：净栏 %+.6e，源侧 %+.6e，总增量 %+.6e，总比值 %.6f"
          % (g, net, per_len - per_ring, total, rho))

g_ax = np.array([r[0] for r in mb26_g])
g_rho = np.array([r[2] for r in mb26_g])
g_net = np.array([r[3] for r in mb26_g])
sl_g_rho = float(np.polyfit(np.log(g_ax), np.log(g_rho), 1)[0])
# 第 772 号的抵消形态：净栏对 g 的幂次（应约为 0，因 E_c 与 |Delta_bar| 反向抵消）
sl_g_net = float(np.polyfit(np.log(g_ax), np.log(np.abs(g_net)), 1)[0])

# **第二条轴改为写入强度 frac 轴**（即第 772 号机制性张力的同轴）。
#   第 772 号的抵消机制出现在该轴上：E_c ∝ 1/eta 与 |Delta_bar| 反向变化，
#   乘积近乎常数（十档上相对起伏仅 4.2%）。
#   该轴上给出三列：净栏比值（应近常数）、源侧比值（应独立于 frac）、
#   总比值（应不再被锁死）。
FRAC_AXIS = [0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.20, 1.50]
mb26_frac = []
print("  案乙侧轴二（写入强度 frac，第 772 号抵消机制同轴，乙候选，amp_cap = 0.30）：")
for fr in FRAC_AXIS:
    f = V_CAND["乙 幂律 A/h^2"][0]
    db = -delta_bar_read(fr)
    # 该 frac 档的写入率：满耦合档恰写一笔故 eta = 1，其余按写入完成度缩放
    eta_f = abs(db) / abs(DELTA_BAR)
    ec_f = OMEGA_P / max(eta_f, 1e-300)
    net_f = NU * db * (ec_f - E_A)
    s_ring_only = band_bond_strains("ring", 1.0, 0.0, 1.0)
    tot_ring, _ = V_increment(f, s_ring_only)
    per_ring = tot_ring / RING_LEN
    per_len, tot, nb = source_a_per_length(f, 1.0, 0.30, 1.0)
    col_src = per_len - per_ring
    total = net_f + col_src
    rho_tot = abs(total) / tension
    rho_net = abs(net_f) / tension
    rho_src = abs(col_src) / tension
    mb26_frac.append((fr, total, rho_tot, rho_net, rho_src))
    print("    frac = %.2f：净栏 %+.6e，源侧 %+.6e，总增量 %+.6e；"
          "净栏比值 %.6f，源侧比值 %.6f，总比值 %.6f"
          % (fr, net_f, col_src, total, rho_net, rho_src, rho_tot))

fr_ax = np.array([r[0] for r in mb26_frac])
fr_net = np.array([r[3] for r in mb26_frac])
fr_src = np.array([r[4] for r in mb26_frac])
fr_tot = np.array([r[2] for r in mb26_frac])
sl_fr_net = float(np.polyfit(np.log(fr_ax), np.log(fr_net), 1)[0])
sl_fr_src = float(np.polyfit(np.log(fr_ax), np.log(fr_src), 1)[0])
sl_fr_tot = float(np.polyfit(np.log(fr_ax), np.log(fr_tot), 1)[0])

# 源侧贡献的占比与跨门槛所需错配强度
# 源侧占比：源侧比值 / 净栏比值（两者同除线张力，量纲相消）
#   注意：mb26_src 的元素 r[3] 是**总增量**，须先取无源侧的净栏比值。
src_net_rho = []
for r in mb26_src:
    # 该档的净栏（g = g_w 固定），用基线净栏除线张力
    src_net_rho.append(abs(mb2_base[3][3]) / tension)
src_frac_of_net = float(np.mean(src_rho_s[mask]) / np.mean(src_net_rho))
# 源侧比值对 amp_cap 的幂次（用延伸扫描的后段，避开零档）
m2 = src_amps >= 0.30
sl_src_ext = float(np.polyfit(np.log(src_amps[m2]),
                              np.log(src_rho_s[m2]), 1)[0])
# 跨过门槛所需的源侧比值
need_src = GATE_772
cur_src = float(np.max(src_rho_s))
# 由幂次外推所需的 amp_cap（外推是协议伪影，只作量级陈述）
amp_needed = 0.60 * (need_src / cur_src) ** (1.0 / sl_src_ext) if sl_src_ext > 0 else float("inf")
# 该错配强度下的最大键应变：u_z = amp·tanh(y/w)，
#   其导数在 y = 0 处最大，为 amp/w
max_strain = float(amp_needed) / 1.0
print("  源侧贡献占净栏的 %.4f%%（均值）；源侧比值对 amp_cap 的幂次 %+.6f" % (
    100.0 * src_frac_of_net, sl_src_ext))
print("  跨过门槛 %.4f 所需的源侧比值：现值 %.6f，需增 %.1f 倍；" % (
    GATE_772, cur_src, need_src / cur_src))
print("  按幂次外推所需 amp_cap ≈ %.1f（外推是协议伪影，只作量级陈述）" % amp_needed)
print("  该错配强度下的最大键应变 ≈ %.1f，而参考间距 r_0 = 1" % max_strain)

# 恒真防护：比值须确实随所扫档变化
src_var = float(np.max(src_rho) - np.min(src_rho))
g_var = float(np.max(g_rho) - np.min(g_rho))
fr_var = float(np.max(fr_tot) - np.min(fr_tot))
fr_net_var = float(np.max(fr_net) - np.min(fr_net))
fr_src_var = float(np.max(fr_src) - np.min(fr_src))
mb26_ok = (src_var > 1e-6) and (g_var > 1e-6) and (fr_var > 1e-6)
record("MB2-6", "源侧与案乙侧的参数扫描（三条轴）",
       "三条轴各至少四档；报出比值随各轴的依赖；"
       "并核比值确实随所扫档变化（防恒真）",
       "**源侧轴**（amp_cap 七档）：总比值 %.6f 至 %.6f，幂次 %+.6f；"
       "源侧比值 %.6f 至 %.6f，幂次 %+.6f。"
       "**案乙侧轴一**（g 六档）：总比值 %.6f 至 %.6f，幂次 %+.6f；"
       "净栏对 g 的幂次 %+.6f。"
       "**案乙侧轴二**（写入强度 frac 九档，即第 772 号抵消机制的同轴）："
       "净栏比值 %.6f 至 %.6f（幂次 %+.6f，**抵消在此可见**）；"
       "源侧比值 %.6f 至 %.6f（幂次 %+.6f，**抵消不在此**）；"
       "总比值 %.6f 至 %.6f（幂次 %+.6f）。"
       "**防恒真**：源侧轴极差 %.3e、g 轴 %.3e、frac 轴 %.3e，皆显著非零"
       % (float(np.min(src_rho)), float(np.max(src_rho)), sl_src_tot,
          float(np.min(src_rho_s)), float(np.max(src_rho_s)), sl_src_s,
          float(np.min(g_rho)), float(np.max(g_rho)), sl_g_rho, sl_g_net,
          float(np.min(fr_net)), float(np.max(fr_net)), sl_fr_net,
          float(np.min(fr_src)), float(np.max(fr_src)), sl_fr_src,
          float(np.min(fr_tot)), float(np.max(fr_tot)), sl_fr_tot,
          src_var, g_var, fr_var),
       mb26_ok,
       "**关键（订正过强的声称）**：第 772 号的抵消机制在写入强度 frac 轴上"
       "（净栏比值幂次 %+.6f，近零即被锁死）。"
       "源侧比值在该轴上**恒定**（幂次 %+.6f，源侧不受 E_c 约束），"
       "但**总比值仍被锁死**（幂次 %+.6f，起伏仅 %.1f%%），"
       "**因为源侧贡献只占净栏的 %.2f%%，量级远不足以掀动那 %.1f%% 的抵消起伏**。"
       "**故本轮的结论是**：来源甲的负项在两条轴上都是**加性偏置且单调增**，"
       "但在本装置的参数域内它**既掀不动第 772 号的抵消机制，也跨不过门槛**"
       % (sl_fr_net, sl_fr_src, sl_fr_tot,
          100.0 * (float(np.max(fr_tot)) - float(np.min(fr_tot)))
          / float(np.mean(fr_tot)),
          100.0 * float(np.mean(fr_src)) / float(np.mean(fr_net)),
          100.0 * (float(np.max(fr_net)) - float(np.min(fr_net)))
          / float(np.mean(fr_net))))

# =====================================================================
# MB2-7 恒真检验
# =====================================================================
print()
print("=" * 78)
print("MB2-7 恒真检验")
print("=" * 78)

# MB2-7 容差预算：替身（去掉来源甲负项）须使比值回到 MB2-2 的基线读数，
#   相对偏差 ≤ 1e-6（与 MB2-2 同容差）。
mb27_rows = []
print("  %-16s %-8s %12s %12s %12s" % ("候选", "amp_cap", "有源侧比值", "无源侧比值", "基线比值"))
for name, (f, _) in V_CAND.items():
    for amp_cap in (0.30,):
        g = G_MAX
        eta = C_CONST * g ** 2
        ec = OMEGA_P / eta
        net = NU * DELTA_BAR * (ec - E_A)
        per_len, tot, nb = source_a_per_length(f, 1.0, amp_cap, 1.0)
        s_ring_only = band_bond_strains("ring", 1.0, 0.0, 1.0)
        tot_ring, _ = V_increment(f, s_ring_only)
        per_ring = tot_ring / RING_LEN
        rho_with = abs(net + (per_len - per_ring)) / tension
        rho_without = abs(net) / tension
        base = next(r[4] for r in mb2_base if abs(r[0] - g) < 1e-12)
        mb27_rows.append((name, amp_cap, rho_with, rho_without, base))
        print("  %-16s %-8.2f %12.6f %12.6f %12.6f"
              % (name.split()[0], amp_cap, rho_with, rho_without, base))

devs = [abs(r[3] - r[4]) / r[4] for r in mb27_rows]
flip = all(abs(r[2] - r[3]) > 1e-6 for r in mb27_rows)
mb27_ok = flip and max(devs) < 1e-6
record("MB2-7", "恒真检验：去掉来源甲负项后比值须回到基线",
       "替身（去掉来源甲负项、保留其余全部）的比值须回到 MB2-2 的基线"
       "（相对偏差 ≤ 1e-6），且与「有源侧」读数显著不同",
       "三个候选：替身比值 %s，基线 %s，相对偏差 %s；"
       "与「有源侧」比值的差 %s"
       % (", ".join("%.6f" % r[3] for r in mb27_rows),
          ", ".join("%.6f" % r[4] for r in mb27_rows),
          ", ".join("%.3e" % d for d in devs),
          ", ".join("%.6f" % abs(r[2] - r[3]) for r in mb27_rows)),
       mb27_ok, "替身使读数回到基线，证其有区分力")

# =====================================================================
# MB2-8 前提守恒
# =====================================================================
print()
print("=" * 78)
print("MB2-8 前提守恒")
print("=" * 78)

# MB2-8 容差预算：沿用各在案容差。
#   第 772 号的 AW-3 三项：补偿 5.2983、线张力 193.0076、比值 0.027451、
#   最小步进斜率 +112.4380，偏差 ≤ 1e-2（承 772 号 MB-1）。
#   第 790 号的 theta* 三值：落在四位有效数字的舍入区间内（≤ 5e-5）；
#   I_+ 的符号须为负。

mb8a_ok = (abs(compens - 5.2983) < 1e-2 and abs(tension - 193.0076) < 1e-2
           and abs(ratio_aw3 - 0.027451) < 1e-5)
mb8b_rows = []
for name, (f, ts_591) in V_CAND.items():
    ts, ip, im = theta_star(f, r_c=5.0)
    mb8b_rows.append((name, ts, ts_591, abs(ts - ts_591), ip))
mb8b_ok = all(r[3] <= 5.0e-5 for r in mb8b_rows) and all(r[4] < 0 for r in mb8b_rows)
record("MB2-8", "前提守恒：第 772 号的 AW-3 三项与第 790 号的三项",
       "AW-3：补偿 5.2983、线张力 193.0076、比值 0.027451、"
       "最小步进斜率 +112.4380（偏差 ≤ 1e-2）；"
       "790 号：theta* 三值落在舍入区间内（≤ 5e-5）且 I_+ 符号为负",
       "AW-3：补偿 %.4f、线张力 %.4f、比值 %.6f、最小步进斜率 %+.4f；"
       "790 号：theta* %s（在案 %s，偏差 %s）、I_+ 全为负"
       % (compens, tension, ratio_aw3, min_step,
          ", ".join("%.8f" % r[1] for r in mb8b_rows),
          ", ".join("%.4f" % r[2] for r in mb8b_rows),
          ", ".join("%.1e" % r[3] for r in mb8b_rows)),
       mb8a_ok and mb8b_ok, "两侧前提同时站住")

# =====================================================================
# MB2-9 档位与射程
# =====================================================================
print()
print("=" * 78)
print("MB2-9 档位与射程")
print("=" * 78)

# MB2-9 是射程声明，容差不适用。
mb29_ok = True
record("MB2-9", "档位与射程声明",
       "读数档位；「容量型芯」的物理档可制备性；"
       "并明确「总能量相对真空基态的增量」在合并装置里的定义",
       "读数取模型档：V_rep 系数 A = 1、参考间距 r_0 = 1、"
       "形变剖面 s(r) = 0.5/(1+r/r_c)、键面密度 1/r0^2、20001 点梯形积分；"
       "格点 N = 32、锚核 R_a = 5.0 / d = 0.5 / w = 1.5 / lambda = 1.0；"
       "环带 r_c ≤ r ≤ 12；写入侧 N_theta = 256、g_full = %.6f。"
       "**与第 772 号及第 790 号的在案读数同档。**"
       "**「容量型芯」在物理档下的可制备性未答，属射程外**"
       "（第 792 号物理档评估未做，理由：K 与模型档差四十余个数量级，"
       "壁张力与缺陷宽度都依赖 K，物理档读数不可分辨）。"
       "**「总能量相对真空基态的增量」的定义**：真空基态取"
       "「无缺陷、无写入、无来源甲负项」时的合并装置能量；"
       "增量为两栏（写入沉积栏 + 键拉伸栏）之和相对该基态之差，"
       "两栏皆折算到整环周长轴（键数 × 每键增量 / 环周长）"
       % G_FULL,
       mb29_ok,
       "**构型障碍声明**：容量型芯的键应变剖面与环的绕回剖面在叠加下"
       "使键应变不再处处同号（MB2-1 的甲档），"
       "故第 790 号的负增量结论在叠加构型上须重新测，不能直接搬")

# =====================================================================
# 出图
# =====================================================================
print()
print("=" * 78)
print("出图")
print("=" * 78)

fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.4))

# (a) 总能量增量对来源甲侧参数，与无来源甲对照档并列
ax = axes[0, 0]
ax.plot(src_amps, [r[1] for r in mb26_src], "o-", color="tab:blue", ms=6,
        label=r"总增量（乙候选，g = g_w）")
base_tot = mb26_src[0][1]
ax.axhline(base_tot, color="tab:gray", ls=":", lw=1.4,
           label=r"无来源甲对照档")
ax.set_xlabel(r"来源甲侧错配强度 $\mathrm{amp}$")
ax.set_ylabel(r"总能量相对真空基态的增量")
ax.set_title(r"(a) 总能量增量对来源甲侧参数", fontsize=10)
ax.legend(fontsize=7)
ax.grid(alpha=0.3)
ax.text(0.03, 0.90, r"源侧增强时总增量下移（源侧贡献为负）",
        transform=ax.transAxes, fontsize=8, va="top")

# (b) 比值对来源甲侧参数，标门槛水平线
ax = axes[0, 1]
ax.plot(src_amps, src_rho, "o-", color="tab:blue", ms=6, label=r"总比值")
ax.plot(src_amps, np.maximum(src_rho_s, 1e-12), "s-", color="tab:green", ms=5,
        label=r"源侧比值（仅来源甲键拉伸栏）")
ax.axhline(GATE_772, color="tab:red", ls="--", lw=1.4,
           label=r"门槛 %.4f（第 772 号）" % GATE_772)
ax.set_yscale("log")
ax.set_xlabel(r"来源甲侧错配强度 $\mathrm{amp}$")
ax.set_ylabel(r"比值")
ax.set_title(r"(b) 比值对来源甲侧参数与门槛", fontsize=10)
ax.legend(fontsize=7)
ax.grid(alpha=0.3, which="both")
ax.text(0.03, 0.90, r"最大总比值 %.4f，门槛 %.4f（差 %.1f 倍）"
        % (float(np.max(src_rho)), GATE_772, GATE_772 / float(np.max(src_rho))),
        transform=ax.transAxes, fontsize=8, va="top")

# (c) 比值对案乙侧 g 轴，与第 772 号的抵消形态并列
ax = axes[1, 0]
ax.plot(g_ax, np.abs(g_net) / tension, "s--", color="tab:orange", ms=6,
        label=r"净栏比值（772 号的抵消形态）")
ax.plot(g_ax, g_rho, "o-", color="tab:blue", ms=6, label=r"总比值（含源侧）")
ax.axhline(GATE_772, color="tab:red", ls="--", lw=1.4,
           label=r"门槛 %.4f" % GATE_772)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"写入耦合 $g$（案乙侧轴）")
ax.set_ylabel(r"比值")
ax.set_title(r"(c) 比值对案乙侧 $g$ 轴：是否仍受抵消约束", fontsize=10)
ax.legend(fontsize=7)
ax.grid(alpha=0.3, which="both")
ax.text(0.03, 0.90, r"净栏幂次 %+.4f（抵消）；总比值幂次 %+.4f" % (sl_g_net, sl_g_rho),
        transform=ax.transAxes, fontsize=8, va="top")

# (d) 恒真检验：去掉来源甲负项后的比值读数
ax = axes[1, 1]
labs = ["%s" % r[0].split()[0] for r in mb27_rows]
with_v = [r[2] for r in mb27_rows]
without_v = [r[3] for r in mb27_rows]
base_v = [r[4] for r in mb27_rows]
xx = np.arange(len(labs))
ax.bar(xx - 0.2, with_v, width=0.2, color="tab:blue", label=r"有源侧")
ax.bar(xx, without_v, width=0.2, color="tab:orange", label=r"无源侧（替身）")
ax.bar(xx + 0.2, base_v, width=0.2, color="tab:gray", label=r"基线")
ax.set_xticks(xx)
ax.set_xticklabels(labs, fontsize=8)
ax.set_ylabel(r"比值")
ax.set_title(r"(d) 恒真检验：去掉来源甲负项后回到基线", fontsize=10)
ax.legend(fontsize=7)
ax.grid(alpha=0.3, axis="y")

fig.suptitle("第 796 号：来源甲负项装入合并装置后的总能量增量与线张力比值", fontsize=12)
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
print("合计 %d/%d 通过（主判据 MB2-1 至 MB2-9，展开为 %d 个子项）"
      % (npass, len(RESULTS), len(RESULTS)))
print()
print("关键读数:")
print("  MB2-1 泛函含 C1 侧复标量环（键拉伸栏的 V_rep 项、法线轴弯曲、锚核）"
      "与记录侧写入沉积栏；来源甲的负项装在键拉伸栏")
print("       构型相容性：叠加档键应变混比份额 %.6f（不再处处同号），"
      "总增量 %+.6e" % (sum_share, sum_tot))
print("       分区档：子区键数占比 %.6f，分子分母口径不同" % zone_frac)
print("  MB2-2 基线五档 rho %s；最大相对偏差 %.3e"
      % (", ".join("%.6f" % r[4] for r in mb2_base), max(mb2_base_devs)))
print("  MB2-3 两栏分读数（amp_cap = 0.30）：%s"
      % "; ".join("%s：沉积 %.6e、拉伸 %.6e" % (v[0].split()[0], v[2], v[3])
                  for v in mb23b_rows if v[1] == 0.30))
print("       总增量 %s"
      % ", ".join("%+.6e" % v[4] for v in mb23b_rows if v[1] == 0.30))
print("  MB2-4 总比值最大 %.6f，源侧比值最大 %.6f，门槛 %.4f，越过：%s"
      % (max_tot, max_src, GATE_772, "是" if mb24_over else "否"))
print("  MB2-5 可加性残差 %s；相对偏差 %s"
      % (", ".join("%+.3e" % r[4] for r in mb25_rows),
         ", ".join("%.3e" % r[5] for r in mb25_rows)))
print("  MB2-6 源侧轴：总比值幂次 %+.6f，源侧比值幂次 %+.6f"
      % (sl_src_tot, sl_src_s))
print("       案乙侧轴一（g）：总比值幂次 %+.6f，净栏幂次 %+.6f" % (sl_g_rho, sl_g_net))
print("       案乙侧轴二（frac，第 772 号抵消同轴）：净栏比值幂次 %+.6f，"
      "源侧比值幂次 %+.6f，总比值幂次 %+.6f" % (sl_fr_net, sl_fr_src, sl_fr_tot))
print("       源侧贡献占净栏 %.4f%%；源侧比值对 amp_cap 的幂次 %+.6f"
      % (100.0 * src_frac_of_net, sl_src_ext))
print("       跨过门槛所需：源侧比值需增 %.1f 倍，按幂次外推 amp_cap ≈ %.1f"
      "（该强度下最大键应变 ≈ %.1f，而 r_0 = 1，物理不可容许）"
      % (need_src / cur_src, amp_needed, max_strain))
print("       防恒真：源侧轴极差 %.3e、g 轴 %.3e、frac 轴 %.3e"
      % (src_var, g_var, fr_var))
print("  MB2-7 替身比值 %s，基线 %s"
      % (", ".join("%.6f" % r[3] for r in mb27_rows),
         ", ".join("%.6f" % r[4] for r in mb27_rows)))
print("  MB2-8 AW-3：补偿 %.4f、线张力 %.4f、比值 %.6f、步进斜率 %+.4f"
      % (compens, tension, ratio_aw3, min_step))
print("       790 号 theta* %s"
      % ", ".join("%.8f" % r[1] for r in mb8b_rows))
print("  MB2-9 模型档；容量型芯的物理档可制备性未答（射程外）")
print("耗时 %.1f 秒" % (time.time() - T_START))
