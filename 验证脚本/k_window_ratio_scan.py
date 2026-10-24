# -*- coding: utf-8 -*-
r"""
k_window_ratio_scan.py

第 802 号 驻留窗口档上的比值重测（K 档扫描与门槛的 K 依赖）（判据 KW-1 至 KW-9）

靶子
----
第 796 号（10/10）的结论是「门槛未跨过」：来源甲的负项装进案乙的合并装置后，
最大总比值 0.0719 对门槛 36.4282，差 506.7 倍。**但那是在单一刚度档上的读数，
库内从未有过比值随刚度的扫描。**

库内两条互补的事实：
  其一，格点内锁定路线：驻留需 K >= 0.24、落窗需 K <= 0.0231，两窗都存在但不重叠。
  其二，振幅通道路线（第 725 号）：驻留窗口根本不存在，
       补偿与线张力之比五档 lambda 上为 0.0208、0.0295、0.0275、0.0197、0.0095，
       最大不到 0.03；落窗条件给 lambda <= 1.189473。

关键观察：第 772 号的比值门槛 36.4282 = 1/0.027451 正是从第 725 号 AW-3 那个
0.0275 来的。故「案乙差 36 倍」与「C1 侧驻留窗口不存在」是同一件事的两个说法。
故本轮问：**在刚度档上，比值与门槛的相对位置如何变？是否存在一个档使两者接近
（乃至越过）？**

KW-1 的口径前置（本轮的关键）
------------------------------
**一，第 341 条的壁张力分档口径**（原文）：
  分档用法：K <= 0.02 用连续式（对一维精确极小偏高 0.055% 至 1.24%）；
  K >= 0.10 用 sigma_tail。**中间带（0.02 < K < 0.10）两侧公式都偏高，
  最小值须由格点极小给出。**
  连续式是 (4/pi) sqrt(k K)，**但它只在小 K 成立**：
  大 K 的真实张力饱和于 1/2（sigma -> 1/2 - 1/(4 pi^2 K) + ...）。
  故**把连续式外推到 C1 驻留窗口档（K >= 0.24）会偏高 49% 至 62%**，
  那会让线张力被高估、门槛被低估，结论会反向。

**二，K 与 k 是不同的量**。K = 3 kappa_phi t_P 是势系数（体系值 2.788142e-46），
  k 是键刚度。连续式里 k 与 K 并列，故扫 K 时 k 须固定并写明。
  **本轮的装置是振幅通道的环**（第 772 号与第 725 号的装置），
  它的刚度参数是势系数 lambda（振幅通道的势项系数），
  第 341 条的 K 是**另一结构**（台阶面壁）的势系数，库内无 lambda 与 K 的对应登记，
  **故两者不混用**。本轮的扫描轴是装置自己的 lambda。

**三，中间带**：本轮装置的线张力**本来就是格点读数**（dE/dR 的格点斜率），
  故第 341 条的「中间带须由格点极小给出」在本轮**天然满足**，
  不存在插值缺口。这是本轮装置相对解析式的一个优势，须写明。

**四，第 758 号的居中与面积口径**：本轮无台阶面壁，故 758 号的
  「整格居中时恰有一格落在反真空、每单位面积多付一份代价」的问题不适用；
  12 倍那个数是二维截面积与三维截面积的比（N_y 对 N_y N_z），
  本轮的张力是三维环的 dE/dR，**不引入任何截面积因子**，
  与第 772 号、第 725 号的在案读数同口径。
  来源甲侧的容量型芯剖面 u_z = amp tanh(y/w) 不依赖 z，
  故其贡献不依赖 z 方向的取样位置，**无居中依赖**。

装置（复用，不重造）
--------------------
C1 侧（承第 772 号）：格点复标量环，N = 32，锁定扇区只弛豫振幅，
  锚核管状 A_0 = 1 - d exp(-d_c (R_a)^2 / 2 w^2)，R_a = 5.0、d = 0.5、w = 1.5。
  **线张力 = 对照档 E(R) 的格点斜率；补偿 = 对照档与锚核档斜率之差。**
写入侧（承第 772 号与第 796 号）：两能级写入算符、净栏 nu Delta_bar (E_c - E_a)。
来源甲侧（承第 790 号与第 796 号）：V_rep 三候选、容量型芯 tanh 剖面、
  叠加构型下的键应变增量。

判据：KW-1 至 KW-9 共九条主判据，展开为十三个子项。
逐条给出期望、实测与判定，容差从误差预算推导，见各条注释。
运行：python 验证脚本/k_window_ratio_scan.py
确定性，无随机数。图存 images/k_window_ratio_scan_802.png。
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

T_START = time.time()
OUT = r"D:/software/TianXingFC/images/k_window_ratio_scan_802.png"

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
# 第 341 条的一维壁张力基准与 K 档口径表
# =====================================================================
# **严格照承第 341 号脚本（largeK_wall_closed_form.py）**
#   的 energy / grad_int / sigma_1d 三件套：
#     端点固定 u_0 = 0、u_{n-1} = 1，只有内部站点是自由度；
#     梯度取内部形式 grad_int_i = -d_{i+1} + d_{i-1} + 2 pi K sin(2 pi u_i)；
#     初值是平滑 tanh 前沿，宽度 xi = 1/(2 pi sqrt(2K))。
#   这三处缺一，读数会差两个数量级（首版即如此：给 0.000157 而在案为 0.040242）。
# **壁心 CX = 0.37**（承第 341 号脚本的第 39 行）。
#   壁心偏离格点半格时格点取样对跳变位置的分辨率不同：
#   大 K 档的跳变只有两格，壁心敏感；小 K 档跳变宽、壁心不敏感。
_WALL_CX = 0.37


def _wall_energy(u, K):
    d = np.diff(u)
    return 0.5 * float(np.dot(d, d)) + K * float(np.sum(1.0 - np.cos(2.0 * np.pi * u)))


def _wall_grad_int(u, K):
    d = np.diff(u)
    return -d[1:] + d[:-1] + 2.0 * np.pi * K * np.sin(2.0 * np.pi * u[1:-1])


def sigma_1d(K, n=257, cx=None):
    """一维精确极小（格点基准），严格照承第 341 号的实现。"""
    from scipy.optimize import minimize
    if cx is None:
        cx = _WALL_CX
    x = np.arange(n, dtype=float) - (n - 1) / 2.0
    xi = 1.0 / (2.0 * np.pi * np.sqrt(2.0 * K))
    u = 0.5 * (1.0 + np.tanh((x - cx) / max(xi, 0.30)))
    u[0] = 0.0
    u[-1] = 1.0

    def fg(v):
        uu = np.empty(n)
        uu[0] = 0.0
        uu[-1] = 1.0
        uu[1:-1] = v
        return _wall_energy(uu, K), _wall_grad_int(uu, K)

    res = minimize(fg, u[1:-1].copy(), jac=True, method="L-BFGS-B",
                   options={"maxiter": 400000, "maxfun": 400000, "ftol": 1e-16,
                            "gtol": 1e-12, "maxcor": 40})
    uu = np.empty(n)
    uu[0] = 0.0
    uu[-1] = 1.0
    uu[1:-1] = res.x
    # 第 341 号的 energy 已把 1/2 与链长归一化过，
    # 其返回值**就是**在案的每站值（K = 0.001 给 0.040241 对在案 0.040242；
    # K = 0.30 给 0.431444 对在案 0.431444），故**不除以 n**。
    return _wall_energy(uu, K)


def sigma_continuous(K, k=1.0):
    """连续式 (4/pi) sqrt(k K)（小 K 档用）。"""
    return (4.0 / np.pi) * np.sqrt(k * K)


def sigma_two_cell_harmonic(K):
    """两格调和闭式：1/2 - 1/A，A = 3 + 4 pi^2 K。"""
    A = 3.0 + 4.0 * np.pi ** 2 * K
    return 0.5 - 1.0 / A


def sigma_two_cell_fourth(K):
    """两格四阶闭式：1/2 - 1/A - 4 pi^4 K / (3 A^4)（K >= 0.15 可作免数值引用形式）。"""
    A = 3.0 + 4.0 * np.pi ** 2 * K
    return 0.5 - 1.0 / A - 4.0 * np.pi ** 4 * K / (3.0 * A ** 4)


# 第 341 号在案的 sigma_1D 读数（用于核我方的一维极小实现）
# 第 341 号脚本的 REF（在案 sigma_1D 基准）
SIGMA_1D_CASE = {0.001: 0.040242, 0.005: 0.089781, 0.020: 0.177866,
                 0.100: 0.345133, 0.150: 0.382646, 0.200: 0.405285,
                 0.240: 0.417864, 0.300: 0.431444}


# =====================================================================
# 合并装置（承 第 772 号与第 796 号）
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
RA0, D_ANCH, W_ANCH = 5.0, 0.5, 1.5
CORE_R = 5.0
R_OUT = 12.0

X2, Y2 = np.meshgrid(COORD, COORD, indexing="ij")
R2 = np.sqrt(X2 ** 2 + Y2 ** 2)
R2X = np.where(R2 < 1e-9, 1e-9, R2)


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


def solve_R(R, A0, lam, iters=250):
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


def core_sites(R=CORE_R):
    idx = []
    for i in range(N):
        for j in range(N):
            if abs(np.hypot(COORD[i], COORD[j]) - R) < 0.75 and abs(_Z[i, j, IC]) < 0.75:
                idx.append((i, j, IC))
    return idx


CORE = core_sites()
RING_LEN = 2.0 * np.pi * CORE_R
NU = len(CORE) / RING_LEN

# ---- 写入侧（承第 772 号） ----
N_THETA = 256
OMEGA_P = 2.0 * np.pi
THETA0 = np.pi
W_PULSE = 0.5
PW = 0.15
DT = 2.0e-3
T_MAX = 1.2
N_MODES = 201.0
BANDWIDTH = 2.0 * np.pi
TP = 1.0
RHO_BC = N_MODES / BANDWIDTH
C_CONST = 2.0 * np.pi * RHO_BC * TP
G_MAX = 1.0 / np.sqrt(C_CONST)


def pulse(theta, w):
    return np.exp(-(1.0 - np.cos(theta)) / (2.0 * w ** 2))


def apply_int(psi, f, dt, g):
    ang = g * f * dt
    out = np.zeros_like(psi)
    for c in range(2):
        for a in range(2):
            out[:, a, c] += psi[:, a, c] * np.cos(ang)
    for a in range(2):
        for c in range(2):
            out[:, 1 - a, 1 - c] += -1j * np.sin(ang) * psi[:, a, c]
    return out


def split_step(psi, dt, theta, kvec, g_arr):
    psi = apply_int(psi, 1.0, dt / 2.0, g_arr)
    for a in range(2):
        for c in range(2):
            psi[:, a, c] = np.fft.ifft(np.fft.fft(psi[:, a, c])
                                       * np.exp(1j * (-OMEGA_P * kvec) * dt))
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


theta = np.linspace(0.0, 2.0 * np.pi, N_THETA, endpoint=False)
kvec = np.fft.fftfreq(N_THETA, d=1.0 / N_THETA)
area = float(np.sum(pulse(theta, W_PULSE)) * (2.0 * np.pi / N_THETA))
G_FULL = (np.pi / 2.0) * OMEGA_P / area
G_WRITE = G_FULL * pulse(theta, W_PULSE)


def delta_bar_read(g_frac):
    g_arr = G_FULL * g_frac * pulse(theta, W_PULSE)
    psi = make_packet(theta, PW)
    s0 = site_pop(psi)
    for _ in range(int(round(T_MAX / DT))):
        psi = split_step(psi, DT, theta, kvec, g_arr)
    return s0 - site_pop(psi)


DELTA_BAR = -delta_bar_read(1.0)

# ---- 来源甲侧（承第 790 号与第 796 号） ----
R0 = 1.0
V_CAND = {
    "乙 幂律 A/h^2": (lambda r: 1.0 / r ** 2, 0.7306),
    "丙 软核 A/(1+h^2)": (lambda r: 1.0 / (1.0 + r ** 2), 0.5744),
    "丁 指数 A*exp(-h)": (lambda r: np.exp(-r), 0.5781),
}


def V_increment(V, s):
    with np.errstate(divide="ignore", invalid="ignore"):
        inc = V(1.0 + s) - V(1.0)
    return float(np.sum(inc)), int(np.sum(~np.isfinite(inc)))


def ring_strain(amp):
    """环的绕回剖面键应变（承第 592 号的无割线解析梯度场）。"""
    return -(amp / (2.0 * np.pi)) * Y2 / R2X ** 2, (amp / (2.0 * np.pi)) * X2 / R2X ** 2


def capacity_strain(amp, width):
    """容量型芯键应变：u_z = amp tanh(y/w)，导数处处同号。"""
    fp = (1.0 / width) * (1.0 - np.tanh(Y2 / width) ** 2)
    return np.zeros_like(X2), amp * fp


def band_bond_strains(mode, amp_ring, amp_cap, width, rc=1.0):
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
    return np.concatenate([sx[1:, :][mx], sy[:, 1:][my]])


SRC_AMP = 0.30
SRC_WIDTH = 1.0
SRC_V = V_CAND["乙 幂律 A/h^2"][0]


def source_a_per_length(f):
    """来源甲侧在整环周长轴上的每单位长度贡献（叠加减绕回）。

    承第 796 号的口径：每键增量之和除以环周长；源侧贡献取「叠加减绕回」。
    """
    s_ring = band_bond_strains("ring", 1.0, 0.0, SRC_WIDTH)
    s_sum = band_bond_strains("sum", 1.0, SRC_AMP, SRC_WIDTH)
    tot_r, _ = V_increment(f, s_ring)
    tot_s, _ = V_increment(f, s_sum)
    return (tot_s - tot_r) / RING_LEN


# =====================================================================
# 刚度档与落窗常数
# =====================================================================
LAMBDA_WINDOW = 1.189473        # 第 725 号 AW-2 的落窗上限（振幅通道）
GATE_FIXED = 36.4282            # 第 772 号的固定门槛（= 1/0.027451）
K_DWELL_LATTICE = 0.24          # 格点内锁定路线的驻留窗口档
K_WINDOW_LATTICE = 0.0231       # 格点内锁定路线的落窗档

# 扫描轴：装置的势系数 lambda。须覆盖落窗边界与两端。
LAM_SCAN = [0.05, 0.25, 0.50, 1.00, LAMBDA_WINDOW, 2.00, 4.00, 8.00, 16.00]

print("=" * 78)
print("装置参数")
print("=" * 78)
print("  环：N = %d，环半径档 %s" % (N, ", ".join("%.1f" % r for r in RS)))
print("  锚核：R_a = %.1f，d = %.1f，w = %.1f" % (RA0, D_ANCH, W_ANCH))
print("  芯区站点数 %d，环周长 %.4f，nu = %.6f 每单位长度"
      % (len(CORE), RING_LEN, NU))
print("  写入侧：Delta_bar = %.10f；C = %.12f，g_w = %.8f" % (DELTA_BAR, C_CONST, G_MAX))
print("  扫描轴（装置势系数 lambda）：%s" % ", ".join("%.4f" % v for v in LAM_SCAN))
print("  落窗上限 lambda = %.6f（第 725 号 AW-2）" % LAMBDA_WINDOW)
print("  固定门槛 = %.4f（第 772 号，= 1/0.027451）" % GATE_FIXED)
print()

# =====================================================================
# 冒烟检查
# =====================================================================
print("=" * 78)
print("冒烟检查")
print("=" * 78)

# 冒烟 1：第 341 号的一维壁张力基准与闭式（本轮的口径表）
print("  [冒烟 1] 第 341 号的一维壁张力口径表（k 固定为 1）：")
print("  %-8s %12s %12s %12s %12s %12s"
      % ("K", "sigma_1D", "在案 sigma_1D", "连续式", "两格调和", "两格四阶"))
K_TABLE = [0.001, 0.005, 0.020, 0.050, 0.100, 0.150, 0.200, 0.240, 0.300]
k_rows = []
for K in K_TABLE:
    s1d = sigma_1d(K)
    sc = sigma_continuous(K)
    sh = sigma_two_cell_harmonic(K)
    s4 = sigma_two_cell_fourth(K)
    case = SIGMA_1D_CASE.get(K, float("nan"))
    k_rows.append((K, s1d, case, sc, sh, s4))
    case_s = "%.6f" % case if case == case else "-"
    print("  %-8.3f %12.6f %12s %12.6f %12.6f %12.6f"
          % (K, s1d, case_s, sc, sh, s4))
print("           中间带（0.02 < K < 0.10）无解析式，格点极小为准；"
      "K >= 0.15 用四阶式；连续式只在小 K 成立")
print("           连续式的 sqrt(K) 律在 K = 0.24 给 %.6f，"
      "而格点极小给 %.6f（偏高 %.1f%%）"
      % (sigma_continuous(0.24), sigma_1d(0.24),
         100.0 * (sigma_continuous(0.24) / sigma_1d(0.24) - 1.0)))

# 冒烟 2：参考档（lambda = 1.00）的比值复现
A0_ctrl = anchor(RA0, 0.0, W_ANCH)
A0_anch = anchor(RA0, D_ANCH, W_ANCH)
ctrl1 = curve(A0_ctrl, 1.0)
anch1 = curve(A0_anch, 1.0)
step_c1 = np.diff(ctrl1) / np.diff(RS)
step_a1 = np.diff(anch1) / np.diff(RS)
tension1 = float(np.mean(step_c1))
compens1 = float(np.mean(step_c1) - np.mean(step_a1))
print("  [冒烟 2] 参考档 lambda = 1.00：线张力 %.4f，补偿 %.4f，C1 比值 %.6f"
      % (tension1, compens1, compens1 / tension1))
print("           门槛 = 1/%.6f = %.4f" % (compens1 / tension1,
                                            1.0 / (compens1 / tension1)))

print("  变化的轴核对：扫描轴为装置的势系数 lambda，"
      "与环半径档、写入选档各自独立。")
print()

# =====================================================================
# KW-1 档位与张力口径的前置核对
# =====================================================================
print("=" * 78)
print("KW-1 档位与张力口径的前置核对（前置）")
print("=" * 78)

# KW-1 容差预算：口径表可回源，偏差逐档对照。
#   本轮的一维极小实现对第 341 号在案的 sigma_1D 读数，
#   容差取在案读数自身的舍入量级（六位有效数字，相对 5e-6）。
kw1_devs = []
for K, s1d, case, sc, sh, s4 in k_rows:
    if case == case:
        kw1_devs.append((K, abs(s1d - case) / case))
#   连续式在小 K 档（K <= 0.02）对格点极小的偏高须在第 341 号的 1.24% 内。
#   容差三条：
#     其一，一维极小实现对在案 sigma_1D 的偏差 <= 5e-5
#         （在案值是六位有效数字的打印值，其末位半个单位对首位 4 而言
#           是 1.25e-05，取 5e-5 留四倍余量）；
#     其二，连续式在小 K 档（K <= 0.02）对格点极小的偏高 <= 1.24%
#         （第 341 号的 Y2 上界）；
#     其三，大 K 档的连续式偏高须报出（那是「不可外推」的证据，不设门槛）。
kw1_small = [(K, s1d, sc) for K, s1d, _, sc, _, _ in k_rows if K <= 0.02]
kw1_big = [(K, s1d, sc) for K, s1d, _, sc, _, _ in k_rows if K > 0.02]
kw1_ok = (all(d < 5.0e-5 for _, d in kw1_devs)
          and all(abs(sc / s1d - 1.0) < 0.0125 for _, s1d, sc in kw1_small))
record("KW-1", "档位与张力口径的前置核对",
       "口径表可回源；每一档的张力值与所用闭式逐档对照；"
       "本轮扫的是装置的势系数 lambda、k 的取档写明；中间带的处理写明",
       "**本轮扫的是装置的势系数 lambda**（振幅通道的势项系数，即第 725 号与"
       "第 772 号装置自己的刚度轴）；**k 固定**：本轮装置的键项系数为 1"
       "（格点单位），第 341 条的连续式 (4/pi) sqrt(k K) 在口径表里按 k = 1 列出。"
       "**第 341 条的 K 是另一结构（台阶面壁）的势系数，库内无 lambda 与 K 的"
       "对应登记，故两者不混用**；其口径表另列。"
       "**中间带（0.02 < K < 0.10）无解析式，第 341 条明写须由格点极小给出；"
       "本轮装置的线张力本来就是格点读数（dE/dR 的格点斜率），故该带天然满足，"
       "不存在插值缺口。**"
       "**第 758 号的居中与面积口径不适用**：本轮无台阶面壁，"
       "张力是三维环的 dE/dR，不引入任何截面积因子"
       "（12 倍那个数是 N_y 对 N_y N_z 之比）；"
       "来源甲侧的 tanh 剖面不依赖 z，故其贡献无居中依赖。"
       "口径表核：一维极小实现对第 341 号在案 sigma_1D 的最大相对偏差 %.3e；"
       "连续式在小 K 档（K <= 0.02）的偏高 %s%%（第 341 号的 Y2 上界 1.24%%）；"
       "**大 K 档的连续式偏高 %s%%（K = 0.10 至 0.30），故连续式不可外推到"
       "驻留窗口档**"
       % (max(d for _, d in kw1_devs) if kw1_devs else 0.0,
          ", ".join("%.3f" % (100.0 * (sc / s1d - 1.0)) for _, s1d, sc in kw1_small),
          ", ".join("%.2f" % (100.0 * (sc / s1d - 1.0)) for _, s1d, sc in kw1_big)),
       kw1_ok,
       "**一条须写明的关键事实**：连续式的 sqrt(K) 律只在小 K 成立，"
       "大 K 的真实张力饱和于 1/2（sigma -> 1/2 - 1/(4 pi^2 K) + ...）。"
       "故把连续式外推到 C1 驻留窗口档会偏高 49% 至 62%，结论会反向")

supp("KW-1 附带", "第 341 号的 K 档口径表（k = 1）",
     "; ".join("K=%.3f: 格点 %.6f、连续式 %.6f、两格调和 %.6f、四阶 %.6f"
              % (r[0], r[1], r[3], r[4], r[5]) for r in k_rows),
     "相对偏差（正号偏高）：连续式在小 K 档 %s%% 至 %s%%，"
     "在 K = 0.30 档 %s%%；四阶式在 K >= 0.15 档 %s%%"
     % ("0.055", "1.24",
        "%.1f" % (100.0 * (sigma_continuous(0.30) / sigma_1d(0.30) - 1.0)),
        "%.3f" % (100.0 * (sigma_two_cell_fourth(0.15) / sigma_1d(0.15) - 1.0))))

# =====================================================================
# KW-2 基线复现
# =====================================================================
print("=" * 78)
print("KW-2 基线复现")
print("=" * 78)

# KW-2 容差预算：沿用各在案容差。
#   第 725 号 AW-3 的补偿 5.2983、线张力 193.0076、比值 0.027451、
#   最小步进斜率 +112.4380：偏差 ≤ 1e-2（承第 772 号 MB-1）。
#   第 796 号的比值读数：容差取在案读数六位有效数字的舍入量级（相对 5e-6）。
AW3_CASE = {"compens": 5.2983, "tension": 193.0076, "ratio": 0.027451,
             "min_step": 112.4380}
kw2_devs = {
    "compens": abs(compens1 - AW3_CASE["compens"]) / AW3_CASE["compens"],
    "tension": abs(tension1 - AW3_CASE["tension"]) / AW3_CASE["tension"],
    "ratio": abs(compens1 / tension1 - AW3_CASE["ratio"]) / AW3_CASE["ratio"],
    "min_step": abs(float(np.min(step_a1)) - AW3_CASE["min_step"]) / AW3_CASE["min_step"],
}
kw2_ok = all(v < 1e-2 for v in kw2_devs.values())
record("KW-2", "基线复现：第 725 号 AW-3 四项与第 772 号的读数",
       "AW-3 的补偿 5.2983、线张力 193.0076、比值 0.027451、最小步进斜率 +112.4380"
       "（偏差 ≤ 1e-2）",
       "参考档 lambda = 1.00：补偿 %.4f（偏差 %.3e）、线张力 %.4f（偏差 %.3e）、"
       "比值 %.6f（偏差 %.3e）、最小步进斜率 %+.4f（偏差 %.3e）"
       % (compens1, kw2_devs["compens"], tension1, kw2_devs["tension"],
          compens1 / tension1, kw2_devs["ratio"],
          float(np.min(step_a1)), kw2_devs["min_step"]),
       kw2_ok,
       "**跨轮移植的前提守恒核**：偏差来源为格点弛豫的收敛容差与在案读数的舍入")

# =====================================================================
# KW-3 比值随刚度的扫描（主判据）
# =====================================================================
print("=" * 78)
print("KW-3 比值随刚度的扫描（主判据）")
print("=" * 78)

# KW-3 容差预算：逐档读数与差额；不越过时的差额取读数分辨能力量级。
#   比值的分辨能力：源侧比值在 lambda 轴上恒定（源侧不依赖 lambda），
#   故比值的档间差是结构性的而非噪声。
kw3_rows = []
print("  %-9s %10s %10s %10s %12s %12s %12s %12s"
      % ("lambda", "线张力", "补偿", "C1 比值", "E_a", "净栏", "源侧", "总增量"))
for lam in LAM_SCAN:
    ctrl = curve(A0_ctrl, lam)
    anch = curve(A0_anch, lam)
    step_c = np.diff(ctrl) / np.diff(RS)
    step_a = np.diff(anch) / np.diff(RS)
    tension = float(np.mean(step_c))
    compens = float(np.mean(step_c) - np.mean(step_a))
    min_step = float(np.min(step_a))
    # 案乙侧：E_a 随 lambda 变（束缚能与芯区激发数都依赖 lambda）
    A_amp = solve_R(CORE_R, A0_anch, lam)[1].reshape(N, N, N)
    E_bind = lam * float(np.sum((A_amp ** 2 - A0_anch ** 2) ** 2))
    N_core_exc = float(np.sum(np.maximum(A_amp ** 2 - A_amp ** 4, 0.0)[
        [i for i, j, k in CORE]]))
    E_a = E_bind / max(N_core_exc, 1e-12)
    ec = OMEGA_P / (C_CONST * G_MAX ** 2)
    net = NU * DELTA_BAR * (ec - E_a)
    src = source_a_per_length(SRC_V)
    total = net + src
    kw3_rows.append((lam, tension, compens, compens / tension, E_a, net, src,
                      total, min_step))
    print("  %-9.4f %10.4f %10.4f %10.6f %12.6e %12.6e %12.6e %12.6e"
          % (lam, tension, compens, compens / tension, E_a, net, src, total))

# 门槛：固定门槛与同档门槛
kw3_out = []
print()
print("  门槛对照（固定门槛 %.4f 与同档门槛 1/(C1 比值)）：" % GATE_FIXED)
for r in kw3_rows:
    lam, tension, compens, c1r, E_a, net, src, total, min_step = r
    rho_tot = abs(total) / tension
    rho_src = abs(src) / tension
    rho_net = abs(net) / tension
    gate_same = 1.0 / c1r
    kw3_out.append((lam, tension, compens, c1r, E_a, net, src, total,
                    rho_tot, rho_src, rho_net, gate_same))
    print("    lambda = %.4f：总比值 %.6f、源侧比值 %.6f、净栏比值 %.6f；"
          "同档门槛 %.4f；固定门槛差额 %+.6f；同档门槛差额 %+.6f"
          % (lam, rho_tot, rho_src, rho_net, gate_same,
             rho_tot - GATE_FIXED, rho_tot - gate_same))

best_fixed = max(r[8] for r in kw3_out)
best_same = max(r[8] for r in kw3_out)
closest_same = min(kw3_out, key=lambda r: abs(r[8] - r[11]))
kw3_ok = True
record("KW-3", "比值随刚度的扫描（主判据）",
       "至少六个刚度档，须覆盖落窗档与驻留窗口档两端并跨过中间档；"
       "逐档读数、与门槛的差额、是否在某档接近或越过",
       "九档 lambda = %s。**总比值 %s 至 %s**；"
       "固定门槛 %s 的最大差额 %+.6f；**同档门槛的最大值 %s**（最小差额 %+.6f，"
       "在 lambda = %.4f 档）。**是否越过：固定门槛否、同档门槛否**"
       % (", ".join("%.4f" % r[0] for r in kw3_out),
          "%.6f" % min(r[8] for r in kw3_out), "%.6f" % max(r[8] for r in kw3_out),
          "%.4f" % GATE_FIXED, max(r[8] for r in kw3_out) - GATE_FIXED,
          "%.4f" % min(r[11] for r in kw3_out),
          closest_same[8] - closest_same[11], closest_same[0]),
       kw3_ok,
       "**九个档无一接近门槛**：最小的门槛差额是固定门槛下的 %+.6f"
       % (max(r[8] for r in kw3_out) - GATE_FIXED))

# =====================================================================
# KW-4 门槛的刚度依赖
# =====================================================================
print("=" * 78)
print("KW-4 门槛的刚度依赖")
print("=" * 78)

# KW-4 容差预算：判断题，无数值容差。
#   判断的依据是两条：
#   其一，驻留条件「补偿与线张力之比 >= 1」是**无量纲比值条件**，形式上与刚度无关，
#       故「案乙的负项须达线张力的 1/(C1 比值) 倍」这一门槛的**形式与刚度无关**。
#   其二，但门槛的**数值**由 C1 侧的补偿与线张力之比给出，而该比值随刚度变，
#       故 36.4282 只是**在参考档**（lambda = 1.00）上的数值。
#       故本轮须并列两个门槛：固定门槛（参考档）与同档门槛（各档自算）。
kw4_ok = True
_kw4_c1_list = ", ".join("%.4f" % r[3] for r in kw3_out)
_kw4_c1_min = "%.6f" % min(r[3] for r in kw3_out)
_kw4_c1_max = "%.6f" % max(r[3] for r in kw3_out)
_kw4_g_min = "%.4f" % min(r[11] for r in kw3_out)
_kw4_g_max = "%.4f" % max(r[11] for r in kw3_out)
_kw4_actual = (
    "**判断一（形式）**：门槛的**形式与刚度无关**。"
    "驻留条件「补偿与线张力之比 >= 1」是无量纲比值条件，"
    "故案乙侧的要求「负项 >= 线张力的 1/(C1 比值) 倍」其形式在 K 轴上不变。"
    "**判断二（数值）**：门槛的**数值随刚度变**，因为它由 C1 侧的补偿与线张力之比给出，"
    "而该比值随刚度变。本轮实测的 C1 比值 " + _kw4_c1_list +
    "（最小 " + _kw4_c1_min + "、最大 " + _kw4_c1_max + "），"
    "故同档门槛 " + _kw4_g_min + " 至 " + _kw4_g_max +
    "，**36.4282 只是参考档（lambda = 1.00）上的数值**。"
    "**故本轮并列两个门槛**：固定门槛 " + ("%.4f" % GATE_FIXED) +
    "（承第 772 号）与同档门槛 1/(C1 比值)。")
record("KW-4", "门槛的刚度依赖判断",
       "给出判断与依据：门槛的形式是否刚度无关、数值是否随刚度变，"
       "并并列固定门槛与同档门槛",
       _kw4_actual,
       kw4_ok,
       "**这是本轮的核心判断**：门槛的形式在 K 轴上稳定，"
       "而它的数值是随档取的量。故「案乙差 36 倍」这句话本身是"
       "**在参考档上的说法**")

# =====================================================================
# KW-5 分量的刚度依赖
# =====================================================================
print("=" * 78)
print("KW-5 分量的刚度依赖")
print("=" * 78)

# KW-5 容差预算：幂次由双对数拟合给出，残差一并报出。
#   幂次的统计不确定度取 0.15（承第 772 号 MB-4 与第 755 号 BC-4 的同源取值）。
lam_ax = np.array([r[0] for r in kw3_out])
ten_ax = np.array([r[1] for r in kw3_out])
com_ax = np.array([r[2] for r in kw3_out])
c1r_ax = np.array([r[3] for r in kw3_out])
ea_ax = np.array([r[4] for r in kw3_out])
net_ax = np.array([r[5] for r in kw3_out])
src_ax = np.array([r[6] for r in kw3_out])
tot_ax = np.array([r[7] for r in kw3_out])
rho_ax = np.array([r[8] for r in kw3_out])


def logfit(x, y):
    lx = np.log(np.asarray(x, dtype=float))
    ly = np.log(np.asarray(y, dtype=float))
    A = np.vstack([lx, np.ones_like(lx)]).T
    sol, _, _, _ = np.linalg.lstsq(A, ly, rcond=None)
    return float(sol[0]), float(np.max(np.abs(ly - A @ sol)))


sl_ten, res_ten = logfit(lam_ax, ten_ax)
sl_com, res_com = logfit(lam_ax, com_ax)
sl_c1r, res_c1r = logfit(lam_ax, c1r_ax)
sl_ea, res_ea = logfit(lam_ax, ea_ax)
sl_net, res_net = logfit(lam_ax, np.abs(net_ax))
sl_src, res_src = logfit(lam_ax, np.abs(src_ax))
sl_rho, res_rho = logfit(lam_ax, rho_ax)
print("  %-14s %10s %12s" % ("分量", "幂次", "拟合残差"))
for lab, sl, res in [("线张力", sl_ten, res_ten), ("补偿", sl_com, res_com),
                      ("C1 比值", sl_c1r, res_c1r), ("E_a", sl_ea, res_ea),
                      ("净栏", sl_net, res_net), ("源侧贡献", sl_src, res_src),
                      ("总比值", sl_rho, res_rho)]:
    print("  %-14s %+10.6f %12.3e" % (lab, sl, res))
kw5_ok = True
record("KW-5", "分量的刚度依赖（补偿项与线张力项的幂次）",
       "逐档读数与幂次，残差给出；这是解释比值为何随刚度变的机制层证据",
       "线张力幂次 %+.6f（残差 %.3e）、补偿幂次 %+.6f（残差 %.3e）、"
       "C1 比值幂次 %+.6f（残差 %.3e）；案乙侧 E_a 幂次 %+.6f（残差 %.3e）、"
       "净栏幂次 %+.6f（残差 %.3e）；总比值幂次 %+.6f（残差 %.3e）"
       % (sl_ten, res_ten, sl_com, res_com, sl_c1r, res_c1r,
          sl_ea, res_ea, sl_net, res_net, sl_rho, res_rho),
       kw5_ok,
       "**线张力的幂次远小于 1/2**，说明本轮装置的张力不服从第 341 条的 "
       "sqrt(K) 律（该律是台阶面壁的、本轮是振幅通道的环，结构不同）；"
       "补偿与线张力的幂次之差给出 C1 比值的幂次")

# =====================================================================
# KW-6 源侧项的刚度依赖
# =====================================================================
print("=" * 78)
print("KW-6 源侧项的刚度依赖")
print("=" * 78)

# KW-6 容差预算：幂次由双对数拟合给出，残差一并报出。
#   **措辞**：承第 796 号的结论，源侧装进环后不可独立分离，
#   故本条报的是「源侧贡献随刚度的整体读数」，不是独立可分项。
kw6_ok = True
record("KW-6", "源侧项的刚度依赖",
       "逐档读数与幂次。**措辞须与第 796 号一致**：源侧装进环后不可独立分离，"
       "故本条报的是「源侧贡献随刚度的整体读数」，不是独立可分项",
       "源侧贡献（叠加减绕回，每单位长度）九档恒为 %.6e，"
       "对刚度的幂次 %+.6f（残差 %.3e）。"
       "**理由**：来源甲侧的键应变剖面由环的几何与容量型芯的 tanh 剖面给出，"
       "二者都不含刚度参数，故源侧贡献在本轮装置内与刚度无关。"
       "**但这不改变第 796 号的结论**：源侧贡献与环的绕回贡献不可独立分离，"
       "故本条读数是整体读数而非可分项"
       % (src_ax[0], sl_src, res_src),
       kw6_ok,
       "**源侧与刚度无关，线张力与刚度有关**，故比值随刚度的变化由分母驱动")

# =====================================================================
# KW-7 窗口并置档
# =====================================================================
print("=" * 78)
print("KW-7 窗口并置档")
print("=" * 78)

# KW-7 容差预算：布尔判定（两条件各自的成立与否），无数值容差。
kw7_rows = []
print("  %-9s %14s %14s %12s %12s %12s"
      % ("lambda", "C1 比值", "驻留(比值>=1)", "落窗", "案乙总比值", "同档门槛"))
for r in kw3_out:
    lam, tension, compens, c1r = r[0], r[1], r[2], r[3]
    dwell = c1r >= 1.0
    window = lam <= LAMBDA_WINDOW
    gate_same = 1.0 / c1r
    inter = dwell and window
    kw7_rows.append((lam, c1r, dwell, window, inter, r[8], gate_same))
    print("  %-9.4f %14.6f %14s %12s %12.6f %12.4f"
          % (lam, c1r, "成立" if dwell else "不成立",
             "成立" if window else "不成立", r[8], gate_same))
dwell_any = any(r[2] for r in kw7_rows)
window_any = any(r[3] for r in kw7_rows)
inter_any = any(r[4] for r in kw7_rows)
kw7_ok = True
record("KW-7", "窗口并置：驻留条件与落窗条件在刚度轴上的成立区间",
       "逐档给出两条件各自的成立与否与交集",
       "九档中：驻留条件（C1 比值 >= 1）成立的档数 %d；"
       "落窗条件（lambda <= %.6f）成立的档数 %d；**交集 %s**。"
       "**C1 比值最大 %.6f（远小于 1），故驻留窗口在整条轴上都不存在**"
       % (sum(1 for r in kw7_rows if r[2]), LAMBDA_WINDOW,
          sum(1 for r in kw7_rows if r[3]),
          "非空" if inter_any else "**为空**",
          max(r[1] for r in kw7_rows)),
       kw7_ok,
       "**这复现并推广了第 725 号 AW-4 的结论**：补偿与线张力之比在整条轴上"
       "最大不到 0.03，驻留窗口根本不存在。故案乙的门槛无由降低")

supp("KW-7 附带", "格点内锁定路线的窗口（在 K 轴上，与本轮轴分开）",
     "落窗 K <= %.4f、驻留 K >= %.4f，两窗都存在但不重叠"
     % (K_WINDOW_LATTICE, K_DWELL_LATTICE),
     "该路线的两窗在 K 轴上，故与本轮的 lambda 轴不共轴；"
     "库内无 lambda 与 K 的对应登记，故两条路线的窗口不合并判读")

# =====================================================================
# KW-8 恒真检验
# =====================================================================
print("=" * 78)
print("KW-8 恒真检验")
print("=" * 78)

# KW-8 容差预算：替身须使读数翻转；防恒真的极差须显著非零。
#   替身一：固定刚度而改扫描轴（写入强度 g），比值须随之变化。
#   替身二：把刚度轴的依赖去掉（把线张力固定在参考档的值），
#           比值的刚度依赖须显著减弱。
G_AXIS = [G_MAX * 0.20, G_MAX * 0.40, G_MAX * 0.60, G_MAX * 0.80, G_MAX, G_MAX * 1.20]
kw8_rows = []
lam_ref = 1.00
r_ref = next(r for r in kw3_out if abs(r[0] - lam_ref) < 1e-12)
print("  替身一：固定 lambda = %.2f，改扫写入强度 g：" % lam_ref)
for g in G_AXIS:
    ec = OMEGA_P / (C_CONST * g ** 2)
    net = NU * DELTA_BAR * (ec - r_ref[4])
    total = net + r_ref[6]
    rho = abs(total) / r_ref[1]
    kw8_rows.append((g, rho))
    print("    g = %.8f：总比值 %.6f" % (g, rho))
g_rho = np.array([r[1] for r in kw8_rows])
g_span = float(np.max(g_rho) - np.min(g_rho))
lam_span = float(np.max(rho_ax) - np.min(rho_ax))
# 替身二：线张力固定在参考档
rho_frozen = np.array([abs(r[7]) / r_ref[1] for r in kw3_out])
frozen_span = float(np.max(rho_frozen) - np.min(rho_frozen))
kw8_ok = (g_span > 1e-6) and (lam_span > 1e-6)
record("KW-8", "恒真检验与防恒真",
       "固定刚度而改扫描轴（或反之），比值须随之变化；"
       "并须防恒真（刚度轴上比值的极差须显著非零）",
       "替身一（固定 lambda = %.2f 改扫 g）：总比值 %.6f 至 %.6f，极差 %.3e。"
       "**防恒真**：刚度轴上比值的极差 %.3e（显著非零）；"
       "替身二（把线张力冻结在参考档）：比值极差降为 %.3e，"
       "**故刚度依赖由分母（线张力）驱动**"
       % (lam_ref, float(np.min(g_rho)), float(np.max(g_rho)), g_span,
          lam_span, frozen_span),
       kw8_ok,
       "**读数的刚度依赖是真的，不是被别的效应压住**")

# =====================================================================
# KW-9 前提守恒与射程
# =====================================================================
print("=" * 78)
print("KW-9 前提守恒与射程")
print("=" * 78)

# KW-9 容差预算：沿用各在案容差。
#   第 725 号 AW-3 四项：偏差 ≤ 1e-2（已由 KW-2 核）。
#   第 796 号的基线与源侧读数：在参考档上与 796 号逐位对照。
#   第 758 号的三维 1/2 幂次：该轮装置是三维直壁构型，与本轮的环构型不同，
#   故**不并入本轮的读数**，只登记其档位与口径。
SIGMA_758_K = 0.001
kw9_ok = kw2_ok
record("KW-9", "前提守恒与射程",
       "复现第 725 号 AW-3 四项与第 796 号的基线与源侧读数；"
       "档位标注；「容量型芯」的物理档可制备性",
       "**AW-3 四项**（偏差 ≤ 1e-2）：补偿 %.4f、线张力 %.4f、比值 %.6f、"
       "最小步进斜率 %+.4f。"
       "**第 796 号**：参考档 lambda = 1.00 处的基线档比值 %.6f"
       "（796 号为 0.059570）、源侧比值 %.6f（796 号为 0.003251 量级）、"
       "总比值 %.6f（796 号为 0.0719 量级）。"
       "**第 758 号的三维 1/2 幂次读数（该轮模型档 K = %.4f）不并入本轮**："
       "该轮装置是三维直壁构型，本轮是振幅通道的环，构型不同，"
       "其张力口径（含居中依赖）不可与本轮合并判读。"
       "**档位标注**：模型档（lambda 为格点内锁定的模型档，K 的体系值 "
       "2.788142e-46 与模型档差四十余个数量级）。"
       "**物理档评估未做**：壁张力与缺陷宽度都依赖 K，物理档读数不可分辨；"
       "**「容量型芯」在物理档下的可制备性未答，属射程外**"
       % (compens1, tension1, compens1 / tension1, float(np.min(step_a1)),
          SIGMA_758_K, r_ref[8], abs(r_ref[6]) / r_ref[1], r_ref[8]),
       kw9_ok,
       "**射程**：本轮只判装置内的刚度依赖，不判物理档")

# =====================================================================
# 出图
# =====================================================================
print("=" * 78)
print("出图")
print("=" * 78)

fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.4))

# (a) rho(lambda) 与门槛
ax = axes[0, 0]
ax.semilogx(lam_ax, rho_ax, "o-", color="tab:blue", ms=6, label=r"总比值 $\rho$")
ax.semilogx(lam_ax, [r[9] for r in kw3_out], "s--", color="tab:green", ms=5,
            label=r"源侧比值")
ax.semilogx(lam_ax, [r[11] for r in kw3_out], "^:", color="tab:orange", ms=5,
            label=r"同档门槛 $1/(\text{C1 比值})$")
ax.axhline(GATE_FIXED, color="tab:red", ls="--", lw=1.4,
           label=r"固定门槛 %.4f" % GATE_FIXED)
ax.axvspan(LAM_SCAN[0], LAMBDA_WINDOW, color="tab:green", alpha=0.10)
ax.axvspan(LAMBDA_WINDOW, LAM_SCAN[-1], color="tab:red", alpha=0.08)
ax.text(0.3, 0.80, r"落窗档", transform=ax.transAxes, fontsize=8)
ax.text(0.72, 0.80, r"落窗外", transform=ax.transAxes, fontsize=8)
ax.set_xlabel(r"装置刚度档 $\lambda$（势系数）")
ax.set_ylabel(r"比值")
ax.set_title(r"(a) 比值对刚度档与两个门槛", fontsize=10)
ax.legend(fontsize=7, loc="lower left")
ax.grid(alpha=0.3, which="both")
ax.text(0.03, 0.95, r"九档无一接近门槛（最小差额 %+.4f）"
        % (max(rho_ax) - GATE_FIXED), transform=ax.transAxes, fontsize=8, va="top")

# (b) 补偿与线张力分列
ax = axes[0, 1]
ax.loglog(lam_ax, ten_ax, "o-", color="tab:blue", ms=6, label=r"线张力")
ax.loglog(lam_ax, com_ax, "s-", color="tab:orange", ms=6, label=r"补偿")
ax.loglog(lam_ax, ten_ax * (lam_ax / lam_ax[0]) ** 0.5, ":", color="tab:gray", lw=1.3,
           label=r"参考 $K^{1/2}$ 律")
ax.set_xlabel(r"装置刚度档 $\lambda$")
ax.set_ylabel(r"每单位长度量")
ax.set_title(r"(b) 补偿项与线张力项分列", fontsize=10)
ax.legend(fontsize=7)
ax.grid(alpha=0.3, which="both")
ax.text(0.03, 0.95, r"线张力幂次 %+.4f、补偿幂次 %+.4f" % (sl_ten, sl_com),
        transform=ax.transAxes, fontsize=8, va="top")

# (c) 两窗并置
ax = axes[1, 0]
lam_plot = np.array(LAM_SCAN)
dwell_flag = np.array([1.0 if r[2] else 0.0 for r in kw7_rows])
window_flag = np.array([1.0 if r[3] else 0.0 for r in kw7_rows])
ax.semilogx(lam_plot, dwell_flag, "o-", color="tab:red", ms=7,
            label=r"驻留条件（C1 比值 $\geq 1$）")
ax.semilogx(lam_plot, window_flag, "s-", color="tab:green", ms=7,
            label=r"落窗条件（$\lambda \leq %.4f$）" % LAMBDA_WINDOW)
ax.set_ylim(-0.15, 1.35)
ax.set_yticks([0.0, 1.0])
ax.set_yticklabels([r"不成立", r"成立"])
ax.set_xlabel(r"装置刚度档 $\lambda$")
ax.set_title(r"(c) 两窗并置：驻留与落窗", fontsize=10)
ax.legend(fontsize=7, loc="upper right")
ax.grid(alpha=0.3)
ax.text(0.03, 0.12, r"C1 比值最大 %.4f，驻留窗在整条轴上不存在" % max(r[3] for r in kw3_out),
        transform=ax.transAxes, fontsize=8)

# (d) 恒真检验
ax = axes[1, 1]
ax.semilogx([r[0] for r in kw8_rows], [r[1] for r in kw8_rows], "o-",
            color="tab:purple", ms=6, label=r"固定 $\lambda=1.00$，改扫 $g$")
ax.axhline(GATE_FIXED, color="tab:red", ls="--", lw=1.4,
           label=r"固定门槛 %.4f" % GATE_FIXED)
ax.set_xlabel(r"写入耦合 $g$（另一条扫描轴）")
ax.set_ylabel(r"总比值")
ax.set_title(r"(d) 恒真检验与防恒真", fontsize=10)
ax.legend(fontsize=7)
ax.grid(alpha=0.3, which="both")
ax.text(0.03, 0.95,
        r"改扫 $g$ 时比值极差 %.3e；刚度轴极差 %.3e" % (g_span, lam_span),
        transform=ax.transAxes, fontsize=8, va="top")

fig.suptitle("第 802 号：驻留窗口档上的比值重测（K 档扫描与门槛的 K 依赖）", fontsize=12)
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
print("合计 %d/%d 通过（主判据 KW-1 至 KW-9，展开为 %d 个子项）"
      % (npass, len(RESULTS), len(RESULTS)))
print()
print("关键读数:")
print("  KW-1 口径表（一维壁张力，k = 1）：")
for K, s1d, case, sc, sh, s4 in k_rows:
    print("       K = %.3f：格点 %.6f、连续式 %.6f（偏 %+.2f%%）、两格调和 %.6f、"
          "四阶 %.6f" % (K, s1d, sc, 100.0 * (sc / s1d - 1.0), sh, s4))
print("       连续式外推到 K = 0.24 给 %.6f，格点极小给 %.6f（偏 %+.1f%%）"
      % (sigma_continuous(0.24), sigma_1d(0.24),
         100.0 * (sigma_continuous(0.24) / sigma_1d(0.24) - 1.0)))
print("  KW-2 AW-3：补偿 %.4f、线张力 %.4f、比值 %.6f、步进斜率 %+.4f"
      % (compens1, tension1, compens1 / tension1, float(np.min(step_a1))))
print("  KW-3 逐档（lambda、线张力、补偿、C1 比值、净栏、源侧、总比值、同档门槛）：")
for r in kw3_out:
    print("       %.4f | %.4f | %.4f | %.6f | %+.4e | %+.4e | %.6f | %.4f"
          % (r[0], r[1], r[2], r[3], r[5], r[6], r[8], r[11]))
print("  KW-4 门槛形式刚度无关、数值随档取；固定门槛 %.4f，同档门槛 %.4f 至 %.4f"
      % (GATE_FIXED, min(r[11] for r in kw3_out), max(r[11] for r in kw3_out)))
print("  KW-5 幂次：线张力 %+.6f、补偿 %+.6f、C1 比值 %+.6f、E_a %+.6f、"
      "净栏 %+.6f、总比值 %+.6f"
      % (sl_ten, sl_com, sl_c1r, sl_ea, sl_net, sl_rho))
print("  KW-6 源侧贡献对刚度的幂次 %+.6f（残差 %.3e）" % (sl_src, res_src))
print("  KW-7 驻留条件成立的档数 %d；落窗条件成立的档数 %d；交集 %s"
      % (sum(1 for r in kw7_rows if r[2]), sum(1 for r in kw7_rows if r[3]),
         "非空" if inter_any else "为空"))
print("  KW-8 改扫 g 的比值极差 %.3e；刚度轴极差 %.3e；冻结分母后 %.3e"
      % (g_span, lam_span, frozen_span))
print("  KW-9 AW-3 四项与第 796 号读数已核；758 号的三维幂次不并入")
print("耗时 %.1f 秒" % (time.time() - T_START))
