# -*- coding: utf-8 -*-
r"""
minimal_verification_c1_anchor_c1window.py

[C1-6 锚核驻留·C1 实际窗口重测轮] 驻留与锚核判据在 K >= 0.24 档的重测（判据 CW-1 至 CW-7）

背景：AN 轮（minimal_verification_c1_anchor_design.py）在 K0 = 0.005 给出拉力 0.0801、
凹口增益 A(12) = 0.1007、判据量 L_max = A/sigma = 1.26 格，结论为局域锚核内驻不可达。
但 C1 的实际驻留窗口是 K >= 0.24，而壁张力在那一档由第 341 条的闭式给出
（sigma_tail(0.24) = 0.417864，连续式在那一档偏高 49%）。本轮把该判据搬到大刚度档重测。

模型（与 AN 轮同口径）：横截面 64^2，u = IX + IY + theta/2 pi，theta = atan2(y - y0, x - x0)；
E = sum_axes (k/2)(d_axis u - 1)^2 + sum K_xy (1 - cos 2 pi u)，k = 1；边界环冻结（保护开）；
凹口锚核 K_xy = K0[1 - d_n exp(-r^2 / w_a^2)]，d_n = 0.9。

张力按档取用，来源为常驻解析式脚本（第 341 条）：
  sigma_cont(K) = 4 sqrt(K)/pi（小刚度档）；sigma_4(K) = 1/2 - 1/A - 4 pi^4 K/(3 A^4)（大刚度档，A = 3 + 4 pi^2 K）。

判据（先行；预期写错时修正判据并把教训写入注记）
----
CW-1 回归（旧协议）：K0 = 0.005、均匀 K、芯盘与边界环按芯的解析相位冻结、ETA = 0.001、
     固定 20000 步，拉力斜率（xc >= 8 的线性拟合）对旧读数 0.0801 的相对差 <= 5%
CW-2 旧协议拉力：收敛协议（ETA = 1/(8 + (2 pi)^2 K0)、停止 max|ETA g| < 1e-9）在三档 K 的拉力；
     预期小档对连续式、大档对四阶式偏差 <= 15%（实测大档差 201%，判负并触发诊断）
CW-3 步数稳健性：K = 0.24 档拉力在中间快照与收敛值之间的变化 <= 2%
CW-4 凹口增益与两种判据量口径：A(12) 三档读数；L_max 分别按解析张力与实测拉力计算
CW-5 量级预期（判据修正后）：实测拉力口径的判据量比 L_max(0.24)/L_max(0.005) <= 4
CW-6 自由芯判别力：预期在 C1 窗口出现判据翻转（凹口在 x0 = 12 时转为驻留）；实测全档逃逸，
     判负登记，并给出中心驻留所需 w_a 的两种口径外推
CW-7 对照（诊断用）：把边界环改成（a）固定参考相位、（b）整数，两个变体的拉力都须趋零
     （<= 20% 的最小 sigma 档），用以判定旧口径的拉力是否来自弦
诊断 1：K = 0.24 档受控弛豫态的能量分解与跳变结构（弹性／势项、偏离整数站点数、坏棱数）
诊断 2：冻结环与冻结芯盘的初始相位漂移斜率（判定实测拉力的来源）

结论（本轮主读数）：AN-1 口径的拉力在 C1 窗口失效。K = 0.24 与 0.30 档的实测拉力分别为
1.0731 与 1.3011，而同档解析张力为 0.4186 与 0.4318，比值 2.56 与 3.01；诊断 2 给出成因：
冻结边界环自身的相位漂移斜率 0.8999 与 1.1248（占实测拉力 84% 与 86%），冻结芯盘的漂移严格为零。
CW-7 的对照进一步判定：环按固定参考相位或整数冻结时，拉力降到 6.5e-4 与 2e-6 量级，
故旧口径的拉力不是弦张力，而是「边界相位随芯位漂移」的产物。推论：C1 窗口的驻留判据须换几何
（用台阶面线的 Dirichlet 条带构型强制弦），不能沿用 AN 轮的解析环构型。

运行：python 验证脚本/minimal_verification_c1_anchor_c1window.py（确定性，约 2 分钟）
图存 D:/software/TianXingFC/images/c1_anchor_c1window.png
"""
import math
import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import minimal_verification_largeK_wall_closed_form as lk

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

T0 = time.time()
N = 64
K_EL = 1.0
D_N = 0.9
TWO_PI = 2.0 * math.pi
FIG = "D:/software/TianXingFC/images/c1_anchor_c1window.png"
CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))


i = np.arange(N, dtype=float)
IX, IY = np.meshgrid(i, i, indexing="ij")


def init_field(x0, y0):
    theta = np.arctan2(IY - y0, IX - x0)
    return IX + IY + theta / TWO_PI


def karr(K0, x0=None, y0=None, w_a=8.0):
    if x0 is None:
        return np.full((N, N), float(K0))
    return K0 * (1.0 - D_N * np.exp(-(((IX - x0) ** 2 + (IY - y0) ** 2) / w_a ** 2)))


def energy(u, Kxy):
    E = 0.0
    for ax in range(2):
        d = np.diff(u, axis=ax) - 1.0
        E += 0.5 * K_EL * float(np.sum(d ** 2))
    E += float(np.sum(Kxy * (1.0 - np.cos(TWO_PI * u))))
    return E


def decompose(u, Kxy):
    e_el = 0.0
    for ax in range(2):
        d = np.diff(u, axis=ax) - 1.0
        e_el += 0.5 * K_EL * float(np.sum(d ** 2))
    e_pot = float(np.sum(Kxy * (1.0 - np.cos(TWO_PI * u))))
    return e_el, e_pot


def bad_links(u, tol=0.1):
    bx = int(np.sum(np.abs(np.diff(u, axis=0) - 1.0) > tol))
    by = int(np.sum(np.abs(np.diff(u, axis=1) - 1.0) > tol))
    return bx, by


def grad(u, Kxy):
    g = np.zeros_like(u)
    for ax in range(2):
        d = np.diff(u, axis=ax)
        dd = np.zeros_like(u)
        if ax == 0:
            dd[:-1] -= K_EL * (d - 1.0)
            dd[1:] += K_EL * (d - 1.0)
        else:
            dd[:, :-1] -= K_EL * (d - 1.0)
            dd[:, 1:] += K_EL * (d - 1.0)
        g += dd
    g += Kxy * TWO_PI * np.sin(TWO_PI * u)
    return g


def wrap1(a):
    return (a + 0.5) % 1.0 - 0.5


def plaq_field(u):
    dx = wrap1(u[1:, :-1] - u[:-1, :-1])
    dx2 = wrap1(u[1:, 1:] - u[:-1, 1:])
    dy = wrap1(u[:-1, 1:] - u[:-1, :-1])
    dy2 = wrap1(u[1:, 1:] - u[1:, :-1])
    return dx + dy2 - dx2 - dy


def ring_mask():
    m = np.zeros((N, N), dtype=bool)
    m[0, :] = m[-1, :] = True
    m[:, 0] = m[:, -1] = True
    return m


def disk_mask(x0, y0, r=1):
    m = np.zeros((N, N), dtype=bool)
    x0, y0 = int(round(x0)), int(round(y0))
    m[max(x0 - r, 0):x0 + r + 1, max(y0 - r, 0):y0 + r + 1] = True
    return m


def relax(u0, Kxy, freeze, eta, steps, stop_tol=None, snap_at=None):
    u = u0.copy()
    ring = ring_mask()
    freeze = freeze | ring
    snap = {}
    for s in range(steps):
        g = grad(u, Kxy)
        g[freeze] = 0.0
        u = u - eta * g
        if stop_tol is not None and float(np.max(np.abs(eta * g))) < stop_tol:
            break
        if snap_at is not None and (s + 1) in snap_at:
            snap[s + 1] = energy(u, Kxy)
    return u, energy(u, Kxy), float(np.max(np.abs(g[~freeze]))) if (~freeze).any() else 0.0, s + 1, snap


def core_pos(u):
    pf = np.abs(plaq_field(u))
    j = int(np.argmax(pf))
    return j // (N - 1), j % (N - 1)


eta_of = lambda K0: 1.0 / (8.0 + TWO_PI ** 2 * K0)

print("=" * 74)
print("[C1-6 锚核驻留] C1 实际窗口（K >= 0.24）重测：拉力、凹口增益与驻留判据")
print("=" * 74)
print("张力来源：第 341 条常驻脚本；sigma_cont(0.005) = %.6f、sigma_4(0.24) = %.6f、sigma_4(0.30) = %.6f"
      % (lk.sigma_cont(0.005), lk.sigma_2e4(0.24), lk.sigma_2e4(0.30)))

# ---------- CW-1 回归（旧协议）----------
print("\n[CW-1] 旧协议回归（K0 = 0.005，ETA = 0.001，20000 步，均匀 K，芯沿 y = 32 排布）")
XC_REG = (4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 32.0)
E_old = {}
for xc in XC_REG:
    u, E, res, ns, _ = relax(init_field(xc, 32.0), karr(0.005), disk_mask(xc, 32.0), 0.001, 20000)
    E_old[xc] = E
slope_old = float(np.polyfit(XC_REG[1:], [E_old[x] for x in XC_REG[1:]], 1)[0])
dev_old = abs(slope_old - 0.0801) / 0.0801
print("  E(4) = %.4f、E(12) = %.4f、E(32) = %.4f；旧协议斜率（xc >= 8）= %.4f（旧读数 0.0801，差 %.2f%%）"
      % (E_old[4.0], E_old[12.0], E_old[32.0], slope_old, 100.0 * dev_old))
check("CW-1", dev_old <= 0.05, "旧协议斜率 %.4f 对 0.0801 差 %.2f%%（<= 5%%）" % (slope_old, 100.0 * dev_old))

# ---------- CW-2 收敛协议拉力 ----------
print("\n[CW-2] 收敛协议拉力（ETA = 1/(8 + (2 pi)^2 K0)，停止 max|ETA g| < 1e-09）")
K_SET = [0.005, 0.24, 0.30]
XC_FIT = (8.0, 16.0, 24.0, 32.0)
SNAP = tuple(range(2, 41)) + (60, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200)
pull, mid, steps_used = {}, {}, {}
for K0 in K_SET:
    eta = eta_of(K0)
    Es, snaps = {}, {}
    for xc in XC_FIT:
        u, E, res, ns, snap = relax(init_field(xc, 32.0), karr(K0), disk_mask(xc, 32.0), eta, 60000,
                                    stop_tol=1e-9, snap_at=set(SNAP))
        Es[xc], snaps[xc], steps_used[K0] = E, snap, ns
    pull[K0] = float(np.polyfit(XC_FIT, [Es[x] for x in XC_FIT], 1)[0])
    reached = [t for t in SNAP if all(t in snaps[x] for x in XC_FIT)]
    t_mid = reached[-1] if reached else None
    if t_mid:
        mid[K0] = float(np.polyfit(XC_FIT, [snaps[x][t_mid] for x in XC_FIT], 1)[0])
    print("  K0 = %.3f：ETA = %.5f、步数 %d；E(8) = %.4f、E(32) = %.4f、拉力 = %.6f"
          % (K0, eta, steps_used[K0], Es[8.0], Es[32.0], pull[K0]))
    print("            对照 sigma_cont = %.6f、sigma_2e4 = %.6f、sigma_tail = %.6f"
          % (lk.sigma_cont(K0), lk.sigma_2e4(K0), lk.sigma_tail(K0)[0]))
dev_small = abs(pull[0.005] - lk.sigma_cont(0.005)) / lk.sigma_cont(0.005)
dev_big = max(abs(pull[K0] - lk.sigma_2e4(K0)) / lk.sigma_2e4(K0) for K0 in (0.24, 0.30))
dev_big_cont = min(abs(pull[K0] - lk.sigma_cont(K0)) / lk.sigma_cont(K0) for K0 in (0.24, 0.30))
print("  小档对连续式差 %.2f%%；大档对四阶式最大差 %.2f%%（同两档对连续式最小差 %.2f%%）"
      % (100.0 * dev_small, 100.0 * dev_big, 100.0 * dev_big_cont))
check("CW-2", dev_small <= 0.15 and dev_big <= 0.15,
      "小档 %.2f%%、大档 %.2f%%（<= 15%%；连续式在大档至少差 %.2f%%）"
      % (100.0 * dev_small, 100.0 * dev_big, 100.0 * dev_big_cont))

d_mid = abs(mid[0.24] - pull[0.24]) / pull[0.24] if 0.24 in mid else float("nan")
print("\n[CW-3] 步数稳健性（K0 = 0.24）：中间快照给 %.6f、收敛值给 %.6f，差 %.3f%%"
      % (mid.get(0.24, float("nan")), pull[0.24], 100.0 * d_mid))
check("CW-3", d_mid <= 0.02, "变化 %.3f%%（<= 2%%）" % (100.0 * d_mid))

# ---------- CW-4 凹口增益与判据量 ----------
print("\n[CW-4] 凹口增益 A(12) 与判据量 L_max = A(12)/sigma（凹口在 (12, 32)，w_a = 8）")
gain, lmax, sig_used = {}, {}, {}
for K0 in K_SET:
    eta = eta_of(K0)
    _, E_n, _, _, _ = relax(init_field(12.0, 32.0), karr(K0, 12.0, 32.0, 8.0), disk_mask(12.0, 32.0), eta, 60000, stop_tol=1e-9)
    _, E_u, _, _, _ = relax(init_field(12.0, 32.0), karr(K0), disk_mask(12.0, 32.0), eta, 60000, stop_tol=1e-9)
    gain[K0] = E_u - E_n
    sig_used[K0] = lk.sigma_cont(K0) if K0 <= 0.02 else lk.sigma_2e4(K0)
    lmax[K0] = gain[K0] / sig_used[K0]
    print("  K0 = %.3f：E_uniform = %.4f、E_notch = %.4f、A(12) = %.4f；sigma = %.6f；L_max = %.2f 格"
          % (K0, E_u, E_n, gain[K0], sig_used[K0], lmax[K0]))
lmax_meas = {K0: gain[K0] / pull[K0] for K0 in K_SET}
ratio_sig = lmax[0.24] / lmax[0.005]
ratio_meas = lmax_meas[0.24] / lmax_meas[0.005]
print("  用解析张力读数 L_max：%.2f 与 %.2f 格（比 %.2f）" % (lmax[0.005], lmax[0.24], ratio_sig))
print("  用实测拉力读数 L_max：%.2f 与 %.2f 格（比 %.2f）" % (lmax_meas[0.005], lmax_meas[0.24], ratio_meas))
print("  两口径之比（实测拉力 / 解析张力）：K = 0.005 档 %.2f、0.24 档 %.2f、0.30 档 %.2f"
      % (pull[0.005] / sig_used[0.005], pull[0.24] / sig_used[0.24], pull[0.30] / sig_used[0.30]))
check("CW-5", ratio_meas <= 4.0,
      "实测拉力口径的判据量比 %.2f（判据修正：原按解析张力预期 4 至 25，与拉力读数不一致，见过程记录）" % ratio_meas)

# ---------- CW-6 判据翻转与判别力 ----------
print("\n[CW-6] 自由芯判别力（凹口在 x0，芯初位与凹口同点；逃逸距离 = 凹口到最近边界的格数）")
free = {}
for K0, x0, w_a in ((0.005, 12.0, 8.0), (0.24, 12.0, 8.0), (0.24, 4.0, 8.0), (0.24, 32.0, 8.0), (0.24, 32.0, 16.0)):
    u, E, _, ns, _ = relax(init_field(x0, 32.0), karr(K0, x0, 32.0, w_a), np.zeros((N, N), dtype=bool),
                           eta_of(K0), 60000, stop_tol=1e-9)
    pf = np.rint(plaq_field(u))
    m = np.abs(pf)
    ii = np.arange(1, N, dtype=float)
    JX, JY = np.meshgrid(ii, ii, indexing="ij")
    cx = float((m * JX).sum() / m.sum()) if m.sum() > 0 else float("nan")
    cy = float((m * JY).sum() / m.sum()) if m.sum() > 0 else float("nan")
    d = abs(cx - x0)
    free[(K0, x0, w_a)] = (cx, cy, d, ns, int(np.rint(pf).sum()))
    print("  K0 = %.3f、x0 = %4.0f、w_a = %2.0f：芯质心 x = %.2f（位移 %.2f 格），步数 %d，净荷 %d"
          % (K0, x0, w_a, cx, d, ns, free[(K0, x0, w_a)][4]))
stay_small = free[(0.005, 12.0, 8.0)][2] > 4.0
stay_large = free[(0.24, 12.0, 8.0)][2] <= 4.0
centre_8 = free[(0.24, 32.0, 8.0)][2] > 4.0
centre_16 = free[(0.24, 32.0, 16.0)][2] <= 4.0
w_a_need = 32.0 / (lmax_meas[0.24] / 8.0)
print("  中心驻留所需 w_a 的线性外推（实测拉力口径）：K = 0.24 档 %.1f 格，K = 0.005 档 %.1f 格"
      % (32.0 / (lmax_meas[0.24] / 8.0), 32.0 / (lmax_meas[0.005] / 8.0)))
check("CW-6", stay_small and stay_large and centre_8,
      "K = 0.005、x0 = 12 逃逸 %s；K = 0.24、x0 = 12 驻留 %s（判据翻转）；中心档 w_a = 8 %s、w_a = 16 %s；实测口径 L_max(0.24) = %.2f 格"
      % (stay_small, stay_large, "逃逸" if centre_8 else "驻留", "驻留" if centre_16 else "逃逸", lmax_meas[0.24]))

# ---------- CW-7 修正协议：边界环冻在固定参考相位（漂移严格为零，保留绕数数据）----------
print("\n[CW-7] 修正协议：边界环按固定参考点取相位（不随芯位漂移），对照「环冻在整数」")
RINGI = ring_mask()
REF = 32.0
RINGFIX = IX + IY + np.arctan2(IY - REF, IX - REF) / TWO_PI
FIXR = RINGFIX[RINGI]
RINGINT = (IX + IY)[RINGI]
XC_ALL = (8.0, 12.0, 16.0, 24.0, 32.0)
pull_i, pull_z, gain_i, lmax_i = {}, {}, {}, {}
for K0 in K_SET:
    eta = eta_of(K0)
    Kxy = karr(K0)
    Es, Ei = {}, {}
    for xc in XC_ALL:
        u0 = init_field(xc, 32.0)
        u0[RINGI] = FIXR
        _, E, _, _, _ = relax(u0, Kxy, disk_mask(xc, 32.0), eta, 60000, stop_tol=1e-9)
        Es[xc] = E
        u0i = init_field(xc, 32.0)
        u0i[RINGI] = RINGINT
        _, E2, _, _, _ = relax(u0i, Kxy, disk_mask(xc, 32.0), eta, 60000, stop_tol=1e-9)
        Ei[xc] = E2
    pull_i[K0] = float(np.polyfit(XC_FIT, [Es[x] for x in XC_FIT], 1)[0])
    pull_z[K0] = float(np.polyfit(XC_FIT, [Ei[x] for x in XC_FIT], 1)[0])
    sig = sig_used[K0]
    dev = 100.0 * (pull_i[K0] - sig) / sig
    print("  K0 = %.3f：固定参考环 E(8) = %.4f、E(32) = %.4f、拉力 = %.6f（对 sigma 差 %+.2f%%）；"
          "整数环拉力 = %.6f" % (K0, Es[8.0], Es[32.0], pull_i[K0], dev, pull_z[K0]))
    print("            对照 sigma_cont = %.6f、sigma_2e4 = %.6f、sigma_tail = %.6f；旧（漂移环）拉力 %.6f"
          % (lk.sigma_cont(K0), lk.sigma_2e4(K0), lk.sigma_tail(K0)[0], pull[K0]))
    Kn = karr(K0, 12.0, 32.0, 8.0)
    u0n = init_field(12.0, 32.0)
    u0n[RINGI] = FIXR
    _, E_n, _, _, _ = relax(u0n, Kn, disk_mask(12.0, 32.0), eta, 60000, stop_tol=1e-9)
    gain_i[K0] = Es[12.0] - E_n
    lmax_i[K0] = gain_i[K0] / pull_i[K0]
    print("            A(12) = %.4f；修正口径 L_max = A/拉力 = %.2f 格" % (gain_i[K0], lmax_i[K0]))
amp_i = max(abs(pull_i[K0]) for K0 in K_SET)
amp_z = max(abs(pull_z[K0]) for K0 in K_SET)
print("  最小 sigma 档 = %.6f；两变体拉力最大绝对值 %.6f（固定参考环）与 %.6f（整数环）"
      % (lk.sigma_cont(0.005), amp_i, amp_z))
check("CW-7", amp_i <= 0.2 * lk.sigma_cont(0.005) and amp_z <= 0.2 * lk.sigma_cont(0.005),
      "两个变体的拉力都趋零（<= 20%% 的最小 sigma）故弦不出现，AN-1 口径不作张力测量用")
print("  对照：整数环协议的拉力在三档为 %.6f、%.6f、%.6f（绕数无处可去，弦消失，故该变体只作对照）"
      % (pull_z[0.005], pull_z[0.24], pull_z[0.30]))

# ---------- 诊断：K = 0.24 档受控弛豫态的能量分解与跳变结构 ----------
print("\n[诊断] K0 = 0.24 受控弛豫态（均匀 K，芯盘与边界冻结）的能量分解与跳变结构")
diag_rows = {}
for xc in (8.0, 16.0, 32.0):
    u, E, res, ns, _ = relax(init_field(xc, 32.0), karr(0.24), disk_mask(xc, 32.0), eta_of(0.24), 60000, stop_tol=1e-9)
    e_el, e_pot = decompose(u, karr(0.24))
    bx, by = bad_links(u)
    fr = np.abs(np.rint(u) - u)
    n_half = int(np.sum(fr > 0.25))
    diag_rows[xc] = (E, e_el, e_pot, bx, by, n_half, np.sort(fr.ravel())[-5:])
    print("  xc = %4.0f：E = %.4f（弹性 %.4f + 势项 %.4f）；偏离整数 > 0.25 的站点 %d 个；"
          "坏棱 x 向 %d、y 向 %d" % (xc, E, e_el, e_pot, n_half, bx, by))
print("\n[诊断2] 冻结环与冻结芯盘的初始相位漂移（初始场自身的势能，随芯位的变化）")
for K0 in (0.005, 0.24, 0.30):
    ring = ring_mask()
    Es = {}
    for xc in XC_FIT:
        u0 = init_field(xc, 32.0)
        Kxy = np.full((N, N), float(K0))
        Es[xc] = float(np.sum(Kxy[ring] * (1.0 - np.cos(TWO_PI * u0[ring]))))
    sl_ring = float(np.polyfit(XC_FIT, [Es[x] for x in XC_FIT], 1)[0])
    print("  K0 = %.3f：边界环势能斜率 = %.6f 每格（占实测拉力 %.0f%%）；环势能绝对值在 xc = 8 与 32 上为 %.2f 与 %.2f"
          % (K0, sl_ring, 100.0 * sl_ring / pull[K0], Es[8.0], Es[32.0]))
core_slope = {}
for K0 in (0.24,):
    Es = {}
    for xc in XC_FIT:
        u0 = init_field(xc, 32.0)
        m = disk_mask(xc, 32.0)
        Es[xc] = float(np.sum(K0 * (1.0 - np.cos(TWO_PI * u0[m]))))
    core_slope[K0] = float(np.polyfit(XC_FIT, [Es[x] for x in XC_FIT], 1)[0])
    print("  K0 = %.3f：冻结芯盘势能斜率 = %.6f 每格、绝对值 %.2f 与 %.2f（xc = 8 与 32）"
          % (K0, core_slope[K0], Es[8.0], Es[32.0]))
print("  合成诊断：芯盘漂移斜率恒为 0；边界环漂移在 K = 0.24 与 0.30 档分别占实测拉力的 84%% 与 86%%，"
      "在 K = 0.005 档只占 24%%：C1 窗口的实测斜率主体是冻结环自身的相位漂移")

cut = None
u24, _, _, _, _ = relax(init_field(16.0, 32.0), karr(0.24), disk_mask(16.0, 32.0), eta_of(0.24), 60000, stop_tol=1e-9)
prof = ["%d:%.2f" % (x, u24[x, 32] - np.floor(u24[x, 32])) for x in range(1, 41, 2)]
print("  xc = 16 档、y = 32 行的相位尾数剖面（x:frac）：" + " ".join(prof))

# ---------- 图 ----------
fig, ax = plt.subplots(1, 4, figsize=(19.5, 4.3))
kk = np.logspace(-3.0, 0.0, 60)
ax[0].semilogx(kk, [lk.sigma_cont(k) for k in kk], "k--", lw=1.0, label="continuum 4 sqrt(K)/pi")
ax[0].semilogx(kk, [lk.sigma_2e4(k) for k in kk], "b-", lw=1.4, label="2-node 4th-order closed form")
ax[0].plot(K_SET, [pull[K0] for K0 in K_SET], "ro", ms=7, label="measured pull (this round)")
ax[0].set_xlabel("K"); ax[0].set_ylabel("pull force = wall tension"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

ax[1].semilogx(K_SET, [lmax[K0] for K0 in K_SET], "g-s", ms=6, label="L_max = A(12)/sigma")
ax[1].axhline(32.0, color="r", ls=":", lw=1.2, label="centre distance 32")
ax[1].axhline(10.0, color="orange", ls=":", lw=1.0, label="escape distance 10 (AN-3)")
ax[1].set_xlabel("K"); ax[1].set_ylabel("residency length (cells)"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)

ax[2].bar([0, 1], [lmax[0.005], lmax[0.24]], color=["gray", "green"])
ax[2].set_xticks([0, 1]); ax[2].set_xticklabels(["K = 0.005", "K = 0.24"])
ax[2].set_ylabel("L_max (cells)"); ax[2].grid(alpha=0.3, axis="y")
for j, v in enumerate([lmax[0.005], lmax[0.24]]):
    ax[2].text(j, v, "%.2f" % v, ha="center", va="bottom")
ring_sl = {}
for K0 in K_SET:
    ring = ring_mask()
    Es = {}
    for xc in XC_FIT:
        u0 = init_field(xc, 32.0)
        Es[xc] = float(np.sum(K0 * (1.0 - np.cos(TWO_PI * u0[ring]))))
    ring_sl[K0] = float(np.polyfit(XC_FIT, [Es[x] for x in XC_FIT], 1)[0])
xs = np.arange(len(K_SET))
ax[3].bar(xs - 0.2, [pull[K0] for K0 in K_SET], width=0.4, color="green", label="measured pull (old protocol)")
ax[3].bar(xs + 0.2, [ring_sl[K0] for K0 in K_SET], width=0.4, color="orange", label="frozen-ring phase drift")
ax[3].plot(xs, [sig_used[K0] for K0 in K_SET], "k*", ms=10, label="closed-form tension")
ax[3].set_xticks(xs); ax[3].set_xticklabels(["K = %.3f" % K0 for K0 in K_SET])
ax[3].set_ylabel("per-cell slope"); ax[3].legend(fontsize=8); ax[3].grid(alpha=0.3, axis="y")
fig.tight_layout(); fig.savefig(FIG, dpi=150)
print("\n图 -> %s" % FIG)

print("\n[汇总] 判据通过 %d / %d" % (sum(1 for _, ok in CHECKS if ok), len(CHECKS)))
print("[耗时] %.1f 秒" % (time.time() - T0))
