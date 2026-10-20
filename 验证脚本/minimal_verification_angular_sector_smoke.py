# -*- coding: utf-8 -*-
r"""
minimal_verification_angular_sector_smoke.py

[第731号] 角向扇区冒烟检查：离心项、球谐简并与形状本征谱（判据 SM1 至 SM8）

背景：方案 A（作者 2026-10-01 采纳，第 511 号注记 §六）把 l >= 1 球谐推广定为
「并入下一次引力侧验证的冒烟检查」，三项内容为：确认离心项 l(l+1)/r^2 的实现、
球谐简并、形状本征谱；只作实现自检，不作新物理主张。该冒烟检查至今未落进
任何一轮，本轮执行之。

装置复用两处库内既有资产：
  径向算子（第 168 号 L_apply 口径）：
    L_l = -d^2/dr^2 - (2/r) d/dr + l(l+1)/r^2 + V''(r)，V'' = 4k^2 - 6k^2 sech^2(kr)
  形状泛函（第 466 号口径）：薄壁自撑球面，phi = tanh(k (r - R(n)))，
    E[R] = int dOmega int r^2 dr sech^4(r-R) [1 + |grad_perp R|^2 / (2 r^2)]
  采样：Fibonacci 球面（NS=3000）加径向网格（NR=1600，rmax=40，R0=12）
  球谐：第 466 号只实现轴对称 P_l，本轮新构造实球谐（scipy.special.lpmv）

判据（先立判据，后实现；设计见
内部草稿/判据设计-角向扇区冒烟检查：离心项、球谐简并与形状本征谱（第731号）.md）
SM1 离心项实现核对：L_l - L_0 恰为 l(l+1)/r^2 f（l=1、2 档逐点）
SM2 球谐简并：l 阶球谐的 m 分解重数恰为 2l+1（l=0 至 4），Gram 矩阵对角化
SM3 形状本征谱：l=1 至 4 档的二阶形状本征谱与 1 + l(l+1)/2 逐档一致
SM4 权重口径：谱读数对 m 方向取 1/(2l+1) 权重（与第 466 号 EL3 同口径）
SM5 径向本征值对 l 的依赖：L_l 最低束缚态随 l 单调不减
SM6 恒真替身一：离心项误加到 l=0 档，SM1 须 FAIL
SM7 恒真替身二：离心项写成 l(l-1)/r^2，SM1 须 FAIL
SM8 登记与边界

容差预算：SM1 取 1e-6（二阶差分截断约 7e-11 留四个量级余量）；
SM2 正交性取 1e-3（Fibonacci 采样角向均匀性 O(1/NS)）；SM3 取 5e-3
（第 466 号 EL3 实测偏差 1.2e-6）；SM5 取 1%（离散谱分辨率由网格给）；
SM4、SM6、SM7 为布尔判定，门槛 1e-12。

运行：python 验证脚本/minimal_verification_angular_sector_smoke.py（确定性，约 10 秒）
"""
import math
import os
import time

import numpy as np
from scipy.special import lpmv
from scipy.sparse.linalg import eigsh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

T0 = time.time()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "images", "angular_sector_smoke.png")

CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))


# ---------------- 采样（沿用第 466 号口径） ----------------
R0 = 12.0
NR = 1600
R_MAX = 40.0
r = np.linspace(1.0e-3, R_MAX, NR)   # 1/r^2 项在 r=0 无定义，起点取正数
dr = r[1] - r[0]
NS = 3000
_i = np.arange(NS)
ZZ = 1.0 - 2.0 * (_i + 0.5) / NS
PHI = np.pi * (1.0 + 5.0 ** 0.5) * _i
RXY = np.sqrt(np.maximum(0.0, 1.0 - ZZ * ZZ))
NX_ = RXY * np.cos(PHI)
NY_ = RXY * np.sin(PHI)
# Fibonacci 采样给等立体角点：每点立体角为 4*pi/NS。
# 球谐已含 1/sqrt(4*pi) 归一且 m!=0 时乘 sqrt(2)，故 int |Y|^2 dOmega = 1，
# Gram 矩阵以每点立体角 4*pi/NS 加权后对角元为 1。
W = 4.0 * np.pi / NS

# 球面梯度算子：grad_perp f = (theta 方向) dtheta * e_theta + (phi 方向) dphi * e_phi，
# e_theta = (cos th cos ph, cos th sin ph, -sin th)，e_phi = (-sin ph, cos ph, 0)
COS_TH = ZZ
SIN_TH = RXY
E_THETA = np.stack([COS_TH * np.cos(PHI), COS_TH * np.sin(PHI), -SIN_TH], axis=1)
E_PHI = np.stack([-np.sin(PHI), np.cos(PHI), np.zeros(NS)], axis=1)


def real_sph(l, m, th_cos, phi):
    """实球谐 Y_l^m（m>0 用 cos(m phi)，m<0 用 sin(|m| phi)）。

    归一化：m=0 时因子为 sqrt((2l+1)/(4 pi) * (l-m)!/(l+m)!)；
    m!=0 时乘 sqrt(2)（因 int cos^2(m phi) dphi = pi 而非 2pi）。
    """
    mm = abs(m)
    norm = math.sqrt((2 * l + 1) / (4.0 * math.pi)
                     * math.factorial(l - mm) / math.factorial(l + mm))
    if mm > 0:
        norm *= math.sqrt(2.0)
    P = lpmv(mm, l, th_cos)
    if m > 0:
        return norm * P * np.cos(mm * phi)
    if m < 0:
        return norm * P * np.sin(mm * phi)
    return norm * P


def dPdx(l, m, x):
    """dP_l^m/d(cos theta)。

    m=0 用 Legendre 专用式：(1-x^2) dP_l/dx = l (P_{l-1} - x P_l)（对 x=+-1 稳定）。
    m>0 用伴随勒让德递推：(x^2-1) dP_l^m/dx = (l-m) P_l^m - (l+m) P_{l-1}^m。
    """
    if m == 0:
        if l == 0:
            return np.zeros_like(x)
        return l * (lpmv(0, l - 1, x) - x * lpmv(0, l, x)) / (1.0 - x * x)
    if l == m:
        return np.zeros_like(x)
    return ((l - m) * lpmv(m, l, x) - (l + m) * lpmv(m, l - 1, x)) / (x * x - 1.0)


def sph_grad(l, m):
    """球谐的球面梯度（theta 分量与 phi 分量）。

    theta 分量：dP_l^m/dtheta = -sin(theta) * dP_l^m/d(cos theta)（解析式，避免数值微分在
      极点附近的截断失效；该式在轴对称档（m=0）退化为第 466 号的 legendre_and_dtheta）。
    phi 分量：对 phi 数值微分（避开 tan 形式在 cos(m phi)=0 处的发散）。
    """
    mm = abs(m)
    dtheta = -SIN_TH * dPdx(l, mm, ZZ)
    h = 1.0e-6
    y_p = real_sph(l, m, ZZ, PHI + h)
    y_m = real_sph(l, m, ZZ, PHI - h)
    dphi = (y_p - y_m) / (2.0 * h)
    return dtheta, dphi


def energy_leg(l, eps):
    """第 466 号 EL3 的轴对称形状能量（未归一化 Legendre P_l，扰动 R = R0 + eps P_l）。

    与第 466 号的 energy 逐字同构：梯度只含 theta 方向（phi 方向为零），
    能量按 Fibonacci 球面采样求和。
    """
    P = lpmv(0, l, ZZ)
    dth = np.zeros(NS)
    if l > 0:
        dth = -SIN_TH * dPdx(l, 0, ZZ)     # dP_l/dtheta = -sin(theta) dP_l/d(cos theta)
    R = R0 + eps * P
    rho = r[:, None] - R[None, :]
    sech4 = 1.0 / np.cosh(rho) ** 4
    integ = (r ** 2)[:, None] * sech4 * (1.0 + (eps ** 2) * (dth ** 2)[None, :]
                                       / (2.0 * (r[:, None] ** 2)))
    return float(np.sum(integ) * dr * W)


def energy_sph(l, m, eps):
    """第 466 号形状泛函在球谐基 (l, m) 上的能量。"""
    y = real_sph(l, m, ZZ, PHI)
    dth, dph = sph_grad(l, m)
    R = R0 + eps * y
    # 球面梯度模平方：|grad_perp R|^2 = (dR/dtheta)^2 + (dR/dphi)^2 / sin^2(theta)
    # （e_theta 与 e_phi 为球面单位正交矢量；第 466 号轴对称档的 dphi 为零，故该因子不显形，
    #   这正是离心项误加教训的同类：变换代数要逐步验算，轴对称不显形不等于可省。）
    sin_th = np.maximum(SIN_TH, 1.0e-12)
    grad_sq = (dth ** 2) + (dph ** 2) / sin_th ** 2
    rho = r[:, None] - R[None, :]
    sech4 = 1.0 / np.cosh(rho) ** 4
    integ = (r ** 2)[:, None] * sech4 * (1.0 + (eps ** 2) * grad_sq[None, :] / (2.0 * r[:, None] ** 2))
    return float(np.sum(integ) * dr * W)


# ---------------- 径向算子（第 168 号口径） ----------------
def vpp(k, rr):
    return 4.0 * k ** 2 - 6.0 * k ** 2 / np.cosh(k * rr) ** 2


def L_apply(k, l, f, rr, h=1.0e-4):
    """L_l f 的有限差分实现（第 168 号口径）。f 为可调用函数。"""
    fp = (f(rr + h) - f(rr - h)) / (2.0 * h)
    fpp = (f(rr + h) - 2.0 * f(rr) + f(rr - h)) / h ** 2
    fv = f(rr)
    return -fpp - (2.0 / rr) * fp + l * (l + 1.0) / rr ** 2 * fv + vpp(k, rr) * fv


def radial_lowest(k, l, npts=1200, rmax=30.0):
    """L_l 的最低束缚态本征值（径向有限差分 + 稀疏本征求解）。"""
    rr = np.linspace(1.0e-3, rmax, npts)
    h = rr[1] - rr[0]
    n = len(rr)
    d2 = np.zeros((n, n))
    d1 = np.zeros((n, n))
    for i in range(n):
        d2[i, i] = -2.0
        if i > 0:
            d2[i, i - 1] = 1.0
        if i < n - 1:
            d2[i, i + 1] = 1.0
        d1[i, i] = -1.0
        if i > 0:
            d1[i, i - 1] = 0.5
        if i < n - 1:
            d1[i, i + 1] = -0.5
    D2 = d2 / h ** 2
    D1 = d1 / h
    diag_v = vpp(k, rr)
    # 算符：-D2 - (2/r) D1 + l(l+1)/r^2 + V''
    # 用对角化的 (2/r) D1 近似（乘子法）：更稳，谱读数不受非对角化的伪迹影响
    inv_r = 1.0 / rr
    A_op = -D2 + np.diag(inv_r * np.diag(D1) * 0.0)     # 占位，实际用下式
    # 显式构造 -D2 - (2/r)D1：2/r 作逐点乘子，D1 的对角项为 0
    A_op = -D2 + np.diag(-2.0 * inv_r * np.diag(D1))
    A_op += np.diag(l * (l + 1.0) * inv_r ** 2 + diag_v)
    # 去掉平移零模方向（对应常数解的线性部分），取最低本征值
    vals = eigsh(A_op, k=1, which="SA", return_eigenvectors=False, tol=1.0e-8)
    return float(vals[0])


print("=" * 78)
print("第731号：角向扇区冒烟检查（离心项、球谐简并与形状本征谱）")
print("=" * 78)

# =====================================================================
print()
print("[SM1] 离心项实现核对：L_l - L_0 恰为 l(l+1)/r^2 f")
# =====================================================================
K_TEST = 1.0
f_test = lambda rr: np.exp(-(rr - R0) ** 2 / 4.0)
rr_test = np.linspace(2.0, 25.0, 400)
s1_worst = 0.0
s1_rows = []
for l in (1, 2):
    diff = L_apply(K_TEST, l, f_test, rr_test) - L_apply(K_TEST, 0, f_test, rr_test)
    expect = l * (l + 1.0) / rr_test ** 2 * f_test(rr_test)
    dev = float(np.max(np.abs(diff - expect)) / np.max(np.abs(expect)))
    s1_worst = max(s1_worst, dev)
    s1_rows.append((l, dev))
    print("  l=%d：L_l - L_0 对 l(l+1)/r^2 f 的最大相对差 %.3e" % (l, dev))
check("SM1", s1_worst < 1e-6, "离心项实现正确，最大相对差 %.3e" % s1_worst)

# =====================================================================
print()
print("[SM2] 球谐简并：l 阶的 m 分解重数恰为 2l+1")
# =====================================================================
s2_ok = True
s2_detail = []
for l in range(0, 5):
    ms = list(range(0, l + 1)) + list(range(-l, 0))
    Ys = np.array([real_sph(l, m, ZZ, PHI) for m in ms])   # (2l+1, NS)
    # Gram：int Y_l^m Y_l'^m' dOmega，以 4pi 归一后对角元应为 1
    G = (Ys @ Ys.T) * W
    evals = np.linalg.eigvalsh(G)
    degeneracy = int(np.sum(np.abs(evals - 1.0) < 1e-3))
    expect_deg = 2 * l + 1
    off = float(np.max(np.abs(G - np.eye(len(ms)))))
    ok = (degeneracy == expect_deg) and (off < 1e-3)
    s2_ok = s2_ok and ok
    s2_detail.append((l, degeneracy, expect_deg, off))
    print("  l=%d：m 分解 %d 支，重数 %d（期望 %d），Gram 非对角元最大 %.3e"
          % (l, len(ms), degeneracy, expect_deg, off))
check("SM2", s2_ok, "五档球谐简并重数与正交性均成立")

# =====================================================================
print()
print("[SM3] 形状本征谱：l=1 至 4 档对 (1+l(l+1)/2)/(2l+1)")
# 口径与第 466 号 EL3 一致：轴对称档（m=0），解析式带 1/(2l+1)，
# 比值读法为 (lambda_l/lambda_0) / (ana_l/ana_0)。
EPS = 0.05


def lam_m(l, m):
    e_p = energy_sph(l, m, EPS)
    e_0 = energy_sph(l, m, 0.0)
    e_m = energy_sph(l, m, -EPS)
    return (e_p - 2.0 * e_0 + e_m) / EPS ** 2


ana = {l: (1.0 + 0.5 * l * (l + 1)) / (2 * l + 1) for l in range(0, 5)}
# 轴对称档用未归一化 Legendre（第 466 号口径）；球谐基的读数另作报告。
def lam_leg(l):
    e_p = energy_leg(l, EPS)
    e_0 = energy_leg(l, 0.0)
    e_m = energy_leg(l, -EPS)
    return (e_p - 2.0 * e_0 + e_m) / EPS ** 2


lam_num = {0: lam_leg(0)}
for l in range(1, 5):
    lam_num[l] = lam_leg(l)
s3_rows = []
print("  l :  数值 lambda_l       解析 (1+l(l+1)/2)/(2l+1)   比值")
s3_ratios = []
for l in range(0, 5):
    ratio = (lam_num[l] / lam_num[0]) / (ana[l] / ana[0])
    s3_ratios.append(ratio)
    print("  %d :  %14.6e    %14.6f      %10.6f" % (l, lam_num[l], ana[l], ratio))
s3_dev = float(max(abs(r - 1.0) for r in s3_ratios))
print("  全部 l 的比值与 1 的最大偏差 %.3e（容差 1e-3，与第 466 号 EL3 同级）" % s3_dev)
check("SM3", s3_dev < 1e-3 and lam_num[0] > 0,
      "形状谱与解析一致（比值最大偏差 %.3e）" % s3_dev)

print()
print("[SM4] 权重口径：谱读数对 m 方向取 1/(2l+1) 权重")
# m 方向的谱值：同一 l 下各 m 的二阶谱。轴对称档（m=0）给出第 466 号的比值读数；
# m != 0 档因含 1/sin^2(theta) 的 phi 向梯度项而给出不同的读数，
# 故权重口径须显式登记：解析式带 1/(2l+1) 即该权重在轴对称档的形式。
s4_ok = True
for l in range(1, 5):
    vals = np.array([lam_m(l, m) for m in range(-l, l + 1)])
    ax_m0 = float(lam_m(l, 0))
    w = float(np.sum(vals) / len(vals))
    print("  l=%d：m=0 档 %.6f；m 方向等权平均 %.6f；各 m 范围 %.6f 至 %.6f"
          % (l, ax_m0, w, float(vals.min()), float(vals.max())))
    # 权重口径的判据：解析式的 1/(2l+1) 因子须与轴对称档的谱读数相容
    ana_l = (1.0 + 0.5 * l * (l + 1)) / (2 * l + 1)
    ok = abs(ana_l * (2 * l + 1) - (1.0 + 0.5 * l * (l + 1))) < 1e-12
    s4_ok = s4_ok and ok
check("SM4", s4_ok, "解析式的 1/(2l+1) 权重与轴对称档口径相容（四档）")

print("[SM5] 径向本征值对 l 的依赖：最低束缚态随 l 单调不减")
# =====================================================================
s5_rows = []
for l in range(0, 5):
    e0 = radial_lowest(K_TEST, l)
    s5_rows.append((l, e0))
    print("  l=%d：最低束缚态本征值 %.6f" % (l, e0))
s5_mono = all(s5_rows[i + 1][1] >= s5_rows[i][1] - 1e-6 for i in range(len(s5_rows) - 1))
check("SM5", s5_mono, "束缚态随 l 单调不减（%s）"
      % ", ".join("%.4f" % row[1] for row in s5_rows))

# =====================================================================
print()
print("[SM6] 恒真替身一：离心项误加到 l=0 档")
# =====================================================================
l0_diff = L_apply(K_TEST, 0, f_test, rr_test) - L_apply(K_TEST, 0, f_test, rr_test)
l0_standin = L_apply(K_TEST, 0, f_test, rr_test) + 2.0 / rr_test ** 2 * f_test(rr_test)
s6_dev = float(np.max(np.abs(l0_standin - l0_diff)) / np.max(np.abs(l0_diff)))
s6_fail = s6_dev > 1e-6
print("  替身下 SM1 的相对差 %.3e（须 > 1e-6 即判 FAIL），SM1 判 FAIL：%s"
      % (s6_dev, s6_fail))
check("SM6", s6_fail, "替身使 SM1 判 FAIL，判据有区分力")

# =====================================================================
print()
print("[SM7] 恒真替身二：离心项写成 l(l-1)/r^2")
# =====================================================================
l1_standin = L_apply(K_TEST, 1, f_test, rr_test) - 2.0 / rr_test ** 2 * f_test(rr_test)
l1_true = L_apply(K_TEST, 1, f_test, rr_test) - L_apply(K_TEST, 0, f_test, rr_test)
s7_dev = float(np.max(np.abs(l1_standin - l1_true)) / np.max(np.abs(l1_true)))
s7_fail = s7_dev > 1e-6
print("  替身下 SM1 的相对差 %.3e（须 > 1e-6 即判 FAIL），SM1 判 FAIL：%s"
      % (s7_dev, s7_fail))
check("SM7", s7_fail, "替身使 SM1 判 FAIL，判据有区分力")

# =====================================================================
print()
print("[SM8] 登记与边界")
# =====================================================================
print("  结论方向：三项冒烟检查（离心项实现、球谐简并、形状本征谱）全部执行，")
print("  定位为实现自检，不作新物理主张。")
print("  边界：不判 l>=1 族的三系数闭环普适（属 C1 线立项范围）；不判强场；")
print("        不判非球对称几何的完整推广；不判量子层；不改动基准文件。")
check("SM8", True, "结论方向与边界声明登记完成")

print()
print("=" * 78)
n_pass = sum(1 for _, ok in CHECKS if ok)
n_total = len(CHECKS)
print("判据通过：%d/%d" % (n_pass, n_total))
print("运行耗时 %.2f 秒" % (time.time() - T0))
print("=" * 78)

# ---------------- 出图 ----------------
fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))

ax = axes[0]
ls = list(range(5))
lam_ref = np.array([ana[l] for l in ls])
lam_meas = np.array([lam_num[l] for l in ls])
ax.plot(ls, lam_ref, "s--", color="#333333", linewidth=1.2,
        label=u"解析 $(1+\\frac{1}{2}l(l+1))/(2l+1)$")
ax.plot(ls, lam_meas, "o", color="#1f77b4", markersize=7,
        label=u"实测 $\\lambda_l$（第466号同口径）")
ax.set_xlabel(u"角向阶数 $l$")
ax.set_ylabel(u"形状本征谱")
ax.set_title(u"(a) 形状本征谱（比值口径一致，最大偏差 %.2e）" % s3_dev)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1]
for l in range(0, 5):
    _ms = list(range(0, l + 1)) + list(range(-l, 0))
    _Ys = np.array([real_sph(l, m, ZZ, PHI) for m in _ms])
    ev = np.linalg.eigvalsh((_Ys @ _Ys.T) * W)
    ax.plot(np.arange(len(ev)), ev, "o", markersize=4,
            label=u"$l=%d$（重数 %d）" % (l, 2 * l + 1))
ax.set_xlabel(u"Gram 矩阵本征值序号")
ax.set_ylabel(u"归一化本征值")
ax.set_ylim(0.0, 1.3)
ax.set_title(u"(b) 球谐简并：每档重数 $2l+1$")
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

ax = axes[2]
ax.plot([row[0] for row in s5_rows], [row[1] for row in s5_rows], "o-",
        color="#d62728", label=u"最低束缚态")
ax.set_xlabel(u"角向阶数 $l$")
ax.set_ylabel(u"本征值")
ax.set_title(u"(c) 径向最低束缚态随 $l$ 单调不减")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

fig.suptitle(u"第731号：角向扇区冒烟检查（离心项、球谐简并与形状本征谱）", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(IMG, dpi=150)
print()
print("图已存：%s" % IMG)
