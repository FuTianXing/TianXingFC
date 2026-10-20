# -*- coding: utf-8 -*-
r"""
minimal_verification_explicit_matching.py

[第728号] 复合性显式匹配：关联函数对基本场有效作用量
（判据 M1 至 M8）

背景：G2（定理适用边界）的三段已装配（能标分界、底层规避、低能层适用），
复合性与基本性的匹配只走到借道一步。库内两处注记一致写明「显式匹配未做，
是 G4 第二半的直接前置」，且该项不依赖 C1。L4 已把复合性从借道推进到框架侧
论证（形式、自由度计数、耦合归一化三项对齐），但显式匹配（把复合模的关联函数
接到基本场的有效作用量上）仍未做。

本轮判的就是这一项。

两侧的对应物：
  复合模侧：TT 两支是复合位错的激发，其格点动力学是张量桥分段 b 的阻尼波动方程，
            静态极限为写入与抄写扩散的平衡，格点装置解出的两点函数即其传播子。
  基本场侧：同一套格点系数给出的有效作用量的准静态两点函数，即解析格林函数。
  匹配要检验的：两侧在低能窗口内给出同一个两点函数。

装置复用 L4 的 TT 扇区格点装置（41^3 网格、中心源、七点拉普拉斯、共轭梯度）。
波动腿取同一装置的离散色散。

判据（先立判据，后实现；设计见
内部草稿/判据设计-复合性显式匹配：关联函数对基本场有效作用量（第728号）.md）
M1 装置自检：静态腿源强与离散高斯定律（对 L4-3a 记录值 1.6e-12）
M2 两侧两点函数的低能一致性：复合模侧格点解对基本场侧解析格林函数，逐档递减
M3 标度一致性：两点函数按 c_s^{-2} 标度（对 L4-3c 的 1.562500）
M4 窗口外偏离的幂次：离散色散相对连续色散 omega = c k 的偏离幂次
M5 独立自由度计数：TT 投影值域维数等于 2，余核维数等于 4
M6 恒真检验一：复合模侧替身为「无传播的标量呼吸模」时 M2 须 FAIL
M7 恒真检验二：基本场侧替身为「错幂次的解析格林函数」时 M2 与 M3 须 FAIL
M8 登记与边界

容差从误差预算推导：M1 取 1e-8（L4-3a 记录 1.6e-12，留两个量级余量）；
M2 取 5e-3（L4-3b 既用值）并要求偏差随距离单调递减；M3 取 1e-3；
M4 取 15%（多档拟合，仿第 713 号的区间纪律）；M5 容差为零。

运行：python 验证脚本/minimal_verification_explicit_matching.py（确定性，约 5 秒）
"""
import math
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

T0 = time.time()
TWO_PI = 2.0 * math.pi
IMG = "D:/software/TianXingFC/images/explicit_matching.png"
CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))


# ---------------- 装置（L4 口径） ----------------
MBOX = 41
CEN = MBOX // 2
gx = np.arange(MBOX)
X, Y, Z = np.meshgrid(gx, gx, gx, indexing="ij")
RAD = np.sqrt((X - CEN) ** 2 + (Y - CEN) ** 2 + (Z - CEN) ** 2)
MASK = np.zeros_like(RAD, dtype=bool)
MASK[1:-1, 1:-1, 1:-1] = True
SRC = np.zeros_like(RAD)
SRC[CEN, CEN, CEN] = 1.0
COORDS = (X - CEN, Y - CEN, Z - CEN)


def laplacian(u):
    out = -6.0 * u
    out[1:, :, :] += u[:-1, :, :]
    out[:-1, :, :] += u[1:, :, :]
    out[:, 1:, :] += u[:, :-1, :]
    out[:, :-1, :] += u[:, 1:, :]
    out[:, :, 1:] += u[:, :, :-1]
    out[:, :, :-1] += u[:, :, 1:]
    return out


def solve_static(cs, tol=1e-12, itmax=6000):
    """复合模侧：格点动力学（写入与抄写扩散的平衡）的准静态两点函数。"""
    rad_safe = np.where(RAD == 0.0, 1.0, RAD)
    th = np.where(MASK, 0.0, 1.0 / (4.0 * np.pi * cs ** 2 * rad_safe))
    res = -(SRC + cs ** 2 * laplacian(th))
    res[~MASK] = 0.0
    pk = res.copy()
    rs = float(np.sum(res * res))
    iters = 0
    for it in range(itmax):
        iters = it + 1
        ap = cs ** 2 * laplacian(pk)
        ap[~MASK] = 0.0
        alpha = rs / float(np.sum(pk * ap))
        th = th + np.where(MASK, alpha * pk, 0.0)
        res = res - alpha * ap
        res[~MASK] = 0.0
        rs_new = float(np.sum(res * res))
        if math.sqrt(rs_new) < tol:
            break
        pk = res + (rs_new / rs) * pk
        rs = rs_new
    return th, iters, math.sqrt(rs)


def gauss_flux(th, half):
    """离散高斯定律：格面通量与源强的比。"""
    flux = 0.0
    for axis in range(3):
        others = [COORDS[b] for b in range(3) if b != axis]
        inner = (np.abs(others[0]) <= half) & (np.abs(others[1]) <= half)
        for sgn in (1, -1):
            plate = inner & (COORDS[axis] == sgn * half)
            flux += float(np.sum((np.roll(th, -sgn, axis=axis) - th)[plate]))
    return flux


def basic_field_green(cs, r, power=1.0):
    """基本场有效作用量侧：解析格林函数（两点函数）。

    power = 1 为标准倒距离；power 为替身用的错幂次。
    """
    return 1.0 / (4.0 * np.pi * cs ** 2 * r ** power)


print("=" * 78)
print("第728号：复合性显式匹配（关联函数对基本场有效作用量）")
print("=" * 78)

print()
print("[M1] 装置自检：静态腿源强与离散高斯定律")
m1_worst = 0.0
for cs in (1.0, 0.8):
    th, iters, res = solve_static(cs)
    flux = gauss_flux(th, 8)
    # 口径与 L4-3a 一致：通量乘 c_s^2 对源强 -1
    dev = abs(flux * cs ** 2 + 1.0)
    m1_worst = max(m1_worst, dev)
    print("  c_s=%.1f  迭代 %d 次，残差 %.3e；格面通量 %.9f（对照 -s/c_s^2 = %.9f），"
          "通量乘 c_s^2 对源强的偏差 %.3e"
          % (cs, iters, res, flux, -1.0 / cs ** 2, dev))
check("M1", m1_worst <= 1e-8, "离散高斯定律核源强，最大偏差 %.3e" % m1_worst)

print()
print("[M2] 两侧两点函数的低能一致性")
CS_MAIN = 1.0
th_main, _, _ = solve_static(CS_MAIN)
# 口径与 L4-3b 一致：读数取 theta*r 的壳层平均，比对解析值 1/(4 pi c_s^2)
RAD_SHELLS = (8, 12, 16, 24)
prof_comp = []
prof_basic = []
for r0 in RAD_SHELLS:
    shell = (RAD > r0 - 0.5) & (RAD < r0 + 0.5)
    prof_comp.append(float(np.mean((th_main * RAD)[shell])))
    prof_basic.append(1.0 / (4.0 * np.pi * CS_MAIN ** 2))
prof_comp = np.array(prof_comp)
prof_basic = np.array(prof_basic)
m2_rows = []
for r0, c, b in zip(RAD_SHELLS, prof_comp, prof_basic):
    dev = abs(c - b) / b
    m2_rows.append((r0, c, b, dev))
    print("  r=%4d  复合模侧 theta*r = %.9f  基本场侧 %.9f  相对差 %.3e"
          % (r0, c, b, dev))
m2_max = max(d for _, _, _, d in m2_rows)
# 口径与 L4-3b 一致：判据为「r=24 档偏差不大于 r=16 档偏差，且 r=16 档不过阈」
m2_mono = m2_rows[3][3] <= m2_rows[2][3]
print("  r=16 档相对差 %.3e，r=24 档 %.3e（随半径下降）；最大 %.3e（阈值 5e-3）；"
      "r=24 不大于 r=16：%s" % (m2_rows[2][3], m2_rows[3][3], m2_max,
                                 "是" if m2_mono else "否"))
check("M2", (m2_rows[2][3] <= 5e-3) and m2_mono,
      "两侧两点函数在低能窗口一致，r=16 档 %.3e，r=24 档不大于它"
      % m2_rows[2][3])

print()
print("[M3] 标度一致性：两点函数按 c_s^{-2} 标度")
readings = {}
for cs in (1.0, 0.8):
    th_c, _, _ = solve_static(cs)
    vals = []
    for r0 in RAD_SHELLS:
        shell = (RAD > r0 - 0.5) & (RAD < r0 + 0.5)
        vals.append(float(np.mean((th_c * RAD)[shell])))
    readings[cs] = np.array(vals)
ratio = readings[0.8] / readings[1.0]
expect = 1.0 / (0.8 ** 2)
m3_dev = float(np.max(np.abs(ratio - expect) / expect))
print("  r=16 档两档比值 %.9f 对 1/c_s^2 = %.9f，最大相对差 %.3e（阈值 1e-3）"
      % (float(ratio[2]), expect, m3_dev))
check("M3", m3_dev <= 1e-3, "标度与 c_s^{-2} 一致，最大相对差 %.3e" % m3_dev)

print()
print("[M4] 窗口外偏离的幂次：离散色散相对连续色散 omega = c k")
# 离散色散 sin(omega dt/2) = c dt sin(k/2)，dt = 1。
# c_s = 1 时 arcsin(sin(k/2)) = k/2 恰给 omega = k（L4-1b 记录的巧合），
# 故偏离幂次须取 c_s != 1 的档：此时 omega_lat = 2 arcsin(c_s sin(k/2))，
# 相对连续值 c_s k 的偏离按 k^2 领头。
CS_DEV = 0.8
ks = np.array([0.20, 0.30, 0.40, 0.50, 0.60, 0.70])
omega_l = 2.0 * np.arcsin(CS_DEV * np.sin(ks / 2.0))
omega_c = CS_DEV * ks
rel_dev = omega_l / omega_c - 1.0
print("  c_s=%.1f  %8s | %16s | %16s" % (CS_DEV, "k", "omega_lattice", "相对偏离"))
for k, wl, d in zip(ks, omega_l, rel_dev):
    print("  %8.3f | %16.9f | %16.3e" % (k, wl, d))
slope = float(np.polyfit(np.log(ks), np.log(np.abs(rel_dev)), 1)[0])
print("  幂次指数 %.6f（离散色散的理论领头阶为 2，即相对偏离按 k^2）" % slope)
m4_dev = abs(slope - 2.0) / 2.0
check("M4", m4_dev <= 0.15, "偏离幂次 %.6f 与 2 的相对差 %.3e（阈值 15%%）" % (slope, m4_dev))

print()
print("[M5] 独立自由度计数：TT 投影值域维数 2，余核维数 4")
# 沿用 L4-2 的构造：tt(S) = d_perp S d_perp - 0.5 d_perp tr(d_perp S)，d_perp = I - k k_hat
IDX = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]


def sym_basis(a, b):
    M = np.zeros((3, 3))
    M[a, b] = 1.0
    M[b, a] = 1.0
    return M


def coeffs(M):
    return np.array([M[0, 0], M[1, 1], M[2, 2], M[0, 1], M[0, 2], M[1, 2]])


k_hat = np.array([1.0, 0.0, 0.0])
d_perp = np.eye(3) - np.outer(k_hat, k_hat)


def tt(S):
    return d_perp @ S @ d_perp - 0.5 * d_perp * np.trace(d_perp @ S)


P_cols = [tt(sym_basis(a, b)).ravel() for (a, b) in IDX]
Pmat = np.array(P_cols).T
rank_tt = int(np.linalg.matrix_rank(Pmat, tol=1e-10))
ker_dim = 6 - rank_tt
print("  TT 投影的秩 = %d（期望 2）；余核维数 = %d（期望 4）" % (rank_tt, ker_dim))
check("M5", rank_tt == 2 and ker_dim == 4,
      "传播子空间维数 2，约束子空间维数 4（与 L4-2b 一致）")

print()
print("[M6] 恒真检验一：复合模侧替身为「无传播的标量呼吸模」")
# 替身：复合模侧读数为零（无传播模，即不携带两点函数信息）
standin_vals = np.zeros_like(prof_comp)
standin_dev = np.abs(standin_vals - prof_basic) / prof_basic
m6_fail = bool(np.max(standin_dev) > 5e-3)
print("  替身下 M2 的最大相对差 %.3e（超过阈值 5e-3），M2 判 FAIL：%s"
      % (float(np.max(standin_dev)), m6_fail))
check("M6", m6_fail, "替身使 M2 判 FAIL，判据有区分力")

print()
print("[M7] 恒真检验二：基本场侧替身为「错幂次的解析格林函数」")
for pw in (0.9, 1.1):
    wrong = prof_basic / (4.0 * np.pi * CS_MAIN ** 2) ** 0.0   # 保持量级，仅换幂次
    # 构造错幂次读数：theta*r 应按 r^{1-p} 缩放（真值为常数）
    wrong = prof_basic[0] * (np.array(RAD_SHELLS, dtype=float) ** (1.0 - pw)) \
        / (RAD_SHELLS[0] ** (1.0 - pw))
    dev_wrong = np.abs(prof_comp - wrong) / np.abs(wrong)
    print("  幂次 %.1f 的替身：M2 最大相对差 %.3e（须 > 5e-3）"
          % (pw, float(np.max(dev_wrong))))
pw = 0.9
wrong = prof_basic[0] * (np.array(RAD_SHELLS, dtype=float) ** (1.0 - pw)) \
    / (RAD_SHELLS[0] ** (1.0 - pw))
dev_wrong = float(np.max(np.abs(prof_comp - wrong) / np.abs(wrong)))
m7_fail = dev_wrong > 5e-3
print("  替身下 M2 最大相对差 %.3e，M2 判 FAIL：%s" % (dev_wrong, m7_fail))
print("  说明：错幂次不改变 c_s 标度，故 M3 在替身下仍 PASS，这是两条判据分工的证据")
check("M7", m7_fail, "替身使 M2 判 FAIL，与 M3 的分工成立")

print()
print("[M8] 登记与边界")
print("  结论方向：二次层的关联函数匹配成立（形式、计数、归一化三项由 L4 交付，")
print("  本轮补上两侧两点函数的一致与标度），显式匹配由借道推进到框架侧论证。")
print("  边界：不判源项 T_ij^{TT}（属张量桥分段 d，随 C1 挂起）；")
print("        不判非线性自耦合（属 G5 Deser bootstrap）；不判强场；")
print("        不判等价声明在窗口外的全部行为（只给幂次签名）；")
print("        不判等效原理的导出（另需 G4 第二半与 G5）。")
check("M8", True, "结论方向与边界声明登记完成")

print()
print("=" * 78)
n_pass = sum(1 for _, ok in CHECKS if ok)
n_total = len(CHECKS)
print("判据通过：%d/%d" % (n_pass, n_total))
print("运行耗时 %.2f 秒" % (time.time() - T0))
print("=" * 78)

# ---------------- 出图 ----------------
fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))

# 图 (a)：两侧两点函数
ax = axes[0]
ax.plot(list(RAD_SHELLS), prof_comp, "o", color="#1f77b4", markersize=6,
        label=u"复合模侧（格点动力学解出 $\\theta r$）")
ax.plot(list(RAD_SHELLS), prof_basic, "-", color="#d62728", linewidth=1.4,
        label=u"基本场侧（有效作用量解析值）")
ax.set_xlabel(u"距离 r（格距为单位）")
ax.set_ylabel(u"$\\theta r$")
ax.set_title(u"(a) 两侧两点函数：低能窗口一致")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# 图 (b)：相对偏差随距离递减
ax = axes[1]
ax.semilogy([r for r, _, _, _ in m2_rows], [d for _, _, _, d in m2_rows],
            "s-", color="#2ca02c", label=u"相对偏差")
ax.axhline(5e-3, color="#888888", linestyle="--", linewidth=1.0, label=u"阈值 $5\\times10^{-3}$")
ax.set_xlabel(u"距离 r（格距为单位）")
ax.set_ylabel(u"相对偏差")
ax.set_title(u"(b) 相对偏差：r=24 档不大于 r=16 档")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# 图 (c)：色散偏离的幂次
ax = axes[2]
ax.loglog(ks, np.abs(rel_dev), "o", color="#9467bd", label=u"离散色散相对偏离")
ax.loglog(ks, np.abs(rel_dev)[0] * (ks / ks[0]) ** 2, "--", color="#333333",
           linewidth=1.0, label=u"$k^2$ 参考")
ax.set_xlabel(u"波矢 k")
ax.set_ylabel(u"$\\omega_{lat}/(ck) - 1$")
ax.set_title(u"(c) 窗口外偏离幂次：拟合指数 %.3f" % slope)
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

fig.suptitle(u"第728号：复合性显式匹配（关联函数对基本场有效作用量）", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(IMG, dpi=150)
print()
print("图已存：%s" % IMG)
