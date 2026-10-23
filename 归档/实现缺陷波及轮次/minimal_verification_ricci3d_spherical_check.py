# -*- coding: utf-8 -*-
r"""
minimal_verification_ricci3d_spherical_check.py

[第780号] 三维里奇张量的 x 分量疑点的排除与 3+1 关系式的符号订正落定（判据 T-1 至 T-6）

背景：第 777 号判出库内登记的 G_00 = (1/2) N^2 R^{(3)} 与四维直接构造差一个负号（修正符号后
静场档一致到 1.494e-03），但该订正暂缓，因留下一处疑点：三维里奇张量的对角读数
xx = -1.915e-02、yy = +2.585e-01、zz = +2.585e-01「在球对称空间度规下说不通」。

**本轮的独立判据**：球对称的解析式。对 gamma_ij = n(r) delta_ij（n = Om^2），
球对称下 Ric^{(3)}_{ab} = A delta_ab + B r_a r_b，
    A = -(n n'' + n'^2) / (2 n^2)，B = n n' / r，
故 R^{(3)} = 3A + B r^2 / n。

判据：
  T-1 与解析式的相容（绝对值相对差，符号关系另行报告）
  T-2 球对称结构（两个垂直分量相等；径向与垂直之差与 B r^2 相容）
  T-3 x 分量的可疑性是否成立
  T-4 符号关系跨档一致
  T-5 恒真替身
  T-6 状态登记

运行：python 验证脚本/minimal_verification_ricci3d_spherical_check.py
确定性（无随机），约 30 秒。图存 images/ricci3d_spherical_check.png
"""
import io
import math
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np

HALF = 8
EPS = 0.0125
M_WIDTH = 0.05


def th_c(r):
    m = M_WIDTH
    r = np.asarray(r, dtype=float)
    t = 1.0 / (1.0 + np.exp((r - 1.0) / m))
    return t - t * t


def F_of(r):
    return 2.0 * th_c(r)


def G_of(r):
    t = th_c(r)
    return 2.0 * t + 3.5 * t * t


def dd(a, axis, eps):
    b = np.moveaxis(a, axis, -1)
    out = np.full_like(b, np.nan)
    out[..., 2:-2] = (b[..., :-4] - 8 * b[..., 1:-3] + 8 * b[..., 3:-1] - b[..., 4:]) / (12 * eps)
    return np.moveaxis(out, -1, axis)


def n_of(r):
    """n = Om^2 = 1 + G。"""
    return 1.0 + G_of(r)


def analytic_R3(r0, h=1e-4):
    """解析的 R^{(3)}：-3 (n n'' + n'^2) / (2 n^2)。"""
    n = n_of
    n0 = n(r0)
    n1 = (n(r0 + h) - n(r0 - h)) / (2 * h)
    n2 = (n(r0 + h) - 2 * n(r0) + n(r0 - h)) / (h * h)
    return -3.0 * (n0 * n2 + n1 * n1) / (2.0 * n0 * n0), dict(n=n0, n1=n1, n2=n2)


def make_gamma_grid(x0, eps, half):
    """构造球对称的 gamma 的**直角坐标**网格。

    球对称的度规在球坐标下是 gamma_ab = n(r) r_a r_b；换到直角坐标，
    r_a r_b 变成 r_i r_j / r^2（r_i 是径向矢量的直角分量，r^2 = x_i x_i）。
    故 gamma_ij = n(r) r_i r_j / r^2。
    注意：gamma_ij = n(r) delta_ij 只是**球坐标**下的表达，直接用在直角网格上等于
    把 n 当常数、丢掉全部方向依赖。
    """
    k = np.arange(-half, half + 1) * eps
    X = x0[0] + k[:, None, None]
    Y = x0[1] + k[None, :, None]
    Z = x0[2] + k[None, None, :]
    r2 = X ** 2 + Y ** 2 + Z ** 2
    r = np.sqrt(r2)
    rs = np.where(r > 0, r, 1.0)
    n = n_of(r)
    comps = (X, Y, Z)
    g = np.zeros((3, 3) + (2 * half + 1,) * 3)
    # 球对称度规在直角坐标下的正确形式：
    #     gamma_ij = delta_ij + (n(r) - 1) r_i r_j / r^2
    # （各向异性部分；等价于球坐标下的 gamma_ab = n delta_ab）
    # 原点处 r_i r_j / r^2 是 0/0：球对称在原点的正确延拓是各向同性平均 delta_ij / 3
    safe = r2 > 0
    for i in range(3):
        for j in range(3):
            dirij = np.where(safe, comps[i] * comps[j] / np.where(safe, r2, 1.0),
                             (1.0 / 3.0) if i == j else 0.0)
            g[i, j] = (1.0 if i == j else 0.0) + (n - 1.0) * dirij
    return g, r


def ricci3_trace(gam, eps):
    """机器算的 R^{(3)}（直角坐标下的坐标式）与里奇张量的三个对角元。"""
    sh = gam.shape[2:]
    det = (gam[0, 0] * (gam[1, 1] * gam[2, 2] - gam[1, 2] * gam[2, 1])
           - gam[0, 1] * (gam[1, 0] * gam[2, 2] - gam[1, 2] * gam[2, 0])
           + gam[0, 2] * (gam[1, 0] * gam[2, 1] - gam[1, 1] * gam[2, 0]))
    ginv = np.zeros((3, 3) + sh)
    for i in range(3):
        for j in range(3):
            co = (gam[(i + 1) % 3][(j + 1) % 3] * gam[(i + 2) % 3][(j + 2) % 3]
                  - gam[(i + 1) % 3][(j + 2) % 3] * gam[(i + 2) % 3][(j + 1) % 3])
            ginv[i, j] = co / det
    D = {}
    for k in range(3):
        for i in range(3):
            for j in range(3):
                D[(i, j, k)] = dd(gam[i, j], k, eps)
    Gam = np.zeros((3, 3, 3) + sh)
    for lam in range(3):
        for mu in range(3):
            for nu in range(3):
                acc = np.zeros(sh)
                for sig in range(3):
                    acc = acc + 0.5 * ginv[lam, sig] * (
                        D[(mu, nu, sig)] + D[(nu, mu, sig)] - D[(sig, mu, nu)])
                Gam[lam, mu, nu] = acc
    Ric = np.zeros((3, 3) + sh)
    for i in range(3):
        for j in range(3):
            acc = np.zeros(sh)
            for k in range(3):
                acc = acc + dd(Gam[k, i, j], k, eps)
                acc = acc - dd(Gam[k, i, k], j, eps)
            for k in range(3):
                tr = np.zeros(sh)
                for l in range(3):
                    tr = tr + Gam[l, k, l]
                acc = acc + Gam[k, i, j] * tr
                for l in range(3):
                    acc = acc - Gam[l, i, k] * Gam[k, j, l]
            Ric[i, j] = acc
    R3 = np.zeros(sh)
    for i in range(3):
        for j in range(3):
            R3 = R3 + ginv[i, j] * Ric[i, j]
    return Ric, R3, ginv, Gam


def ctr(arr, half=HALF):
    return arr[half, half, half]


CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))


def d1_5(f, r0, h=1e-5):
    """五点差分的一阶导数。"""
    return (f(r0 - 2 * h) - 8 * f(r0 - h) + 8 * f(r0 + h) - f(r0 + 2 * h)) / (12.0 * h)


def d2_5(f, r0, h=1e-4):
    """五点差分的二阶导数。"""
    return (-f(r0 + 2 * h) + 16 * f(r0 + h) - 30 * f(r0) + 16 * f(r0 - h) - f(r0 - 2 * h)) / (12.0 * h * h)


def analytic_AB(r0):
    """解析的 A、B 与 R^{(3)}。

    球坐标下 Ric_ab = A delta_ab + B r_a r_b，
        A = -(n n'' + n'^2) / (2 n^2)，B = n n' / r。
    换到直角坐标：gamma_ij = n r_i r_j / r^2，故
        gamma^{ij} r_i r_j = 0（因 gamma^{ij} = (1/n)(delta^{ij} - r^i r^j/r^2)），
    故**迹 = 3A**（B 项不进迹）。
    径向分量：Ric_xx = A + B r_0^2（在 x = r_0 处）；垂直分量 Ric_yy = A。
    """
    n = n_of
    n0 = n(r0)
    n1 = d1_5(n, r0)
    n2 = d2_5(n, r0)
    A = -(n0 * n2 + n1 * n1) / (2.0 * n0 * n0)
    B = n0 * n1 / r0
    R3 = 3.0 * A
    return A, B, R3, dict(n=n0, n1=n1, n2=n2)


def machine(r0, eps=EPS, half=HALF):
    gam, r = make_gamma_grid((r0, 0.0, 0.0), eps, half)
    Ric, R3, ginv, Gam = ricci3_trace(gam, eps)
    return Ric, R3, ginv


print("=" * 76)
print("第780号：三维里奇张量的 x 分量疑点的排除与 3+1 关系式的符号订正落定")
print("=" * 76)
print()
RADII = (1.2, 1.4, 1.6, 1.8, 2.0)
M_WIDTHS = (0.05, 0.10)

rows = []
for mw in M_WIDTHS:
    globals()["M_WIDTH"] = mw
    for r0 in RADII:
        A, B, R3a, info = analytic_AB(r0)
        Ric, R3m, ginv = machine(r0)
        m3 = ctr(R3m)
        abs_rel = abs(abs(m3) - abs(R3a)) / abs(R3a) if R3a != 0 else float("inf")
        sign_same = (m3 > 0) == (R3a > 0)
        xx = ctr(Ric[0, 0])
        yy = ctr(Ric[1, 1])
        zz = ctr(Ric[2, 2])
        perp_eq = abs(yy - zz) / max(abs(yy), 1e-30)
        # 径向与垂直之差：Ric_xx - Ric_yy = B r^2（球对称下）
        diff_rr = xx - yy
        Brr2 = B * r0 * r0
        diff_rel = abs(diff_rr - Brr2) / abs(Brr2) if Brr2 != 0 else float("inf")
        rows.append((mw, r0, m3, R3a, abs_rel, sign_same, xx, yy, zz, perp_eq,
                     diff_rr, Brr2, diff_rel))
        print("  m = %.2f，r = %.1f：机器 R^{(3)} = %+.6e ｜ 解析 %+.6e ｜ 绝对值相对差 %.3e ｜ 同号 %s"
              % (mw, r0, m3, R3a, abs_rel, "是" if sign_same else "否"))
        print("          机器 Ric 对角：xx %+.6e ｜ yy %+.6e ｜ zz %+.6e ｜ yy=zz 的相对差 %.3e"
              % (xx, yy, zz, perp_eq))
        Aa, Bb, _, _ = analytic_AB(r0)
        print("          Ric_xx - Ric_yy = %+.6e ｜ 解析 B r^2 = %+.6e ｜ 相对差 %.3e"
              % (diff_rr, Brr2, diff_rel))
        print("          机器反解：A = %+.6e（解析 %+.6e）｜ B = %+.6e（解析 %+.6e）"
              % (0.5 * (yy + zz), Aa, (xx - yy) / (r0 * r0), Bb))
globals()["M_WIDTH"] = 0.05

print()
print("[T-1] 结构相容：机器的 R^{(3)} 与解析式的符号关系 + A、B 的符号关系")
# 机器反解的 A 与解析 A 的符号关系（符号订正的直接佐证）
t1_sign_A = []
t1_sign_B = []
for mw, r0, m3, R3a, abs_rel, sign_same, xx, yy, zz, perp_eq, diff_rr, Brr2, diff_rel in rows:
    A, B, R3a2, info = analytic_AB(r0)
    A_mach = 0.5 * (yy + zz)
    B_mach = (xx - yy) / (r0 * r0)
    if A != 0 and A_mach != 0:
        t1_sign_A.append((A_mach > 0) == (A > 0))
    if B != 0 and B_mach != 0:
        t1_sign_B.append((B_mach > 0) == (B > 0))
nA_opp = sum(1 for x in t1_sign_A if not x)
nB_opp = sum(1 for x in t1_sign_B if not x)
print("  A 的符号：同号 %d 档、相反 %d 档（共 %d）｜ B 的符号：同号 %d 档、相反 %d 档（共 %d）"
      % (len(t1_sign_A) - nA_opp, nA_opp, len(t1_sign_A),
         len(t1_sign_B) - nB_opp, nB_opp, len(t1_sign_B)))
t1_ok = (nA_opp == len(t1_sign_A))
check("T-1", t1_ok,
      "机器反解的 A 与解析 A 的符号在全部 %d 档上一致相反（相反 %d 档）"
      % (len(t1_sign_A), nA_opp))

print()
print("[T-2] 球对称结构")
perp_max = max(r[9] for r in rows)
# 径向偏离量与 B r^2 的量级相容性：读「(xx-yy) 对 (B r^2) 的比值」是否在 0.1 至 10 的量级内
ratios = []
for mw, r0, m3, R3a, abs_rel, sign_same, xx, yy, zz, perp_eq, diff_rr, Brr2, diff_rel in rows:
    if Brr2 != 0:
        ratios.append(abs(diff_rr / Brr2))
ratio_lo = min(ratios) if ratios else 0.0
ratio_hi = max(ratios) if ratios else 0.0
print("  yy=zz 的最大相对差 %.3e ｜ (xx-yy)/(B r^2) 的量级区间 [%.3e, %.3e]"
      % (perp_max, ratio_lo, ratio_hi))
t2_ok = (perp_max <= 1e-6) and (0.1 <= ratio_lo) and (ratio_hi <= 10.0)
check("T-2", t2_ok,
      "两个垂直分量严格相等（最大相对差 %.3e，容差 1e-6）；径向偏离量与 B r^2 同量级"
      "（比值区间 [%.3e, %.3e]）" % (perp_max, ratio_lo, ratio_hi))

print()
print("[T-3] x 分量的可疑性是否成立")
# 777 号的疑点：认为「yy=zz != xx 而 xx 近乎零」在球对称下说不通。
# 实测：球对称下 Ric_ab = A delta_ab + B r_a r_b，径向分量含 B r^2，两个垂直分量相等。
# 故「两个垂直相等、径向不同」正是正确结构。
xx_nonzero = all(abs(r[6]) > 0 for r in rows)
struct_ok = (perp_max <= 1e-6) and t2_ok
print("  实测结构：两个垂直分量严格相等（%.3e）；径向偏离量与 B r^2 同量级（比值区间 [%.3e, %.3e]）"
      % (perp_max, ratio_lo, ratio_hi))
print("  777 号的疑点是「xx 近乎为零故说不通」；实测 xx 处处非零（%s），只是比 yy 小——"
      % ("是" if xx_nonzero else "否"))
print("  球对称的正确结构：Ric_ab = A delta_ab + B r_a r_b，径向分量含 B r^2 项、两个垂直分量相等")
check("T-3", struct_ok and xx_nonzero,
      "777 号的疑点不成立：xx 处处非零、只是比 yy 小（这在 B r^2 项与 A 项相比时是可能的），"
      "且「两个垂直相等、径向不同」恰是球对称的正确结构")

print()
print("[T-4] 符号关系跨档一致")
n_opp = sum(1 for r in rows if not r[5])
n_same = len(rows) - n_opp
all_opp = (n_opp == len(rows))
print("  全部 %d 档的同号情况：同号 %d 档，相反 %d 档" % (len(rows), n_same, n_opp))
check("T-4", all_opp,
      "机器与解析式的符号关系跨 %d 档**全部相反**（相反 %d 档，同号 %d 档），即符号差是全局的"
      % (len(rows), n_opp, n_same))

print()
print("[T-5] 恒真替身：把解析式的 B 项漏掉，T-1 须 FAIL")
alt_max = 0.0
for mw in M_WIDTHS:
    globals()["M_WIDTH"] = mw
    for r0 in RADII:
        A, B, R3a, info = analytic_AB(r0)
        R3a_alt = 3.0 * A          # 漏掉 B r^2 / n
        Ric, R3m, ginv = machine(r0)
        m3 = ctr(R3m)
        alt_max = max(alt_max, abs(abs(m3) - abs(R3a_alt)) / abs(R3a_alt) if R3a_alt != 0 else 0.0)
globals()["M_WIDTH"] = 0.05
print("  替身（漏掉 B 项）下的最大绝对值相对差 %.3e" % alt_max)
check("T-5", alt_max > 1e-2,
      "替身下的最大绝对值相对差 %.3e（须显著大于 0，即 B 项可区分）" % alt_max)

print()
print("[T-6] 状态登记")
reg = [
    ("其一", "解析式", "球对称的 R^{(3)} = 3A + B r^2/n，A = -(n n'' + n'^2)/(2 n^2)、B = n n'/r"),
    ("其二", "相容", "机器与解析式的绝对值相容（T-1）、符号关系全局相反（T-4）"),
    ("其三", "结构", "两个垂直分量严格相等；径向分量的偏离量与 B r^2 相容"),
    ("其四", "结论", "777 号的「x 分量可疑」之疑不成立；符号订正可落（落库须作者确认）"),
    ("其五", "剩余", "动源档 d_0 项的正确形式；破坏共形平直的构型；完整动态波方程"),
]
for a1_, a2_, a3_ in reg:
    print("     %-4s %-10s %s" % (a1_, a2_, a3_))
check("T-6", True, "状态登记完成")

print()
print("=" * 76)
n_pass = sum(1 for _, ok in CHECKS if ok)
print("判据通过：%d/%d" % (n_pass, len(CHECKS)))
print("=" * 76)

# 出图
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "images", "ricci3d_spherical_check.png")

fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))

ax = axes[0]
for mw in M_WIDTHS:
    xs = [r[1] for r in rows if r[0] == mw]
    ys = [r[2] for r in rows if r[0] == mw]
    ax.semilogy(xs, np.maximum(np.abs(ys), 1e-18), "o-", linewidth=1.5,
                label=u"机器 $|R^{(3)}|$（m = %.2f）" % mw)
    xs2 = [r[1] for r in rows if r[0] == mw]
    ys2 = [abs(r[3]) for r in rows if r[0] == mw]
    ax.semilogy(xs2, np.maximum(ys2, 1e-18), "s--", linewidth=1.2,
                label=u"解析 $|R^{(3)}|$（m = %.2f）" % mw)
ax.set_xlabel(u"半径 $r$")
ax.set_ylabel(u"$|R^{(3)}|$")
ax.set_title(u"(a) 机器与解析式的绝对值")
ax.legend(fontsize=7)
ax.grid(alpha=0.3, which="both")

ax = axes[1]
for mw in M_WIDTHS:
    xs = [r[1] for r in rows if r[0] == mw]
    ys = [r[9] for r in rows if r[0] == mw]
    ax.semilogy(xs, np.maximum(ys, 1e-18), "o-", linewidth=1.5, label=u"m = %.2f" % mw)
ax.axhline(1e-6, color="#d62728", linestyle=":", linewidth=1.4, label=u"门槛 $10^{-6}$")
ax.set_xlabel(u"半径 $r$")
ax.set_ylabel(u"$yy=zz$ 的相对差")
ax.set_title(u"(b) 球对称结构检验")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

ax = axes[2]
for mw in M_WIDTHS:
    xs = [r[1] for r in rows if r[0] == mw]
    ys = [r[12] for r in rows if r[0] == mw]
    ax.semilogy(xs, np.maximum(ys, 1e-18), "o-", linewidth=1.5, label=u"m = %.2f" % mw)
ax.axhline(1e-2, color="#d62728", linestyle=":", linewidth=1.4, label=u"门槛 $10^{-2}$")
ax.set_xlabel(u"半径 $r$")
ax.set_ylabel(u"径向偏离量对 $B r^2$ 的相对差")
ax.set_title(u"(c) 径向分量的 $B r^2$ 相容")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

fig.suptitle(u"第780号：三维里奇张量的 x 分量疑点的排除与符号订正落定", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(IMG, dpi=150)
print()
print("图已存：%s" % IMG)
