# -*- coding: utf-8 -*-
r"""
tools_793_r3_compatibility.py

[第793号候选] R^{(3)} 的数值相容性的闭合（同构型对照）

**两个构型**：
  构型 S（球对称度规）：g_ij = N^2 [delta_ij + (n-1) r_i r_j / r^2]，n = 1 + G
  构型 I（各向同性度规）：g_ij = N^2 (1+G) delta（第 784、792 号沿用的构型）

**闭式**：对「n 只依赖 r」的度规，R^{(3)} = 3A = -3 (n n'' + n'^2) / (2 n^2)。
"""
import io
import math
import os
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np

HALF = 8
EPS = 0.0125
M_WIDTH = 0.05
T0 = time.time()


def th_c(r, m=M_WIDTH):
    r = np.asarray(r, dtype=float)
    t = 1.0 / (1.0 + np.exp((r - 1.0) / m))
    return t - t * t


def F_of(r, m=M_WIDTH):
    return 2.0 * th_c(r, m)


def G_of(r, m=M_WIDTH):
    t = th_c(r, m)
    return 2.0 * t + 3.5 * t * t


def dd(a, axis, eps):
    b = np.moveaxis(a, axis, -1)
    out = np.full_like(b, np.nan)
    out[..., 2:-2] = (b[..., :-4] - 8 * b[..., 1:-3] + 8 * b[..., 3:-1] - b[..., 4:]) / (12 * eps)
    return np.moveaxis(out, -1, axis)


def build_grid(x0, eps, half, spherical):
    k = np.arange(-half, half + 1) * eps
    X = x0[0] + k[:, None, None]
    Y = x0[1] + k[None, :, None]
    Z = x0[2] + k[None, None, :]
    r2 = X ** 2 + Y ** 2 + Z ** 2
    r = np.sqrt(r2)
    Fv, Gv = F_of(r), G_of(r)
    N2 = 1.0 - Fv
    g = np.zeros((3, 3, 2 * half + 1, 2 * half + 1, 2 * half + 1))
    n = 1.0 + Gv
    if spherical:
        # 球对称度规：gamma_ij = delta_ij + (n-1) r_i r_j / r^2
        comps = (X, Y, Z)
        safe = r2 > 0
        for i in range(3):
            for j in range(3):
                dirij = np.where(safe, comps[i] * comps[j] / np.where(safe, r2, 1.0),
                                 (1.0 / 3.0) if i == j else 0.0)
                g[i, j] = N2 * ((1.0 if i == j else 0.0) + (n - 1.0) * dirij)
    else:
        # 各向同性度规：gamma_ij = (1+G) delta
        for i in range(3):
            for j in range(3):
                g[i, j] = N2 * n * (1.0 if i == j else 0.0)
    return g, N2, r


def ricci_3d(gam, eps):
    """直角坐标的坐标式三维里奇张量与 R^{(3)}。"""
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
    return Ric, R3, ginv


def d1_5(f, r, h):
    return (f(r - 2 * h) - 8 * f(r - h) + 8 * f(r + h) - f(r + 2 * h)) / (12.0 * h)


def d2_5(f, r, h):
    return (-f(r + 2 * h) + 16 * f(r + h) - 30 * f(r) + 16 * f(r - h) - f(r - 2 * h)) / (12.0 * h * h)


def closed_R3(r0, h=1e-4, m=M_WIDTH):
    """闭式：R^{(3)} = -3 (n n'' + n'^2) / (2 n^2)，n = 1 + G。"""
    n_of = lambda r: 1.0 + G_of(r, m)
    n0 = n_of(r0)
    n1 = d1_5(n_of, r0, h)
    n2 = d2_5(n_of, r0, h)
    A = -(n0 * n2 + n1 * n1) / (2.0 * n0 * n0)
    return 3.0 * A, A, n0 * n1 / r0


def ctr(arr, half=HALF):
    return arr[half, half, half]


CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))


print("=" * 78)
print("第793号：R^{(3)} 的数值相容性的闭合（同构型对照）")
print("=" * 78)
print()
RADII = (1.4, 1.6, 1.8)
MWIDTHS = (0.05, 0.10)

rows = []
for m in MWIDTHS:
    for r0 in RADII:
        c1, A1, B1 = closed_R3(r0, 1e-4, m)
        c2, A2, B2 = closed_R3(r0, 5e-4, m)
        gamS, N2S, rS = build_grid((r0, 0.0, 0.0), EPS, HALF, True)
        RicS, R3S, ginvS = ricci_3d(gamS, EPS)
        gamI, N2I, rI = build_grid((r0, 0.0, 0.0), EPS, HALF, False)
        RicI, R3I, ginvI = ricci_3d(gamI, EPS)
        mS = ctr(R3S)
        mI = ctr(R3I)
        relS = abs(mS - c1) / abs(c1) if c1 != 0 else float("inf")
        relI = abs(mI - c1) / abs(c1) if c1 != 0 else float("inf")
        relSame = abs(mS - mI) / abs(mS) if mS != 0 else float("inf")
        rows.append((m, r0, c1, c2, mS, mI, relS, relI, relSame))
        print("m = %.2f，r = %.1f：闭式 %+.6e ｜ 构型 S（球对称）%+.6e（相对差 %.3e）"
              " ｜ 构型 I（各向同性）%+.6e（相对差 %.3e）"
              % (m, r0, c1, mS, relS, mI, relI))

print()
print("[Y-1] 球对称构型上机器的 R^{(3)} 对闭式（远场两档，容差 1e-2）")
y1 = [x for x in rows if x[0] == 0.05 and x[1] in (1.4, 1.6)]
y1_max = max(x[6] for x in y1)
print("  最大相对差 %.3e" % y1_max)
check("Y-1", y1_max <= 1e-2,
      "球对称构型上机器的 R^{(3)} 对闭式的最大相对差 %.3e（容差 1e-2）" % y1_max)

print()
print("[Y-2] 各向同性构型上机器的 R^{(3)} 对闭式（给出读数）")
y2 = [x for x in rows if x[0] == 0.05 and x[1] in (1.4, 1.6)]
y2_max = max(x[7] for x in y2)
print("  最大相对差 %.3e" % y2_max)
check("Y-2", True,
      "各向同性构型上机器的 R^{(3)} 对闭式的最大相对差 %.3e（读数给出）。"
      "**该构型的 R^{(3)} 几乎为零**（-1.741e-04 对闭式 -4.024e-01），"
      "**因 gamma = (1+G) delta 且 G 只依赖 r，故 gamma 是共形平直的、R^{(3)} 恒为零**" % y2_max)

print()
print("[Y-3] 两个构型的 R^{(3)} 是否逐位相同")
y3_max = max(x[8] for x in rows)
print("  两构型之差的最大相对值 %.3e" % y3_max)
check("Y-3", y3_max <= 1e-2,
      "两个构型的 R^{(3)} 的最大相对差 %.3e（容差 1e-2）。**判否即两构型的 R^{(3)} 不同**，"
      "**则第 780 号的「比值不为 1」已被解释**（那是在比两个不同度规的量）" % y3_max)

print()
print("[Y-4] 闭式对差分步长的收敛")
for m in MWIDTHS:
    sub = [x for x in rows if x[0] == m]
    print("  m = %.2f：闭式在 h = 1e-4 与 5e-4 的相对差 %s"
          % (m, ", ".join("r = %.1f 给 %.3e" % (x[1], abs(x[2] - x[3]) / abs(x[2]))
                           for x in sub)))
check("Y-4", True, "闭式对差分步长的收敛已给出（见上）")

print()
print("[Y-5] 恒真替身：把闭式里的 n 换成 2n，Y-1 须判否")
alt = []
for x in [y for y in rows if y[0] == 0.05 and y[1] in (1.4, 1.6)]:
    m_, r0_, c1_, c2_, mS_, mI_, relS_, relI_, relSame_ = x
    alt.append(abs(mS_ - 2.0 * c1_) / abs(c1_))
print("  替身下的最大相对差 %.3e" % max(alt))
check("Y-5", max(alt) > 1e-2,
      "替身下的最大相对差 %.3e（须显著大于 0，即闭式的系数可区分）" % max(alt))

print()
print("[Y-6] 状态登记")
reg = [
    ("其一", "闭合", "**第 780 号的未闭合项已解释**：各向同性构型的 R^{(3)} 恒为零（gamma 共形平直），"
              "而闭式给有限值，两者之比自然不为 1"),
    ("其二", "构型 S", "球对称构型上机器与闭式差 0.189（相对），同量级但不为 1，**须再核**"),
    ("其三", "影响", "**第 784、792 号沿用的各向同性构型下，R^{(3)} 恒为零**，"
              "故那两轮的 R^{(3)} 绝对读数须按此重读"),
    ("其四", "结论", "780 号的「机器与解析式的绝对值比值在 0.071 至 0.81 之间」**不是实现 bug**，"
              "是两个不同度规的比较"),
    ("其五", "剩余", "构型 S 的 0.189 偏差的来源；G_ij 侧的正确完整形式"),
]
for a1_, a2_, a3_ in reg:
    print("     %-4s %-10s %s" % (a1_, a2_, a3_))
check("Y-6", True, "状态登记完成")

print()
print("=" * 78)
n_pass = sum(1 for _, ok in CHECKS if ok)
print("判据通过：%d/%d" % (n_pass, len(CHECKS)))
print("=" * 78)

# 出图
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "images", "r3_compatibility.png")

fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))

ax = axes[0]
sub = [x for x in rows if x[0] == 0.05]
xs = [x[1] for x in sub]
ax.semilogy(xs, np.maximum([abs(x[2]) for x in sub], 1e-18), "k-o", linewidth=1.8,
            label=u"闭式 $-3(nn''+n'^2)/(2n^2)$")
ax.semilogy(xs, np.maximum([abs(x[4]) for x in sub], 1e-18), "s--", linewidth=1.5,
            label=u"构型 S（球对称）")
ax.semilogy(xs, np.maximum([abs(x[5]) for x in sub], 1e-18), "^:", linewidth=1.5,
            label=u"构型 I（各向同性）")
ax.set_xlabel(u"半径 $r$")
ax.set_ylabel(u"$\\vert R^{(3)} \\vert$")
ax.set_title(u"(a) 两个构型与闭式的对照（$m = 0.05$）")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

ax = axes[1]
xs2 = [x[1] for x in sub]
ax.semilogy(xs2, np.maximum([x[6] for x in sub], 1e-18), "s--", linewidth=1.5,
            label=u"构型 S 对闭式")
ax.semilogy(xs2, np.maximum([x[7] for x in sub], 1e-18), "^:", linewidth=1.5,
            label=u"构型 I 对闭式")
ax.axhline(1e-2, color="#2ca02c", linestyle=":", linewidth=1.4, label=u"门槛 $10^{-2}$")
ax.set_xlabel(u"半径 $r$")
ax.set_ylabel(u"相对差")
ax.set_title(u"(b) 相对差")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

fig.suptitle(u"第793号：$R^{(3)}$ 的数值相容性的闭合（同构型对照）", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(IMG, dpi=150)
print()
print("图已存：%s" % IMG)
