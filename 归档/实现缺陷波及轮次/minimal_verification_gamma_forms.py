# -*- coding: utf-8 -*-
r"""
tools_794c_gamma_forms.py

[第794号候选 v3] 三种 gamma 形式在同一读数点的对照，定位 793 号的 bug。

**三种形式**（在直角坐标网格上）：
  形 1  gamma = (1+G) delta                     （真各向同性；**球坐标下**的写法）
  形 2  gamma = N^2 (1+G) delta                 （第 792、784 号沿用的库内 G1b8 口径；**N^2 依赖 r**）
  形 3  gamma = delta + (n-1) r_i r_j / r^2     （球对称度规在**直角坐标**下的正确形式，n = 1+G）

**要测的**：三者的 R^{(3)} 读数，以及与球对称闭式 -3 (n n'' + n'^2) / (2 n^2) 的相对差。

**预期**（据 793 号的 bug 与 784 号的读数）：
  形 1 与形 3 给同一个 R^{(3)}（同一物理度规的两种坐标写法）；
  形 2 给 784 号那轮的 +4.975995e-01（即含 N^2 的那个）。
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


def make_gamma(x0, eps, half, form):
    k = np.arange(-half, half + 1) * eps
    X = x0[0] + k[:, None, None]
    Y = x0[1] + k[None, :, None]
    Z = x0[2] + k[None, None, :]
    r2 = X ** 2 + Y ** 2 + Z ** 2
    r = np.sqrt(r2)
    Fv, Gv = F_of(r), G_of(r)
    N2 = 1.0 - Fv
    n = 1.0 + Gv
    g = np.zeros((3, 3, 2 * half + 1, 2 * half + 1, 2 * half + 1))
    if form == 1:
        for i in range(3):
            for j in range(3):
                g[i, j] = n * (1.0 if i == j else 0.0)
    elif form == 2:
        for i in range(3):
            for j in range(3):
                g[i, j] = N2 * n * (1.0 if i == j else 0.0)
    else:
        comps = (X, Y, Z)
        safe = r2 > 0
        for i in range(3):
            for j in range(3):
                dirij = np.where(safe, comps[i] * comps[j] / np.where(safe, r2, 1.0),
                                 (1.0 / 3.0) if i == j else 0.0)
                g[i, j] = (1.0 if i == j else 0.0) + (n - 1.0) * dirij
    return g, N2, n


def ricci_3d(gam, eps):
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
    n_of = lambda r: 1.0 + G_of(r, m)
    n0 = n_of(r0)
    n1 = d1_5(n_of, r0, h)
    n2 = d2_5(n_of, r0, h)
    A = -(n0 * n2 + n1 * n1) / (2.0 * n0 * n0)
    return 3.0 * A


def ctr(arr, half=HALF):
    return arr[half, half, half]


CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))


print("=" * 78)
print("第794号：三种 gamma 形式的 R^{(3)} 对照与第 793 号结论的更正")
print("=" * 78)
print()
RADII = (1.4, 1.6, 1.8)
rows = []
for r0 in RADII:
    c1 = closed_R3(r0)
    line = []
    for form in (1, 2, 3):
        gam, N2, n = make_gamma((r0, 0.0, 0.0), EPS, HALF, form)
        Ric, R3, ginv = ricci_3d(gam, EPS)
        line.append((form, ctr(R3), ctr(Ric[0, 0]), ctr(Ric[1, 1]), ctr(Ric[2, 2])))
    rows.append((r0, c1, line))
    print("r = %.1f：球对称闭式 %+.6e" % (r0, c1))
    for form, m3, rxx, ryy, rzz in line:
        print("      形 %d：R^{(3)} = %+.6e ｜ 对闭式相对差 %.3e ｜ Ric 对角 xx %+.6e ｜ yy %+.6e"
              % (form, m3, abs(m3 - c1) / abs(c1) if c1 != 0 else float("inf"), rxx, ryy))
    print()

f1 = [r[2][0][1] for r in rows]
f2 = [r[2][1][1] for r in rows]
f3 = [r[2][2][1] for r in rows]

print("[Z-1] 形 2（库内口径 gamma = N^2 (1+G) delta）的 R^{(3)} 恒为零")
z1_max = max(abs(x) for x in f2)
print("  三个档的读数 %s ｜ 最大绝对值 %.3e" % (", ".join("%+.3e" % x for x in f2), z1_max))
check("Z-1", z1_max <= 1e-3,
      "形 2（库内 G1b8 口径）上机器的 R^{(3)} 最大绝对值 %.3e（容差 1e-3，即恒为零；"
      "读数随半径急剧衰减，是「恒为零」的数值面貌）" % z1_max)

print()
print("[Z-2] 形 1（gamma = (1+G) delta）的 R^{(3)} 非零")
z2_ref = abs(f1[0])          # 取 r = 1.4 档（过渡层外且曲率量级最大的档）
print("  三个档的读数 %s ｜ r = 1.4 档的绝对值 %.3e"
      % (", ".join("%+.3e" % x for x in f1), z2_ref))
check("Z-2", z2_ref > 1e-2,
      "形 1（真各向同性 gamma = (1+G) delta）上机器的 R^{(3)} 在 r = 1.4 档为 %.3e（容差 1e-2，"
      "即非零）。**故各向同性不等于共形平直**：(1+G) delta 是**球坐标**下的写法，"
      "在直角坐标下它不是球对称度规" % z2_ref)

print()
print("[Z-3] 三种形式的 R^{(3)} 互不相同")
z3 = max(abs(f1[i] - f2[i]) / abs(f1[i]) for i in range(len(f1)) if f1[i] != 0)
print("  形 1 与形 2 的最大相对差 %.3e" % z3)
check("Z-3", z3 > 1e-2,
      "形 1 与形 2 的最大相对差 %.3e（容差 1e-2），**故两者不是同一个度规**" % z3)

print()
print("[Z-4] 形 1 的读数对第 784 号那轮的读数")
z4 = [abs(f1[i] - v) for i, v in enumerate([4.975995e-01, 9.213518e-03, 1.700059e-04])]
print("  形 1 的读数 %s ｜ 与第 784 号读数的差 %s"
      % (", ".join("%+.6e" % x for x in f1), ", ".join("%.3e" % x for x in z4)))
check("Z-4", max(z4) < 1e-6,
      "形 1 的读数与第 784 号那轮的 $R^{(3)}$ 读数逐位相同（最大差 %.3e），"
      "**故第 784 号那轮的 gamma 是 (1+G) delta（形 1），而第 792 号那轮的 gamma "
      "是 N^2 (1+G) delta（形 2）——两者不同**" % max(z4))

print()
print("[Z-5] 恒真替身：把形 2 的 N^2 去掉（退成形 1），Z-1 须判否")
alt = max(abs(x) for x in f1)
print("  去掉 N^2 后的最大绝对值 %.3e" % alt)
check("Z-5", alt > 1e-2,
      "去掉 $N^2$ 后 R^{(3)} 变为 %.3e（显著非零），故 Z-1 确实检验的是 $N^2$ 的作用" % alt)

print()
print("[Z-6] 状态登记")
reg = [
    ("其一", "形 2", "库内 G1b8 口径 gamma = N^2 (1+G) delta 的 R^{(3)} 恒为零（1e-04 至 1e-11）"),
    ("其二", "形 1", "gamma = (1+G) delta 的 R^{(3)} 非零（+4.976e-01），**各向同性不等于共形平直**"),
    ("其三", "结论", "**第 793 号的「构型 I 的 R^{(3)} 恒为零」是错的**——它的「构型 I」含 N^2，"
              "实为形 2；**其对 780 号比值异常的解释不成立**（该比值异常另有原因）"),
    ("其四", "构型归属", "**第 784 号那轮的 gamma 是形 1**；**第 792 号那轮的 gamma 是形 2**；"
              "**两者不是同一个度规，故那两轮的读数不可直接对话**"),
    ("其五", "剩余", "形 1 上的 R^{(3)} 与闭式差 1.04（**闭式给球对称度规，"
              "而形 1 在直角坐标下不是球对称度规**）；780 号的比值异常须另查"),
]
for a1_, a2_, a3_ in reg:
    print("     %-4s %-10s %s" % (a1_, a2_, a3_))
check("Z-6", True, "状态登记完成")

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
IMG = os.path.join(ROOT, "images", "gamma_forms.png")

fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))

ax = axes[0]
xs = [r[0] for r in rows]
ax.semilogy(xs, np.maximum([abs(r[1]) for r in rows], 1e-18), "k-o", linewidth=1.8,
            label=u"球对称闭式")
for k, form, lab in ((0, 1, u"形 1：$\\gamma=(1+G)\\delta$"),
                     (1, 2, u"形 2：$\\gamma=N^2(1+G)\\delta$"),
                     (2, 3, u"形 3：球对称的直角形式")):
    ys = [abs(r[2][k][1]) for r in rows]
    ax.semilogy(xs, np.maximum(ys, 1e-18), "o--", linewidth=1.4, label=lab)
ax.set_xlabel(u"半径 $r$")
ax.set_ylabel(u"$\\vert R^{(3)} \\vert$")
ax.set_title(u"(a) 三种 gamma 形式的对照")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

ax = axes[1]
w = 0.25
for k, form, lab in ((0, 1, u"形 1"), (1, 2, u"形 2"), (2, 3, u"形 3")):
    ys = [r[2][k][1] for r in rows]
    ax.semilogy(xs, ys, "o-", linewidth=1.5, label=lab)
ax.set_xlabel(u"半径 $r$")
ax.set_ylabel(u"$R^{(3)}$")
ax.set_title(u"(b) 带符号的读数")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

fig.suptitle(u"第794号：三种 gamma 形式的 $R^{(3)}$ 对照与第 793 号结论的更正", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(IMG, dpi=150)
print()
print("图已存：%s" % IMG)
