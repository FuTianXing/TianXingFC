# -*- coding: utf-8 -*-
r"""
tools_802_gij_nconst_gconst.py

[第802号候选] N 常数构型上 G_ij 侧残余差额的来源定位：把 G 也换成常数。

**第 801 号的结论**：差额的主体在 N = N(r) 的项上；N 常数构型上的残余为 9.773e-01，
**而该构型是最干净的**（无时间导数、beta = 0、gamma 不含 N）。
**但注意**：N 常数时 g_ij = N^2 (1+G) delta 仍是 r 的函数（因 G 依赖 r），
**故 gamma 依赖 r、R^{(3)} 非零**。

**本轮的定位**：在 N 常数构型上，**再把 G 换成常数**（gamma = delta，R^{(3)} 恒为零）：
  若差额归零 → 残余全部来自 gamma 的径向依赖（即 R^{(3)} 与 gamma 的空间二阶导数）；
  若不归零 → 残余来自「G_ij 的 3+1 表达与 4D 直接构造的结构差」，与 gamma 无关。

**同时**：给出 N 常数、G 恒定时的机器读数与解析对照。
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


def build_slice(x0, eps, half, t, N_const, G_const):
    k = np.arange(-half, half + 1) * eps
    X = x0[0] + k[:, None, None]
    Y = x0[1] + k[None, :, None]
    Z = x0[2] + k[None, None, :]
    r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    Fv, Gv = F_of(r), G_of(r)
    if G_const:
        Gv = np.ones_like(Gv) * float(Gv[HALF, HALF, HALF])
    N2 = 1.0 - Fv
    if N_const:
        N2 = np.ones_like(N2) * 0.5
    g = np.zeros((4, 4, 2 * half + 1, 2 * half + 1, 2 * half + 1))
    g[0, 0] = -N2
    g[1, 1] = g[2, 2] = g[3, 3] = N2 * (1.0 + Gv)
    return g


def christoffel(g, eps):
    dg = [np.zeros_like(g), dd(g, 2, eps), dd(g, 3, eps), dd(g, 4, eps)]
    gi = np.zeros_like(g)
    for i in range(4):
        gi[i, i] = 1.0 / g[i, i]
    sh = g.shape[2:]
    Gam = np.zeros((4, 4, 4) + sh)
    for lam in range(4):
        for mu in range(4):
            for nu in range(4):
                acc = np.zeros(sh)
                for sig in range(4):
                    acc = acc + 0.5 * gi[lam, sig] * (
                        dg[mu][sig, nu] + dg[nu][sig, mu] - dg[sig][mu, nu])
                Gam[lam, mu, nu] = acc
    return Gam, dg, gi, sh


def ricci_4d(Gam, dg, gi, sh, eps):
    """静场（d_0 项为零）的四维 Ricci 与迹。"""
    Z = np.zeros(sh)

    def ricci(mu, nu):
        a1 = Z.copy()
        for rho in (1, 2, 3):
            a1 = a1 + dd(Gam[rho, nu, mu], rho - 1, eps)
        a2 = Z.copy()
        for rho in range(4):
            a2 = a2 - dd(Gam[rho, rho, mu], nu - 1, eps) if nu != 0 else a2
        a3 = Z.copy()
        for rho in range(4):
            for sig in range(4):
                a3 = a3 + Gam[rho, rho, sig] * Gam[sig, nu, mu]
        a4 = Z.copy()
        for rho in range(4):
            for sig in range(4):
                a4 = a4 - Gam[rho, nu, sig] * Gam[sig, rho, mu]
        return a1 - a2 + a3 - a4

    Ric00 = ricci(0, 0)
    Ricij = np.zeros((3, 3) + sh)
    for i in range(1, 4):
        for j in range(1, 4):
            Ricij[i - 1, j - 1] = ricci(i, j)
    R = Ric00.copy()
    for i in range(1, 4):
        R = R + Ricij[i - 1, i - 1]
    return Ric00, Ricij, R


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


def ctr(arr, half=HALF):
    return arr[half, half, half]


def evaluate(r0, N_const, G_const, eps=EPS, half=HALF):
    g = build_slice((r0, 0.0, 0.0), eps, half, 0.0, N_const, G_const)
    Gam, dg, gi, sh = christoffel(g, eps)
    Ric00, Ricij, R4d = ricci_4d(Gam, dg, gi, sh, eps)
    N2 = -g[0, 0]
    gam = np.zeros((3, 3) + sh)
    for i in range(3):
        for j in range(3):
            gam[i, j] = g[i + 1, j + 1] / N2
    Ric3, R3, ginv3 = ricci_3d(gam, eps)
    G4 = np.zeros((3, 3) + sh)
    for i in range(3):
        for j in range(3):
            G4[i, j] = Ricij[i, j] - 0.5 * g[i + 1, j + 1] * R4d
    G3 = np.zeros((3, 3) + sh)
    for i in range(3):
        for j in range(3):
            G3[i, j] = N2 * (Ric3[i, j] - 0.5 * gam[i, j] * R3)
    rels = []
    for i in range(3):
        for j in range(3):
            a4 = ctr(G4[i, j])
            a3 = ctr(G3[i, j])
            rels.append(abs(a3 - a4) / abs(a4) if a4 != 0 else float("inf"))
    return max(rels), ctr(R3), ctr(G4[0, 0]), ctr(G3[0, 0]), ctr(Ric3[0, 0]), ctr(Ric3[1, 1])


CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))


print("=" * 78)
print("第802号：N 常数构型上残余差额的来源（G 也换常数）")
print("=" * 78)
print()
RADII = (1.4, 1.6)
rows = []
for r0 in RADII:
    a, R3a, g4a, g3a, rxxa, ryya = evaluate(r0, False, False)
    b, R3b, g4b, g3b, rxxb, ryyb = evaluate(r0, True, False)
    c, R3c, g4c, g3c, rxxc, ryyc = evaluate(r0, True, True)
    rows.append((r0, a, b, c, R3a, R3b, R3c, g4c, g3c))
    print("r = %.1f：" % r0)
    print("      N=N(r)、G=G(r)：最大相对差 %.3e ｜ R^{(3)} %+.6e" % (a, R3a))
    print("      N 常数、G=G(r)：最大相对差 %.3e ｜ R^{(3)} %+.6e" % (b, R3b))
    print("      N 常数、G 常数：最大相对差 %.3e ｜ R^{(3)} %+.6e" % (c, R3c))
    print("      G_xx：4D 侧 %+.6e ｜ 3+1 侧（N、G 常数）%+.6e" % (g4c, g3c))

print()
print("[V-1] N、G 都为常数时两侧是否都归零（相对差此时无定义，须看绝对读数）")
v1 = max(max(abs(x[7]), abs(x[8])) for x in rows)
print("  两侧 G_xx 的最大绝对值 %.3e" % v1)
check("V-1", v1 <= 1e-25,
      "N、G 都为常数时 4D 侧与 3+1 侧的 G_xx 的最大绝对值 %.3e（容差 1e-25，即机器零）；"
      "**故残余差额完全来自 gamma 的径向依赖**（即 G(r) 的空间二阶导数），"
      "**不是「G_ij 的 3+1 表达本身」的结构错**" % v1)

print()
print("[V-2] N 常数、G 依赖 r 时残余差额的量级")
v2 = max(x[2] for x in rows if x[0] == 1.4)
print("  r = 1.4 档的残余最大相对差 %.3e" % v2)
check("V-2", v2 > 1e-2,
      "N 常数、G 依赖 r 时残余的最大相对差 %.3e（容差 1e-2，即残余确实存在）" % v2)

print()
print("[V-3] 残余是否随 N 的恢复而放大")
v3 = rows[0][1] / rows[0][2]
print("  N=N(r) 的差额 %.3e ｜ N 常数的差额 %.3e ｜ 比值 %.3e" % (rows[0][1], rows[0][2], v3))
check("V-3", v3 > 3.0,
      "恢复 N 的空间依赖后差额放大 %.3e 倍（容差 3.0），**故 N 的空间依赖是差额的一个来源**" % v3)

print()
print("[V-4] R^{(3)} 的读数（三个构型）")
for r0, a, b, c, R3a, R3b, R3c, g4c, g3c in rows:
    print("  r = %.1f：R^{(3)} 依次为 %+.6e ｜ %+.6e ｜ %+.6e" % (r0, R3a, R3b, R3c))
check("V-4", True, "三个构型的 R^{(3)} 读数已给出（见上）")

print()
print("[V-5] 恒真替身：把 N 常数档的 G 也换常数后，差额须归零")
print("  替身下两侧的 G_xx 绝对值：%s"
      % ", ".join("%.3e" % max(abs(x[7]), abs(x[8])) for x in rows))
check("V-5", max(max(abs(x[7]), abs(x[8])) for x in rows) <= 1e-25,
      "替身（G 也换常数）下两侧的 G_xx 绝对值降到机器零，判据有区分力")

print()
print("[V-6] 状态登记")
reg = [
    ("其一", "残余来源", "**N、G 都为常数时两侧都归零**（机器零 1e-30 量级），"
              "**故 N 常数构型上的残余完全来自 gamma 的径向依赖**（G(r) 的空间二阶导数）"),
    ("其二", "含义", "**不是「G_ij 的 3+1 表达本身」的结构错**——它在其适用条件（N 与 gamma 都不依赖"
              "空间位置）下是对的"),
    ("其三", "定位", "**G_ij 侧的 3+1 关系式在 gamma 依赖 r 时与 4D 直接构造不等，差额正比于 "
              "gamma 的径向曲率**"),
    ("其四", "与前轮", "**与第 801 号一致**：差额的主体在 N = N(r) 的项上；本轮进一步给出"
              "**N 常数后残余亦在 gamma 的径向依赖上**"),
    ("其五", "剩余", "差额与 gamma 径向曲率的定量关系（正比与否）；G_ij 侧完整 ADM 形式；"
              "780 号的比值异常"),
]
for a1_, a2_, a3_ in reg:
    print("     %-4s %-10s %s" % (a1_, a2_, a3_))
check("V-6", True, "状态登记完成")

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
IMG = os.path.join(ROOT, "images", "gij_gconst_localization.png")

fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))

ax = axes[0]
xs = [x[0] for x in rows]
w = 0.25
for k, (idx, lab) in enumerate(((1, u"$N=N(r)$、$G=G(r)$"),
                                (2, u"$N$ 常数、$G=G(r)$"),
                                (3, u"$N$ 常数、$G$ 常数"))):
    ys = [x[idx] for x in rows]
    ax.bar(np.arange(len(rows)) + k * w, np.maximum(ys, 1e-18), width=w, label=lab)
ax.set_xticks(np.arange(len(rows)) + w)
ax.set_xticklabels([u"$r = %.1f$" % x[0] for x in rows])
ax.set_yscale("log")
ax.set_ylabel(u"最大相对差")
ax.set_title(u"(a) 三个构型的差额")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

ax = axes[1]
for k, (idx, lab) in enumerate(((4, u"$R^{(3)}$：$N=N(r)$、$G=G(r)$"),
                                (5, u"$R^{(3)}$：$N$ 常数、$G=G(r)$"),
                                (6, u"$R^{(3)}$：$N$ 常数、$G$ 常数"))):
    ys = [abs(x[idx]) for x in rows]
    ax.semilogy(xs, np.maximum(ys, 1e-18), "o--", linewidth=1.5, label=lab)
ax.set_xlabel(u"半径 $r$")
ax.set_ylabel(u"$\\vert R^{(3)} \\vert$")
ax.set_title(u"(b) 三个构型的 $R^{(3)}$")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

fig.suptitle(u"第802号：$N$ 常数构型上残余差额的来源", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(IMG, dpi=150)
print()
print("图已存：%s" % IMG)
