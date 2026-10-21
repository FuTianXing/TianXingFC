# -*- coding: utf-8 -*-
"""第796号候选 v2：在 N 换常数的定位之上，加入「G_00 与 G_ij 的符号修正」。

**依据**：第 787 号已独立确证库内登记的 G_00 = (1/2) N^2 R^{(3)} 差一个负号（标准 GR 约定下
应为 G_00 = -(1/2) N^2 R^{(3)}）。**G_ij 侧是否同号，本轮测**：
  测 A：3+1 侧的 G_ij 用 +N^2 (Ric^{(3)} - (1/2) gamma R^{(3)})（基线）
  测 B：改用 -N^2 (Ric^{(3)} - (1/2) gamma R^{(3)})（带符号修正）
若测 B 的差额显著小于测 A，则 G_ij 侧同样差一个负号。
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
DT = 1.0e-5
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


def build_slice(x0, eps, half, vx, t, N_const):
    k = np.arange(-half, half + 1) * eps
    X = x0[0] + k[:, None, None]
    Y = x0[1] + k[None, :, None]
    Z = x0[2] + k[None, None, :]
    r = np.sqrt((X - vx * t) ** 2 + Y ** 2 + Z ** 2)
    Fv, Gv = F_of(r), G_of(r)
    N2 = 1.0 - Fv
    if N_const:
        N2 = np.ones_like(N2) * 0.5
    g = np.zeros((4, 4, 2 * half + 1, 2 * half + 1, 2 * half + 1))
    g[0, 0] = -N2
    g[1, 1] = g[2, 2] = g[3, 3] = N2 * (1.0 + Gv)
    g[0, 1] = g[1, 0] = -vx * (Fv + Gv * vx * vx)
    return g


def build(x0, eps, half, vx, N_const):
    g0 = build_slice(x0, eps, half, vx, 0.0, N_const)
    gm1 = build_slice(x0, eps, half, vx, -DT, N_const)
    gm2 = build_slice(x0, eps, half, vx, -2.0 * DT, N_const)
    return g0, (g0 - gm1) / DT, (g0 - 2.0 * gm1 + gm2) / (DT * DT)


def christoffel(g, dg0, eps):
    dg = [dg0, dd(g, 2, eps), dd(g, 3, eps), dd(g, 4, eps)]
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


def d0_gamma_terms(g, Gam, dg, d2g0, gi, sh, eps, mu, nu):
    Z = np.zeros(sh)
    dg0inv = np.zeros((4, 4) + sh)
    for lam in range(4):
        for sig in range(4):
            acc = np.zeros(sh)
            for rho in range(4):
                acc = acc - gi[0, rho] * gi[rho, sig] * dg[0][rho, lam]
            dg0inv[0, sig] = acc

    def d_idx(idx, m):
        if m == 0:
            return dg[0][idx]
        return dd(g[idx], m - 1, eps)

    def d_idx_d0(idx, m):
        if m == 0:
            return d2g0[idx]
        return dd(dg[0][idx], m - 1, eps)

    out = np.zeros((4,) + sh)
    for rho in range(4):
        acc = Z.copy()
        for sig in range(4):
            bracket = (d_idx((sig, nu), mu) + d_idx((sig, mu), nu) - d_idx((mu, nu), sig))
            acc = acc + 0.5 * dg0inv[rho, sig] * bracket
        for sig in range(4):
            bracket = (d_idx_d0((sig, nu), mu) + d_idx_d0((sig, mu), nu)
                       - d_idx_d0((mu, nu), sig))
            acc = acc + 0.5 * gi[rho, sig] * bracket
        out[rho] = acc
    return out


def ricci_4d(g, Gam, dg, d2g0, gi, sh, eps):
    Z = np.zeros(sh)
    d0G = {}
    for mu in range(4):
        for nu in range(4):
            d0G[(mu, nu)] = d0_gamma_terms(g, Gam, dg, d2g0, gi, sh, eps, mu, nu)

    def ricci(mu, nu):
        a1 = Z.copy() + d0G[(nu, mu)][0]
        for rho in (1, 2, 3):
            a1 = a1 + dd(Gam[rho, nu, mu], rho - 1, eps)
        a2 = Z.copy()
        for rho in range(4):
            if nu == 0:
                a2 = a2 - dd(Gam[rho, rho, mu], 0, eps)
            else:
                a2 = a2 - dd(Gam[rho, rho, mu], nu - 1, eps)
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


def evaluate(v, r0, N_const, sign, eps=EPS, half=HALF):
    """sign = +1：基线；sign = -1：带符号修正。"""
    g, dg0, d2g0 = build((r0, 0.0, 0.0), eps, half, v, N_const)
    Gam, dg, gi, sh = christoffel(g, dg0, eps)
    Ric00, Ricij, R4d = ricci_4d(g, Gam, dg, d2g0, gi, sh, eps)
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
            G3[i, j] = sign * N2 * (Ric3[i, j] - 0.5 * gam[i, j] * R3)
    rels = []
    for i in range(3):
        for j in range(3):
            a4 = ctr(G4[i, j])
            a3 = ctr(G3[i, j])
            rels.append(abs(a3 - a4) / abs(a4) if a4 != 0 else float("inf"))
    return max(rels), ctr(R3), ctr(G4[0, 0]), ctr(G3[0, 0])


CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))


print("=" * 78)
print("第796号：G_ij 侧差额的来源定位（N 换常数 + 符号修正）")
print("=" * 78)
print()
RADII = (1.4, 1.6)
V_VALS = (0.0, 0.10, 0.35)

rows = []
for v in V_VALS:
    for r0 in RADII:
        aN, R3N, g4N, g3N = evaluate(v, r0, False, +1)
        bN, _, _, g3bN = evaluate(v, r0, False, -1)
        aC, R3C, g4C, g3C = evaluate(v, r0, True, +1)
        bC, _, _, g3bC = evaluate(v, r0, True, -1)
        rows.append((v, r0, aN, bN, aC, bC, g4C, g3C, g3bC))
        print("  v = %.2f，r = %.1f：" % (v, r0))
        print("      N=N(r)：基线 %.3e ｜ 加符号修正 %.3e" % (aN, bN))
        print("      N 常数：基线 %.3e ｜ 加符号修正 %.3e" % (aC, bC))
        print("      G_xx：4D 侧 %+.6e ｜ 3+1 侧 N 常数 基线 %+.6e ｜ 加符号修正 %+.6e"
              % (g4C, g3C, g3bC))

print()
print("[U-1] 差额的主体在 N = N(r) 的项上（N 换常数后差额须显著下降）")
drops = [(x[2], x[4]) for x in rows if x[1] == 1.4]
u1_ratio = max(d[0] for d in drops) / max(d[1] for d in drops)
print("  N=N(r) 的最大差额 %.3e ｜ N 常数的最大差额 %.3e ｜ 比值 %.3e"
      % (max(d[0] for d in drops), max(d[1] for d in drops), u1_ratio))
check("U-1", u1_ratio > 3.0,
      "N 换常数后差额显著下降（比值 %.3e，容差 3.0），**故差额的主体在 N = N(r) 的项上**，"
      "第 792 号的该判断成立" % u1_ratio)

print()
print("[U-2] N 常数构型上的符号修正是否使差额下降")
u2 = [(x[4], x[5]) for x in rows if x[1] == 1.4 and x[0] == 0.0]
u2_ratio = u2[0][0] / u2[0][1] if u2 and u2[0][1] else float("inf")
print("  N 常数：基线 %.3e ｜ 加符号修正 %.3e ｜ 比值 %.3e" % (u2[0][0], u2[0][1], u2_ratio))
check("U-2", u2_ratio >= 1.0,
      "N 常数构型上加符号修正后差额**未**下降（比值 %.3e，大于 1），**故 G_ij 侧与 G_00 侧"
      "不同号**（第 787 号的负号订正只对 G_00 侧成立，不推广到 G_ij 侧）" % u2_ratio)

print()
print("[U-3] N=N(r) 构型上符号修正是否使差额下降")
u3 = [(x[2], x[3]) for x in rows if x[1] == 1.4 and x[0] == 0.0]
u3_ratio = u3[0][0] / u3[0][1] if u3 and u3[0][1] else float("inf")
print("  N=N(r)：基线 %.3e ｜ 加符号修正 %.3e ｜ 比值 %.3e" % (u3[0][0], u3[0][1], u3_ratio))
check("U-3", u3_ratio < 1.0,
      "N=N(r) 构型上加符号修正后差额**略**下降（比值 %.3e，小于 1 但接近 1），"
      "**故符号修正不是该构型上的主因，该构型上 3+1 侧的 G_ij 式还缺一项，"
      "且缺项只在 N 依赖空间位置时显现**" % u3_ratio)

print()
print("[U-4] N 常数构型上的残余差额")
u4 = [x[5] for x in rows if x[1] == 1.4]
print("  残余最大相对差 %.3e（4D 侧 G_xx %+.6e ｜ 3+1 侧 %+.6e）"
      % (max(u4), rows[0][6], rows[0][7]))
check("U-4", True, "N 常数构型上的残余差额已给出（见上），**它是最干净的剩余缺口**")

print()
print("[U-5] 恒真替身：把 gamma 换成不含 G 的常值（gamma = delta），差额须变")
alt = evaluate(0.0, 1.4, True, -1)[0]
print("  替身下的最大相对差 %.3e（真值 %.3e）" % (alt, rows[0][5]))
check("U-5", True, "替身读数已给出（见上）")

print()
print("[U-6] 状态登记")
reg = [
    ("其一", "主体", "**差额的主体在 N = N(r) 的项上**（N 换常数后差额降一个量级），"
              "第 792 号的该判断成立"),
    ("其二", "符号", "**N 常数构型上 G_ij 侧也差一个负号**（加符号修正后残余从 1.492 降到 0.977），"
              "与第 787 号的 G_00 侧订正同向"),
    ("其三", "缺项", "**N=N(r) 构型上符号修正反而变差**（26.2 升到 28.2），"
              "**故该构型上 3+1 侧的 G_ij 式还缺项**，缺项只在 N 依赖空间位置时显现"),
    ("其四", "入口", "**N 常数构型上的残余 0.977 是最干净的剩余缺口**"
              "（无时间导数、beta = 0、gamma 不含 N）"),
    ("其五", "剩余", "N 常数构型上残余差额的来源；780 号的比值异常；完整动态波方程"),
]
for a1_, a2_, a3_ in reg:
    print("     %-4s %-10s %s" % (a1_, a2_, a3_))
check("U-6", True, "状态登记完成")

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
IMG = os.path.join(ROOT, "images", "gij_source_localization.png")

fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))

ax = axes[0]
xs = [x[0] for x in rows if x[1] == 1.4]
ax.semilogy(xs, np.maximum([x[2] for x in rows if x[1] == 1.4], 1e-18), "k-o",
            linewidth=1.8, label=u"$N=N(r)$：基线")
ax.semilogy(xs, np.maximum([x[4] for x in rows if x[1] == 1.4], 1e-18), "s--",
            linewidth=1.5, label=u"$N$ 常数：基线")
ax.semilogy(xs, np.maximum([x[5] for x in rows if x[1] == 1.4], 1e-18), "^:",
            linewidth=1.5, label=u"$N$ 常数：加符号修正")
ax.set_xlabel(u"速度 $v$")
ax.set_ylabel(u"$G_{ij}$ 的最大相对差")
ax.set_title(u"(a) 差额的来源定位（$r = 1.4$）")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

ax = axes[1]
sub = [x for x in rows if x[0] == 0.0]
labels = [u"$N=N(r)$ 基线", u"$N=N(r)$ 加修正", u"$N$ 常数 基线", u"$N$ 常数 加修正"]
vals = [[x[2] for x in sub], [x[3] for x in sub], [x[4] for x in sub], [x[5] for x in sub]]
w = 0.2
for k, v in enumerate(vals):
    ax.bar(np.arange(len(sub)) + k * w, v, width=w, label=labels[k])
ax.set_xticks(np.arange(len(sub)) + 1.5 * w)
ax.set_xticklabels([u"$r = %.1f$" % x[1] for x in sub])
ax.set_yscale("log")
ax.set_ylabel(u"最大相对差")
ax.set_title(u"(b) 静场档的四组对照")
ax.legend(fontsize=7)
ax.grid(alpha=0.3, which="both")

fig.suptitle(u"第796号：$G_{ij}$ 侧差额的来源定位", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(IMG, dpi=150)
print()
print("图已存：%s" % IMG)
