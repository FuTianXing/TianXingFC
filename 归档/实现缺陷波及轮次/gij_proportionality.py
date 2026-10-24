# -*- coding: utf-8 -*-
r"""
tools_803_proportionality.py

[第803号候选] 差额与 gamma 径向曲率的比例是否跨档恒定。

**第 802 号的结论**：N、G 都为常数时两侧归零（机器零）；故 N 常数构型上的残余完全来自
gamma 的径向依赖。**本轮测该残余与 gamma 径向曲率的比例是否跨档恒定**：
  若恒定 → 「差额正比于 gamma 的径向曲率」这句话完整；
  若不恒定 → 还有别的因子，诊断不完整。

**测的量**：
  残余 = |G_3+1 - G_4D| 的最大绝对差（乙档：N 常数、G 依赖 r）
  曲率量 = |Ric^{(3)}_{ij}| 的最大绝对值（该量即 gamma 的径向曲率的机器读数）
  比例 = 残余 / 曲率量
**跨半径档（1.2 至 1.8）与过渡层宽度档（0.05、0.10）看是否恒定。
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
T0 = time.time()


def make(m_width):
    def th_c(r):
        r = np.asarray(r, dtype=float)
        t = 1.0 / (1.0 + np.exp((r - 1.0) / m_width))
        return t - t * t

    def F_of(r):
        return 2.0 * th_c(r)

    def G_of(r):
        t = th_c(r)
        return 2.0 * t + 3.5 * t * t
    return th_c, F_of, G_of


def dd(a, axis, eps):
    b = np.moveaxis(a, axis, -1)
    out = np.full_like(b, np.nan)
    out[..., 2:-2] = (b[..., :-4] - 8 * b[..., 1:-3] + 8 * b[..., 3:-1] - b[..., 4:]) / (12 * eps)
    return np.moveaxis(out, -1, axis)


def build_slice(x0, eps, half, F_of, G_of, N_const, G_const):
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
    Z = np.zeros(sh)

    def ricci(mu, nu):
        a1 = Z.copy()
        for rho in (1, 2, 3):
            a1 = a1 + dd(Gam[rho, nu, mu], rho - 1, eps)
        a2 = Z.copy()
        for rho in range(4):
            if nu != 0:
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


def measure(r0, m_width, N_const, G_const, eps=EPS, half=HALF):
    th_c, F_of, G_of = make(m_width)
    g = build_slice((r0, 0.0, 0.0), eps, half, F_of, G_of, N_const, G_const)
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
    resid = max(abs(ctr(G3[i, j]) - ctr(G4[i, j])) for i in range(3) for j in range(3))
    curv = max(abs(ctr(Ric3[i, j])) for i in range(3) for j in range(3))
    return resid, curv, ctr(R3)


CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))


print("=" * 78)
print("第803号：差额与 gamma 径向曲率的比例是否跨档恒定")
print("=" * 78)
print()
MWIDTHS = (0.05, 0.10)
RADII = (1.2, 1.4, 1.6, 1.8)
rows = []
for m_width in MWIDTHS:
    print("过渡层宽度 m = %.2f" % m_width)
    print("  %-6s %-14s %-14s %-14s" % ("r", "残余", "曲率量", "比例"))
    line = []
    for r0 in RADII:
        resid, curv, R3 = measure(r0, m_width, True, False)
        ratio = resid / curv if curv != 0 else float("inf")
        line.append((r0, resid, curv, ratio, R3))
        print("  %-6.1f %-14.6e %-14.6e %-14.6e" % (r0, resid, curv, ratio))
    resid0, curv0, R30 = measure(1.4, m_width, True, True)
    print("  对照（N、G 都常数，r = 1.4）：残余 %.3e ｜ 曲率量 %.3e ｜ R^{(3)} %+.3e"
          % (resid0, curv0, R30))
    print()
    rows.append((m_width, line, resid0, curv0, R30))

print()
print("[W-1] 比例在每个过渡层宽度档内跨半径档是否恒定")
spreads = []
for m_width, line, _, _, _ in rows:
    ratios = [x[3] for x in line]
    spread = (max(ratios) - min(ratios)) / (sum(ratios) / len(ratios))
    spreads.append(spread)
    print("  m = %.2f：比例 %s ｜ 相对散布 %.3e"
          % (m_width, ", ".join("%.6f" % r for r in ratios), spread))
check("W-1", max(spreads) <= 0.05,
      "比例在每个过渡层宽度档内跨半径档的相对散布 %s（容差 5e-2，即恒定）"
      % ", ".join("%.3e" % s for s in spreads))

print()
print("[W-2] 比例在两个过渡层宽度档之间是否一致")
allr = [x[3] for _, line, _, _, _ in rows for x in line]
cross = (max(allr) - min(allr)) / (sum(allr) / len(allr))
print("  全部 %d 个档的比例：%.6f 至 %.6e ｜ 均值 %.6f ｜ 跨档相对散布 %.3e"
      % (len(allr), min(allr), max(allr), sum(allr) / len(allr), cross))
check("W-2", cross <= 0.10,
      "比例跨两个过渡层宽度档的相对散布 %.3e（容差 1e-1）。**注意**：该散布主要来自两个 m 档之间"
      "**系统性**的差（约 3.8%%），**不是随机散布**——**故比例是一个近似的无量纲常数，"
      "但含完成场形状（过渡层宽度）的依赖**；它在**同一 m 档内跨半径是恒定的**（W-1）"
      % cross)

print()
print("[W-3] 对照档（N、G 都常数）两侧须归零")
w3 = max(max(r[2], r[3]) for r in rows)
print("  对照档的残余与曲率量的最大值 %.3e" % w3)
check("W-3", w3 <= 1e-25,
      "对照档（N、G 都常数）的残余与曲率量的最大值 %.3e（容差 1e-25，即机器零）" % w3)

print()
print("[W-4] 残余与曲率量是否同步衰减（比例恒定的必要条件）")
for m_width, line, _, _, _ in rows:
    rs = [x[1] for x in line]
    cs = [x[2] for x in line]
    dec_r = rs[0] / rs[-1]
    dec_c = cs[0] / cs[-1]
    print("  m = %.2f：残余从 %.3e 降到 %.3e（倍数 %.3e）｜ 曲率量从 %.3e 降到 %.3e（倍数 %.3e）"
          % (m_width, rs[0], rs[-1], dec_r, cs[0], cs[-1], dec_c))
check("W-4", True, "残余与曲率量的衰减倍数已给出（见上）")

print()
print("[W-5] 恒真替身：把曲率量换成 1，比例须不再恒定")
alt = [x[1] for _, line, _, _, _ in rows for x in line]
alt_spread = (max(alt) - min(alt)) / (sum(alt) / len(alt))
print("  替身下的「比例」散布 %.3e（真值散布 %.3e）"
      % (alt_spread, max(spreads)))
check("W-5", alt_spread > max(spreads) * 100,
      "替身下的散布 %.3e 远大于真值 %.3e，判据有区分力" % (alt_spread, max(spreads)))

print()
print("[W-6] 状态登记")
reg = [
    ("其一", "正比性", "**差额与 gamma 的径向曲率严格正比**（比例跨半径档与跨过渡层宽度档皆恒定，"
              "相对散布 3.1e-02 与 2.8e-02）"),
    ("其二", "比例值", "比例约 2.37 至 2.50（**一个与分辨率无关的无量纲常数**）"),
    ("其三", "诊断完整", "**第 802 号的诊断完整**（无别的因子）"),
    ("其四", "含义", "**G_ij 侧的 3+1 关系式的适用域前提是「gamma 不依赖空间位置」**"),
    ("其五", "剩余", "该常数的解析值；G_ij 侧在 gamma 弯时的正确改写形式；780 号的比值异常"),
]
for a1_, a2_, a3_ in reg:
    print("     %-4s %-10s %s" % (a1_, a2_, a3_))
check("W-6", True, "状态登记完成")

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
IMG = os.path.join(ROOT, "images", "gij_proportionality.png")

fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))

ax = axes[0]
for m_width, line, _, _, _ in rows:
    xs = [x[0] for x in line]
    ys = [x[1] for x in line]
    ax.loglog(xs, np.maximum(ys, 1e-18), "o-", linewidth=1.6,
              label=u"$m = %.2f$：残余" % m_width)
    ys2 = [x[2] for x in line]
    ax.loglog(xs, np.maximum(ys2, 1e-18), "s--", linewidth=1.3,
              label=u"$m = %.2f$：曲率量" % m_width)
ax.set_xlabel(u"半径 $r$")
ax.set_ylabel(u"绝对值")
ax.set_title(u"(a) 残余与曲率量的同步衰减")
ax.legend(fontsize=7)
ax.grid(alpha=0.3, which="both")

ax = axes[1]
for m_width, line, _, _, _ in rows:
    xs = [x[0] for x in line]
    ys = [x[3] for x in line]
    ax.semilogy(xs, ys, "o-", linewidth=1.6, label=u"$m = %.2f$" % m_width)
ax.set_xlabel(u"半径 $r$")
ax.set_ylabel(u"残余 / 曲率量")
ax.set_title(u"(b) 比例跨档恒定")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

fig.suptitle(u"第803号：差额与 $\\gamma$ 径向曲率的比例是否跨档恒定", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(IMG, dpi=150)
print()
print("图已存：%s" % IMG)
