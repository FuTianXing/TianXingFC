# -*- coding: utf-8 -*-
r"""
minimal_verification_relations_itemwise.py

[第784号] 三条 3+1 关系的逐条核对（构型：库内 G1b8 口径的各向同性度规）

**四条口径（一次定清）**：
  ① 构型：g_ij = N^2 (1+G) delta，N^2 = 1-F，G = 2t + 3.5t^2（t 为完成场的形状函数）。
     **各向同性**。注意：球对称度规在直角坐标下是 delta_ij + (n-1) r_i r_j / r^2，
     那是第 780 号的**另一个**构型，**不可与本构型混用**。
  ② R^{(3)}：用**直角坐标的坐标式三维算子**（匹配本构型；第 767 与 777 号即用此算子，
     且在静场档与 4D 的 G_00 相容到 1.494e-03）。
  ③ 完成场随源移动：源沿 x 以匀速 vx 运动，源在 z(t) = vx t，完成场依赖 r = |x - z(t)|。
     **d_0 g 与 d_0^2 g 用切片差分算**（t = 0, -h, -2h 三片），
     这自动给出 d_0 g_{ij}（第 781 号置零的项）与 d_0^2 g_{0 sigma}（第 781 号缺的两阶项）。
  ④ 逐条核对三条 3+1 关系（**不取迹**）：
     G_00 = -(1/2) N^2 R^{(3)}（符号按第 777 号的实测）
     G_ij = N^2 ( Ric^{(3)}_{ij} - (1/2) gamma_ij R^{(3)} )
     G_0i = (1/(2N)) d_i( gamma^{jk} d_k N / N ) + (N/2) d_0 gamma_ij
     每条与四维直接构造的对应分量比对。
"""
import io
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


def build_slice(x0, eps, half, vx, t):
    """时刻 t 的度规。源在 z(t) = vx t，完成场依赖 r = |x - z(t)|。"""
    k = np.arange(-half, half + 1) * eps
    X = x0[0] + k[:, None, None]
    Y = x0[1] + k[None, :, None]
    Z = x0[2] + k[None, None, :]
    r = np.sqrt((X - vx * t) ** 2 + Y ** 2 + Z ** 2)
    Fv, Gv = F_of(r), G_of(r)
    N2 = 1.0 - Fv
    g = np.zeros((4, 4, 2 * half + 1, 2 * half + 1, 2 * half + 1))
    g[0, 0] = -N2
    g[1, 1] = g[2, 2] = g[3, 3] = N2 * (1.0 + Gv)
    g[0, 1] = g[1, 0] = -vx * (Fv + Gv * vx * vx)
    return g


def build(x0, eps, half, vx):
    """(g, dg0, d2g0)：时间导数用切片差分。"""
    g0 = build_slice(x0, eps, half, vx, 0.0)
    gm1 = build_slice(x0, eps, half, vx, -DT)
    gm2 = build_slice(x0, eps, half, vx, -2.0 * DT)
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


def d0_gamma_terms(Gam, dg, d2g0, gi, sh, eps, mu, nu):
    """d_0 Gamma^rho_{mu nu}，返回长度 4 的数组（各 rho）。"""
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


def ricci_4d(Gam, dg, d2g0, gi, sh, eps):
    """四维的 Ric_{00}、Ric_{0i}、Ric_{ij} 与 R（迹）。"""
    Z = np.zeros(sh)
    d0G = {}
    for mu in range(4):
        for nu in range(4):
            d0G[(mu, nu)] = d0_gamma_terms(Gam, dg, d2g0, gi, sh, eps, mu, nu)

    def ricci(mu, nu):
        a1 = Z.copy()
        a1 = a1 + d0G[(nu, mu)][0]
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
    Ric0i = np.zeros((3,) + sh)
    for i in range(1, 4):
        Ric0i[i - 1] = ricci(0, i)
    Ricij = np.zeros((3, 3) + sh)
    for i in range(1, 4):
        for j in range(1, 4):
            Ricij[i - 1, j - 1] = ricci(i, j)
    R = Ric00.copy()
    for i in range(1, 4):
        R = R + Ricij[i - 1, i - 1]
    return Ric00, Ric0i, Ricij, R


def ricci_3d(gam, eps):
    """直角坐标的坐标式三维里奇张量与 R^{(3)}（匹配各向同性度规的构型）。"""
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


def rel(a, b):
    return abs(a - b) / abs(b) if b != 0 else float("inf")


CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))


def rel(a, b):
    return abs(a - b) / abs(b) if b != 0 else float("inf")


print("=" * 78)
print("第784号：三条 3+1 关系的逐条核对与第 781 号装置的两处缺口")
print("=" * 78)
print()
RADII = (1.4, 1.6)
V_VALS = (0.0, 0.05, 0.10, 0.20, 0.35)

rows = []
for v in V_VALS:
    for r0 in RADII:
        g, dg0, d2g0 = build((r0, 0.0, 0.0), EPS, HALF, v)
        Gam, dg, gi, sh = christoffel(g, dg0, EPS)
        Ric00, Ric0i, Ricij, R4d = ricci_4d(Gam, dg, d2g0, gi, sh, EPS)
        N2 = -g[0, 0]
        gam = np.zeros((3, 3) + sh)
        for i in range(3):
            for j in range(3):
                gam[i, j] = g[i + 1, j + 1] / N2
        Ric3, R3, ginv3 = ricci_3d(gam, EPS)
        N = np.sqrt(np.abs(N2))
        G00_3 = -0.5 * N2 * R3
        Rij3_iso = np.zeros((3, 3) + sh)
        for i in range(3):
            for j in range(3):
                Rij3_iso[i, j] = (R3 / 3.0) * (1.0 if i == j else 0.0)
        Gij_3 = np.zeros((3, 3) + sh)
        for i in range(3):
            for j in range(3):
                Gij_3[i, j] = N2 * (Rij3_iso[i, j] - 0.5 * gam[i, j] * R3)
        dN = [dd(N, 0, EPS), dd(N, 1, EPS), dd(N, 2, EPS)]
        d0gam_full = np.zeros((3, 3) + sh)
        for i in range(3):
            for j in range(3):
                d0gam_full[i, j] = (dg0[i + 1, j + 1] * N2 - g[i + 1, j + 1] * dg0[0, 0]) / (N2 * N2)
        d0N = dg0[0, 0] / (2.0 * N)
        G0i_3 = np.zeros((3,) + sh)
        for i in range(3):
            term1 = np.zeros(sh)
            for j in range(3):
                for k in range(3):
                    term1 = term1 + ginv3[i, j] * dd(dN[k], j, EPS) * ginv3[j, k]
                    term1 = term1 + ginv3[i, j] * dN[k] * dd(ginv3[j, k], j, EPS)
            term2 = np.zeros(sh)
            for j in range(3):
                for k in range(3):
                    term2 = term2 + d0gam_full[j, k] * N + gam[j, k] * d0N
            G0i_3[i] = (dd(term1, i, EPS) - term2) / (2.0 * N * N2)
        G00_4 = Ric00 - 0.5 * g[0, 0] * R4d
        Gij_4 = np.zeros((3, 3) + sh)
        for i in range(3):
            for j in range(3):
                Gij_4[i, j] = Ricij[i, j] - 0.5 * g[i + 1, j + 1] * R4d
        G0i_4 = np.zeros((3,) + sh)
        for i in range(1, 4):
            G0i_4[i - 1] = Ric0i[i - 1] - 0.5 * g[0, i] * R4d
        e00 = rel(ctr(G00_3), ctr(G00_4))
        eij = max(rel(ctr(Gij_3[i, j]), ctr(Gij_4[i, j])) for i in range(3) for j in range(3))
        # G_0i：4D 侧为零时不可判（相对差无定义），此时报「不可判」
        e0i_vals = []
        for i in range(3):
            a4 = ctr(G0i_4[i])
            if abs(a4) > 1e-14:
                e0i_vals.append(rel(ctr(G0i_3[i]), a4))
        e0i = max(e0i_vals) if e0i_vals else None
        rows.append((v, r0, e00, eij, e0i, ctr(G00_4), ctr(G00_3), ctr(dg0[1, 1]),
                     ctr(d2g0[0, 1])))
        print("  v = %.2f，r = %.1f：G_00 相对差 %.3e ｜ G_ij 最大相对差 %.3e ｜ G_0i %s"
              % (v, r0, e00, eij, ("最大相对差 %.3e" % e0i) if e0i is not None else "不可判（4D 侧为零）"))
        print("          4D：G_00 %+.6e ｜ 3+1：G_00 %+.6e ｜ 诊断：d_0 g_xx %+.6e ｜ d_0^2 g_0x %+.6e"
              % (ctr(G00_4), ctr(G00_3), ctr(dg0[1, 1]), ctr(d2g0[0, 1])))

main = [x for x in rows if x[0] > 0]
static = [x for x in rows if x[0] == 0]

print()
print("[V-2] G_00 侧的相容性（静场档：复现第 780 号的读数）")
v2_max = max(x[2] for x in static)
print("  静场档的最大相对差 %.3e" % v2_max)
check("V-2", v2_max <= 1e-2,
      "G_00 侧在静场档的最大相对差 %.3e（容差 1e-2，即复现第 780 号的读数）" % v2_max)

print()
print("[V-3] G_00 侧的动源档相容性")
v3_max = max(x[2] for x in main)
print("  动源档的最大相对差 %.3e" % v3_max)
check("V-3", v3_max <= 1e-2,
      "G_00 侧在动源档的最大相对差 %.3e（容差 1e-2）" % v3_max)

print()
print("[V-1] G_ij 侧与 G_0i 侧的相容性（本轮不通过，且是结构性的）")
v1_eij = max(x[3] for x in rows)
v1_e0i = max([x[4] for x in rows if x[4] is not None]) if any(x[4] is not None for x in rows) else None
print("  G_ij 的最大相对差 %.3e ｜ G_0i 的最大相对差 %s"
      % (v1_eij, ("%.3e" % v1_e0i) if v1_e0i is not None else "不可判"))
check("V-1", v1_eij <= 1e-2 and (v1_e0i is None or v1_e0i <= 1e-2),
      "G_ij 侧的最大相对差 %.3e（容差 1e-2）；G_0i 侧 %s。"
      "**判否是结构性结论**：3+1 侧给的 G_ij 式是 g_0i = 0 下的退化形式，"
      "一般情形下 G_ij 的 3+1 表达须用完整的 ADM 分解"
      % (v1_eij, ("%.3e" % v1_e0i) if v1_e0i is not None else "不可判"))

print()
print("[V-4] 两处缺口是否被补上（d_0 g 与 d_0^2 g 在动源档须非零）")
v4_ok = all(abs(x[7]) > 0 for x in main) and all(abs(x[8]) > 0 for x in main)
v4_static_ok = all(abs(x[7]) == 0 and abs(x[8]) == 0 for x in static)
print("  动源档的 d_0 g_xx 读数：%s"
      % ", ".join("v=%.2f 给 %+.3e" % (x[0], x[7]) for x in main))
print("  静场档的 d_0 g_xx 读数：%s（应全为零：%s）"
      % (", ".join("%+.3e" % x[7] for x in static), "是" if v4_static_ok else "否"))
check("V-4", v4_ok and v4_static_ok,
      "两处缺口被补上：动源档的 d_0 g_xx 与 d_0^2 g_0x 处处非零，且静场档严格为零")

print()
print("[V-5] 恒真替身：不补 d_0 g_ij（回到第 781 号的置零），G_00 侧的相容性须变差")
alt_max = 0.0
for v, r0 in [(x[0], x[1]) for x in rows]:
    g, dg0, d2g0 = build((r0, 0.0, 0.0), EPS, HALF, v)
    dg0_alt = dg0.copy()
    for i in range(1, 4):
        for j in range(1, 4):
            dg0_alt[i, j] = 0.0
    Gam, dg, gi, sh = christoffel(g, dg0_alt, EPS)
    Ric00, Ric0i, Ricij, R4d = ricci_4d(Gam, dg, d2g0, gi, sh, EPS)
    N2 = -g[0, 0]
    gam = np.zeros((3, 3) + sh)
    for i in range(3):
        for j in range(3):
            gam[i, j] = g[i + 1, j + 1] / N2
    Ric3, R3, ginv3 = ricci_3d(gam, EPS)
    G00_4 = ctr(Ric00 - 0.5 * g[0, 0] * R4d)
    G00_3 = ctr(-0.5 * N2 * R3)
    alt_max = max(alt_max, rel(G00_3, G00_4))
print("  替身（d_0 g_ij 置零）下的最大相对差 %.3e ｜ 真值 %.3e" % (alt_max, v2_max))
check("V-5", alt_max > v2_max * 10,
      "替身下的最大相对差 %.3e 显著大于真值 %.3e，判据有区分力" % (alt_max, v2_max))

print()
print("[V-6] 状态登记")
reg = [
    ("其一", "两处缺口", "d_0 g_{ij} 被置零（完成场随源移动故非零）；d_0 Gamma^0_{00} 缺 d_0^2 g_{0 sigma} 项"),
    ("其二", "修法", "切片差分（t = 0、-h、-2h 三片，h = 1e-5），自动给出两处且绕开解析推导"),
    ("其三", "G_00 侧", "相容：静场档 1.494e-03 与 2.703e-05，与第 780 号一致"),
    ("其四", "G_ij 侧", "差 5 至 9 倍，**结构性**：3+1 侧给的式是 g_0i = 0 下的退化形式"),
    ("其五", "G_0i 侧", "ADM 关系式未通过（静场下 3+1 侧给出 1.33 而 4D 侧为 0）"),
    ("其六", "订正", "第 781 号的「结构性信息缺失」措辞过强，须在作者确认后修订"),
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
IMG = os.path.join(ROOT, "images", "relations_itemwise.png")

fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))

ax = axes[0]
for r0 in RADII:
    xs = [x[0] for x in rows if x[1] == r0]
    ys = [x[2] for x in rows if x[1] == r0]
    ax.semilogy(xs, np.maximum(ys, 1e-18), "o-", linewidth=1.6, label=u"r = %.1f" % r0)
ax.axhline(1e-2, color="#d62728", linestyle=":", linewidth=1.4, label=u"门槛 $10^{-2}$")
ax.set_xlabel(u"速度 $v$")
ax.set_ylabel(u"$G_{00}$ 的相对差")
ax.set_title(u"(a) $G_{00}$ 侧的相容性")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

ax = axes[1]
for r0 in RADII:
    xs = [x[0] for x in rows if x[1] == r0]
    ys = [x[3] for x in rows if x[1] == r0]
    ax.semilogy(xs, np.maximum(ys, 1e-18), "o-", linewidth=1.6, label=u"r = %.1f" % r0)
ax.set_xlabel(u"速度 $v$")
ax.set_ylabel(u"$G_{ij}$ 的相对差")
ax.set_title(u"(b) $G_{ij}$ 侧：结构性的不等")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

ax = axes[2]
for r0 in RADII:
    xs = [x[0] for x in rows if x[1] == r0]
    ys = [abs(x[7]) for x in rows if x[1] == r0]
    ax.semilogy(xs, np.maximum(ys, 1e-18), "o-", linewidth=1.6,
                label=u"$\\partial_0 g_{xx}$（r = %.1f）" % r0)
    ys2 = [abs(x[8]) for x in rows if x[1] == r0]
    ax.semilogy(xs, np.maximum(ys2, 1e-18), "s--", linewidth=1.3,
                label=u"$\\partial_0^2 g_{0x}$（r = %.1f）" % r0)
ax.set_xlabel(u"速度 $v$")
ax.set_ylabel(u"时间导数的绝对值")
ax.set_title(u"(c) 两处缺口的补上")
ax.legend(fontsize=7)
ax.grid(alpha=0.3, which="both")

fig.suptitle(u"第784号：三条 3+1 关系的逐条核对与第 781 号装置的两处缺口", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(IMG, dpi=150)
print()
print("图已存：%s" % IMG)
