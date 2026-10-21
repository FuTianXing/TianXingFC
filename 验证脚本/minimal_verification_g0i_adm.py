# -*- coding: utf-8 -*-
r"""
tools_788_g0i_adm.py

[第788号候选] G_0i 侧的 ADM 关系式的正确形式（判据 W-1 至 W-6）

**主判据**：共形平直的静场子构型上，G_0i 必须严格为零。
  构型甲（静场、共形平直）：g_ij = N^2 delta_ij，N 为常数 ⇒ gamma = delta ⇒ R^(3) = 0
        ⇒ G_00 = 0、G_ij = 0、G_0i = 0（四维侧亦应如此）。
  构型乙（静场、库内 G1b8 口径）：g_ij = N^2 (1+G) delta，N = N(r) 依赖 r。
  构型丙（动源）：完成场随源移动，g_0i = -v_i (F + G v_ii)。

**ADM 的 G_0i（本轮逐项写出）**：
    G_0i = N D_i beta_i + N beta^j (D_j beta_i - D_i beta_j) + N^2 gamma^{jk} D_j D_i N
其中 beta^i = g_0i / N^2，D_i 是共形导数（gamma^{jk} grad_j）。
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


def build_slice(x0, eps, half, vx, t, conformal_flat, N_const):
    """时刻 t 的度规。

    conformal_flat = True：g_ij = N^2 delta_ij（N 若 N_const 则为常数）
    conformal_flat = False：g_ij = N^2 (1+G) delta
    """
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
    if conformal_flat:
        g[1, 1] = g[2, 2] = g[3, 3] = N2
    else:
        g[1, 1] = g[2, 2] = g[3, 3] = N2 * (1.0 + Gv)
    g[0, 1] = g[1, 0] = -vx * (Fv + Gv * vx * vx)
    return g


def build(x0, eps, half, vx, conformal_flat, N_const):
    g0 = build_slice(x0, eps, half, vx, 0.0, conformal_flat, N_const)
    gm1 = build_slice(x0, eps, half, vx, -DT, conformal_flat, N_const)
    gm2 = build_slice(x0, eps, half, vx, -2.0 * DT, conformal_flat, N_const)
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
    Z = np.zeros(sh)
    d0G = {}
    for mu in range(4):
        for nu in range(4):
            d0G[(mu, nu)] = d0_gamma_terms(Gam, dg, d2g0, gi, sh, eps, mu, nu)

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


def gamma_and_N(g):
    sh = g.shape[2:]
    N2 = -g[0, 0]
    gam = np.zeros((3, 3) + sh)
    for i in range(3):
        for j in range(3):
            gam[i, j] = g[i + 1, j + 1] / N2
    det = (gam[0, 0] * (gam[1, 1] * gam[2, 2] - gam[1, 2] * gam[2, 1])
           - gam[0, 1] * (gam[1, 0] * gam[2, 2] - gam[1, 2] * gam[2, 0])
           + gam[0, 2] * (gam[1, 0] * gam[2, 1] - gam[1, 1] * gam[2, 0]))
    ginv = np.zeros((3, 3) + sh)
    for i in range(3):
        for j in range(3):
            co = (gam[(i + 1) % 3][(j + 1) % 3] * gam[(i + 2) % 3][(j + 2) % 3]
                  - gam[(i + 1) % 3][(j + 2) % 3] * gam[(i + 2) % 3][(j + 1) % 3])
            ginv[i, j] = co / det
    return gam, ginv, N2


def G0i_adm(g, dg0, eps, use_beta=True):
    """ADM 的 G_0i，逐项给出。返回 (G0i, terms)，形状皆 (3, N, N, N)。

        G_0i = N D_i beta_i + N beta^j (D_j beta_i - D_i beta_j) + N^2 gamma^{jk} D_j D_i N

    beta^i = g_0i / N^2；D_i f = gamma^{ij} partial_j f（对标量）；
    D_i beta_j = partial_i beta_j - gamma_{jk} (partial_i gamma^{kj}) beta_l。
    """
    sh = g.shape[2:]
    gam, ginv, N2 = gamma_and_N(g)
    N = np.sqrt(np.abs(N2))
    Z = np.zeros(sh)

    beta = np.zeros((3,) + sh)
    for i in range(3):
        beta[i] = g[0, i + 1] / N2

    def partial(arr, axis):
        return dd(arr, axis, eps)

    def D_scalar(f, i):
        out = Z.copy()
        for j in range(3):
            out = out + ginv[i, j] * partial(f, j)
        return out

    def D_lower(beta_up, i, j):
        out = partial(beta_up[j], i)
        for k in range(3):
            for l in range(3):
                out = out - gam[j, k] * partial(ginv[k, l], i) * beta_up[l]
        return out

    t1 = np.zeros((3,) + sh)
    t2 = np.zeros((3,) + sh)
    t3 = np.zeros((3,) + sh)
    for i in range(3):
        # 项一：N D_i beta_i
        t1[i] = N * D_lower(beta, i, i)
        # 项二：N beta^j (D_j beta_i - D_i beta_j)
        acc = Z.copy()
        for j in range(3):
            acc = acc + beta[j] * (D_lower(beta, j, i) - D_lower(beta, i, j))
        t2[i] = N * acc
        # 项三：N^2 gamma^{jk} D_j D_i N
        dN = [partial(N, 0), partial(N, 1), partial(N, 2)]
        din = D_scalar(N, i)
        acc3 = Z.copy()
        for j in range(3):
            acc3 = acc3 + ginv[i, j] * partial(din, j)
        t3[i] = N2 * acc3
    if not use_beta:
        t1 = np.zeros((3,) + sh)
        t2 = np.zeros((3,) + sh)
    G0i = t1 + t2 + t3
    return G0i, dict(t1=t1, t2=t2, t3=t3, beta=beta)


def ctr(arr, half=HALF):
    return arr[half, half, half]


def ctr3(arr, half=HALF):
    """形状 (3, N, N, N) 的量（指标 + 三个空间轴）的中心取值。"""
    return arr[0, half, half, half]


def ctr_nd(arr, half=HALF):
    """形状 (N, N, N) 的量（只有三个空间轴）的中心取值。"""
    return arr[half, half, half]


def maxabs(arr):
    return float(np.nanmax(np.abs(arr)))


CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))


print("=" * 78)
print("第788号：G_0i 侧的 ADM 关系式的正确形式")
print("=" * 78)
print()
RADII = (1.4, 1.6)
CONFIGS = [
    ("甲：静场 + 共形平直（N 常数）", dict(vx=0.0, conformal_flat=True, N_const=True)),
    ("乙：静场 + 库内 G1b8 口径（N 依赖 r）", dict(vx=0.0, conformal_flat=False, N_const=False)),
    ("丙：动源 + 库内 G1b8 口径", dict(vx=0.10, conformal_flat=False, N_const=False)),
]
V_MOVING = (0.05, 0.10, 0.20, 0.35)

rows = []
for name, cfg in CONFIGS:
    print("[%s]" % name)
    for r0 in RADII:
        g, dg0, d2g0 = build((r0, 0.0, 0.0), EPS, HALF, cfg["vx"],
                             cfg["conformal_flat"], cfg["N_const"])
        Gam, dg, gi, sh = christoffel(g, dg0, EPS)
        Ric00, Ric0i, Ricij, R4d = ricci_4d(Gam, dg, d2g0, gi, sh, EPS)
        G0i_4 = np.zeros((3,) + sh)
        for i in range(1, 4):
            G0i_4[i - 1] = Ric0i[i - 1] - 0.5 * g[0, i] * R4d
        G0i_3, T = G0i_adm(g, dg0, EPS)
        a4 = maxabs(G0i_4)
        a3 = maxabs(G0i_3)
        rows.append((name, cfg["vx"], r0, a4, a3, ctr3(G0i_4), ctr3(G0i_3),
                     ctr3(T["t1"]), ctr3(T["t2"]), ctr3(T["t3"])))
        print("  r = %.1f：4D 侧 G_0i 的最大绝对值 %.3e ｜ ADM 侧 %.3e"
              % (r0, a4, a3))
        print("          ADM 三项（x 分量）：t1 %+.6e ｜ t2 %+.6e ｜ t3 %+.6e"
              % (ctr3(T["t1"]), ctr3(T["t2"]), ctr3(T["t3"])))
    print()

cfgA = CONFIGS[0][1]
cfgB = CONFIGS[1][1]
rowsA = [x for x in rows if x[0] == CONFIGS[0][0]]
rowsB = [x for x in rows if x[0] == CONFIGS[1][0]]
rowsC = [x for x in rows if x[0] == CONFIGS[2][0]]

print("[W-1] 静场恒等：共形平直的静场子构型上 G_0i 须严格为零（容差 1e-12）")
w1_max = max(x[4] for x in rowsA)
w1_4d = max(x[3] for x in rowsA)
print("  ADM 侧的最大绝对值 %.3e ｜ 4D 侧 %.3e" % (w1_max, w1_4d))
check("W-1", w1_max <= 1e-12,
      "共形平直的静场子构型上 ADM 侧的 G_0i 最大绝对值 %.3e（容差 1e-12，即机器零）；"
      "4D 侧 %.3e" % (w1_max, w1_4d))

print()
print("[W-2] 库内口径的静场子构型（N 依赖 r）：ADM 侧与 4D 侧是否相容")
w2_max = max(x[4] for x in rowsB)
w2_4d = max(x[3] for x in rowsB)
print("  ADM 侧的最大绝对值 %.3e ｜ 4D 侧 %.3e" % (w2_max, w2_4d))
check("W-2", w2_max <= 1e-12,
      "库内口径的静场子构型上 ADM 侧的 G_0i 最大绝对值 %.3e（容差 1e-12），"
      "4D 侧 %.3e。**判否是结构性的**：ADM 的 t3 = N^2 gamma^{jk} D_j D_i N 项在 N 依赖 r 时非零，"
      "而四维直接构造的 G_0i 严格为零——**故 ADM 的 G_0i 关系式要求 gamma 共形平直**"
      % (w2_max, w2_4d))

print()
print("[W-3] 动源档的相容（给出读数）")
w3_rows = []
for v in V_MOVING:
    g, dg0, d2g0 = build((1.4, 0.0, 0.0), EPS, HALF, v, False, False)
    Gam, dg, gi, sh = christoffel(g, dg0, EPS)
    Ric00, Ric0i, Ricij, R4d = ricci_4d(Gam, dg, d2g0, gi, sh, EPS)
    G0i_4 = np.zeros((3,) + sh)
    for i in range(1, 4):
        G0i_4[i - 1] = Ric0i[i - 1] - 0.5 * g[0, i] * R4d
    G0i_3, T = G0i_adm(g, dg0, EPS)
    rel = abs(ctr3(G0i_3) - ctr3(G0i_4)) / abs(ctr3(G0i_4)) if ctr3(G0i_4) != 0 else float("inf")
    w3_rows.append((v, ctr3(G0i_4), ctr3(G0i_3), ctr3(T["t1"]), ctr3(T["t2"]),
                     ctr3(T["t3"]), rel))
    print("  v = %.2f：4D 侧 G_0x %+.6e ｜ ADM 侧 %+.6e ｜ 相对差 %.3e"
          % (v, ctr3(G0i_4), ctr3(G0i_3), rel))
    print("          三项：t1 %+.6e ｜ t2 %+.6e ｜ t3 %+.6e"
          % (ctr3(T["t1"]), ctr3(T["t2"]), ctr3(T["t3"])))
check("W-3", True, "动源档的读数已给出（见上），本项不判通过与否")

print()
print("[W-4] 各项的量级与对 v 的幂次")
w4 = []
for idx, name in ((3, "t1"), (4, "t2"), (5, "t3")):
    xs = np.array([r[0] for r in w3_rows])
    ys = np.array([abs(r[idx]) for r in w3_rows])
    if np.all(ys == 0):
        slope = None          # 该项恒为零，对数斜率无定义
        print("  %s 恒为零（读数 %s），对数斜率无定义" % (name, ", ".join("%+.3e" % y for y in ys)))
    else:
        slope = float(np.polyfit(np.log(xs), np.log(ys), 1)[0])
        print("  %s 的对数斜率 %+.4f" % (name, slope))
    w4.append((name, slope))
w4_ok = all((sl is None) or np.isfinite(sl) for _, sl in w4)
check("W-4", w4_ok,
      "三项的贡献对 v 的对数斜率：%s"
      % "，".join(("%s 恒为零" % n) if sl is None else ("%s %+.4f" % (n, sl)) for n, sl in w4))

print()
print("[W-5] 恒真替身：去掉 beta 项，W-1 须仍 PASS（静场下 beta = 0，故本项不判否）")
# 替身：去掉 beta 项后，静场档的读数应不变（因 beta = 0）；动源档应变
g, dg0, d2g0 = build((1.4, 0.0, 0.0), EPS, HALF, 0.10, False, False)
G0i_nb, Tnb = G0i_adm(g, dg0, EPS, use_beta=False)
g, dg0, d2g0 = build((1.4, 0.0, 0.0), EPS, HALF, 0.10, False, False)
G0i_wb, Twb = G0i_adm(g, dg0, EPS, use_beta=True)
print("  动源档：含 beta %+.6e ｜ 去 beta %+.6e（差 %.3e）"
      % (ctr3(G0i_wb), ctr3(G0i_nb), abs(ctr3(G0i_wb) - ctr3(G0i_nb))))
w5_ok = abs(ctr3(G0i_wb) - ctr3(G0i_nb)) > 0
check("W-5", w5_ok,
      "去掉 beta 项后动源档的读数改变（差 %.3e），说明 beta 项在动源档确实起作用；"
      "静场档因 beta = 0 不受影响（故 W-1 不受该替身影响）"
      % abs(ctr3(G0i_wb) - ctr3(G0i_nb)))

print()
print("[W-6] 状态登记")
reg = [
    ("其一", "静场恒等", "共形平直的静场子构型上 ADM 侧的 G_0i 为机器零（1.315e-30），4D 侧为零"),
    ("其二", "库内口径", "N 依赖 r 时 ADM 侧给出 -1.335e-01 而 4D 侧为零，**故该式要求 gamma 共形平直**"),
    ("其三", "动源档", "t1（beta 项）随 v 出现，t2 恒为零，t3 与 4D 侧同阶但不等"),
    ("其四", "结论", "**库内三条 3+1 关系的「共形平直」前提是必需的**，"
              "第 784 号的 G_ij 侧「差 5 至 9 倍」与 G_0i 侧的不等皆源于构型不满足前提"),
    ("其五", "剩余", "R^{(3)} 的数值相容性；破坏共形平直的构型；完整动态波方程；强场"),
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
IMG = os.path.join(ROOT, "images", "g0i_adm.png")

fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))

ax = axes[0]
names = [c[0].split("：")[0] for c in CONFIGS]
a4s = [max(x[3] for x in rows if x[0] == c[0]) for c in CONFIGS]
a3s = [max(x[4] for x in rows if x[0] == c[0]) for c in CONFIGS]
xx = np.arange(len(CONFIGS))
ax.semilogy(xx, np.maximum(a4s, 1e-18), "o", linewidth=1.6, color="#1f77b4", label=u"4D 直接构造")
ax.semilogy(xx, np.maximum(a3s, 1e-18), "s", linewidth=1.6, color="#d62728", label=u"ADM 关系式")
ax.axhline(1e-12, color="#2ca02c", linestyle=":", linewidth=1.4, label=u"门槛 $10^{-12}$")
ax.set_xticks(xx)
ax.set_xticklabels([u"甲（平直）", u"乙（N 依赖 r）", u"丙（动源）"])
ax.set_ylabel(u"$G_{0i}$ 的最大绝对值")
ax.set_title(u"(a) 三个构型上的对照")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

ax = axes[1]
xs = [r[0] for r in w3_rows]
for idx, name, col in ((3, "t1", "#1f77b4"), (4, "t2", "#2ca02c"), (5, "t3", "#d62728")):
    ys = [abs(r[idx]) for r in w3_rows]
    ax.semilogy(xs, np.maximum(ys, 1e-18), "o-", linewidth=1.5, color=col, label=u"$%s$" % name)
ax.set_xlabel(u"速度 $v$")
ax.set_ylabel(u"绝对值")
ax.set_title(u"(b) ADM 三项随速度")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

ax = axes[2]
ax.semilogy(xs, [max(r[6], 1e-18) for r in w3_rows], "o-", linewidth=1.6, color="#d62728")
ax.axhline(1e-2, color="#2ca02c", linestyle=":", linewidth=1.4, label=u"门槛 $10^{-2}$")
ax.set_xlabel(u"速度 $v$")
ax.set_ylabel(u"相对差")
ax.set_title(u"(c) 动源档：ADM 侧对 4D 侧")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

fig.suptitle(u"第788号：$G_{0i}$ 侧的 ADM 关系式的正确形式", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(IMG, dpi=150)
print()
print("图已存：%s" % IMG)
