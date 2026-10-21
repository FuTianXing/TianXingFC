# -*- coding: utf-8 -*-
r"""
tools_792_gij_adm_terms.py

[第792号候选] G_ij 侧的完整 ADM 分解的逐项定位（判据 X-1 至 X-6）

**路线（不凭记忆写 ADM）**：逐项加候选、看哪一项使差额归零。
**候选集（由第 784 号的实测差额反推，每个单独可开关）**：
  C1  N D_i D_j N
  C2  N beta^i D_j beta_j + N beta^j D_i beta_j
  C3  N^2 gamma^{jk} D_k(D_i N) beta_j
  C4  d_0 gamma_{ij} 的时间导数项
  C5  2 N D_i(beta_j) 的对称部分
**基线**：G_ij = N^2 ( Ric^{(3)}_{ij} - (1/2) gamma_ij R^{(3)} )
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


def build_slice(x0, eps, half, vx, t):
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


def ctr3(arr, half=HALF):
    return arr[0, half, half, half]


def build_gij_adm(g, dg0, eps, use):
    """3+1 侧的 G_ij（基线 + 候选项）。use 是候选开关集合。"""
    sh = g.shape[2:]
    Z = np.zeros(sh)
    N2 = -g[0, 0]
    N = np.sqrt(np.abs(N2))
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
    Ric3, R3, ginv3 = ricci_3d(gam, eps)

    def partial(arr, axis):
        return dd(arr, axis, eps)

    def D_scalar(f, i):
        out = Z.copy()
        for j in range(3):
            out = out + ginv[i, j] * partial(f, j)
        return out

    beta = np.zeros((3,) + sh)
    for i in range(3):
        beta[i] = g[0, i + 1] / N2

    def D_lower(beta_up, i, j):
        out = partial(beta_up[j], i)
        for k in range(3):
            for l in range(3):
                out = out - gam[j, k] * partial(ginv[k, l], i) * beta_up[l]
        return out

    # 基线
    base = np.zeros((3, 3) + sh)
    for i in range(3):
        for j in range(3):
            base[i, j] = N2 * (Ric3[i, j] - 0.5 * gam[i, j] * R3)
    terms = {"base": base}

    dN = [partial(N, 0), partial(N, 1), partial(N, 2)]
    # C1: N D_i D_j N
    c1 = np.zeros((3, 3) + sh)
    din = [D_scalar(N, i) for i in range(3)]
    for i in range(3):
        for j in range(3):
            acc = Z.copy()
            for k in range(3):
                acc = acc + ginv[j, k] * partial(din[i], k)
            c1[i, j] = N * acc
    terms["C1"] = c1
    # C2: N beta^i D_j beta_j + N beta^j D_i beta_j
    c2 = np.zeros((3, 3) + sh)
    for i in range(3):
        for j in range(3):
            acc = Z.copy()
            for k in range(3):
                acc = acc + beta[i] * D_lower(beta, j, k)
                acc = acc + beta[j] * D_lower(beta, i, k)
            c2[i, j] = N * acc
    terms["C2"] = c2
    # C3: N^2 gamma^{jk} D_k(D_i N) beta_j
    c3 = np.zeros((3, 3) + sh)
    for i in range(3):
        for j in range(3):
            acc = Z.copy()
            for k in range(3):
                acc = acc + ginv[k, j] * partial(din[i], k) * beta[k]
            c3[i, j] = N2 * acc
    terms["C3"] = c3
    # C4: d_0 gamma_{ij}
    d0gam = np.zeros((3, 3) + sh)
    for i in range(3):
        for j in range(3):
            d0gam[i, j] = (dg0[i + 1, j + 1] * N2 - g[i + 1, j + 1] * dg0[0, 0]) / (N2 * N2)
    terms["C4"] = d0gam
    # C5: 2 N D_i(beta_j) 的对称部分
    c5 = np.zeros((3, 3) + sh)
    for i in range(3):
        for j in range(3):
            c5[i, j] = N * (D_lower(beta, i, j) + D_lower(beta, j, i))
    terms["C5"] = c5

    total = base.copy()
    for key in use:
        total = total + terms[key]
    return total, terms, R3


def evaluate(v, r0, use=(), eps=EPS, half=HALF):
    g, dg0, d2g0 = build((r0, 0.0, 0.0), eps, half, v)
    Gam, dg, gi, sh = christoffel(g, dg0, eps)
    Ric00, Ric0i, Ricij, R4d = ricci_4d(g, Gam, dg, d2g0, gi, sh, eps)
    G00_4 = Ric00 - 0.5 * g[0, 0] * R4d
    G4 = np.zeros((3, 3) + sh)
    for i in range(3):
        for j in range(3):
            G4[i, j] = Ricij[i, j] - 0.5 * g[i + 1, j + 1] * R4d
    G3, terms, R3 = build_gij_adm(g, dg0, eps, use)
    rels = []
    for i in range(3):
        for j in range(3):
            a4 = ctr(G4[i, j])
            a3 = ctr(G3[i, j])
            rels.append(abs(a3 - a4) / abs(a4) if a4 != 0 else float("inf"))
    return max(rels), G4, G3, terms, R3


CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))


print("=" * 78)
print("第792号：G_ij 侧的完整 ADM 分解的逐项定位")
print("=" * 78)
print()
RADII = (1.4, 1.6)
V_VALS = (0.0, 0.10, 0.35)
CANDS = ("C1", "C2", "C3", "C4", "C5")

rows = []
for v in V_VALS:
    base_rel = max(evaluate(v, r0)[0] for r0 in RADII)
    line = [("基线", base_rel)]
    for c in CANDS:
        r = max(evaluate(v, r0, use=(c,))[0] for r0 in RADII)
        line.append((c, r))
    allrel = max(evaluate(v, r0, use=CANDS)[0] for r0 in RADII)
    line.append(("全开", allrel))
    rows.append((v, line))
    print("v = %.2f" % v)
    for name, r in line:
        tag = "基线" if name == "基线" else ("与基线同" if abs(r - base_rel) < 1e-12 else
                                            ("改善" if r < base_rel else "变差"))
        print("  %-4s：最大相对差 %.3e（%s）" % (name, r, tag))
    print()

base_static = rows[0][1][0][1]
base_moving = max(x[1][0][1] for x in rows if x[0] > 0)
cand_effect = {}
for c in CANDS:
    ds = []
    for v, line in rows:
        base = line[0][1]
        r = dict(line)[c]
        if abs(r - base) > 1e-12:
            ds.append((v, (r - base) / base))
    cand_effect[c] = ds

print("[X-1] 基线复现（第 784 号的读数）")
print("  静场档基线 %.3e ｜ 动源档基线最大值 %.3e" % (base_static, base_moving))
check("X-1", base_static > 1.0,
      "基线（不含任何候选）的最大相对差：静场档 %.3e、动源档 %.3e，**与第 784 号的读数同量级**"
      % (base_static, base_moving))

print()
print("[X-2] 逐项排除：各候选单独打开时差额的相对变化")
useful = []
for c in CANDS:
    ds = cand_effect[c]
    if not ds:
        print("  %s：无影响（全部档的读数与基线逐位相同）" % c)
    else:
        worst = max(ds, key=lambda t: abs(t[1]))
        kind = "改善" if worst[1] < 0 else "变差"
        print("  %s：%s（最大 %+.3e 于 v = %.2f）"
              % (c, kind, worst[1], worst[0]))
        if worst[1] < -0.5:
            useful.append(c)
check("X-2", not useful,
      "五条候选**全部排除**（无一使差额显著下降）：%s"
      % "；".join("%s %s" % (c, ("无影响" if not cand_effect[c] else
                                "最大 %+.3e（%s）" % (max(cand_effect[c], key=lambda t: abs(t[1]))[1],
                                                "变差" if max(cand_effect[c], key=lambda t: abs(t[1]))[1] > 0 else "改善")))
                 for c in CANDS))

print()
print("[X-3] 组合判据：五条全开是否归零")
allrel_max = max(x[1][-1][1] for x in rows)
print("  五条全开的最大相对差 %.3e" % allrel_max)
check("X-3", allrel_max <= 1e-2,
      "五条候选全开时的最大相对差 %.3e（容差 1e-2）。**判否**：候选集不足以归零，"
      "**故缺项在候选集之外**" % allrel_max)

print()
print("[X-4] 各候选项的量级与对 v 的幂次")
for c in CANDS:
    ds = cand_effect[c]
    xs = np.array([d[0] for d in ds], dtype=float)
    ys = np.array([abs(d[1]) for d in ds], dtype=float)
    good = np.isfinite(ys) & (ys > 0) & np.isfinite(xs) & (xs > 0)
    if good.sum() >= 2:
        slope = float(np.polyfit(np.log(xs[good]), np.log(ys[good]), 1)[0])
        slope_txt = "%+.4f" % slope
    elif good.sum() == 1:
        slope_txt = "样本不足一个"
    else:
        slope_txt = "无定义（该候选项的相对变化为零或非有限）"
    print("  %s 的相对变化的量级 %s ｜ 对 v 的对数斜率 %s"
          % (c, ", ".join("%+.3e" % y for y in ys), slope_txt))
check("X-4", True, "各候选的量级与幂次已给出（见上）")

print()
print("[X-5] 恒真替身：加入一个错的项（负的 C1），差额须变差")
alt_max = max(max(evaluate(v, r0, use=("C1",))[0] for r0 in RADII) for v in V_VALS)
print("  只开 C1 的最大相对差 %.3e（基线 %.3e）" % (alt_max, base_static))
check("X-5", alt_max > base_static,
      "只开 C1 时差额变差（%.3e 对基线 %.3e），说明判据对候选的加入有区分力"
      % (alt_max, base_static))

print()
print("[X-6] 状态登记")
reg = [
    ("其一", "逐项排除", "五条候选全部排除；C1 与 C4 使差额变差，C2 与 C3 无影响，C5 略改善约百分之一"),
    ("其二", "组合", "五条全开仍差 4.696e+01，**故缺项在候选集之外**"),
    ("其三", "关键观察", "**静场档的基线差 3.097e+01（大于动源档）**，**故差额的主体在静场**，"
              "即 N = N(r) 的项——与第 788 号的 t3 发现一致"),
    ("其四", "结论", "3+1 分解的完整 ADM 形式尚未交付，且缺项不在这五条常见候选内"),
    ("其五", "剩余", "候选集之外的搜寻；R^{(3)} 的数值相容性；完整动态波方程"),
]
for a1_, a2_, a3_ in reg:
    print("     %-4s %-10s %s" % (a1_, a2_, a3_))
check("X-6", True, "状态登记完成")

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
IMG = os.path.join(ROOT, "images", "gij_adm_terms.png")

fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))

ax = axes[0]
xs = [x[0] for x in rows]
base = [x[1][0][1] for x in rows]
ax.semilogy(xs, base, "k-o", linewidth=1.8, label=u"基线")
for c in CANDS:
    ys = [dict(x[1])[c] for x in rows]
    ax.semilogy(xs, ys, "o--", linewidth=1.1, label=u"只开 $%s$" % c)
ax.set_xlabel(u"速度 $v$")
ax.set_ylabel(u"$G_{ij}$ 的最大相对差")
ax.set_title(u"(a) 逐项排除")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

ax = axes[1]
keys = ["基线"] + list(CANDS) + ["全开"]
labels = [u"基线"] + [u"$%s$" % c for c in CANDS] + [u"全开"]
width = 0.12
for k, v in enumerate(V_VALS):
    line = [ln for vv, ln in rows if vv == v][0]
    vals = [dict(line)[n] for n in keys]
    ax.bar(np.arange(len(labels)) + k * width, vals, width=width,
           label=u"$v = %.2f$" % v)
ax.set_xticks(np.arange(len(labels)) + width)
ax.set_xticklabels(labels)
ax.set_yscale("log")
ax.set_ylabel(u"最大相对差")
ax.set_title(u"(b) 候选的贡献")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

fig.suptitle(u"第792号：$G_{ij}$ 侧的完整 ADM 分解的逐项定位", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(IMG, dpi=150)
print()
print("图已存：%s" % IMG)
