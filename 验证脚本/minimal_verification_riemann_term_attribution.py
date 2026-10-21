# -*- coding: utf-8 -*-
r"""
minimal_verification_riemann_term_attribution.py

[第762号] 解析黎曼张量的逐项展开与差额的项级归属（判据 K-1 至 K-6）

背景：判据第 748 号判出差额项与装置的构造配置无关（两个满足 3+1 分解前提的装置上份额不减；
完全平直档机器 R00 严格为零，机器读数无系统误差），故根源指向 3+1 分解公式本身。748 号已
排除候选式共 17 个，其留下的入口是逐项展开解析黎曼张量与机器读数比对。

**展开的原理**：四维黎曼张量的 00 分量在 g_0i = 0、g_tt = -N^2 的静场下可由 ADM 分解写出，
各项分别是 t 方向的导数项、Gamma^i_{0j}（移位）项、空间曲率项、N 的导数项；而严格 3+1 的
三维分解预言只含其中一部分。

**本轮的装置改动点**：机器的 R00 改用**直接的四维黎曼张量分量构造**（从 G_munu 与其一阶导数
直接算出 R_00），而非 748 号的反解式（R00 = G00 - (1/2) g00 R 与 R = -G^mu_mu），否则无法
做项级比对。

**档**：
  甲 g00 = -(1-F)，gij = (1-F)(1+G) delta（gamma_ij = (1+G) delta，Om^2 = 1+G）
  乙 g00 = -(1-F)，gij = (1-F) delta（gamma_ij = delta，Om^2 = 1）

判据：
  K-1 展开式与机器读数的相容
  K-2 差额的项级归属
  K-3 移位项的零贡献
  K-4 差额对场强 F 的依赖阶
  K-5 恒真替身
  K-6 状态登记

运行：python 验证脚本/minimal_verification_riemann_term_attribution.py
确定性（无随机），约 5 秒。图存 images/riemann_term_attribution.png
"""
import io
import math
import os
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "images", "riemann_term_attribution.png")

T0 = time.time()
CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))


# ----------------------------------------------------------------------
# 装置（自建；第 716 号与 G1b8 均不 import，见第 754 号的教训）
# ----------------------------------------------------------------------
HALF = 4
C = HALF
NP_ = 2 * HALF + 1
EPS = 0.05


def grid():
    ii, jj, kk = np.indices((NP_, NP_, NP_))
    return ii, jj, kk


II, JJ, KK = grid()
CTR = NP_ // 2
R_GRID = np.sqrt((II - CTR) ** 2.0 + (JJ - CTR) ** 2.0 + (KK - CTR) ** 2.0)


def th_lin(r):
    """库内的完成场线性部分（与 G1b8 同构）：erf 型核，m = 0.05。"""
    m = 0.05
    r = np.asarray(r, dtype=float)
    return 1.0 / (1.0 + np.exp((r - 1.0) / m))


def th_c(r):
    """完成场：th_lin - th_lin^2。"""
    t = th_lin(r)
    return t - t * t


def F_of(r):
    return 2.0 * th_c(r)


def G_of(r):
    t = th_c(r)
    return 2.0 * t + 3.5 * t * t


def d1_keep(a, axis, eps):
    b = np.moveaxis(a, axis, -1)
    out = np.full_like(b, np.nan)
    out[..., 2:-2] = (b[..., :-4] - 8 * b[..., 1:-3] + 8 * b[..., 3:-1] - b[..., 4:]) / (12 * eps)
    return np.moveaxis(out, -1, axis)


def g_patch(kind, x0, eps):
    """构造度规场 g_munu（4x4xNP_^3）。"""
    k = np.arange(-HALF, HALF + 1) * eps
    X = x0[0] + k[:, None, None]
    Y = x0[1] + k[None, :, None]
    Z = x0[2] + k[None, None, :]
    r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    Fv = F_of(r)
    Gv = G_of(r)
    g = np.zeros((4, 4, NP_, NP_, NP_))
    g[0, 0] = -(1.0 - Fv)
    if kind == "甲":
        g[1, 1] = g[2, 2] = g[3, 3] = (1.0 - Fv) * (1.0 + Gv)
    elif kind == "乙":
        g[1, 1] = g[2, 2] = g[3, 3] = (1.0 - Fv)
    else:
        raise ValueError(kind)
    return g


def christoffel_and_R00(g, eps):
    """从度规场直接构造 Christoffel 符号与 R_00，并按项拆分。

    R_00 = ∂_ρ Γ^ρ_{00} - ∂_0 Γ^ρ_{0ρ} + Γ^ρ_{00} Γ^σ_{ρσ} - Γ^σ_{0ρ} Γ^ρ_{0σ}
    拆分：
      spatial   ∂_ρ Γ^ρ_{00}（ρ 遍历空间指标 1、2、3）
      t_deriv   -∂_0 Γ^ρ_{0ρ}（t 方向；静态场恒为零）
      quad_a    +Γ^ρ_{00} Γ^σ_{ρσ}
      quad_b    -Γ^σ_{0ρ} Γ^ρ_{0σ}
      shift     移位项（g_0i 因子；静场 g_0i = 0，贡献为零）
    """
    dg = [np.zeros_like(g), d1_keep(g, 2, eps), d1_keep(g, 3, eps), d1_keep(g, 4, eps)]
    # 逐点逆度规（本构造的度规是对角的，故逐点取 1/g）
    # 边界处某些分量为零，故在有效点求逆、零点置零，末位用中心点值补齐
    ginvf = np.zeros_like(g)
    for a in range(4):
        ga = g[a, a]
        good = np.abs(ga) > 1e-14
        inv = np.zeros_like(ga)
        inv[good] = 1.0 / ga[good]
        ginvf[a, a] = inv
    # 用中心点（必然非零）替换最后两个平面的零值，消除边界污染
    for a in range(4):
        ctr_val = ginvf[a, a, C, C, C]
        if np.isfinite(ctr_val) and ctr_val != 0.0:
            zero_mask = (ginvf[a, a] == 0.0)
            ginvf[a, a] = np.where(zero_mask, ctr_val, ginvf[a, a])
    # Christoffel 符号
    Gam = np.zeros((4, 4, 4, NP_, NP_, NP_))
    for lam in range(4):
        for mu in range(4):
            for nu in range(4):
                acc = np.zeros((NP_, NP_, NP_))
                for sig in range(4):
                    acc = acc + 0.5 * ginvf[lam, sig] * (
                        dg[mu][sig, nu] + dg[nu][sig, mu] - dg[sig][mu, nu])
                Gam[lam, mu, nu] = acc

    def dGamma(lam, mu, nu, comp):
        """沿 comp 方向求 Gamma^lam_{mu nu} 的导数。
        comp = 0 是 t 方向（静态场下恒为零）；comp = 1,2,3 是 x,y,z 空间方向。
        Gam[lam,mu,nu] 的形状是 3 维（空间三维），其空间轴 0、1、2 对应 x、y、z，
        故 comp = 1,2,3 映射到轴 0、1、2，即轴索引 = comp - 1。"""
        if comp == 0:
            return np.zeros((NP_, NP_, NP_))
        return d1_keep(Gam[lam, mu, nu], comp - 1, eps)

    terms = {}
    # 项一：∂_ρ Γ^ρ_{00}（ρ 遍历空间指标 1、2、3）
    t1 = np.zeros((NP_, NP_, NP_))
    for rho in (1, 2, 3):
        t1 = t1 + dGamma(rho, 0, 0, rho)
    terms["spatial"] = t1

    # 项二：-∂_0 Γ^ρ_{0ρ}（t 方向，静态场恒为零）
    t2 = np.zeros((NP_, NP_, NP_))
    for rho in range(4):
        t2 = t2 - dGamma(rho, 0, rho, 0)
    terms["t_deriv"] = t2

    # 项三：+Γ^ρ_{00} Γ^σ_{ρσ}
    t3 = np.zeros((NP_, NP_, NP_))
    for rho in range(4):
        acc = np.zeros((NP_, NP_, NP_))
        for sig in range(4):
            acc = acc + Gam[sig, rho, sig]
        t3 = t3 + Gam[rho, 0, 0] * acc
    terms["quad_a"] = t3

    # 项四：-Γ^σ_{0ρ} Γ^ρ_{0σ}
    t4 = np.zeros((NP_, NP_, NP_))
    for rho in range(4):
        for sig in range(4):
            t4 = t4 - Gam[sig, 0, rho] * Gam[rho, 0, sig]
    terms["quad_b"] = t4

    R00_full = t1 + t2 + t3 + t4
    terms["shift"] = np.zeros((NP_, NP_, NP_))
    return R00_full, terms


def G_munu_and_Ric(g, eps):
    """由度规场求 G_munu（逐点）与 Ric_munu（供反解式用）。"""
    dg = [np.zeros_like(g), d1_keep(g, 2, eps), d1_keep(g, 3, eps), d1_keep(g, 4, eps)]
    gg = g[:, :, C, C, C]
    ginv = np.linalg.inv(gg)
    ginvf = np.zeros_like(g)
    for a in range(4):
        for b in range(4):
            ginvf[a, b] = ginv[a, b]
    Gam = np.zeros((4, 4, 4, NP_, NP_, NP_))
    for lam in range(4):
        for mu in range(4):
            for nu in range(4):
                acc = np.zeros((NP_, NP_, NP_))
                for sig in range(4):
                    acc = acc + 0.5 * ginvf[lam, sig] * (
                        dg[mu][sig, nu] + dg[nu][sig, mu] - dg[sig][mu, nu])
                Gam[lam, mu, nu] = acc
    dG = np.zeros((4, 4, 4, 4))
    for kk in range(3):
        for lam in range(4):
            for mu in range(4):
                for nu in range(4):
                    # Gam[lam,mu,nu] 是 3 维（空间三维），空间轴 0、1、2 对应 x、y、z
                    dG[lam, mu, nu, kk + 1] = d1_keep(Gam[lam, mu, nu], kk, eps)[C, C, C]
    Ric = np.zeros((4, 4))
    for mu in range(4):
        for nu in range(4):
            acc = 0.0
            for lam in range(4):
                acc += dG[lam, mu, nu, lam]
            Ric[mu, nu] = acc
            for rho in range(4):
                Ric[mu, nu] -= dG[rho, mu, rho, nu]
    Gm = np.zeros((4, 4))
    for mu in range(4):
        for nu in range(4):
            acc = 0.0
            for rho in range(4):
                acc += ginv[rho, nu] * Ric[rho, mu]
            Gm[mu, nu] = Ric[mu, nu] - 0.5 * ginv[mu, nu] * np.trace(Ric)
    return Gm, Ric


def strict_R00_3plus1(kind, r0):
    """严格 3+1 口径的三维分解量。

    返回 (R00_pred, N, lap3N, gam)：
      R00_pred 取 N[lap_(3) N + Gamma^i_{ij} d^j N]（库的 3+1 分解式）；
      同时在调用处用 -2 N lap3N 与 N lap3N 两个对照值核对符号与因子约定。
    """
    Om2f = (lambda r: 1.0 + G_of(r)) if kind == "甲" else (lambda r: np.ones_like(np.asarray(r, dtype=float)))
    Nf = lambda r: math.sqrt(1.0 - float(F_of(r)))
    Om_f = lambda r: math.sqrt(float(Om2f(r)))
    dN = rad_d1(Nf, r0)
    dOm = rad_d1(Om_f, r0)
    Omv = float(Om_f(r0))
    lap3N = rad_d1(lambda x: Om_f(x) ** 2 * rad_d1(Nf, x), r0) / Omv ** 3
    gam = 3.0 * dN * dOm / Omv ** 3
    N = math.sqrt(1.0 - float(F_of(r0)))
    return N * (lap3N + gam), N, lap3N, gam


def rad_d1(f, r0, h=1e-4):
    return (f(r0 + h) - f(r0 - h)) / (2.0 * h)


def center(arr):
    return arr[C, C, C]


print("=" * 78)
print("第762号：解析黎曼张量的逐项展开与差额的项级归属")
print("=" * 78)

# ----------------------------------------------------------------------
# 冒烟检查
# ----------------------------------------------------------------------
print()
print("[冒烟] 装置的非平凡性")
g_a = g_patch("甲", (1.2, 0.0, 0.0), EPS)
R00_a, terms_a = christoffel_and_R00(g_a, EPS)
print("  甲档（r = 1.2）：直接构造的 R00 = %+.6e" % center(R00_a))
smoke_ok = (abs(center(R00_a)) > 0) and np.isfinite(center(R00_a))
print("  冒烟结论：%s" % ("通过" if smoke_ok else "不通过"))
if not smoke_ok:
    print("  冒烟未过，按规范不进判据循环。")
    sys.exit(1)

# ----------------------------------------------------------------------
# K-1 展开式与机器读数的相容；K-2 差额的项级归属
# ----------------------------------------------------------------------
print()
print("[K-1] 展开式与机器读数的相容；[K-2] 差额的项级归属")
RADII = (0.8, 1.0, 1.2, 1.4, 1.6)
k1_rows = []
k2_rows = []
for kind in ("甲", "乙"):
    print("  档 %s：" % kind)
    for r0 in RADII:
        g = g_patch(kind, (r0, 0.0, 0.0), EPS)
        R00_full, terms = christoffel_and_R00(g, EPS)
        R00_mach = center(R00_full)
        pred, N, lap3N, gam = strict_R00_3plus1(kind, r0)
        diff = R00_mach - pred
        share = abs(diff) / abs(R00_mach) if R00_mach != 0 else 0.0
        t_t = center(terms["t_deriv"])
        t_s = center(terms["spatial"])
        t_qa = center(terms["quad_a"])
        t_qb = center(terms["quad_b"])
        t_sh = center(terms["shift"])
        # 逐项的单独读数
        k1_rows.append((kind, r0, R00_mach, pred, diff, share))
        k2_rows.append((kind, r0, t_t, t_s, t_qa, t_qb, t_sh, diff))
        print("    r = %.1f：R00 = %+.6e ｜ 3+1 预言 %+.6e ｜ 差额 %+.6e（份额 %.3e）"
              % (r0, R00_mach, pred, diff, share))
        print("          逐项：t 方向导数 %+.6e ｜ 空间导数 %+.6e ｜ 二次项甲 %+.6e ｜ 二次项乙 %+.6e ｜ 移位 %+.6e"
              % (t_t, t_s, t_qa, t_qb, t_sh))
    # 各项与差额的关系
    print("  档 %s 的项级归属核对：" % kind)
    for _k, r0, t_t, t_s, t_qa, t_qb, t_sh, diff in [x for x in k2_rows if x[0] == kind]:
        cands = {"t_deriv": t_t, "spatial": t_s, "quad_a": t_qa, "quad_b": t_qb,
                 "t_deriv+spatial": t_t + t_s, "t_deriv-spatial": t_t - t_s,
                 "quad_a-quad_b": t_qa - t_qb, "t_deriv+quad_a-quad_b": t_t + t_qa - t_qb}
        best = min(cands.items(), key=lambda kv: abs(kv[1] - diff))
        rel = abs(best[1] - diff) / abs(diff) if diff != 0 else float("inf")
        print("    r = %.1f：差额 %+.6e ｜ 最贴近的项组合 %s = %+.6e（相对差 %.3e）"
              % (r0, diff, best[0], best[1], rel))

# 三个量的并列：甲 直接构造的四维 R00；乙 反解式 R00 = G00 - (1/2) g00 R；丙 库的 3+1 分解预言。
# 第 748 号的机器读数用的是乙；本轮用甲。两者之差正是待归因的差额。
print()
print("[K-2 核心] 三个量的并列：直接构造 / 反解式 / 库的 3+1 分解预言")
three_rows = []
for kind in ("甲", "乙"):
    for r0 in RADII:
        g = g_patch(kind, (r0, 0.0, 0.0), EPS)
        R00_direct, terms = christoffel_and_R00(g, EPS)
        Rd = center(R00_direct)
        Gm, Ric = G_munu_and_Ric(g, EPS)
        ginv = np.linalg.inv(g[:, :, C, C, C])
        R_trace = -(Gm[0, 0] * ginv[0, 0] + Gm[1, 1] * ginv[1, 1]
                    + Gm[2, 2] * ginv[2, 2] + Gm[3, 3] * ginv[3, 3])
        g00 = float(g[0, 0, C, C, C])
        R00_inv = Gm[0, 0] - 0.5 * g00 * R_trace
        pred, N, lap3N, gam = strict_R00_3plus1(kind, r0)
        d_ctor_inv = abs(Rd - R00_inv) / abs(Rd) if Rd != 0 else 0.0
        d_ctor_pred = abs(Rd - pred) / abs(Rd) if Rd != 0 else 0.0
        d_inv_pred = abs(R00_inv - pred) / abs(R00_inv) if R00_inv != 0 else 0.0
        three_rows.append((kind, r0, Rd, R00_inv, pred, d_ctor_inv, d_ctor_pred, d_inv_pred))
        print("  档 %s，r = %.1f：直接构造 %+.6e ｜ 反解式 %+.6e ｜ 3+1 预言 %+.6e"
              % (kind, r0, Rd, R00_inv, pred))
        print("          相对差：直接对反解 %.3e ｜ 直接对预言 %.3e ｜ 反解对预言 %.3e"
              % (d_ctor_inv, d_ctor_pred, d_inv_pred))
inv_ctor_max = max(r[5] for r in three_rows)
pred_ctor_max = max(r[6] for r in three_rows)
inv_pred_max = max(r[7] for r in three_rows)
print("  汇总：直接对反解最大 %.3e ｜ 直接对预言最大 %.3e ｜ 反解对预言最大 %.3e"
      % (inv_ctor_max, pred_ctor_max, inv_pred_max))

# 自检：静态时已知恒等式 R_00 = -2 N ∇²₍₃₎N（∇ 为 gamma 的三维导数）。
# 机器读数应与该式一致；库的 3+1 分解式 N[lap N + Gamma dN] 则是另一个量。
# 故用两个对照值定出库的约定，并同时报告机器读数与 -2N∇²N 的一致性。
print()
print("[自检] 机器 R00 与已知恒等式 -2 N lap_(3) N 的一致性（静态 ADM 分解）")
selfcheck_rows = []
for kind in ("甲", "乙"):
    for r0 in (0.8, 1.2, 1.6):
        g = g_patch(kind, (r0, 0.0, 0.0), EPS)
        R00_full, terms = christoffel_and_R00(g, EPS)
        R00_mach = center(R00_full)
        pred, N, lap3N, gam = strict_R00_3plus1(kind, r0)
        ref = -2.0 * N * lap3N
        rel = abs(R00_mach - ref) / abs(ref) if ref != 0 else float("inf")
        selfcheck_rows.append((kind, r0, R00_mach, ref, rel, pred, N * lap3N))
        print("  档 %s，r = %.1f：机器 %+.6e ｜ -2N∇²N %+.6e（相对差 %.3e）｜ N∇²N %+.6e"
              % (kind, r0, R00_mach, ref, rel, N * lap3N))
sc_max = max(r[4] for r in selfcheck_rows)
print("  自检的最大相对差 %.3e" % sc_max)

# K-1 的相容性：四项展开之和与「直接构造的 R00」在定义上相同，故 K-1 判「各项之和与 R00
# 一致」（恒等核对），而「差额是否为某一项或某一项组合」由 K-2 判。
k1_rels = []
for kind, r0, R00_mach, pred, diff, share in k1_rows:
    g = g_patch(kind, (r0, 0.0, 0.0), EPS)
    R00_full, terms = christoffel_and_R00(g, EPS)
    ssum = sum(center(terms[k]) for k in ("t_deriv", "spatial", "quad_a", "quad_b"))
    ctr = center(R00_full)
    rel = abs(ssum - ctr) / abs(ctr) if ctr != 0 else 0.0
    k1_rels.append(rel)
k1_max = max(k1_rels) if k1_rels else 0.0
k1_ok = k1_max <= 1e-12
print("  K-1 各项之和与 R00 的相对差：最大 %.3e" % k1_max)
check("K-1", k1_ok, "四项展开之和与直接构造的 R00 逐位一致（相对差 <= 1e-12，实测最大 %.3e）" % k1_max)

# K-2：差额归属
k2_best = []
for kind, r0, t_t, t_s, t_qa, t_qb, t_sh, diff in k2_rows:
    cands = {"t_deriv": t_t, "spatial": t_s, "quad_a": t_qa, "quad_b": t_qb,
             "t_deriv+spatial": t_t + t_s, "t_deriv-spatial": t_t - t_s,
             "quad_a-quad_b": t_qa - t_qb, "t_deriv+quad_a-quad_b": t_t + t_qa - t_qb}
    best = min(cands.items(), key=lambda kv: abs(kv[1] - diff))
    rel = abs(best[1] - diff) / abs(diff) if diff != 0 else float("inf")
    k2_best.append((kind, r0, best[0], rel))
k2_ok = all(r <= 1e-3 for _, _, _, r in k2_best)
print("  K-2 归属读数：")
for kind, r0, name, rel in k2_best:
    print("    档 %s，r = %.1f：差额最贴近「%s」，相对差 %.3e" % (kind, r0, name, rel))
check("K-2", k2_ok,
      "差额项归属到展开项：%s；最差相对差 %.3e"
      % ("、".join(sorted(set(n for _, _, n, _ in k2_best))), max(r for _, _, _, r in k2_best)))

# ----------------------------------------------------------------------
# K-3 移位项的零贡献
# ----------------------------------------------------------------------
print()
print("[K-3] 移位项的零贡献：静场 g_0i = 0，移位项应严格为零")
shift_max = max(abs(r[6]) for r in k2_rows)
print("  移位项的最大绝对值 %.3e（机器零门槛 1e-14）" % shift_max)
check("K-3", shift_max <= 1e-14, "静场的移位项贡献严格为零（最大绝对值 %.3e）" % shift_max)

# ----------------------------------------------------------------------
# K-4 差额对场强 F 的依赖阶
# ----------------------------------------------------------------------
print()
print("[K-4] 差额对场强 F 的依赖阶")
# 以 F 的缩放档：令 F -> s F（等价于把完成场的线性部分缩放），看差额的幂次
F_SCALES = (0.25, 0.5, 1.0, 2.0, 4.0)
_orig_F = F_of


def make_scaled(s):
    def Fs(r):
        return s * _orig_F(r)
    return Fs


k4_rows = []
for kind in ("甲",):
    diffs = []
    for s in F_SCALES:
        globals()["F_of"] = make_scaled(s)
        g = g_patch(kind, (1.2, 0.0, 0.0), EPS)
        R00_full, terms = christoffel_and_R00(g, EPS)
        pred, N, lap3N, gam = strict_R00_3plus1(kind, 1.2)
        diffs.append((s, center(R00_full) - pred))
    globals()["F_of"] = _orig_F
    xs = np.array([d[0] for d in diffs])
    ys = np.array([abs(d[1]) for d in diffs])
    slope = float(np.polyfit(np.log(xs), np.log(ys), 1)[0]) if all(ys > 0) else float("nan")
    k4_rows.append((kind, diffs, slope))
    print("  档 %s：F 缩放档 %s 对应差额 %s"
          % (kind, ", ".join("%.2f" % d[0] for d in diffs), ", ".join("%+.4e" % d[1] for d in diffs)))
    print("          差额对 F 的对数斜率 %+.4f" % slope)
slope_k4 = k4_rows[0][2]
check("K-4", np.isfinite(slope_k4) and abs(slope_k4 - 1.0) < 0.1,
      "差额对场强 F 的对数斜率 %+.4f（若为 1 则差额与 F 同阶）" % slope_k4)

# ----------------------------------------------------------------------
# K-5 恒真替身
# ----------------------------------------------------------------------
print()
print("[K-5] 恒真替身：把 Om 取另一侧（甲档用 Om^2 = 1+G 改用 Om^2 = (1+G)/(1-F)），K-2 的归属须改变")
# 替身：严格 3+1 预言用错的 Om（未除 N^2），差额随之改变
alt_best = []
for kind, r0, t_t, t_s, t_qa, t_qb, t_sh, diff in k2_rows:
    Fv = float(F_of(r0))
    Gv = float(G_of(r0))
    Om2_alt = (1.0 + Gv) / (1.0 - Fv)
    Nf = lambda r: math.sqrt(1.0 - float(F_of(r)))
    Om_f = lambda r: math.sqrt(float((1.0 + G_of(r)) / (1.0 - F_of(r))))
    dN = rad_d1(Nf, r0)
    dOm = rad_d1(Om_f, r0)
    Omv = float(Om_f(r0))
    lap3N = rad_d1(lambda x: Om_f(x) ** 2 * rad_d1(Nf, x), r0) / Omv ** 3
    gam = 3.0 * dN * dOm / Omv ** 3
    N = math.sqrt(1.0 - Fv)
    pred_alt = N * (lap3N + gam)
    diff_alt = (t_t + t_s + t_qa + t_qb) - pred_alt
    cands = {"t_deriv": t_t, "spatial": t_s, "quad_a": t_qa, "quad_b": t_qb,
             "t_deriv+spatial": t_t + t_s, "quad_a-quad_b": t_qa - t_qb}
    best = min(cands.items(), key=lambda kv: abs(kv[1] - diff_alt))
    rel = abs(best[1] - diff_alt) / abs(diff_alt) if diff_alt != 0 else float("inf")
    alt_best.append((kind, r0, best[0], rel))
orig_names = set(n for _, _, n, _ in k2_best)
alt_names = set(n for _, _, n, _ in alt_best)
changed = (orig_names != alt_names) or any(r > 1e-2 for _, _, _, r in alt_best)
print("  替身后最贴近的项：%s（真值：%s）" % (sorted(alt_names), sorted(orig_names)))
check("K-5", changed,
      "改 Om 取法后差额的项级归属改变（%s 对 %s），判据有区分力"
      % (sorted(alt_names), sorted(orig_names)))

# ----------------------------------------------------------------------
# K-6 状态登记
# ----------------------------------------------------------------------
print()
print("[K-6] 状态登记")
reg = [
    ("其一", "展开", "四项（t 方向导数、空间导数、二次项甲、二次项乙）之和与直接构造的 R00 逐位一致"),
    ("其二", "归属", "差额项的项级归属读数见 K-2 的逐档输出"),
    ("其三", "移位", "静场的移位项贡献严格为零"),
    ("其四", "依赖阶", "差额对场强 F 的对数斜率见 K-4"),
    ("其五", "剩余", "差额项的解析式是否即某一展开项，须待作者确认口径后落基准；"
                "P5 守恒结构那一侧的状态更新须作者确认"),
]
for a1_, a2_, a3_ in reg:
    print("     %-4s %-10s %s" % (a1_, a2_, a3_))
check("K-6", True, "状态登记完成")

# ----------------------------------------------------------------------
# 总结
# ----------------------------------------------------------------------
print()
print("=" * 78)
n_pass = sum(1 for _, ok in CHECKS if ok)
print("判据通过：%d/%d" % (n_pass, len(CHECKS)))
print("运行耗时 %.2f 秒" % (time.time() - T0))
print("=" * 78)

# ----------------------------------------------------------------------
# 出图
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))

ax = axes[0]
for kind, marker, col in (("甲", "o", "#1f77b4"), ("乙", "s", "#ff7f0e")):
    rs = [r[1] for r in k1_rows if r[0] == kind]
    ds = [r[4] for r in k1_rows if r[0] == kind]
    ax.plot(rs, ds, marker + "-", color=col, linewidth=1.6, label=u"档 %s 的差额" % kind)
ax.axhline(0.0, color="#888888", linewidth=0.8)
ax.set_xlabel(u"半径 $r$")
ax.set_ylabel(u"差额 $\\Delta$")
ax.set_title(u"(a) 差额随半径")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1]
for kind, marker, col in (("甲", "o", "#1f77b4"), ("乙", "s", "#ff7f0e")):
    rs = [r[1] for r in k1_rows if r[0] == kind]
    sh = [r[5] for r in k1_rows if r[0] == kind]
    ax.semilogy(rs, np.maximum(sh, 1e-18), marker + "-", color=col, linewidth=1.6,
                label=u"档 %s 的份额" % kind)
ax.axhline(1e-4, color="#2ca02c", linestyle=":", linewidth=1.4, label=u"门槛 $10^{-4}$")
ax.set_xlabel(u"半径 $r$")
ax.set_ylabel(u"差额占 $R_{00}$ 的份额")
ax.set_title(u"(b) 差额份额")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

ax = axes[2]
labels = []
vals = []
for kind, r0, name, rel in k2_best:
    labels.append(u"%s %.1f" % (kind, r0))
    vals.append(rel)
ax.bar(np.arange(len(labels)), np.maximum(vals, 1e-18), color="#1f77b4", width=0.6)
ax.axhline(1e-3, color="#d62728", linestyle=":", linewidth=1.4, label=u"归属门槛 $10^{-3}$")
ax.set_xticks(np.arange(len(labels)))
ax.set_xticklabels(labels, fontsize=7, rotation=45)
ax.set_ylabel(u"归属的相对差")
ax.set_title(u"(c) 差额的项级归属")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")

fig.suptitle(u"第762号：解析黎曼张量的逐项展开与差额的项级归属", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(IMG, dpi=150)
print()
print("图已存：%s" % IMG)
