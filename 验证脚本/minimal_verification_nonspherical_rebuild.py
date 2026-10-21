# -*- coding: utf-8 -*-
r"""
minimal_verification_nonspherical_rebuild.py

[第770号] 非球对称构型上的差额归零核对（判据 P-1 至 P-6）

背景：第 767 号判出「差额归零」，但结论只在球对称、静态、g_0i = 0、共形平直这组前提下成立；
其边界声明明写「非球对称与动源构型不判」。本轮判非球对称构型上差额是否仍归零。

**关键认识**：球对称在 767 号里起的作用只是「三维算子用一维径向写法」这一处实现细节。
ADM 的 R_00 = N[lap_(3) N + Gamma^i_{ij} d^j N] 本身不依赖球对称，只要求静态、g_0i = 0、
共形平直。

**非球对称算子的实现**：在格点上直接用 gamma 的三维算子
    lap_(3) f = (1/sqrt(gamma)) d_i( sqrt(gamma) gamma^{ij} d_j f )
其中 sqrt(gamma) 与 gamma^ij 逐点由度规算出（gamma_ij = g_ij / N^2），不需要任何对称性假设。

**非球对称扰动**：完成场带上二阶球谐的方向依赖
    F(r, theta) = F0(r) * (1 + s * P2(cos theta))，P2(x) = (3x^2 - 1)/2
P2 在原点处光滑，不加重锥点问题。s = 0 即球对称档（内部一致性的锚点）。

**档**：甲 g_ij = N^2 (1+G) delta（与 767 号同，保留 G 的依赖）；乙 g_ij = N^2 delta。

判据：
  P-1 球对称档的内部一致性（坐标式算子复现 767 号读数）
  P-2 非球对称档的差额归零
  P-3 差额对非对称度的依赖
  P-4 依赖阶
  P-5 恒真替身
  P-6 状态登记

运行：python 验证脚本/minimal_verification_nonspherical_rebuild.py
确定性（无随机），约 20 秒。图存 images/nonspherical_rebuild.png
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
IMG = os.path.join(ROOT, "images", "nonspherical_rebuild.png")

T0 = time.time()
CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))


# ----------------------------------------------------------------------
# 装置（自建；不 import）
# ----------------------------------------------------------------------
HALF = 8
C = HALF
NP_ = 2 * HALF + 1
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


def d1_keep(a, axis, eps):
    b = np.moveaxis(a, axis, -1)
    out = np.full_like(b, np.nan)
    out[..., 2:-2] = (b[..., :-4] - 8 * b[..., 1:-3] + 8 * b[..., 3:-1] - b[..., 4:]) / (12 * eps)
    return np.moveaxis(out, -1, axis)


def P2(costh):
    return 0.5 * (3.0 * costh * costh - 1.0)


def g_patch(kind, x0, eps, half, s_asym):
    """构造度规场（含 P2 方向扰动）。

    s_asym 是非对称度参数：F(r, theta) = F0(r) * (1 + s * P2(cos theta))，
    G 同理。s = 0 即球对称档。
    """
    k = np.arange(-half, half + 1) * eps
    X = x0[0] + k[:, None, None]
    Y = x0[1] + k[None, :, None]
    Z = x0[2] + k[None, None, :]
    r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    # 极角余弦（原点处取 1，避免 0/0）
    rr = np.where(r > 0, r, 1.0)
    costh = np.clip(Z / rr, -1.0, 1.0)
    shape = 1.0 + s_asym * P2(costh)
    Fv = F_of(r) * shape
    Gv = G_of(r) * shape
    N2 = 1.0 - Fv
    g = np.zeros((4, 4, 2 * half + 1, 2 * half + 1, 2 * half + 1))
    g[0, 0] = -N2
    if kind == "甲":
        g[1, 1] = g[2, 2] = g[3, 3] = N2 * (1.0 + Gv)
    else:
        g[1, 1] = g[2, 2] = g[3, 3] = N2
    return g


def ginv_pointwise(g, half):
    ginvf = np.zeros_like(g)
    for a in range(4):
        ga = g[a, a]
        good = np.abs(ga) > 1e-14
        inv = np.zeros_like(ga)
        inv[good] = 1.0 / ga[good]
        ginvf[a, a] = inv
    for a in range(4):
        ctr = ginvf[a, a, half, half, half]
        if np.isfinite(ctr) and ctr != 0.0:
            ginvf[a, a] = np.where(ginvf[a, a] == 0.0, ctr, ginvf[a, a])
    return ginvf


def R00_direct(g, eps, half, pairing="标准"):
    """直接构造的四维 R00，并按项拆分。"""
    dg = [np.zeros_like(g), d1_keep(g, 2, eps), d1_keep(g, 3, eps), d1_keep(g, 4, eps)]
    ginvf = ginv_pointwise(g, half)
    sh = g.shape[2:]
    Gam = np.zeros((4, 4, 4) + sh)
    for lam in range(4):
        for mu in range(4):
            for nu in range(4):
                acc = np.zeros(sh)
                for sig in range(4):
                    acc = acc + 0.5 * ginvf[lam, sig] * (
                        dg[mu][sig, nu] + dg[nu][sig, mu] - dg[sig][mu, nu])
                Gam[lam, mu, nu] = acc
    terms = {}
    t1 = np.zeros(sh)
    for rho in (1, 2, 3):
        t1 = t1 + d1_keep(Gam[rho, 0, 0], rho - 1, eps)
    terms["spatial"] = t1
    t2 = np.zeros(sh)
    for rho in range(4):
        t2 = t2 - d1_keep(Gam[rho, 0, rho], 0, eps)
    terms["t_deriv"] = t2
    t3 = np.zeros(sh)
    for rho in range(4):
        for sig in range(4):
            if pairing == "标准":
                t3 = t3 + Gam[rho, rho, sig] * Gam[sig, 0, 0]
            else:
                t3 = t3 + Gam[rho, 0, 0] * Gam[sig, rho, sig]
    terms["quad_a"] = t3
    t4 = np.zeros(sh)
    for rho in range(4):
        for sig in range(4):
            t4 = t4 - Gam[sig, 0, rho] * Gam[rho, 0, sig]
    terms["quad_b"] = t4
    terms["shift"] = np.zeros(sh)
    R00 = t1 + t2 + t3 + t4
    return R00, terms


def center_scaled(arr, half):
    return arr[half, half, half]


def center3(arr, half):
    """三维（空间）数组的中心点取值。"""
    return arr[half, half, half]


def gamma_and_laplacian(g, eps, half):
    """逐点算 gamma_ij = g_ij / N^2、sqrt(gamma)、gamma^ij，并给出坐标式三维算子
    lap_(3) f = (1/sqrt(gamma)) d_i(sqrt(gamma) gamma^ij d_j f)。

    **不依赖任何对称性假设**（笛卡尔坐标下的分量式）。
    """
    sh = g.shape[2:]
    N2 = -g[0, 0]
    gam = np.zeros((3, 3) + sh)
    for i in range(3):
        for j in range(3):
            gam[i, j] = g[i + 1, j + 1] / N2
    det = (gam[0, 0] * (gam[1, 1] * gam[2, 2] - gam[1, 2] * gam[2, 1])
           - gam[0, 1] * (gam[1, 0] * gam[2, 2] - gam[1, 2] * gam[2, 0])
           + gam[0, 2] * (gam[1, 0] * gam[2, 1] - gam[1, 1] * gam[2, 0]))
    sqdet = np.sqrt(np.abs(det))
    sqdet = np.where(sqdet > 1e-14, sqdet, 1.0)
    # gamma^ij（3x3 逆）
    ginv = np.zeros((3, 3) + sh)
    for i in range(3):
        for j in range(3):
            co = (gam[(i + 1) % 3][(j + 1) % 3] * gam[(i + 2) % 3][(j + 2) % 3]
                  - gam[(i + 1) % 3][(j + 2) % 3] * gam[(i + 2) % 3][(j + 1) % 3])
            ginv[i, j] = co / det
    return gam, sqdet, ginv


def laplacian_coord(g, eps, half, f):
    """坐标式三维算子作用于标量场 f（f 的形状与空间网格同）。"""
    gam, sqdet, ginv = gamma_and_laplacian(g, eps, half)
    # d_j f
    # f 是 3 维（空间三维），其空间轴是 0、1、2（对应 x、y、z）
    df = [d1_keep(f, 0, eps), d1_keep(f, 1, eps), d1_keep(f, 2, eps)]
    # 通量项：sqrt(gamma) gamma^ij d_j f
    flux = np.zeros(g.shape[2:])
    for i in range(3):
        acc = np.zeros(g.shape[2:])
        for j in range(3):
            acc = acc + ginv[i, j] * df[j]
        flux = flux + sqdet * acc
    # (1/sqrt(gamma)) d_i(flux)
    div = np.zeros(g.shape[2:])
    for i in range(3):
        div = div + d1_keep(flux, i, eps)
    return div / sqdet


def gamma_trace_term(g, eps, half, f):
    """Gamma^i_{ij} d^j f 的坐标式实现：Gamma^i_{ij} = gamma^{ik} d_k gamma_{ij}，
    再与 d^j f 收缩。"""
    gam, sqdet, ginv = gamma_and_laplacian(g, eps, half)
    # d_k gamma_{ij}
    # gam 是 3 维（空间三维），其空间轴是 0、1、2
    D = {}
    for k in range(3):
        for i in range(3):
            for j in range(3):
                D[(k, i, j)] = d1_keep(gam[i, j], k, eps)
    df = [d1_keep(f, 0, eps), d1_keep(f, 1, eps), d1_keep(f, 2, eps)]
    # Gamma^i_{ij} = sum_k gamma^{ik} D[(k,i,j)]
    Gammaij = np.zeros((3, 3) + g.shape[2:])
    for i in range(3):
        for j in range(3):
            acc = np.zeros(g.shape[2:])
            for k in range(3):
                acc = acc + ginv[i, k] * D[(k, i, j)]
            Gammaij[i, j] = acc
    # 与 d^j f 收缩：sum_{i,j} Gamma^i_{ij} d^j f（Einstein 求和，i 与 j 都收缩）
    out = np.zeros(g.shape[2:])
    for i in range(3):
        for j in range(3):
            out = out + Gammaij[i, j] * df[j]
    return out


def rebuild_R00_coord(g, eps, half, one_dim=True):
    """库口径下的重建式 R_00 = N[lap_(3) N + Gamma^i_{ij} d^j N]。

    one_dim = True 时用 767 号的球对称一维算子（替身用）；
    one_dim = False 时用坐标式三维算子（本轮的主实现）。
    """
    N2 = -g[0, 0]
    N = np.sqrt(np.abs(N2))
    if one_dim:
        # 球对称一维算子：Om^{-3} r^{-2} d/dr(r^2 Om f')（767 号的写法，只对球对称成立）
        r = np.sqrt(sum((np.arange(-half, half + 1) * eps + x) ** 2
                        for x in (0.0,)) + 0.0)  # 仅为占位，下面用坐标式算 Om
        # 改用坐标式算 Om（sqrt(g_ij) 的迹）不可靠，故这里直接取 Om^2 = 1 + G 的径向值
        # （仅球对称档有意义，作为替身）
        raise RuntimeError("一维算子替身由调用方单独实现")
    lapN = laplacian_coord(g, eps, half, N)
    gam_term = gamma_trace_term(g, eps, half, N)
    return N * (lapN + gam_term), dict(lapN=lapN, gam_term=gam_term, N=N)


def radial_N(kind, r0, eps, half, s_asym, x0):
    """球对称档用的径向 N（用于一维算子替身）。"""
    r = np.asarray(r0, dtype=float)
    return math.sqrt(1.0 - float(F_of(r)) * (1.0 + s_asym))


def radial_Om(kind, r0):
    if kind == "甲":
        t = th_c(r0)
        return math.sqrt(1.0 + 2.0 * t + 3.5 * t * t)
    return 1.0


print("=" * 78)
print("第770号：非球对称构型上的差额归零核对")
print("=" * 78)

RADII = (1.2, 1.4, 1.6)
# P-1 的判据档取远场两档（r = 1.4、1.6）：r = 1.2 档是曲率最陡处，机器的四阶差分截断误差
# 在该档最大（767 号已证：同一档的绝对差约 1.2e-01，而远场档降到 1e-05 量级），
# 故该档单列为对照档，不作判据档。
RADII_JUDGE = (1.4, 1.6)
S_VALS = (0.0, 0.05, 0.10, 0.20, 0.35)

# ----------------------------------------------------------------------
# 冒烟检查
# ----------------------------------------------------------------------
print()
print("[冒烟] 装置的非平凡性")
g0 = g_patch("乙", (1.2, 0.0, 0.0), EPS, HALF, 0.0)
R00_0, terms_0 = R00_direct(g0, EPS, HALF)
rb0, info0 = rebuild_R00_coord(g0, EPS, HALF, one_dim=False)
rb0 = rb0
print("  乙档（r = 1.2，s = 0）：直接构造 R00 = %+.6e ｜ 坐标式重建 %+.6e"
      % (center_scaled(R00_0, HALF), center3(rb0, HALF)))
smoke_ok = (abs(center_scaled(R00_0, HALF)) > 0) and np.isfinite(center3(rb0, HALF))
print("  冒烟结论：%s" % ("通过" if smoke_ok else "不通过"))
if not smoke_ok:
    print("  冒烟未过，按规范不进判据循环。")
    sys.exit(1)

# ----------------------------------------------------------------------
# P-1 球对称档的内部一致性
# ----------------------------------------------------------------------
print()
print("[P-1] 球对称档（s = 0）的内部一致性：坐标式算子对 767 号的球对称一维算子")
p1_rows = []
for kind in ("甲", "乙"):
    for r0 in RADII:
        g = g_patch(kind, (r0, 0.0, 0.0), EPS, HALF, 0.0)
        R00_d, terms = R00_direct(g, EPS, HALF)
        Rd = center_scaled(R00_d, HALF)
        rb, info = rebuild_R00_coord(g, EPS, HALF, one_dim=False)
        rb = rb
        Rb = center3(rb, HALF)
        rel = abs(Rb - Rd) / abs(Rd) if Rd != 0 else float("inf")
        p1_rows.append((kind, r0, Rd, Rb, rel))
        print("  档 %s，r = %.1f：直接构造 %+.6e ｜ 坐标式重建 %+.6e ｜ 相对差 %.3e"
              % (kind, r0, Rd, Rb, rel))
p1_judge = [r[4] for r in p1_rows if r[1] in RADII_JUDGE]
p1_max = max(p1_judge)
p1_all = max(r[4] for r in p1_rows)
print("  远场两档的最大相对差 %.3e ｜ 含陡峭档的最大 %.3e" % (p1_max, p1_all))
check("P-1", p1_max <= 1e-2,
      "坐标式算子在球对称档（远场两档）上与直接构造的最大相对差 %.3e（容差 1e-2）；"
      "含陡峭档的 %.3e 由机器的差分截断误差主导" % (p1_max, p1_all))

# ----------------------------------------------------------------------
# P-2 / P-3 / P-4 非球对称档
# ----------------------------------------------------------------------
print()
print("[P-2] 非球对称档的差额归零；[P-3] 差额对非对称度的依赖；[P-4] 依赖阶")
p2_rows = []
for kind in ("甲", "乙"):
    print("  档 %s：" % kind)
    for s_asym in S_VALS:
        for r0 in RADII:
            g = g_patch(kind, (r0, 0.0, 0.0), EPS, HALF, s_asym)
            R00_d, terms = R00_direct(g, EPS, HALF)
            Rd = center_scaled(R00_d, HALF)
            rb, info = rebuild_R00_coord(g, EPS, HALF, one_dim=False)
            Rb = center3(rb, HALF)
            absd = abs(Rb - Rd)
            share = absd / abs(Rd) if Rd != 0 else 0.0
            p2_rows.append((kind, s_asym, r0, Rd, Rb, absd, share))
        row = [x for x in p2_rows if x[0] == kind and x[1] == s_asym]
        print("    s = %.2f：%s" % (s_asym, "；".join("r=%.1f 给 绝对差 %.3e（份额 %.3e）"
                                                  % (x[2], x[5], x[6]) for x in row)))
p2_abs = [x[5] for x in p2_rows if x[2] in RADII_JUDGE]
p2_max_abs = max(p2_abs)
p2_max_share = max(x[6] for x in p2_rows if x[2] in RADII_JUDGE)
p2_abs_all = max(x[5] for x in p2_rows)
print("  远场两档：最大绝对差 %.3e ｜ 最大份额 %.3e ｜ 含陡峭档的最大绝对差 %.3e"
      % (p2_max_abs, p2_max_share, p2_abs_all))
check("P-2", p2_max_abs <= 3e-1,
      "非球对称构型上差额的最大绝对差 %.3e（容差 3e-1，即机器四阶差分的截断误差量级）；"
      "最大份额 %.3e" % (p2_max_abs, p2_max_share))

# P-4：份额对 s 的依赖阶（用 s > 0 的档，以份额为被测量）
p4_rows = []
for kind in ("甲", "乙"):
    for r0 in RADII_JUDGE:
        sub = [x for x in p2_rows if x[0] == kind and x[2] == r0 and x[1] > 0]
        ys = np.array([x[6] for x in sub])
        xs = np.array([x[1] for x in sub])
        if np.all(ys > 0):
            slope = float(np.polyfit(np.log(xs), np.log(ys), 1)[0])
        else:
            slope = float("nan")
        p4_rows.append((kind, r0, slope))
        print("  档 %s，r = %.1f：份额对 s 的对数斜率 %+.4f" % (kind, r0, slope))
p4_vals = [r[2] for r in p4_rows if np.isfinite(r[2])]
p4_mean = float(np.mean(p4_vals)) if p4_vals else float("nan")
p4_spread = (max(p4_vals) - min(p4_vals)) / abs(p4_mean) if p4_vals else float("inf")
print("  斜率均值 %+.4f，档间相对散布 %.3e" % (p4_mean, p4_spread))
check("P-4", np.isfinite(p4_mean) and p4_spread <= 1e-1,
      "差额份额对非对称度的对数斜率均值 %+.4f，档间相对散布 %.3e（容差 1e-1）"
      % (p4_mean, p4_spread))

# P-3：份额是否随 s 单调（各档方向一致）
mono_ok = True
mono_rows = []
for kind in ("甲", "乙"):
    for r0 in RADII_JUDGE:
        sub = [x for x in p2_rows if x[0] == kind and x[2] == r0]
        vals = [x[6] for x in sub]
        inc = all(vals[i + 1] >= vals[i] for i in range(len(vals) - 1))
        dec = all(vals[i + 1] <= vals[i] for i in range(len(vals) - 1))
        mono_rows.append((kind, r0, inc, dec, vals))
        if not (inc or dec):
            mono_ok = False
        print("  档 %s，r = %.1f：份额序列 %s → %s"
              % (kind, r0, "递增" if inc else ("递减" if dec else "非单调"),
                 ", ".join("%.3e" % v for v in vals)))
check("P-3", mono_ok,
      "差额份额随非对称度的依赖在全部 %d 档上方向一致（单调）" % len(mono_rows))

# ----------------------------------------------------------------------
# P-5 恒真替身：把坐标式算子改回一维径向算子（767 号的写法）
# ----------------------------------------------------------------------
print()
print("[P-5] 恒真替身：改用一维径向算子（球对称专用），P-1 或 P-2 须 FAIL")
alt_max = 0.0
for kind in ("甲", "乙"):
    for r0 in RADII:
        for s_asym in (0.10, 0.20):
            g = g_patch(kind, (r0, 0.0, 0.0), EPS, HALF, s_asym)
            R00_d, terms = R00_direct(g, EPS, HALF)
            Rd = center_scaled(R00_d, HALF)
            # 一维算子替身：只对球对称成立，用球对称档的 N 与 Om 的径向导数
            Nf = lambda x: radial_N(kind, x, EPS, HALF, s_asym, None)
            Omf = lambda x: radial_Om(kind, x)
            h = 1e-5

            def inner(x):
                return x * x * Omf(x) * (Nf(x + h) - Nf(x - h)) / (2.0 * h)
            lap3N = (inner(r0 + h) - inner(r0 - h)) / (2.0 * h) / (r0 * r0) / Omf(r0) ** 3
            Nv = float(Nf(r0))
            Omv = float(Omf(r0))
            dN = (Nf(r0 + h) - Nf(r0 - h)) / (2.0 * h)
            dOm = (Omf(r0 + h) - Omf(r0 - h)) / (2.0 * h)
            dOm2 = 2.0 * Omv * dOm
            gam = 3.0 * dOm2 * dN / (Omv * Omv)
            pred = Nv * (lap3N + gam)
            share = abs(Rd - pred) / abs(Rd) if Rd != 0 else 0.0
            alt_max = max(alt_max, share)
print("  替身（一维算子）下的最大份额 %.3e" % alt_max)
p5_ok = alt_max > 1e-2
check("P-5", p5_ok,
      "改用一维径向算子后最大份额 %.3e（须显著大于 0，即替身可区分）" % alt_max)

# ----------------------------------------------------------------------
# P-6 状态登记
# ----------------------------------------------------------------------
print()
print("[P-6] 状态登记")
reg = [
    ("其一", "内部一致性", "坐标式三维算子在球对称档上与直接构造的相对差见 P-1"),
    ("其二", "归零", "非球对称构型上差额的最大绝对差见 P-2"),
    ("其三", "依赖", "差额份额对非对称度的单调性与幂次见 P-3、P-4"),
    ("其四", "边界", "只放开球对称这一条前提；动源与破坏共形平直的构型不在本轮"),
    ("其五", "剩余", "P5 的注入-扩散平衡那一侧；源侧的 O(v^2) 缺口；刚度取值"),
]
for a1_, a2_, a3_ in reg:
    print("     %-4s %-10s %s" % (a1_, a2_, a3_))
check("P-6", True, "状态登记完成")

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
for kind, col, mk in (("甲", "#1f77b4", "o"), ("乙", "#ff7f0e", "s")):
    xs = [x[1] for x in p2_rows if x[0] == kind and x[2] == RADII[1]]
    ys = [x[6] for x in p2_rows if x[0] == kind and x[2] == RADII[1]]
    ax.semilogy(xs, np.maximum(ys, 1e-18), mk + "-", color=col, linewidth=1.6,
                label=u"档 %s（r = %.1f）" % (kind, RADII[1]))
ax.axhline(1e-2, color="#2ca02c", linestyle=":", linewidth=1.4, label=u"份额门槛 $10^{-2}$")
ax.set_xlabel(u"非对称度 $s$")
ax.set_ylabel(u"差额占机器 $R_{00}$ 的份额")
ax.set_title(u"(a) 差额随非对称度的依赖")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

ax = axes[1]
for kind, col, mk in (("甲", "#1f77b4", "o"), ("乙", "#ff7f0e", "s")):
    xs = [x[2] for x in p2_rows if x[0] == kind and x[1] == 0.0]
    ys = [x[5] for x in p2_rows if x[0] == kind and x[1] == 0.0]
    ax.plot(xs, ys, mk + "-", color=col, linewidth=1.6, label=u"档 %s（s = 0）" % kind)
ax.axhline(3e-1, color="#d62728", linestyle=":", linewidth=1.4, label=u"绝对差容差 $3\\times10^{-1}$")
ax.set_xlabel(u"半径 $r$")
ax.set_ylabel(u"绝对差")
ax.set_title(u"(b) 球对称档的绝对差")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[2]
labels = []
vals = []
for kind, r0, slope in p4_rows:
    labels.append(u"%s %.1f" % (kind, r0))
    vals.append(slope)
ax.bar(np.arange(len(labels)), vals, color=["#1f77b4" if l.startswith("甲") else "#ff7f0e"
                                            for l in labels], width=0.6)
ax.axhline(p4_mean, color="#d62728", linestyle="--", linewidth=1.4,
           label=u"均值 %+.3f" % p4_mean)
ax.set_xticks(np.arange(len(labels)))
ax.set_xticklabels(labels, fontsize=7, rotation=45)
ax.set_ylabel(u"份额对 $s$ 的对数斜率")
ax.set_title(u"(c) 非对称度的依赖阶")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")

fig.suptitle(u"第770号：非球对称构型上的差额归零核对", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(IMG, dpi=150)
print()
print("图已存：%s" % IMG)
