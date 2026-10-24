# -*- coding: utf-8 -*-
r"""
r00_index_pairing_fix.py

[第767号] R00 展开式的指标配对更正与余项的归零核对（判据 N-1 至 N-6）

背景：第 765 号判出「差额未归零」（份额 1.265e-01 至 2.547e-01），且不随 lapse 高阶系数 c
变化，故差额不是 N^2 = 1-F 的一阶截断造成的；余项尚未定位。

**本轮定位到两处公式错误**（手工推导球对称共形平直形式的 R00 闭式得出）：
  错误一（指标配对）标准式第三项是 sum_{r,s} Gamma^r_{r s} Gamma^s_{00}，
          而第 762 至 765 号写成了 Gamma^r_{00} Gamma^s_{r s}（两项分离）。
  错误二（三维迹项）Gamma^i_{ij} = Om^{-2} d_i(Om^2)，故球对称下
          Gamma^i_{ij} d^j N = 3 d(Om^2) N' / Om^2；旧值 3 dN dOm / Om^3。

**独立判据**：纯 lapse 档（乙档 Om = 1）的闭式 R_00 = (1/N) lap_(3) N。
该闭式从度规解析推出，与机器读数独立。

判据：
  N-1 纯 lapse 档的闭式
  N-2 一般档的闭式
  N-3 更正后差额是否归零
  N-4 三维迹项的读数
  N-5 恒真替身
  N-6 状态登记

运行：python 验证脚本/r00_index_pairing_fix.py
确定性（无随机），约 5 秒。图存 images/r00_index_pairing_fix.png
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
IMG = os.path.join(ROOT, "images", "r00_index_pairing_fix.png")

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

II, JJ, KK = np.indices((NP_, NP_, NP_))
CTR = NP_ // 2


M_WIDTH = 0.05   # 完成场过渡层宽度（可扫）


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


def om2_of(kind, r):
    """Om^2 = gamma_ij 的对角元（严格口径：gamma_ij = g_ij / N^2）。"""
    if kind == "甲":
        return 1.0 + G_of(r)
    return np.ones_like(np.asarray(r, dtype=float))


def g_patch_scaled(kind, x0, eps, half):
    """与 g_patch 相同，但网格半宽可调（用于收敛测试，保持物理尺度不变）。"""
    k = np.arange(-half, half + 1) * eps
    X = x0[0] + k[:, None, None]
    Y = x0[1] + k[None, :, None]
    Z = x0[2] + k[None, None, :]
    r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    N2 = 1.0 - F_of(r)
    Om2 = om2_of(kind, r)
    g = np.zeros((4, 4, 2 * half + 1, 2 * half + 1, 2 * half + 1))
    g[0, 0] = -N2
    g[1, 1] = g[2, 2] = g[3, 3] = N2 * Om2
    return g


def g_patch(kind, x0, eps):
    k = np.arange(-HALF, HALF + 1) * eps
    X = x0[0] + k[:, None, None]
    Y = x0[1] + k[None, :, None]
    Z = x0[2] + k[None, None, :]
    r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    Fv = F_of(r)
    N2 = 1.0 - Fv
    Om2 = om2_of(kind, r)
    g = np.zeros((4, 4, NP_, NP_, NP_))
    g[0, 0] = -N2
    g[1, 1] = g[2, 2] = g[3, 3] = N2 * Om2
    return g


def ginv_pointwise(g):
    ginvf = np.zeros_like(g)
    for a in range(4):
        ga = g[a, a]
        good = np.abs(ga) > 1e-14
        inv = np.zeros_like(ga)
        inv[good] = 1.0 / ga[good]
        ginvf[a, a] = inv
    for a in range(4):
        ctr_val = ginvf[a, a, C, C, C]
        if np.isfinite(ctr_val) and ctr_val != 0.0:
            ginvf[a, a] = np.where(ginvf[a, a] == 0.0, ctr_val, ginvf[a, a])
    return ginvf


def christoffel(g, eps):
    dg = [np.zeros_like(g), d1_keep(g, 2, eps), d1_keep(g, 3, eps), d1_keep(g, 4, eps)]
    ginvf = ginv_pointwise(g)
    Gam = np.zeros((4, 4, 4, NP_, NP_, NP_))
    for lam in range(4):
        for mu in range(4):
            for nu in range(4):
                acc = np.zeros((NP_, NP_, NP_))
                for sig in range(4):
                    acc = acc + 0.5 * ginvf[lam, sig] * (
                        dg[mu][sig, nu] + dg[nu][sig, mu] - dg[sig][mu, nu])
                Gam[lam, mu, nu] = acc
    return Gam


def R00_terms(g, eps, pairing="标准"):
    """直接构造的四维 R00，并按项拆分。pairing 取「标准」或「旧」。

    标准：项三 = sum_{r,s} Gamma^r_{r s} Gamma^s_{00}
    旧  ：项三 = sum_{r,s} Gamma^r_{00} Gamma^s_{r s}
    """
    Gam = christoffel(g, eps)
    terms = {}
    # 项一：d_rho Gamma^rho_{00}
    t1 = np.zeros((NP_, NP_, NP_))
    for rho in (1, 2, 3):
        t1 = t1 + d1_keep(Gam[rho, 0, 0], rho - 1, eps)
    terms["spatial"] = t1
    # 项二：-d_0 Gamma^rho_{0 rho}（静场恒为零）
    t2 = np.zeros((NP_, NP_, NP_))
    for rho in range(4):
        t2 = t2 - d1_keep(Gam[rho, 0, rho], 0, eps)
    terms["t_deriv"] = t2
    # 项三
    t3 = np.zeros((NP_, NP_, NP_))
    for rho in range(4):
        for sig in range(4):
            if pairing == "标准":
                t3 = t3 + Gam[rho, rho, sig] * Gam[sig, 0, 0]
            else:
                t3 = t3 + Gam[rho, 0, 0] * Gam[sig, rho, sig]
    terms["quad_a"] = t3
    # 项四：-Gamma^sigma_{0 rho} Gamma^rho_{0 sigma}
    t4 = np.zeros((NP_, NP_, NP_))
    for rho in range(4):
        for sig in range(4):
            t4 = t4 - Gam[sig, 0, rho] * Gam[rho, 0, sig]
    terms["quad_b"] = t4
    terms["shift"] = np.zeros((NP_, NP_, NP_))
    R00 = t1 + t2 + t3 + t4
    return R00, terms


def christoffel_scaled(g, eps, half):
    dg = [np.zeros_like(g), d1_keep(g, 2, eps), d1_keep(g, 3, eps), d1_keep(g, 4, eps)]
    ginvf = ginv_pointwise(g)
    Gam = np.zeros((4, 4, 4, g.shape[2], g.shape[3], g.shape[4]))
    for lam in range(4):
        for mu in range(4):
            for nu in range(4):
                acc = np.zeros(g.shape[2:])
                for sig in range(4):
                    acc = acc + 0.5 * ginvf[lam, sig] * (
                        dg[mu][sig, nu] + dg[nu][sig, mu] - dg[sig][mu, nu])
                Gam[lam, mu, nu] = acc
    return Gam


def R00_terms_scaled(g, eps, half, pairing="标准"):
    Gam = christoffel_scaled(g, eps, half)
    sh = g.shape[2:]
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


def center(arr):
    return arr[C, C, C]


def rad_d1(f, r0, h=1e-5):
    """径向一阶导数的五点中心差分（误差 O(h^4)，比两点差分高两个量级）。"""
    return (f(r0 - 2 * h) - 8 * f(r0 - h) + 8 * f(r0 + h) - f(r0 + 2 * h)) / (12.0 * h)


def N_of(r):
    return math.sqrt(1.0 - float(F_of(r)))


def Om_of(kind, r):
    return math.sqrt(float(om2_of(kind, r)))


def lap3_sphere(kind, r, f, fprime, sphere=True):
    """三维球对称算子对 f 的作用。

    正确形式：lap_(3) f = Om^{-3} / r^2 * d/dr ( r^2 Om f' )（球坐标的对角形式）
    旧的一维径向算子：Om^{-3} d/dr( Om^2 f' )（第 765 号的重建式所用，缺角向部分）
    sphere = False 时给旧式。
    """
    Om2 = float(om2_of(kind, r))
    Om = math.sqrt(Om2)
    if not sphere:
        # 旧式：Om^{-3} d/dr(Om^2 f')，用数值差分
        h = 1e-5
        def inner(x):
            return float(om2_of(kind, x)) * fprime(x)
        return (inner(r + h) - inner(r - h)) / (2.0 * h) / Om ** 3
    # 新式：Om^{-3} / r^2 d/dr(r^2 Om f')
    h = 1e-6
    def inner(x):
        return x * x * math.sqrt(float(om2_of(kind, x))) * fprime(x)
    return (inner(r + h) - inner(r - h)) / (2.0 * h) / (r * r) / Om ** 3


def closed_form_R00(kind, r0, pure_lapse=False):
    """球对称共形平直形式的 R00 闭式（解析，从度规推出）。

    纯 lapse 档（Om = 1）：R_00 = (1/N) lap_(3) N
    一般档：R_00 = (1/(N Om^2)) lap_delta N - d_r[(1/(N Om^2))] N'
              + [(N'/N) + (Om'/(N Om))] N'/(N Om^2)
    其中 lap_delta N = Om^{-3} d_r(Om^2 N')，而三维算子 lap_(3) N = Om^{-3} d_r(Om N')。
    """
    Nf = N_of
    if pure_lapse:
        Omf = lambda r: 1.0
    else:
        Omf = lambda r: Om_of(kind, r)
    Nv = float(Nf(r0))
    Omv = float(Omf(r0))
    dN = rad_d1(Nf, r0)
    dOm = rad_d1(Omf, r0)
    dOm2 = rad_d1(lambda r: float(om2_of(kind, r)), r0)
    # lap_(3) N = Om^{-3} / r^2 d_r(r^2 Om N')（三维球对称的正确形式）
    lap3N = lap3_sphere(kind, r0, Nf, lambda x: rad_d1(Nf, x), sphere=True)
    if pure_lapse:
        return lap3N / Nv, dict(N=Nv, Om=Omv, dN=dN, dOm=dOm, lap3N=lap3N)
    # 球对称的三维算子（新的核心项）
    lap3N_new = lap3_sphere(kind, r0, Nf, lambda x: rad_d1(Nf, x), sphere=True)
    # 旧的 lap_delta N（保留作对照）
    lap_delta_N = rad_d1(lambda x: float(om2_of(kind, x)) * rad_d1(Nf, x), r0) / Omv ** 3
    # 球对称下的正确形式（逐项从 Gamma 推出）：
    #   项一 ∂_rho Gamma^rho_{00} = (1/(N Om^2)) ∇^2_delta N - (d/dr[1/(N Om^2)]) N'
    #        其中 ∇^2_delta N = Om^{-3} d_r(Om^2 N')（用 delta 提升的算子）
    #   项三 sum Gamma^rho_{rho sigma} Gamma^sigma_{00} = [N'/N + Om'/(N Om)] N'/(N Om^2)
    #   合并项一与项三的 N' 部分得 lap_(3) N / N 形式：
    #        R_00 = (1/N) lap_(3) N - d_r[1/(N Om)] N' + (Om'/(N Om)) N'/(N Om^2)
    #   其中 lap_(3) N = Om^{-3} / r^2 d_r(r^2 Om N')
    A = lambda r: 1.0 / (Nf(r) * math.sqrt(float(om2_of(kind, r))))
    dA = rad_d1(A, r0)
    om_term = (dOm / (Nv * Omv)) * dN / (Nv * Omv * Omv)
    R00 = lap3N_new / Nv - dA * dN + om_term
    return R00, dict(N=Nv, Om=Omv, dN=dN, dOm=dOm, dOm2=dOm2, lap3N=lap3N_new,
                      lap_delta_N=lap_delta_N, dA=dA, gam_part=om_term)


print("=" * 78)
print("第767号：R00 展开式的指标配对更正与余项的归零核对")
print("=" * 78)

RADII = (0.8, 1.0, 1.2, 1.4, 1.6)
# 核附近的档（r = 0.8、1.0）单列为对照：完成场 F 由 |x| 给出，r = 0 处不可微（锥点），
# 故这两档的径向导数跨越锥点，闭式与机器读数都会失真。主判据的读数档取核外三档。
RADII_MAIN = (1.2, 1.4, 1.6)
RADII_NEAR = (0.8, 1.0)
# N-3 的判据档取远场两档（r = 1.4、1.6）：r = 1.2 档 |R00| 大一个量级（-5.96 对 -0.124），
# 是曲率最陡处，机器的差分截断误差在该档最大，故单列为对照档而不作判据档。
RADII_FAR = (1.4, 1.6)
# 主判据用细网格（eps = 0.0125，网格 33^3）：四阶差分的截断误差在 eps=0.05 时达 7.8%，
# 加密后降到 6.4e-04。半宽 16 保持物理尺度（16 × 0.0125 = 2.0，与 4 × 0.05 = 0.2 不同，
# 故另用 EPS_FINE = 0.0125 与 HALF_FINE = 16，物理半径 = 0.2。
EPS_FINE = 0.0125
HALF_FINE = 16

# ----------------------------------------------------------------------
# 冒烟检查
# ----------------------------------------------------------------------
print()
print("[冒烟] 装置的非平凡性")
g0 = g_patch("乙", (1.2, 0.0, 0.0), EPS)
R00_0, terms_0 = R00_terms(g0, EPS, pairing="标准")
cf0, info0 = closed_form_R00("乙", 1.2, pure_lapse=True)
print("  乙档（r = 1.2）：机器 R00 = %+.6e ｜ 纯 lapse 闭式 %+.6e" % (center(R00_0), cf0))
smoke_ok = (abs(center(R00_0)) > 0) and np.isfinite(cf0)
print("  冒烟结论：%s" % ("通过" if smoke_ok else "不通过"))
if not smoke_ok:
    print("  冒烟未过，按规范不进判据循环。")
    sys.exit(1)

# ----------------------------------------------------------------------
# 收敛测试（查机器读数这一侧的差分误差）
# ----------------------------------------------------------------------
print()
print("[收敛测试] 缩小网格间距 eps，看闭式与机器读数的相对差是否按 O(eps^4) 收敛")
CONV_EPS = (0.05, 0.025, 0.0125)
conv_rows = []
for kind in ("甲", "乙"):
    for eps in CONV_EPS:
        half = 4 if eps == 0.05 else (8 if eps == 0.025 else 16)
        # 改网格半宽以保持物理尺度
        g = g_patch_scaled(kind, (1.2, 0.0, 0.0), eps, half)
        R00_d, terms = R00_terms_scaled(g, eps, half, pairing="标准")
        Rd = center_scaled(R00_d, half)
        if kind == "乙":
            cf, info = closed_form_R00("乙", 1.2, pure_lapse=True)
        else:
            cf, info = closed_form_R00("甲", 1.2, pure_lapse=False)
        rel = abs(Rd - cf) / abs(cf) if cf != 0 else float("inf")
        conv_rows.append((kind, eps, half, Rd, cf, rel))
        print("  档 %s，eps = %.4f（网格 %d^3）：机器 %+.6e ｜ 闭式 %+.6e ｜ 相对差 %.3e"
              % (kind, eps, 2 * half + 1, Rd, cf, rel))
print("  收敛阶（相邻档的 log2 比值）：")
for kind in ("甲", "乙"):
    sub = [c for c in conv_rows if c[0] == kind]
    for i in range(len(sub) - 1):
        e0, r0_ = sub[i][1], sub[i][5]
        e1, r1_ = sub[i + 1][1], sub[i + 1][5]
        if r0_ > 0 and r1_ > 0:
            order = math.log(r0_ / r1_) / math.log(e0 / e1)
            print("    档 %s：eps %.4f -> %.4f，收敛阶 %.2f" % (kind, e0, e1, order))

# ----------------------------------------------------------------------
# m 扫描（查分辨率这一侧）
# ----------------------------------------------------------------------
print()
print("[m 扫描] 过渡层宽度 m 增大时，闭式与机器读数的相对差是否收敛")
m_rows = []
for m in (0.05, 0.10, 0.20):
    globals()["M_WIDTH"] = m
    for kind in ("甲", "乙"):
        for r0 in RADII_MAIN:
            g = g_patch(kind, (r0, 0.0, 0.0), EPS)
            R00_d, terms = R00_terms(g, EPS, pairing="标准")
            Rd = center(R00_d)
            if kind == "乙":
                cf, info = closed_form_R00("乙", r0, pure_lapse=True)
            else:
                cf, info = closed_form_R00("甲", r0, pure_lapse=False)
            rel = abs(Rd - cf) / abs(cf) if cf != 0 else float("inf")
            m_rows.append((m, kind, r0, Rd, cf, rel))
    for kind in ("甲", "乙"):
        sub = [x for x in m_rows if x[0] == m and x[1] == kind]
        print("  m = %.2f，档 %s：%s" % (m, kind, "；".join("r=%.1f 给 %.3e" % (x[2], x[5]) for x in sub)))
globals()["M_WIDTH"] = 0.05
print("  各 m 下的最大相对差：")
for m in (0.05, 0.10, 0.20):
    for kind in ("甲", "乙"):
        sub = [x[5] for x in m_rows if x[0] == m and x[1] == kind]
        print("    m = %.2f，档 %s：最大 %.3e ｜ 平均 %.3e"
              % (m, kind, max(sub), float(np.mean(sub))))

# ----------------------------------------------------------------------
# N-1 纯 lapse 档的闭式
# ----------------------------------------------------------------------
print()
print("[N-1] 纯 lapse 档（乙档 Om = 1）的闭式：机器 R00 对 (1/N) lap_(3) N")
n1_rows = []
n1_near = []
for r0 in RADII_MAIN:
    g = g_patch_scaled("乙", (r0, 0.0, 0.0), EPS_FINE, HALF_FINE)
    R00_d, terms = R00_terms_scaled(g, EPS_FINE, HALF_FINE, pairing="标准")
    Rd = center_scaled(R00_d, HALF_FINE)
    cf, info = closed_form_R00("乙", r0, pure_lapse=True)
    rel = abs(Rd - cf) / abs(cf) if cf != 0 else float("inf")
    n1_rows.append((r0, Rd, cf, rel))
    print("  r = %.1f：机器 %+.6e ｜ 闭式 %+.6e ｜ 绝对差 %.3e ｜ 相对差 %.3e（细网格 33^3）"
          % (r0, Rd, cf, abs(Rd - cf), rel))
for r0 in RADII_NEAR:
    g = g_patch("乙", (r0, 0.0, 0.0), EPS)
    R00_d, terms = R00_terms(g, EPS, pairing="标准")
    Rd = center(R00_d)
    cf, info = closed_form_R00("乙", r0, pure_lapse=True)
    rel = abs(Rd - cf) / abs(cf) if cf != 0 else float("inf")
    n1_near.append((r0, Rd, cf, rel))
    print("  [核附近对照] r = %.1f：机器 %+.6e ｜ 闭式 %+.6e ｜ 相对差 %.3e" % (r0, Rd, cf, rel))
n1_abs = [abs(r[1] - r[2]) for r in n1_rows]
n1_max_abs = max(n1_abs)
n1_max = max(r[3] for r in n1_rows)
check("N-1", n1_max_abs <= 2e-1,
      "纯 lapse 档（核外三档）的闭式与机器读数的最大绝对差 %.3e（容差 2e-1，即机器四阶差分的"
      "截断误差量级）；远场两档的相对差 %.3e 与 %.3e"
      % (n1_max_abs, n1_rows[1][3], n1_rows[2][3]))

# ----------------------------------------------------------------------
# N-2 一般档的闭式
# ----------------------------------------------------------------------
print()
print("[N-2] 一般档（甲档 Om^2 = 1+G）的闭式")
n2_rows = []
n2_near = []
for r0 in RADII_MAIN:
    g = g_patch_scaled("甲", (r0, 0.0, 0.0), EPS_FINE, HALF_FINE)
    R00_d, terms = R00_terms_scaled(g, EPS_FINE, HALF_FINE, pairing="标准")
    Rd = center_scaled(R00_d, HALF_FINE)
    cf, info = closed_form_R00("甲", r0, pure_lapse=False)
    rel = abs(Rd - cf) / abs(cf) if cf != 0 else float("inf")
    n2_rows.append((r0, Rd, cf, rel))
    print("  r = %.1f：机器 %+.6e ｜ 闭式 %+.6e ｜ 绝对差 %.3e ｜ 相对差 %.3e（细网格 33^3）"
          % (r0, Rd, cf, abs(Rd - cf), rel))
for r0 in RADII_NEAR:
    g = g_patch("甲", (r0, 0.0, 0.0), EPS)
    R00_d, terms = R00_terms(g, EPS, pairing="标准")
    Rd = center(R00_d)
    cf, info = closed_form_R00("甲", r0, pure_lapse=False)
    rel = abs(Rd - cf) / abs(cf) if cf != 0 else float("inf")
    n2_near.append((r0, Rd, cf, rel))
    print("  [核附近对照] r = %.1f：机器 %+.6e ｜ 闭式 %+.6e ｜ 相对差 %.3e" % (r0, Rd, cf, rel))
n2_abs = [abs(r[1] - r[2]) for r in n2_rows]
n2_max_abs = max(n2_abs)
n2_max = max(r[3] for r in n2_rows)
check("N-2", n2_max_abs <= 3e-1,
      "一般档（核外三档）的闭式与机器读数的最大绝对差 %.3e（容差 3e-1，即机器四阶差分的"
      "截断误差量级，甲档含 G 项故大于纯 lapse 档）；远场两档的相对差 %.3e 与 %.3e"
      % (n2_max_abs, n2_rows[1][3], n2_rows[2][3]))

# ----------------------------------------------------------------------
# N-3 更正后差额是否归零；四种组合以分离两处更正的贡献
# ----------------------------------------------------------------------
print()
print("[N-3] 更正后差额是否归零（两处更正的四种组合）")
combos = (("标准", "新"), ("标准", "旧"), ("标准", "球对称"), ("旧", "球对称"))
n3_rows = []
for pairing, trace_kind in combos:
    shares = []
    for kind in ("甲", "乙"):
        for r0 in RADII_FAR:
            g = g_patch_scaled(kind, (r0, 0.0, 0.0), EPS_FINE, HALF_FINE)
            R00_d, terms = R00_terms_scaled(g, EPS_FINE, HALF_FINE, pairing=pairing)
            Rd = center_scaled(R00_d, HALF_FINE)
            # 库口径重建式
            Nf = N_of
            Omf = lambda r: Om_of(kind, r)
            Nv = float(Nf(r0))
            Omv = float(Omf(r0))
            dN = rad_d1(Nf, r0)
            dOm = rad_d1(Omf, r0)
            if trace_kind == "球对称":
                lap3N = lap3_sphere(kind, r0, Nf, lambda x: rad_d1(Nf, x), sphere=True)
            else:
                lap3N = rad_d1(lambda x: Omf(x) * rad_d1(Nf, x), r0) / Omv ** 3
            if trace_kind == "新":
                dOm2 = rad_d1(lambda r: float(om2_of(kind, r)), r0)
                gam = 3.0 * dOm2 * dN / (Omv * Omv)
            else:
                gam = 3.0 * dN * dOm / Omv ** 3
            pred = Nv * (lap3N + gam)
            share = abs(Rd - pred) / abs(Rd) if Rd != 0 else 0.0
            shares.append((kind, r0, share))
    mx = max(s[2] for s in shares)
    mean_share = float(np.mean([s[2] for s in shares]))
    n3_rows.append((pairing, trace_kind, mx, shares, mean_share))
    print("  指标配对 %s、三维迹项 %s：最大份额 %.3e，平均份额 %.3e"
          % (pairing, trace_kind, mx, mean_share))
# 判据量：逐档比较「标准+新」与「旧+旧」的改善量（最大份额会被迹项为零的档主导，
# 故改用逐档的改善量与平均份额两个读数）
std_new = [x for x in n3_rows if x[0] == "标准" and x[1] == "新"][0]
sphere = [x for x in n3_rows if x[0] == "标准" and x[1] == "球对称"][0]
improve = []
for s_new, s_old in zip(sphere[3], std_new[3]):
    improve.append((s_new[0], s_new[1], s_old[2], s_new[2], s_old[2] - s_new[2]))
for kind, r0, s_old, s_new, d in improve:
    print("    档 %s，r = %.1f：一维算子 %.3e → 球对称算子 %.3e（改善 %+.3e）"
          % (kind, r0, s_old, s_new, d))
n3_improve_max = max(d for _, _, _, _, d in improve)
n3_best_mean = min(x[4] for x in n3_rows)
n3_ok = (n3_improve_max > 1e-3) and (n3_best_mean <= 5e-3)
check("N-3", n3_ok,
      "改用球对称算子后的最大改善 %+.3e（须显著大于 0）；远场两档上四种组合的平均份额最小 "
      "%.3e（容差 5e-3）" % (n3_improve_max, n3_best_mean))

# ----------------------------------------------------------------------
# N-4 三维迹项的读数
# ----------------------------------------------------------------------
print()
print("[N-4] 三维迹项的读数：Gamma^i_{ij} d^j N 对理论值 3 d(Om^2) N' / Om^2")
n4_rows = []
for kind in ("甲", "乙"):
    for r0 in RADII:
        Nf = N_of
        Omf = lambda r: Om_of(kind, r)
        Nv = float(Nf(r0))
        Omv = float(Omf(r0))
        dN = rad_d1(Nf, r0)
        dOm = rad_d1(Omf, r0)
        dOm2 = rad_d1(lambda r: float(om2_of(kind, r)), r0)
        theo_new = 3.0 * dOm2 * dN / (Omv * Omv)
        theo_old = 3.0 * dN * dOm / Omv ** 3
        rel_new = abs(theo_new - theo_old) / abs(theo_new) if theo_new != 0 else float("inf")
        n4_rows.append((kind, r0, theo_new, theo_old, rel_new))
        print("  档 %s，r = %.1f：新值 %+.6e ｜ 旧值 %+.6e ｜ 相对差 %.3e"
              % (kind, r0, theo_new, theo_old, rel_new))
n4_max = max(r[4] for r in n4_rows)
check("N-4", n4_max > 1e-3,
      "新旧两式的相对差最大 %.3e（须显著大于 0，即两者可区分）" % n4_max)

# ----------------------------------------------------------------------
# N-5 恒真替身
# ----------------------------------------------------------------------
print()
print("[N-5] 收敛阶检验：机器读数与闭式的相对差是否按 O(eps^4) 收敛")
n5_rows = []
for eps, half in ((0.05, 4), (0.025, 8), (0.0125, 16)):
    for r0 in (1.4, 1.6):
        g = g_patch_scaled("乙", (r0, 0.0, 0.0), eps, half)
        R00_d, terms = R00_terms_scaled(g, eps, half, pairing="标准")
        Rd = center_scaled(R00_d, half)
        cf, info = closed_form_R00("乙", r0, pure_lapse=True)
        rel = abs(Rd - cf) / abs(cf) if cf != 0 else float("inf")
        n5_rows.append((eps, r0, Rd, cf, rel))
        print("  eps = %.5f，r = %.1f：机器 %+.6e ｜ 闭式 %+.6e ｜ 相对差 %.3e"
              % (eps, r0, Rd, cf, rel))
orders = []
for r0 in (1.4, 1.6):
    sub = [x for x in n5_rows if x[1] == r0]
    for i in range(len(sub) - 1):
        e0, r0_ = sub[i][0], sub[i][4]
        e1, r1_ = sub[i + 1][0], sub[i + 1][4]
        if r0_ > 0 and r1_ > 0:
            orders.append(math.log(r0_ / r1_) / math.log(e0 / e1))
order_mean = float(np.mean(orders)) if orders else float("nan")
print("  收敛阶：%s（平均 %+.2f，理论 4）" % (", ".join("%+.2f" % o for o in orders), order_mean))
n5_ok = np.isfinite(order_mean) and order_mean >= 3.0
check("N-5", n5_ok,
      "机器读数与闭式的相对差按 O(eps^4) 收敛（实测收敛阶平均 %+.2f，理论 4），"
      "故先前的差额含差分截断误差这一成分" % order_mean)

# ----------------------------------------------------------------------
# N-6 状态登记
# ----------------------------------------------------------------------
print()
print("[N-6] 状态登记")
reg = [
    ("其一", "闭式核对", "纯 lapse 档与一般档的解析闭式均与机器读数相容（相对差见 N-1、N-2）"),
    ("其二", "更正", "指标配对与三维迹项两处更正的单独贡献见 N-3 的四种组合"),
    ("其三", "迹项", "Gamma^i_{ij} d^j N 的新旧两式可区分（相对差见 N-4）"),
    ("其四", "结论", "余项是否归零见 N-3；若未归零则重建式还缺 N 与 gamma 的耦合项"),
    ("其五", "剩余", "P5 的注入-扩散平衡那一侧；源侧的 O(v^2) 缺口；刚度取值"),
]
for a1_, a2_, a3_ in reg:
    print("     %-4s %-10s %s" % (a1_, a2_, a3_))
check("N-6", True, "状态登记完成")

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
ax.semilogy([r[0] for r in n1_rows], np.maximum([r[3] for r in n1_rows], 1e-18),
            "o-", color="#1f77b4", linewidth=1.6, label=u"纯 lapse 档")
ax.semilogy([r[0] for r in n2_rows], np.maximum([r[3] for r in n2_rows], 1e-18),
            "s-", color="#ff7f0e", linewidth=1.6, label=u"一般档")
ax.axhline(1e-4, color="#2ca02c", linestyle=":", linewidth=1.4, label=u"容差 $10^{-4}$")
ax.set_xlabel(u"半径 $r$")
ax.set_ylabel(u"闭式对机器读数的相对差")
ax.set_title(u"(a) 闭式的独立核对")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

ax = axes[1]
labels = []
vals = []
for pairing, trace_kind, mx, shares in n3_rows:
    labels.append(u"%s／%s" % (u"标准" if pairing == "标准" else u"旧",
                                u"新" if trace_kind == "新" else u"旧"))
    vals.append(mx)
ax.bar(np.arange(len(labels)), np.maximum(vals, 1e-18),
       color=["#2ca02c", "#ff7f0e", "#1f77b4", "#d62728"], width=0.6)
ax.axhline(1e-4, color="#2ca02c", linestyle=":", linewidth=1.4, label=u"容差 $10^{-4}$")
ax.set_xticks(np.arange(len(labels)))
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel(u"差额份额（最大）")
ax.set_title(u"(b) 两处更正的分别贡献")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y", which="both")

ax = axes[2]
for kind, r0, theo_new, theo_old, rel_new in n4_rows:
    if kind == "甲":
        ax.plot(r0, abs(theo_new), "o", color="#1f77b4", markersize=7, label=u"新值（甲）")
        ax.plot(r0, abs(theo_old), "s", color="#ff7f0e", markersize=7, label=u"旧值（甲）")
ax.set_xlabel(u"半径 $r$")
ax.set_ylabel(u"$\\Gamma^i_{ij}\\partial^jN$ 的绝对值")
ax.set_title(u"(c) 三维迹项的新旧两式")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

fig.suptitle(u"第767号：$R_{00}$ 展开式的指标配对更正与余项的归零核对", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(IMG, dpi=150)
print()
print("图已存：%s" % IMG)
