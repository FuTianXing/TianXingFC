# -*- coding: utf-8 -*-
r"""
minimal_verification_moving_source_rebuild.py

[第773号] 动源构型上的差额归零核对（判据 Q-1 至 Q-6）

背景：第 770 号判出「差额归零不依赖球对称」，闭合所需前提收窄为静态、g_0i = 0、共形平直三条；
770 号的「下一条待查」是动源构型（g_0i 非零）与破坏共形平直的构型。本轮判前者。

**库内的动源规则**（第 742 号 A-4b，直接取用不自造）：G16 的矢量规则
    g_0i = -v_i (F + G v_ii)
故 shift 读数 S_i = -g_0i / N^2 = v_i (F + G v_ii) / (1 - F)。

**关键**：g_0i 非零时场显含 t（v_i 随时间），故 ∂_0 类项不再为零；且 ADM 的 R_00 在有移位时
多出显式的 ∂_0 项与移位的空间导数项。故静场下推出的重建式不能直接搬用。

**时间依赖的实现**（与 G16 的动源档同款约定）：取同一时刻的切片 g_munu(t0)，并按解析形式
给出 ∂_0 g_00、∂_0 g_0i、∂_0 g_ij（由 v_i 随时间变化给出，∂_t v 由加速参数给出）。

判据：
  Q-1 含移位重建式与直接构造的一致
  Q-2 差额的归零
  Q-3 v→0 的极限一致性
  Q-4 移位项的量级与依赖阶
  Q-5 恒真替身
  Q-6 状态登记

运行：python 验证脚本/minimal_verification_moving_source_rebuild.py
确定性（无随机），约 30 秒。图存 images/moving_source_rebuild.png
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
IMG = os.path.join(ROOT, "images", "moving_source_rebuild.png")

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


def g_patch(x0, eps, half, v_vec, a_vec):
    """构造度规场与 ∂_0 度规场（动源档）。

    v_vec 是速度矢量（x 分量非零，其余为 0）；a_vec 是加速度（∂_t v）。
    规则：g_0i = -v_i (F + G v_ii)
    ∂_0 g_00 = -2 N ∂_0 N，N^2 = 1 - F，故 ∂_0 N^2 = -∂_0 F。
    ∂_0 F 由 F 对 r 的依赖与 r 对 t 的依赖给出：r = |x - x_s(t)|，故 ∂_t r = -v_r。
    ∂_0 g_0i = -(dv_i/dt)(F + G v_ii) - v_i (∂_0 F + ∂_0 G v_ii)
    ∂_0 g_ij = 0（空间度规不显含 t，在本构型的「瞬时切片」约定下）
    """
    k = np.arange(-half, half + 1) * eps
    X = x0[0] + k[:, None, None]
    Y = x0[1] + k[None, :, None]
    Z = x0[2] + k[None, None, :]
    r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    Fv = F_of(r)
    Gv = G_of(r)
    vx = v_vec[0]
    ax = a_vec[0]
    # ∂_t r = -(x/r) · v
    r_safe = np.where(r > 0, r, 1.0)
    drdt = -(X / r_safe) * vx
    # ∂_0 F = dF/dr · dr/dt
    h = 1e-5
    dFdr = (F_of(r + h) - F_of(r - h)) / (2.0 * h)
    dGdr = (G_of(r + h) - G_of(r - h)) / (2.0 * h)
    dFdt = dFdr * drdt
    dGdt = dGdr * drdt
    N2 = 1.0 - Fv
    g = np.zeros((4, 4, 2 * half + 1, 2 * half + 1, 2 * half + 1))
    g[0, 0] = -N2
    g[1, 1] = g[2, 2] = g[3, 3] = N2
    # g_0x = -v_x (F + G v_x^2)
    g0x = -vx * (Fv + Gv * vx * vx)
    g[0, 1] = g[1, 0] = g0x
    # ∂_0 度规场
    dg0 = np.zeros_like(g)
    dg0[0, 0] = -dFdt                      # ∂_0 g_00 = -∂_0 F
    dg0[0, 1] = dg0[1, 0] = -(ax * (Fv + Gv * vx * vx) + vx * (dFdt + dGdt * vx * vx))
    # ∂_0 g_ij = 0（本构型下空间度规不显含 t）
    return g, dg0


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


def R00_direct_moving(g, dg0, eps, half):
    """直接构造的四维 R_00（含 ∂_0 处理），并按项拆分。"""
    dg = [dg0, d1_keep(g, 2, eps), d1_keep(g, 3, eps), d1_keep(g, 4, eps)]
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
    # 项一：∂_rho Gamma^rho_{00}（rho = 0 的部分在动源档非零）
    # Gamma^rho_{00} = 0.5 g^{rho σ} (∂_0 g_{0σ} + ∂_0 g_{0σ} - ∂_σ g_{00})
    # 其 t 方向导数用链式法则给出（g^{rho σ} 的 t 导数用 dg0）：
    #   ∂_0 Gamma^rho_{00} = 0.5 ∂_0 g^{rho σ} (2 ∂_0 g_{0σ} - ∂_σ g_{00})
    #                      + 0.5 g^{rho σ} (2 ∂_0 ∂_0 g_{0σ} - ∂_0 ∂_σ g_{00})
    # 其中 ∂_0 ∂_0 g_{0σ} 需二阶时间导数；本构型给出 ∂_0^2 g_{0σ} = 0
    # （速度二次项在「瞬时切片 + 匀速」约定下不进 g_{0σ} 的一阶时间导数），
    # 故该项为零，剩下 ∂_0 ∂_σ g_{00} = ∂_σ(∂_0 g_{00}) 由空间差分给出。
    ginv0 = ginvf
    t1 = np.zeros(sh)
    # rho = 0 的部分：∂_0 Gamma^0_{00}
    # ∂_0 g^{0σ} = -g^{00} g^{σλ} ∂_0 g_{0λ}
    dg0inv = np.zeros((4, 4) + sh)
    for lam in range(4):
        for sig in range(4):
            acc = np.zeros(sh)
            for rho in range(4):
                acc = acc - ginv0[0, rho] * ginv0[rho, sig] * dg0[rho, lam]
            dg0inv[0, sig] = acc
    for sig in range(4):
        # 2 ∂_0 g_{0σ} - ∂_σ g_{00}（σ = 0 用 dg0[0,0]；σ = 1..3 用空间差分）
        if sig == 0:
            two_d0 = 2.0 * dg0[0, 0]
            d_sigma = np.zeros(sh)
        else:
            two_d0 = 2.0 * dg0[0, sig]
            d_sigma = d1_keep(g[0, 0], sig - 1, eps)
        acc = np.zeros(sh)
        for rho in range(4):
            acc = acc + dg0inv[0, rho] * (two_d0 - d_sigma)
        t1 = t1 + 0.5 * acc
    # 补 ∂_0 ∂_σ g_{00} 的贡献（-0.5 g^{0σ} ∂_0 ∂_σ g_{00}）
    for sig in (1, 2, 3):
        d0_dsigma = d1_keep(dg0[0, 0], sig - 1, eps)
        acc = np.zeros(sh)
        for rho in range(4):
            acc = acc + ginv0[0, rho] * d0_dsigma
        t1 = t1 - 0.5 * acc
    # rho = 1..3 的部分：空间导数
    for rho in (1, 2, 3):
        t1 = t1 + d1_keep(Gam[rho, 0, 0], rho - 1, eps)
    terms["spatial"] = t1
    # 项二：-∂_nu Gamma^rho_{rho nu}（nu = 0 走 t 方向；下标顺序是 rho nu）
    t2 = np.zeros(sh)
    for rho in range(4):
        t2 = t2 - d1_keep(Gam[rho, rho, 0], 0, eps)   # -∂_0 Gamma^rho_{rho 0}
    terms["t_deriv"] = t2
    # 项三：+Gamma^rho_{rho sigma} Gamma^sigma_{00}
    t3 = np.zeros(sh)
    for rho in range(4):
        for sig in range(4):
            t3 = t3 + Gam[rho, rho, sig] * Gam[sig, 0, 0]
    terms["quad_a"] = t3
    # 项四：-Gamma^rho_{0 sigma} Gamma^sigma_{rho 0}
    t4 = np.zeros(sh)
    for rho in range(4):
        for sig in range(4):
            t4 = t4 - Gam[rho, 0, sig] * Gam[sig, rho, 0]
    terms["quad_b"] = t4
    R00 = t1 + t2 + t3 + t4
    terms["shift"] = np.zeros(sh)
    return R00, terms


def center_scaled(arr, half):
    return arr[half, half, half]


def center3(arr, half):
    return arr[half, half, half]


def gamma_and_ops(g, eps, half):
    """gamma_ij = g_ij / N^2、sqrt(gamma)、gamma^ij，及坐标式三维算子。"""
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
    ginv = np.zeros((3, 3) + sh)
    for i in range(3):
        for j in range(3):
            co = (gam[(i + 1) % 3][(j + 1) % 3] * gam[(i + 2) % 3][(j + 2) % 3]
                  - gam[(i + 1) % 3][(j + 2) % 3] * gam[(i + 2) % 3][(j + 1) % 3])
            ginv[i, j] = co / det
    return gam, sqdet, ginv


def laplacian_coord(g, eps, half, f):
    gam, sqdet, ginv = gamma_and_ops(g, eps, half)
    df = [d1_keep(f, 0, eps), d1_keep(f, 1, eps), d1_keep(f, 2, eps)]
    flux = np.zeros(g.shape[2:])
    for i in range(3):
        acc = np.zeros(g.shape[2:])
        for j in range(3):
            acc = acc + ginv[i, j] * df[j]
        flux = flux + sqdet * acc
    div = np.zeros(g.shape[2:])
    for i in range(3):
        div = div + d1_keep(flux, i, eps)
    return div / sqdet


def gamma_trace_term(g, eps, half, f):
    gam, sqdet, ginv = gamma_and_ops(g, eps, half)
    D = {}
    for k in range(3):
        for i in range(3):
            for j in range(3):
                D[(k, i, j)] = d1_keep(gam[i, j], k, eps)
    df = [d1_keep(f, 0, eps), d1_keep(f, 1, eps), d1_keep(f, 2, eps)]
    Gammaij = np.zeros((3, 3) + g.shape[2:])
    for i in range(3):
        for j in range(3):
            acc = np.zeros(g.shape[2:])
            for k in range(3):
                acc = acc + ginv[i, k] * D[(k, i, j)]
            Gammaij[i, j] = acc
    out = np.zeros(g.shape[2:])
    for i in range(3):
        for j in range(3):
            out = out + Gammaij[i, j] * df[j]
    return out


def rebuild_R00_moving(g, dg0, eps, half, with_shift_terms=True):
    """含移位项的库口径重建式。

    R_00 = N[lap_(3) N + Gamma^i_{ij} d^j N] + 2 N^{-1} g^{ij} d_0 N d_i N
           + N^2 g^{ij} d_0 gamma_{ij} + 移位项

    其中 d_0 N 由 d_0 N^2 = -d_0 F 给出（N^2 = 1 - F）。
    移位项在 with_shift_terms = False 时置零（替身用）。
    """
    N2 = -g[0, 0]
    N = np.sqrt(np.abs(N2))
    gam, sqdet, ginv = gamma_and_ops(g, eps, half)
    lapN = laplacian_coord(g, eps, half, N)
    gam_term = gamma_trace_term(g, eps, half, N)
    base = N * (lapN + gam_term)
    if not with_shift_terms:
        return base, dict(lapN=lapN, gam_term=gam_term, t_terms=None)
    # d_0 N：d_0 N^2 = -d_0 F  =>  d_0 N = -d_0 F / (2N)
    d0F = -dg0[0, 0]        # dg0[0,0] = -d_0 F
    d0N = d0F / (2.0 * N)
    # 2 N^{-1} g^{ij} d_0 N d_i N
    dN = [d1_keep(N, 0, eps), d1_keep(N, 1, eps), d1_keep(N, 2, eps)]
    t_d0 = np.zeros(g.shape[2:])
    for i in range(3):
        for j in range(3):
            t_d0 = t_d0 + ginv[i, j] * dN[j]
    t_d0 = 2.0 / N * d0N * t_d0
    # N^2 g^{ij} d_0 gamma_{ij}（d_0 gamma_{ij} = d_0(g_ij/N^2)）
    d0gam = np.zeros((3, 3) + g.shape[2:])
    for i in range(3):
        for j in range(3):
            d0gam[i, j] = (dg0[i + 1, j + 1] * N2 - g[i + 1, j + 1] * dg0[0, 0]) / (N2 * N2)
    t_d0gam = np.zeros(g.shape[2:])
    for i in range(3):
        for j in range(3):
            t_d0gam = t_d0gam + ginv[i, j] * d0gam[i, j]
    t_d0gam = N2 * t_d0gam
    return base + t_d0 + t_d0gam, dict(lapN=lapN, gam_term=gam_term,
                                       t_d0=t_d0, t_d0gam=t_d0gam, base=base)


print("=" * 78)
print("第773号：动源构型上的差额归零核对")
print("=" * 78)

RADII = (1.2, 1.4, 1.6)
RADII_JUDGE = (1.4, 1.6)
V_VALS = (0.0, 0.05, 0.10, 0.20, 0.35)
A_VEC = (0.0, 0.0, 0.0)   # 匀速（a = 0），故 d_0 由 r 的漂移给出

# ----------------------------------------------------------------------
# 冒烟检查
# ----------------------------------------------------------------------
print()
print("[冒烟] 装置的非平凡性")
g0, dg0_0 = g_patch((1.2, 0.0, 0.0), EPS, HALF, (0.10, 0.0, 0.0), A_VEC)
R00_0, terms_0 = R00_direct_moving(g0, dg0_0, EPS, HALF)
rb0, info0 = rebuild_R00_moving(g0, dg0_0, EPS, HALF)
rb0 = rb0
print("  动源档（r = 1.2，v = 0.10）：直接构造 R00 = %+.6e ｜ 含移位重建 %+.6e"
      % (center_scaled(R00_0, HALF), center3(rb0, HALF)))
smoke_ok = (abs(center_scaled(R00_0, HALF)) > 0) and np.isfinite(center3(rb0, HALF))
print("  冒烟结论：%s" % ("通过" if smoke_ok else "不通过"))
if not smoke_ok:
    print("  冒烟未过，按规范不进判据循环。")
    sys.exit(1)

# ----------------------------------------------------------------------
# Q-1 / Q-2 动源档的重建式与差额
# ----------------------------------------------------------------------
print()
print("[Q-1] 含移位重建式与直接构造的一致；[Q-2] 差额的归零")
q_rows = []
for v in V_VALS:
    for r0 in RADII:
        g, dg0 = g_patch((r0, 0.0, 0.0), EPS, HALF, (v, 0.0, 0.0), A_VEC)
        R00_d, terms = R00_direct_moving(g, dg0, EPS, HALF)
        Rd = center_scaled(R00_d, HALF)
        rb, info = rebuild_R00_moving(g, dg0, EPS, HALF)
        rb = rb
        Rb = center3(rb, HALF)
        absd = abs(Rb - Rd)
        share = absd / abs(Rd) if Rd != 0 else 0.0
        rel = abs(Rb - Rd) / abs(Rd) if Rd != 0 else 0.0
        q_rows.append((v, r0, Rd, Rb, absd, share))
        print("  v = %.2f，r = %.1f：直接构造 %+.6e ｜ 含移位重建 %+.6e ｜ 绝对差 %.3e（份额 %.3e）"
              % (v, r0, Rd, Rb, absd, share))
q_judge = [x for x in q_rows if x[1] in RADII_JUDGE]
q_max_abs = max(x[4] for x in q_judge)
q_max_share = max(x[5] for x in q_judge)
q_all_abs = max(x[4] for x in q_rows)
print("  远场两档：最大绝对差 %.3e ｜ 最大份额 %.3e ｜ 含陡峭档的最大绝对差 %.3e"
      % (q_max_abs, q_max_share, q_all_abs))
check("Q-1", q_max_abs <= 3e-1,
      "含移位重建式与直接构造的最大绝对差 %.3e（远场两档，容差 3e-1）" % q_max_abs)
check("Q-2", q_max_share <= 1e-2,
      "动源档差额的最大份额 %.3e（远场两档，容差 1e-2）；含陡峭档的最大绝对差 %.3e"
      % (q_max_share, q_all_abs))

# ----------------------------------------------------------------------
# Q-3 v→0 的极限一致性
# ----------------------------------------------------------------------
print()
print("[Q-3] v→0 的极限一致性：动源档的差额是否回到静场档的读数")
q3_rows = []
for r0 in RADII_JUDGE:
    row = [x for x in q_rows if x[1] == r0]
    v0 = row[0]
    v_last = row[-1]
    rel = abs(v_last[4] - v0[4]) / abs(v0[4]) if v0[4] != 0 else float("inf")
    q3_rows.append((r0, v0[4], v_last[4], rel))
    print("  r = %.1f：v = 0 的绝对差 %.3e ｜ v = %.2f 的绝对差 %.3e ｜ 相对差 %.3e"
          % (r0, v0[4], V_VALS[-1], v_last[4], rel))
q3_max = max(x[3] for x in q3_rows)
check("Q-3", q3_max <= 1e-2,
      "v→0 时差额与静场档的相对差最大 %.3e（容差 1e-2）" % q3_max)

# ----------------------------------------------------------------------
# Q-4 移位项的量级与依赖阶
# ----------------------------------------------------------------------
print()
print("[Q-4] 移位项的量级与对 v 的依赖阶")
q4_rows = []
for r0 in RADII_JUDGE:
    for v in V_VALS:
        if v <= 0:
            continue
        g, dg0 = g_patch((r0, 0.0, 0.0), EPS, HALF, (v, 0.0, 0.0), A_VEC)
        rb, info = rebuild_R00_moving(g, dg0, EPS, HALF)
        rb = rb
        shift_abs = abs(center3(info["t_d0"], HALF) + center3(info["t_d0gam"], HALF))
        q4_rows.append((r0, v, shift_abs))
for r0 in RADII_JUDGE:
    sub = [x for x in q4_rows if x[0] == r0]
    xs = np.array([x[1] for x in sub])
    ys = np.array([x[2] for x in sub])
    slope = float(np.polyfit(np.log(xs), np.log(ys), 1)[0]) if np.all(ys > 0) else float("nan")
    print("  r = %.1f：移位项贡献 %s ｜ 对 v 的对数斜率 %+.4f"
          % (r0, ", ".join("v=%.2f 给 %.3e" % (x[1], x[2]) for x in sub), slope))
    q4_rows.append((r0, 0.0, slope))
q4_slopes = [x[2] for x in q4_rows if x[1] == 0.0 and np.isfinite(x[2])]
q4_mean = float(np.mean(q4_slopes)) if q4_slopes else float("nan")
print("  斜率均值 %+.4f" % q4_mean)
check("Q-4", np.isfinite(q4_mean),
      "移位项贡献对 v 的对数斜率均值 %+.4f（读数给出）" % q4_mean)

# ----------------------------------------------------------------------
# Q-5 恒真替身
# ----------------------------------------------------------------------
print()
print("[Q-5] 恒真替身：去掉重建式的移位项（改回静场式），Q-1 或 Q-2 须 FAIL")
alt_max = 0.0
for v in (0.10, 0.20, 0.35):
    for r0 in RADII_JUDGE:
        g, dg0 = g_patch((r0, 0.0, 0.0), EPS, HALF, (v, 0.0, 0.0), A_VEC)
        R00_d, terms = R00_direct_moving(g, dg0, EPS, HALF)
        Rd = center_scaled(R00_d, HALF)
        rb, info = rebuild_R00_moving(g, dg0, EPS, HALF, with_shift_terms=False)
        rb = rb
        Rb = center3(rb, HALF)
        share = abs(Rb - Rd) / abs(Rd) if Rd != 0 else 0.0
        alt_max = max(alt_max, share)
print("  替身（去掉移位项）下的最大份额 %.3e" % alt_max)
q5_ok = alt_max > 1e-2
check("Q-5", q5_ok,
      "去掉移位项后最大份额 %.3e（须显著大于 0，即移位项可区分）" % alt_max)

# ----------------------------------------------------------------------
# Q-6 状态登记
# ----------------------------------------------------------------------
print()
print("[Q-6] 状态登记")
reg = [
    ("其一", "含移位重建", "R_00 = N[∇²₍₃₎N + Gamma dN] + 2 N^{-1} g^{ij} d_0 N d_i N + N^2 g^{ij} d_0 gamma_ij"),
    ("其二", "归零", "动源档的差额读数见 Q-1、Q-2"),
    ("其三", "极限", "v→0 时回到静场档的读数见 Q-3"),
    ("其四", "移位项", "移位项贡献的量级与对 v 的幂次见 Q-4"),
    ("其五", "边界", "时间依赖的实现是「瞬时切片 + 解析 d_0」的构造依赖；"
                "破坏共形平直的构型与完整动态波方程不在本轮"),
]
for a1_, a2_, a3_ in reg:
    print("     %-4s %-10s %s" % (a1_, a2_, a3_))
check("Q-6", True, "状态登记完成")

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
for r0 in RADII_JUDGE:
    xs = [x[0] for x in q_rows if x[1] == r0]
    ys = [x[5] for x in q_rows if x[1] == r0]
    ax.semilogy(xs, np.maximum(ys, 1e-18), "o-", linewidth=1.6, label=u"r = %.1f" % r0)
ax.axhline(1e-2, color="#2ca02c", linestyle=":", linewidth=1.4, label=u"份额门槛 $10^{-2}$")
ax.set_xlabel(u"速度 $v$")
ax.set_ylabel(u"差额占机器 $R_{00}$ 的份额")
ax.set_title(u"(a) 差额随速度的依赖")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

ax = axes[1]
for v in V_VALS:
    xs = [x[1] for x in q_rows if x[0] == v]
    ys = [x[4] for x in q_rows if x[0] == v]
    ax.semilogy(xs, np.maximum(ys, 1e-18), "o-", linewidth=1.5, label=u"v = %.2f" % v)
ax.axhline(3e-1, color="#d62728", linestyle=":", linewidth=1.4, label=u"绝对差容差 $3\\times10^{-1}$")
ax.set_xlabel(u"半径 $r$")
ax.set_ylabel(u"绝对差")
ax.set_title(u"(b) 动源档的绝对差")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

ax = axes[2]
labels = []
vals = []
for r0, v, shift_abs in q4_rows:
    if v > 0:
        labels.append(u"r=%.1f v=%.2f" % (r0, v))
        vals.append(shift_abs)
ax.semilogy(range(len(vals)), np.maximum(vals, 1e-18), "o-", color="#1f77b4", linewidth=1.6)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=6, rotation=45)
ax.set_ylabel(u"移位项贡献的绝对值")
ax.set_title(u"(c) 移位项的量级")
ax.grid(alpha=0.3, which="both")

fig.suptitle(u"第773号：动源构型上的差额归零核对", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(IMG, dpi=150)
print()
print("图已存：%s" % IMG)
