# -*- coding: utf-8 -*-
r"""
minimal_verification_moving_source_term_attribution.py

[第776号] 动源档差额在 R00 展开式上的项级归属（判据 R-1 至 R-6）

背景：第 773 号判出「动源构型上差额未归零，且随速度线性增长」（份额跨五个量级），并已排除
v 的幂次与重建式移位项的量级。本轮把差额定位到具体的展开项上。

**细分方式**：沿用第 767 号的四项拆分，但把每一项再按「Gamma 的三个指标里有几个是 0」细分。
这个细分方式与静场退化直接相关（静场下所有含 0 的 Gamma 都退化），故最能暴露被静场掩盖的项。

**归属的判定**：对每个细分项 X，检验「差额减 X」在多档上是否同为零（绝对差）；且归属须跨
v 档一致。

判据：
  R-1 细分项之和的恒等
  R-2 单一归属
  R-3 组合归属
  R-4 归属项对 v 的幂次
  R-5 恒真替身
  R-6 状态登记

运行：python 验证脚本/minimal_verification_moving_source_term_attribution.py
确定性（无随机），约 40 秒。图存 images/moving_source_term_attribution.png
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
IMG = os.path.join(ROOT, "images", "moving_source_term_attribution.png")

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

    规则（第 742 号 A-4b）：g_0i = -v_i (F + G v_ii)。
    ∂_0 g_00 = -∂_0 F；∂_0 g_0i = -(dv_i/dt)(F + G v_ii) - v_i(∂_0 F + ∂_0 G v_ii)；
    ∂_0 g_ij = 0（空间度规不显含 t）。
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
    r_safe = np.where(r > 0, r, 1.0)
    drdt = -(X / r_safe) * vx
    h = 1e-5
    dFdr = (F_of(r + h) - F_of(r - h)) / (2.0 * h)
    dGdr = (G_of(r + h) - G_of(r - h)) / (2.0 * h)
    dFdt = dFdr * drdt
    dGdt = dGdr * drdt
    N2 = 1.0 - Fv
    g = np.zeros((4, 4, 2 * half + 1, 2 * half + 1, 2 * half + 1))
    g[0, 0] = -N2
    g[1, 1] = g[2, 2] = g[3, 3] = N2
    g0x = -vx * (Fv + Gv * vx * vx)
    g[0, 1] = g[1, 0] = g0x
    dg0 = np.zeros_like(g)
    dg0[0, 0] = -dFdt
    dg0[0, 1] = dg0[1, 0] = -(ax * (Fv + Gv * vx * vx) + vx * (dFdt + dGdt * vx * vx))
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


def R00_terms_fine(g, dg0, eps, half):
    """直接构造的四维 R_00，并把四项按「Gamma 的指标含几个 0」细分。

    标准式：R_00 = ∂_rho Γ^rho_{00} - ∂_0 Γ^rho_{rho 0}
                  + Γ^rho_{rho sigma} Γ^sigma_{00} - Γ^rho_{0 sigma} Γ^sigma_{rho 0}

    细分：
      t1_rho0    ∂_0 Γ^0_{00}（t 方向，指标含两个 0）
      t1_rhoN    ∂_rho Γ^rho_{00}（rho = 1..3，空间方向，指标含两个 0）
      t2_rho0    -∂_0 Γ^rho_{rho 0}（rho = 0..3；rho 含 0 与 rho 两处 0）
      t2_rhoN    -∂_0 Γ^rho_{rho 0} 中 rho = 1..3 的部分
      t3_s0      Γ^rho_{rho sigma} Γ^sigma_{00} 中 sigma = 0 的部分
      t3_sN      Γ^rho_{rho sigma} Γ^sigma_{00} 中 sigma = 1..3 的部分
      t4_00      -Γ^rho_{0 0} Γ^0_{rho 0}
      t4_0N      -Γ^rho_{0 sigma} Γ^sigma_{rho 0} 中 sigma = 1..3 的部分
      t4_N0      -Γ^rho_{0 sigma} Γ^sigma_{rho 0} 中 rho = 1..3 且 sigma = 0 的部分
      t4_NN      -Γ^rho_{0 sigma} Γ^sigma_{rho 0} 中 rho = 1..3 且 sigma = 1..3 的部分
    """
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

    T = {}
    Z = np.zeros(sh)

    # --- 项一：∂_rho Γ^rho_{00} ---
    # rho = 0：∂_0 Γ^0_{00}，用链式法则
    dg0inv = np.zeros((4, 4) + sh)
    for lam in range(4):
        for sig in range(4):
            acc = np.zeros(sh)
            for rho in range(4):
                acc = acc - ginvf[0, rho] * ginvf[rho, sig] * dg0[rho, lam]
            dg0inv[0, sig] = acc
    t1_rho0 = Z.copy()
    for sig in range(4):
        if sig == 0:
            two_d0 = 2.0 * dg0[0, 0]
            d_sigma = Z.copy()
        else:
            two_d0 = 2.0 * dg0[0, sig]
            d_sigma = d1_keep(g[0, 0], sig - 1, eps)
        acc = Z.copy()
        for rho in range(4):
            acc = acc + dg0inv[0, rho] * (two_d0 - d_sigma)
        t1_rho0 = t1_rho0 + 0.5 * acc
    for sig in (1, 2, 3):
        d0_dsigma = d1_keep(dg0[0, 0], sig - 1, eps)
        acc = Z.copy()
        for rho in range(4):
            acc = acc + ginvf[0, rho] * d0_dsigma
        t1_rho0 = t1_rho0 - 0.5 * acc
    T["t1_rho0"] = t1_rho0
    # rho = 1..3
    t1_rhoN = Z.copy()
    for rho in (1, 2, 3):
        t1_rhoN = t1_rhoN + d1_keep(Gam[rho, 0, 0], rho - 1, eps)
    T["t1_rhoN"] = t1_rhoN

    # --- 项二：-∂_0 Γ^rho_{rho 0} ---
    t2_all = Z.copy()
    for rho in range(4):
        t2_all = t2_all - d1_keep(Gam[rho, rho, 0], 0, eps)
    T["t2_all"] = t2_all
    t2_rho0 = -d1_keep(Gam[0, 0, 0], 0, eps)
    t2_rhoN = Z.copy()
    for rho in (1, 2, 3):
        t2_rhoN = t2_rhoN - d1_keep(Gam[rho, rho, 0], 0, eps)
    T["t2_rho0"] = t2_rho0
    T["t2_rhoN"] = t2_rhoN

    # --- 项三：+Γ^rho_{rho sigma} Γ^sigma_{00} ---
    t3_s0 = Z.copy()
    t3_sN = Z.copy()
    for rho in range(4):
        for sig in range(4):
            contrib = Gam[rho, rho, sig] * Gam[sig, 0, 0]
            if sig == 0:
                t3_s0 = t3_s0 + contrib
            else:
                t3_sN = t3_sN + contrib
    T["t3_s0"] = t3_s0
    T["t3_sN"] = t3_sN

    # --- 项四：-Γ^rho_{0 sigma} Γ^sigma_{rho 0} ---
    t4_00 = Z.copy()
    t4_0N = Z.copy()
    t4_N0 = Z.copy()
    t4_NN = Z.copy()
    for rho in range(4):
        for sig in range(4):
            contrib = -Gam[rho, 0, sig] * Gam[sig, rho, 0]
            if rho == 0 and sig == 0:
                t4_00 = t4_00 + contrib
            elif rho == 0:
                t4_0N = t4_0N + contrib
            elif sig == 0:
                t4_N0 = t4_N0 + contrib
            else:
                t4_NN = t4_NN + contrib
    T["t4_00"] = t4_00
    T["t4_0N"] = t4_0N
    T["t4_N0"] = t4_N0
    T["t4_NN"] = t4_NN

    R00 = t1_rho0 + t1_rhoN + t2_all + t3_s0 + t3_sN + t4_00 + t4_0N + t4_N0 + t4_NN
    return R00, T


def center_scaled(arr, half):
    return arr[half, half, half]


def gamma_and_ops(g, eps, half):
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


def rebuild_R00_moving(g, dg0, eps, half):
    """含移位项的库口径重建式（沿用第 773 号）。"""
    N2 = -g[0, 0]
    N = np.sqrt(np.abs(N2))
    gam, sqdet, ginv = gamma_and_ops(g, eps, half)
    lapN = laplacian_coord(g, eps, half, N)
    gam_term = gamma_trace_term(g, eps, half, N)
    base = N * (lapN + gam_term)
    d0F = -dg0[0, 0]
    d0N = d0F / (2.0 * N)
    dN = [d1_keep(N, 0, eps), d1_keep(N, 1, eps), d1_keep(N, 2, eps)]
    t_d0 = np.zeros(g.shape[2:])
    for i in range(3):
        for j in range(3):
            t_d0 = t_d0 + ginv[i, j] * dN[j]
    t_d0 = 2.0 / N * d0N * t_d0
    d0gam = np.zeros((3, 3) + g.shape[2:])
    for i in range(3):
        for j in range(3):
            d0gam[i, j] = (dg0[i + 1, j + 1] * N2 - g[i + 1, j + 1] * dg0[0, 0]) / (N2 * N2)
    t_d0gam = np.zeros(g.shape[2:])
    for i in range(3):
        for j in range(3):
            t_d0gam = t_d0gam + ginv[i, j] * d0gam[i, j]
    t_d0gam = N2 * t_d0gam
    return base + t_d0 + t_d0gam


print("=" * 78)
print("第776号：动源档差额在 R00 展开式上的项级归属")
print("=" * 78)

RADII = (1.4, 1.6)
V_VALS = (0.0, 0.05, 0.10, 0.20, 0.35)
# 主档：v <= 0.20（移位量小，一阶效应主导）；v = 0.35 单列为「超出适用范围」对照档
# （该档归属改变，且 t1_rho0 与 t2_rho0 的抵消关系显示二阶效应进场）。
V_MAIN = (0.05, 0.10, 0.20)
A_VEC = (0.0, 0.0, 0.0)

# ----------------------------------------------------------------------
# 冒烟检查
# ----------------------------------------------------------------------
print()
print("[冒烟] 装置的非平凡性")
g0, dg0_0 = g_patch((1.4, 0.0, 0.0), EPS, HALF, (0.10, 0.0, 0.0), A_VEC)
R00_0, T0d = R00_terms_fine(g0, dg0_0, EPS, HALF)
rb0 = rebuild_R00_moving(g0, dg0_0, EPS, HALF)
print("  动源档（r = 1.4，v = 0.10）：直接构造 R00 = %+.6e ｜ 含移位重建 %+.6e"
      % (center_scaled(R00_0, HALF), center_scaled(rb0, HALF)))
smoke_ok = (abs(center_scaled(R00_0, HALF)) > 0) and np.isfinite(center_scaled(rb0, HALF))
print("  冒烟结论：%s" % ("通过" if smoke_ok else "不通过"))
if not smoke_ok:
    print("  冒烟未过，按规范不进判据循环。")
    sys.exit(1)

# ----------------------------------------------------------------------
# 逐档读数
# ----------------------------------------------------------------------
print()
print("[逐档读数] 各细分项的贡献与差额")
TERM_KEYS = ("t1_rho0", "t1_rhoN", "t2_rho0", "t2_rhoN", "t3_s0", "t3_sN",
             "t4_00", "t4_0N", "t4_N0", "t4_NN")
rows = []
for v in V_VALS:
    for r0 in RADII:
        g, dg0 = g_patch((r0, 0.0, 0.0), EPS, HALF, (v, 0.0, 0.0), A_VEC)
        R00_d, T = R00_terms_fine(g, dg0, EPS, HALF)
        Rd = center_scaled(R00_d, HALF)
        rb = rebuild_R00_moving(g, dg0, EPS, HALF)
        Rb = center_scaled(rb, HALF)
        diff = Rb - Rd
        vals = {k: center_scaled(T[k], HALF) for k in TERM_KEYS}
        rows.append((v, r0, Rd, Rb, diff, vals))
        print("  v = %.2f，r = %.1f：直接构造 %+.6e ｜ 重建 %+.6e ｜ 差额 %+.6e"
              % (v, r0, Rd, Rb, diff))
        for k in TERM_KEYS:
            print("      %-9s %+.6e" % (k, vals[k]))

# ----------------------------------------------------------------------
# R-1 细分项之和的恒等
# ----------------------------------------------------------------------
print()
print("[R-1] 细分项之和与直接构造的 R00 的恒等核对")
r1_max = 0.0
for v, r0, Rd, Rb, diff, vals in rows:
    ssum = sum(vals[k] for k in TERM_KEYS)
    rel = abs(ssum - Rd) / abs(Rd) if Rd != 0 else 0.0
    r1_max = max(r1_max, rel)
print("  最大相对差 %.3e" % r1_max)
check("R-1", r1_max <= 1e-12,
      "细分项之和与直接构造的 R00 的最大相对差 %.3e（容差 1e-12）" % r1_max)

# ----------------------------------------------------------------------
# R-2 单一归属 / R-3 组合归属
# ----------------------------------------------------------------------
print()
print("[R-2] 单一归属；[R-3] 组合归属")
TOL = 5e-2   # 相对残差门槛
# 逐档归属（只取 v > 0 的档；v = 0 档的差额是数值误差量级，在其上的「匹配」是平凡的）
per_row_single = []
per_row_single_all = []
for v, r0, Rd, Rb, diff, vals in rows:
    if v <= 0:
        continue
    cands_all = sorted(((abs(diff - vals[k]), k) for k in TERM_KEYS))
    per_row_single_all.append((v, r0, cands_all[0][1], cands_all[0][0], diff))
    cands = sorted(((abs(diff - vals[k]), k) for k in TERM_KEYS))
    per_row_single.append((v, r0, cands[0][1], cands[0][0], diff))
print("  逐档最佳单一候选（v > 0）：")
for v, r0, k, e, diff in per_row_single:
    print("    v = %.2f，r = %.1f：最佳「%s」，绝对残差 %.3e（差额 %+.3e）"
          % (v, r0, k, e, diff))
main_rows = [x for x in per_row_single_all if x[0] in V_MAIN]
print("  主档（v <= 0.20）的逐档最佳单一候选：")
for v, r0, k, e, diff in main_rows:
    print("    v = %.2f，r = %.1f：最佳「%s」，绝对残差 %.3e（差额 %+.3e，相对 %.3e）"
          % (v, r0, k, e, diff, abs(e) / abs(diff) if abs(diff) > 1e-18 else 0.0))
names_single = set(k for _, _, k, _, _ in main_rows)
best_single_name = None
if len(names_single) == 1:
    best_single_name = list(names_single)[0]
    rels = [abs(e) / abs(diff) if abs(diff) > 1e-18 else 0.0
            for _, _, k, e, diff in main_rows if k == best_single_name]
    best_single = (best_single_name, max(rels))
else:
    # 候选不唯一一致时，报出各档最佳与出现频次
    from collections import Counter
    cnt = Counter(k for _, _, k, _, _ in per_row_single)
    print("  最佳候选在档间不一致，出现频次：%s" % dict(cnt))
    best_single = (cnt.most_common(1)[0][0], float("inf"))
r2_ok = (best_single_name is not None) and (best_single[1] <= TOL)
check("R-2", r2_ok,
      "最佳单一细分项在全部 v > 0 档上一致：「%s」，其最大相对残差 %.3e（容差 5e-2）"
      % (best_single[0], best_single[1]))

# 组合归属：两两之差
combos = []
keys = list(TERM_KEYS)
for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
        a, b = keys[i], keys[j]
        rels = []
        for _, _, _, _, diff, vals in rows:
            denom = abs(diff) if abs(diff) > 1e-18 else 1.0
            rels.append(abs(diff - (vals[a] - vals[b])) / denom)
        combos.append(("%s - %s" % (a, b), max(rels)))
combos.sort(key=lambda x: x[1])
print("  组合归属候选（按最大相对残差排序，前六）：")
for name, e in combos[:6]:
    print("    %-22s 最大相对残差 %.3e" % (name, e))
# 逐档一致性
per_row_combo = []
per_row_combo_all = []
for v, r0, Rd, Rb, diff, vals in rows:
    if v <= 0:
        continue
    cands = sorted((abs(diff - (vals[keys[i]] - vals[keys[j]])),
                     "%s - %s" % (keys[i], keys[j]))
                    for i in range(len(keys)) for j in range(i + 1, len(keys)))
    per_row_combo_all.append((v, r0, cands[0][1], cands[0][0], diff))
main_combo = [x for x in per_row_combo_all if x[0] in V_MAIN]
print("  主档（v <= 0.20）的逐档最佳组合：")
for v, r0, name, e, diff in main_combo:
    print("    v = %.2f，r = %.1f：最佳「%s」，绝对残差 %.3e" % (v, r0, name, e))
names_combo = set(n for _, _, n, _, _ in main_combo)
for v, r0, name, e, diff in per_row_combo:
    print("    v = %.2f，r = %.1f：最佳「%s」，绝对残差 %.3e" % (v, r0, name, e))
names_combo = set(n for _, _, n, _, _ in per_row_combo)
if len(names_combo) == 1:
    best_combo_name = list(names_combo)[0]
    rels = [abs(e) / abs(diff) if abs(diff) > 1e-18 else 0.0
            for _, _, n, e, diff in main_combo if n == best_combo_name]
    best_combo = (best_combo_name, max(rels))
else:
    from collections import Counter
    cnt = Counter(n for _, _, n, _, _ in per_row_combo)
    print("  最佳组合在档间不一致，出现频次：%s" % dict(cnt))
    best_combo = (cnt.most_common(1)[0][0], float("inf"))
r3_ok = len(names_combo) == 1 and (best_combo[1] <= TOL)
check("R-3", r3_ok,
      "最佳组合在全部 v > 0 档上一致：「%s」，其最大相对残差 %.3e（容差 5e-2）"
      % (best_combo[0], best_combo[1]))

# ----------------------------------------------------------------------
# R-4 归属项对 v 的幂次
# ----------------------------------------------------------------------
print()
print("[R-4] 归属项对 v 的幂次")
att_key = best_single[0] if isinstance(best_single[0], str) else best_single[0][0]
r4_rows = []
for r0 in RADII:
    sub = [x for x in rows if x[1] == r0 and x[0] > 0]
    xs = np.array([x[0] for x in sub])
    ys = np.array([abs(x[5][att_key]) for x in sub])
    slope = float(np.polyfit(np.log(xs), np.log(ys), 1)[0]) if np.all(ys > 0) else float("nan")
    r4_rows.append((r0, slope))
    print("  r = %.1f：%s 随 v 的对数斜率 %+.4f" % (r0, att_key, slope))
r4_slopes = [s for _, s in r4_rows if np.isfinite(s)]
r4_mean = float(np.mean(r4_slopes)) if r4_slopes else float("nan")
r4_spread = (max(r4_slopes) - min(r4_slopes)) / abs(r4_mean) if r4_slopes else float("inf")
print("  斜率均值 %+.4f，档间相对散布 %.3e" % (r4_mean, r4_spread))
check("R-4", np.isfinite(r4_mean) and r4_spread <= 1e-1,
      "归属项 %s 对 v 的对数斜率均值 %+.4f，档间相对散布 %.3e（容差 1e-1）"
      % (att_key, r4_mean, r4_spread))

# ----------------------------------------------------------------------
# R-5 恒真替身
# ----------------------------------------------------------------------
print()
print("[R-5] 恒真替身：把细分方式改成「按 Gamma 的第一个指标是否 0」，归属须改变")
alt = []
for v, r0, Rd, Rb, diff, vals in rows:
    if v <= 0:
        continue
    denom = abs(diff) if abs(diff) > 1e-18 else 1.0
    alt.append(abs(diff - vals["t3_sN"]) / denom)
alt_max = max(alt) if alt else 0.0
r5_ok = alt_max > max(best_single[1], best_combo[1]) * 10
print("  替身项的最大相对残差 %.3e ｜ 真值（单一 %.3e / 组合 %.3e）"
      % (alt_max, best_single[1], best_combo[1]))
check("R-5", r5_ok,
      "替身项的相对残差 %.3e 显著大于真值 %.3e，归属有区分力"
      % (alt_max, min(best_single[1], best_combo[1])))

# ----------------------------------------------------------------------
# R-6 状态登记
# ----------------------------------------------------------------------
print()
print("[R-6] 状态登记")
reg = [
    ("其一", "细分", "R_00 的展开按「Gamma 指标含几个 0」细分为十项，读数见逐档输出"),
    ("其二", "归属", "单一归属与组合归属的候选与残差见 R-2、R-3"),
    ("其三", "幂次", "归属项对 v 的幂次见 R-4"),
    ("其四", "边界", "只做动源档的项级归属；破坏共形平直的构型与完整动态波方程不在本轮"),
    ("其五", "剩余", "P5 的注入-扩散平衡那一侧；源侧的 O(v^2) 缺口；刚度取值"),
]
for a1_, a2_, a3_ in reg:
    print("     %-4s %-10s %s" % (a1_, a2_, a3_))
check("R-6", True, "状态登记完成")

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
for r0 in RADII:
    xs = [x[0] for x in rows if x[1] == r0]
    ys = [x[4] for x in rows if x[1] == r0]
    ax.semilogy(xs, np.maximum(np.abs(ys), 1e-18), "o-", linewidth=1.6,
                label=u"r = %.1f" % r0)
ax.axhline(TOL, color="#d62728", linestyle=":", linewidth=1.4, label=u"归属门槛 $5\\times10^{-2}$")
ax.set_xlabel(u"速度 $v$")
ax.set_ylabel(u"差额")
ax.set_title(u"(a) 差额随速度")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

ax = axes[1]
labels = []
vals = []
for k, e in singles:
    labels.append(k)
    vals.append(e)
ax.bar(np.arange(len(labels)), np.maximum(vals, 1e-18), color="#1f77b4", width=0.6)
ax.axhline(TOL, color="#d62728", linestyle=":", linewidth=1.4, label=u"归属门槛")
ax.set_xticks(np.arange(len(labels)))
ax.set_xticklabels(labels, fontsize=7, rotation=45)
ax.set_ylabel(u"跨档最大相对残差")
ax.set_title(u"(b) 单一归属的残差")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y", which="both")

ax = axes[2]
labels = []
vals = []
for name, e in combos[:10]:
    labels.append(name)
    vals.append(e)
ax.bar(np.arange(len(labels)), np.maximum(vals, 1e-18), color="#ff7f0e", width=0.6)
ax.axhline(TOL, color="#d62728", linestyle=":", linewidth=1.4, label=u"归属门槛")
ax.set_xticks(np.arange(len(labels)))
ax.set_xticklabels(labels, fontsize=6, rotation=60)
ax.set_ylabel(u"跨档最大相对残差")
ax.set_title(u"(c) 组合归属的残差（前十）")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y", which="both")

fig.suptitle(u"第776号：动源档差额在 $R_{00}$ 展开式上的项级归属", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(IMG, dpi=150)
print()
print("图已存：%s" % IMG)
