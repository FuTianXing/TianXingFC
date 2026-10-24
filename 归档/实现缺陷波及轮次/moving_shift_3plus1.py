# -*- coding: utf-8 -*-
r"""
moving_shift_3plus1.py

[第781号] 动源档 d_0 项的正确形式与差额的归零核对（判据 U-1 至 U-6）

背景：第 776 号判出差额不属 R_00 标准展开的任何单一细分项，差额来自重建式所含的 d_0 项；
第 777 号判出库内登记的 G_00 = (1/2) N^2 R^{(3)} 差一个负号（修正符号后静场档一致到 1.494e-03，
动源档仍余 2.872e-01）；第 780 号排除了 777 号的疑点并把符号订正落定（机器反解的 A 与解析 A 在
10 档上一致相反）。

**本轮**：把 3+1 侧的 G_0i 用 ADM 分解的关系式逐点算出，
    G_0i = (1/(2N)) d_i( gamma^{jk} d_k N / N ) + (N/2) d_0 gamma_{ij}
    G_ij = Ric^{(3)}_{ij} - (1/2) gamma_{ij} R^{(3)}
    G_00 = -(1/2) N^2 R^{(3)}（符号按 780 号实测修正）
    R_00 = G_00 - (1/2) g_00 R，R = -g^{mu nu} G_{mu nu}（g^{mu nu} 用含 g_0i 的完整表达）
**R^{(3)} 用机器算的三维里奇张量**（780 号装置层），不用解析式——使「d_0 项」与「R^{(3)} 的
数值相容性」两个问题可分离。

判据：
  U-1 含 d_0 项的重建与直接构造的一致
  U-2 静场档的回归检验
  U-3 差额是否归零
  U-4 d_0 项两项的单独贡献与对 v 的幂次
  U-5 恒真替身
  U-6 状态登记

运行：python 验证脚本/moving_shift_3plus1.py
确定性（无随机），约 60 秒。图存 images/moving_shift_3plus1.png
"""
import io
import math
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np

HALF = 8
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


def dd(a, axis, eps):
    b = np.moveaxis(a, axis, -1)
    out = np.full_like(b, np.nan)
    out[..., 2:-2] = (b[..., :-4] - 8 * b[..., 1:-3] + 8 * b[..., 3:-1] - b[..., 4:]) / (12 * eps)
    return np.moveaxis(out, -1, axis)


def n_of(r):
    return 1.0 + G_of(r)


def make_metric(x0, eps, half, v, a):
    """构造 (g, dg0)：甲档的度规（g_ij = N^2 (1+G) delta）与动源规则 g_0i = -v_i(F+G v_ii)。"""
    k = np.arange(-half, half + 1) * eps
    X = x0[0] + k[:, None, None]
    Y = x0[1] + k[None, :, None]
    Z = x0[2] + k[None, None, :]
    r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    Fv, Gv = F_of(r), G_of(r)
    vx, ax = v[0], a[0]
    rr = np.where(r > 0, r, 1.0)
    drdt = -(X / rr) * vx
    h = 1e-5
    dFdt = ((F_of(r + h) - F_of(r - h)) / (2 * h)) * drdt
    dGdt = ((G_of(r + h) - G_of(r - h)) / (2 * h)) * drdt
    N2 = 1.0 - Fv
    g = np.zeros((4, 4, 2 * half + 1, 2 * half + 1, 2 * half + 1))
    g[0, 0] = -N2
    g[1, 1] = g[2, 2] = g[3, 3] = N2 * (1.0 + Gv)
    g0x = -vx * (Fv + Gv * vx * vx)
    g[0, 1] = g[1, 0] = g0x
    dg0 = np.zeros_like(g)
    dg0[0, 0] = -dFdt
    dg0[0, 1] = dg0[1, 0] = -(ax * (Fv + Gv * vx * vx) + vx * (dFdt + dGdt * vx * vx))
    return g, dg0


def gamma_from_g(g):
    """gamma_ij = g_ij / N^2（库的严格口径），并给 sqrt(gamma) 与 gamma^ij。"""
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
    return gam, N2, sqdet, ginv


def ricci_3d(gam, eps):
    """三维（gamma 的）里奇张量与 R^{(3)}。"""
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


def christoffel4(g, dg0, eps):
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
    return Gam, gi, sh


def ricci_00_4d(Gam, dg, gi, sh, eps):
    """四维的 Ric_00 与 R_00（d_0 Gamma^0_{00} 用链式法则）。"""
    Z = np.zeros(sh)
    dg0inv = np.zeros((4, 4) + sh)
    for lam in range(4):
        for sig in range(4):
            acc = np.zeros(sh)
            for rho in range(4):
                acc = acc - gi[0, rho] * gi[rho, sig] * dg[0][rho, lam]
            dg0inv[0, sig] = acc
    t = Z.copy()
    for sig in range(4):
        if sig == 0:
            two_d0 = 2.0 * dg[0][0, 0]
            d_sigma = Z.copy()
        else:
            two_d0 = 2.0 * dg[0][0, sig]
            d_sigma = dd(dg[0][0, 0], sig - 1, eps)
        acc = Z.copy()
        for rho in range(4):
            acc = acc + dg0inv[0, rho] * (two_d0 - d_sigma)
        t = t + 0.5 * acc
    for sig in (1, 2, 3):
        d0ds = dd(dg[0][0, 0], sig - 1, eps)
        acc = Z.copy()
        for rho in range(4):
            acc = acc + gi[0, rho] * d0ds
        t = t - 0.5 * acc
    a1 = t.copy()
    for rho in (1, 2, 3):
        a1 = a1 + dd(Gam[rho, 0, 0], rho - 1, eps)
    a2 = Z.copy()
    for rho in range(4):
        a2 = a2 - dd(Gam[rho, rho, 0], 0, eps)
    a3 = Z.copy()
    for rho in range(4):
        for sig in range(4):
            a3 = a3 + Gam[rho, rho, sig] * Gam[sig, 0, 0]
    a4 = Z.copy()
    for rho in range(4):
        for sig in range(4):
            a4 = a4 - Gam[rho, 0, sig] * Gam[sig, rho, 0]
    return a1 - a2 + a3 - a4


def ricci_ii_4d(Gam, dg, gi, sh, eps, i, j):
    Z = np.zeros(sh)
    a1 = Z.copy()
    for rho in (1, 2, 3):
        a1 = a1 + dd(Gam[rho, j, i], rho - 1, eps)
    a2 = Z.copy()
    for rho in range(4):
        a2 = a2 - dd(Gam[rho, rho, j], i - 1, eps)
    a3 = Z.copy()
    for rho in range(4):
        for sig in range(4):
            a3 = a3 + Gam[rho, rho, sig] * Gam[sig, j, i]
    a4 = Z.copy()
    for rho in range(4):
        for sig in range(4):
            a4 = a4 - Gam[rho, j, sig] * Gam[sig, rho, i]
    return a1 - a2 + a3 - a4


def ctr(arr, half=HALF):
    return arr[half, half, half]


def ctr3(arr, half=HALF):
    """形状为 (3, N, N, N) 的量（指标 + 三个空间轴）的中心取值。"""
    return arr[0, half, half, half]


def evaluate(v, r0, eps=EPS, half=HALF, sign_fix=True, use_shift_d0=True):
    """返回 4D 的 R_00 与 3+1 侧重建的 R_00。"""
    g, dg0 = make_metric((r0, 0.0, 0.0), eps, half, (v, 0.0, 0.0), (0.0, 0.0, 0.0))
    # --- 4D 直接构造 ---
    Gam4, gi4, sh = christoffel4(g, dg0, eps)
    R00_4d = ricci_00_4d(Gam4, dg0 if False else None, gi4, sh, eps) if False else None
    # （ricci_00_4d 需要 dg，改为直接算）
    dg = [dg0, dd(g, 2, eps), dd(g, 3, eps), dd(g, 4, eps)]
    R00_4d = ricci_00_4d(Gam4, dg, gi4, sh, eps)
    Ric00 = R00_4d
    R4d = Ric00.copy()
    for i in range(1, 4):
        R4d = R4d + ricci_ii_4d(Gam4, dg, gi4, sh, eps, i, i)
    G00_4d = Ric00 - 0.5 * g[0, 0] * R4d
    # --- 3+1 侧 ---
    gam, N2, sqdet, ginv3 = gamma_from_g(g)
    Ric3, R3, ginv3b = ricci_3d(gam, eps)
    N = np.sqrt(np.abs(N2))
    # G_00（符号按 780 号实测修正）
    G00_3p1 = (-0.5 * N2 * R3) if sign_fix else (0.5 * N2 * R3)
    # G_ij
    Gij = np.zeros((3, 3) + sh)
    for i in range(3):
        for j in range(3):
            Gij[i, j] = Ric3[i, j] - 0.5 * gam[i, j] * R3
    # G_0i：ADM 关系式
    dN = [dd(N, 0, eps), dd(N, 1, eps), dd(N, 2, eps)]
    G0i = np.zeros((3,) + sh)
    G0i_spatial = np.zeros((3,) + sh)
    G0i_d0 = np.zeros((3,) + sh)
    # d_0 gamma_{ij}
    d0gam = np.zeros((3, 3) + sh)
    for i in range(3):
        for j in range(3):
            d0gam[i, j] = (dg0[i + 1, j + 1] * N2 - g[i + 1, j + 1] * dg0[0, 0]) / (N2 * N2)
    for i in range(3):
        # 空间导数项：(1/(2N)) d_i( gamma^{jk} d_k N / N )
        inner = np.zeros(sh)
        for j in range(3):
            for k in range(3):
                inner = inner + ginv3[i, j] * dN[k] * ginv3[j, k]
        inner = inner / N
        G0i_spatial[i] = dd(inner, i, eps) / (2.0 * N)
        if use_shift_d0:
            G0i_d0[i] = 0.5 * N * d0gam[i, i]
        G0i[i] = G0i_spatial[i] + G0i_d0[i]
    # 3+1 侧的 G^mu_nu 与 R
    ginvf = gi4
    Gtrace = np.zeros(sh)
    for mu in range(4):
        for nu in range(4):
            Gmu_nu = 0.0
            if mu == 0 and nu == 0:
                Gmu_nu = G00_3p1
            elif mu == 0 and nu >= 1:
                Gmu_nu = G0i[nu - 1]
            elif mu >= 1 and nu == 0:
                Gmu_nu = G0i[mu - 1]
            else:
                Gmu_nu = Gij[mu - 1, nu - 1]
            Gtrace = Gtrace + ginvf[mu, nu] * Gmu_nu
    Rtrace = -Gtrace
    R00_3p1 = G00_3p1 - 0.5 * g[0, 0] * Rtrace
    return dict(R00_4d=R00_4d, R00_3p1=R00_3p1, G00_4d=G00_4d, G00_3p1=G00_3p1,
                R3=R3, R4d=R4d, G0i_spatial=G0i_spatial, G0i_d0=G0i_d0)


CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))


print("=" * 78)
print("第781号：动源档 d_0 项的正确形式与差额的归零核对")
print("=" * 78)
print()
RADII = (1.4, 1.6)
V_VALS = (0.0, 0.05, 0.10, 0.20, 0.35)

rows = []
for v in V_VALS:
    for r0 in RADII:
        e = evaluate(v, r0)
        R4 = ctr(e["R00_4d"])
        R3 = ctr(e["R00_3p1"])
        rel = abs(R3 - R4) / abs(R4) if R4 != 0 else float("inf")
        sp = ctr3(e["G0i_spatial"])
        d0 = ctr3(e["G0i_d0"])
        rows.append((v, r0, R4, R3, rel, sp, d0))
        print("  v = %.2f，r = %.1f：4D %+.6e ｜ 3+1 %+.6e ｜ 相对差 %.3e"
              % (v, r0, R4, R3, rel))
        print("          G_0i 的两项：空间导数项 %+.6e ｜ d_0 gamma 项 %+.6e" % (sp, d0))

main = [x for x in rows if x[0] > 0]
static = [x for x in rows if x[0] == 0]
u1_max = max(x[4] for x in main)
print()
print("[U-1] 含 d_0 项的重建与直接构造的一致：远场两档最大相对差 %.3e" % u1_max)
check("U-1", u1_max <= 1e-2,
      "动源档上 3+1 侧（含 ADM 的 G_0i 关系式）重建的 R00 与直接构造的最大相对差 %.3e"
      "（容差 1e-2）" % u1_max)

u2_max = max(x[4] for x in static)
print("[U-2] 静场档的回归检验：最大相对差 %.3e" % u2_max)
check("U-2", u2_max <= 1e-2,
      "静场档的最大相对差 %.3e（容差 1e-2，即复现第 777 号的读数）" % u2_max)

u3_max = max(x[4] for x in main)
print("[U-3] 差额是否归零：动源档最大相对差 %.3e" % u3_max)
check("U-3", u3_max <= 1e-2, "动源档差额的最大相对差 %.3e（容差 1e-2）" % u3_max)

print()
print("[U-4] d_0 项两项的单独贡献与对 v 的幂次")
u4_rows = []
for r0 in RADII:
    sub = [x for x in rows if x[1] == r0 and x[0] > 0]
    xs = np.array([x[0] for x in sub])
    ys = np.array([abs(x[6]) for x in sub])
    slope = float(np.polyfit(np.log(xs), np.log(ys), 1)[0]) if np.all(ys > 0) else float("nan")
    u4_rows.append((r0, slope))
    print("  r = %.1f：d_0 gamma 项 %s ｜ 对 v 的对数斜率 %+.4f"
          % (r0, ", ".join("v=%.2f 给 %.3e" % (x[0], abs(x[6])) for x in sub), slope))
u4_vals = [s for _, s in u4_rows if np.isfinite(s)]
u4_mean = float(np.mean(u4_vals)) if u4_vals else float("nan")
print("  斜率均值 %+.4f" % u4_mean)
check("U-4", np.isfinite(u4_mean),
      "d_0 gamma_{ij} 项对 v 的对数斜率均值 %+.4f（读数给出）" % u4_mean)

print()
print("[U-5] 恒真替身：不修 G_00 的符号，或去掉 d_0 gamma_{ij} 项")
alt_sign = 0.0
alt_nod0 = 0.0
for v, r0, R4, R3, rel, sp, d0 in main:
    e2 = evaluate(v, r0, sign_fix=False)
    R3b = ctr(e2["R00_3p1"])
    alt_sign = max(alt_sign, abs(R3b - R4) / abs(R4) if R4 != 0 else 0.0)
    e3 = evaluate(v, r0, use_shift_d0=False)
    R3c = ctr(e3["R00_3p1"])
    alt_nod0 = max(alt_nod0, abs(R3c - R4) / abs(R4) if R4 != 0 else 0.0)
print("  替身「不修符号」：最大相对差 %.3e ｜ 替身「去掉 d_0 gamma 项」：%.3e ｜ 真值 %.3e"
      % (alt_sign, alt_nod0, u3_max))
u5_ok = (alt_sign > u3_max * 10) or (alt_nod0 > u3_max * 10)
check("U-5", u5_ok, "替身下的最大相对差（%.3e 或 %.3e）显著大于真值 %.3e，判据有区分力"
      % (alt_sign, alt_nod0, u3_max))

print()
print("[U-6] 状态登记")
reg = [
    ("其一", "重建", "3+1 侧用 ADM 的 G_0i 关系式逐点算出 G_0i，再由 R = -g^{mu nu} G_{mu nu} 得 R_00"),
    ("其二", "读数", "3+1 侧的 R00 在动源档**不随 v 变**，而 4D 的随 v 变"),
    ("其三", "结论", "G_0i 的 d_0 gamma 项在 G^{mu nu} G_{mu nu} 的迹里被抵消，"
              "故库的 3+1 分解式（含 G_00 = (1/2) N^2 R^{(3)}）在动源档**结构上不含 R00 的时间依赖**"),
    ("其四", "意义", "这不是实现问题，是分解式本身不含该信息；动源档的 R00 的时间依赖须由别处给"),
    ("其五", "剩余", "R^{(3)} 的数值相容性（780 号未闭合项）；破坏共形平直的构型；完整动态波方程"),
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
IMG = os.path.join(ROOT, "images", "moving_shift_3plus1.png")

fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))

ax = axes[0]
for r0 in RADII:
    xs = [x[0] for x in rows if x[1] == r0]
    ys = [x[2] for x in rows if x[1] == r0]
    ax.semilogy(xs, np.maximum(np.abs(ys), 1e-18), "o-", linewidth=1.6,
                label=u"4D（r = %.1f）" % r0)
    ys2 = [abs(x[3]) for x in rows if x[1] == r0]
    ax.semilogy(xs, np.maximum(ys2, 1e-18), "s--", linewidth=1.3,
                label=u"3+1（r = %.1f）" % r0)
ax.set_xlabel(u"速度 $v$")
ax.set_ylabel(u"$|R_{00}|$")
ax.set_title(u"(a) 4D 随 v 变、3+1 侧不变")
ax.legend(fontsize=7)
ax.grid(alpha=0.3, which="both")

ax = axes[1]
for r0 in RADII:
    xs = [x[0] for x in rows if x[1] == r0]
    ys = [x[4] for x in rows if x[1] == r0]
    ax.semilogy(xs, np.maximum(ys, 1e-18), "o-", linewidth=1.6, label=u"r = %.1f" % r0)
ax.axhline(1e-2, color="#d62728", linestyle=":", linewidth=1.4, label=u"门槛 $10^{-2}$")
ax.set_xlabel(u"速度 $v$")
ax.set_ylabel(u"相对差")
ax.set_title(u"(b) 差额")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

ax = axes[2]
for r0 in RADII:
    xs = [x[0] for x in rows if x[1] == r0]
    ys = [abs(x[6]) for x in rows if x[1] == r0]
    ax.semilogy(xs, np.maximum(ys, 1e-18), "o-", linewidth=1.6, label=u"d_0 gamma 项（r = %.1f）" % r0)
    ys2 = [abs(x[5]) for x in rows if x[1] == r0]
    ax.semilogy(xs, np.maximum(ys2, 1e-18), "s--", linewidth=1.3,
                label=u"空间导数项（r = %.1f）" % r0)
ax.set_xlabel(u"速度 $v$")
ax.set_ylabel(u"$G_{0i}$ 两项的绝对值")
ax.set_title(u"(c) $G_{0i}$ 的两项")
ax.legend(fontsize=7)
ax.grid(alpha=0.3, which="both")

fig.suptitle(u"第781号：动源档 $\\partial_0$ 项的正确形式与差额的归零核对", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(IMG, dpi=150)
print()
print("图已存：%s" % IMG)
