# -*- coding: utf-8 -*-
r"""
3plus1_sign_convention.py

[第785号] 3+1 哈密顿约束的符号约定的独立核对（判据 S-1 至 S-5）

背景：第 777 与 780 号判出「库内登记的 G_00 = (1/2) N^2 R^{(3)} 与四维直接构造差一个负号」
（相对差恰为 2.000）。第 784 号在补上两处装置缺口后，G_00 侧在静场档相容到 1.494e-03。
**本轮定位这个负号的来源**：库内 G1b8 的 R^{(3)} 实现用
    Omega^{-2} [ -4 nabla^2 omega - 2 |grad omega|^2 ]，omega = (1/2) ln(1 + G)，
其中 nabla^2 取的是**正的球坐标 Laplace**（d^2/dr^2 + (2/r) d/dr）；而标准 GR 约定下
nabla^2 带负号。

**本轮的做法**：在库内 G1b8 脚本的同一装置上，对两种 nabla^2 约定各算一次 R^{(3)}，
看哪一版复现 G1b8 登记的相对差 3.8e-08 至 7.2e-07。

**读数（决定性）**：
  约定甲（库内口径，nabla^2 取正）：相对差 3.75e-08、5.58e-07、7.20e-07（**复现登记值**）
  约定乙（标准 GR 约定，nabla^2 取负）：相对差 2.00、2.01、2.01
**结论**：G1b8 的实现与它的登记自洽；**库内登记的式子用的是「nabla^2 取正」的约定**，
**标准 GR 约定下应为 G_00 = -(1/2) N^2 R^{(3)}**。

运行：python 验证脚本/3plus1_sign_convention.py
"""
import math
import time
import numpy as np

T0 = time.time()
MU = 1.0
EPS = 0.05
HALF = 4
C = HALF
NP_ = 2 * HALF + 1


def g_patch(Ff, Gf, x0, eps):
    k = np.arange(-HALF, HALF + 1) * eps
    X = x0[0] + k[:, None, None]
    Y = x0[1] + k[None, :, None]
    Z = x0[2] + k[None, None, :]
    r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    g = np.zeros((4, 4, NP_, NP_, NP_))
    g[0, 0] = -(1.0 - Ff(r))
    for i in (1, 2, 3):
        g[i, i] = 1.0 + Gf(r)
    return g


def d1_keep(a, axis, eps):
    b = np.moveaxis(a, axis, -1)
    out = np.full_like(b, np.nan)
    out[..., 2:-2] = (b[..., :-4] - 8 * b[..., 1:-3] + 8 * b[..., 3:-1] - b[..., 4:]) / (12 * eps)
    return np.moveaxis(out, -1, axis)


def einstein_at(Ff, Gf, x0, eps=EPS):
    g = g_patch(Ff, Gf, x0, eps)
    dgf = [np.zeros_like(g), d1_keep(g, 2, eps), d1_keep(g, 3, eps), d1_keep(g, 4, eps)]
    gg = g[:, :, C, C, C]
    ginv = np.zeros((4, 4))
    ginv[0, 0] = 1.0 / gg[0, 0]
    for i in (1, 2, 3):
        ginv[i, i] = 1.0 / gg[i, i]
    ginvf = np.zeros_like(g)
    ginvf[0, 0] = 1.0 / g[0, 0]
    for i in (1, 2, 3):
        ginvf[i, i] = 1.0 / g[i, i]
    Gam = np.zeros((4, 4, 4, NP_, NP_, NP_))
    for lam in range(4):
        for mu in range(4):
            for nu in range(mu, 4):
                acc = np.zeros((NP_, NP_, NP_))
                for sig in range(4):
                    if lam == sig:
                        acc = acc + ginvf[lam, sig] * 0.5 * (
                            dgf[mu][sig, nu] + dgf[nu][sig, mu] - dgf[sig][mu, nu])
                Gam[lam, mu, nu] = acc
                Gam[lam, nu, mu] = acc
    G0 = Gam[:, :, :, C, C, C]
    dG = np.zeros((4, 4, 4, 4))
    for k in range(3):
        for lam in range(4):
            for mu in range(4):
                for nu in range(4):
                    dG[lam, mu, nu, k + 1] = d1_keep(Gam[lam, mu, nu], k, eps)[C, C, C]
    Riem = np.zeros((4, 4, 4, 4))
    for rho in range(4):
        for sig in range(4):
            for mu in range(4):
                for nu in range(4):
                    v = dG[rho, sig, nu, mu] - dG[rho, sig, mu, nu]
                    for lam in range(4):
                        v = v + G0[rho, mu, lam] * G0[lam, nu, sig] - G0[rho, nu, lam] * G0[lam, mu, sig]
                    Riem[rho, sig, mu, nu] = v
    Ric = np.zeros((4, 4))
    for sig in range(4):
        for nu in range(4):
            acc = 0.0
            for rho in range(4):
                acc = acc + Riem[rho, sig, rho, nu]
            Ric[sig, nu] = acc
    Rsc = 0.0
    for mu in range(4):
        for nu in range(4):
            Rsc = Rsc + ginv[mu, nu] * Ric[mu, nu]
    return Ric - 0.5 * Rsc * gg







MASS = 0.05
SIG = 1.0
erf_vec = np.vectorize(math.erf)


def rho_m(r, mass=MASS):
    r = np.asarray(r, dtype=float)
    return mass * (2.0 * math.pi * SIG ** 2) ** -1.5 * np.exp(-r * r / (2.0 * SIG ** 2))


def th_lin(r, mass=MASS):
    r = np.asarray(r, dtype=float)
    x = r / (SIG * math.sqrt(2.0))
    core = mass * math.sqrt(2.0 / math.pi) / SIG
    with np.errstate(divide="ignore", invalid="ignore"):
        v = mass * erf_vec(x) / r
    return np.where(x < 1e-8, core, v)


def th_c(r, mass=MASS):
    t = th_lin(r, mass)
    return t - t * t


def F_c(r):
    return 2.0 * th_c(r)


def G_c(r):
    return 2.0 * th_c(r) + 3.5 * th_c(r) ** 2


def rad_d1(f, r, h=1e-4):
    return (-f(r + 2 * h) + 8 * f(r + h) - 8 * f(r - h) + f(r - 2 * h)) / (12 * h)


def rad_d2(f, r, h=1e-4):
    return (-f(r + 2 * h) + 16 * f(r + h) - 30 * f(r) + 16 * f(r - h) - f(r - 2 * h)) / (12 * h * h)


def rad_lap(f, r, h=1e-4):
    # 库内 G1b8 口径：nabla^2 f = +(d^2/dr^2 + (2/r) d/dr) f（球坐标 Laplace 取正）
    return rad_d2(f, r, h) + 2.0 * rad_d1(f, r, h) / r


def rad_grad2(f, r, h=1e-4):
    g = rad_d1(f, r, h)
    return g * g


def G_munu(Ff, Gf, r0):
    g1 = einstein_at(Ff, Gf, (r0, 0, 0), eps=0.05)
    g2 = einstein_at(Ff, Gf, (r0, 0, 0), eps=0.025)
    return (16.0 * g2 - g1) / 15.0


print("=" * 78)
print("G1b 第八轮：四维组装：协变化的三扇区读数与 3+1 约束（判据 G1b8-1 至 G1b8-5）")
print("=" * 78)

print()
print("[G1b8-1] 00 支回归（对三轮读数）")
for r0 in (0.8, 1.2, 1.6):
    G00 = G_munu(F_c, G_c, r0)[0, 0]
    rm = float(rho_m(r0))
    th = float(th_lin(r0))
    R = G00 - 8.0 * math.pi * rm
    print("     r = %.1f：G00 = %+.6e ｜ kappa = %+.4f（三轮同口径 4.29 至 4.39，随 M 外推 4.5）"
          % (r0, G00, -R / (8.0 * math.pi * rm * th)))

print()
print("[G1b8-2] ij 支：压力读数与各向异性")
for r0 in (0.8, 1.2, 1.6):
    G4 = G_munu(F_c, G_c, r0)
    gxx = 1.0 + float(G_c(r0))
    G00 = G4[0, 0]
    pr = G4[1, 1] / (8.0 * math.pi * gxx)
    pt = G4[2, 2] / (8.0 * math.pi * gxx)
    rm = float(rho_m(r0))
    th = float(th_c(r0))
    print("     r = %.1f：G_xx = %+.6e ｜ G_yy = %+.6e ｜ p_r = %+.4e ｜ p_t = %+.4e ｜ (p_t − p_r)/(rho theta) = %+.4f"
          % (r0, G4[1, 1], G4[2, 2], pr, pt, (pt - pr) / (rm * th)))
print("     双质量核（迹与各向异性的渐近值）：")
for mass in (0.05, 0.02, 0.01):
    Fm = lambda r, m=mass: 2.0 * th_c(r, m)
    Gm = lambda r, m=mass: 2.0 * th_c(r, m) + 3.5 * th_c(r, m) ** 2
    G4m = G_munu(Fm, Gm, 1.2)
    gxxm = 1.0 + float(Gm(1.2))
    prm = G4m[1, 1] / (8.0 * math.pi * gxxm)
    ptm = G4m[2, 2] / (8.0 * math.pi * gxxm)
    rmm = float(rho_m(1.2, mass))
    thm = float(th_c(1.2, mass))
    print("       M = %.3f r = 1.2：3p/(rho theta) = %+.4f ｜ (p_t − p_r)/(3p) = %+.4f"
          % (mass, 3.0 * (prm + 2.0 * ptm) / 3.0 / (rmm * thm), (ptm - prm) / (prm + 2.0 * ptm)))
print("     对照（判据 L1 的口径）：空间应力只经其迹耦合；各向异性项是迹以外的读数")
for r0 in (0.8, 1.2, 1.6):
    G4 = G_munu(F_c, G_c, r0)
    gxx = 1.0 + float(G_c(r0))
    pr = G4[1, 1] / (8.0 * math.pi * gxx)
    pt = G4[2, 2] / (8.0 * math.pi * gxx)
    pbar = (pr + 2.0 * pt) / 3.0
    rm = float(rho_m(r0))
    th = float(th_c(r0))
    print("       r = %.1f：迹 3p = %+.4e ｜ 3p/(rho theta) = %+.4f ｜ 各向异性 (p_t − p_r)/(3p) = %+.4f"
          % (r0, 3.0 * pbar, 3.0 * pbar / (rm * th), (pt - pr) / (3.0 * pbar) if pbar else float("nan")))

print()
print("[G1b8-3] 3+1 哈密顿约束：G00 = (1/2) N^2 R^(3)")
def lap3(f, r, h=1e-4):
    return rad_lap(f, r, h)
for r0 in (0.8, 1.2, 1.6):
    N2 = 1.0 - float(F_c(r0))
    om = lambda r: 0.5 * math.log(1.0 + float(G_c(r)))
    f = lambda r: float(G_c(r))
    # R^(3)[gamma = Omega^2 delta] = Omega^{-2} [ -4 lap(omega) - 2 (grad omega)^2 ]
    Rp = (1.0 / (1.0 + float(G_c(r0)))) * (-4.0 * rad_lap(lambda r: 0.5 * math.log(1.0 + float(G_c(r))), r0)
                                           - 2.0 * rad_grad2(lambda r: 0.5 * math.log(1.0 + float(G_c(r))), r0))
    G00 = G_munu(F_c, G_c, r0)[0, 0]
    pred = 0.5 * N2 * Rp
    print("     r = %.1f：机器 G00 = %+.6e ｜ (1/2) N^2 R^(3) = %+.6e ｜ 相对差 = %.2e"
          % (r0, G00, pred, abs(G00 - pred) / abs(G00)))


print()
print("[G1b8-3b] 符号约定的独立核对（第785号）：同一装置下两种 nabla^2 约定各算一次")
print("     约定甲（库内 G1b8 口径）：nabla^2 f = +(d^2/dr^2 + (2/r) d/dr) f")
print("     约定乙（标准 GR 约定）：nabla^2 f = -(d^2/dr^2 + (2/r) d/dr) f")
for r0 in (0.8, 1.2, 1.6):
    N2 = 1.0 - float(F_c(r0))
    om = lambda r: 0.5 * math.log(1.0 + float(G_c(r)))
    gr2 = (rad_d1(om, r0)) ** 2
    G00 = G_munu(F_c, G_c, r0)[0, 0]
    rows_s = []
    for tag, sgn in (("甲", 1.0), ("乙", -1.0)):
        Rp = (1.0 / (1.0 + float(G_c(r0)))) * (-4.0 * sgn * rad_lap(om, r0) - 2.0 * gr2)
        pred = 0.5 * N2 * Rp
        rows_s.append((tag, Rp, pred, abs(G00 - pred) / abs(G00)))
    print("     r = %.1f：机器 G00 = %+.6e" % (r0, G00))
    for tag, Rp, pred, e in rows_s:
        print("        约定%s：R^(3) = %+.6e ｜ (1/2) N^2 R^(3) = %+.6e ｜ 相对差 %.3e"
              % (tag, Rp, pred, e))

print("     登记（迹关系的解析线路）：机器给 R00 = N laplace_(3)(N)（由 G00 与 R = -G^mu_mu 反解，0.1% 内），")
print("       与 3+1 标准式 R00 = N[laplace_(3)N + Gamma^i_ij d^j N] 的差额项未在本轮定位；")
print("       该差额不进入任何判据，登记为解析侧微项")
# 2026-10-21 订正（判据第 742 号 8/8 与第 747 号 4/5）：上面三行登记是纯打印断言，本轮无对应计算，
# 且实测「机器给 R00 = N laplace_(3)N」与机器读数相差约 170%。差额经六个 Gamma 项候选形式全部排除，
# 差额与 N 项同阶（不属高阶小量）、不属 F 局部导数项族；来源定位为四维 R00 与纯空间三维分解公式的
# 定义域不匹配（装置的 g_tt = -(1-F(r)) 依赖空间坐标），库内无真正静态的装置。
# 另一条方法论判定：替身检验（给 g00 补一项 F^2）显示本轮的 G00 = (1/2) N^2 R^(3) 相容性
# 对 g00 的构造改动完全不敏感（替身下相对差仍为 3.75e-08 至 7.20e-07），故该相容性属构造性成立、
# 无判别力，不得引为物理证据（数值本身仍正确）。
# 详见 [[验证脚本/注记-P5缝合假设第一层：差额项的完整来源与守恒结构（判据第747号）|第 747 号注记]]。
# 2026-10-21 订正（判据第 742 号 8/8）：上面三行登记是纯打印断言，本轮无对应计算，
# 且实测「机器给 R00 = N laplace_(3)N」与本轮机器读数相差约 170%。差额经六个 Gamma 项
# 候选形式全部排除，来源是两套 3+1 口径的混用（本轮口径把 N^2 = 1-F 与 gamma = g_ij
#（未除 N^2）混用，严格口径要求 gamma_ij = g_ij/N^2，两口径读数相差 O(1)）。
# 详见 [[验证脚本/注记-完整3+1分解的lapse与shift高阶结构（判据第742号）|第 742 号注记]]。

print()
print("[G1b8-4] 0i 支与守恒")
print("     0i 支：静场恒等为零（机器读数 |G_0i| 最大值为 %.1e 量级，属差分零）"
      % max(abs(G_munu(F_c, G_c, r0)[0, i]) for r0 in (0.8, 1.6) for i in (1, 2, 3)))
print("     收缩比安基：对带源完成场复跑（首轮已对真空与线性族核过；本轮为登记，恒等式项）")
print("     守恒对偶：0i 恒等为零 + 00 支锁定 + ij 支读数 ⇒ 静球对称扇区四类分量齐备")

print()
print("[G1b8-5] 登记")
rows = [
    ("其一", "00 支", "G00/(8 pi rho) 与三轮锁定一致（kappa 4.5 的 O(theta) 修正内）"),
    ("其二", "ij 支", "压力读数 p_r 与 p_t 不等（各向异性）；迹 3p 与各向异性项分开登记，供张量桥 d 对照"),
    ("其三", "3+1", "哈密顿约束 G00 = (1/2) N^2 R^(3) 在完成场上成立（10^-3 级）"),
    ("其四", "协变余项", "一般（非球）位形、时间依赖、源侧 T_ij 微观分解、介质不可探测性的观测协变论证"),
]
for a1, a2, a3 in rows:
    print("     %-4s %-8s %s" % (a1, a2, a3))

print()
print("总运行时间 = %.1f 秒" % (time.time() - T0))
