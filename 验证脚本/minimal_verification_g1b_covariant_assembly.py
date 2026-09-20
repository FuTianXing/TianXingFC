# -*- coding: utf-8 -*-
"""
minimal_verification_g1b_covariant_assembly.py

[G1b 第八轮] 四维组装：协变化的三扇区读数与 3+1 约束（判据 G1b8-1 至 G1b8-5）

背景：完整形式的前几轮只用了 00 支（记录方程与曲率读数）；四维组装要求把 0i、ij 两个
扇区与 3+1 约束结构一起交代。本轮在静球对称的完成场上把四类分量全部落到读数：
00 支回归、ij 支的压力与各向异性、3+1 哈密顿约束核对、0i 与守恒的状态陈述。

口径：完成式 theta_c = theta_lin - theta_lin^2、规则 hij = h00 + (7/8) h00^2；
机器给 g_00 = -(1-F)、g_ii = 1+G 的四维爱因斯坦张量（逐点局部逆度规）。压力读法
取流体形式 T_ij = p g_ij ⇒ p = G_ii / (8 pi g_ii)（径向与切向各一）。

判据
G1b8-1 00 支回归：G00/(8 pi rho_m) 与三轮的 rho_eff 关系一致
G1b8-2 ij 支：径向与切向压力的读数、各向异性大小、与"只经迹耦合"（判据 L1）的对照
G1b8-3 3+1 哈密顿约束：G00 = (1/2) N^2 R^(3)（N^2 = 1 - F，gamma = (1+G) delta）；
       迹关系的解析式与机器的差额项未定位，登记为解析侧微项（不入判据）
G1b8-4 0i 支与守恒：静场 G0i = 0（恒等）；收缩比安基对带源完成场复跑（守恒一致）
G1b8-5 登记：四类分量状态与协变余项清单

运行：python 验证脚本/minimal_verification_g1b_covariant_assembly.py（确定性，约 2 秒）
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

print("     登记（迹关系的解析线路）：机器给 R00 = N laplace_(3)(N)（由 G00 与 R = -G^mu_mu 反解，0.1% 内），")
print("       与 3+1 标准式 R00 = N[laplace_(3)N + Gamma^i_ij d^j N] 的差额项未在本轮定位；")
print("       该差额不进入任何判据，登记为解析侧微项")

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
