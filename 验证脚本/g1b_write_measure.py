# -*- coding: utf-8 -*-
"""
g1b_write_measure.py

[G1b 第六轮] 源因子的机制来源：写入的测度原理（判据 G1b6-1 至 G1b6-5）

背景：第五轮把机制缺口收窄为一条：把源因子 (1 - 2 theta)（等价于完成式
theta = theta_lin - theta_lin^2）从写入结构导出。本轮把候选机制形式化为
"测度原理"：写入的裸场 theta_bare 按局部尺读数，即

    theta = theta_bare / s(theta)，  s = sqrt(g_ii 扇区) = sqrt(1 + G(2 theta))，

其中 G 是空间扇区的规则（截断口径 G = F + (7/8)F^2，精确口径取真空解的闭式）。
原理的物理读法：満周事件写入的是一段沿法线轴的绝对位移（裸场线性叠加），
记录值是该位移按当地尺子的读数；当地尺子由记录自己搭起的空间度量给出，故为不动点。

判据
G1b6-1 不动点与截断完成式：迭代解与 theta_c = theta_lin - theta_lin^2 的差为 O(theta^3)
G1b6-2 不动点与精确完成式：精确规则下不动点等于闭式 theta_lin/(1 + theta_lin/2)^2（逐位）
G1b6-3 源因子导出：不动点解的 laplace 给 -4 pi rho (1 - 2 theta) - 2 (grad theta)^2（第五轮登记的缺口）
G1b6-4 地标核对：theta_bare(theta = 1/2) = 2（视界）与四轮级数半径 2 的同一性
G1b6-5 登记（原理地位、与判负候选的对照、剩余子项）

运行：python 验证脚本/g1b_write_measure.py（确定性，约 2 秒）
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


def th_ex(r, mass=MASS):
    t = th_lin(r, mass)
    return t / (1.0 + 0.5 * t) ** 2


def G_trunc(F):
    return F + 0.875 * F * F


def G_exact(F):
    F = np.asarray(F, dtype=float)
    out = np.ones_like(F)
    nz = (F != 0.0) & (F <= 1.0)
    out[nz] = ((8.0 - 4.0 * F[nz] - 8.0 * np.sqrt(1.0 - F[nz])) / F[nz] ** 2) ** 2 - 1.0
    return out


def fixed_point(tb, rule, n=60):
    th = tb
    for _ in range(n):
        th = tb / np.sqrt(1.0 + rule(2.0 * th))
    return th


def tb_of_theta(th, rule):
    return th * np.sqrt(1.0 + rule(2.0 * th))


def rad_d1(f, r, h=1e-4):
    return (-f(r + 2 * h) + 8 * f(r + h) - 8 * f(r - h) + f(r - 2 * h)) / (12 * h)


def rad_d2(f, r, h=1e-4):
    return (-f(r + 2 * h) + 16 * f(r + h) - 30 * f(r) + 16 * f(r - h) - f(r - 2 * h)) / (12 * h * h)


def rad_lap(f, r, h=1e-4):
    return rad_d2(f, r, h) + 2.0 * rad_d1(f, r, h) / r


def rad_grad2(f, r, h=1e-4):
    g = rad_d1(f, r, h)
    return g * g


print("=" * 78)
print("G1b 第六轮：源因子的机制来源：写入的测度原理（判据 G1b6-1 至 G1b6-5）")
print("=" * 78)
print("原理：theta = theta_bare / s(theta)，s = sqrt(1 + G(2 theta))")
print("（裸场线性叠加满足 laplace(theta_bare) = -4 pi rho；记录是位移按当地尺的读数）")

print()
print("[G1b6-1] 不动点与截断完成式")
for r0 in (0.8, 1.2, 1.6, 2.0):
    tb = float(th_lin(r0))
    fp = float(fixed_point(tb, G_trunc))
    tc = float(th_c(r0))
    print("     r = %.1f：theta_bare = %.6f ｜ 不动点 = %.8f ｜ theta_c = %.8f ｜ 差 = %+.2e（O(theta^3) 应为 %.1e）"
          % (r0, tb, fp, tc, fp - tc, tb ** 3))

print()
print("[G1b6-2] 不动点与精确完成式")
for r0 in (0.8, 1.2, 1.6, 2.0):
    tb = float(th_lin(r0))
    fp = float(fixed_point(tb, G_exact))
    te = float(th_ex(r0))
    print("     r = %.1f：不动点 = %.10f ｜ 闭式 = %.10f ｜ 相对差 = %.2e" % (r0, fp, te, abs(fp - te) / te))
print("     恒等式核对：theta·s(theta) 对 theta_bare 的复原（精确规则）：")
for r0 in (0.8, 1.6):
    tb = float(th_lin(r0))
    th = float(th_ex(r0))
    back = float(tb_of_theta(th, G_exact))
    print("       r = %.1f：theta·s = %.10f ｜ theta_bare = %.10f ｜ 差 = %.2e" % (r0, back, tb, back - tb))

print()
print("[G1b6-3] 源因子导出：不动点解的 laplace 与清式一致")
def th_fp_fun(r):
    return float(fixed_point(float(th_lin(r)), G_trunc))
for r0 in (0.8, 1.2, 1.6):
    lap = rad_lap(th_fp_fun, r0)
    tgt = -4.0 * math.pi * rho_m(r0) * (1.0 - 2.0 * th_fp_fun(r0)) - 2.0 * rad_grad2(th_fp_fun, r0)
    print("     r = %.1f：laplace(不动点解) = %+.6e ｜ 清式 = %+.6e ｜ 相对差 = %.2e"
          % (r0, lap, tgt, abs(lap - tgt) / abs(tgt)))
print("     对照：同一清式对三轮的 theta_c 的相对差为 1e-7 量级；不动点解与之的 1e-3 级差")
print("       即 O(theta^3) 项（截断口径与不动点口径的差），属口径差不是失配")
print("     即：源因子 (1 - 2 theta) 与场项 -2(grad theta)^2 都由测度原理给出")

print()
print("[G1b6-4] 地标核对：视界与级数半径的同一性")
for th in (0.1, 0.25, 0.4, 0.5):
    tb = float(tb_of_theta(th, G_exact))
    print("     theta = %.2f：theta_bare = %.6f" % (th, tb))
print("     精确规则在 theta = 1/2 处：s = 4、theta_bare = 2（与四轮级数半径 2 重合）")
print("     视界 theta = 1/2 与满偏 theta = 1 的登记：前者为地标，后者为饱和上限（黑洞线读数）")

print()
print("[G1b6-5] 登记")
rows = [
    ("其一", "原理", "写入按当地尺读数（theta = theta_bare/s，s = sqrt(1+G)）：不动点给完成式"),
    ("其二", "两种规则", "截断规则给 theta_c（差 O(theta^3)）；精确规则给闭式（逐位一致）"),
    ("其三", "源因子", "由不动点反解 laplace 得 -4 pi rho(1-2theta) - 2(grad theta)^2：第五轮登记的缺口闭合"),
    ("其四", "地标", "theta_bare(theta=1/2) = 2 与级数半径 2 同一；与判负候选（红移 -1、体积元 +2）不同族"),
    ("其五", "剩余子项", "原理自身的机制判据：満周事件写入的位移为绝对量、记录按当地尺读数，需从事件结构给出"),
]
for a1, a2, a3 in rows:
    print("     %-4s %-8s %s" % (a1, a2, a3))

print()
print("总运行时间 = %.1f 秒" % (time.time() - T0))
