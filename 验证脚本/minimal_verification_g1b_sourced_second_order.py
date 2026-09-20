# -*- coding: utf-8 -*-
"""
minimal_verification_g1b_sourced_second_order.py

[G1b 第三轮] 携源剖面的二阶源项与残差消去（判据 G1b3-1 至 G1b3-5）

背景：首轮定下真空二阶靶子，第二轮把真空二阶结构装配为三项平衡（θ 场项 +4、
写入规则迹项 −7、几何非线性 +3，单位 (grad theta_1)^2）并判定真空无需源侧修正；
源侧自耦合的位置只能在携源剖面上判定。本轮取软化高斯源（总质量 M、宽度 sigma），
用同一台 FD 爱因斯坦张量机器做五项核对。

口径：线性层 h00 = 2 theta、hij = h00；记录方程 laplace(theta) = −4 pi rho（本口径
下 G00 = 8 pi rho 在线性层成立，机器约定 G^(1)00 = −laplace(B)，B 为空间扇区系数）；
完成式 theta_c = theta_lin − theta_lin^2，规则 hij = h00 + (7/8) h00^2。

判据
G1b3-1 携源线性层：theta_lin = M erf(r/(sigma sqrt2))/r 满足 laplace = −4 pi rho_m
G1b3-2 二阶源项（θ 语言）：laplace(theta_c) = −4 pi rho_m − 2 (grad theta)^2，
       等价附加源 rho_2 = (grad theta)^2/(2 pi)；与朴素自能归一（方程口径系数 1/2）差 4 倍
G1b3-3 二次型机器核对：G00[F = G = A] = −laplace(A) + 3 A laplace(A) + (3/4)(grad A)^2
G1b3-4 携源残差消去：R = G00 − 8 pi rho_m 对 −36 pi rho_m theta（即 rho_eff = rho_m(1 − 9 theta/2)）
G1b3-5 全局形式：δM = ∫R/8 pi d3x 三路线核对（机器积分、rho theta 解析、场能形式 δM = −(9/4 pi) E_f）

运行：python 验证脚本/minimal_verification_g1b_sourced_second_order.py（确定性，约 1 秒）
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


def rad_d1(f, r, h=1e-4):
    return (-f(r + 2 * h) + 8 * f(r + h) - 8 * f(r - h) + f(r - 2 * h)) / (12 * h)


def rad_d2(f, r, h=1e-4):
    return (-f(r + 2 * h) + 16 * f(r + h) - 30 * f(r) + 16 * f(r - h) - f(r - 2 * h)) / (12 * h * h)


def rad_lap(f, r, h=1e-4):
    return rad_d2(f, r, h) + 2.0 * rad_d1(f, r, h) / r


def rad_grad2(f, r, h=1e-4):
    g = rad_d1(f, r, h)
    return g * g


def g00_of(Ff, Gf, x0, eps):
    return einstein_at(Ff, Gf, x0, eps=eps)[0, 0]


print("=" * 78)
print("G1b 第三轮：携源剖面的二阶源项与残差消去（判据 G1b3-1 至 G1b3-5）")
print("=" * 78)
print("源：高斯团块 M = %.3f，sigma = %.1f（弱场 theta ~ %.3f）" % (MASS, SIG, MASS / SIG))

print()
print("[G1b3-1] 携源线性层：laplace(theta_lin) = −4 pi rho_m")
rr = np.linspace(0.6, 2.2, 25)
resv = []
for r0 in (0.8, 1.2, 1.6, 2.0):
    lap = rad_lap(th_lin, r0)
    tgt = -4.0 * math.pi * rho_m(r0)
    print("     r = %.1f：laplace = %+.6e，−4 pi rho_m = %+.6e，相对差 = %.2e"
          % (r0, lap, tgt, abs(lap - tgt) / abs(tgt)))
for r0 in rr:
    resv.append(abs(rad_lap(th_lin, r0) + 4.0 * math.pi * rho_m(r0)))
print("     最大绝对残差 = %.2e（网格步长 h = 1e-4 的差分精度）" % max(resv))
print("     远场：theta_lin(4) = %.4e 对 M/r = %.4e（相对差 %.2e）"
      % (th_lin(4.0), MASS / 4.0, abs(th_lin(4.0) - MASS / 4.0) / (MASS / 4.0)))

print()
print("[G1b3-2] 二阶源项（theta_c = theta_lin − theta_lin^2）")
print("     带源时 laplace(theta_lin) = −4 pi rho_m 不消失，多出 2 theta laplace(theta) 项，")
print("     故清式为 laplace(theta_c) = −4 pi rho_m (1 − 2 theta) − 2 (grad theta)^2 + O(theta^3)")
for r0 in (0.8, 1.2, 1.6, 2.0):
    lap = rad_lap(th_c, r0)
    tgt = -4.0 * math.pi * rho_m(r0) * (1.0 - 2.0 * th_lin(r0)) - 2.0 * rad_grad2(th_lin, r0)
    print("     r = %.1f：laplace(theta_c) = %+.6e，清式 = %+.6e，相对差 = %.2e"
          % (r0, lap, tgt, abs(lap - tgt) / abs(tgt)))
print("     两面登记：其一是源的重整化因子 (1 − 2 theta)；其二是附加场项 (grad theta)^2/(2 pi)：")
print("       r = 1.0 处场项 %.4e 对 rho_m = %.4e（占 %.1f%%）"
      % (rad_grad2(th_lin, 1.0) / (2.0 * math.pi), rho_m(1.0),
         100.0 * rad_grad2(th_lin, 1.0) / (2.0 * math.pi) / rho_m(1.0)))
print("     对照：朴素自能归一（自能源 (grad theta)^2/2）给方程口径系数 1/2，本式为 2，差 4 倍；")
print("       且本源项不含 (1 − 2 theta) 型重整化，两项形状都不同")

print()
print("[G1b3-3] 二次型机器核对：G00[F = G = A] = −laplace(A) + 3 A laplace(A) + (3/4)(grad A)^2")
def quad_of_A(r0, lam):
    A0 = lambda r: 2.0 * th_lin(r)
    g = g00_of(lambda r: lam * A0(r), lambda r: lam * A0(r), (r0, 0, 0), 0.02)
    return (g + lam * rad_lap(A0, r0)) / lam ** 2

radii = [1.0, 1.4, 1.8, 2.2, 2.6]
cols = []
for r0 in radii:
    q1 = quad_of_A(r0, 0.01)
    q2 = quad_of_A(r0, 0.005)
    q = 2.0 * q2 - q1
    A0, lapA0, grad2A0 = 2.0 * th_lin(r0), rad_lap(lambda r: 2.0 * th_lin(r), r0), rad_grad2(lambda r: 2.0 * th_lin(r), r0)
    cols.append((A0 * lapA0, grad2A0, q))
    print("     r = %.1f：q = %+.6e ｜ A lapA = %+.6e，grad^2 A = %+.6e" % (r0, q, A0 * lapA0, grad2A0))
Mtx = np.array([[c[0], c[1]] for c in cols])
vec = np.array([c[2] for c in cols])
sol, res, rank, sv = np.linalg.lstsq(Mtx, vec, rcond=None)
print("     最小二乘拟合：A lapA 系数 = %.4f（预测 3），(grad A)^2 系数 = %.4f（预测 0.75）" % (sol[0], sol[1]))

print()
print("[G1b3-4] 携源残差消去：R = G00 − 8 pi rho_m 对 −36 pi rho_m theta（rho_eff = rho_m(1 − 9 theta/2)）")
F_c = lambda r: 2.0 * th_c(r)
G_c = lambda r: 2.0 * th_c(r) + 3.5 * th_c(r) ** 2
for mass in (0.05, 0.02, 0.01, 0.005):
    Fm = lambda r, m=mass: 2.0 * th_c(r, m)
    Gm = lambda r, m=mass: 2.0 * th_c(r, m) + 3.5 * th_c(r, m) ** 2
    for r0 in (0.8, 1.2, 1.6):
        g1 = g00_of(Fm, Gm, (r0, 0, 0), 0.05)
        g2 = g00_of(Fm, Gm, (r0, 0, 0), 0.025)
        g00 = (16.0 * g2 - g1) / 15.0
        rm = rho_m(r0, mass)
        th = th_lin(r0, mass)
        R = g00 - 8.0 * math.pi * rm
        pred = -36.0 * math.pi * rm * th
        naive = 4.0 * math.pi * rad_grad2(lambda r: th_lin(r, mass), r0)
        print("     M = %.3f r = %.1f：G00 = %+.6e，R = %+.6e ｜ 预测 %+.6e ｜ 纯物质 0 ｜ 朴素场能 %+.6e"
              % (mass, r0, g00, R, pred, naive))
        print("              测得 −R/(8 pi rho_m theta) = %+.4f（预测 4.5）｜ R/(8 pi rho_m) = %+.5f"
              % (-R / (8.0 * math.pi * rm * th), R / (8.0 * math.pi * rm)))

print()
print("[G1b3-5] 全局形式：δM = ∫R/8 pi d3x 的三条路线（有效源修正的体积分）")
mass_g = 0.02
Fg = lambda r: 2.0 * th_c(r, mass_g)
Gg = lambda r: 2.0 * th_c(r, mass_g) + 3.5 * th_c(r, mass_g) ** 2
grid = np.linspace(0.1, 6.0, 61)
dM_meas = 0.0
dM_rho = 0.0
E_f = 0.0
for i in range(len(grid) - 1):
    ra, rb = grid[i], grid[i + 1]
    w = rb - ra
    ga = (16.0 * g00_of(Fg, Gg, (ra, 0, 0), 0.025) - g00_of(Fg, Gg, (ra, 0, 0), 0.05)) / 15.0
    gb = (16.0 * g00_of(Fg, Gg, (rb, 0, 0), 0.025) - g00_of(Fg, Gg, (rb, 0, 0), 0.05)) / 15.0
    Ra = ga - 8.0 * math.pi * rho_m(ra, mass_g)
    Rb = gb - 8.0 * math.pi * rho_m(rb, mass_g)
    dM_meas += w * 0.5 * (ra * ra * Ra + rb * rb * Rb) / 2.0
    dM_rho += w * 0.5 * (ra * ra * rho_m(ra, mass_g) * th_lin(ra, mass_g)
                         + rb * rb * rho_m(rb, mass_g) * th_lin(rb, mass_g))
    E_f += w * 0.5 * (ra * ra * rad_grad2(lambda r: th_lin(r, mass_g), ra)
                      + rb * rb * rad_grad2(lambda r: th_lin(r, mass_g), rb)) / 2.0
dM_rho *= -4.5 * 4.0 * math.pi
E_f = 4.0 * math.pi * E_f  # 累积量为 (1/2)∫r^2 (gradθ)^2 dr，乘 4 pi 得 ∫(gradθ)^2/2 d3x
E_f_tail = 2.0 * math.pi * mass_g ** 2 / grid[-1]
print("     路线一（机器采样积分，R = G00 − 8 pi rho_m）：δM = %+.6e" % dM_meas)
print("     路线二（解析 −(9/2)∫rho theta d3x）：δM = %+.6e（与路线一差 %.2f%%）"
      % (dM_rho, 100.0 * abs(dM_rho - dM_meas) / abs(dM_rho)))
print("     恒等式核对：∫rho theta d3x = %+.6e ｜ (1/4 pi)∫(grad theta)^2 d3x = %+.6e（含解析尾项 M^2/R）"
      % (4.0 * math.pi * dM_rho / (-4.5 * 4.0 * math.pi), 2.0 * (E_f + E_f_tail) / (4.0 * math.pi)))
print("     路线三（场能形式）：δM = −(9/4 pi) E_f = %+.6e，E_f = %+.6e（含尾项），比值 δM/E_f = %.4f（预测 %.4f）"
      % (-(9.0 / (4.0 * math.pi)) * (E_f + E_f_tail), E_f + E_f_tail,
         dM_meas / (E_f + E_f_tail), -9.0 / (4.0 * math.pi)))
print("     （外区点态登记：r = 5 sigma、M = 0.02 处 Delta/(grad theta)^2 = %.3f，朝 −4 收敛）"
      % ((lambda r0: (lambda g: (g + 2.0 * rad_lap(lambda r: th_c(r, 0.02), r0)) / rad_grad2(lambda r: th_c(r, 0.02), r0))(
          (16.0 * g00_of(lambda r: 2.0 * th_c(r, 0.02), lambda r: 2.0 * th_c(r, 0.02) + 3.5 * th_c(r, 0.02) ** 2, (5.0, 0, 0), 0.025)
           - g00_of(lambda r: 2.0 * th_c(r, 0.02), lambda r: 2.0 * th_c(r, 0.02) + 3.5 * th_c(r, 0.02) ** 2, (5.0, 0, 0), 0.05)) / 15.0))(5.0)))

print()
print("总运行时间 = %.1f 秒" % (time.time() - T0))
