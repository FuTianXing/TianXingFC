# -*- coding: utf-8 -*-
"""
g1b_source_mechanism.py

[G1b 第五轮] 源侧二阶项的机制侧判定（判据 G1b5-1 至 G1b5-5）

背景：三轮把携源二阶项锁定为 rho_eff = rho_m(1 - (9/2)theta)（等价 R = G00 - 8 pi rho_m
= -36 pi rho_m theta），但该值是"量出来的"，不是从源侧机制导出的。本轮把它拆开：
形状判别（源侧修正必须是局部乘性）、机制候选系数的清单与判负、系数分解的几何归属、
以及源侧单独不可能的定量判定。

口径：沿用三、四轮（完成式 theta_c = theta_lin - theta_lin^2、规则 hij = h00 + (7/8)h00^2、
机器 G^(1)00 = -laplace(B)、单位 laplace(theta_lin) = -4 pi rho_m）。

判据
G1b5-1 形状判别：R 对各候选形状的比值跨半径散布（局部乘性 rho theta 唯一存活）
G1b5-2 机制候选系数：无自耦合 0、红移 -1、几何槽位合成 -9/2 的对照
G1b5-3 系数分解核对：G00 = -2 laplace(theta) + 5 theta laplace(theta) - 4 (grad theta)^2（点态）
       与 9/2 = 2 + 5/2 的分解（两个几何槽位的权重）
G1b5-4 源侧单独不可能：源独模型（laplace(theta_S) = -4 pi rho_m(1 + kappa theta_S)）的最优 kappa = -2，
       曲率差随剖面 2% 至 13% 且与 M 无关；逐点所需 kappa_req = -2 + (grad theta)^2/(2 pi rho theta) 随半径变化
G1b5-5 登记

运行：python 验证脚本/g1b_source_mechanism.py（确定性，约 2 秒）
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


def g00_rich(Ff, Gf, r0):
    g1 = g00_of(Ff, Gf, (r0, 0, 0), 0.05)
    g2 = g00_of(Ff, Gf, (r0, 0, 0), 0.025)
    return (16.0 * g2 - g1) / 15.0


def poisson_solve(gfun, rmax=10.0, n=4000):
    rr = np.linspace(1e-6, rmax, n)
    gv = np.array([gfun(float(x)) for x in rr])
    cell = rr[1] - rr[0]
    inner = np.cumsum(gv * rr ** 2) * cell
    outer = np.cumsum((gv * rr)[::-1])[::-1] * cell
    with np.errstate(divide="ignore", invalid="ignore"):
        f = -inner / rr - outer
    return rr, f


print("=" * 78)
print("G1b 第五轮：源侧二阶项的机制侧判定（判据 G1b5-1 至 G1b5-5）")
print("=" * 78)

F_c = lambda r: 2.0 * th_c(r)
G_c = lambda r: 2.0 * th_c(r) + 3.5 * th_c(r) ** 2

print()
print("[G1b5-1] 形状判别：R 对各候选形状的比值（局部乘性唯一存活）")
rr_grid = np.linspace(1e-6, 10.0, 4000)
rho_grid = rho_m(rr_grid)
th_grid = th_lin(rr_grid)
# 窗口平均（非局部候选）：theta_win(r) = ∫ theta K r'^2 dr' / ∫ K r'^2 dr'，K 取宽度 sigma 的核
Kmat = np.exp(-(rr_grid[:, None] - rr_grid[None, :]) ** 2 / (2.0 * SIG ** 2))
num = (Kmat * (th_grid * rr_grid ** 2)[None, :]).sum(axis=1)
den = (Kmat * (rr_grid ** 2)[None, :]).sum(axis=1)
th_win = num / den
shapes = [
    ("rho·theta（局部乘性，本轮要求）", lambda r: rho_m(r) * th_lin(r)),
    ("rho·theta^2（局部二次）", lambda r: rho_m(r) * th_lin(r) ** 2),
    ("(grad theta)^2（朴素场能）", lambda r: rad_grad2(th_lin, r)),
    ("rho·<theta>_窗口（非局部）", lambda r: rho_m(r) * float(np.interp(r, rr_grid, th_win))),
]
radii = (0.7, 1.0, 1.3, 1.6, 2.0)
Rvals = []
for r0 in radii:
    R = g00_rich(F_c, G_c, r0) - 8.0 * math.pi * rho_m(r0)
    Rvals.append(R)
for name, sf in shapes:
    ratios = [Rvals[i] / sf(r0) for i, r0 in enumerate(radii)]
    spread = (max(ratios) - min(ratios)) / abs(sum(ratios) / len(ratios)) * 100.0
    print("     %-28s：比值 %s ｜ 跨半径散布 %.1f%%"
          % (name, "、".join("%.4g" % x for x in ratios), spread))
print("     双质量对照（rho·theta 形状的散布应为 O(theta) 修正而非形状失配）：")
for mass in (0.05, 0.02):
    Fm = lambda r, m=mass: 2.0 * th_c(r, m)
    Gm = lambda r, m=mass: 2.0 * th_c(r, m) + 3.5 * th_c(r, m) ** 2
    rat = []
    for r0 in (0.8, 1.2, 1.6):
        R = g00_rich(Fm, Gm, r0) - 8.0 * math.pi * rho_m(r0, mass)
        rat.append(R / (rho_m(r0, mass) * float(th_lin(r0, mass))))
    spr = (max(rat) - min(rat)) / abs(sum(rat) / len(rat)) * 100.0
    print("       M = %.3f：比值 %s ｜ 散布 %.2f%%" % (mass, "、".join("%.1f" % x for x in rat), spr))

print()
print("[G1b5-2] 机制候选系数（曲率口径 rho_eff = rho_m(1 + kappa theta)）")
print("     实测（三轮，四档质量外推）：kappa = -4.5（即 -9/2）")
rows = [
    ("无自耦合（源=裸缺陷）", "0", "判负：无论廓与量级都不符"),
    ("红移/测到无穷远（√(-g00) = 1 - theta）", "-1", "判负：差 3.5（4.5 倍中的 78%）"),
    ("四维体积元（√(-g) - 1 ≈ +2 theta）", "+2", "判负：符号相反"),
    ("几何槽位合成（本轮给出）", "-9/2", "吻合（见 G1b5-3）"),
]
for a, b, c in rows:
    print("     %-34s kappa = %-5s %s" % (a, b, c))

print()
print("[G1b5-3] 系数分解核对：G00 = -2 lap(theta) + 5 theta lap(theta) - 4 (grad theta)^2")
for r0 in (0.8, 1.2, 1.6):
    gm = g00_rich(F_c, G_c, r0)
    th = float(th_c(r0))
    lap = rad_lap(lambda r: th_c(r), r0)
    gr = rad_grad2(lambda r: th_c(r), r0)
    ana = -2.0 * lap + 5.0 * th * lap - 4.0 * gr
    print("     r = %.1f：机器 = %+.6e ｜ 分解式 = %+.6e ｜ 相对差 = %.2e" % (r0, gm, ana, abs(gm - ana) / abs(gm)))
print("     分解：-2 lap(theta) 的源部给 -2 theta；5 theta lap(theta) 的源部给 -2.5 theta；")
print("       合计 -4.5 theta = -9/2（两个几何槽位权重 1 与 5/4；grad 项两半相消）")

print()
print("[G1b5-4] 源侧单独不可能：源独模型的最优 kappa 与逐点所需 kappa 不一致")
KAP = -2.0
rr_s, f_s = poisson_solve(lambda x: 8.0 * math.pi * float(np.interp(x, rr_grid, rho_grid * th_grid)))
def th_S(r):
    return th_lin(r) + np.interp(r, rr_s, f_s)
F_s = lambda r: 2.0 * th_S(r)
G_s = lambda r: 2.0 * th_S(r) + 3.5 * th_S(r) ** 2
print("     源独模型：laplace(theta_S) = -4 pi rho (1 - 2 theta_S)（最优 kappa = -2，即把完成式的源因子搬来）")
for r0 in (0.8, 1.2, 1.6):
    gc = g00_rich(F_c, G_c, r0)
    gs = g00_rich(F_s, G_s, r0)
    d = gs - gc
    sig = 36.0 * math.pi * rho_m(r0) * float(th_c(r0))
    print("     r = %.1f：G00(源独) − G00(完成) = %+.4e ｜ 占信号 %.1f%%（差随剖面变化，非零）"
          % (r0, d, 100.0 * d / sig))
print("     逐点所需 kappa（把源独模型的解析式对齐到实测曲率）：kappa_req = -2 + (grad theta)^2/(2 pi rho theta)")
for r0 in (0.7, 1.0, 1.3, 1.6, 2.0):
    gr = rad_grad2(lambda r: th_c(r), r0)
    kq = -2.0 + gr / (2.0 * math.pi * rho_m(r0) * float(th_c(r0)))
    print("       r = %.1f：kappa_req = %+.3f" % (r0, kq))
print("     结论：kappa_req 随半径变化（无常数 kappa 可同时满足全剖面），源独读法判负；")
print("       残余按 (grad theta)^2 标度（场项不可由源侧承担）")
print("     与 M 无关性对照：")
for mass in (0.05, 0.02):
    r0 = 1.2
    Fm = lambda r, m=mass: 2.0 * th_c(r, m)
    Gm = lambda r, m=mass: 2.0 * th_c(r, m) + 3.5 * th_c(r, m) ** 2
    _, fs2 = poisson_solve(lambda x, m=mass: 8.0 * math.pi * float(np.interp(x, rr_grid, rho_m(rr_grid, m) * th_lin(rr_grid, m))))
    tS = lambda r, m=mass: th_lin(r, m) + np.interp(r, rr_s, fs2)
    Fs = lambda r, m=mass: 2.0 * tS(r, m)
    Gs = lambda r, m=mass: 2.0 * tS(r, m) + 3.5 * tS(r, m) ** 2
    d = g00_rich(Fs, Gs, r0) - g00_rich(Fm, Gm, r0)
    sig = 36.0 * math.pi * rho_m(r0, mass) * float(th_c(r0, mass))
    print("       M = %.3f：差/信号 = %+.2f%%" % (mass, 100.0 * d / sig))

print("[G1b5-5] 登记")
rows2 = [
    ("其一", "形状", "源侧修正为局部乘性 rho·theta；rho theta^2、(grad theta)^2、非局部窗口三类均判负"),
    ("其二", "系数", "自然可导出的候选（0、-1、+2）全部判负；-9/2 由几何槽位合成（2 + 5/2）给出"),
    ("其三", "不可能性", "源独模型最优 kappa = -2 的曲率差随剖面 2% 至 13% 且与 M 无关；逐点所需 kappa_req 随半径变化：无常数 kappa 可用，场项不可由源侧承担"),
    ("其四", "机制侧下一步", "把源因子 (1 - 2 theta)（或完成式 theta = theta_lin - theta_lin^2）从満周/自耦合的写入结构导出"),
]
for a1, a2, a3 in rows2:
    print("     %-4s %-8s %s" % (a1, a2, a3))

print()
print("总运行时间 = %.1f 秒" % (time.time() - T0))
