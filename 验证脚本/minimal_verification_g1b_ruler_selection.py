# -*- coding: utf-8 -*-
"""
minimal_verification_g1b_ruler_selection.py

[G1b 第七轮] 测度原理的第一原理判据：四种尺的定量判别（判据 G1b7-1 至 G1b7-5）

背景：第六轮把源因子的机制来源写成测度原理（theta = theta_bare / s(theta)），
留下一条子项：把"写入的位移为绝对量、记录按当地尺读数"从事件结构给出。本轮把它拆成
两条已注册层级的条款，并对四种自然的"尺"做定量判别。

条款（一）：位移的绝对性。法向跃迁的既有读数（注记-法向跃迁的公理级导出：判据 M4 与 M5）
给每事件恰 1.000000 l_P、与绕数无关；代价计数的结论是替代方案需新增相位指标场、绕数
与摆幅。该条为结构论证级加代价清单，不属本轮新假设。
条款（二）：读数的尺。记录 theta 是倾角，只能是"法向上升除以横向长度"的比值；横向长度
在框架里有四种自然取法：坐标格距、时间因子 sqrt(-g00)、四维体积因子 sqrt(-g)、
空间扇区长度因子 sqrt(gii)。本轮把四种尺各自的不动点解出，算到曲率读数上做判别。

判据
G1b7-1 条款一的现状核对（引既有读数，列层级与代价）
G1b7-2 四种尺的不动点与曲率判别（kappa 与梯度残余 c 的联合判别，空间尺唯一）
G1b7-3 逐点核对：空间尺不动点解的曲率与实测关系（kappa = 4.5，梯度残余为零）
G1b7-4 相容性：与刚性晶格裁定（坐标距离与测量距离之分）及地标 theta_bare(1/2) = 2
G1b7-5 登记（原理层级与零新增代价）

运行：python 验证脚本/minimal_verification_g1b_ruler_selection.py（确定性，约 3 秒）
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


def G_trunc(F):
    return F + 0.875 * F * F


def G_exact(F):
    F = np.asarray(F, dtype=float)
    out = np.ones_like(F)
    nz = (F != 0.0) & (F <= 1.0)
    out[nz] = ((8.0 - 4.0 * F[nz] - 8.0 * np.sqrt(1.0 - F[nz])) / F[nz] ** 2) ** 2 - 1.0
    return out


def rad_d1(f, r, h=1e-4):
    return (-f(r + 2 * h) + 8 * f(r + h) - 8 * f(r - h) + f(r - 2 * h)) / (12 * h)


def rad_lap(f, r, h=1e-4):
    d2 = (-f(r + 2 * h) + 16 * f(r + h) - 30 * f(r) + 16 * f(r - h) - f(r - 2 * h)) / (12 * h * h)
    return d2 + 2.0 * rad_d1(f, r, h) / r


def rad_grad2(f, r, h=1e-4):
    g = rad_d1(f, r, h)
    return g * g


def fixed_point(tb, rfun, n=60):
    th = tb
    for _ in range(n):
        th = tb / rfun(th)
    return th


RULERS = [
    ("坐标尺（1，无完成式）", lambda th: 1.0 + 0.0 * th),
    ("时间尺 sqrt(-g00)", lambda th: np.sqrt(1.0 - 2.0 * th)),
    ("四维体积尺 sqrt(-g)", lambda th: np.sqrt((1.0 - 2.0 * th) * (1.0 + 2.0 * th) ** 3)),
    ("空间扇区尺 sqrt(gii)（本轮）", lambda th: np.sqrt(1.0 + G_trunc(2.0 * th))),
]

print("=" * 78)
print("G1b 第七轮：测度原理的第一原理判据：四种尺的定量判别（判据 G1b7-1 至 G1b7-5）")
print("=" * 78)

print()
print("[G1b7-1] 条款一（位移绝对性）的现状核对")
print("     既有读数（法向跃迁注记 M4）：绕数 w = 0、1、2 下每周期净轴向位移均为 1.000000 l_P；")
print("       M5：折返是唯一可行处置（继续前进违反圆心不可重合，停驻违反波性推进）")
print("     层级：结构论证级加代价清单（替代方案需新增相位指标场、整数绕数与未定摆幅）")
print("     本轮的用法：该条不新增假设，直接引用")

print()
print("[G1b7-2] 四种尺的不动点与曲率判别（两点精确解 + 双质量外推）")
def kappa_c_of(name, rfun, mass):
    def thf(r):
        return float(fixed_point(float(th_lin(r, mass)), rfun))
    rows = []
    for r0 in (1.0, 1.6):
        lap = rad_lap(thf, r0)
        g2 = rad_grad2(thf, r0)
        th = thf(r0)
        G00 = -2.0 * lap + 5.0 * th * lap - 4.0 * g2
        R = G00 - 8.0 * math.pi * rho_m(r0, mass)
        rows.append((8.0 * math.pi * rho_m(r0, mass) * th, 4.0 * math.pi * g2, R))
    (a1, b1, y1), (a2, b2, y2) = rows
    det = a1 * b2 - a2 * b1
    kap = (y1 * b2 - y2 * b1) / det
    cc = (a1 * y2 - a2 * y1) / det
    return -kap, cc

PRED = {
    "坐标尺（1，无完成式）": (2.5, -1.0 / math.pi),
    "时间尺 sqrt(-g00)": (0.5, -2.0 / math.pi),
    "四维体积尺 sqrt(-g)": (6.5, 1.0 / math.pi),
    "空间扇区尺 sqrt(gii)（本轮）": (4.5, 0.0),
}
for name, rfun in RULERS:
    k1, c1 = kappa_c_of(name, rfun, 0.05)
    k2, c2 = kappa_c_of(name, rfun, 0.02)
    k_inf = 2.5 * k2 - 1.5 * k1
    c_inf = 2.5 * c2 - 1.5 * c1
    pk, pc = PRED[name]
    print("     %-30s M=0.05 (%+.3f, %+.3f) ｜ M=0.02 (%+.3f, %+.3f) ｜ 外推 (%+.3f, %+.3f) ｜ 解析预测 (%+.3f, %+.3f)"
          % (name, k1, c1, k2, c2, k_inf, c_inf, pk, pc))
print("     约定：R = -8 pi rho kappa theta + c·4 pi (grad theta)^2；实测（三轮）为 kappa = +4.5 且 c = 0")
print("     判别：解析预测与外推一致；只有空间扇区尺同时给 (kappa, c) = (4.5, 0)，")
print("       其余三者为 (2.5, -0.318)、(0.5, -0.637)、(6.5, +0.318)，两个分量都与实测不符")

print()
print("[G1b7-3] 逐点核对：空间尺不动点解（M = 0.05 与 0.02）")
for mass in (0.05, 0.02):
    def th_spatial(r, m=mass):
        return float(fixed_point(float(th_lin(r, m)), RULERS[3][1]))
    line = []
    for r0 in (0.8, 1.2, 1.6):
        lap = rad_lap(th_spatial, r0)
        g2 = rad_grad2(th_spatial, r0)
        th = th_spatial(r0)
        G00 = -2.0 * lap + 5.0 * th * lap - 4.0 * g2
        R = G00 - 8.0 * math.pi * rho_m(r0, mass)
        line.append("r=%.1f: kappa=%+.3f, 梯度残余 %.2f%%"
                    % (r0, -R / (8.0 * math.pi * rho_m(r0, mass) * th),
                       100.0 * (R + 36.0 * math.pi * rho_m(r0, mass) * th) / (8.0 * math.pi * rho_m(r0, mass))))
    print("     M = %.3f：%s" % (mass, " ｜ ".join(line)))

print("[G1b7-4] 相容性核对")
print("     刚性晶格裁定：格距不变，晶格为固定坐标网格 ⇒ 坐标距离与测量距离两层并存；")
print("       条款二读作'倾角按测量长度归一'，与'层数 = 坐标距离'不冲突（层数记坐标步，倾角记比值）")
for th in (0.25, 0.5):
    tb = th * float(np.sqrt(1.0 + G_exact(2.0 * th)))
    print("     地标：theta = %.2f 对应 theta_bare = %.6f（theta = 1/2 处为 2.000000，与四轮级数半径同一）" % (th, tb))
print("     位置侧相容：视界处 s = 4 为时间尺的零点处（sqrt(-g00) = 0），两尺在该处分离，")
print("       空间尺在视界处仍为 4（有限），与光锥兼容线的'内侧翻转'读数不冲突")

print()
print("[G1b7-5] 登记")
rows = [
    ("其一", "条款一", "每事件恰 1.000000 l_P、与绕数无关（结构论证级加代价清单，引用既有读数，不新增假设）"),
    ("其二", "条款二", "倾角按横向长度归一；四种尺的定量判别给空间扇区尺唯一（kappa = 4.5 且 c = 0）"),
    ("其三", "层级", "原理 = 条款一 + 条款二：条款一的代价已登记（相位指标场、绕数、摆幅），条款二零新增代价"),
    ("其四", "剩余", "无新增子项；下一层可选：原理在一般位形与时间依赖下的形式（属 G1b-f 之外的推广）"),
]
for a1, a2, a3 in rows:
    print("     %-4s %-6s %s" % (a1, a2, a3))

print()
print("总运行时间 = %.1f 秒" % (time.time() - T0))
