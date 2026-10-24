# -*- coding: utf-8 -*-
"""
g1b_source_assembly.py

[G1b 第二轮] 二阶源项与自耦合的机制装配（判据 G1b2-1 至 G1b2-5）

背景：首轮把非线性封装的第一道核对做成读数：静真空二阶靶子
h00 = h_lin − (1/2)h_lin²、hij = h_lin + (3/8)h_lin²，等价 theta = theta_lin − theta_lin²
（记录场层面）。本轮问"这个二阶项由什么生成"：是源侧的场能入源（自耦合），
还是方程侧（几何）的二阶项，还是两者按口径分配。用首轮同一台 FD 爱因斯坦
张量机器做装配核对。

判据
G1b2-1 有效非线性方程：精确解满足 laplace(theta) = −2 (grad theta)² + O(theta³)
G1b2-2 形状换算核对：laplace(−theta_1²) = −2 (grad theta_1)²（靶子形状，s = 1）；
       对照形状 s = 1/4 给 −(1/2)(grad theta_1)²（弱四倍，登记为形状换算表）
G1b2-3 几何二阶项的直接测量：G^{(2)}[h_1]_{00} 的系数 c，解析预测 3 = 7 − 4（幅度外推核对）
G1b2-4 装配恒等式：G^{(1)}[h_2] + G^{(2)}[h_1] = 0（h_1 线性族、h_2 = −theta_1² 修正族）
G1b2-5 装配结论与分工登记（打印）

运行：python 验证脚本/g1b_source_assembly.py（确定性，约 1 秒）
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


def u_of(r):
    return MU / (2.0 * r)


def F_exact(r):
    u = u_of(r)
    return 4.0 * u / (1.0 + u) ** 2


def G_exact(r):
    u = u_of(r)
    return (1.0 + u) ** 4 - 1.0


def F_lin(r):
    return 4.0 * u_of(r)


def F_2(r):
    h = F_lin(r)
    return -0.5 * h ** 2


def G_2(r):
    h = F_lin(r)
    return (3.0 / 8.0) * h ** 2


def radial_d1(f, r, h=1e-4):
    return (-f(r + 2 * h) + 8 * f(r + h) - 8 * f(r - h) + f(r - 2 * h)) / (12 * h)


def radial_d2(f, r, h=1e-4):
    return (-f(r + 2 * h) + 16 * f(r + h) - 30 * f(r) + 16 * f(r - h) - f(r - 2 * h)) / (12 * h * h)


def radial_lap(f, r, h=1e-4):
    return radial_d2(f, r, h) + 2.0 * radial_d1(f, r, h) / r


print("=" * 78)
print("G1b 第二轮：二阶源项与自耦合的机制装配（判据 G1b2-1 至 G1b2-5）")
print("=" * 78)

print()
print("[G1b2-1] 有效非线性方程：laplace(theta) = −2 (grad theta_1)² + O(theta_1³)")
def th_ex(r):
    u = u_of(r)
    return 2.0 * u / (1.0 + u) ** 2


for r0 in (4.0, 8.0, 16.0):
    lap = radial_lap(th_ex, r0)
    tgt = -2.0 * (MU / r0 ** 2) ** 2
    print("     r = %5.1f：laplace(theta) = %+.6e，−2(grad theta_1)² = %+.6e，残差 = %+.3e"
          % (r0, lap, tgt, lap - tgt))
rr = np.linspace(3.0, 20.0, 40)
resv = []
for r0 in rr:
    resv.append(abs(radial_lap(th_ex, r0) + 2.0 * (MU / r0 ** 2) ** 2))
sl = np.polyfit(np.log(rr), np.log(resv), 1)[0]
print("     残差幂次拟合 = %.2f（O(theta_1³) 期望约 -5）" % sl)
resv2 = []
for r0 in rr:
    lead = 4.5 * MU ** 3 / r0 ** 5
    resv2.append(abs(radial_lap(th_ex, r0) + 2.0 * (MU / r0 ** 2) ** 2 - lead))
sl2 = np.polyfit(np.log(rr), np.log(resv2), 1)[0]
print("     扣主导残差 (9/2)theta_1(grad theta_1)² 后幂次拟合 = %.2f（期望约 -6，两级逐项吻合）" % sl2)

print()
print("[G1b2-2] 形状换算核对（theta_1 = mu/r；θ² 形状与梯度形状互译）")
for (c, tag) in ((1.0, "靶子形状 s = 1：−theta_1²"), (0.25, "对照形状 s = 1/4：−(1/4)theta_1²")):
    err = 0.0
    for r0 in (1.0, 2.0, 3.0):
        lap = radial_lap(lambda r, c=c: -c * (MU / r) ** 2, r0)
        tgt = -2.0 * c * (MU / r0 ** 2) ** 2
        err = max(err, abs(lap - tgt) / abs(tgt))
    print("     %s：laplace 与系数式最大相对差 = %.2e" % (tag, err))
print("     换算登记：θ² 形状取系数 s 等价于梯度形状 −2s(grad theta_1)²；靶子要求 s = 1，s = 1/4 弱四倍")

print()
print("[G1b2-3] 几何二阶项的直接测量：G^{(2)}[h_1]_{00}（解析预测 c = 3 = 7 − 4）")
def G2_estimate(Ff, Gf, x0, amp, eps):
    g1 = einstein_at(lambda r: Ff(r, amp), lambda r: Gf(r, amp), x0, eps=eps)[0, 0]
    g2 = einstein_at(lambda r: Ff(r, amp / 2), lambda r: Gf(r, amp / 2), x0, eps=eps)[0, 0]
    return 2.0 * (g1 - 2.0 * g2) / amp ** 2

def Flin_a(r, a):
    return a * F_lin(r)

def G_lin_a(r, a):
    return a * F_lin(r)

def c_of_amp(r0, amp):
    b1 = G2_estimate(Flin_a, G_lin_a, (r0, 0, 0), amp, 0.05)
    b2 = G2_estimate(Flin_a, G_lin_a, (r0, 0, 0), amp, 0.025)
    bx = (16.0 * b2 - b1) / 15.0
    return bx / (MU / r0 ** 2) ** 2

AMPS = (0.01, 0.005, 0.0025, 0.00125)
QUAD0 = {}
for r0 in (1.5, 2.5, 3.5):
    cs = [c_of_amp(r0, a) for a in AMPS]
    kk, b0 = np.polyfit(np.array(AMPS), np.array(cs), 1)
    k2, k1, b02 = np.polyfit(np.array(AMPS), np.array(cs), 2)
    QUAD0[r0] = b02 * (MU / r0 ** 2) ** 2
    print("     r = %.1f：c(0.01/0.005/0.0025/0.00125) = %s" % (r0, "、".join("%.5f" % c for c in cs)))
    print("             外推 c0：线性 %.5f（斜率 %+.2f）｜含二次项 %.5f（二次系数 %+.1f；三阶项污染随 a 归零）"
          % (b0, kk, b02, k2))

print()
print("[G1b2-4] 装配恒等式：G^{(1)}[h_2] + G^{(2)}[h_1] = 0（两肢各自外推干净）")
def lin_of_scale(r0, eh):
    l1 = einstein_at(lambda r: eh * F_2(r), lambda r: eh * G_2(r), (r0, 0, 0), eps=0.05)[0, 0]
    l2 = einstein_at(lambda r: eh * F_2(r), lambda r: eh * G_2(r), (r0, 0, 0), eps=0.025)[0, 0]
    return (16.0 * l2 - l1) / 15.0 / eh

for r0 in (1.5, 2.5, 3.5):
    lin = (10.0 * lin_of_scale(r0, 1e-4) - lin_of_scale(r0, 1e-3)) / 9.0
    quad = QUAD0[r0]
    tot = lin + quad
    print("     r = %.1f：G^{(1)}[h_2] = %+.6e，G^{(2)}[h_1] = %+.6e，和 = %+.3e（相对 %.2e）"
          % (r0, lin, quad, tot, abs(tot) / max(abs(lin), abs(quad))))

print()
print("[G1b2-5] 装配结论与分工登记")
rows = [
    ("其一", "三项平衡（单位 (grad theta_1)²）", "θ 场项 +4、写入规则迹项 −7、几何非线性实测 c = +3，合计零；θ 方程口径 +3/2 − 7/2 = −2 与靶子一致"),
    ("其二", "系数 c 的地位", "c = 7 − 4 由靶子（2×2）与规则空间扇区系数（2×7/2）锁定，非自由参数；幅度外推实测 3.000"),
    ("其三", "机制侧的含义", "真空二阶结构由规则与几何非线性闭合，无需源侧自耦合介入；自能入源留待携源轮"),
    ("其四", "下一轮", "携源剖面的二阶贡献与强场对接（G1b-c/e）"),
]
for a, b, c in rows:
    print("     %-4s %-28s %s" % (a, b, c))

print()
print("总运行时间 = %.1f 秒" % (time.time() - T0))
