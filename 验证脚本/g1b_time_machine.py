# -*- coding: utf-8 -*-
"""
g1b_time_machine.py

[G1b 第十轮] 含时器械：4D 差分爱因斯坦张量与标量扇区的真空条件（判据 G1b10-1 至 G1b10-5）

背景：前九轮的器械只含空间导数（静态补丁）。含时结构的最小器械是把补丁扩到含时间方向，
并在线性层与真空条件上标定。本轮给出：器械标定（平坦零、静态回归）、线性系数测量
（h00 = A 与 hij = B 两种抽动的 00 支系数）、标量 ansatz（h00 = hij = 2 theta）在真空条件
扫描下是否有非平凡传播解。

口径：4D 补丁 9^4、四阶中心差分、逐点逆度规（对角 ansatz g00 = -(1-F)、gii = 1+G）。

判据
G1b10-1 器械标定：平坦度规零残差；静态完成场在 4D 机上与静态机读数一致
G1b10-2 线性系数：G00 对 A 抽动的 d2/dt2 系数 c_A 与对 B 抽动的 laplace 系数 c_B
G1b10-3 标量 ansatz 真空条件扫描：theta = a cos(kx - omega t)，找是否存在非平凡 (omega, k)
G1b10-4 与既有读数对照：记录场波动方程（c_s = c，张量桥 b / GV4）与约束扇区归属
G1b10-5 登记

运行：python 验证脚本/g1b_time_machine.py（确定性，约 30 秒）
"""
import math
import time
import numpy as np

T0 = time.time()
HALF = 4
C = HALF
NP_ = 2 * HALF + 1


def g_patch4(Ff, Gf, x0, eps):
    k = np.arange(-HALF, HALF + 1) * eps
    Tt = x0[0] + k[:, None, None, None]
    X = x0[1] + k[None, :, None, None]
    Y = x0[2] + k[None, None, :, None]
    Z = x0[3] + k[None, None, None, :]
    Tb, Xb, Yb, Zb = np.broadcast_arrays(Tt, X, Y, Z)
    g = np.zeros((4, 4, NP_, NP_, NP_, NP_))
    g[0, 0] = -(1.0 - Ff(Tb, Xb, Yb, Zb))
    for i in (1, 2, 3):
        g[i, i] = 1.0 + Gf(Tb, Xb, Yb, Zb)
    return g


def d1_keep(a, axis, eps):
    b = np.moveaxis(a, axis, -1)
    out = np.full_like(b, np.nan)
    out[..., 2:-2] = (b[..., :-4] - 8 * b[..., 1:-3] + 8 * b[..., 3:-1] - b[..., 4:]) / (12 * eps)
    return np.moveaxis(out, -1, axis)


def einstein_at4(Ff, Gf, x0, eps):
    g = g_patch4(Ff, Gf, x0, eps)
    dgf = [d1_keep(g, 2 + i, eps) for i in range(4)]
    gg = g[:, :, C, C, C, C]
    ginv = np.zeros((4, 4))
    for i in range(4):
        ginv[i, i] = 1.0 / gg[i, i]
    ginvf = np.zeros_like(g)
    for i in range(4):
        ginvf[i, i] = 1.0 / g[i, i]
    Gam = np.zeros((4, 4, 4, NP_, NP_, NP_, NP_))
    for lam in range(4):
        for mu in range(4):
            for nu in range(mu, 4):
                acc = np.zeros((NP_, NP_, NP_, NP_))
                for sig in range(4):
                    acc = acc + ginvf[lam, sig] * 0.5 * (
                        dgf[mu][sig, nu] + dgf[nu][sig, mu] - dgf[sig][mu, nu])
                Gam[lam, mu, nu] = acc
                Gam[lam, nu, mu] = acc
    G0 = Gam[:, :, :, C, C, C, C]
    dG = np.zeros((4, 4, 4, 4))
    for k in range(4):
        for lam in range(4):
            for mu in range(4):
                for nu in range(4):
                    dG[lam, mu, nu, k] = d1_keep(Gam[lam, mu, nu], k, eps)[C, C, C, C]
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


print("=" * 78)
print("G1b 第十轮：含时器械（判据 G1b10-1 至 G1b10-5）")
print("=" * 78)

MASS = 0.05
SIG = 1.0


def th_lin_s(r):
    r = np.asarray(r, dtype=float)
    x = r / (SIG * math.sqrt(2.0))
    out = np.where(x < 1e-8, MASS * math.sqrt(2.0 / math.pi) / SIG, MASS * np.vectorize(math.erf)(x) / np.maximum(r, 1e-30))
    return out


def th_c_s(r):
    t = th_lin_s(r)
    return t - t * t


def Fs(T, X, Y, Z):
    return 2.0 * th_c_s(np.sqrt(X ** 2 + Y ** 2 + Z ** 2))


def Gs(T, X, Y, Z):
    t = th_c_s(np.sqrt(X ** 2 + Y ** 2 + Z ** 2))
    return 2.0 * t + 3.5 * t ** 2


print()
print("[G1b10-1] 器械标定")
z00 = einstein_at4(lambda T, X, Y, Z: 0.0, lambda T, X, Y, Z: 0.0, (0.0, 1.10, 0.3, -0.2), 0.05)
print("     平坦度规：|G_00| = %.2e（应为机器零）" % abs(z00[0, 0]))
x0s = (0.0, 1.2, 0.0, 0.0)
g4 = einstein_at4(Fs, Gs, x0s, 0.05)
g4b = einstein_at4(Fs, Gs, x0s, 0.025)
g00_4d = (16.0 * g4b[0, 0] - g4[0, 0]) / 15.0
print("     静态完成场（4D 机）：G_00 = %+.6e（静态机三轮读数同口径为 +3.341389e-02）" % g00_4d)
print("       相对差 = %.2e" % (abs(g00_4d - 3.341389e-02) / 3.341389e-02))

print()
print("[G1b10-2] 线性系数：G00 对 A 抽动的 d2/dt2 系数与对 B 抽动的 laplace 系数")
a0, kk, om = 1e-2, 2.0, 3.0
P0 = (0.35, 0.9, 0.3, -0.2)


def Acos(T, X, Y, Z):
    return a0 * np.cos(kk * X - om * T)


def Bcos(T, X, Y, Z):
    return a0 * np.cos(kk * X - om * T)


gA = einstein_at4(Acos, lambda T, X, Y, Z: 0.0, P0, 0.03)
gB = einstein_at4(lambda T, X, Y, Z: 0.0, Bcos, P0, 0.03)
d2A = -om * om * float(Acos(*P0))
lapB = -kk * kk * float(Bcos(*P0))
print("     A 抽动（h00 ≠ 0）：G_00 = %+.4e，A'' = %+.4e，c_A = %+.4f" % (gA[0, 0], d2A, gA[0, 0] / d2A))
print("     B 抽动（hij ≠ 0）：G_00 = %+.4e，lap B = %+.4e，c_B = %+.4f" % (gB[0, 0], lapB, gB[0, 0] / lapB))

print()
print("[G1b10-3] 标量 ansatz（h00 = hij = 2 theta）的真空条件扫描")
print("     取 theta = a cos(kx - omega t)，a = 1e-2，扫 (omega/k) 与 k，测 |G| 的范数")
for kkk in (1.0, 2.0):
    for ratio in (0.0, 0.5, 1.0, 1.5, 2.0):
        oo = ratio * kkk
        def thf(T, X, Y, Z, kk2=kkk, o=oo):
            return 1e-2 * np.cos(kk2 * X - o * T)
        def FF(T, X, Y, Z, kk2=kkk, o=oo):
            return 2.0 * thf(T, X, Y, Z, kk2, o)
        G4 = einstein_at4(FF, FF, P0, 0.03)
        nrm = float(np.linalg.norm(G4))
        print("     k = %.1f，omega/k = %.1f：|G| = %.3e ｜ G_00 = %+.3e ｜ G_xx = %+.3e ｜ G_0x = %+.3e"
              % (kkk, ratio, nrm, G4[0, 0], G4[1, 1], G4[0, 1]))

print()
print("[G1b10-4] 与既有读数对照")
print("     记录场波动方程：张量桥 b / GV4 登记为阻尼波动方程（弱阻尼端 c_s = c，过阻尼端扩散泊松）")
print("     约束扇区（判据 U）：多余模由两条守恒量导出的 4 条约束定值，零空间 2（TT 两支）")
print("     本轮读数与两者的关系：标量 ansatz 的真空条件见 G1b10-3（若无非平凡解，则标量扇区为约束扇区）")
print("     非线性时间侧（完成式 theta_c 的含时残差）与含时源项（携源波包）为下一件，登记")

print()
print("[G1b10-5] 登记")
rows = [
    ("其一", "器械", "4D 补丁（含时间方向）的差分爱因斯坦张量：平坦零、静态回归一致"),
    ("其二", "线性系数", "c_A 与 c_B 由机器读出（对 h00 = A 与 hij = B 两种抽动）"),
    ("其三", "标量扇区", "标量 ansatz 的真空条件扫描给出该扇区是否传播（读数见上）"),
    ("其四", "余项", "非线性时间侧（完成式的含时残差）、含时源项、TT 扇区（需全对称 3x3 空间度规的器械）"),
]
for a1, a2, a3 in rows:
    print("     %-4s %-8s %s" % (a1, a2, a3))

print()
print("总运行时间 = %.1f 秒" % (time.time() - T0))
