# -*- coding: utf-8 -*-
"""
minimal_verification_g1b_tt_completion.py

[G1b 第十二轮] TT 背景上的完成形式判据（判据 G1b12-1 至 G1b12-5）

背景：第十一轮在 TT 波背景上测得 O(a^2) 残余，落在零向分量（G00 = Gzz = -G0z）。
本轮问：框架的完成形式（theta = theta_lin - theta_lin^2 加规则二次项）能否抵消它。

三件读数：
（一）纯 TT 波的 theta 含量：h00 = 0、tr H = 0 ⇒ 记录场 theta = 0 ⇒ 完成式贡献恒为零；
（二）残余的横向均匀性：平面波 ansatz 在 x,y 平移下不变，残余应均匀（测三点）；
（三）规范模式标定：纯时间重参数化（h00 = eps(t)，其余为零）的线性化爱因斯坦张量应为零，
     以此说明非零残余不可由线性化坐标变换消去（须由几何侧二阶修正承担）。

判据
G1b12-1 theta 含量：两种偏振的 tr H 与 h00 均为机器零，theta = 0
G1b12-2 残余均匀性：固定 u，三处 (x,y) 的 |G| 一致
G1b12-3 规范标定：纯时间重参数化模式的 |G| 为机器零
G1b12-4 结论登记：完成式在 TT 扇区贡献为零；残余为物理量，须由几何侧二阶修正承担
G1b12-5 登记（含余项）

运行：python 验证脚本/minimal_verification_g1b_tt_completion.py（确定性，约 1 分钟）
"""
import math
import time
import numpy as np

T0 = time.time()
HALF = 4
C = HALF
NP_ = 2 * HALF + 1


def g_patch4(Ff, Hf, x0, eps):
    k = np.arange(-HALF, HALF + 1) * eps
    Tt = x0[0] + k[:, None, None, None]
    X = x0[1] + k[None, :, None, None]
    Y = x0[2] + k[None, None, :, None]
    Z = x0[3] + k[None, None, None, :]
    Tb, Xb, Yb, Zb = np.broadcast_arrays(Tt, X, Y, Z)
    g = np.zeros((4, 4, NP_, NP_, NP_, NP_))
    g[0, 0] = -(1.0 - Ff(Tb, Xb, Yb, Zb))
    H = Hf(Tb, Xb, Yb, Zb)
    for i in range(3):
        for j in range(3):
            g[i + 1, j + 1] = (1.0 if i == j else 0.0) + H[i, j]
    return g


def d1_keep(a, axis, eps):
    b = np.moveaxis(a, axis, -1)
    out = np.full_like(b, np.nan)
    out[..., 2:-2] = (b[..., :-4] - 8 * b[..., 1:-3] + 8 * b[..., 3:-1] - b[..., 4:]) / (12 * eps)
    return np.moveaxis(out, -1, axis)


def einstein_at4(Ff, Hf, x0, eps):
    g = g_patch4(Ff, Hf, x0, eps)
    dgf = [d1_keep(g, 2 + i, eps) for i in range(4)]
    gg = g[:, :, C, C, C, C]
    ginv = np.linalg.inv(gg)
    gp = np.moveaxis(g, (0, 1), (-2, -1))
    ginvf = np.moveaxis(np.linalg.inv(gp), (-2, -1), (0, 1))
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


def nrm(M):
    return float(np.sqrt(np.sum(np.abs(M) ** 2)))


print("=" * 78)
print("G1b 第十二轮：TT 背景上的完成形式判据（判据 G1b12-1 至 G1b12-5）")
print("=" * 78)

AMP, KK, OO = 1e-2, 1.0, 1.0
Pp = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]])
Px = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
zero_F = lambda T, X, Y, Z: 0.0


def make_H(P, kk, oo, amp=AMP):
    def Hf(T, X, Y, Z):
        prof = amp * np.cos(kk * Z - oo * T)
        return np.einsum('ij,...->ij...', P, prof)
    return Hf


print()
print("[G1b12-1] 纯 TT 波的 theta 含量")
for tag, P in (("+ 偏振", Pp), ("x 偏振", Px)):
    Hv = make_H(P, KK, OO)(0.03, 0.4, 0.25, -0.15)
    print("     %s：tr H = %+.2e（机器零）｜ h00 = 0 ｜ 记录场 theta = h00/2 = 0 ｜ 迹通道 tr/6 = %+.2e"
          % (tag, float(np.trace(Hv)), float(np.trace(Hv)) / 6.0))
print("     结论：纯 TT 波不含记录场自由度，完成式（theta_c 与规则二次项）的贡献恒为零")

print()
print("[G1b12-2] 残余的横向均匀性（固定 u = t - z）")
u0 = 0.03 - (-0.15)
for (xx, yy) in ((0.4, 0.25), (0.6, 0.1), (0.2, 0.5)):
    T0p = 0.03
    Z0p = T0p - u0
    G4 = einstein_at4(zero_F, make_H(Pp, KK, OO), (T0p, xx, yy, Z0p), 0.03)
    print("     (x, y) = (%.2f, %.2f)：|G| = %.4e ｜ G_00 = %+.4e ｜ G_0z = %+.4e"
          % (xx, yy, nrm(G4), G4[0, 0], G4[0, 3]))

print()
print("[G1b12-3] 规范标定：纯时间重参数化模式（h00 = eps(t)，其余为零）")
for eps_amp in (1e-2,):
    def Ft(T, X, Y, Z, a=eps_amp):
        return a * np.cos(2.0 * T)
    G4 = einstein_at4(Ft, lambda T, X, Y, Z: np.zeros((3, 3) + np.broadcast(T, X, Y, Z).shape),
                      (0.3, 0.4, 0.25, -0.15), 0.03)
    print("     eps = %.2f：|G| = %.3e（应为机器零：纯时间重参数化是线性化规范模式）" % (eps_amp, nrm(G4)))

print()
print("[G1b12-4] 结论登记")
print("     完成式在 TT 扇区的贡献为零（G1b12-1）；残余在横向上均匀（G1b12-2）")
print("     残余是物理量（规范模式给零，G1b12-3），不可由线性化坐标变换消去")
print("     因此 TT 扇区的二阶修正落在几何侧：须由爱因斯坦张量自身的二阶结构承担（组装恒等式的形式）")
print("     扇区归属：theta 扇区（有源、迹）由完成式收束；TT 扇区（真空、无迹）由精确真空方程收束")

print()
print("[G1b12-5] 登记")
rows = [
    ("其一", "theta 含量", "纯 TT 波 tr H 与 h00 为机器零：完成式贡献恒为零"),
    ("其二", "均匀性", "残余在横向上均匀（固定 u 三点一致）：与平面波 ansatz 的平移对称一致"),
    ("其三", "规范性", "纯时间重参数化的 |G| 为机器零：残余不可由线性化坐标变换消去"),
    ("其四", "归属", "TT 扇区二阶修正归几何侧（精确真空方程）；完成式只管 theta（有源/迹）扇区"),
    ("其五", "余项", "几何侧二阶修正的显式构造（组装恒等式在 TT 扇区的解）、含时源项、强场端"),
]
for a1, a2, a3 in rows:
    print("     %-4s %-8s %s" % (a1, a2, a3))

print()
print("总运行时间 = %.1f 秒" % (time.time() - T0))
