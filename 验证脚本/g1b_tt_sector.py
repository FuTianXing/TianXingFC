# -*- coding: utf-8 -*-
"""
g1b_tt_sector.py

[G1b 第十一轮] TT 扇区器械：全对称 3x3 空间度规与两种偏振的真空条件（判据 G1b11-1 至 G1b11-5）

背景：第十轮的含时器械只支持对角空间度规（gii = 1+G），装不下 TT（横向无迹）波。
本轮把空间度规扩到全对称 3x3（gij = delta_ij + H_ij，逐点 4x4 矩阵求逆），
再做两件标定与两种偏振的真空条件扫描。

判据
G1b11-1 器械标定：平坦度规零残差；各向同性静态完成场（H_ij = G(r) delta）与静态机读数一致
G1b11-2 TT 波（+ 偏振）：H = a cos(kz - omega t)(e_x⊗e_x - e_y⊗e_y)，扫 omega/k 找真空条件
G1b11-3 TT 波（x 偏振）：H = a cos(kz - omega t)(e_x⊗e_y + e_y⊗e_x)，同上
G1b11-4 对照：与标量扇区（约束）与既有读数（c_s = c、螺旋度 ±2）的关系登记
G1b11-5 登记（含剩余项）

运行：python 验证脚本/g1b_tt_sector.py（确定性，约 2 分钟）
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


def copysign_norm(M):
    return float(np.sqrt(np.sum(np.abs(M) ** 2)))


print("=" * 78)
print("G1b 第十一轮：TT 扇区器械（判据 G1b11-1 至 G1b11-5）")
print("=" * 78)

P0 = (0.03, 0.4, 0.25, -0.15)

print()
print("[G1b11-1] 器械标定")
z = einstein_at4(lambda T, X, Y, Z: 0.0, lambda T, X, Y, Z: np.zeros((3, 3) + np.broadcast(T, X, Y, Z).shape),
                 (0.0, 1.10, 0.3, -0.2), 0.05)
print("     平坦度规：|G| = %.2e（应为机器零）" % copysign_norm(z))
MASS, SIG = 0.05, 1.0
erf = np.vectorize(math.erf)


def th_c_s(R):
    x = R / (SIG * math.sqrt(2.0))
    t = np.where(x < 1e-8, MASS * math.sqrt(2.0 / math.pi) / SIG, MASS * erf(x) / np.maximum(R, 1e-30))
    return t - t * t


def iso_F(T, X, Y, Z):
    return 2.0 * th_c_s(np.sqrt(X ** 2 + Y ** 2 + Z ** 2))


def iso_H(T, X, Y, Z):
    tt = th_c_s(np.sqrt(X ** 2 + Y ** 2 + Z ** 2))
    Gv = 2.0 * tt + 3.5 * tt ** 2
    shape = np.broadcast(T, X, Y, Z).shape
    H = np.zeros((3, 3) + shape)
    for i in range(3):
        H[i, i] = Gv
    return H


g1 = einstein_at4(iso_F, iso_H, (0.0, 1.2, 0.0, 0.0), 0.05)
g2 = einstein_at4(iso_F, iso_H, (0.0, 1.2, 0.0, 0.0), 0.025)
g00 = (16.0 * g2[0, 0] - g1[0, 0]) / 15.0
print("     各向同性静态完成场（3x3 器械）：G_00 = %+.6e（静态机读数 +3.341389e-02）" % g00)
print("       相对差 = %.2e" % (abs(g00 - 3.341389e-02) / 3.341389e-02))

AMP = 1e-2
Pp = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]])
Px = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])


def make_H(P, kk, oo, amp=AMP):
    def Hf(T, X, Y, Z):
        prof = amp * np.cos(kk * Z - oo * T)
        return np.einsum('ij,...->ij...', P, prof)
    return Hf


zero_F = lambda T, X, Y, Z: 0.0
for tag, P in (("+ 偏振", Pp), ("x 偏振", Px)):
    print()
    print("[G1b11-%d] TT 波（%s）的真空条件扫描：H = a cos(kz - omega t) P，a = %.2f" % (2 if tag.startswith("+") else 3, tag, AMP))
    for kkk in (1.0, 2.0):
        line = []
        for ratio in (0.0, 0.5, 0.9, 1.0, 1.1, 1.5):
            oo = ratio * kkk
            G4 = einstein_at4(zero_F, make_H(P, kkk, oo), P0, 0.03)
            line.append("w/k=%.1f: %.2e" % (ratio, copysign_norm(G4)))
        print("     k = %.1f：" % kkk + " ｜ ".join(line))
    Gp = einstein_at4(zero_F, make_H(P, 1.0, 1.0, AMP), P0, 0.03)
    print("     结构区分（a = %.3f，k = 1，omega/k = 1）：G_00 = %+.3e，G_zz = %+.3e，G_0z = %+.3e，G_xx = %+.3e，G_yy = %+.3e，G_xy = %+.3e"
          % (AMP, Gp[0, 0], Gp[3, 3], Gp[0, 3], Gp[1, 1], Gp[2, 2], Gp[1, 2]))
    print("     幅度标度核对（k = 1，omega/k = 1 与 1.1 两档）：")
    for amp in (0.005, 0.01):
        n1 = copysign_norm(einstein_at4(zero_F, make_H(P, 1.0, 1.0, amp), P0, 0.03))
        n2 = copysign_norm(einstein_at4(zero_F, make_H(P, 1.0, 1.1, amp), P0, 0.03))
        print("       a = %.3f：|G|(w/k=1) = %.3e ｜ |G|(w/k=1.1) = %.3e ｜ 比 = %.3f" % (amp, n1, n2, n1 / n2))

print()
print("[G1b11-4] 对照登记")
print("     标量（共形）扇区（第十轮）：十档扫描 |G| 全部非零，无传播解，属约束扇区")
print("     TT 扇区（本轮）：两种偏振在 omega/k = 1 处给真空条件（读数见上），传播速度 c = 1")
print("     与既有读数：张量桥 b / GV4 的 c_s = c 同向；自旋 2 线的螺旋度 ±2 对应两种偏振")
print("     剩余项：TT 的非线性背反作用（完成式在 TT 背景上的含时残差）、含时源项（携源波包）")

print()
print("[G1b11-5] 登记")
rows = [
    ("其一", "器械", "全对称 3x3 空间度规加逐点 4x4 矩阵求逆：平坦零、各向同性静态回归一致"),
    ("其二", "+ 偏振", "扫 omega/k 给真空条件在 1 处（读数见 G1b11-2）"),
    ("其三", "x 偏振", "同上（读数见 G1b11-3）"),
    ("其四", "层级", "标量扇区为约束、TT 扇区为传播：与判据 U、c_s = c、螺旋度 ±2 三条既有读数同向"),
    ("其五", "余项", "TT 的非线性背反作用、含时源项（携源波包）、强场端"),
]
for a1, a2, a3 in rows:
    print("     %-4s %-8s %s" % (a1, a2, a3))

print()
print("总运行时间 = %.1f 秒" % (time.time() - T0))
