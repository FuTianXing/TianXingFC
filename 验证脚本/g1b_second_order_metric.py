# -*- coding: utf-8 -*-
"""
g1b_second_order_metric.py

[G1b 第十三轮] 几何侧二阶修正的显式构造（判据 G1b13-1 至 G1b13-5）

背景：第十二轮判定 TT 扇区的 O(a^2) 残余归几何侧，形式为组装恒等式
G^(1)[H^(2)] + G^(2)[H^(1)] = 0。本轮把它解出来：残余的 u-结构、四个响应基
（h00 与 hij 的常数与 cos(2u) 两个形状）、系数求解与装配核验。

口径：TT 波 H^(1) = a cos(kz - omega t) P_+，k = omega = 1，a = 1e-2；
修正取 H^(2)：h00 = A0 + A2 cos(2u)，hij = (B0 + B2 cos(2u)) delta，u = t - z。
残余与响应都在三维分量 (G00, Gzz, G0z) 上读（第十二轮：其余分量为机器零）。

判据
G1b13-1 残余的 u-结构（扫相位，报三分量）
G1b13-2 四个响应基（单位幅度 1e-2 的线性响应，除幅度回原）
G1b13-3 系数求解与装配核验（最小二乘 + 修正后 |G| 的降阶）
G1b13-4 幅度标度（a = 1e-2 与 5e-3：修正系数按 a^2、剩余按 a^3）
G1b13-5 登记

运行：python 验证脚本/g1b_second_order_metric.py（确定性，约 1 分钟）
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


AMP = 1e-2
Pp = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]])
zero_F = lambda T, X, Y, Z: 0.0
zero_H = lambda T, X, Y, Z: np.zeros((3, 3) + np.broadcast(T, X, Y, Z).shape)


def H_tt(T, X, Y, Z, a=AMP):
    prof = a * np.cos(1.0 * Z - 1.0 * T)
    return np.einsum('ij,...->ij...', Pp, prof)


def null3(G4):
    return np.array([G4[0, 0], G4[3, 3], G4[0, 3]])


print("=" * 78)
print("G1b 第十三轮：几何侧二阶修正的显式构造（判据 G1b13-1 至 G1b13-5）")
print("=" * 78)

print()
print("[G1b13-1] 残余的 u-结构（TT 波，a = %.3f）" % AMP)
T0p, X0p, Y0p = 0.0, 0.4, 0.25
for u in (0.0, 0.4, 0.8, 1.2, 1.6):
    Z0p = T0p - u
    G4 = einstein_at4(zero_F, H_tt, (T0p, X0p, Y0p, Z0p), 0.03)
    v = null3(G4)
    print("     u = %.1f：G00 = %+.4e ｜ Gzz = %+.4e ｜ G0z = %+.4e ｜ |G| = %.3e"
          % (u, v[0], v[1], v[2], nrm(G4)))

print()
print("[G1b13-2] 四个响应基（幅度 1e-2 的线性响应，已归一）")
AMP_R = 1e-2
P0 = (0.35, 0.4, 0.25, 0.35)


def A2f(T, X, Y, Z):
    return AMP_R * np.cos(2.0 * (T - Z))


def A0f(T, X, Y, Z):
    return AMP_R + 0.0 * T


def B2f(T, X, Y, Z):
    prof = AMP_R * np.cos(2.0 * (T - Z))
    shape = np.broadcast(T, X, Y, Z).shape
    H = np.zeros((3, 3) + shape)
    for i in range(3):
        H[i, i] = prof
    return H


def B0f(T, X, Y, Z):
    shape = np.broadcast(T, X, Y, Z).shape
    H = np.zeros((3, 3) + shape)
    for i in range(3):
        H[i, i] = AMP_R
    return H


R = {}
R["A2"] = null3(einstein_at4(A2f, zero_H, P0, 0.03)) / AMP_R
R["A0"] = null3(einstein_at4(A0f, zero_H, P0, 0.03)) / AMP_R
R["B2"] = null3(einstein_at4(zero_F, B2f, P0, 0.03)) / AMP_R
R["B0"] = null3(einstein_at4(zero_F, B0f, P0, 0.03)) / AMP_R
for k in ("A2", "A0", "B2", "B0"):
    print("     R[%s] = (G00, Gzz, G0z) = (%+.4e, %+.4e, %+.4e)" % (k, R[k][0], R[k][1], R[k][2]))

print()
print("[G1b13-3] 系数求解与装配核验")
def residual_at(u, a=AMP):
    Zp = T0p - u
    return null3(einstein_at4(zero_F, lambda T, X, Y, Z, aa=a: H_tt(T, X, Y, Z, aa), (T0p, X0p, Y0p, Zp), 0.03))

import itertools
U_SAMPLES = (0.0, 0.4, 0.8, 1.2, 1.6)
Mrows = []
brows = []
for u in U_SAMPLES:
    r = residual_at(u)
    Mrows.append([R["A2"][0] * math.cos(2 * u), R["A0"][0], R["B2"][0] * math.cos(2 * u), R["B0"][0]])
    brows.append(-r[0])
M4 = np.array(Mrows)
b4 = np.array(brows)
sol, res, rank, sv = np.linalg.lstsq(M4, b4, rcond=None)
A2, A0, B2, B0 = sol
print("     由 G00 分量求解（四点最小二乘）：A2 = %+.4e，A0 = %+.4e，B2 = %+.4e，B0 = %+.4e" % (A2, A0, B2, B0))
print("     拟合残差（G00 分量）＝ %.3e（对残余幅度 %.3e 的比 = %.2e）"
      % (float(np.linalg.norm(M4 @ sol - b4)), float(np.linalg.norm(b4)), float(np.linalg.norm(M4 @ sol - b4)) / float(np.linalg.norm(b4))))

def F_corr(T, X, Y, Z):
    return A0 + A2 * np.cos(2.0 * (T - Z))


def H_corr(T, X, Y, Z):
    prof_c = B0 + B2 * np.cos(2.0 * (T - Z))
    shape = np.broadcast(T, X, Y, Z).shape
    H = np.zeros((3, 3) + shape)
    for i in range(3):
        H[i, i] = prof_c
    H = H + H_tt(T, X, Y, Z)
    return H

for u in (0.4, 1.2):
    Zp = T0p - u
    g_un = einstein_at4(zero_F, lambda T, X, Y, Z: H_tt(T, X, Y, Z), (T0p, X0p, Y0p, Zp), 0.03)
    g_co = einstein_at4(F_corr, H_corr, (T0p, X0p, Y0p, Zp), 0.03)
    print("     u = %.1f：未修正 |G| = %.3e ｜ 修正后 |G| = %.3e ｜ 降幅 = %.1f 倍"
          % (u, nrm(g_un), nrm(g_co), nrm(g_un) / max(nrm(g_co), 1e-30)))

print()
print("[G1b13-4] 幅度标度（修正系数按 a^2、剩余按 a^3）")
for a in (AMP, AMP / 2.0):
    r1 = residual_at(0.4, a)
    M1 = np.array([R["A2"][0] * math.cos(0.8), R["A0"][0], R["B2"][0] * math.cos(0.8), R["B0"][0]])
    s1, _, _, _ = np.linalg.lstsq(M1.reshape(1, 4), np.array([-r1[0]]), rcond=None)
    Zp = T0p - 0.4
    g_un = einstein_at4(zero_F, lambda T, X, Y, Z, aa=a: H_tt(T, X, Y, Z, aa), (T0p, X0p, Y0p, Zp), 0.03)
    Fa = lambda T, X, Y, Z, s=s1: s[1] + s[0] * np.cos(2.0 * (T - Z))
    Ha = lambda T, X, Y, Z, s=s1, aa=a: (lambda base: base + np.einsum('ij,...->ij...', np.eye(3), s[3] + s[2] * np.cos(2.0 * (T - Z))))(H_tt(T, X, Y, Z, aa))
    g_co = einstein_at4(Fa, Ha, (T0p, X0p, Y0p, Zp), 0.03)
    print("     a = %.4f：未修正 |G| = %.3e ｜ 修正后 |G| = %.3e ｜ 比 = %.3f"
          % (a, nrm(g_un), nrm(g_co), nrm(g_co) / nrm(g_un)))

print()
print("[G1b13-5] 登记")
rows = [
    ("其一", "残余结构", "残余为 u 的函数（扫相位读数），落在 (G00, Gzz, G0z) 三分量"),
    ("其二", "响应基", "A2/A0/B2/B0 四个响应基的读数（G00 分量的系数求解由它给出）"),
    ("其三", "装配", "修正后 |G| 的降阶读数；降阶比给出二阶修正的闭合程度"),
    ("其四", "标度", "修正系数按 a^2、剩余按 a^3 的对照"),
    ("其五", "余项", "含时源项（携源波包）、强场端、以及（如需）h0i 修正（本轮 ansatz 无 h0i）"),
]
for a1, a2, a3 in rows:
    print("     %-4s %-8s %s" % (a1, a2, a3))

print()
print("总运行时间 = %.1f 秒" % (time.time() - T0))
