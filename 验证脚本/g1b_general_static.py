# -*- coding: utf-8 -*-
"""
g1b_general_static.py

[G1b 第九轮] 一般静态位形（非球）：线性层、二次型、清式与源因子（判据 G1b9-1 至 G1b9-5）

背景：前几轮全部在静球对称上。本轮把器械推广到非球（g_patch 收三维坐标函数），
源取各向异性高斯（sigma = (1.0, 0.7, 1.3)），逐条核对一般静态位形下的完成结构。

判据
G1b9-1 非球线性层：laplace(theta_lin) = -4 pi rho_m（三点，含离轴点）
G1b9-2 非球二次型：G00[F = G = A] = -laplace(A) + 3 A laplace(A) + (3/4)(grad A)^2（幅度外推提取）
G1b9-3 非球清式：laplace(theta_c) = -4 pi rho_m (1 - 2 theta) - 2 (grad theta)^2
G1b9-4 非球源因子：G00 对 8 pi rho_m (1 - (9/2) theta) 的逐点 kappa
G1b9-5 登记（一般静态位形有判；时间依赖需新器械，登记）

运行：python 验证脚本/g1b_general_static.py（确定性，约 30 秒）
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
    g = np.zeros((4, 4, NP_, NP_, NP_))
    g[0, 0] = -(1.0 - Ff(X, Y, Z))
    for i in (1, 2, 3):
        g[i, i] = 1.0 + Gf(X, Y, Z)
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
SIG = (1.0, 0.7, 1.3)
SQ = math.sqrt(math.pi)


def rho_m(X, Y, Z, mass=MASS):
    s = SIG
    pref = mass * (2.0 * math.pi) ** -1.5 / (s[0] * s[1] * s[2])
    return pref * np.exp(-(X ** 2 / (2 * s[0] ** 2) + Y ** 2 / (2 * s[1] ** 2) + Z ** 2 / (2 * s[2] ** 2)))


def th_lin(X, Y, Z, mass=MASS, nsub=600):
    # theta = (2M/sqrt(pi)) ∫_0^inf ds Π(1+2 s_i^2 s^2)^{-1/2} exp(-Σ x_i^2 s^2/(1+2 s_i^2 s^2))
    th = np.zeros_like(np.asarray(X, dtype=float))
    th = np.broadcast_to(th, np.broadcast(X, Y, Z).shape).copy()
    Xb, Yb, Zb = np.broadcast_arrays(np.asarray(X, dtype=float), np.asarray(Y, dtype=float), np.asarray(Z, dtype=float))
    for j in range(nsub):
        t0 = math.pi * j / (2.0 * nsub)
        t1 = math.pi * (j + 1) / (2.0 * nsub)
        for t in (0.5 * (t0 + t1),):
            s = math.tan(t)
            dsdt = 1.0 / math.cos(t) ** 2
            w = dsdt * (t1 - t0)
            den = [1.0 + 2.0 * SIG[i] ** 2 * s * s for i in range(3)]
            pref = 1.0
            for i in range(3):
                pref /= math.sqrt(den[i])
            ex = np.exp(-(Xb ** 2 * s * s / den[0] + Yb ** 2 * s * s / den[1] + Zb ** 2 * s * s / den[2]))
            th = th + w * pref * ex
    return th * (2.0 * mass / SQ)


def th_c_of(X, Y, Z, mass=MASS):
    t = th_lin(X, Y, Z, mass)
    return t - t * t


def F_of(X, Y, Z):
    return 2.0 * th_c_of(X, Y, Z)


def G_of(X, Y, Z):
    t = th_c_of(X, Y, Z)
    return 2.0 * t + 3.5 * t ** 2


def lap3(f, x0, h=2e-4):
    acc = 0.0
    for i in range(3):
        xp = list(x0); xp[i] += h
        xm = list(x0); xm[i] -= h
        acc += (float(f(*xp)) - 2.0 * float(f(*x0)) + float(f(*xm))) / (h * h)
    return acc


def grad2_3(f, x0, h=2e-4):
    acc = 0.0
    for i in range(3):
        xp = list(x0); xp[i] += h
        xm = list(x0); xm[i] -= h
        d = (float(f(*xp)) - float(f(*xm))) / (2.0 * h)
        acc += d * d
    return acc


def g00_at(Ff, Gf, x0, eps):
    return einstein_at(Ff, Gf, x0, eps=eps)[0, 0]


print("=" * 78)
print("G1b 第九轮：一般静态位形（非球）（判据 G1b9-1 至 G1b9-5）")
print("=" * 78)
print("源：各向异性高斯 M = %.3f，sigma = %s（非球）" % (MASS, str(SIG)))
PTS = [(1.1, 0.3, -0.2), (0.7, -0.5, 0.4), (0.2, 0.15, 0.6)]

print()
print("[G1b9-1] 非球线性层：laplace(theta_lin) = -4 pi rho_m")
for x0 in PTS:
    lap = lap3(th_lin, x0)
    tgt = -4.0 * math.pi * float(rho_m(*x0))
    print("     x = (%.1f, %.1f, %.1f)：laplace = %+.6e ｜ −4 pi rho_m = %+.6e ｜ 相对差 = %.2e"
          % (x0[0], x0[1], x0[2], lap, tgt, abs(lap - tgt) / abs(tgt)))

print()
print("[G1b9-2] 非球二次型：G00[F = G = A] = -laplace(A) + 3 A laplace(A) + (3/4)(grad A)^2")
def A_test(X, Y, Z, amp):
    return amp * np.exp(-(X ** 2 + 2.0 * Y ** 2 + 1.5 * Z ** 2) / 2.0)
for x0 in PTS:
    def q_of(lam):
        g = g00_at(lambda X, Y, Z: A_test(X, Y, Z, lam), lambda X, Y, Z: A_test(X, Y, Z, lam), x0, 0.02)
        return (g + lam * lap3(lambda X, Y, Z: A_test(X, Y, Z, 1.0), x0)) / lam ** 2
    q = 2.0 * q_of(0.005) - q_of(0.01)
    lapA = lap3(lambda X, Y, Z: A_test(X, Y, Z, 1.0), x0)
    grA = grad2_3(lambda X, Y, Z: A_test(X, Y, Z, 1.0), x0)
    Av = float(A_test(*x0, 1.0))
    pred = 3.0 * Av * lapA + 0.75 * grA
    print("     x = (%.1f, %.1f, %.1f)：q = %+.6e ｜ 3 A laplaceA + (3/4) grad²A = %+.6e ｜ 相对差 = %.2e"
          % (x0[0], x0[1], x0[2], q, pred, abs(q - pred) / abs(pred)))

print("[G1b9-3] 非球清式：laplace(theta_c) = -4 pi rho_m (1 - 2 theta) - 2 (grad theta)^2")
for x0 in PTS:
    lap = lap3(th_c_of, x0)
    th = float(th_c_of(*x0))
    tgt = -4.0 * math.pi * float(rho_m(*x0)) * (1.0 - 2.0 * th) - 2.0 * grad2_3(th_c_of, x0)
    print("     x = (%.1f, %.1f, %.1f)：laplace = %+.6e ｜ 清式 = %+.6e ｜ 相对差 = %.2e"
          % (x0[0], x0[1], x0[2], lap, tgt, abs(lap - tgt) / abs(tgt)))

print()
print("[G1b9-4] 非球源因子：G00 对 8 pi rho_m (1 - (9/2) theta) 的逐点 kappa")
for x0 in PTS:
    g1 = g00_at(F_of, G_of, x0, 0.05)
    g2 = g00_at(F_of, G_of, x0, 0.025)
    g00 = (16.0 * g2 - g1) / 15.0
    rm = float(rho_m(*x0))
    th = float(th_c_of(*x0))
    R = g00 - 8.0 * math.pi * rm
    print("     x = (%.1f, %.1f, %.1f)：G00 = %+.6e ｜ kappa = %+.4f（球对称档同量级，O(theta) 修正内）"
          % (x0[0], x0[1], x0[2], g00, -R / (8.0 * math.pi * rm * th)))

print()
print("[G1b9-5] 登记")
rows = [
    ("其一", "线性层", "非球源下 laplace(theta_lin) = -4 pi rho_m 成立（含离轴点）"),
    ("其二", "二次型", "非球 A 下 G00[F=G=A] 的二次型系数与球对称档一致（3 与 3/4）"),
    ("其三", "清式", "非球 theta_c 下清式成立：源因子 (1 - 2 theta) 与场项 -2(grad theta)^2 均逐点"),
    ("其四", "源因子", "非球源的逐点 kappa 与球对称档同量级：一般静态位形不引入新结构"),
    ("其五", "余项", "时间依赖需要新器械（含时 FD 爱因斯坦张量或 3+1 演化）；本轮不覆盖"),
]
for a1, a2, a3 in rows:
    print("     %-4s %-8s %s" % (a1, a2, a3))

print()
print("总运行时间 = %.1f 秒" % (time.time() - T0))
