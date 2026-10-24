# -*- coding: utf-8 -*-
r"""
g1b_cross_generality.py

[G1b 第二十二轮] 闭式解的一般性复核：振幅不等与旋转族斜交角（判据 G22-1 至 G22-5）

背景：第二十一轮给出交叉修正的闭式解（两模式、有理系数）。本轮把它从单构型推到
一般两束交叉的两个方向：其一，振幅不等（$a \ne b$，闭式的 $ab$ 结构应保持）；
其二，斜交角（以绕 $z$ 的旋转族实现：波 1 沿 $z$、偏振随族旋转，波 2 方向与偏振
一起旋转，生成 $30^\circ$、$60^\circ$、$90^\circ$ 的斜交），闭式解应随构型整体旋转。
并设对照：把未旋转的闭式施加到旋转构型上，应无明显降幅。

解析预期：
1) 交叉项是双线性的：$2G_2[h_1, h_2] = ab\cdot(\text{单位幅度双线性型})$，故闭式系数
   只需按 $ab$ 缩放，$a/b$ 任意时降幅相同（到三阶地板）。
2) 旋转族：整个构型（两束的方向与偏振）绕 $z$ 旋转 $\phi$，闭式解随之旋转；
   未旋转的闭式对旋转构型无效（对照读数接近 1）。

判据
G22-1 基线：平坦零、单波完成复核、$\phi = 0$ 与 $a = b$ 的闭式降幅复现
G22-2 振幅不等：两档比值（2 与 0.5）
G22-3 旋转族：$30^\circ$、$60^\circ$、$90^\circ$ 的降幅
G22-4 对照：未旋转闭式施加于旋转构型
G22-5 登记

运行：python 验证脚本/g1b_cross_generality.py（确定性，约 1 分钟）
"""
import math
import time
import numpy as np

T_START = time.time()
HALF = 4
C = HALF
NP_ = 2 * HALF + 1


def g_patch4(Ff, Nf, Hf, x0, eps):
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
    if Nf is not None:
        Nv = Nf(Tb, Xb, Yb, Zb)
        for i in range(3):
            g[0, i + 1] = Nv[i]
            g[i + 1, 0] = Nv[i]
    return g


def d1_keep(a, axis, eps):
    b = np.moveaxis(a, axis, -1)
    out = np.full_like(b, np.nan)
    out[..., 2:-2] = (b[..., :-4] - 8 * b[..., 1:-3] + 8 * b[..., 3:-1] - b[..., 4:]) / (12 * eps)
    return np.moveaxis(out, -1, axis)


def einstein_at4(Ff, Hf, x0, eps, Nf=None):
    g = g_patch4(Ff, Nf, Hf, x0, eps)
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


def ein_ext(Ff, Hf, x0, eps=0.03, Nf=None):
    A = einstein_at4(Ff, Hf, x0, eps, Nf)
    B = einstein_at4(Ff, Hf, x0, eps / 2.0, Nf)
    return (16.0 * B - A) / 15.0


def nrm(M):
    return float(np.sqrt(np.sum(np.abs(M) ** 2)))


zero_F = lambda T, X, Y, Z: 0.0


def zero_H(T, X, Y, Z):
    return np.zeros((3, 3) + np.broadcast(T, X, Y, Z).shape)


def zero_N(T, X, Y, Z):
    return np.zeros((3,) + np.broadcast(T, X, Y, Z).shape)


P1o = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]])
P2o = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
T1o = np.diag([1.0, 1.0, 0.0])
T2o = np.diag([0.0, 1.0, 1.0])


def rotz(phi):
    c, s = math.cos(phi), math.sin(phi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def prof1(u, a):
    return (3.0 * a * a / 16.0) * np.cos(2.0 * u) - (a * a / 8.0) * u * u


def prof2(v, b):
    return (3.0 * b * b / 16.0) * np.cos(2.0 * v) - (b * b / 8.0) * v * v


def phase(kk, T, X, Y, Z):
    return np.cos(kk[0] * T + kk[1] * X + kk[2] * Y + kk[3] * Z)


def field_w1(phi, a, comp=False):
    R = rotz(phi)
    P1 = R @ P1o @ R.T
    def Hf(T, X, Y, Z, a=a, comp=comp, P1=P1):
        u = T - Z
        A = a * np.cos(u)
        b1 = prof1(u, a) if comp else 0.0
        sh = np.broadcast(T, X, Y, Z).shape
        H = np.zeros((3, 3) + sh)
        for i in range(3):
            for j in range(3):
                H[i, j] = A * P1[i, j] + b1 * T1o[i, j]
        return H
    return zero_F, Hf, zero_N


def field_w2(phi, b, comp=False):
    R = rotz(phi)
    P2 = R @ P2o @ R.T
    T2 = R @ T2o @ R.T
    d = R @ np.array([1.0, 0.0, 0.0])
    def Hf(T, X, Y, Z, b=b, comp=comp, P2=P2, T2=T2, d=d):
        v = T - (d[0] * X + d[1] * Y + d[2] * Z)
        B = b * np.cos(v)
        b2 = prof2(v, b) if comp else 0.0
        sh = np.broadcast(T, X, Y, Z).shape
        H = np.zeros((3, 3) + sh)
        for i in range(3):
            for j in range(3):
                H[i, j] = B * P2[i, j] + b2 * T2[i, j]
        return H
    return zero_F, Hf, zero_N


def field_sum(phi, a, b, comp=True, corr=None):
    F1, H1, N1 = field_w1(phi, a, comp)
    F2, H2, N2 = field_w2(phi, b, comp)
    def Ff(T, X, Y, Z):
        if corr is not None:
            return corr[0](T, X, Y, Z)
        return np.zeros_like(T + X + Y + Z)

    def Hf(T, X, Y, Z):
        H = H1(T, X, Y, Z) + H2(T, X, Y, Z)
        if corr is not None:
            H = H + corr[1](T, X, Y, Z)
        return H

    def Nf(T, X, Y, Z):
        sh = np.broadcast(T, X, Y, Z).shape
        if corr is not None:
            return corr[2](T, X, Y, Z)
        return np.zeros((3,) + sh)
    return Ff, Hf, Nf


def corr_field(phi, a, b, rot=True):
    Rc = rotz(phi) if rot else np.eye(3)
    d = Rc @ np.array([1.0, 0.0, 0.0])
    ex = Rc @ np.array([1.0, 0.0, 0.0])
    ey = Rc @ np.array([0.0, 1.0, 0.0])
    ez = np.array([0.0, 0.0, 1.0])
    kp = np.array([2.0, -d[0], -d[1], -1.0 - d[2]])
    km = np.array([0.0, d[0], d[1], -1.0 + d[2]])
    ab = a * b
    nvec_p = -5.0 / 18.0 * (ex + ez)
    nvec_m = 1.0 / 8.0 * (ex + ez)

    def Ff(T, X, Y, Z):
        return ab * (-5.0 / 36.0 * phase(kp, T, X, Y, Z) - 0.25 * phase(km, T, X, Y, Z))

    def Nf(T, X, Y, Z):
        sh = np.broadcast(T, X, Y, Z).shape
        Nv = np.zeros((3,) + sh)
        for i in range(3):
            Nv[i] = ab * (nvec_p[i] * phase(kp, T, X, Y, Z) + nvec_m[i] * phase(km, T, X, Y, Z))
        return Nv

    def Hf(T, X, Y, Z):
        sh = np.broadcast(T, X, Y, Z).shape
        H = np.zeros((3, 3) + sh)
        for kmv, cp, cy in ((kp, 5.0 / 8.0, -1.0), (km, -5.0 / 8.0, 0.0)):
            p = ab * phase(kmv, T, X, Y, Z)
            for i in range(3):
                for j in range(3):
                    H[i, j] += p * (cp * (ex[i] * ex[j] + ez[i] * ez[j]) + cy * ey[i] * ey[j])
        return H
    return Ff, Hf, Nf


def G_at(field, x0, eps=0.03):
    Ff, Hf, Nf = field
    return ein_ext(Ff, Hf, x0, eps, Nf)


PTS = [(0.0, 0.7, 0.3, -0.5), (0.0, 1.1, 0.3, -0.9), (0.0, 0.5, 0.3, -1.3)]

print("=" * 78)
print("G1b 第二十二轮：闭式解的一般性复核（判据 G22-1 至 G22-5）")
print("=" * 78)


def cross_before_after(phi, a, b, rot=True, pts=PTS):
    res = []
    for x0 in pts:
        Gs = G_at(field_sum(phi, a, b, comp=False), x0)
        G1 = G_at(field_w1(phi, a), x0)
        G2 = G_at(field_w2(phi, b), x0)
        Rb = Gs - G1 - G2
        corr = corr_field(phi, a, b, rot=rot)
        Gs2 = G_at(field_sum(phi, a, b, comp=True, corr=corr), x0)
        Ra = Gs2 - G_at(field_w1(phi, a, comp=True), x0) - G_at(field_w2(phi, b, comp=True), x0)
        res.append((nrm(Rb), nrm(Ra)))
    return res


print()
print("[G22-1] 基线：平坦零、单波完成复核、phi = 0 与 a = b 的复现")
zf = ein_ext(zero_F, zero_H, (0.0, 1.10, 0.3, -0.2), 0.05, zero_N)
print("     平坦度规：|G| = %.2e" % nrm(zf))
print("     单波 1 完成：|G| = %.3e ｜ 单波 2 完成：|G| = %.3e"
      % (nrm(G_at(field_w1(0.0, 1e-2, comp=True), PTS[0])), nrm(G_at(field_w2(0.0, 1e-2, comp=True), PTS[0]))))
r0 = cross_before_after(0.0, 1e-2, 1e-2)
for kk, (rb, ra) in enumerate(r0):
    print("     点 %d：前 |R| = %.3e ｜ 后 |R| = %.3e ｜ 降幅 = %.1f 倍" % (kk + 1, rb, ra, rb / max(ra, 1e-30)))
print("     [累计 %.1f 秒]" % (time.time() - T_START))

print()
print("[G22-2] 振幅不等（phi = 0，闭式按 ab 缩放）")
for (a, b) in ((2e-2, 1e-2), (1e-2, 2e-2), (2e-2, 5e-3)):
    r = cross_before_after(0.0, a, b)
    line = "     a = %s, b = %s：" % (a, b)
    for rb, ra in r:
        line += " %.3e->%.3e (%.0f×) ｜" % (rb, ra, rb / max(ra, 1e-30))
    print(line)
print("     [累计 %.1f 秒]" % (time.time() - T_START))

print()
print("[G22-3] 旋转族（斜交角，a = b = 1e-2，闭式随构型旋转）")
for phi in (math.pi / 6, math.pi / 3, math.pi / 2):
    r = cross_before_after(phi, 1e-2, 1e-2, rot=True, pts=PTS[:2])
    line = "     phi = %3.0f 度：" % math.degrees(phi)
    for rb, ra in r:
        line += " %.3e->%.3e (%.0f×) ｜" % (rb, ra, rb / max(ra, 1e-30))
    print(line)
print("     [累计 %.1f 秒]" % (time.time() - T_START))

print()
print("[G22-4] 对照：未旋转闭式施加于旋转构型")
for phi in (math.pi / 6, math.pi / 2):
    r = cross_before_after(phi, 1e-2, 1e-2, rot=False, pts=PTS[:2])
    line = "     phi = %3.0f 度（未旋转闭式）：" % math.degrees(phi)
    for rb, ra in r:
        line += " %.3e->%.3e (%.2f×) ｜" % (rb, ra, rb / max(ra, 1e-30))
    print(line)
print("     [累计 %.1f 秒]" % (time.time() - T_START))

print()
print("[G22-5] 登记")
rows2 = [
    ("其一", "基线", "平坦零、单波完成与 phi = 0 复现"),
    ("其二", "振幅不等", "三档比值的降幅（ab 结构）"),
    ("其三", "旋转族", "30/60/90 度斜交角的降幅"),
    ("其四", "对照", "未旋转闭式对旋转构型无效"),
    ("其五", "余项", "三阶项、任意（非共面）斜交的一般形式、强场端"),
]
for a1, a2, a3 in rows2:
    print("     %-4s %-8s %s" % (a1, a2, a3))

print()
print("总运行时间 = %.1f 秒" % (time.time() - T_START))
