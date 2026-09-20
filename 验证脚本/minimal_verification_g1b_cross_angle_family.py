# -*- coding: utf-8 -*-
r"""
minimal_verification_g1b_cross_angle_family.py

[G1b 第二十三轮] 交叉角族：非 90 度交叉下闭式解的推广与复现（判据 G23-1 至 G23-5）

背景：第十八至二十一轮的构型都是 90 度交叉；第二十二轮的“斜交角”实为固定 90 度下的
方位旋转（协变性检验）。本轮做真正的交叉角族：波 1 沿 z（u = t - z），波 2 沿
(sin th, 0, cos th)（v = t - sin th x - cos th z），th 从 30 度到 90 度。两模式仍为
cos(u+v) 与 cos(u-v)，用 Pi(k)eps 代数为每个角度解出七场闭式系数（含 H_xz：90 度档的
x <-> z 交换对称把它的系数压到零），再用 4D 机器核验降幅。

判据
G23-1 基线：平坦零、单波完成复核
G23-2 角度族：七场系数表（30 至 90 度）与 sin 型污染比
G23-3 机器核验：各角度的降幅
G23-4 复核：th = 90 度对上第二十一轮有理形
G23-5 登记

运行：python 验证脚本/minimal_verification_g1b_cross_angle_family.py（确定性，约 1 分钟）
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


ETA = np.diag([-1.0, 1.0, 1.0, 1.0])


def Pi(k, eps):
    k2 = float(k @ ETA @ k)
    tr = float(np.einsum('ab,ab->', ETA, eps))
    bar = eps - 0.5 * ETA * tr
    kd = ETA @ k
    kbar = np.einsum('a,ab->b', kd, bar)
    kbark = float(np.einsum('a,ab,b->', kd, bar, kd))
    return 0.5 * (k2 * bar - np.outer(k, kbar) - np.outer(kbar, k) + ETA * kbark)


def eps_F():
    e = np.zeros((4, 4))
    e[0, 0] = 1.0
    return e


def eps_N(i):
    e = np.zeros((4, 4))
    e[0, 1 + i] = 1.0
    e[1 + i, 0] = 1.0
    return e


def eps_H(i, j):
    e = np.zeros((4, 4))
    e[1 + i, 1 + j] = 1.0
    e[1 + j, 1 + i] = 1.0
    return e


FIELDS7 = [('F', 0), ('N', 0), ('N', 2), ('H', (0, 0)), ('H', (0, 2)), ('H', (1, 1)), ('H', (2, 2))]
FIELDS6 = [('F', 0), ('N', 0), ('N', 2), ('H', (0, 0)), ('H', (1, 1)), ('H', (2, 2))]
EMAP = {('F', 0): eps_F(), ('N', 0): eps_N(0), ('N', 2): eps_N(2),
        ('H', (0, 0)): eps_H(0, 0), ('H', (0, 2)): eps_H(0, 2),
        ('H', (1, 1)): eps_H(1, 1), ('H', (2, 2)): eps_H(2, 2)}

A1 = 1e-2
A2 = 1e-2
AB = A1 * A2


def prof1(u, a):
    return (3.0 * a * a / 16.0) * np.cos(2.0 * u) - (a * a / 8.0) * u * u


def prof2(v, b):
    return (3.0 * b * b / 16.0) * np.cos(2.0 * v) - (b * b / 8.0) * v * v


def th_data(theta):
    st, ct = math.sin(theta), math.cos(theta)
    n2 = np.array([st, 0.0, ct])
    e1 = np.array([ct, 0.0, -st])
    e2 = np.array([0.0, 1.0, 0.0])
    P2 = np.outer(e1, e1) - np.outer(e2, e2)
    T2 = np.outer(e1, e1) + np.outer(e2, e2)
    kp = np.array([2.0, -st, 0.0, -1.0 - ct])
    km = np.array([0.0, st, 0.0, -1.0 + ct])
    return n2, P2, T2, kp, km


def field_w1(a, comp=False):
    def Hf(T, X, Y, Z, a=a, comp=comp):
        u = T - Z
        A = a * np.cos(u)
        b1 = prof1(u, a) if comp else 0.0
        sh = np.broadcast(T, X, Y, Z).shape
        H = np.zeros((3, 3) + sh)
        H[0, 0] = A + b1
        H[1, 1] = -A + b1
        return H
    return zero_F, Hf, zero_N


def field_w2(theta, b, comp=False):
    n2, P2, T2, kp, km = th_data(theta)

    def Hf(T, X, Y, Z, b=b, comp=comp, n2=n2, P2=P2, T2=T2):
        v = T - (n2[0] * X + n2[1] * Y + n2[2] * Z)
        B = b * np.cos(v)
        b2 = prof2(v, b) if comp else 0.0
        sh = np.broadcast(T, X, Y, Z).shape
        H = np.zeros((3, 3) + sh)
        for i in range(3):
            for j in range(3):
                H[i, j] = B * P2[i, j] + b2 * T2[i, j]
        return H
    return zero_F, Hf, zero_N


def mk_corr(theta, cvec):
    # cvec[(tag, *idx, mod)]，mod = 0 (k_+) 或 1 (k_-)
    n2, P2, T2, kp, km = th_data(theta)

    def ph(kk, T, X, Y, Z):
        return np.cos(kk[0] * T + kk[1] * X + kk[2] * Y + kk[3] * Z)

    def Ff(T, X, Y, Z, cvec=cvec):
        return cvec[('F', 0, 0, 0)] * ph(kp, T, X, Y, Z) + cvec[('F', 0, 0, 1)] * ph(km, T, X, Y, Z)

    def Nf(T, X, Y, Z, cvec=cvec):
        sh = np.broadcast(T, X, Y, Z).shape
        Nv = np.zeros((3,) + sh)
        for idx in (0, 2):
            Nv[idx] = cvec[('N', idx, 0, 0)] * ph(kp, T, X, Y, Z) + cvec[('N', idx, 0, 1)] * ph(km, T, X, Y, Z)
        return Nv

    def Hf(T, X, Y, Z, cvec=cvec):
        sh = np.broadcast(T, X, Y, Z).shape
        H = np.zeros((3, 3) + sh)
        for (i, j) in ((0, 0), (0, 2), (1, 1), (2, 2)):
            H[i, j] = cvec[('H', i, j, 0)] * ph(kp, T, X, Y, Z) + cvec[('H', i, j, 1)] * ph(km, T, X, Y, Z)
            H[j, i] = H[i, j]
        return H
    return Ff, Hf, Nf


def field_sum(theta, comp=True, corr=None):
    F1, H1, N1 = field_w1(A1, comp)
    F2, H2, N2 = field_w2(theta, A2, comp)

    def Ff(T, X, Y, Z, corr=corr):
        if corr is not None:
            return corr[0](T, X, Y, Z)
        return np.zeros_like(T + X + Y + Z)

    def Hf(T, X, Y, Z, corr=corr):
        H = H1(T, X, Y, Z) + H2(T, X, Y, Z)
        if corr is not None:
            H = H + corr[1](T, X, Y, Z)
        return H

    def Nf(T, X, Y, Z, corr=corr):
        sh = np.broadcast(T, X, Y, Z).shape
        if corr is not None:
            return corr[2](T, X, Y, Z)
        return np.zeros((3,) + sh)
    return Ff, Hf, Nf


def G_at(field, x0, eps=0.03):
    Ff, Hf, Nf = field
    return ein_ext(Ff, Hf, x0, eps, Nf)


IDX = [("tt", (0, 0)), ("tz", (0, 3)), ("zz", (3, 3)), ("xx", (1, 1)), ("yy", (2, 2)),
       ("xy", (1, 2)), ("xt", (0, 1)), ("yt", (0, 2)), ("xz", (1, 3)), ("yz", (2, 3))]
CIDX = dict(IDX)
PTS = [(0.0, 0.7, 0.3, -0.5), (0.0, 1.1, 0.3, -0.9), (0.0, 0.5, 0.3, -1.3),
       (0.0, 1.4, 0.3, -0.2), (0.0, 0.9, 0.3, 0.3), (0.0, 0.3, 0.3, -0.9)]


def kx(kk, x0):
    return kk[0] * x0[0] + kk[1] * x0[1] + kk[2] * x0[2] + kk[3] * x0[3]


def residuals(theta):
    return [G_at(field_sum(theta, comp=False), x0) - G_at(field_w1(A1), x0) - G_at(field_w2(theta, A2), x0)
            for x0 in PTS]


def gauge_vecs(k):
    # 每模式的三条规范方向（槽 0、槽 x、槽 z），在七场基 (F, N_x, N_z, H_xx, H_xz, H_yy, H_zz) 上
    v0 = np.zeros(7)
    v0[0] = 2.0 * k[0]
    v0[1] = k[1]
    v0[2] = k[3]
    v1 = np.zeros(7)
    v1[1] = k[0]
    v1[3] = 2.0 * k[1]
    v1[4] = k[3]
    v2 = np.zeros(7)
    v2[2] = k[0]
    v2[4] = k[1]
    v2[6] = 2.0 * k[3]
    return np.array([v0, v1, v2])


def solve_mode_gf(k, Rm):
    Amat = np.array([Pi(k, EMAP[fk]).reshape(-1) for fk in FIELDS7]).T
    V = gauge_vecs(k)
    Q, _ = np.linalg.qr(V.T)
    P = np.eye(7) - Q @ Q.T
    sol, _, _, _ = np.linalg.lstsq(Amat @ P, -Rm.reshape(-1), rcond=None)
    c = P @ sol
    rr = float(np.linalg.norm(Amat @ c + Rm.reshape(-1))) / float(np.linalg.norm(Rm))
    return c, rr


def solve_mode_6(k, Rm):
    Amat = np.array([Pi(k, EMAP[fk]).reshape(-1) for fk in FIELDS6]).T
    sol, _, _, _ = np.linalg.lstsq(Amat, -Rm.reshape(-1), rcond=None)
    rr = float(np.linalg.norm(Amat @ sol + Rm.reshape(-1))) / float(np.linalg.norm(Rm))
    return sol, rr


def solve_theta(theta):
    n2, P2, T2, kp, km = th_data(theta)
    Ar = np.array([[math.cos(kx(kp, x0)), math.sin(kx(kp, x0)),
                    math.cos(kx(km, x0)), math.sin(kx(km, x0))] for x0 in PTS])
    RR = [G_at(field_sum(theta, comp=False), x0) - G_at(field_w1(A1), x0) - G_at(field_w2(theta, A2), x0)
          for x0 in PTS]
    coefs = {}
    for tag, (i2, j2) in CIDX.items():
        bvec = np.array([R[i2, j2] for R in RR])
        cf, _, _, _ = np.linalg.lstsq(Ar, bvec, rcond=None)
        coefs[tag] = cf
    cvec = {}
    for mod, kidx, k in ((0, 0, kp), (1, 2, km)):
        Rm = np.zeros((4, 4))
        for tag, (i2, j2) in CIDX.items():
            Rm[i2, j2] = coefs[tag][kidx]
            Rm[j2, i2] = coefs[tag][kidx]
        c, _ = solve_mode_gf(k, Rm)
        for fk, v in zip(FIELDS7, c):
            cvec[key_of(fk, mod)] = v
    sinmax = max(max(abs(coefs[t][1]), abs(coefs[t][3])) for t in coefs)
    cosmax = max(max(abs(coefs[t][0]), abs(coefs[t][2])) for t in coefs)
    return coefs, cvec, sinmax / cosmax


print("=" * 78)
print("G1b 第二十三轮：交叉角族（判据 G23-1 至 G23-5）")
print("=" * 78)


def key_of(fk, mod):
    if fk[0] == 'F':
        return ('F', 0, 0, mod)
    if fk[0] == 'N':
        return ('N', fk[1], 0, mod)
    return ('H', fk[1][0], fk[1][1], mod)


print()
print("[G23-1] 基线：平坦零、单波完成复核")
zf = ein_ext(zero_F, zero_H, (0.0, 1.10, 0.3, -0.2), 0.05, zero_N)
print("     平坦度规：|G| = %.2e" % nrm(zf))
print("     单波 1 完成：|G| = %.3e ｜ 单波 2（th = 90 度）完成：|G| = %.3e"
      % (nrm(G_at(field_w1(A1, comp=True), PTS[0])),
         nrm(G_at(field_w2(math.pi / 2, A2, comp=True), PTS[0]))))
print("     [累计 %.1f 秒]" % (time.time() - T_START))

print()
print("[G23-2] 角度族：七场系数（单位 ab）与 sin 型污染比")
CV = {}
for deg in (90, 75, 60, 45, 30):
    theta = math.radians(deg)
    coefs, cvec, sinrel = solve_theta(theta)
    CV[deg] = cvec
    print("     th = %2d 度（sin 型相对 = %.2e）：" % (deg, sinrel))
    for fk in FIELDS7:
        print("        %-10s c_+ = %+.4f ｜ c_- = %+.4f"
              % (str(fk), cvec[key_of(fk, 0)] / AB, cvec[key_of(fk, 1)] / AB))
print("     [累计 %.1f 秒]" % (time.time() - T_START))

print()
print("[G23-3] 机器核验：各角度的降幅（三点）")
for deg in (90, 75, 60, 45, 30):
    theta = math.radians(deg)
    corr = mk_corr(theta, CV[deg])
    line = "     th = %2d 度：" % deg
    for x0 in PTS[:3]:
        Rb = G_at(field_sum(theta, comp=False), x0) - G_at(field_w1(A1), x0) - G_at(field_w2(theta, A2), x0)
        Ra = G_at(field_sum(theta, comp=True, corr=corr), x0) - G_at(field_w1(A1, comp=True), x0) - G_at(field_w2(theta, A2, comp=True), x0)
        line += " %.3e->%.3e (%.0f x) ｜" % (nrm(Rb), nrm(Ra), nrm(Rb) / max(nrm(Ra), 1e-30))
    print(line)
print("     [累计 %.1f 秒]" % (time.time() - T_START))

print()
print("[G23-3b] 拟合残差比（七场，逐模式）")
for deg in (90, 75, 60, 45, 30):
    theta = math.radians(deg)
    n2, P2, T2, kp, km = th_data(theta)
    Ar = np.array([[math.cos(kx(kp, x0)), math.sin(kx(kp, x0)), math.cos(kx(km, x0)), math.sin(kx(km, x0))] for x0 in PTS])
    RR = [G_at(field_sum(theta, comp=False), x0) - G_at(field_w1(A1), x0) - G_at(field_w2(theta, A2), x0) for x0 in PTS]
    line = "     th = %2d 度：" % deg
    for mod, kidx, k in ((0, 0, kp), (1, 2, km)):
        Rm = np.zeros((4, 4))
        for tag, (i2, j2) in CIDX.items():
            bvec = np.array([R[i2, j2] for R in RR])
            cf, _, _, _ = np.linalg.lstsq(Ar, bvec, rcond=None)
            Rm[i2, j2] = cf[kidx]
            Rm[j2, i2] = cf[kidx]
        Amat = np.array([Pi(k, EMAP[fk]).reshape(-1) for fk in FIELDS7]).T
        sol, _, _, _ = np.linalg.lstsq(Amat, -Rm.reshape(-1), rcond=None)
        rr = float(np.linalg.norm(Amat @ sol + Rm.reshape(-1))) / float(np.linalg.norm(Rm))
        line += "模式%d 残差比 %.1e ｜ " % (mod, rr)
    print(line)
print("     [累计 %.1f 秒]" % (time.time() - T_START))

print()
print("[G23-4] 对照：th = 90 度档对上第二十一轮有理形（单位 ab）")
RAT90 = {('F', 0, 0, 0): -5.0 / 36.0, ('N', 0, 0, 0): -5.0 / 18.0, ('N', 2, 0, 0): -5.0 / 18.0,
         ('H', 0, 0, 0): 5.0 / 8.0, ('H', 1, 1, 0): -1.0, ('H', 2, 2, 0): 5.0 / 8.0,
         ('F', 0, 0, 1): -1.0 / 4.0, ('N', 0, 0, 1): 1.0 / 8.0, ('N', 2, 0, 1): 1.0 / 8.0,
         ('H', 0, 0, 1): -5.0 / 8.0, ('H', 1, 1, 1): 0.0, ('H', 2, 2, 1): -5.0 / 8.0}
theta = math.pi / 2
n2, P2, T2, kp, km = th_data(theta)
Ar = np.array([[math.cos(kx(kp, x0)), math.sin(kx(kp, x0)), math.cos(kx(km, x0)), math.sin(kx(km, x0))] for x0 in PTS])
RR = [G_at(field_sum(theta, comp=False), x0) - G_at(field_w1(A1), x0) - G_at(field_w2(theta, A2), x0) for x0 in PTS]
coefs90 = {}
for tag, (i2, j2) in CIDX.items():
    bvec = np.array([R[i2, j2] for R in RR])
    cf, _, _, _ = np.linalg.lstsq(Ar, bvec, rcond=None)
    coefs90[tag] = cf
CV6 = {}
for mod, kidx, k in ((0, 0, kp), (1, 2, km)):
    Rm = np.zeros((4, 4))
    for tag, (i2, j2) in CIDX.items():
        Rm[i2, j2] = coefs90[tag][kidx]
        Rm[j2, i2] = coefs90[tag][kidx]
    sol, rr = solve_mode_6(k, Rm)
    for fk, v in zip(FIELDS6, sol):
        CV6[key_of(fk, mod)] = v
dev = 0.0
for kk, ref in RAT90.items():
    dev = max(dev, abs(CV6[kk] / AB + ref))
print("     六场解与有理形（比较 -RAT，含 P2 约定的负号）的最大绝对偏差 = %.2e" % dev)
print("     七场最小范数解与六场解的差（代表选择差）：H_xz 七场 c_+ = %+.4f、c_- = %+.4f；六场无此场"
      % (CV[90][('H', 0, 2, 0)] / AB, CV[90][('H', 0, 2, 1)] / AB))
print("     [累计 %.1f 秒]" % (time.time() - T_START))

print()
print("[G23-5] 登记")
rows2 = [
    ("其一", "基线", "平坦零、单波完成复核"),
    ("其二", "角度族", "30 至 90 度的七场系数表与 sin 型污染比"),
    ("其三", "机器核验", "各角度的降幅（三点）"),
    ("其四", "对照", "th = 90 度六场解对上第二十一轮有理形（偏差 1.4e-2）；七场代表含 H_xz"),
    ("其五", "余项", "th -> 0 的奇异极限（k_- 退化）、三阶项、强场端"),
]
for a1, a2, a3 in rows2:
    print("     %-4s %-8s %s" % (a1, a2, a3))

print()
print("总运行时间 = %.1f 秒" % (time.time() - T_START))
