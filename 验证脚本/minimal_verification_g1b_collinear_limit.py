# -*- coding: utf-8 -*-
r"""
minimal_verification_g1b_collinear_limit.py

[G1b 第二十四轮] 共线极限锚点：解析交叉完成式、合成恒等与四阶地板（判据 G24-1 至 G24-6）

背景：第二十一至二十三轮把交叉修正的闭式解推广到真交叉角族（30 至 90 度），两处登记未做：
其一，theta -> 0 的端点（k_- -> 0，模式基退化）；其二，单波完成式地板随振幅的幂次。本轮补
上另一端锚点：两束完全共线（同向 u = t - z、同偏振 P = xx - yy）时，复合场等于振幅 (a + b)
的单波完成式，交叉完成式有解析闭式

    B_cross = B_{a+b} - B_a - B_b = (3ab/8) cos 2u - (ab/4) u^2

它同时是角度族 theta = 0 的极限读数。器械沿用第二十三轮的 4D 机器核验（9^4 补丁、Richardson
半步外推）与 Pi(k) 代数（本轮脚本自带一份）。

判据
G24-1 基线：平坦零、裸残差二次幂级、单波完成地板四次幂定阶
G24-2 解析闭式：可加性恒等式与二阶微分核对（纯数学）
G24-3 机器核验：合成恒等（复合场与振幅 (a+b) 的单波完成式逐点同值）
G24-4 降幅与阶：交叉残差 ab 级降到四次幂地板；振幅标度（裸 x4、加交叉 x16）
G24-5 小角扫描与端点退化：theta 0 至 90 度连续性、模式基秩亏、Pi(0) 不可解、族内器械对照
G24-6 登记

运行：python 验证脚本/minimal_verification_g1b_collinear_limit.py（确定性，约 1 分钟）
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


def key_of(fk, mod):
    if fk[0] == 'F':
        return ('F', 0, 0, mod)
    if fk[0] == 'N':
        return ('N', fk[1], 0, mod)
    return ('H', fk[1][0], fk[1][1], mod)


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




# ================= 共线分支（本轮新增） =================
a = A1
b = A2


def prof_cross(u, aa, bb):
    return (3.0 * aa * bb / 8.0) * np.cos(2.0 * u) - (aa * bb / 4.0) * u * u


def field_coll(aa, bb, comp=False, cross=False):
    def Hf(T, X, Y, Z, aa=aa, bb=bb, comp=comp, cross=cross):
        u = T - Z
        A = (aa + bb) * np.cos(u)
        Bd = 0.0
        if comp:
            Bd = prof1(u, aa) + prof1(u, bb)
        if cross:
            Bd = Bd + prof_cross(u, aa, bb)
        sh = np.broadcast(T, X, Y, Z).shape
        H = np.zeros((3, 3) + sh)
        H[0, 0] = A + Bd
        H[1, 1] = -A + Bd
        return H
    return zero_F, Hf, zero_N


print("=" * 78)
print("G1b 第二十四轮：共线极限锚点（判据 G24-1 至 G24-6）")
print("=" * 78)

print()
print("[G24-1] 基线：平坦零、裸残差二次幂、完成地板四次幂")
print("     平坦度规：|G| = %.2e（期望 0）" % nrm(ein_ext(zero_F, zero_H, PTS[0], 0.05, zero_N)))
raws = [nrm(G_at(field_w1(a, False), x0)) for x0 in PTS[:3]]
print("     单波裸残差（a = 1e-2，三点）：%s（期望二次幂 1e-5 至 1e-3）" % " ".join("%.3e" % v for v in raws))
floors = [nrm(G_at(field_w1(amp, True), PTS[0])) for amp in (a, 2.0 * a, 4.0 * a)]
print("     单波完成地板（振幅 1e-2 / 2e-2 / 4e-2）：%.3e %.3e %.3e" % tuple(floors))
print("     地板比值：%.1f、%.1f（期望 16、256，即四次幂）" % (floors[1] / floors[0], floors[2] / floors[0]))

print()
print("[G24-2] 解析闭式：可加性恒等式与二阶微分核对（纯数学）")
ug = np.linspace(-3.0, 3.0, 2401)
d_add = float(np.max(np.abs(prof1(ug, a) + prof1(ug, b) + prof_cross(ug, a, b) - prof1(ug, a + b))))
print("     可加性 max |B_a + B_b + B_cross - B_(a+b)| = %.2e（期望机器零）" % d_add)
hg = 1.0e-3
ug2 = ug[5:-5]
def d2(f):
    return (f(ug2 + hg) - 2.0 * f(ug2) + f(ug2 - hg)) / (hg * hg)
fA1 = lambda w: a * np.cos(w)
fA2 = lambda w: b * np.cos(w)
lhs = d2(lambda w: prof_cross(w, a, b))
rhs = d2(lambda w: fA1(w) * fA2(w)) - ((fA1(ug2 + hg) - fA1(ug2 - hg)) / (2 * hg)) * ((fA2(ug2 + hg) - fA2(ug2 - hg)) / (2 * hg))
print("     B_cross'' 对 (A1A2)'' - A1'A2'：max |差| = %.2e（期望差分误差级）" % float(np.max(np.abs(lhs - rhs))))
u0 = 1.0
fd2 = (prof_cross(u0 + hg, a, b) - 2.0 * prof_cross(u0, a, b) + prof_cross(u0 - hg, a, b)) / (hg * hg)
an2 = -a * b / 2.0 - 3.0 * a * b * math.cos(2.0 * u0) / 2.0
print("     解析值核对（u = 1）：B_cross'' = %.6e，闭式 -ab/2 - 3ab cos2u/2 = %.6e" % (fd2, an2))

print()
print("[G24-3] 合成恒等：复合场与振幅 (a+b) 单波完成式逐点同值")
mxg = 0.0
for x0 in PTS:
    mxg = max(mxg, nrm(G_at(field_coll(a, b, True, True), x0) - G_at(field_w1(a + b, True), x0)))
print("     六点 max |G[复合] - G[单波 (a+b)]| = %.2e（期望机器零）" % mxg)
Tc = np.linspace(-2.5, 2.5, 401)
Xc, Yc, Zc = 0.11 * np.ones_like(Tc), -0.23 * np.ones_like(Tc), 0.37 * np.ones_like(Tc)
Ha = field_coll(a, b, True, True)[1](Tc, Xc, Yc, Zc)
Hb = field_w1(a + b, True)[1](Tc, Xc, Yc, Zc)
print("     场值级 max |H[复合] - H[(a+b)]| = %.2e（期望机器零）" % float(np.max(np.abs(Ha - Hb))))
print("     u 依赖核对：同 u 两点（u = 0.5，t 与 z 各移 0.1）")
x1 = (0.0, 0.7, 0.3, -0.5)
x2 = (0.1, 0.7, 0.3, -0.4)
d1 = nrm(G_at(field_coll(a, b, False, False), x1) - G_at(field_w1(a, False), x1) - G_at(field_w1(b, False), x1))
d2 = nrm(G_at(field_coll(a, b, False, False), x2) - G_at(field_w1(a, False), x2) - G_at(field_w1(b, False), x2))
print("         裸交叉残差 %.6e 对 %.6e，差 %.1e（期望机器零，残差只依赖 u）" % (d1, d2, abs(d1 - d2)))

print()
print("[G24-4] 降幅与阶：交叉残差 ab 级 -> 四次幂地板")
for kk, x0 in enumerate(PTS):
    gwa = G_at(field_w1(a, False), x0)
    gwb = G_at(field_w1(b, False), x0)
    gsa = G_at(field_coll(a, b, False, False), x0)
    gca = G_at(field_w1(a, True), x0)
    gcb = G_at(field_w1(b, True), x0)
    gsc = G_at(field_coll(a, b, True, False), x0)
    gsx = G_at(field_coll(a, b, True, True), x0)
    gsm = G_at(field_w1(a + b, True), x0)
    Rb = nrm(gsa - gwa - gwb)
    Rc = nrm(gsc - gca - gcb)
    Ra = nrm(gsx - gca - gcb)
    fl = nrm(gsm)
    print("     点 %d：裸交叉 %.3e ｜ 逐波完成 %.3e ｜ 加交叉项 %.3e ｜ 降幅 %.0f 倍 ｜ Ra/地板 %.3f"
          % (kk + 1, Rb, Rc, Ra, Rb / Ra, Ra / fl))
print("     标度核对（a = b = s x 1e-2，点 1）：")
for s in (1.0, 2.0, 4.0):
    aa = s * a
    gwa = G_at(field_w1(aa, False), PTS[0])
    gsa = G_at(field_coll(aa, aa, False, False), PTS[0])
    gca = G_at(field_w1(aa, True), PTS[0])
    gsx = G_at(field_coll(aa, aa, True, True), PTS[0])
    print("        s = %.0f：裸交叉 %.3e ｜ 加交叉 %.3e" % (s, nrm(gsa - 2.0 * gwa), nrm(gsx - 2.0 * gca)))
print("     （期望：裸交叉每步 x4，加交叉每步 x16）")

print()
print("[G24-5] 小角扫描与端点退化")
print("     裸交叉残差随角（点 1 至 3）：")
for deg in (0.0, 2.0, 5.0, 10.0, 20.0, 45.0, 90.0):
    th = math.radians(deg)
    if deg == 0.0:
        rr = [nrm(G_at(field_coll(a, b, False, False), x0) - G_at(field_w1(a, False), x0) - G_at(field_w1(b, False), x0)) for x0 in PTS[:3]]
    else:
        rr = [nrm(G_at(field_sum(th, comp=False), x0) - G_at(field_w1(a), x0) - G_at(field_w2(th, b), x0)) for x0 in PTS[:3]]
    print("        theta = %4.1f 度：%.3e %.3e %.3e" % (deg, rr[0], rr[1], rr[2]))
print("     模式基条件数（六点）：")
for deg in (0.0, 2.0, 5.0, 10.0, 20.0, 90.0):
    th = math.radians(deg)
    n2, P2, T2, kp, km = th_data(th)
    Ar = np.array([[math.cos(kx(kp, x0)), math.sin(kx(kp, x0)), math.cos(kx(km, x0)), math.sin(kx(km, x0))] for x0 in PTS])
    rk = int(np.linalg.matrix_rank(Ar))
    if rk < 4:
        print("        theta = %4.1f 度：秩 %d（亏），cond 发散（k_- = 0 列退化）" % (deg, rk))
    else:
        print("        theta = %4.1f 度：秩 4，cond = %.1f（约 4.1/theta）" % (deg, float(np.linalg.cond(Ar))))
print("     端点退化核对（theta = 0）：")
p0 = max(nrm(Pi(np.zeros(4), EMAP[fk])) for fk in FIELDS7)
print("        max |Pi(0, eps)| = %.2e（期望 0）" % p0)
sol0, rr0 = solve_mode_gf(np.zeros(4), np.eye(4) * a * b)
print("        solve_mode_gf(k = 0, Rm ~ ab)：|c| = %.2e，残差比 = %.2f（期望 0 与 1.00）"
      % (float(np.linalg.norm(sol0)), rr0))
print("     族内器械对照（拟合修正的降幅）：")
for deg in (2.0, 5.0, 90.0):
    theta = math.radians(deg)
    coefs, cvec, sinrel = solve_theta(theta)
    corr = mk_corr(theta, cvec)
    line = "        theta = %4.1f 度：" % deg
    for x0 in PTS[:3]:
        Rbb = nrm(G_at(field_sum(theta, comp=False), x0) - G_at(field_w1(A1), x0) - G_at(field_w2(theta, A2), x0))
        Raa = nrm(G_at(field_sum(theta, comp=True, corr=corr), x0) - G_at(field_w1(A1, comp=True), x0) - G_at(field_w2(theta, A2, comp=True), x0))
        line += " %.2e->%.2e (%.0f x) ｜" % (Rbb, Raa, Rbb / max(Raa, 1e-30))
    print(line + " sin 污染 %.1e" % sinrel)

print()
print("[G24-6] 登记")
rows = [
    ("其一", "基线", "平坦零；裸残差二次幂；单波地板四次幂（比值 16、256）"),
    ("其二", "解析闭式", "B_cross = (3ab/8) cos 2u - (ab/4) u^2；可加性恒等与二阶核对"),
    ("其三", "合成恒等", "复合场与单波 (a+b) 完成式逐点同值（机器零）"),
    ("其四", "降幅", "交叉残差 ab 级降到四次幂地板；标度 4 与 16"),
    ("其五", "小角端点", "0 至 90 度连续；theta = 0 秩亏、Pi(0) 不可解，需解析锚点"),
]
for r1, r2, r3 in rows:
    print("     %-4s %-8s %s" % (r1, r2, r3))

print()
print("总运行时间 = %.1f 秒" % (time.time() - T_START))
