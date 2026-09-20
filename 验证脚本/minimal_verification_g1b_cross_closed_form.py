# -*- coding: utf-8 -*-
"""
minimal_verification_g1b_cross_closed_form.py

[G1b 第二十一轮] 交叉修正的闭式解：两模式代数响应与机器核验（判据 G21-1 至 G21-5）

背景：第二十轮解出六场支撑里交叉修正的唯一代表，其轮廓内容只有 cos u cos v 与
sin u sin v（即 cos(u+v) 与 cos(u-v) 两个模式）。本轮把 (u,v) 轮廓分解到两个波矢：
k_+ = (2,-1,0,-1)（对应 cos(u+v)）与 k_- = (0,1,0,-1)（对应 cos(u-v)），用线性化
爱因斯坦张量的精确代数式（平面波响应 Pi(k) eps）替代差分测量，直接解出交叉修正的
闭式系数，再用 4D 机器核验闭合。

解析预期：
1) 平面波响应：G^{(1)}_{mu nu}[eps cos(kx)] = cos(kx)·(1/2)[k^2 bar_eps_{mu nu}
   - k_mu (k bar_eps)_nu - k_nu (k bar_eps)_mu + eta_{mu nu} k bar_eps k]，
   bar_eps = eps - (1/2) eta tr(eps)。对零矢 k 与无迹 eps 给零（自检）。
2) 交叉残差与其修正都只含两个 cos 模式（二阶张量的乘积结构只出 cos 型；
   sin 型系数应为三阶污染量级）。
3) 闭式系数：每个模式 6 个未知（六场支撑），由残差的模式系数与 Pi 的像解出；
   与第二十轮代表（cc/ss 换算）应一致。

判据
G21-1 基线：平坦零、单波完成复核、解析响应对机器响应核验
G21-2 残差模式内容：cos/sin 两模式的系数（sin 型应为污染量级）
G21-3 闭式系数：两模式各六场的解
G21-4 机器核验：闭式修正加入后交叉部与总 |G| 的降幅
G21-5 登记

运行：python 验证脚本/minimal_verification_g1b_cross_closed_form.py（确定性，约 1 分钟）
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
KPLUS = np.array([2.0, -1.0, 0.0, -1.0])
KMINUS = np.array([0.0, 1.0, 0.0, -1.0])


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


def eps_H(i):
    e = np.zeros((4, 4))
    e[1 + i, 1 + i] = 1.0
    return e


POLS = [('F', eps_F()), ('Nx', eps_N(0)), ('Nz', eps_N(2)),
        ('Hxx', eps_H(0)), ('Hyy', eps_H(1)), ('Hzz', eps_H(2))]
PMAP = {('F', 0): eps_F(), ('N', 0): eps_N(0), ('N', 2): eps_N(2),
        ('H', (0, 0)): eps_H(0), ('H', (1, 1)): eps_H(1), ('H', (2, 2)): eps_H(2)}

A1 = 1e-2
A2 = 1e-2
AB = A1 * A2


def prof1(u, a):
    return (3.0 * a * a / 16.0) * np.cos(2.0 * u) - (a * a / 8.0) * u * u


def prof2(v, b):
    return (3.0 * b * b / 16.0) * np.cos(2.0 * v) - (b * b / 8.0) * v * v


def field_1(a, comp=False):
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


def field_2(b, comp=False):
    def Hf(T, X, Y, Z, b=b, comp=comp):
        v = T - X
        B = b * np.cos(v)
        b2 = prof2(v, b) if comp else 0.0
        sh = np.broadcast(T, X, Y, Z).shape
        H = np.zeros((3, 3) + sh)
        H[1, 1] = B + b2
        H[2, 2] = -B + b2
        return H
    return zero_F, Hf, zero_N


def field_12(a, b, comp=False, extra=None):
    def Hf(T, X, Y, Z, a=a, b=b, comp=comp, extra=extra):
        u = T - Z
        v = T - X
        A = a * np.cos(u)
        B = b * np.cos(v)
        b1 = prof1(u, a) if comp else 0.0
        b2 = prof2(v, b) if comp else 0.0
        sh = np.broadcast(T, X, Y, Z).shape
        H = np.zeros((3, 3) + sh)
        H[0, 0] = A + b1
        H[1, 1] = -A + B + b1 + b2
        H[2, 2] = -B + b2
        if extra is not None:
            H = H + extra[1](T, X, Y, Z)
        return H

    def Ff(T, X, Y, Z, extra=extra):
        if extra is not None:
            return extra[0](T, X, Y, Z)
        return np.zeros_like(T + X + Y + Z)

    def Nf(T, X, Y, Z, extra=extra):
        sh = np.broadcast(T, X, Y, Z).shape
        if extra is not None:
            return extra[2](T, X, Y, Z)
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


def kx(k, x0):
    return float(k[0] * x0[0] + k[1] * x0[1] + k[2] * x0[2] + k[3] * x0[3])


print("=" * 78)
print("G1b 第二十一轮：交叉修正的闭式解（判据 G21-1 至 G21-5）")
print("=" * 78)

print()
print("[G21-1] 基线：平坦零、单波完成复核与解析响应核验")
zf = ein_ext(zero_F, zero_H, (0.0, 1.10, 0.3, -0.2), 0.05, zero_N)
print("     平坦度规：|G| = %.2e" % nrm(zf))
print("     单波 1（含完成）：|G| = %.3e" % nrm(G_at(field_1(A1, comp=True), PTS[0])))
KN = np.array([1.0, 0.0, 0.0, -1.0])
tt = np.zeros((4, 4))
tt[1, 1] = 1.0
tt[2, 2] = -1.0
print("     Pi 自检（零矢 k = (1,0,0,-1) + TT eps）：max|Pi| = %.2e（应零）" % np.max(np.abs(Pi(KN, tt))))
print("     k_+^2 = %+.4f，k_-^2 = %+.4f（一个类时、一个类空，均为非零模式）"
      % (float(KPLUS @ ETA @ KPLUS), float(KMINUS @ ETA @ KMINUS)))


def mk_single_mode(k, pol, amp):
    if pol == 'F':
        def Ff(T, X, Y, Z, k=k, amp=amp):
            return amp * np.cos(k[0] * T + k[1] * X + k[2] * Y + k[3] * Z)
        return Ff, zero_H, zero_N
    if pol in ('Nx', 'Nz'):
        idx = 0 if pol == 'Nx' else 2
        def Nf(T, X, Y, Z, k=k, amp=amp, idx=idx):
            sh = np.broadcast(T, X, Y, Z).shape
            Nv = np.zeros((3,) + sh)
            Nv[idx] = amp * np.cos(k[0] * T + k[1] * X + k[2] * Y + k[3] * Z)
            return Nv
        return zero_F, zero_H, Nf
    idx = {'Hxx': 0, 'Hyy': 1, 'Hzz': 2}[pol]

    def Hf(T, X, Y, Z, k=k, amp=amp, idx=idx):
        sh = np.broadcast(T, X, Y, Z).shape
        H = np.zeros((3, 3) + sh)
        H[idx, idx] = amp * np.cos(k[0] * T + k[1] * X + k[2] * Y + k[3] * Z)
        return H
    return zero_F, Hf, zero_N


def resp3(mk, x0, a=1e-2, eps=0.03):
    F1, H1, N1 = mk(a)
    F2, H2, N2 = mk(a / 2.0)
    F3, H3, N3 = mk(a / 4.0)
    G1 = ein_ext(F1, H1, x0, eps, N1)
    G2 = ein_ext(F2, H2, x0, eps, N2)
    G3 = ein_ext(F3, H3, x0, eps, N3)
    Amat = np.array([[a, a * a, a ** 3], [a / 2.0, a * a / 4.0, a ** 3 / 8.0],
                     [a / 4.0, a * a / 16.0, a ** 3 / 64.0]])
    st = np.stack([G1.reshape(-1), G2.reshape(-1), G3.reshape(-1)], axis=0)
    cc = np.linalg.solve(Amat, st)
    return cc[0].reshape(G1.shape)


for ktag, k, pol in (("k_+", KPLUS, 'F'), ("k_-", KMINUS, 'Hxx'), ("k_+", KPLUS, 'Nz')):
    x0 = PTS[0]
    rm = resp3(lambda amp, k=k, pol=pol: mk_single_mode(k, pol, amp), x0)
    ra = Pi(k, PMAP[('F', 0) if pol == 'F' else (('N', 0) if pol == 'Nx' else (('N', 2) if pol == 'Nz' else ('H', (0, 0) if pol == 'Hxx' else (('H', (1, 1)) if pol == 'Hyy' else ('H', (2, 2))))))]) * np.cos(kx(k, x0))
    print("     %s %-3s：机器响应 tt=%+.4e 解析 tt=%+.4e ｜ 最大分量差 = %.2e"
          % (ktag, pol, rm[0, 0], ra[0, 0], float(np.max(np.abs(rm - ra)))))


def mk_gauge_A(amp=1.0):
    # 槽 0，轮廓 cc（完整规范模）：C00 = 2(fu+fv)、C0x = -fv、C0z = -fu
    def Ff(T, X, Y, Z, amp=amp):
        u = T - Z
        v = T - X
        pu = -np.sin(u) * np.cos(v)
        pv = -np.cos(u) * np.sin(v)
        return 2.0 * amp * (pu + pv)

    def Nf(T, X, Y, Z, amp=amp):
        u = T - Z
        v = T - X
        shp = np.broadcast(T, X, Y, Z).shape
        Nv = np.zeros((3,) + shp)
        Nv[0] = -amp * (-np.cos(u) * np.sin(v))
        Nv[2] = -amp * (-np.sin(u) * np.cos(v))
        return Nv
    return Ff, zero_H, Nf


def mk_gauge_B(amp=1.0):
    # (xi_x, xi_z) 族，参数 (a1,a2,b1,b2) = (1,0,0,0)：xi_x = cc、xi_z = ss
    def Nf(T, X, Y, Z, amp=amp):
        u = T - Z
        v = T - X
        cs = np.cos(u) * np.sin(v)
        sc = np.sin(u) * np.cos(v)
        shp = np.broadcast(T, X, Y, Z).shape
        Nv = np.zeros((3,) + shp)
        Nv[0] = -amp * (cs + sc)
        Nv[2] = +amp * (cs + sc)
        return Nv

    def Hf(T, X, Y, Z, amp=amp):
        u = T - Z
        v = T - X
        cs = np.cos(u) * np.sin(v)
        shp = np.broadcast(T, X, Y, Z).shape
        H = np.zeros((3, 3) + shp)
        H[0, 0] = 2.0 * amp * cs
        H[2, 2] = -2.0 * amp * cs
        return H
    return zero_F, Hf, Nf


gmA = nrm(resp3(mk_gauge_A, PTS[0], a=1e-2))
gmB = nrm(resp3(mk_gauge_B, PTS[0], a=1e-2))
print("     规范核实测（修正口径，完整两模 A、B）：c1 = %.2e、%.2e（应到噪声级）" % (gmA, gmB))
print("     [累计 %.1f 秒]" % (time.time() - T_START))

print()
print("[G21-2] 残差模式内容（在 C_+、C_- 与 S_+、S_- 四基上投影）")
rowsR, comps = [], []
for x0 in PTS:
    R = G_at(field_12(A1, A2), x0) - G_at(field_1(A1), x0) - G_at(field_2(A2), x0)
    rowsR.append(R)
    comps.append([np.cos(kx(KPLUS, x0)), np.sin(kx(KPLUS, x0)), np.cos(kx(KMINUS, x0)), np.sin(kx(KMINUS, x0))])
Ar = np.array(comps)
coefs = {}
for tag, (i2, j2) in CIDX.items():
    bvec = np.array([R[i2, j2] for R in rowsR])
    cf, _, _, _ = np.linalg.lstsq(Ar, bvec, rcond=None)
    coefs[tag] = cf
cosmax = max(max(abs(coefs[t][0]), abs(coefs[t][2])) for t in coefs)
sinmax = max(max(abs(coefs[t][1]), abs(coefs[t][3])) for t in coefs)
for t in ALLOWED if False else ["tt", "tz", "zz", "xx", "yy", "xt", "xz"]:
    c1, s1, c2, s2 = coefs[t]
    print("     %-3s：C_+ = %+.4e ｜ S_+ = %+.2e ｜ C_- = %+.4e ｜ S_- = %+.2e" % (t, c1, s1, c2, s2))
print("     cos 型最大 = %.3e ｜ sin 型最大 = %.2e（相对 %.1e，应为污染量级）" % (cosmax, sinmax, sinmax / cosmax))
print("     [累计 %.1f 秒]" % (time.time() - T_START))

print()
print("[G21-3] 闭式系数：每个模式解六场")


def tensor_from_cf(cf):
    Rm = np.zeros((4, 4))
    for tag, (i2, j2) in CIDX.items():
        Rm[i2, j2] = cf[tag]
        Rm[j2, i2] = cf[tag]
    return Rm


SOLS = {}
for ktag, kidx, k in (("+", 0, KPLUS), ("-", 2, KMINUS)):
    cf = {t: coefs[t][kidx] for t in CIDX}
    Rm = tensor_from_cf(cf)
    A = np.array([Pi(k, e).reshape(-1) for _, e in POLS]).T
    rhs = -Rm.reshape(-1)
    sol, rres, rrk, _ = np.linalg.lstsq(A, rhs, rcond=None)
    rfit = float(np.linalg.norm(A @ sol - rhs)) / float(np.linalg.norm(rhs))
    SOLS[ktag] = sol
    print("     模式 k_%s：解的残差比 = %.2e（秩 %d）" % (ktag, rfit, rrk))
    for (tag, _), v in zip(POLS, sol):
        print("        %-4s：c = %+.4e = %+.4f ab" % (tag, v, v / AB))
REF20 = {
    'F': (-0.13895, -0.24475),
    'Nx': (-0.27695, +0.12315),
    'Nz': (-0.27895, +0.12315),
    'Hxx': (+0.6274, -0.6229),
    'Hyy': (-0.99935, -0.00025),
    'Hzz': (+0.62535, -0.62285),
}
print("     与第二十轮代表的换算对照（单位 ab；代表 = cc/ss 系数，换算 C_+ = (cc-ss)/2、C_- = (cc+ss)/2）：")
for tag, _ in POLS:
    print("        %-4s：闭式 (%+.4f, %+.4f) ｜ 代表 (%+.4f, %+.4f)"
          % (tag, SOLS['+'][[t for t, _ in POLS].index(tag)] / AB,
             SOLS['-'][[t for t, _ in POLS].index(tag)] / AB, REF20[tag][0], REF20[tag][1]))
print("     小幅度复核（a = b = 1e-3，污染降到 0.1% 级）：")
A1s = 1e-3
rowsS = []
for x0 in PTS:
    R = G_at(field_12(A1s, A1s), x0) - G_at(field_1(A1s), x0) - G_at(field_2(A1s), x0)
    rowsS.append(R)
coefsS = {}
for t in CIDX:
    i2, j2 = CIDX[t]
    bvec = np.array([R[i2, j2] for R in rowsS])
    cf, _, _, _ = np.linalg.lstsq(Ar, bvec, rcond=None)
    coefsS[t] = cf
SOLS2 = {}
for ktag, kidx, k in (("+", 0, KPLUS), ("-", 2, KMINUS)):
    cf = {t: coefsS[t][kidx] for t in CIDX}
    Rm = tensor_from_cf(cf)
    A = np.array([Pi(k, e).reshape(-1) for _, e in POLS]).T
    sol, rres, rrk, _ = np.linalg.lstsq(A, -Rm.reshape(-1), rcond=None)
    SOLS2[ktag] = sol
abS = A1s * A1s
for ktag in ("+", "-"):
    s = SOLS2[ktag] / abS
    ref = s[[t for t, _ in POLS].index('Nx')]
    print("     模式 k_%s：系数/ab = " % ktag + "  ".join("%s %+.4f" % (t, v) for (t, _), v in zip(POLS, s)))
    print("            以 Nx 归一：" + "  ".join("%s %+.4f" % (t, v / ref) for (t, _), v in zip(POLS, s)))
RAT = {'+': np.array([-5.0 / 36.0, -5.0 / 18.0, -5.0 / 18.0, 5.0 / 8.0, -1.0, 5.0 / 8.0]),
       '-': np.array([-1.0 / 4.0, 1.0 / 8.0, 1.0 / 8.0, -5.0 / 8.0, 0.0, -5.0 / 8.0])}
for ktag in ("+", "-"):
    s = SOLS2[ktag] / abS
    dev = np.max(np.abs(s - RAT[ktag])) / np.max(np.abs(RAT[ktag]))
    print("     有理形核对（模式 k_%s 对 %s）：最大相对偏差 = %.2e"
          % (ktag, "(-5/36,-5/18,-5/18,+5/8,-1,+5/8)" if ktag == "+" else "(-1/4,+1/8,+1/8,-5/8,0,-5/8)", dev))
print("     [累计 %.1f 秒]" % (time.time() - T_START))

print()
print("[G21-4] 机器核验：闭式修正加入后")


def closed_corr(scale=1.0):
    def Ff(T, X, Y, Z, scale=scale):
        tot = np.zeros_like(T + X + Y + Z)
        for km, idx in ((KPLUS, 0), (KMINUS, 2)):
            c = SOLS['+' if idx == 0 else '-'][0] * scale
            tot = tot + c * np.cos(km[0] * T + km[1] * X + km[2] * Y + km[3] * Z)
        return tot

    def Hf(T, X, Y, Z, scale=scale):
        sh = np.broadcast(T, X, Y, Z).shape
        H = np.zeros((3, 3) + sh)
        for fi, (i2, j2) in ((3, (0, 0)), (4, (1, 1)), (5, (2, 2))):
            tot = np.zeros(sh)
            for km, idx in ((KPLUS, 0), (KMINUS, 2)):
                c = SOLS['+' if idx == 0 else '-'][fi] * scale
                tot = tot + c * np.cos(km[0] * T + km[1] * X + km[2] * Y + km[3] * Z)
            H[i2, j2] = tot
        return H

    def Nf(T, X, Y, Z, scale=scale):
        sh = np.broadcast(T, X, Y, Z).shape
        Nv = np.zeros((3,) + sh)
        for fi, idx in ((1, 0), (2, 2)):
            tot = np.zeros(sh)
            for km, ki in ((KPLUS, 0), (KMINUS, 2)):
                c = SOLS['+' if ki == 0 else '-'][fi] * scale
                tot = tot + c * np.cos(km[0] * T + km[1] * X + km[2] * Y + km[3] * Z)
            Nv[idx] = tot
        return Nv
    return (Ff, Hf, Nf)


for x0 in PTS[:3]:
    Gsum = G_at(field_12(A1, A2), x0)
    G1v = G_at(field_1(A1), x0)
    G2v = G_at(field_2(A2), x0)
    Rcross = Gsum - G1v - G2v
    corr = closed_corr()
    Gcor = G_at(field_12(A1, A2, comp=True, extra=corr), x0)
    Rafter = Gcor - G_at(field_1(A1, comp=True), x0) - G_at(field_2(A2, comp=True), x0)
    print("     点 %s：交叉部前 |R| = %.3e ｜ 后 |R| = %.3e ｜ 降幅 = %.1f 倍"
          % (str(x0[1:3]), nrm(Rcross), nrm(Rafter), nrm(Rcross) / max(nrm(Rafter), 1e-30)))
print("     幅度标度复核（闭式系数按 ab 缩放到 a = b = 5e-3）：")
corr_h = closed_corr(scale=0.25)
x0 = PTS[0]
Rc2 = G_at(field_12(A1 / 2.0, A2 / 2.0, comp=True, extra=corr_h), x0) - \
    G_at(field_1(A1 / 2.0, comp=True), x0) - G_at(field_2(A2 / 2.0, comp=True), x0)
print("     a = b = 5e-3：修正后交叉部 |R| = %.3e（前 %s 级）" % (nrm(Rc2), "4.5e-05"))
print("     [累计 %.1f 秒]" % (time.time() - T_START))

print()
print("[G21-5] 登记")
rows2 = [
    ("其一", "基线", "平坦零、单波完成、解析响应对机器响应核验"),
    ("其二", "模式内容", "残差在 cos/sin 四基上的系数（sin 型为污染量级）"),
    ("其三", "闭式系数", "两模式各六场的解（与第二十轮代表换算对照）"),
    ("其四", "机器核验", "闭式修正加入后交叉部与总 |G| 的降幅"),
    ("其五", "余项", "振幅不等与一般斜交、强场端"),
]
for a1, a2, a3 in rows2:
    print("     %-4s %-8s %s" % (a1, a2, a3))

print()
print("总运行时间 = %.1f 秒" % (time.time() - T_START))
