# -*- coding: utf-8 -*-
"""
minimal_verification_g1b_uv_class.py

[G1b 第十九轮] (u,v) 类响应分类与交叉修正的最小选择（判据 G19-1 至 G19-6）

背景：第十八轮证明两束交叉平面波的交叉项落在 (u,v) 修正类（u = t - z、v = t - x）的像里，
闭合到三阶污染量级，但四十八列（10 场乘 4 轮廓）的解不唯一、没有规范选择。本轮做三件：
其一，规范核的显式构造与核验（规范模 C = d xi + xi d，四个 xi 槽乘两个轮廓）；
其二，像的结构（40 列矩阵的奇异谱：物理维数与规范方向的个数）；
其三，支撑阶梯（对角、加 F 与 N、全类）与最小范数解的形状，给出交叉修正的规范选择与
结构提示。另附 u 类响应复核（把第十五轮的分类接到 (u,v) 类）。

解析预期：
1) 规范核：C_{mu nu} = d_mu xi_nu + d_nu xi_mu（xi 为 (u,v) 的函数）给零响应，
   读数应在幅度 Richardson 加 eps Richardson 后压到 1e-9 以下。
2) 像的结构：40 列里有 16 个规范组合（4 槽乘 4 轮廓），物理维数上限 24；
   奇异谱应在物理值（O(1) 至 O(10)）与规范方向（噪声级）之间出现断口。
3) 支撑阶梯：从对角类到全类逐级加槽（F、N_x、N_z、N_y、off-diagonal H），
   报告每级的拟合/校验闭合比，定位最少需要哪些槽。
4) u 类复核：u 依赖的迹扰动响应回到第十五轮读数（cos2u 给 -B'' (1,1,-1) 与 -B''/2）。

判据
G19-1 基线：平坦零、单波完成复核与 u 类响应复核
G19-2 规范核：八个规范模的响应上界
G19-3 像的结构：40 列奇异谱与断口
G19-4 支撑阶梯：四级子类的拟合与校验闭合比
G19-5 最小范数解的形状：主要系数与修正幅度
G19-6 登记

运行：python 验证脚本/minimal_verification_g1b_uv_class.py（确定性，约 1 分钟）
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


A1 = 1e-2
A2 = 1e-2


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


def field_12(a, b, comp=False):
    def Hf(T, X, Y, Z, a=a, b=b, comp=comp):
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
        return H
    return zero_F, Hf, zero_N


def G_at(field, x0, eps=0.03):
    Ff, Hf, Nf = field
    return ein_ext(Ff, Hf, x0, eps, Nf)


IDX = [("tt", (0, 0)), ("tz", (0, 3)), ("zz", (3, 3)), ("xx", (1, 1)), ("yy", (2, 2)),
       ("xy", (1, 2)), ("xt", (0, 1)), ("yt", (0, 2)), ("xz", (1, 3)), ("yz", (2, 3))]
ALLOWED = ["tt", "tz", "zz", "xx", "yy", "xt", "xz"]
FORBID = ["ty", "xy", "yz"]
CIDX = dict(IDX)

PTS12 = [(0.0, 0.7, 0.3, -0.5), (0.0, 1.1, 0.3, -0.9), (0.0, 0.5, 0.3, -1.3), (0.0, 1.4, 0.3, -0.2),
         (0.0, 0.9, 0.3, 0.3), (0.0, 0.3, 0.3, -0.9), (0.0, 1.6, 0.3, -1.6), (0.0, 0.2, 0.3, 0.6),
         (0.0, 0.8, 0.3, -1.1), (0.0, 1.2, 0.3, -1.5), (0.0, 0.4, 0.3, 0.2), (0.0, 1.0, 0.3, -0.1)]


def uv_of(x0):
    return (x0[0] - x0[3], x0[0] - x0[1])


print("=" * 78)
print("G1b 第十九轮：(u,v) 类响应分类与交叉修正的最小选择（判据 G19-1 至 G19-6）")
print("=" * 78)

print()
print("[G19-1] 基线：平坦零、单波完成复核与 u 类响应复核")
zf = ein_ext(zero_F, zero_H, (0.0, 1.10, 0.3, -0.2), 0.05, zero_N)
print("     平坦度规：|G| = %.2e" % nrm(zf))
x0s = (0.0, 0.7, 0.3, -0.5)
n1c = nrm(G_at(field_1(A1, comp=True), x0s))
print("     单波 1（含完成）：|G| = %.3e（第十八轮同点读数 %s）" % (n1c, "1.015e-09"))


def u_trace(amp):
    def Hf(T, X, Y, Z, amp=amp):
        p = amp * np.cos(2.0 * (T - Z))
        sh = np.broadcast(T, X, Y, Z).shape
        H = np.zeros((3, 3) + sh)
        for i in range(3):
            H[i, i] = p
        return H
    return zero_F, Hf, zero_N


def resp3(field_mk, x0, a=1e-2, eps=0.03):
    F1, H1, N1 = field_mk(a)
    F2, H2, N2 = field_mk(a / 2.0)
    F3, H3, N3 = field_mk(a / 4.0)
    G1 = ein_ext(F1, H1, x0, eps, N1)
    G2 = ein_ext(F2, H2, x0, eps, N2)
    G3 = ein_ext(F3, H3, x0, eps, N3)
    Amat = np.array([[1.0, a, a * a], [1.0, a / 2.0, a * a / 4.0], [1.0, a / 4.0, a * a / 16.0]])
    st = np.stack([G1.reshape(-1), G2.reshape(-1), G3.reshape(-1)], axis=0)
    c = np.linalg.solve(Amat, st)
    return c[0].reshape(G1.shape)


def resp_hi(field_mk, x0, amp=1e-2, eps=0.03):
    F1f, H1f, N1f = field_mk(amp)
    F2f, H2f, N2f = field_mk(amp / 2.0)
    G1 = ein_ext(F1f, H1f, x0, eps, N1f)
    G2 = ein_ext(F2f, H2f, x0, eps, N2f)
    return (4.0 * G2 - G1) / amp


ru = resp_hi(u_trace, x0s)
print("     u 类复核（迹扇区 cos2u，u = 0.5）：tt=%+.4f zz=%+.4f tz=%+.4f xx=%+.4f yy=%+.4f"
      % (ru[0, 0], ru[3, 3], ru[0, 3], ru[1, 1], ru[2, 2]))
print("     解析：(-B'') 于 (tt,zz,tz) = (+2.1612, +2.1612, -2.1612)，(xx,yy) = +1.0806")
print("     [累计 %.1f 秒]" % (time.time() - T_START))

print()
print("[G19-2] 规范核：八个规范模（4 个 xi 槽乘 2 个轮廓）的响应上界")
GAUGE_PTS = PTS12[:3]


def mk_profile(kind):
    if kind == 'cc':
        return (lambda T, X, Y, Z: np.cos(T - Z) * np.cos(T - X),
                lambda T, X, Y, Z: -np.sin(T - Z) * np.cos(T - X),
                lambda T, X, Y, Z: -np.cos(T - Z) * np.sin(T - X))
    return (lambda T, X, Y, Z: np.sin(T - Z) * np.sin(T - X),
            lambda T, X, Y, Z: np.cos(T - Z) * np.sin(T - X),
            lambda T, X, Y, Z: np.sin(T - Z) * np.cos(T - X))


def gauge_field(slot, kind, amp):
    p, pu, pv = mk_profile(kind)

    def Ff(T, X, Y, Z, amp=amp, slot=slot, pu=pu, pv=pv):
        if slot == 0:
            return 2.0 * amp * (pu(T, X, Y, Z) + pv(T, X, Y, Z))
        return np.zeros_like(T + X + Y + Z)

    def Nf(T, X, Y, Z, amp=amp, slot=slot, pu=pu, pv=pv):
        sh = np.broadcast(T, X, Y, Z).shape
        Nv = np.zeros((3,) + sh)
        if slot == 0:
            Nv[0] = -amp * pv(T, X, Y, Z)
            Nv[2] = -amp * pu(T, X, Y, Z)
        elif slot == 1:
            Nv[0] = amp * (pu(T, X, Y, Z) + pv(T, X, Y, Z))
        elif slot == 2:
            Nv[1] = amp * (pu(T, X, Y, Z) + pv(T, X, Y, Z))
        else:
            Nv[2] = amp * (pu(T, X, Y, Z) + pv(T, X, Y, Z))
        return Nv

    def Hf(T, X, Y, Z, amp=amp, slot=slot, pu=pu, pv=pv):
        sh = np.broadcast(T, X, Y, Z).shape
        H = np.zeros((3, 3) + sh)
        if slot == 1:
            H[0, 0] = -2.0 * amp * pv(T, X, Y, Z)
            H[0, 2] = -amp * pu(T, X, Y, Z)
            H[2, 0] = H[0, 2]
        elif slot == 2:
            H[0, 1] = -amp * pv(T, X, Y, Z)
            H[1, 0] = H[0, 1]
            H[1, 2] = -amp * pu(T, X, Y, Z)
            H[2, 1] = H[1, 2]
        elif slot == 3:
            H[0, 2] = -amp * pv(T, X, Y, Z)
            H[2, 0] = H[0, 2]
            H[2, 2] = -2.0 * amp * pu(T, X, Y, Z)
        return H
    return Ff, Hf, Nf


gmax = 0.0
for slot in range(4):
    for kind in ('cc', 'ss'):
        m = 0.0
        for x0 in GAUGE_PTS:
            r = resp3(lambda amp, slot=slot, kind=kind: gauge_field(slot, kind, amp), x0)
            m = max(m, nrm(r))
        gmax = max(gmax, m)
        print("     xi_%d，轮廓 %-2s：max|响应| = %.2e" % (slot, kind, m))
print("     八个规范模的响应上界 = %.2e（信号标度 O(1)：相对 %.1e）" % (gmax, gmax))
print("     [累计 %.1f 秒]" % (time.time() - T_START))

print()
print("[G19-3] 像的结构：40 列（10 场乘 4 轮廓）奇异谱")
SHAPES = ["cosu cosv", "cosu sinv", "sinu cosv", "sinu sinv"]


def mk_prof(tag):
    if tag == "cosu cosv":
        return lambda T, X, Y, Z: np.cos(T - Z) * np.cos(T - X)
    if tag == "cosu sinv":
        return lambda T, X, Y, Z: np.cos(T - Z) * np.sin(T - X)
    if tag == "sinu cosv":
        return lambda T, X, Y, Z: np.sin(T - Z) * np.cos(T - X)
    return lambda T, X, Y, Z: np.sin(T - Z) * np.sin(T - X)


def mk_corr_full(kind, idx, shape, amp):
    prof = mk_prof(shape)
    if kind == 'F':
        def Ff(T, X, Y, Z, amp=amp, prof=prof):
            return amp * prof(T, X, Y, Z)
        return Ff, zero_H, zero_N
    if kind == 'N':
        def Nf(T, X, Y, Z, amp=amp, prof=prof, idx=idx):
            sh = np.broadcast(T, X, Y, Z).shape
            Nv = np.zeros((3,) + sh)
            Nv[idx] = amp * prof(T, X, Y, Z)
            return Nv
        return zero_F, zero_H, Nf

    def Hf(T, X, Y, Z, amp=amp, prof=prof, idx=idx):
        p = amp * prof(T, X, Y, Z)
        sh = np.broadcast(T, X, Y, Z).shape
        H = np.zeros((3, 3) + sh)
        H[idx[0], idx[1]] = p
        H[idx[1], idx[0]] = p
        return H
    return zero_F, Hf, zero_N


def response_fast(kind, idx, shape, x0, amp=1e-2, eps=0.03):
    F1f, H1f, N1f = mk_corr_full(kind, idx, shape, amp)
    F2f, H2f, N2f = mk_corr_full(kind, idx, shape, amp / 2.0)
    G1 = einstein_at4(F1f, H1f, x0, eps, N1f)
    G2 = einstein_at4(F2f, H2f, x0, eps, N2f)
    return (4.0 * G2 - G1) / amp


FIELDS = [('F', 0)]
for i in range(3):
    FIELDS.append(('N', i))
for ij in ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)):
    FIELDS.append(('H', ij))
cols2 = []
for kind, idx in FIELDS:
    for shape in SHAPES:
        cols2.append((kind, idx, shape))
rows, brows = [], []
for x0 in PTS12:
    R = G_at(field_12(A1, A2), x0) - G_at(field_1(A1), x0) - G_at(field_2(A2), x0)
    resps = [response_fast(k, i, sh, x0) for k, i, sh in cols2]
    for tag in ALLOWED:
        i2, j2 = CIDX[tag]
        rows.append([Rr[i2, j2] for Rr in resps])
        brows.append(-R[i2, j2])
Mfull = np.array(rows)
bfull = np.array(brows)
sv = np.linalg.svd(Mfull, compute_uv=False)
print("     矩阵 %d 行乘 %d 列；奇异值：>=1e-1 的 %d 个，>=1e-2 的 %d 个，>=1e-3 的 %d 个"
      % (Mfull.shape[0], Mfull.shape[1], int((sv >= 1e-1).sum()), int((sv >= 1e-2).sum()), int((sv >= 1e-3).sum())))
print("     前六个 = " + "  ".join("%.2e" % x for x in sv[:6]))
print("     末六个 = " + "  ".join("%.2e" % x for x in sv[-6:]))
print("     第 14 至 20 个 = " + "  ".join("%.2e" % x for x in sv[13:20]))
print("     [累计 %.1f 秒]" % (time.time() - T_START))

print()
print("[G19-4] 支撑阶梯（拟合 8 点 56 行／校验 4 点 28 行）")
NFIT = 8 * len(ALLOWED)
LADDER = [
    ("对角 H（xx,yy,zz）", [('H', (0, 0)), ('H', (1, 1)), ('H', (2, 2))]),
    ("加 F 与 N_x、N_z", [('H', (0, 0)), ('H', (1, 1)), ('H', (2, 2)), ('F', 0), ('N', 0), ('N', 2)]),
    ("加 N_y（全 N）", [('H', (0, 0)), ('H', (1, 1)), ('H', (2, 2)), ('F', 0), ('N', 0), ('N', 1), ('N', 2)]),
    ("全类（含 off-diagonal H）", None),
]
for name, sel in LADDER:
    if sel is None:
        idxs = list(range(len(cols2)))
    else:
        idxs = [i for i, (k, ix, sh) in enumerate(cols2) if (k, ix) in sel]
    Ms = Mfull[:, idxs]
    bs = bfull
    sols, _, rks, _ = np.linalg.lstsq(Ms[:NFIT], bs[:NFIT], rcond=None)
    rfit = float(np.linalg.norm(Ms[:NFIT] @ sols - bs[:NFIT])) / float(np.linalg.norm(bs[:NFIT]))
    rval = float(np.linalg.norm(Ms[NFIT:] @ sols - bs[NFIT:])) / float(np.linalg.norm(bs[NFIT:]))
    print("     %-24s 列数 %2d：拟合闭合比 = %.3e ｜ 校验闭合比 = %.3e" % (name, len(idxs), rfit, rval))
print("     [累计 %.1f 秒]" % (time.time() - T_START))

print()
print("[G19-5] 全类最小范数解的形状（前八个系数，按归一列）")
nrmc = np.linalg.norm(Mfull, axis=0)
Mn = Mfull / nrmc
soln, _, rkn, _ = np.linalg.lstsq(Mn[:NFIT], bfull[:NFIT], rcond=None)
resn = Mn[NFIT:] @ soln - bfull[NFIT:]
rvaln = float(np.linalg.norm(resn)) / float(np.linalg.norm(bfull[NFIT:]))
order = np.argsort(-np.abs(soln))[:8]
for idx in order:
    k, ix, sh = cols2[idx]
    print("     %-2s %-4s %-10s 归一系数 = %+.4f ｜ 实际幅度 = %+.2e" % (k, str(ix), sh, soln[idx], soln[idx] / nrmc[idx]))
print("     最大归一系数 = %.4f ｜ 校验闭合比（归一列解）= %.3e ｜ 注：列强共线，系数属最小范数解、结构信息以支撑阶梯为准" % (np.max(np.abs(soln)), rvaln))
print("     [累计 %.1f 秒]" % (time.time() - T_START))

print()
print("[G19-6] 登记")
rows2 = [
    ("其一", "基线", "平坦零、单波完成复核、u 类响应复核（对上第十五轮读数）"),
    ("其二", "规范核", "八个规范模的响应上界（机器零级，判据核）"),
    ("其三", "像结构", "40 列奇异谱的断口（物理维数与规范方向计数）"),
    ("其四", "支撑阶梯", "四级子类的拟合/校验闭合比：最少需要哪些槽"),
    ("其五", "最小范数解", "主要系数与修正幅度（交叉修正的结构提示）"),
    ("其六", "余项", "交叉修正的闭式规则、振幅不等与一般斜交、强场端"),
]
for a1, a2, a3 in rows2:
    print("     %-4s %-8s %s" % (a1, a2, a3))

print()
print("总运行时间 = %.1f 秒" % (time.time() - T_START))
