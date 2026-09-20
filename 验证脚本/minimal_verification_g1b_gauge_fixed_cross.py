# -*- coding: utf-8 -*-
"""
minimal_verification_g1b_gauge_fixed_cross.py

[G1b 第二十轮] 六场支撑里的规范固定与交叉修正的唯一代表（判据 G20-1 至 G20-6）

背景：第十九轮定出交叉修正的最小支撑（六个场：F、N_x、N_z、H_xx、H_yy、H_zz 乘四个
(u,v) 轮廓 = 24 列），并指出共线基下最小范数解幅度虚高、缺规范选择。本轮做三件：
其一，显式构造六场支撑内的规范子空间（槽 0 的四维族 + (xi_x, xi_z) 满足 (xi_x)_u +
(xi_z)_v = 0 的四维族，共八维）并核验零响应；
其二，在规范正交补里解唯一代表（最小物理幅度解）并核验闭合；
其三，读出唯一代表的逐场逐轮廓结构（交叉修正的形状），并做截断阈稳定性对照。

解析预期：
1) 规范子空间：槽 0 族 C_{00} = 2(f_u + f_v)、C_{0x} = -f_v、C_{0z} = -f_u；
   (xi_x, xi_z) 族 C_{0x} = (xi_x)_u + (xi_x)_v、C_{0z} = (xi_z)_u + (xi_z)_v、
   C_{xx} = -2(xi_x)_v、C_{zz} = -2(xi_z)_u，约束 (xi_z)_v = -(xi_x)_u 使 C_{xz} = 0。
   两族共八维，响应应为零（三点幅度外推下 <= 1e-6 级）。
2) 唯一代表：min |c| s.t. M c = b 且 c 垂直于规范子空间；若收敛，闭合比应回到
   第十九轮的 0.6%/0.5% 地板，且幅度落到 1e-4 量级（物理幅度）。
3) 结构：逐场逐轮廓系数给出交叉修正的形状；做 1e-6/1e-5/1e-4 三档截断阈的稳定性对照。

判据
G20-1 基线：平坦零、单波完成复核与交叉残差复核
G20-2 规范子空间：八个基矢的核验（在测量矩阵内与独立测量两路）
G20-3 唯一代表：规范正交补里的最小幅度解与闭合比
G20-4 结构：唯一代表的逐场逐轮廓系数表
G20-5 稳定性：三档截断阈与幅度读数
G20-6 登记

运行：python 验证脚本/minimal_verification_g1b_gauge_fixed_cross.py（确定性，约 1 分钟）
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
CIDX = dict(IDX)

PTS12 = [(0.0, 0.7, 0.3, -0.5), (0.0, 1.1, 0.3, -0.9), (0.0, 0.5, 0.3, -1.3), (0.0, 1.4, 0.3, -0.2),
         (0.0, 0.9, 0.3, 0.3), (0.0, 0.3, 0.3, -0.9), (0.0, 1.6, 0.3, -1.6), (0.0, 0.2, 0.3, 0.6),
         (0.0, 0.8, 0.3, -1.1), (0.0, 1.2, 0.3, -1.5), (0.0, 0.4, 0.3, 0.2), (0.0, 1.0, 0.3, -0.1)]
NFIT = 8 * len(ALLOWED)

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


FIELDS6 = [('F', 0), ('N', 0), ('N', 2), ('H', (0, 0)), ('H', (1, 1)), ('H', (2, 2))]
cols = []
for kind, idx in FIELDS6:
    for shape in SHAPES:
        cols.append((kind, idx, shape))
CIDX24 = {(k, ix, sh): i for i, (k, ix, sh) in enumerate(cols)}

# profile coefficient algebra: vector c = (cc, cs, sc, ss)


def du_c(c):
    return np.array([c[2], c[3], -c[0], -c[1]])


def dv_c(c):
    return np.array([c[1], -c[0], c[3], -c[2]])


def vec_from_components(comp_map):
    v = np.zeros(24)
    for (kind, idx), c in comp_map.items():
        for s, tag in enumerate(SHAPES):
            key = (kind, idx, tag)
            if key in CIDX24:
                v[CIDX24[key]] += c[s]
    return v


GAUGE = []
for s in range(4):
    k = np.zeros(4)
    k[s] = 1.0
    comp = {('F', 0): 2.0 * (du_c(k) + dv_c(k)), ('N', 0): -dv_c(k), ('N', 2): -du_c(k)}
    GAUGE.append(vec_from_components(comp))
for (a1v, a2v, b1v, b2v) in [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]:
    ax = np.array([a1v, a2v, -b2v, b1v])
    az = np.array([b1v, b2v, -a2v, a1v])
    comp = {('N', 0): du_c(ax) + dv_c(ax), ('N', 2): du_c(az) + dv_c(az),
            ('H', (0, 0)): -2.0 * dv_c(ax), ('H', (2, 2)): -2.0 * du_c(az)}
    GAUGE.append(vec_from_components(comp))
GAUGE = np.array(GAUGE)

print("=" * 78)
print("G1b 第二十轮：六场支撑里的规范固定与交叉修正的唯一代表（判据 G20-1 至 G20-6）")
print("=" * 78)

print()
print("[G20-1] 基线：平坦零、单波完成复核与交叉残差复核")
zf = ein_ext(zero_F, zero_H, (0.0, 1.10, 0.3, -0.2), 0.05, zero_N)
print("     平坦度规：|G| = %.2e" % nrm(zf))
x0s = PTS12[0]
print("     单波 1（含完成）：|G| = %.3e" % nrm(G_at(field_1(A1, comp=True), x0s)))
R0 = G_at(field_12(A1, A2), x0s) - G_at(field_1(A1), x0s) - G_at(field_2(A2), x0s)
print("     交叉残差（点 1）：|R| = %.3e（第十八轮同点 1.803e-04）" % nrm(R0))
print("     [累计 %.1f 秒]" % (time.time() - T_START))

print()
print("[G20-2] 规范子空间：八个基矢的核验")
rows, brows = [], []
for x0 in PTS12:
    R = G_at(field_12(A1, A2), x0) - G_at(field_1(A1), x0) - G_at(field_2(A2), x0)
    resps = [response_fast(k, i, sh, x0) for k, i, sh in cols]
    for tag in ALLOWED:
        i2, j2 = CIDX[tag]
        rows.append([Rr[i2, j2] for Rr in resps])
        brows.append(-R[i2, j2])
M = np.array(rows)
b = np.array(brows)
gmax = 0.0
for gi in range(8):
    r = M @ GAUGE[gi]
    gmax = max(gmax, float(np.linalg.norm(r)))
print("     在测量矩阵内：八个基矢的响应范数上界 = %.3e" % gmax)


def resp3_field(mk, x0, a=1e-2, eps=0.03):
    F1, H1, N1 = mk(a)
    F2, H2, N2 = mk(a / 2.0)
    F3, H3, N3 = mk(a / 4.0)
    G1 = ein_ext(F1, H1, x0, eps, N1)
    G2 = ein_ext(F2, H2, x0, eps, N2)
    G3 = ein_ext(F3, H3, x0, eps, N3)
    Amat = np.array([[1.0, a, a * a], [1.0, a / 2.0, a * a / 4.0], [1.0, a / 4.0, a * a / 16.0]])
    st = np.stack([G1.reshape(-1), G2.reshape(-1), G3.reshape(-1)], axis=0)
    cc = np.linalg.solve(Amat, st)
    return cc[0].reshape(G1.shape)


def mk_gauge_field(vec, amp=1.0):
    parts = {}
    for i, (k, ix, sh) in enumerate(cols):
        if abs(vec[i]) > 0:
            parts[(k, ix, sh)] = amp * vec[i]
    def Ff(T, X, Y, Z):
        tot = np.zeros_like(T + X + Y + Z)
        for (k, ix, sh), a in parts.items():
            if k == 'F':
                tot = tot + a * mk_prof(sh)(T, X, Y, Z)
        return tot
    def Hf(T, X, Y, Z):
        shp = np.broadcast(T, X, Y, Z).shape
        H = np.zeros((3, 3) + shp)
        for (k, ix, sh), a in parts.items():
            if k == 'H':
                H[ix[0], ix[1]] += a * mk_prof(sh)(T, X, Y, Z)
                H[ix[1], ix[0]] = H[ix[0], ix[1]]
        return H
    def Nf(T, X, Y, Z):
        shp = np.broadcast(T, X, Y, Z).shape
        Nv = np.zeros((3,) + shp)
        for (k, ix, sh), a in parts.items():
            if k == 'N':
                Nv[ix] += a * mk_prof(sh)(T, X, Y, Z)
        return Nv
    return Ff, Hf, Nf


gm2 = 0.0
for gi in (0, 4):
    r = resp3_field(lambda amp, gi=gi: mk_gauge_field(GAUGE[gi], amp), PTS12[0], a=1e-3)
    gm2 = max(gm2, nrm(r))
print("     独立测量（基矢 0、4，三点幅度外推）：响应范数上界 = %.3e" % gm2)
print("     [累计 %.1f 秒]" % (time.time() - T_START))

print()
print("[G20-3] 唯一代表：规范正交补里的最小幅度解")
Q, _ = np.linalg.qr(GAUGE.T)
Pproj = np.eye(24) - Q @ Q.T
Mp = M @ Pproj
sols, _, rks, _ = np.linalg.lstsq(Mp[:NFIT], b[:NFIT], rcond=None)
cstar = Pproj @ sols
rfit = float(np.linalg.norm(Mp[:NFIT] @ sols - b[:NFIT])) / float(np.linalg.norm(b[:NFIT]))
rval = float(np.linalg.norm(Mp[NFIT:] @ sols - b[NFIT:])) / float(np.linalg.norm(b[NFIT:]))
gauge_res = float(np.linalg.norm(GAUGE @ cstar))
print("     拟合闭合比 = %.3e ｜ 校验闭合比 = %.3e ｜ 规范残余 = %.3e" % (rfit, rval, gauge_res))
print("     最小幅度 |c*| = %.3e ｜ 最大单系数 = %.3e ｜ 秩 = %d" % (np.linalg.norm(cstar), np.max(np.abs(cstar)), rks))
print("     [累计 %.1f 秒]" % (time.time() - T_START))

print()
print("[G20-4] 唯一代表的逐场逐轮廓系数（×1e5）")
for k, ix in FIELDS6:
    vals = []
    for sh in SHAPES:
        vals.append(cstar[CIDX24[(k, ix, sh)]] * 1e5)
    print("     %-2s %-4s：" % (k, str(ix)) + "  ".join("%-10s %+8.3f" % (SHAPES[i], vals[i]) for i in range(4)))
print("     [累计 %.1f 秒]" % (time.time() - T_START))

print()
print("[G20-5] 稳定性：三档截断阈")
for th in (1e-6, 1e-5, 1e-4):
    U, S, Vt = np.linalg.svd(Mp, full_matrices=False)
    keep = S > th * S[0]
    nk = int(keep.sum())
    scal = U[:, keep].T @ b
    xk = Vt[:nk].T @ (scal / S[keep])
    cth = Pproj @ xk
    rv = float(np.linalg.norm(Mp[NFIT:] @ xk - b[NFIT:])) / float(np.linalg.norm(b[NFIT:]))
    ang = float(np.dot(cth, cstar) / (np.linalg.norm(cth) * np.linalg.norm(cstar) + 1e-30))
    print("     阈 %.0e（保留 %2d 个方向）：|c| = %.3e ｜ 校验闭合比 = %.3e ｜ 与 c* 的夹角余弦 = %.4f"
          % (th, int(keep.sum()), np.linalg.norm(cth), rv, ang))
print("     [累计 %.1f 秒]" % (time.time() - T_START))

print()
print("[G20-6] 登记")
rows2 = [
    ("其一", "基线", "平坦零、单波完成与交叉残差复核"),
    ("其二", "规范子空间", "八个基矢在测量矩阵内与独立测量两路核验"),
    ("其三", "唯一代表", "规范正交补内的最小幅度解、闭合比与规范残余"),
    ("其四", "结构", "逐场逐轮廓系数（六场各四个）"),
    ("其五", "稳定性", "三档截断阈下的 |c|、校验闭合比与夹角余弦"),
    ("其六", "余项", "交叉修正闭式规则的候选形式、振幅不等与一般斜交、强场端"),
]
for a1, a2, a3 in rows2:
    print("     %-4s %-8s %s" % (a1, a2, a3))

print()
print("总运行时间 = %.1f 秒" % (time.time() - T_START))
