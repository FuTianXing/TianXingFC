# -*- coding: utf-8 -*-
"""
g1b_crossing_waves.py

[G1b 第十八轮] 两束交叉平面波的二阶装配：交叉残差、逐波完成与交叉修正候选（判据 G18-1 至 G18-5）

背景：第十五至十七轮把单束（平行传播）平面波的二阶装配与平移源做通。非平面构型的
第一入口是两束交叉平面波：一束沿 z（u = t - z，横向面 x-y），一束沿 x（v = t - x，
横向面 y-z），两者在 y 方向重叠。线性层两束独立；二阶层出现交叉项 2G_2[h_1, h_2]，
它既不被逐波的完成消掉、也不属于任何单束的 u 类或 v 类结构。本轮做三件：
其一，交叉残差的存在、分量结构与 ab 标度；其二，逐波罗森完成之后的残余水平；
其三，交叉修正候选（迹槽、标量槽、矢量槽 N_x、P_xz 槽乘四个 (u,v) 轮廓）的最小二乘
闭合尝试。

解析预期：
1) 对称性：构型在 y -> -y 下不变（禁戒 ty、xy、yz，实测机器零），z 反射不是对称性
   （两束各带自己的传播方向）；残差落七分量 {tt, tz, zz, xx, yy, xt, xz}。
2) 交叉残差 R = G[h1+h2] - G[h1] - G[h2] = 2 G_2[h1, h2] + O(a^3)，按 ab 标度；
   逐波完成（各自横向面的罗森迹）消掉 a^2 与 b^2 部分，残余仍为 O(ab)。
3) 交叉修正的候选空间：(u,v) 轮廓基 {cos u cos v, cos u sin v, sin u cos v, sin u sin v}
   乘槽位 {迹、h00、N_x、P_xz 图案}。判据问：自然基能否把交叉残差压下去，压到多少。

判据
G18-1 基线：平坦零、单波复核（含完成，a^4 标度）
G18-2 交叉残差：分量结构、ab 标度与 (u,v) 形状
G18-3 逐波完成后的残余水平
G18-4 交叉修正候选的拟合（自然槽位基：迹、h00、N_x、P_xz）
G18-5 一般 (u,v) 修正类的闭合（10 场乘 4 轮廓，含 SVD 有效秩）
G18-6 登记

运行：python 验证脚本/g1b_crossing_waves.py（确定性，约 1 分钟）
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

PTS = [(0.0, 0.7, 0.3, -0.5), (0.0, 1.1, 0.3, -0.9), (0.0, 0.5, 0.3, -1.3), (0.0, 1.4, 0.3, -0.2)]


def uv_of(x0):
    return (x0[0] - x0[3], x0[0] - x0[1])


print("=" * 78)
print("G1b 第十八轮：两束交叉平面波的二阶装配（判据 G18-1 至 G18-6）")
print("=" * 78)

print()
print("[G18-1] 基线：平坦零与单波复核（含完成）")
zf = ein_ext(zero_F, zero_H, (0.0, 1.10, 0.3, -0.2), 0.05, zero_N)
print("     平坦度规：|G| = %.2e" % nrm(zf))
x0s = (0.0, 0.7, 0.3, -0.5)
n1a = nrm(G_at(field_1(A1, comp=True), x0s))
n1b = nrm(G_at(field_1(A1 / 2.0, comp=True), x0s))
n2a = nrm(G_at(field_2(A2, comp=True), x0s))
print("     单波 1（含完成）：|G|(a) = %.3e，|G|(a/2) = %.3e，比 = %.2f（a^4 期望 16）"
      % (n1a, n1b, n1a / max(n1b, 1e-30)))
print("     单波 2（含完成）：|G| = %.3e" % n2a)
print("     [累计 %.1f 秒]" % (time.time() - T_START))

print()
print("[G18-2] 交叉残差 R = G[h1+h2] - G[h1] - G[h2]（未完成；四点）")
for x0 in PTS:
    u, v = uv_of(x0)
    Gs = G_at(field_12(A1, A2), x0)
    G1 = G_at(field_1(A1), x0)
    G2 = G_at(field_2(A2), x0)
    R = Gs - G1 - G2
    oth = max(abs(R[i, j]) for tag, (i, j) in IDX if tag in FORBID)
    line = "     (u,v) = (%+.1f, %+.1f)：|R| = %.3e ｜ 分量" % (u, v, nrm(R))
    for tag in ALLOWED:
        i, j = dict(IDX)[tag]
        line += " %s=%+.3e" % (tag, R[i, j])
    print(line)
    print("        禁戒分量（y 反射）最大 = %.2e（相对 |R| %.1e）" % (oth, oth / nrm(R)))
print("     标度：b 对 b/2")
x0z = PTS[1]
Rfull = G_at(field_12(A1, A2), x0z) - G_at(field_1(A1), x0z) - G_at(field_2(A2), x0z)
Rhalf = G_at(field_12(A1, A2 / 2.0), x0z) - G_at(field_1(A1), x0z) - G_at(field_2(A2 / 2.0), x0z)
print("     |R|(b) / |R|(b/2) = %.2f（ab 期望 2）" % (nrm(Rfull) / nrm(Rhalf)))
print("     [累计 %.1f 秒]" % (time.time() - T_START))

print()
print("[G18-3] 逐波完成后的残余水平（四点）")
for x0 in PTS:
    u, v = uv_of(x0)
    Gs = G_at(field_12(A1, A2, comp=True), x0)
    G1 = G_at(field_1(A1, comp=True), x0)
    G2 = G_at(field_2(A2, comp=True), x0)
    Gs0 = G_at(field_12(A1, A2), x0)
    print("     (u,v) = (%+.1f, %+.1f)：|G|（未完成）= %.3e ｜ |G|（含完成）= %.3e ｜ 交叉部 |G_s - G_1 - G_2| = %.3e"
          % (u, v, nrm(Gs0), nrm(Gs), nrm(Gs - G1 - G2)))
print("     [累计 %.1f 秒]" % (time.time() - T_START))

print()
print("[G18-4] 交叉修正候选的拟合（迹、h00、N_x、P_xz 乘四个 (u,v) 轮廓）")
SLOTS = ["T", "F", "Nx", "Pxz"]
SHAPES = ["cosu cosv", "cosu sinv", "sinu cosv", "sinu sinv"]


def mk_prof(tag):
    if tag == "cosu cosv":
        return lambda T, X, Y, Z: np.cos(T - Z) * np.cos(T - X)
    if tag == "cosu sinv":
        return lambda T, X, Y, Z: np.cos(T - Z) * np.sin(T - X)
    if tag == "sinu cosv":
        return lambda T, X, Y, Z: np.sin(T - Z) * np.cos(T - X)
    return lambda T, X, Y, Z: np.sin(T - Z) * np.sin(T - X)


def mk_corr(slot, shape, amp):
    prof = mk_prof(shape)
    if slot == "T":
        def Hf(T, X, Y, Z, amp=amp, prof=prof):
            p = amp * prof(T, X, Y, Z)
            sh = np.broadcast(T, X, Y, Z).shape
            H = np.zeros((3, 3) + sh)
            for i in range(3):
                H[i, i] = p
            return H
        return zero_F, Hf, zero_N
    if slot == "F":
        def Ff(T, X, Y, Z, amp=amp, prof=prof):
            return amp * prof(T, X, Y, Z)
        return Ff, zero_H, zero_N
    if slot == "Nx":
        def Nf(T, X, Y, Z, amp=amp, prof=prof):
            sh = np.broadcast(T, X, Y, Z).shape
            Nv = np.zeros((3,) + sh)
            Nv[0] = amp * prof(T, X, Y, Z)
            return Nv
        return zero_F, zero_H, Nf
    def Hf(T, X, Y, Z, amp=amp, prof=prof):
        p = amp * prof(T, X, Y, Z)
        sh = np.broadcast(T, X, Y, Z).shape
        H = np.zeros((3, 3) + sh)
        H[0, 0] = p
        H[2, 2] = -p
        return H
    return zero_F, Hf, zero_N


def response(slot, shape, x0, amp=1e-2, eps=0.03):
    F1f, H1f, N1f = mk_corr(slot, shape, amp)
    F2f, H2f, N2f = mk_corr(slot, shape, amp / 2.0)
    G1 = ein_ext(F1f, H1f, x0, eps, N1f)
    G2 = ein_ext(F2f, H2f, x0, eps, N2f)
    return (4.0 * G2 - G1) / amp


cols = []
for slot in SLOTS:
    for shape in SHAPES:
        cols.append((slot, shape))
rows, brows = [], []
for x0 in PTS:
    R = G_at(field_12(A1, A2), x0) - G_at(field_1(A1), x0) - G_at(field_2(A2), x0)
    resps = [response(slot, shape, x0) for slot, shape in cols]
    for tag in ALLOWED:
        i, j = dict(IDX)[tag]
        rows.append([Rr[i, j] for Rr in resps])
        brows.append(-R[i, j])
M = np.array(rows)
bv = np.array(brows)
nrmc = np.linalg.norm(M, axis=0)
Mn = M / nrmc
soln, rr, rk, sv = np.linalg.lstsq(Mn, bv, rcond=None)
resn = Mn @ soln - bv
ratio = float(np.linalg.norm(resn)) / float(np.linalg.norm(bv))
print("     列数 %d，行数 %d，秩 %d ｜ 归一化拟合残差比 = %.3e" % (M.shape[1], M.shape[0], rk, ratio))
solA, rrA, rkA, svA = np.linalg.lstsq(M, bv, rcond=None)
resA = M @ solA - bv
ratioA = float(np.linalg.norm(resA)) / float(np.linalg.norm(bv))
print("     绝对范数口径的拟合残差比 = %.3e（列强共线、系数放大）" % ratioA)
mx = np.argsort(-np.abs(soln))[:6]
for idx in mx:
    print("     系数 %-8s %-12s = %+.4e（归一列）" % (cols[idx][0], cols[idx][1], soln[idx]))
for i0 in range(0, len(bv), len(ALLOWED)):
    x0 = PTS[i0 // len(ALLOWED)]
    uu, vv = uv_of(x0)
    seg = resn[i0:i0 + len(ALLOWED)]
    print("     拟合后残余（(u,v) = (%+.1f, %+.1f)）：%s" % (uu, vv, " ".join("%s=%+.2e" % (t, seg[k]) for k, t in enumerate(ALLOWED))))
x0 = PTS[1]
R = G_at(field_12(A1, A2), x0) - G_at(field_1(A1), x0) - G_at(field_2(A2), x0)
print("     单点闭合检验（(u,v) = (%+.1f, %+.1f)）：|R| = %.3e" % (uv_of(x0)[0], uv_of(x0)[1], nrm(R)))
print("     [累计 %.1f 秒]" % (time.time() - T_START))

print()
print("[G18-5] 一般 (u,v) 修正类的闭合（10 场乘 4 轮廓 = 40 列；单 eps 幅度 Richardson）")
FIELDS = [('F', 0)]
for i in range(3):
    FIELDS.append(('N', i))
for ij in ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)):
    FIELDS.append(('H', ij))


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


PTS12 = list(PTS) + [(0.0, 0.9, 0.3, 0.3), (0.0, 0.3, 0.3, -0.9), (0.0, 1.6, 0.3, -1.6), (0.0, 0.2, 0.3, 0.6), (0.0, 0.8, 0.3, -1.1), (0.0, 1.2, 0.3, -1.5), (0.0, 0.4, 0.3, 0.2), (0.0, 1.0, 0.3, -0.1)]
cols2 = []
for kind, idx in FIELDS:
    for shape in SHAPES:
        cols2.append((kind, idx, shape))


def build_rows(ptlist):
    rows, brows = [], []
    for x0 in ptlist:
        R = G_at(field_12(A1, A2), x0) - G_at(field_1(A1), x0) - G_at(field_2(A2), x0)
        resps = [response_fast(k, i, s, x0) for k, i, s in cols2]
        for tag in ALLOWED:
            i2, j2 = dict(IDX)[tag]
            rows.append([Rr[i2, j2] for Rr in resps])
            brows.append(-R[i2, j2])
    return np.array(rows), np.array(brows)


M2, b2 = build_rows(PTS12[:8])
M3, b3 = build_rows(PTS12[8:])
sol2, rr2, rk2, sv2 = np.linalg.lstsq(M2, b2, rcond=None)
res2 = M2 @ sol2 - b2
ratio2 = float(np.linalg.norm(res2)) / float(np.linalg.norm(b2))
res3 = M3 @ sol2 - b3
ratio3 = float(np.linalg.norm(res3)) / float(np.linalg.norm(b3))
s2 = np.linalg.svd(M2, compute_uv=False)
print("     拟合点 8 个（%d 行）／校验点 4 个（%d 行）；列数 %d，秩 %d" % (M2.shape[0], M3.shape[0], M2.shape[1], rk2))
print("     拟合闭合比 = %.3e ｜ 校验闭合比 = %.3e" % (ratio2, ratio3))
print("     奇异值前六个 = " + "  ".join("%.2e" % x for x in s2[:6]))
print("     [累计 %.1f 秒]" % (time.time() - T_START))

print()
print("[G18-6] 登记")
rows2 = [
    ("其一", "基线", "平坦零；单波（含完成）复核与 a^4 标度"),
    ("其二", "交叉残差", "分量结构（允许集）与 ab 标度"),
    ("其三", "逐波完成", "完成前后的 |G| 水平与交叉部读数"),
    ("其四", "交叉修正", "自然槽位基（16 列）的拟合残差比（两口径）"),
    ("其五", "一般类闭合", "10 场乘 4 轮廓（40 列）的闭合比与 SVD 有效秩"),
    ("其六", "余项", "交叉修正的解析形式与最小选择、(u,v) 类响应分类、强场端"),
]
for a1, a2, a3 in rows2:
    print("     %-4s %-8s %s" % (a1, a2, a3))

print()
print("总运行时间 = %.1f 秒" % (time.time() - T_START))
