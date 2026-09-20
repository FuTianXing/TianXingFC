# -*- coding: utf-8 -*-
"""
minimal_verification_g1b_second_order_metric3.py

[G1b 第十五轮] 横向/无迹基元的下一层装配（判据 G1b15-1 至 G1b15-6）

背景：第十四轮以迹扇区（cos2u 与 u^2 两轮廓）在零向三分量（G_tt, G_zz, G_tz）上完成
二阶装配，残余移到横向分量（G_xx = G_yy），总 |G| 仍按 a^2 标度。本轮做下一层：
先给出 u 依赖修正的线性响应分类，再以扩展基元（迹槽加矢量槽）做全分量装配，
检验降阶与标度。

解析预期（u = t - z；对 u 的导数记撇）：
1) 线性响应公式：记 bar_h = h - (1/2) eta h（trace reverse），k 为 u 的梯度，
   T_mu = k^rho bar_h''_{mu rho}；则 G_{mu nu} = (k_mu T_nu + k_nu T_mu)/2
   - (1/2) eta_{mu nu} (k . T)。
2) 分类（对角槽 p_x, p_y, p_z；标量槽 F 进 h_00；矢量槽 N_i 进 g_0i）：
   null3 块：G_tt = G_zz = -G_tz = -(p_x + p_y)''/2；
   横向块：G_xx = G_yy = -p_z''/2 - F''/2 - N_z''；
   混合块：G_xt = -G_xz = -(N_x + C_xz)''/2；
   无迹型（P_+，P_x）响应恒为零。
3) 罗森条件 gamma_1''/gamma_1 + gamma_2''/gamma_2 = 0 给 (p + q)'' = 2 A'' A + A'^2
   （A = a cos u）；对称分裂 p = q = B 给 B = (3 a^2/16) cos 2u - (a^2/8) u^2，
   即第十四轮拟合的系数。
4) 第十四轮修正 = 罗森形 + 仅 z 轴迹扰动 Delta = diag(0, 0, B) 的自身响应：
   残余 G_xx = G_yy = -B''/2 = (3 a^2/8) cos 2u + a^2/8。
5) 补偿条件 N_z'' = -B''/2 给 N_z = -(3 a^2/32) cos 2u + (a^2/16) u^2；
   装配后 a^2 残留全消；领先残差的标度由本轮实测判定（a^3 未出现，落 a^4）。
   若把迹修正支撑在横向两轴（罗森形本体），领先残差同样落 a^4。

判据
G1b15-1 器械核验：ein_ext（eps 半步 Richardson）下平坦零与各向同性静态回归
G1b15-2 基元响应分类：十六基元全分量读数对解析分类（含 TT 型恒零与横向块三来源）
G1b15-3 第十四轮残余定位：全分量读数与两形状拟合（c0 = a^2/8，c2 = 3a^2/8）
G1b15-4 扩展装配：两段拟合（迹两列 + 矢量槽两列）系数对解析；联合四列最小二乘对照
G1b15-5 降阶与标度：五方案（未修正、第十四轮、装配解析、装配拟合、罗森形）
G1b15-6 登记

运行：python 验证脚本/minimal_verification_g1b_second_order_metric3.py（确定性，约 2 分钟）
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


AMP = 1e-2
Pp = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]])
Px = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
zero_F = lambda T, X, Y, Z: 0.0


def zero_H(T, X, Y, Z):
    return np.zeros((3, 3) + np.broadcast(T, X, Y, Z).shape)


def zero_N(T, X, Y, Z):
    return np.zeros((3,) + np.broadcast(T, X, Y, Z).shape)


def shape_cos_u(T, X, Y, Z):
    return np.cos(T - Z)


def shape_cos2(T, X, Y, Z):
    return np.cos(2.0 * (T - Z))


def shape_u2(T, X, Y, Z):
    return (T - Z) ** 2


def mk_diag(axes, shape, amp):
    def Hf(T, X, Y, Z, axes=axes, shape=shape, amp=amp):
        prof = amp * shape(T, X, Y, Z)
        sh = np.broadcast(T, X, Y, Z).shape
        H = np.zeros((3, 3) + sh)
        for i in axes:
            H[i, i] = prof
        return H
    return Hf


def mk_tt(pat, shape, amp):
    def Hf(T, X, Y, Z, pat=pat, shape=shape, amp=amp):
        return np.einsum('ij,...->ij...', pat, amp * shape(T, X, Y, Z))
    return Hf


def mk_scalar(shape, amp):
    def Ff(T, X, Y, Z, shape=shape, amp=amp):
        return amp * shape(T, X, Y, Z)
    return Ff


def mk_vec(slot, shape, amp):
    def Nf(T, X, Y, Z, slot=slot, shape=shape, amp=amp):
        sh = np.broadcast(T, X, Y, Z).shape
        Nv = np.zeros((3,) + sh)
        Nv[slot] = amp * shape(T, X, Y, Z)
        return Nv
    return Nf


def field_from_spec(spec, amp):
    kind = spec[0]
    if kind == 'B3':
        return zero_F, mk_diag((0, 1, 2), spec[1], amp), zero_N
    if kind == 'Bxy':
        return zero_F, mk_diag((0, 1), spec[1], amp), zero_N
    if kind == 'Bz':
        return zero_F, mk_diag((2,), spec[1], amp), zero_N
    if kind == 'F':
        return mk_scalar(spec[1], amp), zero_H, zero_N
    if kind == 'N':
        return zero_F, zero_H, mk_vec(spec[1], spec[2], amp)
    if kind == 'TT':
        return zero_F, mk_tt(spec[1], spec[2], amp), zero_N
    raise ValueError(kind)


BASE = (0.0, 0.4, 0.25, 0.0)


def pt_of_u(u):
    return (BASE[0], BASE[1], BASE[2], BASE[3] - u)


def response(spec, u, amp=AMP, eps=0.03):
    x0 = pt_of_u(u)
    F1, H1, N1 = field_from_spec(spec, amp)
    F2, H2, N2 = field_from_spec(spec, amp / 2.0)
    G1 = ein_ext(F1, H1, x0, eps, N1)
    G2 = ein_ext(F2, H2, x0, eps, N2)
    return (4.0 * G2 - G1) / amp


IDX = [("tt", (0, 0)), ("tz", (0, 3)), ("zz", (3, 3)), ("xx", (1, 1)), ("yy", (2, 2)),
       ("xy", (1, 2)), ("xt", (0, 1)), ("yt", (0, 2)), ("xz", (1, 3)), ("yz", (2, 3))]
CIDX = dict(IDX)


def comp_line(M):
    return " ".join("%s=%+.4f" % (tag, M[i, j]) for tag, (i, j) in IDX)


def field_tt(a):
    return zero_F, mk_tt(Pp, shape_cos_u, a), zero_N


def field_r14(a, axes=(0, 1, 2)):
    B2 = 3.0 * a * a / 16.0
    Bu2 = -a * a / 8.0
    def Hf(T, X, Y, Z, a=a, axes=axes, B2=B2, Bu2=Bu2):
        u = T - Z
        prof = B2 * np.cos(2.0 * u) + Bu2 * u * u
        sh = np.broadcast(T, X, Y, Z).shape
        H = np.zeros((3, 3) + sh)
        for i in axes:
            H[i, i] = prof
        return H + np.einsum('ij,...->ij...', Pp, a * np.cos(u))
    return zero_F, Hf, zero_N


def field_asm(a, c=None):
    if c is None:
        B2 = 3.0 * a * a / 16.0
        Bu2 = -a * a / 8.0
        Nz2 = -3.0 * a * a / 32.0
        Nzu2 = a * a / 16.0
    else:
        B2, Bu2, Nz2, Nzu2 = c
    def Hf(T, X, Y, Z, a=a, B2=B2, Bu2=Bu2):
        u = T - Z
        prof = B2 * np.cos(2.0 * u) + Bu2 * u * u
        sh = np.broadcast(T, X, Y, Z).shape
        H = np.zeros((3, 3) + sh)
        for i in range(3):
            H[i, i] = prof
        return H + np.einsum('ij,...->ij...', Pp, a * np.cos(u))
    def Nf(T, X, Y, Z, Nz2=Nz2, Nzu2=Nzu2):
        u = T - Z
        sh = np.broadcast(T, X, Y, Z).shape
        Nv = np.zeros((3,) + sh)
        Nv[2] = Nz2 * np.cos(2.0 * u) + Nzu2 * u * u
        return Nv
    return zero_F, Hf, Nf


def gnorm(field, u, eps=0.03):
    Ff, Hf, Nf = field
    return nrm(ein_ext(Ff, Hf, pt_of_u(u), eps, Nf))


print("=" * 78)
print("G1b 第十五轮：横向/无迹基元的下一层装配（判据 G1b15-1 至 G1b15-6）")
print("=" * 78)

print()
print("[G1b15-1] 器械核验（ein_ext：eps 半步 Richardson）")
zf = ein_ext(zero_F, zero_H, (0.0, 1.10, 0.3, -0.2), 0.05, zero_N)
print("     平坦度规：|G| = %.2e" % nrm(zf))
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


g00 = ein_ext(iso_F, iso_H, (0.0, 1.2, 0.0, 0.0), 0.05, zero_N)[0, 0]
print("     各向同性静态完成场：G_00 = %+.6e（参照 +3.341389e-02，相对差 %.1e）"
      % (g00, abs(g00 - 3.341389e-02) / 3.341389e-02))
print("     [累计 %.1f 秒]" % (time.time() - T_START))

print()
print("[G1b15-2] 基元响应分类（单位幅度，u = 0.5；解析分类见文档头）")
TABLE = [
    ("B3_cos2u", ('B3', shape_cos2)),
    ("B3_u2", ('B3', shape_u2)),
    ("Bxy_cos2u", ('Bxy', shape_cos2)),
    ("Bxy_u2", ('Bxy', shape_u2)),
    ("Bz_cos2u", ('Bz', shape_cos2)),
    ("Bz_u2", ('Bz', shape_u2)),
    ("F_cos2u", ('F', shape_cos2)),
    ("F_u2", ('F', shape_u2)),
    ("Nz_cos2u", ('N', 2, shape_cos2)),
    ("Nz_u2", ('N', 2, shape_u2)),
    ("Nx_cos2u", ('N', 0, shape_cos2)),
    ("Nx_u2", ('N', 0, shape_u2)),
    ("TTp_cos2u", ('TT', Pp, shape_cos2)),
    ("TTp_u2", ('TT', Pp, shape_u2)),
    ("TTx_cos2u", ('TT', Px, shape_cos2)),
    ("TTx_u2", ('TT', Px, shape_u2)),
]
for tag, spec in TABLE:
    r = response(spec, 0.5)
    print("     %-10s %s ｜ |G| = %.2e" % (tag, comp_line(r), nrm(r)))
print("     [累计 %.1f 秒]" % (time.time() - T_START))

print()
print("[G1b15-3] 第十四轮修正场的残余定位（解析：G_xx = G_yy = (3a^2/8)cos2u + a^2/8，其余 0）")
US = (0.4, 0.8, 1.2, 1.6, 2.0, 2.4)
for u in (0.4, 1.2, 2.0):
    Ff, Hf, Nf = field_r14(AMP)
    G = ein_ext(Ff, Hf, pt_of_u(u), 0.03, Nf)
    pred = 3.0 * AMP ** 2 / 8.0 * math.cos(2.0 * u) + AMP ** 2 / 8.0
    oth = max(abs(G[i, j]) for tag, (i, j) in IDX if tag not in ("xx", "yy"))
    print("     u = %.1f：|G| = %.3e ｜ G_xx = %+.4e（解析 %+.4e）｜ G_yy - G_xx = %+.1e ｜ 其余分量最大 %.1e"
          % (u, nrm(G), G[1, 1], pred, G[2, 2] - G[1, 1], oth))
rows, vals = [], []
for u in US:
    Ff, Hf, Nf = field_r14(AMP)
    G = ein_ext(Ff, Hf, pt_of_u(u), 0.03, Nf)
    rows.append([1.0, math.cos(2.0 * u)])
    vals.append(G[1, 1])
sol2, res2, rank2, sv2 = np.linalg.lstsq(np.array(rows), np.array(vals), rcond=None)
ratio2 = float(np.linalg.norm(np.array(rows) @ sol2 - np.array(vals))) / float(np.linalg.norm(np.array(vals)))
print("     两形状拟合（六点）：c0 = %+.5e（解析 %+.5e）｜ c2 = %+.5e（解析 %+.5e）｜ 拟合残差比 = %.1e"
      % (sol2[0], AMP ** 2 / 8.0, sol2[1], 3.0 * AMP ** 2 / 8.0, ratio2))
print("     [累计 %.1f 秒]" % (time.time() - T_START))

print()
print("[G1b15-4] 扩展装配（迹两列 + 矢量槽两列，六点、全分量）")
SP = [('B3', shape_cos2), ('B3', shape_u2), ('N', 2, shape_cos2), ('N', 2, shape_u2)]
RESP = []
GUN = []
for u in US:
    RESP.append([response(sp, u) for sp in SP])
    Ff, Hf, Nf = field_tt(AMP)
    GUN.append(ein_ext(Ff, Hf, pt_of_u(u), 0.03, Nf))
print("     [响应与残余测量完成，累计 %.1f 秒]" % (time.time() - T_START))
rows, vals = [], []
for iu in range(len(US)):
    for tag in ("tt", "tz", "zz"):
        i, j = CIDX[tag]
        rows.append([RESP[iu][0][i, j], RESP[iu][1][i, j]])
        vals.append(-GUN[iu][i, j])
solA, ra, rankA, svA = np.linalg.lstsq(np.array(rows), np.array(vals), rcond=None)
rows, vals = [], []
for iu in range(len(US)):
    for tag in ("xx", "yy"):
        i, j = CIDX[tag]
        rows.append([RESP[iu][2][i, j], RESP[iu][3][i, j]])
        vals.append(-(GUN[iu][i, j] + solA[0] * RESP[iu][0][i, j] + solA[1] * RESP[iu][1][i, j]))
solB, rb, rankB, svB = np.linalg.lstsq(np.array(rows), np.array(vals), rcond=None)
print("     阶段一（null3 块，迹两列）：B2 = %+.6e（解析 %+.6e）｜ Bu2 = %+.6e（解析 %+.6e）"
      % (solA[0], 3 * AMP ** 2 / 16, solA[1], -AMP ** 2 / 8))
print("     阶段二（横向块，矢量槽两列）：Nz2 = %+.6e（解析 %+.6e）｜ Nzu2 = %+.6e（解析 %+.6e）"
      % (solB[0], -3 * AMP ** 2 / 32, solB[1], AMP ** 2 / 16))
rows, vals = [], []
for iu in range(len(US)):
    for tag, (i, j) in IDX:
        rows.append([RESP[iu][c][i, j] for c in range(4)])
        vals.append(-GUN[iu][i, j])
M4 = np.array(rows)
b4 = np.array(vals)
sol4, r4, rank4, sv4 = np.linalg.lstsq(M4, b4, rcond=None)
ratio4 = float(np.linalg.norm(M4 @ sol4 - b4)) / float(np.linalg.norm(b4))
print("     联合四列（与两段拟合同解，供对照）：(%+.3e, %+.3e, %+.3e, %+.3e) ｜ 拟合残差比 = %.1e"
      % (sol4[0], sol4[1], sol4[2], sol4[3], ratio4))
print("     [累计 %.1f 秒]" % (time.time() - T_START))

print()
print("[G1b15-5] 降阶与标度（u = 0.4；a = 1e-2 与 a/2）")
c_fit = (solA[0], solA[1], solB[0], solB[1])
schemes = [
    ("未修正（纯 TT）", lambda a: field_tt(a)),
    ("第十四轮（迹三轴）", lambda a: field_r14(a)),
    ("装配（解析系数）", lambda a: field_asm(a)),
    ("装配（拟合系数）", lambda a, c=c_fit: field_asm(a, tuple(x * (a / AMP) ** 2 for x in c))),
    ("罗森形（迹横向两轴）", lambda a: field_r14(a, axes=(0, 1))),
]
res = []
for name, fn in schemes:
    n1 = gnorm(fn(AMP), 0.4)
    n2 = gnorm(fn(AMP / 2.0), 0.4)
    res.append((name, n1, n2))
base = res[0][1]
print("     %-22s %-11s %-11s %-6s %-8s" % ("方案", "|G|(a)", "|G|(a/2)", "比", "降幅"))
for name, n1, n2 in res:
    print("     %-22s %.3e  %.3e  %6.2f  %8.1f" % (name, n1, n2, n1 / n2, base / n1))
print("     期望标度：未修正与第十四轮按 a^2（比 4.0）；装配与罗森形按 a^4（比 16.0，若按 a^3 则为 8.0）")
n1r = gnorm(field_asm(AMP), 1.2)
n2r = gnorm(field_asm(AMP / 2.0), 1.2)
nr1 = gnorm(field_r14(AMP, axes=(0, 1)), 1.2)
nr2 = gnorm(field_r14(AMP / 2.0, axes=(0, 1)), 1.2)
print("     u = 1.2 复核：装配比 = %.2f ｜ 罗森形比 = %.2f" % (n1r / n2r, nr1 / nr2))
n2x = gnorm(field_asm(2.0 * AMP), 0.4)
n1x = gnorm(field_asm(AMP), 0.4)
nfl = gnorm(field_asm(0.1 * AMP), 0.4)
print("     倍率复核：a = 2e-2 对 1e-2 的比 = %.2f（a^4 期望 16.0）｜ 底噪探针：a = 1e-3 给 |G| = %.2e" % (n2x / n1x, nfl))
print("     [累计 %.1f 秒]" % (time.time() - T_START))

print()
print("[G1b15-6] 登记")
rows = [
    ("其一", "器械核验", "ein_ext（eps 半步 Richardson）：平坦零、各向同性静态回归与第十四轮一致"),
    ("其二", "基元分类", "十六基元全分量读数对解析分类：TT 型恒零；横向块三来源（p_z、F、N_z）同向"),
    ("其三", "残余定位", "第十四轮残余 = (3a^2/8)cos2u + a^2/8 于 (xx,yy)（两形状拟合对解析）"),
    ("其四", "装配", "两段拟合（迹两列 + 矢量槽两列）系数对解析；联合四列同解"),
    ("其五", "降阶", "五方案 |G| 与 a-标度比：装配与罗森形均落 a^4（比 16，a^3 未出现）"),
    ("其六", "余项", "含时源项（携源波包）、一般非平面波构型、强场端"),
]
for a1, a2, a3 in rows:
    print("     %-4s %-8s %s" % (a1, a2, a3))

print()
print("总运行时间 = %.1f 秒" % (time.time() - T_START))
