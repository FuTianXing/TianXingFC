# -*- coding: utf-8 -*-
"""
g1b_moving_source.py

[G1b 第十六轮] 动源完成形式首判：平移源的 O(v) 扇区与拖曳轮廓（判据 G16-1 至 G16-6）

背景：静态完成式（F = 2 th_cs，H = (2 th_cs + 3.5 th_cs^2) delta）与矢量规则
（h_0i = 4 Theta_i，K 轮）分别建立在两条线上，从未在同一构型里对撞。本轮把源以恒定
速度平移，抽取 G 的奇部（O(v)）与偶部（O(v^2)），做三件事：O(v) 扇区的分量结构；
G_0i/G_00 对 v_i 的无量纲比值与拖曳轮廓的判别（三个候选：线性剖面、完成式剖面、助推
配方）；O(v^2) 偶部的量级与标度。附带平面波扇区一般性三项复核（P_x 偏振、双频轮廓、
分裂无关）。

解析预期：
1) 平移源的 O(v) 度量 = 静态完成式的助推：h_00 = 2 th_cs，h_ij = (2 th_cs + 3.5 th_cs^2) delta，
   h_0i = -v_i (F + Gv_ii) = -v_i (4 th_cs + 3.5 th_cs^2)（助推的坐标变换给负号）。
2) 若线性层即广义相对论线性层，则助推解满足场方程：G_0i/G_00 = T_0i/T_00 = -v_i，与半径
   无关（源剖面的形状被同一因子带出）。三个拖曳候选彼此相差 O(theta) ~ 3%，比值曲线
   随半径的形状可判别哪一个满足 2)。
3) O(v^2) 偶部的量级与 v^2 标度（档比期望 4）；其完整解释留待洛伦兹助推候选。

判据
G16-1 器械与基线：平坦零、v = 0 静态回归
G16-2 O(v) 分量结构：奇部只有 0x 分量
G16-3 比值与轮廓判别：七候选乘三半径的比值表
G16-4 O(v^2) 偶部：量级与 v^2 标度
G16-5 附带：平面波一般性（P_x、双频、分裂）
G16-6 登记

运行：python 验证脚本/g1b_moving_source.py（确定性，约 1 分钟）
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


MASS, SIG = 0.05, 1.0
erf = np.vectorize(math.erf)


def th_lin(R):
    x = R / (SIG * math.sqrt(2.0))
    t = np.where(x < 1e-8, MASS * math.sqrt(2.0 / math.pi) / SIG,
                 MASS * erf(x) / np.maximum(R, 1e-30))
    return t


def th_cs(R):
    t = th_lin(R)
    return t - t * t


def field_move(vvec, drag='none', sgn=-1.0):
    vx, vy, vz = vvec

    def R_of(T, X, Y, Z):
        Xs = X - vx * T
        Ys = Y - vy * T
        Zs = Z - vz * T
        return np.sqrt(Xs * Xs + Ys * Ys + Zs * Zs)

    def Ff(T, X, Y, Z):
        return 2.0 * th_cs(R_of(T, X, Y, Z))

    def Hf(T, X, Y, Z):
        t = th_cs(R_of(T, X, Y, Z))
        Gv = 2.0 * t + 3.5 * t * t
        sh = np.broadcast(T, X, Y, Z).shape
        H = np.zeros((3, 3) + sh)
        for i in range(3):
            H[i, i] = Gv
        return H

    def Nf(T, X, Y, Z):
        sh = np.broadcast(T, X, Y, Z).shape
        Nv = np.zeros((3,) + sh)
        if drag == 'none':
            return Nv
        R = R_of(T, X, Y, Z)
        if drag == 'lin':
            g = th_lin(R)
        elif drag == 'cs':
            g = th_cs(R)
        elif drag == 'full':
            t = th_cs(R)
            g = t + 0.875 * t * t
        else:
            raise ValueError(drag)
        for i in range(3):
            Nv[i] = sgn * 4.0 * g * vvec[i]
        return Nv

    return Ff, Hf, Nf


def G_at(field, x0, eps=0.03):
    Ff, Hf, Nf = field
    return ein_ext(Ff, Hf, x0, eps, Nf)


IDX = [("tt", (0, 0)), ("tz", (0, 3)), ("zz", (3, 3)), ("xx", (1, 1)), ("yy", (2, 2)),
       ("xy", (1, 2)), ("xt", (0, 1)), ("yt", (0, 2)), ("xz", (1, 3)), ("yz", (2, 3))]


def comp_line(M):
    return " ".join("%s=%+.4e" % (tag, M[i, j]) for tag, (i, j) in IDX)


def odd_even(vdir, drag, sgn, r, V0):
    x0 = (0.0, r, 0.0, 0.0)
    vp = (V0 * vdir[0], V0 * vdir[1], V0 * vdir[2])
    vm = (-vp[0], -vp[1], -vp[2])
    Gp = G_at(field_move(vp, drag, sgn), x0)
    Gm = G_at(field_move(vm, drag, sgn), x0)
    G0 = G_at(field_move((0.0, 0.0, 0.0), drag, sgn), x0)
    return G0, (Gp - Gm) / 2.0, (Gp + Gm) / 2.0 - G0


print("=" * 78)
print("G1b 第十六轮：动源完成形式首判：平移源的 O(v) 扇区与拖曳轮廓（判据 G16-1 至 G16-6）")
print("=" * 78)

print()
print("[G16-1] 器械与基线（v = 0 静态回归）")
zf = ein_ext(zero_F, zero_H, (0.0, 1.10, 0.3, -0.2), 0.05, zero_N)
print("     平坦度规：|G| = %.2e" % nrm(zf))
G0s = G_at(field_move((0.0, 0.0, 0.0)), (0.0, 1.2, 0.0, 0.0), eps=0.05)
print("     静态完成场：G_00 = %+.6e（参照 +3.341389e-02，相对差 %.1e）"
      % (G0s[0, 0], abs(G0s[0, 0] - 3.341389e-02) / 3.341389e-02))
print("     [累计 %.1f 秒]" % (time.time() - T_START))

V = 0.05
print()
print("[G16-2] O(v) 分量结构（full, sgn=-1，v = %.2f 沿 x，r = 1.2；归一 = G_00 v）" % V)
G0, odd, ev = odd_even((1.0, 0.0, 0.0), 'full', -1.0, 1.2, V)
norm = G0[0, 0] * V
print("     G_00 = %+.5e ｜ 归一 = %.4e" % (G0[0, 0], norm))
print("     奇部/归一：" + comp_line(odd / norm))
oth = max(abs(odd[i, j]) for tag, (i, j) in IDX if tag != "xt")
print("     非 0x 分量的最大绝对值 |奇部| = %.2e（相对信号 %.2e）" % (oth, oth / abs(odd[0, 1])))
print("     [累计 %.1f 秒]" % (time.time() - T_START))

print()
print("[G16-3] 比值表：ratio(r) = 奇部 G_0x / (G_00 v)（期望 -1 且与 r 无关）")
VARIANTS = [("无拖曳", 'none', 0.0), ("lin+", 'lin', 1.0), ("lin-", 'lin', -1.0),
            ("cs+", 'cs', 1.0), ("cs-", 'cs', -1.0), ("full+", 'full', 1.0), ("full-", 'full', -1.0)]
RS = (0.8, 1.6, 2.4)
for name, drag, sgn in VARIANTS:
    row = []
    for r in RS:
        G0v, oddv, evv = odd_even((1.0, 0.0, 0.0), drag, sgn, r, V)
        row.append(oddv[0, 1] / (G0v[0, 0] * V))
    print("     %-6s ratio(0.8, 1.6, 2.4) = (%+.4f, %+.4f, %+.4f) ｜ 散布 = %.1e"
          % (name, row[0], row[1], row[2], max(row) - min(row)))
print("     [累计 %.1f 秒]" % (time.time() - T_START))

print()
print("[G16-4] O(v^2) 偶部（full, sgn=-1，r = 1.2）")
G0a, odda, eva = odd_even((1.0, 0.0, 0.0), 'full', -1.0, 1.2, V)
mxa = max(abs(eva[i, j]) for tag, (i, j) in IDX)
tagm = [tag for tag, (i, j) in IDX if abs(eva[i, j]) == mxa][0]
G0b, oddb, evb = odd_even((1.0, 0.0, 0.0), 'full', -1.0, 1.2, V / 2.0)
mxb = max(abs(evb[i, j]) for tag, (i, j) in IDX)
print("     max |偶部| = %.3e（在 %s；相对 G_00 = %.2e）" % (mxa, tagm, mxa / abs(G0a[0, 0])))
print("     档比（v = %.3f 对 %.3f）= %.2f（O(v^2) 期望 4）" % (V, V / 2.0, mxa / mxb))
print("     [累计 %.1f 秒]" % (time.time() - T_START))

print()
print("[G16-5] 附带：平面波扇区一般性三项（a = 1e-2 与 a/2，u = 0.4）")
BASE = (0.0, 0.4, 0.25, 0.0)


def pt_of_u(u):
    return (BASE[0], BASE[1], BASE[2], BASE[3] - u)


def field_ttx(a, omega=1.0):
    def Hf(T, X, Y, Z, a=a, omega=omega):
        u = T - Z
        A = a * np.cos(omega * u)
        Bv = (3.0 * a * a / 16.0) * np.cos(2.0 * omega * u) - (a * a * omega * omega / 8.0) * u * u
        sh = np.broadcast(T, X, Y, Z).shape
        H = np.zeros((3, 3) + sh)
        H[0, 1] = A
        H[1, 0] = A
        H[0, 0] = Bv
        H[1, 1] = Bv
        return H
    return zero_F, Hf, zero_N


def field_tt_split(a):
    def Hf(T, X, Y, Z, a=a):
        u = T - Z
        A = a * np.cos(u)
        Bv = (3.0 * a * a / 16.0) * np.cos(2.0 * u) - (a * a / 8.0) * u * u
        sh = np.broadcast(T, X, Y, Z).shape
        H = np.zeros((3, 3) + sh)
        H[0, 0] = A
        H[1, 1] = -A + 2.0 * Bv
        return H
    return zero_F, Hf, zero_N


A0 = 1e-2
for name, mk in [("P_x 偏振", lambda aa: field_ttx(aa)),
                 ("双频（omega=2）", lambda aa: field_ttx(aa, omega=2.0)),
                 ("分裂（p=0, q=2B）", lambda aa: field_tt_split(aa))]:
    n1 = nrm(G_at(mk(A0), pt_of_u(0.4), eps=0.03))
    n2 = nrm(G_at(mk(A0 / 2.0), pt_of_u(0.4), eps=0.03))
    print("     %-18s |G|(a) = %.3e ｜ |G|(a/2) = %.3e ｜ 比 = %.2f（a^4 期望 16）"
          % (name, n1, n2, n1 / max(n2, 1e-30)))
print("     [累计 %.1f 秒]" % (time.time() - T_START))

print()
print("[G16-6] 登记")
rows = [
    ("其一", "基线", "平坦零；v = 0 静态回归对参照值"),
    ("其二", "O(v) 结构", "奇部只落 0x 分量；其余分量读数（供登记）"),
    ("其三", "比值判别", "七候选乘三半径的比值表：哪个轮廓给常数比 -1"),
    ("其四", "O(v^2)", "偶部量级与档比；完整解释留待洛伦兹助推候选"),
    ("其五", "附带", "P_x、双频、分裂三项的 |G| 与 a-标度"),
    ("其六", "余项", "洛伦兹助推候选、源扇区洛伦兹破坏系数、一般非平面波构型"),
]
for a1, a2, a3 in rows:
    print("     %-4s %-8s %s" % (a1, a2, a3))

print()
print("总运行时间 = %.1f 秒" % (time.time() - T_START))
