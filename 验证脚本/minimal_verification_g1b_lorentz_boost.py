# -*- coding: utf-8 -*-
"""
minimal_verification_g1b_lorentz_boost.py

[G1b 第十七轮] 洛伦兹助推候选的完整形式：助推核验、比值解析式与 O(theta) 偏置剖分（判据 G17-1 至 G17-5）

背景：第十六轮把平移源的 O(v) 扇区做通，但比分 G_0x/G_00 对朴素期望 -v 留下 O(theta)
偏置（~0.3 theta）待剖分，助推候选（伽利略对洛伦兹）也未分岔。本轮给出洛伦兹助推的
完整形式：把静态完成式按真洛伦兹变换搬到实验室系（gamma 因子、轮廓收缩、时空混合），
然后做三件：其一，助推构造核验（L 的机器爱因斯坦张量对"静态张量之助推"逐分量一致）；
其二，伽利略候选的 v^2 偏差（与 L 的差按 v^2 标度）；其三，比值的解析式
R = -v (G'_tt + G'_xx) / (G'_tt + v^2 G'_xx)，与"朴素 -v"的偏离即 G'_xx/G'_tt，O(theta)
偏置的就是静态张量自己的应力/能量比。

解析预期：
1) 精确解构造：均匀平移源的解是静态解的洛伦兹助推（连续极限的洛伦兹协变性）。实验室系
   度规 = M^T g' M，M 为 d x'/d x（沿 x 的助推，参数 gamma）。
2) 助推核验：G_{mu nu}(L, x) = M^a_mu M^b_nu G'_{ab}(x')，x' = 助推坐标。若成立，L 的场方程
   质量与静态解同档（~1e-7 相对），伽利略候选只在 O(v^2) 内成立。
3) 比值：R = G_0x/G_00 = -v (G'_tt + G'_xx)/(G'_tt + v^2 G'_xx)。v -> 0 时
   R/(-v) = 1 + G'_xx/G'_tt：偏离由静态解的应力分量给出，即第十六轮的 O(theta) 偏置的
   候选解释之一（分母项），可直接判定。

判据
G17-1 器械与基线：平坦零、v = 0 静态回归
G17-2 助推构造核验：三档速度下 L 的机器 G 对静态 G 之助推（张量变换）
G17-3 伽利略候选的 v^2 偏差：delta G = G(伽利略) - G(L) 的标度与量级
G17-4 比值解析式：R_meas 对 R_pred 与朴素 -v；G'_xx/G'_tt 的对照
G17-5 登记

运行：python 验证脚本/minimal_verification_g1b_lorentz_boost.py（确定性，约 1 分钟）
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


def field_static():
    def Ff(T, X, Y, Z):
        R = np.sqrt(X * X + Y * Y + Z * Z)
        return 2.0 * th_cs(R)

    def Hf(T, X, Y, Z):
        R = np.sqrt(X * X + Y * Y + Z * Z)
        t = th_cs(R)
        Gv = 2.0 * t + 3.5 * t * t
        sh = np.broadcast(T, X, Y, Z).shape
        H = np.zeros((3, 3) + sh)
        for i in range(3):
            H[i, i] = Gv
        return H

    return Ff, Hf, zero_N


def field_boost(v):
    gamma = 1.0 / math.sqrt(1.0 - v * v)

    def stat_of(xp, yp, zp):
        rp = np.sqrt(xp * xp + yp * yp + zp * zp)
        t = th_cs(rp)
        Fp = 2.0 * t
        Gvp = 2.0 * t + 3.5 * t * t
        gtt = -(1.0 - Fp)
        gxx = 1.0 + Gvp
        return gtt, gxx, Gvp

    def coords(T, X, Y, Z):
        tp = gamma * (T - v * X)
        xp = gamma * (X - v * T)
        return tp, xp, Y, Z

    def Ff(T, X, Y, Z):
        tp, xp, yp, zp = coords(T, X, Y, Z)
        gtt, gxx, Gvp = stat_of(xp, yp, zp)
        return 1.0 + gamma * gamma * (gtt + v * v * gxx)

    def Hf(T, X, Y, Z):
        tp, xp, yp, zp = coords(T, X, Y, Z)
        gtt, gxx, Gvp = stat_of(xp, yp, zp)
        sh = np.broadcast(T, X, Y, Z).shape
        H = np.zeros((3, 3) + sh)
        H[0, 0] = gamma * gamma * (v * v * gtt + gxx) - 1.0
        H[1, 1] = Gvp
        H[2, 2] = Gvp
        return H

    def Nf(T, X, Y, Z):
        tp, xp, yp, zp = coords(T, X, Y, Z)
        gtt, gxx, Gvp = stat_of(xp, yp, zp)
        sh = np.broadcast(T, X, Y, Z).shape
        Nv = np.zeros((3,) + sh)
        Nv[0] = -gamma * gamma * v * (gtt + gxx)
        return Nv

    return Ff, Hf, Nf


def field_galilean(v):
    def Ff(T, X, Y, Z):
        R = np.sqrt((X - v * T) ** 2 + Y * Y + Z * Z)
        return 2.0 * th_cs(R)

    def Hf(T, X, Y, Z):
        R = np.sqrt((X - v * T) ** 2 + Y * Y + Z * Z)
        t = th_cs(R)
        Gv = 2.0 * t + 3.5 * t * t
        sh = np.broadcast(T, X, Y, Z).shape
        H = np.zeros((3, 3) + sh)
        for i in range(3):
            H[i, i] = Gv
        return H

    def Nf(T, X, Y, Z):
        R = np.sqrt((X - v * T) ** 2 + Y * Y + Z * Z)
        t = th_cs(R)
        g = t + 0.875 * t * t
        sh = np.broadcast(T, X, Y, Z).shape
        Nv = np.zeros((3,) + sh)
        Nv[0] = -4.0 * g * v
        return Nv

    return Ff, Hf, Nf


def G_at(field, x0, eps=0.03):
    Ff, Hf, Nf = field
    return ein_ext(Ff, Hf, x0, eps, Nf)


def boost_matrix(v):
    gamma = 1.0 / math.sqrt(1.0 - v * v)
    return np.array([[gamma, -gamma * v, 0.0, 0.0],
                     [-gamma * v, gamma, 0.0, 0.0],
                     [0.0, 0.0, 1.0, 0.0],
                     [0.0, 0.0, 0.0, 1.0]])


IDX = [("tt", (0, 0)), ("tz", (0, 3)), ("zz", (3, 3)), ("xx", (1, 1)), ("yy", (2, 2)),
       ("xy", (1, 2)), ("xt", (0, 1)), ("yt", (0, 2)), ("xz", (1, 3)), ("yz", (2, 3))]


print("=" * 78)
print("G1b 第十七轮：洛伦兹助推候选的完整形式（判据 G17-1 至 G17-5）")
print("=" * 78)

print()
print("[G17-1] 器械与基线")
zf = ein_ext(zero_F, zero_H, (0.0, 1.10, 0.3, -0.2), 0.05, zero_N)
print("     平坦度规：|G| = %.2e" % nrm(zf))
G0s = G_at(field_static(), (0.0, 1.2, 0.0, 0.0), eps=0.05)
print("     静态完成场：G_00 = %+.6e（参照 +3.341389e-02，相对差 %.1e）"
      % (G0s[0, 0], abs(G0s[0, 0] - 3.341389e-02) / 3.341389e-02))
print("     [累计 %.1f 秒]" % (time.time() - T_START))

R_MAIN = 1.2
print()
print("[G17-2] 助推构造核验（L 的机器 G 对静态 G 之助推；r = %.1f）" % R_MAIN)
for v in (0.05, 0.1, 0.2):
    gamma = 1.0 / math.sqrt(1.0 - v * v)
    x0 = (0.0, R_MAIN, 0.0, 0.0)
    GL = G_at(field_boost(v), x0)
    tp = gamma * (0.0 - v * R_MAIN)
    xp = gamma * (R_MAIN - 0.0)
    Gst = G_at(field_static(), (tp, xp, 0.0, 0.0))
    M = boost_matrix(v)
    Gpred = M.T @ Gst @ M
    d = GL - Gpred
    print("     v = %.2f：max|G_L - M^T G' M| = %.2e ｜ 相对 |G_L| = %.2e ｜ |G_L| = %.3e"
          % (v, float(np.max(np.abs(d))), float(np.max(np.abs(d))) / nrm(GL), nrm(GL)))
print("     [累计 %.1f 秒]" % (time.time() - T_START))

print()
print("[G17-3] 伽利略候选的 v^2 偏差（delta G = G(伽利略) - G(L)；r = %.1f）" % R_MAIN)
prev = None
for v in (0.05, 0.1, 0.2):
    x0 = (0.0, R_MAIN, 0.0, 0.0)
    GL = G_at(field_boost(v), x0)
    GG = G_at(field_galilean(v), x0)
    d = GG - GL
    mx = float(np.max(np.abs(d)))
    line = "     v = %.2f：max|delta G| = %.3e ｜ 相对 |G_L| = %.2e" % (v, mx, mx / nrm(GL))
    if prev is not None:
        line += " ｜ 档比 = %.2f（v^2 期望 4）" % (mx / prev)
    prev = mx
    print(line)
print("     [累计 %.1f 秒]" % (time.time() - T_START))

print()
print("[G17-4] 比值解析式（R = G_0x/G_00；R_pred = -v (G'_tt + G'_xx)/(G'_tt + v^2 G'_xx)）")
for v in (0.05, 0.1):
    gamma = 1.0 / math.sqrt(1.0 - v * v)
    for r in (0.8, 1.2, 1.6, 2.4):
        x0 = (0.0, r, 0.0, 0.0)
        GL = G_at(field_boost(v), x0)
        Rm = GL[0, 1] / GL[0, 0]
        tp = gamma * (0.0 - v * r)
        xp = gamma * (r - 0.0)
        Gst = G_at(field_static(), (tp, xp, 0.0, 0.0))
        Rd = -v * (Gst[0, 0] + Gst[1, 1]) / (Gst[0, 0] + v * v * Gst[1, 1])
        print("     v = %.2f, r = %.1f：R_meas = %+.6f ｜ R_pred = %+.6f ｜ R/(-v) - 1 = %+.5f ｜ G'_xx/G'_tt = %+.5f"
              % (v, r, Rm, Rd, Rm / (-v) - 1.0, Gst[1, 1] / Gst[0, 0]))
print("     [累计 %.1f 秒]" % (time.time() - T_START))

print()
print("[G17-5] 登记")
rows = [
    ("其一", "基线", "平坦零；v = 0 静态回归对参照值"),
    ("其二", "助推核验", "三档速度下 L 的机器 G 与静态 G 之助推逐分量一致（相对差读数）"),
    ("其三", "分岔", "伽利略对洛伦兹的偏差按 v^2 标度（档比读数）"),
    ("其四", "比值", "R 对 R_pred 与朴素 -v；偏离与 G'_xx/G'_tt 的对照（O(theta) 偏置剖分）"),
    ("其五", "余项", "有限盒与离散归一残差、一般非平面波构型、强场端"),
]
for a1, a2, a3 in rows:
    print("     %-4s %-8s %s" % (a1, a2, a3))

print()
print("总运行时间 = %.1f 秒" % (time.time() - T_START))
