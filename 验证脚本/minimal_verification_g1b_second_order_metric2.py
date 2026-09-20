# -*- coding: utf-8 -*-
"""
minimal_verification_g1b_second_order_metric2.py

[G1b 第十四轮] 装配核验重做：矢量扇区槽与轮廓基元（判据 G1b14-1 至 G1b14-5）

背景：第十三轮的装配核验留下一个判负登记：TT 波残余的常数部
（q0 = a^2 k^2 / 4）在器械的对角类（h00 与 hij 的常数与 cos2u 两形状）内无对应项。
本轮按裁决做两件：给器械加矢量扇区槽（g0i，移位向量），并加轮廓基元（u^2 形迹扰动，
其线性响应对 u 恒定），重做装配核验。

解析预期：迹扰动 hij = B(u) delta 的线性响应为 -B''(u)(1, 1, -1)（u = t - z），
故恒定部与 cos2u 部各需 B_{u2}(u) = -q0 u^2/2 与 B_2 = 3 a^2 k^2 / 16。

判据
G1b14-1 器械扩展标定（含 g0i 槽）：平坦零、各向同性静态回归一致
G1b14-2 矢量扇区响应：Nz = cos2u 与 Nz = u^2 两形状的三分量响应
G1b14-3 轮廓基元响应：迹扇区 u^2 形响应对 u 恒定，与解析式 -2(1,1,-1) 对照
G1b14-4 装配核验：扩展基元下的最小二乘、修正后 |G| 降阶、系数对解析值
G1b14-5 登记

运行：python 验证脚本/minimal_verification_g1b_second_order_metric2.py（确定性，约 1 分钟）
"""
import math
import time
import numpy as np

T0 = time.time()
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




def nrm(M):
    return float(np.sqrt(np.sum(np.abs(M) ** 2)))


AMP = 1e-2
Pp = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]])
zero_F = lambda T, X, Y, Z: 0.0
zero_H = lambda T, X, Y, Z: np.zeros((3, 3) + np.broadcast(T, X, Y, Z).shape)
zero_N = lambda T, X, Y, Z: np.zeros((3,) + np.broadcast(T, X, Y, Z).shape)


def H_tt(T, X, Y, Z, a=AMP):
    prof = a * np.cos(1.0 * Z - 1.0 * T)
    return np.einsum('ij,...->ij...', Pp, prof)


def null3(G4):
    return np.array([G4[0, 0], G4[3, 3], G4[0, 3]])


print("=" * 78)
print("G1b 第十四轮：装配核验重做（矢量扇区槽与轮廓基元，判据 G1b14-1 至 G1b14-5）")
print("=" * 78)
P0 = (0.35, 0.4, 0.25, 0.35)

print()
print("[G1b14-1] 器械扩展标定")
z = einstein_at4(zero_F, zero_H, (0.0, 1.10, 0.3, -0.2), 0.05, Nf=zero_N)
print("     平坦度规（含 g0i 槽）：|G| = %.2e" % nrm(z))
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


g1 = einstein_at4(iso_F, iso_H, (0.0, 1.2, 0.0, 0.0), 0.05, Nf=zero_N)
g2 = einstein_at4(iso_F, iso_H, (0.0, 1.2, 0.0, 0.0), 0.025, Nf=zero_N)
g00 = (16.0 * g2[0, 0] - g1[0, 0]) / 15.0
print("     各向同性静态完成场：G_00 = %+.6e（静态机 +3.341389e-02，相对差 %.1e）"
      % (g00, abs(g00 - 3.341389e-02) / 3.341389e-02))

AMP_R = 1e-2
print()
print("[G1b14-2] 矢量扇区响应（幅度 1e-2，已归一）")
def Nz_cos(T, X, Y, Z):
    sh = np.broadcast(T, X, Y, Z).shape
    Nv = np.zeros((3,) + sh)
    Nv[2] = AMP_R * np.cos(2.0 * (T - Z))
    return Nv


def Nz_u2(T, X, Y, Z):
    sh = np.broadcast(T, X, Y, Z).shape
    Nv = np.zeros((3,) + sh)
    Nv[2] = AMP_R * ((T - Z) ** 2)
    return Nv


for tag, Nf in (("Nz = cos2u", Nz_cos), ("Nz = u^2", Nz_u2)):
    G4 = einstein_at4(zero_F, zero_H, P0, 0.03, Nf=Nf)
    v = null3(G4) / AMP_R
    print("     %s：三分量响应 = (%+.4e, %+.4e, %+.4e) ｜ |G| = %.3e" % (tag, v[0], v[1], v[2], nrm(G4) / AMP_R))

print()
print("[G1b14-3] 轮廓基元响应（迹扇区 u^2 形）")
def B_u2(T, X, Y, Z):
    prof = AMP_R * ((T - Z) ** 2)
    sh = np.broadcast(T, X, Y, Z).shape
    H = np.zeros((3, 3) + sh)
    for i in range(3):
        H[i, i] = prof
    return H


def B_cos(T, X, Y, Z):
    prof = AMP_R * np.cos(2.0 * (T - Z))
    sh = np.broadcast(T, X, Y, Z).shape
    H = np.zeros((3, 3) + sh)
    for i in range(3):
        H[i, i] = prof
    return H


Rb = null3(einstein_at4(zero_F, B_cos, P0, 0.03, Nf=zero_N)) / AMP_R
Ru = null3(einstein_at4(zero_F, B_u2, P0, 0.03, Nf=zero_N)) / AMP_R
print("     B = cos2u：(%+.4e, %+.4e, %+.4e)（解析 -B'' = +4cos2u ⇒ (4, 4, -4)）" % (Rb[0], Rb[1], Rb[2]))
print("     B = u^2  ：(%+.4e, %+.4e, %+.4e)（解析 -B'' = -2 ⇒ (-2, -2, +2)）" % (Ru[0], Ru[1], Ru[2]))

print()
print("[G1b14-4] 装配核验（扩展基元）")
T0p, X0p, Y0p = 0.0, 0.4, 0.25


def residual_null(u, a=AMP):
    Zp = T0p - u
    return null3(einstein_at4(zero_F, lambda T, X, Y, Z, aa=a: H_tt(T, X, Y, Z, aa), (T0p, X0p, Y0p, Zp), 0.03, Nf=zero_N))


US = (0.0, 0.4, 0.8, 1.2, 1.6, 2.0)
Mrows, brows = [], []
for u in US:
    r = residual_null(u)
    Mrows.append([Rb[0] * math.cos(2 * u), Ru[0], 0.0, 0.0])
    brows.append(-r[0])
sol, res, rank, sv = np.linalg.lstsq(np.array(Mrows), np.array(brows), rcond=None)
B2, Bu2 = sol[0], sol[1]
print("     求解（G00 分量，六点）：B2 = %+.5e（解析 %+.5e）｜ B_{u2} = %+.5e（解析 %+.5e）"
      % (B2, 3 * AMP ** 2 / 16.0, Bu2, -AMP ** 2 / 8.0))
print("     拟合残差比 = %.2e" % (float(np.linalg.norm(np.array(Mrows) @ sol - np.array(brows))) / float(np.linalg.norm(brows))))

def F_c(T, X, Y, Z, B2=B2, Bu2=Bu2):
    return 0.0


def H_c(T, X, Y, Z, B2=B2, Bu2=Bu2):
    prof = B2 * np.cos(2.0 * (T - Z)) + Bu2 * ((T - Z) ** 2)
    sh = np.broadcast(T, X, Y, Z).shape
    H = np.zeros((3, 3) + sh)
    for i in range(3):
        H[i, i] = prof
    return H + H_tt(T, X, Y, Z)


for u in (0.4, 1.2, 2.0):
    Zp = T0p - u
    g_un = einstein_at4(zero_F, lambda T, X, Y, Z: H_tt(T, X, Y, Z), (T0p, X0p, Y0p, Zp), 0.03, Nf=zero_N)
    g_co = einstein_at4(F_c, H_c, (T0p, X0p, Y0p, Zp), 0.03, Nf=zero_N)
    print("     u = %.1f：未修正 |G| = %.3e ｜ 修正后 |G| = %.3e ｜ 降幅 = %.1f 倍"
          % (u, nrm(g_un), nrm(g_co), nrm(g_un) / max(nrm(g_co), 1e-30)))

print("     幅度标度：")
def corrected_norm(a):
    r = residual_null(0.4, a)
    B2a = 3.0 * a ** 2 / 16.0
    Bu2a = -a ** 2 / 8.0
    def Htot(T, X, Y, Z, aa=a, b2=B2a, bu=Bu2a):
        prof = b2 * np.cos(2.0 * (T - Z)) + bu * ((T - Z) ** 2)
        sh = np.broadcast(T, X, Y, Z).shape
        H = np.zeros((3, 3) + sh)
        for i in range(3):
            H[i, i] = prof
        return H + H_tt(T, X, Y, Z, aa)
    Zp = T0p - 0.4
    return nrm(einstein_at4(F_c, Htot, (T0p, X0p, Y0p, Zp), 0.03, Nf=zero_N))
n1 = corrected_norm(AMP)
n2 = corrected_norm(AMP / 2.0)
print("       用解析系数修正后：a = %.4f 给 |G| = %.3e ｜ a = %.4f 给 %.3e ｜ 比 = %.2f（按 a^3 应为 8）"
      % (AMP, n1, AMP / 2.0, n2, n1 / n2))
print("       （注：解析系数只对零向三分量成立；修正自身在横向分量上产生响应，故总 |G| 仍按 a^2）")

print()
print("[G1b14-5] 登记")
rows = [
    ("其一", "器械扩展", "加矢量扇区槽 g0i 后：平坦零、各向同性静态回归一致"),
    ("其二", "矢量响应", "Nz 两形状的三分量响应读数（供登记；本轮装配未用）"),
    ("其三", "轮廓基元", "迹扇区 u^2 形响应对 u 恒定，与解析 -2(1,1,-1) 对照"),
    ("其四", "装配", "扩展基元下拟合残差与修正后降阶；系数对解析值（3a^2/16 与 -a^2/8）"),
    ("其五", "余项", "含时源项（携源波包）、强场端、矢量扇区的非线性结构"),
]
for a1, a2, a3 in rows:
    print("     %-4s %-8s %s" % (a1, a2, a3))

print()
print("总运行时间 = %.1f 秒" % (time.time() - T0))
