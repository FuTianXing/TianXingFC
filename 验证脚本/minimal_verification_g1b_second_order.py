# -*- coding: utf-8 -*-
"""
minimal_verification_g1b_second_order.py

[G1b 首轮] 非线性封装的第一道核对：静球对称的二阶靶子与判定项
（判据 G1b-1 至 G1b-6）

背景：G1b（完整形态/非线性封装）是引力线最后一项。线性层结论：h00 = 2θ、
hij = h00 δij（λ = 1）、G00 = −2∇²θ = 8πGρ/c²。本轮把"非线性要补什么"
第一次量化为靶子：

（一）精确真空解（各向同性坐标施瓦西，教科书结果）的二阶展开给出
    h00 = h_lin − (1/2) h_lin² + O(h³)、hij = h_lin + (3/8) h_lin² + O(h³)
    （h_lin = r_s/r），等价地 hij = h00 + (7/8) h00²；
（二）用局部补丁上的四阶有限差分实现爱因斯坦张量机器：
    平坦零残差、精确真空解零残差（四阶收敛）、线性解对照 −2∇²θ、
    收缩比安基恒等式 ∇μG_{μν} = 0；
（三）量出线性同幅 ansatz（hij ≡ h00）在二阶的残差系数。

判据
G1b-1 FD 机制：平坦零残差；精确真空解残差按 eps 四阶收敛
G1b-2 线性核对：G00/(−2∇²θ) 趋于 1
G1b-3 比安基核对：一般静态球对称 F、G 下 ∇μ G_{μν} = 0（FD 精度）
G1b-4 二阶靶子（代数精确）：系数 −1/2 与 +3/8（等价 7/8）
G1b-5 线性 ansatz 的二阶残差系数
G1b-6 判定项清单（打印）

运行：python 验证脚本/minimal_verification_g1b_second_order.py（确定性，约 10 秒）
"""
import math
import time
import numpy as np

T0 = time.time()
MU = 1.0
EPS = 0.05
HALF = 4
C = HALF
NP_ = 2 * HALF + 1


def g_patch(Ff, Gf, x0, eps):
    k = np.arange(-HALF, HALF + 1) * eps
    X = x0[0] + k[:, None, None]
    Y = x0[1] + k[None, :, None]
    Z = x0[2] + k[None, None, :]
    r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    g = np.zeros((4, 4, NP_, NP_, NP_))
    g[0, 0] = -(1.0 - Ff(r))
    for i in (1, 2, 3):
        g[i, i] = 1.0 + Gf(r)
    return g


def d1_keep(a, axis, eps):
    b = np.moveaxis(a, axis, -1)
    out = np.full_like(b, np.nan)
    out[..., 2:-2] = (b[..., :-4] - 8 * b[..., 1:-3] + 8 * b[..., 3:-1] - b[..., 4:]) / (12 * eps)
    return np.moveaxis(out, -1, axis)


def einstein_at(Ff, Gf, x0, eps=EPS, want_bianchi=False):
    g = g_patch(Ff, Gf, x0, eps)
    dgf = [np.zeros_like(g), d1_keep(g, 2, eps), d1_keep(g, 3, eps), d1_keep(g, 4, eps)]
    gg = g[:, :, C, C, C]
    ginv = np.zeros((4, 4))
    ginv[0, 0] = 1.0 / gg[0, 0]
    for i in (1, 2, 3):
        ginv[i, i] = 1.0 / gg[i, i]
    ginvf = np.zeros_like(g)
    ginvf[0, 0] = 1.0 / g[0, 0]
    for i in (1, 2, 3):
        ginvf[i, i] = 1.0 / g[i, i]
    Gam = np.zeros((4, 4, 4, NP_, NP_, NP_))
    for lam in range(4):
        for mu in range(4):
            for nu in range(mu, 4):
                acc = np.zeros((NP_, NP_, NP_))
                for sig in range(4):
                    if lam == sig:
                        acc = acc + ginvf[lam, sig] * 0.5 * (
                            dgf[mu][sig, nu] + dgf[nu][sig, mu] - dgf[sig][mu, nu])
                Gam[lam, mu, nu] = acc
                Gam[lam, nu, mu] = acc
    G0 = Gam[:, :, :, C, C, C]
    dG = np.zeros((4, 4, 4, 4))
    for k in range(3):
        for lam in range(4):
            for mu in range(4):
                for nu in range(4):
                    dG[lam, mu, nu, k + 1] = d1_keep(Gam[lam, mu, nu], k, eps)[C, C, C]
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
    Gmn = Ric - 0.5 * Rsc * gg
    if not want_bianchi:
        return Gmn, gg, ginv, G0
    # G^{mu nu} and its derivatives for the contracted Bianchi identity
    def Gup(Ff2, Gf2, x):
        G2, g2, gi2, _ = einstein_at(Ff2, Gf2, x, eps, want_bianchi=False)
        return gi2 @ G2 @ gi2
    dGu = np.zeros((4, 4, 4))
    for k in range(3):
        pts = [np.array(x0, dtype=float) + m * eps * np.eye(3)[k] for m in (-2, -1, 1, 2)]
        vals = [Gup(Ff, Gf, p) for p in pts]
        dGu[:, :, k + 1] = (vals[0] - 8 * vals[1] + 8 * vals[2] - vals[3]) / (12 * eps)
    Gu0 = ginv @ Gmn @ ginv
    D = np.zeros(4)
    for nu in range(4):
        acc = 0.0
        for mu in range(4):
            acc = acc + dGu[mu, nu, mu]
            for lam in range(4):
                acc = acc + G0[mu, mu, lam] * Gu0[lam, nu] + G0[nu, mu, lam] * Gu0[mu, lam]
        D[nu] = acc
    return Gmn, gg, ginv, G0, D


def F_exact(r):
    u = MU / (2.0 * r)
    return 4.0 * u / (1.0 + u) ** 2


def G_exact(r):
    u = MU / (2.0 * r)
    return (1.0 + u) ** 4 - 1.0


def F_zero(r):
    return np.zeros_like(r)


def G_zero(r):
    return np.zeros_like(r)


def F_soft(r, A=1e-3, a=1.0):
    return 2.0 * A / np.sqrt(r ** 2 + a ** 2)


def G_soft(r, A=1e-3, a=1.0):
    return 2.0 * A / np.sqrt(r ** 2 + a ** 2)


def F_lin_ansatz(r, A=0.2, a=1.0):
    return 2.0 * A / np.sqrt(r ** 2 + a ** 2)


def G_lin_ansatz(r, A=0.2, a=1.0):
    return 2.0 * A / np.sqrt(r ** 2 + a ** 2)


print("=" * 78)
print("G1b 首轮：静球对称的二阶靶子与判定项（判据 G1b-1 至 G1b-6）")
print("=" * 78)

print()
print("[G1b-1] FD 机制：平坦与精确真空解")
p0 = (1.2, 0.0, 0.0)
Gmn0, _, _, _ = einstein_at(F_zero, G_zero, p0)
print("     平坦：max|Gmn| = %.2e（期望 <= 1e-12）" % np.abs(Gmn0).max())
for eps in (0.05, 0.025):
    res = []
    for r0 in (1.0, 1.5, 2.0, 3.0):
        Gm, _, _, _ = einstein_at(F_exact, G_exact, (r0, 0.0, 0.0), eps=eps)
        res.append(np.abs(Gm).max())
    print("     精确真空解：eps = %.3f，max|Gmn| = %s" % (eps, str([round(x, 8) for x in res])))
ratio = []
for r0 in (1.0, 2.0):
    a1 = np.abs(einstein_at(F_exact, G_exact, (r0, 0, 0), eps=0.05)[0]).max()
    a2 = np.abs(einstein_at(F_exact, G_exact, (r0, 0, 0), eps=0.025)[0]).max()
    ratio.append(a1 / a2)
print("     收敛比（eps 减半）：%s（四阶期望约 16）" % str([round(x, 2) for x in ratio]))

print()
print("[G1b-2] 线性核对：G00 对 −2∇²θ（Richardson 外推）")
A = 1e-3
a = 1.0
for r0 in (1.0, 2.0):
    g1 = einstein_at(lambda r: F_soft(r, A, a), lambda r: G_soft(r, A, a), (r0, 0, 0), eps=0.05)[0][0, 0]
    g2 = einstein_at(lambda r: F_soft(r, A, a), lambda r: G_soft(r, A, a), (r0, 0, 0), eps=0.025)[0][0, 0]
    gx = (16.0 * g2 - g1) / 15.0
    ana = -2.0 * (-3.0 * a ** 2 * A / (r0 ** 2 + a ** 2) ** 2.5)
    print("     r = %.1f：G00（外推）= %.8e，−2∇²θ = %.8e，比 = %.5f" % (r0, gx, ana, gx / ana))

print()
print("[G1b-3] 比安基核对：∇μ G_{μν} = 0（一般 F、G）")
for (Ff, Gf, tag) in ((F_exact, G_exact, "精确真空"), (lambda r: F_soft(r, 0.2, 1.0), lambda r: G_soft(r, 0.2, 1.0), "线性族 A=0.2")):
    _, _, _, _, D = einstein_at(Ff, Gf, (1.5, 0.0, 0.0), want_bianchi=True)
    print("     %s：max|∇μG_{μν}| = %.3e" % (tag, np.abs(D).max()))

print()
print("[G1b-4] 二阶靶子（代数精确）")
u = 1e-4
HL = 4.0 * u
h00 = 4.0 * u / (1.0 + u) ** 2
hij = (1.0 + u) ** 4 - 1.0
c00 = (h00 - HL) / HL ** 2
cij = (hij - HL) / HL ** 2
print("     h00 = h_lin + (%.6f) h_lin²；期望 -0.5" % c00)
print("     hij = h_lin + (%.6f) h_lin²；期望 +0.375" % cij)
print("     等价关系：hij − h00 = (%.6f) h_lin²；期望 +0.875" % ((hij - h00) / HL ** 2))

print()
print("[G1b-5] 线性 ansatz（hij ≡ h00）与二阶完成族（hij = h00 + (7/8)h00²）的残差对照")
def G_second(r, A, a):
    th = A / np.sqrt(r ** 2 + a ** 2)
    return 2.0 * th + 3.5 * th ** 2
for r0 in (1.0, 2.0):
    for A2 in (0.05, 0.1, 0.2):
        g1 = einstein_at(lambda rr: F_lin_ansatz(rr, A2, 1.0), lambda rr: G_lin_ansatz(rr, A2, 1.0), (r0, 0, 0), eps=0.05)[0][0, 0]
        g2 = einstein_at(lambda rr: F_lin_ansatz(rr, A2, 1.0), lambda rr: G_lin_ansatz(rr, A2, 1.0), (r0, 0, 0), eps=0.025)[0][0, 0]
        gx1 = (16.0 * g2 - g1) / 15.0
        g1b = einstein_at(lambda rr: F_lin_ansatz(rr, A2, 1.0), lambda rr: G_second(rr, A2, 1.0), (r0, 0, 0), eps=0.05)[0][0, 0]
        g2b = einstein_at(lambda rr: F_lin_ansatz(rr, A2, 1.0), lambda rr: G_second(rr, A2, 1.0), (r0, 0, 0), eps=0.025)[0][0, 0]
        gx2 = (16.0 * g2b - g1b) / 15.0
        ana = -2.0 * (-3.0 * A2 / (r0 ** 2 + 1.0) ** 2.5)
        c2a = (gx1 - ana) / A2 ** 2
        c2b = (gx2 - ana) / A2 ** 2
        print("     r = %.1f，A = %.2f：线性族二阶系数 = %+.4f；完成族二阶系数 = %+.4f"
              % (r0, A2, c2a, c2b))

print()
print("[G1b-6] 判定项清单")
rows = [
    ("G1b-a", "二阶靶子", "静真空二阶系数 h00: −1/2、hij: +3/8（等价 (7/8)）；机制装配必须生成"),
    ("G1b-b", "二阶源项", "记录场修正 θ = θ_lin − θ_lin² + O(θ³)（静球对称）；需自耦合/源二阶装配"),
    ("G1b-c", "残差消去", "线性同幅 ansatz 的二阶残差系数（见 G1b-5）必须被完成形式修掉"),
    ("G1b-d", "守恒一致", "二阶收缩比安基恒等式（本机核对到 FD 精度）"),
    ("G1b-e", "强场消费", "黑洞线读数（视界光锥、gxx 完成形式、停滞律）作为边界条件核对"),
    ("G1b-f", "连续极限", "以上装配后恢复 Gμν = 8πG/c⁴ Tμν 的完整非线性形式"),
]
for k, a1, b1 in rows:
    print("     %-8s %-8s %s" % (k, a1, b1))

print()
print("总运行时间 = %.1f 秒" % (time.time() - T0))
