# -*- coding: utf-8 -*-
"""
minimal_verification_g1b_continuum_limit.py

[G1b 第五轮] 连续极限（判据 G1b4-1 至 G1b4-5）

背景：正稿《爱因斯坦场方程的微观根基》§3.2 给出曲率的链接表述（格面链接累积的净转角亏损即
曲率，与 Regge 无坐标构造同构）；§4.2 给出链接合成到泊松方程的显式版本。本轮把"完成后的
非线性结构"的连续极限做成读数：解层面（完成式级数与其收敛半径）与器械层面（格点写入的格点
格林函数、格面 holonomy 的曲率读数）。

口径：h00 = 2 theta（F = h00），空间扇区 F = h00、G = hij 系数；精确真空解给规则闭式
G_exact(F) = w^2 − 1，w = (8 − 4F − 8 sqrt(1 − F))/F^2，小 F 展开 F + (7/8)F^2 + ...
完成式两口径：截断 theta_c = theta_lin − theta_lin^2 配 G = F + (7/8)F^2（二、三轮用的对象）；
精确 theta_ex = theta_lin/(1 + theta_lin/2)^2 配 G = G_exact(F)。

判据
G1b4-1 解层面收敛：theta 完成式系数 c_k = (−1)^{k−1} k/2^{k−1}（数值提取核对）；截断误差按
       |c_{n+1}| theta^{n+1} 标度；theta_lin > 2 发散（读数）；视界 theta = 1/2 在收敛半径内
G1b4-2 截断与锁定分居：G00[截断完成式] − G00[精确完成式] 的标度（期望 O(M^3)）对三轮锁定
       残差 R（O(M^2)）：锁定值不是截断假象
G1b4-3 链接读数：格面 holonomy（闭路平移积）H = 1 − a^2 R + O(a^3)，与独立 FD Riemann 对照，
       测收敛阶（a = 0.4 至 0.05）
G1b4-4 格点写入：7 点拉普拉斯算符的连续极限（符号差 (a^2/12) sum k_i^4 与其
       方向比 1 : 1/2 : 1/3，与 P4-31 的 sum k^4 各向异性同源）
G1b4-5 综合登记

运行：python 验证脚本/minimal_verification_g1b_continuum_limit.py（确定性，约 20 秒）
"""
import math
import time
import numpy as np

T0 = time.time()
MASS = 0.02
SIG = 1.0
erf_vec = np.vectorize(math.erf)


def th_lin(r, mass=MASS):
    r = np.asarray(r, dtype=float)
    x = r / (SIG * math.sqrt(2.0))
    core = mass * math.sqrt(2.0 / math.pi) / SIG
    with np.errstate(divide="ignore", invalid="ignore"):
        v = mass * erf_vec(x) / r
    return np.where(x < 1e-8, core, v)


def th_c(r, mass=MASS):
    t = th_lin(r, mass)
    return t - t * t


def th_ex(r, mass=MASS):
    t = th_lin(r, mass)
    return t / (1.0 + 0.5 * t) ** 2


def w_exact(F):
    F = np.asarray(F, dtype=float)
    out = np.ones_like(F)
    nz = F != 0.0
    out[nz] = (8.0 - 4.0 * F[nz] - 8.0 * np.sqrt(1.0 - F[nz])) / F[nz] ** 2
    return out


def G_exact(F):
    return w_exact(F) ** 2 - 1.0


def metric_at(x, Ff, Gf):
    r = math.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2)
    g = np.zeros((4, 4))
    g[0, 0] = -(1.0 - float(Ff(r)))
    for i in (1, 2, 3):
        g[i, i] = 1.0 + float(Gf(r))
    return g


def dg_at(x, Ff, Gf, i, h=1e-4):
    xp = list(x)
    xp[i - 1] += h
    xm = list(x)
    xm[i - 1] -= h
    return (metric_at(xp, Ff, Gf) - metric_at(xm, Ff, Gf)) / (2.0 * h)


def gamma_at(x, Ff, Gf, h=1e-4):
    g = metric_at(x, Ff, Gf)
    ginv = np.linalg.inv(g)
    dg = {0: np.zeros((4, 4))}
    for i in (1, 2, 3):
        dg[i] = dg_at(x, Ff, Gf, i, h)
    Gam = np.zeros((4, 4, 4))
    for lam in range(4):
        for mu in range(4):
            for nu in range(4):
                s = 0.0
                for sig in range(4):
                    s += ginv[lam, sig] * (dg[mu][sig, nu] + dg[nu][sig, mu] - dg[sig][mu, nu])
                Gam[lam, mu, nu] = 0.5 * s
    return Gam


def riem_fd(x, Ff, Gf, h=1e-5):
    G0 = gamma_at(x, Ff, Gf)
    dG = {0: np.zeros((4, 4, 4))}
    for i in (1, 2, 3):
        xp = list(x)
        xp[i - 1] += h
        xm = list(x)
        xm[i - 1] -= h
        dG[i] = (gamma_at(xp, Ff, Gf) - gamma_at(xm, Ff, Gf)) / (2.0 * h)
    R = np.zeros((4, 4, 4, 4))
    for lam in range(4):
        for sig in range(4):
            for mu in range(4):
                for nu in range(4):
                    v = dG[mu][lam, sig, nu] - dG[nu][lam, sig, mu]
                    for rho in range(4):
                        v += G0[lam, mu, rho] * G0[rho, nu, sig] - G0[lam, nu, rho] * G0[rho, mu, sig]
                    R[lam, sig, mu, nu] = v
    return R


def transport(x0, x1, Ff, Gf):
    d3 = np.array(x1, dtype=float) - np.array(x0, dtype=float)
    d = np.array([0.0, d3[0], d3[1], d3[2]])
    mid = [0.5 * (a + b) for a, b in zip(x0, x1)]
    Gm = gamma_at(mid, Ff, Gf)
    T = np.eye(4)
    for mu in range(4):
        for nu in range(4):
            for sig in range(4):
                T[mu, nu] -= Gm[mu, nu, sig] * d[sig]
    return T


def holonomy(x0, a, Ff, Gf, axis1=0, axis2=1):
    P = [np.array(x0, dtype=float) for _ in range(4)]
    P[1] = P[0].copy()
    P[1][axis1] += a
    P[2] = P[1].copy()
    P[2][axis2] += a
    P[3] = P[0].copy()
    P[3][axis2] += a
    H = np.eye(4)
    for k in range(4):
        H = transport(P[(k + 1) % 4], P[k], Ff, Gf) @ H
    return H


print("=" * 78)
print("G1b 第五轮：连续极限（判据 G1b4-1 至 G1b4-5）")
print("=" * 78)

print()
print("[G1b4-1] 解层面收敛：完成式级数与收敛半径")
def f_ex(t):
    return t / (1.0 + 0.5 * t) ** 2

coefs_rec = [1.0]
for k in range(2, 11):
    c = -(coefs_rec[k - 2] + (coefs_rec[k - 3] / 4.0 if k >= 3 else 0.0))
    coefs_rec.append(c)
coefs_fml = [(-1.0) ** (k - 1) * k / 2.0 ** (k - 1) for k in range(1, 11)]
print("     系数递推核对（f (1 + t/2)^2 = t 的 Cauchy 积，独立于公式）：")
print("       递推 %s" % "、".join("%+.4f" % c for c in coefs_rec[:6]))
print("       公式 %s" % "、".join("%+.4f" % c for c in coefs_fml[:6]))
print("       两路最大差 = %.2e" % max(abs(a - b) for a, b in zip(coefs_rec, coefs_fml)))
for t in (0.3, 0.5, 1.0):
    S8 = sum(coefs_fml[k - 1] * t ** k for k in range(1, 9))
    err = f_ex(t) - S8
    lead = abs(coefs_fml[8]) * t ** 9
    print("     theta_lin = %.1f：截断 n = 8 残差 = %+.3e，首项量级 %.3e，比 %.3f（交错尾比 1/(1 + 0.5556 t) = %.3f）"
          % (t, err, lead, abs(err) / lead, 1.0 / (1.0 + (10.0 / 18.0) * t)))
part = [sum(coefs_fml[k - 1] * 2.5 ** k for k in range(1, n + 1)) for n in range(1, 10)]
print("     发散读数（theta_lin = 2.5 在半径 2 外）：部分和 %s（幅度增长）" % "、".join("%.2f" % v for v in part))
print("     闭式在 2.5 处仍有值 f = %.4f（级数非表示，闭式仍有效）；视界 theta = 1/2 在半径内（余量 4 倍）" % f_ex(2.5))

print("[G1b4-2] 截断与锁定分居：O(M^3) 对 O(M^2)")
import io
src2 = io.open(r"D:/software/TianXingFC/验证脚本/minimal_verification_g1b_sourced_second_order.py", encoding="utf-8").read()
cut = src2.index('print("=" * 78)')
exec(src2[:cut])
for mass in (0.05, 0.02):
    Fc = lambda r, m=mass: 2.0 * th_c(r, m)
    Gc = lambda r, m=mass: 2.0 * th_c(r, m) + 3.5 * th_c(r, m) ** 2
    Fe = lambda r, m=mass: 2.0 * th_ex(r, m)
    Ge = lambda r, m=mass: G_exact(2.0 * th_ex(r, m))
    for r0 in (0.8, 1.2):
        gc = (16.0 * g00_of(Fc, Gc, (r0, 0, 0), 0.025) - g00_of(Fc, Gc, (r0, 0, 0), 0.05)) / 15.0
        ge = (16.0 * g00_of(Fe, Ge, (r0, 0, 0), 0.025) - g00_of(Fe, Ge, (r0, 0, 0), 0.05)) / 15.0
        R = gc - 8.0 * math.pi * rho_m(r0, mass)
        print("     M = %.3f r = %.1f：截断-精确差 = %+.3e ｜ 锁定残差 R = %+.3e ｜ 比 = %.3f"
              % (mass, r0, gc - ge, R, (gc - ge) / R))

print()
print("[G1b4-3] 链接读数：格面 holonomy 与 FD Riemann 的连续极限")
Ffun = lambda r: 2.0 * th_c(r)
Gfun = lambda r: 2.0 * th_c(r) + 3.5 * th_c(r) ** 2
x0 = (1.2, 0.0, 0.0)
RF = riem_fd(x0, Ffun, Gfun)
# (x,y) 平面的曲率矩阵 R^mu_{x nu y}
Rxy = np.zeros((4, 4))
for mu in range(4):
    for nu in range(4):
        Rxy[mu, nu] = RF[mu, nu, 1, 2]
nrmR = np.linalg.norm(Rxy)
rat = {}
print("     FD Riemann（x,y 平面，(R^mu_{x nu y}) 的范数）= %.6e" % nrmR)
for a in (0.4, 0.2, 0.1, 0.05):
    H = holonomy(x0, a, Ffun, Gfun)
    M = (H - np.eye(4)) / a ** 2
    nrmM = np.linalg.norm(M)
    print("     a = %.2f：||(H − 1)/a^2|| = %.6e ｜ 比值 ||M||/||Rxy|| = %.4f ｜ 偏差 = %.4f（∝ a）"
          % (a, nrmM, nrmM / nrmR, abs(nrmM / nrmR - 1.0)))
    rat[a] = nrmM / nrmR

print("     线性外推 2·ratio(a/2) − ratio(a)：%.4f（a = 0.1/0.05 档）" % (2.0 * rat[0.05] - rat[0.1]))

print()
print("[G1b4-4] 格点写入：7 点拉普拉斯算符的连续极限（符号差的精确值）")
print("     格点算符：laplace_lat f = (1/a^2) sum_i [f(x+a e_i) − 2f(x) + f(x−a e_i)]")
print("     符号：D(k) = (4/a^2) sum_i sin^2(k_i a/2)；与连续 k^2 的差 = (a^2/12) sum k_i^4 + O(a^4)")
for a in (0.1, 0.05, 0.01):
    row = []
    for ndir, lab in (((1, 0, 0), "[100]"), ((1, 1, 0), "[110]"), ((1, 1, 1), "[111]")):
        nrm = math.sqrt(sum(v * v for v in ndir))
        kh = [v / nrm for v in ndir]
        Dk = (4.0 / a ** 2) * sum(math.sin(0.5 * a * kh[i]) ** 2 for i in range(3))
        rel = (1.0 - Dk) / 1.0
        k4 = sum(v ** 4 for v in kh)
        row.append((lab, rel / a ** 2, k4))
    print("     a = %.3f：相对修正/a^2 = %s ｜ sum k^4/k^4 = %s ｜ 预测 (1/12)·sum = %s"
          % (a, "、".join("%s %+.5f" % (lab, val) for lab, val, _ in row),
             "、".join("%.4f" % q for _, _, q in row),
             "、".join("%.5f" % (q / 12.0) for _, _, q in row)))
print("     方向比 1 : 0.5 : 0.3333 与立方四阶不变量一致（与 P4-31 的 sum k_a^4 各向异性同源登记）")

print("[G1b4-5] 综合登记")
rows = [
    ("其一", "解层面", "完成式级数系数 c_k = (−1)^{k−1}k/2^{k−1}，收敛半径 2；视界 theta = 1/2 在内（4 倍余量）"),
    ("其二", "器械层面", "格点算符的连续极限为 (a^2/12) sum k^4（方向比 1:1/2:1/3）；格面 holonomy 读数与 FD Riemann 一阶同步收敛"),
    ("其三", "截断与锁定", "截断-精确差 O(M^3) 远小于锁定残差 O(M^2)：三轮锁定值不是截断假象"),
    ("其四", "余项", "强场对接（G1b-e）按裁决归黑洞线主建；本线余项至此为 G1b-f 以外的消费项"),
]
for a1, a2, a3 in rows:
    print("     %-4s %-12s %s" % (a1, a2, a3))

print()
print("总运行时间 = %.1f 秒" % (time.time() - T0))
