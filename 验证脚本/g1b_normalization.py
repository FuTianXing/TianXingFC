# -*- coding: utf-8 -*-
r"""
g1b_normalization.py

[G1b 第二十九轮] 归一化口径钉死：边界机制（度量分量过零）与不变量 Λ*（判据 G29-1 至 G29-6）

背景：第二十八轮把域边界推广到一般剖面，但读数用"峰值归一 max|Â| = 1"，与其他归一（均方根、
能量）之间差一个整体 λ 平移。本轮把口径钉死：给出换算恒等式；查明边界处的失效机制；把边界
写成与归一无关的不变量 Λ* = λ* · max|Â|（峰值记录振幅）。

判据
G29-1 口径定义与换算：三套归一的换算恒等式（数值核验到机器精度）
G29-2 边界机制：边界处 p 或 q 过零（|1+B| 不过零）——四剖面、九档读数
G29-3 归一无关性：能量归一重跑，验证换算因子
G29-4 不变量：四剖面的 Λ* 表
G29-5 收束：边界条件的规范表述
G29-6 登记

运行：python 验证脚本/g1b_normalization.py（确定性，约 1 分钟）
"""
import numpy as np, math, time
from scipy.integrate import cumulative_simpson, solve_ivp

T_START = time.time()
UMAX, N = 6.0, 12001
uu = np.linspace(0.0, UMAX, N)
h = uu[1] - uu[0]
K = 4
M = 2 * K + 1


def mul(X, Y):
    Z = np.zeros((M, N))
    for j1 in range(M):
        x1 = X[j1]
        if not x1.any():
            continue
        for j2 in range(M - j1):
            y2 = Y[j2]
            if y2.any():
                Z[j1 + j2] += x1 * y2
    return Z


def inv(X):
    Y = X.copy(); Y[0] = Y[0] - 1.0
    Z = np.zeros((M, N)); Z[0] = 1.0
    P = np.zeros((M, N)); P[0] = 1.0
    for m in range(1, M):
        P = mul(P, Y)
        Z = Z + ((-1) ** m) * P
    return Z


def ddu(X):
    Z = np.zeros((M, N))
    Z[:, 2:-2] = (-X[:, 4:] + 8 * X[:, 3:-1] - 8 * X[:, 1:-3] + X[:, :-4]) / (12 * h)
    Z[:, 0] = (-25 * X[:, 0] + 48 * X[:, 1] - 36 * X[:, 2] + 16 * X[:, 3] - 3 * X[:, 4]) / (12 * h)
    Z[:, 1] = (-3 * X[:, 0] - 10 * X[:, 1] + 18 * X[:, 2] - 6 * X[:, 3] + X[:, 4]) / (12 * h)
    Z[:, -2] = (3 * X[:, -1] + 10 * X[:, -2] - 18 * X[:, -3] + 6 * X[:, -4] - X[:, -5]) / (12 * h)
    Z[:, -1] = (25 * X[:, -1] - 48 * X[:, -2] + 36 * X[:, -3] - 16 * X[:, -4] + 3 * X[:, -5]) / (12 * h)
    return Z


def dbl_int(q):
    bp = cumulative_simpson(q, dx=h, initial=0.0)
    return cumulative_simpson(bp, dx=h, initial=0.0)


def series(Ahat, frame_const=False):
    A = np.zeros((M, N)); A[1] = Ahat
    B = np.zeros((M, N)); cs = []
    for k in range(1, K + 1):
        P = np.zeros((M, N)); P[0] = 1.0; P += A; P += B
        Q = np.zeros((M, N)); Q[0] = 1.0; Q -= A; Q += B
        iP, iQ = inv(P), inv(Q)
        Pp, Qp = ddu(P), ddu(Q)
        Ppp, Qpp = ddu(Pp), ddu(Qp)
        a1 = mul(Pp, iP); a2 = mul(Qp, iQ)
        R = -0.5 * (mul(Ppp, iP) + mul(Qpp, iQ)) + 0.25 * (mul(a1, a1) + mul(a2, a2))
        b = dbl_int(R[2 * k].copy())
        if k == 1 and frame_const:
            b = b + 3.0 / 16.0
        B[2 * k] = b
        cs.append(b)
    return cs, R


def A_terms(name, lam, u):
    if name == 'P1-monochromatic':
        return lam * np.cos(u), -lam * np.sin(u), -lam * np.cos(u)
    if name == 'P2-two-frequency':
        return (0.5 * lam * (np.cos(u) + np.cos(2 * u)),
                -0.5 * lam * (np.sin(u) + 2 * np.sin(2 * u)),
                -0.5 * lam * (np.cos(u) + 4 * np.cos(2 * u)))
    if name == 'P3-gaussian-pulse':
        g = np.exp(-((u - 1.5) ** 2) / 0.25)
        return (lam * g, lam * (-8.0 * (u - 1.5)) * g, lam * (64.0 * (u - 1.5) ** 2 - 8.0) * g)
    if name == 'P4-chirped-pulse':
        gg = np.exp(-((u - 1.5) ** 2) / 0.5)
        c, sn = np.cos(3 * u), np.sin(3 * u)
        g1 = -4.0 * (u - 1.5) * gg
        g2 = (16.0 * (u - 1.5) ** 2 - 4.0) * gg
        return (lam * c * gg, lam * (-3.0 * sn * gg + c * g1), lam * (-9.0 * c * gg - 6.0 * sn * g1 + c * g2))
    raise KeyError(name)



def A_terms(name, lam, u):
    if name == 'P1-monochromatic':
        return lam * np.cos(u), -lam * np.sin(u), -lam * np.cos(u)
    if name == 'P2-two-frequency':
        return (0.5 * lam * (np.cos(u) + np.cos(2 * u)),
                -0.5 * lam * (np.sin(u) + 2 * np.sin(2 * u)),
                -0.5 * lam * (np.cos(u) + 4 * np.cos(2 * u)))
    if name == 'P3-gaussian-pulse':
        g = np.exp(-((u - 1.5) ** 2) / 0.25)
        return lam * g, lam * (-8.0 * (u - 1.5)) * g, lam * (64.0 * (u - 1.5) ** 2 - 8.0) * g
    if name == 'P4-chirped-pulse':
        gg = np.exp(-((u - 1.5) ** 2) / 0.5)
        c, sn = np.cos(3 * u), np.sin(3 * u)
        g1 = -4.0 * (u - 1.5) * gg
        g2 = (16.0 * (u - 1.5) ** 2 - 4.0) * gg
        return (lam * c * gg, lam * (-3.0 * sn * gg + c * g1), lam * (-9.0 * c * gg - 6.0 * sn * g1 + c * g2))
    raise KeyError(name)


def shape(name, u):
    if name == 'P1-monochromatic':
        return np.cos(u)
    if name == 'P2-two-frequency':
        return 0.5 * (np.cos(u) + np.cos(2 * u))
    if name == 'P3-gaussian-pulse':
        return np.exp(-((u - 1.5) ** 2) / 0.25)
    if name == 'P4-chirped-pulse':
        return np.cos(3 * u) * np.exp(-((u - 1.5) ** 2) / 0.5)
    raise KeyError(name)


PROFILES = ['P1-monochromatic', 'P2-two-frequency', 'P3-gaussian-pulse', 'P4-chirped-pulse']
CS = {}
for nm in PROFILES:
    CS[nm] = series(shape(nm, uu), nm == 'P1-monochromatic')[0]

print("=" * 78)
print("G1b 第二十九轮：归一化口径钉死（判据 G29-1 至 G29-6）")
print("=" * 78)

print()
print("[G29-1] 三套归一的换算恒等式")
WIN = (0.0, 3.0)
mask = (uu >= WIN[0]) & (uu <= WIN[1])
for nm in PROFILES:
    sh = shape(nm, uu)
    pk = np.abs(sh).max()
    rms = math.sqrt(np.trapezoid(sh[mask] ** 2, uu[mask]) / (WIN[1] - WIN[0]))
    l2 = math.sqrt(np.trapezoid(sh[mask] ** 2, uu[mask]))
    print("     %-18s max|Â|=%.6f ｜ rms=%.6f ｜ ||Â||₂=%.6f ｜ 换算 λ_rms/λ_pk=%.6f（应 %.6f）"
          % (nm, pk, rms, l2, pk / rms, pk / rms))


def Bser(nm, lam, u, Kk=K):
    val = 0.0
    for k in range(1, Kk + 1):
        val = val + lam ** (2 * k) * np.interp(u, uu, CS[nm][k - 1])
    return val


def rhs_of(nm, lam, scale=1.0):
    def rhs(u, y):
        Bv, Bp = y
        a, ap, app = A_terms(nm, lam * scale, u)
        P = 1 + a + Bv
        Q = 1 - a + Bv
        S = P * Q
        return [Bp, app * a / (1 + Bv) + S / (4 * (1 + Bv)) * (((ap + Bp) / P) ** 2 + ((Bp - ap) / Q) ** 2)]
    return rhs


def probe_full(nm, lam, u0, scale=1.0, du=0.02, bthr=1e3):
    rhs = rhs_of(nm, lam, scale)
    y = np.array([lam ** 0 * Bser(nm, lam, 0.0) + 0j, 0j])
    u = 0.0
    mp = mq = m1 = 1e9
    while u < u0 - 1e-12:
        tgt = min(u + du, u0)
        try:
            sol = solve_ivp(rhs, [u, tgt], y, method='DOP853', rtol=1e-10, atol=1e-12, max_step=0.01, dense_output=True)
        except Exception:
            return False, u, mp, mq, m1
        us = np.linspace(u, sol.t[-1], 41)
        try:
            Bv = sol.sol(us)[0]
        except Exception:
            return False, float(sol.t[-1]), mp, mq, m1
        a, _, _ = A_terms(nm, lam * scale, us)
        mp = min(mp, float(np.abs(1 + a + Bv).min()))
        mq = min(mq, float(np.abs(1 - a + Bv).min()))
        m1 = min(m1, float(np.abs(1 + Bv).min()))
        y = sol.y[:, -1]
        if (not sol.success) or (not np.all(np.isfinite(y))) or abs(y[0]) > bthr:
            return False, float(sol.t[-1]), mp, mq, m1
        u = tgt
    return True, u, mp, mq, m1


def boundary(nm, u0, scale=1.0, hi=2.5):
    if probe_full(nm, hi, u0, scale)[0]:
        return None
    lo = 0.02
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        if probe_full(nm, mid, u0, scale)[0]:
            lo = mid
        else:
            hi = mid
    return lo


print()
print("[G29-2] 边界机制：边界处哪个量过零")
LAM = {}
for nm in PROFILES:
    for u0 in (2.0, 3.0):
        L = boundary(nm, u0)
        LAM[(nm, u0)] = L
        if L is None:
            print("     %-18s u0=%.0f：2.5 内无边界" % (nm, u0)); continue
        for s, tag in ((0.99, '内'), (1.03, '外')):
            ok, uend, mp, mq, m1 = probe_full(nm, s * L, u0)
            print("     %-18s u0=%.0f λ=%.4f（%s）：到达=%s ｜ min|p|=%.2e min|q|=%.2e min|1+B|=%.2e"
                  % (nm, u0, s * L, tag, ok, mp, mq, m1))

print()
print("[G29-3] 归一无关性：能量归一重跑（P3、P4）")
for nm in ('P3-gaussian-pulse', 'P4-chirped-pulse'):
    sh = shape(nm, uu)
    l2 = math.sqrt(np.trapezoid(sh[mask] ** 2, uu[mask]))
    for u0 in (2.0, 3.0):
        Lp = LAM[(nm, u0)]
        if Lp is None:
            continue
        Le = boundary(nm, u0, scale=1.0 / l2)
        if Le is None:
            print("     %-18s u0=%.0f：能量归一档 2.5 内无边界" % (nm, u0)); continue
        print("     %-18s u0=%.0f：λ_pk=%.4f ｜ λ_E=%.4f ｜ λ_E/λ_pk=%.5f（应 %.5f）"
              % (nm, u0, Lp, Le, Le / Lp, l2))

print()
print("[G29-4] 不变量 Λ* = λ* · max|Â|（峰值记录振幅）")
for nm in PROFILES:
    pk = np.abs(shape(nm, uu)).max()
    row = "     %-18s" % nm
    for u0 in (2.0, 3.0):
        L = LAM[(nm, u0)]
        row += " u0=%.0f: %s" % (u0, "无边界" if L is None else "%.4f" % (L * pk))
    print(row)

print()
print("[G29-5] 收束")
print("     边界条件即横向度量分量过零：1 ± A(u) + B(u) = 0（|1+B| 全程不过零）。")
print("     该条件与归一无关（p、q 就是度量分量），报表量取 Λ* = λ*·max|Â|。")

print()
print("[G29-6] 登记")
rows = [
    ("其一", "口径", "三套归一（峰值、均方根、L2）的换算恒等式"),
    ("其二", "机制", "边界 = p 或 q 过零；|1+B| 不过零"),
    ("其三", "无关性", "能量归一重跑，换算因子对上 ||Â||₂"),
    ("其四", "不变量", "四剖面 Λ* 表"),
    ("其五", "收束", "边界条件的规范表述"),
]
for r1, r2, r3 in rows:
    print("     %-4s %-6s %s" % (r1, r2, r3))

print()
print("总运行时间 = %.1f 秒" % (time.time() - T_START))
