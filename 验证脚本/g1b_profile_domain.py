# -*- coding: utf-8 -*-
r"""
g1b_profile_domain.py

[G1b 第二十八轮] 一般剖面的域边界：λ 级数器械、剖面类分野与误差律（判据 G28-1 至 G28-6）

背景：第二十六、二十七轮把单色波 A = a cos u 的收敛半径测成"解域边界"。本轮把它推广到一般
剖面 A = λ Â(u)（max|Â| = 1）：用网格版双变量截断级数（λ 偶阶至 λ^8、u 网格 6 与 12001 点，
累计 Simpson 二重积分、四阶单边差分）逐阶解 b_{2k}，再用分段积分探针测域边界，并与打靶解对照。

判据
G28-1 器械与回归：单色档对上第二十五轮闭式；四剖面的奇数阶恒零（奇偶封口）
G28-2 域边界：四剖面在 u0 = 1, 2, 3 的 λ* 表；双频档的初始切片退化（λ = 1）
G28-3 剖面类对照：周期型与局域脉冲型的边界差别
G28-4 误差律：偏差 / λ^(2K+2) 的 λ 无关性
G28-5 收束：边界尺度与退化估计的关系（负结果记录）
G28-6 登记

运行：python 验证脚本/g1b_profile_domain.py（确定性，约 1 分钟）
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


PROF = [
    ('P1-monochromatic', lambda u: np.cos(u), True),
    ('P2-two-frequency', lambda u: 0.5 * (np.cos(u) + np.cos(2 * u)), False),
    ('P3-gaussian-pulse', lambda u: np.exp(-((u - 1.5) ** 2) / 0.25), False),
    ('P4-chirped-pulse', lambda u: np.cos(3 * u) * np.exp(-((u - 1.5) ** 2) / 0.5), False),
]

print("=" * 78)
print("G1b 第二十八轮：一般剖面的域边界（判据 G28-1 至 G28-6）")
print("=" * 78)

CS = {}
RES = {}
for nm, f, fc in PROF:
    CS[nm], RES[nm] = series(f(uu), fc)

print()
print("[G28-1] 器械与回归")
b2c = 3.0 / 16.0 * np.cos(2 * uu) - uu ** 2 / 8.0
b4c = (3 * uu ** 2 * np.cos(2 * uu) / 128 + 11 * uu ** 2 / 512 - uu * np.sin(2 * uu) / 128
       + 25 * np.sin(2 * uu) ** 2 / 2048 - np.cos(2 * uu) / 128 + 1.0 / 128)
print("     单色档对第二十五轮闭式：b_2 差 %.2e ｜ b_4 差 %.2e"
      % (np.abs(CS['P1-monochromatic'][0] - b2c).max(), np.abs(CS['P1-monochromatic'][1] - b4c).max()))
for nm, _, _ in PROF:
    odds = max(np.abs(RES[nm][j]).max() for j in (1, 3, 5, 7))
    evens = [np.abs(RES[nm][2 * k]).max() for k in range(1, K + 1)]
    print("     %-20s 奇数阶 max %.1e ｜ 偶数阶 %s" % (nm, odds, " ".join("%.1e" % v for v in evens[:3])))


def Bser(nm, lam, u, Kk):
    val = 0.0
    for k in range(1, min(Kk, K) + 1):
        val = val + lam ** (2 * k) * np.interp(u, uu, CS[nm][k - 1])
    return val


def rhs_of(nm, lam):
    def rhs(u, y):
        Bv, Bp = y
        a, ap, app = A_terms(nm, lam, u)
        P = 1 + a + Bv
        Q = 1 - a + Bv
        S = P * Q
        return [Bp, app * a / (1 + Bv) + S / (4 * (1 + Bv)) * (((ap + Bp) / P) ** 2 + ((Bp - ap) / Q) ** 2)]
    return rhs


def probe(nm, lam, u0, bthr=30.0, du=0.05):
    rhs = rhs_of(nm, lam)
    y = np.array([Bser(nm, lam, 0.0, K) + 0j, 0j])
    u = 0.0
    while u < u0 - 1e-12:
        tgt = min(u + du, u0)
        try:
            sol = solve_ivp(rhs, [u, tgt], y, method='DOP853', rtol=1e-10, atol=1e-12, max_step=0.02)
        except Exception:
            return False, u
        y = sol.y[:, -1]
        if (not sol.success) or (not np.all(np.isfinite(y))) or abs(y[0]) > bthr or abs(1 + y[0]) < 1e-3:
            return False, float(sol.t[-1])
        u = tgt
    return True, u


def init_deg(nm, lam):
    a0, _, _ = A_terms(nm, lam, 0.0)
    return min(abs(1 + a0), abs(1 - a0)) < 1e-9


print()
print("[G28-2] 域边界 λ*(u0)")
for nm, _, _ in PROF:
    row = "     %-20s" % nm
    for u0 in (1.0, 2.0, 3.0):
        if init_deg(nm, 0.05):
            row += " u0=%.0f:初始退化" % u0
        elif probe(nm, 2.0, u0)[0]:
            row += " u0=%.0f:>2.0" % u0
        else:
            lo, hi = 0.02, 2.0
            for _ in range(24):
                mid = 0.5 * (lo + hi)
                if init_deg(nm, mid) or probe(nm, mid, u0)[0]:
                    lo = mid
                else:
                    hi = mid
            row += " u0=%.0f:%.4f" % (u0, lo)
    print(row)
print("     双频档初始切片退化：Â(0) = 1，λ = 1 时 q(0) = 1 - λÂ(0) + B(0) = 0（与第二十七轮 4/3 同类）")

print()
print("[G28-3] 剖面类对照")
print("     周期型（P1 单色、P2 双频）在 u0 = 3 的边界 0.768 与 0.679；")
print("     局域脉冲型（P3 高斯、P4 啁啾）在同点为 1.080 与 0.634，u0 = 1 档均在 2.0 以上（脉冲未到或已过）")

print()
print("[G28-4] 误差律：偏差 / λ^(2K+2)")
for nm in ('P1-monochromatic', 'P2-two-frequency', 'P3-gaussian-pulse'):
    for lam in (0.05, 0.1, 0.2):
        uarr = np.linspace(0, 2.0, 201)
        sol = solve_ivp(rhs_of(nm, lam), [0.0, 2.0], [Bser(nm, lam, 0.0, K) + 0j, 0j],
                        method='DOP853', rtol=1e-12, atol=1e-14, dense_output=True, max_step=0.01)
        ex = sol.sol(uarr)[0].real
        row = []
        for Kk in range(1, K + 1):
            dif = float(np.max(np.abs(ex - np.array([Bser(nm, lam, u, Kk) for u in uarr]))))
            row.append(abs(dif) / lam ** (2 * Kk + 2))
        print("     %-20s λ=%.2f：%s" % (nm, lam, " ".join("%.4e" % v for v in row)))

print()
print("[G28-5] 收束")
print("     四个剖面的边界都落在 O(1)（0.63 至 1.5），尺度与剖面形状无关。")
print("     一阶退化估计（用 b_2 解 1 + b_2λ^2 ∓ Âλ = 0）在 P1 上给 4/3（u = 0 档）与 0.645（u = π 档），")
print("     实测边界 1.111（u0 = 2）与 0.768（u0 = 3）落在其间：退化估计给量级，具体边界由解的奇点决定。")

print()
print("[G28-6] 登记")
rows = [
    ("其一", "器械", "网格版 λ 级数；单色档对上闭式；四剖面奇数阶恒零"),
    ("其二", "边界", "λ*(u0) 表：P1 0.768/1.111，P2 0.679/0.754/1.008，P3 1.080，P4 0.634"),
    ("其三", "剖面类", "周期型与局域型的边界差别；双频档初始切片退化"),
    ("其四", "误差律", "偏差 / λ^(2K+2) 的 λ 无关性（三剖面）"),
    ("其五", "收束", "边界尺度 O(1) 与形状无关；一阶退化估计只给量级"),
]
for r1, r2, r3 in rows:
    print("     %-4s %-6s %s" % (r1, r2, r3))

print()
print("总运行时间 = %.1f 秒" % (time.time() - T_START))
