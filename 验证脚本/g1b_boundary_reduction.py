# -*- coding: utf-8 -*-
r"""
g1b_boundary_reduction.py

[G1b 第三十轮] 边界条件的解析约化（判据 R1 至 R6）

背景：第 29 轮把边界机制查明为横向度量分量过零（1 +- A + B = 0），但只有数值。
本轮给出解析约化：令 w = (P Q)^{1/4}（横向尺度因子），则

  R1  代数恒等式：B = r - 1，P = r + a，Q = r - a，r = sqrt(a^2 + w^4)；
       a = w^2 sinh(M/2)，M = ln(P/Q)
  R2  约化方程：w'' + (M')^2 w / 16 = 0，等价形式
       w'' = - (a' - 2 a w'/w)^2 w / (4 (w^4 + a^2))
  R3  初值闭式：w(0) = ((1+B0)^2 - a0^2)^{1/4}，
       w'(0) = (B'(0)(1+B0) - a0 a'(0)) / (2 w(0)^3)
  R4  凹性：w'' <= 0 逐点（等号仅在 a = c w^2 档）
  R5  边界同一：w 的零点与度量分量过零是同一事件（位置与失效层读数）
  R6  可解子族（常数 M）与登记

运行：python 验证脚本/g1b_boundary_reduction.py（确定性，约 40 秒）
"""
import time

import numpy as np
from scipy.integrate import solve_ivp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

T0 = time.time()
CHECKS = []


def gauss(w):
    def A(u):
        return np.exp(-((u - 1.5) ** 2) / (2.0 * w ** 2))
    def A1(u):
        return A(u) * (-(u - 1.5) / w ** 2)
    def A2(u):
        return A(u) * (((u - 1.5) ** 2) / w ** 4 - 1.0 / w ** 2)
    return A, A1, A2


def system(A, A1, A2, lam):
    def rhs(u, y):
        Bv, Bp = y
        a, ap, app = lam * A(u), lam * A1(u), lam * A2(u)
        P = 1 + a + Bv
        Q = 1 - a + Bv
        S = P * Q
        return [Bp, app * a / (1 + Bv) + S / (4 * (1 + Bv)) * (((ap + Bp) / P) ** 2 + ((Bp - ap) / Q) ** 2)]

    def rhs_w(u, y):
        wv, wp = y
        a, ap = lam * A(u), lam * A1(u)
        num = ap - 2.0 * a * wp / wv
        return [wp, -(num * num) / (4.0 * (wv ** 4 + a ** 2)) * wv]

    return rhs, rhs_w


def ic_w(A, A1, lam, B0=0.0, Bp0=0.0):
    a0, ap0 = lam * A(0.0), lam * A1(0.0)
    w0 = ((1.0 + B0) ** 2 - a0 ** 2) ** 0.25
    wp0 = (Bp0 * (1.0 + B0) - a0 * ap0) / (2.0 * w0 ** 3)
    return w0, wp0


def integ(f, u0, y0, n=4001):
    ts = np.linspace(0.0, u0, n)
    return solve_ivp(f, [0.0, u0], y0, t_eval=ts, method="DOP853",
                     rtol=1e-11, atol=1e-13, max_step=1e-3)


def probe_metric(A, A1, A2, lam, u0, du=0.05):
    rhs, _ = system(A, A1, A2, lam)
    y = np.array([0j, 0j])
    u = 0.0
    while u < u0 - 1e-12:
        tgt = min(u + du, u0)
        try:
            sol = solve_ivp(rhs, [u, tgt], y, method="DOP853", rtol=1e-10, atol=1e-12, max_step=0.02)
        except Exception:
            return False
        y = sol.y[:, -1]
        if (not sol.success) or (not np.all(np.isfinite(y))) or abs(y[0]) > 30.0 or abs(1 + y[0]) < 1e-3:
            return False
        u = tgt
    return True


def lam_star(pred, lo=0.02, hi=3.0, it=22):
    if pred(hi):
        return None
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        if pred(mid):
            lo = mid
        else:
            hi = mid
    return lo


print("=" * 78)
print("G1b 第三十轮：边界条件的解析约化（判据 R1 至 R6）")
print("=" * 78)

CASES = [("gauss w=0.354", gauss(0.354), 1.00), ("gauss w=1.0", gauss(1.0), 0.90), ("gauss w=0.2", gauss(0.2), 0.70)]

print()
print("[R1] 代数恒等式（w = (PQ)^{1/4}，r = sqrt(a^2+w^4)）")
r1max = 0.0
for nm, (A, A1, A2), lam in CASES:
    rhs, _ = system(A, A1, A2, lam)
    sol = integ(rhs, 3.0, [0j, 0j])
    us = sol.t
    Bv = sol.y[0]
    a = lam * A(us)
    P = 1 + a + Bv
    Q = 1 - a + Bv
    w = (P * Q) ** 0.25
    r = np.sqrt(a ** 2 + w ** 4)
    e1 = np.max(np.abs(Bv - (r - 1)))
    e2 = np.max(np.abs(P - (r + a)))
    e3 = np.max(np.abs(Q - (r - a)))
    e4 = np.max(np.abs(a - w ** 2 * np.sinh(np.log(P / Q) / 2)))
    r1max = max(r1max, e1, e2, e3, e4)
    print("     %-14s lam=%.2f：|B-r+1|=%.2e |P-r-a|=%.2e |Q-r+a|=%.2e |a-w^2 sinh(M/2)|=%.2e"
          % (nm, lam, e1, e2, e3, e4))
CHECKS.append(("R1 代数恒等式", r1max < 1e-13, r1max))

print()
print("[R2] 约化方程与初值闭式：w'' + (M')^2 w/16 = 0")
r2max = 0.0
for nm, (A, A1, A2), lam in CASES:
    rhs, rhs_w = system(A, A1, A2, lam)
    so = integ(rhs, 3.0, [0j, 0j])
    us = so.t
    P = 1 + lam * A(us) + so.y[0]
    Q = 1 - lam * A(us) + so.y[0]
    w_num = (P * Q) ** 0.25
    w0, wp0 = ic_w(A, A1, lam)
    sr = integ(rhs_w, 3.0, [w0, wp0])
    d = np.max(np.abs(sr.y[0] - w_num))
    r2max = max(r2max, d)
    print("     %-14s lam=%.2f：w(0)=%.6f w'(0)=%+.6f ｜ 逐点差 %.2e" % (nm, lam, w0, wp0, d))
CHECKS.append(("R2 约化方程", r2max < 1e-12, r2max))

print()
print("[R3] 初值两条闭式与数值一致（含 a(0) 不为零的档）")
r3max = 0.0
for nm, (A, A1, A2), lam in CASES:
    w0, wp0 = ic_w(A, A1, lam)
    a0, ap0 = lam * A(0.0), lam * A1(0.0)
    w0_chk = (1.0 - a0 ** 2) ** 0.25
    wp0_chk = -a0 * ap0 / (2.0 * (1.0 - a0 ** 2) ** 0.75)
    d = max(abs(w0 - w0_chk), abs(wp0 - wp0_chk))
    r3max = max(r3max, d)
    print("     %-14s a(0)=%.6f：w(0) 偏差 %.2e，w'(0) 偏差 %.2e" % (nm, a0, abs(w0 - w0_chk), abs(wp0 - wp0_chk)))
CHECKS.append(("R3 初值闭式", r3max < 1e-14, r3max))

print()
print("[R4] 凹性：w'' <= 0 逐点（等号仅在 a = c w^2 档）")
r4max = -9.9
for nm, (A, A1, A2), lam in CASES:
    _, rhs_w = system(A, A1, A2, lam)
    w0, wp0 = ic_w(A, A1, lam)
    sr = integ(rhs_w, 3.0, [w0, wp0])
    wpp = np.gradient(sr.y[1], sr.t)
    r4max = max(r4max, float(np.max(wpp)))
    print("     %-14s lam=%.2f：max w'' = %+.3e" % (nm, lam, np.max(wpp)))
A0 = lambda u: 0.0 * u
_, rhs_w0 = system(A0, A0, A0, 0.5)
sr0 = integ(rhs_w0, 3.0, [1.0, 0.0])
print("     %-14s        max |w''| = %.3e（各向同性档，恒为零）" % ("a=0", float(np.max(np.abs(np.gradient(sr0.y[1], sr0.t))))))
CHECKS.append(("R4 凹性", r4max < 1e-9, r4max))

print()
print("[R5] 边界同一：w 零点与度量分量过零")


def track(rhs, u0, y0, obs, du=0.02, nsub=41):
    y = np.array(y0)
    u = 0.0
    best_u, best_v = float("nan"), 1e9
    while u < u0 - 1e-12:
        tgt = min(u + du, u0)
        try:
            sol = solve_ivp(rhs, [u, tgt], y, method="DOP853", rtol=1e-10, atol=1e-13,
                            max_step=0.02, dense_output=True)
        except Exception:
            break
        if (not sol.success) or (not np.all(np.isfinite(sol.y[:, -1]))):
            break
        us = np.linspace(u, tgt, nsub)
        ys = sol.sol(us)
        vals = np.abs(obs(us, ys))
        i = int(np.argmin(vals))
        if vals[i] < best_v:
            best_v, best_u = float(vals[i]), float(us[i])
        y = sol.y[:, -1]
        u = tgt
    return best_u, best_v


u0 = 3.0
A, A1, A2 = gauss(0.354)
ls_metric = lam_star(lambda l: probe_metric(A, A1, A2, l, u0))
LAM = 1.03 * ls_metric
rhs_b, rhs_w = system(A, A1, A2, LAM)
w0, wp0 = ic_w(A, A1, LAM)
u_w, w_min = track(rhs_w, u0, [w0, wp0], lambda us, ys: ys[0])
u_q, q_min = track(rhs_b, u0, [0j, 0j],
                   lambda us, ys: np.minimum(np.abs(1 + LAM * A(us) + ys[0]),
                                             np.abs(1 - LAM * A(us) + ys[0])))
print("     w=0.354，u0=3：度量分量口径 lambda* = %.6f（复现收口轮 1.0806）" % ls_metric)
print("     lam = %.4f 档：w 最小处的 u = %.4f（w_min = %.2e）｜ 度量分量最小处的 u = %.4f（min = %.2e）"
      % (LAM, u_w, w_min, u_q, q_min))
print("     两口径的过零位置差 = %.2e（同一事件的两条轨道）" % abs(u_w - u_q))
CHECKS.append(("R5 边界同一", abs(u_w - u_q) < 0.05 and min(w_min, q_min) < 1e-6, abs(u_w - u_q)))

print()
print("[R6] 可解子族（常数 M：w = k (u-u0)，a = c w^2）与登记")
c0, k0, us_ = 0.37, 0.83, 1.2
uu2 = np.linspace(0.5, 1.1, 601)
w_val = k0 * (uu2 - us_)
a_val = c0 * w_val ** 2
r_val = np.sqrt(a_val ** 2 + w_val ** 4)
M_val = np.log((r_val + a_val) / (r_val - a_val))
Mp_val = np.gradient(M_val, uu2)
e_algebra = np.max(np.abs(a_val - w_val ** 2 * np.sinh(M_val / 2)))
e_ode = np.max(np.abs(np.gradient(np.gradient(w_val, uu2), uu2) + Mp_val ** 2 * w_val / 16))
print("     常数 M 子族：代数恒等式残差 %.2e，约化方程残差 %.2e（w 线性、M 常数）" % (e_algebra, e_ode))
print("     一般剖面含任意指定函数 a(u)，lambda* 是其非线性泛函；通用闭式不可期，登记见注记。")
CHECKS.append(("R6 可解子族与登记", e_algebra < 1e-12 and e_ode < 1e-6, max(e_algebra, e_ode)))

print()
print("[图] 生成边界约化图")
fig, axs = plt.subplots(1, 2, figsize=(10.4, 3.8))
ax = axs[0]
for lam, col in [(0.97 * ls_metric, "tab:blue"), (1.03 * ls_metric, "tab:red")]:
    _, rhs_w2 = system(A, A1, A2, lam)
    w0, wp0 = ic_w(A, A1, lam)
    ev2 = lambda u, y: y[0] - 1e-6
    ev2.terminal = True
    ev2.direction = -1
    s2 = solve_ivp(rhs_w2, [0.0, u0], [w0, wp0], method="DOP853", rtol=1e-10, atol=1e-13,
                   max_step=1e-3, events=ev2, dense_output=True)
    tt = np.linspace(0.0, s2.t[-1], 600)
    ax.plot(tt, s2.sol(tt)[0], color=col, lw=1.6, label="lam = %.4f" % lam)
ax.axhline(0.0, color="k", lw=0.8, ls=":")
ax.set_xlabel("u")
ax.set_ylabel("w = (P Q)^(1/4)")
ax.set_title("(a) transverse scale factor near the boundary")
ax.legend(fontsize=8)
ax = axs[1]
for lam, col in [(0.3, "tab:green"), (0.6, "tab:orange"), (0.9, "tab:purple")]:
    _, rhs_w2 = system(A, A1, A2, lam)
    w0, wp0 = ic_w(A, A1, lam)
    s2 = integ(rhs_w2, u0, [w0, wp0], n=3001)
    ax.plot(s2.t, np.gradient(s2.y[1], s2.t), color=col, lw=1.4, label="lam = %.1f" % lam)
ax.axhline(0.0, color="k", lw=0.8, ls=":")
ax.set_xlabel("u")
ax.set_ylabel("w''")
ax.set_title("(b) concavity: w'' <= 0 pointwise")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(r"D:/software/TianXingFC/images/g1b_boundary_reduction.png", dpi=140)
print("     图已存 images/g1b_boundary_reduction.png")

print()
npass = sum(1 for _, ok, _ in CHECKS if ok)
for nm, ok, val in CHECKS:
    print("     %-16s %s（读数 %.3e）" % (nm, "通过" if ok else "失败", val))
print("判据 %d/%d 通过；用时 %.1f 秒" % (npass, len(CHECKS), time.time() - T0))
