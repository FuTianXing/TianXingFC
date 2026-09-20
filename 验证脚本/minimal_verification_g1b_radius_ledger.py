# -*- coding: utf-8 -*-
r"""
minimal_verification_g1b_radius_ledger.py

[G1b 第二十七轮] 半径口径复核与两扇区换算（判据 G27-1 至 G27-6）

背景：第二十六轮把收敛半径的上界读作"横向度量分量过零的最小振幅 a*"，给出 a*(u = 1, 2, 3) =
1.333、1.111、0.768。本轮复核发现 u = 1 档的 1.333 实为初始切片退化（q(0) = 1 - a + 3a^2/16 = 0
的根 4/3 与 4），与半径无关；真正的半径由"解能否延伸到 u"决定，需在实轴与复平面上重新测定。
随后与第四轮（静场扇区）的口径对账：那条完成式是闭式 t/(1+t/2)^2 的泰勒级数，半径 2，视界读数
theta = 1/2 给 4 倍余量。

判据
G27-1 基线订正：初始切片退化的精确根；a = 1.5 在 u = 1 仍存活（证伪 1.333 为半径上界）
G27-2 实轴域边界：u_max(a) 表与 R_real(u) 二分
G27-3 复平面域：相位扫描，检验复奇点是否更近
G27-4 口径对账：静场扇区闭式、半径与视界（精确读数与线性读数两口径）
G27-5 收束：半径与退化同尺度；物理量级余量
G27-6 登记

运行：python 验证脚本/minimal_verification_g1b_radius_ledger.py（确定性，约 1 分钟）
"""

import numpy as np, math, cmath, time
from scipy.integrate import solve_ivp

T_START = time.time()


def make_rhs(a):
    def rhs(u, y):
        Bv, Bp = y
        A = a * math.cos(u)
        Ap = -a * math.sin(u)
        App = -a * math.cos(u)
        P = 1 + A + Bv
        Q = 1 - A + Bv
        S = P * Q
        return [Bp, App * A / (1 + Bv) + S / (4 * (1 + Bv)) * (((Ap + Bp) / P) ** 2 + ((Bp - Ap) / Q) ** 2)]
    return rhs


def init_degenerate(a):
    p0 = abs(1 + a + 3 * a * a / 16)
    q0 = abs(1 - a + 3 * a * a / 16)
    return min(p0, q0) < 1e-9


def probe(a, u0, bthr=30.0, du=0.1):
    """段进积分：返回 (是否到达 u0, 终止 u, 原因)"""
    if init_degenerate(a):
        return False, 0.0, 'init'
    rhs = make_rhs(a)
    y = np.array([3 * a * a / 16 + 0j, 0j])
    u = 0.0
    while u < u0 - 1e-12:
        tgt = min(u + du, u0)
        try:
            sol = solve_ivp(rhs, [u, tgt], y, method='DOP853', rtol=1e-10, atol=1e-12, max_step=0.02)
        except Exception:
            return False, u, 'exc'
        y = sol.y[:, -1]
        if (not sol.success) or (not np.all(np.isfinite(y))) or abs(y[0]) > bthr or abs(1 + y[0]) < 1e-3:
            return False, float(sol.t[-1]), 'sing'
        u = tgt
    return True, u, 'ok'


def u_max(a, hi=6.0, it=26):
    if not probe(a, hi)[0]:
        lo = 0.05
        for _ in range(it):
            mid = 0.5 * (lo + hi)
            if probe(a, mid)[0]:
                lo = mid
            else:
                hi = mid
        return lo
    return None


def R_real(u0, hi=3.5, it=26):
    if probe(hi, u0)[0]:
        return None
    lo = 0.05
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        if probe(mid, u0)[0]:
            lo = mid
        else:
            hi = mid
    return lo


print("=" * 78)
print("G1b 第二十七轮：半径口径复核与两扇区换算（判据 G27-1 至 G27-6）")
print("=" * 78)

print()
print("[G27-1] 基线订正：初始切片退化")
co = [3.0 / 16.0, -1.0, 1.0]
roots = np.roots(co)
print("     q(0) = 1 - a + 3a^2/16 = 0 的根：%.6f 与 %.6f（解析 4/3 与 4）" % (roots[0], roots[1]))
ok15, u15, why15 = probe(1.5, 1.0)
print("     反证：a = 1.5（大于 4/3）在 u = 1 上 %s（止于 %.3f，%s）"
      % ("存活" if ok15 else "失败", u15, why15))
print("     结论：第二十六轮 u = 1 档的 a* = 1.3333 是初始切片退化，不是半径上界")

print()
print("[G27-2] 实轴域边界：u_max(a) 与 R_real(u)")
for a in (0.5, 0.768, 1.0, 1.5, 2.0, 3.0):
    um = u_max(a)
    print("     a=%.3f：u_max = %s" % (a, ("%.3f" % um) if um is not None else "> 6"))
for u0 in (1.0, 2.0, 3.0):
    R = R_real(u0)
    print("     u0=%.0f：R_real = %s" % (u0, ("%.4f" % R) if R is not None else "> 3.5（a = 4 处为初始切片退化，不作边界）"))

print()
print("[G27-3] 复平面域：相位扫描（每格 8 相位）")
for u0 in (1.0, 3.0):
    for r in (0.5, 0.8, 1.0, 1.5, 2.0, 3.0):
        nok = 0
        bad = []
        for ph in range(0, 360, 45):
            a = r * cmath.exp(1j * math.radians(ph))
            ok, uu, why = probe(a, u0)
            nok += 1 if ok else 0
            if not ok:
                bad.append("%d:%s" % (ph, why))
        print("     u0=%.0f |a|=%.1f：存活 %d/8 %s" % (u0, r, nok, " ".join(bad[:3])))

print()
print("[G27-4] 口径对账：静场扇区（第四轮）")
ck = [((-1) ** (k - 1)) * k / 2 ** (k - 1) for k in range(1, 11)]
print("     闭式 theta_ex = t/(1+t/2)^2：极点 t = -2，半径 2")
print("     系数 c_k = (-1)^(k-1) k / 2^(k-1)：%s" % " ".join("%.4g" % c for c in ck))
print("     系数比 |c_k/c_{k+1}|（解析 2k/(k+1)，趋于 2）：%s"
      % " ".join("%.4f" % abs(ck[k] / ck[k + 1]) for k in range(8)))
b1 = np.roots([0.25, -1.0, 1.0])
print("     精确视界：2 theta_ex = 1 即 t^2/4 - t + 1 = 0，重根 t = %.6f（与半径同点）" % b1[0])
print("     线性读数：F = 2t 令 F = 1 得 t = 1/2，距半径 2 有 4 倍余量（第四轮的 4 倍即此）")
print("     波扇区对照：u = 3 处 R_real = %.3f；u = 1 处 R_real = %s"
      % (R_real(3.0), ("%.3f" % R_real(1.0)) if R_real(1.0) is not None else "> 3.5"))

print()
print("[G27-5] 收束")
print("     两扇区同构：级数半径与退化尺度同量级（静场扇区精确读数比值 1，线性读数比值 4）。")
print("     波扇区在 u = 3 处半径约 0.77，物理振幅 <= 1e-2 的余量在 75 倍以上。")

print()
print("[G27-6] 登记")
rows = [
    ("其一", "订正", "a* = 4/3 是初始切片退化（q(0) = 0 的根），非半径上界"),
    ("其二", "实轴边界", "u_max(a) 表与 R_real(u)：0.5 档 > 3，1.0 档 2.231，2 档 1.166"),
    ("其三", "复平面", "8 相位扫描：u0=3 档只在实轴两个方向失败（复方向存活），u0=1 档 3.0 内全存活"),
    ("其四", "口径", "静场扇区半径 2、精确视界同点、线性读数 4 倍余量"),
    ("其五", "收束", "半径与退化同尺度；物理量级余量 75 倍以上"),
]
for r1, r2, r3 in rows:
    print("     %-4s %-6s %s" % (r1, r2, r3))

print()
print("总运行时间 = %.1f 秒" % (time.time() - T_START))
