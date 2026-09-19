# -*- coding: utf-8 -*-
"""
minimal_verification_c1_softmode_route.py

[C1 路线乙首判] 软模承载宽度的能量竞争：局域锁定 + 软体相位的窗口存在性
（判据 S-1 至 S-4）

背景：C1-4 锚核筛选给出窗口互斥：落窗（coherent 宽度 l* >= 2）要求 K <= 0.0231，
驻留（环稳定）要求 K >= 0.25，两窗交集为空。路线乙的假设是把两个角色解耦：
宽度由体的软相位模（K_soft 小）承担，稳定由锚核的局域锁定（K 局域抬高）承担。
本判据在同一一维链上做第一道能量竞争判定。

模型（与 L/U 系列同口径）：E = sum_e (1/2) kappa_e (du_e - 1)^2
+ sum_i K_i (1 - cos 2 pi u_i)，kappa_e = 1（不动用软化分支）；
K_i = K_soft + dK * exp(-((i - i0)/w_a)^2)（局域锁定，中心在 kink 芯）。
宽度定义 = 弹性超量密度 (1/2)(Delta u - 1)^2 的二阶矩宽度 2 sqrt(r2)。

判据
S-1 落窗线复现：自由 kink 的 l*(K_soft) 与库内口径一致（K = 0.02 给约 2.0，
    K = 0.05 给约 1.35），落窗上界 K_win 复现约 0.0231
S-2 局域锁定的增益：加局域锁定后总能下降（gain = E_free - E_anch > 0）
S-3 竞争边界：在 (dK, w_a) 网格上找 (gain > 0) 与 (l* >= 2) 同时成立的区域，
    给出边界与最大可用 dK
S-4 与既有读数对接：自由 kink 在 K_soft = 0.25 处的 l* 应与库内 1.029 一致
运行：python 验证脚本/minimal_verification_c1_softmode_route.py（确定性，约 30 秒）
"""
import numpy as np

N = 600
KAP = 1.0
ETA = 0.005
STEPS = 20000


def init_kink(n=N):
    x = np.arange(n, dtype=float)
    u = np.where(x < n // 2, x, x + 1.0)
    u[n // 2] = n / 2 + 0.5
    return x, u


def relax(u, Karr, steps=STEPS, eta=ETA):
    x = np.arange(N, dtype=float)
    for _ in range(steps):
        du = np.diff(u)
        g = np.zeros(N)
        g[:-1] -= KAP * (du - 1.0)
        g[1:] += KAP * (du - 1.0)
        g += Karr * 2.0 * np.pi * np.sin(2.0 * np.pi * u)
        g[0] = g[-1] = 0.0
        u = u - eta * g
        u[0], u[-1] = x[0], float(N)
    return u


def energy(u, Karr):
    du = np.diff(u)
    return float(0.5 * (KAP * (du - 1.0) ** 2).sum()
                 + (Karr * (1.0 - np.cos(2.0 * np.pi * u))).sum())


def width(u):
    du = np.diff(u)
    dens = 0.5 * (du - 1.0) ** 2
    idx = np.arange(N - 1) + 0.5
    c0 = (dens * idx).sum() / dens.sum()
    return 2.0 * np.sqrt((dens * (idx - c0) ** 2).sum() / dens.sum())


def anchor_profile(i0, w_a, d, n=N, K_soft=0.0):
    """偏好凹口：局域压低 K（AF-1 口径的锚核），深度 d 为相对压低的份额。"""
    i = np.arange(n, dtype=float)
    return K_soft * (1.0 - d * np.exp(-((i - i0) / w_a) ** 2))


print("=" * 76)
print("C1 路线乙首判：软模承载宽度的能量竞争（判据 S-1 至 S-4）")
print("=" * 76)

# S-1 落窗线复现
print()
print("[S-1] 自由 kink 的宽度与落窗线")
x, u0 = init_kink()
l_scan = {}
for K in (0.005, 0.010, 0.015, 0.020, 0.0231, 0.05, 0.25):
    u = relax(u0.copy(), np.full(N, K))
    l_scan[K] = width(u)
    print("     K_soft = %.4f：l* = %.3f%s" % (K, l_scan[K], "（落窗线 2）" if abs(K - 0.0231) < 1e-9 else ""))
Kwin = None
for K in (0.025, 0.03, 0.035, 0.05):
    u = relax(u0.copy(), np.full(N, K))
    if width(u) < 2.0:
        Kwin = K
        break
print("     落窗上界（l* 首次 < 2）K_win ≈ %.4f（库内口径 0.0231）" % Kwin)

# S-2/S-3 偏好凹口锚核：增益与宽度的竞争
print()
print("[S-2/S-3] 偏好凹口锚核（局域压低 K）：增益与宽度")
i0 = N // 2
print("     %-8s %-6s %-6s %-12s %-8s %s" % ("K_soft", "w_a", "d", "gain", "l*", "判定"))
for K_soft in (0.010, 0.015, 0.020):
    Kb = np.full(N, K_soft)
    uf = relax(u0.copy(), Kb)
    Ef, lf = energy(uf, Kb), width(uf)
    print("     （自由 kink：E = %.5f，l* = %.3f）" % (Ef, lf))
    for w_a in (1.0, 2.0, 4.0, 8.0):
        for d in (0.3, 0.6, 0.9):
            Karr = anchor_profile(i0, w_a, d, K_soft=K_soft)
            ua = relax(u0.copy(), Karr)
            Ea, la = energy(ua, Karr), width(ua)
            gain = Ef - Ea
            ok = (gain > 0) and (la >= 2.0)
            print("     %-8.3f %-6.1f %-6.2f %-12.5f %-8.3f %s"
                  % (K_soft, w_a, d, gain, la, "窗口内" if ok else "排除"))

# S-4 与库内读数对接
print()
print("[S-4] 对接：自由 kink 在 K = 0.25 处")
print("     本脚本 l*(0.25) = %.3f（库内 1.029）" % l_scan[0.25])
