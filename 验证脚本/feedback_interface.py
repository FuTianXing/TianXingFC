# -*- coding: utf-8 -*-
r"""
feedback_interface.py

[C1 负线项来源判据] 键能反馈通道能否生成等效负线项（判据 L-1 至 L-4）

问题：C1 需要一份界面优惠（负线项）以同时满足"驻留"与"宽度窗口"。三路线里
显式负项的代价是"新元素"。本判据核一件事：体系**既有**的键能反馈通道
kappa_e = kappa (1 + c s_e) 能否在界面处生成等效的负线项（即降低壁能）。

模型：一维链 N = 600，E = sum_e ½ kappa_e (Δu_e - 1)^2 + sum_i K (1 - cos 2 pi u_i)，
kappa = 1；站点记录 theta_i = u_i - x_i，链接记录 s_e = (theta_i + theta_{i+1})/2，
kappa_e = kappa (1 + c s_e)（下限 0.1 kappa，防负刚度）。自洽求解：先弛豫再更新
s_e，迭代至收敛（反馈剖面自洽，口径同本体层重建的反馈读数）。

判据（对照为同 K 下 c = 0 的普通壁）
L-1 符号：Delta E = E(反馈) - E(普通) 的符号（负 = 界面优惠）。
L-2 宽度：ell* 的变化（反馈把壁加宽还是收窄）。
L-3 幅度：|Delta E| / E_0 与等效线项系数的量级；刚度下限是否被触及（稳定性边界）。
L-4 结论：反馈通道能否充当等效负线项。

运行：python 验证脚本/feedback_interface.py
确定性、无图，运行约 10 秒。
"""
import numpy as np

N, KAP, ETA, STEPS_OUT, STEPS_IN = 600, 1.0, 0.005, 6, 4000

def relax(u, kappa_e, K, steps=STEPS_IN, eta=ETA):
    for _ in range(steps):
        du = np.diff(u)
        g = np.zeros(N)
        # 弹性项：kappa_e 逐链接
        e = du - 1.0
        gh = kappa_e * e                      # 链接上的力
        g[:-1] -= gh
        g[1:] += gh
        # PN 项
        g += K * 2.0 * np.pi * np.sin(2.0 * np.pi * u)
        g[0] = g[-1] = 0.0
        u = u - eta * g
        u[0], u[-1] = 0.0, float(N)
    return u

def energy(u, kappa_e, K):
    du = np.diff(u)
    return float(0.5 * (kappa_e * (du - 1.0) ** 2).sum() + K * (1.0 - np.cos(2 * np.pi * u)).sum())

def width(u):
    du = np.diff(u)
    dens = 0.5 * (du - 1.0) ** 2
    idx = np.arange(N - 1) + 0.5
    c0 = (dens * idx).sum() / dens.sum()
    return 2.0 * np.sqrt((dens * (idx - c0) ** 2).sum() / dens.sum())

def wall(c, K):
    x = np.arange(N, dtype=float)
    u = np.where(x < N // 2, x, x + 1.0)
    u[N // 2] = N / 2 + 0.5
    ke = np.ones(N - 1)
    floor_hit = False
    for _ in range(STEPS_OUT):
        u = relax(u, ke, K)
        s = 0.5 * ((u - x)[:-1] + (u - x)[1:])
        ke_new = KAP * (1.0 + c * s)
        if (ke_new < 0.1 * KAP).any():
            floor_hit = True
        ke = np.clip(ke_new, 0.1 * KAP, None)
    return u, ke, floor_hit

print("=" * 72)
print("C1 负线项来源判据：键能反馈通道的界面效应（L-1 至 L-4）")
print("=" * 72)
for K in (0.02, 0.005):
    u0, _, _ = wall(0.0, K)
    E0, w0 = energy(u0, np.ones(N - 1), K), width(u0)
    print(f"[K = {K}] 普通壁：E_0 = {E0:.6f}，ell* = {w0:.3f}")
    for c in (0.1, 0.3, 1.0):
        u, ke, hit = wall(c, K)
        E, w = energy(u, ke, K), width(u)
        dE = E - E0
        print(f"   c = {c:4.2f}：E = {E:.6f}，Delta E = {dE:+.6f}（{dE/E0*100:+.3f}%），"
              f"ell* = {w:.3f}，kappa_e 范围 [{ke.min():.3f}, {ke.max():.3f}]，下限触及 = {hit}")
print()
print()
print("[L-3 负号侧扫描] 若反馈系数取负（软化反馈），符号是否翻转、稳定性边界在哪里")
for K in (0.02, 0.005):
    u0, _, _ = wall(0.0, K)
    E0 = energy(u0, np.ones(N - 1), K)
    for c in (-0.1, -0.3, -1.0):
        u, ke, hit = wall(c, K)
        E, w = energy(u, ke, K), width(u)
        print(f"   K = {K:5.3f}, c = {c:5.2f}：Delta E = {E - E0:+.6f}（{(E - E0)/E0*100:+.3f}%），"
              f"kappa_e 范围 [{ke.min():.3f}, {ke.max():.3f}]，下限触及 = {hit}")

print("[L-1/L-4] 读数说明：Delta E > 0 表示反馈抬高壁能（界面反感），不能充当负线项；")
print("          Delta E < 0 才构成等效界面优惠。幅度与稳定性边界见上表两列。")
