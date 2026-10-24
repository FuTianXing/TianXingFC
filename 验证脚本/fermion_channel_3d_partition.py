# -*- coding: utf-8 -*-
r"""
fermion_channel_3d_partition.py

[费米子通道第四轮] 三维面壁的能量均分核对（判据 P1 至 P4）

定位：第二轮的 c_el = 1/2 是一维 FK 链读数；本轮在三维 FK 格点上放一面平面壁
（一维 kink 的三维最近亲），核对（一）复合结构是否稳定存在、（二）宽度标度是否同指数、
（三）能量均分是否仍为 1/2。

边界与分工：真正的三维粒子形态（点/线缺陷与三元组哈密顿量）归 C1 线（任务 C1-6，
未启动，依赖 C1-5）；本轮只做面壁情形，不进入 C1 的形态与驻留判定。

模型：三维 FK 格点 N^3，E = sum[ (k/2) sum_axes (d_axis u - delta_{axis,x})^2 + K (1 - cos 2 pi u) ]，
k = 1，x 向固定边界 u(0) = 0、u(N-1) = N（一条平面壁），y、z 向周期。
读数与一维逐项对齐：ell* = 2 x 弹性超量密度的惯性半径（沿 x 的列平均剖面）。

判据
P1 稳定存在：弛豫后 x 剖面单调、缠绕保持（u(N-1) - u(0) = N）。
P2 宽度标度：ell*(3D) 与一维同 K 读数一致（偏差 <= 5%）。
P3 能量均分：c_el = E_el / (E_el + E_pn) 落在 0.5 +- 0.03。
P4 边界登记：点/线缺陷与三元组哈密顿量归 C1 线，本轮不判。

运行：python 验证脚本/fermion_channel_3d_partition.py
确定性、无图，运行约 20 秒。
"""
import numpy as np

N = 32
K_EL = 1.0
ETA = 0.01
STEPS = 8000
X = np.arange(N, dtype=float).reshape(N, 1, 1)
ONES_YZ = np.ones((1, N, N), dtype=float)

def relax(K, steps=STEPS):
    u = np.broadcast_to(X, (N, N, N)).astype(float).copy()
    half = N // 2
    u[half:, :, :] += 1.0
    u[half, :, :] = half + 0.5
    for _ in range(steps):
        g = np.zeros_like(u)
        dx = np.zeros_like(u)
        dx[:-1, :, :] += u[1:, :, :] - u[:-1, :, :] - 1.0
        dx[1:, :, :] -= u[1:, :, :] - u[:-1, :, :] - 1.0
        g -= K_EL * dx
        for axis in (1, 2):
            d = np.roll(u, -1, axis=axis) - u
            g -= K_EL * (d - np.roll(d, 1, axis=axis))
        g += K * 2.0 * np.pi * np.sin(2.0 * np.pi * u)
        g[0, :, :] = 0.0
        g[-1, :, :] = 0.0
        u = u - ETA * g
        u[0, :, :] = 0.0
        u[-1, :, :] = float(N)
    return u

def readout(u, K):
    dx = np.diff(u, axis=0)
    dy = np.roll(u, -1, axis=1) - u
    dz = np.roll(u, -1, axis=2) - u
    rho = 0.5 * K_EL * (((dx - 1.0) ** 2).sum(axis=(1, 2))
                        + (dy[:N - 1] ** 2).sum(axis=(1, 2))
                        + (dz[:N - 1] ** 2).sum(axis=(1, 2))) / (N * N)
    idx = np.arange(N - 1) + 0.5
    c0 = (rho * idx).sum() / rho.sum()
    ell = 2.0 * np.sqrt((rho * (idx - c0) ** 2).sum() / rho.sum())
    E_el = float(0.5 * K_EL * (((dx - 1.0) ** 2).sum() + (dy ** 2).sum() + (dz ** 2).sum()))
    E_pn = float(K * (1.0 - np.cos(2.0 * np.pi * u)).sum())
    return ell, E_el, E_pn

print("=" * 72)
print("费米子通道第四轮：三维面壁的能量均分核对（P1 至 P4）")
print("=" * 72)
one_d = {0.005: 4.06, 0.01: 2.86, 0.02: 2.01}
for K in [0.005, 0.01, 0.02]:
    u = relax(K)
    prof = u[:, 0, 0]
    mono = bool(np.all(np.diff(prof) > 0))
    wind = float(u[-1, 0, 0] - u[0, 0, 0])
    ell, E_el, E_pn = readout(u, K)
    c_el = E_el / (E_el + E_pn)
    print(f"K = {K:6.3f}: 单调 {mono}，缠绕 {wind:.0f} | ell*(3D) = {ell:.2f}（一维 {one_d[K]:.2f}）"
          f" | c_el = {c_el:.4f}（E_el = {E_el:.4f}, E_pn = {E_pn:.4f}）")
print()
print("[P1] 稳定存在：见上表两列（单调与缠绕）")
print("[P2] 宽度标度：逐值比对一维读数")
print("[P3] 能量均分：c_el 对 1/2 的偏差见上表")
print("[P4] 边界登记：点/线缺陷与三元组哈密顿量（C1-6）归 C1 线，本轮不判")
