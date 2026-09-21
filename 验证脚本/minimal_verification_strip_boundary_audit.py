# -*- coding: utf-8 -*-
r"""
minimal_verification_strip_boundary_audit.py

条带分歧三向对照与 x 向边界修正轮：定位第 333 条「两次自测未复现」的成因。

来源：作者 2026-09-21 指令（执行第 333 条登记的三向对照）。
三个实现，同一几何（竖条 64x128，x 两端固定 u=0 与 1，y 周期）、同一初值（线性斜坡）、同一能量泛函：
  A 第 286 条脚本的原实现（x 用显式差分拉普拉斯、y 用 roll 差分拉普拉斯、dt = 0.2、六万步、正确 Dirichlet）
  B 第 330／333 条的 FFT 隐式实现（x 与 y 都用周期拉普拉斯——x 方向的周期性与几何不符）
  C 本轮修正实现（x 用显式差分拉普拉斯并保留固定边界，y 用 FFT 隐式、势项线性化部分隐式）

判据（先行）：
  W1 A 复现第 286 条读数（K = 0.001 给 0.040242），偏差 <= 1%
  W2 C 与 A 一致（同档偏差 <= 1%）
  W3 B 与 A 之差显著（> 10%），即证实 x 向周期性是分歧成因
  W4 偏差随 K 的模式：K = 0.02 与 0.001 两档上 B 的偏差之比（登记；宽壁受影响更大）
  W5 登记
"""
import numpy as np

NX, NY = 64, 128
REF = {0.02: 0.177866, 0.001: 0.040242}
rows = []


def energy(u, K, k=1.0):
    dx = u[1:, :] - u[:-1, :]
    dy = np.roll(u, -1, axis=1) - u
    e1 = 0.5 * k * float(np.sum(dx ** 2) + np.sum(dy ** 2))
    e2 = K * float(np.sum(1 - np.cos(2 * np.pi * u)))
    return e1 + e2, e1, e2


def init():
    u = np.tile(np.linspace(0.0, 1.0, NX)[:, None], (1, NY))
    u[0, :] = 0.0
    u[-1, :] = 1.0
    return u


def relax_A(K, iters=60000, dt=0.2, k=1.0):
    u = init()
    for it in range(iters):
        lap = np.zeros_like(u)
        lap[1:-1, :] = u[2:, :] + u[:-2, :] - 2.0 * u[1:-1, :]
        lap[1:-1, :] += np.roll(u[1:-1, :], -1, axis=1) + np.roll(u[1:-1, :], 1, axis=1) - 2.0 * u[1:-1, :]
        g = -k * lap + 2.0 * np.pi * K * np.sin(2.0 * np.pi * u)
        g[0, :] = 0.0
        g[-1, :] = 0.0
        step = dt * g
        u = u - step
        u[0, :] = 0.0
        u[-1, :] = 1.0
    return u, float(np.max(np.abs(step)))


def relax_B(K, iters=200000, k=1.0):
    u = init()
    dt = min(0.4, 1.0 / (2 * np.pi ** 2 * K))
    W = (2 * np.pi) ** 2 * K
    fx = np.fft.fftfreq(NX)[:, None] * np.pi
    fy = np.fft.fftfreq(NY)[None, :] * np.pi
    lam = 4 * np.sin(fx) ** 2 + 4 * np.sin(fy) ** 2
    last = 0.0
    for it in range(iters):
        uh = np.fft.fft2(u)
        nonlin = 2 * np.pi * K * np.sin(2 * np.pi * u) - W * u
        uh = (uh - dt * np.fft.fft2(nonlin)) / (1.0 + dt * (k * lam + W))
        un = np.real(np.fft.ifft2(uh))
        un[0, :] = 0.0
        un[-1, :] = 1.0
        last = float(np.max(np.abs(un - u)))
        u = un
        if last < 1e-13:
            break
    return u, last


def relax_C(K, iters=200000, dt=0.15, k=1.0):
    u = init()
    W = (2 * np.pi) ** 2 * K
    lam_y = 4 * np.sin(np.fft.fftfreq(NY)[None, :] * np.pi) ** 2
    last = 0.0
    for it in range(iters):
        # x 方向：显式差分拉普拉斯（保留 Dirichlet 固定边界，不用周期）
        lap_x = np.zeros_like(u)
        lap_x[1:-1, :] = u[2:, :] + u[:-2, :] - 2.0 * u[1:-1, :]
        # y 方向：谱隐式
        uh = np.fft.rfft(u, axis=1)
        non = 2 * np.pi * K * np.sin(2 * np.pi * u) - W * u
        nh = np.fft.rfft(non, axis=1)
        lx = np.fft.rfft(lap_x, axis=1)
        uh = (uh + dt * (k * lx - nh)) / (1.0 + dt * (k * lam_y[:, :uh.shape[1]] + W))
        un = np.fft.irfft(uh, n=NY, axis=1)
        un[0, :] = 0.0
        un[-1, :] = 1.0
        last = float(np.max(np.abs(un - u)))
        u = un
        if last < 1e-13:
            break
    return u, last


out = {}
for K in (0.02, 0.001):
    uA, lA = relax_A(K)
    uB, lB = relax_B(K)
    uC, lC = relax_C(K)
    eA = energy(uA, K)[0] / NY
    eB = energy(uB, K)[0] / NY
    eC = energy(uC, K)[0] / NY
    out[K] = (eA, eB, eC, lA, lB, lC)
    print("K = %.3f：A（第 286 条原实现）%.6f（残差 %.1e）；B（周期 x 的 FFT 隐式）%.6f（%.1e）；C（本轮修正）%.6f（%.1e）" % (K, eA, lA, eB, lB, eC, lC))
    print("          第 286 条登记值 %.6f；B 对 A 偏差 %.1f%%；C 对 A 偏差 %.2f%%" % (REF[K], 100 * abs(eB - eA) / eA, 100 * abs(eC - eA) / eA))

eA1, eB1, eC1 = out[0.001][0], out[0.001][1], out[0.001][2]
eA2, eB2 = out[0.02][0], out[0.02][1]
rows.append(("W1", "A 对第 286 条 K = 0.001 登记值 0.040242", "%.6f（偏差 %.2f%%）" % (eA1, 100 * abs(eA1 - REF[0.001]) / REF[0.001]), "偏差 <= 1%", abs(eA1 - REF[0.001]) / REF[0.001] <= 0.01))
rows.append(("W2", "C（修正后）对 A，K = 0.001", "偏差 %.2f%%" % (100 * abs(eC1 - eA1) / eA1), "<= 1%", abs(eC1 - eA1) / eA1 <= 0.01))
rows.append(("W3", "B（周期 x）对 A，K = 0.001", "偏差 %.1f%%" % (100 * abs(eB1 - eA1) / eA1), "> 10%（证实成因）", abs(eB1 - eA1) / eA1 > 0.10))
d2 = abs(eB2 - eA2) / eA2
rows.append(("W4", "B 的偏差随 K：K = 0.02 与 0.001", "%.1f%% 与 %.1f%%（比 %.1f）" % (100 * d2, 100 * abs(eB1 - eA1) / eA1, (abs(eB1 - eA1) / eA1) / d2), "登记（宽壁受影响更大）", True))
rows.append(("W5", "登记", "登记", "登记", True))

print("")
print("=== 条带分歧三向对照判据表（实测 对 期望）===")
for tag, name, meas, expx, ok in rows:
    print("%-4s %-40s 实测 %-48s 期望 %-26s %s" % (tag, name, meas, expx, "通过" if ok else "不通过"))
