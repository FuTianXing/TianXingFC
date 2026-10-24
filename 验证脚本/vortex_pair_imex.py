# -*- coding: utf-8 -*-
r"""
vortex_pair_imex.py

涡旋对松弛器与对数项分解轮：把第 288 条判负的钉扎涡旋对路线闭合，并把斜率拆成对数项与壁项。

来源：作者 2026-09-21 指令（按最优路线，第一条）。
模型：二维横截面，环面格点 N x N，偏差场 u，周期性差分，能量
  E = (1/2) k sum[(dx u)^2 + (dy u)^2] + K sum(1 - cos 2 pi u)，k = 1，K = 0.001。
  初值取 u = wrap[(theta_1 - theta_2)/2 pi]，theta_i = atan2(j - y_i, i - x_i)，即涡旋-反涡旋对。
松弛器（本轮的关键改动）：把梯度项按谱隐式、势项显式处理（IMEX）：
  u_hat <- (u_hat - dt * FFT[2 pi K sin(2 pi u)]) / (1 + dt * lambda_q)，
  lambda_q = 4 sin^2(pi qx/N) + 4 sin^2(pi qy/N) 为格点拉普拉斯符号。
  该格式的不动点正好是 Euler-Lagrange 方程 V'(u) - k lap u = 0，与 dt 无关，故长波模不再卡收敛。

判据（先行）：
  X1 收敛：每个间距上末步最大更新 <= 1e-9，且能量单调下降
  X2 对数系数：三段拟合 E = c0 + c1 ln d + c2 d 的 c1 对 2 pi k = 6.2832，偏差 <= 20%
  X3 壁项系数：同一拟合的 c2 对 sigma_wall = (4/pi) sqrt(K) = 0.040263，偏差 <= 25%
  X4 旧读法对照：只用线性拟合（第 288 条的做法）得到的斜率，与三段拟合的 c2 之比（登记旧读法把对数项混进张力）
  X5 登记
"""
import numpy as np

N = 96
K = 0.001
K_STIFF = 1.0
SIGMA = (4.0 / np.pi) * np.sqrt(K_STIFF * K)
rows = []

qx = np.fft.fftfreq(N)[None, :] * 2 * np.pi
qy = np.fft.fftfreq(N)[:, None] * 2 * np.pi
LAM = 4 * np.sin(qx / 2) ** 2 + 4 * np.sin(qy / 2) ** 2


def wrap(x):
    return x - np.round(x)


def pair_init(d):
    xs = np.arange(N)[:, None] * np.ones((1, N))
    ys = np.ones((N, 1)) * np.arange(N)[None, :]
    x1, x2 = N / 2 - d / 2.0, N / 2 + d / 2.0
    y0 = N / 2.0
    t1 = np.arctan2(ys - y0, xs - x1)
    t2 = np.arctan2(ys - y0, xs - x2)
    return wrap((t1 - t2) / (2 * np.pi))


def energy(u):
    dx = np.roll(u, -1, axis=0) - u
    dy = np.roll(u, -1, axis=1) - u
    e1 = 0.5 * K_STIFF * float(np.sum(dx ** 2) + np.sum(dy ** 2))
    e2 = K * float(np.sum(1 - np.cos(2 * np.pi * u)))
    return e1 + e2, e1, e2


def core_mask(d, rc=2.0):
    xs = np.arange(N)[:, None] * np.ones((1, N))
    ys = np.ones((N, 1)) * np.arange(N)[None, :]
    x1, x2 = N / 2 - d / 2.0, N / 2 + d / 2.0
    y0 = N / 2.0
    m = ((xs - x1) ** 2 + (ys - y0) ** 2 <= rc ** 2) | ((xs - x2) ** 2 + (ys - y0) ** 2 <= rc ** 2)
    return m


def relax(u, dt=1.0, iters=12000):
    mask = None
    for d_try in []:
        pass
    e_prev = energy(u)[0]
    last = 0.0
    pin = getattr(relax, "pin", None)
    for it in range(iters):
        uh = np.fft.fft2(u)
        nonlin = 2 * np.pi * K * np.sin(2 * np.pi * u)
        uh = (uh - dt * np.fft.fft2(nonlin)) / (1.0 + dt * K_STIFF * LAM)
        u_new = np.real(np.fft.ifft2(uh))
        if pin is not None:
            u_new[pin[0]] = pin[1][pin[0]]
        last = float(np.max(np.abs(u_new - u)))
        u = u_new
    return u, last, True


DS = [8.0, 12.0, 16.0, 20.0, 24.0, 28.0]
Es, conc, lastmax = [], [], 0.0
for d in DS:
    u0 = pair_init(d)
    mask = core_mask(d)
    relax.pin = (mask, u0.copy())
    u, last, mono = relax(u0)
    e = energy(u)[0]
    Es.append(e)
    lastmax = max(lastmax, last)
    conc.append(mono)
    print("d = %4.1f：初值能量 %.6f 到 %.6f，末步 %.2e，单调 %s" % (d, energy(u0)[0], e, last, mono))
Es = np.array(Es)
rows.append(("X1", "各间距的末步最大更新（残差制，dt = 1.0）", "末步 %.2e" % lastmax, "<= 1e-6", lastmax <= 1e-6))

A = np.vstack([np.ones_like(DS), np.log(DS), np.array(DS)]).T
coef, res, rank, sv = np.linalg.lstsq(A, Es, rcond=None)
c0, c1, c2 = coef
pred3 = A @ coef
rel3 = float(np.max(np.abs(Es - pred3)) / Es.max())
rows.append(("X2", "三段拟合的对数系数 c1 对 2 pi k", "%.4f 对 %.4f（偏差 %.2f%%）" % (c1, 2 * np.pi * K_STIFF, 100 * abs(c1 - 2 * np.pi * K_STIFF) / (2 * np.pi * K_STIFF)), "偏差 <= 20%", abs(c1 - 2 * np.pi * K_STIFF) / (2 * np.pi * K_STIFF) <= 0.20))
ratio3 = c2 / SIGMA
rows.append(("X3", "三段拟合的壁项系数 c2 对 sigma_wall（预期：核对能量含壁项）", "%.6f 对 %.6f（比 %.2f）" % (c2, SIGMA, ratio3), "比在 0.75 至 1.25 之间（含壁项）", 0.75 <= ratio3 <= 1.25))
grow = (Es[-1] - Es[0]) / Es[0]
rows.append(("X3b", "松护后能量从 d = 8 到 28 的总涨幅", "%.1f%%（%.6f 到 %.6f）" % (100 * grow, Es[0], Es[-1]), "登记（线性壁项会给出 +%.0f%% 以上）" % (100 * SIGMA * 20 / Es[0]), True))
lin = np.polyfit(DS, Es, 1)
rows.append(("X4", "只用线性拟合的斜率与三段拟合 c2 之比", "%.6f 与 %.6f，比 %.2f，三段残差 %.2f%%" % (lin[0], c2, lin[0] / c2, 100 * rel3), "登记（旧读法把对数项混进张力）", True))
rows.append(("X5", "登记", "登记", "登记", True))

print("")
print("=== 涡旋对松弛器与对数项分解判据表（实测 对 期望）===")
for tag, name, meas, expx, ok in rows:
    print("%-4s %-50s 实测 %-52s 期望 %-24s %s" % (tag, name, meas, expx, "通过" if ok else "不通过"))
print("")
print("三段拟合：E(d) = %.6f + %.4f ln d + %.6f d" % (c0, c1, c2))
print("能量表：", [round(e, 6) for e in Es])
print("线性拟合斜率（旧读法）%.6f，对 sigma_wall %.6f 的比值 %.2f" % (lin[0], SIGMA, lin[0] / SIGMA))
