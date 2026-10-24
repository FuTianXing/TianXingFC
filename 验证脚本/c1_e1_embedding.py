# -*- coding: utf-8 -*-
r"""
c1_e1_embedding.py

[C1 三维嵌入 第一判据 E1] 条带几何的弦张力闭合、横截面分离与沿线稳定性
（判据 E1-1 至 E1-5）

背景：C1 驻留判据在解析环构型上失效（第 342 条：边界相位漂移占拉力 84% 至 86%），
下一步改用强制净跳变的 Dirichlet 条带几何。本判据同时是引力线三维嵌入的第一道：
三维构型若是"二维解沿 z 的直积"，横截面积分才可承载一维源项。

模型（与 C1 各轮同口径）：u = IX + IY + phi，
E = (1/2) sum_axes (d_axis u - 1)^2 + sum K (1 - cos 2 pi u)，k = 1。
条带：x 两端 Dirichlet（phi = 0 与 1），y 周期（横截面内自由），z 周期（沿线）。

判据
E1-1 横截面分离：从 y 无关初态弛豫到收敛，max|d_y phi| 为机器零，
     且 sigma = E / Ny 对一维精确解 sigma_1D 的偏差 <= 0.05%
E1-2 横截面稳定性：在收敛态上加 1e-3 的 y 调制种子，弛豫后衰减（比值 <= 0.5）
E1-3 尺寸稳健性：Ny 加倍时 sigma 变化 <= 0.5%
E1-4 三维直积：NZ = 8 层，逐层偏差与能量可加性 <= 1e-12，z 向梯度能量为机器零
E1-5 沿线稳定性：1e-3 的 z 调制（波长 4 与 8 层）都衰减，短波长衰减更快

运行：python 验证脚本/c1_e1_embedding.py（确定性，约 30 秒）
"""
import math
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "D:/software/TianXingFC/验证脚本")
import largeK_wall_closed_form as lk

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

T0 = time.time()
TWO_PI = 2.0 * math.pi
IMG = "D:/software/TianXingFC/images/c1_e1_embedding.png"
CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))


def grad2(phi, K):
    g = np.zeros_like(phi)
    d = phi[1:, :] - phi[:-1, :]
    g[:-1, :] -= d
    g[1:, :] += d
    dy = np.roll(phi, -1, axis=1) - phi
    g -= dy
    g += np.roll(dy, 1, axis=1)
    g += TWO_PI * K * np.sin(TWO_PI * phi)
    g[0, :] = 0.0
    g[-1, :] = 0.0
    return g


def energy2(phi, K):
    d = phi[1:, :] - phi[:-1, :]
    dy = np.roll(phi, -1, axis=1) - phi
    return 0.5 * float(np.sum(d ** 2) + np.sum(dy ** 2)) + float(np.sum(K * (1.0 - np.cos(TWO_PI * phi))))


def relax2(phi, K, eta, steps, tol=1e-10):
    g = grad2(phi, K)
    for it in range(steps):
        g = grad2(phi, K)
        phi = phi - eta * g
        phi[0, :] = 0.0
        phi[-1, :] = 1.0
        if it % 200 == 0 and np.max(np.abs(g[1:-1, :])) < tol:
            break
    return phi, g, it + 1


def grad3(phi, K):
    g = np.zeros_like(phi)
    d = phi[1:, :, :] - phi[:-1, :, :]
    g[:-1, :, :] -= d
    g[1:, :, :] += d
    dy = np.roll(phi, -1, axis=1) - phi
    g -= dy
    g += np.roll(dy, 1, axis=1)
    dz = np.roll(phi, -1, axis=2) - phi
    g -= dz
    g += np.roll(dz, 1, axis=2)
    g += TWO_PI * K * np.sin(TWO_PI * phi)
    g[0, :, :] = 0.0
    g[-1, :, :] = 0.0
    return g


def relax3(phi, K, eta, steps, every, seed_shape):
    amps = []
    for it in range(steps):
        g = grad3(phi, K)
        phi = phi - eta * g
        phi[0, :, :] = 0.0
        phi[-1, :, :] = 1.0
        if it % every == 0:
            amps.append(float(np.sum(seed_shape * phi) / np.sum(seed_shape ** 2)))
    return phi, amps


print("=" * 78)
print("C1 三维嵌入 E1：条带几何的弦张力闭合、横截面分离与沿线稳定性")
print("=" * 78)

NX, NY = 64, 32
x = np.arange(NX, dtype=float)
ramp = (x / (NX - 1.0))[:, None] * np.ones((1, NY))
yseed = np.sin(np.pi * x / (NX - 1.0))[:, None] * np.cos(TWO_PI * np.arange(NY)[None, :] / NY)

sig = {}
print()
print("[E1-1] 横截面分离：收敛解与张力闭合")
for K in (0.005, 0.24, 0.30):
    eta = 0.1 if K < 0.05 else 0.02
    phi, g, steps = relax2(ramp.copy(), K, eta, 20000)
    e = energy2(phi, K)
    s = e / NY
    sig[K] = (phi, s, steps)
    s1 = lk.sigma_1d(K)
    s1 = float(s1[0] if isinstance(s1, tuple) else s1)
    dy = np.roll(phi, -1, axis=1) - phi
    dev = 100.0 * (s - s1) / s1
    print("     K = %.3f：步数 %d，sigma = %.6f，一维精确 %.6f（差 %+.4f%%），max|d_y phi| = %.2e，max|g| = %.2e"
          % (K, steps, s, s1, dev, np.abs(dy).max(), np.abs(g[1:-1, :]).max()))
    check("E1-1 K=%.3f" % K, abs(dev) <= 0.05 and np.abs(dy).max() <= 1e-12,
          "偏差 %+.4f%%，d_y <= %.2e" % (dev, np.abs(dy).max()))

print()
print("[E1-2] 横截面稳定性：1e-3 的 y 调制种子")
for K in (0.005, 0.24, 0.30):
    phi0, s, _ = sig[K]
    eta = 0.02 if K < 0.05 else 0.005
    steps = 4000
    phib, _, _ = relax2(phi0 + 1e-3 * yseed, K, eta, steps)
    ratio = float(np.sum(yseed * (phib - phi0)) / np.sum(yseed ** 2)) / 1e-3
    print("     K = %.3f：种子 1e-3 经 %d 步后比值 %.3e（能量 %.6f）" % (K, steps, ratio, energy2(phib, K) / NY))
    check("E1-2 K=%.3f" % K, 0.0 <= ratio <= 0.5, "比值 %.3e" % ratio)

print()
print("[E1-3] 尺寸稳健性：Ny 加倍")
K = 0.24
phi0b = np.repeat(ramp[:, :1], 2 * NY, axis=1)
phib, _, _ = relax2(phi0b, K, 0.02, 20000)
sb = energy2(phib, K) / (2 * NY)
d3 = 100.0 * (sb - sig[K][1]) / sig[K][1]
print("     Ny = %d：sigma = %.6f（Ny = %d 给 %.6f，差 %+.4f%%）" % (2 * NY, sb, NY, sig[K][1], d3))
check("E1-3", abs(d3) <= 0.5, "差 %+.4f%%" % d3)

print()
print("[E1-4] 三维直积：NZ = 8 层的一致性与能量可加性")
NZ = 8
phi2 = sig[K][0]
base = np.repeat(phi2[:, :, None], NZ, axis=2)
ph3, _ = relax3(base, K, 0.02, 400, 200, np.zeros_like(base))
dz = np.roll(ph3, -1, axis=2) - ph3
d = ph3[1:, :, :] - ph3[:-1, :, :]
e3 = 0.5 * float(np.sum(d ** 2) + np.sum((np.roll(ph3, -1, axis=1) - ph3) ** 2) + np.sum(dz ** 2)) \
     + float(np.sum(K * (1.0 - np.cos(TWO_PI * ph3))))
e2 = energy2(phi2, K)
lay = float(np.abs(ph3 - phi2[:, :, None]).max())
dE = abs(e3 - NZ * e2) / (NZ * e2)
print("     逐层最大偏差 %.2e ｜ 总能量比 %.12f（差 %.2e）｜ z 向梯度能量 %.2e" % (lay, e3 / (NZ * e2), dE, 0.5 * float(np.sum(dz ** 2))))
check("E1-4", lay <= 1e-12 and dE <= 1e-12 and 0.5 * float(np.sum(dz ** 2)) <= 1e-20,
      "层偏差 %.2e，能量差 %.2e" % (lay, dE))

print()
print("[E1-5] 沿线稳定性：1e-3 的 z 调制种子（波长 4 与 8 层）")
ratio_z = {}
for lam in (4, 8):
    zz = np.arange(NZ, dtype=float)
    seed = 1e-3 * np.sin(np.pi * x / (NX - 1.0))[:, None, None] * np.sin(TWO_PI * zz[None, None, :] / lam)
    shp = seed / 1e-3
    _, amps = relax3(base + seed, K, 0.005, 400, 20, shp)
    ratio_z[lam] = amps[-1] / amps[0]
    print("     波长 %d 层（q_z^2 = %.4f）：投影 %.3e 到 %.3e，比值 %.3e"
          % (lam, TWO_PI ** 2 / lam ** 2, amps[0], amps[-1], amps[-1] / amps[0]))
check("E1-5", ratio_z[4] < ratio_z[8] < 1.0, "比值 4 层 %.3e，8 层 %.3e" % (ratio_z[4], ratio_z[8]))

print()
print("[图] 生成 E1 图")
fig, axs = plt.subplots(1, 2, figsize=(10.2, 3.8))
ax = axs[0]
for K in (0.005, 0.24, 0.30):
    ax.plot(x, sig[K][0][:, 0], lw=1.5, label="K = %.3f" % K)
ax.set_xlabel("x（Dirichlet strip）")
ax.set_ylabel("phi")
ax.set_title("(a) strip cross-section profile")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax = axs[1]
for lam, col in ((4, "tab:red"), (8, "tab:blue")):
    zz = np.arange(NZ, dtype=float)
    seed = 1e-3 * np.sin(np.pi * x / (NX - 1.0))[:, None, None] * np.sin(TWO_PI * zz[None, None, :] / lam)
    shp = seed / 1e-3
    _, amps = relax3(base + seed, K, 0.005, 400, 20, shp)
    ax.semilogy(np.arange(len(amps)) * 20, [max(abs(a), 1e-18) for a in amps], color=col, lw=1.5,
                label="wavelength %d layers" % lam)
ax.set_xlabel("relaxation step")
ax.set_ylabel("z-mode projection")
ax.set_title("(b) decay of along-line modulation")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(IMG, dpi=140)
print("     图已存 images/c1_e1_embedding.png")

print()
npass = sum(1 for _, ok in CHECKS if ok)
for tag, ok in CHECKS:
    print("     %-16s %s" % (tag, "通过" if ok else "失败"))
print("判据 %d/%d 通过；用时 %.1f 秒" % (npass, len(CHECKS), time.time() - T0))
