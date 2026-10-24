# -*- coding: utf-8 -*-
r"""
c1_e2_timed_source.py

[C1 三维嵌入 第二判据 E2] 含时源分量的嵌入：运动壁的动量流、gamma 结构与沿线调制稳定性
（判据 E2-1 至 E2-5）

背景：E1 判据交付了静态源分量的嵌入（条带张力、直积稳定性与能量可加性）。本判据
把它推到含时分量：给壁一个助推，测动量流 T^{01} 的横截面积分、能量随速度的 gamma
结构，以及运动下的沿线调制稳定性。

模型：phi 为条带偏离场（u = IX + IY + phi），x 两端 Dirichlet（0 与 1），
E = sum_links (1/2) (d_x phi)^2 + sum_i [ (1/2) phidot_i^2 + K (1 - cos 2 pi phi_i) ]，
P = - sum_links (1/2)(phidot_i + phidot_{i+1}) d_i。

解析参考：连续 kink 为 f(s) = (2/pi) arctan(exp(A s))，A = 2 pi sqrt(K)；
助推解为 phi(x,t) = f(gamma (x - v t))、phidot = -gamma v f'，
其能量 E = gamma M、动量 P = gamma M v，且满足 P = E v（M 为静止能量）。

判据
E2-1 能量 gamma 结构：三档 v 的 E(v) / E(0) 对 gamma 的偏差 <= 1%
E2-2 动量流归一：P(v) / (gamma M v) 对 1 的偏差 <= 1%
E2-3 场论恒等式：P - E v 的相对偏差 <= 2%
E2-4 横截面嵌入：三维 (y,z) 积分值 = Ny * Nz * 一维值（<= 1e-12）
E2-5 运动下沿线稳定性：boost 构型加 1e-3 的 z 调制，leapfrog 演化后调制不增长
     （比值 <= 1.2），总能量漂移 <= 1e-4

口径说明：壁宽约 1/(2 pi sqrt(K)) 格。C1 驻留窗口 K >= 0.24 的壁宽约 0.3 格，
格点钉扎使运动分量不可测，故本判据在宽壁档（K = 0.0002 与 0.00005）执行；
窗口档只适用静态分量（E1）。

运行：python 验证脚本/c1_e2_timed_source.py（确定性，约 30 秒）
"""
import math
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

T0 = time.time()
TWO_PI = 2.0 * math.pi
IMG = "D:/software/TianXingFC/images/c1_e2_timed_source.png"
CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))


def kink(s, K):
    A = TWO_PI * math.sqrt(K)
    e = np.exp(np.clip(A * s, -300.0, 300.0))
    return (2.0 / math.pi) * np.arctan(e)


def kink_p(s, K):
    A = TWO_PI * math.sqrt(K)
    e = np.exp(np.clip(A * s, -300.0, 300.0))
    return (2.0 / math.pi) * A * e / (1.0 + e * e)


def energy1d(phi, phid, K):
    d = phi[1:] - phi[:-1]
    return float(np.sum(0.5 * d ** 2) + np.sum(0.5 * phid ** 2 + K * (1.0 - np.cos(TWO_PI * phi))))


def momentum1d(phi, phid):
    d = phi[1:] - phi[:-1]
    return float(-np.sum(0.5 * (phid[:-1] + phid[1:]) * d))


def boost_fields(x, x0, v, K):
    g = 1.0 / math.sqrt(1.0 - v * v)
    s = g * (x - x0)
    return kink(s, K), -g * v * kink_p(s, K)


print("=" * 78)
print("C1 三维嵌入 E2：含时源分量的嵌入（运动壁的动量流、gamma 与沿线稳定性）")
print("=" * 78)

N = 1024
x = np.arange(N, dtype=float)
X0 = 512.0
VS = (0.2, 0.4, 0.6)
rows = {}
for K in (0.0002, 0.00005):
    print()
    print("[K = %.5f，壁宽约 %.1f 格] 助推壁的能量与动量" % (K, 1.0 / (TWO_PI * math.sqrt(K))))
    phi0, _ = boost_fields(x, X0, 0.0, K)
    M = energy1d(phi0, np.zeros(N), K)
    for v in VS:
        phi, phid = boost_fields(x, X0, v, K)
        g = 1.0 / math.sqrt(1.0 - v * v)
        E, P = energy1d(phi, phid, K), momentum1d(phi, phid)
        rows[(K, v)] = (E, P, M, g)
        print("     v = %.1f：E/M = %.6f（gamma = %.6f，差 %+.3f%%）｜ P/(gamma M v) = %.6f（差 %+.3f%%）｜ (P - E v)/P = %.2e"
              % (v, E / M, g, 100.0 * (E / M - g) / g, P / (g * M * v), 100.0 * (P / (g * M * v) - 1.0), (P - E * v) / abs(P)))
    for v in VS:
        E, P, M, g = rows[(K, v)]
        check("E2-1 K=%.5f v=%.1f" % (K, v), abs(E / M - g) / g <= 0.01, "E/M 对 gamma 差 %+.3f%%" % (100.0 * (E / M - g) / g))
        check("E2-2 K=%.5f v=%.1f" % (K, v), abs(P / (g * M * v) - 1.0) <= 0.01, "P/(gamma M v) 差 %+.3f%%" % (100.0 * (P / (g * M * v) - 1.0)))
        check("E2-3 K=%.5f v=%.1f" % (K, v), abs(P - E * v) / abs(P) <= 0.02, "(P - E v)/P = %.2e" % ((P - E * v) / abs(P)))

print()
print("[E2-4] 横截面嵌入：三维 (y,z) 积分值与一维值的对照")
K, NY, NZ = 0.0002, 8, 8
v = 0.4
phi1, phid1 = boost_fields(x, X0, v, K)
phi3 = np.repeat(np.repeat(phi1[:, None, None], NY, axis=1), NZ, axis=2)
phid3 = np.repeat(np.repeat(phid1[:, None, None], NY, axis=1), NZ, axis=2)
d3 = phi3[1:, :, :] - phi3[:-1, :, :]
e3 = float(np.sum(0.5 * d3 ** 2) + NY * NZ * np.sum(0.5 * phid1 ** 2 + K * (1.0 - np.cos(TWO_PI * phi1))))
p3 = float(-np.sum(0.5 * (phid3[:-1, :, :] + phid3[1:, :, :]) * d3))
E1, P1 = energy1d(phi1, phid1, K), momentum1d(phi1, phid1)
de = abs(e3 - NY * NZ * E1) / abs(NY * NZ * E1)
dp = abs(p3 - NY * NZ * P1) / abs(NY * NZ * P1)
print("     能量：三维 %.8f 对 %d x 一维 %.8f（差 %.2e）｜ 动量：三维 %.8f 对 %d x 一维 %.8f（差 %.2e）"
      % (e3, NY * NZ, E1, de, p3, NY * NZ, P1, dp))
check("E2-4", de <= 1e-12 and dp <= 1e-12, "能量差 %.2e，动量差 %.2e" % (de, dp))

print()
print("[E2-5] 运动下沿线稳定性：boost 构型加 1e-3 的 z 调制")
K = 0.00005
N3, NY3, NZ3 = 512, 4, 8
x3 = np.arange(N3, dtype=float)
X3 = 256.0
v = 0.3
phi2, vel2 = boost_fields(x3, X3, v, K)
base = np.repeat(np.repeat(phi2[:, None, None], NY3, axis=1), NZ3, axis=2)
vel = np.repeat(np.repeat(vel2[:, None, None], NY3, axis=1), NZ3, axis=2)
shape = kink_p(x3 - X3, K)[:, None, None] * np.sin(TWO_PI * np.arange(NZ3)[None, None, :] / NZ3)
phi = base + 1e-3 * shape
phid = vel.copy()


def force(p):
    lap = np.zeros_like(p)
    lap[1:-1] = p[2:] - 2.0 * p[1:-1] + p[:-2]
    lap[1:-1] += (np.roll(p, -1, axis=1) - 2.0 * p + np.roll(p, 1, axis=1))[1:-1]
    lap[1:-1] += (np.roll(p, -1, axis=2) - 2.0 * p + np.roll(p, 1, axis=2))[1:-1]
    return lap - TWO_PI * K * np.sin(TWO_PI * p)


def energy3(p, q):
    d = p[1:, :, :] - p[:-1, :, :]
    dy = np.roll(p, -1, axis=1) - p
    dz = np.roll(p, -1, axis=2) - p
    return float(np.sum(0.5 * d ** 2) + np.sum(0.5 * q ** 2 + K * (1.0 - np.cos(TWO_PI * p))) + 0.5 * float(np.sum(dy ** 2)) + 0.5 * float(np.sum(dz ** 2)))


shp = np.broadcast_to(shape, phi.shape)
E0 = energy3(phi, phid)
proj0 = float(np.sum(shp * (phi - base)) / np.sum(shp ** 2))
dt = 0.05
hist = [(0.0, proj0, E0)]
for it in range(1, 1601):
    phid = phid + 0.5 * dt * force(phi)
    phi = phi + dt * phid
    phid = phid + 0.5 * dt * force(phi)
    phi[0, :, :] = 0.0
    phi[-1, :, :] = 1.0
    phid[0, :, :] = 0.0
    phid[-1, :, :] = 0.0
    if it % 200 == 0:
        hist.append((it * dt, float(np.sum(shp * (phi - base)) / np.sum(shp ** 2)), energy3(phi, phid)))
proj1, E1t = hist[-1][1], hist[-1][2]
ratio = proj1 / proj0
drift = abs(E1t - E0) / abs(E0)
print("     调制投影 %.3e 到 %.3e（比值 %.3f）｜ 总能量漂移 %.2e" % (proj0, proj1, ratio, drift))
check("E2-5", ratio <= 1.2 and drift <= 1e-4, "比值 %.3f，漂移 %.2e" % (ratio, drift))

print()
print("[图] 生成 E2 图")
fig, axs = plt.subplots(1, 2, figsize=(10.2, 3.8))
ax = axs[0]
for K, col in ((0.0002, "tab:blue"), (0.00005, "tab:green")):
    vv = np.array(VS)
    ax.plot(vv, [(rows[(K, v)][0] / rows[(K, v)][2]) for v in VS], "o-", color=col, lw=1.5,
            label="E/M, K = %.5f" % K)
    ax.plot(vv, [1.0 / math.sqrt(1.0 - v * v) for v in VS], "k--", lw=1.0,
            label="gamma" if K == 0.0002 else None)
ax.set_xlabel("v")
ax.set_ylabel("E / M")
ax.set_title("(a) boosted wall: E/M vs gamma")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax = axs[1]
ax.semilogy([h[0] for h in hist], [max(abs(h[1]), 1e-18) for h in hist], "o-", color="tab:red", lw=1.5)
ax.set_xlabel("time")
ax.set_ylabel("z-mode projection")
ax.set_title("(b) along-line modulation under motion")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(IMG, dpi=140)
print("     图已存 images/c1_e2_timed_source.png")

print()
npass = sum(1 for _, ok in CHECKS if ok)
for tag, ok in CHECKS:
    print("     %-26s %s" % (tag, "通过" if ok else "失败"))
print("判据 %d/%d 通过；用时 %.1f 秒" % (npass, len(CHECKS), time.time() - T0))
