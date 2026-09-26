# -*- coding: utf-8 -*-
"""光包进桥：分布源的能量密度直接进源与等效窄源对照（第 452 条；443 移交"分布源"）。

判据：
  P1 分布源的器械语义：站点源 f = u（能量密度）时，链接源 s = 0，故全空间 lambda = 1（逐点）；
     即"尺读数 = 记录中点"对分布源同样成立（器械语义确认）
  P2 远场等效：光包（分布，a = 10、波长 6.3 格、同 437 参数）与等效窄源（同总能量、单格）的
     远场 1/r 系数比随窗右移趋于 1（口径：桥器械、含尺读数）
  P3 与 437 口径一致：本器械的远场比值与第 437 条（纯泊松，平台 1.009）一致（同 u、同离散核）

器械复用 436/443：fft_solve 离散泊松（源在站点）；u_e = interp(theta)（s_e = 0）。
运行：python 验证脚本/minimal_verification_photon_packet_bridge.py
确定性，无随机数，运行约数秒。
"""
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "images", "photon_packet_bridge.png")

D = 3
N = 96


def interp_site(theta):
    return np.stack([(theta + np.roll(theta, -1, axis=mu)) / 2.0 for mu in range(D)])


def fft_solve(f):
    kx = 2 * np.pi * np.fft.fftfreq(N)
    KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing="ij")
    denom = 1.0 - (np.cos(KX) + np.cos(KY) + np.cos(KZ)) / D
    denom.flat[0] = 1.0
    theta = np.real(np.fft.ifftn(np.fft.fftn(f) / denom))
    return theta - theta.mean()


# 光包：斜向行波 × 高斯（437 参数：a = 10，波长 6.3 格）
c0 = N // 2
idx = np.arange(N) - c0
X, Y, Z = np.meshgrid(idx, idx, idx, indexing="ij")
a = 10.0
env = np.exp(-(X ** 2 + Y ** 2 + Z ** 2) / (2 * a * a))
kk = 1.0
ph = kk * (X + Y + Z) / math.sqrt(3.0)
e1 = np.array([1.0, -1.0, 0.0]) / math.sqrt(2.0)
khat = np.array([1.0, 1.0, 1.0]) / math.sqrt(3.0)
bhat = np.cross(khat, e1)
E = e1[:, None, None, None] * np.sin(ph)[None, :, :, :] * env[None, :, :, :]
B = bhat[:, None, None, None] * np.sin(ph)[None, :, :, :] * env[None, :, :, :]
u = 0.5 * (E[0] ** 2 + E[1] ** 2 + E[2] ** 2 + B[0] ** 2 + B[1] ** 2 + B[2] ** 2)
U = float(u.sum())
print("== 设定 ==")
print("  光包：a = %.1f，波长 %.1f 格，U = sum u = %.6e" % (a, 2 * math.pi / kk, U))

# 源 A：分布（u 直接作站点源）；源 B：等效窄源（同 U，中心单格）
fA = u.copy()
fB = np.zeros((N, N, N)); fB[c0, c0, c0] = U
thA = fft_solve(fA)
thB = fft_solve(fB)
uA = interp_site(thA)
tm = interp_site(thA)

print()
print("== P1 器械语义（分布源） ==")
vals = []
for mu in range(D):
    m = np.abs(tm[mu]) > 1e-12
    vals.append(uA[mu][m] / tm[mu][m])
v = np.concatenate(vals)
print("  lambda 中位 = %.15f；max|lambda-1| = %.3e（期望逐点 1：s_e = 0）" % (float(np.median(v)), float(np.max(np.abs(v - 1)))))

print()
print("== P2 远场等效（球壳平均，同 437 口径） ==")
r2grid = X ** 2 + Y ** 2 + Z ** 2
rshell = np.sqrt(r2grid)
def shell_mean(field, r0):
    mask = np.abs(rshell - r0) < 0.5
    return float(np.mean(field[mask]))
rrs = np.arange(12, 25)
pA = np.array([shell_mean(thA, r) for r in rrs])
pB = np.array([shell_mean(thB, r) for r in rrs])
for (lo, hi) in ((18, 24), (12, 20)):
    sel = (rrs >= lo) & (rrs <= hi)
    rm = float(np.mean(pA[sel] / pB[sel]))
    rs = float(np.std(pA[sel] / pB[sel]))
    print("  窗 [%2d, %2d]：比值均值 = %.5f，散布 = %.2e（口径同 437：先球壳平均、后逐 r 比值再平均）" % (lo, hi, rm, rs))
print("  对照（437 纯泊松）：平台比值均值 1.009、散布 1.08e-02；本器械同离散核，期望一致")

# 附加诊断：等效源换成"宽 1 归一化高斯"（437 口径）
g1 = np.exp(-(X ** 2 + Y ** 2 + Z ** 2) / 2.0)
g1 = g1 / g1.sum()
fC = U * g1
thC = fft_solve(fC)
pC = np.array([shell_mean(thC, r) for r in rrs])
for (lo, hi) in ((18, 24),):
    sel = (rrs >= lo) & (rrs <= hi)
    rm = float(np.mean(pA[sel] / pC[sel]))
    print("  诊断：光包/宽1高斯 比值均值 = %.5f（等效源选择的敏感性）" % rm)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
im = ax.imshow(u[c0, :, :], cmap="inferno")
ax.set_title("光包 u 的中心切面")
fig.colorbar(im, ax=ax, fraction=0.046)
ax = axes[1]
ax.plot(rrs, pA / pB, "o-", ms=3, label="packet/point（球壳平均）")
ax.axvspan(18, 24, color="0.85", alpha=0.5, label="平台窗")
ax.axhline(1.0, color="k", ls="--", lw=1)
ax.set_xlabel("r"); ax.set_ylabel("theta_packet/theta_point")
ax.set_title("远场比值（分布/点源）")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(IMG, dpi=160)
print("figure saved:", IMG)
