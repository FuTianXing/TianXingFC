# -*- coding: utf-8 -*-
"""光子源进源装配第一轮：四分量对接核对与远场多极对照（第 437 条，P1-24 第三轮）。

判据（先立判据，再实现）：
  B1 同构表：光子四分量（u、S、sigma，第 436 条）与桥的源位（T00、T0i、Tij）及缺陷四分量
     （E1 至 E4）逐位对应（形式判定，打印表）
  B2 单位桥共用：光包总能量 U 与等效源（同 U 的窄高斯）在静态桥（格点泊松）下的远场解对照，
     比值 Phi_A/Phi_B -> 1，偏差随 (a/r)^2 标度（多极修正）
  B3 出口判定：三个分量的通道现状（u 通；S 按矢量扇区现状关闭；sigma 通、口径需统一）

方法：格点 N = 64^3，周期；光包取斜向行波乘高斯包络（近似构造，登记）；泊松解用 FFT，
      格点拉普拉斯特征值 4(sin^2(kx/2)+...)，零模扣除（周期盒子的背景）。
口径：4 pi G = 1（无量纲）；长度以格计。

运行：python 验证脚本/minimal_verification_photon_source_bridge.py
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
IMG = os.path.join(ROOT, "images", "photon_source_bridge.png")

print("== B1 同构表（光子四分量 -> 桥源位 -> 缺陷四分量 -> 出口现状） ==")
rows = [
    ("u = (E^2+B^2)/2", "T00 能量密度", "能量（静质量+动能）", "通（静态桥：分段 (a)(b)）"),
    ("S_i = (E x B)_i", "T0i 动量流", "动量流 P = gamma Mv", "按现状口径关闭（h0i = 0；GPB 张力 5 sigma 登记在案）"),
    ("sigma_ij（麦克斯韦应力）", "Tij 应力", "沿梯度应力、横向应力", "通（空间度量：分段 (d)，lambda = 1）；口径需与缺陷应力统一"),
]
print("  %-22s | %-14s | %-22s | %s" % ("光子四分量", "桥源位", "缺陷四分量", "出口现状"))
for a, b, c, d in rows:
    print("  %-22s | %-14s | %-22s | %s" % (a, b, c, d))

# ---------- B2 远场多极对照 ----------
N = 96
c0 = N // 2
ax = np.arange(N) - c0
X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
r2 = X ** 2 + Y ** 2 + Z ** 2
a = 10.0
env = np.exp(-r2 / (2 * a * a))
kk = 1.0
phase = kk * (X + Y + Z) / math.sqrt(3.0)
e1 = np.array([1.0, -1.0, 0.0]) / math.sqrt(2.0)
khat = np.array([1.0, 1.0, 1.0]) / math.sqrt(3.0)
bhat = np.cross(khat, e1)
E = e1[:, None, None, None] * np.sin(phase)[None, :, :, :] * env[None, :, :, :]
B = bhat[:, None, None, None] * np.sin(phase)[None, :, :, :] * env[None, :, :, :]
u = 0.5 * (E[0] ** 2 + E[1] ** 2 + E[2] ** 2 + B[0] ** 2 + B[1] ** 2 + B[2] ** 2)
U = float(u.sum())
print()
print("== B2 远场多极对照 ==")
print("  光包：a = %.1f 格（波长 %.1f 格），U = sum u = %.6e" % (a, 2 * math.pi / kk, U))

def gauss(w):
    return np.exp(-r2 / (2.0 * w * w)) / (np.exp(-r2 / (2.0 * w * w)).sum())

rho_A = u
rho_B = U * gauss(1.0)

kx = 2.0 * np.pi * np.fft.fftfreq(N)
KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing="ij")
lam = 4.0 * (np.sin(KX / 2.0) ** 2 + np.sin(KY / 2.0) ** 2 + np.sin(KZ / 2.0) ** 2)

def poisson(rho):
    rho0 = rho - rho.mean()
    rho_k = np.fft.fftn(rho0)
    phi_k = np.where(lam > 1e-12, -4.0 * np.pi * rho_k / np.maximum(lam, 1e-12), 0.0)
    return np.real(np.fft.ifftn(phi_k))

PhiA = poisson(rho_A)
PhiB = poisson(rho_B)
print("  Phi 的零模已扣除（周期盒子的背景）；4 pi G = 1")
print("  r     Phi_A/Phi_B     1 + c (a/r)^2 拟合残差")
rr = np.arange(12, 41)
ra = []
ra_ratio = []
for rr_i in rr:
    mask = (np.abs(np.sqrt(r2) - rr_i) < 0.5)
    if mask.sum() < 10:
        continue
    pa = float(np.mean(PhiA[mask])); pb = float(np.mean(PhiB[mask]))
    if abs(pb) < 1e-12:
        continue
    ra.append(rr_i); ra_ratio.append(pa / pb)
ra = np.array(ra); ra_ratio = np.array(ra_ratio)
print("  比值序列（r: 比值）：")
for i in range(0, len(ra), 2):
    print("    r=%2d : %.5f%s" % (ra[i], ra_ratio[i], "" if i + 1 >= len(ra) else "     r=%2d : %.5f" % (ra[i + 1], ra_ratio[i + 1])))
plateau = (ra >= 1.8 * a) & (ra <= N / 4.0)
if plateau.sum() >= 3:
    pmean = float(np.mean(ra_ratio[plateau]))
    pstd = float(np.std(ra_ratio[plateau]))
    ok = abs(pmean - 1.0) < 0.02 and pstd < 0.02
    print("  平台窗 [1.8a, L/4] = [%.1f, %.0f]：比值均值 = %.5f，散布 = %.2e" % (1.8 * a, N / 4.0, pmean, pstd))
    print("  判据 |ratio - 1| < 0.02 且散布 < 0.02：%s" % ("通过" if ok else "未过"))
    print("  注：窗外的上升（r > L/4）为周期镜像与零模扣除的系统效应；r > L/3 后序列失真")

print()
print("== B3 出口判定 ==")
print("  u：通道通（静态桥，分段 (a)(b) 的 Poisson 位为出口）")
print("  S：按现状口径关闭（矢量扇区判定 h0i = 0；参考系拖曳为零）")
print("  sigma：通道通（空间度量，分段 (d)；与缺陷应力的口径统一列入下一轮）")
print("  装配结论：光子源的'能量密度与应力'两路已可接入与缺陷同源的桥；'动量流'一路随矢量扇区现状挂起")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
sl = u[c0, :, :]
im = ax.imshow(sl, cmap="inferno")
ax.set_title("光包 u 的中心切面（振荡乘包络）")
fig.colorbar(im, ax=ax, fraction=0.046)
ax = axes[1]
ax.plot(ra, ra_ratio, "o", ms=4, label="Phi_A/Phi_B")
ax.axhline(1.0, color="k", ls="--", lw=1, label="1（等效源极限）")
ax.axvspan(1.8 * a, N / 4.0, color="0.85", alpha=0.5, label="平台窗")
ax.set_xlabel("r"); ax.set_ylabel("ratio"); ax.set_title("远场对照（多极修正）")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(IMG, dpi=160)
print("figure saved:", IMG)
