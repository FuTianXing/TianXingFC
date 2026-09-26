# -*- coding: utf-8 -*-
"""非球对称自撑的第一轮：椭球 kink 的逐点破缺与能量破缺标度（第 464 条；判据 S 的非球对称未验项）。

模型：椭球 hedgehog，phi = tanh(rho)，rho = sqrt(kx^2 x^2 + ky^2 y^2 + kz^2 z^2)，v = 1；
势 V = (k^2/2)(phi^2 - 1)^2（k 为参考平均）；BPS 条件要求 |grad rho| = k 逐点成立。

只依赖方向的量：
  h(n) = sqrt(kx^2 nx^2 + ky^2 ny^2 + kz^2 nz^2)（rho = r h）
  g(n) = |grad rho|(n) = sqrt((kx^4 nx^2 + ky^4 ny^2 + kz^4 nz^2)/(kx^2 nx^2 + ky^2 ny^2 + kz^2 nz^2))

判据：
  NS1 逐点 BPS 破缺：椭球梯度模的方向涨落 max|g/k - 1| 与各向异性度 eps 成正比（线性标度）；
  NS2 能量破缺：(E - E_BPS)/E_BPS = [0.5 int dOmega (g-k)^2/h^3] / [k int dOmega g/h^3]
      与 eps^2 成正比（二次标度）；
  NS3 整体缩放不变：所有 k_i 同乘 lambda 时逐点破缺与能量比值不变（自撑的比例结构不依赖整体标度）。
各向异性参数化：kx = k(1+eps)，ky = kz = k(1-eps/2)（一阶体积保持）。

确定性，约 1 秒。图存 images/ellipsoid_selfsupport.png
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
IMG = os.path.join(ROOT, "images", "ellipsoid_selfsupport.png")


def check(tag, ok, detail):
    print(f"  [{tag}] {'PASS' if ok else 'FAIL'}  {detail}")
    return bool(ok)


# 方向网格（theta, phi）
NTH, NPH = 240, 480
th = np.linspace(0, np.pi, NTH + 1)
ph = np.linspace(0, 2 * np.pi, NPH + 1)
TH, PH = np.meshgrid(th[:-1], ph[:-1], indexing="ij")
NX = np.sin(TH) * np.cos(PH)
NY = np.sin(TH) * np.sin(PH)
NZ = np.cos(TH)
N = np.stack([NX, NY, NZ], axis=-1)
dO = np.sin(TH) * (np.pi / NTH) * (2 * np.pi / NPH)


def gfun(kx, ky, kz, n):
    num = kx ** 4 * n[..., 0] ** 2 + ky ** 4 * n[..., 1] ** 2 + kz ** 4 * n[..., 2] ** 2
    den = kx ** 2 * n[..., 0] ** 2 + ky ** 2 * n[..., 1] ** 2 + kz ** 2 * n[..., 2] ** 2
    return np.sqrt(num / den)


def hfun(kx, ky, kz, n):
    return np.sqrt(kx ** 2 * n[..., 0] ** 2 + ky ** 2 * n[..., 1] ** 2 + kz ** 2 * n[..., 2] ** 2)


k0 = 1.0


def build(eps, lam=1.0):
    kx = lam * k0 * (1 + eps)
    ky = lam * k0 * (1 - eps / 2)
    kz = ky
    return kx, ky, kz


def R_energy(eps, lam=1.0):
    kx, ky, kz = build(eps, lam)
    g = gfun(kx, ky, kz, N)
    h = hfun(kx, ky, kz, N)
    num = 0.5 * np.sum((g - k0 * lam) ** 2 / h ** 3 * dO)
    den = (k0 * lam) * np.sum(g / h ** 3 * dO)
    return float(num / den)


print("== NS1 逐点 BPS 破缺（椭球梯度模的方向涨落） ==")
eps_list = [0.01, 0.02, 0.04, 0.08]
dev = []
for eps in eps_list:
    kx, ky, kz = build(eps)
    gg = gfun(kx, ky, kz, N) / k0
    d = float(np.max(np.abs(gg - 1.0)))
    dev.append(d)
    print(f"  eps = {eps:.2f}：max|g/k - 1| = {d:.9f}，比值/eps = {d/eps:.9f}")
dev_ratio = [dev[i] / eps_list[i] for i in range(len(eps_list))]
spread1 = (max(dev_ratio) - min(dev_ratio)) / max(dev_ratio)
print(f"  比值/eps 的散布 = {spread1:.3e}（期望 < 1e-9，线性标度）")
ns1 = check("NS1", spread1 < 1e-9, f"逐点破缺与 eps 线性（比值 {dev_ratio[0]:.6f} 至 {dev_ratio[-1]:.6f}）")

print("== NS2 能量破缺的二次标度（对数斜率趋于 2） ==")
Rl = []
for eps in eps_list:
    R = R_energy(eps)
    Rl.append(R)
    print(f"  eps = {eps:.2f}：(E-E_BPS)/E_BPS = {R:.9e}，比值/eps^2 = {R/eps**2:.9f}")
slopes = [math.log(Rl[i + 1] / Rl[i]) / math.log(eps_list[i + 1] / eps_list[i]) for i in range(len(Rl) - 1)]
print("  相邻档对数斜率 p = " + "、".join(f"{v:.6f}" for v in slopes))
mids = [(eps_list[i] + eps_list[i + 1]) / 2 for i in range(len(Rl) - 1)]
cslope = (slopes[1] - slopes[0]) / (mids[1] - mids[0])
p0 = slopes[0] - cslope * mids[0]
print(f"  斜率随 eps 线性外推 p(eps->0) = {p0:.6f}，与 2 的偏差 = {abs(p0-2.0):.2e}（期望 < 5e-4）")
ns2 = check("NS2", abs(p0 - 2.0) < 5e-4, f"首阶二次确认（外推 p = {p0:.6f}；高阶趋势已登记）")

print("== NS3 整体缩放不变 ==")
base = R_energy(0.2, lam=1.0)
scale_dev = []
for lam in [2.0, 5.0]:
    scale_dev.append(abs(R_energy(0.2, lam=lam) - base) / base)
print(f"  lam = 2、5 时能量比值相对偏差 = {max(scale_dev):.3e}（期望 < 1e-12）")
ns3 = check("NS3", max(scale_dev) < 1e-12, "整体缩放不改变破缺结构")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
phi_line = np.linspace(0, np.pi / 2, 200)
nline = np.stack([np.cos(phi_line), np.sin(phi_line), np.zeros_like(phi_line)], axis=-1)
for eps in [0.1, 0.2, 0.4]:
    kx, ky, kz = build(eps)
    ax.plot(phi_line, gfun(kx, ky, kz, nline) / k0 - 1.0, label=f"eps = {eps}")
ax.axhline(0.0, color="k", ls="--", lw=1)
ax.set_xlabel("方向角（x 轴到 y 轴）"); ax.set_ylabel("g/k - 1")
ax.set_title("逐点 BPS 破缺的方向分布"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax = axes[1]
ax.loglog(eps_list, Rl, "o-", ms=4, label="(E-E_BPS)/E_BPS")
ref = Rl[0] * (np.array(eps_list) / eps_list[0]) ** 2
ax.loglog(eps_list, ref, "k--", lw=1, label="eps^2 参考")
ax.set_xlabel("各向异性度 eps"); ax.set_ylabel("能量破缺比")
ax.set_title("能量破缺的二次标度"); ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(IMG, dpi=160)
print("figure saved:", IMG)
print("判据：", "全部通过" if (ns1 and ns2 and ns3) else "存在失败")
