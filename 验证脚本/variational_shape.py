# -*- coding: utf-8 -*-
"""变分形状修正：径向化降阶、能量破缺 eps^4 与 l=4 形状扫描（第 465 条；第 464 条椭球 ansatz 的升级）。

模型升级：kink 剖面由椭球坐标函数改为"径向移形剖面"，phi = tanh(k (r - R(n)))，
R(n) 为形状的径向函数（方向 n 上的半径）。BPS 破缺由 |grad rho| 变成 |grad(r-R)|：
一阶偏差被形状梯度垂直于径向的性质抵消，逐点破缺降为 O(eps^2)（非负）：
  g - 1 ~ |grad_perp R|^2 / (2 R^2)。

判据：
  V1 径向化逐点破缺 ~ eps^2（比值 max(g-1)/eps^2 常数；对照椭球坐标 ansatz 的 O(eps)）
  V2 能量破缺 R_E = 0.5 <(g-1)^2> / <g> 的对数斜率外推至 eps -> 0 等于 4
  V3 形状变分：R = R_ell(1 + lambda * Y40(n)) 扫描 lambda，报告 R_E 最小值位置与相对椭球（lambda=0）的改善

确定性（Fib 球面采样），约数秒。图存 images/variational_shape.png
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
IMG = os.path.join(ROOT, "images", "variational_shape.png")


def check(tag, ok, detail):
    print(f"  [{tag}] {'PASS' if ok else 'FAIL'}  {detail}")
    return bool(ok)


# Fib 球面采样
NS = 60000
i = np.arange(NS)
zz = 1 - 2 * (i + 0.5) / NS
phg = np.pi * (1 + 5 ** 0.5) * i
rxy = np.sqrt(np.maximum(0.0, 1 - zz * zz))
N = np.stack([rxy * np.cos(phg), rxy * np.sin(phg), zz], axis=-1)


def R_base(n, eps, lam=0.0):
    a = 1.0 + eps
    b = 1.0 - eps / 2
    A = (n[..., 0] / a) ** 2 + (n[..., 1] / b) ** 2 + (n[..., 2] / b) ** 2
    R0 = 1.0 / np.sqrt(A)
    Y40 = (35 * n[..., 2] ** 4 - 30 * n[..., 2] ** 2 + 3) / 8.0
    return R0 * (1.0 + lam * Y40)


def grad_perp_mod(n, eps, lam):
    ref = np.where(np.abs(n[..., 2:3]) < 0.9, np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]))
    e1 = np.cross(n, ref)
    e1 = e1 / np.linalg.norm(e1, axis=-1, keepdims=True)
    e2 = np.cross(n, e1)
    d = 1e-5
    R0 = R_base(n, eps, lam)
    R1 = (R_base(n + d * e1, eps, lam) - R_base(n - d * e1, eps, lam)) / (2 * d)
    R2 = (R_base(n + d * e2, eps, lam) - R_base(n - d * e2, eps, lam)) / (2 * d)
    return R0, np.sqrt(R1 ** 2 + R2 ** 2)


def g_of(n, eps, lam=0.0):
    R0, gp = grad_perp_mod(n, eps, lam)
    return np.sqrt(1.0 + gp ** 2 / R0 ** 2)


print("== V1 径向化逐点破缺 ~ eps^2 ==")
eps_list = [0.01, 0.02, 0.04, 0.08]
dev = []
for eps in eps_list:
    g = g_of(N, eps)
    d = float(np.max(g - 1.0))
    dev.append(d)
    print(f"  eps = {eps:.2f}：max(g-1) = {d:.9e}，比值/eps^2 = {d/eps**2:.6f}")
ratio1 = [dev[k] / eps_list[k] ** 2 for k in range(len(eps_list))]
spread1 = (max(ratio1) - min(ratio1)) / max(ratio1)
print(f"  比值/eps^2 散布 = {spread1:.3e}（期望 < 5e-2，首阶二次）")
v1 = check("V1", spread1 < 5e-2, f"逐点破缺二次（比值 {ratio1[0]:.4f} 至 {ratio1[-1]:.4f}）")


def R_energy(eps, lam=0.0):
    g = g_of(N, eps, lam)
    return float(0.5 * np.mean((g - 1.0) ** 2) / np.mean(g))


print("== V2 能量破缺 ~ eps^4 ==")
Rl = []
for eps in eps_list:
    R = R_energy(eps)
    Rl.append(R)
    print(f"  eps = {eps:.2f}：R_E = {R:.9e}，比值/eps^4 = {R/eps**4:.6f}")
slopes = [math.log(Rl[k + 1] / Rl[k]) / math.log(eps_list[k + 1] / eps_list[k]) for k in range(len(Rl) - 1)]
print("  相邻档对数斜率 p = " + "、".join(f"{v:.6f}" for v in slopes))
mids = [(eps_list[k] + eps_list[k + 1]) / 2 for k in range(len(Rl) - 1)]
csl = (slopes[1] - slopes[0]) / (mids[1] - mids[0])
p0 = slopes[0] - csl * mids[0]
print(f"  线性外推 p(eps->0) = {p0:.6f}，与 4 的偏差 = {abs(p0-4.0):.2e}（期望 < 1e-3）")
v2 = check("V2", abs(p0 - 4.0) < 1e-3, f"首阶四次确认（外推 p = {p0:.6f}）")

print("== V3 形状变分（Y40 扫描） ==")
eps_v = 0.05
lam_list = np.linspace(-1.0, 1.0, 41) * eps_v
Rv = [R_energy(eps_v, lam) for lam in lam_list]
kmin = int(np.argmin(Rv))
print(f"  eps = {eps_v}：椭球（lambda=0）R_E = {R_energy(eps_v):.9e}")
print(f"  最优 lambda* = {lam_list[kmin]:.6f}（相对 eps 为 {lam_list[kmin]/eps_v:.4f}），R_E(min) = {Rv[kmin]:.9e}")
impr = (R_energy(eps_v) - Rv[kmin]) / R_energy(eps_v)
print(f"  相对椭球的改善 = {impr*100:.3f}%")
v3 = check("V3", True, f"变分扫描完成：最优 lambda* = {lam_list[kmin]:.5f}，改善 {impr*100:.2f}%")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
ax.loglog(eps_list, Rl, "o-", ms=4, label="径向化 R_E")
ax.loglog(eps_list, Rl[0] * (np.array(eps_list) / eps_list[0]) ** 4, "k--", lw=1, label="eps^4 参考")
ax.set_xlabel("各向异性度 eps"); ax.set_ylabel("能量破缺比")
ax.set_title("径向化后的能量破缺（四次标度）"); ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
ax = axes[1]
ax.plot(lam_list / eps_v, Rv, "o-", ms=3)
ax.axvline(0.0, color="k", ls="--", lw=1, label="椭球")
ax.set_xlabel("lambda / eps"); ax.set_ylabel("R_E")
ax.set_title("Y40 形状变分扫描（eps = 0.05）"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(IMG, dpi=160)
print("figure saved:", IMG)
print("判据：", "全部通过" if (v1 and v2 and v3) else "存在失败")
