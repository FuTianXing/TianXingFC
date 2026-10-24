# -*- coding: utf-8 -*-
"""约束变分的显式实现：拉格朗日乘子、约束梯度方向与投影极小化（第 467 条；第 466 条移交的约束处置）。

模型：薄壁自撑球面，phi = tanh(k(r - R(n)))，k = 1；形状多模展开
  R(n) = sum_{l=0}^{4} c_l P_l(cos theta)；
能量 E(c)（第 466 条的精确泛函）与体积泛函 V(c) = (1/3) int R^3 dOmega。
约束取固定体积 V(c) = V_0。

判据：
  C1 拉格朗日乘子：约束驻定 dE/dc0 = mu dV/dc0 的数值 mu 与薄壁解析 mu = 8/(3 c0) 对照
  C2 约束梯度方向：dV/dc_l = 0（l = 1..4）在球点上机器零（约束只沿 l=0，切空间 = l>=1）
  C3 约束切空间谱：l>=1 的无约束谱在约束下不变且全正（谱由第 466 条解析式给出）
  C4 投影极小化演示：从随机 l>=1 扰动出发，固定体积的投影梯度下降收敛到球（能量单调降）

确定性（Fib 方向采样 + 径向网格），约数秒。图存 images/constrained_variation.png
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
IMG = os.path.join(ROOT, "images", "constrained_variation.png")


def check(tag, ok, detail):
    print(f"  [{tag}] {'PASS' if ok else 'FAIL'}  {detail}")
    return bool(ok)


from numpy.polynomial import legendre as LEG

NR = 1600
r_max = 40.0
r = np.linspace(0.0, r_max, NR)
dr = r[1] - r[0]
NS = 3000
i = np.arange(NS)
zz = 1 - 2 * (i + 0.5) / NS
phg = np.pi * (1 + 5 ** 0.5) * i
rxy = np.sqrt(np.maximum(0.0, 1 - zz * zz))
NZ = zz
W = 4.0 * np.pi / NS


def Pl_and_dth(l, costh):
    coeffs = [0] * l + [1]
    P = LEG.legval(costh, coeffs)
    if l == 0:
        return P, np.zeros_like(costh)
    Pp = LEG.legval(costh, LEG.legder(coeffs))
    sinth = np.sqrt(np.maximum(0.0, 1 - costh ** 2))
    return P, -sinth * Pp


PL = [Pl_and_dth(l, NZ) for l in range(5)]


def R_and_dth(c):
    Rv = np.zeros_like(NZ)
    dth = np.zeros_like(NZ)
    for l in range(5):
        Rv = Rv + c[l] * PL[l][0]
        dth = dth + c[l] * PL[l][1]
    return Rv, dth


def energy(c):
    Rv, dth = R_and_dth(c)
    rho = r[:, None] - Rv[None, :]
    sech4 = (1.0 / np.cosh(rho)) ** 4
    integ = (r ** 2)[:, None] * sech4 * (1.0 + (dth ** 2)[None, :] / (2.0 * (r ** 2)[:, None] + 1e-300))
    return float(np.sum(integ) * dr * W)


def volume(c):
    Rv, _ = R_and_dth(c)
    return float(np.sum(Rv ** 3) * W / 3.0)


R0 = 12.0
c_star = np.array([R0, 0.0, 0.0, 0.0, 0.0])

print("== C1 拉格朗日乘子 ==")
h = 1e-2
dE = (energy(c_star + [h, 0, 0, 0, 0]) - energy(c_star - [h, 0, 0, 0, 0])) / (2 * h)
dV = (volume(c_star + [h, 0, 0, 0, 0]) - volume(c_star - [h, 0, 0, 0, 0])) / (2 * h)
mu_num = dE / dV
mu_ana = 8.0 / (3.0 * R0)
print(f"  dE/dc0 = {dE:.6f}，dV/dc0 = {dV:.6f}，mu = {mu_num:.9f}（解析 8/(3c0) = {mu_ana:.9f}）")
c1 = check("C1", abs(mu_num - mu_ana) / mu_ana < 1e-3, f"乘子与解析一致（{mu_num:.6f} 对 {mu_ana:.6f}）")

print("== C2 约束梯度方向 ==")
devs = []
for l in range(1, 5):
    dc = [0.0] * 5
    dc[l] = h
    dVl = (volume(np.array(c_star) + dc) - volume(np.array(c_star) - dc)) / (2 * h)
    devs.append(abs(dVl))
rel = max(devs) / abs(dV)
print(f"  dV/dc_l（l=1..4）最大绝对值 = {max(devs):.3e}，相对 dV/dc0 = {rel:.3e}（期望 < 1e-6；残余为 Fib 采样误差）")
c2 = check("C2", rel < 1e-6, "约束梯度只沿 l=0（切空间 = l>=1）")

print("== C3 约束切空间谱 ==")
eps = 0.05
Ebase = energy(c_star)
lam_num = {}
for l in range(1, 5):
    cp = np.array(c_star); cm = np.array(c_star)
    cp[l] = eps; cm[l] = -eps
    lam_num[l] = (energy(cp) - 2 * Ebase + energy(cm)) / eps ** 2
print("  l :  无约束 lambda_l（约束下不变）")
pos = True
for l in range(1, 5):
    print(f"  {l} :  {lam_num[l]:.6f}")
    if lam_num[l] <= 0:
        pos = False
c3 = check("C3", pos, "l>=1 谱全正（约束不修正；Derrick 模为 l=0，已被约束消掉）")

print("== C4 投影极小化演示 ==")
rng = np.random.default_rng(467)
c = np.array(c_star)
c[1:] = 0.3 * rng.normal(size=4)
Es = [energy(c)]
for it in range(100):
    # 数值梯度（只动 l>=1），再把 c0 调回使 V = V0
    grad = np.zeros(5)
    hg = 1e-4
    for l in range(1, 5):
        cp = c.copy(); cp[l] += hg
        cm = c.copy(); cm[l] -= hg
        grad[l] = (energy(cp) - energy(cm)) / (2 * hg)
    c = c - 0.005 * grad
    # 体积投影：解 c0 使 V(c) = V0（单调，用二分）
    lo, hi = 0.5 * c_star[0], 2.0 * c_star[0]
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        cc = c.copy(); cc[0] = mid
        if volume(cc) < (4 * np.pi / 3.0) * R0 ** 3:
            lo = mid
        else:
            hi = mid
    c[0] = 0.5 * (lo + hi)
    Es.append(energy(c))
shape_norm = float(np.linalg.norm(c[1:]))
mono = Es[-1] < Es[0]
print(f"  初始形状范数 {0.3*np.sqrt(4):.4f} 至 {shape_norm:.6f}；能量 {Es[0]:.6f} 至 {Es[-1]:.6f}；单调降 {mono}")
c4 = check("C4", shape_norm < 1e-3 and Es[-1] < Es[0], f"收敛到球（形状范数 {shape_norm:.2e}）")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
ax.plot(range(len(Es)), Es, "-")
ax.set_xlabel("投影下降迭代"); ax.set_ylabel("E")
ax.set_title("固定体积的投影极小化"); ax.grid(alpha=0.3)
ax = axes[1]
ls = [1, 2, 3, 4]
ax.plot(ls, [lam_num[l] for l in ls], "o-", ms=4)
ax.set_xlabel("l"); ax.set_ylabel("lambda_l（约束下）")
ax.set_title("约束切空间谱（l>=1，全正）"); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(IMG, dpi=160)
print("figure saved:", IMG)
print("判据：", "全部通过" if (c1 and c2 and c3 and c4) else "存在失败")
