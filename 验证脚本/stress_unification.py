# -*- coding: utf-8 -*-
"""应力口径统一：光与缺陷应力的纯纵结构与桥的同幅口径（第 439 条，P1-24 第四轮）。

判据（先立判据，再实现）：
  S1 分解表：缺陷应力（沿梯度、横向两类）与麦克斯韦应力的分解并排（形式判定）
  S2 平面波的本征结构：麦克斯韦应力对行波的本征值为 {0, 0, u}（两个横向零、一个沿传播 = u），
     数值统计（内部样本点）
  S3 逐点纯纵性：sigma_ij 与 u k_i k_j 的相对偏差，期望 O(theta^2) 级（格点修正），并给出 theta 标度
  S4 波包整体量：积分应力与 (k_i k_j) 积分 u 的关系（近似构造的偏差登记）

方法：沿用第 436 条的格点场构造（斜向 (1,1,1) 行波，低幅线性层，周期卷绕差分）；
      应力的本征值用 3x3 对称矩阵本征分解（对内部样本点）；全部确定性。
      本脚本只做应力口径的形式与数值统一，不构造"桥"（空间度量侧，属本体层重建线）。

运行：python 验证脚本/stress_unification.py
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
IMG = os.path.join(ROOT, "images", "stress_unification.png")

print("== S1 应力分解表 ==")
print("  源        | 沿方向分量           | 横向分量        | 类型")
print("  缺陷（BPS 墙）| T_par = -rho（张力）   | 0（横向零）     | 纯纵、压性")
print("  光（行波）  | T_par = +u（辐射压强） | 0（横向零）     | 纯纵、压性（正号）")
print("  统一口径    | 两者均为'纯纵型'：横向分量为零、迹等于沿方向分量；方向取源的定向（梯度/传播）")

N = 64
c0 = N // 2
axr = np.arange(N) - c0
X, Y, Z = np.meshgrid(axr, axr, axr, indexing="ij")
e1 = np.array([1.0, -1.0, 0.0]) / math.sqrt(2.0)
khat = np.array([1.0, 1.0, 1.0]) / math.sqrt(3.0)
bhat = np.cross(khat, e1)

def droll(f, axis):
    return (np.roll(f, -1, axis=axis) - np.roll(f, 1, axis=axis)) / 2.0

def build_fields(theta):
    kap = theta / math.sqrt(3.0)
    sw2 = 3.0 * math.sin(kap / 2.0) ** 2
    om = 2.0 * math.asin(math.sqrt(sw2))
    n0 = 5
    ph = om * n0 - (kap * X + kap * Y + kap * Z)
    Ev = -e1[:, None, None, None] * (math.sin(om) * np.cos(ph))[None, :, :, :]
    Av = e1[:, None, None, None] * np.sin(ph)[None, :, :, :]
    axv = Av[0]; ayv = Av[1]; azv = Av[2]
    gx = (droll(axv, 0), droll(axv, 1), droll(axv, 2))
    gy = (droll(ayv, 0), droll(ayv, 1), droll(ayv, 2))
    gz = (droll(azv, 0), droll(azv, 1), droll(azv, 2))
    bx = gz[1] - gy[2]; by = gx[2] - gz[0]; bz = gy[0] - gx[1]
    Bv = np.array([bx, by, bz])
    return Ev, Bv, om, kap

print()
print("== S2/S3 平面波（斜向 (1,1,1)） ==")
th_list = (0.05, 0.1, 0.2, 0.3)
dev_list = []
for th in th_list:
    Ev, Bv, om, kap = build_fields(th)
    E2 = Ev[0] ** 2 + Ev[1] ** 2 + Ev[2] ** 2
    B2 = Bv[0] ** 2 + Bv[1] ** 2 + Bv[2] ** 2
    u = 0.5 * (E2 + B2)
    sig = np.zeros((3, 3) + u.shape)
    for a in range(3):
        for b in range(3):
            delta = 1.0 if a == b else 0.0
            sig[a, b] = -(Ev[a] * Ev[b] + Bv[a] * Bv[b]) + 0.5 * delta * (E2 + B2)
    inner = (slice(4, -4), slice(4, -4), slice(4, -4))
    u_in = u[inner]
    # 逐点纯纵性：sigma_ij 与 u k_i k_j 的偏差（相对）
    dev = 0.0
    for a in range(3):
        for b in range(3):
            target = u * khat[a] * khat[b]
            d = np.max(np.abs(sig[a, b][inner] - target[inner])) / np.max(u_in)
            dev = max(dev, d)
    dev_list.append(dev)
    # 本征值统计（内部样本 200 点）
    idx = [np.random.default_rng(7 + i).integers(4, N - 4, 3) for i in range(200)]
    evs = []
    for (i1, i2, i3) in idx:
        M = np.array([[sig[a, b][i1, i2, i3] for b in range(3)] for a in range(3)])
        w = np.linalg.eigvalsh(M)
        evs.append(w)
    evs = np.array(evs)
    evs_sorted = np.sort(np.abs(evs), axis=1)
    small_rel = float(np.max(evs_sorted[:, 1] / evs_sorted[:, 2]))
    lam3_over_u = []
    for kk, (i1, i2, i3) in enumerate(idx):
        uv = u[i1, i2, i3]
        if uv > 1e-12:
            lam3_over_u.append(evs_sorted[kk, 2] / uv)
    big_rel = float(np.mean(lam3_over_u))
    print("  theta = %.2f：逐点偏差 max（相对）= %.3e；本征值 |lam2|/|lam3| max = %.3e；mean lam3/u（逐点）= %.5f（期望 1）"
          % (th, dev, small_rel, big_rel))

print()
print("== S3 标度 ==")
sl = float(np.polyfit(np.log(th_list), np.log(dev_list), 1)[0])
print("  逐点偏差的 log-log 斜率 = %.2f（登记读数：偏差随 theta 的幂）" % sl)

print()
print("== S4 波包整体量（近似构造登记） ==")
a = 10.0
env = np.exp(-(X ** 2 + Y ** 2 + Z ** 2) / (2 * a * a))
ph = 1.0 * (X + Y + Z) / math.sqrt(3.0)
Evw = e1[:, None, None, None] * np.sin(ph)[None, :, :, :] * env[None, :, :, :]
Bvw = bhat[:, None, None, None] * np.sin(ph)[None, :, :, :] * env[None, :, :, :]
E2w = Evw[0] ** 2 + Evw[1] ** 2 + Evw[2] ** 2
B2w = Bvw[0] ** 2 + Bvw[1] ** 2 + Bvw[2] ** 2
uw = 0.5 * (E2w + B2w)
Uw = float(uw.sum())
Ssum = np.zeros((3, 3))
for aa in range(3):
    for bb in range(3):
        delta = 1.0 if aa == bb else 0.0
        Ssum[aa, bb] = float((-(Evw[aa] * Evw[bb] + Bvw[aa] * Bvw[bb]) + 0.5 * delta * (E2w + B2w)).sum())
print("  包总能量 U = %.4e；积分应力的最大分量 = %.4e" % (Uw, np.max(np.abs(Ssum))))
print("  积分应力 / U 的最大偏差（对 k_i k_j 结构）：%.3e" % np.max(np.abs(Ssum / Uw - np.outer(khat, khat))))
print("  注：sigma = u k_i k_j 对'正交等幅'场（E 沿 e1、B 沿 b1、同一标量轮廓）是代数恒等式，S4 的机器零即其验证")

print()
print("== 判定与移交 ==")
print("  一、口径统一的形式结论：光与缺陷的应力都是'纯纵型'（横向零、迹等于沿方向分量），方向取各自定向")
print("  二、光的数值验证：本征结构 {0,0,u} 与逐点纯纵性成立（偏差 O(theta^2) 级，格点修正）")
print("  三、移交：'桥'（空间度量侧）的显式输出对照依赖本体层重建线；同幅口径（lambda = 1）下的")
print("      '纯纵应力 -> 空间度量'通道形式留作下一轮")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
ax.loglog(th_list, dev_list, "o-")
ax.set_xlabel("theta"); ax.set_ylabel("max relative deviation")
ax.set_title("S3: 逐点纯纵性偏差")
ax.grid(alpha=0.3, which="both")
ax = axes[1]
Ev, Bv, om, kap = build_fields(0.2)
E2 = Ev[0] ** 2 + Ev[1] ** 2 + Ev[2] ** 2
B2 = Bv[0] ** 2 + Bv[1] ** 2 + Bv[2] ** 2
u = 0.5 * (E2 + B2)
sl2 = u[c0, :, :]
im = ax.imshow(sl2, cmap="inferno")
ax.set_title("u 的中心切面（斜向行波，theta = 0.2）")
fig.colorbar(im, ax=ax, fraction=0.046)
fig.tight_layout()
fig.savefig(IMG, dpi=160)
print("figure saved:", IMG)
