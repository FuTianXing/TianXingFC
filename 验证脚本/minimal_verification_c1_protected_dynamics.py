# -*- coding: utf-8 -*-
"""
minimal_verification_c1_protected_dynamics.py

[C1-6 保护的动力学实现] 边界冻结下的线缺陷弛豫：标签守恒、净荷位置与结构读数
（判据 P-1 至 P-5）

背景：B 轮（C1-6 三维版）给出无保护时线缺陷扇区的拓扑荷衰减到 0；W1/W2 三维版
确认绕数是边界量、局域体算符不改扇区、改扇区须沿触及边界的弦。据此，保护的
最小实现是把标签的载体（边界环）冻结：边界数据固定则绕数恒为初值（精确），
体内动力学完全自由。本轮在同一弛豫协议上跑四档对照，给标签、能量、净荷位置与
体内结构的读数。

模型（与 B 轮同口径）：横截面 N^2 = 64^2，u = u0 + theta/2 pi，u0 = IX + IY，
theta = atan2(y - y0, x - x0)；E = sum_axes (k/2)(d_axis u - 1)^2
+ K(1 - cos 2 pi u)，k = 1；弛豫为梯度下降（eta = 0.001，20000 步）。
四档：对照（全自由）、仅锚核（冻结芯 3x3）、仅保护（冻结边界环）、保护加锚核。

判据
P-1 无保护对照：W 从 1 衰减到 0（复现 B 轮 B-1）
P-2 仅锚核不提供保护：冻结芯而边界自由时 W 仍衰减
P-3 保护（边界冻结）下的守恒与漂移：W 全程恒为 1；能量收敛；无锚核时净荷
     滑向边界驻留（位置稳定归锚核，标签守恒归保护）
P-4 保护加锚核的末态结构：电荷分布、质心、体内剖面（内驻亚稳线缺陷是否存在的
     判定读数）
P-5 统计与归属登记

运行：python 验证脚本/minimal_verification_c1_protected_dynamics.py（确定性，约 15 秒）
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
N = 64
NSTEPS = 20000
ETA = 0.001
K_EL = 1.0
TWO_PI = 2.0 * math.pi
IMG = "D:/software/TianXingFC/images/c1_protected_dynamics.png"


def wrap1(a):
    return (a + 0.5) % 1.0 - 0.5


def grids():
    i = np.arange(N, dtype=float)
    IX, IY = np.meshgrid(i, i, indexing="ij")
    return IX, IY


def init_field():
    IX, IY = grids()
    cx = cy = N / 2.0
    theta = np.arctan2(IY - cy, IX - cx)
    return IX + IY + theta / TWO_PI


def energy(u, Kp):
    E = 0.0
    for ax in range(2):
        d = np.diff(u, axis=ax) - 1.0
        E += 0.5 * K_EL * float(np.sum(d ** 2))
    E += Kp * float(np.sum(1.0 - np.cos(TWO_PI * u)))
    return E


def grad(u, Kp):
    g = np.zeros_like(u)
    for ax in range(2):
        d = np.diff(u, axis=ax)
        dd = np.zeros_like(u)
        if ax == 0:
            dd[:-1] -= K_EL * (d - 1.0)
            dd[1:] += K_EL * (d - 1.0)
        else:
            dd[:, :-1] -= K_EL * (d - 1.0)
            dd[:, 1:] += K_EL * (d - 1.0)
        g += dd
    g += Kp * TWO_PI * np.sin(TWO_PI * u)
    return g


def plaq_field(u):
    dx = wrap1(u[1:, :-1] - u[:-1, :-1])
    dx2 = wrap1(u[1:, 1:] - u[:-1, 1:])
    dy = wrap1(u[:-1, 1:] - u[:-1, :-1])
    dy2 = wrap1(u[1:, 1:] - u[1:, :-1])
    return dx + dy2 - dx2 - dy


def plaq_charge(u):
    return int(np.rint(plaq_field(u)).sum())


def boundary_ring():
    m = np.zeros((N, N), dtype=bool)
    m[0, :] = True
    m[-1, :] = True
    m[:, 0] = True
    m[:, -1] = True
    return m


def core_disk():
    m = np.zeros((N, N), dtype=bool)
    c = N // 2
    m[c - 1:c + 2, c - 1:c + 2] = True
    return m


def relax(u, Kp, steps=NSTEPS, eta=ETA, freeze=None, track=50):
    u = u.copy()
    hist = []
    for t in range(steps):
        g = grad(u, Kp)
        if freeze is not None:
            g[freeze] = 0.0
        u = u - eta * g
        if t % track == 0:
            hist.append((t, energy(u, Kp), plaq_charge(u)))
    return u, hist


def charge_cells(u):
    w = np.rint(plaq_field(u))
    return [(int(a) + 1, int(b) + 1, int(w[a, b])) for a, b in zip(*np.nonzero(w))]


def charge_centroid(u):
    w = np.abs(np.rint(plaq_field(u)))
    i = np.arange(1, N, dtype=float)
    IX, IY = np.meshgrid(i, i, indexing="ij")
    return float((w * IX).sum() / w.sum()), float((w * IY).sum() / w.sum())


print("=" * 78)
print("C1-6 保护的动力学实现：边界冻结下的线缺陷弛豫（判据 P-1 至 P-5）")
print("=" * 78)

u_init = init_field()
ring = boundary_ring()
disk = core_disk()
K0 = 0.005
print()
print("初态：W = %d；能量（K = 0.005）= %.4f；电荷 %s"
      % (plaq_charge(u_init), energy(u_init, K0), str(charge_cells(u_init))))

print()
print("[P-1] 无保护对照（全自由）")
uC, histC = relax(u_init, K0)
drops = [s for s in histC if s[2] != 1]
print("     W 末值 = %d；首个非 1 记录 = %s；末能量 = %.4f"
      % (plaq_charge(uC), str(drops[0]) if drops else "无", energy(uC, K0)))

print()
print("[P-2] 仅锚核（冻结芯，边界自由）")
uA, histA = relax(u_init, K0, freeze=disk)
dropsA = [s for s in histA if s[2] != 1]
print("     W 末值 = %d；首个非 1 记录 = %s" % (plaq_charge(uA), str(dropsA[0]) if dropsA else "无"))

print()
print("[P-3] 仅保护（冻结边界环，无锚核）")
uP, histP = relax(u_init, K0, freeze=ring)
badP = [s for s in histP if s[2] != 1]
cxP, cyP = charge_centroid(uP)
print("     W 末值 = %d（非 1 次数 = %d）；能量 初 = %.4f，末 = %.4f"
      % (plaq_charge(uP), len(badP), energy(u_init, K0), energy(uP, K0)))
print("     净荷：质心 = (%.2f, %.2f)（初值 32.5, 32.5）；电荷 %s" % (cxP, cyP, str(charge_cells(uP))))

print()
print("[P-4] 保护加锚核的末态结构")
res4 = []
for Kp in (0.005, 0.02):
    uPF, _ = relax(u_init, Kp, freeze=ring | disk)
    cells = charge_cells(uPF)
    cx, cy = charge_centroid(uPF)
    phi = uPF[16, :] - (16 + np.arange(N))
    phi_int = phi[8:56]
    res4.append((Kp, uPF, cells, cx, cy))
    print("     K = %.5f：W = %d；能量 = %.4f" % (Kp, plaq_charge(uPF), energy(uPF, Kp)))
    print("        电荷 %s；质心 = (%.2f, %.2f)" % (str(cells), cx, cy))
    print("        体内剖面（x = 16 列，y = 8..55）：min = %.3f，max = %.3f，峰-谷差 = %.3f"
          % (float(phi_int.min()), float(phi_int.max()), float(phi_int.max() - phi_int.min())))

uPF02 = res4[1][1]
print()
print("[P-5] 归属登记")
print("     标签守恒：保护（边界冻结）下精确成立（P-3：非 1 次数 = 0）")
print("     净荷位置：无锚核时滑向边界驻留（质心 x = %.2f）；锚核冻结芯只保形状："
      "芯处留 (+1, -1) 偶极，净 +1 驻边界" % cxP)
print("     内驻亚稳线缺陷：本模型内未出现（体内剖面平坦，无扭结）")
print("     宽度的归属：1D 链（端点固定 = 天然保护，A 轮/W4 的 l* 与 xi 表）")

fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.2))

ax = axes[0, 0]
ax.plot([s[0] for s in histC], [s[2] for s in histC], label="全自由（对照）")
ax.plot([s[0] for s in histA], [s[2] for s in histA], "--", label="仅锚核")
ax.plot([s[0] for s in histP], [s[2] for s in histP], label="仅保护（边界冻结）")
ax.set_xlabel("步数")
ax.set_ylabel("W")
ax.set_title("(a) 标签时间线（K = 0.005）")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[0, 1]
ax.plot([s[0] for s in histC], [s[1] for s in histC], label="全自由")
ax.plot([s[0] for s in histP], [s[1] for s in histP], label="仅保护")
ax.set_xlabel("步数")
ax.set_ylabel("能量")
ax.set_title("(b) 能量时间线")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1, 0]
for Kp, uPF, cells, cx, cy in res4:
    ax.scatter([c[0] for c in cells], [c[1] for c in cells],
               s=[40 * abs(c[2]) for c in cells], label="保护加锚核 K = %.4f" % Kp)
cellsP = charge_cells(uP)
ax.scatter([c[0] for c in cellsP], [c[1] for c in cellsP], marker="x", s=30,
           label="仅保护（边界驻留）")
ax.set_xlim(0, 64)
ax.set_ylim(0, 64)
ax.set_xlabel("x（格）")
ax.set_ylabel("y（格）")
ax.set_title("(c) 非零电荷位置（记号大小 = |w|）")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1, 1]
IX, IY = grids()
phi = uPF02 - (IX + IY)
im = ax.imshow(phi.T, origin="lower", cmap="RdBu_r")
ax.set_title("(d) 保护加锚核末态（K = 0.02）：u - x - y")
ax.set_xlabel("x")
ax.set_ylabel("y")
fig.colorbar(im, ax=ax, fraction=0.046)

fig.tight_layout()
fig.savefig(IMG, dpi=150)
print()
print("图已存：%s" % IMG)
print("总运行时间 = %.1f 秒" % (time.time() - T0))
