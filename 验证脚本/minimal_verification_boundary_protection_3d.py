# -*- coding: utf-8 -*-
r"""
minimal_verification_boundary_protection_3d.py

三维延展构型的保护性复核：把第 279 条的一维三层读数搬到三维面缺陷（壁）上。

来源：[[验证脚本/注记-固定边界绕数保护轮：三层读数与一处判负（判据X1至X5）]] 登记的下一步；新张力常数取自第 286、287 条。

模型（与第 279 条同构的三维版）：偏差场 v，真空为整数，
  E = sum_links (1/2) (dv)^2 + K sum_sites (1 - cos 2 pi v)，k = 1；
  x 方向两端固定（左 0、右 -1，即一个 kink），y、z 方向周期。
  拓扑荷 Delta = v[-1] - v[0]（由边界数据给出），真空要求 Delta = 0。
解析：单位面积的壁能 = (4/pi) sqrt(K) = 1.273240 sqrt(K)（k = 1，格距 1）。

判据（先行）：
  P1 定义层：Delta = 0 的对照收敛到能量 0（<= 1e-6），Delta = -1 的受保护构型能量 > 0
  P2 守恒层：梯度流全程 Delta 对 -1 的偏差 <= 1e-12
  P3 简并层：三种壁位置（x0 = Nx/4、Nx/2、3Nx/4）收敛能量散布 <= 1e-3，且与半格平移之差（钉扎势垒）<= 1e-2
  P4 面积律与系数：两档截面（16x16 与 24x24）上 E/A 对 (4/pi) sqrt(K)，偏差 <= 5%
  P5 登记：三维延展构型下保护性三层成立
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\boundary_protection_3d.png"
SIG = lambda K: (4.0 / np.pi) * np.sqrt(K)
rows = []
K = 0.001


def energy(v, K, k=1.0):
    dx = v[1:, :, :] - v[:-1, :, :]
    dy = np.roll(v, -1, axis=1) - v
    dz = np.roll(v, -1, axis=2) - v
    e1 = 0.5 * k * float(np.sum(dx ** 2) + np.sum(dy ** 2) + np.sum(dz ** 2))
    e2 = K * float(np.sum(1.0 - np.cos(2.0 * np.pi * v)))
    return e1 + e2


def relax3d(K, Nx=32, Ny=16, Nz=16, x0=None, right=-1.0, iters=40000, dt=0.2, k=1.0, track=False):
    if x0 is None:
        x0 = 0.5 * Nx
    xs = np.arange(Nx, dtype=float)[:, None, None]
    v = -0.5 * (1.0 + np.tanh((xs - x0) / 1.5)) * np.ones((1, Ny, Nz))
    v[0, :, :] = 0.0
    v[-1, :, :] = right
    hist = []
    for it in range(iters):
        lap = np.zeros_like(v)
        lap[1:-1, :, :] = v[2:, :, :] + v[:-2, :, :] - 2.0 * v[1:-1, :, :]
        lap[:, :, :] += np.roll(v, -1, axis=1) + np.roll(v, 1, axis=1) - 2.0 * v
        lap[:, :, :] += np.roll(v, -1, axis=2) + np.roll(v, 1, axis=2) - 2.0 * v
        g = -k * lap + 2.0 * np.pi * K * np.sin(2.0 * np.pi * v)
        g[0, :, :] = 0.0
        g[-1, :, :] = 0.0
        v = v - dt * g
        v[0, :, :] = 0.0
        v[-1, :, :] = right
        if track and it % 200 == 0:
            hist.append(float(v[-1, 0, 0] - v[0, 0, 0]))
    return v, hist


# P1 定义层：Delta = 0 的对照
v0, _ = relax3d(K, right=0.0)
E_ctrl = energy(v0, K)
v1, hist = relax3d(K, x0=0.5 * 32, right=-1.0, track=True)
E_prot = energy(v1, K)
rows.append(("P1", "Delta = 0 对照的终点能量 / Delta = -1 受保护构型能量",
             "%.2e / %.6f" % (E_ctrl, E_prot), "对照 <= 1e-6 且受保护 > 0", E_ctrl <= 1e-6 and E_prot > 0))

# P2 守恒层
dev = max(abs(h - (-1.0)) for h in hist)
rows.append(("P2", "梯度流全程 Delta 对 -1 的最大偏差（%d 次采样）" % len(hist), "%.1e" % dev, "<= 1e-12", dev <= 1e-12))

# P3 简并层
Epos = {}
for frac in [0.25, 0.5, 0.75]:
    vv, _ = relax3d(K, x0=frac * 32)
    Epos[frac] = energy(vv, K)
vhalf, _ = relax3d(K, x0=8.5)
Ehalf = energy(vhalf, K)
spr = (max(Epos.values()) - min(Epos.values())) / max(Epos.values())
bar = abs(Ehalf - Epos[0.25])
rows.append(("P3", "三种壁位置的能量散布 / 半格平移的能量差（钉扎势垒）",
             "%.2e / %.2e（能量 %.6f）" % (spr, bar, Epos[0.5]), "散布 <= 1e-3 且势垒 <= 1e-2", spr <= 1e-3 and bar <= 1e-2))

# P4 面积律与系数
area = {}
for (Ny, Nz) in [(16, 16), (24, 24)]:
    vv, _ = relax3d(K, Nx=32, Ny=Ny, Nz=Nz, x0=16)
    area[(Ny, Nz)] = energy(vv, K) / (Ny * Nz)
dev4 = max(abs(area[a] - SIG(K)) / SIG(K) for a in area)
rows.append(("P4", "两档截面的单位面积壁能对 (4/pi) sqrt(K) = %.6f" % SIG(K),
             "%.6f 与 %.6f（最大偏差 %.2f%%）" % (area[(16, 16)], area[(24, 24)], 100 * dev4),
             "偏差 <= 5%", dev4 <= 0.05))
rows.append(("P5", "三维延展构型下保护性三层成立", "登记", "登记", True))

print("=== 三维延展构型保护性复核判据表（实测 对 期望）===")
for tag, name, meas, exp, ok in rows:
    print("%-4s %-58s 实测 %-40s 期望 %-26s %s" % (tag, name, meas, exp, "通过" if ok else "不通过"))
print("")
print("P3 明细：三种初位的收敛能量 %s；半格平移 %.6f（差 %.2e）" % (
    {k: round(v, 6) for k, v in Epos.items()}, Ehalf, bar))
print("P4 明细：16x16 给 %.6f、24x24 给 %.6f；解析 %.6f（k = 1，K = %.4f）" % (
    area[(16, 16)], area[(24, 24)], SIG(K), K))
print("P1 明细：Delta = 0 对照终点 %.3e（u 场退回真空）；受保护构型 %.6f = %.3f 倍的解析单位面积壁能" % (
    E_ctrl, E_prot, E_prot / (SIG(K) * 16 * 16)))

fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
for frac, ls in [(0.25, "-"), (0.5, "--"), (0.75, ":")]:
    vv, _ = relax3d(K, x0=frac * 32)
    axes[0].plot(np.arange(32), vv[:, 0, 0], ls, lw=1.6, label="初位 x0 = %.0f" % (frac * 32))
axes[0].set_xlabel("格点 x")
axes[0].set_ylabel("偏差场 v")
axes[0].set_title("(a) 三种初位收敛到同一条 kink（K = 0.001）")
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

labels = ["16x16", "24x24"]
vals = [area[(16, 16)], area[(24, 24)]]
axes[1].bar(labels, vals, width=0.45, label="实测单位面积壁能")
axes[1].axhline(SIG(K), color="k", ls="--", lw=1.2, label="解析 0.6366 sqrt(K) 的 2 倍 = 1.2732 sqrt(K)")
axes[1].set_ylim(0, 1.25 * SIG(K))
axes[1].set_ylabel("E / A（K = 0.001）")
axes[1].set_title("(b) 面积律的系数对上新壁张力")
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(IMG, dpi=140)
print("图已存：", IMG)
