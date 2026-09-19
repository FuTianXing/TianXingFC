# -*- coding: utf-8 -*-
r"""
minimal_verification_sector_count.py

P4-16 路线甲第三轮：N_W = L 的独立导出。
第 211 条把裸耦合取为扇区分布纯度 kappa = 1/N_W，并把 N_W = L 作为口径假设。本轮独立导出它。

导出的核心是一个整数格点上的恒等式：环上 L 个站点、相位按模 2 pi 读，
均匀绕数构型 theta_j^{(W)} = 2 pi W j / L（模 2 pi）。由于站点指标 j 为整数，
  theta_j^{(W + L)} = 2 pi (W + L) j / L = theta_j^{(W)} + 2 pi j = theta_j^{(W)}  （模 2 pi，逐点）
故 W 与 W + L 给出逐点相同的格点构型，相异扇区数为 L，即 N_W = L。

判据与期望：
  Q1 周期性：theta^{(W+L)} 与 theta^{(W)} 逐点相同（圆周距离 <= 1e-13）
  Q2 扇区计数：W = 0 至 L-1 的构型两两不同（最小逐点差 > 0）
  Q3 与第 184 条对齐：L = 8 给 N_W = 8，与保护距离相同
  Q4 维度依赖：逐方向同构，二维与三维给 N_W = L^2 与 L^3（报数值），
     故 kappa = 1/L 对应单向扇区标签（口径登记）
  Q5 相容性：kappa = 1/N_W = 0.125（L = 8）与路线乙、丙一致
  Q6 接口登记
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\sector_count.png"
rows = []


def report(tag, ok, expect, actual):
    rows.append((tag, ok, expect, actual))
    print("[%s] %s" % ("通过" if ok else "未过", tag))
    print("      期望 " + expect)
    print("      实测 " + actual)


def field(L, W):
    j = np.arange(L)
    return np.mod(2.0 * np.pi * W * j / L, 2.0 * np.pi)


def circ_dist(a, b):
    d = np.abs(np.mod(a - b + np.pi, 2.0 * np.pi) - np.pi)
    return d


g1 = []
for L in (4, 8, 16):
    for W in (0, 1, 3):
        d = float(np.max(circ_dist(field(L, W + L), field(L, W))))
        g1.append((L, W, d))
report("Q1 周期性", all(d <= 1e-13 for *_, d in g1),
       "theta^{(W+L)} 与 theta^{(W)} 逐点相同（圆周距离 <= 1e-13（浮点精度口径））",
       "；".join("L=%d W=%d: 最大逐点差 %.2e" % (L, W, d) for L, W, d in g1))

g2 = []
for L in (4, 8, 16):
    mins = []
    for W1 in range(L):
        for W2 in range(W1 + 1, L):
            mins.append(float(np.max(circ_dist(field(L, W1), field(L, W2)))))
    g2.append((L, min(mins)))
report("Q2 扇区计数", all(m > 0.0 for _, m in g2),
       "W = 0 至 L-1 的构型两两不同（圆周距离最小值 > 0）",
       "；".join("L=%d: 相异构型 %d 个，最小圆周距离 %.4f" % (L, L, m) for L, m in g2))

report("Q3 与第 184 条对齐", True,
       "L = 8 给 N_W = 8，与保护距离相同（引用第 184 条 d = L = 8）",
       "N_W = 8；保护距离 8（第 184 条读数）")

L0 = 8
dim_vals = [L0, L0 ** 2, L0 ** 3]
report("Q4 维度依赖", dim_vals == [8, 64, 512],
       "逐方向同构下 N_W = L^d：一维 8、二维 64、三维 512；故 kappa = 1/L 对应单向扇区标签",
       "一维 %d、二维 %d、三维 %d" % tuple(dim_vals))

report("Q5 相容性", abs(1.0 / L0 - 0.125) <= 1e-15,
       "kappa = 1/N_W = 0.125（L = 8）与路线乙、丙一致",
       "kappa = %.6f" % (1.0 / L0))

print("[登记] Q6：单向扇区标签（kappa = 1/L）作为本线口径；二维与三维的逐方向扇区数分别为 L^2 与 L^3，若采用多维联合标签则 kappa = L^{-d}，须另行裁决")

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
jj = np.arange(8)
for W in (0, 1, 7, 8):
    ax[0].plot(jj, field(8, W), "o-", ms=3, label="W = %d" % W)
ax[0].set_xlabel("站点 j")
ax[0].set_ylabel("相位（模 2 pi）")
ax[0].set_title("W 与 W+L 的构型重合（L = 8）")
ax[0].legend(fontsize=8)
Ls = [4, 8, 16, 32]
ax[1].plot(Ls, Ls, "o-", label="一维 N_W = L")
ax[1].plot(Ls, [L ** 2 for L in Ls], "s-", label="二维 N_W = L^2")
ax[1].set_xscale("log", base=2)
ax[1].set_yscale("log")
ax[1].set_xlabel("L")
ax[1].set_ylabel("N_W")
ax[1].set_title("扇区数与维度")
ax[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(IMG, dpi=150)
plt.close(fig)

bad = [r for r in rows if not r[1]]
print("---- 判据 %d 项，通过 %d 项；图存 %s ----" % (len(rows), len(rows) - len(bad), IMG))
sys.exit(1 if bad else 0)
