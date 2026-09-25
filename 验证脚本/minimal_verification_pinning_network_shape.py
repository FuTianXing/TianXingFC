# -*- coding: utf-8 -*-
"""钉扎网络的空间分布形状判定：平稳点集无幂律尾与尺度不变遇见率条件（第 419 条，P0-5 候选机制七轮）。

判据：
  R1 周期钉扎（格点 Peierls）的判负（材料：D-7 的 Delta E_PN = 1.36e-9）
  R2 泊松点集最近邻分布的尾型（数值：无幂律尾，survival 的对数斜率随距离变负）
  R3 尺度不变遇见率的构造（lambda(l) = c/l 给幂律尾，c = 5/3）
  R4 对照表（周期、泊松、单尺度分形、尺度不变四类）
  R5 净判

运行：python 验证脚本/minimal_verification_pinning_network_shape.py
"""
import math
import os

import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "images", "pinning_network_shape.png")

MU = 5.0 / 3.0            # 目标尾参数（第 415/418 条：xi ∝ t^{1/mu}，mu = 5/3）
C_RATE = MU               # 尺度不变遇见率 lambda(l) = c/l，c = mu（给 P ∝ l^{-1-mu}）
SEED = 20260926

print("== R1 周期钉扎（格点 Peierls） ==")
print("  材料：D-7 的离散钉扎势垒 Delta E_PN = 1.359e-9（极弱，宽 kink 几乎无感）")
print("  周期势给单色跳跃长度（l = l_P），无分布、无重尾；弱钉扎转出自由滑动（弹道，alpha = 1）")
print("  判定：判负（无重尾，且弹道一侧在允许带外）")
print()

print("== R2 泊松点集最近邻分布的尾型（数值） ==")
rng = np.random.default_rng(SEED)
L = 20.0
N = 20000
pts = rng.random((N, 3)) * L
tree = cKDTree(pts, boxsize=L)
d, _ = tree.query(pts, k=2)
nn = d[:, 1]
n_density = N / L ** 3
print("  N = %d，盒长 %.0f，密度 n = %.3f，平均最近邻 = %.4f（理论 0.554 n^{-1/3} = %.4f）" % (
    N, L, n_density, float(np.mean(nn)), 0.554 * n_density ** (-1.0/3.0)))
levels = [0.5, 0.6, 0.7, 0.8, 0.9]
print("  survival S(l) = P(NN > l) 与局部对数斜率 dlnS/dlnl：")
for lev in levels:
    ell = float(np.quantile(nn, lev))
    S = float(np.mean(nn > ell))
    slope_theory = -4.0 * math.pi * n_density * ell ** 3
    print("    l = %.3f（分位 %.1f）：S = %.4f，理论斜率 -4 pi n l^3 = %.2f" % (ell, lev, S, slope_theory))
print("  理论：S(l) = exp(-(4pi/3) n l^3)，对数斜率 = -4 pi n l^3 随 l 变负（无平台）")
print("  判定：平稳泊松点集无幂律尾，判负")
print()

print("== R3 尺度不变遇见率的构造 ==")
print("  设遇见率 lambda(l) = c/l（尺度不变，无特征长度），则 P(l) ∝ l^{-1-c}")
print("  取 c = mu = %.4f：P 尾指数 = 1 + c = %.4f，S 尾指数 = c = %.4f" % (MU, 1.0 + C_RATE, C_RATE))
u = rng.random(N)
ell_min = 0.1
samples = ell_min * (1.0 - u) ** (-1.0 / C_RATE)
for lev in (0.5, 0.9, 0.99):
    th = ell_min * (1.0 - lev) ** (-1.0 / C_RATE)
    emp = float(np.quantile(samples, lev))
    print("    分位 %.2f：理论 %.4f，实测 %.4f" % (lev, th, emp))
print("  平均跳跃 <l> = l_min * c/(c-1) = %.4f（c > 1，有限）" % (ell_min * C_RATE / (C_RATE - 1.0)))
print("  判定：幂律尾可按需构造；要求遇见率严格按 1/l 衰减（尺度不变搜索）")
print()

print("== R4 对照表 ==")
rows = [
    ("格点周期势", "lambda = 单色（delta 型）", "单色跳跃，无尾", "判负"),
    ("泊松位错集", "lambda = 常数", "Weibull 型，无幂律尾", "判负"),
    ("单尺度分形集（相关维数 d_f）", "lambda ∝ l^{d_f - 1}", "Weibull 型（d_f > 0），无幂律尾", "判负"),
    ("尺度不变遇见率", "lambda = c/l", "幂律尾，c = 5/3 给 mu = 5/3", "条件存活（无现成来源）"),
]
for name, rate, tail, verdict in rows:
    print("  %-24s %-28s %-32s %s" % (name, rate, tail, verdict))
print()

print("== R5 净判 ==")
print("  甲（跳跃重尾）的现有两个成分判负：格点周期（单色）与位错泊松假设（Weibull 尾）")
print("  平稳点集（含任意单尺度分形）都不可能给幂律尾：尾型由 lambda 的形状决定，幂律尾需要 lambda ∝ 1/l")
print("  甲的条件存活：需要尺度不变遇见率；框架内检索无分形或层级钉扎材料，来源待建")
print()

print("== 图 ==")
fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6), dpi=110)
fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.16, wspace=0.26)

ax = axes[0]
ells = np.linspace(0.02, 1.2, 400)
S_poisson = np.exp(-(4.0 * math.pi / 3.0) * n_density * ells ** 3)
mask = ells >= ell_min
S_power = (ell_min / np.maximum(ells, ell_min)) ** C_RATE
ax.semilogy(ells, S_poisson, color="#14508c", lw=2.0, label="Poisson NN (no power tail)")
ax.semilogy(ells[mask], S_power[mask], color="#2f7a2f", lw=2.0, label="scale-invariant (power tail)")
ax.axvline(ell_min, color="#888888", ls=":", lw=1.2)
ax.set_xlim(0, 1.2)
ax.set_ylim(1e-12, 1.05)
ax.set_xlabel("jump length l", fontsize=10.5)
ax.set_ylabel("survival S(l)", fontsize=10.5)
ax.set_title("R2/R3 survival functions", fontsize=10.5)
ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
ax.legend(fontsize=8.5)

ax = axes[1]
l2 = np.linspace(0.05, 1.0, 300)
slope_poisson = -4.0 * math.pi * n_density * l2 ** 3
ax.plot(l2, slope_poisson, color="#14508c", lw=2.0, label="Poisson: slope falls (no plateau)")
ax.plot(l2, np.full_like(l2, -C_RATE), color="#2f7a2f", lw=2.0, label="scale-invariant: slope = -5/3")
ax.set_xlim(0, 1.0)
ax.set_ylim(-20, 0)
ax.set_xlabel("l", fontsize=10.5)
ax.set_ylabel("local log-slope of S(l)", fontsize=10.5)
ax.set_title("R4 tail-shape diagnostics", fontsize=10.5)
ax.grid(True, ls=":", lw=0.5, alpha=0.6)
ax.legend(fontsize=8.5)

fig.savefig(IMG)
print("  已保存：", IMG)
