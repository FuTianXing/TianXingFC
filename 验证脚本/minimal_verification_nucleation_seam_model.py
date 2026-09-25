# -*- coding: utf-8 -*-
"""成核-接缝几何的最小模型：alpha = 1 - q 标度定理与 3/5 的几何靶（第 416 条，P0-5 候选机制四轮）。

模型族：持续成核（J(t) = J0 t^p）+ 波前覆盖（半径 R(t) = v0 t^{1-q}/(1-q)）+ Avrami 完成判据
        （单位体积覆盖积分 = 1），畴尺度取核密度的 Voronoi 口径 xi = coeff * N^{-1/3}。

判据：
  V1 标度验证：固定 p，扫 J0，拟合 ln xi vs ln tau 斜率 = 1 - q（q = 0、0.2、0.4、0.6）
  V2 与成核幂次无关：p = 0、1、2 同 q 下斜率一致
  V3 几何靶：带内 alpha 对应 q 窗口
  V4 蒙特卡洛：3D 泊松点集的平均近邻系数（xi 口径）
  V5 对照表：机制 q 值 -> alpha -> 带判定

运行：python 验证脚本/minimal_verification_nucleation_seam_model.py
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
IMG = os.path.join(ROOT, "images", "nucleation_seam_model.png")

LO, HI = 0.5598, 0.6015

def tau_completion(p, q, J0, v0=1.0):
    power = p + 4.0 - 3.0 * q
    return ((power * (1.0 - q) ** 3) / (J0 * v0 ** 3)) ** (1.0 / power)

def N_density(p, q, J0, v0=1.0):
    tau = tau_completion(p, q, J0, v0)
    return J0 * tau ** (p + 1.0) / (p + 1.0)

def fit_slope(xs, ys):
    lx = np.log(xs)
    ly = np.log(ys)
    return float(np.polyfit(lx, ly, 1)[0])

print("== V1 标度验证（固定 p = 0，扫 J0，xi = coeff * N^{-1/3}） ==")
COEFF = 0.554  # 3D 泊松平均近邻系数（V4 用蒙特卡洛核对）
J0_LIST = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
for q in (0.0, 0.2, 0.4, 0.6):
    taus, xis = [], []
    for J0 in J0_LIST:
        tau = tau_completion(0.0, q, J0)
        N = N_density(0.0, q, J0)
        xi = COEFF * N ** (-1.0 / 3.0)
        taus.append(tau)
        xis.append(xi)
    slope = fit_slope(taus, xis)
    print("  q = %.1f：斜率 = %.6f（期望 %.6f），tau 范围 [%.3e, %.3e]，xi 范围 [%.3e, %.3e]" % (
        q, slope, 1.0 - q, min(taus), max(taus), min(xis), max(xis)))
print()

print("== V2 与成核幂次无关（q = 0.4） ==")
for p_ in (0.0, 1.0, 2.0):
    taus, xis = [], []
    for J0 in J0_LIST:
        tau = tau_completion(p_, 0.4, J0)
        N = N_density(p_, 0.4, J0)
        xi = COEFF * N ** (-1.0 / 3.0)
        taus.append(tau)
        xis.append(xi)
    slope = fit_slope(taus, xis)
    print("  p = %.0f：斜率 = %.6f（期望 0.6）" % (p_, slope))
print()

print("== V3 几何靶（带内 alpha <-> q 窗口） ==")
q_lo = 1.0 - HI
q_hi = 1.0 - LO
print("  alpha in [%.4f, %.4f] <-> q in [%.4f, %.4f]" % (LO, HI, q_lo, q_hi))
print("  3/5（alpha = 0.6）<-> q = 2/5 = 0.4")
print("  必要条件 alpha > 1/2（与链条曲线有交点）<-> q < 1/2")
print("  381 条追赶变体 alpha = 1/2 <-> q = 1/2（恰在边界，无定点）")
print()

print("== V4 蒙特卡洛：3D 泊松平均近邻系数 ==")
rng = np.random.default_rng(20260926)
for N in (500, 2000, 8000):
    pts = rng.random((N, 3))
    tree = cKDTree(pts, boxsize=1.0)
    d, _ = tree.query(pts, k=2)
    nn = d[:, 1]
    mean_nn = float(np.mean(nn))
    coeff = mean_nn * N ** (1.0 / 3.0)
    print("  N = %5d：平均近邻 = %.5f，系数 mean_nn * N^{1/3} = %.5f" % (N, mean_nn, coeff))
print("  采用系数 %.3f（与理论口径一致）" % COEFF)
print()

print("== V5 对照表（机制 q -> alpha -> 带判定） ==")
rows = [
    ("视界线性（均匀成核、常速波前）", 0.0),
    ("靶：3/5 几何（q = 2/5）", 0.4),
    ("追赶变体（第 381 条）", 0.5),
    ("KZ 幂律（alpha = 1/3，前提已判负）", 2.0/3.0),
]
for name, q in rows:
    alpha = 1.0 - q
    ok = LO <= alpha <= HI
    print("  %s：q = %.4f，alpha = %.4f，带内 = %s" % (name, q, alpha, ok))
print()

print("== 图 ==")
fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6), dpi=110)
fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.16, wspace=0.24)

ax = axes[0]
for q, col in ((0.0, "#14508c"), (0.2, "#2f7a2f"), (0.4, "#c0392b"), (0.6, "#7030a0")):
    taus, xis = [], []
    for J0 in J0_LIST:
        tau = tau_completion(0.0, q, J0)
        N = N_density(0.0, q, J0)
        taus.append(tau)
        xis.append(COEFF * N ** (-1.0 / 3.0))
    ax.loglog(taus, xis, "o-", color=col, lw=1.8, markersize=6,
              label="q = %.1f (slope %.1f)" % (q, 1.0 - q))
ax.set_xlabel("completion time tau (lattice units)", fontsize=10.5)
ax.set_ylabel("domain scale xi", fontsize=10.5)
ax.set_title("V1 xi ~ tau^{1-q} (independent of p)", fontsize=10.5)
ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
ax.legend(fontsize=8.5, loc="upper left")

ax = axes[1]
qs = np.linspace(0.0, 0.7, 200)
alphas = 1.0 - qs
ax.plot(qs, alphas, color="#c0392b", lw=2.0, label="alpha = 1 - q")
ax.axhspan(LO, HI, color="#ffe9b0", alpha=0.85, label="allowed band")
for name, q, col in (("horizon", 0.0, "#14508c"), ("target 3/5", 0.4, "#2f7a2f"),
                     ("chase", 0.5, "#7030a0"), ("KZ", 2.0/3.0, "#888888")):
    ax.plot([q], [1.0 - q], "o", color=col, markersize=8)
    ax.annotate(name, (q, 1.0 - q), textcoords="offset points", xytext=(7, -10), fontsize=9.5, color=col)
ax.axvline(0.4, color="#2f7a2f", ls=":", lw=1.2)
ax.set_xlim(0.0, 0.75)
ax.set_ylim(0.0, 1.1)
ax.set_xlabel("front deceleration exponent q", fontsize=10.5)
ax.set_ylabel("alpha", fontsize=10.5)
ax.set_title("V5 mechanisms on the q-alpha line", fontsize=10.5)
ax.grid(True, ls=":", lw=0.5, alpha=0.6)
ax.legend(fontsize=8.5, loc="lower left")

fig.savefig(IMG)
print("  已保存：", IMG)
