# -*- coding: utf-8 -*-
"""长程关联点集的遇见率实验：候选乙首轮（第 422 条，P0-5 跨线启发候选乙）。

判据：
  N1 生成器核对：s = 0（白噪声谱）档应接近泊松
  N2 关联扫描：各 s 档的 g(l) = -dlnS/dlnl 形状
  N3 平台检测（滑动窗口口径，与第 419/421 条一致）
  N4 中位 g 对照
  N5 靶位读数：平台值 vs 5/3

方法：3D 高斯随机场（功率谱 P(k) ∝ k^{-s}）阈值化到目标密度，格点占位加亚格随机偏移；
      cKDTree 测最近邻分布。

运行：python 验证脚本/minimal_verification_correlated_point_set.py
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
IMG = os.path.join(ROOT, "images", "correlated_point_set.png")

TARGET = 5.0 / 3.0
N = 128
L = float(N)
RHO = 0.01
SEED = 20260926

def grf_points(s_exp, rho=RHO, seed=SEED):
    rng = np.random.default_rng(seed + int(s_exp * 100))
    kx = np.fft.fftfreq(N) * 2.0 * np.pi
    KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing="ij")
    K = np.sqrt(KX ** 2 + KY ** 2 + KZ ** 2)
    amp = np.zeros_like(K)
    mask = K > 0
    amp[mask] = K[mask] ** (-s_exp / 2.0)
    noise = rng.normal(size=(N, N, N)) + 1j * rng.normal(size=(N, N, N))
    field = np.real(np.fft.ifftn(noise * amp))
    field = (field - field.mean()) / field.std()
    thr = np.quantile(field, 1.0 - rho)
    idx = np.argwhere(field > thr)
    offs = rng.random((len(idx), 3))
    pts = idx + offs
    return pts

def g_curve(pts, grid):
    tree = cKDTree(pts, boxsize=L)
    d, _ = tree.query(pts, k=2)
    nn = d[:, 1]
    S = np.array([np.mean(nn > g_) for g_ in grid])
    lnS = np.log(np.maximum(S, 1e-10))
    gv = -np.diff(lnS) / np.diff(np.log(grid))
    mid = np.sqrt(grid[:-1] * grid[1:])
    return mid, gv, nn, S

def find_plateau(mid, gv, w=10, tol=0.12):
    best = None
    for i in range(0, len(gv) - w + 1):
        win = gv[i:i+w]
        med = float(np.median(win))
        if med <= 0:
            continue
        spread = float(win.max() - win.min()) / med
        if spread <= tol:
            span = math.log(mid[i+w-1]) - math.log(mid[i])
            cand = (med, spread, mid[i], mid[i+w-1], span)
            if best is None or cand[4] > best[4]:
                best = cand
    return best

grid = np.logspace(math.log10(0.5), math.log10(60.0), 64)
S_LIST = [-1.0, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0]

print("== 生成与扫描（N = %d，密度 rho = %.3f） ==" % (N, RHO))
results = {}
for s_exp in S_LIST:
    pts = grf_points(s_exp)
    mid, gv, nn, S = g_curve(pts, grid)
    pl = find_plateau(mid, gv)
    msk = (mid > 1.0) & (mid < 10.0)
    med_g = float(np.median(gv[msk])) if np.any(msk) else float("nan")
    results[s_exp] = (mid, gv, nn, S, pl, med_g)
    if pl is None:
        print("  s = %.1f：点数 %6d，平均 NN = %.3f，无平台；中位 g = %.3f" % (
            s_exp, len(pts), float(np.mean(nn)), med_g))
    else:
        print("  s = %.1f：点数 %6d，平均 NN = %.3f，平台 c = %.3f（范围 [%.2f, %.2f]）；中位 g = %.3f" % (
            s_exp, len(pts), float(np.mean(nn)), pl[0], pl[2], pl[3], med_g))
print()

print("== N1 生成器核对（s = 0 对泊松） ==")
rng = np.random.default_rng(SEED + 999)
pois = rng.random((20000, 3)) * L
mid_p, gv_p, nn_p, S_p = g_curve(pois, grid)
msk = (mid_p > 1.0) & (mid_p < 10.0)
print("  泊松对照：平均 NN = %.3f，中位 g = %.3f" % (float(np.mean(nn_p)), float(np.median(gv_p[msk]))))
print("  s = 0 档：平均 NN = %.3f，中位 g = %.3f" % (
    float(np.mean(results[0.0][2])), results[0.0][5]))
print()

print("== N3/N5 平台与靶位 ==")
plats = {s: results[s][4] for s in S_LIST}
any_plat = [s for s in S_LIST if plats[s] is not None]
if any_plat:
    for s in any_plat:
        print("  s = %.1f：平台 c = %.3f（对 5/3 偏差 %.1f%%）" % (
            s, plats[s][0], (plats[s][0] - TARGET) / TARGET * 100))
else:
    print("  全部档位无平台（严格检测口径）")
print()

print("== 图 ==")
fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6), dpi=110)
fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.16, wspace=0.24)

ax = axes[0]
colors = {-1.0: "#b05a5a", 0.0: "#888888", 0.5: "#14508c", 1.0: "#2f7a2f", 2.0: "#c0392b", 3.0: "#7030a0", 4.0: "#e08a1e"}
for s_exp in S_LIST:
    mid, gv = results[s_exp][0], results[s_exp][1]
    ax.loglog(mid, gv, color=colors[s_exp], lw=1.5, alpha=0.9, label="s = %.1f" % s_exp)
ax.axhline(TARGET, color="#000000", ls="--", lw=1.2)
ax.annotate("5/3", (0.6, TARGET * 1.12), fontsize=10)
ax.set_xlim(0.5, 60)
ax.set_ylim(0.05, 40)
ax.set_xlabel("l", fontsize=10.5)
ax.set_ylabel("g(l)", fontsize=10.5)
ax.set_title("N2/N3 g(l) by correlation exponent s", fontsize=10.5)
ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
ax.legend(fontsize=8.5)

ax = axes[1]
for s_exp in S_LIST:
    S = results[s_exp][3]
    ax.semilogy(grid, S, color=colors[s_exp], lw=1.4, alpha=0.9, label="s = %.1f" % s_exp)
ax.set_xlim(0.5, 60)
ax.set_ylim(1e-4, 1.05)
ax.set_xlabel("l", fontsize=10.5)
ax.set_ylabel("S(l)", fontsize=10.5)
ax.set_title("N2 survival functions", fontsize=10.5)
ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
ax.legend(fontsize=8.5)

fig.savefig(IMG)
print("  已保存：", IMG)
