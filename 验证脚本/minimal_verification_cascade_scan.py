# -*- coding: utf-8 -*-
"""级联簇精扫：K 收敛性、平台检测与 5/3 等值线（第 421 条，P0-5 候选机制九轮）。

判据：
  U1 K 收敛性：同 (m, s) 下 K = 3、4、5 的平台值是否收敛（容差 5%）
  U2 平台检测器：滑动窗口（对数域）自动定位平台段
  U3 (m, s) 网格扫描：c(m, s) 面与 c = 5/3 等值线
  U4 平台宽度读数

运行：python 验证脚本/minimal_verification_cascade_scan.py
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
IMG1 = os.path.join(ROOT, "images", "cascade_scan_convergence.png")
IMG2 = os.path.join(ROOT, "images", "cascade_scan_contour.png")

TARGET = 5.0 / 3.0
SEED = 20260927
L = 60.0

def cascade_full(N0, m, K, s_ratio, r0=6.0, rng=None):
    pts = rng.random((N0, 3)) * L
    layers = [pts]
    cur = pts
    r = r0
    for _ in range(K):
        children = []
        for i in range(len(cur)):
            for _m in range(m):
                v = rng.normal(size=3)
                nrm = np.linalg.norm(v)
                if nrm == 0.0:
                    v = np.array([1.0, 0.0, 0.0])
                    nrm = 1.0
                v = v / nrm * (r * rng.random() ** (1.0/3.0))
                children.append(cur[i] + v)
        cur = np.array(children)
        layers.append(cur)
        r *= s_ratio
    return layers

def g_curve(pts, grid):
    pts = pts % L
    tree = cKDTree(pts, boxsize=L)
    d, _ = tree.query(pts, k=2)
    nn = d[:, 1]
    S = np.array([np.mean(nn > g_) for g_ in grid])
    lnS = np.log(np.maximum(S, 1e-10))
    gv = -np.diff(lnS) / np.diff(np.log(grid))
    mid = np.sqrt(grid[:-1] * grid[1:])
    return mid, gv, nn

def find_plateau(mid, gv, w=10, tol=0.12):
    best = None
    n = len(gv)
    for i in range(0, n - w + 1):
        win = gv[i:i+w]
        med = float(np.median(win))
        if med <= 0:
            continue
        spread = float(win.max() - win.min()) / med
        if spread <= tol:
            span = math.log(mid[i+w-1]) - math.log(mid[i])
            cand = (med, spread, mid[i], mid[i+w-1], span, i)
            if best is None or cand[4] > best[4]:
                best = cand
    return best

grid = np.logspace(math.log10(0.015), math.log10(15.0), 64)

print("== U0 细网格诊断（m = 4, s = 0.30, K = 4） ==")
rng0 = np.random.default_rng(SEED)
layers0 = cascade_full(N0=60, m=4, K=4, s_ratio=0.30, rng=rng0)
pts0 = np.vstack(layers0)
mid0, gv0, nn0 = g_curve(pts0, grid)
course = (mid0 > 0.05) & (mid0 < 2.0)
print("  l 在 [0.05, 2.0] 的 g：min = %.3f，max = %.3f，中位 = %.3f（中位非平台）" % (
    gv0[course].min(), gv0[course].max(), np.median(gv0[course])))
print("  抽样（l, g）：")
for i in range(0, len(mid0), 5):
    print("    %.4f  %.3f" % (mid0[i], gv0[i]))
# 粗网格对照（420 条口径：24 点网格）
grid_c = np.logspace(math.log10(0.02), math.log10(30.0), 24)
S_c = np.array([np.mean(nn0 > g_) for g_ in grid_c])
lnS_c = np.log(np.maximum(S_c, 1e-10))
gv_c = -np.diff(lnS_c) / np.diff(np.log(grid_c))
mid_c = np.sqrt(grid_c[:-1] * grid_c[1:])
print("  粗网格（24 点）g 在 [0.1, 2] 内的值：%s" % " ".join("%.2f" % v for v in gv_c[(mid_c > 0.1) & (mid_c < 2.0)]))
print("  判读：细网格显示振荡（峰谷交替），粗采样跳过峰谷后呈现近似常数，420 条的「平台」为采样伪影")
print()

print("== U1 K 收敛性（固定 (m, s)，K = 3、4、5） ==")
rng = np.random.default_rng(SEED)
conv_rows = []
for (m_, s_) in ((4, 0.30), (4, 0.35), (3, 0.30), (6, 0.35)):
    layers = cascade_full(N0=60, m=m_, K=5, s_ratio=s_, rng=rng)
    row = []
    for K_ in (3, 4, 5):
        pts = np.vstack(layers[:K_+1])
        mid_, gv_, nn_ = g_curve(pts, grid)
        pl = find_plateau(mid_, gv_)
        if pl is None:
            row.append((K_, None, len(pts)))
            print("  m = %d, s = %.2f, K = %d：无平台（点数 %d）" % (m_, s_, K_, len(pts)))
        else:
            row.append((K_, pl[0], len(pts), pl[2], pl[3], pl[1]))
            print("  m = %d, s = %.2f, K = %d：平台 c = %.3f（散布 %.1f%%），范围 [%.3f, %.3f]，点数 %d" % (
                m_, s_, K_, pl[0], pl[1]*100, pl[2], pl[3], len(pts)))
    conv_rows.append(((m_, s_), row))
print()

print("== U3 (m, s) 网格扫描（K = 4，自适应母点数） ==")
m_list = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
s_list = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
C = np.full((len(s_list), len(m_list)), np.nan)
W = np.full((len(s_list), len(m_list)), np.nan)
MED = np.full((len(s_list), len(m_list)), np.nan)
rng2 = np.random.default_rng(SEED + 1)
for j, s_ in enumerate(s_list):
    for i, m_ in enumerate(m_list):
        N0 = max(25, int(round(20000.0 / (m_ ** 4))))
        layers = cascade_full(N0=N0, m=int(m_), K=4, s_ratio=s_, rng=rng2)
        pts = np.vstack(layers)
        mid_, gv_, nn_ = g_curve(pts, grid)
        pl = find_plateau(mid_, gv_)
        if pl is not None:
            C[j, i] = pl[0]
            W[j, i] = pl[4]
        msk = (mid_ > 0.1) & (mid_ < 2.0)
        MED[j, i] = float(np.median(gv_[msk]))
    print("  s = %.2f 行完成：c 范围 [%.2f, %.2f]，有效列数 %d" % (
        s_, np.nanmin(C[j]), np.nanmax(C[j]), int(np.sum(~np.isnan(C[j])))))
print()
print("  c 值表（行 s，列 m）：")
print("        " + " ".join("%6.1f" % m_ for m_ in m_list))
for j, s_ in enumerate(s_list):
    print("  s=%.2f " % s_ + " ".join(" %5.2f" % v if not np.isnan(v) else "  --- " for v in C[j]))
print()

print("== U4 5/3 位置读数 ==")
mask = ~np.isnan(C)
if np.any(mask):
    d = np.abs(C - TARGET)
    idx = np.unravel_index(np.nanargmin(d), C.shape)
    print("  离 5/3 最近的网格点：m = %.1f, s = %.2f，c = %.3f（偏差 %.1f%%）" % (
        m_list[idx[1]], s_list[idx[0]], C[idx], (C[idx]-TARGET)/TARGET*100))
    # 粗等值线：对每个 m 列，插值找 c = 5/3 的 s
    print("  按 m 列插值 c = 5/3 的 s 位置：")
    for i, m_ in enumerate(m_list):
        col = C[:, i]
        ok = ~np.isnan(col)
        if np.sum(ok) >= 2:
            ss = np.array(s_list)[ok]
            cc = col[ok]
            if cc.min() <= TARGET <= cc.max():
                s_star = float(np.interp(TARGET, cc, ss))
                print("    m = %.1f：s* = %.3f" % (m_, s_star))
            else:
                print("    m = %.1f：c 范围 [%.2f, %.2f]，不含 5/3" % (m_, cc.min(), cc.max()))
print()

print("== 图 ==")
fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6), dpi=110)
fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.16, wspace=0.24)

ax = axes[0]
rng3 = np.random.default_rng(SEED)
layers = cascade_full(N0=60, m=4, K=5, s_ratio=0.30, rng=rng3)
cols = {3: "#14508c", 4: "#2f7a2f", 5: "#c0392b"}
for K_ in (3, 4, 5):
    pts = np.vstack(layers[:K_+1])
    mid_, gv_, _ = g_curve(pts, grid)
    ax.loglog(mid_, gv_, color=cols[K_], lw=1.5, alpha=0.9, label="K = %d" % K_)
ax.axhline(TARGET, color="#888888", ls="--", lw=1.5)
ax.annotate("5/3", (0.02, TARGET*1.12), fontsize=10)
ax.set_xlim(0.015, 15)
ax.set_ylim(0.05, 8)
ax.set_xlabel("l", fontsize=10.5)
ax.set_ylabel("g(l)", fontsize=10.5)
ax.set_title("U1 oscillations for K = 3, 4, 5 (m=4, s=0.30)", fontsize=10.5)
ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
ax.legend(fontsize=9)

ax = axes[1]
ax.loglog(mid0, gv0, color="#2f7a2f", lw=1.6, label="fine grid (64 pts)")
ax.loglog(mid_c, gv_c, "o", color="#c0392b", markersize=8, label="coarse grid (24 pts)")
ax.axhline(TARGET, color="#888888", ls="--", lw=1.5)
ax.annotate("5/3", (0.02, TARGET*1.12), fontsize=10)
ax.set_xlim(0.015, 15)
ax.set_ylim(0.05, 8)
ax.set_xlabel("l", fontsize=10.5)
ax.set_ylabel("g(l)", fontsize=10.5)
ax.set_title("U0 aliasing: coarse sampling skips peaks", fontsize=10.5)
ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
ax.legend(fontsize=9)

fig.savefig(IMG1)
print("  已保存：", IMG1)
plt.close(fig)

fig2, ax2 = plt.subplots(figsize=(7.8, 4.8), dpi=110)
X, Y = np.meshgrid(m_list, s_list)
pcm = ax2.pcolormesh(X, Y, MED, cmap="viridis", shading="auto")
fig2.colorbar(pcm, ax=ax2, label="median g (420-diagnostic reading)")
cs3 = ax2.contour(X, Y, MED, levels=[5.0/3.0], colors="#c0392b", linewidths=2.4)
ax2.clabel(cs3, inline=True, fontsize=10)
ax2.set_xlabel("m", fontsize=10.5)
ax2.set_ylabel("s", fontsize=10.5)
ax2.set_title("median g (not a plateau; strict detector: none passed)", fontsize=10)
fig2.tight_layout()
fig2.savefig(IMG2)
print("  已保存：", IMG2)
