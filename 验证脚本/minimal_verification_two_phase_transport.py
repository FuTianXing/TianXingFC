# -*- coding: utf-8 -*-
"""两相混合的渗透型输运判定：Avrami 端部窗口与级联簇检查（第 420 条，P0-5 候选机制八轮）。

判据：
  T1 Avrami 端部：D ∝ phi_s^k 的有效指数 k n f(x)，窗口宽度扫描
  T2 渗透变体：D ∝ (phi_s - phi_c)^{t_P} 的重标等价
  T3 级联簇检查：等级点集的 g(l) = -dlnS/dlnl 是否接近常数（尺度不变）

运行：python 验证脚本/minimal_verification_two_phase_transport.py
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
IMG = os.path.join(ROOT, "images", "two_phase_transport.png")

TARGET = 1.0 / 5.0
SEED = 20260926

def f_avrami(x):
    # f(x) = x e^{-x} / (1 - e^{-x})；d ln phi_s / d ln x = f(x)（phi_s = 1 - e^{-x}）
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)
    small = x < 1e-8
    out[~small] = x[~small] * np.exp(-x[~small]) / (1.0 - np.exp(-x[~small]))
    return out

xs = np.logspace(-4, 2, 4000)

print("== T1 Avrami 端部（phi_s = 1 - exp(-x)，x = (t/tau)^n） ==")
print("  有效指数（对 k = 1）= n f(x)；早期平台值 = n，晚期趋于 0")
for n in (2.5, 3.0, 4.0):
    eff = n * f_avrami(xs)
    idx = np.argmin(np.abs(eff - TARGET))
    phi_at = 1.0 - math.exp(-xs[idx])
    print("  n = %.1f：指数 = 1/5 出现在 x = %.3f（phi_s = %.4f，t/tau = %.3f），此时斜率漂移 = %.3f 每 e 折" % (
        n, xs[idx], phi_at, xs[idx] ** (1.0/n), -TARGET))
print()

print("== T1c 窗口宽度扫描（指数在 [0.16, 0.24] 内） ==")
print("  k | 窗口类型 | 相对宽度（以 e 折计，ln t 宽度）| phi_s 范围")
for k in (0.05, 0.08, 0.2, 0.5, 1.0, 2.0, 5.0):
    n = 4.0
    eff = k * n * f_avrami(xs)
    inwin = (eff >= 0.16) & (eff <= 0.24)
    if not np.any(inwin):
        print("  %.2f | 无 | - | -" % k)
        continue
    # 连续段
    idxs = np.where(inwin)[0]
    segs = []
    start = idxs[0]
    for a, b in zip(idxs[:-1], idxs[1:]):
        if b != a + 1:
            segs.append((start, a))
            start = b
    segs.append((start, idxs[-1]))
    # 取最宽段
    best = max(segs, key=lambda s_: xs[s_[1]] - xs[s_[0]])
    x1, x2 = xs[best[0]], xs[best[1]]
    wid = (math.log(x2) - math.log(x1)) / n
    phi1, phi2 = 1.0 - math.exp(-x1), 1.0 - math.exp(-x2)
    if x1 < 0.1:
        tag = "早期平台侧"
    else:
        tag = "晚期瞬时侧"
    print("  %.2f | %s | %.4f | phi_s in [%.4f, %.4f]" % (k, tag, wid, phi1, phi2))
print("  判读：k 的自然档（>= 1）只给 0.01 至 0.02 e 折的晚期窗口；宽平台需要 k = 1/(5n) = 0.05（早期），无物理支撑")
print()

print("== T2 渗透变体 ==")
print("  D ∝ (phi_s - phi_c)^{t_P} 在 phi_s > phi_c 时近似 D ∝ phi_s^{t_P}（后因子缓变）")
print("  等价于 k -> t_P（输运指数，3D 约 2.0）：落入 k >= 1 的判负档，结论不变")
print()

print("== T3 级联簇检查（等级点集） ==")
rng = np.random.default_rng(SEED)
L = 60.0
def cascade_points(N0, m, K, s_ratio, r0):
    pts = rng.random((N0, 3)) * L
    all_pts = [pts]
    r = r0
    cur = pts
    for _ in range(K):
        children = []
        for i in range(len(cur)):
            for _m in range(m):
                # 子点在母点半径 r 球内（近似：半径 r/2 均匀）
                v = rng.normal(size=3)
                v = v / np.linalg.norm(v) * (r * rng.random() ** (1.0/3.0))
                children.append(cur[i] + v)
        cur = np.array(children)
        all_pts.append(cur)
        r *= s_ratio
    return np.vstack(all_pts)

pts = cascade_points(N0=100, m=4, K=4, s_ratio=0.35, r0=6.0)
pts = pts % L
tree = cKDTree(pts, boxsize=L)
d, _ = tree.query(pts, k=2)
nn = d[:, 1]
print("  点数 = %d（母 100 x 4^4 递归，半径比 0.35）" % len(pts))
grid = np.logspace(math.log10(0.02), math.log10(30.0), 24)
S = np.array([np.mean(nn > g_) for g_ in grid])
lnS = np.log(np.maximum(S, 1e-12))
gvals = -np.diff(lnS) / np.diff(np.log(grid))
mid = np.sqrt(grid[:-1] * grid[1:])
print("  l 与 g(l) = -dlnS/dlnl（对照：尺度不变要求 g ~ 1.67 常数；平稳 Poisson 的 g 随 l 上升）:")
for i in range(0, len(mid), 3):
    print("    l = %.3f：g = %.3f" % (mid[i], gvals[i]))
# 与泊松对照
nn_pois = None
pp = rng.random((len(pts), 3)) * L
tp = cKDTree(pp, boxsize=L)
dp, _ = tp.query(pp, k=2)
nn_pois = dp[:, 1]
Sp = np.array([np.mean(nn_pois > g_) for g_ in grid])
lnSp = np.log(np.maximum(Sp, 1e-12))
gp = -np.diff(lnSp) / np.diff(np.log(grid))
print("  对照泊松（同点数）：g 在 l = %.3f 处为 %.3f，l = %.3f 处为 %.3f（上升型）" % (
    mid[3], gp[3], mid[-4], gp[-4]))
print()

print("== T3b 参数粗扫（平台值 c 是否随参数移动） ==")
print("  m x s -> 平台 g（取 l 在 [0.1, 2] 的中位 g）与 ln m / ln(1/s)")
scan_rows = []
for m_ in (2, 3, 4, 6):
    for s_ in (0.25, 0.35, 0.5):
        pts_ = cascade_points(N0=100, m=m_, K=3, s_ratio=s_, r0=6.0)
        pts_ = pts_ % L
        tr = cKDTree(pts_, boxsize=L)
        dd, _ = tr.query(pts_, k=2)
        nn_ = dd[:, 1]
        S_ = np.array([np.mean(nn_ > g_) for g_ in grid])
        lnS_ = np.log(np.maximum(S_, 1e-12))
        gg = -np.diff(lnS_) / np.diff(np.log(grid))
        mask_ = (mid >= 0.1) & (mid <= 2.0)
        if np.any(mask_):
            c_ = float(np.median(gg[mask_]))
        else:
            c_ = float("nan")
        d_ratio = math.log(m_) / math.log(1.0 / s_)
        scan_rows.append((m_, s_, c_, d_ratio))
        print("  m = %d, s = %.2f：平台 g 中位 = %.3f，ln m / ln(1/s) = %.3f" % (m_, s_, c_, d_ratio))
print("  判读：平台值随参数移动则 5/3 可能在特定参数处可达；不动则需另寻机制")
print()

print("== 净判 ==")
print("  丁（D ∝ phi_s^k）判负：k 自然档给 1% 至 2% 的晚期瞬时窗口；宽平台需要 k = 1/(5n) ≈ 0.05（无物理支撑）")
print("  渗透变体同判负（重标等价）")
print("  级联簇：初步读数见上（g 是否平台）；若为上升型则平稳化，甲仍为唯一条件存活（尺度不变遇见率）")
print()

print("== 图 ==")
fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.4), dpi=110)
fig.subplots_adjust(left=0.06, right=0.99, top=0.86, bottom=0.16, wspace=0.26)

ax = axes[0]
for n, col in ((2.5, "#14508c"), (3.0, "#2f7a2f"), (4.0, "#c0392b")):
    ax.semilogx(xs, n * f_avrami(xs), color=col, lw=1.8, label="n = %.1f (k = 1)" % n)
ax.axhline(TARGET, color="#888888", ls="--", lw=1.4)
ax.annotate("target exponent 1/5", (1e-3, TARGET * 1.15), fontsize=9)
ax.set_xlim(1e-4, 1e2)
ax.set_ylim(0, 4.4)
ax.set_xlabel("x = (t/tau)^n", fontsize=10.5)
ax.set_ylabel("effective exponent", fontsize=10.5)
ax.set_title("T1 Avrami effective exponent", fontsize=10.5)
ax.grid(True, ls=":", lw=0.5, alpha=0.6)
ax.legend(fontsize=8.5)

ax = axes[1]
ks = np.logspace(math.log10(0.03), math.log10(8.0), 60)
wid_list, tag_list = [], []
n = 4.0
for k in ks:
    eff = k * n * f_avrami(xs)
    inwin = (eff >= 0.16) & (eff <= 0.24)
    if not np.any(inwin):
        wid_list.append(np.nan)
        tag_list.append(0)
        continue
    idxs = np.where(inwin)[0]
    segs = []
    start = idxs[0]
    for a, b in zip(idxs[:-1], idxs[1:]):
        if b != a + 1:
            segs.append((start, a))
            start = b
    segs.append((start, idxs[-1]))
    best = max(segs, key=lambda s_: xs[s_[1]] - xs[s_[0]])
    x1, x2 = xs[best[0]], xs[best[1]]
    wid = (math.log(x2) - math.log(x1)) / n
    wid_list.append(wid)
    tag_list.append(1 if x1 < 0.1 else 2)
wid_arr = np.array(wid_list)
ax.loglog(ks, wid_arr, color="#7030a0", lw=2.0)
ax.axhline(0.1, color="#888888", ls=":", lw=1.2)
ax.annotate("window 0.1 e-fold", (0.031, 0.1*1.15), fontsize=8.5)
ax.set_xlim(0.03, 8)
ax.set_ylim(1e-4, 1)
ax.set_xlabel("k (exponent of D ~ phi_s^k)", fontsize=10.5)
ax.set_ylabel("widest window (e-folds in t)", fontsize=10.5)
ax.set_title("T1c window width vs k (n = 4)", fontsize=10.5)
ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)

ax = axes[2]
ax.loglog(mid, gvals, "o-", color="#c0392b", lw=1.8, markersize=4, label="cascade (leveled)")
ax.loglog(mid, np.maximum(gp, 1e-3), "s-", color="#14508c", lw=1.6, markersize=4, label="Poisson (same N)")
ax.axhline(5.0/3.0, color="#2f7a2f", ls="--", lw=1.6)
ax.annotate("scale-invariant target 5/3", (0.03, 1.9), fontsize=9, color="#2f7a2f")
ax.set_xlim(0.02, 30)
ax.set_ylim(1e-2, 1e1)
ax.set_xlabel("l", fontsize=10.5)
ax.set_ylabel("g(l) = -dlnS/dlnl", fontsize=10.5)
ax.set_title("T3 cascade cluster diagnostic", fontsize=10.5)
ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
ax.legend(fontsize=8.5)

fig2, ax4 = plt.subplots(figsize=(6.2, 4.4), dpi=110)
ms_ = [r_[0] for r_ in scan_rows]; ss_ = [r_[1] for r_ in scan_rows]; cs_ = [r_[2] for r_ in scan_rows]
sc = ax4.scatter(ms_, ss_, c=cs_, s=140, cmap="viridis", vmin=1.2, vmax=2.6)
for m_, s_, c_, d_ in scan_rows:
    ax4.annotate("%.2f" % c_, (m_, s_), textcoords="offset points", xytext=(6, 4), fontsize=8.5)
ax4.axhline(1.667, color="#c0392b", ls="--", lw=1.3)
ax4.annotate("target 5/3", (2.0, 1.70), fontsize=9, color="#c0392b")
ax4.set_xlabel("m (children per level)", fontsize=10.5)
ax4.set_ylabel("s (radius ratio)", fontsize=10.5)
ax4.set_title("T3b plateau g vs (m, s)", fontsize=10.5)
fig2.colorbar(sc, ax=ax4, label="plateau g")
fig2.savefig(os.path.join(ROOT, "images", "two_phase_transport_scan.png"))
print("  扫描图已保存：images/two_phase_transport_scan.png")

fig.savefig(IMG)
print("  已保存：", IMG)
