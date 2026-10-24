# -*- coding: utf-8 -*-
"""有效维数 2.4 的几何实现：分形格点波尾的维数扫描（第 428 条，候选丁段三）。

判据：
  Q1 规则 2D 对照（基线）
  Q2 分形几何（Sierpinski 垫片 d_f = log3/log2、地毯 d_f = log8/log3、海绵 d_f = log20/log3）
     上「相邻点响应差」（力型信号）的尾指数
  Q3 跨窗稳定性
  Q4 「指数 vs 维数」对照与 2.4 位置的反推

方法：图拉普拉斯（4 或 6 邻接）稠密对角化，脉冲响应 u(t) = sum_i v_i[s]v_i[o] cos(w_i t)；
      力型信号取相邻两点的响应差；尾用对数分段 RMS 拟合。

运行：python 验证脚本/fractal_tail_scan.py
"""
import math
import os

import numpy as np
from scipy.linalg import eigh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "images", "fractal_tail_scan.png")

def graph_from_mask(mask):
    pts = np.argwhere(mask)
    H, W = mask.shape
    idx = -np.ones((H, W), dtype=int)
    for k, (i, j) in enumerate(pts):
        idx[i, j] = k
    N = len(pts)
    edges = []
    for (i, j) in pts:
        for di, dj in ((1, 0), (0, 1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W and mask[ni, nj]:
                edges.append((idx[i, j], idx[ni, nj]))
    L = np.zeros((N, N))
    for a, b in edges:
        L[a, a] += 1; L[b, b] += 1
        L[a, b] -= 1; L[b, a] -= 1
    return pts, L, edges

def sierpinski_mask(n):
    size = 2 ** n
    ii = np.arange(size)[:, None]
    jj = np.arange(size)[None, :]
    return (ii & jj) == 0

def carpet_mask(n):
    size = 3 ** n
    ii = np.arange(size)[:, None] * np.ones((1, size), dtype=int)
    jj = np.ones((size, 1), dtype=int) * np.arange(size)[None, :]
    keep = np.ones((size, size), bool)
    a, b = ii.copy(), jj.copy()
    for _ in range(n):
        center = (a % 3 == 1) & (b % 3 == 1)
        keep &= ~center
        a //= 3; b //= 3
    return keep

def tail_of_graph(mask, name, t_grid):
    pts, L, edges = graph_from_mask(mask)
    N = len(pts)
    w2, V = eigh(L)
    w2 = np.maximum(w2, 0.0)
    w = np.sqrt(w2)
    center = np.array([mask.shape[0] / 2.0, mask.shape[1] / 2.0])
    d2 = ((pts - center) ** 2).sum(axis=1)
    src = int(np.argmin(d2))
    # 找 src 的一个邻居作为 obs
    obs = None
    for a, b in edges:
        if a == src:
            obs = b; break
        if b == src:
            obs = a; break
    if obs is None:
        return None
    c = V[src, :] * (V[src, :] - V[obs, :])
    vals = []
    for t in t_grid:
        vals.append(float(np.sum(c * np.cos(w * t))))
    return np.array(vals), N

def env(ts, vals, edges=None, win=5):
    vs = np.array(vals)
    ts = np.array(ts)
    mids = []; rms = []
    for i in range(0, len(ts) - win + 1):
        seg = vs[i:i + win]
        mids.append(float(np.exp(np.mean(np.log(ts[i:i + win])))))
        rms.append(float(np.sqrt(np.mean(seg ** 2))))
    return np.array(mids), np.array(rms)

def fit_slope(mids, rms, lo, hi):
    msk = (mids >= lo) & (mids <= hi) & (rms > 0)
    if np.sum(msk) < 5:
        return float("nan")
    return float(np.polyfit(np.log(mids[msk]), np.log(rms[msk]), 1)[0])

geoms = [
    ("regular 2D 64x64", np.ones((64, 64), bool), None),
    ("Sierpinski gasket n=8", sierpinski_mask(8), math.log(3) / math.log(2)),
    ("Sierpinski carpet n=4", carpet_mask(4), math.log(8) / math.log(3)),
]

print("== Q1/Q2 分形波尾扫描（力型信号：相邻响应差） ==")
t_grid = np.logspace(math.log10(2.0), math.log10(400.0), 45)
results = []
for name, mask, df in geoms:
    out = tail_of_graph(mask, name, t_grid)
    if out is None:
        print("  %s：无邻居，跳过" % name)
        continue
    vals, N = out
    edges_env = np.logspace(math.log10(3.0), math.log10(400.0), 10)
    mids, rms = env(t_grid, vals)
    s1 = fit_slope(mids, rms, 20, 150)
    s2 = fit_slope(mids, rms, 40, 300)
    results.append((name, N, df, s1, s2))
    print("  %-22s N = %5d，d_f = %s：尾指数（窗[20,150]）%.3f、（窗[40,300]）%.3f" % (
        name, N, ("%.3f" % df) if df else "2.000", s1, s2))

print()
print("== Q4 指数与维数对照 ==")
for name, N, df, s1, s2 in results:
    if df is None:
        print("  %s：对照" % name)
    else:
        print("  %s：尾指数 %.3f 对 d_f/2 = %.3f、d_f = %.3f" % (name, s1, df/2.0, df))
print("  目标：尾指数 1.2（对应 γF = 1.2、γv = 0.8）；若按 尾指数 = d_x/2 反推，需 d_x = 2.4")
print()

print("== 图 ==")
fig, ax = plt.subplots(figsize=(7.8, 4.8), dpi=110)
colors = {"regular 2D 64x64": "#14508c", "Sierpinski gasket n=8": "#2f7a2f", "Sierpinski carpet n=4": "#c0392b"}
for name, mask, df in geoms:
    out = tail_of_graph(mask, name, t_grid)
    if out is None:
        continue
    vals, N = out
    edges_env = np.logspace(math.log10(3.0), math.log10(400.0), 10)
    mids, rms = env(t_grid, vals)
    ax.loglog(mids, rms, "o-", color=colors[name], lw=1.4, markersize=4, label=name)
ax.set_xlabel("t", fontsize=10.5)
ax.set_ylabel("envelope of edge signal", fontsize=10.5)
ax.set_title("fractal lattice wave tails (edge signal)", fontsize=10.5)
ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
ax.legend(fontsize=8.5)
fig.savefig(IMG)
print("  已保存：", IMG)
