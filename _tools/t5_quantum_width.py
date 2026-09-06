# -*- coding: utf-8 -*-
"""
T5 量子版宽度：Z_2 L=3 toric code 第一激发（涡旋/电荷）的波函数空间分布
若激发波函数在"通量缺陷位置"上的分布局域（单点）→ 点状粒子（宽度 0）
若扩散 → 有限宽度（对照 ℓ >= 2 色散窗口）
结果只打印终端，不写任何文档。
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

L = 3
NE = 2 * L * L
D = 2 ** NE

def eid(t, i, j):
    return i * L + j if t == 'h' else L * L + i * L + j

stars = [[eid('h', i, j), eid('h', i, (j - 1) % L),
          eid('v', i, j), eid('v', (i - 1) % L, j)] for i in range(L) for j in range(L)]
plaqs = [[eid('h', i, j), eid('h', (i + 1) % L, j),
          eid('v', i, j), eid('v', i, (j + 1) % L)] for i in range(L) for j in range(L)]

rows, cols, vals = [], [], []
for pl in plaqs:
    for s in range(D):
        p = 1.0 if sum((s >> e) & 1 for e in pl) % 2 == 0 else -1.0
        rows.append(s); cols.append(s); vals.append(-p)
for st in stars:
    for s in range(D):
        t = s
        for e in st:
            t ^= (1 << e)
        rows.append(s); cols.append(t); vals.append(-1.0)
H = coo_matrix((vals, (rows, cols)), shape=(D, D)).tocsr()

w, v = eigsh(H, k=8, which='SA')
E0 = w[0]
print(f'E0 = {E0:.6f}, 第一激发差 = {w[4]-E0:.6f}, 第二激发差 = {w[5]-E0:.6f}')

def flux_pattern(s):
    """返回通量缺陷（B_p = -1）的 plaquette 位置列表。"""
    bad = []
    for pi, pl in enumerate(plaqs):
        p = 1.0 if sum((s >> e) & 1 for e in pl) % 2 == 0 else -1.0
        if p < 0:
            bad.append(pi)
    return bad

def spread(positions):
    """位置集合的空间散布（直径）。"""
    if len(positions) <= 1:
        return 0.0
    ys = [p // L for p in positions]
    xs = [p % L for p in positions]
    r = 0.0
    for a in range(len(positions)):
        for b in range(len(positions)):
            dy = min(abs(ys[a]-ys[b]), L-abs(ys[a]-ys[b]))
            dx = min(abs(xs[a]-xs[b]), L-abs(xs[a]-xs[b]))
            r = max(r, np.sqrt(dx*dx + dy*dy))
    return r

# 激发态：第一激发（w[1]）的通量缺陷分布
for k in (4, 5):
    psi = v[:, k]
    amp2 = np.abs(psi) ** 2
    # 按通量缺陷模式聚合
    modes = {}
    for s in range(D):
        if amp2[s] < 1e-12:
            continue
        pat = tuple(flux_pattern(s))
        modes[pat] = modes.get(pat, 0.0) + amp2[s]
    print(f'\n激发 {k}（E-E0 = {w[k]-E0:.6f}）通量缺陷模式:')
    for pat, wgt in sorted(modes.items(), key=lambda x: -x[1])[:5]:
        n = len(pat)
        sp = spread(pat)
        print(f'  模式 {pat}（缺陷数 {n}，散布 {sp:.2f} 格距）权重 {wgt:.4f}')
    # 缺陷数期望与平均散布
    exp_n = sum(len(p) * w for p, w in modes.items())
    exp_sp = sum(spread(list(p)) * w for p, w in modes.items())
    print(f'  缺陷数期望 = {exp_n:.3f}，平均散布 = {exp_sp:.3f} 格距')
    print(f"  {'点状粒子（宽度≈0，toric code 无空间延伸）' if exp_sp < 0.5 else f'有限宽度 {exp_sp:.2f}（对照 ℓ≥2）'}")

print('\nT5 量子版完成（仅打印）。')
