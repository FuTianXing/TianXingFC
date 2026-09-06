# -*- coding: utf-8 -*-
"""
疤的数值实现方案：第一步探路（单格点状态枚举 + 状态图）
相位量子化：φ^a = 2π k_a / N（k_a ∈ Z_N），三元组状态 (k_x, k_y, k_z)
绑定能（格点内，κ_φ = 1）：E = Σ_{a<b} [1 - cos(minimage(φ_a - φ_b))]
真空态：三个 k 相等（δ = 0，绑定能 0）
状态图：单步边 = 改变一个 k 一个单位（mod N）
探路目标：状态图连通性；按候选绕数标签分组后的能量障碍结构。
结果只打印终端，不写任何文档。
"""
import numpy as np
from itertools import product
from collections import deque

def minimage(d):
    return (d + np.pi) % (2.0 * np.pi) - np.pi

def binding_energy(k, N, kappa=1.0):
    ph = 2.0 * np.pi * np.array(k) / N
    E = 0.0
    for a, b in [(0, 1), (1, 2), (2, 0)]:
        E += 1.0 - np.cos(minimage(ph[a] - ph[b]))
    return kappa * E

def state_graph(N):
    states = list(product(range(N), repeat=3))
    idx = {s: i for i, s in enumerate(states)}
    adj = [[] for _ in states]
    for s in states:
        i = idx[s]
        for a in range(3):
            for dk in (+1, -1):
                t = list(s)
                t[a] = (t[a] + dk) % N
                t = tuple(t)
                adj[i].append(idx[t])
    return states, adj

def components(states, adj):
    comp = [-1] * len(states)
    ncomp = 0
    for i in range(len(states)):
        if comp[i] >= 0:
            continue
        comp[i] = ncomp
        q = deque([i])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if comp[v] < 0:
                    comp[v] = ncomp
                    q.append(v)
        ncomp += 1
    return comp, ncomp

def saddle_between(states, adj, E, groupA, groupB):
    """两组状态间的最短路径鞍点能量（Dijkstra 变体：最小化路径上的最大能量）。"""
    import heapq
    INF = float('inf')
    Ns = len(states)
    dist = [INF] * Ns
    pq = []
    for i in groupA:
        dist[i] = E[i]
        heapq.heappush(pq, (E[i], i))
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v in adj[u]:
            nd = max(d, E[v])
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return min(dist[i] for i in groupB)

for N in (4, 6, 8):
    print('=' * 64)
    print(f'N = {N}（状态数 {N**3}）')
    states, adj = state_graph(N)
    E = np.array([binding_energy(s, N) for s in states])
    comp, ncomp = components(states, adj)
    print(f'连通分量数: {ncomp}（{"全连通" if ncomp == 1 else "存在分隔!"}）')
    vac = [i for i, s in enumerate(states) if len(set(s)) == 1]
    print(f'真空态（三 k 相等）: {len(vac)} 个, 绑定能 = {E[vac[0]]:.4f}')
    # 能量分布
    print(f'绑定能范围: [{E.min():.4f}, {E.max():.4f}]')
    # 候选标签 A：相对差环绕（k_b - k_a mod N 的环绕数 = 每个差的"方向"）
    # 候选标签 B：跨度的圈数（max k - min k）
    labA = {}   # 相对差符号组合
    labB = {}   # 跨度
    for i, s in enumerate(states):
        d1 = (s[1] - s[0]) % N
        d2 = (s[2] - s[1]) % N
        labA[i] = (d1, d2)
        labB[i] = max(s) - min(s)
    # 按标签 B（跨度）分组看障碍
    print('\n按跨度（max k - min k）分组:')
    for m in sorted(set(labB.values())):
        ids = [i for i in range(len(states)) if labB[i] == m]
        if not ids:
            continue
        print(f'  跨度 {m}: {len(ids)} 态, 绑定能 {E[ids].min():.3f} - {E[ids].max():.3f}, '
              f'含真空组? {"是" if m == 0 else "否"}')
    # 跨度组间鞍点（m=0 组 vs m>=1 组）
    g0 = [i for i in range(len(states)) if labB[i] == 0]
    g1 = [i for i in range(len(states)) if labB[i] >= 1]
    if g0 and g1:
        sad = saddle_between(states, adj, E, g0, g1)
        print(f'跨度 0 组 → 跨度 ≥1 组的路径最大能量（鞍点）: {sad:.4f}（κ_φ = 1）')
    # 真空态 → 非真空态的鞍点
    sad_v = saddle_between(states, adj, E, vac, [i for i in range(len(states)) if i not in vac])
    print(f'真空 → 非真空的路径最大能量（鞍点）: {sad_v:.4f}')
print('\n探路完成（仅打印）。')
