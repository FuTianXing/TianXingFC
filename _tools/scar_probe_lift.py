# -*- coding: utf-8 -*-
"""
疤的数值实现方案：选项 1 提升变量版单格点
核心问题：多圈锁定（δ = 2πm，m 整数圈）能否承载整数绕数并受保护？
三实验：
  1A  cos 势下多圈锁定绑定能（预期退化：cos(2πm)=1，扭转与真空同能量）
  1B  非周期势 V = κ_φ(1-cos δ) + λ(δ/2π)² 的绑定能与解缠障碍（连续 + 离散 N 步）
  1C  提升变量状态图（φ = 2π(m + k/N)）：跨绕数组的鞍点
结果只打印终端，不写任何文档。
"""
import numpy as np
from collections import deque
import heapq

KAPPA = 1.0

print("=" * 66)
print("选项 1：提升变量版单格点（多圈锁定）")
print("=" * 66)

# ---------------- 实验 1A：cos 势绑定能 vs 圈数 m ----------------
print("\n[1A] cos 势 V = κ_φ(1-cos δ)：多圈锁定 δ = 2πm 的绑定能")
print("  m        δ/2π      V(δ)（κ_φ=1）")
for m in range(0, 5):
    d = 2 * np.pi * m
    print(f"  {m}        {m}.0        {1 - np.cos(d):.4f}")
print("  结论预期：cos(2πm) = 1 恒成立，扭转态与真空同能量（无绑定能）")

# ---------------- 实验 1B：非周期势的绑定能与解缠障碍 ----------------
print("\n[1B] 非周期势 V = κ_φ(1-cos δ) + λ(δ/2π)²")
for lam in (0.01, 0.05, 0.1):
    print(f"\n  λ = {lam}")
    print("  m  绑定能(阱底)  解缠障碍(连续)  解缠障碍(N=4)  (N=8)  (N=16)  (N=64)")
    for m in range(1, 5):
        Ebind = lam * m ** 2
        # 连续路径障碍：δ ∈ [2π(m-1), 2πm] 上 V 的最大值（cos 项在奇数 π 处 = 2κ_φ，
        # λ 项在端点最大 = λm²；细扫描取 max）
        ds = np.linspace(2 * np.pi * (m - 1), 2 * np.pi * m, 20001)
        V = KAPPA * (1 - np.cos(ds)) + lam * (ds / (2 * np.pi)) ** 2
        obs_cont = V.max()
        row = f"  {m}  {Ebind:8.4f}   {obs_cont:8.4f}"
        # 离散 N 步：δ 取 2π(m - j/N)，j = 0..N（从 m 圈退到 m-1 圈的 N 步）
        for N in (4, 8, 16, 64):
            js = np.arange(N + 1)
            dsn = 2 * np.pi * (m - js / N)
            Vn = KAPPA * (1 - np.cos(dsn)) + lam * (dsn / (2 * np.pi)) ** 2
            row += f"   {Vn.max():8.4f}"
        print(row)

# ---------------- 实验 1C：提升变量状态图 ----------------
print("\n[1C] 提升变量状态图（φ = 2π(m + k/N)，m 截断 [-M, M]，k ∈ Z_N）")
print("  绕数标签：相对提升差 Δ = (φ_b - φ_a)/2π 的整数部分")
for N in (4, 8):
    M = 2
    # 状态：(m_x, m_y, m_z, k_x, k_y, k_z) —— 绕数 = 任意两个元相子的提升差？三元组
    # 简化为二元组 (m_a, k_a), (m_b, k_b)，绕数 = round((m_b-k_b... 差)
    # 更直接：二元组 φ_a, φ_b，绕数 W = round((φ_b - φ_a)/2π)（提升差圈数）
    states = [(ma, ka, mb, kb) for ma in range(-M, M + 1) for ka in range(N)
              for mb in range(-M, M + 1) for kb in range(N)]
    idx = {s: i for i, s in enumerate(states)}
    adj = [[] for _ in states]
    for s in states:
        i = idx[s]
        for which in (0, 1):
            ma, ka, mb, kb = s
            # 分数步 k ± 1（wrap 时 m ± 1）
            for dk in (+1, -1):
                if which == 0:
                    ka2 = (ka + dk) % N
                    ma2 = ma + (1 if ka + dk >= N else 0) - (1 if ka + dk < 0 else 0)
                    t = (ma2, ka2, mb, kb)
                else:
                    kb2 = (kb + dk) % N
                    mb2 = mb + (1 if kb + dk >= N else 0) - (1 if kb + dk < 0 else 0)
                    t = (ma, ka, mb2, kb2)
                if t in idx:
                    adj[i].append(idx[t])
    # 绑定能（非周期势 λ = 0.05，只用相对差 δ = φ_b - φ_a）
    lam = 0.05
    E = []
    for (ma, ka, mb, kb) in states:
        d = 2 * np.pi * ((mb - ma) + (kb - ka) / N)
        E.append(KAPPA * (1 - np.cos(d)) + lam * (d / (2 * np.pi)) ** 2)
    E = np.array(E)
    # 绕数标签：round(d/2π)
    Wlab = np.array([round((mb - ma) + (kb - ka) / N) for (ma, ka, mb, kb) in states])
    # 连通性
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
    print(f"\n  N = {N}：状态数 {len(states)}，连通分量 {ncomp}"
          f"（{'全连通' if ncomp == 1 else '有分隔'}）")
    # 跨绕数组鞍点（W=0 组 vs W>=1 组）
    def saddle(gA, gB):
        INF = float('inf')
        dist = [INF] * len(states)
        pq = []
        for i in gA:
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
        return min(dist[i] for i in gB)
    g0 = [i for i in range(len(states)) if Wlab[i] == 0]
    g1 = [i for i in range(len(states)) if Wlab[i] >= 1]
    g2 = [i for i in range(len(states)) if Wlab[i] >= 2]
    s01 = saddle(g0, g1)
    s12 = saddle([i for i in range(len(states)) if Wlab[i] == 1], g2)
    print(f"  绕数标签分布: { {w: int((Wlab == w).sum()) for w in sorted(set(Wlab))} }")
    print(f"  绑定能范围: {E.min():.4f} - {E.max():.4f}")
    print(f"  W=0 组 → W≥1 组鞍点: {s01:.4f}")
    print(f"  W=1 组 → W≥2 组鞍点: {s12:.4f}")
    print(f"  （对比：光滑场翻页票价 0.19κ_∥；连续极限单键反相代价 2κ_φ）")
print("\n选项 1 完成（仅打印）。")
