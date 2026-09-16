# -*- coding: utf-8 -*-
r"""
minimal_verification_state_graph.py

[离散化保护的否定轮] 相位量子化下的状态图全连通与多圈锁定退化

定位
----
《绕数承载层》位形侧四重否定的第三、第四支的固化脚本（原探路脚本 _tools/scar_probe_single.py 与
_tools/scar_probe_lift.py，2026-09-06，未固化）。两部分：
  A 相位量子化模型（单格点三元组，k_a 属于 Z_N）：状态图（单步 = 改一个 k 一个单位）的连通性、
     真空态、按候选绕数标签（跨度）分组后的能量结构，以及从真空逃逸的最小代价；
  B 提升变量版（多圈锁定 varphi = 2 pi (m + k/N)）：cos 势下多圈锁定是否退化、非周期势下的阱底
     与解缠障碍、提升变量状态图跨绕数组的鞍点。
判据全为精确组合与确定性数值，本脚本不改动基准文件。

判据（先立判据，再实现；预期写错时修正判据并记录教训）
----
SG-1 状态图全连通：N = 4, 6, 8 的连通分量数 = 1（构造性保护在离散化下不成立）。
SG-2 真空：三 k 相等的 N 个态绑定能 = 0（kappa_phi = 1）。
SG-3 逃逸代价：真空到非真空的路径最大能量（鞍点）= 2(1 - cos 2 pi / N)（逐点：N = 4, 6, 8）。
SG-4 多圈锁定退化：cos 势下 delta = 2 pi m 的阱底绑定能 = 0（扭转与真空同能量）。
SG-5 非周期势 V = kappa(1-cos delta) + lambda (delta / 2 pi)^2：阱底 = lambda m^2 逐点。
SG-6 提升变量状态图：跨绕数组（W = 0 到 W >= 1）鞍点与离散化细度 N 无关（N = 4, 8 同值）。

运行
----
python 验证脚本/minimal_verification_state_graph.py
确定性（纯组合 + 无随机数），运行 < 30 秒。纯打印，不出图。
"""
import time
import heapq
from itertools import product
from collections import deque

import numpy as np

T0 = time.time()
CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print(f"  [{tag}] {'PASS' if ok else 'FAIL'}  {detail}")


def minimage(d):
    return (d + np.pi) % (2.0 * np.pi) - np.pi


def binding_energy(triple, N, kappa=1.0):
    ph = 2.0 * np.pi * np.array(triple) / N
    E = 0.0
    for a, b in [(0, 1), (1, 2), (2, 0)]:
        E += 1.0 - np.cos(minimage(ph[a] - ph[b]))
    return kappa * E


def build_state_graph(N):
    states = list(product(range(N), repeat=3))
    idx = {s: i for i, s in enumerate(states)}
    adj = [[] for _ in states]
    for s in states:
        i = idx[s]
        for a in range(3):
            for dk in (+1, -1):
                t = list(s)
                t[a] = (t[a] + dk) % N
                adj[i].append(idx[tuple(t)])
    return states, adj


def components(nstates, adj):
    comp = [-1] * nstates
    ncomp = 0
    for i in range(nstates):
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
    return ncomp


def saddle(nstates, adj, E, groupA, groupB):
    """两组状态间使路径最大能量最小的鞍点（最小最大路径）。"""
    INF = float("inf")
    dist = [INF] * nstates
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


print("=" * 72)
print("[离散化保护的否定轮] 相位量子化的状态图与多圈锁定")
print("=" * 72)

# ---------------- A 相位量子化单格点三元组 ----------------
for N in (4, 6, 8):
    states, adj = build_state_graph(N)
    E = np.array([binding_energy(s, N) for s in states])
    ncomp = components(len(states), adj)
    vac = [i for i, s in enumerate(states) if len(set(s)) == 1]
    nonvac = [i for i in range(len(states)) if i not in vac]
    sad = saddle(len(states), adj, E, vac, nonvac)
    pred = 2.0 * (1.0 - np.cos(2.0 * np.pi / N))
    print(f"\nN = {N}（状态数 {N ** 3}）：连通分量 {ncomp}；真空态 {len(vac)} 个（E = {E[vac[0]]:.4f}）")
    print(f"    逃离代价（真空 → 非真空鞍点）= {sad:.6f}；2(1-cos 2pi/N) = {pred:.6f}")
    check(f"SG-1/N{N}", ncomp == 1, f"状态图全连通（连通分量 {ncomp}）")
    check(f"SG-2/N{N}", abs(E[vac[0]]) < 1e-12 and len(vac) == N,
          f"真空 {len(vac)} 个、绑定能 {E[vac[0]]:.1e}")
    check(f"SG-3/N{N}", abs(sad - pred) < 1e-9,
          f"逃逸代价 {sad:.6f} = 2(1-cos 2pi/N) {pred:.6f}")

# ---------------- B 提升变量（多圈锁定） ----------------
print("\n" + "-" * 72)
print("提升变量版：多圈锁定 varphi = 2 pi (m + k/N)，delta = 2 pi m")
print("-" * 72)
for lam in (0.05, 0.1):
    print(f"\n  lambda = {lam}")
    print("    m   阱底(cos 势)   阱底(非周期)   lambda m^2   解缠障碍   2k+lambda(m-1/2)^2")
    for m in (1, 2, 3, 4):
        delta = 2.0 * np.pi * m
        well_cos = 1.0 - np.cos(delta)
        well_np = well_cos + lam * (delta / (2.0 * np.pi)) ** 2
        barrier = 2.0 + lam * (m - 0.5) ** 2
        print(f"   {m:2d}   {well_cos:10.6f}   {well_np:12.6f}   {lam * m * m:10.6f}"
              f"   {barrier:9.6f}   {barrier:9.6f}")
        check(f"SG-4/lam{lam}/m{m}", abs(well_cos) < 1e-12,
              f"cos 势下 m = {m} 的阱底 = {well_cos:.1e}（与真空同能量）")
        check(f"SG-5/lam{lam}/m{m}", abs(well_np - lam * m * m) < 1e-12,
              f"非周期势阱底 {well_np:.6f} = lambda m^2 {lam * m * m:.6f}")

# 提升变量状态图：两体相对差 (m, k)，边 = m ± 1 或 k ± 1
print("\n" + "-" * 72)
print("提升变量状态图：跨绕数组鞍点（lambda = 0.05）")
print("-" * 72)
LAM = 0.05
M_TRUNC = 6
for N in (4, 8):
    states = [(m, k) for m in range(-M_TRUNC, M_TRUNC + 1) for k in range(N)]
    idx = {s: i for i, s in enumerate(states)}
    adj = [[] for _ in states]
    # 有效单步只改 k 一个单位，越过 k 的端点时 m 随之加减（相位连续，整圈不作单步）
    for (m, k) in states:
        i = idx[(m, k)]
        for dk in (+1, -1):
            k2 = k + dk
            m2 = m
            if k2 < 0:
                k2, m2 = k2 + N, m2 - 1
            if k2 >= N:
                k2, m2 = k2 - N, m2 + 1
            if -M_TRUNC <= m2 <= M_TRUNC:
                adj[i].append(idx[(m2, k2)])
    E = np.array([1.0 - np.cos(2.0 * np.pi * (m + k / N))
                  + LAM * (m + k / N) ** 2 for (m, k) in states])
    ncomp = components(len(states), adj)
    Wlab = np.array([round(m + k / N) for (m, k) in states])
    g0 = [i for i in range(len(states)) if Wlab[i] == 0]
    g1 = [i for i in range(len(states)) if Wlab[i] >= 1]
    sad = saddle(len(states), adj, E, g0, g1)
    print(f"  N = {N}：状态数 {len(states)}，连通分量 {ncomp}，绕数标签覆盖 "
          f"[{Wlab.min()}, {Wlab.max()}]，跨绕数组鞍点 = {sad:.6f}"
          f"（预期 2 + lambda/4 = {2 + LAM / 4:.6f}）")
    check(f"SG-6/N{N}", ncomp == 1 and abs(sad - (2.0 + LAM / 4.0)) < 1e-9,
          f"全连通 {ncomp == 1}；鞍点 {sad:.6f} = 2 + lambda/4")

print("\n判据汇总：{}/{} 通过".format(sum(1 for _, ok in CHECKS if ok), len(CHECKS)))
for tag, ok in CHECKS:
    if not ok:
        print(f"  未过：{tag}")
print(f"\n[离散化保护的否定轮] 完成；总运行 {time.time() - T0:.1f} s。")
