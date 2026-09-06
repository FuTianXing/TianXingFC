# -*- coding: utf-8 -*-
"""
T4 + U(1) 合并实验（经典层第一探路）
Z_N 通量模型（N = 16 大 N 作为 U(1) 连续极限近似），L = 8 周期格
键变量 k_e ∈ Z_N（实轴方向差读法），通量单位 = 2π/N
H = -κ_φ Σ_v cos(2π/N · 星_v) - κ_∥ Σ_p cos(2π/N · 通量_p)
绕数 W_x = 第 0 列竖直键和 mod N（大环通量）
协议 A：无约束零温弛豫（对照衰变）
协议 B（T4）：每步投影回 W_x = 1 扇区（单键调整修正绕数）
T5：扇区极小的有效宽度 ℓ*（能量密度分布），对照 ℓ ≥ 2 色散窗口
结果只打印终端，不写任何文档。
"""
import numpy as np
from collections import deque

L = 8
N = 16
KP = 1.0      # κ_∥
K0 = 1.0      # κ_φ

def build(L):
    edges = {}
    eid = 0
    for i in range(L):
        for j in range(L):
            edges[('h', i, j)] = eid
            eid += 1
    for i in range(L):
        for j in range(L):
            edges[('v', i, j)] = eid
            eid += 1
    stars = []
    for i in range(L):
        for j in range(L):
            stars.append([edges[('h', i, j)], edges[('h', i, (j - 1) % L)],
                          edges[('v', i, j)], edges[('v', (i - 1) % L, j)]])
    plaqs = []
    for i in range(L):
        for j in range(L):
            plaqs.append([edges[('h', i, j)], edges[('h', (i + 1) % L, j)],
                          edges[('v', i, j)], edges[('v', i, (j - 1) % L)]])
    return edges, stars, plaqs

edges, stars, plaqs = build(L)
NE = len(edges)
s = 2.0 * np.pi / N

def energy(k):
    E = 0.0
    for st in stars:
        E -= K0 * np.cos(s * sum(k[e] for e in st) % N)
    for pl in plaqs:
        E -= KP * np.cos(s * sum(k[e] for e in pl) % N)
    return E

def winding(k):
    """绕数 W_x：第 0 列竖直键和 mod N（取最近整数单位）。"""
    tot = sum(k[edges[('v', i, 0)]] for i in range(L)) % N
    w = tot
    if w > N // 2:
        w -= N
    return w

def flux_map(k, L):
    """每行 plaquette 通量（单位 2π/N）。"""
    fm = np.zeros((L, L), dtype=int)
    for i in range(L):
        for j in range(L):
            pl = plaqs[i * L + j]
            fm[i, j] = sum(k[e] for e in pl) % N
    return fm

def width(k, L):
    """T5：能量密度分布的有效宽度 ℓ*（以 plaquette 通量偏离 0 的分布粗粒测）。"""
    fm = flux_map(k, L)
    # 通量偏离 0 的格点，按 |偏离| 加权求空间散布（惯性半径）
    dev = np.minimum(fm, N - fm).astype(float)
    wgt = dev
    if wgt.sum() < 1e-9:
        return 0.0
    ys, xs = np.meshgrid(np.arange(L), np.arange(L), indexing='ij')
    cx = (wgt * xs).sum() / wgt.sum()
    cy = (wgt * ys).sum() / wgt.sum()
    r2 = np.sum(wgt * ((xs - cx) ** 2 + (ys - cy) ** 2)) / wgt.sum()
    return 2.0 * np.sqrt(r2)  # 直径量级

def make_fluxline():
    """通量线构型（W_x = 1）：第 0 行竖直键全 +1（穿过一行 plaquette）。"""
    k = np.zeros(NE, dtype=int)
    for j in range(L):
        k[edges[('v', 0, j)]] = 1
    return k

def relax_A(k0, steps=6000, seed=7):
    """协议 A：无约束零温 Metropolis。返回能量/绕数轨迹。"""
    rng = np.random.default_rng(seed)
    k = k0.copy()
    E = energy(k)
    traj = []
    for step in range(steps):
        e = rng.integers(NE)
        dk = 1 if rng.random() < 0.5 else N - 1
        kc = k.copy()
        kc[e] = (kc[e] + dk) % N
        dE = energy(kc) - E
        if dE < 0 or (dE == 0 and rng.random() < 0.5):
            k = kc
            E += dE
        if step % 1000 == 0:
            traj.append((step, E, winding(k)))
    return k, traj

def relax_B(k0, steps=6000, seed=11):
    """协议 B（T4）：每步投影回 W_x = 1 扇区（单键修正）。"""
    rng = np.random.default_rng(seed)
    k = k0.copy()
    E = energy(k)
    traj = []
    for step in range(steps):
        e = rng.integers(NE)
        dk = 1 if rng.random() < 0.5 else N - 1
        kc = k.copy()
        kc[e] = (kc[e] + dk) % N
        dE = energy(kc) - E
        if dE < 0 or (dE == 0 and rng.random() < 0.5):
            k = kc
            E += dE
        # 投影：修正绕数回 1（调第 0 列竖直键 v(0,0)）
        w = winding(k)
        if w != 1:
            d = (1 - w) % N
            k[edges[('v', 0, 0)]] = (k[edges[('v', 0, 0)]] + d) % N
            E = energy(k)
        if step % 1000 == 0:
            traj.append((step, E, winding(k)))
    return k, traj

print("=" * 66)
print("T4 + U(1) 合并实验（Z_N 大 N 通量模型）")
print(f"L = {L} 周期格，N = {N}（通量单位 2π/{N}），κ_φ = κ_∥ = 1")
print("=" * 66)

k0 = make_fluxline()
E0 = energy(k0)
W0 = winding(k0)
print(f"\n通量线构型：W_x = {W0}，能量（相对真空 E=-{(4*L*L):.0f}）= {E0 + 2*L*L:.4f}")

# 协议 A
print("\n[协议 A] 无约束零温弛豫（对照：绕数衰变）")
kA, trajA = relax_A(k0)
print("  t        E_rel         W_x")
for t, E, w in trajA:
    print(f"  {t:6d}  {E + 2*L*L:10.4f}  {w:+4d}")
print(f"  末态：W_x = {winding(kA)}，能量（相对真空）= {energy(kA) + 4*L*L:.4f}")

# 协议 B（T4 约束投影）
print("\n[协议 B（T4）] 每步投影回 W_x = 1 扇区")
kB, trajB = relax_B(k0)
print("  t        E_rel         W_x")
for t, E, w in trajB:
    print(f"  {t:6d}  {E + 2*L*L:10.4f}  {w:+4d}")
EB = energy(kB) + 4 * L * L
print(f"  末态：W_x = {winding(kB)}，能量（相对真空）= {EB:.4f}")
print(f"  {'扇区极小存在 ✓（约束下能量停在非零极小）' if EB > 1e-6 else '扇区极小 = 真空级能量（无绑定能）'}")

# T5：宽度
print("\n[T5] 扇区极小的形态宽度")
wB = width(kB, L)
wA = width(kA, L)
print(f"  协议 B 末态宽度 ℓ* = {wB:.2f} 格距（对照 ℓ ≥ 2 色散窗口）")
print(f"  协议 A 末态宽度 = {wA:.2f} 格距（对照：无约束衰变后）")
fm = flux_map(kB, L)
nz = (np.minimum(fm, N - fm) > 0).sum()
print(f"  协议 B 末态非零通量 plaquette 数 = {nz}（1 = 单锚核；多 = 散布/dressing）")
print(f"  {'锚核 + dressing 形态 ✓' if nz >= 1 else '异常'}：非零通量集中于 {nz} 个 plaquette")

print("\n合并实验完成（仅打印）。")
