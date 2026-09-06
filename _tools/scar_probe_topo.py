# -*- coding: utf-8 -*-
"""
纠缠侧探路：绕数 = 关系通量（toric code 风格，经典层第一探路）
键变量 k_e ∈ Z_N（关系自由度，无相位差积分约束 → 通量可非平凡）
H = -κ_φ Σ_v cos(星_v) - κ_∥ Σ_p cos(通量_p)
  星_v = 2π/N · Σ_{e∈v} k_e（顶点四边和，格点内锁定）
  通量_p = 2π/N · Σ_{e∈p} k_e（plaquette 四边和，格点间关系环）
绕数 = 大环通量（周期格中通量线闭合为环）
判据：
  C1 真空涨落票价：基态单键翻转 ΔE（vs 光滑场翻页 0.19）
  C2 通量线构型：局域零温更新下绕数保持或拉回真空
  C3 绕数翻转票价：通量线消失路径的能量剖面（鞍点）
结果只打印终端，不写任何文档。
"""
import numpy as np
from itertools import product

def build(L):
    """2D 周期格：顶点 (i,j)，边水平 h[i,j]（(i,j)-(i,j+1)）、竖直 v[i,j]（(i,j)-(i+1,j)）。
    返回边索引与星/环成员表。"""
    # 边编号：0..2L²-1：水平 L² 条 + 竖直 L² 条
    edges = {}  # (type, i, j) -> id
    eid = 0
    for i in range(L):
        for j in range(L):
            edges[('h', i, j)] = eid
            eid += 1
    for i in range(L):
        for j in range(L):
            edges[('v', i, j)] = eid
            eid += 1
    # 星（顶点 i,j：四条边 h(i,j), h(i,j-1), v(i,j), v(i-1,j)）
    stars = []
    for i in range(L):
        for j in range(L):
            stars.append([edges[('h', i, j)], edges[('h', i, (j - 1) % L)],
                          edges[('v', i, j)], edges[('v', (i - 1) % L, j)]])
    # 环（plaquette (i,j)：h(i,j), h(i+1,j), v(i,j), v(i,j-1)）
    plaqs = []
    for i in range(L):
        for j in range(L):
            plaqs.append([edges[('h', i, j)], edges[('h', (i + 1) % L, j)],
                          edges[('v', i, j)], edges[('v', i, (j - 1) % L)]])
    return edges, stars, plaqs

def energy(k, stars, plaqs, N, kappap, kappa0, twopi):
    s = twopi / N
    E = 0.0
    for st in stars:
        E -= kappa0 * np.cos(s * sum(k[e] for e in st))
    for pl in plaqs:
        E -= kappap * np.cos(s * sum(k[e] for e in pl))
    return E

def ring_winding(k, plaqs, N, L):
    """绕数：周期格中沿 y 方向通量线的净通量 = 一行 plaquette 通量和（mod N）取整。"""
    s = 2 * np.pi / N
    # 第 0 列 plaquette 链（i=0..L-1, j=0）的通量和
    tot = 0.0
    for i in range(L):
        pl = plaqs[i * L + 0]
        tot += sum(k[e] for e in pl)
    W = tot / N
    return W - np.round(W)  # 到最近整数距离（0 = 通量零）

def flux_line(k, plaqs, N, L):
    """绕数（整数）：穿过环的通量线数 = 每列 plaquette 通量和（mod N）沿列求和。"""
    s = 2 * np.pi / N
    cols = []
    for j in range(L):
        tot = 0
        for i in range(L):
            tot += sum(k[e] for e in plaqs[i * L + j])
        cols.append(tot)
    # 列通量和 mod N（周期格闭合环约束：各列差是守恒量）
    return [int(round(c / N)) % N if c % N else 0 for c in cols], cols

print("=" * 66)
print("纠缠侧探路：绕数 = 关系通量（toric code 风格，经典层）")
print("=" * 66)

L = 8
N = 4
edges, stars, plaqs = build(L)
NE = len(edges)
s = 2 * np.pi / N
print(f"L = {L}（周期格），键数 = {NE}，Z_{N} 关系自由度，"
      f"星项 κ_φ = 1，环项 κ_∥ = 1")

k0 = np.zeros(NE, dtype=int)
E0 = energy(k0, stars, plaqs, N, 1.0, 1.0, s)
print(f"真空（全 k=0）：E = {E0:.4f}，绕数 = 0")

# ---------------- C1：真空单键翻转票价 ----------------
print("\n[C1] 真空单键翻转 ΔE（局域扰动的票价）")
dEs = []
for e in range(NE):
    k1 = k0.copy()
    k1[e] = (k1[e] + 1) % N
    dEs.append(energy(k1, stars, plaqs, N, 1.0, 1.0, 2 * np.pi) - E0)
dEs = np.array(dEs)
print(f"  单键 +1: ΔE 范围 [{dEs.min():.4f}, {dEs.max():.4f}]，均值 {dEs.mean():.4f}")
print(f"  对比：光滑场环翻页票价 0.19κ_∥；单格点相位量子化鞍点 2κ_φ(1-cos 2π/N) = {2*(1-np.cos(s)):.4f}")

# ---------------- C2：通量线构型 + 局域零温更新 ----------------
print("\n[C2] 通量线构型（绕数 = 1）在局域零温更新下的演化")
# 构造绕数 1：把第 0 行竖直键全设为 +1（穿过一行 plaquette 的通量线）
kring = k0.copy()
for j in range(L):
    kring[edges[('v', 0, j)]] = 1
W0_list, cols0 = flux_line(kring, plaqs, N, L)
print(f"  初态：通量线构型，列通量和 = {cols0}")
E_r = energy(kring, stars, plaqs, N, 1.0, 1.0, s)
print(f"  初态能量（相对真空）：{E_r - E0:.4f}")
# 零温 Metropolis：随机键 ±1，接受 ΔE <= 0（等能量随机接受 50%）
rng = np.random.default_rng(42)
k = kring.copy()
flip_events = 0
W_traj = []
for step in range(20000):
    e = rng.integers(NE)
    dk = 1 if rng.random() < 0.5 else -1
    kc = k.copy()
    kc[e] = (kc[e] + dk) % N
    dE = energy(kc, stars, plaqs, N, 1.0, 1.0, 2 * np.pi) - energy(k, stars, plaqs, N, 1.0, 1.0, 2 * np.pi)
    if dE < 0 or (dE == 0 and rng.random() < 0.5):
        k = kc
    if step % 2000 == 0:
        _, cols = flux_line(k, plaqs, N, L)
        W_traj.append(cols)
# 末态分析
_, cols_f = flux_line(k, plaqs, N, L)
E_f = energy(k, stars, plaqs, N, 1.0, 1.0, s)
print(f"  演化 20000 步后：列通量 = {cols_f}")
print(f"  能量（相对真空）：{E_f - E0:.4f}（初态 {E_r - E0:.4f}）")
same = (cols_f == cols0)
print(f"  绕数保持：{'是 ✓（通量线未被拉回）' if same else '否 ✗（通量线消失/改变）'}")

# ---------------- C3：通量线消失路径的能量剖面 ----------------
print("\n[C3] 绕数翻转票价：通量线逐键抹除的能量剖面")
# 路径：把通量线上的竖直键逐个归零（局部最优 = 每次抹除一个键，测 ΔE）
kpath = kring.copy()
prof = []
for j in range(L):
    e = edges[('v', 0, j)]
    E_before = energy(kpath, stars, plaqs, N, 1.0, 1.0, 2 * np.pi)
    # 尝试两种抹除：键减 1（k=1 → 0）或加 1（k=1 → 2）
    cands = []
    for dk in (-1, +1):
        kc = kpath.copy()
        kc[e] = (kc[e] + dk) % N
        cands.append((energy(kc, stars, plaqs, N, 1.0, 1.0, 2 * np.pi), kc, dk))
    cands.sort()
    kpath = cands[0][1]
    prof.append(cands[0][0] - E_before)
print(f"  逐键抹除通量线的 ΔE 序列: {[f'{p:+.4f}' for p in prof]}")
print(f"  累计: {sum(prof):+.4f}（理想路径应 ≈ 初态能量 {-E_r + E0:.4f}）")
# 鞍点（最大中间能量）
E_mid = []
kpath = kring.copy()
for j in range(L):
    e = edges[('v', 0, j)]
    for dk in (-1, +1):
        kc = kpath.copy()
        kc[e] = (kc[e] + dk) % N
        E_mid.append(energy(kc, stars, plaqs, N, 1.0, 1.0, 2 * np.pi))
    kpath = kc if dk == -1 else kpath
E_sad = max(E_mid) - E0
print(f"  路径最大中间能量（鞍点，相对真空）: {E_sad:.4f}（对比光滑场 0.19，单格点 2κ_φ(1-cos 2π/N) = {2*(1-np.cos(s)):.4f}）")

print("\n纠缠侧探路完成（仅打印）。")
