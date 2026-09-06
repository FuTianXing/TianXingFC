# -*- coding: utf-8 -*-
"""
Z_2 toric code L=3 环面：验证基态简并与 L 无关（预期 2^(2g) = 4）
态空间 2^18 = 262144，稀疏 eigsh
判据：
  Q2 基态简并 = 4（与 L=2 一致）
  Q1 基态空间内 W_x 投影量子数（可对角化 + 扇区分解 ±1）
  Q3 一阶矩阵元（跨扇区基态经 V = ΣX_e）
结果只打印终端，不写任何文档。
"""
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import eigsh

N = 2
L = 3
NE = 2 * L * L
D = 2 ** NE
print("=" * 60)
print(f"Z_2 toric code：L = {L} 环面，键数 {NE}，态空间 2^{NE} = {D}")
print("=" * 60)

def eid(t, i, j):
    return i * L + j if t == 'h' else L * L + i * L + j

stars = [[eid('h', i, j), eid('h', i, (j - 1) % L),
          eid('v', i, j), eid('v', (i - 1) % L, j)] for i in range(L) for j in range(L)]
plaqs = [[eid('h', i, j), eid('h', (i + 1) % L, j),
          eid('v', i, j), eid('v', i, (j + 1) % L)] for i in range(L) for j in range(L)]

# 键值基（Z 基）：|k_0...k_{NE-1}>，索引 = Σ k_e 2^e
rows, cols, vals = [], [], []
# B_p 对角：(-1)^{Σk}
for pl in plaqs:
    for s in range(D):
        p = 1.0 if sum((s >> e) & 1 for e in pl) % 2 == 0 else -1.0
        rows.append(s); cols.append(s); vals.append(-p)
# A_v：四键翻转
for st in stars:
    for s in range(D):
        t = s
        for e in st:
            t ^= (1 << e)
        rows.append(s); cols.append(t); vals.append(-1.0)
H = coo_matrix((vals, (rows, cols)), shape=(D, D)).tocsr()
print("H 构建完成，非零元 =", H.nnz)

# 大环绕数（对角）：W_x = (-1)^{Σ v(i,0)}、W_y = (-1)^{Σ h(0,j)}
Wx = np.ones(D, dtype=float)
Wy = np.ones(D, dtype=float)
for s in range(D):
    px = sum((s >> eid('v', i, 0)) & 1 for i in range(L)) % 2
    py = sum((s >> eid('h', 0, j)) & 1 for j in range(L)) % 2
    Wx[s] = 1.0 if px == 0 else -1.0
    Wy[s] = 1.0 if py == 0 else -1.0

w, v = eigsh(H, k=24, which='SA')
E0 = w[0]
deg = int((np.abs(w - E0) < 1e-6).sum())
gap = w[deg] - E0
print(f"[Q2] 基态能量 = {E0:.6f}，简并度 = {deg}（预期 2^(2g) = 4，与 L 无关）"
      f"{'✓' if deg == 4 else '✗'}")
print(f"     能隙 = {gap:.6f}")

gs = v[:, :deg]
# Q1：基态空间 W_x 表示
Wproj = np.zeros((deg, deg))
for a in range(deg):
    for b in range(deg):
        Wproj[a, b] = gs[:, a] @ (Wx * gs[:, b])
offdiag = np.max(np.abs(Wproj - np.diag(np.diag(Wproj))))
ew = np.linalg.eigvalsh(Wproj)
print(f"[Q1] 基态空间 W_x 表示：对角化偏离 = {offdiag:.2e}，本征值 = {np.round(ew, 4).tolist()}")
# 扇区分解
plus = int((ew > 0.5).sum())
minus = int((ew < -0.5).sum())
print(f"     W_x = +1 扇区 {plus} 态，W_x = -1 扇区 {minus} 态（预期各 2）")

# Q3：一阶矩阵元（+1 扇区与 -1 扇区基态间经 V = ΣX_e）
V = coo_matrix(([], ([], [])), shape=(D, D))
r3, c3, v3 = [], [], []
for e in range(NE):
    for s in range(D):
        r3.append(s); c3.append(s ^ (1 << e)); v3.append(1.0)
V = coo_matrix((v3, (r3, c3)), shape=(D, D)).tocsr()
gs_p = [gs[:, k] for k in range(deg) if ew[k] > 0.5]
gs_m = [gs[:, k] for k in range(deg) if ew[k] < -0.5]
m11 = 0.0
for a in gs_p:
    for b in gs_m:
        m11 = max(m11, abs(a @ (V @ b)))
print(f"[Q3] 一阶矩阵元 <W=+1|V|W=-1> = {m11:.6e}"
      f"（{'≈ 0 ✓ 能隙保护' if m11 < 1e-8 else '非零'}）")

# T5q：基态通量分布（第一个基态）
psi = gs[:, 0]
amp2 = np.abs(psi) ** 2
amp2 /= amp2.sum()
devs = []
for pl in plaqs:
    vals_pl = np.array([sum((s >> e) & 1 for e in pl) % N for s in range(D)], dtype=float)
    devs.append(float((amp2 * vals_pl).sum()))
print(f"[T5q] 基态各 plaquette 平均通量偏离（Z_2 单位）: {np.round(devs, 4).tolist()}")
print("\nZ_2 L=3 完成（仅打印）。")
