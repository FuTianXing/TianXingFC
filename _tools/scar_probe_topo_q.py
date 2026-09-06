# -*- coding: utf-8 -*-
"""
纠缠侧探路：量子层（Z_2 toric code，L=2 环面周期格）
绕数（电荷）= 大环算符本征值（全局非定域量子数）
H = -κ_φ Σ_v A_v - κ_∥ Σ_p B_p
  A_v = ∏_{e∈v} σ^x_e（顶点项，格点内锁定）
  B_p = ∏_{e∈p} σ^z_e（环项，plaquette 通量）
判据：
  Q1 [H, W] = 0（大环绕数算符与哈密顿量对易，数值验证）
  Q2 基态简并 = 2^{2g} = 4（环面，绕数扇区分割基态）
  Q3 局域扰动（单键场 εΣσ^x）下绕数扇区投影保持（能隙保护）
结果只打印终端，不写任何文档。
"""
import numpy as np

L = 2
NE = 2 * L * L  # 8 键
nbit = NE
D = 2 ** nbit
print("=" * 66)
print("纠缠侧探路：量子层（Z_2 toric code，L=2 环面）")
print(f"键数 = {NE}，希尔伯特空间 = 2^{nbit} = {D} 维")
print("=" * 66)

# 键编号：水平 h[i,j]（i,j=0..1）+ 竖直 v[i,j]
def eid(t, i, j):
    if t == 'h':
        return i * L + j
    return L * L + i * L + j

# 星（顶点 i,j：h(i,j), h(i,j-1), v(i,j), v(i-1,j)）
stars = []
for i in range(L):
    for j in range(L):
        stars.append([eid('h', i, j), eid('h', i, (j - 1) % L),
                      eid('v', i, j), eid('v', (i - 1) % L, j)])
# 环（plaquette (i,j)：h(i,j), h(i+1,j), v(i,j), v(i,j-1)）
plaqs = []
for i in range(L):
    for j in range(L):
        plaqs.append([eid('h', i, j), eid('h', (i + 1) % L, j),
                      eid('v', i, j), eid('v', i, (j - 1) % L)])

def bit(s, e):
    return (s >> e) & 1

# 构造 H 矩阵（256 维稠密可行）
H = np.zeros((D, D))
# B_p 项（对角）：(-1)^{Σ b}
for pl in plaqs:
    d = np.zeros(D)
    for s in range(D):
        d[s] = -1.0 if sum(bit(s, e) for e in pl) % 2 == 1 else 1.0
    H -= np.diag(d)  # -κ_∥ B_p（κ_∥ = 1）
# A_v 项（置换）：翻转顶点四键
for st in stars:
    A = np.zeros((D, D))
    for s in range(D):
        t = s
        for e in st:
            t ^= (1 << e)
        A[t, s] = 1.0
    H -= A  # -κ_φ A_v（κ_φ = 1）
# 大环绕数算符：绕 x 方向 = 固定 j=0 的一列竖直键乘积；绕 y 方向 = 固定 i=0 的一行水平键乘积
W_op = np.zeros((D, D))
W_op2 = np.zeros((D, D))
for s in range(D):
    px = sum(bit(s, eid('v', i, 0)) for i in range(L)) % 2
    py = sum(bit(s, eid('h', 0, j)) for j in range(L)) % 2
    W_op[s, s] = 1.0 if px == 0 else -1.0
    W_op2[s, s] = 1.0 if py == 0 else -1.0

# Q1：对易检验
comm = np.max(np.abs(H @ W_op - W_op @ H))
comm2 = np.max(np.abs(H @ W_op2 - W_op2 @ H))
print(f"\n[Q1] [H, W_x] 最大矩阵元 = {comm:.2e}，[H, W_y] = {comm2:.2e}")
print(f"     {'对易成立 ✓（绕数算符为运动常数）' if comm < 1e-10 and comm2 < 1e-10 else '对易不成立 ✗'}")

# Q2：基态简并（精确对角化）
w, v = np.linalg.eigh(H)
E0 = w[0]
deg = int((np.abs(w - E0) < 1e-8).sum())
gap = w[deg] - E0
print(f"\n[Q2] 基态能量 = {E0:.6f}，简并度 = {deg}（环面 2^(2g) = 4 预期）")
print(f"     能隙 = {gap:.6f}")
for k in range(min(deg, 4)):
    gs = v[:, k]
    wxp = np.sum(gs ** 2 * (W_op.diagonal() > 0))
    print(f"     基态 {k}: W_x = +1 权重 {wxp:.3f}，W_x = -1 权重 {1 - wxp:.3f}")

# Q3：局域扰动下绕数扇区保持（能隙保护签名）
print("\n[Q3] 局域扰动 V = εΣ_e σ^x_e：一阶矩阵元与演化保持")
V = np.zeros((D, D))
for e in range(NE):
    Vx = np.zeros((D, D))
    for s in range(D):
        Vx[s ^ (1 << e), s] = 1.0
    V += Vx
psi_plus = v[:, 0]
psi_minus = v[:, 2]
first_order = np.abs(np.dot(psi_plus, V @ psi_minus))
print(f"  一阶矩阵元 <W=+1|V|W=-1> = {first_order:.6f}"
      f"（{'≈ 0 ✓ 能隙保护（基态间无直接耦合）' if first_order < 1e-10 else '非零 ✗'}）")
leak2 = 0.0
for k in range(1, D):
    dE = w[k] - w[0]
    if dE < 1e-8:
        continue
    leak2 += np.abs(np.dot(psi_plus, V @ v[:, k])) ** 2 / dE
print(f"  二阶泄漏率 Σ|<+|V|k>|²/ΔE = {leak2:.6f}")
for eps in (0.01, 0.1, 0.5):
    Hp = H + eps * V
    wp, vp = np.linalg.eigh(Hp)
    c = vp.T @ psi_plus
    Wbar = 0.0
    for k in range(D):
        Wbar += np.abs(c[k]) ** 2 * (vp[:, k].conj() @ W_op @ vp[:, k]).real
    print(f"  ε = {eps}: 初态 W=+1 演化后 W_x 时间平均 = {Wbar:+.4f}"
          f"（{'保持 ✓' if abs(Wbar - 1.0) < 0.1 else ('部分保持' if Wbar > 0.3 else '丢失 ✗')}）")
print("\n量子层探路完成（仅打印）。")
