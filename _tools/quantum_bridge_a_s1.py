# -*- coding: utf-8 -*-
"""
张量桥分段 (a) S1 探测（v2.1，纯 numpy）：SO(3) 标架写入 + 扩散的稳态空间曲率
2026-09-13 v2.1：rodrigues 修正（单位轴叉乘矩阵归一化），旋转守恒恢复
（v1 scipy 超时；v2 全 numpy 但 rodrigues 未归一化叉乘矩阵，旋转不守恒已修正）

对数近似：log(M) ≈ (M - M^T)/2 的轴向分量（小角，稳态相对旋转 << 1）
指数：Rodrigues 公式批量实现（Kh = K/|w| 归一化单位轴叉乘矩阵）
判据同 v1：J1 空间格面亏损 ~ 1/r；J2 与时间向同阶；J3 远场回平坦
"""
import numpy as np

def skew_vec(M):
    """SO(3) 矩阵 -> 轴向向量近似（反对称部分，小角）：v = sin(theta) n_hat ~ theta n_hat"""
    return np.stack([M[..., 2, 1] - M[..., 1, 2],
                     M[..., 0, 2] - M[..., 2, 0],
                     M[..., 1, 0] - M[..., 0, 1]], axis=-1) / 2.0

def rodrigues(w):
    """批量 Rodrigues：轴角向量 -> SO(3) 矩阵（v2.1：Kh = K/|w| 归一化）"""
    w = np.asarray(w, dtype=np.float64)
    phi = np.linalg.norm(w, axis=-1)
    shape = w.shape[:-1]
    I = np.broadcast_to(np.eye(3), shape + (3, 3)).copy()
    out = I.copy()
    nz = phi > 1e-12
    if nz.any():
        ww = w[nz]
        p = phi[nz]
        K = np.zeros(ww.shape[:-1] + (3, 3))
        K[..., 0, 1] = -ww[..., 2]; K[..., 0, 2] = ww[..., 1]
        K[..., 1, 0] = ww[..., 2];  K[..., 1, 2] = -ww[..., 0]
        K[..., 2, 0] = -ww[..., 1]; K[..., 2, 1] = ww[..., 0]
        Kh = K / p[..., None, None]
        K2 = Kh @ Kh
        out[nz] = I[nz] + np.sin(p)[..., None, None] * Kh + (1.0 - np.cos(p))[..., None, None] * K2
    return out

def tick(F, src, dth, alpha):
    """一拍：SO(3) 扩散（邻居相对旋转对数平均）+ 源注入 R_x(dth)"""
    rel = np.zeros(F.shape[:-2] + (3,), dtype=np.float64)
    for a in range(3):
        for s in (+1, -1):
            nb = np.roll(F, -s, axis=a)
            rel += skew_vec(nb @ np.swapaxes(F, -1, -2))
    rel /= 6.0
    Fn = rodrigues(alpha * rel) @ F
    Fn[src] = rodrigues(np.array([dth, 0.0, 0.0])) @ Fn[src]
    return Fn

def plaquette_eps(F):
    """三方向格面 holonomy 亏损角（Regge，小角近似）"""
    U = [np.roll(F, -1, axis=a) @ np.swapaxes(F, -1, -2) for a in range(3)]
    angs = []
    for a in range(3):
        for b in range(a + 1, 3):
            Ux, Uy = U[a], U[b]
            H = (np.swapaxes(Uy, -1, -2)
                 @ np.roll(np.swapaxes(Ux, -1, -2), -1, axis=b)
                 @ np.roll(Uy, -1, axis=a)
                 @ Ux)
            angs.append(np.linalg.norm(skew_vec(H), axis=-1))
    return angs

print("=" * 72)
print("张量桥分段 (a) S1 探测 v2.1：SO(3) 标架写入+扩散稳态的空间曲率")
print("=" * 72)

N = 17
c2 = N // 2
shape = (N, N, N)
src = (c2, c2, c2)
dth = 0.02
alpha = 0.5
T = 15000

F = np.broadcast_to(np.eye(3), shape + (3, 3)).copy()
print(f"参数：N={N}, 注入 dth={dth}, alpha={alpha}, T={T}")

for t in range(T):
    F = tick(F, src, dth, alpha)
    if t in (1000, 4000, 9000, T - 1):
        phi = np.linalg.norm(skew_vec(F), axis=-1)
        print(f"  t={t:6d}: max|rot|={phi.max():.4f}, |rot(src)|={phi[src]:.4f}")

phi = np.linalg.norm(skew_vec(F), axis=-1)
eps = plaquette_eps(F)
print(f"\n稳态：max|rot|={phi.max():.4f}, 空间格面亏损 max={max(A.max() for A in eps):.3e}")

ii, jj, kk = np.indices(shape)
r_grid = np.sqrt((ii - c2) ** 2 + (jj - c2) ** 2 + (kk - c2) ** 2)
far = r_grid > N // 3
print(f"J3 远场(r>{N//3})：max|rot|={phi[far].max():.3e}, "
      f"max 格面亏损={max(A[far].max() for A in eps):.3e}")

print("\n沿 x 轴采样：")
print(f"  {'r':>4} {'|rot|(时间向)':>14} {'eps_xz':>14} {'eps_xy':>14} {'eps* r²':>12}")
ex = []
for r in range(2, N // 2):
    n = (c2 + r, c2, c2)
    ep = max(A[n] for A in eps)
    ex.append((r, phi[n], ep))
    if r in (3, 5, 7):
        print(f"  {r:4d} {phi[n]:14.6e} {eps[0][n]:14.6e} {eps[1][n]:14.6e} {ep*r*r:12.4e}")

rs = np.array([x[0] for x in ex], dtype=float)
phis = np.array([x[1] for x in ex])
epsx = np.array([x[2] for x in ex])
c_phi = np.polyfit(np.log(rs), np.log(phis), 1)
print(f"\n时间向 |rot| 拟合：~ r^{c_phi[0]:.2f}（期望 -1）")
m = epsx > 0
if m.sum() > 3:
    c_eps = np.polyfit(np.log(rs[m]), np.log(epsx[m]), 1)
    print(f"空间亏损拟合：~ r^{c_eps[0]:.2f}（J1 期望 -1 量级）")
    ratio = np.median(epsx[m] * rs[m] / (phis[m] * rs[m]))
    print(f"J2 系数比（中位）={ratio:.3e}（史瓦西期望 O(1) 量级）")
else:
    print("空间亏损在采样带内为零或过小，J1 未成立")

print("\n探测完成。")
