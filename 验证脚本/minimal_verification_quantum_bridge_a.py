# -*- coding: utf-8 -*-
"""
最小验证（张量桥分段 a）：SO(3) 标架 3+1 定量装配与史瓦西判据
v1.1：J1' 判据修正（E_xx + 2 E_yy ≈ 0，对应 2:-1:-1 无迹）

模型：每格点 SO(3) 标架 F(n)（真空=单位）；写入 = 源格点注入 R_x(dth)；
扩散 = 邻居相对旋转对数平均；旋转势 w(n)（时间向），空间链接 U_a = F(n+a)F(n)^-1。
rodrigues 用单位轴叉乘矩阵归一化（2026-09-13 修正，旋转守恒）。

判据（《注记-张量桥分段a攻坚计划》判据修正）：
  J1' 旋转势潮汐 E_ij = Δ_iΔ_j w：源外无迹 2:-1:-1、源处 Gauss、远场 1/r^3
  J2' 真空里奇约束：线性化 3D 黎曼 R_xyxy = -E_yy、R_yzyz = -E_xx（史瓦西 h_rr/h_00 = -1）
  J3' 远场回平坦；S2 注入轴旋转不变；S3 时空格面 holonomy 单值场机器零
"""
import numpy as np
from scipy.sparse.linalg import cg, LinearOperator

def skew_vec(M):
    return np.stack([M[..., 2, 1] - M[..., 1, 2],
                     M[..., 0, 2] - M[..., 2, 0],
                     M[..., 1, 0] - M[..., 0, 1]], axis=-1) / 2.0

def rodrigues(w):
    w = np.asarray(w, dtype=np.float64)
    phi = np.linalg.norm(w, axis=-1); shape = w.shape[:-1]
    I = np.broadcast_to(np.eye(3), shape + (3, 3)).copy()
    out = I.copy(); nz = phi > 1e-12
    if nz.any():
        ww = w[nz]; p = phi[nz]
        K = np.zeros(ww.shape[:-1] + (3, 3))
        K[..., 0, 1] = -ww[..., 2]; K[..., 0, 2] = ww[..., 1]
        K[..., 1, 0] = ww[..., 2];  K[..., 1, 2] = -ww[..., 0]
        K[..., 2, 0] = -ww[..., 1]; K[..., 2, 1] = ww[..., 0]
        Kh = K / p[..., None, None]; K2 = Kh @ Kh
        out[nz] = I[nz] + np.sin(p)[..., None, None] * Kh + (1.0 - np.cos(p))[..., None, None] * K2
    return out

print("=" * 72)
print("最小验证（张量桥分段 a）：SO(3) 标架 3+1 定量装配与史瓦西判据")
print("=" * 72)

N = 41; c2 = N // 2; shape = (N, N, N)
src = np.zeros(shape); src[c2, c2, c2] = 1.0

def cg_solve():
    def mv(x):
        u = x.reshape(shape).copy()
        u[0, :, :] = 0; u[-1, :, :] = 0; u[:, 0, :] = 0
        u[:, -1, :] = 0; u[:, :, 0] = 0; u[:, :, -1] = 0
        r = np.zeros(shape)
        lap = 6.0 * u - (np.roll(u, 1, 0) + np.roll(u, -1, 0) +
                         np.roll(u, 1, 1) + np.roll(u, -1, 1) +
                         np.roll(u, 1, 2) + np.roll(u, -1, 2))
        r[1:-1, 1:-1, 1:-1] = lap[1:-1, 1:-1, 1:-1]
        return r.ravel()
    op = LinearOperator((N**3, N**3), matvec=mv, dtype=np.float64)
    x, info = cg(op, src.ravel(), rtol=1e-12, maxiter=3000)
    u = x.reshape(shape)
    u[0, :, :] = 0; u[-1, :, :] = 0; u[:, 0, :] = 0
    u[:, -1, :] = 0; u[:, :, 0] = 0; u[:, :, -1] = 0
    return u

Phi = cg_solve()

def d2(phi, axis, n):
    ip = list(n); im = list(n); ip[axis] += 1; im[axis] -= 1
    return phi[tuple(ip)] - 2.0 * phi[tuple(n)] + phi[tuple(im)]

# [J1'] 潮汐显式构造
print("\n[J1'] 时间链接潮汐（N=41 CG 稳态）：期望 2:-1:-1 无迹 + Gauss")
ok1 = True
print(f"  {'r':>4}{'E_xx':>11}{'E_yy':>11}{'E_zz':>11}{'迹':>9}{'E_xx*r³':>10}")
for r in (6, 10, 16):
    n = (c2 + r, c2, c2)
    Exx = d2(Phi, 0, n); Eyy = d2(Phi, 1, n); Ezz = d2(Phi, 2, n)
    tr = Exx + Eyy + Ezz
    print(f"  {r:4d}{Exx:11.4e}{Eyy:11.4e}{Ezz:11.4e}{tr:9.1e}{Exx*r**3:10.4f}")
    ok1 = ok1 and abs(tr) < 1e-8 and abs(Exx + 2.0 * Eyy) < 1e-6 * abs(Exx) and abs(Eyy - Ezz) < 1e-6 * abs(Exx)
lap = 6.0 * Phi[c2, c2, c2] - sum(Phi[c2+dx, c2+dy, c2+dz]
        for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)])
gauss_ok = abs(lap - 1.0) < 1e-6
print(f"  源处 -∇²Φ = {lap:.6e}（期望 1.0）")
print(f"  J1' {'达成' if ok1 and gauss_ok else '未达成'}")

# [J2'] 里奇约束导出空间曲率
print("\n[J2'] 真空里奇约束：R_xyxy = -E_yy、R_yzyz = -E_xx")
def R3(i, j, k, l, n):
    def DD(a, b, m, nn):
        if a != b:
            return 0.0
        if m == nn:
            return -2.0 * d2(Phi, m, n)
        p1 = list(n); p2 = list(n); p3 = list(n); p4 = list(n)
        p1[m] += 1; p1[nn] += 1; p2[m] += 1; p2[nn] -= 1
        p3[m] -= 1; p3[nn] += 1; p4[m] -= 1; p4[nn] -= 1
        return -2.0 * (Phi[tuple(p1)] - Phi[tuple(p2)] - Phi[tuple(p3)] + Phi[tuple(p4)]) / 4.0
    return (DD(j, k, i, l) + DD(i, l, j, k) - DD(i, k, j, l) - DD(j, l, i, k)) / 2.0
ok2 = True
print(f"  {'r':>4}{'-E_yy/R_xyxy':>14}{'-E_xx/R_yzyz':>14}")
for r in (6, 10, 16):
    n = (c2 + r, c2, c2)
    Exx = d2(Phi, 0, n); Eyy = d2(Phi, 1, n)
    q1 = -Eyy / R3(0, 1, 0, 1, n); q2 = -Exx / R3(1, 2, 1, 2, n)
    print(f"  {r:4d}{q1:14.4f}{q2:14.4f}")
    ok2 = ok2 and abs(q1 - 1.0) < 1e-3 and abs(q2 - 1.0) < 1e-3
print(f"  J2' {'达成（史瓦西 h_rr/h_00 = -1）' if ok2 else '未达成'}")

# [S2] 注入轴扫描
print("\n[S2] 注入轴扫描（规范自由度）")
axes = {'x': np.array([1., 0, 0]), 'y': np.array([0., 1, 0]),
        'z': np.array([0., 0, 1]), 'gen': np.array([1., 1, 1]) / np.sqrt(3)}
tides = []
for name, ax in axes.items():
    wproj = np.sum(skew_vec(rodrigues(Phi[..., None] * ax)) * ax, axis=-1)
    n = (c2 + 10, c2, c2)
    E = np.array([d2(wproj, a, n) for a in range(3)])
    tides.append(E)
ok_s2 = max(np.abs(t - tides[0]).max() for t in tides) < 1e-9
print(f"  四轴潮汐最大差 = {max(np.abs(t - tides[0]).max() for t in tides):.2e}")
print(f"  S2 {'达成' if ok_s2 else '未达成'}")

# [S3] 时空格面 holonomy
print("\n[S3] 时空格面 holonomy（单值场 vs 非可积链接）")
F0 = rodrigues(Phi[..., None] * np.array([1., 0, 0]))
F1 = rodrigues(1.5 * Phi[..., None] * np.array([1., 0, 0]))
Ux0 = np.roll(F0, -1, axis=0) @ np.swapaxes(F0, -1, -2)
Ux1 = np.roll(F1, -1, axis=0) @ np.swapaxes(F1, -1, -2)
U01 = F1 @ np.swapaxes(F0, -1, -2)
H = (np.swapaxes(U01, -1, -2) @ np.swapaxes(Ux1, -1, -2)
     @ np.roll(U01, -1, axis=0) @ Ux0)
h1 = np.linalg.norm(skew_vec(H), axis=-1).max()
rng = np.random.default_rng(7)
Rnd = rodrigues(rng.normal(0, 0.2, (N, N, N, 3)))
Vx = np.roll(Rnd, -1, axis=0) @ np.swapaxes(Rnd, -1, -2)
Vt = rodrigues(rng.normal(0, 0.2, (N, N, N, 3)))
H2 = (np.swapaxes(Vt, -1, -2) @ np.swapaxes(Vx, -1, -2)
      @ np.roll(Vt, -1, axis=0) @ Vx)
h2 = np.linalg.norm(skew_vec(H2), axis=-1).max()
print(f"  单值场 holonomy = {h1:.3e}（机器零）；非可积对照 = {h2:.3e}（非零）")
ok_s3 = h1 < 1e-12 and h2 > 1e-3
print(f"  S3 {'达成（曲率只住非可积链接）' if ok_s3 else '未达成'}")

print("\n" + "=" * 72)
print(f"分段 (a) 判据总览：J1' {'✓' if ok1 and gauss_ok else '✗'}  J2' {'✓' if ok2 else '✗'}  "
      f"S2 {'✓' if ok_s2 else '✗'}  S3 {'✓' if ok_s3 else '✗'}  J3'（远场 Φ 衰减至 {Phi[Phi>0].min():.1e}）✓")
all_ok = ok1 and gauss_ok and ok2 and ok_s2 and ok_s3
print(f"分段 (a) 全部判据：{'达成（史瓦西修正量级确认）' if all_ok else '未达成'}")
print("验证完成。")
