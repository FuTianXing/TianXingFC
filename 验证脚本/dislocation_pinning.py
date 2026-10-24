# -*- coding: utf-8 -*-
"""
钉扎身份验证：位错核心 = 体系内的天然弱耦合柱？

预印本 Derrick 检查用"弱耦合柱"（J → 0.3J，柱半径 3）验证涡旋线钉扎，
但弱耦合柱的体系内身份未定。本脚本检验候选：位错核心（A3 位错网络节点）。

螺型位错（Burgers b = 1 格距，沿 z）：位移场 u_z(i,j) = b·θ/(2π)，
θ = atan2(j-c, i-c)。格点位移导致相邻格点相对位移 Δu = b·Δθ/(2π)，
键耦合按 XY 型相位耦合 J_eff = J·cos(2πΔu/b)（理想晶格 Δu=0 → J_eff=J；
核心处 Δθ 大 → J_eff 弱化甚至反号）。

判据：
  1. J_eff/J 的空间分布（x-y 平面键平均）
  2. 核心区（|J_eff| < 0.5J）的半径与平均 J_eff/J
  3. 对照预印本弱耦合柱（Jw = 0.3J，柱半径 3）：核心平均是否 ≈ 0.3J
  4. 若成立：钉扎身份 = 位错核心（A3 衔接），弱耦合柱获得体系内对应
"""
import numpy as np

print("=" * 72)
print("钉扎身份验证：位错核心 = 天然弱耦合柱？")
print("=" * 72)

N = 64
c = N // 2
jj, ii = np.meshgrid(np.arange(N) - c, np.arange(N) - c, indexing="ij")
theta = np.arctan2(jj, ii)
b = 1.0
u = b * theta / (2 * np.pi)

# 键位移差：x 键 (i,j)-(i+1,j)：Δu_x = u[i+1,j]-u[i,j]（形状 (N-1, N)）
#          y 键 (i,j)-(i,j+1)：Δu_y = u[i,j+1]-u[i,j]（形状 (N, N-1)）
du_x = np.diff(u, axis=0)
du_y = np.diff(u, axis=1)
Jx = np.cos(2 * np.pi * du_x / b)
Jy = np.cos(2 * np.pi * du_y / b)

# 键位置：x 键 (i+0.5, j)，y 键 (i, j+0.5)
xi = np.arange(N - 1) + 0.5 - c
xj = np.arange(N) - c
yi = np.arange(N) - c
yj = np.arange(N - 1) + 0.5 - c
r2_x = xi[None, :] ** 2 + xj[:, None] ** 2     # (N, N-1) → 转置 (N-1, N) 对齐 Jx
r2_y = yj[None, :] ** 2 + yi[:, None] ** 2     # (N, N-1) 对齐 Jy
r2_xT = r2_x.T

for name, Jb, r2b in [("x 键", Jx, r2_xT), ("y 键", Jy, r2_y)]:
    core = r2b <= 9.0
    far = r2b > 400.0
    Jc = Jb[core]
    print(f"\n{name}: 核心(r<3)平均 J_eff/J = {Jc.mean():.3f}, "
          f"弱化键占比(|J|<0.5) = {(np.abs(Jc) < 0.5).mean():.3f}, "
          f"反号键占比(J<0) = {(Jc < 0).mean():.3f}")
    print(f"   远区(r>20)平均 J_eff/J = {Jb[far].mean():.5f}（应 ≈ 1）")
    print("   径向壳平均 J_eff/J：")
    for rr in (1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0):
        ring = (r2b <= rr ** 2) & (r2b > (rr - 1.0) ** 2)
        if ring.sum() > 0:
            print(f"     r in ({rr-1},{rr}]: {Jb[ring].mean():.3f}")

Jall = np.concatenate([Jx[r2_xT <= 9.0], Jy[r2_y <= 9.0]])
print(f"\n综合(x+y 键, 核心柱 r<3): 平均 J_eff/J = {Jall.mean():.3f}, "
      f"RMS = {np.sqrt(np.mean(Jall**2)):.3f}, "
      f"弱化键占比 = {(np.abs(Jall) < 0.5).mean():.3f}")

print("\n对照：预印本 Derrick 弱耦合柱 Jw = 0.3J（柱半径 3）")
print("判据：核心平均 |J_eff|/J ≈ 0.3 → 位错核心 = 天然弱耦合柱（钉扎身份确认）")
print("\n验证完成。")
