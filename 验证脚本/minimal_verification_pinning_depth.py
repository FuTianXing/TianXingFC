# -*- coding: utf-8 -*-
"""
钉扎能 vs 弱化强度：位错核心弱化（J_eff ≈ 0.84J）是否足以钉扎？

预印本 Derrick 钉扎验证用 Jw = 0.3J 得阱深约 420。位错核心弹性弱化实测
J_eff/J ≈ 0.844。本脚本测钉扎能对 Jw 的依赖（Jw = 0.84/0.6/0.3），
判断弱钉扎（0.84）是否非零及其强度。

模型：三维 XY 场（同 Derrick [3]），涡旋线沿 z 固定在中心，
弱耦合柱（半径 3，Jw）沿 z 位于距线距离 D 处，弛豫测 E(D)。
阱深 = E(D=8) - E(D=0)。
"""
import numpy as np

print("=" * 72)
print("钉扎能 vs 弱化强度 Jw")
print("=" * 72)

N = 32
c = N // 2
shape = (N, N, N)
Y, X = np.meshgrid(np.arange(N) - c, np.arange(N) - c, indexing="ij")
phi_line = np.broadcast_to(np.arctan2(Y, X)[:, :, None], shape).copy()

def relax(phi, col_mask, Jw, iters=2500, eta=0.05):
    phi = phi.copy()
    for t in range(iters):
        s = np.zeros_like(phi)
        for ax in range(3):
            s += (np.sin(phi - np.roll(phi, 1, ax)) + np.sin(phi - np.roll(phi, -1, ax)))
        s = np.where(col_mask, Jw * s, s)
        phi = phi - eta * s
    return phi

def total_E(phi, col_mask, Jw):
    E = 0.0
    for ax in range(3):
        d1 = np.roll(phi, 1, ax) - phi
        d2 = np.roll(phi, -1, ax) - phi
        E += 0.5 * np.sum((1 - np.cos(d1))) + 0.5 * np.sum((1 - np.cos(d2)))
    for ax in range(3):
        d1 = np.roll(phi, 1, ax) - phi
        d2 = np.roll(phi, -1, ax) - phi
        E -= (1 - Jw) * 0.5 * np.sum(col_mask * (1 - np.cos(d1)))
        E -= (1 - Jw) * 0.5 * np.sum(col_mask * (1 - np.cos(d2)))
    return E

for Jw in (0.84, 0.6, 0.3):
    print(f"\nJw = {Jw}")
    Es = []
    for D in (0, 2, 4, 6, 8):
        i4, j4, k4 = np.indices(shape)
        col = ((i4 - D) ** 2 + j4 ** 2) <= 9.0
        phi_r = relax(phi_line, col, Jw, iters=2500)
        E = total_E(phi_r, col, Jw)
        Es.append(E)
        print(f"  D = {D}: E = {E:.3f}")
    depth = Es[0] - Es[-1]
    print(f"  阱深 = E(D=8) - E(D=0) = {-depth:.3f}（{'阱存在' if depth > 0.5 else '≈0: 无钉扎'}）")

print("\n验证完成。")
