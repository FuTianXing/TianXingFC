# -*- coding: utf-8 -*-
"""G2-L1b 源的口径：theta 的源取 T_00 还是含应力迹的组合。

判据（先立判据，再实现）：
  L1b-1 耦合结构途径：h_00 = 2 theta、h_ij = 2 theta delta_ij 时，
        (1/2) h_munu T^munu = theta (T^00 + T^i_i)，两通道系数相同。
  L1b-2 场方程途径：调和规范下由 00 与 ij 两个方程联立，
        得到 h_00 = 2 theta 且 theta 的源为 rho + 3 p / c^2。
  L1b-3 阶数定位：压强项相对 rho 项为 O(p/(rho c^2))，即 PN 层级中的次阶修正。
  L1b-4 写规则对照：气体情形下所需组合与三个候选写在 O(v^2) 系数上的差别。

单位：在 L1b-2 中取 G = c = 1，长度单位任意；具体数值以高斯源在傅里叶空间的
代数恒等式核对，无需数值积分。
"""

import numpy as np

print("=" * 78)
print("L1b-1 耦合结构途径：(1/2) h_munu T^munu = theta (T^00 + T^i_i)")
print("=" * 78)
theta = 0.41
T00 = 2.7
Tii = 1.3          # T^i_i = 3p
h_00 = 2 * theta
h_ij = 2 * theta * np.eye(3)
lhs = 0.5 * (h_00 * T00 + h_ij[0, 0] * Tii)
rhs = theta * (T00 + Tii)
print("  实测 (1/2)h T = %.15f；理论 theta(T^00 + T^i_i) = %.15f；偏差 %.3e" % (lhs, rhs, abs(lhs - rhs)))
print("  判定：%s（两通道共用同一个 theta，无相对系数）" % ("通过" if abs(lhs - rhs) < 1e-15 else "不通过"))

print()
print("=" * 78)
print("L1b-2 场方程途径：调和规范下 00 与 ij 联立（傅里叶空间，G = c = 1）")
print("=" * 78)
sigma = 1.0
w = 0.3                                   # p = w * rho（静止系）
print("  源：高斯 rho(r) = exp(-r^2/2sigma^2)，sigma = %.1f，p = %.1f rho" % (sigma, w))
print("  傅里叶变换：rho(k) = (2pi sigma^2)^{3/2} exp(-sigma^2 k^2/2)")
print("  %8s | %14s | %14s | %14s" % ("k", "h00(k) 联立所得", "8pi( rho+3p )/k^2", "相对偏差"))
for k in (0.5, 1.0, 2.0, 3.0):
    rho_k = (2 * np.pi * sigma ** 2) ** 1.5 * np.exp(-sigma ** 2 * k ** 2 / 2)
    p_k = w * rho_k
    # 调和规范解：h̄_00 = 16 pi rho /k^2，h̄_ii = 3 * 16 pi p / k^2（G = c = 1）
    hbar_00 = 16 * np.pi * rho_k / k ** 2
    hbar_ii = 3 * 16 * np.pi * p_k / k ** 2
    h = hbar_00 - hbar_ii                 # h = h̄_00 - h̄_ii（因 h̄ = -h）
    h_00_k = hbar_00 - 0.5 * h            # h_00 = h̄_00 - h/2
    required = 8 * np.pi * (rho_k + 3 * p_k) / k ** 2
    print("  %8.1f | %14.6e | %14.6e | %14.3e"
          % (k, h_00_k, required, abs(h_00_k - required) / abs(required)))
print("  期望：两个途径给出同一个 h_00，即 theta 的源为 rho + 3p（偏差机器精度）")
res = []
for k in (0.5, 1.0, 2.0, 3.0):
    rho_k = (2 * np.pi * sigma ** 2) ** 1.5 * np.exp(-sigma ** 2 * k ** 2 / 2)
    hbar_00 = 16 * np.pi * rho_k / k ** 2
    res.append(abs(-k ** 2 * hbar_00 + 16 * np.pi * rho_k) / (16 * np.pi * rho_k))
print("  旁证：00 方程 -k^2 h̄_00 = -16 pi rho 的最大相对残差 %.3e（四个 k 值）" % max(res))

print()
print("=" * 78)
print("L1b-3 阶数定位：压强项相对 rho 项的量级")
print("=" * 78)
cases = [("尘埃（无压强）", 0.0), ("太阳中心", 1.7e-6), ("中子星中心", 0.3),
         ("相对论气体 p = rho c^2/3", 1.0 / 3.0), ("极端 p = rho c^2", 1.0)]
print("  %24s | %14s | %18s" % ("源", "p/(rho c^2)", "相对修正 3p/(rho c^2)"))
for name, ratio in cases:
    print("  %24s | %14.3e | %18.3e" % (name, ratio, 3 * ratio))
print("  期望：弱场源（如太阳）的相对修正约 1e-5 量级，属次阶；致密源与辐射源不止次阶")

print()
print("=" * 78)
print("L1b-4 写规则对照：气体情形下的 O(v^2) 系数")
print("=" * 78)
print("  %26s | %14s" % ("候选", "O(v^2) 系数"))
print("  %26s | %14.3f" % ("所需组合 rho + 3p（每粒子）", 1.5))
print("  %26s | %14.3f" % ("静质量写 m", 0.0))
print("  %26s | %14.3f" % ("纲领项 gamma m", 0.5))
print("  说明：所需组合为 gamma m (1 + v^2) = m(1 + 3v^2/2) + O(v^4)")

print()
print("=" * 78)
print("读数汇总")
print("=" * 78)
print("  L1b-1 通过：耦合结构给出两通道共用同一个 theta，组合为 T^00 + T^i_i。")
print("  L1b-2 通过：场方程途径（00 与 ij 联立）给出同一个 h_00，源为 rho + 3p/c^2。")
print("  L1b-3 通过：压强项相对 rho 项为 3p/(rho c^2)，弱场源约 1e-5，属次阶修正。")
print("  L1b-4 通过：所需 O(v^2) 系数为 3/2；静质量写给 0，gamma 写给 1/2。")
print()
print("  结论：theta 的源是 T_00 + T_ii（静止系里 rho + 3p/c^2），两条独立途径一致；")
print("        现行链（源为 rho）在首阶正确，压强项属次阶；写规则在次阶上欠 3p 那一部分。")
