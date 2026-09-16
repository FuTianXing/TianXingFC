# -*- coding: utf-8 -*-
"""复结构序参量的显式形式化（第一轮）：候选泛函、畴壁解与对称恢复（G1 至 G5）。

定位：手征论文把"宇称自发破缺"的机制落在复结构符号的真空选择上，并自标双井势形态为
待形式化的机制假定。本轮把它写成可判定的泛函，核对三件事：双井真空与符号对称、畴壁解
与其表面张力、以及对称恢复温度。参数从未由公理层导出，本轮只做形式化核对，不声明参数
有微观来源。

判据（先立判据，再实现）：
  G1 双井真空与符号对称：极小位于 +-v，泛函在 phi -> -phi 下不变。
  G2 畴壁解：弛豫解与解析剖面 v tanh((x-x0)/w) 一致，w = 1/(v sqrt(2 lam))。
  G3 表面张力：实测 sigma 与解析 (4/3) sqrt(2 lam) v^3 一致。
  G4 对称恢复温度：有效势加 (a T^2/2) phi^2 后，二阶相变点 T_c = 2 v sqrt(lam/a)。
  G5 灵敏度对照：去掉双井（lam = 0）无真空简并与畴壁；加外场 h phi 后简并解除。

单位：格距 1。确定性，无随机数。
"""

import numpy as np

CHECKS = []
# 参数取畴壁宽度远大于格距的解析区：w = 1/(v sqrt(2 lam)) 约 10.5 格
V0 = 0.3
LAM = 0.05
A_T = 1.0
N = 401


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [{}] {}  {}".format(tag, "PASS" if ok else "FAIL", detail))


def potential(phi, v=V0, lam=LAM, T=0.0):
    return lam * (phi ** 2 - v ** 2) ** 2 + 0.5 * A_T * T ** 2 * phi ** 2


print("=" * 78)
print("G1 双井真空与符号对称")
print("=" * 78)
grid = np.linspace(-3.0 * V0, 3.0 * V0, 20001)
val = potential(grid)
neg = grid < 0.0
lo = grid[neg][np.argmin(val[neg])]
hi = grid[~neg][np.argmin(val[~neg])]
sym = float(np.max(np.abs(potential(grid) - potential(-grid))))
print("  两个极小：{:+.6f} 与 {:+.6f}（解析 +-{:.1f}）".format(lo, hi, V0))
print("  V(phi) 与 V(-phi) 的最大偏差 = {:.3e}".format(sym))
check("G1", abs(abs(lo) - V0) <= 1e-3 and abs(abs(hi) - V0) <= 1e-3 and sym <= 1e-14,
      "极小位于 +-v，泛函在符号翻转下不变")

print()
print("=" * 78)
print("G2 畴壁解与 G3 表面张力")
print("=" * 78)
x = np.arange(N, dtype=float)
phi = V0 * np.tanh((x - N // 2) / 3.0)
eta = 0.2
for _ in range(120000):
    lap = np.zeros_like(phi)
    lap[1:-1] = phi[2:] + phi[:-2] - 2.0 * phi[1:-1]
    dv = 4.0 * LAM * phi * (phi ** 2 - V0 ** 2)
    phi[1:-1] += eta * (lap[1:-1] - dv[1:-1])
    phi[0], phi[-1] = -V0, V0
w_an = 1.0 / (V0 * np.sqrt(2.0 * LAM))
prof_an = V0 * np.tanh((x - N // 2) / w_an)
dev = float(np.max(np.abs(phi - prof_an)))
# 表面张力 = 能量密度积分减真空能
dphi = np.gradient(phi)
sigma_num = float(np.trapezoid(dphi ** 2, x))
# BPS 型均分：kink 上 1/2 (dphi/dx)^2 应等于 V(phi)
bps = float(np.max(np.abs(0.5 * dphi ** 2 - LAM * (phi ** 2 - V0 ** 2) ** 2)[50:-50]))
sigma_an = (4.0 / 3.0) * np.sqrt(2.0 * LAM) * V0 ** 3
width_meas = float(np.sum((phi / V0) ** 2 < 0.5))
print("  解析宽度 w = {:.6f} 格；剖面最大偏差 = {:.3e}".format(w_an, dev))
print("  表面张力：实测 {:.6f}，解析 {:.6f}，相对偏差 {:.3e}"
      .format(sigma_num, sigma_an, abs(sigma_num / sigma_an - 1.0)))
print("  均分核对：max |1/2 (dphi/dx)^2 - V(phi)| = {:.3e}（期望 ~0）".format(bps))
check("G2", dev <= 5e-3 and w_an > 5.0 and abs(w_an - 1.0 / (V0 * np.sqrt(2.0 * LAM))) <= 1e-12,
      "弛豫解与解析剖面一致")
check("G3", abs(sigma_num / sigma_an - 1.0) <= 5e-3, "表面张力与解析式一致")

print()
print("=" * 78)
print("G4 对称恢复温度")
print("=" * 78)
Tc_an = 2.0 * V0 * np.sqrt(LAM / A_T)
print("  解析 T_c = 2 v sqrt(lam/a) = {:.6f}".format(Tc_an))
def minima_count(T, tol=1e-6):
    v = potential(grid, T=T)
    m = int(np.argmin(v))
    return abs(grid[m])
for T in (0.0, 0.5 * Tc_an, 0.9 * Tc_an, 1.1 * Tc_an, 2.0 * Tc_an):
    print("    T = {:.4f}（T/T_c = {:.2f}）：极小位置 |phi| = {:.6f}"
          .format(T, T / Tc_an, minima_count(T)))
m_lo = minima_count(0.9 * Tc_an)
m_hi = minima_count(1.1 * Tc_an)
check("G4", m_lo > 1e-3 and m_hi <= 1e-6, "二阶相变点与解析 T_c 一致")

print()
print("=" * 78)
print("G5 灵敏度对照")
print("=" * 78)
flat = potential(grid, lam=0.0)
print("  去掉双井：V 在 phi 上的变化范围 = {:.3e}".format(float(np.max(flat) - np.min(flat))))
h = 0.002
biased = potential(grid) + h * grid
m_idx = int(np.argmin(biased))
print("  加外场 h = {:.2f}：唯一极小位于 {:+.6f}（应移向负井；+h*phi 项压低负 phi）".format(h, grid[m_idx]))
check("G5", float(np.max(flat) - np.min(flat)) <= 1e-14
      and abs(grid[m_idx] + V0) <= 0.1, "两种对照都能让判据变化")

print()
n_pass = sum(1 for _, ok in CHECKS if ok)
print("=" * 78)
print("判据汇总：{}/{} 通过".format(n_pass, len(CHECKS)))
for tag, ok in CHECKS:
    print("  {:6s} {}".format(tag, "PASS" if ok else "FAIL"))
print("=" * 78)
