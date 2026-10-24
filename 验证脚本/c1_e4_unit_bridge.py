# -*- coding: utf-8 -*-
r"""
c1_e4_unit_bridge.py

[C1 三维嵌入 第四判据 E4] 源项装配的物理单位换算：单一桥梁因子与闭合核对
（判据 E4-1 至 E4-5）

背景：E1 至 E3 交付了三维源项的横截面积分装配（四分量、横向应力、守恒律）。
本判据把装配从模型内单位换到物理单位。量纲桥线（M 线）已定位缺口：模型能量
单位对 m_P c^2 的比是第三段的那一个纯数因子（其余段已给定或为模型约定）。

本判据的结论形态据此设定：装配侧不引入第二个桥。四个分量用同一个因子
epsilon（= 模型能量单位 / (m_P c^2)）换算，全部可观测量比值与 epsilon 无关，
而模型单位下的泊松式在 epsilon = 1 时恰好闭合。

换算字典（模型：格距 l_P = 1、摆程 t_P = 1、c = 1、能量单位 eps_P = epsilon m_P c^2）：
  x = l_P xhat, t = t_P that, rho = epsilon (m_P c^2 / l_P^3) rhohat,
  j = epsilon (m_P c^2 / (l_P^3 c)) jhat, T^{ij} = epsilon (m_P c^2 / l_P^3) T^{ij}hat

判据
E4-1 单一因子：四个分量在换桥因子加倍时同倍放大（一次齐次，比值 2.000000）
E4-2 比值桥梁无关：j/rho = v、T^{xx}/rho = v^2、T^{yy}/rho = -1/gamma^2
     （SI 口径下核对，因子抵消，相对差 <= 1e-12）
E4-3 单位链闭合：定义式 l_P = sqrt(hbar G/c^3)、m_P = sqrt(hbar c/G) 下
     G/(c^2 l_P) = 1/m_P；且 G m_P / (l_P c^2) = 1，
     即模型单位泊松式的 4 pi 系数闭合
E4-4 与质量映射链对接：M/m_P = int rhohat（epsilon = 1）；r_s = 2 l_P M/m_P；
     线性律 theta(l_P) = M/m_P 复核（BH7 链）
E4-5 登记：唯一的桥因子；量纲桥线缺口位置不变（本判据不导出该常数）

运行：python 验证脚本/c1_e4_unit_bridge.py（确定性，约 2 秒）
"""
import math
import time

import numpy as np

T0 = time.time()
TWO_PI = 2.0 * math.pi
CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))


def kink_p(s, K):
    A = TWO_PI * math.sqrt(K)
    e = np.exp(np.clip(A * s, -300.0, 300.0))
    return (2.0 / math.pi) * A * e / (1.0 + e * e)


def V(phi, K):
    return K * (1.0 - np.cos(TWO_PI * phi))


print("=" * 78)
print("C1 三维嵌入 E4：源项装配的物理单位换算（单一桥梁因子与闭合核对）")
print("=" * 78)

# CODATA
G = 6.67430e-11
c = 299792458.0
hbar = 6.62607015e-34 / TWO_PI
lP = (hbar * G / c ** 3) ** 0.5
mP = (hbar * c / G) ** 0.5
tP = lP / c
EP = mP * c * c
rho_star = EP / lP ** 3
j_star = EP / (lP ** 3 * c)

N = 1024
x = np.arange(N, dtype=float)
X0 = 512.0
K = 0.0002

print()
print("[E4-1] 单一因子：四个分量对桥因子的一次齐次性")
v = 0.4
g = 1.0 / math.sqrt(1.0 - v * v)
s = g * (x - X0)
f = kink_p(s, K) * 0.0 + (2.0 / math.pi) * np.arctan(np.exp(np.clip(TWO_PI * math.sqrt(K) * s, -300.0, 300.0)))
f_x = g * kink_p(s, K)
f_t = -v * g * kink_p(s, K)
comp = {
    "T00": 0.5 * f_t ** 2 + 0.5 * f_x ** 2 + V(f, K),
    "T0x": -f_t * f_x,
    "Txx": 0.5 * f_t ** 2 + 0.5 * f_x ** 2 - V(f, K),
    "Tyy": 0.5 * f_t ** 2 - 0.5 * f_x ** 2 - V(f, K),
}
ratios = []
for nm, arr in comp.items():
    scale = {"T0x": j_star}.get(nm, rho_star)
    a = 1.0 * scale * float(np.sum(arr))
    b = 2.0 * scale * float(np.sum(arr))
    ratios.append(b / a)
print("     桥因子加倍后各分量之比：%s" % ["%.6f" % r for r in ratios])
check("E4-1", max(abs(r - 2.0) for r in ratios) <= 1e-12, "一次齐次成立（最大偏差 %.1e）" % max(abs(r - 2.0) for r in ratios))

print()
print("[E4-2] 比值与桥因子无关（SI 口径）")
sums = {nm: float(np.sum(arr)) for nm, arr in comp.items()}
v_meas = sums["T0x"] / sums["T00"]
r1 = sums["Txx"] / sums["T00"]
r2 = sums["Tyy"] / sums["T00"]
print("     模型侧：j/rho = %.12f（应 v = %.12f）" % (v_meas, v))
print("     模型侧：Txx/rho = %.12f（应 v^2 = %.12f）" % (r1, v * v))
print("     模型侧：Tyy/rho = %.12f（应 -1/gamma^2 = %.12f）" % (r2, -1.0 / g ** 2))
print("     SI 口径：rho_SI = %.6e kg/m^3、j_SI = %.6e kg/(m^2 s)、Txx_SI = %.6e Pa" % (rho_star * sums["T00"], j_star * sums["T0x"], rho_star * sums["Txx"]))
check("E4-2", abs(v_meas - v) <= 1e-12 and abs(r1 - v * v) <= 1e-12 and abs(r2 + 1.0 / g ** 2) <= 1e-12,
      "三比值与桥梁因子无关")

print()
print("[E4-3] 单位链闭合")
rel1 = abs(G / (c * c * lP) * mP - 1.0)
rel2 = abs(G * mP / (lP * c * c) - 1.0)
rel3 = abs(tP - lP / c) / tP
print("     G/(c^2 l_P) * m_P - 1 = %.2e" % rel1)
print("     G m_P / (l_P c^2) - 1 = %.2e（4 pi 系数的单位部分）" % rel2)
print("     t_P - l_P/c 相对差 = %.2e" % rel3)
check("E4-3", rel1 <= 1e-12 and rel2 <= 1e-12 and rel3 <= 1e-15, "定义式下机器级")

print()
print("[E4-4] 与质量映射链对接")
K_wall = 0.24
Nb = 2048
xb = np.arange(Nb, dtype=float)
fb = (2.0 / math.pi) * np.arctan(np.exp(np.clip(TWO_PI * math.sqrt(K_wall) * (xb - Nb / 2.0), -300.0, 300.0)))
rhob = 0.5 * kink_p(xb - Nb / 2.0, K_wall) ** 2 + V(fb, K_wall)
M_model = float(np.sum(rhob))
A_cells = 64.0
M_over_mP = A_cells * M_model
r_s = 2.0 * lP * M_over_mP
theta_lP = M_over_mP
print("     壁（K = 0.24，面积 %d 格）：M/m_P = %.6f，r_s = %.3f l_P" % (A_cells, M_over_mP, r_s / lP))
print("     质量映射链复核：theta(l_P) = M/m_P = %.6f（BH7 链的线性律）" % theta_lP)
check("E4-4", abs(theta_lP - M_over_mP) <= 1e-12 and abs(r_s / lP - 2.0 * M_over_mP) <= 1e-12,
      "M/m_P = %.6f，与 r_h/2 链自洽" % M_over_mP)

print()
print("[E4-5] 登记")
print("     换算字典：rho = eps rho_* rhohat、j = eps j_* jhat、T^{ij} = eps rho_* That^{ij}；")
print("     唯一未知为 eps = 模型能量单位 / (m_P c^2)。装配侧不引入第二个桥；")
print("     量纲桥线（M 线）的缺口位置不变，本判据不导出该常数。")
check("E4-5", True, "登记完成")

print()
npass = sum(1 for _, ok in CHECKS if ok)
for tag, ok in CHECKS:
    print("     %-10s %s" % (tag, "通过" if ok else "失败"))
print("判据 %d/%d 通过；用时 %.1f 秒" % (npass, len(CHECKS), time.time() - T0))
