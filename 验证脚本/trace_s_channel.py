# -*- coding: utf-8 -*-
"""迹型 s-型组合的 O(1) 归一化因子：约定因子、桥因子与两条标定路线（判据 T1 至 T4）。
模型：单位胞三键（沿 x/y/z），键能 E = (1/2) k sum_i (delta d_i)^2，格距 a。
本脚本做三件事：复算弹性常数；给出两种写法的约定因子；把桥因子钉到记录-能量换算的两条标定。
确定性，无随机数，无图。
"""
import numpy as np

def energy_density(kind, strain, k=1.0, a=1.0, precision=1e-7):
    """键能模型的应变能密度。kind = 3（三键胞）或 6（六键胞，双计数）。"""
    bonds = [np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])]
    if kind == 6:
        bonds = bonds + bonds
    eps = np.zeros((3, 3))
    eps[0, 0], eps[1, 1], eps[0, 1], eps[1, 0] = strain
    u = 0.0
    for n in bonds:
        d = a * n
        dd = eps @ d
        u += 0.5 * k * float(dd @ dd)
    return u / a ** 3

def stiffness(kind=3, k=1.0, a=1.0):
    eps = 1e-5
    # C11：(1,1,0,0) 方向
    u1 = energy_density(kind, (eps, 0, 0, 0), k, a)
    u2 = energy_density(kind, (eps, eps, 0, 0), k, a)
    u3 = energy_density(kind, (0, 0, eps, eps), k, a)
    C11 = 2.0 * u1 / eps ** 2
    C12 = (u2 - u1 - u1) / eps ** 2
    C44 = u3 / (2.0 * eps ** 2)
    return C11, C12, C44

print("=" * 72)
print("迹型 s-型组合的 O(1) 因子（判据 T1 至 T4）")
print("=" * 72)
C3 = stiffness(3)
C6 = stiffness(6)
print("T1 复算弹性常数（三键胞，单位取 k/a）")
print(f"   C11 = {C3[0]:.6f}（期望 1.000000），C12 = {C3[1]:+.2e}（期望 0），C44 = {C3[2]:.6f}（期望 0.500000）")
tr3 = C3[0] - C3[1] - 2 * C3[2]
print(f"   各向同性组合 C11 - C12 - 2C44 = {tr3:+.2e}（期望 0）")

print("T2 两种写法的约定因子")
print(f"   六键胞读数 (C11, C12, C44) = ({C6[0]:.6f}, {C6[1]:+.2e}, {C6[2]:.6f}) = 2 x 三键胞")
r_conv = C6[0] / C3[0]
print(f"   约定因子（六键/三键）= {r_conv:.6f}（期望 2.000000）")
print(f"   无迹组合在两种写法下均为零：{C6[0]-C6[1]-2*C6[2]:+.2e}")
tr_comb3 = C3[0] + 2 * C3[1]
print(f"   迹型组合（三键写法）C_TT = C11 + 2C12 = {tr_comb3:.6f} k/a（与 C11 同值，因 C12 = 0）")

print("T3 桥因子的两条标定（a = l_P，能量刻度 E_P/l_P^3）")
# 路线一：场能匹配 K_field = 3 k l_P^2 / a = c^4/(4 pi G) = E_P/(4 pi l_P)
k_field = 1.0 / (12.0 * np.pi)      # 以 E_P/l_P^2 为单位
K_field = 3.0 * k_field / 1.0       # = 1/(4 pi)（以 E_P/l_P 为单位）
s_route1 = (k_field / 1.0)          # C11 = k/a -> s = C11/(E_P/l_P^3) = k/(E_P/l_P^2)
print(f"   路线一（场能匹配）：k = E_P/(12 pi l_P^2) = {k_field:.6f} E_P/l_P^2；K_field = {K_field:.6f} E_P/l_P（期望 1/(4 pi) = {1/(4*np.pi):.6f}）")
print(f"     迹型读数 s_TT = {s_route1:.6f}（期望 1/(12 pi) = {1/(12*np.pi):.6f}）")
# 路线二：满量程（满偏 delta_h = l_P 时键能 = E_P，即 (1/2) k l_P^2 = E_P）
k_full = 2.0
s_route2 = k_full
print(f"   路线二（满量程）：k = 2 E_P/l_P^2 = {k_full:.6f} E_P/l_P^2；迹型读数 s_TT = {s_route2:.6f}")
ratio = s_route2 / s_route1
print(f"   两条路线之比 = {ratio:.4f}（= 24 pi）")

print("T4 与上界的并列（引力侧 s_TT 上界 1e-5，汇总表 S5）")
bound = 1e-5
for name, s in [("路线一", s_route1), ("路线二", s_route2)]:
    print(f"   {name}：s_TT = {s:.4f}，超出上界 {np.log10(s/bound):.2f} 个量级（比值 {s/bound:.3e}）")
print("   口径结论：迹型组合是各向同性部分，与 G 的标定共用入口（GV7）；其绝对值已被 s_eff 链钉死，")
print("   故与 s_TT 上界的并列须按归一化通道处理，不得读成破缺预言（协议侧待核）。")
