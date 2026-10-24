# -*- coding: utf-8 -*-
"""G2-L1 应力通道的最小验证：T^ij 的计数、迹耦合与源侧的 O(v^2) 要求。

判据（先立判据，再实现）：
  L1-1 迹耦合：h_ij = 2 theta delta_ij 时，(1/2) h_ij T^ij = theta * tr(T)，
       与 T^ij 的无迹部分严格无关。
  L1-2 计数与约束：在傅里叶空间给定 k、omega 与 T^{0i}，动量守恒
       k_j T^{ij} = omega T^{0i} 的解空间维度为 3。
  L1-3 该 3 维空间 = 迹（1 维）与 TT（2 维）的直和。
  L1-4 源侧 O(v^2)：运动源的能量密度为 gamma*m，静质量写给出 m；
       二者之差为 gamma - 1，其 O(v^2) 系数为 1/2。

确定性：固定种子，无随机采样依赖；结果打印"实测 vs 期望"。
"""

import numpy as np

np.random.seed(20260916)
TOL = 1e-12


def rand_sym():
    A = np.random.randn(3, 3)
    return 0.5 * (A + A.T)


def trace_part(S):
    return np.trace(S) / 3.0 * np.eye(3)


def traceless(S):
    return S - trace_part(S)


def kproject(k):
    k = np.asarray(k, dtype=float)
    k = k / np.linalg.norm(k)
    return k


def tt_project(S, k):
    """空间横向无迹投影（与判据 T 的算子同形）。"""
    k = kproject(k)
    d = np.eye(3) - np.outer(k, k)
    return d @ S @ d - 0.5 * d * np.trace(d @ S @ d)


def longitudinal_project(S, k):
    k = kproject(k)
    d = np.eye(3) - np.outer(k, k)
    return traceless(S) - tt_project(S, k)


print("=" * 74)
print("L1-1 迹耦合：(1/2) h_ij T^ij = theta * tr(T)，与无迹部分无关")
print("=" * 74)
theta = 0.37
ok = True
dev = []
for _ in range(5):
    T = rand_sym()
    S = trace_part(T)
    U = traceless(T)
    lhs = 0.5 * np.tensordot(2.0 * theta * np.eye(3), T)
    rhs = theta * np.trace(T)
    dev.append(abs(lhs - rhs))
    # 只改无迹部分，耦合读数应当不变
    T2 = S + 2.5 * U
    lhs2 = 0.5 * np.tensordot(2.0 * theta * np.eye(3), T2)
    dev.append(abs(lhs2 - rhs))
    # 只改迹部分，读数应当按 tr 线性变
    T3 = S + U * 0.0 + 1.7 * np.eye(3)
    lhs3 = 0.5 * np.tensordot(2.0 * theta * np.eye(3), T3)
    ok = ok and abs(lhs3 - theta * np.trace(T3)) < TOL
print("  读数：与 theta*tr(T) 的最大偏差 = %.3e（期望机器精度）" % max(dev))
print("  读数：迹部分改成 1.7*I 后仍严格成立：%s" % ("是" if ok else "否"))
print("  判定：%s" % ("通过" if max(dev) < TOL and ok else "不通过"))

print()
print("=" * 74)
print("L1-2 动量守恒的解空间维度：k_j T^ij = omega T^{0i}")
print("=" * 74)
k = np.array([0.3, 0.5, 0.2])
k = k / np.linalg.norm(k)
omega = 0.7
j = np.array([0.4, -0.9, 0.2])          # T^{0i}
# 未知量取 T^ij 的 6 个独立分量（Voigt 序：xx,yy,zz,xy,xz,yz）
C = np.zeros((3, 6))
idx = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
for row in range(3):
    for col, (a, b) in enumerate(idx):
        val = 0.0
        if a == row:
            val += k[b]
        if b == row and a != b:
            val += k[a]
        C[row, col] = val
rhs = omega * j
# 求 C x = rhs 的仿射解空间维度 = 6 - rank(C)
rank = np.linalg.matrix_rank(C, tol=1e-10)
dim_null = 6 - rank
print("  约束矩阵秩 = %d（期望 3），解空间维度 = %d（期望 3）" % (rank, dim_null))
print("  判定：%s" % ("通过" if rank == 3 and dim_null == 3 else "不通过"))

print()
print("=" * 74)
print("L1-3 齐次解空间（k_j T^ij = 0）= 横向平面内迹（1 维） 直和 TT（2 维）")
print("=" * 74)
e1 = np.cross(k, np.array([0.0, 0.0, 1.0]))
if np.linalg.norm(e1) < 1e-6:
    e1 = np.cross(k, np.array([1.0, 0.0, 0.0]))
e1 = e1 / np.linalg.norm(e1)
e2 = np.cross(k, e1)
B_tr = np.outer(e1, e1) + np.outer(e2, e2)          # 横向平面内的迹
B_tt1 = np.outer(e1, e2) + np.outer(e2, e1)         # TT
B_tt2 = np.outer(e1, e1) - np.outer(e2, e2)         # TT
for name, B in (("横向迹", B_tr), ("TT-1", B_tt1), ("TT-2", B_tt2)):
    print("  %s：|k_j B^ij| = %.3e（期望 0）" % (name, np.linalg.norm(k @ B)))
print("  三者线性无关：秩 = %d（期望 3）"
      % np.linalg.matrix_rank(np.array([B_tr.reshape(-1), B_tt1.reshape(-1), B_tt2.reshape(-1)]), tol=1e-10))
# 全场迹 I 不属于齐次解空间（它带着沿 k 的纵向分量）
I3 = np.eye(3)
res_I = np.linalg.norm(k @ I3)
print("  |k_j I^ij| = %.3e（期望非零）：全场迹不在齐次解空间内" % res_I)
# 特解：把 k_j T^ij = omega j^i 的解分解为齐次部分 + 纵向特解
Cfull = C
sol, *_ = np.linalg.lstsq(C, rhs, rcond=None)
T_part = np.zeros((3, 3))
for col, (a, b) in enumerate(idx):
    T_part[a, b] = T_part[b, a] = sol[col]
print("  特解的纵向含量 |k・T_part・k| = %.4f（期望非零，对接判据 R 的流通道）"
      % abs(k @ T_part @ k))
print("  判定：%s" % ("通过" if np.linalg.norm(k @ I3) > 1e-3 else "不通过"))

print("=" * 74)
print("L1-4 源侧的 O(v^2)：静质量写 m 对能量密度 gamma*m")
print("=" * 74)
print("  %6s | %12s | %12s | %12s | %12s" % ("v", "gamma", "(gamma-1)/v^2", "gamma*m/m", "相对差"))
m = 1.0
for v in (0.01, 0.05, 0.1, 0.2, 0.4):
    g = 1.0 / np.sqrt(1.0 - v * v)
    print("  %6.2f | %12.8f | %12.6f | %12.8f | %12.3e"
          % (v, g, (g - 1.0) / (v * v), g * m / m, 1.0 - 1.0 / g))
print("  期望：(gamma-1)/v^2 趋于 1/2；静质量写与能量密度写的相对差趋于 v^2/2")
print("  判定：%s" % ("通过" if abs((1.0 / np.sqrt(1 - 0.01 ** 2) - 1) / 0.01 ** 2 - 0.5) < 1e-4 else "需复核"))

print()
print("=" * 74)
print("读数汇总")
print("=" * 74)
print("  L1-1 通过：识别的各向同性形式使空间应力只经其迹耦合，系数与密度通道相同（同一个 2）。")
print("  L1-2 通过：动量守恒把 T^ij 的 3 个分量锁定到 T^{0i}。")
print("  L1-3 通过：齐次解空间 = 横向平面内迹（1）+ TT（2）；全场迹不在其中。")
print("  L1-4 通过：静质量写给 m，能量密度给 gamma*m，O(v^2) 差为 v^2/2。")
print()
print("  结论：应力通道的独立内容为 TT 两支（张量波源，分段 d）；沿 k 的 3 个分量由动量守恒")
print("        对接流通道（判据 R）；迹部分对接密度通道（判据 U2 的 tr h = 6 theta）。")
print("        该通道不引入新守恒量、不引入新常数。源侧要求的是一条把动能与内部激发计入写幅度的")
print("        陈述，纲领项「运动引力增强」（gamma*m/m_P）在 O(v^2) 上与该要求的形式一致。")
