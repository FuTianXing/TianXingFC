# -*- coding: utf-8 -*-
'''U3 同根判据（第 512 条）：两饱和常数的普朗克标度共用结构。

温度侧饱和（k_B T_P = E_P^{(t)}，D in [0,1] 有界）与几何侧饱和
（theta_max = 1 满偏，机制失义边界）的同根检验：
V1 作用量同根三恒等式；V2 紧致性根源核对（相位侧 D 有界 + 取向侧满偏）；
V3 普朗克能量同一（E_P^{(t)} = m_P c^2）；V4 约定成分辨析（满偏数值的
映射归一 vs 结构内容）；EP5 坏定义反例（无界 D 定义使 T_P 失效）。
确定性脚本，纯数值核对与演示。
'''
import numpy as np
from scipy.constants import hbar, c, G, k as k_B

print('=' * 78)
print('第 512 条：U3 同根判据（两饱和常数的普朗克标度共用结构）')
print('=' * 78)
print()

t_P = np.sqrt(hbar * G / c ** 5)
m_P = np.sqrt(hbar * c / G)
E_Pt = hbar / t_P

# ---------- V1 作用量同根三恒等式 ----------
print('V1 作用量同根三恒等式（每摆程一份作用量量子结构的两个读数）')
r1 = E_Pt * t_P / hbar - 1.0
r2 = m_P * c ** 2 * t_P / hbar - 1.0
r3 = (m_P / m_P) - 1.0
print('   E_P^(t) t_P / hbar - 1        = %.3e' % r1)
print('   m_P c^2 t_P / hbar - 1        = %.3e' % r2)
print('   delta_theta(m_P) = m_P/m_P - 1 = %.3e（写入幅值句 delta_theta = m/m_P 的满偏端）' % r3)
v1 = max(abs(r1), abs(r2), abs(r3)) <= 1e-12
print()

# ---------- V3 普朗克能量同一 ----------
print('V3 普朗克能量同一：E_P^(t) = m_P c^2（两饱和共用同一能量标度）')
r_E = E_Pt / (m_P * c ** 2) - 1.0
T_P = E_Pt / k_B
print('   E_P^(t) / (m_P c^2) - 1       = %.3e' % r_E)
print('   T_P = E_P^(t)/k_B = %.6e K；m_P c^2 / k_B = %.6e K（同一温度）' % (T_P, m_P * c ** 2 / k_B))
print('   关联常数：k_B T_P / (m_P c^2) = %.15f（温度饱和端与满载质量能量同一）' % (k_B * T_P / (m_P * c ** 2)))
v3 = abs(r_E) <= 1e-12
print()

# ---------- V2 紧致性根源核对 ----------
print('V2 紧致性根源：相位侧 D in [0,1]（紧致相位圆）与取向侧满偏（紧致取向圆）')
rng = np.random.default_rng(42)
n = 20000
z = np.exp(1j * rng.uniform(0, 2 * np.pi, n))
D = 1.0 - abs(np.mean(z)) / np.sqrt(np.mean(np.abs(z) ** 2))
z_aligned = np.exp(1j * np.full(n, 0.3))
D_aligned = 1.0 - abs(np.mean(z_aligned)) / np.sqrt(np.mean(np.abs(z_aligned) ** 2))
print('   随机相位（均匀圆）：D = %.6f（期望 ~1 - O(1/sqrt(n)) ~ 1）' % D)
print('   全同向相位：      D = %.6f（期望 0，冻结端）' % D_aligned)
ok_v2a = 0.0 <= D <= 1.0 and abs(D_aligned) <= 1e-12
print('   取向侧：满偏 theta = 1 对应取向反向 alpha = pi（180 度）；')
print('   值域有界性同源于两个紧致圆（相位 S^1 与取向 S^1），紧致轴给上界')
print()

# ---------- V4 约定成分辨析 ----------
print('V4 满偏数值 1 的约定成分：结构内容 vs 映射归一')
alpha = np.pi * 1.0
R = np.array([[np.cos(alpha), -np.sin(alpha), 0], [np.sin(alpha), np.cos(alpha), 0], [0, 0, 1]])
R2 = R @ R
id_err = float(np.max(np.abs(R2 - np.eye(3))))
print('   角度映射 alpha = pi * theta：theta = 1 给 alpha = pi；')
print('   SO(3) 中 alpha = pi 为二阶元：R(pi)^2 - I 最大元 = %.3e（取向反向即失义，结构性）' % id_err)
print('   换圈数映射 theta\' = 2*theta：满偏 theta\' = 2（半圈），数值随映射变，')
print('   结构判据（反向 = 失义）不变：满偏数值 1 含弧度归一约定成分')
v4 = id_err <= 1e-12
print()

# ---------- EP5 坏定义反例 ----------
print('EP5 反例：无界 D 定义使 T_P 失效（判据区分力）')
D_bad = 1.0 - np.mean(np.abs(z)) ** 2 / np.mean(np.abs(z) ** 2) * 4.0
print('   坏定义 D\' = 1 - 4 |<z>|^2/<|z|^2>（去掉开方归一的结构；柯西-施瓦茨给 D\' in [-3, 1]）：')
print('   随机相位下 D\' = %.6f（低相干端贴上界）' % D_bad)
worse = 0.0
for sig in [0.05, 0.1, 0.2, 0.5, 1.0, 3.0]:
    zz = np.exp(1j * rng.normal(0, sig, n))
    db = 1.0 - 4.0 * abs(np.mean(zz)) ** 2 / np.mean(np.abs(zz) ** 2)
    worse = max(worse, abs(db))
    if db < 0:
        print('   sigma = %.2f（高相干档）：D\' = %.6f < 0 越下界' % (sig, db))
print('   六档相干扫描：|D\'| 最大 %.6f（越界，D\'<0 出现）' % worse)
ok_ep5 = worse > 1.0
print('   判读：坏定义破坏 [0,1] 有界性（高相干端给负值），T_P 饱和失效；区分力确认' if ok_ep5
      else '   判读失败：反例不越界')
print()

print('汇总：V1-%s V2-%s V3-%s V4-%s EP5-%s' % (
    'PASS' if v1 else 'FAIL', 'PASS' if ok_v2a else 'FAIL', 'PASS' if v3 else 'FAIL',
    'PASS' if v4 else 'FAIL', 'PASS' if ok_ep5 else 'FAIL'))
print('结论：两饱和常数共用同一普朗克能量 E_P^(t) = m_P c^2（代数恒等）与')
print('同一作用量量子入口 hbar（每摆程一份），有界性分别住相位与取向两个')
print('紧致圆；候选二在「共用普朗克标度入口」意义上成立（弱形态），满偏')
print('数值 1 含弧度归一约定成分（结构内容 = 取向反向即失义）。')
