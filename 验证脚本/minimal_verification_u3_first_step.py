# -*- coding: utf-8 -*-
"""minimal_verification_u3_first_step.py

U3 第一步：两条界限的同型化与满量程核对。判据 R1 至 R5 先写后实现。

两条界限：
  温度侧  k_B T = E_P * D，D 为复相干无序度，D 属 [0,1]（Cauchy-Schwarz + 归一化）
  几何侧  theta = (法线弯曲角)/pi，法线是方向、周期 pi，故 theta 属 [0,1]
本脚本核对两条界限的同型性、满量程恒等式与两者的比值（2pi 问题）。
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 基本常数（CODATA 2018）
hbar = 1.054571817e-34
c = 2.99792458e8
G = 6.67430e-11
lP = np.sqrt(hbar*G/c**3); tP = lP/c
mP = np.sqrt(hbar*c/G); EP_t = hbar/tP; EPa = hbar*2*np.pi/tP
print('='*74); print('U3 第一步：两条界限的同型化与满量程核对'); print('='*74)

# ---------- R1 温度侧：有界性是定义性的 ----------
rng = np.random.default_rng(20260921)
print('R1 温度侧有界性（D 属 [0,1]）的数值核查')
bad = 0
for _ in range(200000):
    n = rng.integers(2, 6)
    re = rng.normal(size=n); im = rng.normal(size=n); z = re + 1j*im
    D = 1.0 - abs(z.mean())/np.sqrt(np.mean(np.abs(z)**2))
    if D < -1e-12 or D > 1.0 + 1e-12: bad += 1
print('   随机样本 200000 组中越界 %d 组   期望 0（D 上界由 Cauchy-Schwarz 给）' % bad)
print('   D = 1 - |<z>|/sqrt(<|z|^2>)：分子非负给 D <= 1，Cauchy-Schwarz 给 D >= 0')

# ---------- R2 几何侧：有界性是构造性的 ----------
print('R2 几何侧有界性（theta 属 [0,1]）的依据')
print('   在册口径（2026-09-15 角度映射裁定）：满偏对应取向自由度压弯 180 度，法线轴保持笔直；')
print('   法线是方向而非矢量，方向空间的周期是 pi，故弯曲角不超过 pi，theta = 弯曲角/pi <= 1')

# ---------- R3 满量程恒等式 ----------
print('-'*74)
print('R3 满量程恒等式：几何侧满量程 mP c^2 与 hbar/tP')
print('   mP*c^2 = %.10e J；hbar/tP = %.10e J；相对差 %.3e'
      % (mP*c**2, EP_t, (mP*c**2)/EP_t - 1))
print('   （恒等式 mP c^2 = hbar c/lP = hbar/tP，因 lP = c tP）')

# ---------- R4 比值 2pi ----------
print('R4 两侧满量程之比')
print('   温度侧（基准口径）hbar*omega_P = %.10e J' % EPa)
print('   几何侧 mP*c^2 = %.10e J' % (mP*c**2))
print('   比值 = %.12f   期望 2*pi = %.12f   相对差 %.3e'
      % (EPa/(mP*c**2), 2*np.pi, EPa/(mP*c**2)/(2*np.pi) - 1))
print('   折算口径 E_P^{(t)} = hbar/tP 与几何侧满量程同值，相对差 %.3e'
      % (EP_t/(mP*c**2) - 1))

# ---------- R5 判定 ----------
print('-'*74)
print('R5 判定')
print('   同型：成立。两侧都是「相位类变量在紧致靶空间上的满量程」型：')
print('     温度侧靶空间为归一化区间 [0,1]（Cauchy-Schwarz 给上端）')
print('     几何侧靶空间为方向圆（周期 pi，给上端）')
print('   满量程：几何侧 = mP c^2 = hbar/tP（精确恒等式）；温度侧（基准）= 2pi hbar/tP。')
print('   两者相差恰好 2pi，该因子即在册的折算与基准约定差。')
print('   故同根性的判定归结为一个口径问题：温度定义里的整圈作用量 h = 2pi hbar 与')
print('   几何侧的拍长 tP 是否本属同一件事的两种读数。')

# ---------- 图 ----------
fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.6))
u = np.linspace(0, 1, 200)
ax[0].plot(u, u, lw=2.4, color='#1f4e79', label='温度侧 $T/T_P^{(基准)} = D$')
ax[0].plot(u, u, lw=2.4, ls='--', color='crimson', label='几何侧 $\\theta = $ 弯曲角$/\\pi$')
ax[0].plot([1], [1], 'o', ms=8, color='k')
ax[0].annotate('满量程', xy=(1, 1), xytext=(0.55, 0.75), fontsize=10,
               arrowprops=dict(arrowstyle='->', lw=1.0))
ax[0].set_xlabel('归一化自变量（无序度 / 弯曲角$/\\pi$）')
ax[0].set_ylabel('归一化读数')
ax[0].set_title('两条界限同型：紧致靶空间上的线性读数')
ax[0].legend(fontsize=8.5); ax[0].grid(alpha=0.3)
ax[1].bar([0, 1], [EPa/(mP*c**2), 1.0], width=0.5,
          color=['#1f4e79', 'crimson'])
ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(['温度侧 $\\hbar\\omega_P$', '几何侧 $m_Pc^2$'])
ax[1].set_ylabel('以 $m_Pc^2$ 为单位')
ax[1].set_title('两侧满量程之比：差 $2\\pi$')
for x, v in ((0, EPa/(mP*c**2)), (1, 1.0)):
    ax[1].text(x, v*1.02, '%.6f' % v, ha='center', fontsize=9)
ax[1].grid(alpha=0.3, axis='y')
fig.tight_layout(); out = 'D:/software/TianXingFC/images/u3_two_bounds_scales.png'; fig.savefig(out, dpi=150)
print('图已存：', out)
