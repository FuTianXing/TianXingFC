# -*- coding: utf-8 -*-
"""insertion_slaving.py

奴役机制与插入的三种读法：格距绝对性给出奴役，晚期的三种处理各给一个背景史。
判据 P1 至 P5 先写后实现。
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

Om, OL, H0 = 0.315, 0.685, 67.4     # km/s/Mpc
z = np.linspace(0.0, 1.2, 601)

def H_LCDM(z): return np.sqrt(Om*(1+z)**3 + OL)
def H_max(z):  return np.maximum(np.sqrt(Om*(1+z)**3), 1.0)
def H_slaved(z): return np.sqrt(Om*(1+z)**3)

print('='*74); print('奴役机制与插入的三种读法'); print('='*74)
# P1 奴役：格距绝对值 lP -> 格点数正比物理体积 -> Ndot/N = 3H
print('P1  奴役机制：格距绝对性给 Ndot/N = 3H')
print('   格点数密度 n = 1/lP^3 为绝对量，物理体积按 a^3 走，')
print('   故要维持格距，插入率必须恰为 3H：Ndot/N = d(ln a^3)/dt = 3H')
print('   核对：a 正比 t^n 时 Ndot/N = 3n/t，与 3H = 3n/t 逐位相同（恒等）')

# P2 三种读法的背景史
print('P2  三种读法的 H(z)/H0（取 z = 0.3 与 z = 0）')
for name, f in (('Lambda 型（插入贡献能量密度）', H_LCDM), ('max 型（H = max(H_content, H0)）', H_max), ('纯奴役型（无晚期源）', H_slaved)):
    print('   %-28s H(0.3)/H0 = %.4f   H(0)/H0 = %.4f' % (name, f(np.array([0.3]))[0], f(np.array([0.0]))[0]))

print('P3  max 型的转折点与偏离')
zstar = (1.0/Om)**(1.0/3.0) - 1.0
print('   max 型转折 z* = %.4f（Lambda 型的等分点 z = %.4f）'
      % (zstar, (OL/Om)**(1.0/3.0) - 1.0))
print('   z = 0.3 处 max 型低 %.1f%%，属可判偏离（宇宙钟与 BAO 的 H(z) 单点误差约百分之几）'
      % (100*(1 - H_max(np.array([0.3]))[0]/H_LCDM(np.array([0.3]))[0])))

print('P4  纯奴役型被预算闭合排除')
print('   无晚期源时 H(0)/H0 = sqrt(Om) = %.4f；要合上观测 H0 须 Om = 1（爱因斯坦-德西特），' % np.sqrt(Om))
print('   与 CMB 给的 Om = 0.315 冲突，差 %.1f%%' % (100*(1-np.sqrt(Om))))

print('P5  Lambda 型与观测一致，且给出框架的独有内容')
print('   插入若贡献一个常能量密度项，则 H^2 = H0^2(Om(1+z)^3 + OL)，背景与 LambdaCDM 同形；')
print('   框架的独有内容转为「常能量密度项的机制」，即插入的应力能量。')

# ---------- 图 ----------
fig, ax = plt.subplots(1, 1, figsize=(7.8, 5.0))
ax.plot(z, H_LCDM(z), lw=2.4, color='seagreen', label='Lambda 型（与 LambdaCDM 同形）')
ax.plot(z, H_max(z), lw=2.4, color='crimson', ls='--', label='max 型（$z<z_*$ 纯德西特）')
ax.plot(z, H_slaved(z), lw=2.4, color='gray', ls=':', label='纯奴役型（无晚期源）')
ax.axvline(zstar, color='crimson', ls=':', lw=1.0)
ax.axhline(1.0, color='gray', lw=0.8)
ax.annotate('$z_* = %.3f$' % zstar, xy=(zstar, 1.02), xytext=(zstar+0.06, 0.86),
            fontsize=9, color='crimson')
ax.annotate('$z = 0.3$ 处低 %.0f%%' % (100*(1-H_max(np.array([0.3]))[0]/H_LCDM(np.array([0.3]))[0])),
            xy=(0.3, 1.0), xytext=(0.36, 1.12), fontsize=9, color='crimson')
ax.set_xlabel('$z$'); ax.set_ylabel('$H(z)/H_0$')
ax.set_title('插入的三种读法给出的背景膨胀史')
ax.legend(fontsize=8.5); ax.grid(alpha=0.3)
fig.tight_layout()
out = 'D:/software/TianXingFC/images/insertion_slaving_readings.png'
fig.savefig(out, dpi=150)
print('图已存：', out)
