# -*- coding: utf-8 -*-
"""minimal_verification_lorentz_damage_scan.py

D3 轮：第四档损坏（洛伦兹结构）的定义与扫描。判据 L1 至 L5 先写后实现。
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

EP_t = 1.2209e19*1e9           # 折算普朗克能量（eV），E_P^{(t)} = hbar/t_P
hPi2_6 = np.pi**2/6.0
print('='*74); print('D3：第四档损坏（洛伦兹结构）的定义与扫描'); print('='*74)

# ---------- L1 系数由格距唯一给定（两条口径一致） ----------
print('L1 系数口径：折算约定 1/24 配 E_P^{(t)}，基准约定 pi^2/6 配 E_P')
print('   1/24*(2pi)^2 = %.6f 对 pi^2/6 = %.6f   判：两口径同值'
      % ((2*np.pi)**2/24.0, np.pi**2/6.0))

# ---------- L2 各向异性比 6:3:2 ----------
dirs = {'[100]': 1.0, '[110]': 0.5, '[111]': 1.0/3.0}
vals = {k: hPi2_6*v for k, v in dirs.items()}
print('L2 各向异性比（sum k_a^4/k^4 的三个方向值 1 : 1/2 : 1/3）')
for k, v in vals.items():
    print('   %-6s delta_omega/omega 系数 = %.6f' % (k, v))
r = vals['[100]']/vals['[111]']
print('   [100]:[111] 比 = %.4f   期望 3.0000（即 6:2，三个方向合写为 6:3:2）' % r)

# ---------- L3/L4 无隙与次光速 ----------
k = np.logspace(-6, -1, 200)                  # 以 1/lP 为单位
w = k*(1 - (1.0/24.0)*1.0*k**2)               # [100] 方向
print('L3 无隙性：omega(k) 在 k -> 0 的比值 %.6f   期望 -> 1（无隙）' % (w[0]/k[0]))
print('L4 次光速性：delta_omega/omega 的符号 %s（在册 §6.1 全域次光速定理）'
      % ('负' if w[0]-k[0] < 0 else '正'))

# ---------- L5 敏感度扫描 ----------
print('-'*74)
print('L5 敏感度扫描：若 LV 系数随场强增强为 (1-2theta)^{-n}，找可观测阈值对应的 theta')
def theta_crit(n, E_eV, thresh):
    base = hPi2_6*(E_eV/EP_t)**2
    need = thresh/base
    x = need**(-1.0/n)             # 1-2theta
    return x, 0.5*(1-x)
for E_eV, name in ((1e3, '1 keV'), (1e4, '10 keV')):
    for thresh in (1e-20, 1e-11):
        base = hPi2_6*(E_eV/EP_t)**2
        print('   %s，参考线 %.0e：基准破坏 %.3e，所需增强 %.3e' % (name, thresh, base, thresh/base))
        for n in (1, 5, 10, 20, 50):
            x, th = theta_crit(n, E_eV, thresh)
            print('      n = %2d -> 1-2theta = %.3e（theta = %.9f，距饱和 %.3e）' % (n, x, th, 0.5-th))

# ---------- 图 ----------
fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.8))
E_eV = 1e3
ns = np.linspace(0.5, 60, 400)
for thresh, col in ((1e-20, 'crimson'), (1e-11, 'darkorange')):
    th = [theta_crit(n, E_eV, thresh)[1] for n in ns]
    ax[0].plot(ns, 0.5-np.array(th), lw=2.0, color=col, label='参考线 %.0e' % thresh)
ax[0].set_yscale('log'); ax[0].set_xlabel('增强指数 $n$')
ax[0].set_ylabel('$0.5-\\theta$（距饱和的距离）')
ax[0].set_title('1 keV 光子：洛伦兹破坏达可观测级的边界')
ax[0].legend(fontsize=8.5); ax[0].grid(alpha=0.3)
kk = np.logspace(-7, -0.7, 300)
ax[1].loglog(kk, 1 - (1/24.0)*kk**2, lw=2.2, color='#1f4e79', label='[100]')
ax[1].loglog(kk, 1 - (1/48.0)*kk**2, lw=2.2, color='#2e7d32', label='[110]')
ax[1].loglog(kk, 1 - (1/72.0)*kk**2, lw=2.2, color='crimson', label='[111]')
ax[1].axhline(1.0, color='gray', lw=0.8)
ax[1].set_xlabel('$k\\,l_P$'); ax[1].set_ylabel('$\\omega/(ck)$')
ax[1].set_title('三方向的首阶洛伦兹破坏（比值 6:3:2）')
ax[1].legend(fontsize=8.5); ax[1].grid(alpha=0.3, which='both')
fig.tight_layout(); out = 'D:/software/TianXingFC/images/lorentz_damage_scan.png'; fig.savefig(out, dpi=150)
print('图已存：', out)
