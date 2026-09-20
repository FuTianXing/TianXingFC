# -*- coding: utf-8 -*-
"""minimal_verification_insertion_law.py

插入律的系数：三条候选各自给出什么 p_0，与核合成带对照。
判据 L1 至 L5 先写后实现。
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

lP, tP, c = 1.616255e-35, 5.391247e-44, 2.99792458e8
H0 = 2.2e-18
taun, Q, Tf, TD, g_f, g_D = 879.4, 1.293, 0.8, 0.07, 10.75, 3.36
Yp_obs, Yp_err = 0.2453, 0.0034

def t_std(T, gs): return 2.42/np.sqrt(gs)*T**(-2.0)
def xneq(T):      return 1.0/(1.0+np.exp(Q/T))
def Yp_of_n(n):
    dt = 2.0*n*(t_std(TD, g_D) - t_std(Tf, g_f))
    return 2.0*xneq(Tf)*np.exp(-dt/taun)

print('='*74); print('插入律的系数：三条候选'); print('='*74)
# L1 恒等：p_ins = 3 H tP
print('L1  格点恒等核对：p_ins = 3 H tP')
for n, name in ((0.5, '辐射'), (2.0/3.0, '物质')):
    H = n/1.0                       # 取 t = 1 s
    p = 3*H*tP
    print('   %s段（H = %.4f/t，t = 1 秒）：p_ins = %.4g，p_ins*t/tP = %.4f   期望 3n = %.4f'
          % (name, n, p, p*1.0/tP, 3*n))
print('   结论：给定 H = n/t，p_ins 的系数自动为 3n；p_0 = 3/2 不是被推导出来的，是 H = 1/(2t) 的改写')

# L2 候选二：效率耦合给 H = eta/t
print('L2  候选二（边缘项按效率 eta 供给插入）：H = eta/t，核合成给 eta 带')
grid = np.linspace(0.20, 1.05, 4001)
ok = [x for x in grid if abs(Yp_of_n(x)-Yp_obs) <= Yp_err]
print('   eta 允许带 = [%.4f, %.4f]   期望含 0.5000（辐射段）' % (min(ok), max(ok)))
print('   eta = 0.5 给 Y_p = %.4f；eta = 1.0 给 %.4f（偏离 %+.1f%%）'
      % (Yp_of_n(0.5), Yp_of_n(1.0), 100*(Yp_of_n(1.0)/Yp_obs-1)))

# L3 物质段要求 eta = 2/3，与辐射段的 1/2 冲突
print('L3  同一机制须在两段分别给 eta = 1/2（辐射）与 2/3（物质）')
for eta, name in ((0.5, '辐射'), (2.0/3.0, '物质')):
    print('   %s段：要求 eta = %.4f，落在核合成带内 = %s' % (name, eta, min(ok) <= eta <= max(ok)))
print('   物质段在核合成之后（t_eq = 1.48e12 s），核合成不约束它；须另用 CMB 声学尺度检验（待做）')

# L4 候选一与候选二的可判差别
print('L4  可判差别：边缘是否在膨胀史留痕')
print('   候选一（slaving）：边缘项不进 Friedmann，磨合期不在膨胀史留痕')
print('   候选二（效率耦合）：H 由边缘给定，磨合期的高值平台段给出 H ~ 1/(3tP) ~ %.3g 1/s'
      % (1/(3*tP)))

# L5 过渡时刻
print('L5  过渡时刻')
print('   候选一：插入跟踪到 H = H0 为止，与暗能量主导起点同点（z ~ 0.3）')
print('   候选二：过渡由 eta 换形给出，时刻待定；候选一给出的是 1/(2t) 与 H0 的交点')
for n in (0.5, 2.0/3.0):
    print('     n = %.3f 时 1/(2*%.3f*t) = H0 给 t = %.3g s = %.3g Gyr'
          % (n, n, n/(2*H0), n/(2*H0)/3.156e16))

# ---------- 图 ----------
fig, ax = plt.subplots(1, 1, figsize=(7.6, 5.0))
ee = np.linspace(0.25, 1.0, 600)
ax.plot(ee, [Yp_of_n(e) for e in ee], lw=2.2, color='#1f4e79', label='$Y_p(\\eta)$')
ax.axhspan(Yp_obs-Yp_err, Yp_obs+Yp_err, color='seagreen', alpha=0.30,
           label='观测 $0.2453\\pm0.0034$')
ax.axvline(0.5, color='seagreen', ls='--', lw=1.3, label='辐射段要求 $\\eta = 1/2$')
ax.axvline(2.0/3.0, color='darkorange', ls='-.', lw=1.3, label='物质段要求 $\\eta = 2/3$')
ax.axvline(1.0, color='crimson', ls=':', lw=1.3, label='几何基线 $\\eta = 1$')
ax.set_xlabel('效率 $\\eta$（边缘项供给插入的比例）'); ax.set_ylabel('$Y_p$')
ax.set_title('氦-4 产额对插入效率的灵敏度')
ax.legend(fontsize=8.5); ax.grid(alpha=0.3)
fig.tight_layout()
out = 'D:/software/TianXingFC/images/insertion_law_eta_band.png'
fig.savefig(out, dpi=150)
print('图已存：', out)
