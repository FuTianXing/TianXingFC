# -*- coding: utf-8 -*-
"""minimal_verification_thermal_history.py

热史一致性核对：框架膨胀史与标准分段对照，以及核合成后果。
判据 J1 至 J8 先写后实现，打印「实测 vs 期望」。
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ---------- 常量 ----------
lP, tP, c = 1.616255e-35, 5.391247e-44, 2.99792458e8
H0   = 2.2e-18          # 1/s（67.4 km/s/Mpc）
taun = 879.4            # s，中子寿命
Q    = 1.293            # MeV，中子-质子质量差
Tf, TD = 0.8, 0.07      # MeV，冻结与氘瓶颈
g_f, g_D = 10.75, 3.36  # 有效自由度
Yp_obs, Yp_err = 0.2453, 0.0034   # 观测与误差

def t_std(T, gs):  return 2.42/np.sqrt(gs)*T**(-2.0)   # s，标准辐射主导
def xneq(T):       return 1.0/(1.0+np.exp(Q/T))
def Yp_of_r(r):
    dt = 2.0*r*(t_std(TD, g_D) - t_std(Tf, g_f))
    return 2.0*xneq(Tf)*np.exp(-dt/taun)

print('='*72); print('热史一致性核对'); print('='*72)
# J1 恒等式：H = H0 + c/R  与  p = 3 lP / R
print('J1 恒等式核对（取 t = 1 s，解 R(t)）')
R0 = c*4.083e-8                                  # 结晶时刻边缘半径
tt = np.array([1e-3, 1.0, 1e3])
Rnum = R0*np.exp(H0*tt) + (c/H0)*np.expm1(H0*tt)
Hnum = (R0*np.exp(H0*tt)*H0 + c*np.exp(H0*tt))/Rnum
Hana = H0 + c/Rnum
print('   max |H_num/H_ana - 1| = %.3e   期望 < 1e-12' % np.max(np.abs(Hnum/Hana-1)))
print('   p = 3 lP/R 复算：p(1 s) = %.4g   3 tP/t = %.4g   比 %.4f' %
      (3*lP/Rnum[1], 3*tP/1.0, (3*lP/Rnum[1])/(3*tP/1.0)))

# J2 基线给 Milne
print('J2 基线 p_geo = 3 tP/t 给 H = 1/t（Milne，a 正比 t）')
print('   H_geo(1 s) = %.4g 1/s   与 1/t = 1.0 之比 %.4f   期望 1.0000'
      % (c/Rnum[1], (c/Rnum[1])/1.0))

# J3 标准辐射主导的要求
print('J3 标准辐射主导要求 H = 1/(2t)，即 t_fw/t_std = 2r = 1')
print('   H_std(0.8 MeV) = %.4g 1/s   对应 t_std = %.4f s' % (0.5/t_std(Tf, g_f), t_std(Tf, g_f)))
print('   代数恒等：H = r*H_geo = r/t 与 t_std = 1/(2H) 合取给 t_fw/t_std = 2r')
print('   要复现标准热史须 2r = 1，即 r = 0.5000')

# J4 时间-温度关系
print('J4 同一温度下的时间比 t_fw/t_std = 2r')
for T, gs in ((1.0, 10.75), (0.1, 3.36)):
    print('   T = %.2f MeV：t_std = %.4g s，r=1 给 %.4g s，比值 %.4f'
          % (T, t_std(T, gs), 2*t_std(T, gs), 2.0))

# J5 核合成后果
print('J5 核合成后果（冻结-衰变玩具模型）')
print('   标准（r=0.5）：Y_p = %.4f   观测 %.4f ± %.4f' % (Yp_of_r(0.5), Yp_obs, Yp_err))
print('   基线（r=1.0）：Y_p = %.4f   偏离 %+.1f%%'
      % (Yp_of_r(1.0), 100*(Yp_of_r(1.0)/Yp_obs-1)))
grid = np.linspace(0.30, 1.05, 3001)
ok = [r for r in grid if abs(Yp_of_r(r)-Yp_obs) <= Yp_err]
print('   观测带 ±1.4%% 给出的允许因子 r = %.4f 至 %.4f   期望含 0.5000'
      % (min(ok), max(ok)))
print('   逐点核对 r = 0.5 与 r = 1.0 的 Y_p：%.4f 与 %.4f'
      % (Yp_of_r(0.5), Yp_of_r(1.0)))

# J6 H0 项可略
print('J6 H0 项可略：H0*t(1 MeV) = %.3e   期望 < 1e-15' % (H0*t_std(Tf, g_f)))
# J7 磨合期
print('J7 磨合期（阶段一）须在核合成前结束；其时长未导出，上端仍为开放项')
# J8 假设清单
print('J8 假设：Friedmann 成立、物质为标准辐射、弱作用微物理标准、T 与 R 反比')

# ---------- 图 ----------
fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.0))
rr = np.linspace(0.3, 1.05, 400)
ax[0].plot(rr, [Yp_of_r(r) for r in rr], lw=2.2, color='#1f4e79',
           label=r'$Y_p(r)$（$r$：落位因子）')
ax[0].axhspan(Yp_obs-Yp_err, Yp_obs+Yp_err, color='seagreen', alpha=0.30,
              label='观测 $0.2453\pm0.0034$')
ax[0].axvline(0.5, color='seagreen', ls='--', lw=1.3, label='辐射主导要求 $r = 0.5$')
ax[0].axvline(1.0, color='crimson', ls='--', lw=1.3, label='几何基线 $r = 1$（Milne）')
ax[0].set_xlabel('落位因子 $r = p/p_{\\mathrm{geo}}$')
ax[0].set_ylabel('$Y_p$')
ax[0].set_title('氦-4 产额对落位因子的灵敏度')
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

TT = np.logspace(np.log10(0.05), np.log10(300.0), 300)
ax[1].loglog(TT, [t_std(T, 10.75 if T > 0.5 else 3.36) for T in TT],
             lw=2.2, color='seagreen', label='标准辐射主导')
ax[1].loglog(TT, [2*t_std(T, 10.75 if T > 0.5 else 3.36) for T in TT],
             lw=2.2, color='crimson', label='框架基线 $r = 1$（时间加倍）')
ax[1].axvline(Tf, color='gray', ls=':', lw=1.0, label='冻结 $0.8$ MeV')
ax[1].axvline(TD, color='gray', ls='--', lw=1.0, label='氘瓶颈 $0.07$ MeV')
ax[1].set_xlabel('$T$（MeV）'); ax[1].set_ylabel('$t$（秒）')
ax[1].set_title('时间-温度关系：标准与框架基线')
ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, which='both')
fig.tight_layout()
out = 'D:/software/TianXingFC/images/mdelta_thermal_history_bbn.png'
fig.savefig(out, dpi=150)
print('图已存：', out)
