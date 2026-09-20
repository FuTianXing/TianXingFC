# -*- coding: utf-8 -*-
"""minimal_verification_mdelta_recount.py

损害普查第四步：统一口径下的时标区间读数重算。

口径（本次统一）：
  D  = D(t_*) 分段标度，取代固定值 10^-74 至 10^-90
  s  = pi*w_def^2 / l_P^2（宽度档截面），取代 s = 1
  域 = p <= p_max（由 xi >= w_def 给出）

判据 J1 至 J10 见注记。判据先写、后实现，打印「实测 vs 期望」。
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ---------- 常量 ----------
lP = 1.616255e-35          # m
tP = 5.391247e-44          # s
c  = 2.99792458e8          # m/s
eps   = 4.4373e-47         # 质量论文口径 kappa_phi/omega_P^2
KJ    = 1.7518e-45         # 电子锚 K/J
teq   = 1.48e12            # s
zeq   = 3400.0
f_vis, f_nuc, f_ele = 2.12e-105, 1.06e-105, 1.95e-102
f_tot = 6.71e-105          # 由总物质档 t_* 读数反推（待作者确认）

wdef = 1.0 / (2.0*np.pi*np.sqrt(KJ))          # 以 lP 为单位
sigma = np.pi * wdef**2                        # 以 lP^2 为单位（宽度档）

def xi_of_p(p):    return eps**(-1.0/p)
def fprod_of_p(p): return eps**(3.0/p)
def D_of_t(t):     return (t/teq)**1.5 / zeq**3
def t_of_D(D):     return teq * (D*zeq**3)**(2.0/3.0)
def t_of_p(p, f):  return t_of_D(f/fprod_of_p(p))
def T_of_t(t, gs): return np.sqrt(2.42/np.sqrt(gs)/t)   # MeV

pmax = np.log(1.0/eps)/np.log(wdef)

print('=' * 74)
print('统一口径重算：时标区间读数')
print('=' * 74)
print('XI(p_max)/w_def  = %.4f   期望 1.0000' % (xi_of_p(pmax)/wdef))
print('F_PROD(p_max)    = %.4g   期望 1.82e-65' % fprod_of_p(pmax))
print('P_MAX            = %.4f   期望 2.15（由 xi>=w_def 定）' % pmax)

print('-' * 74)
print('J4 四档角点（p = p_max）')
tiers = [('可见物质', f_vis, 4.09e-8), ('核子', f_nuc, 2.58e-8),
         ('总物质', f_tot, 8.80e-8), ('电子', f_ele, 3.86e-6)]
for name, f, exp in tiers:
    D = f/fprod_of_p(pmax); t = t_of_D(D)
    print('  %-5s f=%.3g  D=%.4g  t_*=%.4g s  期望 %.3g s  比 %.4f'
          % (name, f, D, t, exp, t/exp))

print('-' * 74)
D_c = f_vis/fprod_of_p(pmax); t_c = t_of_D(D_c)
print('J3 角点 D（可见物质档） = %.4g   期望 1.17e-40' % D_c)
print('J7 输运余量：')
l_req = xi_of_p(pmax)                       # A = 1 -> l_req = xi
l_mfp = xi_of_p(pmax)**3 / sigma
ratio = max(1.0, 3.0*l_req/l_mfp)
tau_reach = (l_req/c) * ratio * lP
print('  l_req/l_mfp = %.4f (pi=%.4f)  倍率 %.3f' % (l_req/l_mfp, np.pi, ratio))
print('  tau_reach = %.4g s   t_*min = %.4g s   tau/t_* = %.3g  期望 <=1e-10'
      % (tau_reach, t_c, tau_reach/t_c))
print('J8 湮灭余量：tau_ann = 1e-3 t_* 给 A_min = %.4g  期望 <=1e-10'
      % ( (xi_of_p(pmax)*lP/(c*1e-3*t_c))**1.5 ))
print('J9 因果余量：c*t_*/xi = %.4g  期望 >=1e12' % (c*t_c/(xi_of_p(pmax)*lP)))

print('-' * 74)
print('J5 路线乙表内 t_*=xi/c 一列与密度链反解对照')
for p in (3.0, 4.0, 6.0):
    t_tab = xi_of_p(p)*lP/c
    t_den = t_of_p(p, f_vis)
    print('  p=%4.1f  表内 %.3g s   密度链 %.3g s   比 %.3g' % (p, t_tab, t_den, t_den/t_tab))

print('-' * 74)
print('J10 前沿与区间（可见物质档）')
t_qcd, t_ew = 1.4e-5, 9.15e-12
for name, tt in (('QCD', t_qcd), ('电弱', t_ew)):
    if t_of_p(pmax, f_vis) >= tt:
        print('  %-4s 前沿 t=%.3g s -> 域内处处晚于该时刻，无边界' % (name, tt))
        continue
    hit = None
    for p in np.arange(pmax, 1.50, -1e-5):
        if t_of_p(p, f_vis) >= tt:
            hit = p; break
    print('  %-4s 前沿 t=%.3g s -> p = %s' % (name, tt, ('%.4f' % hit) if hit else '域内无'))
print('  域内 t_* 上端须由热史一致性封顶：')
for cap in (0.1, 1.0, 10.0):
    lo = None
    for p in np.arange(1.50, pmax, 1e-5):
        if t_of_p(p, f_vis) <= cap:
            lo = p; break
    t_lo = t_of_p(lo, f_vis)
    print('    封顶 %.1f s -> p >= %.4f，下端时标 %.3g s，温度 %.4g MeV'
          % (cap, lo, t_lo, T_of_t(t_lo, 61.75)))
print('  角点上端：t_*min = %.4g s，温度 %.4g GeV'
      % (t_c, T_of_t(t_c, 61.75)/1000.0))

print('-' * 74)
print('J6 单调性：p 增大，t_* 必减')
ps = np.linspace(1.80, pmax, 40)
ts = np.array([t_of_p(p, f_vis) for p in ps])
print('  all decreasing =', bool(np.all(np.diff(ts) < 0)))
print('J2/口径复核：sigma = %.4g l_P^2（宽度档）；s = 1 旧档差 %.3g 倍'
      % (sigma, sigma))

# ---------- 图 ----------
fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.0))
pg = np.linspace(1.74, pmax, 400)
tg = np.array([t_of_p(p, f_vis) for p in pg])
ax[0].semilogy(pg, tg, lw=2.0, color='#1f4e79', label='密度链解族 $t_*(p)$（可见物质档）')
ax[0].axvline(pmax, color='crimson', ls='--', lw=1.2,
              label='$p_{\\max}$ = %.3f（$\\xi = w_{\\mathrm{def}}$）' % pmax)
ax[0].axhline(t_qcd, color='seagreen', ls=':', lw=1.2, label='QCD 前沿 $10^{-5}$ s')
ax[0].axhline(1.0, color='darkorange', ls='-.', lw=1.2, label='热史上端 1 s')
ax[0].axhline(tP, color='gray', ls=':', lw=1.0, label='$t_P$')
ax[0].fill_betweenx([tP, 1e6], 1.70, pmax, color='#dbe5f1', alpha=0.45)
for name, f, _ in tiers:
    ax[0].plot([pmax], [t_of_D(f/fprod_of_p(pmax))], 'o', ms=6,
               label='%s档' % name)
ax[0].set_xlabel('幂指数 $p$'); ax[0].set_ylabel('$t_*$（秒）')
ax[0].set_ylim(1e-44, 1e7); ax[0].set_title('统一口径下的时标解族')
ax[0].legend(fontsize=7.5, loc='lower left'); ax[0].grid(alpha=0.3)

ax[1].semilogy(pg, [xi_of_p(p)/wdef for p in pg], lw=2.0, color='#1f4e79',
               label='$\\xi/w_{\\mathrm{def}}$')
ax[1].semilogy(pg, [fprod_of_p(p) for p in pg], lw=2.0, color='crimson',
               label='$f_{\\mathrm{prod}}=\\varepsilon^{3/p}$')
ax[1].axvline(pmax, color='crimson', ls='--', lw=1.2)
ax[1].axhline(1.0, color='gray', ls=':', lw=1.0, label='$\\xi = w_{\\mathrm{def}}$')
ax[1].set_xlabel('幂指数 $p$'); ax[1].set_ylabel('（无量纲）')
ax[1].set_title('关联长度与生产份额随 $p$')
ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
fig.tight_layout()
out = 'D:/software/TianXingFC/images/mdelta_recount_tstar_interval.png'
print('字体解析：', matplotlib.font_manager.findfont('Microsoft YaHei'))
fig.savefig(out, dpi=150)
print('图已存：', out)
