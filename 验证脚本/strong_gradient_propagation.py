# -*- coding: utf-8 -*-
"""strong_gradient_propagation.py

D2 轮：强梯度区（近视界）的微观传播检验。判据 P1 至 P5 先写后实现。

方法：前沿到达法。在含渐变 c(r) 的一维格点上从外端驱动一个陡前沿，
记录各半径的首达时刻 T(r)，由 dT/dr = 1/c(r) 取局域光速。
前沿先于任何反射到达，故该方法对边界反射免疫（前两版的驻波污染由此消除）。
"""
import io, contextlib
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

R = 'D:/software/TianXingFC/'
buf = io.StringIO(); g = {'__name__': '__main__'}
with contextlib.redirect_stdout(buf):
    src = open(R + '验证脚本/light_cone_horizon.py', encoding='utf-8').read()
    exec(compile(src, 'light_cone_horizon', 'exec'), g)
theta1, r_grid, rh = g['theta1'], g['r'], g['rh']
print('='*74); print('D2：强梯度区微观传播检验（前沿到达法）'); print('='*74)
print('复用 D3 剖面：r_h = %.3f，theta_max = %.4f' % (rh, float(theta1.max())))

h, dt = 0.125, 0.1
r_in, r_out = 10.6, 200.0
rr = np.arange(r_in, r_out + 1e-9, h)
th = np.clip(CubicSpline(r_grid, theta1)(rr), 0.0, 0.999)
om = 0.3927
print('格点 h = %.3f，dt = %.3f，驱动 omega = %.4f' % (h, dt, om))

def c_vac(t): return np.clip(1.0 - 2.0*t, 0.0, None)
def c_lin(t): return np.sqrt(np.clip((1.0-2.0*t)/(1.0+2.0*t), 0.0, None))

def front(c_of_t, nstep=8000):
    c = c_of_t(th); J = len(rr)-1
    u = np.zeros(J+1); uo = np.zeros(J+1)
    arr = np.full(J+1, np.nan)
    peak = np.zeros(J+1)
    hist = np.zeros((nstep, J+1))
    for n in range(nstep):
        t = (n+1)*dt
        lap = np.zeros(J+1); lap[1:J] = (u[2:J+1]-2*u[1:J]+u[0:J-1])
        un = 2*u-uo+(c*dt/h)**2*lap
        un[0] = 0.0; un[J] = np.sin(om*t)*(t < 8*np.pi/om)
        uo, u = u, un
        hist[n] = u
        m = np.abs(u)
        new = (arr != arr) & (m > 1e-3) & (t > 1.0)
        arr[new] = t
    return c, arr

print('-'*74)
c_u, T_u = front(lambda t: np.full_like(t, 0.5))
for x in (180.0, 120.0, 60.0):
    j = int(np.argmin(np.abs(rr-x)))
    print('P1 均匀对照 c = 0.5：r = %6.1f 处首达 %.4f，理论 %.4f' % (rr[j], T_u[j], (200.0-rr[j])/0.5 + 1.0))
ju = [int(np.argmin(np.abs(rr-x))) for x in (180.0, 120.0, 60.0)]
sl = np.polyfit(rr[ju], T_u[ju], 1)[0]
print('   由首达时刻斜率给 |c| = %.5f   期望 0.50000（斜率带符号，取绝对值）' % abs(1.0/sl))

print('-'*74)
c_v, T_v = front(c_vac)
m = np.isfinite(T_v) & (rr > 12.5) & (rr < 150)
# 窗宽正负 40 个细格（= 5 个粗单位）：把首达时刻的 dt 量子化平滑掉
W = 40
c_meas = np.full_like(T_v, np.nan)
for j in range(W, len(rr)-W):
    dT = T_v[j-W] - T_v[j+W]; dR = rr[j+W] - rr[j-W]
    c_meas[j] = dR/max(dT, 1e-9)
print('P2/P3 渐变区局域光速：两种完成形式对照')
print('   %6s %8s %10s %10s %9s %10s %9s' % ('r', 'theta', '实测 c', '真空预言', '相对差', '线性预言', '相对差'))
dv, dl = [], []
for x in (100.0, 60.0, 40.0, 25.0, 20.0, 16.0):
    j = int(np.argmin(np.abs(rr-x)))
    a = c_meas[j]/c_v[j]-1; b = c_meas[j]/c_lin(th[j])-1
    dv.append(abs(a)); dl.append(abs(b))
    print('   %6.1f %8.4f %10.5f %10.5f %+8.2e %10.5f %+8.2e'
          % (rr[j], th[j], c_meas[j], c_v[j], a, c_lin(th[j]), b))
print('   对真空完成的最大相对差 %.3e；对线性完成的最大相对差 %.3e' % (max(dv), max(dl)))

print('-'*74)
sel = (rr >= 14.0) & (rr <= 120.0)
tt = np.trapezoid(1.0/np.clip(c_v[sel], 1e-9, None), rr[sel])
tm = T_v[int(np.argmin(np.abs(rr-14.0)))] - T_v[int(np.argmin(np.abs(rr-120.0)))]
print('P4 旅行时间 120 至 14（真空完成）：预言 %.4f，实测 %.4f，相对差 %+.2e   期望 <= 3e-2'
      % (tt, tm, tm/tt-1))
j20 = int(np.argmin(np.abs(rr-20.0)))
print('P5 近内侧测点 r = %.1f：theta = %.4f，预言 c = %.4f，实测 c = %.4f（窗宽 %.1f 以内为有效域）'
      % (rr[j20], th[j20], c_v[j20], c_meas[j20], W*h))
inn = (T_v[j20]-T_v[j20+40])/(rr[j20+40]-rr[j20])
out = (T_v[int(np.argmin(np.abs(rr-160.0)))]-T_v[int(np.argmin(np.abs(rr-180.0)))])/20.0
print('   冻结堆积（每单位 r 的耗时）：内段 %.3f 对远端 %.3f，比 %.2f'
      % (inn, out, inn/out))

fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.8))
m2 = (rr >= 11.5) & (rr <= 100.0)
ax[0].plot(rr[m2], c_v[m2], lw=2.2, color='#1f4e79', label='真空完成 $c=1-2\\theta$')
ax[0].plot(rr[m2], c_lin(th)[m2], lw=1.2, ls=':', color='gray', label='线性完成')
ax[0].plot(rr[m2], c_meas[m2], lw=1.4, ls='--', color='crimson', label='前沿法实测')
ax[0].axvline(rh, color='k', ls=':', lw=1.0); ax[0].set_ylim(0, 1.0)
ax[0].set_xlabel('$r$'); ax[0].set_ylabel('$c_r$'); ax[0].set_title('强梯度区的局域光速')
ax[0].legend(fontsize=8.5); ax[0].grid(alpha=0.3)
msel = (rr > 13.0) & (rr < 100.0)
ax[1].semilogy(rr[msel], c_v[msel], lw=2.0, color='#1f4e79', label='真空完成')
ax[1].semilogy(rr[msel], c_meas[msel], lw=1.3, ls='--', color='crimson', label='实测')
ax[1].axvline(rh, color='k', ls=':', lw=1.0)
ax[1].set_xlabel('$r$'); ax[1].set_ylabel('$c_r$（对数轴）'); ax[1].set_title('近视界的衰减对照')
ax[1].legend(fontsize=8.5); ax[1].grid(alpha=0.3, which='both')
fig.tight_layout(); out = R+'images/strong_gradient_propagation.png'; fig.savefig(out, dpi=150)
print('图已存：', out)
