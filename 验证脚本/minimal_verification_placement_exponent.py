# -*- coding: utf-8 -*-
"""minimal_verification_placement_exponent.py

落位律的运动学判据：约束落在膨胀指数 n 上，不落在常数比例因子上。
判据 K1 至 K6 先写后实现。
"""
import numpy as np

lP, tP, c = 1.616255e-35, 5.391247e-44, 2.99792458e8
taun, Q, Tf, TD, g_f, g_D = 879.4, 1.293, 0.8, 0.07, 10.75, 3.36
Yp_obs, Yp_err = 0.2453, 0.0034

def t_std(T, gs): return 2.42/np.sqrt(gs)*T**(-2.0)
def xneq(T):      return 1.0/(1.0+np.exp(Q/T))
def Yp_of_n(n):
    dt = 2.0*n*(t_std(TD, g_D) - t_std(Tf, g_f))
    return 2.0*xneq(Tf)*np.exp(-dt/taun)

print('='*74); print('落位律的运动学判据'); print('='*74)
# K1: R 正比 t^n 给 H = n/t
print('K1  R 正比 t^n 与 H = n/t 的对照')
for n in (1.0, 1.0/2.0, 2.0/3.0):
    tt = np.geomspace(1.0, 10.0, 20001)
    lnR = np.log(tt**n)
    slope = np.diff(lnR)/np.diff(np.log(tt))   # dlnR/dlnt = n
    print('   n = %.4f：dlnR/dlnt = %.6f   期望 %.4f' % (n, slope[0], n))

# K2: 常速前缘：任意落位系数都只给 n = 1
print('K2  常速前缘（p = alpha*3*lP/R）：H = 1/t 与 alpha 无关')
for alpha in (1.0, 0.5, 0.25):
    v = alpha*c
    R0 = c*4.083e-8
    tt = np.array([1.0, 100.0])
    Rr = R0 + v*tt
    print('   alpha = %.2f：v = %.3g m/s，H*t = %s   期望趋 1'
          % (alpha, v, np.array2string(v/Rr*tt, precision=6)))

# K3: 双条件落位 p 正比 (lP/R)^2 给 n = 1/2
print('K3  双条件落位（p = beta*(lP/R)^2）：解析积分核对')
beta, R0 = 1.0, c*4.083e-8
tt = np.linspace(1.0, 1e6, 400001)
D_micro = lP**2/tP
D_need = (c*t_std(Tf, g_f))**2/t_std(Tf, g_f)
print('   代数上给 n = 1/2（2R dR 正比 dt 即 R 正比 sqrt(t)），但系数越界：')
print('   要求 D ~ R^2/t 在核合成处 = %.3g m^2/s；微观格点值 lP^2/tP = %.3g m^2/s'
      % (D_need, D_micro))
print('   相差 %.3g 倍，即 beta 须为 O(1e43) 量级才能推动尺度级前缘，机制不成立'
      % (D_need/D_micro))

# K4: 核合成给出的 n 允许带
grid = np.linspace(0.30, 1.05, 3001)
ok = [x for x in grid if abs(Yp_of_n(x)-Yp_obs) <= Yp_err]
print('K4  观测带 ±1.4%% 给出的膨胀指数 n = %.4f 至 %.4f   期望含 0.5000'
      % (min(ok), max(ok)))
print('   n = 1（常速前缘）给 Y_p = %.4f，偏离 %+.1f%%' % (Yp_of_n(1.0), 100*(Yp_of_n(1.0)/Yp_obs-1)))
print('   n = 0.5（扩散前缘）给 Y_p = %.4f' % Yp_of_n(0.5))

# K5: 三条候选机制（常数重标）逐一判
print('K5  上一轮三条候选机制的判定：三者都只改常数，不改指数')
cands = [('落位节拍与几何计数单位不一致（每两步一次落位）', 0.5), ('边缘格点只有一半处于缺邻居状态', 0.5), ('常数 3 的口径重标', 0.25)]
for name, alpha in cands:
    print('   %-28s alpha = %.2f -> n = 1.0000，Y_p = %.4f，判：无效'
          % (name, alpha, Yp_of_n(1.0)))
print('K6  有效机制的门槛：任何只把 p 乘常数 alpha 的方案保持 n = 1；')
print('     要落到 n = 0.5 必须改 p 对 R 的幂次（1/R -> 1/R^2），或改前缘的运动学。')
