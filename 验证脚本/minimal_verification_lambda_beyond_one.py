# -*- coding: utf-8 -*-
"""minimal_verification_lambda_beyond_one.py

H3 轮：lambda/J > 1 的开域扫描。判据 K1 至 K5 先写后实现。

复用 U2 最小臂的器械（径向饱和源模型 + 耦合力，三档作用区 x 两臂符号），
沿 lambda 连续延拓到 1 以上，判是否存在物理分支以及边界的位置。
"""
import io, contextlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

R = 'D:/software/TianXingFC/'
g = {'__name__': '__main__'}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(open(R+'验证脚本/minimal_verification_u2_coupling_stress.py', encoding='utf-8').read(),
                 'u2_coupling_stress', 'exec'), g)
solve, readouts, r = g['solve'], g['readouts'], g['r']
base = g['theta1'] if 'theta1' in g else None
if base is None:
    for k in ('theta0', 't0', 'th0'):
        if k in g: base = g[k]; break
print('='*74); print('H3：lambda/J > 1 的开域扫描'); print('='*74)
b0 = readouts(base)
print('K1 基线复核（lambda = 0）：r_h = %.4f（D3 表 10.285）；真空区散布 %.4g'
      % (b0['rh'], b0['spread']))

lams = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
print('-'*74)
print('K2/K3 延拓扫描（三区 x 两臂；守门 = 残差 < 1e-12 且 theta >= 0）')
rec = {}
for key in ('H1', 'H2', 'H3'):
    for arm, sgn in (('M（登记臂，回拉）', +1), ('P（对照臂，推高）', -1)):
        guess = base.copy(); lastok = None
        print('   %s 区 / %s' % (key, arm))
        for lam in lams:
            t, nR = solve(lam, key, guess, sgn)
            ok = (nR < 1e-12) and (float(t.min()) >= -1e-9)
            ro = readouts(t)
            print('      lambda = %6.2f：res = %.2e  theta_min = %+.4f  r_h = %8.4f  '
                  '散布 = %8.2e  D_近 = %7.3f  D_外 = %8.3f  %s'
                  % (lam, nR, float(t.min()), ro['rh'], ro['spread'],
                     ro['D_near'], ro['D_outer'], '守门过' if ok else '守门失败'))
            rec[(key, arm, lam)] = (ok, ro, float(t.min()))
            if not ok:
                break
            guess = t
            lastok = lam
        print('      最后一个守门通过的 lambda = %s' % lastok)

print('-'*74)
print('K4 物理范围对照')
for name, val in (('电子锚', 1.7518e-45), ('可见物质档', 2.12e-105/1.82e-65*1.0)):
    print('   %s 的 lambda/J = %.3g（此行为量级对照，第二行按密度档份额比给出）' % (name, val))
print('   物理可达的 lambda/J 远小于 1（电子锚低 45 个数量级），故本扫描的 >1 段属模型外推区')
print('K5 判定：见注记 §五')
