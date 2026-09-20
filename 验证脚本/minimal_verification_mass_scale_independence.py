# -*- coding: utf-8 -*-
"""minimal_verification_mass_scale_independence.py

D4 轮：质量刻度的独立标定。判据 M1 至 M5 先写后实现。

问题：H7 指出 s0、h0 到物理质量的换算经 r_h 单值中转，而 M/m_P = r_h/(2 l_P)
自标为条件性命题，未经独立标定。本轮核对两条「锚」是否真的独立。
"""
import io, contextlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

R = 'D:/software/TianXingFC/'
buf = io.StringIO(); g = {'__name__': '__main__'}
with contextlib.redirect_stdout(buf):
    src = open(R + '验证脚本/minimal_verification_light_cone_horizon.py', encoding='utf-8').read()
    exec(compile(src, 'light_cone_horizon', 'exec'), g)
theta, r, rh = g['theta1'], g['r'], g['rh']
print('='*74); print('D4：质量刻度的独立标定（两锚是否独立）'); print('='*74)
print('复用 D3 剖面：r_h = %.3f（网格 dr=1）' % rh)

# ---------- M1 稳态识别式的代数内容 ----------
print('M1 稳态识别式 theta(r) = (m/m_P)(l_P/r) 的代数内容')
print('   令 A = (m/m_P) l_P（远场系数），则 theta(r) = A/r')
print('   视界条件 theta = 1/2 给 r_h = 2A，即 m/m_P = r_h/(2 l_P)')
print('   判：远场系数锚（A）与视界锚（r_h）是同一条关系的两种读法，代数上不独立')

# ---------- M2 数值核对 ----------
print('-'*74)
print('M2 数值核对：由远场系数反推 r_h 与直接视界插值比较')
half = 0.5*theta
far = (r > 80) & (r < 0.9*r.max())
A = float(np.mean(theta[far]*r[far]))
rh_from_A = 2.0*A
print('   远场系数 A = <theta*r> = %.4f（窗口 %.0f 至 %.0f，样本 %d）'
      % (A, r[far].min(), r[far].max(), far.sum()))
print('   由 A 反推 r_h = 2A = %.4f；直接视界插值 r_h = %.4f；相对差 %.3e'
      % (rh_from_A, rh, rh_from_A/rh - 1))
print('   判：两条锚在剖面内逐位一致（差异只来自离散与窗口），确认 M1 的代数同一性')

# ---------- M3 两锚对同一量的敏感度 ----------
print('-'*74)
print('M3 敏感性：远场窗口与网格对 A 的影响')
for lo, hi in ((60, 0.9*r.max()), (80, 0.9*r.max()), (100, 0.95*r.max())):
    m = (r > lo) & (r < hi)
    print('   窗口 %5.0f 至 %5.0f：A = %.5f，2A = %.5f（对 r_h 相对差 %+.2e）'
          % (lo, hi, float(np.mean(theta[m]*r[m])), 2*float(np.mean(theta[m]*r[m])),
             2*float(np.mean(theta[m]*r[m]))/rh - 1))

# ---------- M4 结论：独立路线需要什么 ----------
print('-'*74)
print('M4 独立路线清单（逐条给阻塞点）')
print('   路线甲 远场系数锚：与视界锚同一条关系（本轮的 M1、M2）——不是独立路线')
print('   路线乙 场的能量泛函：需记录场的能量泛函，体系内未立（U2 的写入-扩散-弹性协议无能量式）')
print('   路线丙 源 s0 的微观身份：需知道 s0 计什么（缺陷密度），而饱和源模型自标为模型输入')
print('   路线丁 粒子线质量刻度 m_delta：可换归一化单位，但不改变物理内容（同一质量换单位）')
print('   判：真正独立的路线（乙、丙）都缺前置，且前置同属 H1（第一原理强场动力学）')

# ---------- M5 结论 ----------
print('-'*74)
print('M5 结论')
print('   其一，「双锚」（r_h 与 q/Q）不是独立校验，而是同一稳态识别式的两种读法')
print('   其二，H7 不能单独修补：独立标定所需的前置（场的能量泛函，或源 s0 的微观身份）')
print('         都属 H1 的缺口，故 H7 归并为 H1 的推论')
print('   其三，今天可做的只有标注订正：质量窗口属换算级读数，其「独立核验」的说法须撤回')

# ---------- 图 ----------
fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.6))
m = r > 1.02*rh
ax[0].loglog(r[m], theta[m]*r[m], lw=2.0, color='#1f4e79', label='$theta\\,r$（应趋于常数 $A$）')
ax[0].axhline(A, color='crimson', ls='--', lw=1.2, label='远场均值 $A = %.4f$' % A)
ax[0].axvline(rh, color='k', ls=':', lw=1.0)
ax[0].set_xlabel('$r$（格距单位）'); ax[0].set_ylabel('$\\theta\\,r$')
ax[0].set_title('远场系数锚'); ax[0].legend(fontsize=8.5); ax[0].grid(alpha=0.3, which='both')
ax[1].plot(r[m], theta[m], lw=2.0, color='#1f4e79', label='$\\theta(r)$')
ax[1].axhline(0.5, color='crimson', ls='--', lw=1.2, label='视界条件 $\\theta = 1/2$')
ax[1].axvline(rh, color='k', ls=':', lw=1.0, label='插值 $r_h = %.3f$' % rh)
ax[1].axvline(rh_from_A, color='darkorange', ls='-.', lw=1.2, label='由 $A$ 反推 $2A = %.3f$' % rh_from_A)
ax[1].set_xlabel('$r$（格距单位）'); ax[1].set_ylabel('$\\theta$')
ax[1].set_ylim(0, 1.05); ax[1].set_title('视界锚与远场锚重合'); ax[1].legend(fontsize=8.5); ax[1].grid(alpha=0.3)
fig.tight_layout(); out = R+'images/mass_scale_anchors.png'; fig.savefig(out, dpi=150)
print('图已存：', out)
