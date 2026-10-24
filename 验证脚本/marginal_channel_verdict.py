# -*- coding: utf-8 -*-
'''边际通道对照的判定与两通道并置（第 478 条，P4-31 第三步）。

判据 M1 至 M3：边际通道判定表；有量纲路径的 delta 上限复算；边际通道与六维通道的并置。
数值全部取自工作区已抄录值（上界注记 §七 与 §八、第 476 条能量表），不新引入外部来源。
'''
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

IMG = 'D:/software/TianXingFC/images/marginal_channel_verdict.png'
print('=' * 78)
print('第 478 条：边际通道对照的判定与两通道并置')
print('=' * 78)
print()

print('M1 边际通道判定表（对称性保持时框架侧系数为零）')
rows = [
    ('光子 k^(E), k^(B)（d=4）', 1e-35, '构造性取零（反射对称性）'),
    ('光子 kappa_tr（d=4）', 1e-20, '构造性取零（反射对称性）'),
    ('引力 s_bar（d=4）', 1e-11, '构造性取零（反射对称性）'),
    ('费米子 b_tilde（d=3，有量纲）', 1.5e-33, '构造性取零（反射对称性）'),
]
for name, b, src in rows:
    print('   %-28s 阈值 %.1e   框架侧 0   判定：精确通过（零来自 %s）' % (name, b, src))
print('   说明：本判定的意义在于记录零的来源，而不是记录数值（见上界注记 §三 第三步）。')
print()

print('M2 有量纲路径的 delta 上限复算（路径二：k_AF ~ delta M_*, b_tilde ~ delta m_*）')
kAF = 1e-44
btil = 1.5e-33
for tag, M in [('M_* = M_P ~ 1.2e19 GeV', 1.2e19), ('M_* ~ 1 GeV（电弱量级）', 1.0)]:
    print('   %-26s 光子 delta <= %.1e   费米子 delta <= %.1e' % (tag, kAF / M, btil / M))
print('   与上界注记 §八 表核对：1.2e19 档给 1e-63 与 1e-52，1 GeV 档给 1e-44 与 1e-33（一致）')
print('   读法：若框架侧存在百分级破缺（10^-2），两条路径都给出 10^9 至 10^61 个数量级的差距；')
print('       故「格点规则的反射对称性精确成立」由观测逼出，属唯一可行选择。')
print()

print('M3 边际通道与六维通道的并置（两条通道不互相掩盖）')
E = np.array([1e9, 1e12, 1e15, 1e18, 1e20])
xi = (1.0 / 24.0) * (E / 1.221e28) ** 2
print('   六维通道（能标依赖，第 476 条）：')
for e, x in zip(E, xi):
    print('      E = %.0e eV -> xi_eff = %.3e' % (e, x))
print('   边际通道（方向依赖与速度差，与能量无关）：框架侧零，阈值见 M1 表')
print('   并置结论：两通道的观测量不同（前者为到达时间与能标依赖，后者为方向依赖与偏振），')
print('       前者随 E^2 增长、后者与能量无关，一方的观测排除不会掩盖另一方的检验窗口。')
print()

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
names = ['photon k(E),k(B)', 'photon kappa_tr', 'gravity s_bar', 'fermion b_tilde']
vals = [r[1] for r in rows]
ax[0].barh(range(len(vals)), [np.log10(v) for v in vals], color='tab:blue')
ax[0].set_yticks(range(len(vals)))
ax[0].set_yticklabels(names, fontsize=8)
ax[0].set_xlabel('log10 (threshold)')
ax[0].set_title('M1 marginal-channel thresholds (framework side = 0)')
ax[1].loglog(E, xi, 'o-', label='six-dimensional channel xi_eff')
ax[1].axhline(1e-35, color='crimson', ls='--', label='photon k^(E) threshold')
ax[1].set_xlabel('photon energy (eV)')
ax[1].set_ylabel('effective coefficient')
ax[1].set_title('M3 two channels side by side')
ax[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(IMG, dpi=150)
print('图已保存：' + IMG)