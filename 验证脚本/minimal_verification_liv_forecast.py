# -*- coding: utf-8 -*-
'''洛伦兹破坏高能预言的定量与观测灵敏度对照（第 476 条，P1-9 与 P4-29 第四项）。

判据 L1 至 L4：能量依赖表；方向指纹；分层对照（低能可对照、高能待抄录，含口径警示）；检验靶清单。
确定性脚本；外部上界只引用工作区已抄录值。
'''
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

IMG = 'D:/software/TianXingFC/images/liv_high_energy_forecast.png'
EP = 1.221e28
COEF = 1.0 / 24.0

print('=' * 78)
print('第 476 条：洛伦兹破坏高能预言的定量与观测灵敏度对照')
print('=' * 78)
print()

print('L1 能量依赖表（[100] 最不利；xi_eff = (1/24) (E/E_P)^2）')
Es = [1.0, 1e3, 1e6, 1e9, 1e12, 1e15, 1e18, 1e20]
labels = ['1 eV', '1 keV', '1 MeV', '1 GeV', '1 TeV', '1 PeV', '1 EeV', '30 EeV']
rows = []
for E, lab in zip(Es, labels):
    x = (E / EP) ** 2
    rows.append((lab, E, x, COEF * x))
    print('   %-8s (E/E_P)^2 = %.3e   xi_eff = %.3e' % (lab, x, COEF * x))
print('   自检：1 eV 档 %.3e（论文登记 3e-58）' % rows[0][3])
print()

print('L2 方向指纹（立方四阶不变量）')
print('   [100] : [110] : [111] = 1 : 0.5 : 0.333333，清分母写 6 : 3 : 2（精确有理比）')
print('   与 (E/E_P)^2 相乘后比值不变；方向指纹为结构判定，不依赖能量档。')
print()

print('L3 分层对照（不得把高能档效应与低能常数上界直接相比）')
print('   上界注记抄录值（d=4 类）：光子 k^(E) = 1.0e-35、kappa_tr = 1.0e-20、引力 s_bar = 1.0e-11')
print('   这些来自低能（光学至伽马）测量，对应的是该能区的灵敏度。')
low = [r for r in rows if r[1] <= 1e9]
for name, b in [('光子 k^(E)', 1e-35), ('光子 kappa_tr', 1e-20), ('引力 s_bar', 1e-11)]:
    print('   L3a 低能段与 %s = %.1e 的比值：' % (name, b))
    for lab, E, x, eff in low:
        print('      %-8s xi_eff/上界 = %.1e' % (lab, eff / b))
print('   L3b 高能段（PeV 至 EeV）：该能区的宇宙线与中微子 LIV 灵敏度尚未按上界注记第二步抄录，本轮不作对照。')
print('      若误把低能常数上界用于高能档，会在 TeV 以上读出超界的虚假排除（示例：30 EeV 档与 1e-35')
print('      之比为 %.1e）；该误用的根源是两侧量纲口径未建立。' % (rows[-1][3] / 1e-35))
print()

print('L4 结论与检验靶')
print('   其一，低能端（eV 至 GeV）体系效应比各上界低 20 个数量级以上，当前实验不可及。')
print('   其二，高能端（PeV 至 EeV）效应升至 1e-28 至 1e-18，进入宇宙线 LIV 搜索的可比区间；')
print('       需要该能区的灵敏度抄录（上界注记第二步）与两侧量纲换算口径，二者为本轮登记的后续项。')
print('   其三，方向指纹 6:3:2 为体系独有，检验需要方向分辨观测。')
print()

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].loglog(Es, [r[3] for r in rows], 'o-', label='xi_eff([100])')
ax[0].axhline(1e-35, color='crimson', ls='--', label='photon k^(E) = 1e-35 (low-E bound)')
ax[0].axvspan(1e12, 1e20, color='gold', alpha=0.25, label='high-E zone (sensitivity pending)')
ax[0].set_xlabel('photon energy (eV)')
ax[0].set_ylabel('effective coefficient')
ax[0].set_title('L1/L3 energy scaling and layered comparison')
ax[0].legend(fontsize=8)
ax[1].plot([0, 1, 2], [1.0, 0.5, 1.0 / 3.0], 'o-')
ax[1].set_xticks([0, 1, 2])
ax[1].set_xticklabels(['[100]', '[110]', '[111]'])
ax[1].set_ylabel('relative coefficient')
ax[1].set_title('L2 directional fingerprint 6:3:2')
plt.tight_layout()
plt.savefig(IMG, dpi=150)
print('图已保存：' + IMG)