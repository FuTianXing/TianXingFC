# -*- coding: utf-8 -*-
"""primordial_seed_scales.py

原初扰动的来源核算：种子尺度、出生时视界、观测枢轴模三者的量级核对。
判据 S1 至 S5 先写后实现。
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

lP, tP, c = 1.616255e-35, 5.391247e-44, 2.99792458e8
Mpc = 3.0856775814913673e22
T0_eV = 2.348e-4
k_pivot = 0.05                      # Mpc^-1
lam_c = 2*np.pi/k_pivot*Mpc         # 共动波长（米）

def T_MeV(t, gs): return np.sqrt(2.42/np.sqrt(gs)/t)
def lam_phys(t, gs):
    T = T_MeV(t, gs)*1e6            # eV
    return lam_c/(T/T0_eV)

cases = [('下端 t_* = 4.083e-8 秒', 4.083e-8, 61.75), ('上端 t_* = 0.1 秒', 0.1, 10.75)]
xi_lo, xi_hi = 3.8028e21, 6.4e25    # 以 lP 为单位（统一口径重算的允许域）

print('='*76); print('原初扰动来源核算：三把尺子'); print('='*76)
print('枢轴模共动波长 = %.4g 米（k = %.2f Mpc^-1）' % (lam_c, k_pivot))
rows=[]
for name, t, gs in cases:
    T = T_MeV(t, gs); z = T*1e6/T0_eV
    dH = 2*c*t
    lp = lam_phys(t, gs)
    rows.append((name, t, T, z, dH, lp))
    print('-'*76)
    print('%s：T = %.3g MeV，z = %.3g' % (name, T, z))
    print('   出生时视界 2ct_* = %.4g 米' % dH)
    print('   枢轴模在出生时的物理波长 = %.4g 米，与视界之比 = %.4g' % (lp, lp/dH))
    print('   种子尺度域 = %.3g 至 %.3g 米，与视界之比 = %.3g 至 %.3g'
          % (xi_lo*lP, xi_hi*lP, xi_lo*lP/dH, xi_hi*lP/dH))
    print('   种子到下端的跨度 = %.3g 米，即 %.1f 个数量级'
          % (lp/(xi_lo*lP), np.log10(lp/(xi_lo*lP))))

print('='*76)
print('S1 枢轴模在出生时属超视界：两端比值均 >= 1e6 ->',
      all(r[5]/r[4] >= 1e6 for r in rows))
print('S2 种子尺度属深亚视界：最不利一端（种子域上端对下端视界）比值 %.3g，判 <= 1e-9 -> %s'
      % (max(xi_hi*lP/r[4] for r in rows), all(xi_hi*lP/r[4] <= 1e-9 for r in rows)))
spans=[np.log10(r[5]/x) for r in rows for x in (xi_lo*lP, xi_hi*lP)]
print('S3 种子到枢轴的跨度：%.1f 至 %.1f 个数量级' % (min(spans), max(spans)))
need = max(r[5] for r in rows)/2/lP
print('S4 若要求局域关联覆盖枢轴模，须 xi >= %.3g lP；允许域上端 %.3g lP，超出 %.1f 个数量级'
      % (need, xi_hi, np.log10(need/xi_hi)))
print('   判：局域（定居层）通道不可行')
print('S5 结论：能承担枢轴模的只可能是节律层（非局域）对象；')
print('   随之而来的要求是它的统计无特征尺度，否则谱在该尺度上出现转折。')
print('   附：先验纠缠若带关联长度 xi_e，则它在今日的共动尺寸 = xi_e*(1+z) 决定谱的转折位置；')
for name, t, gs, in [(c0[0], c0[1], c0[2]) for c0 in cases]:
    T = T_MeV(t, gs); z = T*1e6/T0_eV
    for x in (xi_lo*lP, xi_hi*lP):
        node = x*(1+z)/Mpc
        print('   %s：xi = %.3g 米 -> 折到今日的共动尺寸 %.3g Mpc（枢轴为 %.3g Mpc）'
              % (name[:10], x, node, lam_c/Mpc))

# ---------- 图 ----------
fig, ax = plt.subplots(1, 1, figsize=(8.2, 5.2))
labels = ['格距 lP', '缺陷宽度 w_def', '种子域下端', '种子域上端',
          '出生视界 2ct_*(下端)', '出生视界 2ct_*(上端)',
          '枢轴模(出生时,下端)', '枢轴模(出生时,上端)', '枢轴模(今日共动)']
vals = [lP, 3.8028e21*lP, xi_lo*lP, xi_hi*lP,
        2*c*4.083e-8, 2*c*0.1,
        lam_phys(4.083e-8, 61.75), lam_phys(0.1, 10.75), lam_c]
cols = ['#888','#1f4e79','#1f4e79','#1f4e79','#c00000','#c00000','#2e7d32','#2e7d32','#2e7d32']
y = np.arange(len(vals))[::-1]
ax.barh(y, np.log10(vals), color=cols, height=0.6)
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel('$\\log_{10}$（米）')
ax.set_title('原初扰动的尺度阶梯：种子、出生视界与观测量')
for yi, v in zip(y, vals):
    ax.text(np.log10(v)+0.4, yi, '%.2g m' % v, va='center', fontsize=8)
ax.grid(alpha=0.3, axis='x')
fig.tight_layout()
out = 'D:/software/TianXingFC/images/primordial_seed_scale_ladder.png'
fig.savefig(out, dpi=150)
print('图已存：', out)
