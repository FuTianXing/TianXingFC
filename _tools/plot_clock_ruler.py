# -*- coding: utf-8 -*-
"""科普配图：钟与尺：引力的两个读数（三栏示意图）。
(a) 旧构造：每站一个数，段上没有自己的读数；(b) 新构造：每段带自己的数；
(c) 两支读数落在同一条曲线上（同点比值恒为一）。
输出：images/clock_ruler_schematic.png（绝对路径写入，工作区移动需同步改）。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, axs = plt.subplots(1, 3, figsize=(13.2, 4.0))
n = 5
xs = np.arange(n)

# (a) 旧构造
ax = axs[0]
ax.plot([xs[0], xs[-1]], [0, 0], '-', color='#999999', lw=1.5, zorder=1)
ax.plot(xs, [0] * n, 'o', ms=11, color='#1f77b4', zorder=2)
for i, x in enumerate(xs):
    ax.text(x, 0.30, r'$\theta_{%d}$' % (i + 1), ha='center', fontsize=9, color='#1f77b4')
for i in range(n - 1):
    ax.text((xs[i] + xs[i + 1]) / 2, -0.30, '？', ha='center', fontsize=13, color='#d62728')
ax.text((xs[0] + xs[-1]) / 2, -0.85, '段上没有自己的数', ha='center', fontsize=10, color='#d62728')
ax.annotate('钟读得到', xy=(xs[2], 0.16), xytext=(xs[2] + 0.4, 0.95), fontsize=10,
            arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=1.3), color='#2ca02c')
ax.text((xs[0] + xs[-1]) / 2, 1.55, '(a) 旧构造：每站一个数', ha='center', fontsize=11)
ax.set_xlim(-0.8, n - 0.2); ax.set_ylim(-1.3, 1.9); ax.axis('off')

# (b) 新构造
ax = axs[1]
for i in range(n - 1):
    ax.plot([xs[i] + 0.12, xs[i + 1] - 0.12], [0, 0], '-', color='#ff7f0e', lw=4, zorder=1)
    ax.annotate('', xy=(xs[i] + 0.42, 0.20), xytext=(xs[i] + 0.14, 0.20),
                arrowprops=dict(arrowstyle='->', color='#ff7f0e', lw=1.3))
    ax.text((xs[i] + xs[i + 1]) / 2, -0.32, r'$u_{%d}$' % (i + 1), ha='center', fontsize=9, color='#ff7f0e')
ax.plot(xs, [0] * n, 'o', ms=11, color='#1f77b4', zorder=2)
ax.text((xs[0] + xs[-1]) / 2, -0.87, '每段带上自己的数（含方向）', ha='center', fontsize=10, color='#ff7f0e')
ax.annotate('钟 ← 站点合成', xy=(xs[3], 0.15), xytext=(xs[3] - 2.1, 0.95), fontsize=10,
            arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=1.3), color='#2ca02c')
ax.annotate('尺 ← 段自己', xy=((xs[1] + xs[2]) / 2, -0.15), xytext=((xs[1] + xs[2]) / 2 + 0.5, 1.5), fontsize=10,
            arrowprops=dict(arrowstyle='->', color='#9467bd', lw=1.3), color='#9467bd')
ax.text((xs[0] + xs[-1]) / 2, 1.95, '(b) 新构造：每段一个数', ha='center', fontsize=11)
ax.set_xlim(-0.8, n - 0.2); ax.set_ylim(-1.3, 2.2); ax.axis('off')

# (c) 两支读数重合
ax = axs[2]
x = np.linspace(0.0, 10.0, 2001)
tent = 0.45 * np.minimum(x, 10.0 - x)
ax.plot(x, tent, '-', color='#1f77b4', lw=1.6, label='记录剖面')
xe = np.arange(1, 10)
ax.plot(xe, 0.45 * np.minimum(xe, 10.0 - xe), 'o', ms=5, color='#2ca02c', label='链接读数')
xs_ = np.arange(0.5, 9.6, 1.0)
ax.plot(xs_, 0.45 * np.minimum(xs_, 10.0 - xs_), 's', ms=4.5, mfc='none', color='#d62728',
        label='站点读数（半格对位）')
ax.text(5.0, 1.65, '两支读数在同一条曲线上重叠', ha='center', fontsize=10)
ax.text(5.0, 1.40, '同点比值恒为 1（λ = 1）', ha='center', fontsize=10, color='#1f77b4')
ax.legend(fontsize=9, loc='upper left', frameon=False)
ax.text(5.0, 2.05, '(c) 钟与尺由同一条曲线供给', ha='center', fontsize=11)
ax.set_xlim(-0.4, 10.4); ax.set_ylim(-0.15, 2.3); ax.axis('off')

fig.tight_layout()
fig.savefig("D:/software/TianXingFC/images/clock_ruler_schematic.png", dpi=150)
print("图已保存：images/clock_ruler_schematic.png")