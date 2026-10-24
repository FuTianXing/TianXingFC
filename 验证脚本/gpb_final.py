# -*- coding: utf-8 -*-
"""
gpb_final.py

GPB 终值定量复核（判据第 646 条）：引力线矢量块的外部余项。

判据：
  U1（拖曳侧）：体系预测的参考系拖曳岁差（标准强度，等于 GR 值）与 GPB 终值的
      偏离不超过 1 sigma。
  U2（测地侧）：体系预测的测地岁差（标准强度）与 GPB 终值的偏离不超过 1 sigma。
  U3（强度区间）：强度取 1 落在 1 sigma 允许区间内。
  U4（恒真反例，须 FAIL）：旧预言（拖曳岁差为零，奇偶门未打开前的框架读数）与
      GPB 终值的偏离须超过 3 sigma。

模型设定（数据来源显式声明）：
  - 观测值取 GPB 终值（Everitt 等，2011）：参考系拖曳 37.2 +- 7.2 毫角秒每年、
    测地岁差 6601.8 +- 18.3 毫角秒每年；库内 2026-09 记录的两项（拖曳 37.2 +- 7.2、
    旧预言偏离 5.17 sigma）与之同源。
  - GR 预测值取 GPB 论文所列：拖曳 39.2、测地 6606.1 毫角秒每年。
  - 体系预测：2026-09-20 补入动量读数项（逐拍沿缺陷世界线的四维流，对时间反演为奇）
    后奇偶门打开，矢量块读数改为标准强度，故体系预测等于 GR 值；这一层是"体系给出
    标准强度"的读数，不是独立数值预言。
  - 网络不可用（WebFetch 超时），数据取自库内记录与公开标准值；若 GPB 数据版本更新
    须同步。

边界声明：本轮是外部数据对照，核验"体系在打开奇偶门后的矢量块读数与 GPB 终值一致"；
不重做矢量块的内部推导（四轮判据 Z/K/P/L 与定理 7 已在库）、不裁决矢量块动力学
（自作用迭代与量子化仍属纲领层）。

可逆性等级：判据敏感。结论依赖外部数据版本（观测值与其误差）与 GR 预测值取值；
换数据版本会移动 sigma，但"体系读数等于标准强度"这一层不变。

协议伪影与物理结论分层：sigma 数值属外部对照读数；"体系矢量块不落入零拖曳旧预言"
这一结构结论不受数据版本影响。
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---------- 数据（GPB 终值） ----------
OBS_DRAG, ERR_DRAG = 37.2, 7.2       # 毫角秒每年
OBS_GEO, ERR_GEO = 6601.8, 18.3
GR_DRAG, GR_GEO = 39.2, 6606.1       # GR 预测值（GPB 论文所列）

# 体系预测 = 标准强度 = GR 值
SYS_DRAG, SYS_GEO = GR_DRAG, GR_GEO

sig_drag = abs(SYS_DRAG - OBS_DRAG) / ERR_DRAG
sig_geo = abs(SYS_GEO - OBS_GEO) / ERR_GEO
sig_old = abs(0.0 - OBS_DRAG) / ERR_DRAG

lo_1sig = (OBS_DRAG - ERR_DRAG) / GR_DRAG
hi_1sig = (OBS_DRAG + ERR_DRAG) / GR_DRAG

results = []
results.append(('U1', u'拖曳侧 sigma（体系标准强度对 GPB 终值）', sig_drag,
                u'<= 1 sigma', sig_drag <= 1.0))
results.append(('U2', u'测地侧 sigma（体系标准强度对 GPB 终值）', sig_geo,
                u'<= 1 sigma', sig_geo <= 1.0))
results.append(('U3', u'强度 1 是否落在 1 sigma 允许区间内',
                1.0 if (lo_1sig <= 1.0 <= hi_1sig) else 0.0,
                u'须为 1', lo_1sig <= 1.0 <= hi_1sig))
results.append(('U4', u'旧预言（零拖曳）sigma（须 FAIL）', sig_old,
                u'> 3 sigma', sig_old > 3.0))

readings = [
    ('R1', u'GPB 终值拖曳（毫角秒每年）', OBS_DRAG, u'误差 %.1f' % ERR_DRAG),
    ('R2', u'GR 预测拖曳（毫角秒每年）', GR_DRAG, u'体系预测同此'),
    ('R3', u'1 sigma 允许的强度区间下界', lo_1sig, u'无量纲强度'),
    ('R4', u'1 sigma 允许的强度区间上界', hi_1sig, u'无量纲强度'),
    ('R5', u'GPB 终值测地（毫角秒每年）', OBS_GEO, u'误差 %.1f' % ERR_GEO),
    ('R6', u'GR 预测测地（毫角秒每年）', GR_GEO, u'体系预测同此'),
]

# ---------- 打印 ----------
print('=' * 78)
print(u'GPB 终值定量复核（第 646 条）')
print('=' * 78)
n_pass = 0
for tag, name, val, expect, ok in results:
    n_pass += ok
    print('%s %-40s 实测 %.12g  期望 %-14s %s' % (
        tag, name, val, expect, 'PASS' if ok else 'FAIL'))
print('-' * 78)
print(u'判据 %d/%d 通过' % (n_pass, len(results)))
print('-' * 78)
for tag, name, val, expect in readings:
    print('%s %-40s 实测 %.12g  （%s）' % (tag, name, val, expect))
print('-' * 78)
print(u'体系（标准强度）对 GPB 终值：拖曳 %.3f sigma、测地 %.3f sigma；'
      u'旧预言（零拖曳）%.2f sigma' % (sig_drag, sig_geo, sig_old))
assert n_pass == len(results), 'FAIL: 判据未全过'

# ---------- 图：sigma 随强度 ----------
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
ax = axes[0]
ss = np.linspace(0.0, 1.5, 300)
ax.plot(ss, np.abs(ss * GR_DRAG - OBS_DRAG) / ERR_DRAG, lw=2, color='C0',
        label=u'拖曳侧 $\\sigma(s)$')
ax.axhline(1.0, color='C1', ls='--', lw=1.2, label=u'$1\\sigma$')
ax.axhline(3.0, color='C3', ls=':', lw=1.2, label=u'$3\\sigma$')
ax.axvline(1.0, color='k', ls='-.', lw=1.2, label=u'体系强度 $s=1$')
ax.axvline(0.0, color='C3', ls='-', lw=1.0, alpha=0.6, label=u'旧预言 $s=0$')
ax.set_xlabel(u'强度 s（相对标准强度）')
ax.set_ylabel(u'对 GPB 终值的 $\\sigma$')
ax.set_title(u'(a) 体系强度落在 $1\\sigma$ 区间，旧预言在 $3\\sigma$ 之外')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1]
labels = [u'体系（标准强度）', u'旧预言（零拖曳）']
vals = [sig_drag, sig_old]
bars = ax.bar(labels, vals, color=['C0', 'C3'], width=0.5)
ax.axhline(1.0, color='C1', ls='--', lw=1.2, label=u'$1\\sigma$')
ax.axhline(3.0, color='C3', ls=':', lw=1.2, label=u'$3\\sigma$')
ax.set_ylabel(u'对 GPB 终值的 $\\sigma$')
ax.set_title(u'(b) 两种预言对 GPB 终值的偏离')
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.1, u'%.2f' % v, ha='center', fontsize=9)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(r'D:\software\TianXingFC\images\gpb_final_check.png', dpi=150)
print(u'figure saved: images/gpb_final_check.png')
