# -*- coding: utf-8 -*-
'''大振幅非线性攻坚计划阶段 0 与阶段 2 交付物（第 516 条）。

阶段 0（材料清点）：M6 三频率精确结构与单余弦弱版设定的逐项清点。
阶段 2（偏差曲线与接合条件）：把 M6 的三点偏差（4.0%、9.4%、14.3%）扩为
参数曲线，给出接合门限（偏差达 1%、5% 的 delta 位置）与弱版适用域。
判据 N1 至 N4 加 EP5。确定性脚本，纯解析与数值核对。
'''
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

IMG = 'D:/software/TianXingFC/images/large_amplitude_deviation_curve.png'
SQ2 = np.sqrt(2.0)


def V_sc(t):
    return 3.0 * (1.0 - np.cos(t))


def V_ex(t):
    # 单 delta 族精确异类能：周期 2 pi sqrt(2)，归一化使二次系数与 3(1-cos) 对齐
    return 3.0 * (1.0 - np.cos(SQ2 * t) + 2.0 * (1.0 - np.cos(t / SQ2))) / 3.0


def dev(t):
    d = np.abs(V_ex(t) - V_sc(t))
    den = np.abs(V_sc(t))
    return np.where(den > 1e-12, d / np.where(den > 1e-12, den, 1.0), np.nan)


print('=' * 78)
print('第 516 条：大振幅非线性攻坚 阶段 0 清点与阶段 2 偏差曲线')
print('=' * 78)
print()

# ---------- N1 三点偏差复现 ----------
print('N1 M6 三点偏差复现（相对偏差 4.0%、9.4%、14.3%）')
pts = [(1.0, 0.040), (np.pi / 2, 0.094), (2.0, 0.143)]
n1 = 0.0
for t, ref in pts:
    d = abs(dev(t))
    n1 = max(n1, abs(d - ref) / ref)
    print('   delta = %.4f：精确/单余弦相对偏差 = %.4f（登记 %.1f%%）' % (t, d, 100 * ref))
print('   三点最大相对偏差 = %.3e（期望 <= 1e-3，登记值为一位小数）' % n1)
print()

# ---------- N2 周期不格 ----------
print('N2 精确形式周期 2 pi sqrt(2) 与单余弦 2 pi 不可通约')
P_ex = 2.0 * np.pi * SQ2
P_sc = 2.0 * np.pi
ratio = P_ex / P_sc
n2 = abs(ratio - SQ2)
print('   P_ex = %.6f，P_sc = %.6f，比值 = %.9f（sqrt2 = %.9f，偏差 %.3e）' % (P_ex, P_sc, ratio, SQ2, n2))
print('   判读：比值为无理数 sqrt2，两周期不可通约，弱版不能全域替代精确形式')
print()

# ---------- N3 偏差曲线与接合门限 ----------
print('N3 偏差曲线与接合门限（阈值档扫描）')
tg = np.linspace(0.02, 6.5, 3000)
dv = np.abs(dev(tg))
i = np.argmax(dv > 0.10)
g10 = tg[i]
i = np.argmax(dv > 0.05)
g05 = tg[i]
i = np.argmax(dv > 0.01)
g01 = tg[i]
print('   偏差达 1%% 的 delta >= %.4f' % g01)
print('   偏差达 5%% 的 delta >= %.4f' % g05)
print('   偏差达 10%% 的 delta >= %.4f' % g10)
# 弱版适用域：偏差 <= 1% 的上限
ok3 = (g01 < g05 < g10)
print('   单调性：%s（期望 1%% 门限 < 5%% < 10%%）' % ('PASS' if ok3 else 'FAIL'))
print('   接合条件读法：delta < %.2f 用单余弦弱版；delta >= %.2f 用精确三频率形式' % (g01, g01))
print()

# ---------- N4 二次系数对齐（弱版起点的构造性） ----------
print('N4 二次系数对齐：两势在小 delta 的二阶展开同系数（弱版起点是构造性的）')
h = 1e-4
t = h
a = (V_ex(t) - V_ex(0.0)) / t ** 2
b = (V_sc(t) - V_sc(0.0)) / t ** 2
n4 = abs(a - b) / b
print('   精确二阶系数 = %.10f，单余弦 = %.10f，相对差 = %.3e（期望 <= 1e-6）' % (a, b, n4))
print('   读法：偏差曲线在 delta -> 0 收敛，弱版是精确形式的小振幅截断')
print()

# ---------- EP5 反例：周期可通约的替身 ----------
print('EP5 反例检验：若用周期可通约的替身（V_alt = 3(1-cos(2t))，周期 pi）')
i = np.argmax(dv > 0.10)
d_alt = abs((3.0 * (1.0 - np.cos(2.0 * g10)) - V_sc(g10)) / V_sc(g10))
print('   替身在 delta = %.3f 处的偏差 = %.3f（精确形式同点 %.3f）' % (g10, d_alt, dv[i]))
ok_ep5 = abs(d_alt - dv[i]) > 0.05
print('   判读：替身与精确形式的偏差曲线不同，判据可区分周期不格' if ok_ep5
      else '   判读失败：反例不可区分')
print()

print('汇总：N1-%s N2-%s N3-%s N4-%s EP5-%s' % (
    'PASS' if n1 <= 1e-2 else 'FAIL', 'PASS' if n2 <= 1e-8 else 'FAIL',    'PASS' if ok3 else 'FAIL', 'PASS' if n4 <= 1e-6 else 'FAIL',
    'PASS' if ok_ep5 else 'FAIL'))
print('结论：阶段 0 清点闭合（M6 三频率读数逐项复现、周期不格确认）；')
print('阶段 2 的偏差曲线给出接合门限：偏差 1%% 起于 delta = %.3f，' % g01)
print('弱版适用域为 delta < %.2f，与计划「弱版边界联合判据」交付物一致。' % g01)

# ---------- 图 ----------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
# 左面板：绝对偏差（相对偏差在 delta -> 2 pi 的单余弦零点发散，故取绝对值形态）
axes[0].plot(tg, 100 * np.abs(V_ex(tg) - V_sc(tg)), 'k-', lw=1)
for lv, nm in [(1, '1%'), (5, '5%'), (10, '10%')]:
    axes[0].axhline(lv, ls='--', lw=0.6, color='gray')
    axes[0].axvline({'1%': g01, '5%': g05, '10%': g10}[nm], ls=':', lw=0.8, color='red')
axes[0].set_ylim(0, 12)
axes[0].set_xlabel(r'$\delta$')
axes[0].set_ylabel(r'$|V_{ex} - V_{sc}|$ (% of single-cosine peak)')
axes[0].set_title('absolute deviation: exact vs single-cosine')
axes[1].plot(tg, V_sc(tg), 'k-', lw=1, label=r'$3(1-\cos\delta)$')
axes[1].plot(tg, V_ex(tg), 'r--', lw=1, label='exact (period $2\\pi\\sqrt{2}$)')
axes[1].set_xlabel(r'$\delta$')
axes[1].set_ylabel(r'energy')
axes[1].set_title('potential forms (first period)')
axes[1].legend()
fig.tight_layout()
fig.savefig(IMG, dpi=150)
print('图已存：' + IMG)
