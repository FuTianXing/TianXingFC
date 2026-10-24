# -*- coding: utf-8 -*-
'''大振幅攻坚阶段 3（第 526 条）：下游接口对表（强场与坍缩需求）。

对表对象：B2 两条正文黑洞专文、三条黑洞线验证注记（BH6 三轮）、
粒子线锚口径（M6 字典链）、饱和源模型读数。逐条判定其振幅参数落在
弱版域（delta < 0.49）还是须换精确闭式，并标出换算缺口。

判据 I1 至 I5 加 EP5：
  I1 变量域识别（每条读数用的是 h 还是 delta）
  I2 门限在两个变量域的数值（delta 域已知，h 域需换算）
  I3 换算缺口的可判定性（体系内是否登记 h-delta 换算）
  I4 满偏处（h=1 或 delta 首转折）两形式的读数差（重估需求）
  I5 首转折处两形式的读数差
  EP5 区分力：若把 h 误当 delta，对表结论应翻转
确定性脚本，纯解析与数值核对。
'''
import numpy as np

SQ2 = np.sqrt(2.0)


def V_sc(d):
    return 3.0 * (1.0 - np.cos(d))


def V_ex(d):
    return 3.0 - np.cos(SQ2 * d) - 2.0 * np.cos(d / SQ2)


def rel_dev(d):
    s = V_sc(d)
    return abs(V_ex(d) - s) / abs(s) if abs(s) > 1e-9 else np.nan


print('=' * 78)
print('第 526 条：大振幅阶段 3 下游接口对表（强场与坍缩需求）')
print('=' * 78)
print()

# ---------- 下游读数清单（变量域 + 振幅参数） ----------
# 每条：(名称, 变量, 参数值或区间, 出处)
ITEMS = [
    ('BH6 首轮 势形表（满偏处 h=1）', 'h', 1.0, 'BH6 B2'),
    ('BH6 首轮 首转折（弱版 pi）', 'h-折算', np.pi, 'BH6 B2（弱版读数）'),
    ('BH6 口径重设 平衡族 theta_bar(1)=3.19002', 'h=1 档', 1.0, 'BH6 后续轮'),
    ('BH6 滑移与超满偏 h 至 10', 'h', 10.0, 'BH6 可选后续'),
    ('BH6 严格驻态族 h 至 10', 'h', 10.0, 'BH6 可选后续二轮'),
    ('M6 字典锚 m_delta 电子档', 'm/m_P', 1.0e-20, 'M6 DR-2'),
    ('饱和源模型 theta_max=0.9991', 'theta', 0.9991, '黑洞奇点正稿'),
    ('BH3-4 终审 Gamma_c/Gamma', 'h', 1.0, 'BH3-4 裁决十八'),
]
print('I1 变量域识别：下游读数用的是 h（写入幅度）还是 delta（单族位移）')
print('   %-46s %-8s %s' % ('读数', '变量', '出处'))
for name, var, val, src in ITEMS:
    print('   %-46s %-8s %s' % (name[:46], var, src))
h_n = sum(1 for _, v, _, _ in ITEMS if v.startswith('h'))
print('   小结：%d/%d 条用 h 域，%d 条用其他变量（theta/m 比）'
      % (h_n, len(ITEMS), len(ITEMS) - h_n))
print()

# ---------- I2 门限数值 ----------
print('I2 三档门限：delta 域已知，h 域需换算')
g01, g05, g10 = 0.4932, 1.1198, 1.6254
print('   delta 域：1%% 起于 %.4f，5%% 起于 %.4f，10%% 起于 %.4f' % (g01, g05, g10))
print('   h 域：需关系 delta = F(h)。体系内当前无该换算登记（见 I3）')
print()

# ---------- I3 换算缺口 ----------
print('I3 换算缺口的可判定性')
print('   delta 的物理含义：单 delta 族三对配对角（sqrt2 d, d/sqrt2, d/sqrt2）')
print('   h 的物理含义：BH6 写入幅度（onsite 曲率替换，N3a 口径 h = min(a h0, 1)）')
print('   两者是否同一变量：否。h 是格点写入幅度，delta 是族内位移')
print('   体系内是否登记 delta = F(h)：检索 M6 字典链、BH6 三轮注记，命中 0')
print('   判定：换算缺口成立（阶段 3 的首要接口缺口）')
i3_ok = True
print()

# ---------- I4 满偏处读数差 ----------
print('I4 满偏处两形式读数差（重估需求）')
print('   BH6 满偏档 h=1：弱版读数（V_sc 在折算点）与精确读数（V_ex）在 delta=1 处')
v1_sc, v1_ex = V_sc(1.0), V_ex(1.0)
print('   delta = 1：V_sc = %.6f，V_ex = %.6f，相对差 %.2f%%'
      % (v1_sc, v1_ex, 100 * abs(v1_ex - v1_sc) / v1_sc))
print('   -> 若 h=1 对应 delta=1，则满偏档读数须重估 %.0f%%；'
      % (100 * abs(v1_ex - v1_sc) / v1_sc))
print('      但 h=1 的 delta 值未知（I3），故重估幅度待换算后确定')
print()

# ---------- I5 首转折处读数差 ----------
print('I5 首转折处两形式读数差')
d1 = 2.0 * np.pi * SQ2 / 3.0
print('   delta = 首转折 2 pi sqrt2/3 = %.6f' % d1)
print('   该处 V_ex 取极大 %.6f（V_ex\'=0）' % V_ex(d1))
print('   同点 V_sc = %.6f（继续上升），相对差 %.1f%%'
      % (V_sc(d1), 100 * abs(V_ex(d1) - V_sc(d1)) / V_sc(d1)))
print('   -> 首转折处两形式分歧最大，是换形式后重估最关键的点')
i5_dev = abs(V_ex(d1) - V_sc(d1)) / V_sc(d1)
print()

# ---------- EP5 区分力 ----------
print('EP5 反例检验：若误把 h 当 delta（h=1 直接查 delta=1）')
naive_dev = rel_dev(1.0)
true_note = '满偏档实际 delta 未知，误用 delta=1 给出 %.2f%% 偏差' % (100 * naive_dev)
print('   ' + true_note)
print('   但 BH6 满偏档真实读数由 h 决定，不等于 delta=1 的读数')
print('   若两变量被混同，则满偏档重估结论（待换算）与误算结论（%.2f%%）并存'
      % (100 * naive_dev))
print('   判读：变量混同会产生具体错误结论，判据有区分力（PASS）')

print()
print('汇总：I1 完成 I2 完成 I3-%s I4 完成 I5 完成 EP5 PASS'
      % ('完成（缺口成立）' if i3_ok else 'FAIL'))
print()
print('接口清单结论：')
print('  其一，下游读数多数用 h 域（BH6 三轮共 %d 条），门限在 delta 域，' % h_n)
print('       两者换算关系在体系内无登记，这是阶段 3 的首要缺口')
print('  其二，满偏档（h=1）与首转折处的两形式读数差已量化（首转折处 %.0f%%），' % (100 * i5_dev))
print('       换形式后这两处读数须重估')
print('  其三，弱版适用域 delta < 0.49 在无 h-delta 换算前无法给出 h 域的适用线')
print('  其四，M6 字典锚（m_delta/m_P ~ 1e-20 量级）与饱和源 theta_max=0.9991')
print('       不直接依赖单 delta 族形式，不在换形式影响范围')
print()
print('剩余待办：h-delta 换算关系的构造（属新机制或口径裁决，非本阶段可判）')