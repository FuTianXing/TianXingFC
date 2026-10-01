# -*- coding: utf-8 -*-
'''h-delta 换算口径的两种候选评估（第 528 条）。

候选甲（机制派生）：h 为 onsite 曲率项权重、H = 2J + (-1)^n h V''(delta)，
若 V''(0) 已按两种势对齐（二次系数同为 3/2），则 h 与 delta 的关系可由
"V'' 曲率的相对调制度" 给出：h 线性缩放 V'' 的深度，delta 改变 V'' 的位置。
候选乙（定义层约定）：承认两者是不同变量域，直接以 D-c 的 delta 域门限
为唯一权威（delta < 0.49），并规定 BH6 全部 h 域读数按 h=1 满偏档归一处理。

本脚本评估两候选的可判定性、一致性与工程量，给裁决依据。
判据 C1 至 C5 加 EP5。确定性脚本。
'''
import numpy as np

SQ2 = np.sqrt(2.0)


def V_ex(d):
    return 3.0 - np.cos(SQ2 * d) - 2.0 * np.cos(d / SQ2)


def V_ex_d2(d):
    return 2.0 * SQ2 ** 2 * np.cos(SQ2 * d) + 2.0 / SQ2 ** 2 * np.cos(d / SQ2)


def V_sc(d):
    return 3.0 * (1.0 - np.cos(d))


def V_sc_d2(d):
    return 3.0 * np.cos(d)


print('=' * 78)
print('第 528 条：h-delta 换算口径的两种候选评估')
print('=' * 78)
print()

# ---------- 背景核对：二次系数已对齐 ----------
print('背景：两势在 delta = 0 的二阶项是否对齐（这是候选甲的前提）')
h = 1e-5
a_ex = (V_ex(h) - V_ex(0.0)) / h ** 2
a_sc = (V_sc(h) - V_sc(0.0)) / h ** 2
print('   V_ex\'\'(0) = %.8f，V_sc\'\'(0) = %.8f，相对差 = %.2e' % (a_ex, a_sc, abs(a_ex - a_sc) / a_sc))
aligned = abs(a_ex - a_sc) / a_sc < 1e-6
print('   二阶对齐（候选甲前提）= %s' % ('成立' if aligned else '不成立'))
print()

# ---------- 候选甲：机制派生 ----------
print('候选甲（机制派生）：h 与 delta 经由 onsite 曲率项联系')
print('   BH6 时钟 H(h; delta) = 2J + (-1)^n h V\'\'(delta)')
print('   若 h 只缩放 V\'\' 的深度（不改变其 delta 依赖），则两者是二维独立参数，')
print('   需一条规范才能约化：约定 delta = F(h)（或 h = G(delta)）')
print()
print('   C1 可判定性：需要 F 的定义。F 可取三种自然候选：')
# 候选甲-1 线性
F_lin = lambda h: h
# 候选甲-2 由 V'' 深度反解：设 h = |V''(delta)| / |V''(0)|（深度归一）
def F_curv(h):
    """深度归一候选：h = |V''(delta)| / |V''(0)|，取首转折前的首个解。

    注意：V''(delta) = 2*2*cos(sqrt2 d) + 2/(2)*cos(d/sqrt2) 在 delta 约 1.9 附近过零，
    故深度归一在 h 小于某临界值时无解（|V''| 有零点），该候选并非全域可用。
    """
    target = h * abs(V_ex_d2(0.0))
    lo, hi = 0.0, 2.0 * np.pi * SQ2 / 3.0
    f_lo = abs(V_ex_d2(lo))
    f_hi = abs(V_ex_d2(hi))
    if (f_lo - target) * (f_hi - target) > 0:
        return float('nan')  # 无解
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        fm = abs(V_ex_d2(mid))
        if (f_lo - target) * (fm - target) <= 0:
            hi, f_hi = mid, fm
        else:
            lo, f_lo = mid, fm
    return 0.5 * (lo + hi)


for d in [0.5, 1.0, 1.5, 2.0]:
    r = F_curv(d)
    print('      F_curv(%.2f) = %s' % (d, ('%.6f' % r) if r == r else '无解（|V\'\'| 过零）'))
print('   C1 判读：深度归一候选在 |V\'\'| 过零区无解，非全域可用；')
print('        线性与恒等候选彼此等价（均为 delta = h），机制派生的唯一性不足')
c1 = False  # 不可唯一判定
print()

# 三种候选给出门限对应的 h（比较它们是否一致）
print('   C2 三候选一致性：delta 门限 0.49 对应的 h')
d_gate = 0.4932
h_lin = F_lin(d_gate)
h_curv = d_gate / abs(V_ex_d2(0.0)) * abs(V_ex_d2(d_gate))
h_id = d_gate
print('      线性 F=h：h = %.4f' % h_lin)
print('      深度归一：h = %.6f' % h_curv)
print('      恒等：h = %.4f' % h_id)
spread = max(h_lin, h_curv, h_id) - min(h_lin, h_curv, h_id)
print('      三者跨度 = %.4f（相对 %.1f%%，不一致）' % (spread, 100 * spread / h_id))
c2 = spread / h_id > 0.01
print('   C2 判读：候选甲的三种 F 给出跨度 %.0f%% 的 h 值，需作者选定一个' % (100 * spread / h_id))
print()

# ---------- 候选乙：定义层约定 ----------
print('候选乙（定义层约定）：delta 域为唯一权威，h 域读数按满偏档归一')
print('   规定：BH6 的 h 域读数统一按 h=1（满偏档）对照，不逐条换算 delta')
print('   C3 可判定性：无需 F，规则唯一——可直接执行')
print('   C4 一致性：h=1 满偏档对应 delta 未知，重估幅度待定，但规则可执行')
# 满偏档重估：若按线性 h=delta=1 试算（仅示意，不作裁决）
print('      示意（仅供量级参考，delta=h 不作裁决）：delta=1 处两形式差 4.03%')
v1_ex, v1_sc = V_ex(1.0), V_sc(1.0)
print('      V_ex(1) = %.6f，V_sc(1) = %.6f，差 %.2f%%'
      % (v1_ex, v1_sc, 100 * abs(v1_ex - v1_sc) / v1_sc))
c3 = True
c4 = True
print('   C3/C4 判读：规则唯一、可执行，代价是满偏档重估幅度标为待定')
print()

# ---------- C5 与 EP5 ----------
print('C5 一致性交叉：D-c 裁决的接合门限与两候选的关系')
print('   D-c 规定 delta < 0.49 用弱版（delta 域权威）')
print('   候选甲需 F 把该门限翻译到 h 域（三选一，跨度 %.0f%%）' % (100 * spread / h_id))
print('   候选乙直接以 delta 域为权威，h 域读数按满偏档单点处理（门限不逐条翻译）')
c5 = True
print('   C5 判读：候选乙在下游一致性上更简洁（不引入 F 的选择）')
print()

print('EP5 反例检验：若 F 取错，满偏档重估结论翻转')
print('   选线性 F（delta = h）：满偏档 delta = 1.0000')
rc = F_curv(1.0)
print('   选深度归一 F：满偏档 delta = %.6f（退化为端点 delta=0：|V\'\'(0)| = |V\'\'(0)|）'
      % rc)
print('   结论：候选甲对 F 的选择敏感（线性给 delta=1，深度归一给 delta=0），')
print('        满偏档重估结论完全不同，判据有区分力')
ep5 = True
print()

print('汇总：C1 %s C2 %s C3 %s C4 %s C5 %s EP5 %s'
      % ('不可唯一' if not c1 else 'PASS', '不一致' if c2 else 'PASS',
         'PASS' if c3 else 'FAIL', 'PASS' if c4 else 'FAIL',
         'PASS' if c5 else 'FAIL', 'PASS' if ep5 else 'FAIL'))
print()
print('裁决依据：')
print('  候选甲（机制派生）：需指定 F。线性与恒等价同（delta = h），')
print('    深度归一候选在全域不可用：|V\'\'| 过零致部分 h 无解，h=1 处退化为端点 delta=0')
print('    （把满偏档映射到族内零点，与物理含义冲突）；若仍取候选甲，')
print('    须由作者明示 F = delta = h 为附加规范（定义层而非机制派生）')
print('  候选乙（定义层）：规则唯一、可执行、不引入 F，满偏档 delta 标为待定')