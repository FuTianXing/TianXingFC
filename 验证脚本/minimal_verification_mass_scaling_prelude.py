# -*- coding: utf-8 -*-
'''质量标度如何进入并合轨道：前置判据（第 554 条）。

撤回第 553 条归属裁决后的前置工作。第 553 条的 N2 判负测的是构造选择
（r ∝ 1/m），而非体系物理。本判据要回答：轨道半径对质量的依赖 r(m)
由体系的哪一条关系确定，以及在什么条件下适用。

候选构造与其体系依据：
  甲 稳态载荷式：theta(R) = G M / (c^2 R)  => R ∝ M
     依据：碰撞写入的稳态格林函数（时空观 §7 承载上限段）；
     适用对象：记录载荷的径向分布（静态），非轨道周期。
  乙 维里定理：R ∝ (M / n)^(1/3)（等体积）=> R ∝ M^(1/3)
     依据：若轨道由均匀介质引力支撑，维里定理定尺度；
     适用条件：需体系给出等体积密度假设。
  丙 普朗克标度入口：R = l_P (m / m_P)^alpha
     依据：格距刚性（晶格刚性原理）与满偏的归一化地位；
     适用条件：alpha 由体系的轨道-格距关系决定，本判据要定 alpha。

判据：
  P0 前置门：三种候选须各自给出可区分的幂（否则判据无区分力）
  Q1 甲的适用性：稳态载荷式是否给出轨道周期（而非静态载荷）
  Q2 乙的前提：等体积假设是否在体系内成立
  Q3 alpha 的可判定性：由满偏归一化 + 格距刚性能否定 alpha
  Q4 EP5：alpha 自由时判据失效（须有额外规范）

运行：python 验证脚本/minimal_verification_mass_scaling_prelude.py
确定性，无随机数。
'''
import math
import time
import numpy as np

T_START = time.time()

TP = LP = 1.0
C = LP / TP
OMEGA_P = 2 * np.pi / TP
KAPPA_PHI = 1.0 / (6 * np.pi * TP)
OMEGA_0 = math.sqrt(3 * OMEGA_P * KAPPA_PHI)
P_PN = 5.0 / 3.0

print('=' * 78)
print('质量标度前置判据（第 554 条）：r(m) 由哪条体系关系确定')
print('=' * 78)
print()

masses = np.array([0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0])


def p_fit(vals):
    return float(np.polyfit(np.log(masses), np.log(vals), 1)[0])


# ---------- P0 前置门 ----------
print('P0 前置门：三种候选须各自给出可区分的幂')
a_jia = 1.0      # R ∝ M
a_yi = 1.0 / 3.0  # R ∝ M^(1/3)
a_bing = 0.5     # R ∝ M^(1/2)（待定的 alpha）
exps = {'甲 稳态载荷式': a_jia, '乙 维里定理': a_yi, '丙 alpha=1/2': a_bing}
for k, v in exps.items():
    print('   %s：R ∝ M^%.4f -> omega 幂 = %.4f' % (k, v, -v))
p0_ok = len(set(round(-v, 3) for v in exps.values())) == 3
print('   [%s] P0 前置门（判据：三候选幂两两可区分）' % ('PASS' if p0_ok else 'FAIL'))
print()

# ---------- Q1 甲的适用性 ----------
print('Q1 甲（稳态载荷式 R ∝ M）的适用性')
print('   体系依据：碰撞写入稳态式 theta(R) = (m/m_P)(l_P/R) = G M/(c^2 R)')
print('   该式的对象：记录载荷的径向分布（静态，每拍写入的稳态）')
print('   并合轨道所需：引力波周期，即轨道尺度随时间的演化')
print('   判读：稳态式给静态载荷的空间分布，不直接给轨道周期；')
print('         若用于轨道，须附加「轨道尺度 = 满偏点尺度」的条件（额外假设）')
q1_ok = False   # 甲不直接适用
print('   [%s] Q1（甲对轨道周期不直接适用，须附加条件）' % '完成')
print()

# ---------- Q2 乙的前提 ----------
print('Q2 乙（维里定理 R ∝ M^(1/3)）的前提')
print('   前提：轨道由均匀介质引力支撑（等体积假设）')
print('   体系状态：碰撞写入的介质是各向异性的（公理 2 的三类正交方向 + 异类耦合）')
print('   判读：体系未给出等体积密度假设；晶格为各向异性格点阵，非均匀介质')
q2_ok = False
print('   [%s] Q2（等体积前提在体系内不成立，乙不适用）' % '完成')
print()

# ---------- Q3 alpha 的可判定性 ----------
print('Q3 alpha 的可判定性：由满偏归一化 + 格距刚性能否定 alpha')
# 满偏条件：m = m_P 时 theta = 1，轨道收缩至 r_s/2（弱场外推口径）
# 格距刚性：l_P 绝对不变，轨道最小尺度不低于 l_P
# 若 R = l_P (m/m_P)^alpha，满偏 m = m_P 给 R = l_P * 1 = l_P
# 但体系登记的满偏尺度是 r_s/2 = G m_P / c^2 = l_P（由 m_P = sqrt(hbar c/G)）
print('   满偏归一化：m = m_P 时 R = r_s/2 = G m_P/c^2')
print('   代入 m_P = sqrt(hbar c/G)：G m_P / c^2 = sqrt(hbar G / c^3) = l_P')
print('   故满偏尺度 = l_P（与格距同阶），与「R = l_P (m/m_P)^alpha」在 m=m_P 处相容')
print('   判读：alpha 在满偏端被归一化约束为 1（任何 alpha 在 m=m_P 给 R=l_P），')
print('         故 alpha 不可由满偏归一化确定；alpha 须由轨道动力学给出')
q3_ok = False   # alpha 不可由现有材料判定
print('   [%s] Q3（alpha 不可由满偏归一化判定，须轨道动力学）' % '完成')
print()

# ---------- Q4 EP5 ----------
print('Q4 EP5：alpha 自由时判据失效（须额外规范）')
om_alphas = {}
for alpha in [0.0, 0.5, 1.0, 5.0 / 6.0, 1.0]:
    r = LP * (masses ** alpha)
    om = (C ** 2 * (1.0 / r) / np.sqrt(C ** 2 * (1.0 / r) ** 2 + OMEGA_0 ** 2)) / r
    om_alphas[alpha] = p_fit(om)
print('   alpha -> omega 质量幂：')
for a, p in om_alphas.items():
    print('      alpha = %.4f -> p = %+.4f' % (a, p))
spread = max(om_alphas.values()) - min(om_alphas.values())
print('   幂的跨度 = %.4f（alpha 自由则幂自由，判据需外部规范才能定）' % spread)
q4_ok = spread > 0.1
print('   [%s] Q4（判据：alpha 自由时幂跨度 > 0.1，即需外部规范）' % ('PASS' if q4_ok else 'FAIL'))
print()

print('汇总：P0-%s Q1 完成 Q2 完成 Q3 完成 Q4-%s'
      % ('PASS' if p0_ok else 'FAIL', 'PASS' if q4_ok else 'FAIL'))
print()
print('前置判据结论：')
print('  其一，甲（稳态载荷式 R ∝ M）：体系有依据（碰撞写入稳态式），')
print('    但其对象是静态载荷的空间分布，不是轨道周期；用于轨道须附加条件。')
print('  其二，乙（维里定理 R ∝ M^(1/3)）：前提（等体积介质）在体系内不成立')
print('    （晶格各向异性），不适用。')
print('  其三，丙（alpha 自由）：满偏归一化在 m = m_P 处对任何 alpha 都给 R = l_P，')
print('    故 alpha 不可由满偏归一化确定；幂跨度 %.4f 证实需外部规范。' % spread)
print('  其四，故 r(m) 的构造在体系内**未确定**，前置判据判为「材料不足」：')
print('    需要轨道动力学的第一原理导出（轨道尺度与格距、与质量标度的关系），')
print('    该内容不在既有材料射程内。')
print()
print('对第 553 条的影响：归属裁决维持撤回（正确的），')
print('  且撤回理由升级：不仅幂差源于构造，且体系内尚无确定 r(m) 的材料。')
print('耗时 %.1fs' % (time.time() - T_START))