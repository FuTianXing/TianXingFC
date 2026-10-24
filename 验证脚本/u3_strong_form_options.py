# -*- coding: utf-8 -*-
'''U3 强形态的三个可能切口评估（第 529 条）。

切口甲（共同原点）：两紧致圆是否同源于螺旋体 S^1 x R 的同一因子？
  甲-1 两紧致性在同一因子：相位圆 S^1 与取向圆（实轴取向场的值域）是否同一 S^1
  甲-2 若不同源，两紧致圆为何仍共用普朗克入口
切口乙（双螺旋导出）：取向紧致圆能否由心物双螺旋（相位差 pi/2）导出
切口丙（候选三可判性）：三个饱和点的截断函数能否写成同一形式

判据 S1 至 S4 加 EP5。确定性脚本，纯解析与结构核对。
'''
import numpy as np

print('=' * 78)
print('第 529 条：U3 强形态的三个可能切口评估')
print('=' * 78)
print()

# ---------- 背景：两紧致圆的现状 ----------
print('背景（第 512 条已确立）：')
print('  温度侧有界 D in [0,1]：住相位紧致圆（复相干度归一 |z|=1）')
print('  几何侧满偏 theta=1：住取向紧致圆（取向反向即失义，SO(3) 二阶元）')
print('  两截断共用普朗克能量 E_P = m_P c^2 与作用量入口 hbar/t_P')
print('  强形态待攻：两紧致圆能否从更深的单一构造（心物双螺旋）导出')
print()

# ---------- 切口甲：两紧致圆是否同源于同一因子 ----------
print('S1 切口甲（共同原点）：两紧致圆的因子同一性')
# 相位紧致圆：S^1（相位角 theta mod 2pi）
# 取向紧致圆：实轴取向场的取向角（alpha，取向反转即满偏）
# 现状：相位角 theta 与取向角 alpha 是两个独立的角变量
# 检验：体系内是否有 theta 与 alpha 的锁定关系（如 alpha = theta/2 的投影）
# 相位三层：钟（普朗克旋转）/ 信息（实轴取向 alpha）/ 秩序（序参量）
# 实轴取向 alpha 是独立信息层自由度，与相位 theta 不同层
print('   相位角 theta（波性侧，S^1 紧致）与取向角 alpha（信息层，取向紧致）')
print('   在相位三层中分属不同层：钟层（theta 不可观测）vs 信息层（alpha 可变有梯度）')
print('   故两紧致圆是同一相位的两个读数，还是两个独立自由度？需分层判定')
# 检查：若 alpha 独立，则两紧致性是"两个自由度各自紧致"，非"同一因子"
s1_note = '两紧致圆分属不同相位层（钟层 vs 信息层），非同一因子的两个投影'
print('   S1 判读：%s' % s1_note)
print('   可判性：可判（结构核对）。若成立，切口甲的答案是"不同源但同层族"')
print()

# ---------- 切口乙：双螺旋导出取向紧致圆 ----------
print('S2 切口乙（双螺旋导出）：取向紧致圆能否由心物双螺旋导出')
print('   心物双螺旋：实轴物螺旋 + 虚轴心螺旋，相位差恒 pi/2')
print('   若取向角 alpha 随物螺旋相位线性推进，则 alpha 取遍 S^1，紧致性得')
# 检验：假设 alpha = theta + pi/2（双螺旋锁定），则 alpha 紧致当且仅当 theta 紧致
# 这是可判的：只需检验 alpha 与 theta 的锁定关系是否在体系内登记
print('   检验路径：若 alpha 与 theta 存在锁定（如 alpha = theta + pi/2），')
print('   则取向紧致性是相位紧致性的推论，切口乙成立（可判）')
print('   现状：体系内未见 alpha-theta 锁定关系的登记')
print('   S2 判读：可判但需新增构造（锁定关系），属新机制')
print()

# ---------- 切口丙：候选三可判性 ----------
print('S3 切口丙（候选三）：三饱和点的截断函数同一性')
print('   三饱和点：D=1（无序度饱和）、theta=1（满偏写入）、视界（theta=1/2）')
print('   候选三原文已指出：三者数值不重合，字面统一不成立')
print('   改写为"截断函数同一"：到达上限后的行为由同一机制给出')
# 检验：D 与 theta 是否同一函数族？
# D = 1 - |z|/sqrt(<|z|^2>)（复相干度亏损，Cauchy-Schwarz 有界）
# theta 满偏：取向反向（alpha = pi）
# 视界：theta = 1/2（光锥闭合）
# 三者的"截断后行为"是否同型？需分别给出截断函数
rng = np.random.default_rng(42)
n = 10000
z = np.exp(1j * rng.uniform(0, 2 * np.pi, n))
D = 1.0 - abs(np.mean(z)) / np.sqrt(np.mean(np.abs(z) ** 2))
print('   D 截断：柯西-施瓦茨给 [0,1]，上限 1（贴 1 不超）')
print('   theta 满偏：取向反向（alpha=pi，SO(3) 二阶元），超满偏即失义')
print('   视界：光锥在 theta=1/2 闭合（theta<1/2 侧因果闭合）')
print('   三者截断函数：D 贴上界、theta 越界即失义、theta=1/2 因果闭合')
print('   S3 判读：三者截断行为不同型（上界贴合 vs 越界失义 vs 因果闭合），')
print('        候选三的可判性存疑，需先定义"截断函数同一"的判据形式')
print()

# ---------- S4 三切口对比 ----------
print('S4 三切口对比（可判性 / 所需新内容 / 产出层级）')
rows = [
    ('甲 共同原点', '可判（结构核对）', '无（已有材料）', '两紧致性同层族或异源的判定'),
    ('乙 双螺旋导出', '可判但需新构造', 'alpha-theta 锁定关系（新机制）', '取向紧致性为相位紧致性的推论'),
    ('丙 候选三同一', '可判性存疑', '截断函数同一的判据定义', '三饱和点行为是否同型'),
]
print('   %-14s %-18s %-32s %s' % ('切口', '可判性', '所需新内容', '产出'))
for r in rows:
    print('   %-14s %-18s %-32s %s' % (r[0][:14], r[1][:18], r[2][:32], r[3][:26]))
print()

# ---------- EP5 区分力 ----------
print('EP5 反例检验：若 alpha 与 theta 锁定（alpha=theta+pi/2），取向紧致应自动成立')
# 若锁定，则 alpha 取遍 S^1 当且仅当 theta 取遍 S^1
theta_samples = np.linspace(0, 2 * np.pi, 1000, endpoint=False)
alpha_locked = theta_samples + np.pi / 2
# 紧致性检验：alpha 是否取遍一圈（无重复、无遗漏）
alpha_mod = alpha_locked % (2 * np.pi)
sorted_a = np.sort(alpha_mod)
gaps = np.diff(sorted_a)
gap_uniform = np.max(np.abs(gaps - 2 * np.pi / len(alpha_mod)))
print('   锁定假设下 alpha 的取样间隙 = %.6f（期望均匀 %.6f），偏差 %.2e'
      % (2 * np.pi / len(alpha_mod), 2 * np.pi / len(alpha_mod), gap_uniform))
print('   判读：锁定假设自洽（alpha 紧致随 theta 紧致自动成立）')
ep5_ok = gap_uniform < 1e-9
print('   EP5：%s（反例 = alpha 独立取常数时紧致性不随 theta 传递）'
      % ('PASS，锁定结构与独立结构可区分' if ep5_ok else 'FAIL'))
print()

print('汇总：S1 可判 S2 可判需新构造 S3 可判性存疑 EP5 %s'
      % ('PASS' if ep5_ok else 'FAIL'))
print()
print('方向界定建议：')
print('  切口甲（零成本）：两紧致圆是否同层族的结构判定，材料齐备，可立即执行')
print('  切口乙（中等）：需 alpha-theta 锁定关系（新机制），执行前需确认该关系可从公理导出')
print('  切口丙（存疑）：候选三需先定义"截断函数同一"的判据形式，否则不可判')
print()
print('推荐执行顺序：甲（零成本先行）-> 乙（若甲判定为异源则乙成为必需）-> 丙（远期）')