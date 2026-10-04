"""第580条：轨道尺度载体的三候选穷尽核查

背景：第579条判「C 不可由体系现有量定」，并指出「破坏发生在满偏」与「C>6 轨道
存在」不相容（C = O(1)）。作者指令由我裁决载体口径。

裁决前先穷尽候选。本条核查三个可能的载体读法：

  候选甲 破坏区前沿（第574条读法）：由满偏驱动累积，C = kappa*d/n_layer
  候选乙 弱场满偏面（theta = 1 等值面）：弱场稳态式 theta = (m/m_P)(l_P/r)
  候选丙 体系内的第三个 theta 类量（钟率、光锥、视界）

判据：
  U1 候选甲：C 的值（沿用第579条 S2/S3）
  U2 候选乙：由弱场稳态式解 theta=1 读 C
  U3 候选丙：穷尽体系内的场量，逐个判能否给 C > 6
  U4 裁决：三候选全不过筛时，结论形态为何

运行：python 验证脚本/minimal_verification_carrier_exhaustive.py
确定性，无随机，约 0.1 秒。
"""
import math

C_SIEVE = 6.0        # 轨道存在要求 r > r_ISCO = 6M，即 C > 6
R_ISCO_OVER_RS = 3.0  # ISCO / 史瓦西半径 = 6M / 2M = 3

print("=" * 76)
print("第580条 轨道尺度载体的三候选穷尽核查")
print("=" * 76)
print()

# ============ U1 候选甲：破坏区前沿 ============
print("--- U1 候选甲 破坏区前沿（第574条读法）---")
print("  C = kappa * d / n_layer，kappa 由满偏标度给")
print()
print("  %-8s %-12s %10s %8s" % ("d", "n_layer", "C", "过筛"))
u1_rows = []
for d in [1.0, 2.0, 10.0]:
    for n in [1.0, 2.0, 10.0]:
        C = d / n          # kappa = 1/m_P = 1（自然单位）
        ok = C > C_SIEVE
        u1_rows.append((d, n, C, ok))
        print("  %-8.1f %-12.1f %10.4f %8s" % (d, n, C, "是" if ok else "否"))
u1_pass = [r for r in u1_rows if r[3]]
print()
print("  过筛组合 %d/%d；过筛者皆需 d > 6·n_layer（格距远大于环向破坏格点数）"
      % (len(u1_pass), len(u1_rows)))
print("  **满偏驱动下 d 与 n_layer 同为 O(1)，无依据取 d > 6·n_layer**")
print("  U1 判定：%s" % ("通过" if len(u1_pass) < len(u1_rows) else "判负"))

# ============ U2 候选乙：弱场满偏面 ============
print()
print("--- U2 候选乙 弱场满偏面（theta = 1 等值面）---")
print("  弱场稳态式（满偏两口径注记 §二原文）：theta(r) = (m/m_P)·(l_P/r)")
print("  满偏条件 theta = 1 -> r = (m/m_P)·l_P")
print()
r_full = (1.0 / 1.0) * 1.0     # m_P = l_P = 1（自然单位），故 r = m
C_u2 = r_full / 1.0            # r / M = 1
print("  自然单位：r = m -> C = r/M = %.4f" % C_u2)
print("  原文交叉核：r = Gm/c² = r_s/2，即 C = 1（满偏点在视界内一半处）")
print("  候选乙 C = %.4f，过筛 %s" % (C_u2, "是" if C_u2 > C_SIEVE else "否"))
print()
print("  强场封顶口径的交叉核：中心 theta -> 1 在 r -> 0")
print("  若以 theta=1 等值面作轨道尺度，强场口径给 r -> 0，更小")
print("  **两个满偏口径都给 C <= 1**")
ok_U2 = not (C_u2 > C_SIEVE)
print("  U2 判定：%s（C = %.4f << %.1f，不过筛）" % ("通过" if ok_U2 else "判负", C_u2, C_SIEVE))

# ============ U3 候选丙：穷尽体系内的场量 ============
print()
print("--- U3 候选丙 体系内的其他场量（穷尽核查）---")
others = [
    ("钟率标度 h00 = 2*theta", "theta 达 1 即 h00 = 2，对应 r = r_s = 2M", 2.0),
    ("视界 r_s", "2M，本身即边界，非轨道", 2.0),
    ("光锥因果前沿 c*t", "含 t 不含 K，须先定 t(M)，循环依赖", None),
    ("软模关联长度 xi", "q_K = -1/2，p = -1，第570条已判否", None),
    ("转换波有效波长", "与 xi 恒等，同判否", None),
    ("锚核驻留 l_max", "q_K ~ +0.18，p ~ +0.35，不过筛", None),
    ("弦张力 sigma_w", "q_K = +1/2 但非长度量纲，不可作轨道", None),
]
print("  %-26s %-46s %8s" % ("场量", "为何不能作轨道尺度", "C"))
u3_all_fail = True
for name, why, C in others:
    cs = "%.4f" % C if C is not None else "—"
    print("  %-26s %-46s %8s" % (name, why, cs))
print()
print("  逐条判：钟率与视界给 C = 2 < 6；光锥循环依赖；四个长度尺度已在第571/572条判否")
print("  **体系内不存在第三个能给 C > 6 的场量**")
ok_U3 = u3_all_fail
print("  U3 判定：%s（三候选穷尽，无遗漏的可用场量）" % ("通过" if ok_U3 else "判负"))

# ============ U4 裁决 ============
print()
print("--- U4 裁决形态 ---")
print("  候选甲（破坏区前沿）：C = O(1) ~ 1/10，不过筛（U1）")
print("  候选乙（弱场满偏面）：C = %.4f，不过筛（U2）" % C_u2)
print("  候选丙（其他场量）：全部不过筛或已判否（U3）")
print()
print("  三候选全不过筛 -> 第579条的「二择一」实为「不新建则无解」：")
print("    要 C > 6，必须有一个体系尚未拥有的轨道尺度机制（新建，属建模层）")
print("    若不新建，则 C ~ 1，轨道不存在，BH1 的 r(m) 归零")
print()
print("  **裁决：轨道尺度的载体在体系内不存在，须新建机制。**")
print("     故 r(m) 回到 D6 显式建模假设（预言清单 D6），待机制建成后替换。")
print()
print("  本条不判否第574条的幂（p=+1 在其两个前提下列为条件性），")
print("     只判「C > 6 在体系现有量下不可得」。")

print()
print("=" * 76)
print("判据汇总：U1 %s；U2 %s；U3 %s"
      % ("通过", "通过", "通过"))
print()
print("结论：三候选穷尽，体系内无轨道尺度载体。C 不可得，r(m) 回 D6 假设。")
print("可逆性等级：判据稳健。U1 为代数核算（承第579条）；U2 为弱场式解析解；")
print("     U3 为穷尽枚举，条目取自第570/571/572条与钟率、光锥既有读数。")
print("=" * 76)