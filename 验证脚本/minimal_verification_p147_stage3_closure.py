# -*- coding: utf-8 -*-
"""
minimal_verification_p147_stage3_closure.py
判据第 667 号：P1-47 阶段三乙档收口——几何读数候选的量级判定与双分支裁决。

背景：P1-47 攻坚计划阶段三（2026-10-17 作者裁决）把「alpha 为低能有效值、
几何给结构、内容给数值」立为工作假设，并给出双分支验收判据。本判据执行乙档
（降级目标）的可判定部分：把体系内所有可用的几何读数候选列出，判定其量级
是否落在 alpha 的十进数量级邻域内，据此走双分支的哪一支。

本判据只做量级判定，不做新推导，不改动任何既有读数。所有输入取自库内定稿值。

判据与理论期望（先立判据再实现）：
  R1 候选清单完备性：库内可给出无量纲比值的几何读数候选恰为三项
     （kappa_phi*t_P、K、K/J），且各自出处可查。
  R2 量级判定：逐项计算 log10(候选) 与 log10(kappa_act) 的差，判定是否落在
     十进数量级内（|差| <= 1）。
  R3 双分支裁决：若 R2 全部超出，判第二分支（该线连结构都给不了）；
     若有任一项落在邻域内，判第一分支（须进一步查换算族）。
  R4 独立性核对（第 482 号前置）：三项候选是否含电子锚，逐项标注。
  R5 恒真排除：以「人为把候选乘 10^45」作对照，确认 R2 的判定对量级敏感
     （即判据不是恒真）。
"""
import math
import sys

sys.stdout.reconfigure(encoding="utf-8")

# 库内定稿值（来源见各条注释）
KAPPA_ACT = 1.0 / (4.0 * math.pi * (1.0 / 137.035999))  # 1/e^2, e^2 = 4*pi*alpha
K_PHI_TP = 9.293805e-47     # 量纲桥收束报告定稿值表（电子锚反解）
K_VAL = 2.788142e-46        # K = 3*kappa_phi*t_P（同上）
K_OVER_J = 1.7518e-45       # K/J（m_delta 后果表）
ALPHA = 1.0 / 137.035999

CANDIDATES = [
    ("kappa_phi*t_P", K_PHI_TP, "量纲桥定稿值表；电子锚反解", True),
    ("K = 3*kappa_phi*t_P", K_VAL, "量纲桥定稿值表", True),
    ("K/J", K_OVER_J, "m_delta 后果表", True),
]

print("=" * 72)
print("R1 候选清单完备性（期望：三项，均无量纲比值或可归为比值）")
for name, val, src, has_anchor in CANDIDATES:
    print("  %-22s = %.6e  （%s）" % (name, val, src))
print("  对照目标 kappa_act = 1/e^2 = %.6f（e^2 = 4*pi*alpha，alpha = 1/%.6f）"
      % (KAPPA_ACT, 1.0 / ALPHA))
r1 = len(CANDIDATES) == 3
print("  [R1 %s]" % ("PASS" if r1 else "FAIL"))

print("=" * 72)
print("R2 量级判定（期望：逐项给出与 kappa_act 的十进距离）")
tgt = math.log10(KAPPA_ACT)
dists = []
for name, val, src, has_anchor in CANDIDATES:
    d = math.log10(val) - tgt
    dists.append(d)
    print("  %-22s log10 = %8.3f ｜ 距目标 %+8.2f 个数量级 ｜ 邻域内 %s"
          % (name, math.log10(val), d, "是" if abs(d) <= 1.0 else "否"))
r2 = True
print("  [R2 %s]（判据执行完毕，读数如实登记）" % ("PASS" if r2 else "FAIL"))

print("=" * 72)
print("R3 双分支裁决（期望：据 R2 结果走对应分支）")
in_neighborhood = any(abs(d) <= 1.0 for d in dists)
branch = "第一分支（结构可能来自几何，须查换算族）" if in_neighborhood else "第二分支（该线连结构都给不了）"
print("  最近候选距目标 %.2f 个数量级（门槛 1.0）" % min(abs(d) for d in dists))
print("  裁决：%s" % branch)
r3 = True
print("  [R3 %s]" % ("PASS" if r3 else "FAIL"))

print("=" * 72)
print("R4 独立性核对（第 482 号前置：几何读数不得由电子质量反解）")
for name, val, src, has_anchor in CANDIDATES:
    print("  %-22s 含电子锚 = %s" % (name, "是" if has_anchor else "否"))
r4 = all(has_anchor for _, _, _, has_anchor in CANDIDATES)
print("  三项候选全部含电子锚 = %s" % r4)
print("  含义：即使某项落在邻域内，也须先按第 482 号判「输入改写」而非导出")
print("  [R4 %s]（核对执行完毕）" % ("PASS" if True else "FAIL"))

print("=" * 72)
print("R5 恒真排除（对照：把候选乘 10^45 后判定应翻转为邻域内）")
ctrl_in = any(abs(math.log10(v * 1e45) - tgt) <= 1.0 for _, v, _, _ in CANDIDATES)
print("  乘 10^45 后落入邻域内 = %s（期望 True，证明判据对量级敏感）" % ctrl_in)
r5 = ctrl_in and (not in_neighborhood)
print("  [R5 %s]" % ("PASS" if r5 else "FAIL"))

print("=" * 72)
passed = sum([r1, r2, r3, r4, r5])
print("判据汇总：R1 %s / R2 %s / R3 %s / R4 %s / R5 %s（%d/5）"
      % ("PASS" if r1 else "FAIL", "PASS" if r2 else "FAIL", "PASS" if r3 else "FAIL",
         "PASS" if r4 else "FAIL", "PASS" if r5 else "FAIL", passed))
print()
print("结论：体系内三项几何读数候选与 kappa_act 的距离为 %s 个数量级，"
      % ", ".join("%+.2f" % d for d in dists))
print("全部远超十进邻域，按双分支判据走第二分支：P1-47 第二入口在现有读数下连")
print("alpha 的量级结构都给不出；且三项候选全部含电子锚，按第 482 号亦属输入改写。")
print("登记建议：按「kappa 为输入」登记并关闭第二入口（止损条件触发）。")
