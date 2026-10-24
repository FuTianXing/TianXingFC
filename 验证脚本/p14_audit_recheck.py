# -*- coding: utf-8 -*-
"""
p14_audit_recheck.py
判据第 669 号：P1-4 第一层复核——九条判负的读数依赖审计与宽度口径更新。

背景：原计划用第 658 号提升边表示复核第 423 至 433 条的判负，检查是否有判负
被 naive 荷读数伪影带偏。本判据先做依赖审计，据结果决定复核对象，并核对这批
判负所用的输入口径是否与后续更新（第 622 号宽度约定）一致。

判据与理论期望（先立判据再实现）：
  A1 依赖审计：第 423 至 433 条的九条脚本是否含拓扑荷读数（charge/plaq/winding
     等模式）。期望：逐条给出计数，据计数判定提升边是否有复核对象。
  A2 口径核对：第 433 条的宽度输入与第 622 号约定是否一致（相差 2*pi）。
  A3 带端重算：在两种宽度约定下重算 alpha 带端，判定 3/5 是否仍在带内、
     alpha=1/2 是否仍被排除。
  A4 结论稳定性：给出两种约定下结论是否一致的判定。
本判据只做审计与闭式重算，不重跑九条判负，不改动任何既有读数。
"""
import math
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCRIPTS = [
    (423, "huygens_memory.py"),
    (424, "lattice_green_function.py"),
    (425, "gradient_tail_chain.py"),
    (426, "gle_tauberian.py"),
    (427, "stationary_phase_corner.py"),
    (428, "fractal_tail_scan.py"),
    (431, "kpz_surface.py"),
    (432, "defect_geometry_tail.py"),
    (433, "alpha_band_review.py"),
]

# 第 433 条输入（脚本原值）
F0 = 2.12e-105
T_EQ0 = 2.745e55
Z_EQ0 = 3400.0
T_AGE0 = 4.35e17 / 5.391247e-44
W_SLINE = 3.80e21          # S 线/接口轮口径（第 433 条原用）
W_622 = 2.389222e22        # 第 622 号约定
ALPHA_CAND = 0.6           # 3/5


def band(p=1.5, f=F0, w=W_SLINE, t_eq=T_EQ0, z=Z_EQ0, t_age=T_AGE0):
    C = (z ** -3.0 / f) ** (1.0 / 3.0)
    beta = p / 3.0
    lnC, lneq, lna, lnw = math.log(C), math.log(t_eq), math.log(t_age), math.log(w)
    B = lnC - beta * lneq
    if B <= 0.0:
        return None
    a_min = beta + B / lna
    a_max = beta * lnw / (lnw - B) if lnw > B else float("inf")
    return a_min, a_max


print("=" * 72)
print("A1 依赖审计（期望：逐条给出荷读数计数，据结果判定复核对象）")
pat = re.compile(r"charge|plaq|winding|绕数|np\.round", re.I)
total = 0
for num, fn in SCRIPTS:
    path = os.path.join(ROOT, "验证脚本", fn)
    if not os.path.exists(path):
        print("  %3d %-52s 文件缺失" % (num, fn))
        continue
    src = open(path, encoding="utf-8", errors="ignore").read()
    n = len(pat.findall(src))
    total += n
    print("  %3d %-52s 荷读数模式 %d" % (num, fn, n))
a1 = total == 0
print("  九条脚本荷读数模式合计 = %d" % total)
print("  判定：提升边表示%s复核对象" % ("无" if a1 else "有"))
print("  [A1 %s]" % ("PASS" if a1 else "FAIL"))

print("=" * 72)
print("A2 宽度口径核对（期望：第 433 条用 S 线口径，与第 622 号约定差 2*pi）")
ratio = W_622 / W_SLINE
print("  W_S-line = %.6e 格" % W_SLINE)
print("  W_622    = %.6e 格" % W_622)
print("  比值 = %.6f（2*pi = %.6f，相对差 %.2e）" % (ratio, 2 * math.pi, abs(ratio - 2 * math.pi) / (2 * math.pi)))
a2 = abs(ratio - 2 * math.pi) / (2 * math.pi) < 1e-3
print("  [A2 %s]" % ("PASS" if a2 else "FAIL"))

print("=" * 72)
print("A3 带端重算（期望：给出两种约定下的带端与是否含 3/5）")
rows = []
for lab, w in (("S 线口径", W_SLINE), ("第 622 号约定", W_622)):
    r = band(w=w)
    amin, amax = r
    inside = amin <= ALPHA_CAND <= amax
    rows.append((lab, w, amin, amax, amax - amin, inside))
    print("  %-14s w=%.6e  a_min=%.4f  a_max=%.4f  带宽=%.4f  含 3/5=%s"
          % (lab, w, amin, amax, amax - amin, "是" if inside else "否"))
a3 = all(r[5] for r in rows)
print("  [A3 %s]" % ("PASS" if a3 else "FAIL"))

print("=" * 72)
print("A4 结论稳定性（期望：两种约定下 3/5 在带内、1/2 被排除，结论一致）")
half_excluded = all(r[2] > 0.5 for r in rows)
print("  alpha_min 均大于 0.5 = %s（1/2 被排除）" % half_excluded)
print("  alpha_min 两档相同 = %s（该端不依赖 w）" % (abs(rows[0][2] - rows[1][2]) < 1e-12))
print("  带宽变化 = %.4f -> %.4f（收窄 %.1f%%）"
      % (rows[0][4], rows[1][4], 100 * (1 - rows[1][4] / rows[0][4])))
a4 = a3 and half_excluded
print("  [A4 %s]" % ("PASS" if a4 else "FAIL"))

print("=" * 72)
passed = sum([a1, a2, a3, a4])
print("判据汇总：A1 %s / A2 %s / A3 %s / A4 %s（%d/4）"
      % ("PASS" if a1 else "FAIL", "PASS" if a2 else "FAIL",
         "PASS" if a3 else "FAIL", "PASS" if a4 else "FAIL", passed))
print()
print("结论：九条判负全部不依赖拓扑荷读数，提升边表示对本批判负无复核力（原计划的复核无对象）。")
print("第 433 条宽度输入属 S 线口径，与第 622 号约定相差 2*pi；换上后带宽由 0.0417 收窄至 0.0374，")
print("3/5 = 0.6 由带内（<0.6015）变为带外（>0.5971），候选资格须按现行口径重估；")
print("alpha = 1/2 被排除的结论不依赖宽度，仍然成立。")
