# -*- coding: utf-8 -*-
"""孤子锚口径一致性（A 组）与装订封口（C 组）的最小验证。
判据 A1 至 A4、C1 至 C2 的算术核对与引用清单扫描。
运行：python 验证脚本/minimal_verification_soliton_anchor_consistency.py
"""
import math
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "images", "soliton_anchor_readings.png")

m_P = 2.176434e-8
m_e = 9.1093837015e-31
c = 299792458.0
MeV_per_J = 6.241509074e12
Ee = m_e * c * c * MeV_per_J

kap = 9.293805e-47
K = 3.0 * kap
m_Phi = 2.0 * math.pi
Jb = 1.0 / (2.0 * math.pi)
KJ = K / Jb

def MeV(v):
    return v * m_P * c * c * MeV_per_J

print("== 常数 ==")
print("m_e c^2 = %.8f MeV ; m_e/m_P = %.6e" % (Ee, m_e / m_P))
print()

print("== A1：三条读数（各线现行定义）==")
m_d = math.sqrt(KJ)
gap_mb = 2.0 * math.pi * math.sqrt(K / m_Phi)
gap_d = 2.0 * math.pi * math.sqrt(KJ)
sol = (4.0 / math.pi) * math.sqrt(KJ)
print("字典质量 m_delta = sqrt(K/J) = %.6e (m_P) -> %.5f MeV  [期望 = 电子，M-E 锚]" % (m_d, MeV(m_d)))
print("缺口（MB-9） 2pi*sqrt(K/m_Phi) = %.6e -> %.5f MeV  [期望 = m_delta]" % (gap_mb, MeV(gap_mb)))
print("缺口（D 线） 2pi*sqrt(K/J)     = %.6e -> %.5f MeV  [期望 = 2pi x m_delta]" % (gap_d, MeV(gap_d)))
print("孤子 4/pi*sqrt(K/J)             = %.6e -> %.5f MeV  [期望 0.6506]" % (sol, MeV(sol)))
print()

print("== A2：恒等式自洽核对 ==")
r_closed = (4.0 / math.pi) / (2.0 * math.pi)
r_table = sol / gap_mb
ratio = r_table / r_closed
print("闭式（同一 m_delta，D 线口径）：M_sol/gap_D = 2/pi^2 = %.6f" % r_closed)
print("定稿两支（MB-9 口径）：      M_sol/gap_MB = %.6f" % r_table)
print("两者之比 = %.6f ; 2pi = %.6f ; 相对差 = %.2e" % (ratio, 2 * math.pi, abs(ratio / (2 * math.pi) - 1)))
print("识别：差额恰好等于 m_Phi（=2pi）。两种缺口的差别只在根号内放 m_Phi 还是放 J：")
print("  MB-9: 2pi*sqrt(K/m_Phi) = sqrt(K/J)（因 m_Phi = 1/J）；D 线: 2pi*sqrt(K/J)。")
print("  判读：可按定义吸收（属于两线缺口定义的因子差，不是任一条链内部的算术错误）。")
print()

print("== A3：口径换算表 ==")
print("theta 场 k_b = 1/(2pi) = %.6f ; u 场 m_Phi = 2pi = %.6f ; 比 = (2pi)^2 = %.4f" % (Jb, m_Phi, m_Phi / Jb))
print("sqrt(K/J) = %.6e ; 2pi*sqrt(K/m_Phi) = %.6e ; 相对差 = %.2e" % (m_d, gap_mb, abs(m_d / gap_mb - 1)))
print("换算行：缺口(MB-9) = 缺口(D 线) / m_Phi ；m_delta = 缺口(MB-9) = 缺口(D 线)/(2pi)。")
print()

print("== C1 与 C2：装订量级封口 ==")
need = Ee / MeV(sol)
shift = abs(MeV(sol) - Ee) / MeV(sol)
print("把孤子移到电子所需因子 = %.6f ; pi/4 = %.6f（恰好等于去掉 4/pi 前因子）" % (need, math.pi / 4.0))
print("所需相对移动量 = %.4f（%.2f%%）" % (shift, 100 * shift))
print("单圈标度 kappa_phi t_P = %.6e ；系数取 O(1) 时相对修正约 %.1e" % (kap, kap))
print("缺口倍数 = %.3e （所需移动量 / kappa_phi t_P），动力学装订无法补足。" % (shift / kap))
print()

print("== 图 ==")
fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), dpi=110)
fig.subplots_adjust(left=0.06, right=0.98, top=0.86, bottom=0.12, wspace=0.28)

ax = axes[0]
names = ["缺口（MB-9）\n= 电子（锚，M-E）", "孤子\n4/pi x 电子", "缺口（D 线）\n2pi x 电子"]
vals = [MeV(gap_mb), MeV(sol), MeV(gap_d)]
cols = ["#7a9cc0", "#c0392b", "#7fbf7f"]
bars = ax.bar([0, 1, 2], vals, color=cols, width=0.55)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.06, "%.4f MeV" % v, ha="center", fontsize=10.5)
ax.axhline(Ee, color="#555555", ls="--", lw=1.0)
ax.text(2.42, Ee + 0.04, "电子 0.5110", ha="right", fontsize=10, color="#555555")
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(names, fontsize=10)
ax.set_ylabel("质量（MeV）", fontsize=11)
ax.set_title("(a) 三条读数的数值", fontsize=13, pad=8)
ax.set_ylim(0, 3.9)

ax = axes[1]
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.set_title("(b) 因子链与两线缺口的差别", fontsize=13, pad=8)
ax.text(0.5, 0.86, "字典质量 m_delta = 电子（锚）", ha="center", fontsize=11, color="#1f3f63",
        bbox=dict(boxstyle="round,pad=0.35", fc="#eaf1f8", ec="#7a9cc0"))
ax.annotate("", xy=(0.5, 0.60), xytext=(0.5, 0.79), arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.6))
ax.text(0.53, 0.695, "x 4/pi = 1.2732", fontsize=11, color="#c0392b")
ax.text(0.5, 0.52, "孤子 = 0.6506 MeV", ha="center", fontsize=11, color="#c0392b",
        bbox=dict(boxstyle="round,pad=0.35", fc="#f6d3cf", ec="#c0392b"))
ax.text(0.5, 0.30, "缺口（D 线）= 2pi x 电子 = 3.2107 MeV", ha="center", fontsize=11, color="#3f6b3f",
        bbox=dict(boxstyle="round,pad=0.35", fc="#eaf7ea", ec="#7fbf7f"))
ax.text(0.5, 0.115, "两线缺口定义相差 m_Phi = 2pi：\nMB-9 放 m_Phi 在根号内（2pi sqrt(K/m_Phi)），D 线放 J（2pi sqrt(K/J)）",
        ha="center", fontsize=10.5, color="#333333")
fig.savefig(IMG)
print("saved:", IMG)
print()

print("== A4：引用清单扫描 ==")
pat = re.compile(r"0\.6506|1\.273240|3\.2107|缺口质量")
for base, dirs, files in os.walk(ROOT):
    if ".git" in base or base.endswith("images") or base.endswith("_tools"):
        continue
    for fn in files:
        if not fn.endswith(".md"):
            continue
        p = os.path.join(base, fn)
        try:
            txt = open(p, encoding="utf-8").read()
        except Exception:
            continue
        n = sum(1 for ln in txt.splitlines() if pat.search(ln))
        if n:
            print("  %2d 处  %s" % (n, os.path.relpath(p, ROOT)))
