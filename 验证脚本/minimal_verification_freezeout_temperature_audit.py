# -*- coding: utf-8 -*-
"""B 路冻结温度审查：相变温度已有读数，所需势垒落在电弱档（第 364 条）。
运行：python 验证脚本/minimal_verification_freezeout_temperature_audit.py
"""
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "images", "freezeout_temperature_audit.png")

exp_reg = 106.7
tab = [(2.15, 4.1e-8, 2.7), (2.10, 4.3e-7, 0.846), (2.05, 5.2e-6, 0.243), (2.00, 7.1e-5, 0.066)]
cands = [("v（定居密度）", 246.0), ("m_H", 125.0), ("QCD 禁闭", 0.150), ("m_p", 0.938)]

print("== 体系已有的相变温度（m_delta 线，标准热史换算）==")
for p, t, T in tab:
    print("  p = %.2f : t_* = %.1e s , T_* = %.3g GeV , 所需势垒 = %.1f GeV" % (p, t, T, exp_reg * T))
print()

print("== 候选势垒在各温度下的指数 ==")
print("  %-12s %s" % ("势垒", "  ".join("T=%.3g GeV" % T for _, _, T in tab)))
for name, dE in cands:
    row = "  %-12s" % name
    for _, _, T in tab:
        row += "  %9.1f" % (dE / T)
    print(row)
print()

print("== 焦点：v 势垒 ==")
T_req = 246.0 / exp_reg
print("  取 DeltaE = v = 246 GeV 时，所需 T = %.3f GeV" % T_req)
for p, t, T in tab:
    print("  p = %.2f : 指数 = %.1f（目标 106.7，差 %.1f）" % (p, 246.0 / T, 246.0 / T - exp_reg))
print()

print("== 图 ==")
fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), dpi=110)
fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.14, wspace=0.26)

ax = axes[0]
Ts = [0.05 * (1.12 ** k) for k in range(0, 45)]
for name, dE in cands:
    ys = [dE / T for T in Ts]
    ax.loglog(Ts, ys, lw=1.8, label=name)
ax.axhline(exp_reg, color="#2f7a2f", ls="--", lw=1.2)
ax.annotate("目标指数 106.7", (0.12, exp_reg * 1.15), fontsize=10.5, color="#2f7a2f")
for _, _, T in tab:
    ax.plot([T], [exp_reg], "o", color="#c0392b", ms=5)
ax.annotate("体系相变温度读数\n(66 MeV 至 2.7 GeV)", (0.08, 2.0), fontsize=10, color="#c0392b")
ax.set_xlabel("T_*（GeV）", fontsize=11)
ax.set_ylabel("指数 = 势垒 / T_*", fontsize=11)
ax.set_title("(a) 势垒与温度的组合", fontsize=12.5, pad=8)
ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
ax.legend(fontsize=9.5, loc="upper right")

ax = axes[1]
names = ["p=2.15\n2.7 GeV", "p=2.10\n846 MeV", "p=2.05\n243 MeV", "p=2.00\n66 MeV"]
vals = [exp_reg * T for _, _, T in tab]
ax.bar([0, 1, 2, 3], vals, color="#7a9cc0", width=0.55)
for i, v in enumerate(vals):
    ax.text(i, v * 1.12, "%.0f GeV" % v, ha="center", fontsize=10.5)
ax.axhline(246.0, color="#c0392b", ls="--", lw=1.4)
ax.annotate("v = 246 GeV", (3.2, 262), fontsize=10.5, color="#c0392b", ha="right")
ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(names, fontsize=10)
ax.set_ylabel("B 路所需势垒（GeV）", fontsize=11)
ax.set_yscale("log")
ax.set_title("(b) 所需势垒落在电弱档", fontsize=12.5, pad=8)
fig.savefig(IMG)
print("saved:", IMG)
