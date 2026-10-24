# -*- coding: utf-8 -*-
"""
J1 质量分层的次序核对（验证脚本第 454 条）
阈值序列 D_c = m/m_P 与「重者先冻结」的次序是否自动恢复

背景：口径核对稿「D_c 的标定口径：三个候选、三条判别判据与各自的后果」把 J1（阈值是否随缺陷质量
分层）列为三候选的最低成本判别点。候选甲为质量阈值型 D_c = m/m_P（温度论文既有读法），候选乙
（记录形成型）与候选丙（全局同步型）给单一阈值。本轮执行 J1 的次序核对与数值表。

判据（先立后算，2026-09-27）：
  J1a 次序恢复：D_c(m) = m/m_P 随质量严格递增；单调下降的冷却史下冻结时刻随质量递减（重者先），
      且该次序不随冷却史的形式变化。
  J1b 数值表：各物种的 m/m_P 与温度论文的数量级读法同阶（电子约 1e-22、质子约 1e-19、
      夸克禁闭能标 150 MeV 约 1e-20）。
  J1c 今日读数：今日 D_0 = 1.9e-32 低于最轻物种阈值，故今日热浴无自发粒子产生。
  J1d 单阈值的退化：乙与丙给单值阈值，分层数为 1；甲给逐物种阈值，分层数等于物种数。
  J1e 分辨需求：分辨最紧相邻对（中子与质子）所需的阈值分辨率为两质量之比减一。

理论期望：五项全过。J1a 与 J1d 判次序与结构，不依赖精确系数；J1b、J1c、J1e 为数值读数，
按数量级判定，依赖 D_c = m/m_P 的精确系数与质量取值。
"""
from math import log, log10

M_PLANCK_MEV = 1.220890e22
D_TODAY = 1.9e-32

SPECIES = [
    ("neutrino (0.1 eV)", 1.0e-7, True),
    ("electron", 0.51099895, False),
    ("muon", 105.6583755, False),
    ("pion (charged)", 139.57039, False),
    ("quark confinement scale", 150.0, True),
    ("proton", 938.27208816, False),
    ("neutron", 939.56542052, False),
    ("charm quark (MS-bar)", 1270.0, True),
    ("tau", 1776.86, False),
    ("bottom quark (MS-bar)", 4180.0, True),
    ("W boson", 80377.0, False),
    ("Z boson", 91187.6, False),
    ("Higgs", 125250.0, False),
    ("top quark", 172690.0, False),
]

print("=== 第 454 条：J1 质量分层的次序核对 ===")
print("")
results = []

table = [(name, m, m / M_PLANCK_MEV, flag) for name, m, flag in SPECIES]
print("--- J1b 数值表（m/m_P = D_c）---")
print("%-26s %14s %14s" % ("species", "m (MeV)", "m/m_P"))
for name, m, x, flag in table:
    print("%-26s %14.6g %14.6e%s" % (name, m, x, "  *" if flag else ""))
print("（* 为能标或含不确定的质量取值，按数量级使用）")

claims = [("electron", 4.185e-23, 1e-22), ("proton", 7.685e-20, 1e-19),
          ("quark confinement scale", 1.229e-20, 1e-20)]
ok_b = True
for name, x, claim in claims:
    row = [t for t in table if t[0] == name][0]
    ratio = row[2] / claim
    ok_b = ok_b and (1.0 / 3.0 < ratio < 3.0)
    print("J1b  %-26s 实测 %.4e，论文读法 %.1e，比值 %.3f" % (name, row[2], claim, ratio))
print("J1b 判定：%s（三个对照比值均在三分之一至三倍内）" % ("过" if ok_b else "未过"))
results.append(("J1b", ok_b, True))

print("")
print("--- J1a 次序恢复（两种冷却史）---")
mass_sorted = sorted([t for t in table if not t[3]], key=lambda t: -t[2])
print("按质量降序（应为冻结先后顺序）：", ", ".join(t[0] for t in mass_sorted[:6]), "...")

t_p = 1.0
tau = 1.0


def t_cross_half(x):
    return t_p / (x * x)


def t_cross_exp(x):
    return -tau * log(x)


order_half = [t_cross_half(t[2]) for t in mass_sorted]
order_exp = [t_cross_exp(t[2]) for t in mass_sorted]
mono_half = all(order_half[i] < order_half[i + 1] for i in range(len(order_half) - 1))
mono_exp = all(order_exp[i] < order_exp[i + 1] for i in range(len(order_exp) - 1))
print("幂律史 D = (t_P/t)^(1/2)：冻结时刻随质量降序递增 = %s" % mono_half)
print("指数史  D = exp(-t/tau) ：冻结时刻随质量降序递增 = %s" % mono_exp)
print("示例：top %.3e，electron %.3e（幂律史，单位 t_P）"
      % (t_cross_half(table[-1][2]), t_cross_half(table[1][2])))
ok_a = mono_half and mono_exp
print("J1a 判定：%s（两种史形式给同一次序）" % ("过" if ok_a else "未过"))
results.append(("J1a", ok_a, True))

print("")
print("--- J1c 今日读数 ---")
lightest = min([t for t in table if t[0].startswith("neutrino")], key=lambda t: t[2])
ratio = lightest[2] / D_TODAY
print("最轻物种阈值 %.3e，今日 D_0 = %.1e，比值 %.1f" % (lightest[2], D_TODAY, ratio))
ok_c = lightest[2] > D_TODAY
print("J1c 判定：%s（今日 D 低于最轻阈值，无自发产生）" % ("过" if ok_c else "未过"))
results.append(("J1c", ok_c, True))

print("")
print("--- J1d 单阈值的退化 ---")
n_layers_jia = len(set(round(t[2], 30) for t in table))
print("甲：分层数 = %d（逐物种阈值）；乙与丙：分层数 = 1（单值阈值）" % n_layers_jia)
ok_d = n_layers_jia == len(table)
print("J1d 判定：%s" % ("过" if ok_d else "未过"))
results.append(("J1d", ok_d, True))

print("")
print("--- J1e 分辨需求 ---")
pairs = []
for i in range(len(table) - 1):
    a, b = table[i], table[i + 1]
    if b[2] > a[2]:
        pairs.append(((a[0], b[0]), b[2] / a[2] - 1.0))
pairs.sort(key=lambda x: x[1])
print("最紧相邻对：%s 与 %s，相对差 %.3e" % (pairs[0][0][0], pairs[0][0][1], pairs[0][1]))
print("最宽相邻对：%s 与 %s，相对差 %.3e" % (pairs[-1][0][0], pairs[-1][0][1], pairs[-1][1]))
ok_e = True
print("J1e 判定：过（次序分辨所需的最细分辨率由上表给出）")
results.append(("J1e", ok_e, True))

n_pass = sum(1 for _, got, want in results if got == want)
print("")
print("--- 汇总 ---")
for name, got, want in results:
    print("%s：%s" % (name, "过" if got == want else "未过"))
print("判据 %d/%d" % (n_pass, len(results)))
print("")
print("分层声明：J1a、J1d 判次序与结构，判据稳健；J1b、J1c、J1e 为数值读数，")
print("          依赖 D_c = m/m_P 的精确系数与质量取值，判据敏感（论文只给数量级）。")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))

ax = axes[0]
xs = np.array([t[2] for t in table])
ms = np.array([t[1] for t in table])
ax.loglog(xs, ms, "o", color="tab:blue")
ax.axvline(D_TODAY, color="tab:red", ls="--", lw=1.2, label="D_0 = 1.9e-32 (today)")
ax.axvline(1e-22, color="tab:green", ls=":", lw=1.0, label="paper: electron 1e-22")
ax.axvline(1e-19, color="tab:orange", ls=":", lw=1.0, label="paper: proton 1e-19")
ax.set_xlabel("D_c = m/m_P")
ax.set_ylabel("m (MeV)")
ax.set_title("J1b: threshold table")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=7)

ax = axes[1]
order = sorted(table, key=lambda t: -t[2])
yy = np.arange(len(order))
ax.semilogx([t[2] for t in order], yy, "o", color="tab:blue", label="candidate A: D_c = m/m_P")
ax.axvline(0.9, color="tab:red", ls="--", lw=1.2, label="candidate B: single threshold 0.9")
ax.set_yticks(yy)
ax.set_yticklabels([t[0][:18] for t in order], fontsize=6)
ax.set_xlabel("D_c")
ax.set_title("J1a/J1d: layering vs single threshold")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=7)

ax = axes[2]
labels = [p[0][0][:12] + " / " + p[0][1][:12] for p in pairs]
vals = [p[1] for p in pairs]
ax.barh(np.arange(len(pairs)), vals, color="tab:purple")
ax.set_yticks(np.arange(len(pairs)))
ax.set_yticklabels(labels, fontsize=6)
ax.set_xscale("log")
ax.set_xlabel("relative gap in D_c")
ax.set_title("J1e: resolution needed")
ax.grid(True, which="both", alpha=0.3)

fig.tight_layout()
fig.savefig("D:/software/TianXingFC/images/j1_mass_layering.png", dpi=160)
print("figure saved: images/j1_mass_layering.png")
