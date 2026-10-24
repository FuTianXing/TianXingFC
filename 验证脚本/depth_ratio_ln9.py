# -*- coding: utf-8 -*-
"""指数深度比 234.44 / 106.70 与 ln 9 的登记观察（第 366 条）。
运行：python 验证脚本/depth_ratio_ln9.py
"""
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "images", "depth_ratio_ln9.png")

rho_b = 4.2075e-28
m_p = 1.67262192e-27
l_P = 1.616255e-35
m_P = 2.176434e-8
m_e = 9.1093837015e-31
M_MeV = 0.6506
m_p_MeV = 938.272
kap_tP = 9.293805e-47
kap0_tP = 2.030
f = 1.531335e-102

ln_f = math.log(1.0 / f)
ln_S = math.log(kap0_tP / kap_tP)
print("== 两个深度 ==")
print("  ln(1/f) = %.4f ; ln(1/S) = %.4f ; 比 = %.5f" % (ln_f, ln_S, ln_f / ln_S))
print("  ln 9 = %.5f ; 相对差 %.1e" % (math.log(9.0), abs(ln_f / ln_S / math.log(9.0) - 1)))
print()

print("== 分解 ==")
n_b = rho_b / m_p
term1 = math.log(1.0 / (n_b * l_P ** 3))
term2 = math.log(m_p_MeV / M_MeV)
print("  ln(1/f) = ln[1/(n_b l_P^3)] - ln(m_p/M) = %.3f - %.3f = %.3f" % (term1, term2, term1 - term2))
t1 = 2.0 * math.log(m_P / m_e)
t2 = math.log(6.0 * math.pi)
t3 = math.log(kap0_tP)
print("  ln(1/S) = 2 ln(m_P/m_e) + ln(6 pi) + ln(kappa0 t_P) = %.3f + %.3f + %.3f = %.3f" % (t1, t2, t3, t1 + t2 + t3))
print()

print("== 精度括号：指数按登记值 106.7 的四位有效数字取值 ==")
for e in (106.65, 106.70, 106.75):
    print("  指数 = %.2f -> 比 = %.5f" % (e, ln_f / e))
print("  ln 9 = %.5f 落在区间内" % math.log(9.0))
print()


print("== 口径敏感性 ==")
depth_num = math.log(1.0 / (n_b * l_P ** 3))
n_e = 0.875 * n_b
depth_e = math.log(1.0 / (n_e * l_P ** 3))
depth_quark = depth_num - math.log(3.0)
depth_mass = ln_f
for name, d in (("混合口径（现行）", depth_mass), ("重子数口径", depth_num), ("电子数口径", depth_e), ("夸克口径", depth_quark)):
    print("  %-14s 深度 = %8.3f ; 比值 = %.5f" % (name, d, d / ln_S))
print("  ln 9 = %.5f ; 混合口径精度括号 = [%.4f, %.4f]" % (math.log(9.0), ln_f / 106.75, ln_f / 106.65))
print("  电子数密度 n_e = %.5f m^-3（= 0.875 n_b）" % n_e)
print("  混合口径隐含计数：每重子 %.0f 个缺陷（= m_p/M）" % (m_p_MeV / M_MeV))
print()
print("== 其他深度对（纪律核对）==")
for a, b, na, nb in ((139.15, 106.70, "139.15", "106.70"), (234.4375, 139.15, "234.44", "139.15")):
    print("  %s / %s = %.4f" % (na, nb, a / b))
print()

print("== 图 ==")
fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6), dpi=110)
fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.14, wspace=0.28)

ax = axes[0]
names = ["kappa_phi 深度\n106.70", "落位概率深度\n139.15", "入口份额深度\n234.44"]
vals = [106.70, 139.15, 234.44]
cols = ["#7a9cc0", "#c99700", "#c0392b"]
ax.bar([0, 1, 2], vals, color=cols, width=0.55)
for i, v in enumerate(vals):
    ax.text(i, v + 5, "%.2f" % v, ha="center", fontsize=11)
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(names, fontsize=10)
ax.set_ylabel("指数深度", fontsize=11)
ax.set_ylim(0, 280)
ax.set_title("(a) 三个已登记的指数深度", fontsize=12.5, pad=8)

ax = axes[1]
lo, hi = ln_f / 106.75, ln_f / 106.65
ax.axhspan(lo, hi, color="#eaf1f8", label="质量口径精度括号")
ax.axhline(math.log(9.0), color="#2f7a2f", ls="--", lw=1.4)
labels4 = ["混合口径\n2.19716", "重子数\n2.26533", "电子数\n2.26666", "夸克\n2.25503"]
vals4 = [depth_mass / ln_S, depth_num / ln_S, depth_e / ln_S, depth_quark / ln_S]
ax.bar([0, 1, 2, 3], vals4, color=["#c0392b", "#aab7c4", "#aab7c4", "#aab7c4"], width=0.5)
for i, v in enumerate(vals4):
    ax.text(i, v + 0.0016, "%.5f" % v, ha="center", fontsize=9.5)
ax.annotate("ln 9 = 2.19722", (3.35, 2.19760), fontsize=10.5, color="#2f7a2f", ha="right")
ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(labels4, fontsize=9)
ax.set_ylim(2.19, 2.28)
ax.set_ylabel("比值", fontsize=11)
ax.set_title("(b) 三种口径的比值与 ln 9", fontsize=12.5, pad=8)
ax.legend(fontsize=9, loc="lower right")
fig.savefig(IMG)
print("saved:", IMG)
