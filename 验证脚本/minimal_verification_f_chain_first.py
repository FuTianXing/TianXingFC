# -*- coding: utf-8 -*-
"""f 线首轮：Kibble 链条四环清点与 f 的口径表（第 398 条）。

判据：
  F1 链条四环清点（产量、稀释、湮灭、对接）
  F2 边界解复核（f_prod、D、乘积、t_* 与可见物质档的吻合度）
  F3 口径表：f 的五档数值与质量/计数两口径的关系核对
  F4 综合：链条闭环度、角点条件与判句

运行：python 验证脚本/minimal_verification_f_chain_first.py
"""
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "images", "f_chain_first.png")

W_DEF = 3.80e21          # 缺陷宽度，l_P
T_EQ = 1.48e12           # 秒
Z_EQ = 3400.0
F_VIS = 2.12e-105        # 可见物质档（R1 裁决）
F_NUC = 1.06e-105
F_TOT = 6.8e-105
F_MASS = 1.531335e-102   # 质量口径（孤子，MB-8 定稿）
F_ELEC = 1.949756e-102   # 质量口径旧值（电子）
M_SOL = 0.65063          # MeV（孤子）
M_VIS = 0.5 * 938.272    # MeV（可见物质平均，R1）

print("== F1 链条四环清点 ==")
print("  环一（产量）：缺陷占据率 f_prod = (l_P/xi)^3；边界解 xi = w_def = 3.80e21 l_P")
print("  环二（稀释）：D = (t_*/t_eq)^{3/2} / z_eq^3（分段膨胀标度）")
print("  环三（湮灭）：A = 1（边界读法；从必需降为可选）")
print("  环四（对接）：f_obs 的质量口径或计数口径（见 F3）")
print()

print("== F2 边界解复核 ==")
f_prod = (1.0 / W_DEF) ** 3
D = 1.17e-40
prod = f_prod * D
print("  f_prod = %.4e；D = %.2e；乘积 = %.4e" % (f_prod, D, prod))
print("  与可见物质档（%.2e）之比 = %.4f（%.2f%% 内吻合）" % (F_VIS, prod / F_VIS, abs(prod / F_VIS - 1) * 100))
t_re = T_EQ * (D * Z_EQ ** 3) ** (2.0 / 3.0)
print("  t_* 反解复核：%.3e 秒（第 364 条区间的下界 4.09e-8 秒）" % t_re)
print()

print("== F3 f 的口径表与两口径关系 ==")
rows = [
    ("质量口径（孤子，定稿）", F_MASS, "rho_b/(M_sol/l_P^3)"),
    ("质量口径（电子，旧值）", F_ELEC, "M = m_e"),
    ("计数口径（可见物质，R1）", F_VIS, "rho_b/(M_vis/l_P^3)"),
    ("计数口径（核子）", F_NUC, "M = m_p"),
    ("计数口径（总物质）", F_TOT, "含暗物质"),
]
for tag, v, norm in rows:
    print("  %-24s f = %.4e（%s）" % (tag, v, norm))
r1 = F_MASS / F_VIS
r2 = M_VIS / M_SOL
print("  关系核对：质量口径/计数口径 = %.2f；M_vis/M_sol = %.2f（差 %.2f%%）" % (r1, r2, abs(r1 / r2 - 1) * 100))
print("  判读：两口径为同一物理量的不同归一，因子为每缺陷对应质量与孤子质量之比；引用处宜区分命名")
print()

print("== F4 综合 ==")
print("  链条闭环度：在边界解（xi = w_def、A = 1）上，产量乘稀释与可见物质档吻合到 %.2f%%；" % (abs(prod / F_VIS - 1) * 100))
print("  该吻合依赖三个角点条件（xi 取健全性下界、A 取 1、t_* 取下界）；放开任一条件，产量上界下降。")
print("  判句：其一，xi = w_def 的独立依据（为何冻结关联恰好取缺陷宽度）；")
print("  其二，湮灭因子 A 的动力学（当前的边界读法）；其三，两口径的命名统一。")
print()

print("== 图 ==")
fig, ax = plt.subplots(figsize=(9.8, 4.4), dpi=110)
fig.subplots_adjust(left=0.08, right=0.97, top=0.87, bottom=0.17)

pts = [
    ("可见物质档 2.12e-105", F_VIS, "#2f7a2f"),
    ("核子档 1.06e-105", F_NUC, "#2f7a2f"),
    ("总物质档 6.8e-105", F_TOT, "#2f7a2f"),
    ("链条乘积 2.13e-105", prod, "#c0392b"),
    ("孤子档 1.53e-102", F_MASS, "#14508c"),
    ("电子档 1.95e-102", F_ELEC, "#14508c"),
]
for i, (tag, v, c) in enumerate(pts):
    y = 0.85 - 0.14 * i
    ax.plot([v], [y], "o", color=c, markersize=9)
    ax.annotate(tag, (v * 1.5, y), fontsize=9.5, color=c, va="center")
ax.annotate("", xy=(F_MASS, 0.25), xytext=(F_VIS, 0.25), arrowprops=dict(arrowstyle="->", color="#666666", lw=1.4))
ax.text(math.sqrt(F_VIS * F_MASS), 0.28, "两口径之比 722", fontsize=9.5, color="#666666", ha="center")
ax.set_xscale("log")
ax.set_xlim(5e-106, 5e-102)
ax.set_ylim(0, 1)
ax.set_yticks([])
ax.set_xlabel("入口份额 f（对数）", fontsize=11)
ax.set_title("f 的五档数值、链条乘积与两口径之比（722 倍）", fontsize=11.5, pad=8)
ax.grid(True, axis="x", which="both", ls=":", lw=0.5, alpha=0.6)
fig.savefig(IMG)
print("  已保存：", IMG)
