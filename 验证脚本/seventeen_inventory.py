# -*- coding: utf-8 -*-
"""17 的出现清单：派生链识别与 34 pi 的精度判定（第 395 条）。

判据：
  N1 清单：S 线全部 17/16.98 出现的位置与其来源链
  N2 精度：106.7 的三口径与 34 pi 的对照（派生 vs 近邻）
  N3 判定：目标数是否携带独立结构；三路归约的最终落点

运行：python 验证脚本/seventeen_inventory.py
"""
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "images", "seventeen_inventory.png")

KAPPA_TP = 9.293805e-47
S_REG = 4.437e-47        # 裁决二十八登记值
K0_TP = 2.030            # 第 359 条反解裸耦合
TWO_PI = 2.0 * math.pi

print("== N1 清单：17/16.98 的出现与来源链 ==")
items = [
    ("计数式常数 16.98", "A 路（第 361 条）", "= 106.7/(2 pi)"),
    ("A 路宽度窗口下端 17.0 步", "第 362、376 条", "= 计数式常数（Delta = E_P）"),
    ("t_sweep 自然点 16.98 t_P", "第 361 条", "同上"),
    ("弛豫率目标 0.0589 每拍", "第 376 条", "= 1/16.98"),
    ("势曲率缺口 16.98 E_P", "第 380 条", "= 计数式常数"),
    ("C 路 1/g^2 = 16.98", "第 393 条", "= 106.7/(2 pi)"),
    ("34 pi = 106.814 观察", "第 361 条 §4", "= 2 pi x 17（提示登记）"),
]
for i, (tag, src, chain) in enumerate(items, 1):
    print("  %d. %-24s 来源：%s；链：%s" % (i, tag, src, chain))
print()

print("== N2 精度：106.7 的三口径与 34 pi ==")
e_a = -math.log(S_REG)              # 从登记 S 反算
e_b = math.log(K0_TP / KAPPA_TP)    # 从 kappa_phi 与裸耦合算
e_c = 106.7                          # 登记报值
e_34 = 34.0 * math.pi
print("  口径甲（S = %.4e 反算）：指数 = %.4f" % (S_REG, e_a))
print("  口径乙（kappa_phi 与裸耦合 %.3f）：指数 = %.4f" % (K0_TP, e_b))
print("  口径丙（登记报值）：指数 = %.4f" % e_c)
print("  34 pi = %.4f" % e_34)
for tag, e in (("甲", e_a), ("乙", e_b), ("丙", e_c)):
    d = e_34 - e
    print("  34 pi 与口径%s之差：%.4f（%.4f%%）" % (tag, d, d / e_34 * 100))
print()
print("  一致性核对：kappa_phi/裸耦合给的 S = %.4e，与登记 S = %.4e 差 %.2f%%" % (
    KAPPA_TP / K0_TP, S_REG, (KAPPA_TP / K0_TP / S_REG - 1) * 100))
print("  （口径甲与乙的 3% 级不一致属清点发现，登记为口径待核项。）")
print()

print("== N3 判定 ==")
n_c = e_c / TWO_PI
n_a = e_a / TWO_PI
d17 = 17.0 - n_c
print("  16.98 的精确值（口径丙）：%.4f；与整数 17 之差：%.4f（%.4f%%）" % (n_c, d17, d17 / 17 * 100))
print("  判读一：清单前 6 处的 16.98 全部是 106.7/(2 pi) 的派生（同一除法操作），无独立信息。")
print("  判读二：34 pi 与 106.7 的差为 0.08% 至 0.13%（随口径），在四位精度下可辨；判为近邻而非等式。")
print("  判读三：无论 34 pi 成立与否，「17 的机制」在体系里没有候选来源（数字近邻不构成机制）。")
print("  落点：三路（A、B、C）的目标数不携带独立结构；机制问题的内容回到「106.7 本身」，")
print("    即 kappa_phi 的层级（= 电子质量锚的下游）。")
print()

print("== 图 ==")
fig, ax = plt.subplots(figsize=(9.4, 4.2), dpi=110)
fig.subplots_adjust(left=0.08, right=0.97, top=0.86, bottom=0.16)

ax.axvline(e_34, color="#c0392b", lw=1.6, label="34 pi = %.3f" % e_34)
for e, tag, c in ((e_a, "口径甲 %.3f" % e_a, "#14508c"), (e_b, "口径乙 %.3f" % e_b, "#2f7a2f"), (e_c, "口径丙 %.3f" % e_c, "#7030a0")):
    ax.axvline(e, color=c, ls="--", lw=1.2)
    ax.annotate(tag, (e, 0.75), rotation=90, fontsize=9, color=c, va="center", ha="right")
ax.axvspan(e_b, e_a, color="#dce9f7", alpha=0.7, label="106.7 的口径带（甲乙丙）")
ax.set_xlim(106.60, 106.90)
ax.set_ylim(0, 1)
ax.set_yticks([])
ax.set_xlabel("指数值", fontsize=11)
ax.set_title("34 pi 与 106.7 三口径的对照（差 0.08% 至 0.13%）", fontsize=12, pad=8)
ax.legend(fontsize=9, loc="upper left")
fig.savefig(IMG)
print("  已保存：", IMG)
