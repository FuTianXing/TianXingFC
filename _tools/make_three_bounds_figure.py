# -*- coding: utf-8 -*-
"""科普配图：三个天花板示意图（三条边界）。

输出：images/three_bounds_schematic.png
用途：科普文章《为什么停在 2.828：贝尔违反的第三个天花板》第六节插图。
确定性脚本，无随机；柱高即三类资源的关联强度上界（S = 4 eps）。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = r"D:\software\TianXingFC\images\three_bounds_schematic.png"

labels = ["共享抽签\n（经典）", "共享量子态\n（体系里的锁定对）", "看人下菜的盒子\n（假想）"]
vals = [2.0, 2.0 * np.sqrt(2.0), 4.0]
colors = ["#718096", "#2b6cb0", "#c53030"]
notes = ["开局前共享一串随机数\n怎么摆布都上不去",
         "开局前共享一个量子态\n相位可以斜着用",
         "输出分布随输入变化\n自然界没有兑现它"]

fig, ax = plt.subplots(figsize=(10.2, 6.0))
bars = ax.bar(labels, vals, color=colors, width=0.52, alpha=0.92, zorder=3)
for b, v, n in zip(bars, vals, notes):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.06, "S = %.3f" % v, ha="center", fontsize=12, zorder=4)
    ax.text(b.get_x() + b.get_width() / 2, v / 2, n, ha="center", va="center", fontsize=9.4,
            color="white", zorder=4, linespacing=1.5)

ax.axhline(2.0, color="#718096", lw=1.0, ls=":", zorder=2)
ax.axhline(2.0 * np.sqrt(2.0), color="#2b6cb0", lw=1.0, ls=":", zorder=2)
ax.axhline(4.0, color="#c53030", lw=1.0, ls=":", zorder=2)
ax.axhline(2.827, color="#2f855a", lw=1.4, ls="--", zorder=5)
ax.text(2.42, 2.827, "实验 2.827", color="#2f855a", fontsize=10, va="center", ha="left")

ax.plot([1], [2.0 * np.sqrt(2.0)], marker="*", ms=17, color="#f6ad55", mec="#744210", zorder=6)
ax.annotate("自然界停在这一档", xy=(1, 2.0 * np.sqrt(2.0)), xytext=(1.34, 3.45),
            fontsize=11, color="#744210",
            arrowprops=dict(arrowstyle="->", color="#744210", lw=1.3), zorder=6)

ax.set_ylim(0, 4.55)
ax.set_ylabel("关联强度 S", fontsize=11)
ax.set_title("三个天花板：为什么世界停在 2.828", fontsize=14)
ax.grid(axis="y", alpha=0.22, zorder=1)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print("[INFO] 柱高：", ["%.6f" % v for v in vals])
print("[INFO] 图已存：", OUT)
