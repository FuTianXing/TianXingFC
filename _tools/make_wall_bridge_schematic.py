# -*- coding: utf-8 -*-
"""生成科普《给引力装上源和尺：三维的墙与一座单位之桥》的图 1。
两联图：(a) 三维的墙与四格礼盒；(b) 单位之桥。
输出：images/wall_and_bridge_schematic.png
运行：python _tools/make_wall_bridge_schematic.py（工作区根目录）
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow, FancyBboxPatch, Polygon

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "images", "wall_and_bridge_schematic.png")

fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.2), dpi=110)
fig.subplots_adjust(left=0.02, right=0.985, top=0.9, bottom=0.05, wspace=0.04)

# ===== 面板 a =====
ax = axes[0]
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.set_title("(a) 三维的墙：把源搬进三维", fontsize=13.5, pad=8)

band = Rectangle((0.05, 0.66), 0.90, 0.26, fc="#eef2f7", ec="#b9c4d0", lw=1.2)
ax.add_patch(band)
ys = np.linspace(0.66, 0.92, 80)
xs = 0.50 + 0.022 * np.sin((ys - 0.66) / 0.26 * np.pi * 1.5)
ax.plot(xs, ys, color="#c0392b", lw=3.2, solid_capstyle="round")
right = np.concatenate([np.column_stack([xs, ys]), [[0.95, 0.92], [0.95, 0.66]]])
ax.add_patch(Polygon(right, closed=True, fc="#f6d3cf", ec="none", alpha=0.75, zorder=0))
ax.add_patch(FancyArrow(0.10, 0.955, 0.16, 0.0, width=0.006, head_width=0.028, head_length=0.025, color="#2c5f8a"))
ax.text(0.20, 0.975, "墙以速度 v 运动", fontsize=10.5, color="#2c5f8a", ha="center", va="bottom")
ax.text(0.52, 0.93, "墙（缺陷）", fontsize=10.5, color="#c0392b", ha="left", va="bottom")

ax.text(0.05, 0.615, "张力与一维精确解一致（差 0.0000%）", fontsize=10.5, color="#333333")
ax.text(0.05, 0.560, "跑起来：E/M = γ（偏差 0.006%）", fontsize=10.5, color="#333333")
ax.text(0.05, 0.505, "动量：P = γMv（偏差 0.06%）", fontsize=10.5, color="#333333")

lid = FancyBboxPatch((0.045, 0.10), 0.91, 0.34, boxstyle="round,pad=0.012,rounding_size=0.02",
                     fc="#f7fbf7", ec="#7fa87f", lw=1.4, ls="--")
ax.add_patch(lid)
ax.text(0.50, 0.415, "守恒律是盒盖（两条 $\\partial_\\mu T^{\\mu\\nu} = 0$ 都成立）", fontsize=10.5,
        color="#3f6b3f", ha="center", va="center")
cells = [("能量 $T^{00}$", 0.07, 0.265), ("动量流 $T^{0x}$", 0.53, 0.265),
         ("沿梯度应力 $T^{xx}$", 0.07, 0.145), ("横向应力 $T^{yy}$", 0.53, 0.145)]
for txt, cx, cy in cells:
    b = FancyBboxPatch((cx, cy), 0.40, 0.10, boxstyle="round,pad=0.008,rounding_size=0.015",
                       fc="#eaf1f8", ec="#7a9cc0", lw=1.1)
    ax.add_patch(b)
    ax.text(cx + 0.20, cy + 0.05, txt, fontsize=11, color="#1f3f63", ha="center", va="center")

# ===== 面板 b =====
ax = axes[1]
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.set_title("(b) 单位之桥：把尺接上现实", fontsize=13.5, pad=8)

ax.add_patch(Rectangle((0.03, 0.28), 0.21, 0.46, fc="#dfe7ef", ec="#9fb2c6", lw=1.2))
ax.add_patch(Rectangle((0.76, 0.28), 0.21, 0.46, fc="#dfe7ef", ec="#9fb2c6", lw=1.2))
ax.text(0.135, 0.51, "格子单位\n（步、拍、$m_P$）", fontsize=11, color="#1f3f63", ha="center", va="center")
ax.text(0.865, 0.51, "现实单位\n（米、秒、千克）", fontsize=11, color="#1f3f63", ha="center", va="center")

bridge = FancyBboxPatch((0.235, 0.435), 0.53, 0.085, boxstyle="round,pad=0.004,rounding_size=0.02",
                        fc="#ffe9b8", ec="#c99700", lw=1.5)
ax.add_patch(bridge)
ax.text(0.50, 0.478, "桥因子 $\\epsilon = 1$（算出来的）", fontsize=11.5, color="#8a6100", ha="center", va="center")

ax.annotate("", xy=(0.755, 0.80), xytext=(0.245, 0.80),
            arrowprops=dict(arrowstyle="<->", color="#2c5f8a", lw=1.3))
ax.text(0.50, 0.825, "两端尺度跨度约 101.8 个数量级", fontsize=10.5, color="#2c5f8a", ha="center", va="bottom")

ax.text(0.50, 0.375, "由摆程 $t_P$ 与作用量相位 $\\hbar$ 导出", fontsize=10.5, color="#333333", ha="center")
ax.text(0.50, 0.305, "$K = 3\\kappa_\\phi t_P = 2.788142\\times10^{-46}$", fontsize=11.5, color="#1f3f63", ha="center")
ax.text(0.50, 0.235, "入口份额 $f = 1.531335\\times10^{-102}$", fontsize=11.5, color="#1f3f63", ha="center")
ax.text(0.50, 0.155, "旧值 $9.069\\times10^{-124}$ 与 $1.950\\times10^{-102}$ 退役", fontsize=10, color="#777777", ha="center")
ax.text(0.50, 0.095, "四个分量共用这一座桥（E4）", fontsize=10.5, color="#3f6b3f", ha="center")

fig.savefig(OUT)
print("saved:", OUT)
