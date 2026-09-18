# -*- coding: utf-8 -*-
"""发布配图：三锚对照静态版（与 可视化/钟与尺_互动示意.html 页尾栏同源）。
① 光偏折（观测区间）；② 真空条件（条件式，83 倍对比底噪）；③ 同源同格林函数（两支读数同在一条曲线上）。
输出：images/anchor_triple_panel.png（绝对路径写入，工作区移动需同步改）。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, axs = plt.subplots(1, 3, figsize=(13.2, 4.2))

ax = axs[0]
ax.axvspan(0.99999, 1.00001, color="#e3ecf9", zorder=0)
ax.axvline(1.0, color="#b3261e", lw=2.0, zorder=3)
ax.text(1.000006, 0.72, "观测带（1e-5 量级）", color="#14508c", fontsize=10, ha="left")
ax.text(1.000012, 0.40, "导出值 1", color="#b3261e", fontsize=11, ha="left")
ax.set_xlim(0.99995, 1.00005)
ax.set_ylim(0, 1)
ax.set_yticks([])
ax.set_title("① 光偏折（观测区间）", fontsize=12, color="#14508c")
ax.set_xlabel("PPN 参数 lambda", fontsize=10)
ax.text(0.02, 0.04, "导出值落在观测带内", transform=ax.transAxes, fontsize=9.5, color="#555555")

ax = axs[1]
ax.bar([0], [83], width=0.5, color="#9aa4ae")
ax.bar([1], [1.4], width=0.5, color="#14508c")
ax.text(0, 86, "λ = 0：83 倍", ha="center", fontsize=10, color="#555555")
ax.text(1, 6, "λ = 1：底噪", ha="center", fontsize=10, color="#14508c")
ax.set_xticks([0, 1])
ax.set_xticklabels(["λ = 0", "λ = 1"], fontsize=10)
ax.set_ylim(0, 100)
ax.set_title("② 真空条件（条件式）", fontsize=12, color="#14508c")
ax.set_ylabel("源外里奇残差倍率", fontsize=10)
ax.text(0.02, 0.04, "λ = 1 落到底噪（前提是场方程成立）", transform=ax.transAxes, fontsize=9.5, color="#555555")

ax = axs[2]
t = np.linspace(0, 1, 300)
ax.plot(t, np.sin(np.pi * t), color="#1f77b4", lw=1.6)
td = np.arange(2, 40, 4) / 40.0
ax.plot(td, np.sin(np.pi * td), "o", ms=5.5, color="#ff7f0e", label="钟的读数")
ax.plot(td, np.sin(np.pi * td), "s", ms=6.0, mfc="none", mec="#2ca02c", label="尺的读数")
ax.legend(fontsize=9.5, frameon=False, loc="upper right")
ax.set_yticks([])
ax.set_ylim(-0.08, 1.28)
ax.set_title("③ 同源同格林函数（结构恒等）", fontsize=12, color="#14508c")
ax.text(0.02, 0.04, "两支读数落在同一条曲线上，同点比值恒为一", transform=ax.transAxes, fontsize=9.5, color="#555555")

fig.text(0.01, 0.015, "注：① 的绝对归一化含一个与 λ 无关的系统因子 1.365（来源未定，登记在案）；② 以真空场方程成立为输入。", fontsize=9, color="#888888")
fig.subplots_adjust(left=0.05, right=0.98, top=0.88, bottom=0.20, wspace=0.28)
plt.savefig("D:/software/TianXingFC/images/anchor_triple_panel.png", dpi=150)
print("saved")
