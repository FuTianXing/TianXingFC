# -*- coding: utf-8 -*-
"""湮灭因子 A 的动力学：钉扎与湮灭的竞赛（第 411 条）。

判据：
  Z1 湮灭率与湮灭时间（缺陷刚形成时）
  Z2 钉扎率与钉扎时间（自然档与 S 线窗口档）
  Z3 钉扎窗口内的湮灭概率与 A 的读数
  Z4 门槛反向与综合（A = 1 升格为结构性结论）

运行：python 验证脚本/annihilation_factor.py
"""
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "images", "annihilation_factor.png")

W_DEF = 3.80e21
SIGMA = math.pi * W_DEF * W_DEF      # 碰撞截面（同款口径），l_P^2
N_DEF = W_DEF ** -3.0                # 刚形成的缺陷密度，l_P^-3
V = 1.0                              # 相对速度，格每拍
T_STAR = 7.562e35                    # 相变时标，拍（第 408 条）

print("== Z1 湮灭率与湮灭时间 ==")
G_ANN = SIGMA * V * N_DEF
T_ANN = 1.0 / G_ANN
print("  缺陷密度（刚形成）：n_def = w_def^-3 = %.4e l_P^-3" % N_DEF)
print("  湮灭率（每缺陷）：Gamma_ann = sigma v n_def = %.4e 每拍" % G_ANN)
print("  湮灭时间：tau_ann = %.4e 拍 = %.3e 秒" % (T_ANN, T_ANN * 5.391247e-44))
print()

print("== Z2 钉扎率与钉扎时间 ==")
for tag, n_pin in (("自然档（每格一位错）", 1.0), ("S 线窗口档（1.298e-45）", 1.298e-45)):
    g_pin = SIGMA * V * n_pin
    t_pin = 1.0 / g_pin
    print("  %-24s Gamma_pin = %.4e 每拍；t_pin = %.4e 拍" % (tag, g_pin, t_pin))
print()

print("== Z3 窗口内湮灭概率与 A ==")
for tag, n_pin in (("自然档", 1.0), ("S 线窗口档", 1.298e-45)):
    g_pin = SIGMA * V * n_pin
    t_pin = 1.0 / g_pin
    p_ann = G_ANN * t_pin
    print("  %-12s 窗口内湮灭概率 = Gamma_ann x t_pin = %.3e -> A = 1 - %.1e" % (tag, p_ann, p_ann))
print()

print("== Z4 门槛反向与综合 ==")
n_gate = N_DEF
print("  湮灭赶上钉扎的门槛：n_pin <= %.3e l_P^-3（同款截面口径）" % n_gate)
print("  对照：自然档 1；S 线窗口档 1.298e-45；该门槛比「每宇宙一位错」（2.6e-148）仍稀 %.0e 倍" % (n_gate / 2.6e-148))
print("  时间线：t_pin（1 拍量级）<< tau_ann（1.2e21 拍）<< t_*（7.6e35 拍）")
print("  综合：A = 1 由「钉扎远快于湮灭」保障（gap 达 20 至 65 个数量级，随钉扎档），从边界假设升格为结构性结论")
print()

print("== 图 ==")
fig, ax = plt.subplots(figsize=(9.8, 4.2), dpi=110)
fig.subplots_adjust(left=0.08, right=0.97, top=0.86, bottom=0.17)

pts = [
    ("t_pin 自然档 2.2e-44 拍", 2.204e-44, "#14508c"),
    ("t_pin S 线档 17.0 拍", 16.98, "#14508c"),
    ("tau_ann 1.2e21 拍", T_ANN, "#c0392b"),
    ("t_* 7.6e35 拍", T_STAR, "#666666"),
]
for i, (tag, v, c) in enumerate(pts):
    y = 0.8 - 0.15 * i
    ax.plot([v], [y], "o", color=c, markersize=9)
    ax.annotate(tag, (v * 2.0, y), fontsize=9.5, color=c, va="center")
ax.axvspan(2.204e-44, 16.98, color="#dce9f7", alpha=0.8, label="钉扎窗口（两档之间）")
ax.set_xscale("log")
ax.set_xlim(1e-45, 1e38)
ax.set_ylim(0, 1)
ax.set_yticks([])
ax.set_xlabel("时间（拍，对数）", fontsize=11)
ax.set_title("钉扎与湮灭的竞赛：钉扎（<= 17 拍）远快于湮灭（1.2e21 拍）", fontsize=11.5, pad=8)
ax.grid(True, axis="x", which="both", ls=":", lw=0.5, alpha=0.6)
ax.legend(fontsize=9, loc="upper left")
fig.savefig(IMG)
print("  已保存：", IMG)
