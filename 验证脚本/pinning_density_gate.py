# -*- coding: utf-8 -*-
"""钉扎密度的门槛：位错间距窗口与自然候选的距离（第 386 条）。

判据（确定性算术 + 材料清点）：
  G1 门槛反解：捕获率窗口 -> 钉扎密度窗口 -> 位错间距窗口
  G2 自然候选对照：五种畴尺度档对应的捕获率与窗口的距离
  G3 结构性论证：位错作为畴内畸变，间距不超过畴尺度
  G4 综合：门槛的物理读法与本轮判定

运行：python 验证脚本/pinning_density_gate.py
"""
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "images", "pinning_density_gate.png")

PROD = 106.7 / (2.0 * math.pi)   # 16.9818
W_DEF = 3.80e21                  # 缺陷构型宽度，l_P
L_P = 1.616255e-35               # 米
SIGMA = math.pi * W_DEF * W_DEF  # 捕获截面，l_P^2
G_LO = 1.0 / (4.0 * PROD)        # 0.0147216 每拍
G_HI = 1.0 / PROD                # 0.0588866 每拍

def spacing(n):
    return (1.0 / n) ** (1.0 / 3.0)

print("== 设定 ==")
print("  捕获截面 sigma = pi w_def^2 = %.4e l_P^2；迁移速度取光速档 v = 1 格每拍" % SIGMA)
print("  捕获率窗口 Gamma = %.5f 至 %.5f 每拍" % (G_LO, G_HI))
print()

print("== G1 门槛反解 ==")
n_lo = G_LO / SIGMA
n_hi = G_HI / SIGMA
s_hi = spacing(n_lo)
s_lo = spacing(n_hi)
print("  钉扎密度窗口：n_pin = %.4e 至 %.4e l_P^-3" % (n_lo, n_hi))
print("  位错间距窗口：%.4e 至 %.4e l_P（约 %.3e 至 %.3e 米）" % (s_lo, s_hi, s_lo * L_P, s_hi * L_P))
print("  窗口宽度：%.2f 个数量级" % math.log10(s_hi / s_lo))
print()

print("== G2 自然候选对照 ==")
cands = [
    ("KZ 幂律（tau_Q 下端）", 19.47),
    ("追赶变体（同档）", 60.76),
    ("视界间距（同档）", 7.38e3),
    ("P4-27 上界时标档", 5.87e8),
    ("W1 档视界（t_* = 4.09e-8 s）", 7.59e35),
]
for tag, xi in cands:
    g_cand = (xi ** -3.0) * SIGMA * 1.0
    if g_cand > G_HI:
        dec = math.log10(g_cand / G_HI)
        side = "太密（捕获太快）"
    else:
        dec = math.log10(G_LO / g_cand)
        side = "太稀（捕获太慢）"
    print("  %-28s 间距 = %9.3g 格 -> Gamma = %9.3g 每拍；距窗口 %.1f 个数量级（%s）" % (tag, xi, g_cand, dec, side))
print()

print("== G3 结构性论证 ==")
print("  位错是畴内畸变（凝固时序错配的空间尺度不超过畴尺度），故位错间距 <= 畴尺度；")
print("  A 路要求位错间距落在 [%.3g, %.3g] 格，必要条件为畴尺度 >= %.3g 格。" % (s_lo, s_hi, s_lo))
print("  同时位错间距的自然估计由涨落相关长度定（不超过畴尺度），要落入窗口需要该相关长度")
print("  与畴尺度同量级且恰好为 1e15 格量级。")
print()

print("== G4 综合 ==")
d1 = math.log10(s_lo / 19.47)
d2 = math.log10(s_lo / 7.38e3)
d3 = math.log10(s_lo / 5.87e8)
d4 = math.log10(7.59e35 / s_hi)
print("  波前动力学档距窗口 %.1f 至 %.1f 个数量级（太密）；P4-27 上界档差 %.1f 个数量级；" % (d2, d1, d3))
print("  W1 档超窗口 %.1f 个数量级（太稀）。五种自然候选全部远离窗口，最近者差 %.1f 个数量级。" % (d4, d3))
print("  判定：吸附动力学首轮判负（假设：钉扎中心密度可由相变时空结构估计；截面取 384 条口径）。")
print()

print("== 图 ==")
fig, ax = plt.subplots(figsize=(9.8, 4.4), dpi=110)
fig.subplots_adjust(left=0.09, right=0.97, top=0.88, bottom=0.18)

ax.axvspan(s_lo, s_hi, color="#c8e6c9", label="A 路要求的位错间距窗口（0.2 个数量级）")
pts = [
    (19.47, "KZ 幂律 19.5", "#c0392b"),
    (60.76, "追赶 60.8", "#c0392b"),
    (7.38e3, "视界 7.4e3", "#c0392b"),
    (5.87e8, "P4-27 档 5.9e8", "#e08a1e"),
    (7.59e35, "W1 档 7.6e35", "#14508c"),
]
for i, (x, tag, c) in enumerate(pts):
    y = 0.72 - 0.14 * i
    ax.plot([x], [y], "o", color=c)
    ax.annotate(tag, (x * 1.7, y), fontsize=9, color=c, va="center")
ax.set_xscale("log")
ax.set_xlim(1, 1e37)
ax.set_ylim(0, 1)
ax.set_yticks([])
ax.set_xlabel("位错平均间距 / l_P（对数）", fontsize=11)
ax.set_title("钉扎密度门槛：五种自然候选全部远离窗口（最近者差 6.2 个数量级）", fontsize=11.5, pad=8)
ax.grid(True, axis="x", which="both", ls=":", lw=0.5, alpha=0.6)
ax.legend(fontsize=9, loc="upper left")
fig.savefig(IMG)
print("  已保存：", IMG)
