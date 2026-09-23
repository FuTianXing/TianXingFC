# -*- coding: utf-8 -*-
"""凝固时间的最小动力学：三候选时标估算与 A 路的单参数压缩（第 384 条）。

判据（确定性算术 + 材料清点）：
  S1 背景对照：感知-校正循环的延迟（1 至 4 拍）
  S2 缺陷形成：相位错配的弛豫（局域 1/kappa；大尺度表面活口的自洽性排除）
  S3 缺陷完成：钉扎捕获率的反解（A 路压缩为单参数）
  S4 综合：S1、S2 判负；A 路存活系于钉扎捕获率一参数

运行：python 验证脚本/minimal_verification_solidification_time.py
"""
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "images", "solidification_time.png")

PROD = 106.7 / (2.0 * math.pi)   # 16.9818
W_DEF = 3.80e21                  # 缺陷构型宽度，l_P
T_GATE1 = PROD                   # 17.0 拍
T_GATE2 = 4.0 * PROD             # 67.9 拍

def d_req(t):
    return math.sqrt(PROD / t)

sigma = math.pi * W_DEF * W_DEF
g_hi = 1.0 / T_GATE1             # 0.0589 每拍
g_lo = 1.0 / T_GATE2             # 0.0147 每拍

print("== 设定 ==")
print("  A 路门槛：t_sweep >= %.1f 拍（Delta = E_P）或 >= %.1f 拍（Delta = E_P/2）" % (T_GATE1, T_GATE2))
print("  等价的捕获率窗口：Gamma = 1/t_sweep = %.4f 至 %.4f 每拍" % (g_lo, g_hi))
print("  校准率结构定标：kappa = 1/t_P（正稿第五节）")
print()

print("== S1 背景对照：感知-校正循环的延迟 ==")
print("  链条：局部偏转 -> 邻居感知 -> 自主校正 -> 逐格传播；各环节下限 1 拍")
for t in (1, 2, 4):
    print("  t = %d 拍 -> Delta = %.4f E_P，超截断 %.1f%%" % (t, d_req(t), (d_req(t) - 1) * 100))
print("  判读：背景凝固为快过程（1 至 4 拍），与 A 路门槛差 4 至 68 倍")
print()

print("== S2 缺陷形成：相位错配的弛豫 ==")
print("  局域版：t = 1/kappa = 1 拍 -> Delta = %.4f E_P（超截断 %.1f%%）" % (d_req(1), (d_req(1) - 1) * 100))
print("  大尺度表面活口：t = w^2/kappa，取 w = 2 pi 给 t = %.1f 拍（看似过门槛）" % ((2 * math.pi) ** 2))
print("  自洽性排除：冻结尺度由 tau(xi) = xi/v 定，tau = xi^2/kappa、v = 1 -> xi = kappa = 1；")
print("    与 w = 2 pi 不自洽（稳态解只有 w = kappa），表面活口不成立")
print()

print("== S3 缺陷完成：钉扎捕获率的反解 ==")
print("  钉扎捕获（同生论文：绕数缺陷被位错钉扎而凝固）：t = 1/(n_pin sigma v_def)")
print("  捕获截面 sigma = pi w_def^2 = %.4e l_P^2" % sigma)
print("  A 路要求 Gamma = n_pin sigma v_def = %.4f 至 %.4f 每拍" % (g_lo, g_hi))
nv_lo = g_lo / sigma
nv_hi = g_hi / sigma
print("  即 n_pin v_def = %.3e 至 %.3e l_P^-2 每拍" % (nv_lo, nv_hi))
print("  若取 v_def = 1 格每拍（光速档）：n_pin = %.3e 至 %.3e 每格" % (nv_lo, nv_hi))
sp1 = (1.0 / nv_hi) ** (1.0 / 3.0)
sp2 = (1.0 / nv_lo) ** (1.0 / 3.0)
print("  对应平均位错间距：%.3e 至 %.3e l_P（约 %.2e 至 %.2e 米）" % (sp1, sp2, sp1 * 1.616255e-35, sp2 * 1.616255e-35))
print("  对照：自然档（每格一位错）捕获瞬时；该窗口要求位错密度比自然档稀约 45 个数量级")
print("  判读：钉扎捕获率是 A 路唯一未排除的量，其两个参数（位错密度、迁移速度）未建模")
print()

print("== S4 综合：A 路的单参数压缩 ==")
print("  S1、S2 判负（快过程）；S3 开口（参数未建模）")
print("  A 路的全部内容压缩为：钉扎捕获率 Gamma = 0.0147 至 0.0589 每拍")
print("  该窗口与第 380 条的弛豫率目标同值（同一自洽结构），与入口份额 f 的缺陷产额同族")
print()

print("== 图 ==")
fig, ax = plt.subplots(figsize=(9.8, 4.2), dpi=110)
fig.subplots_adjust(left=0.27, right=0.97, top=0.88, bottom=0.17)

rows = [
    ("S2 错配弛豫（局域）", 1.0, 1.0, "#c0392b"),
    ("S1 感知-校正（1 至 4 拍）", 1.0, 4.0, "#c0392b"),
    ("S3 钉扎捕获（目标窗口）", T_GATE1, T_GATE2, "#2f7a2f"),
]
ys = list(range(len(rows)))[::-1]
for (tag, t1, t2, c), y in zip(rows, ys):
    ax.barh(y, t2 - t1 + 0.0001, left=t1, height=0.5, color=c, alpha=0.75)
    ax.text(t2 * 1.6, y, ("%.3g 至 %.3g 拍" % (t1, t2)) if t1 != t2 else "%.3g 拍" % t1, va="center", fontsize=9.5, color=c)
ax.axvline(T_GATE1, color="#e08a1e", ls="--", lw=1.2)
ax.axvline(T_GATE2, color="#e08a1e", ls=":", lw=1.2)
ax.set_yticks(ys)
ax.set_yticklabels([r[0] for r in rows], fontsize=10)
ax.set_xscale("log")
ax.set_xlim(1e-1, 1e3)
ax.set_xlabel("凝固时间 / 拍（对数）", fontsize=11)
ax.set_title("凝固时间三候选：仅钉扎捕获的目标窗口覆盖 A 路门槛（橙线）", fontsize=11.5, pad=8)
ax.grid(True, axis="x", which="both", ls=":", lw=0.5, alpha=0.6)
fig.savefig(IMG)
print("  已保存：", IMG)
