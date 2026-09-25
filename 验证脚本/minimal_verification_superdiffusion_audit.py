# -*- coding: utf-8 -*-
"""超扩散候选清点：D_req 与体系输运系数的对照（第 410 条）。

判据：
  Y1 候选清单：体系内全部扩散型系数的数值档
  Y2 对照：D_req 与各档的距离
  Y3 弹道参数化：D = cL 所需的 L 与鉴认
  Y4 综合：空档判定与跨线观察（超校准输运的缺席）

运行：python 验证脚本/minimal_verification_superdiffusion_audit.py
"""
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "images", "superdiffusion_audit.png")

D_REQ = 1.909e7          # 格^2 每拍（第 408 条）
W_DEF = 3.80e21          # 格
K_B = 2.0 * math.pi
K_CONST = 2.788142e-46
M_SOL = 4.0 * math.sqrt(K_B * K_CONST) / math.pi   # 孤子质量（格子单位）
D_Q = 1.0 / M_SOL        # hbar/M 档（格子单位 hbar = 1）
T_P = 5.391247e-44
L_P = 1.616255e-35

print("== Y1 候选清单（体系扩散型系数，格^2 每拍） ==")
cands = [
    ("记录阻尼观测上界（5.6e-60 每拍，作扩散档下界）", 5.6e-60),
    ("缺陷迁移率（gamma_d = 0.5958）", 0.5958),
    ("弹道单步档（c x l_P = 1）", 1.0),
    ("校准扩散上界（2 pi）", 2.0 * math.pi),
    ("弹道 x w_def 档（c x w_def）", W_DEF),
    ("量子扩散档（hbar / M_sol = %.3e）" % D_Q, D_Q),
]
for tag, v in cands:
    print("  %-40s %.4e" % (tag, v))
print("  目标 D_req = %.4e（第 408 条）" % D_REQ)
print()

print("== Y2 对照（距离，个数量级） ==")
for tag, v in cands:
    d = abs(math.log10(D_REQ / v))
    print("  %-40s 距目标 %.2f" % (tag, d))
print()

print("== Y3 弹道参数化 ==")
L_req = D_REQ / 1.0
print("  若 D = c L（弹道等效），所需 L = %.4e 格 = %.3e 米" % (L_req, L_req * L_P))
print("  该长度与体系量的对照：w_def 的 1/%.1e；无独立鉴认" % (W_DEF / L_req))
print()

print("== Y4 综合 ==")
d1 = math.log10(D_REQ / (2.0 * math.pi))
d2 = math.log10(W_DEF / D_REQ)
print("  空档判定：目标落在经典输运档（<= 10）与弹道/量子档（>= 1e21）之间；")
print("    距最近的经典档 %.1f 个数量级，距最近的弹道档 %.1f 个数量级，无候选对照。" % (d1, d2))
print("  跨线观察：两线都要求超校准输运——S 线（朗道-齐纳宽度）需 17 至 68 倍，")
print("    f 线（本判句）需 3.039e6 倍；量级不同而方向相同，「超校准输运的缺席」为共同结构性缺口。")
print()

print("== 图 ==")
fig, ax = plt.subplots(figsize=(10.2, 4.2), dpi=110)
fig.subplots_adjust(left=0.06, right=0.97, top=0.86, bottom=0.17)

pts = [
    ("记录阻尼上界 5.6e-60", 5.6e-60, "#888888"),
    ("迁移率 0.596", 0.5958, "#14508c"),
    ("弹道单步 1", 1.0, "#14508c"),
    ("校准上界 2 pi", 2.0 * math.pi, "#14508c"),
    ("目标 1.9e7", D_REQ, "#c0392b"),
    ("弹道 x w_def 3.8e21", W_DEF, "#666666"),
    ("hbar/M 1.9e22", D_Q, "#666666"),
]
for i, (tag, v, c) in enumerate(pts):
    y = 0.85 - 0.13 * i
    ax.plot([v], [y], "o", color=c, markersize=9)
    ax.annotate(tag, (v * 1.6, y), fontsize=9, color=c, va="center")
ax.axvspan(1e1, 1e21, color="#fff3d6", alpha=0.8, label="空档（无体系候选）")
ax.set_xscale("log")
ax.set_xlim(1e-61, 1e24)
ax.set_ylim(0, 1)
ax.set_yticks([])
ax.set_xlabel("扩散型系数（格² 每拍，对数）", fontsize=11)
ax.set_title("超扩散清点：目标 1.9e7 落在经典档与弹道档之间的空档", fontsize=11.5, pad=8)
ax.grid(True, axis="x", which="both", ls=":", lw=0.5, alpha=0.6)
ax.legend(fontsize=9, loc="upper left")
fig.savefig(IMG)
print("  已保存：", IMG)
