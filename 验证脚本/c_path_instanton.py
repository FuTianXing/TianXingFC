# -*- coding: utf-8 -*-
"""C 路首轮：瞬子作用量的量级二分与 g^2 落点（第 393 条）。

判据：
  C1 标准形式清点：四种瞬子系数约定下的 g^2 要求
  C2 体系耦合谱对照：g^2 目标值与体系常数的距离
  C3 体系作用量候选：四个候选的数值与量级二分
  C4 综合：空白区判定与结构观察（目标 16.98 与 A 路计数式常数同构）

运行：python 验证脚本/c_path_instanton.py
"""
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "images", "c_path_instanton.png")

TARGET_N = 106.7
TARGET_S = TARGET_N / (2.0 * math.pi)   # 16.98
KAPPA_TP = 9.293805e-47
K_CONST = 2.788142e-46
J_CONST = 1.0

print("== C1 标准形式清点（指数 = S_E = c / g^2 = 106.7） ==")
for c, tag in ((2 * math.pi, "2 pi / g^2（theta 型）"),
               (math.pi, "pi / g^2"),
               (4 * math.pi ** 2, "4 pi^2 / g^2"),
               (8 * math.pi ** 2, "8 pi^2 / g^2（SU(2) YM 型）")):
    g2 = c / TARGET_N
    print("  %-26s g^2 = %.4f（g = %.4f）" % (tag, g2, math.sqrt(g2)))
print()

print("== C2 体系耦合谱对照 ==")
spectrum = [
    ("kappa_phi t_P", KAPPA_TP),
    ("K", K_CONST),
    ("J（归一）", J_CONST),
    ("kappa_phi / K", KAPPA_TP / K_CONST),
    ("1 / (2 pi)", 1.0 / (2 * math.pi)),
]
g2_targets = [2 * math.pi / TARGET_N, 8 * math.pi ** 2 / TARGET_N]
for tag, val in spectrum:
    d1 = max(val / g2_targets[0], g2_targets[0] / val)
    d2 = max(val / g2_targets[1], g2_targets[1] / val)
    print("  %-16s = %.4e；与 g^2 = %.4f 差 %.1f 个数量级；与 g^2 = %.4f 差 %.1f 个数量级" % (
        tag, val, g2_targets[0], abs(math.log10(d1)), g2_targets[1], abs(math.log10(d2))))
print("  判读：目标 g^2 的四个档（0.029 至 0.740）在体系耦合谱上无自然落点；")
print("    最接近者 1/(2 pi) = 0.159 与 0.0589 差 2.7 倍（0.43 个数量级），其次 1/3 差 5.7 倍。")
print()

print("== C3 体系作用量候选 ==")
sg = 8.0 * math.sqrt(J_CONST * K_CONST)
wkb = 2.0 * math.sqrt(2.0 * K_CONST) * 0.900
inv_sqrt = 1.0 / math.sqrt(KAPPA_TP)
inv_lin = 1.0 / KAPPA_TP
print("  a) sine-Gordon 瞬子（8 sqrt(JK)）：S_E = %.3e" % sg)
print("  b) WKB 势垒穿透（2 sqrt(2K) x 0.900）：S_E = %.3e" % wkb)
print("  c) 1 / sqrt(kappa_phi t_P)：S_E = %.3e" % inv_sqrt)
print("  d) 1 / (kappa_phi t_P)：S_E = %.3e" % inv_lin)
print("  目标：S_E = %.2f（= 106.7 / 2 pi）" % TARGET_S)
print("  量级二分：候选 a、b 在 1e-22 侧（离目标 23 个数量级）；候选 c、d 在 1e23 侧（离目标 21 与 44 个数量级）")
print()

print("== C4 综合 ==")
print("  空白区判定：目标 16.98 落在 1e-22 与 1e23 之间的空白区；四个自然候选无一接近。")
print("  结构观察：目标 16.98 = 106.7 / (2 pi)，与 A 路计数式常数（(Delta/E_P)^2 w = 16.98）同值；")
print("  两路共享同一反解结构（「17 的机制」），C 路的判定由此归约到该结构。")
print()

print("== 图 ==")
fig, ax = plt.subplots(figsize=(9.8, 4.6), dpi=110)
fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.17)

pts = [
    ("SG 瞬子 %.1e" % sg, sg, "#14508c"),
    ("WKB %.1e" % wkb, wkb, "#14508c"),
    ("目标 16.98", TARGET_S, "#c0392b"),
    ("1/sqrt(kappa t_P) %.0e" % inv_sqrt, inv_sqrt, "#666666"),
    ("1/(kappa t_P) %.0e" % inv_lin, inv_lin, "#666666"),
]
for i, (tag, x, c) in enumerate(pts):
    y = 0.8 - 0.16 * i
    ax.plot([x], [y], "o", color=c, markersize=9)
    ax.annotate(tag, (x * 1.6, y), fontsize=9.5, color=c, va="center")
ax.axvspan(1e-20, 1e21, color="#fff3d6", alpha=0.8, label="空白区（无自然候选）")
ax.set_xscale("log")
ax.set_xlim(1e-24, 1e48)
ax.set_ylim(0, 1)
ax.set_yticks([])
ax.set_xlabel("瞬子作用量 S_E（对数）", fontsize=11)
ax.set_title("C 路：作用量的量级二分与目标值的空白落点", fontsize=12, pad=8)
ax.grid(True, axis="x", which="both", ls=":", lw=0.5, alpha=0.6)
ax.legend(fontsize=9, loc="upper left")
fig.savefig(IMG)
print("  已保存：", IMG)
