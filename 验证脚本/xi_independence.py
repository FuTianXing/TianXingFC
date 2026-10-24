# -*- coding: utf-8 -*-
"""xi 的独立依据：单参数曲线结构与扩散建立的对照（第 408 条）。

判据：
  X1 约束清点：健全性下界、因果上界与 f 方程的单参数结构
  X2 曲线验证：xi ∝ t_*^{1/2}（xi 每加两个数量级、t_* 加四个数量级）
  X3 扩散建立假设：D 的唯一解与其与校准上界的对照
  X4 时间匹配读法与综合判定

运行：python 验证脚本/xi_independence.py
"""
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "images", "xi_independence.png")

W_DEF = 3.80e21
F_VIS = 2.12e-105
T_P = 5.391247e-44
T_EQ_S = 1.48e12
Z_EQ = 3400.0
T_EQ = T_EQ_S / T_P            # 拍
KAPPA_UB = 2.0 * math.pi       # 校准扩散上界（格^2 每拍）

# 由 f = xi^-3 (t/t_eq)^{3/2} z_eq^-3 = F 解 t
def t_star_of_xi(xi):
    return T_EQ * (F_VIS * xi ** 3 * Z_EQ ** 3) ** (2.0 / 3.0)

t_edge = t_star_of_xi(W_DEF)

print("== X1 约束清点 ==")
print("  下界（健全性）：xi >= w_def = %.2e 格" % W_DEF)
print("  上界（因果）：xi <= c t_*（冻结关联不超过相变期间的因果视界）")
print("  f 方程：f = xi^-3 (t_*/t_eq)^{3/2} z_eq^-3 = %.3e（可见物质档）" % F_VIS)
print("  两个未知量、一个方程：xi 与 t_* 呈单参数联动")
print()

print("== X2 曲线验证（xi ∝ t_*^{1/2}） ==")
for xf in (1.0, 100.0, 1.0e4):
    xi = W_DEF * xf
    t = t_star_of_xi(xi)
    fchk = xi ** -3.0 * (t / T_EQ) ** 1.5 * Z_EQ ** -3.0
    print("  xi = %.2e 格（%g x w_def）：t_* = %.3e 拍；f 复核 = %.3e" % (xi, xf, t, fchk))
print("  判读：xi 每乘 100，t_* 乘 1e4，f 保持不变；曲线结构确认")
print()

print("== X3 扩散建立假设 ==")
# xi^2 = D t 与 f 方程联立：f = D^{-3/2} t_eq^{-3/2} z_eq^{-3}
D_req = (F_VIS * (T_EQ ** 1.5) * (Z_EQ ** 3)) ** (-2.0 / 3.0)
print("  假设 xi^2 = D t_*（扩散建立），代入 f 方程消去 t_*：f = D^{-3/2} t_eq^{-3/2} z_eq^-3")
print("  D 的唯一解：D_req = %.3e 格^2 每拍" % D_req)
print("  与校准扩散上界 %.4f 之比：%.3e（xi 方向 %.0f 倍）" % (KAPPA_UB, D_req / KAPPA_UB, math.sqrt(D_req / KAPPA_UB)))
xi_cal = math.sqrt(KAPPA_UB * t_edge)
print("  交叉复核：t_* = %.3e 拍处，校准扩散只能建立 xi = %.3e 格（w_def 的 1/%.0f）" % (t_edge, xi_cal, W_DEF / xi_cal))
print()

print("== X4 时间匹配读法与综合 ==")
print("  边界解处：D_req x t_* = %.3e；w_def^2 = %.3e（差 %.1f%%）" % (
    D_req * t_edge, W_DEF ** 2, abs(D_req * t_edge / W_DEF ** 2 - 1) * 100))
print("  读法：D t_* = w_def^2 即「建立缺陷宽度所需时间 = 相变时标」（时间匹配条件）")
print("  综合：xi 无独立依据；链条为单参数曲线；若取扩散建立，D 必须 = 1.9e7（超校准上界 3.1e6 倍，无依据）")
print()

print("== 图 ==")
fig, ax = plt.subplots(figsize=(9.8, 5.0), dpi=110)
fig.subplots_adjust(left=0.10, right=0.97, top=0.88, bottom=0.14)

ts = [10 ** (30 + k * 0.15) for k in range(0, 121)]
f_line = [math.sqrt(D_req * t) for t in ts]
cal_line = [math.sqrt(KAPPA_UB * t) for t in ts]
ax.loglog(ts, f_line, color="#c0392b", lw=2.0, label="f 曲线（xi = sqrt(D_req t)，D_req = 1.9e7）")
ax.loglog(ts, cal_line, color="#14508c", lw=2.0, label="校准扩散线（xi = sqrt(2 pi t)）")
ax.loglog(ts, ts, color="#888888", lw=1.4, ls="--", label="因果线（xi = c t）")
ax.plot([t_edge], [W_DEF], "o", color="#2f7a2f", markersize=9)
ax.annotate("边界解 (7.8e35, 3.8e21)", (t_edge * 1.3, W_DEF * 1.6), fontsize=9.5, color="#2f7a2f")
ax.set_xlim(1e30, 1e48)
ax.set_ylim(1e15, 1e50)
ax.set_xlabel("相变时标 t_* / 拍（对数）", fontsize=11)
ax.set_ylabel("冻结关联 xi / 格（对数）", fontsize=11)
ax.set_title("两条平行线：f 曲线与校准扩散线相距 1749 倍（xi 方向）", fontsize=11.5, pad=8)
ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
ax.legend(fontsize=8.5, loc="upper left")
fig.savefig(IMG)
print("  已保存：", IMG)
