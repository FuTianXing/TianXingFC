# -*- coding: utf-8 -*-
"""xi 生长律族的联立判定：从单参数曲线到定点（第 413 条，P0-5 候选机制首轮）。

判据：
  X1 框架：生长律与链条联立（曲线 xi_c(t) 的解析式）
  X2 弹道律（alpha = 1）：联立解与健全性判负
  X3 幂律族扫描：允许域与 alpha 上界
  X4 alpha = 3/5 的解与综合

运行：python 验证脚本/xi_growth_law.py
"""
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "images", "xi_growth_law.png")

F_OBS = 2.12e-105
T_EQ = 2.745e55
Z_EQ = 3400.0
W_DEF = 3.80e21
T_UNIV = 4.35e17 / 5.391247e-44   # 宇宙年龄，拍（物理上限）

# 链条曲线：xi_c(t) = C * (t/t_eq)^{1/2}，C = (z_eq^-3 / f)^{1/3}
C_CURVE = (Z_EQ ** -3.0 / F_OBS) ** (1.0 / 3.0)

def xi_chain(t):
    return C_CURVE * (t / T_EQ) ** 0.5

def t_of_alpha(alpha):
    # t^{alpha-1/2} = C / sqrt(t_eq)  ->  t = (C/sqrt(t_eq))^{1/(alpha-1/2)}
    if abs(alpha - 0.5) < 1e-12:
        return None
    return (C_CURVE / math.sqrt(T_EQ)) ** (1.0 / (alpha - 0.5))

print("== X1 框架 ==")
print("  链条曲线：xi_c(t) = C (t/t_eq)^{1/2}，C = (z_eq^-3/f)^{1/3} = %.4e" % C_CURVE)
print("  生长律 xi = t^alpha 与曲线联立：t = (C/sqrt(t_eq))^{1/(alpha-1/2)}")
t_edge = 7.562e35  # 第 408 条边界解的 t
print("  核对：xi_c(%.3e) = %.4e（w_def = %.3e）" % (t_edge, xi_chain(t_edge), W_DEF))
print()

print("== X2 弹道律（alpha = 1） ==")
t_bal = t_of_alpha(1.0)
xi_bal = t_bal ** 1.0
print("  联立解：t = %.4e 拍，xi = %.4e 格" % (t_bal, xi_bal))
print("  健全性 xi >= w_def：比值 %.3e（差 %.1f 个数量级），判负" % (xi_bal / W_DEF, abs(math.log10(xi_bal / W_DEF))))
print()

print("== X3 幂律族扫描 ==")
a_max = math.log(W_DEF) / math.log(t_edge)
a_min = 0.5 + math.log(C_CURVE / math.sqrt(T_EQ)) / math.log(T_UNIV)
print("  alpha 上界（xi = w_def 处）：alpha_max = %.4f" % a_max)
print("  alpha 下界（t = 宇宙年龄 %.2e 拍处）：alpha_min = %.4f" % (T_UNIV, a_min))
print("  允许带：[%.4f, %.4f]，宽度 %.4f" % (a_min, a_max, a_max - a_min))
for alpha in (0.55, a_min + 1e-4, 0.58, 0.60, a_max - 1e-4, 0.65, 0.75, 1.00):
    t = t_of_alpha(alpha)
    xi = t ** alpha
    ok = (xi >= W_DEF) and (t >= 1.0) and (xi <= t) and (t <= T_UNIV)
    print("  alpha = %.4f：t = %.3e 拍，xi = %.3e 格，xi/w_def = %.3e，合法 = %s" % (
        alpha, t, xi, xi / W_DEF, ok))
print()

print("== X4 alpha = 3/5 与综合 ==")
t_35 = t_of_alpha(0.6)
xi_35 = t_35 ** 0.6
print("  alpha = 3/5：t = %.4e 拍 = %.3e 秒，xi = %.4e 格（%.2f w_def）" % (
    t_35, t_35 * 5.391247e-44, xi_35, xi_35 / W_DEF))
print("  该解满足：健全性（xi > w_def）、因果（xi < t）、t 年龄内（t < 宇宙年龄）；所需新参数为零")
print("  KZ 指数对照：nu/(1+nu z) = 3/5 对应 (nu, z) = (3/2, 1)；与 3D Ising 类不符，属超扩散族")
print("  判读：alpha = 3/5 是允许带内的自然分数候选；P0-5 缺口的一类候选机制")
print()

print("== X5 档位敏感性（三档 f） ==")
for name, f in (("可见物质", 2.12e-105), ("核子", 1.06e-105), ("总物质", 6.8e-105)):
    C = (Z_EQ ** -3.0 / f) ** (1.0 / 3.0)
    t_edge_f = T_EQ * (W_DEF / C) ** 2
    a_max_f = math.log(W_DEF) / math.log(t_edge_f)
    xi_univ = C * math.sqrt(T_UNIV / T_EQ)
    a_min_f = math.log(xi_univ) / math.log(T_UNIV)
    print("  %s档：f = %.2e，C = %.4e，t_edge = %.3e 拍，带 = [%.4f, %.4f]，3/5 在带内 = %s" % (
        name, f, C, t_edge_f, a_min_f, a_max_f, (a_min_f <= 0.6 <= a_max_f)))
print()

print("== 图 ==")
fig, ax = plt.subplots(figsize=(9.8, 5.0), dpi=110)
fig.subplots_adjust(left=0.10, right=0.97, top=0.88, bottom=0.14)

ts = [10 ** (25 + k * 0.12) for k in range(0, 101)]
ax.loglog(ts, [xi_chain(t) for t in ts], color="#c0392b", lw=2.2, label="链条曲线 xi ∝ t^{1/2}")
for alpha, c, tag in ((0.55, "#888888", "alpha = 0.55"), (0.6, "#2f7a2f", "alpha = 3/5"), (0.75, "#14508c", "alpha = 0.75"), (1.0, "#7030a0", "alpha = 1（弹道）")):
    ax.loglog(ts, [t ** alpha for t in ts], color=c, lw=1.5, ls="--", label=tag)
ax.axhline(W_DEF, color="#e08a1e", lw=1.6, label="健全性下界 w_def")
ax.plot([t_edge], [W_DEF], "o", color="#c0392b", markersize=8)
ax.annotate("边界解", (t_edge * 1.4, W_DEF * 0.5), fontsize=9.5, color="#c0392b")
ax.plot([t_35], [xi_35], "o", color="#2f7a2f", markersize=8)
ax.annotate("alpha = 3/5 解", (t_35 * 1.4, xi_35 * 2.5), fontsize=9.5, color="#2f7a2f")
ax.plot([t_bal], [xi_bal], "o", color="#7030a0", markersize=8)
ax.annotate("弹道解（违反下界）", (t_bal * 0.3, xi_bal * 0.3), fontsize=9.5, color="#7030a0")
ax.set_xlim(1e25, 1e42)
ax.set_ylim(1e7, 1e45)
ax.set_xlabel("相变时标 t_* / 拍（对数）", fontsize=11)
ax.set_ylabel("冻结关联 xi / 格（对数）", fontsize=11)
ax.set_title("生长律与链条的联立：交点在健全性下界之上的才合法", fontsize=11.5, pad=8)
ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
ax.legend(fontsize=8.5, loc="upper left")
fig.savefig(IMG)
print("  已保存：", IMG)
