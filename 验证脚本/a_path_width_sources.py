# -*- coding: utf-8 -*-
"""A 路宽度来源的三路审查：校准扩散、势曲率、KZ 冻结（第 380 条）。

判据（确定性算术，无自由参数；口径为格点单位 l_P = t_P = 1）：
  W1 校准扩散承载：w = kappa/v（v = 1 格每步），kappa 有结构上界 omega_P = 2 pi
  W2 势曲率承载：w = E_P/Delta（质量参数倒数）
  W3 KZ 冻结承载：w = (tau_Q/t_P)^alpha（alpha = nu/(1+nu z)，取 1/3 与 1/4 两档）
  W4 时标档对照与活口判定

运行：python 验证脚本/a_path_width_sources.py
"""
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "images", "a_path_width_sources.png")

PROD = 106.7 / (2.0 * math.pi)     # 计数式乘积 16.9818
T_P = 5.391247e-44                 # 秒
OMEGA_P = 2.0 * math.pi            # 校准率结构上界（格子单位，正稿第五节）
W_LO = PROD                        # 17.0 步
W_HI = 4.0 * PROD                  # 67.9 步
T_2727 = (10.0 ** -39.4 / T_P, 10.0 ** -34.5 / T_P)   # 因果饱和档，t_P 单位
T_W1 = 4.09e-8 / T_P                                  # W1 档，t_P 单位
T_UNIV = 4.35e17 / T_P                                # 宇宙年龄，t_P 单位（对照）

print("== 设定 ==")
print("  计数式：(Delta/E_P)^2 x w = %.4f；A 路宽度窗口 %.1f 至 %.1f 步" % (PROD, W_LO, W_HI))
print("  校准率结构上界：omega_P = 2 pi = %.4f（格点单位）" % OMEGA_P)
print()

print("== W1 校准扩散承载 ==")
w_max = OMEGA_P
d_min = math.sqrt(PROD / w_max)
print("  判据：w = kappa/v（v = 1）；kappa <= omega_P -> w <= %.4f 步" % w_max)
print("  读数：上限 %.2f 步，对 A 路窗口下端差 %.2f 倍、对上端差 %.2f 倍" % (w_max, W_LO / w_max, W_HI / w_max))
print("  缺口换算：Delta >= %.4f E_P；超截断 %.1f%%（松读法）、%.1f%%（两摆程）" % (
    d_min, (d_min / 1.0 - 1) * 100, (d_min / 0.5 - 1) * 100))
print("  判读：判负")
print()

print("== W2 势曲率承载 ==")
print("  判据：w = E_P/Delta；代入计数式：(Delta/E_P)^2 x (E_P/Delta) = Delta/E_P = %.2f" % PROD)
print("  读数：Delta = %.2f E_P；超截断 %.2f 倍（松读法）、%.2f 倍（两摆程）" % (PROD, PROD, 2 * PROD))
print("  判读：判负")
print()

print("== W3 KZ 冻结承载 ==")
print("  判据：w = (tau_Q/t_P)^alpha")
for alpha, tag in ((1.0 / 3.0, "1/3"), (1.0 / 4.0, "1/4")):
    tq_lo = W_LO ** (1.0 / alpha)
    tq_hi = W_HI ** (1.0 / alpha)
    print("  alpha = %s: 反解窗口 tau_Q = %.2e 至 %.2e t_P" % (tag, tq_lo, tq_hi))
print("  时标档正算（alpha = 1/3）：")
for tq, tag in ((T_2727[0], "P4-27 下界"), (T_2727[1], "P4-27 上界"), (T_W1, "W1 档")):
    w = tq ** (1.0 / 3.0)
    d = math.sqrt(PROD / w)
    print("    %-12s tau_Q = %.2e t_P -> w = %.2f 步，Delta = %.4g E_P" % (tag, tq, w, d))
print("  判读：活口")
print()

print("== W4 交集与对照 ==")
tq_req_lo = W_LO ** 3.0
tq_req_hi = W_HI ** 3.0
overlap_lo = max(T_2727[0], tq_req_lo)
overlap_hi = T_2727[1]
print("  松读法（Delta <= E_P，需 tau_Q >= %.2e）：P4-27 档交集 = [%.2e, %.2e]" % (
    tq_req_lo, max(T_2727[0], tq_req_lo), overlap_hi))
print("  两摆程（Delta <= E_P/2，需 tau_Q >= %.2e）：P4-27 档交集 = [%.2e, %.2e]（非空）" % (
    tq_req_hi, max(T_2727[0], tq_req_hi), overlap_hi))
v_gev = 246.0
E_P_gev = 1.220890e19
dv = v_gev / E_P_gev
w_needed = PROD / (dv * dv)
tq_needed = w_needed ** 3.0
print("  B 路对照：Delta = 246 GeV = %.2e E_P -> 需 w = %.2e 步、tau_Q = %.2e t_P" % (dv, w_needed, tq_needed))
print("  宇宙年龄对照：tau_Q 需求超宇宙年龄 %.1f 个数量级" % math.log10(tq_needed / T_UNIV))
d_kz_hi = math.sqrt(PROD / (T_2727[0] ** (1.0 / 3.0)))
d_kz_lo = math.sqrt(PROD / (T_2727[1] ** (1.0 / 3.0)))
print("  判读：A 路 KZ 档 Delta = %.3f 至 %.3f E_P 与 B 路 %.2e E_P 差 %.1f 至 %.1f 个数量级" % (d_kz_lo, d_kz_hi, dv, math.log10(d_kz_lo / dv), math.log10(d_kz_hi / dv)))
print()

print("== 图 ==")
fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), dpi=110)
fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.14, wspace=0.30)

ax = axes[0]
xs = [10 ** (k / 10.0) for k in range(-20, 21)]
ws = [PROD / (x * x) for x in xs]
ax.loglog(xs, ws, color="#14508c", lw=2.0, label="计数式解集（(缺口/E_P)^2 x w = 16.98）")
ax.axhline(OMEGA_P, color="#c0392b", ls="--", lw=1.4, label="校准扩散上界 w = 2 pi")
ax.loglog(xs, [1.0 / x for x in xs], color="#2f7a2f", ls=":", lw=1.4, label="势曲率承载 w = E_P/缺口")
ax.axvspan(0.01, 0.5, color="#f6d3cf", alpha=0.55)
ax.axvline(0.5, color="#c0392b", ls="--", lw=0.9)
ax.axvline(1.0, color="#2f7a2f", ls=":", lw=0.9)
ax.plot([d_min], [OMEGA_P], "o", color="#c0392b")
ax.annotate("(1.64, 6.28)", (1.8, 3.2), fontsize=10, color="#c0392b")
ax.plot([PROD], [1.0], "o", color="#2f7a2f")
ax.annotate("(16.98, 1)", (6.5, 0.55), fontsize=10, color="#2f7a2f")
ax.set_xlim(1e-2, 1e2)
ax.set_ylim(1e-1, 1e4)
ax.set_xlabel("交叉缺口 / E_P", fontsize=11)
ax.set_ylabel("前沿宽度（步）", fontsize=11)
ax.set_title("(a) 三路承载与截断（粉色为截断内）", fontsize=12.5, pad=8)
ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
ax.legend(fontsize=8.5, loc="lower left")

ax = axes[1]
wgrid = [10 ** (k / 20.0) for k in range(20, 201)]
ax.loglog(wgrid, [w ** 3 for w in wgrid], color="#14508c", lw=2.0, label="KZ，alpha = 1/3（tau_Q = w^3）")
ax.loglog(wgrid, [w ** 4 for w in wgrid], color="#2f7a2f", lw=2.0, ls="--", label="KZ，alpha = 1/4（tau_Q = w^4）")
ax.axvspan(W_LO, W_HI, color="#fff3d6", label="A 路宽度窗口 17 至 68 步")
ax.axhspan(T_2727[0], T_2727[1], color="#dce9f7", label="P4-27 因果饱和档")
ax.axhline(T_W1, color="#7030a0", ls=":", lw=1.6, label="W1 档 7.6e35 t_P")
ax.set_xlim(10, 1e10)
ax.set_ylim(1e2, 1e38)
ax.set_xlabel("前沿宽度（步，对数）", fontsize=11)
ax.set_ylabel("淬火时标 tau_Q / t_P（对数）", fontsize=11)
ax.set_title("(b) KZ 反解：时标与宽度的对应", fontsize=12.5, pad=8)
ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
ax.legend(fontsize=8.5, loc="upper left")

fig.savefig(IMG)
print("  已保存：", IMG)
