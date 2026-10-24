# -*- coding: utf-8 -*-
"""B 路势垒身份审查：定居密度作锁定势垒的两项代价（第 365 条）。
运行：python 验证脚本/barrier_identity_audit.py
"""
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "images", "barrier_identity_audit.png")

exp_reg = 106.7
v = 246.0
E_P_GeV = 1.220890e19
tab = [(2.15, 2.7), (2.10, 0.846), (2.05, 0.243), (2.00, 0.066)]

print("== 1. 势垒与定居密度的比例 ==")
print("  v = %.0f GeV ; v/E_P = %.3e" % (v, v / E_P_GeV))
for p, T in tab:
    need = exp_reg * T
    print("  p = %.2f : 所需势垒 = %6.1f GeV = %.3f v" % (p, need, need / v))
print()

print("== 2. 精度代价：族参数 p 的敏感度 ==")
coef = 213.5
dlnT = -coef / 2.0
sens = abs(dlnT) / 2.15 ** 2
print("  由 t_*(p) 式，ln T = -(213.5/2)(1/p - 0.4651) + 常数，d lnT/dp = %.1f" % sens)
for tol in (0.05, 0.01, 0.001):
    print("  要把指数对上 %.1f%%，需 p 精度 %.2e" % (tol * 100, tol / sens))
print()

print("== 3. 势垒项对指数的灵敏度 ==")
print("  指数 = 势垒/T_*；势垒若取 v，指数 = v/T_*。族内读数：")
for p, T in tab:
    print("    p = %.2f : 指数 = %8.1f（目标 106.7）" % (p, v / T))
print()

print("== 4. 两个代价 ==")
print("  代价一：v 目前是观测输入（未计入参数地位声明的两项输入清单）。")
print("  代价二：v 自身的小量 v/E_P = %.2e 另有登记（10^-17 注记），S 的问题会转为 v 的问题。" % (v / E_P_GeV))
print()

print("== 图 ==")
fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), dpi=110)
fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.14, wspace=0.26)

ax = axes[0]
ps = [2.00 + 0.15 * k / 149.0 for k in range(150)]
Ts = [2.7 * math.exp(-(213.5 / 2.0) * ((1.0 / p - 1.0 / 2.15))) for p in ps]
ys = [v / T for T in Ts]
ax.semilogy(ps, ys, color="#14508c", lw=2.0, label="指数 = v / T_*(p)")
ax.axhline(exp_reg, color="#2f7a2f", ls="--", lw=1.2)
ax.annotate("目标 106.7", (2.005, 118), fontsize=10.5, color="#2f7a2f")
ax.annotate("p ≈ 2.14 处穿线", (2.135, 60), fontsize=10.5, color="#14508c")
ax.set_xlabel("族参数 p", fontsize=11)
ax.set_ylabel("指数（对数）", fontsize=11)
ax.set_title("(a) 定居密度作势垒：族内穿线", fontsize=12.5, pad=8)
ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
ax.legend(fontsize=10, loc="upper right")

ax = axes[1]
labels = ["p=2.15", "p=2.10", "p=2.05", "p=2.00"]
ratios = [exp_reg * T / v for _, T in tab]
ax.bar([0, 1, 2, 3], ratios, color="#7a9cc0", width=0.55)
for i, r in enumerate(ratios):
    ax.text(i, r * 1.05, "%.2f v" % r, ha="center", fontsize=10.5)
ax.axhline(1.0, color="#c0392b", ls="--", lw=1.4)
ax.annotate("v", (3.25, 1.06), fontsize=11, color="#c0392b", ha="right")
ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(labels, fontsize=10.5)
ax.set_ylabel("所需势垒 / v", fontsize=11)
ax.set_title("(b) 所需势垒与 v 的比值", fontsize=12.5, pad=8)
fig.savefig(IMG)
print("saved:", IMG)
