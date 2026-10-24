# -*- coding: utf-8 -*-
"""A 路前沿侧侦察：前沿速度已知为 c，宽度缺失（第 362 条）。
运行：python 验证脚本/front_width_audit.py
"""
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "images", "front_width_audit.png")

exp_reg = 106.7
prod_req = exp_reg / (2.0 * math.pi)

print("== 侦察读数 ==")
print("  前沿速度：体系陈述为光速（波前以光速扫过；因果前沿每拍一格 = c）")
print("  缺失量：前沿宽度 w（每个格点被扫过所需的格数/步数）与交叉缺口 Delta")
print()

print("== 自然档（尖锐前沿 w = 1）==")
exp_nat = 2.0 * math.pi * 1.0
S_nat = math.exp(-exp_nat)
print("  指数 = 2 pi x 1 = %.4f ; S = %.4e" % (exp_nat, S_nat))
print("  目标 S = e^-106.7 = %.4e" % math.exp(-exp_reg))
print("  指数缺口 = %.2f ; S 的倍数差 = %.3e（约 %.1f 个数量级）" % (exp_reg - exp_nat, math.exp(exp_reg - exp_nat), math.log10(math.exp(exp_reg - exp_nat))))
print()

print("== 补足的两条路 ==")
print("  路一：固定 Delta = E_P，所需前沿宽度 w = %.2f 步" % prod_req)
for d in (0.5, 1.0, 2.0):
    print("    Delta/E_P = %.2f -> w = %.2f" % (d, prod_req / (d * d)))
print("  路二：固定 w = 1，所需缺口 Delta/E_P = sqrt(%.2f) = %.3f" % (prod_req, math.sqrt(prod_req)))
for w in (2.0, 4.0, 17.0):
    print("    w = %.0f -> Delta/E_P = %.3f" % (w, math.sqrt(prod_req / w)))
print()

print("== 附带观察：体系内其余指数深度 ==")
f = 1.531335e-102
print("  kappa_phi 的指数深度 = 106.7")
print("  落位概率 3 H0 t_P 的指数深度 = 139.15（元相常数词条）")
print("  入口份额 f 的指数深度 = %.2f" % (-math.log(f)))
print()

print("== 图 ==")
fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6), dpi=110)
fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.12, wspace=0.28)

ax = axes[0]
ws = [0.5 + 0.05 * k for k in range(0, 400)]
Ss = [math.exp(-2.0 * math.pi * w) for w in ws]
ax.semilogy(ws, Ss, color="#c0392b", lw=2.0, label="S = e^(-2 pi w)（Delta = E_P）")
ax.semilogy([1.0], [S_nat], "o", color="#14508c")
ax.annotate("自然档 (w=1, S=1.9e-3)", (1.0, S_nat), textcoords="offset points", xytext=(10, 8), fontsize=10.5, color="#14508c")
ax.axhline(math.exp(-exp_reg), color="#2f7a2f", ls="--", lw=1.2)
ax.annotate("目标 S = e^-106.7", (12.0, math.exp(-exp_reg)), textcoords="offset points", xytext=(0, 8), fontsize=10.5, color="#2f7a2f")
ax.axvline(prod_req, color="#7a9cc0", ls=":", lw=1.2)
ax.annotate("所需宽度 17 步", (prod_req, 1e-20), textcoords="offset points", xytext=(-64, 10), fontsize=10.5, color="#14508c")
ax.set_xlabel("前沿宽度 w（步）", fontsize=11)
ax.set_ylabel("S", fontsize=11)
ax.set_ylim(1e-50, 1.0)
ax.set_xlim(0.5, 20)
ax.set_title("(a) 前沿宽度决定抑制深度", fontsize=12.5, pad=8)
ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)

ax = axes[1]
names = ["kappa_phi\n106.7", "落位概率\n139.15", "入口份额 f\n234.8"]
vals = [106.7, 139.15, -math.log(f)]
ax.bar([0, 1, 2], vals, color=["#c0392b", "#c99700", "#7a9cc0"], width=0.55)
for i, v in enumerate(vals):
    ax.text(i, v + 4, "%.1f" % v, ha="center", fontsize=11)
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(names, fontsize=10.5)
ax.set_ylabel("指数深度", fontsize=11)
ax.set_ylim(0, 280)
ax.set_title("(b) 体系内三个指数深度（仅登记）", fontsize=12.5, pad=8)
fig.savefig(IMG)
print("saved:", IMG)
