# -*- coding: utf-8 -*-
"""存活因子 S 的机制候选：四种指数抑制的参数量级与判句（第 360 条）。
运行：python 验证脚本/survival_factor_candidates.py
"""
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "images", "survival_factor_candidates.png")

t_P = 5.391247e-44
hbar = 1.054571817e-34
k_B = 1.380649e-23
E_P = hbar / t_P
T_P = E_P / k_B
kap_tP = 9.293805e-47
exp_reg = 106.7

print("== 输入 ==")
print("  kappa_phi t_P = %.6e ; 登记指数 ln(1/S) = %.1f" % (kap_tP, exp_reg))
print("  E_P = %.4e J ; T_P = %.4e K" % (E_P, T_P))
for bare in (1.0, 2.0, 2.0 * math.pi / 3.0, math.sqrt(2.0 * math.pi)):
    S = kap_tP / bare
    print("  若裸耦合 kappa_phi0 t_P = %.4f，则 S = %.4e，指数 = %.2f" % (bare, S, -math.log(S)))
print()

print("== 四路线：把指数 106.7 摊到 O(1) 参数上 ==")
print("  A 朗道-齐纳：S = exp(-2 pi Delta^2/(hbar v))；取 Delta = E_P，得扫速")
v_req = 2.0 * math.pi * E_P * E_P / (hbar * exp_reg)
print("     v = %.4e J/s = %.5f * (hbar/t_P^2)（O(1/20)，普朗克附近）" % (v_req, v_req / (hbar / t_P ** 2)))
T_req = E_P / (exp_reg * k_B)
print("  B 热冻结：S = exp(-DeltaE/(k_B T))；取 DeltaE = E_P，得 T = %.4e K = T_P/%.1f" % (T_req, T_P / T_req))
g2 = 2.0 * math.pi / exp_reg
print("  C 瞬子：S = exp(-S_E/hbar)；若 S_E = 2 pi/g^2，得 g^2 = %.5f（弱耦合档）" % g2)
print("  D 基布尔-祖雷克：缺陷密度为幂律，不产生指数；除非叠加激活过程（归入 B）")
print()

print("== 结构性数字观察（只登记，不作断言）==")
print("  34 pi = %.4f（与 106.7 相对差 %.3f%%）；2 pi x 17 = 同上" % (34.0 * math.pi, abs(34.0 * math.pi - exp_reg) / exp_reg * 100))
S34 = math.exp(-34.0 * math.pi)
print("  e^-34pi = %.4e ；与 kappa_phi t_P 相比（若裸耦合取 1）差 %.1f%%" % (S34, abs(S34 / kap_tP - 1) * 100))
print()

print("== 图 ==")
fig, ax = plt.subplots(figsize=(12.6, 4.6), dpi=110)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.set_title("存活因子 S = e^-106.7：四种机制的参数要求", fontsize=14, pad=8)
cards = [
    ("A 朗道-齐纳\n扫速 v = 0.0589 hbar/t_P^2", "#eaf1f8", "#7a9cc0"),
    ("B 热冻结\nT = T_P / 106.7", "#fff3d6", "#c99700"),
    ("C 瞬子\ng^2 = 2 pi / 106.7 = 0.0589", "#f6d3cf", "#c0392b"),
    ("D 基布尔-祖雷克\n幂律，指数需另加激活", "#eeeeee", "#888888"),
]
for i, (t, fc, ec) in enumerate(cards):
    x = 0.03 + (i % 2) * 0.49
    y = 0.56 - (i // 2) * 0.30
    ax.add_patch(FancyBboxPatch((x, y), 0.45, 0.24, boxstyle="round,pad=0.012,rounding_size=0.03", fc=fc, ec=ec, lw=1.4))
    ax.text(x + 0.225, y + 0.12, t, ha="center", va="center", fontsize=11.5, color="#222222")
ax.text(0.5, 0.16, "四路都能容纳 106.7，且所需参数都是 O(1)；判别靠哪一路与同步冻结、普朗克尺度缺口、校准接力自洽",
        ha="center", fontsize=11, color="#333333")
ax.text(0.5, 0.06, "登记观察：106.7 与 34 pi = 106.814 相对差 0.11%（仅提示，不作断言）", ha="center", fontsize=10.5, color="#666666")
fig.savefig(IMG)
print("saved:", IMG)
