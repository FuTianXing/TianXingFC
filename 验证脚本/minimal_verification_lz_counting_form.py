# -*- coding: utf-8 -*-
"""A 路判句：把 S = e^-106.7 化成 (缺口/E_P)^2 x 步数 = 17 的计数形式（第 361 条）。
运行：python 验证脚本/minimal_verification_lz_counting_form.py
"""
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "images", "lz_counting_form.png")

hbar = 1.054571817e-34
t_P = 5.391247e-44
E_P = hbar / t_P
m_e = 9.1093837015e-31
c = 299792458.0
exp_reg = 106.7
prod_req = exp_reg / (2.0 * math.pi)

print("== 输入 ==")
print("  E_P = %.6e J ; 登记指数 = %.1f ; 所需乘积 (缺口/E_P)^2 x 步数 = %.4f" % (E_P, exp_reg, prod_req))
print()

print("== 计数形式 ==")
print("  朗道-齐纳指数 = 2 pi (Delta/E_P)^2 (t_sweep/t_P)，其中扫速 v = E_P / t_sweep。")
print("  判句：缺口平方乘步数 = %.2f。" % prod_req)
print()
print("  满足条件的自然点：")
for d in (0.25, 0.5, 1.0, 2.0, 4.0):
    N = prod_req / (d * d)
    print("    Delta/E_P = %.2f  ->  t_sweep/t_P = %.2f" % (d, N))
print()

print("== 软缺口路线判负 ==")
Delta_soft = m_e * c * c
d_soft = Delta_soft / E_P
N_soft = prod_req / (d_soft * d_soft)
print("  Delta = hbar omega_0 = m_e c^2 = %.4e J，Delta/E_P = %.4e" % (Delta_soft, d_soft))
print("  所需步数 = %.3e（天文数字，软缺口路线判负）" % N_soft)
print()

print("== 结构性数字 ==")
print("  2 pi x %.3f = %.2f ; 34 pi = %.3f（相对差 %.3f%%）" % (prod_req, 2 * math.pi * prod_req, 34 * math.pi, abs(34 * math.pi - exp_reg) / exp_reg * 100))
print()

print("== 图 ==")
fig, ax = plt.subplots(figsize=(10.6, 5.2), dpi=110)
xs = [10 ** (k / 40.0) for k in range(-20, 25)]
ys = [prod_req / (x * x) for x in xs]
ax.loglog(xs, ys, color="#c0392b", lw=2.0, label="满足 S = e^-106.7 的 (Delta/E_P, 步数) 曲线")
ax.fill_between(xs, [1e-3] * len(xs), 1.0, color="#dddddd", alpha=0.5)
ax.axhline(1.0, color="#555555", ls="--", lw=1.0)
ax.text(0.35, 1.3, "步数 < 1：亚普朗克，非自然", fontsize=10, color="#555555")
for d, lab in ((0.5, "(0.5, 68)"), (1.0, "(1, 17)"), (2.0, "(2, 4.2)")):
    ax.plot([d], [prod_req / (d * d)], "o", color="#14508c")
    ax.annotate(lab, (d, prod_req / (d * d)), textcoords="offset points", xytext=(8, 6), fontsize=10.5, color="#14508c")
ax.set_xlabel("缺口 / E_P", fontsize=11)
ax.set_ylabel("所需步数（t_sweep / t_P）", fontsize=11)
ax.set_title("A 路计数形式：(缺口/E_P)^2 x 步数 = 17", fontsize=13, pad=8)
ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
ax.legend(fontsize=10, loc="upper right")
fig.savefig(IMG)
print("saved:", IMG)
