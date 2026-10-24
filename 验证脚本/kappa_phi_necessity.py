# -*- coding: utf-8 -*-
"""kappa_phi 必要性：N1 极限清单与 N2 替代机制的读数复现要求（第 359 条）。
运行：python 验证脚本/kappa_phi_necessity.py
"""
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrow

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "images", "kappa_phi_necessity_map.png")

t_P = 5.391247e-44
m_P = 2.176434e-8
m_e = 9.1093837015e-31
c = 299792458.0
MeV_per_J = 6.241509074e12
hbar = 1.054571817e-34
kap_tP = 9.293805e-47
K = 3.0 * kap_tP
m_Phi = 2.0 * math.pi
J = 1.0 / (2.0 * math.pi)
w_P = 2.0 * math.pi / t_P

print("== N1 清单：kappa_phi 的角色与 kappa_phi -> 0 去向 ==")
roles = [
    ("锚核振幅锁定（粒性侧）", "消失", "绑定能 ∝ kappa_phi 标度（实测 3.13/121.9/359.4）；锁不住则无锚核、无质量标签"),
    ("相对相位能隙 w0^2 = 3 w_P kappa_phi", "消失", "kappa_phi -> 0 时能隙归零，缺口模式不存在（M-E 锚失去对象）"),
    ("sigma 锁定势 K = 3 kappa_phi t_P", "消失", "K -> 0，整周锁定势消失，孤子质量 (4/pi)sqrt(K/J)m_P -> 0"),
    ("色禁闭的几何解释（三元组不可分）", "消失", "异类耦合归零则三轴解耦，三元组可分解"),
    ("质量标度 m_f ~ hbar sqrt(kappa_phi)", "消失", "全部静止质量归零"),
    ("入口份额反解链 f = rho_b/(M rho_P)", "失效", "M -> 0 时分母归零，链条失效"),
    ("键刚度 k_b（同类耦合）", "保留", "公理 4 动能系数，与 kappa_phi 无关"),
    ("光速 c = l_P/t_P", "保留", "三条来源均不依赖介质弹性参数（时空观 §）"),
    ("同幅条件 lambda = 1", "保留", "真空条件导出路径不引用 kappa_phi（GV 轮）"),
    ("桥因子 epsilon = 1", "保留", "由摆程与作用量相位导出（MB-1）"),
    ("惯性系数 2pi 与 1/(2pi)", "保留", "公理 4 动能系数（MB-9）"),
]
for name, fate, why in roles:
    print("  %-34s %-4s %s" % (name, fate, why))
print("  读法：去 kappa_phi 后，质量侧六项同时消失，几何与光侧五项原样保留。")
print()

print("== N1 恒等式核对 ==")
r1 = math.sqrt(6.0 * math.pi * kap_tP)
print("  sqrt(6 pi kappa_phi t_P) = %.6e ; m_e/m_P = %.6e ; 相对差 = %.2e" % (r1, m_e / m_P, abs(r1 / (m_e / m_P) - 1)))
w0 = math.sqrt(3.0 * w_P * (kap_tP / t_P))
print("  sqrt(3 w_P kappa_phi) = %.6e 1/s ; m_e c^2/hbar = %.6e 1/s ; 相对差 = %.2e" % (w0, m_e * c * c / hbar, abs(w0 / (m_e * c * c / hbar) - 1)))
print("  K/J = %.6e ; 6 pi kappa_phi t_P = %.6e ; 相对差 = %.2e" % (K / J, 6.0 * math.pi * kap_tP, abs((K / J) / (6.0 * math.pi * kap_tP) - 1)))
print()

print("== N2 读数复现要求 ==")
print("  缺口 = 电子（0.51100 MeV）、孤子 = 4/pi 倍（0.65063 MeV）、能隙式 w0^2 = 3 w_P kappa_phi、")
print("  换算恒等式 K/J = 6 pi kappa_phi t_P、入口份额 f = 1.531335e-102。")
S = math.exp(-106.7)
kap0 = kap_tP / S
print("  裁决二十八：kappa_phi = kappa_phi0 * S，S = e^-106.7 = %.3e" % S)
print("  反解裸耦合 kappa_phi0 t_P = %.3f（O(1)，裸层无层级；全部小量在 S）" % kap0)
print()

print("== 图 ==")
fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0), dpi=110)
fig.subplots_adjust(left=0.04, right=0.98, top=0.88, bottom=0.08, wspace=0.14)

ax = axes[0]
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.set_title("(a) kappa_phi -> 0：哪些随之消失", fontsize=13, pad=8)
ax.text(0.26, 0.93, "质量侧（六项消失）", ha="center", fontsize=12, color="#b02a2a")
ax.text(0.76, 0.93, "几何与光侧（五项保留）", ha="center", fontsize=12, color="#2f7a2f")
gone = ["锚核锁定", "相对相位能隙", "sigma 锁定势", "色禁闭解释", "质量标度", "入口份额链"]
keep = ["键刚度 k_b", "光速 c", "同幅条件 lambda=1", "桥因子 epsilon=1", "惯性系数 2pi"]
for i, t in enumerate(gone):
    ax.add_patch(FancyBboxPatch((0.04, 0.74 - i * 0.125), 0.44, 0.095, boxstyle="round,pad=0.008,rounding_size=0.02",
                                fc="#f6d3cf", ec="#c0392b", lw=1.1))
    ax.text(0.26, 0.787 - i * 0.125, t, ha="center", va="center", fontsize=10.5, color="#8c1f1f")
for i, t in enumerate(keep):
    ax.add_patch(FancyBboxPatch((0.54, 0.74 - i * 0.125), 0.44, 0.095, boxstyle="round,pad=0.008,rounding_size=0.02",
                                fc="#e3f2e3", ec="#4f8f4f", lw=1.1))
    ax.text(0.76, 0.787 - i * 0.125, t, ha="center", va="center", fontsize=10.5, color="#1f5c1f")
ax.text(0.5, 0.045, "读法：去掉锁扣，剩下的是一张无质量的几何床垫", ha="center", fontsize=10.5, color="#333333")

ax = axes[1]
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.set_title("(b) 层级梯子：小量全在 S 里", fontsize=13, pad=8)
steps = [("裸耦合 kappa_phi0 t_P ≈ 2（O(1)）", "#eaf1f8", "#7a9cc0"),
         ("× 存活因子 S = e^-106.7", "#fff3d6", "#c99700"),
         ("= kappa_phi t_P = 9.29e-47", "#f6d3cf", "#c0392b"),
         ("开根号 √(6 pi …) = 4.19e-23 = m_e/m_P", "#e3f2e3", "#4f8f4f")]
for i, (t, fc, ec) in enumerate(steps):
    ax.add_patch(FancyBboxPatch((0.06, 0.70 - i * 0.155), 0.88, 0.115, boxstyle="round,pad=0.01,rounding_size=0.03",
                                fc=fc, ec=ec, lw=1.3))
    ax.text(0.5, 0.7575 - i * 0.155, t, ha="center", va="center", fontsize=11, color="#222222")
    if i < 3:
        ax.annotate("", xy=(0.5, 0.585 - i * 0.155), xytext=(0.5, 0.695 - i * 0.155),
                    arrowprops=dict(arrowstyle="->", color="#555555", lw=1.4))
fig.savefig(IMG)
print("saved:", IMG)
