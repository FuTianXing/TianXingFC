# -*- coding: utf-8 -*-
"""SME 对接：d=6 立方系数的等效归一、boost 量级与三通道对照（第 367 条）。
运行：python 验证脚本/minimal_verification_lv_sme_mapping.py
"""
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "images", "lv_sme_mapping.png")

E_P_eV = 1.220890e28
l_P = 1.616255e-35
c = 299792458.0
dirs = [("[100]", 1.0), ("[110]", 0.5), ("[111]", 1.0 / 3.0)]

print("== 1. 等效归一：把框架的二次项写成标准 (E/E_QG2)^2 形式 ==")
print("  框架：delta_omega/omega = -(E/E_P)^2 x (S/24)，S = sum n_a^4")
print("  标准：delta_omega/omega = -(E/E_QG2)^2  ==>  E_QG2 = E_P x sqrt(24/S)")
for name, S in dirs:
    print("  %-6s S = %.4f ; E_QG2 = %.3f E_P = %.3e GeV" % (name, S, math.sqrt(24.0 / S), math.sqrt(24.0 / S) * 1.220890e19))
print()

print("== 2. 三通道对照 ==")
print("  (a) 真空双折射（Integral 型）：框架算符对 k 为偶，首阶无双折射 ==> 该通道不适用")
print("  (b) 谱线展宽（ESPRESSO 型）：框架为规则晶格的相干色散，无随机泡沫 ==> 该通道不适用")
print("  (c) 高能计时（LHAASO/Fermi 型）：框架无一次项；二次项如下")
for E, lab in ((1.0, "1 eV"), (1.3e13, "13 TeV"), (1e20, "1e20 eV")):
    d = (E / E_P_eV) ** 2
    print("      E = %-8s : (E/E_P)^2 = %.3e ; [100] delta/omega = %.3e" % (lab, d, d / 24.0))
print()

print("== 3. 代表性能量下的延迟（距离 640 Mpc，z = 0.151）==")
D = 640e6 * 3.085677581e16
tD = D / c
print("  光行时 = %.3e s (%.2f Gyr)" % (tD, tD / 3.1557e16))
for E, lab in ((1.3e13, "13 TeV"), (1e20, "1e20 eV")):
    for name, S in dirs:
        rel = (E / E_P_eV) ** 2 * S / 24.0
        print("  %-8s %-6s : 相对偏离 %.3e ; 延迟 %.3e s" % (lab, name, rel, rel * tD))
print()

print("== 4. boost 量级：运动观测者对纯空间张量的修正 ==")
beta = 370e3 / c
print("  beta = %.3e (370 km/s) ; beta^2 = %.3e" % (beta, beta * beta))
print("  纯空间四阶张量在 boost 下的纯空间分量修正为 O(beta^2)（无 O(beta) 项）")
print("  故系数的速度调制深度约 %.1e 相对量；观测量再乘 (E/E_P)^2" % (beta * beta))
for E, lab in ((13e12, "13 TeV"), (1e20, "1e20 eV")):
    mod = beta * beta * (E / E_P_eV) ** 2 / 24.0
    print("  %-8s : 调制量 %.3e（相对）；对应延迟 %.3e s" % (lab, mod, mod * tD))
print()

print("== 图 ==")
fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), dpi=110)
fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.14, wspace=0.26)

ax = axes[0]
Es = [10 ** (k / 4.0) for k in range(0, 85)]
for name, S in dirs:
    ax.loglog(Es, [(E / E_P_eV) ** 2 * S / 24.0 for E in Es], lw=1.8, label=name)
for E, lab in ((1.0, "1 eV"), (1.3e13, "13 TeV"), (1e20, "1e20 eV")):
    ax.axvline(E, color="#999999", ls=":", lw=1.0)
ax.annotate("1 eV", (1.2, 1e-52), fontsize=9.5, color="#666666")
ax.annotate("13 TeV", (2e13, 1e-52), fontsize=9.5, color="#666666")
ax.annotate("1e20 eV", (1.3e20, 1e-52), fontsize=9.5, color="#666666")
ax.set_xlabel("光子能量（eV）", fontsize=11)
ax.set_ylabel("|delta omega / omega|", fontsize=11)
ax.set_title("(a) 框架的二次偏离与方向族", fontsize=12.5, pad=8)
ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
ax.legend(fontsize=10, loc="upper left")

ax = axes[1]
for name, S in dirs:
    ax.loglog(Es, [(E / E_P_eV) ** 2 * S / 24.0 * tD for E in Es], lw=1.8, label=name)
ax.axhspan(1e-3, 1.0, color="#fff3d6", alpha=0.7)
ax.annotate("毫秒至秒档：现有高能计时分析的量级带", (1.0, 3.0), fontsize=9.5, color="#8a6100")
ax.set_xlabel("光子能量（eV）", fontsize=11)
ax.set_ylabel("延迟（s，640 Mpc）", fontsize=11)
ax.set_title("(b) 延迟与计时敏感带", fontsize=12.5, pad=8)
ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
ax.legend(fontsize=10, loc="upper left")
fig.savefig(IMG)
print("saved:", IMG)
