# -*- coding: utf-8 -*-
"""LIV 约定线：相位与群速度的因子 3、球面平均与 SME 记号对应（第 373 条）。
运行：python 验证脚本/liv_conventions.py
"""
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "images", "liv_conventions.png")

E_P_eV = 1.220890e28
dirs = [("[100]", 1.0), ("[110]", 0.5), ("[111]", 1.0 / 3.0)]
S_avg = 3.0 / 5.0
D1 = 640e6 * 3.085677581e16
t1 = D1 / 299792458.0
t_gpc = 1e9 * 3.085677581e16 / 299792458.0

print("== 1. 相位与群速度：因子 3 ==")
print("  omega = k[1 - (S/24)(k/E_P)^2]  =>  v_ph 偏差 = (S/24)(E/E_P)^2 ; v_g 偏差 = 3 x 上式 = (S/8)(E/E_P)^2")
for name, S in dirs:
    print("  %-6s S = %.4f ; 等效能标：相位 %6.3f E_P ; 群速度 %6.3f E_P" % (name, S, math.sqrt(24.0 / S), math.sqrt(8.0 / S)))
print("  球面平均 S = %.3f ; 等效能标：相位 %.3f E_P ; 群速度 %.3f E_P" % (S_avg, math.sqrt(24.0 / S_avg), math.sqrt(8.0 / S_avg)))
print()

print("== 2. 时延（群速度口径）==")
for E, lab in ((1.3e13, "13 TeV"), (1e20, "1e20 eV")):
    row = "  %-8s" % lab
    for name, S in dirs:
        d = (S / 8.0) * (E / E_P_eV) ** 2 * t1
        row += "  %-6s %.3e s" % (name, d)
    print(row + "   （640 Mpc）")
for name, S in dirs:
    d = (S / 8.0) * (1e20 / E_P_eV) ** 2 * t_gpc
    print("  %-6s 1 Gpc : %.3f s" % (name, d))
print()

print("== 3. 立方谐波分解与球面平均 ==")
vals = [S for _, S in dirs]
print("  S([100]) = %.3f ; S([110]) = %.3f ; S([111]) = %.3f ; 平均 = %.3f" % (vals[0], vals[1], vals[2], S_avg))
print("  相对平均的偏离：%+.1f%%（[100]）与 %+.1f%%（[111]）" % ((vals[0] / S_avg - 1) * 100, (vals[2] / S_avg - 1) * 100))
print("  方向平均即各向同性系数：文献里按各向同性给出的界，对的是平均值那一支")
print()

print("== 4. 与 LHAASO 二次界的余量（界 6e-8 E_P）==")
bound = 6e-8
for lab, val in (("群速度 [100]", math.sqrt(8.0 / 1.0)), ("群速度球面平均", math.sqrt(8.0 / S_avg)), ("相位 [100]", math.sqrt(24.0 / 1.0))):
    print("  %-14s 等效 %.3f E_P ; 余量 %.2e 倍（%.2f 个数量级）" % (lab, val, val / bound, math.log10(val / bound)))
print()

print("== 图 ==")
fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), dpi=110)
fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.14, wspace=0.26)

ax = axes[0]
xs = [k / 200.0 for k in range(0, 201)]
S = []
for x in xs:
    # 路径 [100] -> [110] -> [111]：方向 (1-x, x, 0) 归一
    n = (1 - x, x, 0.0)
    m = math.sqrt(sum(v * v for v in n))
    S.append(sum((v / m) ** 4 for v in n))
ax.plot(xs, S, color="#14508c", lw=2.0, label="沿 [100] 到 [110] 的方向")
for x in (0.0, 0.5, 1.0):
    n = (1 - x, x, 0.0)
    m = math.sqrt(sum(v * v for v in n))
    s = sum((v / m) ** 4 for v in n)
    ax.plot([x], [s], "o", color="#c0392b")
ax.axhline(S_avg, color="#2f7a2f", ls="--", lw=1.2)
ax.annotate("球面平均 3/5", (0.05, S_avg + 0.02), fontsize=10, color="#2f7a2f")
ax.set_xlabel("沿线参数", fontsize=11)
ax.set_ylabel("S = sum n_a^4", fontsize=11)
ax.set_title("(a) 方向因子的谱（含球面平均）", fontsize=12.5, pad=8)
ax.grid(True, ls=":", lw=0.5, alpha=0.6)
ax.legend(fontsize=10, loc="upper right")

ax = axes[1]
names = ["相位 [100]", "群速 [100]", "群速 平均", "LHAASO 界"]
vals2 = [math.sqrt(24.0), math.sqrt(8.0), math.sqrt(8.0 / S_avg), bound * 1.220890e19 / 1.220890e19]
ax.bar([0, 1, 2], [math.sqrt(24.0), math.sqrt(8.0), math.sqrt(8.0 / S_avg)], color=["#aab7c4", "#14508c", "#2f7a2f"], width=0.5)
for i, v in enumerate([math.sqrt(24.0), math.sqrt(8.0), math.sqrt(8.0 / S_avg)]):
    ax.text(i, v * 1.06, "%.2f E_P" % v, ha="center", fontsize=10)
ax.axhline(bound, color="#c0392b", ls="--", lw=1.4)
ax.annotate("6e-8 E_P（LHAASO 二次界）", (2.4, bound * 1.6), fontsize=10, color="#c0392b", ha="right")
ax.set_yscale("log")
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(["相位 [100]", "群速 [100]", "群速球面平均"], fontsize=10)
ax.set_ylabel("等效 E_QG,2（E_P）", fontsize=11)
ax.set_ylim(1e-9, 60)
ax.set_title("(b) 两种约定下的等效标度与文献界", fontsize=12.5, pad=8)
fig.savefig(IMG)
print("saved:", IMG)
