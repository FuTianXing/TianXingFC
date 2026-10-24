# -*- coding: utf-8 -*-
"""生长指数 3/5 的来源判识：KZ 解空间、最简分数与上端紧配合（第 414 条，P0-5 候选机制二轮）。

判据：
  Y1 带内最简分数枚举（分母 ≤ 20）
  Y2 KZ 解空间 nu = 3/(5-3z) 与 nu 趋于无穷的极限线
  Y3 上端紧配合闭式（alpha_max = L/(A+2L)，A/L 与 -1/3 之差）
  Y4 理想 f*（使 alpha_max = 3/5）与观测 f 的比值
  Y5 交叉恒等（2 alpha - 1 = 1/5；t_* = D_req^5）
  Y6 分数形式清点（形式候选，不构成机制）

运行：python 验证脚本/growth_exponent_audit.py
"""
import math
import os
from fractions import Fraction

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "images", "growth_exponent_audit.png")

F_OBS = 2.12e-105
T_EQ = 2.745e55
Z_EQ = 3400.0
W_DEF = 3.80e21
T_UNIV = 4.35e17 / 5.391247e-44

C_CURVE = (Z_EQ ** -3.0 / F_OBS) ** (1.0 / 3.0)
D_REQ = C_CURVE ** 2 / T_EQ

A_MAX = math.log(W_DEF) / math.log(T_EQ * (W_DEF / C_CURVE) ** 2)
A_MIN = 0.5 + math.log(C_CURVE / math.sqrt(T_EQ)) / math.log(T_UNIV)
LO, HI = A_MIN, A_MAX

print("== 框架（与第 413 条同式复算） ==")
print("  C = %.4e，D_req = C^2/t_eq = %.4e 格^2/拍" % (C_CURVE, D_REQ))
print("  允许带：[%.4f, %.4f]" % (LO, HI))
print()

print("== Y1 带内最简分数（分母 <= 20） ==")
found = []
for q in range(2, 21):
    for p in range(math.ceil(LO * q), math.floor(HI * q) + 1):
        f = Fraction(p, q)
        if f.denominator == q and LO <= float(f) <= HI:
            found.append(f)
found = sorted(set(found), key=lambda fr: (fr.denominator, fr))
for fr in found:
    print("  %s = %.6f（分母 %d）" % (fr, float(fr), fr.denominator))
print("  最小分母者：%s；第二：%s" % (found[0], found[1]))
print()

print("== Y2 KZ 解空间：nu = 3/(5-3z) ==")
z_list = [Fraction(1,6), Fraction(1,3), Fraction(1,2), Fraction(2,3), Fraction(5,6),
          Fraction(1,1), Fraction(7,6), Fraction(4,3), Fraction(3,2)]
for z in z_list:
    nu = Fraction(3,1) / (Fraction(5,1) - 3*z)
    alpha = nu / (1 + nu*z)
    print("  z = %s：nu = %s，alpha = %s（%.6f）" % (z, nu, alpha, float(alpha)))
print("  z = 5/3：极点，nu 趋于无穷，alpha 趋于 1/z = 3/5（KT 类线，纯标度）")
print("  最小分母组合：(nu, z) = (3/2, 1)")
print()

print("== Y3 上端紧配合闭式 ==")
A_ = math.log(T_EQ) - 2.0 * math.log(C_CURVE)
L_ = math.log(W_DEF)
a_max_check = L_ / (A_ + 2.0 * L_)
print("  A = ln(t_eq/C^2) = %.5f，L = ln(w_def) = %.5f" % (A_, L_))
print("  A/L = %.6f，-1/3 = %.6f，相对差 = %.3f%%" % (A_ / L_, -1.0/3.0, abs(A_ / L_ + 1.0/3.0) / (1.0/3.0) * 100.0))
print("  alpha_max 复算 = %.6f（对照 %.4f）" % (a_max_check, A_MAX))
print("  形式验证：1/(2 - 1/3) = %.6f" % (1.0 / (2.0 - 1.0/3.0)))
delta = A_ + L_ / 3.0
print("  delta = A + L/3 = %.5f，e^delta = %.5f" % (delta, math.exp(delta)))
print()

print("== Y4 理想 f*（使 alpha_max = 3/5） ==")
C_STAR = math.sqrt(T_EQ * W_DEF ** (1.0/3.0))
F_STAR = Z_EQ ** -3.0 / C_STAR ** 3.0
t_edge_star = T_EQ * (W_DEF / C_STAR) ** 2
a_max_star = math.log(W_DEF) / math.log(t_edge_star)
print("  C* = sqrt(t_eq * w_def^{1/3}) = %.4e" % C_STAR)
print("  f* = z_eq^-3 / C*^3 = %.4e" % F_STAR)
print("  复算 alpha_max(f*) = %.8f（期望 0.6）" % a_max_star)
print("  f*/f_obs = %.4f，f_obs/f* = %.4f" % (F_STAR / F_OBS, F_OBS / F_STAR))
print()

print("== Y5 交叉恒等 ==")
print("  2 alpha - 1（alpha = 3/5）= %.4f；<n_x^4> 球面平均（D=3）= 1/(D+2) = %.4f" % (2*0.6 - 1.0, 1.0/5.0))
t35 = D_REQ ** 5.0
print("  t_* = D_req^{1/(2 alpha - 1)} = D_req^5 = %.4e 拍（对照第 413 条 2.5393e36）" % t35)
print()

print("== Y6 分数形式清点（形式候选，不构成机制） ==")
D = 3
print("  D/(D+2) = %d/%d = %.6f（D = 3）" % (D, D+2, D/(D+2.0)))
print("  D/(2D-1) = %d/%d = %.6f（D = 3）" % (D, 2*D-1, D/(2.0*D-1)))
print("  (D+1)/(2D+1)：D = 2 给 3/5，D = 3 给 4/7")
print("  D/(2D-1) 族：D = 3 给 3/5，D = 4 给 4/7（该族在带内仅此两项；D >= 5 出带）")
print("  D/(D+2) 族：仅 D = 3 在带内（D = 2 给 1/2，D = 4 给 2/3）")
print()

print("== 图 ==")
fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6), dpi=110)
fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.16, wspace=0.22)

ax = axes[0]
ax.axvspan(LO, HI, color="#ffe9b0", alpha=0.8, label="允许带 [%.4f, %.4f]" % (LO, HI))
for fr in found:
    col = "#2f7a2f" if fr.denominator == 5 else "#14508c"
    ax.plot([float(fr)], [0.5], "o", color=col, markersize=9 if fr.denominator == 5 else 7)
    ax.annotate("%s" % fr, (float(fr), 0.5), textcoords="offset points", xytext=(0, 11), ha="center", fontsize=9.5)
ax.set_xlim(0.5, 0.65)
ax.set_ylim(0, 1)
ax.set_yticks([])
ax.set_xlabel("growth exponent alpha", fontsize=10.5)
ax.set_title("Y1 in-band simplest fractions (q <= 20)", fontsize=10.5)
ax.legend(fontsize=8.5, loc="upper left")

ax = axes[1]
zs = [i / 120.0 for i in range(1, 200)]
plot_z = [z for z in zs if z < 5.0/3.0 - 1e-6]
plot_nu = [3.0/(5.0 - 3.0*z) for z in plot_z if 3.0/(5.0 - 3.0*z) < 12.0]
plot_z = plot_z[:len(plot_nu)]
ax.plot(plot_z, plot_nu, color="#c0392b", lw=2.0, label="nu = 3/(5-3z)")
ax.axvline(5.0/3.0, color="#888888", ls="--", lw=1.4)
ax.annotate("z = 5/3 (nu -> inf)", (5.0/3.0, 10.6), fontsize=9.5, ha="right")
for (zn, zd, tag) in ((1,1,"(3/2, 1)"), (2,3,"(1, 2/3)"), (1,3,"(3/4, 1/3)")):
    z = zn/zd
    nu = 3.0/(5.0 - 3.0*z)
    ax.plot([z], [nu], "o", color="#14508c", markersize=7)
    ax.annotate(tag, (z, nu), textcoords="offset points", xytext=(7, 6), fontsize=9.5)
ax.set_xlim(0, 1.9)
ax.set_ylim(0, 12)
ax.set_xlabel("z (dynamic exponent)", fontsize=10.5)
ax.set_ylabel("nu", fontsize=10.5)
ax.set_title("Y2 solution line of alpha = 3/5", fontsize=10.5)
ax.grid(True, ls=":", lw=0.5, alpha=0.6)
ax.legend(fontsize=8.5, loc="upper left")

fig.savefig(IMG)
print("  已保存：", IMG)
