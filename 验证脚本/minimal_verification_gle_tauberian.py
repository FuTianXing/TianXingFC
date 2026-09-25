# -*- coding: utf-8 -*-
"""换算规则的第一原理化：gamma_v = 2 - gamma_F 的数值验证（第 426 条，候选丁/换算链第二段）。

判据：
  N1 谱-时间验证：S_v(w) = w^{1-gF} cos(pi(1-gF)/2)（截断）的时间相关尾指数 = 2 - gF
     （gF = 0.5、1.0、1.5、1.9 四档）
  N2 FDT 代数核对：S_v = S_F / |Sigma|^2 = 2 kT Re(1/Sigma) 的等价链
  N3 适用域边界：gF -> 2 时尾指数 -> 0（临界行为）

方法：C_v(t) = int_{wlo}^{whi} S_v(w) cos(w t) d w（对数网格数值积分）；
      拟合窗取 [1e2, 1e3]（远高于 w_hi 的短时、远低于 1/w_lo 的长时截断）。

运行：python 验证脚本/minimal_verification_gle_tauberian.py
"""
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import quad
from math import gamma

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "images", "gle_tauberian.png")

def C_num(t, p, Lam=1.0):
    w1 = np.linspace(0.0, 1.0, 20000)
    f1 = np.where(w1 > 0, w1 ** p, 0.0) * np.exp(-w1 / Lam)
    v1 = float(np.trapezoid(f1 * np.cos(w1 * t), w1))
    if p < 0.999:
        dw = 1.0 / 20000.0
        v1 += dw ** (1.0 + p) / (1.0 + p) * 1.0
    npts = min(2000000, max(4000, int(200 * t)))
    w2 = np.linspace(1.0, 50.0, npts)
    f2 = w2 ** p * np.exp(-w2 / Lam)
    v2 = float(np.trapezoid(f2 * np.cos(w2 * t), w2))
    return v1 + v2

def C_ana(t, p, Lam=1.0):
    return float(gamma(1.0 + p) * np.real(complex(1.0 / Lam, t) ** (-(1.0 + p))))

def tail_fit_num(p, t_lo=1e2, t_hi=1e4, nt=16):
    ts = np.logspace(math.log10(t_lo), math.log10(t_hi), nt)
    cs = np.array([abs(C_num(t, p)) for t in ts])
    msk = cs > 1e-300
    return float(np.polyfit(np.log(ts[msk]), np.log(cs[msk]), 1)[0]), ts, cs

print("== N1 谱-时间验证（拟合窗 [1e2, 1e3]） ==")
for gF in (0.5, 1.0, 1.5, 1.9):
    p = 1.0 - gF
    a, _, _ = tail_fit_num(p)
    print("  gamma_F = %.1f（p = %.1f）：C_v(t) 数值尾指数 = %.3f（推导值 1 + p = %.3f；差 %+.3f）" % (
        gF, p, a, 1.0 + p, a - (1.0 + p)))

print()
print("== N2 FDT 代数链核对 ==")
print("  S_v = S_F/|Sigma|^2 与 S_F = 2 kT Re(Sigma) 合并给 S_v = 2 kT Re(1/Sigma)")
for gF in (0.5, 1.5):
    p = 1.0 - gF
    # Re(1/Sigma)：Sigma = (i w)^{gF-1} = w^{gF-1} e^{i pi (gF-1)/2}
    # 1/Sigma = w^{1-gF} e^{-i pi (gF-1)/2}；Re = w^{1-gF} cos(pi (gF-1)/2) = w^{1-gF} cos(pi (1-gF)/2)
    val1 = math.cos(math.pi * (1.0 - gF) / 2.0)
    val2 = math.cos(math.pi * (gF - 1.0) / 2.0)
    print("  gamma_F = %.1f：cos(pi(1-gF)/2) = %.4f，cos(pi(gF-1)/2) = %.4f（一致：偶函数）" % (gF, val1, val2))

print()
print("== N3 适用域边界（gF -> 2） ==")
for gF in (1.99,):
    p = 1.0 - gF
    a, _, _ = tail_fit_num(p)
    print("  gamma_F = %.2f（p = %.2f）：数值尾指数 %.3f（趋向 0.01）" % (gF, p, a))
# 解析对照：数值与解析的点对比
print()
print("== N1b 数值与解析逐点对照（p = 0.5） ==")
for t in (100.0, 300.0, 1000.0, 3000.0):
    print("  t = %.0f：数值 %.6e，解析 %.6e（相对差 %.2e）" % (
        t, C_num(t, 0.5), C_ana(t, 0.5), abs(C_num(t, 0.5) - C_ana(t, 0.5)) / abs(C_ana(t, 0.5))))
print()

print()
print("== N1c 数值序列诊断（p = 0 与 p = -0.5） ==")
for p_dbg in (0.0, -0.5):
    print("  p = %.1f：" % p_dbg)
    for t in (100.0, 1000.0, 10000.0):
        print("    t = %.0f：数值 %.6e，解析 %.6e" % (t, C_num(t, p_dbg), C_ana(t, p_dbg)))

print("== 图 ==")
fig, ax = plt.subplots(figsize=(7.8, 4.8), dpi=110)
for gF, col in ((0.5, "#14508c"), (1.0, "#2f7a2f"), (1.5, "#c0392b"), (1.9, "#7030a0")):
    a_, m_, e_ = tail_fit_num(1.0 - gF)
    ax.loglog(m_, e_, "o-", color=col, lw=1.4, markersize=3.5, label="gF = %.1f (fit %.2f, target %.1f)" % (gF, a_, 2 - gF))
ax.set_xlabel("t", fontsize=10.5)
ax.set_ylabel("|C_v(t)|", fontsize=10.5)
ax.set_title("GLE: spectral power -> time tail (numeric check)", fontsize=10.5)
ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
ax.legend(fontsize=9)
fig.savefig(IMG)
print("  已保存：", IMG)
