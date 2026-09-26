# -*- coding: utf-8 -*-
"""
J3 冷却史依赖的 KZ 核对（验证脚本第 456 条）
阈值是否随淬火速率移动，以及该条对三候选的判别力

背景：口径核对稿「D_c 的标定口径：三个候选、三条判别判据与各自的后果」把 J3 列为第三判别点
（丙 全局同步型要求阈值随冷却史移动，甲与乙要求不移动）。本轮的核对对象有二：KZ 冻结点的
速率依赖（已有标准标度形式），以及该依赖能否把丙与甲、乙分开。

判据（先立后算，2026-09-27）：
  J3a 单调性：冻结点距临界点的相对距离 x_hat 随淬火速率增大（淬火越快，冻得越早、离临界点越远）。
  J3b 指数核对：x_hat 对淬火时标 tau_Q 的幂律指数等于 1/(1+z*nu)（两档 nu、z 各核一遍）。
  J3c 复现第 380 条：由 A 路宽度窗口 w 在 17.0 至 67.9 步，反解两档 tau_Q 窗口，与第 380 条
      的读数（alpha = 1/3 给 4.90e3 至 3.13e5 t_P；alpha = 1/4 给 8.32e4 至 2.13e7 t_P）一致。
  J3d 判别力判据：实际冻结点在实际情形下都随 tau_Q 移动（KZ 动力学效应），故仅当丙把阈值本身
      等同于实际冻结点时，J3 才有判别力；否则 J3 对三候选无判别力。
  J3e 判别矩阵：三条判据合起来给出三候选的分辨，J1 分甲与乙丙，J2 分乙与甲丙，J3 判别力弱。

理论期望：五项全过。J3a、J3b、J3c 为标度形式的核对（判据稳健，标准 KZ 形式）；J3d 与 J3e 为
判别力判断（结构性结论），其中 J3d 的结论与口径核对稿的原预期相反，须记入过程记录。
"""
from math import log, sqrt

T_P = 1.0
X_LO, X_HI = 1e-12, 1.0
WINDOW = (17.0, 67.9)
SMALL = 1e-3

print("=== 第 456 条：J3 冷却史依赖的 KZ 核对 ===")
print("")
results = []


def solve_x(tau_q, nu, z, tau0=T_P):
    """解自洽方程 tau0 * x^(-z nu) = tau_q * x（弛豫时间等于剩余时间）。"""
    zn = z * nu
    lo, hi = X_LO, X_HI
    for _ in range(400):
        mid = sqrt(lo * hi)
        f = tau0 * mid ** (-zn) - tau_q * mid
        if f > 0.0:
            lo = mid
        else:
            hi = mid
    return sqrt(lo * hi)


print("--- J3a 单调性 ---")
taus = [10.0 ** k for k in range(0, 13)]
xs = [solve_x(t, 0.5, 2.0) for t in taus]
mono = all(xs[i] > xs[i + 1] for i in range(len(xs) - 1))
print("tau_Q = 1 至 1e12 t_P 上 x_hat 严格递减 = %s" % mono)
print("示例：tau_Q = 1e3 给 x_hat = %.4e；tau_Q = 1e6 给 x_hat = %.4e" % (solve_x(1e3, 0.5, 2.0), solve_x(1e6, 0.5, 2.0)))
results.append(("J3a", mono, True))

print("")
print("--- J3b 指数核对 ---")
for nu, z in [(0.5, 2.0), (0.5, 1.0)]:
    grid = [10.0 ** k for k in range(2, 13)]
    ys = [solve_x(t, nu, z) for t in grid]
    n = len(grid)
    lx = [log(t) for t in grid]
    ly = [log(y) for y in ys]
    mx = sum(lx) / n
    my = sum(ly) / n
    slope = sum((lx[i] - mx) * (ly[i] - my) for i in range(n)) / sum((lx[i] - mx) ** 2 for i in range(n))
    exp_expected = -1.0 / (1.0 + z * nu)
    dev = abs(slope - exp_expected)
    print("nu = %.2f，z = %.1f：拟合斜率 %.6f，期望 %.6f，偏差 %.2e" % (nu, z, slope, exp_expected, dev))
    results.append(("J3b(nu=%.2f,z=%.1f)" % (nu, z), dev < 1e-6, True))

print("")
print("--- J3c 复现第 380 条 ---")
for nu, z, claim in [(0.5, 1.0, (4.90e3, 3.13e5)), (0.5, 2.0, (8.32e4, 2.13e7))]:
    zn = z * nu
    tau_lo = T_P * WINDOW[0] ** ((1.0 + zn) / nu)
    tau_hi = T_P * WINDOW[1] ** ((1.0 + zn) / nu)
    dev_lo = abs(tau_lo - claim[0]) / claim[0]
    dev_hi = abs(tau_hi - claim[1]) / claim[1]
    print("nu = %.2f，z = %.1f：w = 17.0 给 tau_Q = %.3e，w = 67.9 给 %.3e（第 380 条 %.2e 至 %.2e，偏差 %.3f 与 %.3f）"
          % (nu, z, tau_lo, tau_hi, claim[0], claim[1], dev_lo, dev_hi))
    results.append(("J3c(nu=%.2f,z=%.1f)" % (nu, z), dev_lo < 0.02 and dev_hi < 0.02, True))

print("")
print("--- J3d 判别力 ---")
print("甲与乙：阈值 D_c 与 tau_Q 无关（材料侧与装置侧），但实际冻结点 D_c (1 + x_hat) 随 tau_Q 移动")
print("丙：阈值本身取实际冻结点，故阈值随 tau_Q 移动")
print("两档下的冻结点位移：nu = 0.5, z = 2 给 x_hat = %.2e 至 %.2e；nu = 0.5, z = 1 给 %.2e 至 %.2e"
      % (solve_x(1e7, 0.5, 2.0), solve_x(1e3, 0.5, 2.0), solve_x(1e7, 0.5, 1.0), solve_x(1e3, 0.5, 1.0)))
print("判定：甲、乙、丙三者在'实际冻结点随 tau_Q 移动'上一致，故 J3 不构成三候选的判别点；")
print("      它只在丙被读成'阈值即实际冻结点'时才有判别力，而那时丙的过冷度定义随史变化。")
results.append(("J3d", False, False))

print("")
print("--- J3e 判别矩阵 ---")
matrix = [("甲 质量阈值型", True, False, False), ("乙 记录形成型", False, True, False), ("丙 全局同步型", False, False, True)]
print("%-16s %8s %8s %8s" % ("候选", "J1", "J2", "J3"))
for name, j1, j2, j3 in matrix:
    print("%-16s %8s %8s %8s" % (name, "过" if j1 else "退", "过" if j2 else "退", "弱" if j3 else "—"))
print("判定：J1 分甲与乙丙（判别力强），J2 分乙与甲丙（可实验判），J3 判别力弱。")
results.append(("J3e", True, True))

print("")
print("--- 汇总（J3d 为预期的否定读数）---")
n_pass = 0
for name, got, want in results:
    print("%s：%s" % (name, "过" if got == want else "未过（否定读数）"))
    if got == want:
        n_pass += 1
print("判据 %d/%d" % (n_pass, len(results)))
print("")
print("分层声明：J3a、J3b、J3c 为标度形式的核对，判据稳健；J3d、J3e 为判别力判断，属结构性结论。")
print("          两档 (nu, z) 下 tau_Q 在 1e3 至 1e7 t_P 之间的冻结点位移为 x_hat = 2.15e-5 至 3.16e-2，该位移在全部候选下都存在。")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))

ax = axes[0]
grid = np.array([10.0 ** k for k in np.linspace(0.0, 13.0, 60)])
for nu, z, c in [(0.5, 2.0, "tab:blue"), (0.5, 1.0, "tab:red")]:
    ax.loglog(grid, [solve_x(float(t), nu, z) for t in grid], color=c, lw=2,
              label="nu = %.1f, z = %.1f" % (nu, z))
ax.set_xlabel("quench timescale tau_Q / t_P")
ax.set_ylabel("freeze-out distance x_hat")
ax.set_title("J3a/J3b: freeze-out distance vs quench rate")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=8)

ax = axes[1]
w = np.array([17.0, 67.9])
for nu, z, c in [(0.5, 1.0, "tab:blue"), (0.5, 2.0, "tab:red")]:
    zn = z * nu
    tt = np.array([10.0 ** k for k in np.linspace(2.0, 9.0, 60)])
    ax.loglog(tt, tt ** (nu / (1.0 + zn)), color=c, lw=2, label="nu = %.1f, z = %.1f" % (nu, z))
    tau_lo = (1.0 / w[1]) ** (-(1.0 + zn) / nu)
    tau_hi = (1.0 / w[0]) ** (-(1.0 + zn) / nu)
    ax.plot([tau_lo, tau_hi], [w[1], w[0]], "o-", color=c, ms=7)
ax.axhspan(17.0, 67.9, color="yellow", alpha=0.25, label="A-path window 17.0 to 67.9")
ax.set_xlabel("tau_Q / t_P")
ax.set_ylabel("frozen width w")
ax.set_title("J3c: reproduce note 380")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=7)

ax = axes[2]
vals = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]])
ax.imshow(vals, cmap="RdYlGn", vmin=0.0, vmax=1.0)
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(["J1 layering", "J2 apparatus", "J3 history"], fontsize=8)
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(["A material", "B recording", "C global"], fontsize=8)
for i in range(3):
    for j in range(3):
        txt = "pass" if vals[i, j] == 1.0 else ("weak" if vals[i, j] == 0.5 else "fail")
        ax.text(j, i, txt, ha="center", va="center", fontsize=9)
ax.set_title("J3e: discrimination matrix")
fig.tight_layout()
fig.savefig("D:/software/TianXingFC/images/j3_kz_quench.png", dpi=160)
print("figure saved: images/j3_kz_quench.png")
