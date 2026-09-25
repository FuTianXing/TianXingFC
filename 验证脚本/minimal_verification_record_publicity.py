# -*- coding: utf-8 -*-
"""
记录的公共性：读出反作用与复读精度预算（验证脚本第 412 条，探针第三层）

背景：探针词条第三节给出「记录的公共性」——记录写在格点上且不衰减，故可重复读取，任何与该格点耦合的仪器都读到同一份。该层为辨析推论，尚无独立验证。本条给它一组可算判据。

判据（先立后算，2026-09-25）：
  P1 单次读出精度：精度按 eta^(-1/2) 标度（在标准量子测量的线性响应区内）。
  P2 累积读出：重复 N 次，精度按 N^(-1/2) 提高。
  P3 权衡不变量：总反作用与精度的平方之积不依赖 eta 与 N。
  P4 复读条件：给定扰动预算，可读精度有下限；该下限的显式形式由 P3 给出。

模型：记录值 D 在一次写入后固定。读出取弱测量模型的最小形式——
  单次读出噪声 sigma = sigma0 / sqrt(eta)，单次反作用 dD = kappa * eta；
  N 次读出给出精度 sigma / sqrt(N) 与总反作用 N * kappa * eta。
本模型取标准量子测量的权衡关系，体系侧的内容是把该关系接到记录的复读条件上。

理论期望：P1 标度斜率 -1/2；P2 斜率 -1/2；P3 乘积恒为 kappa * sigma0^2；
          P4 精度下限 = sqrt(kappa * sigma0^2 / Delta_max)。
"""
from math import sqrt, log10

SIGMA0 = 1.0
KAPPA = 1.0
ETA_LIST = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
N_LIST = [1, 10, 100, 1000, 10000]


def sigma(eta, n):
    return SIGMA0 / sqrt(eta) / sqrt(n)


def backaction(eta, n):
    return KAPPA * eta * n


def product(eta, n):
    return backaction(eta, n) * sigma(eta, n) ** 2


print("=== 第 412 条：记录的公共性（读出反作用与复读精度预算）===")
print("模型：单次读出噪声 sigma0/sqrt(eta)，单次反作用 kappa*eta，重复 N 次")
print("参数：sigma0 = %.3f，kappa = %.3f" % (SIGMA0, KAPPA))
print("")

print("--- P1 单次读出精度（N = 1）---")
print("eta        精度          反作用        乘积")
rows = []
for e in ETA_LIST:
    s = sigma(e, 1)
    b = backaction(e, 1)
    pr = product(e, 1)
    rows.append((e, s, b, pr))
    print("%-10.1e %-13.6e %-13.6e %-13.6e" % (e, s, b, pr))
print("")

print("--- P2 累积读出（eta = 1e-2）---")
print("N          精度          反作用        乘积")
rows2 = []
for n in N_LIST:
    s = sigma(1e-2, n)
    b = backaction(1e-2, n)
    pr = product(1e-2, n)
    rows2.append((n, s, b, pr))
    print("%-10d %-13.6e %-13.6e %-13.6e" % (n, s, b, pr))
print("")

p1_slope = (log10(rows[-1][1]) - log10(rows[0][1])) / (log10(rows[-1][0]) - log10(rows[0][0]))
p2_slope = (log10(rows2[-1][1]) - log10(rows2[0][1])) / (log10(rows2[-1][0]) - log10(rows2[0][0]))
prods = [r[3] for r in rows] + [r[3] for r in rows2]
spread = max(prods) - min(prods)
print("--- P3 权衡不变量 ---")
print("P1 标度斜率 = %.6f（期望 -0.5）" % p1_slope)
print("P2 标度斜率 = %.6f（期望 -0.5）" % p2_slope)
print("乘积的最大最小差 = %.3e（期望 0），常数 = %.6f" % (spread, KAPPA * SIGMA0 ** 2))
print("")

print("--- P4 复读条件（扰动预算给精度下限）---")
print("扰动预算      精度下限")
for dmax in [1e-1, 1e-2, 1e-3, 1e-4]:
    print("%-13.1e %.6e" % (dmax, sqrt(KAPPA * SIGMA0 ** 2 / dmax)))
print("")

print("--- 一致性的判据化形态（两仪器读同一记录）---")
print("两仪器各自的精度为 sigma_A 与 sigma_B，读数之差的标准差为 sqrt(sigma_A^2 + sigma_B^2)")
for ea, eb, n in [(1e-2, 1e-2, 100), (1e-2, 1e-3, 100), (1e-3, 1e-4, 1000)]:
    sa, sb = sigma(ea, n), sigma(eb, n)
    print("eta_A = %.0e, eta_B = %.0e, N = %d：差的标准差 = %.6e" % (ea, eb, n, sqrt(sa ** 2 + sb ** 2)))
print("")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
IMG = "D:/software/TianXingFC/images/record_publicity.png"

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
ax = axes[0]
for n, mk in [(1, "o-"), (10, "s--"), (100, "^:")]:
    ax.loglog(ETA_LIST, [sigma(e, n) for e in ETA_LIST], mk, label="N = %d" % n)
ax.set_xlabel("耦合率 eta")
ax.set_ylabel("读出精度")
ax.set_title("P1/P2：精度随耦合率与读次数")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

ax = axes[1]
for n, mk in [(1, "o-"), (10, "s--"), (100, "^:")]:
    ax.loglog(ETA_LIST, [backaction(e, n) for e in ETA_LIST], mk, label="N = %d" % n)
ax.set_xlabel("耦合率 eta")
ax.set_ylabel("总反作用")
ax.set_title("反作用随耦合率与读次数")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

ax = axes[2]
etas = [10 ** (-4 + 0.05 * i) for i in range(0, 61)]
for n, mk in [(1, "o-"), (10, "s--"), (100, "^:")]:
    ax.semilogx(etas, [product(e, n) for e in etas], mk, label="N = %d" % n)
ax.axhline(KAPPA * SIGMA0 ** 2, color="gray", linestyle=":", label="常数 %.2f" % (KAPPA * SIGMA0 ** 2))
ax.set_xlabel("耦合率 eta")
ax.set_ylabel("总反作用 x 精度^2")
ax.set_title("P3：权衡不变量")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(IMG, dpi=150)
print("图已保存：" + IMG)
