# -*- coding: utf-8 -*-
"""
两仪器同时读同一记录的相互干扰（验证脚本第 444 条）

背景：第 412 条给「记录可被多处反复读取」一组判据，其注记第六节声明未验「多次读出之间互不干扰」，
并给出验证方式：两通道同时耦合同一寄存器，检验各自反作用是否仍按 kappa*eta 标度而无交叉项。本条执行该验证。

判据（先立后算，2026-09-27）：
  Q1 不同指针时可加：两通道反作用之和等于各自反作用之和，交叉项为零。
  Q2 同一指针时有交叉项：总反作用含 2*eta_A*eta_B*sigma^2 一项。
  Q3 交叉项占比：2*eta_A*eta_B/(eta_A^2+eta_B^2)，在 eta_A = eta_B 时取 1（等于主体反作用）。
  Q4 设计条件：读出通道须耦合到不同指针自由度，公共性方可落在弱耦合区。

模型：记录寄存器取位置变量 x，读取出 von Neumann 弱测量，耦合 H = eta * x * p_ptr；
  系统动量受反作用 p -> p + eta * p_ptr，方差的增量为 eta^2 * sigma_ptr^2（单位质量）。
  两通道同时作用时，指针是否同一自由度决定交叉项的有无。

理论期望：Q1 给出 eta_A^2 + eta_B^2 倍 sigma^2；Q2 给出 (eta_A+eta_B)^2 倍 sigma^2；
          Q3 的占比为 2*eta_A*eta_B/(eta_A^2+eta_B^2)，对称点在 1。
"""
import numpy as np

SIGMA_PTR = 1.0


def backaction(etas, same_pointer):
    if same_pointer:
        return float(sum(etas)) ** 2 * SIGMA_PTR ** 2
    return float(sum(e ** 2 for e in etas)) * SIGMA_PTR ** 2


def crosstalk(etas):
    return backaction(etas, True) - backaction(etas, False)


def crosstalk_ratio(a, b):
    base = a ** 2 + b ** 2
    return 0.0 if base == 0 else 2 * a * b / base


print("=== 第 444 条：两仪器同时读同一记录的相互干扰 ===")
print("模型：von Neumann 弱测量，反作用增量 = eta^2 * sigma_ptr^2（单位质量），sigma_ptr = %.3f" % SIGMA_PTR)
print("")

print("--- Q1 / Q2：可加性与交叉项（eta_A = 0.1 固定）---")
print("eta_B     不同指针      同一指针      交叉项        占比")
rows = []
for b in [0.00, 0.05, 0.10, 0.20, 0.40, 0.80]:
    etas = [0.1, b]
    sep = backaction(etas, False)
    same = backaction(etas, True)
    cross = crosstalk(etas)
    ratio = crosstalk_ratio(0.1, b)
    rows.append((b, sep, same, cross, ratio))
    print("%-9.2f %-13.6e %-13.6e %-13.6e %.4f" % (b, sep, same, cross, ratio))
print("")

print("--- Q3：占比的对称点 ---")
print("eta_A = eta_B = 0.2 时占比 = %.6f（期望 1.0）" % crosstalk_ratio(0.2, 0.2))
print("eta_B 趋于 0 时占比趋于 %.6f（期望 0）" % crosstalk_ratio(0.2, 1e-12))
print("")

print("--- Q4：设计条件的定量形态 ---")
print("若要求交叉项不超过主体反作用的 5%，则两通道耦合率之比须满足：")
for target in [0.05, 0.10, 0.20]:
    lo, hi = 1e-6, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if crosstalk_ratio(1.0, mid) < target:
            lo = mid
        else:
            hi = mid
    print("  占比 < %.2f：eta_B/eta_A < %.6f（或 > %.6f）" % (target, lo, 1.0 / lo))
print("")


def mc_backaction(a, b, same, n=400000, seed=20260927):
    rng = np.random.default_rng(seed)
    if same:
        ptr = rng.normal(0.0, SIGMA_PTR, size=n)
        dp = (a + b) * ptr
    else:
        dp = a * rng.normal(0.0, SIGMA_PTR, size=n) + b * rng.normal(0.0, SIGMA_PTR, size=n)
    return float(np.var(dp))


print("--- 蒙特卡洛交叉验证（M = 400000，种子 20260927）---")
for a, b in [(0.1, 0.1), (0.1, 0.3)]:
    print("eta_A = %.2f, eta_B = %.2f：不同指针 采样 %.6e 解析 %.6e；同一指针 采样 %.6e 解析 %.6e" % (
        a, b, mc_backaction(a, b, False), backaction([a, b], False),
        mc_backaction(a, b, True), backaction([a, b], True)))
print("")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
IMG = "D:/software/TianXingFC/images/dual_readout_crosstalk.png"

bs = np.linspace(0.0, 1.0, 200)
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
ax = axes[0]
ax.plot(bs, [backaction([0.1, b], False) for b in bs], label="不同指针（可加）")
ax.plot(bs, [backaction([0.1, b], True) for b in bs], label="同一指针（含交叉）")
ax.set_xlabel("eta_B（eta_A = 0.1）")
ax.set_ylabel("反作用增量")
ax.set_title("Q1/Q2：两种情形的反作用")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(bs, [crosstalk([0.1, b]) for b in bs], color="tab:red")
ax.set_xlabel("eta_B（eta_A = 0.1）")
ax.set_ylabel("交叉项")
ax.set_title("Q2：交叉项随耦合率")
ax.grid(alpha=0.3)

ax = axes[2]
ratios = [crosstalk_ratio(1.0, b) for b in np.linspace(1e-4, 1.0, 300)]
ax.semilogx(np.linspace(1e-4, 1.0, 300), ratios)
ax.axhline(1.0, color="gray", linestyle=":", label="对称点占比 = 1")
ax.set_xlabel("eta_B/eta_A")
ax.set_ylabel("交叉项占比")
ax.set_title("Q3：占比的对称性")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(IMG, dpi=150)
print("图已保存：" + IMG)
