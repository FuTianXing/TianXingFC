# -*- coding: utf-8 -*-
"""
跨层事件结构类四条件：合成判据与缺项反证（验证脚本第 450 条）

背景：判据草案「跨层事件的构成条件：十一项条件的判据化」给出 C1 至 C11，首轮取结构类四项
（C1 逐层性、C2 三件输入、C3 双通道、C4 档划分对齐）。本轮为复核轮加缺项反证：
C1、C3、C4 汇总既有读数；C2 补一项新读数（缺项时的后果）。

判据（先立后算，2026-09-27）：
  C1 逐层性：单点锚井共模失败（既有读数：M2 v1）；逐层阵列各层读数可分开。
  C2 三件输入：三项齐备则读数有定义；缺阈值或缺持有时间时读数不立（新增反证）。
  C3 双通道：单通道不可行（既有读数：第 387 条反证）。
  C4 档划分对齐：不同语义类不共档（既有读数：第 403 条穷举，反例为零）。

新增反证的形式：C2a 阈值不足（信号低于阈值）→ 误判率上升；C2b 持有时间不足 → 记录衰减。

理论期望：C2a 在信噪比 1 处误判率约 0.1587、在 5 处约 2.9e-7；C2b 在 tau = 1000 拍时标处
          衰减约 1e-3，在 tau = 0.1 拍时标处衰减约 e^-10。
"""
from math import erf, sqrt, exp

def phi(z):
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def error_rate(snr):
    return phi(-snr)


def decay(tau_over_beat):
    return exp(-1.0 / tau_over_beat)


print("=== 第 450 条：跨层事件结构类四条件 ===")
print("本轮为复核轮加缺项反证：C1、C3、C4 汇总既有读数，C2 补新读数")
print("")

print("--- C1 逐层性（既有读数：M2 v1 与第 383 条）---")
print("单点锚井：共模失败（v1 反证）；逐层阵列：各层指针独立，漏检集在逐层档为空")
print("判定：C1 成立（判据稳健，方向不随实现翻）")
print("")

print("--- C3 双通道（既有读数：第 387 条反证）---")
print("单通道形态不可行；双通道须保持弱于写入（eta_h <= eta_w^2/S^2）")
print("判定：C3 成立（判据稳健）")
print("")

print("--- C4 档划分对齐（既有读数：第 403 条穷举）---")
print("穷举 Bell(9) = 21147 种切法，反例为零；无损切法计数 125 与 780 与 Bell 数一致")
print("判定：C4 成立（判据稳健）")
print("")

print("--- C2a 缺阈值反证（新增）---")
print("信噪比 d/sigma   误判率")
for snr in [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]:
    print("%-16.2f %.6e" % (snr, error_rate(snr)))
print("判定：信噪比不足时误判率显著（1 处 %.4f），阈值条件为读数可分的必要件" % error_rate(1.0))
print("")

print("--- C2b 缺持有时间反证（新增）---")
print("tau / (N_拍 t_P)   一程后的相对残留")
for ratio in [0.1, 1.0, 10.0, 100.0, 1000.0]:
    print("%-19.2f %.6e" % (ratio, decay(ratio)))
print("判定：持有时间不足时记录在拍时标内衰减殆尽（0.1 处残留 %.3e）" % decay(0.1))
print("")

print("--- 合成判定 ---")
print("C1 至 C4 全数成立；其中 C2 的缺项后果由本轮新增反证给出")
print("第二层验收标准的复核：新方程须能反推 C8（持有时间）与 C5（阈值），本表为第一层的当前状态")
print("")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
IMG = "D:/software/TianXingFC/images/structural_conditions.png"

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
snrs = [0.01 * i for i in range(1, 901)]
ax = axes[0]
ax.semilogy(snrs, [max(error_rate(s), 1e-16) for s in snrs])
ax.axvline(1.0, color="tab:red", linestyle=":", label="信噪比 1")
ax.axvline(5.0, color="tab:green", linestyle=":", label="信噪比 5")
ax.set_xlabel("信噪比 d/sigma")
ax.set_ylabel("误判率")
ax.set_title("C2a：缺阈值时的误判率")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

ratios = [10 ** (-1 + 0.01 * i) for i in range(0, 401)]
ax = axes[1]
ax.loglog(ratios, [max(decay(r), 1e-300) for r in ratios])
ax.axhline(1e-3, color="gray", linestyle=":", label="残留 1e-3")
ax.set_xlabel("tau / (N_拍 t_P)")
ax.set_ylabel("一程后相对残留")
ax.set_title("C2b：缺持有时间时的记录衰减")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

ax = axes[2]
labels = ["C1 逐层性", "C2 三件输入", "C3 双通道", "C4 档对齐"]
values = [1, 1, 1, 1]
bars = ax.barh(labels, values, color="tab:blue")
for b, v in zip(bars, values):
    ax.text(v * 0.5, b.get_y() + b.get_height() / 2, "成立", va="center", ha="center", color="white")
ax.set_xlim(0, 1.2)
ax.set_xlabel("判定（1 = 成立）")
ax.set_title("结构类四条件合成本轮状态")
ax.grid(alpha=0.3, axis="x")

plt.tight_layout()
plt.savefig(IMG, dpi=150)
print("图已保存：" + IMG)
