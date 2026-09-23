# -*- coding: utf-8 -*-
"""P8a 前沿剖面最小模型：紫外截断封掉缺口路线，宽度翻译为弛豫率（第 376 条）。

判据驱动（全部为确定性算术，无自由参数）：
  P1 尖锐前沿（w = 1）所需的缺口
  P2 紫外截断下的宽度下限（两种截断读法）
  P3 宽度到前沿弛豫率的换算
  P4 与体系两档速率的对照（格点更新率、记录阻尼观测上界）

运行：python 验证脚本/minimal_verification_front_profile_minimal.py
"""
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "images", "front_profile_minimal.png")

PROD = 106.7 / (2.0 * math.pi)   # 计数式乘积，(Delta/E_P)^2 x w
T_P = 5.391247e-44               # 秒
G_OBS_S = 1.04e-16               # 记录阻尼观测上界，每秒

print("== 设定 ==")
print("  计数式（第 361 条）：(Delta/E_P)^2 x w = %.4f" % PROD)
print("  输入：指数目标 S = e^-106.7；t_P = %.6e s；记录阻尼上界 %.2e s^-1" % (T_P, G_OBS_S))
print()

print("== P1 尖锐前沿所需的缺口 ==")
print("  预期：w = 1 时计数式要求 Delta = sqrt(%.2f) x E_P = %.2f E_P" % (PROD, math.sqrt(PROD)))
print("  读数：w = 1 -> %.2f E_P；w = 2 -> %.2f E_P；w = 4 -> %.2f E_P" % (
    math.sqrt(PROD), math.sqrt(PROD / 2.0), math.sqrt(PROD / 4.0)))
print("  判读：缺口路线要求缺口超过普朗克档，与截断（P2）矛盾，判负")
print()

print("== P2 紫外截断下的宽度下限 ==")
print("  截断（采样定理）：Delta <= E_P/2（两摆程约定）；松读法给出 Delta <= E_P")
for d, tag in ((0.5, "Delta = E_P/2（两摆程）"), (1.0, "Delta = E_P（松读法）")):
    print("  %s -> w >= %.1f 步（缺口取截断上界时宽度取下限）" % (tag, PROD / (d * d)))
print("  判读：缺口路线封死；A 路若成立，宽度须不小于 17 步或 68 步")
print()

print("== P3 宽度翻译为前沿弛豫率 ==")
print("  运动学关系：前沿以每步一格推进（v = c），宽度 w = v/gamma，即 gamma = 1/w 每步")
for w, tag in ((PROD, "w = 17 步（松读法）"), (PROD / 0.25, "w = 68 步（两摆程）")):
    print("  %s -> gamma = %.4f t_P^-1" % (tag, 1.0 / w))
print("  判读：A 路的前沿形成过程须以 0.0147 至 0.0589 t_P^-1 的速率弛豫")
print()

print("== P4 与体系两档速率的对照 ==")
g_hi = 1.0 / PROD
g_lo = 1.0 / (PROD / 0.25)
g_obs = G_OBS_S * T_P
print("  裸格点档（w = 1 的隐含弛豫率）：1 t_P^-1")
print("  所需档：%.4f 至 %.4f t_P^-1（%.2e 至 %.2e s^-1）" % (g_lo, g_hi, g_lo / T_P, g_hi / T_P))
print("  记录阻尼上界（S2G）：%.2e s^-1 = %.1e t_P^-1" % (G_OBS_S, g_obs))
print("  比较一：所需档比裸格点档低 %.2f 个数量级" % math.log10(1.0 / g_hi))
print("  比较二：所需档比记录阻尼上界高 %.1f 个数量级（两个 gamma 的物理身份未建立，只登记位置）" % math.log10(g_hi / g_obs))
print()

print("== 图 ==")
fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), dpi=110)
fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.14, wspace=0.28)

ax = axes[0]
ds = [0.05 * (1.15 ** k) for k in range(0, 40)]
ws = [PROD / (d * d) for d in ds]
ax.loglog(ds, ws, color="#14508c", lw=2.0, label="满足 S = e^-106.7 的（缺口, 宽度）解集")
ax.axvspan(0, 0.5, color="#f6d3cf", alpha=0.6)
ax.annotate("紫外截断内（Delta <= E_P/2）", (0.02, 4e4), fontsize=10, color="#8c1f1f")
ax.axvline(0.5, color="#c0392b", ls="--", lw=1.2)
ax.axvline(1.0, color="#2f7a2f", ls=":", lw=1.2)
ax.annotate("E_P/2", (0.52, 3e3), fontsize=10, color="#c0392b")
ax.annotate("E_P", (1.05, 3e3), fontsize=10, color="#2f7a2f")
ax.plot([0.5], [PROD / 0.25], "o", color="#c0392b")
ax.plot([1.0], [PROD], "o", color="#2f7a2f")
ax.annotate("(E_P/2, 68 步)", (0.53, 90), fontsize=10, color="#c0392b")
ax.annotate("(E_P, 17 步)", (1.05, 24), fontsize=10, color="#2f7a2f")
ax.set_xlabel("交叉缺口 / E_P", fontsize=11)
ax.set_ylabel("所需前沿宽度（步）", fontsize=11)
ax.set_title("(a) 截断后宽度的下限（粉色为截断内）", fontsize=12.5, pad=8)
ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
ax.legend(fontsize=9.5, loc="upper right")

ax = axes[1]
ax.set_xscale("log")
ax.set_xlim(1e-62, 10)
ax.set_ylim(0, 1)
ax.axvline(1.0, color="#7a9cc0", lw=3, label="裸格点档 1 t_P^-1")
ax.axvspan(1.0 / (PROD / 0.25), 1.0 / PROD, color="#fff3d6", label="所需档 0.015 至 0.059")
ax.axvline(g_obs, color="#c0392b", lw=3, label="记录阻尼上界 5.6e-60")
ax.set_yticks([])
ax.set_xlabel("弛豫率（t_P^-1，对数）", fontsize=11)
ax.set_title("(b) 所需弛豫率与已知两档", fontsize=12.5, pad=8)
ax.legend(fontsize=9.5, loc="upper left")
fig.savefig(IMG)
print("  已保存：", IMG)
