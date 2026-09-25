# -*- coding: utf-8 -*-
"""q = 2/5 的来源清点与首轮判定：稀释继承下界与校准保护分数（第 417 条，P0-5 候选机制五轮）。

判据：
  P1 稀释继承类：速度线性继承「随膨胀衰减量」的 q 值集合与下界
  P2 共动膨胀读法：q = 膨胀幂指数（辐射 1/2）与创生指数型的适用性
  P3 校准衰减自由幂：q = m * s 的条件曲线（q = 2/5 的目标）
  P4 锁定时间增长：q = m（k ∝ t^m），待建口径
  P5 部分保护混合：q = w * q_dil 的保护分数

运行：python 验证脚本/minimal_verification_q_source_audit.py
"""
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "images", "q_source_audit.png")

Q_LO, Q_HI = 0.3985, 0.4402   # 允许带在 q 口径的窗口（第 416 条）
Q_TARGET = 2.0 / 5.0
N_RAD = 0.5                    # 辐射期膨胀幂指数 a ∝ t^{1/2}

print("== P1 稀释继承类（速度 ∝ 衰减量 X，X ∝ t^{-m}） ==")
props = [
    ("共动波数 k ∝ 1/a", 0.5),
    ("场振幅 ∝ 1/a（红移）", 0.5),
    ("温度 ∝ 1/a", 0.5),
    ("数密度 ∝ a^{-3}", 1.5),
    ("能量密度 ∝ a^{-4}", 2.0),
]
qs = [m for _, m in props]
for name, m in props:
    q = m
    flag = "带内" if Q_LO <= q <= Q_HI else "带外"
    print("  %s：m = %.4f，q = %.4f，%s" % (name, m, q, flag))
print("  稀释继承 q 值集合：{%s}；下界 = %.4f" % (", ".join("%.4f" % v for v in sorted(set(qs))), min(qs)))
print("  靶 q = 2/5 = %.4f，比下界小 %.4f（低 %.1f%%）" % (Q_TARGET, min(qs) - Q_TARGET, (min(qs)-Q_TARGET)/min(qs)*100))
print()

print("== P2 共动膨胀读法 ==")
print("  共动推进指数 q = 1 - n；辐射期 n = %.2f -> q = %.4f（与稀释继承下界一致），带外" % (N_RAD, 1.0 - N_RAD))
print("  创生膨胀为指数型（R(t) = (R0 + c/H0) e^{H0 t} - c/H0），不产生幂律 n 与幂律 q，判「不适用」")
print()

print("== P3 校准衰减自由幂（v ∝ A^s，A ∝ t^{-m}，q = m s） ==")
for s in (1.0, 0.5, 2.0):
    m = Q_TARGET / s
    print("  响应指数 s = %.1f：需 m = %.4f（A ∝ t^{-%.4f}）" % (s, m, m))
print("  对照：纯稀释给出的最小 m 为 0.5（1/a 类），大于所需 m（s = 1 时 0.4），故纯稀释不足")
print("  即：需要有效衰减指数 m_eff < 0.5，慢于任何单次稀释继承")
print()

print("== P4 锁定时间增长（每格锁定拍数 k(t) ∝ t^m，q = m） ==")
print("  靶条件：k(t) ∝ t^{2/5}（每格锁定时间按 t^{0.4} 增长）")
print("  状态：无框架内机制读数，待建（口径登记）")
print()

print("== P5 部分保护混合（q = w q_dil + (1-w) q_src，q_src = 0） ==")
for q_dil in (0.5, 1.0):
    w = Q_TARGET / q_dil
    print("  若稀释分量 q_dil = %.2f：w = %.4f，保护分数 1 - w = %.4f" % (q_dil, w, 1.0 - w))
print("  取最小稀释（q_dil = 0.5，1/a 类）时保护分数为 1/5（形式读数，不称机制）")
print()

print("== 图 ==")
fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6), dpi=110)
fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.16, wspace=0.24)

ax = axes[0]
ax.axvspan(Q_LO, Q_HI, color="#ffe9b0", alpha=0.9, label="allowed band (q)")
marks = [
    ("dilution family (1/2, 1, 3/2, 2)", [0.5, 1.0, 1.5, 2.0], "#14508c"),
    ("target 2/5", [0.4], "#2f7a2f"),
    ("chase / comoving (1/2)", [0.5], "#7030a0"),
    ("KZ (2/3)", [2.0/3.0], "#888888"),
]
for name, xs, col in marks:
    ax.plot(xs, [0.5]*len(xs), "o", color=col, markersize=8, label=name)
ax.axvline(0.5, color="#aaaacc", ls=":", lw=1.4)
ax.annotate("dilution lower bound q = 1/2", (0.5, 0.72), fontsize=9.5, ha="center", color="#14508c")
ax.set_xlim(0.25, 2.15)
ax.set_ylim(0, 1)
ax.set_yticks([])
ax.set_xlabel("q", fontsize=10.5)
ax.set_title("P1 candidates on the q axis", fontsize=10.5)
ax.legend(fontsize=8.5, loc="upper right")

ax = axes[1]
ws = [i/200.0 for i in range(201)]
ax.plot(ws, [w*0.5 for w in ws], color="#c0392b", lw=2.0, label="q = w * (1/2)")
ax.axhspan(Q_LO, Q_HI, color="#ffe9b0", alpha=0.9, label="allowed band (q)")
ax.plot([0.8], [0.4], "o", color="#2f7a2f", markersize=8)
ax.annotate("w = 4/5 (protection 1/5)", (0.8, 0.4), textcoords="offset points", xytext=(-140, 12), fontsize=9.5, color="#2f7a2f")
ax.set_xlim(0, 1.0)
ax.set_ylim(0, 0.55)
ax.set_xlabel("dilution weight w", fontsize=10.5)
ax.set_ylabel("q", fontsize=10.5)
ax.set_title("P5 mixed attenuation: q = w * q_dil", fontsize=10.5)
ax.grid(True, ls=":", lw=0.5, alpha=0.6)
ax.legend(fontsize=8.5, loc="upper left")

fig.savefig(IMG)
print("  已保存：", IMG)
