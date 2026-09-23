# -*- coding: utf-8 -*-
"""KZ 适用性判定：临界冻结不适用、注入-接缝机制与畴尺度候选（第 381 条）。

判据（材料清点 + 确定性算术）：
  V1 KZ 前提清点与三条反证（零动力不动点、交错无吸引性、写入型有序化）
  V2 体系机制对照（注入 + 多核 + 波前 + 接缝；间距与宽度的概念区分）
  V3 候选畴标度的重算（KZ 幂律两档、追赶变体、视界线性）
  V4 A 路窗口的候选对照（含缺陷构型宽度）

运行：python 验证脚本/minimal_verification_kz_applicability.py
"""
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "images", "kz_applicability.png")

PROD = 106.7 / (2.0 * math.pi)   # 计数式乘积 16.9818
W_LO = PROD                       # 17.0 步
W_HI = 4.0 * PROD                 # 67.9 步
T_LO = 10.0 ** -39.4 / 5.391247e-44   # P4-27 下端，t_P
T_HI = 10.0 ** -34.5 / 5.391247e-44   # P4-27 上端，t_P
W_DEF = 3.80e21                   # 缺陷构型宽度，l_P
E_P_EV = 1.220890e28              # eV

print("== 设定 ==")
print("  计数式：(Delta/E_P)^2 x w = %.4f；A 路窗口 [%.1f, %.1f] 步" % (PROD, W_LO, W_HI))
print("  时标档：P4-27 [%.2e, %.2e] t_P" % (T_LO, T_HI))
print("  候选标度：KZ 幂律 tau_Q^alpha、追赶变体 sqrt(tau_Q/(2 t_P))、视界线性 c tau_Q、缺陷构型宽度 %.2e l_P" % W_DEF)
print()

print("== V1 KZ 前提清点（材料判定） ==")
print("  前提一（连续相变临界点）：不成立。零动力不动点 (D,O,G) = (0,0,0)：原点处所有项系数为零，")
print("    物侧动力学完全静止，离开原点唯一机制是种子注入 f_tx（《从混沌到兴盛》§1.3、§2.1）。")
print("  前提二（临界减速）：不成立。原点给出的是完全静止，不是弛豫时间的幂律发散；")
print("    校准通道线性化谱为 kappa k^2，任意 kappa > 0 稳定、无阈值（正稿第四节）。")
print("  前提三（无序态被动力学吸引到有序）：不成立。交错基态无吸引性（数值判定：一般初值 180 次碰撞")
print("    对相锁值 120 次，末段仍带同向畴）；反相由创生规则写入、由校准维持（正稿第三节）。")
print("  判定：标准 KZ 临界冻结的前提三条全不成立，不适用。")
print()

print("== V2 体系机制对照 ==")
print("  实际机制：种子注入 + 多核独立成核 + 波前以 c 扩张 + 相遇接缝凝固")
print("    （《粒子与时空的同生》§3.3；多核成案 2026-09-08）。")
print("  分层：Kibble 因果论证适用（体系内定案）；Zurek 淬火标度律标准形式不适用（V1）。")
print("  概念区分：上述机制给出接缝间距（畴尺度）；A 路的 w 是波前过渡区宽度（前沿扫过交叉区的时间）。")
print("  两者在标准 KZ 里同为一尺度 xi_freeze；在无临界减速的体系里分离，这一分离是本轮的核心读数。")
print()

print("== V3 候选畴标度的重算 ==")
w_kz13 = T_LO ** (1.0 / 3.0)
print("  候选 A（KZ 幂律，若强行沿用）：")
print("    tau_Q = %.2e t_P -> w = %.2f 步（alpha = 1/3）" % (T_LO, w_kz13))
tq_kz_lo = W_LO ** 3.0
tq_kz_hi = W_HI ** 3.0
print("    A 路窗口反解：tau_Q = %.2e 至 %.2e t_P" % (tq_kz_lo, tq_kz_hi))
w_chase = math.sqrt(T_LO / 2.0)
print("  候选 B（追赶变体，本轮构造：kappa 从 0 线性增长到 1/t_P，模式衰减判据给相干尺度）:")
print("    L = sqrt(int kappa dt) = sqrt(tau_Q/(2 t_P)) 步")
print("    tau_Q = %.2e t_P -> w = %.2f 步" % (T_LO, w_chase))
tq_ch_lo = 2.0 * W_LO * W_LO
tq_ch_hi = 2.0 * W_HI * W_HI
print("    A 路窗口反解：tau_Q = %.2e 至 %.2e t_P" % (tq_ch_lo, tq_ch_hi))
print("  候选 C（视界线性，Kibble 间距读出）：w = c tau_Q")
print("    tau_Q = %.2e t_P -> w = %.2e 步" % (T_LO, T_LO))
print("    A 路窗口反解：tau_Q = %.1f 至 %.1f t_P" % (W_LO, W_HI))
print("  读数：三个候选对同一 tau_Q 给 %.2f / %.2f / %.2e 步，跨度 %.1f 倍" % (
    w_kz13, w_chase, T_LO, T_LO / w_kz13))
print()

print("== V4 A 路窗口的候选对照（含缺陷构型宽度） ==")
print("  校准界面宽度（第 380 条）：<= 2 pi = 6.28 步，判负。")
d_def = math.sqrt(PROD / W_DEF)
print("  缺陷构型宽度 w_def = %.2e l_P：Delta = sqrt(16.98/w_def) = %.3e E_P = %.2e eV" % (
    W_DEF, d_def, d_def * E_P_EV))
print("  判读：窗口内的候选为 KZ 幂律（贴下端 %.1f 步）与追赶变体（%.1f 步）；两者的物理身份" % (w_kz13, w_chase))
print("    （LZ 的 t_sweep 对应畴尺度还是过渡宽度）登记为下一轮判句。")
print()

print("== 图 ==")
fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), dpi=110)
fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.14, wspace=0.30)

ax = axes[0]
tqs = [10 ** (k / 10.0) for k in range(20, 101)]
ax.loglog(tqs, [t ** (1.0 / 3.0) for t in tqs], color="#14508c", lw=2.0, label="KZ 幂律，alpha = 1/3")
ax.loglog(tqs, [t ** 0.25 for t in tqs], color="#14508c", lw=1.4, ls=":", label="KZ 幂律，alpha = 1/4")
ax.loglog(tqs, [math.sqrt(t / 2.0) for t in tqs], color="#2f7a2f", lw=2.0, label="追赶变体 sqrt(tau/2)")
ax.loglog(tqs, tqs, color="#666666", lw=1.6, ls="--", label="视界线性 c tau")
ax.axhspan(W_LO, W_HI, color="#fff3d6", label="A 路宽度窗口 17 至 68 步")
ax.axvspan(T_LO, T_HI, color="#dce9f7", alpha=0.7, label="P4-27 时标档")
ax.set_xlim(1e2, 1e10)
ax.set_ylim(1, 1e10)
ax.set_xlabel("淬火时标 tau_Q / t_P（对数）", fontsize=11)
ax.set_ylabel("候选尺度（步，对数）", fontsize=11)
ax.set_title("(a) 四种标度对同一时标的读数", fontsize=12.5, pad=8)
ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
ax.legend(fontsize=8.5, loc="upper left")

ax = axes[1]
ax.set_xscale("log")
ax.set_xlim(1, 1e22)
ax.set_ylim(0, 1)
ax.axvspan(W_LO, W_HI, color="#fff3d6", label="A 路宽度窗口")
ax.axvspan(0.5, 2 * math.pi, color="#f6d3cf", label="校准界面宽度（380 判负）")
ax.axvline(0.5, color="#c0392b", lw=1.0)
ax.plot([w_kz13], [0.72], "o", color="#14508c")
ax.annotate("KZ 幂律 19.5", (w_kz13 * 1.25, 0.70), fontsize=9.5, color="#14508c")
ax.plot([w_chase], [0.52], "o", color="#2f7a2f")
ax.annotate("追赶 60.7", (w_chase * 1.25, 0.50), fontsize=9.5, color="#2f7a2f")
ax.plot([T_LO], [0.30], "o", color="#666666")
ax.annotate("视界间距 7.4e3", (T_LO * 1.3, 0.28), fontsize=9.5, color="#666666")
ax.axvline(W_DEF, color="#7030a0", lw=1.8)
ax.annotate("缺陷构型宽度 3.8e21", (W_DEF / 4.0, 0.82), fontsize=9.5, color="#7030a0", ha="right")
ax.set_yticks([])
ax.set_xlabel("宽度（步，对数）", fontsize=11)
ax.set_title("(b) 候选尺度与 A 路窗口（tau_Q 固定 7.4e3 t_P）", fontsize=12.5, pad=8)
ax.legend(fontsize=8.5, loc="upper left")
fig.savefig(IMG)
print("  已保存：", IMG)
