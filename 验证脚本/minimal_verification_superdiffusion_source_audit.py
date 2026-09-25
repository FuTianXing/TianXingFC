# -*- coding: utf-8 -*-
"""超扩散输运的候选机制清点：长时尾接口与短程结构判负（第 418 条，P0-5 候选机制六轮）。

判据：
  Q1 alpha = 3/5 的等价表述（长时尾接口的数学换算）
  Q2 候选清单与判定（八条候选：判负五、待建三）
  Q3 判负基线的材料核对（记忆上界、过阻尼、校准瞬时、六档最近者）
  Q4 三接口条件式（跳跃重尾、速度尾、迁移率时变）
  Q5 接口的框架材料位置
  Q6 净判与 P0-5 状态

运行：python 验证脚本/minimal_verification_superdiffusion_source_audit.py
"""
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "images", "superdiffusion_source_audit.png")

ALPHA = 3.0 / 5.0
D_REQ = 1.909e7
T_STAR = 2.5393e36
T_UNIV = 4.35e17 / 5.391247e-44

z_exp = 1.0 / ALPHA
msd_exp = 2.0 / z_exp
d_growth = 2.0 * ALPHA - 1.0
gamma_tail = 2.0 - msd_exp

print("== Q1 alpha = 3/5 的等价表述（长时尾接口） ==")
print("  z（动力学指数，xi ∝ t^{1/z}）= 1/alpha = %.4f" % z_exp)
print("  MSD 指数（<x^2> ∝ t^m）= 2/z = %.4f" % msd_exp)
print("  D_eff 增长指数 = 2 alpha - 1 = %.4f" % d_growth)
print("  速度自关联尾指数 gamma（C_v ∝ s^{-gamma}）= 2 - m = %.4f" % gamma_tail)
print("  语义：gamma < 1 给超扩散主指数 2-gamma；gamma = 1 临界（对数修正）；gamma > 1 正常扩散加修正（Alder 尾 3/2 属此类）")
print("  核对链：alpha = 0.6 -> z = 1.6667 -> m = 1.2 -> gamma = 0.8 -> D 增长 0.2")
print()

print("== Q2 候选清单与判定 ==")
cands = [
    ("长程时间记忆核（GLE 类）", "判负", "记忆时间上界 tau_m <= t_P（第 G4-1 条：碰撞写入每拍一次的两层结构）"),
    ("校准通道输运", "判负", "校准瞬时无损（2026-09-26 裁决）；校准谱 kappa k^2 扩散型（第 380 条）"),
    ("弹道加随机化（持续随机行走）", "判负", "长时极限为正常扩散（z = 2）；超扩散只在瞬态"),
    ("守恒量长时尾（Alder 型）", "判负", "尾指数 3/2 > 1，只给修正不改主指数"),
    ("过阻尼晶格扩散（迁移率档）", "判负", "gamma_d = 0.5958 过阻尼加马尔可夫，给正常扩散"),
    ("跳跃长度重尾（Levy 型，mu = 5/3）", "待建", "需钉扎网络或势垒间距的重尾分布；第 386、D-7 条的分布形状未定"),
    ("迁移率时变（gamma_eff ∝ t^{-1/5}）", "待建", "gamma_d 为常数标定；有效摩擦的时变性无材料"),
    ("缺陷增殖（n(t) ∝ t^{1/5}）", "待建", "需产生机制的定量；无材料"),
    ("两相混合的渗透型输运（凝固分数控制）", "待建", "D ∝ phi_s(t)^k；Avrami 早期指数 4、晚期饱和，均非 1/5，需临界区结构"),
]
for name, verdict, basis in cands:
    print("  %-32s %s  %s" % (name, verdict, basis))
print()

print("== Q3 判负基线的材料核对 ==")
print("  记忆时间：tau_m <= t_P（碰撞写入每拍一次的两层结构，频率依赖至 O((omega t_P)^2)）")
print("  过阻尼：gamma_d = 0.5958（迁移率 18.9 的反解，D 线常数标定）")
print("  校准：瞬时完成且无损，无格点延迟（2026-09-26 作者裁决）")
print("  六档最近者：校准扩散上界 2 pi = %.4f（距目标 %.2f 个数量级）" % (2*math.pi, math.log10(D_REQ/(2*math.pi))))
print()

print("== Q4 三接口条件式 ==")
print("  甲 跳跃重尾：P(l) ∝ l^{-1-mu}，mu = 5/3（尾指数 8/3）")
print("  乙 速度尾：C_v(s) ∝ s^{-4/5}")
print("  丙 迁移率时变：gamma_eff ∝ t^{-1/5}（D ∝ t^{1/5}）")
print()

print("== Q5 接口的框架材料位置 ==")
print("  甲：第 386 条（钉扎密度门槛）与 D-7（钉扎势垒）有势垒读数，分布形状未建模")
print("  乙：弹性/序参量慢模式无材料（校准通道经裁决关闭）")
print("  丙：gamma 在既有标定中为常数，时变性无材料")
print("  丁：两相混合候选需凝固分数 phi_s(t) 与环境耦合 D(phi) 的联合模型；初判 Avrami 端部指数（4 与饱和）均不合 1/5")
print()

print("== Q6 净判 ==")
print("  超扩散输运在框架内无现成活口：五条判负（记忆、校准、PRW、守恒尾、过阻尼），三条待建（重尾、时变、增殖）")
print("  P0-5 缺口性质：需要补的是一类长时结构（幂律尾或重尾）的机制；放大倍数（3e6 倍）属表象")
print("  现有结构（碰撞接力的马尔可夫性、校准瞬时、过阻尼）在结构上均为短时或短程，与长时幂律需求相反")
print()

print("== 图 ==")
fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6), dpi=110)
fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.16, wspace=0.26)

ax = axes[0]
ax.axvspan(0.0, 1.0, color="#dff0d8", alpha=0.9, label="superdiffusive (gamma < 1)")
ax.axvspan(1.0, 2.2, color="#f2f2f2", alpha=0.9, label="normal + correction (gamma > 1)")
ax.axvline(1.0, color="#888888", ls="--", lw=1.4)
ax.plot([gamma_tail], [0.5], "o", color="#2f7a2f", markersize=9)
ax.annotate("target gamma = 4/5", (gamma_tail, 0.5), textcoords="offset points", xytext=(10, 14), fontsize=9.5, color="#2f7a2f")
ax.plot([1.5], [0.5], "o", color="#14508c", markersize=8)
ax.annotate("Alder tail 3/2", (1.5, 0.5), textcoords="offset points", xytext=(-10, -20), fontsize=9.5, color="#14508c")
ax.annotate("Markovian (no power tail)", (1.9, 0.75), fontsize=9.5, color="#c0392b")
ax.set_xlim(0, 2.2)
ax.set_ylim(0, 1)
ax.set_yticks([])
ax.set_xlabel("velocity autocorrelation tail exponent gamma", fontsize=10.5)
ax.set_title("Q1 long-time tail interface", fontsize=10.5)
ax.legend(fontsize=8.5, loc="upper right")

ax = axes[1]
ts = [T_STAR * (10 ** (k * 0.05)) for k in range(0, 201)]
ds = [D_REQ * (t / T_STAR) ** d_growth for t in ts]
ts = [t for t in ts if t <= T_UNIV]
ds = ds[:len(ts)]
ax.loglog(ts, ds, color="#c0392b", lw=2.2, label="required D_eff(t) ~ t^{1/5}")
tiers = [
    ("mobility 0.596", 0.5958, "#14508c"),
    ("ballistic step 1", 1.0, "#14508c"),
    ("calib upper 2 pi", 6.2832, "#2f7a2f"),
    ("c w_def 3.8e21", 3.80e21, "#888888"),
    ("quantum 1.9e22", 1.876e22, "#888888"),
]
for name, val, col in tiers:
    ax.axhline(val, color=col, ls=":", lw=1.2)
    ax.annotate(name, (ts[0]*1.2, val*1.25), fontsize=8.0, color=col)
ax.set_xlim(ts[0], T_UNIV)
ax.set_ylim(1e-1, 1e24)
ax.set_xlabel("time (lattice units)", fontsize=10.5)
ax.set_ylabel("D_eff", fontsize=10.5)
ax.set_title("Q6 requirement: a growing curve, not a constant tier", fontsize=10.5)
ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
ax.legend(fontsize=8.5, loc="lower right")

fig.savefig(IMG)
print("  已保存：", IMG)
