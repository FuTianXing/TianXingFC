# -*- coding: utf-8 -*-
"""LZ 的 t_sweep 物理身份判定：Delta 硬约束、快过程排除与缺口归约（第 382 条）。

判据（确定性算术 + 结构性论证）：
  T1 联合域：t_sweep 与 Delta 由计数式绑定，Delta 的紫外截断给 t_sweep 下限
  T2 快过程排除：t_sweep 取 1 步与 2 pi 步时的 Delta 需求
  T3 缺陷能隙读法的排除：Delta = sqrt(kappa_phi t_P) 反解 t_sweep
  T4 候选清单与相变时标档对照（归约结论）

运行：python 验证脚本/minimal_verification_tsweep_identity.py
"""
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "images", "tsweep_identity.png")

PROD = 106.7 / (2.0 * math.pi)      # 16.9818
T_P = 5.391247e-44                  # 秒
KAPPA_TP = 9.293805e-47             # kappa_phi t_P
D_DEF = math.sqrt(KAPPA_TP)         # 缺陷能隙，无量纲（以 E_P 为单位）
T_LO = 10.0 ** -39.4 / T_P          # P4-27 下端，t_P
T_HI = 10.0 ** -34.5 / T_P          # P4-27 上端，t_P
T_W1 = 4.09e-8 / T_P                # W1 档，t_P

def d_req(t):
    return math.sqrt(PROD / t)

print("== 设定 ==")
print("  计数式：t_sweep = %.4f / (Delta/E_P)^2（t_P 单位）" % PROD)
print("  输入：kappa_phi t_P = %.6e；缺陷能隙 sqrt(kappa_phi t_P) = %.4e（以 E_P 为单位）" % (KAPPA_TP, D_DEF))
print()

print("== T1 联合域与 t_sweep 下限 ==")
print("  截断：Delta <= E_P（松读法）或 <= E_P/2（两摆程）")
for d, tag in ((1.0, "Delta = E_P"), (0.5, "Delta = E_P/2")):
    print("  %s -> t_sweep >= %.1f t_P" % (tag, PROD / (d * d)))
print("  判读：t_sweep >= 17.0（松读法）或 >= 67.9（两摆程）")
print()

print("== T2 快过程排除 ==")
for t, tag in ((1.0, "单拍写入"), (2.0 * math.pi, "校准上界 2 pi")):
    dr = d_req(t)
    print("  t_sweep = %.4g t_P（%s）-> Delta = %.4f E_P，超截断 %.1f%%（松读法）" % (
        t, tag, dr, (dr / 1.0 - 1) * 100))
print("  判读：两个快过程档均超截断，排除")
print()

print("== T3 缺陷能隙读法的排除 ==")
t_req = PROD / (D_DEF * D_DEF)
print("  读法：若 Delta = 缺陷能隙 = %.4e E_P，则 t_sweep = %.4e t_P" % (D_DEF, t_req))
print("  换算：%.4e t_P = %.4g 秒 = %.3g 小时" % (t_req, t_req * T_P, t_req * T_P / 3600.0))
print("  对照：P4-27 上界 %.2e t_P 的 %.2e 倍（%.1f 个数量级）" % (T_HI, t_req / T_HI, math.log10(t_req / T_HI)))
print("         W1 档 %.2e t_P 的 %.2e 倍（%.1f 个数量级）" % (T_W1, t_req / T_W1, math.log10(t_req / T_W1)))
print("  判读：所需时标远超全部登记档，排除")
print()

print("== T4 候选清单与档位对照 ==")
rows = [
    ("单拍写入", 1.0),
    ("校准上界 2 pi", 2.0 * math.pi),
    ("KZ 幂律（tau_Q 下端）", T_LO ** (1.0 / 3.0)),
    ("追赶（tau_Q 下端）", math.sqrt(T_LO / 2.0)),
    ("视界间距（tau_Q 下端）", T_LO),
    ("相变时标（P4-27 上端）", T_HI),
    ("缺陷固有时间", t_req),
]
for tag, t in rows:
    print("  %-22s t_sweep = %10.4g t_P -> Delta = %10.4g E_P" % (tag, t, d_req(t)))
print("  判读：主群候选（KZ/追赶）在窗口与截断内；视界与时标档在截断内但超窗口；两个快过程与缺陷能隙档被排除")
print()

print("== 图 ==")
fig, ax = plt.subplots(figsize=(9.6, 5.4), dpi=110)
fig.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.13)

xs = [10 ** (-25 + k * 0.1) for k in range(0, 281)]
ys = [PROD / (x * x) for x in xs]
ax.loglog(xs, ys, color="#14508c", lw=2.0, label="计数式 t = 16.98/(Delta/E_P)^2")
ax.axvspan(1e-25, 0.5, color="#f6d3cf", alpha=0.55)
ax.axvline(0.5, color="#c0392b", ls="--", lw=1.0)
ax.axvline(1.0, color="#2f7a2f", ls=":", lw=1.0)
ax.axhspan(T_LO, T_HI, color="#dce9f7", alpha=0.65)
pts = [
    (4.1196, 1.0, "#c0392b", "单拍写入"),
    (1.6439, 2.0 * math.pi, "#c0392b", "校准上界"),
    (0.9338, T_LO ** (1.0 / 3.0), "#14508c", "KZ 幂律"),
    (0.5285, math.sqrt(T_LO / 2.0), "#2f7a2f", "追赶"),
    (0.0480, T_LO, "#666666", "视界/时标下端"),
]
for x, y, c, tag in pts:
    ax.plot([x], [y], "o", color=c)
    ax.annotate(tag, (x * 1.6, y * 1.6), fontsize=9, color=c)
ax.plot([D_DEF], [t_req], "o", color="#7030a0")
ax.annotate("缺陷能隙 %.1e" % D_DEF, (D_DEF * 3.0, t_req * 0.5), fontsize=9, color="#7030a0")
ax.set_xlim(1e-25, 30)
ax.set_ylim(1, 1e52)
ax.set_xlabel("交叉缺口 / E_P（对数）", fontsize=11)
ax.set_ylabel("t_sweep / t_P（对数）", fontsize=11)
ax.set_title("(a) t_sweep 与缺口的联合域、候选点与时标档（浅蓝带为 P4-27 档）", fontsize=12, pad=8)
ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
ax.legend(fontsize=9, loc="lower left")
fig.savefig(IMG)
print("  已保存：", IMG)
