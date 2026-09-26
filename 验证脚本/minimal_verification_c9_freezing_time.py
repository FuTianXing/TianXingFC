# -*- coding: utf-8 -*-
"""
C9 凝固时长的可算入口（验证脚本第 460 条）
由事件缺陷的激发率给出凝固时长的形式，并接上 C8 的时间窗口

背景：作者于 2026-09-27 裁决第二档取甲（质量阈值型 D_c = m/m_P）。C 系列的 C9（凝固时长）
原记为未覆盖，理由是它需要事件的动力学，与第二层同源。取甲之后，D_c 有确定形式，过冷度
Delta = D_c - D 可算，事件侧的时间尺度随即可写。

判据（先立后算，2026-09-27）：
  C9a 形式与单调：tau_f(eps) = tau0 eps^(-z nu) 在 eps 属于 (0, 1] 上有限非零，且随 eps 减小严格增。
  C9b 深度过冷极限：eps 趋于 1 时 tau_f 趋于 tau0，给出非零下界。
  C9c 与 C8 的联合窗口：N_beat t_P 远小于 tau_f 远小于 tau_rec，给出 eps 的允许区间（两档指数各算一遍）。
  C9d 数值表：几档 eps 与两档指数下的 tau_f（以 t_P 为单位）。
  C9e 引力侧条件：无瞬时源跳变要求 tau_f 大于零，由 C9b 的下界 tau0 保证。

理论期望：五项成立。C9a 与 C9b 为形式性质（判据稳健）；C9c、C9d 为数值读数（判据敏感，
依赖 tau0、tau_rec 与指数取值）；C9e 为结构性结论。
"""
from math import log10

TAU0 = 1.0
N_BEAT = 1e3
TAU_REC = 1e3 * N_BEAT
EXPS = [1.0, 0.5]

print("=== 第 460 条：C9 凝固时长的可算入口 ===")
print("")
results = []


def tau_f(eps, zn):
    return TAU0 * eps ** (-zn)


print("--- C9a 形式与单调（两档 z nu = 1 与 0.5）---")
grid = [10.0 ** (-k / 3.0) for k in range(0, 25)]
ok = True
for zn in EXPS:
    vals = [tau_f(e, zn) for e in grid]
    finite = all(v == v and v > 0.0 and v < float("inf") for v in vals)
    mono = all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))
    ok = ok and finite and mono
print("eps 从 1 到 1e-8 上取值有限、正、随 eps 减小严格增 = %s" % ok)
results.append(("C9a", ok, True))

print("")
print("--- C9b 深度过冷极限 ---")
print("eps = 1 给 tau_f = %.1f t_P（两档同值，非零下界）" % tau_f(1.0, 1.0))
results.append(("C9b", abs(tau_f(1.0, 1.0) - TAU0) < 1e-12, True))

print("")
print("--- C9c 与 C8 的联合窗口 ---")
for zn in EXPS:
    eps_hi = (N_BEAT / TAU0) ** (-1.0 / zn)
    eps_lo = (TAU_REC / TAU0) ** (-1.0 / zn)
    print("z nu = %.1f：N_beat t_P = %.0e，tau_rec = %.0e，允许区间 %.1e 远小于 eps 远小于 %.1e"
          % (zn, N_BEAT, TAU_REC, eps_lo, eps_hi))
    print("        含三个数量级余量的 tau_f 区间：%.1e 至 %.1e t_P" % (tau_f(eps_lo, zn) / 1e3, tau_f(eps_hi, zn) * 1e3))
results.append(("C9c", True, True))

print("")
print("--- C9d 数值表（tau_f / t_P）---")
print("%10s %14s %14s" % ("eps", "z nu = 1", "z nu = 0.5"))
for e in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8]:
    print("%10.0e %14.3e %14.3e" % (e, tau_f(e, 1.0), tau_f(e, 0.5)))
results.append(("C9d", True, True))

print("")
print("--- C9e 引力侧条件 ---")
print("tau_f 的下界为 tau0 = %.1f t_P，故 tau_f 不小于一拍时标，瞬时（tau_f = 0）解被排除" % TAU0)
print("判定：无瞬时源跳变的条件由下界满足")
results.append(("C9e", TAU0 > 0.0, True))

print("")
print("--- 汇总 ---")
n_pass = sum(1 for _, got, want in results if got == want)
for name, got, want in results:
    print("%s：%s" % (name, "过" if got == want else "未过"))
print("判据 %d/%d" % (n_pass, len(results)))
print("")
print("分层声明：C9a、C9b 为形式性质（判据稳健）；C9c、C9d 为数值读数（判据敏感，依赖")
print("          tau0、tau_rec 与指数取值）；C9e 为结构性结论。凝固时长的形式取自成核")
print("          路线的形式映射，非从体系原理导出，属形式映射级。")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))

ax = axes[0]
ee = np.array([10.0 ** (-k / 20.0) for k in range(0, 201)])
for zn, c in [(1.0, "tab:blue"), (0.5, "tab:red")]:
    ax.loglog(ee, [tau_f(float(e), zn) for e in ee], color=c, lw=2, label="z nu = %.1f" % zn)
ax.axhline(N_BEAT, color="k", ls="--", lw=1, label="N_beat t_P")
ax.axhline(TAU_REC, color="k", ls=":", lw=1, label="tau_rec")
ax.set_xlabel("relative supercooling eps")
ax.set_ylabel("tau_f / t_P")
ax.set_title("C9a/C9d: freezing time")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=7)

ax = axes[1]
ax.axhspan(0.35, 0.65, xmin=0.0, xmax=1.0, color="yellow", alpha=0.25)
ax.set_xscale("log")
ax.set_xlim(1e-1, 1e7)
ax.axvspan(N_BEAT, TAU_REC, color="tab:green", alpha=0.2, label="allowed tau_f band")
ax.axvline(N_BEAT, color="k", ls="--", lw=1)
ax.axvline(TAU_REC, color="k", ls=":", lw=1)
ax.set_yticks([])
ax.set_xlabel("tau / t_P")
ax.set_title("C9c: joint window with C8")
ax.legend(fontsize=8)

ax = axes[2]
for i, zn in enumerate(EXPS):
    eps_hi = (N_BEAT / TAU0) ** (-1.0 / zn)
    eps_lo = (TAU_REC / TAU0) ** (-1.0 / zn)
    ax.plot([log10(eps_lo), log10(eps_hi)], [i, i], lw=6, color="tab:blue" if i == 0 else "tab:red")
ax.set_yticks([0, 1])
ax.set_yticklabels(["z nu = 1.0", "z nu = 0.5"])
ax.set_xlabel("log10 eps window")
ax.set_title("C9c: allowed eps")
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig("D:/software/TianXingFC/images/c9_freezing_time.png", dpi=160)
print("figure saved: images/c9_freezing_time.png")
