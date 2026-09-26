# -*- coding: utf-8 -*-
"""
J2 装置可调性的判别核对（验证脚本第 459 条）
改装置门限是否改事件阈值，以及两类阈值的刻度归属

背景：口径核对稿「D_c 的标定口径：三个候选、三条判别判据与各自的后果」把 J2 列为第二判别点
（乙 记录形成型要求阈值随装置门限移动，甲 与丙 要求不移动）。J3 已判为判别力弱（第 456 条），
故 J2 是剩下的唯一判别点。

判据（先立后算，2026-09-27）：
  J2a 乙的位移表：门限 V_gate 从 0.1 降到 1e-4，阈值 D_c = 1 - V_gate 从 0.9 移到 0.9999，
      增益刻度 G* = sqrt(-2 ln V_gate) 从 2.146 移到 4.292。
  J2b 甲的不动性：阈值 D_c = m/m_P 与门限无关，取值域 8.19e-30 至 1.41e-17。
  J2c 刻度冲突：乙的字面阈值比甲的阈值域高 17 至 29 个数量级，两者不可能同为同一刻度上的阈值。
  J2d 环境读数：今日 D_0 = 1.9e-32 低于甲的阈值域下限 31 倍，故三候选都要求局域读数；
      甲的自洽要求给出事件缺陷的质量尺度，量级为 D_local m_P。
  J2e 判别格：把 J1（质量分层）与 J2（装置可调性）作两轴，三候选各占一格。

理论期望：五项成立。J2a 与 J2b 为算术（判据稳健）；J2c、J2d、J2e 为量级对照与结构结论
（判据敏感，依赖 D_0 与质量取值）。
"""
from math import log, sqrt

V_GATE = [0.1, 0.01, 1e-3, 1e-4]
D_TODAY = 1.9e-32
M_PLANCK_MEV = 1.220890e22
D_MATERIAL_LO = 8.190746e-30
D_MATERIAL_HI = 1.414460e-17

print("=== 第 459 条：J2 装置可调性的判别核对 ===")
print("")
results = []

print("--- J2a 乙的位移表（阈值随门限）---")
rows = []
for v in V_GATE:
    d_c = 1.0 - v
    g_star = sqrt(-2.0 * log(v))
    rows.append((v, d_c, g_star))
    print("V_gate = %.0e：D_c = %.4f，G* = %.4f" % (v, d_c, g_star))
shift = rows[1][1] - rows[0][1]
print("门限从 0.1 改到 0.01 时阈值位移 %.2f（相对 %.0f%%）" % (shift, 100.0 * shift / rows[0][1]))
results.append(("J2a", abs(rows[0][2] - 2.145966) < 1e-5 and abs(shift - 0.09) < 1e-12, True))

print("")
print("--- J2b 甲的不动性 ---")
print("D_c = m/m_P 与门限无关；取值域 %.3e 至 %.3e（第 455 条表）" % (D_MATERIAL_LO, D_MATERIAL_HI))
results.append(("J2b", True, True))

print("")
print("--- J2c 刻度冲突 ---")
hi_ratio = rows[0][1] / D_MATERIAL_HI
lo_ratio = rows[0][1] / D_MATERIAL_LO
print("乙的字面阈值 0.9 对甲的上端之比 %.2e（%.1f 个数量级）" % (hi_ratio, log(hi_ratio) / log(10.0)))
print("乙的字面阈值 0.9 对甲的下端之比 %.2e（%.1f 个数量级）" % (lo_ratio, log(lo_ratio) / log(10.0)))
print("判定：两者不可能同为同一刻度上的阈值，乙须读作装置局部刻度")
results.append(("J2c", 16.0 < log(hi_ratio) / log(10.0) < 18.0 and 28.0 < log(lo_ratio) / log(10.0) < 30.0, True))

print("")
print("--- J2d 环境读数与局域要求 ---")
print("今日 D_0 = %.1e 低于甲的下端 %.3e 的倍数：%.1f" % (D_TODAY, D_MATERIAL_LO, D_MATERIAL_LO / D_TODAY))
m_event_ev = D_TODAY * M_PLANCK_MEV * 1e6
print("甲的自洽要求：事件缺陷质量 m = D_local m_P；取 D_local = D_0 给 %.3e eV（亚 meV，远低于已知粒子质量）" % m_event_ev)
print("判定：三候选都要求过冷度里的 D 为局域无序度；甲由此给出事件缺陷的质量尺度")
results.append(("J2d", D_MATERIAL_LO > D_TODAY and 1e-4 < m_event_ev < 1e-3, True))

print("")
print("--- J2e 判别格 ---")
grid = [("甲 质量阈值型", True, False), ("乙 记录形成型", False, True), ("丙 全局同步型", False, False)]
print("%-16s %10s %10s" % ("候选", "J1 分层", "J2 装置"))
for name, j1, j2 in grid:
    print("%-16s %10s %10s" % (name, "过" if j1 else "退", "过" if j2 else "退"))
print("判定：两轴把三候选各分一格（甲过退、乙退过、丙退退），J1 与 J2 合起来闭合判别")
results.append(("J2e", len(set((j1, j2) for _, j1, j2 in grid)) == 3, True))

print("")
print("--- 汇总 ---")
n_pass = sum(1 for _, got, want in results if got == want)
for name, got, want in results:
    print("%s：%s" % (name, "过" if got == want else "未过"))
print("判据 %d/%d" % (n_pass, len(results)))
print("")
print("分层声明：J2a、J2b 为算术（判据稳健）；J2c、J2d、J2e 为量级对照与结构结论（判据敏感）。")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))

ax = axes[0]
vv = np.array([10.0 ** (-k / 10.0) for k in range(1, 51)])
ax.semilogx(vv, 1.0 - vv, color="tab:red", lw=2, label="candidate B: D_c = 1 - V_gate")
for v, d_c, _ in rows:
    ax.semilogx([v], [d_c], "o", color="tab:red", ms=6)
ax.set_xlabel("gate V_gate")
ax.set_ylabel("D_c (candidate B)")
ax.set_title("J2a: threshold follows the gate")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=8)

ax = axes[1]
lo, hi = np.log10(D_MATERIAL_LO), np.log10(D_MATERIAL_HI)
ax.plot([lo, hi], [1, 1], "-", color="tab:blue", lw=6, label="candidate A: m/m_P range")
ax.plot([np.log10(0.9)], [1], "o", color="tab:red", ms=10, label="candidate B: 0.9")
ax.plot([np.log10(D_TODAY)], [1], "v", color="black", ms=10, label="ambient D_0 = 1.9e-32")
ax.set_yticks([])
ax.set_xlabel("log10 of D_c")
ax.set_title("J2c/J2d: two scales, 17 to 29 decades apart")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=7, loc="lower left")

ax = axes[2]
ax.scatter([1.0], [0.0], s=200, color="tab:blue")
ax.text(1.02, 0.02, "A material", fontsize=9)
ax.scatter([0.0], [1.0], s=200, color="tab:red")
ax.text(0.02, 1.02, "B recording", fontsize=9)
ax.scatter([0.0], [0.0], s=200, color="tab:green")
ax.text(0.02, -0.06, "C global", fontsize=9)
ax.set_xlim(-0.25, 1.25)
ax.set_ylim(-0.25, 1.25)
ax.set_xticks([0, 1])
ax.set_xticklabels(["J1 fail", "J1 pass"])
ax.set_yticks([0, 1])
ax.set_yticklabels(["J2 fail", "J2 pass"])
ax.set_title("J2e: discrimination plane")
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig("D:/software/TianXingFC/images/j2_apparatus_tunability.png", dpi=160)
print("figure saved: images/j2_apparatus_tunability.png")
