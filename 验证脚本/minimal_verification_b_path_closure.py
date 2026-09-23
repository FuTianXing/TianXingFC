# -*- coding: utf-8 -*-
"""B 路收尾判定：三件闭环条件的可达性与精度缺口（第 390 条）。

判据（确定性算术 + 材料清点）：
  B1 复核：温度族读数的敏感度与穿线点（复核第 364、365 条算术）
  B2 件二（p 精度）：当前自由度与要求精度的对比（多口径）
  B3 件一与件三：决策类型判定（裁决请求）与建模任务判定
  B4 综合：B 路状态（条件性候选，三件未齐；件二有定量缺口）

运行：python 验证脚本/minimal_verification_b_path_closure.py
"""
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "images", "b_path_closure.png")

V_GEV = 246.0
TARGET = 106.7
# 温度族四点读数（第 364 条）
PS = [2.00, 2.05, 2.10, 2.15]
TS = [0.066, 0.243, 0.846, 2.7]   # GeV

print("== B1 复核：敏感度与穿线点 ==")
for i in range(3):
    slope = (math.log(TS[i+1]) - math.log(TS[i])) / (PS[i+1] - PS[i])
    print("  段 p = %.2f 至 %.2f：d lnT / dp = %.1f" % (PS[i], PS[i+1], slope))
# 穿线点（在 2.10 与 2.15 之间插值，指数 = v/T）
e1 = V_GEV / TS[2]
e2 = V_GEV / TS[3]
frac = (math.log(e1) - math.log(TARGET)) / (math.log(e1) - math.log(e2))
p_star = PS[2] + 0.05 * frac
print("  指数穿线点：p* = %.4f（目标指数 %.1f，v = %.0f GeV）" % (p_star, TARGET, V_GEV))
print()

print("== B2 件二：p 精度要求与当前自由度 ==")
SENS = 23.1
d_1pct = 0.01 / SENS
d_5pct = 0.05 / SENS
d_2x = math.log(2.0) / SENS
print("  要求（敏感度 %.1f）：指数对上 1%% 需 dp = %.3e；5%% 需 %.3e；2 倍需 %.3e" % (SENS, d_1pct, d_5pct, d_2x))
df_family = PS[3] - PS[0]
print("  当前自由度口径一（族宽）：dp ~ %.2f" % df_family)
# 口径二：xi 自由度给 t_* 九个数级移动；辐射主导 T ~ t^{-1/2}
dlnT = 9.0 * math.log(10.0) / 2.0
df_xi = dlnT / SENS
print("  当前自由度口径二（xi 折算，九个数级时标移动、T ~ t^{-1/2}）：dp ~ %.3f" % df_xi)
print("  缺口：口径一 %.2e / %.0e = %.0f 倍（%.1f 个数量级）" % (df_family, d_1pct, df_family / d_1pct, math.log10(df_family / d_1pct)))
print("        口径二 %.3f / %.0e = %.0f 倍（%.1f 个数量级）" % (df_xi, d_1pct, df_xi / d_1pct, math.log10(df_xi / d_1pct)))
print("  连 2 倍指数精度（需 %.3e）也不够：口径一差 %.0f 倍" % (d_2x, df_family / d_2x))
print()

print("== B3 件一与件三 ==")
print("  件一（v 的输入地位）：两个选项。列入输入清单属框架决策（裁决请求）；")
print("    推导 v 的来源属新机制（v/E_P = 2.02e-17 的小量另有登记，属换位）。")
print("  件三（O(1) 系数）：所需为 0.03 至 1.17 v 范围内选定一个系数（随族成员变化）；")
print("    需要凝聚势的形状（定居密度的势模型），属可建模的小任务，尚未执行。")
print()

print("== B4 综合 ==")
print("  B 路不是判负，而是三件未齐：件一属框架决策、件二有 %.1f 至 %.1f 个数量级的定量缺口、" % (
    math.log10(df_family / d_1pct), math.log10(df_xi / d_1pct)))
print("  件三属未执行的建模。B 路的状态取「条件性候选（三件未齐）」。")
print()

print("== 图 ==")
fig, ax = plt.subplots(figsize=(9.8, 4.6), dpi=110)
fig.subplots_adjust(left=0.09, right=0.97, top=0.88, bottom=0.16)

eps = [V_GEV / t for t in TS]
ax.semilogy(PS, eps, "o-", color="#14508c", lw=2.0, label="指数 = v / T_*(p)（第 364 条读数）")
ax.axhline(TARGET, color="#c0392b", ls="--", lw=1.2, label="目标指数 106.7")
ax.plot([p_star], [TARGET], "o", color="#c0392b")
ax.annotate("穿线点 p* = %.4f" % p_star, (p_star + 0.005, TARGET * 1.6), fontsize=9.5, color="#c0392b")
ax.axvspan(p_star - d_1pct, p_star + d_1pct, color="#2f7a2f", alpha=0.85, label="要求精度带（1%%，宽 9e-4）")
ax.axvspan(p_star - 0.075, p_star + 0.075, color="#dce9f7", alpha=0.7, label="当前族宽带（约 0.15）")
ax.set_xlim(2.00, 2.30)
ax.set_ylim(50, 4000)
ax.set_xlabel("族参数 p", fontsize=11)
ax.set_ylabel("指数（对数）", fontsize=11)
ax.set_title("B 路件二：要求精度带（绿，宽 9e-4）与当前自由度带（蓝，宽 0.15）在图上相差千倍", fontsize=11, pad=8)
ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
ax.legend(fontsize=8.5, loc="upper right")
fig.savefig(IMG)
print("  已保存：", IMG)
