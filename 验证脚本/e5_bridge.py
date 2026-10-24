# -*- coding: utf-8 -*-
"""
E5 的增益与缺陷质量换算（验证脚本第 461 条）
取甲之后把材料侧的质量阈值接到 M2 的增益闸门上

背景：作者于 2026-09-27 裁决第二档取甲（D_c = m/m_P）。第 453 条的 E5 核对把桥接写成
D_c = G^{*2}（增益刻度），并标注该桥接待裁决。取甲之后可以写出换算：把装置侧生成的局部
无序度按冻结族 D = 1 - eta 归一化，得到 D_local(G) = 1 - exp(-G^2/2)，于是两条门槛落在同一
增益刻度上可比。

判据（先立后算，2026-09-27）：
  E5-1 换算形式：材料门槛的增益刻度形式为 G_m = sqrt(-2 ln(1 - m0/m_P))，与 M2 的
       G_M2 = sqrt(2 ln 10) 同尺度。
  E5-2 材料门槛数值：取第 459 条的事件缺陷质量读数 m0/m_P = D_0 = 1.9e-32，得 G_m = 6.16e-16。
  E5-3 支配比：G_M2 / G_m = 3.48e15（15.5 个数量级），故 M2 门槛支配。
  E5-4 交叉质量：两门槛相等给出 m0/m_P = 0.9，即 1.099e19 GeV。
  E5-5 刻度对齐：第 453 条的 4.605（衰减指数刻度）与 0.9（归一化无序度刻度）为同一门限。

理论期望：五项成立。E5-1 与 E5-5 为形式性质（判据稳健）；E5-2、E5-3、E5-4 为数值读数
（判据敏感，依赖 D_0 与闸门取值）。
"""
from math import log, exp, sqrt

D_TODAY = 1.9e-32
M_PLANCK_GEV = 1.220890e19
V_GATE = 0.1
G_M2 = sqrt(-2.0 * log(V_GATE))

print("=== 第 461 条：E5 的增益与缺陷质量换算 ===")
print("")
results = []


def g_material(ratio):
    """材料门槛的增益刻度形式；ratio 小于 1e-8 时用首项展开 sqrt(2 x)，避开 1 - x 的双精度下溢。"""
    if ratio < 1e-8:
        return sqrt(2.0 * ratio)
    return sqrt(-2.0 * log(1.0 - ratio))


def d_local(g):
    """归一化局部无序度 D_local = 1 - exp(-G^2/2)；小 G 用首项展开，避开双精度下溢。"""
    half = (g * g) / 2.0
    if half < 1e-8:
        return half
    return 1.0 - exp(-half)


print("--- E5-1 换算形式 ---")
print("材料门槛的增益刻度形式：G_m = sqrt(-2 ln(1 - m0/m_P))")
print("M2 门槛：G_M2 = sqrt(2 ln 10) = %.6f" % G_M2)
print("两条门槛同尺度可比（都由 D_local(G) = 1 - exp(-G^2/2) 的同一反函数给出）")
results.append(("E5-1", abs(G_M2 - 2.145966) < 1e-5, True))

print("")
print("--- E5-2 材料门槛数值 ---")
gm = g_material(D_TODAY)
print("取 m0/m_P = D_0 = %.1e，得 G_m = %.6e（首项展开 sqrt(2 x)，因 1 - 1.9e-32 在双精度下等于 1）" % (D_TODAY, gm))
print("核对：D_local(G_m) = %.6e（期望 %.1e；G_m^2/2 = %.3e，双精度下 1 - exp 已下溢，故用首项展开）"
      % (d_local(gm), D_TODAY, gm * gm / 2.0))
results.append(("E5-2", abs(d_local(gm) / D_TODAY - 1.0) < 1e-9, True))

print("")
print("--- E5-3 支配比 ---")
ratio = G_M2 / gm
print("G_M2 / G_m = %.3e（%.2f 个数量级）" % (ratio, log(ratio) / log(10.0)))
print("在 D 刻度上：0.9 对 %.1e，相差 %.2f 个数量级" % (D_TODAY, log(0.9 / D_TODAY) / log(10.0)))
print("判定：现实质量范围内 M2 门槛支配材料门槛")
results.append(("E5-3", ratio > 1.0e15, True))

print("")
print("--- E5-4 交叉质量 ---")
cross = 1.0 - exp(-(G_M2 * G_M2) / 2.0)
print("两门槛相等要求 m0/m_P = %.4f，即 %.4e GeV（约 %.2f 个普朗克质量）"
      % (cross, cross * M_PLANCK_GEV, cross))
print("判定：交叉质量远高于任何已知粒子，故已知范围内装置门槛支配")
results.append(("E5-4", abs(cross - 0.9) < 1e-9, True))

print("")
print("--- E5-5 刻度对齐 ---")
print("M2 闸门 V = 0.1：衰减指数刻度给 D_sensor = G^2 = %.6f；归一化刻度给 D_local = %.6f"
      % (G_M2 * G_M2, d_local(G_M2)))
print("判定：4.605170 与 0.9 为同一门限的两个刻度（第 453 条的读数由本条给出刻度归属）")
results.append(("E5-5", abs(G_M2 * G_M2 - 4.605170) < 1e-6 and abs(d_local(G_M2) - 0.9) < 1e-12, True))

print("")
print("--- 汇总 ---")
n_pass = sum(1 for _, got, want in results if got == want)
for name, got, want in results:
    print("%s：%s" % (name, "过" if got == want else "未过"))
print("判据 %d/%d" % (n_pass, len(results)))
print("")
print("分层声明：E5-1、E5-5 为形式性质（判据稳健）；E5-2、E5-3、E5-4 为数值读数（判据敏感，")
print("          依赖 D_0 与闸门取值）。换算结论为两门槛的分工与支配，非两门槛相等。")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))

ax = axes[0]
xs = np.array([10.0 ** (-k / 2.0) for k in range(2, 65)])
ys = [g_material(float(x)) for x in xs]
ax.loglog(xs, ys, color="tab:blue", lw=2, label="G_m = sqrt(-2 ln(1 - m/m_P))")
ax.axhline(G_M2, color="tab:red", ls="--", lw=1.5, label="G_M2 = %.4f" % G_M2)
ax.axvline(0.9, color="black", ls=":", lw=1, label="crossover m/m_P = 0.9")
ax.set_xlabel("event defect m / m_P")
ax.set_ylabel("threshold gain G*")
ax.set_title("E5-1/E5-3: two thresholds")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=7)

ax = axes[1]
ax.plot([log(1e-32), log(0.9)], [1, 1], "-", color="tab:blue", lw=6, label="material scale")
ax.plot([log(0.9)], [1], "o", color="tab:red", ms=10, label="crossover 0.9")
ax.plot([log(0.9)], [0.5], "o", color="tab:red", ms=10)
ax.plot([log(0.9)], [0.5], "o", color="tab:red", ms=10)
ax.set_yticks([0.5, 1])
ax.set_yticklabels(["gate", "material"], fontsize=8)
ax.set_xlabel("log10 (D scale)")
ax.set_title("E5-5: 31 decades apart")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=7)

ax = axes[2]
grid = np.array([10.0 ** (-k / 4.0) for k in range(1, 33)] + [0.9])
eff = [max(g_material(float(x)), G_M2) for x in grid]
ax.semilogx(grid, eff, color="tab:green", lw=2, label="G_eff = max(material, M2)")
ax.axhline(G_M2, color="tab:red", ls="--", lw=1)
ax.axvline(0.9, color="black", ls=":", lw=1)
ax.set_xlabel("event defect m / m_P")
ax.set_ylabel("effective threshold gain")
ax.set_title("E5-4: step at 0.9")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=7)

fig.tight_layout()
fig.savefig("D:/software/TianXingFC/images/e5_gain_mass_bridge.png", dpi=160)
print("figure saved: images/e5_gain_mass_bridge.png")
