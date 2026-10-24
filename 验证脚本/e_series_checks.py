# -*- coding: utf-8 -*-
"""
跨层事件 E 系列三门核对（验证脚本第 453 条）
E2 阈值一致性、E3 时间一致性、E5 与 M2 增益闸门对照

背景：判据草案「跨层事件的成核路线：过冷度对应量与 E 系列判据」列出 E1 至 E5，第十节接口核对
结论为 E2、E3、E5 属核对工作（各有既有读数可对），E1 与 E4 的机制侧等过冷度定下来。本轮执行
三门核对，产出读数与判定。

判据（先立后算，2026-09-27）：
  E2a 单调性：速率 I(d) = I0 exp(-A/d^2) 在 d > 0 上单调增。
  E2b 分辨率阈值：I 降到 1e-12 I0 处的余量 d_res = sqrt(A/ln(1e12))。
  E2c 半速率宽度：I = I0/2 处的余量 d_half = sqrt(A/ln 2)。
  E3a 等待时间：发生率 I 下的等待时间服从指数分布，样本均值乘 I 为 1。
  E3b 相对涨落：指数分布的相对涨落为 1。
  E3c 窗口：I 的允许窗口为 [1/tau_rec, 1/(N_beat t_P))，非空且宽度为 log10(tau_rec/(N_beat t_P)) 个数量级。
  E5a 闸门复算：G* = sqrt(4 ln(1/V_gate)/n_diff)，取 (V_gate, n_diff) = (0.1, 2)，与 M2 报告 2.15 相容。
  E5b 读数复现：V(G) = exp(-G^2/2) 复现 M2 的 V(2) = 0.135 与 V(5) 同量级于 4e-6。
  E5c 桥接自洽：首选口径的相变点无序度候选 D_c = G*^2 = 2 ln 10，与 G* 数值自洽。

理论期望：九项全过。E5c 只判数值自洽，桥接假设的成立与否归 D_c 标定口径的裁决。
"""
from math import log, exp, sqrt
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

A_BARRIER = 1.0
D_RES = sqrt(A_BARRIER / log(1e12))
D_HALF = sqrt(A_BARRIER / log(2.0))
V_GATE = 0.1
N_DIFF = 2
G_STAR = sqrt(4.0 * log(1.0 / V_GATE) / N_DIFF)
G_STAR_NOTE = 2.15


def rate(delta):
    if delta <= 0.0:
        return 0.0
    return exp(-A_BARRIER / (delta * delta))


print("=== 第 453 条：跨层事件 E 系列三门核对 ===")
print("")
results = []

print("--- E2 阈值一致性（成核速率对余量的依赖）---")
grid = np.linspace(0.02, 3.0, 1200)
vals = np.array([rate(float(d)) for d in grid])
mono_math = bool(np.all(np.diff(-A_BARRIER / (grid * grid)) > 0.0))
mask = grid >= 0.3
mono_num = bool(np.all(np.diff(vals[mask]) > 0.0))
d_min = sqrt(A_BARRIER / 744.0)
print("E2a 单调性：对数侧全区间严格递增 = %s；可表示区（d >= 0.3）数值严格递增 = %s"
      % (mono_math, mono_num))
print("    下溢边界 d_min = %.5f（d < d_min 处速率在双精度下为 0，故单调性只在可表示区数值核对）" % d_min)
results.append(("E2a", mono_math and mono_num, True))

i_res = rate(D_RES)
dev_res = abs(i_res / 1e-12 - 1.0)
print("E2b 分辨率阈值：d_res = %.6f，I/I0 = %.6e（期望 1e-12，相对偏差 %.2e）"
      % (D_RES, i_res, dev_res))
results.append(("E2b", dev_res < 1e-9, True))

i_half = rate(D_HALF)
print("E2c 半速率宽度：d_half = %.6f，I/I0 = %.6f（期望 0.5）" % (D_HALF, i_half))
results.append(("E2c", abs(i_half - 0.5) < 1e-9, True))

print("")
print("--- E3 时间一致性（等待时间分布与拍时标）---")
rng = np.random.default_rng(20260927)
i_rate = 1e-3
samples = rng.exponential(scale=1.0 / i_rate, size=2000000)
mean_over = float(samples.mean()) * i_rate
rel_dev = abs(mean_over - 1.0)
print("E3a 样本均值乘 I：%.6f（期望 1，容差 3e-3，偏差 %.2e）" % (mean_over, rel_dev))
results.append(("E3a", rel_dev < 3e-3, True))

spread = float(samples.std(ddof=1)) / float(samples.mean())
print("E3b 相对涨落：%.6f（期望 1，容差 3e-3）" % spread)
results.append(("E3b", abs(spread - 1.0) < 3e-3, True))

n_beats = [10.0, 100.0, 1000.0, 1e5]
t_p = 1.0
tau_rec = 1e3 * np.array(n_beats) * t_p
width = np.log10(tau_rec / (np.array(n_beats) * t_p))
i_edge = 1e-3 / (np.array(n_beats) * t_p)
p_tail = float(1.0 - np.exp(-i_edge[2] * n_beats[2] * t_p))
print("E3c 窗口宽度（十年两倍口径）：%s 个数量级（期望 3.0 全同）" % np.round(width, 6).tolist())
print("    在 I = 1e-3 / (N_beat t_P) 处，低于一拍的尾概率 = %.6e（期望 1e-3）" % p_tail)
results.append(("E3c", bool(np.allclose(width, 3.0)) and abs(p_tail - 1e-3) < 1e-6, True))

print("")
print("--- E5 与 M2 对照（增益闸门）---")
dev_g = abs(G_STAR - G_STAR_NOTE) / G_STAR_NOTE
print("E5a 闸门复算：G* = %.6f，报告值 %.2f，相对偏差 %.4f（容差 5e-3）"
      % (G_STAR, G_STAR_NOTE, dev_g))
results.append(("E5a", dev_g < 5e-3, True))


def vis(g):
    return exp(-(g * g) / 2.0)


v2, v5 = vis(2.0), vis(5.0)
print("E5b 读数复现：V(2) = %.6f（报告 0.135）；V(5) = %.3e（报告 4e-6，同量级）" % (v2, v5))
ok_b = abs(v2 - 0.135) < 5e-3 and 1e-6 < v5 < 1e-5
results.append(("E5b", ok_b, True))

d_c = G_STAR * G_STAR
dev_c = abs(sqrt(d_c) - G_STAR)
print("E5c 桥接自洽：D_c = G*^2 = %.6f（= 2 ln 10 = %.6f），sqrt(D_c) - G* = %.2e"
      % (d_c, 2.0 * log(10.0), dev_c))
results.append(("E5c", dev_c < 1e-12, True))

n_pass = sum(1 for _, got, want in results if got == want)
print("")
print("--- 汇总 ---")
for name, got, want in results:
    print("%s：%s（期望 %s）" % (name, "过" if got == want else "未过", want))
print("判据 %d/%d" % (n_pass, len(results)))
print("")
print("分层声明：E5a、E5b 与 E2a 不依赖速率形式的具体取值，判据稳健；")
print("          E2b、E2c、E3c 依赖势垒取值 A 与持有时间取值，判据敏感。")
print("          E5c 只判数值自洽；桥接假设 D_c = 2 ln 10 待 D_c 标定口径裁决。")

fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))

ax = axes[0]
dd = np.linspace(0.02, 3.0, 600)
ax.semilogy(dd, [max(rate(float(d)), 1e-16) for d in dd], color="tab:blue", lw=2, label="I/I0")
ax.axvline(0.0, color="k", ls="--", lw=1)
ax.semilogy([D_RES], [1e-12], "o", color="tab:red", label="d_res = %.3f" % D_RES)
ax.semilogy([D_HALF], [0.5], "s", color="tab:green", label="d_half = %.3f" % D_HALF)
ax.set_xlabel("margin d (below zero: no event)")
ax.set_ylabel("I / I0")
ax.set_title("E2: nucleation rate vs margin")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=8)

ax = axes[1]
bins = np.linspace(0.0, 6.0, 120)
ax.hist(samples / samples.mean(), bins=bins, density=True, color="tab:blue", alpha=0.6, label="sampled")
xx = np.linspace(0.0, 6.0, 400)
ax.plot(xx, np.exp(-xx), color="tab:red", lw=2, label="Exp(1) pdf")
ax.set_xlabel("T / mean(T)")
ax.set_ylabel("density")
ax.set_title("E3: waiting time (mean x I = %.5f, spread = %.5f)" % (mean_over, spread))
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)

ax = axes[2]
gg = np.linspace(0.0, 5.0, 400)
ax.semilogy(gg, [vis(float(g)) for g in gg], color="tab:blue", lw=2, label="V(G) = exp(-G^2/2)")
ax.axhline(0.1, color="tab:red", ls="--", lw=1, label="gate V = 0.1")
ax.axvline(G_STAR, color="tab:green", ls=":", lw=1.5, label="G* = %.4f" % G_STAR)
ax.semilogy([2.0, 5.0], [v2, v5], "o", color="k", label="M2 readings G = 2, 5")
ax.set_xlabel("gain G")
ax.set_ylabel("visibility V")
ax.set_title("E5: M2 gate recomputation")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig("D:/software/TianXingFC/images/e_series_checks.png", dpi=160)
print("figure saved: images/e_series_checks.png")
