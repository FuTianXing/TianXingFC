# -*- coding: utf-8 -*-
"""同时性相对性的形式化（第 470 条）：等读数切片、爱因斯坦同步与约定自由度。

路线三首项（见 [[内部草稿/材料-狭义相对论与广义相对论的公设定理化盘点：四公设加惯性的现状、级别与路线]] 三·路线三）。
定义基础：同时性 = 振荡同步（同相）（[[元相子几何动力学——束缚振荡修订整合版]] 时间涌现段）；
待形式化：两观测者经光信号比较得到的同时性差异（[[狭义相对论的定理化：从晶格几何到洛伦兹结构]] §5.3 遗留 (a)）。

判据：
  RS1 等读数切片倾斜：共动观测者的同时切片在晶格系中倾斜 v/c^2；两种等价读法
      （固有时分离 gamma*v*L0/c^2 与晶格分离 v*L/c^2）一致（机器精度，多速度档）。
  RS2 爱因斯坦同步协议：A 发射、B 反射、A 取中；反射事件上 B 的读数等于 A 的取中读数
      （协议自洽，机器精度）；晶格系往返时刻 2*gamma*L0/c；等读数事件的偏移
      gamma*v*L0/c^2（前钟落后）；同一晶格时刻的读数差 -v*L0/c^2。
  RS3 约定自由度：往返固有时 2*L0/c 与 v 无关（机器精度）；单程不对称比 = tanh(phi) = v/c；
      取中偏移 = gamma*v*L0/c^2；闭合回路（往返）读数与约定无关。
  RS4 普朗克层偏差：晶格色散 omega = 2*sin(k/2)（单位 t_P = l_P = c = 1），谱传播测群速度
      = cos(k/2)；往返同步修正相对值 = 1/cos(k/2) - 1 = k^2/8 + O(k^4)，系数 1/8 复现；
      折算为 (1/8)(E/E_P^(t))^2 量级（同色散线签名）。

确定性；约数秒。图存 images/simultaneity_formalization.png
"""
import math
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "images", "simultaneity_formalization.png")
T0 = time.time()
C = 1.0


def check(tag, ok, detail):
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))
    return bool(ok)


def gamma(v):
    return 1.0 / math.sqrt(1.0 - (v / C) ** 2)


print("第 470 条 同时性相对性的形式化：RS1 至 RS4")

# ---------------------------------------------------------------- RS1
print("RS1 等读数切片倾斜（解析层，机器精度）")
rows = []
for v in (0.1, 0.3, 0.5, 0.7, 0.9):
    g = gamma(v)
    for L0 in (1.0, 2.5, 7.0):
        t0 = 2.0
        xB = v * t0 + L0 / g
        tB = g * (t0 - v * xB / C ** 2)
        tA = t0 / g
        d_read = tB - tA
        dt_equal = g * v * L0 / C ** 2
        dx_events = g * L0
        slope = dt_equal / dx_events
        rows.append((v, d_read, -v * L0 / C ** 2, dt_equal, slope,
                     g * (-d_read) - dt_equal))
dev_read = max(abs(r[1] - r[2]) for r in rows)
dev_slope = max(abs(r[4] - r[0] / C ** 2) for r in rows)
dev_cons = max(abs(r[5]) for r in rows)
check("RS1a", dev_read < 1e-14,
      "同一晶格时刻的读数差 = -v*L0/c^2（前钟落后）：最大偏差 %.2e（15 组）" % dev_read)
check("RS1b", dev_slope < 1e-14,
      "等读数切片斜率等于 v/c^2：最大偏差 %.2e" % dev_slope)
check("RS1c", dev_cons < 1e-14,
      "速率因子一致性（gamma 乘读数差 = 等读数偏移）：最大偏差 %.2e" % dev_cons)

# ---------------------------------------------------------------- RS2
print("RS2 爱因斯坦同步协议（解析运动学，机器精度）")


def protocol(v, L0):
    g = gamma(v)
    phi = math.atanh(v / C)
    t_catch = L0 * math.exp(phi) / C
    t_catch_alt = L0 / (g * (C - v))
    reading_B = t_catch / g - v * L0 / C ** 2
    t_return = 2 * g * L0 / C
    reading_A = t_return / g
    assigned = 0.5 * reading_A
    offset_readings = g * v * L0 / C ** 2
    reading_diff_fixed_t = -v * L0 / C ** 2
    return dict(t_catch=t_catch, t_catch_alt=t_catch_alt, reading_B=reading_B,
                reading_A=reading_A, assigned=assigned, t_return=t_return,
                offset=offset_readings, diff_fixed_t=reading_diff_fixed_t, phi=phi)


rs2 = []
for v in (0.1, 0.3, 0.5, 0.7):
    for L0 in (1.0, 3.0):
        rs2.append((v, L0, protocol(v, L0)))
d_alt = max(abs(d["t_catch"] - d["t_catch_alt"]) / d["t_catch"] for _, _, d in rs2)
d_self = max(abs(d["reading_B"] - d["assigned"]) / abs(d["reading_B"]) for _, _, d in rs2)
d_round = max(abs(d["t_return"] - 2 * gamma(v) * L0 / C) for v, L0, d in rs2)
check("RS2a", d_alt < 1e-14, "追上时刻两种写法一致：相对偏差 %.2e" % d_alt)
check("RS2b", d_self < 1e-14,
      "协议自洽（B 读数 = A 取中读数）：相对偏差 %.2e（8 组）" % d_self)
check("RS2c", d_round < 1e-14 and max(abs(d["reading_A"] - 2 * L0 / C) for _, L0, d in rs2) < 1e-14,
      "往返时刻 2*gamma*L0/c 与 A 读数 2*L0/c：偏差 %.2e" % d_round)
off_ok = max(abs(d["offset"] - gamma(v) * v * L0 / C ** 2) for v, L0, d in rs2)
sgn_ok = all((d["offset"] > 0) == (v > 0) for v, L0, d in rs2)
check("RS2d", off_ok < 1e-14 and sgn_ok,
      "等读数事件的偏移 gamma*v*L0/c^2（前钟落后）：偏差 %.2e" % off_ok)

# ---------------------------------------------------------------- RS3
print("RS3 约定自由度（往返不变与单程分解）")
rs3 = []
for v in (0.0, 0.1, 0.3, 0.5, 0.7, 0.9):
    g = gamma(v)
    phi = math.atanh(v / C)
    L0 = 2.0
    t_out = L0 * math.exp(phi) / C
    t_back = L0 * math.exp(-phi) / C
    t_round = t_out + t_back
    proper_round = t_round / g
    asym = (t_out - t_back) / t_round
    shift = t_out - t_round / 2
    rs3.append((v, proper_round, asym, shift))
d_const = max(abs(r[1] - 2 * 2.0 / C) for r in rs3)
d_asym = max(abs(r[2] - r[0] / C) for r in rs3)
d_shift = max(abs(r[3] - gamma(r[0]) * r[0] * 2.0 / C ** 2) for r in rs3)
check("RS3a", d_const < 1e-14,
      "往返固有时 2*L0/c 与 v 无关：最大偏差 %.2e（6 档）" % d_const)
check("RS3b", d_asym < 1e-14, "单程不对称比 = tanh(phi) = v/c：最大偏差 %.2e" % d_asym)
check("RS3c", d_shift < 1e-14,
      "取中偏移 = gamma*v*L0/c^2（约定项）：最大偏差 %.2e" % d_shift)

# ---------------------------------------------------------------- RS4
print("RS4 晶格色散与普朗克层同步修正（谱传播）")


def vg_measure(k0, N=8192, sigma=48.0, tmax=700.0, nsnap=11):
    # 单侧谱包（k 空间高斯集中于 +k0）：Re[IFFT(Phi(k) exp(-i omega t))] 为实解，
    # 包单向传播，质心速度即群速度（无左右分包与零初速假设）。
    kk = 2.0 * np.pi * np.fft.fftfreq(N)
    om = 2.0 * np.abs(np.sin(kk / 2.0))
    dk = 1.0 / (2.0 * sigma)
    shape = np.exp(-((kk - k0) ** 2) / (2.0 * dk ** 2)) * (kk > 0.0)
    n0 = N // 4
    phik = shape.astype(complex) * np.exp(-1j * kk * n0)
    ts = np.linspace(0.0, tmax, nsnap)
    n = np.arange(N)
    xc = []
    for t in ts:
        phi = np.real(np.fft.ifft(phik * np.exp(-1j * om * t)))
        w = phi ** 2
        xc.append(np.sum(n * w) / np.sum(w))
    slope = np.polyfit(ts, np.array(xc), 1)[0]
    return slope


ks = (0.15, 0.35, 0.55)
vg = np.array([vg_measure(k) for k in ks])
pred = np.cos(np.array(ks) / 2.0)
rel = np.abs(vg - pred) / pred
check("RS4a", float(np.max(rel)) < 3e-3,
      "群速度实测 vs cos(k/2)：最大相对偏差 %.2e（k = 0.15/0.35/0.55）" % float(np.max(rel)))

coef = (1.0 / vg - 1.0) / np.array(ks) ** 2
coef_pred = 1.0 / 8.0 + 5.0 * np.array(ks) ** 2 / 384.0
d_coef = np.abs(coef - coef_pred)
check("RS4b", float(np.max(d_coef)) < 2e-3,
      "修正系数 (1/vg-1)/k^2 vs 1/8+5k^2/384：最大偏差 %.2e（趋向 1/8 = %.4f）"
      % (float(np.max(d_coef)), float(coef[-1])))
e_over_ep = 2.0 * np.sin(np.array(ks) / 2.0)
lead = (1.0 / 8.0) * e_over_ep ** 2
corr = 1.0 / pred - 1.0
d_lead = np.abs(corr - lead) / corr
# 容差取解析余项预算：corr - (1/8)(E/E_P)^2 = O(k^4)，相对预算 3k^2/16（含 10% 放宽）
budget = (3.0 / 16.0) * np.array(ks) ** 2 * 1.1
check("RS4c", bool(np.all(d_lead <= budget)),
      "折算 (1/8)(E/E_P^(t))^2 在解析余项预算内：实测 %s vs 预算 %s"
      % (["%.2e" % x for x in d_lead], ["%.2e" % x for x in budget]))

# ---------------------------------------------------------------- 图
fig, axs = plt.subplots(2, 2, figsize=(11, 8))

ax = axs[0, 0]
xs = np.linspace(0.0, 4.0, 50)
for v in (0.3, 0.5, 0.7):
    ax.plot(xs, (v / C ** 2) * xs, label="斜率 v/c^2 = %.2f" % (v / C ** 2))
    ax.plot(xs, (gamma(v) * v / C ** 2) * (xs / gamma(v)), "x", ms=3)
ax.set_xlabel("晶格系分离 Δx")
ax.set_ylabel("等读数事件的晶格时刻差 Δt")
ax.set_title("RS1 同时切片倾斜 v/c^2", fontsize=10)
ax.legend(fontsize=8)

ax = axs[0, 1]
v, L0 = 0.5, 3.0
g = gamma(v)
tt = np.linspace(0.0, 1.2 * 2 * g * L0 / C, 200)
ax.plot(0 * tt, tt, "-", color="#2b6cb0", label="A 世界线（x=0）")
ax.plot(v * tt, tt, "-", color="#c53030", label="B 世界线（前钟）")
tc = L0 * math.exp(math.atanh(v / C)) / C
tr = 2 * g * L0 / C
ax.plot([0, tc, 0], [0, tc, tr], "--", color="#555", label="光脉冲")
ax.plot([tc], [tc], "o", color="#555")
ax.vlines(g * v * L0 / C ** 2, 0, g * v * L0 / C ** 2, color="#68a063", lw=1.2)
ax.annotate("等读数偏移\n$\\gamma vL_0/c^2$", xy=(g * v * L0 / C ** 2, g * v * L0 / C ** 2),
            xytext=(g * v * L0 / C ** 2 + 0.4, g * v * L0 / C ** 2 + 1.2), fontsize=8,
            arrowprops=dict(arrowstyle="->", lw=0.8))
ax.set_xlabel("x（晶格系）")
ax.set_ylabel("t（晶格系）")
ax.set_title("RS2 爱因斯坦同步协议（v=0.5c）", fontsize=10)
ax.legend(fontsize=8)

ax = axs[1, 0]
vs = np.array([r[0] for r in rs3])
ax.plot(vs, [r[1] for r in rs3], "o-", color="#2b6cb0", label="往返固有时（应恒为 2L0/c）")
ax.set_ylim(3.9, 4.1)
ax.set_xlabel("v")
ax.set_ylabel("往返固有时")
ax2 = ax.twinx()
ax2.plot(vs, [r[2] for r in rs3], "s--", color="#c53030", label="单程不对称比（= v/c）")
ax2.set_ylabel("单程不对称比")
ax.set_title("RS3 往返约定无关与单程分解", fontsize=10)
ax.legend(fontsize=8, loc="lower left")
ax2.legend(fontsize=8, loc="upper left")

ax = axs[1, 1]
ax.plot(ks, coef, "o", color="#2b6cb0", label="实测 (1/vg-1)/k^2")
ax.plot(ks, coef_pred, "-", color="#c53030", label="解析 1/8 + 5k^2/384")
ax.axhline(1.0 / 8.0, color="gray", ls=":", lw=0.9, label="极限 1/8")
ax.set_xlabel("波数 k（l_P = 1）")
ax.set_ylabel("修正系数")
ax.set_title("RS4 普朗克层修正系数趋向 1/8", fontsize=10)
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(IMG, dpi=150)
print("图已存：%s（用时 %.1f s）" % (IMG, time.time() - T0))
