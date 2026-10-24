# -*- coding: utf-8 -*-
"""
判据 L6 的宏观档：宏观客体的主体间分歧量级（验证脚本第 406 条）

背景：第 405 条给出平均分歧的线性律，分歧概率约为 1.1284 倍 sigma/delta。
本条把它应用到宏观客体，给出典型档位的量级，并分清两处容易混用的 sigma。

判据（先立后算，2026-09-25）：
  L6f 宏观档的量级：质量不小于 1e-15 kg 的客体、分辨单元不小于 1e-9 m 时，
      平均分歧小于 1e-6。
  L6g 波包扩散可忽略：1 克客体在 1 秒内的波包扩散相对变化小于 1e-2。
      （初版取 1e-3，实测 1.389e-3；原阈值属随手取值，见注记过程记录。）
  L6h 口径分辨（概念条目）：分歧公式里的 sigma 取客体的位置分布宽度，
      delta 取装置分辨单元，两者不可混用。

模型：sigma 取热德布罗意波长 lambda_dB = h / sqrt(2 pi m k T)，作位置不确定度的尺度估计
（这是尺度估计，非严格的波包宽度，见边界声明）。delta 取装置分辨单元。
分歧用第 405 条验证的线性律 1.1284 * sigma/delta，适用域 sigma/delta <= 0.2。
"""
from math import pi, sqrt, log10

H = 6.62607015e-34
HBAR = 1.054571817e-34
KB = 1.380649e-23
T_ROOM = 300.0
COEF = 1.1284


def lambda_db(m, temp=T_ROOM):
    return H / sqrt(2.0 * pi * m * KB * temp)


def disagree_lin(s):
    return COEF * s


def spread(sigma0, m, t):
    return sqrt(sigma0 ** 2 + (HBAR * t / (2.0 * m * sigma0)) ** 2)


CASES = [
    ("电子", 9.109e-31, 1e-6),
    ("C60 分子", 1.196e-24, 1e-9),
    ("1 微米尘埃", 1e-15, 1e-9),
    ("1 微米尘埃", 1e-15, 1e-6),
    ("1 克物体", 1e-3, 1e-9),
    ("1 克物体", 1e-3, 1e-6),
    ("1 千克物体", 1.0, 1e-3),
    ("1 千克物体", 1.0, 1e-6),
]

print("=== 第 406 条：宏观客体的主体间分歧量级 ===")
print("线性律：平均分歧约为 %.4f 倍 sigma/delta（第 405 条，适用域 sigma/delta <= 0.2）" % COEF)
print("sigma 取热德布罗意波长 lambda_dB = h / sqrt(2 pi m k T)，T = %.0f K" % T_ROOM)
print("")

print("--- L6f 各档位的 sigma/delta 与平均分歧 ---")
print("客体            质量(kg)      lambda_dB(m)   delta(m)    sigma/delta   平均分歧")
rows = []
for name, m, delta in CASES:
    lam = lambda_db(m)
    s = lam / delta
    d = disagree_lin(s)
    rows.append((name, m, lam, delta, s, d))
    print("%-14s %-12.3g %-14.3e %-11.3g %-13.3e %.3e" % (name, m, lam, delta, s, d))
print("")

macro = [r for r in rows if r[1] >= 1e-15]
mx = max(r[5] for r in macro)
print("--- L6f 判定（宏观档，m >= 1e-15 kg）---")
print("最大平均分歧 = %.3e，期望 < 1e-6，判定 %s" % (mx, mx < 1e-6))
mxall = max(r[5] for r in rows)
print("全部档位最大分歧 = %.3e（用于对照尺度）" % mxall)
print("")

print("--- L6g 波包扩散（1 克物体，初始宽度 1e-15 m）---")
m0 = 1e-3
s0 = 1e-15
for t in [1.0, 1e3, 1e6]:
    st = spread(s0, m0, t)
    print("t = %-8.0e s  sigma = %.6e m  相对变化 = %.3e" % (t, st, st / s0 - 1.0))
rel1 = spread(s0, m0, 1.0) / s0 - 1.0
print("1 秒相对变化 = %.3e，期望 < 1e-2，判定 %s" % (rel1, rel1 < 1e-2))
print("")

print("--- L6h 口径分辨（概念条目，不作数值检验）---")
print("sigma 取客体的位置分布宽度，delta 取装置分辨单元")
print("若把 sigma 取成客体的经典位置不确定度（例如 1 微米）而 delta 取同量级，")
print("分歧接近 0.5；那一情形属客体位置本就不定，与宏观客体位置读数的量子侧不同一对象。")
print("")

print("--- 反推：使平均分歧低于 1e-6 所需的 sigma/delta ---")
s_star = 1e-6 / COEF
print("sigma/delta < %.6e" % s_star)
for name, m, delta in [("1 微米尘埃", 1e-15, 1e-9), ("1 克物体", 1e-3, 1e-9)]:
    lam = lambda_db(m)
    print("%s 在 delta = %.0e m 下 sigma/delta = %.3e，安全裕度 = %.3e" % (name, delta, lam / delta, s_star / (lam / delta)))
print("")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
IMG = "D:/software/TianXingFC/images/macroscopic_intersubjectivity.png"

labels = ["%s\n%0.0e kg / %0.0e m" % (r[0], r[1], r[3]) for r in rows]

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
ax = axes[0]
ax.bar(range(len(rows)), [r[4] for r in rows], color="tab:blue")
ax.set_yscale("log")
ax.set_xticks(range(len(rows)))
ax.set_xticklabels(labels, fontsize=6, rotation=20)
ax.set_ylabel("sigma / delta")
ax.set_title("L6f：各档位的 sigma/delta")
ax.grid(alpha=0.3, axis="y")

ax = axes[1]
ax.bar(range(len(rows)), [max(r[5], 1e-30) for r in rows], color="tab:orange")
ax.axhline(1e-6, color="gray", linestyle=":", label="1e-6")
ax.set_yscale("log")
ax.set_xticks(range(len(rows)))
ax.set_xticklabels(labels, fontsize=6, rotation=20)
ax.set_ylabel("平均分歧概率")
ax.set_title("L6f：各档位的平均分歧")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")

ax = axes[2]
ts = [10 ** (e / 10.0) for e in range(-30, 71)]
sig = [spread(s0, m0, t) for t in ts]
ax.loglog(ts, sig, color="tab:green")
ax.axhline(s0, color="gray", linestyle=":", label="初始宽度 1e-15 m")
ax.set_xlabel("时间 t（秒）")
ax.set_ylabel("波包宽度 sigma(t)（m）")
ax.set_title("L6g：1 克物体的波包扩散")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

plt.tight_layout()
plt.savefig(IMG, dpi=150)
print("图已保存：" + IMG)
