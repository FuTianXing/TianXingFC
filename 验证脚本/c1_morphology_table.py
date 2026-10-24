# -*- coding: utf-8 -*-
"""
c1_morphology_table.py

[C1-6 L2 形态参数表] 参数字典与重测核对：宽度、壁张力与锚核凹口（判据 L2-1 至 L2-4）

背景：C1-6 的两侧分工设计（§10.2）把 L2 定义为"半径、厚度与锚核参数的表（供
哈密顿量写项）"，现值为部分（脱钉阈值 0.15、K_c(4) = 0.25，引自 C1-3、C1-4）。
本轮把表补齐：给出参数字典，重测其中可廉价复核的三组（1D 宽度 l*(K)、壁张力
sigma_w(K)、锚核凹口增益），其余条目按来源引用登记。

模型（与 A 轮同口径）：FK 链，E = sum 0.5*kappa_e*(du - 1)^2
+ sum K_i*(1 - cos 2 pi u)，kappa_e = 1，N = 400，梯度下降 eta = 0.005，
20000 步；宽度 = 弹性超量密度二阶矩宽 2*sqrt(r2)；壁张力 = 弛豫 kink 的总能量
（背景能量为零）；凹口 = K_i = K[1 - d*exp(-((i - i0)/w_a)^2)]，增益 =
E_free - E_notch。

判据
L2-1 字典与表：每条参数有符号、含义、进入的哈密顿量项与来源
L2-2 重测一致性：l*(K) 与库内读数（A 轮、S 轮）逐档差 <= 3%
L2-3 标度：l* 对 K 的幂次拟合 -0.5 附近；sigma_w 与 4*sqrt(K)/pi 差 <= 5%
L2-4 凹口增益：为正、对 d 与 w_a 单调不减（与 S 轮同向）

运行：python 验证脚本/c1_morphology_table.py（确定性，约 20 秒）
"""
import math
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

T0 = time.time()
N = 400
KAP = 1.0
ETA = 0.005
STEPS = 20000
KS = (0.0005, 0.002, 0.005, 0.00633, 0.01, 0.02, 0.05)
REF = {0.0005: 12.90, 0.002: 6.44, 0.005: 4.06, 0.00633: 3.60, 0.01: 2.86, 0.02: 2.01, 0.05: 1.35}
IMG = "D:/software/TianXingFC/images/c1_morphology_table.png"


def init_kink(n=N):
    x = np.arange(n, dtype=float)
    u = np.where(x < n // 2, x, x + 1.0)
    u[n // 2] = n / 2 + 0.5
    return x, u


def relax(u, Karr, steps=STEPS, eta=ETA):
    x = np.arange(N, dtype=float)
    u = u.copy()
    for _ in range(steps):
        du = np.diff(u)
        g = np.zeros(N)
        g[:-1] -= KAP * (du - 1.0)
        g[1:] += KAP * (du - 1.0)
        g += Karr * 2.0 * np.pi * np.sin(2.0 * np.pi * u)
        g[0] = g[-1] = 0.0
        u = u - eta * g
        u[0], u[-1] = x[0], float(N)
    return u


def energy(u, Karr):
    du = np.diff(u)
    return float(0.5 * (KAP * (du - 1.0) ** 2).sum()
                 + (Karr * (1.0 - np.cos(2.0 * np.pi * u))).sum())


def width(u):
    du = np.diff(u)
    dens = 0.5 * (du - 1.0) ** 2
    idx = np.arange(N - 1) + 0.5
    c0 = (dens * idx).sum() / dens.sum()
    return 2.0 * np.sqrt((dens * (idx - c0) ** 2).sum() / dens.sum())


print("=" * 78)
print("C1-6 L2 形态参数表：参数字典与重测核对（判据 L2-1 至 L2-4）")
print("=" * 78)

print()
print("[L2-1] 参数字典（符号 | 含义 | 数值或区间 | 进入项 | 来源）")
doc = [
    ("R", "环半径（驻留态）", "R_c(4) = 4.0（K = 0.25）；R_c 随 K 减小而增大", "环的形态", "C1-3（引）"),
    ("alpha*", "脱钉阈值", "0.15", "锚核钉扎深度", "C1-3（引）"),
    ("K_c(4)/K_c(5)", "格点锁定驻留阈值", "0.25 / 0.24", "锚核项", "C1-4（引）"),
    ("K", "钉住强度（锚核项）", "窗口 K <= 1/48（宽）、K <= 1/(16 pi^2)（色散）", "K(1-cos 2 pi u)", "W4/A（重测）"),
    ("d, w_a", "凹口深度/宽度", "0.3-0.9 / 2-8；增益正且单调", "锚核项局域调制", "S 轮（重测）"),
    ("l*", "包层宽度（数值口径）", "见下表；l* = (pi/sqrt3) xi", "包层（宽度）", "A/W4（重测）"),
    ("xi", "软模关联长度", "1/(2 pi sqrt K)", "波性侧宽度机制", "W4（重测）"),
    ("sigma_w", "壁张力（每单位长度）", "E_kink(K)；连续预测 4 sqrt(K)/pi", "切割能（漂移判据）", "本轮（新增）"),
    ("w_perp", "线缺陷横向宽", "负读数（无内驻形态；归属 1D/面壁）", "—", "P 轮（引）"),
    ("t_wall", "三维面壁宽度", "4.06 / 2.86 / 2.01（K = 0.002/0.00633/0.02）", "面壁形态", "费米子四轮（引）"),
]
for a, b, c, dd, e in doc:
    print("     %-14s | %-22s | %-44s | %-16s | %s" % (a, b, c, dd, e))

print()
print("[L2-2/L2-3] 宽度与壁张力重测")
x, u0 = init_kink()
rows = []
for K in KS:
    uf = relax(u0.copy(), np.full(N, K))
    l = width(uf)
    Ek = energy(uf, np.full(N, K))
    sigma_pred = 4.0 * math.sqrt(K) / math.pi
    ref = REF[K]
    rows.append((K, l, Ek, sigma_pred, ref))
    print("     K = %.5f：l* = %6.3f（库内 %5.2f，差 %+.2f%%）；sigma_w = %.4f（预测 %.4f，差 %+.2f%%）"
          % (K, l, ref, 100.0 * (l - ref) / ref, Ek, sigma_pred, 100.0 * (Ek - sigma_pred) / sigma_pred))
fit = np.polyfit(np.log([r[0] for r in rows[:6]]), np.log([r[1] for r in rows[:6]]), 1)[0]
sig_dev = max(abs(r[2] - r[3]) / r[3] for r in rows)
print("     宽度幂次拟合 = %.3f（期望 -0.5 附近）；壁张力最大相对差 = %.2f%%" % (fit, 100.0 * sig_dev))

print()
print("[L2-4] 锚核凹口增益（K = 0.01）")
K0 = 0.01
ufree = relax(u0.copy(), np.full(N, K0))
Efree = energy(ufree, np.full(N, K0))
i0 = N // 2
i = np.arange(N, dtype=float)
gains = {}
for d_ in (0.3, 0.6, 0.9):
    for w_a in (2.0, 4.0, 8.0):
        Karr = K0 * (1.0 - d_ * np.exp(-((i - i0) / w_a) ** 2))
        ua = relax(u0.copy(), Karr)
        gain = Efree - energy(ua, Karr)
        gains[(d_, w_a)] = gain
        print("     d = %.1f，w_a = %.1f：增益 = %+.4f" % (d_, w_a, gain))
mono_d = all(gains[(0.3, w)] <= gains[(0.6, w)] <= gains[(0.9, w)] + 1e-12 for w in (2.0, 4.0, 8.0))
mono_w = all(gains[(d, 2.0)] <= gains[(d, 4.0)] <= gains[(d, 8.0)] + 1e-12 for d in (0.3, 0.6, 0.9))
print("     对 d 单调不减：%s；对 w_a 单调不减：%s" % (mono_d, mono_w))

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

ax = axes[0]
Kg = np.logspace(math.log10(0.0003), math.log10(0.06), 100)
ax.loglog(Kg, [math.pi / math.sqrt(3.0) / (2.0 * math.pi * math.sqrt(k)) for k in Kg], "-", label="(pi/sqrt3) xi")
ax.loglog([r[0] for r in rows], [r[1] for r in rows], "o", ms=5, label="l*（重测）")
ax.axhline(2.0, color="k", ls="--", lw=1)
ax.axvline(1.0 / 48.0, color="r", ls=":", lw=1, label="K = 1/48")
ax.axvline(1.0 / (16.0 * math.pi ** 2), color="b", ls=":", lw=1, label="K = 1/(16 pi^2)")
ax.set_xlabel("K")
ax.set_ylabel("l*")
ax.set_title("(a) 宽度与窗口线")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

ax = axes[1]
ax.loglog([r[0] for r in rows], [r[2] for r in rows], "o", ms=5, label="sigma_w（重测）")
ax.loglog(Kg, [4.0 * math.sqrt(k) / math.pi for k in Kg], "-", label="4 sqrt(K)/pi")
ax.set_xlabel("K")
ax.set_ylabel("sigma_w")
ax.set_title("(b) 壁张力")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

ax = axes[2]
ds = (0.3, 0.6, 0.9)
was = (2.0, 4.0, 8.0)
M = np.array([[gains[(d_, w)] for w in was] for d_ in ds])
im = ax.imshow(M, cmap="viridis", aspect="auto")
ax.set_xticks(range(len(was)))
ax.set_xticklabels(["%.0f" % w for w in was])
ax.set_yticks(range(len(ds)))
ax.set_yticklabels(["%.1f" % d_ for d_ in ds])
ax.set_xlabel("w_a")
ax.set_ylabel("d")
ax.set_title("(c) 凹口增益（K = 0.01）")
fig.colorbar(im, ax=ax, fraction=0.046)

fig.tight_layout()
fig.savefig(IMG, dpi=150)
print()
print("图已存：%s" % IMG)
print("总运行时间 = %.1f 秒" % (time.time() - T0))
