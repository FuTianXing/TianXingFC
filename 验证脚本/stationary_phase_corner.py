# -*- coding: utf-8 -*-
"""gamma_F = d/2 的解析化：BZ 角点半稳相求和与数值对照（第 427 条，候选丁解析段）。

判据：
  P1 d = 1：解析闭式 C1(t) = (2 sqrt(pi t))^{-1} cos(2t - pi/4) 与 MC 数值对照
  P2 d = 2、3：角点半稳相和（k_x = pi 的 2^{d-1} 个角点）与 MC 数值对照
  P3 包络斜率统一对照（解析与数值均给 t^{-d/2}）

解析要点：相位 omega(k) = 2 sqrt(sum sin^2(k_a/2)) 的驻点为 BZ 角点（k_a in {0, pi}）；
权重 sin^2(k_x/2)/omega 仅在 k_x = pi 的角点非零（2^{d-1} 个）；各角点非退化（除全零角点，
其贡献为高阶），半稳相各给 t^{-d/2}。

运行：python 验证脚本/stationary_phase_corner.py
"""
import math
import os
from itertools import product

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "images", "stationary_phase_corner.png")

def c_ana(t, d):
    total = 0.0
    for bits in product((0, 1), repeat=d):
        if bits[0] != 1:
            continue
        m = sum(bits)
        om = 2.0 * math.sqrt(m)
        w = 2.0 / om  # 权重 4 sin^2(kx/2)/(2 omega)，在 kx = pi 处给 2/omega
        sgn = (d - m) - m
        amp = w / (2.0 * math.pi) ** d * (2.0 * math.pi / t) ** (d / 2.0) * (2.0 * math.sqrt(m)) ** (d / 2.0)
        total += amp * math.cos(om * t + math.pi * sgn / 4.0)
    return total

def c_num(t, d, M, seed):
    r = np.random.default_rng(20260926 + seed)
    acc = 0.0
    done = 0
    while done < M:
        mb = min(200000, M - done)
        ks = r.uniform(-np.pi, np.pi, (mb, d))
        s2 = sum(np.sin(ks[:, i] / 2.0) ** 2 for i in range(d))
        w = 2.0 * np.sqrt(s2)
        sin2 = 4.0 * np.sin(ks[:, 0] / 2.0) ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            wgt = np.where(w > 1e-12, sin2 / (2.0 * w), 0.0)
        acc += float(np.sum(wgt * np.cos(w * t)))
        done += mb
    return acc / M

def envelope(ts, vals, edges):
    mids = []; rms = []
    for i in range(len(edges) - 1):
        msk = (ts >= edges[i]) & (ts < edges[i + 1])
        if not np.any(msk):
            continue
        mids.append(math.sqrt(edges[i] * edges[i + 1]))
        rms.append(float(np.sqrt(np.mean(np.array(vals)[msk] ** 2))))
    return np.array(mids), np.array(rms)

print("== P1/P2 逐点对照（解析 vs 数值） ==")
t_probe = [200.0, 500.0, 1000.0, 2000.0, 5000.0]
for d, M, seed in ((1, 1000000, 1), (2, 2000000, 2), (3, 2000000, 3)):
    devs = []
    print("  d = %d：" % d)
    for t in t_probe:
        an = c_ana(t, d)
        nu = c_num(t, d, M, seed)
        dev = abs(nu - an) / max(abs(an), 1e-30)
        devs.append(dev)
        print("    t = %.0f：解析 %+.5e，数值 %+.5e（相对差 %.2f）" % (t, an, nu, dev))
    print("    相对差中位 = %.2f" % float(np.median(devs)))

print()
print("== P3 包络对照（斜率目标 d/2） ==")
edges = np.logspace(2.0, 4.0, 9)
for d, M, seed in ((1, 1000000, 1), (2, 2000000, 2), (3, 2000000, 3)):
    ts = np.logspace(2.0, 4.0, 25)
    num = [c_num(t, d, M // 4, seed) for t in ts]
    ana = [c_ana(t, d) for t in ts]
    mN, eN = envelope(ts, num, edges)
    mA, eA = envelope(ts, ana, edges)
    sN = float(np.polyfit(np.log(mN), np.log(eN), 1)[0])
    sA = float(np.polyfit(np.log(mA), np.log(eA), 1)[0])
    print("  d = %d：数值包络斜率 %.3f，解析包络斜率 %.3f（目标 -%.1f）" % (d, sN, sA, d / 2.0))

print()
print("== 图 ==")
fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.4), dpi=110)
fig.subplots_adjust(left=0.06, right=0.99, top=0.86, bottom=0.16, wspace=0.24)
for ax, (d, M, seed) in zip(axes, ((1, 1000000, 1), (2, 2000000, 2), (3, 2000000, 3))):
    ts = np.logspace(2.0, 4.0, 25)
    num = [c_num(t, d, M // 4, seed) for t in ts]
    ana = [c_ana(t, d) for t in ts]
    mN, eN = envelope(ts, num, edges)
    mA, eA = envelope(ts, ana, edges)
    ax.loglog(mN, eN, "o", color="#c0392b", markersize=5, label="numeric (MC)")
    ax.loglog(mA, eA, "-", color="#14508c", lw=1.8, label="analytic (corner sum)")
    ref = d / 2.0
    ax.loglog(mA, 10 ** (-ref * np.log10(mA) + (np.log10(eA[len(eA)//2]) + ref * np.log10(mA[len(mA)//2]))),
              color="#888888", ls=":", lw=1.2, label="t^-%.1f" % ref)
    ax.set_xlabel("t", fontsize=10.5)
    ax.set_ylabel("envelope", fontsize=10.5)
    ax.set_title("d = %d force-corr envelope" % d, fontsize=10.5)
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
    ax.legend(fontsize=8)
fig.savefig(IMG)
print("  已保存：", IMG)
