# -*- coding: utf-8 -*-
"""传递链首轮：力关联尾的二分之 d 律与速度尾反推（第 425 条，候选丁/换算链第一段）。

判据：
  M1 1D 力关联尾 γF（梯度权重、同点、蒙特卡洛连续 k 积分）
  M2 2D 力关联尾 γF
  M3 3D 力关联尾 γF
  M4 反推：γv = 2 - γF（过阻尼 + FDT 口径）与靶 4/5 对照

方法：同点二时间关联 C(t) = < w(k) cos(omega_k t) >_k，梯度权重 w = 4 sin^2(kx/2)/(2 omega)；
      k 用蒙特卡洛连续采样（无周期格点的回归伪影），M 按维度设置；
      包络 = 对数分段 RMS；噪声地板 = 蒙特卡洛标准误的 3 倍；
      尾指数 = 在 3σ 以上段内的双对数拟合。

运行：python 验证脚本/gradient_tail_chain.py
"""
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "images", "gradient_tail_chain.png")

def mc_tail(dims, M, nseg, per, tmin, tmax, seed, batch=200000):
    r = np.random.default_rng(20260926 + seed)
    edges = np.logspace(math.log10(tmin), math.log10(tmax), nseg + 1)
    ts_all = np.concatenate([np.linspace(edges[i], edges[i + 1], per) for i in range(nseg)])
    C = np.zeros(len(ts_all)); C2 = np.zeros(len(ts_all))
    done = 0
    while done < M:
        mb = min(batch, M - done)
        ks = r.uniform(-np.pi, np.pi, (mb, dims))
        s2 = sum(np.sin(ks[:, i] / 2.0) ** 2 for i in range(dims))
        w = 2.0 * np.sqrt(s2)
        sin2 = 4.0 * np.sin(ks[:, 0] / 2.0) ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            wgt = np.where(w > 1e-12, sin2 / (2.0 * w), 0.0)
        c = wgt[:, None] * np.cos(w[:, None] * ts_all[None, :])
        C += c.sum(axis=0); C2 += (c ** 2).sum(axis=0)
        done += mb
    C /= M; C2 /= M
    se = np.sqrt(np.maximum(C2 - C * C, 0.0) / M)
    mids = []; rms = []; sig = []
    idx = 0
    for i in range(nseg):
        seg = C[idx:idx + per]; segse = se[idx:idx + per]
        mids.append(math.sqrt(edges[i] * edges[i + 1]))
        rms.append(float(np.sqrt(np.mean(seg ** 2))))
        sig.append(float(np.mean(segse)))
        idx += per
    return np.array(mids), np.array(rms), np.array(sig)

def fitc(m, e, s, lo, hi):
    msk = (m >= lo) & (m <= hi) & (e > 3.0 * s)
    if np.sum(msk) < 6:
        return float("nan"), 0
    return float(np.polyfit(np.log(m[msk]), np.log(e[msk]), 1)[0]), int(np.sum(msk))

print("== M1/M2/M3 力关联尾（梯度权重，同点，蒙特卡洛） ==")
m1, e1, s1 = mc_tail(1, 1000000, 60, 12, 10.0, 1e5, seed=1)
m2, e2, s2 = mc_tail(2, 4000000, 40, 8, 10.0, 3e4, seed=2)
m3, e3, s3 = mc_tail(3, 4000000, 40, 8, 10.0, 3e4, seed=3)

def report(name, m, e, s):
    idx = np.where(e > 3.0 * s)[0]
    if len(idx) == 0:
        print("  %s：无可用段" % name)
        return float("nan")
    a, n = fitc(m, e, s, m[idx[0]], m[idx[-1]])
    print("  %s：可用段 [%.1f, %.1f]（%d 段）；γF = %.3f（%d 点）" % (name, m[idx[0]], m[idx[-1]], len(idx), a, n))
    return a

a1 = report("1D（M=1e6）", m1, e1, s1)
a2 = report("2D（M=4e6）", m2, e2, s2)
a3 = report("3D（M=4e6）", m3, e3, s3)
print()

print("== M4 反推（γv = 2 - γF，过阻尼 + FDT 口径） ==")
for name, gF in (("1D", a1), ("2D", a2), ("3D", a3)):
    gFp = abs(gF)
    gv = 2.0 - gFp
    print("  %s：尾指数 γF = %.3f（绝对值）-> γv = 2 - γF = %.3f（靶 4/5 = 0.800；差 %+.3f）" % (name, gFp, gv, gv - 0.8))
print("  规律观察：γF（绝对值）为 %.2f、%.2f、%.2f，约随 d/2（1/2、1、3/2）；γv 随维度下降，靶 0.8 位于 2D 与 3D 之间" % (abs(a1), abs(a2), abs(a3)))
d_eff = 2.0 * (2.0 - 0.8)
print("  若按 γF = d/2 口径，靶 γv = 0.8 对应 d_eff = %.1f（2D 与 3D 之间）" % d_eff)
print()

print("== 图 ==")
fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.4), dpi=110)
fig.subplots_adjust(left=0.06, right=0.99, top=0.86, bottom=0.16, wspace=0.24)
for ax, (m, e, s, a, name) in zip(axes, ((m1, e1, s1, a1, "1D"), (m2, e2, s2, a2, "2D"), (m3, e3, s3, a3, "3D"))):
    ax.loglog(m, e, "o-", color="#c0392b", lw=1.4, markersize=3.5, label="force corr envelope")
    ax.loglog(m, 3.0 * s, color="#999999", ls="-", lw=1.0, label="3 sigma floor")
    ax.loglog(m, 10 ** (a * np.log10(m) + (np.log10(e[len(e)//2]) - a * np.log10(m[len(e)//2]))),
              color="#888888", ls="--", lw=1.3, label="fit %.2f" % a)
    for ref, col in ((1.2, "#2f7a2f"), (1.0, "#14508c"), (0.5, "#7030a0")):
        ax.loglog(m, 10 ** (-ref * np.log10(m) + (np.log10(e[len(e)//2]) + ref * np.log10(m[len(e)//2]))),
                  color=col, ls=":", lw=1.0, alpha=0.8, label="t^-%.1f" % ref)
    ax.set_xlabel("t", fontsize=10.5)
    ax.set_ylabel("envelope", fontsize=10.5)
    ax.set_title("%s force-correlation tail" % name, fontsize=10.5)
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
    ax.legend(fontsize=7.5)
fig.savefig(IMG)
print("  已保存：", IMG)
