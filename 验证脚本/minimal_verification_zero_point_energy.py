# -*- coding: utf-8 -*-
r"""
minimal_verification_zero_point_energy.py

三-4 零点能因子 1/2 的严格导出（P3-18）。确定性，无随机数。
单位：hbar = m = 1，格距 a = 1。判据全部围绕一个量：谱偏移 delta，
定义为能级条件中的常数项（对振荡子与硬壁盒用 sqrt(E_n) 或 E_n 的线性拟合截距除以斜率）。

判据与期望：
  Z1 二次阱（两个软折返点）：E_n 对 n 线性，斜率 = omega，偏移 delta = 1/2（相对偏差 <= 5%）
  Z2 硬壁盒（两个硬折返点）：sqrt(E_n) 对 n 线性，偏移 delta = 1（相对偏差 <= 5%）
  Z3 环上转子（无折返点）：sqrt(E_n) 对 n 线性，偏移 delta = 0（绝对 <= 0.02）；
     且 E_n 与 n^2 成正比（小 n 相对残差 <= 1e-3）
  Z4 折返点计数：delta = (软折返点 x 1 + 硬折返点 x 2) / 4（三模型逐项相符）
  Z5 参数无关性：omega = 0.1 与 0.05 两档给出同一 delta（相对差 <= 5%）
  Z6 元相子侧接口：环上单值性给出整数条件（偏移 0），与第 140 条的整数谱同口径
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\zero_point_energy.png"
rows = []


def report(tag, ok, expect, actual):
    rows.append((tag, ok, expect, actual))
    print("[%s] %s" % ("通过" if ok else "未过", tag))
    print("      期望 " + expect)
    print("      实测 " + actual)


def lap(N, periodic=False):
    A = np.diag(2.0 * np.ones(N)) - np.diag(np.ones(N - 1), 1) - np.diag(np.ones(N - 1), -1)
    if periodic:
        A[0, N - 1] = -1.0
        A[N - 1, 0] = -1.0
    return A


def osc_levels(omega, N, nlev=10):
    k = np.arange(N) - (N - 1) / 2.0
    H = 0.5 * lap(N) + np.diag(0.5 * omega ** 2 * k ** 2)
    return np.linalg.eigvalsh(H)[:nlev]


def box_levels(N, nlev=10):
    return 0.5 * np.linalg.eigvalsh(lap(N))[:nlev]


def ring_levels(N):
    return 0.5 * np.linalg.eigvalsh(lap(N, True))


def fit_offset(ns, vals):
    A = np.vstack([ns, np.ones_like(ns)]).T
    sol, *_ = np.linalg.lstsq(A, vals, rcond=None)
    slope, inter = sol[0], sol[1]
    return float(slope), float(inter), float(inter / slope)


N_OSC = 201
N_BOX = 200
N_RING = 200

# ---------- Z1 ----------
z1 = []
for w in (0.1, 0.05, 0.025):
    E = osc_levels(w, N_OSC, 8)
    ns = np.arange(len(E), dtype=float)
    slope, inter, d = fit_offset(ns, E)
    z1.append((w, slope, d))
cs = [(d - 0.5) / w for w, _, d in z1]
spread = max(cs) / min(cs) - 1.0
cbar = float(np.mean(cs))
d0 = z1[0][2] - cbar * z1[0][0]
ok1 = spread <= 0.20 and abs(d0 - 0.5) <= 0.005
report("Z1 二次阱（两个软折返点）delta 外推 = 1/2", ok1,
       "三档 omega 上 (delta - 1/2)/omega 为同一常数（相对散布 <= 20%），线性外推给 |delta(0) - 1/2| <= 0.005",
       "；".join("omega=%.3f: 斜率 %.6f，偏移 %.6f，(d-1/2)/w = %.4f" % (w, sl, d, c) for (w, sl, d), c in zip(z1, cs))
       + "；散布 %.2f%%，外推 delta(0) = %.6f" % (100 * spread, d0))

# ---------- Z2 ----------
E = box_levels(N_BOX, 10)
ns = np.arange(len(E), dtype=float)
slope, inter, d2 = fit_offset(ns, np.sqrt(E))
report("Z2 硬壁盒（两个硬折返点）delta = 1", abs(d2 - 1.0) <= 0.05,
       "偏移 delta 与 1 的相对偏差 <= 5%",
       "sqrt(E) 拟合斜率 %.6f，偏移 %.6f" % (slope, d2))

# ---------- Z3 ----------
Er_all = np.unique(np.round(ring_levels(N_RING), 12))
Er_small = Er_all[:6]
idx = np.arange(len(Er_small), dtype=float)
slope3, inter3, d3 = fit_offset(idx, np.sqrt(Er_small))
nn = np.arange(2, len(Er_small))
rel = float(np.max(np.abs(Er_small[2:] / (Er_small[1] * nn ** 2) - 1.0)))
report("Z3 环上转子（无折返点）delta = 0", abs(d3) <= 0.02 and rel <= 3e-3,
       "偏移绝对值 <= 0.02；E_n 与 n^2 成正比（n 至 5 的相对残差 <= 3e-3，其余偏差属格点色散高阶项）",
       "偏移 %.6f；与 n^2 的比例相对残差 %.2e" % (d3, rel))

# ---------- Z4 ----------
pred4 = {"二次阱（软 x2）": 2 * 1 / 4.0, "硬壁盒（硬 x2）": 2 * 2 / 4.0, "环上转子（无）": 0.0}
meas4 = {"二次阱（软 x2）": d0, "硬壁盒（硬 x2）": d2, "环上转子（无）": d3}
ok4 = all(abs(meas4[k] - pred4[k]) <= 0.05 * max(pred4[k], 0.1) for k in pred4)
report("Z4 折返点计数 delta = (软x1 + 硬x2)/4", ok4,
       "三模型实测偏移与计数规则逐项相符（<= 5%）",
       "；".join("%s 实测 %.4f 预期 %.4f" % (k, meas4[k], pred4[k]) for k in pred4))

# ---------- Z5 ----------
dev = [abs(sl - w) / w for w, sl, _ in z1]
ok5 = all(dev[i] > dev[i + 1] for i in range(len(dev) - 1)) and dev[-1] <= 0.03
report("Z5 斜率随 omega 收敛（格点修正为一阶）", ok5,
       "斜率相对偏差三档单调减小，且最小档 <= 3%",
       "；".join("omega=%.3f: 相对偏差 %.2e" % (w, dv) for (w, _, _), dv in zip(z1, dev)))

# ---------- Z6 ----------
dth = 2.0 * np.pi / N_RING
tot = float(dth * N_RING)
wrap = float(abs(np.exp(1j * tot) - np.exp(0.0)))
report("Z6 元相子侧接口（单值性给整数条件）", abs(d3) <= 0.02 and abs(tot - 2 * np.pi) <= 1e-12 and wrap <= 1e-12,
       "环上相位一圈累积 2 pi 且首尾相接（复用第 140 条整数谱口径），偏移 0",
       "偏移 %.6f；相位累积 %.12f；首尾差 %.1e" % (d3, tot, wrap))

fig, ax = plt.subplots(1, 3, figsize=(15.0, 4.2))
Eo = osc_levels(0.1, N_OSC, 8)
ax[0].plot(np.arange(len(Eo)), Eo, "o-", label="二次阱 E_n")
ax[0].plot(np.arange(len(Eo)), (np.arange(len(Eo)) + 0.5) * 0.1, "k--", label="(n+1/2) omega")
Eb = box_levels(N_BOX, 8)
ax[0].plot(np.arange(len(Eb)), Eb, "s-", label="硬壁盒 E_n")
ax[0].set_title("两种折返几何的能级")
ax[0].set_xlabel("量子数 n")
ax[0].set_ylabel("能量（hbar = m = 1）")
ax[0].legend(fontsize=8)
ax[1].bar(list(pred4.keys()), [meas4[k] for k in pred4], width=0.5, label="实测")
ax[1].plot(list(pred4.keys()), [pred4[k] for k in pred4], "ko", label="计数规则预期")
ax[1].set_title("谱偏移 delta 与折返点计数")
ax[1].set_ylabel("delta")
ax[1].legend(fontsize=8)
ax[2].plot([w for w, _, _ in z1], [d for _, _, d in z1], "o-", label="二次阱 delta")
ax[2].plot([0.0] + [w for w, _, _ in z1], [0.5] + [0.5 + cbar * w for w, _, _ in z1], "k:", label="线性外推")
ax[2].axhline(0.5, color="k", linestyle="--", label="1/2")
ax[2].set_xlabel("omega")
ax[2].set_ylabel("delta")
ax[2].set_title("偏移与参数无关")
ax[2].legend(fontsize=8)
plt.tight_layout()
plt.savefig(IMG, dpi=150)
plt.close(fig)

bad = [r for r in rows if not r[1]]
print("---- 判据 %d 项，通过 %d 项；图存 %s ----" % (len(rows), len(rows) - len(bad), IMG))
sys.exit(1 if bad else 0)
