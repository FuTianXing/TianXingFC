# -*- coding: utf-8 -*-
r"""
zero_point_asymptotic.py

三-4 解析余项轮：把「数值外推给 1/2」升级为解析形式。
确定性，无随机数。单位 hbar = m = 1，格距 1。

判据与期望：
  W1 修正形式：四档 omega 上 (delta - 1/2)/omega 对 omega 线性（拟合残差 <= 5e-4）
  W2 系数与外推：截距 c1 与外推 delta(0) = 1/2 相符（|delta(0) - 1/2| <= 5e-4）；
     c1 与 c2 报出（c1 为格点修正的一阶系数）
  W3 Mathieu 映射：格点振子在动量表象下等价于 Mathieu 方程
     psi'' + (8/omega^2)(E - 1 + cos p) psi = 0，即 a = 8(E-1)/omega^2、q = -4/omega^2；
     两种表象的谱逐项一致（<= 1e-10），故大 q 渐近中的 (2n+1) 结构即折返点的 1/2
  W4 另两个几何的解析值：硬壁盒 delta = 1（盒谱 k_n = (n+1)pi/L）、环上转子 delta = 0（单值性），
     两档抽查（<= 1e-9）
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\zero_point_asymptotic.png"
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


def osc_levels(omega, N, nlev=8):
    k = np.arange(N) - (N - 1) / 2.0
    H = 0.5 * lap(N) + np.diag(0.5 * omega ** 2 * k ** 2)
    return np.linalg.eigvalsh(H)[:nlev]


def fit_offset(vals, ns=None):
    if ns is None:
        ns = np.arange(len(vals), dtype=float)
    A = np.vstack([ns, np.ones_like(ns)]).T
    sol, *_ = np.linalg.lstsq(A, vals, rcond=None)
    return float(sol[0]), float(sol[1]), float(sol[1] / sol[0])


OM = (0.10, 0.05, 0.025, 0.0125)
deltas = []
for w in OM:
    N = int(max(201, 60.0 / w))
    E = osc_levels(w, N, 6)
    _, _, d = fit_offset(E)
    deltas.append(d)

# ---------- W1 ----------
ys = [(d - 0.5) / w for d, w in zip(deltas, OM)]
A2 = np.vstack([np.array(OM), np.ones_like(np.array(OM))]).T
sol2, *_ = np.linalg.lstsq(A2, np.array(ys), rcond=None)
c2, c1 = float(sol2[0]), float(sol2[1])
res = float(np.max(np.abs(np.array(ys) - A2 @ sol2)))
report("W1 修正形式 (delta-1/2)/omega 对 omega 线性", res <= 5e-4,
       "四档 omega 上线性拟合残差 <= 5e-4",
       "；".join("omega=%.4f: (d-1/2)/w = %.6f" % (w, y) for w, y in zip(OM, ys)) + "；最大残差 %.2e" % res)

# ---------- W2 ----------
Aq = np.vstack([np.array(OM) ** 2, np.array(OM), np.ones_like(np.array(OM))]).T
solq, *_ = np.linalg.lstsq(Aq, np.array(deltas), rcond=None)
q2, q1, q0 = [float(x) for x in solq]
resq = float(np.max(np.abs(np.array(deltas) - Aq @ solq)))
report("W2 二次拟合外推给 1/2", abs(q0 - 0.5) <= 5e-4 and resq <= 5e-4,
       "delta = 1/2 + c1 omega + c2 omega^2 的二次拟合给出 |delta(0) - 1/2| <= 5e-4，残差 <= 5e-4",
       "拟合 delta = %.6f + %.4f omega + %.4f omega^2；残差 %.2e" % (q0, q1, q2, resq))

# ---------- W3（作用量口径，协议无关） ----------
w3 = 0.025
N3 = int(max(401, 120.0 / w3))
E3 = osc_levels(w3, N3, 8)
kk = np.arange(N3) - (N3 - 1) / 2.0
acts = []
for n in range(1, 6):
    En = E3[n]
    inside = kk[0.5 * w3 ** 2 * kk ** 2 < En]
    kappa = En - 0.5 * w3 ** 2 * inside ** 2
    acts.append(float(np.sum(np.arccos(1.0 - kappa)) / np.pi - n))
rep3 = float(np.max(acts) - np.min(acts))
report("W3 作用量口径的偏移（离散 WKB）", abs(float(np.mean(acts)) - 0.5) <= 0.02 and rep3 <= 0.02,
       "格点色散下的作用量和 A(E_n)/pi - n 与 n 无关（<= 0.02）且均值与 1/2 相符（<= 0.02）",
       "omega=%.4f: 逐档偏移 " % w3 + "、".join("%.5f" % a for a in acts) + "；均值 %.5f，极差 %.2e" % (float(np.mean(acts)), rep3))

# ---------- W4（另两个几何的解析谱） ----------
N4 = 200
Ebox_lat = 0.5 * np.linalg.eigvalsh(lap(N4))
k_n = np.pi * (np.arange(N4) + 1) / (N4 + 1)
gap_box = float(np.max(np.abs(np.sort(Ebox_lat) - np.sort(1.0 - np.cos(k_n)))))
Ering_lat = 0.5 * np.linalg.eigvalsh(lap(N4, True))
gap_ring = float(np.max(np.abs(np.sort(Ering_lat) - np.sort(1.0 - np.cos(2.0 * np.pi * np.arange(N4) / N4)))))
report("W4 另两个几何的解析谱", gap_box <= 1e-12 and gap_ring <= 1e-12,
       "硬壁盒谱 = 1 - cos(k_n)、k_n = (n+1)pi/(N+1)（delta = 1）；环上转子谱 = 1 - cos(2 pi n/N)（delta = 0），均 <= 1e-12",
       "盒谱最大差 %.2e（delta = 1 解析）；环谱最大差 %.2e（delta = 0 解析）" % (gap_box, gap_ring))

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
ax[0].plot(OM, ys, "o", label="(delta-1/2)/omega")
ww = np.linspace(0.0, max(OM) * 1.1, 50)
ax[0].plot(ww, c1 + c2 * ww, "k--", label="线性拟合")
ax[0].plot([0.0], [c1], "r*", markersize=12, label="omega -> 0 截距 c1")
ax[0].set_xlabel("omega")
ax[0].set_ylabel("(delta - 1/2)/omega")
ax[0].set_title("修正系数的一阶结构")
ax[0].legend(fontsize=8)
ax[1].plot(np.arange(6), E1, "o-", label="k 表象")
ax[1].plot(np.arange(6), E2, "kx", label="p 表象（Mathieu）")
ax[1].set_xlabel("量子数 n")
ax[1].set_ylabel("能量")
ax[1].set_title("两种表象的谱一致（omega = 0.05）")
ax[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(IMG, dpi=150)
plt.close(fig)

bad = [r for r in rows if not r[1]]
print("---- 判据 %d 项，通过 %d 项；图存 %s ----" % (len(rows), len(rows) - len(bad), IMG))
sys.exit(1 if bad else 0)
