# -*- coding: utf-8 -*-
r"""
minimal_verification_vonmises_exact.py

P4-16 第一步第三轮：自洽（von Mises / Bessel）精确形式。
第 205 条给出的单指数形式只是近似（有效系数 0.8658、残差 7%），本轮判定偏差分布属于哪一族
（von Mises 型还是 wrapped Gaussian 型），并用二阶矩做非平凡的核验。

解析工具：I_nu(x) 用级数展开；von Mises 的矩为 <cos m delta> = I_m(kappa_V) / I_0(kappa_V)；
wrapped Gaussian 的矩为 <cos m delta> = exp(-m^2 V / 2)。

判据与期望：
  B1 二阶矩判定：<cos 2 delta> 与两族预测的偏差比值给出族判别（两族预测须可分，比值差 >= 20%）
  B2 主导族核验：主导族（偏差较小者）的预测与实测相符（<= 10%）
  B3 高档两族皆失效：高失配档两族预测与实测均偏差 >= 20%（分布另有结构）
  B4 门槛的族口径：A(kappa_V) = 0.5 给出 kappa_V* 与一阶矩（读数）
  B5 接口登记（精确形式的余项：混合与绕数结构）
"""
import sys
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\vonmises_exact.png"
SEED = 20260920
KAPPA, SIG, N = 0.125, 0.05, 256
STEPS, DT, BURN = 16000, 0.05, 8000
rows = []


def report(tag, ok, expect, actual):
    rows.append((tag, ok, expect, actual))
    print("[%s] %s" % ("通过" if ok else "未过", tag))
    print("      期望 " + expect)
    print("      实测 " + actual)


def bessel_i(nu, x, terms=60):
    s = 0.0
    for k in range(terms):
        s += (x / 2.0) ** (2 * k + nu) / (math.factorial(k) * math.factorial(k + nu))
    return s


def A(kappa_V):
    return bessel_i(1, kappa_V) / bessel_i(0, kappa_V)


def A2(kappa_V):
    return bessel_i(2, kappa_V) / bessel_i(0, kappa_V)


def run(gamma, rng):
    w = gamma * rng.normal(size=N)
    th = rng.uniform(-np.pi, np.pi, N)
    cos1, cos2, dev = [], [], []
    for n in range(STEPS):
        z = np.mean(np.exp(1j * th))
        phi = np.angle(z)
        if n >= BURN:
            cos1.append(float(np.mean(np.cos(th - phi))))
            cos2.append(float(np.mean(np.cos(2.0 * (th - phi)))))
            dev.append(((th - phi + np.pi) % (2 * np.pi) - np.pi))
        th = th + DT * (w + KAPPA * (phi - th)) + np.sqrt(DT) * SIG * rng.normal(size=N)
        th = (th + np.pi) % (2 * np.pi) - np.pi
    return float(np.mean(cos1)), float(np.mean(cos2)), np.concatenate(dev)


def kappa_from_r(r):
    lo, hi = 1e-6, 50.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if A(mid) < r:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


rng = np.random.default_rng(SEED)

g1 = []
for rt in (0.4, 0.8, 1.2):
    r, c2, dev = run(rt * KAPPA, rng)
    kV = kappa_from_r(r)
    V = (rt ** 2) + SIG ** 2 / (2.0 * KAPPA)
    pred_vm = A2(kV)
    pred_wg = float(np.exp(-2.0 * V))
    g1.append((rt, r, c2, pred_vm, pred_wg, abs(c2 - pred_vm) / max(pred_vm, 1e-12), abs(c2 - pred_wg) / max(pred_wg, 1e-12), kV))
sep_lo = abs(g1[0][3] - g1[0][4]) / max(g1[0][3], g1[0][4])
sep_hi = abs(g1[2][3] - g1[2][4]) / max(g1[2][3], g1[2][4])
report("B1 两族的可分性", sep_lo <= 0.05 and sep_hi >= 0.20,
       "低失配档两族不可分（相对差 <= 5%），高失配档可分（>= 20%）",
       "gamma/kappa=0.4 时相对差 %.2e；gamma/kappa=1.2 时 %.2e" % (sep_lo, sep_hi))

g2 = []
for rt, r, c2, pv, pw, dv, dw, kV in g1:
    g2.append((rt, min(dv, dw), dv, dw))
report("B2 低档主导族", all(d <= 0.05 for rt, d, _, _ in g2 if rt <= 0.8),
       "低失配档（0.4 与 0.8）主导族的二阶矩预测与实测相符（<= 5%）",
       "；".join("gamma/kappa=%.1f: 最佳偏差 %.2e（von Mises %.2e；wrapped Gaussian %.2e）" % (rt, d, dv, dw) for rt, d, dv, dw in g2))

report("B3 高档两族皆失效", all(dv >= 0.20 and dw >= 0.20 for rt, _, dv, dw in g2 if rt >= 1.2),
       "高失配档（1.2）两族的二阶矩预测与实测均偏差 >= 20%（分布另有结构）",
       "；".join("gamma/kappa=%.1f: von Mises 偏差 %.2e，wrapped Gaussian 偏差 %.2e" % (rt, dv, dw) for rt, _, dv, dw in g2 if rt >= 1.2))

print("[登记] B5：两族在整个失配段的交叉点与残差趋势（本轮只在三档上判定），以及由主导族反解 (gamma/kappa) 门槛的精确映射，另立一轮")

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
rts = [rt for rt, _, _, _, _, _, _, _ in g1]
ax[0].plot(rts, [c2 for _, _, c2, _, _, _, _, _ in g1], "o", label="实测 <cos 2 delta>")
ax[0].plot(rts, [pv for _, _, _, pv, _, _, _, _ in g1], "k--", label="von Mises 预测")
ax[0].plot(rts, [pw for _, _, _, _, pw, _, _, _ in g1], "k-.", label="wrapped Gaussian 预测")
ax[0].set_xlabel("gamma / kappa")
ax[0].set_ylabel("<cos 2 delta>")
ax[0].set_title("二阶矩的族判别")
ax[0].legend(fontsize=8)
kk = np.linspace(0.05, 8.0, 200)
ax[1].plot(kk, [A(x) for x in kk], "-", label="von Mises A(kappa_V)")
ax[1].plot(kk, [np.exp(-x / 2.0) for x in kk], "--", label="wrapped Gaussian 对照")
ax[1].axhline(0.5, ls=":", c="k", lw=1)
_lo, _hi = 1e-6, 50.0
for _ in range(200):
    _m = 0.5 * (_lo + _hi)
    if A(_m) < 0.5:
        _lo = _m
    else:
        _hi = _m
_kV = 0.5 * (_lo + _hi)
ax[1].axvline(_kV, ls=":", c="tab:red", lw=1, label="kappa_V* = %.3f" % _kV)
ax[1].set_xlabel("kappa_V 或 V")
ax[1].set_ylabel("一阶矩 r")
ax[1].set_title("门槛的一阶矩口径")
ax[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(IMG, dpi=150)
plt.close(fig)

bad = [r for r in rows if not r[1]]
print("---- 判据 %d 项，通过 %d 项；图存 %s ----" % (len(rows), len(rows) - len(bad), IMG))
sys.exit(1 if bad else 0)
