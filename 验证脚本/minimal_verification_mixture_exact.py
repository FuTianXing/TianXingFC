# -*- coding: utf-8 -*-
r"""
minimal_verification_mixture_exact.py

P4-16 第一步第四轮：混合分布的精确形式。
第 206 条判定门槛附近两族（von Mises 与 wrapped Gaussian）皆失效，本轮给出混合形式并检验：
  p(delta) = w * vonMises(kappa_V) + (1 - w) * (1 / 2 pi)
即"锁定分量 + 脱锁（相位均匀）分量"。该形式的矩为
  <cos m delta> = w * I_m(kappa_V) / I_0(kappa_V)     （m >= 1，均匀分量不贡献）
故两个独立比值 <cos 2d>/<cos d> 与 <cos 3d>/<cos 2d> 必须指向同一个 kappa_V（非平凡判据），
脱锁权重为 w = <cos d> / A_1(kappa_V)。

判据与期望：
  M1 低频档混合式成立：双比值解出的 kappa_V 相符（<= 5%）
  M2 高频档混合式失效：双比值偏差 >= 30% 且 w > 1
  M3 中档过渡：w > 1 或矩预测偏差 >= 20%
  M4 低频档矩预测：<cos 2 delta> 与 w A_2(kappa_V) 相符（<= 5%）
  M5 接口登记（脱锁分量的结构与 w 的解析律）
"""
import sys
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\mixture_exact.png"
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


def Am(m, kV):
    return bessel_i(m, kV) / bessel_i(0, kV)


def solve_kV(ratio, m1, m2):
    lo, hi = 1e-6, 20.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if Am(m2, mid) / Am(m1, mid) < ratio:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def run(gamma, rng):
    w = gamma * rng.normal(size=N)
    th = rng.uniform(-np.pi, np.pi, N)
    acc = []
    for n in range(STEPS):
        z = np.mean(np.exp(1j * th))
        phi = np.angle(z)
        if n >= BURN:
            d = (th - phi + np.pi) % (2 * np.pi) - np.pi
            acc.append([float(np.mean(np.cos(m * d))) for m in (1, 2, 3)])
        th = th + DT * (w + KAPPA * (phi - th)) + np.sqrt(DT) * SIG * rng.normal(size=N)
        th = (th + np.pi) % (2 * np.pi) - np.pi
    return np.mean(np.array(acc), axis=0)


rng = np.random.default_rng(SEED)
g1 = []
for rt in (0.4, 0.8, 1.2):
    c = run(rt * KAPPA, rng)
    kV2 = solve_kV(c[1] / c[0], 1, 2)
    kV3 = solve_kV(c[2] / c[1], 2, 3)
    g1.append((rt, c[0], c[1], c[2], kV2, kV3, abs(kV2 - kV3) / kV2))
lo = g1[0]
hi = g1[2]
report("M1 低频档混合式成立", lo[6] <= 0.05,
       "低失配档（0.4）双比值解出的 kappa_V 相符（<= 5%）",
       "由二阶 %.4f、由三阶 %.4f（相对 %.2e）" % (lo[4], lo[5], lo[6]))

g2 = []
for rt, c1, c2, c3, kV2, kV3, dev in g1:
    kV = 0.5 * (kV2 + kV3)
    wgt = c1 / Am(1, kV)
    pred = wgt * Am(2, kV)
    g2.append((rt, wgt, c2, pred, abs(c2 - pred) / pred))
report("M2 高频档混合式失效", hi[6] >= 0.30 and g2[2][1] > 1.0,
       "高失配档（1.2）双比值偏差 >= 30% 且解出的 w > 1（违反概率解释）",
       "双比值相对偏差 %.2e；w = %.4f" % (hi[6], g2[2][1]))

report("M3 中档过渡", g2[1][1] > 1.0 or abs(g2[1][2] - g2[1][3]) / g2[1][3] >= 0.20,
       "中档（0.8）上 w > 1 或矩预测偏差 >= 20%（报出数值）",
       "w = %.4f；矩预测偏差 %.2e" % (g2[1][1], g2[1][4]))

report("M4 低频档矩预测", g2[0][4] <= 0.05,
       "低频档（0.4）<cos 2 delta> 与 w A_2(kappa_V) 相符（<= 5%）",
       "实测 %.4f 对预测 %.4f（相对 %.2e）" % (g2[0][2], g2[0][3], g2[0][4]))

print("[读数] 混合权重：w = %.4f（0.4 档）、%.4f（0.8 档）、%.4f（1.2 档）；高频档 w 超过 1 说明均匀（脱锁）分量不适用"
      % (g2[0][1], g2[1][1], g2[2][1]))
print("[登记] M5：脱锁分量的结构（均匀分量已被否定，改为双 von Mises 或带结构的宽分量）与 w(gamma/kappa) 的解析律，另立一轮；本轮只登记，不作判据")

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
rts = [rt for rt, _, _, _, _, _, _ in g1]
ax[0].plot(rts, [c2 for _, _, c2, _, _, _, _ in g1], "o", label="实测 <cos 2 delta>")
ax[0].plot(rts, [pr for _, _, _, pr, _ in g2], "k--", label="混合式预测")
ax[0].plot(rts, [c3 for _, _, _, c3, _, _, _ in g1], "s", ms=4, label="实测 <cos 3 delta>")
ax[0].set_xlabel("gamma / kappa")
ax[0].set_ylabel("矩")
ax[0].set_title("混合式对二阶与三阶矩的覆盖")
ax[0].legend(fontsize=8)
ax[1].plot(rts, [wg for _, wg, _, _, _ in g2], "o-", label="混合权重 w")
ax[1].axhline(1.0, ls=":", c="k", lw=1)
ax[1].set_xlabel("gamma / kappa")
ax[1].set_ylabel("w")
ax[1].set_title("脱锁权重随失配比的下降")
ax[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(IMG, dpi=150)
plt.close(fig)

bad = [r for r in rows if not r[1]]
print("---- 判据 %d 项，通过 %d 项；图存 %s ----" % (len(rows), len(rows) - len(bad), IMG))
sys.exit(1 if bad else 0)
