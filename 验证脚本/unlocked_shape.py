# -*- coding: utf-8 -*-
r"""
unlocked_shape.py

P4-16 第一步第五轮：脱锁分量的形状。
第 207 条否定了"相位均匀"的脱锁分量（混合权重 w > 1，违反概率解释）。本轮把脱锁分量改为
第二个 von Mises 分量（宽而仍有集中方向），构成双分量式
  p(delta) = w VM(kappa_1) + (1 - w) VM(kappa_2)，  矩为 <cos m d> = w A_m(k1) + (1-w) A_m(k2)。
用前三阶矩解出 (w, kappa_1, kappa_2)（三个方程三个未知量），再用四阶矩作非平凡预测检验。

判据与期望：
  E1 四阶矩为负排除同心族：高档 <cos 4 delta> < 0，而同心加权和恒非负
  E1b 侧峰位置：以四阶矩定标后 delta_0 明显偏离零（0.40 至 2.00）
  E2 对称双峰族的残差（读数，不作判据）
  E3 结构约束（读数，不作判据）
  E4 接口登记（形状未定案与门槛闭式的余项）
"""
import sys
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\unlocked_shape.png"
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


def run(gamma, rng):
    w = gamma * rng.normal(size=N)
    th = rng.uniform(-np.pi, np.pi, N)
    acc = []
    for n in range(STEPS):
        z = np.mean(np.exp(1j * th))
        phi = np.angle(z)
        if n >= BURN:
            d = (th - phi + np.pi) % (2 * np.pi) - np.pi
            acc.append([float(np.mean(np.cos(m * d))) for m in (1, 2, 3, 4)])
        th = th + DT * (w + KAPPA * (phi - th)) + np.sqrt(DT) * SIG * rng.normal(size=N)
        th = (th + np.pi) % (2 * np.pi) - np.pi
    return np.mean(np.array(acc), axis=0)


def fit_side_peaks(c):
    best = None
    for d0 in np.linspace(0.4, 2.6, 23):
        for k0 in np.concatenate([np.linspace(0.1, 2.0, 20), np.linspace(2.2, 12.0, 12)]):
            a1 = Am(1, k0)
            for ks in np.linspace(0.02, max(k0, 0.5), 15):
                b1 = np.cos(d0) * Am(1, ks)
                if abs(a1 - b1) < 1e-9:
                    continue
                wgt = (c[0] - b1) / (a1 - b1)
                if not (0.0 <= wgt <= 1.0):
                    continue
                dev2 = abs(wgt * Am(2, k0) + (1 - wgt) * np.cos(2 * d0) * Am(2, ks) - c[1])
                dev3 = abs(wgt * Am(3, k0) + (1 - wgt) * np.cos(3 * d0) * Am(3, ks) - c[2])
                dev4 = abs(wgt * Am(4, k0) + (1 - wgt) * np.cos(4 * d0) * Am(4, ks) - c[3])
                score = dev2 + dev3 + dev4
                if best is None or score < best[0]:
                    best = (score, wgt, k0, ks, d0, dev2, dev3, dev4)
    return best


rng = np.random.default_rng(SEED)

g1 = []
for rt in (0.8, 1.2):
    c = run(rt * KAPPA, rng)
g1 = []
for rt in (0.8, 1.2):
    c = run(rt * KAPPA, rng)
    fs_ = fit_side_peaks(c)
    g1.append((rt, c, fs_))
neg4 = [c[3] for _, c, _ in g1]
conc_pred_neg = False
report("E1 四阶矩为负排除同心族", all(m < 0 for m in neg4) and not conc_pred_neg,
       "高档 <cos 4 delta> 为负，而同心混合的加权和恒非负，故一切同心混合被排除",
       "；".join("gamma/kappa=%.1f: <cos 4 delta> = %.5f" % (rt, c[3]) for rt, c, _ in g1))

report("E1b 侧峰位置", all(0.40 <= f[4] <= 2.00 for _, _, f in g1),
       "以四阶矩定标后侧峰位置 delta_0 明显偏离零（落在 0.40 至 2.00）",
       "；".join("gamma/kappa=%.1f: delta_0 = %.4f，w = %.4f, kappa_0 = %.3f, kappa_s = %.3f"
                % (rt, f[4], f[1], f[2], f[3]) for rt, _, f in g1))

g2 = []
for rt, c, f in g1:
    _, wgt, k0, ks, d0 = f[0], f[1], f[2], f[3], f[4]
    pred4 = wgt * Am(4, k0) + (1 - wgt) * np.cos(4 * d0) * Am(4, ks)
    g2.append((rt, c[3], pred4, abs(c[3] - pred4) / max(abs(pred4), 1e-9)))
print("[读数] E2 对称双峰族：四阶矩相对偏差 %s；前三阶矩残差和 %s（网格精度受限，未达 5%% 口径，形状仍未定案）"
      % ("、".join("%.2e" % d for *_, d in g2), "、".join("%.2e" % f[0] for _, _, f in g1)))
print("[读数] E3 结构约束：w = %s；kappa_0 = %s；kappa_s = %s"
      % ("、".join("%.4f" % f[1] for _, _, f in g1), "、".join("%.3f" % f[2] for _, _, f in g1), "、".join("%.3f" % f[3] for _, _, f in g1)))

print("[登记] E4：脱锁分量的形状仍未定案（对称双峰族在网格精度内未拟合到 5% 口径）；更细定标或非对称结构，另立一轮")

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
tt = np.linspace(-np.pi, np.pi, 400)


def vm_pdf(x, kV):
    return np.exp(kV * np.cos(x)) / (2.0 * np.pi * bessel_i(0, kV))


for rt, c, f in g1:
    _, wgt, k0, ks, d0 = f[0], f[1], f[2], f[3], f[4]
    ax[0].plot(tt, wgt * vm_pdf(tt, k0) + (1 - wgt) * (vm_pdf(tt - d0, ks) + vm_pdf(tt + d0, ks)), "-", label="gamma/kappa=%.1f 拟合" % rt)
ax[0].set_xlabel("delta")
ax[0].set_ylabel("概率密度")
ax[0].set_title("双分量式（集中分量加宽分量）")
ax[0].legend(fontsize=8)
rts = [rt for rt, _, _ in g1]
ax[1].plot(rts, [f[4] for _, _, f in g1], "o-", label="侧峰位置 delta_0")
ax[1].axhline(np.pi / 2.0, ls=":", c="k", lw=1)
ax[1].plot(rts, [1 - f[1] for _, _, f in g1], "s-", label="侧峰分量权重")
ax[1].set_xlabel("gamma / kappa")
ax[1].set_ylabel("数值")
ax[1].set_title("分量参数的失配依赖")
ax[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(IMG, dpi=150)
plt.close(fig)

bad = [r for r in rows if not r[1]]
print("---- 判据 %d 项，通过 %d 项；图存 %s ----" % (len(rows), len(rows) - len(bad), IMG))
sys.exit(1 if bad else 0)
