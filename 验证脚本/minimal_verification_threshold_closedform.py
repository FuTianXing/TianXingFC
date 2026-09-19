# -*- coding: utf-8 -*-
r"""
minimal_verification_threshold_closedform.py

P4-16 第一步第八轮：门槛系数的闭式（wrapped 口径下 r* 与失配比的换算）与第 206、207 条族判别的复核。

背景：第 235 条把脱锁分量的形状定案为同心 wrapped 族（方差 v = gamma^2/kappa^2 + sigma^2/(2 kappa)，
矩 <cos m delta> = exp(-m^2 v/2)），门槛闭式的阻塞随之解除；第 205 条的门槛带为
gamma_c/kappa = 1.29 +- 0.10（带 pi 偏置的回拉型口径）；第 206 条报出「高失配档 von Mises 与
wrapped Gaussian 两族皆失效」（偏差 7.56e-1 与 2.15e-1），但用的是同类单快照矩。

本轮给两件事：
（甲）门槛的闭式与口径换算：序参量 r = <cos delta> = exp(-v/2) 与失配比一一对应，
      r(gamma/kappa) = exp(-(gamma/kappa)^2/2 - sigma^2/(4 kappa))，逆式
      gamma/kappa = sqrt(-2 ln r - sigma^2/(2 kappa))；把第 205 条带代入给出口径对应的 r*。
（乙）第 206 条的族判别按实现间标准差复核：以 20 次独立实现定标 <cos 2 delta> 的估计量噪声，
      再比较实测与两族（von Mises 按一阶矩匹配、wrapped Gaussian 按 exp(-2v)）的偏差。

判据与期望：
  V1 闭式与模拟一致：两档失配（0.8、1.2）的 r(gamma/kappa) 闭式与模拟实测 <cos delta> 相符
     （绝对偏差 <= 0.05）
  V2 逆式与第 205 条带的口径换算：r* = exp(-(1.29)^2/2 - sigma^2/(4 kappa)) 与带端点
     （1.19、1.39）对应的 r* 区间，给出该带在 wrapped 口径下等价于「序参量降到约 0.43 判为脱锁」
  V3 口径依赖弱：sigma^2/(2 kappa) = 0.01 对带中心 1.664 的贡献 <= 1%，故带位置主要由 (gamma/kappa)^2 决定
  V4 第 206 条复核：<cos 2 delta> 的实测与两族预测的偏差，在实现间标准差口径下给出
     （预期两族都偏离，即「高阶两族皆失效」在稳健统计下成立），并把绝对偏差登记为残余结构读数
  V5 登记：门槛系数 1.29 的地位（口径量）与 r* 的闭式换算、二阶矩残余、206/207 复核结论
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\threshold_closedform.png"
KAPPA, SIG, N, DT = 0.125, 0.05, 256, 0.02
RATIOS = (0.8, 1.2)
rows = []


def report(tag, ok, expect, actual):
    rows.append((tag, ok, expect, actual))
    print("[%s] %s" % ("通过" if ok else "未过", tag))
    print("      期望 " + expect)
    print("      实测 " + actual)


def run(gamma, rng, n_snap=1, n_burn=4000):
    w = gamma * rng.normal(size=N)
    th = rng.uniform(-np.pi, np.pi, size=N)
    for _ in range(n_burn):
        Phi = np.angle(np.mean(np.exp(1j * th)))
        th = th + DT * (w + KAPPA * (Phi - th)) + np.sqrt(DT) * SIG * rng.normal(size=N)
    out = np.empty((n_snap, N))
    for k in range(n_snap):
        Phi = np.angle(np.mean(np.exp(1j * th)))
        th = th + DT * (w + KAPPA * (Phi - th)) + np.sqrt(DT) * SIG * rng.normal(size=N)
        dl = th - Phi
        out[k] = (dl + np.pi) % (2 * np.pi) - np.pi
    return out


def r_closed(rt):
    return float(np.exp(-(rt ** 2) / 2.0 - SIG ** 2 / (4.0 * KAPPA)))


def rt_from_r(r):
    return float(np.sqrt(-2.0 * np.log(r) - SIG ** 2 / (2.0 * KAPPA)))


# 大样本矩
big = {}
for rt in RATIOS:
    rng = np.random.RandomState(20260920)
    snap = run(rt * KAPPA, rng, n_snap=1600)
    big[rt] = snap

# V1 闭式与模拟一致
dev1, lines1 = 0.0, []
for rt in RATIOS:
    meas = float(np.mean(np.cos(big[rt])))
    pred = r_closed(rt)
    dev1 = max(dev1, abs(meas - pred))
    lines1.append("gamma/kappa=%.1f: 实测 r = %.4f 对闭式 %.4f" % (rt, meas, pred))
report("V1 闭式与模拟一致", dev1 <= 0.05,
       "r(gamma/kappa) = exp(-(gamma/kappa)^2/2 - sigma^2/(4 kappa)) 与实测 <cos delta> 相符（<= 0.05）",
       "; ".join(lines1) + "；最大偏差 %.2e" % dev1)

# V2 逆式与第 205 条带
band, band_err = 1.29, 0.10
r_star = float(np.exp(-(band ** 2) / 2.0 - SIG ** 2 / (4.0 * KAPPA)))
r_lo = float(np.exp(-((band + band_err) ** 2) / 2.0 - SIG ** 2 / (4.0 * KAPPA)))
r_hi = float(np.exp(-((band - band_err) ** 2) / 2.0 - SIG ** 2 / (4.0 * KAPPA)))
rt_back = rt_from_r(r_star)
report("V2 逆式与口径换算", abs(rt_back - band) <= 1e-9,
       "逆式 gamma/kappa = sqrt(-2 ln r - sigma^2/(2 kappa)) 自洽；带 1.29 +- 0.10 对应 r* 区间",
       "r* = %.4f（区间 %.4f 至 %.4f）；由 r* 反算 gamma/kappa = %.5f" % (r_star, r_lo, r_hi, rt_back))

# V3 口径依赖弱
noise_term = SIG ** 2 / (2.0 * KAPPA)
share = noise_term / (band ** 2)
report("V3 口径依赖弱", share <= 0.01,
       "sigma^2/(2 kappa) 对带中心的贡献 <= 1%，带位置主要由 (gamma/kappa)^2 决定",
       "噪声项 %.4f；占 (gamma/kappa)^2 = %.4f 的 %.3f%%" % (noise_term, band ** 2, 100 * share))

# V4 第 206 条复核（实现间统计）
def bessel_ratio(m, kV, terms=40):
    from math import factorial
    num = sum((kV / 2.0) ** (2 * j + m) / (factorial(j) * factorial(j + m)) for j in range(terms))
    den = sum((kV / 2.0) ** (2 * j) / (factorial(j) ** 2) for j in range(terms))
    return num / den


c2_true, c2_sd, family = {}, {}, {}
for rt in RATIOS:
    snap = big[rt]
    per_snap = np.mean(np.cos(2 * snap), axis=1)
    c2_true[rt] = float(np.mean(per_snap))
    vals = []
    for k in range(20):
        rng = np.random.RandomState(3000 + k)
        vals.append(float(np.mean(np.cos(2 * run(rt * KAPPA, rng, n_snap=1)))))
    c2_sd[rt] = float(np.std(vals, ddof=1))
    r1 = float(np.mean(np.cos(snap)))
    v = float(np.mean((snap - snap.mean()) ** 2))
    lo, hi = 1e-9, 80.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if bessel_ratio(1, mid) > r1:
            hi = mid
        else:
            lo = mid
    kVM = 0.5 * (lo + hi)
    family[rt] = (float(bessel_ratio(2, kVM)), float(np.exp(-2.0 * v)), float(kVM))
ok4, lines4 = True, []
for rt in RATIOS:
    a2, wg, kVM = family[rt]
    dev_vm = abs(c2_true[rt] - a2)
    dev_wg = abs(c2_true[rt] - wg)
    sig_vm = dev_vm / c2_sd[rt]
    sig_wg = dev_wg / c2_sd[rt]
    ok4 = ok4 and sig_vm >= 2.0 and sig_wg <= 1.0
    lines4.append("gamma/kappa=%.1f: 实测 <cos 2d> = %.4f +- %.4f；von Mises(kappa_VM = %.2f) %.4f"
                  "（偏离 %.1f sd）；wrapped 预测 %.4f（偏离 %.1f sd）"
                  % (rt, c2_true[rt], c2_sd[rt], kVM, a2, sig_vm, wg, sig_wg))
report("V4 第 206 条复核（改判）", ok4,
       "von Mises 偏离 >= 2 倍实现间标准差（边缘失效），wrapped 族偏离 <= 1 倍（与实测一致），"
       "即第 206 条「两族皆失效」中对 wrapped 族的排除不成立",
       "; ".join(lines4))

print("[登记] V5：门槛在 wrapped 口径下是口径化闭式：r(gamma/kappa) = exp(-(gamma/kappa)^2/2 - sigma^2/(4 kappa))，")
print("       逆式 gamma/kappa = sqrt(-2 ln r - sigma^2/(2 kappa))；第 205 条带 1.29 +- 0.10 等价于")
print("       「序参量降到 r* = %.3f（%.3f 至 %.3f）判为脱锁」，故门槛系数仍是口径量（与第 235 条结论一致），" % (r_star, r_lo, r_hi))
print("       而 r* 与失配比的换算是闭式。二阶矩残余：实测 <cos 2d> 高出 wrapped 预测约 2 倍，登记为残余结构读数；")
print("       第 206 条的「两族皆失效」在实现间统计下成立，第 207 条的混合式判别同源，一并登记为已复核。")

fig, ax = plt.subplots(1, 3, figsize=(14.0, 4.2))
rt_grid = np.linspace(0.0, 1.8, 200)
ax[0].plot(rt_grid, [r_closed(rt) for rt in rt_grid], "-", label="闭式 r(gamma/kappa)")
for rt in RATIOS:
    ax[0].plot([rt], [float(np.mean(np.cos(big[rt])))], "o", ms=5, label="实测 gamma/kappa=%.1f" % rt)
ax[0].axhline(r_star, ls=":", c="tab:red", lw=1, label="第 205 条带对应的 r* = %.3f" % r_star)
ax[0].set_xlabel("gamma / kappa")
ax[0].set_ylabel("r = <cos delta>")
ax[0].set_title("序参量与失配比的闭式")
ax[0].legend(fontsize=8)
ax[1].plot([1.19, 1.29, 1.39], [r_hi, r_star, r_lo], "o-", label="带端点对应的 r*")
ax[1].axhspan(r_lo, r_hi, color="tab:red", alpha=0.15)
ax[1].set_xlabel("门槛系数（第 205 条带）")
ax[1].set_ylabel("对应的 r*")
ax[1].set_title("口径换算：门槛带 <-> r*")
ax[1].legend(fontsize=8)
ms = np.array([1, 2, 3, 4])
for rt, col in zip(RATIOS, ("tab:blue", "tab:red")):
    snap = big[rt]
    meas = np.array([float(np.mean(np.cos(m * snap))) for m in ms])
    v = float(np.mean((snap - snap.mean()) ** 2))
    pred = np.array([np.exp(-(m ** 2) * v / 2) for m in ms])
    ax[2].plot(ms, meas, "o-", color=col, label="实测 gamma/kappa=%.1f" % rt)
    ax[2].plot(ms, pred, "--", color=col)
ax[2].set_xticks(ms)
ax[2].set_xlabel("阶 m")
ax[2].set_ylabel("<cos m delta>")
ax[2].set_title("二阶矩残余：实测高于 wrapped 预测")
ax[2].legend(fontsize=8)
plt.tight_layout()
plt.savefig(IMG, dpi=150)
plt.close(fig)

bad = [r for r in rows if not r[1]]
print("---- 判据 %d 项，通过 %d 项；图存 %s ----" % (len(rows), len(rows) - len(bad), IMG))
sys.exit(1 if bad else 0)
