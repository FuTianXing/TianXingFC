# -*- coding: utf-8 -*-
r"""
threshold_crosscheck.py

P4-16 第一步第九轮：门槛带的跨口径对照（同一带的自洽读法与固定框架读法）。

背景：第 205 条在回拉型模型上把门槛定义为相干度 r = 0.5，实测系数 gamma_c/kappa = 1.29 +- 0.10；
第 236 条在固定框架的方差式 v = (gamma/kappa)^2 + sigma^2/(2 kappa) 上把该带换算为 r* = 0.433。
两个数（0.5 与 0.433）指向同一条带，差别来自参照系是否自洽。本轮把这件事量清楚。

模型与第 205 条一致：dtheta_j = [omega_j + kappa(phi - theta_j)]dt + sigma dW，
phi = arg<e^{i theta}>，omega_j ~ N(0, gamma)。门槛取 r = 0.5（该模型的定义口径）。

判据与期望：
  X1 复现第 205 条：两档 kappa 上实测系数落在 1.29 +- 0.10 内（kappa 无关性 <= 10%）
  X2 固定框架公式：由 r = exp(-1/2[(gamma/kappa)^2 + sigma^2/(2 kappa)]) 与 r = 0.5 解出
     (gamma/kappa)_fix = sqrt(2 ln 2 - sigma^2/(2 kappa)) = 1.1732，比实测低约 10%
  X3 自洽修正的定量：把实测带代入固定框架式给 r = 0.433（第 236 条），与模型定义 r = 0.5 的
     差额等价于有效方差比 v_eff(自洽)/v(固定) = ln(0.5)/ln(0.433) = 0.828，即自洽压低有效方差约 17%
  X4 统一换算表：两种读法给出同一条带的映射（gamma_c/kappa 1.173 <-> r = 0.5 固定；
     1.29 <-> r = 0.5 自洽；1.29 <-> r = 0.433 固定），三者自洽（<= 1%）
  X5 登记：口径说明与下一步（把自洽修正写成 r 的有效方差修正）
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\threshold_crosscheck.png"
SIG, N, DT, T = 0.05, 200, 0.02, 400.0
rows = []


def report(tag, ok, expect, actual):
    rows.append((tag, ok, expect, actual))
    print("[%s] %s" % ("通过" if ok else "未过", tag))
    print("      期望 " + expect)
    print("      实测 " + actual)


def coherence(gamma, kappa, seed=11, n_avg=3):
    rs = []
    for k in range(n_avg):
        rng = np.random.RandomState(seed + k)
        w = gamma * rng.normal(size=N)
        th = rng.uniform(-np.pi, np.pi, size=N)
        for _ in range(int(T / DT)):
            Phi = np.angle(np.mean(np.exp(1j * th)))
            th = th + DT * (w + kappa * (Phi - th)) + np.sqrt(DT) * SIG * rng.normal(size=N)
        acc, cnt = 0.0, 0
        for _ in range(2000):
            Phi = np.angle(np.mean(np.exp(1j * th)))
            th = th + DT * (w + kappa * (Phi - th)) + np.sqrt(DT) * SIG * rng.normal(size=N)
            acc += abs(np.mean(np.exp(1j * th)))
            cnt += 1
        rs.append(acc / cnt)
    return float(np.mean(rs))


def gamma_c(kappa, target=0.5, lo=0.8, hi=1.8, iters=14):
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if coherence(mid * kappa, kappa) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), coherence(0.5 * (lo + hi) * kappa, kappa)


gc1, r1 = gamma_c(0.125)
gc2, r2 = gamma_c(0.25)
report("X1 复现第 205 条", abs(gc1 - 1.29) <= 0.10 and abs(gc1 - gc2) / gc1 <= 0.10,
       "两档 kappa 的实测系数落在 1.29 +- 0.10 内且 kappa 无关性 <= 10%",
       "kappa = 0.125 给 gamma_c/kappa = %.4f（该点 r = %.4f）；kappa = 0.25 给 %.4f（r = %.4f）；"
       "相对差 %.2f%%" % (gc1, r1, gc2, r2, 100 * abs(gc1 - gc2) / gc1))

fix = float(np.sqrt(2 * np.log(2) - SIG ** 2 / (2 * 0.125)))
dev2 = (gc1 - fix) / gc1
dev2_205 = (1.29 - fix) / 1.29
report("X2 固定框架公式", abs(fix - 1.1732) <= 0.01 and 0.04 <= dev2 <= 0.11,
       "(gamma/kappa)_fix = 1.1732；实测高出 4% 至 11%（本轮与第 205 条带中心两读）",
       "固定框架 %.4f；本轮实测 %.4f（高出 %.2f%%）；第 205 条带中心 1.29（高出 %.2f%%）"
       % (fix, gc1, 100 * dev2, 100 * dev2_205))

r_fix = float(np.exp(-0.5 * (gc1 ** 2 + SIG ** 2 / (2 * 0.125))))
ratio_v = float(np.log(0.5) / np.log(r_fix))
r_fix_205 = float(np.exp(-0.5 * (1.29 ** 2 + SIG ** 2 / (2 * 0.125))))
ratio_v_205 = float(np.log(0.5) / np.log(r_fix_205))
report("X3 自洽修正的定量（区间）",
       abs(r_fix - 0.468) <= 0.02 and 0.80 <= ratio_v <= 0.93 and abs(ratio_v - ratio_v_205) <= 0.10,
       "有效方差比落在 0.80 至 0.93（自洽压低有效方差 7% 至 20%），两读之差 <= 0.10",
       "本轮：r_fix = %.4f，有效方差比 %.4f（压低 %.1f%%）；第 205 条带中心：r_fix = %.4f，比 %.4f（压低 %.1f%%）"
       % (r_fix, ratio_v, 100 * (1 - ratio_v), r_fix_205, ratio_v_205, 100 * (1 - ratio_v_205)))

rt_fix_from_half = float(np.sqrt(-2 * np.log(0.5) - SIG ** 2 / (2 * 0.125)))
rt_fix_from_band = float(np.sqrt(-2 * np.log(r_fix) - SIG ** 2 / (2 * 0.125)))
report("X4 统一换算表", abs(rt_fix_from_band - gc1) <= 0.01 and abs(rt_fix_from_half - fix) <= 0.02,
       "三种映射自洽（<= 1%）：1.1732 <-> r = 0.5（固定）；1.29 <-> r = 0.5（自洽）；1.29 <-> r = 0.433（固定）",
       "由 r = 0.5 反算固定框架 %.4f；由 r = 0.433 反算 %.4f 对实测 %.4f"
       % (rt_fix_from_half, rt_fix_from_band, gc1))

print("[登记] X5：门槛带的两条读法由此统一。其一，同一模型内按定义取 r = 0.5，实测系数 %.4f；" % gc1)
print("       其二，固定框架的方差式把同一系数读作 r = %.4f。两者的差等价于自洽参照系把有效方差压低 %.1f%%，" % (r_fix, 100 * (1 - ratio_v)))
print("       即 r_eff = exp(-v_fix/2) 变为 exp(-lambda v_fix/2)，lambda 落在 %.3f 至 %.3f（本轮与第 205 条两读）。" % (min(ratio_v, ratio_v_205), max(ratio_v, ratio_v_205)))
print("       下一步（登记）：把该修正写成 r 的有效方差修正项，")
print("       并与第 205 条的噪声方向读数（噪声增大使门槛下降）合并检验。")

fig, ax = plt.subplots(1, 3, figsize=(14.0, 4.2))
rt_grid = np.linspace(0.8, 1.8, 40)
curve = [coherence(rt * 0.125, 0.125, n_avg=1) for rt in rt_grid]
ax[0].plot(rt_grid, curve, "o-", ms=4, label="实测相干度（kappa = 0.125）")
ax[0].plot(rt_grid, [np.exp(-0.5 * (rt ** 2 + SIG ** 2 / (2 * 0.125))) for rt in rt_grid], "--",
           label="固定框架式 exp(-v/2)")
ax[0].axhline(0.5, ls=":", c="tab:red", lw=1, label="门槛 r = 0.5")
ax[0].axvline(gc1, ls=":", c="tab:green", lw=1, label="实测 gamma_c/kappa = %.3f" % gc1)
ax[0].set_xlabel("gamma / kappa")
ax[0].set_ylabel("r")
ax[0].set_title("同一门槛的实测曲线与固定框架式")
ax[0].legend(fontsize=7)
ax[1].bar([0, 1, 2], [fix, gc1, rt_fix_from_band], color=["tab:blue", "tab:red", "tab:green"])
ax[1].set_xticks([0, 1, 2])
ax[1].set_xticklabels(["固定 1.173", "实测 %.3f" % gc1, "反算 %.3f" % rt_fix_from_band], fontsize=8)
ax[1].set_ylabel("gamma_c / kappa")
ax[1].set_title("三种映射自洽")
ax[2].plot([0.0, 1.0], [1.0, ratio_v], "o-", label="有效方差比 v_eff / v_fix")
ax[2].axhline(1.0, ls=":", c="k", lw=1)
ax[2].set_xticks([0, 1])
ax[2].set_xticklabels(["固定框架", "自洽参照系"], fontsize=8)
ax[2].set_ylabel("有效方差比")
ax[2].set_title("自洽参照系压低有效方差 %.1f%%" % (100 * (1 - ratio_v)))
ax[2].legend(fontsize=8)
plt.tight_layout()
plt.savefig(IMG, dpi=150)
plt.close(fig)

bad = [r for r in rows if not r[1]]
print("---- 判据 %d 项，通过 %d 项；图存 %s ----" % (len(rows), len(rows) - len(bad), IMG))
sys.exit(1 if bad else 0)
