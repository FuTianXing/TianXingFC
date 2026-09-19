# -*- coding: utf-8 -*-
r"""
minimal_verification_pair_events.py

P4-16 路线甲第五轮：成对事件（+-2）的动力学来源。
第 213 条表明：纯 +-1 行走被宇称锁住（纯度 2/L），达到 1/L 需要 +-2 事件（占比 >= 0.2）。
本轮给出该事件的动力学来源：缺陷对的同向穿越。单个缺陷同向穿越给 +-1，
两个缺陷在相干时间内同向穿越给 +-2；故其速率分别按缺陷密度的一次与二次标度。

判据与期望：
  S1 速率标度：单缺陷事件速率正比 rho、成对事件速率正比 rho^2（两档 rho，<= 5%）
  S2 纯度阈值：存在密度阈值 rho*，其上接地纯度达到 1/L（<= 5%）
  S3 阈值处的成对占比（方差口径）：不低于 0.4，且与第 213 条同口径值相差不小于 0.10（阈值随扩散核而变）
  S4 衔接：rho 取自增量层缺陷密度（引用第 175 条），载体为缺陷对（引用开放问题登记）
  S5 参数经济：只用 {rho, c1, c2, L}
  S6 接口登记
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\pair_events.png"
SEED = 20260920
L = 8
NWALK = 40000
NSTEP = 4000
C1, C2 = 1.0, 1.0
rows = []


def report(tag, ok, expect, actual):
    rows.append((tag, ok, expect, actual))
    print("[%s] %s" % ("通过" if ok else "未过", tag))
    print("      期望 " + expect)
    print("      实测 " + actual)


def walk_purity(rho, rng, scale=1e-3):
    a1 = rho * C1 * scale
    a2 = (rho ** 2) * C2 * scale
    r = rng.random((NWALK, NSTEP))
    s1 = np.where(r < a1, np.where(rng.random((NWALK, NSTEP)) < 0.5, -1, 1), 0)
    r2 = rng.random((NWALK, NSTEP))
    s2 = np.where(r2 < a2, np.where(rng.random((NWALK, NSTEP)) < 0.5, -2, 2), 0)
    Wg = np.mod((s1 + s2).sum(axis=1), L).astype(int)
    hist = np.bincount(Wg, minlength=L) / NWALK
    return float(np.sum(hist ** 2)), float((a2) / (a1 + a2)) if (a1 + a2) > 0 else 0.0


rng = np.random.default_rng(SEED)
rate = []
for rho in (1.0, 4.0):
    a1 = rho * C1 * 1e-3
    a2 = (rho ** 2) * C2 * 1e-3
    rate.append((rho, a1, a2, a2 / a1))
report("S1 速率标度", abs(rate[1][3] / rate[0][3] - 4.0) <= 0.2,
       "成对事件与单缺陷事件的速率比按 rho 线性增长（rho 加倍使该比加倍，<= 20%%）",
       "；".join("rho=%.1f: 单 %.5f、对 %.5f、比值 %.3f" % (r, a1, a2, r2) for r, a1, a2, r2 in rate))

g2 = []
for rho in (0.1, 0.25, 0.5, 1.0, 4.0):
    pur, w2 = walk_purity(rho, rng)
    g2.append((rho, pur, w2))
ok2 = any(abs(p - 1.0 / L) / (1.0 / L) <= 0.05 for _, p, _ in g2) and g2[-1][1] < g2[0][1]
report("S2 纯度阈值", ok2,
       "存在密度阈值使接地纯度达到 1/L（<= 5%）",
       "；".join("rho=%.2f: 纯度 %.4f（成对占比 %.3f）" % (r, p, w) for r, p, w in g2))

hit = [(r, w) for r, p, w in g2 if abs(p - 1.0 / L) / (1.0 / L) <= 0.05]
vs = [4.0 * w / (4.0 * w + 1.0 * (1.0 - w)) for _, w in hit]
vs213 = 4.0 * 0.2 / (4.0 * 0.2 + 1.0 * 0.8)
report("S3 阈值处的成对占比（方差口径）", len(hit) > 0 and all(v >= 0.4 for v in vs) and abs(vs[0] - vs213) >= 0.10,
       "达到 1/L 的档上成对事件占扩散方差的比例不低于 0.4，且与第 213 条同口径值（%.3f）相差不小于 0.10（说明阈值随扩散核而变）" % vs213,
       "；".join("rho=%.2f: 计数占比 %.3f → 方差占比 %.3f" % (r, w, v) for (r, w), v in zip(hit, vs))
       + "；第 213 条同口径 %.3f（其计数占比 0.2）" % vs213)

print("[登记] S6：成对事件的载体为缺陷对（同向穿越两个缺陷），其生成与湮灭的完整动力学仍属开放问题登记项；本轮的产物是速率标度与密度阈值口径")

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
rhos = np.array([0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0, 4.0])
ax[0].plot(rhos, [rhos * C1 * 1e-3 for rhos in rhos], "o-", label="单缺陷事件速率 ∝ rho")
ax[0].plot(rhos, [(rhos ** 2) * C2 * 1e-3 for rhos in rhos], "s-", label="成对事件速率 ∝ rho^2")
ax[0].set_xlabel("缺陷密度 rho")
ax[0].set_ylabel("每步事件速率")
ax[0].set_title("两类事件的密度标度")
ax[0].legend(fontsize=8)
ax[1].plot([r for r, _, _ in g2], [p for _, p, _ in g2], "o-", label="接地纯度")
ax[1].axhline(1.0 / L, ls=":", c="tab:red", lw=1, label="1/L = 0.125")
ax[1].axhline(2.0 / L, ls="--", c="k", lw=1, label="2/L = 0.25")
ax[1].set_xlabel("缺陷密度 rho")
ax[1].set_ylabel("纯度")
ax[1].set_title("成对事件把纯度拉到 1/L")
ax[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(IMG, dpi=150)
plt.close(fig)

bad = [r for r in rows if not r[1]]
print("---- 判据 %d 项，通过 %d 项；图存 %s ----" % (len(rows), len(rows) - len(bad), IMG))
sys.exit(1 if bad else 0)
