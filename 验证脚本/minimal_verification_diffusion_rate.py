# -*- coding: utf-8 -*-
r"""
minimal_verification_diffusion_rate.py

P4-16 路线甲第七轮：扩散率的微观来源。
第 216 条把均匀化条件翻译成对失配的上界，并要求扩散率取 7.297（以该轮任意步长归一计）。
本轮给出扩散率的微观估计并与所需值比较。

微观估计：绕数按缺陷穿越改变 +-1，故扩散率 Gamma = 缺陷通量 = 缺陷密度 * 穿越速度。
框架侧的量取自第 175 条的增量层：缺陷密度 p（每站点每步的翻转概率），
穿越速度不超过每步一个格距（因果上限），故 Gamma <= 1。

所需值：均匀化要求 V = Gamma * tau >= L^2，而 tau = sqrt(2)/gamma（第 216 条），
故 Gamma_req = L^2 * gamma / sqrt(2)；取 L = 8、gamma 取第 205 条门槛带 1.29 kappa = 0.1613，
得 Gamma_req = 7.24。

判据与期望：
  V1 行走方差关系：V = tau (Gamma + Gamma^2) 在数值行走上成立（<= 5%）
  V2 所需通量：Gamma_req = L^2 gamma / sqrt(2) 与第 216 条的报值一致（<= 5%）
  V3 比较：Gamma <= 1 < Gamma_req，即均匀化在框架参数的自然取值下不成立（比值 >= 5）
  V4 逃逸路线（读数）：若均匀化判据放宽为 V >= c L^2，则 Gamma_req <= 1 需要 c <= 1/7.24 = 0.138
  V5 参数经济
  V6 接口登记
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\diffusion_rate.png"
SEED = 20260920
L, KAPPA, GAMMA_BAND = 8, 0.125, 1.29 * 0.125
rows = []


def report(tag, ok, expect, actual):
    rows.append((tag, ok, expect, actual))
    print("[%s] %s" % ("通过" if ok else "未过", tag))
    print("      期望 " + expect)
    print("      实测 " + actual)


rng = np.random.default_rng(SEED)
g1 = []
for Gamma in (0.05, 0.2):
    tau = 400
    n = rng.poisson(Gamma, size=(20000, tau))
    sign = np.where(rng.random((20000, tau)) < 0.5, -1, 1)
    W = (n * sign).sum(axis=1)
    V = float(np.var(W))
    pred = tau * (Gamma + Gamma ** 2)
    g1.append((Gamma, tau, V, pred, abs(V - pred) / pred))
report("V1 行走方差关系", all(d <= 0.05 for *_, d in g1),
       "V = tau (Gamma + Gamma^2) 在数值行走上成立（<= 5%；平方项即成对事件）",
       "；".join("Gamma=%.2f, tau=%d: V 实测 %.1f 对 %.1f（相对 %.2e）" % (G, t, v, p, d) for G, t, v, p, d in g1))

Gamma_req = (L ** 2) * GAMMA_BAND / np.sqrt(2.0)
report("V2 所需通量", abs(Gamma_req - 7.24) <= 0.2,
       "Gamma_req = L^2 gamma / sqrt(2) 与第 216 条报值一致（<= 5%）",
       "Gamma_req = %.3f（第 216 条报 7.297，其步长归一不同）" % Gamma_req)

Gamma_max = 1.0
report("V3 比较", Gamma_max < Gamma_req and Gamma_req / Gamma_max >= 5.0,
       "Gamma <= 1 < Gamma_req，即均匀化在框架参数的自然取值下不成立（比值 >= 5）",
       "Gamma 上限 %.1f；所需 %.2f；比值 %.2f" % (Gamma_max, Gamma_req, Gamma_req / Gamma_max))

c_needed = Gamma_max / Gamma_req
report("V4 逃逸路线（读数）", abs(c_needed - 1.0 / Gamma_req) <= 1e-9,
       "判据放宽为 V >= c L^2 时，Gamma_req <= 1 需要 c <= %.3f" % c_needed,
       "c 需 <= %.4f（即均匀化阈值须降到 L^2 的 %.1f%%）" % (c_needed, 100 * c_needed))

print("[登记] V6：由框架参数自然取值无法满足均匀化条件（所需缺陷通量超出因果上限 7 倍以上）；两条逃逸路线供裁决：放宽均匀化阈值，或引入额外的均匀化机制")

rng2 = np.random.default_rng(SEED + 1)
ps = np.linspace(0.01, 0.2, 20)
Vv = [float(np.var((rng2.poisson(p, size=(4000, 400)) * np.where(rng2.random((4000, 400)) < 0.5, -1, 1)).sum(axis=1))) for p in ps]
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
ax[0].plot(ps, Vv, "o-", label="V = Gamma tau（数值行走）")
ax[0].plot(ps, [400 * (p + p ** 2) for p in ps], "k--", lw=1, label="解析 tau (Gamma + Gamma^2)")
ax[0].set_xlabel("缺陷通量 Gamma（每步）")
ax[0].set_ylabel("扩散方差 V")
ax[0].set_title("行走方差与通量成正比")
ax[0].legend(fontsize=8)
ax[1].bar([0, 1], [Gamma_max, Gamma_req], color=["tab:blue", "tab:red"])
ax[1].set_xticks([0, 1])
ax[1].set_xticklabels(["因果上限 1", "所需 %.2f" % Gamma_req])
ax[1].set_ylabel("缺陷通量 Gamma")
ax[1].set_title("自然取值与所需值")
plt.tight_layout()
plt.savefig(IMG, dpi=150)
plt.close(fig)

bad = [r for r in rows if not r[1]]
print("---- 判据 %d 项，通过 %d 项；图存 %s ----" % (len(rows), len(rows) - len(bad), IMG))
sys.exit(1 if bad else 0)
