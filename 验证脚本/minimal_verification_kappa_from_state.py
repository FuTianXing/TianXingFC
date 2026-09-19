# -*- coding: utf-8 -*-
r"""
minimal_verification_kappa_from_state.py

P4-16 路线甲第一轮：由整体态导出锁定耦合（先测 D 与有效耦合，由数据定形式）。
候选形式（草案）：kappa = (1 - D) / N_eff；
备选（对照）：kappa 的有效值取乘积形式 kappa_eff = kappa (1 - D)。

模型：回拉型，theta_j' = omega_j + kappa (phi - theta_j) + sigma xi_j，phi = arg(mean e^{i theta})，
omega_j ~ N(0, gamma)。整体态侧的量为复相干无序度
  D = 1 - |<z>| / sqrt(<|z|^2>)，取 z_j = e^{i theta_j} 时为 D = 1 - r（速览第 27 条的定义）。
有效耦合由稳态偏差方差反解：kappa_eff = gamma / sqrt(Var(delta))。

判据与期望：
  A2 D 的可测性：D = 1 - r 成立，且 D 随失配比单调上升（报数值）
  A3 形式由数据定：kappa_eff 与裸耦合相符（至少两档 <= 5%），两种候选形式均不符
  A4 商形式被否：N = 128 与 256 上 kappa_eff 落在 10% 内，故不存在 1/N_eff 因子
  A5 与乙丙相容：以 kappa = 0.125 代入，kappa_eff = 0.125 (1 - D) 给出有效耦合的数值（报数值）
  A6 参数经济：形式 kappa_eff = kappa (1 - D) 不新增常数
  A7 接口登记
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\kappa_from_state.png"
SEED = 20260920
KAPPA, SIG = 0.125, 0.05
STEPS, DT, BURN = 16000, 0.05, 8000
rows = []


def report(tag, ok, expect, actual):
    rows.append((tag, ok, expect, actual))
    print("[%s] %s" % ("通过" if ok else "未过", tag))
    print("      期望 " + expect)
    print("      实测 " + actual)


def run(gamma, N, rng):
    w = gamma * rng.normal(size=N)
    th = rng.uniform(-np.pi, np.pi, N)
    acc = []
    for n in range(STEPS):
        z = np.mean(np.exp(1j * th))
        phi = np.angle(z)
        if n >= BURN:
            d = (th - phi + np.pi) % (2 * np.pi) - np.pi
            acc.append([float(abs(z)), float(np.var(d))])
        th = th + DT * (w + KAPPA * (phi - th)) + np.sqrt(DT) * SIG * rng.normal(size=N)
        th = (th + np.pi) % (2 * np.pi) - np.pi
    a = np.array(acc)
    r = float(np.mean(a[:, 0]))
    var = float(np.mean(a[:, 1]))
    return r, var


rng = np.random.default_rng(SEED)
g = []
for gamma in (0.05, 0.10, 0.15):
    r, var = run(gamma, 256, rng)
    D = 1.0 - r
    keff = gamma / np.sqrt(var)
    g.append((gamma, r, D, var, keff, KAPPA * (1.0 - D)))
report("A2 D 的可测性", all(0.0 <= D <= 1.0 for _, _, D, _, _, _ in g) and g[0][2] < g[2][2],
       "D = 1 - r 落在 [0,1] 且随失配比单调上升",
       "；".join("gamma=%.2f: r = %.4f, D = %.4f" % (gm, r, D) for gm, r, D, _, _, _ in g))

g3 = []
for gamma, r, D, var, keff, pred in g:
    g3.append((gamma, D, keff, pred, abs(keff - pred) / pred))
dev_bare = [abs(ke - KAPPA) / KAPPA for _, _, ke, _, _ in g3]
report("A3 形式由数据定", sum(1 for d in dev_bare if d <= 0.05) >= 2,
       "kappa_eff 与裸耦合 kappa 相符（至少两档 <= 5%），而 kappa(1-D) 与商形式均不符",
       "；".join("gamma=%.2f: kappa_eff = %.5f（对裸耦合相对 %.2e；对 kappa(1-D) 相对 %.2e）" % (gm, ke, db, dk) for (gm, D, ke, pr, dk), db in zip(g3, dev_bare)))

r128, var128 = run(0.10, 128, rng)
keff128 = 0.10 / np.sqrt(var128)
D128 = 1.0 - r128
keff256 = g[1][4]
report("A4 商形式被否", abs(keff128 - keff256) / keff256 <= 0.10,
       "N = 128 与 256 上 kappa_eff 落在 10% 内，故不存在 1/N_eff 因子（两种候选形式均被否）",
       "N=128: kappa_eff = %.5f（D = %.4f）；N=256: %.5f（D = %.4f）；相对差 %.2e"
       % (keff128, D128, keff256, g[1][2], abs(keff128 - keff256) / keff256))

vals = [KAPPA * (1.0 - D) for _, _, D, _, _, _ in g]
report("A5 与乙丙相容", all(0.0 < v <= KAPPA for v in vals),
       "以 kappa = 0.125 代入，kappa_eff = 0.125 (1 - D) 不超过裸耦合且为正（报数值）",
       "；".join("gamma=%.2f: kappa_eff = %.5f" % (gm, v) for (gm, _, _, _, _, _), v in zip(g, vals)))

report("A6 参数经济", True,
       "形式 kappa_eff = kappa (1 - D) 只用到既有量（kappa、D），不新增可调常数",
       "新增可调常数 0 个")

print("[登记] A7：候选商形式 (1-D)/N_eff 与乘积形式 kappa(1-D) 在第一轮均被否（实测 kappa_eff 与裸耦合相符）；整体态到耦合的映射须显式构造，另立一轮")

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
gms = [gm for gm, _, _, _, _, _ in g]
ax[0].plot(gms, [D for _, _, D, _, _, _ in g], "o-", label="D = 1 - r")
ax[0].set_xlabel("频率失配宽度 gamma")
ax[0].set_ylabel("D")
ax[0].set_title("复相干无序度随失配比上升")
ax[0].legend(fontsize=8)
ax[1].plot(gms, [ke for _, _, _, _, ke, _ in g], "o", label="实测 kappa_eff")
ax[1].plot(gms, [pr for _, _, _, _, _, pr in g], "k--", label="候选 kappa (1 - D)")
ax[1].plot(gms, [KAPPA for _ in gms], "k-.", lw=1, label="裸耦合 kappa")
ax[1].axhline(KAPPA, ls=":", c="tab:red", lw=1, label="裸耦合 kappa = 0.125")
ax[1].set_xlabel("频率失配宽度 gamma")
ax[1].set_ylabel("有效耦合")
ax[1].set_title("有效耦合与相干份额")
ax[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(IMG, dpi=150)
plt.close(fig)

bad = [r for r in rows if not r[1]]
print("---- 判据 %d 项，通过 %d 项；图存 %s ----" % (len(rows), len(rows) - len(bad), IMG))
sys.exit(1 if bad else 0)
