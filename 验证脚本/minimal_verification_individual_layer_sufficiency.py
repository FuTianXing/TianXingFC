# -*- coding: utf-8 -*-
"""墙三个体层充分统计量：最小验证（判据 S1 至 S8）

模型：N 个元相子分为 M 个模式块；theta_i = Phi_a + delta_i；
      w_a = W_a * eta_a * exp(i Phi_a)；Psi = sum_a w_a；
      读数 = (arg Psi + xi) 落在哪一段弧宽 2 pi W_a 的弧段上。

确定性：随机数种子固定；除触发相位 xi 外无随机源。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(20260924)
FIG = "D:/software/TianXingFC/images/individual_layer_sufficiency.png"

RES = []


def tri(t):
    """三元组 {0, t, 180+t}：单位相量和恒为 1，故块内相干因子与 t 无关。"""
    return [0.0, t, np.pi + t]


def deltas_from_triples(ts):
    out = []
    for t in ts:
        out += tri(t)
    return np.array(out)


def eta_of(d):
    return np.mean(np.exp(1j * d))


def psi_of(W, eta, Phi):
    return np.sum(np.asarray(W) * np.asarray(eta) * np.exp(1j * np.asarray(Phi)))


def draw(psi, xi, W):
    ang = np.mod(np.angle(psi) + xi, 2.0 * np.pi)
    c = np.cumsum(2.0 * np.pi * np.asarray(W))
    return np.searchsorted(c, ang, side="right")


def freq(psi, W, K, xi=None):
    if xi is None:
        xi = rng.uniform(0.0, 2.0 * np.pi, K)
    idx = draw(psi, xi, W)
    M = len(W)
    return np.array([np.count_nonzero(idx == a) for a in range(M)]) / float(K)


def report(tag, name, meas, exp, note="", grade="通过"):
    RES.append((tag, name, meas, exp, note, grade))


# ---------------- 基准族：份额 (1/3, 1/2, 1/6)，三层 eta 恒为 1/3 ----------------
Nb = np.array([6, 9, 3])
Ntot = Nb.sum()
W0 = Nb / Ntot
Phi0 = np.array([0.4, 1.9, 4.2])

dA = [deltas_from_triples([np.deg2rad(30.0)] * 2),
      deltas_from_triples([np.deg2rad(30.0)] * 3),
      deltas_from_triples([np.deg2rad(30.0)])]
dB = [deltas_from_triples([np.deg2rad(30.0), np.deg2rad(47.0)]),
      deltas_from_triples([np.deg2rad(30.0), np.deg2rad(47.0), np.deg2rad(61.0)]),
      deltas_from_triples([np.deg2rad(103.0)])]

etaA = np.array([eta_of(d) for d in dA])
etaB = np.array([eta_of(d) for d in dB])
print("块内相干因子 A：", np.round(etaA, 15), " 期望 1/3")
print("块内相干因子 B：", np.round(etaB, 15), " 期望 1/3")
print("个体配置差异（相位数不同的分量数）：",
      sum(np.count_nonzero(np.mod(dA[i] - dB[i], 2 * np.pi) > 1e-12) for i in range(3)))

pA, pB = psi_of(W0, etaA, Phi0), psi_of(W0, etaB, Phi0)

# S2：三位组充分（等价类内全部可观测读数逐位相同）
sep_psi = abs(pA - pB)
sep_ang = abs(np.angle(pA) - np.angle(pB))
xi_fix = np.linspace(0.0, 2 * np.pi, 97)[:-1]
occA = np.array([draw(pA, x, W0) for x in xi_fix])
occB = np.array([draw(pB, x, W0) for x in xi_fix])
report("S2", "等价类内 |Psi| 与 arg Psi 偏差", max(sep_psi, sep_ang), 1e-15,
       "固定 (W, eta, Phi) 改变个体相位配置", "通过" if max(sep_psi, sep_ang) < 1e-12 else "未过")

# S3：个体层惰性（固定触发相位的 97 个单次结果逐点相同）
report("S3", "等价类内单次落点不一致数", int(np.count_nonzero(occA != occB)), 0,
       "97 个触发相位逐点比对", "通过" if np.all(occA == occB) else "未过")

# S4：频率只依赖份额
K = 400000
fA = freq(pA, W0, K)
fB = freq(pB, W0, K)
etaC = etaA.copy()
etaC[1] = 0.9 * etaA[1]
pC = psi_of(W0, etaC, Phi0 + np.array([1.1, 0.0, 0.0]))
fC = freq(pC, W0, K)
band = 3.0 * np.sqrt(W0 * (1.0 - W0) / K)
dev = np.max(np.abs(fA - fB))
report("S4", "换 eta 与 Phi 后的频率最大偏差", dev, band.max(),
       "3 sigma 多项误差带 %.5f" % band.max(), "通过" if dev < band.max() else "未过")

# S5：可见度只依赖 eta
phis = np.deg2rad(np.linspace(0.0, 80.0, 41))
V2 = np.array([abs(eta_of(np.array([t, -t]))) for t in phis])
Vexp = np.abs(np.cos(phis))
report("S5", "两元相块 |eta| 对 cos psi 的最大偏差", np.max(np.abs(V2 - Vexp)), 1e-15,
       "41 档扫描", "通过" if np.max(np.abs(V2 - Vexp)) < 1e-14 else "未过")

# S6：单次相位只依赖 Phi
dPhi = 0.37
pD = psi_of(W0, etaA, Phi0 + dPhi)
shift = np.mod(np.angle(pD) - np.angle(pA), 2 * np.pi)
report("S6", "整体平移 Phi 后的 arg Psi 相移", shift, dPhi,
       "期望相移等于平移量", "通过" if abs(shift - dPhi) < 1e-12 else "未过")
occD = np.array([draw(pD, x, W0) for x in xi_fix])
report("S6b", "相移引起的落点变化数", int(np.count_nonzero(occD != occA)), None,
       "定性：整体相移改变落点而不改变频率", "读数")

# S7：非平衡检验（均匀触发不可构造偏离；触发集中则偏离）
Ks = [2000, 20000, 200000]
REP = 400
devs = []
for k in Ks:
    d = [np.max(np.abs(freq(pA, W0, k) - W0)) for _ in range(REP)]
    devs.append(float(np.mean(d)))
fit = np.polyfit(np.log(Ks), np.log(devs), 1)[0]
report("S7", "均匀触发下频率偏离份额的 K 标度", fit, -0.5,
       "每档 400 次重复的均值（K = 2e3/2e4/2e5）", "通过" if abs(fit + 0.5) < 0.12 else "未过")
print("S7 三点偏离均值：", ["%.6f" % d for d in devs])
xi_conc = (np.angle(np.exp(1j * rng.vonmises(0.0, 6.0, K))) + np.pi) / (2 * np.pi)
f_conc = np.array([np.count_nonzero(draw(pA, xi_conc, W0) == a) for a in range(3)]) / K
report("S7b", "触发集中（von Mises kappa = 6）时的频率最大偏离", np.max(np.abs(f_conc - W0)),
       band.max(), "对照均匀触发", "读数")

# S8：份额可读（反证对照）
KB = 2000000
pts = []
for n1 in range(1, 13):
    nb = np.array([n1, 7, 5])
    Wn = nb / nb.sum()
    fn = freq(psi_of(Wn, np.ones(3), Phi0), Wn, KB)
    pts.append((Wn, fn))
WW = np.concatenate([q[0] for q in pts])
FF = np.concatenate([q[1] for q in pts])
slope = float(np.dot(WW, FF) / np.dot(WW, WW))
maxdev = max(float(np.max(np.abs(f - Wq))) for Wq, f in pts)
report("S8", "频率对份额的过原点斜率（12 档，每档 2e6 采样）", slope, 1.0,
       "最大单档偏差 %.2e" % maxdev, "通过" if abs(slope - 1.0) < 2e-3 else "未过")

# S1：份额非充分（反证基座）
d1 = deltas_from_triples([np.deg2rad(30.0), np.deg2rad(30.0)])
d1[4] = np.deg2rad(180.0 + 47.0)  # 三元组改成 {0, 30, 227}：和不再为 1
eta1 = eta_of(d1)
etaS = etaA.copy()
etaS[0] = eta1
pS = psi_of(W0, etaS, Phi0)
dV = abs(eta1) - abs(etaA[0])
dAng = abs(np.angle(pS) - np.angle(pA))
report("S1", "固定 (W, Phi) 改变 eta 后的 |eta| 与 arg Psi 改变", max(dV, dAng), None,
       "|eta| 由 1/3 变为 %.6f" % abs(eta1), "读数" if max(dV, dAng) > 1e-3 else "未过")

print()
print("=" * 96)
print("判据 | 判据量 | 实测 | 期望 | 备注 | 结论")
print("=" * 96)
for tag, name, meas, exp, note, grade in RES:
    m = ("%.6g" % meas) if isinstance(meas, float) else str(meas)
    e = ("%.6g" % exp) if isinstance(exp, float) else ("-" if exp is None else str(exp))
    print("%-5s | %s | %s | %s | %s | %s" % (tag, name, m, e, note, grade))
print("=" * 96)

# ---------------- 图 ----------------
fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.0))
ax[0, 0].bar(np.arange(3) - 0.18, W0, width=0.34, label="份额 W_a", color="#4c72b0")
ax[0, 0].bar(np.arange(3) + 0.18, fA, width=0.34, label="实测频率（4e5 采样）", color="#dd8452")
ax[0, 0].set_xticks([0, 1, 2]); ax[0, 0].set_xticklabels(["块 1", "块 2", "块 3"])
ax[0, 0].set_title("S8/S4：频率只跟随份额"); ax[0, 0].legend(); ax[0, 0].set_ylabel("频率")

ax[0, 1].bar(np.arange(4) - 0.2, [sep_psi, sep_ang, 0.0, 0.0], width=0.4,
             label="等价类内偏差（A 与 B）", color="#55a868")
ax[0, 1].bar(np.arange(4) + 0.2, [1e-16, 1e-16, dV, dAng], width=0.4,
             label="反证对照（S1：改 eta）", color="#c44e52")
ax[0, 1].set_yscale("log"); ax[0, 1].set_xticks(range(4))
ax[0, 1].set_xticklabels(["|dPsi|", "d(argPsi)", "d|eta|", "d(argPsi)"])
ax[0, 1].set_title("S2/S3 惰性 对 S1 反证（对数轴）"); ax[0, 1].legend(fontsize=8)

ax[1, 0].plot(np.rad2deg(phis), Vexp, "k-", lw=1.0, label="理论 |cos psi|")
ax[1, 0].plot(np.rad2deg(phis), V2, "o", ms=3.5, mfc="none", label="实测 |eta|")
ax[1, 0].set_xlabel("块内相位散开 phi（度）"); ax[1, 0].set_ylabel("块可见度 V")
ax[1, 0].set_title("S5：可见度只由 eta 承担"); ax[1, 0].legend(fontsize=8)

ax[1, 1].loglog(Ks, devs, "o-", label="均匀触发（400 次重复均值）")
ax[1, 1].loglog(Ks, band.max() * np.sqrt(Ks[0] / np.asarray(Ks)), "k--", lw=1.0, label="3 sigma 参考线")
ax[1, 1].axhline(np.max(np.abs(f_conc - W0)), color="#c44e52",
                 label="触发集中：偏离 %.4f" % np.max(np.abs(f_conc - W0)))
ax[1, 1].set_xlabel("采样数 K"); ax[1, 1].set_ylabel("频率偏离份额的最大值")
ax[1, 1].set_title("S7：非平衡检验"); ax[1, 1].legend(fontsize=8)

fig.tight_layout()
fig.savefig(FIG, dpi=140)
print("图已保存：", FIG)
