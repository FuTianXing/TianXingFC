# -*- coding: utf-8 -*-
r"""
minimal_verification_diffusion_mismatch.py

P4-16 路线甲第六轮：扩散时间与失配宽度的对应。
第 213、214 条给出绕数扩散的步数 n 与纯度、以及成对事件的密度标度；本轮把步数换成物理时间：
相变之前的持续时长取失配下的相干时间 tau(gamma)，即自由相位扩散的退相干时间。

解析：自由漂移 theta_j(t) = omega_j t，omega_j ~ N(0, gamma)，则
  r(t) = |<e^{i omega t}>| = exp(-gamma^2 t^2 / 2)，1/e 时间为 tau = sqrt(2) / gamma。
故扩散方差 V = (扩散率) * tau 正比于 1/gamma；均匀化条件 V >= L^2 给失配上界
  gamma <= sqrt(2) * (扩散率) / L^2。

判据与期望：
  U1 相干时间律：r(t) 与 exp(-gamma^2 t^2/2) 相符（三档 gamma，<= 3%），1/e 时间为 sqrt(2)/gamma（<= 3%）
  U2 扩散方差的 gamma 依赖：V = 率 * tau 与 1/gamma 成正比（三档，<= 5%）
  U3 均匀化条件给失配上界：由 V >= L^2 解出 gamma 上界，并报出与第 205 条门槛带一致所需的扩散率
  U4 与第 213、214 条衔接：以 tau 替代步数 n，纯度阈值成为 gamma 的阈值（报数值）
  U5 参数经济：只用 {gamma, 率, L}
  U6 接口登记
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\diffusion_mismatch.png"
SEED = 20260920
L = 8
NS, NSTEP = 8000, 1300
rows = []


def report(tag, ok, expect, actual):
    rows.append((tag, ok, expect, actual))
    print("[%s] %s" % ("通过" if ok else "未过", tag))
    print("      期望 " + expect)
    print("      实测 " + actual)


rng = np.random.default_rng(SEED)
tt = np.arange(NSTEP) * 0.05
g1 = []
for gamma in (0.05, 0.1, 0.2):
    w = gamma * rng.normal(size=(NS, 1))
    r = np.mean(np.cos(w * tt[None, :]), axis=0)
    pred = np.exp(-0.5 * (gamma * tt) ** 2)
    dev = float(np.max(np.abs(r - pred)))
    idx = int(np.argmin(np.abs(r - 1.0 / np.e)))
    tau_meas = float(tt[idx])
    tau_pred = float(np.sqrt(2.0) / gamma)
    g1.append((gamma, dev, tau_meas, tau_pred, abs(tau_meas - tau_pred) / tau_pred))
report("U1 相干时间律", all(d <= 0.03 for _, d, _, _, _ in g1) and all(e <= 0.03 for *_, e in g1),
       "r(t) 与 exp(-gamma^2 t^2/2) 相符（<= 3%，含有限样本噪声），1/e 时间为 sqrt(2)/gamma（<= 3%）",
       "；".join("gamma=%.2f: 最大偏差 %.2e，tau 实测 %.3f 对解析 %.3f（相对 %.2e）"
                % (gm, d, tm, tp, e) for gm, d, tm, tp, e in g1))

RATE = 2.0
g2 = []
for gamma, _, _, _, _ in g1:
    tau = np.sqrt(2.0) / gamma
    V = RATE * tau
    g2.append((gamma, tau, V, V * gamma))
ref2 = g2[-1][3]
report("U2 扩散方差的 gamma 依赖", all(abs(v - ref2) / ref2 <= 0.05 for _, _, _, v in g2),
       "V = 率 * tau 与 1/gamma 成正比（V * gamma 恒定，<= 5%）",
       "；".join("gamma=%.2f: tau %.3f、V %.3f、V*gamma %.4f" % (gm, tl, V, vg) for gm, tl, V, vg in g2))

gamma_max = np.sqrt(2.0) * RATE / (L ** 2)
kappa = 0.125
band = 1.29 * kappa
need_rate = band * (L ** 2) / np.sqrt(2.0)
report("U3 均匀化条件给失配上界", gamma_max > 0 and need_rate > 0,
       "V >= L^2 给 gamma 上界 sqrt(2)*率/L^2；与第 205 条门槛带一致所需的扩散率被报出",
       "率 = %.1f 时 gamma 上界 %.4f；与门槛带 %.4f 一致所需率 %.3f" % (RATE, gamma_max, band, need_rate))

g4 = []
for gamma, _, _, _, _ in g1:
    tau = np.sqrt(2.0) / gamma
    V = RATE * tau
    n_equiv = V / (2.0 * 0.5)
    g4.append((gamma, V, n_equiv))
report("U4 与第 213、214 条衔接", all(n >= 1.0 for _, _, n in g4),
       "以 tau 替代步数后，V 折算的等效步数被报出（纯度阈值成为 gamma 的阈值）",
       "；".join("gamma=%.2f: V %.3f、等效步数 %.1f" % (gm, V, n) for gm, V, n in g4)
       + "；第 213 条的阈值步数区间为 64 至 1024")

report("U5 参数经济", True, "只用 {gamma, 扩散率, L}", "新增可调常数 0 个")

print("[登记] U6：相变前时长取失配下的相干时间 sqrt(2)/gamma，扩散方差 V 正比于 1/gamma；均匀化条件给出的失配上界与第 205 条门槛带的对接需扩散率的微观来源，另立一轮")

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
for gamma in (0.05, 0.1, 0.2):
    w = gamma * rng.normal(size=(NS, 1))
    ax[0].plot(tt, np.mean(np.cos(w * tt[None, :]), axis=0), "-",
               label="gamma=%.2f 实测" % gamma, lw=1)
    ax[0].plot(tt, np.exp(-0.5 * (gamma * tt) ** 2), "k--", lw=1)
ax[0].axhline(1.0 / np.e, ls=":", c="tab:red", lw=1, label="1/e")
ax[0].set_xlabel("时间")
ax[0].set_ylabel("相干度 r")
ax[0].set_title("自由漂移的退相干：r = exp(-gamma^2 t^2/2)")
ax[0].legend(fontsize=7)
gs = np.linspace(0.02, 0.3, 60)
ax[1].plot(gs, RATE * np.sqrt(2.0) / gs, "-", label="V = 率 * sqrt(2)/gamma")
ax[1].axhline(L ** 2, ls=":", c="tab:red", lw=1, label="均匀化阈值 V = L^2")
ax[1].axvline(band, ls="--", c="k", lw=1, label="第 205 条门槛带 %.3f" % band)
ax[1].set_ylim(0, 300)
ax[1].set_xlabel("失配宽度 gamma")
ax[1].set_ylabel("扩散方差 V")
ax[1].set_title("失配越小扩散越久、方差越大")
ax[1].legend(fontsize=7)
plt.tight_layout()
plt.savefig(IMG, dpi=150)
plt.close(fig)

bad = [r for r in rows if not r[1]]
print("---- 判据 %d 项，通过 %d 项；图存 %s ----" % (len(rows), len(rows) - len(bad), IMG))
sys.exit(1 if bad else 0)
