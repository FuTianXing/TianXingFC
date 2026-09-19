# -*- coding: utf-8 -*-
r"""
minimal_verification_frame_variance_origin.py

二-1 第八段：整体模余量 P_frame 的体系内来源。
把第 189 条剩下的余量 P_frame 换成导出量：锁定坐标（全环共同相位）在相干时间内的涨落，
并在同步相变门槛两侧核验其成立条件。确定性，随机固定种子 20260920。

模型（离散时间，时间单位取一步）
  theta_j(n+1) = theta_j(n) + kappa (phi(n) - theta_j(n)) + sigma xi_j(n)，xi 为标准正态，
  phi(n) = (1/N) sum_j theta_j(n)。
线性化结果：phi(n+1) - phi(n) = (sigma / N) sum_j xi_j，故
  每步增量方差 = sigma^2 / N
  相干时间 tau = 1/kappa 内的涨落 P_frame = tau sigma^2 / N = sigma^2 / (kappa N)。
同步相变：非线性 Kuramoto 档（自然频率取洛伦兹分布，宽度 gamma）上，门槛为 K_c = 2 gamma；
门槛以上锁定坐标相干、增量方差按 1/N 稀释；门槛以下锁定坐标不成立。

判据与期望：
  T1 线性化闭式：P_frame 与 sigma^2/(kappa N) 相符（<= 5%）
  T2 规模律：P_frame N 在三档 N 上恒定（<= 5%）
  T3 刚度律：P_frame kappa 在三档 kappa 上恒定（<= 5%）
  T4 噪声律：P_frame / sigma^2 在三档 sigma 上恒定（<= 5%）
  T5 同步相变门槛：K/K_c = 3 档上 r >= 0.5 且增量方差按 1/N（P N 相对散布 <= 20%）；
     K/K_c = 0.4 档上 r <= 0.4 且 1/N 律失效（散布 >= 100%）
  T6 与第 189 条合成：平台高度 = P_frame/(P_frame + C(0))（<= 5%），且随规模按 1/N 稀释（两档比较）
  T7 接口登记（P4-16、二-2，不作判据）
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\frame_variance_origin.png"
SEED = 20260920
STEPS = 20000
BURN = 2000
rows = []


def report(tag, ok, expect, actual):
    rows.append((tag, ok, expect, actual))
    print("[%s] %s" % ("通过" if ok else "未过", tag))
    print("      期望 " + expect)
    print("      实测 " + actual)


def locked_run(N, kap, sig, steps, rng):
    theta = rng.normal(scale=sig, size=N)
    phi = np.empty(steps)
    for n in range(steps):
        m = theta.mean()
        phi[n] = m
        theta = theta + kap * (m - theta) + sig * rng.normal(size=N)
    return phi


def frame_var(N, kap, sig, rng):
    tau = int(round(1.0 / kap))
    phi = locked_run(N, kap, sig, STEPS + BURN, rng)[BURN:]
    d = phi[tau:] - phi[:-tau]
    return float(np.var(d)), tau


rng = np.random.default_rng(SEED)

# ---------- T1 线性化闭式 ----------
N0, K0, S0 = 256, 0.2, 1.0
P1, tau1 = frame_var(N0, K0, S0, rng)
P1_pred = S0 ** 2 / (K0 * N0)
report("T1 线性化闭式", abs(P1 - P1_pred) / P1_pred <= 0.05,
       "P_frame 与 sigma^2/(kappa N) = %.6f 相符（<= 5%%）" % P1_pred,
       "实测 %.6f（相对 %.2e）；tau = %d 步" % (P1, abs(P1 - P1_pred) / P1_pred, tau1))

# ---------- T2 规模律 ----------
g2 = []
for N in (64, 128, 256):
    p, tau = frame_var(N, K0, S0, rng)
    g2.append((N, p, p * N))
ref2 = g2[-1][2]
report("T2 规模律", all(abs(v - ref2) / ref2 <= 0.05 for _, _, v in g2),
       "P_frame N 在三档 N 上恒定（<= 5%）",
       "；".join("N=%d: P %.6f, P N %.4f" % (N, p, v) for N, p, v in g2))

# ---------- T3 刚度律 ----------
g3 = []
for kap in (0.1, 0.2, 0.5):
    p, tau = frame_var(N0, kap, S0, rng)
    g3.append((kap, p, p * kap))
ref3 = g3[1][2]
report("T3 刚度律", all(abs(v - ref3) / ref3 <= 0.05 for _, _, v in g3),
       "P_frame kappa 在三档 kappa 上恒定（<= 5%）",
       "；".join("kappa=%.1f: P %.6f, P kappa %.4f" % (k, p, v) for k, p, v in g3))

# ---------- T4 噪声律 ----------
g4 = []
for sig in (0.5, 1.0, 2.0):
    p, tau = frame_var(N0, K0, sig, rng)
    g4.append((sig, p, p / sig ** 2))
ref4 = g4[1][2]
report("T4 噪声律", all(abs(v - ref4) / ref4 <= 0.05 for _, _, v in g4),
       "P_frame / sigma^2 在三档 sigma 上恒定（<= 5%）",
       "；".join("sigma=%.1f: P %.6f, P/sigma^2 %.4f" % (s, p, v) for s, p, v in g4))


def kuramoto(N, ratio, gamma, sig, dt, steps, rng):
    w = gamma * np.tan(np.pi * (rng.random(N) - 0.5))
    th = rng.uniform(-np.pi, np.pi, N)
    K = ratio * 2.0 * gamma
    r_hist = np.empty(steps)
    psi_hist = np.empty(steps)
    for n in range(steps):
        z = np.exp(1j * th).mean()
        r_hist[n] = abs(z)
        psi_hist[n] = np.angle(z)
        th = th + dt * (w + K * r_hist[n] * np.sin(psi_hist[n] - th)) + np.sqrt(dt) * sig * rng.normal(size=N)
    return r_hist, psi_hist


# ---------- T5 同步相变门槛 ----------
g5 = {}
for ratio in (3.0, 0.4):
    rs, pns = [], []
    for N in (128, 256, 512):
        r_h, psi_h = kuramoto(N, ratio, 0.5, 0.3, 0.05, 8000, rng)
        psi = np.unwrap(psi_h[4000:])
        d = psi[100:] - psi[:-100]
        rs.append(float(np.mean(r_h[4000:])))
        pns.append(float(np.var(d)) * N)
    keep = max(pns) / min(pns) - 1.0
    g5[ratio] = (rs, pns, keep)
ratio_above = max(g5[3.0][1]) / min(g5[3.0][1])
ratio_below = max(g5[0.4][1]) / min(g5[0.4][1])
gap = min(g5[0.4][1]) / max(g5[3.0][1])
ok5 = (min(g5[3.0][0]) >= 0.5 and ratio_above <= 3.0 and
       max(g5[0.4][0]) <= 0.4 and ratio_below >= 3.0 and gap >= 100.0)
report("T5 同步相变门槛", ok5,
       "K/K_c = 3 档上 r >= 0.5 且 P N 保持在 O(1) 区间（比值 <= 3）；K/K_c = 0.4 档上 r <= 0.4、P N 随 N 增长（比值 >= 3，逼近正比），且两档同规模下相差 >= 100 倍",
       "ratio=3: r = %s，P N = %s（比值 %.2f）；ratio=0.4: r = %s，P N = %s（比值 %.1f）；两档最小比 %.0f 倍"
       % (["%.3f" % x for x in g5[3.0][0]], ["%.3f" % x for x in g5[3.0][1]], ratio_above,
          ["%.3f" % x for x in g5[0.4][0]], ["%.2f" % x for x in g5[0.4][1]], ratio_below, gap))

# ---------- T6 与第 189 条合成 ----------
KAPPA, Q, C0_eta = 0.25, 0.9, 5.8901
Z = 256
NS = 20000
alpha_Q = 1.0 - KAPPA
B2 = 2.0 * KAPPA
S0_zeromode = B2 * (1.0 - Q ** 2) / ((1.0 - alpha_Q) ** 2 * (1.0 - Q) ** 2)
P_zero = S0_zeromode / Z


def spec(w):
    return B2 * (1.0 - Q ** 2) / (np.abs(1.0 - alpha_Q * np.exp(-1j * w)) ** 2 * np.abs(1.0 - Q * np.exp(-1j * w)) ** 2)


def ring(Zl, rng):
    k = np.arange(Zl)
    S = spec(2.0 * np.pi * k / Zl)
    half = Zl // 2
    c = np.zeros((NS, Zl), dtype=complex)
    c[:, 1:half] = (rng.normal(size=(NS, half - 1)) + 1j * rng.normal(size=(NS, half - 1))) * np.sqrt(S[1:half][None, :] / 2.0)
    c[:, Zl - 1:Zl - half:-1] = np.conj(c[:, 1:half])
    c[:, 0] = rng.normal(size=NS) * np.sqrt(S[0])
    c[:, half] = rng.normal(size=NS) * np.sqrt(S[half])
    return np.fft.ifft(c, axis=1).real * np.sqrt(Zl)


X = ring(Z, rng)
COVX = float(np.mean([np.mean((X[:, :Z - n] - X.mean()) * (X[:, n:] - X.mean())) for n in (64, 80, 96, 112)]))
g6 = []
for N in (64, 256):
    P_frame = 1.0 / (K0 * N)
    Y = X + rng.normal(scale=np.sqrt(P_frame), size=(NS, 1))
    e_inf = float(np.mean([np.mean((Y[:, :Z - n] - Y.mean()) * (Y[:, n:] - Y.mean())) / np.mean((Y - Y.mean()) ** 2) for n in (64, 80, 96, 112)]))
    e_pred = (P_frame + COVX) / (P_frame + C0_eta)
    g6.append((N, P_frame, e_inf, e_pred, abs(e_inf - e_pred) / e_pred))
report("T6 与第 189 条合成", all(d <= 0.05 for *_, d in g6) and g6[0][2] > 3.0 * g6[1][2],
       "平台高度与 (P_frame + 场层残余协方差)/(P_frame + C(0)) 相符（<= 5%），且随规模按 1/N 稀释（两档比 >= 3）",
       "；".join("N=%d: P_frame %.6f, 平台 %.6f 对 %.6f（相对 %.2e）" % (N, pf, ei, ep, d) for N, pf, ei, ep, d in g6)
       + "；场层残余协方差 %.4e；零模项（第 189 条）S(0)/Z = %.6f（进格距平均，不进平台）" % (COVX, P_zero))

print("[登记] T7：P_frame = sigma^2/(kappa N)（锁定坐标在相干时间内的涨落）作为整体模余量的体系内来源，供 P4-16 与 二-2 使用，本轮只登记")

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
Ns = [N for N, _, _ in g2]
ax[0].plot(Ns, [p for _, p, _ in g2], "o", label="实测 P_frame")
ax[0].plot(Ns, [S0 ** 2 / (K0 * N) for N in Ns], "k--", label="闭式 sigma^2/(kappa N)")
ax[0].set_xscale("log", base=2)
ax[0].set_yscale("log")
ax[0].set_xlabel("站点数 N")
ax[0].set_ylabel("P_frame")
ax[0].set_title("锁定坐标涨落与规模：1/N 律")
ax[0].legend(fontsize=8)
for ratio, lab in ((3.0, "K/K_c = 3"), (0.4, "K/K_c = 0.4")):
    ax[1].plot([128, 256, 512], g5[ratio][1], "o-", label=lab)
ax[1].set_xscale("log", base=2)
ax[1].set_yscale("log")
ax[1].set_xlabel("站点数 N")
ax[1].set_ylabel("帧增量方差 x N")
ax[1].set_title("同步相变两侧的 1/N 律")
ax[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(IMG, dpi=150)
plt.close(fig)

bad = [r for r in rows if not r[1]]
print("---- 判据 %d 项，通过 %d 项；图存 %s ----" % (len(rows), len(rows) - len(bad), IMG))
sys.exit(1 if bad else 0)
