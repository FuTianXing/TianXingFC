# -*- coding: utf-8 -*-
r"""
locked_mode_variance.py

二-1 第七段：共享分量方差 P 的体系内来源（场层零模）。
把第 174 / 183 / 184 条里作为输入的 P 换成导出量：场层（第 181、182 条的两点指数闭式）
在环上的零模方差。
确定性，随机固定种子 20260919。

模型：噪声取 AR(1)（q = 1 - 2p），场层取漏积分响应（alpha = 1 - kappa，beta^2 = 2 kappa），
其谱为 S(w) = beta^2 (1 - q^2) / (|1 - alpha e^{-iw}|^2 |1 - q e^{-iw}|^2)。
环长 Z 上按谱抽样，得到与第 182 条闭式同分布的场。

解析（两条闭合式）
  场层方差：C(0) = a + b
  谱零频值：S(0) = beta^2 (1 - q^2) / ((1 - alpha)^2 (1 - q)^2)
           = a (1 + q)/(1 - q) + b (1 + alpha)/(1 - alpha)
  零模方差：P = S(0)/Z（环上恰好一个 k = 0 模，其方差为 S(0)/Z）
  平台高度：E_inf = P / C(0)（配对相干在大格距上的平台）

判据与期望：
  R1 谱抽样与闭式一致：Var(X) 与 C(0) = a + b 相符（<= 5%），两种 S(0) 闭式彼此相符（<= 1e-9）
  R2 零模方差：实测（逐实现环均值，再对集合求方差）与 P = S(0)/Z 相符（<= 5%）
  R3 求和规则：零模方差等于配对协方差在格距上的平均（<= 5%）
  R4 场层单独不给出平台：大格距窗上 |Cov(n)| / P <= 5%（平台另需整体模）
  R5 平台分解与反解：注入三档整体模，平台高度反解 P_hat 与注入值相符（<= 5%），且 P_frame = P_hat - P >= 0
  R6 有限尺寸标度：Z = 256、512、1024 上 P Z 恒定（<= 5%）
  R7 参数依赖：换 q（p = 0.10）核验闭式（<= 5%）并报 S(0) 单调性
  R8 口径判别：读数加 2 pi W（整数 W）后差值读数逐项相同（<= 1e-12）
  R9 接口登记（P4-16、二-2，不作判据）
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\locked_mode_variance.png"
SEED = 20260919
KAPPA = 0.25
NS = 20000
rows = []


def report(tag, ok, expect, actual):
    rows.append((tag, ok, expect, actual))
    print("[%s] %s" % ("通过" if ok else "未过", tag))
    print("      期望 " + expect)
    print("      实测 " + actual)


def closed(q):
    alpha = 1.0 - KAPPA
    beta2 = 2.0 * KAPPA
    den = (alpha - q) * (1.0 - alpha * q)
    a = -beta2 * q / den
    b = beta2 * alpha * (1.0 - q ** 2) / (den * (1.0 - alpha ** 2))
    return alpha, beta2, a, b


def spec(w, q, alpha, beta2):
    return beta2 * (1.0 - q ** 2) / (np.abs(1.0 - alpha * np.exp(-1j * w)) ** 2 * np.abs(1.0 - q * np.exp(-1j * w)) ** 2)


def ring_field(Z, q, ns, rng):
    alpha, beta2, a, b = closed(q)
    k = np.arange(Z)
    S = spec(2.0 * np.pi * k / Z, q, alpha, beta2)
    half = Z // 2
    c = np.zeros((ns, Z), dtype=complex)
    c[:, 1:half] = (rng.normal(size=(ns, half - 1)) + 1j * rng.normal(size=(ns, half - 1))) * np.sqrt(S[1:half][None, :] / 2.0)
    c[:, Z - 1:Z - half:-1] = np.conj(c[:, 1:half])
    c[:, 0] = rng.normal(size=ns) * np.sqrt(S[0])
    c[:, half] = rng.normal(size=ns) * np.sqrt(S[half])
    return np.fft.ifft(c, axis=1).real * np.sqrt(Z)


def stats(x, n):
    a = x[:, :x.shape[1] - n]
    b = x[:, n:]
    mu = float(x.mean())
    return float(np.mean((x - mu) ** 2)), float(np.mean((a - mu) * (b - mu)))


rng = np.random.default_rng(SEED)
Q = 0.9
Z0 = 256
alpha, beta2, a, b = closed(Q)
C0 = a + b
S0_spec = beta2 * (1.0 - Q ** 2) / ((1.0 - alpha) ** 2 * (1.0 - Q) ** 2)
S0_series = a * (1.0 + Q) / (1.0 - Q) + b * (1.0 + alpha) / (1.0 - alpha)
X = ring_field(Z0, Q, NS, rng)
R = (1, 2, 4, 8, 16, 24, 32, 48, 64)
report("R1 谱抽样与闭式一致",
       abs(float(np.mean(X ** 2)) - C0) / C0 <= 0.05 and abs(S0_spec - S0_series) / S0_spec <= 1e-9,
       "Var(X) 与 C(0) = a + b = %.4f 相符（<= 5%%）；两种 S(0) 闭式相符（<= 1e-9）" % C0,
       "Var(X) %.4f（相对 %.2e）；S(0) 谱式 %.4f 对级数式 %.4f（相对 %.2e）"
       % (float(np.mean(X ** 2)), abs(float(np.mean(X ** 2)) - C0) / C0, S0_spec, S0_series, abs(S0_spec - S0_series) / S0_spec))

# ---------- R2 零模方差 ----------
zero = np.fft.fft(X, axis=1)[:, 0] / Z0
P_meas = float(np.var(zero))
P_closed = S0_spec / Z0
report("R2 零模方差", abs(P_meas - P_closed) / P_closed <= 0.05,
       "实测零模方差与 P = S(0)/Z = %.6f 相符（<= 5%%）" % P_closed,
       "实测 %.6f（相对 %.2e）；S(0) = %.4f，Z = %d" % (P_meas, abs(P_meas - P_closed) / P_closed, S0_spec, Z0))

# ---------- R3 求和规则 ----------
lags = np.arange(Z0)
C_all = np.array([stats(X, int(n))[1] for n in lags])
sum_rule = float(np.mean(C_all))
report("R3 求和规则", abs(sum_rule - P_closed) / P_closed <= 0.05,
       "配对协方差在格距上的平均等于零模方差 P = %.6f（<= 5%%）" % P_closed,
       "格距平均 %.6f（相对 %.2e）" % (sum_rule, abs(sum_rule - P_closed) / P_closed))

# ---------- R4 场层单独不给出平台 ----------
win = (64, 80, 96, 112, 128)
ratio = max(abs(stats(X, n)[1]) for n in win) / P_closed
report("R4 场层单独不给出平台", ratio <= 0.05,
       "大格距窗 %s 上 |Cov(n)| / P <= 5%%" % (win,),
       "最大比值 %.4f；窗内 Cov 从 %.3e 到 %.3e" % (ratio, stats(X, win[0])[1], stats(X, win[-1])[1]))

# ---------- R5 平台分解与反解 ----------
g5b = []
for mult in (1.0, 3.0, 10.0):
    P_inj = mult * P_closed
    Y = X + rng.normal(scale=np.sqrt(P_inj), size=(NS, 1))
    win5 = (64, 80, 96, 112)
    e_inf = float(np.mean([stats(Y, n)[1] / stats(Y, n)[0] for n in win5]))
    var_y = stats(Y, win5[0])[0]
    p_hat = var_y * e_inf
    g5b.append((mult, P_inj, p_hat, p_hat - P_closed, abs(p_hat - P_inj) / P_inj))
report("R5 平台分解与反解", all(d <= 0.05 and pf >= -0.05 * P_closed for *_, pf, d in g5b),
       "三档整体模注入下由平台反解的 P 与注入值相符（<= 5%），且余量 P_frame = P_hat - P 非负（统计容差 5%）",
       "；".join("注入 %.4f: 反解 %.4f（相对 %.2e），余量 %.4f" % (pi, ph, d, pf) for m, pi, ph, pf, d in g5b))

# ---------- R4 有限尺寸标度 ----------
g4 = []
for Z in (256, 512, 1024):
    Y = ring_field(Z, Q, NS, rng) if Z != Z0 else X
    z = np.fft.fft(Y, axis=1)[:, 0] / Z
    pm = float(np.var(z))
    g4.append((Z, pm, S0_spec / Z, pm * Z))
ref = g4[0][3]
report("R6 有限尺寸标度", all(abs(pz - ref) / ref <= 0.05 for _, _, _, pz in g4),
       "Z = 256、512、1024 上 P Z 恒定（散布 <= 5%）",
       "；".join("Z=%d: P %.6f 对 %.6f；P Z %.4f" % (Z, pm, pc, pz) for Z, pm, pc, pz in g4))

# ---------- R5 参数依赖 ----------
g5 = []
for q in (0.9, 0.8):
    al, b2, aq, bq = closed(q)
    Y = ring_field(Z0, q, NS, rng)
    z = np.fft.fft(Y, axis=1)[:, 0] / Z0
    pm = float(np.var(z))
    s0 = b2 * (1.0 - q ** 2) / ((1.0 - al) ** 2 * (1.0 - q) ** 2)
    g5.append((q, pm, s0 / Z0, float(np.mean(Y ** 2)), aq + bq))
report("R7 参数依赖", all(abs(pm - pc) / pc <= 0.05 for _, pm, pc, _, _ in g5) and g5[0][2] > g5[1][2],
       "两档 q 上零模方差与闭式相符（<= 5%），且 S(0) 随 q 增大而增大",
       "；".join("q=%.1f: P %.6f 对 %.6f（Var(X) %.4f 对 C(0) %.4f）" % (q, pm, pc, vx, c0) for q, pm, pc, vx, c0 in g5))

# ---------- R7 口径判别 ----------
W = 3
y1 = X + 2.0 * np.pi * W
d0 = X[:, 64:] - X[:, :Z0 - 64]
d1 = y1[:, 64:] - y1[:, :Z0 - 64]
dev = float(np.max(np.abs(d1 - d0)))
report("R8 口径判别（2 pi 平移不可观测）", dev <= 1e-12,
       "读数加 2 pi W（W = %d）后差值读数逐项相同（<= 1e-12）" % W,
       "最大偏差 %.3e" % dev)

print("[登记] R9：P = S(0)/Z（场层零模方差）作为锁定模方差的体系内来源，供 P4-16 与 二-2 使用，本轮只登记")

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
ax[0].plot(R, [stats(X, n)[1] / stats(X, n)[0] for n in R], "o-", label="场层实测（环上）")
ax[0].axhline(P_closed / C0, ls="--", c="k", lw=1, label="P/C(0) = %.4f（零模口径）" % (P_closed / C0))
Y = X + rng.normal(scale=np.sqrt(3.0 * P_closed), size=(NS, 1))
ax[0].plot(R, [stats(Y, n)[1] / stats(Y, n)[0] for n in R], "s-", label="加整体模（3 P）")
ax[0].set_xscale("log", base=2)
ax[0].set_xlabel("格距 n")
ax[0].set_ylabel("配对相干 E(n)")
ax[0].set_title("零模给出的共享分量与整体模余量")
ax[0].legend(fontsize=8)
Zs = [Z for Z, _, _, _ in g4]
ax[1].plot(Zs, [S0_spec / Z for Z in Zs], "k--", label="P = S(0)/Z")
ax[1].plot(Zs, [pz / Z for Z, _, _, pz in g4], "o", label="实测零模方差")
ax[1].set_xscale("log", base=2)
ax[1].set_yscale("log")
ax[1].set_xlabel("环长 Z")
ax[1].set_ylabel("P")
ax[1].set_title("零模方差与环长：P Z 恒定")
ax[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(IMG, dpi=150)
plt.close(fig)

bad = [r for r in rows if not r[1]]
print("---- 判据 %d 项，通过 %d 项；图存 %s ----" % (len(rows), len(rows) - len(bad), IMG))
sys.exit(1 if bad else 0)
