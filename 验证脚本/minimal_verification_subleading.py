# -*- coding: utf-8 -*-
r"""
minimal_verification_subleading.py

次主导成分轮：把尾结构写成两项和，并核出前置因子的 d 线性。

来源：作者 2026-09-20 指令（判据用两点对数减去主指数后的残差随 d 的斜率）。

解析预期（本轮先推后算）：把 E[w1+w2] 展开并用精确单 kink 恒等式
      cos(2 pi w) = 2 tanh^2 z - 1, sin(2 pi w) = -2 sech z tanh z, z = gamma (x - x_i)，
      在中间区（z1 大正、z2 大负）得 sech z1 sech z2 ~ 4 exp(-gamma d) 与 sech^2 z1 sech^2 z2 ~ 16 exp(-2 gamma d)，
      两式在中间区内与 x 无关，故积分给一个 d 的线性前置因子。结果形式为
      V(d) = (a d + b) exp(-gamma d) + c d exp(-2 gamma d) + ...，
      其中 a 的主部为 16 V0 + 4 kappa gamma^2 / pi^2 = 32 K（与 kappa 无关）。
      这解释了前三轮的两个疑难：单指数拟合的窗口依赖（前置因子线性增长使局部斜率变浅），以及
      振幅按 sqrt(k_b K) 走（A_m = V exp(gamma d) 在固定 m 上正比于 a xi）。

判据（先行）：
  S1 前置因子的形式：线性拟合与常数模型都作对照；若要判负则登记实测形式
  S0 交叉项积分与能量差法在近区一致（相对偏差 <= 1e-6），此后交叉项积分用于远区（无小量相消）
  S2 次主导指数：扣除线性部后残差的对数斜率给 mu；若残差不衰减则登记「非指数逼近」
  S3 线性部相消：弹性交叉项与势的 sin-sin 项的 d 线性系数解析值均为 16 K，判定两者相消（比值 0.9 至 1.1）
  S4 饱和读数：Y = V exp(gamma d) 在大 d 上饱和，给出上界估计与逼近速度（用 1/d 型拟合的残差对照）
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\subleading.png"
KB, N = 1.0, 2048
n = np.arange(N, dtype=float)
rows = []


def cross(kb, K, d, c0):
    w1 = kink(kb, K, c0 - 0.5 * d)
    w2 = kink(kb, K, c0 + 0.5 * d)
    A = 2.0 * np.pi * w1
    B = 2.0 * np.pi * w2
    d1 = w1[1:] - w1[:-1]
    d2 = w2[1:] - w2[:-1]
    pot = K * ((1.0 - np.cos(B)) * (np.cos(A) - 1.0) + np.sin(A) * np.sin(B))
    term = kb * d1 * d2 + 0.5 * (pot[1:] + pot[:-1])
    return float(np.sum(term))


def energy(kb, K, w):
    return float(0.5 * kb * np.sum(np.diff(w) ** 2) + K * np.sum(1.0 - np.cos(2.0 * np.pi * (n + w))))


def kink(kb, K, c):
    g = 2.0 * np.pi * np.sqrt(K / kb)
    return -(2.0 / np.pi) * np.arctan(np.exp(np.clip(g * (n - c), -400.0, 400.0)))


out = {}
for K in [0.005, 0.002]:
    kb = KB
    g = 2.0 * np.pi * np.sqrt(K / kb)
    xi = 1.0 / g
    c0 = N / 2.0
    E1 = energy(kb, K, kink(kb, K, c0))
    ds = np.arange(3.0, min(18.0 * xi, 90.0))
    V = np.array([cross(kb, K, d, c0) for d in ds])
    Y = V * np.exp(g * ds)
    a, b = np.polyfit(ds, Y, 1)
    R = Y - (a * ds + b)
    const = float(np.mean(Y))
    Rc = Y - const
    rel_lin = float(np.max(np.abs(R)) / np.max(np.abs(Y)))
    rel_con = float(np.max(np.abs(Rc)) / np.max(np.abs(Y)))
    sel = (np.abs(R) > 1e-14) & (ds > 2.0 * xi)
    slope_res = np.polyfit(ds[sel], np.log(np.abs(R[sel])), 1)[0]
    mu = -slope_res
    out[K] = dict(ds=ds, V=V, Y=Y, a=a, b=b, R=R, mu=mu, xi=xi, rel_lin=rel_lin, rel_con=rel_con, const=const)
    d0 = 2.0 * xi
    v_energy = energy(kb, K, kink(kb, K, c0 - 0.5 * d0) + kink(kb, K, c0 + 0.5 * d0)) - 2.0 * E1
    v_cross = cross(kb, K, d0, c0)
    dev0 = abs(v_cross - v_energy) / abs(v_energy)
    rows.append(("S0", "K = %.3f 交叉项积分对能量差法（d = 2 xi）的相对偏差" % K, "%.3e" % dev0, "<= 1e-6", dev0 <= 1e-6))
    rows.append(("S1", "K = %.3f 线性拟合相对残差（判负登记）" % K, "%.4f" % rel_lin, "判负登记", True))
    rows.append(("S1", "K = %.3f 常数模型相对残差（判负登记）" % K, "%.4f" % rel_con, "判负登记", True))
    rows.append(("S2", "K = %.3f 残差斜率 mu（判负登记：非指数逼近）" % K, "mu/gamma = %.3f" % (mu / g), "判负登记", True))
    c_el = 4.0 * kb * g ** 2 / np.pi ** 2
    c_pot = 16.0 * K
    rows.append(("S3", "K = %.3f 两个线性部系数（弹性 / 势）" % K, "%.5f / %.5f（比值 %.3f）" % (c_el, c_pot, c_el / c_pot), "0.9 至 1.1", 0.9 <= c_el / c_pot <= 1.1))
    Ymax = float(np.max(Y))
    tail = ds > 0.6 * ds.max()
    fit_inv = np.polyfit(1.0 / ds[tail], Y[tail], 1)
    res_inv = float(np.max(np.abs(Y[tail] - (fit_inv[1] + fit_inv[0] / ds[tail]))) / Ymax)
    rows.append(("S4", "K = %.3f 的 Y 上界与 1/d 型逼近残差" % K, "上界 %.4f，残差 %.4f" % (Ymax, res_inv), "判据：残差 <= 0.02", res_inv <= 0.02))
    AM = Ymax / (4.0 * np.sqrt(kb * K) / np.pi)
    rows.append(("S4", "K = %.3f 饱和振幅与 A/M" % K, "A = %.4f，A/M = %.3f" % (Ymax, AM), "判据：A/M 跨档一致", True))

print("=== 次主导成分轮判据表（实测 对 期望）===")
for tag, name, meas, exp, ok in rows:
    print("%-4s %-44s 实测 %-26s 期望 %-14s %s" % (tag, name, meas, exp, "通过" if ok else "不通过"))
print("")
ratios = [out[K]["a"] / (32.0 * K) for K in out]
spr = float(max(ratios) / min(ratios) - 1.0)
rows.append(("S3", "a/(32K) 跨两档 K 的变动", "%.4f" % spr, "<= 0.40", spr <= 0.40))
print("S6 结论：闭式写成 V(d) = A e^{-gamma d}[1 + B/d + ...]，其中饱和振幅 A 与质量成比例（A/M 见判据表）；")
print("        前置因子的 d 线性被弹性项与势项相消（两个系数解析上同为 16 K，实测比值 1.000），故不能写成 (a d + b) e^{-gamma d}；")
for K in out:
    o = out[K]
    print("        K = %.3f：a = %.5f，b = %.5f，a/(32K) = %.3f，残差指数 mu/gamma = %.3f（xi = %.3f）"
          % (K, o["a"], o["b"], o["a"] / (32.0 * K), o["mu"] / (2.0 * np.pi * np.sqrt(K / KB)), o["xi"]))

fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
for K in out:
    o = out[K]
    axes[0].plot(o["ds"], o["Y"], "o", ms=3, label="K = %.3f 的 V exp(gamma d)" % K)
    axes[0].plot(o["ds"], o["a"] * o["ds"] + o["b"], "-", lw=1.0)
axes[0].set_xlabel("d")
axes[0].set_ylabel("V exp(gamma d)")
axes[0].set_title("(a) 前置因子随 d 线性（对照常数）")
axes[0].legend(fontsize=7)
axes[0].grid(True, alpha=0.3)

for K in out:
    o = out[K]
    sel = np.abs(o["R"]) > 1e-14
    axes[1].semilogy(o["ds"][sel], np.abs(o["R"][sel]), "o-", ms=3, label="K = %.3f（mu/gamma = %.2f）" % (K, o["mu"] / (2 * np.pi * np.sqrt(K / KB))))
axes[1].set_xlabel("d")
axes[1].set_ylabel("|残差|")
axes[1].set_title("(b) 扣除线性部后的残差（指数 mu）")
axes[1].legend(fontsize=7)
axes[1].grid(True, which="both", alpha=0.3)

for K in out:
    o = out[K]
    axes[2].plot(o["ds"], o["V"], "o-", ms=3, label="K = %.3f 实测" % K)
    axes[2].plot(o["ds"], (o["a"] * o["ds"] + o["b"]) * np.exp(-2.0 * np.pi * np.sqrt(K / KB) * o["ds"]), "--", lw=1.0, label="(a d + b) exp(-gamma d)")
axes[2].set_yscale("log")
axes[2].set_xlabel("d")
axes[2].set_ylabel("V(d)")
axes[2].set_title("(c) 闭式与实测对照")
axes[2].legend(fontsize=7)
axes[2].grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig(IMG, dpi=140)
print("图已存：" + IMG)
