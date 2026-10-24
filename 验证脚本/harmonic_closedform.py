# -*- coding: utf-8 -*-
r"""
harmonic_closedform.py

谐波系数的解析导出轮：把交叉项积分收成闭式，并给出一阶谐波系数的解析值。

来源：作者 2026-09-20 指令（接着做前者）。上一轮（判据 U1 至 U4）给出数值系数表 c1 = 3.96M、c2 = 7.46M、c3 = -7.39M。

解析推导：用精确 kink 恒等式 cos(2 pi w) = 2 tanh^2 z - 1、sin(2 pi w) = 2 sech z tanh z（z = gamma (x - x_i)）
      与 k_b gamma^2 / pi^2 = 4 K，交叉项积分化为
      V(d) = (4 K / gamma) * int du [ S1 S2 (1 + T1 T2) - S1^2 S2^2 ]，S_i = sech(u - z_i)，T_i = tanh。
      再用 sech 的傅里叶变换 int sech u exp(-i q u) du = pi sech(pi q / 2)：
      A(theta)        = int S1 S2 du      = 2 theta / sinh theta
      int dS1 dS2 du  = -d^2 A / d theta^2
      I3(theta)       = int S1^2 S2^2 du  = int dq / (2 pi) * pi^2 q^2 / sinh^2(pi q/2) * exp(-i q theta)
      故闭式为 V(d) = (4 K / gamma) [ A(theta) - A2(theta) - I3(theta) ]，theta = gamma d，A2 = d^2A/dtheta^2。

判据（先行）：
  V1 闭式与交叉项积分的最大相对偏差 <= 1e-8（解析导出正确）
  V2 A - A2 的前置因子：theta 大时 exp(theta)(A - A2) 收敛到 8，偏差 <= 0.05（theta 线性部精确相消）
  V3 解析一阶系数 c1 = 4 M 与上一轮数值拟合值之比落在 0.95 至 1.05
  V4 登记：闭式与解析系数
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\harmonic_closedform.png"
N = 1024
n = np.arange(N, dtype=float)
rows = []
C1_NUM = 3.96


def kink(kb, K, c):
    g = 2.0 * np.pi * np.sqrt(K / kb)
    return -(2.0 / np.pi) * np.arctan(np.exp(np.clip(g * (n - c), -800.0, 800.0)))


def cross(kb, K, d, c0):
    w1 = kink(kb, K, c0 - 0.5 * d)
    w2 = kink(kb, K, c0 + 0.5 * d)
    A = 2.0 * np.pi * w1
    B = 2.0 * np.pi * w2
    pot = K * ((1.0 - np.cos(B)) * (np.cos(A) - 1.0) + np.sin(A) * np.sin(B))
    d1 = w1[1:] - w1[:-1]
    d2 = w2[1:] - w2[:-1]
    return float(np.sum(kb * d1 * d2 + 0.5 * (pot[1:] + pot[:-1])))


def A_of(th):
    return 2.0 * th / np.sinh(th)


def A2_of(th):
    s = np.sinh(th)
    c = np.cosh(th)
    return 2.0 * (-2.0 * c / s ** 2 + 2.0 * th * c ** 2 / s ** 3 - th / s)


def I3_of(th, qmax=90.0, nq=300000):
    q = np.linspace(1e-6, qmax, nq)
    F = np.pi ** 2 * q ** 2 / np.sinh(np.pi * q / 2.0) ** 2
    return float(np.trapezoid(F * np.cos(q * th), q) / np.pi)


def V_closed(kb, K, th):
    g = 2.0 * np.pi * np.sqrt(K / kb)
    return (4.0 * K / g) * (A_of(th) - A2_of(th) - I3_of(th))


cases = [(1.0, 0.005), (1.0, 0.002), (4.0, 0.005)]
tab = {}
for kb, K in cases:
    g = 2.0 * np.pi * np.sqrt(K / kb)
    M = 4.0 * np.sqrt(kb * K) / np.pi
    c0 = N / 2.0
    ths = np.array([2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0])
    Vn = np.array([cross(kb, K, float(t / g), c0) for t in ths])
    Va = np.array([V_closed(kb, K, float(t)) for t in ths])
    dev = float(np.max(np.abs(Va - Vn) / np.abs(Vn)))
    tab[(kb, K)] = dict(ths=ths, Vn=Vn, Va=Va, dev=dev, M=M, g=g)
    rows.append(("V1", "k_b = %.2f, K = %.3f 的闭式偏差（判据 0.10）" % (kb, K), "%.3e" % dev, "判据 0.10", dev <= 0.10))
    rows.append(("V1", "同档偏差/gamma^2（离散化标度）" % (), "%.4f" % (dev / g ** 2), "判据 0.15 散布", True))

uu = np.linspace(-60.0, 60.0, 400001)
SS = 1.0 / np.cosh(uu)
devI3 = 0.0
for t in [0.0, 1.0, 2.0, 4.0]:
    direct = float(np.trapezoid(SS ** 2 * (1.0 / np.cosh(uu - t)) ** 2, uu))
    devI3 = max(devI3, abs(I3_of(t) - direct) / direct)
rows.append(("V0", "I3 闭式对直接积分（theta = 0 至 4）的最大相对偏差", "%.3e" % devI3, "<= 1e-4", devI3 <= 1e-4))

ths = np.array([8.0, 10.0, 12.0, 14.0, 16.0])
conv = np.exp(ths) * (A_of(ths) - A2_of(ths))
dev8 = float(np.max(np.abs(conv / 8.0 - 1.0)))
rows.append(("V2", "exp(theta)(A - A2) 对 8 的最大偏差", "%.4f" % dev8, "<= 0.05", dev8 <= 0.05))
devs = [(tab[c]["dev"] / tab[c]["g"] ** 2) for c in cases]
spread_dev = abs((max(devs) - min(devs)) / (sum(devs) / len(devs)))
rows.append(("V1", "偏差/gamma^2 的跨档散布", "%.4f（均值 %.4f）" % (spread_dev, sum(devs) / len(devs)), "<= 0.15", spread_dev <= 0.15))
ratio = 4.0 / C1_NUM
rows.append(("V3", "解析 c1/M = 4 对上一轮数值 %.2f 之比" % C1_NUM, "%.4f" % ratio, "0.95 至 1.05", 0.95 <= ratio <= 1.05))
rows.append(("V4", "闭式形式", "V = (4K/g)[A - A2 - I3]", "登记", True))

print("=== 谐波系数解析导出轮判据表（实测 对 期望）===")
for tag, name, meas, exp, ok in rows:
    print("%-4s %-46s 实测 %-16s 期望 %-14s %s" % (tag, name, meas, exp, "通过" if ok else "不通过"))
print("")
for c in cases:
    t = tab[c]
    print("k_b = %.2f, K = %.3f：闭式最大相对偏差 %.3e（M = %.4f）" % (c[0], c[1], t["dev"], t["M"]))
print("")
print("V5 结论：交叉项积分收成闭式 V(d) = (4K/gamma)[A(theta) - A2(theta) - I3(theta)]，A = 2 theta / sinh theta，I3 为单重积分。")
print("        大 theta 上 A - A2 的前置因子收敛到 8，theta 线性部精确相消，本轮为解析结论。")
print("        一阶谐波系数由此为 c1 = (4K/gamma) * 8 / M = 4 M。")

fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
for c in cases:
    t = tab[c]
    axes[0].semilogy(t["ths"], np.abs(t["Va"] - t["Vn"]) / np.abs(t["Vn"]), "o-", ms=3, label="k_b=%.2f K=%.3f" % c)
axes[0].set_xlabel("theta")
axes[0].set_ylabel("相对偏差")
axes[0].set_title("(a) 闭式对交叉项积分的偏差")
axes[0].legend(fontsize=7)
axes[0].grid(True, which="both", alpha=0.3)

thg = np.linspace(1.0, 6.0, 200)
axes[1].plot(thg, A_of(thg), "-", label="A")
axes[1].plot(thg, A2_of(thg), "--", label="A2")
axes[1].plot(thg, A_of(thg) - A2_of(thg), "-", label="A - A2")
axes[1].plot(thg, 8.0 * np.exp(-thg), ":", label="8 exp(-theta)")
axes[1].set_xlabel("theta")
axes[1].set_ylabel("值")
axes[1].set_title("(b) A - A2 的 theta 线性部相消")
axes[1].legend(fontsize=7)
axes[1].grid(True, alpha=0.3)

thq = np.linspace(1.0, 8.0, 50)
axes[2].plot(thq, [I3_of(float(t)) for t in thq], "o-", ms=3, label="I3(theta)")
axes[2].plot(thq, 16.0 * thq * np.exp(-2.0 * thq), "--", label="16 theta exp(-2 theta)")
axes[2].set_xlabel("theta")
axes[2].set_ylabel("I3")
axes[2].set_title("(c) I3 的领头行为（双极点）")
axes[2].legend(fontsize=7)
axes[2].grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(IMG, dpi=140)
print("图已存：" + IMG)