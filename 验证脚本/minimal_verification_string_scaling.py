# -*- coding: utf-8 -*-
r"""
minimal_verification_string_scaling.py

弦张力标度轮：多档 K 上测弦张力，定其标度（环的稳定半径依赖这个标度）。

来源：作者 2026-09-20 指令（继续）。上一轮[[验证脚本/注记-弦张力轮]]登记「张力的 K 标度未定」（σ/√K 两档散布 65%）。

模型：2D 横截面，E = sum [ (1/2) k (gx^2 + gy^2) + K (1 - cos 2 pi u) ]，k = 1；
      涡旋对构型 u = (1/2pi)[arg(z-z1) - arg(z-z2)]，棱差取解析梯度的格点投影，势项周期单值。
      张力 σ 取 E(d) 对间距 d 的线性斜率。

判据（先行）：
  U1 σ 对 K 的幂律拟合：指数 q 与残差（残差 <= 0.03），候选 q = 1（指数每格代价）或 q = 1/2（光滑墙张力）
  U2 若 q ≈ 1，则 c = σ/K 的跨档散布 <= 0.20
  U3 与解析对照：沿切割的势能代价每格 2K，给出 σ 中该部分与实测的比值（量级读数）
  U4 登记：标度与对环稳定半径的推论
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\string_scaling.png"
rows = []


def coords(N):
    a = np.arange(N, dtype=float) - 0.5 * (N - 1)
    return np.meshgrid(a, a, indexing="ij")


def pair(N, d):
    X, Y = coords(N)
    x1, x2 = -0.5 * d, 0.5 * d
    r1 = (X - x1) ** 2 + Y ** 2 + 1e-12
    r2 = (X - x2) ** 2 + Y ** 2 + 1e-12
    u = (np.arctan2(Y, X - x1) - np.arctan2(Y, X - x2)) / (2.0 * np.pi)
    gx = (1.0 / (2.0 * np.pi)) * (Y / r1 - Y / r2)
    gy = (1.0 / (2.0 * np.pi)) * (-(X - x1) / r1 + (X - x2) / r2)
    return u, gx, gy


def energy(u, gx, gy, K, k=1.0):
    return 0.5 * k * float(np.sum(gx ** 2 + gy ** 2)) + K * float(np.sum(1.0 - np.cos(2.0 * np.pi * u)))


Ks = [0.02, 0.01, 0.005, 0.002, 0.001]
ds = np.array([6.0, 8.0, 10.0, 12.0, 14.0, 16.0])
N = 96
sig = []
for K in Ks:
    Ep = np.array([energy(*pair(N, float(d)), K) for d in ds])
    s, b = np.polyfit(ds, Ep, 1)
    rel = float(np.max(np.abs(Ep - (b + s * ds))) / np.max(np.abs(Ep)))
    sig.append(float(s))
    rows.append(("U1", "K = %.3f 的张力 sigma（线性拟合残差）" % K, "%.4f（%.4f）" % (s, rel), "残差 <= 0.05", rel <= 0.05))
sig = np.array(sig)
Karr = np.array(Ks)
q, lc = np.polyfit(np.log(Karr), np.log(sig), 1)
pred = np.exp(lc) * Karr ** q
res = np.abs(np.log(sig / pred))
rows.append(("U1", "幂律指数 q（σ = c K^q）与最大十进残差", "%.4f / %.4f" % (q, float(res.max())), "残差 <= 0.03", float(res.max()) <= 0.03))
cs = sig / Karr
spr = float((cs.max() - cs.min()) / cs.mean())
rows.append(("U2", "若 q ≈ 1 则 σ/K 的跨档散布（均值）", "%.4f（%.2f）" % (spr, float(cs.mean())), "<= 0.20", spr <= 0.20))
ratio = float(cs.mean()) / 2.0
rows.append(("U3", "σ 中每格 2K 部分与实测均值之比（量级读数）", "%.1f" % ratio, "量级读数", True))
rows.append(("U4", "登记：标度与对环稳定半径的推论", "见 §三", "登记", True))

print("=== 弦张力标度轮判据表（实测 对 期望）===")
for tag, name, meas, exp, ok in rows:
    print("%-4s %-50s 实测 %-18s 期望 %-16s %s" % (tag, name, meas, exp, "通过" if ok else "不通过"))
print("")
print("U5 结论：张力对 K 的幂律指数 q = %.4f（残差 %.4f 十进对数），σ/K 的跨档均值为 %.2f（散布 %.2f）。" % (q, float(res.max()), float(cs.mean()), spr))
print("        若 q 接近 1，则张力线性于刚度，环的能量 ≈ 2 pi R sigma 随刚度线性；若接近 1/2，则为光滑墙的 sqrt(K) 标度。")
print("        解析对照：切割处每格的势能代价为 2K，σ 的该部分与实测均值的比为 %.1f。" % ratio)

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
axes[0].loglog(Karr, sig, "o-", label="σ(K) 实测")
axes[0].loglog(Karr, np.exp(lc) * Karr ** q, "--", label="拟合 σ = %.3f K^%.3f" % (np.exp(lc), q))
axes[0].loglog(Karr, cs.mean() * Karr, ":", label="线性参考 %.2f K" % cs.mean())
axes[0].set_xlabel("K")
axes[0].set_ylabel("弦张力 σ")
axes[0].set_title("(a) 张力对刚度的标度")
axes[0].legend(fontsize=8)
axes[0].grid(True, which="both", alpha=0.3)

axes[1].semilogx(Karr, cs, "o-", ms=3, label="σ/K")
axes[1].set_xlabel("K")
axes[1].set_ylabel("σ/K")
axes[1].set_title("(b) σ/K 的跨档稳定性（散布 %.2f）" % spr)
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(IMG, dpi=140)
print("图已存：" + IMG)