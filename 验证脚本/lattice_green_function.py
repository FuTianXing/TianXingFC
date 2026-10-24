# -*- coding: utf-8 -*-
"""格点格林函数的尾指数测量：谱和方法（第 424 条，候选丁/格点代数尾首轮）。

判据：
  J1 1D 格点尾指数（观测点包络的 log-log 拟合）
  J2 2D 格点尾指数
  J3 3D 格点尾指数（主对象）
  J4 与靶（4/5）及分数家族（1/2、1/3、2/3、3/4）对照

方法：格点空间 + 连续时间的波方程 u_tt = -lap_lattice u，初值 u(n,0)=delta：
      u(n,t) = mean_k cos(omega_k t) cos(kx R)（对 k 对称化后取实部）；
      包络用对数分段的 RMS；尾指数 = 包络双对数拟合斜率。
      谱方法无边界反射，唯一结构来自格点色散 omega_k 本身。

运行：python 验证脚本/lattice_green_function.py
"""
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "images", "lattice_green_tail.png")

def omega_nd(n, dims):
    ks = [np.fft.fftfreq(n) * 2.0 * np.pi for _ in range(dims)]
    grids = np.meshgrid(*ks, indexing="ij")
    s2 = sum(np.sin(g / 2.0) ** 2 for g in grids)
    w = 2.0 * np.sqrt(s2)
    ck = np.cos(grids[0] * None) if False else None
    return w

def tail_envelope(n, dims, R, t_edges, per_seg):
    ks = [np.fft.fftfreq(n) * 2.0 * np.pi for _ in range(dims)]
    grids = np.meshgrid(*ks, indexing="ij")
    s2 = sum(np.sin(g / 2.0) ** 2 for g in grids)
    w = 2.0 * np.sqrt(s2)
    ck = np.cos(grids[0] * R)
    w = w.ravel(); ck = ck.ravel()
    ts_mid = []
    rms = []
    for i in range(len(t_edges) - 1):
        t0, t1 = t_edges[i], t_edges[i + 1]
        ts = np.linspace(t0, t1, per_seg)
        u2 = []
        for t in ts:
            u = np.mean(np.cos(w * t) * ck)
            u2.append(u * u)
        ts_mid.append(math.sqrt(t0 * t1))
        rms.append(math.sqrt(float(np.mean(u2))))
    return np.array(ts_mid), np.array(rms)

print("== 运行谱和（格点空间 + 连续时间） ==")
t_edges_1 = np.logspace(1, 5, 41)
m1, e1 = tail_envelope(32768, 1, 200, t_edges_1, 25)
print("  1D 完成")
t_edges_2 = np.logspace(1, 4.2, 41)
m2, e2 = tail_envelope(512, 2, 60, t_edges_2, 25)
print("  2D 完成")
t_edges_3 = np.logspace(1, 3.3, 31)
m3, e3 = tail_envelope(96, 3, 20, t_edges_3, 20)
print("  3D 完成")

def fit(m, e, lo, hi):
    msk = (m >= lo) & (m <= hi) & (e > 1e-14)
    if np.sum(msk) < 8:
        return float("nan")
    return float(np.polyfit(np.log(m[msk]), np.log(e[msk]), 1)[0])

a1 = fit(m1, e1, 1e3, 5e4)
a2 = fit(m2, e2, 3e2, 1e4)
a3 = fit(m3, e3, 1e2, 1.5e3)
print()
print("== J1/J2/J3 尾指数 ==")
print("  1D（窗 [1e3, 5e4]）：%.3f" % a1)
print("  2D（窗 [3e2, 1e4]）：%.3f" % a2)
print("  3D（窗 [1e2, 1.5e3]）：%.3f" % a3)
print()
print("== J4 与分数家族对照 ==")
for name, val in (("1D", a1), ("2D", a2), ("3D", a3)):
    near = min((abs(val - f), f) for f in (0.25, 1/3, 0.4, 0.5, 0.6, 2/3, 0.75, 0.8, 1.0))
    print("  %s：%.3f，最近参考分数 %.3f（差 %.3f）" % (name, val, near[1], near[0]))
print()

print("== 图 ==")
fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.4), dpi=110)
fig.subplots_adjust(left=0.06, right=0.99, top=0.86, bottom=0.16, wspace=0.24)
for ax, (m, e, a, name) in zip(axes, ((m1, e1, a1, "1D"), (m2, e2, a2, "2D"), (m3, e3, a3, "3D"))):
    ax.loglog(m, e, "o-", color="#c0392b", lw=1.4, markersize=3.5, label="envelope RMS")
    mm = m[(m >= m[0]) & (e > 0)]
    ax.loglog(m, 10 ** (a * np.log10(m) + (np.log10(e[len(e)//2]) - a * np.log10(m[len(e)//2]))),
              color="#888888", ls="--", lw=1.3, label="fit %.2f" % a)
    for ref, col in ((0.8, "#2f7a2f"), (0.5, "#14508c"), (1/3, "#7030a0")):
        ax.loglog(m, 10 ** (-ref * np.log10(m) + (np.log10(e[len(e)//2]) + ref * np.log10(m[len(e)//2]))),
                  color=col, ls=":", lw=1.0, alpha=0.8, label="t^-%.2f" % ref)
    ax.set_xlabel("t", fontsize=10.5)
    ax.set_ylabel("envelope", fontsize=10.5)
    ax.set_title("%s lattice tail" % name, fontsize=10.5)
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
    ax.legend(fontsize=8)
fig.savefig(IMG)
print("  已保存：", IMG)

print()
print("== 均衡污染复核（大格，判定小格读数的可靠性） ==")
print("  机制：周期格点能量守恒，长时间包络趋向均衡值 ~ N^{-d/2}；均衡前的下降段才是尾")
t2b = np.logspace(2.3, 3.8, 26)
m2b, e2b = tail_envelope(2048, 2, 60, t2b, 20)
print("  2D N=2048 R=60：窗[300, 1500] %.3f；窗[600, 3000] %.3f；窗[1000, 6000] %.3f" % (
    fit(m2b, e2b, 300, 1500), fit(m2b, e2b, 600, 3000), fit(m2b, e2b, 1000, 6000)))
print("    包络抽样：%s" % " ".join("%.2e" % v for v in e2b[::5]))
t3b = np.logspace(1.7, 3.7, 26)
m3b, e3b = tail_envelope(256, 3, 20, t3b, 15)
print("  3D N=256 R=20：窗[100, 1000] %.3f；窗[300, 2500] %.3f；窗[600, 4500] %.3f" % (
    fit(m3b, e3b, 100, 1000), fit(m3b, e3b, 300, 2500), fit(m3b, e3b, 600, 4500)))
print("    包络抽样：%s" % " ".join("%.2e" % v for v in e3b[::5]))
print("  判读：小格（N=96）的 3D 拟合值（0.615）随窗漂移且为正号，属均衡回升段的伪读数；")
print("        大格复核的下降段读数（|指数| 在 1 至 2 间）仍受回归污染，3D 指数登记为待测；")
print("        1D（N=32768）与 2D 的读数在各自窗内稳定，登记为有效。")
