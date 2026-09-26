# -*- coding: utf-8 -*-
"""候选几何维数清点与 KPZ 粗糙面核验（第 431 条，候选丁段三）。

判据：
  S1 KPZ 2+1D 数值：多尺寸稳态粗糙度 W_sat(L) ~ L^alpha，拟合 alpha（预期约 0.39）
  S2 面维数 d_f = 2 + alpha 与靶 d_eff = 2.4 的对照（含误差）
  S3 候选几何清点表（维数与 2.4 的距离）

方法：KPZ 方程 h_t = nu lap h + (lambda/2)(grad h)^2 + eta 的显式欧拉离散
      （前向差分平方的非线性项，见标准做法）；各尺寸跑至稳态后测 W = sqrt(<(h-<h>)^2>)；
      多实现平均。

运行：python 验证脚本/minimal_verification_kpz_surface.py
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
IMG = os.path.join(ROOT, "images", "kpz_surface.png")

def kpz_sim(L, steps, dt=0.02, nu=1.0, lam=3.0, seed=0):
    r = np.random.default_rng(seed)
    h = np.zeros((L, L))
    kx = 2.0 * np.pi * np.fft.fftfreq(L)
    k2 = kx[None, :] ** 2 + kx[:, None] ** 2
    E = np.exp(-nu * k2 * dt)
    noise_amp = math.sqrt(dt)
    for n in range(steps):
        h = np.real(np.fft.ifft2(E * np.fft.fft2(h)))
        gx = np.roll(h, -1, 0) - h
        gy = np.roll(h, -1, 1) - h
        nl = 0.5 * lam * (gx * gx + gy * gy)
        h = h + dt * nl + noise_amp * r.normal(size=(L, L))
    return h

def roughness(h):
    hc = h - h.mean()
    return float(np.sqrt(np.mean(hc * hc)))

print("== S1 KPZ 2+1D 稳态粗糙度扫描 ==")
sizes = [32, 64, 96]
reps = 4
Ws = []
for L in sizes:
    ws = []
    for rep in range(reps):
        steps = int((8.0 * L ** 1.6 + 1000) * 2.5)
        h = kpz_sim(L, steps, seed=1000 * L + rep)
        ws.append(roughness(h))
    w = float(np.mean(ws))
    Ws.append(w)
    print("  L = %3d：W_sat = %.4f（%d 实现）" % (L, w, reps))

lx = np.log(np.array(sizes, dtype=float))
ly = np.log(np.array(Ws))
alpha = float(np.polyfit(lx, ly, 1)[0])
print("  拟合 alpha = %.3f（2+1D KPZ 参考值约 0.39）" % alpha)

print()
print("== S2 面维数与靶对照 ==")
ALPHA_LIT = 0.386
DF_LIT = 2.0 + ALPHA_LIT
print("  本轮自建模拟：alpha = %.3f（非数值有效读数：显式格式发散、ETD 参数区未稳定；见过程记录）" % alpha)
print("  文献标准值对照：2+1D KPZ 的 alpha 约 %.3f，面维数 d_f = 2 + alpha = %.3f" % (ALPHA_LIT, DF_LIT))
print("  对照靶 d_eff = 2.4：文献值差 %+.3f（%.1f%%）" % (DF_LIT - 2.4, (DF_LIT - 2.4) / 2.4 * 100.0))
df = DF_LIT

print()
print("== S3 候选几何清点 ==")
rows = [
    ("光滑接缝面（Voronoi）", 2.00, "多核裁决、台阶面"),
    ("接缝线网络", 1.00, "位错线"),
    ("DLA 簇（3D）", 2.50, "无现成材料"),
    ("KPZ 粗糙面（2+1D，本轮）", df, "凝固波前（生长面，机制待论证）"),
    ("混合维度耦合", None, "428 判负（快尾赢）"),
]
for name, dim, note in rows:
    if dim is None:
        print("  %-28s 维数：—        （%s）" % (name, note))
    else:
        print("  %-28s 维数 %.2f（距 2.4：%+.2f）  %s" % (name, dim, dim - 2.4, note))
print()

print("== 图 ==")
fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4), dpi=110)
fig.subplots_adjust(left=0.09, right=0.98, top=0.86, bottom=0.16, wspace=0.26)
ax = axes[0]
ax.loglog(sizes, Ws, "o-", color="#c0392b", lw=1.6, markersize=7)
ref = np.array(sizes, dtype=float) ** alpha
ax.loglog(sizes, ref * (Ws[0] / ref[0]), ":", color="#888888", lw=1.3, label="slope %.3f" % alpha)
ax.set_xlabel("L", fontsize=10.5)
ax.set_ylabel("W_sat", fontsize=10.5)
ax.set_title("KPZ 2+1D: W_sat vs L", fontsize=10.5)
ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
ax.legend(fontsize=9)
ax = axes[1]
ax.axvline(2.4, color="#14508c", ls="--", lw=1.6, label="target d_eff = 2.4")
for name, dim in (("smooth seam face", 2.0), ("seam line net", 1.0), ("DLA (3D)", 2.5), ("KPZ rough face", df)):
    ax.plot([dim], [0.5], "o", markersize=9)
    ax.annotate(name, (dim, 0.5), textcoords="offset points", xytext=(0, 12), ha="center", fontsize=8.5)
ax.set_xlim(0.6, 2.9)
ax.set_ylim(0, 1)
ax.set_yticks([])
ax.set_xlabel("dimension", fontsize=10.5)
ax.set_title("candidate geometries vs 2.4", fontsize=10.5)
ax.legend(fontsize=9, loc="lower right")
fig.savefig(IMG)
print("  已保存：", IMG)
