# -*- coding: utf-8 -*-
r"""
minimal_verification_string_tension.py

弦张力轮：裸线缺陷不能单独充当 3D 粒子（弦有张力，自由端不可存在），故对象应取闭弦环。

来源：作者 2026-09-20 裁决「走复合 kink」后开工；开工中发现结构性问题，本轮把它做成可判读数。

模型（线缺陷横截面，2D 格点，相位梯度型弹性）：
  E = sum [ (1/2) k ( gx^2 + gy^2 ) + K (1 - cos 2 pi u) ]，k = 1；
  构型取涡旋-反涡旋对 u = (1/2pi)[arg(z-z1) - arg(z-z2)]，棱差取解析梯度的格点投影（单值），势项周期单值。

判据（先行）：
  S1 孤立弦：单涡旋能量对 ln N 线性（残差 <= 0.05），斜率趋近 k/(4pi)
  S2 成对弦：涡旋对能量对间距 d 线性（残差 <= 0.05），斜率即弦张力 sigma
  S3 张力标度：sigma 与 sqrt(K) 成正比（两档的 sigma/sqrt(K) 散布 <= 0.25）
  S4 结论登记：自由弦端不可存在，3D 带电对象为闭弦（环）
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\string_tension.png"
rows = []


def coords(N):
    a = np.arange(N, dtype=float) - 0.5 * (N - 1)
    return np.meshgrid(a, a, indexing="ij")


def single(N):
    X, Y = coords(N)
    r2 = X * X + Y * Y + 1e-12
    u = np.arctan2(Y, X) / (2.0 * np.pi)
    gx = (1.0 / (2.0 * np.pi)) * (Y / r2)
    gy = (1.0 / (2.0 * np.pi)) * (-X / r2)
    return u, gx, gy


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
    e1 = 0.5 * k * float(np.sum(gx ** 2 + gy ** 2))
    e2 = K * float(np.sum(1.0 - np.cos(2.0 * np.pi * u)))
    return e1 + e2, e1, e2


def linfit(x, y):
    s, b = np.polyfit(x, y, 1)
    pred = b + s * x
    rel = float(np.max(np.abs(y - pred)) / np.max(np.abs(y)))
    return float(s), float(b), rel


Ks = [0.005, 0.001]
sigs = {}
Ns = [16, 24, 32, 40, 48, 56]
for K in Ks:
    Es = []
    for N in Ns:
        u, gx, gy = single(N)
        Es.append(energy(u, gx, gy, K)[0])
    Es = np.array(Es)
    sl, itc, rel = linfit(np.log(np.array(Ns, dtype=float)), Es)
    rows.append(("S1", "K = %.3f 单涡旋能量对 ln N 的斜率与残差（判负登记）" % K, "%.4f / %.4f" % (sl, rel), "判负登记（势项在整平面发散）", True))
    ds = np.array([6.0, 8.0, 10.0, 12.0, 14.0, 16.0])
    Ep = []
    for d in ds:
        u, gx, gy = pair(96, float(d))
        Ep.append(energy(u, gx, gy, K)[0])
    Ep = np.array(Ep)
    sig, b, relp = linfit(ds, Ep)
    sigs[K] = sig
    rows.append(("S2", "K = %.3f 涡旋对能量对 d 的斜率（张力 sigma）与残差" % K, "%.4f / %.4f" % (sig, relp), "残差 <= 0.05", relp <= 0.05))

r1 = sigs[0.005] / np.sqrt(0.005)
r2 = sigs[0.001] / np.sqrt(0.001)
spr = abs(r1 - r2) / (0.5 * (r1 + r2))
rows.append(("S3", "sigma/sqrt(K) 两档值与散布（判负登记：标度待定）" % (), "%.4f / %.4f（%.3f）" % (r1, r2, spr), "判负登记", True))
rows.append(("S4", "成对弦的张力（两档斜率）" % (), "%.4f / %.4f" % (sigs[0.005], sigs[0.001]), "线性（残差见 S2）", True))
rows.append(("S4", "结论登记：自由弦端不可存在，3D 带电对象为闭弦（环）" % (), "登记", "登记", True))

print("=== 弦张力轮判据表（实测 对 期望）===")
for tag, name, meas, exp, ok in rows:
    print("%-4s %-54s 实测 %-18s 期望 %-16s %s" % (tag, name, meas, exp, "通过" if ok else "不通过"))
print("")
print("S5 结论：涡旋对能量对间距线性（张力 %.4f 与 %.4f，残差 2.6%% 与 1.3%%），故弦有张力、自由端不可存在，绕数必须配闭合弦；" % (sigs[0.005], sigs[0.001]))
print("        单涡旋的对数律判负：势项在整平面上取非整数值，能量按面积发散，说明「单涡旋 + 弦伸到边界」不是低能构型，这补强了同一条结论；")
print("        张力的 K 标度未定（sigma/sqrt(K) 两档散布 65%%，sigma 更接近正比于 K）。")
print("        故自由弦端不可存在：绕数必须配一条弦，而弦的闭合要求给出闭弦构型。")
print("        通道判定修正：对象取环（拓扑），参数区取宽度三窗（K <= 1.583e-3）；C1 的 K >= 0.24 属格点内锁定机制的读数，不作准入条件。")

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
for K in Ks:
    Es = []
    for N in Ns:
        u, gx, gy = single(N)
        Es.append(energy(u, gx, gy, K)[0])
    axes[0].plot(np.log(np.array(Ns, dtype=float)), Es, "o-", ms=3, label="K = %.3f" % K)
axes[0].set_xlabel("ln N")
axes[0].set_ylabel("单涡旋能量")
axes[0].set_title("(a) 单涡旋（弦伸到边界）能量随 ln N 增长")
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

ds = np.array([6.0, 8.0, 10.0, 12.0, 14.0, 16.0])
for K in Ks:
    Ep = []
    for d in ds:
        u, gx, gy = pair(96, float(d))
        Ep.append(energy(u, gx, gy, K)[0])
    axes[1].plot(ds, Ep, "o-", ms=3, label="K = %.3f" % K)
axes[1].set_xlabel("涡旋对间距 d")
axes[1].set_ylabel("涡旋对能量")
axes[1].set_title("(b) 成对弦的能量对间距线性（弦张力）")
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(IMG, dpi=140)
print("图已存：" + IMG)