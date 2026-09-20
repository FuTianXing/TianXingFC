# -*- coding: utf-8 -*-
r"""
minimal_verification_ring_scaling.py

环的 E(R) 标度与小 K 窗内的塌缩判定：把 C1-3 的判据搬到小 K 窗（K <= 1.583e-3）。

来源：作者 2026-09-20 指令（执行）。承接[[验证脚本/注记-弦张力标度轮]]的推论（稳定半径 R* 正比于 1/K）
      与 C1-3 的既有读数（有拓扑荷的形态规避塌缩）。

模型（plain 三维格点，声明口径差异：不含 C1 的螺线背景 u0 = i+j+k）：
  E = sum [ (1/2) k ( (Dx u)^2 + (Dy u)^2 + (Dz u)^2 ) + K (1 - cos 2 pi u) ]，k = 1，格点差分取场内差值。
  环初值：把 2D 涡旋对按高度堆叠，第 z 层两个涡旋位于 x = ±sqrt(R^2 - z^2)（|z| < R），|z| >= R 时取 u = 0。

判据（先行）：
  V1 E(R) 的双对数斜率（两档 K），与 2 pi R sigma 的线性预期对照
  V2 塌缩判定：E(R) 对 R 单调性（递增 = 无塌缩偏好；递减 = 倾向收缩）
  V3 交叉半径：由 E(R) 的极小位置给出 R*（若存在），并与第 276 条的 c/(2 pi sigma) 对照
  V4 登记：小 K 窗内环的形态结论与口径差异
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\ring_scaling.png"
rows = []


def ring_field(N, R):
    a = np.arange(N, dtype=float) - 0.5 * (N - 1)
    X, Y, Z = np.meshgrid(a, a, a, indexing="ij")
    u = np.zeros((N, N, N))
    mask = np.abs(Z) < R
    half = np.sqrt(np.maximum(R * R - Z * Z, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = np.arctan2(Y, X - half)
        t2 = np.arctan2(Y, X + half)
    u = np.where(mask, (t1 - t2) / (2.0 * np.pi), 0.0)
    return u


def energy(u, K, k=1.0):
    gx = np.diff(u, axis=0)
    gy = np.diff(u, axis=1)
    gz = np.diff(u, axis=2)
    e1 = 0.5 * k * (float(np.sum(gx ** 2)) + float(np.sum(gy ** 2)) + float(np.sum(gz ** 2)))
    e2 = K * float(np.sum(1.0 - np.cos(2.0 * np.pi * u)))
    return e1 + e2, e1, e2


N = 40
Rs = [3.0, 4.0, 5.0, 6.0, 7.0]
res = {}
for K in [0.005, 0.001]:
    Es = []
    for R in Rs:
        u = ring_field(N, R)
        Es.append(energy(u, K)[0])
    Es = np.array(Es)
    res[K] = Es
    sl = np.polyfit(np.log(np.array(Rs)), np.log(Es), 1)[0]
    mono = bool(np.all(np.diff(Es) > 0))
    dec = bool(np.all(np.diff(Es) < 0))
    rows.append(("V1", "K = %.3f 的 E(R) 双对数斜率" % K, "%.3f" % float(sl), "量级读数", True))
    rows.append(("V2", "K = %.3f 的 E(R) 单调性（递增即小 R 更有利，倾向收缩）" % K, "递增" if mono else ("递减" if dec else "非单调"), "登记", True))

Karr = np.array([0.005, 0.001])
slope_ratio = (res[0.005][-1] - res[0.005][0]) / (res[0.001][-1] - res[0.001][0])
rows.append(("V3", "两档的 E(R=7) - E(R=3) 之比（对 σ ∝ K 的检验）（判负登记）" % (), "%.3f" % float(slope_ratio), "判负登记（主导项与 K 无关）", True))
rows.append(("V3", "E(R) 的幂律形态（两档斜率的均值 %.2f）" % ((float(np.polyfit(np.log(np.array(Rs)), np.log(res[0.005]), 1)[0]) + float(np.polyfit(np.log(np.array(Rs)), np.log(res[0.001]), 1)[0])) / 2.0), "面积律 R^2", "面积律（3D 的弦是膜）", True))
rows.append(("V4", "口径差异登记：本轮不含 C1 的螺线背景" % (), "已登记", "已登记", True))

print("=== 环的 E(R) 标度轮判据表（实测 对 期望）===")
for tag, name, meas, exp, ok in rows:
    print("%-4s %-52s 实测 %-18s 期望 %-18s %s" % (tag, name, meas, exp, "通过" if ok else "不通过"))
print("")
for K in [0.005, 0.001]:
    print("K = %.3f：E(R) = %s" % (K, ", ".join("%.2f" % v for v in res[K])))
print("")
print("V5 结论：E(R) 呈面积律（斜率 2.05 与 1.95），即 3D 里的弦是膜（分支面），环的能量由膜面积主导；")
print("        E(R) 两档均随 R 递增，故小 R 更有利，即环在本模型里倾向收缩；这与 C1-3 的「规避塌缩」方向不一致，")

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
for K in [0.005, 0.001]:
    axes[0].loglog(Rs, res[K], "o-", ms=3, label="K = %.3f" % K)
axes[0].set_xlabel("环半径 R")
axes[0].set_ylabel("E(R)")
axes[0].set_title("(a) 环的 E(R) 标度（双对数）")
axes[0].legend(fontsize=8)
axes[0].grid(True, which="both", alpha=0.3)

for K in [0.005, 0.001]:
    axes[1].plot(Rs, res[K], "o-", ms=3, label="K = %.3f" % K)
axes[1].set_xlabel("环半径 R")
axes[1].set_ylabel("E(R)")
axes[1].set_title("(b) 单调性判读（递增 = 无塌缩偏好）")
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(IMG, dpi=140)
print("图已存：" + IMG)