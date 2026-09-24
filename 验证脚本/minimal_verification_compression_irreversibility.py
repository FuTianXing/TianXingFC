# -*- coding: utf-8 -*-
"""
判据 L2 的最小验证：压缩的不可逆方向（粗粒化信息损失，验证脚本第 401 条）

判据（先立后算，2026-09-25，承内部草稿/注记-认知错视与分层粗粒化 的 L2）：
  L2a 信息不增（数据处理不等式）：I(X;Z) <= I(X;Y) <= H(X)，两处均不取等（有损）。
  L2b 不可逆：由 Z 恢复 X 的逐位最优准确率 <= 由 Y 的准确率 < 1；损失 DeltaI = I(X;Y) - I(X;Z) > 0。
  L2c 损失随粗粒化细化而减小：档数 q 从 2 到 7，I(X;Z) 单调不减，q = 7 时与 I(X;Y) 取等。
  L2d 任务性能与信息量不同步：逐位恢复准确率不随 q 单调（q = 3 低于其两侧）。

模型：X 为 24 个独立均匀比特，分 4 块每块 6 位；Y_k 为块内和；Z_k 为 Y_k 的 q 档量化。
理论期望：I(X;Y) = 4 * H_B(6, 1/2) = 9.3334 bit；acc(Y) = 0.65625；acc(Z_3) = 0.578125。
"""
from math import comb
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

IMG = "D:/software/TianXingFC/images/compression_irreversibility.png"
SEED = 20260925
NB = 6
KB = 4
NBIT = NB * KB
M_SAMPLE = 400000

p_s = np.array([comb(NB, s) for s in range(NB + 1)], dtype=float) / 2.0 ** NB


def entropy(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def bin_bounds(q):
    edges = np.linspace(0.0, NB + 1.0, q + 1)
    return [int(round(e)) for e in edges]


def coarse_dist(q):
    b = bin_bounds(q)
    return np.array([p_s[b[i]:b[i + 1]].sum() for i in range(q)])


def acc_analytic(q):
    b = bin_bounds(q)
    total = 0.0
    for i in range(q):
        lo, hi = b[i], b[i + 1]
        ps = p_s[lo:hi]
        ss = np.arange(lo, hi, dtype=float)
        w = ps.sum()
        if w <= 0:
            continue
        p1 = float((ps * ss).sum() / w) / NB
        total += w * max(p1, 1.0 - p1)
    return float(total)


print("=== 第 401 条：压缩的不可逆方向 ===")
print("模型：%d 个独立均匀比特，%d 块每块 %d 位；Y 为块内和，Z 为 Y 的 q 档量化" % (NBIT, KB, NB))
print("")

H_X = float(NBIT)
I_XY = KB * entropy(p_s)
Q3 = 3
I_XZ3 = KB * entropy(coarse_dist(Q3))
delta_I = I_XY - I_XZ3
acc_Y = acc_analytic(NB + 1)
acc_Z3 = acc_analytic(Q3)

print("--- L2a 信息不增 ---")
print("H(X)                               : 实测 %.6f bit   期望 %.6f" % (H_X, 24.0))
print("I(X;Y)                             : 实测 %.6f bit   期望 %.6f" % (I_XY, 9.333448))
print("I(X;Z)  q = 3                      : 实测 %.6f bit   期望 %.6f" % (I_XZ3, 3.906517))
print("判定 I(X;Z) <= I(X;Y) <= H(X)      : %s" % (I_XZ3 < I_XY < H_X))
print("")
print("--- L2b 不可逆 ---")
print("DeltaI = I(X;Y) - I(X;Z)           : 实测 %.6f bit   期望 > 0" % delta_I)
print("逐位恢复准确率 用 Y                : 实测 %.6f   期望 %.6f" % (acc_Y, 0.65625))
print("逐位恢复准确率 用 Z（q = 3）       : 实测 %.6f   期望 %.6f" % (acc_Z3, 0.578125))
print("无损口径（完全不粗粒化）           : 1.000000（定义）")
print("判定 acc(Z) <= acc(Y) < 1          : %s" % (acc_Z3 < acc_Y < 1.0))
print("")

rng = np.random.default_rng(SEED)
X = rng.integers(0, 2, size=(M_SAMPLE, KB, NB))
Y = X.sum(axis=2)
est_Y = (Y > NB / 2).astype(int)
est_Y[Y == NB // 2] = 0
acc_Y_mc = float((est_Y[:, :, None] == X).mean())
b3 = bin_bounds(Q3)
Z3 = np.digitize(Y, b3[1:-1])
est_Z3 = (Z3 == Q3 - 1).astype(int)
acc_Z3_mc = float((est_Z3[:, :, None] == X).mean())
print("--- 采样交叉验证（M = %d，种子 %d） ---" % (M_SAMPLE, SEED))
print("用 Y 的逐位准确率                  : 蒙特卡洛 %.6f   解析 %.6f" % (acc_Y_mc, acc_Y))
print("用 Z 的逐位准确率                  : 蒙特卡洛 %.6f   解析 %.6f" % (acc_Z3_mc, acc_Z3))
print("")

QS = list(range(2, NB + 2))
I_q = [KB * entropy(coarse_dist(q)) for q in QS]
acc_q = [acc_analytic(q) for q in QS]
mono = all(I_q[i] <= I_q[i + 1] + 1e-12 for i in range(len(QS) - 1))
print("--- L2c 档数扫描（I(X;Z) 单调不减，q = 7 取等） ---")
for q, iq, aq in zip(QS, I_q, acc_q):
    print("q = %d  I(X;Z) = %.6f bit   acc = %.6f" % (q, iq, aq))
print("单调性判定                         : %s" % mono)
print("q = 7 与 I(X;Y) 之差               : %.3e bit   期望 0" % abs(I_q[-1] - I_XY))
print("")
acc_vals = sorted(set(round(v, 12) for v in acc_q))
print("--- L2d 任务性能与信息量不同步 ---")
print("出现的准确率取值                   : %s" % ", ".join("%.6f" % v for v in acc_vals))
print("acc 是否随 I 单调                  : %s" % all(acc_q[i] <= acc_q[i + 1] + 1e-12 for i in range(len(QS) - 1)))
print("q = 2 与 q = 7 的准确率相等        : %s" % (abs(acc_q[0] - acc_q[-1]) < 1e-12))
print("")

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
ax = axes[0]
ax.bar(["H(X)", "I(X;Y)", "I(X;Z) q=3"], [H_X, I_XY, I_XZ3], color=["tab:gray", "tab:blue", "tab:orange"])
for i, v in enumerate([H_X, I_XY, I_XZ3]):
    ax.text(i, v + 0.4, "%.3f" % v, ha="center", fontsize=9)
ax.set_ylabel("比特")
ax.set_title("L2a：信息沿压缩方向不增")
ax.grid(alpha=0.3, axis="y")

ax = axes[1]
ax.bar(["随机猜测", "用 Z 恢复", "用 Y 恢复", "无损"], [0.5, acc_Z3, acc_Y, 1.0],
       color=["tab:gray", "tab:orange", "tab:blue", "tab:green"])
for i, v in enumerate([0.5, acc_Z3, acc_Y, 1.0]):
    ax.text(i, v + 0.015, "%.4f" % v, ha="center", fontsize=9)
ax.set_ylim(0.0, 1.15)
ax.set_ylabel("逐位恢复准确率")
ax.set_title("L2b：丢失不可追回")
ax.grid(alpha=0.3, axis="y")

ax = axes[2]
ax.plot(QS, I_q, "o-", color="tab:blue", label="I(X;Z)（比特）")
ax.set_xlabel("量化档数 q")
ax.set_ylabel("I(X;Z)（比特）", color="tab:blue")
ax2 = ax.twinx()
ax2.plot(QS, acc_q, "s--", color="tab:red", label="逐位恢复准确率")
ax2.set_ylabel("逐位恢复准确率", color="tab:red")
ax.set_title("L2c / L2d：信息单调而任务不单调")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(IMG, dpi=150)
print("图已保存：" + IMG)
