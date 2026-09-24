# -*- coding: utf-8 -*-
"""
判据 L4 的最小验证：切法与任务对齐（装置响应类型与读数质量，验证脚本第 402 条）

背景：第 401 条记下一处读数，粗粒化的切法是否与任务对齐，比它保留多少信息更重要。
本条把这一读数放到 M6 与裁决点 4 的口径上：装置响应类型即切法。

判据（先立后算，2026-09-25）：
  L4a 对齐无损（准确率口径）：等距 q = 2 的切法对任务 A 与任务 B 的准确率都等于用 Y 的准确率。
  L4b 错位有损（准确率口径）：等距 q = 3 的切法对两个任务的准确率都低于用 Y 的准确率。
  L4c 指标依赖（本条核心）：对任务 B，等距 q = 2 的互信息低于用 Y，而准确率与用 Y 相同；
      同一装置在两个指标下给出相反结论。
  L4d 对齐切法：把 Y = 3 单独成档的切法 S*，对任务 B 的互信息与准确率都无损。

模型：X 为 24 个独立均匀比特，分 4 块每块 6 位；Y_k 为块内和；Z 为 Y 的分档读数。
任务 A：逐位恢复 X 的每一位。任务 B：恢复块内多数方向（Y 与 3 的比较；平局时按 0.5 计）。
理论期望：acc_A(Y) = 0.656250，acc_A(q=3) = 0.578125；acc_B(Y) = 0.843750，acc_B(q=3) = 0.609375；
I_B(Y) = 2.750000 比特，I_B(q=2) = 1.921372 比特而 acc_B(q=2) = 0.843750。
"""
from math import comb
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

IMG = "D:/software/TianXingFC/images/coarse_graining_alignment.png"
SEED = 20260925
NB = 6
KB = 4
NBIT = NB * KB
MID = NB // 2
M_SAMPLE = 400000

p_s = np.array([comb(NB, s) for s in range(NB + 1)], dtype=float) / 2.0 ** NB


def entropy(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def h2(p):
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p))


def equi_bounds(q):
    edges = np.linspace(0.0, NB + 1.0, q + 1)
    return [int(round(e)) for e in edges]


def taskA_acc(edges):
    acc = 0.0
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        ps = p_s[lo:hi]
        ss = np.arange(lo, hi, dtype=float)
        w = ps.sum()
        if w <= 0:
            continue
        p1 = float((ps * ss).sum() / w) / NB
        acc += w * max(p1, 1.0 - p1)
    return float(acc)


def taskA_mi(edges):
    dist = [p_s[edges[i]:edges[i + 1]].sum() for i in range(len(edges) - 1)]
    return float(KB * entropy(dist))


def block_p0(lo, hi):
    tot = 0.0
    for s in range(lo, hi):
        if s < MID:
            tot += p_s[s]
        elif s == MID:
            tot += 0.5 * p_s[s]
    return tot


def taskB_acc(edges):
    acc = 0.0
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        w = p_s[lo:hi].sum()
        if w <= 0:
            continue
        p0 = block_p0(lo, hi) / w
        acc += w * max(p0, 1.0 - p0)
    return float(acc)


def taskB_mi(edges):
    h_cond = 0.0
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        w = p_s[lo:hi].sum()
        if w <= 0:
            continue
        h_cond += w * h2(block_p0(lo, hi) / w)
    return float(KB * (1.0 - h_cond))


print("=== 第 402 条：切法与任务对齐 ===")
print("模型：%d 个独立均匀比特，%d 块每块 %d 位；Y 为块内和，Z 为 Y 的分档读数" % (NBIT, KB, NB))
print("任务 A = 逐位恢复 X；任务 B = 恢复块内多数方向（Y 与 %d 的比较）" % MID)
print("")

FULL = equi_bounds(NB + 1)
STAR = [0, MID, MID + 1, NB + 1]

print("--- 基准（读全 Y，不粗粒化） ---")
print("acc_A(Y) = %.6f    acc_B(Y) = %.6f" % (taskA_acc(FULL), taskB_acc(FULL)))
print("I_A(Y)   = %.6f 比特   I_B(Y) = %.6f 比特" % (taskA_mi(FULL), taskB_mi(FULL)))
print("")

QS = list(range(2, NB + 2))
rows = []
for q in QS:
    e = equi_bounds(q)
    rows.append((q, taskA_acc(e), taskA_mi(e), taskB_acc(e), taskB_mi(e)))
e2 = equi_bounds(2)
e3 = equi_bounds(3)

print("--- 等距切法扫描 ---")
print("q    acc_A     I_A       acc_B     I_B")
for q, aa, ia, ab, ib in rows:
    print("%d    %.6f  %.6f  %.6f  %.6f" % (q, aa, ia, ab, ib))
print("")

print("--- L4a 对齐无损（准确率口径）---")
print("acc_A(q=2) = %.6f  基准 %.6f  判定 %s" % (rows[0][1], taskA_acc(FULL), rows[0][1] >= taskA_acc(FULL) - 1e-12))
print("acc_B(q=2) = %.6f  基准 %.6f  判定 %s" % (rows[0][3], taskB_acc(FULL), rows[0][3] >= taskB_acc(FULL) - 1e-12))
print("")
print("--- L4b 错位有损（准确率口径）---")
print("acc_A(q=3) = %.6f  基准 %.6f  下降量 %.6f" % (rows[1][1], taskA_acc(FULL), taskA_acc(FULL) - rows[1][1]))
print("acc_B(q=3) = %.6f  基准 %.6f  下降量 %.6f" % (rows[1][3], taskB_acc(FULL), taskB_acc(FULL) - rows[1][3]))
print("")
print("--- L4c 指标依赖（核心）---")
print("任务 B 在 q = 2：I_B = %.6f 比特（基准 %.6f，有损）" % (rows[0][4], taskB_mi(FULL)))
print("                  acc_B = %.6f（基准 %.6f，无损）" % (rows[0][3], taskB_acc(FULL)))
print("两指标结论相反                     : %s" % (rows[0][4] < taskB_mi(FULL) - 1e-12 and rows[0][3] >= taskB_acc(FULL) - 1e-12))
print("")
print("--- L4d 对齐切法 S*（把 Y = %d 单独成档）---" % MID)
print("acc_B(S*) = %.6f  基准 %.6f  %s" % (taskB_acc(STAR), taskB_acc(FULL), taskB_acc(STAR) >= taskB_acc(FULL) - 1e-12))
print("I_B(S*)   = %.6f  基准 %.6f  %s" % (taskB_mi(STAR), taskB_mi(FULL), abs(taskB_mi(STAR) - taskB_mi(FULL)) < 1e-12))
print("I_A(S*)   = %.6f  基准 %.6f" % (taskA_mi(STAR), taskA_mi(FULL)))
print("")

rng = np.random.default_rng(SEED)
Xm = rng.integers(0, 2, size=(M_SAMPLE, KB, NB))
Ym = Xm.sum(axis=2)
tie = (Ym == MID)
truth = (Ym > MID).astype(int)
guessY = truth
hitY = np.where(tie, 0.5, (guessY == truth).astype(float))
acc_B_Y_mc = float(hitY.mean())
Z2 = np.digitize(Ym, e2[1:-1])
guessZ2 = (Z2 == 1).astype(int)
hitZ2 = np.where(tie, 0.5, (guessZ2 == truth).astype(float))
acc_B_Z2_mc = float(hitZ2.mean())
Z3 = np.digitize(Ym, e3[1:-1])
guessZ3 = (Z3 == 2).astype(int)
hitZ3 = np.where(tie, 0.5, (guessZ3 == truth).astype(float))
acc_B_Z3_mc = float(hitZ3.mean())
print("--- 蒙特卡洛交叉验证（M = %d，种子 %d）---" % (M_SAMPLE, SEED))
print("acc_B 用 Y    : 采样 %.6f  解析 %.6f" % (acc_B_Y_mc, taskB_acc(FULL)))
print("acc_B 用 Z(q=2): 采样 %.6f  解析 %.6f" % (acc_B_Z2_mc, taskB_acc(e2)))
print("acc_B 用 Z(q=3): 采样 %.6f  解析 %.6f" % (acc_B_Z3_mc, taskB_acc(e3)))
print("")

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
ax = axes[0]
ax.plot(QS, [r[4] for r in rows], "o-", color="tab:blue", label="I_B（比特）")
ax.set_xlabel("等距档数 q")
ax.set_ylabel("I_B（比特）", color="tab:blue")
ax2 = ax.twinx()
ax2.plot(QS, [r[3] for r in rows], "s--", color="tab:red", label="acc_B")
ax2.set_ylabel("acc_B", color="tab:red")
ax.set_title("任务 B：信息与准确率的分歧")
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(QS, [r[1] for r in rows], "o-", label="任务 A（逐位）")
ax.plot(QS, [r[3] for r in rows], "s-", label="任务 B（多数方向）")
ax.set_xlabel("等距档数 q")
ax.set_ylabel("最优准确率")
ax.set_title("两任务的准确率随切法变化")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[2]
labels = ["q=2", "q=3", "S*"]
accs = [taskB_acc(e2), taskB_acc(e3), taskB_acc(STAR)]
mis = [taskB_mi(e2), taskB_mi(e3), taskB_mi(STAR)]
xpos = np.arange(3)
ax.bar(xpos - 0.2, [v / max(mis) for v in mis], width=0.4, label="I_B（归一）", color="tab:blue")
ax.bar(xpos + 0.2, accs, width=0.4, label="acc_B", color="tab:red")
ax.axhline(taskB_acc(FULL), color="gray", linestyle=":", label="acc_B 基准")
ax.set_xticks(xpos)
ax.set_xticklabels(labels)
ax.set_ylim(0.0, 1.15)
ax.set_title("三种切法在任务 B 上的对比")
ax.legend(fontsize=7)
ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(IMG, dpi=150)
print("图已保存：" + IMG)
