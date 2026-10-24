# -*- coding: utf-8 -*-
"""
判据 L4 的独立复现：对齐判据形式在无平局模型上的穷举检验（验证脚本第 403 条）

背景：第 402 条在 6 位块（有平局）模型上得到读数，装置无损当且仅当档边界包含任务的判别边界。
本条换一个结构不同的模型，以穷举代替扫描，检验该等价性，并修正该判据形式的表述。

模型（与 402 的两处结构差异）：
  1. 块长由 6 位改为 8 位。偶数块长使中点落在某个 Y 值上（此处 y = 4 归低侧），任务无平局歧义。
  2. 目标量由二值改为三值语义分档 L = {0,1,2}、M = {3,4,5}、H = {6,7,8}，判别边界有两处（2|3 与 5|6）。

判据（先立后算，2026-09-25；表述经本轮修正）：
  L4e 充分性：凡不同语义类不共档的切法，准确率为 1 且条件熵为 0。
  L4f 必要性：凡存在两个不同语义类共档的切法，准确率小于 1 或条件熵大于 0。
  L4g 等价性：对全部切法，「不同语义类不共档」与「无损」逐一对应，反例数为 0。
  L4h 指标一致：准确率与互信息的判定完全相同（本条模型无平局，故两指标不分歧）。

判据表述的修正（本轮过程记录）：初版把对齐条件写作「相邻判别边界各自分割」，即只查 2|3 与 5|6。
该条件过宽，漏掉非相邻的跨语义共档（例如 Y = 0 与 Y = 3 同档），使充分性方向出现大量假反例。
正确条件是「不同语义类不共档」，本稿同时报告两种定义下的反例数以便追溯。

穷举范围：Y 的全部 9 个取值划分，共 Bell(9) = 21147 种。二值对照任务 T2 = [y > 4] 同样穷举。
理论期望：任务 T3 的无损切法应等于 Bell(3)^3 = 125 种；任务 T2 的应等于 Bell(5) * Bell(4) = 52 * 15 = 780 种。
"""
from math import comb, log2
from collections import defaultdict

NB = 8
NV = NB + 1
KB = 3
PARTITIONS = 21147

p_s = [comb(NB, s) / 2.0 ** NB for s in range(NV)]
SEM3 = [0 if s <= 2 else (1 if s <= 5 else 2) for s in range(NV)]
T2 = [1 if s > 4 else 0 for s in range(NV)]


def entropy(p):
    out = 0.0
    for x in p:
        if x > 0.0:
            out -= x * log2(x)
    return out


def all_partitions(n):
    labels = [0] * n

    def rec(i, mx):
        if i == n:
            yield tuple(labels)
            return
        for lab in range(mx + 2):
            labels[i] = lab
            yield from rec(i + 1, max(mx, lab))

    yield from rec(0, -1)


def aligned(labels, target):
    for a in range(NV):
        for b in range(a + 1, NV):
            if target[a] != target[b] and labels[a] == labels[b]:
                return False
    return True


def aligned_narrow(labels, target):
    for s in range(NV - 1):
        if target[s] != target[s + 1] and labels[s] == labels[s + 1]:
            return False
    return True


def evaluate(labels, target, ncat):
    groups = defaultdict(list)
    for s in range(NV):
        groups[labels[s]].append(s)
    acc = 0.0
    h_cond = 0.0
    for ss in groups.values():
        w = sum(p_s[s] for s in ss)
        prob = [0.0] * ncat
        for s in ss:
            prob[target[s]] += p_s[s]
        prob = [x / w for x in prob]
        acc += w * max(prob)
        h_cond += w * entropy(prob)
    return acc, h_cond


H3 = entropy([sum(p_s[s] for s in range(NV) if SEM3[s] == c) for c in range(3)])
H2 = entropy([sum(p_s[s] for s in range(NV) if T2[s] == c) for c in range(2)])

print("=== 第 403 条：对齐判据形式的独立复现（穷举）===")
print("模型：块长 %d 位（无平局），Y 在 0 至 %d，独立块数 %d" % (NB, NB, KB))
print("任务 T3 为三值语义分档 L/M/H（边界 2|3 与 5|6）；对照任务 T2 = [y > 4]（边界 4|5）")
print("穷举范围：Bell(%d) = %d 种切法" % (NV, PARTITIONS))
print("基准熵：H(T3) = %.6f 比特，H(T2) = %.6f 比特" % (H3, H2))
print("")

data = {"T3": [], "T2": []}
narrow_bad = {"T3": 0, "T2": 0}
for labels in all_partitions(NV):
    for key, target, ncat in [("T3", SEM3, 3), ("T2", T2, 2)]:
        acc, hc = evaluate(labels, target, ncat)
        ok = abs(acc - 1.0) < 1e-12 and hc < 1e-12
        sp = aligned(labels, target)
        data[key].append((sp, ok, hc, acc))
        if aligned_narrow(labels, target) and not sp:
            narrow_bad[key] += 1

print("--- L4g 等价性交叉表（行 = 不同语义类不共档，列 = 无损）---")
for key, name in [("T3", "任务 T3"), ("T2", "任务 T2")]:
    rs = data[key]
    a = sum(1 for sp, ok, hc, acc in rs if sp and ok)
    b = sum(1 for sp, ok, hc, acc in rs if sp and not ok)
    c = sum(1 for sp, ok, hc, acc in rs if (not sp) and ok)
    d = sum(1 for sp, ok, hc, acc in rs if (not sp) and not ok)
    print("%s：不共档且无损 %d ｜ 不共档却有损 %d ｜ 共档却无损 %d ｜ 共档且有损 %d" % (name, a, b, c, d))
    print("        反例数（b + c） = %d，期望 0" % (b + c))
    print("        窄定义（只查相邻边界）漏掉的共档情形 = %d 种" % narrow_bad[key])
print("")

print("--- L4e 充分性 / L4f 必要性 ---")
for key, name in [("T3", "任务 T3"), ("T2", "任务 T2")]:
    rs = data[key]
    print("%s：不共档组全部无损 %s ｜ 共档组全部有损 %s" % (
        name,
        all(ok for sp, ok, hc, acc in rs if sp),
        all(not ok for sp, ok, hc, acc in rs if not sp)))
print("")

print("--- L4h 指标一致性 ---")
for key, name in [("T3", "任务 T3"), ("T2", "任务 T2")]:
    rs = data[key]
    print("%s：准确率判定与条件熵判定完全相同 %s" % (
        name, all((abs(acc - 1.0) < 1e-12) == (hc < 1e-12) for sp, ok, hc, acc in rs)))
print("")

n3 = sum(1 for sp, ok, hc, acc in data["T3"] if ok)
n2 = sum(1 for sp, ok, hc, acc in data["T2"] if ok)
print("无损切法计数：任务 T3 = %d，理论 Bell(3)^3 = 125，判定 %s" % (n3, n3 == 125))
print("无损切法计数：任务 T2 = %d，理论 Bell(5)*Bell(4) = 780，判定 %s" % (n2, n2 == 780))
print("")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
IMG = "D:/software/TianXingFC/images/alignment_exhaustive.png"

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
ax = axes[0]
xs = [t[2] for t in data["T3"]]
ys = [t[3] for t in data["T3"]]
cs = ["tab:blue" if t[0] else "tab:red" for t in data["T3"]]
ax.scatter(xs, ys, s=3, c=cs, alpha=0.35)
ax.set_xlabel("条件熵 H(T3 与档)（比特）")
ax.set_ylabel("最优准确率")
ax.set_title("穷举 %d 种切法（蓝 = 不同语义类不共档）" % PARTITIONS)
ax.grid(alpha=0.3)

ax = axes[1]
cats = ["不共档且无损", "不共档却有损", "共档却无损", "共档且有损"]
w = 0.35
for idx, key in enumerate(["T3", "T2"]):
    rs = data[key]
    vals = [
        sum(1 for sp, ok, hc, acc in rs if sp and ok),
        sum(1 for sp, ok, hc, acc in rs if sp and not ok),
        sum(1 for sp, ok, hc, acc in rs if (not sp) and ok),
        sum(1 for sp, ok, hc, acc in rs if (not sp) and not ok),
    ]
    ax.bar([v + idx * w for v in range(4)], [max(v, 0.5) for v in vals], width=w, label=key)
ax.set_yscale("log")
ax.set_xticks(range(4))
ax.set_xticklabels(cats, fontsize=7, rotation=12)
ax.set_ylabel("切法计数（对数轴）")
ax.set_title("L4g 交叉表（中间两类应为零）")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")

ax = axes[2]
labels_x = ["窄定义\n（只查相邻边界）", "正确定义\n（跨语义不共档）"]
vals3 = [narrow_bad["T3"], 0]
vals2 = [narrow_bad["T2"], 0]
xpos = [0, 1]
ax.bar([v - 0.17 for v in xpos], [max(v, 0.5) for v in vals3], width=0.34, label="任务 T3")
ax.bar([v + 0.17 for v in xpos], [max(v, 0.5) for v in vals2], width=0.34, label="任务 T2")
ax.set_yscale("log")
ax.set_xticks(xpos)
ax.set_xticklabels(labels_x, fontsize=8)
ax.set_ylabel("假反例数（对数轴）")
ax.set_title("判据表述修正的效果")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(IMG, dpi=150)
print("图已保存：" + IMG)
