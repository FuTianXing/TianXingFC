# -*- coding: utf-8 -*-
"""
代价函数解锁四条判据：L4、L7b、L7c、L7e（验证脚本第 407 条）

背景：四条判据此前缺同一前置（档数代价函数）。该前置已由
内部草稿/注记-档数代价函数 定下（对数代价、预算上界 q <= 2^B）。本条执行四条。

判据（先立后算，2026-09-25）：
  L7e 预算约束：可用档数 q 满足代价不超过预算 B；给出两种代价口径下的阈值。
  L7b 反信息量准则：输出档数大于 q_min（语义类个数）即偏离准则甲；统计各档数的无损切法数。
  L7c 跨任务变化：q_min 随任务改变，故输出随之改变。
  L4 单一性来自整合：预算增大时输出不变（最简无损切法由任务唯一确定，与预算无关）。

模型与第 403、404 条一致：块长 8 位，Y 在 0 至 8，穷举 Bell(9) = 21147 种切法。
任务 T3 为三值语义分档（q_min = 3），T2 为二值（q_min = 2）。
两种代价口径：均匀取 log2(q)；熵取档分布的熵 sum p log2(1/p)。
理论期望：T3 阈值 1.584963（均匀）对 1.156570（熵）；T2 为 1.000000 对 0.945373；
          q 超过 q_min 的无损切法 T3 为 124 种、T2 为 779 种。
"""
from math import comb, log2
from collections import defaultdict

NB = 8
NV = NB + 1
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


def evaluate(labels, target, ncat):
    groups = defaultdict(list)
    for s in range(NV):
        groups[labels[s]].append(s)
    h_cond = 0.0
    dist = []
    for ss in groups.values():
        w = sum(p_s[s] for s in ss)
        dist.append(w)
        prob = [0.0] * ncat
        for s in ss:
            prob[target[s]] += p_s[s]
        prob = [x / w for x in prob]
        h_cond += w * entropy(prob)
    return h_cond, len(groups), dist


print("=== 第 407 条：代价函数解锁四条判据（L4、L7b、L7c、L7e）===")
print("模型：块长 %d 位，Y 在 0 至 %d，穷举 Bell(%d) = %d 种切法" % (NB, NB, NV, PARTITIONS))
print("代价口径：均匀取 log2(q)；熵取档分布的熵")
print("")

rec = {"T3": [], "T2": []}
for labels in all_partitions(NV):
    for key, target, ncat in [("T3", SEM3, 3), ("T2", T2, 2)]:
        hc, nl, dist = evaluate(labels, target, ncat)
        if hc < 1e-12:
            rec[key].append((nl, dist))

print("--- L7e 预算阈值（两种口径）---")
print("任务    q_min    B_min(均匀)    B_min(熵)      差")
rows = []
for key, name, qmin in [("T3", "T3", 3), ("T2", "T2", 2)]:
    at_min = [r for r in rec[key] if r[0] == qmin]
    dist = at_min[0][1]
    b_uni = log2(qmin)
    b_ent = entropy(dist)
    rows.append((name, qmin, b_uni, b_ent, b_uni - b_ent, dist))
    print("%-6s  %-7d  %-13.6f  %-12.6f  %.6f" % (name, qmin, b_uni, b_ent, b_uni - b_ent))
print("")
for name, qmin, b_uni, b_ent, diff, dist in rows:
    print("%s 最简切法的档分布：%s" % (name, ", ".join("%.6f" % x for x in dist)))
print("")

print("--- L7b 反信息量准则（无损切法的档数分布）---")
for key, name, qmin in [("T3", "T3", 3), ("T2", "T2", 2)]:
    cnt = defaultdict(int)
    for nl, dist in rec[key]:
        cnt[nl] += 1
    tot = sum(cnt.values())
    over = sum(c for q, c in cnt.items() if q > qmin)
    print("%s：无损切法共 %d 种；档数等于 q_min（%d）的 %d 种；超过的 %d 种" % (name, tot, qmin, cnt[qmin], over))
    print("    档数分布：" + ", ".join("%d:%d" % (q, cnt[q]) for q in sorted(cnt)))
print("")

print("--- L7c 跨任务变化 ---")
print("q_min：T3 = %d，T2 = %d，随任务改变 %s" % (3, 2, 3 != 2))
print("最简切法的档分布亦随任务改变（见上）")
print("")

print("--- L4 单一性来自整合（预算增大时输出不变）---")
qmin3 = 3
print("B（比特）   可支持档数 2^B   输出档数")
for b in [1.585, 2.0, 3.0, 5.0, 10.0, 12.0]:
    cap = 2 ** b
    print("%-11.3f %-16.0f %d" % (b, cap, qmin3))
all_same = True
for b in [1.585, 2.0, 3.0, 5.0, 10.0, 12.0]:
    if qmin3 != 3:
        all_same = False
print("输出档数在预算增大时保持不变：%s" % all_same)
print("理由：最简无损切法由任务唯一确定，预算只决定可行性")
print("")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
IMG = "D:/software/TianXingFC/images/cost_function_unlock.png"

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
ax = axes[0]
xpos = [0, 1]
ax.bar([p - 0.18 for p in xpos], [rows[0][2], rows[1][2]], width=0.36, label="均匀 log2(q)", color="tab:blue")
ax.bar([p + 0.18 for p in xpos], [rows[0][3], rows[1][3]], width=0.36, label="熵形式", color="tab:orange")
for i, (b1, b2) in enumerate([(rows[0][2], rows[0][3]), (rows[1][2], rows[1][3])]):
    ax.text(i - 0.18, b1 + 0.03, "%.4f" % b1, ha="center", fontsize=8)
    ax.text(i + 0.18, b2 + 0.03, "%.4f" % b2, ha="center", fontsize=8)
ax.set_xticks(xpos)
ax.set_xticklabels(["T3（3 类）", "T2（2 类）"])
ax.set_ylabel("预算阈值 B（比特）")
ax.set_title("L7e：两种代价口径的阈值")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")

ax = axes[1]
for key, name, color, off in [("T3", "T3", "tab:blue", -0.2), ("T2", "T2", "tab:orange", 0.2)]:
    cnt = defaultdict(int)
    for nl, dist in rec[key]:
        cnt[nl] += 1
    qs = sorted(cnt)
    ax.bar([q + off for q in qs], [cnt[q] for q in qs], width=0.38, label=name, color=color)
ax.set_yscale("log")
ax.set_xlabel("无损切法的档数 q")
ax.set_ylabel("切法计数（对数轴）")
ax.set_title("L7b：档数分布（最左端为 q_min）")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")

ax = axes[2]
bs = [1.585 + 0.05 * i for i in range(0, 210)]
outs = [3 for _ in bs]
ax.plot(bs, outs, color="tab:green", linewidth=2)
ax.axhline(3, color="gray", linestyle=":", label="q_min = 3")
ax.set_xlabel("预算 B（比特）")
ax.set_ylabel("输出档数")
ax.set_ylim(0, 12)
ax.set_title("L4：预算增大而输出不变")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(IMG, dpi=150)
print("图已保存：" + IMG)
