# -*- coding: utf-8 -*-
"""
判据 L7a 与 L7d 的最小验证：整合的目标函数（验证脚本第 404 条）

背景：判据草案「研究方向-整合的目标函数」把「最简洁最有用」中的「有用」立成判据组 L7。
准则甲（任务对齐）取最简无损切法，准则乙（信息量最大）取最细划分。本条执行 L7a 与 L7d。

判据（先立后算，2026-09-25）：
  L7a 档数最小性：对指定任务，最简无损切法的档数等于该任务的语义类个数。
  L7a+ 唯一性：达到最小档数的无损切法（不计档标号）唯一。
  L7d 与 L4 相容：全部无损切法都满足「不同语义类不共档」。
  L7c 预备读数：最简档数随任务改变。
  对照（非判据）：准则乙的档数与对 X 的信息量，用以量化「简洁」的代价。

模型与第 403 条一致：块长 8 位，Y 在 0 至 8，穷举 Bell(9) = 21147 种切法。
任务 T3 为三值语义分档（3 类），T2 为二值（2 类）。
理论期望：最简无损档数 T3 = 3、T2 = 2；达到最小档数的无损切法各 1 种；
          准则乙档数 9、I(Z;X) = H(Y) = 2.544198 比特；准则甲（T3 最简）I(Z;X) = 1.156570 比特。
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
    z_dist = []
    for ss in groups.values():
        w = sum(p_s[s] for s in ss)
        z_dist.append(w)
        prob = [0.0] * ncat
        for s in ss:
            prob[target[s]] += p_s[s]
        prob = [x / w for x in prob]
        h_cond += w * entropy(prob)
    return h_cond, len(groups), entropy(z_dist)


H_T3 = entropy([sum(p_s[s] for s in range(NV) if SEM3[s] == c) for c in range(3)])
H_T2 = entropy([sum(p_s[s] for s in range(NV) if T2[s] == c) for c in range(2)])
H_Y = entropy(p_s)

print("=== 第 404 条：整合的目标函数（判据 L7a 与 L7d）===")
print("模型：块长 %d 位，Y 在 0 至 %d，穷举 Bell(%d) = %d 种切法" % (NB, NB, NV, PARTITIONS))
print("任务 T3 为三值语义分档（3 类），T2 为二值（2 类）")
print("基准熵：H(T3) = %.6f，H(T2) = %.6f，H(Y) = %.6f 比特" % (H_T3, H_T2, H_Y))
print("")

rec = {"T3": [], "T2": []}
for labels in all_partitions(NV):
    for key, target, ncat in [("T3", SEM3, 3), ("T2", T2, 2)]:
        hc, nl, hz = evaluate(labels, target, ncat)
        rec[key].append((nl, hc < 1e-12, aligned(labels, target), hz))

print("--- L7d 与 L4 相容 ---")
for key, name in [("T3", "任务 T3"), ("T2", "任务 T2")]:
    rs = [r for r in rec[key] if r[1]]
    print("%s：无损切法 %d 种，全部满足不同语义类不共档 %s" % (name, len(rs), all(r[2] for r in rs)))
print("")

print("--- L7a 档数最小性 ---")
mins = {}
for key, name, want in [("T3", "任务 T3", 3), ("T2", "任务 T2", 2)]:
    rs = [r for r in rec[key] if r[1]]
    mn = min(r[0] for r in rs)
    cnt = sum(1 for r in rs if r[0] == mn)
    mins[key] = mn
    print("%s：最简无损档数 = %d，期望 %d，判定 %s" % (name, mn, want, mn == want))
    print("        达到最小档数的无损切法计 %d 种（不计档标号），期望 1，判定 %s" % (cnt, cnt == 1))
print("")

print("--- L7c 预备读数（跨任务）---")
print("最简档数 T3 = %d，T2 = %d，随任务改变 %s" % (mins["T3"], mins["T2"], mins["T3"] != mins["T2"]))
print("")

print("--- 对照：准则乙（信息量最大 = 最细划分，每档单值）---")
print("准则乙档数 = %d" % NV)
print("准则乙对 X 的信息量 I(Z;X) = H(Y) = %.6f 比特" % H_Y)
print("准则甲（T3 最简）档数 = %d" % mins["T3"])
print("准则甲对 X 的信息量 I(Z;X) = %.6f 比特" % H_T3)
print("准则甲对任务 T3 的损失 = %.6f 比特（0 即无损）" % (H_T3 - H_T3))
print("简洁的代价 = 放弃对 X 的 %.6f 比特，而任务性能不损失" % (H_Y - H_T3))
print("")

for key, name in [("T3", "任务 T3"), ("T2", "任务 T2")]:
    rs = [r for r in rec[key] if r[1]]
    nl_min = min(r[0] for r in rs)
    at_min = [r for r in rs if r[0] == nl_min]
    print("%s 最简无损切法的档数-信息量坐标：(档数 %d, I(Z;X) = %.6f 比特)" % (name, nl_min, at_min[0][3]))
print("")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
IMG = "D:/software/TianXingFC/images/integration_target.png"

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
ax = axes[0]
for key, name, color in [("T3", "任务 T3", "tab:blue"), ("T2", "任务 T2", "tab:orange")]:
    rs = [r for r in rec[key] if r[1]]
    levels = sorted(set(r[0] for r in rs))
    counts = [sum(1 for r in rs if r[0] == lv) for lv in levels]
    ax.bar([lv + (0.0 if key == "T3" else 0.35) for lv in levels], [max(c, 0.5) for c in counts],
           width=0.35, label=name, color=color)
ax.set_yscale("log")
ax.set_xlabel("无损切法的档数")
ax.set_ylabel("切法计数（对数轴）")
ax.set_title("无损切法的档数分布（左端即最简）")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")

ax = axes[1]
ax.bar(["准则甲\n（T3 最简）", "准则乙\n（最细划分）"], [mins["T3"], NV], color=["tab:blue", "tab:gray"])
for i, v in enumerate([mins["T3"], NV]):
    ax.text(i, v + 0.1, str(v), ha="center", fontsize=10)
ax.set_ylabel("档数")
ax.set_title("两条准则的档数")
ax.grid(alpha=0.3, axis="y")

ax = axes[2]
ax.bar(["准则甲", "准则乙"], [H_T3, H_Y], color=["tab:blue", "tab:gray"])
for i, v in enumerate([H_T3, H_Y]):
    ax.text(i, v + 0.03, "%.4f" % v, ha="center", fontsize=10)
ax.set_ylabel("I(Z;X)（比特）")
ax.set_title("两条准则对 X 的信息量")
ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(IMG, dpi=150)
print("图已保存：" + IMG)
