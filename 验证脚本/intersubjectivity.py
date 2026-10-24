# -*- coding: utf-8 -*-
"""
判据 L6 的最小验证：主体间性（验证脚本第 405 条）

背景：作者就主体间性给出补充机制。除判据草案原有的「公开事件」路径外，另一条路径是统计集中：
位置波包的高概率集中在极小区段，各主体独立抽取也会高度趋同，残余差异相当于噪音。
本条把两条路径落成可算判据并给出定量分界。

判据（先立后算，2026-09-25；阈值口径经本轮修正）：
  L6a 公开事件路径：抽取为公开事件、落点写入共享记录时，两主体的读数差异恒为零（定义性）。
  L6b 统计集中路径：两主体独立抽取时，分歧概率由 sigma/delta 决定，分条件值与平均值两口径。
  L6c 差异的可发现性：无共享记录时差异不可发现（等于噪音）；有共享记录时差异可测，须 L6b 满足。
  L6d 分界阈值：使**平均**分歧概率小于 1e-6 的 sigma/delta 上界。
  L6e 边界对准的不可消除性：波包中心落在分辨单元边界时，分歧不随波包收窄而消失，恒为 0.5。

模型：位置读数 x ~ N(mu, sigma^2)，分辨单元宽 delta（取 delta = 1），两主体独立抽取。
条件分歧按 mu 取格中心（最优）与格边界（最差）；平均分歧按 mu 在格内均匀分布积分。
理论期望：平均分歧约等于 1.1 倍 sigma/delta；阈值约 9e-7；sigma/delta = 0.1 时平均分歧约 0.11。
"""
from math import erfc, exp, log, log10, pi, sqrt

SQ2 = sqrt(2.0)
SQ2PI = sqrt(2.0 * pi)


def tail(u):
    if u <= 0.0:
        return 1.0 - 0.5 * erfc(-u / SQ2)
    if u < 25.0:
        return 0.5 * erfc(u / SQ2)
    return exp(-0.5 * u * u) / SQ2PI / u * (1.0 - 1.0 / (u * u))


def same_cell_exact(s, offset):
    kmax = min(int(12.0 / s) + 6, 4000)
    total = 0.0
    for k in range(-kmax, kmax + 1):
        a = (k + 0.5 - offset) / s
        b = (k - 0.5 - offset) / s
        q = tail(b) - tail(a)
        if q > 0.0:
            total += q * q
    return total


def disagree_two_cell(s, offset):
    off = offset if offset <= 0.5 else 1.0 - offset
    p0 = 1.0 - tail((0.5 - off) / s)
    return 2.0 * p0 * (1.0 - p0)


def disagree_cond(s, offset):
    if s < 0.05:
        return disagree_two_cell(s, offset)
    return 1.0 - same_cell_exact(s, offset)


def disagree_mean_small(s, m=4000):
    # 两格解析：平均分歧 = 4 s * int_0^{0.5/s} Phi(1-Phi) du
    U = min(0.5 / s, 12.0)
    h = U / m
    tot = 0.0
    for i in range(m + 1):
        u = i * h
        w = 1.0 if 0 < i < m else 0.5
        p = 1.0 - tail(u)
        tot += w * p * (1.0 - p)
    return 4.0 * s * tot * h


def disagree_mean(s, n=4001):
    if s < 0.1:
        return disagree_mean_small(s)
    tot = 0.0
    for i in range(n):
        tot += disagree_cond(s, (i + 0.5) / n)
    return tot / n


print("=== 第 405 条：主体间性 ===")
print("模型：x ~ N(mu, sigma^2)，分辨单元宽 delta = 1，两主体独立抽取")
print("条件分歧按 mu 对准格中心（最优）与格边界（最差）；平均分歧按 mu 在格内均匀")
print("")

print("--- L6a 公开事件路径 ---")
print("落点写入共享记录时两主体读数差异 = 0（定义性，不作数值检验）")
print("")

SCAN = [1e-6, 1e-4, 1e-3, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
print("--- L6b 统计集中路径：分歧概率（三个口径）---")
print("sigma/delta   分歧(格中心)   分歧(格边界)   分歧(平均)")
rows = []
for s in SCAN:
    dc = disagree_cond(s, 0.0)
    db = disagree_cond(s, 0.5)
    dm = disagree_mean(s)
    rows.append((s, dc, db, dm))
    print("%-13.3g %.6e    %.6f      %.6e" % (s, dc, db, dm))
print("")

print("--- 平均分歧与 sigma/delta 的比例（小 s 段）---")
for s in [1e-8, 1e-6, 1e-4, 0.01, 0.05, 0.1, 0.2]:
    dm = disagree_mean(s)
    print("sigma/delta = %-6.3g  平均分歧 = %.6e  比值 = %.4f" % (s, dm, dm / s))
print("")

print("--- L6d 分界阈值（平均分歧 < 1e-6）---")
lo, hi = 1e-8, 0.2
for _ in range(80):
    mid = 0.5 * (lo + hi)
    if disagree_mean(mid, 2001) < 1e-6:
        lo = mid
    else:
        hi = mid
print("数值阈值 sigma/delta = %.6e，期望约 9e-07" % lo)
print("对照：条件阈值（波包中心恰在格中心，分歧 < 1e-6）为 0.0995，两者相差五个数量级")
print("")

print("--- L6e 边界对准的不可消除性 ---")
print("sigma/delta 趋于 0 时格中心对准的分歧趋于 0，而格边界对准的分歧恒为 0.5")
print("sigma/delta = 1e-6 边界对准：分歧 = %.10f" % disagree_cond(1e-6, 0.5))
print("该情形同时支配平均分歧：平均分歧的量级由它贡献")
print("")

print("--- L6c 差异的可发现性（形式化，不作数值检验）---")
print("无共享记录：差异不可发现，等于噪音；有共享记录：差异可测，此时须 L6d 满足")
print("")

print("--- 宏观举例（平均分歧的量级）---")
for s in [1e-3, 1e-6, 1e-9, 1e-20]:
    dm = disagree_mean(s, 2001)
    lg = log10(dm) if dm > 0 else float("-inf")
    print("sigma/delta = %.0e  平均分歧 = %.3e   log10 = %.1f" % (s, dm, lg))
print("")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
IMG = "D:/software/TianXingFC/images/intersubjectivity.png"

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
xs = [r[0] for r in rows]
ax = axes[0]
ax.loglog(xs, [max(r[1], 1e-30) for r in rows], "o-", label="格中心对准（条件）")
ax.loglog(xs, [max(r[2], 1e-30) for r in rows], "s--", label="格边界对准（条件）")
ax.loglog(xs, [max(r[3], 1e-30) for r in rows], "^-", label="平均（mu 在格内均匀）")
ax.set_xlabel("sigma / delta")
ax.set_ylabel("分歧概率")
ax.set_title("L6b：三个口径的分歧概率")
ax.legend(fontsize=7)
ax.grid(alpha=0.3, which="both")

ax = axes[1]
ax.loglog(xs, [max(r[3], 1e-40) for r in rows], "^-", color="tab:green", label="平均分歧")
ax.axhline(1e-6, color="gray", linestyle=":", label="1e-6")
ax.axvline(lo, color="tab:red", linestyle=":", label="阈值 %.2e" % lo)
ax.set_xlabel("sigma / delta")
ax.set_ylabel("平均分歧概率")
ax.set_title("L6d：平均分歧的阈值（约 9e-7）")
ax.legend(fontsize=7)
ax.grid(alpha=0.3, which="both")

ax = axes[2]
ss = [1e-3, 1e-6, 1e-9, 1e-20]
lgs = [log10(disagree_mean(s, 2001)) for s in ss]
ax.bar(range(len(ss)), [-v for v in lgs], color="tab:blue")
for i, v in enumerate(lgs):
    ax.text(i, -v + 1.0, "1e%.0f" % v, ha="center", fontsize=8)
ax.set_xticks(range(len(ss)))
ax.set_xticklabels(["1e-3", "1e-6", "1e-9", "1e-20"])
ax.set_xlabel("sigma / delta")
ax.set_ylabel("负的 log10(平均分歧)")
ax.set_title("宏观举例：平均分歧的量级")
ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(IMG, dpi=150)
print("图已保存：" + IMG)
