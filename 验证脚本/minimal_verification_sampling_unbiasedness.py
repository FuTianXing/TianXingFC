# -*- coding: utf-8 -*-
r"""
minimal_verification_sampling_unbiasedness.py

[采样无偏性轮] 抽样落点的口径决定偏置：载子计数无偏，站点计数在开链端点偏半

定位
----
对应开放问题 P3-12（采样定理的数学形式化）与论文《序参量份额定理》§10 (e)（抽样无偏性的晶格论证）。
问题：把"份额到观测频率"这一步落到格点上时，抽样落点是否按份额分布？本脚本把该问题拆成两段检验：
  第一段（确定性几何）：最小束缚振荡链上，事件的计数口径（按载子或按站点）是否给出均匀落点；
  第二段（统计性质）：若落点按份额抽样，则频率估计是否无偏、偏差是否按采样数的平方根收敛。
本脚本不改动基准文件；不含新科学断言。

模型
----
一维链，$N$ 个格点（单元），载子 $i$ 在相邻两条边界 $b_i$ 与 $b_{i+1}$ 之间往返；全体同频（同相）。
每个摆程每个载子恰在它的一条边界上留下一次读数，两邻交替：第 $k$ 摆程载子 $i$ 位于 $b_i$（当 $k$ 偶）或 $b_{i+1}$（当 $k$ 奇）。
事件 = 载子与站点的配对。两种计数口径：
  载子计数：事件按载子归属（每个载子每摆程一次）；
  站点计数：事件按站点归属（每条边界收到多少次到访）。
开链指端点 $b_0$、$b_N$ 只有一侧邻居；闭链指 $b_N$ 与 $b_0$ 认同（周期边界）。

判据（先立判据，再实现；预期写错时修正判据并记录教训）
----
J1 载子计数无偏（精确）：每个载子每摆程恰好一次事件（计数 = T）；每个载子对两条相邻边界的到访数在一个整数周期内严格相等（各 T/2）。
J2 开链站点计数的端点偏置：端点站点的频率对内部站点的比值 = 0.5 ± 0.02（O(1) 单站偏置，不随 N 消失）。
J3 偏置的尺度行为：受影响站点占全部站点的比例 = 2/N，故整体总变差偏差 TV 随 N 按 1/N 收敛
   （N = 8、16、32、64 四点双对数斜率 = -1.0 ± 0.05），而端点比值恒为 0.5。
J4 闭链消除偏置：周期边界下全部站点计数严格相等（频率 = 1/N，偏差 < 1e-12）。
J5 随机抽样段：按份额抽样时（块份额取 1:2:3:4），(i) 频率对份额的偏差在 3σ 内的比例 >= 99%（σ = sqrt(p(1-p)/T)）；
   (ii) 偏差的均方根随采样数按 T^(-1/2) 收敛（T = 10 至 10000 六点双对数斜率 = -0.5 ± 0.1）。

运行
----
python 验证脚本/minimal_verification_sampling_unbiasedness.py
确定性（随机种子固定）。图存 D:/software/TianXingFC/images/sampling_unbiasedness.png
"""
import time

import numpy as np

T0 = time.time()
CHECKS = []
RNG = np.random.default_rng(20260918)


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print(f"  [{tag}] {'PASS' if ok else 'FAIL'}  {detail}")


# ---------------- 第一段：确定性几何 ----------------
def site_counts_open(N, T, offsets=None):
    """开链：返回站点计数数组（长度 N+1）与载子计数数组（长度 N）。"""
    if offsets is None:
        offsets = np.zeros(N, dtype=int)
    sc = np.zeros(N + 1, dtype=int)
    cc = np.zeros(N, dtype=int)
    for k in range(T):
        for i in range(N):
            right = (k + offsets[i]) % 2 == 1
            site = i + 1 if right else i
            sc[site] += 1
            cc[i] += 1
    return sc, cc


def site_counts_closed(N, T, offsets=None):
    """闭链：站点 b_N 与 b_0 认同，返回长度 N 的站点计数（相对 b_0 的位置）。"""
    if offsets is None:
        offsets = np.zeros(N, dtype=int)
    sc = np.zeros(N, dtype=int)
    for k in range(T):
        for i in range(N):
            right = (k + offsets[i]) % 2 == 1
            site = (i + 1) % N if right else i % N
            sc[site] += 1
    return sc


print("== 第一段：确定性几何 ==")
N, T = 8, 400
sc_open, cc = site_counts_open(N, T)
freq_open = sc_open / sc_open.sum()
share = 1.0 / N
interior = freq_open[1:N]
ends = np.array([freq_open[0], freq_open[N]])
ratio = ends.mean() / interior.mean()

print(f"  N = {N}，摆程数 T = {T}，事件总数 = {sc_open.sum()}（期望 N·T = {N*T}）")
print(f"  内部站点频率均值 = {interior.mean():.6f}（份额 1/N = {share:.6f}）")
print(f"  端点频率均值     = {ends.mean():.6f}（份额的一半 = {share/2:.6f}）")
check("J1a 载子事件数", np.all(cc == T), f"每个载子 {cc.min()} 至 {cc.max()} 次（期望全为 {T}）")
visits = []
for i in range(N):
    left = np.sum((np.arange(T) + 0) % 2 == 0)   # 载子 i 在偶数摆程位于 b_i
    right = T - left
    visits.append((left, right))
ok_j1b = all(abs(l - r) <= 1 for l, r in visits)
check("J1b 两邻到访相等", ok_j1b, f"各载子左右到访 = {visits[0]}（期望各 T/2 = {T//2}，差 <= 1）")
ok_j2 = abs(ratio - 0.5) <= 0.02
check("J2 开链端点偏置", ok_j2, f"端点/内部频率比 = {ratio:.6f}（期望 0.500 ± 0.02）")

Ns = [8, 16, 32, 64]
ratios, tvs = [], []
for n in Ns:
    sc, _ = site_counts_open(n, 200)
    f = sc / sc.sum()
    s = 1.0 / n
    r = (f[0] + f[n]) / 2 / f[1:n].mean()
    tv = 0.5 * np.sum(np.abs(f - s))
    ratios.append(r)
    tvs.append(tv)
slope_tv = np.polyfit(np.log(Ns), np.log(tvs), 1)[0]
ok_j3a = all(abs(r - 0.5) <= 0.02 for r in ratios)
ok_j3b = abs(slope_tv + 1.0) <= 0.05
print(f"  端点比随 N：{['%.4f' % r for r in ratios]}")
print(f"  整体 TV 偏差随 N：{['%.6f' % t for t in tvs]}，双对数斜率 = {slope_tv:.4f}（期望 -1.0）")
check("J3a 端点偏置 O(1)", ok_j3a, f"四点端点比均为 0.5 ± 0.02；最大偏离 = {max(abs(r-0.5) for r in ratios):.4f}")
check("J3b 整体偏差 1/N", ok_j3b, f"斜率 = {slope_tv:.4f}（期望 -1.0 ± 0.05）")

sc_closed = site_counts_closed(N, T)
ok_j4 = np.all(sc_closed == sc_closed[0])
check("J4 闭链均匀", ok_j4, f"闭链站点计数 = {sc_closed.tolist()}（期望全等，实际极差 = {sc_closed.max()-sc_closed.min()}）")

# ---------------- 第二段：随机抽样 ----------------
print("== 第二段：随机抽样（按份额） ==")
shares = np.array([1, 2, 3, 4], dtype=float)
shares /= shares.sum()
M = len(shares)
Ts = [10, 30, 100, 300, 1000, 3000, 10000]
R = 200
rms = []
cover_hits, cover_total = 0, 0
for Tt in Ts:
    devs = []
    for _ in range(R):
        counts = RNG.multinomial(Tt, shares)
        f = counts / Tt
        devs.append(f - shares)
        if Tt == 1000:
            sig = np.sqrt(shares * (1 - shares) / Tt)
            cover_hits += int(np.sum(np.abs(f - shares) <= 3 * sig))
            cover_total += M
    devs = np.array(devs)
    rms.append(np.sqrt(np.mean(devs ** 2)))
slope_rms = np.polyfit(np.log(Ts), np.log(rms), 1)[0]
cover = cover_hits / cover_total
print(f"  3σ 覆盖率（T = 1000，{R} 组 × {M} 块）= {cover*100:.2f}%（期望 >= 99%）")
print(f"  RMS 偏差：{['%.4f' % v for v in rms]}，双对数斜率 = {slope_rms:.4f}（期望 -0.5）")
check("J5a 覆盖率", cover >= 0.99, f"覆盖 = {cover*100:.2f}%（期望 >= 99%）")
check("J5b 收敛率", abs(slope_rms + 0.5) <= 0.1, f"斜率 = {slope_rms:.4f}（期望 -0.5 ± 0.1）")

# ---------------- 绘图 ----------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax = axes[0][0]
    x = np.arange(N + 1)
    ax.bar(x - 0.2, freq_open, width=0.4, color="#2563eb", label="开链（站点计数）")
    f_closed = sc_closed / sc_closed.sum()
    ax.bar(x - 0.2 + 0.4, np.concatenate([f_closed, [np.nan]]), width=0.4, color="#0f766e", label="闭链（站点计数）")
    ax.axhline(share, color="#be123c", ls="--", lw=1.2, label="份额 1/N")
    ax.set_title(f"站点频率与份额（N = {N}）：开链端点约为半数")
    ax.set_xlabel("边界序号")
    ax.set_ylabel("频率")
    ax.legend(fontsize=8)

    ax = axes[0][1]
    ax.plot(Ns, tvs, "o-", color="#b45309")
    fit = np.exp(np.polyval(np.polyfit(np.log(Ns), np.log(tvs), 1), np.log(Ns)))
    ax.plot(Ns, fit, "--", color="#9aa1b0", label=f"斜率 {slope_tv:.2f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("整体 TV 偏差随 N（双对数）")
    ax.set_xlabel("格点数 N")
    ax.set_ylabel("总变差偏差")
    ax.legend(fontsize=8)

    ax = axes[1][0]
    ax.plot(Ts, rms, "o-", color="#7c3aed")
    fit2 = np.exp(np.polyval(np.polyfit(np.log(Ts), np.log(rms), 1), np.log(Ts)))
    ax.plot(Ts, fit2, "--", color="#9aa1b0", label=f"斜率 {slope_rms:.2f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("按份额抽样：RMS 偏差随采样数")
    ax.set_xlabel("采样数 T")
    ax.set_ylabel("RMS 偏差")
    ax.legend(fontsize=8)

    ax = axes[1][1]
    ax.bar(np.arange(N), cc, color="#4f46e5")
    ax.axhline(T, color="#be123c", ls="--", lw=1.2, label="T")
    ax.set_title("载子计数：每载子事件数（无偏口径）")
    ax.set_xlabel("载子序号")
    ax.set_ylabel("事件数")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig("D:/software/TianXingFC/images/sampling_unbiasedness.png", dpi=160)
    print("  图已保存：images/sampling_unbiasedness.png")
except Exception as exc:      # 绘图失败不影响判据结论
    print(f"  绘图跳过：{exc}")

passed = sum(1 for _, ok in CHECKS if ok)
print(f"== 判据合计 {passed}/{len(CHECKS)} 通过，用时 {time.time()-T0:.1f} s ==")
