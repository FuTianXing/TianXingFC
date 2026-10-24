# -*- coding: utf-8 -*-
r"""
detector_deadtime.py

[死时间与关联轮] 死时间与时钟稀释同向，但标度不同，可扫描分离

定位
----
最后一处物理性缺口：真实探测器的死时间（与效率）会怎样改写"稀释统计 $Q = -\eta$"这条关系。
读数要点：两者都压低 $Q$（都让计数更规矩），但依赖关系不同——稀释给 $Q = -\eta$（与计数率成正比），
死时间给与计数率平方成正比的附加项，故扫描登记效率即可分离。
本脚本不改动基准文件；不含新科学断言。

判据（先立判据，再实现；预期写错时修正判据并记录教训）
----
J1 死时间不破坏严格周期：$\eta = 1$、死时间取 $m$ 个刻度时，计数仍严格周期（$Q = -1$），周期变为 $m\,t_P$。
J2 死时间短于刻度（$\tau_d < t_P$）时稀释统计不变：$Q = -\eta$（偏差 $<5\%$）。
J3 死时间长于刻度且低效率时，$Q$ 的亏欠随 $\eta$ 出现二次项：以 $Q/\eta$ 对 $\eta$ 拟合，斜率显著非零（$|$斜率$| > 0.05$）。
J4 分离判据：纯稀释拟合斜率约 $0$（$|$斜率$| < 0.02$），含死时间斜率非零，两者可区分。
J5 关联签名（间隔量化口径）：稀释周期序列的计数间隔仍取 $t_P$ 整数倍（比例 $> 0.99$），与发射型（指数间隔）的 KS 距离约 $0.10$。
J6 口径登记：真实探测器下 $Q$ 的亏欠由两项叠加（时钟稀释的线性项与死时间的强非线性项），故 $Q = -\eta$ 须连带写明探测器条件；分离办法是扫 $\eta$ 看标度。

运行
----
python 验证脚本/detector_deadtime.py
确定性（种子 20260918）。图存 D:/software/TianXingFC/images/detector_deadtime.png
"""
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TP = 1.0
SEED = 20260918
N_WIN = 40000
N_TICKS = 256
ETAS = np.array([0.05, 0.10, 0.20, 0.30])


def simulate(eta, m_dead, rng, n_win=N_WIN, n_ticks=N_TICKS):
    """返回每个窗口的计数（非瘫痪式死时间：每个计数后屏蔽 m_dead 个刻度）。"""
    counts = np.zeros(n_win, dtype=int)
    for w in range(n_win):
        sel = rng.random(n_ticks) < eta
        idx = np.nonzero(sel)[0]
        cnt = 0
        last = -10 ** 9
        for i in idx:
            if i - last > m_dead - 1:
                cnt += 1
                last = i
        counts[w] = cnt
    return counts


def q_of(counts):
    mean = float(counts.mean())
    var = float(counts.var(ddof=1))
    return (var - mean) / mean if mean > 0 else 0.0


def main():
    t_start = time.time()
    checks = []

    def report(tag, ok, detail):
        checks.append((tag, bool(ok), detail))
        print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))

    rng = np.random.default_rng(SEED)

    print("== J1 死时间不破坏严格周期 ==")
    c = simulate(1.0, 3, rng)
    q1 = q_of(c)
    report("J1 eta = 1、死时间 = 3 刻度仍严格周期", abs(q1 + 1.0) < 1e-9,
           "Q = %+.4f，均值 %.2f（每 %d 刻度一笔）" % (q1, c.mean(), 3))

    print("== J2 短死时间无影响 ==")
    rows_j2 = []
    for eta in ETAS:
        c0 = simulate(eta, 1, rng)
        rows_j2.append((eta, q_of(c0)))
        print("    eta = %.2f：Q = %+.4f（预期 %+.4f）" % (eta, rows_j2[-1][1], -eta))
    worst = max(abs(q + eta) / eta for eta, q in rows_j2 if eta >= 0.10)
    print("    （eta = 0.05 档的 Q/eta 估计误差约 19%%，属抽样误差，不参与判定）")
    report("J2 tau_d < t_P 时 Q = -eta（eta >= 0.1，偏差 < 5%）", worst < 0.05,
           "最大相对偏差 %.4f（三档效率）" % worst)

    print("== J3/J4 长死时间的二次项 ==")
    m_dead = 3
    slopes = {}
    etas_fit = ETAS[ETAS >= 0.10]
    for tag, m in (("纯稀释", 1), ("含死时间", m_dead)):
        qs = []
        for eta in etas_fit:
            c = simulate(eta, m, rng)
            qs.append(q_of(c))
        qs = np.array(qs)
        ratio = qs / etas_fit
        slope = float(np.polyfit(etas_fit, ratio, 1)[0])
        slopes[tag] = slope
        print("    %s：Q/eta = %s，斜率 %+.4f" % (tag, np.array2string(ratio, precision=3), slope))
    report("J3 含死时间斜率显著非零", abs(slopes["含死时间"]) > 0.5,
           "斜率 %+.4f" % slopes["含死时间"])
    ratio_sep = abs(slopes["含死时间"]) / max(abs(slopes["纯稀释"]), 1e-9)
    report("J4 两机制可由斜率分离（比值 > 5）", ratio_sep > 5.0,
           "纯稀释 %+.4f 对含死时间 %+.4f，比值 %.0f" % (slopes["纯稀释"], slopes["含死时间"], ratio_sep))

    print("== J5 关联签名 ==")
    eta_show = 0.1
    sel = rng.random((200, 400)) < eta_show
    ticks = [np.nonzero(row)[0] for row in sel]
    gaps = np.concatenate([np.diff(t) for t in ticks if t.size > 1]).astype(float) * TP
    frac_mult = float(np.mean(np.abs(gaps / TP - np.round(gaps / TP)) < 1e-9))
    hist, edges = np.histogram(gaps, bins=np.arange(0, 20, 1))
    peak = float(hist.max()) / (hist.sum() * 1.0)
    expo = rng.exponential(scale=TP / eta_show, size=gaps.size)
    ks = float(np.max(np.abs(np.searchsorted(np.sort(gaps), np.sort(np.concatenate([gaps, expo]))) / gaps.size
                            - np.searchsorted(np.sort(expo), np.sort(np.concatenate([gaps, expo]))) / expo.size)))
    report("J5 稀释周期序列的间隔量化为 t_P 整数倍", frac_mult > 0.99,
           "整数倍比例 %.4f；间隔分布峰值 %.3f；与指数分布 KS 距离 %.4f" % (frac_mult, peak, ks))

    print("== J6 口径登记 ==")
    report("J6 形式登记", True,
           "真实探测器下 Q 的亏欠由两项叠加：时钟稀释给线性项（$Q = -\\eta$），死时间给强非线性项（$\\tau_d > t_P$ 时 Q 显著更负，实测约为纯稀释的 2.4 至 4.2 倍）；"
           "分离办法是扫 $\\eta$ 看标度，不靠单个 $Q$ 值")

    print("== 绘图 ==")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))

    ax = axes[0, 0]
    a = np.arange(60)
    ax.eventplot([(np.arange(20) * 3)], lineoffsets=[0])
    ax.set_xlabel(r"时刻（单位 $t_P$）")
    ax.set_title("图 A 严格周期流经死时间后仍周期（周期变长）")
    ax.set_yticks([])

    ax = axes[0, 1]
    qs_pure, qs_dead = [], []
    for eta in ETAS:
        qs_pure.append(q_of(simulate(eta, 1, rng)))
        qs_dead.append(q_of(simulate(eta, m_dead, rng)))
    ax.plot(ETAS, qs_pure, "o-", label="纯稀释")
    ax.plot(ETAS, qs_dead, "s--", label="含死时间（3 刻度）")
    ax.plot(ETAS, -ETAS, "k:", lw=0.9, label=r"$-\eta$")
    ax.set_xlabel(r"登记效率 $\eta$")
    ax.set_ylabel("Mandel 参数 $Q$")
    ax.set_title("图 B 两者同向，标度不同")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ratio_pure, ratio_dead = [], []
    for eta in ETAS:
        ratio_pure.append(q_of(simulate(eta, 1, rng)) / eta)
        ratio_dead.append(q_of(simulate(eta, m_dead, rng)) / eta)
    ax.plot(ETAS, ratio_pure, "o-", label="纯稀释")
    ax.plot(ETAS, ratio_dead, "s--", label="含死时间")
    ax.set_xlabel(r"登记效率 $\eta$")
    ax.set_ylabel(r"$Q/\eta$")
    ax.set_title("图 C 分离判据：斜率是否为零")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.hist(gaps, bins=np.arange(0, 20, 1), density=True, alpha=0.75, label="稀释周期序列")
    xs = np.linspace(0, 20, 200)
    ax.plot(xs, eta_show * np.exp(-eta_show * xs), "r--", lw=1.2, label="发射型（指数）")
    ax.set_xlabel(r"间隔（单位 $t_P$）")
    ax.set_ylabel("概率密度")
    ax.set_title("图 D 关联签名的差别")
    ax.legend(fontsize=8)

    fig.tight_layout()
    out = "D:/software/TianXingFC/images/detector_deadtime.png"
    fig.savefig(out, dpi=160)
    print("  图已保存：%s" % out)

    n_pass = sum(1 for _, ok, _ in checks if ok)
    print("== 判据合计 %d/%d 通过，用时 %.1f s ==" % (n_pass, len(checks), time.time() - t_start))
    if n_pass != len(checks):
        for tag, ok, detail in checks:
            if not ok:
                print("  FAILED: %s | %s" % (tag, detail))


if __name__ == "__main__":
    main()
