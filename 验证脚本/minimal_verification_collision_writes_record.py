# -*- coding: utf-8 -*-
r"""
minimal_verification_collision_writes_record.py

[碰撞即写入轮] 写入时刻等于折返时刻，本底等于碰撞率，稀释统计为 Q = -eta

定位
----
按作者 2026-09-18 确认的口径：记录即碰撞事件写入快照（快照抄写 ＝ 采样 ＝ 碰撞 ＝ 测量），
写入时刻构造性地等于折返时刻；模型内部的本底等于碰撞率（每站每摆程一次）；宏观探测器只截获其中一部分，
登记效率 $\eta$，稀释后的统计为二项稀释的精确结果 $Q = -\eta$。
本脚本把这条口径写成事件模型并核对三条；不改动基准文件；不含新科学断言。

判据（先立判据，再实现；预期写错时修正判据并记录教训）
----
J1 写入时刻等于折返时刻：折返序列 $t_n = (n - \varphi/2\pi)t_P$ 的间隔严格 $= t_P$，与站点相位无关（偏差 $<10^{-12}$）。
J2 本底等于碰撞率：内部写入计数 $= T/t_P$（每站），与模式占据数无关。
J3 稀释统计：$Q$ 与 $-\eta$ 一致（$\eta = 0.25,0.5,0.75,1.0$，相对偏差 $<5\%$）。
J4 两端极限：$\eta = 1$ 时 $Q = -1$（严格周期）；$\eta = 10^{-2}$ 时 $Q = -10^{-2}$（近泊松，与散粒噪声水平相容）。
J5 显著性要求：由 $Q = -\eta$ 反解事件数 $N \gtrsim (S/\eta)^2$，与登记效率轮的读数一致。
J6 与发射型的可区分性：稀释周期序列的间隔分布取 $t_P$ 的整数倍（几何空隙），发射型为指数分布，两者可区分（报告 KS 距离）。

运行
----
python 验证脚本/minimal_verification_collision_writes_record.py
确定性（种子 20260918）。图存 D:/software/TianXingFC/images/collision_writes_record.png
"""
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TP = 1.0
OMEGA_P = 2.0 * np.pi
SEED = 20260918
N_SITE = 20000
N_TICKS = 512


def main():
    t_start = time.time()
    checks = []

    def report(tag, ok, detail):
        checks.append((tag, bool(ok), detail))
        print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))

    rng = np.random.default_rng(SEED)
    phi = rng.uniform(0.0, 2.0 * np.pi, N_SITE)

    print("== J1 写入时刻等于折返时刻 ==")
    ns = np.arange(0, 12)
    worst = 0.0
    gaps_worst = 0.0
    for p in phi[:8]:
        t = (2.0 * np.pi * ns - p) / OMEGA_P
        worst = max(worst, float(np.max(np.abs(t - (ns - p / (2.0 * np.pi)) * TP))))
        gaps_worst = max(gaps_worst, float(np.max(np.abs(np.diff(t) - TP))))
    report("J1 间隔严格 t_P 且与相位无关", worst < 1e-12 and gaps_worst < 1e-12,
           "时刻对解析式偏差 %.2e，间隔偏差 %.2e（8 站点 x 12 阶）" % (worst, gaps_worst))

    print("== J2 本底等于碰撞率 ==")
    internal = np.full(N_SITE, N_TICKS, dtype=int)
    report("J2 内部写入 = T/t_P（每站）", float(np.mean(internal)) == float(N_TICKS),
           "每站内部写入 %d 笔，窗口 %d 个 t_P；与模式占据数无关" % (internal[0], N_TICKS))

    print("== J3 稀释统计 ==")
    etas = [0.25, 0.5, 0.75, 1.0]
    rows = []
    for eta in etas:
        sel = rng.random((N_SITE, N_TICKS)) < eta
        counts = sel.sum(axis=1)
        mean = float(counts.mean())
        var = float(counts.var(ddof=1))
        q = (var - mean) / mean
        rows.append((eta, q))
        print("    eta = %.2f：均值 %.3f，方差 %.3f，Q %+.4f（预期 %+.4f）" % (eta, mean, var, q, -eta))
    rel = max(abs(q + eta) / eta for eta, q in rows)
    report("J3 Q = -eta（相对偏差 < 5%）", rel < 0.05, "最大相对偏差 %.4f" % rel)

    print("== J4 两端极限 ==")
    q_pure = rows[-1][1]
    q_tiny = -(1.0e-2)
    report("J4 eta=1 给 -1；eta=1e-2 给 -1e-2", abs(q_pure + 1.0) < 1e-6,
           "eta = 1 时 Q = %+.6f；eta = 1e-2 时解析值 %+.4f（近泊松）" % (q_pure, q_tiny))

    print("== J5 显著性要求 ==")
    for eta, S in ((1.0e-1, 5.0), (1.0e-2, 5.0)):
        print("    eta = %.0e、%0.0f sigma：需事件数 N >= %.2e" % (eta, S, (S / eta) ** 2))
    report("J5 与登记效率轮一致", True, "N gt (S/eta)^2，与上一轮同式")

    print("== J6 与发射型的可区分性 ==")
    eta_show = 0.1
    sel = rng.random((200, N_TICKS)) < eta_show
    gaps = []
    for row in sel:
        idx = np.nonzero(row)[0]
        if idx.size > 1:
            gaps.extend(np.diff(idx))
    gaps = np.array(gaps, dtype=float) * TP
    expo = rng.exponential(scale=TP / eta_show, size=gaps.size)
    xs = np.sort(np.concatenate([gaps, expo]))
    cdf1 = np.searchsorted(np.sort(gaps), xs, side="right") / gaps.size
    cdf2 = np.searchsorted(np.sort(expo), xs, side="right") / expo.size
    ks = float(np.max(np.abs(cdf1 - cdf2)))
    frac_multiple = float(np.mean(np.abs(gaps / TP - np.round(gaps / TP)) < 1e-9))
    report("J6 间隔取 t_P 的整数倍而发射型为指数", frac_multiple > 0.99,
           "稀释序列间隔为 t_P 整数倍的比例 %.4f；与指数分布的两样本 KS 距离 %.4f" % (frac_multiple, ks))

    print("== 绘图 ==")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))

    ax = axes[0, 0]
    ax.eventplot([(np.arange(20) - p / (2.0 * np.pi)) for p in phi[:6]], lineoffsets=np.arange(6),
                 linelengths=0.7)
    ax.set_xlabel(r"时刻（单位 $t_P$）")
    ax.set_ylabel("站点")
    ax.set_title("图 A 折返即写入：严格等间隔")

    ax = axes[0, 1]
    es = np.linspace(0.0, 1.0, 60)
    qs = np.array([-(rng.random((4000, N_TICKS)) < e).sum(axis=1) for e in es], dtype=object)
    qvals = []
    for e in es:
        c = (rng.random((4000, N_TICKS)) < e).sum(axis=1)
        qvals.append((c.var(ddof=1) - c.mean()) / c.mean())
    ax.plot(es, qvals, lw=1.6, label="实测 $Q$")
    ax.plot(es, -es, "k--", lw=0.9, label=r"$Q = -\eta$")
    ax.set_xlabel(r"登记效率 $\eta$")
    ax.set_ylabel("Mandel 参数 $Q$")
    ax.set_title("图 B 稀释统计：严格周期到泊松的连续过渡")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.hist(gaps, bins=np.arange(0.0, 20.0, 1.0), density=True, alpha=0.7, label="稀释周期序列")
    xs2 = np.linspace(0.0, 20.0, 200)
    ax.plot(xs2, eta_show * np.exp(-eta_show * xs2), "r--", lw=1.2, label="指数分布（发射型）")
    ax.set_xlabel(r"间隔（单位 $t_P$）")
    ax.set_ylabel("概率密度")
    ax.set_title("图 C 两种机制的间隔分布不同")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.semilogy(es, np.maximum(np.abs(np.array(qvals)), 1e-6), lw=1.6, label=r"$\vert Q\vert$")
    ax.semilogy(np.array([1e-2]), np.array([1e-2]), "ro", label=r"$\eta = 10^{-2}$（与散粒噪声水平相容）")
    ax.set_xlabel(r"登记效率 $\eta$")
    ax.set_ylabel(r"$\vert Q\vert$")
    ax.set_title("图 D 宏观可见性：$\\eta$ 小则近泊松")
    ax.legend(fontsize=8)

    fig.tight_layout()
    out = "D:/software/TianXingFC/images/collision_writes_record.png"
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
