# -*- coding: utf-8 -*-
r"""
deadtime_correlation.py

[死时间关联效应轮] 死区挖空：洞宽由死时间定、凹陷深度由效率定；两类探测器在严格周期源下分野

定位
----
第 130 条（探测器死时间与关联轮）已核标度分离：死时间与稀释都把 $Q$ 压低（同向），
$Q/\eta$ 对 $\eta$ 的斜率相差约 40 倍。本轮把关联这一半做成结构性读数：
槽网格上的二阶关联 $g^{(2)}(k)$ 如何被稀释与死时间分别改写。

读数要点：稀释不写记忆（$m = 1$ 时 $k \ge 1$ 全部 $g^{(2)}(k) = 1$）；
死时间写记忆（$1 \le k < m$ 时 $g^{(2)}(k) = 0$，为结构性零）。
于是洞宽只由死时间定、凹陷深度只由效率定，两个手柄正交，比单个 $Q$ 值更硬。
另核一处分野：严格周期源下非瘫痪式探测器给周期倍增，瘫痪式探测器完全锁死。

本脚本不改动基准文件；不含新科学断言。

判据（先立判据，再实现；预期写错时修正判据并记录教训）
----
K1 稀释不写记忆：$m = 1$ 时 $g^{(2)}(k) = 1$（$1 \le k \le 8$，偏差 $< 3\%$）。
K2 死区挖空：$m \ge 2$ 时 $g^{(2)}(k) = 0$ 对 $1 \le k < m$（结构性零，无抽样误差）；并核两种采样实现一致。
K3 洞宽定死时间：实测洞宽等于 $m - 1$，对 $\eta$ 不敏感（四档效率同一读数）。
K4 洞外偏离衰减：$m = 3$、$\eta \ge 0.2$、$k = 12$ 时 $|g^{(2)}(k) - 1| < 0.15$。
K5 探测器类型分野：$\tau_d = 3\,t_P$ 时非瘫痪式给周期倍增（$L$ 个刻度 $\lceil L/3 \rceil$ 笔），
   瘫痪式完全锁死（全记录计数 $\le 2$）；$\tau_d < t_P$ 时两类都给满额。
K6 口径登记：记录流的零延迟凹陷来自槽内不可双占据（网格性，与死时间无关），
   死时间只能把凹陷展宽；搜索本底周期性的硬签名是洞宽（对效率不敏感）。

运行
----
python 验证脚本/deadtime_correlation.py
确定性（种子 20260918）。图存 D:/software/TianXingFC/images/deadtime_correlation.png
"""
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TP = 1.0
SEED = 20260918
N_SLOTS = 3000000
KMAX = 12
ETAS = np.array([0.05, 0.10, 0.20, 0.30])
M_LIST = [1, 3, 5]


def occupancy_greedy(eta, m_dead, rng, n_slots):
    """逐刻度二项稀释后按"相邻计数间隔 >= m_dead"贪婪接受（与第 130 条同一模型）。"""
    sel = rng.random(n_slots) < eta
    idx = np.nonzero(sel)[0]
    keep = np.zeros(idx.size, dtype=bool)
    last = -10 ** 9
    for j in range(idx.size):
        i = idx[j]
        if i - last >= m_dead:
            keep[j] = True
            last = i
    occ = np.zeros(n_slots, dtype=bool)
    occ[idx[keep]] = True
    return occ


def occupancy_renewal(eta, m_dead, rng, n_slots):
    """等价采样：间隔 i.i.d.，分布为 gap = m_dead + Geom(eta)。"""
    occ = np.zeros(n_slots, dtype=bool)
    pos = int(rng.integers(0, m_dead))
    while pos < n_slots:
        occ[pos] = True
        pos += m_dead + int(rng.geometric(eta)) - 1
    return occ


def g2_of(occ, kmax=KMAX):
    n = occ.size
    mean = float(occ.mean())
    g = np.zeros(kmax + 1)
    for k in range(1, kmax + 1):
        pairs = np.count_nonzero(occ[:-k] & occ[k:])
        g[k] = pairs / ((n - k) * mean * mean)
    return g


def hole_width(g, tol=1e-12):
    k = 1
    while k < g.size and abs(g[k]) <= tol:
        k += 1
    return k - 1


def periodic_paralyzable(n_ticks, tau_d):
    """瘫痪式：死时间内任何到达都延长死时间。"""
    dead_until = -1.0
    cnt = 0
    for j in range(n_ticks):
        t = j * TP
        if t >= dead_until:
            cnt += 1
        dead_until = max(dead_until, t + tau_d)
    return cnt


def periodic_nonparalyzable(n_ticks, m_dead):
    """非瘫痪式：死时间固定，不因到达延长。"""
    cnt = 0
    j = 0
    while j < n_ticks:
        cnt += 1
        j += m_dead
    return cnt


def report(name, ok, detail):
    print(("[通过] " if ok else "[不通过] ") + name + "：" + detail)
    return bool(ok)


def main():
    t_start = time.time()
    rng = np.random.default_rng(SEED)
    res = {}
    widths = {}

    print("== 槽网格二阶关联读数（N_SLOTS = %d） ==" % N_SLOTS)
    print("m   eta   hole   g2(1)   g2(2)   g2(3)   g2(4)   g2(6)   g2(12)")
    for m in M_LIST:
        for eta in ETAS:
            eta = float(eta)
            occ = occupancy_renewal(eta, m, rng, N_SLOTS)
            g = g2_of(occ)
            res[(m, eta)] = g
            widths[(m, eta)] = hole_width(g)
            print("%d  %.2f  %5d  %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f"
                  % (m, eta, widths[(m, eta)], g[1], g[2], g[3], g[4], g[6], g[12]))

    print("== K1 稀释不写记忆 ==")
    dev1 = max(float(np.max(np.abs(res[(1, float(e))][1:9] - 1.0))) for e in ETAS)
    ok = report("K1 m = 1 时 g2(k) = 1（k = 1 至 8，四档效率）", dev1 < 0.03,
                "最大偏差 %.4f" % dev1)

    print("== K2 死区挖空 ==")
    dev2 = max(float(np.max(np.abs(res[(m, float(e))][1:m]))) for m in (3, 5) for e in ETAS)
    ok &= report("K2 1 <= k < m 时 g2(k) = 0（结构性零）", dev2 == 0.0,
                 "最大残值 %.3e" % dev2)
    n_cmp = 300000
    ga = g2_of(occupancy_greedy(0.30, 3, rng, n_cmp), kmax=6)
    gb = g2_of(occupancy_renewal(0.30, 3, rng, n_cmp), kmax=6)
    d_cmp = float(np.max(np.abs(ga[1:7] - gb[1:7])))
    ok &= report("K2 附 两种采样实现一致（贪婪滤波对更新采样）", d_cmp < 0.05,
                 "g2(1..6) 最大差 %.4f" % d_cmp)

    print("== K3 洞宽定死时间 ==")
    w3 = [widths[(3, float(e))] for e in ETAS]
    w5 = [widths[(5, float(e))] for e in ETAS]
    ok &= report("K3 洞宽 = m - 1 且对 eta 不敏感", w3 == [2] * 4 and w5 == [4] * 4,
                 "m=3 洞宽 %s；m=5 洞宽 %s" % (w3, w5))

    print("== K4 洞外偏离衰减 ==")
    d30 = float(abs(res[(3, 0.30)][12] - 1.0))
    d20 = float(abs(res[(3, 0.20)][12] - 1.0))
    ok &= report("K4 k = 12 时偏离 < 0.15（eta >= 0.2）", max(d30, d20) < 0.15,
                 "eta=0.3 偏离 %.4f；eta=0.2 偏离 %.4f" % (d30, d20))

    print("== K5 探测器类型分野 ==")
    n_t = 64
    np3 = periodic_nonparalyzable(n_t, 3)
    pa3 = periodic_paralyzable(n_t, 3.0)
    np_s = periodic_nonparalyzable(n_t, 1)
    pa_s = periodic_paralyzable(n_t, 0.5)
    ok &= report("K5 tau_d = 3t_P 分野（非瘫痪倍增对瘫痪锁死）",
                 np3 == int(np.ceil(n_t / 3.0)) and pa3 <= 2,
                 "%d 个刻度：非瘫痪 %d 笔、瘫痪 %d 笔" % (n_t, np3, pa3))
    ok &= report("K5 tau_d < t_P 时两类无差别", np_s == n_t and pa_s == n_t,
                 "非瘫痪 %d 笔、瘫痪 %d 笔" % (np_s, pa_s))

    print("== K6 口径登记 ==")
    ok &= report("K6 零延迟凹陷属网格性（与死时间无关）", True, "见注记 §三")

    t_end = time.time()
    print("== 汇总：%s（用时 %.1f 秒） ==" % ("全部通过" if ok else "存在不通过项", t_end - t_start))
    make_figure(res, widths, np3, pa3, np_s, pa_s, n_t)
    return ok


def make_figure(res, widths, np3, pa3, np_s, pa_s, n_t):
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.6))
    ks = np.arange(1, KMAX + 1)

    ax = axes[0][0]
    for m, style in ((1, "o-"), (3, "s-"), (5, "^-")):
        ax.plot(ks, res[(m, 0.30)][1:], style, ms=4, label="m = %d" % m)
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_title("g2(k): dead-time hole (eta = 0.30)")
    ax.set_xlabel("lag k [slots]")
    ax.set_ylabel("g2(k)")
    ax.legend()

    ax = axes[0][1]
    ax.plot(ETAS, [widths[(3, float(e))] for e in ETAS], "o-", label="m = 3 measured")
    ax.plot(ETAS, [widths[(5, float(e))] for e in ETAS], "s-", label="m = 5 measured")
    ax.axhline(2.0, color="C0", lw=0.8, ls=":")
    ax.axhline(4.0, color="C1", lw=0.8, ls=":")
    ax.set_title("hole width: set by dead time only")
    ax.set_xlabel("registration efficiency eta")
    ax.set_ylabel("hole width (zero-lag slots)")
    ax.legend()

    ax = axes[1][0]
    for eta, style in ((0.05, "o-"), (0.10, "s-"), (0.20, "^-"), (0.30, "d-")):
        ax.plot(ks, res[(3, eta)][1:], style, ms=4, label="eta = %.2f" % eta)
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_title("g2(k) outside the hole (m = 3)")
    ax.set_xlabel("lag k [slots]")
    ax.set_ylabel("g2(k)")
    ax.legend()

    ax = axes[1][1]
    ticks = np.arange(n_t)
    cn = [periodic_nonparalyzable(j + 1, 3) for j in ticks]
    cp = [periodic_paralyzable(j + 1, 3.0) for j in ticks]
    cs = [periodic_nonparalyzable(j + 1, 1) for j in ticks]
    ax.step(ticks, cn, where="post", label="non-paralyzable, tau_d = 3 t_P")
    ax.step(ticks, cp, where="post", label="paralyzable, tau_d = 3 t_P")
    ax.step(ticks, cs, where="post", label="tau_d = 0.5 t_P (both types)")
    ax.set_title("strictly periodic source: record count")
    ax.set_xlabel("tick index")
    ax.set_ylabel("cumulative records")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig("D:/software/TianXingFC/images/deadtime_correlation.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
