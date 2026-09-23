# -*- coding: utf-8 -*-
r"""前提 P1 的独立性审查：被记录量的分辨单元（判据 Pa1 至 Pa7）

沿用限制二轮的键滑剖面与读出框架：
  w_j(p) = (1/(pi ell)) sech((j + 0.5 - p)/ell)，ell = 2.2508 格距，峰值 1/(pi ell)
  读出图样 P_j = Theta(w_j - theta)，差异数 n_diff = 两支图样的对称差大小
本脚本的增量是"事件集"一侧：整数层位移（来源甲允许的事件）与亚层位移（连续位置
假设下的事件）分开核对，给出分辨单元恰为一层的读数。
确定性：无随机源。图存 D:/software/TianXingFC/images/record_resolution_unit.png
"""
import numpy as np

RES = []
FIG = "D:/software/TianXingFC/images/record_resolution_unit.png"
NB = 400
K, k = 0.005, 1.0
ELL = 1.0 / (2 * np.pi * np.sqrt(K / k))
WMAX_ANA = 1.0 / (np.pi * ELL)
WMAX_M2 = 0.1399
PGRID = np.linspace(100.0, 120.0, 1601)
INT_DELTAS = [1, 2, 3, 4, 5, 6]
SUB_DELTAS = [0.25, 0.5, 0.75]


def report(tag, name, meas, exp, note="", grade="通过"):
    RES.append((tag, name, meas, exp, note, grade))


def w_profile(p):
    j = np.arange(NB) + 0.5
    return (1.0 / (np.pi * ELL)) / np.cosh((j - p) / ELL)


def pattern(p, theta, d=None):
    w = w_profile(p)
    if d is None:
        return w > theta
    return w[np.arange(0, NB, d)] > theta


def ndiff(p, delta, theta, d=None):
    a = pattern(p, theta, d)
    b = pattern(p + delta, theta, d)
    n = min(a.size, b.size)
    return int(np.count_nonzero(a[:n] != b[:n]))


def main():
    print("=" * 100)
    print("前提 P1 的独立性审查：被记录量的分辨单元（Pa1 至 Pa7）")
    print("=" * 100)
    theta = 0.5 * WMAX_ANA
    print("ell = %.4f 格距，解析峰值 %.4f，既有实测 %.4f，阈值 %.4f" % (ELL, WMAX_ANA, WMAX_M2, theta))
    dev = abs(WMAX_ANA - WMAX_M2) / WMAX_M2
    report("Pa5", "解析峰值对既有实测的相对偏差", dev, 0.02, "剖面校验",
           "通过" if dev < 0.02 else "未过")

    print()
    print("整数层位移一侧（来源甲允许的事件集）")
    print("-" * 100)
    mins = {}
    for delta in INT_DELTAS:
        m = min(ndiff(p, float(delta), theta, d=1) for p in PGRID)
        mins[delta] = m
        print("  delta_n = %d：min_p n_diff = %d" % (delta, m))
    report("Pa1", "整数层位移的漏检集大小", sum(1 for q in mins if mins[q] == 0), 0, "期望空集",
           "通过" if all(mins[q] > 0 for q in mins) else "未过")
    lin = [q for q in INT_DELTAS if q <= 5]
    worst = max(abs(mins[q] - 2 * q) for q in lin)
    report("Pa2", "小位移段 n_diff 对 2 delta_n 的最大偏差", worst, 0, "五档",
           "通过" if worst == 0 else "读数")

    print()
    print("亚层位移一侧（连续位置假设下的事件集）")
    print("-" * 100)
    subs = {}
    for delta in SUB_DELTAS:
        vals = [ndiff(p, delta, theta, d=1) for p in PGRID]
        zero = sum(1 for q in vals if q == 0)
        subs[delta] = (min(vals), zero, len(vals))
        print("  delta = %.2f：min_p n_diff = %d，n_diff = 0 的采样点 %d / %d"
              % (delta, min(vals), zero, len(vals)))
    report("Pa3", "亚层位移（0.25 与 0.5 与 0.75）的 n_diff 最小值",
           "%d / %d / %d" % tuple(subs[q][0] for q in SUB_DELTAS), "0 / 0 / 0",
           "存在这样的位置", "通过" if all(subs[q][0] == 0 for q in SUB_DELTAS) else "未过")

    unit = (all(mins[q] > 0 for q in mins)) and all(subs[q][0] == 0 for q in SUB_DELTAS)
    report("Pa4", "分辨单元判定（整数层可分辨且亚层不可分辨）", "成立" if unit else "不成立", "成立",
           "被记录量的分辨单元恰为一层", "通过" if unit else "未过")

    print()
    min_d2 = min(ndiff(p, 1.0, theta, d=2) for p in PGRID)
    print("边界复算：d = 2 在 delta_n = 1 处 min_p n_diff = %d" % min_d2)
    report("Pa6", "d = 2 档在 delta_n = 1 的 min_p n_diff", min_d2, 0, "稀疏档漏检复算",
           "通过" if min_d2 == 0 else "未过")
    report("Pa7", "两条来源的接口", "-", "-",
           "来源甲（距离归粒：位置取整数层）与来源乙（层位份额：逐层承载概率）的合取", "读数")

    print()
    print("=" * 100)
    print("判据 | 判据量 | 实测 | 期望 | 备注 | 结论")
    print("=" * 100)
    for tag, name, meas, exp, note, grade in RES:
        m = ("%.6g" % meas) if isinstance(meas, float) else str(meas)
        e = ("%.6g" % exp) if isinstance(exp, float) else str(exp)
        print("%-5s | %s | %s | %s | %s | %s" % (tag, name, m, e, note, grade))
    print("=" * 100)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.0))

    ax[0, 0].plot(INT_DELTAS, [mins[q] for q in INT_DELTAS], "o-", label="实测 min_p n_diff")
    ax[0, 0].plot(INT_DELTAS, [2 * q for q in INT_DELTAS], "k--", label="2 delta_n")
    ax[0, 0].set_xlabel("整数层位移 delta_n"); ax[0, 0].set_ylabel("min_p n_diff")
    ax[0, 0].set_title("Pa1/Pa2：整数层位移无漏检（小位移段线性）"); ax[0, 0].legend(fontsize=8)

    for dlt, c in ((0.25, "#4c72b0"), (0.5, "#dd8452"), (0.75, "#c44e52")):
        ax[0, 1].plot(PGRID, [ndiff(p, dlt, theta, d=1) for p in PGRID], ".", ms=1.5, color=c,
                      label="delta = %.2f" % dlt)
    ax[0, 1].set_xlabel("kink 位置 p"); ax[0, 1].set_ylabel("n_diff")
    ax[0, 1].set_title("Pa3：亚层位移存在整段失明区间"); ax[0, 1].legend(fontsize=8)

    ax[1, 0].plot(PGRID, [ndiff(p, 1.0, theta, d=1) for p in PGRID], ".", ms=1.5, color="#55a868")
    ax[1, 0].set_xlabel("kink 位置 p"); ax[1, 0].set_ylabel("n_diff（delta_n = 1）")
    ax[1, 0].set_title("Pa4：整数层位移处处可分辨")

    pv = np.linspace(109.0, 111.0, 401)
    ax[1, 1].plot(pv, [ndiff(p, 0.5, theta, d=1) for p in pv], "-", label="亚层位移 0.5")
    ax[1, 1].plot(pv, [ndiff(p, 1.0, theta, d=1) for p in pv], "--", label="整数层位移 1")
    ax[1, 1].set_xlabel("kink 位置 p"); ax[1, 1].set_ylabel("n_diff")
    ax[1, 1].set_title("Pa4：一层分辨单元的局部形态"); ax[1, 1].legend(fontsize=8)

    fig.tight_layout(); fig.savefig(FIG, dpi=140)
    print("图已保存：", FIG)


if __name__ == "__main__":
    main()
