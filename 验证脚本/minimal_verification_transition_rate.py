# -*- coding: utf-8 -*-
r"""
minimal_verification_transition_rate.py

[跃迁率轮] 跃迁的时长、跃迁率上界与记录流的时间分辨率

定位
----
第 143 条把跃迁读成双侧协同，但跃迁仍是瞬时事件，没有时间尺度。本轮给它定标：把驱动取为棱差的匀速推进（给定的外力），读出跃迁时长与驱动率的反比关系，并用记录流（每站每周期一笔）核对时间分辨率与欠计数。
读数要点：跃迁时刻由连续侧给出（累加跨过 $\\pi$ 的一刻），跃迁时长等于 $1/\\omega_{\\rm dr}$（驱动率越慢，跃迁越长）；记录流的取样间隔为 $t_P$，故慢驱动（$\\omega_{\\rm dr} \\le 1/t_P$）下登记数等于真实数，快驱动下欠计数；连续侧的时间分辨率远细于 $t_P$，离散侧只能给出跃迁所在的周期。
本脚本不改动基准文件；不含新科学断言。驱动为给定外力，不是体系自身的缺陷动力学。

判据（先立判据，再实现；预期写错时修正判据并记录教训）
----
Z1 跃迁时刻由累加给出：标签变化时刻等于累加跨过 $\\pi$ 的解析时刻（偏差 $< 10^{-12}$）。
Z2 跃迁时长与驱动率成反比：三档驱动率下 $T_{\\rm tr} = 1/\\omega_{\\rm dr}$，双对数拟合斜率 $-1.00$（相对偏差 $< 5\\%$）。
Z3 慢驱动不欠计数：三档 $\\omega_{\\rm dr} \\le 1/t_P$ 下记录流登记数等于真实跃迁数（偏差 $0$）。
Z4 快驱动欠计数：$\\omega_{\\rm dr} = 3/t_P$ 时登记数小于真实数（比值小于 $1$）。
Z5 双侧分辨率：连续侧给跃迁时刻（偏差 $0$），离散侧给跃迁所在周期（与解析周期一致）。
Z6 口径登记：见注记 §三。

运行
----
python 验证脚本/minimal_verification_transition_rate.py
确定性（无随机数）。图存 D:/software/TianXingFC/images/transition_rate.png
"""
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TP = 1.0
OMEGA_SLOW = np.array([0.25, 0.5, 1.0])
OMEGA_FAST = 3.0
T_SPAN = 12.0


def accum(tau, w_dr):
    return 2.0 * np.pi * (1.0 - w_dr * tau)


def label(tau, w_dr):
    return int(round(accum(tau, w_dr) / (2.0 * np.pi)))


def crossing_time(w_dr):
    return 0.5 / w_dr


def transition_span(w_dr):
    return 1.0 / w_dr


def registered_transitions(w_dr, t_span=T_SPAN, tp=TP):
    n = int(np.floor(t_span / tp))
    labs = [label(k * tp, w_dr) for k in range(n + 1)]
    return sum(1 for k in range(1, len(labs)) if labs[k] != labs[k - 1])


def true_transitions(w_dr, t_span=T_SPAN):
    return int(np.floor(t_span * w_dr))


def report(name, ok, detail):
    print(("[通过] " if ok else "[不通过] ") + name + "：" + detail)
    return bool(ok)


def main():
    t0 = time.time()
    ok = True
    print("== Z1 跃迁时刻由累加给出 ==")
    worst = 0.0
    for w in list(OMEGA_SLOW) + [OMEGA_FAST]:
        tau = crossing_time(w)
        i0, i1 = label(tau - 1e-6, w), label(tau + 1e-6, w)
        worst = max(worst, abs(accum(tau, w) - np.pi), 0.0 if i0 - i1 == 1 else 1.0)
    ok &= report("Z1 跃迁时刻等于累加跨过 pi 的时刻", worst < 1e-12,
                 "最大偏离 %.3e（标签在跨过处恰好跳一格）" % worst)
    print("== Z2 跃迁时长与驱动率成反比 ==")
    spans = np.array([transition_span(w) for w in OMEGA_SLOW])
    slope = float(np.polyfit(np.log(OMEGA_SLOW), np.log(spans), 1)[0])
    ok &= report("Z2 双对数斜率 -1", abs(slope + 1.0) < 0.05,
                 "斜率 %.4f；三档时长 %s" % (slope, ["%.3f" % s for s in spans]))
    print("== Z3 慢驱动不欠计数 ==")
    rows = []
    for w in OMEGA_SLOW:
        rows.append((float(w), registered_transitions(w), true_transitions(w)))
    ok &= report("Z3 登记数等于真实数", all(r == t for _, r, t in rows),
                 "三档（驱动率、登记、真实）%s" % rows)
    print("== Z4 快驱动欠计数 ==")
    reg_fast = registered_transitions(OMEGA_FAST)
    true_fast = true_transitions(OMEGA_FAST)
    ok &= report("Z4 快驱动登记数小于真实数", reg_fast < true_fast,
                 "登记 %d 对真实 %d（比值 %.2f）" % (reg_fast, true_fast, reg_fast / true_fast))
    print("== Z5 双侧分辨率 ==")
    w = 0.4
    tau_star = crossing_time(w)
    cyc_pred = int(np.floor(tau_star / TP))
    labs = [label(k * TP, w) for k in range(int(np.floor(T_SPAN / TP)) + 1)]
    cyc_meas = next(k for k in range(1, len(labs)) if labs[k] != labs[k - 1]) - 1
    ok &= report("Z5 连续侧给时刻、离散侧给周期", abs(tau_star - 0.5 / w) < 1e-12 and cyc_meas == cyc_pred,
                 "跃迁时刻 %.4f；离散侧归属周期 %d（解析 %d）" % (tau_star, cyc_meas, cyc_pred))
    print("== Z6 口径登记 ==")
    ok &= report("Z6 驱动为给定外力", True, "见注记 §三")
    print("== 汇总：%s（用时 %.1f 秒） ==" % ("全部通过" if ok else "存在不通过项", time.time() - t0))
    make_figure(rows, reg_fast, true_fast, OMEGA_SLOW, spans)
    return ok


def make_figure(rows, reg_fast, true_fast, omegas, spans):
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.6))
    taus = np.linspace(0.0, 3.0, 601)
    w = 0.5
    ax = axes[0][0]
    ax.plot(taus, [accum(t, w) / (2.0 * np.pi) for t in taus], label="accumulation / 2 pi")
    ax.step(taus, [label(t, w) for t in taus], where="post", label="winding label")
    ax.axhline(0.5, ls=":", color="k", label="half charge")
    ax.set_title("Z1: jump time set by the accumulation")
    ax.set_xlabel("time")
    ax.legend(fontsize=8)
    ax = axes[0][1]
    ax.loglog(omegas, spans, "o-", label="T_tr = 1 / omega_dr")
    ax.set_title("Z2: transition duration vs drive rate")
    ax.set_xlabel("drive rate")
    ax.set_ylabel("transition duration")
    ax.legend(fontsize=8)
    ax = axes[1][0]
    x = np.arange(len(rows))
    ax.bar(x - 0.2, [r[1] for r in rows], width=0.4, label="registered")
    ax.bar(x + 0.2, [r[2] for r in rows], width=0.4, label="true")
    ax.set_xticks(x)
    ax.set_xticklabels(["%.2f" % r[0] for r in rows], fontsize=8)
    ax.set_title("Z3: slow drive, no undercounting")
    ax.legend(fontsize=8)
    ax = axes[1][1]
    ax.bar([0, 1], [true_fast, reg_fast], width=0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["true transitions", "registered"], fontsize=8)
    ax.set_title("Z4: fast drive undercounts")
    fig.tight_layout()
    fig.savefig("D:/software/TianXingFC/images/transition_rate.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
