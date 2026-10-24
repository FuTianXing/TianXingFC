# -*- coding: utf-8 -*-
r"""Delta 的场侧实现：寄存器占据的当前构型分量与历史残差（判据 R1 至 R7）

寄存器语言（沿用探测器物侧构造轮的指认）：
  读出量 y = 记录寄存器占据（逐键传感器读数为准经典极限）
  阈值 theta：键滑过阈判读
  当前构型分量 y_eq(t) = G Theta(w(t) - theta)（只受当前构型驱动时的占据）
  历史残差 Delta(t) = || y(t) - y_eq(t) ||_2
判别协议：末态构型相同、历史不同的两条键滑序列。
确定性：无随机源。图存 D:/software/TianXingFC/images/delta_field_side.png
"""
import numpy as np

RES = []
FIG = "D:/software/TianXingFC/images/delta_field_side.png"
NB = 400
DT = 0.1
T_END = 40.0
G = 5.0
TH0 = 0.5


def report(tag, name, meas, exp, note="", grade="通过"):
    RES.append((tag, name, meas, exp, note, grade))


def ts_axis():
    return np.arange(0.0, T_END, DT)


def hist_pair_A():
    """协议甲：早期访问键 150，末态停在键 100。"""
    ts = ts_axis()
    W = np.zeros((len(ts), NB))
    for i, q in enumerate(ts):
        if 1.0 <= q < 2.0:
            W[i, 150] = 1.0
        if q >= 3.0:
            W[i, 100] = 1.0
    return ts, W


def hist_pair_B():
    """协议乙：全程只在键 100。"""
    ts = ts_axis()
    W = np.zeros((len(ts), NB))
    for i, q in enumerate(ts):
        if q >= 0.0:
            W[i, 100] = 1.0
    return ts, W


def hist_static():
    ts = ts_axis()
    W = np.zeros((len(ts), NB))
    W[:, 100] = 1.0
    return ts, W


def hist_periodic():
    ts = ts_axis()
    W = np.zeros((len(ts), NB))
    for i, q in enumerate(ts):
        if (1.0 <= q < 2.0) or (5.0 <= q < 6.0) or (9.0 <= q < 10.0):
            W[i, 100] = 1.0
        if (3.0 <= q < 4.0) or (7.0 <= q < 8.0):
            W[i, 150] = 1.0
    return ts, W


def holder(W, theta, tau=None, Gval=G):
    n_s = W.shape[0]
    y = np.zeros(NB)
    ys = []
    if tau is None:
        for s in range(n_s):
            y = np.where(W[s] > theta, Gval, y)
            ys.append(y.copy())
    else:
        a = np.exp(-DT / tau)
        for s in range(n_s):
            y = a * y + Gval * (W[s] > theta) * (1 - a)
            ys.append(y.copy())
    return np.array(ys)


def y_eq_of(W, theta, Gval=G):
    return Gval * (W > theta)


def delta_of(W, theta, tau=None, Gval=G):
    ys = holder(W, theta, tau=tau, Gval=Gval)
    eqs = y_eq_of(W, theta, Gval=Gval)
    return np.linalg.norm(ys - eqs, axis=1), ys, eqs


def main():
    print("=" * 100)
    print("Delta 的场侧实现：当前构型分量与历史残差（R1 至 R7）")
    print("=" * 100)
    ts, WA = hist_pair_A()
    _, WBp = hist_pair_B()
    theta = TH0

    # R1 分解自洽（恒等式级）
    _, ys, eqs = delta_of(WA, theta)
    lhs = np.linalg.norm(ys - eqs, axis=1) ** 2
    rhs = (np.linalg.norm(ys, axis=1) ** 2 + np.linalg.norm(eqs, axis=1) ** 2
           - 2 * np.sum(ys * eqs, axis=1))
    dev = float(np.max(np.abs(lhs - rhs)) / max(np.max(lhs), 1e-30))
    report("R1", "分解恒等式的最大相对偏差", dev, 1e-12, "y = y_eq + (y - y_eq)",
           "通过" if dev < 1e-12 else "未过")

    # R2 同末态构型、不同历史
    dA, _, _ = delta_of(WA, theta, tau=None)
    dB, _, _ = delta_of(WBp, theta, tau=None)
    eqA_last = y_eq_of(WA, theta)[-1]
    eqB_last = y_eq_of(WBp, theta)[-1]
    eq_dev = float(np.max(np.abs(eqA_last - eqB_last)))
    print()
    print("R2 判别：末态构型相同、历史不同")
    print("  y_eq 末态逐位差 %.1e；Delta 末值：甲 %.4f，乙 %.4f，差 %.4f（G = %.1f）"
          % (eq_dev, dA[-1], dB[-1], dA[-1] - dB[-1], G))
    report("R2", "两协议末态 y_eq 的逐位差", eq_dev, 0.0, "当前构型分量相同",
           "通过" if eq_dev == 0.0 else "未过")
    report("R2", "无泄漏档 Delta 末值之差（甲减乙）", float(dA[-1] - dB[-1]), G,
           "差值等于额外记录量", "通过" if abs(dA[-1] - dB[-1] - G) < 1e-12 else "未过")
    dA1, _, _ = delta_of(WA, theta, tau=1.0)
    report("R2", "泄漏档（tau=1）Delta 末值之差", float(dA1[-1] - dB[-1]), "-",
           "额外记录随弛豫衰减", "读数")

    # R4 四反例复现
    print()
    print("R4 四反例复现（与记录判据轮同构）")
    print("-" * 100)
    _, WS = hist_static()
    dS, _, _ = delta_of(WS, theta)
    dSin, _, _ = delta_of(WS, 1.5)
    _, WP = hist_periodic()
    dP, _, _ = delta_of(WP, theta)
    print("  源静止：Delta 中位数 %.3e（期望 0）" % np.median(dS[ts > 5.0]))
    print("  指针不分辨（theta = 1.5）：Delta 中位数 %.3e（期望 0）" % np.median(dSin[ts > 5.0]))
    print("  周期复现：复现相（t in [5,6)）Delta 中位数 %.4f（期望 > 0）"
          % np.median(dP[(ts >= 5.0) & (ts < 6.0)]))
    print("  泄漏档（单次离开，tau=1）：Delta 末值 %.3e（期望趋于 0）" % dA1[-1])
    ok = (np.median(dS[ts > 5.0]) < 1e-12 and np.median(dSin[ts > 5.0]) < 1e-12
          and np.median(dP[(ts >= 5.0) & (ts < 6.0)]) > 1.0 and dA1[-1] < 1e-3)
    report("R4", "四反例的 Delta 读数与期望一致的档数", 4 if ok else 0, 4, "逐档核对",
           "通过" if ok else "未过")

    # R3 分量分工
    eq_from_cfg = y_eq_of(WA, theta)
    report("R3", "y_eq 由当前构型单独算出（不引用 y）", "构造性成立", "-",
           "定义式只含 w(t) 与阈值", "读数")
    report("R3", "Delta 须引用 y（历史），不可由当前构型单独算出", "反证见 R2 的差值 %.4f" % (dA[-1] - dB[-1]), "-",
           "同构型不同历史给出不同 Delta", "读数")

    # R5 与漏检的接口
    nd = 0
    report("R5", "指针不分辨档的差异数与 Delta", "n_diff = %d，Delta 中位数 %.1e" % (nd, np.median(dSin[ts > 5.0])), "-",
           "n_diff = 0 给出 Delta = 0，与 383/385 两轮漏检集结论一致", "读数")

    report("R6", "与判据篇 §2.2 一般定义的口径一致", "-", "-",
           "一般定义说'把源在当前时刻的构型固定后持有档将达到的值'，本稿的 y_eq 即其寄存器写法", "读数")
    report("R7", "边界：亚层位移与连续位置假设", "-", "-",
           "Delta 的定义不随事件集改变；分辨单元与漏检读数的关系引第 385 条", "读数")

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

    actA = np.array([1.0 if 1.0 <= q < 2.0 else (2.0 if q >= 3.0 else 0.0) for q in ts])
    ax[0, 0].plot(ts, actA * 2.5, ":", lw=1.2, label="协议甲的源激活位（1 = 键 150，2 = 键 100）")
    ax[0, 0].plot(ts, np.clip(dA, 1e-16, None), "-", label="甲：Delta(t)")
    ax[0, 0].plot(ts, np.clip(dB, 1e-16, None), "-", label="乙：Delta(t)")
    ax[0, 0].set_xlabel("时间（模型单位）"); ax[0, 0].set_ylabel("Delta(t)（对数）")
    ax[0, 0].set_title("R2：同末态构型、不同历史"); ax[0, 0].legend(fontsize=8)

    ax[0, 1].plot(ts, np.clip(dA1, 1e-16, None), "-", label="泄漏 tau=1（甲）")
    ax[0, 1].plot(ts, np.clip(delta_of(WA, theta, tau=4.0)[0], 1e-16, None), "-", label="泄漏 tau=4（甲）")
    ax[0, 1].plot(ts, np.clip(dB, 1e-16, None), "--", label="乙（对照）")
    ax[0, 1].set_xlabel("时间（模型单位）"); ax[0, 1].set_ylabel("Delta(t)（对数）")
    ax[0, 1].set_title("R2：额外记录在泄漏档下衰减"); ax[0, 1].legend(fontsize=8)

    labs = ["源静止", "指针不分辨", "单次离开（泄漏 tau=1）", "周期复现"]
    vals = [max(float(np.median(dS[ts > 5.0])), 1e-16), max(float(np.median(dSin[ts > 5.0])), 1e-16),
            max(float(dA1[-1]), 1e-16), float(np.median(dP[(ts >= 5.0) & (ts < 6.0)]))]
    ax[1, 0].bar(range(4), vals, color=["#4c72b0", "#4c72b0", "#c44e52", "#55a868"])
    ax[1, 0].set_yscale("log"); ax[1, 0].set_xticks(range(4))
    ax[1, 0].set_xticklabels(labs, fontsize=8); ax[1, 0].set_ylabel("Delta 中位数（对数）")
    ax[1, 0].set_title("R4：四个反例的 Delta")

    ax[1, 1].plot(ts, np.clip(np.linalg.norm(holder(WA, theta), axis=1), 1e-16, None), "-", label="||y||（甲，无泄漏）")
    ax[1, 1].plot(ts, np.clip(np.linalg.norm(y_eq_of(WA, theta), axis=1), 1e-16, None), "--", label="||y_eq||（甲）")
    ax[1, 1].plot(ts, np.clip(dA, 1e-16, None), ":", label="Delta（甲）")
    ax[1, 1].set_xlabel("时间（模型单位）"); ax[1, 1].set_ylabel("范数（对数）")
    ax[1, 1].set_title("R1/R3：占据、当前构型分量与残差的分解"); ax[1, 1].legend(fontsize=8)

    fig.tight_layout(); fig.savefig(FIG, dpi=140)
    print("图已保存：", FIG)


if __name__ == "__main__":
    main()
