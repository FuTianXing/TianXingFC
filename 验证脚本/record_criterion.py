# -*- coding: utf-8 -*-
r"""记录判据：独立性的定义域、反例与判据级表述（判据 U1 至 U8）

受控键滑序列（不依赖 FK 积分，尺度归一：越阈键取 w = 1.0，指针阈值 theta = 0.5）：
  保持档三种：无泄漏 latch（越阈即置 G、不衰减）、泄漏型 tau、指针阈值放大档。
  当前构型平衡值 y_eq(t) = G Theta(w(t) - theta)；偏离 Delta(t) = || y(t) - y_eq(t) ||_2。
四档源形态：
  A 源静止：键 100 全程 = 1.0
  B 源单次离开：键 100 在 t in [1,2) = 1.0，其后为空
  C 源周期往返：键 100 在 [1,2) 与 [5,6) 与 [9,10)；键 150 在 [3,4) 与 [7,8)
  D 指针不分辨：与 A 同源，但 theta = 1.5（无键越阈）
确定性：无随机源。图存 D:/software/TianXingFC/images/record_criterion.png
"""
import numpy as np

RES = []
FIG = "D:/software/TianXingFC/images/record_criterion.png"
NB = 400
DT_S = 0.1
T_END = 40.0
G = 5.0
TH0 = 0.5


def report(tag, name, meas, exp, note="", grade="通过"):
    RES.append((tag, name, meas, exp, note, grade))


def W_static():
    ts = np.arange(0.0, T_END, DT_S)
    W = np.zeros((len(ts), NB))
    W[:, 100] = 1.0
    return ts, W


def W_single_leave():
    ts = np.arange(0.0, T_END, DT_S)
    W = np.zeros((len(ts), NB))
    for i, q in enumerate(ts):
        if 1.0 <= q < 2.0:
            W[i, 100] = 1.0
    return ts, W


def W_periodic():
    ts = np.arange(0.0, T_END, DT_S)
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
    ys, eqs = [], []
    if tau is None:
        for s in range(n_s):
            y = np.where(W[s] > theta, Gval, y)
            ys.append(y.copy()); eqs.append(Gval * (W[s] > theta))
    else:
        a = np.exp(-DT_S / tau)
        for s in range(n_s):
            y = a * y + Gval * (W[s] > theta) * (1 - a)
            ys.append(y.copy()); eqs.append(Gval * (W[s] > theta))
    return np.array(ys), np.array(eqs)


def Dev(W, theta, tau=None):
    ys, eqs = holder(W, theta, tau=tau)
    return np.linalg.norm(ys - eqs, axis=1), ys


def main():
    print("=" * 100)
    print("记录判据：独立性的定义域、反例与判据级表述（U1 至 U8）")
    print("=" * 100)
    ts, WA = W_static()
    _, WB = W_single_leave()
    _, WC = W_periodic()

    # U2 源静止档
    dA_lat, _ = Dev(WA, TH0, tau=None)
    dA_leak, _ = Dev(WA, TH0, tau=1.0)
    late = ts > 5.0
    report("U2", "源静止档：无泄漏 Delta 中位数", float(np.median(dA_lat[late])), 0.0,
           "记录不成立（无信息可记）", "通过" if np.median(dA_lat[late]) < 1e-12 else "未过")
    report("U2", "源静止档：泄漏档末值 Delta", float(dA_leak[-1]), 0.0,
           "同样归零（中位数含早期过渡，故取末值）",
           "通过" if dA_leak[-1] < 1e-9 else "未过")

    # U5 指针不分辨档
    dD_lat, _ = Dev(WA, 1.5, tau=None)
    report("U5", "指针不分辨档：Delta 中位数", float(np.median(dD_lat)), 0.0,
           "无键越阈，记录不成立", "通过" if np.median(dD_lat) < 1e-12 else "未过")
    dB_lat_probe, _ = Dev(WB, 1.5, tau=None)
    dB_lat_ref, _ = Dev(WB, TH0, tau=None)
    report("U5", "单次离开档下指针阈值 0.5 对 1.5 的 t>5 Delta 中位数对照",
           "%.4f / %.4f" % (np.median(dB_lat_ref[ts > 5.0]), np.median(dB_lat_probe[ts > 5.0])), "-",
           "可分辨档保留、不分辨档归零", "读数")

    # U3 源单次离开档
    dB_lat, yB_lat = Dev(WB, TH0, tau=None)
    dB_l1, _ = Dev(WB, TH0, tau=1.0)
    dB_l4, _ = Dev(WB, TH0, tau=4.0)
    report("U3", "单次离开档：t = 40 时 Delta（无泄漏 / tau=1 / tau=4）",
           "%.3f / %.3e / %.3e" % (dB_lat[-1], dB_l1[-1], dB_l4[-1]), "-",
           "无泄漏保留、泄漏归零", "读数")
    report("U3", "无泄漏对泄漏(tau=1) 的 t=40 保留比",
           float(dB_lat[-1] / max(dB_l1[-1], 1e-300)), "-", "主判据的分离度",
           "通过" if dB_lat[-1] > 1.0 and dB_l1[-1] < 1e-3 else "未过")

    # U6 泄漏档衰减率
    rates = {}
    for tau in (0.25, 1.0, 4.0):
        d, _ = Dev(WB, TH0, tau=tau)
        m = ts > 2.0
        p = np.polyfit(ts[m], np.log(np.clip(d[m], 1e-300, None)), 1)
        rates[tau] = float(p[0])
        print("  U6 tau=%4.2f：衰减率 %+.6f（期望 %+.6f）" % (tau, p[0], -1.0 / tau))
    dev = max(abs(rates[t] + 1.0 / t) * t for t in rates)
    report("U6", "泄漏档衰减率对 -1/tau 的最大相对偏差", dev, 0.05, "三档 tau",
           "通过" if dev < 0.05 else "未过")

    # U4 源周期往返档
    dC_lat, yC_lat = Dev(WC, TH0, tau=None)
    m1 = (ts >= 5.0) & (ts < 6.0)
    m2 = (ts >= 3.0) & (ts < 4.0)
    m0 = (ts >= 0.0) & (ts < 1.0)
    report("U4", "周期往返档：源复现相位（t in [5,6)）的 Delta 中位数",
           float(np.median(dC_lat[m1])), "-", "历史保留，不归零",
           "通过" if np.median(dC_lat[m1]) > 1.0 else "未过")
    report("U4", "周期往返档：首个离开相（t in [3,4)）的 Delta 中位数",
           float(np.median(dC_lat[m2])), "-", "p1 与 p2 两处记录同时在位", "读数")
    report("U4", "周期往返档：初始相（t in [0,1)）的 Delta 中位数",
           float(np.median(dC_lat[m0])), 0.0, "尚无记录", "通过" if np.median(dC_lat[m0]) < 1e-12 else "未过")
    report("U4", "复现相减初始相的 Delta 差",
           float(np.median(dC_lat[m1]) - np.median(dC_lat[m0])), "-",
           "初始相为零，故差值即复现相的记录量", "读数")

    # U7 四档极限与判据逐档一致
    print()
    print("U7 判据级表述的逐档核对：记录成立 <=> 源移开后 Delta 不衰减")
    print("-" * 100)
    table = [("A 源静止", dA_lat, "不成立"), ("B 单次离开（无泄漏）", dB_lat, "成立"),
             ("B 单次离开（泄漏 tau=1）", dB_l1, "不成立"), ("C 周期往返（无泄漏）", dC_lat, "成立"),
             ("D 指针不分辨", dD_lat, "不成立")]
    ok = True
    for lab, d, verdict in table:
        lim = float(np.median(d[ts > 5.0]))
        est = "成立" if lim > 1.0 else "不成立"
        agree = (est == verdict)
        ok = ok and agree
        print("  %-24s：t>5 的 Delta 中位数 %10.4e，判据判读「%s」，期望「%s」，%s"
              % (lab, lim, est, verdict, "一致" if agree else "不一致"))
    report("U7", "五档判读与判据的一致档数", 5 if ok else 0, 5, "逐档核对",
           "通过" if ok else "未过")
    report("U8", "三处结论的同源性", "-", "-",
           "kappa 条件式、指针分辨、持有时间皆由本判据派生", "读数")

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

    ax[0, 0].semilogy(ts, np.clip(dB_lat, 1e-16, None), "-", label="无泄漏 latch")
    ax[0, 0].semilogy(ts, np.clip(dB_l1, 1e-16, None), "-", label="泄漏 tau=1")
    ax[0, 0].semilogy(ts, np.clip(dB_l4, 1e-16, None), "-", label="泄漏 tau=4")
    ax[0, 0].axvline(2.0, color="gray", ls=":", lw=1.0)
    ax[0, 0].set_xlabel("时间（模型单位，虚线处源移开）"); ax[0, 0].set_ylabel("Delta(t)")
    ax[0, 0].set_title("U3/U6：单次离开档的主判据与泄漏失效"); ax[0, 0].legend(fontsize=8)

    ax[0, 1].semilogy(ts, np.clip(dA_lat, 1e-16, None), "-", label="无泄漏 latch")
    ax[0, 1].semilogy(ts, np.clip(dA_leak, 1e-16, None), "-", label="泄漏 tau=1")
    ax[0, 1].semilogy(ts, np.clip(dD_lat, 1e-16, None), "--", label="指针不分辨（theta=1.5）")
    ax[0, 1].set_xlabel("时间（模型单位）"); ax[0, 1].set_ylabel("Delta(t)")
    ax[0, 1].set_title("U2/U5：两个恒零档（源静止、指针不分辨）"); ax[0, 1].legend(fontsize=8)

    ax[1, 0].semilogy(ts, np.clip(dC_lat, 1e-16, None), "-", label="Delta(t)，无泄漏")
    act = np.zeros_like(ts)
    for i, q in enumerate(ts):
        if (1.0 <= q < 2.0) or (5.0 <= q < 6.0) or (9.0 <= q < 10.0):
            act[i] = 1.0
        if (3.0 <= q < 4.0) or (7.0 <= q < 8.0):
            act[i] = 2.0
    ax[1, 0].plot(ts, act * 5.0, ":", lw=1.2, label="源激活键位（1 = 键 100，2 = 键 150）")
    ax[1, 0].set_xlabel("时间（模型单位）"); ax[1, 0].set_ylabel("Delta(t)（对数）")
    ax[1, 0].set_title("U4：源周期往返，复现相上记录不归零"); ax[1, 0].legend(fontsize=8)

    labels = [q[0] for q in table]
    lims = [max(float(np.median(q[1][ts > 5.0])), 1e-16) for q in table]
    ax[1, 1].bar(range(len(labels)), lims, color=["#4c72b0", "#55a868", "#c44e52", "#55a868", "#c44e52"])
    ax[1, 1].set_yscale("log"); ax[1, 1].set_xticks(range(len(labels)))
    ax[1, 1].set_xticklabels(labels, rotation=18, ha="right", fontsize=8)
    ax[1, 1].axhline(1.0, color="gray", ls="--", lw=1.0, label="判据阈值（记录成立）")
    ax[1, 1].set_ylabel("t>5 的 Delta 中位数"); ax[1, 1].set_title("U7：五档判读")
    ax[1, 1].legend(fontsize=8)

    fig.tight_layout(); fig.savefig(FIG, dpi=140)
    print("图已保存：", FIG)


if __name__ == "__main__":
    main()
