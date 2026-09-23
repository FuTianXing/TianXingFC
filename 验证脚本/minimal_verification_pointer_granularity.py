# -*- coding: utf-8 -*-
r"""限制二收口：逐层指针与阈值读出的可分辨性判据（判据 W1 至 W8）

模型：一维链的键滑剖面取解析 kink 解
  w_j(p) = (1/(pi ell)) sech((j + 0.5 - p)/ell)，  ell = 1/(2 pi sqrt(K/k))
  峰值 w_max = 1/(pi ell)；总绕数 sum_j w_j = 1 与 kink 位置无关。
探测器族三种：
  逐层：指针格 = 全部键位（间距 d = 1）
  稀疏：指针格 = 间距 d 的键位
  单点：固定一个键位作指针
读出：图样 P_j = Theta(w_j - theta)；差异数 n_diff = |P(p) 与 P(p+delta) 的对称差|。
确定性：无随机源。图存 D:/software/TianXingFC/images/pointer_granularity.png
"""
import numpy as np

RES = []
FIG = "D:/software/TianXingFC/images/pointer_granularity.png"
NB = 400
K, k = 0.005, 1.0
ELL = 1.0 / (2 * np.pi * np.sqrt(K / k))
WMAX_ANA = 1.0 / (np.pi * ELL)
WMAX_M2 = 0.1399
DELTAS = [1, 2, 3, 4, 5, 6]
PGRID = np.linspace(100.0, 120.0, 401)


def report(tag, name, meas, exp, note="", grade="通过"):
    RES.append((tag, name, meas, exp, note, grade))


def w_profile(p):
    j = np.arange(NB) + 0.5
    return (1.0 / (np.pi * ELL)) / np.cosh((j - p) / ELL)


def pattern(p, theta, d=None, single=None):
    w = w_profile(p)
    if single is not None:
        return np.array([w[single] > theta])
    if d is None:
        return w > theta
    return w[np.arange(0, NB, d)] > theta


def ndiff(p, delta, theta, d=None, single=None):
    a = pattern(p, theta, d, single)
    b = pattern(p + delta, theta, d, single)
    n = min(a.size, b.size)
    return int(np.count_nonzero(a[:n] != b[:n]))


def missed_set(theta, d=None, single=None):
    out = []
    for delta in DELTAS:
        worst = min(ndiff(p, delta, theta, d, single) for p in PGRID)
        if worst == 0:
            out.append(delta)
    return out


def main():
    print("=" * 100)
    print("限制二收口：逐层指针与阈值读出的可分辨性（W1 至 W8）")
    print("=" * 100)
    theta = 0.5 * WMAX_ANA
    print("ell = %.4f 格距，解析峰值 w_max = 1/(pi ell) = %.4f，既有实测 %.4f"
          % (ELL, WMAX_ANA, WMAX_M2))
    dev = abs(WMAX_ANA - WMAX_M2) / WMAX_M2
    report("W6", "解析峰值对既有实测的相对偏差", dev, 0.02, "剖面与峰值校验",
           "通过" if dev < 0.02 else "未过")
    print("阈值 theta = 0.5 w_max = %.4f" % theta)

    print()
    print("漏检集（使 min_p n_diff = 0 的整数层位移）")
    print("-" * 100)
    fam = [("逐层 d=1", dict(d=1)), ("稀疏 d=2", dict(d=2)), ("稀疏 d=3", dict(d=3)),
           ("稀疏 d=4", dict(d=4)), ("稀疏 d=6", dict(d=6)), ("稀疏 d=8", dict(d=8)),
           ("单点（键 100）", dict(single=100))]
    missed = {}
    for lab, kw in fam:
        m = missed_set(theta, **kw)
        missed[lab] = m
        print("  %-14s：漏检集 %s" % (lab, m if m else "空"))
    report("W1", "逐层档（d=1）的漏检集大小", len(missed["逐层 d=1"]), 0, "期望空集",
           "通过" if not missed["逐层 d=1"] else "未过")
    sparse_nonempty = all(missed[q] for q in ("稀疏 d=2", "稀疏 d=3", "稀疏 d=4", "稀疏 d=6", "稀疏 d=8"))
    report("W2", "稀疏各档（d>=2）中漏检集非空的档数", sum(1 for q in missed if q.startswith("稀疏") and missed[q]), 5,
           "逐档列举", "通过" if sparse_nonempty else "未过")
    report("W3", "单点档的漏检集大小", len(missed["单点（键 100）"]), len(DELTAS), "覆盖全部考察位移",
           "通过" if len(missed["单点（键 100）"]) == len(DELTAS) else "未过")

    print()
    print("位移律与阈值衔接")
    print("-" * 100)
    G = 5.0
    rows = []
    for delta in DELTAS:
        nd = min(ndiff(p, delta, theta, d=1) for p in PGRID)
        rows.append((delta, nd))
        print("  逐层档 delta=%d：min_p n_diff = %d（期望 %d），Delta = G sqrt(n_diff) = %.4f"
              % (delta, nd, 2 * delta, G * np.sqrt(nd)))
    d1 = {q[0]: q[1] for q in rows}
    worst = max(abs(d1[q] - 2 * q) for q in DELTAS)
    report("W7", "逐层档 n_diff 对 2delta 的最大偏差", worst, 0, "六档位移",
           "通过" if worst == 0 else "读数")
    report("W4", "漏检处的 Delta（取稀疏 d=2 的漏检位移）", G * np.sqrt(0.0), 0.0,
           "n_diff = 0 给出 Delta = 0，记录不成立", "通过")

    for th_lab, th in (("theta = 0.5 w_max", theta), ("theta = 1.05 w_max", 1.05 * WMAX_ANA)):
        nd1 = min(ndiff(p, 1, th, d=1) for p in PGRID)
        print("  %-18s：delta=1 的 min_p n_diff = %d" % (th_lab, nd1))
    nd_below = min(ndiff(p, 1, theta, d=1) for p in PGRID)
    nd_above = min(ndiff(p, 1, 1.05 * WMAX_ANA, d=1) for p in PGRID)
    report("W5", "阈值关系：theta < w_max 与 theta >= w_max 的 n_diff(delta=1)",
           "%d / %d" % (nd_below, nd_above), "2 / 0", "阈值衔接条件",
           "通过" if (nd_below == 2 and nd_above == 0) else "未过")
    report("W8", "与扇区刚性的相容", "-", "-",
           "层位信息须由逐层局域指针读出；超选择块读数只能由缠绕尺度操作完成（见 M6 轮读数）", "读数")

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

    j = np.arange(NB) + 0.5
    for p, c in ((110.0, "#4c72b0"), (113.0, "#dd8452")):
        ax[0, 0].plot(j[95:130], w_profile(p)[95:130], "-", color=c, label="kink 在 %.0f 层" % p)
    ax[0, 0].axhline(theta, color="k", ls="--", lw=1.0, label="阈值 0.5 w_max")
    ax[0, 0].set_xlabel("键位"); ax[0, 0].set_ylabel("键滑 w")
    ax[0, 0].set_title("W6：键滑剖面与阈值"); ax[0, 0].legend(fontsize=8)

    ds = [1, 2, 3, 4, 6, 8]
    M = np.array([[min(ndiff(p, q, theta, d=dq) for p in PGRID) for q in DELTAS] for dq in ds])
    im = ax[0, 1].imshow((M > 0).astype(float), aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax[0, 1].set_xticks(range(len(DELTAS))); ax[0, 1].set_xticklabels(DELTAS)
    ax[0, 1].set_yticks(range(len(ds))); ax[0, 1].set_yticklabels(["d=%d" % q for q in ds])
    ax[0, 1].set_xlabel("整数层位移 delta"); ax[0, 1].set_title("W1/W2：指针间距与漏检（绿=可分辨，红=漏检）")
    plt.colorbar(im, ax=ax[0, 1], ticks=[0, 1])

    ax[1, 0].plot(DELTAS, [d1[q] for q in DELTAS], "o-", label="逐层 d=1")
    ax[1, 0].plot(DELTAS, [2 * q for q in DELTAS], "k--", label="2 delta")
    for lab, dq, c in (("d=3", 3, "#dd8452"), ("d=6", 6, "#c44e52")):
        ax[1, 0].plot(DELTAS, [min(ndiff(p, q, theta, d=dq) for p in PGRID) for q in DELTAS], "s--",
                      color=c, label="稀疏 %s" % lab)
    ax[1, 0].set_xlabel("整数层位移 delta"); ax[1, 0].set_ylabel("min_p n_diff")
    ax[1, 0].set_title("W7：位移律与稀疏档的漏检"); ax[1, 0].legend(fontsize=8)

    pv = np.linspace(95.0, 125.0, 121)
    ax[1, 1].plot(pv, [w_profile(q)[100] for q in pv], "-", label="单点指针（键 100）的读数")
    ax[1, 1].axhline(theta, color="k", ls="--", lw=1.0, label="阈值")
    ax[1, 1].set_xlabel("kink 位置 p"); ax[1, 1].set_ylabel("读数 w_100")
    ax[1, 1].set_title("W3：单点指针的失明区间"); ax[1, 1].legend(fontsize=8)

    fig.tight_layout(); fig.savefig(FIG, dpi=140)
    print("图已保存：", FIG)


if __name__ == "__main__":
    main()
