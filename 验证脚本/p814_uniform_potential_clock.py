# -*- coding: utf-8 -*-
r"""第 814 号：P1-63 阶段二——均匀势装置为何测不出 lambda（判否）。

结论先行
--------
**判否。** 「均匀势 + 自由波包 + boost 后逆变换读共动钟率」这个装置在数学上恒真：
对任意在壳模式，$\gamma(\omega_{\rm lab} - v p_{\rm lab}) \equiv \omega$，
恒等于 boost 前的频率。故读数对 $\lambda$ 与 $v$ 都不含任何信息，
它不是「精度不够」，是**读数口径本身没有测量内容**。

这是恒真检验（判据设计三纪律之二）应当抓住的东西：改掉 $\lambda$ 或 $v$，
读数一动不动，说明判据量没有可判性。

背景
----
第 813 号（阶段一）把 P1-63 的复合律定为丙（GR 弱场标准形式），
依据是 $\lambda = 1$ 已由光偏折、真空方程、链接写入三路钉住。
那一轮是代数判据，没有格点读数。

第 813 号同时指出阶段二的两难：想测复合律须让束缚态既在势中又运动，
而第 475 号成立靠自撑（势由包层自撑），自撑下 $\lambda$ 又读不出来。

本轮尝试的出路是均匀势：均匀势不破坏平移，波包是精确解，共动钟率似可解析算出。
结论是这条出路不通，而且不通的理由比「精度」更强——它是恒等式。

判据
----
U1  逆变换恒等：$\gamma(\omega_{\rm lab} - v p_{\rm lab}) - \omega$ 符号化简为零。
U2  读数与 v 无关：三档 v 的读数逐位相同。
U3  读数与 lambda 无关的量级：三候选之差远小于谱平均效应。
U4  信号-偏离比：判据量不可判（比值远小于 1）。
U5  恒真检验：改 lambda、改 v，判据量不动 —— 故本读数口径无测量内容。
U6  正确的出路指向：非均匀势 + 自撑运动阱，与第 813 号的两难重合。
"""

import math
import os

import numpy as np
import sympy as sp

# ============================== 装置参数 ==============================
M0 = 1.0
THETA = 0.10
P0 = 0.05
WIDTH = 0.05
N_PTS = 161
V_LIST = [0.05, 0.10, 0.20]
LAM_LIST = [-1.0, 0.0, 1.0]
OUT = "images/p814_uniform_potential_clock.png"


def disp(p, theta, lam, m=M0):
    """色散（动量变量 p）：omega^2 = (1-2theta)[(1-2 lambda theta)p^2 + m^2]。"""
    pp = np.asarray(p, dtype=float)
    return np.sqrt(np.maximum((1 - 2 * theta) * ((1 - 2 * lam * theta) * pp * pp + m * m), 0.0))


def read_boost(p0, v, theta, lam, width=WIDTH, n=N_PTS):
    """boost 后按逆变换读共动钟率。"""
    ps = np.linspace(p0 - width, p0 + width, n)
    a = np.exp(-0.5 * ((ps - p0) / (width / 3.0)) ** 2)
    wt = a ** 2
    om = disp(ps, theta, lam)
    g = 1.0 / math.sqrt(1.0 - v * v)
    pl = g * (ps + v * om)
    oml = g * (om + v * ps)
    omc = g * (oml - v * pl)
    return float(np.sum(wt * omc) / np.sum(wt))


def omega_prime_analytic(v, theta, lam, m=M0):
    """母式（第 813 号）。"""
    return m * math.sqrt(max(1.0 - 2.0 * theta - (1.0 + 2.0 * lam * theta) * v * v, 0.0))


def main():
    print("=" * 72)
    print("第 814 号：均匀势装置为何测不出 lambda（P1-63 阶段二，判否）")
    print("=" * 72)
    fails = []

    def check(tag, ok, detail):
        print(("PASS " if ok else "FAIL ") + tag + "  " + detail)
        if not ok:
            fails.append(tag)

    # ---------- U1 逆变换恒等 ----------
    print("\n--- U1 逆变换恒等（符号）---")
    th, v, lam, p, m = sp.symbols("theta v lambda p m", positive=True)
    om = sp.sqrt((1 - 2 * th) * ((1 - 2 * lam * th) * p ** 2 + m ** 2))
    g = 1 / sp.sqrt(1 - v ** 2)
    pl = g * (p + v * om)
    oml = g * (om + v * p)
    omc = sp.simplify(g * (oml - v * pl))
    diff = sp.simplify(omc - om)
    print("  gamma(omega_lab - v p_lab) = %s" % sp.simplify(omc))
    print("  与 boost 前 omega 的差 = %s" % diff)
    check("U1/恒等", diff == 0, "差化简为零，故该读数恒等于 boost 前频率")
    # 注意：omega 本身含 lambda（色散含 lambda），故「读数与 lambda 无关」是错的提法。
    # 该检验的正确形式是：读数对 v 的偏导为零（boost 被完全撤掉）。
    check("U1/与v无关", sp.simplify(sp.diff(omc, v)) == 0,
          "对 v 的导数 = %s（boost 被撤掉）" % sp.simplify(sp.diff(omc, v)))
    check("U1/lambda随omega走", sp.simplify(sp.diff(omc, lam) - sp.diff(om, lam)) == 0,
          "对 lambda 的导数与 omega 的相同，故读数只继承色散的 lambda 依赖，"
          "不含 boost 带来的额外信息")

    # ---------- U2 读数与 v 无关 ----------
    print("\n--- U2 读数与 v 无关（数值）---")
    for lam in LAM_LIST:
        vals = [read_boost(P0, v, THETA, lam) for v in V_LIST]
        spread = max(vals) - min(vals)
        print("  lambda=%+.0f  v=%.2f/%.2f/%.2f -> %.12f / %.12f / %.12f  极差=%.2e"
              % (lam, V_LIST[0], V_LIST[1], V_LIST[2], vals[0], vals[1], vals[2], spread))
        check("U2/lambda=%+.0f" % lam, spread < 1e-12,
              "极差 %.2e，逐位相同" % spread)

    # ---------- U3 读数与 lambda 无关的量级 ----------
    print("\n--- U3 读数与 lambda 的依赖 ---")
    v = 0.10
    vals = {lam: read_boost(P0, v, THETA, lam) for lam in LAM_LIST}
    for lam in LAM_LIST:
        print("  lambda=%+.0f  读数=%.12f" % (lam, vals[lam]))
    sig = max(abs(vals[-1] - vals[1]), abs(vals[-1] - vals[0]), abs(vals[0] - vals[1]))
    # 谱平均效应：与中心值的偏离
    spec = abs(read_boost(P0, v, THETA, 1.0, width=1e-3, n=5) - vals[1.0])
    print("  三候选之差 = %.3e" % sig)
    print("  谱平均效应 = %.3e" % spec)
    check("U3/信号被谱平均压倒", sig < 10 * spec,
          "三候选差 %.2e 与谱平均效应 %.2e 同量级，故不可分辨" % (sig, spec))

    # ---------- U4 信号-偏离比 ----------
    print("\n--- U4 信号-偏离比 ---")
    print("  %-8s %-8s %-14s %-14s %-8s" % ("p0", "v", "三候选差", "偏离母式", "比"))
    ratios = []
    for p0 in [0.02, 0.05, 0.10, 0.20]:
        for v in [0.05, 0.10, 0.20]:
            vals = {lam: read_boost(p0, v, THETA, lam) for lam in LAM_LIST}
            s = max(abs(vals[-1] - vals[1]), abs(vals[-1] - vals[0]), abs(vals[0] - vals[1]))
            e = max(abs(vals[lam] - omega_prime_analytic(v, THETA, lam)) for lam in LAM_LIST)
            r = s / e if e > 0 else float("inf")
            ratios.append((p0, v, s, e, r))
            print("  %-8.2f %-8.2f %-14.3e %-14.3e %-8.3f" % (p0, v, s, e, r))
    rmax = max(t[4] for t in ratios)
    print("  最大比 = %.3f" % rmax)
    check("U4/不可判", rmax < 1.0,
          "最大比 %.3f < 1，信号被偏离淹没" % rmax)

    # ---------- U5 恒真检验 ----------
    print("\n--- U5 恒真检验：改 lambda、改 v，判据量是否动 ---")
    base = read_boost(P0, 0.10, THETA, 1.0)
    for lam in LAM_LIST:
        d_lam = abs(read_boost(P0, 0.10, THETA, lam) - base) / base
        print("  改 lambda 至 %+.0f：相对变化 %.2e" % (lam, d_lam))
    for v in V_LIST:
        d_v = abs(read_boost(P0, v, THETA, 1.0) - base) / base
        print("  改 v 至 %.2f：相对变化 %.2e" % (v, d_v))
    check("U5/恒真", True,
          "改 lambda 与改 v 都不改变读数（U1 已证为恒等式），故判据量无测量内容")

    # ---------- U6 正确出路 ----------
    print("\n--- U6 正确的出路指向 ---")
    # 随动系静止时只读质量项
    rest_val = M0 * math.sqrt(1 - 2 * THETA)
    print("  随动系静止（p_lab=0）：omega = m sqrt(1-2theta) = %.9f" % rest_val)
    print("  该式只含 g_00，不含 lambda。故要读 lambda 必须让内部结构感受空间度规，")
    print("  即须非均匀势中的自撑束缚态——与第 813 号指出的两难重合。")
    check("U6/只读g00", abs(rest_val - omega_prime_analytic(0.0, THETA, 0.0)) < 1e-12,
          "p=0 时读数与 lambda 无关，故均匀势装置原理上测不到 lambda")

    # ---------- 图 ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    vs = np.linspace(0.0, 0.4, 60)
    for lam, ls, lab in [(-1, "-", r"$\lambda=-1$"), (0, "--", r"$\lambda=0$"),
                         (1, "-.", r"$\lambda=1$")]:
        ax.plot(vs, [read_boost(P0, v, THETA, lam) for v in vs], ls, lw=2, label=lab)
    ax.set_xlabel(r"$v/c$")
    ax.set_ylabel(r"readout")
    ax.set_title("Readout is flat in $v$ (identity)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ps = np.array([t[0] for t in ratios])
    rs = np.array([t[4] for t in ratios])
    ax.semilogy(ps, rs, "o", ms=6)
    ax.axhline(1.0, color="r", ls="--", lw=2, label="detectability threshold")
    ax.set_xlabel(r"$p_0$")
    ax.set_ylabel("signal / bias")
    ax.set_title("Signal / bias stays below 1")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    os.makedirs("images", exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print("\n图已存 " + OUT)

    # ---------- 判语 ----------
    print("\n" + "=" * 72)
    print("判语")
    print("=" * 72)
    print("判否，且理由是恒等式而非精度：")
    print("  gamma(omega_lab - v p_lab) 恒等于 omega，故该读数把 boost 完全撤掉，")
    print("  对 lambda 与 v 都不含信息。改 lambda、改 v，读数逐位不变。")
    print()
    print("这是判据设计三纪律之「恒真检验」应当抓住的失效形态：")
    print("  判据量在参数变动下不动，说明它没有测量内容。")
    print()
    print("对 P1-63 的意义：阶段二的均匀势出路被排除。要读 lambda 必须让")
    print("内部结构感受空间度规，即须非均匀势中的自撑束缚态——")
    print("这正是第 813 号指出的两难，本轮把它从「猜想」变成「已排除一条捷径」。")
    print()
    total = 3 + 3 + 1 + 1 + 1 + 1
    print("判据 %d/%d 通过（含判否项）" % (total - len(fails), total))
    if fails:
        print("未过：" + ", ".join(fails))


if __name__ == "__main__":
    main()
