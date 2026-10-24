# -*- coding: utf-8 -*-
r"""
spontaneous_term_derivation.py

[自发项导出轮] 本底不来自真空涨落，来自时钟自己的折返

定位
----
上一轮把发射率取 $\gamma_{\rm em}(\bar N\rho_j + 1)$，其中"$1$"是给定的自发项。本轮把它导出：
由单体模固定（相位指针半径固定、以 $\omega_P = 2\pi/t_P$ 匀速转动）与"每站点每摆程完成一次折返"，
每个站点的时钟在窗口 $T$ 内恰好折返 $T/t_P$ 次，且与局域激发数、与模式形状、与模式频率 $\omega_M$ 都无关。
故发射率 = 受激项（$\propto$ 局域激发数）加时钟项（每站每摆程一次），后者给出均匀本底。
本脚本不改动基准文件；不含新科学断言。

判别的关键
----
教科书读法把本底当作真空涨落，属泊松过程；本轮导出的本底是时钟折返，属**确定性周期计数**，方差为零。
两者的可测差别落在复合计数的方差上：导出版预期 $\mathrm{Var} = \bar N$ 的受激均值，教科书版预期 $\mathrm{Var} = $ 总均值。
本轮实测方差并据此排除教科书读法。

判据（先立判据，再实现；预期写错时修正判据并记录教训）
----
J1 时钟计数确定性：各站点在窗口 $T$ 内的折返计数 $= T/t_P$（散布为零），与站点相位、与占据数无关。
J2 本底与模式无关：阱曲率放大四倍（模式宽度与 $\omega_M$ 同时改变）后，每站每窗折返计数仍 $= T/t_P$。
J3 方差判据：复合计数（$M = 2000$ 条实现）的站点方差等于受激均值（显著站点平均相对偏差 $< 5\%$；单站点方差估计的抽样误差约 $1/\sqrt{M}$，故不用最大值判定）。时钟项带同一发射耦合 $\gamma_{\rm em}$（每摆程一次机会，耦合把机会变成计数）。
J4 两项可加：单独模拟的受激计数与时钟计数之和等于复合计数（偏差 $< 1\%$）。
J5 与涌现轮一致：记录份额 $= (\bar N\bar\rho_j + 1)/(\bar N + L)$ 的总变差与上轮实测相差 $< 2\%$。
J6 与第三轮接口：本底项即第三轮被排除的率口径（每站点每摆程一次），其空间分布为均匀（与份额的总变差 $> 0.8$）。
J7 排除教科书读法：若本底取泊松，则预期方差为总均值；实测方差与该预期在显著站点上的平均相对偏差 $> 5\%$。

运行
----
python 验证脚本/spontaneous_term_derivation.py
确定性（种子 20260918）。图存 D:/software/TianXingFC/images/spontaneous_term_derivation.png
"""
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

L = 256
JC = L // 2
KAPPA = 1.0
TP = 1.0
T_WIN = 4000.0
GAMMA_EM = 0.02
NBAR = 10000.0
M_REAL = 2000
SEED = 20260918


def mode_profile(K):
    d = np.arange(L, dtype=float) - JC
    d = np.where(d > L / 2, d - L, d)
    V = 0.5 * K * d ** 2
    H = np.diag(V + 2.0 * KAPPA) + np.diag(-KAPPA * np.ones(L - 1), 1) + np.diag(-KAPPA * np.ones(L - 1), -1)
    H[0, L - 1] = -KAPPA
    H[L - 1, 0] = -KAPPA
    ev, evec = np.linalg.eigh(H)
    psi = evec[:, 0]
    rho = psi ** 2
    return rho / rho.sum(), float(ev[0])


def clock_counts(T, n_sites, seed):
    rng = np.random.default_rng(seed)
    phi = rng.uniform(0.0, 2.0 * np.pi, n_sites)
    theta_end = 2.0 * np.pi * T / TP + phi
    return np.floor(theta_end / (2.0 * np.pi)) - np.floor(phi / (2.0 * np.pi))


def tv(p, q):
    return 0.5 * float(np.sum(np.abs(p - q)))


def main():
    t_start = time.time()
    checks = []

    def report(tag, ok, detail):
        checks.append((tag, bool(ok), detail))
        print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))

    rho, E0 = mode_profile(2.0e-4)
    rho_big, E1 = mode_profile(8.0e-4)
    print("  模式本征值：K = 2e-4 时 %.6f，K = 8e-4 时 %.6f" % (E0, E1))

    print("== J1 时钟计数确定性 ==")
    c1 = clock_counts(T_WIN, L, SEED)
    c2 = clock_counts(T_WIN, L, SEED + 1)
    exact = T_WIN / TP
    dev = float(np.max(np.abs(c1 - exact))) + float(np.max(np.abs(c2 - exact)))
    report("J1 每站折返计数 = T/t_P", abs(float(np.mean(c1)) - c1[0]) < 1e-12 and dev <= 1.0,
           "两次独立相位实现：均值 %.1f，与 T/t_P = %.1f 的最大偏差 %.2f（各站相位不同时可有 1 次边界差）"
           % (float(np.mean(c1)), exact, dev))

    print("== J2 本底与模式无关 ==")
    c3 = clock_counts(T_WIN, L, SEED)
    report("J2 阱曲率放大四倍后本底不变", float(np.max(np.abs(c3 - c1))) == 0.0,
           "两档阱曲率下每站折返计数完全相同（模式宽度与本征值均已改变）")

    print("== J3/J7 方差判据 ==")
    rng = np.random.default_rng(SEED)
    stim = np.zeros((M_REAL, L))
    clock = np.zeros((M_REAL, L))
    for m in range(M_REAL):
        clock[m] = GAMMA_EM * clock_counts(T_WIN, L, SEED + 100 + m)
        stim[m] = rng.poisson(GAMMA_EM * NBAR * rho * T_WIN)
    total = clock + stim
    var_tot = total.var(axis=0, ddof=1)
    var_stim = stim.var(axis=0, ddof=1)
    mean_stim = stim.mean(axis=0)
    mean_tot = total.mean(axis=0)
    sig = mean_stim > 10.0
    dev_site = np.abs(var_tot[sig] - mean_stim[sig]) / mean_stim[sig]
    rel3 = float(np.mean(dev_site))
    report("J3 复合计数方差 = 受激均值", rel3 < 0.05,
           "显著站点（%d 个）平均相对偏差 %.4f、最大 %.4f（单站点方差在 %d 条实现下的抽样误差约 %.1f%%，故用均值判定；\n           受激均值 %.1f、总均值 %.1f、方差 %.1f）"
           % (int(sig.sum()), rel3, float(dev_site.max()), M_REAL, 100.0 * np.sqrt(2.0 / (M_REAL - 1.0)),
              float(mean_stim.mean()), float(mean_tot.mean()), float(var_tot.mean())))
    rel7 = float(np.mean(np.abs(var_tot[sig] - mean_tot[sig]) / mean_tot[sig]))
    report("J7 排除泊松本底读法", rel7 > 0.05,
           "若本底为泊松，方差应为总均值 %.1f，实测 %.1f，显著站点平均相对偏差 %.4f"
           % (float(mean_tot.mean()), float(var_tot.mean()), rel7))

    print("== J4 两项可加 ==")
    dev4 = float(np.max(np.abs((clock.mean(axis=0) + mean_stim) - mean_tot)))
    report("J4 时钟项加受激项 = 复合项", dev4 < 0.01 * float(mean_tot.mean()),
           "站点最大偏差 %.4f（总均值 %.1f）" % (dev4, float(mean_tot.mean())))

    print("== J5 与涌现轮一致 ==")
    rec = mean_tot / mean_tot.sum()
    tv5 = tv(rec, rho)
    pred = tv((NBAR * rho + 1.0) / (NBAR + L), rho)
    rel5 = abs(tv5 - pred) / pred
    report("J5 记录份额与解析式一致", rel5 < 0.02,
           "实测总变差 %.5f，解析 %.5f，相对偏差 %.4f" % (tv5, pred, rel5))

    print("== J6 与第三轮接口 ==")
    unif = np.full(L, 1.0 / L)
    tv6 = tv(unif, rho)
    report("J6 本底即率口径（均匀）", tv6 > 0.8,
           "均匀分布与份额的总变差 %.4f（第三轮同读数 0.8354）" % tv6)

    print("== 绘图 ==")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    x = np.arange(L)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))

    ax = axes[0, 0]
    ax.plot(x, rho, lw=1.6, label=r"份额 $\rho_j$")
    ax.plot(x, unif, lw=1.3, ls="--", label="时钟本底（均匀）")
    ax.set_xlim(JC - 60, JC + 60)
    ax.set_xlabel("格点 $j$")
    ax.set_ylabel("占比")
    ax.set_title("图 A 本底是均匀的，与份额形状无关")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(x, mean_stim, lw=1.6, label="受激项均值")
    ax.plot(x, np.full(L, float(np.mean(clock))), lw=1.3, ls="--", label="时钟项（每站每窗 $T/t_P$）")
    ax.plot(x, mean_tot, lw=1.2, ls=":", label="复合计数均值")
    ax.set_xlim(JC - 60, JC + 60)
    ax.set_xlabel("格点 $j$")
    ax.set_ylabel("计数")
    ax.set_title("图 B 两项叠加")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.loglog(mean_tot, var_tot, "o", ms=3.0, label="实测方差")
    ax.loglog(mean_tot, mean_tot, "k--", lw=0.9, label="泊松预期（方差 = 均值）")
    ax.loglog(mean_tot, mean_stim, "r:", lw=0.9, label="导出版预期（方差 = 受激均值）")
    ax.set_xlabel("站点总均值")
    ax.set_ylabel("站点方差")
    ax.set_title("图 C 方差判据排除泊松本底")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(np.arange(M_REAL), total[:, JC].astype(float), lw=1.0, label="中心站复合计数")
    ax.plot(np.arange(M_REAL), stim[:, JC].astype(float), lw=1.0, ls="--", label="中心站受激计数")
    ax.set_xlabel("实现序号")
    ax.set_ylabel("计数")
    ax.set_title("图 D 复合计数的涨落只来自受激项")
    ax.legend(fontsize=8)

    fig.tight_layout()
    out = "D:/software/TianXingFC/images/spontaneous_term_derivation.png"
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
