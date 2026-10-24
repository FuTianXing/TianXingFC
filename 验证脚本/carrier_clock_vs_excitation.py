# -*- coding: utf-8 -*-
r"""
carrier_clock_vs_excitation.py

[载子钟与激发数轮] 率口径给不出份额：本体每站一次 vs 每激发一次

定位
----
第三轮机制面执行，设定按作者 2026-09-18 裁决：载子取元相子本体，离散时间格点演化（一步 $t_P$），
事件取相位折返的净绕数（格点分辨率），初态取束缚模式加小扰动，边界默认闭链。
要判定的是：按"每个本体每摆程记一次"统计的净绕数率，能否给出份额 $N_a/N$。
本脚本不改动基准文件；不含新科学断言。

模型
----
一维闭链 $L = 128$，$l_P = t_P = 1$。包络场服从离散非线性薛定谔方程
  $i\dot A_j = \omega_j A_j + \kappa(A_{j+1} + A_{j-1} - 2A_j) + \alpha|A_j|^2 A_j$，
阱取平方型 $\omega_j = \omega_0 - \tfrac12 K (j-j_c)^2$，$\kappa$ 为最近邻耦合，$\alpha$ 为 Kerr 非线性。
基态由虚时梯度流迭代求得（每 50 步查一次站点率的均匀度作为收敛判据），时间推进用 Strang 分裂步。

读数定义
----
份额取模式权重 $n_j/\sum n$（每个激发一次，对应"每载子每摆程一次"）。
率口径取站点相位推进率 $\dot\varphi_j = -\mathrm{Im}(\dot A_j/A_j)$ 的占比，对应"每个本体每摆程记一次净绕数"。

判据（先立判据，再实现；预期写错时修正判据并记录教训）
----
J1a 线性定常模式：$\dot\varphi_j$ 各站点相同（相对散布 $<10^{-10}$），率占比与份额的总变差 $> 0.2$。
J1b 解耦极限 $\kappa \to 0$：率占比 $= \omega_j/\sum\omega_j$（误差 $<10^{-12}$），与份额的总变差 $> 0.2$。
J2a 非线性定常模式（$\alpha = 4, 64, 256$）：率仍均匀（相对散布 $<10^{-8}$）。定常方程给出
   $\omega_M = \omega_j + \kappa(\Delta A/A)_j + \alpha n_j$ 为常数，故率口径的均匀性与 $\alpha$ 无关（恒等式级）。
J2b 率口径不携带形状信息：对全部 $\alpha$，$\mathrm{TV}$(率占比, 份额) 与 $\mathrm{TV}$(均匀, 份额) 逐点相等（差 $<10^{-9}$），即率口径的偏差完全由份额形状决定。原预期（强非线性把率口径拉向份额）被推翻：TV 随 $\alpha$ 下降来自份额自身被排斥非线性压平。
J3 激发数口径：占比 $= n_j/\sum n$，归一化与逐点偏差 $<10^{-15}$（恒等式级）。
J4 剖面稳定性：加 $1\%$ 扰动后演化 $T = 5000\,t_P$，$\mathrm{TV}(n(T), n(0)) < 0.05$（线性与非线性各一组）。
J5 失败的结构性：把阱曲率放大 4 倍（模式更窄），率口径与份额的总变差仍 $> 0.2$。
J6 范数守恒：分裂步的每个因子模为 1，范数只受累积舍入影响，$10^5$ 步下的相对漂移 $< 10^{-9}$（$T = 5000\,t_P$）。

运行
----
python 验证脚本/carrier_clock_vs_excitation.py
确定性（种子 20260918）。图存 D:/software/TianXingFC/images/carrier_clock_vs_excitation.png
"""
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

L = 128
JC = L // 2
KAPPA = 1.0
OMEGA0 = 4.0
SEED = 20260918


def trap_profile(K):
    j = np.arange(L, dtype=float)
    d = j - JC
    d = np.where(d > L / 2, d - L, d)
    return OMEGA0 + 0.5 * K * d ** 2


def lap(A):
    return np.roll(A, -1) + np.roll(A, 1) - 2.0 * A


def ground_mode(omega, kappa, alpha=0.0, dtau0=0.02, max_steps=200000, tol=1e-10):
    A = np.exp(-0.5 * ((np.arange(L) - JC) / 12.0) ** 2).astype(complex)
    A /= np.linalg.norm(A)
    for step in range(max_steps):
        dtau = dtau0 / (1.0 + alpha * float(np.max(np.abs(A) ** 2)))
        A = A - dtau * (omega * A + kappa * lap(A) + alpha * np.abs(A) ** 2 * A)
        A /= np.linalg.norm(A)
        if step % 50 == 0:
            r = phase_rate(A, omega, kappa, alpha)
            if (r.max() - r.min()) / abs(r.mean()) < tol:
                break
    return A


def phase_rate(A, omega, kappa, alpha):
    f = -1j * (omega * A + kappa * lap(A) + alpha * np.abs(A) ** 2 * A)
    return -np.imag(f / A)


def evolve(A, omega, kappa, alpha, T, dt=0.05):
    n_steps = int(round(T / dt))
    kfreq = 2.0 * np.pi * np.fft.fftfreq(L, d=1.0)
    kin = 4.0 * kappa * np.sin(kfreq / 2.0) ** 2
    def half_phase(Aa, factor):
        return Aa * np.exp(-1j * factor * (omega + alpha * np.abs(Aa) ** 2))
    for _ in range(n_steps):
        A = half_phase(A, dt / 2.0)
        A = np.fft.ifft(np.fft.fft(A) * np.exp(1j * kin * dt))
        A = half_phase(A, dt / 2.0)
    return A


def tv(p, q):
    return 0.5 * float(np.sum(np.abs(p - q)))


def share_of(A):
    n = np.abs(A) ** 2
    return n / n.sum()


def rate_share(A, omega, kappa, alpha):
    r = phase_rate(A, omega, kappa, alpha)
    return r / r.sum()


def main():
    t_start = time.time()
    checks = []

    def report(tag, ok, detail):
        checks.append((tag, bool(ok), detail))
        print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))

    K0 = 2.0e-3
    omega = trap_profile(K0)
    A0 = ground_mode(omega, KAPPA)
    share = share_of(A0)

    print("== J1a 线性定常模式的率口径 ==")
    r_lin = phase_rate(A0, omega, KAPPA, 0.0)
    spread = float((r_lin.max() - r_lin.min()) / abs(r_lin.mean()))
    tv_lin = tv(r_lin / r_lin.sum(), share)
    report("J1a 站点率相同", spread < 1e-10, "相对散布 %.2e，模式频率 %.6f" % (spread, r_lin.mean()))
    report("J1a 率口径不能承载份额", tv_lin > 0.2, "率占比与份额的总变差 %.4f" % tv_lin)

    print("== J1b 解耦极限 ==")
    r_free = phase_rate(A0, omega, 0.0, 0.0)
    dev_free = float(np.max(np.abs(r_free - omega)))
    tv_free = tv(r_free / r_free.sum(), share)
    report("J1b 率 = 局域自然频率", dev_free < 1e-12,
           "最大偏差 %.2e（率占比 = omega_j/Sum omega_j，TV = %.4f）" % (dev_free, tv_free))

    print("== J2 非线性下的率口径 ==")
    for al in (4.0, 64.0, 256.0):
        A_nl = ground_mode(omega, KAPPA, alpha=al)
        r_nl = phase_rate(A_nl, omega, KAPPA, al)
        sp = float((r_nl.max() - r_nl.min()) / abs(r_nl.mean()))
        report("J2a 率仍均匀（alpha = %.0f）" % al, sp < 1e-8,
               "相对散布 %.2e，模式频率 %.6f" % (sp, r_nl.mean()))
    alphas = np.array([0.0, 1.0, 4.0, 16.0, 64.0, 256.0, 1024.0])
    tvs = []
    tvs_unif = []
    shares = []
    for al in alphas:
        A_al = ground_mode(omega, KAPPA, alpha=al)
        sh = share_of(A_al)
        shares.append(sh)
        tvs.append(tv(rate_share(A_al, omega, KAPPA, al), sh))
        tvs_unif.append(tv(np.full(L, 1.0 / L), sh))
    tvs = np.array(tvs)
    tvs_unif = np.array(tvs_unif)
    gap = float(np.max(np.abs(tvs - tvs_unif)))
    report("J2b 率口径不携带形状信息", gap < 1e-9,
           "TV(率, 份额) 与 TV(均匀, 份额) 的最大差 %.2e；TV = %s"
           % (gap, np.array2string(tvs, precision=4)))
    report("J2b 份额被排斥非线性压平", bool(np.all(np.diff(tvs_unif) < 0)),
           "TV(均匀, 份额) = %s（alpha = 1024 时降至 %.3f，故均匀抽样与份额近乎不可区分）"
           % (np.array2string(tvs_unif, precision=4), tvs_unif[-1]))

    print("== J3 激发数口径 ==")
    dev3 = float(np.max(np.abs(share - np.abs(A0) ** 2 / np.sum(np.abs(A0) ** 2))))
    report("J3 占比 = |A_j|^2 归一化", abs(share.sum() - 1.0) < 1e-15 and dev3 < 1e-15,
           "归一化偏差 %.2e，逐点偏差 %.2e" % (abs(share.sum() - 1.0), dev3))

    print("== J4 剖面稳定性 ==")
    rng = np.random.default_rng(SEED)
    for tag, al in (("线性", 0.0), ("非线性 alpha = 4", 4.0)):
        A_ic = ground_mode(omega, KAPPA, alpha=al)
        n_before = np.abs(A_ic) ** 2
        pert = A_ic * np.exp(1j * 0.01 * rng.standard_normal(L))
        pert /= np.linalg.norm(pert)
        A_T = evolve(pert, omega, KAPPA, al, T=5000.0)
        drift = tv(np.abs(A_T) ** 2 / np.sum(np.abs(A_T) ** 2), n_before / n_before.sum())
        report("J4 剖面漂移（%s）" % tag, drift < 0.05, "TV = %.4f" % drift)

    print("== J5 失败的结构性 ==")
    omega5 = trap_profile(4.0 * K0)
    A5 = ground_mode(omega5, KAPPA)
    tv5 = tv(rate_share(A5, omega5, KAPPA, 0.0), share_of(A5))
    report("J5 模式更窄时仍失败", tv5 > 0.2, "总变差 %.4f（阱曲率放大 4 倍）" % tv5)

    print("== J6 范数守恒 ==")
    A_ic = ground_mode(omega, KAPPA)
    A_T = evolve(A_ic.copy(), omega, KAPPA, 0.0, T=5000.0)
    dn = abs(np.sum(np.abs(A_T) ** 2) - np.sum(np.abs(A_ic) ** 2))
    report("J6 范数守恒", dn < 1e-9, "范数漂移 %.2e（T = 5000 t_P、10^5 步，属累积舍入）" % dn)

    print("== 绘图 ==")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))

    ax = axes[0, 0]
    ax.plot(np.arange(L), share, lw=1.6, label="份额（激发数口径 $n_j/\\sum n$）")
    ax.plot(np.arange(L), r_lin / r_lin.sum(), lw=1.4, ls="--", label="率口径（$\\alpha = 0$，均匀）")
    ax.plot(np.arange(L), rate_share(ground_mode(omega, KAPPA, alpha=4.0), omega, KAPPA, 4.0),
            lw=1.4, ls=":", label="率口径（$\\alpha = 4$）")
    ax.set_xlabel("格点 $j$")
    ax.set_ylabel("占比")
    ax.set_title("图 A 率口径抹掉空间结构")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.semilogx(alphas[1:], tvs[1:], "o-")
    ax.axhline(0.5, color="k", lw=0.9, ls="--")
    ax.text(alphas[2], 0.52, "0.5 参考线", fontsize=8)
    ax.set_xlabel(r"非线性强度 $\alpha$")
    ax.set_ylabel("率占比与份额的总变差")
    ax.set_title("图 B 非线性也不把率口径拉向份额")

    ax = axes[1, 0]
    A_T_plot = evolve((A0 * np.exp(1j * 0.01 * np.random.default_rng(SEED).standard_normal(L)))
                      / np.linalg.norm(A0 * np.exp(1j * 0.01 * np.random.default_rng(SEED).standard_normal(L))),
                      omega, KAPPA, 0.0, T=5000.0)
    ax.plot(np.arange(L), share, lw=1.6, label=r"$n_j(0)$")
    ax.plot(np.arange(L), share_of(A_T_plot), lw=1.4, ls="--", label=r"$n_j(T)$，$T = 5000\,t_P$")
    ax.set_xlabel("格点 $j$")
    ax.set_ylabel("份额")
    ax.set_title("图 C 份额剖面在动力学中稳定")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(np.arange(L), (omega - omega.min()) / (omega.max() - omega.min()), lw=1.6,
            label=r"$(\omega_j-\omega_{\min})/\Delta\omega$")
    ax.plot(np.arange(L), share, lw=1.4, ls="--", label="份额")
    ax.set_xlabel("格点 $j$")
    ax.set_ylabel("归一化")
    ax.set_title("图 D 解耦极限：率口径给出阱形分布")
    ax.legend(fontsize=8)

    fig.tight_layout()
    out = "D:/software/TianXingFC/images/carrier_clock_vs_excitation.png"
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
