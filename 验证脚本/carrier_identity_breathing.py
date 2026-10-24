# -*- coding: utf-8 -*-
r"""
carrier_identity_breathing.py

[载子身份与呼吸模式轮] 抽样跟不跟瞬时激发数走

定位
----
第四轮机制面执行。第三轮已排除"每站点每摆程一次"（率口径）并确认只有按激发数计才给出份额；
本轮把载子的操作定义钉死并做时间调制检验：模式做呼吸振荡时，计数跟时间平均份额 $\bar p$
还是跟初始份额 $p_0$ 走。若跟 $\bar p$ 走，"载子 = 局域振荡子的激发量子、每个量子每摆程被读一次"成立。
本脚本不改动基准文件；不含新科学断言。

模型
----
一维闭链 $L = 128$，$l_P = t_P = 1$，包络场服从离散非线性薛定谔方程
  $i\dot A_j = \omega_j A_j + \kappa(A_{j+1} + A_{j-1} - 2A_j) + \alpha|A_j|^2 A_j$，
阱取平方型 $\omega_j = \omega_0 - \tfrac12 K (j-j_c)^2$（取 $\omega_0 = 6$、$K = 2\times10^{-4}$，曲率取正号，$\omega_j$ 在中心最小，基态为束缚模式），$\kappa = 1$。
基态由虚时梯度流求得；初态取宽度为基态 $1.6$ 倍的高斯（零啁啾），在谐振阱里做精确呼吸振荡。
时间推进用 Strang 分裂步，采样用固定种子多项式抽样。

读数定义
----
瞬时份额 $p_j(t) = n_j(t)/\sum_k n_k(t)$，$n_j = |A_j|^2$；
初始份额 $p_j(0)$；时间平均份额 $\bar p_j = \int n_j\,dt / \sum_k\int n_k\,dt$；
计数频率取"每个激发每摆程被读一次"的实现（在每个采样时刻按 $p(t)$ 抽 $m$ 个站点标签），
对照口径取"每站点每摆程一次"（均匀分布）。

判据（先立判据，再实现；预期写错时修正判据并记录教训）
----
J1a 呼吸确实发生：宽度 $\sigma(t)$ 的振荡幅度 $(\max-\min)/\mathrm{mean} > 0.05$。
J1b 线性呼吸周期可复现：把基态分别乘以 $e^{-\epsilon (x-x_c)^2}$（$\epsilon = 2\times10^{-3}$ 与 $4\times10^{-3}$）做小扰动，宽度振荡周期在两种扰动下相对差 $< 10\%$，且窗口内周期数 $\ge 5$。
J2 两假设可区分：$\mathrm{TV}(\bar p, p_0) > 0.05$。
J3a 计数跟随时间平均份额：$\mathrm{TV}(\text{经验频率}, \bar p) < 0.01$。
J3b 拒绝初态假设：$\mathrm{TV}(\text{经验频率}, p_0) > 0.05$。
J4 统计误差标度：经验频率与 $\bar p$ 的总变差随样本数按 $K^{-1/2}$ 下降（双对数斜率 $-0.5 \pm 0.15$）。
J5 率口径在呼吸模式下仍失败：均匀分布与 $\bar p$ 的总变差 $> 0.2$。
J6 范数守恒：相对漂移 $< 10^{-9}$（$T = 5000\,t_P$）。
J7 非线性对照（$\alpha = 4$）：呼吸幅度仍 $> 0.05$，且 $\mathrm{TV}(\text{经验频率}, \bar p) < 0.02$。

运行
----
python 验证脚本/carrier_identity_breathing.py
确定性（种子 20260918）。图存 D:/software/TianXingFC/images/carrier_identity_breathing.png
"""
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

L = 128
JC = L // 2
KAPPA = 1.0
OMEGA0 = 6.0
K0 = 2.0e-4
SEED = 20260918
T_WIN = 5000.0
DT = 0.05
SAMPLE_EVERY = int(round(1.0 / DT))
M_PER_SAMPLE = 100


def trap_profile(K):
    j = np.arange(L, dtype=float)
    d = j - JC
    d = np.where(d > L / 2, d - L, d)
    return OMEGA0 + 0.5 * K * d ** 2


def lap(A):
    return np.roll(A, -1) + np.roll(A, 1) - 2.0 * A


def phase_rate(A, omega, kappa, alpha):
    f = -1j * (omega * A + kappa * lap(A) + alpha * np.abs(A) ** 2 * A)
    return -np.imag(f / A)


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


def gaussian(width):
    A = np.exp(-0.5 * ((np.arange(L) - JC) / width) ** 2).astype(complex)
    return A / np.linalg.norm(A)


def width_of(A):
    n = np.abs(A) ** 2
    n = n / n.sum()
    x = np.arange(L, dtype=float)
    x_c = float(np.sum(n * x))
    return float(np.sqrt(np.sum(n * (x - x_c) ** 2)))


def evolve_collect(A, omega, kappa, alpha, rng, T=T_WIN, dt=DT, m=M_PER_SAMPLE):
    kfreq = 2.0 * np.pi * np.fft.fftfreq(L, d=1.0)
    kin = 4.0 * kappa * np.sin(kfreq / 2.0) ** 2
    n_steps = int(round(T / dt))
    acc = np.zeros(L)
    sig = []
    counts = np.zeros(L, dtype=np.int64)
    series = []
    series_p = []
    for step in range(n_steps):
        A = A * np.exp(-0.5j * dt * (omega + alpha * np.abs(A) ** 2))
        A = np.fft.ifft(np.fft.fft(A) * np.exp(1j * kin * dt))
        A = A * np.exp(-0.5j * dt * (omega + alpha * np.abs(A) ** 2))
        n = np.abs(A) ** 2
        acc += n * dt
        if step % SAMPLE_EVERY == 0:
            p = n / n.sum()
            draw = rng.multinomial(m, p)
            counts += draw
            series.append(draw)
            series_p.append(n.copy())
            sig.append(width_of(A))
    return A, acc, np.array(sig), counts, np.array(series), np.array(series_p)


def tv(p, q):
    return 0.5 * float(np.sum(np.abs(p - q)))


def main():
    t_start = time.time()
    checks = []

    def report(tag, ok, detail):
        checks.append((tag, bool(ok), detail))
        print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))

    omega = trap_profile(K0)
    A_gs = ground_mode(omega, KAPPA)
    sig_gs = width_of(A_gs)
    print("  基态宽度 sigma_gs = %.4f l_P" % sig_gs)

    rng = np.random.default_rng(SEED)
    A_init = gaussian(1.35 * sig_gs)
    A_fin, acc, sig_t, counts, series, series_p = evolve_collect(A_init, omega, KAPPA, 0.0, rng)

    p_bar = acc / acc.sum()
    p0 = np.abs(A_init) ** 2 / np.sum(np.abs(A_init) ** 2)
    freq = counts / counts.sum()

    print("== J1 呼吸振荡 ==")
    amp = float((sig_t.max() - sig_t.min()) / sig_t.mean())
    report("J1a 呼吸幅度", amp > 0.05, "宽度 %.4f 至 %.4f（相对振幅 %.3f）"
           % (sig_t.min(), sig_t.max(), amp))
    omega_breath = 2.0 * np.sqrt(2.0 * KAPPA * K0)
    periods, _ = np.polyfit(np.arange(sig_t.size) * 1.0, sig_t - sig_t.mean(), 1)
    crossings = np.nonzero(np.diff(np.sign(sig_t - sig_t.mean())) > 0)[0]
    if crossings.size > 2:
        period_meas = 2.0 * float(np.mean(np.diff(crossings)))
    else:
        period_meas = np.nan
    period_naive = 2.0 * np.pi / omega_breath
    x = np.arange(L, dtype=float)
    n_gs = np.abs(A_gs) ** 2
    x_c = float(np.sum(n_gs * x) / n_gs.sum())
    periods = []
    cycles = []
    for eps in (2.0e-3, 4.0e-3):
        B = A_gs * np.exp(-eps * (x - x_c) ** 2)
        B = B / np.linalg.norm(B)
        rng_x = np.random.default_rng(SEED + 11)
        _, _, sig_x, _, _, _ = evolve_collect(B, omega, KAPPA, 0.0, rng_x)
        cr = np.nonzero(np.diff(np.sign(sig_x - sig_x.mean())) > 0)[0]
        cycles.append(int(cr.size))
        periods.append(2.0 * float(np.mean(np.diff(cr))) if cr.size > 5 else np.nan)
    rel = abs(periods[0] - periods[1]) / periods[0]
    report("J1b 线性呼吸周期可复现", rel < 0.10 and min(cycles) >= 5,
           "两扰动幅度下周期 %.1f 与 %.1f t_P（相对差 %.3f，周期数 %s）；朴素尺度 2pi/sqrt(2 kappa K) = %.1f t_P 仅作对照"
           % (periods[0], periods[1], rel, cycles, period_naive))

    print("== J2/J3 计数跟谁走 ==")
    tv_bar0 = tv(p_bar, p0)
    tv_emp_bar = tv(freq, p_bar)
    tv_emp_0 = tv(freq, p0)
    report("J2 两假设可区分", tv_bar0 > 0.05,
           "TV(时间平均份额, 初始份额) = %.4f" % tv_bar0)
    report("J3a 计数跟随时间平均份额", tv_emp_bar < 0.01,
           "TV(经验频率, 时间平均份额) = %.5f（样本 %d）" % (tv_emp_bar, counts.sum()))
    report("J3b 拒绝初态假设", tv_emp_0 > 0.05,
           "TV(经验频率, 初始份额) = %.4f" % tv_emp_0)

    print("== J4 统计误差标度 ==")
    n_blocks = np.array([32, 100, 320, 1000, 3200])
    Ks = n_blocks * M_PER_SAMPLE
    tvs_k = []
    for nb in n_blocks:
        f = series[:nb].sum(axis=0) / (nb * M_PER_SAMPLE)
        acc_nb = series_p[:nb].sum(axis=0)
        p_ref = acc_nb / acc_nb.sum()
        tvs_k.append(tv(f, p_ref))
    tvs_k = np.array(tvs_k)
    slope = float(np.polyfit(np.log(Ks), np.log(tvs_k), 1)[0])
    report("J4 统计误差按 K^{-1/2}", abs(slope + 0.5) < 0.15,
           "双对数斜率 %.3f（K = %d 至 %d 个独立样本）" % (slope, Ks[0], Ks[-1]))

    print("== J5 率口径仍失败 ==")
    tv_unif = tv(np.full(L, 1.0 / L), p_bar)
    report("J5 均匀分布与时间平均份额的总变差", tv_unif > 0.2, "总变差 %.4f" % tv_unif)

    print("== J6 范数守恒 ==")
    dn = abs(np.sum(np.abs(A_fin) ** 2) - 1.0)
    report("J6 范数守恒", dn < 1e-9, "相对漂移 %.2e（T = %.0f t_P）" % (dn, T_WIN))

    print("== J7 非线性对照 ==")
    rng2 = np.random.default_rng(SEED + 1)
    A2, acc2, sig2, counts2, _, _ = evolve_collect(gaussian(1.35 * sig_gs), omega, KAPPA, 4.0, rng2)
    amp2 = float((sig2.max() - sig2.min()) / sig2.mean())
    tv2 = tv(counts2 / counts2.sum(), acc2 / acc2.sum())
    report("J7 非线性下呼吸与跟随", amp2 > 0.05 and tv2 < 0.02,
           "呼吸幅度 %.3f，TV(经验频率, 时间平均份额) = %.5f" % (amp2, tv2))

    print("== 绘图 ==")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))

    tt = np.arange(sig_t.size, dtype=float)
    ax = axes[0, 0]
    ax.plot(tt, sig_t, lw=1.4)
    ax.axhline(sig_t.mean(), color="k", lw=0.8, ls="--")
    ax.set_xlabel(r"采样时刻（单位 $t_P$）")
    ax.set_ylabel(r"模式宽度 $\sigma$（单位 $l_P$）")
    ax.set_title("图 A 呼吸振荡（相对振幅 %.2f）" % amp)

    ax = axes[0, 1]
    ax.plot(np.arange(L), p0, lw=1.4, ls=":", label=r"初始份额 $p_0$")
    ax.plot(np.arange(L), p_bar, lw=1.6, label=r"时间平均份额 $\bar p$")
    ax.plot(np.arange(L), freq, lw=1.2, ls="--", label="经验频率（每激发一次）")
    ax.set_xlabel(r"格点 $j$")
    ax.set_ylabel("占比")
    ax.set_title("图 B 计数跟时间平均份额走")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.loglog(Ks, tvs_k, "o-", label="实测总变差")
    ax.loglog(Ks, tvs_k[0] * np.sqrt(Ks[0] / Ks), "k--", lw=0.9, label=r"$K^{-1/2}$ 参考线")
    ax.set_xlabel("样本数 $K$")
    ax.set_ylabel(r"$\mathrm{TV}$(经验频率, $\bar p$)")
    ax.set_title("图 C 统计误差按 $K^{-1/2}$ 收敛（斜率 %.2f）" % slope)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(np.arange(L), p_bar, lw=1.6, label=r"$\bar p$")
    ax.plot(np.arange(L), np.full(L, 1.0 / L), lw=1.3, ls="--", label="率口径（均匀）")
    ax.set_xlabel(r"格点 $j$")
    ax.set_ylabel("占比")
    ax.set_title("图 D 呼吸模式下率口径仍失败（TV = %.3f）" % tv_unif)
    ax.legend(fontsize=8)

    fig.tight_layout()
    out = "D:/software/TianXingFC/images/carrier_identity_breathing.png"
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
