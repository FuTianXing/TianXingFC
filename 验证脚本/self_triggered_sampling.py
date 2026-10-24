# -*- coding: utf-8 -*-
r"""
self_triggered_sampling.py

[自触发采样轮] 时钟由相位自己给出：临界条件 \theta = 2\pi n 的采样序列与鲁棒性

定位
----
对应开放问题 P3-12 判据 (i)（采样点自触发：$\theta = 2\pi n$ 由相位自身的临界条件给出，
不依赖外部时钟）。判据 (ii)（无偏性）与 (iii)（连续极限与带限）已在前两轮做出读数，
本轮补判据 (i)。本脚本不改动基准文件；不含新科学断言。

模型
----
单位取 $t_P = 1$，故 $\omega_P = 2\pi$。站点 $i$ 的相位取线性推进
  $\theta_i(t) = \omega_P t + \varphi_i$，
其中 $\varphi_i$ 是该站点相对参考站点的相位偏置。采样时刻定义为该相位自己的临界条件
  $\theta_i(t) = 2\pi n$，解出 $t_{i,n} = (2\pi n - \varphi_i)/\omega_P$。
检测只用相位值本身：先在参考网格上找 $\mathrm{mod}(\theta, 2\pi)$ 的跨越，再对该阶的
$g_n(t) = \theta(t) - 2\pi n$ 二分求根（向量化）。

噪声情形的口径（本轮订正）
----
物侧的采样格距就是 $t_P$：一次摆程给一次触发。所以噪声鲁棒性只在格点分辨率上提问，
  $\theta_k = \omega_P k\,t_P + \varphi_i + \sigma\xi_k$（$\xi_k$ 独立标准正态，固定种子），
触发计数取净绕数（相位指针完成的整圈数），它等于上穿次数减下穿次数。
若把参考网格细化到 $t_P$ 以下，会额外检测到大量"抖动穿越"（布朗路径在每一阶附近来回），
那不是物侧触发（详见 J6）。这一条是本轮判据先写错、复跑数据揭示后订正的，教训记在过程记录里。

判据（先立判据，再实现）
----
J1 自触发即解：二分求根所得 $t_{i,n}$ 与解析式 $(2\pi n - \varphi_i)/\omega_P$ 的最大偏差
   $< 10^{-10}$；均匀偏置与缺陷型偏置两组剖面（各 8 站点 x 12 阶）都成立。
J2 间隔刚性：同一站点相邻两次触发的间隔 $= t_P$，最大偏差 $< 10^{-12}$，与 $\varphi_i$ 无关。
J3 速率不变：窗口 $[0, T]$ 内每站点触发次数与 $T/t_P$ 的差不超过 1（取整数窗与含小数窗）；
   相对误差不超过 $1/T$（$T = 2^k + 0.37$，$k = 4..9$）。
J4 偏移即相位差：$t_{i,n} - t_{j,n} = (\varphi_j - \varphi_i)/\omega_P$，与 $n$ 无关
   （最大偏差 $< 10^{-12}$）：相位差只搬移时刻，不改动时钟速率。
J5 格点层面的噪声鲁棒（32 条实现，$T = 2^{13}$ 至 $2^{20}$ 个 $t_P$）：(a) 净绕数的系综均值 $= T/t_P$
   （偏差 $< 3$ 倍标准误）；(b) 净绕数的系综标准差随 $T$ 按 $T^{1/2}$ 增长（双对数斜率 $0.5 \pm 0.1$）；
   (c) 该标准差随 $\sigma$ 线性（$\sigma = 0.4$ 与 $0.1$ 之比 $= 4 \pm 1$）；(d) 单步跨越超过一圈的占比为 0。
J6 参考网格伪影的登记：把参考网格细化到 $2.5\times10^{-3}\,t_P$ 后，$\sigma = 0.4$ 出现额外触发
   （局部跨越计数大于净绕数），$\sigma = 0.1$ 不出现；阈值量级为 $\omega_P\sqrt{dt} \approx 0.314$。
   故物侧触发必须定义在格点分辨率上。
J7 采样定理不受损：以自触发时刻为采样点，对 $\nu < 0.5$ 的信号复原数字频率（误差 $< 10^{-5}$）；
   复原相位 $= \psi + 2\pi\nu\,\tau/t_P$（$\tau$ 为采样栅格的平移量，误差 $< 10^{-10}$，即站点相位只造成频率依赖的整体相位平移）；
   $\nu = 0.52 > 0.5$ 时按混叠复原为 $0.48$（与带限轮的读数一致）。

运行
----
python 验证脚本/self_triggered_sampling.py
确定性（随机种子固定为 20260918）。图存 D:/software/TianXingFC/images/self_triggered_sampling.png
"""
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TP = 1.0
OMEGA_P = 2.0 * np.pi
SEED = 20260918
N_SITE = 8


def make_phases(seed):
    return np.random.default_rng(seed).uniform(0.0, 2.0 * np.pi, N_SITE)


def make_defect_phases():
    x = np.arange(N_SITE)
    return 1.7 * np.sin(2.0 * np.pi * x / (N_SITE * 0.5)) + 0.4 * np.cos(2.0 * np.pi * x / 3.0)


def find_crossings(theta_fn, t0, t1, dt):
    ts = np.arange(t0, t1 + dt, dt)
    frac = np.mod(theta_fn(ts), 2.0 * np.pi)
    d = np.diff(frac)
    up_idx = np.nonzero(d < 0.0)[0]
    dn_idx = np.nonzero(d > np.pi)[0]
    out = {}
    for tag, idx, is_up in (("up", up_idx, True), ("dn", dn_idx, False)):
        if idx.size == 0:
            out[tag] = (np.empty(0), np.empty(0, dtype=int))
            continue
        vals = theta_fn(ts)
        n = np.floor(vals[idx] / (2.0 * np.pi)).astype(int) + (1 if is_up else 0)
        a = ts[idx].copy()
        b = ts[idx + 1].copy()
        for _ in range(50):
            m = 0.5 * (a + b)
            neg = (theta_fn(m) - 2.0 * np.pi * n) < 0.0
            a = np.where(neg, m, a)
            b = np.where(neg, b, m)
        out[tag] = (0.5 * (a + b), n)
    return out


def fit_digital_frequency(y, nu_grid):
    n = np.arange(y.size, dtype=float)
    best = (None, np.inf)
    for nu in nu_grid:
        c = np.cos(2.0 * np.pi * nu * n)
        s = np.sin(2.0 * np.pi * nu * n)
        A = np.column_stack([c, s])
        coef, res, rank, sv = np.linalg.lstsq(A, y, rcond=None)
        r = float(np.sum((y - A @ coef) ** 2))
        if r < best[1]:
            best = (nu, r)
    return best[0], best[1]


def main():
    t_start = time.time()
    phis = make_phases(SEED)
    phis_def = make_defect_phases()
    checks = []

    def report(tag, ok, detail):
        checks.append((tag, bool(ok), detail))
        print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))

    print("== J1 自触发即解 ==")
    for tag, pl in (("J1a 均匀偏置", phis), ("J1b 缺陷型偏置", phis_def)):
        w = 0.0
        for p in pl:
            cr = find_crossings(lambda t, p=p: OMEGA_P * t + p, 0.0, 12.0 * TP, 2e-3)
            times, orders = cr["up"]
            analytic = (2.0 * np.pi * orders - p) / OMEGA_P
            w = max(w, float(np.max(np.abs(times - analytic))))
        report(tag, w < 1e-10, "最大偏差 %.2e（8 站点 x 12 阶）" % w)

    print("== J2 间隔刚性 ==")
    gaps_all = []
    for p in phis:
        times, _ = find_crossings(lambda t, p=p: OMEGA_P * t + p, 0.0, 12.0 * TP, 2e-3)["up"]
        gaps_all.append(np.diff(times))
    gaps = np.concatenate(gaps_all)
    dev2 = float(np.max(np.abs(gaps - TP)))
    report("J2 间隔 = t_P", dev2 < 1e-12, "最大偏差 %.2e（%d 个间隔）" % (dev2, gaps.size))

    print("== J3 速率不变 ==")
    ks = np.arange(4, 10)
    Ts = 2.0 ** ks + 0.37
    abs_err = []
    rel = []
    for T in Ts:
        worst_abs = 0.0
        for p in phis:
            cnt = find_crossings(lambda t, p=p: OMEGA_P * t + p, 0.0, T, 2e-3)["up"][0].size
            worst_abs = max(worst_abs, abs(cnt - T / TP))
        abs_err.append(worst_abs)
        rel.append(worst_abs / (T / TP))
    rel = np.array(rel)
    report("J3a 计数误差 <= 1", float(np.max(abs_err)) <= 1.0, "最大绝对误差 %.3f" % float(np.max(abs_err)))
    report("J3b 相对误差 <= 1/T", bool(np.all(rel <= 1.0 / Ts * (1.0 + 1e-12))),
           "最大相对误差 %.3e（T = %.2f t_P 处上界 %.3e）" % (float(np.max(rel)), Ts[-1], 1.0 / Ts[-1]))

    print("== J4 偏移即相位差 ==")
    dev4 = 0.0
    base_t, base_o = find_crossings(lambda t: OMEGA_P * t + phis[0], 0.0, 12.0 * TP, 2e-3)["up"]
    for p in phis[1:]:
        ti, oi = find_crossings(lambda t, p=p: OMEGA_P * t + p, 0.0, 12.0 * TP, 2e-3)["up"]
        common = np.intersect1d(base_o, oi)
        tb = base_t[np.searchsorted(base_o, common)]
        tii = ti[np.searchsorted(oi, common)]
        dev4 = max(dev4, float(np.max(np.abs((tii - tb) - (phis[0] - p) / OMEGA_P))))
    report("J4 时刻差 = 相位差/omega_P", dev4 < 1e-12, "最大偏差 %.2e" % dev4)

    print("== J5 格点层面的噪声鲁棒 ==")
    ks5 = np.arange(13, 21)
    Ts5 = TP * 2.0 ** ks5
    sigmas = [0.1, 0.4]
    M = 32
    n_max = int(Ts5[-1] / TP)
    wind = {s: np.zeros((M, ks5.size)) for s in sigmas}
    steps = {s: [] for s in sigmas}
    for s in sigmas:
        for m in range(M):
            rng = np.random.default_rng(SEED + 1000 + m)
            W = np.concatenate([[0.0], np.cumsum(rng.standard_normal(n_max))])
            for j, T in enumerate(Ts5):
                n = int(T / TP)
                wind[s][m, j] = n + int(round(s * W[n] / (2.0 * np.pi)))
                st = np.round(s * np.diff(W[: n + 1]) / (2.0 * np.pi))
                steps[s].append(float(np.count_nonzero(np.abs(st) > 1)) / max(st.size, 1))
    for s in sigmas:
        mean_dev = float(np.max(np.abs(wind[s].mean(axis=0) - Ts5 / TP)))
        sem = float(np.max(wind[s].std(axis=0, ddof=1) / np.sqrt(M)))
        report("J5a 净绕数均值 = T/t_P（sigma = %.1f）" % s, mean_dev < 3.0 * max(sem, 1e-12),
               "最大偏差 %.4f，3 倍标准误 %.4f" % (mean_dev, 3.0 * sem))
    ratio = float(np.max(wind[0.4].std(axis=0, ddof=1) / np.maximum(wind[0.1].std(axis=0, ddof=1), 1e-12)))
    report("J5c 标准差正比于 sigma", 3.0 < ratio < 5.0,
           "sigma = 0.4 与 0.1 的标准差之比（最大）%.3f（期望 4）" % ratio)
    for s in sigmas:
        sd = wind[s].std(axis=0, ddof=1)
        sl = float(np.polyfit(np.log2(Ts5), np.log2(sd), 1)[0])
        report("J5b 抖动按 T^0.5（sigma = %.1f）" % s, abs(sl - 0.5) < 0.1, "斜率 %.4f（期望 0.5）" % sl)
    worst_short = max(float(np.max(steps[s])) for s in sigmas)
    report("J5d 无额外触发（sigma <= 1）", worst_short == 0.0, "单步跨越多圈的占比最大 %.4f" % worst_short)

    print("== J6 参考网格伪影的登记 ==")
    dt_f = 2.5e-3
    T_f = 64.0 * TP
    n_grid = int(T_f / dt_f) + 1
    t_grid = np.arange(n_grid) * dt_f
    rng = np.random.default_rng(SEED + 7)
    W = np.concatenate([[0.0], np.cumsum(rng.standard_normal(n_grid - 1) * np.sqrt(dt_f))])
    extra = {}
    for s in (0.1, 0.4):
        th_fn = lambda t, W=W, s=s: OMEGA_P * t + phis[0] + s * np.interp(t, t_grid, W)
        cr = find_crossings(th_fn, 0.0, T_f, dt_f)
        n_local = cr["up"][0].size + cr["dn"][0].size
        net = cr["up"][0].size - cr["dn"][0].size
        extra[s] = n_local - abs(net)
        print("  sigma = %.1f：局部跨越 %d 次，净绕数 %+d，额外触发 %d 次" % (s, n_local, net, n_local - abs(net)))
    report("J6 细网格额外触发随 sigma 出现", extra[0.4] > 0 and extra[0.4] > extra[0.1],
           "额外触发 sigma = 0.1 -> %d 次，sigma = 0.4 -> %d 次（阈值量级 omega_P*sqrt(dt) = %.3f）" % (extra[0.1], extra[0.4], OMEGA_P * np.sqrt(dt_f)))

    print("== J7 采样定理不受损 ==")
    omega_tests = [0.05, 0.2, 0.3, 0.45]
    psi = 0.7
    phi_t = phis[3]
    worst_nu = 0.0
    worst_ph = 0.0
    for nu in omega_tests:
        times, _ = find_crossings(lambda t: OMEGA_P * t + phi_t, 0.0, 70.0 * TP, 2e-3)["up"]
        times = times[:64]
        y = np.cos(2.0 * np.pi * nu * times + psi)
        nu_c, _ = fit_digital_frequency(y, np.linspace(0.001, 0.499, 5000))
        nu_f, _ = fit_digital_frequency(y, np.linspace(nu_c - 5e-4, nu_c + 5e-4, 2001))
        worst_nu = max(worst_nu, abs(nu_f - nu))
        n = np.arange(y.size, dtype=float)
        A = np.column_stack([np.cos(2.0 * np.pi * nu * n), np.sin(2.0 * np.pi * nu * n)])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        ph_rec = float(np.arctan2(-coef[1], coef[0]))
        tau_eff = float(times[0] % TP)
        pred = psi + 2.0 * np.pi * nu * tau_eff / TP
        worst_ph = max(worst_ph, abs((ph_rec - pred + np.pi) % (2.0 * np.pi) - np.pi))
    report("J7a 数字频率复原（nu < 0.5）", worst_nu < 1e-5, "最大误差 %.2e" % worst_nu)
    report("J7b 相位复原 = psi + 2*pi*nu*tau/t_P", worst_ph < 1e-10,
           "最大偏差 %.2e（tau = %.6f t_P）" % (worst_ph, tau_eff))
    times, _ = find_crossings(lambda t: OMEGA_P * t + phi_t, 0.0, 70.0 * TP, 2e-3)["up"]
    times = times[:64]
    y = np.cos(2.0 * np.pi * 0.52 * times + psi)
    nu_a, _ = fit_digital_frequency(y, np.linspace(0.4, 0.499, 5000))
    nu_a, _ = fit_digital_frequency(y, np.linspace(nu_a - 5e-4, nu_a + 5e-4, 2001))
    report("J7c 混叠：nu = 0.52 复原为 0.48", abs(nu_a - 0.48) < 1e-5, "复原 nu = %.6f" % nu_a)

    print("== 绘图 ==")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))

    ax = axes[0, 0]
    t_show = np.linspace(0.0, 4.0 * TP, 2000)
    for i in (0, 1, 2):
        ph = phis[i]
        ax.plot(t_show, np.mod(OMEGA_P * t_show + ph, 2.0 * np.pi), lw=1.0,
                label=r"站点 %d：$\varphi$ = %.2f" % (i, ph))
        tc = find_crossings(lambda t, p=ph: OMEGA_P * t + p, 0.0, 4.0 * TP, 2e-3)["up"][0]
        ax.plot(tc, np.zeros_like(tc), "v", ms=5.0)
    ax.set_xlabel(r"时间 $t$（单位 $t_P$）")
    ax.set_ylabel(r"相位折返值（$\mathrm{mod}\ 2\pi$）")
    ax.set_title("图 A 相位自己的临界条件给出采样时刻")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for i, g in enumerate(gaps_all):
        ax.plot(np.arange(g.size), g, ".", ms=3.0, label="站点 %d" % i if i < 3 else None)
    ax.axhline(TP, color="k", lw=1.0, ls="--")
    ax.set_ylim(TP - 0.02, TP + 0.02)
    ax.set_xlabel("间隔序号")
    ax.set_ylabel(r"触发间隔（单位 $t_P$）")
    ax.set_title(r"图 B 间隔刚性：全部落在 $t_P$ 上，与 $\varphi_i$ 无关")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.loglog(Ts, rel, "o-", label="实测相对误差")
    ax.loglog(Ts, 1.0 / Ts, "k--", lw=1.0, label=r"$1/T$ 参考线")
    ax.set_xlabel(r"窗口长度 $T$（单位 $t_P$）")
    ax.set_ylabel("计数相对误差")
    ax.set_title("图 C 速率不变：相对误差落在 $1/T$ 之下")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    for s in sigmas:
        ax.loglog(Ts5, wind[s].std(axis=0, ddof=1), "o-", label=r"$\sigma$ = %.1f" % s)
    sd0 = wind[0.4].std(axis=0, ddof=1)
    ax.loglog(Ts5, sd0[0] * np.sqrt(Ts5 / Ts5[0]), "k--", lw=1.0, label=r"$T^{1/2}$ 参考线")
    ax.set_xlabel(r"窗口长度 $T$（单位 $t_P$）")
    ax.set_ylabel("净绕数标准差（32 条实现）")
    ax.set_title("图 D 噪声只加抖动：长期速率不变，抖动按 $T^{1/2}$")
    ax.legend(fontsize=8)

    fig.tight_layout()
    out = "D:/software/TianXingFC/images/self_triggered_sampling.png"
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
