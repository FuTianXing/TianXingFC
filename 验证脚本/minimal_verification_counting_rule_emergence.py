# -*- coding: utf-8 -*-
r"""
minimal_verification_counting_rule_emergence.py

[计数规则涌现轮] 受激发射给份额、自发发射给均匀本底

定位
----
机制面末步的首轮执行，设定按作者 2026-09-18 裁决：载子取玻色型、记录通道取逐站点独立计数器、
动力学取幺正加弱耗散、范围一维闭链、接受自发项本底并如实登记其标度。
要判定的是：把"按激发数计"换成受激发射统计后，记录的块统计能否自己给出份额，以及自发本底带来的偏离标度。
本脚本不改动基准文件；不含新科学断言。

模型
----
一维闭链 $L = 256$，格距与节拍取 $l_P = t_P = 1$。场服从紧束缚算符 $H = -\kappa\,\Delta + V$（$V_j = \tfrac12K(j-j_c)^2$，$\kappa = 1$），
初态取宽度为基态 $1.35$ 倍的高斯（做呼吸振荡），按幺正演化（另跑一组弱耗散对照）。
每站点独立计数器：在时刻 $t$、站点 $j$ 的发射率为
  $\Gamma_j(t) = \gamma_{\text{em}}\big(\bar N\,\rho_j(t) + 1\big)$，
其中 $\rho_j(t)$ 是瞬时场密度份额、$\bar N$ 是模式总占据数、$1$ 为自发项（每站一个真空单位，故真空总计数为 $L$）。
记录统计由泊松过程产生。判准一（不注入）由此满足：抽样速率只由场密度与占据数给出，脚本中不出现按份额数组抽样的路径。

解析预期
----
记录占比为 $\big(\bar N\bar\rho_j + 1\big)/(\bar N + L)$，其中 $\bar\rho$ 为窗口内时间平均场密度份额。故
  $\mathrm{TV}(\text{记录}, \bar\rho) = \dfrac{\sum_j |1 - L\bar\rho_j|}{2(\bar N + L)}$，
随占据数按 $(\bar N + L)^{-1}$ 下降；达到既有精度 $0.006$ 所需占据数 $\bar N^* = \sum_j|1 - L\bar\rho_j|/(2\times0.006) - L$。

判据（先立判据，再实现；预期写错时修正判据并记录教训）
----
J1 高占据端涌现成立：$\bar N = 10^4$ 时记录占比与时间平均场密度的总变差 $< 0.03$。
J2 解析式核对：两档占据数下实测总变差与解析式的相对偏差 $< 10\%$。
J3 可失败通道标度：总变差对 $(\bar N + L)$ 的双对数斜率 $= -1 \pm 0.1$。
J4 阈值占据数：$\bar N^* > 10^4$（达到 $0.006$ 精度所需的模式占据数属宏观量级）。
J5 不注入（跟随瞬时场）：记录占比与时间平均场密度的总变差与解析式相符（$<0.03$），且与初始份额的总变差大于前者的五倍（$0.11896$ 对 $0.02102$）。阈值 $0.02$ 曾设在解析下界 $0.0207$ 之下，已订正。
J6 口径排他：同一动力学下，站点时钟口径（均匀）与时间平均场密度的总变差 $> 0.2$；碰撞口径（按 $\rho_j\rho_{j\pm1}$ 加权）与它的总变差 $> 0.05$。
J7 范式与归一化：份额归一化误差 $< 10^{-12}$；计数总数与理论期望计数之比在 $1 \pm 0.01$。
J8 弱耗散对照：加入 $\gamma_{\text{loss}} = 0.01$ 后，$\bar N = 10^4$ 的总变差相对变化 $< 10\%$。

运行
----
python 验证脚本/minimal_verification_counting_rule_emergence.py
确定性（种子 20260918）。图存 D:/software/TianXingFC/images/counting_rule_emergence.png
"""
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

L = 256
JC = L // 2
KAPPA = 1.0
K_TRAP = 2.0e-4
SEED = 20260918
T_WIN = 4000.0
DT = 0.05
SNAP_EVERY = int(round(1.0 / DT))
GAMMA_EM = 0.02
NBARS = [10.0, 100.0, 1000.0, 10000.0]


def trap_and_mode():
    d = np.arange(L, dtype=float) - JC
    d = np.where(d > L / 2, d - L, d)
    V = 0.5 * K_TRAP * d ** 2
    H = np.diag(V + 2.0 * KAPPA) + np.diag(-KAPPA * np.ones(L - 1), 1) + np.diag(-KAPPA * np.ones(L - 1), -1)
    H[0, L - 1] = -KAPPA
    H[L - 1, 0] = -KAPPA
    ev, evec = np.linalg.eigh(H)
    psi = evec[:, 0]
    rho = psi ** 2
    return rho / rho.sum(), V, d


def width_of(rho):
    x = np.arange(L, dtype=float)
    xc = float(np.sum(rho * x))
    return float(np.sqrt(np.sum(rho * (x - xc) ** 2)))


def evolve_field(rho_gs, V, gamma_loss=0.0, width_ratio=1.35):
    x = np.arange(L, dtype=float)
    w0 = width_of(rho_gs)
    A = np.exp(-0.5 * ((x - JC) / (width_ratio * w0)) ** 2).astype(complex)
    A /= np.linalg.norm(A)
    kfreq = 2.0 * np.pi * np.fft.fftfreq(L, d=1.0)
    kin = 4.0 * KAPPA * np.sin(kfreq / 2.0) ** 2
    n_steps = int(round(T_WIN / DT))
    snaps = []
    for step in range(n_steps):
        A = A * np.exp(-0.5 * DT * V)
        A = np.fft.ifft(np.fft.fft(A) * np.exp(-1j * kin * DT))
        A = A * np.exp(-0.5 * DT * V)
        if gamma_loss > 0.0:
            A = A * np.exp(-0.5 * gamma_loss * DT)
        if step % SNAP_EVERY == 0:
            n = np.abs(A) ** 2
            snaps.append(n / n.sum())
    return np.array(snaps)


def tv(p, q):
    return 0.5 * float(np.sum(np.abs(p - q)))


def main():
    t_start = time.time()
    checks = []

    def report(tag, ok, detail):
        checks.append((tag, bool(ok), detail))
        print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))

    rho_gs, V, d = trap_and_mode()
    sig = width_of(rho_gs)
    print("  基态宽度 sigma_gs = %.3f l_P" % sig)

    snaps = evolve_field(rho_gs, V)
    rho_bar = snaps.mean(axis=0)
    rho_bar = rho_bar / rho_bar.sum()
    rho0 = snaps[0]

    print("== J5 不注入（跟随瞬时场） ==")
    rng = np.random.default_rng(SEED)
    chi = float(np.sum(np.abs(1.0 - L * rho_bar)))
    rows = []
    for nbar in NBARS:
        counts = np.zeros(L)
        for s in snaps:
            rate = GAMMA_EM * (nbar * s + 1.0) * 1.0
            counts += rng.poisson(rate)
        rec = counts / counts.sum()
        tv_rec = tv(rec, rho_bar)
        tv_init = tv(rec, rho0)
        pred = chi / (2.0 * (nbar + L))
        expected = GAMMA_EM * (nbar + L) * snaps.shape[0]
        rows.append(dict(nbar=nbar, tv_rec=tv_rec, tv_init=tv_init, pred=pred,
                         ratio=counts.sum() / expected))
        print("  Nbar = %8.0f  计数 %8d  总变差(记录, 时间平均场) %.5f  解析 %.5f  总变差(记录, 初始) %.5f"
              % (nbar, int(counts.sum()), tv_rec, pred, tv_init))

    report("J5 记录跟随瞬时场", rows[-1]["tv_rec"] < 0.03 and rows[-1]["tv_init"] > 5.0 * rows[-1]["tv_rec"],
           "Nbar = 1e4：对时间平均场 %.5f，对初始份额 %.5f" % (rows[-1]["tv_rec"], rows[-1]["tv_init"]))

    print("== J1/J2 涌现与解析式 ==")
    report("J1 高占据端总变差 < 0.03", rows[-1]["tv_rec"] < 0.03, "%.5f" % rows[-1]["tv_rec"])
    worst = max(abs(r["tv_rec"] - r["pred"]) / r["pred"] for r in rows)
    report("J2 与解析式的相对偏差 < 10%", worst < 0.10,
           "最大相对偏差 %.4f（四档）" % worst)

    print("== J3 可失败通道标度 ==")
    xs = np.array([r["nbar"] + L for r in rows])
    ys = np.array([r["tv_rec"] for r in rows])
    slope = float(np.polyfit(np.log(xs), np.log(ys), 1)[0])
    report("J3 双对数斜率 = -1 +/- 0.1", abs(slope + 1.0) < 0.1, "斜率 %.4f" % slope)

    print("== J4 阈值占据数 ==")
    nstar = chi / (2.0 * 0.006) - L
    report("J4 阈值占据数 > 1e4", nstar > 1.0e4,
           "达到 0.006 精度所需占据数 %.3e（chi = %.1f）" % (nstar, chi))

    print("== J6 口径排他 ==")
    unif = np.full(L, 1.0 / L)
    tv_unif = tv(unif, rho_bar)
    coll = rho_bar * (np.roll(rho_bar, 1) + np.roll(rho_bar, -1))
    coll = coll / coll.sum()
    tv_coll = tv(coll, rho_bar)
    report("J6a 站点时钟口径仍失败", tv_unif > 0.2, "均匀与时间平均场的总变差 %.4f" % tv_unif)
    report("J6b 碰撞口径仍失败", tv_coll > 0.05, "密度平方加权与它的总变差 %.4f" % tv_coll)

    print("== J7 范式与归一化 ==")
    bad = max(abs(r["ratio"] - 1.0) for r in rows)
    report("J7 计数总数与期望一致", abs(rho_bar.sum() - 1.0) < 1e-12 and bad < 0.01,
           "归一化 %.2e，计数比最大偏差 %.4f" % (abs(rho_bar.sum() - 1.0), bad))

    print("== J8 弱耗散对照 ==")
    snaps_l = evolve_field(rho_gs, V, gamma_loss=0.01)
    rho_bar_l = snaps_l.mean(axis=0)
    rho_bar_l = rho_bar_l / rho_bar_l.sum()
    rng2 = np.random.default_rng(SEED + 5)
    counts_l = np.zeros(L)
    for s in snaps_l:
        counts_l += rng2.poisson(GAMMA_EM * (1.0e4 * s + 1.0))
    rec_l = counts_l / counts_l.sum()
    tv_l = tv(rec_l, rho_bar_l)
    rel = abs(tv_l - rows[-1]["tv_rec"]) / rows[-1]["tv_rec"]
    report("J8 弱耗散下结论稳定", rel < 0.10,
           "含损耗 %.5f 对幺正 %.5f，相对变化 %.3f" % (tv_l, rows[-1]["tv_rec"], rel))

    print("== 绘图 ==")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    x = np.arange(L)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))

    ax = axes[0, 0]
    ax.plot(x, rho_bar, lw=1.6, label=r"时间平均场密度 $\bar\rho_j$")
    ax.plot(x, rho0, lw=1.2, ls=":", label=r"初始场密度 $\rho_j(0)$")
    ax.set_xlim(JC - 60, JC + 60)
    ax.set_xlabel("格点 $j$")
    ax.set_ylabel("占比")
    ax.set_title("图 A 呼吸使两者分离（不注入判准的前提）")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    rng3 = np.random.default_rng(SEED + 9)
    counts_hi = np.zeros(L)
    for s in snaps:
        counts_hi += rng3.poisson(GAMMA_EM * (1.0e4 * s + 1.0))
    rec_hi = counts_hi / counts_hi.sum()
    ax.plot(x, rho_bar, lw=1.6, label=r"$\bar\rho_j$")
    ax.plot(x, rec_hi, lw=1.3, ls="--", label="记录占比（$\\bar N = 10^4$）")
    ax.set_xlim(JC - 60, JC + 60)
    ax.set_xlabel("格点 $j$")
    ax.set_ylabel("占比")
    ax.set_title("图 B 高占据端记录复现份额")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.loglog(xs, ys, "o-", label="实测总变差")
    ax.loglog(xs, ys[0] * xs[0] / xs, "k--", lw=0.9, label=r"$(\bar N + L)^{-1}$ 参考线")
    ax.set_xlabel(r"$\bar N + L$")
    ax.set_ylabel("总变差（记录对时间平均场）")
    ax.set_title("图 C 自发本底的标度（斜率 %.2f）" % slope)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(x, rho_bar, lw=1.6, label=r"$\bar\rho_j$")
    ax.plot(x, unif, lw=1.2, ls="--", label="站点时钟口径（均匀）")
    ax.plot(x, coll, lw=1.2, ls=":", label=r"碰撞口径（$\propto\rho_j\rho_{j\pm1}$）")
    ax.set_xlim(JC - 60, JC + 60)
    ax.set_xlabel("格点 $j$")
    ax.set_ylabel("占比")
    ax.set_title("图 D 同一动力学下两个口径仍失败")
    ax.legend(fontsize=8)

    fig.tight_layout()
    out = "D:/software/TianXingFC/images/counting_rule_emergence.png"
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
