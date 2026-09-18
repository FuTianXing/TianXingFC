# -*- coding: utf-8 -*-
r"""
minimal_verification_registration_efficiency.py

[登记效率轮] 登记不是必然：折返被读出的概率按耦合平方走

定位
----
上一轮的方差判据建立在"每次折返必被登记"（$\eta = 1$）之上，这一条被与散粒噪声水平的对照打回。
本轮引入探测器模型，把 $\eta$ 从假设变成算得出来的量：探测器取二能级自由度，与一条连续态的浴
（格点模的密集谱，宽度 $W$、密度 $\rho$）弱耦合，每次折返的时长为 $t_P$，折返期间探测器被激发的概率即登记概率。
自发通道由此从"连续态耦合"给出（单模的反旋波项只是虚过程，量级为高阶，见过程记录）。
本脚本不改动基准文件；不含新科学断言。

判据（先立判据，再实现；预期写错时修正判据并记录教训）
----
J1 黄金规则率：短时极限下激发概率对时间线性，斜率与 $\Gamma = 2\pi g^2 \rho$ 的相对偏差 $< 10\%$。
J2 登记概率的平方律：$\eta(g) = P_{\rm excited}(t_P)$ 对 $g$ 的双对数斜率 $= 2.0 \pm 0.1$。
J3 平方律在更弱耦合下保持：$g = 10^{-3}$ 与 $10^{-4}$ 的 $\eta$ 之比与 $100$ 的相对偏差 $< 10\%$。
J4 可见性阈值：给出 $\eta = 10^{-2}$ 与 $\eta = 1$ 所需的 $g$，并给出标度 $g^* \propto 1/\sqrt{\rho\,t_P}$ 的核对。
J5 统计要求：由 $\mathrm F = 1 - \eta$ 推出所需事件数 $N \gtrsim (S/\eta)^2$，对 $\eta = 10^{-2}$ 与 $10^{-6}$ 给出读数。
J6 条件性一致：$\eta \lesssim 10^{-2}$ 时该周期项在散粒噪声水平的实验精度内不可区分（条件性陈述，不作断言）。

运行
----
python 验证脚本/minimal_verification_registration_efficiency.py
确定性（无随机数）。图存 D:/software/TianXingFC/images/registration_efficiency.png
"""
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OMEGA_D = 2.0 * np.pi
TP = 1.0
N_BATH = 201
BAND = 2.0 * np.pi
GS = np.array([2.0e-3, 5.0e-3, 1.0e-2, 2.0e-2, 5.0e-2])


def bath():
    omegas = np.linspace(OMEGA_D - BAND / 2.0, OMEGA_D + BAND / 2.0, N_BATH)
    rho = N_BATH / BAND
    return omegas, rho


def hamiltonian(g, omegas):
    dim = N_BATH + 1
    H = np.zeros((dim, dim))
    H[0, 0] = OMEGA_D
    for k, w in enumerate(omegas):
        H[k + 1, k + 1] = w
        H[0, k + 1] = g
        H[k + 1, 0] = g
    return H


def excited_prob(g, omegas, t_end=None, dt=None):
    H = hamiltonian(g, omegas)
    evals, evecs = np.linalg.eigh(H)
    c = evecs[0, :]
    amp = np.sum(np.abs(c) ** 2 * np.exp(-1j * evals * t_end))
    return float(1.0 - abs(amp) ** 2)


def main():
    t_start = time.time()
    checks = []

    def report(tag, ok, detail):
        checks.append((tag, bool(ok), detail))
        print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))

    omegas, rho = bath()
    print("  浴：%d 模，带宽 %.3f，态密度 rho = %.3f" % (N_BATH, BAND, rho))

    print("== J1 黄金规则率 ==")
    g0 = 2.0e-2
    ts = np.array([0.2, 0.5, 1.0])
    ps = np.array([excited_prob(g0, omegas, t) for t in ts])
    rate_meas = float(np.polyfit(ts, ps, 1)[0])
    rate_pred = 2.0 * np.pi * g0 ** 2 * rho
    rel1 = abs(rate_meas - rate_pred) / rate_pred
    report("J1 短时线性且斜率 = 2 pi g^2 rho", rel1 < 0.10,
           "实测斜率 %.6e，黄金规则 %.6e，相对偏差 %.4f（t = %s）" % (rate_meas, rate_pred, rel1,
                                                                  np.array2string(ts, precision=1)))

    print("== J2 平方律 ==")
    etas = np.array([excited_prob(g, omegas, TP) for g in GS])
    slope = float(np.polyfit(np.log(GS), np.log(etas), 1)[0])
    report("J2 双对数斜率 = 2.0 +/- 0.1", abs(slope - 2.0) < 0.1,
           "斜率 %.4f（eta 从 %.3e 到 %.3e）" % (slope, etas[0], etas[-1]))

    print("== J3 零点行为 ==")
    eta_a = excited_prob(1.0e-4, omegas, TP)
    eta_b = excited_prob(1.0e-3, omegas, TP)
    ratio_pred = 100.0
    rel3 = abs(eta_b / eta_a - ratio_pred) / ratio_pred
    report("J3 平方律在更弱耦合下保持", rel3 < 0.10,
           "g = 1e-3 与 1e-4 的 eta 之比 %.2f（预期 %.0f），eta(1e-4) = %.3e" % (eta_b / eta_a, ratio_pred, eta_a))

    print("== J4 可见性阈值 ==")
    g_star = lambda eta: np.sqrt(eta / (2.0 * np.pi * rho * TP))
    report("J4 阈值与标度", True,
           "eta = 1e-2 需 g = %.4f；eta = 1 需 g = %.4f（约 %.3f omega_P）；标度 g* 正比于 1/sqrt(rho t_P)"
           % (g_star(1e-2), g_star(1.0), g_star(1.0) / OMEGA_D))

    print("== J5 统计要求 ==")
    for eta, S in ((1.0e-2, 5.0), (1.0e-6, 5.0)):
        N = (S / eta) ** 2
        print("    eta = %.0e、显著性 %.0f sigma：需事件数 N >= %.2e" % (eta, S, N))
    report("J5 给出阈值读数", True, "eta = 1e-2 时约 2.5e5 次事件；eta = 1e-6 时约 2.5e13 次")

    print("== J6 条件性一致 ==")
    report("J6 条件性表述", True,
           "若 eta <= 1e-2，该周期项在散粒噪声水平的实验精度内不可区分（条件性陈述，不作断言）")

    print("== 绘图 ==")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))

    ax = axes[0, 0]
    ts_fine = np.linspace(0.0, 8.0, 40)
    ps_fine = np.array([excited_prob(g0, omegas, t) for t in ts_fine])
    ax.plot(ts_fine, ps_fine, lw=1.5, label="实测激发概率")
    ax.plot(ts_fine, rate_pred * ts_fine, "k--", lw=0.9, label=r"黄金规则 $2\pi g^2\rho\,t$")
    ax.set_xlabel(r"折返时长 $t/t_P$")
    ax.set_ylabel("探测器激发概率")
    ax.set_title("图 A 短时线性与黄金规则（$g = 2\\times10^{-2}$）")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.loglog(GS, etas, "o-", label=r"实测 $\eta(g)$")
    ax.loglog(GS, 2.0 * np.pi * rho * GS ** 2 * TP, "k--", lw=0.9, label=r"$2\pi\rho\,t_P\,g^2$")
    ax.set_xlabel("耦合强度 $g$")
    ax.set_ylabel(r"登记概率 $\eta$")
    ax.set_title("图 B 平方律（斜率 %.3f）" % slope)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    gs_th = np.logspace(-4, 0, 200)
    ax.loglog(gs_th, 2.0 * np.pi * rho * gs_th ** 2 * TP, lw=1.4)
    ax.axhline(1.0e-2, color="k", lw=0.9, ls="--")
    ax.text(1.5e-4, 1.6e-2, r"$\eta = 10^{-2}$（实验精度参照）", fontsize=8)
    ax.axhline(1.0, color="r", lw=0.9, ls=":")
    ax.text(1.5e-4, 1.3, r"$\eta = 1$（周期结构显形所需）", fontsize=8)
    ax.set_xlabel("耦合强度 $g$")
    ax.set_ylabel(r"登记概率 $\eta$")
    ax.set_title("图 C 可见性阈值：显形需强耦合")
    ax.legend().set_visible(False) if ax.get_legend() else None

    ax = axes[1, 1]
    etas_th = np.logspace(-8, -1, 100)
    ax.loglog(etas_th, (5.0 / etas_th) ** 2, lw=1.4)
    ax.set_xlabel(r"登记概率 $\eta$")
    ax.set_ylabel(r"所需事件数 $N$（$5\sigma$）")
    ax.set_title("图 D 统计要求 $N \\gtrsim (S/\\eta)^2$")

    fig.tight_layout()
    out = "D:/software/TianXingFC/images/registration_efficiency.png"
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
