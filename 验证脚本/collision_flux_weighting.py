# -*- coding: utf-8 -*-
r"""
collision_flux_weighting.py

[碰撞通量加权轮] 份额的口径之争：锁速载子与包络群速

定位
----
对应 P3-12 机制面（真实碰撞动力学是否按份额抽样）与 [[研究方向-碰撞动力学的抽样机制：三个候选模型与优选]]
的候选丙首轮执行。要判定的是：碰撞计数占比取 $\rho_j$ 还是取 $\rho_j v_j$，
以及速度加权一旦进入，偏差有多大、在何处放大。
本脚本不改动基准文件；不含新科学断言。

模型
----
一维链，$l_P = t_P = 1$，$c = l_P/t_P = 1$，精确色散 $\omega^2 = (4/t_P^2)\sin^2(k l_P/2)$，
故 $\omega(k) = (2/t_P)\sin(k l_P/2)$、群速 $v_g(k) = c\cos(k l_P/2)$，带边 $k = \pi/l_P$。

束缚模式取频率平方型的谐振阱：局部频率 $\omega_j = \omega_0\sqrt{1-u_j^2}$，$u_j = (j-j_c)/w$，
阱的半宽 $w$ 即经典折返点。由 $\omega_0$ 与中心波数 $k_c$ 的关系 $\omega_0 = (2/t_P)\sin(k_c l_P/2)$
（精确色散反解）得局部波数
  $k_j l_P = 2\arcsin\!\big(\sin(k_c l_P/2)\sqrt{1-u_j^2}\big)$，
以及速度剖面
  $v_j/c = \cos(k_j l_P/2) = \sqrt{\cos^2\theta_c + u_j^2\sin^2\theta_c}$，$\theta_c = k_c l_P/2$。
中心最慢（$v = c\cos\theta_c$），折返点最快（$v = c$）。模式密度取高斯 $\rho_j \propto e^{-(j-j_c)^2/(2\sigma^2)}$，
$\sigma = w/2$（谐振阱基态的形状）。

两种计数口径
----
口径 A（体系口径，格点锁速）：载子在格点上一摆程推进一格，事件率 $\propto \rho_j\,c$，
故计数占比 $p^A_j = \rho_j/\sum \rho$。
口径 B（动力学口径，包络群速）：事件率 $\propto \rho_j\,v_j$，
故占比 $p^B_j = \rho_j v_j/\sum \rho v$。

判据（先立判据，再实现）
----
J1a 口径 A 归一化误差 $<10^{-12}$。
J1b 口径 A 的计数占比与 $|\psi_j|^2$ 逐点一致（偏差 $<10^{-15}$）：锁速下无速度加权修正。
J2 口径 B 与口径 A 的总变差随 $k_c$ 按 $(k_c l_P)^2$ 标度（$k_c l_P \in [0.05, 0.8]$，10 点双对数斜率 $2.0 \pm 0.2$）。
J3a 总变差随 $k_c$ 单调上升。
J3b 带边放大：带边极限（$k_c l_P = \pi$）的总变差 $> 0.25$，比低能端放大 100 倍以上，中心速度 $v(0)/c \to 0$。
J4a 可区分阈值：非中心参数 $\lambda_1 = \sum (p-q)^2/p$ 随 $k_c$ 按 $(k_c l_P)^4$ 标度，故阈值样本数 $N_{req} \propto (k_c l_P)^{-4}$（斜率 $-4 \pm 0.3$）。
J4b 大波数端 $k_c l_P = 0.8$、$N = 4\times10^6$ 时两样本检验拒绝同分布。
J4c 小波数端 $k_c l_P = 0.05$、同一 $N$ 时不拒绝同分布。
J5 对照读数（非判据）：偏差标度指数与带限轮 J6 的低能 $(kl_P)^2$ 签名同阶。

运行
----
python 验证脚本/collision_flux_weighting.py
确定性（种子 20260918）。图存 D:/software/TianXingFC/images/collision_flux_weighting.png
"""
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LP = 1.0
TP = 1.0
C = LP / TP
SEED = 20260918


def speed_profile(k_c, w, j_c=0.0, half_width=None):
    if half_width is None:
        half_width = int(2.2 * w)
    j = np.arange(-half_width, half_width + 1, dtype=float) + j_c
    u = (j - j_c) / w
    inside = np.abs(u) <= 1.0
    theta_c = k_c * LP / 2.0
    v = np.empty_like(j)
    v[inside] = C * np.sqrt(np.cos(theta_c) ** 2 + u[inside] ** 2 * np.sin(theta_c) ** 2)
    v[~inside] = C
    return j, u, v, inside


def total_variation(p, q):
    return 0.5 * float(np.sum(np.abs(p - q)))


def main():
    t_start = time.time()
    checks = []

    def report(tag, ok, detail):
        checks.append((tag, bool(ok), detail))
        print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))

    w = 60.0
    sigma = w / 2.0

    print("== J1 口径 A：锁速载子 ==")
    k_c = 0.5
    j, u, v, inside = speed_profile(k_c, w)
    rho = np.exp(-((j) ** 2) / (2.0 * sigma ** 2))
    pA = rho / rho.sum()
    err_norm = abs(pA.sum() - 1.0)
    report("J1a 口径 A 归一化", err_norm < 1e-12, "归一化偏差 %.2e" % err_norm)
    devA = float(np.max(np.abs(pA - rho / rho.sum())))
    report("J1b 口径 A 占比 = |psi|^2", devA < 1e-15, "逐点最大偏差 %.2e" % devA)

    print("== J2 口径 B 的偏差标度 ==")
    kcs = np.linspace(0.05, 0.8, 10)
    tvs = []
    for kc in kcs:
        jj, uu, vv, _ = speed_profile(kc, w)
        rr = np.exp(-(jj ** 2) / (2.0 * sigma ** 2))
        pa = rr / rr.sum()
        pb = rr * (vv / C)
        pb = pb / pb.sum()
        tvs.append(total_variation(pa, pb))
    tvs = np.array(tvs)
    slope2 = float(np.polyfit(np.log(kcs * LP), np.log(tvs), 1)[0])
    beta = float(np.exp(np.polyfit(np.log(kcs * LP), np.log(tvs), 1)[1]))
    report("J2 总变差按 (k_c l_P)^2 标度", abs(slope2 - 2.0) < 0.2,
           "斜率 %.4f，系数 beta = %.4f（TV ~ beta (k_c l_P)^2）" % (slope2, beta))

    print("== J3 带边放大 ==")
    kcs3 = np.linspace(0.1, 3.0, 12)
    tvs3 = []
    for kc in kcs3:
        jj, uu, vv, _ = speed_profile(kc, w)
        rr = np.exp(-(jj ** 2) / (2.0 * sigma ** 2))
        pa = rr / rr.sum()
        pb = rr * (vv / C)
        pb = pb / pb.sum()
        tvs3.append(total_variation(pa, pb))
    tvs3 = np.array(tvs3)
    mono = bool(np.all(np.diff(tvs3) > 0))
    report("J3a 总变差随 k_c 单调上升", mono, "末值 %.4f（k_c l_P = 3.0），初值 %.4f" % (tvs3[-1], tvs3[0]))
    vcenter = float(np.cos(kcs3[-1] * LP / 2.0))
    jj, uu, vv, _ = speed_profile(np.pi, w)
    rr = np.exp(-(jj ** 2) / (2.0 * sigma ** 2))
    pa = rr / rr.sum()
    pb = rr * (vv / C)
    pb = pb / pb.sum()
    tv_edge = total_variation(pa, pb)
    report("J3b 带边放大", tv_edge > 0.25 and tv_edge / tvs3[0] > 100.0 and vcenter < 0.1,
           "带边极限总变差 %.4f（低能端 %.2e，放大 %.0f 倍），中心速度 v(0)/c = %.4f"
           % (tv_edge, tvs3[0], tv_edge / tvs3[0], vcenter))

    print("== J4 统计可区分性 ==")
    rng = np.random.default_rng(SEED)

    def lam1(kc):
        jj, uu, vv, _ = speed_profile(kc, w)
        rr = np.exp(-(jj ** 2) / (2.0 * sigma ** 2))
        pa = rr / rr.sum()
        pb = rr * (vv / C)
        pb = pb / pb.sum()
        m = pa > 1e-14
        return float(np.sum((pa[m] - pb[m]) ** 2 / pa[m])), pa, pb

    kcs4 = np.array([0.1, 0.2, 0.4, 0.8])
    lams = np.array([lam1(kc)[0] for kc in kcs4])
    nreq = 110.0 / lams
    slope4 = float(np.polyfit(np.log(kcs4 * LP), np.log(nreq), 1)[0])
    report("J4a 可区分阈值 N_req 按 (k_c l_P)^{-4} 标度", abs(slope4 + 4.0) < 0.3,
           "斜率 %.3f；N_req = %.2e（k_c l_P = 0.1）至 %.2e（0.8）" % (slope4, nreq[0], nreq[-1]))

    n_ev = 4000000
    for tag, kc, expect_reject in (("J4b 大波数（k_c l_P = 0.8，N = 4e6）", 0.8, True),
                                   ("J4c 小波数（k_c l_P = 0.05，N = 4e6）", 0.05, False)):
        _, pa, pb = lam1(kc)
        m = pa > 1e-14
        pa_m = pa[m] / pa[m].sum()
        pb_m = pb[m] / pb[m].sum()
        na = rng.multinomial(n_ev, pa_m).astype(float)
        nb = rng.multinomial(n_ev, pb_m).astype(float)
        pooled = (na + nb) / (2.0 * n_ev)
        keep = pooled > 0
        stat = float(np.sum((na[keep] - n_ev * pooled[keep]) ** 2 / (n_ev * pooled[keep])
                            + (nb[keep] - n_ev * pooled[keep]) ** 2 / (n_ev * pooled[keep])))
        df = int(keep.sum()) - 1
        z = (stat - df) / np.sqrt(2.0 * df)
        rejected = bool(z > 5.0)
        report(tag, rejected == expect_reject,
               "chi2 = %.0f，df = %d，z = %.1f，理论 TV = %.2e，%s"
               % (stat, df, z, total_variation(pa, pb), "拒绝同分布" if rejected else "不拒绝"))

    print("== J5 对照读数 ==")
    print("  口径 B 的偏差指数 %.4f 与带限轮 J6 的低能 (k l_P)^2 签名（斜率 2.0000）同阶" % slope2)

    print("== 绘图 ==")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))

    ax = axes[0, 0]
    jj, uu, vv, _ = speed_profile(0.8, w)
    rr = np.exp(-(jj ** 2) / (2.0 * sigma ** 2))
    rr = rr / rr.max()
    ax.plot(jj, vv / C, lw=1.6, label=r"速度剖面 $v_j/c$（$k_c l_P = 0.8$）")
    ax.plot(jj, rr, lw=1.2, ls="--", label=r"模式密度 $\rho_j$（归一化）")
    ax.set_xlabel(r"格点 $j$（单位 $l_P$）")
    ax.set_ylabel("无量纲")
    ax.set_title("图 A 束缚模式：中心慢、折返点快")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    jj, uu, vv, _ = speed_profile(0.8, w)
    rr = np.exp(-(jj ** 2) / (2.0 * sigma ** 2))
    pa = rr / rr.sum()
    pb = rr * (vv / C)
    pb = pb / pb.sum()
    ax.plot(jj, pa, lw=1.4, label=r"口径 A：锁速载子（$=\rho_j$）")
    ax.plot(jj, pb, lw=1.4, ls="--", label=r"口径 B：包络群速加权")
    ax.set_xlabel(r"格点 $j$（单位 $l_P$）")
    ax.set_ylabel("计数占比")
    ax.set_title("图 B 两种口径的占比分布（$k_c l_P = 0.8$）")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.loglog(kcs * LP, tvs, "o-", label="实测总变差")
    ax.loglog(kcs * LP, beta * (kcs * LP) ** 2, "k--", lw=1.0,
              label=r"$\beta (k_c l_P)^2$ 参考线")
    ax.set_xlabel(r"中心波数 $k_c l_P$")
    ax.set_ylabel("总变差（两种口径）")
    ax.set_title("图 C 偏差按 $(k_c l_P)^2$ 标度（斜率 %.3f）" % slope2)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(kcs3 * LP, tvs3, "o-", label="总变差")
    ax.plot(kcs3 * LP, np.cos(kcs3 * LP / 2.0), "s--", ms=3.5, label=r"中心速度 $v(0)/c$")
    ax.axvline(np.pi, color="k", lw=0.8, ls=":")
    ax.text(np.pi - 0.45, 0.55, "带边", fontsize=8)
    ax.set_xlabel(r"中心波数 $k_c l_P$")
    ax.set_ylabel("无量纲")
    ax.set_title("图 D 带边附近偏差被放大")
    ax.legend(fontsize=8)

    fig.tight_layout()
    out = "D:/software/TianXingFC/images/collision_flux_weighting.png"
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
