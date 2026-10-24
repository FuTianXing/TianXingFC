# -*- coding: utf-8 -*-
"""
判据第 629 号：Born 判据物理档重估（第 624 号）的输入审计。
审计对象：多体论文 §4.1 的势深估计 U_0 ~ hbar*omega_0*(xi/l_P)^3（权重体积图像）
          与同一论文 §4.3 的精确相互作用 V(d) = M*sum_N[...]e^{-N theta} 的档位一致性。

模型（一维锁定势格点，SG 口径，逐字沿用第 627 号与多体论文 §4.3 设定）：
  E = sum_n [ (k_b/2)(theta_{n+1}-theta_n)^2 + K (1 - cos theta_n) ]
  单 kink：theta = 4 arctan(exp((x-x_c)/xi))，xi = sqrt(k_b/K)，质量 M = 8 sqrt(k_b K)，
          能隙 omega_0 = sqrt(K)。
  §4.3 相互作用：V(d) = E[w_1 + w_2] - 2 E[w_1]（两条 kink 的叠加，不做松弛，
                与 §4.3 定义逐字一致）。

判据（先立后测）：
  Q1 冒烟：V(d) 有限、为正（排斥）、随 d 单调衰减。
  Q2 标度：V(d = xi) 随 xi 的幂律指数为 -1（即 V ~ hbar*omega_0，无 n^3 因子）。
  Q3 §4.1 估计审计：比值 U_0^{§4.1}/U_0^{§4.3} 在模型档为 O(1)，在物理档为 n^3 量级。
  Q4 能量上界：物理档 U_0^{§4.1} 超两缺陷总能量（2M）与体系紫外截断 E_max 的数量级。
  Q5 修正 Born 参数：以 §4.3 精确幅值代入，物理档 Born 比值为 O(1)（而非 1e68）。
  Q6 判据有牙：翻转由 n^3 因子承载；去掉该因子翻转消失（n^3 是承重输入）。

单位：格点单位（hbar = c = l_P = 1，k_b = 1）。物理档 n = m_P/m_e = 2.389222e22。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\born_input_audit.png"

KB = 1.0
N_MODEL = 2.25                 # 模型档宽度（K = 0.005 演示参数，第 259 条注记口径）
N_PHYS = 2.389222e22           # 物理档宽度（第 622 号恒等式 = m_P/m_e，格）
XI_SCAN = np.array([2.0, 4.0, 8.0, 16.0, 32.0])


def thomas(a, b, c, d):
    n = len(b)
    cp = np.zeros(n)
    dp = np.empty(n)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i - 1] * cp[i - 1]
        cp[i] = (c[i] / m) if (i < n - 1) else 0.0
        dp[i] = (d[i] - a[i - 1] * dp[i - 1]) / m
    x = np.empty(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def relax(th, kb, K, iters=2000):
    th = th.copy()
    res = 1.0
    for _ in range(iters):
        R = kb * (th[2:] + th[:-2] - 2 * th[1:-1]) - K * np.sin(th[1:-1])
        res = np.max(np.abs(R))
        if res < 1e-13:
            break
        diag = -2 * kb - K * np.cos(th[1:-1])
        off = kb * np.ones(len(R) - 1)
        th[1:-1] += thomas(off, diag, off, -R)
    return th, res


def energy(th, kb, K):
    d = np.diff(th)
    return float(np.sum(kb / 2 * d ** 2) + np.sum(K * (1 - np.cos(th))))


def kink_profile(n, xc, xi):
    arg = np.clip((n - xc) / xi, -300.0, 300.0)
    return 4.0 * np.arctan(np.exp(arg))


def overlap_energy(xi, d_mult, kb=KB):
    """V(d) = E[w_1+w_2] - 2 E[w_1]，两条 kink 叠加（§4.3 定义）。"""
    K = kb / xi ** 2
    N = 2 * int(60 * xi) + 1
    n = np.arange(N, dtype=float)
    nc = (N - 1) / 2.0
    th1, res1 = relax(kink_profile(n, nc, xi), kb, K)
    E1 = energy(th1, kb, K)
    d = d_mult * xi
    w = kink_profile(n, nc - d / 2.0, xi) + kink_profile(n, nc + d / 2.0, xi)
    V = energy(w, kb, K) - 2.0 * E1
    return V, E1, res1


def main():
    line = "-" * 78
    print(line)
    print("判据第 629 号：Born 判据物理档重估的输入审计（§4.1 vs §4.3）")
    print(line)

    # ---------------- Q1 冒烟 ----------------
    print("\n[Q1] 冒烟检查（xi = 8）")
    xi0 = 8.0
    Vd = []
    for mult in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
        V, E1, res = overlap_energy(xi0, mult)
        Vd.append(V)
        print("  d = %.1f xi  V(d) = %+.6e  (V*xi = %+.5f)  单 kink 残差 %.1e"
              % (mult, V, V * xi0, res))
    Vd = np.array(Vd)
    mono = np.all(np.diff(Vd) < 0)
    q1_ok = (np.all(np.isfinite(Vd)) and Vd[0] > 0 and mono)
    print("  有限=%s | 全正（排斥）=%s | 随 d 单调衰减=%s -> %s"
          % (np.all(np.isfinite(Vd)), bool(np.all(Vd > 0)), mono,
             "通过" if q1_ok else "不通过"))

    # ---------------- Q2 标度 ----------------
    print("\n[Q2] 相互作用幅值的 xi 标度（d = xi 处）")
    Vs, Ms, W0s = [], [], []
    for xi in XI_SCAN:
        V, E1, res = overlap_energy(xi, 1.0)
        K = KB / xi ** 2
        w0 = np.sqrt(K)
        Vs.append(V)
        Ms.append(E1)
        W0s.append(w0)
        print("  xi=%6g  V=%.6e  V*xi=%.5f  M=%.6e  omega_0=%.6e  V/omega_0=%.5f  V/M=%.5f"
              % (xi, V, V * xi, E1, w0, V / w0, V / E1))
    Vs = np.array(Vs)
    Ms = np.array(Ms)
    W0s = np.array(W0s)
    slope_V = np.polyfit(np.log(XI_SCAN), np.log(Vs), 1)[0]
    slope_M = np.polyfit(np.log(XI_SCAN), np.log(Ms), 1)[0]
    spread = (Vs * XI_SCAN).max() / (Vs * XI_SCAN).min() - 1.0
    q2_ok = abs(slope_V + 1.0) <= 0.05 and spread < 0.02
    print("  V 的幂律指数 = %.4f（期望 -1，容差 0.05）| V*xi 相对散布 = %.2e（< 0.02）-> %s"
          % (slope_V, spread, "通过" if q2_ok else "不通过"))
    print("  M（孤子质量）的幂律指数 = %.4f（同族，M ~ omega_0 ~ 1/xi，无 n^3）" % slope_M)
    print("  结论：V ~ hbar*omega_0（量级为缺陷自身能量），不含权重体积因子 n^3")

    # ---------------- Q3 §4.1 估计审计 ----------------
    print("\n[Q3] §4.1 势深估计 vs §4.3 精确幅值的档位比")
    cV = float(np.mean(Vs * XI_SCAN))          # V = cV / xi
    print("  §4.3 精确幅值：V = c_V / xi，c_V = %.4f（格点单位）" % cV)
    print("  §4.1 估计：U_0 = omega_0 * n^3 = (1/xi) * xi^3 = xi^2（格点单位）")
    ratio_model = N_MODEL ** 3 / cV
    ratio_phys = N_PHYS ** 3 / cV
    print("  比值 U_0^{§4.1}/U_0^{§4.3} = xi^3 / c_V")
    print("    模型档 n = %.2f : %.3f（O(1)，被 §4.1 的 O(1) 待定吞掉）"
          % (N_MODEL, ratio_model))
    print("    物理档 n = %.4e : %.4e（= n^3 量级）" % (N_PHYS, ratio_phys))
    q3_ok = (1.0 < ratio_model < 10.0) and (ratio_phys > 1e60)
    print("  判定（模型档 O(1) 且物理档 > 1e60）-> %s" % ("通过" if q3_ok else "不通过"))

    # ---------------- Q4 能量上界 ----------------
    print("\n[Q4] 能量上界：物理档 U_0^{§4.1} 对缺陷总能量与紫外截断")
    U0_41_phys = N_PHYS ** 2                    # 格点单位
    M_phys = cV / N_PHYS if False else (8.0 * KB / N_PHYS)   # M = 8/xi（格点单位）
    E_max = 2.0 * np.sqrt(3.0)                  # 体系能带上限（格点单位，第 8.3 节）
    print("  U_0^{§4.1}(物理档) = n^2 = %.4e（格点单位）" % U0_41_phys)
    print("  单缺陷质量 M = 8/xi = %.4e；两缺陷总能量 2M = %.4e" % (M_phys, 2 * M_phys))
    print("  体系紫外截断 E_max = 2 sqrt(3) = %.4f（元相子基模同量级）" % E_max)
    print("  U_0^{§4.1} / (2M) = %.4e（相互作用能为两缺陷总能量的多少倍）"
          % (U0_41_phys / (2 * M_phys)))
    print("  U_0^{§4.1} / E_max = %.4e" % (U0_41_phys / E_max))
    q4_ok = (U0_41_phys / (2 * M_phys)) > 1e60
    print("  模型无关上界：相互作用能 <= 参与缺陷的总能量；此处超界 %.1e 倍 -> %s"
          % (U0_41_phys / (2 * M_phys), "通过（上界被违反）" if q4_ok else "不通过"))

    # ---------------- Q5 修正 Born 参数 ----------------
    print("\n[Q5] 以 §4.3 精确幅值修正后的 Born 参数（物理档，两口径）")
    # Born 比值 R = U_0 / [hbar^2/(m xi^2)]；格点单位下 hbar^2/(m xi^2) = omega_0
    print("  Born 分母 hbar^2/(m xi^2)：物理档 xi = hbar/(mc) 时 = m c^2 = omega_0（格点单位）")
    # 口径甲：d = xi 处的相互作用幅值 V = c_V/xi = c_V * omega_0
    R_41 = N_PHYS ** 3                          # = U_0^{§4.1}/omega_0
    R_43a = cV                                  # = U_0^{§4.3}/omega_0（d=xi 口径）
    # 口径乙：§4.3 的 N=1 截断幅值（论文自标 §4.1 形式 = N=1 截断），V(0)|_{N=1} = 4M
    M_phys_unit = 8.0 * KB / N_PHYS             # M = 8 sqrt(k_b K) = 8/xi
    U0_N1 = 4.0 * M_phys_unit                   # = 32 omega_0
    R_43b = U0_N1 / (KB / N_PHYS)               # 除以 omega_0 = sqrt(K) = 1/xi
    print("  口径甲（d = xi 幅值）：R^{§4.3} = %.4f，含 4pi^2 约定因子 R_2 = %.4f"
          % (R_43a, 4 * np.pi ** 2 * R_43a))
    print("  口径乙（N=1 截断幅值 4M = 32 omega_0）：R^{§4.3} = %.4f，含 4pi^2 因子 R_2 = %.4f"
          % (R_43b, 4 * np.pi ** 2 * R_43b))
    print("  第 624 号口径：R^{§4.1} = n^3 = %.4e，R_2 = %.4e"
          % (R_41, 4 * np.pi ** 2 * R_41))
    q5_ok = (R_41 > 1e60) and (0.5 < 4 * np.pi ** 2 * R_43a < 1e3) \
        and (0.5 < 4 * np.pi ** 2 * R_43b < 1e4)
    print("  修正后物理档 Born 参数为 O(1) 至 O(10^3)（临界区，非 1e68 的强势区），"
          "两口径同向 -> %s" % ("通过" if q5_ok else "不通过"))

    # ---------------- Q6 判据有牙 ----------------
    print("\n[Q6] 判据有牙：翻转由 n^3 因子承载")
    # 去掉 n^3：用 §4.3 精确幅值代入 624 的完整原式
    # 624 原式 R_2 = 4pi^2 n^5 (m/m_P)^2；把 U_0 的 n^3 换成 c_V/omega_0 的常数因子
    # 等价地：R_2^{修正} = R_2^{624} * (c_V / n^3)
    R2_624 = 4 * np.pi ** 2 * N_PHYS ** 3
    R2_fix = R2_624 * cV / N_PHYS ** 3
    ratio_alg = R2_624 / R2_fix
    expect = N_PHYS ** 3 / cV
    print("  第 624 号读数 R_2 = %.4e" % R2_624)
    print("  把 U_0 的 n^3 换成 §4.3 精确幅值后 R_2 = %.4f" % R2_fix)
    print("  增强因子 R_2^{624}/R_2^{修正} = %.6e；代数预期 n^3/c_V = %.6e；相对差 = %.2e"
          % (ratio_alg, expect, abs(ratio_alg - expect) / expect))
    # 判据：增强因子严格等于 n^3/c_V（代数恒等，机器精度），去掉后落入临界区
    q6_ok = (abs(ratio_alg - expect) / expect < 1e-12) and (R2_fix < 1e3)
    print("  结论：翻转严格由 n^3 因子承载（增强因子对 n^3/c_V 机器精度重合），去掉即落入"
          "临界区（O(1) 至 O(100)）-> %s" % ("通过" if q6_ok else "不通过"))

    # ---------------- Q7 约定核对与巧合点 ----------------
    print("\n[Q7] §4.3 级数约定核对与 §4.1 巧合点")
    # (a) 级数对数值叠加能，核对约定。残差来自格点离散化（O(1/xi^2)），故做收敛检验：
    #     偏差应随 xi 翻倍而降约 4 倍（协议伪影特征），而非固定偏差（约定不一致特征）。
    def series_vs_numeric(xi_test):
        M_u = 8.0 * KB / xi_test
        errs = []
        for mult in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
            th = mult
            N = np.arange(1, 80)
            eps = np.where(N % 2 == 1, 1.0, 0.0)
            terms = (4 * N - 2 * (N ** 2 - eps) * th) * np.exp(-N * th)
            V_ser = M_u * terms.sum()
            V_num = overlap_energy(xi_test, mult)[0]
            errs.append(abs(V_ser - V_num) / abs(V_num))
        return np.max(errs)
    d8 = series_vs_numeric(8.0)
    d16 = series_vs_numeric(16.0)
    d32 = series_vs_numeric(32.0)
    print("  (a) 级数对数值叠加能最大相对偏差：xi=8 -> %.3e，xi=16 -> %.3e，xi=32 -> %.3e"
          % (d8, d16, d32))
    print("      偏差比 d8/d16 = %.2f、d16/d32 = %.2f（期望约 4，格点离散化 O(1/xi^2) 特征）"
          % (d8 / d16, d16 / d32))
    q7a_ok = (d8 / d16 > 2.5) and (d16 / d32 > 2.5) and (d32 < 1e-3)
    print("      残差为协议伪影（随 xi 按 1/xi^2 收敛），约定逐位一致 -> %s"
          % ("通过" if q7a_ok else "不通过"))
    M_unit = 8.0 * KB / xi0
    n_star = (4.0 * M_unit / (KB / xi0)) ** (1.0 / 3.0)
    print("  (b) §4.3 首阶（N=1）幅值 = 4M = 4*%.4f = %.4f（格点单位）" % (M_unit, 4 * M_unit))
    print("      §4.1 估计 omega_0 n^3 等于 4M 当且仅当 n^3 = 4M/omega_0 = %.4f，即 n* = %.4f"
          % (4 * M_unit / (KB / xi0), n_star))
    print("      模型档 n = %.2f 落在巧合点附近（相对差 %.1f%%），故该误标从未暴露"
          % (N_MODEL, abs(N_MODEL - n_star) / n_star * 100))
    q7b_ok = abs(n_star - 3.17) < 0.05
    print("  (c) 巧合点 n* = %.4f 为 O(1)（非普遍成立，n 大时估计必然失效）-> %s"
          % (n_star, "通过" if q7b_ok else "不通过"))
    q7_ok = q7a_ok and q7b_ok

    # ---------------- 图 ----------------
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.3))

    ax = axes[0]
    ds = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0])
    for xi in [4.0, 8.0, 16.0]:
        Vv = [overlap_energy(xi, m)[0] for m in ds]
        ax.semilogy(ds, np.abs(Vv), "o-", lw=1.3, label="xi=%g" % xi)
    ax.set_xlabel(r"间距 $d/\xi$")
    ax.set_ylabel(r"$|V(d)|$（格点单位）")
    ax.set_title("(a) 两缺陷叠加能随间距衰减")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.25)

    ax = axes[1]
    ax.loglog(XI_SCAN, Vs, "o-", lw=1.5, label=r"$V(d=\xi)$ 实测")
    ax.loglog(XI_SCAN, cV / XI_SCAN, "k--", lw=1.2, label=r"$c_V/\xi \sim \hbar\omega_0$")
    ax.loglog(XI_SCAN, XI_SCAN ** 2, "r:", lw=1.2, label=r"$\S4.1$ 估计 $\hbar\omega_0 n^3\sim\xi^2$")
    ax.set_xlabel(r"缺陷宽度 $\xi$（格）")
    ax.set_ylabel("相互作用幅值")
    ax.set_title("(b) 精确幅值随 $1/\\xi$ 衰减，§4.1 估计随 $\\xi^2$ 增长")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.25)

    ax = axes[2]
    labels = ["模型档\nn=2.25", "物理档\nn=2.4e22"]
    vals = [ratio_model, ratio_phys]
    ax.bar(labels, vals, color=["C0", "C3"], width=0.55)
    ax.set_yscale("log")
    ax.set_ylabel(r"$U_0^{\S4.1}/U_0^{\S4.3}$")
    ax.set_title("(c) §4.1 估计相对精确幅值的超估倍数")
    for i, v in enumerate(vals):
        ax.text(i, v * 1.5, "%.3g" % v, ha="center", fontsize=9)
    ax.grid(True, which="both", alpha=0.25)

    fig.suptitle("判据第 629 号：Born 物理档重估的输入审计（§4.1 权重体积 $n^3$ 为伪因子）",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(IMG, dpi=150)
    print("\n图已存：%s" % IMG)

    # ---------------- 汇总 ----------------
    res = {"Q1": q1_ok, "Q2": q2_ok, "Q3": q3_ok, "Q4": q4_ok, "Q5": q5_ok, "Q6": q6_ok,
           "Q7": q7_ok}
    print("\n" + line)
    print("判据汇总：" + " | ".join("%s %s" % (k, "通过" if v else "不通过")
                                    for k, v in res.items()))
    print("合计 %d/%d" % (sum(res.values()), len(res)))
    print(line)


if __name__ == "__main__":
    main()
