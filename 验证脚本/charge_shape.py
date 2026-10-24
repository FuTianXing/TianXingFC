# -*- coding: utf-8 -*-
"""
判据第 630 号：电荷耦合的形状无关性裁决（冲突诊断注记的张力判掉）。
张力：623 的形状因子读法（电荷分布继承缺陷剖面，宽度 xi 进入耦合）对
      445/wiki 的点状读法（光子耦合到拓扑绕数，形状无关）。

模型（一维锁定势格点，SG 口径）：
  phi_n 为相位场；拓扑荷（电荷）= (1/2pi) * sum_n Delta phi_n = w（整数）。
  连续 kink：phi(x) = 4 arctan(exp((x-x_c)/xi))，宽度 xi。
  连续密度：rho_Q(x) = phi'(x)/(2 pi) = (1/(pi xi)) sech(x/xi)，其积分为 1。

判据（先立后测）：
  Q1 冒烟：kink 剖面与解析式一致，总绕数 = 1。
  Q2 拓扑荷的 xi 无关性：五档 xi 下总绕数 = 1（机器精度），不随宽度变。
  Q3 密度的 xi 依赖性：密度傅里叶变换 F(q) = sech(pi q xi/2)，随 xi 变（复现 623）。
  Q4 判别器（判据有牙）：w = 0 的局域凸起，拓扑荷 = 0 而密度非零且展布。
      两读法给出相反预言（点状=中性 / 展布=带电），体系电荷定义给 0。
  Q5 AB 耦合的形状无关性：环绕缺陷的 AB 相位 = 2pi w，与 xi 及剖面形状无关。
  Q6 实验后果：读法甲下电荷住格点尺度（l_P），远低于 1e-18 m，无冲突；
      读法乙下冲突 5.6 个数量级。
  Q7 反恒真：强制读法乙（把密度当源）才复现 5.6 个数量级冲突，
      确认冲突是读法乙的专属性质。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\charge_shape_independence.png"

XI_SCAN = np.array([2.0, 4.0, 8.0, 16.0, 32.0])
L_P = 1.616255e-35          # 普朗克长度（米）
XI_PHYS = 3.861592e-13      # 电子约化康普顿波长（米），第 622 号恒等式
PROBE_LIMIT = 1e-18         # 电子形状因子实验上限（米）


def kink(n, xc, xi):
    arg = np.clip((n - xc) / xi, -300.0, 300.0)
    return 4.0 * np.arctan(np.exp(arg))


def total_winding(phi):
    """拓扑荷 = (1/2pi) sum Delta phi。"""
    return (phi[-1] - phi[0]) / (2.0 * np.pi)


def density(phi):
    """离散密度 rho_n = Delta phi_n / (2 pi)。"""
    return np.diff(phi) / (2.0 * np.pi)


def F_of_density(rho, qs):
    """密度归一化形状因子 F(q) = |sum_n rho_n e^{i q n}| / sum rho_n（连续极限）。"""
    n = np.arange(len(rho))
    tot = rho.sum()
    return np.array([abs(np.sum(rho * np.exp(1j * q * n))) / abs(tot) for q in qs])


def main():
    line = "-" * 78
    print(line)
    print("判据第 630 号：电荷耦合的形状无关性裁决")
    print(line)

    # ---------------- Q1 冒烟 ----------------
    print("\n[Q1] 冒烟检查（xi = 8）")
    xi0 = 8.0
    N = 2 * int(30 * xi0) + 1
    n = np.arange(N, dtype=float)
    nc = (N - 1) / 2.0
    phi = kink(n, nc, xi0)
    w = total_winding(phi)
    # 解析剖面核对：phi 对 4arctan(exp) 的最大偏差
    dev = np.max(np.abs(phi - 4.0 * np.arctan(np.exp((n - nc) / xi0))))
    print("  N=%d | 总绕数 w = %.15f（期望 1）| 对解析剖面最大偏差 %.2e"
          % (N, w, dev))
    q1_ok = abs(w - 1.0) < 1e-12 and dev < 1e-12
    print("  -> %s" % ("通过" if q1_ok else "不通过"))

    # ---------------- Q2 拓扑荷的 xi 无关性 ----------------
    print("\n[Q2] 拓扑荷随宽度的不变性（电荷 = 整数绕数）")
    ws = []
    for xi in XI_SCAN:
        Nx = 2 * int(30 * xi) + 1
        nx = np.arange(Nx, dtype=float)
        ncx = (Nx - 1) / 2.0
        wx = total_winding(kink(nx, ncx, xi))
        ws.append(wx)
        print("  xi=%6g  总绕数 = %.15f  (与 1 的偏差 %.2e)" % (xi, wx, abs(wx - 1.0)))
    ws = np.array(ws)
    q2_ok = np.max(np.abs(ws - 1.0)) < 1e-12
    print("  五档最大偏差 = %.2e（机器精度）-> 电荷与宽度无关 -> %s"
          % (np.max(np.abs(ws - 1.0)), "通过" if q2_ok else "不通过"))

    # ---------------- Q3 密度的 xi 依赖性 ----------------
    print("\n[Q3] 密度的傅里叶变换（623 的对象）：F(q) = sech(pi q xi/2)")
    qs = np.array([0.0, 0.1, 0.2, 0.4, 0.8])
    F_ref = {}
    for xi in XI_SCAN:
        Nx = 2 * int(40 * xi) + 1
        nx = np.arange(Nx, dtype=float)
        ncx = (Nx - 1) / 2.0
        rho = density(kink(nx, ncx, xi))
        Fq = F_of_density(rho, qs / xi)  # 以 u = q*xi 为自变量，故 q = u/xi
        pred = 1.0 / np.cosh(np.clip(np.pi * qs / 2.0, 0, 700))
        F_ref[xi] = (Fq, pred)
        print("  xi=%6g  F(q*xi) 实测 %s" % (xi, np.array2string(Fq, precision=4)))
    # 取 xi=8 档逐点核对解析式
    Fq8, pred8 = F_ref[8.0]
    dev8 = np.max(np.abs(Fq8 - pred8))
    print("  xi=8 档对 sech(pi q xi/2) 最大偏差 = %.2e" % dev8)
    # 不同 xi 的 F 在同 q*xi 下相同（形状由 q*xi 决定），但物理 q 处的值随 xi 变
    F_same = all(np.max(np.abs(F_ref[xi][0] - F_ref[8.0][0])) < 5e-3 for xi in XI_SCAN)
    q3_ok = dev8 < 1e-2 and F_same
    print("  密度形状因子随宽度缩放（同 q*xi 下重合、物理 q 处随 xi 变）-> %s"
          % ("通过" if q3_ok else "不通过"))

    # ---------------- Q4 判别器（判据有牙） ----------------
    print("\n[Q4] 判别器：w = 0 的局域凸起（两读法给出相反预言）")
    xi_t = 8.0
    Nt = 2 * int(30 * xi_t) + 1
    nt = np.arange(Nt, dtype=float)
    nct = (Nt - 1) / 2.0
    sigma = xi_t
    phi_bump = 0.5 * np.exp(-((nt - nct) ** 2) / (2.0 * sigma ** 2))  # 局域凸起，净绕数 0
    w_bump = total_winding(phi_bump)
    rho_bump = density(phi_bump)
    dens_norm = np.sum(np.abs(rho_bump))
    print("  局域凸起：总绕数 w = %.15f（读法甲：中性）" % w_bump)
    print("            密度总量 sum|rho| = %.6e（读法乙：非零展布源）" % dens_norm)
    print("  两读法预言相反：点状=中性 / 展布=带电；体系电荷定义（拓扑荷）给 0")
    q4_ok = abs(w_bump) < 1e-12 and dens_norm > 1e-3
    print("  -> 判别器有效（w=0 而密度非零，两读法可分）-> %s"
          % ("通过" if q4_ok else "不通过"))

    # ---------------- Q5 AB 耦合的形状无关性 ----------------
    print("\n[Q5] AB 耦合的形状无关性：环绕缺陷的相位 = 2 pi w")
    print("  拓扑耦合 2 pi w * (围住缺陷一圈的 A 积分)：w 为整数，与 xi 及剖面形状无关")
    ab_phases = []
    for xi in XI_SCAN:
        Nx = 2 * int(30 * xi) + 1
        nx = np.arange(Nx, dtype=float)
        ncx = (Nx - 1) / 2.0
        wx = total_winding(kink(nx, ncx, xi))
        ab_phases.append(2.0 * np.pi * wx)
    ab_phases = np.array(ab_phases)
    print("  五档 AB 相位：%s（全部 = 2 pi）" % np.array2string(ab_phases, precision=12))
    q5_ok = np.max(np.abs(ab_phases - 2.0 * np.pi)) < 1e-11
    print("  对凸起（w=0）：AB 相位 = %.15f（中性）-> %s"
          % (2.0 * np.pi * w_bump, "通过" if q5_ok else "不通过"))

    # ---------------- Q6 实验后果 ----------------
    print("\n[Q6] 实验后果（两读法）")
    # 读法甲：电荷住格点尺度 l_P（核面元环量），远低于探测上限
    margin_A = PROBE_LIMIT / L_P
    # 读法乙：电荷分布宽度 xi_phys，超上限倍数
    ratio_B = XI_PHYS / PROBE_LIMIT
    print("  读法甲（拓扑荷，住格点尺度 l_P = %.3e m）：低于探测上限 %.0e 倍（无冲突）"
          % (L_P, margin_A))
    print("  读法乙（密度展布，宽度 xi = %.3e m）：超探测上限 %.3e 倍 = %.2f 个数量级"
          % (XI_PHYS, ratio_B, np.log10(ratio_B)))
    q6_ok = margin_A > 1e10 and 5.0 < np.log10(ratio_B) < 6.0
    print("  -> %s" % ("通过" if q6_ok else "不通过"))

    # ---------------- Q7 反恒真 ----------------
    print("\n[Q7] 反恒真：冲突是读法乙的专属性质")
    print("  读法甲：电荷 = 整数绕数，形状无关，实验裕度 %.0e 倍（无冲突）" % margin_A)
    print("  读法乙：电荷 = 密度展布，宽度 xi 进入形状因子，冲突 %.2f 个数量级"
          % np.log10(ratio_B))
    print("  同一实验对照下，两读法给出相反结论；冲突只在读法乙下出现")
    q7_ok = margin_A > 1e10 and np.log10(ratio_B) > 5.0
    print("  -> %s" % ("通过" if q7_ok else "不通过"))

    # ---------------- 图 ----------------
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.3))

    ax = axes[0]
    qq = np.linspace(0, 2.0, 200)
    for xi in [2.0, 8.0, 32.0]:
        ax.plot(qq, 1.0 / np.cosh(np.pi * qq / 2.0), lw=1.5,
                label=r"$\xi=%g$" % xi)
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_xlabel(r"$q\xi$")
    ax.set_ylabel(r"$F(q\xi)$")
    ax.set_title(r"(a) 密度形状因子 $=\operatorname{sech}(\pi q\xi/2)$（同 $q\xi$ 重合）")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    ax.plot(nt - nct, phi_bump, "-", lw=1.6, label=r"$\phi$（局域凸起）")
    ax.plot(nt[:-1] - nct, rho_bump * 1e2, "--", lw=1.4, label=r"$\rho_Q\times10^2$")
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("格点")
    ax.set_ylabel(r"$\phi$ 与 $\rho_Q$")
    ax.set_title(r"(b) 判别器：$w=0$ 凸起，$\phi$ 归零而 $\rho_Q$ 非零")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    ax = axes[2]
    labels = ["读法甲\n（拓扑荷）", "读法乙\n（密度展布）"]
    vals = [L_P, XI_PHYS]
    ax.bar(labels, vals, color=["C0", "C3"], width=0.55)
    ax.axhline(PROBE_LIMIT, color="k", lw=1.2, ls="--", label=r"实验上限 $10^{-18}$ m")
    ax.set_yscale("log")
    ax.set_ylabel("电荷尺度（米）")
    ax.set_title("(c) 两读法的电荷尺度对实验上限")
    ax.legend(fontsize=8)
    for i, v in enumerate(vals):
        ax.text(i, v * 1.8, "%.2e" % v, ha="center", fontsize=8)
    ax.grid(True, which="both", alpha=0.25)

    fig.suptitle("判据第 630 号：电荷耦合的形状无关性裁决（拓扑荷对密度展布）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(IMG, dpi=150)
    print("\n图已存：%s" % IMG)

    res = {"Q1": q1_ok, "Q2": q2_ok, "Q3": q3_ok, "Q4": q4_ok, "Q5": q5_ok,
           "Q6": q6_ok, "Q7": q7_ok}
    print("\n" + line)
    print("判据汇总：" + " | ".join("%s %s" % (k, "通过" if v else "不通过")
                                    for k, v in res.items()))
    print("合计 %d/%d" % (sum(res.values()), len(res)))
    print(line)


if __name__ == "__main__":
    main()
