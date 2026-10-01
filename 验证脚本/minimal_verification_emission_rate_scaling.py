# -*- coding: utf-8 -*-
"""发射率标度判据（P1-50 耦合通道的结构层验证，README 第 524 条）。

模型层设定（声明）：一维紧邻 hopping 链（带宽 [-2,2]）作为真空模式连续谱的
最小实现，局域缺陷振子（能量 omega0）与链中央格点局域耦合 g。这是
Weisskopf-Wigner / Fermi 黄金规则的标准玩具，本判据锁定的是"发射率 =
2 pi |g|^2 x 态密度"这一结构定律与态密度的维数效应，不裁决体系的耦合
通道（体系通道按方案注记第九节定 cos 高阶项，机制本身仍属 P1-50 开放）。

判据
----
E1（黄金规则）：三档带内缺陷频率 omega0 = 0.5, 1.0, 1.5，实测 Gamma
    （P_d(t) 早期指数拟合）对 2 pi g^2 rho_+(omega0) 之比在 [0.8, 1.2]
    （20% 预算：拟合窗偏差 + 有限链谱密度修正 O(1/N) + 非指数修正 O(g^2)）。
E2（带外反例，FAIL 预期，EP5 齿）：omega0 = 2.5 > 带边 2，发射通道不存在，
    缺陷与场杂化出束缚态，P_d(t) 后段平均 > 0.5（对照带内同窗 < 0.05）。
E3（态密度维数效应）：1D 范霍夫奇点使 rho_+(omega) 向带边发散，
    Gamma(1.9)/Gamma(1.0) 理论 = rho(1.9)/rho(1.0) = 2.77，
    实测比在 [2.08, 3.46]（正负 25%）。

容差来源：20% 与 25% 从拟合窗系统偏差、有限链谱密度分辨率（eta=0.02 高斯
展宽、N=1200）与 Wigner-Weisskopf 非指数修正 O(g^2) 推导；E2 的 0.5 从
束缚态权重（g=0.15 时约 0.96）推导，裕度充分。
判据可逆性：判据稳健（黄金规则是结构定律，换 N、换档位不翻）。
"""

import numpy as np

N = 1200
G_COUPLING = 0.15
DT = 0.2
T_END = 200.0
FIT_LO, FIT_HI = 0.25, 0.9


def rho_plus(w):
    """一维紧邻链正频态密度（解析连续极限）。"""
    return 1.0 / (np.pi * np.sqrt(4.0 - w * w))


def build_H(omega0):
    """单激发扇区：链 hopping + 缺陷对角 + 局域耦合。实对称。"""
    n = N + 1
    diags = np.zeros(n)
    diags[:N] = 0.0
    diags[N] = omega0
    off = -G_COUPLING * 0  # 占位，下面显式构建 hopping 与耦合
    H = np.zeros((n, n))
    for i in range(N - 1):
        H[i, i + 1] = H[i + 1, i] = -1.0
    H[N, N] = omega0
    n0 = N // 2
    H[N, n0] = H[n0, N] = G_COUPLING
    return H


def gamma_measured(omega0, H):
    """P_d(t) = |<d|exp(-iHt)|d>|^2 的早期指数拟合。"""
    w, v = np.linalg.eigh(H)
    amp = v[N, :] ** 2  # |<d|m>|^2
    ts = np.arange(0.0, T_END + DT, DT)
    pd = np.array([np.sum(amp * np.cos(w * t)) for t in ts]) ** 2 \
        + np.array([np.sum(amp * np.sin(w * t)) for t in ts]) ** 2
    win = (pd >= FIT_LO) & (pd <= FIT_HI) & (ts > 0)
    if win.sum() < 5:
        return np.nan, pd, ts
    coef = np.polyfit(ts[win], np.log(pd[win]), 1)
    return -coef[0], pd, ts


def main():
    print("=== 冒烟检查 ===")
    H_test = build_H(1.0)
    w = np.linalg.eigvalsh(H_test)
    print(f"谱范围实测 [{w.min():.4f}, {w.max():.4f}] | "
          f"期望约 [-2, 2] 加缺陷杂化（g={G_COUPLING}）")
    print(f"理论 Gamma(1.0) = 2*pi*g^2*rho(1.0) = "
          f"{2 * np.pi * G_COUPLING ** 2 * rho_plus(1.0):.5f}")

    print("\n=== E1：黄金规则（三档带内频率） ===")
    e1_pass = True
    gammas = {}
    curves = {}
    for w0 in (0.5, 1.0, 1.5):
        H = build_H(w0)
        gm, pd, ts = gamma_measured(w0, H)
        gt = 2.0 * np.pi * G_COUPLING ** 2 * rho_plus(w0)
        ratio = gm / gt
        ok = 0.8 <= ratio <= 1.2
        e1_pass &= ok
        gammas[w0] = gm
        curves[w0] = (ts, pd)
        print(f"omega0={w0:.1f}: Gamma 实测 {gm:.5f} | 理论 {gt:.5f} | "
              f"比值 {ratio:.4f}（[0.8,1.2] {'PASS' if ok else 'FAIL'}）")
    print(f"E1（黄金规则）: {'PASS' if e1_pass else 'FAIL'}")

    print("\n=== E3：态密度维数效应（1D 范霍夫奇点） ===")
    if np.isnan(gammas.get(1.9, np.nan)):
        H = build_H(1.9)
        gm19, pd19, ts19 = gamma_measured(1.9, H)
        gammas[1.9] = gm19
        curves[1.9] = (ts19, pd19)
    r_meas = gammas[1.9] / gammas[1.0]
    r_theo = rho_plus(1.9) / rho_plus(1.0)
    ok3 = 0.75 * r_theo <= r_meas <= 1.25 * r_theo
    print(f"Gamma(1.9)/Gamma(1.0) 实测 = {r_meas:.4f} | "
          f"理论 rho 比 = {r_theo:.4f} | "
          f"（[0.75x, 1.25x] {'PASS' if ok3 else 'FAIL'}）")
    print(f"E3: {'PASS' if ok3 else 'FAIL'}")

    print("\n=== E2：带外反例（FAIL 预期，EP5 齿） ===")
    H = build_H(2.5)
    w, v = np.linalg.eigh(H)
    amp = v[N, :] ** 2
    ts_full = np.arange(0.0, T_END + DT, DT)
    pd_full = np.array([np.sum(amp * np.cos(w * t)) for t in ts_full]) ** 2 \
        + np.array([np.sum(amp * np.sin(w * t)) for t in ts_full]) ** 2
    ts = np.arange(150.0, T_END + DT, DT)
    pd_late = pd_full[ts_full >= 150.0]
    late_mean = pd_late.mean()
    # 带内对照（omega0=1.0 同窗）
    w1, v1 = np.linalg.eigh(build_H(1.0))
    amp1 = v1[N, :] ** 2
    pd1_late = np.array([np.sum(amp1 * np.cos(w1 * t)) for t in ts]) ** 2 \
        + np.array([np.sum(amp1 * np.sin(w1 * t)) for t in ts]) ** 2
    inband_mean = pd1_late.mean()
    ok2 = (late_mean > 0.5) and (inband_mean < 0.05)
    print(f"带外 omega0=2.5: P_d 后段平均 实测 {late_mean:.4f} | 期望 > 0.5（束缚态不衰）")
    print(f"带内 omega0=1.0 对照: P_d 后段平均 实测 {inband_mean:.4f} | 期望 < 0.05")
    print(f"E2: {'PASS' if ok2 else 'FAIL'}")

    n_pass = sum([e1_pass, ok3, ok2])
    print(f"\n判据 {n_pass}/3（E1/E3/E2，E2 兼 EP5 齿：移出能带发射即消失）")
    print("边界声明：本验证锁定结构定律（发射率=2pi|g|^2x态密度）与 1D")
    print("范霍夫效应，不裁决体系耦合通道（cos 高阶项，见方案注记第九节），")
    print("不给出爱因斯坦 A 系数数值（需 3D 态密度与偶极矩阵元，属 P1-50 余项）。")

    # 绘图（图存 images/，注记与论文嵌入）
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    colors = {0.5: "tab:blue", 1.0: "tab:green", 1.5: "tab:orange",
              1.9: "tab:purple"}
    for w0, (tc, pc) in curves.items():
        axes[0].semilogy(tc, np.clip(pc, 1e-12, None), color=colors[w0],
                         lw=1.2, label=f"$\\omega_0={w0}$ (in band)")
    axes[0].semilogy(ts_full, np.clip(pd_full, 1e-12, None), "r-",
                     lw=1.2, label="$\\omega_0=2.5$ (bound, out of band)")
    axes[0].set_xlabel("$t$"); axes[0].set_ylabel("$P_d(t)$")
    axes[0].legend(fontsize=7); axes[0].set_title(
        "defect excitation decay vs out-of-band survival")
    ws = sorted(gammas)
    gt_list = [2.0 * np.pi * G_COUPLING ** 2 * rho_plus(w0) for w0 in ws]
    gm_list = [gammas[w0] for w0 in ws]
    xpos = np.arange(len(ws))
    axes[1].bar(xpos - 0.2, gm_list, width=0.4, label="measured $\\Gamma$")
    axes[1].bar(xpos + 0.2, gt_list, width=0.4, label="golden rule $2\\pi g^2\\rho$")
    axes[1].set_xticks(xpos, [f"$\\omega_0={w0}$" for w0 in ws])
    axes[1].set_ylabel("$\\Gamma$"); axes[1].legend(fontsize=8)
    axes[1].set_title("golden rule check (E1)")
    wgrid = np.linspace(0.05, 1.99, 300)
    axes[2].plot(wgrid, rho_plus(wgrid), "b-")
    for w0 in ws:
        axes[2].axvline(w0, color=colors[w0], lw=0.8, ls=":")
    axes[2].set_xlabel("$\\omega$"); axes[2].set_ylabel(
        "$\\rho_+(\\omega)$")
    axes[2].set_title("1D van Hove density of states (E3)")
    fig.tight_layout()
    fig.savefig(r"D:\software\TianXingFC\images\emission_rate_scaling_524.png",
                dpi=150)
    print("图已存 images/emission_rate_scaling_524.png")


if __name__ == "__main__":
    main()
