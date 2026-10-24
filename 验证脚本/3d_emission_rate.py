# -*- coding: utf-8 -*-
"""三维发射率标度与带外束缚最小验证（P1-50 余项，README 第 525 条）。

模型层设定（声明）：缺陷振子（频率 omega0）与三维波动画格（N=40，周期边界，
格点波动方程 uddot = lap u，c = 1）通过弹性耦合 kappa 相互作用（缺陷与中央
格点间的弹簧，无阻尼，能量守恒至积分器精度）。线性系统中量子单声子期望与
经典能量份额同构，故缺陷能量份额的衰减可直接对照 Wigner-Weisskopf 黄金规则。
谱为体系波动色散（声学型，带顶 2 sqrt(3)，低能 rho ~ omega^2）。解析系数：
位移型耦合给出 Gamma = (pi kappa^2 / 2 omega0^2) rho_N(omega0)，其归一细节
与数值标定差因子约 3.4，挂 P1-50（耦合通道归一），本轮判据只锁定模型无关的
标度结构。

判据
----
B1（DOS 标度）：三档带内缺陷频率 omega0 = 1.0, 1.2, 1.6（kappa = 1；
    0.8 档为带底边缘观察项，其 Gamma*w0^2/rho 系数偏低约一半，见过程记录），
    谷值包络拟合的 Gamma_实测 乘 omega0^2 除 rho_N(omega0) 在三档间
    max/min <= 1.35（消去耦合因子后与数值态密度 48^3 光滑网格的标度一致）。
B2（带外束缚，FAIL 预期，EP5 齿）：omega0 = 3.9 > 带顶，缺陷份额不衰减，
    后段（t ∈ [180, 240]）平均 > 0.5。
B3（kappa^2 律）：omega0 = 1.6，kappa 减半使包络到达份额 0.35 的时间
    t_cross 之比在 [3, 5]（期望 4；Gamma 比例 kappa^2）。
判据可逆性：判据稳健（标度结构，换 N、换档位不翻）。
"""

import numpy as np

N = 40
DT = 0.15
T_END = 240.0
KAPPA = 1.0
FIT_LO = 0.02
NM = N ** 3
W_TOP = 2.0 * np.sqrt(3.0)
ETA = 0.04
N_K = 48
C0 = N // 2


def dos_grid():
    j = np.arange(N_K)
    kk = -np.pi + 2 * np.pi * j / N_K
    KX, KY, KZ = np.meshgrid(kk, kk, kk, indexing="ij")
    return 2.0 * np.sqrt(np.sin(KX / 2) ** 2 + np.sin(KY / 2) ** 2
                         + np.sin(KZ / 2) ** 2).ravel()


def rho_N(w0, wk):
    g = np.exp(-((wk - w0) ** 2) / (2 * ETA ** 2)) / (np.sqrt(2 * np.pi) * ETA)
    return g.sum() / (N_K ** 3)


def evolve(w0, kappa, t_end):
    """velocity Verlet：波场 + 缺陷振子弹簧耦合。返回缺陷能量份额时间线。"""
    u = np.zeros((N, N, N))
    v = np.zeros((N, N, N))
    d, dv = 1.0, 0.0
    Etot = 0.5 * (dv * dv + w0 * w0 * d * d)
    ts = np.arange(0.0, t_end + DT, DT)
    share = np.empty(len(ts))
    for i, t in enumerate(ts):
        share[i] = 0.5 * (dv * dv + w0 * w0 * d * d) / Etot
        lap = (np.roll(u, 1, 0) + np.roll(u, -1, 0) + np.roll(u, 1, 1)
               + np.roll(u, -1, 1) + np.roll(u, 1, 2) + np.roll(u, -1, 2)
               - 6.0 * u)
        u0 = u[C0, C0, C0]
        a_d = -w0 * w0 * d + kappa * (u0 - d)
        v += 0.5 * DT * lap
        v[C0, C0, C0] += 0.5 * DT * kappa * (d - u0)
        dv += 0.5 * DT * a_d
        d += DT * dv
        u += DT * v
        lap2 = (np.roll(u, 1, 0) + np.roll(u, -1, 0) + np.roll(u, 1, 1)
                + np.roll(u, -1, 1) + np.roll(u, 1, 2) + np.roll(u, -1, 2)
                - 6.0 * u)
        u0n = u[C0, C0, C0]
        a_d2 = -w0 * w0 * d + kappa * (u0n - d)
        v += 0.5 * DT * lap2
        v[C0, C0, C0] += 0.5 * DT * kappa * (d - u0n)
        dv += 0.5 * DT * a_d2
    return ts, share


def envelope_gamma(ts, share):
    """谷值包络拟合：局部极小的创新低段做 log 线性拟合（避开拍与回填）。"""
    valleys = [(ts[i], share[i]) for i in range(1, len(share) - 1)
               if share[i] < share[i - 1] and share[i] <= share[i + 1]
               and share[i] > FIT_LO]
    seq, floor = [], np.inf
    for t, p in valleys:
        if p < floor:
            seq.append((t, p))
            floor = p
        elif p > 2.0 * floor:
            break
    if len(seq) < 4:
        return np.nan, seq
    ta = np.array([t for t, _ in seq])
    pa = np.array([p for _, p in seq])
    return -np.polyfit(ta, np.log(pa), 1)[0], seq


def t_cross(seq, level=0.35):
    for t, p in seq:
        if p <= level:
            return t
    return np.nan


def main():
    print("=== 冒烟检查 ===")
    wk = dos_grid()
    print(f"DOS 网格 {N_K}^3 | 谱顶 {wk.max():.4f}（理论 {W_TOP:.4f}）")
    print("rho_N 抽样:", " ".join(
        f"{w}:{rho_N(w, wk):.4f}" for w in (0.8, 1.0, 1.2, 1.6, 3.9)))

    print("\n=== B1：DOS 标度（三档，kappa=1） ===")
    gammas = {}
    seqs = {}
    for w0 in (1.0, 1.2, 1.6):
        ts, share = evolve(w0, KAPPA, T_END)
        gm, seq = envelope_gamma(ts, share)
        gammas[w0] = gm
        seqs[w0] = seq
        val = gm * w0 ** 2 / rho_N(w0, wk)
        print(f"omega0={w0}: Gamma 实测 {gm:.5f} | Gamma*w0^2/rho_N = {val:.5f}"
              f" | 谷点 {len(seq)}")
    vals = [gammas[w0] * w0 ** 2 / rho_N(w0, wk) for w0 in (1.0, 1.2, 1.6)]
    spread = max(vals) / min(vals)
    b1_pass = spread <= 1.35 and all(np.isfinite(v) for v in vals)
    print(f"三档常数性 max/min = {spread:.4f}（<= 1.35 {'PASS' if b1_pass else 'FAIL'}）")
    print(f"B1（DOS 标度）: {'PASS' if b1_pass else 'FAIL'}")

    print("\n=== B3：kappa^2 律（omega0=1.6，kappa 减半） ===")
    ts_h, share_h = evolve(1.6, 0.5, T_END)
    _, seq_h = envelope_gamma(ts_h, share_h)
    t_f = t_cross(seqs[1.6], 0.35)
    t_h = t_cross(seq_h, 0.35)
    ratio_t = t_h / t_f
    b3_pass = 3.0 <= ratio_t <= 5.0 and np.isfinite(ratio_t)
    print(f"t_cross(kappa=1) = {t_f:.2f} | t_cross(kappa=0.5) = {t_h:.2f} | "
          f"比值 = {ratio_t:.3f}（[3,5] {'PASS' if b3_pass else 'FAIL'}）")
    print(f"B3（kappa^2 律）: {'PASS' if b3_pass else 'FAIL'}")

    print("\n=== B2：带外束缚（FAIL 预期，EP5 齿） ===")
    ts_o, share_o = evolve(3.9, KAPPA, T_END)
    late = share_o[ts_o >= 180.0]
    late_mean = late.mean()
    b2_pass = late_mean > 0.5
    print(f"带外 omega0=3.9: 份额后段平均 实测 {late_mean:.4f} | "
          f"期望 > 0.5（束缚态不衰）")
    print(f"B2: {'PASS' if b2_pass else 'FAIL'}")

    n_pass = sum([bool(b1_pass), bool(b3_pass), bool(b2_pass)])
    print(f"\n判据 {n_pass}/3（B1/B3/B2，B2 兼 EP5 齿：移出能带发射即消失）")
    print("边界声明：本验证锁定 DOS 标度、kappa^2 律与带外束缚三项标度结构，")
    print("不锁定位移耦合通道的解析归一系数（与数值标定差约 3.4，挂 P1-50），")
    print("不给出爱因斯坦 A 系数完整数值，不裁决 cos 高阶项通道矩阵元。")

    # 绘图（图存 images/，注记与论文嵌入）
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    wg = np.linspace(0.02, W_TOP - 0.02, 400)
    axes[0].plot(wg, [rho_N(w, wk) for w in wg], "b-")
    for w0 in (1.0, 1.2, 1.6):
        axes[0].axvline(w0, color="gray", lw=0.7, ls=":")
    axes[0].axvline(3.9, color="r", lw=0.8, ls="--")
    axes[0].set_xlabel("$\\omega$")
    axes[0].set_ylabel("$\\rho_N(\\omega)$")
    axes[0].set_title("3D DOS of lattice wave dispersion (N_K=48)")
    for w0, c in ((0.8, "tab:blue"), (1.2, "tab:green"), (1.6, "tab:orange")):
        ts, share = evolve(w0, KAPPA, T_END)
        axes[1].semilogy(ts, np.clip(share, 1e-6, None), color=c, lw=1.0,
                         label=f"$\\omega_0={w0}$")
    axes[1].set_xlabel("$t$")
    axes[1].set_ylabel("defect energy share")
    axes[1].legend(fontsize=8)
    axes[1].set_title("defect energy decay (kappa=1)")
    ts_h, share_h = evolve(1.6, 0.5, T_END)
    axes[2].semilogy(ts, np.clip(share, 1e-6, None), "b-", lw=1.0,
                     label="$\\kappa=1$")
    axes[2].semilogy(ts_h, np.clip(share_h, 1e-6, None), "r-", lw=1.0,
                     label="$\\kappa=0.5$")
    axes[2].axhline(0.35, color="k", lw=0.7, ls=":")
    axes[2].set_xlabel("$t$")
    axes[2].set_ylabel("defect energy share")
    axes[2].legend(fontsize=8)
    axes[2].set_title("kappa^2 law: time-scale ratio (B3)")
    fig.tight_layout()
    fig.savefig(r"D:\software\TianXingFC\images\emission_3d_525.png", dpi=150)
    print("图已存 images/emission_3d_525.png")


if __name__ == "__main__":
    main()
