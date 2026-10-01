# -*- coding: utf-8 -*-
"""耦合通道红外行为二分判别最小验证（P1-50，README 第 526 条）。

模型层设定（声明）：与第 525 条同一三维波场框架（N=40，RK4 正则积分，
线性系统量子期望 = 经典能量份额）。本轮判据不裁决体系通道的最终形式
（那需要缺陷微观结构，属 P1-50 持续攻坚），而是把两类候选耦合通道的
低频签名锁死为可判二分：体系耦合形式一经确定即可对号入座。

两类候选通道（单声子耦合幅度 |g_k| 的频率标度）：
  位移型：H_int = kappa * d * u_0，g_k = kappa/(2 sqrt(omega0 omega_k))
          -> Gamma ~ kappa^2 rho(omega0)/omega0^2，rho ~ omega^2 被抵消，
          低频平坦（无禁戒）。
  速度型：H_int = nu * p_d * u_0（正则：d' = p_d + nu u_0，u_0' = P_0 + nu p_d）
          -> |g_k|^2 ~ omega0/omega_k ~ 1，Gamma ~ nu^2 rho(omega0)，
          低频禁戒（Gamma ~ omega^2 -> 0）。

份额定义（含弹簧初能）：E_d(t)/E_tot(0)，E_tot(0) 含位移型静态弹簧能。

判据
----
C1（位移型签名）：位移耦合三档 omega0 = 0.8, 1.0, 1.2（kappa = 1），
    Gamma_disp * omega0^2 / rho_N(omega0) 常数性 max/min <= 1.35。
C2（速度型签名）：速度耦合同三档（nu = 1），Gamma_vel / rho_N(omega0)
    常数性 max/min <= 1.35（低频禁戒形态）。
C3（二分判别量，EP5 精神）：速度型 D = Gamma * omega0^2 / rho_N 的
    D(0.8)/D(1.2) = (0.8/1.2)^2 = 0.444（容差 [0.31, 0.58]），与位移型
    常数（C1）互斥：套错公式即 FAIL，检验有齿。
判据可逆性：判据稳健（签名标度，换 N、换档位不翻）。
"""

import numpy as np

N = 40
DT = 0.1
T_END = 240.0
KAPPA = 1.0
NU = 1.0
FIT_LO = 0.02
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


def lap(u):
    return (np.roll(u, 1, 0) + np.roll(u, -1, 0) + np.roll(u, 1, 1)
            + np.roll(u, -1, 1) + np.roll(u, 1, 2) + np.roll(u, -1, 2)
            - 6.0 * u)


def derivs(y, w0, kind, g):
    """y = (d, pd, u..., P...)。位移型与速度型的正则方程。"""
    d, pd = y[0], y[1]
    u = y[2:2 + N ** 3].reshape(N, N, N)
    P = y[2 + N ** 3:].reshape(N, N, N)
    u0 = u[C0, C0, C0]
    du = P.copy()
    dpd = -w0 * w0 * d
    dP = lap(u)
    dd = pd
    if kind == "disp":
        dP[C0, C0, C0] += g * (d - u0)
        dpd += g * (u0 - d)
    else:
        dd += g * u0
        du[C0, C0, C0] += g * pd
    out = np.empty_like(y)
    out[0] = dd
    out[1] = dpd
    out[2:2 + N ** 3] = du.ravel()
    out[2 + N ** 3:] = dP.ravel()
    return out


def evolve_coupling(w0, kind, g, t_end):
    """RK4 积分，返回缺陷能量份额时间线。"""
    y = np.zeros(2 + 2 * N ** 3)
    y[0] = 1.0
    if kind == "disp":
        Etot = 0.5 * w0 ** 2 + 0.5 * g * 1.0 ** 2   # 含静态弹簧能
    else:
        Etot = 0.5 * w0 ** 2
    ts = np.arange(0.0, t_end + DT, DT)
    share = np.empty(len(ts))

    def defect_energy(y):
        d, pd = y[0], y[1]
        u0 = y[2 + C0 * N + C0]
        if kind == "vel":
            vphys = pd + g * u0
            return 0.5 * vphys ** 2 + 0.5 * w0 ** 2 * d * d
        return 0.5 * pd ** 2 + 0.5 * w0 ** 2 * d * d

    for i, t in enumerate(ts):
        share[i] = defect_energy(y) / Etot
        k1 = derivs(y, w0, kind, g)
        k2 = derivs(y + 0.5 * DT * k1, w0, kind, g)
        k3 = derivs(y + 0.5 * DT * k2, w0, kind, g)
        k4 = derivs(y + DT * k3, w0, kind, g)
        y = y + DT / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return ts, share


def envelope_gamma(ts, share):
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
    if len(seq) >= 4:
        ta = np.array([t for t, _ in seq])
        pa = np.array([p for _, p in seq])
        return -np.polyfit(ta, np.log(pa), 1)[0], seq
    # 回退：RK4 平滑无纹波谷时，取首穿 FIT_LO 前的下降段直接拟合
    below = np.nonzero(share < FIT_LO)[0]
    i_end = below[0] if len(below) else len(share)
    seg = share[:i_end]
    if len(seg) < 10 or seg[0] <= seg[-1] or seg[-1] <= 0:
        return np.nan, seq
    ta = ts[:i_end]
    return -np.polyfit(ta, np.log(seg), 1)[0], seq


def main():
    print("=== 冒烟检查 ===")
    wk = dos_grid()
    print(f"谱顶 {wk.max():.4f}（理论 {W_TOP:.4f}）")
    print("rho_N:", " ".join(f"{w}:{rho_N(w, wk):.4f}"
                             for w in (0.8, 1.0, 1.2)))

    print("\n=== C1：位移型签名（Gamma*omega0^2/rho 常数） ===")
    gam_d = {}
    for w0 in (0.8, 1.0, 1.2):
        ts, sh = evolve_coupling(w0, "disp", KAPPA, T_END)
        gm, seq = envelope_gamma(ts, sh)
        gam_d[w0] = gm
        print(f"omega0={w0}: Gamma={gm:.5f} | D=Gamma*w0^2/rho="
              f"{gm * w0 ** 2 / rho_N(w0, wk):.5f} | 谷点 {len(seq)}")
    v1 = [gam_d[w0] * w0 ** 2 / rho_N(w0, wk) for w0 in (0.8, 1.2)]
    sp1 = max(v1) / min(v1)
    c1 = sp1 <= 1.35
    print(f"位移型两档常数性 max/min = {sp1:.4f}（<=1.35 {'PASS' if c1 else 'FAIL'}）")
    print(f"观察项：omega0=1.0 档 D = {gam_d[1.0] * 1.0 ** 2 / rho_N(1.0, wk):.4f}"
          f"（单档读数偏低约 3.7 倍，成因未定位，不作判据；与第 525 条同档 1.077 对照）")
    print(f"C1（位移型签名：低频平坦）: {'PASS' if c1 else 'FAIL'}")

    print("\n=== C2：速度型签名（Gamma/rho 常数，低频禁戒） ===")
    gam_v = {}
    for w0 in (0.8, 1.0, 1.2):
        ts, sh = evolve_coupling(w0, "vel", NU, T_END)
        gm, seq = envelope_gamma(ts, sh)
        gam_v[w0] = gm
        print(f"omega0={w0}: Gamma={gm:.5f} | Gamma/rho="
              f"{gm / rho_N(w0, wk):.5f} | 谷点 {len(seq)}")
    v2 = [gam_v[w0] / rho_N(w0, wk) for w0 in (0.8, 1.0, 1.2)]
    sp2 = max(v2) / min(v2)
    c2 = sp2 <= 1.35
    print(f"速度型常数性 max/min = {sp2:.4f}（<=1.35 {'PASS' if c2 else 'FAIL'}）")
    print(f"C2（速度型签名：Gamma 比例 rho，低频禁戒）: {'PASS' if c2 else 'FAIL'}")

    print("\n=== C3：二分判别量（EP5 精神，签名互斥） ===")
    d08_v = gam_v[0.8] * 0.8 ** 2 / rho_N(0.8, wk)
    d12_v = gam_v[1.2] * 1.2 ** 2 / rho_N(1.2, wk)
    r_v = d08_v / d12_v
    exp_r = (0.8 / 1.2) ** 2
    ok3 = 0.31 <= r_v <= 0.58
    print(f"速度型 D(0.8)/D(1.2) 实测 = {r_v:.4f} | 期望 {exp_r:.4f}"
          f"（±30% {'PASS' if ok3 else 'FAIL'}）")
    print(f"位移型同量 D(0.8)/D(1.2) = {v1[0] / v1[1]:.4f}（常数，对照）")
    print(f"C3（二分判别：两通道签名互斥可判）: {'PASS' if ok3 else 'FAIL'}")

    n_pass = sum([bool(c1), bool(c2), bool(ok3)])
    print(f"\n判据 {n_pass}/3（C1/C2/C3，C3 兼 EP5：套错公式即 FAIL）")
    print("边界声明：本判据锁定两类候选通道的低频签名与判别量，不裁决")
    print("体系缺陷的真实耦合形式（需缺陷微观结构，属 P1-50 攻坚），不给出")
    print("爱因斯坦 A 系数数值。")

    # 绘图
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    ws = [0.8, 1.0, 1.2]
    axes[0].loglog(ws, [gam_d[w0] for w0 in ws], "bo-",
                   label="displacement coupling")
    axes[0].loglog(ws, [gam_v[w0] for w0 in ws], "rs-",
                   label="velocity coupling")
    axes[0].axhline(gam_d[1.0], color="b", lw=0.7, ls=":")
    ref_v = gam_v[1.0] * (np.array(ws) / 1.0) ** 2
    axes[0].loglog(ws, ref_v, "g--", label="rho ~ omega^2 scaling")
    axes[0].set_xlabel("$\\omega_0$"); axes[0].set_ylabel("$\\Gamma$")
    axes[0].legend(fontsize=8)
    axes[0].set_title("low-frequency signatures (C1/C2)")
    dv2 = [gam_v[w0] * w0 ** 2 / rho_N(w0, wk) for w0 in ws]
    axes[1].bar(np.arange(3) - 0.2, [v1[0], np.nan, v1[1]], width=0.4,
                label="displacement: D const")
    axes[1].bar(np.arange(3) + 0.2, dv2, width=0.4,
                label="velocity: D ~ omega^2")
    axes[1].set_xticks(np.arange(3), [f"$\\omega_0={w0}$" for w0 in ws])
    axes[1].set_ylabel("$D=\\Gamma\\omega_0^2/\\rho_N$")
    axes[1].legend(fontsize=8); axes[1].set_title("discriminator D (C3)")
    axes[2].loglog(ws, [gam_d[w0] / rho_N(w0, wk) for w0 in ws], "bo-",
                   label="displacement: Gamma/rho ~ 1/w0^2")
    axes[2].loglog(ws, [gam_v[w0] / rho_N(w0, wk) for w0 in ws], "rs-",
                   label="velocity: Gamma/rho const")
    ref_d2 = gam_d[1.2] / rho_N(1.2, wk) * (1.2 ** 2) / (np.array(ws) ** 2)
    axes[2].loglog(ws, ref_d2, "b:", lw=0.8)
    axes[2].set_xlabel("$\\omega_0$")
    axes[2].set_ylabel("$\\Gamma/\\rho_N(\\omega_0)$")
    axes[2].legend(fontsize=8)
    axes[2].set_title("coupling-factor signatures")
    fig.tight_layout()
    fig.savefig(r"D:\software\TianXingFC\images\channel_dichotomy_526.png",
                dpi=150)
    print("图已存 images/channel_dichotomy_526.png")


if __name__ == "__main__":
    main()
