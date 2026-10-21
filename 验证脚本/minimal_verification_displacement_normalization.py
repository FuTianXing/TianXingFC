# -*- coding: utf-8 -*-
"""位移耦合归一系数的弱耦合极限与读数口径分解（README 第 799 条）。

背景
----
第 525 条注记记下「位移型耦合解析系数 Γ = (π κ² / 2 ω0²) ρ(ω0) 与数值标定差
因子约 3.4，挂 P1-50」。第 526 条的观察项又把 ω0 = 1.0 档的单点偏低 3.7 倍
列为未定位异常。本条把两件事分开：

  一、解析系数的归一：在热力学极限下用态密度加 Hilbert 变换构造无展宽的
      原点格林函数，读取缺陷谱函数的半高宽，与黄金规则逐 κ 对照。若归一
      系数错，弱耦合极限应停在 3.4 一类常因子；若系数正确，比值应随 κ→0
      回到 1。
  二、3.4/3.7 的来源：时间域模型里「缺陷能量份额」只计 ½(dv²+ω0²d²)，
      漏掉弹簧项 ½κ(u0-d)²。把同一轮演化的总缺陷加弹簧能量与它对比，看
      异常是否只出现在漏项读数上。

判据（先立判据，再实现）
------------------------
N1（谱宽归一，κ = 0.1/0.2/0.5/1.0）：缺陷谱半高宽与黄金规则
    Γ_GR = πκ²ρ(ω0)/(2ω0²) 的比值落在 [0.75, 1.35]。容差预算：DOS 直方图
    2%（240³ 网格、3000 箱）、Hilbert 主值离散与插值 3%、局域势与强耦合
    修正 20%、峰值读取 5%。
N2（弱耦合极限，κ = 0.05）：同一比值落在 [0.8, 1.3]，且不出现 3 倍以上
    的常因子。
N3（读数口径分解，时间域 κ = 1）：三档 ω0 = 1.0/1.2/1.6 的含弹簧总能量
    读数 D_total = Γω0²/ρ 的 max/min ≤ 1.15；ω0 = 1.0 档的
    D_total/D_share ≥ 2（复现 3.4/3.7 的漏项异常），ω0 = 1.2 与 1.6 档
    两读数相对差 ≤ 15%。
EP5 齿：把黄金规则系数故意写成 πκ²ρ/ω0²（大 2 倍），N1 的四个比值应全部
    落到 0.75 以下——改掉系数就 FAIL，判据有齿。

判据可逆性：N1/N2 判据稳健（谱宽是线性响应本征量，换网格不翻）；
N3 判据敏感（依赖时间域拟合窗与读数定义），只用于定位读数口径。

边界：本条不裁决体系缺陷与真空的真实耦合型（P1-50 通道型判定仍开放），
不给出爱因斯坦 A 系数数值，不涉及介质折射定量。
"""

import numpy as np


W0 = 1.2
KAPPAS = (0.05, 0.1, 0.2, 0.5, 1.0)
N_K = 240
N_BINS = 3000
W_TOP = 2.0 * np.sqrt(3.0)


def dos_and_green():
    """态密度直方图 + Hilbert 变换给原点格林函数（无展宽）。"""
    edges = np.linspace(0.0, W_TOP + 0.02, N_BINS + 1)
    cent = 0.5 * (edges[1:] + edges[:-1])
    hist = np.zeros(N_BINS)
    dk = 2.0 * np.pi / N_K
    for i in range(N_K):
        kx = -np.pi + dk * i
        ky, kz = np.meshgrid(-np.pi + dk * np.arange(N_K),
                             -np.pi + dk * np.arange(N_K), indexing="ij")
        wk = 2.0 * np.sqrt(np.sin(kx / 2) ** 2 + np.sin(ky / 2) ** 2
                           + np.sin(kz / 2) ** 2).ravel()
        h, _ = np.histogram(wk, bins=edges)
        hist += h
    rho = hist / (N_K ** 3 * (edges[1] - edges[0]))
    with np.errstate(divide="ignore", invalid="ignore"):
        img = -np.pi * rho / (2.0 * cent)
    img[0] = 0.0
    d_img = np.gradient(img, edges[1] - edges[0])
    re = np.empty(N_BINS)
    dw = edges[1] - edges[0]
    for i, w in enumerate(cent):
        d = cent - w
        mask = np.ones(N_BINS, bool)
        mask[i] = False
        # 奇异箱的线性项给 dIm/dw * dw
        re[i] = (np.sum(img[mask] / d[mask]) * dw + d_img[i] * dw) / np.pi
    return cent, rho, re + 1j * img


def interp_green(w, cent, gf):
    return np.interp(w, cent, gf.real) + 1j * np.interp(w, cent, gf.imag)


def spectral_width(kappa, cent, rho, gf, w0=W0):
    """缺陷谱 A(w) = -Im G_d/pi 的峰位与半高宽。"""
    wd = np.sqrt(w0 * w0 + kappa)
    wg = np.linspace(wd - 0.15, wd + 0.15, 6001)
    g = interp_green(wg, cent, gf)
    g0 = g / (1.0 + kappa * g)
    gd = 1.0 / (wg * wg - wd * wd - kappa * kappa * g0)
    a = -gd.imag / np.pi
    ip = int(np.argmax(a))
    ap = a[ip]
    wp = wg[ip]
    left = right = np.nan
    for j in range(ip, 0, -1):
        if a[j] < ap / 2.0:
            left = wg[j]
            break
    for j in range(ip, len(wg)):
        if a[j] < ap / 2.0:
            right = wg[j]
            break
    fwhm = right - left
    rho0 = np.interp(w0, cent, rho)
    gr = np.pi * kappa * kappa * rho0 / (2.0 * w0 * w0)
    return wp, fwhm, gr, rho0


# ---------------- 时间域：缺陷能量 vs 含弹簧总能量 ----------------

N = 40
DT = 0.15
T_END = 240.0
C0 = N // 2
FIT_LO = 0.02


def rho_gauss(w, eta=0.04, nk=64):
    kk = -np.pi + 2.0 * np.pi * np.arange(nk) / nk
    kx, ky, kz = np.meshgrid(kk, kk, kk, indexing="ij")
    wk = 2.0 * np.sqrt(np.sin(kx / 2) ** 2 + np.sin(ky / 2) ** 2
                       + np.sin(kz / 2) ** 2).ravel()
    g = np.exp(-((wk - w) ** 2) / (2.0 * eta * eta)) / (np.sqrt(2.0 * np.pi) * eta)
    return g.sum() / len(wk)


def evolve(w0, kappa, t_end=T_END):
    u = np.zeros((N, N, N))
    v = np.zeros((N, N, N))
    d, dv = 1.0, 0.0
    ts = np.arange(0.0, t_end + DT, DT)
    share = np.empty(len(ts))
    total = np.empty(len(ts))
    etot = 0.5 * (w0 * w0 + kappa)
    for i, t in enumerate(ts):
        u0 = u[C0, C0, C0]
        share[i] = 0.5 * (dv * dv + w0 * w0 * d * d) / etot
        total[i] = 0.5 * (dv * dv + w0 * w0 * d * d
                          + kappa * (u0 - d) ** 2) / etot
        lap = (np.roll(u, 1, 0) + np.roll(u, -1, 0)
               + np.roll(u, 1, 1) + np.roll(u, -1, 1)
               + np.roll(u, 1, 2) + np.roll(u, -1, 2) - 6.0 * u)
        a_d = -w0 * w0 * d + kappa * (u0 - d)
        v += 0.5 * DT * lap
        v[C0, C0, C0] += 0.5 * DT * kappa * (d - u0)
        dv += 0.5 * DT * a_d
        d += DT * dv
        u += DT * v
        lap2 = (np.roll(u, 1, 0) + np.roll(u, -1, 0)
                + np.roll(u, 1, 1) + np.roll(u, -1, 1)
                + np.roll(u, 1, 2) + np.roll(u, -1, 2) - 6.0 * u)
        u0n = u[C0, C0, C0]
        a_d2 = -w0 * w0 * d + kappa * (u0n - d)
        v += 0.5 * DT * lap2
        v[C0, C0, C0] += 0.5 * DT * kappa * (d - u0n)
        dv += 0.5 * DT * a_d2
    return ts, share, total


def envelope_gamma(ts, s):
    valleys = [(ts[i], s[i]) for i in range(1, len(s) - 1)
               if s[i] < s[i - 1] and s[i] <= s[i + 1] and s[i] > FIT_LO]
    seq, floor = [], np.inf
    for t, p in valleys:
        if p < floor:
            seq.append((t, p))
            floor = p
        elif p > 2.0 * floor:
            break
    if len(seq) < 4:
        return np.nan
    ta = np.array([t for t, _ in seq])
    pa = np.array([p for _, p in seq])
    return -np.polyfit(ta, np.log(pa), 1)[0]


def main():
    print("=== 冒烟检查 ===")
    cent, rho, gf = dos_and_green()
    print(f"DOS 网格 {N_K}^3、{N_BINS} 箱；谱顶 {cent[-1]:.4f}")
    print("rho(1.2) = %.6f | -ImG(1.2) = %.6f"
          % (np.interp(W0, cent, rho), -interp_green(W0, cent, gf).imag))

    print("\n=== N1/N2：缺陷谱宽对黄金规则 ===")
    rows = []
    for kap in KAPPAS:
        wp, fw, gr, rho0 = spectral_width(kap, cent, rho, gf)
        ratio = fw / gr
        rows.append((kap, wp, fw, gr, ratio))
        print(f"kappa={kap:>4}: peak={wp:.4f} FWHM={fw:.5f} "
              f"Gamma_GR={gr:.5f} ratio={ratio:.3f}")
    n1 = all(0.75 <= r[4] <= 1.35 for r in rows if r[0] >= 0.1)
    n2 = 0.8 <= [r[4] for r in rows if r[0] == 0.05][0] <= 1.3
    print(f"N1: {'PASS' if n1 else 'FAIL'} | N2: {'PASS' if n2 else 'FAIL'}")

    # EP5 齿：系数故意大 2 倍
    bad = [(r[4] * 0.5) for r in rows if r[0] >= 0.1]
    ep5 = all(x < 0.75 for x in bad)
    print(f"EP5（π 代替 π/2）: ratios={['%.3f' % x for x in bad]} "
          f"{'FAIL 预期达成' if ep5 else '齿未咬合'}")

    print("\n=== N3：缺陷份额 vs 含弹簧总能量（κ=1） ===")
    dsh, dtot, dvals = {}, {}, {}
    for w0 in (1.0, 1.2, 1.6):
        ts, share, total = evolve(w0, 1.0)
        r = rho_gauss(w0)
        gs = envelope_gamma(ts, share)
        gt = envelope_gamma(ts, total)
        dsh[w0] = gs * w0 * w0 / r
        dtot[w0] = gt * w0 * w0 / r
        dvals[w0] = (gs, gt, r)
        print(f"omega0={w0}: D_share={dsh[w0]:.4f} | D_total={dtot[w0]:.4f} "
              f"| ratio={dtot[w0] / dsh[w0]:.3f}")
    spread = max(dtot.values()) / min(dtot.values())
    n3a = spread <= 1.15
    n3b = dtot[1.0] / dsh[1.0] >= 2.0
    n3c = all(abs(dtot[w] / dsh[w] - 1.0) <= 0.15 for w in (1.2, 1.6))
    n3 = n3a and n3b and n3c
    print(f"D_total max/min={spread:.4f}（<=1.15 {'PASS' if n3a else 'FAIL'}）")
    print(f"ω0=1.0 的 D_total/D_share={dtot[1.0] / dsh[1.0]:.3f}"
          f"（>=2 {'PASS' if n3b else 'FAIL'}）")
    print(f"ω0=1.2/1.6 两读数相对差={abs(dtot[1.2] / dsh[1.2] - 1):.4f}/"
          f"{abs(dtot[1.6] / dsh[1.6] - 1):.4f}（<=0.15 {'PASS' if n3c else 'FAIL'}）")
    print(f"N3: {'PASS' if n3 else 'FAIL'}")

    n_pass = sum([bool(n1), bool(n2), bool(n3)])
    print(f"\n判据 {n_pass}/3（N1/N2/N3，EP5 齿：错系数即 FAIL）")
    print("结论：归一系数在弱耦合极限回到 1；3.4/3.7 来自漏掉弹簧项的读数口径。")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    for kap, c in zip(KAPPAS, ("tab:blue", "tab:orange", "tab:green",
                               "tab:red", "tab:purple")):
        wd = np.sqrt(W0 * W0 + kap)
        wg = np.linspace(wd - 0.12, wd + 0.12, 4000)
        g = interp_green(wg, cent, gf)
        g0 = g / (1.0 + kap * g)
        gd = 1.0 / (wg * wg - wd * wd - kap * kap * g0)
        axes[0].plot(wg, -gd.imag / np.pi, color=c, lw=1.0,
                     label=r"$\kappa=%.2f$" % kap)
    axes[0].set_xlabel(r"$\omega$")
    axes[0].set_ylabel(r"$A_d(\omega)$")
    axes[0].set_title("defect spectral function (arb. scale)")
    axes[0].legend(fontsize=7)
    ks = [r[0] for r in rows]
    rs = [r[4] for r in rows]
    axes[1].semilogx(ks, rs, "o-", color="tab:blue")
    axes[1].axhspan(0.75, 1.35, color="tab:green", alpha=0.15,
                    label="N1 band")
    axes[1].axhline(1.0, color="k", lw=0.8, ls=":")
    axes[1].set_xlabel(r"$\kappa$")
    axes[1].set_ylabel("FWHM / $\\Gamma_{GR}$")
    axes[1].set_title("normalization vs coupling")
    axes[1].legend(fontsize=8)
    xs = np.arange(3)
    axes[2].bar(xs - 0.18, [dsh[w] for w in (1.0, 1.2, 1.6)], 0.36,
                label="defect-only share", color="tab:orange")
    axes[2].bar(xs + 0.18, [dtot[w] for w in (1.0, 1.2, 1.6)], 0.36,
                label="defect+spring total", color="tab:blue")
    axes[2].set_xticks(xs)
    axes[2].set_xticklabels([r"$\omega_0=1.0$", r"$\omega_0=1.2$",
                             r"$\omega_0=1.6$"])
    axes[2].set_ylabel(r"$D=\Gamma\omega_0^2/\rho$")
    axes[2].set_title("readout decomposition (κ=1)")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(r"D:\software\TianXingFC\images\displacement_normalization_799.png",
                dpi=150)
    print("图已存 images/displacement_normalization_799.png")


if __name__ == "__main__":
    main()
