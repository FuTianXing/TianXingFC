# -*- coding: utf-8 -*-
r"""
minimal_verification_flux_weighting_3d_defect.py

[通量加权的三维各向异性与局域长度轮] 方向依赖与尺度不变性

定位
----
接 [[注记-碰撞通量加权的最小验证：锁速载子口径闭合与包络群速口径的偏差]]（首轮），
执行 [[研究方向-碰撞动力学的抽样机制：三个候选模型与优选]] 计划中的后两项：
三维 [111] 方向的速度各向异性（原 J6）与局域长度扫描（原 J5）。
读数的意义在于：包络群速口径的份额偏差既依赖方向，也依赖模式的中心波数，
而它与模式的空间宽度无关。本脚本不改动基准文件；不含新科学断言。

模型
----
三维精确色散 $\omega^2 = (4/t_P^2)\sum_a \sin^2(k_a l_P/2)$，群速分量
  $v_a = \partial\omega/\partial k_a = c\,\sin(k_a l_P)/(t_P\,\omega)$，$c = l_P/t_P$。
一维特例给出 $v/c = \cos(k l_P/2)$；沿 [111]（$k_a = k/\sqrt{3}$）给出
  $|v|/c = \cos\big(k l_P/(2\sqrt{3})\big)$，即色散修正的辐角被 $1/\sqrt{3}$ 压低。
份额偏差沿首轮的构造：频率平方型阱 $\omega_j = \omega_0\sqrt{1-u_j^2}$、$u_j = (j-j_c)/w$，
高斯模式密度 $\rho_j \propto e^{-(j-j_c)^2/(2\sigma^2)}$、$\sigma = w/2$，
比较载子锁速口径 $p^A_j \propto \rho_j$ 与包络群速口径 $p^B_j \propto \rho_j v_j$，偏差取总变差。

判据（先立判据，再实现）
----
J1 [100] 方向的速度衰减系数：$1 - v/c$ 对 $(k l_P)^2$ 的拟合系数 $= 1/8 \pm 0.001$。
J2 [111] 方向的系数 $= 1/24 \pm 0.001$（同法拟合）。
J3 群速分量核对：解析式 $v_a = c\sin(k_a l_P)/(t_P\omega)$ 与数值偏导数的相对误差 $< 10^{-8}$（三维一般方向）。
J4 各向异性比：同一 $|k|$ 下两方向的系数之比 $= 3.0 \pm 0.05$；份额偏差之比同值（$3.0 \pm 0.15$）。
J5 尺度不变性：固定中心波数，变阱半宽 $w \in \{10,20,40,80,160\}$，总变差的相对散布 $< 1\%$。
J6 固定缺陷强度下的局域长度标度：$w = \omega_0/\Omega$（$\Omega$ 为阱曲率参数，固定），
   总变差对 $\xi \equiv w$ 的双对数斜率 $= 2.0 \pm 0.3$（即局域长度越大偏差越大）。
J7 分布形状依赖：固定 $k_c$ 与 $w$，改变 $\sigma/w \in \{0.25,0.5,0.75,1.0\}$，总变差单调变化（报告读数）。

运行
----
python 验证脚本/minimal_verification_flux_weighting_3d_defect.py
确定性（无随机数）。图存 D:/software/TianXingFC/images/flux_weighting_3d_defect.png
"""
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LP = 1.0
TP = 1.0
C = LP / TP


def omega3d(kx, ky, kz):
    return (2.0 / TP) * np.sqrt(np.sin(kx * LP / 2.0) ** 2
                                + np.sin(ky * LP / 2.0) ** 2
                                + np.sin(kz * LP / 2.0) ** 2)


def speed_profile(k_c, w, sigma_ratio=0.5, j_c=0.0):
    half = int(2.2 * w)
    j = np.arange(-half, half + 1, dtype=float) + j_c
    u = (j - j_c) / w
    theta_c = k_c * LP / 2.0
    v = np.where(np.abs(u) <= 1.0,
                 C * np.sqrt(np.cos(theta_c) ** 2 + np.minimum(np.abs(u), 1.0) ** 2 * np.sin(theta_c) ** 2),
                 C)
    rho = np.exp(-(j - j_c) ** 2 / (2.0 * (sigma_ratio * w) ** 2))
    return j, u, v, rho


def total_variation(p, q):
    return 0.5 * float(np.sum(np.abs(p - q)))


def tv_flux(k_c, w, sigma_ratio=0.5, direction="100"):
    j, u, v, rho = speed_profile(k_c, w, sigma_ratio)
    if direction == "111":
        theta_c = k_c * LP / (2.0 * np.sqrt(3.0))
        inside = np.abs(u) <= 1.0
        v = C * np.sqrt(np.cos(theta_c) ** 2 + np.minimum(np.abs(u), 1.0) ** 2 * np.sin(theta_c) ** 2)
    pa = rho / rho.sum()
    pb = rho * (v / C)
    pb = pb / pb.sum()
    return total_variation(pa, pb)


def main():
    t_start = time.time()
    checks = []

    def report(tag, ok, detail):
        checks.append((tag, bool(ok), detail))
        print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))

    print("== J1/J2 方向相关的速度衰减系数 ==")
    ks = np.linspace(0.02, 0.30, 12)
    c100 = 1.0 - np.cos(ks * LP / 2.0)
    c111 = 1.0 - np.cos(ks * LP / (2.0 * np.sqrt(3.0)))
    a100 = float(np.polyfit((ks * LP) ** 2, c100, 1)[0])
    a111 = float(np.polyfit((ks * LP) ** 2, c111, 1)[0])
    report("J1 [100] 系数 = 1/8", abs(a100 - 0.125) < 0.001, "拟合 %.6f（1/8 = %.6f）" % (a100, 0.125))
    report("J2 [111] 系数 = 1/24", abs(a111 - 1.0 / 24.0) < 0.001,
           "拟合 %.6f（1/24 = %.6f）" % (a111, 1.0 / 24.0))

    print("== J3 群速分量核对（三维一般方向） ==")
    rng = np.random.default_rng(0)
    worst = 0.0
    h = 1e-6
    for _ in range(6):
        k = rng.uniform(0.05, 1.2, 3)
        w0 = omega3d(k[0], k[1], k[2])
        for a in range(3):
            kp = k.copy()
            kp[a] += h
            km = k.copy()
            km[a] -= h
            fd = (omega3d(*kp) - omega3d(*km)) / (2.0 * h)
            an = C * np.sin(k[a] * LP) / (TP * w0)
            worst = max(worst, abs(fd - an) / max(abs(an), 1e-30))
    report("J3 解析式对数值偏导数", worst < 1e-8, "最大相对误差 %.2e（6 组方向 x 3 分量）" % worst)

    print("== J4 各向异性比 ==")
    ratio_coef = a100 / a111
    k_c_test = 0.8
    tv100 = tv_flux(k_c_test, 60.0, direction="100")
    tv111 = tv_flux(k_c_test, 60.0, direction="111")
    ratio_tv = tv100 / tv111
    report("J4 系数之比 = 3", abs(ratio_coef - 3.0) < 0.05, "比值 %.4f" % ratio_coef)
    report("J4 份额偏差之比 = 3", abs(ratio_tv - 3.0) < 0.15,
           "TV[100] = %.6f，TV[111] = %.6f，比 %.4f" % (tv100, tv111, ratio_tv))

    print("== J5 尺度不变性 ==")
    ws = np.array([10.0, 20.0, 40.0, 80.0, 160.0])
    tvs_w = np.array([tv_flux(0.4, w) for w in ws])
    spread = float((tvs_w.max() - tvs_w.min()) / tvs_w.mean())
    report("J5 偏差与阱半宽无关", spread < 0.01,
           "相对散布 %.2e（TV = %.6f 至 %.6f）" % (spread, tvs_w.min(), tvs_w.max()))

    print("== J6 固定缺陷强度下的局域长度标度 ==")
    Omega = 0.01
    omega0s = np.linspace(0.15, 0.80, 7)
    xis = omega0s / Omega
    tvs_xi = []
    for w0 in omega0s:
        kc = 2.0 * np.arcsin(TP * w0 / 2.0) / LP
        tvs_xi.append(tv_flux(kc, w0 / Omega))
    tvs_xi = np.array(tvs_xi)
    slope6 = float(np.polyfit(np.log(xis), np.log(tvs_xi), 1)[0])
    report("J6 总变差对 xi 的双对数斜率 = 2", abs(slope6 - 2.0) < 0.3,
           "斜率 %.4f（xi = %.1f 至 %.1f l_P，TV = %.3e 至 %.3e）" % (slope6, xis[0], xis[-1], tvs_xi[0], tvs_xi[-1]))

    print("== J7 分布形状依赖 ==")
    ratios = np.array([0.25, 0.5, 0.75, 1.0])
    tvs_s = np.array([tv_flux(0.8, 60.0, sigma_ratio=r) for r in ratios])
    mono = bool(np.all(np.diff(tvs_s) > 0) or np.all(np.diff(tvs_s) < 0))
    report("J7 形状依赖单调", mono, "TV = %s" % np.array2string(tvs_s, precision=5))

    print("== 绘图 ==")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))

    ax = axes[0, 0]
    kk = np.linspace(0.0, 2.0, 400)
    ax.plot(kk, 1.0 - np.cos(kk * LP / 2.0), lw=1.6, label=r"[100]：$1-v/c$")
    ax.plot(kk, 1.0 - np.cos(kk * LP / (2.0 * np.sqrt(3.0))), lw=1.6,
            label=r"[111]：$1-v/c$")
    ax.plot(kk, a100 * (kk * LP) ** 2, "k--", lw=0.9, label=r"$(1/8)(k l_P)^2$")
    ax.plot(kk, a111 * (kk * LP) ** 2, "k:", lw=0.9, label=r"$(1/24)(k l_P)^2$")
    ax.set_xlabel(r"$k l_P$")
    ax.set_ylabel(r"$1 - v/c$")
    ax.set_title("图 A 方向相关的色散修正")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(ws, tvs_w, "o-", label=r"$k_c l_P = 0.4$")
    ax.plot(ws, [tv_flux(0.8, w) for w in ws], "s--", label=r"$k_c l_P = 0.8$")
    ax.set_xscale("log")
    ax.set_xlabel(r"阱半宽 $w$（单位 $l_P$）")
    ax.set_ylabel("总变差")
    ax.set_title("图 B 尺度不变：偏差只由中心波数决定")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.loglog(xis, tvs_xi, "o-", label="实测总变差")
    ax.loglog(xis, tvs_xi[0] * (xis / xis[0]) ** 2, "k--", lw=0.9, label=r"$\xi^2$ 参考线")
    ax.set_xlabel(r"局域长度 $\xi$（单位 $l_P$）")
    ax.set_ylabel("总变差")
    ax.set_title("图 C 固定缺陷强度：偏差按 $\\xi^2$ 上升（斜率 %.2f）" % slope6)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(ratios, tvs_s, "o-")
    ax.set_xlabel(r"$\sigma/w$")
    ax.set_ylabel("总变差")
    ax.set_title("图 D 偏差对模式形状的依赖（$k_c l_P = 0.8$）")

    fig.tight_layout()
    out = "D:/software/TianXingFC/images/flux_weighting_3d_defect.png"
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
