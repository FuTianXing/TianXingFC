# -*- coding: utf-8 -*-
"""
判据第 628 号：锚核落点对能量质心（首矩轮）。

问题：锚核（锁定处）是否等于缺陷能量的质心？
模型：一维锁定势格点上的静态扭结（与第 627 号同模型）。
  能量密度 eps(x) = 4 K sech^2(x/xi)，总能量 E = 8 sqrt(k_b K)，xi = sqrt(k_b/K)。
解析期望：
  单体（对称剖面）：首矩 x_cm = Sigma (n-n_pin) eps / Sigma eps = 0，与钉重合。
  两瓣叠加像（分离瓣极限，线性构造 eps = q eps_A + (1-q) eps_B）：首矩由线性性精确给
    x_cm = q x_A + (1-q) x_B；钉在 A 瓣上，分家量 = (1-q) D（q = 1/2 给 D/2）。
  交叉项：分离瓣极限下被干涉禁闭压制，量级 s(D) 随 D/xi 指数下降。
  首矩盲性：点源（乙：质量集中在锚核处）与展布源（取甲）首矩相同，首矩不能判定两者。

判据 R1 冒烟 / R2 单体重合 / R3 两瓣分家（兼作 R2 的反例）/ R4 交叉项压制 / R5 首矩盲性。
"""
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = r"D:\software\TianXingFC"
IMG = os.path.join(ROOT, "images")
os.makedirs(IMG, exist_ok=True)

KB = 1.0


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


def solve_sg_kink(xi, kb=KB, iters=80):
    N = 2 * int(10 * xi) + 1
    nc = (N - 1) // 2
    K = kb / xi**2
    n = np.arange(N)
    th = 4.0 * np.arctan(np.exp((n - nc) / xi))
    for _ in range(iters):
        R = kb * (th[2:] + th[:-2] - 2 * th[1:-1]) - K * np.sin(th[1:-1])
        if np.max(np.abs(R)) < 1e-13:
            break
        diag = -2.0 * kb - K * np.cos(th[1:-1])
        off = kb * np.ones(len(R) - 1)
        th[1:-1] += thomas(off, diag, off, -R)
    return th, K, nc


def eps_sg(th, kb, K):
    d = np.diff(th)
    link = (kb / 2.0) * d**2
    e = np.zeros_like(th)
    e[1:-1] = 0.5 * (link[:-1] + link[1:]) + K * (1.0 - np.cos(th[1:-1]))
    return e


def centroid(e, ref):
    """相对 ref 的一阶矩（能量质心偏离）。"""
    E = e.sum()
    x = np.arange(len(e), dtype=float)
    return (e * (x - ref)).sum() / E


def sech2_profile(n, x0, xi, A=1.0):
    arg = np.clip((n - x0) / xi, -300.0, 300.0)
    return A / np.cosh(arg) ** 2


def main():
    line = "-" * 78
    print(line)
    print("判据第 628 号：锚核落点对能量质心（首矩轮）")
    print(line)

    # -------- R1 冒烟 --------
    xi0 = 40.0
    th0, K0, nc0 = solve_sg_kink(xi0)
    e0 = eps_sg(th0, KB, K0)
    R0 = KB * (th0[2:] + th0[:-2] - 2 * th0[1:-1]) - K0 * np.sin(th0[1:-1])
    reflex = np.max(np.abs(e0 - e0[::-1])) / e0.max()          # 关于中心的镜像对称
    xcm0 = centroid(e0, nc0)
    print("\n[R1] 冒烟检查（xi=40）")
    print("  格点 N=%d | max|残差|=%.2e | 峰位=%d(=nc %d)" % (len(th0), np.max(np.abs(R0)), int(np.argmax(e0)), nc0))
    print("  能量密度镜像不对称度=%.2e | 首矩（相对钉）=%.3e" % (reflex, xcm0))
    r1_ok = (np.max(np.abs(R0)) < 1e-10) and (reflex < 1e-5)

    # -------- R2 单体重合 --------
    print("\n[R2] 单体：能量质心对锚核落点")
    print("  %-8s %-16s %-16s %-12s" % ("xi", "x_cm(格)", "x_cm/xi", "判定"))
    xi_scan = np.array([10.0, 20.0, 40.0, 80.0, 160.0])
    cm_single = np.array([])
    for xi in xi_scan:
        th, K, nc = solve_sg_kink(xi)
        e = eps_sg(th, KB, K)
        cm = centroid(e, nc)
        cm_single = np.append(cm_single, cm)
        print("  %-8g %-16.3e %-16.3e %-12s" % (xi, cm, cm / xi, "重合"))
    # 误差预算：首矩应严格为零（对称剖面）；实测落在求解器噪声底（有限差分与固定端条件的
    # 浮点不对称，量级 ~1e-9 相对），相对宽度 |x_cm|/xi 远小于 1。容差取 |x_cm| < 1e-3 格。
    r2_ok = np.max(np.abs(cm_single)) < 1e-3
    print("  [R2] |x_cm| 上界=%.2e 格（容差 1e-3 格）| 相对宽度上界=%.2e -> %s"
          % (np.max(np.abs(cm_single)), np.max(np.abs(cm_single / xi_scan)), "通过" if r2_ok else "不通过"))

    # -------- R3 两瓣分家（兼作 R2 的反例） --------
    print("\n[R3] 两瓣叠加像：分家量 = (1-q) D（钉在权重 q 的 A 瓣上）")
    xi = 40.0
    n = np.arange(2 * int(10 * xi) + 1)
    ncen = (len(n) - 1) // 2
    print("  %-8s %-10s %-10s %-16s %-16s %-10s" % ("q", "D/xi", "D(格)", "x_cm(实测)", "期望(1-q)D", "偏差"))
    r3_ok = True
    dev3_max = 0.0
    for q, D in [(0.5, 5 * xi), (0.5, 10 * xi), (0.5, 20 * xi), (0.25, 10 * xi), (0.75, 10 * xi)]:
        xA, xB = ncen - D / 2.0, ncen + D / 2.0
        eA = sech2_profile(n, xA, xi)
        eB = sech2_profile(n, xB, xi)
        e_img = q * eA + (1.0 - q) * eB
        xcm = centroid(e_img, ncen) + ncen          # 绝对坐标
        pred = q * xA + (1.0 - q) * xB
        sep = xcm - xA                              # 钉在 A 瓣
        pred_sep = (1.0 - q) * D
        dev = abs(sep - pred_sep)
        dev3_max = max(dev3_max, dev)
        print("  %-8.2f %-10g %-10g %-16.4f %-16.4f %-10.2e" % (q, D / xi, D, sep, pred_sep, dev))
    r3_ok = dev3_max < 0.5
    print("  [R3] 分家量实测对解析 (1-q)D 最大偏差=%.2e 格（容差 0.5）-> %s" % (dev3_max, "通过" if r3_ok else "不通过"))
    print("       反例作用：R2 的「单体」前提一旦去掉（改两瓣），分家量从 0 跳到 (1-q)D，故 R2 有牙")

    # -------- R4 交叉项压制 --------
    print("\n[R4] 分离瓣极限的线性性：交叉项随 D/xi 指数下降")
    print("  %-10s %-16s %-16s" % ("D/xi", "s(D)=sum(eA eB)/sum(eA^2)", "ln s"))
    Ds = np.array([2.0, 4.0, 6.0, 8.0, 10.0]) * xi
    s_arr = np.array([])
    for D in Ds:
        eA = sech2_profile(n, ncen - D / 2.0, xi)
        eB = sech2_profile(n, ncen + D / 2.0, xi)
        s = (eA * eB).sum() / (eA * eA).sum()
        s_arr = np.append(s_arr, s)
        print("  %-10g %-16.3e %-16.3f" % (D / xi, s, np.log(s)))
    # 渐近斜率取 D/xi >= 4 段拟合；解析渐近为 -2（sech^2 尾重叠 e^{-2D/xi}），容差 0.5
    fit = np.polyfit((Ds / xi)[1:], np.log(s_arr)[1:], 1)
    r4_ok = (abs(fit[0] + 2.0) <= 0.5) and (s_arr[-1] < 1e-6)
    print("  [R4] ln s 对 D/xi 的渐近斜率=%.4f（解析 -2，容差 0.5）| D=10xi 处 s=%.2e（容差 1e-6）-> %s"
          % (fit[0], s_arr[-1], "通过" if r4_ok else "不通过"))
    print("       即在 D >= 5xi 档上首矩线性式成立，误差按 s(D) 压制")

    # -------- R5 首矩盲性（如实登记） --------
    print("\n[R5] 首矩盲性：取甲（展布源）对 乙（点源在锚核）")
    xi = 40.0
    th, K, nc = solve_sg_kink(xi)
    e = eps_sg(th, KB, K)
    x = np.arange(len(e), dtype=float)
    cm_spread = (e * (x - nc)).sum() / e.sum()
    sig_spread = np.sqrt((e * (x - nc) ** 2).sum() / e.sum())
    e_pt = np.zeros_like(e)
    e_pt[nc] = 1.0
    cm_pt = (e_pt * (x - nc)).sum() / e_pt.sum()
    sig_pt = np.sqrt((e_pt * (x - nc) ** 2).sum() / e_pt.sum())
    print("  取甲展布源：首矩=%.3e 格 | 二阶矩 sigma=%.4f 格" % (cm_spread, sig_spread))
    print("  乙点源（钉处）：首矩=%.3e 格 | 二阶矩 sigma=%.4f 格" % (cm_pt, sig_pt))
    print("  [R5] 首矩差=%.2e（两者不可分辨）；二阶矩差=%.4f 格（可分辨）-> 首矩不能判定取甲/乙，须配第 627 号" % (abs(cm_spread - cm_pt), sig_spread - sig_pt))
    r5_ok = (abs(cm_spread - cm_pt) < 1e-3) and (sig_spread - sig_pt > 1.0)

    # -------- 图 --------
    fig, ax = plt.subplots(1, 3, figsize=(15.0, 4.2))
    xi = 40.0
    th, K, nc = solve_sg_kink(xi)
    e = eps_sg(th, KB, K)
    u = np.arange(len(e)) - nc
    ax[0].plot(u, e / e.max(), lw=1.4)
    ax[0].axvline(0.0, color="C3", ls="--", lw=1.2, label="锚核落点")
    ax[0].axvline(centroid(e, nc), color="C0", ls=":", lw=1.6, label="能量质心")
    ax[0].set_xlim(-3 * xi, 3 * xi)
    ax[0].set_xlabel("n - n_c（格）")
    ax[0].set_ylabel("能量密度 / 峰值")
    ax[0].set_title("(a) 单体：质心与锚核落点重合")
    ax[0].legend(fontsize=8)

    q, D = 0.5, 10 * xi
    n = np.arange(2 * int(10 * xi) + 1)
    ncen = (len(n) - 1) // 2
    eA = sech2_profile(n, ncen - D / 2.0, xi)
    eB = sech2_profile(n, ncen + D / 2.0, xi)
    e_img = q * eA + (1 - q) * eB
    ax[1].plot(n - ncen, e_img / e_img.max(), lw=1.4)
    ax[1].axvline(-D / 2.0, color="C3", ls="--", lw=1.2, label="钉（A 瓣）")
    ax[1].axvline(centroid(e_img, ncen), color="C0", ls=":", lw=1.6, label="能量质心")
    ax[1].set_xlim(-D, D)
    ax[1].set_xlabel("n - n_c（格）")
    ax[1].set_ylabel("能量密度 / 峰值")
    ax[1].set_title("(b) 两瓣（q=1/2）：质心在中点，钉在一瓣")
    ax[1].legend(fontsize=8)

    ax[2].semilogy(Ds / xi, s_arr, "o-")
    ax[2].set_xlabel("D / xi")
    ax[2].set_ylabel("交叉项 s(D)")
    ax[2].set_title("(c) 交叉项随瓣间距指数压制")
    fig.tight_layout()
    out = os.path.join(IMG, "anchor_vs_centroid.png")
    fig.savefig(out, dpi=150)
    print("\n[图] 已存 %s" % out)

    print("\n" + line)
    print("判据汇总: R1 %s | R2 %s | R3 %s | R4 %s | R5 %s"
          % tuple("通过" if v else "不通过" for v in (r1_ok, r2_ok, r3_ok, r4_ok, r5_ok)))
    print(line)


if __name__ == "__main__":
    main()
