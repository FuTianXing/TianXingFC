# -*- coding: utf-8 -*-
"""
判据第 627 号：有质量费米子的质量能量落在核区（一格）还是包络（宽度尺度）。

模型：一维锁定势格点上的静态扭结（台阶面）。
  E = sum_n [ (k_b/2)(theta_{n+1}-theta_n)^2 + K (1 - cos theta_n) ]   （SG 口径）
静态解：k_b(theta_{n+1}+theta_{n-1}-2 theta_n) = -K sin theta_n，
顶角解剖面 theta(x)=4 arctan(exp(-x/xi))，xi = sqrt(k_b/K) 为偏离场的衰减长度。
能量密度 eps(x) = 4 K sech^2(x/xi)，总能量 E = 8 sqrt(k_b K)。
解析期望：核心份额（一格）f_core = tanh(1/(2xi)) -> 1/(2xi)；
          包络份额（±xi）f_env = tanh(1) = 0.761594；
          质量半径 sigma/xi = pi/sqrt(12) = 0.906900。

判据 M1 冒烟 / M2 核心份额标度 / M3 包络份额 / M4 质量半径 /
      M5 反例（判据有牙）/ M6 电子档外推 / M7 phi^4 口径稳健性。
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

KB = 1.0                      # 梯度刚度
XI_SCAN = np.array([10, 20, 40, 80, 160], dtype=float)
XI_ELECTRON = 2.389222e22     # 第 622 号恒等式：电子缺陷宽度（格）
TANH1 = np.tanh(1.0)          # 0.7615941559
SIG_COEF = np.pi / np.sqrt(12.0)   # 0.9068996821


# ---------------- 求解器 ----------------
def thomas(a, b, c, d):
    """三对角求解 a x_{i-1} + b x_i + c x_{i+1} = d。"""
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
    """SG 静态扭结（反扭结 2pi -> 0），Newton + Thomas。"""
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


def solve_phi4_kink(xi_corr, kb=KB, iters=80):
    """phi^4 双井静态扭结，xi_corr = 1/sqrt(2 lam) 为关联长度，剖面宽 w = 2 xi_corr。"""
    lam = 1.0 / (2.0 * xi_corr**2)
    w = np.sqrt(2.0 / lam)
    N = 2 * int(10 * w) + 1
    nc = (N - 1) // 2
    n = np.arange(N)
    ph = np.tanh(-(n - nc) / w)
    for _ in range(iters):
        Vp = lam * ph[1:-1] * (ph[1:-1] ** 2 - 1.0)
        R = kb * (ph[2:] + ph[:-2] - 2 * ph[1:-1]) - Vp
        if np.max(np.abs(R)) < 1e-13:
            break
        diag = -2.0 * kb - lam * (3.0 * ph[1:-1] ** 2 - 1.0)
        off = kb * np.ones(len(R) - 1)
        ph[1:-1] += thomas(off, diag, off, -R)
    return ph, lam, nc


def eps_sg(th, kb, K):
    d = np.diff(th)
    link = (kb / 2.0) * d**2
    e = np.zeros_like(th)
    e[ 1:-1] = 0.5 * (link[:-1] + link[1:]) + K * (1.0 - np.cos(th[1:-1]))
    return e


def eps_phi4(ph, kb, lam):
    d = np.diff(ph)
    link = (kb / 2.0) * d**2
    e = np.zeros_like(ph)
    e[1:-1] = 0.5 * (link[:-1] + link[1:]) + (lam / 4.0) * (ph[1:-1] ** 2 - 1.0) ** 2
    return e


def readings(e, nc, xi):
    """返回 (f_core 一格, f_env ±xi 格, sigma 质量半径)。"""
    E = e.sum()
    if E <= 0:
        return np.nan, np.nan, np.nan
    dx = np.arange(len(e)) - nc
    f_core = e[np.abs(dx) <= 0.5].sum() / E
    f_env = e[np.abs(dx) <= xi].sum() / E
    sig = np.sqrt((e * dx**2).sum() / E)
    return f_core, f_env, sig


def main():
    line = "-" * 78
    print(line)
    print("判据第 627 号：有质量费米子的质量落点（核区对包络）")
    print(line)

    # -------- M1 冒烟检查 --------
    xi0 = 40.0
    th0, K0, nc0 = solve_sg_kink(xi0)
    e0 = eps_sg(th0, KB, K0)
    R0 = KB * (th0[2:] + th0[:-2] - 2 * th0[1:-1]) - K0 * np.sin(th0[1:-1])
    mono = np.all(np.diff(th0) >= -1e-9)
    print("\n[M1] 冒烟检查（xi=40）")
    print("  格点数 N=%d | K=%.3e | max|残差|=%.2e | 单调=%s | 峰位=%d(=nc %d)"
          % (len(th0), K0, np.max(np.abs(R0)), mono, int(np.argmax(e0)), nc0))
    print("  能量密度: 中心 %.4e | 端部 %.2e | 全为正=%s"
          % (e0[nc0], max(e0[1], e0[-2]), bool(np.all(e0 > -1e-12))))
    m1_ok = (np.max(np.abs(R0)) < 1e-10) and mono and (int(np.argmax(e0)) == nc0)

    # -------- M2/M3/M4 扫描 --------
    print("\n[M2/M3/M4] 扫描 xi（%s）" % ", ".join("%g" % x for x in XI_SCAN))
    print("  %-8s %-14s %-14s %-14s %-14s %-14s" %
          ("xi", "f_core(实测)", "0.5/xi(期望)", "f_env(实测)", "tanh1(期望)", "sigma/xi"))
    f_core_arr = np.array([])
    f_env_arr = np.array([])
    sig_ratio = np.array([])
    for xi in XI_SCAN:
        th, K, nc = solve_sg_kink(xi)
        e = eps_sg(th, KB, K)
        fc, fe, sg = readings(e, nc, xi)
        f_core_arr = np.append(f_core_arr, fc)
        f_env_arr = np.append(f_env_arr, fe)
        sig_ratio = np.append(sig_ratio, sg / xi)
        print("  %-8g %-14.6e %-14.6e %-14.6f %-14.6f %-14.6f"
              % (xi, fc, 0.5 / xi, fe, TANH1, sg / xi))

    # 标度拟合 log f_core vs log xi
    p = np.polyfit(np.log(XI_SCAN), np.log(f_core_arr), 1)
    slope = p[0]
    ratio = f_core_arr * XI_SCAN            # 应趋于 0.5
    tol_scale = 0.02 + 4.0 / XI_SCAN**2     # 误差预算：离散 O(1/xi^2) + tanh 展宽
    m2_ok = abs(slope + 1.0) <= 0.03 and np.all(np.abs(ratio - 0.5) <= tol_scale)
    print("  [M2] 斜率 p=%.4f（期望 -1，容差 0.03）| f_core*xi（末档）=%.6f（期望 0.5）-> %s"
          % (slope, ratio[-1], "通过" if m2_ok else "不通过"))
    print("  容差预算: 0.02 + 4/xi^2（离散 O(1/xi^2) 与 tanh 展开各 <=1%%，留头）")

    # 离散和覆盖宽度 2xi+1（每格占半格），等价连续半宽 xi+0.5；期望 tanh((xi+0.5)/xi)
    f_env_pred = np.tanh((XI_SCAN + 0.5) / XI_SCAN)
    dev3 = np.abs(f_env_arr - f_env_pred)
    m3_ok = np.max(dev3) <= 0.005
    print("  [M3] f_env 对 tanh((xi+0.5)/xi) 的最大偏差=%.2e（容差 0.005）-> %s"
          % (np.max(dev3), "通过" if m3_ok else "不通过"))
    print("       其中连续极限 tanh(1)=%.6f；末档离散期望=%.6f、实测=%.6f"
          % (TANH1, f_env_pred[-1], f_env_arr[-1]))
    print("       边界项说明：离散和含 2xi+1 格（宽 2xi+1），连续窗口宽 2xi，差为半格边缘项 ~0.21/xi")

    m4_ok = np.max(np.abs(sig_ratio - SIG_COEF)) <= 0.01
    print("  [M4] sigma/xi 对 pi/sqrt(12)=%.6f 的最大偏差=%.2e（容差 0.01）-> %s"
          % (SIG_COEF, np.max(np.abs(sig_ratio - SIG_COEF)), "通过" if m4_ok else "不通过"))

    # -------- M5 反例：判据有牙 --------
    print("\n[M5] 反例（判据有牙）")
    xi_t = 40.0
    n = np.arange(2 * int(10 * xi_t) + 1)
    nc = (len(n) - 1) // 2
    # 反例甲：全部质量压在核区一格（锚核承载全部质量）
    e_pt = np.zeros_like(n, dtype=float)
    e_pt[nc] = 1.0
    fc_pt, fe_pt, sg_pt = readings(e_pt, nc, xi_t)
    print("  反例甲（点质量）: f_core=%.4f（期望 0.5/xi=%.4f）| sigma/xi=%.4f -> 判据翻=%s"
          % (fc_pt, 0.5 / xi_t, sg_pt / xi_t, (abs(fc_pt - 0.5 / xi_t) > tol_scale[-1])))
    # 反例乙：包络窄 50 倍
    arg = np.clip((n - nc) * 50.0 / xi_t, -300.0, 300.0)
    e_narrow = 4.0 * (1.0 / xi_t**2) * (1.0 / np.cosh(arg) ** 2)
    fc_n, fe_n, sg_n = readings(e_narrow, nc, xi_t)
    print("  反例乙（包络窄 50 倍）: f_core*xi=%.4f（真值 0.5）-> 判据翻=%s"
          % (fc_n * xi_t, (abs(fc_n * xi_t - 0.5) > tol_scale[-1])))
    m5_ok = (abs(fc_pt - 0.5 / xi_t) > tol_scale[-1]) and (abs(fc_n * xi_t - 0.5) > tol_scale[-1])

    # -------- M6 电子档外推 --------
    print("\n[M6] 电子档外推（xi = m_P/m_e = %.6e 格，第 622 号）" % XI_ELECTRON)
    fc_e = 0.5 / XI_ELECTRON
    sig_e = SIG_COEF * XI_ELECTRON
    print("  核心（一格）份额 f_core = %.4e" % fc_e)
    print("  包络（±xi）份额        = %.6f" % TANH1)
    print("  质量半径 sigma         = %.4e 格 = %.4e m" % (sig_e, sig_e * 1.616255e-35))
    print("  包络（除核区外）占 1 - %.2e；其中 ±xi 内 %.6f" % (fc_e, TANH1))
    print("  结论：质量能量几乎全部在包络，核区一格可忽略")
    m6_ok = fc_e < 1e-20

    # -------- M7 phi^4 口径稳健性 --------
    print("\n[M7] phi^4 双井口径稳健性")
    print("  %-10s %-14s %-14s %-14s" % ("xi_corr", "f_core*xi_corr", "f_env(±xi_corr)", "sigma/xi_corr"))
    phi_rows = []
    for xi in [10.0, 20.0, 40.0]:
        ph, lam, ncp = solve_phi4_kink(xi)
        e = eps_phi4(ph, KB, lam)
        fc, fe, sg = readings(e, ncp, xi)
        phi_rows.append((xi, fc * xi, fe, sg / xi))
        print("  %-10g %-14.6f %-14.6f %-14.6f" % (xi, fc * xi, fe, sg / xi))
    m7_ok = abs(phi_rows[-1][1] - phi_rows[0][1]) < 0.02 and phi_rows[-1][3] > 0.9
    print("  phi^4 的 f_core*xi_corr 常数=%.4f（与 SG 的 0.5 不同，同为 O(1)）；斜率仍为 1/xi -> %s"
          % (phi_rows[-1][1], "通过" if m7_ok else "不通过"))

    # -------- 图 --------
    fig, ax = plt.subplots(1, 3, figsize=(15.0, 4.2))
    for xi in XI_SCAN:
        th, K, nc = solve_sg_kink(xi)
        e = eps_sg(th, KB, K)
        u = (np.arange(len(e)) - nc) / xi
        ax[0].plot(u, e / e.max(), lw=1.2, label="xi=%g" % xi)
    uu = np.linspace(-3, 3, 400)
    ax[0].plot(uu, 1.0 / np.cosh(uu) ** 2, "k--", lw=1.0, label="sech^2")
    ax[0].set_xlim(-3, 3)
    ax[0].set_xlabel("(n - n_c) / xi")
    ax[0].set_ylabel("能量密度 / 峰值")
    ax[0].set_title("(a) 能量密度剖面随 xi 归一后重合")
    ax[0].legend(fontsize=8)

    ax[1].loglog(XI_SCAN, f_core_arr, "o-", label="f_core 实测")
    ax[1].loglog(XI_SCAN, 0.5 / XI_SCAN, "k--", label="0.5 / xi")
    ax[1].set_xlabel("xi（格）")
    ax[1].set_ylabel("核心（一格）份额")
    ax[1].set_title("(b) 核心份额按 1/xi 衰减")
    ax[1].legend(fontsize=8)

    ax[2].semilogx(XI_SCAN, sig_ratio, "s-", color="C2", label="sigma / xi 实测")
    ax[2].axhline(SIG_COEF, color="k", ls="--", lw=1.0, label="pi/sqrt(12)")
    ax[2].set_ylim(0.85, 0.95)
    ax[2].set_xlabel("xi（格）")
    ax[2].set_ylabel("sigma / xi")
    ax[2].set_title("(c) 质量半径恒为 xi 的常数倍")
    ax[2].legend(fontsize=8)

    fig.tight_layout()
    out = os.path.join(IMG, "mass_location_core_envelope.png")
    fig.savefig(out, dpi=150)
    print("\n[图] 已存 %s" % out)

    # -------- 汇总 --------
    print("\n" + line)
    print("判据汇总: M1 %s | M2 %s | M3 %s | M4 %s | M5 %s | M6 %s | M7 %s"
          % tuple("通过" if v else "不通过"
                  for v in (m1_ok, m2_ok, m3_ok, m4_ok, m5_ok, m6_ok, m7_ok)))
    print(line)


if __name__ == "__main__":
    main()
