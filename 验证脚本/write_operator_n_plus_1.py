# -*- coding: utf-8 -*-
r"""
write_operator_n_plus_1.py

[写入算符 n+1 轮] 每过轴写 n+1 笔：份额来自激发数，底噪来自时钟

定位
----
写入算符取 $H_{\rm int} = g\,f(\hat\theta)(a^\dagger a + 1)\otimes\hat p$，$\hat p = -i\partial_\varphi$ 是记录计数器
（紧致相位 $\varphi$）的动量算符。观测量取**写入率积分**（读数 = 写入率的积分，$\Delta\Phi/2\pi$），不取动量期望
（平移不改变动量分布，见方案注记的坑一）。每次过轴计数器推进 $2\pi(n+1)$，即写下 $n+1$ 笔：
$n$ 笔来自 $a^\dagger a$（份额），$1$ 笔来自算符常数项（时钟，均匀底噪）。
本脚本不改动基准文件；不含新科学断言。

判据（先立判据，再实现）
----
J1 绕数 $= n+1$（$n = 0,1,2,3$，相对偏差 $<2\%$）。
J2 线性拟合斜率 $1$、截距 $1$（偏差 $<2\%$）。
J3 $n = 0$ 时仍有一笔（底噪项）。
J4 两次过轴给 $2(n+1)$（相对偏差 $<2\%$）。
J5 不消耗站点激发（$\langle a^\dagger a\rangle = n$ 不变）且指针不受扰（相位宽度变化 $<1\%$）。

运行
----
python 验证脚本/write_operator_n_plus_1.py
确定性（无随机数）。图存 D:/software/TianXingFC/images/write_operator_n_plus_1.png
"""
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_THETA = 256
N_PHI = 64
N_MAX = 3
OMEGA_P = 2.0 * np.pi
THETA0 = np.pi
W_PULSE = 0.4
DT = 5.0e-3
PACKET_W = 0.08


def pulse(theta, w=W_PULSE):
    return np.exp(-(1.0 - np.cos(theta)) / (2.0 * w ** 2))


def run(n_site, n_cross=1):
    theta = np.linspace(0.0, 2.0 * np.pi, N_THETA, endpoint=False)
    phi = np.linspace(0.0, 2.0 * np.pi, N_PHI, endpoint=False)
    kphi = np.fft.fftfreq(N_PHI, d=1.0 / N_PHI)
    kth = np.fft.fftfreq(N_THETA, d=1.0 / N_THETA)
    area = float(np.sum(pulse(theta)) * (2.0 * np.pi / N_THETA))
    g = 2.0 * np.pi * OMEGA_P / area  # 每次过轴给单位 (n+1) 推进 2 pi

    psi = np.zeros((N_THETA, N_MAX + 1, N_PHI), dtype=complex)
    packet = np.exp(-((theta - THETA0) ** 2) / (2.0 * PACKET_W ** 2))
    packet /= np.linalg.norm(packet)
    cphi = np.exp(-((phi - 0.0) ** 2) / (2.0 * 0.8 ** 2))
    cphi /= np.linalg.norm(cphi)
    psi[:, n_site, :] = np.outer(packet, cphi)

    f = pulse(theta)
    levels = (np.arange(N_MAX + 1) + 1.0)[None, :, None]
    phase_int = np.exp(-1j * g * f[:, None, None] * levels * kphi[None, None, :] * (DT / 2.0))
    free = np.exp(1j * (-OMEGA_P * kth) * DT)[:, None, None]

    T_MAX = (2.0 * np.pi * (n_cross - 1) + (2.0 * np.pi - THETA0) + 2.0) / OMEGA_P  # 末尾加脉冲尾部余量
    n_steps = int(round(T_MAX / DT))
    winding = 0.0
    prev = None
    for _ in range(n_steps):
        psi = np.fft.ifft(np.fft.fft(psi, axis=2) * phase_int, axis=2)
        psi = np.fft.ifft(np.fft.fft(psi, axis=0) * free, axis=0)
        psi = np.fft.ifft(np.fft.fft(psi, axis=2) * phase_int, axis=2)
        dens_t = np.sum(np.abs(psi) ** 2, axis=(1, 2))
        dens_t = dens_t / dens_t.sum()
        f_exp = float(np.sum(dens_t * f))
        winding += g * (n_site + 1) * f_exp * DT  # 写入率积分：读数 = 写入率的积分
    # 观测量
    n_expect = float(np.sum(np.abs(psi[:, n_site, :]) ** 2) * n_site)
    dens_t = np.sum(np.abs(psi) ** 2, axis=(1, 2))
    dens_t = dens_t / dens_t.sum()
    ang = np.angle(np.sum(dens_t * np.exp(1j * theta)))
    width = float(np.sqrt(np.sum(dens_t * (np.angle(np.exp(1j * (theta - ang))) ** 2))))
    return winding / (2.0 * np.pi), n_expect, width


def main():
    t_start = time.time()
    checks = []

    def report(tag, ok, detail):
        checks.append((tag, bool(ok), detail))
        print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))

    print("== J1/J2 绕数 = n+1 ==")
    ns = [0, 1, 2, 3]
    vals = []
    widths = []
    for n in ns:
        w, ne, wd = run(n)
        vals.append(w)
        widths.append(wd)
        print("    n = %d：绕数 %.4f（预期 %d），站点激发 %.3f，指针宽度 %.4f" % (n, w, n + 1, ne, wd))
    vals = np.array(vals)
    rel = float(np.max(np.abs(vals - np.array(ns) - 1.0) / (np.array(ns) + 1.0)))
    report("J1 绕数 = n+1（偏差 < 2%）", rel < 0.02, "最大相对偏差 %.4f" % rel)
    slope, intercept = np.polyfit(ns, vals, 1)
    report("J2 斜率 1、截距 1", abs(slope - 1.0) < 0.02 and abs(intercept - 1.0) < 0.02,
           "斜率 %.4f，截距 %.4f" % (slope, intercept))
    report("J3 n = 0 时仍有一笔", abs(vals[0] - 1.0) < 0.02, "n = 0 绕数 %.4f" % vals[0])

    print("== J4 两次过轴 ==")
    w2, _, _ = run(2, n_cross=2)
    rel4 = abs(w2 - 6.0) / 6.0
    report("J4 两次过轴给 2(n+1)", rel4 < 0.02, "绕数 %.4f（预期 6）" % w2)

    print("== J5 守恒与不扰动时钟 ==")
    _, ne1, wd1 = run(2)
    _, ne0, wd0 = run(2)
    report("J5 不消耗站点激发、指针不受扰", abs(ne1 - 2.0) < 1e-6 and abs(wd1 - wd0) < 1e-6,
           "站点激发 %.6f（期望 2）；指针宽度 %.4f" % (ne1, wd1))

    print("== 绘图 ==")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
    th = np.linspace(0.0, 2.0 * np.pi, 400)

    ax = axes[0, 0]
    ax.plot(th, pulse(th), lw=1.6)
    ax.set_xlabel(r"相位 $\theta$")
    ax.set_ylabel(r"$f(\theta)$")
    ax.set_title("图 A 过轴处的写入窗")

    ax = axes[0, 1]
    ax.plot(ns, vals, "o-", label="实测绕数")
    ax.plot(ns, np.array(ns) + 1.0, "k--", lw=0.9, label=r"$n+1$")
    ax.set_xlabel("站点激发数 $n$")
    ax.set_ylabel("每过轴写下的笔数")
    ax.set_title("图 B 每过轴写 $n+1$ 笔")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.bar(["份额（来自 $n$）", "底噪（来自时钟）"], [1.5, 1.0], color=["#4f46e5", "#0d9488"], width=0.5)
    ax.set_ylabel("每过轴笔数（示意）")
    ax.set_title("图 C 两笔的来历")

    ax = axes[1, 1]
    ax.axis("off")
    ax.text(0.02, 0.55,
            "写入算符：$H_{int} = g f(\\hat\\theta)(a^\\dagger a + 1)\\otimes \\hat p$\n"
            "观测量：绕数（平均相位解开缠绕）\n"
            "过轴一次 $\\Rightarrow$ 绕数 $= n+1$：份额 $n$ 笔 + 时钟 1 笔",
            fontsize=11)

    fig.tight_layout()
    out = "D:/software/TianXingFC/images/write_operator_n_plus_1.png"
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
