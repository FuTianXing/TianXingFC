#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
second_order_source_lock_wave.py

判据 WV：次阶源锁定复验·第三段（波动扇区的时间响应）。

背景
----
第一段（SS）核对源侧条件，第二段（TR）把约束结构推到时间维。
第三段进入波动扇区：基本动力学取阻尼波动方程
    theta_tt - c^2 theta_xx + gamma theta_t = s(x, t),  c = 1,
核对响应侧结构（第五力风险真正所在的一侧）：
  WV1 静态桥接：多模源的稳态响应等于约束解（泊松像）；
  WV2 前缘与群速：零直流发射波包的速度等于 c_s = c，前缘无超光速分量；
  WV3 阻尼与色散：单模余振的幅度按 exp(-gamma t / 2) 衰减，频率等于
      sqrt(k^2 - gamma^2/4)（c_s = c 的传播支）；
  WV4 频率响应：单支格林函数核对（共振落在色散 omega = k 上）；
  WV5 可解性条件：净源非零时零模线性增长（周期模型的口径）。

运行：python 验证脚本/second_order_source_lock_wave.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "images", "second_order_source_lock_wave.png")

L = 200.0
N = 1024
dx = L / N
x = np.linspace(-L / 2, L / 2, N, endpoint=False)
kfft = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
DT = 0.01


def d2(a):
    return np.real(np.fft.ifft(-(kfft ** 2) * np.fft.fft(a)))


def poisson(S):
    Sc = S - np.mean(S)
    return np.real(np.fft.ifft(-np.fft.fft(Sc) / (kfft ** 2 + (kfft == 0.0))))


def run(sfun, T, gamma, keep_times=None, dt=DT):
    """RK4 积分 theta_tt = theta_xx - gamma theta_t + s(x,t)，零初值。"""
    th = np.zeros(N)
    v = np.zeros(N)
    nt = int(round(T / dt))
    keep = {}
    want = None if keep_times is None else np.asarray(keep_times, dtype=float)
    for i in range(nt + 1):
        t = i * dt
        if want is not None:
            j = np.argmin(np.abs(want - t))
            if abs(want[j] - t) < 0.5 * dt:
                keep[round(float(want[j]), 6)] = th.copy()
        if i == nt:
            break
        def acc(a, b, tt):
            return d2(a) - gamma * b + sfun(tt)
        k1t, k1v = v, acc(th, v, t)
        k2t, k2v = v + 0.5 * dt * k1v, acc(th + 0.5 * dt * k1t, v + 0.5 * dt * k1v, t + 0.5 * dt)
        k3t, k3v = v + 0.5 * dt * k2v, acc(th + 0.5 * dt * k2t, v + 0.5 * dt * k2v, t + 0.5 * dt)
        k4t, k4v = v + dt * k3v, acc(th + dt * k3t, v + dt * k3v, t + dt)
        th = th + dt * (k1t + 2 * k2t + 2 * k3t + k4t) / 6.0
        v = v + dt * (k1v + 2 * k2v + 2 * k3v + k4v) / 6.0
    return th, v, keep


def kick_profile(t, t0=6.0, sigma=1.5):
    """零直流时间包络（高斯导数）：int g dt = 0。"""
    u = (t - t0) / sigma
    return -(u / sigma) * np.exp(-0.5 * u * u)


def main():
    print("=" * 78)
    print("second_order_source_lock_wave.py")
    print("判据 WV：次阶源锁定复验·第三段（波动扇区的时间响应）")
    print("=" * 78)

    ok = {}
    bump0 = np.exp(-(x / 2.0) ** 2)          # 发射源：不做均值扣除（避免全域均匀分量）
    bumpm = bump0 - np.mean(bump0)           # 净源核对用

    # ---------- WV1 静态桥接（多模源） ----------
    print()
    print("WV1 静态桥接：多模源的稳态响应对约束解（泊松像）")
    k0a = 2.0 * np.pi / L * 10.0
    smulti = np.cos(k0a * x) + 0.5 * np.cos(2.0 * k0a * x)
    ref1 = -poisson(smulti)
    devs1 = []
    for gamma in (0.4, 0.5):
        T = 100.0
        th, _, _ = run(lambda t: smulti, T, gamma)
        dev = np.max(np.abs(th - ref1)) / np.max(np.abs(ref1))
        devs1.append(dev)
        print("   gamma = %.1f : 稳态偏差 = %.2e（衰减率 gamma/2，T = %.0f）" % (gamma, dev, T))
    ok["WV1"] = max(devs1) < 1e-6

    # ---------- WV2 前缘与群速 ----------
    print()
    print("WV2 前缘与群速（零直流发射，gamma = 0.02）")
    def kick(t):
        return bump0 * kick_profile(t)
    times = (15.0, 20.0, 25.0, 30.0, 35.0)
    th_keep = run(kick, 35.0, 0.02, keep_times=list(times))[2]
    pos = []
    for tt in times:
        arr = th_keep[round(tt, 6)]
        xr = x[x > 3.0]
        ar = arr[x > 3.0]
        pos.append(xr[np.argmax(np.abs(ar))])
    vfit = np.polyfit(times, pos, 1)[0]
    early = np.max(np.abs(th_keep[round(15.0, 6)][x > 20.0]))
    late_scale = np.max(np.abs(th_keep[round(35.0, 6)]))
    print("   波包峰值位置：%s" % ", ".join("%.2f" % p for p in pos))
    print("   拟合群速 = %.4f（期望 1）" % vfit)
    print("   因果前缘：t = 15 时 x > 20 的响应 = %.2e（相对晚场 %.2e）"
          % (early, early / late_scale))
    ok["WV2"] = abs(vfit - 1.0) < 0.02 and early / late_scale < 1e-3

    # ---------- WV3 阻尼与色散（单模余振） ----------
    print()
    print("WV3 阻尼与色散：单模余振（源在 t = 20 关闭）")
    k0b = 2.0 * np.pi / L * 8.0
    ib3 = int(np.argmin(np.abs(kfft - k0b)))
    rates, freq_dev = [], []
    for gamma in (0.15, 0.3):
        W = 40.0 if gamma < 0.2 else 20.0
        ts = np.arange(20.0, 20.0 + W + 1e-9, 0.25)
        def src(t, g=gamma):
            turn = (1.0 - np.exp(-t / 2.0)) * (1.0 - np.exp(-(20.0 - t) / 2.0)) * (t < 20.0)
            return np.cos(k0b * x) * turn
        keep = run(src, 20.0 + W, gamma, keep_times=list(ts))[2]
        z = np.array([np.fft.fft(keep[round(float(tt), 6)])[ib3].real for tt in ts])
        # 驻模的系数会过零：用阻尼正弦的最小二乘扫描拟合 (a, omega)
        om_pred = np.sqrt(k0b ** 2 - gamma ** 2 / 4.0)
        ag = np.linspace(0.5 * gamma / 2.0, 1.5 * gamma / 2.0, 61)
        wg = np.linspace(0.85 * om_pred, 1.15 * om_pred, 121)
        best = None
        for a_ in ag:
            E = np.exp(-a_ * ts)
            for om_ in wg:
                A = np.vstack([E * np.cos(om_ * ts), E * np.sin(om_ * ts)]).T
                coef, *_ = np.linalg.lstsq(A, z, rcond=None)
                r = float(np.sum((z - A @ coef) ** 2))
                if best is None or r < best[0]:
                    best = (r, a_, om_)
        a_fit, om_fit = best[1], best[2]
        rates.append(-a_fit); freq_dev.append(abs(om_fit - om_pred) / om_pred)
        print("   gamma = %.2f : 衰减率 = %.4f（期望 %.4f）；频率 = %.4f（期望 %.4f）"
              % (gamma, -a_fit, -gamma / 2.0, om_fit, om_pred))
    ok["WV3"] = (all(abs(r + g / 2.0) / (g / 2.0) < 0.05 for r, g in zip(rates, (0.15, 0.3)))
                 and max(freq_dev) < 0.02)

    # ---------- WV4 频率响应 ----------
    print()
    print("WV4 频率响应：单支格林函数核对（gamma = 0.15）")
    gamma4 = 0.15
    k0 = 2.0 * np.pi / L * 8.0
    omegas = k0 * np.array([0.4, 0.7, 0.9, 1.0, 1.1, 1.4, 1.8])
    devs4, amps4 = [], []
    for om in omegas:
        T = 80.0
        def src(t, om_=om):
            turn = 1.0 - np.exp(-t / 2.0)
            return np.cos(k0 * x) * np.cos(om_ * t) * turn
        keep = run(src, T, gamma4, keep_times=list(np.arange(64.0, 80.0, 0.25)))[2]
        ts = np.array(sorted(keep.keys()))
        ib = int(np.argmin(np.abs(kfft - k0)))
        ks = np.array([np.fft.fft(keep[t])[ib] for t in ts])
        A = np.vstack([np.cos(om * ts), np.sin(om * ts)]).T
        coef, *_ = np.linalg.lstsq(A, ks.real, rcond=None)
        amp = np.sqrt(coef[0] ** 2 + coef[1] ** 2) * 2.0 / N
        pred = 1.0 / np.sqrt((k0 ** 2 - om ** 2) ** 2 + (gamma4 * om) ** 2)
        devs4.append(abs(amp - pred) / pred)
        amps4.append(amp)
        print("   omega/k0 = %.1f : 响应幅度 = %.4f , 解析 = %.4f , 相对偏差 = %.2e"
              % (om / k0, amp, pred, abs(amp - pred) / pred))
    ok["WV4"] = max(devs4) < 0.05

    # ---------- WV5 可解性条件（净源零） ----------
    print()
    print("WV5 可解性条件：净源 epsilon 驱动零模线性增长")
    gamma5 = 0.5
    slopes = []
    for eps in (0.0, 0.002, 0.005):
        def src(t, e=eps):
            return bumpm + e
        keep = run(src, 60.0, gamma5, keep_times=[20.0, 30.0, 40.0, 50.0, 60.0])[2]
        means = np.array([np.mean(keep[t]) for t in (20.0, 30.0, 40.0, 50.0, 60.0)])
        slope = np.polyfit([20.0, 30.0, 40.0, 50.0, 60.0], means, 1)[0]
        slopes.append(slope)
        print("   epsilon = %.3f : 零模斜率 = %.6f（期望 %.6f）" % (eps, slope, eps / gamma5))
    ok["WV5"] = abs(slopes[1] - 0.002 / gamma5) < 0.02 * (0.002 / gamma5) and abs(slopes[0]) < 1e-6

    # ---------- 图 ----------
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.6))
    ax = axes[0]
    th, _, _ = run(lambda t: smulti, 100.0, 0.4)
    ax.plot(x, th, label="波动稳态")
    ax.plot(x, ref1, "--", label="约束解")
    ax.set_xlim(-20, 20); ax.set_xlabel("x"); ax.set_ylabel("theta")
    ax.set_title("(a) 静态桥接"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot(times, pos, "o", label="波包峰值位置")
    ax.plot(times, np.array(times) - 6.0, "--", label="速度 1 参考")
    ax.set_xlabel("t"); ax.set_ylabel("x_peak")
    ax.set_title("(b) 群速 = c"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax = axes[2]
    preds = 1.0 / np.sqrt((k0 ** 2 - omegas ** 2) ** 2 + (gamma4 * omegas) ** 2)
    ax.plot(omegas / k0, preds, "-", label="解析单支")
    ax.plot(omegas / k0, amps4, "o", ms=6, label="响应读数")
    ax.set_xlabel("omega / k0"); ax.set_ylabel("幅度")
    ax.set_title("(c) 频率响应"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, dpi=160)
    print()
    print("图已存：%s" % FIG)

    print()
    print("=" * 78)
    print("总判定：" + "，".join("%s %s" % (k, "通过" if v_ else "失败") for k, v_ in ok.items()))
    print("=" * 78)


if __name__ == "__main__":
    main()
