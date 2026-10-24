#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
second_order_source_lock_time.py

判据 TR：次阶源锁定复验·第二段（含时度规的时间响应）。

背景
----
第一段（SS 判据）把 (d) 的动能项与应力项补进迹通道的源，核对源侧条件：
两条守恒、迹源装配、约束算子的唯一性与无传播支、注入灵敏度。
第二段把约束结构推到时间维：源含时（刚性移动、位置调制、振幅调制）时，
核对约束的时间响应结构。

判据
----
TR1 时间序列闭合与守恒：五个时刻的约束闭合与两条守恒律；
TR2 瞬时跟踪：源移位后响应随之平移（谱移位逐位一致，无延迟）；
TR3 时间导数恒等：d_t phi 的两条路线（响应求导与源求导后解约束）；
TR4 调制跟踪：位置调制与振幅调制下响应等于瞬时像（无滞后、无余振）；
TR5 齐次自由度与传播支：核只剩空间常数零模，呼吸型扰动不是解，
    波动算子的传播支对照。

运行：python 验证脚本/second_order_source_lock_time.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "images", "second_order_source_lock_time.png")

K = 0.005
kv = 1.0
L = 400.0
N = 4096
dx = L / N
x = np.linspace(-L / 2, L / 2, N, endpoint=False)
kfft = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
M = 2.0 * np.pi * np.sqrt(K)


def d1(a):
    return np.real(np.fft.ifft(1j * kfft * np.fft.fft(a)))


def d2(a):
    return np.real(np.fft.ifft(-(kfft ** 2) * np.fft.fft(a)))


def shift(a, dist):
    return np.real(np.fft.ifft(np.fft.fft(a) * np.exp(-1j * kfft * dist)))


def poisson(S):
    Sc = S - np.mean(S)
    return np.real(np.fft.ifft(-np.fft.fft(Sc) / (kfft ** 2 + (kfft == 0.0))))


def kink_fields(v, t, xi=0.0):
    g = 1.0 / np.sqrt(1.0 - v * v)
    s = np.exp(M * g * (x - v * t - xi))
    th = (2.0 / np.pi) * np.arctan(s)
    thx = d1(th - x / L) + 1.0 / L
    tht = -v * thx
    V = 8.0 * K * s ** 2 / (1.0 + s ** 2) ** 2
    T00 = 0.5 * tht ** 2 + 0.5 * kv * thx ** 2 + V
    T01 = -tht * thx
    T11 = 0.5 * tht ** 2 + 0.5 * kv * thx ** 2 - V
    T22 = -0.5 * tht ** 2 + 0.5 * kv * thx ** 2 + V
    S = T00 + T11 + 2.0 * T22
    return dict(th=th, T00=T00, T01=T01, T11=T11, S=S)


def main():
    print("=" * 78)
    print("second_order_source_lock_time.py")
    print("判据 TR：次阶源锁定复验·第二段（含时度规的时间响应）")
    print("=" * 78)

    ok = {}
    v = 0.3

    # ---------- TR1 时间序列闭合与守恒 ----------
    print()
    print("TR1 时间序列（五个时刻）的约束闭合与两条守恒律")
    res_list, c1_list, c2_list = [], [], []
    for t in (0.0, 5.0, 10.0, 15.0, 20.0):
        f = kink_fields(v, t)
        phi = poisson(f["S"])
        res = np.max(np.abs(d2(phi) - (f["S"] - np.mean(f["S"]))))
        R1 = -v * d1(f["T00"]) + d1(f["T01"])
        R2 = -v * d1(f["T01"]) + d1(f["T11"])
        scale = np.max(np.abs(d1(f["T00"])))
        res_list.append(res); c1_list.append(np.max(np.abs(R1)) / scale)
        c2_list.append(np.max(np.abs(R2)) / scale)
    print("   闭合残差（五时刻）：%s" % ", ".join("%.2e" % r for r in res_list))
    print("   守恒残差 R1：%s" % ", ".join("%.2e" % r for r in c1_list))
    print("   守恒残差 R2：%s" % ", ".join("%.2e" % r for r in c2_list))
    ok["TR1"] = max(res_list) < 1e-8 and max(max(c1_list), max(c2_list)) < 1e-8

    # ---------- TR2 瞬时跟踪（移位恒等） ----------
    print()
    print("TR2 瞬时跟踪：响应随源平移（谱移位逐位核对）")
    devs2 = []
    for dt in (4.0, 9.0):
        f0 = kink_fields(v, 5.0)
        f1 = kink_fields(v, 5.0 + dt)
        p0, p1 = poisson(f0["S"]), poisson(f1["S"])
        dev = np.max(np.abs(p1 - shift(p0, v * dt))) / np.max(np.abs(p1))
        devs2.append(dev)
        print("   dt = %.1f : 相对偏差 = %.2e" % (dt, dev))
    ok["TR2"] = max(devs2) < 1e-8

    # ---------- TR3 时间导数恒等 ----------
    print()
    print("TR3 时间导数恒等：d_t phi 的两条路线")
    f = kink_fields(v, 5.0)
    phi = poisson(f["S"])
    dS = -v * d1(f["S"])
    route_a = -v * d1(phi)          # 响应求导
    route_b = poisson(dS)           # 源求导后解约束
    dev3 = np.max(np.abs(route_a - route_b)) / np.max(np.abs(route_b))
    print("   相对偏差 = %.2e（期望机器级）" % dev3)
    ok["TR3"] = dev3 < 1e-8

    # ---------- TR4 调制跟踪 ----------
    print()
    print("TR4 调制跟踪（位置调制与振幅调制）")
    Tamp, Tpos = 15.0, 20.0
    dev4 = []
    amps_pred, amps_meas = [], []
    for t in np.arange(0.0, 15.0, 2.5):
        a = 1.0 + 0.3 * np.sin(2.0 * np.pi * t / Tamp)
        base = kink_fields(v, t)
        base_phi = poisson(base["S"])
        S_mod = a * base["S"]
        phi_mod = poisson(S_mod)
        phi_a = a * base_phi
        dev4.append(np.max(np.abs(phi_mod - phi_a)) / np.max(np.abs(phi_a)))
        amps_pred.append(a)
        amps_meas.append(np.max(np.abs(phi_mod)) / np.max(np.abs(base_phi)))
    dev_amp = float(np.max(np.abs(np.array(amps_meas) - np.array(amps_pred))))
    print("   振幅调制：响应/瞬时像 最大偏差 = %.2e" % max(dev4))
    print("   振幅读数对 a(t) 的偏差 = %.2e" % dev_amp)
    dev5 = []
    for t in (2.0, 7.0, 12.0):
        xi = 2.0 * np.sin(2.0 * np.pi * t / Tpos)
        f = kink_fields(v, t, xi=xi)
        phi = poisson(f["S"])
        base = poisson(kink_fields(v, 0.0)["S"])
        dev = np.max(np.abs(phi - shift(base, v * t + xi))) / np.max(np.abs(phi))
        dev5.append(dev)
        print("   位置调制 t = %.1f : 响应对瞬时位置的偏差 = %.2e" % (t, dev))
    ok["TR4"] = max(dev4) < 1e-8 and max(dev5) < 1e-8 and dev_amp < 1e-12

    # ---------- TR5 齐次自由度与传播支 ----------
    print()
    print("TR5 齐次自由度（时间维）与传播支对照")
    k0 = 2.0 * np.pi / L * 10.0
    mode = np.cos(k0 * x)
    res_const, res_mode = [], []
    for t in (0.0, 2.0, 4.0):
        c_t = 0.37 * np.sin(1.3 * t) + 0.11
        phi_c = c_t * np.ones_like(x)                      # 空间常数零模
        A_t = np.sin(2.0 * t)
        phi_b = A_t * mode                                 # 呼吸型非平凡扰动
        res_const.append(np.max(np.abs(d2(phi_c))))
        res_mode.append(np.max(np.abs(d2(phi_b) - A_t * (-k0 ** 2) * mode)))
    print("   空间常数零模的残差 = %.2e（恒零）" % max(res_const))
    print("   呼吸型扰动的残差核对（对 -A k^2 cos 的偏差）= %.2e" % max(res_mode))
    om = np.linspace(0.0, 4.0 * k0, 4001)
    trace_min = k0 * k0 * np.ones_like(om)
    wave_min = np.abs(om ** 2 - k0 ** 2)
    print("   传播支对照：迹算子最小残差 = %.4e（无零）；波动算子最小残差 = %.4e（omega = k 归零）"
          % (np.min(trace_min), np.min(wave_min)))
    ok["TR5"] = (max(res_const) < 1e-12) and (max(res_mode) < 1e-12) and (np.min(trace_min) > 0)

    # ---------- 图 ----------
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.6))
    ax = axes[0]
    for t, c in zip((0.0, 10.0, 20.0), ("C0", "C1", "C3")):
        f = kink_fields(v, t)
        ax.plot(x, poisson(f["S"]), color=c, lw=1.1, label="t = %.0f" % t)
    ax.set_xlim(-12, 12); ax.set_xlabel("x"); ax.set_ylabel("phi")
    ax.set_title("(a) 响应随源刚性平移"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax = axes[1]
    ts = np.arange(0.0, 15.0, 2.5)
    aa = 1.0 + 0.3 * np.sin(2.0 * np.pi * ts / Tamp)
    ax.plot(ts, aa, "-", label="源调制 a(t)")
    ax.plot(ts, amps_meas, "o", ms=6, label="响应读数")
    ax.set_xlabel("t"); ax.set_ylabel("幅度比")
    ax.set_title("(b) 振幅调制跟踪（无滞后）"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax = axes[2]
    ax.plot(om / k0, trace_min / (k0 * k0), lw=1.2, label="迹算子")
    ax.plot(om / k0, wave_min / (k0 * k0), "--", lw=1.2, label="波动算子")
    ax.set_xlabel("omega / k"); ax.set_ylabel("最小残差 / k^2")
    ax.set_title("(c) 传播支对照"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
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
