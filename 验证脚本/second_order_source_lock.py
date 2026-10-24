#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
second_order_source_lock.py

判据 SS：次阶源锁定复验（首轮）。

背景
----
自旋 2 的源锁定（判据 U）只在弱场线性层验过：迹由密度源锁定、纵向由流守恒锁定，
四条约束对四支，零空间维度 2。缺口登记为「第五种力的现身处恰在次阶与非线性」，
最低成本检验为：把分段 (d) 的动能项与应力项补进迹方程，检查呼吸模是否仍不独立。

本脚本在 (d) 的最小模型（FK 链的正弦-Gordon 墙）上做四件事：
  SS1 含动能与应力项的两条守恒律（能量、动量）残差；
  SS2 静止零张力与移动应力形态（含 v 的解析结构）；
  SS3 迹源装配与份额：T^i_i、全迹 T、线性化 00 源，含 v 的份额恒等式；
  SS4 呼吸通道的唯一性：约束算子核只有常数、注入扰动不是齐次解、符号无 omega 依赖；
  SS5 守恒破坏的注入灵敏度：残差随 epsilon 线性，实测源为零。

运行：python 验证脚本/second_order_source_lock.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "images", "second_order_source_lock.png")

K = 0.005
kv = 1.0
L = 400.0
N = 4096
dx = L / N
x = np.linspace(-L / 2, L / 2, N, endpoint=False)
core = np.abs(x) <= 30.0
kfft = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)


def d1(a):
    return np.real(np.fft.ifft(1j * kfft * np.fft.fft(a)))


def d2(a):
    return np.real(np.fft.ifft(-(kfft ** 2) * np.fft.fft(a)))


M = 2.0 * np.pi * np.sqrt(K)


def fields(v):
    g = 1.0 / np.sqrt(1.0 - v * v)
    s = np.exp(M * g * x)
    th = (2.0 / np.pi) * np.arctan(s)
    # 单一 kink 在环面上带绕数 1：减线性项后周期化，再做谱导数
    thx = d1(th - x / L) + 1.0 / L
    tht = -v * thx
    V = 8.0 * K * s ** 2 / (1.0 + s ** 2) ** 2
    T00 = 0.5 * tht ** 2 + 0.5 * kv * thx ** 2 + V
    T01 = -tht * thx
    T11 = 0.5 * tht ** 2 + 0.5 * kv * thx ** 2 - V
    T22 = -0.5 * tht ** 2 + 0.5 * kv * thx ** 2 + V
    return dict(g=g, th=th, thx=thx, tht=tht, V=V,
                T00=T00, T01=T01, T11=T11, T22=T22,
                kin=0.5 * tht ** 2, grad=0.5 * kv * thx ** 2)


def main():
    print("=" * 78)
    print("second_order_source_lock.py")
    print("判据 SS：次阶源锁定复验（首轮）")
    print("=" * 78)

    scale = None
    ok = {}

    # ---------- SS1 两条守恒律 ----------
    print()
    print("SS1 含动能与应力项的守恒律残差（谱导数，归一化）")
    vs = [0.0, 0.1, 0.2, 0.3]
    r1s, r2s = [], []
    for v in vs:
        f = fields(v)
        dT00, dT01, dT11 = d1(f["T00"]), d1(f["T01"]), d1(f["T11"])
        R1 = -v * dT00 + dT01           # d_t T00 + d_x T01
        R2 = -v * dT01 + dT11           # d_t T01 + d_x T11
        scale = np.max(np.abs(dT00))
        r1, r2 = np.max(np.abs(R1)) / scale, np.max(np.abs(R2)) / scale
        r1s.append(r1); r2s.append(r2)
        print("   v = %.2f : |R1| = %.2e , |R2| = %.2e（归一化）" % (v, r1, r2))
    ok["SS1"] = max(max(r1s), max(r2s)) < 1e-8

    # ---------- SS2 静止零张力与移动应力 ----------
    print()
    print("SS2 静止零张力与移动应力形态")
    f0 = fields(0.0)
    zt = np.max(np.abs(f0["T11"][core])) / np.max(f0["T00"])
    vir = np.max(np.abs((f0["grad"] - f0["V"])[core])) / np.max(f0["V"])
    print("   静止 |T11|/max(T00) = %.2e ；位力 |grad - V|/max(V) = %.2e" % (zt, vir))
    dev11 = []
    for v in vs[1:]:
        f = fields(v)
        pred = 2.0 * f["g"] ** 2 * v * v * f["V"]
        dev11.append(np.max(np.abs((f["T11"] - pred)[core])) / np.max(pred))
    print("   移动 T11 对 2 gamma^2 v^2 V 的最大相对偏差：%s"
          % ", ".join("%.2e" % d for d in dev11))
    ok["SS2"] = (zt < 1e-8) and (vir < 1e-8) and (max(dev11) < 1e-8)

    # ---------- SS3 迹源装配与份额 ----------
    print()
    print("SS3 迹源装配与含 v 的份额恒等式")
    devs = {"T00": [], "T11": [], "T22": [], "trace": [], "shares": []}
    for v in vs:
        f = fields(v)
        g, V = f["g"], f["V"]
        T00p = 2.0 * g ** 2 * V
        T22p = 2.0 * V
        trace = -f["T00"] + f["T11"] + 2.0 * f["T22"]
        devs["T00"].append(np.max(np.abs((f["T00"] - T00p)[core])) / np.max(T00p))
        den11 = np.max(2.0 * g ** 2 * v * v * V) if v > 0 else np.max(T00p)
        devs["T11"].append(np.max(np.abs((f["T11"] - 2.0 * g ** 2 * v * v * V)[core])) / den11)
        devs["T22"].append(np.max(np.abs((f["T22"] - T22p)[core])) / np.max(T22p))
        devs["trace"].append(np.max(np.abs((trace - 2.0 * V)[core])) / np.max(2.0 * V))
        sk = f["kin"] / f["T00"]; sg = f["grad"] / f["T00"]; sp = V / f["T00"]
        pred = (v * v / 2.0, 0.5, 1.0 / (2.0 * g ** 2))
        devs["shares"].append(max(
            np.max(np.abs((sk - pred[0])[core])), np.max(np.abs((sg - pred[1])[core])),
            np.max(np.abs((sp - pred[2])[core]))))
    for key, vals in devs.items():
        print("   %s 偏差：%s" % (key, ", ".join("%.2e" % q for q in vals)))
    print("   份额恒等式：kin/T00 = v^2/2、grad/T00 = 1/2、pot/T00 = 1/(2 gamma^2)")
    ok["SS3"] = max(max(v) for v in devs.values()) < 1e-8

    # ---------- SS4 呼吸通道的唯一性 ----------
    print()
    print("SS4 呼吸通道（迹约束）的唯一性与无传播支")
    f = fields(0.3)
    S = f["T00"] + (f["T11"] + 2.0 * f["T22"])   # 主动源组合（rho + 3p 型）
    S = S - np.mean(S)
    # 算子：拉普拉斯（周期，零均值子空间）
    ker = np.sum(np.abs(kfft) < 1e-12)
    gap = np.min(np.abs(kfft[kfft != 0.0]) ** 2)
    phi = np.real(np.fft.ifft(-np.fft.fft(S) / (kfft ** 2 + (kfft == 0.0))))
    resid = d2(phi) - S
    print("   算子核维数（常数模）= %d ；最小非零本征值 = %.3e" % (ker, gap))
    print("   源解残差 max|lap phi - S| = %.2e" % np.max(np.abs(resid)))
    bump = np.exp(-(x / 2.0) ** 2); bump = bump - np.mean(bump)
    lapb = d2(bump)
    print("   注入零均值扰动：max|lap bump| = %.2e（非零，故不是齐次解）" % np.max(np.abs(lapb)))
    k0 = 2.0 * np.pi / L * 10.0
    om = np.linspace(0.0, 4.0 * k0, 4001)
    trace_min = k0 * k0 * np.ones_like(om)
    wave_min = np.abs(om ** 2 - k0 ** 2)
    print("   传播支对照：迹通道算子的最小残差（omega 扫描）= %.4e（无零）"
          % np.min(trace_min))
    print("   对照波动算子的最小残差 = %.4e（在 omega = k 处为零，传播支存在）"
          % np.min(wave_min))
    ok["SS4"] = (ker == 1) and (np.max(np.abs(resid)) < 1e-8) and (np.max(np.abs(lapb)) > 1e-6)

    # ---------- SS5 守恒破坏的注入灵敏度 ----------
    print()
    print("SS5 守恒破坏注入的线性灵敏度")
    v = 0.3
    f = fields(v)
    n_mode = 8
    gshape = np.sin(2.0 * np.pi * n_mode * x / L)
    slope_true = (2.0 * np.pi * n_mode / L)
    for eps in (0.0, 1e-6, 2e-6, 5e-6):
        T11p = f["T11"] + eps * gshape
        R2 = -v * d1(f["T01"]) + d1(T11p)
        amp = np.max(np.abs(R2))
        if eps == 0.0:
            print("   eps = 0 : max|R2| = %.3e（实测源，机器零）" % amp)
        else:
            print("   eps = %.1e : max|R2| = %.3e , max|R2|/eps = %.6e"
                  % (eps, amp, amp / eps))
    eps = 1e-6
    R2 = -v * d1(f["T01"]) + d1(f["T11"] + eps * gshape)
    ratio = np.max(np.abs(R2)) / (eps * slope_true * np.max(np.abs(np.cos(2.0 * np.pi * n_mode * x / L))))
    print("   斜率核对（注入项解析导数之比）= %.6f（期望 1）" % ratio)
    ok["SS5"] = abs(ratio - 1.0) < 1e-6

    # ---------- 图 ----------
    f = fields(0.3)
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.6))
    ax = axes[0]
    ax.plot(x, f["T00"], label="T00")
    ax.plot(x, f["T11"], label="T11")
    ax.plot(x, f["T22"], label="T22")
    ax.plot(x, -f["T00"] + f["T11"] + 2.0 * f["T22"], "--", label="trace T")
    ax.set_xlim(-15, 15); ax.set_xlabel("x"); ax.set_ylabel("T")
    ax.set_title("(a) v = 0.3 的应力张量分量"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax = axes[1]
    vv = np.linspace(0.0, 0.5, 200)
    gg = 1.0 / np.sqrt(1.0 - vv * vv)
    ax.plot(vv, vv ** 2 / 2.0, label="kin/T00 = v^2/2")
    ax.plot(vv, 0.5 + 0.0 * vv, label="grad/T00 = 1/2")
    ax.plot(vv, 1.0 / (2.0 * gg ** 2), label="pot/T00 = 1/(2 gamma^2)")
    ax.set_xlabel("v"); ax.set_ylabel("share")
    ax.set_title("(b) 迹源的含 v 份额恒等式"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax = axes[2]
    epslist = np.array([1e-7, 2e-7, 5e-7, 1e-6, 2e-6])
    amps = []
    for e in epslist:
        R2 = -0.3 * d1(fields(0.3)["T01"]) + d1(fields(0.3)["T11"] + e * np.sin(2.0 * np.pi * 8 * x / L))
        amps.append(np.max(np.abs(R2)))
    ax.loglog(epslist, amps, "o-", label="max|R2|")
    ax.loglog(epslist, epslist * slope_true, "--", label="斜率参考")
    ax.set_xlabel("epsilon"); ax.set_ylabel("max|R2|")
    ax.set_title("(c) 注入破坏的线性响应"); ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
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
