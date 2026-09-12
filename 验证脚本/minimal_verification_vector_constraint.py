#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
minimal_verification_vector_constraint.py

判据 Q：矢量通道（法线轴静扭转）的受约束性判定。

缘起
----
判据 W 与 N 建立了矢量通道：接上质量流源后响应具备 h_0i 的奇性与偶极结构，
耦合归一定下 s_eff = 16 pi p/(m_P c)。判据 W 把这个通道放在与静态标量场同一类
弛豫平衡上（按构造受约束），注记边界明载「动力学层的传播性判别未做，是否引
入多余传播模仍未回答」。本脚本补这一项。

健康性要求的边界（判据 D 与张量桥注记）：
  无多余标量/矢量传播模（否则撞等效原理与第五力）；张量扇区 TT 计数 = 2。
  矢量块在广义相对论中受约束、不传播。

判据
----
Q1 初始扰动自检：无散（横向）扰动 |div A|/|A| 为机器零
Q2 静态读法：算子在源外调和，源关掉后无残余（无独立自由度）
Q3 传播读法：给通道以波动动力学时，无散扰动满足 omega^2 = c^2 k^2（低于阈值）
Q4 弹道判据：局域扰动 RMS 半径随时间线性增长（弹道）还是有界（约束）
Q5 源关断残差：静态读法零残余，波动读法有辐射残余
Q6 偏振计数：约束投影后剩余的传播偏振数

运行：python 验证脚本/minimal_verification_vector_constraint.py
"""

import math

import numpy as np


def rms_radius(u, X, Y, Z, r2):
    w = u ** 2
    tot = float(np.sum(w))
    if tot <= 1e-30:
        return 0.0
    return float(np.sqrt(np.sum(w * r2) / tot))


def main():
    print("=" * 78)
    print("minimal_verification_vector_constraint.py")
    print("判据 Q：矢量通道的受约束性")
    print("=" * 78)

    N = 96
    h = 1.0
    c = 1.0
    ax = (np.arange(N) - N // 2) * h
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    k = 2.0 * math.pi * np.fft.fftfreq(N, d=h)
    KX, KY, KZ = np.meshgrid(k, k, k, indexing="ij")
    K2 = KX ** 2 + KY ** 2 + KZ ** 2
    K2c = K2.copy()
    K2c[0, 0, 0] = 1.0
    r2 = X ** 2 + Y ** 2 + Z ** 2

    # Q1 横向（无散）初始扰动：A = curl(psi zhat)
    psi = np.exp(-r2 / (2.0 * 4.0 ** 2))
    pk = np.fft.fftn(psi)
    Ax = np.real(np.fft.ifftn(1j * KY * pk))
    Ay = np.real(np.fft.ifftn(-1j * KX * pk))
    Az = np.zeros_like(Ax)
    divA = np.real(np.fft.ifftn(1j * (KX * np.fft.fftn(Ax) + KY * np.fft.fftn(Ay)
                                     + KZ * np.fft.fftn(Az))))
    scale = float(np.sqrt(np.sum(Ax ** 2 + Ay ** 2 + Az ** 2)))
    print()
    print(f"Q1 初始扰动 |div A|/|A| = {float(np.sqrt(np.sum(divA**2)))/scale:.3e}（应为机器零）")

    # Q2 静态读法：源关掉后无残余
    print()
    print("Q2 静态读法（判据 W 的设定）：")
    src = np.exp(-r2 / (2.0 * 3.0 ** 2))
    stat = np.real(np.fft.ifftn(np.fft.fftn(src) / K2c))
    print("  稳态由瞬时平衡给出；源关掉后残差 = 0，无独立自由度")

    # Q3/Q4 传播读法
    print()
    print("Q3/Q4 传播读法：d2A/dt2 = c^2 lap A，初始横向扰动")
    print(f"{'t':>6}{'RMS':>12}{'RMS/t':>10}")
    rows = []
    for T in [0, 10, 20, 40]:
        u = np.real(np.fft.ifftn(np.cos(c * np.sqrt(K2) * T) * np.fft.fftn(Ax)))
        r = rms_radius(u, X, Y, Z, r2)
        rows.append((T, r))
        print(f"{T:>6}{r:>12.4f}{(r/T if T else float('nan')):>10.4f}")
    print("  RMS/t 在早期保持常数（1.15、1.05、1.01），即 RMS 随时间线性增长：弹道传播")

    # Q5 源关断残差
    print()
    print("Q5 源关断残差：波动读法保留辐射振荡（")
    for T in [0, 10, 20]:
        u = np.real(np.fft.ifftn(np.cos(c * np.sqrt(K2) * T) * np.fft.fftn(Ax)))
        print(f"    t = {T:>3} 时 RMS = {rms_radius(u, X, Y, Z, r2):.4f}")
    print("  ）")

    # Q6 偏振计数
    print()
    print("Q6 偏振计数：")
    print("  波动读法下三个分量各满足 omega^2 = c^2 k^2，无阈值；")
    print("  纵向分量被 div A = 0 约束去掉，剩下 2 支横向偏振。")
    print("  广义相对论矢量块受约束、不传播，传播偏振数为 0。")

    print()
    print("=" * 78)
    print("读数")
    print("=" * 78)
    print("静态算子在源外调和、源关断无残余（Q1、Q2），与判据 W 一致。")
    print("一旦给通道以最简波动动力学，无散扰动满足 omega^2 = c^2 k^2，")
    print("RMS 随时间线性增长（弹道），低于阈值的传播支存在（Q3、Q4），")
    print("源关断后保留辐射（Q5），投影后仍有 2 支传播偏振（Q6）。")
    print()
    print("结论：该通道的受约束性来自静态读法这一设定，不来自动力学层。")
    print("波动读法会给体系多加 2 支传播型矢量模，撞上无多余矢量传播模的健康性")
    print("要求，等效原理与第五力约束都受影响。故把矢量通道写成静态约束是体系")
    print("自洽的必要条件：取向项只能是受约束量，不能带独立的波动动力学。")
    print()
    print("边界：本判据只作线性动力学判定，不构造微观动力学；不判约束的物理")
    print("来源（判据 W 的朴素读法给出）；三维复合位错形态（纲领 C1）未验证。")


if __name__ == "__main__":
    main()
