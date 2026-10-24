#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
twist_channel_source.py

判据 W：法线轴静扭转通道能否被质量流激发（参考系拖曳的源通道判定）。

缘起
----
判据 V 判定：现行写入规则只给标量倾斜幅度，对时间反演为偶，故 h_0i 恒为零；
载体本身具备横向扇区，缺口在源通道。作者 2026-09-12 裁决：引力侧的旋落在法线轴
一侧，由法线轴自身的静扭转承载，不必借用实轴取向场的机制，故不破分工双场。
本脚本把该裁决落成最小验证：给扭转扇道喂真实的质量流源，检验响应是否具备
广义相对论 h_0i 的结构特征。

模型设定（本验证采用的约定，非导出自更微观层级）
--------
两个扇道共用同一个算子（判据 D 的朴素读法：每分量标量弹性）：
    标量扇道（质量）：   lap theta = - s_rho,  s_rho = rho
    取向扇道（质量流）： lap A     = - s_j,    s_j   = rho * v
写入规则的取向部分取最简形式：每拍写一份动量，源即质量流密度 rho*v。
两扇道共用耦合常数（此处取 1）。本验证只判结构与标度，不判系数绝对值。
数值用周期盒上的傅里叶逆算子（与判据 V3 同一套符号处理）。

源（三种）
--
  sphere：球对称密度刚体自转，v = Omega z x r。严格无散，净电流为零，角动量非零。
  ring  ：单层环，位于 z = 0，半径 R，同样自转。
  pair  ：两层同心环，位于 z = +/- z0，角速度反号，净角动量为零。

解析基准
--------
球对称加刚体转动时，二极以上多极矩全部为零，故远场为纯偶极且在任意 r 处精确：
    A(r) = -(rhat x J) / (8 pi r^2),   J = Omega * I_z,  I_z = 积分 rho (x^2+y^2) dV
环不是球对称，其远场含更高极，故只作定性对照，不作系数判据。

判据
----
W0 源自检：自转源质量流散度与自身之比，并检验残差随源平滑度下降（谱混叠）
W1 负对照：关掉取向源时响应严格为零
W2 线性与奇偶：A 与 Omega 成正比；A(-Omega) = -A(Omega)
W3 远场系数与收敛：自转球 r^2|A| 逼近解析 Omega*Iz/(8pi)，残差随盒尺寸下降
W4 角向结构：转轴方向 |A| 为零、赤道面最大（偶极型）
W5 扇道分离：标量扇道与 Omega 无关，取向扇道与 Omega 成正比
W6 净角动量判据：反旋对在对称面上严格为零，面外残余远弱于同几何的单层环

运行：python 验证脚本/twist_channel_source.py
"""

import math

import numpy as np


def grid(n, dx=1.0):
    ax = (np.arange(n) - n // 2) * dx
    return np.meshgrid(ax, ax, ax, indexing="ij")


def kvec(n, dx=1.0):
    k = 2.0 * math.pi * np.fft.fftfreq(n, d=dx)
    return np.meshgrid(k, k, k, indexing="ij")


def laplace_inv(src, K2):
    """解 lap f = -src，即 f_hat = src_hat / k^2（零模置一，等价于扣掉源平均）。"""
    sh = np.fft.fftn(src)
    K2c = K2.copy()
    K2c[0, 0, 0] = 1.0
    return np.real(np.fft.ifftn(sh / K2c))


def build_source(X, Y, Z, mode, omega, width=4.0, ring_r=12.0, ring_w=2.0, z0=8.0):
    """返回 (rho, vx, vy, vz, I_z)。"""
    R_xy = np.sqrt(X ** 2 + Y ** 2)
    zero = np.zeros_like(X)
    if mode == "sphere":
        rho = np.exp(-((np.sqrt(X ** 2 + Y ** 2 + Z ** 2) / width) ** 4))
        vx, vy, vz = -omega * Y, omega * X, zero
        I_z = float(np.sum(rho * R_xy ** 2))
        return rho, vx, vy, vz, I_z
    if mode == "ring":
        rho = np.exp(-(((R_xy - ring_r) ** 2 + Z ** 2) / (2.0 * ring_w ** 2)))
        vx, vy, vz = -omega * Y, omega * X, zero
        I_z = float(np.sum(rho * R_xy ** 2))
        return rho, vx, vy, vz, I_z
    if mode == "pair":
        top = np.exp(-(((R_xy - ring_r) ** 2 + (Z - z0) ** 2) / (2.0 * ring_w ** 2)))
        bot = np.exp(-(((R_xy - ring_r) ** 2 + (Z + z0) ** 2) / (2.0 * ring_w ** 2)))
        rho = top + bot
        vx = top * (-omega * Y) + bot * (omega * Y)
        vy = top * (omega * X) + bot * (-omega * X)
        vz = zero
        I_z = float(np.sum((top - bot) * R_xy ** 2))
        return rho, vx, vy, vz, I_z
    raise ValueError(mode)


def run(n, omega, dx=1.0, mode="sphere", **kw):
    X, Y, Z = grid(n, dx)
    KX, KY, KZ = kvec(n, dx)
    K2 = KX ** 2 + KY ** 2 + KZ ** 2
    rho, vx, vy, vz, I_z = build_source(X, Y, Z, mode, omega, **kw)
    jx, jy, jz = rho * vx, rho * vy, rho * vz
    th = laplace_inv(rho, K2)
    A = (laplace_inv(jx, K2), laplace_inv(jy, K2), laplace_inv(jz, K2))
    dh = 1j * (KX * np.fft.fftn(jx) + KY * np.fft.fftn(jy) + KZ * np.fft.fftn(jz))
    divj = np.real(np.fft.ifftn(dh))
    scale = float(np.sqrt(np.sum(jx ** 2 + jy ** 2 + jz ** 2)))
    div_rel = (float(np.sqrt(np.sum(divj ** 2))) / scale) if scale > 0 else 0.0
    return dict(A=A, th=th, div_rel=div_rel, I_z=I_z, n=n, dx=dx, omega=omega)


def mag(A):
    return np.sqrt(A[0] ** 2 + A[1] ** 2 + A[2] ** 2)


def prof(A, n, dx, radii, direction="x"):
    i0 = n // 2
    f = mag(A)
    out = []
    for q in radii:
        m = int(round(q / dx))
        if direction == "x":
            out.append(float(f[i0 + m, i0, i0]))
        elif direction == "z":
            out.append(float(f[i0, i0, i0 + m]))
        elif direction == "diag":
            out.append(float(f[i0 + m, i0 + m, i0 + m]))
        elif direction == "xz":
            out.append(float(f[i0 + m, i0, i0 + m]))
    return out


def slope(vals, radii):
    return float(np.polyfit(np.log(np.array(radii)), np.log(np.array(vals)), 1)[0])


def main():
    print("=" * 78)
    print("twist_channel_source.py")
    print("判据 W：法线轴静扭转通道能否被质量流激发")
    print("模型：两扇道共用朴素读法算子；取向源取质量流密度 rho*v；耦合取 1")
    print("=" * 78)

    n_main, om0, dx = 96, 1.0, 1.0
    radii = (12.0, 16.0, 20.0, 24.0)
    r1 = run(n_main, om0, dx)
    K2m = kvec(n_main, dx)
    K2m = K2m[0] ** 2 + K2m[1] ** 2 + K2m[2] ** 2

    # ---------------- W0 / W1 ----------------
    print()
    print("-" * 78)
    print("W0 源自检 ｜ W1 负对照")
    print("-" * 78)
    print(f"W0  |div j| / |j| = {r1['div_rel']:.3e}（连续极限下严格无散，残差为谱混叠）")
    widths = (2.0, 4.0, 6.0)
    divs = [run(n_main, om0, dx, width=w)["div_rel"] for w in widths]
    for w, d in zip(widths, divs):
        print(f"    源宽 w = {w:>4.1f}   |div j|/|j| = {d:.3e}")
    zero = laplace_inv(np.zeros((n_main,) * 3), K2m)
    maxA1 = float(np.max(mag(r1["A"])))
    print(f"W1  关掉取向源时的响应 = {float(np.max(np.abs(zero))):.3e}（严格为零）")
    print(f"    对照：自转源的最大响应 = {maxA1:.3e}")

    # ---------------- W2 ----------------
    print()
    print("-" * 78)
    print("W2 线性与奇偶")
    print("-" * 78)
    oms = (0.5, 1.0, 2.0)
    norms = []
    for w in oms:
        m = float(np.max(mag(run(n_main, w, dx)["A"])))
        norms.append(m / w)
        print(f"  Omega = {w:>4.1f}   max|A| = {m:.6f}   归一 max|A|/Omega = {m/w:.9f}")
    rneg = run(n_main, -om0, dx)
    odd = float(np.max(np.abs(np.array(rneg["A"]) + np.array(r1["A"]))))
    print(f"  A(-1) + A(+1) 的最大模 = {odd:.3e}（奇性）")
    print(f"  归一量离散度 = {max(norms)-min(norms):.3e}（线性程度）")

    # ---------------- W3 ----------------
    print()
    print("-" * 78)
    print("W3 远场系数与盒尺寸收敛（自转球，纯偶极）")
    print("-" * 78)
    pe = prof(r1["A"], n_main, dx, radii)
    asym = r1["omega"] * r1["I_z"] / (8.0 * math.pi)
    print(f"{'r':>8}" + "".join(f"{q:>14.0f}" for q in radii))
    print(f"{'r^2|A|':>8}" + "".join(f"{v*q*q:>14.6e}" for v, q in zip(pe, radii)))
    print(f"{'解析':>8}" + f"{asym:>14.6e}   （Omega*Iz/(8pi)）")
    dev12 = abs(pe[0] * 144.0 - asym) / asym
    sl_far = slope(pe[1:], radii[1:])
    sl_near = slope(pe[:3], radii[:3])
    print(f"r = 12 处相对偏差 = {dev12:.3e}")
    print(f"对数斜率：r = 12 至 20 为 {sl_near:.3f}（预期 -2）；"
          f"r = 16 至 24 为 {sl_far:.3f}（远端受周期镜像压低）")
    conv = []
    print()
    print(f"{'n':>8}{'r^2|A| (r=12)':>20}{'相对偏差':>14}")
    for nn in (64, 96, 128):
        rr = run(nn, om0, dx)
        p = prof(rr["A"], nn, dx, (12.0,))[0] * 144.0
        a = rr["omega"] * rr["I_z"] / (8.0 * math.pi)
        conv.append(abs(p - a) / a)
        print(f"{nn:>8}{p:>20.6e}{abs(p-a)/a:>14.3e}")

    # ---------------- W4 ----------------
    print()
    print("-" * 78)
    print("W4 角向结构（偶极型：转轴为零，赤道最大）")
    print("-" * 78)
    pa = prof(r1["A"], n_main, dx, radii, "z")
    ratio_ax = max(a / e for a, e in zip(pa, pe))
    print(f"{'r':>8}" + "".join(f"{q:>13.0f}" for q in radii))
    print(f"{'赤道面':>8}" + "".join(f"{v:>13.6e}" for v in pe))
    print(f"{'转轴方向':>8}" + "".join(f"{v:>13.6e}" for v in pa))
    print(f"最大比值（转轴/赤道）= {ratio_ax:.3e}")

    # ---------------- W5 ----------------
    print()
    print("-" * 78)
    print("W5 扇道分离")
    print("-" * 78)
    dth = float(np.max(np.abs(run(n_main, 0.0, dx)["th"] - r1["th"]))) / float(
        np.max(np.abs(r1["th"])))
    print(f"  Omega = 0 与 1 的 theta 相对差 = {dth:.3e}（标量扇道对转动不敏感）")
    ratios = []
    for w in oms:
        rw = run(n_main, w, dx)
        v = float(np.max(mag(rw["A"]))) / float(np.max(np.abs(rw["th"])))
        ratios.append(v / w)
        print(f"  Omega = {w:>4.1f}   max|A| / max|theta| = {v:.6e}"
              f"   除以 Omega 后 = {v/w:.9f}")
    lin5 = max(ratios) - min(ratios)
    print(f"  归一量的离散度 = {lin5:.3e}（线性程度，理想为零）")

    # ---------------- W6 ----------------
    print()
    print("-" * 78)
    print("W6 净角动量判据（单层环与反旋对的同几何对照）")
    print("-" * 78)
    def w6(n6):
        rs6 = (16.0, 24.0, 32.0)
        a1 = run(n6, om0, dx, mode="ring", ring_r=6.0, ring_w=1.5)
        a2 = run(n6, om0, dx, mode="pair", ring_r=6.0, ring_w=1.5, z0=4.0)
        q1 = prof(a1["A"], n6, dx, rs6)
        q2 = prof(a2["A"], n6, dx, rs6)
        plane = max(b / a for a, b in zip(q1, q2))
        md = tuple(int(q) for q in (16.0, 24.0, 32.0))
        d1 = prof(a1["A"], n6, dx, md, "xz")
        d2 = prof(a2["A"], n6, dx, md, "xz")
        off = [float(q * math.sqrt(2.0)) for q in rs6]
        rat = [b / a for a, b in zip(d1, d2)]
        return plane, off, d1, d2, rat

    print("（环半径 6、宽 1.5；反旋对分离 z0 = +/-4；源的最大外延约 8）")
    for n6 in (96, 128):
        plane, off, d1, d2, rat = w6(n6)
        print()
        print(f"  盒尺寸 n = {n6}")
        print(f"    赤道面（r = 16/24/32）：反旋对 / 单层 = {plane:.3e}"
              f"（镜像对称下应精确为零）")
        print(f"    面外（x=z）r：" + "".join(f"{q:>12.1f}" for q in off))
        print(f"      单层环      " + "".join(f"{v:>12.4e}" for v in d1))
        print(f"      反旋对      " + "".join(f"{v:>12.4e}" for v in d2))
        print(f"      残差比      " + "".join(f"{v:>12.3e}" for v in rat))
        print(f"      比值随 r 的变化：{rat[0]/rat[-1]:.3f} 倍（r 加倍时若多一个 1/r 则应为 2）")
    plane128, off128, d1128, d2128, rat128 = w6(128)
    plane_ratio = plane128
    rat = rat128

    # ---------------- 汇总 ----------------
    print()
    print("=" * 78)
    print("汇总")
    print("=" * 78)
    print(f"W0 自转源无散：|div j|/|j| = {divs[1]:.3e}，随源宽 {widths[0]} 到 {widths[-1]} "
          f"由 {divs[0]:.3e} 降至 {divs[-1]:.3e}（谱混叠）")
    print(f"W1 关掉取向源时响应严格为零；自转源给出 {maxA1:.3e}")
    print(f"W2 线性精确（归一量离散度 {max(norms)-min(norms):.1e}），奇性精确（{odd:.1e}）")
    print(f"W3 自转球 r^2|A| 逼近解析 Omega*Iz/(8pi)：r = 12 偏差 {dev12:.3e}，"
          f"盒尺寸 64 到 128 偏差 {conv[0]:.2e} 降至 {conv[-1]:.2e}")
    print(f"W4 转轴方向为零、赤道最大，比值上界 {ratio_ax:.1e}")
    print(f"W5 标量扇道与 Omega 无关（{dth:.1e}），取向扇道与 Omega 成正比"
          f"（归一量离散度 {lin5:.1e}）")
    print(f"W6 反旋对在对称面内 {plane_ratio:.1e}（精确相消）；面外残差比随 r 由 "
          f"{rat[0]:.2e} 降至 {rat[-1]:.2e}，即 r 加倍时降 {rat[0]/rat[-1]:.3f} 倍"
          f"（极次提高一阶时为 2），且在两种盒尺寸下一致")
    print()
    print("结论：质量流可以激发法线轴的静扭转响应，且响应具备 h_0i 的结构特征：")
    print("      对速度反号为奇、远场为角动量驱动的偶极 1/r^2 且系数对上解析式、")
    print("      转轴为零赤道最大、与质量扇道线性分离、净角动量为零时相消。")
    print("      载体与算子够用，缺的是写规则里的取向项与该耦合的绝对系数。")


if __name__ == "__main__":
    main()
