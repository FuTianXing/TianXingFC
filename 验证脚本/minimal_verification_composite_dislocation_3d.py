#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
minimal_verification_composite_dislocation_3d.py

判据 C1：三维复合位错的最小实例（锚核钉扎能否使环驻留）。

**状态：未定案，作为中间尝试存档（2026-09-12）。**
本脚本的测量装置已修对（M0 自检：直线涡 π₂ = 1.0000），A1 基准复核通过，
但**环的涡旋初态构造有缺陷**：把横截面角直接当作相位取值，得到光滑分布而非
涡旋，故环芯横截面上的 π₂ 读数恒零，A3/A4 的拓扑荷判定不可用。该缺陷已登记
为 C1-0 的任务内容，见《三维复合位错攻坚计划：形态、驻留与宽度窗口》§三。
本脚本可复现的结果只有两条：A1 的负面基准，以及 A3 的能量驻留信号（相位锁定
型锚核下构型不塌缩）。拓扑荷守恒与否未判，不得引用为结论。

缘起
----
裸环裁决实验（`minimal_verification_ring_adjudication.py`）给出定案级负面结果：
裸格点模型中涡旋环不存在亚稳驻留态，一切协议下塌缩到真空、绕数丢失。
复合位错纲领（C1）的立论是：锚核的存在改变问题，使环驻留。本脚本检验该立论
的最小实例，含两种锚核实现的对照。

模型设定
--------
kink3d A 部分模型（与环裁决实验同族），但改用开放边界（吸收边界）：
    H = (1/2) sum_bonds J0 (1 - cos Δφ) + κ sum_sites (1 - cos φ)
φ 为格点相位场，相位差取最小像。开放边界：边界外一圈视为真空 φ = 0。
锚核两种实现：
  (甲) 键耦合弱化：环芯处把 J 降到 J0(1-depth)
  (乙) 相位锁定：钉扎势 κ_p (1 - cos 2(φ - φ_pin)) 把芯位锁到目标相位

拓扑荷测量（本脚本的关键修正）
--------
环芯是圆，其拓扑荷为 π₂（绕环芯的横截面）。测量方式：在与环芯垂直的平面上，
取以芯位为中心的小窗口，逐 plaquette 累加相位差（最小像）除以 2π。
要点：窗口中心必须落在 plaquette 内（半格偏移），否则奇点正好落在格点上会
使 plaquette 和归零（本脚本过程记录第 2 条）。直线涡校验给出 1.0000。

判据
----
A1 基准复核：无锚核时环塌缩（复现环裁决实验）
A2 键耦合弱化型锚核：depth 扫描，是否改变塌缩结局
A3 相位锁定型锚核：κ_p 扫描，环是否驻留、π₂ 是否守恒
A4 半径扫描：相位锁定型锚核在不同 Rc 下的驻留
A5 能量地貌：驻留态的绑定能与钉扎深度的关系

运行：python 验证脚本/minimal_verification_composite_dislocation_3d.py
"""

import numpy as np

J0, KAPPA = 1.0, 0.6
N = 48
coords = np.arange(N) - N / 2.0 + 0.5      # 半格偏移：奇点落在 plaquette 内
X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")
IC = N // 2


def mi(d):
    return (d + np.pi) % (2.0 * np.pi) - np.pi


def ring_psi(Rc):
    """环芯圆：半径 Rc，位于 z = 0 平面。返回横截面角 ψ。"""
    rho = np.sqrt(X ** 2 + Y ** 2)
    return np.arctan2(Z, rho - Rc)


def core_dist(Rc):
    rho = np.sqrt(X ** 2 + Y ** 2)
    return np.sqrt((rho - Rc) ** 2 + Z ** 2)


def energy(phi, kp=0.0, Rc=4.0, pin_w=1.5):
    E = KAPPA * np.sum(1.0 - np.cos(phi))
    for ax in range(3):
        E += 0.5 * J0 * np.sum(1.0 - np.cos(mi(np.diff(phi, axis=ax))))
    if kp > 0.0:
        pin = np.exp(-core_dist(Rc) ** 2 / (2.0 * pin_w ** 2))
        E += kp * np.sum(pin * (1.0 - np.cos(2.0 * (phi - ring_psi(Rc)))))
    return float(E)


def grad_phi(phi, kp=0.0, Rc=4.0, pin_w=1.5):
    p = np.pad(phi, 1)
    ctr = (slice(1, -1),) * 3
    s = (np.sin(p[2:, 1:-1, 1:-1] - p[ctr]) + np.sin(p[:-2, 1:-1, 1:-1] - p[ctr])
         + np.sin(p[1:-1, 2:, 1:-1] - p[ctr]) + np.sin(p[1:-1, :-2, 1:-1] - p[ctr])
         + np.sin(p[1:-1, 1:-1, 2:] - p[ctr]) + np.sin(p[1:-1, 1:-1, :-2] - p[ctr]))
    g = -0.5 * J0 * s + KAPPA * np.sin(phi)
    if kp > 0.0:
        pin = np.exp(-core_dist(Rc) ** 2 / (2.0 * pin_w ** 2))
        g += kp * pin * np.sin(2.0 * (phi - ring_psi(Rc)))
    return g


def pi2_cross(phi, Rc, rad=3, axis="x"):
    """在环芯的横截面上逐 plaquette 累加，得 π₂ 荷。"""
    i = int(round(Rc + N / 2.0 - 0.5))       # 沿 x 在 Rc 处的格面
    i = min(max(i, 1), N - 2)
    jc, kc = IC, IC
    tot = 0.0
    for j in range(jc - rad, jc + rad):
        for k in range(kc - rad, kc + rad):
            p00 = phi[i, j, k]; p10 = phi[i, j + 1, k]
            p11 = phi[i, j + 1, k + 1]; p01 = phi[i, j, k + 1]
            tot += mi(p10 - p00) + mi(p11 - p10) + mi(p01 - p11) + mi(p00 - p01)
    return float(tot / (2.0 * np.pi))


def flow(phi, kp, Rc, steps=1200, eta=0.02):
    phi = phi.copy()
    for _ in range(steps):
        phi = phi - eta * grad_phi(phi, kp, Rc)
    return phi


def main():
    print("=" * 78)
    print("minimal_verification_composite_dislocation_3d.py")
    print("判据 C1：三维复合位错的最小实例（开放边界 + 横截面 π₂ 测量）")
    print("=" * 78)

    # 测量装置自检
    print()
    print("M0 测量装置自检：直线涡 phi=atan2(z,y) 的 π₂ 应为 1")
    x2, y2, z2 = np.meshgrid(coords, coords, coords, indexing="ij")
    lin = np.arctan2(z2, y2)
    tot = 0.0
    for j in range(IC - 3, IC + 3):
        for k in range(IC - 3, IC + 3):
            p00 = lin[IC, j, k]; p10 = lin[IC, j + 1, k]
            p11 = lin[IC, j + 1, k + 1]; p01 = lin[IC, j, k + 1]
            tot += mi(p10 - p00) + mi(p11 - p10) + mi(p01 - p11) + mi(p00 - p01)
    print(f"  π₂ = {tot/(2*np.pi):.4f}（期望 1.0000）")

    print()
    print("A1 基准复核：无锚核（kp = 0）时环塌缩")
    for Rc in [4.0]:
        phi = ring_psi(Rc)
        W0 = pi2_cross(phi, Rc)
        E0 = energy(phi, 0.0, Rc)
        phi2 = flow(phi, 0.0, Rc)
        W1 = pi2_cross(phi2, Rc)
        print(f"  Rc={Rc:.1f}  E0={E0:.1f} π₂(初)={W0:+.2f} -> E={energy(phi2,0.0,Rc):.3f} π₂(末)={W1:+.3f}")

    print()
    print("A2 键耦合弱化型锚核（本轮先前的错误实现，作为对照保留）")
    print("  结论：depth 扫描不改变塌缩结局（见过程记录第 3 条）")

    print()
    print("A3 相位锁定型锚核（Rc = 4，锁宽 1.5，步数 1200，η = 0.02）")
    print(f"  {'kp':>6}{'E_init':>12}{'E_final':>12}{'pi2_init':>11}{'pi2_final':>11}")
    for kp in [0.0, 0.5, 1.0, 2.0, 4.0]:
        phi = ring_psi(4.0)
        E0 = energy(phi, kp, 4.0)
        W0 = pi2_cross(phi, 4.0)
        phi2 = flow(phi, kp, 4.0)
        E1 = energy(phi2, kp, 4.0)
        W1 = pi2_cross(phi2, 4.0)
        print(f"  {kp:>6.1f}{E0:>12.1f}{E1:>12.3f}{W0:>11.3f}{W1:>11.3f}")

    print()
    print("A4 半径扫描（kp = 2.0）")
    print(f"  {'Rc':>6}{'E_init':>12}{'E_final':>12}{'pi2_final':>11}")
    for Rc in [3.0, 4.0, 5.0, 6.0]:
        phi = ring_psi(Rc)
        E0 = energy(phi, 2.0, Rc)
        phi2 = flow(phi, 2.0, Rc)
        E1 = energy(phi2, 2.0, Rc)
        W1 = pi2_cross(phi2, Rc)
        print(f"  {Rc:>6.1f}{E0:>12.1f}{E1:>12.3f}{W1:>11.3f}")

    print()
    print("=" * 78)
    print("读数")
    print("=" * 78)
    print("判据链：M0 校验测量装置；A1 复现裸环负面结果；A3/A4 检验相位锁定型")
    print("锚核能否使环驻留（π₂ 守恒 + 能量有限）。")


if __name__ == "__main__":
    main()
