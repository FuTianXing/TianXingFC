# -*- coding: utf-8 -*-
"""
裁决实验三（修正重跑）：含辐射塌缩对照
对应《注记-两种危险为什么不成立》实验三 + 评审修正。

修正上次两个协议错误：
  1. 绕数沿环管中心线回路测（绕环一圈的相位提升和），不绕 z 轴
  2. 相位场用提升变量 phi_tilde（不取模连续场），演化中提升差只能连续变化、
     不能跳 2π 整数倍 → 相滑移（跳墙）被显式禁止，过阻尼流不可穿越分支切割

三种动力学对照：
  A 无阻尼 Verlet（复现 derrick_3d [2c] 弹道基准：收缩能量保留、可撞墙）
  B 过阻尼梯度流（耗散极限：绕数严格保持、停在能量极小）
  C 阻尼 Verlet（中间态：辐射耗散，预期停在驻留点、绕数保持）

环构型：环形涡旋（torus），xy 平面半径 R=9、管半径 rho0=2，
相位 phi = n * alpha * g(d)，alpha = atan2(y,x) 沿环方向角，g = 环管权重。
提升变量 = 不取模的 phi（缠绕 2πn 沿环连续）。
全部结果只打印终端。已固化（2026-09-06），对应注记《注记-裁决实验四件套复跑》。

评估修正（2026-09-06，两层勘误）：
  其一，本文件头部原称"提升变量（不取模的 phi）"；实际 make_ring 用 arctan2
  构造，存储值已折绕在 (−π·g, π·g]，并非提升坐标。
  其二，"提升差不能跳 2π 整数倍 → 相滑移被显式禁止"不成立：连续流下键相位差
  可连续穿过 π（无需跳 2π），且全局单值提升场的环量和恒为零、根本装不下涡旋。
  本脚本运行结果（三种动力学绕数全部丢失，见注记实验三）与修正后的分析一致。
  ring_winding（截面排序最小像差和）为正确测量，W0 = +1 校验通过。
"""
import numpy as np

print("=" * 72)
print("裁决实验三（修正重跑）：含辐射塌缩对照")
print("=" * 72)

N = 48
c = N // 2
R0 = 9.0
RHO0 = 2.0
N_LOOP = 48          # 环中心线回路的采样点数

Y, X = np.meshgrid(np.arange(N) - c, np.arange(N) - c, indexing="ij")
Z = np.broadcast_to((np.arange(N) - c)[:, None, None], (N, N, N))
X3 = np.broadcast_to(X[:, :, None], (N, N, N))
Y3 = np.broadcast_to(Y[:, :, None], (N, N, N))
rho_xy = np.sqrt(X3 ** 2 + Y3 ** 2)
alpha = np.arctan2(Y3, X3)                      # 极角（沿环方向角）
d_ring = np.sqrt((rho_xy - R0) ** 2 + Z ** 2)   # 到环管中心线距离
g = 0.5 * (1.0 - np.tanh((d_ring - RHO0) / 1.0))  # 管权重：管内 1 管外 0


def make_ring(n_topo=1.0):
    """环形涡旋（正确构造）：涡旋线弯成环，相位在管截面内缠绕 2πn。
    phi = n * atan2(rho_xy - R0, Z) * g（截面坐标：径向偏移 + z），管外衰减。"""
    return n_topo * np.arctan2(rho_xy - R0, Z) * g


def ring_winding(phi):
    """绕管截面小环测绕数：中心线 θ0=0 处法平面（xy 平面 +x 方向），
    截面小环 (rho-R0)²+z² <= r_c² 的格点按截面角 beta = atan2(rho-R0, z) 排序，
    相位提升和 / 2π。"""
    r_c = 1.2
    sec = (np.abs(rho_xy - R0) <= r_c) & (np.abs(Z) <= r_c) & (g > 0.7)
    beta = np.arctan2(rho_xy[sec] - R0, Z[sec])
    ph = phi[sec]
    if ph.size < 4:
        return 0.0
    order = np.argsort(beta)
    ps = ph[order]
    W = 0.0
    for k in range(len(ps) - 1):
        d = ps[k + 1] - ps[k]
        W += (d + np.pi) % (2 * np.pi) - np.pi      # 最小像差和（derrick_3d 同款）
    d = ps[0] - ps[-1]
    W += (d + np.pi) % (2 * np.pi) - np.pi
    return W / (2 * np.pi)


def ring_radius(phi):
    """环半径：相位梯度幅值加权的 |r_xy| 质心。"""
    grad = np.zeros_like(phi)
    for ax in range(3):
        grad += (np.roll(phi, 1, ax) - np.roll(phi, -1, ax)) ** 2
    wgt = np.sqrt(grad) * g
    s = wgt.sum()
    if s < 1e-12:
        return np.nan
    rxy = np.sqrt(X3 ** 2 + Y3 ** 2)
    return (wgt * rxy).sum() / s


def energy_xy(phi):
    E = 0.0
    for ax in range(3):
        d1 = np.roll(phi, 1, ax) - phi
        d2 = np.roll(phi, -1, ax) - phi
        E += 0.5 * np.sum(1.0 - np.cos(d1)) + 0.5 * np.sum(1.0 - np.cos(d2))
    return E


def grad_flow(phi, iters=4000, eta=0.05):
    """过阻尼梯度流：提升变量连续演化，提升差不能跳 2π 倍数（相滑移禁止）。"""
    phi = phi.copy()
    rec = []
    for t in range(iters):
        s = np.zeros_like(phi)
        for ax in range(3):
            s += (np.sin(phi - np.roll(phi, 1, ax)) + np.sin(phi - np.roll(phi, -1, ax)))
        phi = phi - eta * s
        if t % 500 == 0:
            rec.append((t, energy_xy(phi), ring_winding(phi), ring_radius(phi)))
    return phi, rec


def verlet(phi, iters=4000, dt=0.02, gamma=0.0):
    """Verlet 动力学（提升变量）。gamma = 阻尼（辐射通道）。"""
    phi = phi.copy()
    vel = np.zeros_like(phi)
    a = np.zeros_like(phi)
    for ax in range(3):
        a += (np.sin(phi - np.roll(phi, 1, ax)) + np.sin(phi - np.roll(phi, -1, ax)))
    a = -a
    rec = []
    for t in range(iters):
        phi = phi + dt * vel + 0.5 * dt ** 2 * a
        a_new = np.zeros_like(phi)
        for ax in range(3):
            a_new += (np.sin(phi - np.roll(phi, 1, ax)) + np.sin(phi - np.roll(phi, -1, ax)))
        a_new = -a_new
        vel = vel + 0.5 * dt * (a + a_new)
        if gamma > 0:
            vel = vel * (1.0 - gamma * dt)
        a = a_new
        if t % 500 == 0:
            rec.append((t, energy_xy(phi) + 0.5 * np.sum(vel ** 2),
                        ring_winding(phi), ring_radius(phi)))
    return phi, rec


def report(name, rec, E0, W0):
    print(f"\n  [{name}]")
    for t, E, W, R in rec:
        print(f"    t={t:5d}: E={E:9.2f}  W={W:+7.3f}  环半径 R={R:6.2f}")
    Wf, Rf = rec[-1][2], rec[-1][3]
    print(f"    绕数保持：W0={W0:+.3f} → Wf={Wf:+.3f}"
          f"（{'保持 ✓' if abs(Wf - W0) < 0.3 else '解开 ✗'}）")
    print(f"    环驻留：R0={R0:.1f} → Rf={Rf:.2f}"
          f"（{'停在驻留点附近' if not np.isnan(Rf) and 0.5 < Rf < R0 else '塌缩/漂移'}）")


# 初始构型
phi0 = make_ring(1.0)
W0 = ring_winding(phi0)
E0 = energy_xy(phi0)
print(f"\n初始环形涡旋：R0 = {R0}，绕数 W0 = {W0:+.3f}（期望 1.0），能量 E0 = {E0:.2f}")
if abs(W0 - 1.0) > 0.2:
    print("警告：初始绕数测量异常，回路采样不足或构型问题，结果需谨慎解读")

# A. 无阻尼弹道（[2c] 基准复现）
phiA, recA = verlet(phi0.copy(), iters=4000, dt=0.02, gamma=0.0)
report("A 无阻尼 Verlet（弹道，[2c] 基准）", recA, E0, W0)

# B. 过阻尼梯度流（耗散极限，相滑移禁止）
phiB, recB = grad_flow(phi0.copy(), iters=4000, eta=0.05)
report("B 过阻尼梯度流（耗散极限）", recB, E0, W0)

# C. 阻尼 Verlet（中间态，辐射耗散）
phiC, recC = verlet(phi0.copy(), iters=4000, dt=0.02, gamma=0.05)
report("C 阻尼 Verlet（辐射耗散）", recC, E0, W0)

print("\n判定：B/C 绕数保持且环停在有限半径 → 耗散下环不散（粒子自稳定）；"
      "若 B/C 绕数解开 → 耗散下也散（钉扎需回来）。")
print("\n实验三重跑完成（仅打印，未写任何文档）。")
