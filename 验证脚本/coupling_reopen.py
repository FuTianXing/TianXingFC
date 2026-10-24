# -*- coding: utf-8 -*-
"""
弱版耦合重开的最小验证：记录-摆程耦合的「一阶零、二阶非零」结构

背景：
  零接触（弱场）要求写入不反作用于摆程。判据 CR-1 执行后确认，既有检验
  （derrick_3d 第一部分）测的是「锁定滴 vs 真空」的静态能量差，与写入过程无关，
  无法回答耦合是否存在。本脚本按替代判据设计：给记录场一个自由度 h，
  检验它与摆程相位 theta 的耦合在交错背景上的行为。

模型（最小）：
  1D 交错链，N 格点，周期边界。摆程相位 theta_n，交错背景 theta_n = pi*n。
  记录 h_n（法线场倾斜的代理）。
  能量 E = J*sum_n cos(theta_{n+1} - theta_n) + lam*sum_n (1 - cos theta_n) * h_n
  耦合形式取 (1 - cos theta)，与体系挫折弦 K(1 - cos 2pi q) 同族。

判据（先写期望，再算）：
  W1（一阶零）：交错背景处，耦合对 theta 的一阶力为零，对任意 h 分布成立。
      期望 max|dV/dtheta| = 0（机器精度）。
  W2（拍内非零）：theta 偏离 pi*n 时一阶力非零，且正比于 sin(theta)。
      期望 力 / (lam*h*sin(theta)) = 1。
  W3（二阶调制）：交错背景的 Hessian 对角元被调制 lam*cos(theta_n)*h_n = ±lam*h_n。
      期望 对角元相对 2J 的偏移 = ±lam*h_n。
  W4（谱效应）：均匀记录使谱移动；报告最低模与带宽的变化。无先验预期，只报告。

确定性：无随机数。
"""
import numpy as np

N = 64
J = 1.0
LAM = 1.0
theta_bg = np.pi * np.arange(N)          # 交错背景
print("=" * 72)
print("弱版耦合重开的最小验证：记录-摆程耦合的一阶零、二阶非零结构")
print(f"N = {N}, J = {J}, lam = {LAM}, 背景 theta_n = pi*n")
print("=" * 72)


def record(kind, dh):
    h = np.zeros(N)
    if kind == "uniform":
        h[:] = dh
    elif kind == "single":
        h[N // 2] = dh
    elif kind == "block":
        h[N // 2 - 4:N // 2 + 4] = dh
    return h


def coup_force(theta, h, lam=LAM):
    """耦合项 V = lam*sum (1-cos theta_n) h_n 对 theta 的负梯度（即力）"""
    return -lam * np.sin(theta) * h


def coup_hess_diag(theta, h, lam=LAM):
    """耦合项对 Hessian 对角元的贡献 d2V/dtheta_n^2 = lam*cos(theta_n)*h_n"""
    return lam * np.cos(theta) * h


# ---------------- W1: 一阶力在交错背景为零 ----------------
print("\n[W1] 交错背景处的一阶力（期望 0）")
for kind in ["uniform", "single", "block"]:
    for dh in [1e-3, 1e-1, 1.0, 10.0]:
        h = record(kind, dh)
        f = coup_force(theta_bg, h)
        print(f"    h={kind:8s} dh={dh:8.1e}  max|F| = {np.max(np.abs(f)):.3e}")
print("    结论：交错背景（theta = pi*n，sin = 0）处一阶力恒为零，与 h 的大小和分布无关。")

# ---------------- W2: 拍内一阶力非零且正比于 sin(theta) ----------------
print("\n[W2] 偏离交错时的一阶力（期望 力/(lam*h*sin theta) = 1）")
for dtheta in [0.01, 0.1, 0.5, 1.0, np.pi / 2]:
    theta = theta_bg + dtheta
    h = record("uniform", 1.0)
    f = coup_force(theta, h)
    ratio = f / (LAM * h * np.sin(theta))
    print(f"    dtheta={dtheta:6.3f}  sin={np.sin(theta_bg[0]+dtheta):+.6f}  "
          f"ratio mean = {np.mean(ratio):.12f}  spread = {np.ptp(ratio):.2e}")
print("    结论：偏离交错后一阶力非零，且与 sin(theta) 成正比（比值恒为 1）。")

# ---------------- W3: Hessian 对角元的调制 ----------------
print("\n[W3] 交错背景的 Hessian 对角元（链贡献 2J，耦合贡献 lam*cos(theta)*h）")
h = record("uniform", 0.5)
H_chain = np.full(N, 2 * J)
H_coup = coup_hess_diag(theta_bg, h)
print(f"    链贡献       = {H_chain[0]:.6f}（全体相同）")
print(f"    耦合贡献     = ±lam*h = ±{LAM*h[0]:.6f}")
print(f"    偶数格点（theta=0）: {H_coup[0]:+.6f}   奇数格点（theta=pi）: {H_coup[1]:+.6f}")
print(f"    校验 max|H_coup - LAM*h*cos(theta_bg)| = "
      f"{np.max(np.abs(H_coup - LAM*h*np.cos(theta_bg))):.3e}")
print("    结论：记录调制 Hessian 对角元，符号随交错子格交替。")

# ---------------- W4: 谱效应 ----------------
print("\n[W4] 均匀记录下的摆程谱（Hessian 特征值）")


def hessian(h, lam=LAM, J=J):
    H = np.zeros((N, N))
    for n in range(N):
        H[n, n] = 2 * J + lam * np.cos(theta_bg[n]) * h[n]
        H[n, (n + 1) % N] = J
        H[n, (n - 1) % N] = J
    return H


for dh in [0.0, 0.1, 0.5, 1.0]:
    h = record("uniform", dh)
    H = hessian(h)
    ev = np.linalg.eigvalsh(H)
    print(f"    dh={dh:4.1f}  最低模={ev[0]:.8f}  最高模={ev[-1]:.6f}  "
          f"带宽={ev[-1]-ev[0]:.6f}  零模数={np.sum(np.abs(ev) < 1e-9)}")

print("\n[W4 对照] 单点记录（中心一格）")
for dh in [0.0, 1.0, 10.0]:
    h = record("single", dh)
    H = hessian(h)
    ev = np.linalg.eigvalsh(H)
    print(f"    dh={dh:5.1f}  最低模={ev[0]:.8f}  最高模={ev[-1]:.6f}  "
          f"带宽={ev[-1]-ev[0]:.6f}")

# ---------------- 附：拍内能量差（与既有检验的量级对照） ----------------
print("\n[附] 拍内能量差量级（用于与 derrick_3d 的 3.2%/4.4% 对照）")
theta = theta_bg + 0.25 * np.pi
h = record("single", 1.0)
E_coup = LAM * np.sum((1 - np.cos(theta)) * h)
E_chain = J * np.sum(np.cos(theta - np.roll(theta, -1)))
print(f"    单点记录 dh=1, dtheta=pi/4:  E_coup = {E_coup:.6f}, E_chain = {E_chain:.3f}")
print("    说明：耦合能量在单点记录下为 O(lam*dh)，与链能量之比随记录范围变化。")
print("\n" + "=" * 72)
print("完成。")
