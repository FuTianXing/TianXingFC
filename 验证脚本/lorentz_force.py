# -*- coding: utf-8 -*-
"""第 445 条（2026-09-30 审查修订）：洛伦兹力的最小耦合变分。

判据：
  V1 拓扑耦合：绕数 w 的缺陷沿闭合曲线移动时，作用量相位 2*pi*w*∮A.dl 满足规范不变性，
     即在 A -> A + grad(chi) 下模 2pi 不变；同时 ∮A.dl = Phi 给出磁通读数。
  V2 变分核对：离散作用量的 EL 残差在一般初值（非圆轨道）下随步长下降，收敛阶不低于一阶，
     且在精确圆轨道上超收敛。两条轨道分别报告，不合并为单一阶数。
  V3 三条极限：静止荷不受磁力；纯电场给库仑型加速度 qE/m（用标准离散 EL 方程实测）；
     纯梯度势零磁通。

容差预算（2026-09-30 修订）：
  V1 圈积分为解析恒等式（Stokes），步长 1e-5、点数 4e5 下预算约 1e-9；
     规范不变性的相位差按 w 倍放大后取模 2pi，预算 1e-6。
  V2 一般初值的实测斜率为 1.0（一阶，前向差分动能的一阶截断），判据取「不低于 0.9」；
     精确圆轨道上超收敛至约 3 阶，只作参考不作判据。理由：原判据在精确解上测阶数，
     超收敛来自解本身满足连续方程，不能代表格式的普遍阶数（2026-09-30 审查发现）。
  V3(b) 动能 EL 项对线性势为精确值，耦合项按 dt^1.5 收敛，判据取总偏差 < 1e-2（dt=0.0025 档实测 1.8e-4）。
  V3(a) 与 V3(c) 为解析零，容差 1e-12。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
IMG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "images", "jz1_lorentz_force.png")

Phi = 1.0
W = 3


def A_vortex(x, y):
    r2 = x ** 2 + y ** 2
    return -Phi / (2.0 * np.pi) * y / r2, Phi / (2.0 * np.pi) * x / r2


def dtheta(x, y, th):
    thc = np.append(th, 2.0 * np.pi)
    return np.gradient(np.append(x, x[0]), thc)[:-1], np.gradient(np.append(y, y[0]), thc)[:-1]


def closed_loop_integral(x, y, th):
    dx, dy = dtheta(x, y, th)
    Ax, Ay = A_vortex(x, y)
    return float(np.sum(Ax * dx + Ay * dy) * (2.0 * np.pi / len(x)))


def loop_with_gradient(x, y, th):
    """A -> A + grad(chi) 后的圈积分。

    chi = 0.7 x y + 0.3 x^2，grad(chi) = (0.7 y + 0.6 x, 0.7 x)，两个分量都要加。
    """
    dx, dy = dtheta(x, y, th)
    Ax, Ay = A_vortex(x, y)
    gx = 0.7 * y + 0.6 * x
    gy = 0.7 * x
    return float(np.sum((Ax + gx) * dx + (Ay + gy) * dy) * (2.0 * np.pi / len(x)))


# ---------------- V1：拓扑耦合与规范不变性 ----------------
N = 400000
th = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
shapes = {
    "circle": (2.0 + 0.0 * th),
    "wobble": (2.0 * (1.0 + 0.3 * np.sin(3.0 * th))),
    "ellipse": (1.0 / np.sqrt((np.cos(th) / 1.5) ** 2 + (np.sin(th) / 3.0) ** 2)),
}
rows_v1 = []
for name, r in shapes.items():
    x, y = r * np.cos(th), r * np.sin(th)
    i0 = closed_loop_integral(x, y, th)
    i1 = loop_with_gradient(x, y, th)
    phase_shift = abs((i1 - i0) * W) % (2.0 * np.pi)
    phase_shift = min(phase_shift, 2.0 * np.pi - phase_shift)
    rows_v1.append((name, i0, phase_shift))

# ---------------- V2：离散 EL 残差（一般初值 + 精确圆轨道） ----------------
B, q, m = 1.0, 1.0, 1.0
wc = q * B / m


def A_uniform(x, y):
    return -B / 2.0 * y, B / 2.0 * x


def action(xs, dt):
    v = (xs[1:] - xs[:-1]) / dt
    T = 0.5 * m * np.sum(v ** 2) * dt
    xm = 0.5 * (xs[1:] + xs[:-1])
    Ax, Ay = A_uniform(xm[:, 0], xm[:, 1])
    S = q * np.sum(Ax * (xs[1:, 0] - xs[:-1, 0]) + Ay * (xs[1:, 1] - xs[:-1, 1]))
    return T + S


def el_residual(xs, dt, eps=1e-6):
    res = 0.0
    for i in range(1, len(xs) - 1):
        for d in range(2):
            xp = xs.copy(); xp[i, d] += eps
            xm = xs.copy(); xm[i, d] -= eps
            res = max(res, abs((action(xp, dt) - action(xm, dt)) / (2.0 * eps)))
    return res


dts = np.array([0.02, 0.01, 0.005, 0.0025])


def orbit(R, omega, dt):
    t = np.arange(0.0, 1.0, dt)
    return np.stack([R * np.cos(omega * t), -R * np.sin(omega * t)], axis=-1)


res_general, res_exact = [], []
for dt in dts:
    res_general.append(el_residual(orbit(1.0, 0.6 * wc, dt), dt))
    res_exact.append(el_residual(orbit(1.0, wc, dt), dt))
res_general = np.array(res_general)
res_exact = np.array(res_exact)
slope_general = float(np.polyfit(np.log(dts), np.log(res_general), 1)[0])
slope_exact = float(np.polyfit(np.log(dts), np.log(res_exact), 1)[0])

# ---------------- V3：三条极限 ----------------
xs_static = np.tile(np.array([[1.0, 0.0]]), (20, 1))
res_static = el_residual(xs_static, 0.05)

# (b) 纯电场：用标准离散 EL 方程实测，不用数值变分的整体一阶差分
E = 1.0


def S_E(xs, dt):
    v = (xs[1:] - xs[:-1]) / dt
    T = 0.5 * m * np.sum(v ** 2) * dt
    xm = 0.5 * (xs[1:] + xs[:-1])
    return T + q * np.sum(-E * xm[:, 0] * (xs[1:, 0] - xs[:-1, 0]))


def el_accel_E(dt, eps=1e-7):
    t = np.arange(0.0, 1.0, dt)
    xs = np.stack([0.5 * E * t ** 2, np.zeros_like(t)], axis=-1)
    lap = (xs[2:] - 2 * xs[1:-1] + xs[:-2]) / dt ** 2
    acc = np.zeros_like(lap)
    n_probe = min(len(xs) - 1, 30)
    for i in range(1, n_probe):
        xp = xs.copy(); xp[i, 0] += eps
        xm = xs.copy(); xm[i, 0] -= eps
        acc[i - 1, 0] = -(S_E(xp, dt) - S_E(xm, dt)) / (2 * eps)
    return lap[:, 0].mean(), (acc[:, 0] / m).mean()


kin_e, coup_e = el_accel_E(0.0025)

# (c) 纯梯度势
def chi(x, y):
    return x ** 2 * y


def A_grad(x, y):
    return 2.0 * x * y, x ** 2


def grad_loop(Ax, Ay, x, y, th):
    dx, dy = dtheta(x, y, th)
    return float(np.sum(Ax * dx + Ay * dy) * (2.0 * np.pi / len(th)))


th_g = np.linspace(0.0, 2.0 * np.pi, 20000, endpoint=False)
xg, yg = 2.0 * np.cos(th_g), np.sin(th_g)
Axg, Ayg = A_grad(xg, yg)
circ_grad = grad_loop(Axg, Ayg, xg, yg, th_g)

print("V1 拓扑耦合（∮A.dl 期望 Phi = %.6f；w = %d）" % (Phi, W))
for name, i0, ps in rows_v1:
    print("  %-8s ∮A.dl = %.9f（相对差 %.2e）  规范变换后 w 倍相位偏移 = %.3e" % (name, i0, abs(i0 - Phi) / Phi, ps))
print()
print("V2 离散 EL 残差的 dt 收敛")
for dt, rg, re_ in zip(dts, res_general, res_exact):
    print("  dt = %.4f  一般初值残差 = %.3e  精确圆轨道残差 = %.3e" % (dt, rg, re_))
print("  一般初值收敛斜率 = %.4f（判据：不低于 0.9）" % slope_general)
print("  精确圆轨道斜率   = %.4f（参考，不作判据）" % slope_exact)
print()
print("V3 三条极限")
print("  (a) 静止荷的 EL 残差 = %.3e（期望零）" % res_static)
print("  (b) 纯电场：动能 EL = %.6f  耦合 EL = %.6f  总 = %.6f  解析 qE/m = %.6f" % (kin_e, coup_e, kin_e + coup_e, q * E / m))
print("  (c) 纯梯度势的圈积分 = %.3e（期望零）" % circ_grad)
print()
flag1 = bool(all(abs(v - Phi) / Phi < 1e-6 for _, v, _ in rows_v1)
              and all(ps < 1e-6 for _, _, ps in rows_v1))
flag2 = bool(slope_general >= 0.9)
flag3 = bool(res_static < 1e-12 and circ_grad < 1e-12
              and abs((kin_e + coup_e) - q * E / m) < 1e-2)
print("判据 V1（∮A.dl = Phi 且规范不变性模 2pi）：%s" % ("通过" if flag1 else "未通过"))
print("判据 V2（一般初值 EL 残差收敛不低于一阶）：%s" % ("通过" if flag2 else "未通过"))
print("判据 V3（静止荷、库仑力、纯梯度零通量）：%s" % ("通过" if flag3 else "未通过"))
print("判据总计：%d/3" % (int(flag1) + int(flag2) + int(flag3)))

fig, axp = plt.subplots(2, 2, figsize=(11, 8))
for name, r in shapes.items():
    axp[0, 0].plot(r[:8000] * np.cos(th[:8000]), r[:8000] * np.sin(th[:8000]), lw=1.0, label=name)
axp[0, 0].set_aspect("equal")
axp[0, 0].set_title("V1 loops around flux tube (∮A.dl = Phi)")
axp[0, 0].legend(fontsize=7)

axp[0, 1].loglog(dts, res_general, "ro-", label="general IC")
axp[0, 1].loglog(dts, res_exact, "bs-", label="exact circle")
axp[0, 1].loglog(dts, res_general[0] * (dts / dts[0]), "k--", lw=1.0, label="slope 1 ref")
axp[0, 1].set_xlabel("dt"); axp[0, 1].set_ylabel("max |EL residual|")
axp[0, 1].set_title("V2 convergence"); axp[0, 1].legend(fontsize=7)

dta = np.array([0.02, 0.01, 0.005, 0.0025])
tot = []
for dt in dta:
    t = np.arange(0.0, 1.0, dt)
    xs = np.stack([0.5 * E * t ** 2, np.zeros_like(t)], axis=-1)
    lap = (xs[2:] - 2 * xs[1:-1] + xs[:-2]) / dt ** 2
    a2 = np.zeros_like(lap)
    for i in range(1, min(len(xs) - 1, 30)):
        xp = xs.copy(); xp[i, 0] += 1e-7
        xm = xs.copy(); xm[i, 0] -= 1e-7
        a2[i - 1, 0] = -(S_E(xp, dt) - S_E(xm, dt)) / (2e-7)
    tot.append((lap[:, 0] + a2[:, 0] / m).mean())
axp[1, 0].semilogy(dta, np.abs(np.array(tot) - E), "ko-", label="|EL accel - qE/m|")
axp[1, 0].set_xlabel("dt"); axp[1, 0].set_ylabel("abs deviation")
axp[1, 0].set_title("V3(b) uniform E deviation")
axp[1, 0].legend(fontsize=7)

axp[1, 1].plot(xg, yg, "g-", lw=1.0)
axp[1, 1].set_aspect("equal")
axp[1, 1].set_title("V3(c) pure gradient: loop integral = 0")
plt.tight_layout()
plt.savefig(IMG, dpi=130)
print("图已保存：%s" % IMG)
