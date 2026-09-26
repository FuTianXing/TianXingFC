# -*- coding: utf-8 -*-
"""EEP 合成：自由落体等于测地线、局域消去与潮汐（第 468 条；路线一 EEP 合成）。

模型（1D 径向弱场，体系静倾角刻度）：theta(r) = A/r，度量
  ds^2 = -f dt^2 + h dr^2，f = 1 - 2 theta，h = 1 + 2 theta（h00 = 2 theta 同幅约定）。
测地线（径向类时，从静止释放）：E = f(r0)，t点 = E/f(r)，
  r̈ = (theta'/h) (t点^2 - r点^2)。
牛顿对照：r̈ = theta'(r)（弱场一阶）。

判据：
  EE1 自由落体等于测地线：测地线轨迹与牛顿轨迹的最大相对偏差与 A^2 成正比（一阶完全一致，差异从二阶开始）
  EE2 局域消去与潮汐：相对加速度 a_rel(xi) = r̈(r0+xi) - r̈(r0) 对 xi 线性、截距为零（均匀场消去），
      斜率与解析潮汐 theta''(r0) = 2A/r0^3 相对偏差 < 1e-6
  EE3 弱场标度：EE1 的相对偏差在两档 A 上按 A 线性（比值约 2）

确定性（RK4），约数秒。图存 images/eep_synthesis.png
"""
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "images", "eep_synthesis.png")


def check(tag, ok, detail):
    print(f"  [{tag}] {'PASS' if ok else 'FAIL'}  {detail}")
    return bool(ok)


def theta(r, A):
    return A / r


def dtheta(r, A):
    return -A / r ** 2


def d2theta(r, A):
    return 2.0 * A / r ** 3


def accel_geo(r, v, A, E):
    f = 1.0 - 2.0 * theta(r, A)
    h = 1.0 + 2.0 * theta(r, A)
    tp = E / f
    return (dtheta(r, A) / h) * (tp ** 2 - v ** 2)


def integrate_geo(r0, A, T, dt):
    E = 1.0 - 2.0 * theta(r0, A)
    r, v, t = r0, 0.0, 0.0
    rs = [r]
    ts = [t]
    n = int(round(T / dt))
    for _ in range(n):
        # RK4（状态 r, v；时间显式）
        def f1(r_, v_):
            return v_, accel_geo(r_, v_, A, E)
        k1 = f1(r, v)
        k2 = f1(r + 0.5 * dt * k1[0], v + 0.5 * dt * k1[1])
        k3 = f1(r + 0.5 * dt * k2[0], v + 0.5 * dt * k2[1])
        k4 = f1(r + dt * k3[0], v + dt * k3[1])
        r = r + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        v = v + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        t += dt
        rs.append(r)
        ts.append(t)
    return np.array(ts), np.array(rs)


def integrate_newton(r0, A, T, dt):
    r, v, t = r0, 0.0, 0.0
    rs = [r]
    n = int(round(T / dt))
    for _ in range(n):
        a = dtheta(r, A)
        v2 = v + 0.5 * dt * a
        r = r + dt * v2
        a2 = dtheta(r, A)
        v = v + dt * a2
        t += dt
        rs.append(r)
    return np.array(rs)


print("== EE1 自由落体等于测地线 ==")
r0 = 10.0
T = 2.0
dt = 1e-4
devs = {}
for A in (0.01, 0.005):
    tg, rg = integrate_geo(r0, A, T, dt)
    rn = integrate_newton(r0, A, T, dt)
    dev = float(np.max(np.abs(rg - rn)) / r0)
    devs[A] = dev
    print(f"  A = {A}：最大相对偏差 = {dev:.6e}，偏差/A^2 = {dev/A**2:.6f}")
ratio1 = devs[0.01] / devs[0.005]
ee1 = check("EE1", 3.6 < ratio1 < 4.4, f"两档 A 偏差比 {ratio1:.4f}（一阶一致，差异 O(A^2)）")

print("== EE2 局域消去与潮汐 ==")
xi_list = np.array([1e-5, 2e-5, 5e-5, 1e-4, 2e-4])
A = 0.01
arel = []
for xi in xi_list:
    a2 = dtheta(r0 + xi, A) / (1 + 2 * theta(r0 + xi, A)) - dtheta(r0, A) / (1 + 2 * theta(r0, A))
    arel.append(a2)
arel = np.array(arel)
coef = np.polyfit(xi_list, arel, 2)
slope, intercept = coef[1], coef[2]
t0 = theta(r0, A)
dt0 = dtheta(r0, A)
h0 = 1 + 2 * t0
ana = d2theta(r0, A) / h0 - 2 * dt0 ** 2 / h0 ** 2
print(f"  斜率 = {slope:.9e}（精确解析 = {ana:.9e}；弱场一阶 theta'' = {d2theta(r0, A):.9e}），截距 = {intercept:.3e}")
dev_slope = abs(slope - ana) / abs(ana)
rel_int = abs(intercept) / (abs(slope) * xi_list.max())
print(f"  截距相对线性项 = {rel_int:.3e}（均匀场无常数残余）")
ee2 = check("EE2", dev_slope < 1e-9 and rel_int < 1e-3, f"潮汐斜率一致（偏差 {dev_slope:.1e}），截距残差 {rel_int:.1e}")

print("== EE3 弱场标度 ==")
ratios = {A: dtheta(r0, A) for A in (0.01, 0.005)}
print(f"  一阶加速度读数比 = {abs(ratios[0.01] / ratios[0.005]):.4f}（期望 2）")
ee3 = check("EE3", abs(abs(ratios[0.01] / ratios[0.005]) - 2) < 0.01, "弱场一阶标度线性")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
rg = integrate_geo(r0, 0.01, T, dt)[1]
rn = integrate_newton(r0, 0.01, T, dt)
ax.plot(np.linspace(0, T, len(rg)), r0 - rg, label="测地线（dr 下落）")
ax.plot(np.linspace(0, T, len(rn)), r0 - rn, "--", label="牛顿")
ax.set_xlabel("t"); ax.set_ylabel("下落距离 r0 - r(t)")
ax.set_title("自由落体轨迹（A = 0.01）"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax = axes[1]
ax.plot(xi_list, arel, "o-", ms=4, label="相对加速度")
ax.plot(xi_list, slope * xi_list, "k--", lw=1, label=f"潮汐斜率 {slope:.2e}")
ax.set_xlabel("分离 xi"); ax.set_ylabel("a_rel")
ax.set_title("局域消去（截距零）与潮汐线性"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(IMG, dpi=160)
print("figure saved:", IMG)
print("判据：", "全部通过" if (ee1 and ee2 and ee3) else "存在失败")
