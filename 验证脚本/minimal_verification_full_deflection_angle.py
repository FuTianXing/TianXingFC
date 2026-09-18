# -*- coding: utf-8 -*-
"""满偏角的极限地位：机制失义边界、视界过零与双覆盖锚（判据 J1 至 J5）。
映射：alpha(theta) = 180° * theta（作者 2026-09-15 裁定，线性）。
取向读法：取向向量 o(theta) = cos(alpha) n + sin(alpha) e（偏离法线方向 alpha），
对齐量 A(theta) = o(theta) . n = cos(alpha)。
验证对象是映射的几何结构（SO(3)/SU(2) 复合、双覆盖与方向读数），不模拟格点动力学。
输出：images/full_deflection_angle.png（绝对路径写入，工作区移动需同步改）。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

n0 = np.array([0.0, 0.0, 1.0])
e0 = np.array([1.0, 0.0, 0.0])

def rot(deg, axis):
    a = np.deg2rad(deg)
    k = np.asarray(axis, float); k = k / np.linalg.norm(k)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)

def ori(theta):
    a = np.deg2rad(180.0 * theta)
    return np.cos(a) * n0 + np.sin(a) * e0

A = lambda t: float(ori(t) @ n0)

def quat(theta, axis=(0, 0, 1)):
    a2 = np.deg2rad(90.0 * theta)
    k = np.asarray(axis, float); k = k / np.linalg.norm(k)
    return np.array([np.cos(a2), np.sin(a2) * k[0], np.sin(a2) * k[1], np.sin(a2) * k[2]])

def qmul(p, q):
    w1, x1, y1, z1 = p; w2, x2, y2, z2 = q
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                     w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2,
                     w1*z2 + x1*y2 - y1*x2 + z1*w2])

def angle_of(R):
    return np.rad2deg(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)))

print("=== J1a 共轴可加性（线性映射在共轴族上严格可加）===")
worst = 0.0; worst_case = None
for t1 in [0.10, 0.25, 0.5, 0.9, 1.0, 1.3]:
    for t2 in [0.05, 0.30, 0.75]:
        for axis in [(0, 0, 1), (1, 2, 3), (-2, 1, 0.5)]:
            d = np.abs(rot(180 * t1, axis) @ rot(180 * t2, axis) - rot(180 * (t1 + t2), axis)).max()
            if d > worst: worst, worst_case = d, (t1, t2)
print(f"  最大元素偏差 = {worst:.3e}（期望 <= 1e-12），最坏样本 {worst_case[0]}+{worst_case[1]}")

print("=== J1b 增益恒常（无饱和）===")
errs = [abs(angle_of(rot(180 * t, (0, 0, 1))) - 180 * t) for t in np.linspace(0.02, 0.98, 25)]
print(f"  角复原最大偏差 = {max(errs):.3e} 度（期望 <= 1e-9，theta 覆盖 0.02 至 0.98）")

print("=== J1c 异轴不可交换（诚实边界）===")
dx = rot(60, (1, 0, 0)) @ rot(60, (0, 1, 0)) - rot(60, (0, 1, 0)) @ rot(60, (1, 0, 0))
print(f"  60° 异轴换序偏差 = {np.abs(dx).max():.3f}（量级 O(alpha^2)；框架内角度挂在同一法线轴上，取共轴支）")

print("=== J2 视界过零（theta = 1/2, alpha = 90°）===")
print(f"  A(1/2) = {A(0.5):+.3e}（期望 0）；A(0.49) = {A(0.49):+.6f}，A(0.51) = {A(0.51):+.6f}（期望异号）")

print("=== J3 满偏反转（theta = 1, alpha = 180°）===")
ev = np.sort(np.linalg.eigvals(rot(180.0, (0, 0, 1))).real)
print(f"  A(1) = {A(1.0):+.15f}（期望 -1）；特征值 = {ev}（期望 [-1, -1, 1]）")
print(f"  R(1)^2 - I 最大元素偏差 = {np.abs(rot(180, (0,0,1)) @ rot(180, (0,0,1)) - np.eye(3)).max():.3e}（期望 <= 1e-12）")

print("=== J4 双覆盖锚（SU(2) 提升）===")
q1 = quat(1.0); q05 = quat(0.5)
print(f"  q(1)^2 = {np.round(qmul(q1, q1), 15)}（期望 [-1, 0, 0, 0]）")
print(f"  q(1)^4 = {np.round(qmul(qmul(q1, q1), qmul(q1, q1)), 15)}（期望 [1, 0, 0, 0]：旋量阶 4）")
print(f"  q(2) = {np.round(quat(2.0), 15)}（期望 [-1, 0, 0, 0]：SO(3) 恒等，SU(2) 变号）")
per = [np.abs(quat(t + 2) + quat(t)).max() for t in np.linspace(0, 3, 13)]
print(f"  |q(theta+2) + q(theta)| 最大 = {max(per):.3e}（期望 <= 1e-15：周期 4 覆盖）")
print(f"  q(1/2) = {np.round(q05, 12)}（w 分量 = sqrt(2)/2）")

print("=== J5 非紧致相容（theta > 1 映射仍良定）===")
for t in [1.5, 2.0, 3.7, 10.0, 100.0]:
    print(f"  theta = {t:6.1f}: alpha = {180*t:9.1f} 度, A = {A(t):+.6f}, q.w = {quat(t)[0]:+.6f}")
print("  读数：theta = 2 在 SO(3) 回到对齐（A = +1）而 SU(2) 为 -1；机制失义在 theta = 1 已经发生。")

fig, axs = plt.subplots(1, 2, figsize=(11.0, 3.8))
ts = np.linspace(0, 3, 600)
ax = axs[0]
ax.axhline(0, color='#bbbbbb', lw=1)
ax.plot(ts, np.cos(np.pi * ts), color='#1f77b4', lw=1.8)
ax.plot([0.5, 1.0], [0.0, -1.0], 'o', ms=7, color='#b3261e')
ax.annotate("视界 90 度", (0.5, 0.0), textcoords="offset points", xytext=(8, 10), fontsize=10, color='#2ca02c')
ax.annotate("满偏 180 度", (1.0, -1.0), textcoords="offset points", xytext=(8, 10), fontsize=10, color='#b3261e')
ax.set_xlabel("记录值 theta", fontsize=10); ax.set_ylabel("对齐量 A = cos(180 theta)", fontsize=10)
ax.set_title("J2 与 J3：视界过零、满偏反转", fontsize=11, color='#14508c')
ax.set_ylim(-1.25, 1.25)
ax = axs[1]
ax.axhline(-1, color='#bbbbbb', lw=1, ls='--')
ax.plot(ts, np.cos(np.pi * ts / 2), color='#ff7f0e', lw=1.8)
ax.plot([1.0, 2.0], [0.0, -1.0], 'o', ms=7, color='#b3261e')
ax.annotate("theta = 2：q = -1", (2.0, -1.0), textcoords="offset points", xytext=(-96, 10), fontsize=10, color='#b3261e')
ax.set_xlabel("记录值 theta", fontsize=10); ax.set_ylabel("四元数实部 cos(90 theta)", fontsize=10)
ax.set_title("J4：双覆盖（theta 加 2 变号）", fontsize=11, color='#14508c')
ax.set_ylim(-1.25, 1.25)
fig.subplots_adjust(left=0.07, right=0.98, top=0.87, bottom=0.16, wspace=0.24)
plt.savefig("D:/software/TianXingFC/images/full_deflection_angle.png", dpi=150)
print("图已存：images/full_deflection_angle.png")
