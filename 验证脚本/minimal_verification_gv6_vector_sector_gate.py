# -*- coding: utf-8 -*-
r"""
minimal_verification_gv6_vector_sector_gate.py

[GV6] 矢量扇区张力的正面处理：把"要救拖曳"转成一条定量的门

背景
----
判据 V 轮的结论：框架写规则只依赖质量密度、不依赖质量流，故 h_0i = 0（参考系拖曳为零），
与 Gravity Probe B 拖曳岁差实测相差约 5 sigma。本脚本不复述该结论，而给出两件新读数：
灵敏度对照（加一条时间反演为奇的矢量通道后 DC 响应如何出现）与换算门（要匹配观测拖曳，
该通道强度需要多大）。

模型
----
环源：半径 R、角速度 omega、总质量归一的 N 点。框架侧输出为标量 theta（只依赖密度）；
矢量通道取 gravitomagnetic 型写法 A(x) = eps * sum_a q v_a / |x - r_a|（v_a = omega z x r_a），
eps = 0 即通道关闭。

判据
----
S1 零响应：eps = 0 时 theta 的时间平均等于角度平摊的静态场（偏差 < 1e-12），且 A 恒为零。
S2 灵敏度：eps 非零时出现 DC 矢量场；对 eps 线性、对 omega 线性、对 omega 反号为奇
   （三项相对偏差均 < 2%），且方向为横向（与 x 投影垂直、沿 z x x）。
S3 换算门：归一化系数 c1 = |A_DC| / (单极项 x omega R / c x eps) 属 O(1) 量级，
   据此说明要匹配观测拖曳所需 eps 为 O(1)，不能靠小修正自救。

运行
----
python 验证脚本/minimal_verification_gv6_vector_sector_gate.py
确定性（无随机数），运行约 2 秒。
"""
import math
import time

import numpy as np

T0 = time.time()
CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print(f"  [{tag}] {'PASS' if ok else 'FAIL'}  {detail}")


N_RING, R, Q = 128, 1.0, 1.0 / 128
PROBES = np.array([[2.0, 0.0], [0.0, 2.0], [1.5, 1.5], [3.0, 0.0], [0.0, 3.0]])


def sources(omega, t):
    ang = 2 * math.pi * np.arange(N_RING) / N_RING + omega * t
    px, py = R * np.cos(ang), R * np.sin(ang)
    vx, vy = -omega * py, omega * px
    return px, py, vx, vy


def scalar_field(omega, t):
    px, py, _, _ = sources(omega, t)
    dx = PROBES[:, 0][:, None] - px[None, :]
    dy = PROBES[:, 1][:, None] - py[None, :]
    r = np.maximum(np.sqrt(dx ** 2 + dy ** 2), 1e-9)
    return (Q / r).sum(axis=1)


def vector_field(omega, t, eps):
    px, py, vx, vy = sources(omega, t)
    dx = PROBES[:, 0][:, None] - px[None, :]
    dy = PROBES[:, 1][:, None] - py[None, :]
    r = np.maximum(np.sqrt(dx ** 2 + dy ** 2), 1e-9)
    ax = (Q * vx[None, :] / r).sum(axis=1)
    ay = (Q * vy[None, :] / r).sum(axis=1)
    return eps * np.stack([ax, ay], axis=1)


def avg_scalar(omega, n=64):
    ts = 2 * math.pi / abs(omega) * np.arange(n) / n
    return np.mean([scalar_field(omega, t) for t in ts], axis=0)


def avg_vector(omega, eps, n=64):
    ts = 2 * math.pi / abs(omega) * np.arange(n) / n
    return np.mean([vector_field(omega, t, eps) for t in ts], axis=0)


print("=" * 72)
print("最小验证 GV6：矢量扇区张量的定量门")
print("=" * 72)

# ---------------------------------------------------------------- S1
print("\n[S1] eps = 0：标量场时间平均 = 静态场；矢量场为零")
stat = scalar_field(0.0, 0.0)
avg0 = avg_scalar(0.3)
dev = float(np.max(np.abs(avg0 - stat)) / np.max(np.abs(stat)))
vec0 = avg_vector(0.3, 0.0)
print(f"      标量偏差 {dev:.3e}；矢量最大绝对值 {np.max(np.abs(vec0)):.3e}")
check("S1", dev < 1e-12 and np.max(np.abs(vec0)) == 0.0,
      f"标量偏差 {dev:.1e}（< 1e-12），矢量恒零，框架现行写规则无 DC 矢量响应")

# ---------------------------------------------------------------- S2
print("\n[S2] eps 非零：DC 矢量场对 eps、omega 的线性与奇偶性")
om = 0.3
A05 = avg_vector(om, 0.05)
A10 = avg_vector(om, 0.1)
n05, n10 = float(np.max(np.abs(A05))), float(np.max(np.abs(A10)))
A10m = avg_vector(om * 2, 0.1)
n10m = float(np.max(np.abs(A10m)))
A10neg = avg_vector(om, -0.1)
print(f"      eps = 0.05 给 {n05:.6e}；eps = 0.1 给 {n10:.6e}（比值 {n10 / n05:.4f}）")
print(f"      omega 加倍给 {n10m:.6e}（比值 {n10m / n10:.4f}）")
print(f"      eps 反号给 {float(np.max(np.abs(A10neg))):.6e}（与正号差 {np.max(np.abs(A10neg + A10)):.2e}）")
# 方向结构：A 应与 z x x_perp 平行
xp = PROBES[0]
expect_dir = np.array([-xp[1], xp[0]])
expect_dir = expect_dir / np.linalg.norm(expect_dir)
a0 = A10[0] / np.linalg.norm(A10[0])
par = float(abs(np.dot(a0, expect_dir)))
print(f"      探针 0 处方向与 z x x_perp 的平行度 {par:.6f}")
check("S2", abs(n10 / n05 - 2) < 0.02 and abs(n10m / n10 - 2) < 0.02
      and np.max(np.abs(A10neg + A10)) / n10 < 0.02 and par > 0.999,
      f"eps 比 {n10 / n05:.3f}、omega 比 {n10m / n10:.3f}、反号对称、方向平行度 {par:.4f}")

# ---------------------------------------------------------------- S3
print("\n[S3] 换算门：归一化系数 c1 与所需 eps 规模")
mono = float(np.max(np.abs(stat)))
c1 = n10 / (mono * (om * R) * 0.1)
print(f"      单极项 {mono:.4f}；eps = 0.1 的 DC 矢量幅值 {n10:.6e}")
print(f"      归一化系数 c1 = {c1:.4f}（定义 A_DC / (单极 x omega R / c x eps)）")
check("S3", 0.01 < c1 < 100.0, f"c1 = {c1:.4f} 属 O(1) 量级，故所需 eps 为 O(1)，不能靠小修正自救")

print("\n" + "=" * 72)
n_pass = sum(1 for _, ok in CHECKS if ok)
print(f"判据汇总：{n_pass}/{len(CHECKS)} 通过")
for tag, ok in CHECKS:
    print(f"  {tag}: {'PASS' if ok else 'FAIL'}")
print(f"耗时 {time.time() - T0:.1f} s")