# -*- coding: utf-8 -*-
r"""
vortex_delta_deinteg.py

[维度外推审计 · 周期边界版] 第 598 条藏身术在 2D 涡旋芯是否成立
（判据 V1 至 V3，第 600 条；首版开边界留档教训：不守绕数、涡旋真空化）

拓扑扇区（2D 的 FK 扣 Q）：周期环面上每行 q 的 x 环绕数标签 Qx[q] 与
每列 p 的 y 环绕数标签 Qy[p]，初值处整化、全程固定；wrap 键应变
dv_wrap = v[0] - v[N-1] + Q。行/列环绕数守恒与 1D 的 Q 机制同构，
涡旋荷由行环绕数的非均匀分布承载（错排线）。

判据（先立理论期望再实现）
V1（涡旋存活）：行环绕数标签守恒（终态整化与初值一致）、能量单调。
  先立：周期边界守荷，J = k = 1 涡旋存活。
V2（核心）：芯区 delta^12 非整残留（delta/2π 到最近整数平均距离）：
  连续角向 0.25（藏身术失效、三维供货窗口重开）对整数化 <0.05
  （第 598 条判否外推成立、主路线出口关闭）。中间态另行判读。
V3（区分力对照）：y 无关一维墙压进同款 2D 模型，应复现整数化。
可逆性：判据敏感（网格、初值、扇区标签、弛豫协议绑定）。
模型：N = 64 × 64 周期环面三分量，k = J = 1，eta = 0.3/(8 + 8π²J)。
运行：python 验证脚本/vortex_delta_deinteg.py
"""
import math
import time

import numpy as np

T_START = time.time()
N = 64
J_M = 1.0
K_EL = 1.0
TWO_PI = 2.0 * math.pi
RC = 6.0
PAIRS = ((0, 1), (0, 2), (1, 2))

print('=' * 76)
print('维度外推审计（周期边界版）：第 598 条藏身术在 2D 涡旋芯（第 600 条）')
print('=' * 76)
print()
print('环面三分量，行/列环绕数标签 Qx[q]、Qy[p]（初值整化全程固定），')
print('wrap 键应变 = v[0] - v[N-1] + Q；涡旋初值 atan2 场。')
print()


def init_vortex(N):
    n = np.arange(N)
    x, y = np.meshgrid(n, n, indexing='ij')
    x0 = y0 = (N - 1) / 2.0
    v1 = np.arctan2(y - y0, x - x0) / TWO_PI
    return np.stack([v1, np.zeros((N, N)), np.zeros((N, N))])


def init_1dwall(N):
    n = np.arange(N)
    prof = 0.5 * (1.0 + np.tanh((n - N / 2) / 4.0))
    return np.stack([np.tile(prof[:, None], (1, N)),
                     np.zeros((N, N)), np.zeros((N, N))])


def sector(v3):
    """行环绕数 Qx[q]（a=0 链）与列环绕数 Qy[p]。"""
    vx = v3[0]
    qx = np.array([int(round(float(vx[0, q] - vx[N - 1, q]))) for q in range(N)])
    qy = np.array([int(round(float(vx[p, 0] - vx[p, N - 1]))) for p in range(N)])
    return qx, qy


def bonds(v3, qx, qy):
    """三链的 x/y 键应变（wrap 含扇区标签）。"""
    dvx = np.empty((3, N, N))
    dvy = np.empty((3, N, N))
    for a in range(3):
        dvx[a, :-1, :] = v3[a, 1:, :] - v3[a, :-1, :]
        dvx[a, -1, :] = v3[a, 0, :] - v3[a, -1, :] + (qx if a == 0 else 0.0)
        dvy[a, :, :-1] = v3[a, :, 1:] - v3[a, :, :-1]
        dvy[a, :, -1] = v3[a, :, 0] - v3[a, :, -1] + (qy if a == 0 else 0.0)
    return dvx, dvy


def energy(v3, qx, qy):
    dvx, dvy = bonds(v3, qx, qy)
    e_el = 0.5 * K_EL * (float(np.sum(dvx ** 2)) + float(np.sum(dvy ** 2)))
    e_het = 0.0
    for (a, b) in PAIRS:
        e_het += J_M * float(np.sum(1.0 - np.cos(TWO_PI * (v3[a] - v3[b]))))
    return e_el, e_het, e_el + e_het


def grad(v3, qx, qy):
    dvx, dvy = bonds(v3, qx, qy)
    g = np.empty_like(v3)
    for a in range(3):
        # 键项：∂E/∂v[p] = -(dv[p] 作为头) + (dv[p-1] 作为尾)，roll 形式
        # 自动含 wrap 键（Q 为常数，导数只走 dv）
        gx = -dvx[a] + np.roll(dvx[a], 1, axis=0)
        gy = -dvy[a] + np.roll(dvy[a], 1, axis=1)
        g[a] = K_EL * (gx + gy)
    for (a, b) in PAIRS:
        d = TWO_PI * (v3[a] - v3[b])
        s = J_M * TWO_PI * np.sin(d)
        g[a] += s
        g[b] -= s
    return g


def relax(v0, qx, qy, nsteps=40000):
    v = v0.copy()
    eta = min(0.01, 0.3 / (8.0 + 8.0 * math.pi ** 2 * J_M))
    e_prev = energy(v, qx, qy)[2]
    mono = True
    for step in range(nsteps):
        v = v - eta * grad(v, qx, qy)
        if step % 4000 == 3999:
            e = energy(v, qx, qy)[2]
            if e > e_prev + 1e-9:
                mono = False
            if abs(e - e_prev) < 1e-10 * max(1.0, abs(e_prev)):
                return v, e, True, mono
            e_prev = e
    return v, energy(v, qx, qy)[2], False, mono


def resid_delta(v3, r_c=RC):
    c = (N - 1) / 2.0
    n = np.arange(N)
    x, y = np.meshgrid(n, n, indexing='ij')
    mask = (x - c) ** 2 + (y - c) ** 2 <= r_c ** 2
    fr = (TWO_PI * (v3[0] - v3[1]))[mask] / TWO_PI
    d = np.abs(fr - np.round(fr))
    d = np.minimum(d, 1.0 - d)
    return float(np.mean(d)), int(mask.sum())


print('--- V1 涡旋存活（周期边界，40k 步）---')
v0 = init_vortex(N)
qx0, qy0 = sector(v0)
nz_x = int(np.sum(np.abs(qx0)))
nz_y = int(np.sum(np.abs(qy0)))
v_end, e_end, conv, mono = relax(v0, qx0, qy0)
qx1, qy1 = sector(v_end)
keep_x = int(np.sum(np.abs(qx1)))
keep_y = int(np.sum(np.abs(qy1)))
print('   E: %.2f -> %.4f（收敛 %s，单调 %s）' % (energy(v0, qx0, qy0)[2], e_end, conv, mono))
print('   扇区标签：|Qx| 和 %d -> %d，|Qy| 和 %d -> %d（守恒判据：标签恒定）'
      % (nz_x, keep_x, nz_y, keep_y))
v1_ok = mono and np.array_equal(qx0, qx1) and np.array_equal(qy0, qy1)
print('   [%s] V1' % ('PASS' if v1_ok else 'FAIL'))
print()

print('--- V2 芯区 delta^12 非整残留（核心）---')
r2, npts = resid_delta(v_end)
print('   R_delta = %.4f（芯区 %d 点；连续角向 0.25 对整数化 <0.05）' % (r2, npts))
if 0.05 <= r2 < 0.15:
    verdict2 = '中间态（部分连续）：另行判读'
elif r2 >= 0.15:
    verdict2 = '连续/楼梯角向：藏身术失效，三维供货窗口重开'
else:
    verdict2 = '整数化：第 598 条判否外推成立，主路线出口关闭'
print('   判读：%s' % verdict2)
print()

print('--- V3 一维对照（y 无关扭转行，应整数化）---')
v0_1d = init_1dwall(N)
qx1d, qy1d = sector(v0_1d)
v1d, e1d, conv1, mono1 = relax(v0_1d, qx1d, qy1d)
r3, _ = resid_delta(v1d)
print('   R_delta(1D退化) = %.4f（E: %.2f -> %.4f，收敛 %s）'
      % (r3, energy(v0_1d, qx1d, qy1d)[2], e1d, conv1))
v3_ok = r3 < 0.05
print('   [%s] V3' % ('PASS' if v3_ok else 'FAIL'))
print()

print('=' * 76)
print('汇总（第 600 条）：V1 %s，V2 R_delta = %.4f，V3 %s'
      % ('PASS' if v1_ok else 'FAIL', r2, 'PASS' if v3_ok else 'FAIL'))
print('判读：%s' % verdict2)
print()
print('运行时间 %.1f 秒' % (time.time() - T_START))
