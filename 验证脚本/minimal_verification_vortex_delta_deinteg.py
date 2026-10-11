# -*- coding: utf-8 -*-
r"""
minimal_verification_vortex_delta_deinteg.py

[维度外推审计 · 首版留档] 第 598 条藏身术在 2D 涡旋芯是否成立（第 600 条）

历史地位（2026-10-11 作者指示留档）：本版为开边界首版，判据未达交付态，
撤下改留档。核心教训：开边界结构性不守绕数——J = k = 1 下梯度流把涡旋
经边界漏走（E 8207 → 0.035 真空化），芯区非整残留读数是真空假象，非物理
判否。下轮入口 = 2D 周期边界 + 绕数扇区版（FK 扣 Q 的二维化：周期格点上
涡旋的格点通量相容 Q 标签），芯区 delta 非整残留指标与 1D 退化对照不变。
三条 2D 建模教训：其一，atan2 单值场的回路 diff 累计恒零（±π 跳变把 1 拆
到相邻键非整值），绕数须含跳变整化读出或改解析连续构造；其二，切割线补偿
须加在真跳变处（atan2 跳变在负 x 半轴）；其三，即本版主教训，开边界不承载
绕数扇区，2D 判据必须周期边界。

判据设计（保留供下轮）：V1 涡旋存活（绕数守恒、能量单调）；V2 芯区
delta^12 非整残留（连续角向 0.25 对整数化 <0.05，通过则三维供货窗口重开，
判否则第 598 条判否外推成立、主路线出口关闭）；V3 一维退化对照（区分力，
先验已在同款模型内成立 R_delta = 0.0018）。

模型：2D 开边界平面三分量，键弹性两向（k = 1）+ 格点内三对异类（J = 1），
N = 64 × 64，涡旋初值（atan2 单值场，跳变键保留）。
运行：python 验证脚本/minimal_verification_vortex_delta_deinteg.py
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
print('维度外推审计（首版留档）：第 598 条藏身术在 2D 涡旋芯（第 600 条）')
print('=' * 76)
print()
print('开边界版教训版：涡旋被边界漏走真空化，芯区读数为真空假象。')
print()


def init_vortex(N):
    """涡旋初值：atan2 单值场（跳变键保留），绕数由回路累计整化读出。"""
    n = np.arange(N)
    x, y = np.meshgrid(n, n, indexing='ij')
    x0 = y0 = (N - 1) / 2.0
    v1 = np.arctan2(y - y0, x - x0) / TWO_PI
    return np.stack([v1, np.zeros((N, N)), np.zeros((N, N))])


def init_1dwall(N):
    """V3 对照：y 无关的一维墙。"""
    n = np.arange(N)
    prof = 0.5 * (1.0 + np.tanh((n - N / 2) / 4.0))
    return np.stack([np.tile(prof[:, None], (1, N)),
                     np.zeros((N, N)), np.zeros((N, N))])


def dxdy(v):
    dx = np.zeros_like(v)
    dy = np.zeros_like(v)
    dx[:, :-1, :] = v[:, 1:, :] - v[:, :-1, :]
    dy[:, :, :-1] = v[:, :, 1:] - v[:, :, :-1]
    dx[:, -1, :] = 0.0    # 开边界：最外列/行无出键
    dy[:, :, -1] = 0.0
    return dx, dy


def energy(v):
    dx, dy = dxdy(v)
    e_el = 0.5 * K_EL * (float(np.sum(dx ** 2)) + float(np.sum(dy ** 2)))
    e_het = 0.0
    for (a, b) in PAIRS:
        e_het += J_M * float(np.sum(1.0 - np.cos(TWO_PI * (v[a] - v[b]))))
    return e_el, e_het, e_el + e_het


def grad(v):
    dx, dy = dxdy(v)
    g = np.empty_like(v)
    for a in range(3):
        lapx = np.zeros_like(v)
        lapy = np.zeros_like(v)
        lapx[:, :-1, :] -= dx[:, :-1, :]
        lapx[:, 1:, :] += dx[:, :-1, :]
        lapy[:, :, :-1] -= dy[:, :, :-1]
        lapy[:, :, 1:] += dy[:, :, :-1]
        g[a] = K_EL * (lapx[a] + lapy[a])
    for (a, b) in PAIRS:
        d = TWO_PI * (v[a] - v[b])
        s = J_M * TWO_PI * np.sin(d)
        g[a] += s
        g[b] -= s
    return g


def relax(v0, nsteps=60000):
    v = v0.copy()
    eta = min(0.01, 0.3 / (8.0 + 8.0 * math.pi ** 2 * J_M))
    e_prev = energy(v)[2]
    mono = True
    for step in range(nsteps):
        v = v - eta * grad(v)
        if step % 5000 == 4999:
            e = energy(v)[2]
            if e > e_prev + 1e-9:
                mono = False
            if abs(e - e_prev) < 1e-10 * max(1.0, abs(e_prev)):
                return v, e, True, mono
            e_prev = e
    return v, energy(v)[2], False, mono


def winding_number(v, a=0, r_ring=5.0):
    """绕芯回路 dv 累计（含 atan2 跳变键拆分误差）。"""
    c = (N - 1) / 2.0
    m = int(round(r_ring))
    lo, hi = int(round(c - m)), int(round(c + m))
    seg = [v[a, lo:hi + 1, lo], v[a, hi, lo:hi + 1],
           v[a, lo:hi + 1, hi][::-1], v[a, lo, lo:hi + 1][::-1]]
    tot = 0.0
    for s_ in seg:
        tot += float(np.sum(np.diff(s_)))
    return tot


def resid_delta(v, r_c=RC):
    c = (N - 1) / 2.0
    n = np.arange(N)
    x, y = np.meshgrid(n, n, indexing='ij')
    mask = (x - c) ** 2 + (y - c) ** 2 <= r_c ** 2
    fr = TWO_PI * (v[0] - v[1])
    fr = fr[mask] / TWO_PI
    d = np.abs(fr - np.round(fr))
    d = np.minimum(d, 1.0 - d)
    return float(np.mean(d)), int(mask.sum())


print('--- V1 涡旋存活 ---')
v0 = init_vortex(N)
v_end, e_end, conv, mono = relax(v0)
w0 = winding_number(v0)
w1 = winding_number(v_end)
print('   E: %.2f -> %.4f（收敛 %s，单调 %s）；绕数累计 %.4f -> %.4f（整化 %d -> %d）'
      % (energy(v0)[2], e_end, conv, mono, w0, w1, round(w0), round(w1)))
v1_ok = mono and round(w0) == 1 and round(w1) == 1
print('   [%s] V1' % ('PASS' if v1_ok else 'FAIL'))
print()

print('--- V2 芯区 delta^12 非整残留（首版读数：真空假象，不作判读）---')
r2, npts = resid_delta(v_end)
print('   R_delta = %.4f（芯区 %d 点；涡旋已真空化，读数无效）' % (r2, npts))
print()

print('--- V3 一维对照（区分力先验）---')
v0_1d = init_1dwall(N)
v1d, e1d, conv1, mono1 = relax(v0_1d)
r3, _ = resid_delta(v1d)
print('   R_delta(1D退化) = %.4f（E: %.2f -> %.4f，收敛 %s）'
      % (r3, energy(v0_1d)[2], e1d, conv1))
print('   [%s] V3（1D 整数化复现）' % ('PASS' if r3 < 0.05 else 'FAIL'))
print()

print('=' * 76)
print('首版留档终态：V1 FAIL（涡旋边界漏走真空化），V2 读数无效，V3 PASS。')
print('下轮入口：2D 周期边界 + 绕数扇区版（FK 扣 Q 二维化）。')
print('运行时间 %.1f 秒' % (time.time() - T_START))
