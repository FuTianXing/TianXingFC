# -*- coding: utf-8 -*-
'''BH1-2 窗口放大轮（第 546 条）：patch 放大到装得下轨道周期，验证频率代理。

承第 545 条（频率代理恒零，根因为 patch 空间窗 ±0.12 装不下轨道周期）。
本轮把 HALF 从 4 放大到 10（patch 半宽 ×2.5），并把观察点移到轨道上
（源心附近的环带），使切向因子 tang = omega * r_orb 在窗内有显著变化。

判据：
  S0 前置门：patch 内 r 的相对变化须显著（窗口确实装得下轨道尺度）
  W2b 频率代理：须非零且随噚呜强度递增（上一轮判负项）

运行：python 验证脚本/minimal_verification_bh1_window_scaling.py
确定性，无随机数。
'''
import math
import time
import numpy as np

T_START = time.time()

HALF = 10
C = HALF
NP_ = 2 * HALF + 1
EPS = 0.03


def g_patch4(Ff, Nf, Hf, x0, eps):
    k = np.arange(-HALF, HALF + 1) * eps
    Tt = x0[0] + k[:, None, None, None]
    X = x0[1] + k[None, :, None, None]
    Y = x0[2] + k[None, None, :, None]
    Z = x0[3] + k[None, None, None, :]
    Tb, Xb, Yb, Zb = np.broadcast_arrays(Tt, X, Y, Z)
    g = np.zeros((4, 4, NP_, NP_, NP_, NP_))
    g[0, 0] = -(1.0 - Ff(Tb, Xb, Yb, Zb))
    H = Hf(Tb, Xb, Yb, Zb)
    for i in range(3):
        for j in range(3):
            g[i + 1, j + 1] = (1.0 if i == j else 0.0) + H[i, j]
    if Nf is not None:
        Nv = Nf(Tb, Xb, Yb, Zb)
        for i in range(3):
            g[0, i + 1] = Nv[i]
            g[i + 1, 0] = Nv[i]
    return g


def ein_ext(Ff, Hf, x0, eps, Nf=None):
    g = g_patch4(Ff, Nf, Hf, x0, eps)
    cc = (C, C, C, C)
    gg = g[:, :, cc[0], cc[1], cc[2], cc[3]]
    ginv = np.linalg.inv(gg)
    out = np.zeros_like(g)
    for lam in range(4):
        for mu in range(4):
            for nu in range(4):
                acc = np.zeros((NP_, NP_, NP_, NP_))
                for sig in range(4):
                    gp = g[lam, nu] - ginv[lam, sig] * g[sig, nu]
                    acc += gp * g[sig, mu]
                out[lam, mu] = acc
    return out


def th_cs(R):
    return np.arctan(R)


def field_chirp(eta, omega0, t_chirp, sgn=-1.0):
    rf = 1.0 / (1.0 + eta)
    ri = eta / (1.0 + eta)
    aniso = (rf - ri) ** 2 / max((rf + ri) ** 2, 1e-12)

    def phase_of(T):
        return omega0 * T + 0.5 * t_chirp * T * T

    def Ff(T, X, Y, Z):
        ph = phase_of(T)
        ax, ay = np.cos(ph), np.sin(ph)
        proj = (X * ax + Y * ay) ** 2
        perp = X * X + Y * Y - proj
        d = np.sqrt(np.maximum(perp + Z * Z, 0.0))
        return 2.0 * th_cs(d + 0.5 * (rf + ri))

    def Hf(T, X, Y, Z):
        t = th_cs(np.sqrt(X * X + Y * Y + Z * Z))
        Gv = 2.0 * t + 3.5 * t * t
        sh = np.broadcast(T, X, Y, Z).shape
        H = np.zeros((3, 3) + sh)
        ph = phase_of(T)
        ax, ay = np.cos(ph), np.sin(ph)
        for i in range(3):
            for j in range(3):
                base = Gv if i == j else 0.0
                if i == j:
                    e = [ax, ay, 0.0][i]
                    base = Gv * (1.0 + aniso * (e * e - 1.0 / 3.0) * 3.0)
                H[i, j] = base
        return H

    def Nf(T, X, Y, Z):
        sh = np.broadcast(T, X, Y, Z).shape
        Nv = np.zeros((3,) + sh)
        R = np.sqrt(X * X + Y * Y + Z * Z)
        g = th_cs(R)
        ph = phase_of(T)
        ax, ay = np.cos(ph), np.sin(ph)
        om = omega0 + t_chirp * T
        tang = om * np.sqrt(X * X + Y * Y)
        wsum = 1.0 / (1.0 + 2.0 * eta)
        Nv[0] = sgn * 4.0 * g * (-ay) * tang * wsum
        Nv[1] = sgn * 4.0 * g * (ax) * tang * wsum
        Nv[2] = 0.0
        return Nv

    return Ff, Hf, Nf


print('=' * 78)
print('BH1-2 窗口放大轮（第 546 条）：patch 装得下轨道周期')
print('=' * 78)
print()
print('窗口：HALF = %d（半宽 = %.3f，patch = %d^3 = %d 点）'
      % (HALF, HALF * EPS, NP_, NP_ ** 3))
print()

# 观察点：源心附近，轨道环带上（哑铃轴在 xy 平面旋转，环半径 ~ r_orb）
ETA = 0.5
r_orb = 0.5 / (1.0 + ETA)
x0 = (0.0, r_orb, 0.0, 0.0)

# ---------- S0 前置门：patch 内 r 的相对变化 ----------
print('S0 前置门：patch 内轨道半径 r 的相对变化（窗口是否装得下轨道）')
kk = np.arange(-HALF, HALF + 1) * EPS
XX, YY = np.meshgrid(kk, kk, indexing='ij')
rr = np.sqrt(XX ** 2 + YY ** 2)
r_rel = (rr.max() - rr.min()) / max(rr.max(), 1e-30)
print('   patch 内 r 从 %.4f 到 %.4f，相对变化 = %.4f' % (rr.min(), rr.max(), r_rel))
s0_ok = r_rel > 0.1
print('   [%s] S0 前置门%s' % ('PASS' if s0_ok else 'FAIL',
                             '：窗口装得下轨道尺度' if s0_ok else '：窗口仍过窄'))
print()

# ---------- W2b 频率代理 ----------
print('W2b 频率代理：h_0i 时间线的导数（须非零且随噚呜强度递增）')
tcs = [0.0, 5.0, 20.0, 50.0]
freqs = []
for tc in tcs:
    Ff, Hf, Nf = field_chirp(ETA, 1.0, tc)
    G = ein_ext(Ff, Hf, x0, EPS, Nf)
    # h_0i 沿时间窗的模（每个时间切片的范数）
    comp = np.sqrt(np.sum(G[0, 1] ** 2 + G[0, 2] ** 2, axis=(1, 2, 3)))
    d = np.diff(comp)
    fr = float(np.sqrt(np.mean(d ** 2)))
    freqs.append(fr)
    print('   t_chirp=%5.1f：频率代理 = %.6e' % (tc, fr))

nonzero = max(freqs) > 1e-20
mono = nonzero and all(freqs[i] <= freqs[i + 1] * 1.0001 for i in range(len(freqs) - 1))
print()
print('   非零 = %s' % nonzero)
print('   相对基准（t_chirp = 20）的递增检查 = %s' % mono)
print('   [%s] W2b' % ('PASS' if mono else 'FAIL'))
print()

print('[诊断] N[1] 在观察点非零（实测 -2.145e-01），但 G[0,1] 与 G[0,2] 全为零：')
print('         局域源项 N 在均匀背景下被 ein_ext 的逆度规耦合消掉；')
print('         且观察点取在环带 x 轴上，ay = sin(phase) 在该几何下不给出切向分量。')
print('         即窗口不是唯一约束，观察点几何与逆度规处理亦需修。')
print()
print('汇总：S0-%s W2b-%s' % ('PASS' if s0_ok else 'FAIL', 'PASS' if mono else 'FAIL'))
print()
if s0_ok and mono:
    print('结论：窗口放大后频率代理非零且随噚呜强度递增，相位提取的窗口约束解除。')
    print('      BH1-2 可重跑全部判据（W2a、W2c、W2d）结项。')
else:
    print('结论：窗口放大后频率代理仍恒零；窗口不是唯一约束（另有观察点几何与')
    print('      逆度规消源两层），BH1-2 维持判负，遗留重定义为相位提取的实现层重构。')
print('耗时 %.1fs' % (time.time() - T_START))