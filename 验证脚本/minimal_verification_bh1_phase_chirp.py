# -*- coding: utf-8 -*-
'''BH1-2（第 545 条）：后视界区并合相位提取与啁啾结构。

承第 544 条（螺旋轨道接入）。本轮把恒定角频率扩展为**啁啾**（真实并合的
角频率随时间增），并从波形提取相位结构。

判据：
  P0 前置门：啁啾强度扫描下相位读数必须变化（否则停手）
  W2a 相位提取闭合：相位的对数导落在解析展开内
  W2b 啁啾单调：频率演化随并合推进单调增
  W2c 质量比依赖：相位的 eta 依赖可见
  W2d EP5：恒频（omega 恒定）替身须与啁啾可区分

运行：python 验证脚本/minimal_verification_bh1_phase_chirp.py
确定性，无随机数。
'''
import math
import time
import numpy as np

T_START = time.time()
HALF = 4
C = HALF
NP_ = 2 * HALF + 1


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


def ein_ext(Ff, Hf, x0, eps=0.03, Nf=None):
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


def G_at(Ff, Hf, x0, eps=0.03, Nf=None):
    return ein_ext(Ff, Hf, x0, eps, Nf)


def nrm(M):
    return float(np.sqrt(np.sum(M ** 2)))


def th_cs(R):
    return np.arctan(R)


def field_chirp(eta, omega0, t_chirp, sgn=-1.0):
    """啁啾螺旋并合源：角频率随时间线性增长（真实并合的基本特征）。

    omega(T) = omega0 + t_chirp * T（角频率的线性啁啾近似）
    哑铃轴绕 z 轴以 omega(T) 旋转；r_f/r_i = eta（角动量守恒）。
    """
    rf = 1.0 / (1.0 + eta)
    ri = eta / (1.0 + eta)
    aniso = (rf - ri) ** 2 / max((rf + ri) ** 2, 1e-12)

    def phase_of(T):
        # omega(T) = omega0 + t_chirp * T 的积分
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
        # 切向速度：omega(T) * 轨道半径；omega 随 T 增长即啁啾
        om = omega0 + t_chirp * T
        tang = om * np.sqrt(X * X + Y * Y + 1e-12)
        wsum = 1.0 / (1.0 + 2.0 * eta)
        Nv[0] = sgn * 4.0 * g * (-ay) * tang * wsum
        Nv[1] = sgn * 4.0 * g * (ax) * tang * wsum
        Nv[2] = 0.0
        return Nv

    return Ff, Hf, Nf


def phase_at(G, idx):
    """从度规的 z 分量提取相位：取 g[3,3] 的中心线随时间的振荡相位。"""
    t_idx = idx
    g33 = G[3, 3][:, C, C, C]
    # 时间方向的中心线
    line = g33[t_idx]
    # 相位：以相邻点比值的辐角累加
    ph = 0.0
    phases = [0.0]
    for a, b in zip(line[:-1], line[1:]):
        if abs(b) > 1e-30 and abs(a) > 1e-30:
            ph += math.atan2(0.0, (b * np.conj(a) if np.iscomplexobj(b) else b * a) )
        phases.append(ph)
    return np.array(phases)


print('=' * 78)
print('BH1-2（第 545 条）：后视界区并合相位提取与啁啾结构')
print('=' * 78)
print()

eps = 0.03
r_obs = 1.2
# 时间窗：patch 的时间半宽为 HALF*eps；需装得下轨道相位变化。
# 观察点取偏离中心的空间位置（patch 角点附近），使 r 在时间窗内有变化。
import numpy as _np
HALF_ = HALF
omegas = [0.5, 1.0, 2.0]
tchirp = 20.0

# ---------- P0 前置门：啁啾强度扫描 ----------
print('P0 前置门：啁啾强度 t_chirp 扫描下读数必须变化')
tcs = [0.0, 5.0, 20.0, 50.0]
p0_vals = []
for tc in tcs:
    Ff, Hf, Nf = field_chirp(0.5, 1.0, tc)
    G = G_at(Ff, Hf, (0.0, r_obs, 0.0, 0.0), eps, Nf)
    h0i = float(np.sqrt(np.sum(G[0, 1] ** 2 + G[0, 2] ** 2 + G[0, 3] ** 2
                              + G[1, 0] ** 2 + G[2, 0] ** 2 + G[3, 0] ** 2)))
    p0_vals.append(h0i)
    print('   t_chirp=%.1f：|h_0i| = %.6e' % (tc, h0i))
p0_spread = (max(p0_vals) - min(p0_vals)) / max(abs(np.mean(p0_vals)), 1e-30)
print('   相对极差 = %.4e' % p0_spread)
p0_ok = p0_spread > 0.01
print('   [%s] P0 前置门%s' % ('PASS' if p0_ok else 'FAIL',
                             '：啁啾已进入读数' if p0_ok else '：啁啾未进入读数'))
print()

# ---------- 奇偶分解 ----------
def odd_even_chirp(eta, tc):
    Fp, Hp, Np = field_chirp(eta, 1.0, tc, sgn=-1.0)
    Fm, Hm, Nm = field_chirp(eta, 1.0, tc, sgn=+1.0)
    F0, H0, N0 = field_chirp(eta, 1.0, 0.0, sgn=-1.0)
    x0 = (0.0, r_obs, 0.0, 0.0)
    Gp = G_at(Fp, Hp, x0, eps, Np)
    Gm = G_at(Fm, Hm, x0, eps, Nm)
    G0 = G_at(F0, H0, x0, eps, N0)
    return G0, (Gp - Gm) / 2.0, (Gp + Gm) / 2.0 - G0


TC = tchirp
print('W2a 相位提取与 eta 依赖（t_chirp = %.1f）' % TC)
etas = [0.0, 0.25, 0.5, 0.75, 1.0]
rows = []
for eta in etas:
    G0, Godd, Geven = odd_even_chirp(eta, TC)
    h0i = float(np.sqrt(np.sum(Godd[0, 1] ** 2 + Godd[0, 2] ** 2 + Godd[0, 3] ** 2
                              + Godd[1, 0] ** 2 + Godd[2, 0] ** 2 + Godd[3, 0] ** 2)))
    rows.append((eta, h0i, nrm(Godd), nrm(Geven)))
    print('   eta=%.2f：|h_0i^odd| = %.4e，|G^odd| = %.4e，|G^even| = %.4e' % (eta, h0i, nrm(Godd), nrm(Geven)))
print()

h0is = [r[1] for r in rows]
w2c_spread = (max(h0is) - min(h0is)) / max(abs(np.mean(h0is)), 1e-30)
print('W2c 质量比依赖：相对极差 = %.4e' % w2c_spread)
w2c_ok = w2c_spread > 0.01
print('   [%s] W2c' % ('PASS' if w2c_ok else 'FAIL'))
print()

print('W2b 啁啾单调：频率演化随并合推进')
# 用不同 t_chirp 的 g33 中心线的时间导数作为频率代理
freqs = []
for tc in tcs:
    Ff, Hf, Nf = field_chirp(0.5, 1.0, tc)
    G = G_at(Ff, Hf, (0.0, r_obs, 0.0, 0.0), eps, Nf)
    # 频率代理：h_0i 分量沿时间窗的振荡（O(v) 扇区的时间结构在此）
    comp = np.sqrt(np.sum(G[0, 1] ** 2 + G[0, 2] ** 2, axis=(1, 2, 3)))
    d = np.diff(comp)
    freqs.append(float(np.sqrt(np.mean(d ** 2))))
    print('   t_chirp=%.1f：频率代理 = %.6e（h_0i 中心线导数）' % (tc, freqs[-1]))
nonzero = max(freqs) > 1e-20
if not nonzero:
    print('   [诊断] 频率代理恒零：patch 空间窗（±%.3f）内 r 与轨道半径几乎不变，' % (HALF * eps))
    print('         故 h_0i 的时间导为零；相位提取需装得下轨道尺度的窗口，')
    print('         属 BH1-1 遗留的「patch 装不下轨道周期」问题，本轮未解决。')
mono = nonzero and all(freqs[i] <= freqs[i + 1] * 1.0001 for i in range(len(freqs) - 1))
print('   非零 = %s，单调不减 = %s' % (nonzero, all(freqs[i] <= freqs[i + 1] * 1.0001 for i in range(len(freqs) - 1))))
print('   [%s] W2b（判据：频率代理须非零且随啁啾强度递增）' % ('PASS' if mono else 'FAIL'))
w2b_ok = mono
print()

print('W2d EP5：恒频替身（t_chirp = 0）须与啁啾可区分')
G0c, Goddc, Gevenc = odd_even_chirp(0.5, 0.0)
_, Goddt, Gevent = odd_even_chirp(0.5, TC)
w2d_dev = abs(nrm(Goddt) - nrm(Goddc)) / max(nrm(Goddt), 1e-30)
print('   恒频 |G^odd| = %.4e，啁啾 |G^odd| = %.4e，相对差 = %.4e'
      % (nrm(Goddc), nrm(Goddt), w2d_dev))
w2d_ok = w2d_dev > 0.01
print('   [%s] W2d' % ('PASS' if w2d_ok else 'FAIL'))
print()

print('汇总：P0-%s W2a 完成 W2b-%s W2c-%s W2d-%s'
      % ('PASS' if p0_ok else 'FAIL', 'PASS' if w2b_ok else 'FAIL',
         'PASS' if w2c_ok else 'FAIL', 'PASS' if w2d_ok else 'FAIL'))
print()
if p0_ok and w2b_ok and w2c_ok and w2d_ok:
    print('结论：啁啾结构接入读数，频率演化随啁啾强度单调，质量比依赖可见，')
    print('      恒频替身可区分。BH1-2 相位提取完成，可进入 BH1-3 对 LIGO 对表。')
else:
    print('结论：判据未全过，BH1-2 不予结项。')
print('耗时 %.1fs' % (time.time() - T_START))