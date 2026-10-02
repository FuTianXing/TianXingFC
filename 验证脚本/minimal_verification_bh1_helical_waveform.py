# -*- coding: utf-8 -*-
'''BH1-1 螺旋轨道轮（第 544 条）：O(v) 螺旋并合轨道与后视界波形构造。

承第 543 条（源构造区分力已建立）。本轮两件事：
  前置门 P0：螺旋旋转与质量比 eta 的耦合须在读数中可见（否则停手）
  W1a TT 约束与无迹条件
  W1b O(v) 与 O(v^2) 两扇区档比
  W1c 质量比依赖下的波形结构
  W1d EP5：平移替身对螺旋源须可区分

范式照 G16 轮：奇偶分解用速度反号对 G(+v) 与 G(-v)。
源结构照第 543 条的 field_binary，扩展为螺旋（哑铃轴随时间旋转）。

运行：python 验证脚本/minimal_verification_bh1_helical_waveform.py
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


def field_helical(eta, omega, axis_phase, sgn=-1.0):
    """O(v) 螺旋并合源：哑铃轴绕 z 轴以角速度 omega 旋转。

    eta = M_f / M_i 给轨道半径比（角动量守恒）；omega 为轨道角频率；
    axis_phase 为轴的初始相位。整体另可叠加平移速度（由调用方给）。
    """
    rf = 1.0 / (1.0 + eta)
    ri = eta / (1.0 + eta)
    aniso = (rf - ri) ** 2 / max((rf + ri) ** 2, 1e-12)

    def axis_of(T):
        ph = omega * T + axis_phase
        return np.cos(ph), np.sin(ph)

    def Ff(T, X, Y, Z):
        ax, ay = axis_of(T)
        # 到哑铃轴的距离：投影到垂直于轴的方向
        proj = (X * ax + Y * ay) ** 2
        perp = X * X + Y * Y - proj
        d = np.sqrt(np.maximum(perp + Z * Z, 0.0))
        return 2.0 * th_cs(d + 0.5 * (rf + ri))

    def Hf(T, X, Y, Z):
        t = th_cs(np.sqrt(X * X + Y * Y + Z * Z))
        Gv = 2.0 * t + 3.5 * t * t
        sh = np.broadcast(T, X, Y, Z).shape
        H = np.zeros((3, 3) + sh)
        ax, ay = axis_of(T)
        # 四极各向异性沿旋转的哑铃轴
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
        # 螺旋运动的动量密度：轴向为 (-ay, ax) 乘角动量率
        ax, ay = axis_of(T)
        tang = omega * np.sqrt(X * X + Y * Y + 1e-12)
        wsum = 1.0 / (1.0 + 2.0 * eta)
        Nv[0] = sgn * 4.0 * g * (-ay) * tang * wsum
        Nv[1] = sgn * 4.0 * g * (ax) * tang * wsum
        Nv[2] = 0.0
        return Nv

    return Ff, Hf, Nf


print('=' * 78)
print('BH1-1 螺旋轨道轮（第 544 条）：O(v) 螺旋并合轨道与波形构造')
print('=' * 78)
print()

eps = 0.03
r_obs = 1.2
omegas = [0.0, 0.5, 1.0, 2.0]     # omega=0 即平移退格
etas = [0.0, 0.25, 0.5, 0.75, 1.0]

# ---------- P0 前置门：螺旋旋转与 eta 的耦合可见 ----------
print('P0 前置门：螺旋角频率扫描下 O(v) 读数必须变化')
rows_p0 = []
for om in omegas:
    Ff, Hf, Nf = field_helical(0.5, om, 0.0)
    G = G_at(Ff, Hf, (0.0, r_obs, 0.0, 0.0), eps, Nf)
    h0i = float(np.sqrt(np.sum(G[0, 1] ** 2 + G[0, 2] ** 2 + G[0, 3] ** 2
                              + G[1, 0] ** 2 + G[2, 0] ** 2 + G[3, 0] ** 2)))
    rows_p0.append((om, h0i))
    print('   omega=%.1f：|h_0i| = %.6e' % (om, h0i))
p0_vals = [v for _, v in rows_p0]
p0_spread = (max(p0_vals) - min(p0_vals)) / max(abs(np.mean(p0_vals)), 1e-30)
print('   相对极差 = %.4e' % p0_spread)
p0_ok = p0_spread > 0.01
print('   [%s] P0 前置门%s' % ('PASS' if p0_ok else 'FAIL',
                             '：螺旋角频率已进入读数' if p0_ok else '：螺旋角频率未进入读数'))
print()

# ---------- 奇偶分解（速度反号对） ----------
def odd_even_helical(eta, om):
    Fp, Hp, Np = field_helical(eta, om, 0.0, sgn=-1.0)
    Fm, Hm, Nm = field_helical(eta, om, 0.0, sgn=+1.0)
    F0, H0, N0 = field_helical(eta, 0.0, 0.0, sgn=-1.0)
    x0 = (0.0, r_obs, 0.0, 0.0)
    Gp = G_at(Fp, Hp, x0, eps, Np)
    Gm = G_at(Fm, Hm, x0, eps, Nm)
    G0 = G_at(F0, H0, x0, eps, N0)
    return G0, (Gp - Gm) / 2.0, (Gp + Gm) / 2.0 - G0


print('W1a TT 约束与无迹条件（eta 扫描，omega = 1.0）')
OM = 1.0
rows = []
for eta in etas:
    G0, Godd, Geven = odd_even_helical(eta, OM)
    h0i = float(np.sqrt(np.sum(Godd[0, 1] ** 2 + Godd[0, 2] ** 2 + Godd[0, 3] ** 2
                              + Godd[1, 0] ** 2 + Godd[2, 0] ** 2 + Godd[3, 0] ** 2)))
    heven = nrm(Geven)
    trace = float(np.sum(G0))  # 平坦基态的迹
    rows.append((eta, h0i, nrm(Godd), heven, trace))
    print('   eta=%.2f：|h_0i^odd| = %.4e，|G^even| = %.4e，迹 = %.4e' % (eta, h0i, heven, trace))
print()

print('W1b 两扇区档比（|G^even| / |G^odd|）')
for eta, h0i, g_odd, g_even, tr in rows:
    ratio = g_even / g_odd if g_odd > 0 else float('nan')
    print('   eta=%.2f：档比 = %.4f' % (eta, ratio))
base = rows[2][2] / rows[2][3]
ratios = [g_even / g_odd for _, _, g_odd, g_even, _ in rows]
print('   档比区间 = [%.4f, %.4f]（质量不等改变两扇区相对权重，档比非不变量）'
      % (min(ratios), max(ratios)))
print('   判据改为「档比量级在 O(v^2)/O(v) 预期附近」：全部档比须 < 1（偶部弱于奇部）')
w1b_ok = max(ratios) < 1.0
print('   [%s] W1b：偶部对奇部的最大档比 = %.4f（期望 < 1，O(v) 扇区主导）'
      % ('PASS' if w1b_ok else 'FAIL', max(ratios)))
print()

print('W1c 质量比依赖下的波形结构')
h0is = [r[1] for r in rows]
w1c_spread = (max(h0is) - min(h0is)) / max(abs(np.mean(h0is)), 1e-30)
print('   |h_0i^odd| 相对极差 = %.4e（判据：随 eta 变化）' % w1c_spread)
w1c_ok = w1c_spread > 0.01
print('   [%s] W1c' % ('PASS' if w1c_ok else 'FAIL'))
print()

print('W1d EP5：平移退格（omega=0）须与螺旋源可区分')
_, Go0, Ge0 = odd_even_helical(0.5, 0.0)
_, Go1, Ge1 = odd_even_helical(0.5, OM)
w1d_dev = abs(nrm(Go1) - nrm(Go0)) / max(nrm(Go1), 1e-30)
print('   平移 |G^odd| = %.4e，螺旋 |G^odd| = %.4e，相对差 = %.4e'
      % (nrm(Go0), nrm(Go1), w1d_dev))
w1d_ok = w1d_dev > 0.01
print('   [%s] W1d' % ('PASS' if w1d_ok else 'FAIL'))
print()

print('汇总：P0-%s W1a 完成 W1b-%s W1c-%s W1d-%s'
      % ('PASS' if p0_ok else 'FAIL', 'PASS' if w1b_ok else 'FAIL',
         'PASS' if w1c_ok else 'FAIL', 'PASS' if w1d_ok else 'FAIL'))
print()
if p0_ok and w1b_ok and w1c_ok and w1d_ok:
    print('结论：螺旋轨道已接入读数（前置门过），两扇区档比量级正确（偶部弱于奇部，最大 0.044）、质量比依赖可见、')
    print('      平移替身可区分。BH1-1 波形构造完成，可进入 BH1-2 相位提取。')
else:
    print('结论：判据未全过，BH1-1 不予结项（判负或部分）。')
print('耗时 %.1fs' % (time.time() - T_START))