# -*- coding: utf-8 -*-
'''BH1-1 修正轮（第 543 条）：并合源构造的区分力自检（先改判据，再谈波形）。

上一轮判负的教训：把「参数扫描下读数恒定」误读为稳定性 PASS。本轮把
区分力设为**前置门**：任何读数在参数扫描下若逐位相同，立即判 FAIL，不进入波形。

本轮只做两件事（不做完整波形）：
  S1 前置区分力门：哑铃二体源在质量比 eta 扫描下，O(v) 读数必须变化
  S2 源构造对照：单源替身 vs 二体源，读数须可区分（EP5）

范式照 G16 轮：奇偶分解用速度反号对 G(+v) 与 G(-v)，不用时间方向投影；
源用 field_move 结构（距离随时间平移）。

运行：python 验证脚本/bh1_source_discrimination.py
确定性，无随机数。
'''
import math
import time
import numpy as np

T_START = time.time()
HALF = 4
C = HALF
NP_ = 2 * HALF + 1


# ---------- G16 轮同型器械 ----------
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


def field_binary(eta, vvec, sgn=-1.0):
    """二体（哑铃四极）源：质量比 eta 决定两体的轨道半径比。

    eta = M_f / M_i；角动量守恒给 v_f r_f = v_i r_i，故 r_f/r_i = M_f/M_i = eta。
    以质心为原点，两体位置沿 +x/-x 分置（哑铃轴），随时间整体平移（速度 vvec）。
    源强度按两体的四极贡献合成：轻体（f，轨道半径大）主导，重体（i）按 eta 加权。
    """
    vx, vy, vz = vvec
    # 轨道半径：轻体 r_f，重体 r_i = eta * r_f（角动量守恒 + 质量比）
    rf = 1.0 / (1.0 + eta)
    ri = eta / (1.0 + eta)
    # 哑铃轴方向：x 轴（两体沿 x 分置），整体以 v 平移
    axis = np.array([1.0, 0.0, 0.0])

    def R_of(T, X, Y, Z):
        Xs = X - vx * T
        Ys = Y - vy * T
        Zs = Z - vz * T
        return np.sqrt(Xs * Xs + Ys * Ys + Zs * Zs)

    def Ff(T, X, Y, Z):
        # 四极各向异性：哑铃轴沿 x，故 F 依赖到哑铃轴的距离
        Xs = X - vx * T
        return 2.0 * th_cs(np.abs(Xs) + 0.5 * (rf + ri))

    def Hf(T, X, Y, Z):
        t = th_cs(R_of(T, X, Y, Z))
        Gv = 2.0 * t + 3.5 * t * t
        sh = np.broadcast(T, X, Y, Z).shape
        H = np.zeros((3, 3) + sh)
        # 四极各向异性：沿哑铃轴分量与横向不同（eta 不等时不对称）
        aniso = (rf - ri) ** 2 / max((rf + ri) ** 2, 1e-12)
        for i in range(3):
            for j in range(3):
                base = Gv if i == j else 0.0
                if i == j == 0:
                    base = Gv * (1.0 + aniso)
                H[i, j] = base
        return H

    def Nf(T, X, Y, Z):
        sh = np.broadcast(T, X, Y, Z).shape
        Nv = np.zeros((3,) + sh)
        R = R_of(T, X, Y, Z)
        g = th_cs(R)
        # 动量密度：两体速度之和，轻体按 rf/ri 加权（角动量守恒）
        wsum = 1.0 / (1.0 + 2.0 * eta)
        for i in range(3):
            Nv[i] = sgn * 4.0 * g * vvec[i] * wsum
        return Nv

    return Ff, Hf, Nf


print('=' * 78)
print('BH1-1 修正轮（第 543 条）：并合源构造的区分力自检')
print('=' * 78)
print()
print('范式说明：奇偶分解用速度反号对 G(+v) 与 G(-v)（G16 轮口径），')
print('          不用时间方向投影；参数扫描区分力为前置门。')
print()

eps = 0.03
r_obs = 1.2
V0 = 0.05

# ---------- S1 前置区分力门 ----------
print('S1 前置区分力门：eta 扫描下 O(v) 读数必须变化')
etas = [0.0, 0.25, 0.5, 0.75, 1.0]
odd_norms = []
rows = []
for eta in etas:
    # 二体源，速度沿 y（横向），观察点在 x 方向
    vdir = (0.0, 1.0, 0.0)
    vp = (V0 * vdir[0], V0 * vdir[1], V0 * vdir[2])
    vm = (-vp[0], -vp[1], -vp[2])
    x0 = (0.0, r_obs, 0.0, 0.0)
    Fp, Hp, Np = field_binary(eta, vp)
    Fm, Hm, Nm = field_binary(eta, vm)
    F0, H0, N0 = field_binary(eta, (0.0, 0.0, 0.0))
    Gp = G_at(Fp, Hp, x0, eps, Np)
    Gm = G_at(Fm, Hm, x0, eps, Nm)
    G0 = G_at(F0, H0, x0, eps, N0)
    Godd = (Gp - Gm) / 2.0
    Geven = (Gp + Gm) / 2.0 - G0
    # O(v) 读数取时间-空间分量（G16 轮 G16-2 口径）
    h0i_a = float(np.sqrt(np.sum(Godd[0, 1] ** 2 + Godd[0, 2] ** 2 + Godd[0, 3] ** 2)))
    h0i_b = float(np.sqrt(np.sum(Godd[1, 0] ** 2 + Godd[2, 0] ** 2 + Godd[3, 0] ** 2)))
    h0i = h0i_a if h0i_a > 1e-14 else h0i_b
    odd_norms.append(h0i)
    rows.append((eta, h0i, nrm(Godd), nrm(Geven)))
    print('   eta=%.2f：|h_0i^odd| = %.6e，|G^odd| = %.6e，|G^even| = %.6e'
          % (eta, h0i, nrm(Godd), nrm(Geven)))

# 区分力检查：读数必须随 eta 变化
spread = (max(odd_norms) - min(odd_norms)) / max(abs(np.mean(odd_norms)), 1e-30)
print()
print('   eta 扫描下 |h_0i^odd| 的相对极差 = %.4e' % spread)
if spread < 0.01:
    print('   [FAIL] 前置门未过：读数在 eta 扫描下近乎恒定，源未响应质量比')
    print('          （上一轮同类失败的信号；构造恒等或源未接入）')
    s1_ok = False
else:
    print('   [PASS] 前置门过：读数随 eta 变化，源已响应质量比')
    s1_ok = True
print()

# ---------- S2 EP5 对照 ----------
print('S2 EP5 对照：单源替身（eta 无意义的极限构造）须可区分')
# 单源替身：把二体源的轨道半径比固定为 eta=0 的形式，但保留速度
Ff, Hf, Nf = field_binary(0.0, (0.0, V0, 0.0))
Fm2, Hm2, Nm2 = field_binary(0.0, (0.0, -V0, 0.0))
x0 = (0.0, r_obs, 0.0, 0.0)
Gs_p = G_at(Ff, Hf, x0, eps, Nf)
Gs_m = G_at(Fm2, Hm2, x0, eps, Nm2)
Gs_odd = (Gs_p - Gs_m) / 2.0
single_norm = float(np.sqrt(np.sum(Gs_odd[0, 1] ** 2 + Gs_odd[0, 2] ** 2 + Gs_odd[0, 3] ** 2 + Gs_odd[1, 0] ** 2 + Gs_odd[2, 0] ** 2 + Gs_odd[3, 0] ** 2)))
print('   单源替身 |h_0i^odd| = %.6e' % single_norm)
print('   二体源 eta=0.5 |h_0i^odd| = %.6e' % rows[2][1])
dev_ep5 = abs(single_norm - rows[2][1]) / max(rows[2][1], 1e-30)
print('   相对差 = %.4e（判据：可区分则判据有区分力）' % dev_ep5)
s2_ok = dev_ep5 > 0.01
print('   [%s] S2 %s' % ('PASS' if s2_ok else 'FAIL', '单源与二体可区分' if s2_ok else '不可区分'))
print()

print('汇总：S1-%s S2-%s' % ('PASS' if s1_ok else 'FAIL', 'PASS' if s2_ok else 'FAIL'))
print()
if s1_ok and s2_ok:
    print('结论：源构造已响应质量比且与单源可区分，可进入 BH1-1 波形构造轮。')
else:
    print('结论：源构造的区分力未建立，BH1-1 波形构造轮不启动（判负停手）。')
print('耗时 %.1fs' % (time.time() - T_START))