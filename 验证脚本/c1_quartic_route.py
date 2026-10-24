# -*- coding: utf-8 -*-
r"""
c1_quartic_route.py

[C1 路线甲筛] 四次项（Hopf 后备）的可行性筛选：Derrick 标度、驻留与落窗
（判据 H-1 至 H-4）

背景：格点路线与路线乙（软体相 + 局域凹口）在三维均判负，失效模式是面积律
（壁能量正比面积，塌缩无阻）。路线甲在模型里加梯度四次项：
    E = sum_edges [ (1/2) k (du_e - 1)^2 + g (du_e - 1)^4 ] + sum_i K (1 - cos 2 pi u_i)
四次项是 C1-4a 裁决登记的两条实现形式中的「四次项（Hopf 路线）」，此前未试。

Derrick 标度（三维球泡壁，半径与壁厚同乘 lambda；高度固定）：
    E_2 = sum (dh)^2 正比 lambda（面积 lambda^2 除壁厚 lambda）
    E_4 = sum (dh)^4 正比 lambda^-1（面积 lambda^2 除壁厚^3 lambda^3）
    故 E(lambda) = A lambda + B lambda^-1，驻点 lambda* = sqrt(B/A) 存在。

判据（先写期望，再算）
H-1 标度指数：球泡初态按 lambda 缩放后 E_2、E_4 的幂次（期望 +1 与 -1）
H-2 驻留：K_soft = 0.02（落窗内）下扫 g，fate_of 口径判定环是否驻留
H-3 落窗核对：驻留态的环壁宽度 l*（弹性超量密度二阶矩）是否 >= 2
H-4 一致性：g = 0 应回到路线乙判负（collapse）

运行：python 验证脚本/c1_quartic_route.py（确定性，约 60 秒）
"""
import time
import numpy as np

T_START = time.time()
k = 1.0
N = 32
ETA = 0.01
STEPS = 2400

IX, IY, IZ = np.indices((N, N, N))
u0 = (IX + IY + IZ).astype(float)
CX = CY = CZ = N // 2
XC, YC, ZC = IX - CX, IY - CY, IZ - CZ
RR = np.sqrt(XC * XC + YC * YC + ZC * ZC)


def bubble(Rc, w, amp=1.0):
    return amp * 0.5 * (1.0 - np.tanh((RR - Rc) / w))


def shift(u):
    return u - u0


def edge_parts(u):
    """返回三个轴上的边缘差 dh_e = du_e - 1（背景差为 1，故即位移差）。"""
    return [np.diff(u, axis=ax) - 1.0 for ax in range(3)]


def energies(u, Kp, gq):
    E2 = E4 = 0.0
    for dh in edge_parts(u):
        E2 += 0.5 * k * float(np.sum(dh ** 2))
        E4 += gq * float(np.sum(dh ** 4))
    EK = Kp * float(np.sum(1.0 - np.cos(2.0 * np.pi * u)))
    return E2, E4, EK


def grad(u, Kp, gq):
    p = np.pad(u, 1, mode="edge")
    s = (p[2:, 1:-1, 1:-1] + p[:-2, 1:-1, 1:-1] + p[1:-1, 2:, 1:-1]
         + p[1:-1, :-2, 1:-1] + p[1:-1, 1:-1, 2:] + p[1:-1, 1:-1, :-2])
    g_out = k * (6.0 * u - s)
    if gq != 0.0:
        for ax in range(3):
            dh = np.diff(u, axis=ax) - 1.0
            f = 4.0 * gq * dh ** 3           # dE4/d(dh)
            # 正确装配：对边缘两端分别加减
            if ax == 0:
                g_out[:-1] -= f
                g_out[1:] += f
            elif ax == 1:
                g_out[:, :-1] -= f
                g_out[:, 1:] += f
            else:
                g_out[:, :, :-1] -= f
                g_out[:, :, 1:] += f
    g_out += 2.0 * np.pi * Kp * np.sin(2.0 * np.pi * u)
    return g_out


def pin(u):
    u[0, :, :] = u0[0, :, :]; u[-1, :, :] = u0[-1, :, :]
    u[:, 0, :] = u0[:, 0, :]; u[:, -1, :] = u0[:, -1, :]
    u[:, :, 0] = u0[:, :, 0]; u[:, :, -1] = u0[:, :, -1]


def run_gf(u_init, Kp, gq, steps=STEPS):
    u = u_init.copy()
    tr = []
    for t in range(steps):
        u = u - ETA * grad(u, Kp, gq)
        pin(u)
        if (t % 200 == 0) or (t == steps - 1):
            h = shift(u)
            tr.append((float(np.sum(abs(h) > 0.5)), float(h.max())))
    return u, tr


def radius_of(u):
    h = shift(u)
    wgt = np.abs(h)
    if wgt.sum() <= 1e-9:
        return float("nan")
    c0 = (wgt * RR).sum() / wgt.sum()
    r2 = (wgt * (RR - c0) ** 2).sum() / wgt.sum()
    return float(np.sqrt(r2))


def width_wall(u, Rc, half=5.0):
    dens = np.zeros((N, N, N))
    for dh in edge_parts(u):
        dd = np.zeros((N, N, N))
        if dh.shape[0] == N - 1:
            dd[:-1] = dh
        elif dh.shape[1] == N - 1:
            dd[:, :-1] = dh
        else:
            dd[:, :, :-1] = dh
        dens += 0.5 * dd ** 2
    m = (np.abs(RR - Rc) <= half)
    wgt = dens[m]
    if wgt.sum() <= 0:
        return float("nan")
    rho = RR[m]
    c0 = (wgt * rho).sum() / wgt.sum()
    return 2.0 * float(np.sqrt((wgt * (rho - c0) ** 2).sum() / wgt.sum()))


print("=" * 78)
print("C1 路线甲筛：四次项的 Derrick 标度、驻留与落窗（H-1 至 H-4）")
print("=" * 78)
RC0, W0 = 4.0, 2.0
u_init = u0 + bubble(RC0, W0)
print("球泡初态（整格位移，落在势的极小）：Rc = %.1f，w = %.1f，位移峰值 = %.3f，位移大于 0.5 的格点数 = %d"
      % (RC0, W0, shift(u_init).max(), int(np.sum(np.abs(shift(u_init)) > 0.5))))

# H-1 标度指数
print()
print("[H-1] Derrick 标度指数（半径与壁厚同乘 lambda，高度固定）")
base = None
for lam in (0.8, 1.0, 1.25, 1.6):
    us = u0 + bubble(RC0 * lam, W0 * lam)
    E2, E4, _ = energies(us, 0.0, 1.0)
    if base is None:
        base = (E2, E4)
    print("     lambda = %.2f：E_2 = %9.1f（比 %.3f），E_4 = %10.1f（比 %.3f）"
          % (lam, E2, E2 / base[0], E4, E4 / base[1]))
lam = np.array([0.8, 1.0, 1.25, 1.6])
E2s, E4s = [], []
for L in lam:
    us = u0 + bubble(RC0 * L, W0 * L)
    a, b, _ = energies(us, 0.0, 1.0)
    E2s.append(a); E4s.append(b)
print("     E_2 幂次拟合 = %.3f（期望 +1）；E_4 幂次拟合 = %.3f（期望 -1）"
      % (np.polyfit(np.log(lam), np.log(E2s), 1)[0], np.polyfit(np.log(lam), np.log(E4s), 1)[0]))

# H-2/H-3 筛
print()
print("[H-2/H-3] K_soft = 0.02 下扫四次项 g（球泡）")
for gq in (0.0, 0.5, 2.0, 8.0, 30.0, 120.0):
    uf, tr = run_gf(u_init, 0.02, gq)
    ext_f, peak_f = tr[-1]
    ext_0, peak_0 = tr[0]
    survive = ext_f > 0.3 * ext_0 and peak_f > 0.9 * peak_0
    R_f = radius_of(uf)
    lw = width_wall(uf, R_f) if survive else float("nan")
    print("     g = %6.1f：%s（位移>0.25 的格点数 %d -> %d，峰值 %.3f -> %.3f，R = %.2f，l* = %.3f）"
          % (gq, "驻留" if survive else "塌缩/退化", int(ext_0), int(ext_f), peak_0, peak_f, R_f, lw))

print()
print(f"总耗时 {time.time() - T_START:.1f} s")
