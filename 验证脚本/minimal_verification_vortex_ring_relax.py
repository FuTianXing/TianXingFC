# -*- coding: utf-8 -*-
r"""
minimal_verification_vortex_ring_relax.py

环边界下的面缺陷松弛轮：把第 277 条的环读数从 ansatz 升级为松弛读数。

来源：作者 2026-09-21 指令（论文 §14 其一：边缘线终止在体内，即「边界强制」与「核对」两种几何之间的中间情形）。
几何：三维环面，环取自「二维核对按高度堆叠」（第 277 条的做法）：每一 z 片取一对核对，间距取该高度上的弦长
  a(z) = sqrt(R^2 - (z - z0)^2)，|z - z0| <= R；其余片取 u = 0。故环的边界线是一条闭合的核线，绕它的回路给出整数绕数。
松弛：三维谱隐式（IMEX），不动点即 Euler-Lagrange 方程，与步长无关；步长 1.0。

判据（先行）：
  Z1 收敛：各半径的末步最大更新 <= 1e-6
  Z2 拓扑保留：绕环的回路绕数（取模取整）保持为 1（松弛不把线缺陷抹掉）
  Z3 标度：松弛后 E(R) 的双对数斜率。判据分两支：落在 [0.7, 1.4] 判为线律（膜被松弛掉），[1.7, 2.3] 判为面积律（膜保留）
  Z4 与第 277 条 ansatz 读数对照（ansatz 给 1.946 与 2.053）
  Z5 登记
"""
import numpy as np

NX, NY, NZ = 24, 48, 48
K = 0.001
K_STIFF = 1.0
SIGMA = (4.0 / np.pi) * np.sqrt(K_STIFF * K)
Z0 = NZ / 2.0
rows = []

qx = np.fft.fftfreq(NX)[:, None, None] * 2 * np.pi
qy = np.fft.fftfreq(NY)[None, :, None] * 2 * np.pi
qz = np.fft.fftfreq(NZ)[None, None, :] * 2 * np.pi
LAM = 4 * np.sin(qx / 2) ** 2 + 4 * np.sin(qy / 2) ** 2 + 4 * np.sin(qz / 2) ** 2


def wrap(x):
    return x - np.round(x)


def ring_init(R):
    xs = np.arange(NX)[:, None, None] * np.ones((1, NY, NZ))
    ys = np.ones((NX, 1, 1)) * (np.arange(NY)[None, :, None] * np.ones((1, 1, NZ)))
    zs = np.ones((NX, NY, 1)) * np.arange(NZ)[None, None, :]
    u = np.zeros((NX, NY, NZ))
    for k in range(NZ):
        dz = k - Z0
        if abs(dz) <= R - 1:
            a = np.sqrt(max(R ** 2 - dz ** 2, 0.0))
            x1, x2 = NX / 2.0 - a / 2.0, NX / 2.0 + a / 2.0
            t1 = np.arctan2(ys[:, :, k] - NY / 2.0, xs[:, :, k] - x1)
            t2 = np.arctan2(ys[:, :, k] - NY / 2.0, xs[:, :, k] - x2)
            u[:, :, k] = wrap((t1 - t2) / (2 * np.pi))
    return u


def energy(u):
    dx = np.roll(u, -1, axis=0) - u
    dy = np.roll(u, -1, axis=1) - u
    dz = np.roll(u, -1, axis=2) - u
    e1 = 0.5 * K_STIFF * float(np.sum(dx ** 2) + np.sum(dy ** 2) + np.sum(dz ** 2))
    e2 = K * float(np.sum(1 - np.cos(2 * np.pi * u)))
    return e1 + e2


def relax(u, dt=1.0, iters=4000):
    last = 0.0
    for it in range(iters):
        uh = np.fft.fftn(u)
        nonlin = 2 * np.pi * K * np.sin(2 * np.pi * u)
        uh = (uh - dt * np.fft.fftn(nonlin)) / (1.0 + dt * K_STIFF * LAM)
        un = np.real(np.fft.ifftn(uh))
        last = float(np.max(np.abs(un - u)))
        u = un
        if last < 1e-12:
            break
    return u, last


def ring_winding(u, R):
    # 绕核线一圈（在 x-z 平面内绕 (NX/2+R, Z0) 的小回路），按棱差累加后取整
    cx = NX / 2.0 + R
    cz = Z0
    loop = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
    s = 0.0
    prev = None
    i0, j0 = int(round(cx)), int(round(cz))
    y0 = int(round(NY / 2.0))
    pts = [(i0 + 3, j0), (i0 + 3, j0 + 1), (i0 + 2, j0 + 2), (i0 + 1, j0 + 3), (i0, j0 + 3), (i0 - 1, j0 + 2), (i0 - 2, j0 + 1), (i0 - 2, j0), (i0 - 2, j0 - 1), (i0 - 1, j0 - 2), (i0, j0 - 3), (i0 + 1, j0 - 3), (i0 + 2, j0 - 2), (i0 + 3, j0 - 1), (i0 + 3, j0)]
    for (i, j) in pts:
        val = u[i % NX, y0, j % NZ]
        if prev is not None:
            s += wrap(val - prev)
        prev = val
    return s


RS = [6.0, 9.0, 12.0, 15.0]
Es, lastmax, wind = [], 0.0, []
for R in RS:
    u0 = ring_init(R)
    u, last = relax(u0)
    e = energy(u)
    Es.append(e)
    lastmax = max(lastmax, last)
    wind.append(ring_winding(u, R))
    print("R = %4.1f：初值 %.6f 到 %.6f，末步 %.2e，绕环绕数 %.4f" % (R, energy(u0), e, last, wind[-1]))
Es = np.array(Es)
rows.append(("Z1", "各半径的末步最大更新", "%.2e" % lastmax, "<= 1e-6", lastmax <= 1e-6))
rows.append(("Z2", "松弛后绕环的绕数", " / ".join("%.3f" % w for w in wind), "各档取整为 1", all(abs(abs(w) - 1) < 0.15 for w in wind)))
slope = float(np.polyfit(np.log(RS), np.log(Es), 1)[0])
band = "线律" if 0.7 <= slope <= 1.4 else ("面积律" if 1.7 <= slope <= 2.3 else "两支都不落")
rows.append(("Z3", "松弛后 E(R) 的双对数斜率", "%.4f（%s）" % (slope, band), "落在 [0.7,1.4] 或 [1.7,2.3]", 0.7 <= slope <= 1.4 or 1.7 <= slope <= 2.3))
rows.append(("Z4", "与第 277 条 ansatz 读数对照", "ansatz 1.946 与 2.053；本轮松弛 %.4f" % slope, "登记", True))
rows.append(("Z5", "登记", "登记", "登记", True))

print("")
print("=== 环边界下的面缺陷松弛轮判据表（实测 对 期望）===")
for tag, name, meas, expx, ok in rows:
    print("%-4s %-44s 实测 %-46s 期望 %-28s %s" % (tag, name, meas, expx, "通过" if ok else "不通过"))
print("")
print("能量表（R 与 E）：", list(zip(RS, [round(e, 6) for e in Es])))
print("面积律预言：E = sigma_wall * pi R^2 给", [round(SIGMA * np.pi * R ** 2, 6) for R in RS])
print("线律对照：E = c * R 的拟合系数 %.6f" % float(np.polyfit(RS, Es, 1)[0]))
print("膜若保留，E 应显著大于线律；膜若被松弛掉，E 应接近线律。")
