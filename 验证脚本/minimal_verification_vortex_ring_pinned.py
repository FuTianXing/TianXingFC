# -*- coding: utf-8 -*-
r"""
minimal_verification_vortex_ring_pinned.py

强钉扎档核环轮：K = 0.24 下核环存活吗，能量按面积律还是线律。

来源：作者 2026-09-21 指令（第 326 条登记的下一步）。
几何与松弛器同第 326 条：三维环面，核环初值取自每 z 片的核对（间距取弦长），三维谱隐式松弛，不另加钉扎机制
（强钉扎档的含义即格点势本身的锁定效应，K = 0.24 即 C1-4 的驻留阈值）。
与第 326 条的唯一区别：K 从 0.001 提到 0.24（另加一档 0.30 作对照）。

判据（先行）：
  Y1 收敛：各半径末步最大更新 <= 1e-6
  Y2 存活：松弛后能量 > 1e-6 且绕核环的绕数取整为 1
  Y3 标度：存活档上 E(R) 的双对数斜率，落在 [0.7,1.4] 判线律、[1.7,2.3] 判面积律
  Y4 对照：第 277 条 ansatz 给 1.946 与 2.053；第 326 条无钉扎档四档全零
  Y5 登记
"""
import numpy as np

NX, NY, NZ = 16, 32, 32
K_STIFF = 1.0
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


def energy(u, K):
    dx = np.roll(u, -1, axis=0) - u
    dy = np.roll(u, -1, axis=1) - u
    dz = np.roll(u, -1, axis=2) - u
    e1 = 0.5 * K_STIFF * float(np.sum(dx ** 2) + np.sum(dy ** 2) + np.sum(dz ** 2))
    e2 = K * float(np.sum(1 - np.cos(2 * np.pi * u)))
    return e1 + e2, e1, e2


def relax(u, K, dt=0.2, iters=30000):
    last = 0.0
    for it in range(iters):
        uh = np.fft.fftn(u)
        nonlin = 2 * np.pi * K * np.sin(2 * np.pi * u)
        uh = (uh - dt * np.fft.fftn(nonlin)) / (1.0 + dt * K_STIFF * LAM)
        un = np.real(np.fft.ifftn(uh))
        last = float(np.max(np.abs(un - u)))
        u = un
        if last < 1e-13:
            break
    return u, last


def ring_winding(u, R):
    i0, j0 = int(round(NX / 2.0 + R)), int(round(Z0))
    y0 = int(round(NY / 2.0))
    pts = [(i0 + 3, j0), (i0 + 3, j0 + 1), (i0 + 2, j0 + 2), (i0 + 1, j0 + 3), (i0, j0 + 3),
           (i0 - 1, j0 + 2), (i0 - 2, j0 + 1), (i0 - 2, j0), (i0 - 2, j0 - 1), (i0 - 1, j0 - 2),
           (i0, j0 - 3), (i0 + 1, j0 - 3), (i0 + 2, j0 - 2), (i0 + 3, j0 - 1), (i0 + 3, j0)]
    s, prev = 0.0, None
    for (i, j) in pts:
        val = u[i % NX, y0, j % NZ]
        if prev is not None:
            s += wrap(val - prev)
        prev = val
    return s


for K in (0.24, 0.30):
    RS = [3.0, 4.0, 5.0]
    Es, wind, lastmax = [], [], 0.0
    for R in RS:
        u0 = ring_init(R)
        u, last = relax(u0, K)
        e, e1, e2 = energy(u, K)
        Es.append(e)
        wind.append(ring_winding(u, R))
        lastmax = max(lastmax, last)
        print("K = %.2f，R = %.1f：初值 %.4f 到 %.6f（梯度 %.4f + 势项 %.4f），末步 %.2e，绕数 %.4f" % (K, R, energy(u0, K)[0], e, e1, e2, last, wind[-1]))
    alive = all(e > 1e-6 for e in Es) and all(abs(abs(w) - 1) < 0.15 for w in wind)
    rows.append(("Y1", "K = %.2f：各半径末步最大更新" % K, "%.2e" % lastmax, "<= 1e-6", lastmax <= 1e-6))
    rows.append(("Y2", "K = %.2f：存活（能量 > 1e-6 且绕数取整为 1）" % K, "能量 %s；绕数 %s" % (["%.4f" % e for e in Es], ["%.2f" % w for w in wind]), "存活", alive))
    if alive:
        slope = float(np.polyfit(np.log(RS), np.log(Es), 1)[0])
        band = "线律" if 0.7 <= slope <= 1.4 else ("面积律" if 1.7 <= slope <= 2.3 else "两支都不落")
        rows.append(("Y3", "K = %.2f：E(R) 双对数斜率" % K, "%.4f（%s）" % (slope, band), "落在 [0.7,1.4] 或 [1.7,2.3]", 0.7 <= slope <= 1.4 or 1.7 <= slope <= 2.3))
    else:
        rows.append(("Y3", "K = %.2f：E(R) 双对数斜率" % K, "不存活，无定义", "登记", True))
rows.append(("Y4", "对照", "第 277 条 ansatz 1.946 与 2.053；第 326 条无钉扎档四档全零", "登记", True))
rows.append(("Y5", "登记", "登记", "登记", True))

print("")
print("=== 强钉扎档核环轮判据表（实测 对 期望）===")
for tag, name, meas, expx, ok in rows:
    print("%-4s %-40s 实测 %-56s 期望 %-26s %s" % (tag, name, meas, expx, "通过" if ok else "不通过"))
print("")
print("说明：本档的 sigma_wall = (4/pi) sqrt(K) 在 K = 0.24 给 %.4f，kink 宽度 1/(2 pi sqrt(2K)) = %.3f 格（亚格点，属强离散档）。" % ((4.0 / np.pi) * np.sqrt(K_STIFF * 0.24), 1.0 / (2 * np.pi * np.sqrt(2 * 0.24))))
print("若存活且呈面积律，则单位面积系数应与上面这个 sigma_wall 同量级；若呈线律，则膜被松弛掉、只剩核线的线张力。")
