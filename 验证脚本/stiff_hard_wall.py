# -*- coding: utf-8 -*-
r"""
stiff_hard_wall.py

刚性区松弛器与壁张力切换轮：修好 K >= 0.3 的松弛器，并测壁张力的档位切换。

来源：作者 2026-09-21 指令（第 329 条登记的方法待办的下一步）。
问题：第 329 条在 K = 0.30 档发散（能量由 222 反向增长到 1985）。诊断：一阶显式势项要求
  dt * (2 pi)^2 K <= 2，K = 0.30 时 dt <= 0.169，而当时取 0.2，越界。
修法：IMEX 的势项加曲率稳定化（把势项的线性化部分隐式，取最大曲率 W = (2 pi)^2 K）：
  (1 + dt (k lambda_q + W)) u_hat <- u_hat - dt F[2 pi K sin(2 pi u) - W u]，
  并令 dt = min(0.4, 1/(2 pi^2 K))；同时加能量单调性断言（反向增长即中止报错）。

几何：二维条带壁（x 两端固定为 0 与 1，y 周期），读数取 E / Ny，即单位长度壁能（第 286 条的干净壁几何）。
预期：小 K 档走连续式 sigma_wall = (4/pi) sqrt(K)；大 K 档壁宽落到亚格点，退化为一格锐壁，其单位长度能量为
  1/2（梯度项，每格一条棱差为 1 的键）加势项 0（跳变两侧都落在整数值上），故离散下限为 1/2。
  两式相等给出切换点 K* = pi^2 / 64 = 0.15421。

判据（先行）：
  T1 收敛：K = 0.30 档能量单调下降且末步 <= 1e-10（旧格式在同档发散）
  T2 小 K 复现：K = 0.001 档张力对第 286 条读数 0.0402，偏差 <= 2%
  T3 两支切换：六档 K 的实测张力落在 max 判别式 min 的那一支的 5% 内（即取两值中较小者）
  T4 切换点：实测切换落在 pi^2/64 = 0.1542 的 +-0.03 内
  T5 登记
"""
import numpy as np

NX, NY = 24, 64
rows = []


def wall_energy(u, K, k=1.0):
    dx = u[1:, :] - u[:-1, :]
    dy = np.roll(u, -1, axis=1) - u
    e1 = 0.5 * k * float(np.sum(dx ** 2) + np.sum(dy ** 2))
    e2 = K * float(np.sum(1 - np.cos(2 * np.pi * u)))
    return e1 + e2, e1, e2


def relax_wall(K, dt=None, iters=200000, k=1.0):
    if dt is None:
        dt = min(0.4, 1.0 / (2 * np.pi ** 2 * K))
    u = np.linspace(0.0, 1.0, NX)[:, None] * np.ones((1, NY))
    u[0, :] = 0.0
    u[-1, :] = 1.0
    W = (2 * np.pi) ** 2 * K
    lam = 4 * np.sin(np.fft.fftfreq(NX)[:, None] * np.pi) ** 2 + 4 * np.sin(np.fft.fftfreq(NY)[None, :] * np.pi) ** 2
    e_prev = wall_energy(u, K, k)[0]
    mono = True
    last = 0.0
    for it in range(iters):
        uh = np.fft.fft2(u)
        nonlin = 2 * np.pi * K * np.sin(2 * np.pi * u) - W * u
        uh = (uh - dt * np.fft.fft2(nonlin)) / (1.0 + dt * (k * lam + W))
        un = np.real(np.fft.ifft2(uh))
        un[0, :] = 0.0
        un[-1, :] = 1.0
        last = float(np.max(np.abs(un - u)))
        u = un
        e_now = wall_energy(u, K, k)[0]
        if e_now > e_prev + 1e-9:
            mono = False
            break
        e_prev = e_now
        if last < 1e-13:
            break
    return u, last, mono


res = {}
mono = {}
last = {}
for K in (0.001, 0.05, 0.10, 0.15, 0.20, 0.24, 0.30):
    u, l, m = relax_wall(K)
    e, e1, e2 = wall_energy(u, K)
    res[K] = e / NY
    mono[K] = m
    last[K] = l
    print("K = %.3f：dt = %.4f，单位长度壁能 %.6f（梯度 %.4f 势项 %.4f），末步 %.2e，单调 %s" % (K, min(0.4, 1.0 / (2 * np.pi ** 2 * K)), res[K], e1 / NY, e2 / NY, l, m))

KSTAR = np.pi ** 2 / 64.0
rows.append(("T1", "K = 0.30 档收敛（旧格式在同档发散）", "末步 %.2e，单调 %s" % (last[0.30], mono[0.30]), "末步 <= 1e-10 且单调", last[0.30] <= 1e-10 and mono[0.30]))
rows.append(("T2", "K = 0.001 档张力对第 286 条读数 0.0402", "%.6f（偏差 %.2f%%）" % (res[0.001], 100 * abs(res[0.001] - 0.0402) / 0.0402), "偏差 <= 2%", abs(res[0.001] - 0.0402) / 0.0402 <= 0.02))
worst = 0.0
detail = []
for K in (0.05, 0.10, 0.15, 0.20, 0.24, 0.30):
    cont = (4.0 / np.pi) * np.sqrt(K)
    sharp = 0.5
    pred = min(cont, sharp)
    dev = abs(res[K] - pred) / pred
    worst = max(worst, dev)
    detail.append("K=%.2f: %.4f 对 %.4f" % (K, res[K], pred))
rows.append(("T3", "六档张力对两支中较小者", "；".join(detail) + "（最大偏差 %.1f%%）" % (100 * worst), "偏差 <= 5%", worst <= 0.05))
cross = None
Ks = sorted(res)
for i in range(len(Ks) - 1):
    a, b = Ks[i], Ks[i + 1]
    da = abs(res[a] - (4 / np.pi) * np.sqrt(a))
    db = abs(res[b] - 0.5)
    if abs(res[a] - (4 / np.pi) * np.sqrt(a)) < abs(res[a] - 0.5) and abs(res[b] - 0.5) < abs(res[b] - (4 / np.pi) * np.sqrt(b)):
        cross = (a + b) / 2.0
        break
rows.append(("T4", "实测切换点与闭式 pi^2/64 = %.4f" % KSTAR, "实测 %s" % ("%.3f" % cross if cross else "未落在网格内"), "偏差 <= 0.03", cross is not None and abs(cross - KSTAR) <= 0.03))
rows.append(("T5", "登记", "登记", "登记", True))

print("")
print("=== 刚性区松弛器与壁张力切换轮判据表（实测 对 期望）===")
for tag, name, meas, expx, ok in rows:
    print("%-4s %-40s 实测 %-64s 期望 %-26s %s" % (tag, name, meas, expx, "通过" if ok else "不通过"))
print("")
print("两支预言：连续式 (4/pi) sqrt(K) 与离散锐壁下限 1/2；相等处 K* = pi^2/64 = %.5f。" % KSTAR)
print("张力表：", {round(k, 3): round(v, 5) for k, v in res.items()})
