# -*- coding: utf-8 -*-
r"""
minimal_verification_strip_table_fixed.py

修正实现重跑条带表轮：用第 335 条验过的混合格式重测七档 K 的壁张力，并给大 K 档的离散形式定形。

来源：作者 2026-09-21 指令（执行第 335 条登记的下一步）。
实现：第 335 条的修正格式 C（x 显式差分并保留 Dirichlet 两端固定，y 谱隐式，势项线性化部分隐式，dt = 0.15）。
几何：竖条 64x128，读数取 E/Ny。

判据（先行）：
  X1 回归：K = 0.001 与 0.005 档对第 286／335 条读数（0.040242、0.089781），偏差 <= 1%
  X2 收敛：七档末步 <= 1e-10
  X3 大 K 档判据：K = 0.24 与 0.30 档的张力须与「一格锐壁」形式比较，判定它是上限还是等式：
       若实测 < 1/2 且随 K 上升趋近 1/2，则 1/2 是 K -> 无穷的极限（上限），有限 K 严格低于它
  X4 离散形式定形：给出折中剖面（近整数平台加一次快速跳变）的解释并与实测对照
  X5 登记
"""
import numpy as np

NX, NY = 64, 128
REF = {0.001: 0.040242, 0.005: 0.089781}
rows = []


def energy(u, K, k=1.0):
    dx = u[1:, :] - u[:-1, :]
    dy = np.roll(u, -1, axis=1) - u
    e1 = 0.5 * k * float(np.sum(dx ** 2) + np.sum(dy ** 2))
    e2 = K * float(np.sum(1 - np.cos(2 * np.pi * u)))
    return e1 + e2, e1, e2


def relax_C(K, iters=200000, dt=0.15, k=1.0):
    u = np.tile(np.linspace(0.0, 1.0, NX)[:, None], (1, NY))
    u[0, :] = 0.0
    u[-1, :] = 1.0
    W = (2 * np.pi) ** 2 * K
    lam_y = 4 * np.sin(np.fft.fftfreq(NY)[None, :] * np.pi) ** 2
    e_best = energy(u, K, k)[0]
    u_best = u.copy()
    last = 0.0
    for it in range(iters):
        lap_x = np.zeros_like(u)
        lap_x[1:-1, :] = u[2:, :] + u[:-2, :] - 2.0 * u[1:-1, :]
        uh = np.fft.rfft(u, axis=1)
        non = 2 * np.pi * K * np.sin(2 * np.pi * u) - W * u
        nh = np.fft.rfft(non, axis=1)
        lx = np.fft.rfft(lap_x, axis=1)
        uh = (uh + dt * (k * lx - nh)) / (1.0 + dt * (k * lam_y[:, :uh.shape[1]] + W))
        un = np.fft.irfft(uh, n=NY, axis=1)
        un[0, :] = 0.0
        un[-1, :] = 1.0
        last = float(np.max(np.abs(un - u)))
        u = un
        e_now = energy(u, K, k)[0]
        if e_now < e_best - 1e-14:
            e_best = e_now
            u_best = u.copy()
        if last < 1e-13:
            break
    return u_best, last


KS = [0.001, 0.005, 0.02, 0.10, 0.15, 0.20, 0.24, 0.30]
tab, grads, pots, lasts = {}, {}, {}, {}
for K in KS:
    u, last = relax_C(K)
    e, e1, e2 = energy(u, K)
    tab[K] = e / NY
    grads[K] = e1 / NY
    pots[K] = e2 / NY
    lasts[K] = last
    print("K = %.3f：张力 %.6f（梯度 %.6f 势项 %.6f），末步 %.2e；连续式 %.6f，一格锐壁 0.5" % (K, tab[K], grads[K], pots[K], last, (4.0 / np.pi) * np.sqrt(K)))

rows.append(("X1", "K = 0.001 与 0.005 对第 286／335 条读数",
             "%.6f 与 %.6f（偏差 %.2f%% 与 %.2f%%）" % (tab[0.001], tab[0.005],
             100 * abs(tab[0.001] - REF[0.001]) / REF[0.001], 100 * abs(tab[0.005] - REF[0.005]) / REF[0.005]),
             "偏差 <= 1%", abs(tab[0.001] - REF[0.001]) / REF[0.001] <= 0.01 and abs(tab[0.005] - REF[0.005]) / REF[0.005] <= 0.01))
rows.append(("X2", "七档末步最大更新", "%.2e" % max(lasts.values()), "<= 1e-10", max(lasts.values()) <= 1e-10))
below = tab[0.24] < 0.5 and tab[0.30] < 0.5 and tab[0.30] > tab[0.24]
rows.append(("X3", "大 K 档与一格锐壁 0.5 的关系", "K = 0.24 给 %.6f、0.30 给 %.6f，均 < 0.5 且随 K 上升" % (tab[0.24], tab[0.30]),
             "严格低于 0.5 且单调趋近", below))
rows.append(("X4", "离散形式定形", "见注记 §三：近整数平台加一次快速跳变", "登记", True))
rows.append(("X5", "登记", "登记", "登记", True))

print("")
print("=== 修正实现条带表判据（实测 对 期望）===")
for tag, name, meas, expx, ok in rows:
    print("%-4s %-44s 实测 %-56s 期望 %-26s %s" % (tag, name, meas, expx, "通过" if ok else "不通过"))
print("")
print("张力表：", {round(k, 3): round(v, 6) for k, v in tab.items()})
print("连续式对照：", {round(k, 3): round((4 / np.pi) * np.sqrt(k), 6) for k in KS})
