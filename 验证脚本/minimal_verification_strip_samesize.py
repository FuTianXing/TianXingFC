# -*- coding: utf-8 -*-
r"""
minimal_verification_strip_samesize.py

同尺寸条带小 K 档重测轮：判第 330 条的 T2，并复核第 286 条的两档读数。

来源：作者 2026-09-21 指令（执行「同尺寸条带 64x128 重测小 K 档」）。
几何与第 286 条的干净壁完全同尺寸：竖条 $64\times128$，x 两端固定为 0 与 1，y 周期，读数取 $E/N_y$。
松弛器用第 330 条修好的曲率稳定化 IMEX：势项线性化部分隐式（$W = (2\pi)^2K$ 进分母），
$\Delta t = \min(0.4, 1/(2\pi^2K))$，并带能量单调性断言。

判据（先行）：
  U1 复现第 286 条五档读数（0.177866 / 0.126605 / 0.089781 / 0.056878 / 0.040242），偏差 <= 2%
  U2 五档对解析式 sigma_wall = (4/pi) sqrt(K)，偏差 <= 2%
  U3 收敛：五档末步 <= 1e-10 且能量单调
  U4 有限尺寸量化：本轮 K = 0.001 档对第 330 条 24x64 条带的 0.048270 之差
  U5 登记
"""
import numpy as np

NX, NY = 64, 128
REF = {0.02: 0.177866, 0.01: 0.126605, 0.005: 0.089781, 0.002: 0.056878, 0.001: 0.040242}
rows = []


def wall_energy(u, K, k=1.0):
    dx = u[1:, :] - u[:-1, :]
    dy = np.roll(u, -1, axis=1) - u
    e1 = 0.5 * k * float(np.sum(dx ** 2) + np.sum(dy ** 2))
    e2 = K * float(np.sum(1 - np.cos(2 * np.pi * u)))
    return e1 + e2, e1, e2


def relax_wall(K, iters=200000, k=1.0):
    dt = min(0.4, 1.0 / (2 * np.pi ** 2 * K))
    u = np.linspace(0.0, 1.0, NX)[:, None] * np.ones((1, NY))
    u[0, :] = 0.0
    u[-1, :] = 1.0
    W = (2 * np.pi) ** 2 * K
    fx = np.fft.fftfreq(NX)[:, None] * np.pi
    fy = np.fft.fftfreq(NY)[None, :] * np.pi
    lam = 4 * np.sin(fx) ** 2 + 4 * np.sin(fy) ** 2
    e_prev = wall_energy(u, K, k)[0]
    e_best = e_prev
    u_best = u.copy()
    mono, last, bumps = True, 0.0, 0
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
        if e_now < e_best - 1e-14:
            e_best = e_now
            u_best = u.copy()
        elif e_now > e_prev + 1e-9:
            bumps += 1
        e_prev = e_now
        if last < 1e-13:
            break
    return u_best, last, (bumps == 0), dt, bumps


meas, dev_ref, dev_ana, last_, mono_ = {}, {}, {}, {}, {}
for K in (0.02, 0.01, 0.005, 0.002, 0.001):
    u, last, mono, dt, bumps = relax_wall(K)
    e, e1, e2 = wall_energy(u, K)
    meas[K] = e / NY
    ana = (4.0 / np.pi) * np.sqrt(K)
    dev_ref[K] = abs(meas[K] - REF[K]) / REF[K]
    dev_ana[K] = abs(meas[K] - ana) / ana
    last_[K] = last
    mono_[K] = mono
    print("K = %.3f：dt = %.4f，张力 %.6f（第 286 条 %.6f，偏差 %.2f%%；解析 %.6f，偏差 %.2f%%），末步 %.2e，单调 %s"
          % (K, dt, meas[K], REF[K], 100 * dev_ref[K], ana, 100 * dev_ana[K], last, mono) + ("（能量极小于末态，已取最优态）" if not mono else ""))

w1 = max(dev_ref.values())
w2 = max(dev_ana.values())
rows.append(("U1", "五档对第 286 条干净壁读数", "最大偏差 %.2f%%" % (100 * w1), "<= 2%", w1 <= 0.02))
rows.append(("U2", "五档对解析式 sigma_wall", "最大偏差 %.2f%%" % (100 * w2), "<= 2%", w2 <= 0.02))
rows.append(("U3", "五档收敛（末步与单调）", "最大末步 %.2e；单调 %s" % (max(last_.values()), all(mono_.values())), "末步 <= 1e-10 且单调", max(last_.values()) <= 1e-10 and all(mono_.values())))
fs = 0.048270
rows.append(("U4", "有限尺寸量化：本轮 K = 0.001 对第 330 条 24x64 的 %.6f" % fs, "%.6f，差 %.1f%%" % (meas[0.001], 100 * abs(meas[0.001] - fs) / fs), "登记", True))
rows.append(("U5", "登记", "登记", "登记", True))

print("")
print("=== 同尺寸条带小 K 档重测轮判据表（实测 对 期望）===")
for tag, name, measS, expx, ok in rows:
    print("%-4s %-44s 实测 %-52s 期望 %-24s %s" % (tag, name, measS, expx, "通过" if ok else "不通过"))
print("")
print("张力表（本轮 64x128）：", {round(k, 3): round(v, 6) for k, v in meas.items()})
print("第 286 条读数对照：", {round(k, 3): round(v, 6) for k, v in REF.items()})
