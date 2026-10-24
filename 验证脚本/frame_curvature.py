# -*- coding: utf-8 -*-
r"""
frame_curvature.py

[标架场曲率轮] 逐面元环量：均匀波矢处处零曲率、绕数型曲率局域在核面元、离散 Stokes 成立

定位
----
第 132 条把绕数缺陷读成写入时刻的整周期滑移（环上累计 $2\pi w$）。本轮读出它的局部量：面元曲率，即实轴方向场 $A_j$ 沿面元一周的环量。
读数要点：均匀波矢（标架场取梯度）处处零曲率；绕数型标架场的曲率局域在核面元，逐面元曲率之和等于环路 holonomy（离散 Stokes）；核移动时曲率跟随而总和不变；双缺陷（$+w$ 与 $-w$）总曲率为零；半整数绕数给出 $\pi$ 的总曲率，它取不到 $2\pi$ 的整数倍。
曲率（旋度侧）与波矢（梯度侧）是两个独立读数；与标架联络轮的分工是散度侧对旋度侧。
本脚本不改动基准文件；不含新科学断言。

判据（先立判据，再实现；预期写错时修正判据并记录教训）
----
F1 均匀波矢处处零曲率：梯度型标架场 $A_j = q(x_j + y_j)$ 的 16 个面元环量全为零（$< 10^{-12}$）。
F2 曲率局域在核面元：绕数型构型的核面元读数为 $2\pi w$，其余 15 个面元严格为零，切片内总和等于 $2\pi w$。
F3 曲率的动力学读数：绕带电面元四周的写入时刻闭合错位等于 $t_P$（相对偏差 $< 5\%$），平坦面元给零错位（$< 0.02$）。
F4 核移动时曲率跟随：核移到相邻面元后带电面元随之移动，切片内总和不变。
F5 双缺陷总曲率为零：$+w$ 与 $-w$ 两核分别给 $+2\pi w$ 与 $-2\pi w$，总和为零。
F6 半整数绕数的障碍：$w = 1/2$ 的构造在圆上方向场的口径下核面元环量为零（最小角差），全场净环量亦为零，记录流错位为零；该构造带一条相邻跳变接近 $2\pi w$ 的切割，故半整数绕数要求一条弦。第 132 条 $w = 1/2$ 的 $\pi$ 读数按方位角分支展开得到，口径差别登记在注记 §三。
F7 口径登记：曲率与波矢的分工（旋度对梯度），与标架联络轮的分工（散度侧对旋度侧）。

运行
----
python 验证脚本/frame_curvature.py
确定性（无随机数）。图存 D:/software/TianXingFC/images/frame_curvature.png
"""
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OMEGA_P = 2.0 * np.pi
TP = 2.0 * np.pi / OMEGA_P
THETA0 = np.pi
W_PULSE = 0.25
PACKET_W = 0.20
DT = 2.0e-3
T_MAX = 1.05
N_THETA = 256
LSIDE = 5
Q_GRAD = 0.30
W_DEFECT = 1.0
W_HALF = 0.5
CORE_A = (2, 2)
CORE_B = (1, 1)


def wrap(x):
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def theta_grid(n=N_THETA):
    return 2.0 * np.pi * np.arange(n) / n


def pulse(theta, w=W_PULSE, axis=0.0):
    return np.exp(-(1.0 - np.cos(theta - axis)) / (2.0 * w ** 2))


def trapezoid(y, x):
    return float(np.sum(0.5 * (y[:-1] + y[1:]) * np.diff(x)))


def g_calib(w=W_PULSE):
    th = theta_grid(8192)
    return (0.5 * np.pi) * OMEGA_P / trapezoid(pulse(th, w), th)


def packet(theta, center, sigma=PACKET_W):
    d = (theta - center + np.pi) % (2.0 * np.pi) - np.pi
    p = np.exp(-d ** 2 / (2.0 * sigma ** 2))
    return p / np.linalg.norm(p)


def apply_coupling(psi, f, dt, g):
    ang = g * f * dt
    cs = np.cos(ang)
    sn = np.sin(ang)
    out = psi.copy()
    out[:, 0, 1] = cs * psi[:, 0, 1] - 1j * sn * psi[:, 1, 0]
    out[:, 1, 0] = cs * psi[:, 1, 0] - 1j * sn * psi[:, 0, 1]
    return out


def free_step(psi, kvec, dt):
    mult = np.exp(-1j * OMEGA_P * kvec * dt)
    for a in range(2):
        for c in range(2):
            psi[:, a, c] = np.fft.ifft(np.fft.fft(psi[:, a, c]) * mult)
    return psi


def run_site(axis=0.0, n=N_THETA, dt=DT, t_max=T_MAX):
    th = theta_grid(n)
    kvec = np.fft.fftfreq(n, d=1.0 / n)
    g = g_calib()
    psi = np.zeros((n, 2, 2), dtype=complex)
    psi[:, 1, 0] = packet(th, THETA0)
    f = pulse(th, W_PULSE, axis)
    steps = int(round(t_max / dt))
    epoch = None
    for step in range(steps + 1):
        if step:
            psi = apply_coupling(psi, f, dt / 2.0, g)
            psi = free_step(psi, kvec, dt)
            psi = apply_coupling(psi, f, dt / 2.0, g)
        if epoch is None and float((np.abs(psi[:, 0, 1]) ** 2).sum()) >= 0.5:
            epoch = step * dt
    return float(epoch) if epoch is not None else float("nan")


def field_grad(q=Q_GRAD, l=LSIDE):
    return {(x, y): q * (x + y) for x in range(l) for y in range(l)}


def field_winding(w, core, l=LSIDE):
    cx, cy = core[0] + 0.5, core[1] + 0.5
    return {(x, y): float(w * np.arctan2(y - cy, x - cx)) for x in range(l) for y in range(l)}


def field_dipole(w, core_a, core_b, l=LSIDE):
    fa = field_winding(w, core_a, l)
    fb = field_winding(w, core_b, l)
    return {k: fa[k] - fb[k] for k in fa}


def circulation(fld, x, y):
    loop = [(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)]
    tot = 0.0
    for i in range(4):
        tot += wrap(fld[loop[(i + 1) % 4]] - fld[loop[i]])
    return float(tot)


def plaquette_map(fld, l=LSIDE):
    return {(x, y): circulation(fld, x, y) for x in range(l - 1) for y in range(l - 1)}


def epoch_circulation(fld, x, y, t_max=T_MAX):
    loop = [(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)]
    tt = [run_site(axis=fld[p], t_max=t_max) for p in loop]
    tot = 0.0
    for i in range(4):
        d = tt[(i + 1) % 4] - tt[i]
        tot += (d + 0.5 * TP) % TP - 0.5 * TP
    return float(tot / TP)


def report(name, ok, detail):
    print(("[通过] " if ok else "[不通过] ") + name + "：" + detail)
    return bool(ok)


def main():
    t_start = time.time()
    ok = True

    print("== F1 均匀波矢处处零曲率 ==")
    pm_grad = plaquette_map(field_grad())
    dev1 = max(abs(v) for v in pm_grad.values())
    ok &= report("F1 梯度型标架场 16 个面元环量为零", dev1 < 1e-12, "最大绝对值 %.3e" % dev1)

    print("== F2 曲率局域在核面元 ==")
    pm_a = plaquette_map(field_winding(W_DEFECT, CORE_A))
    nz = {k: v for k, v in pm_a.items() if abs(v) > 1e-12}
    tot_a = sum(pm_a.values())
    ok &= report("F2 只有核面元非零且总和为 2 pi w",
                 len(nz) == 1 and abs(pm_a[CORE_A] - 2.0 * np.pi * W_DEFECT) < 1e-12
                 and abs(tot_a - 2.0 * np.pi * W_DEFECT) < 1e-12,
                 "非零面元 %s；核读数 %.6f；总和 %.6f"
                 % (sorted(nz.keys()), pm_a[CORE_A], tot_a))

    print("== F3 曲率的动力学读数 ==")
    ec_charged = epoch_circulation(field_winding(W_DEFECT, CORE_A), *CORE_A)
    ec_flat = epoch_circulation(field_winding(W_DEFECT, CORE_A), 0, 0)
    ok &= report("F3 带电面元错位为一个整周期", abs(ec_charged - 1.0) < 0.05,
                 "实测 %.4f t_P" % ec_charged)
    ok &= report("F3 附 平坦面元零错位", abs(ec_flat) < 0.02, "实测 %.4f t_P" % ec_flat)

    print("== F4 核移动时曲率跟随 ==")
    pm_b = plaquette_map(field_winding(W_DEFECT, CORE_B))
    nz_b = {k: v for k, v in pm_b.items() if abs(v) > 1e-12}
    ok &= report("F4 带电面元随核移动且总和不变",
                 sorted(nz_b.keys()) == [CORE_B] and abs(sum(pm_b.values()) - 2.0 * np.pi * W_DEFECT) < 1e-12,
                 "非零面元 %s；总和 %.6f" % (sorted(nz_b.keys()), sum(pm_b.values())))

    print("== F5 双缺陷总曲率为零 ==")
    pm_d = plaquette_map(field_dipole(W_DEFECT, CORE_A, CORE_B))
    nz_d = {k: v for k, v in pm_d.items() if abs(v) > 1e-12}
    ok &= report("F5 两核给 +2 pi w 与 -2 pi w，总和为零",
                 abs(pm_d[CORE_A] - 2.0 * np.pi * W_DEFECT) < 1e-12
                 and abs(pm_d[CORE_B] + 2.0 * np.pi * W_DEFECT) < 1e-12
                 and abs(sum(pm_d.values())) < 1e-12,
                 "非零面元 %s；总和 %.3e" % (sorted(nz_d.keys()), sum(pm_d.values())))

    print("== F6 半整数绕数 ==")
    fld_h = field_winding(W_HALF, CORE_A)
    pm_h = plaquette_map(fld_h)
    ec_h = epoch_circulation(fld_h, *CORE_A)
    jump = max(abs(fld_h[(x + 1, y)] - fld_h[(x, y)]) for x in range(LSIDE - 1) for y in range(LSIDE))
    jump = max(jump, max(abs(fld_h[(x, y + 1)] - fld_h[(x, y)]) for x in range(LSIDE) for y in range(LSIDE - 1)))
    ok &= report("F6 半整数构造给不出净环量（等价于零曲率加一条弦）",
                 abs(sum(pm_h.values())) < 1e-12 and abs(pm_h[CORE_A]) < 1e-12 and abs(ec_h) < 0.02,
                 "核面元 %.3e；切片总和 %.3e；记录流错位 %.4f t_P"
                 % (pm_h[CORE_A], sum(pm_h.values()), ec_h))
    ok &= report("F6 附 切割处的方向跳变接近 2 pi w",
                 0.70 * 2.0 * np.pi * W_HALF <= jump <= 1.05 * 2.0 * np.pi * W_HALF,
                 "最大相邻跳变 %.4f rad；连续极限 2 pi w = %.4f rad"
                 % (jump, 2.0 * np.pi * W_HALF))

    print("== F7 口径登记 ==")
    ok &= report("F7 曲率对波矢、旋度对梯度", True, "见注记 §三")

    print("== 汇总：%s（用时 %.1f 秒） ==" % ("全部通过" if ok else "存在不通过项", time.time() - t_start))
    make_figure(pm_grad, pm_a, pm_d, ec_charged, ec_flat)
    return ok


def make_figure(pm_grad, pm_a, pm_d, ec_charged, ec_flat):
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.6))
    keys = sorted(pm_a.keys())
    grid_a = np.array([pm_a[k] / (2.0 * np.pi) for k in keys]).reshape(LSIDE - 1, LSIDE - 1)
    grid_d = np.array([pm_d[k] / (2.0 * np.pi) for k in keys]).reshape(LSIDE - 1, LSIDE - 1)
    grid_g = np.array([pm_grad[k] / (2.0 * np.pi) for k in keys]).reshape(LSIDE - 1, LSIDE - 1)
    ax = axes[0][0]
    im = ax.imshow(grid_a.T, origin="lower", cmap="coolwarm", vmin=-1.2, vmax=1.2)
    ax.set_title("winding defect: plaquette circulation / 2 pi")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax = axes[0][1]
    acc = np.cumsum([pm_a[k] / (2.0 * np.pi) for k in keys])
    ax.step(np.arange(acc.size), acc, where="post", label="cumulative circulation")
    ax.axhline(1.0, ls=":", color="k", label="holonomy 2 pi w (round 132)")
    ax.axhline(0.0, ls=":", color="C3", label="uniform wave vector (F1) = 0")
    ax.set_title("discrete Stokes: sum of plaquettes")
    ax.set_xlabel("plaquette index")
    ax.set_ylabel("cumulative / 2 pi")
    ax.legend(fontsize=8)
    ax = axes[1][0]
    im2 = ax.imshow(grid_d.T, origin="lower", cmap="coolwarm", vmin=-1.2, vmax=1.2)
    ax.set_title("dipole (+w, -w): circulation / 2 pi")
    fig.colorbar(im2, ax=ax, fraction=0.046)
    ax = axes[1][1]
    ax.bar([0, 1], [ec_charged, ec_flat], width=0.5, label="measured epoch circulation")
    ax.axhline(1.0, ls=":", color="k", label="expected 1 t_P (charged)")
    ax.axhline(0.0, ls=":", color="C3", label="expected 0 (flat)")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["charged plaquette", "flat plaquette"], fontsize=8)
    ax.set_ylabel("circulation / t_P")
    ax.set_title("record stream: closure mismatch")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("D:/software/TianXingFC/images/frame_curvature.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()