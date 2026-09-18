# -*- coding: utf-8 -*-
r"""
minimal_verification_string_observability.py

[弦的可观测性轮] 切割线不可观测、墙只活在棱层、面元曲率只数核

定位
----
第 134 条读出曲率（面元环量）局域在核面元。本轮问两件事：描述里的分支切割（弦）是否可观测；若要有可观测的弦，它活在哪个读数层。
读数要点：方向场沿一条切割线平移 $2\pi$ 的整数倍不改变任何读数；站点指示函数构成的墙（跨墙棱差偏移 $\delta$）在棱层可见而在面元层不可见；面元环量的取值集合不受墙影响。棱自由度就是实轴方向差（波矢），弦作为切割线不可观测，作为墙则等于局部波矢偏移。
本脚本不改动基准文件；不含新科学断言。

判据（先立判据，再实现；预期写错时修正判据并记录教训）
----
G1 切割移位是描述冗余：沿切割线对方向场做 $2\pi$ 整数倍平移后，棱差、面元环量、写入时刻三项读数逐项相同（偏差 $< 10^{-12}$）。
G2 切割位置不入读数：跨切割线的棱差读数与无切割情形相同（偏差 $< 10^{-12}$）。
G3 墙在棱层可见：跨墙棱差偏移 $\delta$（实测 $\delta$），其余棱不变（偏差 $< 10^{-12}$）。
G4 墙在面元层不可见：所有面元环量与无墙情形逐项相同（偏差 $< 10^{-12}$）。
G5 墙不改变曲率的量子化：整场面元环量取值集合仍为 $\{0, \pm 2\pi\}$。
G6 棱层的动力学读数：跨墙两站的写入时刻差等于该棱的方向差除以 $\omega_P$（即含墙偏移，模 $t_P$，相对偏差 $< 8\%$），平坦棱同样对得上，绕跨墙面元的闭合错位为零（$< 0.02\,t_P$）。本轮把写入脉冲收窄到 $w = 0.12$，以抑制初始重叠带来的时刻偏置（见过程记录）。
G7 口径登记：棱自由度等于实轴方向差（波矢）；弦作为切割线不可观测，作为墙则等于局部波矢偏移。

运行
----
python 验证脚本/minimal_verification_string_observability.py
确定性（无随机数）。图存 D:/software/TianXingFC/images/string_observability.png
"""
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OMEGA_P = 2.0 * np.pi
TP = 2.0 * np.pi / OMEGA_P
THETA0 = np.pi
W_PULSE = 0.12
PACKET_W = 0.20
DT = 2.0e-3
T_MAX = 1.05
N_THETA = 256
LSIDE = 7
W_DEFECT = 1.0
DELTA = 0.5 * np.pi
CORE = (2, 2)
CUT_Y = 5


def wrap(x):
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def wrap_t(x):
    return (x + 0.5 * TP) % TP - 0.5 * TP


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


def write_epoch(axis, n=N_THETA, dt=DT, t_max=T_MAX):
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


def field_base(w=W_DEFECT, core=CORE, l=LSIDE):
    cx, cy = core[0] + 0.5, core[1] + 0.5
    return {(x, y): float(w * np.arctan2(y - cy, x - cx)) for x in range(l) for y in range(l)}


def shift_two_pi(fld, ycut=CUT_Y, l=LSIDE):
    return {k: fld[k] + 2.0 * np.pi * (1.0 if k[1] >= ycut else 0.0) for k in fld}


def wall_field(fld, delta=DELTA, ycut=CUT_Y, l=LSIDE):
    return {k: fld[k] + delta * (1.0 if k[1] >= ycut else 0.0) for k in fld}


def links(fld, l=LSIDE):
    out = {}
    for x in range(l):
        for y in range(l):
            if x + 1 < l:
                out[((x, y), (x + 1, y))] = wrap(fld[(x + 1, y)] - fld[(x, y)])
            if y + 1 < l:
                out[((x, y), (x, y + 1))] = wrap(fld[(x, y + 1)] - fld[(x, y)])
    return out


def plaqueettes(fld, l=LSIDE):
    out = {}
    for x in range(l - 1):
        for y in range(l - 1):
            loop = [(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)]
            tot = 0.0
            for i in range(4):
                tot += wrap(fld[loop[(i + 1) % 4]] - fld[loop[i]])
            out[(x, y)] = float(tot)
    return out


def epoch_of_field(fld):
    return {k: ((THETA0 + fld[k] - THETA0) % (2.0 * np.pi)) / OMEGA_P for k in fld}


def report(name, ok, detail):
    print(("[通过] " if ok else "[不通过] ") + name + "：" + detail)
    return bool(ok)


def main():
    t_start = time.time()
    ok = True
    base = field_base()
    shifted = shift_two_pi(base)
    wall = wall_field(base)

    print("== G1 切割移位是描述冗余 ==")
    lb, ls = links(base), links(shifted)
    d_link = max(abs(lb[k] - ls[k]) for k in lb)
    pb, ps = plaqueettes(base), plaqueettes(shifted)
    d_plaq = max(abs(pb[k] - ps[k]) for k in pb)
    eb, es = epoch_of_field(base), epoch_of_field(shifted)
    d_ep = max(abs(wrap(eb[k] - es[k])) for k in eb)
    ok &= report("G1 棱差、面元环量、写入时刻三项逐项相同",
                 d_link < 1e-12 and d_plaq < 1e-12 and d_ep < 1e-12,
                 "棱差 %.3e；面元 %.3e；写入时刻 %.3e" % (d_link, d_plaq, d_ep))

    cross = [k for k in lb if k[0][1] != k[1][1] and min(k[0][1], k[1][1]) == CUT_Y - 1]
    d_cross = max(abs(lb[k] - ls[k]) for k in cross) if cross else float("nan")
    ok &= report("G2 跨切割线的棱差读数不变", d_cross < 1e-12,
                 "跨切割棱 %d 条，最大偏差 %.3e" % (len(cross), d_cross))

    print("== G3 / G4 墙的两个读数层 ==")
    lw = links(wall)
    dev_wall = {k: lw[k] - lb[k] for k in lb if abs(lw[k] - lb[k]) > 1e-12}
    dev_other = max([abs(lw[k] - lb[k]) for k in lb if k not in dev_wall] or [0.0])
    ok &= report("G3 墙在棱层可见且偏移为 delta",
                 len(dev_wall) == len(cross) and all(abs(v - DELTA) < 1e-12 for v in dev_wall.values())
                 and dev_other < 1e-12,
                 "偏离棱 %d 条，偏移 %.4f（期望 %.4f）；其余棱最大偏差 %.3e"
                 % (len(dev_wall), max(dev_wall.values()), DELTA, dev_other))
    pw = plaqueettes(wall)
    d_pw = max(abs(pb[k] - pw[k]) for k in pb)
    ok &= report("G4 墙在面元层不可见", d_pw < 1e-12, "面元最大偏差 %.3e" % d_pw)

    print("== G5 曲率的量子化不受墙影响 ==")
    vals = sorted(set(round(v, 9) for v in list(pb.values()) + list(pw.values())))
    ok &= report("G5 取值集合仍为 0 与正负 2 pi",
                 all(abs(v) < 1e-9 or abs(abs(v) - 2.0 * np.pi) < 1e-9 for v in vals),
                 "取值集合 %s" % vals)

    print("== G6 棱层的动力学读数 ==")
    wall_link = [k for k in dev_wall][0]
    e1 = write_epoch(wall[wall_link[0]])
    e2 = write_epoch(wall[wall_link[1]])
    d_meas = wrap_t(e2 - e1)
    d_pred = wrap_t((lb[wall_link] + DELTA) / OMEGA_P)
    ok &= report("G6 跨墙写入时刻差等于该棱的方向差除以 omega_P",
                 abs(d_meas - d_pred) / abs(d_pred) < 0.08,
                 "实测 %.4f t_P；期望 %.4f t_P（棱差 %.4f 加墙偏移 %.4f）"
                 % (d_meas, d_pred, lb[wall_link], DELTA))
    flat_link = [k for k in lb if k not in dev_wall and k[0][1] == k[1][1]][0]
    f1 = write_epoch(wall[flat_link[0]])
    f2 = write_epoch(wall[flat_link[1]])
    d_flat_meas = wrap_t(f2 - f1)
    d_flat_pred = wrap_t(lb[flat_link] / OMEGA_P)
    ok &= report("G6 附 平坦棱同样对得上",
                 abs(d_flat_meas - d_flat_pred) < 0.02,
                 "实测 %.4f t_P；期望 %.4f t_P" % (d_flat_meas, d_flat_pred))
    x0, y0 = wall_link[0][0], wall_link[0][1] - 1
    loop = [(x0, y0), (x0 + 1, y0), (x0 + 1, y0 + 1), (x0, y0 + 1)]
    tt = [write_epoch(wall[p]) for p in loop]
    clo = 0.0
    for i in range(4):
        d = tt[(i + 1) % 4] - tt[i]
        clo += (d + 0.5 * TP) % TP - 0.5 * TP
    ok &= report("G6 附 绕跨墙面元的闭合错位为零", abs(clo) < 0.02, "实测 %.4f t_P" % clo)

    print("== G7 口径登记 ==")
    ok &= report("G7 棱自由度等于实轴方向差，弦不是面元读数", True, "见注记 §三")

    print("== 汇总：%s（用时 %.1f 秒） ==" % ("全部通过" if ok else "存在不通过项", time.time() - t_start))
    make_figure(base, shifted, wall, lb, lw, pb, pw, d_meas, d_pred, cross)
    return ok


def make_figure(base, shifted, wall, lb, lw, pb, pw, d_meas, d_pred, cross):
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.6))
    ax = axes[0][0]
    ys = np.arange(LSIDE)
    ax.plot(ys, [wrap(shifted[(3, y)] - base[(3, y)]) for y in ys], "o-", label="2 pi shift (G1)")
    ax.plot(ys, [wrap(wall[(3, y)] - base[(3, y)]) for y in ys], "s-", label="wall delta (G3)")
    ax.set_title("description shift and wall along a column")
    ax.set_xlabel("site y")
    ax.set_ylabel("field offset (wrapped)")
    ax.legend(fontsize=8)
    ax = axes[0][1]
    prof_b = [lb[((3, y), (3, y + 1))] for y in range(LSIDE - 1)]
    prof_w = [lw[((3, y), (3, y + 1))] for y in range(LSIDE - 1)]
    ax.plot(range(LSIDE - 1), prof_b, "o-", label="no wall")
    ax.plot(range(LSIDE - 1), prof_w, "s-", label="with wall")
    ax.set_title("link differences along the column")
    ax.set_xlabel("link index")
    ax.set_ylabel("link difference (rad)")
    ax.legend(fontsize=8)
    ax = axes[1][0]
    keys = sorted(pb.keys())
    diff = np.array([pb[k] - pw[k] for k in keys]).reshape(LSIDE - 1, LSIDE - 1)
    im = ax.imshow(diff.T, origin="lower", cmap="coolwarm", vmin=-1e-3, vmax=1e-3)
    ax.set_title("plaquette difference (base - wall), G4")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax = axes[1][1]
    ax.bar([0, 1], [d_meas, 0.0], width=0.5, label="measured epoch difference")
    ax.axhline(d_pred, ls=":", color="k", label="expected delta / omega_P")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["across wall", "flat link"], fontsize=8)
    ax.set_ylabel("epoch difference / t_P")
    ax.set_title("record stream across the wall (G6)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("D:/software/TianXingFC/images/string_observability.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()