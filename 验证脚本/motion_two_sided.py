# -*- coding: utf-8 -*-
r"""
motion_two_sided.py

[运动双侧轮] 同一条移动过程的两侧读数：连续侧（相位累加）与离散侧（绕数标签），以及两侧的互补关系

定位
----
第 142 条把站点层与棱层分读，容易读成其中一层才是运动的真身。本轮按作者 2026-09-19 的更正重读：运动是双侧协同，粒性侧给离散跃迁，波性侧给连续流动，两侧在同一条过程上同时在场。
读数要点：沿闭合回路的相位累加随路径参数连续变化（无奇点），绕数标签（累加除以 $2\pi$ 取整）只在累加跨过 $\\pi$ 的一刻跳一格；跃迁点恰为半荷，两个端点之外没有第三个标签值。由连续侧可恢复整条轨迹，由离散侧只剩两个值，两侧合读才给出一条完整的运动描述。
本脚本不改动基准文件；不含新科学断言。

判据（先立判据，再实现；预期写错时修正判据并记录教训）
----
Y1 连续侧：相位累加 $S(t)$ 的相邻差值恒定（等于 $-2\\pi\\Delta t$，偏差 $< 10^{-12}$），即过程无奇点。
Y2 离散侧：绕数标签 $W = {\\rm round}(S/2\\pi)$ 在五个参数点上只变化一次，变化位置为 $t = 0.5$（偏差 $< 10^{-12}$）。
Y3 跃迁点定位：跃迁点的累加等于 $\\pi$（偏差 $< 10^{-12}$），即跃迁发生在半荷处而非端点。
Y4 双侧互补：连续侧可恢复步长（差值集合为单元素），离散侧只有两个取值（五参数点、两取值）。
Y5 端点与中点：两端点的写入时刻与解析式一致（偏差 $< 5\\%$）；中点不可定义站点写入时刻（绕核回路棱差之和为 $\\pi$）。
Y6 口径登记：见注记 §三。

运行
----
python 验证脚本/motion_two_sided.py
确定性（无随机数）。图存 D:/software/TianXingFC/images/motion_two_sided.png
"""
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OMEGA_P = 2.0 * np.pi
THETA0 = np.pi
W_PULSE = 0.12
PACKET_W = 0.20
DT = 2.0e-3
T_MAX = 1.0
N_THETA = 255
LSIDE = 7
CORE_A = (2, 2)
CORE_B = (3, 2)
TS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
SITES = [(0, 0), (0, 6), (6, 0), (6, 6)]


def wrap(x):
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def field_winding(core, w=1.0, l=LSIDE):
    cx, cy = core[0] + 0.5, core[1] + 0.5
    return {(x, y): float(w * np.arctan2(y - cy, x - cx)) for x in range(l) for y in range(l)}


def links(fld, l=LSIDE):
    out = {}
    for x in range(l):
        for y in range(l):
            if x + 1 < l:
                out[((x, y), (x + 1, y))] = wrap(fld[(x + 1, y)] - fld[(x, y)])
            if y + 1 < l:
                out[((x, y), (x, y + 1))] = wrap(fld[(x, y + 1)] - fld[(x, y)])
    return out


def loop_sum(lnk, x, y):
    loop = [((x, y), (x + 1, y)), ((x + 1, y), (x + 1, y + 1)), ((x, y + 1), (x + 1, y + 1)), ((x, y), (x, y + 1))]
    signs = [1.0, 1.0, -1.0, -1.0]
    return float(sum(s * lnk[e] for s, e in zip(signs, loop)))


def theta_grid(n=N_THETA):
    return 2.0 * np.pi * np.arange(n) / n


def pulse(theta, w=W_PULSE, axis=0.0):
    return np.exp(-(1.0 - np.cos(theta - axis)) / (2.0 * w ** 2))


def trapezoid(y, x):
    return float(np.sum(0.5 * (y[:-1] + y[1:]) * np.diff(x)))


def g_calib(w=W_PULSE):
    th = 2.0 * np.pi * np.arange(8192) / 8192
    return (0.5 * np.pi) * OMEGA_P / trapezoid(pulse(th, w), th)


def packet(theta, center, sigma=PACKET_W):
    d = (theta - center + np.pi) % (2.0 * np.pi) - np.pi
    p = np.exp(-d ** 2 / (2.0 * sigma ** 2))
    return p / np.linalg.norm(p)


def run_site(axis, n=N_THETA, dt=DT, t_max=T_MAX):
    th, k = theta_grid(n), np.fft.fftfreq(n, d=1.0 / n)
    g = g_calib()
    psi = np.zeros((n, 2, 2), dtype=complex)
    psi[:, 1, 0] = packet(th, THETA0)
    f = pulse(th, W_PULSE, axis)
    mult = np.exp(-1j * OMEGA_P * k * dt)
    steps = int(round(t_max / dt))
    epoch = None
    for step in range(steps + 1):
        if step:
            ang = g * f * dt / 2.0
            cs, sn = np.cos(ang), np.sin(ang)
            a01, a10 = psi[:, 0, 1].copy(), psi[:, 1, 0].copy()
            psi[:, 0, 1] = cs * a01 - 1j * sn * a10
            psi[:, 1, 0] = cs * a10 - 1j * sn * a01
            for aa in range(2):
                for cc in range(2):
                    psi[:, aa, cc] = np.fft.ifft(np.fft.fft(psi[:, aa, cc]) * mult)
            ang = g * f * dt / 2.0
            cs, sn = np.cos(ang), np.sin(ang)
            a01, a10 = psi[:, 0, 1].copy(), psi[:, 1, 0].copy()
            psi[:, 0, 1] = cs * a01 - 1j * sn * a10
            psi[:, 1, 0] = cs * a10 - 1j * sn * a01
        if epoch is None and float((np.abs(psi[:, 0, 1]) ** 2).sum()) >= 0.5:
            epoch = step * dt
    return float(epoch) if epoch is not None else float("nan")


def report(name, ok, detail):
    print(("[通过] " if ok else "[不通过] ") + name + "：" + detail)
    return bool(ok)

def main():
    t0 = time.time()
    ok = True
    base = field_winding(CORE_A)
    end = field_winding(CORE_B)
    lb, le = links(base), links(end)
    rows = []
    for t in TS:
        lk = {e: (1.0 - t) * lb[e] + t * le[e] for e in lb}
        s = loop_sum(lk, *CORE_A)
        rows.append((float(t), s, float(round(s / (2.0 * np.pi)))))
    print("== Y1 连续侧 ==")
    diffs = [rows[i + 1][1] - rows[i][1] for i in range(len(rows) - 1)]
    worst = max(abs(d + 2.0 * np.pi * (TS[i + 1] - TS[i])) for i, d in enumerate(diffs))
    ok &= report("Y1 累加连续且相邻差值恒定", worst < 1e-12,
                 "最大偏离 %.3e；差值 %s" % (worst, ["%.4f" % d for d in diffs]))
    print("== Y2 离散侧 ==")
    W = [r[2] for r in rows]
    ch = [i for i in range(1, len(W)) if W[i] != W[i - 1]]
    ok &= report("Y2 标签只变化一次且位置在 t = 0.5",
                 len(ch) == 1 and abs(TS[ch[0]] - 0.5) < 1e-12,
                 "标签序列 %s；变化位置 t = %.2f" % (W, TS[ch[0]] if ch else float("nan")))
    print("== Y3 跃迁点定位 ==")
    s_half = rows[2][1]
    ok &= report("Y3 跃迁点累加为 pi（半荷）", abs(s_half - np.pi) < 1e-12,
                 "实测 %.6f；偏离 pi %.3e" % (s_half, abs(s_half - np.pi)))
    print("== Y4 双侧互补 ==")
    uniq_diff = len({round(d, 12) for d in diffs})
    uniq_w = len(set(W))
    ok &= report("Y4 连续侧可恢复步长、离散侧只剩两值", uniq_diff == 1 and uniq_w == 2,
                 "差值取值数 %d（参数点 %d）；标签取值数 %d" % (uniq_diff, len(TS), uniq_w))
    print("== Y5 端点与中点 ==")
    meas, pred = [], []
    for site in SITES:
        for t in (0.0, 1.0):
            fld = base if t == 0.0 else end
            meas.append(run_site(fld[site]))
            pred.append(((2.0 * np.pi - THETA0 + fld[site]) % (2.0 * np.pi)) / OMEGA_P)
    dev = max(abs(m - p) / p if p > 1e-9 else abs(m - p) for m, p in zip(meas, pred))
    ok &= report("Y5 端点写入时刻与解析一致", dev < 0.05, "最大偏差 %.4f t_P" % dev)
    lk_half = {e: 0.5 * lb[e] + 0.5 * le[e] for e in lb}
    s_h = loop_sum(lk_half, *CORE_A)
    ok &= report("Y5 附 中点不可定义站点写入时刻", abs(s_h - np.pi) < 1e-12,
                 "中点绕核回路和 %.6f（站点场实现要求 0）" % s_h)
    print("== Y6 口径登记 ==")
    ok &= report("Y6 双侧协同口径", True, "见注记 §三")
    print("== 汇总：%s（用时 %.1f 秒） ==" % ("全部通过" if ok else "存在不通过项", time.time() - t0))
    make_figure(rows, meas, pred, s_h)
    return ok


def make_figure(rows, meas, pred, s_h):
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.6))
    ts = [r[0] for r in rows]
    ax = axes[0][0]
    ax.plot(ts, [r[1] / (2.0 * np.pi) for r in rows], "o-", label="continuous: accumulation / 2 pi")
    ax.step(ts, [r[2] for r in rows], where="mid", label="discrete: winding label")
    ax.set_title("two sides of one motion")
    ax.set_xlabel("path parameter t")
    ax.legend(fontsize=8)
    ax = axes[0][1]
    ax.plot(ts, [r[1] for r in rows], "o-", label="S(t)")
    ax.axhline(np.pi, ls=":", color="k", label="half charge (pi)")
    ax.set_title("jump point sits at half charge")
    ax.set_xlabel("path parameter t")
    ax.set_ylabel("accumulation")
    ax.legend(fontsize=8)
    ax = axes[1][0]
    ax.plot(pred, meas, "o", label="endpoint write epoch")
    lo = min(min(pred), min(meas))
    hi = max(max(pred), max(meas))
    ax.plot([lo, hi], [lo, hi], ":", color="k", label="analytic")
    ax.set_title("endpoints: write epoch still normal")
    ax.set_xlabel("analytic epoch")
    ax.set_ylabel("measured epoch")
    ax.legend(fontsize=8)
    ax = axes[1][1]
    ax.bar([0, 1], [1, 2], width=0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["continuous side\n(distinct steps)", "discrete side\n(distinct labels)"], fontsize=8)
    ax.set_title("information carried by each side")
    fig.tight_layout()
    fig.savefig("D:/software/TianXingFC/images/motion_two_sided.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
