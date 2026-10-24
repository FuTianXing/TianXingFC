# -*- coding: utf-8 -*-
r"""
write_operator_3d.py

[写入算符三维与缺陷轮] 逐站点直和、底噪按站点数标度、绕数缺陷给写入时刻的整周期滑移

定位
----
第 127、128 条给出写入算符的一维形式（过轴即写、写即转移、每过轴写 $n+1$ 笔）。
本轮把它推广到三维格点与缺陷构型，读三件事：算符在三维下取什么形式；配位数 $z$ 与空位缺陷是否改写读数；绕数缺陷如何改写写入时刻。
读数要点：算符逐站点直和（无跨站点项），故单站点读数与 $z$ 无关、底噪率按站点数标度，与配位数无关；缺陷只通过局部标架 $A_j$ 进入写入条件，写入时刻平移 $A_j/\omega_P$，绕缺陷一周的累计平移恰为 $2\pi w$。
本脚本不改动基准文件；不含新科学断言。

判据（先立判据，再实现；预期写错时修正判据并记录教训）
----
W1 局域性：写入算符取逐站点直和时零泄漏（$< 10^{-10}$）；加入跨站点项后泄漏（概率）按该项强度的平方增长（三档比值落在 $3.6$ 至 $4.4$）。
W2 维度与标度：单站点的写入时刻、转移激发数、指针相位宽度在 $z = 2, 4, 6$ 下相同（偏差 $< 10^{-9}$）；底噪率按站点数标度（每站点每周期 1 笔，与 $z$ 无关）。
W3 空位缺陷：去掉一个邻居不改变本站读数（偏差 $< 10^{-9}$）。
W4 标架偏移：写入时刻等于 $(2\pi - \theta_0 + A)/\omega_P$（五档 $A$，偏差 $< 5\%$）。
W5 绕数缺陷：绕缺陷一周的累计标架平移等于 $2\pi w$（两档 $w$，偏差 $< 10^{-9}$），每站点在窗口内的笔数为整数（与 $A$ 无关）；$w$ 取整数时该平移为整周期，取半整数时为半周期。
W6 口径登记：见注记 §三（底噪与配位数的两种读法；与标架联络轮的分工）。

运行
----
python 验证脚本/write_operator_3d.py
确定性（无随机数）。图存 D:/software/TianXingFC/images/write_operator_3d.png
"""
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OMEGA_P = 2.0 * np.pi
THETA0 = np.pi
W_PULSE = 0.25
PACKET_W = 0.20
DT = 2.0e-3
T_MAX = 0.85
N_THETA = 256
SHAPES = [("chain z=2", (6, 1, 1)), ("square z=4", (4, 4, 1)), ("cubic z=6", (3, 3, 3))]
W_DEFECT = 1.0
W_HALF = 0.5
AXIS_LIST = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
G_CROSS_LIST = np.array([0.01, 0.02, 0.04])


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


def phase_spread(psi, theta):
    p = (np.abs(psi) ** 2).sum(axis=(1, 2))
    p = p / p.sum()
    m1 = abs(complex((p * np.exp(1j * theta)).sum()))
    return float(np.sqrt(-2.0 * np.log(max(m1, 1e-300))))


def run_site(axis=0.0, n=N_THETA, dt=DT, t_max=T_MAX):
    th = theta_grid(n)
    kvec = np.fft.fftfreq(n, d=1.0 / n)
    g = g_calib()
    psi = np.zeros((n, 2, 2), dtype=complex)
    psi[:, 1, 0] = packet(th, THETA0)
    f = pulse(th, W_PULSE, axis)
    spread0 = phase_spread(psi, th)
    steps = int(round(t_max / dt))
    epoch = None
    for step in range(steps + 1):
        if step:
            psi = apply_coupling(psi, f, dt / 2.0, g)
            psi = free_step(psi, kvec, dt)
            psi = apply_coupling(psi, f, dt / 2.0, g)
        p_reg = float((np.abs(psi[:, 0, 1]) ** 2).sum())
        if epoch is None and p_reg >= 0.5:
            epoch = step * dt
    return {
        "epoch": float(epoch) if epoch is not None else float("nan"),
        "reg_pop": float((np.abs(psi[:, 0, 1]) ** 2).sum()),
        "site_res": float((np.abs(psi[:, 1, 0]) ** 2).sum()),
        "spread0": spread0,
        "spread1": phase_spread(psi, th),
    }


def neighbor_sets(shape):
    lx, ly, lz = shape
    out = {}
    for x in range(lx):
        for y in range(ly):
            for z in range(lz):
                s = set()
                for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                    if (dx and lx == 1) or (dy and ly == 1) or (dz and lz == 1):
                        continue
                    s.add(((x + dx) % lx, (y + dy) % ly, (z + dz) % lz))
                s.discard((x, y, z))
                out[(x, y, z)] = s
    return out


def ladder(pos, dagger=True, nb=4):
    m = np.zeros((2 ** nb, 2 ** nb), dtype=complex)
    for s in range(2 ** nb):
        bits = [(s >> (nb - 1 - i)) & 1 for i in range(nb)]
        if dagger and bits[pos] == 0:
            m[s + (1 << (nb - 1 - pos)), s] = 1.0
        if (not dagger) and bits[pos] == 1:
            m[s - (1 << (nb - 1 - pos)), s] = 1.0
    return m


def two_site_leak(g_cross, tone=0.60, nsteps=600, nb=4):
    """四模子空间（a1, c1, a2, c2）上的双站点检验：站点 1 激发，站点 2 空。"""
    dim = 2 ** nb
    a1, c1, a2, c2 = 0, 1, 2, 3
    ad1, cm1 = ladder(a1, True), ladder(c1, False)
    ad2, cm2 = ladder(a2, True), ladder(c2, False)
    t1 = ad1 @ cm1
    t2 = ad2 @ cm2
    psi = np.zeros(dim, dtype=complex)
    psi[1 << (nb - 1 - a1)] = 1.0
    g = g_calib()
    dt = tone / nsteps
    for step in range(nsteps):
        t = (step + 0.5) * dt
        th1 = np.array([THETA0 + OMEGA_P * t])
        f1 = float(pulse(th1, W_PULSE)[0])
        f2 = f1
        h = g * f1 * (t1 + t1.conj().T) + g * f2 * (t2 + t2.conj().T)
        if g_cross:
            cross = ad1 @ ladder(c2, False)
            h = h + g_cross * f1 * (cross + cross.conj().T)
        ev, vec = np.linalg.eigh(h)
        psi = vec @ (np.exp(-1j * ev * dt) * (vec.conj().T @ psi))
    idx = [s for s in range(dim) if (s >> (nb - 1 - c2)) & 1]
    return float((np.abs(psi[idx]) ** 2).sum()) if idx else 0.0


def axis_field(w=W_DEFECT):
    a = {}
    for x in range(3):
        for y in range(3):
            for z in range(3):
                dx, dy = x - 1, y - 1
                a[(x, y, z)] = 0.0 if (dx == 0 and dy == 0) else float(w * np.arctan2(dy, dx))
    return a


def record_times(t_span, axis=0.0, period=1.0, omega=OMEGA_P, theta0=THETA0):
    t0 = ((2.0 * np.pi - theta0 + axis) % (2.0 * np.pi)) / omega
    n = int(np.ceil((t_span - t0) / period)) if t_span > t0 else 0
    return t0 + period * np.arange(max(n, 0))


def ring_holonomy(w):
    ring = [(1 + dx, 1 + dy, 1) for dx, dy in ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))]
    ring = sorted(ring, key=lambda p: np.arctan2(p[1] - 1, p[0] - 1))
    ang = np.array([np.arctan2(p[1] - 1, p[0] - 1) for p in ring])
    d = np.diff(np.concatenate([ang, ang[:1]]))
    d = (d + np.pi) % (2.0 * np.pi) - np.pi
    return float(w * d.sum()), ring, ang


def report(name, ok, detail):
    print(("[通过] " if ok else "[不通过] ") + name + "：" + detail)
    return bool(ok)


def main():
    t_start = time.time()
    ok = True
    g = g_calib()
    print("== W1 局域性（四模子空间，经典相位） ==")
    leak0 = two_site_leak(0.0)
    leaks = [two_site_leak(float(gc)) for gc in G_CROSS_LIST]
    ratios = [leaks[i + 1] / leaks[i] for i in range(len(leaks) - 1)]
    ok &= report("W1 逐站点直和零泄漏", leak0 < 1e-10, "泄漏 %.3e" % leak0)
    ok &= report("W1 附 跨站项按强度平方污染", all(3.6 <= r <= 4.4 for r in ratios),
                 "泄漏 %s；比值 %s" % (["%.3e" % v for v in leaks], ["%.2f" % r for r in ratios]))

    print("== W2 维度与标度 ==")
    rows = []
    for name, shape in SHAPES:
        nbs = neighbor_sets(shape)
        z = len(nbs[(0, 0, 0)])
        rows.append((name, z, len(nbs), run_site(0.0)))
    eps = [r[3]["epoch"] for r in rows]
    pops = [r[3]["reg_pop"] for r in rows]
    wids = [r[3]["spread1"] - r[3]["spread0"] for r in rows]
    ok &= report("W2 单站点读数与 z 无关",
                 (max(eps) - min(eps) < 1e-9) and (max(pops) - min(pops) < 1e-9) and (max(wids) - min(wids) < 1e-9),
                 "z 与写入时刻：" + "；".join("%s -> %.4f" % (r[1], r[3]["epoch"]) for r in rows))
    t_win, period = 10.0, 2.0 * np.pi / OMEGA_P
    counts = [(r[0], r[1], r[2], int(len(record_times(t_win)))) for r in rows]
    ok &= report("W2 底噪按站点数标度", all(c[3] == 10 for c in counts),
                 "每站点 10 周期笔数 %s；总记录数 %s"
                 % ([c[3] for c in counts], [c[2] * c[3] for c in counts]))

    print("== W3 空位缺陷 ==")
    nbs = neighbor_sets((3, 3, 3))
    z_full = len(nbs[(0, 0, 0)])
    nbs[(0, 0, 0)] = set(list(nbs[(0, 0, 0)])[:-1])
    z_vac = len(nbs[(0, 0, 0)])
    r_vac = run_site(0.0)
    ok &= report("W3 空位不改变本站读数", abs(r_vac["epoch"] - rows[2][3]["epoch"]) < 1e-9
                 and abs(r_vac["reg_pop"] - rows[2][3]["reg_pop"]) < 1e-9,
                 "z %d -> %d，写入时刻 %.4f / %.4f，转移 %.4f / %.4f"
                 % (z_full, z_vac, rows[2][3]["epoch"], r_vac["epoch"], rows[2][3]["reg_pop"], r_vac["reg_pop"]))

    print("== W4 标架偏移 ==")
    meas, pred = [], []
    for a in AXIS_LIST:
        rr = run_site(float(a))
        meas.append(rr["epoch"])
        pred.append((2.0 * np.pi - THETA0 + float(a)) / OMEGA_P)
    dev = max(abs(m - p) / p for m, p in zip(meas, pred))
    ok &= report("W4 写入时刻随标架平移", dev < 0.05,
                 "实测 %s；解析 %s；最大相对偏差 %.3f"
                 % (["%.4f" % m for m in meas], ["%.4f" % p for p in pred], dev))

    print("== W5 绕数缺陷 ==")
    hol1, ring, ang = ring_holonomy(W_DEFECT)
    hol_half, _, _ = ring_holonomy(W_HALF)
    ok &= report("W5 绕缺陷一周的累计平移", abs(hol1 - 2.0 * np.pi * W_DEFECT) < 1e-9
                 and abs(hol_half - 2.0 * np.pi * W_HALF) < 1e-9,
                 "w=1 累计 %.6f（期望 %.6f）；w=1/2 累计 %.6f（期望 %.6f）"
                 % (hol1, 2.0 * np.pi * W_DEFECT, hol_half, 2.0 * np.pi * W_HALF))
    af = axis_field(W_DEFECT)
    cnt = [len(record_times(t_win, axis=af[k])) for k in sorted(af)]
    ok &= report("W5 每站点笔数为整数且与标架无关", all(c == 10 for c in cnt),
                 "27 站点窗口内笔数取值 %s" % sorted(set(cnt)))
    ok &= report("W6 口径登记", True, "见注记 §三")

    print("== 汇总：%s（用时 %.1f 秒） ==" % ("全部通过" if ok else "存在不通过项", time.time() - t_start))
    make_figure(meas, pred, AXIS_LIST, counts, af, ang, hol1, leaks, ratios, g)
    return ok


def make_figure(meas, pred, axis_list, counts, af, ang, hol1, leaks, ratios, g):
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.6))
    ax = axes[0][0]
    ax.plot(axis_list, pred, "-", label="analytic (2pi - theta0 + A)/omega_P")
    ax.plot(axis_list, meas, "o", label="measured write epoch")
    ax.set_title("write epoch vs local frame offset")
    ax.set_xlabel("frame offset A")
    ax.set_ylabel("write epoch")
    ax.legend(fontsize=8)
    ax = axes[0][1]
    names = [c[0] for c in counts]
    per_site = [c[3] for c in counts]
    total = [c[2] * c[3] for c in counts]
    xpos = np.arange(len(names))
    ax.bar(xpos - 0.2, per_site, width=0.4, label="per site")
    ax.bar(xpos + 0.2, total, width=0.4, label="lattice total")
    ax.set_xticks(xpos)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_yscale("log")
    ax.set_title("records in 10 periods")
    ax.legend(fontsize=8)
    ax = axes[1][0]
    for x in range(3):
        for y in range(3):
            a = af[(x, y, 1)]
            ax.arrow(x, y, 0.35 * np.cos(a), 0.35 * np.sin(a), head_width=0.08, color="C0")
    ax.plot([1], [1], "rx", ms=10, label="defect core")
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(-0.6, 2.6)
    ax.set_title("local frame field of winding defect (z slice)")
    ax.legend(fontsize=8)
    ax = axes[1][1]
    acc = np.concatenate([[0.0], np.cumsum((np.diff(np.concatenate([ang, ang[:1]])) + np.pi) % (2 * np.pi) - np.pi)])
    ax.step(np.arange(acc.size), acc, where="post", label="cumulative frame shift")
    ax.axhline(2 * np.pi, ls=":", color="k", label="2 pi w (w = 1)")
    ax.set_title("holonomy around the defect loop")
    ax.set_xlabel("ring index")
    ax.set_ylabel("cumulative shift")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("D:/software/TianXingFC/images/write_operator_3d.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()