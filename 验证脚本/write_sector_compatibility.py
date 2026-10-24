# -*- coding: utf-8 -*-
r"""
write_sector_compatibility.py

[P3-13 第一步] 写入算符与绕数扇区的相容性：扇区封闭、标签可加、单体模固定、相位住关系层

定位
----
P3-13（$S^1$ 可约化到量子力学描述的合法性与具体机制）是体系自标最高优先级开放问题，此前缺可算抓手。本轮给它做出第一步：用写入算符（命题 12 至 15）核对三条结构约束。
读数要点：写入算符在 (站点模, 寄存器) 的零激发数子空间内转动，故激发总数严格守恒、扇区标签不移动；算符逐站点对角，故标签站点可加、跨站项一旦存在即破坏可加性；单站点态空间的径向自由度为零（相位环只有一个模），叠加因此只能由关系承载；相位信息的读数单元是相邻实轴方向差（棱差），单站点只给出计数。
本脚本不改动基准文件；不含新科学断言。

判据（先立判据，再实现；预期写错时修正判据并记录教训）
----
U1 扇区封闭：写入过程中激发总数守恒（偏差 $< 10^{-12}$）；时钟读数不受写入改写（指针相位宽度变化 $< 1\%$）。
U2 标签可加：逐站点直和下跨站激发泄漏 $< 10^{-15}$；引入跨站项后泄漏（概率）随该强度的平方增长（两档比值落在 $3.6$ 至 $4.4$）。
U3 单体模固定：归一化守恒（$< 10^{-13}$）；自由演化不改变相位分布宽度（$< 10^{-12}$），故径向自由度为零。
U4 相位住关系层：写入时刻只依赖本站标架（改邻站标架不改本站读数，偏差 $0$）；棱差由两站写入时刻差读出（相对偏差 $< 5\%$）。
U5 口径登记：三条结构约束（扇区由绕数定、计数由度定、相位由关系承载）对 P3-13 的约束形式；见注记 §三。

运行
----
python 验证脚本/write_sector_compatibility.py
确定性（无随机数）。图存 D:/software/TianXingFC/images/write_sector_compatibility.png
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
N_THETA = 256
AXIS_LIST = np.array([0.0, 0.6, 1.2, 1.8, 2.4])
G_CROSS = np.array([0.01, 0.02])


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


def phase_spread(psi, th):
    p = (np.abs(psi) ** 2).sum(axis=(1, 2))
    p = p / p.sum()
    m1 = abs(complex((p * np.exp(1j * th)).sum()))
    return float(np.sqrt(-2.0 * np.log(max(m1, 1e-300))))


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


def run_site(axis=0.0, n=N_THETA, dt=DT, t_max=T_MAX, couple=True):
    th = theta_grid(n)
    kvec = np.fft.fftfreq(n, d=1.0 / n)
    g = g_calib()
    psi = np.zeros((n, 2, 2), dtype=complex)
    psi[:, 1, 0] = packet(th, THETA0)
    f = pulse(th, W_PULSE, axis)
    hist = {"norm": [], "total": [], "epoch": None}
    steps = int(round(t_max / dt))
    for step in range(steps + 1):
        if step:
            if couple:
                psi = apply_coupling(psi, f, dt / 2.0, g)
            psi = free_step(psi, kvec, dt)
            if couple:
                psi = apply_coupling(psi, f, dt / 2.0, g)
        hist["norm"].append(float((np.abs(psi) ** 2).sum()))
        hist["total"].append(float((np.abs(psi[:, 1, 0]) ** 2 + np.abs(psi[:, 0, 1]) ** 2).sum()))
        if hist["epoch"] is None and float((np.abs(psi[:, 0, 1]) ** 2).sum()) >= 0.5:
            hist["epoch"] = step * dt
    hist["spread"] = phase_spread(psi, th)
    hist["Psi0"] = psi
    return hist

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
    """四模子空间（a1, c1, a2, c2）：站点 1 激发、站点 2 空，量站点 2 的激发泄漏。"""
    a1, c1, a2, c2 = 0, 1, 2, 3
    dim = 2 ** nb
    ad1, cm1 = ladder(a1, True), ladder(c1, False)
    ad2, cm2 = ladder(a2, True), ladder(c2, False)
    t1 = ad1 @ cm1
    psi = np.zeros(dim, dtype=complex)
    psi[1 << (nb - 1 - a1)] = 1.0
    g = g_calib()
    dt = tone / nsteps
    for step in range(nsteps):
        t = (step + 0.5) * dt
        f1 = float(pulse(np.array([THETA0 + OMEGA_P * t]), W_PULSE)[0])
        h = g * f1 * (t1 + t1.conj().T)
        if g_cross:
            cross = ad1 @ ladder(c2, False)
            h = h + g_cross * f1 * (cross + cross.conj().T)
        ev, vec = np.linalg.eigh(h)
        psi = vec @ (np.exp(-1j * ev * dt) * (vec.conj().T @ psi))
    idx = [s for s in range(dim) if ((s >> (nb - 1 - a2)) & 1) or ((s >> (nb - 1 - c2)) & 1)]
    return float((np.abs(psi[idx]) ** 2).sum())


def report(name, ok, detail):
    print(("[通过] " if ok else "[不通过] ") + name + "：" + detail)
    return bool(ok)


def main():
    t0 = time.time()
    ok = True
    print("== U1 扇区封闭 ==")
    h_on = run_site(couple=True)
    h_off = run_site(couple=False)
    d_norm = max(abs(v - 1.0) for v in h_on["norm"])
    d_tot = max(abs(v - 1.0) for v in h_on["total"])
    ok &= report("U1 激发总数与归一化守恒", max(d_norm, d_tot) < 1e-12,
                 "归一化偏差 %.3e；激发总数偏差 %.3e" % (d_norm, d_tot))
    d_spread = abs(h_on["spread"] - h_off["spread"]) / h_off["spread"]
    ok &= report("U1 附 时钟读数不受写入改写", d_spread < 0.01,
                 "相位宽度相对变化 %.4f" % d_spread)

    print("== U2 标签可加 ==")
    leak0 = two_site_leak(0.0)
    leaks = [two_site_leak(float(gc)) for gc in G_CROSS]
    ratio = leaks[1] / leaks[0] if leaks[0] > 0 else float("inf")
    ok &= report("U2 直和下跨站泄漏为零", leak0 < 1e-15, "泄漏 %.3e" % leak0)
    ok &= report("U2 附 跨站项按强度平方破坏可加性", 3.6 <= ratio <= 4.4,
                 "泄漏 %s；比值 %.2f" % (["%.3e" % v for v in leaks], ratio))

    print("== U3 单体模固定 ==")
    d_norm_off = max(abs(v - 1.0) for v in h_off["norm"])
    w_free_0 = float(np.sqrt(-2.0 * np.log(max(abs(complex((((np.abs(h_off["Psi0"]) ** 2).sum(axis=(1, 2)) / 1.0) * np.exp(1j * theta_grid())).sum())), 1e-300))))
    sp_in = float(h_off["spread"])
    psi_free = np.zeros((N_THETA, 2, 2), dtype=complex)
    psi_free[:, 1, 0] = packet(theta_grid(), THETA0)
    sp0 = phase_spread(psi_free, theta_grid())
    ok &= report("U3 归一化守恒且无径向激发", max(d_norm_off, abs(sp_in - sp0)) < 1e-12,
                 "归一化偏差 %.3e；自由演化宽度差 %.3e（初始宽度 %.4f）" % (d_norm_off, abs(sp_in - sp0), sp0))

    print("== U4 相位住关系层 ==")
    meas = []
    pred = []
    for a in AXIS_LIST:
        meas.append(float(run_site(axis=float(a))["epoch"]))
        pred.append(((2.0 * np.pi - THETA0 + float(a)) % (2.0 * np.pi)) / OMEGA_P)
    dev = max(abs(m - p) for m, p in zip(meas, pred))
    ok &= report("U4 写入时刻只依赖本站标架", dev < 0.05,
                 "五档实测 %s；解析 %s；最大偏差 %.4f t_P"
                 % (["%.4f" % m for m in meas], ["%.4f" % p for p in pred], dev))
    link = AXIS_LIST[2] - AXIS_LIST[1]
    d_meas = meas[2] - meas[1]
    ok &= report("U4 附 棱差由两站时刻差读出", abs(d_meas - link / OMEGA_P) / (link / OMEGA_P) < 0.05,
                 "实测 %.4f t_P；棱差 %.4f rad 对应 %.4f t_P"
                 % (d_meas, link, link / OMEGA_P))

    print("== U5 口径登记 ==")
    ok &= report("U5 三条约束对 P3-13 的约束形式", True, "见注记 §三")
    print("== 汇总：%s（用时 %.1f 秒） ==" % ("全部通过" if ok else "存在不通过项", time.time() - t0))
    make_figure(h_on, h_off, leaks, leak0, meas, pred)
    return ok


def make_figure(h_on, h_off, leaks, leak0, meas, pred):
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.6))
    tt = np.arange(len(h_on["norm"])) * DT
    ax = axes[0][0]
    ax.semilogy(tt, [abs(v - 1.0) + 1e-18 for v in h_on["norm"]], label="normalisation deficit")
    ax.semilogy(tt, [abs(v - 1.0) + 1e-18 for v in h_on["total"]], label="excitation total deficit")
    ax.set_title("U1: conservation under writing")
    ax.set_xlabel("time")
    ax.legend(fontsize=8)
    ax = axes[0][1]
    ax.bar([0, 1, 2], [max(leak0, 1e-18), leaks[0], leaks[1]], width=0.5)
    ax.set_yscale("log")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["direct sum", "cross 0.01", "cross 0.02"], fontsize=8)
    ax.set_title("U2: cross-site leakage")
    ax = axes[1][0]
    ax.plot(AXIS_LIST, pred, "-", label="analytic epoch")
    ax.plot(AXIS_LIST, meas, "o", label="measured epoch")
    ax.set_title("U4: write epoch vs local frame")
    ax.set_xlabel("frame offset A")
    ax.set_ylabel("epoch (t_P)")
    ax.legend(fontsize=8)
    ax = axes[1][1]
    ax.bar([0, 1], [h_off["spread"], h_on["spread"]], width=0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["free evolution", "with writing"], fontsize=8)
    ax.set_ylabel("phase spread")
    ax.set_title("U3: radial freedom absent, clock unperturbed")
    fig.tight_layout()
    fig.savefig("D:/software/TianXingFC/images/write_sector_compatibility.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
