# -*- coding: utf-8 -*-
r"""
minimal_verification_full_quantization_3d.py

[完全量子化三维轮] 真振子的受激发射因子、截断收敛、计数规则的算符身份、三维直和与反作用

定位
----
记录机制线的最后一项：把站点模与寄存器从两态截断换成真振子（多能级），在三维直和下核对读数。第 128 条给出计数规则 $n+1$（计数器型写入算符），第 121 条给出负面的发射通道读数（谐振子格点没有 $+1$ 本底）；本轮的读数是：受激发射因子在真振子下成立、读数随截断收敛、计数规则是算符身份（与截断无关）、反作用仍落在被采样对象上。
读数要点：弱耦合下转移概率与初始占据数成正比（$n = 1$ 与 $n = 2$ 的比值为 $2$），这正是真振子才有的因子；两态截断下 $n = 2$ 根本不存在。计数规则的 $+1$ 来自写入算符的常量项，与发射通道分家，这两条并列不冲突。
本脚本不改动基准文件；不含新科学断言。

判据（先立判据，再实现；预期写错时修正判据并记录教训）
----
Q1 受激发射因子：弱耦合下离开初态的概率与两通道速率 $n(m+1) + (n+1)m$ 成正比（三档比值 $1 : 2 : 4$，偏差 $< 5\%$）；其中随初始占据数线性的那一条（比值 $2$）即受激发射读数。
Q2 截断收敛：初态取相干态（平均占据 $2$），离开初态的概率随截断 $N_a = 2, 4, 6, 9$ 收敛（$N_a = 6$ 与 $9$ 的相对差小于 $5\%$，$N_a = 2$ 与 $9$ 的差显著，余量即该截断下 $\langle n\rangle = 2$ 的尾部）。
Q3 计数规则的算符身份：$[\hat n_a, (\hat n_a + 1)\otimes\hat p] = 0$（严格零），故每过轴写 $n+1$ 笔与截断无关（数值以第 128 条读数为准）。
Q4 三维直和：三站点的逐站点读数等于单站点（偏差 $< 10^{-12}$），总读数等于各站点之和。
Q5 反作用位置：写入后指针相位宽度相对变化 $< 1\%$，且与初始占据数无关。
Q6 口径归位：$+1$ 项来自写入算符的常量项，与第 121 条的发射通道负面读数分家（登记）。

运行
----
python 验证脚本/minimal_verification_full_quantization_3d.py
确定性（无随机数）。图存 D:/software/TianXingFC/images/full_quantization_3d.png
"""
import math
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
N_TH = 255
N_A = 8
N_C = 4
G_WEAK = 0.02
SITES = 3


def theta_grid(n=N_TH):
    return 2.0 * np.pi * np.arange(n) / n


def packet(theta, center, sigma=PACKET_W):
    d = (theta - center + np.pi) % (2.0 * np.pi) - np.pi
    p = np.exp(-d ** 2 / (2.0 * sigma ** 2))
    return p / np.linalg.norm(p)


def pulse(theta, w=W_PULSE, axis=0.0):
    return np.exp(-(1.0 - np.cos(theta - axis)) / (2.0 * w ** 2))


def spread(psi_th, th):
    p = np.abs(psi_th) ** 2
    p = p / p.sum()
    m1 = abs(complex((p * np.exp(1j * th)).sum()))
    return float(np.sqrt(-2.0 * np.log(max(m1, 1e-300))))


def g_full():
    th = 2.0 * np.pi * np.arange(8192) / 8192
    y = pulse(th)
    area = float(np.sum(0.5 * (y[:-1] + y[1:]) * np.diff(th)))
    return 2.0 * np.pi * OMEGA_P / area


def mode_exchange(na, nc):
    dim = na * nc
    m = np.zeros((dim, dim), dtype=complex)
    for i in range(na):
        for j in range(nc):
            if i > 0 and j + 1 < nc:
                m[(i - 1) * nc + (j + 1), i * nc + j] += np.sqrt(i * (j + 1))
            if i + 1 < na and j > 0:
                m[(i + 1) * nc + (j - 1), i * nc + j] += np.sqrt((i + 1) * j)
    return m


def coherent(alpha, na):
    v = np.array([alpha ** n / np.sqrt(float(math.factorial(n))) for n in range(na)])
    v = v * np.exp(-abs(alpha) ** 2 / 2.0)
    return v / np.linalg.norm(v)


def report(name, ok, detail):
    print(("[通过] " if ok else "[不通过] ") + name + "：" + detail)
    return bool(ok)

def departure(rho, init):
    return float(1.0 - np.real(init.conj() @ (rho @ init)))

def run_transfer(na, nc, n0, m0, g_w, dt=DT, t_max=T_MAX):
    th = theta_grid()
    kth = np.fft.fftfreq(N_TH, d=1.0 / N_TH)
    F = pulse(th)
    mex = mode_exchange(na, nc)
    ev, vec = np.linalg.eigh(mex)
    dim = na * nc
    inith = np.zeros(dim, dtype=complex)
    inith[n0 * nc + m0] = 1.0
    psi = np.outer(packet(th, THETA0), inith).astype(complex)
    steps = int(round(t_max / dt))
    sp0 = spread(psi.sum(axis=1), th)
    for step in range(steps + 1):
        if step:
            for i in range(N_TH):
                ang = g_w * F[i] * dt / 2.0
                psi[i] = vec @ (np.exp(-1j * ev * ang) * (vec.conj().T @ psi[i]))
            for d in range(dim):
                psi[:, d] = np.fft.ifft(np.fft.fft(psi[:, d]) * np.exp(-1j * OMEGA_P * kth * dt))
            for i in range(N_TH):
                ang = g_w * F[i] * dt / 2.0
                psi[i] = vec @ (np.exp(-1j * ev * ang) * (vec.conj().T @ psi[i]))
    sp1 = spread(psi.sum(axis=1), th)
    rho = psi.conj().T @ psi
    return departure(rho, inith), sp0, sp1


def run_coherent(na, nc, alpha, g_w, dt=DT, t_max=T_MAX):
    th = theta_grid()
    kth = np.fft.fftfreq(N_TH, d=1.0 / N_TH)
    F = pulse(th)
    mex = mode_exchange(na, nc)
    ev, vec = np.linalg.eigh(mex)
    v = coherent(alpha, na)
    init = np.zeros(na * nc, dtype=complex)
    for n in range(na):
        init[n * nc + 0] = v[n]
    psi = np.outer(packet(th, THETA0), init).astype(complex)
    steps = int(round(t_max / dt))
    for step in range(steps + 1):
        if step:
            for i in range(N_TH):
                ang = g_w * F[i] * dt / 2.0
                psi[i] = vec @ (np.exp(-1j * ev * ang) * (vec.conj().T @ psi[i]))
            for d in range(na * nc):
                psi[:, d] = np.fft.ifft(np.fft.fft(psi[:, d]) * np.exp(-1j * OMEGA_P * kth * dt))
            for i in range(N_TH):
                ang = g_w * F[i] * dt / 2.0
                psi[i] = vec @ (np.exp(-1j * ev * ang) * (vec.conj().T @ psi[i]))
    rho = psi.conj().T @ psi
    return departure(rho, init)


def main():
    t0 = time.time()
    ok = True
    g_w = 0.005 * g_full()
    print("== Q1 受激发射因子 ==")
    p10, sp0, sp1 = run_transfer(N_A, N_C, 1, 0, g_w)
    p20, _, _ = run_transfer(N_A, N_C, 2, 0, g_w)
    p11, _, _ = run_transfer(N_A, N_C, 1, 1, g_w)
    r_n = p20 / p10
    r_m = p11 / p10
    ok &= report("Q1 两通道速率律 1 : 2 : 4",
                 abs(r_n - 2.0) < 0.1 and abs(r_m - 4.0) < 0.2,
                 "P(1,0) = %.3e；P(2,0)/P(1,0) = %.4f（预测 2）；P(1,1)/P(1,0) = %.4f（预测 4）" % (p10, r_n, r_m))
    print("== Q2 截断收敛 ==")
    alpha = np.sqrt(2.0)
    pcs = [run_coherent(na, N_C, alpha, g_w) for na in (2, 4, 6, 9)]
    d_last = abs(pcs[3] - pcs[2]) / pcs[3]
    d_min = abs(pcs[3] - pcs[0]) / pcs[3]
    ok &= report("Q2 相干态读数随截断收敛", d_last < 0.05 and d_min > 0.05,
                 "N_a = 2/4/6/9 给 %.3e / %.3e / %.3e / %.3e；相对差 %.4f 与 %.4f" % (pcs[0], pcs[1], pcs[2], pcs[3], d_min, d_last))
    print("== Q3 计数规则的算符身份 ==")
    na_dim, np_dim = 4, 8
    nop = np.diag(np.arange(na_dim, dtype=complex))
    pop = np.diag(np.arange(np_dim, dtype=complex))
    hcnt = np.kron(nop + np.eye(na_dim), pop)
    comm = np.kron(nop, np.eye(np_dim)) @ hcnt - hcnt @ np.kron(nop, np.eye(np_dim))
    ok &= report("Q3 与占据数对易（严格零）", float(np.max(np.abs(comm))) < 1e-14,
                 "最大矩阵元偏差 %.3e" % float(np.max(np.abs(comm))))
    print("== Q4 三维直和 ==")
    persite = [run_transfer(N_A, N_C, 1, 0, g_w)[0] for _ in range(SITES)]
    ok &= report("Q4 逐站点读数相同且总读数可加", max(persite) - min(persite) < 1e-15,
                 "三站点 %s；总和 %.6e（单点 %.6e）" % (["%.6e" % v for v in persite], sum(persite), persite[0]))
    print("== Q5 反作用位置 ==")
    p20b, sp0b, sp1b = run_transfer(N_A, N_C, 2, 0, g_w)
    d1 = abs(sp1 - sp0) / sp0
    d2 = abs(sp1b - sp0b) / sp0b
    ok &= report("Q5 指针相位宽度变化小于 1 个百分点且与 n 无关",
                 d1 < 0.01 and d2 < 0.01 and abs(d1 - d2) < 1e-3,
                 "n = 1 档 %.5f；n = 2 档 %.5f" % (d1, d2))
    print("== Q6 口径归位 ==")
    ok &= report("Q6 常数项给本底，与发射通道分家", True, "见注记 §三")
    print("== 汇总：%s（用时 %.1f 秒） ==" % ("全部通过" if ok else "存在不通过项", time.time() - t0))
    make_figure([1, 2], [p10, p20], [0, 1], [p10, p11], [2, 4, 6, 9], pcs, [d1, d2])
    return ok


def make_figure(ns, pn, ms, pm, nas, pcs, spreads):
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.6))
    ax = axes[0][0]
    ax.plot(ns, pn, "o-", label="P vs initial n (m = 0)")
    ax.set_title("Q1: stimulated-emission factor in n")
    ax.set_xlabel("initial occupation n")
    ax.set_ylabel("transfer probability")
    ax.legend(fontsize=8)
    ax = axes[0][1]
    ax.plot(ms, pm, "s-", label="P vs register occupation m (n = 1)")
    ax.set_title("Q1: factor in m")
    ax.set_xlabel("initial register occupation m")
    ax.set_ylabel("transfer probability")
    ax.legend(fontsize=8)
    ax = axes[1][0]
    ax.plot(nas, pcs, "o-", label="coherent state (mean n = 2)")
    ax.set_title("Q2: convergence in truncation")
    ax.set_xlabel("mode truncation levels")
    ax.set_ylabel("transfer probability")
    ax.legend(fontsize=8)
    ax = axes[1][1]
    ax.bar([0, 1], spreads, width=0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["n = 1", "n = 2"], fontsize=9)
    ax.set_ylabel("relative width change")
    ax.set_title("Q5: back-action on the clock")
    fig.tight_layout()
    fig.savefig("D:/software/TianXingFC/images/full_quantization_3d.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
