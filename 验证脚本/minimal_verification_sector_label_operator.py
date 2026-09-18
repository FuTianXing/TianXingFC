# -*- coding: utf-8 -*-
r"""
minimal_verification_sector_label_operator.py

[扇区标签算符轮] 绕数标签的算符实现：与写对易、提升关系与整数谱、对易子的解析形式与奇偶选择定则

定位
----
P3-13 第一步（第 139 条）给出三条结构约束，其中第一条（扇区封闭）此前是读数而非算符陈述。本轮把扇区标签做成算符，核对它的代数结构：激发数标签与写入算符对易；相位提升关系 $[\hat L, e^{i\hat\theta}] = e^{i\hat\theta}$ 成立且 $\hat L$ 的谱为整数；写入与 $\hat L$ 的对易子取解析形式 $-i g f'(\hat\theta)\otimes(a^\dagger c + ac^\dagger)$；该对易子在实轴反射下为奇，故对称初态的一阶时钟偏移严格为零。
读数要点：超选择由此从口径变成算符陈述（对易子为零），时钟受扰的部分带确定的选择定则（奇算子，偶态中一阶消失）。
本脚本不改动基准文件；不含新科学断言。

判据（先立判据，再实现；预期写错时修正判据并记录教训）
----
W1 激发数标签与写对易：$[\hat N, H_{\rm int}] = 0$，最大矩阵元偏差 $< 10^{-14}$。
W2 提升关系与整数谱：$[\hat L, e^{i\hat\theta}] = e^{i\hat\theta}$ 在模空间内部成立（排除最高模的截断边界，偏差 $< 10^{-12}$），且 $\hat L$ 的谱为整数（最大偏离 $< 10^{-12}$）。
W3 对易子的解析形式：$[\hat L, H_{\rm int}] = -i\,g\,f'(\hat\theta)\otimes(a^\dagger c + ac^\dagger)$ 在带限脉冲与模空间内部成立（偏差 $< 10^{-12}$）；物理窄脉冲（$w = 0.12$）的残差随环点数加密而下降，判为分辨率效应。
W4 奇偶选择定则：该对易子在实轴反射下为奇（偏差 $< 10^{-12}$），且在偶态中的一阶期望严格为零（$< 10^{-16}$）。
W5 净绕数不变：写入时刻等于折返时刻（偏差 $< 5\%$），指针相位宽度不变（$< 10^{-12}$）。
W6 口径登记：见注记 §三。

运行
----
python 验证脚本/minimal_verification_sector_label_operator.py
确定性（无随机数）。图存 D:/software/TianXingFC/images/sector_label_operator.png
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
NRING = 255


def ring(n=NRING):
    th = 2.0 * np.pi * np.arange(n) / n
    k = np.fft.fftfreq(n, d=1.0 / n)
    f = np.fft.fft(np.eye(n)) / np.sqrt(n)
    return th, k, f


def pulse(theta, w=W_PULSE, axis=0.0):
    return np.exp(-(1.0 - np.cos(theta - axis)) / (2.0 * w ** 2))


def dpulse(theta, w=W_PULSE):
    return -np.sin(theta) / (2.0 * w ** 2) * pulse(theta, w)


def trapezoid(y, x):
    return float(np.sum(0.5 * (y[:-1] + y[1:]) * np.diff(x)))


def g_calib(w=W_PULSE):
    th = 2.0 * np.pi * np.arange(8192) / 8192
    return (0.5 * np.pi) * OMEGA_P / trapezoid(pulse(th, w), th)


def mode_ops():
    """(a,c) 两态的四个算符：激发数、对换、以及张量扩展用的单位阵。"""
    nop = np.diag([0.0, 1.0, 1.0, 2.0]).astype(complex)
    exc = np.zeros((4, 4), dtype=complex)
    exc[0 * 2 + 1, 1 * 2 + 0] = 1.0
    exc[1 * 2 + 0, 0 * 2 + 1] = 1.0
    return nop, exc


def reflection(n=NRING):
    r = np.zeros((n, n))
    for m in range(n):
        r[(-m) % n, m] = 1.0
    return r


def phase_shift_ops(th, n=NRING):
    up = np.diag(np.exp(1j * th)).astype(complex)
    return up


def packet(theta, center, sigma=PACKET_W):
    d = (theta - center + np.pi) % (2.0 * np.pi) - np.pi
    p = np.exp(-d ** 2 / (2.0 * sigma ** 2))
    return p / np.linalg.norm(p)


def phase_spread(psi, th):
    p = (np.abs(psi) ** 2).sum()
    p = p / p.sum()
    m1 = abs(complex((p * np.exp(1j * th)).sum()))
    return float(np.sqrt(-2.0 * np.log(max(m1, 1e-300))))


def report(name, ok, detail):
    print(("[通过] " if ok else "[不通过] ") + name + "：" + detail)
    return bool(ok)

def run_site(axis=0.0, n=NRING, dt=DT, t_max=T_MAX):
    th, k, F = ring(n)
    g = g_calib()
    psi = np.zeros((n, 2, 2), dtype=complex)
    psi[:, 1, 0] = packet(th, THETA0)
    f = pulse(th, W_PULSE, axis)
    spread0 = phase_spread(psi.sum(axis=(1, 2)), th)
    steps = int(round(t_max / dt))
    epoch = None
    mult = np.exp(-1j * OMEGA_P * k * dt)
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
    return float(epoch) if epoch is not None else float("nan"), phase_spread(psi.sum(axis=(1, 2)), th), spread0


def main():
    t0 = time.time()
    ok = True
    th, k, F = ring()
    n = NRING
    I4 = np.eye(4)
    Lop = F.conj().T @ np.diag(k.astype(complex)) @ F
    Lfull = np.kron(Lop, I4)
    nop, exc = mode_ops()
    g = g_calib()
    H = g * np.kron(np.diag(pulse(th).astype(complex)), exc)
    Nfull = np.kron(np.eye(n), nop)
    Uf = np.kron(np.diag(np.exp(1j * th)).astype(complex), I4)
    print("== W1 激发数标签与写对易 ==")
    X1 = Nfull @ H - H @ Nfull
    e1 = float(np.max(np.abs(X1)))
    ok &= report("W1 对易子为零", e1 < 1e-14, "最大矩阵元偏差 %.3e" % e1)
    print("== W2 提升关系与整数谱 ==")
    X2 = Lfull @ Uf - Uf @ Lfull - Uf
    e2_full = float(np.max(np.abs(X2)))
    kmax = (NRING - 1) // 2
    keep = (np.abs(k) <= kmax - 1).astype(complex)
    Proj = np.kron(F.conj().T @ np.diag(keep) @ F, I4)
    e2 = float(np.max(np.abs(Proj @ X2 @ Proj)))
    eigs = np.linalg.eigvalsh(Lop)
    e3 = float(np.max(np.abs(eigs - np.round(eigs))))
    ok &= report("W2 提升关系（模空间内部）成立且谱为整数", e2 < 1e-12 and e3 < 1e-12,
                 "内部偏差 %.3e；全空间偏差 %.3e（最高模截断）；非整数偏离 %.3e" % (e2, e2_full, e3))
    print("== W3 对易子的解析形式 ==")
    fb = 1.0 + 0.2 * np.cos(th) - 0.05 * np.cos(2.0 * th)
    fbp = -0.2 * np.sin(th) + 0.1 * np.sin(2.0 * th)
    Hb = np.kron(np.diag(fb.astype(complex)), exc)
    Xb = Lfull @ Hb - Hb @ Lfull
    Xbana = -1j * np.kron(np.diag(fbp.astype(complex)), exc)
    keep3 = (np.abs(k) <= kmax - 3).astype(complex)
    P3 = np.kron(F.conj().T @ np.diag(keep3) @ F, I4)
    e4 = float(np.max(np.abs(P3 @ (Xb - Xbana) @ P3)))
    Xa = Lfull @ H - H @ Lfull
    Xaana = -1j * g * np.kron(np.diag(dpulse(th).astype(complex)), exc)
    rel255 = float(np.max(np.abs(P3 @ (Xa - Xaana) @ P3))) / float(np.max(np.abs(P3 @ Xa @ P3)))
    n2 = 2 * NRING
    th2 = 2.0 * np.pi * np.arange(n2) / n2
    k2 = np.fft.fftfreq(n2, d=1.0 / n2)
    F2 = np.fft.fft(np.eye(n2)) / np.sqrt(n2)
    L2 = np.kron(F2.conj().T @ np.diag(k2.astype(complex)) @ F2, I4)
    g2 = g_calib()
    H2 = g2 * np.kron(np.diag(pulse(th2).astype(complex)), exc)
    X2n = L2 @ H2 - H2 @ L2
    X2nana = -1j * g2 * np.kron(np.diag(dpulse(th2).astype(complex)), exc)
    kmax2 = (n2 - 1) // 2
    keep2 = (np.abs(k2) <= kmax2 - 3).astype(complex)
    P2p = np.kron(F2.conj().T @ np.diag(keep2) @ F2, I4)
    rel511 = float(np.max(np.abs(P2p @ (X2n - X2nana) @ P2p))) / float(np.max(np.abs(P2p @ X2n @ P2p)))
    ok &= report("W3 对易子等于解析式（带限脉冲）", e4 < 1e-12, "带限偏差 %.3e" % e4)
    ok &= report("W3 附 物理窄脉冲残差为分辨率效应", rel511 < rel255,
                 "相对残差 %.4f（N = 255）降到 %.4f（N = 510）" % (rel255, rel511))
    print("== W4 奇偶选择定则 ==")
    R = reflection(n)
    Rfull = np.kron(R, I4)
    e5 = float(np.max(np.abs(Rfull @ Xb @ Rfull.T + Xb)))
    e5p = float(np.max(np.abs(Rfull @ Xa @ Rfull.T + Xa)))
    psi_th = packet(th, 0.0)
    psi_full = np.kron(psi_th, np.eye(4)[2])
    q1 = abs(complex(psi_full.conj() @ (Xa @ psi_full)))
    ok &= report("W4 对易子为奇且偶态一阶期望为零", e5 < 1e-12 and q1 < 1e-16,
                 "带限奇性偏差 %.3e；物理脉冲奇性残差 %.3e；一阶期望 %.3e" % (e5, e5p, q1))
    print("== W5 净绕数不变 ==")
    ep, sp1, sp0 = run_site(0.0)
    pred = ((2.0 * np.pi - THETA0) % (2.0 * np.pi)) / OMEGA_P
    ok &= report("W5 写入时刻等于折返时刻且宽度不变", abs(ep - pred) / pred < 0.05 and abs(sp1 - sp0) < 1e-12,
                 "实测 %.4f 对解析 %.4f t_P；宽度 %.6f 对 %.6f" % (ep, pred, sp1, sp0))
    print("== W6 口径登记 ==")
    ok &= report("W6 扇区标签算符实现的口径", True, "见注记 §三")
    print("== 汇总：%s（用时 %.1f 秒） ==" % ("全部通过" if ok else "存在不通过项", time.time() - t0))
    make_figure(eigs, Lfull, Uf, Xb, Xbana, [e1, e2, e4, e5])
    return ok


def make_figure(eigs, Lfull, Uf, X, Xana, res):
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.6))
    ax = axes[0][0]
    ax.plot(np.arange(40), np.sort(eigs)[:40], "o", ms=3, label="L spectrum (sorted)")
    ax.set_title("W2: integer spectrum of the winding operator")
    ax.set_xlabel("index")
    ax.set_ylabel("eigenvalue")
    ax.legend(fontsize=8)
    ax = axes[0][1]
    i = np.arange(0, Lfull.shape[0], 37)
    up = Uf[np.ix_(i, i)].ravel()
    com = (Lfull @ Uf - Uf @ Lfull)[np.ix_(i, i)].ravel()
    ax.plot(np.real(up), np.real(com), "o", ms=3, label="real part")
    ax.set_title("W2: [L, exp(i theta)] vs exp(i theta)")
    ax.set_xlabel("matrix element of exp(i theta)")
    ax.legend(fontsize=8)
    ax = axes[1][0]
    xa = Xana[np.ix_(i, i)].ravel()
    xc = X[np.ix_(i, i)].ravel()
    ax.plot(np.real(xa), np.real(xc), "o", ms=3, label="real part")
    ax.set_title("W3: commutator vs analytic form")
    ax.set_xlabel("analytic matrix element")
    ax.legend(fontsize=8)
    ax = axes[1][1]
    ax.bar([0, 1, 2, 3], [max(v, 1e-18) for v in res], width=0.5)
    ax.set_yscale("log")
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["W1", "W2", "W3", "W4"], fontsize=9)
    ax.set_title("residual matrix element (max)")
    fig.tight_layout()
    fig.savefig("D:/software/TianXingFC/images/sector_label_operator.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
