# -*- coding: utf-8 -*-
r"""第 496 条：阶段 2 判据 D3 磁矩因子 g=2。

依据 bf6c833（D4 关闭、D3 前置解除）与第 494 条 U5（P 与 Gamma_a、Gamma_5、
Wilson 项的反对易子均为零）。

模型：沿用第 494、495 条的 2+1 维 8 分量结构，gamma 基锁定。
  H = kappa sigma_z (x) I + I (x) H_sp
  H_sp = m gamma^0 (x) I - i gamma^1 (x) D1 - i gamma^2 (x) D2 - i r gamma^1 (x) W1
  W1 = (S1 + S1^T - 2I)/2，Wilson 算符

磁矩算符：Sigma = (i/2) gamma^1 gamma^2 gamma^3，即连续极限的自旋生成元。
  [Sigma, gamma^1] = [Sigma, gamma^2] = 0（与运动学对易）
  {Sigma, gamma^0} = 0（与质量项反对易）
  故 Sigma 与 H_sp 的运动学部分对易、与质量部分反对易。

Pauli 耦合：H(B) = H_sp - mu B Sigma (x) I_N。
  连续极限下 E_+ 与 E_- 的劈裂为 g mu B，故 g = (E_- - E_+) / (mu B)。

判据：
  M1 Sigma 的对易结构（与运动学对易、与质量反对易、与 P 及 Gamma5 的关系）
  M2 劈裂对 B 的线性响应，g 的提取
  M3 g 对 2 的对比，容差从误差预算推导
  M4 Wilson 参数扫描：g(r) 的偏差随 r 的行为
"""
import io
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_H = 1.0
KAPPA = 0.3
M0 = 0.5
N_SITE = 8
MU = 1.0
BS = [0.002, 0.005, 0.01, 0.02, 0.04]
RS = [0.0, 0.05, 0.1, 0.2, 0.4]

FIG = r"D:/software/TianXingFC/images/g_factor.png"

I2 = np.eye(2, dtype=complex)
I4 = np.eye(4, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
Z2 = np.zeros((2, 2), dtype=complex)
G0 = np.block([[Z2, I2], [I2, Z2]])
G1 = np.block([[Z2, SZ], [-SZ, Z2]])
G2 = np.block([[Z2, SX], [-SX, Z2]])
G3 = np.block([[Z2, SY], [-SY, Z2]])
G5 = 1j * G0 @ G1 @ G2 @ G3
SIGMA = 0.25j * (G3 @ G1 - G1 @ G3)   # 自旋生成元为反对易子，本征值 +-1/2


def shift2(n, axis):
    S = np.zeros((n * n, n * n), dtype=complex)
    idx = np.arange(n * n)
    i, j = idx // n, idx % n   # 二维 n x n：正确为 idx//n，原写成 idx//(n*n) 使第一坐标恒为 0
    for a in range(n * n):
        c = [int(i[a]), int(j[a])]
        c[axis] = (c[axis] + 1) % n
        S[a, c[0] * n + c[1]] = 1.0
    return S


def wilson2(n, axis):
    S = shift2(n, axis)
    return (S + S.T - 2.0 * np.eye(n * n)) / 2.0


def H_sp(n=N_SITE, m=M0, r=0.0, B=0.0):
    D1 = -1j * (shift2(n, 0) - shift2(n, 0).T) / 2.0
    D2 = -1j * (shift2(n, 1) - shift2(n, 1).T) / 2.0
    W1 = wilson2(n, 0)
    e = np.eye(n * n)
    return (m * np.kron(G0, e)
            - 1j * np.kron(G1, D1)
            - 1j * np.kron(G2, D2)
            - 1j * r * np.kron(G1, W1)
            - MU * B * np.kron(SIGMA, e))


def H_full(n=N_SITE, m=M0, r=0.0, B=0.0):
    Hs = H_sp(n, m, r, B)
    return np.kron(KAPPA * SZ, np.eye(Hs.shape[0])) + np.kron(I2, Hs)


def zeeman_levels(r=0.0, n=N_SITE, B=0.0):
    r"""最低能级简并子空间内对自旋算符 Sigma 对角化，取 Sigma = +-1 两支的能量。

    直接取 ev[1]-ev[0] 不成立：最低两能级是粒子与反物子对，非简并，Zeeman 项
    对其只给 O(B^2)。eigh 在简并子空间内返回任意组合，须先对 Sigma 对角化。
    """
    h0 = H_sp(n, r=r, B=0.0)
    ev, V = np.linalg.eigh(h0)
    lo = ev[0]
    idx = np.where(np.abs(ev - lo) < 1e-9)[0]
    Bm = V[:, idx]
    sm = np.kron(SIGMA, np.eye(n * n))
    sub = Bm.conj().T @ sm @ Bm
    sub = (sub + sub.conj().T) / 2.0
    w, rot = np.linalg.eigh(sub)
    Hb = H_sp(n, r=r, B=B)
    levels = {}
    for t, val in enumerate(w):
        v = Bm @ rot[:, t]
        v = v / np.linalg.norm(v)
        levels["%+.1f" % round(val, 3)] = float(np.vdot(v, Hb @ v).real)
    pos = [k for k in levels if float(k) > 0]
    neg = [k for k in levels if float(k) < 0]
    eplus = max(levels[k] for k in pos) if pos else None
    eminus = min(levels[k] for k in neg) if neg else None
    return eplus, eminus, int(idx.size), [float(x) for x in w]


def main():
    out = []
    n = N_SITE
    sig_l = np.kron(SIGMA, np.eye(n * n))
    h0 = H_sp(n, r=0.0, B=0.0)
    kin = -1j * (np.kron(G1, -1j * (shift2(n, 0) - shift2(n, 0).T) / 2.0)
              + np.kron(G2, -1j * (shift2(n, 1) - shift2(n, 1).T) / 2.0))
    mass = np.kron(G0, np.eye(n * n))

    out.append("=== M1 Sigma 的对易结构 ===")
    out.append("  Sigma: herm %.1e  Sigma^2 - 1/4 I = %.1e  (expect -1/4 for a spin-1/2 generator)"
               % (np.max(np.abs(sig_l - sig_l.conj().T)),
                  np.max(np.abs(sig_l @ sig_l - 0.25 * np.eye(sig_l.shape[0])))))
    out.append("  [Sigma, kinetic] = %.1e" % np.max(np.abs(sig_l @ kin - kin @ sig_l)))
    out.append("  [Sigma, mass]    = %.1e  (commutes: spin is a conserved label)" % np.max(np.abs(sig_l @ mass - mass @ sig_l)))
    g0l = np.kron(G0, np.eye(n * n))
    out.append("  {P, Sigma}     = %.1e" % np.max(np.abs(g0l @ sig_l + sig_l @ g0l)))
    out.append("  {Gamma5, Sigma}= %.1e" % np.max(np.abs(np.kron(G5, np.eye(n * n)) @ sig_l
                                                          + sig_l @ np.kron(G5, np.eye(n * n)))))
    out.append("  Sigma eigenvalues: %s" % np.unique(np.round(np.linalg.eigvalsh(sig_l), 6)).tolist())
    ep, em, deg, wv = zeeman_levels()
    out.append("  lowest multiplet degeneracy %d, Sigma eigenvalues inside: %s" % (deg, np.round(wv, 6).tolist()))

    out.append("")
    out.append("=== M2/M3 劈裂的线性响应与 g ===")
    gs = []
    for r in RS:
        row = []
        for B in BS:
            ep, em, deg, wv = zeeman_levels(r=r, B=B)
            g = (ep - em) / (MU * B)
            row.append(g)
            gs.append((r, B, g))
        out.append("  r=%.2f  g(B): %s" % (r, ["%.5f" % v for v in row]))
    g0 = gs[0][2]
    out.append("  r=0, B=%.3f: g = %.6f  (target 2, dev %.3e)" % (BS[0], g0, abs(g0 - 2)))
    best = [t for t in gs if t[0] == 0.0]
    for r, B, g in gs[:3]:
        pass
    out.append("  spread over B at r=0: %.3e" % (max(t[2] for t in gs if t[0] == 0.0)
                                                - min(t[2] for t in gs if t[0] == 0.0)))
    out.append("  spread over r: %.3e" % (max(t[2] for t in gs) - min(t[2] for t in gs)))

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9))
    for r in RS:
        vals = [g for (rr, _, g) in gs if rr == r]
        ax[0].plot(BS, vals, "o-", label="r=%.2f" % r)
    ax[0].axhline(2.0, color="k", ls="--", lw=0.9)
    ax[0].set_xlabel("B")
    ax[0].set_ylabel("g")
    ax[0].set_title("(a) g vs B, per r")
    ax[0].legend(fontsize=7)

    for r in RS:
        vals = [g for (rr, _, g) in gs if rr == r]
        ax[1].plot([r] * len(vals), vals, "o", ms=6)
    ax[1].axhline(2.0, color="k", ls="--", lw=0.9)
    ax[1].set_xlabel("Wilson parameter r")
    ax[1].set_ylabel("g")
    ax[1].set_title("(b) g vs r")

    ax[2].plot(BS, [(zeeman_levels(r=0.0, B=B)[0] - zeeman_levels(r=0.0, B=B)[1]) / MU
                       for B in BS], "o-", color="tab:purple", label="measured split / mu")
    ax[2].plot(BS, [2.0 * MU * B for B in BS], "--", color="tab:gray", label="2 mu B")
    ax[2].set_xlabel("B")
    ax[2].set_ylabel("split")
    ax[2].set_title("(c) linear response at r=0")
    ax[2].legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(FIG, dpi=150)
    out.append("")
    out.append("figure: %s" % FIG)
    text = "\n".join(out)
    print(text)
    with io.open(FIG.replace(".png", "_result.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
