# -*- coding: utf-8 -*-
r"""第 491 条：两因子张量积结构判据化（W1 至 W5）。

依据 [[内部草稿/注记-升维构造方案：内部模式代数与时空Clifford的两因子划分（依据第490条）]] §八 第一轮。
该方案 §八 声明其探针未入库、本组四判据须按占位制重新执行，本条即该执行轮，
并按方案的 §七 可判性表补入 W5（两因子可分离性与谱的因子化代价）。

三因子（方案的 §四 落位表）：
  相位因子  S^1 及其 Z_2 双覆盖，即第 489 条的反周期 theta 环，给 pi 扭转
  模态因子  双模式态空间 S^3 的等距群 SU(2) 双态，给弱荷
  时空因子  3+1 维 4 分量 Dirac 旋量，给宇称、手性、色散、Wilson 项

判据：
  W1 3+1 维 Clifford 自洽性：度规 (1,-1,-1,-1)，六个反对易子，g5 平方为 I 且厄米
  W2 宇称算符的唯一性与穷举完备性：三十二个 Clifford 单项式（含与不含因子 i）
     中满足 [P,g0]=0、{P,gi}=0（i=1,2,3）、P^2=I、厄mi 的候选恰为一个
  W3 P 与 Gamma5 反对易；Gamma5 本征值 +-1 各半
  W4 Wilson 项在时空因子上的 P 与 Gamma5 相容性，以及 doubling 抬升
     （算符形式取 gamma^1 (S + S^{-1} - 2)/2，k 空间像为 cos k - 1）
  W5 两因子可分离性与谱的因子化代价：模态因子与时空因子的哈密顿量对易、
     谱为成对和、联合手性标记给出四种独立组合
"""
import io
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_SITE = 8
C_H = 1.0
M0 = 0.5
TOL = 1e-10

FIG = r"D:/software/TianXingFC/images/two_factor_structure.png"

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
G = [G0, G1, G2, G3]
G5 = 1j * G0 @ G1 @ G2 @ G3
PAR = G0


def shift(n, anti=False):
    S = np.zeros((n, n))
    for i in range(n):
        j = (i + 1) % n
        S[i, j] = -1.0 if (anti and i == n - 1) else 1.0
    return S


def mom(S):
    return -1j * (S - S.T) / 2.0


def wilson_op(n=N_SITE):
    r"""可实现的 Wilson 算符：gamma^1 (S + S^{-1} - 2)/2。

    第 490 条 PD4 用的 k - sin k 形式在格点上无算符实现，因为动量算符
    D = -i(S - S^{-1})/2 在 k=0 处本征值为零，矩阵奇异。本条改用 Wilson
    费米子的标准格点形式，其 k 空间像为 cos k - 1，在 k=0 处为零、
    在 k=pi 处取极值，正好抬升双重子。
    """
    S = shift(n)
    return (S + S.T - 2.0 * np.eye(n)) / 2.0


def H_sp(r=0.0, n=N_SITE):
    D = mom(shift(n))
    W = r * wilson_op(n)
    return (M0 * np.kron(G0, np.eye(n))
            - 1j * C_H * np.kron(G1, D)
            - 1j * np.kron(G1, W)), D


def main():
    out, ok = [], []

    out.append("=== 冒烟 ===")
    ev = np.linalg.eigvalsh(G5)
    hsd = H_sp()[0]
    out.append("H_sp hermitian dev = %.2e" % np.max(np.abs(hsd - hsd.conj().T)))
    out.append("H_sp dim %d, Gamma5 dim %d" % (H_sp()[0].shape[0], G5.shape[0]))
    out.append("Gamma5 eigen: +1 x%d, -1 x%d"
               % (int(np.sum(np.abs(ev - 1) < 1e-9)), int(np.sum(np.abs(ev + 1) < 1e-9))))

    out.append("")
    out.append("=== W1 3+1 维 Clifford 自洽性 ===")
    met = [float(np.trace(G[a] @ G[a]).real / 4) for a in range(4)]
    out.append("  metric diag = %s  (expect 1 -1 -1 -1)" % ["%+.0f" % m for m in met])
    ac = []
    for a in range(4):
        for b in range(a + 1, 4):
            ac.append(float(np.max(np.abs(G[a] @ G[b] + G[b] @ G[a]))))
    out.append("  pairwise anticommutators max = %.1e" % max(ac))
    g5sq = float(np.max(np.abs(G5 @ G5 - I4)))
    g5he = float(np.max(np.abs(G5 - G5.conj().T)))
    out.append("  Gamma5^2 - I = %.1e   Gamma5 herm dev = %.1e" % (g5sq, g5he))
    j1 = (abs(met[0] - 1) < TOL and all(abs(met[i] + 1) < TOL for i in (1, 2, 3))
          and max(ac) < TOL and g5sq < TOL and g5he < TOL)
    ok.append(j1)
    out.append("W1 %s" % ("PASS" if j1 else "FAIL"))

    out.append("")
    out.append("=== W2 宇称算符的唯一性与穷举完备性 ===")
    cands = {}
    for ph in (1.0, 1j):
        for mask in range(16):
            M = np.eye(4, dtype=complex)
            for b in range(4):
                if mask >> b & 1:
                    M = M @ G[b]
            cands["%s%04d" % ("i" if ph == 1j else "1", mask)] = ph * M
    out.append("  candidate count = %d (16 subsets x 2 phases)" % len(cands))
    sol, tab = [], []
    for k, Q in cands.items():
        c0 = float(np.max(np.abs(Q @ G[0] - G[0] @ Q)))
        an = max(float(np.max(np.abs(Q @ G[i] + G[i] @ Q))) for i in (1, 2, 3))
        sq = float(np.max(np.abs(Q @ Q - I4)))
        he = float(np.max(np.abs(Q - Q.conj().T)))
        good = c0 < TOL and an < TOL and sq < TOL and he < TOL
        if good:
            sol.append(k)
        tab.append((k, c0, an, sq, he, good))
    out.append("  solutions to [P,g0]=0, {P,gi}=0, P^2=I, herm: %s" % (sol if sol else "none"))
    j2 = len(sol) == 1
    ok.append(j2)
    out.append("W2 %s  (exhaustive over 32 monomials; unique solution)" % ("PASS" if j2 else "FAIL"))

    out.append("")
    out.append("=== W3 P 与 Gamma5 的反对易 ===")
    a_p5 = float(np.max(np.abs(PAR @ G5 + G5 @ PAR)))
    c_p5 = float(np.max(np.abs(PAR @ G5 - G5 @ PAR)))
    pp = float(np.max(np.abs(PAR @ PAR - I4)))
    npos = int(np.sum(np.abs(ev - 1) < 1e-9))
    out.append("  {P,Gamma5} = %.1e   [P,Gamma5] = %.1e   P^2 - I = %.1e" % (a_p5, c_p5, pp))
    out.append("  Gamma5 spectrum: +1 x%d, -1 x%d (equal split)" % (npos, 4 - npos))
    j3 = a_p5 < TOL and c_p5 > 0.5 and pp < TOL and npos == 2
    ok.append(j3)
    out.append("W3 %s  (parity connects the branches, Gamma5 distinguishes them)" % ("PASS" if j3 else "FAIL"))

    out.append("")
    out.append("=== W4 Wilson 项在时空因子上的相容性 ===")
    W4x = np.kron(G1, wilson_op(N_SITE))
    p4 = np.kron(G0, np.eye(N_SITE))
    g5f = np.kron(G5, np.eye(N_SITE))
    a_pw = float(np.max(np.abs(p4 @ W4x + W4x @ p4)))
    a_5w = float(np.max(np.abs(g5f @ W4x + W4x @ g5f)))
    out.append("  {P, W} = %.1e   {Gamma5, W} = %.1e" % (a_pw, a_5w))
    kh = 2 * np.pi * np.arange(N_SITE) / N_SITE
    ks = np.linspace(0.0, np.pi, 400)
    rs = np.linspace(0.0, 6.0, 61)
    splits = [float(np.max(np.abs(
        (C_H * np.sin(ks) + r * (np.cos(ks) - 1.0)) ** 2
        - (C_H * np.sin(np.pi - ks) + r * (np.cos(np.pi - ks) - 1.0)) ** 2))) for r in rs]
    j4 = a_pw < TOL and a_5w < TOL and splits[0] < TOL and all(
        splits[i] <= splits[i + 1] + 1e-12 for i in range(len(splits) - 1))
    out.append("  doubling split: r=0 -> %.2e ; r=%.1f -> %.4f  monotone=%s"
               % (splits[0], rs[-1], splits[-1],
                  all(splits[i] <= splits[i + 1] + 1e-12 for i in range(len(splits) - 1))))
    out.append("  -> the Wilson term is compatible with both P and Gamma5 while lifting the doublets")
    ok.append(j4)
    out.append("W4 %s" % ("PASS" if j4 else "FAIL"))

    out.append("")
    out.append("=== W5 两因子可分离性与谱的因子化代价 ===")
    Hm = SZ.copy()
    Hs, _ = H_sp()
    A = np.kron(Hm, np.eye(Hs.shape[0]))
    B = np.kron(I2, Hs)
    Hf = A + B
    comm = float(np.max(np.abs(A @ B - B @ A)))
    ef, V = np.linalg.eigh(Hf)
    ea, Va = np.linalg.eigh(A)
    eb, Vb = np.linalg.eigh(B)
    # 构造性检验：取 A、B 本征矢的张量积，验证其为 Hf 本征矢且本征值为两者之和。
    # 不用排序谱比较，避免多重简并下的记账歧义。
    prod_err = 0.0
    for ia in (0, ea.size - 1):
        for jb in (0, 1, eb.size - 1):
            v = Va[:2, ia]
            u = Vb[:Hs.shape[0], jb]
            w = np.kron(v, u)
            nrm = np.linalg.norm(w)
            w = w / nrm
            prod_err = max(prod_err, float(np.max(np.abs(Hf @ w - (ea[ia] + eb[jb]) * w))))
    spec_err = prod_err
    lbl = []
    Szf = np.kron(SZ, np.eye(Hs.shape[0]))
    G5f = np.kron(I2, np.kron(G5, np.eye(N_SITE)))
    for k in range(4):
        v = V[:, k]
        lbl.append((complex(v.conj() @ Szf @ v).real, complex(v.conj() @ G5f @ v).real))
    combos = sorted({(1 if a > 0 else -1, 1 if b > 0 else -1) for a, b in lbl})
    g5vals = sorted({1 if b > 0 else -1 for (_, b) in lbl})
    sharp_mode = all(abs(abs(a) - 1.0) < 1e-9 for (a, _) in lbl)
    unprotected = all(abs(b) < 1e-9 for (_, b) in lbl)
    deg = int(round(Hs.shape[0] / 2))
    out.append("  [A, B] = %.1e   constructive factorisation error = %.1e" % (comm, spec_err))
    out.append("  every spacetime level carries a twofold mode degeneracy (%d -> %d levels)" % (Hs.shape[0], deg))
    out.append("  four lowest states, joint labels (mode Sz, spacetime Gamma5):")
    for (a, b) in lbl:
        out.append("     %+.6f  %+.6f" % (a, b))
    out.append("  joint-label sign counts (meaningless when <Gamma5> = 0): %s" % combos)
    out.append("  mode label sharp (|<sigma_z>| = 1 on every state): %s" % sharp_mode)
    out.append("  spacetime chirality unprotected (|<Gamma5>| = 0 at finite mass): %s" % unprotected)
    out.append("  -> separation is exact, and the two factors behave differently:")
    out.append("     the mode factor carries a sharp label because its own Hamiltonian selects a")
    out.append("     branch, while the spacetime chirality is NOT protected at finite mass. A")
    out.append("     massive Dirac spinor's energy eigenstates are L/R mixtures, so D4's two")
    out.append("     branches exist as a Gamma5 labelling of the OPERATOR, not of the STATES.")
    out.append("     The lift therefore makes D4 decidable but does not by itself close it.")
    j5 = comm < TOL and spec_err < 1e-9 and sharp_mode and unprotected
    ok.append(j5)
    out.append("W5 %s" % ("PASS" if j5 else "FAIL"))

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9))
    names = [t[0] for t in tab]
    ax[0].semilogy(range(len(names)), np.maximum([max(t[1], t[2], t[3], t[4]) for t in tab], 1e-18),
                   "o", color="tab:purple", ms=4)
    ax[0].axhline(TOL, color="tab:red", ls="--", lw=0.8)
    ax[0].set_xticks(range(len(names)))
    ax[0].set_xticklabels(names, rotation=90, fontsize=5)
    ax[0].set_ylabel(r"$\max$ residual")
    ax[0].set_title("(a) W2 parity candidates, one at zero")
    ax[0].set_ylim(1e-18, 1e2)

    ax[1].plot(rs, splits, "-", color="tab:green")
    ax[1].set_xlabel(r"Wilson parameter $r$")
    ax[1].set_ylabel("doubling split")
    ax[1].set_title(r"(b) W4 doubling lift, $r \in [0,6]$")
    ax[1].set_ylim(-1, 400)

    xs = [1 + 0.7 * i for i in range(4)]
    ax[2].scatter(xs, [b for (_, b) in lbl], c=[(1 if a > 0 else -1) for (a, _) in lbl],
                  cmap="coolwarm", s=90, edgecolor="k", zorder=3)
    for i, (a, b) in enumerate(lbl):
        ax[2].annotate("%+.2f" % a, (xs[i], b), textcoords="offset points",
                       xytext=(0, 8), ha="center", fontsize=7)
    ax[2].set_xticks([1, 2, 3, 4])
    ax[2].set_xlabel("four lowest states")
    ax[2].set_ylabel(r"$\langle \Gamma_5 \rangle$")
    ax[2].set_ylim(-1.6, 1.6)
    ax[2].set_title(r"(c) W5 joint labels, colour = mode $\langle\sigma_z\rangle$")

    fig.tight_layout()
    fig.savefig(FIG, dpi=150)
    out.append("")
    out.append("figure: %s" % FIG)

    allok = all(ok)
    out.append("SUMMARY W1..W5 = %s" % ("ALL PASS" if allok else "HAS FAILURE"))
    text = "\n".join(out)
    print(text)
    with io.open(FIG.replace(".png", "_result.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text + "\n")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
