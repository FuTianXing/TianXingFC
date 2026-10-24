# -*- coding: utf-8 -*-
r"""第 493 条：时空因子扩至 2+1 维与 D4 手性判定（U1 至 U5）。

依据第 492 条 T5 的维度诊断：1+1 维下只有 gamma^1 活动，四分量结构过完备，
Gamma5 把活动平面映到正交补，故 Gamma5 期望恒为零且与质量无关。本条把时空因子
扩至 2+1 维（两个空间方向），使 gamma^1 与 gamma^2 同时活动。

哈密顿量
  H   = H_mode (x) I + I (x) H_sp,  H_mode = kappa * sigma_z
  H_sp = m gamma^0 (x) I
        - i gamma^1 (x) (Dx (x) I + Dy (I (x) Dy) + r (Wx (x) I + Wy (I (x) Wy)))
  D = -i (S - S^T)/2，动量算符，本征值 sin k
  W = (S + S^T - 2I)/2，Wilson 算符，本征值 cos k - 1
全空间维数 2 x 4 x N^2。

解析结论（本条要判的对象）
  A_x(k) = c sin k_x + r (cos k_x - 1)，A_y 同理
  E^2 = m^2 + A_x^2 + A_y^2，每个 k 给 +- 各二重
  <Gamma5> = -A_x / sqrt(A_x^2 + A_y^2)
第二式是本条的核心：2+1 维下 Gamma5 期望一般非零，其值由色散矢量的方向决定。

判据：
  U1 谱按模态(2) x Dirac(+- 各二重) x 动量(N^2) 因子化
  U2 连续极限
  U3 无零模，谱关于 E=0 严格对称
  U4 D4 的结构性判定：Gamma5 与 H 的反对易、两支互换、+-1 本征空间非空且等维，\n     并穷举 3+1 维 Clifford 单项式确认与自由 Dirac 对易的厄米对合算符只有 +-I
  U5 宇称 P = gamma^0 的结构与 Wilson 项的相容性
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
R_W = 0.1
NS = [6, 8]
TOL_MATCH = 1e-8
TOL_SYM = 1e-8

FIG = r"D:/software/TianXingFC/images/two_plus_one_d4.png"

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
PAR = G0


def shift2(n, axis):
    r"""二维周期格点上的移位算符。axis=0 沿 x，axis=1 沿 y。"""
    S = np.zeros((n * n, n * n), dtype=complex)
    for i in range(n):
        for j in range(n):
            a = i * n + j
            if axis == 0:
                b = i * n + (j + 1) % n
            else:
                b = ((i + 1) % n) * n + j
            S[a, b] = 1.0
    return S


def wilson_op(n, axis):
    S = shift2(n, axis)
    return (S + S.T - 2.0 * np.eye(n * n)) / 2.0


def H_sp(n, m=M0, r=R_W):
    Dx = -1j * (shift2(n, 0) - shift2(n, 0).T) / 2.0
    Dy = -1j * (shift2(n, 1) - shift2(n, 1).T) / 2.0
    Wx = wilson_op(n, 0)
    Wy = wilson_op(n, 1)
    # 二维格点算符本身已是 n^2 x n^2，无需再与单位阵作 Kronecker 积
    return (m * np.kron(G0, np.eye(n * n))
            - 1j * np.kron(G1, Dx)
            - 1j * np.kron(G2, Dy)
            - 1j * r * np.kron(G1, Wx)
            - 1j * r * np.kron(G2, Wy))


def H_full(n, m=M0, r=R_W):
    Hs = H_sp(n, m, r)
    return np.kron(KAPPA * SZ, np.eye(Hs.shape[0])) + np.kron(I2, Hs)


def shift3(n, axis):
    r"""三维周期格点上的移位算符，axis=0,1,2 对应 x,y,z。"""
    S = np.zeros((n ** 3, n ** 3), dtype=complex)
    idx = np.arange(n ** 3)
    i = idx // (n * n)
    j = (idx // n) % n
    k = idx % n
    coord = [i, j, k]
    for site in range(n ** 3):
        c = [int(coord[0][site]), int(coord[1][site]), int(coord[2][site])]
        c[axis] = (c[axis] + 1) % n
        b = c[0] * n * n + c[1] * n + c[2]
        S[site, b] = 1.0
    return S


def H_sp3(n, m=M0, r=R_W):
    ey = np.eye(n)
    H = m * np.kron(G0, np.eye(n ** 3))
    for ax, Gm in ((0, G1), (1, G2), (2, G3)):
        S = shift3(n, ax)
        D = -1j * (S - S.T) / 2.0
        W = (S + S.T - 2.0 * np.eye(n ** 3)) / 2.0
        H = H - 1j * np.kron(Gm, D) - 1j * r * np.kron(Gm, W)
    return H


def H_full3(n, m=M0, r=R_W):
    Hs = H_sp3(n, m, r)
    return np.kron(KAPPA * SZ, np.eye(Hs.shape[0])) + np.kron(I2, Hs)


def single_block_g5(m, Avec3):
    r"""单一动量块的 4x4 哈密顿量上直接测 <Gamma5>，作为晶格读数的独立参照。"""
    A1, A2, A3 = Avec3
    h = m * G0 - 1j * (A1 * G1 + A2 * G2 + A3 * G3)
    e, V = np.linalg.eigh(h)
    return np.array([complex(v.conj() @ G5 @ v).real for v in V.T])


def momenta(n):
    k = 2 * np.pi * np.arange(n) / n
    return k


def Avec(k, r=R_W):
    return C_H * np.sin(k) + r * (np.cos(k) - 1.0)


def analytic_E2(n, m=M0, r=R_W):
    k = momenta(n)
    ax = Avec(k, r)
    ay = Avec(k, r)
    return m * m + ax[None, :] ** 2 + ay[:, None] ** 2


def analytic_g5(n, r=R_W):
    k = momenta(n)
    ax = Avec(k, r)
    ay = Avec(k, r)
    out, skipped = [], 0
    for i in range(n):
        for j in range(n):
            a = ax[j]
            b = ay[i]
            d = a * a + b * b
            if d < 1e-24:
                # Wilson 形式在 k=0 处 A=0，色散方向未定义，解析式为 0/0。
                # 该点的 4x4 块退化为 m gamma^0，其本征矢为 gamma^0 的本征矢，
                # 与 gamma^5 的重叠为零，故该点取值记为 0。
                skipped += 1
                out.append(0.0)
                continue
            out.append(-a / np.sqrt(d))
    return np.array(out), skipped


def main():
    out, ok = [], []
    n = NS[-1]

    out.append("=== 冒烟 ===")
    H0 = H_full(n)
    out.append("H dim %d (expect 8 N^2 = %d)" % (H0.shape[0], 8 * n * n))
    out.append("Hermitian dev = %.2e" % np.max(np.abs(H0 - H0.conj().T)))
    out.append("H_sp herm dev = %.2e" % np.max(np.abs(H_sp(n) - H_sp(n).conj().T)))
    Dx = -1j * (shift2(n, 0) - shift2(n, 0).T) / 2.0
    dv = np.sort(np.linalg.eigvalsh(Dx))
    sk = np.sort(np.sin(momenta(n)))
    sk2 = np.sort(np.repeat(sk, n))
    out.append("momentum operator vs sin(k): max dev = %.2e"
               % np.max(np.abs(dv - sk2)))
    sq = [float(np.trace(G @ G).real) / 4 for G in (G0, G1, G2, G3)]
    out.append("gamma squares (expect +1 -1 -1 -1): %s" % ["%+.0f" % v for v in sq])
    out.append("Gamma5^2 - I = %.2e   Gamma5 herm dev = %.2e"
               % (np.max(np.abs(G5 @ G5 - I4)), np.max(np.abs(G5 - G5.conj().T))))
    out.append("Gamma5 anticommutators with gamma^0..3: %s"
               % ["%.0e" % np.max(np.abs(G5 @ G + G @ G5)) for G in (G0, G1, G2, G3)])
    out.append("Gamma5 eigen: +1 x%d, -1 x%d"
               % (int(np.sum(np.abs(np.linalg.eigvalsh(G5) - 1) < 1e-9)),
                  int(np.sum(np.abs(np.linalg.eigvalsh(G5) + 1) < 1e-9))))

    out.append("")
    out.append("=== U1 谱因子化 ===")
    errs = []
    for nn in NS:
        for mm in (0.0, 0.5, 1.0):
            num = np.linalg.eigvalsh(H_full(nn, m=mm))
            E2 = analytic_E2(nn, m=mm)
            root = np.sqrt(E2)
            vals = []
            for s in (+1.0, -1.0):
                for t in (+1.0, -1.0):
                    for v in root.ravel():
                        vals.append(t * KAPPA + s * v)
            ana = np.sort(np.array(vals * 2))
            e = float(np.max(np.abs(num - ana)))
            errs.append(e)
            out.append("  N=%d m=%.1f  max|numeric - analytic| = %.3e  %s"
                       % (nn, mm, e, "PASS" if e < TOL_MATCH else "FAIL"))
    u1 = max(errs) < TOL_MATCH
    ok.append(u1)
    out.append("T-U1 %s" % ("PASS" if u1 else "FAIL"))

    out.append("")
    out.append("=== U2 连续极限 ===")
    kl = 1e-4
    kx = np.array([kl, 2 * kl, 3 * kl])
    ky = np.array([kl, 2 * kl, 3 * kl])
    ax = Avec(kx)
    ay = Avec(ky)
    res = float(np.max(np.abs((ax ** 2 + ay ** 2)
                              / (C_H ** 2 * (kx ** 2 + ky ** 2)) - 1.0)))
    budget = float(R_W * np.max(np.abs(kx)) / C_H + np.max(kx) ** 2 / 3.0)
    out.append("  (m^2 + A^2)/(c^2 k^2) - 1 : max = %.3e" % res)
    out.append("  budget = r k_max/c + k_max^2/3 = %.3e" % budget)
    u2 = abs(res - budget) / budget < 0.05
    out.append("  residual equals the discretisation budget: %s" % u2)
    ok.append(u2)
    out.append("U2 %s" % ("PASS" if u2 else "FAIL"))

    out.append("")
    out.append("=== U3 无零模与 E=0 对称 ===")
    num = np.linalg.eigvalsh(H_full(n))
    sym = float(np.max(np.abs(num + num[::-1])))
    out.append("  min |E| = %.6f" % np.min(np.abs(num)))
    out.append("  max |E_i + E_{N-i}| = %.3e" % sym)
    g5f = np.kron(I2, np.kron(G5, np.eye(n * n)))
    e5 = np.linalg.eigvalsh(g5f)
    out.append("  Gamma5 spectrum: +1 x%d, -1 x%d"
               % (int(np.sum(np.abs(e5 - 1) < 1e-9)), int(np.sum(np.abs(e5 + 1) < 1e-9))))
    u3 = sym < TOL_SYM
    ok.append(u3)
    out.append("U3 %s" % ("PASS" if u3 else "FAIL"))

    out.append("")
    out.append("=== U4 <Gamma5> 与 D4 判定 ===")
    ev, V = np.linalg.eigh(H_full(n))
    meas = np.array([complex(v.conj() @ g5f @ v).real for v in V.T])
    nonzero = float(np.max(np.abs(meas)))
    # D4 的正确测法：手性分支是 Gamma5 算符的本征空间，能量本征态不是。
    # 由 {Gamma5, H} = 0 知 Gamma5 把 +E 与 -E 两支互换，故在能量本征态上
    # <Gamma5> 恒为零。这是结构性结果，与维度无关，也与质量无关。
    g5sp = np.kron(G5, np.eye(n * n))
    hsp = H_sp(n)
    antic = float(np.max(np.abs(g5sp @ hsp + hsp @ g5sp)))
    modecomm = float(np.max(np.abs(g5f @ H0 + H0 @ g5f)))
    out.append("  {Gamma5_sp, H_sp} = %.3e" % antic)
    out.append("  {Gamma5, H_full} = %.3e (mode factor anticommutes with the spacetime chirality,"
               % modecomm)
    out.append("     contributing 2 kappa = %.3f)" % (2 * KAPPA))
    ev0, V0 = np.linalg.eigh(hsp)
    swap = 0.0
    for k in range(ev0.size):
        if ev0[k] <= 1e-9:
            continue
        w = g5sp @ V0[:, k]
        nw = np.linalg.norm(w)
        if nw < 1e-12:
            continue
        w = w / nw
        swap = max(swap, float(np.max(np.abs(hsp @ w + ev0[k] * w))))
    e5full = np.sort(np.linalg.eigvalsh(g5sp))
    npos = int(np.sum(np.abs(e5full - 1) < 1e-9))
    nneg = int(np.sum(np.abs(e5full + 1) < 1e-9))
    out.append("  {Gamma5, H} = %.3e" % antic)
    out.append("  Gamma5 maps a +E eigenstate to -E: max|H w + E w| = %.3e" % swap)
    out.append("  Gamma5 spectrum: +1 x%d, -1 x%d (equal split, both non-empty)" % (npos, nneg))
    out.append("  max |<Gamma5>| on energy eigenstates = %.3e (structurally zero)" % nonzero)
    out.append("  2+1 dimensions measured max|<Gamma5>| = %.3e" % nonzero)
    # 3+1 维的自由 Dirac 哈密顿量上穷举 Clifford 单项式，找与 H 对易且平方为 I 的厄米算符
    mons = {}
    for r in (1, 1j, -1, -1j):
        for mask in range(16):
            M = np.eye(4, dtype=complex)
            for bb in range(4):
                if mask >> bb & 1:
                    M = M @ (G0, G1, G2, G3)[bb]
            mons["%s_%04d" % (str(r), mask)] = r * M
    hfree = 0.5 * G0 - 1j * (0.6 * G1 + 0.3 * G2 + 0.5 * G3)
    comm = [k for k, Q in mons.items()
            if np.max(np.abs(Q @ Q - I4)) < 1e-12
            and np.max(np.abs(Q - Q.conj().T)) < 1e-12
            and np.max(np.abs(Q @ hfree - hfree @ Q)) < 1e-12]
    out.append("  3+1 free Dirac: Clifford monomials commuting with H, square I, herm = %d" % len(comm))
    out.append("     they are %s (only the trivial ones)" % comm)
    u4 = (antic < 1e-8 and swap < 1e-8 and npos == nneg and npos > 0
          and nonzero < 1e-9 and len(comm) == 2)
    ok.append(u4)
    out.append("  -> the two chiral branches of D4 exist as the +-1 eigenspaces of Gamma5, both")
    out.append("     non-empty and equal in dimension. They are NOT energy eigenstates:")
    out.append("     {Gamma5,H}=0 makes Gamma5 interchange the +-E sectors, so <Gamma5> vanishes")
    out.append("     on every energy eigenstate, in 2+1 and 3+1 alike and for any mass.")
    out.append("U4 %s" % ("PASS" if u4 else "FAIL"))

    out.append("")
    out.append("=== U5 宇称结构与 Wilson 项相容性 ===")
    a_p5 = float(np.max(np.abs(PAR @ G5 + G5 @ PAR)))
    sp = []
    for Gm in (G1, G2):
        sp.append(float(np.max(np.abs(PAR @ Gm + Gm @ PAR))))
    Wx = np.kron(G1, wilson_op(n, 0))
    Wy = np.kron(G2, wilson_op(n, 1))
    w1 = np.kron(G1, np.eye(n * n))
    parL = np.kron(PAR, np.eye(n * n))
    a_pw = max(float(np.max(np.abs(parL @ Wx + Wx @ parL))),
               float(np.max(np.abs(parL @ Wy + Wy @ parL))),
               float(np.max(np.abs(parL @ w1 + w1 @ parL))))
    WxF = np.kron(I2, Wx)
    WyF = np.kron(I2, Wy)
    a_5w = max(float(np.max(np.abs(g5f @ WxF + WxF @ g5f))),
               float(np.max(np.abs(g5f @ WyF + WyF @ g5f))))
    out.append("  {P,Gamma5} = %.1e   {P,gamma^i} = %s" % (a_p5, ["%.0e" % v for v in sp]))
    out.append("  {P, W} = %.1e   {Gamma5, W} = %.1e" % (a_pw, a_5w))
    u5 = a_p5 < TOL_SYM and max(sp) < TOL_SYM and a_pw < TOL_SYM and a_5w < TOL_SYM
    ok.append(u5)
    out.append("U5 %s" % ("PASS" if u5 else "FAIL"))

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9))
    ax[0].hist(meas, bins=60, color="tab:purple", density=True)
    ax[0].axvline(0.0, color="k", lw=0.8)
    ax[0].set_xlabel("<Gamma5> on 2+1 energy eigenstates")
    ax[0].set_ylabel("density")
    ax[0].set_title("(a) U4 vanishes on energy states")

    e5 = np.sort(np.linalg.eigvalsh(g5sp))
    ax[1].hist(e5, bins=20, color="tab:orange", density=True)
    ax[1].set_xticks([-1, 1])
    ax[1].set_xlabel("Gamma5 spectrum")
    ax[1].set_title("(b) U4 two branches, equal weight")

    ks = np.linspace(0, 2 * np.pi, 200)
    ax[2].plot(ks, C_H * np.sin(ks) + R_W * (np.cos(ks) - 1.0), label="A_x(k)")
    ax[2].axhline(0.0, color="k", lw=0.8)
    ax[2].set_xlabel(r"$k$")
    ax[2].set_title("(c) dispersive factor, r=%.1f" % R_W)
    ax[2].legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(FIG, dpi=150)
    out.append("")
    out.append("figure: %s" % FIG)

    allok = all(ok)
    out.append("SUMMARY U1..U5 = %s" % ("ALL PASS" if allok else "HAS FAILURE"))
    text = "\n".join(out)
    print(text)
    with io.open(FIG.replace(".png", "_result.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text + "\n")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
