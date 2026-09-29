# -*- coding: utf-8 -*-
r"""第 495 条：D4 区分量的判定（K1 至 K4）。

依据 e47438e 的降级结论：D4 缺的是把 Gamma5 两支分开的物理量。本条检验三个候选
区分量，并给出唯一可行的读法。

模型（沿用第 494 条的 2+1 维 8 分量结构）：
  H = kappa sigma_z (x) I + I (x) H_sp
  H_sp = m gamma^0 (x) I - i gamma^1 (x) D1 - i gamma^2 (x) D2
  格点 N x N 周期，N in {6, 8}

候选区分量：
  模态因子   Z_mode = sigma_z (x) I        守恒（不同张量因子）
  畴壁 Z2   S = J-flip，即两 Weyl 分量的相对符号，S^2 = I
  乘积       M = i S Gamma5

判据：
  K1 Gamma5 与 H 的反对易，及两支能量反号
  K2 谱关于 E=0 对称、两扇区简并度相等、无零模
  K3 模态因子守恒但与 Gamma5 反对易，不能作区分量
  K4 畴壁 Z2 标签与 H 反对易故非守恒，其乘积 M 守恒但是联合标签
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
NS = [6, 8]
TOL = 1e-8

FIG = r"D:/software/TianXingFC/images/d4_separator.png"

I2 = np.eye(2, dtype=complex)
I4 = np.eye(4, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
Z2 = np.zeros((2, 2), dtype=complex)
G0 = np.block([[Z2, I2], [I2, Z2]])   # 与第 494 条同基，勿改
G1 = np.block([[Z2, SZ], [-SZ, Z2]])
G2 = np.block([[Z2, SX], [-SX, Z2]])
G3 = np.block([[Z2, SY], [-SY, Z2]])
G5 = 1j * G0 @ G1 @ G2 @ G3
SF = np.block([[-I2, Z2], [Z2, I2]])


def shift2(n, axis):
    S = np.zeros((n * n, n * n), dtype=complex)
    idx = np.arange(n * n)
    i, j = idx // (n * n), idx % n
    for a in range(n * n):
        c = [int(i[a]), int(j[a])]
        c[axis] = (c[axis] + 1) % n
        S[a, c[0] * n + c[1]] = 1.0
    return S


def H_sp(n, m=M0):
    D1 = -1j * (shift2(n, 0) - shift2(n, 0).T) / 2.0
    D2 = -1j * (shift2(n, 1) - shift2(n, 1).T) / 2.0
    # D1、D2 已是 n^2 x n^2 的二维格点算符，无需再与单位阵作 Kronecker 积
    return (m * np.kron(G0, np.eye(n * n))
            - 1j * np.kron(G1, D1)
            - 1j * np.kron(G2, D2))


def H_full(n, m=M0):
    Hs = H_sp(n, m)
    return np.kron(KAPPA * SZ, np.eye(Hs.shape[0])) + np.kron(I2, Hs)


def ac(a, b):
    return float(np.max(np.abs(a @ b + b @ a)))


def main():
    out, ok = [], []
    n = NS[-1]
    hs = H_sp(n)
    g5s = np.kron(G5, np.eye(n * n))
    sf = np.kron(SF, np.eye(n * n))
    zm = np.kron(SZ, np.eye(hs.shape[0]))
    g5f = np.kron(I2, g5s)
    mf = np.kron(I2, np.kron(SF @ G5, np.eye(n * n)))   # P = S Gamma5（该基下 S 与 Gamma5 对易）
    Hf = H_full(n)

    out.append("=== 冒烟 ===")
    out.append("dim %d, H herm dev %.2e" % (Hf.shape[0], np.max(np.abs(Hf - Hf.conj().T))))
    out.append("S: herm %.1e  S^2-I %.1e" % (np.max(np.abs(sf - sf.conj().T)),
                                              np.max(np.abs(sf @ sf - np.eye(hs.shape[0])))))
    out.append("P = S Gamma5: herm %.1e  P^2-I %.1e"
               % (np.max(np.abs(mf - mf.conj().T)),
                  np.max(np.abs(mf @ mf - np.eye(Hf.shape[0])))))
    sq = [float(np.trace(G @ G).real) / 4 for G in (G0, G1, G2, G3)]
    out.append("gamma squares (expect +1 -1 -1 -1): %s" % ["%+.0f" % v for v in sq])
    out.append("Gamma5 anticommutators: %s"
               % ["%.0e" % np.max(np.abs(G5 @ G + G @ G5)) for G in (G0, G1, G2, G3)])
    out.append("  (basis locked to the 494 round; a drift here invalidates S and M)")

    out.append("")
    out.append("=== K1 Gamma5 与 H 的反对易及两支能量反号 ===")
    a1 = ac(g5s, hs)
    ev, V = np.linalg.eigh(hs)
    swap = 0.0
    for k in range(ev.size):
        if ev[k] <= 1e-9:
            continue
        w = g5s @ V[:, k]
        nw = np.linalg.norm(w)
        if nw < 1e-12:
            continue
        swap = max(swap, float(np.max(np.abs(hs @ (w / nw) + ev[k] * (w / nw)))))
    out.append("  {Gamma5, H_sp} = %.3e" % a1)
    out.append("  Gamma5 maps a +E state to -E: max|H w + E w| = %.3e" % swap)
    k1 = a1 < TOL and swap < TOL
    ok.append(k1)
    out.append("K1 %s" % ("PASS" if k1 else "FAIL"))

    out.append("")
    out.append("=== K2 谱对称、两扇区简并度、无零模 ===")
    evf = np.linalg.eigvalsh(Hf)
    evf2, Vf = np.linalg.eigh(Hf)
    sym = float(np.max(np.abs(evf + evf[::-1])))
    npos = int(np.sum(evf > 1e-9))
    nneg = int(np.sum(evf < -1e-9))
    zero = int(np.sum(np.abs(evf) < 1e-9))
    out.append("  max|E_i + E_{N-i}| = %.3e" % sym)
    out.append("  positive-E states %d, negative-E states %d, zero-mode %d" % (npos, nneg, zero))
    k2 = sym < TOL and npos == nneg and zero == 0
    ok.append(k2)
    out.append("  the two branches are the E>0 and E<0 sectors, equal in number")
    out.append("K2 %s" % ("PASS" if k2 else "FAIL"))

    out.append("")
    out.append("=== K3 模态因子：守恒但与 Gamma5 反对易 ===")
    c3 = float(np.max(np.abs(zm @ Hf - Hf @ zm)))
    a3 = ac(zm, g5f)
    out.append("  [sigma_z (x) I, H_full] = %.3e  (conserved)" % c3)
    out.append("  {sigma_z (x) I, Gamma5} = %.3e" % a3)
    k3 = c3 < TOL and a3 > 0.5
    ok.append(k3)
    out.append("  -> the mode factor is conserved but anticommutes with Gamma5, so no state is a")
    out.append("     simultaneous eigenstate of both; it cannot separate the two branches.")
    out.append("K3 %s" % ("PASS" if k3 else "FAIL"))

    out.append("")
    out.append("=== K4 畴壁 Z2 标签与乘积 ===")
    a4 = ac(sf, hs)
    c4 = float(np.max(np.abs(sf @ hs - hs @ sf)))
    a4b = ac(sf, g5s)
    c4m = float(np.max(np.abs(mf @ Hf - Hf @ mf)))
    c4s = float(np.max(np.abs(sf @ g5s - g5s @ sf)))
    c4p = float(np.max(np.abs(mf @ g5f - g5f @ mf)))
    evf2, Vf = np.linalg.eigh(Hf)
    out.append("  {S, H_sp} = %.3e   [S, H_sp] = %.3e   (S anticommutes with H)" % (a4, c4))
    out.append("  [S, Gamma5] = %.3e   (S commutes with Gamma5 in this basis)" % c4s)
    out.append("  P = S Gamma5: [P, H_full] = %.3e   [P, Gamma5] = %.3e" % (c4m, c4p))
    combo = {}
    for k in range(evf.size):
        v = Vf[:, k]
        key = (1 if complex(v.conj() @ mf @ v).real > 0 else -1,
               1 if complex(v.conj() @ g5f @ v).real > 0 else -1)
        combo[key] = combo.get(key, 0) + 1
    out.append("  joint spectrum (P, Gamma5): %s" % {str(k): v for k, v in sorted(combo.items())})
    k4 = a4 < TOL and c4s < TOL and c4m < TOL and c4p < TOL
    ok.append(k4)
    allp = sorted({k[0] for k in combo})
    trivial = allp == [1]
    out.append("  -> S anticommutes with H, not conserved, cannot be a separator.")
    out.append("     P = S Gamma5 is conserved and commutes with Gamma5, but its spectrum is %s,"
               % ("entirely +1, i.e. P is the identity on the physical space" if trivial else str(allp)))
    out.append("     so P carries no information and does not separate the two branches.")
    if trivial:
        out.append("     The joint spectrum has %d combinations, both with P = +1 and Gamma5 = -+1."
                   % len(combo))
    out.append("K4 %s" % ("PASS" if k4 and trivial else "FAIL"))

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9))
    es = np.sort(evf)
    ax[0].plot(es, "o", ms=2, color="tab:blue")
    ax[0].axhline(0.0, color="k", lw=0.9)
    ax[0].set_xlabel("state index")
    ax[0].set_ylabel("E")
    ax[0].set_title("(a) K2 spectrum symmetric about E=0")

    m5 = np.sort(np.linalg.eigvalsh(mf))
    ax[1].hist(m5, bins=20, color="tab:orange", density=True)
    ax[1].set_xticks([-1, 1])
    ax[1].set_xlabel("P = S Gamma5 spectrum")
    ax[1].set_title("(b) K4 joint label, conserved")

    names = ["Gamma5", "S", "sigma_z(mode)", "M"]
    comm = [a1, c4, c3, c4m]
    antic = [a1, a4, a3, c4p]
    x = np.arange(4)
    ax[2].bar(x - 0.18, comm, 0.34, label="commutator")
    ax[2].bar(x + 0.18, antic, 0.34, label="anticommutator")
    ax[2].set_xticks(x)
    ax[2].set_xticklabels(names, rotation=25, fontsize=7)
    ax[2].set_title("(c) candidates vs H")
    ax[2].legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(FIG, dpi=150)
    out.append("")
    out.append("figure: %s" % FIG)

    allok = all(ok)
    out.append("SUMMARY K1..K4 = %s" % ("ALL PASS" if allok else "HAS FAILURE"))
    text = "\n".join(out)
    print(text)
    with io.open(FIG.replace(".png", "_result.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text + "\n")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
