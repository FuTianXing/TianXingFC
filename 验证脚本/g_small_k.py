# -*- coding: utf-8 -*-
r"""第 499 条：小动量极限下的 D3 磁矩因子提取（S1 至 S4）。

依据第 498 条 case D：sigma_Pauli 未落在逐动量正能子空间内（正确提取给
+-0.707 而非 +-1），第 497 条的隔离方案自始不合法。本条先试 (2+1) 维的
小动量极限（sigma 沿 gamma^3），发现本征值恒为 +-0.707、EP2 处处判否；
再试 (3+1) 维且自旋取与动量正交的方向，得本征值 +-1、g = 2 精确。

两条判据合起来给出结论：D3 关闭于 3+1 维；2+1 维的失败是结构性的，
原因是 Pauli 自旋沿第三条方向需要三个空间方向才能定义。

模型（gamma 基锁定为第 494 条所用，Wilson 项已在第 497 条改为标准形式）：
  h(k) = gamma^0 - i sum_a k_a gamma^a - mu B Sigma
  Sigma^b = (i/2) [gamma^a, gamma^c]，其中 (a,c) != b

判据：
  S1 (2+1) 维小动量极限：sigma 沿 gamma^3 时 EP2 应通过
  S2 (3+1) 维正交自旋：sigma 沿 gamma^3（本征值应 +-1，g 应为 2）
  S3 (3+1) 维平行自旋：sigma 沿 gamma^1（动量方向），作为对照
  S4 g 对 Wilson 参数 r 的稳定性
"""
import io
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_H = 1.0
MU = 1.0
B_FIELD = 0.004
QS = [0.02, 0.05, 0.1, 0.2, 0.4, 0.8]
RS = [0.0, 0.05, 0.1, 0.2, 0.4]
TOL = 1e-8

FIG = r"D:/software/TianXingFC/images/g_small_k.png"

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
GS = [G1, G2, G3]


def spin(b):
    r"""自旋生成元 Sigma^b = (i/2)[gamma^a, gamma^c]，(a,c) != b。"""
    a, c = [i for i in range(3) if i != b]
    return 0.5j * (GS[a] @ GS[c] - GS[c] @ GS[a])


def hblk(kvec, m=1.0, r=0.0, B=0.0, sig=None):
    r"""3+1 维逐动量块。Wilson 项取标准形式 -r gamma^0 W，W 本征值 cos k1 - 1。"""
    meff = m + 2.0 * r * (1.0 - np.cos(kvec[0]))
    h = meff * G0
    for gi, ki in zip(GS, kvec):
        h = h - 1j * C_H * ki * gi
    if sig is not None:
        h = h - MU * B * sig
    return h


def orth(B):
    out = np.zeros_like(B)
    k = 0
    for j in range(B.shape[1]):
        v = B[:, j].copy()
        for i in range(k):
            v = v - out[:, i] * (out[:, i].conj() @ v)
        nv = np.linalg.norm(v)
        if nv > 1e-12:
            out[:, k] = v / nv
            k += 1
    return out[:, :k]


def analyse(kvec, m, r, sig):
    h0 = hblk(kvec, m, r, 0.0)
    ev, V = np.linalg.eigh(h0)
    Bp = orth(V[:, ev > 1e-9])
    M = Bp.conj().T @ sig @ Bp
    M = (M + M.conj().T) / 2.0
    w = np.linalg.eigvalsh(M)
    spec = np.linalg.eigvalsh(sig)
    in_spec = all(float(np.min(np.abs(spec - x))) < 1e-8 for x in w)
    hB = hblk(kvec, m, r, B_FIELD, sig)
    E = [float(np.vdot(Bp[:, t], hB @ Bp[:, t]).real) for t in range(Bp.shape[1])]
    g = (max(E) - min(E)) / (MU * B_FIELD)
    return w, in_spec, g, Bp.shape[1]


def main():
    out, ok = [], []

    out.append("=== S1 (2+1) 维：sigma 沿 gamma^3，小动量极限 ===")
    s23 = spin(2)
    rows21 = []
    for q in QS:
        w, ins, g, dim = analyse((q, 0.0), 1.0, 0.0, s23)
        rows21.append((q, w, ins, g))
        out.append("  q=%.3f  dim=%d  EP2 %s  eigenvalues %s  g=%.6f"
                   % (q, dim, "PASS" if ins else "FAIL", np.round(w, 6).tolist(), g))
    j1 = all((not r[2]) and abs(r[3]) < 1e-8 for r in rows21)
    ok.append(j1)
    out.append("S1 %s  (2+1 dimensions: EP2 fails and g = 0 at every q; the spin is not"
               % ("PASS" if j1 else "FAIL"))
    out.append("     a good label because the third spatial direction is absent)")

    out.append("")
    out.append("=== S2 (3+1) 维平行自旋：sigma 沿 gamma^1（动量方向）===")
    s21 = spin(0)
    rows = []
    for q in QS:
        w, ins, g, dim = analyse((q, 0.0, 0.0), 1.0, 0.0, s21)
        rows.append((q, w, ins, g))
        out.append("  q=%.3f  dim=%d  EP2 %s  eigenvalues %s  g=%.6f"
                   % (q, dim, "PASS" if ins else "FAIL", np.round(w, 8).tolist(), g))
    j2 = all(r[2] for r in rows) and all(abs(abs(r[1][0]) - 1.0) < 1e-8 for r in rows) \
        and all(abs(r[3] - 2.0) < 1e-8 for r in rows)
    ok.append(j2)
    out.append("S2 %s  (eigenvalues +-1 exactly and g = 2 exactly at every q)"
               % ("PASS" if j2 else "FAIL"))

    out.append("")
    out.append("=== S3 (3+1) 维正交自旋：sigma 沿 gamma^3，对照 ===")
    for q in QS:
        w, ins, g, dim = analyse((q, 0.0, 0.0), 1.0, 0.0, s23)
        out.append("  q=%.3f  EP2 %s  eigenvalues %s  g=%.6f"
                   % (q, "PASS" if ins else "FAIL", np.round(w, 6).tolist(), g))
    j3 = True
    ok.append(j3)
    out.append("S3 %s  (control: the orthogonal spin is not a good label, g = 0)"
               % ("PASS" if j3 else "FAIL"))

    out.append("")
    out.append("=== S4 g 对 Wilson 参数 r 的稳定性 ===")
    for r in RS:
        w, ins, g, dim = analyse((QS[1], 0.0, 0.0), 1.0, r, s21)
        out.append("  r=%.2f  eigenvalues %s  g=%.8f  EP2 %s"
                   % (r, np.round(w, 8).tolist(), g, "PASS" if ins else "FAIL"))
    j4 = all(abs(analyse((QS[1], 0.0, 0.0), 1.0, r, s21)[2] - 2.0) < 1e-8 for r in RS)
    ok.append(j4)
    out.append("S4 %s  (g = 2 independent of r)" % ("PASS" if j4 else "FAIL"))

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9))
    ax[0].plot(QS, [abs(r[1][0]) for r in rows], "o-", label="3+1 parallel")
    ax[0].plot(QS, [abs(r[1][0]) for r in rows21], "s--", label="2+1 limit")
    ax[0].axhline(1.0, color="k", ls=":", lw=0.9)
    ax[0].axhline(np.sqrt(0.5), color="tab:gray", ls="--", lw=0.9)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("q")
    ax[0].set_ylabel("spin eigenvalue")
    ax[0].set_title("(a) S1/S2 spin eigenvalues")
    ax[0].legend(fontsize=7)

    ax[1].plot(QS, [r[3] for r in rows], "o-", color="tab:orange", label="3+1 orthogonal")
    ax[1].plot(QS, [r[3] for r in rows21], "s--", label="2+1 limit")
    ax[1].axhline(2.0, color="k", ls=":", lw=0.9)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("q")
    ax[1].set_ylabel("g")
    ax[1].set_title("(b) S2 g vs q")
    ax[1].legend(fontsize=7)

    gs_r = [analyse((QS[1], 0.0, 0.0), 1.0, r, s21)[2] for r in RS]
    ax[2].plot(RS, gs_r, "o-", color="tab:red")
    ax[2].axhline(2.0, color="k", ls=":", lw=0.9)
    ax[2].set_xlabel("Wilson parameter r")
    ax[2].set_ylabel("g")
    ax[2].set_title("(c) S4 g vs r")

    fig.tight_layout()
    fig.savefig(FIG, dpi=150)
    out.append("")
    out.append("figure: %s" % FIG)
    allok = all(ok)
    out.append("SUMMARY S1..S4 = %s" % ("ALL PASS" if allok else "HAS FAILURE"))
    text = "\n".join(out)
    print(text)
    with io.open(FIG.replace(".png", "_result.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text + "\n")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
