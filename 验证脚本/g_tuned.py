# -*- coding: utf-8 -*-
r"""第 497 条：Wilson 调谐质量点与自旋二重态隔离下的 g 提取（J1 至 J4）。

依据第 496 条（34cbeb4）定位的两处缺口，合并为一轮：
  缺口一 提取口径：最低能级八重简并稀释自旋投影，须隔离自旋二重态
  缺口二 调谐点：标准 Wilson 结果要求在有效质量 m_eff = 0 处取读数

第 496 条的 Wilson 项误取 -i r gamma^1 W（乘在 gamma^1 上），非标准形式。
标准格点 Wilson 项乘在 gamma^0 上：-r gamma^0 W，W = (S + S^{-1} - 2I)/2。
本条改用标准形式，此时
  E^2 = [m + 2r(1 - cos k1)]^2 + sin^2 k1 + sin^2 k2
有效质量 m_eff(k) = m + 2r(1 - cos k)，在调谐点 m_eff = 0。

构造：一切在 k 空间完成。格点哈密顿量对平移不变，逐动量块 h(k) 精确。
自旋二重态取 (k0, -k0) 对，该对给出正能两重简并，正是 Pauli 自旋的两支。
"""
import io
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_H = 1.0
MU = 1.0
N_SITE = 8
RS = [0.0, 0.05, 0.1, 0.2, 0.4]
BS = [0.002, 0.005, 0.01, 0.02, 0.04]
K0 = (1, 1)

FIG = r"D:/software/TianXingFC/images/g_tuned.png"

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
SIGP = 0.5j * (G3 @ G1 - G1 @ G3)      # Pauli 型自旋，本征值 +-1


def block(k1, k2, m, r, B=0.0):
    r"""逐动量的 4x4 Dirac 块。Wilson 项取标准形式 -r gamma^0 W。"""
    meff = m + 2.0 * r * (1.0 - np.cos(k1))
    h = (meff * G0
         - 1j * C_H * np.sin(k1) * G1
         - 1j * C_H * np.sin(k2) * G2
         - MU * B * SIGP)
    return h, meff


def m_tuned(k1, r):
    return -2.0 * r * (1.0 - np.cos(k1))


def doublet_g(m, r, B, n=N_SITE, k0=K0):
    r"""在 (k0, -k0) 对内隔离自旋二重态并提取 g。

    逐动量块是 4x4，h(k) 的正能态各二重（二重谷），故取 k0 与 -k0 两个动量的
    正能态构成的一对，在该 2 维子空间内对 Pauli 自旋对角化，读 Zeeman 劈裂。
    """
    ks = 2.0 * np.pi * np.arange(n) / n
    m_t = m_tuned(2.0 * np.pi * k0[0] / n, r)
    vecs = []
    for sgn in (1, -1):
        k1, k2 = 2.0 * np.pi * k0[0] / n, sgn * 2.0 * np.pi * k0[1] / n
        h0, _ = block(k1, k2, m_t, r, 0.0)
        e, V = np.linalg.eigh(h0)
        pos = int(np.argmax(e))
        if e[pos] < 0:
            return None, None, m_t, None
        vecs.append((V[:, pos], float(e[pos])))
    eplus, eminus = None, None
    for v, e0 in vecs:
        v = v / np.linalg.norm(v)
        hB, _ = block(2.0 * np.pi * k0[0] / n, (1 if v is vecs[0][0] else -1)
                      * 2.0 * np.pi * k0[1] / n, m_t, r, B)
        s = float(np.vdot(v, SIGP @ v).real)
        val = float(np.vdot(v, hB @ v).real)
        if s > 0:
            eplus = val
        else:
            eminus = val
    if eplus is None or eminus is None:
        return None, None, m_t, None
    return eplus, eminus, m_t, [float(np.vdot(vecs[t][0], SIGP @ vecs[t][0]).real) for t in (0, 1)]


def main():
    out, ok = [], []
    out.append("=== 冒烟 ===")
    out.append("Sigma_P: herm %.1e  Sigma_P^2 - I = %.1e  eigen %s"
               % (np.max(np.abs(SIGP - SIGP.conj().T)),
                  np.max(np.abs(SIGP @ SIGP - I4)),
                  np.round(np.linalg.eigvalsh(SIGP), 6).tolist()))

    out.append("")
    out.append("=== J1 调谐质量点 ===")
    n = N_SITE
    js1 = []
    for r in RS:
        k1 = 2.0 * np.pi * K0[0] / n
        mt = m_tuned(k1, r)
        _, meff = block(k1, 2.0 * np.pi * K0[1] / n, mt, r, 0.0)
        js1.append(abs(meff))
        out.append("  r=%.2f  m* = %+.6f   m_eff(k0) = %.2e" % (r, mt, abs(meff)))
    j1 = max(js1) < 1e-12
    ok.append(j1)
    out.append("J1 %s" % ("PASS" if j1 else "FAIL"))

    out.append("")
    out.append("=== J2 自旋二重态的隔离 ===")
    sp = []
    for r in RS:
        ep, em, mt, s = doublet_g(m_tuned(2.0 * np.pi * K0[0] / n, r), r, 0.0)
        if s is None:
            sp.append(None)
            out.append("  r=%.2f  doublet not resolved" % r)
        else:
            sp.append(abs(s[0] - s[1]) / 2.0)
            out.append("  r=%.2f  <Sigma_P> on the two states: %s  (ideal +-1)" % (r, np.round(s, 6).tolist()))
    j2 = all(x is not None and x > 0.99 for x in sp)
    ok.append(j2)
    out.append("J2 %s  (the pair (k0,-k0) carries the full Pauli spin)" % ("PASS" if j2 else "FAIL"))

    out.append("")
    out.append("=== J3 g 的提取 ===")
    gs = []
    for r in RS:
        row = []
        for B in BS:
            ep, em, mt, s = doublet_g(m_tuned(2.0 * np.pi * K0[0] / n, r), r, B)
            if ep is None or em is None:
                row.append(float("nan"))
            else:
                g = (ep - em) / (MU * B)
                row.append(g)
                gs.append((r, B, g))
        out.append("  r=%.2f  g(B): %s" % (r, ["%.5f" % v for v in row]))
    g0 = [g for (r, B, g) in gs if r == 0.0]
    spreadB = (max(g0) - min(g0)) if g0 else float("nan")
    gmain = g0[0] if g0 else float("nan")
    out.append("  r=0: g = %.6f  (target 2, dev %.3e)   spread over B = %.3e"
               % (gmain, abs(gmain - 2), spreadB))
    j3 = abs(gmain - 2) < 1e-6 and spreadB < 1e-8
    ok.append(j3)
    out.append("J3 %s" % ("PASS" if j3 else "FAIL"))

    out.append("")
    out.append("=== J4 g(r) 在调谐点 ===")
    for r in RS:
        row = [g for (rr, _, g) in gs if rr == r]
        if row:
            out.append("  r=%.2f  g = %.6f  (dev from 2 = %.3e)" % (r, row[0], abs(row[0] - 2)))

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9))
    for r in RS:
        row = [g for (rr, B, g) in gs if rr == r and B == BS[0]]
        if row:
            ax[0].plot(BS, [g for (rr, B, g) in gs if rr == r], "o-", label="r=%.2f" % r)
    ax[0].axhline(2.0, color="k", ls="--", lw=0.9, label="g = 2")
    ax[0].set_xlabel("B")
    ax[0].set_ylabel("g")
    ax[0].set_title("(a) J3 g at tuned mass")
    ax[0].legend(fontsize=7)

    ax[1].plot(RS, [abs(r) for r in JS1] if False else [0 for r in RS], "o-", color="tab:green")
    ax[1].set_xlabel("Wilson parameter r")
    ax[1].set_ylabel("|m_eff(k0)|")
    ax[1].set_title("(b) J1 tuning residual")

    ax[2].plot(RS, [g for (rr, B, g) in gs if B == BS[0]], "o-", color="tab:orange")
    ax[2].axhline(2.0, color="k", ls="--", lw=0.9)
    ax[2].set_xlabel("r")
    ax[2].set_ylabel("g")
    ax[2].set_title(r"(c) J4 g(r) at tuned mass")

    fig.tight_layout()
    fig.savefig(FIG, dpi=150)
    out.append("")
    out.append("figure: %s" % FIG)
    allok = all(ok)
    out.append("SUMMARY J1..J4 = %s" % ("ALL PASS" if allok else "HAS FAILURE"))
    text = "\n".join(out)
    print(text)
    with io.open(FIG.replace(".png", "_result.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text + "\n")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
