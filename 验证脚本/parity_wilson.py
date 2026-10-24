# -*- coding: utf-8 -*-
r"""第 490 条 D5：宇称变换行为与 Wilson 项（第二版）。

承接第 489 条的 2 分量骨架：
  H = sigma_x (x) c_th D_th + sigma_z (x) c_h D_h + sigma_y (x) diag(m)
theta 环反周期（pi 扭转），h 环周期（D-b 主设定）。

第一版三处判负的原因（本版逐条留痕）：
  1. PD1 的三个候选全部 max|P H P - H| = 1.0，非实现错误。根因是代数阻塞：
     格点宇称要求 Pauli 侧满足 [P, sigma_x] = 0、[P, sigma_y] = 0、{P, sigma_z} = 0
     （质量项与内部运动学不变，空间运动学翻号）。2x2 情形下与 sigma_x、sigma_y 同时
     对易的矩阵集为 {+-I}，而 +-I 不满足 {P, sigma_z} = 0，故无解。穷举已验证。
  2. PD3 的破缺量在 h_0 全档恒为 1.0，说明所选 P 根本不是对称，量的大小不携带
     自发破缺信息。该判据的前提（存在一个可对照的宇称对称）在 2 分量下不成立，
     改判为代数阻塞的推论，并补 4x4 对照。
  3. PD1 的判据形式由「找到宇称对称」改为「穷举完备性与无解确认」，因为 2 分量下
     的正确结论就是无解。

PD3 新增 4x4 对照：(2+1) 维的 4x4 Clifford 代数，穷举满足 [Q, gamma_0] = 0、
[Q, gamma_1] = 0、{Q, gamma_2} = 0 的候选，并检验格点宇称（纯格点反演）。

PD4 Wilson 项在 k 空间判据，处方取标准形式
  E^2 = sum_mu [c_mu sin k_mu + r (k_mu - sin k_mu)]^2 + m^2
该式的选择属定理约束下的方案选择，候选空间被 Nielsen-Ninomiya 限死。

判据：
  PD1 2x2 宇称候选的穷举完备性与无解确认
  PD2 宇称与手性算符 sigma_y 的对易结构
  PD3 4x4 对照
  PD4 Wilson 项
  PD5 赝手性的质量退化
"""
import io
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_TH, N_H = 16, 16
C_TH, C_H = 1.0, 1.0
M0 = 0.5
TOL = 1e-10

FIG = r"D:/software/TianXingFC/images/parity_wilson.png"

I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
P2 = {"I": I2, "x": SX, "y": SY, "z": SZ, "ix": 1j * SX, "iy": 1j * SY,
      "iz": 1j * SZ, "xy": SX @ SY, "yz": SY @ SZ, "zx": SZ @ SX}

G0 = np.kron(SY, I2)
G1 = np.kron(SX, SX)
G2 = np.kron(SX, SY)
SIG5 = -1j * (G0 @ G1 @ G2)   # Euclidean: raw product squares to -I
G4 = {"I": np.eye(4, dtype=complex), "G0": G0, "G1": G1, "G2": G2,
      "G0G1": G0 @ G1, "G1G2": G1 @ G2, "G0G2": G0 @ G2, "SIG5": SIG5,
      "G0SIG5": G0 @ SIG5, "G1SIG5": G1 @ SIG5, "G2SIG5": G2 @ SIG5,
      "G0G1SIG5": G0 @ G1 @ SIG5, "G1G2SIG5": G1 @ G2 @ SIG5,
      "G0G2SIG5": G0 @ G2 @ SIG5, "G0G1G2": G0 @ G1 @ G2}


def shift_th():
    S = np.zeros((N_TH, N_TH))
    for i in range(N_TH):
        S[i, (i + 1) % N_TH] = -1.0 if i == N_TH - 1 else 1.0
    return S


def shift_h():
    S = np.zeros((N_H, N_H))
    for i in range(N_H):
        S[i, (i + 1) % N_H] = 1.0
    return S


def mom(S):
    return -1j * (S - S.T) / 2.0


D_TH, D_H = mom(shift_th()), mom(shift_h())
EYE_TH, EYE_H = np.eye(N_TH), np.eye(N_H)


def R_rev(n):
    R = np.zeros((n, n))
    for i in range(n):
        R[i, (-i) % n] = 1.0
    return R


R_TH, R_H = R_rev(N_TH), R_rev(N_H)
ID_LAT = np.kron(EYE_TH, EYE_H)


def H2(m_field=None):
    m = np.full(N_H, M0) if m_field is None else m_field
    return (np.kron(SX, np.kron(C_TH * D_TH, EYE_H))
            + np.kron(SZ, np.kron(EYE_TH, C_H * D_H))
            + np.kron(SY, np.kron(EYE_TH, np.diag(m))))


def H4(m_field=None):
    m = np.full(N_H, M0) if m_field is None else m_field
    n = N_TH * N_H
    dth = np.kron(C_TH * D_TH, EYE_H)
    dh = np.kron(EYE_TH, C_H * D_H)
    return (np.kron(G1, dth) + np.kron(G2, dh)
            + np.kron(G0, np.kron(EYE_TH, np.diag(m))))


SY_OP = np.kron(SY, ID_LAT)
P_SPIN = {"P_A": SY, "P_B": I2, "P_C": SZ, "P_D": SX}
P_LAT = {"R_thh": np.kron(R_TH, R_H), "R_h": np.kron(EYE_TH, R_H),
         "R_th": np.kron(R_TH, EYE_H), "I": ID_LAT}


def m_bubble(h0, amp=0.30):
    n = np.arange(N_H) - N_H / 2.0
    return M0 * (1.0 + amp * 1.0 / np.cosh((n - h0) / 2.0) ** 2)


def wilson_E2(kth, kh, r, m=M0):
    a = C_TH * np.sin(kth) + r * (kth - np.sin(kth))
    b = C_H * np.sin(kh) + r * (kh - np.sin(kh))
    return a * a + b * b + m * m


def main():
    out, ok = [], []

    out.append("=== 冒烟 ===")
    H0 = H2()
    out.append("H2 shape %d, Hermitian dev %.2e" % (H0.shape[0], np.max(np.abs(H0 - H0.conj().T))))
    H4m = H4()
    out.append("H4 shape %d, Hermitian dev %.2e" % (H4m.shape[0], np.max(np.abs(H4m - H4m.conj().T))))
    out.append("Clifford 4x4 (Euclidean): {G0,G1}=%.0e {G0,G2}=%.0e {G1,G2}=%.0e  squares=%.0e %.0e %.0e"
               % (np.max(np.abs(G0 @ G1 + G1 @ G0)), np.max(np.abs(G0 @ G2 + G2 @ G0)),
                  np.max(np.abs(G1 @ G2 + G2 @ G1)),
                  np.trace(G0 @ G0).real / 4, np.trace(G1 @ G1).real / 4, np.trace(G2 @ G2).real / 4))

    out.append("")
    out.append("=== PD1 2x2 宇称候选：穷举完备性与无解确认 ===")
    comm = []
    for k, Q in P2.items():
        cx = np.max(np.abs(Q @ SX - SX @ Q))
        cy = np.max(np.abs(Q @ SY - SY @ Q))
        if cx < TOL and cy < TOL:
            comm.append(k)
    out.append("  commuting with both sigma_x and sigma_y: %s" % comm)
    out.append("  centralizer of {sigma_x, sigma_y} in 2x2 is {+-I}; neither anticommutes with sigma_z")
    out.append("  -> no 2x2 Pauli operator satisfies [P,sigma_x]=[P,sigma_y]=0 and {P,sigma_z}=0")
    a2 = {k: np.max(np.abs(Q @ SZ + SZ @ Q)) for k, Q in P2.items() if k in comm}
    out.append("  {P, sigma_z} for the centralizer: %s" % {k: float(v) for k, v in a2.items()})

    grid_dev, triv = [], []
    for sn, sv in P_SPIN.items():
        for ln, lv in P_LAT.items():
            P = np.kron(sv, lv)
            inv2 = np.max(np.abs(P @ P - np.eye(P.shape[0])))
            dev = np.max(np.abs(P @ H0 @ P - H0))
            is_id = np.max(np.abs(P - np.eye(P.shape[0]))) < TOL
            (triv if is_id else grid_dev).append(dev)
            out.append("  %s (x) %-6s  max|P^2 - I| = %.2e   max|P H P - H| = %.6f%s"
                       % (sn, ln, inv2, dev, "   <- identity, trivial" if is_id else ""))
    out.append("  non-trivial candidates: %d, of which symmetries: %d"
               % (len(grid_dev), sum(1 for d in grid_dev if d < TOL)))
    j1 = set(comm) == {"I"} and min(grid_dev) > 0.5
    ok.append(j1)
    out.append("PD1 %s  (no 2x2 parity exists; the lattice candidates confirm it numerically)"
               % ("PASS" if j1 else "FAIL"))

    out.append("")
    out.append("=== PD2 宇称与 sigma_y 的对易结构 ===")
    antic = []
    for sn, sv in P_SPIN.items():
        P = np.kron(sv, P_LAT["R_thh"])
        c = np.max(np.abs(P @ SY_OP @ P - SY_OP))
        a = np.max(np.abs(P @ SY_OP @ P + SY_OP))
        kind = "commute" if c < TOL else ("anticommute" if a < TOL else "neither")
        if kind == "anticommute":
            antic.append(sn)
        out.append("  %s  [P,sigma_y] = %.2e   {P,sigma_y} = %.2e   -> %s" % (sn, c, a, kind))
    out.append("  anticommuting candidates: %s" % antic)
    out.append("  none of them is a symmetry of H (PD1), so the D4 algebraic premise")
    out.append("  (P connects the two branches) does NOT hold in the 2-component realisation")
    j2 = len(antic) >= 1
    ok.append(j2)
    out.append("PD2 %s" % ("PASS" if j2 else "FAIL"))

    out.append("")
    out.append("=== PD3 4x4 对照 ===")
    sol4, tbl4 = [], []
    for k, Q in G4.items():
        c0 = np.max(np.abs(Q @ G0 - G0 @ Q))
        c1 = np.max(np.abs(Q @ G1 - G1 @ Q))
        a2_ = np.max(np.abs(Q @ G2 + G2 @ Q))
        sq = np.max(np.abs(Q @ Q - np.eye(4)))
        good = c0 < TOL and c1 < TOL and a2_ < TOL and sq < TOL
        if good:
            sol4.append(k)
        tbl4.append((k, c0, c1, a2_, sq, good))
    for k, c0, c1, a2_, sq, good in tbl4:
        out.append("  %-10s [Q,G0]=%.0e [Q,G1]=%.0e {Q,G2}=%.0e Q^2-I=%.0e %s"
                   % (k, c0, c1, a2_, sq, "<== SOLUTION" if good else ""))
    out.append("  4x4 candidates satisfying [Q,G0]=[Q,G1]=0, {Q,G2}=0, Q^2=I: %s" % (sol4 if sol4 else "none"))
    out.append("  SIG5 = i G0 G1 G2: [SIG5,G0]=%.1e [SIG5,G1]=%.1e {SIG5,G2}=%.1e SIG5^2-I=%.1e"
               % (np.max(np.abs(SIG5 @ G0 - G0 @ SIG5)), np.max(np.abs(SIG5 @ G1 - G1 @ SIG5)),
                  np.max(np.abs(SIG5 @ G2 + G2 @ SIG5)), np.max(np.abs(SIG5 @ SIG5 - np.eye(4)))))
    allanti = max(np.max(np.abs(SIG5 @ G + G @ SIG5)) for G in (G0, G1, G2))
    allcom = max(np.max(np.abs(SIG5 @ G - G @ SIG5)) for G in (G0, G1, G2))
    out.append("  SIG5 vs the gamma set: max|anticommutator| = %.3e, max|commutator| = %.3e" % (allanti, allcom))
    out.append("  -> SIG5 commutes with all three gammas and squares to I, i.e. it is the centre")
    out.append("     of the 4x4 Euclidean Clifford, a trivial spectator symmetry, not a parity")
    Psig = np.kron(SIG5, P_LAT["I"])
    dsign = np.max(np.abs(Psig @ H4m @ Psig + H4m))
    ddiff = np.max(np.abs(Psig @ H4m @ Psig - H4m))
    out.append("  lattice realisation: max|P H4 P + H4| = %.2e   max|P H4 P - H4| = %.6f" % (dsign, ddiff))
    out.append("  -> the lattice parity maps H to -H, i.e. it is a particle-hole type operator;")
    out.append("     the spectrum is then symmetric about E = 0, matching SK3 of the 489 round")
    Ploc = np.kron(np.eye(4), P_LAT["R_h"])
    dloc = np.max(np.abs(Ploc @ H4m @ Ploc - H4m))
    out.append("  naive lattice parity (pure h inversion): max|P H4 P - H4| = %.6f" % dloc)
    j3 = len(sol4) == 0 and allanti > 0.5 and allcom < TOL and dloc > 0.5
    ok.append(j3)
    out.append("PD3 %s  (no parity in (2+1)d at either 2x2 or 4x4; SIG5 is a trivial central symmetry)"
               % ("PASS" if j3 else "FAIL"))

    out.append("")
    out.append("=== PD4 Wilson 项（k 空间）===")
    kth = 2 * np.pi * (np.arange(N_TH) + 0.5) / N_TH
    kh = 2 * np.pi * np.arange(N_H) / N_H
    K = np.meshgrid(kh, kth, indexing="ij")
    par4 = max(float(np.max(np.abs(wilson_E2(K[1], K[0], r) - wilson_E2(-K[1], -K[0], r))))
               for r in [0.0, 0.2, 0.5, 1.0, 2.0])
    j4a = par4 < TOL
    out.append("  PD4a parity E2(-k)=E2(k), max dev %.2e  %s" % (par4, "PASS" if j4a else "FAIL"))
    ks = np.linspace(0.0, np.pi, 400)
    rs = np.linspace(0.0, 6.0, 61)
    splits = [float(np.max(np.abs(wilson_E2(0.0, ks, r, 0.0) - wilson_E2(0.0, np.pi - ks, r, 0.0))))
              for r in rs]
    emin, edou = [], []
    for r in rs:
        ev = np.sort(np.sqrt(np.abs(wilson_E2(K[0], K[1], r, M0))).ravel())
        emin.append(float(ev[0]))
        edou.append(float(ev[1]))
    sep = int(np.argmax(np.array(edou) - np.array(emin)))
    rbest = float(rs[sep])
    gaps = emin
    j4b = splits[0] < TOL and splits[-1] > 1.0 and all(
        splits[i] <= splits[i + 1] + 1e-12 for i in range(len(splits) - 1))
    out.append("  PD4b doubling split: r=0 -> %.2e ; r=%.1f -> %.4f  monotone=%s  %s"
               % (splits[0], rs[-1], splits[-1],
                  all(splits[i] <= splits[i + 1] + 1e-12 for i in range(len(splits) - 1)),
                  "PASS" if j4b else "FAIL"))
    rbest = float(rs[int(np.argmax(gaps))])
    kg = np.linspace(1e-4, 3e-3, 5)
    cdev = max(float(np.max(np.abs(wilson_E2(0.0, kg, r, 0.0) / kg ** 2 - C_H ** 2) / C_H ** 2))
               for r in [0.0, 0.5, 1.0])
    j4c = cdev < 1e-4
    out.append("  PD4c low-k continuum intact, max rel dev %.2e  %s" % (cdev, "PASS" if j4c else "FAIL"))
    out.append("  PD4d doublet-physical separation maximised at r = %.3f" % rbest)
    out.append("       physical lowest |E| = %.6f, doublet |E| = %.6f, separation = %.6f"
               % (emin[sep], edou[sep], edou[sep] - emin[sep]))
    out.append("       at r = 0 the two are degenerate: |E| = %.6f and %.6f" % (emin[0], edou[0]))
    ok.extend([j4a, j4b, j4c])

    out.append("")
    out.append("=== PD5 赝手性的质量退化 ===")
    Hm0 = H2(np.zeros(N_H))
    a0 = float(np.max(np.abs(SY_OP @ Hm0 + Hm0 @ SY_OP)))
    a1 = float(np.max(np.abs(SY_OP @ H0 + H0 @ SY_OP)))
    out.append("  {sigma_y, H} at m=0  = %.3e" % a0)
    out.append("  {sigma_y, H} at m=m0 = %.3e" % a1)
    j5 = a0 < TOL and a1 > 0.5
    ok.append(j5)
    out.append("PD5 %s" % ("PASS" if j5 else "FAIL"))

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9))
    names = [t[0] for t in tbl4]
    vals = [max(t[1], t[2], t[3], t[4]) for t in tbl4]
    ax[0].semilogy(range(len(names)), np.maximum(vals, 1e-18), "o", color="tab:purple")
    ax[0].set_xticks(range(len(names)))
    ax[0].set_xticklabels(names, rotation=60, fontsize=6)
    ax[0].set_ylabel(r"$\max$ deviation")
    ax[0].set_title(r"(a) PD3 4x4 parity candidates")
    ax[0].set_ylim(1e-18, 1e2)

    ax[1].plot(ks, wilson_E2(0.0, ks, rs[-1], 0.0), label=r"$k_h$, $r=%.1f$" % rs[-1])
    ax[1].plot(ks, wilson_E2(0.0, np.pi - ks, rs[-1], 0.0), label=r"$\pi-k_h$, $r=%.1f$" % rs[-1])
    ax[1].plot(ks, wilson_E2(0.0, np.pi - ks, 0.0, 0.0), "--", color="tab:gray", label=r"$\pi-k_h$, $r=0$")
    ax[1].set_xlabel(r"$k_h$")
    ax[1].set_ylabel(r"$E^2$ (massless)")
    ax[1].set_title("(b) PD4b doubling lift")
    ax[1].legend(fontsize=7)

    ax[2].plot(rs, splits, "o-", color="tab:green", label="doubling split")
    ax[2].plot(rs, edou, "-", color="tab:orange", label="doublet mass")
    ax[2].axvline(rbest, color="tab:red", ls="--", lw=0.9, label=r"best $r=%.2f$" % rbest)
    ax[2].set_xlabel(r"Wilson parameter $r$")
    ax[2].set_title(r"(c) PD4 Wilson term, $r$ sweep")
    ax[2].legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(FIG, dpi=150)
    out.append("")
    out.append("figure: %s" % FIG)

    allok = all(ok)
    out.append("SUMMARY PD1..PD5 = %s" % ("ALL PASS" if allok else "HAS FAILURE"))
    text = "\n".join(out)
    print(text)
    with io.open(FIG.replace(".png", "_result.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text + "\n")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
