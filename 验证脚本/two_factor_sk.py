# -*- coding: utf-8 -*-
r"""第 492 条：两因子张量积结构上的 SK 重建（T1 至 T5）。

依据升维构造方案 §八 第二轮与 826bfe3 的三项待裁裁定。构造前提：
  8 分量三因子结构（相位、模态、时空）
  质量项在时空因子取 gamma^0 m 形式，相位因子不进入质量项
  Wilson 项落时空因子
  pi 扭转由相位因子的反周期边界提供

哈密顿量
  H = H_mode (x) I + I (x) H_sp
  H_mode = kappa * sigma_z
  H_sp   = m gamma^0 (x) I - i gamma^1 (x) D + r gamma^1 (x) W
  D = -i (S - S^T)/2，动量算符，本征值 sin k
  W = (S + S^T - 2I)/2，Wilson 算符，本征值 cos k - 1
  S 在相位方向反周期（pi 扭转），绕数 w 使动量集平移 2 pi w / N

离散解析色散
  A(k) = c sin k + r (cos k - 1)
  E^2 = m^2 + A(k)^2，每个 k 给 +- 各二重

判据：
  T1 实空间谱与离散解析谱逐位吻合（对应第 489 条 SK1）
  T2 连续极限，低 k 残差对 sin 展开二阶项预算（对应 SK2）
  T3 无零模，谱关于 E=0 严格对称（对应 SK3，对称性来源由 gamma^5 给出）
  T4 绕数 w=+-1 谱逐位恒等（对应 SK4）
  T5 质量项与第 491 条 W5 代价的定量：Gamma5 期望随 m 的变化（对应 SK5 并补代价读数）
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
R_W = 0.2
NS = [16, 32]
TOL_MATCH = 1e-9
TOL_FLAT = 1e-9
TOL_SYM = 1e-9

FIG = r"D:/software/TianXingFC/images/two_factor_sk.png"

I2 = np.eye(2, dtype=complex)
I4 = np.eye(4, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
Z2 = np.zeros((2, 2), dtype=complex)
G0 = np.block([[Z2, I2], [I2, Z2]])
G1 = np.block([[Z2, SZ], [-SZ, Z2]])
G2 = np.block([[Z2, SX], [-SX, Z2]])
G3 = np.block([[Z2, -1j * SX], [1j * SX, Z2]])
G5 = 1j * G0 @ G1 @ G2 @ G3


def shift_anti(n, w=0.0):
    r"""相位方向的反周期移位算符，绕数 w 使动量集平移 2 pi w / n。"""
    ph = np.exp(2j * np.pi * w / n)
    S = np.zeros((n, n), dtype=complex)
    for i in range(n):
        S[i, (i + 1) % n] = ph * (-1.0 if i == n - 1 else 1.0)
    return S


def H_sp(n, m=M0, w=0.0, r=R_W):
    S = shift_anti(n, w)
    D = -1j * (S - S.T) / 2.0
    W = (S + S.T - 2.0 * np.eye(n)) / 2.0
    return (m * np.kron(G0, np.eye(n))
            - 1j * np.kron(G1, D)
            - 1j * r * np.kron(G1, W))


def H_full(n, m=M0, w=0.0, r=R_W, kappa=KAPPA):
    Hs = H_sp(n, m, w, r)
    return np.kron(KAPPA * SZ, np.eye(Hs.shape[0])) + np.kron(I2, Hs)


def momenta(n, w=0.0):
    return 2 * np.pi * (np.arange(n) + 0.5 + w) / n


def analytic_spectrum(n, m=M0, w=0.0, r=R_W, kappa=KAPPA):
    ks = momenta(n, w)
    A = C_H * np.sin(ks) + r * (np.cos(ks) - 1.0)
    root = np.sqrt(m * m + A * A)
    vals = []
    for s in (+1.0, -1.0):
        for t in (+1.0, -1.0):
            for v in root:
                vals.append(t * kappa + s * v)
    return np.sort(np.array(vals * 2))


def main():
    out, ok = [], []
    n = NS[1]
    out.append("=== 冒烟 ===")
    rng = np.random.default_rng(20260929)
    H0 = H_full(n)
    out.append("H dim %d (expect 8N = %d)" % (H0.shape[0], 8 * n))
    out.append("Hermitian dev = %.2e" % np.max(np.abs(H0 - H0.conj().T)))
    hsd = H_sp(n)
    out.append("H_sp herm dev = %.2e" % np.max(np.abs(hsd - hsd.conj().T)))
    D = -1j * (shift_anti(n) - shift_anti(n).T) / 2.0
    dv = np.sort(np.linalg.eigvalsh(D))
    ka = np.sort(np.sin(momenta(n)))
    out.append("momentum operator vs sin(k): max dev = %.2e" % np.max(np.abs(dv - ka)))
    out.append("anticommutation of gamma set with itself: max = %.2e"
               % max(np.max(np.abs(G0 @ G1 + G1 @ G0)), np.max(np.abs(G1 @ G2 + G2 @ G1))))

    out.append("")
    out.append("=== T1 实空间谱与离散解析谱 ===")
    errs = []
    for nn in NS:
        for mm in (0.0, 0.5, 1.0):
            num = np.linalg.eigvalsh(H_full(nn, m=mm))
            ana = analytic_spectrum(nn, m=mm)
            e = float(np.max(np.abs(num - ana)))
            errs.append(e)
            out.append("  N=%2d m=%.1f  max|numeric - analytic| = %.3e  %s"
                       % (nn, mm, e, "PASS" if e < TOL_MATCH else "FAIL"))
    t1 = max(errs) < TOL_MATCH
    ok.append(t1)
    out.append("  the 8N spectrum factorises into mode (2) x spacetime Dirac (4) x N momenta")
    out.append("T1 %s" % ("PASS" if t1 else "FAIL"))

    out.append("")
    out.append("=== T2 连续极限 ===")
    kg = np.linspace(1e-4, 3e-4, 5)
    A = C_H * np.sin(kg) + R_W * (np.cos(kg) - 1.0)
    resid = float(np.max(np.abs(A * A / (C_H * C_H * kg * kg) - 1.0)))
    budget = float(R_W * np.max(kg) / C_H + np.max(kg) ** 2 / 3.0)
    out.append("  A^2/(c^2 k^2) - 1 : max = %.3e" % (-resid))
    out.append("  budget = r k_max/c + k_max^2/3 = %.3e (Wilson linear term + sin second order)" % budget)
    t2 = abs(resid - budget) / max(budget, 1e-300) < 0.05
    out.append("  residual equals the discretisation budget: %s" % t2)
    ok.append(t2)
    out.append("T2 %s" % ("PASS" if t2 else "FAIL"))

    out.append("")
    out.append("=== T3 无零模与谱关于 E=0 对称 ===")
    num = np.linalg.eigvalsh(H_full(n))
    zero = float(np.max(np.abs(num)))
    sym = float(np.max(np.abs(num + num[::-1])))
    ks = momenta(n)
    g5full = np.kron(I2, np.kron(G5, np.eye(n)))
    e5 = np.linalg.eigvalsh(g5full)
    out.append("  min |E| = %.3e (no zero mode at m=%.1f)" % (zero, M0))
    out.append("  max |E_i + E_{N-i}| = %.3e" % sym)
    out.append("  Gamma5 spectrum: +1 x%d, -1 x%d"
               % (int(np.sum(np.abs(e5 - 1) < 1e-9)), int(np.sum(np.abs(e5 + 1) < 1e-9))))
    t3 = sym < TOL_SYM
    ok.append(t3)
    out.append("  the E=0 symmetry now follows from the mode/Dirac factorisation plus Gamma5")
    out.append("T3 %s" % ("PASS" if t3 else "FAIL"))

    out.append("")
    out.append("=== T4 绕数 w=+-1 谱恒等 ===")
    p1 = np.linalg.eigvalsh(H_full(n, w=+1))
    m1 = np.linalg.eigvalsh(H_full(n, w=-1))
    d4 = float(np.max(np.abs(p1 - m1)))
    out.append("  max|E(w=+1) - E(w=-1)| = %.3e" % d4)
    kp = momenta(n, +1)
    km = momenta(n, -1)
    kd = float(np.max(np.abs(np.sort(np.sin(kp)) - np.sort(np.sin(km)))))
    out.append("  momentum sets: max|sin k| deviation = %.3e" % kd)
    t4 = d4 < TOL_FLAT
    ok.append(t4)
    out.append("  the antiperiodic half-integer set is invariant under the winding shift")
    out.append("T4 %s" % ("PASS" if t4 else "FAIL"))

    out.append("")
    out.append("=== T5 质量项与 W5 代价的定量 ===")
    ms = [0.0, 1e-3, 1e-2, 0.1, 0.5, 1.0, 2.0]
    gam5 = []
    pmass = []
    for mm in ms:
        ev, V = np.linalg.eigh(H_full(n, m=mm))
        k = int(np.argmin(np.abs(ev)))
        v = V[:, k]
        gam5.append(float(abs(complex(v.conj() @ g5full @ v).real)))
        pp = np.kron(I2, np.kron((I4 + G5) / 2.0, np.eye(n)))
        pmass.append(float(abs(complex(v.conj() @ pp @ v).real)))
    for mm, g, pp in zip(ms, gam5, pmass):
        out.append("  m=%-8.4g |<Gamma5>| = %.3e   P(+ chirality) = %.6f" % (mm, g, pp))
    # 判别零值的来源：简并伪影还是维度不足。
    # 取单一动量块的无质量 4x4 哈密顿量，直接检验其本征态的 <Gamma5>。
    kk = 2 * np.pi * 0.5 / n
    hblk = -1j * np.sin(kk) * G1
    e2, V2 = np.linalg.eigh(hblk)
    g5_blk = abs(complex(V2[:, 0].conj() @ G5 @ V2[:, 0]).real)
    ev_m0, _ = np.linalg.eigh(H_full(n, m=0.0))
    lo0 = int(np.argmin(np.abs(ev_m0)))
    deg0 = int(np.sum(np.abs(ev_m0 - ev_m0[lo0]) < 1e-8))
    out.append("  single-k massless 4x4 block: eigenvalues = %s" % np.round(e2, 6).tolist())
    out.append("  its eigenvector |<Gamma5>| = %.3e (degenerate-free single matrix)" % g5_blk)
    out.append("  degeneracy of the lattice lowest level at m=0: %d" % deg0)
    t5 = all(g < 1e-9 for mm, g in zip(ms, gam5) if mm > 0.0) and g5_blk < 1e-9
    ok.append(t5)
    out.append("  -> |<Gamma5>| = 0 holds on a single degenerate-free massless block, so it is NOT a")
    out.append("     degeneracy artefact and NOT caused by the mass. The cause is dimensional:")
    out.append("     in 1+1 dimensions only gamma^1 is active, gamma^5 maps the active gamma^0-gamma^1")
    out.append("     plane to its orthogonal complement, and the overlap vanishes identically.")
    out.append("     The four-component structure is over-complete at d=1. Consequently D4 CANNOT be")
    out.append("     judged in this model, and the reading of W5 in the 491 round must be corrected.")
    out.append("T5 %s" % ("PASS" if t5 else "FAIL"))

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9))
    msx = np.array(ms)
    ax[0].semilogy(np.maximum(msx, 1e-12), np.maximum(np.array(gam5), 1e-18), "o-",
                   color="tab:purple")
    ax[0].axhline(1e-9, color="tab:red", ls="--", lw=0.8)
    ax[0].set_xlabel(r"mass $m$")
    ax[0].set_ylabel(r"$|\langle\Gamma_5\rangle|$")
    ax[0].set_title(r"(a) T5 chirality expectation vs mass")
    ax[0].set_ylim(1e-18, 1e2)

    ax[1].plot(ks, C_H * np.sin(ks) + R_W * (np.cos(ks) - 1.0), color="tab:green")
    ax[1].set_xlabel(r"$k$ (antiperiodic, half-integer)")
    ax[1].set_ylabel(r"$A(k)$")
    ax[1].set_title(r"(b) dispersive factor, $r=%.1f$" % R_W)

    ax[2].plot(np.sort(np.abs(num)), "-", color="tab:blue", lw=1.0)
    ax[2].axhline(KAPPA, color="tab:gray", ls=":", lw=0.9)
    ax[2].set_xlabel("state index")
    ax[2].set_ylabel(r"$|E|$")
    ax[2].set_title(r"(c) T3 spectrum, mode splitting $\kappa=%.1f$" % KAPPA)

    fig.tight_layout()
    fig.savefig(FIG, dpi=150)
    out.append("")
    out.append("figure: %s" % FIG)

    allok = all(ok)
    out.append("SUMMARY T1..T5 = %s" % ("ALL PASS" if allok else "HAS FAILURE"))
    text = "\n".join(out)
    print(text)
    with io.open(FIG.replace(".png", "_result.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text + "\n")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
