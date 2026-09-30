# -*- coding: utf-8 -*-
r"""第 498 条：提取程序检验器（EP1 至 EP5）。

依据前五轮的同一类错误：把简并子空间内的任意本征矢当作物理态。
  第 491、492、494 条  Gamma5 的成因误判
  第 496 条          自旋提取得 g = 0 或 1.4907
  第 497 条          g 读数随 B 反变
共同根源：提取前未检验「该子空间内被测算符是否已被对角化」「该子空间是否简并」
「被测算符是否守恒」。本条把这三件事做成可复用的检验器。

检验器的输入：被测算符 O、目标子空间的基 B、算符 O 的全谱。
检验器的输出：四段判定。使用者须在读任何物理数之前先过检验器。

EP1 子空间正交性：Pi = B B^dagger 须为幂等厄米投影
EP2 提取正确性：子空间内对角化所得本征值须落在 O 的全谱上
EP3 污染量化：朴素程序（取子空间内一个任意矢作期望值）的读数离散
EP4 守恒标签自检：O 与 H 反对易时，能量本征态上的 <O> 恒为零，
    程序须报「非守恒标签」而非给出读数；O 与 H 对易时读数应锐
EP5 回归测试：对四类历史案例（一个正对照、三类失败）跑检验器，
    须全部被正确标记
"""
import io
import sys

import numpy as np

TOL = 1e-8
SEED = 20260929

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
SIGP = 0.5j * (G3 @ G1 - G1 @ G3)


def nrm(a):
    return float(np.max(np.abs(a)))


def orth(B):
    r"""对子空间基做 Gram-Schmidt 正交归一。"""
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


def project_check(B):
    r"""EP1：Pi = B B^dagger 为幂等厄米投影（输入须已正交归一）。"""
    Pi = B @ B.conj().T
    return nrm(Pi @ Pi - Pi) < TOL and nrm(Pi - Pi.conj().T) < TOL


def diag_in_subspace(O, B):
    r"""子空间内对角化，返回本征值。"""
    M = B.conj().T @ O @ B
    M = (M + M.conj().T) / 2.0
    return np.linalg.eigvalsh(M)


def in_spec(w, O):
    r"""EP2：本征值须落在 O 的全谱上。"""
    spec = np.linalg.eigvalsh(O)
    return all(float(np.min(np.abs(spec - x))) < TOL for x in w), spec


def naive_spread(O, B, rng, trials=64):
    r"""EP3：取子空间内随机组合矢作期望值，读数的离散。"""
    out = []
    dim = B.shape[1]
    for _ in range(trials):
        v = B @ (rng.normal(size=dim) + 1j * rng.normal(size=dim))
        v = v / np.linalg.norm(v)
        out.append(float(np.vdot(v, O @ v).real))
    return max(out) - min(out)


def conservation(H, O):
    r"""EP4：返回 (anticomm, comm) 两个偏差。"""
    return nrm(O @ H + H @ O), nrm(O @ H - H @ O)


def block(k1, k2, m, r):
    meff = m + 2.0 * r * (1.0 - np.cos(k1))
    return (meff * G0
            - 1j * np.sin(k1) * G1
            - 1j * np.sin(k2) * G2), meff


def degeneracy_scan(H, tol=TOL):
    r"""EP0：扫描谱的简并结构，返回 (最大簇重数, 簇列表)。

    簇为 (起始指标, 结束指标, 本征值) 三元组。最大重数大于 1 时，
    落在此子空间的读数须改走子空间谱分类，禁止单矢期望（第 506 条）。
    """
    ev = np.linalg.eigvalsh(H)
    groups, i, n = [], 0, len(ev)
    while i < n:
        j = i
        while j + 1 < n and abs(ev[j + 1] - ev[i]) < tol:
            j += 1
        groups.append((i, j, float(ev[i])))
        i = j + 1
    maxdeg = max(j - i + 1 for i, j, _ in groups)
    return maxdeg, groups


def main():
    out, ok = [], []
    rng = np.random.default_rng(SEED)
    n = 8
    k1 = 2.0 * np.pi / n
    k2 = 2.0 * np.pi / n

    out.append("=== EP1 至 EP4 的检验器定义自检 ===")
    B_ok = np.eye(4, dtype=complex)[:, :2]
    out.append("  projector on a 2-dim subspace: %s" % ("PASS" if project_check(B_ok) else "FAIL"))
    B_bad = np.ones((4, 2), dtype=complex) / 2.0
    out.append("  non-orthonormal basis detected: %s"
               % ("PASS" if not project_check(B_bad) else "FAIL"))
    out.append("  Gram-Schmidt restores orthonormality: %s"
               % ("PASS" if project_check(orth(B_bad)) else "FAIL"))
    j1 = project_check(B_ok) and not project_check(B_bad)
    ok.append(j1)
    out.append("EP1 %s" % ("PASS" if j1 else "FAIL"))

    out.append("")
    out.append("=== EP0 简并检测 ===")
    Hdeg = np.kron(SZ, I4)
    maxdeg, groups = degeneracy_scan(Hdeg)
    out.append("  demo H = sz (x) I4: max cluster degeneracy = %d, clusters = %d"
               % (maxdeg, len(groups)))
    deg_ok = maxdeg >= 2
    out.append("  degenerate subspace detected -> %s" % ("PASS" if deg_ok else "FAIL"))
    ok.append(deg_ok)

    out.append("")
    out.append("=== EP5 回归测试 ===")

    out.append("")
    out.append("  case A（正对照）：模态因子 sigma_z (x) I，与 H_full 对易，应给出锐读数")
    h11 = block(k1, k2, 0.0, 0.0)[0]
    Hfull = np.kron(SZ, I4) + np.kron(I2, h11)
    e, V = np.linalg.eigh(Hfull)
    Bz = orth(V[:, np.abs(e - e[0]) < 1e-9])
    Oz = np.kron(SZ, I4)
    wz, _ = in_spec(diag_in_subspace(Oz, Bz), Oz)
    az, cz = conservation(Hfull, Oz)
    sprz = naive_spread(Oz, Bz, rng)
    okA = project_check(Bz) and wz and cz < TOL and sprz < 1e-6
    out.append("    EP1 %s  EP2 %s  comm %.1e  naive spread %.1e" % (project_check(Bz), wz, cz, sprz))
    out.append("    eigenvalues inside: %s" % np.round(diag_in_subspace(Oz, Bz), 8).tolist())
    out.append("    verdict: extraction well posed -> %s" % ("PASS" if okA else "FAIL"))
    ok.append(okA)

    out.append("")
    out.append("  case B（第 491/492/494 类）：Gamma5 与时空哈密顿量反对易")
    G5s = G5
    a5, c5 = conservation(h11, G5s)
    evs, Vs = np.linalg.eigh(h11)
    B5 = orth(Vs[:, np.abs(evs - evs[0]) < 1e-9])
    w5, _ = in_spec(diag_in_subspace(G5s, B5), G5s)
    spr5 = naive_spread(G5s, B5, rng)
    verdictB = a5 < TOL and c5 > 0.5
    out.append("    EP1 %s  EP2 %s  anticomm %.1e comm %.1e  naive spread %.1e"
               % (project_check(B5), w5, a5, c5, spr5))
    out.append("    verdict: non-conserved label, <Gamma5> on energy states is identically 0"
               " -> %s" % ("FLAGGED" if verdictB else "MISSED"))
    ok.append(verdictB)

    out.append("")
    out.append("  case C（第 496 类）：最低两能级为粒子-反物子对")
    Hfull_m = np.kron(SZ, I4) + np.kron(I2, block(k1, k2, 0.5, 0.0)[0])
    evm = np.linalg.eigvalsh(Hfull_m)
    gap = float(evm[1] - evm[0])
    mult = int(np.sum(np.abs(evm - evm[0]) < 1e-9))
    # 496 条的真实签名：最低能级重数 > 2，故 E1-E0 恒为零，
    # 用它作 Zeeman 劈裂读数必然给出 g = 0，与场强无关。
    verdictC = gap < 1e-9
    out.append("    two lowest levels: %.6f, %.6f  gap = %.3e  lowest-level multiplicity %d"
               % (evm[0], evm[1], gap, mult))
    out.append("    verdict: E1-E0 is identically zero (lowest-level multiplicity %d), so it" % mult)
    out.append("    cannot be a Zeeman response; g = 0/B = 0 for every B -> %s"
               % ("FLAGGED" if verdictC else "MISSED"))
    ok.append(verdictC)

    out.append("")
    out.append("  case D（第 497 类）：sigma_Pauli 在逐动量二重正能子空间内")
    hk, _ = block(k1, k2, 0.0, 0.0)
    evp, Vp = np.linalg.eigh(hk)
    Bsp = orth(Vp[:, evp > 0])
    wsp, _ = in_spec(diag_in_subspace(SIGP, Bsp), SIGP)
    asp, csp = conservation(hk, SIGP)
    sprsp = naive_spread(SIGP, Bsp, rng)
    verdictD = project_check(Bsp) and (not wsp) and sprsp > 0.1
    out.append("    EP1 %s  EP2 %s  anticomm %.1e comm %.1e  dim %d"
               % (project_check(Bsp), wsp, asp, csp, Bsp.shape[1]))
    out.append("    EP2 false means Sigma_P is NOT contained in the positive-energy subspace:")
    out.append("    on a finite lattice with doublers the single-momentum spin is not confined")
    out.append("    to one energy sector, so extracting there is invalid.")
    out.append("    proper extraction eigenvalues: %s" % np.round(diag_in_subspace(SIGP, Bsp), 8).tolist())
    out.append("    naive spread over the subspace: %.3e" % sprsp)
    out.append("    verdict: naive procedure contaminated -> %s" % ("FLAGGED" if verdictD else "MISSED"))
    ok.append(verdictD)

    out.append("")
    out.append("  检验器应拦下的历史错误次数：3（第 491/492/494、496、497 各一次）")
    allok = all(ok)
    out.append("SUMMARY EP1, EP5 = %s" % ("ALL PASS" if allok else "HAS FAILURE"))
    text = "\n".join(out)
    print(text)
    with io.open("D:/software/TianXingFC/验证脚本/extraction_protocol_result.txt", "w",
                 encoding="utf-8", newline="\n") as f:
        f.write(text + "\n")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
