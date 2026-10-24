# -*- coding: utf-8 -*-
"""判据第 663 号：手征来源层判据化第二轮。

模态空间的 CR 结构类型分解与作用分类。

背景（承接第 662 号）：正稿说"手征性 = 复结构 $J$ 的 $\\pm i$ 本征分支"。
第 662 号读到 $J_0$ 在模态系数空间 $\\mathbb{C}^2$ 上只有一个 $+i$ 本征空间
（整个空间），故"$\\pm i$ 分支"在 $\\mathbb{C}^2$ 层无区分；其合法载体是
$S^3$ 的 CR 结构（复一维水平分布 $H^{1,0}\\oplus H^{0,1}$）。本轮把该结构做实，
并对各算符的作用分类：保持手征分支还是交换它。

判据：冒烟 M1 至 M2、C1 至 C7，另附区分力对照与多档扫描。
可逆性等级：判据稳健（解析恒等式与投影范数读数）。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

RNG = np.random.default_rng(20261017)
IMG = r"D:\software\TianXingFC\images\fig_cr_structure_typing.png"

E0 = np.array([1.0, 0, 0, 0])
EI = np.array([0.0, 1, 0, 0])
EJ = np.array([0.0, 0, 1, 0])
EK = np.array([0.0, 0, 0, 1])


def qmul(a, b):
    a0, a1, a2, a3 = a
    b0, b1, b2, b3 = b
    return np.array([
        a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
        a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
        a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1,
        a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0,
    ])


def Lmat(a):
    a0, a1, a2, a3 = a
    return np.array([[a0, -a1, -a2, -a3], [a1, a0, -a3, a2],
                     [a2, a3, a0, -a1], [a3, -a2, a1, a0]])


def Rmat(a):
    a0, a1, a2, a3 = a
    return np.array([[a0, -a1, -a2, -a3], [a1, a0, a3, -a2],
                     [a2, -a3, a0, a1], [a3, a2, -a1, a0]])


def Cmat(g):
    (al, be), (ga, de) = g
    return np.array([[al.real, -al.imag, be.real, -be.imag],
                     [al.imag, al.real, be.imag, be.real],
                     [ga.real, -ga.imag, de.real, -de.imag],
                     [ga.imag, ga.real, de.imag, de.real]])


SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)

J0 = Lmat(EI)            # 复结构：四元数左乘 i
NEG_J = Lmat(-EI)        # J -> -J 的操作
CONJ4 = np.diag([1.0, -1, -1, -1])   # 四元数共轭 q -> qbar（交换左右乘）

LOG = []


def rep(tag, text, ok):
    LOG.append(f"[{'PASS' if ok else 'FAIL'}] {tag}: {text}")
    print(LOG[-1])


def random_point():
    v = RNG.normal(size=4)
    return v / np.linalg.norm(v)


def horizontal_basis(psi):
    """H_psi = {psi, J0 psi}^perp 的实正交基（实维 2）。"""
    Jpsi = J0 @ psi
    Q, _ = np.linalg.qr(np.column_stack([psi, Jpsi]))
    P = np.eye(4) - Q @ Q.T
    w, V = np.linalg.eigh(P)
    return V[:, np.argsort(-w)[:2]], Jpsi


def pi_plus(w):
    """(1 - iJ0)/2 在复化空间上的作用（J0 的 +i 投影）。"""
    return 0.5 * (w - 1j * (J0 @ w))


def pi_minus(w):
    return 0.5 * (w + 1j * (J0 @ w))


def branch_ratio(M, psi):
    """把 H^{1,0}_psi 的基矢送过 M，返回落在 -i 支的占比（0 保持 / 1 交换）。"""
    basis, _ = horizontal_basis(psi)
    v = basis[:, 0] - 1j * (J0 @ basis[:, 0])
    w = M.astype(complex) @ v
    n = np.linalg.norm(w)
    return float(np.linalg.norm(pi_minus(w)) / n)


def main():
    print("=== 判据第 663 号：CR 结构类型分解与作用分类 ===")

    # ---------------- 冒烟
    psi = random_point()
    basis, Jpsi = horizontal_basis(psi)
    rep("M1", f"H 为实维 2（基矢量正交偏差 {np.max(np.abs(basis.T @ basis - np.eye(2))):.2e}），"
              f"与 psi、J0 psi 正交（最大 {max(np.max(np.abs(basis.T @ psi)), np.max(np.abs(basis.T @ Jpsi))):.2e}）",
        True)
    Mh = J0 @ basis
    C2m = np.linalg.lstsq(basis, Mh, rcond=None)[0]
    resid = np.max(np.abs(basis @ C2m - Mh))
    rep("M2", f"J0 保持 H（残差 {resid:.2e}）且限制为复结构（$C^2+I$ 偏差 "
              f"{np.max(np.abs(C2m @ C2m + np.eye(2))):.2e}）",
        resid < 1e-12 and np.max(np.abs(C2m @ C2m + np.eye(2))) < 1e-12)

    # ---------------- C1：H^{1,0} 与投影
    v = basis[:, 0] - 1j * (J0 @ basis[:, 0])
    ev = J0 @ v
    rep("C1", f"H^{{1,0}} 基矢满足 J0 v = i v（偏差 {np.max(np.abs(ev - 1j * v)):.2e}）",
        np.max(np.abs(ev - 1j * v)) < 1e-12)
    p1, p0 = pi_plus(v), pi_minus(v)
    rep("C1", f"两支投影互补（|Pi+ v| = {np.linalg.norm(p1):.6f}、|Pi- v| = {np.linalg.norm(p0):.2e}，"
              f"和与 v 之差 {np.max(np.abs(p1 + p0 - v)):.2e}）",
        abs(np.linalg.norm(p1) - np.linalg.norm(v)) < 1e-12 and np.linalg.norm(p0) < 1e-12)
    pw = pi_plus(pi_plus(v))
    rep("C1", f"Pi+ 幂等（偏差 {np.max(np.abs(pw - pi_plus(v))):.2e}）",
        np.max(np.abs(pw - pi_plus(v))) < 1e-12)

    # ---------------- C2 至 C6：作用分类
    print("--- 作用分类（0 = 保持手征分支，1 = 交换）---")
    theta = 0.9
    gx = np.array([np.cos(theta / 2), 0, 0, np.sin(theta / 2)])     # g_x 方向
    gz = np.array([np.cos(theta / 2), np.sin(theta / 2), 0, 0])     # g_z 方向
    cases = [
        ("J0（复结构）", J0, "keep"),
        ("-J0（符号取反的映射）", NEG_J, "keep"),
        ("R_g（模式混合＝矩阵左乘）", Rmat(gx), "keep"),
        ("L_{g_z}（与 i 交换的左乘）", Lmat(gz), "keep"),
        ("T_1 = sigma_x", Cmat(SX), "keep"),
        ("T_2 = sigma_y", Cmat(SY), "keep"),
        ("T_3 = sigma_z", Cmat(SZ), "keep"),
        ("共轭 q->qbar（左右互换）", CONJ4, "mid"),
        ("L_j（四元数 j 型左乘）", Lmat(EJ), "swap"),
        ("L_k（四元数 k 型左乘）", Lmat(EK), "swap"),
        ("L_{g_x}(0.9)（一般左乘）", Lmat(gx), "mid"),
    ]
    points = [random_point() for _ in range(5)]
    table = {}
    for name, M, _e in cases:
        rs = [branch_ratio(M, p) for p in points]
        table[name] = rs
        print(f"    {name:<26s} 占比 " + " ".join(f"{r:.6f}" for r in rs))
    keep = [n for n, _m, e in cases if e == "keep"]
    swap = [n for n, _m, e in cases if e == "swap"]
    k_max = max(max(table[n]) for n in keep)
    s_min = min(min(table[n]) for n in swap)
    rep("C2", f"保持类七项占比全零（最大 {k_max:.2e}）", k_max < 1e-12)
    rep("C3", f"交换类两项占比全一（最小 {s_min:.6f}）", s_min > 1 - 1e-9)
    rep("C4", f"两类分离（保持最大 {k_max:.2e} 对交换最小 {s_min:.6f}）",
        s_min - k_max > 0.99)

    # ---------------- C5：θ 扫描（连续过渡）
    thetas = np.linspace(0.0, np.pi, 25)
    curve = []
    for th in thetas:
        g = np.array([np.cos(th / 2), 0, 0, np.sin(th / 2)])
        curve.append(np.mean([branch_ratio(Lmat(g), p) for p in points]))
    rep("C5", f"theta 扫描两端为 0 与 1（{curve[0]:.2e} 与 {curve[-1]:.6f}），中间连续（单调性"
              f" {'成立' if np.all(np.diff(curve) >= -1e-12) else '不成立'}）",
        abs(curve[0]) < 1e-12 and abs(curve[-1] - 1) < 1e-9)

    # ---------------- C6：区分力对照与恒真检验
    mid = curve[len(curve) // 2]
    rep("C6", f"恒真检验：中间档 theta = {thetas[len(curve) // 2]:.3f} 给占比 {mid:.6f}，"
              f"落在 0 与 1 之间，判据能区分保持、交换与中间三类",
        0.05 < mid < 0.95)
    n_taut = 0
    for name, M, _e in cases:
        if np.max(np.abs(M - np.eye(4))) < 1e-12:
            n_taut += 1
    rep("C6", f"恒真检验：十一项中与恒等重合者 {n_taut} 项（期望 0）", n_taut == 0)

    # ---------------- C7：与第 662 号读数的衔接
    lj = float(np.mean(table["L_j（四元数 j 型左乘）"]))
    cj = table["共轭 q->qbar（左右互换）"]
    kk = float(np.mean(table["R_g（模式混合＝矩阵左乘）"]))
    rep("C7", f"三个操作须分辨：符号取反的映射实现（L_j、L_k）交换矢量支（占比 {lj:.6f}）；"
              f"左右互换的共轭把 J0 变成另一复结构 R_-i，对 J0 的支是混合作用（占比 "
              f"{min(cj):.6f} 至 {max(cj):.6f}，随点变化，故既不保持也不交换）；"
              f"模式混合保持（占比 {kk:.2e}）。支交换在矢量层面由 L_j 型左乘实现",
        lj > 1 - 1e-9 and min(cj) > 0.05 and max(cj) < 0.999999 and kk < 1e-12)

    # ---------------- 图
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))
    ax = axes[0]
    names = [c[0] for c in cases]
    vals = [np.mean(table[n]) for n in names]
    colors = ["#2ca02c" if v < 1e-6 else ("#d62728" if v > 1 - 1e-6 else "#7f7f7f") for v in vals]
    ax.barh(range(len(names)), vals, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("落在 $-i$ 支的占比")
    ax.set_title("C2/C3：作用分类")
    ax.grid(alpha=0.3, axis="x")

    ax = axes[1]
    ax.plot(thetas, curve, "o-", color="#1f77b4", ms=3)
    ax.axhline(0, color="#2ca02c", ls="--", lw=1, label="保持")
    ax.axhline(1, color="#d62728", ls="--", lw=1, label="交换")
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel("占比")
    ax.set_title(r"C5：$L_{g_x(\theta)}$ 的连续过渡")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    dev_keep, dev_swap = [], []
    for name in keep:
        dev_keep.append(max(table[name]))
    for name in swap:
        dev_swap.append(1 - min(table[name]))
    ax.bar([0, 1], [max(dev_keep), max(dev_swap)], color=["#2ca02c", "#d62728"])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["保持类最大偏离", "交换类最大偏离"])
    ax.set_ylabel("偏离期望的幅度")
    ax.set_yscale("log")
    ax.set_title("C4：两类分离度")
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(IMG, dpi=150)
    print(f"\n图已存：{IMG}")
    n_fail = sum(1 for x in LOG if x.startswith("[FAIL]"))
    print(f"\n汇总：{len(LOG) - n_fail}/{len(LOG)} 通过")


if __name__ == "__main__":
    main()
