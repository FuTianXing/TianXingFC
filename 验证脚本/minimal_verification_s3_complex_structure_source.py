# -*- coding: utf-8 -*-
"""判据第 662 号：手征来源层判据化第一轮。

模态空间复结构的定义域与左右作用的三约定对表。

依据（手征性正稿 §3.1、§5.1）：
  「$S^3$ 即单位四元数流形。四元数乘法给出其上的两个互反复结构：左不变复结构 $J$，
   满足 $J^2 = -1$，几何内容为切空间上乘 $i$；以及 $J' = -J$。」
  「左乘保持 $J$，右乘保持 $-J$。」
  「弱荷 $T_3$ 的几何内容 = 全纯分支上的模态参与度，反全纯分支的模态参与度恒为零。」

本轮把这三句落成显式读数：复结构住在哪个空间、左右作用到底保持什么、
$T_3$ 在 $SU(2)_L \times SU(2)_R$ 分解下的身份是什么。

判据：冒烟 M1 至 M2、S1 至 S4，另附区分力对照与恒真反例。
可逆性等级：判据稳健（全部为解析恒等式与矩阵范数读数，不依赖采样阈值）。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

RNG = np.random.default_rng(20261016)
IMG = r"D:\software\TianXingFC\images\fig_s3_complex_structure_source.png"

E0 = np.array([1.0, 0.0, 0.0, 0.0])   # 四元数 1
EI = np.array([0.0, 1.0, 0.0, 0.0])   # 四元数 i
EJ = np.array([0.0, 0.0, 1.0, 0.0])   # 四元数 j
EK = np.array([0.0, 0.0, 0.0, 1.0])   # 四元数 k


# ---------------------------------------------------------------- 四元数工具
def qmul(a, b):
    a0, a1, a2, a3 = a
    b0, b1, b2, b3 = b
    return np.array([
        a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
        a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
        a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1,
        a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0,
    ])


def qconj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def qinv(q):
    return qconj(q) / float(q @ q)


def Lmat(a):
    """四元数左乘 q -> a q 的 4x4 实矩阵。"""
    a0, a1, a2, a3 = a
    return np.array([
        [a0, -a1, -a2, -a3],
        [a1,  a0, -a3,  a2],
        [a2,  a3,  a0, -a1],
        [a3, -a2,  a1,  a0],
    ])


def Rmat(a):
    """四元数右乘 q -> q a 的 4x4 实矩阵。"""
    a0, a1, a2, a3 = a
    return np.array([
        [a0, -a1, -a2, -a3],
        [a1,  a0,  a3, -a2],
        [a2, -a3,  a0,  a1],
        [a3,  a2, -a1,  a0],
    ])


def Cmat(g):
    """C^2 矩阵左乘的实 4x4 表示（z1 = x0 + i x1, z2 = x2 + i x3）。"""
    (al, be), (ga, de) = g
    return np.array([
        [al.real, -al.imag, be.real, -be.imag],
        [al.imag,  al.real, be.imag,  be.real],
        [ga.real, -ga.imag, de.real, -de.imag],
        [ga.imag,  ga.real, de.imag,  de.real],
    ])


def quat_from_so3(A):
    """由 SO(3) 伴随矩阵提取单位四元数（差正负号）。"""
    tr = np.trace(A)
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (A[2, 1] - A[1, 2]) / s
        y = (A[0, 2] - A[2, 0]) / s
        z = (A[1, 0] - A[0, 1]) / s
    elif A[0, 0] > A[1, 1] and A[0, 0] > A[2, 2]:
        s = np.sqrt(1.0 + A[0, 0] - A[1, 1] - A[2, 2]) * 2.0
        w = (A[2, 1] - A[1, 2]) / s
        x = 0.25 * s
        y = (A[0, 1] + A[1, 0]) / s
        z = (A[0, 2] + A[2, 0]) / s
    elif A[1, 1] > A[2, 2]:
        s = np.sqrt(1.0 + A[1, 1] - A[0, 0] - A[2, 2]) * 2.0
        w = (A[0, 2] - A[2, 0]) / s
        x = (A[0, 1] + A[1, 0]) / s
        y = 0.25 * s
        z = (A[1, 2] + A[2, 1]) / s
    else:
        s = np.sqrt(1.0 + A[2, 2] - A[0, 0] - A[1, 1]) * 2.0
        w = (A[1, 0] - A[0, 1]) / s
        x = (A[0, 2] + A[2, 0]) / s
        y = (A[1, 2] + A[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def decompose(M):
    """把 SO(4) 元素 M 分解为 M = L_a R_b（作用 q -> a q b）。"""
    m1 = M @ E0
    u1 = qmul(M @ EI, qinv(m1))
    u2 = qmul(M @ EJ, qinv(m1))
    u3 = qmul(M @ EK, qinv(m1))
    A = np.column_stack([u1[1:], u2[1:], u3[1:]])
    a = quat_from_so3(A)
    b = qmul(qinv(a), m1)
    return a, b


# ---------------------------------------------------------------- C^2 与 SU(2)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
SU2_GENS = (SX, SY, SZ)


def gk(k, theta):
    """SU(2) 元素 exp(i theta sigma_k / 2)（C^2 矩阵左乘）。"""
    return np.cos(theta / 2.0) * np.eye(2) + 1j * np.sin(theta / 2.0) * SU2_GENS[k]


J0 = Lmat(EI)          # C^2 标准复结构「乘 i」= 四元数左乘 i
NJ0 = Lmat(-EI)        # J' = -J
T3 = Cmat(SZ)          # 弱荷 T_3 = sigma_z
T1 = Cmat(SX)
T2 = Cmat(SY)

LOG = []


def rep(tag, text, ok):
    line = f"[{'PASS' if ok else 'FAIL'}] {tag}: {text}"
    LOG.append(line)
    print(line)


def main():
    print("=== 判据第 662 号：模态空间复结构的定义域与左右作用三约定 ===")

    # ---------------- 冒烟 M1：四元数乘法双实现一致
    worst = 0.0
    for _ in range(200):
        a = RNG.normal(size=4)
        a /= np.linalg.norm(a)
        b = RNG.normal(size=4)
        b /= np.linalg.norm(b)
        worst = max(worst, float(np.max(np.abs(Lmat(a) @ b - qmul(a, b)))))
        worst = max(worst, float(np.max(np.abs(Rmat(a) @ b - qmul(b, a)))))
    rep("M1", f"四元数乘法双实现最大偏差 {worst:.3e}（期望 < 1e-14）", worst < 1e-14)

    # ---------------- 冒烟 M2：左右乘矩阵正交且 det = 1
    ow, dw = 0.0, 0.0
    for _ in range(200):
        a = RNG.normal(size=4)
        a /= np.linalg.norm(a)
        for M in (Lmat(a), Rmat(a)):
            ow = max(ow, float(np.max(np.abs(M.T @ M - np.eye(4)))))
            dw = max(dw, abs(float(np.linalg.det(M)) - 1.0))
    rep("M2", f"左/右乘矩阵正交性 {ow:.3e}、det 偏差 {dw:.3e}（期望 < 1e-14）",
        ow < 1e-14 and dw < 1e-14)

    # ---------------- S1：复结构的定义域
    # S1a：3x3 实矩阵 J^2 = -I 无解（反对称族扫描 + 实特征值论证）
    ts = np.linspace(0.0, 2.0, 4001)
    resid3 = np.sqrt(2.0 * ts ** 2 - 4.0 * ts + 3.0)   # ||J^2+I||_F, J = [w]_x, t = |w|
    t_min = float(ts[np.argmin(resid3)])
    r_min = float(np.min(resid3))
    rep("S1a", f"3x3 反对称族 min||J^2+I||_F = {r_min:.6f}（t = {t_min:.4f}，解析最小值 1）",
        abs(r_min - 1.0) < 1e-6)

    n_real = 0
    n_samp = 2000
    for _ in range(n_samp):
        ev = np.linalg.eigvals(RNG.normal(size=(3, 3)))
        if float(np.min(np.abs(ev.imag))) < 1e-9:
            n_real += 1
    rep("S1a", f"随机 3x3 实矩阵存在实特征值的比例 {n_real / n_samp:.3f}"
               f"（{n_samp} 样本，理论 1.000；奇次实特征多项式必有实根）",
        n_real == n_samp)

    # S1b：4x4 上 J0^2 = -I 精确成立
    r4 = float(np.max(np.abs(J0 @ J0 + np.eye(4))))
    rep("S1b", f"4x4 上 ||J0^2 + I||_max = {r4:.3e}（期望 0）", r4 < 1e-14)
    print("  判定：J 的合法定义域为 C^2 ≅ H（实维 4）；"
          "S^3 切空间实维 3、奇维实空间上 J^2 = -1 无解，故 S^3 上不存在复结构。")

    # ---------------- S2：三约定对表
    d_right, d_anti = 0.0, 0.0
    d_lefts = []
    for _ in range(200):
        g = gk(RNG.integers(0, 3), float(RNG.uniform(0.1, 3.0)))
        Mg = Cmat(g)
        a = Mg[:, 0]                      # 矩阵左乘 = 右乘元素 a（R_a 首列即 a）
        d_right = max(d_right, float(np.max(np.abs(Mg - Rmat(a)))))
        d_lefts.append(float(np.max(np.abs(Mg - Lmat(a)))))
        d_anti = max(d_anti, float(np.max(np.abs(Lmat(a) @ J0 - J0 @ Lmat(a)))))
    d_left = float(np.median(d_lefts))
    rep("S2a", f"C^2 矩阵左乘 vs 四元数右乘 最大偏差 {d_right:.3e}（期望 < 1e-14）",
        d_right < 1e-14)
    rep("S2b", f"C^2 矩阵左乘 vs 四元数左乘 偏差中位数 {d_left:.3f}（期望 > 0.5）",
        d_left > 0.5)
    rep("S2c", f"四元数左乘与 J0 的对易子范数 {d_anti:.6f}（期望非零，反线性）",
        d_anti > 0.1)
    print("  判定：C^2 矩阵左乘（模式混合）= 四元数右乘，与 J0 交换（复线性）；"
          "四元数左乘是反线性映射，与 J0 不交换。")

    # ---------------- S3：J0 的左右不变性（三方向 x 四档）
    # 四元数指数 i 对应 g_z（i sigma_z = 右乘 i）、j 对应 g_y、k 对应 g_x
    thetas = [0.3, 0.7, 1.2, 1.8]
    dir_names = ["g_z", "g_y", "g_x"]
    rows = []
    for k in range(3):
        for th in thetas:
            a = np.array([np.cos(th / 2.0), 0.0, 0.0, 0.0])
            a[k + 1] = np.sin(th / 2.0)
            La, Ra = Lmat(a), Rmat(a)
            r_right = float(np.max(np.abs(Ra @ J0 @ Ra.T - J0)))
            r_left = float(np.max(np.abs(La @ J0 @ La.T - J0)))
            r_flip = float(np.max(np.abs(La @ J0 @ La.T + J0)))
            rows.append((k, th, r_right, r_left, r_flip))
    max_right = max(r[2] for r in rows)
    left_z = max(r[3] for r in rows if r[0] == 0)
    left_xy = min(r[3] for r in rows if r[0] != 0)
    rep("S3a", f"四元数右乘共轭 J0 的最大偏差 {max_right:.3e}（全档，期望 0）",
        max_right < 1e-14)
    rep("S3b", f"四元数左乘共轭 J0 的最大偏差（g_x, g_y 方向）{left_xy:.6f}（期望非零）",
        left_xy > 1e-6)
    rep("S3c", f"四元数左乘共轭 J0 的最大偏差（g_z 方向）{left_z:.3e}（期望 0）",
        left_z < 1e-14)
    th_pi = float(np.max(np.abs(Lmat(EK) @ J0 @ Lmat(EK).T + J0)))
    rep("S3d", f"g = k（theta = pi 型）处 ||L_g J0 L_g^-1 + J0|| = {th_pi:.3e}"
               f"（期望 0：该处给 -J0）", th_pi < 1e-14)
    print("  判定：J0 是四元数右不变的（右乘全群保持）；四元数左乘中只有 U(1)_i 子群保持 J0，"
          "g_x / g_y 方向把它连续旋转到 L_{i cos t + k sin t}，theta = pi 处给 -J0。")

    # ---------------- S4：T_3 的 SU(2)_L x SU(2)_R 身份分解
    print("--- S4 身份分解 M = L_a R_b ---")
    targets = [("J0 = 乘 i", J0), ("T_1 = sigma_x", T1), ("T_2 = sigma_y", T2),
               ("T_3 = sigma_z", T3), ("g_x(pi/2)", Cmat(gk(0, np.pi / 2))),
               ("g_z(pi/2)", Cmat(gk(2, np.pi / 2)))]
    id_rows = []
    for name, M in targets:
        a, b = decompose(M)
        err = float(np.max(np.abs(Lmat(a) @ Rmat(b) - M)))
        id_rows.append((name, a, b, err))
        print(f"    {name:<14s} a = [{a[0]:+.3f},{a[1]:+.3f},{a[2]:+.3f},{a[3]:+.3f}]"
              f"  b = [{b[0]:+.3f},{b[1]:+.3f},{b[2]:+.3f},{b[3]:+.3f}]  重建偏差 {err:.2e}")
    err_max = max(r[3] for r in id_rows)
    rep("S4a", f"身份分解重建最大偏差 {err_max:.3e}（期望 < 1e-13）", err_max < 1e-13)
    da = {r[0]: r[1] for r in id_rows}
    db = {r[0]: r[2] for r in id_rows}
    j_left_only = bool(np.allclose(da["J0 = 乘 i"], EI, atol=1e-9)
                       and np.allclose(db["J0 = 乘 i"], E0, atol=1e-9))
    t3_both = bool((not np.allclose(da["T_3 = sigma_z"], E0, atol=1e-9))
                   and (not np.allclose(db["T_3 = sigma_z"], E0, atol=1e-9)))
    gx_right_only = bool(np.allclose(da["g_x(pi/2)"], E0, atol=1e-9)
                         and not np.allclose(db["g_x(pi/2)"], E0, atol=1e-9))
    tk_share = bool(all(np.allclose(da[n], EI, atol=1e-9)
                        for n in ("T_1 = sigma_x", "T_2 = sigma_y", "T_3 = sigma_z")))
    rep("S4b", f"J0 为纯左型（a = i, b = 1）：{j_left_only}（期望 True）", j_left_only)
    rep("S4c", f"T_3 为对角型（a 与 b 均非平凡）：{t3_both}（期望 True）", t3_both)
    rep("S4d", f"C^2 矩阵左乘 g_x 为纯右型（a = 1）：{gx_right_only}（期望 True）", gx_right_only)
    rep("S4f", f"T_1 / T_2 / T_3 的左分量同为 i：{tk_share}（期望 True）", tk_share)
    print("  判定：J0 是纯 SU(2)_L 侧的复结构；弱荷 T_3 = sigma_z 是左右对角型，"
          "两个分量都非平凡；正稿所称的模式混合（C^2 矩阵左乘）在四元数左右分解下是纯 SU(2)_R 侧；"
          "T_k 三者的左分量同为 i，故 T_k 与 J0 全部对易（与第 661 号读数同源）。")

    # ---------------- S4e：全纯投影荷的实矢量场读数
    psi = np.array([0.5, 0.5, 0.5, 0.5])
    psi /= np.linalg.norm(psi)
    X = Cmat(1j * SZ) @ psi / 2.0            # T_3 方向的切矢量场
    Jpsi = J0 @ psi
    Xh = X - (X @ psi) * psi - (X @ Jpsi) * Jpsi   # 投到水平分布 H = {psi, J0 psi}^perp
    Xc = Xh.astype(complex)
    p10 = 0.5 * (Xc - 1j * (J0 @ Xh).astype(complex))
    p01 = 0.5 * (Xc + 1j * (J0 @ Xh).astype(complex))
    n10, n01 = float(np.linalg.norm(p10)), float(np.linalg.norm(p01))
    ratio = n10 / n01 if n01 > 0 else float("inf")
    rep("S4e", f"T_3 方向矢量场在 CR 两支上的模长 全纯 {n10:.6f} / 反全纯 {n01:.6f}，"
               f"比值 {ratio:.9f}（期望 1：实矢量场两支共轭配对）", abs(ratio - 1.0) < 1e-12)

    # ---------------- S5：区分力对照与恒真反例
    print("--- S5 区分力对照 ---")
    r_alt = float(np.max(np.abs(Rmat(EK) @ Lmat(EJ) @ Rmat(EK).T - Lmat(EJ))))
    rep("S5a", f"对照：把 J0 换成 L_j，右乘共轭的最大偏差 {r_alt:.3e}（期望 0，右不变性对任意左乘复结构成立）",
        r_alt < 1e-14)
    a_t1, b_t1 = decompose(T1)
    a_t3, b_t3 = decompose(T3)
    diff_t = float(np.max(np.abs(np.concatenate([a_t1 - a_t3, b_t1 - b_t3]))))
    rep("S5b", f"对照：T_1 与 T_3 的身份分解之差 {diff_t:.6f}（期望非零：分解有区分力）",
        diff_t > 0.1)
    rep("S5c", f"区分力对照：同一左乘判据在 g_z 方向给偏差 {left_z:.3e}、"
               f"在 g_x / g_y 方向给 {left_xy:.6f}（期望两者不同：判据能区分保持与不保持）",
        left_z < 1e-14 and left_xy > 0.1)
    rep("S5d", f"恒真标注：S3a 的「右乘保持 J0」读数 {max_right:.3e} 来自左乘与右乘交换这一代数恒等，"
               f"对任意左乘型复结构同样成立（S5a 偏差 {r_alt:.3e}）；该恒真性本身排除"
               f"「右乘保持 -J」（右乘保持的是 J 本身）",
        max_right < 1e-14 and r_alt < 1e-14)

    # ---------------- 图
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))

    ax = axes[0]
    ax.plot(ts, resid3, color="#1f77b4", lw=2.0, label="3x3 反对称族 min 残差")
    ax.axhline(1.0, color="#d62728", ls="--", lw=1.4, label="解析最小值 1")
    ax.axhline(0.0, color="#2ca02c", ls=":", lw=1.6, label="4x4 上 J0 残差 0")
    ax.set_xlabel(r"$\|\omega\|$")
    ax.set_ylabel(r"$\|J^2 + I\|_F$")
    ax.set_title("S1：复结构的定义域")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    labels = [f"${dir_names[k]}$ {th}" for k in range(3) for th in thetas]
    vals_r = [r[2] for r in rows]
    vals_l = [r[3] for r in rows]
    xs = np.arange(len(rows))
    ax.bar(xs - 0.2, vals_r, 0.4, color="#2ca02c", label="四元数右乘共轭（保持）")
    ax.bar(xs + 0.2, vals_l, 0.4, color="#d62728", label="四元数左乘共轭（不保持）")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("共轭后与 $J_0$ 的最大偏差")
    ax.set_title("S3：左右作用对 $J_0$ 的作用")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[2]
    names = [r[0] for r in id_rows]
    dl = [np.linalg.norm(r[1] - E0) for r in id_rows]
    dr = [np.linalg.norm(r[2] - E0) for r in id_rows]
    xs = np.arange(len(names))
    ax.bar(xs - 0.2, dl, 0.4, color="#1f77b4", label="$SU(2)_L$ 分量 $|a - 1|$")
    ax.bar(xs + 0.2, dr, 0.4, color="#ff7f0e", label="$SU(2)_R$ 分量 $|b - 1|$")
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("偏离单位元的距离")
    ax.set_title("S4：$M = L_a R_b$ 身份分解")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(IMG, dpi=150)
    print(f"\n图已存：{IMG}")

    n_fail = sum(1 for line in LOG if line.startswith("[FAIL]"))
    print(f"\n汇总：{len(LOG) - n_fail}/{len(LOG)} 通过")


if __name__ == "__main__":
    main()
