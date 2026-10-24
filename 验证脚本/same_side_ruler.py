# -*- coding: utf-8 -*-
r"""判据第 674 号：同侧判据（侧标签与分支行为的二维分类）。

背景：
  - 第 671 号判定 P1-58 支路一的真实缺口是「同侧判据」：$[S_z\otimes I_4,\ I_2\otimes\Gamma_5]=0$
    只在「不混分支」的意义上成立，无法分辨算符是否加载在模态单支上；第 663 号的占比模板
    迁移到时空侧后失真（恒等算符也判交换）。
  - 本轮把「同侧」拆成两个独立坐标：侧标签 $s$（部分迹重构残差）与分支行为 $b$（该侧
    二分支投影下的块分解）。核心断言是二维不可约化：同一算符在错误的尺下读数会翻转。

口径澄清（本轮顺带修正第 671 号的失真归因）：
  - 模态侧的二分支来自复结构 $J_m = \mathrm{i}\sigma_z$（$J_m^2 = -I$），投影
    $\Pi_\pm = (I \mp \mathrm{i}J_m)/2$，本征值 $\pm\mathrm{i}$。
  - 时空侧的二分支来自手征算符 $\Gamma_5$（$\Gamma_5^2 = I$），投影 $P_\pm = (I \pm \Gamma_5)/2$，
    本征值 $\pm 1$。
  - 两类投影都合法（厄米、幂等、互补），块分解判据对两者通用；第 663 号模板内部假定
    $J^2 = -I$（以 $h - \mathrm{i}Jh$ 构造本征矢），套到 $\Gamma_5$ 上时该构造不给出本征矢，
    失真的主因是两类分支混用，复维不匹配是次要因素。

判据：冒烟 M1 至 M3、侧标签 S1 至 S5、分支行为 B1 至 B6，另附两则跨侧套用反例。
可逆性等级：判据稳健（解析构造与范数读数）。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

I2 = np.eye(2, dtype=complex)
I4 = np.eye(4, dtype=complex)
I8 = np.eye(8, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
SZH = SZ / 2.0

# 时空侧：第 667 号构造（4 分量 Cl(3) 表示）
G0 = np.kron(SZ, I2)
G1 = np.kron(SX, I2)
G2 = np.kron(I2, SY)
G5 = 1j * G0 @ G1 @ G2

# 两侧的二分支生成元：复结构型与手征型
J_M = 1j * SZ     # 模态侧复结构，J^2 = -I
C_S = G5          # 时空侧手征算符，C^2 = I

LOG = []


def rep(tag, cond, txt):
    LOG.append(f"[{'PASS' if cond else 'FAIL'}] {tag}: {txt}")
    print(LOG[-1])


def proj_complex(J):
    """复结构型二分支投影（J^2 = -I）：本征值 +i 与 -i。"""
    n = J.shape[0]
    return 0.5 * (np.eye(n) - 1j * J), 0.5 * (np.eye(n) + 1j * J)


def proj_chiral(C):
    """手征型二分支投影（C^2 = I）：本征值 +1 与 -1。"""
    n = C.shape[0]
    return 0.5 * (np.eye(n) + C), 0.5 * (np.eye(n) - C)


def branch_from_projectors(M, Pp, Pm):
    """块分解：返回 (保持度, 交换度)；M 为零时返回 None。"""
    if np.linalg.norm(M) < 1e-14:
        return None
    keep = np.linalg.norm(Pp @ M @ Pp + Pm @ M @ Pm)
    swap = np.linalg.norm(Pp @ M @ Pm + Pm @ M @ Pp)
    tot = np.sqrt(keep ** 2 + swap ** 2)
    return float(keep / tot), float(swap / tot)


def ptr_s(M):
    """对时空因子求部分迹（8x8 -> 2x2）。"""
    return np.einsum("aibi->ab", M.reshape(2, 4, 2, 4))


def ptr_m(M):
    """对模态因子求部分迹（8x8 -> 4x4）。"""
    return np.einsum("aiaj->ij", M.reshape(2, 4, 2, 4))


def recon_m(M):
    return np.kron(ptr_s(M) / 4.0, I4)


def recon_s(M):
    return np.kron(I2, ptr_m(M) / 2.0)


def side_readout(M):
    rm = float(np.linalg.norm(M - recon_m(M)))
    rs = float(np.linalg.norm(M - recon_s(M)))
    return rm, rs


def side_label(rm, rs, tol=1e-12):
    if rm < tol and rs < tol:
        return "退化"
    if rm < tol:
        return "模态"
    if rs < tol:
        return "时空"
    return "双侧"


def main():
    print("=== 判据第 674 号：同侧判据（侧标签与分支行为的二维分类） ===")
    rng = np.random.default_rng(20261017)

    # ---------------- M1：部分迹与重构恒等式
    w1 = 0.0
    for _ in range(50):
        A = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        B = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        M = np.kron(A, B)
        w1 = max(w1, float(np.linalg.norm(ptr_s(M) - A * np.trace(B))))
        w1 = max(w1, float(np.linalg.norm(ptr_m(M) - np.trace(A) * B)))
        w1 = max(w1, float(np.linalg.norm(recon_m(np.kron(A, I4)) - np.kron(A, I4))))
        w1 = max(w1, float(np.linalg.norm(recon_s(np.kron(I2, B)) - np.kron(I2, B))))
    rep("M1", w1 < 1e-12, f"部分迹与重构恒等式最大偏差 {w1:.2e}（期望 < 1e-12）")

    # ---------------- M2：两类投影的合法性
    Pp_m, Pm_m = proj_complex(J_M)
    Pp_s, Pm_s = proj_chiral(C_S)
    r = []
    for Pp, Pm in ((Pp_m, Pm_m), (Pp_s, Pm_s)):
        r.append(float(np.linalg.norm(Pp @ Pp - Pp)))
        r.append(float(np.linalg.norm(Pm @ Pm - Pm)))
        r.append(float(np.linalg.norm(Pp + Pm - np.eye(Pp.shape[0]))))
        r.append(float(np.linalg.norm(Pp - Pp.conj().T)))
    rk_m = (int(np.linalg.matrix_rank(Pp_m)), int(np.linalg.matrix_rank(Pm_m)))
    rk_s = (int(np.linalg.matrix_rank(Pp_s)), int(np.linalg.matrix_rank(Pm_s)))
    rep("M2", max(r) < 1e-13 and rk_m == (1, 1) and rk_s == (2, 2),
        f"两类投影均厄米幂等互补（最大偏差 {max(r):.1e}）；"
        f"复结构型秩 {rk_m}、手征型秩 {rk_s}")

    # ---------------- M3：块分解范数守恒
    w3 = 0.0
    for _ in range(50):
        A = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        B = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        for M, (Pp, Pm) in ((A, (Pp_m, Pm_m)), (B, (Pp_s, Pm_s))):
            keep = np.linalg.norm(Pp @ M @ Pp + Pm @ M @ Pm)
            swap = np.linalg.norm(Pp @ M @ Pm + Pm @ M @ Pp)
            w3 = max(w3, abs(float(np.sqrt(keep ** 2 + swap ** 2)) - float(np.linalg.norm(M))))
    rep("M3", w3 < 1e-12, f"块分解范数守恒最大偏差 {w3:.2e}（期望 < 1e-12）")

    # ---------------- S：侧标签
    ops = [
        ("S1", r"$S_z\otimes I_4$", np.kron(SZH, I4), "模态"),
        ("S2", r"$I_2\otimes\Gamma_5$", np.kron(I2, G5), "时空"),
        ("S3", r"$I_2\otimes\Gamma_1$", np.kron(I2, G1), "时空"),
        ("S4", r"$\sigma_z\otimes\Gamma_5$", np.kron(SZ, G5), "双侧"),
        ("S5", r"$I_8$", I8, "退化"),
    ]
    side_rows = []
    for tag, name, M, exp in ops:
        rm, rs = side_readout(M)
        lab = side_label(rm, rs)
        side_rows.append((name, rm, rs, lab))
        rep(tag, lab == exp,
            f"{name}：模态侧残差 {rm:.3e}、时空侧残差 {rs:.3e} → {lab}（期望 {exp}）")

    # ---------------- B：分支行为（正确尺：约化到该侧因子，用该侧分支）
    # B3 刻意避开 $\Gamma_5$ 自身：投影 $P_\pm$ 由 $\Gamma_5$ 定义，用 $\Gamma_5$ 测自己的
    # 分支行为是恒真项（EP5）。改取时空侧与 $\Gamma_5$ 对易的非平凡算符 $\Gamma_0\Gamma_1$。
    cases = [
        ("B1", r"$S_z$", SZH, (Pp_m, Pm_m), "保持"),
        ("B2", r"$S_x$", SX / 2.0, (Pp_m, Pm_m), "交换"),
        ("B3", r"$\Gamma_0\Gamma_1$", G0 @ G1, (Pp_s, Pm_s), "保持"),
        ("B4", r"$\Gamma_1$", G1, (Pp_s, Pm_s), "交换"),
        ("B5", r"$\Gamma_2$", G2, (Pp_s, Pm_s), "保持"),
        ("B6", r"$(\sigma_x+\sigma_z)/2$", (SX + SZ) / 2.0, (Pp_m, Pm_m), "混合"),
    ]
    b_rows = []
    for tag, name, A, (Pp, Pm), exp in cases:
        k, s = branch_from_projectors(A, Pp, Pm)
        b_rows.append((name, k, s))
        if exp == "保持":
            ok = k > 1.0 - 1e-12
        elif exp == "交换":
            ok = s > 1.0 - 1e-12
        else:
            ok = abs(k - s) < 1e-12 and abs(k - 1.0 / np.sqrt(2.0)) < 1e-12
        rep(tag, ok, f"{name}：保持度 {k:.6f}、交换度 {s:.6f}（期望 {exp}）")

    # ---------------- R：跨侧套用（区分力）
    M = np.kron(SX / 2.0, I4)
    P8p, P8m = proj_chiral(np.kron(I2, G5))
    k_bad, s_bad = branch_from_projectors(M, P8p, P8m)
    k_ok, s_ok = branch_from_projectors(SX / 2.0, Pp_m, Pm_m)
    rep("R1", k_bad > 1.0 - 1e-12 and s_ok > 1.0 - 1e-12,
        f"{r'$S_x\otimes I_4$'}：跨侧套用（8 维 + 手征型）读保持度 {k_bad:.6f}，"
        f"正确尺（约化 + 复结构型）读交换度 {s_ok:.6f}，读数翻转")

    M = np.kron(I2, G1)
    Jm8 = np.kron(J_M, I4)
    P8p2, P8m2 = proj_complex(Jm8)
    k_bad2, s_bad2 = branch_from_projectors(M, P8p2, P8m2)
    k_ok2, s_ok2 = branch_from_projectors(G1, Pp_s, Pm_s)
    rep("R2", k_bad2 > 1.0 - 1e-12 and s_ok2 > 1.0 - 1e-12,
        f"{r'$I_2\otimes\Gamma_1$'}：跨侧套用（8 维 + 复结构型）读保持度 {k_bad2:.6f}，"
        f"正确尺（约化 + 手征型）读交换度 {s_ok2:.6f}，读数翻转")

    # ---------------- C：对易与侧别的关系（修正第 671 号的归因）
    # 671 号以 $[S_z\otimes I_4,\ I_2\otimes\Gamma_5]=0$ 作为「同侧」的证据。本轮扫描表明：
    # 任何纯模态算符与任何纯时空算符都对易，对易是同侧的充分条件而非必要条件，
    # 且它恰恰是「异侧且各居纯因子」的伴随特征，不能用来分辨同侧。
    pairs = [
        (r"$S_z\otimes I_4$ 与 $I_2\otimes\Gamma_5$", SZH, G5),
        (r"$S_x\otimes I_4$ 与 $I_2\otimes\Gamma_1$", SX / 2.0, G1),
        (r"$S_y\otimes I_4$ 与 $I_2\otimes\Gamma_2$", SY / 2.0, G2),
        (r"$(\sigma_x+\sigma_z)/2\otimes I_4$ 与 $I_2\otimes\Gamma_1$", (SX + SZ) / 2.0, G1),
    ]
    com = []
    for name, A, B in pairs:
        c2 = float(np.linalg.norm(np.kron(A, I4) @ np.kron(I2, B)
                                  - np.kron(I2, B) @ np.kron(A, I4)))
        com.append(c2)
    rep("C1", max(com) < 1e-13,
        f"四组纯模态 × 纯时空算符对易子最大范数 {max(com):.2e}（期望 0）："
        f"异侧且各居纯因子时对易恒成立")

    # C2：同侧算符可对易也可不对易，故「对易」不蕴含侧别
    same_side = float(np.linalg.norm(SX @ SZ - SZ @ SX))
    same_comm = float(np.linalg.norm(SZ @ (SZ @ SZ) - (SZ @ SZ) @ SZ))
    rep("C2", same_side > 1.0 and same_comm < 1e-13,
        f"同侧算符：{r'$S_x$'} 与 {r'$S_z$'} 对易子范数 {same_side:.6f}（不对易）、"
        f"{r'$S_z$'} 与 {r'$S_z^2$'} 对易子范数 {same_comm:.2e}（对易）："
        f"同侧两种情形都出现，对易不蕴含侧别")
    print("  逻辑定位：异侧且各居纯因子 ⇒ 对易（C1）；故不对易 ⇒ 同侧，"
          "但逆命题不成立（对易无法推出同侧）。第 671 号把对易读作同侧的证据，方向反了。")

    # ---------------- T：类型与复维的分离（修正第 671 号的失真归因）
    # 671 号把模板失真归因于「$\Gamma_5$ 本征空间各复维 2 而模板要求各复维 1」。
    # 分离实验：先固定类型换复维，再固定复维换类型，看哪一项真正决定成败。
    # T1：手征型尺、复维 2（4 分量时空侧）
    k4, s4 = branch_from_projectors(G0 @ G1, Pp_s, Pm_s)
    # T2：手征型尺、复维 1（2 分量玩具 C = sigma_z）
    P2p, P2m = proj_chiral(SZ)
    k2, s2 = branch_from_projectors(SX, P2p, P2m)
    rep("T1", k4 > 1.0 - 1e-12 and s2 > 1.0 - 1e-12,
        f"同一手征型尺，复维 2（保持度 {k4:.6f}）与复维 1（交换度 {s2:.6f}）都给出正确读数："
        f"复维不是模板失真的主因")

    # T3：663 号的本征矢构造 $v = h - \mathrm{i}Jh$ 在两类型下的成败扫描。
    # 该式依赖 $J^2 = -I$；$C^2 = +I$ 时它不给出本征矢（除非 $h$ 恰为本征矢）。
    rng2 = np.random.default_rng(674)
    fail_J, fail_C = 0, 0
    n_s = 400
    for _ in range(n_s):
        h2 = rng2.normal(size=2) + 1j * rng2.normal(size=2)
        h2 /= np.linalg.norm(h2)
        v = h2 - 1j * (J_M @ h2)
        fail_J += int(np.linalg.norm(J_M @ v - 1j * v) > 1e-10)
        h4 = rng2.normal(size=4) + 1j * rng2.normal(size=4)
        h4 /= np.linalg.norm(h4)
        vc = h4 - 1j * (G5 @ h4)
        ok = min(np.linalg.norm(G5 @ vc - vc), np.linalg.norm(G5 @ vc + vc))
        fail_C += int(ok > 1e-10)
    rep("T2", fail_J == 0 and fail_C > 0.95 * n_s,
        f"$h-\\mathrm{{i}}Jh$ 构造在 {n_s} 个随机 $h$ 上的失败数："
        f"复结构型 {fail_J} 个（恒成功），手征型 {fail_C} 个（几乎恒失败）")

    # ---------------- 图
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    ax = axes[0]
    names = [row[0] for row in side_rows]
    rm_v = [max(row[1], 1e-18) for row in side_rows]
    rs_v = [max(row[2], 1e-18) for row in side_rows]
    x = np.arange(len(names))
    ax.bar(x - 0.2, rm_v, 0.4, label="模态侧残差", color="#2a6fdb")
    ax.bar(x + 0.2, rs_v, 0.4, label="时空侧残差", color="#d1603d")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.axhline(1e-12, color="gray", ls="--", lw=1)
    ax.set_ylabel("重构残差（Frobenius，对数）")
    ax.set_title("(a) 侧标签：部分迹重构残差")
    ax.legend(fontsize=8)

    ax = axes[1]
    names = [row[0] for row in b_rows]
    kk = [row[1] for row in b_rows]
    ss = [row[2] for row in b_rows]
    x = np.arange(len(names))
    ax.bar(x, kk, 0.6, label="保持度", color="#3a9d5d")
    ax.bar(x, ss, 0.6, bottom=kk, label="交换度", color="#e0a03a")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("块分解占比")
    ax.set_title("(b) 分支行为：正确尺（约化后）")
    ax.legend(fontsize=8)

    ax = axes[2]
    labels = [r"$S_x\otimes I_4$", r"$I_2\otimes\Gamma_1$"]
    bad = [k_bad, k_bad2]
    good = [s_ok, s_ok2]
    x = np.arange(2)
    ax.bar(x - 0.18, bad, 0.36, label="跨侧套用：读作保持度", color="#b03a3a")
    ax.bar(x + 0.18, good, 0.36, label="正确尺：读作交换度", color="#2a6fdb")
    for i in range(2):
        ax.text(x[i] - 0.18, 0.5, "保持\n1.000", ha="center", va="center",
                color="white", fontsize=9)
        ax.text(x[i] + 0.18, 0.5, "交换\n1.000", ha="center", va="center",
                color="white", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("占比")
    ax.set_title("(c) 反例：同一算符，两把尺给出相反标签")
    ax.legend(fontsize=8)

    fig.tight_layout()
    out = "D:/software/TianXingFC/images/fig_same_side_ruler.png"
    fig.savefig(out, dpi=160)
    print(f"\n图已保存：{out}")

    n_pass = sum(1 for line in LOG if line.startswith("[PASS]"))
    print(f"\n判据通过 {n_pass}/{len(LOG)}")


if __name__ == "__main__":
    main()
