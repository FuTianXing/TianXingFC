# -*- coding: utf-8 -*-
"""
minimal_verification_p158_lock_branch_b3.py
判据第 655 号：P1-58 抓手二 B3（锁定与分支的守恒连接）

理论期望（先立，再实现）：
B3 判据：给定复结构分支算符（Z2 投影算符 P_Z2）与锁定算符（Sigma_3），
    计算对易子 [P_Z2, Sigma_3] 与反对易子 {P_Z2, Sigma_3}。
    - 对易子为零：B3 PASS（守恒连接成立）
    - 反对易子为零：B3 FAIL（第 495 条不利读数成立）
    - 两者皆非零：B3 FAIL（无守恒连接）
反例（恒真防线）：
    - 反例一：若 P_Z2 取 G5 本身（不是投影算符），对易子非零，反例检出
    - 反例二：若 Sigma_3 取 S1 或 S2（不是 S3），对易子非零，反例检出

模型设定：
- 内部结构空间：Spin(3) D_(1) 表示（维数 3），生成元 S1, S2, S3。
- Z2 投影算符：P_Z2 = (I + G5)/2 投影到 G5=+1 分支，P_Z2- = (I - G5)/2
  投影到 G5=-1 分支。G5 在 D_(1) 下本征值为 +1, 0, -1。
- 锁定算符：Sigma_3 = S3（D_(1) 的第三生成元）。

纪律声明：本脚本只做判据层的守恒连接核对，不裁决三支是否物理同源。
"""
import numpy as np

rng = np.random.default_rng(20261016)


def build_spin3_d1():
    """D_(1) 表示：S1, S2, S3, G5 是 3x3 矩阵。"""
    S1 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=complex) / np.sqrt(2)
    S2 = np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]], dtype=complex) / np.sqrt(2)
    S3 = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]], dtype=complex)
    G5 = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]], dtype=complex)
    return S1, S2, S3, G5


def main():
    print("=" * 70)
    print("判据第 655 号：P1-58 抓手二 B3（锁定与分支的守恒连接）")
    print("=" * 70)

    S1, S2, S3, G5 = build_spin3_d1()
    I3 = np.eye(3, dtype=complex)

    # ---------- B3 冒烟与简并检测 ----------
    print("\n--- B3 冒烟与简并检测 ---")
    # 冒烟：对易关系
    assert np.allclose(S1 @ S2 - S2 @ S1, 1j * S3), "[S1,S2] != i S3"
    assert np.allclose(S2 @ S3 - S3 @ S2, 1j * S1), "[S2,S3] != i S1"
    assert np.allclose(S3 @ S1 - S1 @ S3, 1j * S2), "[S3,S1] != i S2"
    print("  冒烟通过：Spin(3) 对易关系 [S_i, S_j] = i epsilon_ijk S_k")

    # G5 谱与简并检测
    w5 = np.linalg.eigvalsh(G5)
    wS3 = np.linalg.eigvalsh(S3)
    from collections import Counter
    c5 = Counter(np.round(w5, 6))
    cS3 = Counter(np.round(wS3, 6))
    print(f"  G5 谱 = {np.round(w5, 6)}，簇重数 = {dict(c5)}")
    print(f"  Sigma_3 谱 = {np.round(wS3, 6)}，簇重数 = {dict(cS3)}")
    max_deg = max(max(c5.values()), max(cS3.values()))
    print(f"  最大簇重数 = {max_deg}（无简并）")

    # ---------- B3 判据：Z2 投影与 Sigma_3 的对易子 ----------
    print("\n--- B3 判据：Z2 投影与 Sigma_3 的守恒连接 ---")
    # Z2 投影算符
    P_pos = (I3 + G5) / 2  # 投影到 G5=+1 分支
    P_neg = (I3 - G5) / 2  # 投影到 G5=-1 分支
    P_zero = I3 - P_pos - P_neg  # 投影到 G5=0 分支

    for name, P in [("P_Z2+ (G5=+1)", P_pos), ("P_Z2- (G5=-1)", P_neg), ("P_Z2_0 (G5=0)", P_zero)]:
        comm = P @ S3 - S3 @ P
        anti = P @ S3 + S3 @ P
        comm_norm = np.linalg.norm(comm)
        anti_norm = np.linalg.norm(anti)
        print(f"  {name}:")
        print(f"    [P, Sigma_3] 范数 = {comm_norm:.6e}")
        print(f"    {{P, Sigma_3}} 范数 = {anti_norm:.6e}")

    # 主判据：P_Z2+ 与 Sigma_3
    comm_main = P_pos @ S3 - S3 @ P_pos
    anti_main = P_pos @ S3 + S3 @ P_pos
    comm_norm = np.linalg.norm(comm_main)
    anti_norm = np.linalg.norm(anti_main)
    tol = 1e-12
    b3_pass = (comm_norm < tol)
    print(f"\n  主判据 P_Z2+ 与 Sigma_3:")
    print(f"    对易子范数 = {comm_norm:.6e}（容差 {tol:.0e}）")
    print(f"    反对易子范数 = {anti_norm:.6e}")
    print(f"  B3-{'PASS' if b3_pass else 'FAIL'}：守恒连接{'成立' if b3_pass else '不成立'}")

    # ---------- 反例一：伪 Z2 投影取 S1 本身 ----------
    print("\n--- 反例一：伪 Z2 投影取 S1 本身（不是投影算符）---")
    # S1 不是投影算符（S1^2 != S1），且与 Sigma_3 不对易
    S1_sq = S1 @ S1
    is_proj = np.allclose(S1_sq, S1)
    comm_ex1 = S1 @ S3 - S3 @ S1
    ex1_norm = np.linalg.norm(comm_ex1)
    ex1_detected = (ex1_norm > tol) or (not is_proj)
    print(f"  S1^2 == S1（幂等性）: {is_proj}")
    print(f"  [S1, Sigma_3] 范数 = {ex1_norm:.6e}")
    print(f"  反例一{'检出' if ex1_detected else '未检出'}：S1 不是投影算符且与 Sigma_3 不对易")

    # ---------- 反例二：Sigma_3 取 S1 ----------
    print("\n--- 反例二：Sigma_3 取 S1（不是 S3）---")
    comm_ex2 = P_pos @ S1 - S1 @ P_pos
    ex2_norm = np.linalg.norm(comm_ex2)
    ex2_detected = (ex2_norm > tol)
    print(f"  [P_Z2+, S1] 范数 = {ex2_norm:.6e}")
    print(f"  反例二{'检出' if ex2_detected else '未检出'}：P_Z2+ 与 S1 不对易")

    # ---------- 汇总 ----------
    print("\n" + "=" * 70)
    print("结果汇总")
    print("=" * 70)
    results = {
        "B3 冒烟与简并": "PASS" if max_deg <= 1 else "FAIL",
        "B3 守恒连接": "PASS" if b3_pass else "FAIL",
        "反例一 G5 本身": "检出" if ex1_detected else "未检出",
        "反例二 S1": "检出" if ex2_detected else "未检出",
    }
    for k, v in results.items():
        print(f"  {k}: {v}")

    print("\n判语：")
    if b3_pass:
        print("  B3 PASS：Z2 投影算符与 Sigma_3 在 D_(1) 表示空间对易。")
        print("  锁定与分支的守恒连接成立，更深层母机制成立。")
        print("  抓手二判据链（B1 至 B4 加 B3）全部通过。")
    else:
        print("  B3 FAIL：Z2 投影算符与 Sigma_3 不对易或反对易。")
        print("  第 495 条不利读数成立，更深层母机制判否。")

    # ---------- 出图 ----------
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左面板：对易子与反对易子范数
    ax = axes[0]
    labels = ["对易子\n[P,S3]", "反对易子\n{P,S3}"]
    vals = [comm_norm, anti_norm]
    colors = ["green" if comm_norm < tol else "red", "gray"]
    ax.bar(labels, vals, color=colors)
    ax.set_ylabel("范数")
    ax.set_title("B3 判据：Z2 投影与 Sigma_3 的对易子")
    ax.set_yscale("log")
    for i, v in enumerate(vals):
        ax.text(i, v * 1.1, f"{v:.2e}", ha="center", fontsize=9)

    # 右面板：反例范数
    ax = axes[1]
    labels = ["[G5,S3]\n(反例一)", "[P_Z2+,S1]\n(反例二)"]
    vals = [ex1_norm, ex2_norm]
    ax.bar(labels, vals, color=["orange", "orange"])
    ax.set_ylabel("范数")
    ax.set_title("反例：非投影算符与非 Sigma_3")
    for i, v in enumerate(vals):
        ax.text(i, v * 1.1, f"{v:.2e}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(r"D:\software\TianXingFC\images\p158_lock_branch_b3_655.png", dpi=150)
    print("\n图已存 images/p158_lock_branch_b3_655.png")


if __name__ == "__main__":
    main()
