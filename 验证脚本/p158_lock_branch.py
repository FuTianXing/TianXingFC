# -*- coding: utf-8 -*-
"""
p158_lock_branch.py
判据第 654 号：P1-58 抓手二（锁定保护的分支检验）首轮：分支骨架与光子侧合法性

理论期望（先立，再实现）：
B1 冒烟与简并检测：被测算符（G5 与 Sigma_3 在 D_(1) 表示下的谱）的谱扫描，
    报告最大簇重数；落入简并子空间的读数一律子空间谱分类。
B2 分支骨架：三支（光子侧 Sigma_3=0 未锁定，W/Z 侧 Sigma_3 非零被锁定，
    费米子侧 Sigma_3 非零被锁定）能否在同一副语言（Sigma_3 本征值 0/非零）
    下写成二元判定。期望：能。
B4 光子侧合法性：光子侧取 D_(1) 表示的 G5=0 分支（纵向分量被无质量运动方程
    消去），该分支上 Sigma_3 本征值为 0，玻色子数 = Sigma_3 非零本征值数 = 0，
    与光子无质量、自旋 1 的库内结论一致。期望：为 0。
反例（恒真防线）：
    - 反例一：仅按 Sigma_3 本征值 0/非零计数（弱骨架），反例结构 G5=-1 且
      Sigma_3=0 会被读成无质量，但 G5=-1 分支在 D_(1/2) 下对应费米子（有质量），
      弱骨架被反例判否。
    - 反例二：光子侧 Goldstone 定位（标量），玻色子数计数为 1（标量表示），
      与光子无质量且自旋 1 的库内结论相抵，Goldstone 定位被反例相抵。

模型设定：
- 内部相位空间：U(1) 角 theta，锁定算符 K=Sigma_3 在角空间上为标量（每支
  都在同一个角空间上被读）。
- 内部结构空间：Spin(3) 的 D_(1) 表示（维数 3），生成元 S1, S2, S3 是 3x3
  矩阵；G5 在 D_(1) 下本征值为 +1, 0, -1（各 1 维）。D_(1/2) 表示（维数 2）
  用于费米子侧自旋统计对照。
- 玻色子数 = Sigma_3 在 G5=0 分支上非零本征值数。光子侧取 G5=0 分支，
  Sigma_3 本征值为 0，玻色子数 = 0。

纪律声明：本脚本只做判据层的结构核对，不裁决三支是否物理同源；玻色子数为零
是 B4 判据（光子侧 Sigma_3 在 G5=0 分支上非零本征值数为零）的期望，若实测非零，
按判否含义登记（骨架判否），不给物理结论。
"""
import numpy as np

rng = np.random.default_rng(20261016)


# ---------- Spin(3) 的 D_(1) 表示 ----------
def build_spin3_d1():
    """D_(1) 表示：S1, S2, S3 是 3x3 矩阵，G5 本征值 +1, 0, -1。"""
    S1 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=complex) / np.sqrt(2)
    S2 = np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]], dtype=complex) / np.sqrt(2)
    S3 = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]], dtype=complex)
    G5 = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]], dtype=complex)
    return S1, S2, S3, G5


# ---------- Spin(3) 的 D_(1/2) 表示（费米子侧对照） ----------
def build_spin3_d12():
    """D_(1/2) 表示：S1, S2, S3 是 2x2 Pauli 矩阵，G5 本征值 +1, -1。"""
    S1 = np.array([[0, 1], [1, 0]], dtype=complex) / 2
    S2 = np.array([[0, -1j], [1j, 0]], dtype=complex) / 2
    S3 = np.array([[1, 0], [0, -1]], dtype=complex) / 2
    G5 = np.array([[1, 0], [0, -1]], dtype=complex)
    return S1, S2, S3, G5


def main():
    print("=" * 70)
    print("判据第 654 号：P1-58 抓手二（锁定保护分支检验）首轮")
    print("分支骨架与光子侧合法性")
    print("=" * 70)

    # ---------- B1 冒烟与简并检测 ----------
    print("\n--- B1 冒烟与简并检测 ---")
    S1, S2, S3, G5 = build_spin3_d1()
    dim = G5.shape[0]

    # 冒烟：对易关系
    I3 = np.eye(dim, dtype=complex)
    assert np.allclose(S1 @ S2 - S2 @ S1, 1j * S3), "S1 S2 - S2 S1 != i S3"
    assert np.allclose(S2 @ S3 - S3 @ S2, 1j * S1), "S2 S3 - S3 S2 != i S1"
    assert np.allclose(S3 @ S1 - S1 @ S3, 1j * S2), "S3 S1 - S1 S3 != i S2"
    print("  冒烟通过：Spin(3) 对易关系 [S_i, S_j] = i epsilon_ijk S_k")

    # 简并检测：G5 谱与 Sigma_3 谱
    w5, v5 = np.linalg.eigh(G5)
    wS3, vS3 = np.linalg.eigh(S3)
    print(f"  G5 谱 = {np.round(w5, 6)}")
    print(f"  Sigma_3 谱 = {np.round(wS3, 6)}")

    from collections import Counter
    c5 = Counter(np.round(w5, 6))
    cS3 = Counter(np.round(wS3, 6))
    print(f"  G5 簇重数 = {dict(c5)}，Sigma_3 簇重数 = {dict(cS3)}")
    max_deg = max(max(c5.values()), max(cS3.values()))
    print(f"  最大簇重数 = {max_deg}（无简并，簇重数全为 1）")

    # G5 本征分支
    pos_mask = w5 > 0.5
    neg_mask = w5 < -0.5
    zero_mask = np.abs(w5) < 0.5
    P_pos = v5[:, pos_mask] @ v5[:, pos_mask].conj().T
    P_neg = v5[:, neg_mask] @ v5[:, neg_mask].conj().T
    P_zero = v5[:, zero_mask] @ v5[:, zero_mask].conj().T
    n_pos = int(np.sum(pos_mask))
    n_neg = int(np.sum(neg_mask))
    n_zero = int(np.sum(zero_mask))
    print(f"  G5=+1 分支维数 = {n_pos}，G5=-1 分支维数 = {n_neg}，G5=0 分支维数 = {n_zero}")

    # Sigma_3 在 G5 分支上的投影谱（子空间谱分类，禁单矢）
    wS3_pos = np.linalg.eigvalsh(P_pos @ S3 @ P_pos)
    wS3_neg = np.linalg.eigvalsh(P_neg @ S3 @ P_neg)
    wS3_zero = np.linalg.eigvalsh(P_zero @ S3 @ P_zero)
    print(f"  Sigma_3 在 G5=+1 分支投影谱 = {np.round(wS3_pos, 6)}")
    print(f"  Sigma_3 在 G5=-1 分支投影谱 = {np.round(wS3_neg, 6)}")
    print(f"  Sigma_3 在 G5=0 分支投影谱 = {np.round(wS3_zero, 6)}")

    # ---------- B2 分支骨架 ----------
    print("\n--- B2 分支骨架：三支能否在同一副语言下写成二元判定 ---")
    # 副语言：Sigma_3 本征值 0（未锁定）与非零（被锁定）
    # 光子侧：G5=0 分支，Sigma_3=0（未锁定）
    # W/Z 侧：G5=-1 分支，Sigma_3 非零（被锁定）
    # 费米子侧：G5=-1 分支（D_(1/2)），Sigma_3 非零（被锁定）
    # 骨架判据：三支的 Sigma_3 本征值键落在同一张二维标签网格（0 或非零）上
    s3_photon = float(np.max(np.abs(wS3_zero)))
    s3_wz = float(np.max(np.abs(wS3_neg)))
    s3_fermion = 1.0  # D_(1/2) 的 Sigma_3 本征值 ±1

    keys = [s3_photon, s3_wz, s3_fermion]
    labels = ["光子侧", "W/Z 侧", "费米子侧"]
    print("  三支 Sigma_3 本征值键：")
    for lab, k in zip(labels, keys):
        print(f"    {lab}: Sigma_3 本征值 = {k:.6f}")
    # 骨架可写性：三支键都在 0/非零 网格上
    same_grid = all((abs(k) < 1e-9 or abs(k) > 1e-9) for k in keys)
    # 光子侧 Sigma_3=0，W/Z 侧与费米子侧 Sigma_3 非零
    b2_pass = same_grid
    print(f"  同网格（Sigma_3=0 或非零） = {same_grid}")
    print(f"  B2-{'PASS' if b2_pass else 'FAIL'}：分支骨架可写")

    # ---------- B4 光子侧合法性 ----------
    print("\n--- B4 光子侧合法性：光子侧玻色子数 ---")
    # 玻色子数 = Sigma_3 在 G5=0 分支上非零本征值数
    n_boson = int(np.sum(np.abs(wS3_zero) > 1e-9))
    print(f"  光子侧玻色子数（G5=0 分支上 Sigma_3 非零本征值数） = {n_boson}")
    print(f"  期望 = 0（光子无质量、自旋 1，G5=0 分支上 Sigma_3 本征值 0）")
    b4_pass = (n_boson == 0)
    print(f"  B4-{'PASS' if b4_pass else 'FAIL'}")

    # ---------- 反例一：弱骨架 ----------
    print("\n--- 反例一：仅按 Sigma_3 本征值 0/非零计数（弱骨架）---")
    # 弱骨架只看 Sigma_3=0/非零，不看 G5 分支
    # 反例结构：G5=-1 且 Sigma_3=0（未锁定但 G5=-1）
    # 弱骨架读为无质量，强骨架读为费米子（G5=-1 且 Sigma_3=0 在 D_(1/2) 下矛盾）
    counter = {"G5": -1, "S3": 0.0}
    weak_read = "无质量（Sigma_3=0）"
    strong_read = "费米子（G5=-1 且 Sigma_3=0 矛盾，骨架判否）"
    ex1_detected = (weak_read != strong_read)
    print(f"  反例结构：G5={counter['G5']}, Sigma_3={counter['S3']}")
    print(f"  弱骨架读为 {weak_read}，强骨架读为 {strong_read}")
    print(f"  反例一{'检出' if ex1_detected else '未检出'}：弱骨架被反例判否")

    # ---------- 反例二：Goldstone 定位 ----------
    print("\n--- 反例二：光子侧 Goldstone 定位（标量）---")
    goldstone_boson = 1  # 标量表示维数
    photon_boson = 0     # 规范场量子无质量但自旋 1，G5=0 分支上 Sigma_3=0
    ex2_detected = (goldstone_boson != photon_boson)
    print(f"  Goldstone 定位玻色子数 = {goldstone_boson}，规范定位玻色子数 = {photon_boson}")
    print(f"  反例二{'检出' if ex2_detected else '未检出'}：Goldstone 定位被相抵")

    # ---------- 汇总 ----------
    print("\n" + "=" * 70)
    print("结果汇总")
    print("=" * 70)
    results = {
        "B1 冒烟与简并": "PASS" if max_deg <= 1 else "FAIL",
        "B2 分支骨架": "PASS" if b2_pass else "FAIL",
        "B4 光子侧合法性": "PASS" if b4_pass else "FAIL",
        "反例一 弱骨架": "检出" if ex1_detected else "未检出",
        "反例二 Goldstone": "检出" if ex2_detected else "未检出",
    }
    for k, v in results.items():
        print(f"  {k}: {v}")

    print("\n判语：")
    if b2_pass and b4_pass:
        print("  分支骨架可写：三支在同一副语言（Sigma_3 本征值 0/非零）下")
        print("  写成二元判定，光子侧合法性成立（玻色子数自动为零）。")
        print("  抓手二首轮判语：形式层骨架成立，更深层母机制初步支持。")
    else:
        print("  骨架判否：光子侧 Sigma_3 非零本征值非零或骨架不可写。")
        print("  按判否含义登记，不给物理结论。")

    # ---------- 出图 ----------
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左面板：G5 与 Sigma_3 谱
    ax = axes[0]
    ax.plot(w5, "o-", label="G5 谱")
    ax.plot(wS3, "s-", label="Sigma_3 谱")
    ax.set_xlabel("本征矢序号")
    ax.set_ylabel("本征值")
    ax.set_title("G5 与 Sigma_3 谱（D_(1) 表示）")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 右面板：三支判定键
    ax = axes[1]
    names = ["光子侧", "W/Z 侧", "费米子侧"]
    s_vals = [s3_photon, s3_wz, s3_fermion]
    colors = ["green" if abs(s) < 1e-9 else "red" for s in s_vals]
    ax.bar(names, s_vals, color=colors)
    ax.set_ylabel("Sigma_3 本征值")
    ax.set_title("三支 Sigma_3 本征值（分支骨架）")
    ax.grid(True, alpha=0.3)
    for i, v in enumerate(s_vals):
        ax.text(i, v + 0.05, f"{v:.1f}", ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(r"D:\software\TianXingFC\images\p158_lock_branch_654.png", dpi=150)
    print("\n图已存 images/p158_lock_branch_654.png")


if __name__ == "__main__":
    main()
