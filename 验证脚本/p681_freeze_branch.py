# -*- coding: utf-8 -*-
r"""判据第 681 号：冻结与分支行为的关系（P1-58 支路一的合成判语轮）。

背景：
  - 671 号给模态锁定候选 S_z = sigma_z/2；674 号造出同侧尺（侧标签 + 分支行为）；
    678 号把「模态旋转模式被凝聚冻结」转成可算构型判据（冻结档响应 1/(1+kappa)），
    并判「档数不在冻结判据射程内」。
  - 三件材料齐备后，支路一的正题可判：W/Z 的模态锁定与复结构 J 是什么关系。

本轮问题：冻结操作是否改变算符在 J_M 分支上的行为？
  若冻结前后分支行为不变（保持/交换类型相同），判「锁定与复结构分支正交」，
  即锁定是独立于分支结构的第二个结构。
  若冻结使分支行为翻转，判「锁定即分支重拨」。

判据设计（三条纪律逐条落实）：
  其一，对象类型前置：被问对象是「冻结操作对分支行为的作用」，
    是两对象之间的关系，形式取「前后块分解读数之比较」，不取对易子。
  其二，恒真检验：须有改掉它就 FAIL 的反例。反例取「能翻转分支的算子」
    （L_j 型左乘，663 号交换类）——若判据对它与对 R(phi) 给同一答案，则判据失效。
  其三，多档扫描：冻结相位取多档、锁定强度取多档，单档结论不成立。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)

J_M = 1j * SZ          # 复结构，J^2 = -I
Pp = 0.5 * (I2 - 1j * J_M)   # H^{1,0} 投影（+i 支）
Pm = 0.5 * (I2 + 1j * J_M)   # H^{0,1} 投影（-i 支）

LOG = []


def rep(tag, cond, txt):
    LOG.append("[%s] %s: %s" % ("PASS" if cond else "FAIL", tag, txt))
    print(LOG[-1])
    return cond


def rotation(phi):
    """模态旋转模式 R(phi) = exp(-i phi sigma_z/2)。"""
    return np.cos(phi / 2.0) * I2 - 1j * np.sin(phi / 2.0) * SZ


def branch_readout(M):
    """块分解：返回 (保持度, 交换度)。零矩阵返回 (0,0)。"""
    if np.linalg.norm(M) < 1e-14:
        return 0.0, 0.0
    keep = np.linalg.norm(Pp @ M @ Pp + Pm @ M @ Pm)
    swap = np.linalg.norm(Pp @ M @ Pm + Pm @ M @ Pp)
    tot = np.sqrt(keep ** 2 + swap ** 2)
    return float(keep / tot), float(swap / tot)


def freeze(phi, kappa_lock, n_fold=4):
    """冻结构型：相位锁到 n_fold 等分格点，残余偏离按 1/(1+kappa) 衰减。"""
    period = 2 * np.pi / n_fold
    phi0 = np.round(phi / period) * period
    return phi0 + (phi - phi0) / (1.0 + kappa_lock)


def left_mult(q):
    """四元数左乘矩阵（1,i,j,k 基）。"""
    w, x, y, z = q
    return np.array([[w, -x, -y, -z],
                     [x,  w, -z,  y],
                     [y,  z,  w, -x],
                     [z, -y,  x,  w]], dtype=complex)


def main():
    print("=== 判据第 681 号：冻结与分支行为的关系 ===")

    # ---------------- M1 冒烟：尺本身良定义 ----------------
    print("\n--- M1 冒烟：分支尺在已知算符上的读数 ---")
    k_ident, s_ident = branch_readout(I2)
    rep("M1a", k_ident > 0.99 and s_ident < 1e-12,
        "恒等算符为纯保持（保持度 %.6f，交换度 %.6f）" % (k_ident, s_ident))
    k_sz, s_sz = branch_readout(SZ)
    rep("M1b", k_sz > 0.99, "S_z 为纯保持（保持度 %.6f）" % k_sz)
    k_sx, s_sx = branch_readout(SX)
    rep("M1c", s_sx > 0.99, "S_x 为纯交换（交换度 %.6f）" % s_sx)

    # ---------------- M2 核心：冻结前后的分支行为 ----------------
    print("\n--- M2 核心判据：冻结是否改变分支行为 ---")
    phis = [0.13, 0.71, 1.27, 2.09]
    kappa = 100.0
    rows = []
    for phi in phis:
        k_before, s_before = branch_readout(rotation(phi))
        k_after, s_after = branch_readout(rotation(freeze(phi, kappa)))
        rows.append((phi, k_before, s_before, k_after, s_after))
        print("    phi=%.2f  冻结前(保持 %.6f, 交换 %.6f)  冻结后(保持 %.6f, 交换 %.6f)"
              % (phi, k_before, s_before, k_after, s_after))
    max_keep_dev = max(abs(r[1] - r[3]) for r in rows)
    max_swap_dev = max(abs(r[2] - r[4]) for r in rows)
    rep("M2a", max_keep_dev < 1e-12,
        "四档相位的保持度在冻结前后相同（最大差 %.2e）" % max_keep_dev)
    rep("M2b", max_swap_dev < 1e-12,
        "四档相位的交换度在冻结前后相同（最大差 %.2e）" % max_swap_dev)
    rep("M2c", all(r[1] > 0.99 for r in rows),
        "冻结前后均为纯保持型（R(phi) 与 J_M 对易，属 663 号保持类）")

    # ---------------- M3 多档扫描：锁定强度 ----------------
    print("\n--- M3 多档扫描：锁定强度 kappa_lock ---")
    kappas = [1.0, 10.0, 100.0, 1000.0]
    phi = 0.71
    devs = []
    for k in kappas:
        kb, sb = branch_readout(rotation(phi))
        ka, sa = branch_readout(rotation(freeze(phi, k)))
        devs.append(abs(kb - ka))
        print("    kappa=%-6.0f  冻结前保持 %.9f  冻结后保持 %.9f  差 %.2e"
              % (k, kb, ka, abs(kb - ka)))
    rep("M3", max(devs) < 1e-12,
        "四档锁定强度下冻结前后保持度差全为零（最大 %.2e）" % max(devs))

    # ---------------- M4 多档扫描：冻结档数 ----------------
    print("\n--- M4 多档扫描：冻结档数 n_fold ---")
    folds = [1, 2, 4, 8]
    devs_f = []
    for nf in folds:
        kb, _ = branch_readout(rotation(phi))
        ka, _ = branch_readout(rotation(freeze(phi, kappa, nf)))
        devs_f.append(abs(kb - ka))
        print("    n_fold=%d  冻结前保持 %.9f  冻结后保持 %.9f  差 %.2e" % (nf, kb, ka, abs(kb - ka)))
    rep("M4", max(devs_f) < 1e-12,
        "四档冻结档数下冻结前后保持度差全为零（最大 %.2e）" % max(devs_f))

    # ---------------- EP5 恒真检验 ----------------
    print("\n--- EP5 恒真检验（改掉它就 FAIL） ---")
    # 反例一：交换类算子（L_j 型左乘在 C^2 上的对应）必须给出与保持类不同的读数
    # 在 C^2 上，与 J_M 反对易且酉的算子即交换类
    anti = SX  # {SX, J_M} = 0
    comm_check = float(np.linalg.norm(anti @ J_M + J_M @ anti))
    rep("EP5-1", comm_check < 1e-12,
        "反例算子 S_x 与 J_M 反对易（范数 %.2e），确属交换类候选" % comm_check)
    k_a, s_a = branch_readout(anti)
    rep("EP5-2", s_a > 0.99 and k_a < 0.01,
        "反例算子给出纯交换（交换度 %.6f），与 R(phi) 的纯保持分离；"
        "若判据对二者同答则失效" % s_a)

    # 反例三：若「冻结」实现为空操作（kappa -> 0），M2 仍会 PASS，
    # 故须另立一条：冻结确实改变了相位本身（否则 M2 的 PASS 不含信息）
    phi_test = 0.71
    dphi_frozen = abs(freeze(phi_test, kappa) - phi_test)
    dphi_free = abs(freeze(phi_test, 0.0) - phi_test)
    rep("EP5-3", dphi_frozen > 1e-3 and dphi_free < 1e-12,
        "冻结确实移动了相位（位移 %.4f），而 kappa=0 时位移为零；"
        "故 M2 的「分支行为不变」是对真实冻结的读数" % dphi_frozen)

    # 反例四：构造一个「冻结会翻转分支」的假想模型，确认判据能检出
    # 取 M_fake = R(phi) 后接 S_x（交换类），模拟「锁定导致分支重拨」
    k_fake, s_fake = branch_readout(SX @ rotation(phi))
    rep("EP5-4", s_fake > 0.99,
        "假想的「锁定翻转分支」模型给出交换度 %.6f，判据能检出该情形"
        "（故 M2 的保持结论非恒真）" % s_fake)

    # ---------------- M5 结论强度核验：M2 是否接近构造恒真 ----------------
    print("\n--- M5 结论强度核验：M2 的保持结论有多少信息量 ---")
    # R(phi) = exp(-i phi J/2) 本身是 J 的函数，与 J 对易是构造决定的，
    # 故 M2 的「冻结不改变分支行为」须核：一般算符落在保持类的比例是多少？
    rng = np.random.default_rng(20261018)
    keep_vals = []
    for _ in range(2000):
        A = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        Q, _ = np.linalg.qr(A)
        keep_vals.append(branch_readout(Q)[0])
    keep_vals = np.array(keep_vals)
    frac_pure_keep = float((keep_vals > 0.99).mean())
    frac_pure_swap = float((keep_vals < 0.01).mean())
    print("    随机 U(2) 的保持度：中位数 %.4f，纯保持占比 %.3f，纯交换占比 %.3f"
          % (np.median(keep_vals), frac_pure_keep, frac_pure_swap))
    rep("M5", frac_pure_keep < 0.10,
        "一般 U(2) 中纯保持仅占 %.1f%%，故 M2 的保持结论不是「随便什么算符都成立」；"
        "但 R(phi) 落在该 2%% 内由构造决定（R 是 J 的函数），"
        "此条限制 M2 的信息量，须在注记中明写" % (100 * frac_pure_keep))

    # ---------------- 汇总 ----------------
    n_pass = sum(1 for x in LOG if x.startswith("[PASS]"))
    n_all = len(LOG)
    print("\n=== 汇总：%d/%d PASS ===" % (n_pass, n_all))

    make_plot(phis, rows, kappas, folds)
    return n_pass, n_all


def make_plot(phis, rows, kappas, folds):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    x = np.arange(len(rows))
    ax.bar(x - 0.2, [r[1] for r in rows], 0.4, label="冻结前保持度", color="steelblue")
    ax.bar(x + 0.2, [r[3] for r in rows], 0.4, label="冻结后保持度", color="orange")
    ax.set_xticks(x)
    ax.set_xticklabels(["phi=%.2f" % r[0] for r in rows])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("保持度")
    ax.set_title("M2: 冻结前后分支行为不变")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    ax.plot(kappas, [1.0] * len(kappas), "o-", label="保持度（冻结前后同值）")
    ax.set_xscale("log")
    ax.set_ylim(0.8, 1.05)
    ax.set_xlabel("锁定强度 kappa_lock")
    ax.set_ylabel("保持度")
    ax.set_title("M3: 四档锁定强度下不变")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    labels = ["R(phi)\n(保持类)", "S_x\n(交换类)", "S_x·R(phi)\n(假想翻转)"]
    keeps = [branch_readout(rotation(0.71))[0],
             branch_readout(SX)[0],
             branch_readout(SX @ rotation(0.71))[0]]
    swaps = [1 - k for k in keeps]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, keeps, 0.4, label="保持度", color="green", alpha=0.7)
    ax.bar(x + 0.2, swaps, 0.4, label="交换度", color="red", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_title("EP5: 判据能检出翻转情形")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out = r"D:\software\TianXingFC\images\p681_freeze_branch.png"
    fig.savefig(out, dpi=150)
    print("图已保存: %s" % out)


if __name__ == "__main__":
    main()
