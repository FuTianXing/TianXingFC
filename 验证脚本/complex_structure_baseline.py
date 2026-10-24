# -*- coding: utf-8 -*-
r"""判据第 685 号：复结构基准的选定（$\mathbb{C}^2$ 上复结构族的分类与模式混合的唯一基准）。

背景：
  - 第 662 号用 $J_0=L_{\mathrm{i}}$（四元数左乘 $\mathrm{i}$，在 $\mathbb{C}^2$ 上即乘 $\mathrm{i}$），
    读出 $T_1$、$T_2$、$T_3$ 三者与 $J_0$ 全对易。
  - 第 680 号用 $J_M=\mathrm{i}\sigma_z\otimes I_4$，读出只有 $T_3$ 对易、$T_1$ 与 $T_2$
    的对易子范数 $4$。
  - 两轮读数看似矛盾。本轮查明：$\mathbb{C}^2$ 上满足 $J^2=-I$ 的厄米复结构不唯一，
    $\{I,\sigma_x,\sigma_y,\sigma_z\}$ 各乘 $\mathrm{i}$ 都是合法复结构，它们与生成元的
    对易性各不相同。所以「哪个 $J$」是一个必须判定的问题，不能默认为某一个。
  - 正稿 §5.1 称「模式混合（模态态矢量的矩阵左乘）保持 $J$」。本轮检验哪一个 $J$ 满足这条，
    并给出复结构族的分类。

判据：
  X1  复结构族的列举：$\mathrm{i}$ 乘四个 Pauli 矩阵，全部满足 $J^2=-I$、反厄米。
  X2  族内对易性分类：按与 $T_1$、$T_2$、$T_3$ 的对易谱分档。
  X3  模式混合的基准唯一性：$SU(2)$ 左乘（模式混合）保持哪些 $J$。
  X4  第 662 号与第 680 号的读数在同一框架下复现，说明两轮不矛盾。
反例：$\mathrm{i}\sigma_z$ 不被 $SU(2)$ 全群保持（只有 $U(1)$ 子群保持），故它不能作模式混合的基准。

判据：冒烟 M1 至 M2、X1 至 X4、R1 至 R2。
可逆性等级：判据稳健（解析构造与范数读数）。
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
GENS = {"T1": SX, "T2": SY, "T3": SZ}

# 四个候选复结构
CANDS = {
    "i*I": 1j * I2,
    "i*sx": 1j * SX,
    "i*sy": 1j * SY,
    "i*sz": 1j * SZ,
}

LOG = []


def rep(tag, cond, txt):
    LOG.append(f"[{'PASS' if cond else 'FAIL'}] {tag}: {txt}")
    print(LOG[-1])


def su2(k, theta):
    """SU(2) 元素 exp(i theta sigma_k / 2)。"""
    return np.cos(theta / 2.0) * I2 + 1j * np.sin(theta / 2.0) * (SX, SY, SZ)[k]


def main():
    print("=== 判据第 685 号：复结构基准的选定 ===")

    # ---------------- M1：四个候选都满足复结构定义
    r = []
    for name, J in CANDS.items():
        r.append(float(np.linalg.norm(J @ J + I2)))
        r.append(float(np.linalg.norm(J + J.conj().T)))
    rep("M1", max(r) < 1e-14,
        f"四个候选的 $J^2+I$ 与反厄米偏差最大 {max(r):.1e}（期望 0）："
        f"$\\{{\\mathrm{{i}}I,\\ \\mathrm{{i}}\\sigma_x,\\ \\mathrm{{i}}\\sigma_y,\\ \\mathrm{{i}}\\sigma_z\\}}$ 全是合法复结构")

    # ---------------- M2：复结构族是连续的（一般族参数化）
    # J(n) = i (n·sigma)，n 为单位矢量；n = (0,0,1) 给 i*sz
    rng = np.random.default_rng(685)
    worst = 0.0
    for _ in range(200):
        n = rng.normal(size=3)
        n /= np.linalg.norm(n)
        J = 1j * (n[0] * SX + n[1] * SY + n[2] * SZ)
        worst = max(worst, float(np.linalg.norm(J @ J + I2)))
    rep("M2", worst < 1e-14,
        f"随机单位矢量 200 个，$\\mathrm{{i}}(\\mathbf n\\cdot\\boldsymbol\\sigma)^2+I$ 最大偏差 "
        f"{worst:.1e}（期望 0）：复结构族是连续族（Bloch 球面）")

    # ---------------- X1：族内对易性分类
    table = {}
    for name, J in CANDS.items():
        table[name] = {k: float(np.linalg.norm(J @ G - G @ J)) for k, G in GENS.items()}
    # i*I 与三者全对易；i*sx 只与 sx 对易；等等
    ok = (all(v < 1e-13 for v in table["i*I"].values())
          and table["i*sx"]["T1"] < 1e-13 and table["i*sx"]["T2"] > 1.0
          and table["i*sy"]["T2"] < 1e-13 and table["i*sy"]["T1"] > 1.0
          and table["i*sz"]["T3"] < 1e-13 and table["i*sz"]["T1"] > 1.0)
    rep("X1", ok,
        "对易谱分类：$\\mathrm{i}I$ 与三者全对易；$\\mathrm{i}\\sigma_k$ 只与 $\\sigma_k$ 对易"
        f"（$\\mathrm{{i}}\\sigma_x$ 对 $T_1$: {table['i*sx']['T1']:.1e}、对 $T_2$: {table['i*sx']['T2']:.3f}）")

    # X1b：一般族的对易谱随 n 连续变化
    ns = np.linspace(0.0, 1.0, 11)
    comm_t3 = []
    for t in ns:
        n = np.array([0.0, np.sqrt(1.0 - t ** 2) if t < 1 else 0.0, t])
        n = n / np.linalg.norm(n)
        J = 1j * (n[0] * SX + n[1] * SY + n[2] * SZ)
        comm_t3.append(float(np.linalg.norm(J @ SZ - SZ @ J)))
    rep("X1", comm_t3[0] > 1.0 and comm_t3[-1] < 1e-13,
        f"沿 Bloch 球从 $\\sigma_x$ 转到 $\\sigma_z$：与 $T_3$ 的对易子由 {comm_t3[0]:.3f} "
        f"连续降到 {comm_t3[-1]:.1e}：对易性是基准的函数")

    # ---------------- X2：模式混合保持哪些 J
    # 模式混合 = SU(2) 矩阵左乘。检验 J -> U J U^dagger 在 U 取全群时是否不变。
    keep = {}
    for name, J in CANDS.items():
        worst_dev = 0.0
        for k in range(3):
            for th in (0.3, 1.0, 2.0, np.pi):
                U = su2(k, th)
                dev = float(np.linalg.norm(U @ J @ U.conj().T - J))
                worst_dev = max(worst_dev, dev)
        keep[name] = worst_dev
    rep("X2", keep["i*I"] < 1e-13,
        f"$SU(2)$ 全群作用下 $J\\to UJU^\\dagger$ 的最大偏差："
        f"$\\mathrm{{i}}I$: {keep['i*I']:.1e}（不变）；"
        f"$\\mathrm{{i}}\\sigma_x$: {keep['i*sx']:.3f}、$\\mathrm{{i}}\\sigma_y$: {keep['i*sy']:.3f}、"
        f"$\\mathrm{{i}}\\sigma_z$: {keep['i*sz']:.3f}（均改变）")

    # ---------------- X3：基准唯一性
    # 只有 i*I 被全群保持，故它是模式混合的唯一相容基准。
    others = [keep[k] for k in ("i*sx", "i*sy", "i*sz")]
    rep("X3", keep["i*I"] < 1e-13 and min(others) > 1.0,
        f"唯一性：只有 $\\mathrm{{i}}I$ 被 $SU(2)$ 全群保持（其余最小偏差 {min(others):.3f}），"
        f"故模式混合的基准唯一")

    # ---------------- X4：两轮读数复现
    # 662 号：用 i*I，三生成元全对易（对易子 0）
    d662 = [table["i*I"][k] for k in ("T1", "T2", "T3")]
    # 680 号：用 i*sz，只有 T3 对易（对易子 0），T1/T2 非零
    d680 = [table["i*sz"][k] for k in ("T1", "T2", "T3")]
    rep("X4", max(d662) < 1e-13 and abs(d680[2]) < 1e-13 and d680[0] > 1.0,
        f"同一框架下复现：基准 $\\mathrm{{i}}I$ 给 {[f'{x:.0e}' for x in d662]}（全对易，第 662 号）；"
        f"基准 $\\mathrm{{i}}\\sigma_z$ 给 {[f'{x:.1f}' for x in d680]}（只 $T_3$ 对易，第 680 号）")

    # ---------------- R1：i*sz 只被 U(1) 保持
    # 绕 z 转的 U(1) 子群保持 i*sz；绕 x 转不保持
    Uz = su2(2, 1.0)
    Ux = su2(0, 1.0)
    rz = float(np.linalg.norm(Uz @ CANDS["i*sz"] @ Uz.conj().T - CANDS["i*sz"]))
    rx = float(np.linalg.norm(Ux @ CANDS["i*sz"] @ Ux.conj().T - CANDS["i*sz"]))
    rep("R1", rz < 1e-13 and rx > 0.5,
        f"$\\mathrm{{i}}\\sigma_z$ 在 $U(1)_z$ 下不变（{rz:.1e}）、在绕 x 转下改变（{rx:.3f}）："
        f"它只被子群保持，不能作全群基准")

    # ---------------- R2：i*I 与 i*sz 给出的「手征支」不同
    # 检验两者的 +i 本征空间
    dims = {}
    for name in ("i*I", "i*sz"):
        w = np.linalg.eigvals(CANDS[name])
        dims[name] = int(np.sum(np.abs(w - 1j) < 1e-9))
    d_ii, d_isz = dims["i*I"], dims["i*sz"]
    rep("R2", d_ii == 2 and d_isz == 1,
        f"+i 本征空间维数：$\\mathrm{{i}}I$ 给 {d_ii}（整个 $\\mathbb{{C}}^2$）、"
        f"$\\mathrm{{i}}\\sigma_z$ 给 {d_isz}（一维复直线）：两者的分支结构不同")

    # ---------------- 图
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.5))

    ax = axes[0]
    labels = list(CANDS.keys())
    x = np.arange(len(labels))
    for i, (k, col) in enumerate(zip(("T1", "T2", "T3"), ("#d1603d", "#3a9d5d", "#2a6fdb"))):
        ax.bar(x + (i - 1) * 0.26, [table[n][k] for n in labels], 0.26,
               label=f"与 ${k}$", color=col)
    ax.set_xticks(x)
    ax.set_xticklabels([r"$\mathrm{i}\,I$", r"$\mathrm{i}\sigma_x$", r"$\mathrm{i}\sigma_y$", r"$\mathrm{i}\sigma_z$"])
    ax.set_ylabel("对易子范数")
    ax.set_title("(a) 对易谱随基准变化")
    ax.legend(fontsize=8)
    ax.text(0, 0.08, "全 0", ha="center", fontsize=9, color="#2a6fdb")
    ax.set_ylim(0, 3.4)

    ax = axes[1]
    ax.bar(range(4), [keep[n] for n in labels],
           color=["#2a6fdb", "#d1603d", "#d1603d", "#d1603d"])
    ax.set_xticks(range(4))
    ax.set_xticklabels([r"$\mathrm{i}\,I$", r"$\mathrm{i}\sigma_x$", r"$\mathrm{i}\sigma_y$", r"$\mathrm{i}\sigma_z$"])
    ax.set_ylabel(r"$SU(2)$ 全群作用下的最大改变量")
    ax.set_title("(b) 只有 $\\mathrm{i}I$ 被全群保持")
    ax.axhline(0, color="k", lw=0.8)

    ax = axes[2]
    ax.bar(range(2), [d_ii, d_isz], color=["#2a6fdb", "#e0a03a"])
    ax.set_xticks(range(2))
    ax.set_xticklabels([r"$\mathrm{i}I$", r"$\mathrm{i}\sigma_z$"])
    ax.set_ylabel(r"$+i$ 本征空间维数")
    ax.set_title("(c) 分支结构不同")
    ax.set_ylim(0, 2.5)
    for i, v in enumerate([d_ii, d_isz]):
        ax.text(i, v + 0.06, str(v), ha="center", fontsize=10)

    fig.tight_layout()
    out = "D:/software/TianXingFC/images/fig_complex_structure_baseline.png"
    fig.savefig(out, dpi=160)
    print(f"\n图已保存：{out}")

    n_pass = sum(1 for line in LOG if line.startswith("[PASS]"))
    print(f"\n判据通过 {n_pass}/{len(LOG)}")


if __name__ == "__main__":
    main()
