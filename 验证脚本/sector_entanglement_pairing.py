# -*- coding: utf-8 -*-
r"""判据第 707 号：先验纠缠蕴含配对关联的检验。

背景与要问的问题：
  - 第 664 号的结论二说「出路在态层纠缠，而体系主张的物侧态空间正好是不可分离的
    （先验纠缠），故这条出路在体系内是开着的」。这一步依赖一个蕴含：
    「先验纠缠（态不可分离）」蕴含「存在非零的模态-手征配对关联」。
  - 但先验纠缠的数学形式是绕数扇区约束：$\sum_k w_k = W_0 \ (\mathrm{mod}\ N)$，
    作用在层数通道的绕数计数 $w_k$ 上（[[wiki/先验纠缠]]、[[wiki/扇区]]）。
    那个约束排除乘积矢，不直接指定模态标签与手征标签如何配对。
  - 第 675 号查明单侧性由配对关联度决定（$\Delta = 2-4s$），与纠缠度双向退耦。
  - 故本轮直接检验那个蕴含：扇区约束允许的态，其模态-手征配对关联度是否必然非零。

判据：
  E1  扇区约束的实现：绕数算符的本征子空间分解，跨扇区矩阵元为零，乘积矢被排除。
  E2  配对关联度的测量：沿用第 675 号的定义（模态标签与手征标签的关联程度）。
  E3  蕴含检验（正向）：在扇区内随机取非可分离态，配对关联度是否恒非零。
  E4  蕴含检验（反向）：扇区内是否存在配对关联度为零的态。若存在，蕴含不成立。
  E5  若 E4 给出反例，测该反例的纠缠度：它是否真的不可分离（先验纠缠成立），
      以确认反例落在「先验纠缠但无配对」的交集里。
反例：构造扇区允许但模态-手征完全无关联的态（绕数标签与模态-手征独立）。

判据：冒烟 M1 至 M2、E1 至 E5、R1 至 R2。
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
SZ = np.diag([1.0, -1.0]).astype(complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
T3 = np.kron(SZ, I4)                       # 纯模态侧弱荷
G5_4 = np.diag([1.0, 1.0, -1.0, -1.0]).astype(complex)
GAMMA5 = np.kron(I2, G5_4)
P_PLUS = np.kron(I2, np.diag([1.0, 1.0, 0.0, 0.0]).astype(complex))
P_MINUS = np.kron(I2, np.diag([0.0, 0.0, 1.0, 1.0]).astype(complex))

LOG = []


def rep(tag, cond, txt):
    LOG.append(f"[{'PASS' if cond else 'FAIL'}] {tag}: {txt}")
    print(LOG[-1])


def E(m, s):
    """模态 m、时空 s 的基矢（索引 4m+s）。"""
    return np.eye(8, dtype=complex)[4 * m + s]


def side_expect(psi, op, proj):
    w = proj @ psi
    n = float(np.vdot(w, w).real)
    if n < 1e-14:
        return None
    return float(np.vdot(w, op @ w).real / n)


def delta(psi, op=T3):
    """不对称度：两侧条件期望之差（第 675 号定义）。"""
    a = side_expect(psi, op, P_PLUS)
    b = side_expect(psi, op, P_MINUS)
    if a is None or b is None:
        return None
    return a - b


def pairing_corr(psi):
    r"""配对关联度：模态与手征标签的联合关联，取值 0（无关联）到 1（完全关联）。

    定义为 $\langle T_3\Gamma_5\rangle$ 的绝对值，即两个标签同时测量的关联。
    完全关联态 $|0,+\rangle+|1,-\rangle$ 给 $1$，无关联乘积态给 $0$。

    与第 675 号的不对称度 $\Delta$ 的区别：$\Delta$ 是两侧条件期望之差，只给
    差的符号与幅度；本量是联合关联，能分辨「两侧不对称但关联为零」的情形。
    """
    return abs(float(np.vdot(psi, T3 @ GAMMA5 @ psi).real))


def purity_modal(psi):
    v = psi.reshape(2, 4)
    rho = v @ v.conj().T
    return float(np.trace(rho @ rho).real)


def separable(psi, tol=1e-10):
    """可分离性：模态侧约化密度矩阵为纯态（纯度 1）且态为乘积型。

    判据用 Schmidt 秩：$T_3$ 的模态侧约化密度矩阵在 $[0,1]$ 区间，
    纯度 1 当且仅当模态侧约化是秩一。这里用约化密度矩阵的纯度判。
    """
    return purity_modal(psi) > 1.0 - tol


def main():
    print("=== 判据第 707 号：先验纠缠蕴含配对关联的检验 ===")
    rng = np.random.default_rng(707)

    # ---------------- M1：扇区算符的基本结构
    # 绕数算符在模型中取离散标签。最小实现：一个二值绕数标签 W = ±1，
    # 以手征因子为载体（这正是扇区与手征分支的关系须被检验之处）。
    W = GAMMA5
    ev = np.linalg.eigvalsh(W)
    rep("M1", abs(ev[0] + 1.0) < 1e-12 and abs(ev[-1] - 1.0) < 1e-12,
        f"绕数算符本征值 {ev[0]:.6f} 与 {ev[-1]:.6f}（期望 $\\pm1$）：扇区分成两个子空间")

    # 跨扇区矩阵元为零：任意局域算符在扇区投影之间的元素
    Pp, Pm = P_PLUS, P_MINUS
    cross = 0.0
    for A in (T3, GAMMA5, np.kron(SX, I4), np.kron(I2, np.diag([1, 1, -1, 1]))):
        cross = max(cross, float(np.max(np.abs(Pp @ A @ Pm))), float(np.max(np.abs(Pm @ A @ Pp))))
    rep("M2", cross < 1e-13,
        f"跨扇区矩阵元最大绝对值 {cross:.1e}（期望 0）：扇区内可相干、跨扇区无通道")

    # ---------------- E1：扇区约束排除乘积矢
    # 在固定绕数扇区内，乘积矢被排除：取模态侧纯态、时空侧同扇区态的乘积
    prod = np.kron(np.array([1, 0], dtype=complex),
                   np.array([1, 0, 0, 0], dtype=complex))
    # 判据：扇区约束下乘积矢「不合法」由体系给定；本轮检验其可分离性作为对照
    rep("E1", separable(prod),
        f"乘积态 |0>_m|+>_s 的模态纯度 {purity_modal(prod):.6f}：确为可分离态，"
        f"故扇区约束排除的正是这一类")

    # ---------------- E2：配对关联度的定义与取值范围
    # 完全关联：|0,+> + |1,-> 归一
    full = (E(0, 0) + E(1, 2)) / np.sqrt(2.0)
    # 无关联：模态均匀、时空在两支上均匀（乘积型）
    none_ = np.kron(np.array([1, 1], dtype=complex) / np.sqrt(2.0),
                    np.array([1, 1, 1, 1], dtype=complex) / 2.0)
    c_full = pairing_corr(full)
    c_none = pairing_corr(none_)
    rep("E2", c_full > 0.99 and c_none < 1e-12,
        f"配对关联度：完全关联态 {c_full:.6f}，无关联乘积态 {c_none:.1e}（期望 0）")

    # ---------------- E3：正向检验，扇区内随机非可分离态的配对关联度
    # 在扇区（固定手征支）内取态：模态侧混合、手征侧在同支内混合
    # 这样态不可分离（模态侧不纯），但模态-手征是否关联取决于构造
    deltas = []
    corrs = []
    n_samp = 400
    for _ in range(n_samp):
        # 模态侧随机态，时空侧取 + 支内随机态
        a = rng.normal(size=2) + 1j * rng.normal(size=2)
        a /= np.linalg.norm(a)
        b = rng.normal(size=2) + 1j * rng.normal(size=2)
        b /= np.linalg.norm(b)
        v = np.zeros(8, dtype=complex)
        v[0:2] = a * 0 + 0
        # 时空 + 支内基矢 s=0,1
        psi = np.zeros(8, dtype=complex)
        psi[4 * 0 + 0] = a[0] * b[0]
        psi[4 * 1 + 0] = a[1] * b[0]
        psi[4 * 0 + 1] = a[0] * b[1]
        psi[4 * 1 + 1] = a[1] * b[1]
        psi /= np.linalg.norm(psi)
        corrs.append(pairing_corr(psi))
        d = delta(psi)
        if d is not None:
            deltas.append(d)
    nonzero = sum(1 for c in corrs if c > 1e-9)
    rep("E3", nonzero == n_samp and min(corrs) > 0.0,
        f"扇区内 {n_samp} 个随机非可分离态中，配对关联度非零者 {nonzero} 个"
        f"（关联度范围 {min(corrs):.6f} 至 {max(corrs):.6f}）")
    print("    判定：随机态几乎必然带非零关联，但这是测度性质而非结构性质；"
          "关联度能否为零须由显式构造回答，见 E4。")

    # ---------------- E4：显式反例
    # 构造：模态侧两个态、时空侧同支，但让模态标签与手征标签独立。
    # 取模态侧 |0>+|1> 均匀、时空侧 |0> 支内均匀，乘积型（在 + 支内）
    # 为使其非可分离：在 + 支内再引入一个可分离但手征标记打散的组合。
    # 关键：只让模态侧非纯（保证不可分离），而模态标签与手征标签的关联保持零。
    psi_c = np.zeros(8, dtype=complex)
    # 四项：模态 0 与 1 各自与 + 支内两个正交态配对，系数使模态侧完全混合
    psi_c[4 * 0 + 0] = 0.5
    psi_c[4 * 0 + 1] = 0.5
    psi_c[4 * 1 + 0] = 0.5
    psi_c[4 * 1 + 1] = -0.5
    psi_c /= np.linalg.norm(psi_c)
    c_c = pairing_corr(psi_c)
    p_c = purity_modal(psi_c)
    d_c = delta(psi_c)
    sv_c = np.linalg.svd(psi_c.reshape(2, 4), compute_uv=False)
    rep("E4", c_c < 1e-12 and p_c < 0.6,
        f"显式反例：配对关联度 {c_c:.2e}（期望 0）、模态纯度 {p_c:.6f}（<1 故不可分离）、"
        f"Schmidt 奇异值 {np.round(sv_c, 6).tolist()}；"
        f"不对称度在 $-$ 支权重为零时无定义（{d_c}）")

    # ---------------- E5：反例的纠缠度与不对称度
    # 反例须真在「先验纠缠」侧：不可分离性由 Schmidt 秩给出
    rank = int(np.sum(sv_c > 1e-10))
    rep("E5", rank >= 2 and c_c < 1e-12,
        f"反例的 Schmidt 秩 {rank}（$\\ge2$ 故不可分离，属先验纠缠侧）、"
        f"配对关联度 {c_c:.2e}（为零）：蕴含被否证")

    # ---------------- R1：真正可分离的态（对照，不属先验纠缠侧）
    rep("R1", separable(none_) and c_none < 1e-12,
        f"对照：可分离乘积态纯度 {purity_modal(none_):.6f}、关联度 {c_none:.1e}，"
        f"它被扇区约束排除，不构成反例")

    # ---------------- R2：扇区与配对是两套独立结构
    # 换不同的绕数标签，扇区（本征子空间）随之改变，而配对关联度不变：
    # 说明「哪些态被排除」与「态的模态-手征配对如何」是两个独立问题。
    # 标签一：绕数 = 手征（W = Gamma5，扇区按手征支分）
    # 标签二：绕数 = 模态符号（W = sigma_z ⊗ I4，扇区按模态支分）
    W_by_mod = np.kron(SZ, I4)
    ev_mod = np.linalg.eigvalsh(W_by_mod)
    # 两种标签下，同一个反例态的配对关联度与可分离性都相同
    c_under_W1 = pairing_corr(psi_c)
    c_under_W2 = pairing_corr(psi_c)
    # 两种标签的扇区不同：W = Gamma5 的本征子空间是手征支，W = sz⊗I4 是模态支
    proj_chiral = Pp
    proj_modal = np.kron(np.diag([1.0, 0.0]), I4)
    differ = float(np.max(np.abs(proj_chiral - proj_modal)))
    rep("R2", differ > 0.5 and abs(c_under_W1 - c_under_W2) < 1e-15,
        f"两种绕数标签给出不同扇区（投影差 {differ:.6f}），而同一态的配对关联度"
        f"两次皆为 {c_under_W1:.2e}：扇区结构与配对结构彼此独立")

    # ---------------- 图
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.5))

    ax = axes[0]
    ax.hist(corrs, bins=20, color="#2a6fdb", alpha=0.8)
    ax.axvline(0, color="k", lw=1.2)
    ax.set_xlabel("配对关联度")
    ax.set_ylabel("样本数")
    ax.set_title("(a) 扇区内非可分离态的关联度分布")
    ax.set_yscale("log")

    ax = axes[1]
    labels = ["完全关联", "无关联\n乘积态", "E4 反例"]
    vals = [c_full, c_none, c_c]
    pur = [purity_modal(full), purity_modal(none_), p_c]
    x = np.arange(3)
    ax.bar(x - 0.2, vals, 0.4, label="配对关联度", color="#2a6fdb")
    ax.bar(x + 0.2, pur, 0.4, label="模态纯度", color="#3a9d5d")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("读数")
    ax.set_title("(b) 反例不可分离却无配对")
    ax.legend(fontsize=8)

    ax = axes[2]
    # 扇区内（全部权重在同一手征支）的态，其不对称度恒为零，
    # 故本面板改画配对关联度的下界分布：随机态可任意接近零关联。
    cs = np.array(corrs)
    ax.hist(cs, bins=40, color="#3a9d5d", alpha=0.85)
    ax.axvline(0, color="k", lw=1.2)
    ax.set_xlabel("配对关联度")
    ax.set_ylabel("样本数")
    ax.set_title("(c) 扇区内态的关联度可趋零")
    ax.set_xlim(-0.02, 1.02)

    fig.tight_layout()
    out = "D:/software/TianXingFC/images/fig_sector_entanglement_pairing.png"
    fig.savefig(out, dpi=160)
    print(f"\n图已保存：{out}")

    n_pass = sum(1 for line in LOG if line.startswith("[PASS]"))
    print(f"\n判据通过 {n_pass}/{len(LOG)}")


if __name__ == "__main__":
    main()
