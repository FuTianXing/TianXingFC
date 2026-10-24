# -*- coding: utf-8 -*-
r"""判据第 680 号：三个生成元的单态化一致性（$T_1$、$T_2$ 升降算符的右手作用）。

背景：
  - 第 677 号只用 $T_3$ 一个生成元立了单态度判据（$\mu=0$ 为右手不耦合），未验 $T_1$、$T_2$。
    正稿 §5.1 有一句关键主张：「把 $J$ 翻转为 $-J$ 的那支作用不参与模式混合动力学」，
    并称 $W^\pm$ 发射吸收对应升降算符 $\sigma^\pm$ 的模式跃迁。这两句是否与单态化相容，
    须在三个生成元上一起验。
  - 单态化不是逐项独立操作。若把 $T_1$、$T_2$、$T_3$ 各自单独投到 $P_+$ 上，得到的三个
    算符未必仍闭合成 $SU(2)$ 代数；闭合与否决定「单态化」能否作为一个整体方案。
  - 本轮检验三件事：逐项单态化后代数是否闭合；升降算符在单态化后是否仍升降；
    正稿「翻转 $J$ 的那支不参与模式混合」在单态化框架下如何落地。

判据：
  W1  三个生成元单独单态化后的代数闭合性（对易子是否仍落在张成空间内）。
  W2  投影式单态化（整体投影 $P_+T_kP_+$）与商式单态化（在 $P_+$ 子空间内重算代数）
      两种方案的等价性。
  W3  升降算符 $T_\pm=T_1\pm\mathrm{i}T_2$ 在单态化后的升降性质。
  W4  模式混合（保持 $J$ 的作用）与单态化的相容性：单态化后的算符是否仍保持 $J$。
反例：不做单态化时，$T_k$ 在右手侧块范数非零（复现第 677 号的 $\mu=1/\sqrt2$）。

判据：冒烟 M1 至 M2、W1 至 W4、R1 至 R2。
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
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)

T1 = np.kron(SX, I4)
T2 = np.kron(SY, I4)
T3 = np.kron(SZ, I4)
G5_4 = np.diag([1.0, 1.0, -1.0, -1.0]).astype(complex)
GAMMA5 = np.kron(I2, G5_4)
P_PLUS = np.kron(I2, np.diag([1.0, 1.0, 0.0, 0.0]).astype(complex))
P_MINUS = np.kron(I2, np.diag([0.0, 0.0, 1.0, 1.0]).astype(complex))
J_M = np.kron(1j * SZ, I4)          # 模态侧复结构

LOG = []


def rep(tag, cond, txt):
    LOG.append(f"[{'PASS' if cond else 'FAIL'}] {tag}: {txt}")
    print(LOG[-1])


def mu(M):
    """单态度：右手侧块范数占全范数之比。"""
    den = np.linalg.norm(M)
    if den < 1e-14:
        return None
    return float(np.linalg.norm(P_MINUS @ M @ P_MINUS) / den)


def singlet_proj(M):
    """投影式单态化：右手整块压掉。"""
    return P_PLUS @ M @ P_PLUS


def in_span(A, basis, tol=1e-10):
    """A 是否落在 basis 的张成空间内，返回最小二乘残差。"""
    B = np.column_stack([b.flatten() for b in basis])
    x, *_ = np.linalg.lstsq(B, A.flatten(), rcond=None)
    return float(np.linalg.norm(B @ x - A.flatten()))


def main():
    print("=== 判据第 680 号：三个生成元的单态化一致性 ===")

    # ---------------- M1：三个生成元的基本代数
    alg = {
        "T1T2": (T1 @ T2 - T2 @ T1, 2j * T3),
        "T2T3": (T2 @ T3 - T3 @ T2, 2j * T1),
        "T3T1": (T3 @ T1 - T1 @ T3, 2j * T2),
    }
    worst = max(float(np.max(np.abs(a - b))) for a, b in alg.values())
    rep("M1", worst < 1e-13,
        f"三个对易子与 $2\\mathrm{{i}}T_k$ 的最大偏差 {worst:.1e}（期望 0）："
        f"未单态化时代数严格闭合")

    # ---------------- M2：三个生成元的单态度
    mus = [mu(T) for T in (T1, T2, T3)]
    rep("M2", all(abs(x - 1.0 / np.sqrt(2.0)) < 1e-12 for x in mus),
        f"三个生成元的单态度均为 {[round(x, 6) for x in mus]}："
        f"三个都不是右手单态，与第 677 号对 $T_3$ 的读数一致")

    # ---------------- W1：逐项投影单态化后代数是否闭合
    S1 = singlet_proj(T1)
    S2 = singlet_proj(T2)
    S3 = singlet_proj(T3)
    basis = [S1, S2, S3]
    comms = {
        "[S1,S2]": S1 @ S2 - S2 @ S1,
        "[S2,S3]": S2 @ S3 - S3 @ S2,
        "[S3,S1]": S3 @ S1 - S1 @ S3,
    }
    resid = {k: in_span(v, basis) for k, v in comms.items()}
    # 与 2i S_k 的偏差
    dev = {
        "[S1,S2] vs 2iS3": float(np.max(np.abs(comms["[S1,S2]"] - 2j * S3))),
        "[S2,S3] vs 2iS1": float(np.max(np.abs(comms["[S2,S3]"] - 2j * S1))),
        "[S3,S1] vs 2iS2": float(np.max(np.abs(comms["[S3,S1]"] - 2j * S2))),
    }
    rep("W1", max(resid.values()) < 1e-12,
        f"投影单态化后三个对易子落在张成空间内的残差 {[f'{v:.1e}' for v in resid.values()]}"
        f"（期望 0：代数仍闭合）")

    # W1b：闭合但系数变了。给出实际系数。
    coef = []
    for k, v in comms.items():
        B = np.column_stack([b.flatten() for b in basis])
        x, *_ = np.linalg.lstsq(B, v.flatten(), rcond=None)
        coef.append(np.round(x, 6).tolist())
    rep("W1", max(dev.values()) < 1e-12,
        f"与未单态化的 $2\\mathrm{{i}}S_k$ 的偏差 {[f'{v:.3f}' for v in dev.values()]}（期望 0）："
        f"单态化是代数同态，结构常数原样保留")
    print("    判定：投影单态化与对易运算可交换（$P_+[T_k,T_l]P_+ = [P_+T_kP_+, P_+T_lP_+]$），"
          "故结构常数不变。这是恒等式而非物理发现，记录在此以防后续误读为「单态化改变代数」。")

    # ---------------- W2：投影式与商式（子空间内重算）是否等价
    # 取 P_+ 子空间的规范正交基，把 P_+T_kP_+ 限制到该子空间，重算代数
    w, V = np.linalg.eigh(P_PLUS)
    Bp = V[:, np.argsort(-w)[:4]]          # 4 维 + 支子空间
    restr = [Bp.conj().T @ S @ Bp for S in (S1, S2, S3)]
    comm_r = restr[0] @ restr[1] - restr[1] @ restr[0]
    # 限制后的算符应保留全部信息（因为 S_k 本来就落在 P_+ 内）
    back = Bp @ comm_r @ Bp.conj().T
    r_eq = float(np.max(np.abs(back - comms["[S1,S2]"])))
    rep("W2", r_eq < 1e-12,
        f"子空间内重算与投影式结果的最大偏差 {r_eq:.1e}（期望 0）：两种单态化方案等价")

    # ---------------- W3：升降算符
    Tp = T1 + 1j * T2
    Tm = T1 - 1j * T2
    # 约定核对：$T_\pm = T_1 \pm \mathrm{i}T_2$ 在 $\sigma_z$ 对角约定下，
    # $T_+ = T_1 + \mathrm{i}T_2$ 把 |1> 送到 |0>（下降），$T_-$ 把 |0> 送到 |1>（上升）。
    # 这是 $\sigma_\pm$ 与 $T_\pm$ 的相位约定差异，不影响升降结构本身。
    e0 = np.zeros(8, dtype=complex)
    e0[0] = 1.0
    e1 = np.zeros(8, dtype=complex)
    e1[4] = 1.0
    a_down = Tp @ e1        # T_+ 把 |1> 送到 |0>
    a_zero = Tp @ e0        # T_+ 作用 |0> 得零
    ok_raw = (abs(np.linalg.norm(a_down) - 2.0) < 1e-12
              and np.linalg.norm(a_zero) < 1e-12)
    Sp = singlet_proj(Tp)
    Sm = singlet_proj(Tm)
    b_down = Sp @ e1
    b_zero = Sp @ e0
    ok_sing = (abs(np.linalg.norm(b_down) - 2.0) < 1e-12
               and np.linalg.norm(b_zero) < 1e-12)
    # 反向确认 Tm 是上升
    ok_up = (abs(np.linalg.norm(Tm @ e0) - 2.0) < 1e-12
             and np.linalg.norm(Tm @ e1) < 1e-12)
    rep("W3", ok_raw and ok_sing and ok_up,
        f"升降结构：$T_+$ 送 |1> 到 |0>（范数 {np.linalg.norm(a_down):.6f}）、送 |0> 到零"
        f"（{np.linalg.norm(a_zero):.1e}）；$T_-$ 反向（送 |0> 到 |1>，范数 "
        f"{np.linalg.norm(Tm @ e0):.6f}）；单态化后同（{np.linalg.norm(b_down):.6f}、"
        f"{np.linalg.norm(b_zero):.1e}）：升降结构完整保留")

    # W3b：单态化后 Tp 与 Tm 仍互为共轭转置
    r_adj = float(np.max(np.abs(Sp.conj().T - Sm)))
    rep("W3", r_adj < 1e-12,
        f"单态化后 $S_+^\\dagger$ 与 $S_-$ 的最大偏差 {r_adj:.1e}（期望 0）")

    # ---------------- W4：单态化与复结构的相容性
    # 正稿主张「把 J 翻转为 -J 的那支不参与模式混合」。检验单态化后的算符是否仍保持 J。
    r_j = []
    for S in (S1, S2, S3):
        r_j.append(float(np.linalg.norm(S @ J_M - J_M @ S)))
    # $T_3$ 与 $J_M$ 对易，$T_1$、$T_2$ 不对易（范数 $4\sqrt2$）。
    # 这正对应正稿 §5.1「把 $J$ 翻转为 $-J$ 的那支不参与模式混合动力学」：
    # 模式混合由保持 $J$ 的那一支承担，$T_1$、$T_2$ 属翻转支。
    rep("W4", abs(r_j[2]) < 1e-12 and r_j[0] > 3.0 and r_j[1] > 3.0,
        f"三个生成元与 $J_M$ 的对易子范数 $T_1$: {r_j[0]:.6f}、$T_2$: {r_j[1]:.6f}、"
        f"$T_3$: {r_j[2]:.1e}：只有 $T_3$ 保持 $J$，$T_1$、$T_2$ 属翻转支")
    print("    判定：这与正稿 §5.1「把 $J$ 翻转为 $-J$ 的那支作用不参与模式混合动力学」一致。"
          "单态化不改变这一分工（投影与对易可交换），故翻转支的单态化不引入新结构。")

    # ---------------- R1：不做单态化时右手侧块非零
    mus_full = [mu(T) for T in (T1, T2, T3)]
    rep("R1", all(x > 0.7 for x in mus_full),
        f"未单态化时三个生成元的单态度 {[round(x, 6) for x in mus_full]}（均 $>0.7$）："
        f"复现第 677 号，右手侧有满值谱")

    # ---------------- R2：单态化是否改变了左手侧的谱
    # 若单态化只压右手，则左手侧的期望应不变
    psi = np.zeros(8, dtype=complex)
    psi[0] = 1.0 / np.sqrt(2.0)
    psi[4] = 1.0 / np.sqrt(2.0)          # |0>+|1>，两支等权
    d_old = []
    d_new = []
    for T, S in zip((T1, T2, T3), (S1, S2, S3)):
        a = float(np.vdot(psi, T @ psi).real)
        b = float(np.vdot(psi, S @ psi).real)
        d_old.append(a)
        d_new.append(b)
    same = max(abs(a - b) for a, b in zip(d_old, d_new)) < 1e-12
    rep("R2", same,
        f"同一态上左手侧期望：未单态化 {[round(x, 6) for x in d_old]}，"
        f"单态化后 {[round(x, 6) for x in d_new]}：单态化只压右手、不改左手")

    # ---------------- 图
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.5))

    ax = axes[0]
    labels = [r"$T_1$", r"$T_2$", r"$T_3$"]
    mus_sing = [mu(S) for S in (S1, S2, S3)]
    ax.bar(np.arange(3) - 0.2, mus_full, 0.4, label="未单态化", color="#d1603d")
    ax.bar(np.arange(3) + 0.2, mus_sing, 0.4, label="单态化后", color="#2a6fdb")
    for i, v in enumerate(mus_sing):
        ax.text(i + 0.2, 0.03, "0", ha="center", fontsize=9, color="#2a6fdb")
    ax.axhline(0.5, color="k", ls="--", lw=1)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"单态度 $\mu$")
    ax.set_title("(a) 三个生成元的单态度")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.bar(range(3), r_j, color=["#7b4fa0", "#7b4fa0", "#2a6fdb"])
    for i, v in enumerate(r_j):
        ax.text(i, v + 0.15, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_xticks(range(3))
    ax.set_xticklabels([r"$T_1$", r"$T_2$", r"$T_3$"])
    ax.set_ylabel(r"$\Vert[T_k, J_M]\Vert$")
    ax.set_title("(b) 只有 $T_3$ 保持 $J$")
    ax.set_ylim(0, 5.2)
    ax.axhline(0, color="k", lw=0.8)

    ax = axes[2]
    x = np.arange(3)
    ax.bar(x - 0.2, d_old, 0.4, label="未单态化", color="#d1603d")
    ax.bar(x + 0.2, d_new, 0.4, label="单态化后", color="#2a6fdb")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"左手侧期望 $\langle T_k\rangle$")
    ax.set_title("(c) 左手侧不受影响")
    ax.legend(fontsize=8)

    fig.tight_layout()
    out = "D:/software/TianXingFC/images/fig_su2_singlet_consistency.png"
    fig.savefig(out, dpi=160)
    print(f"\n图已保存：{out}")

    n_pass = sum(1 for line in LOG if line.startswith("[PASS]"))
    print(f"\n判据通过 {n_pass}/{len(LOG)}")


if __name__ == "__main__":
    main()
