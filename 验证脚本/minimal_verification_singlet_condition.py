# -*- coding: utf-8 -*-
r"""判据第 677 号：单态条件（右手不耦合与两侧不对称的区分）。

背景：
  - 第 675 号把单侧性定位为配对关联度（$\Delta=2-4s$），但那里测的量是「$T_3$ 在两个
    手征扇区的条件期望之差」。这个量测的是「两侧不对称」。
  - 标准表述里「$SU(2)_L$ 只作用左手费米子」是更强的一句话：右手是单态，弱荷耦合
    恒等于零。两侧期望差为 $2$ 可以来自「左手 $+1$、右手 $-1$」（右手仍耦合，只是反号），
    这在结构上不同于「右手恒不耦合」。两者不可互代。
  - 本轮把两个条件分开立判据，并检验第 675 号的配对关联度结论在单态条件下是否仍成立。

判据：
  V1  两个条件的分离：构造「两侧不对称但右手仍耦合」的算符（$\sigma_z$ 型），它的
      单态度为零而不对称度为满值；构造「右手不耦合」的算符（$P_+$ 型），单态度为满值。
  V2  单态度定义：$\mu(M)=\lVert P_- M P_-\rVert/\lVert M\rVert$（右手侧块范数占比），
      零为完全单态，与不对称度 $\Delta$ 独立。
  V3  配对关联度在单态条件下：把弱荷算符换成 $P_+$ 型（右手恒零），重新扫配对份额，
      看 $\Delta$ 是否仍走 $2-4s$ 还是变为 $1-2s$。
  V4  标准表述的正确读法：$SU(2)_L$ 只作用左手要求 $\mu=0$；此条件比 $\Delta\neq0$ 强。
反例：$\mu=0$ 而 $\Delta=0$（右手不耦合且左手也不耦合）；$\mu=1$ 而 $\Delta=2$（只有右手耦合）。

判据：冒烟 M1 至 M2、V1 至 V4、R1 至 R2。
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
T3 = np.kron(SZ, I4)                       # sigma_z 型弱荷：两侧都有谱，右手反号
G5_4 = np.diag([1.0, 1.0, -1.0, -1.0]).astype(complex)
GAMMA5 = np.kron(I2, G5_4)
P_PLUS = np.kron(I2, np.diag([1.0, 1.0, 0.0, 0.0]).astype(complex))
P_MINUS = np.kron(I2, np.diag([0.0, 0.0, 1.0, 1.0]).astype(complex))

# 单态型弱荷：只在 + 支（左手）上有谱，- 支（右手）恒零
T3_SINGLET = P_PLUS @ T3 @ P_PLUS


def E(m, s):
    return np.eye(8, dtype=complex)[4 * m + s]


LOG = []


def rep(tag, cond, txt):
    LOG.append(f"[{'PASS' if cond else 'FAIL'}] {tag}: {txt}")
    print(LOG[-1])


def side_expect(psi, op, proj):
    w = proj @ psi
    n = float(np.vdot(w, w).real)
    if n < 1e-14:
        return None
    return float(np.vdot(w, op @ w).real / n)


def delta(psi, op):
    a = side_expect(psi, op, P_PLUS)
    b = side_expect(psi, op, P_MINUS)
    if a is None or b is None:
        return None
    return a - b


def mu_singlet(M):
    """单态度：右手侧块范数占全范数之比。0 为右手完全不耦合。"""
    den = np.linalg.norm(M)
    if den < 1e-14:
        return None
    return float(np.linalg.norm(P_MINUS @ M @ P_MINUS) / den)


def purity_modal(psi):
    v = psi.reshape(2, 4)
    rho = v @ v.conj().T
    return float(np.trace(rho @ rho).real)


def main():
    print("=== 判据第 677 号：单态条件与两侧不对称的区分 ===")

    # ---------------- M1：两个算符的基本性质
    r1 = float(np.linalg.norm(T3 @ GAMMA5 - GAMMA5 @ T3))
    r2 = float(np.linalg.norm(T3_SINGLET @ GAMMA5 - GAMMA5 @ T3_SINGLET))
    rep("M1", r1 < 1e-14 and r2 < 1e-14,
        f"两个弱荷都与手征对易（{r1:.1e}、{r2:.1e}）：均为纯模态侧算符")

    # ---------------- M2：单态型算符的谱结构
    ev_full = np.linalg.eigvalsh(T3)
    ev_sing = np.linalg.eigvalsh(T3_SINGLET)
    zc_full = int(np.sum(np.abs(ev_full) < 1e-12))
    zc_sing = int(np.sum(np.abs(ev_sing) < 1e-12))
    rep("M2", zc_full == 0 and zc_sing == 4,
        f"sigma_z 型零本征值 {zc_full} 个、单态型 {zc_sing} 个（右手 4 维全零）："
        f"单态型把右手整块压掉")

    # ---------------- V1：两个条件的分离
    mu_t3 = mu_singlet(T3)
    mu_sing = mu_singlet(T3_SINGLET)
    # 不对称度：取一个两支都有权重的态
    psi = (E(0, 0) + E(1, 2)) / np.sqrt(2.0)
    d_t3 = delta(psi, T3)
    d_sing = delta(psi, T3_SINGLET)
    rep("V1", abs(mu_t3 - 1.0 / np.sqrt(2.0)) < 1e-12 and mu_sing < 1e-12,
        f"单态度：sigma_z 型 {mu_t3:.6f}（右手仍耦合），单态型 {mu_sing:.2e}（右手不耦合）；"
        f"同一态上不对称度 {d_t3:.6f} 对 {d_sing:.6f}：两个量各自独立")
    print("    判定：sigma_z 型的单态度为 1/sqrt(2) 而非 0，故「两侧不对称」不等于「右手单态」。")

    # ---------------- V2：单态度的独立性（扫描混合）
    # M(t) = cos(t) * sigma_z 型 + sin(t) * 单态型，看 mu 与 Delta 能否独立变化
    ts = np.linspace(0.0, np.pi / 2, 21)
    mus, dels = [], []
    for t in ts:
        M = np.cos(t) * T3 + np.sin(t) * T3_SINGLET
        mus.append(mu_singlet(M))
        dels.append(delta(psi, M))
    rep("V2", abs(mus[0] - 1.0 / np.sqrt(2.0)) < 1e-12 and mus[-1] < 1e-12
        and abs(dels[0] - 2.0) < 1e-12,
        f"混合扫描：单态度由 {mus[0]:.6f} 降到 {mus[-1]:.2e}，"
        f"不对称度由 {dels[0]:.6f} 变到 {dels[-1]:.6f}（并非同步降为零）")

    # ---------------- V3：配对关联度在单态条件下重测
    m0 = np.array([1, 0], dtype=complex)
    m1 = np.array([0, 1], dtype=complex)
    s_p = np.array([1, 0, 0, 0], dtype=complex)
    s_m = np.array([0, 0, 1, 0], dtype=complex)

    def paired_state(share):
        """配对份额 share：0 完全关联（0 配 +、1 配 -），0.5 打乱，1 反配。"""
        v = (np.sqrt(1.0 - share) * (np.kron(m0, s_p) + np.kron(m1, s_m))
             + np.sqrt(share) * (np.kron(m0, s_m) + np.kron(m1, s_p)))
        return v / np.linalg.norm(v)

    shares = [0.0, 0.25, 0.5, 0.75, 1.0]
    d_full = [delta(paired_state(s), T3) for s in shares]
    d_sing = [delta(paired_state(s), T3_SINGLET) for s in shares]
    exp_full = [2.0 - 4.0 * s for s in shares]
    exp_sing = [1.0 - 2.0 * s for s in shares]
    e1 = max(abs(a - b) for a, b in zip(d_full, exp_full))
    e2 = max(abs(a - b) for a, b in zip(d_sing, exp_sing))
    rep("V3", e1 < 1e-12 and e2 < 1e-12,
        f"sigma_z 型 $\\Delta$ = {[round(x, 4) for x in d_full]}（解析 $2-4s$，偏差 {e1:.1e}）；"
        f"单态型 $\\Delta$ = {[round(x, 4) for x in d_sing]}（解析 $1-2s$，偏差 {e2:.1e}）")
    print("    判定：配对关联度的函数形式不变（线性、过零点在 s=1/2），但斜率减半，"
          "因为单态型的右手期望恒为零，可用幅度只剩左手一半。")

    # ---------------- V4：标准表述的正确读法
    # 「SU(2)_L 只作用左手」要求 mu = 0；此条件比 Delta != 0 强。
    # 给一个 Delta = 2 但 mu = 1/sqrt2 的反例，说明弱条件不能替代强条件。
    rep("V4", mu_t3 > 0.5 and abs(d_t3 - 2.0) < 1e-12,
        f"反例成立：sigma_z 型有满值不对称度（$\\Delta$ = {d_t3:.6f}）而单态度 {mu_t3:.6f}，"
        f"故「只作用左手」不能由「两侧不对称」推出")

    # ---------------- R1：右手不耦合且左手也不耦合
    M_zero = np.zeros((8, 8), dtype=complex)
    mu_zero = mu_singlet(M_zero)
    rep("R1", mu_zero is None,
        f"零算符：单态度无定义（{mu_zero}），须显式排除，否则 0/0")

    # R1b：只有左手耦合（真正的单态）
    only_left = P_PLUS @ T3 @ P_PLUS
    mu_ol = mu_singlet(only_left)
    d_ol = delta(psi, only_left)
    rep("R1", mu_ol < 1e-12 and abs(d_ol - 1.0) < 1e-12,
        f"只有左手耦合（单态）：单态度 {mu_ol:.2e}，不对称度 {d_ol:.6f}"
        f"（右手恒零故差等于左手期望）")

    # ---------------- R2：只有右手耦合
    only_right = P_MINUS @ T3 @ P_MINUS
    mu_or = mu_singlet(only_right)
    d_or = delta(psi, only_right)
    rep("R2", abs(mu_or - 1.0) < 1e-12 and abs(d_or - 1.0) < 1e-12,
        f"只有右手耦合：单态度 {mu_or:.6f}（满值），不对称度 {d_or:.6f}"
        f"（= 0 减右手期望 -1，故为正）：单态度能分辨耦合住在哪一侧，"
        f"不对称度只能给差的符号，与住哪侧无关")

    # ---------------- 图
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.5))

    ax = axes[0]
    ax.plot(ts, mus, "o-", color="#1f77b4", ms=3, label=r"单态度 $\mu$")
    ax.set_xlabel(r"混合参数 $t$（$0$ 为 $\sigma_z$ 型，$\pi/2$ 为单态型）")
    ax.set_ylabel(r"单态度 $\mu$", color="#1f77b4")
    ax.tick_params(axis="y", labelcolor="#1f77b4")
    ax.set_ylim(-0.05, 1.05)
    ax2 = ax.twinx()
    ax2.plot(ts, dels, "s-", color="#d62728", ms=3, label=r"不对称度 $\Delta$")
    ax2.set_ylabel(r"不对称度 $\Delta$", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax.set_title("(a) 单态度与不对称度不同步")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(shares, d_full, "o-", color="#2a6fdb", ms=4, label=r"$\sigma_z$ 型：$2-4s$")
    ax.plot(shares, d_sing, "s-", color="#3a9d5d", ms=4, label=r"单态型：$1-2s$")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("配对打乱份额 $s$")
    ax.set_ylabel(r"$\Delta$")
    ax.set_title("(b) 配对关联度在两型下的形式")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    labels = [r"$\sigma_z$ 型", r"单态型", "只有左手", "只有右手"]
    mus_bar = [mu_t3, mu_sing, mu_ol, mu_or]
    ax.bar(range(4), mus_bar, color=["#7f7f7f", "#3a9d5d", "#2a6fdb", "#d1603d"])
    ax.axhline(0.5, color="k", ls="--", lw=1)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(r"单态度 $\mu$")
    for i, v in enumerate(mus_bar):
        ax.text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_title("(c) 单态度判定耦合住在哪一侧")
    ax.set_ylim(0, 1.18)

    fig.tight_layout()
    out = "D:/software/TianXingFC/images/fig_singlet_condition.png"
    fig.savefig(out, dpi=160)
    print(f"\n图已保存：{out}")

    n_pass = sum(1 for line in LOG if line.startswith("[PASS]"))
    print(f"\n判据通过 {n_pass}/{len(LOG)}")


if __name__ == "__main__":
    main()
