# -*- coding: utf-8 -*-
"""判据第 664 号：跨侧桥的判据化。

张量积框架下「单侧加载」的不可能性与纠缠出路的检验。

背景：第 663 号读到模态侧对 CR 分支中性（弱荷 T_k 全在保持类），故「弱作用只作用
左手」这份偏心必须在桥那一层找。本轮把桥的形式做成判据：在 8 分量结构
（模态 2 维 × 时空 4 维）上，检验「弱荷在两手征扇区的期望是否可能不同」。

判据：冒烟 M1 至 M2、B1 至 B5，另附对照组与多档扫描。
可逆性等级：判据稳健（解析恒等式与条件期望读数）。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

RNG = np.random.default_rng(20261017)
IMG = r"D:\software\TianXingFC\images\fig_cross_bridge_single_side.png"

SZ = np.diag([1.0, -1.0]).astype(complex)
G5 = np.diag([1.0, 1.0, -1.0, -1.0]).astype(complex)

I2, I4 = np.eye(2, dtype=complex), np.eye(4, dtype=complex)
T3 = np.kron(SZ, I4)                       # 纯模态侧弱荷
GAMMA5 = np.kron(I2, G5)                   # 纯时空侧手征
P_PLUS = np.kron(I2, np.diag([1.0, 1.0, 0.0, 0.0]).astype(complex))
P_MINUS = np.kron(I2, np.diag([0.0, 0.0, 1.0, 1.0]).astype(complex))
COUPLED = T3 @ GAMMA5                      # 对照：手征-模态耦合算符

LOG = []


def rep(tag, text, ok):
    LOG.append(f"[{'PASS' if ok else 'FAIL'}] {tag}: {text}")
    print(LOG[-1])


def rand_state(n):
    v = RNG.normal(size=n) + 1j * RNG.normal(size=n)
    return v / np.linalg.norm(v)


def side_expect(psi, op, proj):
    """扇区 proj 内 op 的条件期望；扇区权重为零时返回 None。"""
    w = proj @ psi
    n = np.vdot(w, w).real
    if n < 1e-14:
        return None
    return float(np.vdot(w, op @ w).real / n)


def side_gap(psi, op):
    a = side_expect(psi, op, P_PLUS)
    b = side_expect(psi, op, P_MINUS)
    if a is None or b is None:
        return None
    return abs(a - b)


def entangled(theta):
    """cos(th)|0>_m|+>_s + sin(th)|1>_m|->_s（|+> 与 |-> 为 gamma5 本征态）。"""
    v = np.zeros(8, dtype=complex)
    v[0] = np.cos(theta)
    v[6] = np.sin(theta)
    return v


def main():
    print("=== 判据第 664 号：跨侧桥的判据化 ===")

    # ---------------- 冒烟
    c = float(np.max(np.abs(T3 @ GAMMA5 - GAMMA5 @ T3)))
    rep("M1", f"[T_3, Gamma_5] 的范数 {c:.2e}（期望 0，复现第 661 号）", c < 1e-14)
    dims = []
    for mp in (P_PLUS, P_MINUS):
        for mz in (np.kron(np.diag([1.0, 0.0]), I4), np.kron(np.diag([0.0, 1.0]), I4)):
            P = mp @ mz
            dims.append(int(round(np.trace(P).real)))
    rep("M2", f"四扇区维数 {dims}（期望各 2）", dims == [2, 2, 2, 2])

    # ---------------- B1：张量积态下两侧期望相同
    gaps = []
    for _ in range(200):
        psi = np.kron(rand_state(2), rand_state(4))
        g = side_gap(psi, T3)
        if g is not None:
            gaps.append(g)
    rep("B1", f"张量积态（{len(gaps)} 个）上 T_3 的两侧期望差最大 {max(gaps):.2e}"
              f"（期望 0：纯模态侧算符不区分手征扇区）", max(gaps) < 1e-13)

    # ---------------- B2：纠缠态下两侧期望可以不同
    th = np.linspace(1e-6, np.pi / 2 - 1e-6, 200)
    gaps_e = [side_gap(entangled(t), T3) for t in th]
    max_gap = max(g for g in gaps_e if g is not None)
    rep("B2", f"纠缠态族上 T_3 的两侧期望差最大 {max_gap:.6f}（期望非零，上界 2）",
        max_gap > 1.9)

    # ---------------- B3：对照，耦合算符在张量积态下也能分开
    gaps_c = []
    for _ in range(200):
        psi = np.kron(rand_state(2), rand_state(4))
        g = side_gap(psi, COUPLED)
        if g is not None:
            gaps_c.append(g)
    rep("B3", f"对照：耦合算符 T_3 Gamma_5 在张量积态上两侧期望差最大 {max(gaps_c):.6f}"
              f"（期望非零：它不是纯模态侧算符）", max(gaps_c) > 0.1)

    # ---------------- B4：纠缠度与期望差的关系
    th2 = np.linspace(0.0, np.pi / 2, 41)
    curve = []
    for t in th2:
        g = side_gap(entangled(t), T3)
        curve.append(np.nan if g is None else g)
    finite = [x for x in curve if not np.isnan(x)]
    rep("B4", f"纠缠参数扫描：可读档 {len(finite)}/{len(curve)}，期望差由近 0 升到 "
              f"{max(finite):.6f}，单调性 {'成立' if np.all(np.diff(finite) >= -1e-9) else '不成立'}",
        len(finite) >= 38 and max(finite) > 1.9)

    # ---------------- B5：恒真检验与反例
    n_taut = 0
    for name, op in (("T_3", T3), ("T_3 Gamma_5", COUPLED)):
        if np.max(np.abs(op - np.eye(8))) < 1e-12:
            n_taut += 1
    rep("B5", f"恒真检验：两项算符中与恒等重合者 {n_taut} 项（期望 0）", n_taut == 0)
    psi_prod = np.kron(np.array([1, 0], dtype=complex), np.array([1, 0, 0, 0], dtype=complex))
    rep("B5", f"反例：乘积态 |0>_m|+>_s 上 T_3 的 -支权重为零（读数为 "
              f"{side_expect(psi_prod, T3, P_MINUS)}），说明判据须取两侧权重都非零的态",
        side_expect(psi_prod, T3, P_MINUS) is None)

    # ---------------- 图
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))

    ax = axes[0]
    xs, ys = [], []
    for _ in range(120):
        psi = np.kron(rand_state(2), rand_state(4))
        a = side_expect(psi, T3, P_PLUS)
        b = side_expect(psi, T3, P_MINUS)
        if a is not None and b is not None:
            xs.append(a)
            ys.append(b)
    ax.plot([-1.05, 1.05], [-1.05, 1.05], "k--", lw=1, label="相等线")
    ax.scatter(xs, ys, s=12, color="#1f77b4", alpha=0.7)
    ax.set_xlabel(r"$+$ 支内 $\langle T_3\rangle$")
    ax.set_ylabel(r"$-$ 支内 $\langle T_3\rangle$")
    ax.set_title("B1：张量积态两侧相等")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(th2, curve, "o-", color="#d62728", ms=3)
    ax.set_xlabel(r"纠缠参数 $\theta$")
    ax.set_ylabel(r"两侧 $\langle T_3\rangle$ 之差")
    ax.set_title("B4：纠缠度与单侧性")
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.bar([0, 1], [max(gaps), max(gaps_c)], color=["#2ca02c", "#ff7f0e"])
    ax.set_xticks([0, 1])
    ax.set_xticklabels([r"纯弱荷 $T_3$", r"耦合算符 $T_3\Gamma_5$"])
    ax.set_ylabel("张量积态上两侧期望差最大值")
    ax.set_title("B1/B3：纯模态侧算符与耦合算符")
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(IMG, dpi=150)
    print(f"\n图已存：{IMG}")
    n_fail = sum(1 for x in LOG if x.startswith("[FAIL]"))
    print(f"\n汇总：{len(LOG) - n_fail}/{len(LOG)} 通过")


if __name__ == "__main__":
    main()
