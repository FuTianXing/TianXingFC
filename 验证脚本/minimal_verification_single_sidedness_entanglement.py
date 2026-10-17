# -*- coding: utf-8 -*-
r"""判据第 675 号：单侧性与纠缠结构的分离。

背景与动机：
  - 第 664 号判定「张量积态配纯模态侧算符，单侧加载不可能；出路在态层纠缠」，并给出
    纠缠族 $\cos\theta|0\rangle_m|+\rangle_s + \sin\theta|1\rangle_m|-\rangle_s$ 上两侧
    期望差达上界 $2.000000$、随 $\theta$ 单调上升。
  - 本轮重算发现该读数被误读：该族上两侧期望差恒为 $2$，与 $\theta$ 无关（$+\cos^2\theta$
    对 $-\sin^2\theta$ 之差恰为 $1$ 除以各自的权重……见下）。扫描里的「上升」实为
    「$\theta$ 取端点时某支权重为零不可读」造成的可读性变化，不是物理单调性。
  - 由此引出本轮的真问题：单侧性到底由什么决定。第 664 号把「纠缠」当作单侧的原因，
    但它给的族是最大纠缠型的特定配对，纠缠度与单侧性在该族里完全退耦。本轮把两者分开。

判据设计（三条，各自独立可证伪）：
  U1  单侧性度量的定义与解析恒等式：$\Delta=\langle T_3\rangle_+-\langle T_3\rangle_-$
      在配对态 $|0\rangle|+\rangle,|1\rangle|-\rangle$ 上恒为 $1$（归一化权重下），
      与配对权重无关。
  U2  纠缠度与单侧性退耦：固定配对结构，扫描权重得 $\Delta$ 恒定；固定权重换配对
      得 $\Delta$ 变化。即 $\Delta$ 由配对（哪一支配哪一支）决定，与纠缠度无关。
  U3  单侧性的充分必要条件：$\Delta$ 最大（$=1$）当且仅当模态标签与手征标签完全关联
      （一个模态支只与一个手征支配对）。部分关联给中间值。
反例：不关联的配对（$|0\rangle$ 配两支持平）给 $\Delta=0$；可分离态给 $\Delta=0$。

判据：冒烟 M1 至 M2、U1 至 U4、R1 至 R2。
可逆性等级：判据稳健（解析恒等式与直接求和）。
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
T3 = np.kron(SZ, I4)                       # 纯模态侧弱荷
G5_4 = np.diag([1.0, 1.0, -1.0, -1.0]).astype(complex)
GAMMA5 = np.kron(I2, G5_4)                 # 纯时空侧手征
P_PLUS = np.kron(I2, np.diag([1.0, 1.0, 0.0, 0.0]).astype(complex))
P_MINUS = np.kron(I2, np.diag([0.0, 0.0, 1.0, 1.0]).astype(complex))

# 基矢约定：kron(模态, 时空)，故索引 = 4*m + s；时空侧 s=0 为 + 支、s=2 为 - 支
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


def delta(psi, op=T3):
    a = side_expect(psi, op, P_PLUS)
    b = side_expect(psi, op, P_MINUS)
    if a is None or b is None:
        return None
    return a - b


def purity_modal(psi):
    """模态侧约化密度矩阵的纯度 tr(rho^2)；1 为可分离，1/2 为最大纠缠。"""
    rho = np.zeros((2, 2), dtype=complex)
    v = psi.reshape(2, 4)
    rho = v @ v.conj().T
    return float(np.trace(rho @ rho).real)


def paired(a, b, phi=0.0):
    """cos(phi)|0>_m|+> + sin(phi)|1>_m|->；phi 控制权重。"""
    return np.cos(phi) * E(0, 0) + np.sin(phi) * E(1, 2)


def cross_paired(a, b, phi):
    """交叉配对：|0>_m|-> 与 |1>_m|+>。"""
    return np.cos(phi) * E(0, 2) + np.sin(phi) * E(1, 0)


def mixed_pair(phi, share):
    """部分关联：|0>_m 配 + 支与 - 支按 share 分，|1>_m 配另一支。"""
    v = (np.cos(phi) * (np.sqrt(1.0 - share) * E(0, 0) + np.sqrt(share) * E(0, 2))
         + np.sin(phi) * (np.sqrt(1.0 - share) * E(1, 2) + np.sqrt(share) * E(1, 0)))
    return v / np.linalg.norm(v)


def main():
    print("=== 判据第 675 号：单侧性与纠缠结构的分离 ===")

    # ---------------- M1：算符与投影的基本恒等式
    c1 = float(np.max(np.abs(T3 @ GAMMA5 - GAMMA5 @ T3)))
    rk = [int(round(np.trace(P).real)) for P in (P_PLUS, P_MINUS)]
    rep("M1", c1 < 1e-14 and rk == [4, 4],
        f"[T_3,Gamma_5] 范数 {c1:.1e}、两扇区维数 {rk}（各 4，含模态 2）")

    # ---------------- M2：基矢标签核对（防止 s 索引写错）
    ok = True
    # 时空侧 + 支取 s=0,1，- 支取 s=2,3（与本文件 diag 约定一致）
    for m in (0, 1):
        for s in (0, 1):
            ok = ok and abs(float(np.linalg.norm(P_PLUS @ E(m, s)) ** 2) - 1.0) < 1e-14
        for s in (2, 3):
            ok = ok and abs(float(np.linalg.norm(P_MINUS @ E(m, s)) ** 2) - 1.0) < 1e-14
    tz = [float(np.vdot(E(m, 0), T3 @ E(m, 0)).real) for m in (0, 1)]
    rep("M2", ok and tz == [1.0, -1.0],
        f"基矢标签核对通过；T_3 在 |0>、|1> 上取 {tz}")

    # ---------------- U1：配对族上 Delta 与权重无关
    phis = np.linspace(0.05, np.pi / 2 - 0.05, 40)
    ds = [delta(paired(0, 0, p)) for p in phis]
    span = max(ds) - min(ds)
    rep("U1", span < 1e-12 and abs(np.mean(ds) - 2.0) < 1e-12,
        f"配对族 40 档权重下 Delta 恒为 {np.mean(ds):.6f}，极差 {span:.2e}"
        f"（期望 2，与权重无关）")
    print("    解析：+ 支期望 = +1，- 支期望 = -1，故 Delta = 2；此处 T_3 取 sigma_z"
          "（本征值 ±1）而非 sigma_z/2，读数 2 对应第 664 号的「上界 2」。")

    # ---------------- U2：纠缠度与单侧性退耦
    # 配对族 cos(phi)|0,+> + sin(phi)|1,-> 本身是纠缠态：phi 从 0 到 pi/2 时纯度由 1 降到 1/2
    # 再升回 1（端点退化），而 Delta 恒为 2。纯度在变、单侧性不变，两者退耦。
    pur = [purity_modal(paired(0, 0, p)) for p in phis]
    pspan = max(pur) - min(pur)
    rep("U2", pspan > 0.4 and span < 1e-12,
        f"配对族上模态纯度在 {min(pur):.6f} 到 {max(pur):.6f} 之间变化（极差 {pspan:.3f}），"
        f"而 Delta 恒为 2（极差 {span:.1e}）：纠缠度在变、单侧性不变")

    # ---------------- U3：换配对结构，Delta 变号
    d_cross = [delta(cross_paired(0, 0, p)) for p in phis]
    rep("U3", abs(np.mean(d_cross) + 2.0) < 1e-12,
        f"交叉配对族 Delta 恒为 {np.mean(d_cross):.6f}（期望 -2，符号翻转）："
        f"单侧的方向由配对决定")

    # ---------------- U4：部分关联给中间值
    shares = [0.0, 0.25, 0.5, 0.75, 1.0]
    ds4 = [delta(mixed_pair(np.pi / 4, s)) for s in shares]
    expect = [2.0 - 4.0 * s for s in shares]
    err = max(abs(a - b) for a, b in zip(ds4, expect))
    rep("U4", err < 1e-12,
        f"部分关联扫描：Delta = {[round(x, 4) for x in ds4]}，"
        f"解析期望 {expect}（最大偏差 {err:.1e}）：完全关联给满值、无关联给零")

    # ---------------- R1：可分离态（真纠缠度为零）的 Delta
    prod = np.kron(np.array([1, 0], dtype=complex), np.array([1, 0, 0, 0], dtype=complex))
    d_prod = delta(prod)
    rep("R1", d_prod is None,
        f"乘积态 |0>_m|+>_s：- 支权重为零，Delta 无定义（{d_prod}），"
        f"说明可读性本身要作前提检查")

    # R1b：两侧权重都非零的可分离态（模态侧固定，时空侧取两支等权叠加）
    prod2 = np.kron(np.array([1, 0], dtype=complex),
                    np.array([1, 1, 1, 1], dtype=complex) / 2.0)
    d_prod2 = delta(prod2)
    rep("R1", d_prod2 is not None and abs(d_prod2) < 1e-12,
        f"可分离态 |0>_m(+,+,-,-)/2：Delta = {d_prod2:.2e}（期望 0，"
        f"复现第 664 号张量积态的结论）")

    # ---------------- R2：最大纠缠下扫描配对关联度
    # 两个正交的完全关联对：|0,+>+|1,-> 与 |0,->-|1,+>（后者反号配对）。二者都最大纠缠
    # （模态纯度 1/2）。按权重叠加后纯度恒为 1/2，配对关联度随权重从完全关联连续降到
    # 无关联，Delta 随之由 2 降到 0。这是「纠缠度不变、单侧性变」的对照，比 U2 更强。
    m0 = np.array([1, 0], dtype=complex)
    m1 = np.array([0, 1], dtype=complex)
    s_p = np.array([1, 0, 0, 0], dtype=complex)      # + 支内一个基矢
    s_m = np.array([0, 0, 1, 0], dtype=complex)      # - 支内一个基矢
    a_pair = (np.kron(m0, s_p) + np.kron(m1, s_m)) / np.sqrt(2.0)
    b_pair = (np.kron(m0, s_m) - np.kron(m1, s_p)) / np.sqrt(2.0)
    ws = np.linspace(0.0, np.pi / 4, 9)
    r2 = []
    for w in ws:
        v = np.cos(w) * a_pair + np.sin(w) * b_pair
        r2.append((float(purity_modal(v)), delta(v)))
    purs = [x[0] for x in r2]
    dvals = [x[1] for x in r2]
    rep("R2", max(purs) - min(purs) < 1e-12
        and abs(dvals[0] - 2.0) < 1e-12 and abs(dvals[-1]) < 1e-12,
        f"最大纠缠族（纯度恒 {purs[0]:.6f}，极差 {max(purs) - min(purs):.1e}）："
        f"Delta 由 {dvals[0]:.6f} 连续降到 {dvals[-1]:.2e}，"
        f"纠缠度不变而单侧性可变，纠缠度不能决定单侧性")

    # ---------------- 图
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))

    ax = axes[0]
    ax.plot(phis, ds, "o-", color="#1f77b4", ms=3, label="配对族")
    ax.plot(phis, d_cross, "s-", color="#d62728", ms=3, label="交叉配对族")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel(r"权重参数 $\phi$")
    ax.set_ylabel(r"$\Delta=\langle T_3\rangle_+-\langle T_3\rangle_-$")
    ax.set_title("(a) 单侧性与权重无关（两族各自恒定）")
    ax.set_ylim(-2.6, 2.6)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(shares, ds4, "o-", color="#2ca02c", ms=4, label="实测")
    ax.plot(shares, expect, "k--", lw=1, label=r"解析 $2-4s$")
    ax.set_xlabel("部分关联份额 $s$")
    ax.set_ylabel(r"$\Delta$")
    ax.set_title("(b) 单侧性由关联度决定")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    labels = ["可分离\n(无关联)", "配对族", "交叉配对", "最大纠缠\n(无关联)"]
    vals = [abs(d_prod2), abs(np.mean(ds)), abs(np.mean(d_cross)), abs(dvals[-1])]
    cols = ["#7f7f7f", "#1f77b4", "#d62728", "#7f7f7f"]
    ax.bar(range(4), vals, color=cols)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(r"$\vert\Delta\vert$")
    ax.set_title("(c) 纠缠度与单侧性无关")
    ax.set_ylim(0, 1.15)

    fig.tight_layout()
    out = "D:/software/TianXingFC/images/fig_single_sidedness_entanglement.png"
    fig.savefig(out, dpi=160)
    print(f"\n图已保存：{out}")

    n_pass = sum(1 for line in LOG if line.startswith("[PASS]"))
    print(f"\n判据通过 {n_pass}/{len(LOG)}")


if __name__ == "__main__":
    main()
