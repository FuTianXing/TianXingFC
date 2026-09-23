# -*- coding: utf-8 -*-
r"""弹性遍历通道的相干代价：M2 缓降读数的换算、宏观约束与可逆性判别（判据 R1 至 R6）

第一部分：由 M2 三个无探测器协议的读数提取每格距代价，反解允许遍历距离与宏观阈值。
第二部分：可逆性判别。分支 {|A>,|B>} 与一个散射模式（截断 Fock），
          前向脉冲为分支 B 侧位移 D(alpha)，反向脉冲为 D(-alpha)。
          从属读法：模式保留，反向脉冲精确还原（V = 1）。
          独立读法：模式每次脉冲后被丢弃并重挂真空，反向脉冲无法还原。
确定性：无随机源。图存 D:/software/TianXingFC/images/traversal_decoherence_cost.png
"""
import numpy as np
from scipy.linalg import expm

RES = []
FIG = "D:/software/TianXingFC/images/traversal_decoherence_cost.png"
M2 = [("甲-1 零模踢", 6, 0.0025), ("甲-2 宽脉冲", 61, 0.0433), ("甲-3 踢+阻尼", 0, 1.5e-6)]
LP_M = 1.616255e-35
NF = 26


def report(tag, name, meas, exp, note="", grade="通过"):
    RES.append((tag, name, meas, exp, note, grade))


def displace(alpha, n):
    a = np.diag(np.sqrt(np.arange(1, n)), 1)
    return expm(alpha * a.T.conj() - np.conj(alpha) * a)


VAC = np.zeros(NF, dtype=complex); VAC[0] = 1.0
KA = np.kron(np.array([1.0, 0.0], dtype=complex), VAC)
KB = np.kron(np.array([0.0, 1.0], dtype=complex), VAC)


def tr_mode(rho):
    r = rho.reshape(2, NF, 2, NF)
    return np.einsum("aibi->ab", r)


def tr_branch(rho):
    r = rho.reshape(2, NF, 2, NF)
    return np.einsum("aiaj->ij", r)


def coh_branch(rho):
    """分支可见度 V = 2 |rho_AB| / (rho_AA + rho_BB)，分支约化态迹为一。"""
    rho_s = tr_mode(rho)
    return float(2.0 * abs(rho_s[0, 1]) / np.real(np.trace(rho_s)))


def U_branch_disp(vec, alpha):
    """U = |A><A| (x) I + |B><B| (x) D(alpha)，作用在 2*NF 维矢量上。"""
    out = np.zeros_like(vec)
    vA = vec[0:NF]; vB = vec[NF:2 * NF]
    out[0:NF] = vA
    out[NF:2 * NF] = displace(alpha, NF) @ vB
    return out


def Ud(alpha):
    """块对角算符 diag(I_NF, D(alpha))，对应 |A><A|(x)I + |B><B|(x)D(alpha)。"""
    M = np.zeros((2 * NF, 2 * NF), dtype=complex)
    M[0:NF, 0:NF] = np.eye(NF)
    M[NF:2 * NF, NF:2 * NF] = displace(alpha, NF)
    return M


def forward_and_inverse(alpha):
    """返回 (前向后的分支相干, 从属读法反向后的分支相干, 独立读法反向后的分支相干)。"""
    vacproj = np.outer(VAC, VAC.conj())
    # 前向脉冲
    psi = (KA + np.kron(np.array([0.0, 1.0]), displace(alpha, NF) @ VAC)) / np.sqrt(2.0)
    rho = np.outer(psi, psi.conj())
    v_fwd = coh_branch(rho)
    # 从属读法：模式保留，反向脉冲精确还原
    U = Ud(-alpha)
    v_sub = coh_branch(U @ rho @ U.conj().T)
    # 独立读法：脉冲后模式被丢弃并重挂真空，再做反向脉冲
    rho_s = tr_mode(rho)
    re_att = np.kron(rho_s, vacproj)
    v_ind = coh_branch(U @ re_att @ U.conj().T)
    return v_fwd, v_sub, v_ind


def part1():
    print("=" * 100)
    print("第一部分：代价律提取与宏观约束（R1、R2、R5）")
    print("=" * 100)
    print("协议 | 遍历格距 N | D_env | 每格距代价 delta | kappa = delta/2")
    deltas = []
    for name, N, D in M2:
        if N > 0:
            d = D / N; deltas.append(d)
            print("%-14s | %6d | %.4f | %.3e | %.3e" % (name, N, D, d, d / 2))
        else:
            print("%-14s | %6d | %.1e | 无遍历，不适用 | -" % (name, N, D))
    d_mean = float(np.mean(deltas)); kap = d_mean / 2.0
    print("两支均值：delta = %.3e，kappa = %.3e" % (d_mean, kap))
    report("R1", "两支每格距代价的相对差", float(abs(deltas[1] - deltas[0]) / d_mean), "-",
           "同量级（%.2e 与 %.2e）" % (deltas[0], deltas[1]), "读数")
    N_max = 0.2107 / d_mean
    report("R2", "V>=0.9 允许的遍历格距（若代价为真）", N_max, "-", "由 D<=0.2107 反解", "读数")
    print()
    print("宏观约束（若代价为真，delta <= -2 ln(V_min)/N）")
    print("-" * 100)
    print("路径  | 遍历格距 N = L/l_P | 允许的 delta | 允许的 kappa")
    rows = []
    for L, lab in [(1e-9, "1 nm"), (1e-6, "1 um"), (1e-3, "1 mm"), (1.0, "1 m")]:
        N = L / LP_M; dmax = 0.2107 / N; rows.append((lab, N, dmax))
        print("%-6s| %.3e | %.3e | %.3e" % (lab, N, dmax, dmax / 2))
    report("R5", "1 m 路径允许的每格距代价", rows[-1][2], "-",
           "对照 M2 推断的 %.2e" % d_mean, "读数")
    return d_mean, kap, N_max, rows


def part2():
    print()
    print("=" * 100)
    print("第二部分：可逆性判别（R3、R4）")
    print("=" * 100)
    print("alpha | V 前向（两种读法同值） | V 前向+反向（从属） | V 前向+反向（独立） | 解析 exp(-a^2/2) 与 exp(-a^2)")
    alphas = [0.2, 0.4, 0.6]
    w_sub, w_ind, w_ana = 0.0, 0.0, 0.0
    vs = []
    for a in alphas:
        v_fwd, v_sub, v_ind = forward_and_inverse(a)
        ana = np.exp(-a * a / 2.0); ana2 = np.exp(-a * a)
        w_sub = max(w_sub, abs(v_sub - 1.0))
        w_ind = max(w_ind, abs(v_ind - ana2))
        w_ana = max(w_ana, abs(v_fwd - ana))
        vs.append((a, v_fwd, v_sub, v_ind, ana, ana2))
        print("%.2f  | %.6f | %.12f | %.6f | %.6f / %.6f" % (a, v_fwd, v_sub, v_ind, ana, ana2))
    report("R3", "从属读法反向后的可见度对 1 的最大偏差", w_sub, 1e-8,
           "三档 alpha；模式保留", "通过" if w_sub < 1e-8 else "未过")
    report("R3", "独立读法反向后的可见度对 exp(-a^2) 的最大偏差", w_ind, 1e-10,
           "三档 alpha；不恢复", "通过" if w_ind < 1e-10 else "未过")
    report("R4", "前向可见度对 exp(-a^2/2) 的最大偏差", w_ana, 1e-10,
           "瞬时可见度不能判别两种读法", "通过" if w_ana < 1e-10 else "未过")
    return vs


def figure(d_mean, N_max, rows, vs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.0))

    Ns = np.array([m[1] for m in M2 if m[1] > 0], float)
    Ds = np.array([m[2] for m in M2 if m[1] > 0], float)
    ax[0, 0].loglog(Ns, Ds, "o", ms=8, label="M2 读数（甲-1、甲-2）")
    ax[0, 0].loglog(Ns, d_mean * Ns, "--", label="线性律 delta=%.2e/格距" % d_mean)
    ax[0, 0].axhline(1.5e-6, color="gray", lw=1.0, label="甲-3（无遍历）")
    ax[0, 0].set_xlabel("遍历格距 N"); ax[0, 0].set_ylabel("D_env")
    ax[0, 0].set_title("R1：代价随遍历距离累积"); ax[0, 0].legend(fontsize=8)

    Nv = np.logspace(0, 5.6, 200)
    ax[0, 1].loglog(Nv, 0.2107 / Nv, "-", label="V>=0.9 允许上限")
    ax[0, 1].loglog([N_max], [d_mean], "r*", ms=14, label="M2 推断 delta")
    ax[0, 1].set_xlabel("遍历格距 N"); ax[0, 1].set_ylabel("允许的每格距代价 delta")
    ax[0, 1].set_title("R2：允许代价随遍历距离下降"); ax[0, 1].legend(fontsize=8)

    labs = [q[0] for q in rows]; dm = [q[2] for q in rows]
    ax[1, 0].semilogy(labs, dm, "o-", label="宏观所需 delta 上限")
    ax[1, 0].axhline(d_mean, color="r", ls="--", label="M2 推断 delta = %.2e" % d_mean)
    ax[1, 0].set_ylabel("每格距代价 delta"); ax[1, 0].set_title("R5：宏观约束与 M2 推断的落差")
    ax[1, 0].legend(fontsize=8)

    w = 0.2; idx = np.arange(len(vs))
    ax[1, 1].bar(idx - w, [q[1] for q in vs], w, label="仅前向（两读法同值）")
    ax[1, 1].bar(idx, [q[2] for q in vs], w, label="前向+反向（从属）")
    ax[1, 1].bar(idx + w, [q[3] for q in vs], w, label="前向+反向（独立）")
    ax[1, 1].set_xticks(idx); ax[1, 1].set_xticklabels(["alpha=%.1f" % q[0] for q in vs])
    ax[1, 1].set_ylim(0, 1.15); ax[1, 1].set_ylabel("分支相干 |rho_AB|")
    ax[1, 1].set_title("R3/R4：可逆性判别"); ax[1, 1].legend(fontsize=8)

    fig.tight_layout(); fig.savefig(FIG, dpi=140)
    print("图已保存：", FIG)


def main():
    d_mean, kap, N_max, rows = part1()
    vs = part2()
    print()
    print("=" * 100)
    print("判据 | 判据量 | 实测 | 期望 | 备注 | 结论")
    print("=" * 100)
    for tag, name, meas, exp, note, grade in RES:
        m = ("%.6g" % meas) if isinstance(meas, float) else str(meas)
        e = ("%.6g" % exp) if isinstance(exp, float) else str(exp)
        print("%-5s | %s | %s | %s | %s | %s" % (tag, name, m, e, note, grade))
    print("=" * 100)
    figure(d_mean, N_max, rows, vs)


if __name__ == "__main__":
    main()
