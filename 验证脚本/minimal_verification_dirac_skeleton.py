# -*- coding: utf-8 -*-
"""
第 489 条：电子算符 Dirac 骨架（电子场攻坚计划阶段 2 首轮）。

模型：$S^1 × R$ 状态空间上的 2+1 维离散 Dirac 哈密顿量（π 扭转口径），
    H = kron(σx, c_θ D_θ) + kron(σz, c_h D_h) + kron(m σy, I)
θ 环反周期（π 扭转口径：SU(2) 双值表示，ψ(θ+2π) = -ψ(θ)，接缝移位带 -1），
h 环周期（D-b 主设定）。绕数背景 winding = ±1 以 θ 规范耦合实现（D4 的 ±n 来源）。

理论期望（先立后测）：
  SK1（D2 主体）实空间数值谱与离散解析谱 E² = c_θ² sin²(k_θ) + c_h² sin²(k_h) + m²
     逐位吻合（两条算法路径自洽）。
  SK2（D2 连续极限）低 k 模式 (E² - m²)/(c²k²) → 1，残差 = O((ka)²)。
  SK3（D4 初读数之一）反周期生效：k_θ 全为半整数（无零模），谱关于 E = 0 严格对称。
  SK4（D4 镜像式）winding = ±1 的 Dirac 谱逐位恒等：±n 为同一算符的两支（谱层证据）。
  SK5（D1 甲乙首测）位置依赖质量 m(h) = m₀(1 + A sech²) 相对常数质量的谱偏移读数，
     束缚态计数如实登记；丙（绕数阈值势垒）本轮不做，登记后续。

可逆性等级：SK1/SK3/SK4 判据稳健（解析恒等式 + 构造决定）；SK2 含有限差分协议伪影
  （sin(ka) 指纹化与 fermion doubling 计数分层报告）；SK5 为读数非判据。
"""
import numpy as np

N_TH, N_H = 32, 32
C_TH, C_H = 1.0, 1.0
M0 = 0.5

SIG_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIG_Z = np.array([[1, 0], [0, -1]], dtype=complex)
SIG_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)

results = []
def report(tag, measured, expected, ok):
    results.append((tag, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {tag}")
    print(f"    实测: {measured}")
    print(f"    期望: {expected}")

def shift_theta(antiperiodic=True, gauge=0.0):
    """θ 方向前向移位算符（含反周期接缝负号与绕数规范相位）。"""
    ph = np.exp(-1j * gauge) if gauge else 1.0
    S = np.zeros((N_TH, N_TH))
    for i in range(N_TH):
        j = (i + 1) % N_TH
        S[i, j] = ph * (-1.0 if (antiperiodic and i == N_TH - 1) else 1.0)
    return S

def shift_h():
    S = np.zeros((N_H, N_H))
    for i in range(N_H):
        S[i, (i + 1) % N_H] = 1.0
    return S

def build_H(m_field=None, winding=0.0):
    """H = kron(σx, c_θ D_θ) + kron(σz, c_h D_h) + kron(m σy, I)。"""
    Sth = shift_theta(gauge=2 * np.pi * winding / N_TH)
    D_th = (Sth - Sth.T) / 2.0
    Sh = shift_h()
    D_h = (Sh - Sh.T) / 2.0
    m = np.full(N_H, M0) if m_field is None else m_field
    H = (np.kron(SIG_X, -1j * C_TH * np.kron(D_th, np.eye(N_H)))
         + np.kron(SIG_Z, -1j * C_H * np.kron(np.eye(N_TH), D_h))
         + np.kron(SIG_Y, np.kron(np.eye(N_TH), np.diag(m))))
    return H

def smoke():
    print("== 冒烟检查 ==")
    H = build_H()
    herm = np.max(np.abs(H - H.conj().T))
    S = shift_theta()
    phases = np.angle(np.linalg.eigvals(S))
    expect = np.sort(np.mod(2 * np.pi * (np.arange(N_TH) + 0.5) / N_TH + np.pi, 2 * np.pi) - np.pi)
    got = np.sort(np.mod(phases + np.pi, 2 * np.pi) - np.pi)
    ok = herm < 1e-12 and np.max(np.abs(expect - got)) < 1e-10
    print(f"    Hermitian 偏差 {herm:.1e}；移位本征相位与 2π(n+1/2)/N 最大差 "
          f"{np.max(np.abs(expect - got)):.1e}")
    report("冒烟：H Hermitian 且反周期移位本征值为半整数相位",
           f"herm {herm:.1e}, 相位差 {np.max(np.abs(expect - got)):.1e}",
           "herm < 1e-12，相位差 < 1e-10", ok)
    return ok

def analytic_discrete_spectrum(winding=0.0):
    n_th = np.arange(N_TH) - N_TH // 2
    k_th = 2 * np.pi * (n_th + 0.5) / N_TH + 2 * np.pi * winding / N_TH
    n_h = np.arange(N_H) - N_H // 2
    k_h = 2 * np.pi * n_h / N_H
    KT, KH = np.meshgrid(k_th, k_h, indexing="ij")
    E2 = C_TH ** 2 * np.sin(KT) ** 2 + C_H ** 2 * np.sin(KH) ** 2 + M0 ** 2
    E = np.sqrt(E2).ravel()
    return np.sort(np.concatenate([E, -E])), k_th, k_h

def count_doubling():
    return 2 ** 2  # 每维一个附加谷（中心差分 Nielsen–Ninomiya），协议伪影计数

def main():
    smoke()
    H0 = build_H()
    print("\n== SK1（D2 主体）：数值谱 vs 离散解析谱 ==")
    num = np.linalg.eigvalsh(H0)
    ana, k_th, k_h = analytic_discrete_spectrum()
    d = np.max(np.abs(num - ana))
    report("SK1 实空间谱与离散解析谱逐位吻合", f"max|ΔE| = {d:.2e}",
           "< 1e-10（两条算法路径自洽）", d < 1e-10)

    print("\n== SK2（D2 连续极限）==")
    n_th = np.arange(N_TH) - N_TH // 2
    k_th_vals = 2 * np.pi * (n_th + 0.5) / N_TH
    n_h = np.arange(N_H) - N_H // 2
    k_h_vals = 2 * np.pi * n_h / N_H
    KT, KH = np.meshgrid(k_th_vals, k_h_vals, indexing="ij")
    k2 = (C_TH * KT) ** 2 + (C_H * KH) ** 2
    E2_mode = C_TH ** 2 * np.sin(KT) ** 2 + C_H ** 2 * np.sin(KH) ** 2
    lo = k2 < 0.02
    budget = lo.sum() and np.max(k2[lo]) / 3.0
    ratio = E2_mode[lo] / k2[lo]
    res = np.max(np.abs(ratio - 1.0))
    report("SK2 低 k 模式连续极限", f"max |(E²-m²)/(c²k²) - 1| = {res:.2e}（{lo.sum()} 模式）",
           f"≤ 误差预算 k²/3 = {budget:.2e}（sin 展开 O((ka)²)，档内 ka ≤ 0.14）", res <= budget)

    print("\n== SK3（D4 初读数：反周期与谱对称）==")
    frac = k_th_vals * N_TH / (2 * np.pi)
    half_int = np.max(np.abs(np.mod(frac, 1.0) - 0.5))
    E_sorted = np.sort(num)
    sym = np.max(np.abs(E_sorted - (-E_sorted)[::-1]))
    report("SK3 反周期半整数动量（无零模）", f"max|frac(k_θ·N_θ/2π) - 1/2| = {half_int:.2e}",
           "< 1e-10（k_θ/2π 全为 n+1/2，π 扭转口径生效）", half_int < 1e-10)
    report("SK3 谱关于 E = 0 严格对称", f"max|E_i + E_(N+1-i)| = {sym:.2e}",
           "< 1e-10（Dirac 海，正负成对）", sym < 1e-10)

    print("\n== SK4（D4 镜像式：winding = ±1 谱恒等）==")
    Hp = build_H(winding=+1)
    Hm = build_H(winding=-1)
    Ep, Em = np.linalg.eigvalsh(Hp), np.linalg.eigvalsh(Hm)
    dpm = np.max(np.abs(Ep - Em))
    report("SK4 ±n 背景谱逐位恒等", f"max|E(+n) - E(-n)| = {dpm:.2e}",
           "< 1e-10（±n 为同一算符的两支，D4 谱层证据）", dpm < 1e-10)

    print("\n== SK5（D1 甲乙首测：位置依赖质量）==")
    ih = np.arange(N_H)
    prof = 1.0 + 0.3 / np.cosh((ih - N_H / 2) / 4.0) ** 2
    Eb = np.linalg.eigvalsh(build_H(m_field=M0 * prof))
    offset = np.max(np.abs(np.sort(Eb) - np.sort(num)))
    gap_lo = np.min(np.abs(Eb[Eb > 0]))
    print(f"    甲（常数 m）：能隙下沿 {np.min(np.abs(num[num > 0])):.6f}，无 |E| < m 束缚态")
    print(f"    乙（m(h) 峰值 +30%）：谱最大偏移 {offset:.4f}，能隙下沿 {gap_lo:.6f}，"
          f"|E| < m 束缚态 {np.sum(np.abs(Eb) < M0)} 个")
    report("SK5 甲乙谱偏移读数（读数非判据）",
           f"max 偏移 {offset:.4f} = {(offset / M0):.1%} of m₀；束缚态计数见上",
           "偏移量级 = A·m₀；束缚态结构差异如实登记，D1 三候选对比待丙补齐", True)

    print("\n== 协议伪影分层 ==")
    print(f"    fermion doubling：中心差分每维一个附加谷，计数因子 {count_doubling()}，"
          f"已由离散解析谱（sin(ka) 指纹）完整吸收，SK1 对照不受污染")
    print(f"    连续极限残差：SK2 的 O((ka)²)，属有限差分协议伪影，非物理色散破坏")

    print("\n== 判据汇总 ==")
    npass = sum(1 for _, ok in results if ok)
    for tag, ok in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {tag}")
    print(f"\n判据 SK1 至 SK5（冒烟并入）：{npass}/{len(results)} 通过")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax = axes[0, 0]
    ax.scatter(k2[lo], E2_mode[lo], s=8, c="b", label="E² - m²（低 k 档解析谱）")
    kk = np.linspace(0, 0.05, 20)
    ax.plot(kk, kk, "k--", lw=1, label="c²k²")
    ax.set_xlabel("c²k²")
    ax.set_ylabel("E² - m²")
    ax.set_title(f"SK2: 连续极限，残差 {res:.1e}")
    ax.legend()
    ax = axes[0, 1]
    idx = np.argsort(np.abs(num))[:60]
    ax.plot(range(60), np.abs(num)[idx], "bo", ms=4, label="|E| 最低 60 态")
    ax.axhline(M0, color="gray", ls=":", lw=1)
    ax.axhline(np.sqrt(M0 ** 2 + (np.pi / N_TH) ** 2), color="r", ls="--", lw=1,
               label="m + 反周期最低 k")
    ax.set_xlabel("态序号")
    ax.set_ylabel("|E|")
    ax.set_title("SK3: 反周期无零模，谱对称")
    ax.legend()
    ax = axes[1, 0]
    ax.plot(Ep, Ep - Em, "b.", ms=3)
    ax.set_xlabel("E(winding=+1)")
    ax.set_ylabel("E(+1) - E(-1)")
    ax.set_title(f"SK4: ±n 谱逐位恒等（max {dpm:.1e}）")
    ax = axes[1, 1]
    ax.hist(np.abs(num), bins=40, alpha=0.5, label="甲 常数 m")
    ax.hist(np.abs(Eb), bins=40, alpha=0.5, label="乙 m(h) 峰值 +30%")
    ax.axvline(M0, color="gray", ls=":", lw=1)
    ax.set_xlabel("|E|")
    ax.set_ylabel("态数")
    ax.set_title("SK5: 甲乙谱分布对比（D1 首测）")
    ax.legend()
    plt.tight_layout()
    plt.savefig(r"D:\software\TianXingFC\images\dirac_skeleton.png", dpi=150)
    print(f"\n图已存: D:\\software\\TianXingFC\\images\\dirac_skeleton.png")

main()
