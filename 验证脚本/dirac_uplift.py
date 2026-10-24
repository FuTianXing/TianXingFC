# -*- coding: utf-8 -*-
"""
第 492 条：升维第二轮——两因子 Dirac 与 D4 显式投影（电子场攻坚计划阶段 2）。

模型：全空间 8 分量 = 内部模态 2（σ_z 标签，恒等乘客）⊗ 时空 3+1 维 Dirac 4（Dirac 基），
    H = I₂ ⊗ [ α_θ c_θ K_θ + α_h c_h K_h + β m ]，
    α_i = γ⁰γ^i（Hermitian），β = γ⁰；K_θ 为反周期差分（π 扭转口径），K_h 周期（D-b 主设定）。
θ 环 N_θ = 16 反周期、h 环 N_h = 16 周期；γ³ 不激活但保留在 γ⁵ = iγ⁰γ¹γ²γ³ 的定义中。

理论期望（先立后测）：
  UP1（D2 主体）8 分量实空间谱 vs 离散解析谱逐位吻合（每模式 8 支：
     ±√(c_θ²sin²k_θ + c_h²sin²k_h + m²) 各内部 2 重）。
  UP2（D2 连续极限）低 k 档 (E²−m²)/(c²k²) → 1，残差 ≤ sin 展开预算 k²/3。
  UP3（宇称）P = γ⁰ 在三十二个 Clifford 单项式中穷举唯一（复核 491 W2）；
     PH(k)P⁻¹ = H(−k)（θ 与 h 动量同时翻号）。
  UP4（D4 关闭读数，本轮核心）显式投影构造：手征投影态 ⟨Γ₅⟩ = ±1 精确
     （对照 491 W5 质量本征态的零）；Weyl 块谱 E = ±c|k|（单支无质量）；质量项为块外耦合。
  UP5（±n 镜像迁移）winding = ±1 规范平移的 8 分量谱逐位恒等。

可逆性等级：UP1/UP3/UP4/UP5 判据稳健（解析恒等式与穷举）；UP2 含 sin 展开协议伪影，
  容差从误差预算 k²/3 推导。前提：D-b 主设定周期边界、D-c 取 π 扭转奇偶、
  D4 关闭机制取显式投影构造主线（2026-09-29 作者裁定）。
"""
import numpy as np

N_TH, N_H = 16, 16
C_TH, C_H = 1.0, 1.0
M0 = 0.5

I2 = np.eye(2)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
S2 = [SX, SY, SZ]

def gamma_matrices():
    """Dirac 基 3+1 维 γ。γ⁰ = diag(I,-I)；γ^i = [[0,σ_i],[-σ_i,0]]；γ⁵ = [[0,I],[I,0]]。"""
    g0 = np.zeros((4, 4), dtype=complex)
    g0[:2, :2] = I2
    g0[2:, 2:] = -I2
    g5 = np.zeros((4, 4), dtype=complex)
    g5[:2, 2:] = I2
    g5[2:, :2] = I2
    gs = []
    for s in S2:
        g = np.zeros((4, 4), dtype=complex)
        g[:2, 2:] = s
        g[2:, :2] = -s
        gs.append(g)
    return g0, gs, g5

G0, GVEC, G5 = gamma_matrices()

results = []
def report(tag, measured, expected, ok):
    results.append((tag, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {tag}")
    print(f"    实测: {measured}")
    print(f"    期望: {expected}")

def shift(dim, N, antiperiodic, gauge=0.0):
    ph = np.exp(-1j * gauge) if gauge else 1.0
    S = np.zeros((N, N), dtype=complex)
    for i in range(N):
        j = (i + 1) % N
        S[i, j] = ph * (-1.0 if (antiperiodic and i == N - 1) else 1.0)
    return S

def build_H(winding=0.0, m=None):
    """8 分量 H = I₂ ⊗ [ α_θ c_θ K_θ + α_h c_h K_h + β m ]。

    K_θ 反周期含绕数规范（gauge = 2π w / N_TH）；K_h 周期。
    时空块：H_4 = γ⁰(γ¹ c_θ K_θ + γ² c_h K_h + m)，等价 α·p + βm。
    """
    Sth = shift(0, N_TH, True, gauge=2 * np.pi * winding / N_TH)
    D_th = (Sth - Sth.conj().T) / 2.0
    Sh = shift(1, N_H, False)
    D_h = (Sh - Sh.conj().T) / 2.0
    mm = M0 if m is None else m
    K_th = -1j * C_TH * np.kron(D_th, np.eye(N_H))
    K_h = -1j * C_H * np.kron(np.eye(N_TH), D_h)
    H4 = (np.kron(G0 @ GVEC[0], K_th)
          + np.kron(G0 @ GVEC[1], K_h)
          + np.kron(G0, mm * np.eye(N_TH * N_H)))
    return np.kron(I2, H4)

def analytic_spectrum(winding=0.0):
    n_th = np.arange(N_TH) - N_TH // 2
    k_th = 2 * np.pi * (n_th + 0.5) / N_TH + 2 * np.pi * winding / N_TH
    n_h = np.arange(N_H) - N_H // 2
    k_h = 2 * np.pi * n_h / N_H
    KT, KH = np.meshgrid(k_th, k_h, indexing="ij")
    E2 = C_TH ** 2 * np.sin(KT) ** 2 + C_H ** 2 * np.sin(KH) ** 2 + M0 ** 2
    E = np.sqrt(E2).ravel()
    return np.sort(np.concatenate([np.repeat(E, 4), np.repeat(-E, 4)])), k_th, k_h

def projector_states(H):
    """取若干本征态作手征投影，返回投影态的 ⟨Γ₅⟩ 与归一化能量。"""
    ev, V = np.linalg.eigh(H)
    Gfull = np.kron(I2, np.kron(G5, np.eye(N_TH * N_H)))
    out = []
    for idx in (0, ev.shape[0] // 4, ev.shape[0] // 2, ev.shape[0] - 1):
        v = V[:, idx]
        for sgn in (+1, -1):
            P = (np.eye(H.shape[0]) + sgn * Gfull) / 2.0
            w = P @ v
            nrm = np.linalg.norm(w)
            if nrm < 1e-12:
                out.append((idx, sgn, None, None))
                continue
            w = w / nrm
            exp_g5 = np.real(w.conj() @ (Gfull @ w))
            out.append((idx, sgn, exp_g5, float(ev[idx])))
    return out

def main():
    print("== 冒烟检查 ==")
    H = build_H()
    herm = np.max(np.abs(H - H.conj().T))
    anti = np.max(np.abs(G0 @ G5 + G5 @ G0))
    g5sq = np.max(np.abs(G5 @ G5 - np.eye(4)))
    ok = herm < 1e-12 and anti < 1e-12 and g5sq < 1e-12
    print(f"    Hermitian {herm:.1e}；{{γ⁰,γ⁵}} {anti:.1e}；γ⁵²−1 {g5sq:.1e}")
    report("冒烟：8 分量 H Hermitian，γ⁰ 与 γ⁵ 反对易，γ⁵² = 1",
           f"herm {herm:.1e}, anti {anti:.1e}, γ⁵² {g5sq:.1e}",
           "全部 < 1e-12", ok)

    print("\n== UP1（D2 主体）：8 分量谱 vs 离散解析谱 ==")
    num = np.linalg.eigvalsh(H)
    ana, k_th, k_h = analytic_spectrum()
    d = np.max(np.abs(num - ana))
    report("UP1 实空间 8 分量谱与离散解析谱逐位吻合", f"max|ΔE| = {d:.2e}",
           "< 1e-10（每模式 8 支：±E 各内部 2 重）", d < 1e-10)

    print("\n== UP2（D2 连续极限）==")
    n_th = np.arange(N_TH) - N_TH // 2
    k_th_vals = 2 * np.pi * (n_th + 0.5) / N_TH
    n_h = np.arange(N_H) - N_H // 2
    k_h_vals = 2 * np.pi * n_h / N_H
    KT, KH = np.meshgrid(k_th_vals, k_h_vals, indexing="ij")
    k2 = (C_TH * KT) ** 2 + (C_H * KH) ** 2
    E2_mode = C_TH ** 2 * np.sin(KT) ** 2 + C_H ** 2 * np.sin(KH) ** 2
    lo = k2 < 0.06
    budget = np.max(k2[lo]) / 3.0
    ratio = E2_mode[lo] / k2[lo]
    res = np.max(np.abs(ratio - 1.0))
    report("UP2 低 k 模式连续极限", f"max |(E²−m²)/(c²k²) − 1| = {res:.2e}（{lo.sum()} 模式）",
           f"≤ 误差预算 k²/3 = {budget:.2e}（sin 展开 O((ka)²)）", res <= budget)

    print("\n== UP3（宇称：穷举唯一复核 + PH(k)P⁻¹ = H(−k)）==")
    # 三十二个 Clifford 单项式穷举（含与不含 i 因子）
    monos = [np.eye(4)]
    import itertools
    basis = [G0] + GVEC
    for r in range(1, 5):
        for combo in itertools.combinations(range(4), r):
            M = np.eye(4, dtype=complex)
            for c in combo:
                M = M @ basis[c]
            monos.append(M)
            monos.append(1j * M)
    uniq, count = [], 0
    for M in monos:
        c1 = np.max(np.abs(M @ G0 - G0 @ M))
        a1 = np.max(np.abs(M @ GVEC[0] + GVEC[0] @ M))
        a2 = np.max(np.abs(M @ GVEC[1] + GVEC[1] @ M))
        a3 = np.max(np.abs(M @ GVEC[2] + GVEC[2] @ M))
        psq = np.max(np.abs(M @ M - np.eye(4)))
        her = np.max(np.abs(M - M.conj().T))
        if c1 < 1e-10 and a1 < 1e-10 and a2 < 1e-10 and a3 < 1e-10 and psq < 1e-10 and her < 1e-10:
            uniq.append(M)
            count += 1
    only_g0 = len(uniq) == 1 and np.max(np.abs(uniq[0] - G0)) < 1e-10
    # PH(k)P⁻¹ = H(−k)：γ⁰ 共轭把 γ¹、γ² 反号
    conv = np.max(np.abs(G0 @ (GVEC[0] @ G0) @ G0 + GVEC[0] @ G0))  # γ⁰α_θγ⁰ = −α_θ
    report("UP3 宇称算符穷举唯一（P = γ⁰）", f"三十二单项式中合格者 {count} 个，是否仅 γ⁰：{only_g0}",
           "恰 1 个且为 γ⁰（复核 491 W2）", only_g0 and count == 1)
    report("UP3 宇称共轭翻动量（γ⁰γ¹γ⁰ = −γ¹）", f"max|γ⁰α_θγ⁰ + α_θ| = {conv:.1e}",
           "< 1e-12（PH(k)P⁻¹ = H(−k) 的代数根源）", conv < 1e-12)

    print("\n== UP4（D4 关闭读数：显式投影构造）==")
    proj = projector_states(H)
    bad = [p for p in proj if p[2] is None or abs(abs(p[2]) - 1.0) > 1e-10]
    good = [p for p in proj if p[2] is not None]
    exps = [p[2] for p in good]
    print(f"    投影态 ⟨Γ₅⟩：{[f'{e:+.6f}' for e in exps]}（对照 491 W5 质量本征态全零）")
    ok4 = len(good) > 0 and all(abs(abs(e) - 1.0) < 1e-10 for e in exps)
    report("UP4 显式投影态 ⟨Γ₅⟩ = ±1 精确", f"{len(good)} 个投影态全部 ±1，失败 {len(bad)} 个",
           "|⟨Γ₅⟩| = 1（Weyl 分量有确定手性；对照质量本征态为零）", ok4)
    # Weyl 块谱：P₊ 子空间内 H 的块（无质量档 m = 0）
    H0 = build_H(m=0.0)
    Gfull = np.kron(I2, np.kron(G5, np.eye(N_TH * N_H)))
    evG, UG = np.linalg.eigh(Gfull)
    idxp = np.where(np.abs(evG - 1.0) < 1e-10)[0]
    Up = UG[:, idxp]
    Wb = Up.conj().T @ H0 @ Up
    Ew = np.linalg.eigvalsh(Wb)
    pmag = np.sqrt((C_TH ** 2 * np.sin(KT) ** 2 + C_H ** 2 * np.sin(KH) ** 2).ravel())
    ana_w = np.sort(np.concatenate([np.repeat(pmag, 2), np.repeat(-pmag, 2)]))
    nmin = min(len(Ew), len(ana_w))
    dw = np.max(np.abs(np.sort(Ew)[:nmin] - ana_w[:nmin]))
    report("UP4 Weyl 块谱（m = 0 投影块）", f"max|ΔE| = {dw:.2e}（对解析 Weyl 谱 |E| = c|sin k|，{len(Ew)} 维）",
           "< 1e-10（单支无质量色散；质量为块外耦合）", dw < 1e-10)

    print("\n== UP5（±n 镜像迁移）==")
    Hp = build_H(winding=+1)
    Hm = build_H(winding=-1)
    Ep, Em = np.linalg.eigvalsh(Hp), np.linalg.eigvalsh(Hm)
    dpm = np.max(np.abs(Ep - Em))
    report("UP5 ±n 背景 8 分量谱逐位恒等", f"max|E(+n) − E(−n)| = {dpm:.2e}",
           "< 1e-10（π 扭转口径的镜像在升维结构下保持）", dpm < 1e-10)

    print("\n== 协议伪影分层 ==")
    print("    fermion doubling：中心差分每维一附加谷，因子 4；Wilson 项落时空因子，")
    print("    与显式投影的关系登记后续（Wilson 破坏手征对称，与投影构造的相容性未判）")
    print("\n== 判据汇总 ==")
    npass = sum(1 for _, ok in results if ok)
    for tag, ok in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {tag}")
    print(f"\n判据 UP1 至 UP5（冒烟并入）：{npass}/{len(results)} 通过")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax = axes[0, 0]
    lo_idx = np.argsort(np.abs(num))[:80]
    ax.plot(range(80), np.abs(num)[lo_idx], "b.", ms=4, label="|E| 最低 80 态（8 支结构）")
    ax.axhline(np.sqrt(M0 ** 2 + (np.pi / N_TH) ** 2), color="r", ls="--", lw=1,
               label="m + 反周期最低 k")
    ax.axhline(M0, color="gray", ls=":", lw=1)
    ax.set_xlabel("态序号")
    ax.set_ylabel("|E|")
    ax.set_title("UP1: 8 分量谱最低态（内部 2 重简并可见）")
    ax.legend()
    ax = axes[0, 1]
    ax.scatter(k2[lo], E2_mode[lo], s=25, c="b", label="E² − m²（低 k 档解析谱）")
    kk = np.linspace(0, 0.02, 20)
    ax.plot(kk, kk, "k--", lw=1, label="c²k²")
    ax.set_xlabel("c²k²")
    ax.set_ylabel("E² − m²")
    ax.set_title(f"UP2: 连续极限，残差 {res:.1e}")
    ax.legend()
    ax = axes[1, 0]
    picks = [(i, s, e) for (i, s, e, _) in proj if e is not None][:8]
    ax.bar(range(len(picks)), [p[2] for p in picks],
           color=["b" if p[1] > 0 else "r" for p in picks])
    ax.axhline(1.0, color="gray", ls=":", lw=1)
    ax.axhline(-1.0, color="gray", ls=":", lw=1)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel("投影态样本")
    ax.set_ylabel("Gamma5 期望")
    ax.set_title("UP4: 显式投影态手性期望 = ±1")
    ax = axes[1, 1]
    ax.plot(Ep, Ep - Em, "b.", ms=3)
    ax.set_xlabel("E(winding=+1)")
    ax.set_ylabel("E(+1) − E(−1)")
    ax.set_title(f"UP5: ±n 谱逐位恒等（max {dpm:.1e}）")
    plt.tight_layout()
    plt.savefig(r"D:\software\TianXingFC\images\dirac_uplift.png", dpi=150)
    print(f"\n图已存: D:\\software\\TianXingFC\\images\\dirac_uplift.png")

main()
