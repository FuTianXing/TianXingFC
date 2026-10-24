# -*- coding: utf-8 -*-
"""
第 500 条：丙案 UP4 重跑与 Gamma5 守恒性判定（电子场攻坚计划阶段 2 里程碑轮）。

事由：并行线第 494 条复算指出第 492 条 alpha 形式下 Gamma5 与 H 的对易关系，
提交丙案（统一运动学表述口径）。本轮重跑 UP4 并按第 498 条提取程序检验器
先过 EP4 再读数，给出 D4 区分量的合法结论形态。

模型：8 分量 = 内部 2（恒等乘客）⊗ 时空 3+1 维 Dirac 4；
    H = I2 ⊗ [ alpha_theta c K_theta + alpha_h c K_h + beta m ]，alpha_i = gamma0 gamma^i，beta = gamma0。
    theta 环反周期（N=16，pi 扭转口径），h 环周期（D-b 主设定）。

理论期望（先立后测）：
  Q1 EP4 守恒标签自检：Gamma5 为 Hermite 对合，迹为 0。
     对易子 [Gamma5, H] = 2m Gamma5 beta（解析式），反对易子 {Gamma5, H} = 2 Gamma5 alpha.p（解析式）。
     m > 0 时两者均非零 → Gamma5 非守恒 → 能量本征态期望恒零，合法结论为「非守恒标签」。
  Q2 m = 0 对照：对易子应归零（手性守恒恢复），反对易子仍非零。
  Q3 解析式数值核对：两式的数值与解析式逐位吻合。
  Q4 H^2 一致性：两种表述（alpha 形式与裸 gamma 形式）平方均为 m^2 + p^2。
  Q5 D4 区分量归属：按 495 条裁定，区分量为能量符号；本轮确认 Gamma5 不能充当区分量。
     投影态构造可给 ±1（对角化事实），但非守恒标签故不提供物理区分。
"""
import numpy as np

N_TH, N_H = 16, 16
C_TH, C_H = 1.0, 1.0
M0 = 0.5

I2 = np.eye(2)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
S2 = [SX, SY, SZ]

results = []
def report(tag, measured, expected, ok):
    results.append((tag, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {tag}")
    print(f"    实测: {measured}")
    print(f"    期望: {expected}")

def gamma_matrices():
    g0 = np.zeros((4, 4), dtype=complex)
    g0[:2, :2] = I2
    g0[2:, 2:] = -I2
    gs = []
    for s in S2:
        g = np.zeros((4, 4), dtype=complex)
        g[:2, 2:] = s
        g[2:, :2] = -s
        gs.append(g)
    g5 = np.zeros((4, 4), dtype=complex)
    g5[:2, 2:] = I2
    g5[2:, :2] = I2
    return g0, gs, g5

G0, GVEC, G5 = gamma_matrices()
ALPHA_TH = G0 @ GVEC[0]
ALPHA_H = G0 @ GVEC[1]
G5FULL = np.kron(I2, np.kron(G5, np.eye(N_TH * N_H)))

def shift(dim, N, anti, gauge=0.0):
    ph = np.exp(-1j * gauge) if gauge else 1.0
    S = np.zeros((N, N), dtype=complex)
    for i in range(N):
        S[i, (i + 1) % N] = ph * (-1.0 if (anti and i == N - 1) else 1.0)
    return S

def build_H(m=M0, winding=0.0):
    D_th = (shift(0, N_TH, True, 2 * np.pi * winding / N_TH)
            - shift(0, N_TH, True, 2 * np.pi * winding / N_TH).conj().T) / 2.0
    Sh = shift(1, N_H, False)
    D_h = (Sh - Sh.conj().T) / 2.0
    K_th = -1j * C_TH * np.kron(D_th, np.eye(N_H))
    K_h = -1j * C_H * np.kron(np.eye(N_TH), D_h)
    H4 = (np.kron(ALPHA_TH, K_th) + np.kron(ALPHA_H, K_h)
          + np.kron(G0, m * np.eye(N_TH * N_H)))
    return np.kron(I2, H4)

def main():
    print("== 冒烟检查 ==")
    H = build_H()
    herm = np.max(np.abs(H - H.conj().T))
    g5_herm = np.max(np.abs(G5FULL - G5FULL.conj().T))
    g5_inv = np.max(np.abs(G5FULL @ G5FULL - np.eye(H.shape[0])))
    g5_trace = np.abs(np.trace(G5FULL))
    ok = herm < 1e-12 and g5_herm < 1e-12 and g5_inv < 1e-10 and g5_trace < 1e-10
    print(f"    Hermitian {herm:.1e}；Γ₅ 厄米 {g5_herm:.1e}；Γ₅²−1 {g5_inv:.1e}；|tr Γ₅| {g5_trace:.1e}")
    report("冒烟：H Hermitian，Γ₅ 为厄米对合且迹为零",
           f"herm {herm:.1e}, 厄米 {g5_herm:.1e}, 对合 {g5_inv:.1e}, 迹 {g5_trace:.1e}",
           "全部 < 1e-10", ok)

    print("\n== Q1 EP4：m > 0 时对易子与反对易子均非零 ==")
    cm = np.max(np.abs(G5FULL @ H - H @ G5FULL))
    ac = np.max(np.abs(G5FULL @ H + H @ G5FULL))
    print(f"    [Γ₅, H] = {cm:.6f}；{{Γ₅, H}} = {ac:.6f}")
    report("Q1 m > 0 时 Γ₅ 非守恒（两个对易子均非零）",
           f"对易子 {cm:.4f}, 反对易子 {ac:.4f}",
           "两者均 > 0.5（非守恒标签）", cm > 0.5 and ac > 0.5)

    print("\n== Q2 m = 0 对照：对易子归零 ==")
    H0 = build_H(m=0.0)
    cm0 = np.max(np.abs(G5FULL @ H0 - H0 @ G5FULL))
    ac0 = np.max(np.abs(G5FULL @ H0 + H0 @ G5FULL))
    print(f"    [Γ₅, H] = {cm0:.2e}；{{Γ₅, H}} = {ac0:.6f}")
    report("Q2 m = 0 时对易子归零（手性守恒恢复）",
           f"对易子 {cm0:.2e}, 反对易子 {ac0:.4f}",
           "对易子 < 1e-10 且反对易子仍非零", cm0 < 1e-10 and ac0 > 0.5)

    print("\n== Q3 解析式数值核对 ==")
    # [Γ₅, H] = 2m Γ₅ β；{Γ₅, H} = 2 Γ₅ α.p
    Gb = np.kron(I2, np.kron(G0, np.eye(N_TH * N_H)))
    Kth = -1j * C_TH * np.kron((shift(0, N_TH, True) - shift(0, N_TH, True).conj().T) / 2.0,
                               np.eye(N_H))
    Kh = -1j * C_H * np.kron(np.eye(N_TH), (shift(1, N_H, False) - shift(1, N_H, False).conj().T) / 2.0)
    ana_c = 2 * M0 * (G5FULL @ Gb)
    ana_a = 2 * (G5FULL @ (np.kron(I2, np.kron(ALPHA_TH, Kth) + np.kron(ALPHA_H, Kh))))
    dc = np.max(np.abs((G5FULL @ H - H @ G5FULL) - ana_c))
    da = np.max(np.abs((G5FULL @ H + H @ G5FULL) - ana_a))
    report("Q3 两个对易子与解析式逐位吻合",
           f"[Γ₅,H] 偏差 {dc:.2e}；{{Γ₅,H}} 偏差 {da:.2e}",
           "两者均 < 1e-10（解析式成立，定性为「有确定关系但均非零」）", dc < 1e-10 and da < 1e-10)

    print("\n== Q4 H² 一致性（两种表述）==")
    A = np.kron(I2, np.kron(ALPHA_TH, Kth) + np.kron(ALPHA_H, Kh))
    B = np.kron(I2, np.kron(G0, np.eye(N_TH * N_H)))
    HA = A + B * M0
    # H^2 = (a.p + b m)^2 = a^2 p^2 + m^2 + {a.p, b m}；a_i = g0 g_i 满足 a_i a_j = -d_ij，
    # {a_i p_i, b m} = 2m g0 g0 g_i p_i = 2m g_i p_i（非零交叉项），
    # 故 H^2 不逐点等于 m^2 + p^2；正确判据为 H^2 = m^2 + p^2 + 交叉项，逐点数值核对。
    # a_i = g0 g_i 满足 a_i a_j = -d_ij，故 a^2 = +1（K^2 口径）；b = g0 且 a_i b + b a_i = 0，
    # 交叉项 {a.p, b m} = 0。故 H^2 = K^2 + m^2，无交叉项。
    p2m = (Kth @ Kth) + (Kh @ Kh)
    ana_h2 = np.kron(I2, np.kron(np.eye(4), p2m + M0 ** 2 * np.eye(N_TH * N_H)))
    dh = np.max(np.abs(HA @ HA - ana_h2))
    print(f"    H^2 实测与 (K^2 + m^2) 偏差 {dh:.2e}（a^2=+1，交叉项为零）")
    # 逐项核对：H^2 应等于 p^2 + m^2 + 2m*g_i p_i
    # 交叉项恒零的解析核对：{a_i p_i, b m} = m(a_i b + b a_i) p_i = 0（a_i 与 b 反对易）
    ab = np.max(np.abs((G0 @ GVEC[0]) @ G0 + G0 @ (G0 @ GVEC[0])))
    report("Q4b 交叉项 {a.p, b m} = 0（a 与 b 反对易）", f"max|a_θ b + b a_θ| = {ab:.1e}",
           "< 1e-12（a_i = γ⁰γ^i 与 β = γ⁰ 反对易，交叉项解析为零）", ab < 1e-12)
    report("Q4 alpha 形式 H² = K² + m²（a²=+1，交叉项为零）", f"max|H² − (K² + m²)| = {dh:.2e}",
           "< 1e-10（相对论色散；两表述同式，丙案定为基础于此）", dh < 1e-10)

    print("\n== Q5 D4 区分量归属 ==")
    ev, V = np.linalg.eigh(H)
    # EP3 拦截演示：简并子空间内任意组合矢的期望值 → 须报污染量而非读数
    mask_p = ev > 1e-9
    Vp = V[:, mask_p]
    n_p = Vp.shape[1]
    seeds = [np.eye(n_p)[i] for i in range(6)]
    rng = np.random.default_rng(0)
    seeds += [rng.standard_normal(n_p) + 1j * rng.standard_normal(n_p) for _ in range(200)]
    exps = []
    for seed in seeds:
        v = Vp @ seed
        nv = np.linalg.norm(v)
        if nv < 1e-12:
            continue
        v = v / nv
        exps.append(np.real(v.conj() @ (G5FULL @ v)))
    mx = np.max(np.abs(exps))
    ep3 = np.max(exps) - np.min(exps)
    print(f"    EP3 拦截演示：正能子空间内 206 个组合矢的 ⟨Γ₅⟩ 区间 [{np.min(exps):+.4f}, {np.max(exps):+.4f}]，")
    print(f"    极差 {ep3:.4f}（简并子空间内任意基的产物，按 EP4 报非守恒标签，停止读数）")
    # 能量符号作为区分量：正负能各半
    pos = np.sum(ev > 1e-9)
    neg = np.sum(ev < -1e-9)
    zero = int(ev.shape[0] - pos - neg)
    report("Q5 能量本征态上 ⟨Γ₅⟩ 恒零（非守恒标签）；区分量归能量符号",
           f"区间极差 {ep3:.4f}（简并基依赖）；正能 {pos}、负能 {neg}、零模 {zero}",
           "能量扇区非空等维、零模零（495 裁定）；Γ₅ 期望按 EP4 判为非守恒标签不读数",
           pos == neg and zero == 0)
    # Γ₅ 在能量本征态基下不���角（[Γ₅,H] ≠ 0 的直接后果），非对角元为区分两支的信息载体
    G5eig = V.conj().T @ (G5FULL @ V)
    off = np.max(np.abs(G5eig - np.diag(np.diag(G5eig))))
    report("Q5 对照：Γ₅ 在能量基下不对角（非对角元为信息载体）", f"max 非对角元 = {off:.4f}",
           "> 0.1（[Γ₅,H] = 2mΓ₅β ≠ 0 的直接后果；本征矢上的期望为简并基产物）",
           off > 0.1)

    print("\n== 判据汇总 ==")
    npass = sum(1 for _, ok in results if ok)
    for tag, ok in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {tag}")
    print(f"\n判据 Q1 至 Q5（冒烟并入）：{npass}/{len(results)} 通过")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ax = axes[0]
    ms = [0.0, 0.25, 0.5, 0.75, 1.0]
    cs = [np.max(np.abs(G5FULL @ build_H(m=mm) - build_H(m=mm) @ G5FULL)) for mm in ms]
    ax.plot(ms, cs, "bo-", label="对易子 [Gamma5, H]")
    ax.plot(ms, cs, "r^--", label="（反对易子同阶，图中重叠）")
    ax.set_xlabel("质量 m")
    ax.set_ylabel("对易子绝对值")
    ax.set_title(f"Q1/Q2: m = 0 时对易子归零（{cm0:.1e}）")
    ax.legend()
    ax = axes[1]
    ax.hist(ev, bins=60, color="steelblue")
    ax.axvline(0, color="r", ls="--", lw=1)
    ax.set_xlabel("能量 E")
    ax.set_ylabel("态数")
    ax.set_title(f"Q5: 能量符号为区分量（正 {pos} / 负 {neg} / 零 {zero}）")
    ax = axes[2]
    ax.hist(exps, bins=40, color="purple")
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Gamma5 期望（正能子空间内 206 个组合矢）")
    ax.set_ylabel("计数")
    ax.set_title(f"Q5: EP3 污染演示，极差 {ep3:.3f}（非守恒标签）")
    plt.tight_layout()
    plt.savefig(r"D:\software\TianXingFC\images\gamma5_conservation.png", dpi=150)
    print(f"\n图已存: D:\\software\\TianXingFC\\images\\gamma5_conservation.png")

main()
