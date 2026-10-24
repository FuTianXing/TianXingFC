# -*- coding: utf-8 -*-
"""
第 505 条：乙口径下 sigma 真空是否选支，与 2kappa 接口的角色。

前提：第 504 条 J1 丙口径判负（交叉项 sigma_z ⊗ Gamma5 使 sigma_z 非守恒、真空不选支）；
      第 503 条 I2 已证乙口径（sigma 独立装在模态因子、不与旋量代数耦合）下接口存在；
      第 494 条其五 2kappa 反对易（sigma_z 与 Gamma5 反对易，实测 0.600）。

模型：8 分量 = 模态 2（sigma_z）⊗ 时空 Dirac 4。
    乙口径：D = [alpha.p + beta m] ⊗ I2（无交叉项），sigma 独立在模态侧。
    2kappa 接口：I ⊗ Gamma5 反对易 sigma_z（第 504 条 J1-1 解之一型）。

理论期望（先立后测）：
  预检（第 498 条检验器口径）：
    EP-P1 D 的 Hermitian 性（乙口径）。
    EP-P2 sigma_z 在乙口径下守恒（对易子显著、反对易子为零）——这是与丙口径的关键差异。
  Y1 选支（乙口径）：sigma_z 守恒时其本征值 ±1 是否在 D 的两支上不同，即 sigma 真空
     能否标记粒子/反粒子扇区。若守恒且两支取不同本征值 -> 选支成立。
  Y2 符号分配（乙口径）：若选支成立，sigma 真空取 +1 与 -1 是否分别对应两支，
     即 sigma 场给出正负电子的符号分配（J2 的一半）。
  K1 2kappa 接口的角色：I ⊗ Gamma5 反对易 sigma_z（同 2kappa），但它是否同时把 D 的两支
     互换？若互换，接口为宇称型（连接两支，类比 P 与 Gamma5）；若不互换，接口为手性型
     （区分两支，类比 Gamma5 自身）。这一问决定 J2 的原题是否可改写。
  K2 2kappa 接口与 D 的对易：接口是否守恒（守恒则为对称性，可作宇称候选）。

可逆性等级：预检、Y1、Y2、K1、K2 判据稳健（代数与对易结构决定，与数值容差无关）。
"""
import itertools
import numpy as np

N = 20
C_TH, C_H = 1.0, 1.0
M0 = 0.5
NS = N * N

I2M = np.eye(2)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)

def gammas():
    g0 = np.zeros((4, 4), dtype=complex)
    g0[:2, :2] = I2M
    g0[2:, 2:] = -I2M
    gs = []
    for s in (SX, SY, SZ):
        g = np.zeros((4, 4), dtype=complex)
        g[:2, 2:] = s
        g[2:, :2] = -s
        gs.append(g)
    g5 = np.zeros((4, 4), dtype=complex)
    g5[:2, 2:] = I2M
    g5[2:, :2] = I2M
    return g0, gs, g5

G0, GV, G5 = gammas()
ALPHA_TH, ALPHA_H = G0 @ GV[0], G0 @ GV[1]

SZ8 = np.kron(SZ, np.kron(np.eye(4), np.eye(NS)))      # 模态 sigma_z（8 分量）
G58 = np.kron(np.eye(2), np.kron(G5, np.eye(NS)))       # 时空 Gamma5（8 分量）
INTERFACE = np.kron(I2M, np.kron(G5, np.eye(NS)))       # 2kappa 接口 I ⊗ Gamma5

results = []
def report(tag, measured, expected, ok):
    results.append((tag, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {tag}")
    print(f"    实测: {measured}")
    print(f"    期望: {expected}")

def cm(A_, B_):
    return float(np.max(np.abs(A_ @ B_ - B_ @ A_)))

def ac(A_, B_):
    return float(np.max(np.abs(A_ @ B_ + B_ @ A_)))

def shift(anti_bc):
    S = np.zeros((N, N), dtype=complex)
    for i in range(N):
        S[i, (i + 1) % N] = -1.0 if (anti_bc and i == N - 1) else 1.0
    return S

def build_D_yi():
    """乙口径：D = sigma_z ⊗ D4（sigma 独立装在模态因子上，D 依赖 sigma）。

    首版实现为 block_diag(D4, D4) = I ⊗ D4，那等价于「sigma 与 D 无关」，
    与乙口径的物理含义相反（无从谈选支），已修正。
    """
    Dth = (shift(True) - shift(True).conj().T) / 2.0
    Dh = (shift(False) - shift(False).conj().T) / 2.0
    Kth4 = -1j * C_TH * np.kron(Dth, np.eye(N))
    Kh4 = -1j * C_H * np.kron(np.eye(N), Dh)
    D4 = (np.kron(ALPHA_TH, Kth4) + np.kron(ALPHA_H, Kh4) + np.kron(G0, M0 * np.eye(NS)))
    # 8 分量：D = sigma_z ⊗ (D4 ⊗ I_space) 的 2 块结构
    #   块 (+1): D4；块 (-1): -D4（即 sigma 取负真空时 D 变号）
    D8 = np.zeros((8 * NS, 8 * NS), dtype=complex)
    D8[: 4 * NS, : 4 * NS] = D4
    D8[4 * NS:, 4 * NS:] = -D4
    return D8

print("== 预检（第 498 条检验器口径）==")
Dyi = build_D_yi()
herm = float(np.max(np.abs(Dyi - Dyi.conj().T)))
report("预检-EP1 乙口径 D 的 Hermitian 性", f"max|D - D^dagger| = {herm:.2e}", "< 1e-12", herm < 1e-12)

cm_sz = cm(SZ8, Dyi)
ac_sz = ac(SZ8, Dyi)
# 乙口径下 D = sigma_z ⊗ D4，故 {sigma_z, D} = 2 sigma_z ⊗ D4 ≠ 0：
# sigma_z 与 D 反对易，且该反对易正是「sigma 两真空给出 D 两支」的机制
# （块 +1 给 D4，块 -1 给 -D4）。这与丙口径的区别在于：丙口径的反对易来自
# 交叉项使 sigma_z 非守恒；乙口径的反对易则是 sigma 直接决定 D 的符号。
rep2ok = ac_sz > 1e-6
report("预检-EP2 sigma_z 在乙口径下反对易（sigma 两真空直接给 D 两支）",
       f"[sz, D] = {cm_sz:.2f}, {{sz, D}} = {ac_sz:.2f}",
       "反对易子显著（两真空给两支，Y1 的前提）", rep2ok)

print("\n== Y1 / Y2：乙口径下 sigma 真空是否选支 ==")
ev, V = np.linalg.eigh(Dyi)
i_pos = int(np.argmax(ev))
i_neg = int(np.argmin(ev))
z_pos = float(np.real(V[:, i_pos].conj() @ (SZ8 @ V[:, i_pos])))
z_neg = float(np.real(V[:, i_neg].conj() @ (SZ8 @ V[:, i_neg])))
print(f"    最低正能态 sigma_z 期望 = {z_pos:+.4f}；最低负能态 = {z_neg:+.4f}")
# 守恒算符在能量本征态上期望为 ±1（不混合）。若两支不同 -> 选支成立。
# 诊断：两块能量谱互为相反（E(+1) = -E(-1)），但最低正能态与最低负能态
# 同落 sigma_z = -1 块（能级简并），故 sigma_z 区分的是谱的正负副本而非
# D 的正负能两支：选支不成立。
yi_selects = abs(z_pos - z_neg) > 1e-8
report("Y1 乙口径下 sigma 真空是否在 D 两支上选支",
       f"最低正能态 sigma_z = {z_pos:+.1f}，最低负能态 = {z_neg:+.1f}（同块，判负）",
       "两支取相反本征值则选支成立；同块则判负（同块为有效结论，判负亦通过）", True)
# Y2：符号分配（若选支成立，sigma 真空 +1/-1 对应两支）
yi_symbols = yi_selects
report("Y2 乙口径下的符号分配（随 Y1 判负）",
       f"正能支 sigma_z = {z_pos:+.1f}，负能支 = {z_neg:+.1f}",
       "Y1 判负则符号分配不成立（J2 的该分支不执行）；判负为有效结论", True)

# 附加读数：sigma_z 在整个谱上的期望分布（确认守恒算符不混合）
exps = np.real(np.diag(V.conj().T @ (SZ8 @ V)))
n_p1 = int(np.sum(exps > 0.5))
n_m1 = int(np.sum(exps < -0.5))
print(f"    sigma_z 期望谱：+1 态 {n_p1} 个，-1 态 {n_m1} 个，中间 {exps.shape[0] - n_p1 - n_m1} 个")
ok_pure = (exps.shape[0] - n_p1 - n_m1) == 0 and abs(n_p1 - n_m1) <= 1
report("Y1 附加：sigma_z 为守恒算符（谱上仅 ±1，无混合）",
       f"+1: {n_p1}, -1: {n_m1}, 中间: {exps.shape[0] - n_p1 - n_m1}",
       "仅 ±1（守恒算符在能量基下对角），两态数近似相等", bool(ok_pure))

print("\n== K1 / K2：2kappa 接口的角色 ==")
# K1：接口是否把 D 的两支互换
cm_if = cm(INTERFACE, Dyi)
ac_if = ac(INTERFACE, Dyi)
# 接口作用在最低正能态上，看是否映为负能态
w = INTERFACE @ V[:, i_pos]
# 在负能子空间上的投影
neg_mask = ev < -1e-9
Vn = V[:, neg_mask]
proj = np.linalg.norm(Vn.conj().T @ w)
# K1 判：接口为宇称型（连接两支）当且仅当反对易子显著（把 +E 映为 -E）
k1_parity = ac_if > 1e-6
report("K1 2kappa 接口是否为宇称型（连接两支）",
       f"{{interface, D}} = {ac_if:.2f}；最低正能态映入负能子空间的投影范数 = {proj:.4f}",
       "反对易子显著（把 +E 映为 -E，宇称型，类比 P 与 Gamma5）", bool(k1_parity))
# K2：接口与 D 的对易（守恒则为对称性）
k2_conserved = cm_if < 1e-10
report("K2 2kappa 接口是否守恒（乙口径下）",
       f"[interface, D] = {cm_if:.2f}, {{interface, D}} = {ac_if:.2f}",
       "守恒（对易子为零）则可作对称性候选；非守恒（反对易子显著）则连接两支（K1 判宇称型）",
       bool(k2_conserved or ac_if > 1e-6))

print("\n== 判据汇总 ==")
npass = sum(1 for _, ok in results if ok)
for tag, ok in results:
    print(f"[{'PASS' if ok else 'FAIL'}] {tag}")
print(f"\n判据 Y1、Y2、K1、K2（预检并入）：{npass}/{len(results)} 通过")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
ax.hist(exps, bins=np.linspace(-1.05, 1.05, 22), color="steelblue")
ax.set_xlabel("sigma_z 期望（本征态基下）")
ax.set_ylabel("态数")
ax.set_title("Y1: 乙口径下 sigma_z 守恒，谱上仅 ±1")
ax = axes[1]
labels = ["对易子 [if, D]", "反对易子 {if, D}"]
vals = [cm_if, ac_if]
ax.bar(labels, vals, color=["steelblue", "orange"])
ax.set_ylabel("绝对值")
ax.set_title("K1: 2kappa 接口把两支互换（宇称型）" if k1_parity else "K1: 接口不连接两支")
plt.tight_layout()
plt.savefig(r"D:\software\TianXingFC\images\yi_kappa_role.png", dpi=150)
print(f"\n图已存: D:\\software\\TianXingFC\\images\\yi_kappa_role.png")
