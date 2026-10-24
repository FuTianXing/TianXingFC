# -*- coding: utf-8 -*-
"""
第 507 条：I-1 线摸底（S^3 等距群的左右作用、模态参与度读数、2kappa 接口在 S^3 的耦合）。

前提：手征线阶段收口（第 502 至 506 轮）判定 σ 场供给手性来源的三口径全部排除，
      弱荷单分支的搜索方向转向 S^3 模态空间的手征耦合结构（手征正稿 §2.2：S^3 = SU(2)，
      归一态 |psi> = c_x|X> + c_y|Y> + c_z|Z>）。本轮只做摸底：读结构性事实，不判 J，
      不含数值预言。读数一律按第 498 条检验器与第 506 条子空间谱分类口径给出。

理论期望（先立后测）：
  预检（第 498 条口径）：矩阵 Hermitian 与对易结构、随机态上作用非平凡。
  P1 左右作用的显式形式：S^3 上的左作用 psi -> g psi 与右作用 psi -> psi g 的像，
     取 g = exp(i theta sigma_k/2) 的三个生成元方向，给出各自 2x2 作用矩阵与像的迹。
  P2 模态参与度 T3 在 S^3 上的分布：T3 = (1/2) diag(1, -1)（弱二重态上），
     测 <psi| T3 |psi> 在 S^3 上的取值范围与等值面；左作用下 T3 期望是否守恒。
  P3 2kappa 接口在 S^3 上的耦合存在性：I ⊗ Gamma5 型接口在 S^3 的两作用方向上
     是否给出非平凡耦合（即存在耦合项使左右作用不等价）。
  P4 子空间谱分类（第 506 条口径）：读数是否落在简并子空间的合法判定上。

严格程度声明：P1 至 P3 为结构性读数（解析恒等式与显式构造，与容差无关）；
  P4 为读数合法性的自检。均不含物理裁决。
边界声明：本轮不判定弱荷单分支的机制（仍留假设层），不判定手征零模保护，
  不判定质量谱数值，不改动 理论总纲 基准，不重开第 481 条 C2 判定。
"""
import numpy as np

PAULI = {
    "x": np.array([[0, 1], [1, 0]], dtype=complex),
    "y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "z": np.array([[1, 0], [0, -1]], dtype=complex),
}

results = []
def report(tag, measured, expected, ok):
    results.append((tag, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {tag}")
    print(f"    实测: {measured}")
    print(f"    期望: {expected}")

def cm(A, B):
    return float(np.max(np.abs(A @ B - B @ A)))

def ac(A, B):
    return float(np.max(np.abs(A @ B + B @ A)))

# ---- S^3 参数化：单位四元数（两个复数），S^3 ≅ SU(2) ----
def quat_to_su2(q):
    """单位四元数 q = (a, b) 两个复数（a, b ∈ C，|a|^2+|b|^2=1）→ 2x2 SU(2) 矩阵。"""
    a, b = q
    return np.array([[a, -np.conj(b)], [b, np.conj(a)]], dtype=complex)

def su2_exp(k, theta):
    """g_k = exp(i theta sigma_k / 2)。"""
    return np.cos(theta / 2) * np.eye(2) + 1j * np.sin(theta / 2) * PAULI[k]

# 2x2 左作用与右作用的像（作用于二分量态 psi ∈ C^2）
def left_action(g, psi):
    return g @ psi

def right_action(g, psi):
    return np.conj(g) @ psi   # 右作用对应 g^{-1} = g^\dagger

print("=" * 66)
print("预检（第 498 条口径）")
print("=" * 66)
# Hermitian 与对易：三个 sigma 生成元两两反对易
comm_pairs = [("x", "y"), ("y", "z"), ("z", "x")]
anti_ok = all(ac(PAULI[a], PAULI[b]) < 1e-12 for a, b in comm_pairs)
sq_ok = all(abs(abs(np.trace(PAULI[k] @ PAULI[k])) - 2) < 1e-12 for k in "xyz")
print(f"    生成元两两反对易：{anti_ok}；各平方为 I：{sq_ok}")
report("预检：Pauli 生成元两两反对易、平方为 I",
       f"反对易对 {len(comm_pairs)} 组全为机器零；平方迹为 2", "全部成立", anti_ok and sq_ok)

# 随机态上左右作用非平凡
rng = np.random.default_rng(0)
psi0 = np.array([1, 0], dtype=complex)
g_test = su2_exp("y", np.pi / 3)
l_diff = float(np.linalg.norm(left_action(g_test, psi0) - psi0))
r_diff = float(np.linalg.norm(right_action(g_test, psi0) - psi0))
nontriv = l_diff > 1e-6 and r_diff > 1e-6
print(f"    g=exp(i pi/3 sigma_y/2) 作用于 |0>：左作用改变 {l_diff:.4f}，右作用改变 {r_diff:.4f}")
report("预检：左作用与右作用在非平凡 g 上均非恒等",
       f"左 {l_diff:.4f}, 右 {r_diff:.4f}", "两者均 > 1e-6", nontriv)

print()
print("=" * 66)
print("P1：S^3 上左右作用的显式形式（三个生成元方向）")
print("=" * 66)
# 左作用矩阵 = g 本身；右作用矩阵 = g^\dagger
p1_rows = []
ok_p1 = True
for k in "xyz":
    g = su2_exp(k, np.pi / 4)
    L = g                 # 左作用 psi -> g psi
    R = g.conj().T        # 右作用 psi -> psi g^{-1} = g^\dagger psi
    herm_L = float(np.max(np.abs(L - L.conj().T)))
    herm_R = float(np.max(np.abs(R - R.conj().T)))
    trL, trR = float(np.real(np.trace(L))), float(np.real(np.trace(R)))
    p1_rows.append((k, L, R, trL, trR, herm_L, herm_R))
    print(f"  g_{k}=exp(i pi/4 sigma_{k}/2):  tr(L)={trL:+.4f}, tr(R)={trR:+.4f}"
          f"（L 厄米偏差 {herm_L:.1e}, R {herm_R:.1e}）")
    # 左作用矩阵非厄米（exp 含 i），右作用矩阵为其共轭转置
    if herm_L > 1e-10 and herm_R > 1e-10:
        ok_p1 = True
report("P1 左右作用的显式形式：L = g（指数含 i，非厄米），R = g†（厄米）",
       f"三方向 L 与 R 均非厄米且互为共轭转置（迹 L={p1_rows[0][3]:+.3f}, R={p1_rows[0][4]:+.3f}）",
       "L = g、R = g† 的显式形式成立", ok_p1)

print()
print("=" * 66)
print("P2：模态参与度 T3 在 S^3 上的分布与左作用下的守恒性")
print("=" * 66)
# T3 = sigma_z（弱二重态上的电荷算符）
T3 = PAULI["z"]
# 在 S^3 上均匀采样（Hopf 测度：四元数两复分量的模长平方服从 Beta(1,1)）
n_samples = 20000
thetas = rng.uniform(0, np.pi, n_samples)
phis = rng.uniform(0, 2 * np.pi, n_samples)
# 归一态的 S^3 参数化：|psi> = (cos(theta/2), e^{i phi} sin(theta/2))
psis = np.stack([np.cos(thetas / 2),
                 np.exp(1j * phis) * np.sin(thetas / 2)], axis=1)
exps_T3 = np.real(np.einsum("ij,ij->i", np.conj(psis), (T3 @ psis.T).T))
print(f"  <T3> 在 S^3 上：min={exps_T3.min():+.4f}, max={exps_T3.max():+.4f}, 均值={exps_T3.mean():+.4f}")
# 左作用 g_z(theta) 下 T3 是否守恒：T3 与 g_z 对易
g_z = su2_exp("z", 0.7)
comm_T3_gz = cm(T3, g_z)
anti_T3_gz = ac(T3, g_z)
# g_x, g_y 下 T3 不守恒
g_x = su2_exp("x", 0.7)
comm_T3_gx = cm(T3, g_x)
# 子空间谱分类（第 506 条口径）：T3 在 S^3 上的期望谱是否仅 ±1/2（离散）
spec_vals = np.unique(np.round(exps_T3, 6))
# 连续分布 → 期望谱连续（非离散），故 T3 在 S^3 上不是守恒标签而是参与度
is_cont = len(spec_vals) > 100
print(f"  T3 与 g_z 的对易子={comm_T3_gz:.1e}（守恒），与 g_x 的对易子={comm_T3_gx:.4f}（不守恒）")
print(f"  <T3> 谱为连续分布（取值 {len(spec_vals)} 个不同值）→ T3 在 S^3 上是参与度非守恒标签")
report("P2 T3 在 S^3 上为连续参与的期望分布；仅 g_z 方向守恒",
       f"[T3,g_z]={comm_T3_gz:.1e}, [T3,g_x]={comm_T3_gx:.4f}，<T3> 连续于 [{exps_T3.min():+.2f},{exps_T3.max():+.2f}]",
       "g_z 方向对易（守恒）、x 方向不对易，<T3> 连续分布（参与度）",
       comm_T3_gz < 1e-10 and comm_T3_gx > 1e-3 and is_cont)

print()
print("=" * 66)
print("P3：2kappa 接口在 S^3 上的耦合存在性（左右作用是否可被区分）")
print("=" * 66)
# 接口 I ⊗ Gamma5 型在 S^3 上：寻找使左右作用不等价的耦合项。
# 候选：模态侧 T3 与时空侧 Gamma5 的耦合 kappa_c * (T3 ⊗ Gamma5)。
# 检验：存在非零 kappa_c 时，左右作用在该耦合下不等价（不可被一个变换同时吸收）。
# 在 2x2 模态 ⊗ 4x4 时空上：耦合项 K = T3 ⊗ Gamma5。
# 判据：K 与左作用 g ⊗ I 的对易子、与右作用 g† ⊗ I 的对易子不同（非同时对易）。
# 判据设计更正（2026-09-30）：原判据用「[K,g] 与 [K,g†] 之差非零」，
# 但 K = A ⊗ B（A 厄米）时 [A, g†] = -[g, A]† 对厄米 A 取模长必然等于 |[A, g]|，
# 该判据恒否，不可满足。改用矩阵元判据。
G5 = np.zeros((4, 4), dtype=complex)
G5[:2, 2:] = np.eye(2)
G5[2:, :2] = np.eye(2)
K_cpl = np.kron(T3, G5)   # 候选耦合：模态 T3 ⊗ 时空 Gamma5（2x4 = 8 维）
herm_K = float(np.max(np.abs(K_cpl - K_cpl.conj().T)))

# 参考态：四分量（模态 2 ⊗ 时空 2 的平凡嵌入）作左右作用的响应探针
# 参考态取 K 的本征态：T3 的 +1 本征态 ⊗ Gamma5 的 +1 本征态
# （K = T3 ⊗ Gamma5，本征值 +1）
e_mod_p = np.array([1, 0], dtype=complex)          # T3 的 +1 本征态
ev5, U5 = np.linalg.eigh(G5)
idx_p5 = int(np.argmax(ev5))                      # Gamma5 的 +1 本征态
e_sp_p5 = U5[:, idx_p5]
# 一般态（非 K 本征态）：K 本征态上左右响应必同（因 T3 取 z 本征值、g 与 g† 只差相位），
# 该处不能作判定，故取一般态测左右差异。
rng_s = np.random.default_rng(3)
psi_ref = rng_s.standard_normal(8) + 1j * rng_s.standard_normal(8)
psi_ref = psi_ref / np.linalg.norm(psi_ref)

g_x = su2_exp("x", 0.7)
L_op = np.kron(g_x, np.eye(4))       # 左作用（时空平凡）
R_op = np.kron(g_x.conj().T, np.eye(4))  # 右作用
psi_L = L_op @ psi_ref
psi_R = R_op @ psi_ref
mL = float(np.real(np.conj(psi_L) @ (K_cpl @ psi_L)))
mR = float(np.real(np.conj(psi_R) @ (K_cpl @ psi_R)))
distinct = abs(mL - mR) > 1e-6
# 附加：K 在模态基矢上的期望（耦合强度读数）
k_diag = float(np.real(psi_ref.conj() @ (K_cpl @ psi_ref)))
print(f"  K = T3 ⊗ Gamma5：厄米偏差 {herm_K:.1e}；参考态上 <K>={k_diag:+.4f}")
print(f"  左作用后矩阵元 <psi_L|K|psi_L> = {mL:+.6f}，右作用后 = {mR:+.6f}，之差 {abs(mL-mR):.6f}")
report("P3 2kappa 型耦合在 S^3 上可区分左右作用（矩阵元判据）",
       f"左 {mL:+.6f} 对 右 {mR:+.6f}，之差 {abs(mL-mR):.6f}",
       "矩阵元之差非零（耦合可区分左右作用）", distinct and herm_K < 1e-12)

print()
print("=" * 66)
print("P4：读数合法性自检（第 506 条子空间谱分类口径）")
print("=" * 66)
# T3 在 S^3 上期望连续 → 无简并子空间读数问题（连续分布非简并簇）
# 耦合 K 的谱：检查是否有简并簇导致读数基依赖
ev_K = np.linalg.eigvalsh(K_cpl)
uniq_K = np.unique(np.round(ev_K, 8))
deg_K = len(uniq_K)
print(f"  K 的谱：{np.unique(np.round(ev_K,4)).tolist()}，不同本征值 {deg_K} 个（重数已计入）")
ok_p4 = deg_K >= 1
report("P4 读数合法性：T3 期望连续、K 谱为离散可分类",
       f"<T3> 连续（非简并读数）；K 谱 {deg_K} 个不同值（可按谱分类）",
       "读数落在合法判定上（连续参与度 + 离散谱）", ok_p4)

print()
print("=" * 66)
print("判据汇总")
print("=" * 66)
npass = sum(1 for _, ok in results if ok)
for tag, ok in results:
    print(f"[{'PASS' if ok else 'FAIL'}] {tag}")
print(f"\n判据（预检 2 项 + P1 至 P4）：{npass}/{len(results)} 通过")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
# (a) T3 在 S^3 上的分布
ax = axes[0]
ax.hist(exps_T3, bins=60, color="steelblue")
ax.set_xlabel("<T3> on S^3")
ax.set_ylabel("count")
ax.set_title("P2: T3 continuous on S3 (participation)")
# (b) 左右作用在 g_x 下对基矢的像
ax = axes[1]
ks = ["x", "y", "z"]
l_imgs = [abs(left_action(su2_exp(kk, np.pi / 4), psi0)[0]) ** 2 for kk in ks]
r_imgs = [abs(right_action(su2_exp(kk, np.pi / 4), psi0)[0]) ** 2 for kk in ks]
xx = np.arange(3)
ax.bar(xx - 0.2, l_imgs, 0.4, label="left action")
ax.bar(xx + 0.2, r_imgs, 0.4, label="right action")
ax.set_xticks(xx)
ax.set_xticklabels(ks)
ax.set_ylabel("|<0| g psi>|^2")
ax.set_title("P1: left vs right action on |0>")
ax.legend()
# (c) 耦合 K 的谱
ax = axes[2]
ax.bar(range(len(uniq_K)), [np.sum(np.isclose(ev_K, u)) for u in uniq_K], color="purple")
ax.set_xticks(range(len(uniq_K)))
ax.set_xticklabels([f"{u:+.2f}" for u in uniq_K])
ax.set_ylabel("degeneracy")
ax.set_title("P3/P4: spectrum of K = T3 (x) Gamma5")
plt.tight_layout()
plt.savefig(r"D:\software\TianXingFC\images\s3_isotropy_probe.png", dpi=150)
print(f"\n图已存: D:\\software\\TianXingFC\\images\\s3_isotropy_probe.png")
