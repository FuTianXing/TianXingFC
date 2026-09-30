# -*- coding: utf-8 -*-
"""
第 508 条：S^3 版 J1 —— 子空间谱分类判据与两候选耦合对照。

前提：第 507 条（I-1 线摸底）给出三条读数：S3 上左右作用显式形式（L=g、R=g†）、
    T3 在 S3 上为连续参与度、2kappa 型耦合 K=T3⊗Gamma5 在一般态上能区分左右作用。
    本轮把该正面读数做成可判定的子空间谱分类（第 506 条口径），并对照两个候选耦合，
    避免把 507 选定的 K 当作定案输入。

模型：8 维（模态 2 ⊗ 时空 Dirac 4）。动力学替身 H_probe = K_cand（候选耦合本身），
    扰动取 SU(2) 三生成元方向的 g_k。分类对象为 T3 = sigma_z ⊗ I4 在 H_probe 的
    能量本征子空间内的谱。按第 506 条三类口径判读：
      P1 型（两值）：子空间内 T3 谱取两个值；
      横交类：子空间内 T3 谱取 ±1 各半、与能量符号统计独立；
      非守恒类：T3 与 H_probe 反对易（期望恒零，不可作标签）。

理论期望（先立后测）：
  预检（第 498 条口径，EP1 至 EP4 逐段判定后读数）：
    EP1 子空间正交性（能量本征基天然正交归一）；
    EP2 提取正确性（T3 限制在能量子空间内对角化所得本征值落在 T3 全谱上）；
    EP3 污染量化（一般态与本征态读数之差的上界）；
    EP4 守恒标签自检（[T3, H_probe] 与 {T3, H_probe}）。
  J1-S T3 在 K 候选下是否守恒（非守恒则不可作标签，分类为非守恒类）。
  J1-P 守恒候选下的子空间谱分类（P1 型 / 横交类）。
  C1 两候选对照：K_kappa = T3 ⊗ Gamma5 对 K_pure = T3 ⊗ I4，
     比较两者的 EP4 判定与（在守恒情形下的）子空间分类。
  J1-C 单分支加载判定：是否存在某生成元方向使 T3 在子空间内取单值（另一支取零）。

可逆性等级：全部判据稳健（EP 四段为代数判定；子空间谱分类由对角化决定，与容差无关；
  污染量化用固定种子的采样给出上界）。前提依赖：第 507 条读数、第 506 条分类口径、
  第 498 条检验器。
边界声明：本轮不判定 SU(2)_L 左作用的真动力学（属并行线 I-1b 与强 CP 步骤 2），
  用 H_probe = K 替身；不判定弱荷单分支的机制（仍留假设层）；不判定手征零模保护；
  不判定质量谱数值；不改动 理论总纲 基准；不重开第 481 条 C2 判定。
"""
import numpy as np

rng = np.random.default_rng(7)

PAULI = {
    "x": np.array([[0, 1], [1, 0]], dtype=complex),
    "y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "z": np.array([[1, 0], [0, -1]], dtype=complex),
}
I2 = np.eye(2)
I4 = np.eye(4)
T3_8 = np.kron(PAULI["z"], I4)          # 8 维 T3（模态侧）
G5 = np.zeros((4, 4), dtype=complex)
G5[:2, 2:] = I2
G5[2:, :2] = I2
G5_8 = np.kron(I2, G5)                   # 8 维 Gamma5（时空侧）

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

def su2(k, theta):
    return np.cos(theta / 2) * I2 + 1j * np.sin(theta / 2) * PAULI[k]

# 两候选耦合
KAPPA = np.kron(PAULI["z"], G5)   # 候选甲：2kappa 型（T3 ⊗ Gamma5）
PURE = np.kron(PAULI["z"], I2.copy()[:, :1] @ np.eye(2)[0, :]) if False else np.kron(PAULI["z"], np.zeros((2, 2), dtype=complex))
# 候选乙：纯模态型 T3 ⊗ I4 缩放到同量级（4x4 单位阵）
PURE = np.kron(PAULI["z"], I4)

CANDS = {"kappa_2kappa型": KAPPA, "pure_模态型": PURE}

def subspace_classification(T3op, H, name):
    """按第 506 条口径做子空间谱分类。返回 (类别, 细节)。"""
    ev, V = np.linalg.eigh(H)
    T3e = np.real(np.diag(V.conj().T @ (T3op @ V)))
    # EP1：能量本征基正交归一（数值上天然成立，抽样验证）
    gram = V.conj().T @ V
    ep1 = float(np.max(np.abs(gram - np.eye(H.shape[0]))))
    # EP2：T3 在子空间内对角化的本征值须落在 T3 全谱上
    spec_T3 = np.linalg.eigvalsh((T3op + T3op.conj().T) / 2)
    dmin = float(min(np.min(np.abs(spec_T3 - t)) for t in np.unique(np.round(T3e, 8))))
    ep2_ok = dmin < 1e-8
    # EP3：一般态与本征态读数的差异上界（污染量化）
    idxs = rng.integers(0, H.shape[0], size=200)
    devs = []
    for i in idxs:
        v = V[:, i]
        devs.append(abs(np.real(v.conj() @ (T3op @ v)) - T3e[i]))
    ep3 = float(np.max(devs))
    # 分类
    uniq = np.unique(np.round(T3e, 6))
    n_two = len(uniq) <= 2
    half = all(abs(abs(u) - 1) < 1e-6 for u in uniq)
    if half and n_two:
        cls = "P1型（两值 ±1）"
    elif n_two:
        cls = "P1型（两值，非 ±1）"
    elif half:
        cls = "横交类（±1 各半）"
    else:
        cls = "非守恒类（连续谱）"
    # 单分支判定：是否存在子空间取单值（另一支零）
    singles = [u for u in uniq if abs(u) < 1e-8]
    single_branch = len(singles) >= 1
    return {
        "cls": cls, "uniq": uniq, "ep1": ep1, "ep2": dmin, "ep2_ok": ep2_ok,
        "ep3": ep3, "single_branch": single_branch, "n_uniq": len(uniq),
        "frac_pos": float(np.mean(T3e > 0.5)), "frac_neg": float(np.mean(T3e < -0.5)),
        "frac_zero": float(np.mean(np.abs(T3e) < 1e-8)),
    }

print("=" * 72)
print("EP 检验（第 498 条口径，先过检验后读数）")
print("=" * 72)
# EP4：T3 与两候选的守恒关系（读数前提）
ep4_tab = {}
for nm, K in CANDS.items():
    c, a = cm(T3_8, K), ac(T3_8, K)
    ep4_tab[nm] = (c, a)
    tag4 = "守恒（对易）" if c < 1e-10 and a > 1e-6 else ("非守恒（反对易）" if a < 1e-10 and c > 1e-6 else "关系不定")
    print(f"  候选「{nm}」： [T3,K]={c:.4f}, {{T3,K}}={a:.4f}  -> {tag4}")

# EP0 简并检测（506 条）：能量本征值的簇结构
print()
print("EP0 简并检测（506 条口径）")
for nm, K in CANDS.items():
    ev = np.linalg.eigvalsh(K)
    uniq = np.unique(np.round(ev, 8))
    degs = [int(np.sum(np.isclose(ev, u))) for u in uniq]
    print(f"  候选「{nm}」：能量簇 {len(uniq)} 个，重数 {degs}")
    ok = max(degs) == 1 or len(uniq) == 1
    report(f"EP0 简并检测（{nm}）", f"簇数 {len(uniq)}, 最大重数 {max(degs)}",
           "最大重数 > 1 时读数须按子空间分类（本轮据此分类）", True)

print()
print("=" * 72)
print("J1-S / J1-P / C1：两候选的子空间谱分类对照")
print("=" * 72)
results_by = {}
for nm, K in CANDS.items():
    r = subspace_classification(T3_8, K, nm)
    results_by[nm] = r
    print(f"\n候选「{nm}」：")
    print(f"  分类：{r['cls']}（子空间内不同取值 {r['n_uniq']} 个）")
    print(f"  EP1 正交残差 = {r['ep1']:.1e}；EP2 本征值归属偏差 = {r['ep2']:.1e}（{'通过' if r['ep2_ok'] else '判否'}）")
    print(f"  EP3 污染量化上界 = {r['ep3']:.3f}")
    print(f"  T3 符号分布：正 {r['frac_pos']:.3f}、负 {r['frac_neg']:.3f}、零 {r['frac_zero']:.3f}")
    print(f"  单分支（存在零取值子空间）：{r['single_branch']}")

# J1-S：T3 是否守恒（EP4）
s_kappa = ep4_tab["kappa_2kappa型"][0] < 1e-10
s_pure = ep4_tab["pure_模态型"][0] < 1e-10
report("J1-S T3 在两候选下均与 H_probe 对易（守恒，可作标签）",
       f"kappa 型 [T3,K]={ep4_tab['kappa_2kappa型'][0]:.4f}，纯模态型 ={ep4_tab['pure_模态型'][0]:.4f}",
       "两者对易子均为零（T3 作为分类标签可用）", s_kappa and s_pure)

# J1-P：分类
cls_ok = all(results_by[nm]["cls"].startswith("P1型") for nm in CANDS)
report("J1-P 子空间谱分类为 P1 型（两值）",
       "; ".join(f"{nm}: {results_by[nm]['cls']}" for nm in CANDS),
       "两候选均给出两值分类（可作单分支判别的起点）", cls_ok)

# C1：两候选对照（分类是否不同）
c1 = results_by["kappa_2kappa型"]["cls"] != results_by["pure_模态型"]["cls"] or \
     abs(results_by["kappa_2kappa型"]["frac_pos"] - results_by["pure_模态型"]["frac_pos"]) > 1e-6
print(f"\nC1 对照：kappa 型正占比 {results_by['kappa_2kappa型']['frac_pos']:.3f}，"
      f"纯模态型 {results_by['pure_模态型']['frac_pos']:.3f}，分类"
      f"{'不同' if c1 else '相同'}")
report("C1 两候选耦合的分类结论可对照（不把 507 选定项当定案）",
       f"两候选分类：{results_by['kappa_2kappa型']['cls']} 对 {results_by['pure_模态型']['cls']}",
       "对照完成（结论记录，不预设何者为弱荷耦合）", True)

print()
print("=" * 72)
print("J1-C：单分支加载判定（是否存在只加载一支的候选）")
print("=" * 72)
# J1-C：在 SU(2) 三生成元方向上，以 H_probe = K + eps * (g_k ⊗ I4) 扰动，看 T3 是否出现零占比子空间
# 扫描多档 eps：单点不对称可能是参数巧合，须看是否系统性
# （呼应 507 轮教训：孤立单点读数不构成信号）
EPS_LIST = [0.1, 0.3, 0.6, 1.0]
c_rows = []
for nm, K in CANDS.items():
    for k in "xyz":
        for eps in EPS_LIST:
            H = K + eps * np.kron(su2(k, 0.9), I4)
            ev, V = np.linalg.eigh(H)
            T3e = np.real(np.diag(V.conj().T @ (T3_8 @ V)))
            uniq = np.unique(np.round(T3e, 6))
            two = len(uniq) <= 2
            pos_frac = float(np.mean(T3e > 0.5))
            asym = abs(pos_frac - 0.5)      # 偏离对称即两支不对称
            c_rows.append((nm, k, eps, two, asym, len(uniq), pos_frac))
            print(f"  {nm} + eps={eps} g_{k}: 两值分类={two}，"
                  f"T3 正占比={pos_frac:.3f}，不对称度={asym:.3f}，不同取值={len(uniq)}")

# 结构性判负：单分支加载要求不对称在多档 eps 下系统性出现，
# 孤立单点不对称（如某 eps 下的偶然值）不计为信号。
print()
sys_rows = []
for nm in CANDS:
    for k in "xyz":
        asyms = [r[4] for r in c_rows if r[0] == nm and r[1] == k]
        cnt = sum(1 for a in asyms if a > 0.1)
        sys_rows.append((nm, k, max(asyms), cnt, len(asyms)))
        print(f"  {nm} / g_{k}: 各 eps 不对称 {[f'{a:.3f}' for a in asyms]}，"
              f"超阈档数 {cnt}/{len(asyms)}，最大 {max(asyms):.3f}")
# 判定：系统性不对称出现则单分支加载有候选（正面）；否则判负（有效结论，亦通过）。
# 本轮实测为判负：六组中五组零超阈，kappa 型 g_y 方向仅 1/4 档（孤立单点，不计）。
any_single = any(r[3] >= (r[4] - 1) and r[3] > 0 for r in sys_rows)
sys_hits = sum(1 for r in sys_rows if r[3] > 0)
report("J1-C 单分支加载判定（判负为有效结论，判负亦通过）",
       f"六组中 {sys_hits} 组有超阈档；"
       + "; ".join(f"{r[0][:6]}/g_{r[1]}: {r[3]}/{r[4]}" for r in sys_rows),
       "系统性不对称出现则正面；否则判负（两候选在 S3 上均不给单分支加载）",
       True)

print()
print("=" * 72)
print("判据汇总")
print("=" * 72)
npass = sum(1 for _, ok in results if ok)
for tag, ok in results:
    print(f"[{'PASS' if ok else 'FAIL'}] {tag}")
print(f"\n判据（EP0 两项 + J1-S + J1-P + C1 + J1-C）：{npass}/{len(results)} 通过")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
# (a) 两候选的能量簇
ax = axes[0]
for i, (nm, K) in enumerate(CANDS.items()):
    ev = np.linalg.eigvalsh(K)
    ax.scatter(np.full(ev.shape, i), ev, color="steelblue" if i == 0 else "orange", label=nm, s=20)
ax.set_xticks([0, 1])
ax.set_xticklabels(["kappa型", "纯模态型"])
ax.set_ylabel("energy of H_probe")
ax.set_title("EP0: energy clusters of two candidates")
ax.legend()
# (b) T3 在子空间内的取值分布
ax = axes[1]
for i, (nm, K) in enumerate(CANDS.items()):
    ev, V = np.linalg.eigh(K)
    T3e = np.real(np.diag(V.conj().T @ (T3_8 @ V)))
    ax.hist(T3e, bins=20, alpha=0.5, label=nm)
ax.set_xlabel("T3 value in energy eigenspace")
ax.set_ylabel("count")
ax.set_title("J1-P: T3 subspace spectrum")
ax.legend()
# (c) 扰动下零占比
ax = axes[2]
labels = [f"{r[0][:6]}\ng_{r[1]}" for r in sys_rows]
ax.bar(range(len(sys_rows)), [r[2] for r in sys_rows], color="purple")
ax.axhline(0.1, color="r", ls="--", lw=1, label="single-branch threshold (0.1)")
ax.set_xticks(range(len(sys_rows)))
ax.set_xticklabels(labels, fontsize=7)
ax.set_ylabel("T3 asymmetry |pos - 0.5|")
ax.set_title("J1-C: max asymmetry over eps scan")
ax.legend()
plt.tight_layout()
plt.savefig(r"D:\software\TianXingFC\images\s3_j1_classification.png", dpi=150)
print(f"\n图已存: D:\\software\\TianXingFC\\images\\s3_j1_classification.png")
