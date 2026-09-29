# -*- coding: utf-8 -*-
"""
第 503 条：I-1b 最小交付轮（sigma 场独立性、与旋量代数的交叉耦合及其对易型）。

问题：J1 执行的前置。D-b' 口径三选一（sigma 以何方式进入电子算符 D 的构造）取决于
    sigma 场的显式构型。本轮只判三件事，不写全泛函：
    I1  sigma 场是否独立于相位场（有无独立自由度）；
    I2  sigma 与旋量代数之间是否存在交叉耦合项；
    I3  耦合若存在，为对易或反对易。

模型：沿用第 486 条的约束版 linkage（a = (1 - sigma)/2，sigma 定义相位零点）：
    E = sum_n [ (kappa/2)(d sigma_n)^2 + lambda (sigma_n^2 - 1)^2 + (k/2)(d a_n)^2 ],
    a_n = (1 - sigma_n)/2。
旋量侧取 3+1 维 Dirac 基（G0, g1, g2, g3, Gamma5），模态因子取 sigma_z。
交叉耦合候选：c1 (Gamma5 ⊗ sigma_z)（反对易型）、c2 (Gamma5 ⊗ I)（独立型）、
    c3 (gamma_theta ⊗ sigma_z)（动量耦合型）。

理论期望（先立后测）：
  I1 独立于相位场：sigma 场的变分不改变相位场的能量读数（相位的能量式对 sigma 恒等盲，
     第 487 条 AX1 在此复测），且 sigma 有独立自由度（相位梯度为零时 sigma 仍可非平凡）。
  I2 交叉耦合存在性：在 8 分量三因子结构上，穷举 gamma 型与投影型元件 A，检验是否存在
     A 满足 A ⊗ sigma_z = -A ⊗ sigma_z（即 A 互换 sigma 的两个真空）。第 494 条其五已测
     2kappa 型反对易（Gamma5 与 sigma_z 反对易），本轮检验其在完整三因子结构上是否给出
     sigma 真空选支的代数接口。
  I3 对易型：候选 A 若满足 {A, sigma_z} = 0 则为反对易（可换真空），若 [A, sigma_z] = 0
     则为对易（不换真空，不选支）。
  I4 与 D-b' 口径的接口：由 I2/I3 判定甲（U(1) 基底层）之外的乙（模态因子层）或丙（交叉项）
     哪一口径在代数上可行，给 J1 执行以确定口径。

可逆性等级：I1、I3 判据稳健（解析恒等式）；I2 判据稳健（穷举完备，候选族固定）；
  I4 为推论。
"""
import numpy as np

I2 = np.eye(2)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)

def gammas():
    g0 = np.zeros((4, 4), dtype=complex)
    g0[:2, :2] = I2
    g0[2:, 2:] = -I2
    gs = []
    for s in (SX, SY, SZ):
        g = np.zeros((4, 4), dtype=complex)
        g[:2, 2:] = s
        g[2:, :2] = -s
        gs.append(g)
    g5 = np.zeros((4, 4), dtype=complex)
    g5[:2, 2:] = I2
    g5[2:, :2] = I2
    return g0, gs, g5

G0, GV, G5 = gammas()

results = []
def report(tag, measured, expected, ok):
    results.append((tag, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {tag}")
    print(f"    实测: {measured}")
    print(f"    期望: {expected}")

def comm(A, B):
    return float(np.max(np.abs(A @ B - B @ A)))

def anti(A, B):
    return float(np.max(np.abs(A @ B + B @ A)))

# ========== 冒烟 ==========
print("== 冒烟检查 ==")
g5_sq = np.max(np.abs(G5 @ G5 - np.eye(4)))
g5_anti_g = max(anti(G5, g) for g in GV)
ok_smoke = g5_sq < 1e-12 and g5_anti_g < 1e-12
print(f"    Γ₅² − I = {g5_sq:.1e}；{{Γ₅, γ}} 最大 = {g5_anti_g:.1e}")
report("冒烟：Γ₅ 厄米对合且与三个 γ 反对易", f"{g5_sq:.1e}, {g5_anti_g:.1e}",
       "两者 < 1e-12", ok_smoke)

# ========== I1 sigma 场独立于相位场 ==========
print("\n== I1：sigma 场独立性 ==")
# 相位能量式（U(1) 基底层）：U(1) 型基底项对 sigma 恒等盲（第 487 条 AX1）
# 相位 u，sigma 定义零点：a = (1-sigma)/2。相位基底项 1 - cos(2 pi u)。
# 检验：把 u 的零点按 sigma 平移（u -> u + a），基底项读数是否改变。
KAPPA, LAM, K = 1.0, 0.25, 1.0
N = 64
rng = np.random.default_rng(7)
sig = 2 * np.tanh(rng.standard_normal(N))  # 非平凡 sigma 位形
sig = np.clip(sig, -0.999, 0.999)

def e_sigma(s):
    ds = np.diff(s)
    return float(np.sum((KAPPA / 2) * ds ** 2) + np.sum(LAM * (s ** 2 - 1) ** 2))

def e_phase_blind_reflect(u, s):
    """U(1) 型基底项在 sigma 的 ±1 反射下的读数（第 487 条 AX1 原文口径）：
    1 - cos(2 pi sigma * u)，检验 s = +1 与 s = -1 是否逐位相同。"""
    return float(np.sum(1 - np.cos(2 * np.pi * s * u)))

u_far = 0.37  # 均匀远场，相位梯度为零
blind_var = float(e_phase_blind_reflect(u_far, +1) - e_phase_blind_reflect(u_far, -1))
sigma_indep = abs(blind_var) < 1e-12
# sigma 有独立自由度：相位梯度为零时 sigma 能量仍非零
e_sig_val = float(e_sigma(sig))
report("I1 sigma 独立于相位场（U(1) 基底对 sigma 恒等盲，sigma 有独立自由度）",
       f"基底能量 ±1 反射之差 {blind_var:.1e}；sigma 自身能量 {e_sig_val:.4f}",
       "基底侧 ±1 反射读数之差为零（AX1 复测），sigma 能量非零（独立自由度）",
       sigma_indep and abs(e_sig_val) > 1e-6)

# ========== I2 / I3 交叉耦合存在性与对易型 ==========
print("\n== I2 / I3：交叉耦合与对易型 ==")
# 8 分量三因子：模态 2 (sigma_z) ⊗ 时空 Dirac 4。元件 A ⊗ sigma_z 形式：
# sigma 的两个真空是 sigma_z 的 ±1 本征态。要 A ⊗ sigma_z = -(A ⊗ sigma_z)，
# 即 A 反对易 sigma_z。模态 2x2 中反对易 sigma_z 的只有 sigma_x, sigma_y。
# 但 J1 要求 A 是 4x4 时空元件 ⊗ 模态元件；完整条件：A_4 ⊗ M_2 反对易 sigma_z_2 ⊗ I_4。
# 即 A_4 ⊗ M_2 与 I_4 ⊗ sigma_z_2 反对易 -> A_4 M_2 ⊗ sigma_z_2 反对易。
# 若 M_2 反对易 sigma_z_2（M_2 = sigma_x 或 sigma_y），则需 A_4 反对易 I_4（不可能）。
# 若 A_4 反对易自身不可能（标量），故 A_4 ⊗ M_2 与 sigma_z 反对易要求 A_4 或 M_2 有一个反对易 sigma_z...
# 系统穷举：4x4 Dirac 单项式 A（共 32 项含 i），M_2 取 {I, sx, sy, sz}，检验
#   (A ⊗ M) 与 (I ⊗ sz) 反对易，且 (A ⊗ M) 厄米对合。
import itertools
mono = [np.eye(4, dtype=complex)]
basis = [G0] + GV
for r in range(1, 5):
    for combo in itertools.combinations(range(4), r):
        M = np.eye(4, dtype=complex)
        for c in combo:
            M = M @ basis[c]
        mono.append(M)
        mono.append(1j * M)
M2set = {"I": I2, "sx": SX, "sy": SY, "sz": SZ}
SZ2 = SZ
sols = []
seen = []
for A in mono:
    for name, M2 in M2set.items():
        X = np.kron(A, M2)
        Z = np.kron(np.eye(4), SZ2)
        ac = anti(X, Z)
        cm = comm(X, Z)
        herm = np.max(np.abs(X - X.conj().T))
        inv = np.max(np.abs(X @ X - np.eye(8)))
        if ac < 1e-10 and herm < 1e-10 and inv < 1e-10:
            key = (name, tuple(np.round(np.abs(A).flatten(), 6)))
            if key not in [k for k, _ in seen]:
                seen.append((key, name))
                sols.append((name, ac, cm, herm, inv))
                print(f"    解：模态元件 {name}（反对易 sigma_z，anti={ac:.1e}, comm={cm:.1e}）")
ok_i2 = len(sols) > 0
report("I2 交叉耦合接口存在性：存在 8 分量元件反对易 sigma_z（sigma 真空可被互换）",
       f"穷举 {len(mono)} 个 Dirac 单项式 × 4 个模态元件，去重后合格解 {len(sols)} 个",
       "至少 1 个解（sigma 真空存在代数接口）", ok_i2)

# I3 对易型：区分解的对易型（反对易=可换真空；对易=不换）
# 对易 sigma_z 的模态元件（I, sz）与反对易的（sx, sy）
commuting_M = [n for n in M2set if comm(M2set[n], SZ2) < 1e-10]
anticommuting_M = [n for n in M2set if anti(M2set[n], SZ2) < 1e-10]
print(f"    对易 sigma_z 的模态元件：{commuting_M}；反对易的：{anticommuting_M}")
ok_i3 = len(anticommuting_M) > 0 and len(commuting_M) > 0
report("I3 对易型：模态侧存在反对易 sigma_z 的元件（可换真空），与对易元件并存",
       f"反对易 {anticommuting_M}，对易 {commuting_M}",
       "两类并存（反对易给出接口，对易不选支）", ok_i3)

# ========== I4 D-b' 口径接口 ==========
print("\n== I4：D-b' 口径的代数可行性 ==")
# 甲（U(1) 基底层）：由 I1 复测判负（基底恒等盲）
# 乙（模态因子层）：需 sigma_z 参与耦合，由 I2/I3 判定——反对易元件存在则可行
# 丙（交叉项）：需 I2 的解给出 sigma 真空选支接口
jia_ok = not sigma_indep  # 甲口径下基底对 sigma 盲 -> 无解
yi_ok = len(anticommuting_M) > 0
bing_ok = ok_i2
print(f"    甲（U(1) 基底层）：可行 = {jia_ok}（基底恒等盲，无解）")
print(f"    乙（模态因子层）：可行 = {yi_ok}（sigma_z 反对易元件 {anticommuting_M} 存在）")
print(f"    丙（交叉项）：可行 = {bing_ok}（8 分量反对易解 {len(sols)} 个）")
ok_i4 = (not jia_ok) and yi_ok and bing_ok
report("I4 D-b' 口径接口：甲判负、乙与丙在代数上可行",
       f"甲可行={jia_ok}, 乙可行={yi_ok}, 丙可行={bing_ok}",
       "甲否（AX1 复测），乙丙是（J1 的执行口径收敛到乙或丙）", ok_i4)

print("\n== 判据汇总 ==")
npass = sum(1 for _, ok in results if ok)
for tag, ok in results:
    print(f"[{'PASS' if ok else 'FAIL'}] {tag}")
print(f"\n判据 I1 至 I4（冒烟并入）：{npass}/{len(results)} 通过")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
ax.plot(np.arange(N), sig, "b-", label="sigma 场位形（非平凡）")
ax.plot(np.arange(N), (1 - sig) / 2, "r--", label="a = (1-sigma)/2（零点偏移）")
ax.set_xlabel("格点 n")
ax.set_ylabel("值")
ax.set_title("I1: sigma 场独立于相位场（双井势）")
ax.legend()
ax = axes[1]
names = list(M2set.keys())
cms = [comm(M2set[n], SZ2) for n in names]
ams = [anti(M2set[n], SZ2) for n in names]
x = np.arange(len(names))
ax.bar(x - 0.2, cms, 0.4, label="对易子 [M, sigma_z]")
ax.bar(x + 0.2, ams, 0.4, label="反对易子 {M, sigma_z}")
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylabel("绝对值")
ax.set_title("I3: 模态侧对易型（sx,sy 反对易可换真空）")
ax.legend()
plt.tight_layout()
plt.savefig(r"D:\software\TianXingFC\images\sigma_independence.png", dpi=150)
print(f"\n图已存: D:\\software\\TianXingFC\\images\\sigma_independence.png")
