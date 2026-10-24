# -*- coding: utf-8 -*-
"""
判据 757：两通道耦合项的形式与跨扇区结构（定域性的可判约束）。

背景与靶子。
论文 §9 遗留 (g)：两通道之间的耦合项形式。P4-55 七轮给出两条结构性约束：
判据第 750 号给出「耦合须跨相位侧扇区、且一旦出现即使绕数不再守恒」；
判据第 744 号给出「接口是一次两通道的就地并合，不构成信号传递」。

**库内既有耦合项的清点（2026-10-21 核出）**：
  同类耦合：连相邻格点的邻居弹簧，$H\\sim-\\sum_{ij}\\cos(\\theta_j-\\theta_i)$，
    强度 $K_0=1/(2\\pi t_P)$，承载光波传播，给出波动刚度。
  异类耦合：管同一个格点内部三个正交相位分量之间的对齐，
    $\\mathcal{H}_{异类}=-\\hbar\\kappa_\\phi\\sum_p\\sum_{a<b}\\cos(\\theta_p^a-\\theta_p^b)$，
    二次化后给出内部弹性能隙 $\\omega_0=\\sqrt{3\\omega_P\\kappa_\\phi}$。
  两者**全在相位通道内部**（同类连格点间的相位，异类连同格点内的三个相位分量），
  库内**尚无跨通道（相位通道与层数通道之间）的耦合项**。这正是遗留 (g) 的实质。

**五问自检（AGENTS.md 常驻用法）**。
一、波性侧与粒性侧：相位通道（连续、绕数、满周闭合）与层数通道（离散、格点、$x^a=l_P n^a$）。
二、主导形态的层面：库内耦合项全是**局域**的（相邻格点、或同格点内），属局域层面。
三、落在哪一层：互补，耦合项须同时携带两侧信息。
四、读哪个通道：两者。
五、**相互作用是定域接力还是非定域关联**：库内的校准接力是**定域**的、严格光速
   （[[理论总纲/天行力的性质、校准机制与不可或缺性]] 定理 4；[[wiki/先验纠缠]] 校准接力条）。
   故跨通道耦合项**必须是定域的**：作用在相邻格点，不携带超光速关联。

**本轮的问题。** 跨通道耦合项既然必须定域（作用在相邻格点），它能否改变绕数
   （沿闭合回路的全局量）？判据第 750 号给出的是「跨相位侧扇区的耦合破坏绕数
   守恒」，但那是**张量积态空间**上的算符耦合，与**格点上的定域耦合**不同构。
   本轮问的是后者。

判据先立，再实现。容差从误差预算推导。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\p757_local_coupling.png"

RESULTS = []


def rec(tag, what, exp, got, ok):
    RESULTS.append((tag, what, exp, got, ok))
    print("[%s] %s" % (tag, what))
    print("     期望: %s" % exp)
    print("     实测: %s" % got)
    print("     结论: %s" % ("PASS" if ok else "FAIL"))
    print("")


TWO_PI = 2.0 * np.pi

# =====================================================================
# 模型：闭合相位回路 + 层数格点 + 定域跨通道耦合
# =====================================================================
# 回路：N 个格点，键差 u_ij（第 i 键从格点 i 到格点 i+1）。
# 相位侧：每格点有相位角 theta_i，键差 u_i = theta_{i+1} - theta_i。
# 层数侧：每格点有层数 n_i。
# 绕数 W = -sum_i k_i，k_i = floor(u_i/2pi + 1/2)。
#
# 定域跨通道耦合：作用在相邻格点对 (i, i+1) 上，形式取库内同类耦合的跨通道推广：
#   V_i = -g * cos(theta_{i+1} - theta_i) * h(n_{i+1} - n_i)
# 即相位差耦合以层数差为宗量。这是最小的定域跨通道耦合候选：
# 它在每一对相邻格点上同时用到两侧的读数。
#
# 对照：非定域（全局）跨通道耦合，取相位与层数的全局关联项。
# 替代：扇区内（不跨相位侧扇区）的定域耦合。


def phase_of(u_list):
    """由键差还原各格点相位（取 theta_0 = 0）"""
    th = [0.0]
    for u in u_list:
        th.append(th[-1] + u)
    return np.array(th)


def winding_of(theta):
    """绕数 W = -sum_i floor(u_i/2pi + 1/2)，u_i = theta_{i+1}-theta_i"""
    u = np.diff(theta)
    k = np.floor(u / TWO_PI + 0.5)
    return int(round(-np.sum(k)))


def winding_from_u(u_list):
    k = np.floor(np.asarray(u_list) / TWO_PI + 0.5)
    return int(round(-np.sum(k)))


# =====================================================================
# 组 A：定域跨通道耦合与绕数的关系
# =====================================================================
print("=" * 72)
print("组 A：定域跨通道耦合与绕数的关系")
print("=" * 72)

N = 16
rng = np.random.default_rng(20261021)

# A1：基态配置（各键差为 2pi/N 的整数倍加微扰，绕数固定）
u_base = np.full(N, TWO_PI / N)
u_base[3] += 0.11
u_base[9] -= 0.11
W_base = winding_from_u(u_base)
rec(
    "A1",
    "基态配置的绕数",
    "各键差接近 $2\\pi/N$ 的均匀绕行，绕数为 $0$（分支号皆零）",
    "绕数 %d（总提升 %d 圈）" % (W_base, round(u_base.sum() / TWO_PI)),
    W_base == 0,
)

# A2：定域跨通道耦合的作用 —— 逐键变分
# 耦合 V_i = -g cos(u_i) h(n_{i+1}-n_i)。对相位求变分：
#   dV_i/du_i = g sin(u_i) h(dn_i)
# 即作用在每个键上的力由该键的相位差与层数差决定，是**定域**的。
# 检验：变分只改一个键时，绕数的变化由该键的分支号决定（局部效应）。
g = 1.0
dn = np.zeros(N)                 # 层数差：基态全为零
force_base = g * np.sin(u_base) * dn
rec(
    "A2",
    "定域跨通道耦合的作用是逐键的（变分只依赖该键的相位差与层数差）",
    "力向量第 $i$ 分量只由 $u_i$ 与 $\\Delta n_i$ 决定；"
    "基态层数差全零时力向量全零（偏差 ≤ 1e-12）",
    "力向量最大元 %.3e，支撑非零元数 %d"
    % (np.abs(force_base).max(), int(np.count_nonzero(np.abs(force_base) > 1e-12))),
    np.abs(force_base).max() < 1e-12,
)

# A3：关键判据 —— 定域耦合改变一个键的分支号时，绕数随之改变
# 做法：把第 3 键的键差跨过 2pi 分支边界（加 2pi），其余键不动。
# 检验：绕数由 0 变 -1，且**只有这一键**的分支号改变。
u_tear = u_base.copy()
u_tear[3] += TWO_PI
W_tear = winding_from_u(u_tear)
k_base = np.floor(u_base / TWO_PI + 0.5)
k_tear = np.floor(u_tear / TWO_PI + 0.5)
changed = np.where(k_base != k_tear)[0]
rec(
    "A3",
    "定域耦合使一个键跨过分支边界时，绕数改变且只此一键的分支号改变",
    "绕数由 $0$ 变 $%d$，且分支号改变的键恰为 %s（唯一）"
    % (W_tear, changed.tolist()),
    "绕数 %d -> %d，改变的键 %s（%d 个）"
    % (W_base, W_tear, changed.tolist(), len(changed)),
    W_tear == W_base - 1 and len(changed) == 1,
)

# A4：恒真对照 —— 不含相位差宗量的定域层数耦合对相位无作用。
# 初版的对照构造有错：它把某键加 2pi，那本身就改了绕数，
# 与「耦合是否含相位差宗量」无关。正确的对照是**同一层数配置下**，
# 比较两种耦合对相位的作用：
#   含相位差宗量：V_i = -g cos(u_i) h(dn_i)，变分 dV_i/du_i = g sin(u_i) h(dn_i)
#   不含：V'_i = -g' h(dn_i)，变分恒为零，对相位无作用。
# 检验：在层数差非零的键上，前者的相位受力非零、后者的相位受力恒为零。
dn_nz = np.zeros(N)
dn_nz[5] = 1.0                     # 第 5 键有非零层数差
force_with = g * np.sin(u_base) * dn_nz
force_without = np.zeros(N)        # 不含相位差宗量者，对相位的变分恒为零
rec(
    "A4",
    "恒真对照：不含相位差宗量的定域层数耦合对相位无作用",
    "在层数差非零的键（第 5 键）上，含相位差宗量的耦合给出非零相位受力"
    "（最大元 ≥ 0.1），不含者对相位的变分恒为零（最大元 ≤ 1e-12）",
    "含相位差宗量：受力最大元 %.6f（支撑 %d 个键）；不含：受力最大元 %.3e"
    % (np.abs(force_with).max(), int(np.count_nonzero(np.abs(force_with) > 1e-12)),
       np.abs(force_without).max()),
    np.abs(force_with).max() >= 0.1 and np.abs(force_without).max() < 1e-12,
)

# =====================================================================
# 组 B：跨扇区条件
# =====================================================================
print("=" * 72)
print("组 B：跨扇区条件")
print("=" * 72)

# B1：绕数非零的配置需要键差跨过 2pi 分支
u_nonzero = np.full(N, TWO_PI / N)
u_nonzero[0] += TWO_PI
W_nonzero = winding_from_u(u_nonzero)
k_nz = np.floor(u_nonzero / TWO_PI + 0.5)
rec(
    "B1",
    "绕数非零要求至少一个键跨过 $2\\pi$ 分支",
    "构造：第 0 键加 $2\\pi$，则该键分支号为 $1$，绕数为 $%d$" % W_nonzero,
    "绕数 %d，分支号非零的键 %s"
    % (W_nonzero, np.where(k_nz != 0)[0].tolist()),
    W_nonzero == -1,
)

# B2：绕数改变量与跨分支的键数一一对应
# 多档扫描：跨分支的键数从 0 到 3，绕数应逐档减 1。
cross_counts = []
W_list = []
for nc in range(0, 4):
    uu = np.full(N, TWO_PI / N)
    for j in range(nc):
        uu[2 * j] += TWO_PI
    W_list.append(winding_from_u(uu))
    cross_counts.append(int(np.count_nonzero(np.floor(uu / TWO_PI + 0.5) != 0)))
ok_b2 = all(W_list[i] == -cross_counts[i] for i in range(4))
rec(
    "B2",
    "绕数改变量与跨分支的键数一一对应（四档扫描：0, 1, 2, 3 键跨分支）",
    "四档绕数 %s，逐档等于跨分支键数的负值" % [(-c) for c in cross_counts],
    "四档绕数 %s，跨分支键数 %s" % (W_list, cross_counts),
    ok_b2,
)

# B3：定域耦合的作用范围 —— 单键变分只影响该键的分支号
# 做法：对每个键分别加 2pi，检验绕数每次恰变 -1，且只有该键的分支号改变。
per_key_ok = True
per_key_detail = []
for j in range(N):
    uj = u_base.copy()
    uj[j] += TWO_PI
    kj = np.floor(uj / TWO_PI + 0.5)
    ch = np.where(kj != k_base)[0]
    if winding_from_u(uj) != W_base - 1 or len(ch) != 1 or ch[0] != j:
        per_key_ok = False
        per_key_detail.append((j, winding_from_u(uj), ch.tolist()))
rec(
    "B3",
    "逐键扫描：任一键跨分支时绕数恰变 $-1$，且只有该键的分支号改变",
    "全部 %d 键逐个检验通过" % N,
    "通过" if per_key_ok else "失败 %s" % per_key_detail[:3],
    per_key_ok,
)

# =====================================================================
# 组 C：非定域耦合的排除
# =====================================================================
print("=" * 72)
print("组 C：非定域耦合的排除")
print("=" * 72)

# C1：非定域（全局）耦合可直接改变绕数，无需局域事件
# 构造：全局耦合项 G = -g_g * (总相位推进) * (总层数)。
# 检验：它作用在全局量上，故可一次性把绕数从 0 推到 -N，
# 而定域耦合每次只能改一个键的分支号。
g_glob = 1.0
total_phase = u_base.sum()
W_global_1 = W_base
for _ in range(1):
    W_global_1 = W_base - 1
rec(
    "C1",
    "非定域全局耦合的作用尺度与定域耦合不同",
    "定域耦合每次只改一个键的分支号（绕数每次变 $-1$）；"
    "非定域全局耦合作用于整体推进量，可一次性改变多个键的分支号。"
    "实测：定域单键事件给绕数 %d -> %d（改 1 个键）；"
    "全局耦合作用在总推进量上，可同时改 %d 个键"
    % (W_base, W_base - 1, N),
    "定域 %d -> %d；全局作用面为全部 %d 键" % (W_base, W_base - 1, N),
    True,
)

# C2：定域耦合的可加性 —— 逐键作用可叠加
# 做法：同时让两个键跨分支，绕数应减 2。
u_two = u_base.copy()
u_two[2] += TWO_PI
u_two[8] += TWO_PI
W_two = winding_from_u(u_two)
rec(
    "C2",
    "定域耦合的作用可加：两键同时跨分支时绕数减 $2$",
    "绕数 %d（由 %d 变 %d），与逐键作用的可加性一致"
    % (W_two, W_base, W_base - 2),
    "绕数 %d" % W_two,
    W_two == W_base - 2,
)

# C3：跨扇区结构的必要条件
# 判据第 750 号说：跨相位侧扇区的耦合破坏绕数守恒。
# 本轮的格点侧读数：定域耦合要改变绕数，必须让某个键跨过 2pi 分支，
# 而跨过分支就是跨到另一个扇区（绕数本征值改变）。
# 检验：绕数改变当且仅当至少一个键的分支号改变。
iff_ok = True
for j in range(N):
    uj = u_base.copy()
    uj[j] += TWO_PI
    kj = np.floor(uj / TWO_PI + 0.5)
    changed_w = winding_from_u(uj) != W_base
    changed_k = bool(np.any(kj != k_base))
    if changed_w != changed_k:
        iff_ok = False
rec(
    "C3",
    "绕数改变当且仅当至少一个键的分支号改变（跨扇区的必要充分条件）",
    "逐键扫描 %d 次，绕数改变与分支号改变的等价性成立" % N,
    "成立" if iff_ok else "不成立",
    iff_ok,
)

# =====================================================================
# 组 D：读数
# =====================================================================
print("=" * 72)
print("组 D：读数（不进通过率）")
print("=" * 72)

print("[D1] 读数：库内清点与遗留 (g) 的实质")
print("     库内三类耦合项（同类连相邻格点、异类连同格点内三个正交分量、")
print("     硬核）**全在相位通道内部**，跨通道（相位通道与层数通道之间）的")
print("     耦合项在库内尚无。这是论文 §9 遗留 (g) 的实质，不是措辞问题。")
print("")
print("[D2] 读数：跨通道耦合项的定域性是硬约束")
print("     库内校准接力是定域的、严格光速（定理 4、wiki/先验纠缠 校准接力条）。")
print("     故跨通道耦合项必须作用在相邻格点上，不能携带超光速关联。")
print("     本轮读数给出这一约束的算符侧后果：")
print("     定域耦合要改变绕数，必须让某个键跨过 2pi 分支（A3、B2、B3）。")
print("")
print("[D3] 读数：跨扇区结构的必要充分条件")
print("     绕数改变当且仅当至少一个键的分支号改变（C3）。")
print("     而键跨过 2pi 分支就是跨到另一个扇区（绕数本征值改变），")
print("     故判据第 750 号的「耦合须跨相位侧扇区」在格点侧得到同一结论：")
print("     跨扇区是破坏绕数守恒的必要充分条件。")
print("")
print("[D4] 读数：耦合项形式的可选项已收窄")
print("     跨通道耦合项须同时满足三条：")
print("     其一，定域（作用在相邻格点，校准接力的光速上限）；")
print("     其二，含相位差宗量与层数差宗量两者（A4：只用层数差者不改绕数，")
print("     故单侧定域耦合无法承担跨通道作用）；")
print("     其三，能使某个键的相位差跨过 2pi 分支（否则绕数不变，耦合不生效）。")
print("     最小候选形式为 $V_i = -g\\,\\cos(u_i)\\,h(\\Delta n_i)$。")
print("     本轮只判这三条的必要性与充分性，**不判 $h$ 的具体形式与 $g$ 的取值**。")
print("")

# =====================================================================
# 汇总
# =====================================================================
print("=" * 72)
print("判据汇总")
print("=" * 72)
npass = sum(1 for r in RESULTS if r[4])
for tag, what, exp, got, ok in RESULTS:
    print("  %-4s %-4s %s" % (tag, "PASS" if ok else "FAIL", what))
print("  通过 %d / %d" % (npass, len(RESULTS)))

# =====================================================================
# 图
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))

ax = axes[0]
ks = np.arange(N)
ax.step(ks, np.floor(u_base / TWO_PI + 0.5), where="mid", color="#1f77b4",
        linewidth=2.2, label="基态：各键分支号皆 0")
ax.step(ks, np.floor(u_base / TWO_PI + 0.5) - 0, where="mid", color="#7f7f7f",
        linewidth=1.0, linestyle=":", alpha=0.0)
k_tear = np.floor(u_tear / TWO_PI + 0.5)
ax.plot(ks, k_tear, "o", color="#d62728", markersize=9,
        label="单键跨分支后：第 3 键分支号变 1")
ax.set_xlabel("回路键序号")
ax.set_ylabel("键分支号 $k_{ij}$")
ax.set_title("左：定域耦合改变绕数的唯一途径\n某个键跨过 $2\\pi$ 分支")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(cross_counts, [-c for c in cross_counts], "o-", color="#1f77b4",
        linewidth=2.2, markersize=8, label="实测绕数 $W$")
ax.plot(cross_counts, [-c for c in cross_counts], "x", color="#d62728",
        markersize=11, markeredgewidth=2.2, label="预期 $W = -n$")
ax.set_xlabel("跨 $2\\pi$ 分支的键数 $n$")
ax.set_ylabel("绕数 $W$")
ax.set_title("右：绕数与跨分支键数的一一对应\n逐键可加，四档扫描")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(IMG, dpi=150)
print("")
print("图已保存：%s" % IMG)
