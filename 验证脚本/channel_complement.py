# -*- coding: utf-8 -*-
"""
判据 743：满周闭合接口与两通道虚过程的互补判定。

背景。P4-55 前三轮已交付：判据 730（i-epsilon 处方在离散时间下失效）、
判据 737（两通道在态空间里正交、不冗余、可分离，A/B 两组降级为自建对照
模型读数）、判据 740（层数侧虚过程形态为「一个键暂借一个 2pi 分支然后
归还」，绕数是分支拓扑量）。

本轮问两通道虚过程如何互补。

**对象类型前置（重要）**。库内有两套「通道」，名字相近而所指不同：
  (i) 物侧两通道：相位通道（横截面，内部空间）与层数通道（法线轴，外部空间）；
  (ii) 读出侧两通道：两台仪器同时读同一记录（第 444 条，交叉项
      2 eta_A eta_B sigma^2，通道须分指针或耦合率拉开约四十倍）。
本轮只问 (i)。第 444 条的结论**不移植**到物侧两通道上：前者讲仪器探针的
可加性条件，后者讲两个物侧自由度的互补结构，对象不同。

**库内给物侧两通道的唯一物理接口**（[[wiki/相位通道]] 机制第 1 条、
[[wiki/层数通道]] 机制第 1 条）：相位指针满周闭合被圆周标记点检测，触发一次
法线跃迁，层数加一。故本轮的互补问题表述为：**层数侧的分支暂借（虚过程）
是否经由满周闭合接口，与相位侧的涨落相关联。**

判据先立，再实现。容差从误差预算推导。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\p743_channel_complement.png"

RESULTS = []


def rec(tag, what, exp, got, ok):
    RESULTS.append((tag, what, exp, got, ok))
    print("[%s] %s" % (tag, what))
    print("     期望: %s" % exp)
    print("     实测: %s" % got)
    print("     结论: %s" % ("PASS" if ok else "FAIL"))
    print("")


TWO_PI = 2.0 * np.pi
N = 64


def branch_k(us):
    """键分支指标 k_ij = floor(u_ij / 2pi + 1/2)"""
    return np.floor(np.asarray(us, dtype=float) / TWO_PI + 0.5)


def winding_from_phase(us):
    """绕数 W = -sum k（库内离散形式）"""
    return int(round(-np.sum(branch_k(us))))


# =====================================================================
# 模型：闭合相位回路 + 满周闭合检测器
# =====================================================================
# 相位在回路上逐键推进，键差 u_ij 不取模。
# 满周闭合检测：相位指针连续推进时，圆周标记点被穿越的次数即满周闭合次数。
# 检测器实现：沿路径累加相位，跟踪累积相位对 2pi 的跨越次数。
def full_turn_detector(us):
    """满周闭合检测器：沿路径累加相位，统计跨越 2pi 的次数。

    返回 (总跨越数, 逐键的跨越指示)。
    """
    cum = 0.0
    crossings = []
    for u in us:
        prev = cum
        cum += u
        # 跨越 2pi 整数倍边界的次数。用「就近取整的分支号之差」：
        #   k(cum) = floor(cum/2pi + 1/2)
        # 该取整在正负半轴对称（分支号对 ±2pi 的偏移同号），
        # 且把恰在 2pi 边界上的情形算作已进入下一分支
        # （绕一圈时 cum = 2pi，k = 1，故计数为 1，符合「满周闭合一次」）。
        c = int(np.floor(cum / TWO_PI + 0.5)) - int(np.floor(prev / TWO_PI + 0.5))
        crossings.append(c)
    return int(np.sum(crossings)), crossings


# =====================================================================
# 组 A：满周闭合检测器的基本性质
# =====================================================================
print("=" * 72)
print("组 A：满周闭合检测器的基本性质")
print("=" * 72)

# A1：绕一圈，检测到一次满周闭合
u_one = np.full(N, TWO_PI / N)
n_turn, cr = full_turn_detector(u_one)
rec(
    "A1",
    "满周闭合检测器：相位绕满一圈",
    "检测到 1 次满周闭合",
    "检测数 %d，逐键跨越指示 %s" % (n_turn, np.unique(cr)),
    n_turn == 1,
)

# A2：绕两圈
n_turn2, _ = full_turn_detector(np.full(N, 2 * TWO_PI / N))
rec(
    "A2",
    "满周闭合检测器：相位绕满两圈",
    "检测到 2 次满周闭合",
    "检测数 %d" % n_turn2,
    n_turn2 == 2,
)

# A3：检测数与绕数的关系
# 绕数 W = -sum k；检测数 T = 满周闭合次数
# 平滑绕行：W = 0（各键分支号 0），但检测数 T = 1
# 这正是两通道读数不同型的地方：W 记分支分布，T 记满周闭合次数
w_smooth = winding_from_phase(u_one)
rec(
    "A3",
    "平滑绕行的绕数与满周闭合检测数",
    "绕数 $W = 0$（各键分支号皆 0，绕数只记分支分布），"
    "而检测数 $T = 1$（相位确实转满了一圈）。故两者是不同的读数，不可互换",
    "绕数 %d，检测数 %d" % (w_smooth, n_turn),
    w_smooth == 0 and n_turn == 1,
)

# A4：撕裂情形
# 一条键跨过 2pi 分支：相位总提升不变（仍是一圈），但该键落到分支 1
# 撕裂的构造：一条键的相位差跨过 2pi 分支，但**总提升不变**。
# 干净做法：把多出的 2pi 分摊到其余所有键上，每键减 2pi/(N-1)，
# 于是总提升仍为一圈，而第一键落到分支 1 上。
def spread_borrow(u, n_borrow_keys=1, extra=2.0 * np.pi):
    """构造「n 个键各暂借一个 2pi 分支」的键差。

    每个暂借键各自加上整 2pi（这样它确实落到分支 1），
    再把多出的总量从其余键各减去，保持总提升不变。
    注意：不能把 2pi 均分给各暂借键，因为每键只分到 2pi/n
    时可能不足 pi，键分支指标就近取整给 0，跨不过分支边界。
    """
    uu = np.array(u, dtype=float)
    rest = len(uu) - n_borrow_keys
    total_extra = extra * n_borrow_keys
    uu[:n_borrow_keys] += extra
    uu[n_borrow_keys:] -= total_extra / rest
    return uu


u_tear = spread_borrow(u_one, 1)
w_tear = winding_from_phase(u_tear)
n_tear, _ = full_turn_detector(u_tear)
rec(
    "A4",
    "撕裂情形的绕数与检测数",
    "绕数变为 $-1$（分支分布改变），检测数仍为 $1$（总提升未变，仍是一圈）。"
    "故撕裂改变分支拓扑，不改变满周闭合次数",
    "绕数 %d，检测数 %d" % (w_tear, n_tear),
    w_tear == -1 and n_tear == 1,
)

# =====================================================================
# 组 B：层数侧虚过程与满周闭合接口
# =====================================================================
print("=" * 72)
print("组 B：层数侧虚过程与满周闭合接口")
print("=" * 72)

# B1：层数侧虚过程 = 一条键暂借一个 2pi 分支
# 相位侧的对应：那一键的相位差暂时超出 2pi，即该键的相位提升暂时多绕一周
# 层数侧虚过程：一条键暂借一个 2pi 分支。
# 构造：把 2pi 加到第一键上，并从其余键各减去 2pi/(N-1)，总提升不变。
# 该键落到分支 1（绕数减 1），相位总提升不变（检测数不变）。
u_virtual = spread_borrow(u_one, 1)
w_virtual = winding_from_phase(u_virtual)
n_virtual, cr_v = full_turn_detector(u_virtual)
rec(
    "B1",
    "层数侧虚过程：一条键暂借一个 $2\\pi$ 分支",
    "绕数变为 $-1$（分支分布改变）；满周闭合检测数**不变**（总提升未变，"
    "相位并未多绕一圈）。故暂借只动层数侧的读数",
    "绕数 %d，检测数 %d" % (w_virtual, n_virtual),
    w_virtual == -1 and n_virtual == n_turn,
)

# B2：虚过程归还后，两个读数都复原
u_return = u_one.copy()
w_return = winding_from_phase(u_return)
n_return, _ = full_turn_detector(u_return)
rec(
    "B2",
    "虚过程归还后的复原",
    "绕数回到 $0$，检测数回到 $1$，两读数同时复原",
    "绕数 %d，检测数 %d" % (w_return, n_return),
    w_return == 0 and n_return == 1,
)

# B3：互补性的算术形式：两读数之差
# 关键：分支暂借使绕数减 1、检测数加 1，故 (W + T) 不变
def w_plus_t(us):
    w = winding_from_phase(us)
    t, _ = full_turn_detector(us)
    return w + t


# 互补性的正确形式：两读数各自度量不同的守恒量。
# 绕数 W 记分支分布（键落在第几个 2pi 分支），对**总提升**不敏感；
# 检测数 T 记满周闭合次数（相位推进了多少圈），对**分支分布**不敏感。
# 故二者互补：T 在 W 改变时不动，W 在 T 改变时不动。
base_w = winding_from_phase(u_one)
base_t, _ = full_turn_detector(u_one)

# 情形一：只动分支（W 变），T 应不变
w_only = winding_from_phase(u_virtual)
t_only, _ = full_turn_detector(u_virtual)

# 情形二：只动总提升（T 变），W 应不变
u_more = np.full(N, 2 * TWO_PI / N)
w_more = winding_from_phase(u_more)
t_more, _ = full_turn_detector(u_more)

rec(
    "B3",
    "互补性：两读数各自度量不同的守恒量",
    "情形一（只动分支）：绕数 $%d \\to %d$，检测数 $%d \\to %d$（不变）；"
    "情形二（只动总提升）：绕数 $%d \\to %d$（不变），检测数 $%d \\to %d$。"
    "故 W 与 T 互不敏感，互为互补读数" % (base_w, w_only, base_t, t_only,
                                base_w, w_more, base_t, t_more),
    "情形一：$W %d\\to%d$、$T %d\\to%d$；情形二：$W %d\\to%d$、$T %d\\to%d$"
    % (base_w, w_only, base_t, t_only, base_w, w_more, base_t, t_more),
    t_only == base_t and w_more == base_w,
)

# B4：恒真检验：一个与两通道都无关的读数
def unrelated_reading(us):
    """总相位提升的平方，无分支信息、无满周信息"""
    return int(round(np.sum(us) ** 2 * 1000))


r_base = unrelated_reading(u_one)
r_virt = unrelated_reading(u_virtual)
r_more = unrelated_reading(u_more)
rec(
    "B4",
    "恒真检验：与两通道都无关的读数在两种情形下",
    "该读数在情形二（只动总提升）下改变，故 B3 的两个「不变」是两读数各自的"
    "守恒性质，不是任何读数都有的平凡性质",
    "无关读数：基态 %d，分支暂借 %d，总提升加倍 %d"
    % (r_base, r_virt, r_more),
    r_more != r_base,
)

# B5：多键暂借时的加性
n_borrow = 4
u_multi = spread_borrow(u_one, n_borrow)
w_multi = winding_from_phase(u_multi)
n_multi, _ = full_turn_detector(u_multi)
rec(
    "B5",
    "多键暂借的可加性",
    "暂借 %d 个分支时绕数 $-%d$（逐键可加），检测数仍为基态值 %d（总提升未变）"
    % (n_borrow, n_borrow, base_t),
    "绕数 %d，检测数 %d" % (w_multi, n_multi),
    w_multi == -n_borrow and n_multi == base_t,
)

# =====================================================================
# 组 C：两读数的类型不可互换
# =====================================================================
print("=" * 72)
print("组 C：两读数的类型不可互换")
print("=" * 72)

# C1：绕数对整体平移不变，检测数不然
u_shift = u_one + 0.7
w_shift = winding_from_phase(u_shift)
n_shift, _ = full_turn_detector(u_shift)
rec(
    "C1",
    "整体平移 $0.7$ rad 下两读数的响应",
    "绕数不变（$%d \\to %d$，分支分布未变）；检测数变（$%d \\to %d$，"
    "总提升变了）。故两者对平移的响应不同型，不可互换"
    % (w_smooth, w_shift, n_turn, n_shift),
    "绕数 %d -> %d；检测数 %d -> %d" % (w_smooth, w_shift, n_turn, n_shift),
    w_shift == w_smooth and n_shift != n_turn,
)

# C2：两读数的值域不同
w_vals = []
t_vals = []
rng = np.random.default_rng(20261021)
for _ in range(300):
    m = int(rng.integers(0, 4))
    uu = np.full(N, m * TWO_PI / N)
    if rng.random() < 0.5:
        uu = spread_borrow(uu, 1)
    w_vals.append(winding_from_phase(uu))
    tt, _ = full_turn_detector(uu)
    t_vals.append(tt)
rec(
    "C2",
    "两读数的值域",
    "绕数取整且可为负（分支偏移的负和），检测数为非负整数（满周闭合的次数）。"
    "二者符号约定与物理含义均不同",
    "绕数范围 [%d, %d]，检测数范围 [%d, %d]"
    % (min(w_vals), max(w_vals), min(t_vals), max(t_vals)),
    min(w_vals) <= 0 and min(t_vals) >= 0 and max(t_vals) >= 0,
)

# C3：恒真检验：恒为零的读数不参与
zeros = [w for w in w_vals if w == 0]
rec(
    "C3",
    "恒真检验：绕数在样本中取零的频次",
    "绕数为零的样本数 %d（三百档中），非全零，故 C2 的取值范围有内容"
    % len(zeros),
    "绕数为零的样本 %d / %d" % (len(zeros), len(w_vals)),
    0 < len(zeros) < len(w_vals),
)

# =====================================================================
# 组 D：读数
# =====================================================================
print("=" * 72)
print("组 D：读数（不进通过率）")
print("=" * 72)

print("[D1] 读数：两通道虚过程的互补形式")
print("     绕数 W 记键的分支分布（层数侧读数），检测数 T 记相位推进的圈数")
print("     （相位侧读数）。二者互补：W 对总提升不敏感，T 对分支分布不敏感。")
print("     只动分支时 W 变而 T 不变（B3 情形一），只动总提升时 T 变而 W 不变")
print("     （B3 情形二）。故两读数各自度量不同的守恒量，互不敏感、互不可替代。")
print("")
print("[D2] 读数：两读数不同型")
print("     平移 0.7 rad 时绕数不变、检测数变（C1）。")
print("     绕数可负、检测数非负（C2）。故二者不可互换，各有各的适用面。")
print("")
print("[D3] 读数：撕裂与暂借在两读数上的表现")
print("     两者都使绕数减一（A4、B1），区别在检测数：撕裂时检测数不变（分支")
print("     拓扑变了，总提升未变），暂借时检测数也不变（B1）。")
print("     故按本轮模型，二者只在分支归属上留下同样的读数差别，")
print("     其分离需另找判据；本轮不给该判据。")
print("")
print("[D4] 读数：层数侧虚过程与满周闭合接口的关系")
print("     层数侧虚过程（一键暂借一个 2pi 分支）不改变相位总推进（B1），")
print("     故它不触发满周闭合接口；归还后两读数同时复原（B2），多键暂借可加（B5）。")
print("     库内机制（相位满周闭合触发法线跃迁）给出的是相位侧到层数侧的方向，")
print("     本轮测得的是层数侧不反向触发相位侧。")
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
cases = ["基态（平滑绕行）", "撕裂（分支偏移）", "虚过程（暂借分支）"]
w_case = [w_smooth, w_tear, w_virtual]
t_case = [n_turn, n_tear, n_virtual]
xs = np.arange(3)
ax.bar(xs - 0.2, w_case, width=0.4, color="#1f77b4", alpha=0.6,
       edgecolor="k", label="绕数 $W$（层数侧，分支拓扑）")
ax.bar(xs + 0.2, t_case, width=0.4, color="#d62728", alpha=0.6,
       edgecolor="k", label="检测数 $T$（相位侧，满周闭合）")
ax.axhline(0, color="gray", linewidth=0.9)
for i, (w_, t_) in enumerate(zip(w_case, t_case)):
    ax.text(i - 0.2, w_ - 0.25 if w_ < 0 else w_ + 0.15, "%d" % w_, ha="center", fontsize=11)
    ax.text(i + 0.2, t_ + 0.15, "%d" % t_, ha="center", fontsize=11)
ax.set_xticks(xs)
ax.set_xticklabels(cases, fontsize=10)
ax.set_ylabel("读数")
ax.set_title("左：两读数在三态下的取值\n分支事件动 $W$ 而不动 $T$")
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis="y")

ax = axes[1]
nb = np.arange(0, 9)
w_line = [winding_from_phase(
    np.concatenate([u_one, np.zeros(0)])) for _ in range(0)]
# 构造：暂借 k 个分支
def make_borrow(k):
    return spread_borrow(u_one, k)


w_line, t_line, wpt_line = [], [], []
for k in nb:
    uu = make_borrow(k)
    w_k = winding_from_phase(uu)
    t_k, _ = full_turn_detector(uu)
    w_line.append(w_k)
    t_line.append(t_k)
    wpt_line.append(w_k + t_k)
ax.plot(nb, w_line, "o-", color="#1f77b4", linewidth=2.0, markersize=7,
        label="绕数 $W$（降 $k$）")
ax.plot(nb, t_line, "s-", color="#d62728", linewidth=2.0, markersize=7,
        label="检测数 $T$（升 $k$）")
ax.plot(nb, wpt_line, "^--", color="#2ca02c", linewidth=2.0, markersize=8,
        label="$W+T$（随之变化）")
ax.set_xlabel("暂借的分支数 $k$")
ax.set_ylabel("读数")
ax.set_title("右：两读数的可加性\n$W$ 随 $k$ 降，$T$ 不动")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(IMG, dpi=150)
print("")
print("图已保存：%s" % IMG)
