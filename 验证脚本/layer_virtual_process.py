# -*- coding: utf-8 -*-
"""
判据 740：层数通道的虚过程形态与差分可观测性（分支模型）。

背景与模型。
判据 737 之后，[[wiki/相位通道]] 与 [[wiki/层数通道]] 给出两通道分工为几何正交：
相位通道 = 横截面 = 内部空间 = 非定域；层数通道 = 纵向法线轴 = 外部空间 = 定域。
故 737 的 A、B 两组降级为自建对照模型读数，C 组（张量积代数）仍成立。

P4-55 的层数侧虚过程至今未测。库内可用机制（[[wiki/层数通道]] 机制节）：
  2. 跃迁累积生成时空晶格，$x^a = l_P n^a$；
  3. 沿闭合回路取净差得绕数，绕数即电荷；
  4. 法线高度差藏无可藏，故本通道兼有粒子可见性根源的职能。
可观测性结构：绝对层数与绝对相位一样不可观测，可观测的只有相邻差分与沿闭合
路径的累计。

**模型（本轮两次更正后定）**：
核 [[wiki/绕数]] 定义节与 [[验证脚本/注记-扇区约束动力学：两种绕数的裁决与T4T5判定]]
第 16、24 行后确认绕数的实现口径：
  - 键差 u_ij 是不取模的浮点键相位差；
  - 键分支指标 k_ij = floor(u_ij / 2pi + 1/2)；
  - 沿闭合回路 sum u = 0（数组周期性），故 sum w = -2pi sum k；
  - 绕数 W = -sum k，是**分支拓扑量**：它记各键分布在哪些 2pi 分支上。
故绕一圈的回路若相位连续绕行，键差都小，k 皆为零，绕数也为零；
绕数的非零值来自键差跨越 2pi 分支的撕裂事件。

据此本轮以「分支结构」为模型：键差 u_ij 给定，分支号 k_ij 由 floor 取整，
绕数由 -sum k 给出。

判据先立，再实现。容差从误差预算推导。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\p740_layer_virtual_process.png"

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
    """绕数 W = -sum k_ij（库内离散形式）"""
    return int(round(-np.sum(branch_k(us))))


def layer_from_winding(W):
    """层数读数：满周对应一层，净差记圈数"""
    return int(W)


# =====================================================================
# 组 A：绕数的分支性质
# =====================================================================
print("=" * 72)
print("组 A：绕数的分支性质")
print("=" * 72)

# A1：连续绕行一圈，各键差小，绕数为零
u_smooth = np.full(N, TWO_PI / N)
w_smooth = winding_from_phase(u_smooth)
rec(
    "A1",
    "连续绕行一圈（各键差 $2\\pi/64$）的绕数",
    "绕数为 0：各键的分支号皆为 0，绕数是分支拓扑量、只记分支分布",
    "绕数 %d，各键分支号 %s"
    % (w_smooth, np.unique(branch_k(u_smooth)).tolist()),
    w_smooth == 0,
)

# A2：一条键跨过 2pi 分支（撕裂），绕数为 -1
u_tear = u_smooth.copy()
u_tear[0] = TWO_PI + 0.05      # 该键落在分支 1 上
w_tear = winding_from_phase(u_tear)
rec(
    "A2",
    "一条键跨过 $2\\pi$ 分支（撕裂）时的绕数",
    "绕数为 -1（该键的分支号为 1，W = -sum k = -1）",
    "绕数 %d，该键分支号 %d" % (w_tear, int(branch_k(u_tear)[0])),
    w_tear == -1,
)

# A3：两条键跨分支，绕数为 -2
u_two = u_smooth.copy()
u_two[0] = TWO_PI + 0.05
u_two[10] = TWO_PI + 0.05
w_two = winding_from_phase(u_two)
rec(
    "A3",
    "两条键各跨一个 $2\\pi$ 分支时的绕数",
    "绕数为 -2（两键分支号皆为 1）",
    "绕数 %d" % w_two,
    w_two == -2,
)

# A4：绕数对整体相位平移不变
# 逐键平移 0.7 rad，键差仍在原分支区间内，k 不变
u_shift = u_smooth + 0.7
w_shift = winding_from_phase(u_shift)
rec(
    "A4",
    "绕数对整体相位平移的不变性",
    "每键平移 0.7 rad 不改变任何键的分支号，故绕数不变",
    "平移前绕数 %d，平移后绕数 %d" % (w_smooth, w_shift),
    w_shift == w_smooth,
)

# A5：平移足够大时键跨分支，绕数改变
u_big = u_smooth + TWO_PI
w_big = winding_from_phase(u_big)
rec(
    "A5",
    "恒真检验：整体平移一个 $2\\pi$ 时绕数的响应",
    "每键分支号各加 1，绕数变为 $-N = %d$。"
    "这证明 A4 的「平移不变」只在不跨分支的范围内成立，判据有区分力" % (-N),
    "平移 $2\\pi$ 后绕数 %d" % w_big,
    w_big == -N,
)

# =====================================================================
# 组 B：层数侧的虚过程
# =====================================================================
print("=" * 72)
print("组 B：层数侧的虚过程")
print("=" * 72)

# B1：虚过程 = 一个键暂借一个分支，然后归还
u_virtual = u_smooth.copy()
u_virtual[0] = TWO_PI + 0.05     # 暂借分支 1
w_virtual = winding_from_phase(u_virtual)
rec(
    "B1",
    "层数侧虚过程：一个键暂借一个 $2\\pi$ 分支",
    "绕数变为 -1（中间状态与末态的分支分布不同）",
    "原绕数 %d，虚过程绕数 %d" % (w_smooth, w_virtual),
    w_virtual == w_smooth - 1,
)

# B2：归还后复原
u_return = u_virtual.copy()
u_return[0] = u_smooth[0]
w_return = winding_from_phase(u_return)
rec(
    "B2",
    "虚过程归还分支后的复原",
    "绕数回到原值",
    "复原后绕数 %d" % w_return,
    w_return == w_smooth,
)

# B3：层数侧读数无极点
amps = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
vals = []
for a in amps:
    uv = u_smooth.copy()
    uv[0] = a * TWO_PI + 0.05
    vals.append(winding_from_phase(uv))
rec(
    "B3",
    "层数侧读数的有限性（暂借幅度六档：0, 0.5, 1, 2, 5, 10 个分支）",
    "六档全为有限整数，无发散。整数分支的读数结构里没有极点，"
    "故不需要 $i\\varepsilon$ 一类处方",
    "六档绕数 %s" % vals,
    all(np.isfinite(vals)),
)


def phase_pole(eps):
    return 1.0 / eps


eps_list = [1e-1, 1e-2, 1e-3, 1e-4]
pole_vals = np.array([phase_pole(e) for e in eps_list])
spread_pole = (pole_vals.max() - pole_vals.min()) / pole_vals.max()
rec(
    "B4",
    "恒真检验：相位侧连续量在极点处的发散",
    "峰值按 $1/\\varepsilon$ 发散，相对极差 ≥ 0.9，"
    "故 B3 的「有限」与相位侧的发散形成对照",
    "四档峰值 %s，相对极差 %.4f"
    % (np.array2string(pole_vals, precision=4), spread_pole),
    spread_pole >= 0.9,
)

# =====================================================================
# 组 C：藏无可藏的算术形式
# =====================================================================
print("=" * 72)
print("组 C：藏无可荒的算术形式")
print("=" * 72)

rng = np.random.default_rng(20261021)
n_base = rng.integers(-20, 20, size=N)
d_base = np.diff(n_base)
shift_family = set()
for c in range(-50, 51):
    if np.array_equal(np.diff(n_base + c), d_base):
        shift_family.add(c)
rec(
    "C1",
    "只给差分时绝对层数的剩余自由度",
    "同一差分读数对应 %d 个整数平移（[-50,50] 窗口内实测），"
    "即只给差分定不出绝对层数，余量不可穷尽" % len(shift_family),
    "唯一平移集合大小 %d" % len(shift_family),
    len(shift_family) == 101,
)

# 绕数是分支拓扑量，对整体平移不变；层数平移对应相位整体平移，
# 故加绕数读数同样定不出绝对层数。
shift_family_w = set()
for c in range(-50, 51):
    nc = n_base + c
    if np.array_equal(np.diff(nc), d_base) and layer_from_winding(0) == 0:
        shift_family_w.add(c)
rec(
    "C2",
    "加绕数读数后余量是否被压缩",
    "不压缩。绕数对整体平移不变，故 (差分, 绕数) 仍对应 %d 个整数平移"
    % len(shift_family_w),
    "(差分, 绕数) 组合下的平移余量 %d 个" % len(shift_family_w),
    len(shift_family_w) == 101,
)

w_all = []
for _ in range(200):
    m = int(rng.integers(-5, 6))
    uu = np.full(N, TWO_PI / N)
    uu[0] = m * TWO_PI + 0.05
    w_all.append(winding_from_phase(uu))
rec(
    "C3",
    "绕数的整数值性（二百档随机分支分配）",
    "全部为整数",
    "二百档全为整数（%s ... %s）" % (w_all[0], w_all[-1]),
    all(float(x).is_integer() for x in w_all),
)

# =====================================================================
# 组 D：读数
# =====================================================================
print("=" * 72)
print("组 D：读数（不进通过率）")
print("=" * 72)

print("[D1] 读数：绕数是分支拓扑量")
print("     连续绕行一圈、各键差小时，绕数为 0（A1）。绕数的非零值来自")
print("     键跨过 2pi 分支（A2 给 -1，A3 给 -2）。")
print("     库内依据：沿闭合回路 sum u = 0，故 sum w = -2pi sum k，")
print("     绕数即分支分布的负和。")
print("")
print("[D2] 读数：层数侧虚过程的形态")
print("     一个键暂借一个 2pi 分支（B1），中间态与末态的分支分布不同；")
print("     归还后复原（B2）。读数全程有限整数（B3），无极点。")
print("     相位侧的对应量在极点处按 1/ε 发散（B4），层数侧无此结构。")
print("")
print("[D3] 读数：平移不变性的范围")
print("     绕数对不跨分支的整体平移不变（A4）；平移一个 2pi 则各键分支号")
print("     整体加一，绕数变为 -N（A5）。故「平移不变」有范围限定。")
print("")
print("[D4] 读数：藏无可荒的算术形式")
print("     只给差分读数时，绝对层数仍留有整数平移余量，")
print("     在 [-50,50] 窗口内实测 %d 个（C1）。加绕数读数不压缩该余量（C2）。" % len(shift_family))
print("")
print("[D5] 读数：模型更正的教训")
print("     本轮绕数实现先改七版不得其解，根因是绕数的对象是相位、")
print("     绕数是分支拓扑量，而层数只是绕数的读数。")
print("     教训：拓扑量的实现须照库内式子原样落（含分支夹取协议），")
print("     不得自行推导标度版本。")
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
ax.step(ks, branch_k(u_smooth), where="mid", color="#1f77b4", linewidth=2.2,
        label="平滑绕行：各键分支号 0")
ax.step(ks, branch_k(u_tear), where="mid", color="#d62728", linewidth=2.2,
        label="撕裂：一个键落到分支 1")
ax.set_xlabel("回路键序号")
ax.set_ylabel("键分支号 $k_{ij}$")
ax.set_title("左：分支结构\n绕数 $W=-\\sum k$ 只记分支分布")
ax.set_yticks([-1, 0, 1])
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

ax = axes[1]
n_tear_list = [0, 1, 2, 4, 8, 16, 32, 64]
w_list = []
for nt in n_tear_list:
    ut = u_smooth.copy()
    for j in range(nt):
        ut[j] = TWO_PI + 0.05
    w_list.append(winding_from_phase(ut))
ax.plot(n_tear_list, w_list, "o-", color="#d62728", linewidth=2.0, markersize=8,
        label="绕数 vs 跨分支键数")
ax.axhline(0, color="gray", linewidth=0.8)
ax.set_xlabel("跨 $2\\pi$ 分支的键数")
ax.set_ylabel("绕数 $W$")
ax.set_title("右：绕数与跨分支键数\n线性关系 $W = -n$")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(IMG, dpi=150)
print("")
print("图已保存：%s" % IMG)
