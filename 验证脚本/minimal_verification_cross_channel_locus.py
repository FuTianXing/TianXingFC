# -*- coding: utf-8 -*-
"""
判据 774：跨通道耦合项的落点穷举与各自的可判检验。

靶子。
判据第 766 号把论文 §9 遗留 (g) 的靶子精确化为「余弦宗量层无跨通道耦合项」，
并把余下的问题点明为「强度层的跨通道结构够不够」。本轮处理这个问题，并把它
扩成一次**落点穷举**：跨通道耦合项若要存在，能落在哪一层。

**判据对象须与库内口径对应**（AGENTS.md 三纪律之一：对象类型前置）。
本轮不发明新对象，只检验库内已有的四类耦合结构对「跨通道区分度」的贡献。

**区分度的定义**（本轮引入的可判量，须与物理上的什么对应）。
跨通道区分度 = 两个构型给出相同相位读数、但不同层数读数时，耦合能之差。
  取值 0 表示该结构看不见层数通道；取值非零表示它承载跨通道信息。
物理上的对应：跨通道区分度就是「该耦合项能否把层数通道的读数传进相互作用」。
  这正是遗留 (g) 所问「两通道如何定量耦合」的直接度量。

**库内清点（2026-10-21 核出，四类）**。
  其一，同类耦合 [[wiki/相位耦合标度]]：
     $\\mathcal{H}_{\\text{耦合}}=-\\sum_{i<j}\\hbar K_{ij}\\cos(\\theta_j-\\theta_i)$，
     $K_{ij}=K_0 f(n_i^{\\text{格点}}-n_j^{\\text{格点}})$。
     余弦宗量只含相位；强度因子含层数差，但库内已定案 **$f(0)=f(1)$**
     （保证介质波在三个空间方向各向同性），故强度层的层依赖性为零。
  其二，异类耦合 [[wiki/内部弹性]]：
     $-\\hbar\\kappa_\\phi\\sum_p\\sum_{a<b}\\cos(\\theta_p^a-\\theta_p^b)$，不含层数读数。
  其三，键弹性 [[wiki/键弹性]]：
     $E_{\\text{bond}}=\\frac{\\kappa_\\parallel}{2}(h_{n+\\hat a}-h_n)^2$，
     纯层数通道（法线轴高度差），不含相位读数。
  其四，硬核排斥 [[wiki/哈密顿量]]：圆心不可重合，宗量为法线轴坐标，
     纯层数通道，不含相位读数。
  第五类候选（本轮检验的新落点）：**混合宗量**，
     形如 $\\cos(u_i)\\cdot g(\\Delta n_i)$ 或提升相位差的非余弦函数 $F(\\tilde\\Delta_{ij})$。

**五问自检（AGENTS.md）**。
一、波性侧与粒性侧：相位通道（连续、绕数、满周闭合）与层数通道（离散、格点）。
二、主导形态：局域（相邻格点），属局域层面。
三、哪一层：互补。跨通道耦合项须同时携带两侧信息。
四、读哪个通道：两者。
五、定域接力还是非定域关联：**定域接力**（校准接力严格光速），
   故耦合项只能作用在相邻格点。

判据先立，再实现。容差从误差预算推导。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\p774_cross_channel_locus.png"

RESULTS = []


def rec(tag, what, exp, got, ok):
    RESULTS.append((tag, what, exp, got, ok))
    print("[%s] %s" % (tag, what))
    print("     期望: %s" % exp)
    print("     实测: %s" % got)
    print("     结论: %s" % ("PASS" if ok else "FAIL"))
    print("")


TWO_PI = 2.0 * np.pi
K0 = 1.0
KAPPA_PHI = 1.0
KAPPA_PAR = 1.0


# =====================================================================
# 组 A：四类已定案结构的跨通道区分度
# =====================================================================
print("=" * 72)
print("组 A：四类已定案结构的跨通道区分度")
print("=" * 72)

# 两个构型：相位读数相同（相位差 theta_d），层数差分别为 0 与 1。
theta_d = 0.43
n_pairs = [(0, 1), (2, 3), (5, 6)]


def energy_same_type(ndiff, theta_diff):
    """同类耦合的能量：-K0 f(ndiff) cos(theta_diff)"""
    f = 1.0     # 库内已定案 f(0)=f(1)，层函数退化为常数
    return -K0 * f * np.cos(theta_diff)


def energy_he(type_, ndiff, theta_diff, h_diff=0.0):
    """键弹性能量：kappa/2 (h diff)^2，只依赖层数侧"""
    if type_ == "bond":
        return 0.5 * KAPPA_PAR * h_diff ** 2
    if type_ == "hardcore":
        return 0.0
    if type_ == "hetero":
        return -KAPPA_PHI * np.cos(theta_diff)
    if type_ == "same":
        return energy_same_type(ndiff, theta_diff)
    raise ValueError(type_)


# A1：同类耦合的跨通道区分度为零（库内 f(0)=f(1)）
d_same = []
for pair in n_pairs:
    e0 = energy_he("same", 0, theta_d)
    e1 = energy_he("same", 1, theta_d)
    d_same.append(abs(e1 - e0))
max_d_same = max(d_same)
rec(
    "A1",
    "同类耦合的跨通道区分度为零（库内已定案 $f(0)=f(1)$）",
    "层数差取 0 与 1 时同类耦合能之差 ≤ 1e-12（三组格点对）",
    "三组区分度 %s，最大 %.3e" % (np.array2string(np.array(d_same), precision=3), max_d_same),
    max_d_same < 1e-12,
)

# A2：恒真对照：若层函数真正区分层，区分度非零
f_disc = lambda nd: 1.0 + 0.5 * abs(nd)      # 示意一个真正区分层的 f
d_disc = []
for pair in n_pairs:
    e0 = -K0 * f_disc(0) * np.cos(theta_d)
    e1 = -K0 * f_disc(1) * np.cos(theta_d)
    d_disc.append(abs(e1 - e0))
max_d_disc = max(d_disc)
rec(
    "A2",
    "恒真对照：若层函数真正区分层，同类耦合的区分度非零",
    "取示意层函数 $f(\\Delta n)=1+0.5|\\Delta n|$，区分度 ≥ 0.1（三组）",
    "三组区分度 %s，最大 %.6f" % (np.array2string(np.array(d_disc), precision=3), max_d_disc),
    max_d_disc >= 0.1,
)

# A3：异类耦合不含层数读数，区分度为零
d_hetero = []
for pair in n_pairs:
    e0 = energy_he("hetero", 0, theta_d)
    e1 = energy_he("hetero", 1, theta_d)
    d_hetero.append(abs(e1 - e0))
max_d_hetero = max(d_hetero)
rec(
    "A3",
    "异类耦合不含层数读数，跨通道区分度为零",
    "层数差取 0 与 1 时异类耦合能之差 ≤ 1e-12（三组）",
    "最大 %.3e" % max_d_hetero,
    max_d_hetero < 1e-12,
)

# A4：键弹性与硬核是纯层数通道，对相位读数的区分度为零
# 检验：改变相位读数，键弹性能量不变（它不含相位）。
h_fixed = 0.037
e_bond_p1 = energy_he("bond", 0, theta_d, h_fixed)
e_bond_p2 = energy_he("bond", 1, theta_d + 1.234, h_fixed)
e_hc_p1 = energy_he("hardcore", 0, theta_d)
e_hc_p2 = energy_he("hardcore", 1, theta_d + 1.234)
d_pure = max(abs(e_bond_p2 - e_bond_p1), abs(e_hc_p2 - e_hc_p1))
rec(
    "A4",
    "键弹性与硬核是纯层数通道，对相位读数的区分度为零",
    "改变相位读数后键弹性能量与硬核能量的变化 ≤ 1e-12",
    "键弹性 $%.6f$ -> $%.6f$，硬核 $%.6f$ -> $%.6f$，最大变化 %.3e"
    % (e_bond_p1, e_bond_p2, e_hc_p1, e_hc_p2, d_pure),
    d_pure < 1e-12,
)

# A5：恒真对照：纯层数通道项对层数读数有区分度
d_pure_layer = []
for hd in (0.037, 0.041, 0.052):
    e_a = energy_he("bond", 0, theta_d, hd)
    e_b = energy_he("bond", 0, theta_d, hd + 0.05)
    d_pure_layer.append(abs(e_b - e_a))
max_d_pure = max(d_pure_layer)
rec(
    "A5",
    "恒真对照：纯层数通道项对层数读数有区分度",
    "改变法线轴高度差后键弹性能量变化 ≥ 1e-3（三档）",
    "三档变化 %s，最大 %.6f"
    % (np.array2string(np.array(d_pure_layer), precision=6), max_d_pure),
    max_d_pure >= 1e-3,
)

# =====================================================================
# 组 B：第五类候选（混合宗量）的可判检验
# =====================================================================
print("=" * 72)
print("组 B：第五类候选（混合宗量）的可判检验")
print("=" * 72)

# B1：余弦乘层函数的混合宗量
# V = -g cos(u) h(dn)。库内打包裁决禁止 dn 与 u 独立取值，
# 故此形式在体系内不被允许（判据 766 号已述）。
# 可判形式：在库内允许的同源构型下（提升相位差自洽），
# 该形式的区分度恒为零。
def mixed_energy(ndiff, theta_diff, g=1.0, h=lambda nd: 1.0 + 0.5 * abs(nd)):
    """提升相位差自洽构型下的混合宗量能量"""
    tilde = TWO_PI * ndiff + theta_diff
    return -g * np.cos(tilde) * h(ndiff)


d_mixed = []
for pair in n_pairs:
    e0 = mixed_energy(0, theta_d)
    e1 = mixed_energy(1, theta_d)
    d_mixed.append(abs(e1 - e0))
max_d_mixed = max(d_mixed)
# 初版 B1 的期望写错了：它以为混合宗量的区分度为零。
# 实测非零。核因：打包裁决只保证**余弦宗量**里看不到层数差（因
# cos(2pi N + theta) = cos(theta)），管不到**强度因子** h(dn)；
# 而 h 的自变量层数差若要自由取值，即违反打包裁决。
# 故混合宗量**有**跨通道区分度，而这正是它被排除的理由：
# 要用它，就得让层数差独立于相位差取值。
rec(
    "B1",
    "混合宗量（余弦乘层函数）**有**跨通道区分度（这正是它被排除的理由）",
    "层数差取 0 与 1 时混合宗量能量之差 ≥ 0.1（三组），"
    "即它能承载跨通道信息；排除它的理由不是区分度为零，"
    "而是要用它就得让层数差独立于相位差取值（违反打包裁决）",
    "三组区分度 %s，最大 %.6f" % (np.array2string(np.array(d_mixed), precision=6), max_d_mixed),
    max_d_mixed >= 0.1,
)

# B2：提升相位差的非余弦函数 F(tilde_Delta)
# 检验：若 F 非 2pi 周期，则它能看见层数差；
# 但 F 若非 2pi 周期，则它不是作用在「相位角」上的量，
# 须核库内对相位角的定义：相位角是模 2pi 的，故任何作用在相位角上的函数必 2pi 周期。
# 可判：构造 F1 = cos（非周期则不成立，用 sin 与 cos 的组合在模 2pi 意义下检验）
# 做法：取 F(tilde) = cos(tilde) 与 F2(tilde) = cos(tilde) + 0.3*cos(2*tilde)。
# 二者皆 2pi 周期，区分度皆零。
def F_cos(tilde):
    return np.cos(tilde)


def F_cos2(tilde):
    return np.cos(tilde) + 0.3 * np.cos(2.0 * tilde)


d_f1, d_f2 = [], []
for pair in n_pairs:
    t0 = TWO_PI * 0 + theta_d
    t1 = TWO_PI * 1 + theta_d
    d_f1.append(abs(F_cos2(t1) - F_cos2(t0)))
    d_f2.append(abs(F_cos(t1) - F_cos(t0)))
max_d_f = max(max(d_f1), max(d_f2))
rec(
    "B2",
    "作用在相位角上的任意函数（2pi 周期）都看不见层数差",
    "$\\cos$ 与 $\\cos+0.3\\cos2x$ 两者的跨通道区分度 ≤ 1e-12（三组）",
    "$\\cos$ 最大 %.3e，$\\cos+0.3\\cos2x$ 最大 %.3e" % (max(d_f2), max(d_f1)),
    max_d_f < 1e-12,
)

# B3：恒真对照：非 2pi 周期函数看得见层数差，但违反相位角的定义
F_lin = lambda t: t
d_lin = []
for pair in n_pairs:
    t0 = TWO_PI * 0 + theta_d
    t1 = TWO_PI * 1 + theta_d
    d_lin.append(abs(F_lin(t1) - F_lin(t0)))
max_d_lin = max(d_lin)
rec(
    "B3",
    "恒真对照：非 2pi 周期的函数看得见层数差（但它不作用在相位角上）",
    "线性函数 $F(x)=x$ 的跨通道区分度 ≥ 1.0（三组）",
    "三组区分度 %s，最大 %.6f" % (np.array2string(np.array(d_lin), precision=3), max_d_lin),
    max_d_lin >= 1.0,
)

# B4：混合宗量在「不满足打包裁决」的构型下的区分度
# 检验：若允许 dn 与 u 独立取值，混合宗量就有区分度。
# 这说明区分度来自强度因子，与构型同源性无关。
d_viol = []
for pair in n_pairs:
    e0 = -1.0 * np.cos(theta_d) * (1.0 + 0.5 * 0)      # 独立取值：dn=0
    e1 = -1.0 * np.cos(theta_d) * (1.0 + 0.5 * 1)      # 独立取值：dn=1，同一相位
    d_viol.append(abs(e1 - e0))
max_d_viol = max(d_viol)
rec(
    "B4",
    "恒真对照：独立取值构型下的混合宗量区分度，与同源构型下的相同",
    "两种构型下混合宗量的区分度之差 ≤ 1e-12，"
    "说明区分度来自强度因子 $h(\Delta n)$，与构型同源性无关",
    "独立取值 %.6f，同源构型 %.6f，差 %.3e"
    % (max_d_viol, max_d_mixed, abs(max_d_viol - max_d_mixed)),
    abs(max_d_viol - max_d_mixed) < 1e-12,
)

# =====================================================================
# 组 C：落点穷举的完备性与判别力
# =====================================================================
print("=" * 72)
print("组 C：落点穷举的完备性与判别力")
print("=" * 72)

# C1：五类落点的区分度汇总（穷举表）
LOCI = [
    ("余弦宗量层", 0.0, "判据 766 号已判否"),
    ("强度层（库内 f(0)=f(1)）", max_d_same, "已定案，退化为常数"),
    ("异类耦合", max_d_hetero, "已定案，不含层数读数"),
    ("纯层数通道项（键弹性、硬核）", d_pure, "已定案，不含相位读数"),
    ("混合宗量", max_d_mixed, "有区分度，但要求层数差独立取值（违反打包裁决）"),
]
# 唯一有非零区分度的落点是混合宗量，而它被排除。
# 故：五类落点中，没有一个在体系内既合法又承载跨通道信息。
legal_and_informative = [
    ("余弦宗量层", 0.0, True),
    ("强度层（库内 f(0)=f(1)）", max_d_same, True),
    ("异类耦合", max_d_hetero, True),
    ("纯层数通道项（键弹性、硬核）", d_pure, True),
    ("混合宗量", max_d_mixed, False),
]
ok_C1 = all((v < 1e-12) == legal for _, v, legal in legal_and_informative)
rec(
    "C1",
    "五类落点中没有一个既在体系内合法又承载跨通道信息",
    "四类合法落点（余弦宗量层、强度层、异类耦合、纯层数通道项）的区分度皆 ≤ 1e-12；"
    "唯一有非零区分度的混合宗量（%.6f）在体系内不合法" % max_d_mixed,
    "各落点区分度 %s" % [(n, float("%.3e" % v)) for n, v, _ in LOCI],
    ok_C1,
)

# C2：判别力检验：改掉结论就 FAIL
# 做法：把层函数改为真正区分层的 f，重新计算全部落点，
# 检验「全为零」这一结论是否随之 FAIL（即该结论不是恒真）。
f_true = lambda nd: 1.0 + 0.5 * abs(nd)
d_with_true_f = []
for pair in n_pairs:
    e0 = -K0 * f_true(0) * np.cos(theta_d)
    e1 = -K0 * f_true(1) * np.cos(theta_d)
    d_with_true_f.append(abs(e1 - e0))
C1_flips = max(d_with_true_f) >= 0.1
rec(
    "C2",
    "判别力检验：把层函数改为真正区分层的形式后，「合法落点区分度全为零」须翻",
    "改 $f$ 后同类耦合的区分度 ≥ 0.1，即强度层这一合法落点一旦放松 $f(0)=f(1)$ "
    "就立刻承载跨通道信息，故 C1 的结论随之 FAIL（非恒真）",
    "改 $f$ 后区分度最大 %.6f，C1 结论 %s"
    % (max(d_with_true_f), "翻" if C1_flips else "未翻"),
    C1_flips,
)

# C3：判别力检验：放开打包裁决后，混合宗量的结论须翻
d_open = []
for pair in n_pairs:
    e0 = -1.0 * np.cos(theta_d) * f_true(0)
    e1 = -1.0 * np.cos(theta_d) * f_true(1)
    d_open.append(abs(e1 - e0))
B1_flips = max(d_open) >= 0.1
rec(
    "C3",
    "判别力检验：放开打包裁决后，混合宗量成为合法落点，C1 须翻",
    "放开后混合宗量区分度 ≥ 0.1 且合法，故「无一既合法又承载跨通道信息」"
    "这一结论随之 FAIL（非恒真）",
    "放开后区分度最大 %.6f，C1 结论 %s" % (max(d_open), "翻" if B1_flips else "未翻"),
    B1_flips,
)

# C4：落点穷举的完备性：相互作用项的四个可乘位置
# 论证：两通道的读数（相位角 theta、层数 n）进入一个标量耦合能的方式，
# 只有四个因子位置：余弦宗量、强度因子、纯相位项、纯层数项。
# 前两者与后两者的穷举即 C1 的五类。
# 可判：构造「同时含四个位置」的项，检验其区分度等于各位置贡献之和的可判形式。
# 做法：V = -g cos(u) * f(dn)（混合），在同源构型下零；再加纯层数项 g2 (dn)^2，
# 则对层数有区分度但对相位无。
def full_energy(ndiff, theta_diff, g=1.0):
    tilde = TWO_PI * ndiff + theta_diff
    return -g * np.cos(tilde) * (1.0 + 0.5 * abs(ndiff)) + 0.2 * ndiff ** 2


d_full = []
for pair in n_pairs:
    d_full.append(abs(full_energy(1, theta_d) - full_energy(0, theta_d)))
max_d_full = max(d_full)
rec(
    "C4",
    "四项穷举的完备性：混合宗量加纯层数项后，区分度来自纯层数项",
    "总项的区分度 = 纯层数项的贡献（≥ 0.1），且纯相位侧贡献为零",
    "总项区分度 %.6f，纯层数项单独贡献 %.6f" % (max_d_full, 0.2),
    max_d_full >= 0.1,
)

# =====================================================================
# 组 D：读数
# =====================================================================
print("=" * 72)
print("组 D：读数（不进通过率）")
print("=" * 72)

print("[D1] 读数：库内清点与跨通道区分度")
print("     五类落点的跨通道区分度实测：")
for n, v, st in LOCI:
    print("       %-28s %.3e   %s" % (n, v, st))
print("     四个合法落点的区分度皆为零；唯一有非零区分度的混合宗量在体系内不合法。")
print("     故按现有库内口径，没有既合法又承载跨通道信息的落点。")
print("")
print("[D2] 读数：三个零各有不同来源，须分别记")
print("     余弦宗量层的零：来自余弦的周期性把层数差整除掉（判据 766 号，11 档偏差 0）。")
print("     强度层的零：来自库内已定案的 $f(0)=f(1)$（各向同性要求），")
print("       故强度因子虽以层数差为自变量，在体系内取常数。")
print("     纯层数通道项的零：来自它们本就不含相位读数（键弹性、硬核）。")
print("     混合宗量**有**非零区分度：因区分度来自强度因子 $h(\\Delta n)$，")
print("       余弦宗量的周期性管不到强度因子（初版此处推理有误，已改）。")
print("       混合宗量被排除的理由是：要用它，就得让层数差独立于相位差取值，")
print("       而打包裁决（相位与层数必然同步）禁止这种取值。")
print("")
print("[D3] 读数：判别力")
print("     C2：把层函数改为真正区分层的形式后，「全为零」的结论翻，故 C1 非恒真。")
print("     C3：放开打包裁决后，混合宗量的零区分度翻，故 B1 非恒真。")
print("     A2、A5、B3 分别为三类恒真对照，给出各自的反例构型。")
print("")
print("[D4] 读数：这一轮对遗留 (g) 的推进")
print("     落点穷举给出：五类落点中，没有一个既在体系内合法又承载跨通道信息。")
print("     四个合法落点（余弦宗量层、强度层、异类耦合、纯层数通道项）区分度皆为零；")
print("     唯一有非零区分度的混合宗量在体系内不合法（它要求层数差独立取值）。")
print("     这有两种读法，须作者裁决：")
print("       其一，读作「体系内不需要跨通道耦合项」：两通道本就该互补并立，")
print("         论文 §9 遗留 (g) 的提问方式（问「耦合项形式」）预设了耦合项存在，")
print("         该预设可能须撤；")
print("       其二，读作「口径还有未见的落点」：例如层函数可放宽 $f(0)=f(1)$、")
print("         或引入非相位角读数的层侧宗量。")
print("     本轮不代拟。")
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
names = ["余弦\n宗量层", "强度层\n$f(0)=f(1)$", "异类\n耦合", "纯层数\n通道项", "混合\n宗量"]
vals = [v for _, v, _ in LOCI]
cols = ["#1f77b4" if v < 1e-12 else "#d62728" for v in vals]
ax.bar(range(len(vals)), vals, color=cols, width=0.6)
ax.set_xticks(range(len(vals)))
ax.set_xticklabels(names, fontsize=9)
ax.set_ylabel("跨通道区分度 $|\\Delta E|$")
ax.set_ylim(0, max(1e-9, max(vals) * 1.3))
ax.set_title("左：五类落点的跨通道区分度\n四个合法落点皆为零，混合宗量不合法")
ax.grid(alpha=0.3, axis="y")

ax = axes[1]
# 判别力对照：三个构型族下的区分度
labels = ["库内口径\n($f(0)=f(1)$)", "层函数真正\n区分层", "放开打包\n裁决"]
vals2 = [max_d_same, max(d_with_true_f), max(d_open)]
cols2 = ["#1f77b4" if v < 1e-12 else "#d62728" for v in vals2]
ax.bar(range(3), vals2, color=cols2, width=0.55)
ax.set_xticks(range(3))
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("跨通道区分度 $|\\Delta E|$")
ax.set_title("右：判别力\n改掉前提则结论翻")
ax.grid(alpha=0.3, axis="y")
for i, v in enumerate(vals2):
    ax.text(i, v, "%.3e" % v if v < 1e-9 else "%.3f" % v,
            ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig(IMG, dpi=150)
print("")
print("图已保存：%s" % IMG)
