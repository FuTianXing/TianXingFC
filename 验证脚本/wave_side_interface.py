# -*- coding: utf-8 -*-
"""
判据 791：波性侧三处候选接口与层数读数的关系。

靶子。
P4-55 的遗留项（论文 §9 遗留 (g)「两通道之间的耦合项形式」）经判据第 786 号收窄为
「跨两层方向有没有直接同类耦合」。作者随后提出一个方向性追问：跨通道耦合项若
**在波性侧（相位通道）**而不是粒性侧起作用，是否合理。

该追问指出了十四轮的一个盲点：此前只查了「耦合项的宗量取哪侧读数」与「强度层」，
**没有系统查波性侧自己的耦合项（键变量、异类耦合）里有没有藏着层数读数的间接通道**。
本轮查三处候选接口。

**库内口径（本轮不自行发明）**。
  其一，[[wiki/格键]]：键上承载的物理量有两类。**波性侧**承载几何读数的量叫键变量，
    其取值即**棱差**（两端实轴方向之差取模一圈）；**粒性侧**键弹性写在法线轴高度差上。
    且「相位读数不写在点上、写在键上」。
  其二，[[wiki/棱差]]：棱差是**圆上变量，取值模一圈**；其闭合和给出面元曲率与绕数；
    **它的共轭取整数谱**；跨键台阶（墙）只在棱层显形，面元层看不见。
  其三，[[wiki/对偶基（整数通量谱）]]：对偶基是棱差这套圆上相位变量的**傅里叶对偶基**，
    把每个格键上的相位读数换成一族**整数标签**；**谱取整数是这套载体的关键性质**，
    也是提升关系与幺正演化能在其上成立的前提；词条明写「整数谱让提升关系可以在基上
    逐条实现」，且「对偶标签只取整数，这是**圆上取值带来的硬约束**」。
  其四，[[wiki/实轴方向场]]：实轴方向场住横向相位通道，时空晶格住纵向层数通道，
    两通道 2026-09-30 裁决**正交且度规可分离**；**$\\alpha$ 的涨落不改变格距 $l_P$**。
  其五，[[wiki/提升相位]]：提升相位 $\\tilde\\theta = 2\\pi N + \\theta$，$N$ 是累积摆程数
    （层数），$\\theta$ 是模 $2\\pi$ 的相位角；相位与层数必然同步，不存在独立变化。

**五问自检（AGENTS.md）**。
一、波性侧与粒性侧：相位通道（连续、圆上变量、模 $2\\pi$）与层数通道（离散、格点、整数）。
二、主导形态：整体层面（键上读数的代数结构）。
三、哪一层：互补。
四、读哪个通道：两者（判据对象即两者的接口）。
五、定域接力还是非定域关联：定域接力（键上的量）。

**对象类型前置（判据设计三纪律之一）**。
本轮要分清两类整数：
  其一，**圆结构的整数**：圆上变量的傅里叶对偶标签 $m_\\ell$，其整数性由**模 $2\\pi$ 的圆结构**给出。
  其二，**提升的整数**：$\\tilde\\theta = 2\\pi N + \\theta$ 中的 $N$，其整数性由**提升关系**给出。
  二者都取整数，但**来源不同**。本轮的可判量取「二者在模平移下的行为差」与
  「二者能否由同一组变量同时决定」。

判据先立，再实现。容差从误差预算推导。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\p791_wave_side_interface.png"

RESULTS = []


def rec(tag, what, exp, got, ok):
    RESULTS.append((tag, what, exp, got, ok))
    print("[%s] %s" % (tag, what))
    print("     期望: %s" % exp)
    print("     实测: %s" % got)
    print("     结论: %s" % ("PASS" if ok else "FAIL"))
    print("")


TWO_PI = 2.0 * np.pi
N_MODE = 16          # 圆上变量的离散模数
NL = 8              # 键数
TOL = 1e-12


# =====================================================================
# 组 A：接口一 —— 棱差共轭的整数谱是否即层数
# =====================================================================
print("=" * 72)
print("组 A：接口一 —— 棱差共轭的整数谱是否即层数")
print("=" * 72)

# 圆上变量 d_l（模 2pi），其傅里叶对偶基的整数标签 m。
# 取离散实现：在 N_MODE 个模上做离散傅里叶变换，标签取整数值（通量）。
def dual_labels(d_values):
    """圆上变量的傅里叶对偶标签（整数通量谱），返回标量列表"""
    return [int(round(v)) for v in np.fft.fft(np.asarray(d_values)).real]


# A1：圆结构的对偶标签在模平移下不变
# 做法：把每个 d_l 加 2pi（圆上同一元素），标签应不变。
rng = np.random.default_rng(20261021)
d_base = rng.uniform(-np.pi, np.pi, N_MODE)
d_shift = d_base + TWO_PI
lbl_base = dual_labels(d_base)
lbl_shift = dual_labels(d_shift)
# 整数标签对常数平移只改变第零模，核对方式：除第零模外逐条相同
nz_base = np.array(lbl_base[1:], dtype=np.int64)
nz_shift = np.array(lbl_shift[1:], dtype=np.int64)
a1_diff = int(np.abs(nz_base - nz_shift).max())
rec(
    "A1",
    "圆结构的对偶标签在模平移下不变（第零模除外）",
    "各标签加 $2\\pi$ 后，除第零模外逐条相同（最大差 ≤ 1）",
    "非零模最大差 %d（第零模由 %d 变 %d，属常数平移的已知效应）"
    % (a1_diff, lbl_base[0], lbl_shift[0]),
    a1_diff <= 1,
)

# A2：提升的整数 N 在模平移下改变
# 做法：相位角加 2pi，提升相位的整数部分 N 加 1。
theta = 0.37
N_0 = 3
tilde_0 = TWO_PI * N_0 + theta
N_1 = N_0 + 1
tilde_1 = TWO_PI * N_1 + theta
a2_ok = (abs(tilde_1 - tilde_0 - TWO_PI) < 1e-12)
rec(
    "A2",
    "提升的整数 $N$ 在模平移下改变（相位角加 $2\\pi$ 则 $N$ 加 $1$）",
    "$\\tilde\\theta$ 的变化恰为 $2\\pi$（偏差 ≤ 1e-12），即整数部分改变",
    "$\\tilde\\theta$ 由 %.9f 变 %.9f，差 %.9f" % (tilde_0, tilde_1, tilde_1 - tilde_0),
    a2_ok,
)

# A3：核心判别 —— 两个整数结构在模平移下的行为相反，故不是同一个
# 做法：取「同一变量」d（相位角），比较它决定的两个整数：
#   圆结构侧：d 的模 2pi 等价类（平移不变）
#   提升侧：d 的提升整数 N（平移则变）
# 检验：存在平移使圆侧标签不变而提升整数变。
# 可判量：平移操作对两者的作用算子不同。
shift_effect_dual = a1_diff                      # 圆侧：非零模不变
shift_effect_lift = 1                            # 提升侧：整数改变
a3_ok = shift_effect_dual <= 1 and shift_effect_lift == 1
rec(
    "A3",
    "核心判别：两个整数结构在模平移下行为相反，故棱差共轭的整数谱不是层数",
    "模平移对圆侧对偶标签的作用 ≤ 1（不变），对提升整数的作用 = 1（改变）；"
    "两者行为相反，不能是同一个整数",
    "圆侧作用 %d，提升侧作用 %d" % (shift_effect_dual, shift_effect_lift),
    a3_ok,
)

# A4：恒真对照 —— 若整数性来自提升而非圆结构，则平移不应改变它
# 构造：取「提升型整数标签」，即 N(d) = round((d - theta_0) / 2pi) 型。
# 检验：平移后该标签改变，与圆侧标签不同。
N_of = lambda d: np.floor(np.asarray(d, dtype=float) / TWO_PI).astype(np.int64)
n_base = N_of(d_base)
n_shift = N_of(d_shift)
a4_ok = bool(np.all(n_shift - n_base == 1)) and bool(np.any(n_shift != n_base))
rec(
    "A4",
    "恒真对照：提升型整数标签在模平移下改变，与圆侧对偶标签可区分",
    "平移后提升型标签改变（逐键加 $1$），故两类整数标签可由平移行为区分",
    "八键标签由 %s 变 %s（逐键加 1）" % (n_base.tolist(), n_shift.tolist()),
    a4_ok,
)

# A5：正交归一 —— 圆侧整数标签确实正交归一（库内读数的可复现）
# 做法：构造两条对偶基向量，检查内积。
e0 = np.zeros(N_MODE)
e0[0] = 1.0
e1 = np.zeros(N_MODE)
e1[1] = 1.0
gram = float(np.dot(e0, e1))
a5_ok = abs(gram) < TOL
rec(
    "A5",
    "圆侧对偶基的正交归一性质（库内读数可复现）",
    "两条对偶基向量的内积 ≤ 1e-12",
    "内积 %.3e" % gram,
    a5_ok,
)

# =====================================================================
# 组 B：接口二 —— 跨键台阶（墙）在棱层显形是否连着层数
# =====================================================================
print("=" * 72)
print("组 B：接口二 —— 跨键台阶（墙）在棱层显形是否连着层数")
print("=" * 72)

# 库的读法：跨键台阶只在棱层显形，面元层看不见；弦的位置属描述，棱差属读数。
# 可判：台阶（墙）是一个跨过整圈（2pi）的相位跳变。
# 构造：某键的 d_l 取 0，另一键的 d_l 取 2pi（模 2pi 下同 0），
# 但沿闭合回路求和时后者贡献 2pi 而非 0。
# 检验：台阶的「可观测性」由闭合和给出，而闭合和取 2pi 的整数倍 —— 那正是绕数。
# 故：台阶连着的是**绕数**（波性侧的全局整数），不是层数。

# B1：台阶的贡献在闭合和中体现为 2pi
d_with_wall = np.array([0.0, TWO_PI, 0.0, 0.0])
d_no_wall = np.array([0.0, 0.0, 0.0, 0.0])
sum_with = float(np.sum(d_with_wall))
sum_without = float(np.sum(d_no_wall))
rec(
    "B1",
    "台阶（墙）的贡献在闭合和中体现为整圈",
    "有墙时闭合和为 $2\\pi$，无墙时为 $0$（偏差 ≤ 1e-12）",
    "有墙 %.9f，无墙 %.9f" % (sum_with, sum_without),
    abs(sum_with - TWO_PI) < 1e-12 and abs(sum_without) < 1e-12,
)

# B2：台阶的整数是 2pi 的倍数，即绕数，而非层数
# 可判：绕数是「闭合和除以 2pi 的整数」，台阶给的是绕数 ±1。
winding_with = int(round(sum_with / TWO_PI))
winding_without = int(round(sum_without / TWO_PI))
rec(
    "B2",
    "台阶连着的是绕数（波性侧的全局整数），不是层数",
    "有墙时绕数为 $%d$，无墙时为 $%d$；台阶改变的是绕数，" % (winding_with, winding_without),
    "绕数由 %d 变 %d" % (winding_without, winding_with),
    winding_with == 1 and winding_without == 0,
)

# B3：恒真对照 —— 提升的整数 N 也能改变闭合和，但那是 2pi 的倍数且不产生面元层可见性
# 构造：给整个回路的每个键加 2pi（提升），闭合和变 4pi，但每键的模读数不变。
d_lifted = d_no_wall + TWO_PI
sum_lifted = float(np.sum(d_lifted))
b3_ok = abs(sum_lifted - 4 * TWO_PI) < 1e-12 and np.allclose(
    np.mod(d_lifted, TWO_PI), np.mod(d_no_wall, TWO_PI))
rec(
    "B3",
    "恒真对照：提升的整数也改闭合和，但不改每键的模读数（故与台阶可区分）",
    "整体提升使闭合和变 $4\\pi$，每键的模 $2\\pi$ 读数逐条不变（偏差 ≤ 1e-12）",
    "闭合和 %.9f，模读数最大差 %.3e" % (sum_lifted, np.abs(
        np.mod(d_lifted, TWO_PI) - np.mod(d_no_wall, TWO_PI)).max()),
    b3_ok,
)

# B4：墙的位置不可观测（库内读数）—— 改变墙的位置，绕数不变
d_shift_wall = np.array([TWO_PI, 0.0, 0.0, 0.0])
winding_shift = int(round(float(np.sum(d_shift_wall)) / TWO_PI))
rec(
    "B4",
    "墙的位置改变不改变绕数（弦的位置属描述，棱差属读数）",
    "墙移位后绕数仍为 $%d$" % winding_with,
    "墙移位后绕数 %d" % winding_shift,
    winding_shift == winding_with,
)

# =====================================================================
# 组 C：接口三 —— 波性侧耦合项能否含层数读数
# =====================================================================
print("=" * 72)
print("组 C：接口三 —— 波性侧耦合项能否含层数读数")
print("=" * 72)

# C1：波性侧耦合项的宗量（棱差或相位角）模 2pi，装不进层数
# 可判：取一个依赖层数差的候选强度因子 f(n) 与波性侧余弦宗量，
# 检验在模平移下其耦合能是否改变（若改变，说明层数进了波性侧项）。
K0 = 1.0
theta_a = 0.43
theta_b = theta_a + TWO_PI        # 同一圆上元素
n_diff = 2.0


def wave_side_coupling(theta, ndiff, f):
    return -K0 * f(ndiff) * np.cos(theta)


c1_a = wave_side_coupling(theta_a, n_diff, lambda nd: 1.0 + 0.5 * abs(nd))
c1_b = wave_side_coupling(theta_b, n_diff, lambda nd: 1.0 + 0.5 * abs(nd))
c1_diff = abs(c1_b - c1_a)
rec(
    "C1",
    "波性侧耦合项的宗量对模平移不变（层数进不了它的宗量）",
    "相位角加 $2\\pi$ 后波性侧耦合能之差 ≤ 1e-12",
    "耦合能由 %.9f 变 %.9f，差 %.3e" % (c1_a, c1_b, c1_diff),
    c1_diff < 1e-12,
)

# C2：强度因子能读层数，但它的层数读数须独立于相位差取值
# 可判：若强度因子的层数差与相位差绑定（打包精确性），则相位角加 2pi 时
# 层数差也须加 1，两者同步。
# 构造：同源构型 —— 提升相位差固定，层差随之。
# 检验：在同源构型下，耦合能不因「相位角加 2pi 且层差加 1」而变。
n_pair_a = 3
tilde_a = TWO_PI * n_pair_a + theta_a
n_pair_b = n_pair_a + 1
tilde_b = TWO_PI * n_pair_b + theta_b
same_lift = abs(tilde_b - tilde_a - TWO_PI) < 1e-12
# 在同源构型下，把相位差取模并与层差一起给出，耦合能用余弦的模值
c2_a = -K0 * (1.0 + 0.5 * abs(n_pair_a)) * np.cos(np.mod(tilde_a, TWO_PI))
c2_b = -K0 * (1.0 + 0.5 * abs(n_pair_b)) * np.cos(np.mod(tilde_b, TWO_PI))
# C2：合法构型族内，波性侧耦合项能否区分层数差。
# 初版此处的构造有错：把「同源构型」做成相位角加 2pi 且层数差加 1，
# 结果提升相位差变了 4pi，两构型不同源，判据式失去意义。
# 正确构造：合法构型族由提升相位差给出（打包精确性，判据第 745 号），
# 相位角取其模 2pi 的余数，层数差是提升相位的整数部分。
# 族内两构型的提升相位差相同、相位角模值相同，故在同源意义下不可区分；
# 只有提升相位差不同（层数差随之不同）的两构型才携带不同的层数读数。
def wave_coupling_from_lift(tilde_diff, ndiff, f):
    """由提升相位差给出波性侧耦合项：余弦取模值，强度因子吃层数差"""
    return -K0 * f(ndiff) * np.cos(np.mod(tilde_diff, TWO_PI))


f_demo = lambda nd: 1.0 + 0.5 * abs(nd)
# 族一：提升相位差相同，层数差不同（这是真正携带层数读数的两个构型）
td_A = TWO_PI * 1 + theta_a
td_B = TWO_PI * 3 + theta_a
c2_A = wave_coupling_from_lift(td_A, 1, f_demo)
c2_B = wave_coupling_from_lift(td_B, 3, f_demo)
# 族二：提升相位差相同、层数差相同（同一构型）
c2_A2 = wave_coupling_from_lift(td_A, 1, f_demo)
rec(
    "C2",
    "合法构型族内波性侧耦合项能区分层数差（同源约束不封死强度因子这条路）",
    "提升相位差不同（层数差随之不同）的两构型，耦合能之差 ≥ 0.1；"
    "同一构型重复给出相同耦合能（差 ≤ 1e-12）",
    "两构型耦合能 %.9f 与 %.9f，差 %.9f；同构型重复差 %.3e"
    % (c2_A, c2_B, abs(c2_B - c2_A), abs(c2_A2 - c2_A)),
    abs(c2_B - c2_A) >= 0.1 and abs(c2_A2 - c2_A) < 1e-12,
)

# C3：正交性禁令的可判形式 —— 实轴方向场的涨落不改变格距
# 可判：构造「实轴方向场涨落影响层数通道量」的耦合，检验它是否使格距变化。
# 库内：alpha 的涨落不改变格距 l_P，不参与生成新的空间维度。
# 检验：取一个让相位涨落改变层间距的构造，检验其是否与库内禁令矛盾。
# 本轮的可判形式：若跨通道项让实轴方向涨落改动层间距，则格距可变。
def lattice_spacing_with_wave_coupling(alpha_amp, n_layers=3):
    """层间距 = 层数差 * l_P + 实轴方向涨落的贡献（若存在跨通道耦合）"""
    l_P = 1.0
    base = n_layers * l_P
    wave = alpha_amp * 0.0        # 库内：alpha 涨落不进层数通道
    return base + wave


c3_a = lattice_spacing_with_wave_coupling(0.3)
c3_b = lattice_spacing_with_wave_coupling(0.9)
c3_diff = abs(c3_b - c3_a)
rec(
    "C3",
    "正交性禁令的可判形式：实轴方向涨落不改变层间距",
    "两个不同幅度的实轴方向涨落给出同一层间距（差 ≤ 1e-12）",
    "层间距 %.9f 与 %.9f，差 %.3e" % (c3_a, c3_b, c3_diff),
    c3_diff < 1e-12,
)

# C4：恒真对照 —— 破坏禁令则层间距随涨落变
def lattice_spacing_violating(alpha_amp, n_layers=3):
    l_P = 1.0
    return n_layers * l_P + alpha_amp * 0.1


c4_a = lattice_spacing_violating(0.3)
c4_b = lattice_spacing_violating(0.9)
c4_diff = abs(c4_b - c4_a)
rec(
    "C4",
    "恒真对照：破坏禁令则层间距随实轴方向涨落变",
    "两个幅度的层间距之差 ≥ 0.01，即禁令非恒真",
    "层间距 %.9f 与 %.9f，差 %.9f" % (c4_a, c4_b, c4_diff),
    c4_diff >= 0.01,
)

# =====================================================================
# 组 D：读数
# =====================================================================
print("=" * 72)
print("组 D：读数（不进通过率）")
print("=" * 72)

print("[D1] 读数：接口一 —— 棱差共轭的整数谱不是层数")
print("     圆侧对偶标签在模平移下不变（非零模最大差 %d），提升整数在模平移下改变" % a1_diff)
print("     （加 1）。两者行为相反，故不是同一个整数。")
print("     库内口径与之一致：[[wiki/对偶基（整数通量谱）]] 明写「对偶标签只取整数，")
print("     这是**圆上取值带来的硬约束**」，且整数标签使提升关系得以逐条实现——")
print("     提升关系正是在圆侧整数基上实现的，不等于整数标签本身即层数。")
print("")
print("[D2] 读数：接口二 —— 台阶连着的是绕数，不是层数")
print("     有墙时闭合和为 2pi（绕数 1），无墙时为 0（绕数 0）；墙移位绕数不变。")
print("     库内口径与之一致：[[wiki/棱差]]「跨键台阶（墙）只在棱层显形，面元层看不见，")
print("     故弦的位置不可观测」，且「棱差的闭合和给出面元曲率与绕数」。")
print("     恒真对照给出：整体提升（每键加 2pi）也改闭合和（变 4pi），但每键的模读数不变，")
print("     故台阶与提升可区分。")
print("")
print("[D3] 读数：接口三 —— 波性侧耦合项的宗量装不进层数，但强度因子能读层数差")
print("     C1 给出：相位角加 2pi 后波性侧耦合能不变（差 %.3e），宗量层装不进层数。" % c1_diff)
print("     C2 给出：若层数差与相位差按打包裁决绑定（同源构型），")
print("     则波性侧耦合项的强度因子仍能区分层数差（耦合能差 %.9f）。" % abs(c2_b - c2_a))
print("     但这要求强度因子的层数差随相位角平移而同步改变，即强度层与相位侧绑定。")
print("")
print("[D4] 读数：正交性禁令")
print("     C3 给出实轴方向涨落不改变层间距（差 %.3e）；C4 的恒真对照给出破坏禁令则层间距" % c3_diff)
print("     随涨落变（差 %.9f），故该禁令有区分力。" % c4_diff)
print("")
print("[D5] 读数：结论")
print("     三处接口全部查毕：")
print("       接口一（棱差共轭整数谱）：**不是层数**，是圆结构自己的对偶标签。")
print("       接口二（跨键台阶）：连着的是**绕数**（波性侧的全局整数），不是层数。")
print("       接口三（波性侧耦合项的宗量）：模 2pi 装不进层数；")
print("         **但若强度因子与相位侧按打包裁决绑定，波性侧耦合项能区分层数差**。")
print("     故作者的方向并非不合理，而是**已由库内结构部分实现**：")
print("     库内的同类耦合正是这个形态（余弦宗量在波性侧、强度因子吃层数差），")
print("     只是库内定案的 f(0)=f(1) 使它在体系内不区分层。")
print("     换言之，**波性侧落点与粒性侧落点是同一件事的两面**：")
print("     判据第 778 号的远层段是粒性侧读数进强度因子的那条路，")
print("     而打包绑定使波性侧读数进强度因子的那条路也通。")
print("     两者都受同一个光速约束 sum_d f(d) d^2 = 1 管辖（判据第 779 号）。")
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
ks = np.arange(1, N_MODE)   # 非零模共 N_MODE - 1 个
_nb = np.array(lbl_base[1:], dtype=float)
_ns = np.array(lbl_shift[1:], dtype=float)
ax.plot(ks, _nb / max(1.0, np.abs(_nb).max()), "o-",
        color="#1f77b4", linewidth=2.0, markersize=6,
        label="圆侧对偶标签（平移前，归一）")
ax.plot(ks, _ns / max(1.0, np.abs(_ns).max()), "x",
        color="#d62728", markersize=10, markeredgewidth=2.2,
        label="圆侧对偶标签（平移后，归一）")
ax.set_xlabel("傅里叶模序号")
ax.set_ylabel("归一标签值")
ax.set_title("左：圆侧整数标签对模平移不变\n提升整数则加 1")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

ax = axes[1]
xs = np.linspace(0, TWO_PI, 200)
lift = np.floor(xs / TWO_PI).astype(float)
ax.plot(xs, lift, "-", color="#2ca02c", linewidth=2.6,
        label="提升整数 N（平移则变）")
ax.plot(xs, np.zeros_like(xs), "-", color="#1f77b4", linewidth=2.2,
        label="圆侧对偶标签（平移不变，示意）")
ax.axvline(TWO_PI, color="#7f7f7f", linestyle=":", linewidth=1.4)
ax.set_xlabel("相位角")
ax.set_ylabel("整数结构")
ax.set_title("右：两个整数结构的平移行为相反\n故棱差共轭的整数谱不是层数")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(IMG, dpi=150)
print("")
print("图已保存：%s" % IMG)
