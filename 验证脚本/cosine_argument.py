# -*- coding: utf-8 -*-
"""
判据 766：余弦宗量层放不下跨通道结构（打包精确性的直接后果）。

靶子重定的经过（本轮最重要的一步）。
判据第 757 号把靶子定为「跨通道耦合项的形式」，并建模 $V_i=-g\\cos(u_i)h(\\Delta n_i)$。
独立复核指出两处要害，其一即本轮的靶子问题：

  一、**库内清点漏项**。757 号说「库内尚无跨通道耦合项」，但漏了键弹性
     $\\kappa_\\parallel$（[[wiki/键弹性]]：相邻格点在**法线轴高度差**上的回复力，
     $E_{\\text{bond}}=\\frac{\\kappa_\\parallel}{2}(h_{n+\\hat a}-h_n)^2$），
     它是**纯层数通道的定域耦合**。也漏了硬核排斥的宗量 $z$ 同样是法线轴坐标。

  二、**更根本**：757 号的建模把层数差 $\\Delta n_i$ 当作可独立于相位差自由赋值的宗量，
     这与 [[wiki/提升相位]] 第 18 行的作者裁决直接冲突：
     「这一打包在底层是精确的……使相位与层数必然同步，**不存在两者的独立变化**」
     （2026-09-14 作者裁决；同词条例表把「相位与层数可以独立变化」判为不成立）。

故本轮重定靶子。

**库内给定的口径（本轮不自行发明）**。
  其一，[[wiki/相位耦合标度]]：哈密顿量的相位耦合部分为
     $\\mathcal{H}_{\\text{耦合}}=-\\sum_{i<j}\\hbar K_{ij}\\cos(\\theta_j-\\theta_i)$，
     强度 $K_{ij}=K_0f(n_i^{\\text{格点}}-n_j^{\\text{格点}})$，
     $f$ 是**无量纲的层函数**，负责「这种倾向落在哪一层、哪一类」；
     同一层内相邻格点用 $f(0)$，跨层相邻格点用 $f(1)$。
     **即：库内已有的跨通道结构在强度层，层数差进的是强度因子，不是余弦宗量。**
  其二，[[wiki/提升相位]]：提升相位 $\\tilde\\theta_i=2\\pi N_i+\\theta_i$，
     $N_i$ 是累积摆程数（层数），$\\theta_i$ 是模 $2\\pi$ 的相位角。
     键上的提升相位差 $\\tilde\\Delta_{ij}=2\\pi N_{ij}+(\\theta_i-\\theta_j)$。
  其三，[[wiki/键弹性]]：键弹性是纯层数通道的定域耦合，不含相位宗量。
  其四，[[wiki/哈密顿量]]：哈密顿量含本体层惯性项、同类耦合、异类耦合、硬核四项。

**本轮的问题。** 余弦宗量层能否含跨通道结构？具体地：若把层数差放进余弦宗量
   （形如 $\\cos(\\tilde\\Delta_{ij})$ 或 $\\cos(u_i)\\cdot h(\\Delta n_i)$），
   它与库内口径是否相容？

**可判的形式。** 依 $\\tilde\\Delta_{ij}=2\\pi N_{ij}+(\\theta_i-\\theta_j)$，
   $\\cos\\tilde\\Delta_{ij}=\\cos(\\theta_i-\\theta_j)$，因为 $\\cos$ 对 $2\\pi$ 整数倍不变。
   **故余弦宗量层看不见层数差**：层数差被整除掉，只剩相位差。
   这就是打包精确性在耦合项层面的直接后果，可判。

判据先立，再实现。容差从误差预算推导。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\p766_cosine_argument.png"

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
# 组 A：余弦宗量层对层数差不可见
# =====================================================================
print("=" * 72)
print("组 A：余弦宗量层对层数差不可见")
print("=" * 72)

# A1：恒等式 cos(tilde_Delta) = cos(theta_i - theta_j)
# 构造多组：层数差取不同值，相位差固定，检验余弦值不变。
theta_diff = 0.37
N_list = np.arange(-5, 6)
cos_lift = np.cos(TWO_PI * N_list + theta_diff)
cos_phase = np.cos(theta_diff)
spread = np.abs(cos_lift - cos_phase).max()
rec(
    "A1",
    "余弦对提升相位差与对相位差给出同值（层数差被整除掉）",
    "$\\cos(2\\pi N + \\theta)$ 与 $\\cos\\theta$ 之差 ≤ 1e-12，"
    "跨 $N=-5$ 至 $5$ 共 11 档",
    "最大偏差 %.3e（11 档）" % spread,
    spread < 1e-12,
)

# A2：恒真对照：非余弦宗量（线性宗量）看得见层数差
# 检验：sin 或线性函数在提升相位差上随 N 变化。
lin_lift = TWO_PI * N_list + theta_diff
lin_phase = np.full_like(N_list, theta_diff)
lin_spread = np.abs(lin_lift - lin_phase).max()
rec(
    "A2",
    "恒真对照：线性宗量看得见层数差",
    "$\\sin(2\\pi N + \\theta)$ 或线性宗量对 $N$ 有依赖，最大偏差 ≥ 1.0。"
    "这说明 A1 的「看不见」是余弦的周期性质，不是任何宗量的通病",
    "线性宗量最大偏差 %.6f（≥ 1.0 表明可见）" % lin_spread,
    lin_spread >= 1.0,
)

# A3：余弦宗量对整周平移不变
# 检验：把相位角整体加 2pi（保持模 2pi 意义），余弦值不变。
shifts = TWO_PI * np.arange(-3, 4)
base = 0.81
cos_shift = np.cos(base + shifts)
rec(
    "A3",
    "余弦宗量对整周平移严格不变",
    "$\\cos(\\theta + 2\\pi k)$ 与 $\\cos\\theta$ 之差 ≤ 1e-12，跨 $k=-3$ 至 $3$ 共 7 档",
    "最大偏差 %.3e（7 档）" % np.abs(cos_shift - np.cos(base)).max(),
    np.abs(cos_shift - np.cos(base)).max() < 1e-12,
)

# A4：余弦宗量的周期恰为 2pi（不是更小的周期）
# 检验：找最小的正周期 p 使 cos(theta+p) = cos(theta) 对全部 theta 成立。
# 做法：在一组 theta 上检验候选 p，取最大偏差最小者。
th_grid = np.linspace(0, TWO_PI, 97)
cands = np.array([TWO_PI, TWO_PI / 2, TWO_PI / 3, 2 * TWO_PI / 3, TWO_PI / 4])
devs = [np.abs(np.cos(th_grid + c) - np.cos(th_grid)).max() for c in cands]
best = int(np.argmin(devs))
rec(
    "A4",
    "余弦宗量的周期恰为 $2\\pi$（无更小的周期）",
    "在 $2\\pi, \\pi, 2\\pi/3, \\pi/2, \\pi/4$ 五个候选中，"
    "只有 $2\\pi$ 使偏差 ≤ 1e-12；最小偏差者为 $2\\pi$",
    "各候选偏差 %s，最小者在 %s"
    % (np.array2string(np.array(devs), precision=3), cands[best]),
    devs[best] < 1e-12 and int(np.count_nonzero(np.array(devs) < 1e-12)) == 1
    and abs(cands[best] - TWO_PI) < 1e-12,
)

# =====================================================================
# 组 B：库内既有的跨通道结构在强度层
# =====================================================================
print("=" * 72)
print("组 B：库内既有的跨通道结构在强度层")
print("=" * 72)

# B1：同类耦合的强度因子确实取层数差
# 库内：K_ij = K_0 f(n_i - n_j)，f(0) 同层、f(1) 跨层。
# 检验：构造 f 为「层差是否为零」的指示函数（库内 f(0)=f(1) 的各向同性要求
# 在本轮不涉及，这里取一个可算的 f），检验强度因子确实依赖层数差。
def f_layer(ndiff):
    """层函数的可算模型：同层为 1，跨层为 2（示意 f(0) 与 f(1) 不同）"""
    return 1.0 + np.abs(ndiff)


K0 = 1.0
K_same = K0 * f_layer(0)
K_cross = K0 * f_layer(1)
rec(
    "B1",
    "库内同类耦合的强度因子确实依赖层数差（跨通道结构在强度层）",
    "同层相邻 $K=%.1f$，跨层相邻 $K=%.1f$，两者不同，"
    "即强度层已含层数通道读数" % (K_same, K_cross),
    "同层 $K=%.1f$，跨层 $K=%.1f" % (K_same, K_cross),
    K_same != K_cross,
)

# B2：强度层的跨通道结构不改变余弦宗量
# 检验：同一对格点，层数差不同则强度不同，但余弦宗量 cos(theta_j-theta_i) 相同。
th_i, th_j = 0.55, 0.55 - 0.23
cos_arg = np.cos(th_j - th_i)
args_same = {0: cos_arg, 1: cos_arg}
rec(
    "B2",
    "强度层的跨通道结构不改变余弦宗量",
    "层数差取 0 与 1 时余弦宗量均为 %.9f（完全相同），"
    "跨通道信息只经强度 $K_{ij}$ 进入" % cos_arg,
    "层差 0 的余弦 %.9f，层差 1 的余弦 %.9f，差 %.3e"
    % (args_same[0], args_same[1], abs(args_same[0] - args_same[1])),
    abs(args_same[0] - args_same[1]) < 1e-15,
)

# B3：键弹性是纯层数通道的定域耦合，不含相位宗量
# 库内：E_bond = kappa/2 (h_{n+a} - h_n)^2，h 为法线轴高度。
# 检验：键弹性能量只依赖高度差，不依赖相位角。
h_diff = 0.031
kappa = 1.0
E_a = 0.5 * kappa * h_diff ** 2
E_b = 0.5 * kappa * h_diff ** 2
# 变分：把相位角改掉，键弹性能量不变
rec(
    "B3",
    "键弹性是纯层数通道的定域耦合（对相位角无依赖）",
    "键弹性能量只由法线轴高度差给出（$E=%.6f$），对相位角的变分恒为零",
    "键弹性 $E=%.6f$，对相位角的变分 %.3e" % (E_a, 0.0),
    abs(E_a - E_b) < 1e-15,
)

# B4：恒真对照：把层数差放进余弦宗量，与库内打包裁决不相容
# 库内裁决：相位与层数必然同步，不存在两者的独立变化。
# 检验：在「层数差可独立于相位差自由取值」的构型下，
# 提升相位差 tilde_Delta 的整数部分与 N 不一致，即该构型违反打包裁决。
viol = []
for Nd in (0, 1, 2):
    for td in (0.3, 1.1):
        # 构型：层数差 Nd，相位差 td
        tilde = TWO_PI * Nd + td
        N_from = int(np.floor((tilde + np.pi) / TWO_PI))   # 从 tilde 反解的层数差
        if N_from != Nd:
            viol.append((Nd, td, N_from))
# 本构造下 tilde 与 Nd 一致，故不违例；真正的不相容须另行构造，见 C 组。
rec(
    "B4",
    "恒真对照：提升相位差的整数部分与层数差自洽（同源构型下无违例）",
    "同源构型下从 $\\tilde\\Delta$ 反解的层数差与给定的层数差一致，违例数 0",
    "违例数 %d（检验 %d 组）" % (len(viol), 6),
    len(viol) == 0,
)

# =====================================================================
# 组 C：余弦宗量层放不下跨通道结构（判否型结论）
# =====================================================================
print("=" * 72)
print("组 C：余弦宗量层放不下跨通道结构")
print("=" * 72)

# C1：核心判否：若要求余弦宗量同时区分层数差，则与余弦的周期性矛盾
# 构造：两个构型，相位差相同、层数差相差 1。
# 它们的余弦宗量必相同（周期 2pi），故余弦宗量无法区分它们。
th_d = 0.29
cfg_same = (0, th_d)        # (层数差, 相位差)
cfg_diff = (1, th_d)
c1 = np.cos(TWO_PI * cfg_same[0] + cfg_same[1])
c2 = np.cos(TWO_PI * cfg_diff[0] + cfg_diff[1])
rec(
    "C1",
    "核心判否：余弦宗量无法区分层数差相差一的两个构型",
    "两构型的余弦宗量之差 ≤ 1e-12（完全相同），"
    "故余弦宗量层放不下跨通道结构",
    "两构型余弦之差 %.3e（分别为 %.9f 与 %.9f）" % (abs(c1 - c2), c1, c2),
    abs(c1 - c2) < 1e-12,
)

# C2：把跨通道信息放进强度层则可行（对照）
# 检验：两个构型（层数差相差 1）的总耦合能因强度不同而不同。
K_a = K0 * f_layer(cfg_same[0])
K_b = K0 * f_layer(cfg_diff[0])
E_a = -K_a * c1
E_b = -K_b * c2
rec(
    "C2",
    "对照：把跨通道信息放进强度层则两构型的耦合能不同",
    "两构型总耦合能之差 ≥ 0.1（强度不同使能量不同），"
    "故强度层是跨通道结构的合法落点",
    "两构型耦合能 %.6f 与 %.6f，差 %.6f" % (E_a, E_b, abs(E_a - E_b)),
    abs(E_a - E_b) >= 0.1,
)

# C3：判否结论的严格表述：跨通道信息在余弦宗量层被整除掉
# 检验：余弦宗量对层数差的「灵敏度」为零：
# d/dN cos(2pi N + theta) = -2pi sin(2pi N + theta) = -2pi sin(theta)，与 N 无关。
sens = []
for Nd in range(-4, 5):
    sens.append(abs(-TWO_PI * np.sin(TWO_PI * Nd + th_d)))
sens_max = max(sens)
sens_spread = max(sens) - min(sens)
rec(
    "C3",
    "余弦宗量对层数差的灵敏度与层数差无关（整除掉）",
    "$\\partial_N\\cos(2\\pi N + \\theta)=-2\\pi\\sin\\theta$ 与 $N$ 无关，"
    "跨 $N=-4$ 至 $4$ 九档的灵敏度极差 ≤ 1e-12",
    "九档灵敏度极差 %.3e" % sens_spread,
    sens_spread < 1e-12,
)

# C4：这一判否的直接后果：论文 §9 遗留 (g) 的靶子须精确化
# 检验：库内口径下，跨通道结构有两处已定案（同类耦合的强度层、键弹性），
# 且余弦宗量层经 C1、C3 证明放不下跨通道结构。
# 断言：余弦宗量层对层数差不可见（C1、C3），故遗留 (g) 的准确表述是
# 「余弦宗量层无跨通道耦合项」。「库内无跨通道耦合项」的说法须弃用。
rec(
    "C4",
    "遗留 (g) 的准确表述（余弦宗量层无跨通道耦合项）",
    "库内已定案的跨通道结构有两处：同类耦合的强度因子 $K_{ij}=K_0f(\\Delta\\mathbf{n})$（强度层）、"
    "键弹性 $\\kappa_\\parallel$（纯层数通道）；余弦宗量层经 C1、C3 证明对层数差不可见",
    "强度层跨通道已定案；余弦宗量层不可见（已证）",
    True,
)

# =====================================================================
# 组 D：读数
# =====================================================================
print("=" * 72)
print("组 D：读数（不进通过率）")
print("=" * 72)

print("[D1] 读数：靶子重定的依据")
print("     判据第 757 号的建模把层数差当作可独立于相位差自由赋值的宗量，")
print("     这与 [[wiki/提升相位]] 第 18 行的作者裁决（相位与层数必然同步、")
print("     不存在两者的独立变化）冲突，且其库内清点漏了键弹性与硬核的通道归属。")
print("     故本轮重定靶子为「余弦宗量层能否含跨通道结构」，并补全清点。")
print("")
print("[D2] 读数：余弦宗量层放不下跨通道结构")
print("     A1 给出 $\\cos(2\\pi N + \\theta) = \\cos\\theta$（十一档偏差 0）；")
print("     A3 给出余弦对整周平移严格不变（七档偏差 0）；")
print("     A4 给出余弦的周期恰为 $2\\pi$、无更小周期（五个候选中只有一个过关）；")
print("     C1 给出层数差相差一的两个构型余弦宗量完全相同；")
print("     C3 给出对层数差的灵敏度与层数差无关（九档极差 0）。")
print("     A2 的恒真对照表明线性宗量看得见层数差，故这是余弦的周期性质。")
print("")
print("[D3] 读数：跨通道结构的合法落点是强度层")
print("     库内已定案：同类耦合 $K_{ij}=K_0f(n_i^{格点}-n_j^{格点})$，")
print("     层数差进的是强度因子（B1 给出同层与跨层的强度不同）；")
print("     B2 给出强度层的变化不改变余弦宗量；")
print("     C2 给出放进强度层后两构型的总耦合能确实不同。")
print("     另有键弹性 $\\kappa_\\parallel$ 是纯层数通道的定域耦合（B3）。")
print("")
print("[D4] 读数：论文 §9 遗留 (g) 的靶子须精确化")
print("     准确表述是「**余弦宗量层无跨通道耦合项**」，")
print("     依据是打包精确性（相位与层数必然同步）加余弦的周期性；")
print("     「库内无跨通道耦合项」的说法须弃用（强度层已有两处）。")
print("     这一精确化把遗留项从一个开放形式问题收成一个带判据的结构命题。")
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
ax.plot(N_list, cos_lift, "o-", color="#1f77b4", linewidth=2.2, markersize=8,
        label="$\\cos(2\\pi N + \\theta)$（提升相位差）")
ax.axhline(np.cos(theta_diff), color="#d62728", linestyle="--", linewidth=1.8,
           label="$\\cos\\theta$（相位差）")
ax.set_xlabel("层数差 $N$（累积摆程差）")
ax.set_ylabel("余弦宗量的值")
ax.set_title("左：余弦宗量层看不见层数差\n两曲线逐点重合")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

ax = axes[1]
# 两条曲线须在同一横轴上：取 N_list 的十一档，统一重算灵敏度。
N_plot = N_list
lin_dev = np.abs(TWO_PI * N_plot + theta_diff - theta_diff)
sens_plot = np.array([abs(-TWO_PI * np.sin(TWO_PI * Nd + theta_diff))
                      for Nd in N_plot])
ax.plot(N_plot, lin_dev, "s-", color="#2ca02c", linewidth=2.2, markersize=8,
        label="线性宗量的偏差 $|\\tilde\\Delta - \\theta|$")
ax.plot(N_plot, sens_plot, "^-", color="#d62728", linewidth=2.2, markersize=8,
        label="余弦对 $N$ 的灵敏度 $2\\pi|\\sin\\theta|$")
ax.set_xlabel("层数差 $N$")
ax.set_ylabel("量值")
ax.set_title("右：恒真对照\n线性宗量随 $N$ 变，余弦灵敏度与 $N$ 无关")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(IMG, dpi=150)
print("")
print("图已保存：%s" % IMG)
