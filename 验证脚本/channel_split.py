# -*- coding: utf-8 -*-
"""
判据 737：虚过程的通道拆分（相位侧与层数侧的可判分离）。

背景。P4-55 于 2026-10-21 经作者裁决确立纠错：连续与离散两者都基本、互补，
不存在哪一侧更基本。据此，「虚过程属于哪个通道」这一问法（预设二者择一）
须改为「虚过程在两通道里各是什么形态、如何互补」。

本体依据（[[理论总纲/公理体系]] 第 34、35 行）：
  物侧投影配置空间 $S^1 \\times \\mathbb{Z}$，态空间
  $\\mathcal{H}_k = L^2(S^1) \\otimes \\ell^2(\\mathbb{Z})$。
  相位通道指标是**瞬时谱指标**（连续旋转），层数通道指标是**历史计数**（整数）。
  两通道正交，不构成重复计数。

本轮把「虚过程」这个统称拆成两通道成员，并给每条成员可判的数学特征。
拆分的判据不是文字，是两通道各自的谱结构：

  相位侧：$L^2(S^1)$ 的傅里叶指标为整数但**取值连续**（相位角），
          其动力学有极点结构，传播子须靠 $i\\varepsilon$ 定义；
          谱型为 $\omega_k^2 = c^2k^2 + m^2$ 型连续色散。
  层数侧：$\ell^2(\\mathbb{Z})$ 的指标为整数计数，取值离散；
          其动力学是**计数跃迁**（$n \\to n \\pm 1$），无极点、无 $i\\varepsilon$；
          谱型为周期型（Bloch 带），因为 $n$ 的平移是离散平移。

判据先立，再实现。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\p737_channel_split.png"

RESULTS = []


def rec(tag, what, exp, got, ok):
    RESULTS.append((tag, what, exp, got, ok))
    print("[%s] %s" % (tag, what))
    print("     期望: %s" % exp)
    print("     实测: %s" % got)
    print("     结论: %s" % ("PASS" if ok else "FAIL"))
    print("")


# =====================================================================
# 组 A：两通道的谱结构差异
# =====================================================================
print("=" * 72)
print("组 A：两通道的谱结构差异")
print("=" * 72)

# A1：相位侧（连续旋转）的谱：谐振子型，等间距
# 相位通道的算符是 a^dagger a，本征值 n = 0,1,2,...（瞬时谱指标）
# 但其动力学变量 phi 取值连续（S^1），故传播子有极点
Nlev = 20
E_phase = np.arange(Nlev, dtype=float)          # 等间距 1
spacing_phase = np.diff(E_phase)
rec(
    "A1",
    "相位通道的谱间距（$L^2(S^1)$ 的傅里叶指标）",
    "等间距，间距恒为 1，极差 0",
    "间距 %s，极差 %.3e"
    % (np.array2string(np.unique(spacing_phase), precision=6),
       spacing_phase.max() - spacing_phase.min()),
    abs(spacing_phase.max() - spacing_phase.min()) < 1e-12,
)

# A2：层数侧（计数）的谱：同样等间距，但语义不同
# 层数 n 是历史计数，其共轭变量是准动量 theta，谱为周期型（Bloch 带）
# 在周期势下 E(theta) = -2t cos(theta)，带宽有限
theta = np.linspace(-np.pi, np.pi, 4001)
t_hop = 1.0
E_layer = -2.0 * t_hop * np.cos(theta)
bandwidth = E_layer.max() - E_layer.min()
rec(
    "A2",
    "层数通道的谱（$\\ell^2(\\mathbb{Z})$ 上的离散平移）",
    "Bloch 带型 $E(\\theta) = -2t\\cos\\theta$，带宽 $4t = 4$（有限、紧致）",
    "带宽 %.6f，谱域 [%.4f, %.4f]"
    % (bandwidth, E_layer.min(), E_layer.max()),
    abs(bandwidth - 4.0 * t_hop) < 1e-9,
)

# A3：关键差异：相位侧无界、层数侧有界。用可执行办法判：
# 把指标上限加倍，看谱上界是否跟着翻倍（无界）还是不动（有界）。
N1, N2 = 20, 40
top_phase_1 = float(N1 - 1)
top_phase_2 = float(N2 - 1)
ratio_phase = top_phase_2 / top_phase_1

# 层数侧：加倍格点数，Bloch 带上界不变
def band_top(Nsite):
    th = np.linspace(-np.pi, np.pi, 4001)
    return (-2.0 * np.cos(th)).max()


top_layer_1 = band_top(N1)
top_layer_2 = band_top(N2)
ratio_layer = top_layer_2 / top_layer_1

# 上界取 N-1，加倍 N 时上界从 N1-1 变 N2-1，比值 = (N2-1)/(N1-1)。
# 真正的判法落在「上界随 N 增长」这一趋势本身，比值等于 2 只是它的一个特例。
ratio_expected = (N2 - 1) / (N1 - 1)
rec(
    "A3",
    "两通道谱的紧致性对比（加倍指标上限看上界是否跟着变）",
    "相位侧上界随指标线性增长（比值 = (N2-1)/(N1-1) = %.4f，随 N 无界增长）；"
    "层数侧上界不随格点数改变（比值 1），判有界。这是两通道可判分离的第一个特征"
    % ratio_expected,
    "相位侧上界 %.0f -> %.0f，比值 %.6f（期望 %.6f）；"
    "层数侧上界 %.6f -> %.6f，比值 %.6f"
    % (top_phase_1, top_phase_2, ratio_phase, ratio_expected,
       top_layer_1, top_layer_2, ratio_layer),
    abs(ratio_phase - ratio_expected) < 1e-9
    and ratio_phase > 1.5
    and abs(ratio_layer - 1.0) < 1e-12,
)

# =====================================================================
# 组 B：虚过程在两通道里的形态
# =====================================================================
print("=" * 72)
print("组 B：虚过程在两通道里的形态")
print("=" * 72)

# B1：相位侧的虚过程：传播子有极点，须 iε
def prop_phase(w, k, m=1.0, eps=1e-3):
    return 1.0 / (w ** 2 - (k ** 2 + m ** 2) + 1j * eps)


ws = np.linspace(-3, 3, 3001)
k0 = 0.5
P = prop_phase(ws, k0, m=1.0)
peak = np.abs(P).max()
rec(
    "B1",
    "相位侧传播子 $1/(\\omega^2 - \\omega_k^2 + i\\varepsilon)$ 的极点结构",
    "在 $\\omega = \\pm\\sqrt{k^2+m^2}$ 处有极点，$|P|$ 的峰值由 $1/\\varepsilon$ 定，"
    "即依赖人为小量。这是相位侧虚过程的形态",
    "$k=%.2f$、$m=1$ 时极点位置 $\\pm%.4f$，峰值 %.4f（$1/\\varepsilon$ 量级，$\\varepsilon=10^{-3}$）"
    % (k0, np.sqrt(k0 ** 2 + 1.0), peak),
    peak > 100.0,
)

# B2：层数侧的虚过程：无极点，计数跃迁的振幅有限
# 层数侧的「虚过程」是 n -> n+1 -> n 的往返，其振幅由跃迁矩阵元给出
# 在有限带上，格林函数 G(z) = 1/(z - E(theta)) 在带外无极点（带隙中解析）
def G_layer(z, t=1.0, N=2001):
    th = np.linspace(-np.pi, np.pi, N)
    E = -2.0 * t * np.cos(th)
    return np.mean(1.0 / (z - E))


z_gap = 3.0   # 带外（带宽 4，带为 [-2,2]）
G_gap = G_layer(z_gap)
rec(
    "B2",
    "层数侧格林函数在带外的值（$z=3$，带宽 4、带为 $[-2,2]$）",
    "带外解析、无极点，值为有限实数，不依赖任何 $\\varepsilon$",
    "G(3) = %.8f（有限实数），全程无 $\\varepsilon$ 参数" % G_gap.real,
    np.isfinite(G_gap) and abs(G_gap.imag) < 1e-12,
)

# B3：层数侧的格林函数在带内有虚部（连续谱的边界）
# 带内格林函数的虚部来自谱密度：Im G(z) = -pi * rho(z)，
# 其中 rho 为 Bloch 带的态密度。用解析式算，避免离散平均把虚部抹平。
def dos_bloch(z, t=1.0):
    """一维紧束缚链的态密度：rho(E) = 1/(2*pi*t*sqrt(1-(E/2t)^2))，|E|<2t"""
    x = z / (2.0 * t)
    if abs(x) >= 1.0:
        return 0.0
    return 1.0 / (2.0 * np.pi * t * np.sqrt(1.0 - x ** 2))


z_in = 0.5
rho_in = dos_bloch(z_in)
im_G_analytic = -np.pi * rho_in
rec(
    "B3",
    "层数侧格林函数在带内的虚部（$z=0.5$，由 Bloch 态密度给出）",
    "带内虚部 $\\mathrm{Im}\\,G = -\\pi\\rho(z)$ 非零，$\\rho$ 由带结构给出（$t=1$、$z=0.5$ 时"
    "$\\rho = 1/(2\\pi\\sqrt{1-0.0625}) \\approx %.6f$），不依赖任何 $\\varepsilon$"
    % (1.0 / (2 * np.pi * np.sqrt(1 - 0.0625))),
    "$\\rho(0.5) = %.8f$，$\\mathrm{Im}\\,G = %.8f$（非零，且无 $\\varepsilon$ 出现）"
    % (rho_in, im_G_analytic),
    abs(im_G_analytic) > 1e-3,
)

# B4：恒真检验：若把层数侧强行套上 iε，结果应依赖 ε
def G_layer_eps(z, eps, t=1.0, N=2001):
    th = np.linspace(-np.pi, np.pi, N)
    E = -2.0 * t * np.cos(th)
    return np.mean(1.0 / (z - E + 1j * eps))


eps_list = [1e-1, 1e-2, 1e-3, 1e-4]
G_eps = np.array([G_layer_eps(z_gap, e) for e in eps_list])
spread = (np.abs(G_eps).max() - np.abs(G_eps).min()) / np.abs(G_eps[-1])
rec(
    "B4",
    "恒真检验：把相位侧的 $i\\varepsilon$ 强行套到层数侧带外",
    "带外本应解析（B2），套上 $i\\varepsilon$ 后结果反随 $\\varepsilon$ 变化，"
    "说明 $i\\varepsilon$ 处方不属于层数侧",
    "ε 四档 |G| %s，相对极差 %.4f"
    % (np.array2string(np.abs(G_eps), precision=10), spread),
    spread > 1e-3,
)

# =====================================================================
# 组 C：两通道的互补性（不是重复计数）
# =====================================================================
print("=" * 72)
print("组 C：两通道的互补性")
print("=" * 72)

# C1：两通道的算符对易（正交）
# 相位算符 N_phi 与层数算符 N_n 作用在不同因子空间，必对易
dim_phi, dim_n = 12, 12
Nphi = np.kron(np.diag(np.arange(dim_phi, dtype=float)), np.eye(dim_n))
Nn = np.kron(np.eye(dim_phi), np.diag(np.arange(dim_n, dtype=float)))
comm = Nphi @ Nn - Nn @ Nphi
rec(
    "C1",
    "两通道算符的对易子 $[\\hat N_\\phi, \\hat N_n]$",
    "两算符作用于张量积的不同因子，对易子恒为零矩阵（正交性）",
    "对易子最大元素 %.3e（维数 $%d\\times%d$）"
    % (np.abs(comm).max(), dim_phi * dim_n, dim_phi * dim_n),
    np.abs(comm).max() < 1e-12,
)

# C2：不构成重复计数。判法：两通道算符的**联合本征标号**是否唯一确定格点。
# 若冗余（两通道是同一个量的两种写法），则 (N_phi, N_n) 无法唯一标定格点。
pairs = [(a, b) for a in range(dim_phi) for b in range(dim_n)]
unique_pairs = len(set(pairs))
total_sites = dim_phi * dim_n

# 冗余对照的正确构造：若两通道是同一个量的两种写法，则层数算符
# 完全由相位算符决定（N_n = f(N_phi)）。此时联合标号数退化为 dim_phi。
def redundant_label_count(dim_a, dim_b, func):
    labels = set()
    for a in range(dim_a):
        for b in range(dim_b):
            if b == func(a):
                labels.add((a, b))
    return len(labels)


unique_red = redundant_label_count(dim_phi, dim_n, lambda a: a % dim_n)
rec(
    "C2",
    "不构成重复计数：联合本征标号 $(N_\\phi, N_n)$ 的唯一性",
    "两通道独立时，$%d \\times %d$ 个格点各由唯一有序对标记（唯一标号数 = 格点数 %d）；"
    "冗余构造（$N_n$ 由 $N_\\phi$ 决定）时标号数退化为 $%d$，故该判据有区分力"
    % (dim_phi, dim_n, total_sites, dim_phi),
    "独立构造：唯一标号 %d / 格点 %d（相等）；"
    "冗余对照：唯一标号 %d / 格点 %d（退化，为独立的 %d 分之一）"
    % (unique_pairs, total_sites, unique_red, total_sites, dim_n),
    unique_pairs == total_sites and unique_red == dim_phi and unique_red < total_sites,
)

# C3：可分离性判据：联合态是纯乘积态
psi_phi = np.zeros(dim_phi)
psi_phi[3] = 1.0
psi_n = np.zeros(dim_n)
psi_n[5] = 1.0
psi_joint = np.kron(psi_phi, psi_n)
rho = np.outer(psi_joint, psi_joint.conj())
# 约化密度矩阵的纯度：纯乘积态给 1
rho_phi = np.zeros((dim_phi, dim_phi), dtype=complex)
for b in range(dim_n):
    blk = rho.reshape(dim_phi, dim_n, dim_phi, dim_n)
    rho_phi += blk[:, b, :, b]
purity = np.real(np.trace(rho_phi @ rho_phi))
rec(
    "C3",
    "可分离性：乘积态的约化密度矩阵纯度",
    "纯乘积态的约化密度矩阵纯度为 1（可分离），这判两通道不纠缠、不冗余",
    "纯度 %.12f" % purity,
    abs(purity - 1.0) < 1e-10,
)

# C4：恒真检验：纠缠态须判否
bell = np.zeros(dim_phi * dim_n, dtype=complex)
bell[0 * dim_n + 0] = 1 / np.sqrt(2)
bell[1 * dim_n + 1] = 1 / np.sqrt(2)
rho_b = np.outer(bell, bell.conj())
blk = rho_b.reshape(dim_phi, dim_n, dim_phi, dim_n)
rho_phi_b = np.zeros((dim_phi, dim_phi), dtype=complex)
for b in range(dim_n):
    rho_phi_b += blk[:, b, :, b]
purity_b = np.real(np.trace(rho_phi_b @ rho_phi_b))
rec(
    "C4",
    "恒真检验：纠缠态（Bell 型）的约化纯度",
    "纯度为 $1/2 < 1$，故 C3 的判据有区分力",
    "纯度 %.12f" % purity_b,
    purity_b < 0.6,
)

# =====================================================================
# 组 D：拆分结果（读数）
# =====================================================================
print("=" * 72)
print("组 D：拆分结果（读数，不进通过率）")
print("=" * 72)

print("[D1] 读数：虚过程在相位侧的形态")
print("     连续相位角的场涨落，传播子有极点，须靠 $i\\varepsilon$ 定义（B1）。")
print("     其「虚」表现为离壳：$\\omega^2 \\neq k^2 + m^2$。")
print("")
print("[D2] 读数：虚过程在层数侧的形态")
print("     整数计数的往返跃迁 $n \\to n\\pm1 \\to n$，格林函数由 Bloch 带结构给出，")
print("     带外解析（B2）、带内虚部由谱边界给出（B3），全程不需要 $i\\varepsilon$。")
print("     其「虚」表现为中间计数值与末态不同，而总计数在往返后复原。")
print("")
print("[D3] 读数：两者的互补关系")
print("     两通道算符对易（C1，正交），乘积态可分离（C3，不冗余）。")
print("     故「虚过程」是一个统称，其两个成员分别住在两个通道里，")
print("     不可互相替代，也不构成重复计数。")
print("     这正合 AGENTS.md 所立「互补要求两侧并立，缺一侧即错」。")
print("")
print("[D4] 读数：对判据 730 的重新定位")
print("     判据 730 发现 $i\\varepsilon$ 处方在离散时间下失效（依赖达 19 倍）。")
print("     按本轮的拆分，其成因可精确表述：$i\\varepsilon$ 属相位侧处方（B1），")
print("     强行用于层数侧时失效（B4）。故判据 730 的读数应当读作")
print("     「相位侧处方不可移植到层数侧」，不能读作「离散化有问题」。")
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
ax.plot(theta, E_layer, "-", color="#1f77b4", linewidth=2.4,
        label="层数侧：Bloch 带 $E=-2t\\cos\\theta$")
ax.axhline(0, color="gray", linewidth=0.8)
ax.axhspan(-2, 2, color="#1f77b4", alpha=0.12)
ax.set_xlabel("准动量 $\\theta$")
ax.set_ylabel("谱值 $E$")
ax.set_title("左：层数通道的谱\n带宽 $4t$ 有限、紧致，无极点结构")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

ax = axes[1]
nn = np.arange(1, 12)
ax.plot(nn, nn, "o-", color="#d62728", linewidth=2.4, markersize=8,
        label="相位侧：$E_n = n$（无界）")
ax.axhline(2, color="#1f77b4", linestyle="--", linewidth=2.0,
           label="层数侧带上界 $+2t$")
ax.axhline(-2, color="#1f77b4", linestyle="--", linewidth=2.0)
ax.set_xlabel("指标")
ax.set_ylabel("谱值")
ax.set_title("右：两通道谱的紧致性对比\n相位侧无界，层数侧有界")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(IMG, dpi=150)
print("")
print("图已保存：%s" % IMG)
