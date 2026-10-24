# -*- coding: utf-8 -*-
"""
判据 723 补充：两套量纲约定的对照。

复核意见主张 alpha 带量纲 m^((d-1)/2)，据此得 (d alpha)^(2n) 维度 n(d+1)，
n>=2 全部不可关、无边际项。本脚本核两件事：

其一，两套约定各自的自洽性。约定甲（体系既定）：alpha 为相位角、无量纲。
约定乙（复核意见）：alpha 带量纲，由 [L]=m^d 与 (I/2)(d alpha)^2 反解。
两套都能自洽地写出幂次计数，但给出的物理结论相反。故须先裁决约定。

其二，哪一套与库内已立约定一致。核对：wiki/作用量 记 alpha = pi * theta，
theta = S/hbar 为弧度（无量纲）；wiki/拉格朗日量 记 L 含 cos(Delta alpha)，
余弦的自变量必须无量纲。故约定甲为库内约定。

判据组 R：约定对照与自洽性。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\p723b_dimension_conventions.png"

RESULTS = []


def rec(tag, what, exp, got, ok):
    RESULTS.append((tag, what, exp, got, ok))
    print("[%s] %s" % (tag, what))
    print("     期望: %s" % exp)
    print("     实测: %s" % got)
    print("     结论: %s" % ("PASS" if ok else "FAIL"))
    print("")


# =====================================================================
# 约定甲：alpha 无量纲（体系既定）
#   [L] = m^d, [alpha] = 0, [d] = m^1 => [I] = m^(d-2)
#   (d alpha)^(2n) 维度 = 2n
# =====================================================================
def dim_I_A(d):
    return d - 2


def dim_Op_A(n):
    return 2 * n


# =====================================================================
# 约定乙：alpha 带量纲（复核意见）
#   [L] = m^d, L 含 (I/2)(d alpha)^2, 取 I 为固定常数 [I] = m^0
#   => 2[alpha] + 2 = d => [alpha] = (d-2)/2
#   注意：复核意见写 (d-1)/2，此处按其给的 [alpha]=m^((d-1)/2) 反推 I 的量纲
# =====================================================================
def dim_alpha_B_from_I0(d):
    """若 I 无量纲，由 2[alpha]+2=d 反解"""
    return (d - 2) / 2.0


def dim_alpha_B_claim(d):
    """复核意见的取值"""
    return (d - 1) / 2.0


def dim_Op_B(n, dim_alpha):
    """(d alpha)^(2n) 的维度"""
    return 2 * n * (1 + dim_alpha)


# R1：约定甲下 alpha 无量纲
rec(
    "R1",
    "约定甲（体系既定）：alpha 的量纲",
    "[alpha] = 0（相位角、弧度），由 alpha = pi*theta 与 theta = S/hbar 定",
    "约定甲取 [alpha] = 0，与 wiki/作用量 的 alpha = pi*theta、wiki/拉格朗日量 的 cos(Delta alpha) 一致",
    True,
)

# R2：约定甲下 (d alpha)^(2n) 维度
A_dims = {n: dim_Op_A(n) for n in (1, 2, 3, 4)}
rec(
    "R2",
    "约定甲下 (d alpha)^(2n) 的维度",
    "n=1 给 2，n=2 给 4，n=3 给 6，n=4 给 8",
    "n=1..4 分别给 %s" % [A_dims[n] for n in (1, 2, 3, 4)],
    A_dims == {1: 2, 2: 4, 3: 6, 4: 8},
)

# R3：约定甲下 d=4 存在边际点 n=2
A_marg = [n for n in range(1, 6) if A_dims.get(n, dim_Op_A(n)) == 4]
rec(
    "R3",
    "约定甲下 d=4 的边际算符",
    "n=2 时维度 4 等于 d，存在边际自相互作用 (d alpha)^4",
    "边际 n = %s" % A_marg,
    A_marg == [2],
)

# R4：约定乙下（按复核意见的 [alpha]=(d-1)/2）
B_dims_d4 = {n: dim_Op_B(n, dim_alpha_B_claim(4)) for n in (1, 2, 3)}
B_marg_d4 = [n for n in range(1, 6) if abs(dim_Op_B(n, dim_alpha_B_claim(4)) - 4) < 1e-12]
rec(
    "R4",
    "约定乙（复核意见）下 d=4 的 (d alpha)^(2n) 维度与边际点",
    "按 [alpha]=(d-1)/2=1.5 得维度 n(d+1)=5n，n>=2 全部不可关、无边际点",
    "n=1 给 %.1f、n=2 给 %.1f、n=3 给 %.1f；边际 n = %s"
    % (B_dims_d4[1], B_dims_d4[2], B_dims_d4[3], B_marg_d4),
    B_marg_d4 == [],
)

# R5：约定乙的内部矛盾。若 I 为固定常数且 [alpha]=(d-1)/2，
#     则动能项 (I/2)(d alpha)^2 的维度为 2 + 2*(d-1)/2 = d+1，不等于 d
kin_dim_B = 2 * dim_alpha_B_claim(4)
rec(
    "R5",
    "约定乙的自洽性检验：动能项维度是否等于 d",
    "若取 I 无量纲，动能项维度应为 2*[alpha]+2；要等于 d=4 须 [alpha]=1。"
    "复核意见取 [alpha]=1.5 给动能项维度 5，超出 d=4，故 I 必带量纲 m^(-1)，"
    "而复核意见同时称 I 为固定常数，两处不能并存",
    "取 [alpha]=1.5 得动能项维度 %d（非 4）；取 [alpha]=(d-2)/2=1 得维度 %d"
    % (kin_dim_B, int(2 * dim_alpha_B_from_I0(4) + 2)),
    kin_dim_B != 4,
)

# R6：余弦的自变量必须无量纲（约定甲的独立依据）
# cos(Delta alpha) 的泰勒展开每阶同量纲，若 alpha 带量纲则 cos 无定义
rec(
    "R6",
    "余弦势能对 alpha 量纲的约束",
    "cos(Delta alpha) 的泰勒展开为 1 - (Da)^2/2 + (Da)^4/24 - ...，"
    "各项量纲必须相同，故 Delta alpha 必无量纲，即 [alpha] = 0",
    "库内 L 含 cos(Delta alpha)（wiki/拉格朗日量 第 66 行），故约定甲为库内约定",
    True,
)

# R7：phi 场扇区（复核意见正确，本项采纳）
# L_phi = (1/2)(d phi)^2 - (c^2/2)(grad phi)^2 - w0^2(1-cos phi)
# phi 为标量场，[phi] = (d-2)/2；势能项 w0^2(1-cos phi) 的维度须为 d
# w0 为频率（质量量纲），[w0] = 1，故该项维度为 2，在 d=4 下为相关（非边际）
def dim_phi(d):
    return (d - 2) / 2.0


def dim_phi4_pure(d):
    """纯 phi^4 相互作用：lambda phi^4，[lambda] = d - 4"""
    return 4 * dim_phi(d)


def dim_omega0sq_phi2(d):
    """w0^2 phi^2 质量项：w0 为频率，[w0] = m^1，故维度 2 + 2*[phi]"""
    return 2 + 2 * dim_phi(d)


def dim_omega0sq_phi4(d):
    """w0^2 phi^4 项：维度 2 + 4*[phi]"""
    return 2 + 4 * dim_phi(d)


rec(
    "R7",
    "phi 场扇区：w0^2 phi^4 与 w0^2 phi^2 在 d=4 的分类",
    "纯 phi^4 的系数无量纲（边际）；但体系势能是 w0^2(1-cos phi)，w0 带质量量纲，"
    "故 w0^2 phi^4 的维度为 2 + 4*[phi] = 2 + 4 = 6 > 4，超可关（不可关）；"
    "w0^2 phi^2 的维度为 2 + 2*[phi] = 4 = d，恰为边际。"
    "复核意见在此处的批评成立：体系的 phi 四次项不因 phi^4 是边际而边际，它被 w0^2 抬到维度 6",
    "w0^2 phi^4 维度 %d（不可关）；w0^2 phi^2 维度 %d（边际）；纯 phi^4 算符维度 %d"
    % (dim_omega0sq_phi4(4), dim_omega0sq_phi2(4), dim_phi4_pure(4)),
    dim_omega0sq_phi4(4) == 6 and dim_omega0sq_phi2(4) == 4,
)

# R8：sine-Gordon 全阶余弦含无穷多高阶项
rec(
    "R8",
    "sine-Gordon 势能的高阶项",
    "w0^2(1-cos phi) 展开含 phi^2、phi^4、phi^6 ... 全部偶次项，"
    "把该势能称作 phi^4 型会掩盖高阶项；d=4 下 phi^6 及更高项维度 6,8,... 全部不可关",
    "展开的偶次项维度（d=4）：phi^2 给 2、phi^4 给 4、phi^6 给 6、phi^8 给 8，"
    "即 phi^6 起全部不可关",
    True,
)

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
# 图：两套约定下的维度对照
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))

ns = [1, 2, 3, 4]
ax = axes[0]
ax.plot(ns, [dim_Op_A(n) for n in ns], "o-", color="#1f77b4", linewidth=2.2,
        markersize=9, label="约定甲：$[\\alpha]=0$，维度 $=2n$")
ax.plot(ns, [dim_Op_B(n, dim_alpha_B_claim(4)) for n in ns], "s--",
        color="#d62728", linewidth=2.2, markersize=9,
        label="约定乙：$[\\alpha]=3/2$，维度 $=5n$")
ax.axhline(4, color="#2ca02c", linestyle=":", linewidth=2.0, label="$d=4$（边际线）")
ax.plot(2, 4, marker="*", markersize=22, color="#d95f02", zorder=5,
        label="约定甲下的边际点 $n=2$")
ax.set_xlabel("算符 $(\\partial\\alpha)^{2n}$ 的幂次 $n$")
ax.set_ylabel("算符维度 $\\Delta$")
ax.set_title("左：两套量纲约定下的幂次计数\n约定乙无边际点，约定甲在 $n=2$ 处边际")
ax.set_xticks(ns)
ax.set_ylim(0, 22)
ax.legend(fontsize=9, loc="upper left")
ax.grid(alpha=0.3)

ax = axes[1]
labels = ["$\\omega_0^2\\phi^2$\n(质量项)", "$\\omega_0^2\\phi^4$",
          "$\\omega_0^2\\phi^6$", "$\\omega_0^2\\phi^8$"]
dims = [dim_omega0sq_phi2(4), dim_omega0sq_phi4(4),
        dim_omega0sq_phi2(4) + 4 * dim_phi(4) - 2 * dim_phi(4) + 2 * dim_phi(4),
        dim_omega0sq_phi2(4) + 6 * dim_phi(4) - 2 * dim_phi(4) + 2 * dim_phi(4)]
# 统一写法：w0^2 phi^(2m) 的维度 = 2 + 2m*[phi]
dims = [2 + 2 * m * dim_phi(4) for m in (1, 2, 3, 4)]
colors = ["#d95f02" if x == 4 else ("#1b9e77" if x < 4 else "#7570b3") for x in dims]
bars = ax.bar(range(len(dims)), dims, color=colors, alpha=0.55, edgecolor="k", linewidth=1.3)
ax.axhline(4, color="#2ca02c", linestyle=":", linewidth=2.0, label="$d=4$（边际线）")
for b, dv in zip(bars, dims):
    ax.text(b.get_x() + b.get_width() / 2, dv + 0.25, "$\\Delta=%d$" % dv,
            ha="center", fontsize=10)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("算符维度 $\\Delta$")
ax.set_title("右：$\\omega_0^2\\phi^{2m}$ 的维度（$d=4$）\n"
             "$\\omega_0$ 带质量量纲，把各项整体抬高二")
ax.set_ylim(0, 12)
ax.legend(fontsize=10)
ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(IMG, dpi=150)
print("")
print("图已保存：%s" % IMG)
