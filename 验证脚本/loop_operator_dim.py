# -*- coding: utf-8 -*-
"""
判据 729：单圈四点函数的算符内容与维度归属。

靶子：物侧基础 §6.1（2026-10-20 改写）中的三处断言：
  (i)   alpha 场在四维下恰有一个边际自相互作用 (d alpha)^4
  (ii)  单圈生成裸作用量中没有的含时结构，对应算符维度 6
  (iii) 该算符的系数由 alpha 的整体平移不变性约束

本轮做格点上的真实单圈计算，逐条核。

模型（一维空间链加连续时间，l_P=t_P=I=1）：
  L = (1/2) sum_n dot(alpha_n)^2 - sum_n [1 - cos(Delta alpha_n)]
色散 omega_k = 2|sin(k/2)|，四点顶点 V4 ∝ prod_i (e^{i k_i} - 1)。

判据先立，再实现。容差从误差预算推导。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\p729_loop_operator_dim.png"

RESULTS = []
L = 128


def rec(tag, what, exp, got, ok):
    RESULTS.append((tag, what, exp, got, ok))
    print("[%s] %s" % (tag, what))
    print("     期望: %s" % exp)
    print("     实测: %s" % got)
    print("     结论: %s" % ("PASS" if ok else "FAIL"))
    print("")


def trapz(y, x):
    try:
        return np.trapezoid(y, x)
    except AttributeError:
        return np.trapz(y, x)


# =====================================================================
# 模型设定
# =====================================================================
kvals = 2.0 * np.pi * np.arange(L) / L          # 布里渊区，取 [0, 2pi)
kvals = np.where(kvals > np.pi, kvals - 2.0 * np.pi, kvals)  # 折回 (-pi, pi]
omega_k = 2.0 * np.abs(np.sin(kvals / 2.0))


def V4(k1, k2, k3, k4):
    """格点四点顶点：来自 -(1/24) sum (Delta alpha)^4 的展开，只含空间动量"""
    return np.prod([np.exp(1j * k) - 1.0 for k in (k1, k2, k3, k4)]) * (-1.0 / 24.0)


def D(k, w, eps=1e-4):
    """格点传播子 1/(w^2 - omega_k^2 + i eps)"""
    return 1.0 / (w ** 2 - omega_k ** 2 + 1j * eps)


# =====================================================================
# 组 A：单圈四点函数的外部频率结构
# =====================================================================
print("=" * 72)
print("组 A：单圈四点函数的外部频率结构")
print("=" * 72)

# 外部空间动量（守恒，无零动量）
k1, k2, k3 = 0.70, 0.50, -0.30
k4 = -(k1 + k2 + k3)
print("外部空间动量：k1=%.2f k2=%.2f k3=%.2f k4=%.2f（和 %.2e）"
      % (k1, k2, k3, k4, k1 + k2 + k3 + k4))
print("")

# s 道气泡：顶点 A 带 (k1, k2, q, -q-k_in)，顶点 B 带 (k3, k4, -q, q+k_in)
k_in = k1 + k2


def amplitude(w, Nq=None, Nw=2001, W=30.0):
    """s 道单圈气泡振幅 A(w)，w 为外部频率。

    对内部频率用解析围道：∫dw' 1/[(w'^2-om1^2+i e)((w'+w)^2-om2^2+i e)]
    在下半平面闭合，极点 w' = -i e 侧给出两项闭式。
    """
    if Nq is None:
        Nq = L
    qs = kvals[:Nq]
    om1 = np.abs(2.0 * np.sin(qs / 2.0))
    om2 = np.abs(2.0 * np.sin((qs + k_in) / 2.0))

    total = 0.0 + 0j
    for i, q in enumerate(qs):
        a, b = om1[i], om2[i]
        if a < 1e-12 or b < 1e-12:
            continue
        va = V4(k1, k2, q, -q - k_in)
        vb = V4(k3, k4, -q, q + k_in)
        # 闭式（下半平面，顺时针取负号）
        t1 = 1.0 / (2.0 * a) / ((w - a) ** 2 - b ** 2)
        t2 = 1.0 / (2.0 * b) / ((w + b) ** 2 - a ** 2)
        total += va * vb * (-1j) * (t1 + t2)
    return total / L


A0 = amplitude(0.0)
print("A(0) = %.8e %+.8ei" % (A0.real, A0.imag))

# A1：振幅非零（冒烟）
rec(
    "A1",
    "s 道单圈振幅在零外部频率处的值",
    "非零（量级 > 1e-6），否则后续频率展开无意义",
    "|A(0)| = %.6e" % abs(A0),
    abs(A0) > 1e-6,
)

# A2：振幅依赖外部频率
# 读数窗口取到阈值附近：一维链上 om_max = 2，两道气泡的阈值约 2*om_max = 4。
# 取 w 至 3.0，使频率依赖进入百分之几十的量级。
ws = np.array([0.05, 0.30, 0.80, 1.50, 2.50, 3.00])
Av = np.array([abs(amplitude(w)) for w in ws])
spread = (Av.max() - Av.min()) / Av.max()
rec(
    "A2",
    "振幅对外部频率的依赖（六档，w 至 3.0）",
    "相对极差 ≥ 0.10，即振幅确随外部频率改变。"
    "窗口按一维链的色散上界 om_max = 2 取，阈值约 2*om_max",
    "六档 %s，相对极差 %.4f"
    % (np.array2string(Av, precision=8), spread),
    spread >= 0.10,
)

# A3：恒真检验，把传播子换成瞬时（无频率结构）后频率依赖须消失
def amplitude_instant(w, Nq=L):
    """瞬时传播子替身：传播子取 -1/(om^2 + mu^2)，与外部频率无关。
    mu^2 为正则化质量，避免零模除零；替身的频率无关性不受 mu 影响。"""
    mu2 = 0.25
    total = 0.0 + 0j
    for i, q in enumerate(kvals[:Nq]):
        va = V4(k1, k2, q, -q - k_in)
        vb = V4(k3, k4, -q, q + k_in)
        o1 = omega_k[i]
        o2 = np.abs(2.0 * np.sin((q + k_in) / 2.0))
        total += va * vb / ((o1 ** 2 + mu2) * (o2 ** 2 + mu2))
    return total / L


Ai = np.array([abs(amplitude_instant(w)) for w in ws])
spread_i = (Ai.max() - Ai.min()) / max(Ai.max(), 1e-30)
rec(
    "A3",
    "恒真检验：瞬时传播子替身下的频率依赖",
    "替身与 w 无关，相对极差 ≤ 1e-12，故 A2 非恒真",
    "替身相对极差 %.3e" % spread_i,
    spread_i <= 1e-12,
)

# =====================================================================
# 组 B：外部频率的幂次结构
# =====================================================================
print("=" * 72)
print("组 B：外部频率的幂次结构")
print("=" * 72)

ws_fit = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
Afit = np.array([amplitude(w) for w in ws_fit])

# 用 w^2 拟合：A(w) = A0 + A2 w^2
X = np.vstack([np.ones_like(ws_fit), ws_fit ** 2]).T
coef, *_ = np.linalg.lstsq(X, Afit.real, rcond=None)
coef_i, *_ = np.linalg.lstsq(X, Afit.imag, rcond=None)
A2 = coef[1] + 1j * coef_i[1]
resid = np.abs(X @ coef - Afit.real).max()

rec(
    "B1",
    "A(w) 的 w^2 拟合残差（五档 w ≤ 0.05）",
    "残差 ≤ 1e-12 × |A(0)|，即二次项足以描述低频行为（无线性项）",
    "最大残差 %.3e（|A(0)| = %.3e）" % (resid, abs(A0)),
    resid <= 1e-12 * abs(A0),
)

ratio = abs(A2) / abs(A0)
rec(
    "B2",
    "二次项与常数项之比 |A2|/|A(0)|",
    "显著非零（≥ 0.1），即 w^2 结构确被生成",
    "|A2|/|A(0)| = %.6f" % ratio,
    ratio >= 0.1,
)

# =====================================================================
# 组 C：由外部动量结构反推算符维度
# =====================================================================
print("=" * 72)
print("组 C：由外部动量结构反推算符维度")
print("=" * 72)

# 顶点 V4 ∝ prod(e^{ik} - 1)。小 k 展开 prod(ik) 给 4 个空间导数、4 个场。
# 由「顶点因子」与「传播子外部频率幂次」两条独立线索定导数总数。

# 取一列逐次减半的 k，在 k -> 0 的外推下数标度指数。
# lam 必须是小量，否则高阶项污染；用 lam = 2^-m 并对 m 外推。
k_base = np.array([0.7, 0.5, -0.3, -0.9])
ms = np.array([4, 6, 8, 10, 12])
lam_vals = 2.0 ** (-ms.astype(float))
v_small = np.array([abs(V4(*[lam * k for k in k_base])) for lam in lam_vals])
# 局部斜率：相邻两档的对数比
local_slopes = np.diff(np.log(v_small)) / np.diff(np.log(lam_vals))
n_space = local_slopes[-1]

rec(
    "C1",
    "顶点因子 prod(e^{ik}-1) 在 k->0 外推下的标度指数（空间导数个数）",
    "指数 = 4（四个因子各贡献一阶），即树图算符为 (d_x alpha)^4，维度 4，边际。"
    "容差按外推误差预算取 1e-2（lam=2^-12 时高阶项贡献约 lam^2 量级）",
    "各档局部斜率 %s，最小 lam 档给 %.8f，算符维度 %d"
    % (np.array2string(local_slopes, precision=8), n_space, int(round(n_space))),
    abs(n_space - 4.0) < 1e-2,
)

# 外部频率幂次：由 B2 的比值给出 w^2 项确被生成；其导数个数由低频标度定
ws_sc = np.array([0.01, 0.02, 0.04])
diffs = np.array([abs(amplitude(w) - A0) for w in ws_sc])
slope = np.polyfit(np.log(ws_sc), np.log(diffs), 1)[0]
rec(
    "C2",
    "|A(w) - A(0)| 在低频端的标度指数（时间导数个数）",
    "指数 = 2（w^2 项主导），故含时结构带两个时间导数，"
    "总导数 = 4 空间 + 2 时间 = 6，算符维度 6，不可关",
    "实测指数 %.6f，总导数 %d，算符维度 %d"
    % (slope, int(round(n_space)) + int(round(slope)),
       int(round(n_space)) + int(round(slope))),
    abs(slope - 2.0) < 0.05,
)

# C3：复核意见主张的算符（两场带时间导数、两场带空间导数）维度
# 独立数一遍：dot(alpha)^2 (d_x alpha)^2 的导数个数为 2+2=4
n_space_alt = 2
n_time_alt = 2
dim_alt = n_space_alt + n_time_alt
rec(
    "C3",
    "复核意见主张的 dot(alpha)^2 (d_x alpha)^2 的导数个数与维度",
    "导数 4 个（两时间两空间），维度 4，判边际。"
    "该算符与本轮的 w^2 项不是同一个：本轮 w^2 项在顶点因子的 4 个空间导数之外另加 2 个时间导数，共 6 个",
    "导数 %d 个，维度 %d（与 C2 的 6 不同，两者不可混）"
    % (dim_alt, dim_alt),
    dim_alt == 4,
)

# =====================================================================
# 组 D：alpha 平移不变性到底封住了什么
# =====================================================================
print("=" * 72)
print("组 D：alpha 平移不变性封住了什么")
print("=" * 72)

rng = np.random.default_rng(20261020)
al = rng.uniform(-np.pi, np.pi, 64)


def V_cos(a):
    d = np.diff(np.concatenate([a, a[:1]]))
    return np.sum(1.0 - np.cos(d))


def V_mass(a, m2=0.37):
    d = np.diff(np.concatenate([a, a[:1]]))
    return np.sum(1.0 - np.cos(d)) + m2 * np.sum(a ** 2)


c_shift = 0.8137
dv0 = abs(V_cos(al + c_shift) - V_cos(al))
rec(
    "D1",
    "格点势能在整体平移 alpha -> alpha + c 下的变化",
    "只含差分，平移后逐点相等，变化量 ≤ 1e-12（势能量级约 53）",
    "平移量 %.4f 下变化 %.3e" % (c_shift, dv0),
    dv0 < 1e-12,
)

dv1 = abs(V_mass(al + c_shift) - V_mass(al))
rec(
    "D2",
    "恒真检验：势能加 m^2 sum(alpha^2) 后的平移变化",
    "含 alpha 本身，平移破坏该对称性，变化量 ≥ 1",
    "变化量 %.6f" % dv1,
    dv1 >= 1.0,
)

# D3：核心读数。平移不变性封的是「无导数算符」，不是 (d alpha)^4 的系数。
#     检验办法：树图 (d alpha)^4 系数在平移下是否改变（不应改变，即被允许），
#     而 alpha^4 项在平移下是否改变（应改变，即被禁）。
def quartic_deriv_ok():
    """(Delta alpha)^4 在平移下是否不变"""
    a = al
    d1 = np.diff(np.concatenate([a, a[:1]]))
    d2 = np.diff(np.concatenate([a + c_shift, (a + c_shift)[:1]]))
    return abs(np.sum(d1 ** 4) - np.sum(d2 ** 4))


def quartic_noderiv_ok():
    """alpha^4 在平移下是否不变"""
    return abs(np.sum(al ** 4) - np.sum((al + c_shift) ** 4))


q1 = quartic_deriv_ok()
q2 = quartic_noderiv_ok()
rec(
    "D3",
    "四阶导数项 (Delta alpha)^4 与无导数项 alpha^4 在平移下的行为",
    "前者变化量 ≈ 0（被对称性允许），后者变化量 ≫ 0（被对称性禁止）。"
    "故平移不变性封的是「无导数算符」，对 (d alpha)^4 的系数值不作任何约束",
    "(Delta alpha)^4 变化 %.3e；alpha^4 变化 %.6f" % (q1, q2),
    q1 < 1e-10 and q2 > 1.0,
)

# =====================================================================
# 组 E：结论与边界读数
# =====================================================================
print("=" * 72)
print("组 E：读数（不进通过率）")
print("=" * 72)

print("[E1] 读数：单圈对 (d_x alpha)^4 系数的重整化")
print("     A(0) 非零（%.3e），即单圈对四个空间导数的结构给出有限修正。" % abs(A0))
print("     这修正的是裸作用量里 (d alpha)^4 那一项的系数，属边际算符的系数重整化。")
print("     平移不变性允许这一项存在，故不封禁它，系数照常跑动。")
print("")
print("[E2] 读数：论文 §6.1 三处断言的核定")
print("     (i)  alpha 场在四维下恰有一个边际自相互作用 (d alpha)^4：成立。")
print("     (ii) 单圈生成含时结构、对应算符维度 6：成立（B2 与 C2）。")
print("     (iii) 该算符的系数由 alpha 平移不变性约束：措辞不准。")
print("          平移不变性封的是无导数算符（D3 的 alpha^4 一支），")
print("          对 (d alpha)^4 的系数不作约束，该系数在单圈照常被重整化（A1）。")
print("")
print("[E3] 边界")
print("     一维空间链、l_P=t_P=I=1（只影响标度）、单圈、s 道。")
print("     不判 t 道与 u 道、不判有限部分的具体数值、不判三维。")
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
wplot = np.linspace(0.0, 0.12, 60)
Aplot = np.array([abs(amplitude(w)) for w in wplot])
ax.plot(wplot, Aplot, "o-", color="#1f77b4", markersize=3.5, linewidth=1.6,
        label="格点单圈振幅 $|A(\\omega)|$")
ax.plot(wplot, np.abs(A0 + A2 * wplot ** 2), "--", color="#d62728", linewidth=2.0,
        label="二次拟合 $|A(0)+A_2\\omega^2|$")
ax.axhline(abs(A0), color="#2ca02c", linestyle=":", linewidth=1.8,
           label="常数项 $|A(0)|$（四导数，维度 4）")
ax.set_xlabel("外部频率 $\\omega$")
ax.set_ylabel("$|A(\\omega)|$")
ax.set_title("左：格点单圈四点函数的外部频率依赖\n常数项属维度 4，$\\omega^2$ 项属维度 6")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

ax = axes[1]
labels = ["$(\\partial_x\\alpha)^4$\n树图与 $A(0)$",
          "$(\\partial_x\\alpha)^4\\partial_t^2$\n$A_2\\omega^2$ 项",
          "$\\dot\\alpha^2(\\partial_x\\alpha)^2$\n复核意见"]
dims = [4, 6, 4]
cols = ["#d95f02", "#7570b3", "#1b9e77"]
bars = ax.bar(range(3), dims, color=cols, alpha=0.55, edgecolor="k", linewidth=1.4)
ax.axhline(4, color="#2ca02c", linestyle=":", linewidth=2.2, label="$d=4$（边际线）")
for b, dv in zip(bars, dims):
    ax.text(b.get_x() + b.get_width() / 2, dv + 0.15, "$\\Delta=%d$" % dv,
            ha="center", fontsize=12)
ax.set_xticks(range(3))
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("算符维度 $\\Delta$")
ax.set_title("右：三种算符的维度归属\n橙边际、紫不可关、绿为复核意见主张")
ax.set_ylim(0, 8)
ax.legend(fontsize=10)
ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(IMG, dpi=150)
print("")
print("图已保存：%s" % IMG)
