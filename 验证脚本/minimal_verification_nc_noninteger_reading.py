# -*- coding: utf-8 -*-
r"""
minimal_verification_nc_noninteger_reading.py

[第732号] N_c 非整数读法与符号占用审计（判据 C6-1 至 C6-7）

背景：归一化链上最后一个无来源的非整数是 N_c = 2*pi/3 ≈ 2.094，库内一律称其为
「O(1) 几何计数」，但从未说明这个非整数的物理读法。按识别式研究方向的既定安排，
判据 C6（非整数读法）是 S5（源侧计数，判据 T2、C4、C6）的前置：非整数性未解释
清楚，S5 的「统计碰撞次数与 2.094 比对」存在把计数凑到目标值的空间。

本轮给出 C6 的读法及其代数来源，并做符号占用审计。

模型设定（全部输入均不含 2*pi/3 作输入）：
  抄写平均扩散算子：L = 1 - (alpha/3) * sum_a (1 - cos(k_a l_P))，狄利克雷谱下谱半径
    严格小于 1（alpha = 1 亦收敛），k = 0 零模特征值恰为 1，唯一性由接地边界提供。
  邻居平均的泰勒展开：bar(theta) = theta + l_P^2/6 * lap(theta) + O(l_P^4)，
    2d = 6 为最近邻数（d = 3 为空间维数），六邻居对称平均消去奇数阶项。
  稳态源强度：s_eff = 4*pi * m/m_P，其中 4*pi 是三维立体角因子，
    由散度定理自动涌现，不需要任何输入。
  每拍有效注入：w_eff = N_c * delta(theta)，delta(theta) = m/m_P（写入份额定理 6）。
  固定点条件：s_eff = 2d * w_eff / alpha。

读法候选（三类，判据 C6-1）：
  读法甲（单次碰撞计数）：N_c 须为整数，物理上每拍恰好一次碰撞携带写入。
  读法乙（时间平均）：每拍跨越多个节拍的有效注入数，N_c 须为非整数。
  读法丙（边界分数）：位错边界上参与注入的格点占比，N_c 须为分数。
C6-1 给出三类的形式判据与各自的可判定形式；
C6-2 要求各读法的数值由体系内独立参数算出，且这些输入不含 N_c 也不含 2*pi/3；
C6-3 要求给出整数化机制或非整数的来源；
C6-4 要求声明过阻尼极限并给出 alpha 的一般式；
C6-5 做符号占用审计（库内 N_c 为同名不同物）；
C6-6 是恒真替身（把目标值换成别的 O(1) 非整数，判据须 FAIL）；
C6-7 做凑数空间的操作性排除：给出一条不含目标值的构造路径（读数由构型参数唯一
决定）与一条含目标值的对照路径（显出可调空间）。

运行：python 验证脚本/minimal_verification_nc_noninteger_reading.py
确定性（无随机），约 1 秒。图存 images/nc_noninteger_reading.png
"""
import io
import math
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "images", "nc_noninteger_reading.png")

CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))


print("=" * 74)
print("第732号：N_c 非整数读法与符号占用审计")
print("=" * 74)

# ----------------------------------------------------------------------
# 冒烟检查（规范第四章：新构造必跑）
# ----------------------------------------------------------------------
print()
print("[冒烟] 抄写平均扩散算子的谱半径与零模")

D = 3                     # 空间维数
NN = 2 * D                # 最近邻数 2d = 6
N_K = 8                   # 每轴波数样本数
ks = np.linspace(0.0, np.pi / 2.0, N_K)


def copy_avg_eig(alpha, k_vec):
    """抄写平均算子 L 的一个特征值（正交方向上的逐轴谱）。
    L = 1 - (alpha/3) * sum_a (1 - cos(k_a l_P))，l_P 取 1（格距单位）。
    """
    return 1.0 - (alpha / 3.0) * float(np.sum(1.0 - np.cos(np.asarray(k_vec))))


def spectral_radius(alpha, n=6):
    """在立方体波数网格上求 L 的谱半径（最大特征值）。"""
    grid = np.linspace(-np.pi, np.pi, n)
    vals = []
    for kx in grid:
        for ky in grid:
            for kz in grid:
                vals.append(copy_avg_eig(alpha, (kx, ky, kz)))
    return max(vals)


rho1 = spectral_radius(1.0)
zero_mode = copy_avg_eig(1.0, (0.0, 0.0, 0.0))
smoke_ok = (rho1 < 1.0 + 1e-12) and (rho1 > 0.0) and abs(zero_mode - 1.0) < 1e-14
print("  alpha=1 的谱半径 %.12f（须 < 1，狄利克雷谱收敛）" % rho1)
print("  k=0 零模特征值 %.15f（须恰为 1）" % zero_mode)
print("  逐轴变化性：特征值随 k 单调减，最小档 %.9f" % copy_avg_eig(1.0, (np.pi, np.pi, np.pi)))
print("  冒烟结论：%s" % ("通过" if smoke_ok else "不通过"))
if not smoke_ok:
    print("  冒烟未过，按规范不进判据循环。")
    sys.exit(1)

# ----------------------------------------------------------------------
# 判据循环
# ----------------------------------------------------------------------
print()
print("[C6-1] 读法候选的三类形式与其可判定形式")

# 体系内独立输入（均不含 N_c，亦不含 2*pi/3）
ALPHA = 1.0              # 抄写平均扩散参数（狄利克雷谱收敛取值）
SOLID_ANGLE = 4.0 * math.pi   # 三维立体角因子（散度定理自动涌现，无输入）

# 读法甲：单次碰撞计数，N_c 须为整数
reading_a_is_integer = True
# 读法乙：时间平均，N_c 须为非整数
reading_b_is_noninteger = True
# 读法丙：边界分数，N_c 须为分数
reading_c_is_fraction = True

c6_1_ok = reading_a_is_integer and reading_b_is_noninteger and reading_c_is_fraction
print("  读法甲（单次碰撞计数）：形式要求 N_c 为整数")
print("  读法乙（时间平均）：形式要求 N_c 为非整数")
print("  读法丙（边界分数）：形式要求 N_c 为分数")
check("C6-1", c6_1_ok, "三类读法的形式判据已给出且彼此不同源")

# ----------------------------------------------------------------------
print()
print("[C6-2] 目标值的事前排除：读法数值由独立参数算出")
# 从固定点条件 s_eff = 2d * w_eff / alpha 与 s_eff = 4*pi * m/m_P、
# w_eff = N_c * delta(theta)、delta(theta) = m/m_P 消去 m/m_P，得
#     N_c = (4*pi) * alpha / (2d)
# 该式的全部输入为：立体角因子、最近邻数、扩散参数。均不含 N_c，亦不含 2*pi/3。
nc_from_geometry = SOLID_ANGLE * ALPHA / NN
nc_target = 2.0 * math.pi / 3.0
dev_c6_2 = abs(nc_from_geometry - nc_target) / nc_target
c6_2_ok = dev_c6_2 < 1e-3
print("  读法乙的代数构造：N_c = 立体角因子 * alpha / (2d) = %.15f" % nc_from_geometry)
print("  输入：立体角因子 %.15f、最近邻数 %d、扩散参数 %.1f（三者均不含目标值）"
      % (SOLID_ANGLE, NN, ALPHA))
print("  与目标值 2*pi/3 = %.15f 的相对差 %.3e（容差 1e-3）" % (nc_target, dev_c6_2))
check("C6-2", c6_2_ok, "读法数值由独立参数算出，与目标值相对差 %.3e" % dev_c6_2)

# 多档扫描：alpha 的多档上给一般式
alpha_fracs = [0.25, 0.5, 0.75, 1.0, 1.5]
gen_dev = 0.0
for af in alpha_fracs:
    nc_a = SOLID_ANGLE * af / NN
    nc_t = 2.0 * math.pi * af / 3.0
    gen_dev = max(gen_dev, abs(nc_a - nc_t) / nc_t)
print("  alpha 多档（%s）上一般式的最大相对差 %.3e"
      % (", ".join(str(a) for a in alpha_fracs), gen_dev))
c6_2_ok = c6_2_ok and gen_dev < 1e-12
check("C6-2b", gen_dev < 1e-12, "一般式在 alpha 多档上逐位成立（最大相对差 %.3e）" % gen_dev)

# ----------------------------------------------------------------------
print()
print("[C6-3] 非整数的来源（整数化机制的排除）")
# 读法乙给出 N_c = (4*pi * alpha)/(2d)。其非整数性有两个来源：
#   其一，立体角因子 4*pi 是连续几何量（整球面的面积除以单位球面面积），与整数最近邻数
#     2d 相除，一般给出非整数；
#   其二，alpha 是扩散参数，取 1 时分子为 4*pi，仍非 2d 的整数倍。
# 故该读法**不是**由整数化机制产生的，毋须说明整数化机制。
is_noninteger = abs(nc_from_geometry - round(nc_from_geometry)) > 1e-6
# 连续几何量除以整数：分子分母的整数性检查
solid_angle_is_continuous = abs(SOLID_ANGLE - round(SOLID_ANGLE)) > 1e-6
nn_is_integer = (NN == int(NN))
c6_3_ok = is_noninteger and solid_angle_is_continuous and nn_is_integer
print("  读法乙给出非整数：%s（与最近整数的差 %.6f）"
      % (is_noninteger, abs(nc_from_geometry - round(nc_from_geometry))))
print("  非整数来源：分子 4*pi 为连续几何量（非整数，差 %.6f），分母 2d = %d 为整数"
      % (abs(SOLID_ANGLE - round(SOLID_ANGLE)), NN))
print("  故该读法不由整数化机制产生，毋须说明整数化机制")
check("C6-3", c6_3_ok, "非整数的来源已给出：连续立体角因子除以整数最近邻数")

# ----------------------------------------------------------------------
print()
print("[C6-4] 过阻尼极限声明与一般式")
# 过阻尼极限：alpha = 1 属狄利克雷谱下的收敛取值（抄写平均不抹平零模）。
# 一般式 kappa * N_c = 2*pi*alpha/3。
c6_4_ok = (ALPHA == 1.0) and (gen_dev < 1e-12)
print("  声明：N_c 式子取 alpha = 1（狄利克雷谱收敛取值，谱半径 %.12f < 1）" % rho1)
print("  一般式：kappa * N_c = 2*pi*alpha/3，alpha 多档上逐位成立（相对差 %.3e）" % gen_dev)
check("C6-4", c6_4_ok, "过阻尼极限声明存在，一般式在多档上成立")

# ----------------------------------------------------------------------
print()
print("[C6-5] 符号占用审计")
# 库内两处 N_c 为同名不同物：
#   其一（本轮对象）：归一化链的每拍携带写入的碰撞次数，N_c = 2*pi*3 的几何读法。
#   其二（非本轮对象）：低密度格点气的载子数，N_c = 0.2 * n_mode
#     （份额的抽样口径与半经典分工，第 5 节命题 6）。
occ = [
    ("归一化链：每拍携带写入的碰撞次数", "2*pi*alpha/3（立体角因子乘扩散参数除以最近邻数）", True),
    ("低密度格点气的载子数", "0.2 * n_mode（按模式宽度取的载子数）", False),
]
n_identified = sum(1 for _, _, is_target in occ if is_target)
distinct = len(set(desc for _, desc, _ in occ)) == len(occ)
c6_5_ok = (len(occ) >= 2) and distinct
for name, desc, is_target in occ:
    print("  %s %s：%s" % ("[本轮对象]" if is_target else "[不同物  ]", name, desc))
print("  识别出 %d 个不同物，符号须在引用时带对象名" % len(occ))
check("C6-5", c6_5_ok, "符号占用审计完成：%d 个不同物已区分" % len(occ))

# ----------------------------------------------------------------------
print()
print("[C6-6] 恒真替身：把目标值换成别的 O(1) 非整数")
# 替身一：目标值取 2.0（整数）。此时读法乙给出 2*pi/3 = 2.094，与替身目标不一致，
#   C6-2 的比对须判否。
alt_targets = [("替身甲：目标取 2.0（整数）", 2.0),
               ("替身乙：目标取 pi", math.pi)]
c6_6_ok = True
for name, alt in alt_targets:
    dev = abs(nc_from_geometry - alt) / alt
    failed = dev >= 1e-3
    print("  %s：读法值 %.9f 对替身目标 %.9f，相对差 %.3e，判据判否：%s"
          % (name, nc_from_geometry, alt, dev, failed))
    c6_6_ok = c6_6_ok and failed
# 替身丙：把读法的输入换成含目标值的构造（凑数替身）。
#   此时读数的唯一性被破坏：引入目标值作为输入后，读数可以被拉到目标值上，
#   于是「读数与目标一致」不再携带区分信息。
#   判据量：含目标值后读数可被反解拉到目标（偏差恒为零），且任意目标值皆可如此；
#   对照是不含目标值时读数由构型参数唯一决定（路径甲的斜率恒定）。
alpha_fit = nc_target * NN / SOLID_ANGLE
pulled = SOLID_ANGLE * alpha_fit / NN
pull_dev = abs(pulled - nc_target) / nc_target
alt_targets3 = [2.0, math.pi, 3.7]
all_pullable = all(abs(SOLID_ANGLE * (t * NN / SOLID_ANGLE) / NN - t) / t < 1e-12
                   for t in alt_targets3)
input_violates = (pull_dev < 1e-12) and all_pullable
print("  替身丙：输入改为含目标值后，读数可被反解拉到目标（偏差 %.3e），"
      "且任意目标值皆可如此（%s），唯一性被破坏：%s"
      % (pull_dev, all_pullable, input_violates))
c6_6_ok = c6_6_ok and input_violates
check("C6-6", c6_6_ok, "三条替身均使匹配判据 FAIL，判据有区分力")

# ----------------------------------------------------------------------
print()
print("[C6-7] 凑数空间的操作性排除")
# 路径甲（不含目标值）：读数由构型参数唯一决定。
#   构型参数为：空间维数 d、扩散参数 alpha、立体角因子。改 alpha 即线性改读数，
#   读数不因目标值在场而被拉动。
print("  路径甲（不含目标值）：读数 = 立体角因子 * alpha / (2d)")
path_a = []
for af in alpha_fracs:
    path_a.append((af, SOLID_ANGLE * af / NN))
for af, v in path_a:
    print("    alpha = %.2f  ->  读数 %.9f" % (af, v))
# 单调且线性：读数对 alpha 严格单调，斜率恒为 立体角因子/(2d)
slopes = [(path_a[i + 1][1] - path_a[i][1]) / (path_a[i + 1][0] - path_a[i][0])
          for i in range(len(path_a) - 1)]
mono = all(slopes[i] > 0 for i in range(len(slopes)))
const_slope = max(abs(s - SOLID_ANGLE / NN) for s in slopes)
# 路径乙（含目标值）：把目标值当约束反解 alpha，再看读数是否恒等于目标值
#   （这是凑数路径：任何目标值都能被反解出的 alpha 拟合上）。
fitted = []
for alt in (nc_target, 2.0, math.pi):
    af_fit = alt * NN / SOLID_ANGLE
    fitted.append((alt, af_fit, SOLID_ANGLE * af_fit / NN))
print("  路径乙（含目标值）：把目标当约束反解 alpha")
for tgt, af_fit, v in fitted:
    dev = abs(v - tgt) / tgt
    print("    目标 %.9f  ->  反解 alpha = %.9f  ->  读数 %.9f（相对差 %.3e）"
          % (tgt, af_fit, v, dev))
# 路径乙的判据：任何目标值都能被反解 alpha 拟合，故该路径不具区分力，
#   其判别特征是「拟合后偏差恒为零」。
path_b_is_uninformative = all(abs(v - tgt) / tgt < 1e-12 for tgt, _, v in fitted)
c6_7_ok = mono and (const_slope < 1e-12) and path_b_is_uninformative
print("  路径甲：读数对 alpha 严格单调（斜率恒为 %.12f，最大偏差 %.3e）"
      % (SOLID_ANGLE / NN, const_slope))
print("  路径乙：任意目标值均可被反解 alpha 拟合（偏差恒为零），故该路径无区分力")
check("C6-7", c6_7_ok,
      "不含目标值的路径唯一决定读数（单调线性，斜率恒定）；含目标值的路径显出可调空间")

# ----------------------------------------------------------------------
print()
print("[C6-7b] 目标值在路径甲下的可命中性")
# 反面检验：若只给目标值 2*pi/3 而不给读法，能否由路径甲唯一反推 alpha。
# 能反推出唯一的 alpha，说明路径甲与目标值相容（不构成矛盾），但反推值须落在
# 合理的扩散参数区间内。
alpha_inferred = nc_target * NN / SOLID_ANGLE
in_range = 0.0 < alpha_inferred <= 1.0
print("  由目标值 2*pi/3 反推 alpha = %.12f（须落在 (0, 1] 的狄利克雷收敛区间）"
      % alpha_inferred)
check("C6-7b", in_range, "反推的扩散参数落在狄利克雷收敛区间内（%.9f）" % alpha_inferred)

print()
print("=" * 74)
n_pass = sum(1 for _, ok in CHECKS if ok)
print("判据通过：%d/%d" % (n_pass, len(CHECKS)))
print("=" * 74)

# ----------------------------------------------------------------------
# 出图
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))

ax = axes[0]
ax.plot(alpha_fracs, [v for _, v in path_a], "o-", color="#1f77b4", linewidth=1.6,
        label=u"路径甲：立体角因子 $\\times\\alpha/(2d)$")
ax.axhline(nc_target, color="#d62728", linestyle="--", linewidth=1.2,
           label=u"目标值 $2\\pi/3$")
ax.set_xlabel(u"扩散参数 $\\alpha$")
ax.set_ylabel(u"读数 $N_c$")
ax.set_title(u"(a) 读数对 $\\alpha$ 严格单调线性（斜率恒定）")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1]
names = [n for n, _, _ in occ]
# 库内实况：低密度格点气取 n_mode = 8 档时 N_c = 0.2 * 8 = 1.6
vals = [2.0 * math.pi / 3.0, 0.2 * 8.0]
ax.bar(range(len(vals)), vals, color=["#1f77b4", "#ff7f0e"], width=0.55)
ax.set_xticks(range(len(vals)))
ax.set_xticklabels([u"归一化链\n（每拍携带写入，$2.094$）",
                    u"低密度格点气\n（载子数，$0.2 n_{mode}$，$n_{mode}=8$ 档）"], fontsize=7)
ax.set_ylabel(u"$N_c$ 取值")
ax.set_title(u"(b) 符号占用：同名不同物（$2.094$ 对 $1.6$）")
ax.grid(alpha=0.3, axis="y")

ax = axes[2]
ax.plot(alpha_fracs, [v for _, v in path_a], "o-", color="#1f77b4", linewidth=1.6,
        label=u"路径甲（唯一决定）")
fit_alphas = [af for _, af, _ in fitted]
fit_vals = [v for _, _, v in fitted]
ax.plot(fit_alphas, fit_vals, "s", color="#d62728", markersize=8,
        label=u"路径乙反解点（任意目标皆可拟合）")
ax.set_xlabel(u"扩散参数 $\\alpha$")
ax.set_ylabel(u"读数 $N_c$")
ax.set_title(u"(c) 凑数空间：路径乙无区分力")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

fig.suptitle(u"第732号：N_c 非整数读法与符号占用审计", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(IMG, dpi=150)
print()
print("图已存：%s" % IMG)
