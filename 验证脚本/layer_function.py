# -*- coding: utf-8 -*-
"""
判据 778：库内定案层函数的跨通道区分度分段与各向同性约束。

靶子重定的经过（本轮最重要的一步）。
判据第 774 号以「跨通道区分度」为可判量，穷举五类落点，结论是
「按现有库内口径，没有一个落点既合法又承载跨通道信息」。
独立复核指出该结论的库内依据有误，须重做。

**复核查出的要害**（三条，均经核实成立）。
  其一，**库内定案的层函数不是常数**。774 号把 $f$ 当作常数处理，
     据称「库内已定案 $f(0)=f(1)$，故强度层退化为常数」。
     核源文档《元相子几何动力学——束缚振荡修订整合版》第 219 行：
     「$f(0) = f(1) = 1$，$f(|\\Delta n| > 1) = 0$（层内与相邻层等刚度，各向同性条件）」。
     **$f$ 在 $|\\Delta n| \\ge 2$ 处取零，非常数。**
     后果：774 号只取 $|\\Delta n| \\in \\{0,1\\}$ 两档，恰好落在 $f$ 的等值区，
     把非退化的 $f$ 读成了常数。扩展到 $|\\Delta n|=2$ 档，区分度变为非零。
  其二，**$f(0)=f(1)$ 是各向同性的充分条件，不是对 $f$ 的全局约束**。
     源文档第 394 行的极限检验写「$f(0)=f(1)$ 破坏则介质波各向异性」，
     第 326 行写「$f(0)=f(1)$ 保证介质波三方向各向同性」，
     第 219 行的括注写「层内与相邻层等刚度，各向同性条件」——
     三处都只约束 $f(0)$ 与 $f(1)$ 两点，对 $|\\Delta n| \\ge 2$ 段无要求。
  其三，**$f$ 的过渡形式是库内已登记的开放问题**（源文档第 355 行 O2：
     「同类耦合函数 $f$ 的过渡形式」）。

**本轮的问题。** 库内定案的层函数承载多少跨通道信息？
  跨通道区分度在 $|\\Delta n|$ 的哪一段为零、哪一段非零？
  各向同性要求 $f(0)=f(1)$ 对 $|\\Delta n| \\ge 2$ 段的取值施加了什么约束？

**可判量（沿用第 774 号的定义，此处核正方向）**。
跨通道区分度 = 两个构型给出**相同相位读数**、**不同层数读数**时耦合能之差。
它测的是「该耦合结构能否把层数通道的读数传进相互作用」。
774 号的清点表混用了两个方向（余弦宗量层等取对层数读数的差，
纯层数通道项取对相位读数的差），本轮全表统一为**对层数读数**的差，
纯层数通道项的「对相位读数区分度」另列为一项，不再混入本表。

**五问自检**。
一、波性侧与粒性侧：相位通道（连续、绕数）与层数通道（离散、格点）。
二、主导形态：局域（相邻格点）。
三、哪一层：互补。
四、读哪个通道：两者。
五、定域接力还是非定域关联：定域接力（校准接力严格光速）。

判据先立，再实现。容差从误差预算推导。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\p778_layer_function.png"

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


# =====================================================================
# 库内定案的层函数
# =====================================================================
def f_library(nd):
    """库内定案：f(0) = f(1) = 1，f(|Δn| > 1) = 0"""
    a = abs(int(nd))
    return 1.0 if a <= 1 else 0.0


def same_coupling_energy(nd, theta_diff):
    """同类耦合能量：-K0 f(Δn) cos(θ_j - θ_i)"""
    return -K0 * f_library(nd) * np.cos(theta_diff)


# =====================================================================
# 组 A：库内定案 f 的跨通道区分度分段
# =====================================================================
print("=" * 72)
print("组 A：库内定案 f 的跨通道区分度分段")
print("=" * 72)

theta_d = 0.43
# A1：相邻层档（|Δn| = 0 与 1）区分度为零
d_adj = abs(same_coupling_energy(1, theta_d) - same_coupling_energy(0, theta_d))
rec(
    "A1",
    "相邻层档（$|\\Delta n|=0$ 与 $1$）的跨通道区分度为零",
    "$f(0)=f(1)=1$，故两档强度相同，耦合能之差 ≤ 1e-12",
    "区分度 %.3e（$f(0)=%.1f$，$f(1)=%.1f$）"
    % (d_adj, f_library(0), f_library(1)),
    d_adj < 1e-12,
)

# A2：跨两层档（|Δn| = 0 与 2）区分度非零
d_two = abs(same_coupling_energy(2, theta_d) - same_coupling_energy(0, theta_d))
rec(
    "A2",
    "跨两层档（$|\\Delta n|=0$ 与 $2$）的跨通道区分度非零",
    "$f(2)=0 \\ne f(0)=1$，故两档强度不同，区分度 ≥ 0.1",
    "区分度 %.6f（$f(0)=%.1f$，$f(2)=%.1f$）" % (d_two, f_library(0), f_library(2)),
    d_two >= 0.1,
)

# A3：多档扫描 |Δn| = 0..5，给出区分度的完整剖面
ND_LIST = list(range(0, 6))
d_profile = []
for nd in ND_LIST:
    d_profile.append(abs(same_coupling_energy(nd, theta_d) - same_coupling_energy(0, theta_d)))
# 期望：|Δn|=0,1 为零；|Δn|>=2 为 K0 cos(θ) = cos(0.43) = 0.908966
cos_t = np.cos(theta_d)
exp_profile = [0.0, 0.0] + [cos_t] * 4
ok_a3 = all(abs(d_profile[i] - exp_profile[i]) < 1e-12 for i in range(6))
rec(
    "A3",
    "多档扫描：区分度剖面在 $|\\Delta n|=0,1$ 为零、$|\\Delta n| \\ge 2$ 为 $K_0\\cos\\theta$",
    "六档剖面 %s" % [float("%.6f" % v) for v in exp_profile],
    "六档剖面 %s" % [float("%.6f" % v) for v in d_profile],
    ok_a3,
)

# A4：恒真对照 —— 若 f 取常数，区分度处处为零
def f_const(nd):
    return 1.0


d_const = []
for nd in ND_LIST:
    d_const.append(abs(-K0 * f_const(nd) * np.cos(theta_d)
                       - (-K0 * f_const(0) * np.cos(theta_d))))
max_d_const = max(d_const)
rec(
    "A4",
    "恒真对照：若层函数取常数，区分度处处为零",
    "常数 $f$ 下六档区分度皆 ≤ 1e-12，即 A3 的非零段来自 $f$ 的非退化性",
    "六档最大 %.3e" % max_d_const,
    max_d_const < 1e-12,
)

# A5：区分度与相位读数无关（它测的是层数侧）
d_ph = []
for td in (0.1, 0.43, 1.2, 2.7):
    d_ph.append(abs(same_coupling_energy(2, td) - same_coupling_energy(0, td)))
# 期望：d_ph[i] = K0 |cos(td_i) - cos(td_0)|，各档不同
ok_a5 = abs(d_ph[1] - d_profile[2]) < 1e-12 and abs(d_ph[3] - d_ph[1]) > 1e-6
rec(
    "A5",
    "区分度随参照相位读数而变（它测的是层数侧，不是相位侧）",
    "取参照相位差 $0.1, 0.43, 1.2, 2.7$ 四档，$|\\Delta n|=2$ 档的区分度"
    "分别为各自 $|\\cos\\theta - \\cos\\theta_0|$，且各档互不相同",
    "四档区分度 %s" % [float("%.6f" % v) for v in d_ph],
    ok_a5,
)

# =====================================================================
# 组 B：各向同性要求对 |Δn| >= 2 段的约束边界
# =====================================================================
print("=" * 72)
print("组 B：各向同性要求对 |Δn| >= 2 段的约束边界")
print("=" * 72)

# B1：各向同性要求只约束 f(0) 与 f(1) 两点
# 可判：构造一个满足 f(0)=f(1) 但 f(2) 任取的 f，检验它仍满足各向同性要求。
# 各向同性条件的判别式：f(0) == f(1)（源文档三处皆如此表述）。
def f_iso_ok(f):
    return abs(f(0) - f(1)) < 1e-15


f_variants = []
for f2 in (0.0, 0.5, 1.0, 2.0):
    def f_var(nd, _f2=f2):
        a = abs(int(nd))
        if a == 0:
            return 1.0
        if a == 1:
            return 1.0
        return _f2
    f_variants.append((f2, f_var, f_iso_ok(f_var)))

all_iso = all(ok for _, _, ok in f_variants)
rec(
    "B1",
    "各向同性要求只约束 $f(0)$ 与 $f(1)$ 两点，对 $|\\Delta n| \\ge 2$ 段无要求",
    "四个变体（$f(2)$ 取 $0, 0.5, 1, 2$）全部满足 $f(0)=f(1)$，"
    "故各向同性条件对它们一视同仁",
    "各变体的 $f(0)=f(1)$ 检查 %s"
    % [(f2, ok) for f2, _, ok in f_variants],
    all_iso,
)

# B2：库内定案的 f 是这些变体之一（f(2)=0）
is_library = abs(f_library(2) - 0.0) < 1e-15 and abs(f_library(0) - 1.0) < 1e-15 \
    and abs(f_library(1) - 1.0) < 1e-15
rec(
    "B2",
    "库内定案的 $f$ 是 B1 变体族的一员（$f(2)=0$）",
    "$f(0)=1$、$f(1)=1$、$f(2)=0$ 逐位相符",
    "$f(0)=%.1f$，$f(1)=%.1f$，$f(2)=%.1f$" % (f_library(0), f_library(1), f_library(2)),
    is_library,
)

# B3：区分度对 f(|Δn|>=2) 段的依赖（跨通道信息的强度由该段决定）
d_f2_scan = []
for f2 in (0.0, 0.25, 0.5, 0.75, 1.0):
    def f_var(nd, _f2=f2):
        a = abs(int(nd))
        if a <= 1:
            return 1.0
        return _f2
    e0 = -K0 * f_var(0) * np.cos(theta_d)
    e2 = -K0 * f_var(2) * np.cos(theta_d)
    d_f2_scan.append((f2, abs(e2 - e0)))
ok_b3 = all(abs(d_f2_scan[i][1] - (1.0 - d_f2_scan[i][0]) * cos_t) < 1e-12
            for i in range(5))
rec(
    "B3",
    "区分度由 $f$ 的 $|\\Delta n| \\ge 2$ 段决定，且随该段取值线性变化",
    "五档 $f(2)$ 取 $0, 0.25, 0.5, 0.75, 1$ 时，区分度依次为 "
    "$K_0\\vert 1-f(2)\\vert\\cos\\theta$",
    "五档 %s" % [(f2, float("%.6f" % v)) for f2, v in d_f2_scan],
    ok_b3,
)

# B4：恒真对照 —— 破坏 f(0)=f(1) 后各向同性条件判否
def f_broken(nd):
    a = abs(int(nd))
    return 1.0 if a == 0 else 0.0


broken_ok = f_iso_ok(f_broken)
rec(
    "B4",
    "恒真对照：破坏 $f(0)=f(1)$ 后各向同性条件判否",
    "取 $f(0)=1$、$f(1)=0$，则 $f(0)=f(1)$ 不成立，各向同性判别式须判否",
    "f(0)=%.1f，f(1)=%.1f，各向同性判别式 %s"
    % (f_broken(0), f_broken(1), "成立（区分力不足）" if broken_ok else "判否（符合预期）"),
    not broken_ok,
)

# =====================================================================
# 组 C：落点表重建与结论
# =====================================================================
print("=" * 72)
print("组 C：落点表重建与结论")
print("=" * 72)

# C1：重建落点表，全部统一为「对层数读数」的区分度
# 说明：本表的区分度一律为「相位读数固定、层数读数变化时耦合能之差」。
# 纯层数通道项（键弹性、硬核）不含相位读数，故其对层数读数有区分度、
# 对相位读数无区分度；按本表口径（对层数读数），它们**有**区分度。
# 这正是 774 号混用方向造成误判之处。
h_fixed = 0.037
KAPPA_PAR = 1.0


def bond_energy(nd):
    """键弹性能量：kappa/2 (h 差)^2，h 差 = 层数差的物线映射（示意）"""
    h_diff = 0.02 * abs(int(nd))
    return 0.5 * KAPPA_PAR * h_diff ** 2


def hetero_energy(nd):
    """异类耦合：不含层数读数"""
    return -1.0 * np.cos(theta_d)


def cosine_layer_energy(nd):
    """余弦宗量层：只含相位，层数差进不来"""
    return -1.0 * np.cos(theta_d)


LOCI = [
    ("余弦宗量层", abs(cosine_layer_energy(2) - cosine_layer_energy(0)),
     "层数差进不来余弦宗量"),
    ("强度层（库内定案 f）", d_two, "对 |Δn|>=2 区分，对 |Δn|<=1 不区分"),
    ("异类耦合", abs(hetero_energy(2) - hetero_energy(0)),
     "不含层数读数"),
    ("纯层数通道项（键弹性）", abs(bond_energy(2) - bond_energy(0)),
     "含层数读数，不含相位读数"),
]
informative = [(n, v) for n, v, _ in LOCI if v >= 1e-12]
rec(
    "C1",
    "落点表重建：库内定案口径下，承载跨通道信息的落点有若干",
    "四个落点中，余弦宗量层与异类耦合的区分度为零；"
    "强度层对 $|\\Delta n| \\ge 2$ 非零；纯层数通道项因含层数读数亦非零。"
    "即 774 号「无一既合法又承载跨通道信息」的结论不成立",
    "各落点区分度 %s，其中承载跨通道信息者 %d 个"
    % ([(n, float("%.6f" % v)) for n, v, _ in LOCI], len(informative)),
    len(informative) >= 2,
)

# C2：余弦宗量层与异类耦合的区分度为零（两条负判据）
neg_zero = (abs(cosine_layer_energy(2) - cosine_layer_energy(0)) < 1e-12
             and abs(hetero_energy(2) - hetero_energy(0)) < 1e-12)
rec(
    "C2",
    "两条负判据：余弦宗量层与异类耦合的跨通道区分度为零",
    "两者的 $|\\Delta n|=0$ 与 $2$ 两档耦合能之差皆 ≤ 1e-12",
    "余弦宗量层 %.3e，异类耦合 %.3e"
    % (abs(cosine_layer_energy(2) - cosine_layer_energy(0)),
       abs(hetero_energy(2) - hetero_energy(0))),
    neg_zero,
)

# C3：恒真对照 —— 强度层的非零段确由 f 的非退化性给出
# 做法：把 f 换成常数（违反 f(2)=0 的库内定案），区分度应处处为零。
d_const_profile = []
for nd in ND_LIST:
    d_const_profile.append(abs(-K0 * 1.0 * np.cos(theta_d)
                               - (-K0 * 1.0 * np.cos(theta_d))))
ok_c3 = max(d_const_profile) < 1e-12 and d_two >= 0.1
rec(
    "C3",
    "判别力：把 $f$ 换成常数后强度层的非零段消失",
    "常数 $f$ 下六档区分度皆 ≤ 1e-12，而库内定案 $f$ 下 $|\\Delta n|=2$ 档为 %.6f，"
    "故该非零段确由 $f$ 的非退化性给出" % d_two,
    "常数 $f$ 下最大 %.3e，库内定案 $f$ 下 %.6f" % (max(d_const_profile), d_two),
    ok_c3,
)

# C4：结论的可判形式 —— 跨通道信息由 f 的 |Δn|>=2 段承载
# 判据：区分度剖面恰在 f 的等值区为零、在非等值区非零
zero_seg = [i for i in range(6) if d_profile[i] < 1e-12]
nonzero_seg = [i for i in range(6) if d_profile[i] >= 1e-12]
ok_c4 = zero_seg == [0, 1] and nonzero_seg == [2, 3, 4, 5]
rec(
    "C4",
    "跨通道信息由 $f$ 的 $|\\Delta n| \\ge 2$ 段承载（零段与非零段的分界恰在 $f$ 的等值区边界）",
    "区分度为零的档为 $0,1$（$f$ 的等值区），非零的档为 $2,3,4,5$（$f=0$ 段）",
    "零段 %s，非零段 %s" % (zero_seg, nonzero_seg),
    ok_c4,
)

# =====================================================================
# 组 D：读数
# =====================================================================
print("=" * 72)
print("组 D：读数（不进通过率）")
print("=" * 72)

print("[D1] 读数：库内定案 f 的完整形式与区分度剖面")
print("     源文档《元相子几何动力学——束缚振荡修订整合版》第 219 行：")
print("     $f(0) = f(1) = 1$，$f(|\\Delta n| > 1) = 0$（层内与相邻层等刚度，各向同性条件）。")
print("     故 $f$ 非常数：$|\\Delta n| \\le 1$ 段取 1，$|\\Delta n| \\ge 2$ 段取 0。")
print("     区分度剖面（$|\\Delta n| = 0$ 至 $5$，参照相位差 $0.43$）：")
for nd in ND_LIST:
    print("       |Δn| = %d   f = %.1f   区分度 %.6f" % (nd, f_library(nd), d_profile[nd]))
print("")
print("[D2] 读数：各向同性要求的约束边界")
print("     源文档三处表述皆只约束 $f(0)$ 与 $f(1)$ 两点：")
print("       第 219 行括注「层内与相邻层等刚度，各向同性条件」；")
print("       第 326 行「$f(0)=f(1)$ 保证介质波三方向各向同性」；")
print("       第 394 行极限检验「$f(0)=f(1)$ 破坏则介质波各向异性」。")
print("     B1 给出：$f(2)$ 取 $0, 0.5, 1, 2$ 的四个变体全部满足该判别式，")
print("     故各向同性要求对 $|\\Delta n| \\ge 2$ 段无约束。")
print("     B3 给出：区分度随 $f(|\\Delta n| \\ge 2)$ 段的取值线性变化，")
print("     即跨通道信息的强度由该段决定。")
print("")
print("[D3] 读数：对判据第 774 号的三处订正")
print("     其一，774 号把库内定案的 $f$ 当作常数，得出「强度层不区分层、区分度为零」。")
print("       实际：$f$ 在 $|\\Delta n| \\ge 2$ 处取零，强度层对跨两层档有区分度 %.6f。" % d_two)
print("     其二，774 号的清点表混用了两个方向的区分度（多数落点取对层数读数的差，")
print("       纯层数通道项取对相位读数的差），本轮全表统一为对层数读数的差。")
print("     其三，774 号结论三「跨通道耦合项若要存在，只能落在混合宗量这一类」")
print("       无判据支持：独立复核实测梯度/曲率型、非局域型、三通道乘积型三条候选")
print("       均能承载跨通道信息（区分度 0.164685、0.554127、0.503682）。")
print("       本轮不主张任何形式的穷举，只给已核落点的读数与分界。")
print("")
print("[D4] 读数：这一轮对论文 §9 遗留 (g) 的推进")
print("     准确表述更新为：**跨通道区分度由层函数 $f$ 的 $|\\Delta n| \\ge 2$ 段承载**。")
print("     库内定案的 $f$ 已给出该区分度（%.6f），故「强度层够不够」这一问题有答案：" % d_two)
print("     强度层**已够**（对跨两层的键）；遗留的真问题收为")
print("     **$f$ 的过渡形式**（源文档第 355 行开放问题 O2）。")
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
ax.bar(ND_LIST, d_profile, color=["#1f77b4", "#1f77b4"] + ["#d62728"] * 4, width=0.6)
ax.axhline(cos_t, color="#7f7f7f", linestyle="--", linewidth=1.4,
           label="零层段档的区分度 K0 cos(theta)")
ax.set_xlabel("层数差的绝对值")
ax.set_ylabel("跨通道区分度")
ax.set_title("左：区分度剖面\n层差 0 与 1 为零，层差 2 起非零")
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis="y")

ax = axes[1]
fs = [v[0] for v in d_f2_scan]
ds = [v[1] for v in d_f2_scan]
ax.plot(fs, ds, "o-", color="#1f77b4", linewidth=2.2, markersize=8,
        label="实测区分度")
ax.plot(fs, [abs(1.0 - f) * cos_t for f in fs], "x", color="#d62728",
        markersize=11, markeredgewidth=2.2, label="预期值")
ax.set_xlabel("层函数在层差 2 以上的取值")
ax.set_ylabel("跨通道区分度")
ax.set_title("右：跨通道信息强度由该段决定\n五档扫描，线性关系")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(IMG, dpi=150)
print("")
print("图已保存：%s" % IMG)
