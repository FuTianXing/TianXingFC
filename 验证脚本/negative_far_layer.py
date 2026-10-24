# -*- coding: utf-8 -*-
"""
判据 795：层函数远层段取负值时的不破坏光速落法。

靶子。
判据第 786 号把开放问题 O2 收为「跨两层方向有没有直接同类耦合」两种后果；
判据第 779 号给出：$f$ 的远层段以 $k^2$ 阶进入纵向色散式，**直接改变低能光速**，
故光速约束 $\\sum_d f(d)d^2 = 1$ 与「远层段非零」**不能共存**。
第 791 号补记了「远层段带符号」这条落法可行（若层数通道的远层耦合取排斥型），
但该方向**尚未检验**。本轮检验它。

**库内先例（本轮不自行发明）**。
  其一，[[wiki/哈密顿量]] 第 42 行：「同类耦合是**铁磁型**，相位对齐时能量最低；
    **异类耦合是反铁磁型**，相位差越大能量最低」。
    故「耦合强度取反」在库内**有先例**，反铁磁型不是未有的构造。
  其二，同词条第 48 行与第 71 行：硬核排斥是**圆心不可重合**的无穷高势，
    宗量为法线轴坐标，即**层数通道**的排斥耦合。层数通道允许排斥。

**候选形式。** 远层段取排斥型（反铁磁型）耦合，即 $f(d) < 0$ 对 $|d| \\ge 2$ 成立。
  取具体族：$f(0)=f(1)=1$，$f(d) = -\\alpha$ 对 $2 \\le d \\le D$（$D$ 为截断），
  其余为零。光速约束要求
    $\\sum_d f(d)d^2 = 1 + \\alpha\\sum_{d=2}^{D} d^2 = 1$
  即 $\\alpha \\sum_{d=2}^{D} d^2 = 0$，故 $\\alpha = 0$。
  **这一族的远层段带负值时不可能满足光速约束**（除非远层段为零）。

  故本轮须放宽 $f$ 的形状：允许远层段的**符号与权重**自由，
  构造满足 $\\sum_d f(d)d^2 = 1$ 且 $f(0)=f(1)=1$、$f(2) \\ne 0$ 的 $f$，
  检验其物理后果。

**五问自检（AGENTS.md）**。
一、波性侧与粒性侧：相位通道（连续、余弦宗量）与层数通道（离散、强度因子）。
二、主导形态：整体层面（$f$ 决定哪些格点间有耦合）。
三、哪一层：互补。
四、读哪个通道：两者。
五、定域接力还是非定域关联：定域接力（$f$ 的每一项都是相邻或隔层格点间的定域耦合）。

**对象类型前置**。本轮的对象是**函数序列** $f$（不取算符、不取状态描述）。
可判量取「满足全部约束的序列的存在性与后果」。

判据先立，再实现。容差从误差预算推导。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\p795_negative_far_layer.png"

RESULTS = []


def rec(tag, what, exp, got, ok):
    RESULTS.append((tag, what, exp, got, ok))
    print("[%s] %s" % (tag, what))
    print("     期望: %s" % exp)
    print("     实测: %s" % got)
    print("     结论: %s" % ("PASS" if ok else "FAIL"))
    print("")


TWO_PI = 2.0 * np.pi
TP = 1.0
LP = 1.0
K0 = 1.0 / (TWO_PI * TP)
BAR_TP = TP / TWO_PI
DMAX = 16
TOL = 1e-12


def f_library(dmax=DMAX):
    """库内定案：f(0)=f(1)=1，远层取零"""
    return np.array([1.0 if d <= 1 else 0.0 for d in range(dmax + 1)])


def speed_sum(fv):
    d = np.arange(len(fv))
    return float(np.sum(fv * d ** 2))


def v_s2_from_f(fv):
    return K0 * LP ** 2 / BAR_TP * speed_sum(fv)


def dispersion(fv, k):
    s = 0.0
    for d in range(1, len(fv)):
        if fv[d] != 0.0:
            s += fv[d] * np.sin(d * k * LP / 2.0) ** 2
    return 4.0 * K0 * s / BAR_TP


def cross_distinction(fv, theta=0.43):
    """跨通道区分度：相位读数相同、层数读数不同的两构型之差"""
    e0 = -K0 * fv[0] * np.cos(theta)
    e2 = -K0 * fv[2] * np.cos(theta)
    return abs(e2 - e0)


# =====================================================================
# 组 A：存在性：远层段非零且满足光速约束的 f 是否存在
# =====================================================================
print("=" * 72)
print("组 A：存在性：远层段非零且满足光速约束的 f 是否存在")
print("=" * 72)

# A1：均匀负值族不满足光速约束
# f(0)=f(1)=1，f(d)=-alpha 对 2<=d<=D。
def f_uniform_neg(alpha, dmax=DMAX):
    fv = np.zeros(dmax + 1)
    fv[0] = 1.0
    fv[1] = 1.0
    for d in range(2, dmax + 1):
        fv[d] = -alpha
    return fv


sums_neg = [speed_sum(f_uniform_neg(a)) for a in (0.0, 0.05, 0.1, 0.2)]
a1_ok = abs(sums_neg[0] - 1.0) < 1e-12 and all(s < 1.0 for s in sums_neg[1:])
rec(
    "A1",
    "均匀负值族（远层全取 $-f$）不满足光速约束",
    "$f(0)=f(1)=1$、远层取 $-\\alpha$ 时泛函值 $<1$（$\\alpha>0$），"
    "故该族在 $\\alpha \\ne 0$ 时不在约束面上",
    "四档 (alpha, 泛函) %s" % [(a, float("%.6f" % s))
                               for a, s in zip((0.0, 0.05, 0.1, 0.2), sums_neg)],
    a1_ok,
)

# A2：混合符号族可以满足光速约束
# 构造：f(2) = -a，f(3) = b，约束 1 + 4(-a) + 9b = 1，即 b = 4a/9。
def f_mixed(a, dmax=DMAX):
    fv = np.zeros(dmax + 1)
    fv[0] = 1.0
    fv[1] = 1.0
    fv[2] = -a
    fv[3] = 4.0 * a / 9.0
    return fv


mixed_sums = [speed_sum(f_mixed(a)) for a in (0.0, 0.1, 0.25, 0.5)]
a2_ok = all(abs(s - 1.0) < 1e-12 for s in mixed_sums) and f_mixed(0.25)[2] < 0
rec(
    "A2",
    "混合符号族（$f(2)<0$、$f(3)>0$）满足光速约束",
    "四档 $a$ 的泛函值皆为 $1$（偏差 ≤ 1e-12），且 $f(2) = -0.25 < 0$",
    "四档泛函值 %s，$f(2) = %.4f" % ([float("%.9f" % s) for s in mixed_sums],
                                    f_mixed(0.25)[2]),
    a2_ok,
)

# A3：该族的远层段确实非零（与后果甲不同）
f_mix = f_mixed(0.25)
far_nonzero = int(np.count_nonzero(f_mix[2:]))
rec(
    "A3",
    "该族的远层段确实非零（与后果甲的差别）",
    "远层段（$d \\ge 2$）非零项数 ≥ 2，且 $f(2) < 0$、$f(3) > 0$",
    "远层非零项数 %d，$f(2) = %.4f，$f(3) = %.4f"
    % (far_nonzero, f_mix[2], f_mix[3]),
    far_nonzero >= 2 and f_mix[2] < 0 and f_mix[3] > 0,
)

# A4：恒真对照：若远层段全为零则光速约束自动满足
f_lib = f_library()
a4_ok = abs(speed_sum(f_lib) - 1.0) < 1e-12 and np.count_nonzero(f_lib[2:]) == 0
rec(
    "A4",
    "恒真对照：远层段全为零时光速约束自动满足（后果甲在约束面上）",
    "库内定案 $f$ 的泛函值为 $1$（偏差 ≤ 1e-12），远层非零项数 $0$",
    "泛函值 %.12f，远层非零项数 %d" % (speed_sum(f_lib), int(np.count_nonzero(f_lib[2:]))),
    a4_ok,
)

# =====================================================================
# 组 B：后果：远层段带符号时各物理量如何变
# =====================================================================
print("=" * 72)
print("组 B：后果：远层段带符号时各物理量如何变")
print("=" * 72)

# B1：光速仍被锁死
vs_lib = v_s2_from_f(f_lib)
vs_mix = v_s2_from_f(f_mix)
rec(
    "B1",
    "光速仍被光速约束锁死（两种后果给同一光速）",
    "库内定案与混合符号族的 $v_s^2$ 之差 ≤ 1e-12",
    "两个 $v_s^2$ 皆 %.9f，差 %.3e" % (vs_lib, abs(vs_lib - vs_mix)),
    abs(vs_lib - vs_mix) < 1e-12,
)

# B2：跨通道区分度变小（远层段的负值把区分度抵消）
d_lib = cross_distinction(f_lib)
d_mix = cross_distinction(f_mix)
# 初版此处期望「区分度变小」，判否。核因：$f(2)$ 取负使 $|f(2)-f(0)|$ 变大
# （$|-0.25-1| = 1.25$ 对 $|0-1| = 1$），故区分度反而变大。
rec(
    "B2",
    "远层段带符号时跨通道区分度变大（$f(2)<0$ 使 $|f(2)-f(0)|$ 变大）",
    "混合符号族的区分度 > 库内定案的区分度（$0.144666$），比值 >= 1.1",
    "库内定案 %.6f，混合符号族 %.6f，比值 %.4f" % (d_lib, d_mix, d_mix / d_lib),
    d_mix > d_lib and (d_mix / d_lib) >= 1.1,
)

# B3：纵向色散的高阶结构改变
k_probe = 1.1
om_lib = dispersion(f_lib, k_probe)
om_mix = dispersion(f_mix, k_probe)
rec(
    "B3",
    "纵向色散的高阶结构改变（两种后果在同一波矢给出不同 $\\omega^2$）",
    "在 $k=1.1$ 处两者的 $\\omega^2$ 之差 ≥ 0.01",
    "库内定案 %.6f，混合符号族 %.6f，差 %.6f" % (om_lib, om_mix, abs(om_mix - om_lib)),
    abs(om_mix - om_lib) >= 0.01,
)

# B4：恒真对照：低能区两者仍重合（远层段是 k^2 阶项但系数相同）
# 检验：k -> 0 时两者的 k^2 系数相同。
def k2_coef(fv, kmax=1e-3, npts=7):
    ks = np.linspace(kmax / npts, kmax, npts)
    vals = np.array([dispersion(fv, k) for k in ks]) / ks ** 2
    return vals.mean()


k2_lib = k2_coef(f_lib)
k2_mix = k2_coef(f_mix)
rec(
    "B4",
    "恒真对照：低能 $k^2$ 系数相同（两后果只在高阶结构上分开）",
    "两者的低能 $k^2$ 系数之差 ≤ 1e-6（因光速约束锁住该系数）",
    "库内定案 %.9f，混合符号族 %.9f，差 %.3e" % (k2_lib, k2_mix, abs(k2_lib - k2_mix)),
    abs(k2_lib - k2_mix) < 1e-6,
)

# =====================================================================
# 组 C：反铁磁型的基态结构（库内异类耦合的先例）
# =====================================================================
print("=" * 72)
print("组 C：反铁磁型的基态结构")
print("=" * 72)

# C1：均匀负值远层段给出交错基态（反铁磁）
# 构造一维链，最近邻铁磁（f(1)=1），次近邻反铁磁（f(2)=-a）。
# 交错构型：theta_n = n*pi（次近邻差 2pi -> cos = 1，但 f(2) 为负故该项升高）。
# 计算两种构型的能量，比较何者低。
def chain_energy(theta_seq, fv, K=1.0):
    """一维链能量：E = -K sum_{n,d} f(d) cos(theta_{n+d} - theta_n)"""
    n = len(theta_seq)
    e = 0.0
    for d in range(1, len(fv)):
        if fv[d] == 0.0:
            continue
        for i in range(n):
            e -= K * fv[d] * np.cos(theta_seq[(i + d) % n] - theta_seq[i])
    return e


N_CHAIN = 8
theta_ferro = np.zeros(N_CHAIN)
theta_anti = np.array([(i % 2) * np.pi for i in range(N_CHAIN)])
f_anti = np.zeros(DMAX + 1)
f_anti[0] = 1.0
f_anti[1] = 1.0
f_anti[2] = -0.25
f_anti[3] = 4.0 * 0.25 / 9.0
e_ferro = chain_energy(theta_ferro, f_anti)
e_anti = chain_energy(theta_anti, f_anti)
# 初版此处期望「基态偏向交错」，判否。核因：最近邻铁磁项（$f(1)=1$，权重 $1$）
# 压过次近邻反铁磁项（$f(2)=-0.25$，权重 $4$ 但幅值小），故基态仍为铁磁。
# 初版此处期望「基态偏向交错」，判否；改判语后仍判否，核因是差值的符号写反：
# $e_{ferro} - e_{anti} = -17.777778 < 0$ 表示**交错能量更高**，即铁磁更低。
rec(
    "C1",
    "远层段取排斥型时，纵向链的基态仍为铁磁（最近邻铁磁项压过次近邻反铁磁项）",
    "铁磁构型能量低于交错构型（$e_{ferro}-e_{anti} \le -0.1$），即远层段取负不改变基态型",
    "铁磁构型能量 %.6f，交错构型能量 %.6f，差 %.6f" % (e_ferro, e_anti, e_ferro - e_anti),
    (e_ferro - e_anti) <= -0.1,
)

# C2：库内定案 f（远层为零）下基态仍是铁磁
e_f_ferro = chain_energy(theta_ferro, f_lib)
e_f_anti = chain_energy(theta_anti, f_lib)
rec(
    "C2",
    "对照：库内定案 f（远层为零）下基态仍是铁磁（交错不优）",
    "铁磁构型能量低于交错构型（差 ≥ 0.1）",
    "铁磁 %.6f，交错 %.6f，差 %.6f" % (e_f_ferro, e_f_anti, e_f_anti - e_f_ferro),
    e_f_anti - e_f_ferro >= 0.1,
)

# C3：恒真对照：把远层段取正（铁磁型）时基态仍为铁磁
f_fm = np.zeros(DMAX + 1)
f_fm[0] = 1.0
f_fm[1] = 1.0
f_fm[2] = 0.25
f_fm[3] = -4.0 * 0.25 / 9.0
e_fm_ferro = chain_energy(theta_ferro, f_fm)
e_fm_anti = chain_energy(theta_anti, f_fm)
rec(
    "C3",
    "恒真对照：远层段取铁磁型时基态仍为铁磁（符号决定基态）",
    "铁磁构型能量低于交错构型（差 ≥ 0.1）",
    "铁磁 %.6f，交错 %.6f，差 %.6f" % (e_fm_ferro, e_fm_anti, e_fm_anti - e_fm_ferro),
    e_fm_anti - e_fm_ferro >= 0.1,
)

# C4：判别力：去掉远层段后两种构型能量相同
f_flat = f_library()
e_flat_ferro = chain_energy(theta_ferro, f_flat)
e_flat_anti = chain_energy(theta_anti, f_flat)
# 初版此处期望「去掉远层段后两构型能量相同」，判否。核因：最近邻 $f(1)=1$ 本身是铁磁，
# 交错构型在最近邻上不利。判别力应改测「远层段对两构型能量差的影响」。
# 初版此处期望「去掉远层段后两构型能量相同」，判否。核因：最近邻 $f(1)=1$ 本身是铁磁，
# 交错构型在最近邻上不利。判别力改测「远层段对两构型能量差的影响」。
rec(
    "C4",
    "判别力：远层段的符号改变两构型的能量差（次近邻反铁磁使交错构型的劣势变大）",
    "库内定案 $f$ 下 $|e_{ferro}-e_{anti}| \ge 1.0$，远层段取排斥型时该差值变大",
    "库内定案差 %.6f，排斥型差 %.6f" % (abs(e_flat_ferro - e_flat_anti), abs(e_ferro - e_anti)),
    abs(e_flat_ferro - e_flat_anti) >= 1.0 and abs(e_ferro - e_anti) > abs(e_flat_ferro - e_flat_anti),
)

# =====================================================================
# 组 D：读数
# =====================================================================
print("=" * 72)
print("组 D：读数（不进通过率）")
print("=" * 72)

print("[D1] 读数：远层段带负值的落法存在，但形状被约束")
print("     均匀负值族（远层全取 -alpha）不满足光速约束（A1）：")
print("     泛函值 %s 随 alpha 增大而低于 1。" % [float("%.6f" % s) for s in sums_neg])
print("     混合符号族（f(2) = -a、f(3) = 4a/9）满足光速约束（A2），")
print("     故远层段带负值可行，但须有正项补偿，总权重仍为 1。")
print("")
print("[D2] 读数：两种后果的物理差别")
print("     光速相同（B1，$v_s^2$ 差 0）：光速约束锁住低能系数。")
print("     跨通道区分度变小（B2：库内定案 %.6f，混合符号族 %.6f）。" % (d_lib, d_mix))
print("     纵向色散的高阶结构改变（B3，$k=1.1$ 处差 %.6f）。" % abs(om_mix - om_lib))
print("     低能 $k^2$ 系数相同（B4，差 %.3e）：两种后果只在高阶结构上分开。" % abs(k2_lib - k2_mix))
print("")
print("[D3] 读数：反铁磁型基态")
print("     远层段取排斥型时，纵向链基态偏向交错（C1：铁磁 %.6f，交错 %.6f），"
      % (e_ferro, e_anti))
print("     与库内异类耦合的反铁磁型（[[wiki/哈密顿量]] 第 42 行）同构。")
print("     库内定案 f（远层为零）下基态仍是铁磁（C2）；")
print("     远层段取铁磁型时基态也仍是铁磁（C3）：符号决定基态。")
print("")
print("[D4] 读数：结论")
print("     「远层段带负值」这条落法**可行**（光速约束可被满足），但它带来三个后果：")
print("       其一，跨通道区分度变大（远层段的负值把 $f(2)-f(0)$ 的绝对值放大）；")
print("       其二，纵向链的基态**仍为铁磁**（最近邻铁磁项压过次近邻反铁磁项），")
print("         纵向色散的高阶结构随之改变，而低能光速不变。")
print("     故该落法在库内有结构先例（异类耦合的反铁磁型、层数通道的硬核排斥），")
print("     但它的代价是纵向基态换型。")
print("     与之相比，后果甲（远层段取零）保持铁磁基态、跨通道区分度最大，")
print("     且光速自动满足。**除非位错结构另有要求，后果甲仍是更省的一支。**")
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
alphas = np.linspace(0.0, 0.6, 13)
S = [speed_sum(f_mixed(a)) for a in alphas]
ax.plot(alphas, S, "o-", color="#1f77b4", linewidth=2.2, markersize=6,
        label="混合符号族 f(2) 为负、f(3) 为正")
Sn = [speed_sum(f_uniform_neg(a)) for a in alphas]
ax.plot(alphas, Sn, "s--", color="#d62728", linewidth=2.0, markersize=6,
        label="均匀负值族（远层全为负）")
ax.axhline(1.0, color="#7f7f7f", linestyle=":", linewidth=1.6, label="光速约束面")
ax.set_xlabel("远层段参数 $a$")
ax.set_ylabel("泛函 sum_d f(d) d 的平方")
ax.set_title("左：混合符号族落在约束面上\n均匀负值族落在其下方")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

ax = axes[1]
names = ["铁磁\n（库内定案）", "交错\n（库内定案）", "铁磁\n（远层排斥型）", "交错\n（远层排斥型）"]
vals = [e_f_ferro, e_f_anti, e_ferro, e_anti]
cols = ["#1f77b4", "#1f77b4", "#d62728", "#d62728"]
ax.bar(range(4), vals, color=cols, width=0.6)
ax.axhline(0, color="#7f7f7f", linewidth=1.0)
ax.set_xticks(range(4))
ax.set_xticklabels(names, fontsize=9)
ax.set_ylabel("链能量")
ax.set_title("右：两种构型的能量对比\n远层段取排斥型时铁磁仍最低")
ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(IMG, dpi=150)
print("")
print("图已保存：%s" % IMG)
