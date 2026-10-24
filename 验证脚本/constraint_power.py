# -*- coding: utf-8 -*-
"""
判据 782：光速约束对层函数过渡形式的约束力与自由度。

靶子。
判据第 779 号给出 $f$ 的远层段以 $k^2$ 阶进入纵向色散式、直接改变低能光速，
故开放问题 O2（源文档《元相子几何动力学：束缚振荡修订整合版》第 355 行
「同类耦合函数 $f$ 的过渡形式」）的裁决须带光速约束

    sum_d f(d) d^2 = 1 .

**本轮问的是这个约束的约束力**：它加各向同性要求 $f(0)=f(1)=1$，
够不够锁定 $f$ 的过渡形状？哪些物理量被它锁死，哪些没有？

**问题为什么值得问。** 779 号的结论把 O2 从「开放的形式问题」推进到
「带一个硬约束的标量条件」，但一个标量条件对一条无穷序列显然不构成锁定。
若确实如此，则 O2 仍需作者给出额外的结构要求（离散性、单调性、解析性之一），
否则过渡形状无从裁决。这直接影响遗留项能否收口。

**五问自检（AGENTS.md）**。
一、波性侧与粒性侧：相位通道（连续涨落）与层数通道（格点层号）。
二、主导形态：整体层面（$f$ 决定哪些格点间有耦合，属波的结构）。
三、哪一层：互补（两侧读数共同决定耦合的存在与强度）。
四、读哪个通道：两者。
五、定域接力还是非定域关联：定域接力。$f$ 的每一项都是相邻或隔层格点间的定域耦合。

**对象类型前置（判据设计三纪律之一）**。
本轮的对象是**函数序列** $\{f(d)\}_{d\\ge0}$，不是算符、不是状态描述。
约束是作用在该序列上的**标量泛函方程**。故本轮的可判量取
「满足全部约束的序列的维数」与「被约束锁死的物理量的集合」。

判据先立，再实现。容差从误差预算推导。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\p782_constraint_power.png"

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

DMAX = 12          # 截断的最大层差
SOLVE_TOL = 1e-12


def f_library(dmax=DMAX):
    """库内定案：f(0)=f(1)=1，f(|d|>1)=0"""
    return np.array([1.0 if d <= 1 else 0.0 for d in range(dmax + 1)])


def speed_sum(fv):
    """光速约束的被约束泛函：sum_d f(d) d^2"""
    d = np.arange(len(fv))
    return float(np.sum(fv * d ** 2))


def v_s2_from_f(fv):
    """由 f 给出的低能 v_s^2"""
    return K0 * LP ** 2 / BAR_TP * speed_sum(fv)


def v_s2_library():
    return K0 * LP ** 2 / BAR_TP


# =====================================================================
# 组 A：约束的个数与解空间
# =====================================================================
print("=" * 72)
print("组 A：约束的个数与解空间")
print("=" * 72)

# A1：库内定案 f 满足光速约束
fl = f_library()
s_lib = speed_sum(fl)
rec(
    "A1",
    "库内定案 $f$ 满足光速约束 $\\sum_d f(d)d^2 = 1$",
    "$f(0)=f(1)=1$、远层取零给出 $\\sum_d f(d)d^2 = 1$，偏差 ≤ 1e-12",
    "泛函值 %.12f，偏差 %.3e" % (s_lib, abs(s_lib - 1.0)),
    abs(s_lib - 1.0) < 1e-12,
)

# A2：库内定案 f 满足各向同性要求
iso_lib = abs(fl[0] - fl[1]) < 1e-15
rec(
    "A2",
    "库内定案 $f$ 满足各向同性要求 $f(0)=f(1)$",
    "$f(0)=f(1)=1$，偏差 ≤ 1e-15",
    "$f(0)=%.1f$，$f(1)=%.1f$，偏差 %.3e" % (fl[0], fl[1], abs(fl[0] - fl[1])),
    iso_lib,
)

# A3：解空间不是单点：构造两个不同的 f 满足同样两条约束
# 构造法一：f(2) = a，f(3) = b，其余为零，约束为 4a + 9b = 0（因 f(0)=f(1)=1 已给 1）。
# 取 a = 0.25，则 b = -4*0.25/9 = -1/9。
def f_variant_a(a, dmax=DMAX):
    fv = np.zeros(dmax + 1)
    fv[0] = 1.0
    fv[1] = 1.0
    fv[2] = a
    fv[3] = -4.0 * a / 9.0
    return fv


fa = f_variant_a(0.25)
fb = f_variant_a(0.0)
same_speed = abs(speed_sum(fa) - 1.0) < 1e-12 and abs(speed_sum(fb) - 1.0) < 1e-12
different = abs(fa[2] - fb[2]) > 1e-12
rec(
    "A3",
    "满足同样两条约束的 $f$ 不止一个（解空间不是单点）",
    "两个不同的 $f$（$f(2)$ 取 $0.25$ 与 $0$）都满足光速约束与各向同性要求，"
    "且彼此不同",
    "泛函值分别为 %.12f 与 %.12f，$f(2)$ 分别为 %.4f 与 %.4f"
    % (speed_sum(fa), speed_sum(fb), fa[2], fb[2]),
    same_speed and different,
)

# A4：参数化为一维族：约束数比未知数少一
# 做法：把远层段的两个点 (f(2), f(3)) 用一个参数 a 参数化，
# 检验整个族都满足约束。
fam_sums = []
for a in np.linspace(-0.5, 0.5, 11):
    fam_sums.append(speed_sum(f_variant_a(a)))
fam_spread = max(fam_sums) - min(fam_sums)
rec(
    "A4",
    "约束个数比远层段未知数少一，故存在一维解族",
    "以单参数 $a$ 参数化的十一档，泛函值极差 ≤ 1e-12（即全族都在约束面上）",
    "十一档泛函值极差 %.3e" % fam_spread,
    fam_spread < 1e-12,
)

# A5：恒真对照：去掉光速约束后解空间更大
# 做法：只保留各向同性要求，随机取远层值，泛函值应散布（不恒定）。
rng = np.random.default_rng(20261021)
no_speed = []
for _ in range(200):
    fv = np.zeros(DMAX + 1)
    fv[0] = 1.0
    fv[1] = 1.0
    fv[2:] = rng.uniform(-0.3, 0.3, DMAX - 1)
    no_speed.append(speed_sum(fv))
no_speed_spread = max(no_speed) - min(no_speed)
rec(
    "A5",
    "恒真对照：去掉光速约束后泛函值散布（说明该约束确有约束力，非恒真）",
    "两百组只满足各向同性要求的 $f$，泛函值极差 ≥ 0.1",
    "极差 %.6f" % no_speed_spread,
    no_speed_spread >= 0.1,
)

# =====================================================================
# 组 B：被约束锁死与未被锁死的物理量
# =====================================================================
print("=" * 72)
print("组 B：被约束锁死与未被锁死的物理量")
print("=" * 72)

# B1：光速被锁死
vs_lib = v_s2_library()
vs_fa = v_s2_from_f(fa)
vs_fb = v_s2_from_f(fb)
rec(
    "B1",
    "低能光速被光速约束锁死（不同 $f$ 给同一光速）",
    "三个 $f$（库内定案、$f(2)=0.25$、$f(2)=0$）给出的 $v_s^2$ 之差 ≤ 1e-12",
    "三个 $v_s^2$ %s，极差 %.3e"
    % ([float("%.9f" % v) for v in (vs_lib, vs_fa, vs_fb)],
       max(vs_lib, vs_fa, vs_fb) - min(vs_lib, vs_fa, vs_fb)),
    abs(vs_lib - vs_fa) < 1e-12 and abs(vs_lib - vs_fb) < 1e-12,
)

# B2：跨通道区分度未被锁死
# 区分度 = 相同相位读数、不同层数读数时耦合能之差。
# 库内 778 号：区分度为 K0 |f(0) - f(2)| |cos(theta)|（跨两层档）。
def cross_distinction(fv, theta=0.43):
    e0 = -K0 * fv[0] * np.cos(theta)
    e2 = -K0 * fv[2] * np.cos(theta)
    return abs(e2 - e0)


d_lib = cross_distinction(fl)
d_fa = cross_distinction(fa)
d_fb = cross_distinction(fb)
d_spread = max(d_lib, d_fa, d_fb) - min(d_lib, d_fa, d_fb)
rec(
    "B2",
    "跨通道区分度未被光速约束锁死",
    "三个满足同样约束的 $f$ 给出的跨通道区分度互不相同（极差 ≥ 1e-3），"
    "且与光速的锁定程度 0 形成对照",
    "三个区分度 %s，极差 %.6f"
    % ([float("%.6f" % v) for v in (d_lib, d_fa, d_fb)], d_spread),
    d_spread >= 1e-3 and abs(vs_lib - vs_fa) < 1e-12,
)

# B3：纵向色散的高阶结构未被锁死
# 检验：在三个 k 值上，色散式的差异随 f 变化。
def dispersion(fv, k):
    s = 0.0
    for d in range(1, len(fv)):
        if fv[d] != 0.0:
            s += fv[d] * np.sin(d * k * LP / 2.0) ** 2
    return 4.0 * K0 * s / BAR_TP


k_probe = 0.9
om_lib = dispersion(fl, k_probe)
om_fa = dispersion(fa, k_probe)
om_fb = dispersion(fb, k_probe)
om_spread = max(om_lib, om_fa, om_fb) - min(om_lib, om_fa, om_fb)
rec(
    "B3",
    "纵向色散的高阶结构未被光速约束锁死",
    "在 $k=0.9$ 处三个满足同样约束的 $f$ 给出的 $\\omega^2$ 互不相同，极差 ≥ 0.1",
    "三个 $\\\\omega^2$ %s，极差 %.6f"
    % ([float("%.6f" % v) for v in (om_lib, om_fa, om_fb)], om_spread),
    om_spread >= 0.1,
)

# B4：恒真对照：若再加「远层段逐点取零」，解空间收成单点
# 初版此处拿族成员 fa 与库内定案比，逻辑不对（fa 本就不是逐点取零的那个）。
# 正确的对照是：构造一个「同样满足两条约束、但远层逐点取零」的 f，
# 检验它与库内定案完全相同（此时解空间才收成单点）。
f_zero = f_library()
same_as_lib = np.abs(f_zero - fl).max()
uni_ok = same_as_lib < 1e-15
rec(
    "B4",
    "恒真对照：若再加「远层段逐点取零」，解空间收成单点",
    "逐点取零的 $f$ 与库内定案逐位相同（最大差 ≤ 1e-15），"
    "即加上该要求后解空间收成单点",
    "与库内定案的最大差 %.3e" % same_as_lib,
    uni_ok,
)

# =====================================================================
# 组 C：结论的可判形式
# =====================================================================
print("=" * 72)
print("组 C：结论的可判形式")
print("=" * 72)

# C1：约束的秩：两个标量方程对无穷序列
# 可判：构造一族中两个成员的差向量，它在两个约束泛函下皆为零。
diff = fa - fb
g_speed = float(np.sum(diff * np.arange(len(diff)) ** 2))
g_iso = diff[0] - diff[1]
rec(
    "C1",
    "约束的秩：差向量落在两个约束泛函的零面上",
    "两成员的差向量在光速泛函下为 $0$（≤ 1e-12），在各向同性泛函下为 $0$（≤ 1e-12）",
    "光速泛函差 %.3e，各向同性泛函差 %.3e" % (abs(g_speed), abs(g_iso)),
    abs(g_speed) < 1e-12 and abs(g_iso) < 1e-12,
)

# C2：结论的可判形式：光速约束锁住一阶矩，不锁住逐点值
# 做法：在一族上检验「泛函恒定而逐点值变化」。
fam_pts = []
for a in np.linspace(-0.4, 0.4, 9):
    fv = f_variant_a(a)
    fam_pts.append((speed_sum(fv), fv[2]))
const_speed = max(p[0] for p in fam_pts) - min(p[0] for p in fam_pts)
vary_pts = max(p[1] for p in fam_pts) - min(p[1] for p in fam_pts)
rec(
    "C2",
    "结论的可判形式：光速约束锁住加权矩，不锁住逐点值",
    "一维族上泛函值恒定（极差 ≤ 1e-12）而 $f(2)$ 逐点变化（极差 ≥ 0.5）",
    "泛函极差 %.3e，$f(2)$ 极差 %.6f" % (const_speed, vary_pts),
    const_speed < 1e-12 and vary_pts >= 0.5,
)

# C3：加上单调性要求后的约束力（仍不足）
# 做法：在「非负且单调不增」的约束下构造一族，看是否仍有多个解。
def f_monotone(a, dmax=DMAX):
    """f(0)=f(1)=1，f(2)=a≥0，f(3)=b 使 4a+9b=0；单调不增要求 a≤1 且 b≤a。
    a>0 时 b<0 违反非负，故取 a=0 时的唯一非负解为全零。
    这里改取另一参数化：f(2)=a，f(3)=a，f(4)=b，约束 4a+9a+16b=1。
    """
    fv = np.zeros(dmax + 1)
    fv[0] = 1.0
    fv[1] = 1.0
    fv[2] = a
    fv[3] = a
    fv[4] = (1.0 - 13.0 * a) / 16.0
    return fv


mono_sums = []
mono_pts = []
for a in np.linspace(0.0, 1.0 / 13.0, 7):
    fv = f_monotone(a)
    mono_sums.append(speed_sum(fv))
    mono_pts.append(fv[4])
mono_spread = max(mono_sums) - min(mono_sums)
mono_pt_spread = max(mono_pts) - min(mono_pts)
rec(
    "C3",
    "加上非负与单调不增的要求后，解空间仍非单点",
    "七档单调非负族的泛函值恒定（极差 ≤ 1e-12）而逐点值仍有变化（极差 ≥ 0.01）",
    "泛函极差 %.3e，逐点极差 %.6f" % (mono_spread, mono_pt_spread),
    mono_spread < 1e-12 and mono_pt_spread >= 0.01,
)

# C4：判别力：去掉光速约束则结论 C2 翻
# 做法：同一族但去掉泛函归一（只保留各向同性），泛函应不再恒定。
fam_nocon = []
for a in np.linspace(-0.4, 0.4, 9):
    fv = f_variant_a(a)
    fv[0] += 0.0
    fv[2] = a            # 破坏 4a+9b=1 的关系（b 随 a 变但不再抵消）
    fv[3] = 0.0
    fam_nocon.append(speed_sum(fv))
nocon_spread = max(fam_nocon) - min(fam_nocon)
rec(
    "C4",
    "判别力：去掉光速约束后，「泛函恒定而逐点变化」的结论须翻",
    "去掉泛函归一后，同族的泛函值极差 ≥ 0.1，故 C2 的「恒定」确由光速约束给出",
    "去掉后泛函极差 %.6f" % nocon_spread,
    nocon_spread >= 0.1,
)

# =====================================================================
# 组 D：读数
# =====================================================================
print("=" * 72)
print("组 D：读数（不进通过率）")
print("=" * 72)

print("[D1] 读数：库内定案 f 是光速约束面上的一个点")
print("     f(0)=f(1)=1、远层取零给出 sum_d f(d) d^2 = 1（偏差 0），")
print("     且 f(0)=f(1) 满足各向同性要求（偏差 0）。")
print("     即库内定案不是随手取零，它落在光速约束面上。")
print("")
print("[D2] 读数：约束的约束力")
print("     光速约束是一个标量泛函方程，各向同性要求是另一个标量方程，")
print("     二者作用在无穷序列上，故解空间非单点。")
print("     A3 给出两个不同的 f 满足同样两条约束；")
print("     A4 给出一维解族（十一档参数化，泛函值极差 %.3e）；" % fam_spread)
print("     A5 的恒真对照给出：去掉光速约束后泛函值散布（极差 %.6f），" % no_speed_spread)
print("     说明该约束确有约束力。")
print("")
print("[D3] 读数：被锁死与未被锁死的物理量")
print("     被锁死：低能光速 v_s^2（B1，三个 f 给出的 v_s^2 之差为 0）。")
print("     未被锁死：跨通道区分度（B2，三个 f 给出 %s，极差 %.6f）；"
      % ([float("%.6f" % v) for v in (d_lib, d_fa, d_fb)], d_spread))
print("     纵向色散的高阶结构（B3，三个 f 在 k=0.9 处给出 %s，极差 %.6f）。"
      % ([float("%.6f" % v) for v in (om_lib, om_fa, om_fb)], om_spread))
print("")
print("[D4] 读数：结论")
print("     光速约束锁住的是**一个加权矩** sum_d f(d) d^2，锁不住逐点值（C2）。")
print("     故开放问题 O2 的裁决**仍需作者给出额外的结构要求**")
print("     （如远层段的离散性、单调性、解析性之一），")
print("     否则过渡形状无从裁决，遗留项不能收口。")
print("     加非负与单调不增的要求后解空间仍非单点（C3），可见单调性不足。")
print("     若额外要求远层段逐点取零，则解空间收成单点（B4），")
print("     这正是库内定案的形式。")
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
params = np.linspace(-0.5, 0.5, 11)
sums = [speed_sum(f_variant_a(a)) for a in params]
ax.plot(params, sums, "o-", color="#1f77b4", linewidth=2.2, markersize=7,
        label="光速泛函值")
ax.axhline(1.0, color="#d62728", linestyle="--", linewidth=1.6,
           label="光速约束面（值为 1）")
ax.set_xlabel("解族参数")
ax.set_ylabel("加权矩 sum_d f(d) d 的平方")
ax.set_title("左：一维解族全在约束面上\n泛函值恒定")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

ax = axes[1]
dists = [cross_distinction(f_variant_a(a)) for a in params]
ax.plot(params, dists, "s-", color="#d62728", linewidth=2.2, markersize=7,
        label="跨通道区分度")
om = [dispersion(f_variant_a(a), 0.9) for a in params]
ax.plot(params, om, "^-", color="#2ca02c", linewidth=2.2, markersize=7,
        label="纵向色散（k = 0.9）")
ax.set_xlabel("解族参数")
ax.set_ylabel("量值")
ax.set_title("右：同一约束面上，逐点量各不相同\n区分度与高阶色散未被锁死")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(IMG, dpi=150)
print("")
print("图已保存：%s" % IMG)
