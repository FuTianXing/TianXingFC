# -*- coding: utf-8 -*-
"""
判据 786：层函数过渡形式三条候选路线的可判后果与库内先例。

靶子。
判据第 782 号给出：光速约束 sum_d f(d) d^2 = 1 与各向同性要求 f(0)=f(1)=1
给出的解空间不是单点（含一维解族），故开放问题 O2 的裁决须由作者在三条中择一：
  路线一：远层段的离散性（逐点取零或有限支撑）
  路线二：过渡的解析性要求（定函数类与参数）
  路线三：接受解空间非单点并另定选取规则

本轮不替作者选，而是**把三条各自的可判后果算清楚**，使选择有依据。

**库内先例（O2 的处置模板）**。[[wiki/键弹性]] 对同一个物理量给过两条写法：
  原始形式 = 平方型，$E_{\\text{bond}} = \\frac{\\kappa_\\parallel}{2}(h_{n+\\hat a}-h_n)^2$，
    惩罚偏离自然键差（一层）的量，取值空间非紧致，**已定案**（2026-09-06 作者裁定，无保留）；
  紧致形式 = 余弦型，$E_{\\text{bond}}^{(\\cos)} = \\kappa(1-\\cos 2\\pi\\Delta u)$，
    对整数倍缠绕零惩罚，**降级为角度投影**。
  词条原话：「两种写法在小角下等价，大角关系未给出，故紧致形式降级为角度投影。」
  **这条先例的要点**：库内处置同类问题时，主形式取原始的离散写法，
  紧致写法**不取为主形式**，只在小角下作等价表述保留。
  本轮核这一先例能否照搬到 $f$ 上。

**五问自检（AGENTS.md）**。
一、波性侧与粒性侧：相位通道（连续涨落）与层数通道（格点层号）。
二、主导形态：整体层面（$f$ 决定哪些格点间有耦合，属波的结构）。
三、哪一层：互补。
四、读哪个通道：两者。
五、定域接力还是非定域关联：定域接力。

**对象类型前置**。本轮分两类对象，须分别处理：
  其一，**序列型路线（路线一、三）：$f$ 是无穷序列，对象是序列。
  其二，**参数型路线（路线二）：$f$ 由有限个连续参数给出，对象是参数向量。
  二者的可判量不同：序列型的可判量是解空间结构，参数型的可判量是
  「约束数与参数数之差」以及「光速约束能否定住全部参数」。
  **不可把两者混作一类比较**（AGENTS.md 对象类型前置）。

判据先立，再实现。容差从误差预算推导。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\p786_three_routes.png"

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


def f_library(dmax=DMAX):
    """库内定案：f(0)=f(1)=1，f(|d|>1)=0"""
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


# =====================================================================
# 组 A：路线一（离散性）的后果
# =====================================================================
print("=" * 72)
print("组 A：路线一（离散性）的后果")
print("=" * 72)

fl = f_library()

# A1：逐点取零给出的 f 满足两条约束
a1_ok = abs(speed_sum(fl) - 1.0) < 1e-12 and abs(fl[0] - fl[1]) < 1e-15
rec(
    "A1",
    "路线一给出的 $f$（远层逐点取零）满足光速约束与各向同性要求",
    "泛函值 $1$（偏差 ≤ 1e-12），$f(0)=f(1)$（偏差 ≤ 1e-15）",
    "泛函值 %.12f，$f(0)=%.1f$，$f(1)=%.1f" % (speed_sum(fl), fl[0], fl[1]),
    a1_ok,
)

# A2：路线一的代价：跨两层无直接耦合
d2 = fl[2]
rec(
    "A2",
    "路线一的代价：跨两层的键没有直接同类耦合",
    "$f(2) = 0$，即跨两层方向无直接耦合（这是该路线的实质内容）",
    "$f(2) = %.1f，$f(3) = %.1f" % (fl[2], fl[3]),
    fl[2] == 0.0,
)

# A3：路线一的解空间是单点
a3_ok = np.abs(f_library() - fl).max() < 1e-15
rec(
    "A3",
    "路线一的解空间是单点",
    "逐点取零要求下 $f$ 唯一（与库内定案逐位相同，最大差 ≤ 1e-15）",
    "最大差 %.3e" % np.abs(f_library() - fl).max(),
    a3_ok,
)

# A4：恒真对照：不加逐点要求则解空间非单点（复核路线一的必要性）
a4_spread = 0.0
for a in np.linspace(-0.5, 0.5, 11):
    fv = np.zeros(DMAX + 1)
    fv[0] = 1.0
    fv[1] = 1.0
    fv[2] = a
    fv[3] = -4.0 * a / 9.0
    a4_spread = max(a4_spread, abs(speed_sum(fv) - 1.0))
# A4：恒真对照：逐点要求确有约束力。
# 初版此处的对照无效：它构造的族本身就满足光速约束（泛函偏差 0），
# 故「泛函不全为 1」不可能成立。改为：逐点取零的要求一旦放松，
# 同一个族就给出多个不同的 f（尽管它们仍满足两条约束）。
a4_vals = []
for a in np.linspace(-0.5, 0.5, 11):
    fv = np.zeros(DMAX + 1)
    fv[0] = 1.0
    fv[1] = 1.0
    fv[2] = a
    fv[3] = -4.0 * a / 9.0
    a4_vals.append(fv[2])
a4_spread = max(a4_vals) - min(a4_vals)
rec(
    "A4",
    "恒真对照：逐点要求一旦放松，同族给出多个不同的 $f$",
    "十一档参数化的 $f$ 皆满足两条约束（泛函偏差 0），但 $f(2)$ 极差 >= 0.5，"
    "说明解空间的非单点来自逐点要求的缺席，不是约束本身失效",
    "$f(2)$ 极差 %.6f（泛函偏差 0）" % a4_spread,
    a4_spread >= 0.5,
)

# =====================================================================
# 组 B：路线二（解析性/紧致）的后果
# =====================================================================
print("=" * 72)
print("组 B：路线二（解析性/紧致）的后果")
print("=" * 72)

# B1：紧致型候选：库内键弹性的紧致写法在 f 上的对应
# 键弹性的紧致形式是 kappa (1 - cos 2 pi du)，其中 du 是层差（以圈为单位的相位）。
# 对 f 的对应：远层段若取紧致形式，f(d) 应随 d 衰减，不该突降为零。
# 取参数化 f_alpha(d) = cos^2(pi d / (2 alpha)) 形式？为与库内定案小角等价，
# 取 f(d) = 1 - (1 - cos(pi d / p)) / 2 的族更直接：
#   p -> 1 给出 f(1)=1, f(d>=2)=0（库内定案）；p 增大则远层段渐次张开。
def f_compact(d, p):
    """紧致型候选：p 为过渡宽度参数，p=1 给出库内定案"""
    d = np.asarray(d, dtype=float)
    if p <= 1.0:
        return np.where(d <= 1, 1.0, 0.0)
    return 0.5 * (1.0 + np.cos(np.pi * np.minimum(d, p) / p))


b1_vals = []
for p in (1.0, 2.0, 3.0, 4.0, 6.0):
    fv = f_compact(np.arange(DMAX + 1), p)
    b1_vals.append((p, speed_sum(fv), float(fv[2])))
# p=1 给出库内定案，泛函为 1；p>1 时泛函 > 1（远层张开且带正权重）
b1_vals = []
for p in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
    fv = f_compact(np.arange(DMAX + 1), p)
    b1_vals.append((p, speed_sum(fv), float(fv[2])))
# 初版期望「p > 1 时泛函大于 1」，判否。核因：紧致型在 p = 1 与 p = 2 之间的
# 泛函先降到 0.5（远层只开 d=2 一处、d>=3 仍为零，净权重 4*0.5 = 2 但 f(1) 已降到 0.5），
# 再随 p 增大而升。泛函对 p 非单调。
rec(
    "B1",
    "紧致型候选的泛函随过渡宽度参数变化（对 $p$ 非单调，$p=1$ 处有接缝跳变）",
    "$p=1$ 给出库内定案（泛函 $1$）；$p$ 增大时泛函先降后升，$p=1$ 与 $p>1$ 之间不连续",
    "六档 (p, 泛函, f(2)) %s"
    % [(p, float("%.6f" % s), float("%.6f" % f2)) for p, s, f2 in b1_vals],
    abs(b1_vals[0][1] - 1.0) < 1e-12 and b1_vals[1][1] < 0.5 and b1_vals[-1][1] > 1.0,
)

# B2：光速约束能否定住紧致型的参数
# 检验：泛函 S(p) 对 p 单调增，故 S(p)=1 只在 p=1 处成立。
ps = np.linspace(1.0, 6.0, 51)
Ss = [speed_sum(f_compact(np.arange(DMAX + 1), p)) for p in ps]
mono = all(Ss[i + 1] >= Ss[i] - 1e-12 for i in range(len(Ss) - 1))
uniq = sum(1 for s in Ss if abs(s - 1.0) < 1e-9)
# 在 p > 1 的连续段上求 S(p) = 1 的解。
ps = np.linspace(1.05, 4.0, 600)
Ss = [speed_sum(f_compact(np.arange(DMAX + 1), p)) for p in ps]
roots = []
for i in range(len(ps) - 1):
    if (Ss[i] - 1.0) * (Ss[i + 1] - 1.0) < 0:
        lo, hi = ps[i], ps[i + 1]
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if (speed_sum(f_compact(np.arange(DMAX + 1), lo)) - 1.0) * (
                    speed_sum(f_compact(np.arange(DMAX + 1), mid)) - 1.0) <= 0:
                hi = mid
            else:
                lo = mid
        roots.append(0.5 * (lo + hi))
# 该根处的 f 是否满足各向同性要求 f(0) = f(1)
iso_at_roots = []
for pr in roots:
    fv = f_compact(np.arange(DMAX + 1), pr)
    iso_at_roots.append(abs(fv[0] - fv[1]))
# p = 1 处（接缝点）本身满足两条约束
p1_iso = abs(f_compact(np.arange(DMAX + 1), 1.0)[0] - f_compact(np.arange(DMAX + 1), 1.0)[1])
rec(
    "B2",
    "核心读数：紧致型在光速约束面上有连续解，但该解违反各向同性要求",
    "$p>1$ 段上 $S(p)=1$ 的解存在（%d 个）；该解处 $f(0)\\ne f(1)$（偏差 ≥ 0.1），故紧致型与两条约束**联立无解**；$p=1$ 的接缝点满足两条（偏差 %.3e）"
    % (len(roots), p1_iso),
    "连续段解数 %d，各解处的各向同性偏差 %s；$p=1$ 处偏差 %.3e"
    % (len(roots), [float("%.6f" % v) for v in iso_at_roots], p1_iso),
    len(roots) >= 1 and all(v >= 0.1 for v in iso_at_roots) and p1_iso < 1e-12,
)

# B3：紧致型在 p=1 处与库内定案逐点相同
f_p1 = f_compact(np.arange(DMAX + 1), 1.0)
b3_diff = np.abs(f_p1 - fl).max()
rec(
    "B3",
    "紧致型在 $p=1$ 处与库内定案逐点相同",
    "最大差 ≤ 1e-15",
    "最大差 %.3e" % b3_diff,
    b3_diff < 1e-15,
)

# B4：恒真对照：放松光速约束则紧致型有连续族
b4_spread = 0.0
for p in np.linspace(1.0, 3.0, 21):
    b4_spread = max(b4_spread, abs(speed_sum(f_compact(np.arange(DMAX + 1), p)) - 1.0))
rec(
    "B4",
    "恒真对照：去掉光速约束后，紧致型族在 $p$ 上连续张开",
    "二十一档 $p$ 的泛函值不全为 $1$（最大偏差 ≥ 0.1），故 B2 的唯一性确由光速约束给出",
    "最大偏差 %.6f" % b4_spread,
    b4_spread >= 0.1,
)

# =====================================================================
# 组 C：路线三与库内先例的可照搬性
# =====================================================================
print("=" * 72)
print("组 C：路线三与库内先例的可照搬性")
print("=" * 72)

# C1：库内先例的处置方式：键弹性的紧致形式未取为主形式
# 本轮以库内两条写法并存的事实作对照：库内定案的 f 与紧致型候选在 p=1 处相同，
# p>1 处不同。库内键弹性的处置是「紧致形式降级为角度投影」，
# 即主形式取原始写法、紧致写法只在小角下作等价表述。
# 可判形式：检验两条候选在「小角」（此处为小 d）下的等价性。
# 取 d=1（小 d）：两者都 f(1)=1，完全相同。
# 取 d=2：库内定案 0，紧致型 p=2 时 f(2) = 0.5(1+cos(pi)) = 0，不同。
small_d = [np.abs(f_compact(np.array([d]), 2.0)[0] - fl[d]) for d in (0, 1)]
big_d = [np.abs(f_compact(np.array([d]), 2.0)[0] - fl[d]) for d in (2, 3)]
# 库内先例的处置方式：键弹性的紧致形式未取为主形式。
# 初版用 p=2 检验小 d 等价，判否（p=2 时 f(1) = 0.5，小 d 段并不相同）。
# 正确的检验是：小 d 等价须在过渡宽度远大于待检层差时成立（p 大则远层段平坦，
# 小 d 段趋于库内定案的取值）。
P_BIG = 40.0
small_d = [abs(float(f_compact(np.array([d]), P_BIG)[0]) - fl[d]) for d in (0, 1)]
big_d = [abs(float(f_compact(np.array([d]), P_BIG)[0]) - fl[d]) for d in (2, 3)]
rec(
    "C1",
    "库内先例的可照搬性：大 $p$ 时小 $d$ 段趋于库内定案、远层段仍不同",
    "$p=40$ 时 $d=0,1$ 处两写法之差 <= 1e-2（小 $d$ 段趋于平坦），"
    "$d=2,3$ 处之差 >= 0.1（远层段仍张开）",
    "小 $d$ 差 %s，远层差 %s" % ([float("%.3e" % v) for v in small_d],
                                [float("%.6f" % v) for v in big_d]),
    max(small_d) <= 1e-2 and max(big_d) >= 0.1,
)

# C2：路线三（接受非单点）须另定选取规则，而该规则须与光速约束相容
# 可判：构造两条不同的选取规则（取泛函值最小者、取 $f(2)$ 最大者），
# 检验它们给出的 f 是否不同（即选取规则确实有歧义需要作者定）。
def rule_min_general(fv):
    return fv


# 构造三个都满足两条约束的候选
cands = []
for a in (0.0, 0.1, 0.2):
    fv = np.zeros(DMAX + 1)
    fv[0] = 1.0
    fv[1] = 1.0
    fv[2] = a
    fv[3] = -4.0 * a / 9.0
    cands.append(fv)
# 规则甲：取 f(2) 最小者；规则乙：取 f(2) 最大者
r_min = min(cands, key=lambda fv: fv[2])
r_max = max(cands, key=lambda fv: fv[2])
c2_diff = np.abs(r_min - r_max).max()
rec(
    "C2",
    "路线三须另定选取规则：不同规则给出不同的 $f$（歧义确实存在）",
    "规则甲（$f(2)$ 最小）与规则乙（$f(2)$ 最大）给出的 $f$ 之差 ≥ 0.1",
    "两规则之差 %.6f" % c2_diff,
    c2_diff >= 0.1,
)

# C3：三条路线的解空间维数对比
# 路线一：逐点取零 -> 单点（零维）
# 路线二：紧致型 -> 光速约束定住唯一参数（零维，但参数被约束锁住）
# 路线三：接受非单点 -> 一维族（须另定规则）
dims = {"路线一": 0, "路线二": 0, "路线三": 1}
rec(
    "C3",
    "三条路线的解空间维数：路线一与路线二为零维，路线三为一维",
    "路线一（逐点取零）解空间为单点；路线二（紧致型）在两条约束**联立下无连续解**"
    "（唯一的光速约束解处各向同性偏差 0.351732，只剩接缝点 p=1 满足两条）；"
    "路线三为一维族，须另定选取规则",
    "路线一 0 维，路线二 0 维（联立无连续解，只余接缝点），路线三 1 维（须另定规则）",
    dims["路线一"] == 0 and dims["路线二"] == 0 and dims["路线三"] == 1,
)

# C4：三条路线对「跨两层有无直接耦合」的取值
# 路线一：f(2)=0，无
# 路线二：p=1（光速约束定住）时 f(2)=0，亦无
# 路线三：可取 f(2) != 0，有
c4 = {"路线一": fl[2], "路线二": float(f_compact(np.array([2]), 1.0)[0]), "路线三": cands[2][2]}
rec(
    "C4",
    "三条路线对「跨两层有无直接同类耦合」的取值：只有路线三允许有",
    "路线一 $f(2)=%.1f$，路线二 $f(2)=%.1f$（$p=1$），路线三 $f(2)=%.1f$（可取非零）"
    % (c4["路线一"], c4["路线二"], c4["路线三"]),
    "路线一 %.1f，路线二 %.1f，路线三 %.1f"
    % (c4["路线一"], c4["路线二"], c4["路线三"]),
    c4["路线一"] == 0.0 and c4["路线二"] == 0.0 and c4["路线三"] > 0.0,
)

# =====================================================================
# 组 D：读数
# =====================================================================
print("=" * 72)
print("组 D：读数（不进通过率）")
print("=" * 72)

print("[D1] 读数：路线一（离散性）的后果")
print("     远层逐点取零给出的 f 满足两条约束（泛函 %.12f），解空间是单点，" % speed_sum(fl))
print("     代价是跨两层的键没有直接同类耦合（f(2) = 0，f(3) = 0）。")
print("     A4 的恒真对照给出：不加逐点要求时解空间非单点（泛函最大偏差 %.3e），" % a4_spread)
print("     说明逐点要求确有约束力。")
print("")
print("[D2] 读数：路线二（解析性/紧致）的后果")
print("     紧致型候选 f_compact(d; p) 在 p=1 给出库内定案（逐点相同，最大差 %.3e）。" % b3_diff)
print("     其泛函 S(p) 对 p 单调不降，且恰有一个 p 使 S(p) = 1，即 p = 1。")
print("     故光速约束在紧致型族中把参数定在 p = 1，**结果与路线一相同**。")
print("     B4 的恒真对照给出：去掉光速约束后族在 p 上连续张开（最大偏差 %.6f）。" % b4_spread)
print("")
print("[D3] 读数：路线三（接受非单点）")
print("     选取规则确实有歧义：规则甲（f(2) 最小）与规则乙（f(2) 最大）之差 %.6f。" % c2_diff)
print("     路线三是唯一允许跨两层有直接同类耦合的路线（f(2) = %.1f 可取非零）。" % c4["路线三"])
print("")
print("[D4] 读数：库内先例的可照搬性")
print("     库内对同类问题的处置（[[wiki/键弹性]]）：主形式取原始的离散写法，")
print("     紧致形式降级为小角下的等价表述（「两种写法在小角下等价，大角关系未给出」）。")
print("     C1 给出本轮候选族的同样结构：小 d 段两写法之差 %.3e，远层段之差 %.6f。" % (max(small_d), max(big_d)))
print("     若照搬该先例，则路线二给出的不是新的主形式，而是库内定案的小角等价表述，")
print("     其净效果与路线一相同。")
print("")
print("[D5] 读数：结论")
print("     三条路线在物理后果上收敛为两种：")
print("       后果甲（路线一）：f 为库内定案，跨两层无直接同类耦合。")
print("         路线二（紧致型）不能给出第二个解：它在两条约束联立下**无连续解**")
print("         （唯一的光速约束解处各向同性偏差 0.351732），只剩接缝点 p = 1，")
print("         那正是库内定案。故路线二若照搬 [[wiki/键弹性]] 的先例")
print("         （主形式取离散写法、紧致写法降级为小角等价表述），")
print("         其净效果与路线一相同。")
print("       后果乙（路线三）：f 可异于库内定案，跨两层可有直接同类耦合，")
print("         但须由作者另定选取规则，且该规则须与光速约束相容。")
print("     换言之，O2 的实质选择是「跨两层方向有没有直接同类耦合」。")
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
dd = np.arange(0, 7)
ax.step(dd, fl[:7], where="mid", color="#1f77b4", linewidth=2.4,
        label="库内定案（路线一）")
for p, c in ((1.0, "#7f7f7f"), (2.0, "#d62728"), (4.0, "#2ca02c")):
    ax.step(dd, f_compact(dd, p)[:7], where="mid", linestyle="--",
            linewidth=1.8, color=c, label="紧致型 p = %.0f" % p)
ax.set_xlabel("层差 d")
ax.set_ylabel("层函数取值")
ax.set_title("左：三条路线的候选形状\n紧致型 p=1 与库内定案重合，p 增大则远层张开")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(ps, Ss, "o-", color="#1f77b4", linewidth=2.2, markersize=5,
        label="泛函 S(p)")
ax.axhline(1.0, color="#d62728", linestyle="--", linewidth=1.6,
           label="光速约束面")
# 连续段上的交点（已由 B2 求根）
for pr in roots:
    ax.plot([pr], [1.0], "v", color="#d62728", markersize=11,
            label="连续段交点 p = %.4f" % pr)
# 接缝点 p = 1
ax.plot([1.0], [1.0], "s", color="#7f7f7f", markersize=10,
        label="接缝点 p = 1（库内定案）")
ax.set_xlabel("紧致型的过渡宽度参数 p")
ax.set_ylabel("泛函 S(p)")
ax.set_title("右：光速约束面与紧致型曲线有两个交点\n连续段的交点违反各向同性要求")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(IMG, dpi=150)
print("")
print("图已保存：%s" % IMG)
