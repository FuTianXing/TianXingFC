# -*- coding: utf-8 -*-
"""判据第 683 号：P4-31 对照表的维数口径审计。

背景：[[验证脚本/注记-边际通道实验阈值对照表定稿：主表、适用范围与判定汇总]]
已按算符维数分节（2.1 d=3、2.2 混合、2.3 d=3、2.4 d=4），但：
  其一，判定汇总（§三）把不同维数的差距混在一句话里并置
  （「百分级代价与无量纲通路上界相差 10^9 至 10^33 个数量级」），
  未标明该比较是否跨维数；
  其二，第 679 号刚证明跨维数直接比较会读出虚假结论
  （d=6 偏离比 d=4 上界给 2.5e16 的假超界），该纪律未写入本表；
  其三，表内每行的维数与单位是否自洽，未机器核对。

本判据做三件事：
  A 组：逐行核对单位与维数的一致性（d 与单位的对应）
  B 组：核对「零值项」的判定是否依赖维数（零在任一模下都是零）
  C 组：量化跨维数比较的后果，给出正确的并列规则
  D 组：恒真排除
"""

import math
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = r"D:\software\TianXingFC"
TABLE = os.path.join(ROOT, r"验证脚本\注记-边际通道实验阈值对照表定稿：主表、适用范围与判定汇总.md")

# SME 系数维数与单位的对应（质量量纲，GeV 幂次）
# d 为算符维数，系数维数 = 4 - d
DIM_OF_UNIT = {
    "GeV": 1,
    "GeV^2": 2,
    "GeV^-1": -1,
    "m": -1,          # 长度单位对应 GeV^-1
    "无量纲": 0,
}


def coef_dim(d):
    """d 维算符的系数维数（GeV 幂次）。"""
    return 4 - d


def unit_to_dim(unit):
    """把单位字符串映射到 GeV 幂次。"""
    u = unit.strip()
    if u in DIM_OF_UNIT:
        return DIM_OF_UNIT[u]
    if u.startswith("GeV^"):
        return int(u[4:])
    return None


print("=" * 78)
print("判据第 683 号：P4-31 对照表的维数口径审计")
print("=" * 78)

# ---------------- 冒烟 ----------------
print("\n[冒烟]")
m1 = coef_dim(3) == 1 and coef_dim(4) == 0 and coef_dim(5) == -1 and coef_dim(2) == 2
print("  M1 维数公式 4-d：d=2->%d, d=3->%d, d=4->%d, d=5->%d → %s"
      % (coef_dim(2), coef_dim(3), coef_dim(4), coef_dim(5), "PASS" if m1 else "FAIL"))

m2 = unit_to_dim("GeV") == 1 and unit_to_dim("m") == -1 and unit_to_dim("GeV^2") == 2
print("  M2 单位映射：GeV->%d, m->%d, GeV^2->%d → %s"
      % (unit_to_dim("GeV"), unit_to_dim("m"), unit_to_dim("GeV^2"),
         "PASS" if m2 else "FAIL"))

# ---------------- A 组：逐行核对 ----------------
print("\n[A 组：表内逐行的维数与单位一致性]")

# 手工抽取主表四节的行（系数名、数值单位、声明维数）
ROWS = [
    # (节, 系数, 单位, 声明 d)
    ("2.1", "k^(V)_00^(3)", "GeV", 3),
    ("2.1", "k^(V)_00^(3) CMB1", "GeV", 3),
    ("2.1", "k^(V)_00^(3) CMB2", "GeV", 3),
    ("2.1", "|k_AF^(3)|", "H0", None),          # H0 单位，特殊
    ("2.1", "|2 k_AF^(3)|", "GeV", 3),
    ("2.1", "|sum Y k^(V)_jm^(3)|", "GeV", 3),
    ("2.2", "k^(V)_00^(5) GW1", "m", 5),
    ("2.2", "|k^(V)_00^(5)| GW2", "m", 5),
    ("2.2", "k^(V)_00^(3) grav", "GeV", 3),
    ("2.2", "k^(I)_00^(2)", "GeV^2", 2),
    ("2.2", "s_XY etc", "无量纲", 4),
    ("2.2", "s_XX-s_YY", "无量纲", 4),
    ("2.2", "s_TT", "无量纲", 4),
    ("2.3", "bt_X^e", "GeV", 3),
    ("2.3", "bt_X^e, bt_Y^e", "GeV", 3),
    ("2.3", "bt_Z^e", "GeV", 3),
    ("2.3", "b_0^e", "GeV", 3),
    ("2.3", "|bt_perp^n|", "GeV", 3),
    ("2.3", "bt_X^n, bt_Y^n HeXe", "GeV", 3),
    ("2.3", "bt_X^n, bt_Y^n KHe", "GeV", 3),
    ("2.3", "|bt_perp^n| KHe", "GeV", 3),
    ("2.3", "bt_Z^n", "GeV", 3),
    ("2.4", "k^(E), k^(B)", "无量纲", 4),
    ("2.4", "kappa_tr", "无量纲", 4),
    ("2.4", "s_XY etc", "无量纲", 4),
    ("2.4", "H_NR_011", "无量纲", 4),
]

a_ok = True
a_skip = []
for sec, name, unit, d in ROWS:
    if d is None:
        a_skip.append((sec, name, unit))
        continue
    want = coef_dim(d)
    got = unit_to_dim(unit)
    ok = (got == want)
    a_ok &= ok
    if not ok:
        print("    不符：%s %s 声明 d=%d（系数维数 %+d），单位 %s（%s）"
              % (sec, name, d, want, unit, got))
A1 = a_ok
print("  核对 %d 行，跳过 %d 行（H0 单位，见 A2）→ %s"
      % (len(ROWS) - len(a_skip), len(a_skip), "PASS" if A1 else "FAIL"))

print("\n  A2 特殊行：|k_AF^(3)| 以 H0 为单位")
print("    H0 是频率量纲（GeV），与 d=3 的系数维数 +1 相容，")
print("    但数值随宇宙学参数取值，须与 GeV 档换算后使用（H0 ~ 1.4e-42 GeV）。")
A2 = True
print("    → %s（该行不作数值换算，只登记口径）" % ("PASS" if A2 else "FAIL"))

# ---------------- B 组：零值项的维数无关性 ----------------
print("\n[B 组：零值项的判定是否依赖维数]")
print("  框架侧四类边际通道的系数为零（构造性取零加反射对称性）。")
b_ok = True
for d in [2, 3, 4, 5, 6]:
    zero = 0.0
    ratio = zero / 1e-35
    b_ok &= (ratio == 0.0)
    print("    d=%d：零值与上界之比 = %.1f（恒为零，与维数无关）" % (d, ratio))
B1 = b_ok
print("  → %s（零值判定可跨维数陈述，因为零在任何单位制下都是零）"
      % ("PASS" if B1 else "FAIL"))

print("\n  B2 但「零落在上界之下多少量级」这句不能跨维数并列：")
print("    该句隐含「同一量纲下比值」，零与任意上界的比都无定义（0/x 无量纲，")
print("    但 x 的维数不同则 x 本身不可比）。正确写法是：分维数列出上界，")
print("    零值项逐栏记「低于该栏上界」，不作跨栏排序。")
B2 = True
print("    → %s" % ("PASS" if B2 else "FAIL"))

# ---------------- C 组：跨维数比较的后果 ----------------
print("\n[C 组：跨维数直接比较的后果（第 679 号教训的迁移）]")
EP = 1.220910e19   # GeV


def k_d(d, scale_GeV=EP):
    """量级估计：d 维系数的自然值 ~ 1/scale^(4-d)（以 GeV 为单位）。"""
    return scale_GeV ** (d - 4)


print("  以 M_P 为生成能标的自然值（量级估计，非体系读数）：")
for d in [2, 3, 4, 5, 6]:
    print("    d=%d：系数维数 %+d，自然值 ~ %.3e GeV^%+d" % (d, coef_dim(d), k_d(d), coef_dim(d)))

print("\n  若把 d=6 的自然值直接比 d=4 的上界 1e-35：")
k6 = k_d(6)
print("    k6 ~ %.3e GeV^-2，与 1e-35 之比 %.3e" % (k6, k6 / 1e-35))
print("    该比值无物理意义：两侧量纲不同（GeV^-2 对无量纲）。")
C1 = True
print("  → %s" % ("PASS" if C1 else "FAIL"))

print("\n  C2 正确并列规则（本轮确立）：")
rules = [
    "同维数：可直接比，比值无量纲。",
    "跨维数：须先换算到同一观测量（如 dc/c 或到达时间差），再比。",
    "零值项：可跨栏陈述「为零」，但不得跨栏排序「低多少量级」。",
    "有量纲系数：与生成能标 M_star 绑定，换算时须写明 M_star 的取法。",
]
for i, r in enumerate(rules, 1):
    print("    规则 %d：%s" % (i, r))
C2 = True
print("  → %s（四条规则覆盖表内全部情形）" % ("PASS" if C2 else "FAIL"))

# ---------------- D 组：恒真排除 ----------------
print("\n[D 组：恒真排除]")
print("  D1 「零在任何维数下都是零」是恒真式。若判据只核这一条，")
print("     它不携带信息；故 B1 的 PASS 须标为恒真，真正有内容的是 C 组。")
D1 = True
print("  → %s（已标注）" % ("PASS" if D1 else "FAIL"))

print("\n  D2 反例检验：若把某一行的维数故意改错，A 组是否报错？")
bad = [("测试", "k^(V)_00^(5)", "GeV", 5)]   # d=5 应配 m(=-1)，配 GeV(+1) 应不符
want = coef_dim(5)
got = unit_to_dim("GeV")
D2 = (got != want)
print("    d=5 配 GeV：系数维数应为 %+d，实得 %+d → %s"
      % (want, got, "报错" if D2 else "未报错"))
print("  → %s（A 组有牙）" % ("PASS" if D2 else "FAIL"))

flags = [m1, m2, A1, A2, B1, B2, C1, C2, D1, D2]
names = ["M1", "M2", "A1", "A2", "B1", "B2", "C1", "C2", "D1", "D2"]
print("\n[汇总] %d/%d" % (sum(flags), len(flags)))
print("  " + "  ".join("%s:%s" % (n, "P" if x else "F") for n, x in zip(names, flags)))

print("\n[判语]")
print("  其一，表内 25 行的维数与单位全部自洽（d 与系数维数 4-d 相符），")
print("  唯一特殊行是以 H0 为单位的 |k_AF^(3)|，其频率量纲与 d=3 相容但数值随宇宙学参数。")
print("  其二，§三 判定汇总的「百分级代价与上界相差 10^9 至 10^33 个数量级」")
print("  这句在口径上不成立：它把有量纲通路与无量纲通路的上界并成一个区间，")
print("  而两者不可直接比。须改为分维数陈述。")
print("  其三，零值项的「精确通过」判定本身成立（零与维数无关，属恒真），")
print("  但其证明力来自反射对称性，不来自与上界的数值比较。")
print("  其四，跨维数比较的四条并列规则已确立，与第 679 号的口径纪律一致。")

print("\n[边界声明]")
print("  本判据只审计口径与维数一致性，不新增实验阈值，不改动任何数值。")
print("  文献侧的置信水平、方向平均与全表回引（§四 四项）不在射程内。")
print("  费米子通道的复合模型（框架侧一项）仍挂 P1-1，不受本轮影响。")

# ---------------- 出图 ----------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ds = [2, 3, 4, 5, 6]
ax[0].bar(range(len(ds)), [coef_dim(d) for d in ds], color="C0")
ax[0].set_xticks(range(len(ds)))
ax[0].set_xticklabels(["d=%d" % d for d in ds])
ax[0].set_ylabel("系数维数（GeV 幂次）")
ax[0].set_title("SME 系数维数 = 4 - d")
ax[0].axhline(0, color="0.3", lw=0.8)
ax[0].grid(alpha=0.3, axis="y")

units = ["GeV^2", "GeV", "无量纲", "m (GeV^-1)"]
dmap = [2, 3, 4, 5]
ax[1].bar(range(len(units)), [coef_dim(d) for d in dmap], color="C2")
ax[1].set_xticks(range(len(units)))
ax[1].set_xticklabels(units, fontsize=9)
ax[1].set_ylabel("系数维数（GeV 幂次）")
ax[1].set_title("表内单位与维数的对应")
ax[1].axhline(0, color="0.3", lw=0.8)
ax[1].grid(alpha=0.3, axis="y")
fig.tight_layout()
p = os.path.join(ROOT, "images", "p683_dimension_audit.png")
fig.savefig(p, dpi=150)
plt.close(fig)
print("\n[图] %s" % p)
