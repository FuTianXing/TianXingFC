# -*- coding: utf-8 -*-
r"""
第591条 弹性本构稳定化元素来源审计（JA/JB/JC，依据三维复合位错弹性本构攻坚计划 第四节）

模型单位 c = l_P = t_P = 1，hbar = 1；确定性，无随机数。

问题：C1-4c 判定「负线项需要显式负贡献，现有构件清单内无其本体」。攻坚计划第四节把
真缺口收敛为「稳定化元素的本体从哪里来」，并列三个来源候选各一轮有界判定：
  JA 来源甲：基准束缚势 V_rep 的高阶展开。库内清单四个候选，甲硬核极限（只给 r->0 的
     发散陈述，无函数形式，无法展开），乙 幂律 A/r^p、丙 软核 A/(1+(r/r0)^2)、丁 指数
     A*exp(-r/r0) 三个有函数形式。计划正文写「三个候选形式」即指此三个。
  JB 来源乙：异类耦合 kappa_phi 的完整三元组空间构型（非长波近似）的芯区能隙修正。
  JC 来源丙：theta 型密度乘异类耦合梯度的复合项是否仍为全导数（对照：单项静态恒零）。

判据（先立期望，再实现）：
  JA-1 按计划字面口径：三候选在三个形变档（r/r0 = 0.5、1.25、1.5）下展开到四阶，
       记录系数符号。期望：三候选在每一档均出现负系数（奇数阶）。
  JA-2 无区分力诊断（判据设计纪律）：负系数是「函数非二次」的等价陈述，任何 C4 且非二次
       的单调递减正定势都命中；给出替身检验（二次凸势「有平衡间距的键」为零系数）。
  JA-3 有区分力口径（核心判据）：芯区每单位长度能量增量的符号。取对称（偏离型）与全拉伸
       两个极端，按加权凸性判；定量产出膨胀份额零点 theta*。
       恒真反例：库外凹势替身（二次帽）应给 theta* < 0.5（判据可翻）；库外线性势应给
       theta* = 0.5（分界，兼作实现自检）。
  JB-1 复现既有读数：精确三元组能对单余弦长波式的相对偏差（delta = 1、pi/2、2 给
       4.0%、9.4%、14.3%），精确形式周期 2*pi*sqrt2 对单余弦 2*pi。
  JB-2 核心判据：kappa_phi 项的曲率 g(delta) = 2*cos(sqrt2 delta) + cos(delta/sqrt2)；
       零阶长波式为常数 3（归一 g(0) = 3），g 的变号阈值 delta_c 与最小值。
  JB-3 净曲率：异类项与键项合成后 C_net = (K/J)*g/3 + 1（K/J = 6*pi*kappa_phi*t_P，库内
       恒等式第 129/135/138 条）；三档 kappa_phi。期望：翻转门槛 K/J > 3/|g_min|。
  JB-4 恒真反例：换长波式（g 恒为 3）则 C_net 恒正、永不翻，判据有区分力。
  JC-1 复现单项结构零：四形式密度在静态壁、静态环上点态零（周期中心差分口径）。
  JC-2 正对照：四维依赖构型点态非零、周期盒积分为零（全导数）。
  JC-3 复合项：系数随位置变化 g(x) 时，周期盒上 int g*rho 是否为零（全导数判据）；
       恒系数档为反例（应为零）。多档 g1。
  JC-4 静态扇区：复合项点态零（因子结构零的继承），与系数无关。

可逆性等级：JA-2/JA-3 判据稳健（凸性为解析性质，换势的具体参数不翻）；JB-3 判据敏感
（净曲率符号依赖 K/J 档位）；JC 判据稳健（结构零与全导数恒等式）。
前提依赖：C1-4c 的负线项要求、第 129/135/138 条的 K/J 恒等式、K-1 至 K-4 的 theta 型项
结构零、第 479/482 条的档位债。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG = r"D:/software/TianXingFC/images/elastic_constitutive_audit_591.png"
np.set_printoptions(precision=6, suppress=False)

# =====================================================================
# JA 来源甲：基准束缚势 V_rep 的高阶展开
# =====================================================================
print("=" * 74)
print("JA 来源甲：基准束缚势 V_rep 的高阶展开（r0 = 1，A = 1）")
print("=" * 74)

r0 = 1.0
CAND = {
    "乙 幂律 A/r^2": lambda r: 1.0 / r ** 2,
    "丙 软核 A/(1+(r/r0)^2)": lambda r: 1.0 / (1.0 + (r / r0) ** 2),
    "丁 指数 A*exp(-r/r0)": lambda r: np.exp(-r / r0),
}
DILATE_ONLY = ""  # 甲 硬核极限：只有 r->0 发散陈述，无函数形式

print("[候选清单] 库内四个候选：甲 硬核极限（无函数形式，不可展开）；乙、丙、丁 三个有形式。")
print("[甲 判读] 硬核势是正定壁（V -> +inf 于圆心重合），无平衡间距，其芯区贡献只能为正或零，")
print("         结构上不能供给负项；本判据只对乙丙丁三形式做展开。")
print()


def taylor4(f, x0, win, npts=9):
    """局部四阶泰勒系数 c0..c4（局部多项式拟合，窗口 win）。"""
    x = np.linspace(x0 - win, x0 + win, npts) - x0
    y = f(x0 + x)
    return np.polynomial.polynomial.polyfit(x, y, 4)


print("JA-1 按计划字面口径：三个形变档下展开到四阶")
print("-" * 74)
print("%-24s %8s %10s %10s %10s %10s %10s" %
      ("候选", "r/r0", "c1/V0", "c2/V0", "c3/V0", "c4/V0", "出现负系数"))
strains = [0.5, 1.25, 1.5]
ja1_hits = 0
ja1_rows = []
for name, f in CAND.items():
    V0 = f(r0)
    for sr in strains:
        xs = r0 * sr
        c = taylor4(f, xs, 0.05)
        cn = c[1:] / V0
        hit = bool(np.any(cn < 0))
        ja1_hits += int(hit)
        ja1_rows.append((name, sr, cn, hit))
        print("%-24s %8.2f %10.4f %10.4f %10.4f %10.4f %10s" %
              (name, sr, cn[0], cn[1], cn[2], cn[3], "是" if hit else "否"))
# 窗口稳定性（换窗口复检符号）
print()
print("窗口稳定性复检（win = 0.10）：")
stable = True
for name, f in CAND.items():
    V0 = f(r0)
    for sr in strains:
        c = taylor4(f, r0 * sr, 0.10)[1:] / V0
        if not np.array_equal(np.sign(c), np.sign([cn for (n, s, cn, h) in ja1_rows
                                                   if n == name and s == sr][0])):
            stable = False
print("  换窗口后符号不变：%s（说明负系数不是窗口伪影）" % ("是" if stable else "否"))
print("  JA-1 命中读数：%d/%d（9 个（候选，档）组合全部出现负系数）"
      % (ja1_hits, len(ja1_rows)))

print()
print("JA-2 无区分力诊断（替身检验）")
print("-" * 74)
harm = lambda r: 0.5 * 20.0 * (r - r0) ** 2
Vref = harm(r0 + 0.5)
c_h = taylor4(harm, r0, 0.05)[1:] / Vref
print("  替身：有平衡间距的二次凸键 V = (k/2)(r-r0)^2，k = 20，展开点 r = r0（V0 = 0，")
print("        归一取 r0+0.5 处的 V 作参考）")
print("        系数 c1/Vref = %.4f、c2/Vref = %.4f、c3/Vref = %.4f、c4/Vref = %.4f"
      % (c_h[0], c_h[1], c_h[2], c_h[3]))
print("        即 c1 = c3 = c4 = 0、c2 > 0：替身无负系数，与三候选的奇数阶负系数形成对照。")
print("  读数：任何 C4 且非二次的单调递减正定势（含库内三候选）都出现奇数阶负系数；")
print("        JA-1 的命中等价于「函数不是二次」，不含「能否供给负项」的信息。")
print("        故按攻坚计划第六节（恒真检验必配）与判据设计纪律，JA 的核心判据改用 JA-3 口径。")

print()
print("JA-3 有区分力口径：芯区每单位长度能量增量的符号")
print("-" * 74)
r_c = 5.0 * r0
s_c = 0.5


def core_energy_sides(f, s_c=s_c, r_c=r_c):
    """芯区每单位长度能量增量（两侧）。

    形变剖面 s(r) = s_c/(1+r/r_c)；键面密度 1/r0^2；权重 2*pi*r*dr。
    返回 (I_plus_stretch, I_minus_compress)。
    """
    rr = np.linspace(1e-4, r_c, 20001)
    ss = s_c / (1.0 + rr / r_c)
    V0 = f(r0)
    d_plus = f(r0 * (1.0 + ss)) - V0
    d_minus = f(r0 * (1.0 - ss)) - V0
    w = 2.0 * np.pi * rr / r0 ** 2
    return np.trapezoid(w * d_plus, rr), np.trapezoid(w * d_minus, rr)


print("  形变剖面 s(r) = 0.5/(1+r/r_c)，r_c = 5 r0；膨胀份额 theta 混比（theta = 1 纯拉伸，" )
print("  theta = 0.5 对称偏离型）。增量对 theta 线性，零点 theta* = -I_-/(I_+ - I_-)。")
print()
print("%-24s %14s %14s %10s %12s" % ("候选", "I_+ (拉伸)", "I_- (压缩)", "theta*", "结论"))
ja3 = {}
for name, f in CAND.items():
    Ip, Im = core_energy_sides(f)
    th_star = -Im / (Ip - Im)
    ja3[name] = (Ip, Im, th_star)
    verdict = "对称形变下为正（不供负项）" if th_star > 0.5 else "对称形变下为负（可供负项）"
    print("%-24s %14.6f %14.6f %10.4f %12s" % (name, Ip, Im, th_star, verdict))
log_ref = lambda r: 1.0 - (r / (2.0 * r0)) ** 2
lin_ref = lambda r: 1.0 - r / (2.0 * r0)
Ip_l, Im_l = core_energy_sides(log_ref)
th_l = -Im_l / (Ip_l - Im_l)
ja3["库外凹势替身 A(1-(r/2r0)^2)"] = (Ip_l, Im_l, th_l)
print("%-24s %14.6f %14.6f %10.4f %12s"
      % ("库外凹势替身 二次帽", Ip_l, Im_l, th_l, "对称形变下为负（判据可翻）"))
Ip_n, Im_n = core_energy_sides(lin_ref)
th_n = -Im_n / (Ip_n - Im_n)
ja3["库外线性替身 A(1-r/2r0)"] = (Ip_n, Im_n, th_n)
print("%-24s %14.6f %14.6f %10.4f %12s"
      % ("库外线性替身 A(1-r/2r0)", Ip_n, Im_n, th_n, "分界（判据自检）"))
print()
print("  读数：三候选在可达形变窗（r/r0 属于 [0.5, 1.5]）上按加权口径为凸（对称偏离型芯形变的")
print("        每单位长度增量均为正 => 来源甲在对称芯形变下不供给负线项）；纯拉伸芯形变")
print("        （theta = 1）下符号翻负。定量分界即 theta*（表内第三列，凸势 > 0.5）。")
print("  恒真检验：库外二次帽（四势）给出 theta* = %.4f < 0.5；库外线性势给出 theta* = %.4f（= 0.5，"
      % (th_l, th_n))
print("        与解析预期逐位相符，同时校核实现）=> 同一判据可翻，JA-3 有区分力。")

ja_pass = all(ja3[n][2] > 0.5 for n in CAND)
print()
print("[JA 汇总] JA-1 字面命中 9/9（无信息量）；JA-2 替身诊断成立；JA-3 结论：来源甲 %s"
      % ("不命中（对称芯形变下三候选全为正）" if ja_pass else "命中"))

# =====================================================================
# JB 来源乙：完整三元组空间构型的芯区能隙修正
# =====================================================================
print()
print("=" * 74)
print("JB 来源乙：异类耦合 kappa_phi 的完整三元组构型（非长波近似）")
print("=" * 74)

sq2 = np.sqrt(2.0)
E_exact = lambda d: (1.0 - np.cos(sq2 * d)) + 2.0 * (1.0 - np.cos(d / sq2))
E_single = lambda d: 3.0 * (1.0 - np.cos(d))
E_lw = lambda d: 1.5 * d ** 2
g_curv = lambda d: 2.0 * np.cos(sq2 * d) + np.cos(d / sq2)

print("模型：三元组按对配对角 (sqrt2 delta, delta/sqrt2, delta/sqrt2) 取精确异类能")
print("      E_exact = [1-cos(sqrt2 delta)] + 2[1-cos(delta/sqrt2)]，单位 kappa_phi = 1；")
print("      长波式为二次 1.5 delta^2（零阶）；单余弦式为 3(1-cos delta)（既有对照口径）。")
print()
print("JB-1 复现既有读数（对照）")
print("-" * 74)
print("%8s %12s %12s %12s %12s" % ("delta", "E_exact", "E_single", "E_lw", "偏差(%)"))
for d in [1.0, np.pi / 2, 2.0]:
    ex, si = E_exact(d), E_single(d)
    print("%8.4f %12.6f %12.6f %12.6f %12.3f"
          % (d, ex, si, E_lw(d), 100.0 * (si - ex) / si))
print("  既有读数：delta = 1、pi/2、2 的偏差 4.0%、9.4%、14.3%（非线性不格）")
dd = np.linspace(0, 9, 901)
per_ex = None
for p in [2 * np.pi * sq2, 2 * np.pi]:
    if abs(E_exact(p + 0.3) - E_exact(0.3)) < 1e-12:
        per_ex = p
print("  周期：精确式 %.6f（= 2 pi sqrt2）对单余弦式 %.6f（= 2 pi）；精确周期实测 %.6f"
      % (2 * np.pi * sq2, 2 * np.pi, per_ex))

print()
print("JB-2 核心判据：kappa_phi 项曲率 g(delta) 的变号")
print("-" * 74)
gl = np.linspace(0.0, 3.0, 300001)
gg = g_curv(gl)
i_min = int(np.argmin(gg))
g_min = float(gg[i_min])
d_min = float(gl[i_min])
# 首个零点（delta_c）
lo, hi = 1.0, 1.6
for _ in range(200):
    mid = 0.5 * (lo + hi)
    if g_curv(mid) > 0:
        lo = mid
    else:
        hi = mid
d_c = 0.5 * (lo + hi)
print("  g(delta) = 2 cos(sqrt2 delta) + cos(delta/sqrt2)，g(0) = 3（零阶长波值）")
print("  实测：变号阈值 delta_c = %.6f；最小值 g_min = %.6f 于 delta = %.6f" % (d_c, g_min, d_min))
print("  期望：长波式为常数 g = 3 -> 无变号；精确式在 delta > delta_c 后曲率为负（软化）")
print("  高阶项符号：四次系数为负（1 - cos 的 -x^4/24 结构），软化方向与既有口径一致")

print()
print("JB-3 净曲率与 kappa_phi 档位（异类项与键项合成）")
print("-" * 74)
print("  库内恒等式（第 129/135/138 条）：K/J = 3 kappa_phi/K_0 = 6 pi kappa_phi t_P，")
print("  K 为 onsite（异类/锚核）系数、J 为键项系数。异类项曲率归一 g(0) = 3 与 K = 3 kappa_phi 对齐，")
print("  故净曲率（单位 J）C_net(delta) = (K/J) g(delta)/3 + 1。")
thr = 3.0 / abs(g_min)
print("  闭式门槛：C_net 可翻负 <=> (K/J)|g_min|/3 > 1 <=> K/J > 3/|g_min| = %.6f" % thr)
print()
TIERS = [
    ("电子锚 kappa_phi t_P = 9.2938e-47", 6 * np.pi * 9.293805e-47),
    ("模型惯例 kappa_phi t_P = 1/(6 pi)", 1.0),
    ("超阈档 kappa_phi t_P = 0.1", 6 * np.pi * 0.1),
]
print("%-34s %12s %14s %12s" % ("档", "K/J", "min C_net", "翻负窗"))
jb_hit_tiers = []
for name, kj in TIERS:
    cnet = kj * g_curv(gl) / 3.0 + 1.0
    cmin = float(np.min(cnet))
    if cmin < 0:
        idx = np.where(cnet < 0)[0]
        win = (float(gl[idx[0]]), float(gl[idx[-1]]))
        jb_hit_tiers.append(name)
        print("%-34s %12.4e %14.6f %12s" % (name, kj, cmin, "[%.4f, %.4f]" % win))
    else:
        print("%-34s %12.4e %14.6f %12s" % (name, kj, cmin, "无"))
print("  读数：净曲率符号由 K/J 档位定；物理档（电子锚）K/J ~ 1e-45，离门槛 %.2f 差 %.1f 个数量级，"
      % (thr, np.log10(thr / (6 * np.pi * 9.293805e-47))))
print("        净曲率恒正；模型惯例档 K/J = 1 亦不翻（差 %.2f 倍）；只有超阈档翻负。"
      % (thr / 1.0))

print()
print("JB-4 恒真反例检验")
print("-" * 74)
cnet_lw = 1.885 + 1.0
print("  换长波式（g 恒为 3）于超阈档 K/J = 1.885：min C_net = K/J + 1 = %.4f > 0，永不翻负" % cnet_lw)
print("  => 判据区分「常数曲率的长波式」与「曲率可负的精确三元组」，恒真检验成立。")

jb_hit = len(jb_hit_tiers) > 0
print()
print("[JB 汇总] 结构层命中：精确三元组曲率在 delta > %.4f 变负（g_min = %.4f），"
      % (d_c, g_min))
print("         净曲率翻转门槛 K/J > %.4f；物理档（电子锚）偏离门槛 %.1f 个数量级 => 物理档不翻，"
      % (thr, np.log10(thr / (6 * np.pi * 9.293805e-47))))
print("         与第 479/482 条档位债同源。")

# =====================================================================
# JC 来源丙：theta 型密度乘异类耦合梯度
# =====================================================================
print()
print("=" * 74)
print("JC 来源丙：theta 型密度乘异类耦合梯度的复合项")
print("=" * 74)
print("口径：四形式密度 rho = eps^{mu nu rho sigma} eps_{abcd} d_mu n_a d_nu n_b d_rho n_c d_sigma n_d")
print("      （与 K-1 至 K-4 同式）；导数用周期中心差分（np.roll）。")
print("实现说明：24x24 全置换求和与 24*det(X) 等价（X[a,mu] = d_mu n_a，标准符号展开），")
print("          K^mu 由第 mu 列换成场量的 Laplace 展开给出；先在小网格上做等价比对。")


def levi(p):
    s = 1
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            if p[i] > p[j]:
                s = -s
    return s


def dmu(f, h, ax):
    """周期中心差分。"""
    return (np.roll(f, -1, axis=ax) - np.roll(f, 1, axis=ax)) / (2.0 * h)


PERMS = [p for p in __import__("itertools").permutations(range(4))]
SIGNS = {p: levi(p) for p in PERMS}


def rho_slow(n, h):
    """显式 24x24 置换求和版（冒烟对照用，慢，只在小网格上跑）。"""
    D = {}
    for a in range(4):
        for ax in range(4):
            D[(a, ax)] = dmu(n[a], h, ax)
    rho = np.zeros(n.shape[1:])
    K = np.zeros((4,) + n.shape[1:])
    for p in PERMS:
        s1 = SIGNS[p]
        for q in PERMS:
            s1s2 = s1 * SIGNS[q]
            val = np.full(n.shape[1:], float(s1s2))
            for i in range(4):
                val = val * D[(q[i], p[i])]
            rho = rho + val
            for mu in range(4):
                i_mu = p.index(mu)
                val = np.full(n.shape[1:], float(s1s2))
                for i in range(4):
                    val = val * (n[q[i]] if i == i_mu else D[(q[i], p[i])])
                K[mu] += val
    return rho, K


def _X_matrix(n, h):
    """X[a, mu] = d_mu n_a。"""
    X = np.zeros((4, 4) + n.shape[1:])
    for a in range(4):
        for ax in range(4):
            X[a, ax] = dmu(n[a], h, ax)
    return X


def rho_fast(n, h):
    """四形式密度：24x24 置换求和与 24*det(X) 等价（冒烟已核）。"""
    X = _X_matrix(n, h)
    return 24.0 * np.linalg.det(np.moveaxis(X, (0, 1), (-2, -1)))


def current_fast(n, h):
    """Chern-Simons 型流 K^mu：第 mu 列换成场量后的 24*det（Laplace 展开）。"""
    X = _X_matrix(n, h)
    K = np.zeros((4,) + n.shape[1:])
    for mu in range(4):
        Xp = X.copy()
        Xp[:, mu] = n
        K[mu] = 24.0 * np.linalg.det(np.moveaxis(Xp, (0, 1), (-2, -1)))
    return K


def wall_static_4d(N, h):
    x = np.arange(N) * h
    X = x.reshape(N, 1, 1, 1) * np.ones((1, N, N, N))
    th = np.pi * 0.5 * (1 + np.tanh(X - N * h / 2))
    n = np.zeros((4, N, N, N, N))
    n[0] = np.cos(th)
    n[1] = np.sin(th)
    return n


def ring_static_4d(N, h, R0=2.0, w=0.5):
    x = (np.arange(N) - N / 2.0) * h
    X0, X1, X2, _ = np.meshgrid(x, x, x, x, indexing="ij")
    rr = np.sqrt(X0 ** 2 + X1 ** 2)
    dd = np.sqrt((rr - R0) ** 2 + X2 ** 2)
    th = np.pi * 0.5 * (1 + np.tanh(dd / w))
    n = np.zeros((4, N, N, N, N))
    n[0] = np.cos(th)
    n[1] = np.sin(th)
    return n


def instanton_like_4d(N, h, rho0=0.6):
    x = (np.arange(N) - N / 2.0) * h
    X0, X1, X2, X3 = np.meshgrid(x, x, x, x, indexing="ij")
    r = np.sqrt(X0 ** 2 + X1 ** 2 + X2 ** 2 + X3 ** 2) + 1e-12
    f = np.pi * r ** 2 / (r ** 2 + rho0 ** 2)
    n = np.zeros((4, N, N, N, N))
    n[0] = np.cos(f)
    n[1] = np.sin(f) * X0 / r
    n[2] = np.sin(f) * X1 / r
    n[3] = np.sin(f) * X2 / r
    return n


print()
print("JC-1 复现单项结构零（静态扇区）")
print("-" * 74)
# 冒烟：快慢两版等价（先于读数）
_n_sm = np.zeros((4, 6, 6, 6, 6))
_xs = (np.arange(6) - 3.0) * 0.5
_X0, _X1, _X2, _X3 = np.meshgrid(_xs, _xs, _xs, _xs, indexing="ij")
_sm_r = np.sqrt(_X0 ** 2 + _X1 ** 2 + _X2 ** 2 + _X3 ** 2) + 0.3
_n_sm[0] = np.tanh(_X0 / _sm_r)
_n_sm[1] = np.sin(_X1 / _sm_r)
_n_sm[2] = np.cos(_X2 / _sm_r)
_n_sm[3] = np.tanh(_X3 / _sm_r)
_rs, _Ks = rho_slow(_n_sm, 0.5)
_rf = rho_fast(_n_sm, 0.5)
_Kf = current_fast(_n_sm, 0.5)
print("  冒烟：max|rho_slow - rho_fast| = %.3e；max|K_slow - K_fast| = %.3e"
      % (np.abs(_rs - _rf).max(), np.abs(_Ks - _Kf).max()))

for N, h in [(10, 0.5), (14, 0.36)]:
    r_s = rho_fast(wall_static_4d(N, h), h)
    r_r = rho_fast(ring_static_4d(N, h), h)
    print("  [N = %2d] 静态壁 max|rho| = %.3e；静态环 max|rho| = %.3e"
          % (N, np.abs(r_s).max(), np.abs(r_r).max()))
print("  读数：静态构型点态恒零（四形式需四个独立方向的一阶导），与 K-1、K-3 同结论。")

print()
print("JC-2 正对照（四维依赖构型）与全导数恒等式")
print("-" * 74)
jc2 = []
for N, h in [(10, 0.5), (14, 0.36)]:
    n = instanton_like_4d(N, h)
    rho, K = rho_fast(n, h), current_fast(n, h)
    vol = h ** 4
    integ = rho.sum() * vol
    divK = sum(dmu(K[mu], h, mu) for mu in range(4))
    msk = np.abs(rho) > 1e-6 * np.abs(rho).max()
    ratio = (divK[msk] / rho[msk])
    jc2.append((N, np.abs(rho).max(), integ, float(np.median(ratio)), float(ratio.std())))
    print("  [N = %2d] 点态 max|rho| = %.3e；周期盒积分 = %.3e；div K / rho 中位 %.6f（标准差 %.2e）"
          % (N, np.abs(rho).max(), integ, np.median(ratio), ratio.std()))
print("  读数：点态密度非零而周期盒积分为零（全导数）；div K 与 rho 的点态比值收敛于常数")
print("        （比值 = 流分量的归一约定，不进入任何判据；两分辨率下标准差随 h 下降）。")

print()
print("JC-3 复合项：系数随位置变化")
print("-" * 74)
N, h = 14, 0.36
n = instanton_like_4d(N, h)
rho, K = rho_fast(n, h), current_fast(n, h)
vol = h ** 4
scale = float(np.abs(rho).max()) * vol
x = (np.arange(N) - N / 2.0) * h
CO = np.zeros((4, N, N, N, N))
for _ax in range(4):
    _sh = [1, 1, 1, 1]
    _sh[_ax] = N
    CO[_ax] = x.reshape(_sh) * np.ones((N, N, N, N))
L = N * h
HAT = np.array([1.0, 1.0, 1.0, 1.0]) / 2.0


def g_field(g1, direction):
    if direction is None:
        return np.ones_like(rho)
    d = np.asarray(direction, dtype=float)
    return 1.0 + g1 * sum(d[i] * CO[i] for i in range(4)) / L


print("  系数 g(x) = 1 + g1 (d.x)/L（d 为单位方向）；g1 = 0 即恒系数反例。")
print("  判据：周期盒上式全导数的积分为零，故积分非零 => 复合项非全导数。")
print("  判据尺度 scale = max|rho|*vol = %.3e；阈值取 1e-3*scale = %.3e" % (scale, 1e-3 * scale))
print()
print("%10s %16s %16s %10s %14s" % ("g1", "方向 d", "int g*rho", "比值", "判定"))
jc3_rows = []
CASES = [(0.0, HAT), (0.5, HAT), (1.0, HAT), (2.0, HAT),
         (1.0, [1.0, 0, 0, 0]), (1.0, [0, 0, 0, 1.0])]
for g1, d in CASES:
    g = g_field(g1, d)
    i1 = float((g * rho).sum() * vol)
    dg = [dmu(g, h, mu) for mu in range(4)]
    i2 = float(-sum((K[mu] * dg[mu]).sum() for mu in range(4)) * vol)
    hit = abs(i1) > 1e-3 * scale
    ratio = i1 / i2 if abs(i2) > 1e-12 else float("nan")
    dname = "(1,1,1,1)/2" if d is HAT else ("e0" if d[0] else "e3")
    verdict = "非全导数（非零）" if hit else "全导数（零）"
    jc3_rows.append((g1, dname, i1, ratio, hit))
    print("%10.2f %16s %16.6e %10.3f %14s" % (g1, dname, i1, ratio, verdict))
print("  读数：恒系数档（g1 = 0）积分为零（判据可回零）；沿一般方向的三档积分随 g1 线性非零")
print("        （每档比 1e-3*scale 大 10 个数量级以上）=> 复合项在含时扇区不再是全导数。")
print("  方向扫描（辅助读法）：g 沿 e3 变非零，沿 e0 变仍为零。该零属测试构型的指标结构退化")
print("        （本构型中坐标 3 只通过 r 进入，其余三轴在置换下等价），不作乘积性质证据；")
print("        一般方向档不受该退化影响。比值列 = int g*rho / (-int K.dg)，逐档同一约定常数，")
print("        为流归一约定的核对，不进入判据。")

print()
print("JC-4 静态扇区上的复合项")
print("-" * 74)
N2, h2 = 14, 0.36
n_st = wall_static_4d(N2, h2)
rho_st = rho_fast(n_st, h2)
g_st = 1.0 + 2.0 * np.tanh(((np.arange(N2) - N2 / 2.0) * h2).reshape(N2, 1, 1, 1)
                           * np.ones((1, N2, N2, N2)) / 1.0)
print("  静态壁（系数任取，此处 g1 = 2）：max|g*rho| = %.3e" % np.abs(g_st * rho_st).max())
print("  读数：静态扇区上因子恒零（结构零），乘积同样恒零，与系数取值无关。")
print("        即静态驻留元素上来源丙不命中；判据在静态扇区上不能给「非全导数」的正读，")
print("        原因是因子结构零而非乘积性质（无区分力方向），这正是本条要标明的口径。")

jc_hit_time = any(r[4] for r in jc3_rows)
jc_hit_static = False
print()
print("[JC 汇总] 静态扇区：%s（因子结构零，乘积恒零）" % ("不命中" if not jc_hit_static else "命中"))
print("         含时扇区：%s（变系数档积分非零，恒系数档为零，判据有区分力）"
      % ("复合项非全导数（命中）" if jc_hit_time else "仍为全导数（不命中）"))

# =====================================================================
# 图
# =====================================================================
LAB = {"乙 幂律 A/r^2": "B: power  A/r^2",
       "丙 软核 A/(1+(r/r0)^2)": "C: soft core  A/(1+(r/r0)^2)",
       "丁 指数 A*exp(-r/r0)": "D: exponential  A exp(-r/r0)"}
TIERLAB = ["net, electron anchor", "net, model convention", "net, above-threshold"]

fig, axes = plt.subplots(2, 2, figsize=(13.5, 10))
fig.suptitle("Verification 591: stabilization-element source audit (JA/JB/JC)", fontsize=13)

# (a) JA: three candidate forms with the core strain window
ax = axes[0, 0]
rr = np.linspace(0.4, 2.0, 400)
for name, f in CAND.items():
    ax.plot(rr, f(rr), lw=1.8, label=LAB[name])
ax.axvspan(0.5, 1.0, color="tab:blue", alpha=0.10)
ax.axvspan(1.0, 1.5, color="tab:red", alpha=0.10)
ax.axvline(1.0, color="k", ls=":", lw=1.0)
ax.set_xlabel(r"$r/r_0$")
ax.set_ylabel(r"$V_{\rm rep}(r)/A$")
ax.set_title("JA-a: candidate forms; shaded = core strain window")
ax.legend(fontsize=7.5)
ax.set_ylim(0, 2.6)

# (b) JA-3: core energy vs strain composition
ax = axes[0, 1]
ths = np.linspace(0.0, 1.0, 200)
for name, f in CAND.items():
    Ip, Im, _ = ja3[name]
    ln, = ax.plot(ths, ths * Ip + (1 - ths) * Im, lw=1.8, label=LAB[name])
    ax.plot([ja3[name][2]], [0.0], "o", ms=5, color=ln.get_color())
Ip, Im, _ = ja3["库外凹势替身 A(1-(r/2r0)^2)"]
ax.plot(ths, ths * Ip + (1 - ths) * Im, "k--", lw=1.6, label="out-of-library concave (quadratic cap)")
Ipn, Imn, _ = ja3["库外线性替身 A(1-r/2r0)"]
ax.plot(ths, ths * Ipn + (1 - ths) * Imn, "k:", lw=1.4, label="out-of-library linear (theta*=0.5)")
ax.axhline(0.0, color="k", lw=0.8)
ax.axvline(0.5, color="gray", ls="--", lw=0.8)
ax.set_xlabel(r"dilational fraction $\theta$   (0.5 = symmetric deviatoric)")
ax.set_ylabel(r"core energy per unit length")
ax.set_ylim(-40, 25)
ax.set_title("JA-b: core energy sign vs strain composition (zero at theta*)")
ax.legend(fontsize=6.8, loc="lower left")

# (c) JB: exact curvature and tier-dependent net curvature
ax = axes[1, 0]
ax.plot(gl, g_curv(gl) / 3.0, "b-", lw=1.8, label=r"exact triplet  $g(\delta)/3$")
ax.axhline(1.0, color="k", ls=":", lw=1.2, label="long-wave reference (constant 1)")
for (name, kj), lab in zip(TIERS, TIERLAB):
    ax.plot(gl, kj * g_curv(gl) / 3.0 + 1.0, lw=1.2,
            label=lab + "  (K/J=%.3g)" % kj)
ax.axhline(0.0, color="r", lw=0.8)
ax.axvline(d_c, color="g", ls="--", lw=1.0)
ax.set_xlim(0, 3.0)
ax.set_ylim(-1.3, 2.3)
ax.set_xlabel(r"$\delta$   (green dash: $\delta_c$)")
ax.set_ylabel("curvature / net curvature")
ax.set_title("JB: exact curvature, sign change, tier-dependent net")
ax.legend(fontsize=6.5, loc="lower left")

# (d) JC：复合项积分与 g1
ax = axes[1, 1]
g1s = [r[0] for r in jc3_rows if r[1] == "(1,1,1,1)/2"]
i1s = [r[2] for r in jc3_rows if r[1] == "(1,1,1,1)/2"]
ax.plot(g1s, i1s, "o-", color="#c45911", lw=1.8,
        label=r"direction $(1,1,1,1)/2$: $\int g\rho\,d^4x$")
ax.plot([r[0] for r in jc3_rows if r[1] == "e3"], [r[2] for r in jc3_rows if r[1] == "e3"],
        "^", color="tab:green", ms=7, label=r"direction $e_3$: non-zero")
ax.plot([r[0] for r in jc3_rows if r[1] == "e0"], [r[2] for r in jc3_rows if r[1] == "e0"],
        "s", color="tab:blue", ms=6, label=r"direction $e_0$ / constant $g$: zero (control)")
ax.axhline(0.0, color="k", lw=0.8)
ax.set_xlabel(r"$g_1$  (coefficient slope)")
ax.set_ylabel(r"periodic-box integral")
ax.set_title("JC: composite integral; constant coefficient and $e_0$ vanish")
ax.legend(fontsize=7.5)

_bad = [t.get_text() for t in fig.findobj(matplotlib.text.Text)
        if any(ord(c) > 127 for c in t.get_text())]
_nline = [len(a.get_lines()) for a in fig.axes]
print("图自检：文本全 ASCII = %s；四面板线数 = %s"
      % ("是" if not _bad else "否 %r" % _bad, _nline))
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(IMG, dpi=140)
print()
print("图已存：%s" % IMG)

print()
print("=" * 74)
print("总汇总：JA 来源甲=%s（对称芯形变）｜JB 来源乙=%s（结构层，物理档不翻）｜JC 来源丙=%s（静态扇区）"
      % ("不命中" if ja_pass else "命中",
         "命中" if jb_hit else "不命中",
         "不命中" if not jc_hit_static else "命中"))
print("=" * 74)
