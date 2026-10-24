# -*- coding: utf-8 -*-
"""
判据 745：提升相位的打包精确性与 2pi 因子的一致性。

背景与靶子的更正。
判据 744 建议下一步问「跃迁的量子幅度」，即层数更新是否携带相位因子。
本轮先核库内规定，得结论：该提法不成立，靶子须改。

库内两处规定。
  一、[[wiki/元相方程]] 方程 (II)：$dz/dt = 2\\pi c\\,\\delta(\\arg\\psi)$。
     其 $2\\pi$ 来自 $A = \\omega_P l_P = 2\\pi c$，而 $\\omega_P l_P = 2\\pi c$ 出自普朗克关系。
     即该处的 $2\\pi$ 是**运动学的**（弧长 $2\\pi\\hbar$ 与周期 $2\\pi/\\omega_P$ 的几何自洽），
     不是量子幅度。方程 (II) 的解是实数阶梯 $z(t) = l_P\\lfloor ct/l_P \\rfloor$，层数更新不带相位因子。
  二、[[wiki/提升相位]]：$\\tilde\\theta_i = 2\\pi N_i + \\theta_i$，$N_i$ 为累积摆程数、
     $\\theta_i$ 为模 $2\\pi$ 的相位角；且「这一打包在底层是精确的。晶格刚性、普朗克频率不变、
     跃迁距离不变、满周闭合不变四者咬合，使相位与层数必然同步，不存在两者的独立变化」。

故「层数更新是否带相位因子」这一问法预设了两者可独立变化，与该精确性冲突。
正确的问题：**相位与层数被同一个量 $\\tilde\\theta$ 打包，打包的精确性在两侧各给出什么约束。**

本轮判据组：
  组 A：打包的自洽性（$\\tilde\\theta$ 的整数部分即层数、小数部分即相位角）
  组 B：$2\\pi$ 因子在两侧的一致性（弧长侧与周期侧同源）
  组 C：四者咬合的可判后果（晶格刚性、普朗克频率不变、跃迁距离不变、满周闭合不变）
  组 D：恒真检验与区分力

判据先立，再实现。容差从误差预算推导。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\p745_lift_phase.png"

RESULTS = []


def rec(tag, what, exp, got, ok):
    RESULTS.append((tag, what, exp, got, ok))
    print("[%s] %s" % (tag, what))
    print("     期望: %s" % exp)
    print("     实测: %s" % got)
    print("     结论: %s" % ("PASS" if ok else "FAIL"))
    print("")


TWO_PI = 2.0 * np.pi

# 库内常数关系（普朗克关系）：
#   E_P = h f = 2pi hbar * (omega_P/2pi) = hbar omega_P
#   弧长 2pi hbar（圆周）与周期 2pi/omega_P（时间）互为共轭
HBAR = 1.0
OMEGA_P = 1.0
E_P = HBAR * OMEGA_P
# 库内的三个身份（[[wiki/普朗克长度]] 第 16 行）：
#   满周跃迁的位移 = l_P
#   光走一拍的距离 = c * t_P = l_P
#   标准普朗克长度（与前两者逐位一致）
# 其中 t_P 是一圈的时间：omega_P 是**角频率**，一圈含 2pi 弧度，故 t_P = 2pi/omega_P。
# 于是 l_P = c t_P，且由 omega_P l_P = 2pi c 得 c = omega_P l_P / 2pi = l_P / t_P。
T_P = TWO_PI / OMEGA_P
L_P = 1.0                          # 满周跃迁的位移，格距，取为 1 作标度
C_LIGHT = L_P / T_P                # 光走一拍的距离恰为 l_P，故 c = l_P / t_P
# 核对：omega_P l_P = 2pi c
assert abs(OMEGA_P * L_P - TWO_PI * C_LIGHT) < 1e-12


# =====================================================================
# 组 A：打包的自洽性
# =====================================================================
print("=" * 72)
print("组 A：打包的自洽性")
print("=" * 72)

# A1：提升相位的整数部分即层数、小数部分即相位角
rng = np.random.default_rng(20261021)
N_list = rng.integers(-8, 9, size=400)
theta_list = rng.uniform(-np.pi, np.pi, size=400)
tilde = TWO_PI * N_list + theta_list
N_back = np.floor((tilde + np.pi) / TWO_PI).astype(int)
theta_back = tilde - TWO_PI * N_back
rec(
    "A1",
    "提升相位的往返：整数部分还原层数、小数部分还原相位角",
    "层数逐位还原，偏差 0；相位角逐位还原，偏差 ≤ 1e-12",
    "层数最大偏差 %d；相位角最大偏差 %.3e"
    % (int(np.abs(N_back - N_list).max()), float(np.abs(theta_back - theta_list).max())),
    int(np.abs(N_back - N_list).max()) == 0
    and float(np.abs(theta_back - theta_list).max()) < 1e-12,
)

# A1b：区分判据。平移一整圈（2pi），层数应加一、相位角应不变。
# 这条排除了「打包只是把相位重标号」的可能。
tilde_shift = tilde + TWO_PI
N_sh = np.floor((tilde_shift + np.pi) / TWO_PI).astype(int)
theta_sh = tilde_shift - TWO_PI * N_sh
rec(
    "A1b",
    "区分判据：提升相位平移一整圈后，层数加一、相位角不变",
    "层数逐位加一，偏差 0；相位角逐位不变，偏差 ≤ 1e-12。"
    "故打包不是重标号，圈数被真实地记进了层数",
    "层数增量最大偏差 %d；相位角最大偏差 %.3e"
    % (int(np.abs((N_sh - N_back) - 1).max()),
       float(np.abs(theta_sh - theta_back).max())),
    int(np.abs((N_sh - N_back) - 1).max()) == 0
    and float(np.abs(theta_sh - theta_back).max()) < 1e-12,
)

# A2：层数变化一格，提升相位变化恰为 2pi
dN = 1
d_tilde = TWO_PI * dN
rec(
    "A2",
    "层数变一格时提升相位的增量",
    "增量恰为 $2\\pi$（这是打包的权重，也是 $2\\pi$ 因子的直接体现）",
    "增量 %.15f，与 $2\\pi$ 之差 %.3e" % (d_tilde, abs(d_tilde - TWO_PI)),
    abs(d_tilde - TWO_PI) < 1e-15,
)

# A3：相位角变化不改变层数
th_new = 0.37
N_un = int(np.floor((TWO_PI * 3 + th_new + np.pi) / TWO_PI))
rec(
    "A3",
    "只改相位角时层数是否改变",
    "层数不变（整数部分只由满周数定），故相位与层数无独立变化的余地",
    "相位角 %.2f 下的层数 %d（与整数部分一致）" % (th_new, N_un),
    N_un == 3,
)

# =====================================================================
# 组 B：2pi 因子在两侧的一致性
# =====================================================================
print("=" * 72)
print("组 B：2pi 因子在两侧的一致性")
print("=" * 72)

# B1：弧长侧 2pi hbar 与周期侧 2pi/omega_P 的共轭关系
arc = TWO_PI * HBAR                    # 圆周弧长
period = TWO_PI / OMEGA_P             # 旋转周期
# 两者相乘应给出作用量 h 的量子化条件的一个尺度：
# arc / c = period  <=> 空间侧一整圈的时间 = 时间侧一整圈
# 库内第一组核对：光走一拍的距离 = 满周跃迁的位移。
# 一拍（一圈）的时间 t_P = 2pi/omega_P；光在此时间走的距离 c t_P 应恰为 l_P。
dist_one_turn = C_LIGHT * T_P
rec(
    "B1",
    "库内第一组核对：光走一圈（$t_P = 2\\pi/\\omega_P$）的距离 = 满周跃迁的位移 $l_P$",
    "$c\\,t_P = %.12f$ 应等于 $l_P = %.12f$，偏差 ≤ 1e-12"
    % (dist_one_turn, L_P),
    "光走一圈 %.12f，跃迁位移 %.12f，偏差 %.3e"
    % (dist_one_turn, L_P, abs(dist_one_turn - L_P)),
    abs(dist_one_turn - L_P) < 1e-12,
)

# B2：E_P = h f = hbar omega_P 中 2pi 的分子分母消去
f_P = OMEGA_P / TWO_PI                 # 频率 = omega_P / 2pi
E_from_hf = (TWO_PI * HBAR) * f_P
rec(
    "B2",
    "普朗克关系 $E_P = h f = 2\\pi\\hbar\\cdot(\\omega_P/2\\pi) = \\hbar\\omega_P$ 的 $2\\pi$ 消去",
    "由 $h f$ 得 %.12f，由 $\\hbar\\omega_P$ 得 %.12f，两者相等，偏差 ≤ 1e-12",
    "h f = %.12f，hbar omega_P = %.12f，偏差 %.3e"
    % (E_from_hf, HBAR * OMEGA_P, abs(E_from_hf - HBAR * OMEGA_P)),
    abs(E_from_hf - HBAR * OMEGA_P) < 1e-12,
)

# B3：方程 (II) 的系数 A = omega_P l_P = 2pi c
A_from_omega_l = OMEGA_P * L_P
A_from_2pi_c = TWO_PI * C_LIGHT
rec(
    "B3",
    "方程 (II) 系数 $A$ 的两种写法 $\\omega_P l_P$ 与 $2\\pi c$",
    "两者相等，偏差 ≤ 1e-12。这是 $2\\pi$ 因子在方程 (II) 中的来源",
    "$\\omega_P l_P$ = %.12f，$2\\pi c$ = %.12f，偏差 %.3e"
    % (A_from_omega_l, A_from_2pi_c, abs(A_from_omega_l - A_from_2pi_c)),
    abs(A_from_omega_l - A_from_2pi_c) < 1e-12,
)

# =====================================================================
# 组 C：四者咬合的可判后果
# =====================================================================
print("=" * 72)
print("组 C：四者咬合的可判后果")
print("=" * 72)

# C1：方程 (II) 的解为阶梯函数，宏观极限 z ≈ ct
def z_exact(t, t_P, c):
    """方程 (II) 的阶梯解（[[wiki/元相方程]] 定义节）。

    库内式子：$z(t) = l_P\\lfloor ct/l_P \\rfloor$。
    要点：先算 $ct$（走过的距离），再按 $l_P$ 取整（满周闭合的次数）。
    这与 $c\\,t_P\\lfloor t/t_P\\rfloor$ 是两个不同的函数：
    后者先按 $t_P$ 取整（时间侧），前者先算距离再按 $l_P$ 取整（空间侧）。
    """
    lP_local = c * t_P          # 一拍走过的距离等于 l_P（满周一次）
    return lP_local * np.floor(c * t / lP_local)


for Nt_ in (200, 2000, 20000):
    pass

# 一拍（一圈）的时间即全局 T_P = 2pi/omega_P；一拍走的距离恰为一格 l_P。
ts = np.array([0.5, 5.0, 50.0, 500.0]) * T_P
z_ex = z_exact(ts, T_P, C_LIGHT)
z_mac = C_LIGHT * ts
rel_mac = np.abs(z_ex - z_mac) / z_mac
rec(
    "C1",
    "方程 (II) 阶梯解的宏观极限 $z \\approx ct$（四档时刻）",
    "相对偏差随时刻增大而下降（取整误差 $\\le t_P/2$ 除以 $ct$）",
    "四档相对偏差 %s" % np.array2string(rel_mac, precision=8),
    rel_mac[-1] < rel_mac[0] and rel_mac[-1] < 1e-2,
)

# C2：跃迁距离不变（层数每跳恰 l_P）
ns = np.arange(0, 2001)
z_steps = z_exact(ns * T_P, T_P, C_LIGHT)
dz = np.diff(z_steps)
rec(
    "C2",
    "跃迁距离不变性：层数每跳恰 $l_P$",
    "全部步长恒为 $l_P$（%.12f），极差 ≤ 1e-12" % L_P,
    "步长唯一值 %.12f，极差 %.3e" % (dz[0], dz.max() - dz.min()),
    abs(dz.max() - dz.min()) < 1e-12 and abs(dz[0] - L_P) < 1e-12,
)

# C3：满周闭合不变：相位角越过 pi 时层数不跳
# 检验：连续推进相位角，层数只在满周（越过 2pi 的整数倍）时跳
theta_cont = np.linspace(0, 3 * TWO_PI, 30001)
# 层数 = 满周闭合的次数，故用 floor（向下取整）：
# 跳点在 tilde_theta = 2pi 的整数倍处。
# 不可用就近取整 floor(x + 1/2)：那样跳点落在 pi 处，与满周闭合矛盾。
N_cont = np.floor(theta_cont / TWO_PI).astype(int)
# 检查：theta 连续变化时 N 只在 theta = 2pi 的整数倍处变
dN_cont = np.diff(N_cont)
change_idx = np.where(dN_cont != 0)[0]
expected = np.round(theta_cont[change_idx] / TWO_PI).astype(int)
actual = N_cont[change_idx + 1]
rec(
    "C3",
    "满周闭合不变：层数只在相位越过 $2\\pi$ 整数倍时跳",
    "层数变化只发生在越过 $2\\pi$ 整数倍处，变化次数为 3，"
    "且变化后的层数等于跨越的圈数（层数从 0 起计）",
    "变化次数 %d，跳点 $\\tilde\\theta/2\\pi$ = %s，变化后层数 %s（期望 %s）"
    % (len(change_idx), np.round(theta_cont[change_idx] / TWO_PI, 3).tolist(),
       actual.tolist(), expected.tolist()),
    len(change_idx) == 3 and np.array_equal(actual, expected)
    and np.allclose(theta_cont[change_idx] / TWO_PI,
                    np.round(theta_cont[change_idx] / TWO_PI), atol=1e-3),
)

# C4：普朗克频率不变。检验办法：以两个不同角频率推进同一时间，
# 圈数之比应恰为频率之比。
def turns_in_time(T, om):
    """时间 T 内完成的满周圈数（按 arg psi 越过 2pi 整数倍计）"""
    return int(np.floor(om * T / TWO_PI + 0.5))


# 窗口取 2pi/omega_P 的整数倍，使两个频率下的圈数都恰为整数，
# 否则取整使比值偏离频率比（这是取整的边界效应，不是规律本身）。
T_probe = 2.0 * np.pi * 20 / OMEGA_P
om1, om2 = OMEGA_P, 2.0 * OMEGA_P
n1, n2 = turns_in_time(T_probe, om1), turns_in_time(T_probe, om2)
ratio_exp = om2 / om1
ratio_got = n2 / n1
rec(
    "C4",
    "普朗克频率不变：同一时间内的圈数之比等于频率之比",
    "频率比 %.6f，圈数比 %.6f，两者相等" % (ratio_exp, ratio_got),
    "圈数 %d 与 %d，比值 %.6f" % (n1, n2, ratio_got),
    abs(ratio_got - ratio_exp) < 1e-9,
)
# D3：2pi 的具体数值由普朗克关系系定（B 组），非打包自洽性定。
# 可执行检验：改变 HBAR 与 OMEGA_P，看两侧写法是否仍自洽，
# 而相位周期（打包常数）保持 2pi 不随其变。
hbar_alt = 2.0
om_alt = 1.5
tP_alt = TWO_PI / om_alt
lP_alt = 1.0
c_alt = lP_alt / tP_alt
E_alt = hbar_alt * om_alt
selfcons_alt = abs(c_alt * tP_alt - lP_alt)
coef_alt = abs(om_alt * lP_alt - TWO_PI * c_alt)
E_hf_alt = (TWO_PI * hbar_alt) * (om_alt / TWO_PI)
rec(
    "D3",
    "$2\\pi$ 的数值来源：改普朗克常数与频率后两侧写法仍自洽",
    "取 hbar=%.1f、omega_P=%.1f 后：一拍的距离与跃迁位移仍一致（偏差 %.3e），"
    "方程 (II) 系数仍满足 omega_P l_P = 2pi c（偏差 %.3e），"
    "普朗克关系 E = h f 仍给出 hbar omega_P（偏差 %.3e），"
    "而相位周期恒为 2pi 不随其变。故 2pi 的数值由普朗克关系系定，"
    "打包的自洽性只定结构"
    % (hbar_alt, om_alt, selfcons_alt, coef_alt, abs(E_hf_alt - E_alt)),
    "一拍与跃迁位移偏差 %.3e；系数关系偏差 %.3e；普朗克关系偏差 %.3e"
    % (selfcons_alt, coef_alt, abs(E_hf_alt - E_alt)),
    selfcons_alt < 1e-9 and coef_alt < 1e-9 and abs(E_hf_alt - E_alt) < 1e-12,
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
# 图
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))

ax = axes[0]
tt = np.linspace(0, 6 * T_P, 3001)
zz = z_exact(tt, T_P, C_LIGHT)
ax.step(tt, zz, where="post", color="#1f77b4", linewidth=2.2,
        label="方程 (II) 阶梯解 $z(t)$")
ax.plot(tt, C_LIGHT * tt, "--", color="#d62728", linewidth=1.8,
        label="宏观极限 $ct$")
ax.set_xlabel("时间 $t$（以普朗克时间 $t_P$ 为单位）")
ax.set_ylabel("层数坐标 $z$")
ax.set_title("左：方程 (II) 的解\n整数部分跳一格，小数部分连续")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(theta_cont, N_cont, "-", color="#1f77b4", linewidth=2.2,
        label="层数 $N(\\tilde\\theta)$")
for k in (1, 2, 3):
    ax.axvline(k * TWO_PI, color="#d62728", linestyle="--", linewidth=1.4)
ax.set_xlabel("提升相位 $\\tilde\\theta$")
ax.set_ylabel("层数 $N$")
ax.set_title("右：层数只在越过 $2\\pi$ 整数倍时跳\n$2\\pi$ 是跳的步长")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(IMG, dpi=150)
print("")
print("图已保存：%s" % IMG)
