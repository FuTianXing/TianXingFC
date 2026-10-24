r"""amplitude_basis.py

判据 M1 至 M7：方向色散偏差的幅度基数 sum_a n_a^4 的三套取值口径、
两套 E_P 约定的配对规则，以及 L3 旧读数隐含基数的反解。

模型：三维最近邻色散 omega^2 = (4/t_P^2) sum_a sin^2(k_a l_P/2)，l_P = c = 1。
相位偏差 delta_omega/omega = -(1/24)(E/E_P^(t))^2 sum_a n_a^4
                        = -(pi^2/6)(E/E_P)^2 sum_a n_a^4，
其中 E_P^(t) = hbar/t_P、E_P = hbar*omega_P = 2pi E_P^(t)。

幅度基数 sum_a n_a^4 的三套取值：
  A 单方向（给定波矢，最不利方向 [100]）  s_A = 1
  B 同族三值平均（[100]/[110]/[111]）     s_B = (1 + 1/2 + 1/3)/3 = 11/18
  C 球面平均 <sum_a n_a^4> = 3<n_x^4>     s_C = 3/5

判据
M1 三套基数取值 = 1、11/18、3/5（比值 90:55:54，即 1 : 11/18 : 3/5）
M2 两套 E_P 约定在 s 相同处逐位同值（s_A 与 s_C 各测一次）
M3 系数与能标必须配对：1/24 配 E_P^(t)、pi^2/6 配 E_P
M4 混用口径相对配套口径偏高 4pi^2 = 39.478（逐能量）
M5 L3 旧读数反解隐含基数 s = 1，非 pi^2/6 或 3/5
M6 L3 新旧比值链：同能量跨口径 = 4pi^2；同口径跨能量 = 能量平方比
M7 自检锚：E = E_P^(t) 处 delta_omega/omega = 1/24 = 4.1667%（精确）

运行：python 验证脚本/amplitude_basis.py
确定性、无随机数，运行约 1 秒。
"""
import math

E_Pt_GeV = 1.220890e19                 # E_P^(t) = hbar/t_P（折算普朗克能量）
C2PI = 2.0 * math.pi
E_P_basis_GeV = C2PI * E_Pt_GeV        # 体系基准 E_P = hbar*omega_P = 2pi E_P^(t)

S_A = 1.0                              # 单方向（最不利 [100]）
S_B = (1.0 + 0.5 + 1.0 / 3.0) / 3.0    # 同族三值平均 = 11/18
S_C = 3.0 / 5.0                        # 球面平均 <sum_a n_a^4> = 3 <n_x^4>

CASES = [("LHC 13 TeV", 13e12), ("IceCube 1 PeV", 1e15),
         ("UHECR 1e20 eV", 1e20), ("3e20 eV", 3e20)]

# L3 材料旧读数（口径混用）与更正后读数（配套口径、s = 1）
OLD = [1.865e-30, 1.104e-26, 1.104e-16, 9.932e-16]
NEW = [4.724e-32, 2.795e-28, 2.795e-18, 2.516e-17]


def coeff_t(E):
    return (E / (E_Pt_GeV * 1e9)) ** 2 / 24.0


def coeff_b(E):
    return (E / (E_P_basis_GeV * 1e9)) ** 2 * math.pi ** 2 / 6.0


print("=" * 74)
print("M1 幅度基数 sum_a n_a^4 的三套取值")
print("=" * 74)
print("  A 单方向（最不利 [100]）        s_A = %.6f" % S_A)
print("  B 同族三值平均（[100]/[110]/[111]）s_B = %.6f (= 11/18)" % S_B)
print("  C 球面平均 <sum n_a^4>          s_C = %.6f (= 3/5)" % S_C)
print("  比值 s_A : s_B : s_C = %.4f : %.4f : %.4f" % (S_A, S_B, S_C))
print("  化作整数 90 : 55 : 54（即 1 : 11/18 : 3/5）")
print("  方向比恒为 1 : 1/2 : 1/3 = 6 : 3 : 2（与 s 的取值无关）")

print()
print("=" * 74)
print("M2 两套 E_P 约定同值性 / M3 系数配对")
print("=" * 74)
print("  %6s | %14s | %14s | %14s" % ("s", "折算(1/24)", "基准(pi^2/6)", "两者之比"))
for lab, s in (("s_A", S_A), ("s_C", S_C)):
    vt = coeff_t(1e20) * s
    vb = coeff_b(1e20) * s
    print("  %6s | %14.6e | %14.6e | %14.6f" % (lab, vt, vb, vb / vt))
print("  结论：同一 s 下两套约定逐位同值（比 = 1.000000），配对规则成立")
print("  配对规则：1/24 必须配 E_P^(t)；pi^2/6 必须配 E_P = 2pi E_P^(t)")

print()
print("=" * 74)
print("M4 混用口径相对配套口径的偏高倍数（应恒为 4pi^2）")
print("=" * 74)
print("  %14s | %14s | %14s | %14s" % ("情形", "配套(s=1)", "混用(s=1)", "偏高倍数"))
for name, E in CASES:
    vt = coeff_t(E)
    vm = (E / (E_Pt_GeV * 1e9)) ** 2 * math.pi ** 2 / 6.0
    print("  %14s | %14.4e | %14.4e | %14.4f" % (name, vt, vm, vm / vt))
print("  4pi^2 = %.6f" % (4.0 * math.pi ** 2))

print()
print("=" * 74)
print("M5 L3 旧读数反解隐含基数（旧值 = 混用口径 x s）")
print("=" * 74)
print("  %14s | %14s | %14s | %10s" % ("情形", "旧读数", "混用口径(s=1)", "隐含 s"))
for (name, E), old in zip(CASES, OLD):
    mix = (E / (E_Pt_GeV * 1e9)) ** 2 * math.pi ** 2 / 6.0
    print("  %14s | %14.4e | %14.4e | %10.6f" % (name, old, mix, old / mix))
print("  结论：隐含 s = 1.0000，即旧表用的就是单方向最不利基数 s = 1")
print("  换言之 L3 旧表的错误只有一处：pi^2/6 与 E_P^(t) 搭配，基数并无第二重错误")

print()
print("=" * 74)
print("M6 比值链核对")
print("=" * 74)
print("  同能量跨口径（旧/新，应 = 4pi^2 = 39.4784）：")
for (name, E), old, new in zip(CASES, OLD, NEW):
    print("    %14s : %.4f" % (name, old / new))
print("  同口径跨能量（PeV 到 1e20 eV，应 = (1e20/1e15)^2 = 1e10）：")
print("    新表 PeV -> 1e20 eV : %.4f" % (NEW[2] / NEW[1]))
print("    新表 1e20 eV -> 3e20 eV（应 = 9）：%.4f" % (NEW[3] / NEW[2]))

print()
print("=" * 74)
print("M7 自检锚：E = E_P^(t) 处")
print("=" * 74)
print("  配套口径 s = 1 ：delta_omega/omega = 1/24 = %.6f = %.4f%%" % (1.0 / 24.0, 100.0 / 24.0))
print("  混用口径       ：%.4f = %.1f%%（荒谬，作为反证）" % (math.pi ** 2 / 6.0, 100.0 * math.pi ** 2 / 6.0))
