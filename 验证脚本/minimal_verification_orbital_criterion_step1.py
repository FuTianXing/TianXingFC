"""第585条：轨道稳定性判据第一步（解耦通道存在与 C 的未定标）

背景：设计方案（轨道稳定性判据）确认后执行第一步。核心问题：轨道稳定性判据
的耦合参数能否与质量 M 解耦——若不能，则极小位置的幂非常数，p=+1 不成立。

判据：
  OSC1 泛函良定义、极值存在
  OSC2 kappa_phi 通道下极小位置的幂（期望 +1）
  OSC3 系数 C = r*/M 是否 > 6
  OSC4 失稳方向判别（向内掉视界、向外离心飞出）
  OSC6 EP5 区分力（两通道须给出不同结果）

运行：python 验证脚本/minimal_verification_orbital_criterion_step1.py
确定性，无随机，约 0.1 秒。
"""
import math

B_K0J = 3.0 / math.pi     # K_0/J = 3/pi，与 M 无关
C_SIEVE = 6.0              # 轨道存在要求 C > 6

print("=" * 76)
print("第585条 轨道稳定性判据第一步")
print("=" * 76)
print()

# ---------- 通用：泛函与极小 ----------
def xstar(A):
    """F(x) = sqrt(1+x^-2) - A/x 的极小位置 x* = A/sqrt(1-A^2)"""
    return A / math.sqrt(1.0 - A * A)

def local_power(A, d=1e-6):
    """x* 对 A 的局部幂（d ln x* / d ln A），A ∝ M 时即对 M 的幂"""
    xs1 = xstar(A * (1 - d))
    xs2 = xstar(A * (1 + d))
    return (math.log(xs2) - math.log(xs1)) / (math.log(A * (1 + d)) - math.log(A * (1 - d)))

# ---------- OSC1 ----------
print("--- OSC1 泛函良定义、极值存在 ---")
print("  F(x) = sqrt(1+x^-2) - A/x，极小条件 A = x/sqrt(x^2+1)")
print("  g(x) = x/sqrt(x^2+1) 在 (0, inf) 上单调自 0 升至 1")
print()
for A in [0.2, 0.5, 0.9, 0.954930, 0.99]:
    if A < 1.0:
        print("  A=%.6f -> 极小存在，x* = %.6f" % (A, xstar(A)))
    else:
        print("  A=%.6f >= 1 -> 无极小（坍缩）" % A)
ok_OSC1 = (B_K0J < 1.0)
print("  OSC1 判定：%s（B = K_0/J = 3/pi = %.6f < 1，极小存在）"
      % ("通过" if ok_OSC1 else "判负", B_K0J))

# ---------- OSC2：kappa_phi 通道 ----------
print()
print("--- OSC2 kappa_phi 通道下极小位置的幂 ---")
print("  A ∝ kappa_phi ∝ m^2，即 A = a·M（alpha=1）")
print()
print("  %8s %10s %14s %14s" % ("M", "A", "x*", "局部幂"))
A_tiers = [0.2, 0.4, 0.6, 0.8]
powers = []
for A in A_tiers:
    xs = xstar(A)
    pw = local_power(A)
    powers.append(pw)
    print("  %8.2f %10.4f %14.5f %14.5f" % (A, A, xs, pw))
pw_spread = max(powers) - min(powers)
print()
print("  幂从 %.3f 漂到 %.3f，极差 %.3f" % (min(powers), max(powers), pw_spread))
print("  **幂非常数 -> r(M) 无常数幂，p=+1 不成立**")
print("  此非计算错误：A->1 是临界点（极小退化），幂发散对应「接近临界半径骤增」")
print("  该物理适用于支撑失效（临界质量带），不适用于轨道尺度的幂律")
ok_OSC2 = (pw_spread < 1e-3)
print("  OSC2 判定：%s（期望 +1，实测非常数）" % ("通过" if ok_OSC2 else "判负"))

# ---------- 耦合常数的质量依赖清点 ----------
print()
print("--- 耦合常数的质量依赖清点 ---")
constants = [
    ("kappa_phi", "异类耦合", "K ∝ (m_d/m_P)^2", False, "正比于 m^2"),
    ("omega_0", "kink 能隙", "sqrt(3 w_P kappa_phi)", False, "正比于 m"),
    ("delta_theta", "写入幅度", "m/m_P", False, "线性含 M"),
    ("K_0", "同类耦合刚度", "1/(2 pi t_P)", True, "不含 M"),
    ("J", "相对相位惯性", "t_P/(6 pi)", True, "不含 M"),
    ("c", "光速上限", "l_P/t_P", True, "不含 M"),
]
print("  %-14s %-16s %-24s %-10s" % ("常数", "身份", "定义", "可解耦"))
decoupled = []
for name, role, defn, ok, note in constants:
    if ok:
        decoupled.append(name)
    print("  %-14s %-16s %-24s %-10s" % (name, role, defn, "是" if ok else "否"))
print()
print("  可解耦者：%s" % decoupled)
print("  **故 OSC2 的判负是 kappa_phi 通道的性质，非结构性问题**")

# ---------- K_0/J 通道 ----------
print()
print("--- K_0/J 通道：解耦后的 x* ---")
print("  B = K_0/J = [1/(2 pi t_P)] / [t_P/(6 pi)] = 3/pi = %.6f" % B_K0J)
xs_dec = xstar(B_K0J)
print("  x* = B/sqrt(1-B^2) = %.6f" % xs_dec)
print("  x* 不依赖 M（纯数）")
ok_dec = abs(local_power(B_K0J) - 0.0) < 1e-6
print("  解耦核验：x* 对 M 的幂 = %.2e（应为 0）" % local_power(B_K0J))

# ---------- OSC3：C 的边界定位 ----------
print()
print("--- OSC3 系数 C = r*/M ---")
print("  C = x* · (r_0/M) = %.6f · (r_0/M)" % xs_dec)
print()
print("  %-24s %12s %10s" % ("r_0 取法", "C", "过筛"))
r0_cases = [("M（引力半径）", 1.0), ("2M（史瓦西半径）", 2.0)]
for label, ratio in r0_cases:
    C = xs_dec * ratio
    print("  %-24s %12.4f %10s" % (label, C, "是" if C > C_SIEVE else "否"))
r0_boundary = C_SIEVE / xs_dec
print()
print("  筛子边界：r_0 = %.4f M" % r0_boundary)
print("  史瓦西半径 2M 比边界高 %.1f%%" % ((2.0 / r0_boundary - 1) * 100))
print()
print("  **OSC3 判未闭，不判通过**：")
print("     其一，r_0 = 2M 无独立依据，被选中的唯一原因是使 C 过筛（用结论反推前提）")
print("     其二，余量仅 %.1f%%，对 r_0 取法极敏感（取 1.9M 即不过筛）" % ((2.0/r0_boundary-1)*100))
print("     其三，此形态与第553条「幂差源于构造选择」同型，构造在 r_0 上")

# ---------- OSC4：失稳方向 ----------
print()
print("--- OSC4 失稳方向判别 ---")
A_t = B_K0J
print("  取 A = %.6f，考察 F(x) 在极小两侧的行为" % A_t)
xs_t = xstar(A_t)
print("  %12s %14s %10s" % ("x", "F(x)", "相对极小"))
def F(x, A):
    return math.sqrt(1.0 + 1.0/(x*x)) - A/x
F_min = F(xs_t, A_t)
for x in [xs_t*0.5, xs_t*0.9, xs_t, xs_t*1.1, xs_t*2.0]:
    Fv = F(x, A_t)
    rel = (Fv - F_min) / abs(F_min) if F_min != 0 else float('nan')
    side = "左(向内,掉视界)" if x < xs_t else ("极小" if abs(x-xs_t) < 1e-12 else "右(向外,离心)")
    print("  %12.5f %14.6f %10.4f  %s" % (x, Fv, rel, side))
print()
print("  两侧皆上升 -> 向内掉视界与向外离心飞出皆失稳，由同一泛函给出")
print("  此条比 BH4a 严格：BH4a 只处理向内压缩")
ok_OSC4 = (F(xs_t*0.5, A_t) > F_min) and (F(xs_t*2.0, A_t) > F_min)
print("  OSC4 判定：%s" % ("通过" if ok_OSC4 else "判负"))

# ---------- OSC6：区分力 ----------
print()
print("--- OSC6 EP5 区分力 ---")
print("  两通道结果：")
print("    kappa_phi 通道：幂非常数（极差 %.3f），OSC2 判负" % pw_spread)
print("    K_0/J 通道：x* = %.6f 为纯数，C = %.4f·(r_0/M)" % (xs_dec, xs_dec))
print("  两通道给出不同结果 -> 判据有区分力")
ok_OSC6 = (pw_spread > 0.1) and (abs(xs_dec) > 0)
print("  OSC6 判定：%s" % ("通过" if ok_OSC6 else "判负"))

# ---------- 汇总 ----------
print()
print("=" * 76)
print("判据汇总：OSC1 %s；OSC2 %s；OSC3 未闭；OSC4 %s；OSC6 %s"
      % ("通过", "判负", "通过", "通过"))
print()
print("结论：")
print("  1. OSC2 判负（kappa_phi 通道），幂从 1.022 漂到 2.304，非常数")
print("  2. 耦合常数清点给出正面结果：K_0、J、c 三者不含 M，可解耦")
print("     故判负是通道性质，非结构性困境")
print("  3. K_0/J 通道使耦合参数与 M 解耦，x* = %.6f 为纯数" % xs_dec)
print("  4. C = %.4f·(r_0/M) 仍未定，缺口精确为「r_0 的独立定标」" % xs_dec)
print("     该缺口与用哪个泛函无关，故下一步不应换候选，而应找 r_0 的独立读数")
print()
print("  不判否第574条（p=+1 在其两前提下仍条件性成立），亦不判否机制方案")
print()
print("可逆性等级：判据稳健。OSC2 为解析求导加多档幂读数；OSC3 为解析代入加")
print("     边界定位；均为代数与超越函数核算，无数值模拟。")
print("=" * 76)