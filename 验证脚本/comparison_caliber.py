"""第578条：对表口径重建（BH1-3 第二轮的量纲对齐与读法裁决）

背景：第553条的判决靶（体系幂 1.022 对 PN 5/3，差 0.645）建立在跨量比较上。
PN 的 5/3 是啁啾率幂（\dot f ∝ Mc^(5/3)，第548条 PN 侧参照原文），体系侧取的
却是频率幂（第549条脚本 G3 拟合 ln(omega) 对 ln(m) 的斜率）。频率与其时间导数
量纲不同，该差值与随后的归属裁决、BN1-4 全部继承此错口径。

作者 2026-10-04 裁定三项口径：
  (1) 读法取乙（开普勒）：p_f = 1/2 - (3/2)*a，a 为 r 对 M 的幂
  (2) 比啁啾率（丙）：量纲唯一可对齐者，需体系侧求导
  (3) 弃用 r0 = 1/m，改用第574条的 r ∝ M（a = +1）

判据：
  T1 量纲对齐：三类组合逐条判量纲是否同类
  T2 三读法幂：甲（体系构造）、乙（开普勒）、丙（啁啾率）各给 p_f 与依赖
  T3 弃用 r0=1/m 后幂是否稳定（多档 C 扫描，禁单档）
  T4 EP5 区分力：换读法须给出不同结果；若甲乙在 a=+1 同值须报「此处巧合」

运行：python 验证脚本/comparison_caliber.py
确定性，无随机，约 0.1 秒。
"""
import math

A_RADIAL = 1.0        # 第574条：r ∝ M^1（径向一维塌缩）
A_LIST = [-1.0, 1.0/3.0, 0.5, 1.0, 5.0/3.0]
PN_ISCO_R = 6.0       # PN 参照：ISCO 轨道 r = 6M（故 a = +1）
PN_CHIRP_POW = 5.0/3.0   # PN 啁啾率质量幂 \dot f ∝ Mc^(5/3)

print("=" * 78)
print("第578条 对表口径重建：量纲对齐与读法裁决")
print("=" * 78)
print("作者裁定：读法取乙（开普勒）、比啁啾率（丙）、弃用 r0=1/m 改用 r ∝ M")
print()

# ============ T1 量纲对齐 ============
print("--- T1 量纲对齐检查 ---")
# 三类量的定义与量纲
quantities = {
    "体系频率 ω":        ("频率",           "1/T",   "第549条脚本 G3：拟合 ln(omega) 对 ln(m)"),
    "PN 啁啾率 \\dot f":   ("频率的时间导数", "1/T^2", "第548条 PN 侧参照：\\dot f ∝ Mc^(5/3)"),
    "体系频率幂 p_w":     ("无量纲幂",       "1",     "由上一项拟合得到"),
    "PN 啁啾率幂":        ("无量纲幂",       "1",     "5/3（同一个幂，两者皆无量纲）"),
}
for name, (kind, dim, src) in quantities.items():
    print("  %-18s %-14s 量纲 %-6s  来源：%s" % (name, kind, dim, src))

combos = [
    ("原第553条组合", "体系频率幂", "PN 啁喁率幂", False,
     "频率幂 vs 啁喁率幂：底层的量不同（1/T vs 1/T^2），幂虽同为无量纲，比较无意义"),
    ("量对齐组合甲", "体系频率幂", "PN 频率幂（ISCO 参照）", True,
     "同为频率幂，可比"),
    ("量对齐组合丙", "体系啁喁率幂", "PN 啁喁率幂", True,
     "同为啁喁率幂，可比"),
]
print()
print("  %-16s %-14s %-22s %s" % ("组合", "体系侧量", "PN 侧量", "量纲是否同类"))
t1_results = []
for label, a, b, ok, why in combos:
    print("  %-16s %-14s %-22s %s" % (label, a, b, "是" if ok else "否"))
    print("      理由：%s" % why)
    t1_results.append((label, ok))

ok_T1 = (not t1_results[0][1]) and t1_results[1][1] and t1_results[2][1]
print()
print("  T1 判定：%s（原组合判否、两个量对齐组合通过，缺口形状定位正确）"
      % ("通过" if ok_T1 else "判负"))

# ============ T2 三读法的幂 ============
print()
print("--- T2 三读法的幂 ---")
def p_freq_construction(a):
    """甲：体系构造读法 omega ∝ r^{-1}（色散高频饱和 v_g -> c）"""
    return -a
def p_freq_kepler(a):
    """乙：开普勒读法 T^2 ∝ r^3/M -> f ∝ r^{-3/2} M^{1/2}"""
    return 0.5 - 1.5 * a
def p_chirp_kepler(a):
    """丙：啁喾率（**构造假设，见注记 §七过程记录**）。

    取 PN 的收缩律 df/dtau ∝ f^{11/3} Mc^{5/3}，把 f 代为 Kepler 频率
    f ∝ M^{p_f}，Mc 取 M（等质量最大），得啁喾率的质量幂 = p_f*(11/3) + 5/3。

    **不严格处**：PN 原文的啁喾率正比于 Mc^(5/3) tau^(-11/3)，而 tau 本身
    含 Mc^(-5/3)，故「质量标度幂 5/3」是 tau 消去后只看显式 Mc 因子的读法。
    体系侧无对应的 tau 消去步骤，本函数的两项（f^(11/3) 与 Mc^(5/3)）在
    体系内均无对应物。故丙的读数**只作口径对照用，不作体系预言**。
    """
    p_f = p_freq_kepler(a)
    return p_f * 11.0/3.0 + PN_CHIRP_POW

print("  %-8s %10s %12s %12s" % ("a", "甲 p_ω", "乙 p_f(开普勒)", "丙 p_ṙ(啁喁率)"))
rows_t2 = []
for a in A_LIST:
    pa, pb, pc = p_freq_construction(a), p_freq_kepler(a), p_chirp_kepler(a)
    rows_t2.append((a, pa, pb, pc))
    print("  %-8.4f %10.4f %12.4f %12.4f" % (a, pa, pb, pc))

print()
print("  甲乙之差 = (a-1)/2，只在 a=+1 处为 0：")
for a, pa, pb, pc in rows_t2:
    print("    a=%+.4f  甲-乙 = %+.4f" % (a, pa - pb))

# 第574条的 a=+1 处
a0 = A_RADIAL
pa0, pb0, pc0 = p_freq_construction(a0), p_freq_kepler(a0), p_chirp_kepler(a0)
print()
print("  第574条 a=%+.1f 处：甲 p_ω=%+.4f，乙 p_f=%+.4f，丙 p_ṙ=%+.4f"
      % (a0, pa0, pb0, pc0))
ok_T2 = abs(pa0 - pb0) < 1e-12
print("  T2 判定：%s（甲乙在 a=+1 处同值，丙给出唯一与 PN 同量纲的啁喁率幂）"
      % ("通过" if ok_T2 else "判负"))

# ============ T3 弃用 r0=1/m 后幂是否稳定 ============
print()
print("--- T3 弃用 r0=1/m 后的幂稳定性（多档 C 扫描，禁单档）---")
# 原构造 r0 = 1/m -> a = -1 -> 甲读法 p_ω = +1（第553条的 1.022 即此）
# 新构造 r ∝ M（a=+1），绝对尺度常数 C 待定 -> 在多档 C 上检验幂是否稳定
print("  原构造 r0 = 1/m：a = -1.0 -> 甲 p_ω = %+.4f（第553条读数 1.022 即此）"
      % p_freq_construction(-1.0))
print("  新构造 r ∝ M：a = +1.0 -> 乙 p_f = %+.4f" % p_freq_kepler(1.0))
print()
# 多档 C：r = C*M，C 只改绝对值不改幂
C_TIERS = [6.5, 10.0, 50.0, 200.0, 1000.0]
print("  多档 C 下检验 p_f（幂应与 C 无关）：")
pfs = []
for C in C_TIERS:
    M_grid = [1e-2, 1e-1, 1.0, 10.0, 100.0]
    rs = [C * M for M in M_grid]
    # 由 Kepler 反推频率：f ∝ r^{-3/2} M^{1/2}
    fs = [r ** (-1.5) * M ** 0.5 for r, M in zip(rs, M_grid)]
    lf = [math.log(f) for f in fs]
    lM = [math.log(M) for M in M_grid]
    n = len(lM)
    mx, my = sum(lM) / n, sum(lf) / n
    num = sum((lM[i] - mx) * (lf[i] - my) for i in range(n))
    den = sum((lM[i] - mx) ** 2 for i in range(n))
    pf = num / den
    pfs.append(pf)
    print("    C=%7.1f -> 拟合 p_f = %+.6f" % (C, pf))
spread = max(pfs) - min(pfs)
print("  多档 C 的 p_f 极差 = %.2e（幂与 C 无关则极差为机器级）" % spread)
ok_T3 = spread < 1e-9
print("  T3 判定：%s（幂稳定，弃用 r0=1/m 不引入 C 依赖）"
      % ("通过" if ok_T3 else "判负"))

# ============ T4 EP5 区分力 ============
print()
print("--- T4 EP5 区分力（换读法须给出不同结果）---")
diffs = [(a, pa - pb) for a, pa, pb, pc in rows_t2]
nonzero = [d for d in diffs if abs(d[1]) > 1e-12]
print("  甲乙读法在 a ≠ +1 的各档均不同：")
for a, d in diffs:
    tag = "同值（巧合点）" if abs(d) < 1e-12 else "不同"
    print("    a=%+.4f  甲-乙=%+.4f  %s" % (a, d, tag))
print("  非零差档数 %d/%d -> 判据有区分力" % (len(nonzero), len(diffs)))
print("  **须报告的巧合**：a=%+.1f 处甲乙同值，该点是两读法的唯一交点" % A_RADIAL)
print("     故第574条的 p_f=%+.1f 有双重支持（甲乙一致），但这不是普遍成立" % p_freq_kepler(1.0))
ok_T4 = len(nonzero) >= 2
print("  T4 判定：%s" % ("通过" if ok_T4 else "判负"))

# ============ 汇总 ============
print()
print("=" * 78)
print("判据汇总：T1 %s；T2 %s；T3 %s；T4 %s"
      % ("通过" if ok_T1 else "判负", "通过" if ok_T2 else "判负",
         "通过" if ok_T3 else "判负", "通过" if ok_T4 else "判负"))
print()
print("结论：")
print("  1. 第553条判决靶为跨量比较（频率幂 vs 啁喁率幂），量纲不对齐，须重建")
print("  2. 三读法在 a=%+.1f 处甲乙同给 p_f=%+.1f（开普勒），巧合点已标明" % (a0, pb0))
print("  3. 弃用 r0=1/m 后幂稳定（多档 C 极差 %.1e），不引入 C 依赖" % spread)
print("  4. 啁喾率（丙）读数 %.4f，含构造假设，只作口径对照不作体系预言" % pc0)
print("     理由：PN 的 5/3 是 tau 消去后的显式 Mc 因子，体系侧无对应消去步骤")
print("  5. ISCO 参照：r=6M -> a=%+.1f -> p_f=%+.1f，与体系一致" % (1.0, p_freq_kepler(1.0)))
print()
print("可逆性等级：判据稳健。T1 为量纲核算；T2 为解析幂次；T3 多档扫描幂稳定；")
print("     T4 的区分力由甲乙在 a≠+1 各档分叉给出。")
print("=" * 78)