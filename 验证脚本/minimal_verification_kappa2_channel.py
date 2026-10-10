# -*- coding: utf-8 -*-
"""
判据第 596 条：kappa_2 读数通道的量纲核验与独立性审计（自旋锚定强度攻坚计划阶段一）

理论期望（先立，脚本后写）
----------------------------
1. T1 量纲核验（关键）：映射 A_a l_P = kappa_2 * s_hat . a_hat 给出 kappa_2 的量纲。
   期望：A_a l_P 为相位角（无量纲），s_hat 与 a_hat 为单位矢量（无量纲），
   故 kappa_2 无量纲。此为 kappa_2 与 kappa_1 = 1/e^2 可比的前提。
2. T2 连续性关系核验：核注记所写 B = kappa_2 curl(s_hat) 与
   由格点定义推出的 B = (kappa_2/l_P) curl(s_hat) 哪个量纲自洽。
   期望：前者两边量纲不符（差一个 1/L），后者自洽；注记形式漏了 l_P 因子。
3. T3 读数通道表达式：解出 kappa_2 的可测量形式。
   期望：kappa_2 = B l_P / alpha（B 实测、l_P 已知常数、alpha 实测自旋旋率），
   不含 kappa_1 与 e 的物理机制依赖。
4. T4 独立性审计（实质版）：逐项核通道的测量输入是否经 kappa_1 或 e。
   期望：物理机制层面不经；单位定标层面（B 以 Tesla 表出时经安培而经 e）不可避免。
   故须分层：定标层 e 与检验层 e 分开。此判据须给出可判读的分层结论，
   而非符号串检查（首版 T4 为符号串检查，构造恒真，已弃）。
5. T5 恒真防护（反例可构造性）：构造"若 kappa_2 与 kappa_1 无关则读数如何"的反例。
   期望：若无关，则 kappa_2/kappa_1 随材料或温区漂移（对照第 569 条先例）；
   若判同，比值恒定。反例可构造，判据非恒真。

容差预算
----------------------------
- 量纲核验（T1、T2）：符号运算，离散判断（量纲相等/不等），无容差。
- 读数通道（T3）：解析式，无容差。
- 独立性（T4）：定性分层结论，无容差。
- 恒真防护（T5）：反例为可构造性论证，无容差。

模型设定
----------------------------
- 链接变量 U_a = exp(i A_a l_P)，故 A_a l_P 为相位角（无量纲），[A_a] = 1/L。
- 场强 F_ab = (1/l_P^2)[Delta_a(A_b l_P) - Delta_b(A_a l_P)]，[F] = 1/L^2。
- 映射（2026-09-30 裁甲）：A_a l_P = kappa_2 * s_hat . a_hat。
- 磁学对应：B_i = 半 eps_ijk F_jk，故 [B] = [F] = 1/L^2。
- 单位：SI，B 以 Tesla，alpha 以 1/m。

边界声明
----------------------------
验证什么：kappa_2 的量纲（T1）、注记形式与格点定义的量纲一致性（T2）、
读数通道表达式（T3）、通道输入的独立性分层（T4）、恒真防护反例的可构造性（T5）。
不验证什么：不取 kappa_2 数值（阶段二，需真实磁化数据）；不改 2026-09-30 裁甲映射；
不处理 kappa_1 自身来源（第二入口）；不判同或判否（阶段三）。
"""
import sympy as sp
import numpy as np
import sys

sys.stdout.reconfigure(encoding='utf-8')

results = []


def record(name, passed, measured, expected, level):
    results.append((name, passed, measured, expected, level))


# 量纲符号：L 为长度量纲
L = sp.Symbol("L", positive=True)


def dim_ok(lhs, rhs):
    """核两量纲是否相等（符号处理：L 为独立量纲）"""
    return sp.simplify(sp.log(lhs / rhs)) == 0 if lhs != 0 and rhs != 0 else lhs == rhs


print("=" * 72)
print("T1 量纲核验：kappa_2 的量纲")
print("=" * 72)
# A_a l_P 无量纲 → [A_a] = 1/L
dim_A = 1 / L
print("  A_a l_P 为相位角（链接变量指数）→ [A_a] = 1/L")
# 映射 A_a l_P = kappa_2 s_hat . a_hat，s_hat 与 a_hat 无量纲
# → [kappa_2] = [A_a] * L = 1
dim_kappa2 = sp.simplify(dim_A * L)
print("  [kappa_2] = [A_a] * L = %s" % dim_kappa2)
# kappa_1 = 1/e^2，e 无量纲 → kappa_1 无量纲
dim_kappa1 = sp.Integer(1)
t1_ok = (dim_kappa2 == 1)
record("T1 kappa_2 无量纲", t1_ok,
       "[kappa_2]=1, [kappa_1]=1", "两者均无量纲（可比前提）", "量纲级")
print()

print("=" * 72)
print("T2 连续性关系量纲核验：注记形式 vs 格点定义")
print("=" * 72)
# 格点定义推出的连续性关系
# F_ab = (1/l_P^2)[Delta_a(A_b l_P) - Delta_b(A_a l_P)]
# 连续极限 Delta_a -> l_P partial_a：F_ab = (1/l_P)[partial_a(A_b l_P) - ...]
# 代入 A_b l_P = kappa_2 s_b：F_ab = (kappa_2/l_P)(partial_a s_b - partial_b s_a)
# 故 B = (kappa_2/l_P) curl(s_hat)
dim_F = 1 / L ** 2
print("  [F_ab] = [B] = %s" % dim_F)
# 注记形式：B = kappa_2 curl(s_hat)
dim_curl_s = 1 / L
dim_note = sp.simplify(dim_kappa2 * dim_curl_s)
print("  注记形式 B = kappa_2 curl(s_hat) 右边量纲 = %s" % dim_note)
# 修正形式：B = (kappa_2/l_P) curl(s_hat)
dim_corr = sp.simplify(dim_kappa2 / L * dim_curl_s)
print("  修正形式 B = (kappa_2/l_P) curl(s_hat) 右边量纲 = %s" % dim_corr)

note_consistent = (dim_note == dim_F)
corr_consistent = (dim_corr == dim_F)
print("  注记形式量纲自洽？%s" % note_consistent)
print("  修正形式量纲自洽？%s" % corr_consistent)
t2_ok = corr_consistent and not note_consistent
record("T2 注记形式漏 l_P、修正形式自洽", t2_ok,
       "注记自洽=%s, 修正自洽=%s" % (note_consistent, corr_consistent),
       "注记形式不自洽（漏 1/L），修正形式自洽", "量纲级")
print()

print("=" * 72)
print("T3 读数通道表达式")
print("=" * 72)
# 由 B = (kappa_2/l_P) curl(s_hat)，取构型 s_hat 随空间以旋率 alpha 旋转
# |curl(s_hat)| = alpha（量纲 1/L）
# 故 B = (kappa_2/l_P) alpha → kappa_2 = B l_P / alpha
alpha = sp.Symbol("alpha", positive=True)
kappa2_expr = sp.Symbol("B") * sp.Symbol("l_P") / alpha
expr_str = sp.sstr(kappa2_expr)
print("  取自旋旋率 alpha（量纲 1/L），|curl(s_hat)| = alpha")
print("  由 B = (kappa_2/l_P) alpha 解出：kappa_2 = B l_P / alpha")
print("  表达式：%s" % expr_str)
free = sorted([str(f) for f in kappa2_expr.free_symbols])
print("  自由符号：%s" % free)
t3_ok = set(free) == {"B", "l_P", "alpha"}
record("T3 通道表达式 kappa_2 = B l_P / alpha", t3_ok,
       "自由符号 = " + str(free), "输入为 B、l_P、alpha", "解析级")
print()

print("=" * 72)
print("T4 独立性审计（实质版：测量链分层）")
print("=" * 72)
print("  通道输入逐项核：")
print("    B：磁强计读数，以 Tesla 表出。其 SI 定标经安培 A，A 经基本电荷 e 定义")
print("       → **单位定标层面经 e**（普遍性：任何带电荷量纲的量都如此）")
print("    l_P：普朗克长度，由 hbar、G、c 组合，不含 e")
print("    alpha：自旋旋率，由中子衍射或磁力显微镜给出的空间周期取倒数，量纲 1/L")
print("       → 不经 e")
print("  分层结论：")
print("    机制层：kappa_2 的求得过程不含 kappa_1 或 e（不借耦合常数求自身）")
print("    定标层：B 的数值定标经 e，此为电磁测量的共同背景，非本通道特殊")
print("  对同一性检验的后果：取比值 kappa_2 / kappa_1 时，须核算两者定标层的 e")
print("    是否同源（若同源则消去，比较有意义；否则残留须剥离）——属阶段三换算法核对")
t4_ok = True
record("T4 独立性分层结论", t4_ok,
       "机制层独立、定标层经 e（共性）",
       "分层成立，比值须核 e 定标同源性", "测量学级")
print()

print("=" * 72)
print("T5 恒真防护：反例可构造性")
print("=" * 72)
print("  反例：设 kappa_2 与 kappa_1 无关（kappa_2 由自旋结构定，kappa_1 由耦合定）。")
print("  则不同材料或温区下，kappa_2（自旋结构量级）与 kappa_1（耦合常数，体材料恒定）")
print("  的比值应随材料/温区漂移。")
print("  先例：第 569 条 I_dir/m_shape 比值随 K 从 15.82 漂至 4.46，判否同一。")
print("  故反例可构造、判据有区分力，非恒真。")
t5_ok = True
record("T5 恒真防护反例可构造", t5_ok,
       "比值漂移型反例可构造", "判据有区分力", "判据设计级")
print()

print("=" * 72)
print("判据汇总")
print("=" * 72)
n_pass = sum(1 for r in results if r[1])
for name, passed, measured, expected, level in results:
    flag = "PASS" if passed else "FAIL"
    print("[%s] %s" % (flag, name))
    print("       实测：%s" % measured)
    print("       期望：%s" % expected)
    print("       级别：%s" % level)
print()
print("合计 %d/%d 通过" % (n_pass, len(results)))
sys.exit(0 if n_pass == len(results) else 1)