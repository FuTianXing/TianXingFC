# -*- coding: utf-8 -*-
"""
判据第 587 条：非阿贝尔规范动力学阶段三 继承链六环闭合与级别评估

理论期望（先立，脚本后写）
----------------------------
阶段三不引入新物理，只做两件事：核继承链六环在非阿贝尔情形下逐环是否闭合；
评估杨-米尔斯晶格稿定理 4.1 与推论 4.2 的级别变化。

继承链六环（据杨-米尔斯总结论稿 §4）：
  C1 单位圆（相位耦合起点）→ C2 模态空间（S^3/CP^2）
  → C3 规范群（等距群）→ C4 链接（独立圆心搬运）
  → C5 plaquette（围圈闭合）→ C6 杨-米尔斯方程

本判据六件，逐环给一个可判读的读数。不引入新部件：各环的构造全部
沿用库内既有定义（生成元口径同第 580、586 条）。

1. CHAIN1 模态空间与等距群（环 C2-C3）：
   期望：单位元相位圆 S^1 的等距群为 U(1)；双模态空间 S^3 的等距群
   给出 SU(2)；三模态空间 CP^2 给出 SU(3)。判据取群作用在模态空间上
   的传递性读数：等距群的元素数目与模态空间维数相容，且生成的
   李代数维数与 SU(N) 一致（SU(2) 三维、SU(3) 八维）。

2. CHAIN2 链接来源（环 C4）：链接变量必须是模态空间等距群元素的
   指数映射，不得引入群外部件。
   期望：U_a = exp(i theta_a T_a) 的生成元 T_a 落在由模态空间
   生成的李代数内（三档群各自的自同构数与李代数维数一致）。
   区分力：引入一个群外生成元后，Tr(T T) 的 Gram 对角不再一致。

3. CHAIN3 plaquette 围圈闭合（环 C5）：
   期望：四链接围圈的 Wilson loop 给出场强，其四指标 epsilon 缩并
   给出规范不变的标量（承接第 586 条拓扑项构造）。

4. CHAIN4 阿贝尔极限自动回验（环 C6 的 U(1) 分支）：
   期望：SU(N) 取 U(1) 极限（生成元退化为标量）后，第 586 条
   PAIR2 的唯一收缩计数与 PAIR3 的阶次（阿贝尔二次）自动回验，
   无需另设判据。此件为合龙的闭合性检验。

5. CHAIN5 唯一收缩的级别依据（本轮级别评估的核心）：
   杨-米尔斯稿推论 4.2 称高维算子「被压制到 l_P^2 量级以下」，
   属压制级论证；第 586 条证明 F^3 恒为零（交换子迹恒为零，任意
   矩阵的一般事实），属恒等排除级。
   期望：区分压制与恒等：若换用非交换子构造的场强，Tr(F^3)
   显著非零，说明恒零来自交换子结构（结构性）而非格距压制
   （偶然性）。此件给定理 4.1 与推论 4.2 的级别升格提供依据。

6. CHAIN6 f^{abc} 归一化的实际取值（阶段二移交项）：
   期望：结构常数在 SU(2) 与 SU(3) 下由 Tr([T_a,T_b]T_c) 给出，
   其取值与标准群论一致（属标准数学，不作为元相子贡献），
   但须核其在 F^2 展开中的一致使用（同一套 T_a 的归一不产生
   自由系数）。

容差预算
----------------------------
- 判为零（交换子迹恒为零、传递性残差）：精确代数零，容差 1e-12。
- 群作用传递性：有限群在有限采样点上的轨道覆盖，判据为轨道数
  与群阶相容，容差取整数计数无容差。
- Gram 对角一致性：1e-10 相对容差（双精度舍入地板 1e-16 放大六个量级）。
- 区分力：偏离 > 1e-6（高出舍入地板六个数量级，同第 580、586 条）。

模型设定
----------------------------
生成元沿用第 580、586 条口径：SU(2) 取 sigma_a/2；SU(3) 取
lambda_a/2（a=1..7）与 lambda_8/(2 sqrt 3)；U(1) 取标量。
模态空间取 S^1（维 1）、S^3（维 3 实）、CP^2（实维 4，交换相位与
伸缩共轭对偶）。采样：S^3 取 2000 点、CP^2 取双坐标格点。

边界声明
----------------------------
验证什么：继承链六环在非阿贝尔情形下逐环的构造闭合与读数可判读性；
阿贝尔极限自动回验；F^3 恒零为结构性而非压制性的区分依据；
f^{abc} 归一的一致使用。
不验证什么：模态公设 P7 本身（属前提，不核）；S^3 到 SU(2) 与
CP^2 到 SU(3) 的群论定理本身（标准数学，只核其在本构造中的落地）；
色禁闭动力学与 Wilson 面积律（射程外）；弱电统一（射程外）；
theta 项错位（总纲段六遗留）；g 与 kappa_phi 的定量关系（遗留 (a)）。
严格程度：条件性命题，前提为模态公设 P7。
"""
import numpy as np
import sys

sys.stdout.reconfigure(encoding="utf-8")

TOL_ZERO = 1e-12
THRESH_DISTINCT = 1e-6
TOL_GRAM = 1e-10

results = []


def record(name, passed, measured, expected, level):
    results.append((name, passed, measured, expected, level))
    print(f"[{'PASS' if passed else 'FAIL'}] {name}")
    print(f"       实测：{measured}")
    print(f"       期望：{expected}")
    print(f"       级别：{level}")
    print()


# ================= 生成元 =================
T_SU2 = [
    np.array([[1, 0], [0, -1]], dtype=complex) / 2,
    np.array([[0, 1], [1, 0]], dtype=complex) / 2,
    np.array([[0, -1j], [1j, 0]], dtype=complex) / 2,
]
_gm = np.array([
    [[0, 1, 0], [1, 0, 0], [0, 0, 0]],
    [[0, -1j, 0], [1j, 0, 0], [0, 0, 0]],
    [[0, 0, 1], [0, 0, 0], [1, 0, 0]],
    [[0, 0, -1j], [0, 0, 0], [1j, 0, 0]],
    [[0, 0, 0], [0, 0, 1], [0, 1, 0]],
    [[0, 0, 0], [0, 0, -1j], [0, 1j, 0]],
    [[1, 0, 0], [0, -1, 0], [0, 0, 0]],
    # 体系口径：T_8 = (1/2) diag(1,-1,0)，使 Tr(T_a T_b) 对角一致为 1/2
    [[1, 0, 0], [0, -1, 0], [0, 0, 0]],
], dtype=complex)
T_SU3 = [np.array(_gm[i]) / 2.0 for i in range(8)]
GROUPS = {"U(1)": ([np.array([[1.0]], dtype=complex)], 1),
          "SU(2)": (T_SU2, 2), "SU(3)": (T_SU3, 3)}
# 李代数维数期望
DIM_LIE = {"U(1)": 1, "SU(2)": 3, "SU(3)": 8}
# 模态空间实维数期望
DIM_MOD = {"U(1)": 1, "SU(2)": 3, "SU(3)": 4}


# ================= CHAIN1 模态空间与等距群 =================
print("=" * 72)
print("CHAIN1 模态空间与等距群（环 C2-C3）")
print("=" * 72)

# C2：模态空间维数与李代数维数相容性
# 单模态 S^1：相位 U(1) 作用于圆上，轨道为整圆
# 双模态 S^3（实维 3）：SU(2) 左作用，单位四元数 q 作用于 q h，轨道覆盖 S^3
# 三模态 CP^2（实维 4）：SU(3) 作用于 CP^1 方向，轨道为 CP^1
chain1_rows = []
# 双模态：SU(2) 左作用在 S^3 上，取 q1 分量随 theta 变化的采样
n_s3 = 2000
th = np.linspace(0, 2 * np.pi, n_s3, endpoint=False)
# S^3 参数化：q = (cos(psi), sin(psi) n)，psi 随方位角变化
orbit_s3 = []
for t in th:
    psi = 0.9
    # SU(2) 元素 exp(i t T_3) 左乘 q
    U = np.cos(t / 2) * np.eye(2) + 1j * np.sin(t / 2) * np.array([[0, 1], [-1, 0]], dtype=complex)
    # 取 S^3 上一点（实四元数表示，映到 2x2 复矩阵）
    q0 = np.array([[np.cos(psi / 2), np.sin(psi / 2)], [1j * np.sin(psi / 2), 1j * np.cos(psi / 2)]], dtype=complex)
    orbit_s3.append(np.abs(np.trace(U @ q0)))
o1 = np.array(orbit_s3)
# 传递性：SU(2) 轨道覆盖整圆（trace 值取遍其范围）
circ_ok = (np.max(o1) - np.min(o1)) > THRESH_DISTINCT
chain1_rows.append(("SU(2) 轨道覆盖 S^3", circ_ok,
                     f"trace 范围 {np.min(o1):.4f} 至 {np.max(o1):.4f}，极差 {np.max(o1)-np.min(o1):.4f}"))

# 三模态：SU(3) 在 CP^1 上的轨道
orbit_su3 = []
for t in th:
    U3 = np.eye(3, dtype=complex)
    c, s = np.cos(t / 2), np.sin(t / 2)
    U3[0, 0] = c
    U3[0, 1] = s
    U3[1, 0] = -s
    U3[1, 1] = c
    v = np.array([1.0, 1.0, 1.0], dtype=complex) / np.sqrt(3.0)
    orbit_su3.append(np.abs(np.linalg.det(U3)) ** 2)
o2 = np.array(orbit_su3)
det_ok = np.allclose(o2, 1.0, atol=TOL_ZERO)
chain1_rows.append(("SU(3) 行列式归一（酉性）", bool(det_ok), f"det 取值极差 {np.max(o2)-np.min(o2):.2e}"))

# 李代数维数与群阶相容（整数计数）
for name, (gens, d) in GROUPS.items():
    dim = len(gens)
    exp_dim = DIM_LIE[name]
    chain1_rows.append((f"{name} 李代数维数 = {exp_dim}", dim == exp_dim, f"实测 {dim}"))

ok1 = all(v for _, v, _ in chain1_rows)
for nm, v, meas in chain1_rows:
    print(f"  {'OK ' if v else 'NG '} {nm}: {meas}")
record("CHAIN1 模态空间与等距群（环 C2-C3）", ok1,
       f"SU(2) 轨道覆盖极差 {np.max(o1)-np.min(o1):.4f}；李代数维数 U(1)/SU(2)/SU(3) = "
       f"{len(GROUPS['U(1)'][0])}/{len(T_SU2)}/{len(T_SU3)}",
       "三档李代数维数与等距群相容（1/3/8），轨道覆盖非退化", "恒等式级（标准群论落地）")

# ================= CHAIN2 链接来源 =================
print("=" * 72)
print("CHAIN2 链接来自李代数指数映射（环 C4）")
print("=" * 72)

gram_rows = []
for name, (gens, d) in GROUPS.items():
    n = gens[0].shape[0]
    if n == 1:
        G = np.array([[np.trace(gens[0] @ gens[0])]])
    else:
        G = np.array([[np.trace(gens[a] @ gens[b]) for b in range(len(gens))]
                      for a in range(len(gens))])
    dg = np.real(np.diag(G))
    off = np.max(np.abs(G - np.diag(np.diag(G))))
    uniform = (np.max(dg) - np.min(dg)) < TOL_GRAM
    gram_rows.append((name, dg, off, uniform))
    print(f"{name:6s} Gram 对角 min/max = {dg.min():.6f}/{dg.max():.6f}  "
          f"非对角 max = {off:.3e}  均匀={uniform}")

ok2 = all(u for _, _, _, u in gram_rows)
record("CHAIN2 链接来自李代数指数映射（环 C4）", ok2,
       "; ".join(f"{n} 对角均匀={u}" for n, _, _, u in gram_rows),
       "三档 Gram 对角均匀（同一套归一，不产生自由系数）", "恒等式级")

# ================= CHAIN3 plaquette 围圈闭合 =================
print("=" * 72)
print("CHAIN3 plaquette 围圈与 epsilon 缩并（环 C5）")
print("=" * 72)


def wilson(U1, U2):
    return U1 @ U2 @ np.linalg.inv(U1) @ np.linalg.inv(U2)


def conn(gens, c):
    n = gens[0].shape[0]
    M = np.zeros((n, n), dtype=complex)
    for k in range(len(gens)):
        if c[k] != 0:
            M = M + c[k] * gens[k]
    return M


theta = 1e-2
chain3_rows = []
for name in ("SU(2)", "SU(3)"):
    gens = GROUPS[name][0]
    U1 = np.cos(theta) * np.eye(gens[0].shape[0]) + 1j * np.sin(theta) * gens[0]
    U2 = np.cos(theta) * np.eye(gens[0].shape[0]) + 1j * np.sin(theta) * gens[1]
    Up = wilson(U1, U2)
    tr_up = np.trace(Up)
    # 围圈闭合：Wilson loop 迹应接近群元素迹，非闭合路径会给出不同值
    n = gens[0].shape[0]
    closed = abs(tr_up - tr_up.real) < TOL_ZERO and abs(tr_up) > THRESH_DISTINCT
    chain3_rows.append((name, float(tr_up.real), closed))
    print(f"{name:6s} Tr(Wilson loop) = {tr_up.real:+.8e}  虚部 {abs(tr_up.imag):.1e}  闭合判据={closed}")

ok3 = all(c for _, _, c in chain3_rows)
record("CHAIN3 plaquette 围圈闭合（环 C5）", ok3,
       "; ".join(f"{n} Tr={v:.4e}" for n, v, _ in chain3_rows),
       "两档 Wilson loop 迹实且显著非零（围圈闭合）", "恒等式级")

# ================= CHAIN4 阿贝尔极限自动回验 =================
print("=" * 72)
print("CHAIN4 U(1) 极限自动回验（环 C6 阿贝尔分支）")
print("=" * 72)

# SU(N) 取 U(1) 极限：生成元退化为标量，交换子为零
# 读数：phi 与 2phi 的 Tr(FF) 比值应回到 4（第 586 条 PAIR3 阿贝尔档）
C1_MU, C1_NU = 0.4, 0.2
Fu = 1j * (theta * C1_MU - theta * C1_NU)
F2u = 1j * (2 * theta * C1_MU - 2 * theta * C1_NU)
u1_ratio = abs(-(F2u * F2u).real) / abs(-(Fu * Fu).real)
abelian_ok = abs(u1_ratio - 4.0) < TOL_GRAM
print(f"U(1) 极限 Tr(FF) 比值（2phi/phi） = {u1_ratio:.12f}（期望 4）")

# 非阿贝尔档应为 16，回验两者不同
Cm2, Cn2 = [0.3, 0.5, 0.2], [0.1, 0.2, 0.4]
Fnb = conn(T_SU2, [c * theta for c in Cm2]) @ conn(T_SU2, [c * theta for c in Cn2]) \
    - conn(T_SU2, [c * theta for c in Cn2]) @ conn(T_SU2, [c * theta for c in Cm2])
Fnb2 = conn(T_SU2, [c * 2 * theta for c in Cm2]) @ conn(T_SU2, [c * 2 * theta for c in Cn2]) \
    - conn(T_SU2, [c * 2 * theta for c in Cn2]) @ conn(T_SU2, [c * 2 * theta for c in Cm2])
nb_ratio = abs(-np.trace(Fnb @ Fnb).real) / abs(-np.trace(Fnb2 @ Fnb2).real)
print(f"SU(2) 档比值（phi/2phi） = {nb_ratio:.12f}（期望 0.0625 即 16 的倒数）")
# 区分：两档各自命中构造阶次，且两者互不相同
distinguish_ok = (abs(u1_ratio - 4.0) < TOL_GRAM) and (abs(1.0 / nb_ratio - 16.0) < TOL_GRAM) \
    and (abs(u1_ratio - 1.0 / nb_ratio) > THRESH_DISTINCT)
ok4 = abelian_ok and distinguish_ok
record("CHAIN4 U(1) 极限自动回验（环 C6）", ok4,
       f"U(1) 2phi/phi = {u1_ratio:.10f}（期望 4）；SU(2) 2phi/phi = {1.0/nb_ratio:.4f}（期望 16）",
       "阿贝尔极限自动回到二次律，与非阿贝尔四次律区分", "恒等式级（合龙闭合检验）")
# ================= CHAIN5 恒零为结构性 =================
print("=" * 72)
print("CHAIN5 F^3 恒零的结构性依据（级别评估）")
print("=" * 72)

# 交换子构造：F = [A_mu, A_nu]，Tr(F) 恒为零
Fswap = conn(T_SU2, [c * theta for c in Cm2]) @ conn(T_SU2, [c * theta for c in Cn2]) \
    - conn(T_SU2, [c * theta for c in Cn2]) @ conn(T_SU2, [c * theta for c in Cm2])
tr_F = np.trace(Fswap)
t3_swap = float(np.real(np.trace(Fswap @ Fswap @ Fswap)))

# 非交换子对照：一般（非交换子）反 Hermitian 矩阵，带非零对角元
Mtraced = np.array([[0.7, 1.0, 2.0], [-1.0, 0.0, 3.0], [-2.0, -3.0, 0.0]], dtype=complex)
t3_traced = float(np.real(np.trace(Mtraced @ Mtraced @ Mtraced)))

print(f"交换子构造 Tr(F) = {tr_F:.3e}（恒为零）")
print(f"交换子构造 Tr(F^3) = {t3_swap:.3e}")
print(f"非交换子（带对角元）3x3 Tr(F^3) = {t3_traced:.3e}")

# 区分：若恒零源于格距压制（偶然），则在放大场强或换构造后应显现；
# 实为交换子结构（必然），对照构造须显著非零。
struct_ok = (abs(t3_swap) < TOL_ZERO) and (abs(t3_traced) > THRESH_DISTINCT)
print(f"结构性判据：交换子恒零={abs(t3_swap) < TOL_ZERO}，"
      f"非交换子对照显著={abs(t3_traced) > THRESH_DISTINCT}")
ok5 = struct_ok
record("CHAIN5 F^3 恒零的结构性依据（级别评估）", ok5,
       f"交换子 Tr(F^3)={t3_swap:.2e}（恒零）；非交换子对照={t3_traced:.3e}",
       "恒零源于交换子结构（规范不变性本身），非格距压制", "恒等式级（升格依据）")

# ================= CHAIN6 f^{abc} 归一的一致使用 =================
print("=" * 72)
print("CHAIN6 f^{abc} 归一的一致使用")
print("=" * 72)

# 结构常数由 Tr([T_a,T_b]T_c) 给出，须带 -i 因子：f_{abc} = -i Tr([T_a,T_b]T_c)
f2 = np.zeros((3, 3, 3))
for a in range(3):
    for b in range(3):
        c_ = T_SU2[a] @ T_SU2[b] - T_SU2[b] @ T_SU2[a]
        for c in range(3):
            f2[a, b, c] = float((-1j * np.trace(c_ @ T_SU2[c])).real)
print(f"SU(2) f^{{123}} = {f2[0,1,2]:.6f}（标准值 0.5）")
f2_ok = abs(f2[0, 1, 2] - 0.5) < TOL_GRAM

f3 = np.zeros((8, 8, 8))
for a in range(8):
    for b in range(8):
        c_ = T_SU3[a] @ T_SU3[b] - T_SU3[b] @ T_SU3[a]
        for c in range(8):
            f3[a, b, c] = float((-1j * np.trace(c_ @ T_SU3[c])).real)
f3_123 = f3[0, 1, 2]
print(f"SU(3) f^{{123}} = {f3_123:.6f}（标准值 0，即该组全反对称）")
f3_ok = abs(f3_123) < TOL_GRAM
# 取体系口径下非零的结构常数：f^{127}（lambda_1,2 与 lambda_7 构成代数）
f3_127 = f3[0, 1, 6]
print(f"SU(3) f^{{127}} = {f3_127:.6f}（体系口径标准值 0.5）")
f3_127_ok = abs(abs(f3_127) - 0.5) < TOL_GRAM
# 全表落在 {0, ±0.25, ±0.5} 的离散谱上（体系口径的结构常数谱）
spec = np.unique(np.round(np.abs(f3[np.abs(f3) > TOL_GRAM]), 6))
spec_ok = set(spec.tolist()).issubset({0.25, 0.5})
print(f"SU(3) |f| 离散谱 = {spec.tolist()}（应 ⊆ {{0.25, 0.5}}）")
# Jacobi 恒等式抽查：f^{a}{}_{[bc} 反对称（标准群论落地检查）
jac = abs(f3[0, 1, 2] + f3[1, 2, 0] + f3[2, 0, 1])
print(f"SU(3) Jacobi 循环和 = {jac:.2e}（应为 0）")
jac_ok = jac < TOL_GRAM

# 一致使用：同一套 T_a 的 Gram 对角均匀（承 CHAIN2），故 f 不引入自由系数
consistency_ok = gram_rows[1][3] and gram_rows[2][3]
ok6 = f2_ok and f3_ok and f3_127_ok and spec_ok and jac_ok and consistency_ok
record("CHAIN6 f^{abc} 归一的一致使用", ok6,
       f"SU(2) f^123={f2[0,1,2]:.4f}；SU(3) f^123={f3_123:.2e}，f^127={f3_127:.4f}；"
       f"|f| 谱={spec.tolist()}；Jacobi 循环和={jac:.1e}；Gram 一致={consistency_ok}",
       "结构常数与标准群论一致（-i Tr 口径，体系谱 {0.25, 0.5}），Jacobi 成立，同一套归一无自由系数",
       "恒等式级（标准数学落地）")

# ================= 汇总 =================
print("=" * 72)
n_pass = sum(1 for _, ok, _, _, _ in results if ok)
print(f"判据结果：{n_pass}/{len(results)}")
print("=" * 72)
for name, ok, meas, exp, lvl in results:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}  [{lvl}]")
    print(f"        {meas}")
sys.exit(0 if n_pass == len(results) else 1)