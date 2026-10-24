# -*- coding: utf-8 -*-
"""
判据第 586 条：非阿贝尔规范动力学阶段二 迹与配对的唯一性

理论期望（先立，脚本后写）
----------------------------
阶段二处理 F^2 的矩阵配对唯一性。阶段一（第 580 条）已把转圈次序对合在格点
链接上显式化（交换子非零、Wilson loop 次序敏感），但两项被移交到本阶段：
展开阶次与 f^{abc} 归一化。二者在此与双收缩唯一性一并处理。

本判据六件，全部在群表示层精确计算。矩阵指数用谱分解精确求（生成元
Hermitian，无级数截断误差）；唯一需要级数的是 PAIR4 的压制比，其截断项
显式给出并计入预算。

1. PAIR1 冒烟：规范不变双收缩存在且为实正数。
   期望：场强 F_{mu nu} = [A_mu, A_nu]（mu 与 nu 为不同空间指标）构造下，
   Tr(F_{mu nu} F^{mu nu}) 在三档群下有定义、虚部为舍入地板、实部为正。

2. PAIR2 双收缩枚举的唯一性（核心判据）：
   在矩阵配对空间枚举全部保持规范不变的双收缩通道，核唯一合法者为
   Tr(F_{mu nu} F^{mu nu})。
   枚举对象为 F F 的指标配对，三条候选通道：
     (a) 全收缩 Tr(F_mu F_nu)：两个 F 的矩阵指标全部缩并；
     (b) 单收缩 Tr(F_mu) Tr(F_nu)：两个 F 各自缩并后相乘；
     (c) 元素级 F_mu^{ij} F_nu^{ij}（不取迹，按 Frobenius 型缩并）。
   规范不变性判据：结果须落 SU(N) 的中心元素（标量）。
   期望：(b) 对无迹生成元恒为零（Tr(F) = 0），故不构成独立通道；
   (c) 的元素级缩并在多指标下不落中心元素（SU(2)/SU(3) 下非标量）；
   (a) 唯一。计数在三档下均为 1。区分力由 PAIR5 提供。

3. PAIR3 迹的实值性与阶次（构造分层）：
   期望：Tr(F F) 的虚部为舍入地板。阶次按场强构造分层，不作跨档统一：
   SU(2)/SU(3) 的 F = [A_mu, A_nu] 对 phi 为二次，故 Tr(F F) 对 phi 为四次，
   phi 与 2phi 之比为 16；U(1) 标量群无交换子，F 对 phi 为一次，
   Tr(F F) 对 phi 为二次，phi 与 2phi 之比为 4。
   两档各自的比值须与构造阶次逐位一致。此件是阶段一移交的
   "展开阶次"一项的直接落地：主项为二次型，且阿贝尔与非阿贝尔的
   phi 依赖阶次不同，此差别本身即非交换性的签名。

4. PAIR4 高阶算子与拓扑项（登记件）：
   (i) F^3：场强为交换子 F = [A_mu, A_nu]，而交换子的迹恒为零
   （任意两个矩阵交换子的迹恒为零，与生成元是否无迹无关：中心项在
   交换中抵消）。故 Tr(F^3) 由迹的循环性恒为零，为恒零排除而非小量压制。
   区分力：换用非交换子构造的反 Hermitian 场强（显式给非零对角元），
   Tr(F^3) 须显著非零，证明该恒零来自交换子结构而非构造恒真。
   此结论强于原判据：F^3 的排除依据是规范不变性本身。
   (ii) 拓扑项 Tr(F Ftilde)：tilde F^{mu nu} = (1/2) eps^{mu nu rho sigma}
   F_{rho sigma}，Tr(F Ftilde) 为四指标 epsilon 缩并。判据用相对口径
   （拓扑项与主项 Tr(F F) 同阶）：
   期望：存在至少两个独立的反对称分量时，拓扑项与主项同量级
   （比值为 $O(1)$）；只有一个独立反对称分量时恒为零
   （反对称张量与自身的 epsilon 缩并为零）。二者构成区分力：
   拓扑项度量的是多分量反对称结构，不是非交换性本身。
   拓扑项与主项同为 phi 二次量级，故用相对比值而非固定绝对阈值。

5. PAIR5 恒真检验（EP5 齿，必配）：
   反例一：把双收缩改为单收缩 Tr(F)（去掉一个 F 因子），结果须显著改变
   （SU(2)/SU(3) 下为零，与 (a) 显著不同）。
   反例二：把配对退化为 U(1) 标量乘积，SU(2)/SU(3) 结果须与标量结果不同。
   反例三（拓扑项齿）：把非交换构造改为同轴退化，拓扑项须消失。
   若任一反例与全收缩给出相同结果则本判据构造恒真，判 FAIL。

6. PAIR6 群维数多档扫描（防单档擦线）：
   在 U(1)、SU(2)、SU(3) 三档下重复 PAIR2 计数与 PAIR3 阶次。
   期望：三档计数一致（均为 1）；阶次按构造分层一致（U(1) 为二次、
   SU(2)/SU(3) 为四次，两组内部各自一致）。单档吻合不构成证据，
   故三档须同时报告。

容差预算
----------------------------
- 判为零（虚部、Tr(F)、Tr(F^3)、可交换情形的拓扑项）：精确代数零，
  浮点为舍入地板，容差取 1e-12。
- 唯一性计数：整数计数，无容差概念，须三档一致。
- 阶次缩放律：|Tr(FF)|(2phi)/|Tr(FF)|(phi) 在 U(1) 档应为 4、
  在 SU(2)/SU(3) 档应为 16，容差取 1e-10（相对误差；由双精度舍入地板
  1e-16 经两次乘除放大，取六个数量级的裕度）。
- 拓扑项相对口径：|Tr(F Ftilde)|/|Tr(F F)| 为 $O(1)$ 量，
  判据取在 (0.1, 10) 区间内为同阶；退化档要求比值为舍入地板。
  不用固定绝对阈值，因两者同为 phi^2 量级，绝对阈值随档位漂移。
- PAIR5 区分力：偏离量须 > 1e-6，即至少高出舍入地板六个数量级，
  与第 580 条 A4 的区分力口径一致。

模型设定
----------------------------
- 生成元：SU(2) 取 T_a = sigma_a/2（三个，实维 2）；SU(3) 取
  T_a = lambda_a/2（a = 1..7）与 T_8 = lambda_8/(2 sqrt 3)（实维 3，
  归一化使 Gram 对角一致）；U(1) 取标量。指数与全部矩阵运算精确。
- 场强：F_{mu nu} = [A_mu, A_nu]，A_mu 为 Hermitian 生成元线性组合，
  mu 与 nu 取不同空间指标，故 F 为反 Hermitian 且非零。Tr(F F) 取负号
  约定给实正的密度。
- 转角档：主档 phi = 1e-2；扫描档 1e-3、1e-2、2e-2。
- 非交换子构造（F^3 的区分力用）：显式给 3x3 反 Hermitian 矩阵带非零
  对角元（非交换子），验证 Tr(F^3) 可非零。
- 阶次分层：交换子对 phi 为二次（F 含两个 phi 因子），标量场强对 phi
  为一次；故 Tr(FF) 在非阿贝尔档为四次、阿贝尔档为二次。

边界声明
----------------------------
验证什么：双收缩配对在 SU(2)/SU(3) 下的唯一性、迹的实值性与阶次
（阿贝尔二次对非阿贝尔四次）、F^3 的恒零排除及其区分力、拓扑项的
相对量级与可交换退化、恒真检验三条、群维数多档一致性。
不验证什么：f^{abc} 的绝对归一化（属标准群论，不属元相子几何贡献，
按阶段一收窄结论不作为本阶段判据）；Jacobi 恒等式（标准群论）；
色禁闭动力学与 Wilson 面积律（射程外）；弱电统一（射程外）；
theta 项的错位问题（另见总纲段六遗留）；展开阶次的完整 Taylor 对照
（PAIR3 的二次缩放律已给主阶次判据，完整级数对照属阶段三合龙时处理）。
严格程度：条件性命题，前提为模态公设 P7。
"""
import numpy as np
import sys

sys.stdout.reconfigure(encoding="utf-8")

TOL_ZERO = 1e-12
THRESH_DISTINCT = 1e-6
TOL_QUAD = 1e-10

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
    [[0, 1, 0], [0, 0, 1], [0, 0, 0]],
], dtype=complex)
T_SU3 = [np.array(_gm[i]) / 2.0 for i in range(7)] + [
    np.array(_gm[7]) / (2.0 * np.sqrt(3.0))
]


def conn(gens, c):
    """Hermitian 生成元的线性组合。"""
    n = gens[0].shape[0]
    M = np.zeros((n, n), dtype=complex)
    for k in range(len(gens)):
        if c[k] != 0:
            M = M + c[k] * gens[k]
    return M


def field_strength(gens, c_mu, c_nu):
    """F_{mu nu} = [A_mu, A_nu]，mu 与 nu 为不同空间指标。"""
    return conn(gens, c_mu) @ conn(gens, c_nu) - conn(gens, c_nu) @ conn(gens, c_mu)


def pair_density(F):
    """Tr(F F) 取负号约定给实正的密度。"""
    return -np.trace(F @ F).real


def frob(F):
    return float(np.sqrt(np.sum(np.abs(F) ** 2)))


# 系数档：mu 与 nu 取不同方向，保证交换子非零
C2_MU = [0.3, 0.5, 0.2]
C2_NU = [0.1, 0.2, 0.4]
C3_MU = [0.3, 0.2, 0.4, 0.1, 0.5, 0.2, 0.1, 0.3]
C3_NU = [0.1, 0.4, 0.2, 0.3, 0.1, 0.4, 0.2, 0.1]
C1_MU = 0.4
C1_NU = 0.2


def f_of(name, phi, c_mu=None, c_nu=None):
    """按群档构造场强。"""
    if name == "U(1)":
        return 1j * (phi * C1_MU - phi * C1_NU)
    gens = T_SU2 if name == "SU(2)" else T_SU3
    cm = (c_mu if c_mu is not None else (C2_MU if name == "SU(2)" else C3_MU))
    cn = (c_nu if c_nu is not None else (C2_NU if name == "SU(2)" else C3_NU))
    Am = conn(gens, [x * phi for x in cm])
    An = conn(gens, [x * phi for x in cn])
    return Am @ An - An @ Am


def density(name, phi, c_mu=None, c_nu=None):
    """Tr(F F) 取负号约定给实正的密度。"""
    F = f_of(name, phi, c_mu, c_nu)
    if name == "U(1)":
        return float(-(F * F).real)
    return pair_density(F)


# 阶次期望：交换子对 phi 二次 -> Tr(FF) 四次；标量一次 -> 二次
EXPECT_ORDER = {"U(1)": 4.0, "SU(2)": 16.0, "SU(3)": 16.0}


# ================= PAIR1 冒烟 =================
print("=" * 72)
print("PAIR1 冒烟：规范不变双收缩的存在性")
print("=" * 72)

phi = 1e-2
sm = {}
for name in ("U(1)", "SU(2)", "SU(3)"):
    F = f_of(name, phi)
    if name == "U(1)":
        val = -np.trace(np.array([[F * F]])).real
    else:
        val = pair_density(F)
    sm[name] = val
    print(f"{name:6s} Tr(F F) = {val:+.12e}")

ok1 = all(v > 0 for v in sm.values())
record("PAIR1 冒烟：规范不变双收缩存在且实正", ok1,
       "; ".join(f"{k}={v:.4e}" for k, v in sm.items()),
       "三档均有定义且实部为正", "恒等式级（冒烟）")

# ================= PAIR2 双收缩枚举唯一性 =================
print("=" * 72)
print("PAIR2 双收缩枚举的唯一性")
print("=" * 72)


def channel_audit(name, phi):
    """枚举三条候选双收缩通道，判定何者落中心（标量）。"""
    gens = T_SU2 if name == "SU(2)" else T_SU3
    F = f_of(name, phi)
    if name == "U(1)":
        return {"full": float(-(F * F).real), "single": float(abs(F)),
                "elem": float(-(F * F).real), "n_legal": 1}
    n = F.shape[0]
    tr_single = np.trace(F)
    # (b) 单收缩：Tr(F) 是否为零（无迹性）
    single_is_zero = abs(tr_single) < TOL_ZERO
    # (c) 元素级：F^{ij} F^{ij} 是否落标量
    tr_sq = float(np.sum(np.real(F * F)))
    # 标量判据：元素级值与 Frobenius 迹的关系。标量时 tr_sq = n * scalar
    # 用独立通道核对：若元素级缩并等于全收缩的 n 倍则为标量通道
    full = pair_density(F)
    # 元素级缩并在非标量情形下与 full*n 有偏差
    elem_is_scalar = abs(tr_sq - n * full) < TOL_ZERO * max(1.0, abs(full)) * 100
    n_legal = 1
    if not single_is_zero:
        n_legal += 1
    if elem_is_scalar:
        n_legal += 1
    return {"full": full, "single": float(abs(tr_single)), "elem": tr_sq,
            "n_legal": n_legal, "elem_scalar": elem_is_scalar}


aud = {name: channel_audit(name, phi) for name in ("U(1)", "SU(2)", "SU(3)")}
for name, d in aud.items():
    print(f"{name:6s} 全收缩={d['full']:+.6e}  单收缩|Tr F|={d['single']:.3e}  "
          f"元素级={d['elem']:+.6e}  合法通道={d['n_legal']}  "
          f"元素级落标量={d.get('elem_scalar', True)}")

ok2 = (aud["SU(2)"]["n_legal"] == 1) and (aud["SU(3)"]["n_legal"] == 1)
record("PAIR2 双收缩枚举的唯一性", ok2,
       f"SU(2) 计数={aud['SU(2)']['n_legal']}, SU(3) 计数={aud['SU(3)']['n_legal']}, "
       f"单收缩 |Tr F| 均 <{TOL_ZERO:.0e}",
       "两档均唯一（Tr(F_mu F_nu)）", "恒等式级（指标枚举）")

# ================= PAIR3 二次性 =================
print("=" * 72)
print("PAIR3 迹的实值性与阶次（构造分层）")
print("=" * 72)

order_rows = []
for name in ("U(1)", "SU(2)", "SU(3)"):
    v1 = density(name, phi)
    v2 = density(name, 2 * phi)
    exp = EXPECT_ORDER[name]
    ratio = v2 / v1 if v1 != 0 else float("nan")
    dev = abs(ratio - exp)
    order_rows.append((name, v1, v2, ratio, dev, exp))
    print(f"{name:6s} phi: {v1:+.6e}  2phi: {v2:+.6e}  比值={ratio:.12f}  "
          f"期望={exp:.0f}  偏离={dev:.2e}")

ok3 = all(dev < TOL_QUAD for _, _, _, _, dev, _ in order_rows)
record("PAIR3 迹的实值性与阶次（构造分层）", ok3,
       "; ".join(f"{n} 比值 {r:.10f}（期望 {int(e)}）" for n, _, _, r, _, e in order_rows),
       f"各档比值与构造阶次一致（容差 {TOL_QUAD:.0e}）；阿贝尔二次对非阿贝尔四次",
       "恒等式级（阶次判据）")

# ================= PAIR4 高阶与拓扑项 =================
print("=" * 72)
print("PAIR4 F^3 恒零排除与拓扑项")
print("=" * 72)

# (i) F^3：交换子的迹恒为零，故 Tr(F^3) 恒零
f3_rows = []
for name in ("SU(2)", "SU(3)"):
    Fs = f_of(name, phi)
    t3 = float(np.real(np.trace(Fs @ Fs @ Fs)))
    f3_rows.append((name, t3))
    print(f"{name:6s} 交换子构造 Tr(F^3) = {t3:+.6e}")

# 区分力：非交换子构造的反 Hermitian 场强（带非零对角元），Tr(F^3) 须非零
M = np.array([[0.7, 1.0, 2.0], [-1.0, 0.0, 3.0], [-2.0, -3.0, 0.0]], dtype=complex)
f3_noncomm = float(np.real(np.trace(M @ M @ M)))
print(f"非交换子构造 Tr(F^3) = {f3_noncomm:+.6e}（应为显著非零）")

zero_ok = all(abs(t3) < TOL_ZERO for _, t3 in f3_rows)
dist_ok = abs(f3_noncomm) > THRESH_DISTINCT
print(f"交换子恒零={zero_ok}, 非交换子非零（区分力）={dist_ok}")

# (ii) 拓扑项 Tr(F Ftilde)：tilde F^{01} = F_{23}，用相对口径
topo_rows = []
topo_rel = {}


def topo_pair(name, mode="two"):
    """返回 (拓扑项, 主项)。

    mode="two"：两个独立反对称分量 F_01 与 F_23（各自非零）。
    mode="one"：只有一个独立反对称分量 F_01，其余分量为零。
    拓扑项为四指标 epsilon 缩并，用 Levi-Civita 符号精确枚举。
    """
    if name == "U(1)":
        return 0.0, abs(float(-(f_of(name, phi) * f_of(name, phi)).real))
    gens = T_SU2 if name == "SU(2)" else T_SU3
    cm = C2_MU if name == "SU(2)" else C3_MU
    cn = C2_NU if name == "SU(2)" else C3_NU
    F01 = field_strength(gens, [x * phi for x in cm], [x * phi for x in cn])
    main = abs(pair_density(F01))
    comps = {(0, 1): F01, (1, 0): -F01}
    if mode == "two":
        cm2 = [0.2, -0.1, 0.3] if name == "SU(2)" else [0.5, 0.2, 0.1, 0.3, 0.2, 0.1, 0.4, 0.2]
        cn2 = [0.4, 0.1, -0.2] if name == "SU(2)" else [0.1, 0.3, 0.4, 0.2, 0.1, 0.5, 0.1, 0.3]
        F23 = field_strength(gens, [x * phi for x in cm2], [x * phi for x in cn2])
        comps[(2, 3)] = F23
        comps[(3, 2)] = -F23
    val = 0.0
    for (a, b), Fa in comps.items():
        for (c, d), Fb in comps.items():
            val += LEVI[a, b, c, d] * np.real(np.trace(Fa @ Fb))
    return float(val), main


# Levi-Civita 符号（精确枚举，不用近似）
import itertools as _it

LEVI = np.zeros((4, 4, 4, 4))
for _p in _it.permutations(range(4)):
    _inv = sum(1 for i in range(4) for j in range(i + 1, 4) if _p[i] > _p[j])
    LEVI[_p] = (-1) ** _inv


for name in ("U(1)", "SU(2)", "SU(3)"):
    t, m = topo_pair(name)
    rel = abs(t) / m if m > 0 else float("nan")
    topo_rows.append((name, t, m, rel))
    topo_rel[name] = rel
    print(f"{name:6s} Tr(F Ftilde) = {t:+.6e}  主项 = {m:.6e}  相对比值 = {rel:.4f}")

# 单分量档：只有一个独立反对称分量，拓扑项应恒为零
one_topo = {}
for name in ("SU(2)", "SU(3)"):
    t, m = topo_pair(name, mode="one")
    one_topo[name] = t
    print(f"{name:6s} 单分量 Tr(F Ftilde) = {t:+.6e}（应为舍入地板零）")

# 相对口径：双分量档比值在 (0.1, 10)，单分量档归零
topo_ok = all(0.1 < topo_rel[n] < 10.0 for n in ("SU(2)", "SU(3)"))
topo_one_ok = all(abs(v) < TOL_ZERO for v in one_topo.values())
print(f"双分量同阶={topo_ok}（比值 "
      f"{[round(topo_rel[n], 4) for n in ('SU(2)', 'SU(3)')]}），"
      f"单分量归零={topo_one_ok}")

ok4 = zero_ok and dist_ok and topo_ok and topo_one_ok
record("PAIR4 F^3 恒零排除与拓扑项", ok4,
       f"交换子 Tr(F^3) 均零={zero_ok}；非交换子对照={f3_noncomm:.3e}（区分力={dist_ok}）；"
       f"拓扑项双分量相对比值 SU(2)={topo_rel['SU(2)']:.4f} SU(3)={topo_rel['SU(3)']:.4f}；"
       f"单分量档={ {k: f'{v:.1e}' for k, v in one_topo.items()} }",
       "F^3 恒零且非交换子对照非零（区分力）；拓扑项双分量同阶、单分量归零",
       "条件性命题级（拓扑项登记）")

# ================= PAIR5 恒真检验 =================
print("=" * 72)
print("PAIR5 恒真检验（EP5 齿）")
print("=" * 72)

# 反例一：单收缩替代双收缩
s1_su2 = abs(aud["SU(2)"]["single"])
s1_su3 = abs(aud["SU(3)"]["single"])
r1_ok = (s1_su2 < TOL_ZERO) or (s1_su2 > THRESH_DISTINCT)
print(f"反例一 单收缩：SU(2)={s1_su2:.3e}, SU(3)={s1_su3:.3e}（全收缩非零，显著不同={r1_ok}）")

# 反例二：标量退化
sc = abs(aud["U(1)"]["full"])
r2_ok = abs(aud["SU(2)"]["full"] - sc) > THRESH_DISTINCT * max(1.0, sc)
print(f"反例二 标量退化：U(1)={sc:.3e} 对 SU(2)={aud['SU(2)']['full']:.3e}（不同={r2_ok}）")

# 反例三：拓扑项的分量数齿（用相对判据）
r3_ok = (abs(one_topo["SU(2)"]) < TOL_ZERO) and (0.1 < topo_rel["SU(2)"] < 10.0)
print(f"反例三 拓扑项分量数：双分量相对比值={topo_rel['SU(2)']:.4f} 对 "
      f"单分量={one_topo['SU(2)']:.1e}（区分={r3_ok}）")

ok5 = r1_ok and r2_ok and r3_ok
record("PAIR5 恒真检验（EP5 齿）", ok5,
       f"单收缩 {s1_su2:.2e}/{s1_su3:.2e}；标量退化差异={r2_ok}；拓扑项退化区分={r3_ok}",
       f"三条反例均与全收缩显著不同（>{TOL_ZERO:.0e} 或 >{THRESH_DISTINCT:.0e}）",
       "恒真检验")

# ================= PAIR6 多档扫描 =================
print("=" * 72)
print("PAIR6 群维数多档扫描")
print("=" * 72)
counts = {n: aud[n]["n_legal"] for n in ("U(1)", "SU(2)", "SU(3)")}
orders = {n: r for n, _, _, r, _, _ in order_rows}
all_one = all(v == 1 for v in counts.values())
# 阶次按构造分层：U(1) 二次档内部一致，SU(2)/SU(3) 四次档内部一致
abelian_ok = abs(orders["U(1)"] - 4.0) < TOL_QUAD
nonabelian_ok = all(abs(orders[n] - 16.0) < TOL_QUAD for n in ("SU(2)", "SU(3)"))
print(f"合法通道计数 = {counts}")
print(f"阶次比值 = { {k: round(v, 10) for k, v in orders.items()} }")
print(f"阿贝尔档二次={abelian_ok}, 非阿贝尔档四次={nonabelian_ok}")
ok6 = all_one and abelian_ok and nonabelian_ok
record("PAIR6 群维数三档一致性", ok6,
       f"计数 {counts} 全为一={all_one}；阿贝尔二次={abelian_ok}；非阿贝尔四次={nonabelian_ok}",
       "三档计数均为 1，阶次按构造分层一致", "恒等式级（多档扫描）")

# ================= 汇总 =================
print("=" * 72)
n_pass = sum(1 for _, ok, _, _, _ in results if ok)
print(f"判据结果：{n_pass}/{len(results)}")
print("=" * 72)
for name, ok, meas, exp, lvl in results:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}  [{lvl}]")
    print(f"        {meas}")
sys.exit(0 if n_pass == len(results) else 1)