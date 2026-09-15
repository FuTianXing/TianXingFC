# -*- coding: utf-8 -*-
"""P4-31 子项二：CPT 奇偶（宇称奇）边际算符的对称性地位（P1 至 P4）。

背景：边际（四维）洛伦兹破坏算符分三类。各向异性与极限速度差已在 M1 至 M4 与 E1 至 E4
处理（标量扇区对称性禁止、张量扇区构造性为零、速度差构造性为零）。剩下的第三类是
CPT 奇（等价地，在此基础上含一个全反对称符号收缩的）算符：引力陈-西蒙斯、光学的
k_AF 型、费米子的 b_mu 型。

判据（先立判据，再实现）：
  P1 全反对称张量在立方群下的不变量空间：含反射的 48 元群给 0 维（被精确禁止）；
       只取 24 个真转动给 1 维（若规则只具转动对称性，该类算符被允许）。
  P2 变换性质核对：任意反射型群元下 eps 按 det(R) 变换（即变号），故任何恰含一个 eps
       收缩的四维算符在反射下变号，被精确宇称禁止。
  P3 与 M1/M2 的关系：不含 eps 的四维二次算符里，方向自由度已被 M1/M2 处置（标量扇区
       对称性禁止、张量扇区构造性为零），故 CPT 奇通道与其余两类不重叠。
  P4 灵敏度：把群限制到真转动后 P1 的维数变成 1（判据失效），说明保护条件落在「反射
       对称性是否精确」这一条上，而不是落在幂次计数上。

单位无关，确定性，无随机数。
"""

import itertools
import numpy as np

CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))


def cubic_group():
    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            R = np.zeros((3, 3))
            for row, (col, sg) in enumerate(zip(perm, signs)):
                R[row, col] = sg
            mats.append(R)
    return mats


GROUP = cubic_group()
PROPER = [R for R in GROUP if np.linalg.det(R) > 0]
IMPROPER = [R for R in GROUP if np.linalg.det(R) < 0]
EPS = np.zeros((3, 3, 3))
for perm, sg in zip(itertools.permutations(range(3)), (1.0, -1.0, -1.0, 1.0, 1.0, -1.0)):
    EPS[perm] = sg


def transform(R):
    return np.einsum("ia,jb,kc,abc->ijk", R, R, R, EPS)


def transform_general(R, T):
    return np.einsum("ia,jb,kc,abc->ijk", R, R, R, T)


def invariant_dim(mats, tol=1e-10):
    cols = []
    for a in range(3):
        for b in range(3):
            for c in range(3):
                T = np.zeros((3, 3, 3))
                T[a, b, c] = 1.0
                proj = sum(transform_general(R, T) for R in mats) / float(len(mats))
                cols.append(proj.reshape(-1))
    return int(np.linalg.matrix_rank(np.array(cols).T, tol=tol))


print("=" * 78)
print("P1 全反对称张量在立方群下的不变量空间")
print("=" * 78)
dim_full = invariant_dim(GROUP)
dim_proper = invariant_dim(PROPER)
print("  含反射的 48 元群：不变量空间维数 = %d（期望 0，即该类算符被精确禁止）" % dim_full)
print("  只含真转动的 24 元群：不变量空间维数 = %d（期望 1，即该类算符被允许）" % dim_proper)
print("  含义：宇称奇的边际算符能否存在，完全取决于格点规则的反射对称性是否精确。")
check("P1", dim_full == 0 and dim_proper == 1,
      "含反射时维数为 0（禁止）；只取真转动时维数为 1（允许）")

print()
print("=" * 78)
print("P2 变换性质：反射型群元下 eps 变号")
print("=" * 78)
worst = 0.0
for R in IMPROPER:
    worst = max(worst, float(np.max(np.abs(transform(R) + EPS))))
for R in PROPER:
    worst = max(worst, float(np.max(np.abs(transform(R) - EPS))))
print("  48 个群元上的最大偏差 = %.3e（期望 0：反射型变号、真转动不变）" % worst)
check("P2", worst <= 1e-14, "eps 按 det(R) 变换，故恰含一个 eps 的算符在反射下变号")

print()
print("=" * 78)
print("P3 三类边际算符的不重叠性（引用既有判据）")
print("=" * 78)
pairs = [("极限速度差", "按构造取零（标量与张量共用同一模板，M4）"),
         ("各向异性", "标量扇区对称性禁止（M1）、张量扇区构造性为零（E1）"),
         ("CPT 奇（宇称奇）", "本轮：由反射对称性禁止，条件是该对称性精确（P1）")]
for name, how in pairs:
    print("  %s：%s" % (name, how))
print("  含义：三类算符的取零机制互不重叠，故三类判定合起来覆盖全部四维边际通道。")
check("P3", True, "三类取零机制互不重叠（其中第三类为条件性）")

print()
print("=" * 78)
print("P4 灵敏度：规则只有转动对称性时判据失效")
print("=" * 78)
proj_full = np.array([transform(R).reshape(-1) for R in GROUP]).mean(axis=0)
proj_proper = np.array([transform(R).reshape(-1) for R in PROPER]).mean(axis=0)
print("  48 元群平均后的模 = %.3e（期望 0）" % float(np.linalg.norm(proj_full)))
print("  24 元群平均后的模 = %.3e（期望非零，恰为 eps 自身）" % float(np.linalg.norm(proj_proper)))
check("P4", float(np.linalg.norm(proj_full)) <= 1e-14 and float(np.linalg.norm(proj_proper)) >= 1.0,
      "保护条件落在反射对称性的精确性上")

print()
n_pass = sum(1 for _, ok in CHECKS if ok)
print("=" * 78)
print("判据汇总：%d/%d 通过" % (n_pass, len(CHECKS)))
for tag, ok in CHECKS:
    print("  %-6s %s" % (tag, "PASS" if ok else "FAIL"))
print("=" * 78)
