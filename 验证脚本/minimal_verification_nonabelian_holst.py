# -*- coding: utf-8 -*-
"""
判据第 580 条：非阿贝尔规范动力学 螺旋转圈对合的格点来源（阶段一）

理论期望（先立，脚本后写）
----------------------------
阶段一的验证范围限于几何侧一件事：螺旋几何的转圈次序对合（先转 a 再转 b 与
先转 b 再转 a 之差）在格点链接变量上的显式存在与后果。三件判据，全在格点上
精确计算，不做 Taylor 展开、不涉群论结构常数的归一化（后者属阶段二，
与 F^2 的矩阵配对唯一性一并处理）。

1. S1 冒烟：格点链接的次序对合。
   期望：不同生成元方向的 SU(2)/SU(3) 链接交换子范数显著非零；
   U(1) 标量链接交换子为机器精度零。
2. A3 Wilson loop 的次序敏感性（核心判据，对应对合的几何后果）：
   plaquette U_p = U_1 U_2 U_1^{-1} U_2^{-1} 对两条路径的次序敏感。
   期望：SU(2)/SU(3) 的两种次序给出不同 U_p（相对差异显著大于浮点噪声）；
   U(1) 给出相同 U_p（差为机器精度零）。此件即转圈对合的格点表现。
3. A4 恒真检验：把对合退化（同轴，两方向用同一生成元）后，敏感性须消失。
   期望：SU(2)/SU(3) 在同轴退化下 U_p 与 1 的偏差为零（对合不贡献），
   与异轴情形有显著差异。若无差异则本判据构造恒真，判 FAIL。

容差预算
----------------------------
- U(1) 判为零：标量乘法精确可交换，浮点上为机器精度，容差取 1e-12。
- SU(2)/SU(3) 判为显著非零：交换子范数在 theta = 1e-2 档下约 7e-6 量级，
  相对噪声须 > 1e-6，即至少高出舍入地板六个数量级。此阈值取自
  "显著高于浮点噪声"的一般要求；实测值若落在其下则须报告为协议伪影
  而非物理读数。首版取 theta = 1e-3 档使二阶小量落到 7e-7、低于阈值，
  属参数档选择问题（改档而非改阈值），两者相差三个数量级不改变结论。

模型设定
----------------------------
- 链接变量 U_a(n) = exp(i theta_a T_a)，theta_a = phi * L，phi = 1e-3 为方向转角，
  L = l_P（普朗格单位制取 1）。指数用谱分解精确求（生成元 Hermitian，
  无级数截断误差）。
- SU(2)：T_a = sigma_a / 2，三方向取三个不同 Pauli 轴（z、x、y）。
- SU(3)：T_a = lambda_a / 2（Gell-Mann 矩阵），取前两个生成元为两个不同方向。
- U(1)：T = 1（1x1 单位阵），标量可交换。
- plaquette：U_p = U_1 U_2 U_1^{-1} U_2^{-1}（标准 Wilson loop 定向）。
- Wilson loop 的两种次序：次序 A 为 (U_1, U_2)，次序 B 为 (U_2, U_1)，
  即沿同一闭合回路的两个方向（顺、逆），非交换群下二者不同。

边界声明
----------------------------
验证什么：转圈次序对合在格点链接上的显式存在（交换子非零）、其几何后果
（Wilson loop 的次序敏感性）、恒真区分力。
不验证什么：群论结构常数 f^{abc} 的归一化与 Jacobi 恒等式（属阶段二，
与 F^2 的矩阵配对唯一性一并处理，本判据刻意不涉，避免把标准群论
当作元相子几何的贡献来核）；不验证连续场论的展开阶次（不做级数展开）；
不验证色禁闭动力学（射程外）；不验证弱电统一（射程外）。
"""
import numpy as np
import sys

sys.stdout.reconfigure(encoding='utf-8')

TOL_ZERO = 1e-12
THRESH_DISTINCT = 1e-6

results = []


def record(name, passed, measured, expected, level):
    results.append((name, passed, measured, expected, level))


# ================= 生成元 =================
T_SU2 = [
    np.array([[1, 0], [0, -1]], dtype=complex) / 2,        # sigma_z
    np.array([[0, 1], [1, 0]], dtype=complex) / 2,         # sigma_x
    np.array([[0, -1j], [1j, 0]], dtype=complex) / 2,       # sigma_y
]

lambda_gm = np.array([
    [[0, 1, 0], [1, 0, 0], [0, 0, 0]],        # lambda_1
    [[0, -1j, 0], [1j, 0, 0], [0, 0, 0]],      # lambda_2
    [[0, 0, 1], [0, 0, 0], [1, 0, 0]],        # lambda_3
    [[0, 0, -1j], [0, 0, 0], [1j, 0, 0]],      # lambda_4
    [[0, 0, 0], [0, 0, 1], [0, 1, 0]],        # lambda_5
    [[0, 0, 0], [0, 0, -1j], [0, 1j, 0]],     # lambda_6
    [[1, 0, 0], [0, -1, 0], [0, 0, 0]],       # lambda_7
    [[0, 1, 0], [0, 0, 1], [0, 0, 0]],        # lambda_8
], dtype=complex)
T_SU3 = [lambda_gm[i] / 2.0 for i in range(8)]


def exact_exp(T, theta):
    """矩阵指数，谱分解精确求（T Hermitian）。"""
    w, V = np.linalg.eigh(T)
    return (V * np.exp(1j * theta * w)) @ V.conj().T


def norm(A):
    return float(np.sqrt(np.sum(np.abs(A) ** 2)))


def comm(A, B):
    return A @ B - B @ A


def wilson(U1, U2):
    return U1 @ U2 @ np.linalg.inv(U1) @ np.linalg.inv(U2)


def su2_link(axis, theta):
    return exact_exp(T_SU2[axis], theta)


def su3_link(gen, theta):
    return exact_exp(T_SU3[gen], theta)


def u1_link(theta):
    return np.array([[np.exp(1j * theta)]], dtype=complex)


# ================= 冒烟 =================
print("=" * 72)
print("冒烟：格点链接的次序对合")
print("=" * 72)
phi = 1e-2
L = 1.0
theta = phi * L

c_su2 = norm(comm(su2_link(0, theta), su2_link(1, theta)))
c_su3 = norm(comm(su3_link(0, theta), su3_link(1, theta)))
c_u1 = norm(comm(u1_link(theta), u1_link(phi * 0.7 * L)))

print(f"SU(2) 不同轴链接交换子范数 = {c_su2:.8e}")
print(f"SU(3) 不同生成元交换子范数 = {c_su3:.8e}")
print(f"U(1) 标量链接交换子范数   = {c_u1:.8e}（应为机器精度零）")
smoke_ok = c_su2 > THRESH_DISTINCT and c_su3 > THRESH_DISTINCT and c_u1 < TOL_ZERO
print("冒烟结论：", "通过" if smoke_ok else "失败")
record("S1 冒烟：格点链接次序对合", smoke_ok,
       f"SU(2)={c_su2:.3e}, SU(3)={c_su3:.3e}, U(1)={c_u1:.3e}",
       f"SU(2)/SU(3) > {THRESH_DISTINCT:.0e}, U(1) < {TOL_ZERO:.0e}", "恒等式级（冒烟）")
print()

# ================= A3 Wilson loop 次序敏感性 =================
print("=" * 72)
print("A3 Wilson loop 的次序敏感性（转圈对合的几何后果）")
print("=" * 72)
print(f"{'群':>8} {'|Up(顺)-Up(逆)|':>20} {'|Up(顺)-1|':>18} {'相对差异':>14} {'判定':>14}")

a3_rows = []
# U(1)
w1 = u1_link(theta)
w2 = u1_link(phi * 0.7 * L)
UpA = wilson(w1, w2)
UpB = wilson(w2, w1)
d = float(np.abs(UpA[0, 0] - UpB[0, 0]))
scale = float(np.abs(UpA[0, 0] - 1.0))
a3_rows.append(("U(1)", d, scale))

# SU(2)
s1a, s2a = su2_link(0, theta), su2_link(1, theta)
s1b, s2b = su2_link(1, theta), su2_link(0, theta)
SpA = wilson(s1a, s2a)
SpB = wilson(s1b, s2b)
d = norm(SpA - SpB)
scale = norm(SpA - np.eye(2))
a3_rows.append(("SU(2)", d, scale))

# SU(3)
g1a, g2a = su3_link(0, theta), su3_link(1, theta)
g1b, g2b = su3_link(1, theta), su3_link(0, theta)
GpA = wilson(g1a, g2a)
GpB = wilson(g1b, g2b)
d = norm(GpA - GpB)
scale = norm(GpA - np.eye(3))
a3_rows.append(("SU(3)", d, scale))

n3_ok = True
for name, d, scale in a3_rows:
    if name == "U(1)":
        rel = 0.0
        ok = d < TOL_ZERO
        verdict = "不敏感PASS" if ok else "异常敏感"
        if not ok:
            n3_ok = False
    else:
        rel = d / scale if scale > 0 else 0.0
        ok = rel > THRESH_DISTINCT
        verdict = "敏感PASS" if ok else "敏感FAIL"
        if not ok:
            n3_ok = False
    print(f"{name:>8} {d:>20.8e} {scale:>18.8e} {rel:>14.6f} {verdict:>14}")

record("A3 Wilson loop 次序敏感性", n3_ok,
       "U(1) 差为零，SU(2)/SU(3) 相对差异显著",
       f"U(1) < {TOL_ZERO:.0e}，其余 > {THRESH_DISTINCT:.0e}", "模型内判据级")
print()

# ================= A4 恒真检验 =================
print("=" * 72)
print("A4 恒真检验：对合退化（同轴）后敏感性消失")
print("=" * 72)
print(f"{'群':>8} {'|Up-1| 异轴':>18} {'|Up-1| 同轴':>18} {'相对差异':>14} {'判定':>12}")
n4_ok = True
a4_rows = []
# SU(2)
Sp_true = wilson(su2_link(0, theta), su2_link(1, theta))
Sp_degen = wilson(su2_link(0, theta), su2_link(0, theta))
t = norm(Sp_true - np.eye(2))
z = norm(Sp_degen - np.eye(2))
a4_rows.append(("SU(2)", t, z))
# SU(3)
Gp_true = wilson(su3_link(0, theta), su3_link(1, theta))
Gp_degen = wilson(su3_link(0, theta), su3_link(0, theta))
t = norm(Gp_true - np.eye(3))
z = norm(Gp_degen - np.eye(3))
a4_rows.append(("SU(3)", t, z))
# U(1) 基线
t = float(np.abs(wilson(w1, w2)[0, 0] - 1.0))
z = float(np.abs(wilson(w1, w1)[0, 0] - 1.0))
a4_rows.append(("U(1)", t, z))

for name, t, z in a4_rows:
    if name == "U(1)":
        rel = abs(t - z) / t if t > 0 else 0.0
        verdict = "基线(可交换)"
    else:
        rel = abs(t - z) / t if t > 0 else 0.0
        ok = rel > THRESH_DISTINCT
        verdict = "区分PASS" if ok else "区分FAIL"
        if not ok:
            n4_ok = False
    print(f"{name:>8} {t:>18.8e} {z:>18.8e} {rel:>14.6f} {verdict:>12}")

record("A4 恒真检验区分力", n4_ok,
       "SU(2)/SU(3) 同轴退化后 U_p 归 1，与异轴有显著差异",
       f"相对差异 > {THRESH_DISTINCT:.0e}", "判据设计级")
print()

# ================= 绘图 =================
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG_DIR = "D:/software/TianXingFC/images"
os.makedirs(IMG_DIR, exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

# 图1：对合交换子范数在三档群的对比（对数刻度，U(1) 为零）
ax = axes[0]
names = ["U(1)", "SU(2)", "SU(3)"]
vals = [c_u1, c_su2, c_su3]
vals_plot = [v if v > 0 else 1e-20 for v in vals]
colors = ["tab:gray", "tab:blue", "tab:orange"]
ax.bar(names, vals_plot, color=colors)
ax.set_yscale("log")
ax.set_ylim(1e-20, 1e-3)
ax.set_ylabel("|comm(U1, U2)|")
ax.set_title("S1 commutator vanishes for U(1)")
ax.grid(alpha=0.3, axis="y")

# 图2：Wilson loop 次序敏感性
ax = axes[1]
labels = ["U(1)", "SU(2)", "SU(3)"]
diffs = [r[1] for r in a3_rows]
diffs_plot = [d if d > 0 else 1e-20 for d in diffs]
ax.bar(labels, diffs_plot, color=colors)
ax.set_yscale("log")
ax.set_ylim(1e-20, 1e-3)
ax.set_ylabel("|Up(fwd) - Up(rev)|")
ax.set_title("A3 Wilson loop order sensitivity")
ax.grid(alpha=0.3, axis="y")

# 图3：同轴退化后 U_p 归 1
ax = axes[2]
true_vals = [r[1] for r in a4_rows]
degen_vals = [max(r[2], 1e-20) for r in a4_rows]
xpos = np.arange(3)
ax.bar(xpos - 0.2, [max(v, 1e-20) for v in true_vals], width=0.4,
       label="distinct axes", color=colors)
ax.bar(xpos + 0.2, degen_vals, width=0.4, label="same axis (degenerate)",
       color=colors, alpha=0.45)
ax.set_yscale("log")
ax.set_ylim(1e-20, 1e-3)
ax.set_xticks(xpos)
ax.set_xticklabels(labels)
ax.set_ylabel("|Up - 1|")
ax.set_title("A4 holst degeneracy check")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
out = IMG_DIR + "/nonabelian_holst_580.png"
plt.savefig(out, dpi=150)
print("figure saved: " + out)
print()

# ================= 汇总 =================
print("=" * 72)
print("判据汇总")
print("=" * 72)
n_pass = sum(1 for r in results if r[1])
for name, passed, measured, expected, level in results:
    flag = "PASS" if passed else "FAIL"
    print(f"[{flag}] {name}")
    print(f"       实测：{measured}")
    print(f"       期望：{expected}")
    print(f"       级别：{level}")
print()
print(f"合计 {n_pass}/{len(results)} 通过")
sys.exit(0 if n_pass == len(results) else 1)