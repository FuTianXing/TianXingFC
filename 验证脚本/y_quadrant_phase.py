# -*- coding: utf-8 -*-
"""P1-29 值侧开放问题三：Y 象限的相位语言操作合成（双原语链与最优角表）

定位：值侧论文《齐雷尔森界的体系内来源》开放问题三的执行轮。此前全空间口径（S = 2 sqrt2 恒饱和）
由相关矩阵结构给出（脚本 368 的三奇异值 (1, |z|, |z|)）；缺的是相位语言侧的显式操作协议与最优角表。
本轮把两条链的等效可观测量算成矩阵，给出角表，并在模型内核对上限。

模型（与值侧论文第 2 节同口径）：相关矩阵 T = diag(1) 直和 B，(Z, X, Y) 基，
  B = [[c, s], [s, -c]]，c = cos(gamma)、s = sin(gamma)（锐锁定，|z| = 1）；
  相关读数 E(a, b) = a^T T b；CHSH 量 S = E(a1,b1) + E(a1,b2) + E(a2,b1) - E(a2,b2)。
相位语言原语：层位占据 Z、表针奇偶 X、零模踢 U_z(theta) = diag(1, e^{i theta})、支重组合 H（50/50）。

判据先行（C1 至 C6）：
  C1 链甲（零模踢 theta -> 支重组合 -> 层位读出）的等效可观测量 = cos(theta) X - sin(theta) Y，
     即表针象限 (X, Y) 平面内方向角 -theta 的方向 (0, cos(theta), -sin(theta))；踢角与方向角差一负号。
  C2 子午面链（零模踢 -pi/2 -> 重组合 -> 踢 phi -> 重组合 -> 踢 pi/2 -> 读出，含第二次重组合）
     的等效可观测量落在 (Z, X) 平面，方向 (cos(phi), sin(phi), 0)；本链即开放问题所说的第二次重组合协议。
  C3 子午面限制上限：设置限制在 span(Z, X) 时 max S = 2 sqrt(1 + cos^2 gamma)（对值侧命题 3 的闭式）。
  C4 全空间上限：设置用链甲（(X, Y) 平面）时 max S = 2 sqrt2，对一切 gamma 恒饱和。
  C5 象限对齐：最优角表为 Alice {gamma/2, gamma/2 + pi/2}、Bob {gamma/2 - pi/4, gamma/2 + pi/4}
     （零模踢角），在该角表上 S = 2 sqrt2 精确；把角表固定成 gamma 无关（对齐偏移 delta = 0）时
     S 在 gamma != 0 处掉下来（gamma = pi/2 处为 0）。
  C6 判定与角表：输出可直接施工的协议表（每侧两条设置，各一条链甲）与结论登记。

边界：不重做 368 与其他既有读数；子午面链的物理实现细节（第二条重组合的场侧构造）不展开，只核可观测量；
不主张公理级结论；不触碰基准文档。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

FIG = r"D:\software\TianXingFC\images\y_quadrant_phase_protocol.png"
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = (X + Z) / np.sqrt(2.0)


def Uz(t):
    return np.diag([1.0 + 0j, np.exp(1j * t)])


def bloch(M):
    """展开到 (Z, X, Y) 分量。"""
    return np.array([np.real(np.trace(M @ Z) / 2), np.real(np.trace(M @ X) / 2), np.real(np.trace(M @ Y) / 2)])


def Tmat(g):
    c, s = np.cos(g), np.sin(g)
    T = np.zeros((3, 3))
    T[0, 0] = 1.0
    T[1, 1], T[1, 2], T[2, 1], T[2, 2] = c, s, s, -c
    return T


def chsh(T, a1, a2, b1, b2):
    return float(a1 @ T @ b1 + a1 @ T @ b2 + a2 @ T @ b1 - a2 @ T @ b2)


def sub_matrix(T, inplane):
    """取出对应平面的 2x2 子块。"""
    idx = [1, 2] if inplane else [0, 1]
    return T[np.ix_(idx, idx)]


def horodecki2(M):
    """2x2 相关块的 CHSH 上限闭式（Horodecki）：2 sqrt(s1^2 + s2^2)。"""
    sv = np.linalg.svd(M, compute_uv=False)
    return float(2.0 * np.sqrt((sv ** 2).sum()))


def opt_plane(T, inplane, ngrid=13, rounds=6):
    """平面族上的 CHSH 最大：粗网格 + 坐标上升细化（确定性）。"""
    def vec(t):
        return np.array([0.0, np.cos(t), np.sin(t)]) if inplane else np.array([np.cos(t), np.sin(t), 0.0])
    G = np.linspace(0, np.pi, ngrid)
    best, arg = -9e9, None
    for a1 in G:
        for a2 in G:
            for b1 in G:
                for b2 in G:
                    val = chsh(T, vec(a1), vec(a2), vec(b1), vec(b2))
                    if val > best:
                        best, arg = val, [a1, a2, b1, b2]
    for it in range(rounds):
        step = np.pi / (ngrid - 1) / (2.0 ** it)
        improved = True
        while improved:
            improved = False
            for k in range(4):
                for d in (-step, step):
                    cand = list(arg)
                    cand[k] += d
                    val = chsh(T, vec(cand[0]), vec(cand[1]), vec(cand[2]), vec(cand[3]))
                    if val > best:
                        best, arg, improved = val, cand, True
    return float(best), arg


def gate(name, ok, detail):
    print("[%s] %s：%s" % ("PASS" if ok else "FAIL", name, detail))
    assert ok, name


print("=" * 78)
print("P1-29 值侧开放问题三：Y 象限的相位语言操作合成（判据先行）")
print("=" * 78)

# ---------- C1 链甲的等效可观测量 ----------
th = 0.7
MA = Uz(th).conj().T @ X @ Uz(th)
target = np.cos(th) * X - np.sin(th) * Y
ok = np.abs(MA - target).max() < 1e-15
bA = bloch(MA)
gate("C1 链甲（踢-重组合-读出）", ok, "等效可观测量 = cos(theta) X - sin(theta) Y（矩阵差 %.1e）；即方向角为 -theta，踢角与方向角差一负号；方向 (Z,X,Y) = (%.1e, %.6f, %.6f)（theta = 0.7）" %
     (np.abs(MA - target).max(), bA[0], bA[1], bA[2]))

# ---------- C2 子午面链（含第二次重组合） ----------
phi = 0.9
V = Uz(-np.pi / 2) @ H @ Uz(phi) @ H @ Uz(np.pi / 2)
MM = V.conj().T @ Z @ V
cand = [("cos*Z+sin*X", np.cos(phi) * Z + np.sin(phi) * X), ("cos*Z-sin*X", np.cos(phi) * Z - np.sin(phi) * X)]
pick, diff = None, 9e9
for name, m in cand:
    d = np.abs(MM - m).max()
    if d < diff:
        pick, diff = name, d
bM = bloch(MM)
gate("C2 子午面链（含第二次重组合）", diff < 1e-15, "等效可观测量 = %s（矩阵差 %.1e）；方向 (Z,X,Y) = (%.6f, %.6f, %.1e)" %
     (pick, diff, bM[0], bM[1], bM[2]))

# ---------- C3 子午面限制上限 ----------
gam_list = np.linspace(0.0, np.pi, 9)
rows3 = []
ok3 = True
for g in gam_list:
    T = Tmat(g)
    m, _ = opt_plane(T, inplane=False)
    exp = 2.0 * np.sqrt(1.0 + np.cos(g) ** 2)
    ok3 = ok3 and abs(m - exp) < 5e-3
    rows3.append((g, m, exp))
ok3 = ok3 and all(abs(r[1] - horodecki2(sub_matrix(Tmat(r[0]), inplane=False))) < 5e-3 for r in rows3)
gate("C3 子午面限制上限", ok3, "九档 gamma：数值最大与 2 sqrt(1 + cos^2 gamma) 的最大偏差 %.1e；与 Horodecki 闭式一致（gamma = %.4f 处 %.6f 对 %.6f）" %
     (max(abs(r[1] - r[2]) for r in rows3), rows3[-1][0], rows3[-1][1], rows3[-1][2]))

# ---------- C4 全空间上限 ----------
rows4 = []
ok4 = True
for g in gam_list:
    T = Tmat(g)
    m, _ = opt_plane(T, inplane=True)
    ok4 = ok4 and abs(m - 2.0 * np.sqrt(2.0)) < 5e-3
    rows4.append((g, m))
ok4 = ok4 and all(abs(r[1] - horodecki2(sub_matrix(Tmat(r[0]), inplane=True))) < 5e-3 for r in rows4)
gate("C4 全空间上限（链甲平面）", ok4, "九档 gamma 的数值最大与 2 sqrt2 = %.9f 的最大偏差 %.1e；与 Horodecki 闭式一致" %
     (2.0 * np.sqrt(2.0), max(abs(r[1] - 2.0 * np.sqrt(2.0)) for r in rows4)))


def table_S(g, aligned=True):
    T = Tmat(g)
    base = g / 2.0 if aligned else 0.0
    def v(t):
        return np.array([0.0, np.cos(t), np.sin(t)])
    return chsh(T, v(base), v(base + np.pi / 2), v(base - np.pi / 4), v(base + np.pi / 4))


# ---------- C5 象限对齐 ----------
g0 = np.pi / 3
s_al = table_S(g0, aligned=True)
s_mis = table_S(g0, aligned=False)
s_mis_half = table_S(np.pi / 2, aligned=False)
ok5 = abs(s_al - 2.0 * np.sqrt(2.0)) < 1e-12 and s_mis < s_al - 1.0 and abs(s_mis_half) < 1e-12
gate("C5 象限对齐", ok5, "gamma = pi/3：对齐角表 S = %.12f（对 2 sqrt2 差 %.1e）；不对齐 S = %.6f；gamma = pi/2 且不对齐 S = %.1e" %
     (s_al, abs(s_al - 2.0 * np.sqrt(2.0)), s_mis, s_mis_half))

# ---------- C6 判定与角表 ----------
print("[INFO] C6 协议表（每侧两条设置，各一条链甲：零模踢 -> 支重组合 -> 层位读出）：")
print("[INFO]   A1：踢 -gamma/2；A2：踢 -gamma/2 - pi/2；B1：踢 -gamma/2 + pi/4；B2：踢 -gamma/2 - pi/4")
print("[INFO]   （表中为踢角；链甲把踢角 theta 实现为方向角 -theta，故方向角取负号）")
print("[INFO]   在该角表上 S = 2 sqrt2 精确（锐锁定，逐档核对）；子午面链（需第二次重组合）上限为 2 sqrt(1 + cos^2 gamma)。")

# ---------- 出图 ----------
fig, ax = plt.subplots(1, 2, figsize=(11.6, 4.4))
gg = np.linspace(-np.pi, np.pi, 241)
ax[0].plot(gg, 2.0 * np.sqrt(1.0 + np.cos(gg) ** 2), "-", lw=1.8, color="#b7791f", label="子午面链上限 2 sqrt(1 + cos^2 gamma)")
ax[0].plot([r[0] for r in rows3], [r[1] for r in rows3], "o", ms=4.5, color="#744210", label="数值最大（九档 gamma）")
ax[0].axhline(2 * np.sqrt(2), color="#2b6cb0", lw=1.6, ls="--", label="链甲平面（表针象限）恒饱和 2 sqrt2")
ax[0].axhline(2.0, color="#718096", lw=1.0, ls=":", label="经典界 2")
ax[0].set_xlabel("gamma（锁定相对表针）")
ax[0].set_ylabel("CHSH 量 S 的上限")
ax[0].set_title("两条链的上限：子午面随 gamma 起伏，链甲恒饱和")
ax[0].legend(fontsize=7.6)
ax[0].grid(alpha=0.25)

g1 = np.pi / 3
dd = np.linspace(-np.pi, np.pi, 241)
sv = [chsh(Tmat(g1), np.array([0, np.cos(d), np.sin(d)]), np.array([0, np.cos(d + np.pi / 2), np.sin(d + np.pi / 2)]),
           np.array([0, np.cos(d - np.pi / 4), np.sin(d - np.pi / 4)]), np.array([0, np.cos(d + np.pi / 4), np.sin(d + np.pi / 4)])) for d in dd]
ax[1].plot(dd, sv, "-", lw=1.8, color="#2f855a", label="角表整体对齐偏移 delta（gamma = pi/3）")
ax[1].axvline(g1 / 2, color="#c53030", lw=1.2, ls="--", label="象限对齐 delta = gamma/2 = %.4f" % (g1 / 2))
ax[1].axhline(2 * np.sqrt(2), color="#b7791f", lw=1.0, ls=":", label="2 sqrt2 = %.6f" % (2 * np.sqrt(2)))
ax[1].set_xlabel("对齐偏移 delta")
ax[1].set_ylabel("S")
ax[1].set_title("角表必须与象限轴对齐：偏移即掉下来")
ax[1].legend(fontsize=7.6)
ax[1].grid(alpha=0.25)
fig.tight_layout()
fig.savefig(FIG, dpi=150)
print("[INFO] 图已存：%s" % FIG)
print("=" * 78)
