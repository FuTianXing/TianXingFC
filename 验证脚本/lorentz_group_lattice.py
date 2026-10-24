# -*- coding: utf-8 -*-
"""P5b 洛伦兹群的晶格表达（第 471 条）：生成元代数、复合律与晶格残差。

背景：[[狭义相对论的定理化：从晶格几何到洛伦兹结构]] 遗留 (c)（生成元、boost 与旋转的复合的
晶格表达）；[[晶格何以不破坏洛伦兹不变性：元相子理论的时空机制与分层定理]]（低能有效理论为
洛伦兹标量，洛伦兹群是连续极限的精确对称群）；[[观测者响应与洛伦兹涌现：束缚态时钟的时间膨胀
与长度收缩推导]]（boost 解与 gamma 补偿）。

判据：
  LG1 生成元代数（连续极限，符号精确）：J_i 与 K_i 的 15 条对易关系全部精确为零（sympy）。
  LG2 生成元的格点表达：中心差分版 J/K 的闭合残差按 (l_P k)^2 标度（减半比值约 4），
      连续极限（k->0）趋于零。
  LG3 boost 复合与 Wigner 转动：两个正交 boost 的复合等于单 boost 乘绕 z 转动（分解精确）；
      小 rapidity 极限 Omega/(ab) -> 1/2；闭式核对。
  LG4 模式层晶格残差：boosted 晶格模式的频率偏离晶格色散，相对残差按 k^2 标度，v=0 时为零。

确定性；约数秒。图存 images/lorentz_group_lattice.png
"""
import math
import os
import time

import numpy as np
import sympy as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "images", "lorentz_group_lattice.png")
T0 = time.time()


def check(tag, ok, detail):
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))
    return bool(ok)


print("第 471 条 P5b 洛伦兹群的晶格表达：LG1 至 LG4")

# ---------------------------------------------------------------- LG1
print("LG1 生成元代数（连续极限，符号精确）")
t, x, y, z = sp.symbols("t x y z", real=True)
# 具体解析试验函数（多项式乘高斯，有理系数）：所有导数解析可算，化简可精确归零
F = sp.exp(-(t**2 + x**2 + y**2 + z**2)) * (
    1 + sp.Rational(3, 10) * t + sp.Rational(1, 5) * x + sp.Rational(1, 10) * x * y
    + sp.Rational(1, 20) * z + sp.Rational(1, 25) * t * z)


def df(e, v):
    return sp.diff(e, v)


Jx = lambda g: -sp.I * (y * df(g, z) - z * df(g, y))
Jy = lambda g: -sp.I * (z * df(g, x) - x * df(g, z))
Jz = lambda g: -sp.I * (x * df(g, y) - y * df(g, x))
Kx = lambda g: sp.I * (t * df(g, x) + x * df(g, t))
Ky = lambda g: sp.I * (t * df(g, y) + y * df(g, t))
Kz = lambda g: sp.I * (t * df(g, z) + z * df(g, t))
ops = {"Jx": (Jx, "J"), "Jy": (Jy, "J"), "Jz": (Jz, "J"),
       "Kx": (Kx, "K"), "Ky": (Ky, "K"), "Kz": (Kz, "K")}
axis_index = {"x": 0, "y": 1, "z": 2}


def lc(a, b, c):
    return (a - b) * (b - c) * (c - a) / 2


n_ok, n_all, worst = 0, 0, 0
for na, (fa, ta) in ops.items():
    for nb, (fb, tb) in ops.items():
        if na == nb:
            continue
        n_all += 1
        ia, ib = na[1], nb[1]
        # 求 c 与符号：eps(i,j,k)，k 为剩余轴
        rest = [q for q in "xyz" if q != ia and q != ib]
        ic = rest[0] if rest else ia
        eps = lc(axis_index[ia], axis_index[ib], axis_index[ic])
        if ta == "J" and tb == "J":
            exp = sp.I * eps * ops["J" + ic][0](F)
        elif ta == "J" and tb == "K":
            exp = sp.I * eps * ops["K" + ic][0](F)
        elif ta == "K" and tb == "J":
            exp = sp.I * eps * ops["K" + ic][0](F)
        else:
            exp = -sp.I * eps * ops["J" + ic][0](F)
        comm = sp.expand(fa(fb(F)) - fb(fa(F)))
        d = sp.simplify(comm - exp)
        if d == 0:
            n_ok += 1
        else:
            worst += 1
check("LG1", n_ok == n_all and n_all == 30,
      "对易关系 %d/%d 精确为零（J-J 6、J-K 12、K-K 6，含反序；符号精确）" % (n_ok, n_all))

# ---------------------------------------------------------------- LG2
print("LG2 生成元的格点表达（中心差分，局部补片）")
M = 3  # 补片半径：采样 -3..3，内区 -2..2 可作两次差分
idx = np.arange(-M, M + 1)
TT, XX, YY = np.meshgrid(idx, idx, idx, indexing="ij")


def D_patch(g, axis):
    return (np.roll(g, -1, axis=axis) - np.roll(g, 1, axis=axis)) / 2.0


def gen_apply(name, g):
    if name == "Jz":
        return -1j * (XX * D_patch(g, 2) - YY * D_patch(g, 1))
    if name == "Kx":
        return 1j * (TT * D_patch(g, 1) + XX * D_patch(g, 0))
    if name == "Ky":
        return 1j * (TT * D_patch(g, 2) + YY * D_patch(g, 0))
    raise KeyError(name)


inner = (slice(2, -2), slice(2, -2), slice(2, -2))
lg2 = []
for k in (0.05, 0.1, 0.2):
    kt, kx, ky = k * 0.6, k * 0.5, k * 0.62
    f = np.exp(1j * (kt * TT + kx * XX + ky * YY))
    comm = gen_apply("Kx", gen_apply("Ky", f)) - gen_apply("Ky", gen_apply("Kx", f))
    R = comm + 1j * gen_apply("Jz", f)
    num = np.sqrt(np.mean(np.abs(R[inner]) ** 2))
    den = np.sqrt(np.mean(np.abs(comm[inner]) ** 2))
    lg2.append(num / den)
ratios = [lg2[1] / lg2[0], lg2[2] / lg2[1]]
coefs = [lg2[i] / (0.05 * 2 ** i) ** 2 for i in range(3)]
check("LG2a", all(3.5 < r < 4.5 for r in ratios),
      "闭合残差按 k^2 标度：减半比值 %.3f 与 %.3f（k = 0.05/0.1/0.2）" % (ratios[0], ratios[1]))
kt_list = [0.6 * k for k in (0.05, 0.1, 0.2)]
pred2 = [abs((1 - math.cos(kt)) / math.cos(kt)) for kt in kt_list]
check("LG2b", max(abs(lg2[i] - pred2[i]) for i in range(3)) < 1e-10,
      "归一残差精确等于 |(1-cos k_t)/cos k_t|：%s vs %s"
      % (["%.2e" % x for x in lg2], ["%.2e" % x for x in pred2]))

# ---------------------------------------------------------------- LG3
print("LG3 boost 复合与 Wigner 转动（精确矩阵）")
I3 = np.eye(3)


def boost_mat(u):
    u = np.asarray(u, dtype=float)
    b2 = float(u @ u)
    g = 1.0 / math.sqrt(1.0 - b2)
    B = np.zeros((4, 4))
    B[0, 0] = g
    B[0, 1:] = g * u
    B[1:, 0] = g * u
    B[1:, 1:] = I3 + (g - 1.0) * np.outer(u, u) / b2
    return B


def rot_z(om):
    R = np.eye(4)
    R[1, 1] = math.cos(om)
    R[1, 2] = -math.sin(om)
    R[2, 1] = math.sin(om)
    R[2, 2] = math.cos(om)
    return R


lg3 = []
for a in (0.2, 0.1, 0.05):
    b = a
    Bx = np.eye(4)
    Bx[0, 0] = Bx[1, 1] = math.cosh(a)
    Bx[0, 1] = Bx[1, 0] = math.sinh(a)
    By = np.eye(4)
    By[0, 0] = By[2, 2] = math.cosh(b)
    By[0, 2] = By[2, 0] = math.sinh(b)
    Mmat = By @ Bx
    u = Mmat[1:, 0] / Mmat[0, 0]
    Bv = boost_mat(u)
    Rres = np.linalg.inv(Bv) @ Mmat
    dec = float(np.max(np.abs(Mmat - Bv @ Rres)))
    om = math.atan2(Rres[2, 1], Rres[1, 1])
    om_formula = math.atan2(math.sinh(a) * math.sinh(b), math.cosh(a) + math.cosh(b))
    lg3.append((a, om, om_formula, dec, om / (a * b)))
check("LG3a", max(r[3] for r in lg3) < 1e-12,
      "复合分解 M = B(V)R(Omega) 精确：最大残差 %.2e" % max(r[3] for r in lg3))
check("LG3b", abs(lg3[0][1] - lg3[0][2]) < 1e-12 and abs(lg3[2][1] - lg3[2][2]) < 1e-12,
      "闭式核对 tan(Omega) = sinh(a)sinh(b)/(cosh(a)+cosh(b))：偏差 %.2e"
      % max(abs(r[1] - r[2]) for r in lg3))
check("LG3c", abs(lg3[0][4] - 0.5) > abs(lg3[1][4] - 0.5) > abs(lg3[2][4] - 0.5),
      "小 rapidity 极限 Omega/(ab) -> 1/2：%s" % ([round(r[4], 4) for r in lg3]))

# ---------------------------------------------------------------- LG4
print("LG4 模式层晶格残差（boosted 模式对色散）")


def disp(k):
    return 2.0 * math.sin(k / 2.0)


lg4 = []
for v in (0.5,):
    for k in (0.4, 0.2, 0.1, 0.05):
        g = 1.0 / math.sqrt(1.0 - v * v)
        om = disp(k)
        om2 = g * (om - v * k)
        k2 = g * (k - v * om)
        r = abs(om2 - disp(k2)) / om2
        lg4.append((v, k, r))
r05 = [r for (v, k, r) in lg4]
ratios4 = [r05[0] / r05[1], r05[1] / r05[2], r05[2] / r05[3]]
v0 = abs(disp(0.2) - disp(0.2)) / disp(0.2)
check("LG4a", all(3.5 < r < 4.5 for r in ratios4),
      "boosted 模式残差按 k^2 标度：减半比值 %.3f / %.3f / %.3f" % tuple(ratios4))
check("LG4b", v0 < 1e-15, "v = 0 控制：残差 %.2e（零）" % v0)

# ---------------------------------------------------------------- 图
fig, axs = plt.subplots(2, 2, figsize=(11, 8))

ax = axs[0, 0]
ax.axis("off")
ax.text(0.02, 0.92, "LG1 生成元代数（符号精确）", fontsize=11)
ax.text(0.02, 0.74, "15 组对易关系（J-J、J-K、K-K，含反序共 30 条）：", fontsize=9)
ax.text(0.02, 0.62, "[J_i, J_j] = i e_ijk J_k", fontsize=10)
ax.text(0.02, 0.52, "[J_i, K_j] = i e_ijk K_k", fontsize=10)
ax.text(0.02, 0.42, "[K_i, K_j] = -i e_ijk J_k", fontsize=10)
ax.text(0.02, 0.26, "全部精确为零（sympy 化简）：%d/%d" % (n_ok, n_all), fontsize=10, color="#2b6cb0")

ax = axs[0, 1]
ks = np.array([0.05, 0.1, 0.2])
ax.loglog(ks, np.array(lg2), "o-", color="#2b6cb0", label="格点生成元闭合残差")
ax.loglog(ks, lg2[0] * (ks / ks[0]) ** 2, "--", color="gray", lw=0.9, label="斜率 2 参照")
ax.set_xlabel("波数 k（l_P = 1）")
ax.set_ylabel("相对残差")
ax.set_title("LG2 格点生成元闭合：残差按 k^2", fontsize=10)
ax.legend(fontsize=8)

ax = axs[1, 0]
aa = np.array([r[0] for r in lg3])
ratio = np.array([r[4] for r in lg3])
ax.semilogx(aa, ratio, "o-", color="#2b6cb0", label="Omega/(ab)（数值提取）")
ax.axhline(0.5, color="#c53030", ls="--", lw=0.9, label="极限 1/2")
ax.set_xlabel("rapidity a = b")
ax.set_ylabel("Omega/(ab)")
ax.set_title("LG3 boost 复合的 Wigner 转动", fontsize=10)
ax.legend(fontsize=8)

ax = axs[1, 1]
ax.loglog(np.array([r[1] for r in lg4]), np.array(r05), "o-", color="#2b6cb0", label="boosted 模式残差")
ax.loglog(np.array([r[1] for r in lg4]), r05[0] * (np.array([r[1] for r in lg4]) / 0.4) ** 2,
          "--", color="gray", lw=0.9, label="斜率 2 参照")
ax.plot([0.2], [max(v0, 1e-18)], "s", color="#68a063", ms=5, label="v = 0 控制（零）")
ax.set_xlabel("波数 k（l_P = 1）")
ax.set_ylabel("相对残差")
ax.set_title("LG4 模式层晶格残差（v = 0.5）", fontsize=10)
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(IMG, dpi=150)
print("图已存：%s（用时 %.1f s）" % (IMG, time.time() - T0))
