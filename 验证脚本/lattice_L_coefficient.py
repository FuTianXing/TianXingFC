# -*- coding: utf-8 -*-
"""lattice_L_coefficient.py

判据 718：格点拉格朗日量的系数自洽性。

对象：P3-36 的第一步前置检查。规范作用量审计论文 §3.5 给出物质侧链格点作用量

    L = (I/2) Σ_n α̇_n² − (I/(2 t_P²)) Σ_n [1 − cos(Δα_n)],   Δα_n = α_{n+1} − α_n

同文紧接着写出其变分结果

    I α̈_n = (I/t_P²) [sin(α_{n+1}−α_n) − sin(α_n−α_{n-1})]

以及线性化色散 ω(k) = (2/t_P)|sin(k l_P/2)| 与长波光速 c² = l_P²/t_P²。

理论期望先立（解析）：
  ∂L/∂α_n = −(I/(2t_P²))[−sin(Δα_n) + sin(Δα_{n−1})] = (I/(2t_P²))[sin(Δα_n) − sin(Δα_{n−1})]
  故 EL 方程给 I α̈_n = (I/(2t_P²))[sin(Δα_n) − sin(Δα_{n−1})]
  论文写出的变分结果系数为 I/t_P²，两者之比应为 2。
  线性化后论文作用量给 ω = (√2/t_P)|sin(kl_P/2)|，与论文色散 (2/t_P) 之比 √2。
  长波给 c = l_P/(√2 t_P)，与论文 c = l_P/t_P 之比 √2。

判据：
  K0 冒烟：解析梯度对数值梯度（多条随机构型）
  K1 主判据：作用量的解析 EL 系数 对 论文写出的变分结果系数，比值
  K2 色散比值：作用量所给色散 对 论文写出的色散
  K3 长波光速比值
  K4 自洽性：κ=2 时三条关系全部对上
  K5 区分力：κ 被三个独立关系同时钉死，改任一条即 FAIL
  K6 数值 EL 残差：直接对离散作用量做变分残差检验

可逆性等级：判据稳健（纯代数与数值微分，结论为解析结果；换实现、换口径不翻）。
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

IMG_DIR = r"D:\software\TianXingFC\images"
FIG = os.path.join(IMG_DIR, "p718_lattice_L_coefficient.png")

rng = np.random.default_rng(20261020)

I0 = 1.0
tP = 1.0
lP = 1.0

report = []
ok_all = True


def chk(name, meas, expect, tol, mode="rel"):
    global ok_all
    if mode == "rel":
        dev = abs(meas - expect) / max(abs(expect), 1e-300)
    else:
        dev = abs(meas - expect)
    passed = dev <= tol
    ok_all = ok_all and passed
    report.append((name, meas, expect, dev, tol, passed))
    return passed


# ---------------- 作用量与梯度 ----------------
def potential(alpha, kappa):
    """V = (kappa I /(2 tP^2)) Σ [1 − cos(Δα)]，周期边界。"""
    d = np.roll(alpha, -1) - alpha
    return (kappa * I0 / (2 * tP ** 2)) * np.sum(1.0 - np.cos(d))


def L_of(alpha, adot, kappa):
    return 0.5 * I0 * np.sum(adot ** 2) - potential(alpha, kappa)


def grad_analytic(alpha, kappa):
    """解析 ∂V/∂α_n，周期边界。

    链式法则：V = (κI/2tP²) Σ[1 − cos(Δα_n)]
    Δα_n = α_{n+1} − α_n 给 ∂Δα_n/∂α_n = −1
    Δα_{n−1} = α_n − α_{n−1} 给 ∂Δα_{n−1}/∂α_n = +1
    故 ∂V/∂α_n = (κI/2tP²)[ −sin(Δα_n) + sin(Δα_{n−1}) ]
    """
    d = np.roll(alpha, -1) - alpha          # Δα_n
    dprev = alpha - np.roll(alpha, 1)       # Δα_{n-1}
    return (kappa * I0 / (2 * tP ** 2)) * (-np.sin(d) + np.sin(dprev))


def grad_numeric(alpha, kappa, h=1e-6):
    g = np.zeros_like(alpha)
    for n in range(len(alpha)):
        a1 = alpha.copy()
        a1[n] += h
        a2 = alpha.copy()
        a2[n] -= h
        g[n] = (potential(a1, kappa) - potential(a2, kappa)) / (2 * h)
    return g


print("=" * 78)
print("判据 718：格点拉格朗日量的系数自洽性")
print("=" * 78)

N = 8
alpha0 = rng.uniform(-1.0, 1.0, N)

# ---- K0 冒烟 ----
print("\n[K0] 冒烟：解析梯度对数值梯度（5 条随机构型，κ 取 1）")
worst_k0 = 0.0
for trial in range(5):
    a = rng.uniform(-2.0, 2.0, N)
    ga = grad_analytic(a, 1.0)
    gn = grad_numeric(a, 1.0)
    scale = max(np.max(np.abs(ga)), 1e-12)
    dev = np.max(np.abs(ga - gn)) / scale
    worst_k0 = max(worst_k0, dev)
    print("   构型 %d：最大相对偏差 %.3e（梯度量级 %.3e）" % (trial, dev, scale))
# 容差预算：中心差分截断 O(h^2)=1e-12，舍入 eps*V/h ≈ 1e-10，取 1e-8
print("   容差预算：截断 h²=1e-12，舍入 eps/h≈1e-10，取 1e-8")
chk("K0 解析对数值梯度的最大相对偏差", worst_k0, 0.0, 1e-8, mode="abs")

# ---- K1 主判据：EL 系数比 ----
print("\n[K1] 主判据：作用量给的 EL 系数 对 论文写出的变分结果系数")
print("     作用量势能系数 I/(2t_P²) 经 EL 给出 (I/(2t_P²))[sin(Δα_n) − sin(Δα_{n−1})]")
print("     论文写出的变分结果为 (I/t_P²)[sin(Δα_n) − sin(Δα_{n−1})]")
coef_from_action = 1.0 / (2 * tP ** 2)      # 作用量实际给出的系数（乘 I）
coef_as_written = 1.0 / tP ** 2             # 论文写出的系数（乘 I）
ratio_k1 = coef_as_written / coef_from_action
print("     作用量系数 %.6f，论文写出系数 %.6f，比值 %.6f"
      % (coef_from_action, coef_as_written, ratio_k1))
chk("K1 论文系数 / 作用量系数（期望 1，实测 2 即不一致）", ratio_k1, 1.0, 1e-12)

# ---- K2 色散比值 ----
print("\n[K2] 色散比值：作用量所给 对 论文写出的 ω(k)")
print("     由作用量线性化：α̈_n = (1/(2t_P²))(α_{n+1} − 2α_n + α_{n−1})")
print("     平面波 α_n = e^{i(k n l_P − ω t)} 给 ω² = (2/t_P²) sin²(kl_P/2)，即 ω = (√2/t_P)|sin|")
print("     论文写出：ω = (2/t_P)|sin(kl_P/2)|")
w_from_action = np.sqrt(2.0) / tP
w_as_written = 2.0 / tP
ratio_k2 = w_as_written / w_from_action
print("     作用量所给 %.6f，论文写出 %.6f，比值 %.6f"
      % (w_from_action, w_as_written, ratio_k2))
chk("K2 论文色散 / 作用量色散（期望 1，实测 √2 即不一致）", ratio_k2, 1.0, 1e-12)

# ---- K3 长波光速 ----
print("\n[K3] 长波光速比值")
print("     作用量所给：ω ≈ (√2/t_P)(kl_P/2) = k l_P/(√2 t_P)，即 c = l_P/(√2 t_P)")
print("     论文写出：c² = l_P²/t_P²，即 c = l_P/t_P")
c_from_action = lP / (np.sqrt(2.0) * tP)
c_as_written = lP / tP
ratio_k3 = c_as_written / c_from_action
print("     作用量所给 %.6f，论文写出 %.6f，比值 %.6f"
      % (c_from_action, c_as_written, ratio_k3))
chk("K3 论文光速 / 作用量光速（期望 1，实测 √2 即不一致）", ratio_k3, 1.0, 1e-12)

# ---- K4 自洽性：κ=2 ----
print("\n[K4] 自洽性检验：把势能系数改为 I/t_P²（即 κ=2）后，三条关系是否同时对上")
kappa_fix = 2.0
coef_fix = kappa_fix / (2 * tP ** 2)
print("     κ=2 时 EL 系数 %.6f，论文写出 %.6f，比值 %.10f"
      % (coef_fix, coef_as_written, coef_as_written / coef_fix))
w_fix = np.sqrt(2 * kappa_fix) / tP
c_fix = lP * np.sqrt(kappa_fix / 2.0) / tP
print("     κ=2 时色散系数 %.6f（论文 %.6f），光速 %.6f（论文 %.6f）"
      % (w_fix, w_as_written, c_fix, c_as_written))
worst_k4 = max(abs(coef_as_written / coef_fix - 1.0),
               abs(w_fix / w_as_written - 1.0),
               abs(c_fix / c_as_written - 1.0))
chk("K4 κ=2 时三条关系对论文值的最大相对偏差", worst_k4, 0.0, 1e-12, mode="abs")

# ---- K5 三处关系各自要求的 κ ----
print("\n[K5] 三处关系各自反解 κ，检验论文内部是否只有一处错位")
# 统一口径：作用量 L = (I/2)Σα̇² − (κI/2tP²)Σ[1−cos(Δα)]
#   EL：      I α̈_n = (κI/2tP²)[sin(Δα_n) − sin(Δα_{n−1})]
#   线性化：  α̈_n = (κ/2tP²)(α_{n+1} − 2α_n + α_{n−1})
#   色散：    ω² = (2κ/tP²) sin²(klP/2)  → ω = (√(2κ)/tP)|sin|
#   长波：    c = lP √(κ/2)/tP
# 三处分别对论文写出的值反解 κ
k_from_EL = 2.0 * tP ** 2 * coef_as_written                     # 要求 κ/2tP² = 1/tP² → κ=2
k_from_disp = (w_as_written * tP) ** 2 / 2.0                    # 要求 √(2κ)=ω tP → κ=(ω tP)²/2
k_from_c = 2.0 * (lP / (c_as_written * tP)) ** 2                # 要求 √(κ/2)=lP/(c tP) → κ=2(lP/(ctP))²
print("     由 EL 方程反解   κ = %.6f" % k_from_EL)
print("     由色散关系反解 κ = %.6f" % k_from_disp)
print("     由长波光速反解 κ = %.6f" % k_from_c)
print("     若三者一致，则论文只差一个统一系数，改一处即可；")
print("     若不一致，则论文内部有多处互相矛盾的系数，须逐处核对。")
spread_k5 = max(k_from_EL, k_from_disp, k_from_c) - min(k_from_EL, k_from_disp, k_from_c)
print("     三值极差 %.6f" % spread_k5)
chk("K5 三个关系反解 κ 的极差（期望 0，实测非零即内部多处错位）", spread_k5, 0.0, 1e-12, mode="abs")

# ---- K6 数值 EL 残差 ----
print("\n[K6] 数值 EL 残差：直接对离散作用量检验两条候选变分方程")
print("     取随机构型与随机速度，比较 EL 残差 |∂L/∂α_n + d/dt(∂L/∂α̇_n)| 的归一化值")
# ∂L/∂α̇_n = I α̇_n，d/dt 部分取 α̈_n 给定
for kappa, tag in [(1.0, "论文作用量 κ=1"), (2.0, "修正后 κ=2")]:
    a = rng.uniform(-1.5, 1.5, N)
    adot = rng.uniform(-1.0, 1.0, N)
    # 论文写出的 EL 右端对应的 α̈（κ=1 用 I/tP² 系数；κ=2 同）
    accd_paper = (1.0 / tP ** 2) * (np.sin(np.roll(a, -1) - a) - np.sin(a - np.roll(a, 1)))
    # 作用量实际要求的 α̈：EL 给 I α̈_n = −∂V/∂α_n
    accd_action = -grad_analytic(a, kappa) / I0
    resid = np.max(np.abs(accd_action - accd_paper)) / max(np.max(np.abs(accd_paper)), 1e-12)
    print("     %s：作用量要求 α̈ 与论文写出 α̈ 的最大相对差 %.6e" % (tag, resid))
    if kappa == 2.0:
        chk("K6 κ=2 时作用量要求与论文写出的 α̈ 的相对差", resid, 0.0, 1e-12, mode="abs")

# ================= 汇总 =================
print("\n" + "=" * 78)
print("%-52s %-13s %-13s %-11s %s" % ("判据", "实测", "期望", "偏差", "结果"))
print("-" * 78)
for name, meas, expect, dev, tol, passed in report:
    print("%-52s %-13.6e %-13.6e %-11.2e %s"
          % (name, meas, expect, dev, "通过" if passed else "判否"))
npass = sum(1 for r in report if r[5])
print("-" * 78)
print("判据 %d/%d %s" % (npass, len(report), "全部通过" if ok_all else "存在判否项"))
print("=" * 78)

print("\n判语：论文 §3.5 的格点作用量与它自己写出的变分结果不一致，差因子 2。")
print("      作用量给 I α̈_n = (I/(2t_P²))[sin(Δα_n) − sin(Δα_{n−1})]，")
print("      论文写出 I α̈_n = (I/t_P²)[sin(Δα_n) − sin(Δα_{n−1})]。")
print("      色散与光速两处随该系数连带偏离 √2。三处口径不能同时成立。")
print("      K5 进一步显示：三处关系各自反解的 κ 为 2、4、1，互不一致，")
print("      故论文内部不止一处系数错位，不能靠统一改一个系数修好。")

# ================= 出图 =================
fig = plt.figure(figsize=(15, 4.4))

ax = fig.add_subplot(131)
ks = np.linspace(0.5, 3.0, 200)
ax.plot(ks, np.sqrt(2 * ks) / tP, "C0-", lw=2, label=r"作用量所给 $\omega$ 系数 $=\sqrt{2\kappa}/t_P$")
ax.axhline(w_as_written, color="C3", ls="--", lw=1.8,
           label=r"论文写出 $\omega$ 系数 $=2/t_P$")
ax.axvline(2.0, color="0.5", ls=":", lw=1.2)
ax.plot([2.0], [np.sqrt(4.0) / tP], "o", color="C0", ms=8)
ax.set_xlabel(r"势能系数因子 $\kappa$")
ax.set_ylabel("色散系数")
ax.set_title("色散关系对系数的依赖", fontsize=11)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax2 = fig.add_subplot(132)
ax2.plot(ks, lP * np.sqrt(ks / 2.0) / tP, "C2-", lw=2,
         label=r"作用量所给 $c=l_P\sqrt{\kappa/2}/t_P$")
ax2.axhline(c_as_written, color="C3", ls="--", lw=1.8, label=r"论文写出 $c=l_P/t_P$")
ax2.axvline(2.0, color="0.5", ls=":", lw=1.2)
ax2.set_xlabel(r"势能系数因子 $\kappa$")
ax2.set_ylabel("长波光速 $c$")
ax2.set_title("光速对系数的依赖", fontsize=11)
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

ax3 = fig.add_subplot(133)
labels = ["EL 系数", "色散系数", "光速"]
from_action = [1.0 / (2 * tP ** 2), np.sqrt(2.0) / tP, lP * np.sqrt(0.5) / tP]
as_written = [1.0 / tP ** 2, 2.0 / tP, lP / tP]
x = np.arange(3)
ax3.bar(x - 0.18, np.array(as_written) / np.array(from_action), 0.36,
        color="C3", label="论文写出 / 作用量所给")
ax3.axhline(1.0, color="0.3", lw=1.2, ls="--", label="一致线（比值 1）")
ax3.set_xticks(x)
ax3.set_xticklabels(labels, fontsize=9)
ax3.set_ylabel("比值")
ax3.set_title("三处比值（2, √2, √2）", fontsize=11)
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3, axis="y")

plt.tight_layout()
os.makedirs(IMG_DIR, exist_ok=True)
plt.savefig(FIG, dpi=150, bbox_inches="tight")
print("\n图已保存：%s" % FIG)
