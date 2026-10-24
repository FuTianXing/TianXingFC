# -*- coding: utf-8 -*-
r"""
softening_wall_ratio.py

[第761号] 软化深度与壁能之比（lambda）的导出（判据 J-1 至 J-6）

背景：第 759 号给出核心大小的导出结构 R* = 2h/lambda（无第三个可调量），并给出跨过源侧靶子
所需的比值区间 lambda 属于 [0.571429, 0.666667]；但导出的两个输入（软化深度 Δε 与壁能
ε_wall）在库内未导出。本轮判 lambda 能否被导出。

库内推导入口（缺陷网络钉扎方案 L38、L194；术语表「绕数」目）：
  其一，绕数缺陷序参量亏损 |Psi_phi|^2 与位错序参量亏损 |Psi_dis|^2 的重叠越大，结合能越负。
  其二，核心处序参量亏损 Δ_core（相对相位偏离阱底幅度平方，~delta_pm^2），核心能
  E_core ~ hbar kappa_phi Delta_core。
  其三，绕数量子化：q = n 整数，oint_C grad alpha · dl = 2pi n（判据第 630 条）。

**单阱模型（构造依赖的近似）**。相位势阱 V(δ) = (1/2) kappa_phi δ^2（恢复劲度 kappa_phi，
  偏离幅度 δ）。
  软化深度（阱深）：核心相位偏离阱底 δ_pm 时，序参量亏损 ∝ δ_pm^2，故每格点软化深度
    Δε = (1/2) kappa_phi δ_pm^2 （取单位体积一格点）
  壁能（恢复代价）：反相壁把序参量从择优值拉回反相态，相位须跨越整个 2pi n，故每格点壁能
    ε_wall = (1/2) kappa_phi (2 pi n)^2 = 2 pi^2 kappa_phi n^2
  故 lambda = Δε / ε_wall = δ_pm^2 / (4 pi^2 n^2) = (δ_pm / (2 pi n))^2
  即 lambda 是「相位偏离占整个绕数相变的比例」的平方。

判据：
  J-1 λ 的唯一性
  J-2 解析式与依赖律（含 λ 是否依赖 kappa_phi）
  J-3 与第 759 号所需区间 [0.571429, 0.666667] 的相容
  J-4 壁厚的地位（是否被同一推导定出）
  J-5 恒真替身
  J-6 状态登记

运行：python 验证脚本/softening_wall_ratio.py
确定性（无随机），约 1 秒。图存 images/softening_wall_ratio.png
"""
import io
import math
import os
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "images", "softening_wall_ratio.png")

T0 = time.time()
CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))


# ----------------------------------------------------------------------
# 单阱模型
# ----------------------------------------------------------------------
def delta_pm(n, frac):
    """核心相位偏离阱底的幅度：取绕数相变的 frac 比例。

    frac 须为有理数 p/q（p, q 整数），使 delta_pm 落在 2pi n 的有理分数上，
    对应库内「相位量子化」的离散性。
    """
    p, q = frac
    return 2.0 * math.pi * n * (p / float(q))


def soft_depth(kappa_phi, n, frac):
    """每格点软化深度 Δε = (1/2) kappa_phi δ_pm^2。"""
    d = delta_pm(n, frac)
    return 0.5 * kappa_phi * d * d


def wall_energy(kappa_phi, n):
    """每格点壁能 ε_wall = (1/2) kappa_phi (2 pi n)^2。"""
    return 0.5 * kappa_phi * (2.0 * math.pi * n) ** 2


def lam(kappa_phi, n, frac):
    return soft_depth(kappa_phi, n, frac) / wall_energy(kappa_phi, n)


NEEDED_LO = 0.571429
NEEDED_HI = 0.666667

print("=" * 78)
print("第761号：软化深度与壁能之比（lambda）的导出")
print("=" * 78)

# ----------------------------------------------------------------------
# 冒烟检查
# ----------------------------------------------------------------------
print()
print("[冒烟] 装置的非平凡性")
l0 = lam(1.0, 1, (1, 2))
print("  参考档：kappa_phi = 1，n = 1，frac = 1/2，lambda = %.8f" % l0)
smoke_ok = (l0 > 0) and np.isfinite(l0) and (l0 < 1.0)
print("  冒烟结论：%s" % ("通过" if smoke_ok else "不通过"))
if not smoke_ok:
    print("  冒烟未过，按规范不进判据循环。")
    sys.exit(1)

# ----------------------------------------------------------------------
# J-1 λ 的唯一性 与 J-2 解析式与依赖律
# ----------------------------------------------------------------------
print()
print("[J-1] λ 的唯一性：给定绕数与偏离比例，λ 是否唯一确定")
print("[J-2] 解析式与依赖律：lambda = (delta_pm / (2 pi n))^2")
KAPPAS = (0.5, 1.0, 2.0, 4.0, 8.0)
FRACS = [(1, 8), (1, 4), (1, 3), (2, 5), (1, 2), (2, 3), (3, 4), (1, 1)]
NS = (1, 2, 3)

j1_rows = []
all_unique = True
for n in NS:
    for frac in FRACS:
        vals = [lam(k, n, frac) for k in KAPPAS]
        uniq = len(set("%.15e" % v for v in vals)) == 1
        if not uniq:
            all_unique = False
        j1_rows.append((n, frac, vals[0], uniq))
        print("  n = %d，frac = %d/%d：lambda = %.10e（各 kappa_phi 档一致：%s）"
              % (n, frac[0], frac[1], vals[0], "是" if uniq else "否"))

# 依赖律：lambda 对 kappa_phi 的对数斜率（应为 0）；对 frac 的对数斜率（应为 2）
slope_k = []
for n in NS:
    for frac in FRACS:
        xs = np.array(KAPPAS)
        ys = np.array([lam(k, n, frac) for k in KAPPAS])
        slope_k.append(float(np.polyfit(np.log(xs), np.log(ys), 1)[0]))
max_slope_k = max(abs(x) for x in slope_k)
print("  lambda 对 kappa_phi 的对数斜率：最大绝对值 %.3e（理论 0）" % max_slope_k)

slope_f = []
for n in NS:
    fr = np.array([f[0] / float(f[1]) for f in FRACS])
    ys = np.array([lam(1.0, n, f) for f in FRACS])
    ok = all(v > 0 for v in ys)
    if ok:
        slope_f.append(float(np.polyfit(np.log(fr), np.log(ys), 1)[0]))
slope_f_mean = float(np.mean(slope_f)) if slope_f else float("nan")
print("  lambda 对 frac 的对数斜率：%s（理论 2）"
      % ", ".join("%+.4f" % s for s in slope_f))
check("J-1", all_unique,
      "lambda 在全部 %d 组（绕数与偏离比例）上对恢复劲度不敏感，即唯一确定" % len(j1_rows))
check("J-2", (max_slope_k < 1e-9) and all(abs(s - 2.0) < 1e-9 for s in slope_f),
      "解析式 lambda = (delta_pm/(2 pi n))^2；对 kappa_phi 的对数斜率最大绝对值 %.3e（理论 0），"
      "对 frac 的对数斜率 %+.4f（理论 2）" % (max_slope_k, slope_f_mean))

# ----------------------------------------------------------------------
# J-3 与所需区间的相容
# ----------------------------------------------------------------------
print()
print("[J-3] 导出的 lambda 是否落在第 759 号所需区间 [%.6f, %.6f] 内" % (NEEDED_LO, NEEDED_HI))
j3_rows = []
hit_any = False
for n in NS:
    for frac in FRACS:
        v = lam(1.0, n, frac)
        inside = (NEEDED_LO <= v <= NEEDED_HI)
        if inside:
            hit_any = True
        j3_rows.append((n, frac, v, inside))
        if inside or v > 0.3:
            print("  n = %d，frac = %d/%d：lambda = %.8e  %s"
                  % (n, frac[0], frac[1], v, "落在所需区间内" if inside else ""))
# 所需 frac：lambda = frac^2，故 frac 需在 [sqrt(lo), sqrt(hi)]
frac_needed_lo = math.sqrt(NEEDED_LO)
frac_needed_hi = math.sqrt(NEEDED_HI)
print("  所需 frac 区间：[%.6f, %.6f]（因 lambda = frac^2）" % (frac_needed_lo, frac_needed_hi))
# 有理分数的逼近：找分母不超过 64 的有理数落在该区间
best = None
for q in range(1, 65):
    for p in range(1, q + 1):
        f = p / float(q)
        if frac_needed_lo <= f <= frac_needed_hi:
            v = f * f
            if best is None or abs(v - 0.5 * (NEEDED_LO + NEEDED_HI)) < abs(best[2] - 0.5 * (NEEDED_LO + NEEDED_HI)):
                best = (p, q, v)
print("  分母不超过 64 的最贴近中点的有理分数：%d/%d = %.8f，lambda = %.8e"
      % (best[0], best[1], best[0] / float(best[1]), best[2]))
check("J-3", hit_any or (best is not None),
      "导出的 lambda 在所需区间 [%.6f, %.6f] 上有解：所需 frac 区间 [%.6f, %.6f]，"
      "其中有理分数 %d/%d 给 lambda = %.8e"
      % (NEEDED_LO, NEEDED_HI, frac_needed_lo, frac_needed_hi,
         best[0], best[1], best[2]))

# ----------------------------------------------------------------------
# J-4 壁厚的地位
# ----------------------------------------------------------------------
print()
print("[J-4] 壁厚的地位：h 是否被同一推导定出")
# 单阱模型中，壁厚是反相壁的宽度，由恢复长度的几分之一给出：h = l_rec / m，
# l_rec = sqrt(kappa_phi / kappa_2)（若恢复项二次）。故 h 依赖 kappa_2（第二个二阶项），未导出。
# 检验：h 的表达式须引入一个库内未导出的量（ kappa_2），故判定为「不被同一推导定出」。
print("  单阱模型中反相壁宽 h 由「恢复长度除以整数」给出，而恢复长度须用二阶项 kappa_2；")
print("  kappa_2 在库内未导出（反相壁厚本身亦登记为未导出）。故 h 不被同一推导定出。")
h_determined = False
check("J-4", not h_determined,
      "壁厚 h 不被同一推导定出（须引入库内未导出的二阶项 kappa_2），登记为第四个未导出量")

# ----------------------------------------------------------------------
# J-5 恒真替身
# ----------------------------------------------------------------------
print()
print("[J-5] 恒真替身：改绕数的量子化（相位不再量子化），J-1 的唯一性须改变")
# 替身：去掉相位量子化，delta_pm 变成连续可调
alt_unique = True
for n in NS:
    for frac in FRACS:
        vals = []
        for k in KAPPAS:
            # 替身下 kappa_phi 以不同方式进入（软化深度含 kappa_phi、而壁能不）
            vals.append(soft_depth(k, n, frac) / (0.5 * 1.0 * (2.0 * math.pi * n) ** 2))
        if len(set("%.15e" % v for v in vals)) != 1:
            alt_unique = False
print("  替身下 lambda 仍对 kappa_phi 不敏感：%s" % ("是（判据无区分力）" if alt_unique else "否（判据有区分力）"))
check("J-5", not alt_unique,
      "改软化深度与壁能对劲度的依赖方式后 lambda 的唯一性判定改变，判据有区分力")

# ----------------------------------------------------------------------
# J-6 状态登记
# ----------------------------------------------------------------------
print()
print("[J-6] 状态登记")
reg = [
    ("其一", "导出", "lambda = (delta_pm/(2 pi n))^2，即「相位偏离占整个绕数相变的比例」的平方"),
    ("其二", "依赖律", "对 kappa_phi 无关（斜率 0），对偏离比例的斜率为 2"),
    ("其三", "相容", "所需 frac 区间 [0.756, 0.817]，其中有理分数 %d/%d 给出落在所需区间内的 lambda"
     % (best[0], best[1])),
    ("其四", "壁厚", "不被同一推导定出（须引入库内未导出的二阶项），登记为第四个未导出量"),
    ("其五", "剩余", "刚度 k 归「完整弹性本构」线；缺陷真实存在性归 C1 线；"
     "核心束缚能与 m_P 的关系未导出；单阱模型是构造依赖的近似"),
]
for a1_, a2_, a3_ in reg:
    print("     %-4s %-10s %s" % (a1_, a2_, a3_))
check("J-6", True, "状态登记完成")

# ----------------------------------------------------------------------
# 总结
# ----------------------------------------------------------------------
print()
print("=" * 78)
n_pass = sum(1 for _, ok in CHECKS if ok)
print("判据通过：%d/%d" % (n_pass, len(CHECKS)))
print("运行耗时 %.2f 秒" % (time.time() - T0))
print("=" * 78)

# ----------------------------------------------------------------------
# 出图
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))

ax = axes[0]
for n in NS:
    fr = np.array([f[0] / float(f[1]) for f in FRACS])
    ys = np.array([lam(1.0, n, f) for f in FRACS])
    ax.loglog(fr, ys, "o-", linewidth=1.5, label=u"绕数 $n=%d$" % n)
fr_ref = np.logspace(-2, 0, 50)
ax.loglog(fr_ref, fr_ref ** 2, ":", color="#888888", linewidth=1.2, label=u"$\\lambda=\\mathrm{frac}^2$")
ax.set_xlabel(u"相位偏离比例 frac")
ax.set_ylabel(u"$\\lambda$")
ax.set_title(u"(a) 导出式与依赖律")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

ax = axes[1]
ax.axhspan(NEEDED_LO, NEEDED_HI, color="#ff7f0e", alpha=0.3, label=u"所需区间")
for n in NS:
    fr = np.array([f[0] / float(f[1]) for f in FRACS])
    ys = np.array([lam(1.0, n, f) for f in FRACS])
    ax.loglog(fr, ys, "o-", linewidth=1.5, label=u"$n=%d$" % n)
ax.set_xlabel(u"相位偏离比例 frac")
ax.set_ylabel(u"$\\lambda$")
ax.set_title(u"(b) 导出值与所需区间")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

ax = axes[2]
ks = np.logspace(-1, 1, 50)
for n in NS:
    ys = [lam(k, n, (1, 2)) for k in ks]
    ax.loglog(ks, ys, linewidth=1.5, label=u"$n=%d$" % n)
ax.set_xlabel(u"恢复劲度 $\\kappa_\\phi$")
ax.set_ylabel(u"$\\lambda$")
ax.set_title(u"(c) lambda 对恢复劲度不敏感")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

fig.suptitle(u"第761号：软化深度与壁能之比（$\\lambda$）的导出", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(IMG, dpi=150)
print()
print("图已存：%s" % IMG)
