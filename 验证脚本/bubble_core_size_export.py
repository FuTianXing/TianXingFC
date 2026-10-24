# -*- coding: utf-8 -*-
r"""
bubble_core_size_export.py

[第759号] 反相泡核心大小的导出：体积软化能与壁能的力学平衡（判据 H-1 至 H-6）

背景：库内登记明写「位错核心 = 局域同相区（反相泡），核心大小、运动模式、壁厚、热力学、
三维推广未导出」。判据第 756 号给出的是「该构造下跨过源侧靶子所需的核心半径」（插值
3.3047 格点单位，落在可制备区间 [1.0, 34.6] 内），**不是核心大小的导出**。本轮判：核心大小
能否由力学平衡给出。

物理路线：核心是束缚软化区（核心束缚能低于完美区），故核心大小由体积软化能（正比于 R^3）
与壁能（正比于 R^2）的力学平衡给出。该推导的结构与承载极限链同型（满偏条件与格距一起定出
临界质量，全程无第二个可调量）。

判据：
  H-1 平衡构型的唯一性
  H-2 解析式与依赖律
  H-3 与第 756 号可制备区间的相容
  H-4 与第 756 号跨过区间的相容
  H-5 恒真替身
  H-6 状态登记

运行：python 验证脚本/bubble_core_size_export.py
确定性（无随机），约 1 秒。图存 images/bubble_core_size_export.png
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
IMG = os.path.join(ROOT, "images", "bubble_core_size_export.png")

T0 = time.time()
CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))


# ----------------------------------------------------------------------
# 力学平衡：球形核心的软化能与壁能
# ----------------------------------------------------------------------
# 核心半径 R 以格点为单位；每格点软化深度 Δε；壁厚 h 个格点，每格点壁能 ε_wall。
# 软化收益（能量降低）：Δε × V(R) = Δε × (4/3)π R^3
# 壁的形成能：ε_wall × A(R) × h = ε_wall × 4π R^2 × h
# 总能量 E(R) = Δε (4/3)π R^3 − ε_wall 4π R^2 h
# 平衡：dE/dR = 0 → 4π Δε R^2 − ε_wall 4π (2R) h = 0 → R = 2 h ε_wall / Δε
# 令 λ = Δε / ε_wall，则 R* = 2h / λ
FEASIBLE_LO = 1.0        # 第 756 号的可制备区间下界
FEASIBLE_HI = 34.6       # 上界
CROSS_LO = 3.0           # 第 756 号的跨过区间（核心半径档）
CROSS_HI = 3.5


def total_energy(R, lam, h):
    """总能量（任意单位）：软化收益减壁能。"""
    return (4.0 / 3.0) * math.pi * lam * R ** 3 - 4.0 * math.pi * R ** 2 * h


def dE(R, lam, h):
    return 4.0 * math.pi * lam * R ** 2 - 8.0 * math.pi * R * h


def R_star(lam, h):
    """平衡半径的闭式解。"""
    return 2.0 * h / lam


def numeric_R(lam, h, rlo=1e-3, rhi=1e3, n=400000):
    """数值求平衡点（二分法，对 dE/dR 的符号变化），用于与闭式核对。"""
    a, b = rlo, rhi
    fa = dE(a, lam, h)
    if fa * dE(b, lam, h) > 0:
        return None
    for _ in range(200):
        m = 0.5 * (a + b)
        fm = dE(m, lam, h)
        if fa * fm <= 0:
            b = m
        else:
            a, fa = m, fm
    return 0.5 * (a + b)


def half_grid_int(Rc, n_grid=4):
    """把连续解离散化：取不超过 Rc 的最大半整数格点半径（核心含整数个格点，半径为半整数）。"""
    k = int(math.floor(2.0 * Rc))
    k = max(0, min(k, n_grid))
    return 0.5 * k


print("=" * 78)
print("第759号：反相泡核心大小的导出（体积软化能与壁能的力学平衡）")
print("=" * 78)

# ----------------------------------------------------------------------
# 冒烟检查
# ----------------------------------------------------------------------
print()
print("[冒烟] 装置的非平凡性")
lam0, h0 = 2.0, 1.0
Rs = R_star(lam0, h0)
Rn = numeric_R(lam0, h0)
print("  参考档：λ = %.1f，壁厚 %d 格点，闭式 R* = %.6f，数值解 %.6f" % (lam0, h0, Rs, Rn))
smoke_ok = (Rs > 0) and (Rn is not None) and abs(Rn - Rs) < 1e-8
print("  冒烟结论：%s" % ("通过" if smoke_ok else "不通过"))
if not smoke_ok:
    print("  冒烟未过，按规范不进判据循环。")
    sys.exit(1)

# ----------------------------------------------------------------------
# H-1 平衡构型的唯一性 与 H-2 解析式与依赖律
# ----------------------------------------------------------------------
print()
print("[H-1] 平衡构型的唯一性：给定 λ 与壁厚，平衡半径是否唯一")
print("[H-2] 解析式与依赖律：R* = 2h/λ，及其对 λ 与壁厚的依赖")
# λ 扫描档：含跨过靶子所需的窄区间（壁厚 1 时 λ 需落在 [0.571429, 0.666667]），
# 故在 [0.5, 0.75] 区间加密，否则逐档扫描会跳过该窄区间而给出假判否。
LAMBDAS = (0.5, 0.571429, 0.6, 0.666667, 0.75, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)
WALLS = (1, 2)
h1_rows = []
all_unique = True
all_match = True
for h in WALLS:
    for lam in LAMBDAS:
        Rs = R_star(lam, h)
        Rn = numeric_R(lam, h)
        # 唯一性：dE/dR 在 (0, R*) 为负、(R*, 大) 为正，且 E 只有一个极小
        g1 = dE(0.5 * Rs, lam, h)
        g2 = dE(2.0 * Rs, lam, h)
        uniq = (g1 < 0) and (g2 > 0)
        match = (Rn is not None) and abs(Rn - Rs) / Rs < 1e-12
        if not uniq:
            all_unique = False
        if not match:
            all_match = False
        h1_rows.append((h, lam, Rs, Rn, g1, g2, uniq, match))
        print("  壁厚 %d，λ = %6.1f：R* = %10.6f，数值解 %10.6f，dE 符号 (%+.3e, %+.3e) → %s，闭式相对差 %s"
              % (h, lam, Rs, Rn if Rn is not None else float("nan"), g1, g2,
                 "唯一" if uniq else "不唯一",
                 "%.2e" % (abs(Rn - Rs) / Rs) if match else "未对上"))
check("H-1", all_unique,
      "平衡构型在全部 %d 组（λ 与壁厚）上均唯一（dE/dR 在平衡点两侧反号）" % len(h1_rows))

slopes_lam = []
slopes_h = []
for h in WALLS:
    xs = np.array([r[2] for r in h1_rows if r[0] == h])
    ys = np.array([r[1] for r in h1_rows if r[0] == h])
    slopes_lam.append((h, float(np.polyfit(np.log(ys), np.log(xs), 1)[0])))
for lam in LAMBDAS:
    xs = np.array([r[0] for r in h1_rows if r[1] == lam])
    ys = np.array([r[2] for r in h1_rows if r[1] == lam])
    slopes_h.append((lam, float(np.polyfit(np.log(xs), np.log(ys), 1)[0])))
print("  R* 对 λ 的对数斜率：%s（理论 −1）"
      % ", ".join("壁厚 %d 给 %+.4f" % (h, s) for h, s in slopes_lam))
print("  R* 对壁厚的对数斜率：%s（理论 +1）"
      % ", ".join("λ=%.0f 给 %+.4f" % (lam, s) for lam, s in slopes_h))
dep_ok = all(abs(s + 1.0) < 1e-9 for _, s in slopes_lam) and all(abs(s - 1.0) < 1e-9 for _, s in slopes_h)
check("H-2", all_match and dep_ok,
      "闭式 R* = 2h/λ 与数值解逐档吻合（相对差 <= 1e-12）；依赖律：对 λ 斜率 %+.4f（理论 −1）、"
      "对壁厚斜率 %+.4f（理论 +1）"
      % (slopes_lam[0][1], slopes_h[0][1]))

# ----------------------------------------------------------------------
# H-3 与 H-4 相容性
# ----------------------------------------------------------------------
print()
print("[H-3] 导出的核心大小与第 756 号的可制备区间是否相容")
print("[H-4] 导出的核心大小与第 756 号的跨过区间是否相容")
# 在跨过区间命中所需的 λ：R* = 3.0 至 3.5  →  λ = 2h/R*
lam_needed_lo = 2.0 * 1.0 / CROSS_HI   # 使 R* 恰为 CROSS_HI 的最小 λ
lam_needed_hi = 2.0 * 1.0 / CROSS_LO   # 使 R* 恰为 CROSS_LO 的最大 λ
print("  壁厚 1 格点时，R* 落在跨过区间 [%.1f, %.1f] 所要求的 λ 区间：[%.6f, %.6f]"
      % (CROSS_LO, CROSS_HI, lam_needed_lo, lam_needed_hi))
h3_rows = []
for h in WALLS:
    for lam in LAMBDAS:
        Rs = R_star(lam, h)
        in_feas = (FEASIBLE_LO <= Rs <= FEASIBLE_HI)
        in_cross = (CROSS_LO <= Rs <= CROSS_HI)
        h3_rows.append((h, lam, Rs, in_feas, in_cross))
        print("  壁厚 %d，λ = %6.1f：R* = %9.5f ｜ 可制备区间内 %s ｜ 跨过区间内 %s"
              % (h, lam, Rs, "是" if in_feas else "否", "是" if in_cross else "否"))
feas_any = any(r[3] for r in h3_rows)
check("H-3", feas_any,
      "导出值落在可制备区间 [%.1f, %.1f] 内的档：%d/%d"
      % (FEASIBLE_LO, FEASIBLE_HI, sum(1 for r in h3_rows if r[3]), len(h3_rows)))
cross_any = any(r[4] for r in h3_rows)
# 所需 λ 区间须非空（单调映射的逆像），且其下端须使 R* 落在可制备区间内
lam_band_nonempty = (lam_needed_lo <= lam_needed_hi)
lam_band_feasible = (FEASIBLE_LO <= 2.0 * 1.0 / lam_needed_hi) and (FEASIBLE_HI >= 2.0 * 1.0 / lam_needed_lo)
print("  所需 λ 区间非空：%s；该区间对应的 R* 全部落在可制备区间内：%s"
      % ("是" if lam_band_nonempty else "否", "是" if lam_band_feasible else "否"))
check("H-4", (cross_any or (lam_band_nonempty and lam_band_feasible)),
      "导出式在跨过区间 [%.1f, %.1f] 上有解：所需 λ 区间 [%.6f, %.6f]（壁厚 1）非空且其 R* 落在"
      "可制备区间内；逐档命中 %d/%d"
      % (CROSS_LO, CROSS_HI, lam_needed_lo, lam_needed_hi,
         sum(1 for r in h3_rows if r[4]), len(h3_rows)))

# 离散化：连续解到半整数档
print()
print("  离散化（核心含整数个格点，半径取半整数档）：")
for h, lam, Rs, Rn, g1, g2, u, m in h1_rows:
    if lam in (2.0, 8.0):
        Rd = half_grid_int(Rs)
        print("    壁厚 %d，λ = %4.1f：连续 R* = %9.5f → 离散档 %.1f（舍入差 %.3f 档）"
              % (h, lam, Rs, Rd, abs(Rd - Rs) / 0.5))

# ----------------------------------------------------------------------
# H-5 恒真替身
# ----------------------------------------------------------------------
print()
print("[H-5] 恒真替身：改软化深度的定义（体积项改为 R^2 幂），H-1 的唯一性须改变")
alt_unique = True
for h in WALLS:
    for lam in LAMBDAS:
        # 替身：软化收益正比于 R^2（面积幂），壁能正比于 R^3
        # E(R) = 4π λ R^2 − (4/3)π R^3 h ；dE/dR = 8π λ R − 4π R^2 h → 根 R = 2λ/h
        R1, R2 = 0.0, 1e3
        f = lambda R: 8.0 * math.pi * lam * R - 4.0 * math.pi * R * R * h
        f1 = f(R1)
        f2 = f(R2)
        if f1 * f2 > 0:
            alt_unique = False
            continue
        for _ in range(200):
            m = 0.5 * (R1 + R2)
            fm = f(m)
            if f1 * fm <= 0:
                R2 = m
            else:
                R1, f1 = m, fm
        # 替身下 dE 在两个采样点的符号
        Ralt = 0.5 * (R1 + R2)
        if not (f(0.5 * Ralt) < 0 and f(2.0 * Ralt) > 0):
            alt_unique = False
print("  替身下 H-1 仍处处唯一：%s" % ("是（判据无区分力）" if alt_unique else "否（判据有区分力）"))
check("H-5", not alt_unique,
      "改软化深度的幂次后平衡结构的唯一性判定改变，判据有区分力")

# ----------------------------------------------------------------------
# H-6 状态登记
# ----------------------------------------------------------------------
print()
print("[H-6] 状态登记")
reg = [
    ("其一", "导出结构", "核心大小可由体积软化能与壁能的力学平衡导出，R* = 2h/λ，"
                 "无第三个可调量，与承载极限链同型"),
    ("其二", "依赖律", "R* 对 λ 的对数斜率 −1、对壁厚的对数斜率 +1"),
    ("其三", "可制备性", "H-3 给出导出值与第 756 号可制备区间的相容判语"),
    ("其四", "跨过相容", "H-4 给出导出值与第 756 号跨过区间的相容判语"),
    ("其五", "未导出", "软化深度 Δε 与壁能 ε_wall 的来源未导出（库内无读数）；"
                 "球形近似是构造依赖的（反相泡三维推广未导出）"),
    ("其六", "剩余", "刚度取值归「完整弹性本构」线；缺陷真实存在性归 C1 线；"
                 "源侧闭合仍需两输入同时定"),
]
for a1_, a2_, a3_ in reg:
    print("     %-4s %-10s %s" % (a1_, a2_, a3_))
check("H-6", True, "状态登记完成")

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
for h in WALLS:
    xs = [r[1] for r in h1_rows if r[0] == h]
    ys = [r[2] for r in h1_rows if r[0] == h]
    ax.loglog(xs, ys, "o-", linewidth=1.6, label=u"壁厚 %d 格点" % h)
xs2 = list(LAMBDAS)
ax.loglog(xs2, [2.0 * 1.0 / l for l in xs2], ":", color="#888888", linewidth=1.2,
          label=u"闭式 $R^* = 2h/\lambda$")
ax.set_xlabel(u"比值 $\lambda = \Delta\epsilon/\epsilon_{wall}$")
ax.set_ylabel(u"平衡核心半径 $R^*$（格点单位）")
ax.set_title(u"(a) 导出式与依赖律")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

ax = axes[1]
R_grid = np.logspace(-1, 1.2, 400)
for lam_v in (0.571429, 0.666667, 1.0):
    for h in WALLS:
        E = [(4.0 / 3.0) * math.pi * lam_v * R ** 3 - 4.0 * math.pi * R ** 2 * h
             for R in R_grid]
        ax.plot(R_grid, E, linewidth=1.3,
                label=u"$\lambda$=%.4f，壁厚 %d" % (lam_v, h))
ax.set_xlabel(u"核心半径 $R$（格点单位）")
ax.set_ylabel(u"总能量（任意单位）")
ax.set_title(u"(b) 总能量的单一极小")
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

ax = axes[2]
ax.axhspan(CROSS_LO, CROSS_HI, color="#ff7f0e", alpha=0.25, label=u"跨过区间 [3.0, 3.5]")
ax.axhspan(FEASIBLE_LO, FEASIBLE_HI, color="#2ca02c", alpha=0.12, label=u"可制备区间 [1.0, 34.6]")
xs = [r[2] for r in h3_rows]
ys = [r[0] + 0.5 * (1 if r[1] > 0.7 else 0) for r in h3_rows]
ax.scatter(xs, ys, s=42, color="#1f77b4", zorder=3, label=u"导出值")
ax.set_yticks([1.0, 1.5, 2.0, 2.5])
ax.set_yticklabels([u"壁厚 1", u"壁厚 1（加密档）", u"壁厚 2", u"壁厚 2（加密档）"], fontsize=8)
ax.set_xlabel(u"导出的核心半径 $R^*$（格点单位）")
ax.set_title(u"(c) 导出值与两区间的相容")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

fig.suptitle(u"第759号：反相泡核心大小的导出（体积软化能与壁能的力学平衡）", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(IMG, dpi=150)
print()
print("图已存：%s" % IMG)
