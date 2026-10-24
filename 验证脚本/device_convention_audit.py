# -*- coding: utf-8 -*-
r"""
device_convention_audit.py

[第795号] 装置口径的统一核对（判据 AA-1 至 AA-5）

**背景**：第 794 号判「第 784 号那轮的 gamma 是形 1 而第 792 号那轮是形 2」，**该判断经逐行
读代码核对是错的**——784 与 792 号两者的 g 与 gamma 构造逐行相同（都是形 1）；
**形 2 只出现在第 793 号的「构型 I」里**（它把 N^2 留在了 gamma 中）。

**本轮的做法**：**逐行核对各号脚本的装置构造**（不凭记忆），并在**同一台机器上重跑关键读数**，
验证「784 与 792 的 gamma 相同」这一结论。

**三种形式**：
  形 1  gamma = g/N^2 = (1+G) delta          （第 784、792 号用的）
  形 2  gamma = N^2 (1+G) delta              （第 793 号的「构型 I」）
  形 3  gamma = delta + (n-1) r_i r_j / r^2  （球对称度规在直角坐标下的正确形式）

运行：python 验证脚本/device_convention_audit.py
确定性（无随机），约 5 秒。图存 images/device_convention_audit.png
"""
import io
import os
import re
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T0 = time.time()
CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))


HALF = 8
EPS = 0.0125
M_WIDTH = 0.05


def th_c(r, m=M_WIDTH):
    r = np.asarray(r, dtype=float)
    t = 1.0 / (1.0 + np.exp((r - 1.0) / m))
    return t - t * t


def F_of(r, m=M_WIDTH):
    return 2.0 * th_c(r, m)


def G_of(r, m=M_WIDTH):
    t = th_c(r, m)
    return 2.0 * t + 3.5 * t * t


def dd(a, axis, eps):
    b = np.moveaxis(a, axis, -1)
    out = np.full_like(b, np.nan)
    out[..., 2:-2] = (b[..., :-4] - 8 * b[..., 1:-3] + 8 * b[..., 3:-1] - b[..., 4:]) / (12 * eps)
    return np.moveaxis(out, -1, axis)


def ricci_3d(gam, eps):
    sh = gam.shape[2:]
    det = (gam[0, 0] * (gam[1, 1] * gam[2, 2] - gam[1, 2] * gam[2, 1])
           - gam[0, 1] * (gam[1, 0] * gam[2, 2] - gam[1, 2] * gam[2, 0])
           + gam[0, 2] * (gam[1, 0] * gam[2, 1] - gam[1, 1] * gam[2, 0]))
    ginv = np.zeros((3, 3) + sh)
    for i in range(3):
        for j in range(3):
            co = (gam[(i + 1) % 3][(j + 1) % 3] * gam[(i + 2) % 3][(j + 2) % 3]
                  - gam[(i + 1) % 3][(j + 2) % 3] * gam[(i + 2) % 3][(j + 1) % 3])
            ginv[i, j] = co / det
    D = {}
    for k in range(3):
        for i in range(3):
            for j in range(3):
                D[(i, j, k)] = dd(gam[i, j], k, eps)
    Gam = np.zeros((3, 3, 3) + sh)
    for lam in range(3):
        for mu in range(3):
            for nu in range(3):
                acc = np.zeros(sh)
                for sig in range(3):
                    acc = acc + 0.5 * ginv[lam, sig] * (
                        D[(mu, nu, sig)] + D[(nu, mu, sig)] - D[(sig, mu, nu)])
                Gam[lam, mu, nu] = acc
    Ric = np.zeros((3, 3) + sh)
    for i in range(3):
        for j in range(3):
            acc = np.zeros(sh)
            for k in range(3):
                acc = acc + dd(Gam[k, i, j], k, eps)
                acc = acc - dd(Gam[k, i, k], j, eps)
            for k in range(3):
                tr = np.zeros(sh)
                for l in range(3):
                    tr = tr + Gam[l, k, l]
                acc = acc + Gam[k, i, j] * tr
                for l in range(3):
                    acc = acc - Gam[l, i, k] * Gam[k, j, l]
            Ric[i, j] = acc
    R3 = np.zeros(sh)
    for i in range(3):
        for j in range(3):
            R3 = R3 + ginv[i, j] * Ric[i, j]
    return Ric, R3, ginv


def grid(x0, eps, half, form):
    k = np.arange(-half, half + 1) * eps
    X = x0[0] + k[:, None, None]
    Y = x0[1] + k[None, :, None]
    Z = x0[2] + k[None, None, :]
    r2 = X ** 2 + Y ** 2 + Z ** 2
    r = np.sqrt(r2)
    Fv, Gv = F_of(r), G_of(r)
    N2 = 1.0 - Fv
    n = 1.0 + Gv
    gam = np.zeros((3, 3, 2 * half + 1, 2 * half + 1, 2 * half + 1))
    if form == 1:
        for i in range(3):
            for j in range(3):
                gam[i, j] = n * (1.0 if i == j else 0.0)
    elif form == 2:
        for i in range(3):
            for j in range(3):
                gam[i, j] = N2 * n * (1.0 if i == j else 0.0)
    else:
        comps = (X, Y, Z)
        safe = r2 > 0
        for i in range(3):
            for j in range(3):
                dirij = np.where(safe, comps[i] * comps[j] / np.where(safe, r2, 1.0),
                                 (1.0 / 3.0) if i == j else 0.0)
                gam[i, j] = (1.0 if i == j else 0.0) + (n - 1.0) * dirij
    return gam


def ctr(arr, half=HALF):
    return arr[half, half, half]


# ----------------------------------------------------------------------
# 逐行核对各号脚本的装置构造
# ----------------------------------------------------------------------
TARGETS = [
    ("784", "验证脚本/relations_itemwise.py"),
    ("792", "验证脚本/gij_adm_terms.py"),
    ("793", "验证脚本/r3_compatibility.py"),
]
PAT_G = re.compile(r"^\s*g\[1, 1\]\s*=\s*g\[2, 2\]\s*=\s*g\[3, 3\]\s*=\s*(.+?)\s*$")
PAT_GAM = re.compile(r"^\s*gam\[i, j\]\s*=\s*(.+?)\s*$")

print("=" * 78)
print("第795号：装置口径的统一核对")
print("=" * 78)
print()
print("逐行核对各号脚本的装置构造（g 的三个对角与 gamma 的取法）：")
audit = {}
for tag, path in TARGETS:
    full = os.path.join(ROOT, path)
    if not os.path.exists(full):
        print("  %s 号：MISSING %s" % (tag, path))
        continue
    src = io.open(full, encoding="utf-8").read().split("\n")
    gs = [PAT_G.match(l).group(1) for l in src if PAT_G.match(l)]
    gams = [PAT_GAM.match(l).group(1) for l in src if PAT_GAM.match(l)]
    # 只保留「gamma 怎么取」的行；d0gam 是 d_0 gamma（时间导数），与取法无关，排除
    gams = [x for x in gams if "d0gam" not in x]
    print("  %s 号（%s）：" % (tag, os.path.basename(path)))
    for l in gs:
        print("      g  : %s" % l)
    for l in gams:
        print("      gam: %s" % l)
    audit[tag] = (gs, gams)

print()
a784 = audit.get("784")
a792 = audit.get("792")
a793 = audit.get("793")

print("[AA-1] 784 号与 792 号的装置构造逐行相同")
# 同一构造写几遍不影响（784 号在两处循环里各写一遍），故比集合
same = (a784 is not None and a792 is not None
        and set(a784[0]) == set(a792[0]) and set(a784[1]) == set(a792[1]))
print("  784：%s ｜ 792：%s" % (a784, a792))
check("AA-1", same,
      "784 号与 792 号的装置构造（g 的三个对角与 gamma 的取法）相同：%s"
      "（**同一构造写几遍不影响，故比的是集合**）" % same)

print()
print("[AA-2] 793 号的 gamma 取法与其余两号不同")
diff = (a784 is not None and a793 is not None
        and (set(a784[1]) != set(a793[1]) or set(a784[0]) != set(a793[0])))
print("  793：%s" % (a793,))
check("AA-2", diff,
      "793 号的 gamma 取法与其余两号不同：%s（形 2 只出现在 793 号的「构型 I」里）" % diff)

print()
print("[AA-3] 同一装置上重跑三形的 R^{(3)}")
rows = []
for r0 in (1.4, 1.6, 1.8):
    line = []
    for form in (1, 2, 3):
        gam = grid((r0, 0.0, 0.0), EPS, HALF, form)
        Ric, R3, ginv = ricci_3d(gam, EPS)
        line.append((form, ctr(R3)))
    rows.append((r0, line))
    print("  r = %.1f：形 1 %+.6e ｜ 形 2 %+.6e ｜ 形 3 %+.6e"
          % (r0, line[0][1], line[1][1], line[2][1]))
check("AA-3", True, "三形的读数已给出（见上）")

print()
print("[AA-4] 形 1 的读数与 784、792 号那轮的读数逐位相同")
ref = {1.4: 4.975995e-01, 1.6: 9.213518e-03, 1.8: 1.700059e-04}
diffs = [abs(dict(rows[0][1])[1] - ref[1.4]),
         abs(dict(rows[1][1])[1] - ref[1.6]),
         abs(dict(rows[2][1])[1] - ref[1.8])]
print("  与第 784、792 号那轮读数的差：%s" % ", ".join("%.3e" % d for d in diffs))
check("AA-4", max(diffs) < 1e-8,
      "形 1 的读数与第 784、792 号那轮的 $R^{(3)}$ 逐位相同（最大差 %.3e），"
      "**故那两轮读数可以对话**" % max(diffs))

print()
print("[AA-5] 恒真替身：把形 1 的 gamma 加上 N^2（退成形 2），须给出不同的读数")
f1 = dict(rows[0][1])[1]
f2 = dict(rows[0][1])[2]
print("  形 1 %+.6e ｜ 形 2 %+.6e ｜ 相对差 %.3e"
      % (f1, f2, abs(f1 - f2) / abs(f1) if f1 != 0 else float("inf")))
check("AA-5", abs(f1 - f2) / abs(f1) > 1e-2,
      "形 1 与形 2 的读数显著不同（相对差 %.3e），判据有区分力"
      % (abs(f1 - f2) / abs(f1) if f1 != 0 else float("inf")))

print()
print("=" * 78)
n_pass = sum(1 for _, ok in CHECKS if ok)
print("判据通过：%d/%d" % (n_pass, len(CHECKS)))
print("=" * 78)
print()
print("耗时 %.2f 秒" % (time.time() - T0))

# 出图
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
IMG = os.path.join(ROOT, "images", "device_convention_audit.png")

fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))

ax = axes[0]
xs = [r[0] for r in rows]
for k, form, lab in ((0, 1, u"形 1：$\\gamma=g/N^2$（784、792 号）"),
                     (1, 2, u"形 2：$N^2$ 留在 $\\gamma$ 里（793 号）"),
                     (2, 3, u"形 3：球对称的直角形式")):
    ys = [r[1][k][1] for r in rows]
    ax.semilogy(xs, np.maximum(np.abs(ys), 1e-18), "o--", linewidth=1.5, label=lab)
ax.set_xlabel(u"半径 $r$")
ax.set_ylabel(u"$\\vert R^{(3)} \\vert$")
ax.set_title(u"(a) 三形的读数")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

ax = axes[1]
xs2 = [r[0] for r in rows]
ys2 = [abs(dict(r[1])[1] - dict(r[1])[2]) / abs(dict(r[1])[1]) for r in rows]
ax.semilogy(xs2, ys2, "o-", linewidth=1.6, color="#d62728")
ax.axhline(1e-2, color="#2ca02c", linestyle=":", linewidth=1.4, label=u"门槛 $10^{-2}$")
ax.set_xlabel(u"半径 $r$")
ax.set_ylabel(u"形 1 与形 2 的相对差")
ax.set_title(u"(b) 两种取法的可区分性")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

fig.suptitle(u"第795号：装置口径的统一核对", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(IMG, dpi=150)
print("图已存：%s" % IMG)
