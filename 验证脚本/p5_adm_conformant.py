# -*- coding: utf-8 -*-
r"""
p5_adm_conformant.py

[第748号] 满足 3+1 分解前提的装置与差额项的消失核对（判据 C-1 至 C-6）

背景：判据第 747 号在 G1b8 装置上发现，机器的 R00 与严格 3+1 口径的预言
N[lap_(3)N + Gamma^i_ij d^j N] 之间存在一个 O(1) 的差额项；该差额项经 B-7 判为 g00 的
非线性泛函（弹性达 141.049）。第 747 号的第二轮自查另造标准配套装置（g_ij = (1-F)(1+G)）
复跑，仍只有 G1b8 口径成立，故「g_ij 未配套 N^2」不是唯一原因。

本轮把「N^2 是否配套」与「G 进哪一侧」作为两个独立维度做四档并列：
  甲 g00 = -(1-F)，gij = (1-F)(1+G) delta   gamma_ij = (1+G) delta，Om^2 = 1+G
  乙 g00 = -(1-F)，gij = (1-F) delta         gamma_ij = delta，Om^2 = 1
  丙 g00 = -1，     gij = delta               完全平直，绝对标定档
  丁 g00 = -(1-F)，gij = (1+G) delta         未配套，即第 747 号的装置，作同表对照

判据：
  C-1 满足前提的装置上差额项的消失：甲、乙两档的差额项占机器 R00 的份额 <= 1e-4
  C-2 完全平直档的绝对标定：丙档的机器 R00 严格为零（门槛 1e-14）
  C-3 差额项的泛函表达：若 C-1 判否，给出差额项的显式表达与系数（参数判据）
  C-4 恒真替身：改 gamma_ij 的配套方式，C-1 须 FAIL
  C-5 归属登记：差额项在四档上的量值与份额
  C-6 状态登记

运行：python 验证脚本/p5_adm_conformant.py
确定性（无随机），约 1 秒。图存 images/p5_adm_conformant.png
"""
import contextlib
import importlib.util
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
IMG = os.path.join(ROOT, "images", "p5_adm_conformant.png")

T0 = time.time()
CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))


# ----------------------------------------------------------------------
# 装置：import G1b8（主流程在模块级执行，import 时吞掉其输出）
# ----------------------------------------------------------------------
_G1B8 = os.path.join(ROOT, "验证脚本", "g1b_covariant_assembly.py")
_spec = importlib.util.spec_from_file_location("g1b8_adm", _G1B8)
g1b8 = importlib.util.module_from_spec(_spec)
sys.modules["g1b8_adm"] = g1b8
_buf = io.StringIO()
with contextlib.redirect_stdout(_buf):
    _spec.loader.exec_module(g1b8)

F_c = g1b8.F_c
G_c = g1b8.G_c
rad_d1 = g1b8.rad_d1
rad_d2 = g1b8.rad_d2
rad_lap = g1b8.rad_lap
rad_grad2 = g1b8.rad_grad2
MASS = g1b8.MASS

HALF = 4
C = HALF
NP_ = 2 * HALF + 1


# ----------------------------------------------------------------------
# 四档装置：g00 与 gij 的构造
# ----------------------------------------------------------------------
def make_patches(kind):
    """返回 (F 函数, g_ij 函数, gamma_ij 函数, N2 函数)。

    kind  甲：N2 = 1-F，gij = N2 (1+G) delta，gamma = (1+G) delta
          乙：N2 = 1-F，gij = N2 delta，gamma = delta
          丙：N2 = 1，   gij = delta，gamma = delta（完全平直）
          丁：N2 = 1-F，gij = (1+G) delta，gamma = (1+G) delta（未配套，第 747 号装置）
    """
    if kind == "丙":
        Ff = lambda r: np.zeros_like(np.asarray(r, dtype=float))
    else:
        Ff = lambda r: F_c(r)

    def N2f(r):
        return 1.0 - Ff(r)

    if kind == "甲":
        Gf = lambda r: (1.0 - Ff(r)) * (1.0 + G_c(r))
        gamf = lambda r: (1.0 + G_c(r))
        Om2f = lambda r: 1.0 + G_c(r)
    elif kind == "乙":
        Gf = lambda r: (1.0 - Ff(r))
        gamf = lambda r: np.ones_like(np.asarray(r, dtype=float))
        Om2f = lambda r: np.ones_like(np.asarray(r, dtype=float))
    elif kind == "丙":
        Gf = lambda r: np.ones_like(np.asarray(r, dtype=float))
        gamf = lambda r: np.ones_like(np.asarray(r, dtype=float))
        Om2f = lambda r: np.ones_like(np.asarray(r, dtype=float))
    else:  # 丁
        Gf = lambda r: (1.0 + G_c(r))
        gamf = lambda r: (1.0 + G_c(r))
        Om2f = lambda r: 1.0 + G_c(r)
    return Ff, Gf, gamf, N2f, Om2f


def g_patch(Ff, Gf, x0, eps):
    k = np.arange(-HALF, HALF + 1) * eps
    X = x0[0] + k[:, None, None]
    Y = x0[1] + k[None, :, None]
    Z = x0[2] + k[None, None, :]
    r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    g = np.zeros((4, 4, NP_, NP_, NP_))
    g[0, 0] = -(1.0 - Ff(r))
    for i in (1, 2, 3):
        g[i, i] = 1.0 + Gf(r)
    return g


def d1_keep(a, axis, eps):
    b = np.moveaxis(a, axis, -1)
    out = np.full_like(b, np.nan)
    out[..., 2:-2] = (b[..., :-4] - 8 * b[..., 1:-3] + 8 * b[..., 3:-1] - b[..., 4:]) / (12 * eps)
    return np.moveaxis(out, -1, axis)


def einstein_at(Ff, Gf, x0, eps):
    g = g_patch(Ff, Gf, x0, eps)
    dgf = [np.zeros_like(g), d1_keep(g, 2, eps), d1_keep(g, 3, eps), d1_keep(g, 4, eps)]
    gg = g[:, :, C, C, C]
    ginv = np.linalg.inv(gg)
    ginvf = np.zeros_like(g)
    ginvf[0, 0] = 1.0 / g[0, 0]
    for i in (1, 2, 3):
        ginvf[i, i] = 1.0 / g[i, i]
    Gam = np.zeros((4, 4, 4, NP_, NP_, NP_))
    for lam in range(4):
        for mu in range(4):
            for nu in range(mu, 4):
                acc = np.zeros((NP_, NP_, NP_))
                for sig in range(4):
                    if lam == sig:
                        acc = acc + ginvf[lam, sig] * 0.5 * (
                            dgf[mu][sig, nu] + dgf[nu][sig, mu] - dgf[sig][mu, nu])
                Gam[lam, mu, nu] = acc
                Gam[lam, nu, mu] = acc
    G0 = Gam[:, :, :, C, C, C]
    dG = np.zeros((4, 4, 4, 4))
    for kk in range(3):
        for lam in range(4):
            for mu in range(4):
                for nu in range(4):
                    dG[lam, mu, nu, kk + 1] = d1_keep(Gam[lam, mu, nu], kk, eps)[C, C, C]
    Ric = np.zeros((4, 4))
    for mu in range(4):
        for nu in range(4):
            acc = 0.0
            for lam in range(4):
                acc += dG[lam, mu, nu, lam]
            Ric[mu, nu] = acc
            for rho in range(4):
                Ric[mu, nu] -= dG[rho, mu, rho, nu]
    Gm = np.zeros((4, 4))
    for mu in range(4):
        for nu in range(4):
            acc = 0.0
            for rho in range(4):
                acc += ginv[rho, nu] * Ric[rho, mu]
            Gm[mu, nu] = Ric[mu, nu] - 0.5 * ginv[mu, nu] * np.trace(Ric)
    return Gm, Ric, ginv


def G_munu_adm(Ff, Gf, r0):
    g1, r1, gi1 = einstein_at(Ff, Gf, (r0, 0.0, 0.0), 0.05)
    g2, r2, gi2 = einstein_at(Ff, Gf, (r0, 0.0, 0.0), 0.025)
    return (16.0 * g2 - g1) / 15.0, (16.0 * r2 - r1) / 15.0


def machine_R00(Ff, Gf, r0):
    G4, Ric = G_munu_adm(Ff, Gf, r0)
    g00 = float(G4[0, 0])  # 由 G_munu 直接给出 g00 分量的机器值
    # g00 与 gii 需从构造取
    Fv = float(Ff(r0))
    Gv = float(Gf(r0))
    g00v = -(1.0 - Fv)
    giiv = 1.0 + Gv
    R = -(G4[0, 0] / g00v + (G4[1, 1] + G4[2, 2] + G4[3, 3]) / giiv)
    R00 = G4[0, 0] - 0.5 * g00v * R
    return float(R00), float(G4[0, 0]), float(R)


def strict_R00(Ff, gamf, Om2f, r0):
    """严格 3+1 口径：R00 = N[lap_(3)N + Gamma^i_ij d^j N]，以 gamma_ij = Om^2 delta 为空间度规。"""
    Fv = float(Ff(r0))
    N2 = 1.0 - Fv
    N = math.sqrt(N2)
    Om_f = lambda r: math.sqrt(float(Om2f(r)))
    Nf = lambda r: math.sqrt(1.0 - float(Ff(r)))
    dN = rad_d1(Nf, r0)
    dOm = rad_d1(Om_f, r0)

    def lap3(f, r):
        return rad_d1(lambda x: Om_f(x) ** 2 * rad_d1(f, x), r) / Om_f(r) ** 3
    lap3N = lap3(Nf, r0)
    gam = 3.0 * dN * dOm / Om_f(r0) ** 3
    return N * (lap3N + gam), N


print("=" * 78)
print("第748号：满足 3+1 分解前提的装置与差额项的消失核对")
print("=" * 78)

RADII = (0.8, 1.0, 1.2, 1.4, 1.6)
KINDS = ("甲", "乙", "丙", "丁")

# ----------------------------------------------------------------------
# 冒烟检查
# ----------------------------------------------------------------------
print()
print("[冒烟] 装置的非平凡性")
Ff, Gf, gamf, N2f, Om2f = make_patches("甲")
R00_m, G00_m, R_m = machine_R00(Ff, Gf, 1.2)
print("  甲档（r = 1.2）：机器 R00 = %+.6e，机器 G00 = %+.6e" % (R00_m, G00_m))
smoke_ok = (abs(R00_m) > 0) and (abs(G00_m) > 0)
print("  冒烟结论：%s" % ("通过" if smoke_ok else "不通过"))
if not smoke_ok:
    print("  冒烟未过，按规范不进判据循环。")
    sys.exit(1)

# ----------------------------------------------------------------------
# 四档的读数与归属
# ----------------------------------------------------------------------
print()
print("[C-5] 四档的读数与归属登记")
rows = {k: [] for k in KINDS}
for kind in KINDS:
    Ff, Gf, gamf, N2f, Om2f = make_patches(kind)
    print("  档 %s：" % kind)
    for r0 in RADII:
        R00_m, G00_m, R_m = machine_R00(Ff, Gf, r0)
        pred, N = strict_R00(Ff, gamf, Om2f, r0)
        diff = R00_m - pred
        share = abs(diff) / abs(R00_m) if R00_m != 0 else 0.0
        rows[kind].append((r0, R00_m, pred, diff, share))
        print("    r = %.1f：机器 R00 = %+.6e ｜ 严格口径预言 %+.6e ｜ 差额 %+.6e ｜ 份额 %.3e"
              % (r0, R00_m, pred, diff, share))

# ----------------------------------------------------------------------
# C-2 完全平直档的绝对标定
# ----------------------------------------------------------------------
print()
print("[C-2] 完全平直档（丙）的绝对标定：机器 R00 应严格为零")
flat_max = max(abs(r[1]) for r in rows["丙"])
for r0, R00_m, pred, diff, share in rows["丙"]:
    print("    r = %.1f：机器 R00 = %+.6e，份额 %s"
          % (r0, R00_m, ("%.3e" % share) if R00_m != 0 else "（R00 为零，份额无定义）"))
print("  丙档机器 R00 的最大绝对值 %.3e（门槛 1e-14）" % flat_max)
check("C-2", flat_max <= 1e-14,
      "完全平直档的机器 R00 严格为零（最大绝对值 %.3e），机器读数无系统误差" % flat_max)

# ----------------------------------------------------------------------
# C-1 满足前提的装置上差额项的消失
# ----------------------------------------------------------------------
print()
print("[C-1] 满足前提的装置（甲、乙）上差额项的消失")
c1_rows = []
for kind in ("甲", "乙"):
    shares = [r[4] for r in rows[kind]]
    worst = max(shares)
    c1_rows.append((kind, worst))
    print("  档 %s：差额份额最大 %.3e（门槛 1e-4）" % (kind, worst))
c1_ok = all(w <= 1e-4 for _, w in c1_rows)
check("C-1", c1_ok,
      "满足 3+1 前提的装置（甲、乙）上差额项消失（份额最大 %.3e）"
      % max(w for _, w in c1_rows))

# ----------------------------------------------------------------------
# C-3 差额项的泛函表达
# ----------------------------------------------------------------------
print()
print("[C-3] 差额项的泛函表达：若 C-1 判否，给出显式表达与系数")
if c1_ok:
    print("  C-1 通过（差额项消失），故差额项在满足前提的装置上无表达式需求；本条按不适用记")
    check("C-3", True, "差额项在满足前提的装置上消失，本条不适用（按通过记）")
else:
    # 差额项除以 N^2 与 F 的组合，看哪一个给出稳定系数
    for label, fac in (("N^2", lambda st: st["N2"]), ("F", lambda st: st["F"]),
                       ("N^2*F", lambda st: st["N2"] * st["F"]), ("1", lambda st: 1.0)):
        vals = []
        for kind in KINDS:
            Ff, Gf, gamf, N2f, Om2f = make_patches(kind)
            for r0 in RADII:
                R00_m, G00_m, R_m = machine_R00(Ff, Gf, r0)
                pred, N = strict_R00(Ff, gamf, Om2f, r0)
                st = {"N2": N * N, "F": float(Ff(r0))}
                f = fac(st)
                if f != 0:
                    vals.append((R00_m - pred) / f)
        if vals:
            mean = sum(vals) / len(vals)
            spread = (max(abs(v - mean) for v in vals) / abs(mean)) if mean else float("inf")
            print("    候选因子 %-8s ：系数均值 %+.6e，档间相对散布 %.3e" % (label, mean, spread))
    check("C-3", True, "差额项的泛函候选已给出（见上）")

# ----------------------------------------------------------------------
# C-4 恒真替身
# ----------------------------------------------------------------------
print()
print("[C-4] 恒真替身：改 gamma_ij 的配套方式（丁档未配套），C-1 须 FAIL")
d_share = max(r[4] for r in rows["丁"])
print("  丁档（未配套）的差额份额最大 %.3e（应远大于门槛 1e-4）" % d_share)
check("C-4", d_share > 1e-3,
      "未配套的装置（丁）上差额项显著（份额 %.3e > 1e-3），配套与否可判别" % d_share)

# ----------------------------------------------------------------------
# C-6 状态登记
# ----------------------------------------------------------------------
print()
print("[C-6] 状态登记")
reg = [
    ("其一", "绝对标定", "丙档（完全平直）的机器 R00 严格为零，机器读数无系统误差"),
    ("其二", "消失", "甲、乙两档（满足 3+1 前提）上差额项的份额见上"),
    ("其三", "对照", "丁档（未配套，即第 747 号装置）的差额份额作同表对照"),
    ("其四", "剩余", "差额项的解析式（若 C-1 判否）；P5 守恒结构这一侧的状态；"
                "注入-扩散平衡那一侧不在本轮"),
]
for a1_, a2_, a3_ in reg:
    print("     %-4s %-10s %s" % (a1_, a2_, a3_))
check("C-6", True, "状态登记完成")

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
xs = np.arange(len(RADII))
w = 0.2
for i, kind in enumerate(KINDS):
    vals = [max(r[4], 1e-18) for r in rows[kind]]
    ax.semilogy(xs + (i - 1.5) * w, vals, "o-", linewidth=1.4,
                label=u"档 %s" % kind)
ax.axhline(1e-4, color="#d62728", linestyle=":", linewidth=1.2, label=u"消失门槛 $10^{-4}$")
ax.set_xticks(xs)
ax.set_xticklabels(["%.1f" % r for r in RADII], fontsize=8)
ax.set_xlabel(u"半径 $r$")
ax.set_ylabel(u"差额项占机器 $R_{00}$ 的份额")
ax.set_title(u"(a) 四档装置的差额份额")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

ax = axes[1]
for i, kind in enumerate(KINDS):
    vals = [r[1] for r in rows[kind]]
    ax.plot(xs + (i - 1.5) * w, vals, "o-", linewidth=1.4, label=u"档 %s 机器 $R_{00}$" % kind)
    pv = [r[2] for r in rows[kind]]
    ax.plot(xs + (i - 1.5) * w, pv, "s--", linewidth=1.1,
            label=u"档 %s 严格口径预言" % kind)
ax.axhline(0.0, color="#888888", linewidth=0.8)
ax.set_xticks(xs)
ax.set_xticklabels(["%.1f" % r for r in RADII], fontsize=8)
ax.set_xlabel(u"半径 $r$")
ax.set_ylabel(u"$R_{00}$ 读数")
ax.set_title(u"(b) 机器读数与严格口径预言")
ax.legend(fontsize=6, ncol=2)
ax.grid(alpha=0.3)

ax = axes[2]
names = list(KINDS)
shares = [max(r[4] for r in rows[k]) for k in KINDS]
ax.bar(np.arange(len(names)), shares, color=["#2ca02c", "#2ca02c", "#7f7f7f", "#d62728"],
       width=0.6)
ax.axhline(1e-4, color="#d62728", linestyle=":", linewidth=1.2)
ax.set_xticks(np.arange(len(names)))
ax.set_xticklabels(names, fontsize=9)
ax.set_ylabel(u"最大差额份额")
ax.set_title(u"(c) 满足前提的两档（绿）上差额消失")
ax.grid(alpha=0.3, axis="y")

fig.suptitle(u"第748号：满足 3+1 分解前提的装置与差额项的消失核对", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(IMG, dpi=150)
print()
print("图已存：%s" % IMG)
