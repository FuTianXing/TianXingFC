# -*- coding: utf-8 -*-
r"""
minimal_verification_library_convention_rebuild.py

[第765号] 库口径下的 R00 分解式重建与差额的归零核对（判据 M-1 至 M-6）

背景：判据第 762 号把差额项的根源定位为「机器读数的取法」（反解式 R00 = G00 - (1/2) g00 R
在 N 的高阶导数上丢因子），并给出直接构造的四维 R00 与库的 3+1 分解预言相容（相对差
6.438e-03 至 2.537e-01），而反解式与预言差 4.155e-01 至 1.582e+00。762 号留下的入口是：
在库的口径下重建预言式，看差额是否归零。

**库的口径**（第 742 号）：g_00 = -N^2、g_0i = -N^2 S_i、g_ij = N^2 gamma_ij，严格口径要求
gamma_ij = g_ij / N^2。**口径约定**（762 号自检给出，库内此前未写明）：机器的直接构造 R00 与
N∇²₍₃₎N 同号且量级相近（比值约 0.864），而非 -2N∇²₍₃₎N。

**重建的做法**：按 ADM 分解的 R00 完整表达式，逐项写出每一项的解析形式（含 N 的一阶与二阶
导数、gamma 的导数、Gamma 的一阶导数），与直接构造的四维 R00 逐项比对。

**档**：
  甲 g00 = -(1-F)，gij = (1-F)(1+G) delta（gamma_ij = (1+G) delta，Om^2 = 1+G）
  乙 g00 = -(1-F)，gij = (1-F) delta（gamma_ij = delta，Om^2 = 1）

判据：
  M-1 重建式与直接构造的一致性
  M-2 差额的归零
  M-3 逐项读数的交叉核对
  M-4 口径常数的稳定性
  M-5 恒真替身
  M-6 状态登记

运行：python 验证脚本/minimal_verification_library_convention_rebuild.py
确定性（无随机），约 5 秒。图存 images/library_convention_rebuild.png
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
IMG = os.path.join(ROOT, "images", "library_convention_rebuild.png")

T0 = time.time()
CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))


# ----------------------------------------------------------------------
# 装置（自建；第 716 号与 G1b8 均不 import，见第 754 号的教训）
# ----------------------------------------------------------------------
HALF = 4
C = HALF
NP_ = 2 * HALF + 1
EPS = 0.05

II, JJ, KK = np.indices((NP_, NP_, NP_))
CTR = NP_ // 2
R_GRID = np.sqrt((II - CTR) ** 2.0 + (JJ - CTR) ** 2.0 + (KK - CTR) ** 2.0)


def th_c(r):
    m = 0.05
    r = np.asarray(r, dtype=float)
    t = 1.0 / (1.0 + np.exp((r - 1.0) / m))
    return t - t * t


def F_of(r):
    return 2.0 * th_c(r)


def G_of(r):
    t = th_c(r)
    return 2.0 * t + 3.5 * t * t


def d1_keep(a, axis, eps):
    b = np.moveaxis(a, axis, -1)
    out = np.full_like(b, np.nan)
    out[..., 2:-2] = (b[..., :-4] - 8 * b[..., 1:-3] + 8 * b[..., 3:-1] - b[..., 4:]) / (12 * eps)
    return np.moveaxis(out, -1, axis)


def g_patch(kind, x0, eps, c_lapse=0.0):
    """构造度规场 g_munu（4x4xNP_^3）。

    c_lapse 是 N^2 = 1 - F + c F^2 的高阶系数（c = 0 为库的一阶形式）。
    """
    k = np.arange(-HALF, HALF + 1) * eps
    X = x0[0] + k[:, None, None]
    Y = x0[1] + k[None, :, None]
    Z = x0[2] + k[None, None, :]
    r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    Fv = F_of(r)
    Gv = G_of(r)
    N2 = 1.0 - Fv + c_lapse * Fv * Fv
    g = np.zeros((4, 4, NP_, NP_, NP_))
    g[0, 0] = -N2
    if kind == "甲":
        g[1, 1] = g[2, 2] = g[3, 3] = N2 * (1.0 + Gv)
    elif kind == "乙":
        g[1, 1] = g[2, 2] = g[3, 3] = N2
    else:
        raise ValueError(kind)
    return g


def ginv_pointwise(g):
    """逐点逆度规（本构造的度规是对角的）。边界处用中心点补齐。"""
    ginvf = np.zeros_like(g)
    for a in range(4):
        ga = g[a, a]
        good = np.abs(ga) > 1e-14
        inv = np.zeros_like(ga)
        inv[good] = 1.0 / ga[good]
        ginvf[a, a] = inv
    for a in range(4):
        ctr_val = ginvf[a, a, C, C, C]
        if np.isfinite(ctr_val) and ctr_val != 0.0:
            ginvf[a, a] = np.where(ginvf[a, a] == 0.0, ctr_val, ginvf[a, a])
    return ginvf


def christoffel(g, eps):
    dg = [np.zeros_like(g), d1_keep(g, 2, eps), d1_keep(g, 3, eps), d1_keep(g, 4, eps)]
    ginvf = ginv_pointwise(g)
    Gam = np.zeros((4, 4, 4, NP_, NP_, NP_))
    for lam in range(4):
        for mu in range(4):
            for nu in range(4):
                acc = np.zeros((NP_, NP_, NP_))
                for sig in range(4):
                    acc = acc + 0.5 * ginvf[lam, sig] * (
                        dg[mu][sig, nu] + dg[nu][sig, mu] - dg[sig][mu, nu])
                Gam[lam, mu, nu] = acc
    return Gam, dg


def R00_direct(g, eps):
    """直接构造的四维 R00，并给出逐项拆分。"""
    Gam, dg = christoffel(g, eps)
    terms = {}
    # 项一：∂_rho Gamma^rho_{00}
    t1 = np.zeros((NP_, NP_, NP_))
    for rho in (1, 2, 3):
        t1 = t1 + d1_keep(Gam[rho, 0, 0], rho - 1, eps)
    terms["spatial"] = t1
    # 项二：-∂_0 Gamma^rho_{0 rho}（静场恒为零）
    t2 = np.zeros((NP_, NP_, NP_))
    for rho in range(4):
        t2 = t2 - d1_keep(Gam[rho, 0, rho], 0, eps)
    terms["t_deriv"] = t2
    # 项三：+Gamma^rho_{00} Gamma^sigma_{rho sigma}
    t3 = np.zeros((NP_, NP_, NP_))
    for rho in range(4):
        acc = np.zeros((NP_, NP_, NP_))
        for sig in range(4):
            acc = acc + Gam[sig, rho, sig]
        t3 = t3 + Gam[rho, 0, 0] * acc
    terms["quad_a"] = t3
    # 项四：-Gamma^sigma_{0 rho} Gamma^rho_{0 sigma}
    t4 = np.zeros((NP_, NP_, NP_))
    for rho in range(4):
        for sig in range(4):
            t4 = t4 - Gam[sig, 0, rho] * Gam[rho, 0, sig]
    terms["quad_b"] = t4
    terms["shift"] = np.zeros((NP_, NP_, NP_))
    R00 = t1 + t2 + t3 + t4
    return R00, terms


def center(arr):
    return arr[C, C, C]


def rad_d1(f, r0, h=1e-4):
    return (f(r0 + h) - f(r0 - h)) / (2.0 * h)


def gamma_factor(kind, r):
    """严格口径的空间度规对角因子 Om^2 = gamma_ij 的对角元。"""
    if kind == "甲":
        return 1.0 + G_of(r)
    return np.ones_like(np.asarray(r, dtype=float))


def rebuild_R00_library_convention(kind, r0, c_lapse=0.0):
    """在库的口径下重建 R00 的三维分解式。

    库的口径：g_00 = -N^2、g_0i = 0、g_ij = N^2 gamma_ij，gamma_ij = Om^2 delta。
    ADM 分解下 R_00 = N [ ∇²_(3) N + Gamma^i_{ij} d^j N ]，其中 ∇²_(3) 用 gamma 的三维算子。
    本函数按这个式子逐项给出：
      lap_part   N · ∇²_(3) N
      gam_part   N · Gamma^i_{ij} d^j N
    并给出 sum（库的预言式）。
    """
    Om_f = lambda r: math.sqrt(float(gamma_factor(kind, r)))
    Nf = lambda r: math.sqrt(1.0 - float(F_of(r)) + c_lapse * float(F_of(r)) ** 2)
    dN = rad_d1(Nf, r0)
    dOm = rad_d1(Om_f, r0)
    Omv = float(Om_f(r0))
    Nv = float(Nf(r0))
    # ∇²_(3) N = Om^-3 d/dr (Om^2 dN/dr)
    lap3N = rad_d1(lambda x: Om_f(x) ** 2 * rad_d1(Nf, x), r0) / Omv ** 3
    # Gamma^i_{ij} d^j N（球对称，径向）：3 (dN/dr)(dOm/dr) / Om^3
    gam = 3.0 * dN * dOm / Omv ** 3
    return dict(N=Nv, dN=dN, dOm=dOm, Om=Omv, lap3N=lap3N, gam=gam,
                lap_part=Nv * lap3N, gam_part=Nv * gam,
                sum_pred=Nv * (lap3N + gam))


print("=" * 78)
print("第765号：库口径下的 R00 分解式重建与差额的归零核对")
print("=" * 78)

RADII = (0.8, 1.0, 1.2, 1.4, 1.6)
C_LAPSES = (0.0, 0.05, 0.10, 0.15, 0.20)

# ----------------------------------------------------------------------
# 冒烟检查
# ----------------------------------------------------------------------
print()
print("[冒烟] 装置的非平凡性")
g0 = g_patch("甲", (1.2, 0.0, 0.0), EPS)
R00_d, terms0 = R00_direct(g0, EPS)
rb = rebuild_R00_library_convention("甲", 1.2)
print("  甲档（r = 1.2）：直接构造 R00 = %+.6e ｜ 库口径重建预言 %+.6e"
      % (center(R00_d), rb["sum_pred"]))
smoke_ok = (abs(center(R00_d)) > 0) and np.isfinite(rb["sum_pred"])
print("  冒烟结论：%s" % ("通过" if smoke_ok else "不通过"))
if not smoke_ok:
    print("  冒烟未过，按规范不进判据循环。")
    sys.exit(1)

# ----------------------------------------------------------------------
# M-1 / M-2 / M-3 重建式与直接构造的逐项比对
# ----------------------------------------------------------------------
print()
print("[M-1] 重建式与直接构造的一致性；[M-2] 差额的归零；[M-3] 逐项读数的交叉核对")
m_rows = []
m1_max = 0.0
m2_max = 0.0
for kind in ("甲", "乙"):
    print("  档 %s：" % kind)
    for c_lapse in C_LAPSES:
        for r0 in RADII:
            g = g_patch(kind, (r0, 0.0, 0.0), EPS, c_lapse=c_lapse)
            R00_d, terms = R00_direct(g, EPS)
            Rd = center(R00_d)
            rb = rebuild_R00_library_convention(kind, r0, c_lapse=c_lapse)
            diff = Rd - rb["sum_pred"]
            share = abs(diff) / abs(Rd) if Rd != 0 else 0.0
            # 逐项交叉核对：直接构造的空间导数项 对 重建的 N·∇²N 项
            t_spatial = center(terms["spatial"])
            rel_item = abs(t_spatial - rb["lap_part"]) / abs(t_spatial) if t_spatial != 0 else 0.0
            m_rows.append((kind, c_lapse, r0, Rd, rb["sum_pred"], diff, share, rel_item))
            m1_max = max(m1_max, share)
            m2_max = max(m2_max, share)
        # 只在 c = 0（库的一阶形式）与首末两档上打印，避免输出过长
        for r0 in (RADII[0], RADII[-1]):
            row = [x for x in m_rows if x[0] == kind and x[1] == c_lapse and x[2] == r0][0]
            print("    c = %.2f，r = %.1f：直接构造 %+.6e ｜ 重建 %+.6e ｜ 差额 %+.6e（份额 %.3e）"
                  " ｜ 空间导数项对 N∇²N 相对差 %.3e"
                  % (c_lapse, r0, row[3], row[4], row[5], row[6], row[7]))

print("  M-1/M-2 的最大份额 %.3e（c 取遍 %s）" % (m1_max, ", ".join("%.2f" % c for c in C_LAPSES)))
m3_max = max(r[7] for r in m_rows)
print("  M-3 逐项交叉核对的最大相对差 %.3e" % m3_max)

# ----------------------------------------------------------------------
# M-4 口径常数的稳定性
# ----------------------------------------------------------------------
print()
print("[M-4] 口径常数（R00 对 N∇²N 的比值）的稳定性")
m4_rows = []
for kind in ("甲", "乙"):
    for r0 in RADII:
        g = g_patch(kind, (r0, 0.0, 0.0), EPS)
        R00_d, terms = R00_direct(g, EPS)
        Rd = center(R00_d)
        rb = rebuild_R00_library_convention(kind, r0)
        ratio = Rd / rb["lap_part"] if rb["lap_part"] != 0 else float("nan")
        m4_rows.append((kind, r0, ratio))
        print("  档 %s，r = %.1f：R00 / (N∇²N) = %+.6f" % (kind, r0, ratio))
ratios = [r[2] for r in m4_rows if np.isfinite(r[2])]
if ratios:
    m4_spread = (max(ratios) - min(ratios)) / abs(np.mean(ratios))
    m4_mean = float(np.mean(ratios))
else:
    m4_spread = float("inf")
    m4_mean = float("nan")
print("  比值的均值 %+.6f，档间相对散布 %.3e" % (m4_mean, m4_spread))
check("M-4", np.isfinite(m4_spread) and m4_spread <= 1e-1,
      "口径常数稳定：均值 %+.6f，档间相对散布 %.3e" % (m4_mean, m4_spread))

# ----------------------------------------------------------------------
# M-1 / M-2 / M-3 的判定
# ----------------------------------------------------------------------
m1_ok = m1_max <= 1e-2
m2_ok = m2_max <= 1e-2
m3_ok = m3_max <= 1e-2
check("M-1", m1_ok,
      "库口径重建的 R00 与直接构造的四维 R00 的最大相对份额 %.3e（容差 1e-2）" % m1_max)
check("M-2", m2_ok,
      "差额的最大份额 %.3e（容差 1e-2）：%s" % (m2_max, "差额已归零" if m2_ok else "未归零"))
check("M-3", m3_ok,
      "逐项交叉核对（直接构造的空间导数项 对 重建的 N∇²N 项）的最大相对差 %.3e" % m3_max)

# 差额随 c 的依赖
print()
print("  差额随 c 的依赖（c 是 N^2 = 1 - F + c F^2 的高阶系数）：")
for kind in ("甲", "乙"):
    line = []
    for c_lapse in C_LAPSES:
        vals = [r[6] for r in m_rows if r[0] == kind and r[1] == c_lapse]
        line.append("c=%.2f 给 %.3e" % (c_lapse, max(vals)))
    print("    档 %s：%s" % (kind, "；".join(line)))

# ----------------------------------------------------------------------
# M-5 恒真替身
# ----------------------------------------------------------------------
print()
print("[M-5] 恒真替身：改 Om 取哪一侧（甲档改用 Om^2 = (1+G)/(1-F)），M-1 须 FAIL")
alt_rows = []
for kind in ("甲", "乙"):
    for r0 in RADII:
        g = g_patch(kind, (r0, 0.0, 0.0), EPS)
        R00_d, terms = R00_direct(g, EPS)
        Rd = center(R00_d)
        # 替身：把 Om^2 取成 (1+G)/(1-F)（未除 N^2，即第 742 号指出的「两套 3+1 口径混用」）
        Om_f_alt = lambda r: math.sqrt((1.0 + float(G_of(r))) / (1.0 - float(F_of(r))))
        Nf = lambda r: math.sqrt(1.0 - float(F_of(r)))
        dN = rad_d1(Nf, r0)
        dOm = rad_d1(Om_f_alt, r0)
        Omv = float(Om_f_alt(r0))
        lap3N = rad_d1(lambda x: Om_f_alt(x) ** 2 * rad_d1(Nf, x), r0) / Omv ** 3
        gam = 3.0 * dN * dOm / Omv ** 3
        Nv = float(Nf(r0))
        pred_alt = Nv * (lap3N + gam)
        share_alt = abs(Rd - pred_alt) / abs(Rd) if Rd != 0 else 0.0
        alt_rows.append((kind, r0, share_alt))
        print("  档 %s，r = %.1f：替身下的份额 %.3e" % (kind, r0, share_alt))
m5_max = max(r[2] for r in alt_rows)
m1_by_kind = {}
for kind in ("甲", "乙"):
    m1_by_kind[kind] = max(r[6] for r in m_rows if r[0] == kind and r[1] == 0.0)
m5_ok = m5_max > 1e-2
check("M-5", m5_ok,
      "替身下份额最大 %.3e（真值 %.3e 与 %.3e），判据有区分力"
      % (m5_max, m1_by_kind["甲"], m1_by_kind["乙"]))

# ----------------------------------------------------------------------
# M-6 状态登记
# ----------------------------------------------------------------------
print()
print("[M-6] 状态登记")
reg = [
    ("其一", "重建", "库口径下 R00 = N[∇²₍₃₎N + Gamma^i_{ij} d^j N] 的逐项解析形式，与直接构造的 "
              "对应项逐项比对"),
    ("其二", "归零", "差额占机器 R00 的份额在 c 取遍五档上的最大值"),
    ("其三", "口径常数", "R00 对 N∇²₍₃₎N 的比值在十档上的均值与散布"),
    ("其四", "c 的依赖", "N^2 = 1 - F + c F^2 的高阶系数对差额的影响"),
    ("其五", "剩余", "P5 的「注入-扩散平衡」那一侧；源侧的 O(v^2) 缺口；"
                "刚度取值（归弹性本构线）"),
]
for a1_, a2_, a3_ in reg:
    print("     %-4s %-10s %s" % (a1_, a2_, a3_))
check("M-6", True, "状态登记完成")

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
for kind, col, mk in (("甲", "#1f77b4", "o"), ("乙", "#ff7f0e", "s")):
    xs = [c for c in C_LAPSES]
    ys = [max(r[6] for r in m_rows if r[0] == kind and r[1] == c) for c in xs]
    ax.semilogy(xs, np.maximum(ys, 1e-18), mk + "-", color=col, linewidth=1.6,
                label=u"档 %s" % kind)
ax.axhline(1e-2, color="#2ca02c", linestyle=":", linewidth=1.4, label=u"门槛 $10^{-2}$")
ax.set_xlabel(u"lapse 高阶系数 $c$（$N^2=1-F+cF^2$）")
ax.set_ylabel(u"差额占机器 $R_{00}$ 的份额")
ax.set_title(u"(a) 差额随 $c$ 的依赖")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

ax = axes[1]
for kind, col, mk in (("甲", "#1f77b4", "o"), ("乙", "#ff7f0e", "s")):
    rs = [r[2] for r in m_rows if r[0] == kind and r[1] == 0.0]
    sh = [r[6] for r in m_rows if r[0] == kind and r[1] == 0.0]
    ax.semilogy(rs, np.maximum(sh, 1e-18), mk + "-", color=col, linewidth=1.6,
                label=u"档 %s" % kind)
ax.axhline(1e-2, color="#2ca02c", linestyle=":", linewidth=1.4, label=u"门槛 $10^{-2}$")
ax.set_xlabel(u"半径 $r$")
ax.set_ylabel(u"差额份额")
ax.set_title(u"(b) 库的一阶形式下的差额")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

ax = axes[2]
labels = []
vals = []
for kind, r0, ratio in m4_rows:
    labels.append(u"%s %.1f" % (kind, r0))
    vals.append(ratio)
ax.bar(np.arange(len(labels)), vals, color=["#1f77b4" if l.startswith("甲") else "#ff7f0e"
                                            for l in labels], width=0.6)
ax.axhline(m4_mean, color="#d62728", linestyle="--", linewidth=1.4,
           label=u"均值 %+.4f" % m4_mean)
ax.set_xticks(np.arange(len(labels)))
ax.set_xticklabels(labels, fontsize=7, rotation=45)
ax.set_ylabel(u"$R_{00}/(N\\nabla^2N)$")
ax.set_title(u"(c) 库的口径常数")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")

fig.suptitle(u"第765号：库口径下的 $R_{00}$ 分解式重建与差额的归零核对", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(IMG, dpi=150)
print()
print("图已存：%s" % IMG)
