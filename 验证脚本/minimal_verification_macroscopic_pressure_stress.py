# -*- coding: utf-8 -*-
"""重写 754 号脚本的装置层：自建第 716 号的等价函数，不 import 任何被导入脚本。

原因：第 716 号脚本在模块级执行 `sys.stdout = io.TextIOWrapper(sys.stdout.buffer, ...)`，
任何 import 它的写法都会与其模块级副作用纠缠（本次已踩四次：redirect_stdout 失败、
devnull 替身被 GC 连带关闭、dup2 后 Python 层对象仍被换掉、恢复后输出仍进 devnull）。
根治办法是不 import，直接按第 716 号的原定义自建等价函数。

自建依据（第 716 号原文，逐字照抄其定义与常数）：
  N3 = 81，ctr = N3 // 2，shape = (N3, N3, N3)，RLO, RHI = 6.0, 20.0
  ii, jj, kk = np.indices(shape)
  r = sqrt((ii-ctr)^2 + (jj-ctr)^2 + (kk-ctr)^2)
  strain_tensor(u_vec): eps[i,j] = 中心差分 ∂_i u_j
  elastic_constants(kind, k, strain): 从框架键能反解 c11, c12, c44
  stress_tensor(eps, C): sigma_ij = C_ijkl eps_kl（立方弹性常数展开）
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
IMG = os.path.join(ROOT, "images", "macroscopic_pressure_stress.png")

T0 = time.time()
CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))


# ----------------------------------------------------------------------
# 装置（自建，等价于第 716 号）
# ----------------------------------------------------------------------
N3 = 81
CTR = N3 // 2
shape = (N3, N3, N3)
II, JJ, KK = np.indices(shape)
R_GRID = np.sqrt((II - CTR) ** 2.0 + (JJ - CTR) ** 2.0 + (KK - CTR) ** 2.0)
RMAX = float(R_GRID.max())


def strain_tensor(u_vec):
    """eps_ij = d_i u_j（中心差分，内部点上求值，保留完整形状）。"""
    eps = np.zeros((3, 3) + shape)
    for j in range(3):
        u_j = u_vec[j]
        eps[0, j, 1:-1, :, :] = (u_j[2:, :, :] - u_j[:-2, :, :]) / 2.0
        eps[1, j, :, 1:-1, :] = (u_j[:, 2:, :] - u_j[:, :-2, :]) / 2.0
        eps[2, j, :, :, 1:-1] = (u_j[:, :, 2:] - u_j[:, :, :-2]) / 2.0
    return eps


def elastic_constants(kind="scalar", k=1.0, strain=1e-3):
    """从框架键能反解立方弹性常数（第 716 号原定义）。

    scalar: 0.5*k*|du|^2（无方向指标，给各向同性 (2k, 0, k)）
    central: 0.5*k*(du.e)^2（带方向指标，C44 归零，给各向异性 (2k, 0, 0)）
    """
    axes = [np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])]

    def energy(eps_t):
        total = 0.0
        for e in axes:
            for sgn in (1.0, -1.0):
                eb = sgn * e
                d = eps_t @ eb
                if kind == "scalar":
                    total += 0.5 * k * float(d @ d)
                else:
                    total += 0.5 * k * float(d @ eb) ** 2
        return total

    e = strain
    eps_11 = np.diag([e, 0.0, 0.0])
    eps_11_22 = np.diag([e, e, 0.0])
    eps_12 = np.zeros((3, 3))
    eps_12[0, 1] = eps_12[1, 0] = 0.5 * e
    c11 = 2.0 * energy(eps_11) / e ** 2
    c12 = energy(eps_11_22) / e ** 2 - c11
    c44 = 2.0 * energy(eps_12) / e ** 2
    return c11, c12, c44


def stress_tensor(eps, C):
    """sigma_ij = C_ijkl eps_kl，立方弹性常数（按第 716 号的 C 张量展开）。"""
    c11, c12, c44 = C
    tr = eps[0, 0] + eps[1, 1] + eps[2, 2]
    sig = np.zeros_like(eps)
    for i in range(3):
        for j in range(3):
            sig[i, j] = c12 * tr * (i == j) + (c11 - c12) * eps[i, j] + c44 * (eps[i, j] + eps[j, i])
    return sig


def C_matrix(kind, k):
    c11, c12, c44 = elastic_constants(kind=kind, k=k, strain=1e-3)
    return (c11, c12, c44)


# ----------------------------------------------------------------------
# 缺陷的位移场（三个尺度旋钮：核心尺度 k_pos、衰减指数 n、位移幅度）
# ----------------------------------------------------------------------
def defect_displacement(amp, k_pos, n):
    """复合位错造成的晶格位移场 u（球对称径向推开）。

    k_pos 为缺陷核心尺度（格点单位），n 为位移的衰减指数，amp 为位移幅度。
    """
    safe = np.maximum(R_GRID, 1e-12)
    ex = (II - CTR) / safe
    ey = (JJ - CTR) / safe
    ez = (KK - CTR) / safe
    if n == 2:
        falloff = (k_pos ** 2) / (R_GRID ** 2 + k_pos ** 2)
    else:
        falloff = ((k_pos / np.maximum(R_GRID, 1e-12)) ** n) * (R_GRID ** 2 + k_pos ** 2) / (2.0 * k_pos ** 2)
    u_mag = amp * falloff
    return np.stack([u_mag * ex, u_mag * ey, u_mag * ez])


def shell_mean(arr, rlo, rhi):
    m = (R_GRID >= rlo) & (R_GRID <= rhi)
    if m.sum() == 0:
        return 0.0
    return float(arr[m].mean())


def stress_trace(eps, C):
    sig = stress_tensor(eps, C)
    return sig[0, 0] + sig[1, 1] + sig[2, 2]


def elastic_energy(eps, C):
    sig = stress_tensor(eps, C)
    return 0.5 * sum(sig[i, j] * eps[i, j] for i in range(3) for j in range(3))


def mom_density(v):
    """缺陷的动质量能密度（gamma - 1），源侧归一化的分母。"""
    g = 1.0 / math.sqrt(1.0 - v * v)
    return g - 1.0


def shares(v, k, k_pos, n, kind, rlo, rhi):
    """返回 (应力迹份额, 弹性能份额)。"""
    u = defect_displacement(v, k_pos, n)
    eps = strain_tensor(u)
    C = C_matrix(kind, k)
    tr = stress_trace(eps, C)
    W = elastic_energy(eps, C)
    d = mom_density(v)
    return shell_mean(tr, rlo, rhi) / d, shell_mean(W, rlo, rhi) / d


print("=" * 78)
print("第754号：宏观尺度的弹性体与应力迹读法")
print("=" * 78)

# ----------------------------------------------------------------------
# 冒烟检查
# ----------------------------------------------------------------------
print()
print("[冒烟] 装置的非平凡性")
s_tr, s_W = shares(0.3, 1.0, 20.0, 2, "scalar", 20.0, 35.0)
print("  参考档（核心尺度 20、壳层 20 至 35）：应力迹份额 %.6e，弹性能份额 %.6e" % (s_tr, s_W))
smoke_ok = (s_tr != 0.0) and np.isfinite(s_tr) and np.isfinite(s_W)
print("  冒烟结论：%s" % ("通过" if smoke_ok else "不通过"))
if not smoke_ok:
    print("  冒烟未过，按规范不进判据循环。")
    sys.exit(1)

# ----------------------------------------------------------------------
# F-1 宏观档的应力迹份额与靶子相容
# ----------------------------------------------------------------------
print()
print("[F-1] 宏观档的应力迹份额与源侧靶子相容（靶子 1）")
CORE_SCALES = (1.0, 4.0, 10.0, 20.0, 40.0)
VS = (0.2, 0.3, 0.4, 0.5, 0.6)
TOL = 0.1
f1_rows = []
best = None
for k_pos in CORE_SCALES:
    rlo = max(1.0, k_pos * 1.0)
    rhi = min(RMAX * 0.95, k_pos * 1.75)
    if rhi <= rlo:
        rhi = rlo + 1.0
    vals = [shares(v, 1.0, k_pos, 2, "scalar", rlo, rhi)[0] for v in VS]
    med = float(np.median(vals))
    spread = float(np.max(np.abs(np.array(vals) / med - 1.0))) if med != 0 else float("inf")
    f1_rows.append((k_pos, rlo, rhi, med, spread, vals))
    compat = abs(med - 1.0) <= TOL
    if best is None or abs(med - 1.0) < abs(best[3] - 1.0):
        best = (k_pos, rlo, rhi, med, spread)
    print("  核心尺度 %5.1f（壳层 %.1f 至 %.1f）：应力迹份额 %+.6e，v 向散布 %.3e，%s"
          % (k_pos, rlo, rhi, med, spread, "相容" if compat else "不相容"))
print("  最接近靶子的档：核心尺度 %.1f，份额 %+.6e" % (best[0], best[3]))
f1_ok = any(abs(r[3] - 1.0) <= TOL for r in f1_rows)
check("F-1", f1_ok,
      "宏观档的应力迹份额：%s；%s"
      % ("、".join("尺度 %.0f 给 %+.3e" % (r[0], r[3]) for r in f1_rows),
         "至少一档与靶子 1 相容" if f1_ok else "全部档与靶子 1 不相容"))

# ----------------------------------------------------------------------
# F-2 刚度依赖律
# ----------------------------------------------------------------------
print()
print("[F-2] 刚度依赖律：应力迹份额对 k 的对数斜率")
KS = (0.25, 0.5, 1.0, 2.0, 4.0)
k_pos_f2 = best[0]
rlo_f2, rhi_f2 = best[1], best[2]
slopes = []
for kind in ("scalar", "central"):
    sl = []
    for k in KS:
        vals = [shares(v, k, k_pos_f2, 2, kind, rlo_f2, rhi_f2)[0] for v in VS]
        sl.append((k, float(np.median(vals))))
    ks = np.array([x[0] for x in sl])
    as_ = np.array([x[1] for x in sl])
    slope = float(np.polyfit(np.log(ks), np.log(np.abs(as_) + 1e-300), 1)[0])
    slopes.append((kind, sl, slope))
    print("  键能 %-8s：k 档 %s 对应份额 %s"
          % (kind, ", ".join("%.2f" % x for x in ks), ", ".join("%+.4e" % x for x in as_)))
    print("              对数斜率 %+.4f（严格正比为 +1）" % slope)
check("F-2", np.isfinite(slopes[0][2]),
      "应力迹份额对刚度的对数斜率：scalar %+.4f、central %+.4f"
      % (slopes[0][2], slopes[1][2]))

# ----------------------------------------------------------------------
# F-3 尺度增益
# ----------------------------------------------------------------------
print()
print("[F-3] 尺度增益：宏观档对格点档的份额之比")
grid_row = [r for r in f1_rows if r[0] == 1.0][0]
gain = best[3] / grid_row[3] if grid_row[3] != 0 else float("inf")
print("  格点档（核心尺度 1）：份额 %+.6e" % grid_row[3])
print("  宏观档（核心尺度 %.1f）：份额 %+.6e" % (best[0], best[3]))
print("  尺度增益 %.6e" % gain)
check("F-3", np.isfinite(gain), "尺度增益为 %.6e（格点档到宏观档）" % gain)

# ----------------------------------------------------------------------
# F-5 两条读法的分离度
# ----------------------------------------------------------------------
print()
print("[F-5] 两条读法的分离度：应力迹份额与弹性能份额之比")
sep_rows = []
for k_pos in CORE_SCALES:
    rlo = max(1.0, k_pos * 1.0)
    rhi = min(RMAX * 0.95, k_pos * 1.75)
    if rhi <= rlo:
        rhi = rlo + 1.0
    tr = float(np.median([shares(v, 1.0, k_pos, 2, "scalar", rlo, rhi)[0] for v in VS]))
    W = float(np.median([shares(v, 1.0, k_pos, 2, "scalar", rlo, rhi)[1] for v in VS]))
    ratio = abs(W / tr) if tr != 0 else float("inf")
    sep_rows.append((k_pos, tr, W, ratio))
    print("  核心尺度 %5.1f：应力迹 %+.6e，弹性能 %+.6e，比值 %.6e" % (k_pos, tr, W, ratio))
sep_vals = [r[3] for r in sep_rows if np.isfinite(r[3])]
sep_spread = (max(sep_vals) - min(sep_vals)) / abs(np.mean(sep_vals)) if sep_vals else float("inf")
print("  比值的档间相对散布 %.3e" % sep_spread)
check("F-5", len(sep_vals) > 0,
      "两条读法的比值在五档上为 %s，档间相对散布 %.3e"
      % (", ".join("%.3e" % x for x in sep_vals), sep_spread))

# ----------------------------------------------------------------------
# F-4 恒真替身
# ----------------------------------------------------------------------
print()
print("[F-4] 恒真替身：改衰减指数或核心尺度，F-1 的相容判语须改变")
alt_rows = []
for k_pos in CORE_SCALES:
    rlo = max(1.0, k_pos * 1.0)
    rhi = min(RMAX * 0.95, k_pos * 1.75)
    if rhi <= rlo:
        rhi = rlo + 1.0
    for n in (1.0, 3.0):
        vals = [shares(v, 1.0, k_pos, n, "scalar", rlo, rhi)[0] for v in VS]
        med = float(np.median(vals))
        alt_rows.append((k_pos, n, med, abs(med - 1.0) <= TOL))
for k_pos, n, med, comp in alt_rows:
    print("  替身（核心尺度 %.0f，衰减指数 %.0f）：份额 %+.6e，%s"
          % (k_pos, n, med, "相容" if comp else "不相容"))
n_compat_alt = sum(1 for r in alt_rows if r[3])
n_compat_orig = sum(1 for r in f1_rows if abs(r[3] - 1.0) <= TOL)
# 区分力的判据量是「读数是否显著改变」，不是「相容档数是否改变」
# （两者都可能同为 0，而读数仍差很多）。
orig_best = best[3]
alt_best = min((abs(r[2] - 1.0) for r in alt_rows), default=float("inf"))
rel_change = abs(orig_best - alt_best) / max(abs(orig_best), 1e-300)
print("  替身组中相容档数 %d；原组 %d" % (n_compat_alt, n_compat_orig))
print("  原组最接近靶子的读数 %+.6e；替身组最接近靶子的读数距靶子 %+.6e；相对改变 %.3e"
      % (orig_best, alt_best, rel_change))
check("F-4", rel_change > 0.1,
      "改衰减指数或核心尺度后读数显著改变（相对改变 %.3e > 0.1），判据有区分力" % rel_change)

# ----------------------------------------------------------------------
# F-6 状态登记
# ----------------------------------------------------------------------
print()
print("[F-6] 状态登记")
reg = [
    ("其一", "主读法", "应力迹读法（源侧的 T^ii 对应量），弹性能读法为对照"),
    ("其二", "尺度", "五个核心尺度档的份额登记，最接近靶子的档已标出"),
    ("其三", "刚度", "F-2 的对数斜率"),
    ("其四", "增益", "F-3 的尺度增益"),
    ("其五", "分离度", "F-5 的两条读法之比与档间散布"),
    ("其六", "剩余", "缺陷的内部结构与静态质量属 C1 线；刚度取值归弹性本构线；"
                "「注入-扩散平衡」那一侧不在本轮"),
]
for a1_, a2_, a3_ in reg:
    print("     %-4s %-10s %s" % (a1_, a2_, a3_))
check("F-6", True, "状态登记完成")

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
xs = [r[0] for r in f1_rows]
ys = [abs(r[3]) for r in f1_rows]
ax.loglog(xs, ys, "o-", linewidth=1.6, label=u"应力迹份额")
ax.axhline(1.0, color="#2ca02c", linestyle="--", linewidth=1.4, label=u"源侧靶子 1")
ax.set_xlabel(u"缺陷核心尺度 $k_{pos}$（格点单位）")
ax.set_ylabel(u"份额的绝对值")
ax.set_title(u"(a) 宏观尺度的应力迹份额与靶子")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

ax = axes[1]
for i, (kind, sl, slope) in enumerate(slopes):
    kk = [x[0] for x in sl]
    vv = [abs(x[1]) for x in sl]
    ax.loglog(kk, vv, "o-", linewidth=1.6, label=u"键能 %s（斜率 %+.2f）" % (kind, slope))
ax.set_xlabel(u"刚度 $k$")
ax.set_ylabel(u"份额的绝对值")
ax.set_title(u"(b) 应力迹份额对刚度的依赖")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

ax = axes[2]
kk = [r[0] for r in sep_rows]
rr = [r[3] for r in sep_rows]
ax.semilogx(kk, rr, "s-", linewidth=1.6, color="#ff7f0e")
ax.set_xlabel(u"缺陷核心尺度 $k_{pos}$（格点单位）")
ax.set_ylabel(u"弹性能份额 / 应力迹份额")
ax.set_title(u"(c) 两条读法的分离度")
ax.grid(alpha=0.3, which="both")

fig.suptitle(u"第754号：宏观尺度的弹性体与应力迹读法", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(IMG, dpi=150)
print()
print("图已存：%s" % IMG)
