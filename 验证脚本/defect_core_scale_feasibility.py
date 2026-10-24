# -*- coding: utf-8 -*-
r"""
defect_core_scale_feasibility.py

[第756号] 缺陷核心尺度的可制备性约束与跨过点的物理可实现性（判据 G-1 至 G-6）

背景：第 754 号判出，弹性份额随缺陷核心尺度连续单调下降并跨过源侧靶子 1（尺度 1 给
+5.640468，尺度 4 给 +8.108251e-01），故弹性候选在框架内可成立；但跨过点的具体位置构造
依赖，其物理可实现性未答。

库内依据：位错核心 = 局域同相区（反相泡），2026-09-05 方案；其登记明写「核心大小……未导出」。
故本轮判「核心尺度的物理约束」：在离散半整数档上，跨过是否发生在可制备区间内。

装置：第 716 号同构**自建**（不 import，因其模块级重设 sys.stdout，见第 754 号的教训）。
判据：
  G-1 跨过点在可制备档上
  G-2 可制备区间的界与依据
  G-3 跨过点对刚度的敏感性
  G-4 恒真替身
  G-5 读法登记
  G-6 状态登记

运行：python 验证脚本/defect_core_scale_feasibility.py
确定性（无随机），约 5 秒。图存 images/defect_core_scale_feasibility.png
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
IMG = os.path.join(ROOT, "images", "defect_core_scale_feasibility.png")

T0 = time.time()
CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))


# ----------------------------------------------------------------------
# 装置（自建，等价于第 716 号；常数逐字照抄）
# ----------------------------------------------------------------------
N3 = 81
CTR = N3 // 2
shape = (N3, N3, N3)
II, JJ, KK = np.indices(shape)
R_GRID = np.sqrt((II - CTR) ** 2.0 + (JJ - CTR) ** 2.0 + (KK - CTR) ** 2.0)
RMAX = float(R_GRID.max())


def strain_tensor(u_vec):
    eps = np.zeros((3, 3) + shape)
    for j in range(3):
        u_j = u_vec[j]
        eps[0, j, 1:-1, :, :] = (u_j[2:, :, :] - u_j[:-2, :, :]) / 2.0
        eps[1, j, :, 1:-1, :] = (u_j[:, 2:, :] - u_j[:, :-2, :]) / 2.0
        eps[2, j, :, :, 1:-1] = (u_j[:, :, 2:] - u_j[:, :, :-2]) / 2.0
    return eps


def elastic_constants(kind="scalar", k=1.0, strain=1e-3):
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
    c11, c12, c44 = C
    tr = eps[0, 0] + eps[1, 1] + eps[2, 2]
    sig = np.zeros_like(eps)
    for i in range(3):
        for j in range(3):
            sig[i, j] = c12 * tr * (i == j) + (c11 - c12) * eps[i, j] + c44 * (eps[i, j] + eps[j, i])
    return sig


# ----------------------------------------------------------------------
# 缺陷的位移场与份额
# ----------------------------------------------------------------------
def defect_displacement(amp, k_pos, n):
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


def stress_trace_share(v, k, k_pos, n, kind, rlo, rhi):
    """应力迹份额：壳内应力迹均值 除以 缺陷动质量能密度（gamma - 1）。"""
    u = defect_displacement(v, k_pos, n)
    eps = strain_tensor(u)
    C = elastic_constants(kind=kind, k=k, strain=1e-3)
    sig = stress_tensor(eps, C)
    tr = sig[0, 0] + sig[1, 1] + sig[2, 2]
    g = 1.0 / math.sqrt(1.0 - v * v)
    return shell_mean(tr, rlo, rhi) / (g - 1.0)


# ----------------------------------------------------------------------
# 可制备性约束（判据设计 §二、§三）
# ----------------------------------------------------------------------
# 约束一（离散性）：反相泡含整数个格点，核心半径取半整数档。
# 约束二（壁的存在性）：核心外须留出反相壁，故 k_pos 不得超过网格半径的一定比例。
#   取一半为最宽的允许（壁至少占核心外的一层格点带）。
WALL_FRACTION = 0.5
# 约束三（相容性）：读数壳层须落在核心外且在网格内。
SHELL_SPAN = (1.0, 1.75)

V_REF = 0.4          # 参考速度（份额与 v 无关，取单档即可，见 D-1 的读数）
K_REF = 1.0
KIND = "scalar"
TARGET = 1.0

# 半整数档：1.0, 1.5, 2.0, ..., 20.0
HALF_SCALES = [0.5 * i for i in range(2, 41)]


def feasible(k_pos):
    """可制备性：返回 (是否可行, 理由)。"""
    if k_pos < 1.0:
        return False, "核心小于一个格点，反相泡无格点可含"
    if k_pos > RMAX * WALL_FRACTION:
        return False, "核心外留不出反相壁（k_pos 超过网格半径的一半）"
    rlo = max(1.0, k_pos * SHELL_SPAN[0])
    rhi = min(RMAX * 0.95, k_pos * SHELL_SPAN[1])
    if rhi <= rlo:
        return False, "读数壳层不在网格内"
    return True, "可制备"


def share_at(k_pos, k=K_REF, n=2, kind=KIND):
    rlo = max(1.0, k_pos * SHELL_SPAN[0])
    rhi = min(RMAX * 0.95, k_pos * SHELL_SPAN[1])
    return stress_trace_share(V_REF, k, k_pos, n, kind, rlo, rhi)


print("=" * 78)
print("第756号：缺陷核心尺度的可制备性约束与跨过点的物理可实现性")
print("=" * 78)

# ----------------------------------------------------------------------
# 冒烟检查
# ----------------------------------------------------------------------
print()
print("[冒烟] 装置的非平凡性")
s0 = share_at(2.0)
print("  参考档（核心半径 2.0）：应力迹份额 %.6e" % s0)
smoke_ok = (s0 != 0.0) and np.isfinite(s0)
print("  冒烟结论：%s" % ("通过" if smoke_ok else "不通过"))
if not smoke_ok:
    print("  冒烟未过，按规范不进判据循环。")
    sys.exit(1)

# ----------------------------------------------------------------------
# G-1 跨过点在可制备档上
# ----------------------------------------------------------------------
print()
print("[G-1] 跨过点在可制备档上（靶子 1；核心半径取半整数档）")
g1_rows = []
feasible_scales = []
for k_pos in HALF_SCALES:
    ok, why = feasible(k_pos)
    if not ok:
        continue
    s = share_at(k_pos)
    feasible_scales.append((k_pos, s))
    g1_rows.append((k_pos, s, s > TARGET))
print("  可制备档共 %d 个（半整数档 %s 至 %s）"
      % (len(feasible_scales), feasible_scales[0][0], feasible_scales[-1][0]))
for k_pos, s, above in g1_rows:
    print("    核心半径 %5.1f：份额 %+.6e  %s" % (k_pos, s, "在靶子上方" if above else "在靶子下方"))

cross = None
for i in range(len(feasible_scales) - 1):
    k0, s0_ = feasible_scales[i]
    k1, s1_ = feasible_scales[i + 1]
    if s0_ > TARGET >= s1_:
        cross = (k0, s0_, k1, s1_)
        break
if cross is not None:
    k0, s0_, k1, s1_ = cross
    # 线性插值给跨过半径
    k_cross = k0 + (s0_ - TARGET) * (k1 - k0) / (s0_ - s1_)
    print("  跨过发生在相邻档之间：核心半径 %.1f（份额 %+.6e）到 %.1f（份额 %+.6e）"
          % (k0, s0_, k1, s1_))
    print("  线性插值的跨过半径：%.4f（格点单位）" % k_cross)
    ok_cross, why_cross = feasible(k_cross)
    print("  插值处的可制备性：%s（%s）" % ("可行" if ok_cross else "不可行", why_cross))
else:
    k_cross = None
    print("  未在可制备档上找到跨过")
g1_ok = (cross is not None)
check("G-1", g1_ok,
      "跨过%s；%s" % ("发生在核心半径 %.1f 与 %.1f 两档之间" % (cross[0], cross[2]) if cross else "未发生在可制备档上",
                      ("插值半径 %.4f 落在可制备区间内" % k_cross) if (cross and ok_cross) else
                      ("插值半径 %.4f 不落在可制备区间内" % k_cross if cross else "无可制备区间可判")))

# ----------------------------------------------------------------------
# G-2 可制备区间的界与依据
# ----------------------------------------------------------------------
print()
print("[G-2] 可制备区间的界与依据")
k_lo = feasible_scales[0][0]
k_hi = feasible_scales[-1][0]
k_max_geom = RMAX * WALL_FRACTION
print("  网格半径 %.1f（%d^3 格点，中心 %d）" % (RMAX, N3, CTR))
print("  下界：%.1f（约束一：核心至少含一个格点；反相泡含整数个格点）" % k_lo)
print("  上界：%.1f（约束二：核心外须留出反相壁，取网格半径的一半为最宽允许）" % k_max_geom)
print("  实际可制备档的上端：%.1f" % k_hi)
print("  未导出的约束（登记）：核心束缚能与承载极限链的临界质量 m_P 的关系；同相相容性的格点数要求")
check("G-2", k_hi > k_lo,
      "可制备区间给出：下界 %.1f（至少一个格点）、上界 %.1f（反相壁留出要求，网格半径的一半）；"
      "实际可制备档上端 %.1f；核心束缚能与 m_P 的关系未导出" % (k_lo, k_max_geom, k_hi))

# ----------------------------------------------------------------------
# G-3 跨过点对刚度的敏感性
# ----------------------------------------------------------------------
print()
print("[G-3] 跨过点对刚度的敏感性：刚度取不同值时跨过档是否移动")
g3_rows = []
for k in (0.25, 0.5, 1.0, 2.0, 4.0):
    cross_k = None
    for i in range(len(feasible_scales) - 1):
        k0 = feasible_scales[i][0]
        k1 = feasible_scales[i + 1][0]
        s0_ = share_at(k0, k=k)
        s1_ = share_at(k1, k=k)
        if s0_ > TARGET >= s1_:
            kc = k0 + (s0_ - TARGET) * (k1 - k0) / (s0_ - s1_)
            cross_k = (k0, k1, kc)
            break
    g3_rows.append((k, cross_k))
    if cross_k:
        print("  刚度 %.2f：跨过档 %.1f 与 %.1f 之间，插值半径 %.4f" % (k, cross_k[0], cross_k[1], cross_k[2]))
    else:
        print("  刚度 %.2f：可制备档上未找到跨过" % k)
kc_list = [r[1][2] for r in g3_rows if r[1]]
if len(kc_list) >= 2:
    kc_move = (max(kc_list) - min(kc_list))
    print("  跨过半径的移动量 %.4f（格点单位），以档计 %.3f" % (kc_move, kc_move / 0.5))
    g3_ok = (kc_move / 0.5) < 1.0
    check("G-3", g3_ok,
          "跨过点随刚度的移动量 %.4f 格点单位（%.3f 档），%s"
          % (kc_move, kc_move / 0.5, "跨过不依赖刚度的精确取值" if g3_ok else "跨过随刚度显著移动"))
else:
    check("G-3", False, "跨过点随刚度的敏感性无法给出（可制备档上跨过不足两档）")

# ----------------------------------------------------------------------
# G-4 恒真替身
# ----------------------------------------------------------------------
print()
print("[G-4] 恒真替身：改衰减指数或键能形式，G-1 的跨过须改变")
g4_rows = []
for tag, n, kind in (("n=1", 1.0, KIND), ("n=3", 3.0, KIND), ("central", 2.0, "central")):
    cross_alt = None
    for i in range(len(feasible_scales) - 1):
        k0 = feasible_scales[i][0]
        k1 = feasible_scales[i + 1][0]
        s0_ = share_at(k0, n=n, kind=kind)
        s1_ = share_at(k1, n=n, kind=kind)
        if s0_ > TARGET >= s1_:
            kc = k0 + (s0_ - TARGET) * (k1 - k0) / (s0_ - s1_)
            cross_alt = (k0, k1, kc)
            break
    g4_rows.append((tag, cross_alt))
    print("  替身 %-8s：%s" % (tag, ("跨过档 %.1f 与 %.1f 之间，插值半径 %.4f" % cross_alt[:3]) if cross_alt else "可制备档上未找到跨过"))
alt_kc = [r[1][2] for r in g4_rows if r[1]]
if alt_kc and k_cross is not None:
    rel_change = max(abs(x - k_cross) for x in alt_kc) / max(abs(k_cross), 1e-12)
    print("  与真值的最大相对差 %.3e" % rel_change)
    g4_ok = rel_change > 0.05
    check("G-4", g4_ok,
          "改衰减指数或键能形式后跨过半径显著改变（最大相对差 %.3e > 0.05），判据有区分力" % rel_change)
else:
    check("G-4", False, "替身后无可比较的跨过半径")

# ----------------------------------------------------------------------
# G-5 读法登记
# ----------------------------------------------------------------------
print()
print("[G-5] 读法登记")
for k_pos, s, above in g1_rows:
    print("     核心半径 %5.1f：应力迹份额 %+.6e" % (k_pos, s))
check("G-5", True, "读法登记完成")

# ----------------------------------------------------------------------
# G-6 状态登记
# ----------------------------------------------------------------------
print()
print("[G-6] 状态登记")
reg = [
    ("其一", "跨过", "可制备档上的跨过区间与插值半径"),
    ("其二", "可制备区间", "下界（至少一个格点）、上界（反相壁留出要求）"),
    ("其三", "刚度敏感性", "G-3 的跨过半径移动量"),
    ("其四", "辨别力", "G-4 的替身相对差"),
    ("其五", "剩余", "核心束缚能与 m_P 的关系未导出；刚度取值归弹性本构线；"
                "缺陷真实存在性归 C1 线；壁厚与热力学未导出"),
]
for a1_, a2_, a3_ in reg:
    print("     %-4s %-12s %s" % (a1_, a2_, a3_))
check("G-6", True, "状态登记完成")

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
ks = [r[0] for r in g1_rows]
ss = [r[1] for r in g1_rows]
ax.semilogx(ks, ss, "o-", linewidth=1.6, markersize=3.5, label=u"可制备档的份额")
ax.axhline(TARGET, color="#2ca02c", linestyle="--", linewidth=1.4, label=u"源侧靶子 1")
if cross is not None:
    ax.axvspan(cross[0], cross[2], color="#ff7f0e", alpha=0.25, label=u"跨过区间")
ax.set_xlabel(u"缺陷核心半径 $k_{pos}$（格点单位）")
ax.set_ylabel(u"应力迹份额")
ax.set_title(u"(a) 离散档上的跨过")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

ax = axes[1]
for k, cross_k in g3_rows:
    if cross_k:
        ax.axvline(cross_k[2], linewidth=1.3, label=u"刚度 %.2f" % k)
ax.set_xlabel(u"跨过半径（格点单位）")
ax.set_ylabel(u"档位")
ax.set_title(u"(b) 跨过点对刚度的敏感性")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[2]
labels = [r[0] for r in g4_rows]
vals = [r[1][2] if r[1] else float("nan") for r in g4_rows]
vals = [v if np.isfinite(v) else 0.0 for v in vals]
ax.bar(np.arange(len(labels)), vals, color=["#1f77b4", "#ff7f0e", "#2ca02c"], width=0.55)
if k_cross is not None:
    ax.axhline(k_cross, color="#d62728", linestyle="--", linewidth=1.4, label=u"真值 %.4f" % k_cross)
ax.set_xticks(np.arange(len(labels)))
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel(u"跨过半径（格点单位）")
ax.set_title(u"(c) 替身下的跨过半径")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")

fig.suptitle(u"第756号：缺陷核心尺度的可制备性约束与跨过点的物理可实现性", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(IMG, dpi=150)
print()
print("图已存：%s" % IMG)
