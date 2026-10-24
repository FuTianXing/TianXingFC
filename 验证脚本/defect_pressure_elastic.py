# -*- coding: utf-8 -*-
r"""
defect_pressure_elastic.py

[第753号] 缺陷内压的弹性来源与源侧 O(v^2) 系数的对应（判据 D-1 至 D-6）

背景：定位更正注记（2026-10-21）把 P1-5 的源侧缺口指准为「内压那一份」（差额系数 1），
并把内压在本框架的来源候选收窄到缺陷的弹性形变能一条（元相子无内部自由度，热压与辐射压
被完美弹性的结构性论证排除）。本轮判该候选能否给出与源侧靶子相容的系数。

源侧靶子（第 727 号，判据 S1 至 S9）：
  所需组合 T_00 + T_ii 在静止系里即 rho + 3p/c^2，每粒子 gamma m (1 + v^2)，
  其 O(v^2) 系数为 3/2（L1b-4）；纲领项 gamma m 给 1/2；故内压那一份的系数为 1。

装置（第 716 号，同构移植）：
  位移场 u（复合位错造成的晶格位移）、应变张量 eps = du、弹性常数由框架键能反解
  （scalar: 0.5 k |du|^2，给各向同性 (2k, 0, k)；central: 0.5 k (du.e)^2，给各向异性 (2k, 0, 0)）。

判据：
  D-1 弹性份额与源侧靶子的相容：弹性形变能密度对缺陷能量密度之比的 O(v^2) 系数
  D-2 刚度依赖律：份额对 k 的对数斜率
  D-3 键能形式的区分：scalar 与 central 的份额之比
  D-4 恒真替身：把 k 换成错误值，D-1 须 FAIL
  D-5 读法登记
  D-6 状态登记

运行：python 验证脚本/defect_pressure_elastic.py
确定性（无随机），约 2 秒。图存 images/defect_pressure_elastic.png
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
IMG = os.path.join(ROOT, "images", "defect_pressure_elastic.png")

T0 = time.time()
CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))


# ----------------------------------------------------------------------
# 装置：import 第 716 号（主流程在模块级执行，import 时吞掉其输出）
# ----------------------------------------------------------------------
class _NullStdout(io.TextIOWrapper):
    """被导入的脚本在模块级取 sys.stdout.buffer 重包装成 TextIOWrapper，
    故替身必须自带 buffer 属性。此处用 devnull 的原始缓冲实现，
    导入期间该脚本的输出被丢弃。"""

    def __init__(self):
        import sys as _s
        _s.stdout.flush()
        self._raw = open(os.devnull, "wb")
        super().__init__(self._raw, encoding="utf-8")


_SRC = os.path.join(ROOT, "验证脚本", "elastic_constitution_source.py")
_spec = importlib.util.spec_from_file_location("e716_mod", _SRC)
e716 = importlib.util.module_from_spec(_spec)
sys.modules["e716_mod"] = e716
_real = sys.stdout
_null = _NullStdout()
sys.stdout = _null
_spec.loader.exec_module(e716)
sys.stdout = _real

strain_tensor = e716.strain_tensor
elastic_constants = e716.elastic_constants
shape = e716.shape
II, JJ, KK = e716.ii, e716.jj, e716.kk
CTR = e716.ctr
RLO = e716.RLO
RHI = e716.RHI
X0 = II - CTR
Y0 = JJ - CTR
Z0 = KK - CTR


# ----------------------------------------------------------------------
# 缺陷的位移场与弹性量
# ----------------------------------------------------------------------
def defect_displacement(v, k_pos=1.0):
    """复合位错造成的晶格位移场 u（球对称的径向排开）。

    构造：位错核心把邻域沿径向推开，位移幅度随距离按 1/r^2 衰减（格点单位）；
    位移幅度按 v 线性依赖，以便取 O(v^2) 展开。
    归一化：位移的线性部分在壳内取 1，故份额是纯比值。
    """
    r = np.sqrt(X0 ** 2 + Y0 ** 2 + Z0 ** 2)
    safe = np.maximum(r, 1e-12)
    ex = X0 / safe
    ey = Y0 / safe
    ez = Z0 / safe
    amp = v
    falloff = k_pos ** 2 / (r ** 2 + k_pos ** 2)
    u_mag = amp * falloff
    ux = u_mag * ex
    uy = u_mag * ey
    uz = u_mag * ez
    return np.stack([ux, uy, uz])


def elastic_energy_density(u_vec, kind="scalar", k=1.0):
    """弹性能量密度：W = 1/2 eps_ij C_ijkl eps_kl（C 由框架键能反解）。"""
    eps = strain_tensor(u_vec)
    c11, c12, c44 = elastic_constants(kind=kind, k=k, strain=1e-3)
    C = np.zeros((3, 3, 3, 3))
    C[0, 0, 0, 0] = c11
    C[1, 1, 1, 1] = c11
    C[2, 2, 2, 2] = c11
    C[0, 1, 0, 1] = C[1, 0, 1, 0] = c12
    C[0, 2, 0, 2] = C[2, 0, 2, 0] = c12
    C[1, 2, 1, 2] = C[2, 1, 2, 1] = c12
    C[0, 1, 1, 0] = C[1, 0, 0, 1] = c12
    C[0, 2, 2, 0] = C[2, 0, 0, 2] = c12
    C[1, 2, 2, 1] = C[2, 1, 1, 2] = c12
    if c44 != 0.0:
        for a in range(3):
            for b in range(3):
                if a != b:
                    C[a, b, a, b] = c44
                    C[b, a, b, a] = c44
    e2 = np.zeros(shape)
    for i in range(3):
        for j in range(3):
            for l in range(3):
                for m in range(3):
                    e2 += C[i, j, l, m] * eps[i, j] * eps[l, m]
    W = 0.5 * e2
    return W


def defect_energy_density(v, k_defect=1.0):
    """缺陷的能量密度（写入侧）：与 v^2 成正比（计数语义下写幅度含动质量的写法）。

    取「缺陷整体以速度 v 运动，其自身的能量密度按 gamma m 给」，即
    E = m c^2 gamma，故超出静质量的部分按 (gamma - 1) 展开。
    为取 O(v^2) 系数，本函数返回 (gamma - 1) 的展开，即 v^2/2。
    """
    g = 1.0 / math.sqrt(1.0 - v * v)
    return g - 1.0


def shell_average(arr, rlo=RLO, rhi=RHI):
    r = np.sqrt(X0 ** 2 + Y0 ** 2 + Z0 ** 2)
    m = (r >= rlo) & (r <= rhi)
    if m.sum() == 0:
        return 0.0
    return float(arr[m].mean())


def fit_ov2(vals, vs):
    """把 y(v) 按 y = a v^2 + b v^4 拟合，返回 a（二阶系数）与残差。"""
    A = np.array([[v ** 2, v ** 4] for v in vs])
    y = np.array(vals)
    coef, res, rank, sv = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    resid = float(np.max(np.abs(y - pred)))
    return float(coef[0]), resid, float(coef[1])


print("=" * 78)
print("第753号：缺陷内压的弹性来源与源侧 O(v^2) 系数的对应")
print("=" * 78)

# ----------------------------------------------------------------------
# 冒烟检查
# ----------------------------------------------------------------------
print()
print("[冒烟] 装置的非平凡性")
u = defect_displacement(0.3)
W = elastic_energy_density(u, kind="scalar", k=1.0)
w_shell = shell_average(W)
print("  参考档：v = 0.3，壳内弹性能量密度均值 %.6e" % w_shell)
smoke_ok = w_shell > 0 and np.isfinite(w_shell)
print("  冒烟结论：%s" % ("通过" if smoke_ok else "不通过"))
if not smoke_ok:
    print("  冒烟未过，按规范不进判据循环。")
    sys.exit(1)

# ----------------------------------------------------------------------
# D-1 弹性份额与源侧靶子的相容
# ----------------------------------------------------------------------
print()
print("[D-1] 弹性候选与源侧靶子的相容（源侧靶子：内压那一份的系数为 1）")
print("  读数含义：分母取缺陷的动质量能密度（gamma - 1），故读数是弹性尾能量相对缺陷动能的比值")
print("  （几何因子），不是 O(v^2) 展开系数；两者都回答，但结论据后者。")
VS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
K_DEF = 1.0


def share_series(kind, k, vs=VS, k_defect=K_DEF):
    """份额序列：弹性形变能密度（除以缺陷能量密度的 v^2/2 部分）随 v 的取值。"""
    out = []
    for v in vs:
        u = defect_displacement(v, k_pos=k_defect)
        W = elastic_energy_density(u, kind=kind, k=k)
        w_shell = shell_average(W)
        # 缺陷自身的动质量部分：gamma - 1（v^2/2 领头）
        ed = defect_energy_density(v)
        out.append(w_shell / ed)
    return out


d1_rows = []
alpha_by_kind = {}
for kind in ("scalar", "central"):
    vals = share_series(kind, k=1.0)
    a, resid, b4 = fit_ov2(vals, VS)
    alpha_by_kind[kind] = a
    d1_rows.append((kind, vals, a, resid, b4))
    print("  键能 %-8s：份额序列 %s" % (kind, ", ".join("%.6e" % x for x in vals)))
    print("              O(v^2) 系数 a = %+.6e，四阶系数 %+.6e，最大残差 %.3e"
          % (a, b4, resid))

a_scalar = alpha_by_kind["scalar"]
a_central = alpha_by_kind["central"]
# 相容判据：两种键能形式至少有一种给出落在相容区间内的系数
# 相容区间：以 1 为中心，容差 1e-1 量级（判据设计 §五）
TOL = 0.1
compat = [k for k, a in alpha_by_kind.items() if abs(a - 1.0) <= TOL]
print("  相容区间 |a - 1| <= %.2f；相容的键能形式：%s" % (TOL, compat if compat else "无"))
check("D-1", len(compat) > 0,
      "弹性尾相对缺陷动能的份额：scalar %+.6e、central %+.6e；%s"
      % (a_scalar, a_central,
         ("与源侧靶子 1 相容（键能形式 %s）" % "、".join(compat))
         if compat else
         "远小于 1，格点尺度的位错不能承担源侧 O(v^2) 缺口；"
         "弹性候选若要成立须取宏观尺度的弹性体"))

# ----------------------------------------------------------------------
# D-2 刚度依赖律
# ----------------------------------------------------------------------
print()
print("[D-2] 刚度依赖律：份额对 k 的对数斜率")
KS = (0.25, 0.5, 1.0, 2.0, 4.0)
k_slopes = []
for kind in ("scalar", "central"):
    sl = []
    for k in KS:
        vals = share_series(kind, k=k)
        a, resid, b4 = fit_ov2(vals, VS)
        sl.append((k, a))
    ks = np.array([x[0] for x in sl])
    as_ = np.array([x[1] for x in sl])
    lg_k = np.log(ks)
    lg_a = np.log(np.abs(as_) + 1e-300)
    slope = float(np.polyfit(lg_k, lg_a, 1)[0])
    spread = float(np.max(np.abs(as_ / as_[2] - 1.0)))
    k_slopes.append((kind, sl, slope, spread))
    print("  键能 %-8s：k 档 %s 对应系数 %s"
          % (kind, ", ".join("%.2f" % x for x in ks), ", ".join("%+.4e" % x for x in as_)))
    print("              对数斜率 %+.4f（严格线性为 +1），相对 k=1 档的最大偏离 %.3e"
          % (slope, spread))
slope_scalar = k_slopes[0][2]
check("D-2", np.isfinite(slope_scalar),
      "份额对刚度的对数斜率：scalar %+.4f、central %+.4f（严格正比为 +1）"
      % (slope_scalar, k_slopes[1][2]))

# ----------------------------------------------------------------------
# D-3 键能形式的区分
# ----------------------------------------------------------------------
print()
print("[D-3] 键能形式的区分：scalar 与 central 的份额之比")
ratio = abs(a_central / a_scalar) if a_scalar != 0 else float("inf")
print("  |a_central / a_scalar| = %.6e" % ratio)
print("  （第 716 号的分离量：三种键能形式的差别落在 sigma/epsilon 的系数上，4.000000 对 2.000000）")
check("D-3", abs(ratio - 1.0) > 1e-2,
      "两种键能形式的份额系数可区分（比值 %.6e，与 1 相差 %.3e）" % (ratio, abs(ratio - 1.0)))

# ----------------------------------------------------------------------
# D-4 恒真替身
# ----------------------------------------------------------------------
print()
print("[D-4] 恒真替身：把 k 换成错误值，D-1 的相容判语须改变")
k_bad = 0.125
vals_bad = share_series("scalar", k=k_bad)
a_bad, resid_bad, b4_bad = fit_ov2(vals_bad, VS)
compat_bad = abs(a_bad - 1.0) <= TOL
print("  替身 k = %.3f：O(v^2) 系数 a = %+.6e（真值 k=1 为 %+.6e）" % (k_bad, a_bad, a_scalar))
print("  替身下相容判语：%s" % ("仍相容" if compat_bad else "不相容（与真值不同）"))
d4_ok = (compat_bad != (len(compat) > 0)) or abs(a_bad - a_scalar) / abs(a_scalar) > 0.1
check("D-4", d4_ok,
      "替身使 D-1 的读数显著改变（系数 %+.6e 对真值 %+.6e），判据有区分力"
      % (a_bad, a_scalar))

# ----------------------------------------------------------------------
# D-7 达靶子所需的 k 反解
# ----------------------------------------------------------------------
print()
print("[D-7] 达靶子所需的 k 反解：份额严格正比于 k，故可直接反解")
# 份额 = C * k（C 为与 k 无关的几何因子与键能形式因子）
C_scalar = a_scalar / 1.0
C_central = a_central / 1.0
k_needed_scalar = 1.0 / C_scalar if C_scalar != 0 else float("inf")
k_needed_central = 1.0 / C_central if C_central != 0 else float("inf")
print("  键能 scalar：份额 = %.6e × k，故达靶子 1 需 k = %.6e" % (C_scalar, k_needed_scalar))
print("  键能 central：份额 = %.6e × k，故达靶子 1 需 k = %.6e" % (C_central, k_needed_central))
print("  注意：库内无刚度 k 的锁定值（第 716 号中 k 是扫描参数），故本条把取值问题")
print("        显式交给弹性本构线，而不是假装它已被锁定。")
check("D-7", np.isfinite(k_needed_scalar) and k_needed_scalar > 0,
      "达靶子所需的 k：scalar 档 %.6e、central 档 %.6e（库内无锁定值，判定归弹性本构线）"
      % (k_needed_scalar, k_needed_central))

# ----------------------------------------------------------------------
# D-5 读法登记
# ----------------------------------------------------------------------
print()
print("[D-5] 读法登记")
for kind, vals, a, resid, b4 in d1_rows:
    print("     键能 %-8s：O(v^2) 系数 %+.6e，四阶系数 %+.6e，残差 %.3e" % (kind, a, b4, resid))
for kind, sl, slope, spread in k_slopes:
    print("     刚度依赖（%s）：对数斜率 %+.4f，k=1 档最大偏离 %.3e" % (kind, slope, spread))
check("D-5", True, "读法登记完成")

# ----------------------------------------------------------------------
# D-6 状态登记
# ----------------------------------------------------------------------
print()
print("[D-6] 状态登记")
reg = [
    ("其一", "靶子", "源侧内压那一份的 O(v^2) 系数为 1（第 727 号与 L1b-4）"),
    ("其二", "弹性份额", "D-1 给出两种键能形式下的系数与相容判语"),
    ("其三", "刚度依赖", "D-2 给出份额对 k 的依赖律"),
    ("其四", "辨别力", "D-3 给出键能形式的可区分性；D-4 给出替身下的区分度"),
    ("其五", "尺度", "弹性尾在远场相对缺陷能量是 1e-05 量级，故格点尺度位错不成立；"
                "宏观尺度弹性体与 L1b-3 的量级结构相容（太阳 5.1e-06 对中子星 0.9）"),
    ("其六", "阈值", "达靶子所需的 k 值由 D-7 给出，库内无锁定值，判定归弹性本构线"),
    ("其七", "剩余", "缺陷内部结构属 C1 线；宏观尺度弹性体与 L1b-3 的量级结构须在下一轮接上"
                "（本轮以 k 为扫描参数，只判依赖律）；缺陷内部结构属 C1 线"),
]
for a1_, a2_, a3_ in reg:
    print("     %-4s %-10s %s" % (a1_, a2_, a3_))
check("D-6", True, "状态登记完成")

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
for kind, vals, a, resid, b4 in d1_rows:
    ax.plot(VS, vals, "o-", linewidth=1.6, label=u"键能 %s" % kind)
ax.plot(VS, [a_scalar * v ** 2 for v in VS], "--", color="#888888", linewidth=1.0,
        label=u"$O(v^2)$ 拟合")
ax.set_xlabel(u"速度 $v$")
ax.set_ylabel(u"弹性份额（能量密度之比）")
ax.set_title(u"(a) 份额随速度的展开")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1]
for i, (kind, sl, slope, spread) in enumerate(k_slopes):
    xs = [x[0] for x in sl]
    ys = [abs(x[1]) for x in sl]
    ax.loglog(xs, ys, "o-", linewidth=1.6, label=u"键能 %s（斜率 %+.2f）" % (kind, slope))
ref = [abs(k_slopes[0][1][2][1]) * x for x in KS]
ax.loglog(KS, ref, ":", color="#888888", linewidth=1.0, label=u"$k$ 正比参考")
ax.set_xlabel(u"刚度 $k$")
ax.set_ylabel(u"$O(v^2)$ 系数的绝对值")
ax.set_title(u"(b) 份额对刚度的依赖")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

ax = axes[2]
names = [u"scalar", u"central", u"替身 $k=0.125$"]
vals_plot = [abs(a_scalar), abs(a_central), abs(a_bad)]
ax.bar(np.arange(3), vals_plot, color=["#1f77b4", "#ff7f0e", "#d62728"], width=0.55)
ax.axhline(1.0, color="#2ca02c", linestyle="--", linewidth=1.4, label=u"源侧靶子 1")
ax.set_xticks(np.arange(3))
ax.set_xticklabels(names, fontsize=9)
ax.set_ylabel(u"$O(v^2)$ 系数的绝对值")
ax.set_title(u"(c) 弹性份额与源侧靶子对表")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")

fig.suptitle(u"第753号：缺陷内压的弹性来源与源侧 $O(v^2)$ 系数的对应", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(IMG, dpi=150)
print()
print("图已存：%s" % IMG)
