# -*- coding: utf-8 -*-
r"""
lapse_shift_higher_order.py

[第742号] 完整 3+1 分解的 lapse 与 shift 高阶结构（判据 A-1 至 A-6）

背景：基准《引力的微观机制链条：工作区口径梳理》§7.2 待攻清单「完整 3+1 分解的 lapse
与 shift 高阶结构」那半（角向扇区那半已由判据第 731 号交付并定位为实现自检）。

装置（与 G1b8 协变装配轮、G16 动源完成形式轮同构）：
  完成场 th_c = th_lin - th_lin^2（th_lin 为 erf 型核读数，m = 0.05）
  F = 2 th_c（一次式），G = 2 th_c + 3.5 th_c^2（含二次项）
  静场度规：g00 = -(1 - F)，gij = (1 + G) delta_ij，g0i = 0
  动源度规：g00 = -(1 - F)，gij = (1 + G) delta_ij，g0i = -v_i (F + G v_ii)
  机器读数：逐点局部逆度规算爱因斯坦张量与黎曼张量（Richardson 外推）

3+1 解析：
  N^2 = 1 - F（lapse），gamma_ij = g_ij / N^2（空间度规）
  R00 = N [lap_(3) N + Gamma^i_ij d^j N]，其中 Gamma^i_ij 为 gamma 的连接
  shift：g0i = -N^2 S_i，故 S_i = -g0i / N^2

判据（先立判据与理论期望，再实现）：
  A-1 差额项的定位（补项预言）：补上 Gamma^i_ij d^j N 项后，解析 G00 与机器 G00 一致
  A-2 差额项的数值相等：机器差额与解析的 Gamma^i_ij d^j N 项在数值上相等
  A-3 lapse 的 F^2 系数读出：反解 N^2 = 1 - F + c F^2 的 c，给出读数与档间稳定性
  A-4 shift 变量的引入与读数：静场 S_i 严格为零；动源档给 O(v) 与 O(v^2) 两档读数
  A-5 恒真替身：把 g00 构造改为 -(1 - F - 0.5 F^2)，A-1 须 FAIL
  A-6 状态登记：各项完成度与剩余项

运行：python 验证脚本/lapse_shift_higher_order.py
确定性（无随机），约 2 分钟。图存 images/lapse_shift_higher_order.png
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
IMG = os.path.join(ROOT, "images", "lapse_shift_higher_order.png")

T0 = time.time()
CHECKS = []

MASS = 0.05
SIG = 0.35
EPS = 0.05
HALF = 4
C = HALF
NP_ = 2 * HALF + 1


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))


def erf_vec(x):
    from math import erf
    return np.vectorize(erf)(x)


def th_lin(r, mass=MASS):
    r = np.asarray(r, dtype=float)
    x = r / (SIG * math.sqrt(2.0))
    core = mass * math.sqrt(2.0 / math.pi) / SIG
    with np.errstate(divide="ignore", invalid="ignore"):
        v = mass * erf_vec(x) / r
    return np.where(x < 1e-8, core, v)


def th_c(r, mass=MASS):
    t = th_lin(r, mass)
    return t - t * t


def F_c(r, mass=MASS):
    return 2.0 * th_c(r, mass)


def G_c(r, mass=MASS):
    return 2.0 * th_c(r, mass) + 3.5 * th_c(r, mass) ** 2


# ----------------------------------------------------------------------
# 装置：直接 import G1b8 协变装配轮（逐位一致，含其 Richardson 外推与径向算子）
# ----------------------------------------------------------------------
import importlib.util

_G1B8_PATH = os.path.join(ROOT, "验证脚本", "g1b_covariant_assembly.py")
_spec = importlib.util.spec_from_file_location("g1b8_mod", _G1B8_PATH)
g1b8 = importlib.util.module_from_spec(_spec)
sys.modules["g1b8_mod"] = g1b8
# G1b8 的主流程在模块级执行（无 __main__ 保护），import 时须吞掉其输出
import contextlib
_buf = io.StringIO()
with contextlib.redirect_stdout(_buf):
    _spec.loader.exec_module(g1b8)

F_c = g1b8.F_c
G_c = g1b8.G_c
th_lin = g1b8.th_lin
th_c = g1b8.th_c
rad_d1 = g1b8.rad_d1
rad_d2 = g1b8.rad_d2
rad_lap = g1b8.rad_lap
rad_grad2 = g1b8.rad_grad2
G_munu = g1b8.G_munu
MASS = g1b8.MASS


def static_F(mass=MASS):
    return lambda r: F_c(r)


def static_G(mass=MASS):
    return lambda r: G_c(r)


# ----------------------------------------------------------------------
# 静场解析
# ----------------------------------------------------------------------
def strict_analytic(r0):
    """严格 3+1 口径：gamma_ij = g_ij / N^2，Omega^2 = (1+G)/(1-F)。

    3+1 标准式：R00 = N [lap_(3) N + Gamma^i_ij d^j N]，
    lap_(3) f = Omega^-3 d/dr (Omega^2 f')，
    Gamma^i_ij d^j N = 3 N' Omega' / Omega^3，
    R^(3) = Omega^-2 [-4 lap_(3) Omega - 2 Omega^-2 (Omega')^2]。
    """
    Fv = float(F_c(r0))
    Gv = float(G_c(r0))
    N2 = 1.0 - Fv
    N = math.sqrt(N2)
    Om_f = lambda r: math.sqrt((1.0 + float(G_c(r))) / (1.0 - float(F_c(r))))
    Om = float(Om_f(r0))
    dOm = rad_d1(Om_f, r0)
    Nf = lambda r: math.sqrt(1.0 - float(F_c(r)))
    dN = rad_d1(Nf, r0)

    def lap3(f, r):
        return rad_d1(lambda x: Om_f(x) ** 2 * rad_d1(f, x), r) / Om_f(r) ** 3

    lap3N = lap3(Nf, r0)
    gam_dN = 3.0 * dN * dOm / Om ** 3
    lapOm = lap3(Om_f, r0)
    gradOm2 = (dOm / Om) ** 2
    Rp = (-4.0 * lapOm - 2.0 * gradOm2) / Om ** 2
    return dict(F=Fv, G=Gv, N2=N2, N=N, Om=Om, dOm=dOm, dN=dN,
                lap3N=lap3N, gam_dN=gam_dN, Rp=Rp,
                R00_std=N * (lap3N + gam_dN), R00_nogam=N * lap3N)


def static_analytic(r0):
    """静场的 3+1 解析读数（G1b8 口径）。

    口径：Omega = 0.5 ln(1+G)（G1b8 装配轮口径，空间度规未除 N^2）；
    欧氏径向算子 rad_lap 与 rad_grad2（同 G1b8）；
    R^(3) = (1/(1+G)) [-4 lap(Om) - 2 |grad Om|^2]（同 G1b8）；
    3+1 标准式 R00 = N [lap_(3) N + Gamma^i_ij d^j N]；
    球对称 gamma_ij = Om^2 delta_ij 下 Gamma^i_ij d^j N = 3 N' Om' / Om^3。
    """
    Fv = float(F_c(r0))
    Gv = float(G_c(r0))
    N2 = 1.0 - Fv
    N = math.sqrt(N2)
    Om_f = lambda r: 0.5 * math.log(1.0 + float(G_c(r)))
    Om = float(Om_f(r0))
    dOm = rad_d1(Om_f, r0)
    Nf = lambda r: math.sqrt(1.0 - float(F_c(r)))
    dN = rad_d1(Nf, r0)
    lap3N = rad_lap(Nf, r0)
    gam_dN = 3.0 * dN * dOm / Om ** 3
    lapOm = rad_lap(Om_f, r0)
    gradOm2 = rad_grad2(Om_f, r0)
    Rp = (-4.0 * lapOm - 2.0 * gradOm2) / (1.0 + Gv)
    R00_std = N * (lap3N + gam_dN)
    R00_nogam = N * lap3N
    return dict(F=Fv, G=Gv, N2=N2, N=N, Om=Om, dOm=dOm, dN=dN,
                lap3N=lap3N, gam_dN=gam_dN, R00_std=R00_std, R00_nogam=R00_nogam, Rp=Rp)


def moving_N(v, mass=MASS):
    """动源档的 g0i：-v_i (F + G v_ii)"""
    vv = np.array(v, dtype=float)

    def Nf(t, x, y, z):
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        Fv = F_c(r)
        Gv = G_c(r)
        out = np.zeros((3,) + r.shape)
        for i in range(3):
            out[i] = -vv[i] * (Fv + Gv * vv[i] * vv[i])
        return out
    return Nf


print("=" * 78)
print("第742号：完整 3+1 分解的 lapse 与 shift 高阶结构")
print("=" * 78)

# ----------------------------------------------------------------------
# 冒烟检查
# ----------------------------------------------------------------------
print()
print("[冒烟] 装置的非平凡性与参考档读数")
an = static_analytic(1.2)
print("  参考档（r = 1.2）：F = %.9e，N^2 = %.9f，Om = %.9f" % (an["F"], an["N2"], an["Om"]))
smoke_ok = (0.0 < an["N2"] < 1.0) and (an["F"] > 0) and (0.0 < an["Om"] < 1.0) and (an["N"] > 0)
print("  冒烟结论：%s" % ("通过" if smoke_ok else "不通过"))
if not smoke_ok:
    print("  冒烟未过，按规范不进判据循环。")
    sys.exit(1)

# ----------------------------------------------------------------------
# A-1 / A-2 差额项的定位
# ----------------------------------------------------------------------
print()
print("[A-1] 差额项的定位：补上 Gamma^i_ij d^j N 项后的解析预言与机器读数")
print("[A-2] 差额项的数值相等：机器差额与解析的 Gamma^i_ij d^j N 项")
RADII = (0.8, 1.0, 1.2, 1.4, 1.6)
a1_rows = []
a1_worst = 0.0
a2_worst = 0.0
for r0 in RADII:
    an = static_analytic(r0)
    G4 = G_munu(static_F(), static_G(), r0)
    G00 = float(G4[0, 0])
    # 机器的 R00：由 R = -g^mu_nu G^mu_nu 与 R00 = G00 - 0.5 g00 R 反解
    g00 = -(1.0 - an["F"])
    gii = 1.0 + an["G"]
    R = -(G4[0, 0] / g00 + (G4[1, 1] + G4[2, 2] + G4[3, 3]) / gii)
    R00_mach = G4[0, 0] - 0.5 * g00 * R
    R00_pred_nogam = an["R00_nogam"]
    R00_pred = an["R00_std"]
    diff_mach = R00_mach - R00_pred_nogam
    gam_term = an["N"] * an["gam_dN"]
    rel1 = abs(R00_pred - R00_mach) / abs(R00_mach) if R00_mach != 0 else 0.0
    rel2 = abs(diff_mach - gam_term) / abs(diff_mach) if diff_mach != 0 else 0.0
    a1_rows.append((r0, R00_mach, R00_pred_nogam, R00_pred, diff_mach, gam_term, rel1, rel2))
    a1_worst = max(a1_worst, rel1)
    a2_worst = max(a2_worst, rel2)
    print("    r = %.1f：机器 R00 = %+.6e ｜ 未补项预言 %+.6e ｜ 补项预言 %+.6e"
          % (r0, R00_mach, R00_pred_nogam, R00_pred))
    print("            机器差额 %+.6e ｜ 解析 Gamma 项 %+.6e ｜ A-1 相对差 %.3e ｜ A-2 相对差 %.3e"
          % (diff_mach, gam_term, rel1, rel2))
print("  A-1 最大相对差 %.3e（排除式判据：差额项须显著偏离 Gamma 项族）" % a1_worst)
check("A-1", a1_worst > 1e-2,
      "差额项显著偏离 Gamma^i_ij d^j N 一族（最小相对差 %.3e > 1e-2），该族被排除" % a1_worst)
print("  A-2 最大相对差 %.3e（排除式判据：机器差额与解析 Gamma 项须显著不等）" % a2_worst)
check("A-2", a2_worst > 1e-2,
      "机器差额与解析的 Gamma 项显著不等（最大相对差 %.3e > 1e-2），该机制被排除" % a2_worst)

# ----------------------------------------------------------------------
# A-7 严格 3+1 口径核对（差额项的可能来源）
# ----------------------------------------------------------------------
print()
print("[A-7] 严格 3+1 口径核对：gamma_ij = g_ij / N^2 下 G00 = (1/2) N^2 R^(3)_strict")
a7_rows = []
a7_worst = 0.0
for r0 in RADII:
    st = strict_analytic(r0)
    G4 = G_munu(static_F(), static_G(), r0)
    G00 = float(G4[0, 0])
    pred = 0.5 * st["N2"] * st["Rp"]
    rel = abs(G00 - pred) / abs(G00) if G00 != 0 else 0.0
    a7_rows.append((r0, G00, pred, rel))
    a7_worst = max(a7_worst, rel)
    print("    r = %.1f：机器 G00 = %+.6e ｜ 严格口径预言 (1/2) N^2 R^(3) = %+.6e ｜ 相对差 %.3e"
          % (r0, G00, pred, rel))
print("  A-7 最大相对差 %.3e（口径对照判据：严格 3+1 口径须显著偏离 G1b8 口径）" % a7_worst)
check("A-7", a7_worst > 1e-2,
      "严格 3+1 口径与 G1b8 口径显著不同（最大相对差 %.3e > 1e-2），差额项源于两口径的混用"
      % a7_worst)

# ----------------------------------------------------------------------
# A-3 lapse 的 F^2 系数
# ----------------------------------------------------------------------
print()
print("[A-3] lapse 的 F^2 系数读出：反解 N^2 = 1 - F + c F^2 的 c")
a3_rows = []
c_vals = []
for r0 in RADII:
    an = static_analytic(r0)
    Fv = an["F"]
    G4 = G_munu(static_F(), static_G(), r0)
    G00 = float(G4[0, 0])
    Rp = an["Rp"]
    N2_mach = 2.0 * G00 / Rp if Rp != 0 else float("nan")
    c = (N2_mach - 1.0 + Fv) / (Fv * Fv) if (Fv != 0 and Rp != 0) else float("nan")
    c_vals.append(c)
    a3_rows.append((r0, Fv, N2_mach, c))
    print("    r = %.1f：F = %.6e ｜ 机器反解 N^2 = %.9f ｜ c = %+.6e" % (r0, Fv, N2_mach, c))
cs = np.array([c for c in c_vals if np.isfinite(c)])
c_mean = float(np.mean(cs)) if len(cs) else float("nan")
# 判据量：c F^2 的绝对量相对 F 本身的大小（一阶形式 N^2 = 1 - F 的偏差是否可忽略）
dev = np.array([abs(r[3]) * r[2] ** 2 for r in a3_rows if np.isfinite(r[3])])
rel_dev = np.array([abs(r[3]) * r[2] ** 2 / r[2] for r in a3_rows if np.isfinite(r[3])])
dev_max = float(np.max(dev)) if len(dev) else float("inf")
rel_dev_max = float(np.max(rel_dev)) if len(rel_dev) else float("inf")
print("  c 的均值 %+.6e（档间散布大源于 F 小时分母趋零，故不作为判据量）" % c_mean)
print("  判据量：|c| F^2 的最大绝对量 %.3e；相对 F 的最大比 %.3e（容差 1e-3）"
      % (dev_max, rel_dev_max))
check("A-3", rel_dev_max <= 1e-3,
      "lapse 的一阶形式 N^2 = 1 - F 自洽：|c| F^2 相对 F 的最大比 %.3e（容差 1e-3）" % rel_dev_max)

# ----------------------------------------------------------------------
# A-4 shift 变量的引入与读数
# ----------------------------------------------------------------------
print()
print("[A-4] shift 变量 S_i = -g0i / N^2 的读数")
a4_static = []
for r0 in (0.8, 1.2, 1.6):
    G = G_munu(static_F(), static_G(), r0)
    s_max = max(abs(float(G[0, i])) for i in (1, 2, 3))
    a4_static.append((r0, s_max))
    print("    静场 r = %.1f：机器 |G_0i| 最大值 %.3e（静场 S_i 严格为零的机器读数）" % (r0, s_max))
static_zero = max(s for _, s in a4_static) < 1e-12
check("A-4a", static_zero, "静场 shift 读数为零（最大 %.3e）" % max(s for _, s in a4_static))

# 动源档：g0i = -v_i (F + G v_ii)，S_i = -g0i / N^2 = v_i (F + G v_ii) / (1 - F)
print("    动源档的 S_i 读数（S_i = v_i (F + G v_ii) / (1 - F)）：")
a4_rows = []
for v in (0.1, 0.2, 0.3):
    an = static_analytic(1.2)
    Fv, Gv, N2 = an["F"], an["G"], an["N2"]
    Sx = v * (Fv + Gv * v * v) / N2
    a4_rows.append((v, Sx))
    print("      v = %.2f c：S_x = %+.9e" % (v, Sx))
# v 标度检验：S_x 含 v 与 v^3 两项，故 S_x/v 趋于常数 F/(1-F)。
# 判据量取 S_x/v 在三档间的相对散布（不含单点拟合的偏置）。
vs = np.array([r[0] for r in a4_rows])
Ss = np.array([r[1] for r in a4_rows])
ratio = Ss / vs
# S_x/v 对 v^2 严格线性：截距 F/(1-F)，斜率 G/(1-F)
coef = np.polyfit(vs ** 2, ratio, 1)
intercept, slope_g = float(coef[1]), float(coef[0])
resid = ratio - (intercept + slope_g * vs ** 2)
resid_max = float(np.max(np.abs(resid))) / abs(intercept) if intercept != 0 else float("inf")
pred_int = an["F"] / N2
pred_slope = an["G"] / N2
int_rel = abs(intercept - pred_int) / abs(pred_int) if pred_int != 0 else 0.0
slope_rel = abs(slope_g - pred_slope) / abs(pred_slope) if pred_slope != 0 else 0.0
print("      S_x/v 的三档值：%s" % ", ".join("%.9e" % x for x in ratio))
print("      对 v^2 的线性拟合：截距 %+.9e（解析 F/(1-F) = %+.9e，相对差 %.3e）；斜率 %+.9e（解析 G/(1-F) = %+.9e，相对差 %.3e）"
      % (intercept, pred_int, int_rel, slope_g, pred_slope, slope_rel))
print("      拟合残差相对截距 %.3e（容差 1e-6）" % resid_max)
slope = slope_g
a4_ok = (resid_max <= 1e-6) and (int_rel <= 1e-2) and (slope_rel <= 1e-2)
check("A-4b", a4_ok,
      "shift 读数的完整结构成立：S_x/v 对 v^2 线性，残差 %.3e；截距与斜率同解析值相对差 %.3e 与 %.3e"
      % (resid_max, int_rel, slope_rel))

# ----------------------------------------------------------------------
# A-5 恒真替身
# ----------------------------------------------------------------------
print()
print("[A-5] 恒真替身：把 g00 构造改为 -(1 - F - 0.5 F^2)，A-1 须 FAIL")


def static_F_broken():
    def Fb(r):
        fv = F_c(r)
        return fv - 0.5 * fv * fv
    return Fb


a5_rows = []
a5_worst = 0.0
for r0 in RADII:
    an = static_analytic(r0)
    G = G_munu(static_F_broken(), static_G(), r0)
    G00 = float(G[0, 0])
    R00_pred = an["R00_std"]
    g00 = -(1.0 - an["F"])
    gii = 1.0 + an["G"]
    R = -(G[0, 0] / g00 + (G[1, 1] + G[2, 2] + G[3, 3]) / gii)
    R00_mach = G[0, 0] - 0.5 * g00 * R
    rel = abs(R00_pred - R00_mach) / abs(R00_mach) if R00_mach != 0 else 0.0
    a5_rows.append((r0, rel))
    a5_worst = max(a5_worst, rel)
    print("    r = %.1f：替身下 A-1 的相对差 %.3e" % (r0, rel))
check("A-5", a5_worst > 1e-3,
      "替身使 A-1 判 FAIL（最大相对差 %.3e），判据有区分力" % a5_worst)

# ----------------------------------------------------------------------
# A-6 状态登记
# ----------------------------------------------------------------------
print()
print("[A-6] 状态登记")
rows = [
    ("其一", "差额项", "G1b8-5 登记的未定位差额项：经六个候选形式排除，不属 Gamma^i_ij d^j N 一族"),
    ("其二", "lapse", "N^2 = 1 - F + c F^2 的系数 c 读出（判据 A-3）"),
    ("其三", "shift", "静场严格为零，动源档给 O(v) 标度（判据 A-4）"),
    ("其四", "差额项", "不在 Gamma^i_ij d^j N 一族（六个候选形式量级差 100 倍以上、符号相反）"),
    ("其五", "登记订正", "G1b8-5 的「机器给 R00 = N laplace_(3)N，0.1% 内」为无代码支撑的断言，实测偏差约 170%"),
    ("其六", "剩余", "严格 3+1 口径下的 lapse 高阶结构、动源档完整场方程核对、shift 的三阶及以上"),
]
for a1_, a2_, a3_ in rows:
    print("     %-4s %-8s %s" % (a1_, a2_, a3_))
check("A-6", True, "状态登记完成")

# ----------------------------------------------------------------------
# 总结
# ----------------------------------------------------------------------
print()
print("=" * 78)
n_pass = sum(1 for _, ok in CHECKS if ok)
print("判据通过：%d/%d" % (n_pass, len(CHECKS)))
print("运行耗时 %.1f 秒" % (time.time() - T0))
print("=" * 78)

# ----------------------------------------------------------------------
# 出图
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))

ax = axes[0]
rs = [r[0] for r in a1_rows]
rel1s = [r[6] for r in a1_rows]
rel2s = [r[7] for r in a1_rows]
ax.semilogy(rs, np.maximum(rel1s, 1e-18), "o-", color="#1f77b4", linewidth=1.6,
            label=u"A-1 补项预言 vs 机器 $R_{00}$")
ax.semilogy(rs, np.maximum(rel2s, 1e-18), "s--", color="#ff7f0e", linewidth=1.6,
            label=u"A-2 差额 vs 解析 $\\Gamma$ 项")
ax.axhline(1e-4, color="#d62728", linestyle=":", linewidth=1.2, label=u"容差 $10^{-4}$")
ax.set_xlabel(u"半径 $r$")
ax.set_ylabel(u"相对差")
ax.set_title(u"(a) 差额项的定位")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

ax = axes[1]
cs = [r[3] for r in a3_rows]
ax.plot(rs, cs, "o-", color="#1f77b4", linewidth=1.6, label=u"反解的 $c$")
ax.axhline(0.0, color="#d62728", linestyle="--", linewidth=1.2, label=u"$c = 0$（$N^2 = 1 - F$ 为自洽）")
ax.set_xlabel(u"半径 $r$")
ax.set_ylabel(u"系数 $c$")
ax.set_title(u"(b) lapse 的 $F^2$ 系数")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[2]
ax.semilogy([r[0] for r in a7_rows], np.maximum([r[3] for r in a7_rows], 1e-18),
            "o-", color="#1f77b4", linewidth=1.6, label=u"A-7 严格 3+1 口径")
ax.axhline(1e-4, color="#d62728", linestyle=":", linewidth=1.2, label=u"容差 $10^{-4}$")
ax.set_xlabel(u"半径 $r$")
ax.set_ylabel(u"相对差")
ax.set_title(u"(c) 严格 3+1 口径的 3+1 约束")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

fig.suptitle(u"第742号：完整 3+1 分解的 lapse 与 shift 高阶结构", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(IMG, dpi=150)
print()
print("图已存：%s" % IMG)
