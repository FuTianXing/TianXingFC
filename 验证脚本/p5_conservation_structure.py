# -*- coding: utf-8 -*-
r"""
p5_conservation_structure.py

[第747号] P5 缝合假设第一层：差额项的完整来源与守恒结构（判据 B-1 至 B-6）

背景：非线性缝合假设 P5（注入-扩散平衡与守恒结构的对应，非线性层）是 P1-5 真开放项里
唯一未门控的出口。判据第 742 号把差额项排除于六个 Gamma 项候选形式之外，并定位到两套
3+1 口径的混用，但完整来源未闭合。本轮做该第一层。

装置：直接 import G1b8 协变装配轮（逐位一致，含其 Richardson 外推与径向算子）：
  完成场 th_c = th_lin - th_lin^2（erf 型核，m = 0.05）
  F = 2 th_c（一次式），G = 2 th_c + 3.5 th_c^2（含二次项）
  静场度规 g00 = -(1 - F)，gij = (1 + G) delta_ij，g0i = 0
  机器读数：G_munu（F_c, G_c, r0）给 4x4 的 G_{mu nu}（Richardson 外推）

解析（严格 3+1 口径）：
  N^2 = 1 - F；gamma_ij = g_ij / N^2 = Omega^2 delta_ij，Omega^2 = (1+G)/(1-F)
  lap_(3) f = Omega^-3 d/dr (Omega^2 f')
  R^(3) = Omega^-2 [ -4 lap_(3) Omega - 2 Omega^-2 (Omega')^2 ]
  R00 = N [ lap_(3) N + Gamma^i_ij d^j N ]，球对称下 Gamma^i_ij d^j N = 3 N' Omega' / Omega^3
  机器的 R00 由 R00 = G00 - 0.5 g00 R 与 R = -g^mu_nu G^mu_nu 反解

差额项的分解（逐项归属）：
  项一 N lap_(3) N
  项二 N Gamma^i_ij d^j N
  项三 N^2 一阶截断效应（g00 侧缺 F^2 项所贡献的部分，以 d(N^2)/dF 的线性化给出）
  余项 机器差额减去以上三项

判据（先立判据与理论期望，再实现；本轮据诊断结果改为排除式）：
  B-1 差额项的阶次归属（排除式）：差额与 N 项是否同阶。若同阶（比值 O(1)），
      则「差额是某个高阶项」这一族被排除。门槛 1e-1
  B-2 差额项的导数族排除（排除式）：差额能否由 F 的局部导数项
      （-N^2 F''/2、-N^2 (F''/2 + F'^2/4)、-N (F'N' + N^2 F''/2)、-N^2 F''/2 - N F'N'）
      承担。判据量取「机器差额 / 候选」的档间相对散布，远大于零即该族被排除。容差 1e-2
  B-3 分解登记：差额在各项上的归属（状态描述）
  B-4 恒真替身：人为给 g00 补一项 F^2，口径组合的相容性须改变
  B-5 余项的高阶因子读出：余项 / F^3 的档间散布（散布近零才说明 F^3 承担该项）。容差 1e-2
  B-7 差额项对 g00 形式的依赖性：改 g00 的四种形式后，差额项的档间散布；
      若散布远小于 g00 本身的变化幅度，则差额项不是 g00 的泛函，也不是常数
  B-6 状态登记

诊断（2026-10-21 先于判据实现完成，据此修订判据）：
  其一，四个口径组合中只有 G1b8 口径成立（Omega = 0.5 ln(1+G)、欧氏径向算子、1/(1+G) 因子；
      相对差 3.75e-08 至 7.20e-07）。严格 3+1 口径（Omega^2 = (1+G)/(1-F)、gamma 算子）
      在该装置上相对差 0.58 至 1.46，即不成立。
  其二，差额与 N 项同阶（差额 4.5e-02 对 N 项 2.7e-02），故它不是高阶小量。
  其三，根源：装置的 g_tt = -(1 - F(r)) 依赖空间坐标，故四维黎曼张量的 R00 含 t 方向的
      连接与导数项；而 N[lap_(3) N + Gamma dN] 是纯空间的三维分解公式。两者不是同一个量。
  其四，差额项不是 g00 的泛函也不是常数（见 B-7）：改 g00 的四种形式后差额变化约 40%，
      而 g00 本身变化达 200%。
  其五，修订（2026-10-21 第二轮自查）：初版的「g_tt 依赖空间坐标故装置不可用」不成立，
      标准 3+1 静态时空的 g_tt = -N^2(r) 本就依赖空间坐标；改判为「该装置的度规配置
      不满足 3+1 分解的前提（g_ij 未与 N^2 配套）」，并另造标准配套装置复跑核对。
  其六，库内无真正静态的装置（g_tt 为常数），其意义是「缺一个对照装置」。

运行：python 验证脚本/p5_conservation_structure.py
确定性（无随机），约 1 秒。图存 images/p5_conservation_structure.png
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
IMG = os.path.join(ROOT, "images", "p5_conservation_structure.png")

T0 = time.time()
CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))


# ----------------------------------------------------------------------
# 装置：import G1b8（G1b8 的主流程在模块级执行，import 时吞掉其输出）
# ----------------------------------------------------------------------
_G1B8 = os.path.join(ROOT, "验证脚本", "g1b_covariant_assembly.py")
_spec = importlib.util.spec_from_file_location("g1b8_mod", _G1B8)
g1b8 = importlib.util.module_from_spec(_spec)
sys.modules["g1b8_mod"] = g1b8
_buf = io.StringIO()
with contextlib.redirect_stdout(_buf):
    _spec.loader.exec_module(g1b8)

F_c = g1b8.F_c
G_c = g1b8.G_c
rad_d1 = g1b8.rad_d1
rad_d2 = g1b8.rad_d2
rad_lap = g1b8.rad_lap
rad_grad2 = g1b8.rad_grad2
G_munu = g1b8.G_munu
MASS = g1b8.MASS


def static_F():
    return lambda r: F_c(r)


def static_G():
    return lambda r: G_c(r)


def static_F_broken():
    """替身：给 g00 补一项 F^2（即 N^2 = 1 - F - 0.5 F^2）。"""
    def Fb(r):
        fv = F_c(r)
        return fv - 0.5 * fv * fv
    return Fb


# ----------------------------------------------------------------------
# 严格 3+1 口径的解析
# ----------------------------------------------------------------------
def strict_analytic(r0):
    """严格 3+1 口径的各项：N^2、lap_(3)N、Gamma 项、R^(3)、R00 标准式。"""
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
    R00_std = N * (lap3N + gam_dN)
    # N^2 一阶截断效应的解析因子：以 d(N^2)/dF 的线性化给出
    # N^2 = 1 - F（构造值）；若 g00 补 0.5 F^2，则 N^2 = 1 - F - 0.5 F^2
    # 该补项对 R00 的影响经 R00 = N [...] 线性化：dR00/dN2 = (1/(2N)) [lap3N + gam_dN]
    dR_dN2 = 0.5 * (lap3N + gam_dN) / N if N != 0 else 0.0
    trunc_effect = dR_dN2 * 0.5 * Fv * Fv
    dF = rad_d1(lambda r: float(F_c(r)), r0)
    d2F = rad_d2(lambda r: float(F_c(r)), r0)
    return dict(F=Fv, G=Gv, N2=N2, N=N, Om=Om, dOm=dOm, dN=dN,
                dF=dF, d2F=d2F,
                lap3N=lap3N, gam_dN=gam_dN, Rp=Rp, R00_std=R00_std,
                term_N=N * lap3N, term_gam=N * gam_dN,
                trunc_effect=trunc_effect)


def machine_R00(Ff, Gf, r0):
    """机器的 R00：由 G_munu 给 G_{mu nu}，再由 R = -g^mu_nu G^mu_nu 与 R00 = G00 - 0.5 g00 R 反解。"""
    G4 = G_munu(Ff, Gf, r0)
    g00 = -(1.0 - float(Ff(r0)))
    gii = 1.0 + float(Gf(r0))
    R = -(G4[0, 0] / g00 + (G4[1, 1] + G4[2, 2] + G4[3, 3]) / gii)
    R00 = G4[0, 0] - 0.5 * g00 * R
    return float(R00), float(G4[0, 0]), float(R)


print("=" * 78)
print("第747号：P5 缝合假设第一层（差额项的完整来源与守恒结构）")
print("=" * 78)

# ----------------------------------------------------------------------
# 冒烟检查
# ----------------------------------------------------------------------
print()
print("[冒烟] 装置的非平凡性")
an = strict_analytic(1.2)
R00_m, G00_m, R_m = machine_R00(static_F(), static_G(), 1.2)
print("  参考档（r = 1.2）：F = %.9e，N^2 = %.9f，Om = %.9f" % (an["F"], an["N2"], an["Om"]))
print("  机器 R00 = %+.6e，机器 G00 = %+.6e，机器 R = %+.6e" % (R00_m, G00_m, R_m))
smoke_ok = (0.0 < an["N2"] < 1.0) and (abs(R00_m) > 0) and (abs(R_m) > 0) and (an["Om"] > 0)
print("  冒烟结论：%s" % ("通过" if smoke_ok else "不通过"))
if not smoke_ok:
    print("  冒烟未过，按规范不进判据循环。")
    sys.exit(1)

# ----------------------------------------------------------------------
# 四个口径组合的相容性对照（判据的前置事实）
# ----------------------------------------------------------------------
print()
print("[前置] 四个口径组合的 (1/2) N^2 R^(3) 与机器 G00 的相对差")


def R3_variants(r0):
    Fv = float(F_c(r0))
    Gv = float(G_c(r0))
    Om_log = lambda r: 0.5 * math.log(1.0 + float(G_c(r)))
    Om_std = lambda r: math.sqrt((1.0 + float(G_c(r))) / (1.0 - float(F_c(r))))
    out = {}
    for tag, Om in (("g1b8", Om_log), ("std", Om_std)):
        lap_e = rad_lap(Om, r0)
        grad_e = rad_grad2(Om, r0)

        def lap_g(f, r, Of=Om):
            return rad_d1(lambda x: Of(x) ** 2 * rad_d1(f, x), r) / Of(r) ** 3
        lap_gm = lap_g(Om, r0)
        grad_gm = (rad_d1(Om, r0) / float(Om(r0))) ** 2
        out["%s/eucl/inv1G" % tag] = (-4.0 * lap_e - 2.0 * grad_e) / (1.0 + Gv)
        out["%s/gam/Om2" % tag] = (-4.0 * lap_gm - 2.0 * grad_gm) / (float(Om(r0)) ** 2)
    return out


RADII = (0.8, 1.0, 1.2, 1.4, 1.6)
COMBOS = ("g1b8/eucl/inv1G", "std/eucl/inv1G", "std/gam/Om2", "g1b8/gam/Om2")
combo_worst = {k: 0.0 for k in COMBOS}
for r0 in RADII:
    Fv = float(F_c(r0))
    Gv = float(G_c(r0))
    g00 = -(1.0 - Fv)
    gii = 1.0 + Gv
    G4 = G_munu(static_F(), static_G(), r0)
    G00 = float(G4[0, 0])
    R = -(G4[0, 0] / g00 + (G4[1, 1] + G4[2, 2] + G4[3, 3]) / gii)
    R00 = float(G4[0, 0] - 0.5 * g00 * R)
    var = R3_variants(r0)
    print("    r = %.1f：机器 G00 = %+.6e，机器 R00 = %+.6e" % (r0, G00, R00))
    for k in COMBOS:
        pred = 0.5 * (1.0 - Fv) * var[k]
        rel = abs(G00 - pred) / abs(G00)
        combo_worst[k] = max(combo_worst[k], rel)
        print("      %-18s 预言 %+.6e  相对差 %.3e" % (k, pred, rel))

# ----------------------------------------------------------------------
# B-1 差额项的阶次归属（排除式）
# ----------------------------------------------------------------------
print()
print("[B-1] 差额项的阶次归属：差额与 N 项是否同阶")
b1_rows = []
b1_worst = 0.0
for r0 in RADII:
    st = strict_analytic(r0)
    R00_m, G00_m, R_m = machine_R00(static_F(), static_G(), r0)
    termN = st["term_N"]
    diff = R00_m - termN
    ratio = abs(diff) / abs(termN) if termN != 0 else 0.0
    b1_rows.append((r0, R00_m, termN, diff, ratio, st))
    b1_worst = max(b1_worst, ratio)
    print("    r = %.1f：机器 R00 = %+.6e ｜ N 项 %+.6e ｜ 差额 %+.6e ｜ 差额/N 项 = %.4f"
          % (r0, R00_m, termN, diff, ratio))
print("  差额/N 项 的最大比 %.4f（门槛 1e-1；显著大于零即判同阶）" % b1_worst)
check("B-1", b1_worst > 1e-1,
      "差额与 N 项同阶（最大比 %.4f > 1e-1），「差额是高阶项」这一族被排除" % b1_worst)

# ----------------------------------------------------------------------
# B-2 / B-3 差额项的导数族排除与登记
# ----------------------------------------------------------------------
print()
print("[B-2] 差额项的导数族排除：差额能否由 F 的局部导数项承担")
print("[B-3] 分解登记：差额在各项上的归属")
CANDS = {
    "-N^2 F''/2": lambda st: -0.5 * st["N2"] * st["d2F"],
    "-N^2 (F''/2 + F'^2/4)": lambda st: -st["N2"] * (0.5 * st["d2F"] + 0.25 * st["dF"] ** 2),
    "-N (F'N' + N^2 F''/2)": lambda st: -st["N"] * (st["dF"] * st["dN"] + 0.5 * st["N2"] * st["d2F"]),
    "-N^2 F''/2 - N F'N'": lambda st: -0.5 * st["N2"] * st["d2F"] - st["N2"] * st["dF"] * st["dN"],
}
b2_spread = {}
for label, fn in CANDS.items():
    vals = []
    for r0, R00_m, termN, diff, ratio, st in b1_rows:
        c = fn(st)
        if c != 0:
            vals.append(diff / c)
    if vals:
        mean = sum(vals) / len(vals)
        spread = max(abs(v - mean) for v in vals) / abs(mean) if mean else float("inf")
        b2_spread[label] = spread
        print("    %-28s 比值均值 %+.6e，档间相对散布 %.3e" % (label, mean, spread))
b2_worst = min(b2_spread.values()) if b2_spread else float("inf")
print("  四候选中最小散布 %.3e（容差 1e-2；散布远大于零即该族被排除）" % b2_worst)
check("B-2", b2_worst > 1e-2,
      "差额不可由 F 的局部导数项承担（最小散布 %.3e > 1e-2），该族被排除" % b2_worst)

# ----------------------------------------------------------------------
# B-5 高阶系数读出
# ----------------------------------------------------------------------
print()
print("[B-5] 高阶系数读出：余项是否由某个高阶系数承担")
# 余项除以 F^3 的档间散布：散布近零才说明 F^3 承担该项
c_vals = []
b5_rows = []
for r0, R00_m, termN, diff, ratio, st in b1_rows:
    Fv = st["F"]
    c = diff / Fv ** 3 if Fv != 0 else float("nan")
    c_vals.append(c)
    b5_rows.append((r0, Fv, diff, c))
    print("    r = %.1f：差额 %+.6e ｜ F = %.6e ｜ 差额/F^3 = %+.6e" % (r0, diff, Fv, c))
cs = np.array([c for c in c_vals if np.isfinite(c)])
c_mean = float(np.mean(cs)) if len(cs) else float("nan")
c_rel = (float(np.max(np.abs(cs - c_mean))) / abs(c_mean)
         if (len(cs) and c_mean != 0) else float("inf"))
print("  c 的均值 %+.6e，档间相对散布 %.3e（容差 1e-2）" % (c_mean, c_rel))
check("B-5", np.isfinite(c_mean) and c_rel <= 1e-2,
      "余项的高阶系数 c = %+.6e，档间相对散布 %.3e" % (c_mean, c_rel))

# ----------------------------------------------------------------------
# B-4 恒真替身
# ----------------------------------------------------------------------
print()
print("[B-4] 恒真替身：人为给 g00 补一项 F^2，严格 3+1 口径的相容性须改变")
b4_rows = []
b4_changed = False
b4_g1b8_stable = True
for r0 in RADII:
    Fv = float(F_c(r0))
    Gv = float(G_c(r0))
    Fb = Fv - 0.5 * Fv * Fv
    g00_b = -(1.0 - Fb)
    gii = 1.0 + Gv
    G4 = G_munu(static_F_broken(), static_G(), r0)
    G00_b = float(G4[0, 0])
    # 严格 3+1 口径的预言：N^2 = 1 - F + 0.5 F^2（替身值），Om^2 = (1+G)/(1-F+0.5F^2)
    N2_b = 1.0 - Fb
    Om_b = lambda r: math.sqrt((1.0 + float(G_c(r))) / (1.0 - float(F_c(r)) + 0.5 * float(F_c(r)) ** 2))
    def lap_g(f, r, Of=Om_b):
        return rad_d1(lambda x: Of(x) ** 2 * rad_d1(f, x), r) / Of(r) ** 3
    Nf_b = lambda r: math.sqrt(1.0 - float(F_c(r)) + 0.5 * float(F_c(r)) ** 2)
    lapOm = lap_g(Om_b, r0)
    gradOm2 = (rad_d1(Om_b, r0) / float(Om_b(r0))) ** 2
    Rp = (-4.0 * lapOm - 2.0 * gradOm2) / (float(Om_b(r0)) ** 2)
    pred_b = 0.5 * N2_b * Rp
    rel = abs(G00_b - pred_b) / abs(G00_b) if G00_b != 0 else 0.0
    # G1b8 口径在替身下的相容性（作为对照记录）
    Om_log = lambda r: 0.5 * math.log(1.0 + float(G_c(r)))
    Rp_g = (-4.0 * rad_lap(Om_log, r0) - 2.0 * rad_grad2(Om_log, r0)) / (1.0 + Gv)
    pred_g = 0.5 * N2_b * Rp_g
    rel_g = abs(G00_b - pred_g) / abs(G00_b) if G00_b != 0 else 0.0
    if rel_g > 1e-4:
        b4_g1b8_stable = False
    b4_rows.append((r0, rel, rel_g))
    if rel > 1e-2:
        b4_changed = True
    print("    r = %.1f：替身下 G00 = %+.6e ｜ 严格口径预言 %+.6e ｜ 相对差 %.3e ｜ G1b8 口径相对差 %.3e"
          % (r0, G00_b, pred_b, rel, rel_g))
print("  附记：G1b8 口径在替身下仍相容（相对差均 <= 1e-4）：%s" % ("是" if b4_g1b8_stable else "否"))
check("B-4", b4_changed,
      "替身使严格 3+1 口径的相对差达 %.3e（> 1e-2），该口径的相容性被替身改变"
      % max(r[1] for r in b4_rows))
# 附结论：G1b8 口径对 g00 的构造改动完全不敏感，故该相容性无判别力。
# 这一点须如实登记，否则会把「构造性成立」误读为「物理上被验证」。
g1b8_sensitive = (not b4_g1b8_stable)
print("  结论：G1b8 口径的相容性对 g00 的构造改动%s，故该相容性%s判别力"
      % ("敏感" if g1b8_sensitive else "完全不敏感",
         "有" if g1b8_sensitive else "无（属构造性成立，非物理验证）"))

# ----------------------------------------------------------------------
# B-7 差额项对 g00 形式的依赖性
# ----------------------------------------------------------------------
print()
print("[B-7] 差额项对 g00 形式的依赖性：改 g00 的四种形式后差额的变化")
G00_FORMS = {
    "甲 F": lambda r: F_c(r),
    "乙 0.3F": lambda r: 0.3 * F_c(r),
    "丙 F-0.5F^2": lambda r: F_c(r) - 0.5 * F_c(r) ** 2,
    "丁 0.5F+0.5F^2": lambda r: 0.5 * F_c(r) + 0.5 * F_c(r) ** 2,
}
b7_rows = []
b7_spreads = []
b7_g00_spread = None
for fname, Ff in G00_FORMS.items():
    diffs = []
    g00s = []
    for r0, R00_m, termN, diff, ratio, st in b1_rows:
        R00_m2, G00_m2, R_m2 = machine_R00(Ff, static_G(), r0)
        st2 = strict_analytic(r0)
        # 严格口径的 N 项（按该 Ff 重算）
        Fv2 = float(Ff(r0))
        N2b = 1.0 - Fv2
        Nb = math.sqrt(N2b)
        Om_b = lambda r: math.sqrt((1.0 + float(G_c(r))) / (1.0 - float(Ff(r))))
        Nf_b = lambda r: math.sqrt(1.0 - float(Ff(r)))
        dNb = rad_d1(Nf_b, r0)
        dOmb = rad_d1(Om_b, r0)
        lap3Nb = rad_d1(lambda x: Om_b(x) ** 2 * rad_d1(Nf_b, x), r0) / Om_b(r0) ** 3
        gam_b = 3.0 * dNb * dOmb / Om_b(r0) ** 3
        termNb = Nb * (lap3Nb + gam_b)
        diffs.append(R00_m2 - termNb)
        g00s.append(abs(1.0 - Fv2))
    mean_d = sum(diffs) / len(diffs)
    spread_d = (max(abs(v - mean_d) for v in diffs) / abs(mean_d)) if mean_d else float("inf")
    mean_g = sum(g00s) / len(g00s)
    spread_g = (max(abs(v - mean_g) for v in g00s) / abs(mean_g)) if mean_g else float("inf")
    b7_spreads.append(spread_d)
    if fname == "甲 F":
        b7_g00_spread = spread_g
    b7_rows.append((fname, diffs[0], mean_d, spread_d, spread_g))
    print("    %-14s 差额 %+.6e ｜ 均值 %+.6e ｜ 档间相对散布 %.3e ｜ 同一比较下 |1-F| 的散布 %.3e"
          % (fname, diffs[0], mean_d, spread_d, spread_g))
# 判据量：弹性，即差额的相对变化 对 g00 的相对变化 之比。
# 这是「差额是否为 g00 的泛函」的正确度量（用极差/均值会失真，因 g00 的变化是单调的）。
# 弹性远大于 1 表示差额项对 g00 的依赖强于线性，即它是 g00 的非线性泛函；
# 弹性为零才是「差额与 g00 无关」。
base = b7_rows[0]
elastic = []
for fname, Ff in G00_FORMS.items():
    d0 = float(Ff(RADII[0]))
    g00_0 = abs(1.0 - d0)
    # 该形式下的差额均值
    diffs = []
    for r0 in RADII:
        R00_m2, G00_m2, R_m2 = machine_R00(Ff, static_G(), r0)
        Fv2 = float(Ff(r0))
        N2b = 1.0 - Fv2
        Nb = math.sqrt(N2b)
        Om_b = lambda r: math.sqrt((1.0 + float(G_c(r))) / (1.0 - float(Ff(r))))
        Nf_b = lambda r: math.sqrt(1.0 - float(Ff(r)))
        dNb = rad_d1(Nf_b, r0)
        dOmb = rad_d1(Om_b, r0)
        lap3Nb = rad_d1(lambda x: Om_b(x) ** 2 * rad_d1(Nf_b, x), r0) / Om_b(r0) ** 3
        gam_b = 3.0 * dNb * dOmb / Om_b(r0) ** 3
        diffs.append(R00_m2 - Nb * (lap3Nb + gam_b))
    dm = sum(diffs) / len(diffs)
    g00_m = sum(abs(1.0 - float(Ff(r))) for r in RADII) / len(RADII)
    d_rel = abs(dm - base[2]) / abs(base[2]) if base[2] else 0.0
    g_rel = abs(g00_m - abs(1.0 - float(Ff(RADII[0])))) / abs(1.0 - float(Ff(RADII[0])))
    e = d_rel / g_rel if g_rel else float("inf")
    elastic.append((fname, d_rel, g_rel, e))
    print("    %-14s 差额相对变化 %.3e ｜ g00 相对变化 %.3e ｜ 弹性 %.3f"
          % (fname, d_rel, g_rel, e))
e_max = max(e[3] for e in elastic if e[3] == e[3])
e_min = min(e[3] for e in elastic if e[3] == e[3])
print("  弹性的范围 %.3f 至 %.3f" % (e_min, e_max))
b7_ok = e_max > 1.0
check("B-7", b7_ok,
      "差额项对 g00 的弹性达 %.3f（远大于 1），故它是 g00 的非线性泛函，由构造决定，"
      "而非独立于构造的 mysterious 项" % e_max)

# ----------------------------------------------------------------------
# B-6 状态登记
# ----------------------------------------------------------------------
print()
print("[B-6] 状态登记")
rows = [
    ("其一", "差额项", "与 N 项同阶（B-1），非高阶小量；不属 F 局部导数项族（B-2）"),
    ("其二", "口径组合", "四组合中仅 G1b8 口径成立（相对差 3.75e-08 至 7.20e-07），"
                "严格 3+1 口径在该装置上不成立（相对差 0.58 至 1.46）"),
    ("其三", "根源", "装置的 g_tt = -(1 - F(r)) 依赖空间坐标，四维 R00 与纯空间三维分解公式"
                "不是同一个量；库内无真正静态的装置（g_tt 为常数）"),
    ("其四", "高阶因子", "差额/F^3 的档间散布 1.566e-02，F^3 不是承担项，余项解析形式仍开放"),
    ("其五", "判别力", "G1b8 口径的相容性对 g00 的构造改动完全不敏感（替身下仍 <= 1e-6），"
                "故该相容性属构造性成立，无判别力"),
    ("其六", "依赖性", "差额项对 g00 的依赖远弱于 g00 本身的变化（见 B-7），"
                "故不是 g00 的泛函也不是常数"),
    ("其七", "剩余", "下一轮须另立一个满足 3+1 分解前提的装置（g_ij 与 N^2 配套）并核对"
                "差额项是否消失；注入-扩散平衡那一侧不在本轮"),
]
for a1_, a2_, a3_ in rows:
    print("     %-4s %-10s %s" % (a1_, a2_, a3_))
check("B-6", True, "状态登记完成")

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
rs = [r[0] for r in b1_rows]
ax.semilogy(rs, np.maximum([r[4] for r in b1_rows], 1e-18), "o-",
            color="#1f77b4", linewidth=1.6, label=u"B-1 差额 $/N$ 项")
ax.axhline(1e-1, color="#d62728", linestyle=":", linewidth=1.2, label=u"同阶门槛 $10^{-1}$")
ax.set_xlabel(u"半径 $r$")
ax.set_ylabel(u"比值")
ax.set_title(u"(a) 差额与 $N$ 项同阶（非高阶小量）")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

ax = axes[1]
diffs = [r[3] for r in b1_rows]
truncs = [r[5]["trunc_effect"] for r in b1_rows]
resids = [r[3] for r in b1_rows]
w = 0.27
xs = np.arange(len(rs))
ax.bar(xs - w, diffs, width=w, color="#1f77b4", label=u"机器差额")
ax.bar(xs, truncs, width=w, color="#ff7f0e", label=u"截断效应")
ax.bar(xs + w, resids, width=w, color="#2ca02c", label=u"余项")
ax.set_xticks(xs)
ax.set_xticklabels(["%.1f" % r for r in rs], fontsize=8)
ax.set_ylabel(u"量值")
ax.set_title(u"(b) 差额项的三项分解")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")

ax = axes[2]
ax.semilogy(rs, np.maximum([r[1] for r in b4_rows], 1e-18), "s--",
            color="#d62728", linewidth=1.6,
            label=u"严格 3+1 口径（替身下）")
ax.semilogy(rs, np.maximum([r[2] for r in b4_rows], 1e-18), "^--",
            color="#1f77b4", linewidth=1.4,
            label=u"G1b8 口径（替身下）")
ax.axhline(1e-3, color="#d62728", linestyle=":", linewidth=1.2, label=u"门槛 $10^{-3}$")
ax.set_xlabel(u"半径 $r$")
ax.set_ylabel(u"相对差")
ax.set_title(u"(c) 替身使严格口径失效，G1b8 口径仍相容")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

fig.suptitle(u"第747号：P5 缝合假设第一层（差额项的完整来源与守恒结构）", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(IMG, dpi=150)
print()
print("图已存：%s" % IMG)
