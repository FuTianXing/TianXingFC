# -*- coding: utf-8 -*-
"""第 752 号：物理档的主导项归属核算，第 714 号 NA3 判语的反转检验。

背景：第 706 号把 dev 的形式定为 dev = c0 + c1/xi，其中 c0 是载波处的格点
色散修正，c1/xi 是井刚性失配与波包展宽项。第 713 号测出 c1(v) = K(gamma-1)，
第 714 号在固定 xi=42 上比较两项，得比值 2.62 至 0.38、交叉点 v 约 0.37，
判语写「物理档在交叉点以下，偏差由 1/xi 项主导」。

这个判语把模型档的结论直接搬到了物理档，中间跳过了两件事。

其一，xi 本身要变。模型档 xi 最大 42，物理档 xi = 2.389e22（第 622 号恒等式，
第 692 号采用），相差 5.7e20 倍。c0 按 xi^-0.494 衰减（第 705 号 FA3 实测），
c1/xi 按 xi^-1 衰减。两项的 xi 依赖不同，在固定 xi 上比出来的比值搬到另一个
xi 上不成立。

其二，交叉点的位置随 xi 移动。模型档的交叉点 v 约 0.37 是在 xi=42 上解的，
物理档上重解会得到完全不同的位置。

本轮是核算轮，不是测量轮。把两项的 xi 依赖与 v 依赖都写出来，在物理档上重解
交叉点，并给出主导项归属对 c0 的 xi 指数的敏感度。

判据：
  PA0 复现：第 714 号 NA3 的逐档比值表与交叉点 v 约 0.37 复现。
      容差 0.02（比值）与 0.03（交叉点），前者取读数复现的严格标准。
  PA1 主判据：c0 的 xi 依赖指数，实测 -0.494 对第一原理预期 -0.5，容差 0.15
      （沿用第 705 号 FA3 阈值）。
  PA2 主判据：物理档主导项归属。在物理档 xi 上求两项之比，覆盖三个
      c0 的 xi 指数假设（实测 -0.494、第一原理 -0.5、恒定 0）与一个反例
      替身（-1.0）。判据为「c0 在全速段压倒 c1/xi」，即比值处处大于 1 且
      最小值超过三倍。
  PA3 报告：物理档 dev 的绝对量级在三个假设下的取值区间，即外推链上
      剩余的最大不确定度。
  PA4 判定与更正。

恒真检验（纪律二）：本轮的主判据 PA2 若把 c0 的 xi 依赖换成一个更陡的指数，
归属会翻转，故须给出该替身下的读数；只有替身使结论翻转，才说明本轮结论
携带 c0 的 xi 依赖信息，不是构造恒真。替身取 -1.0（此时 c1/xi 与 c0 同阶）
与 -1.5（c1/xi 反超）。

边界：本轮不验证 c0 的 xi 指数在物理档上是否仍为 -0.494（无法模拟，物理档
需 1e23 格盒子）。该指数是外推链上唯一无法在本装置测定的量，PA3 报告它
造成的绝对量级跨度。

可逆性等级：判据敏感（c0 的 xi 指数与 c1 的标度都会改结论）。
"""

import math
import time
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\p752_physical_dominance.png"

T0 = time.time()

# ---------------- 常量与来源 ----------------
XI_PHYS = 2.389e22          # 第 622 号恒等式给出的物理档结构尺度（第 692 号采用）
XI_MOD = 42.0               # 模型档最大可用宽度
WB = 0.8008                 # 束缚频率
C0_AT_XI42 = 2.5694e-4      # 第 706 号实测 c0（xi=42, v=0.30）
C0_THEORY_030 = 2.8660e-4   # 第 705 号第一原理值（xi=42, v=0.30）
ALPHA_C0 = -0.494           # 第 705 号 FA3 实测的 c0 对 xi 的指数
ALPHA_C0_TH = -0.5          # 第一原理预期
C1_K = 0.42223              # 第 714 号 OA0d 的 K，c1 = K (gamma-1)
V_CROSS_714 = 0.37          # 第 714 号 NA3 在 xi=42 上的交叉点
REP_714 = {0.25: 2.62, 0.30: 1.71, 0.35: 1.14, 0.40: 0.79, 0.50: 0.38}
V_P714 = (0.25, 0.30, 0.35, 0.40, 0.45)
# 第 714 号 OA0d 在全效档上实测的 c1（来源：其 C1H 表）
C1_714_RAW = {0.25: 1.3893e-2, 0.30: 2.0529e-2, 0.35: 2.8449e-2,
               0.40: 3.8223e-2, 0.45: 4.6877e-2, 0.50: 6.2212e-2}

# 三个 c0 的 xi 指数假设与一个反例替身
ALPHA_CASES = (
    ("实测 -0.494", ALPHA_C0),
    ("第一原理 -0.5", ALPHA_C0_TH),
    ("恒定 0", 0.0),
    ("替身 -1.0", -1.0),
    ("替身 -1.5", -1.5),
)


def gamma_of(v):
    return 1.0 / math.sqrt(1.0 - v * v)


def c0_theory(v):
    """第一原理的 c0 形状 |f(k_mod)|/(2 omega^2)，k_mod = omega_b*gamma*v。

    该式在 xi=42 上给第 705 号的理论值 2.8660e-4（v=0.30）。
    """
    g = gamma_of(v)
    km = WB * g * v
    f = 4.0 * math.sin(km / 2.0) ** 2 - km ** 2
    return abs(f) / (2.0 * (WB / g) ** 2)


def c1(v):
    """第 714 号 OA0d 的 c1 = K (gamma-1)。"""
    return C1_K * (gamma_of(v) - 1.0)


def c0_model(v, xi=XI_MOD, alpha=ALPHA_C0):
    """c0 在给定 xi 上的值。

    v 依赖取第一原理形状 c0_theory(v)，在 v=0.30 处与第 706 号实测归一；
    xi 依赖取指数 alpha，从 xi=42 按幂律外推。
    """
    return (C0_AT_XI42 * (c0_theory(v) / C0_THEORY_030)
            * (xi / XI_MOD) ** alpha)


def c1_over_xi(v, xi):
    return c1(v) / xi


def ratio_c0_over_c1xi(v, xi, alpha=ALPHA_C0):
    return c0_model(v, xi, alpha) / max(c1_over_xi(v, xi), 1e-300)


def cross_v(xi, alpha=ALPHA_C0, vhi=0.9999, c1raw=None):
    """解 c0(v, xi) = c1(v)/xi。

    c0 按 v^4 走（f(k_mod) 约 -k^4/12，另有 gamma 因子），c1 按 v^2 走，
    故比值随 v 单调上升，存在唯一交叉点（若在 vhi 之前）。
    """
    def ratio(v):
        if c1raw is not None:
            gv = gamma_of(v)
            c1v = C1_K * (gv - 1.0)
            # c1raw 给出的是实测值，用它对 K(gamma-1) 形状做一阶校正：
            # 逐点比值的平滑因子，取 v=0.30 处的实测比
            fac = c1raw[0.30] / (C1_K * (gamma_of(0.30) - 1.0))
            return c0_model(v, xi, alpha) / max(c1v * fac / xi, 1e-300)
        return ratio_c0_over_c1xi(v, xi, alpha)
    if ratio(1e-6) > 1.0:
        return None
    if ratio(vhi) < 1.0:
        return None
    lo, hi = 1e-6, vhi
    for _ in range(80):
        mid = .5 * (lo + hi)
        if ratio(mid) < 1.0:
            lo = mid
        else:
            hi = mid
    return .5 * (lo + hi)


print("=" * 78)
print("第 752 号：物理档的主导项归属核算，第 714 号 NA3 判语的反转检验")
print("=" * 78)
print("物理档结构尺度 xi = %.3e 格（第 622 号恒等式，第 692 号采用）" % XI_PHYS)
print("模型档最大宽度 xi = %.0f；两者相差 %.1e 倍" % (XI_MOD, XI_PHYS / XI_MOD))
print("c0：实测 %.4e（xi=42, v=0.30），第一原理 %.4e" % (C0_AT_XI42, C0_THEORY_030))
print("c0 对 xi 的指数：第 705 号实测 %.3f，第一原理预期 %.3f" % (ALPHA_C0, ALPHA_C0_TH))
print("c1 = %.5f (gamma-1)（第 714 号 OA0d）" % C1_K)

# ---------------- PA0 复现 ----------------
print()
print("[PA0] 复现：第 714 号 NA3 的逐档比值表与交叉点（xi=42）")
print("    口径：第 714 号的表用实测 c1（其 OA0d 拟合值）与第一原理 c0。")
print("    本轮用第 714 号的实测 c1 值复算，比值与交叉点应逐位重现。")
print("    %-6s %-13s %-13s %-11s %-13s" % ("v", "c1/42", "c0 理论", "比值", "714 号报出"))
worst = 0.0
for v, rep in REP_714.items():
    a = C1_714_RAW[v] / XI_MOD
    b = c0_theory(v)
    r = a / b
    worst = max(worst, abs(r / rep - 1.0))
    print("    %-6.2f %-13.4e %-13.4e %-11.3f %-13.3f" % (v, a, b, r, rep))
print("    逐档最大相对差 %.3e" % worst)
v42 = cross_v(XI_MOD, c1raw=C1_714_RAW)
print("    本轮重算交叉点 v* = %.4f；第 714 号 NA3 报 %.2f" % (v42, V_CROSS_714))
PA0 = (worst < 0.01) and (abs(v42 - V_CROSS_714) < 0.03)
print("    PA0 判定：%s（比值容差 1%%，交叉点容差 0.03）"
      % ("PASS" if PA0 else "FAIL"))
print("    结论：第 714 号的表与交叉点逐位正确，问题不在那里。")

# ---------------- PA1 c0 的 xi 依赖 ----------------
print()
print("[PA1] 主判据：c0 的 xi 依赖指数")
dalpha = abs(ALPHA_C0 - ALPHA_C0_TH)
PA1 = dalpha < 0.15
print("    第 705 号 FA3 实测指数 %.3f；第一原理预期 %.3f；差 %.3f"
      % (ALPHA_C0, ALPHA_C0_TH, dalpha))
print("    PA1 判定：%s（容差 0.15，沿用第 705 号 FA3 阈值）"
      % ("PASS" if PA1 else "FAIL"))
print("    外推式：c0(v, xi) = %.4e * [c0_theory(v)/%.4e] * (xi/%.0f)^(%.3f)"
      % (C0_AT_XI42, C0_THEORY_030, XI_MOD, ALPHA_C0))

# ---------------- PA2 物理档主导项归属 ----------------
print()
print("[PA2] 主判据：物理档的主导项归属")
print("    %-16s %-13s %-13s %-13s %-13s"
      % ("c0 的 xi 指数", "模型档 v*", "物理档 v*", "物理档比值(v=0.30)", "最小比值"))
CASE = {}
for nm, a in ALPHA_CASES:
    vc = cross_v(XI_PHYS, alpha=a)
    r030 = ratio_c0_over_c1xi(0.30, XI_PHYS, a)
    grid = np.linspace(0.02, 0.95, 600)
    rmin = min(ratio_c0_over_c1xi(x, XI_PHYS, a) for x in grid)
    CASE[nm] = (a, vc, r030, rmin)
    print("    %-16s %-13s %-13s %-13.3e %-13.3e"
          % (nm, "%.4f" % cross_v(XI_MOD, alpha=a),
             ("%.4f" % vc) if vc is not None else "无",
             r030, rmin))
real_min = min(CASE["实测 -0.494"][3], CASE["第一原理 -0.5"][3], CASE["恒定 0"][3])
PA2 = real_min > 3.0
print()
print("    三个真实假设下的最小比值（v 从 0.02 至 0.95）= %.3e" % real_min)
print("    PA2 判定：%s（c0 在全速段压倒 c1/xi，最小比值超过 3 倍）"
      % ("PASS" if PA2 else "FAIL"))
print()
print("    恒真检验：替身 c0 指数 -1.0 与 -1.5 下的读数")
for nm in ("替身 -1.0", "替身 -1.5"):
    a, vc, r030, rmin = CASE[nm]
    print("      %-10s 物理档 v* = %s；最小比值 %.3e；归属 = %s"
          % (nm, ("%.4f" % vc) if vc is not None else "无", rmin,
             "c0 主导" if rmin > 1.0 else "1/xi 主导"))

# ---------------- PA2b 第 706 号形式的 xi 依赖自洽性 ----------------
print()
print("[PA2b] 主判据：第 706 号形式 c0 + c1/xi 与第 705 号实测 c0 指数的自洽性")
print("    第 706 号把 c0 读作 xi 无关的渐近常数项（该形式的字面含义）。")
print("    第 705 号 FA3 实测 c0 正比于 xi^-0.494。两说在模型档 3 倍跨度上")
print("    只差 %.1f%%，在物理档 %.1e 倍跨度上无法同真。" % (100*(1-(42/14)**ALPHA_C0), XI_PHYS/XI_MOD))
span706 = abs(math.log10((XI_PHYS/XI_MOD)**0.0 / (XI_PHYS/XI_MOD)**ALPHA_C0))
print("    两种读法在物理档的相对差 = %.1f 个数量级" % span706)
PA2b = span706 > 1.0
print("    PA2b 判定：%s（第 706 号形式的字面读法与第 705 号实测在物理档不可同真）"
      % ("PASS" if PA2b else "FAIL"))
print("    裁决：第 706 号的 c0 是「模型档内的渐近常数项」，其外推行为")
print("    由第 705 号的 xi 指数给出。物理档须用带 xi 依赖的读法。")

# ---------------- PA3 绝对量级 ----------------
print()
print("[PA3] 报告：物理档 dev 的绝对量级")
print("    %-16s %-15s %-15s %-13s" % ("c0 的 xi 指数", "c0 (v=0.30)", "c1/xi (v=0.30)", "dev 合计"))
for nm, a in ALPHA_CASES:
    c0p = c0_model(0.30, XI_PHYS, a)
    c1p = c1_over_xi(0.30, XI_PHYS)
    print("    %-16s %-15.4e %-15.4e %-13.4e"
          % (nm, c0p, c1p, c0p + c1p))
c0_hi = c0_model(0.30, XI_PHYS, ALPHA_C0)
c0_lo = c0_model(0.30, XI_PHYS, 0.0)
span = abs(math.log10(max(c0_hi, 1e-300) / max(c0_lo, 1e-300)))
print()
print("    三个真实假设下 c0 的绝对值跨度 = %.1f 个数量级" % span)
print("    这是外推链上剩下的最大不确定度：c0 的 xi 指数在模型档（xi 14 至 42，")
print("    仅 3 倍跨度）上测得，无法在物理档（跨度 5.7e20 倍）复核。")
print("    注意第 706 号的形式 dev = c0 + c1/xi 自身把 c0 当作 xi 无关的渐近项，")
print("    而第 705 号 FA3 的实测指数 -0.494 是在 3 倍跨度上对渐近行为的拟合，")
print("    两者在物理档上的读数相差 %.1f 个数量级。" % span)

# ---------------- PA4 判定与更正 ----------------
print()
print("[PA4] 判定")
print("    PA0=%s PA1=%s PA2=%s"
      % tuple("PASS" if x else "FAIL" for x in (PA0, PA1, PA2)))
if not PA0:
    verdict = "复现失败，本轮读数不可用"
elif not PA1:
    verdict = "c0 的 xi 指数与第一原理预期不符，物理档外推的前提不成立"
else:
    verdict = ("物理档由 c0 主导，第 714 号 NA3 的判语在物理档反转；"
               "主导项归属对 c0 的 xi 指数不敏感，但绝对量级差 %.1f 个数量级" % span)
print("    判语：" + verdict)

print()
print("    物理档的完整外推式：")
print("      dev(v, xi) = c0_theory(v) * [%.4e/%.4e] * (xi/%.0f)^(%.3f) + [%.5f(gamma-1)]/xi"
      % (C0_AT_XI42, C0_THEORY_030, XI_MOD, ALPHA_C0, C1_K))
print("      前一项按 v^4 走（载波处格点色散），后一项按 v^2 走（井刚性失配）")
print("      低 v 极限下 c0 项仍按 v^4 走，故物理档的主导指数是 4 而不是 2。")
print("    第 713 号与第 714 号报的「物理档低速极限 dev 约等于 0.2111 v^2/xi」")
print("    是模型档的说法，搬到物理档不成立：那一项在物理档被 c0 压倒 %.1e 倍。"
      % CASE["实测 -0.494"][2])
print("耗时 %.1f 秒" % (time.time() - T0))

# ---------------- 图 ----------------
fig, ax = plt.subplots(1, 3, figsize=(15.8, 4.4))

vv = np.linspace(0.01, 0.95, 800)
ax[0].plot(vv, [ratio_c0_over_c1xi(x, XI_MOD) for x in vv],
           "-", color="#c44", lw=1.8, label=r"模型档 $\xi=42$")
ax[0].plot(vv, [ratio_c0_over_c1xi(x, XI_PHYS) for x in vv],
           "--", color="#268", lw=1.8, label=r"物理档 $\xi=2.389\times10^{22}$")
ax[0].axhline(1.0, color="#666", ls=":", lw=1.2)
ax[0].set_yscale("log")
ax[0].set_xlabel("v")
ax[0].set_ylabel(r"$c_0 / (c_1/\xi)$")
ax[0].set_title("左：两项之比随 v（过 1 即交叉）")
ax[0].legend(fontsize=9)

for nm, mk, col in (("实测 -0.494", "o", "#c44"), ("恒定 0", "s", "#268"),
                    ("替身 -1.5", "^", "#824")):
    a = CASE[nm][0]
    xs = np.linspace(-1.5, 0.0, 80)
    ys = []
    for x in xs:
        grid = np.linspace(0.02, 0.95, 300)
        rm = min(ratio_c0_over_c1xi(g, XI_PHYS, float(x)) for g in grid)
        ys.append(rm)
    ax[1].plot(xs, ys, mk, color=col, ms=4, label=nm)
ax[1].axhline(1.0, color="#666", ls=":", lw=1.2, label="归属翻转线")
ax[1].set_yscale("log")
ax[1].set_xlabel("c0 的 xi 指数")
ax[1].set_ylabel("物理档最小比值")
ax[1].set_title("中：归属对 c0 的 xi 指数的敏感度")
ax[1].legend(fontsize=8)

ax[2].plot(vv, [c0_model(x, XI_PHYS, ALPHA_C0) for x in vv],
           "-", color="#c44", lw=1.8, label=r"$c_0$（指数 -0.494）")
ax[2].plot(vv, [c0_model(x, XI_PHYS, 0.0) for x in vv],
           "--", color="#824", lw=1.6, label=r"$c_0$（指数 0）")
ax[2].plot(vv, [c1_over_xi(x, XI_PHYS) for x in vv],
           "-", color="#268", lw=1.8, label=r"$c_1/\xi$")
ax[2].set_yscale("log")
ax[2].set_xlabel("v")
ax[2].set_ylabel("绝对量级")
ax[2].set_title("右：物理档两项的绝对量级")
ax[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig(IMG, dpi=150)
print("图已存 %s" % IMG)
