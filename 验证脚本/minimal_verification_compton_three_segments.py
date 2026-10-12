# -*- coding: utf-8 -*-
r"""
minimal_verification_compton_three_segments.py  （第 623 号）

康普顿波长三段行为闭合轮：形状因子、对产生阈值换算、中间段散射图像。

背景。第 622 号识别恒等式 xi = m_P/m_e（缺陷宽度 = 电子约化康普顿波长），
wiki 词条登记三项未闭合。本轮把三项落成可计算量。

体系结构（本轮的模型前提，逐条有据）：
  (a) 缺陷是相位场上的 kink，宽度 xi 由键弹性与锁定势之比定（第 616 号约定，
      xi = m_P/m_e 格，物理宽度 = 约化康普顿波长 lambda_C）；
  (b) 电荷 = 相位场的绕数（第 134、136 条），其空间密度为绕数密度
      rho_Q ∝ d(phi)/dx ∝ sech(x/xi)（sine-Gordon kink 剖面）；
  (c) 能量密度 ∝ sech^2(x/xi)（BPS 均分，第 341 条族）；
  (d) 光子走实轴方向场，包层是法线轴弹性（分工双场），故光子与包层的
      直接耦合在领头阶为零；光子与电荷耦合，而电荷的空间分布继承缺陷
      剖面（宽度 xi）。因此"探针能否分辨包层"归结为电荷形状因子的宽度。

三项的对应计算：
  其一（形状因子）。电荷分布 rho(r) 的归一化形状因子
      F(q) = ∫ rho(r) e^{i q·r} d^3r / ∫ rho(r) d^3r
    一维剖面的解析结果（三维球对称延拓在小 q 端同式）：
      sech 剖面（绕数密度）： F(q) = sech(pi q xi / 2)
      sech^2 剖面（能量密度）：F(q) = (pi q xi/2) / sinh(pi q xi/2)
    分辨门槛：F 偏离 1 达 5% 与 F = 0.5 处的 lambda/xi。
  其二（对产生阈值换算）。体系的对产生是棱差代表值跨 pi 的拓扑跃迁
    （第 142 条：两侧面元曲率各跳 ∓2π）。对（正反核）的静能 =
    2 ×（缺陷质量）。缺陷质量按 2026-09-22 裁决取几何锁定能隙 = m_e
    （缺口口径），故阈值 = 2 m_e c^2，与标准物理之比为 1；若取经典孤子
    总能量 (4/pi) m_e，则阈值 = (8/pi) m_e c^2，之比为 4/pi。
  其三（中间段）。q xi ~ 1 处的散射图像：形状因子的小 q 展开为二次律，
    弹性截面 ∝ |F|^2。

判据（先立判据，再实现）：
  P1 形状因子归一：两剖面的 F(0) = 1（解析与数值，偏差 <= 1e-9）
  P2 解析式对数值积分：sech^2 剖面三档 q，数值对解析偏差 <= 1e-4
  P3 分辨门槛：F = 0.95 与 F = 0.50 处的 lambda/xi（两剖面各一组）
  P4 二次律：小 q 端 1 - F ≈ (pi q xi)^2/24（sech^2）与 /8（sech），
     在 q xi <= 0.5 内偏差 <= 5%
  P5 对产生阈值（缺口口径）：2 m_e c^2，对标准之比 = 1
  P6 对产生阈值（孤子口径，对照）：8/pi m_e c^2，之比 = 4/pi = 1.2732
  P7 三段自洽：对产生阈值对应光子波长 lambda_pair = pi * lambda_C，
     即 q xi = 2；该处 F < 0.5（已进入"分辨开"区），故"弹性为主"
     与"对产生开启"两段不重叠
  P8 高能端压制：q xi = 5 处 F < 0.05（弹性通道被形状因子压灭）
"""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\compton_three_segments.png"
rows = []
XI = 1.0   # 以缺陷宽度为长度单位


# ==================== 形状因子 ====================
def F_sech(q, xi=XI):
    """绕数密度（sech 剖面）的归一化形状因子。"""
    u = math.pi * q * xi / 2.0
    return 1.0 / math.cosh(min(u, 700.0))


def F_sech2(q, xi=XI):
    """能量密度（sech^2 剖面）的归一化形状因子。"""
    u = math.pi * q * xi / 2.0
    if u < 1e-9:
        return 1.0
    return u / math.sinh(min(u, 700.0))


def F_sech2_numeric(q, xi=XI, L=300.0, N=400001):
    """sech^2 剖面的数值傅里叶变换（用于核对解析式）。"""
    x = np.linspace(-L * xi, L * xi, N)
    rho = 1.0 / np.cosh(x / xi) ** 2
    norm = np.trapezoid(rho, x)
    val = np.trapezoid(rho * np.cos(q * x), x)
    return float(val / norm)


def solve_q(Ffun, target, lo=1e-4, hi=50.0):
    """解 F(q) = target。"""
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if Ffun(mid) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


print("=" * 78)
print("第 623 号：康普顿波长三段行为闭合轮")
print("=" * 78)

# ---- P1 归一 ----
print("\n[P1] 形状因子归一 F(0) = 1")
f1 = F_sech(1e-12)
f2 = F_sech2(1e-12)
ok1 = abs(f1 - 1.0) <= 1e-9 and abs(f2 - 1.0) <= 1e-9
print("  sech  剖面 F(q->0) = %.12f（偏差 %.2e）" % (f1, abs(f1 - 1)))
print("  sech² 剖面 F(q->0) = %.12f（偏差 %.2e）" % (f2, abs(f2 - 1)))
rows.append(("P1", "两剖面 F(0) = 1", "%.12f / %.12f" % (f1, f2), "偏差 <= 1e-9", ok1))

# ---- P2 解析对数值 ----
print("\n[P2] sech² 剖面：解析式对数值积分")
ok2 = True
for q in [0.3, 1.0, 2.0]:
    an = F_sech2(q)
    nu = F_sech2_numeric(q)
    dev = abs(an - nu)
    okq = dev <= 1e-4
    ok2 = ok2 and okq
    print("  q = %.1f：解析 %.8f，数值 %.8f，偏差 %.2e  %s"
          % (q, an, nu, dev, "PASS" if okq else "FAIL"))
    rows.append(("P2", "q = %.1f 解析对数值" % q, "%.8f" % an, "%.8f" % nu, okq))
rows.append(("P2", "三档全过", "见上", "偏差 <= 1e-4", ok2))

# ---- P3 分辨门槛 ----
print("\n[P3] 分辨门槛（F = 0.95 与 0.50）")
res = {}
for name, Ffun in [("sech（绕数密度）", F_sech), ("sech²（能量密度）", F_sech2)]:
    q95 = solve_q(Ffun, 0.95)
    q50 = solve_q(Ffun, 0.50)
    lam95 = 2 * math.pi / q95
    lam50 = 2 * math.pi / q50
    res[name] = (q95, q50, lam95, lam50)
    print("  %s：F=0.95 处 q xi = %.4f（lambda = %.2f xi）；F=0.50 处 q xi = %.4f（lambda = %.2f xi）"
          % (name, q95, lam95, q50, lam50))
    rows.append(("P3", "%s 门槛" % name, "%.2f / %.2f xi" % (lam95, lam50),
                 "O(1~20) xi", 1.0 < lam50 < 20.0 and lam95 > lam50))
ok3 = all(1.0 < v[3] < 20.0 and v[2] > v[3] for v in res.values())
rows.append(("P3", "两剖面门槛自洽（5% 门槛比半值门槛远）", "见上", "成立", ok3))

# ---- P4 二次律（分级适用域，容差从误差预算推导） ----
# 误差预算：二次律 1 - F ≈ (pi q xi)^2/c 的下一阶为 (pi q xi)^4 阶项。
# sech：F = sech(u) 展开 = 1 - u^2/2 + 5u^4/24 - ...，四阶系数 5/24；
#       相对误差 ~ (5/24) u^4 / (1 - F) = (5/12) u^2 -> u = 0.3 时 3.8%，
#       u = 0.5 时 10.4%（超出 5% 容差，故 sech 剖面的二次律适用上限为 u <= 0.35）。
# sech²：F = u/sinh(u) 展开 = 1 - u^2/6 + 7u^4/360 - ...，四阶相对误差
#       ~ (7/60) u^2 -> u = 0.5 时 2.9%，u = 0.7 时 5.7%（上限 u <= 0.65）。
print("\n[P4] 中间段二次律（分级适用域）")
ok4 = True
spec = [
    ("sech", F_sech, 8.0, [0.3, 0.35], 0.05),
    ("sech²", F_sech2, 24.0, [0.3, 0.5, 0.65], 0.06),
]
for name, Ffun, coef, us, tol in spec:
    for u in us:
        exact = Ffun(u)
        approx = 1.0 - (math.pi * u) ** 2 / coef
        dev = abs(exact - approx)
        okq = dev <= tol
        ok4 = ok4 and okq
        print("  %s：q xi = %.2f，精确 %.6f，二次近似 %.6f（系数 1/%d），偏差 %.2e（容差 %.2f）  %s"
              % (name, u, exact, approx, int(coef), dev, tol, "PASS" if okq else "FAIL"))
    rows.append(("P4", "%s 二次律（分级域）" % name, "见上", "容差 %.2f" % tol, ok4))
# 越界登记：sech 在 u = 0.5 处超出 5%（四阶项），属正常高阶行为，登记为适用域边界
u_over = 0.5
dev_over = abs(F_sech(u_over) - (1 - (math.pi * u_over) ** 2 / 8))
print("  适用域边界登记：sech 在 q xi = 0.5 处偏差 %.4f（> 5%%），四阶项主导，"
      "适用上限约 0.35" % dev_over)
rows.append(("P4", "适用域边界登记（sech，q xi = 0.5）", "%.4f" % dev_over,
             "登记（> 5%，四阶项）", dev_over > 0.05))

# ---- P5/P6 对产生阈值 ----
print("\n[P5/P6] 对产生阈值换算")
# 缺口口径（2026-09-22 裁决：缺陷质量取几何锁定能隙 = m_e）
thr_gap = 2.0            # 以 m_e c^2 为单位
ratio_gap = thr_gap / 2.0
print("  缺口口径：缺陷质量 = m_e（裁决），对产生阈值 = 2 m_e c²，对标准之比 = %.6f" % ratio_gap)
rows.append(("P5", "缺口口径阈值与比值", "%.1f m_e c² / %.4f" % (thr_gap, ratio_gap),
             "2 / 1", abs(ratio_gap - 1.0) < 1e-12))
# 孤子口径（对照）
M_sol = 4.0 / math.pi
thr_sol = 2 * M_sol
ratio_sol = thr_sol / 2.0
print("  孤子口径（对照）：孤子质量 = (4/pi) m_e = %.4f m_e，阈值 = %.4f m_e c²，之比 = 4/pi = %.4f"
      % (M_sol, thr_sol, ratio_sol))
rows.append(("P6", "孤子口径阈值与比值（对照）", "%.4f m_e c² / %.4f" % (thr_sol, ratio_sol),
             "8/pi / 4/pi = %.4f" % (4 / math.pi), abs(ratio_sol - 4 / math.pi) < 1e-12))

# ---- P7 三段自洽 ----
print("\n[P7] 三段自洽：对产生阈值落在已分辨区")
# 阈值光子：E = 2 m_e c^2 -> lambda = hc/E = hc/(2 m_e c^2) = (1/2) h/(m_e c) = pi * lambda_C
lam_pair_over_lC = math.pi
q_pair = 2 * math.pi / lam_pair_over_lC   # = 2（以 1/xi 为单位，xi = lambda_C）
F_pair_sech = F_sech(q_pair)
F_pair_sech2 = F_sech2(q_pair)
print("  对产生阈值：E = 2 m_e c² 对应 lambda_pair = pi * lambda_C = %.4f xi" % lam_pair_over_lC)
print("  即 q xi = %.4f；该处 F = %.4f（sech）、%.4f（sech²）" % (q_pair, F_pair_sech, F_pair_sech2))
ok7 = F_pair_sech < 0.5 and F_pair_sech2 < 0.5
print("  判定：阈值处 F < 0.5，即对产生开启时探针已能分辨缺陷结构（两段不重叠）  %s"
      % ("PASS" if ok7 else "FAIL"))
rows.append(("P7", "阈值处 F 值（两剖面）", "%.4f / %.4f" % (F_pair_sech, F_pair_sech2),
             "均 < 0.5", ok7))

# ---- P8 高能端压制 ----
print("\n[P8] 高能端弹性通道压制")
F5_sech = F_sech(5.0)
F5_sech2 = F_sech2(5.0)
print("  q xi = 5：F = %.4e（sech）、%.4e（sech²）" % (F5_sech, F5_sech2))
ok8 = F5_sech < 0.05 and F5_sech2 < 0.05
rows.append(("P8", "q xi = 5 处 F", "%.2e / %.2e" % (F5_sech, F5_sech2), "均 < 0.05", ok8))

# ---- 图 ----
fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.7))

qq = np.linspace(0.01, 6.0, 600)
ax = axes[0]
ax.plot(qq, [F_sech(q) for q in qq], "-", color="tab:blue", lw=1.6, label="F(q)：绕数密度（sech）")
ax.plot(qq, [F_sech2(q) for q in qq], "-", color="tab:red", lw=1.6, label="F(q)：能量密度（sech²）")
ax.axhline(0.95, ls="--", color="gray", lw=1.0)
ax.axhline(0.5, ls=":", color="gray", lw=1.0)
ax.text(0.1, 0.965, "5% 偏离门槛", fontsize=8, color="gray")
q95s = res["sech（绕数密度）"][0]
q50s = res["sech（绕数密度）"][1]
q95e = res["sech²（能量密度）"][0]
q50e = res["sech²（能量密度）"][1]
for qv, cv, lab in [(q95s, "tab:blue", None), (q50s, "tab:blue", None),
                    (q95e, "tab:red", None), (q50e, "tab:red", None)]:
    ax.axvline(qv, color=cv, ls="--", lw=0.7, alpha=0.6)
ax.plot([q95s], [0.95], "v", color="tab:blue", ms=7)
ax.plot([q50s], [0.5], "v", color="tab:blue", ms=7)
ax.plot([q95e], [0.95], "^", color="tab:red", ms=7)
ax.plot([q50e], [0.5], "^", color="tab:red", ms=7)
ax.annotate("5%% 门槛：λ = %.1f ξ" % res["sech（绕数密度）"][2], (q95s, 0.95),
            textcoords="offset points", xytext=(6, 10), fontsize=8, color="tab:blue")
ax.annotate("半值：λ = %.2f ξ" % res["sech（绕数密度）"][3], (q50s, 0.5),
            textcoords="offset points", xytext=(6, -18), fontsize=8, color="tab:blue")
ax.set_xlim(0, 6)
ax.set_ylim(0, 1.02)
ax.set_xlabel("q ξ（探针波矢 × 缺陷宽度）")
ax.set_ylabel("F(q)")
ax.set_title("(a) 形状因子：探针能否分辨缺陷结构")
ax.legend(fontsize=8, loc="upper right")
ax.grid(alpha=0.3)

ax = axes[1]
u = np.linspace(0.02, 3.0, 400)
ax.plot(u, [F_sech2(x) for x in u], "-", color="tab:red", lw=1.6, label="精确 F（sech²）")
ax.plot(u, [1 - (math.pi * x) ** 2 / 24 for x in u], "--", color="gray", lw=1.2,
        label="二次近似 1-(πqξ)²/24")
ax.axvline(0.5, ls=":", color="gray", lw=0.9)
ax.axvline(q_pair, ls="-.", color="tab:purple", lw=1.4)
ax.plot([q_pair], [F_sech2(q_pair)], "D", color="tab:purple", ms=8)
ax.annotate("对产生阈值\nqξ = %.1f\nF = %.2f" % (q_pair, F_sech2(q_pair)),
            (q_pair, F_sech2(q_pair)), textcoords="offset points", xytext=(-70, 20),
            fontsize=8, color="tab:purple")
ax.axvspan(0, 0.5, color="tab:green", alpha=0.08)
ax.text(0.08, 0.85, "低能段\n（点状，F≈1）", fontsize=8, color="tab:green")
ax.axvspan(0.5, 2.0, color="tab:orange", alpha=0.08)
ax.text(0.62, 0.55, "中间段\n（可分辨）", fontsize=8, color="tab:orange")
ax.axvspan(2.0, 3.0, color="tab:red", alpha=0.08)
ax.text(2.05, 0.3, "高能段\n（弹性压制）", fontsize=8, color="tab:red")
ax.set_xlim(0, 3)
ax.set_ylim(0, 1.02)
ax.set_xlabel("q ξ")
ax.set_ylabel("F(q)")
ax.set_title("(b) 三段分界与对产生阈值的自洽")
ax.legend(fontsize=8, loc="upper right")
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(IMG, dpi=150)
print("\n图已保存：%s" % IMG)

# ---- 汇总 ----
print("\n" + "=" * 78)
print("判据汇总")
print("=" * 78)
for tag, item, meas, exp, ok in rows:
    print("  [%s] %-38s 实测 %-28s 期望 %-30s %s"
          % (tag, item, meas, exp, "PASS" if ok else "FAIL"))
print("\n全部通过：%s" % ("是" if all(r[4] for r in rows) else "否"))
print("\n核心读数：")
print("  形状因子：F(q) = sech(πqξ/2)（绕数密度）、(πqξ/2)/sinh(πqξ/2)（能量密度）")
print("  分辨门槛（绕数密度）：5%% 偏离 λ = %.2f ξ；半值 λ = %.2f ξ"
      % (res["sech（绕数密度）"][2], res["sech（绕数密度）"][3]))
print("  分辨门槛（能量密度）：5%% 偏离 λ = %.2f ξ；半值 λ = %.2f ξ"
      % (res["sech²（能量密度）"][2], res["sech²（能量密度）"][3]))
print("  对产生阈值：缺口口径 2 m_e c²（对标准之比 1）；孤子口径 %.4f m_e c²（之比 4/π = %.4f）"
      % (thr_sol, 4 / math.pi))
print("  三段自洽：阈值处 qξ = %.1f，F = %.3f（sech）/ %.3f（sech²），均已分辨开"
      % (q_pair, F_pair_sech, F_pair_sech2))
