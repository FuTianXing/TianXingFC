# -*- coding: utf-8 -*-
"""BH4a：支撑失效的最大质量零阶判据（相空间下界 + 相对论化软化 + 引力竞争）。

体系输入（三项，出处见验证注记）：
  A 引力势阱：E_g = -(3/5) G M^2 / R（碰撞写入的静态势阱，定案）。
  B 缺陷动能：波包压缩下界 k >= (N/V)^{1/3}（波性侧傅里叶共轭），能动量关系
    E = sqrt(m^2 c^4 + hbar^2 c^2 k^2)（洛伦兹结构定理化；kink 色散期望 P1-33 待验证）。
  C 传播上限：v <= c（每拍一格几何恒等式），使高动量端 E -> hbar c k（软化）。

无量纲化（每颗粒能量，x = R/R_N，R_N = (hbar/mc)(3N/4pi)^{1/3}）：
  f(x) = sqrt(1 + x^-2) - (3/5) w / x，w = (4pi/3)^{1/3} N^{2/3} (m/m_P)^2。
  解析：极小存在 iff a = 0.6 w < 1，极小位置 x* = sqrt(1/a^2 - 1)。
  临界 N_c^{2/3} = (5/3)(3/4pi)^{1/3} (m_P/m)^2，M_c = C_lo * m_P^3/m^2。

判据（先立判据，再实现；均可证伪）：
  B1a 极小位置：a = 0.5 时数值最小点与解析 x* = sqrt(3) 偏差 <= 2%，极小值复核
       f* = sqrt(3)/2（<= 1%）。
  B1b 极小消失（超临界）：a = 2 时网格全域 f' > 0（无局部极小；压缩端无下界）。
  B1c 临界平坦：a = 1 时 f' > 0 全域且 f(x->0) -> 0+（极小退化到压缩端）。
  B2 量级对照：临界质量带（下界口径 C_lo；极端相对论费米气参考口径 C_F）
       中子星与 TOV（2.0-2.5 M_sun）、白矮星与钱德拉塞卡（1.44 M_sun）在 0.3-3 倍内。
  B3 无相对论对照：非相对论动能下（f_NR = 1/(2x^2) - 0.6w/x）N = 100 N_c 仍有有限极小
       （解析 x* = 1/(0.6w)，偏差 <= 1%）；相对论模型同 w 无局部极小。
       => 传播上限（相对论化软化）是"最大质量"的必要输入。
  B4 温度侧不活跃：两星体每颗粒引力束缚能与 E_P 之比 < 1e-15（无序度上限在恒星尺度不活跃）。
  B5 Lane-Emden n = 3 对照：数值解 omega_3 与文献 2.01824 偏差 <= 0.1%；标准组装
       M_Ch = (omega_3 sqrt(3 pi)/2) m_P^3/(mu_e m_u)^2 与名义值 1.44 M_sun 偏差 <= 3%
       （展示零阶判据与标准精密解同构）。

确定性（无随机数），运行数秒。图存 D:/software/TianXingFC/images/bh4a_max_mass.png。
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

G     = 6.67430e-11
c     = 299792458.0
hbar  = 1.054571817e-34
m_n   = 1.67492749804e-27
m_u   = 1.66053906660e-27
M_sun = 1.98840987e30
EP    = np.sqrt(hbar * c**5 / G)
mP    = np.sqrt(hbar * c / G)

CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))


def f_rel(x, w):
    return np.sqrt(1.0 + 1.0 / x**2) - 0.6 * w / x


def f_nr(x, w):
    return 0.5 / x**2 - 0.6 * w / x


def golden_min(func, a, b, iters=300):
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    c1 = b - gr * (b - a)
    c2 = a + gr * (b - a)
    f1 = func(c1)
    f2 = func(c2)
    for _ in range(iters):
        if f1 < f2:
            b = c2
            c2 = c1
            f2 = f1
            c1 = b - gr * (b - a)
            f1 = func(c1)
        else:
            a = c1
            c1 = c2
            f1 = f2
            c2 = a + gr * (b - a)
            f2 = func(c2)
    return 0.5 * (a + b)


print("==== BH4a 支撑失效的最大质量零阶判据 ====")
print("常数：m_P = %.6e kg, E_P = %.6e J, M_sun = %.6e kg" % (mP, EP, M_sun))
print("")

# ---------- 解析系数 ----------
w_c = 5.0 / 3.0
C_lo = ((5.0 / 3.0) * (3.0 / (4.0 * np.pi)) ** (1.0 / 3.0)) ** 1.5
Ck_F = 0.75 * (9.0 * np.pi / 4.0) ** (1.0 / 3.0)          # 极端相对论费米气动能系数
C_F = (Ck_F / 0.6) ** 1.5
print("临界条件 a = 0.6 w = 1（a = 0.6 (4pi/3)^{1/3} N^{2/3} (m/m_P)^2）")
print("下界口径系数 C_lo = %.5f（k = n^{1/3}）" % C_lo)
print("费米气参考系数 C_F = %.5f（动能系数 %.4f 折算）" % (C_F, Ck_F))
print("")

# ---------- B1a 极小位置 ----------
print("[B1a] a = 0.5 极小位置与极小值")
w1 = 0.5 / 0.6
x_num = golden_min(lambda x: f_rel(x, w1), 0.5, 5.0)
x_ana = np.sqrt(3.0)
dev = abs(x_num - x_ana) / x_ana
f_min_num = f_rel(x_num, w1)
f_min_ana = np.sqrt(3.0) / 2.0
dev_f = abs(f_min_num - f_min_ana) / f_min_ana
check("B1a-位置", dev <= 0.02, "x_num = %.6f, x_ana = %.6f, 偏差 %.4f%%" % (x_num, x_ana, dev * 100))
check("B1a-极小值", dev_f <= 0.01, "f_num = %.6f, f_ana = %.6f, 偏差 %.4f%%" % (f_min_num, f_min_ana, dev_f * 100))

# ---------- B1b / B1c 单调性与临界平坦 ----------
xs = np.logspace(-1.3, 1.7, 3001)   # 0.05 至 50

w2 = 2.0 / 0.6
fs2 = f_rel(xs, w2)
fp2 = np.gradient(fs2, xs)
check("B1b-无局部极小", np.all(fp2 > 0), "min f' = %.4f（a = 2，全域 f' > 0）" % fp2.min())

w3 = 1.0 / 0.6
fs3 = f_rel(xs, w3)
fp3 = np.gradient(fs3, xs)
x_small = 0.01
f_small = f_rel(x_small, w3)
check("B1c-临界单调", np.all(fp3 > 0), "min f' = %.4f（a = 1，全域 f' > 0）" % fp3.min())
check("B1c-退化到压缩端", 0.0 < f_small < 0.01, "f(0.01) = %.6f（趋 0+）" % f_small)

# ---------- B2 量级对照 ----------
print("")
print("[B2] 临界质量量级与对照")
mP3_over_mn2 = mP**3 / m_n**2
mP3_over_mu2 = mP**3 / m_u**2
M_ns_lo = C_lo * mP3_over_mn2 / M_sun
M_ns_F = C_F * mP3_over_mn2 / M_sun
M_wd_lo = C_lo * mP3_over_mu2 / 4.0 / M_sun
M_wd_F = C_F * mP3_over_mu2 / 4.0 / M_sun
print("  m_P^3/m_n^2 = %.6e kg = %.4f M_sun" % (mP3_over_mn2, mP3_over_mn2 / M_sun))
print("  m_P^3/m_u^2 = %.6e kg = %.4f M_sun" % (mP3_over_mu2, mP3_over_mu2 / M_sun))
print("  中子星带：%.4f 至 %.4f M_sun（对照 TOV 2.0 至 2.5）" % (M_ns_lo, M_ns_F))
print("  白矮星带：%.4f 至 %.4f M_sun（对照钱德拉塞卡 1.44）" % (M_wd_lo, M_wd_F))

print("  带宽度（费米口径/下界口径）= %.2f 倍" % (M_ns_F / M_ns_lo))
ok_ns = (M_ns_lo <= 2.0) and (2.5 <= M_ns_F)
ok_wd = (M_wd_lo <= 1.44) and (1.44 <= M_wd_F)
check("B2-中子星", ok_ns,
      "TOV 2.0 至 2.5 落在带内；对照/下界 = %.3f, 对照/费米 = %.3f"
      % (2.2 / M_ns_lo, 2.2 / M_ns_F))
check("B2-白矮星", ok_wd,
      "钱德拉塞卡 1.44 落在带内；对照/下界 = %.3f, 对照/费米 = %.3f"
      % (1.44 / M_wd_lo, 1.44 / M_wd_F))

# ---------- B3 无相对论对照 ----------
print("")
print("[B3] 无相对论化对照（N = 100 N_c）")
w100 = w_c * 100.0 ** (2.0 / 3.0)
x_nr_num = golden_min(lambda x: f_nr(x, w100), 0.005, 1.0)
x_nr_ana = 1.0 / (0.6 * w100)
dev_nr = abs(x_nr_num - x_nr_ana) / x_nr_ana
check("B3-非相对论仍有极小", dev_nr <= 0.01,
      "x*_num = %.6f, x*_ana = %.6f, 偏差 %.4f%%" % (x_nr_num, x_nr_ana, dev_nr * 100))
fs100 = f_rel(xs, w100)
fp100 = np.gradient(fs100, xs)
check("B3-相对论同 w 无局部极小", np.all(fp100 > 0), "min f' = %.4f（单调下行至 -inf）" % fp100.min())

# ---------- B4 温度侧不活跃 ----------
print("")
print("[B4] 温度（无序度）上限的活跃性")
E_ns = G * m_n * 2.0 * M_sun / 12e3
E_wd = G * m_u * 1.44 * M_sun / 5e6
r_ns = E_ns / EP
r_wd = E_wd / EP
print("  中子星（2 M_sun, 12 km）每核子束缚能 %.4e J，比 E_P：%.3e" % (E_ns, r_ns))
print("  白矮星（1.44 M_sun, 5000 km）每核子束缚能 %.4e J，比 E_P：%.3e" % (E_wd, r_wd))
check("B4-不活跃", r_ns < 1e-15 and r_wd < 1e-15, "两比值均 < 1e-15（温度上限不进入判据）")

# ---------- B5 Lane-Emden n = 3 ----------
print("")
print("[B5] Lane-Emden n = 3 与钱德拉塞卡对照")


def lane_emden_n3(h=1e-4, xi_max=15.0):
    xi0 = 1e-4
    th = 1.0 - xi0**2 / 6.0 + xi0**4 / 40.0
    dth = -xi0 / 3.0 + xi0**3 / 10.0
    xi = xi0
    while xi < xi_max:
        def deriv(x_, t_, d_):
            return d_, -2.0 * d_ / x_ - t_**3
        k1 = deriv(xi, th, dth)
        k2 = deriv(xi + h / 2, th + h * k1[0] / 2, dth + h * k1[1] / 2)
        k3 = deriv(xi + h / 2, th + h * k2[0] / 2, dth + h * k2[1] / 2)
        k4 = deriv(xi + h, th + h * k3[0], dth + h * k3[1])
        th_n = th + h * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6.0
        dth_n = dth + h * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6.0
        if th_n <= 0.0:
            frac = th / (th - th_n)
            xi1 = xi + frac * h
            dth1 = dth + frac * (dth_n - dth)
            return xi1, dth1
        xi, th, dth = xi + h, th_n, dth_n
    raise RuntimeError("no zero crossing")


xi1, dth1 = lane_emden_n3()
omega3 = -xi1**2 * dth1
omega3_ref = 2.018236
dev_o = abs(omega3 - omega3_ref) / omega3_ref
M_Ch = omega3 * np.sqrt(3.0 * np.pi) / 2.0 * mP**3 / (2.0 * m_u) ** 2 / M_sun
M_Ch_ref = 1.44
dev_M = abs(M_Ch - M_Ch_ref) / M_Ch_ref
print("  xi_1 = %.6f, omega_3 = %.6f（文献 2.018236）" % (xi1, omega3))
print("  M_Ch = %.4f M_sun（名义 1.44，偏差 %.2f%%）" % (M_Ch, dev_M * 100))
check("B5-omega3", dev_o <= 1e-3, "偏差 %.5f%%" % (dev_o * 100))
check("B5-M_Ch", dev_M <= 0.03, "偏差 %.3f%%" % (dev_M * 100))

# ---------- 图 ----------
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
fig, axs = plt.subplots(1, 2, figsize=(11.2, 4.4))

ax = axs[0]
x_plot = np.logspace(np.log10(0.3), np.log10(50), 800)
for a_val, style, lab in [
    (0.5, "-", "a = 0.5（束缚：极小存在）"),
    (1.0, "--", "a = 1（临界：极小退化）"),
    (2.0, "-.", "a = 2（坍缩：无极小）"),
]:
    ax.plot(x_plot, f_rel(x_plot, a_val / 0.6), style, lw=1.8, label=lab)
ax.plot([x_ana], [f_min_ana], "ko", ms=5)
ax.annotate("极小 x* = √3", xy=(x_ana, f_min_ana), xytext=(2.0, 0.35),
            arrowprops=dict(arrowstyle="->", lw=0.9), fontsize=8)
ax.axhline(0.0, color="gray", lw=0.6)
ax.set_xscale("log")
ax.set_ylim(-3.3, 2.2)
ax.set_xlabel("x = R / R_N（R_N 为量子半径标度）")
ax.set_ylabel("E / (N m c²)")
ax.set_title("每颗粒能量泛函：极小存在性随 a 的转变", fontsize=9)
ax.legend(fontsize=7.5)

ax = axs[1]
xs_cat = [0, 1]
ax.bar(0, M_ns_F - M_ns_lo, bottom=M_ns_lo, width=0.35, color="lightsteelblue",
       edgecolor="black", lw=0.8, label="零阶判据带（C = 1.05 至 3.72）")
ax.bar(1, M_wd_F - M_wd_lo, bottom=M_wd_lo, width=0.35, color="lightsteelblue",
       edgecolor="black", lw=0.8)
ax.axhspan(2.0, 2.5, color="orange", alpha=0.25)
ax.axhline(1.44, color="red", ls="--", lw=1.2)
ax.plot([0.5], [2.25], "s", color="orange", ms=6)
ax.plot([1.5], [1.44], "s", color="red", ms=6)
ax.text(0.62, 2.3, "TOV 2.0 至 2.5", fontsize=8, color="darkorange")
ax.text(1.32, 1.15, "钱德拉塞卡 1.44", fontsize=8, color="red")
ax.set_xticks(xs_cat)
ax.set_xticklabels(["中子星\n(m = m_n)", "白矮星\n(μe = 2)"])
ax.set_yscale("log")
ax.set_ylim(0.2, 12)
ax.set_ylabel("临界质量（M_sun）")
ax.set_title("临界质量带与实测上限对照", fontsize=9)
ax.legend(fontsize=7.5, loc="upper right")
ax.grid(axis="y", ls=":", alpha=0.5)

fig.tight_layout()
fig.savefig("D:/software/TianXingFC/images/bh4a_max_mass.png", dpi=150)
print("")
print("图已保存：images/bh4a_max_mass.png")

# ---------- 汇总 ----------
n_fail = sum(1 for _, ok in CHECKS if not ok)
print("")
print("==== 判据汇总：%d 项，%d PASS，%d FAIL ====" % (len(CHECKS), len(CHECKS) - n_fail, n_fail))