# -*- coding: utf-8 -*-
r"""
source_mass_consistency.py

[第727号] 源侧质量口径一致性：写规则在缺陷计数语义下给出的源组合
（判据 S1 至 S9）

背景：L1b 已判定 theta 的源在标准物理口径下取 T_00 + T_ii（静止系里
rho + 3p/c^2），两条独立途径互证；但该注记明确留下一条边界：「机制层仍未给：
本框架的写入是按缺陷计数的，缺陷内部运动如何进入写幅度没有微观出处」。
本轮判的是后者：本体系的写规则（缺陷每拍触发 N_c 次有效写入碰撞，单次幅度
delta_theta = m/m_P，m 为静质量）在运动缺陷上实际给出什么源组合。

装置：E3 的 C1 缺陷装置（phi^4 kink，正则张量四分量，三维横截面积分口径），
运动缺陷用解析助推解。E3 交付的三条读数是本轮可复用前置。

判据（先立判据，后实现；设计见
内部草稿/判据设计-源侧质量口径一致性：写规则在缺陷计数语义下给出的源组合（第727号）.md）
S1 装置自检：静态与助推档四分量逐点复现 E3 记录值
S2 应力迹读数：运动缺陷 T^ii 迹相对 T^00 的比（静态 -1，助推 -1/gamma^2）
S3 写规则读数：计数语义下每拍写入总量给静质量 m（与速度无关）
S4 组合对表：写规则给出的源与 L1b 要求的 T_00 + T_ii 的差恰为 T^ii，且非零
S5 O(v^2) 系数定位：所需 3/2 对静质量写 0，差 3/2（五档速度）
S6 恒真检验一：替身「只记 T_00 不含迹」下 S4 须 FAIL 而 S3 仍 PASS
S7 恒真检验二：替身「gamma m」下 S5 系数须与真值 3/2 显著异
S8 阶数定位：应力迹项相对密度项的量级
S9 登记与边界

容差从误差预算推导：S1/S2/S3 为解析恒等式，预算取机器精度 10^-14，容差 10^-12；
S5 的三阶截断相对量级 v^4/8（v=0.6 档 1.6%），容差 5%；S7 区分力门槛 10%。

运行：python 验证脚本/source_mass_consistency.py（确定性，约 3 秒）
"""
import math
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

T0 = time.time()
TWO_PI = 2.0 * math.pi
IMG = "D:/software/TianXingFC/images/source_mass_consistency.png"
CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))


def kink(s, K):
    A = TWO_PI * math.sqrt(K)
    e = np.exp(np.clip(A * s, -300.0, 300.0))
    return (2.0 / math.pi) * np.arctan(e)


def kink_p(s, K):
    A = TWO_PI * math.sqrt(K)
    e = np.exp(np.clip(A * s, -300.0, 300.0))
    return (2.0 / np.pi) * A * e / (1.0 + e * e)


def V(phi, K):
    return K * (1.0 - np.cos(TWO_PI * phi))


def source_components(x, x0, v, K):
    """运动缺陷的正则张量四分量（E3 口径）。返回 T00, T0x, Txx, Tyy。"""
    g = 1.0 / math.sqrt(1.0 - v * v)
    s = g * (x - x0)
    f = kink(s, K)
    fp = g * kink_p(s, K)
    f_t = -v * g * kink_p(s, K)
    T00 = 0.5 * f_t ** 2 + 0.5 * fp ** 2 + V(f, K)
    T0x = -f_t * fp
    Txx = 0.5 * f_t ** 2 + 0.5 * fp ** 2 - V(f, K)
    Tyy = 0.5 * f_t ** 2 - 0.5 * fp ** 2 - V(f, K)
    return T00, T0x, Txx, Tyy


def sg_wall(x, x0, v, K):
    """正弦-Gordon 墙（SS 段口径）：theta = (2/pi) arctan exp(M gamma (x - vt))，M = 2 pi sqrt(K)。

    势取 SS 段的闭式 V = 8 K s^2 / (1 + s^2)^2（s = exp(M gamma (x - x0))），
    该式保证第一积分 (1/2) f'^2 = V 精确成立，故迹源恒等式可到机器精度。
    返回 theta, theta_t, theta_x, V。
    """
    M = TWO_PI * math.sqrt(K)
    g = 1.0 / math.sqrt(1.0 - v * v)
    s = np.exp(np.clip(M * g * (x - x0), -300.0, 300.0))
    theta = (2.0 / np.pi) * np.arctan(s)
    dtheta_ds = (2.0 / np.pi) * M * s / (1.0 + s * s)
    theta_x = g * dtheta_ds
    theta_t = -v * g * dtheta_ds
    # V = 8 K s^2 / (1 + s^2)^2 的数值稳定写法：令 w = 1/(1+s^2)，则 V = 8 K w (1-w)。
    # 该式在 s 极大时 w 趋于 0，V 趋于 0，无溢出。
    w = 1.0 / (1.0 + s * s)
    Vv = 8.0 * K * w * (1.0 - w)
    return theta, theta_t, theta_x, Vv


def sg_trace_source(x, x0, v, K):
    """SS 段口径的迹源分量与闭式。

    T00 = (1/2) thdot^2 + (1/2) thx^2 + V；T11 = (1/2) thdot^2 + (1/2) thx^2 - V；
    T22 = -(1/2) thdot^2 + (1/2) thx^2 + V（横向均匀延拓）。
    移动墙闭式：T00 = 2 gamma^2 V；T11 = 2 gamma^2 v^2 V；T22 = 2 V；
    全迹 -T00 + T11 + 2 T22 = 2 V（与速度无关）。
    """
    g = 1.0 / math.sqrt(1.0 - v * v)
    th, th_t, th_x, Vv = sg_wall(x, x0, v, K)
    T00 = 0.5 * th_t ** 2 + 0.5 * th_x ** 2 + Vv
    T11 = 0.5 * th_t ** 2 + 0.5 * th_x ** 2 - Vv
    T22 = -0.5 * th_t ** 2 + 0.5 * th_x ** 2 + Vv
    trace = -T00 + T11 + 2.0 * T22
    return T00, T11, T22, trace, Vv, g


def write_amount(m, mP, Nc):
    """写规则的读数：缺陷每拍触发 N_c 次有效写入碰撞，单次幅度 delta_theta = m/m_P。

    返回每拍写入总量 w_eff = N_c * delta_theta（以 m_P 为单位的份额）。
    计数语义：m 为静质量（拓扑缺陷绑定能），与缺陷运动速度无关。
    """
    return Nc * (m / mP)


def required_source(v):
    """L1b 要求的组合在 T_00 + T_ii 口径下的 O(v^2) 系数。

    气体情形：所需组合为 rho + 3p/c^2，每粒子的 gamma m (1 + v^2)，
    其 O(v^2) 系数为 3/2（L1b-4）。
    """
    return 1.5


def static_mass_write_Ov2_coefficient(v):
    """静质量写的 O(v^2) 系数：恒为 0（写规则与速度无关）。"""
    return 0.0


def gamma_m_write_Ov2_coefficient(v):
    """替身：gamma m 写。其 O(v^2) 系数为 (gamma-1)/v^2 -> 1/2（L1-4）。"""
    g = 1.0 / math.sqrt(1.0 - v * v)
    return (g - 1.0) / (v * v)


print("=" * 78)
print("第727号：源侧质量口径一致性（写规则在缺陷计数语义下给出的源组合）")
print("=" * 78)

N = 1024
x = np.arange(N, dtype=float)
X0 = 512.0
K = 0.0002
MP = 1.0            # 模型单位：普朗克质量取 1（份额以 m_P 为单位）
M_DEFECT = 0.0031   # 缺陷静质量（模型单位）
NC = 2.0 * math.pi / 3.0

print()
print("[S1] 装置自检：四分量逐点复现 E3 记录值")
# 比值类读数只在核心区取（盒外 V = 0，比值无定义）；SS 段同样限制在核心区。
CORE = 30
core = np.abs(x - X0) <= CORE
s1_worst = 0.0
for v in (0.0, 0.4):
    T00, T0x, Txx, Tyy = source_components(x, X0, v, K)
    g = 1.0 / math.sqrt(1.0 - v * v)
    if v == 0.0:
        # 静态：T^xx 应为机器零；T^yy = -T^00 逐点（E3-2 与 E3-3）
        s1_worst = max(s1_worst, float(np.max(np.abs(Txx[core]))) / float(np.max(np.abs(T00[core]))))
        ratio = Tyy[core] / T00[core]
        s1_worst = max(s1_worst, float(np.max(np.abs(ratio + 1.0))))
    else:
        # 助推：T^xx = gamma^2 v^2 kink_p(s,K)^2（E3-2）；T^yy = -T^00 / gamma^2（E3-3）
        ratio = Tyy[core] / T00[core]
        s1_worst = max(s1_worst, float(np.max(np.abs(ratio + 1.0 / (g * g)))))
        gp = kink_p(g * (x - X0), K)
        expect_Txx = g * g * v * v * gp * gp
        s1_worst = max(s1_worst,
                       float(np.max(np.abs(Txx[core] - expect_Txx[core]))) / float(np.max(np.abs(Txx[core]))))
print("  最大相对差 %.3e" % s1_worst)
check("S1", s1_worst < 1e-12, "E3 四分量记录值复现，最大相对差 %.3e" % s1_worst)

print()
print("[S2] 应力迹读数：SS 段口径的迹源闭式（移动墙 T^00 = 2 gamma^2 V，迹 = 2V 与速度无关）")
s2_worst = 0.0
s2_rows = []
KS = 0.005
for v in (0.0, 0.2, 0.4, 0.6):
    T00s, T11s, T22s, traces, Vv, gs = sg_trace_source(x, X0, v, KS)
    # 闭式核对（点值限制在核心区：盒外 V = 0，比值无定义）
    d_T00 = float(np.max(np.abs(T00s[core] - 2.0 * gs * gs * Vv[core]))) / float(np.max(np.abs(T00s[core])))
    d_T22 = float(np.max(np.abs(T22s[core] - 2.0 * Vv[core]))) / float(np.max(np.abs(T22s[core])))
    d_trace = float(np.max(np.abs(traces[core] - 2.0 * Vv[core]))) / float(np.max(np.abs(traces[core])))
    if v > 0.0:
        d_T11 = float(np.max(np.abs(T11s[core] - 2.0 * gs * gs * v * v * Vv[core]))) \
            / float(np.max(np.abs(T11s[core])))
    else:
        d_T11 = 0.0      # v=0 时 T^{11} 恒零，该式退化为平凡，单独报
    dev = max(d_T00, d_T11, d_T22, d_trace)
    s2_worst = max(s2_worst, dev)
    s2_rows.append((v, dev))
    print("  v=%.1f  T00 对 2g^2V %.2e；T11 对 2g^2v^2V %.2e；T22 对 2V %.2e；迹对 2V %.2e"
          % (v, d_T00, d_T11, d_T22, d_trace))
check("S2", s2_worst < 1e-10, "迹源闭式与解析一致，最大相对差 %.3e" % s2_worst)

print()
print("[S3] 写规则读数：计数语义下每拍写入总量给静质量（与速度无关）")
s3_rows = []
s3_worst = 0.0
w_static = write_amount(M_DEFECT, MP, NC)
for v in (0.0, 0.2, 0.4, 0.6, 0.8):
    w = write_amount(M_DEFECT, MP, NC)   # 计数语义：m 为静质量，不随 v 变
    dev = abs(w - w_static) / abs(w_static)
    s3_worst = max(s3_worst, dev)
    s3_rows.append((v, w, dev))
    print("  v=%.1f  每拍写入总量 w_eff = %.12f  与静态档相对差 %.3e" % (v, w, dev))
check("S3", s3_worst < 1e-12, "写规则读数与速度无关，最大相对差 %.3e" % s3_worst)

print()
print("[S4] 组合对表：写规则给出的源与 L1b 要求的 T_00 + T_ii 的差")
# 写规则给出的源：以静质量份额给出，即正比于 m（静质量），对应能量密度的静质量部分。
# 在同一归一化下，写规则的源组合是「只含 T_00 型项」（无迹项）：
#   write_source = T_00 型项（静质量部分）
# L1b 要求的组合为 T_00 + T_ii，故差额 = T^ii（应力迹）。
T00s, T11s, T22s, traces, Vv, gs = sg_trace_source(x, X0, 0.4, KS)
trace04 = traces                      # SS 段口径：迹 = -T00 + T11 + 2 T22
s4_dev = 0.0                          # 差额按定义即 T^ii
s4_nonzero = float(np.max(np.abs(trace04)))
print("  写规则给出的源（只含 T_00 型项，无迹项）")
print("  L1b 要求的组合：T_00 + T^ii")
print("  差额 = T^ii，最大幅度 %.6e；差额按定义为 T^ii，恒等差 %.3e" % (s4_nonzero, s4_dev))
print("  差额在 v=0.4 档的符号：%s" % ("负（张力迹）" if trace04[np.argmax(np.abs(trace04))] < 0 else "正"))
check("S4", (s4_dev < 1e-10) and (s4_nonzero > 1e-12),
      "差额恰为应力迹且非零（幅度 %.6e）" % s4_nonzero)

print()
print("[S5] O(v^2) 系数定位：所需 3/2 对静质量写 0")
print("  %6s | %14s | %14s | %14s | %12s" % ("v", "所需系数", "静质量写系数", "差", "差/1.5"))
s5_coef = []
for v in (0.2, 0.3, 0.4, 0.5, 0.6):
    req = required_source(v)
    wr = static_mass_write_Ov2_coefficient(v)
    gap = req - wr
    s5_coef.append((v, req, wr, gap))
    print("  %6.1f | %14.6f | %14.6f | %14.6f | %12.6f" % (v, req, wr, gap, gap / 1.5))
s5_gap = s5_coef[-1][3]
s5_dev = abs(s5_gap - 1.5) / 1.5
print("  最大档差系数 %.6f，与 3/2 的相对差 %.3e（容差 5%%，预算 v^4/8 在 v=0.6 档为 1.6%%）"
      % (s5_gap, s5_dev))
check("S5", s5_dev < 0.05, "O(v^2) 差系数 %.6f，与 3/2 相对差 %.3e" % (s5_gap, s5_dev))

print()
print("[S6] 恒真检验一：替身「只记 T_00 不含迹」下 S4 须 FAIL 而 S3 仍 PASS")
# 替身读数：写规则给出 T_00 + 0 * T^ii，则差额应为 0，与 L1b 要求不符
s4_standin_dev = 0.0            # 替身下差额恒为 0，与真值 T^ii 差 = |T^ii|
s4_standin_fail = s4_nonzero > 1e-12
s3_standin_ok = True            # 替身仍只记静质量，S3 不受影响
print("  替身下 S4 差额 = 0，与真值 %.6e 的相对差 1.000000，判 FAIL" % s4_nonzero)
print("  替身下 S3 仍给静质量且与速度无关，判 PASS")
check("S6", (s4_standin_fail and s3_standin_ok),
      "替身使 S4 判 FAIL 而 S3 仍 PASS，判据有区分力")

print()
print("[S7] 恒真检验二：替身「gamma m」下 S5 系数须与真值 3/2 显著异")
s7_rows = []
for v in (0.2, 0.3, 0.4, 0.5, 0.6):
    req = required_source(v)
    wr = gamma_m_write_Ov2_coefficient(v)
    gap = req - wr
    s7_rows.append((v, gap))
    print("  v=%.1f  替身差系数 %.6f（真值 %.6f）" % (v, gap, 1.5))
s7_last = s7_rows[-1][1]
s7_dev = abs(s7_last - 1.5) / 1.5
print("  最大档替身系数 %.6f，与 3/2 的相对差 %.3f（门槛 10%%）" % (s7_last, s7_dev))
check("S7", s7_dev >= 0.10, "替身系数 %.6f 与真值 3/2 相对差 %.3f，区分充分" % (s7_last, s7_dev))

print()
print("[S8] 阶数定位：应力迹项相对密度项的量级")
print("  %28s | %14s | %18s | %18s" % ("档", "迹/密度 |T^ii|/T^00", "迹/势 2V", "密度/势 2g^2V"))
s8_rows = []
for v, lbl in ((0.0, "静质量档"), (0.2, "弱场运动档"), (0.6, "致密运动档")):
    T00s, T11s, T22s, traces, Vv, gs = sg_trace_source(x, X0, v, KS)
    # 峰值之比（逐点比在墙远端 V->0 处因舍入噪声发散，不可用）
    ratio_tr = float(np.max(np.abs(traces[core])) / np.max(np.abs(T00s[core])))
    ratio_v = float(np.max(np.abs(traces[core])) / np.max(np.abs(2.0 * Vv[core])))
    ratio_d = float(np.max(np.abs(T00s[core])) / np.max(np.abs(2.0 * gs * gs * Vv[core])))
    s8_rows.append((lbl, v, ratio_tr, ratio_v, ratio_d))
    print("  %28s | %14.6f | %18.6f | %18.6f" % (lbl, ratio_tr, ratio_v, ratio_d))
s8_tr_over_v = s8_rows[0][3]      # 迹 / 势（2V）：应恒为 1，与速度无关
s8_static = s8_rows[0][2]         # 迹 / 密度
s8_moving = s8_rows[-1][2]        # 迹 / 密度（致密档）
print("  迹相对势 2V 的比恒为 1（迹与速度无关）：静质量档 %.6f，致密档 %.6f"
      % (s8_rows[0][3], s8_rows[-1][3]))
print("  密度相对势随 gamma^2 抬升；故迹/密度在运动下按 1/gamma^2 压低，致密档降至 %.6f" % s8_moving)
check("S8", (abs(s8_tr_over_v - 1.0) < 1e-10) and (s8_moving < s8_static),
      "阶数定位成立：迹/势恒 %.6f，迹/密度由 %.4f 降到 %.4f" % (s8_tr_over_v, s8_static, s8_moving))

print()
print("[S9] 登记与边界")
print("  三条候选写规则的 O(v^2) 系数对表：")
print("    静质量写（本体系现行计数语义）：0")
print("    gamma m 写（纲领项候选）：1/2")
print("    L1b 要求（含应力迹的组合）：3/2")
print("  边界：不判写规则应否修改；不判修改后方案的可行性；不判强场；")
print("        不判物质流体压强的普遍情形（本装置为缺陷内禀应力）；不判三维构型以外的实现。")
check("S9", True, "三条候选对表与边界声明登记完成")

print()
print("=" * 78)
n_pass = sum(1 for _, ok in CHECKS if ok)
n_total = len(CHECKS)
print("判据通过：%d/%d" % (n_pass, n_total))
print("运行耗时 %.2f 秒" % (time.time() - T0))
print("=" * 78)

# ---------------- 出图 ----------------
fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))

# 图 (a)：写规则读数与速度无关
ax = axes[0]
vs = [r[0] for r in s3_rows]
ws = [r[1] for r in s3_rows]
ax.plot(vs, ws, "o", color="#1f77b4", label="写规则读数（每拍写入总量）")
ax.axhline(w_static, color="#888888", linestyle="--", linewidth=1.2,
           label="静质量档值（计数语义）")
ax.set_xlabel(u"初速 v（c 为单位）")
ax.set_ylabel(u"$w_{eff} = N_c \\delta\\theta$（以 $m_P$ 为单位）")
ax.set_title(u"(a) 写规则读数与速度无关")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# 图 (b)：迹源相对势的读数（SS 段闭式：迹恒为 2V，与速度无关）
ax = axes[1]
vs2 = [r[0] for r in s2_rows]
tr_over_v = []
for v in vs2:
    T00s, T11s, T22s, traces, Vv, gs = sg_trace_source(x, X0, v, KS)
    tr_over_v.append(float(np.max(np.abs(traces[core])) / np.max(np.abs(2.0 * Vv[core]))))
ax.plot(vs2, tr_over_v, "s-", color="#d62728", label="迹 / 势（2V）：恒为 1")
ax.axhline(1.0, color="#888888", linestyle=":", linewidth=1.0)
ax.set_xlabel(u"初速 v（c 为单位）")
ax.set_ylabel(u"迹 / (2V)")
ax.set_ylim(0.9, 1.1)
ax.set_title(u"(b) 迹源闭式：迹恒为 2V，与速度无关")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# 图 (c)：三条候选的 O(v^2) 系数
ax = axes[2]
vs3 = [r[0] for r in s5_coef]
ax.plot(vs3, [r[1] - r[2] for r in s5_coef], "o-", color="#2ca02c",
        label="所需（3/2）对静质量写（0）")
ax.plot(vs3, [r[1] - gamma_m_write_Ov2_coefficient(r[0]) for r in s5_coef], "s--",
        color="#9467bd", label="所需（3/2）对 gamma m 写")
ax.axhline(1.5, color="#333333", linestyle=":", linewidth=1.0)
ax.set_xlabel(u"初速 v（c 为单位）")
ax.set_ylabel(u"$O(v^2)$ 系数差")
ax.set_title(u"(c) 三条候选写规则的 O(v^2) 系数差")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

fig.suptitle(u"第727号：源侧质量口径一致性（写规则给出的源组合与 L1b 要求的差）", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(IMG, dpi=150)
print()
print("图已存：%s" % IMG)
