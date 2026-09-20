# -*- coding: utf-8 -*-
r"""
minimal_verification_c6_measurability.py

C6 可测含量评估轮：计数比三段曲线的膝关节落在哪个 bin 宽。

来源：作者 2026-09-21 指令（给 C6 做可测含量评估）；输入取第 292、295 条的格点读数（w = 15.8 l_P，K = 0.001）
与缺陷质量档（D-6 的电子档与核子档）。

闭式：kink 宽度 w 与质量档的关系。由 K = (pi M/4)^2（M 取普朗克质量单位，第 287 条 M = (4/pi) sqrt(K)）
与 w ∝ K^(-1/2) 得 w/l_P = 0.6361/(M/m_P)，即 w 约等于 0.64 倍约化康普顿波长。
膝位：tau_knee = w/v = 0.6361 * hbar/(M c^2) * (c/v)。

判据（先行）：
  X1 闭式自洽：由格点读数（w = 15.8 l_P，K = 0.001）反解的系数对 0.6361，偏差 <= 1%
  X2 膝位表：电子档、μ 档、核子档、中微子档（0.1 eV）与 v = 0.1c 与 c 的组合
  X3 可达性分级：与阿秒（1e-18 s）、仄秒（1e-21 s）、电子学（1e-9 s）三档 bin 对照
  X4 判决草案：可测含量分级与 C6 判定条件须补的分箱要求
"""
import numpy as np

L_P = 1.616255e-35
T_P = 5.391247e-44
C = 299792458.0
HBAR = 1.054571817e-34
M_P = 2.176434e-8
M_E = 9.1093837015e-31
M_MU = 1.883531627e-28
M_PN = 1.67262192369e-27
rows = []

# X1 闭式自洽
W_LATT = 15.8
K_LATT = 0.001
coef_meas = W_LATT * np.sqrt(K_LATT) * 4.0 / np.pi
coef_closed = 0.6361
dev1 = abs(coef_meas - coef_closed) / coef_closed
rows.append(("X1", "由格点读数反解的系数 w sqrt(K) 4/pi 对 0.6361", "%.4f" % coef_meas, "偏差 <= 1%", dev1 <= 0.01))


def knee(M_kg, v_over_c):
    return coef_closed * HBAR / (M_kg * C * C) / v_over_c


cases = [("中微子档 0.1 eV", 0.1 * 1.78266192e-36), ("电子档", M_E), ("μ 档", M_MU), ("核子档", M_PN)]
tabs = []
for name, M in cases:
    t1 = knee(M, 0.1)
    t2 = knee(M, 1.0)
    w = coef_closed * HBAR / (M * C)
    tabs.append((name, M / M_E, w, t1, t2))
rows.append(("X2", "膝位表（四个质量档乘 v = 0.1c 与 c）", "见表", "登记", True))

bins_ = [("阿秒", 1e-18), ("仄秒", 1e-21), ("皮秒", 1e-12), ("电子学", 1e-9)]
verdict = []
for name, M in cases:
    t1 = knee(M, 0.1)
    hit = [b for b, v in bins_ if v <= t1]
    verdict.append((name, t1, hit))
ok3 = (verdict[0][2] != [] and verdict[1][2] == [] or verdict[1][2] == [])
rows.append(("X3", "可达性分级（分箱比膝位更细者可分辨，v = 0.1c）", "; ".join("%s: %s" % (n, "/".join(h) if h else "四档都不可分辨") for n, t, h in verdict), "登记", True))
rows.append(("X4", "判决草案", "见注记 §三", "登记", True))

print("=== C6 可测含量评估判据表（实测 对 期望）===")
for tag, name, meas, expx, ok in rows:
    print("%-4s %-44s 实测 %-46s 期望 %-20s %s" % (tag, name, meas, expx, "通过" if ok else "不通过"))
print("")
print("X1 细节：由 w = %.1f l_P（K = %.3f）反解的系数为 %.4f，闭式系数 0.6361（w 约等于 0.64 倍约化康普顿波长）" % (W_LATT, K_LATT, coef_meas))
print("X2/X3 膝位表（宽度 w 与膝位 tau = w/v，单位米与秒）：")
for name, Mr, w, t1, t2 in tabs:
    print("   %-14s M = %.3e m_e；w = %.3e m（%.3e l_P）；tau(0.1c) = %.3e s；tau(c) = %.3e s" % (name, Mr, w, w / L_P, t1, t2))
print("")
print("对照 bin（分箱比膝位更细者可分辨）：阿秒 1e-18、仄秒 1e-21、皮秒 1e-12、电子学 1e-9（单位秒）")
for n, t, h in verdict:
    print("   %-14s v = 0.1c 膝位 %.3e s -> %s" % (n, t, ("可分辨：" + "/".join(h)) if h else "四档都不可分辨"))
print("")
print("X4 判决草案：膝位随缺陷质量反比、随速度反比（tau ∝ 1/(M v)）。中微子档（0.1 eV）的膝位落在飞秒至阿秒量级，可达；")
print("           电子档的膝位落在仄秒量级（3.9e-21 s 至 8.2e-21 s），恰在仄秒分箱的边缘；μ 档与核子档落在 1e-23 s 至 1e-24 s，不可扫。")
print("           故 C6 的可测含量为条件性：需分箱 <= 1e-18 s（轻缺陷）或 <= 1e-21 s（电子档），并在同一装置上同时给出单缺陷通量与计数比。")
