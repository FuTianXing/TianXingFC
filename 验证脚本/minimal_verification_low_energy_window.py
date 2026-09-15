# -*- coding: utf-8 -*-
"""G2-L3 等价声明的精度边界：低能与标准量子场论的偏离量级与能标窗口。

判据（先立判据，再实现）：
  L3-1 低能精确性：格点色散在 u = k l_P -> 0 时给出 omega/(ck) -> 1。
  L3-2 首阶系数：偏差的首阶系数等于 -(1/24)(sum_a n_a^4)，方向比 [100]:[110]:[111] = 3:1.5:1；
       换算到体系基准约定（E_P = hbar*omega_P）为 -pi^2/6 * (sum_a n_a^4)(E/E_P)^2。
  L3-3 次阶：残差按 u^4 标度，故误差界由首阶项控制。
  L3-4 窗口表：给定精度 eps，窗口上界 (E/E_P) = sqrt(eps / C_worst)，C_worst = pi^2/6。
  L3-5 实际能量对照：LHC、PeV 中微子、最高能宇宙线三个能量上的偏离量级。

单位：格距 l_P = 1，c = 1（即 omega 以 1/t_P 为单位），故 u = k，E/E_P^(t) = u。
确定性，无随机数。
"""

import numpy as np

E_P_GeV = 1.220890e19          # 普朗克能量（体系基准约定 E_P = hbar*omega_P）
E_P_eV = E_P_GeV * 1e9

print("=" * 76)
print("L3-1 低能精确性：omega/(c k) -> 1")
print("=" * 76)
dirs = {"[100]": np.array([1.0, 0.0, 0.0]),
        "[110]": np.array([1.0, 1.0, 0.0]) / np.sqrt(2),
        "[111]": np.array([1.0, 1.0, 1.0]) / np.sqrt(3)}


def omega_over_ck(u, n):
    """格点色散 omega = 2 sqrt(sum_a sin^2(u_a/2))（l_P = 1, c = 1）。"""
    ua = u * n
    return 2.0 * np.sqrt(np.sum(np.sin(ua / 2.0) ** 2)) / u


print("  %8s | %14s | %14s | %14s" % ("u", "[100]", "[110]", "[111]"))
for u in (1e-2, 1e-3, 1e-4, 1e-6):
    row = "  %8.0e |" % u
    for n in dirs.values():
        row += " %14.12f |" % omega_over_ck(u, n)
    print(row)
print("  期望：三个方向都趋于 1（低能精确）")

print()
print("=" * 76)
print("L3-2 首阶系数与方向比")
print("=" * 76)
print("  %6s | %14s | %14s | %12s" % ("方向", "实测 (w/ck-1)/u^2", "理论 -(1/24)sum n_a^4", "相对偏差"))
C_t = {}
for name, n in dirs.items():
    u = 1e-4
    val = (omega_over_ck(u, n) - 1.0) / u ** 2
    theo = -np.sum(n ** 4) / 24.0
    C_t[name] = -theo
    print("  %6s | %14.10f | %14.10f | %12.3e" % (name, val, theo, abs(val - theo) / abs(theo)))
print("  方向比（取绝对值）：%.4f : %.4f : %.4f（期望 3 : 1.5 : 1）"
      % tuple(C_t[k] / min(C_t.values()) for k in ("[100]", "[110]", "[111]")))
print("  体系基准约定（E_P = hbar*omega_P）下的系数 = (2pi)^2/24 = %.6f 倍" % ((2 * np.pi) ** 2 / 24))
print("  故最不利方向（[100]）的系数 C_worst = pi^2/6 = %.6f" % (np.pi ** 2 / 6))

print()
print("=" * 76)
print("L3-3 次阶：残差按 u^4 标度")
print("=" * 76)
n = dirs["[100]"]
print("  %10s | %16s | %16s | %12s" % ("u", "残差 (w/ck-1-首阶)", "残差 / u^4", "(u^4 理论量级)"))
for u in (1e-1, 5e-2, 2e-2):
    first = -(1 / 24.0) * u ** 2
    resid = omega_over_ck(u, n) - 1.0 - first
    print("  %10.1e | %16.6e | %16.6e | %12.3e" % (u, resid, resid / u ** 4, u ** 4))
print("  期望：残差 / u^4 趋于常数（首阶之后是 u^4 阶），故误差界由首阶项控制")

print()
print("=" * 76)
print("L3-4 窗口表：给定精度 eps 的能标上界")
print("=" * 76)
C_worst = np.pi ** 2 / 6
print("  %10s | %14s | %16s" % ("精度 eps", "(E/E_P) 上界", "对应能量（eV）"))
for eps in (1e-2, 1e-6, 1e-12, 1e-18, 1e-24):
    ratio = np.sqrt(eps / C_worst)
    print("  %10.0e | %14.3e | %16.3e" % (eps, ratio, ratio * E_P_eV))
print("  期望：窗口随 sqrt(eps) 收缩；低能层（E << E_P）在全部所列精度下都成立")

print()
print("=" * 76)
print("L3-5 实际能量对照")
print("=" * 76)
cases = [("LHC 质子 13 TeV", 13e12), ("IceCube 中微子 1 PeV", 1e15),
         ("最高能宇宙线 1e20 eV", 1e20), ("极限宇宙线 3e20 eV", 3e20)]
print("  %22s | %12s | %14s" % ("情形", "E/E_P", "偏离量级 C_worst*(E/E_P)^2"))
for name, E in cases:
    r = E / E_P_eV
    print("  %22s | %12.3e | %14.3e" % (name, r, C_worst * r ** 2))
print("  期望：在可及最高能量上，偏离约 1e-16 或更小")
