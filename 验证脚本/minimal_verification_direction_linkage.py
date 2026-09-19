# -*- coding: utf-8 -*-
r"""
minimal_verification_direction_linkage.py

[G2-L3 附带复核] 方向系数的两面：相位系数与群速系数（判据 F1 至 F5）

背景：色散论文 §5 给出相位衰减系数表（[100]/[110]/[111] = 1/24、1/48、1/72），
P1-25 记录给出群速衰减读数（[100] 1/8、[111] 1/24，比 2.9965）。两者是同一展开的
两阶读数（群速系数 = 3 × 相位系数），本判据显式复算并核对；同时复核 L3 材料的
口径（基准系数 pi^2/6 与折算能标 E_P^(t) 混用会偏高 4pi^2）并给出更正后的数值表。

模型：三维最近邻色散 omega^2 = (4/t_P^2) sum_a sin^2(k_a l_P/2)，l_P = c = 1。
相位系数 C_ph(n) = lim (omega/(ck) - 1)/u^2，期望 -(1/24) sum n_a^4。
群速系数 C_gr(n) = lim (1 - domega/du)/u^2，期望 (1/8) sum n_a^4。

判据
F1 相位系数三方向 = 1/24、1/48、1/72（比 6:3:2）。
F2 群速系数三方向 = 1/8、1/16、1/24（比 6:3:2，且严格 3 倍于相位系数）。
F3 两约定同值：delta_omega/omega = (1/24)(E/E_P^(t))^2 sum n^4
   = (pi^2/6)(E/E_P)^2 sum n^4，其中 E_P^(t) = hbar/t_P、E_P = hbar*omega_P = 2pi E_P^(t)。
F4 混用对照：pi^2/6 与 E_P^(t) 搭配的口径把全部数值抬高 4pi^2 = 39.478。
F5 真实能量表与窗口（[100] 方向）。

运行：python 验证脚本/minimal_verification_direction_linkage.py
确定性、无随机数，运行约 1 秒。
"""
import numpy as np

E_Pt_GeV = 1.220890e19                 # E_P^(t) = hbar/t_P（折算普朗克能量）
C2PI = 2.0 * np.pi
E_P_basis_GeV = C2PI * E_Pt_GeV        # 体系基准 E_P = hbar*omega_P = 2pi E_P^(t)

dirs = {"[100]": np.array([1.0, 0.0, 0.0]),
        "[110]": np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0),
        "[111]": np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)}


def omega_over_u(u, n):
    return 2.0 * np.sqrt(np.sum(np.sin(u * n / 2.0) ** 2)) / u


def domega_du(u, n):
    s = np.sum(np.sin(u * n / 2.0) ** 2)
    return (1.0 / np.sqrt(s)) * np.sum((n / 2.0) * np.sin(u * n))


u = 1e-4
print("=" * 74)
print("F1 相位系数（-[omega/(ck)-1]/u^2）与 F2 群速系数（[1-domega/du]/u^2）")
print("=" * 74)
print("  %6s | %13s | %13s | %13s | %13s" % ("方向", "相位实测", "-(1/24)Sn^4", "群速实测", "(1/8)Sn^4"))
P, G = {}, {}
for name, n in dirs.items():
    c_ph = -(omega_over_u(u, n) - 1.0) / u ** 2
    c_gr = (1.0 - domega_du(u, n)) / u ** 2
    P[name], G[name] = c_ph, c_gr
    print("  %6s | %13.10f | %13.10f | %13.10f | %13.10f"
          % (name, c_ph, np.sum(n ** 4) / 24.0, c_gr, np.sum(n ** 4) / 8.0))
print("  相位比：%.4f : %.4f : %.4f" % tuple(P[k] / min(P.values()) for k in ("[100]", "[110]", "[111]")))
print("  群速比：%.4f : %.4f : %.4f" % tuple(G[k] / min(G.values()) for k in ("[100]", "[110]", "[111]")))

print()
print("=" * 74)
print("F3 两约定同值 / F4 混用对照")
print("=" * 74)
cases = [("LHC 13 TeV", 13e12), ("IceCube 1 PeV", 1e15),
         ("UHECR 1e20 eV", 1e20), ("3e20 eV", 3e20)]
print("  %14s | %14s | %14s | %20s" % ("情形", "折算(1/24)", "基准(pi^2/6)", "混用读数（偏高倍数）"))
for name, E in cases:
    val_t = (E / (E_Pt_GeV * 1e9)) ** 2 / 24.0
    val_b = (E / (E_P_basis_GeV * 1e9)) ** 2 * np.pi ** 2 / 6.0
    val_mix = (E / (E_Pt_GeV * 1e9)) ** 2 * np.pi ** 2 / 6.0
    print("  %14s | %14.3e | %14.3e | %14.3e（x %.2f）" % (name, val_t, val_b, val_mix, val_mix / val_t))

print()
print("=" * 74)
print("F5 窗口：eps 对应的能量上界（[100]）")
print("=" * 74)
for eps in (1e-2, 1e-18):
    u_max = np.sqrt(24.0 * eps)
    print("  eps = %.0e：E/E_P^(t) <= %.4e，E <= %.3e eV；E/E_P（基准）<= %.4e"
          % (eps, u_max, u_max * E_Pt_GeV * 1e9, u_max / C2PI))
print("  对照：混用口径（pi^2/6 x E_P^(t)）偏高 4pi^2 = %.3f" % (4.0 * np.pi ** 2))
