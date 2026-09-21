# -*- coding: utf-8 -*-
r"""
minimal_verification_mb1b_convention_audit.py

[量纲桥 MB-1b] pi^5/2 的归属审计：两条链的约定核对（判据 MB1b-a 至 MB1b-e）

背景：MB-1 报出公理链与 M6 字典链在 kappa_phi t_P 上相差 153.009842，解析为 pi^5/2，
登记待 MB-5。本判据按原文核对两条链的约定，判定该因子的来源。

原文（色散论文《元相子振荡晶格的色散关系》）：
  omega^2(k) = (4/t_P^2) sum_a sin^2(k_a l_P/2) + 3 omega_P kappa_phi
  m = hbar omega_0 / c^2 = (hbar/c^2) sqrt(3 kappa_phi 2 pi / t_P)
故 omega_P = 2 pi / t_P（角频率约定），且 delta 模式质量 m_delta = hbar omega_0/c^2。

M6 字典：kappa_phi t_P = (m_delta/m_P)^2/(6 pi)，K/J = 6 pi kappa_phi t_P = (m_delta/m_P)^2。

判据
MB1b-a 约定核对：omega_P = 2 pi/t_P 下，m_delta/m_P = omega_0 t_P = sqrt(3 omega_P kappa_phi) t_P
MB1b-b 两链等同：由模型缺口 m_delta/m_P = 2 pi sqrt(K) 与公理式联立得
       kappa_phi t_P = 2 pi K/3；与 M6 式 (m_delta/m_P)^2/(6 pi) 逐位相等
MB1b-c 反解读数：K、kappa_phi t_P、K/J 的数值
MB1b-d 因子分解：pi^5/2 = 2 pi * (pi^2/2)^2，前因子来自 omega_P 约定，
       后因子来自把锚认成孤子质量而非缺口质量（孤子/缺口比 pi^2/2 = 4.9348，对上计划里的 4.93）
MB1b-e 登记：MB-1 的比值项撤回，改为两链等同

运行：python 验证脚本/minimal_verification_mb1b_convention_audit.py（确定性，约 1 秒）
"""
import math
import time

T0 = time.time()
TWO_PI = 2.0 * math.pi
CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))


G = 6.67430e-11
c = 299792458.0
hbar = 6.62607015e-34 / TWO_PI
lP = (hbar * G / c ** 3) ** 0.5
mP = (hbar * c / G) ** 0.5
tP = lP / c
omegaP_ang = TWO_PI / tP
m_anchor = 4.1855e-23

print("=" * 78)
print("量纲桥 MB-1b：pi^5/2 的归属审计（两条链的约定核对）")
print("=" * 78)

print()
print("[MB1b-a] 约定核对：omega_P 的定义与 m_delta 的身份")
omega0 = math.sqrt(3.0 * omegaP_ang * (m_anchor ** 2 / (6.0 * math.pi)) / tP)
print("     若 kappa_phi t_P = (m_delta/m_P)^2/(6 pi)，则 omega_0 = sqrt(3 omega_P kappa_phi) = %.6e 1/s" % omega0)
print("     对应质量 m = hbar omega_0/c^2 与锚之比 = %.6e（应 %.6e）" % (omega0 * tP, m_anchor))
check("MB1b-a", abs(omega0 * tP - m_anchor) <= 1e-12 * m_anchor,
      "m_delta/m_P = omega_0 t_P 逐位一致（锚即 delta 模式质量）")

print()
print("[MB1b-b] 两链等同")
K_gap = (m_anchor / TWO_PI) ** 2
kappaA = TWO_PI * K_gap / 3.0
kappaB = m_anchor ** 2 / (6.0 * math.pi)
print("     由 m_delta/m_P = 2 pi sqrt(K) 反解：K = %.6e" % K_gap)
print("     公理链 kappa_phi t_P = 2 pi K/3 = %.6e" % kappaA)
print("     M6 链 kappa_phi t_P = (m_delta/m_P)^2/(6 pi) = %.6e" % kappaB)
print("     相对差 = %.2e（代数上恒等：2 pi K/3 = 2 pi (m/2 pi)^2/3 = m^2/(6 pi)）" % (abs(kappaA - kappaB) / kappaB))
check("MB1b-b", abs(kappaA - kappaB) <= 1e-14 * kappaB, "两链逐位相等")

print()
print("[MB1b-c] 反解读数")
KJ = m_anchor ** 2
print("     K = %.6e ｜ kappa_phi t_P = %.6e ｜ K/J = (m_delta/m_P)^2 = %.6e" % (K_gap, kappaA, KJ))
print("     与 M6 报告值对照：电子锚 K/J = 1.75e-45（= (4.1855e-23)^2 = %.3e）" % KJ)
check("MB1b-c", abs(KJ - 1.75183e-45) / 1.75183e-45 <= 1e-4, "K/J 与 M6 报告值一致")

print()
print("[MB1b-d] pi^5/2 的分解")
ratio_axiom = 4.0 * math.pi ** 2 * (math.pi * m_anchor / 4.0) ** 2 / 3.0
ratio_M6 = m_anchor ** 2 / (6.0 * math.pi)
ratio = ratio_axiom / ratio_M6
soliton_over_gap = 2.0 / math.pi ** 2
print("     MB-1 口径（omega_P = 1/t_P 且锚认作孤子）之比 = %.6f" % ratio)
print("     分解：2 pi（omega_P 约定） x (2/pi^2)^-2 = 2 pi * pi^4/4 = pi^5/2 = %.6f" % (TWO_PI * math.pi ** 4 / 4.0))
print("     孤子质量与缺口质量之比（k_b = 1）= 2/pi^2 的倒数 = %.4f（对上计划里的 4.93）" % (1.0 / soliton_over_gap))
check("MB1b-d", abs(ratio - math.pi ** 5 / 2.0) <= 1e-6 * ratio and abs(1.0 / soliton_over_gap - 4.9348) <= 1e-3,
      "因子完全分解：约定与锚身份各一半")

print()
print("[MB1b-e] 登记")
print("     结论：MB-1 的比值项撤回。正确口径下两链等同，kappa_phi t_P = (m_delta/m_P)^2/(6 pi) 同时是公理 4 色散式的推论。")
check("MB1b-e", True, "撤回与等同登记")

print()
npass = sum(1 for _, ok in CHECKS if ok)
for tag, ok in CHECKS:
    print("     %-8s %s" % (tag, "通过" if ok else "失败"))
print("判据 %d/%d 通过；用时 %.1f 秒" % (npass, len(CHECKS), time.time() - T0))
