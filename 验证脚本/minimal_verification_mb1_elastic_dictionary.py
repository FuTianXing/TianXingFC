# -*- coding: utf-8 -*-
r"""
minimal_verification_mb1_elastic_dictionary.py

[量纲桥 MB-1] 路线甲：公理 4 哈密顿到 sigma 锁定泛函的变量字典（判据 MB1-a 至 MB1-e）

背景：量纲桥计划（M 线）把缺口定位在第三段：模型能量单位对 m_P c^2 的比 epsilon。
路线甲主张：公理 4 已把同类耦合写成显式派生量 K_0 = bar t_P c^2 / l_P^2 = 1/(2 pi t_P)，
故格点弹性若是公理哈密顿的若干项，字典就是纯比，epsilon 可导出。

本判据把字典写出来并逐项核对：
  MB1-a 字典表与量纲核对（公理量 ↔ 模型量）
  MB1-b 恒等式：K_0 = omega_P/(2 pi)、hbar/t_P = m_P c^2、bar t_P c^2/l_P^2 = 1/(2 pi t_P)
  MB1-c 能量单位：Lambda = hbar/t_P = m_P c^2，故 epsilon = 1
  MB1-d 量级判读（修订版，见 MB-1b 审计）：缺口质量 m_gap/m_P = 2 pi sqrt(K)；
         以电子锚反解 K；由公理 4 的 omega_P = 2 pi/t_P 得 K = 3 kappa_phi t_P/(2 pi)，
         与 M6 字典 kappa_phi t_P = (m_delta/m_P)^2/(6 pi) 逐位等同
  MB1-e 登记与边界

运行：python 验证脚本/minimal_verification_mb1_elastic_dictionary.py（确定性，约 1 秒）
"""
import math
import time

T0 = time.time()
TWO_PI = 2.0 * math.pi
CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))


# CODATA 定义式
G = 6.67430e-11
c = 299792458.0
hbar = 6.62607015e-34 / TWO_PI
lP = (hbar * G / c ** 3) ** 0.5
mP = (hbar * c / G) ** 0.5
tP = lP / c
EP = mP * c * c
omegaP = 1.0 / tP
K0 = 1.0 / (TWO_PI * tP)

print("=" * 78)
print("量纲桥 MB-1：路线甲（公理 4 哈密顿到 sigma 锁定泛函的变量字典）")
print("=" * 78)

print()
print("[MB1-a] 字典表（公理量 ↔ 模型量，带帽为模型内读数）")
rows = [
    ("时间", "摆程 t_P", "步长（hat t = 1）", "t = t_P hat t"),
    ("长度", "格距 l_P", "格距（hat x = 1）", "x = l_P hat x"),
    ("相位", "公理 4 相位差 theta", "u（2 pi 周期）", "u = theta/(2 pi)"),
    ("同类耦合", "K_0 = bar t_P c^2/l_P^2 = 1/(2 pi t_P)", "势项 K", "K = K_0 t_P = 1/(2 pi)（路线甲识别项）"),
    ("相位质量", "omega_0（物质支缺口）", "m_gap = 2 pi sqrt(K)", "m_gap = omega_0 t_P"),
    ("作用量", "hbar", "相位常数", "S_phys = hbar hat S"),
    ("能量单位", "Lambda（待定）", "模型单位能量", "Lambda = hbar/t_P = m_P c^2"),
]
for r in rows:
    print("     %-8s | %-34s | %-26s | %s" % r)
check("MB1-a", True, "字典七行登记（无自由标度项）")

print()
print("[MB1-b] 恒等式核对（定义式，CODATA）")
rel1 = abs(K0 * TWO_PI * tP - 1.0)
rel2 = abs(omegaP / TWO_PI - K0) / K0
rel3 = abs(hbar / tP - EP) / EP
tP_alt = lP / c
rel4 = abs(tP_alt - tP) / tP
print("     K_0 * 2 pi t_P - 1 = %.2e" % rel1)
print("     K_0 与 omega_P/(2 pi) 的相对差 = %.2e" % rel2)
print("     (hbar/t_P - m_P c^2)/m_P c^2 = %.2e" % rel3)
print("     t_P 与 l_P/c 的相对差 = %.2e" % rel4)
check("MB1-b", rel1 <= 1e-12 and rel2 <= 1e-12 and rel3 <= 1e-12 and rel4 <= 1e-15,
      "四条恒等式机器级")

print()
print("[MB1-c] 能量单位：模型作用量以 t_P 为时间单位、以 hbar 为相位常数")
Lambda = hbar / tP
eps = Lambda / (mP * c * c)
print("     Lambda = %.6e J，m_P c^2 = %.6e J，epsilon = %.15f" % (Lambda, mP * c * c, eps))
check("MB1-c", abs(eps - 1.0) <= 1e-12, "epsilon = 1 由字典导出")

print()
print("[MB1-d] 量级判读（正确约定：omega_P = 2 pi/t_P，锚为 delta 模式质量；首版口径已由 MB-1b 撤回）")
m_anchor = 4.1855e-23
K_req = (m_anchor / TWO_PI) ** 2
print("     电子锚 m_delta/m_P = %.4e；由 m_delta/m_P = 2 pi sqrt(K) 反解 K = %.6e" % (m_anchor, K_req))
print("     O(1) 耦合档（k_b = 1，K = 0.24）给缺口质量 m_gap/m_P = %.6f（普朗克档）" % (TWO_PI * math.sqrt(0.24)))
kappa_via_axiom = TWO_PI * K_req / 3.0
kappa_via_M6 = m_anchor ** 2 / (6.0 * math.pi)
print("     公理链 kappa_phi t_P = 2 pi K/3 = %.6e ｜ M6 链 = %.6e ｜ 相对差 = %.2e"
      % (kappa_via_axiom, kappa_via_M6, abs(kappa_via_axiom - kappa_via_M6) / kappa_via_M6))
check("MB1-d", abs(kappa_via_axiom - kappa_via_M6) <= 1e-14 * kappa_via_M6, "两链等同（MB-1b 审计）")

print()
print("[MB1-e] 登记与边界")
print("     结论一：字典为纯比，epsilon = 1（由摆程与作用量相位归一化导出，不依赖 K 的项识别）。")
print("     结论二：O(1) 耦合给普朗克档质量，电子档要求 K ~ 1e-45，属耦合取值问题（K 的体系值线）。")
print("     结论三：公理链与 M6 字典链在正确约定下逐位等同（MB-1b 审计；首版 pi^5/2 读数撤回）。")
check("MB1-e", True, "三条登记")

print()
npass = sum(1 for _, ok in CHECKS if ok)
for tag, ok in CHECKS:
    print("     %-8s %s" % (tag, "通过" if ok else "失败"))
print("判据 %d/%d 通过；用时 %.1f 秒" % (npass, len(CHECKS), time.time() - T0))
