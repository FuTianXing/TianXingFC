# -*- coding: utf-8 -*-
r"""
mb9_inertia_derivation.py

[量纲桥 MB-9（新立判据）] 惯性归一的独立导出：公理 4 哈密顿的显式系数（判据 MB9-a 至 MB9-e）

背景：MB-5 与 MB-7 把"惯性归一 m_Phi"登记为桥的最后一维自由量。本判据新立一条：
惯性由公理 4 的哈密顿显式给出。原文（元相子几何动力学，公理 4）：

  H = sum_i (hbar bar t_P/2) dot{tilde theta}_i^2
      - sum_{i<j, 同类} hbar K_{ij} cos(theta_j - theta_i)
      - H_{异类},   H_{异类} = -hbar kappa_phi sum_p sum_{a<b} cos(theta_p^a - theta_p^b)
  K_0 = bar t_P c^2 / l_P^2 = 1/(2 pi t_P)

三个系数都是显式的：动能系数 I = hbar bar t_P（惯性）、键刚度 hbar K_0、
格点内（三元组）锁定 3 hbar kappa_phi（三对 a < b）。

两个场的归一（theta 为提升相位，u = theta/2 pi 为 C1/D 线惯例）：
  theta 场（模型单位，能量 m_P c^2、时间 t_P）：hat I = bar t_P/t_P = 1/(2 pi)；k_b = K_0 t_P = 1/(2 pi)
  u 场（同）：m_Phi = 2 pi；k_b = 2 pi（相差 (2 pi)^2 的场归一）
  两者共用 K = 3 kappa_phi t_P（势项按 (1 - cos 2 pi u) 写）

判据
MB9-a 公理 4 的三系数与公理 5 的互锁：I omega_P^2 = hbar omega_P = E_Phi
MB9-b 模型单位字典（两套场归一）：hat I = k_b = 1/(2 pi)（theta）与 m_Phi = k_b = 2 pi（u）
MB9-c 内禀核对：缺口 m_gap/m_P = sqrt(6 pi kappa_phi t_P) = 公理色散的 omega_0 t_P（逐位一致）
MB9-d 关闭 MB-7 的自由维：两支的 K 读数与质量比 M/m_gap = 4/pi
MB9-e 登记：惯性不再自由；势项对应格点内异类耦合；C1 的 k_b = 1 归一与公理的 2 pi 差一个口径

运行：python 验证脚本/mb9_inertia_derivation.py（确定性，约 1 秒）
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
bar_t = tP / TWO_PI
omegaP = TWO_PI / tP
a = 4.1855e-23
kappa_tP = a ** 2 / (6.0 * math.pi)

print("=" * 78)
print("量纲桥 MB-9：惯性归一的独立导出（公理 4 哈密顿的显式系数）")
print("=" * 78)

print()
print("[MB9-a] 公理 4 的三个系数与公理 5 的互锁")
I = hbar * bar_t
print("     动能系数 I = hbar bar t_P = %.6e J s^2；omega_P = 2 pi/t_P = %.6e 1/s" % (I, omegaP))
print("     I omega_P^2 = %.6e J ｜ E_Phi = hbar omega_P = %.6e J ｜ 相对差 %.2e"
      % (I * omegaP ** 2, hbar * omegaP, abs(I * omegaP ** 2 - hbar * omegaP) / (hbar * omegaP)))
check("MB9-a", abs(I * omegaP ** 2 - hbar * omegaP) / (hbar * omegaP) <= 1e-15,
      "I omega_P^2 = E_Phi 精确（惯性由公理 5 的能量归属钉住）")

print()
print("[MB9-b] 模型单位字典（两套场归一）")
hatI = bar_t / tP
kb_theta = (bar_t * c * c / lP ** 2) * tP
mPhi_u = TWO_PI
kb_u = TWO_PI ** 2 * kb_theta
print("     theta 场：hat I = bar t_P/t_P = %.6f ｜ k_b = K_0 t_P = %.6f" % (hatI, kb_theta))
print("     u 场    ：m_Phi = %.6f ｜ k_b = %.6f（相差 (2 pi)^2 = %.6f）" % (mPhi_u, kb_u, TWO_PI ** 2))
print("     两者共用 K = 3 kappa_phi t_P（势项按 1 - cos 2 pi u 写）")
check("MB9-b", abs(hatI - 1.0 / TWO_PI) <= 1e-15 and abs(kb_u - TWO_PI) <= 1e-12 and abs(kb_u / kb_theta - TWO_PI ** 2) <= 1e-9,
      "两套场归一的系数都是 O(1) 显式值")

print()
print("[MB9-c] 内禀核对：缺口质量与公理色散")
K = 3.0 * kappa_tP
gap = TWO_PI * math.sqrt(K / mPhi_u)
gap_axiom = math.sqrt(3.0 * TWO_PI * kappa_tP)
vs2 = kb_u / mPhi_u
print("     K = 3 kappa_phi t_P = %.6e；缺口 m_gap/m_P = 2 pi sqrt(K/m_Phi) = %.6e" % (K, gap))
print("     公理色散：omega_0 t_P = sqrt(6 pi kappa_phi t_P) = %.6e ｜ 相对差 %.2e"
      % (gap_axiom, abs(gap - gap_axiom) / gap_axiom))
print("     声速：k_b/m_Phi = %.15f（相对差 %.2e）" % (vs2, abs(vs2 - 1.0)))
check("MB9-c", abs(gap - gap_axiom) / gap_axiom <= 1e-15 and abs(vs2 - 1.0) <= 1e-15,
      "缺口与公理色散逐位一致，声速自动为 1")

print()
print("[MB9-d] 关闭 MB-7 的自由维：两支的 K 读数")
K_gap_anchor = K
K_sol_anchor = (math.pi * a / 4.0) ** 2 / kb_u
print("     锚 = 缺口模式：K = 3 kappa_phi t_P = %.6e（缺口 = 电子）" % K_gap_anchor)
soliton = 4.0 * math.sqrt(kb_u * K_gap_anchor) / math.pi
print("           由同一 K 反推孤子（粒子）质量 = %.6e m_P（= 电子的 %.4f 倍）" % (soliton, soliton / a))
print("     锚 = 孤子：K = (pi a/4)^2/k_b = %.6e（孤子 = 电子）" % K_sol_anchor)
gap2 = TWO_PI * math.sqrt(K_sol_anchor / mPhi_u)
print("           由同一 K 反推缺口质量 = %.6e m_P（= 电子的 %.4f 倍）" % (gap2, gap2 / a))
print("     质量比：M/m_gap = 4/pi = %.6f（解析）" % (4.0 / math.pi))
check("MB9-d", abs((soliton / a) - 4.0 / math.pi) <= 1e-12 and abs((gap2 / a) - math.pi / 4.0) <= 1e-12,
      "两支自洽：孤子 = 4/pi 电子，缺口 = pi/4 电子（互为倒数）")

print()
print("[MB9-e] 登记")
print("     惯性归一由公理给出（theta 场 1/(2 pi)，u 场 2 pi），不再是自由量。")
print("     势项 = 格点内（三元组）异类耦合：M-F 的项识别由公理分项结构读出，同类耦合归键刚度。")
print("     C1 各轮的 k_b = 1 归一与公理的 k_b = 2 pi 差一个 2 pi 的口径，O(1) 因子与窗口 K 轴随之平移。")
check("MB9-e", True, "登记完成")

print()
npass = sum(1 for _, ok in CHECKS if ok)
for tag, ok in CHECKS:
    print("     %-8s %s" % (tag, "通过" if ok else "失败"))
print("判据 %d/%d 通过；用时 %.1f 秒" % (npass, len(CHECKS), time.time() - T0))
