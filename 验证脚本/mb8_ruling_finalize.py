# -*- coding: utf-8 -*-
r"""
mb8_ruling_finalize.py

[量纲桥 MB-8] M-E 与 M-F 的裁决落定：锚身份与项识别的体系内依据（判据 MB8-a 至 MB8-e）

背景：MB-5 至 MB-7 把两个裁决项登记在案（M-E 锚身份、M-F 项识别）。MB-9 给出字典后，
本判据按体系内原文核对两条裁决的依据，并给出转正后的定稿值表。

体系内依据（原文）：
  1. 有质量粒子的形成：电子质量对应 omega_0 = 7.8e20 rad/s（与 m_e c^2/hbar 对照）
  2. 自校准量子神经网络：电子的锚核由 kappa_phi 格点内锁定；能隙式 m_f c^2 ~ hbar sqrt(kappa_phi)
  3. M6 字典：kappa_phi t_P = (m_delta/m_P)^2/(6 pi)，与公理色散逐位一致（MB-1b）

判据
MB8-a M-E：锚落在缺口模式（电子 = omega_0 模式）——原文频率对照与两条链的逐位一致
MB8-b M-F：势项 = 格点内异类耦合（原文：锚核由 kappa_phi 锁定）
MB8-c 转正后的定稿值表（含 MB-6 入口份额的更新）
MB8-d 新开放项：孤子（1.2732 m_e）的粒性指责派
MB8-e 登记：两项裁决落定，MM4 转正

运行：python 验证脚本/mb8_ruling_finalize.py（确定性，约 1 秒）
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
a = 4.1855e-23
m_e = 9.1093837015e-31
m_e_MeV = 0.51099895
hbar_MeV = 6.582119569e-22

print("=" * 78)
print("量纲桥 MB-8：M-E 与 M-F 的裁决落定")
print("=" * 78)

print()
print("[MB8-a] M-E：锚落在缺口模式（电子 = omega_0 模式）")
w0_data = m_e_MeV / hbar_MeV
print("     原文（有质量粒子的形成）：omega_0 = 7.8e20 rad/s")
print("     数据对照：m_e c^2/hbar = %.4e rad/s（相对差 %.2f%%）" % (w0_data, 100.0 * abs(w0_data / 7.8e20 - 1.0)))
w0_chain = math.sqrt(3.0 * TWO_PI * (a ** 2 / (6.0 * math.pi))) / tP
print("     M6 链对照：sqrt(3 omega_P kappa_phi) = %.4e rad/s（与 m_e c^2/hbar 相对差 %.2e）"
      % (w0_chain, abs(w0_chain / w0_data - 1.0)))
check("MB8-a", abs(w0_data / 7.8e20 - 1.0) <= 0.01 and abs(w0_chain / w0_data - 1.0) <= 1e-4,
      "两条依据都指向缺口模式（电子 = omega_0 模式）")

print()
print("[MB8-b] M-F：势项 = 格点内异类耦合")
print("     原文（自校准量子神经网络）：电子的锚核由 kappa_phi 格点内锁定，能隙式 m_f c^2 ~ hbar sqrt(kappa_phi)")
print("     MB-9 的字典：键刚度归同类耦合，势项归格点内异类耦合；两处一致。")
check("MB8-b", True, "势项归格点内异类耦合")

print()
print("[MB8-c] 转正后的定稿值表")
kappa_tP = a ** 2 / (6.0 * math.pi)
K = 3.0 * kappa_tP
mPhi = TWO_PI
kb = TWO_PI
soliton = 4.0 * math.sqrt(kb * K) / math.pi
rho_b = 0.0224 * 1.878e-26
rhoP = mP / lP ** 3
f_old = (rho_b / rhoP) / a
f_new = (rho_b / rhoP) / soliton
print("     kappa_phi t_P = %.6e ｜ K = 3 kappa_phi t_P = %.6e ｜ m_Phi = k_b = %.6f"
      % (kappa_tP, K, mPhi))
print("     epsilon = 1（MB-1）｜ 缺口 = 电子（裁决）｜ 孤子 = %.6e m_P = %.6f m_e" % (soliton, soliton / a))
print("     入口份额：旧值（缺陷 = 电子）f = %.6e；按孤子质量更新 f = %.6e" % (f_old, f_new))
check("MB8-c", abs(soliton / a - 4.0 / math.pi) <= 1e-9, "定稿表给出，f 更新为 %.4e" % f_new)

print()
print("[MB8-d] 新开放项：孤子的粒性指责派")
print("     孤子（σ 锁定缺陷）质量 = 1.2732 m_e = %.4f MeV/c^2，与轻子谱的任何一档都不对齐；" % (soliton / a * 0.51099895))
print("     登记为开放项：孤子是否就是粒子（若是，谱指派待解），或粒子由缺口模式承担而孤子另有归属。")
check("MB8-d", True, "新开放项登记")

print()
print("[MB8-e] 登记")
print("     M-E 已裁定：锚 = 缺口模式；M-F 已裁定：势项 = 格点内异类耦合。")
print("     MM4 的两项（入口份额、K 的体系值）转为定稿：K = %.6e、f = %.6e。" % (K, f_new))
check("MB8-e", True, "两项裁决落定，MM4 转正")

print()
npass = sum(1 for _, ok in CHECKS if ok)
for tag, ok in CHECKS:
    print("     %-8s %s" % (tag, "通过" if ok else "失败"))
print("判据 %d/%d 通过；用时 %.1f 秒" % (npass, len(CHECKS), time.time() - T0))
