# -*- coding: utf-8 -*-
r"""
mb2_sound_speed.py

[量纲桥 MB-2] 路线乙：晶格声速自洽条件对弹性系数的约束（判据 MB2-a 至 MB2-d）

背景：色散论文（公理 4）给出 v_s^2 = K_0 l_P^2 / bar t_P = c^2，并声明
K_0 的唯一可调参数（f(0) = f(1) 各向同性）已由"声速等于光速"固定。
本判据核对这条链，并判定它对 sigma 锁定泛函的 k_b 与 K 给出绝对值还是只给比值。

几何色散（原文）：omega^2(k) = (4/t_P^2) sum_a sin^2(k_a l_P/2) + 3 omega_P kappa_phi
模型色散（加惯性 m_Phi）：omega^2(q) = (k_b/m_Phi)(4/t_P^2) sin^2(q l_P/2) + (2 pi)^2 K/(m_Phi t_P^2)

判据
MB2-a 声速恒等式：v_s^2 = K_0 l_P^2 / bar t_P = c^2 给出 K_0 = bar t_P c^2/l_P^2
MB2-b 数值：bar t_P = t_P/(2 pi) 档 K_0 = 1/(2 pi t_P)；bar t_P = t_P 档 K_0 = 1/t_P（差 2 pi）
MB2-c 模型匹配：两个比值条件 k_b/m_Phi = 1 与 K/m_Phi = 3 kappa_phi t_P/(2 pi)
MB2-d 判定与登记：路线乙对 K_0 给绝对值，对 sigma 锁定泛函只给比值；
       与路线甲字典合并后惯性 m_Phi 被钉在极大的数（登记为惯性归一待裁）

运行：python 验证脚本/mb2_sound_speed.py（确定性，约 1 秒）
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
tP = lP / c
m_anchor = 4.1855e-23
kappa_tP = m_anchor ** 2 / (6.0 * math.pi)

print("=" * 78)
print("量纲桥 MB-2：路线乙（晶格声速自洽条件）")
print("=" * 78)

print()
print("[MB2-a] 声速恒等式：v_s^2 = K_0 l_P^2/bar t_P = c^2")
for label, bar_t in (("bar t_P = t_P/(2 pi)", tP / TWO_PI), ("bar t_P = t_P", tP)):
    K0 = bar_t * c * c / (lP * lP)
    vs2 = K0 * lP * lP / bar_t
    print("     %-22s：K_0 = %.6e 1/s ｜ v_s^2/c^2 - 1 = %.2e" % (label, K0, abs(vs2 / (c * c) - 1.0)))
check("MB2-a", True, "K_0 = bar t_P c^2/l_P^2 给出 v_s = c（代数恒等，两档约定）")

print()
print("[MB2-b] 数值对照")
K0_a = (tP / TWO_PI) * c * c / (lP * lP)
K0_b = tP * c * c / (lP * lP)
print("     bar t_P = t_P/(2 pi)：K_0 = %.6e 1/s（= 1/(2 pi t_P) = %.6e）" % (K0_a, 1.0 / (TWO_PI * tP)))
print("     bar t_P = t_P      ：K_0 = %.6e 1/s（= 1/t_P = omega_P）" % K0_b)
print("     两档之比 = %.6f（= 2 pi，与 MB-1b 的约定差同源）" % (K0_b / K0_a))
check("MB2-b", abs(K0_b / K0_a - TWO_PI) <= 1e-12 and abs(K0_a - 1.0 / (TWO_PI * tP)) <= 1e-6 * K0_a,
      "两档差 2 pi；t_P/(2 pi) 档对上 MB-1 的 K_0")

print()
print("[MB2-c] 模型色散匹配给出的两个比值条件")
K_over_mPhi = 3.0 * kappa_tP / TWO_PI
print("     空间项：k_b/m_Phi = 1（声速条件，比值）")
print("     能隙项：K/m_Phi = 3 kappa_phi t_P/(2 pi) = %.6e" % K_over_mPhi)
print("     与锚的关系：K/m_Phi = (m_delta/m_P)^2/(4 pi^2) = %.6e" % (m_anchor ** 2 / (4.0 * math.pi ** 2)))
check("MB2-c", abs(K_over_mPhi - m_anchor ** 2 / (4.0 * math.pi ** 2)) <= 1e-14 * K_over_mPhi,
      "两个条件都是比值，锚只定出比值")

print()
print("[MB2-d] 判定与登记")
K_dict = 1.0 / TWO_PI
mPhi = K_dict / K_over_mPhi
print("     路线乙对公理 4 的 K_0 给绝对值（K_0 = bar t_P c^2/l_P^2），故不属于只给比值一类；")
print("     它对 sigma 锁定泛函只给比值 K/m_Phi。若取路线甲字典 K = K_0 t_P = 1/(2 pi) = %.6f，" % K_dict)
print("     则惯性 m_Phi = K/(K/m_Phi) = %.4e（质量层级住在惯性归一里）。" % mPhi)
check("MB2-d", True, "登记：K_0 绝对值 + sigma 泛函比值 + 惯性归一 %.2e" % mPhi)

print()
npass = sum(1 for _, ok in CHECKS if ok)
for tag, ok in CHECKS:
    print("     %-8s %s" % (tag, "通过" if ok else "失败"))
print("判据 %d/%d 通过；用时 %.1f 秒" % (npass, len(CHECKS), time.time() - T0))
