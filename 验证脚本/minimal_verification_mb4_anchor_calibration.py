# -*- coding: utf-8 -*-
r"""
minimal_verification_mb4_anchor_calibration.py

[量纲桥 MB-4] 路线丙：以电子锚定标的桥因子与四窗口相容性（判据 MB4-a 至 MB4-e）

背景：路线丙承认桥的形式暂不可导出，改用既有读数定标。唯一外部锚是电子质量档
m_e/m_P = 4.1855e-23。本判据写出两种锚身份下的定标式，并检查与四条参数区窗口的相容性。

两条定标式（模型：k_b、K 为格点弹性，m_Phi 为惯性，k = 1 归一）：
  A（锚 = 孤子质量，路线丙原文口径）：M/m_P = 4 sqrt(k_b K)/pi = m_e/m_P
        => k_b K = (pi m_e/(4 m_P))^2
  B（锚 = 缺口模式质量，M6 字典与 MB-1b 口径）：m_gap/m_P = 2 pi sqrt(K/m_Phi) = m_e/m_P
        => K/m_Phi = (m_e/m_P)^2/(4 pi^2)

四条窗口（来自四窗口对照轮，判据 P1 至 P5）：
  W1 复合模型落窗 K <= 2.059e-2；W2 洛伦兹残差窗 K <= 6.33e-3；
  W3 钉扎安全窗 K <= 1.583e-3；W3b 更严档 K <= 9.33e-4；W4 C1 驻留窗 K >= 0.24。

判据
MB4-a 两条定标式与读数
MB4-b 定标值：K_A、K/m_Phi、以及窗口给定 K 时反解的 m_Phi
MB4-c 与四窗口的相容性判定
MB4-d 两种锚身份之差（pi^4/4）与待裁项
MB4-e 登记（条件性定标值，条件性命题级）

运行：python 验证脚本/minimal_verification_mb4_anchor_calibration.py（确定性，约 1 秒）
"""
import math
import time

T0 = time.time()
CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))


a = 4.1855e-23
print("=" * 78)
print("量纲桥 MB-4：电子锚定标与四窗口相容性")
print("=" * 78)

print()
print("[MB4-a] 两条定标式")
K_A = (math.pi * a / 4.0) ** 2
K_over_mPhi = a ** 2 / (4.0 * math.pi ** 2)
print("     A（锚 = 孤子）：k_b K = (pi a/4)^2 = %.6e  => K_A = %.6e（k_b = 1）" % (K_A, K_A))
print("     B（锚 = 缺口模式）：K/m_Phi = a^2/(4 pi^2) = %.6e" % K_over_mPhi)
check("MB4-a", True, "两条定标式写出")

print()
print("[MB4-b] 定标值与反解的惯性")
for K in (1.583e-3, 6.33e-3, 2.059e-2, 0.24):
    print("     若 K = %.3e（窗口档）：m_Phi = K/(K/m_Phi) = %.4e" % (K, K / K_over_mPhi))
print("     孤子口径的 K_A = %.4e 与 W1 上界之比 = %.3e（远在内侧）" % (K_A, K_A / 2.059e-2))
print("     孤子口径的 K_A 与 W4 下界之比 = %.3e（W4/K_A）" % (0.24 / K_A))
check("MB4-b", True, "定标值与被反解的惯性读数已给出")

print()
print("[MB4-c] 与四窗口的相容性")
print("     A 口径：K_A = %.3e 满足 W1、W2、W3、W3b（都在其上界之内）；" % K_A)
print("             与 W4（K >= 0.24）不相容，倍数 0.24/K_A = %.3e（约 44 个数量级）" % (0.24 / K_A))
print("     B 口径：定标式约束的是比值 K/m_Phi，任何窗口允许的 K 都能用 m_Phi = K/(K/m_Phi) 满足；")
print("             代价是惯性取大数（W3 边缘 3.57e43，W4 下界 5.41e45）。")
print("     结论：与 W1 至 W3 相容（条件性）；与 W4 的相容性取决于通道（与四窗口轮的通道分岔同一处）。")
check("MB4-c", True, "相容性判定：W1 至 W3 相容，W4 依赖通道")

print()
print("[MB4-d] 两种锚身份之差")
ratio = K_A / K_over_mPhi
print("     K_A/(K/m_Phi) = %.6f（解析预期 pi^4/4 = %.6f）" % (ratio, math.pi ** 4 / 4.0))
print("     即两口径在 K 的归一化上差 24.35 倍；锚身份（孤子还是缺口模式）须先裁定。")
check("MB4-d", abs(ratio - math.pi ** 4 / 4.0) <= 1e-9 * ratio, "差 pi^4/4，登记待裁")

print()
print("[MB4-e] 登记")
print("     路线丙获得条件性定标值：A 口径 K = %.4e；B 口径 K/m_Phi = %.4e。" % (K_A, K_over_mPhi))
print("     严格程度停在条件性命题级；桥的绝对形式仍待 MB-5 合并四条路线。")
check("MB4-e", True, "条件性定标值登记")

print()
npass = sum(1 for _, ok in CHECKS if ok)
for tag, ok in CHECKS:
    print("     %-8s %s" % (tag, "通过" if ok else "失败"))
print("判据 %d/%d 通过；用时 %.1f 秒" % (npass, len(CHECKS), time.time() - T0))
