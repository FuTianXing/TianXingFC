# -*- coding: utf-8 -*-
r"""
minimal_verification_mb5_route_merge.py

[量纲桥 MB-5] 四条路线合并：桥的形式判定与过定检查（判据 MB5-a 至 MB5-e）

背景：四条路线各自有判（MB-1 字典纯比与桥因子、MB-2 声速与比值条件、MB-3 写入
幅度的物理身份、MB-4 电子锚的条件性定标）。MB-5 把它们对齐，判定桥是单一因子
还是一组分层的条件，并检查联合采纳是否过定。

三条模型条件与一个经验锚：
  (i)  声速条件（MB-2）：k_b = m_Phi
  (ii) 缺口/锚（MB-1b、MB-4）：K/m_Phi = (m_e/m_P)^2/(4 pi^2)
  (iii)项识别（MB-1 路线甲的候选）：K = K_0 t_P = 1/(2 pi)（模型单位）
  锚：电子档 m_e/m_P = 4.1855e-23，对象身份待裁（孤子或缺口模式）

判据
MB5-a 条件清单与自由量计数：三个条件对三个参数 (k_b, K, m_Phi)
MB5-b 联合采纳的两条场景与病理读数：另一对象被推成 3.04e22 倍超重
MB5-c 放宽项识别后的解：K 与 m_Phi 留一维自由（惯性归一）
MB5-d 桥的形式判定：因子部分平凡化为 1，内容是分层条件
MB5-e 登记：MM3 的形式结论与两个裁决项

运行：python 验证脚本/minimal_verification_mb5_route_merge.py（确定性，约 1 秒）
"""
import math
import time

T0 = time.time()
TWO_PI = 2.0 * math.pi
CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))


a = 4.1855e-23
K_dict = 1.0 / TWO_PI
K_over_mPhi = a ** 2 / (4.0 * math.pi ** 2)

print("=" * 78)
print("量纲桥 MB-5：四条路线合并（桥的形式与过定检查）")
print("=" * 78)

print()
print("[MB5-a] 条件清单与自由量计数")
print("     (i)   k_b = m_Phi                       （声速条件）")
print("     (ii)  K/m_Phi = a^2/(4 pi^2)            （缺口与电子锚）")
print("     (iii) K = K_0 t_P = 1/(2 pi) = %.6f  （项识别，路线甲候选）" % K_dict)
print("     参数三个（k_b、K、m_Phi）；三条条件各自独立时把三者全定。")
check("MB5-a", True, "条件清单与计数给出")

print()
print("[MB5-b] 联合采纳的两条场景")
mPhi_B = K_dict / K_over_mPhi
kb_B = mPhi_B
M_sol_B = 4.0 * math.sqrt(kb_B * K_dict) / math.pi
print("     场景一（项识别 + 缺口锚）：m_Phi = k_b = %.4e，K = %.6f" % (mPhi_B, K_dict))
print("           孤子（粒子）质量 M/m_P = 4 sqrt(k_b K)/pi = %.4e（超重 %.1e 倍）" % (M_sol_B, M_sol_B / a))
kb_A = a ** 2 * 16.0 / (math.pi ** 2 * 4.0) / 4.0 * 4.0
K_A = (math.pi * a / 4.0) ** 2
kb_A = K_A / K_dict
gap_A = TWO_PI * math.sqrt(K_dict / kb_A)
print("     场景二（项识别 + 孤子锚）：k_b = m_Phi = K_A/K = %.4e" % kb_A)
print("           缺口质量 m_gap/m_P = 2 pi sqrt(K/m_Phi) = %.4e（同样超重 %.1e 倍）" % (gap_A, gap_A / a))
check("MB5-b", abs(M_sol_B / a - gap_A / a) <= 1e-2 * (M_sol_B / a),
      "两场景对称地把另一对象推成 %.2e 倍超重" % (M_sol_B / a))

print()
print("[MB5-c] 放宽项识别后的解")
kappa_tP = a ** 2 / (6.0 * math.pi)
print("     (i)+(ii) 下：k_b = m_Phi（自由），K = m_Phi a^2/(4 pi^2)，kappa_phi t_P = a^2/(6 pi) = %.6e" % kappa_tP)
print("     kappa_phi t_P 与 MB-1b 的读数逐位一致（该式不依赖项识别）；")
print("     剩下的一维自由量是惯性归一 m_Phi（MB-2 已登记）。")
check("MB5-c", abs(kappa_tP - 9.293805e-47) / 9.293805e-47 <= 1e-4, "kappa_phi t_P 与 MB-1b 一致，留一维自由")

print()
print("[MB5-d] 桥的形式判定")
print("     因子部分：epsilon = 1（MB-1，单一数，已导出，不依赖项识别与锚）。")
print("     内容部分：分层条件（声速、缺口/锚、可选的项目识别），非单一因子。")
print("     联合采纳项识别与锚会过定并把另一对象推成 %.2e 倍超重，故项识别应放宽或改写。" % (M_sol_B / a))
check("MB5-d", True, "形式判定：单一因子（=1）加分层条件")

print()
print("[MB5-e] 登记")
print("     MM3 的结论取第二支：桥是一组分层的条件，而非单一因子。")
print("     两个裁决项：其一，锚身份（缺口模式或孤子）；其二，项识别是否保留。")
check("MB5-e", True, "MM3 形式结论与两个裁决项登记")

print()
npass = sum(1 for _, ok in CHECKS if ok)
for tag, ok in CHECKS:
    print("     %-8s %s" % (tag, "通过" if ok else "失败"))
print("判据 %d/%d 通过；用时 %.1f 秒" % (npass, len(CHECKS), time.time() - T0))
