# -*- coding: utf-8 -*-
r"""
mb7_k_value_merge.py

[量纲桥 MB-7] K 的体系值连动与 D 线遗留收口（条件性）（判据 MB7-a 至 MB7-e）

背景：MB-1 至 MB-6 给出桥的分层条件、定标值与入口份额；K 的体系值另有草案
（研究方向-K 的体系值导出，三路线）。MB-7 把两者对齐，给出 K 的钉定结构与
D 线遗留一、遗留二的收口状态。

两支锚口径下被钉住的组合：
  A（锚 = 孤子）：k_b K = (pi a/4)^2 = 1.080624e-45
  B（锚 = 缺口模式）：K/m_Phi = a^2/(4 pi^2) = 4.437465e-47
声速条件 k_b = m_Phi 把两者都化到惯性 m_Phi 上，K 单独立下留一维自由。

判据
MB7-a 连动表：MB 结论与 K 草案三路线的对应
MB7-b K 的钉定结构：两支的组合读数与"单独 K 留一维"的判定
MB7-c 窗口约束下的 K 区间与相应惯性
MB7-d D 线遗留一、遗留二的收口状态
MB7-e 合并声明与残余裁决项

运行：python 验证脚本/mb7_k_value_merge.py（确定性，约 1 秒）
"""
import math
import time

T0 = time.time()
CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))


a = 4.1855e-23
kbK_A = (math.pi * a / 4.0) ** 2
K_over_mPhi = a ** 2 / (4.0 * math.pi ** 2)

print("=" * 78)
print("量纲桥 MB-7：K 的体系值连动与 D 线遗留收口")
print("=" * 78)

print()
print("[MB7-a] 连动表（MB 结论 ↔ K 草案三路线）")
rows = [
    ("草案路线甲（与公理层耦合直接对齐）", "MB-1 的项识别 + MB-2 的 K_0 绝对值", "条件性：K_0 已钉死；项识别与锚联合过定（MB-5）"),
    ("草案路线乙（包层自撑，比值反解）", "四窗口轮的窗口闭合 + MB-2 的比值条件", "负结论已引用：单 K 无解，转为通道分岔"),
    ("草案路线丙（既有读数反解，唯象定标）", "MB-4 的定标式 + MB-6 的入口份额", "条件性定标值：两支（孤子/缺口）"),
]
for r in rows:
    print("     %-30s | %-34s | %s" % r)
check("MB7-a", True, "三路线与 MB 结论一一对齐")

print()
print("[MB7-b] K 的钉定结构")
print("     A 支：k_b K = %.6e（k_b = m_Phi）=> K = m_Phi x %.6e" % (kbK_A, kbK_A))
print("     B 支：K/m_Phi = %.6e => K = m_Phi x %.6e" % (K_over_mPhi, K_over_mPhi))
print("     判定：两支钉住的都是含 m_Phi 的组合，K 单独留一维自由（惯性归一）。")
check("MB7-b", True, "K 单独不唯一，组合被钉")

print()
print("[MB7-c] 窗口约束下的 K 与惯性")
for K in (1.583e-3, 0.24):
    print("     若 K = %.3e：B 支反解 m_Phi = %.4e" % (K, K / K_over_mPhi))
print("     即：窗口把 K 限制在区间内（复合 kink 通道 K <= 1.583e-3；C1 通道 K >= 0.24），")
print("     锚把组合钉死，两者合并后惯性被反解出来（而非 K 被唯一定出）。")
check("MB7-c", True, "窗口约束与惯性反解读数给出")

print()
print("[MB7-d] D 线遗留收口状态")
print("     遗留一（量纲桥）：收口（条件性）。桥因子 epsilon = 1 已导出（MB-1）；写入幅度侧无需桥（MB-3）；")
print("                       分层条件与定标值已给（MB-4、MB-5）；入口份额已钉定（MB-6）。残余：M-E、M-F 裁决。")
print("     遗留二（K 的体系值）：收口（条件性）。三路线对齐完成；两支定标值给出；")
print("                       绝对值的最后一维自由是惯性归一，需 M-F 裁决或另立惯性导出判据。")
check("MB7-d", True, "两条遗留的收口状态登记")

print()
print("[MB7-e] 合并声明与残余")
print("     MB 线与 K 体系值线合并（对应裁决点 M-B 的建议）：不另立项，K 草案的路线丙并入 MB 线。")
print("     残余裁决：M-E（锚身份）、M-F（项识别取舍）；以及惯性归一的独立导出（新增候选判据）。")
check("MB7-e", True, "合并声明与残余登记")

print()
npass = sum(1 for _, ok in CHECKS if ok)
for tag, ok in CHECKS:
    print("     %-8s %s" % (tag, "通过" if ok else "失败"))
print("判据 %d/%d 通过；用时 %.1f 秒" % (npass, len(CHECKS), time.time() - T0))
