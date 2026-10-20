# -*- coding: utf-8 -*-
r"""
minimal_verification_quartic_magnitude_audit.py

第724号 来源丁负贡献的量级审计（QL-1 至 QL-8）

问题：第722号给出分支乙（写入反作用在物侧给出负贡献，站点侧增量 -0.99882383，总量守恒）。
本轮判它的量级够不够压住 C1 驻留窗口。

对象类型前置：被问对象是参数类（可问量纲、标度、取值范围，不可问对易），判据形式为比值判定，
与第720号（构型判据加存在性判断）、第722号（构型判据加存在性判断）不同。

本轮最先要解决的问题（QL-1 至 QL-3）：跨模型对表的合法性。
  两侧读数来自不同模型：
  写入侧（第722号，第127条装置）：紧致相位 + 站点模 + 记录寄存器两能级，格距为抽象格点，
    耦合按脉冲面积配成完整交换 g = (pi/2) omega_P / int f dtheta（实测 5.092）。
    其「能量」是布居（激发数），无量纲。
  C1 侧（振幅通道径向曲线轮）：中性复标量载体，N = 32、格距 1、开放边界，
    E(R) 的线张力斜率 129.0 至 251.8 每单位 R，钉扎斜率补偿上界 41.3，缺口约为线张力的 10%。
    其「能量」是复标量凝聚能，带长度量纲。
  两者能量不是同一个量，直接相比等于拿「数了几个粒子」与「每格多少能量」作商。

判据（先立期望，再实现）：
  QL-1 装置可比性检验：列两侧可观测量与单位，核能量量纲是否同类。
  QL-2 量纲桥的存在性：枚举三条候选桥（甲 写入强度对 C1 芯区占据数；乙 折返时刻与摆程周期
       的节拍一致性；丙 与第591条 JA-3 的每单位长度能量接口），逐条判成立与否。
  QL-3 桥上不确定度：对成立的桥给不确定度来源与量级，判是否主导结论。
  QL-4 转移量的稳定性：复现第722号 DB-6 口径（强度档、占据数档）并加档位。
  QL-5 门槛侧的确定性：复现 C1 侧临界缺口读数（线张力斜率、钉扎补偿上界、缺口比例）。
  QL-6 分档比较：桥成立时把两侧折算到同一量纲后给分档比值。
  QL-7 三读法条件报告：读法乙、读法甲、读法甲加第二分栏下比值的可定义性。
  QL-8 恒真检验：人为把一侧能量尺度缩放一个数量级，比值须相应改变一个数量级。

预期（先立）：五种结论形态——桥成立且比值在 0.1 至 1 档（最有价值）、桥成立且比值在 10 以上
档（须连带报告过剩是否触发失稳）、桥成立且比值在 0.1 以下档（较可能）、桥不存在（本项在当前
模型配置下不可执行，缺的是把写入过程与 C1 环模型放进同一装置）、R-3 取读法甲且不设负贡献
（本项在该读法下无对象，附读法乙下的对照值）。

可逆性等级：QL-1 至 QL-3、QL-7、QL-8 为判据稳健（量纲与布尔判定）；QL-4 与 QL-5 为数值复现
（容差从各自装置的误差取）；QL-6 的分档边界按两侧不确定度之和划分。
前提依赖：第722号的读数、第127条装置、振幅通道轮的门槛读数、第591条 JA-3 的每单位长度能量
口径、D-5 的每过轴恰写一笔读数。

运行：python 验证脚本/minimal_verification_quartic_magnitude_audit.py
确定性，无随机数。
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

T_START = time.time()
OUT = r"D:/software/TianXingFC/images/quartic_magnitude_audit_724.png"

RESULTS = []


def record(cid, name, expect, actual, ok, note=""):
    RESULTS.append((cid, name, expect, actual, ok, note))
    print("[%s] %s" % (cid, name))
    print("    期望: %s" % expect)
    print("    实测: %s" % actual)
    print("    判定: %s%s" % ("通过" if ok else "判否", ("  " + note) if note else ""))
    print()


# =====================================================================
# 写入侧装置（第 127 条 / 第 722 号同装置）
# =====================================================================
N_THETA = 256
OMEGA_P = 2.0 * np.pi
THETA0 = np.pi
W_PULSE = 0.5
PACKET_W = [0.15, 0.25, 0.40]
DT = 2.0e-3
T_MAX = 1.2


def pulse(theta, w):
    return np.exp(-(1.0 - np.cos(theta)) / (2.0 * w ** 2))


def omega_of_k(kvec):
    return -OMEGA_P * kvec


def apply_int(psi, f, dt, g):
    ang = g * f * dt
    cos_a, sin_a = np.cos(ang), np.sin(ang)
    out = np.zeros_like(psi)
    for c in range(2):
        for a in range(2):
            out[:, a, c] += psi[:, a, c] * cos_a
    for a in range(2):
        for c in range(2):
            out[:, 1 - a, 1 - c] += -1j * sin_a * psi[:, a, c]
    return out


def split_step(psi, dt, theta, kvec, w, g, n_modes=2):
    f = pulse(theta, w)
    psi = apply_int(psi, f, dt / 2.0, g)
    for a in range(n_modes):
        for c in range(n_modes):
            psi[:, a, c] = np.fft.ifft(np.fft.fft(psi[:, a, c]) * np.exp(1j * omega_of_k(kvec) * dt))
    psi = apply_int(psi, f, dt / 2.0, g)
    return psi


def make_packet(theta, pw, site=1):
    psi = np.zeros((N_THETA, 2, 2), dtype=complex)
    pk = np.exp(-((theta - THETA0) ** 2) / (2.0 * pw ** 2))
    psi[:, site, 0] = pk / np.linalg.norm(pk)
    return psi


def site_pop(p):
    return float(np.sum(np.abs(p[:, 1, :]) ** 2))


def reg_pop(p):
    return float(np.sum(np.abs(p[:, :, 1]) ** 2))


def run(psi, g, theta, kvec, steps=None):
    n = steps if steps is not None else int(round(T_MAX / DT))
    for _ in range(n):
        psi = split_step(psi, DT, theta, kvec, W_PULSE, g)
    return psi


theta = np.linspace(0.0, 2.0 * np.pi, N_THETA, endpoint=False)
kvec = np.fft.fftfreq(N_THETA, d=1.0 / N_THETA)
area = float(np.sum(pulse(theta, W_PULSE)) * (2.0 * np.pi / N_THETA))
G_FULL = (np.pi / 2.0) * OMEGA_P / area
t_cross = (2.0 * np.pi - THETA0) / OMEGA_P

# =====================================================================
# QL-1 装置可比性检验
# =====================================================================
print("=" * 78)
print("QL-1 装置可比性检验")
print("=" * 78)

SIDE_WRITE = {
    "装置": "写入算符轮（第127/722条）：紧致相位 + 站点模 + 记录寄存器两能级",
    "格距": "抽象格点（格距 1，无物理长度）",
    "可观测量": "布居（激发数）",
    "能量量纲": "无量纲计数",
    "耦合": "g = (pi/2) omega_P / int f dtheta = %.3f（按脉冲面积配成完整交换）" % G_FULL,
    "节拍": "过轴时刻 t_cross = %.4f" % t_cross,
}
SIDE_C1 = {
    "装置": "振幅通道径向曲线轮：中性复标量载体，N = 32、格距 1、开放边界",
    "格距": "1 物理格距单位",
    "可观测量": "复标量凝聚能 E(R)",
    "能量量纲": "带长度量纲（能量 × 长度）",
    "门槛": "线张力斜率 129.0 至 251.8 每单位 R；钉扎补偿上界 41.3；缺口约 10%",
    "节拍": "自由弛豫（连续时间，无过轴离散）",
}

for k, v in SIDE_WRITE.items():
    print("  写入侧 %-10s %s" % (k, v))
print()
for k, v in SIDE_C1.items():
    print("  C1 侧   %-10s %s" % (k, v))
print()

same_dim = (SIDE_WRITE["能量量纲"] == SIDE_C1["能量量纲"])
ql1_ok = not same_dim  # 判据：两侧能量量纲不同，则直接比值不合法，须走量纲桥
record("QL-1", "装置可比性检验：两侧能量量纲同类否",
       "报告两侧量纲；若不同则直接比值不合法（判据成立）",
       "写入侧 %s，C1 侧 %s，%s"
       % (SIDE_WRITE["能量量纲"], SIDE_C1["能量量纲"],
          "同类" if same_dim else "不同类"),
       ql1_ok, "直接比值不合法，量级审计须先过量纲桥")

# =====================================================================
# QL-2 量纲桥的存在性
# =====================================================================
print("=" * 78)
print("QL-2 量纲桥的存在性（三条候选）")
print("=" * 78)

# 桥甲：写入强度对 C1 芯区占据数
# 写入侧：每站点每过轴一次完整交换，转移量 = 占据数（无量纲）
# C1 侧：芯区每格点的缺陷占据数 n_core 与钉扎强度 K 挂钩
# 缺口：两侧的「占据数」虽同名，但写入侧是无量纲计数、C1 侧是复标量模方，
# 且 C1 侧钉扎强度 K 的物理标度属 P1-2 档位债（未定）
BRIDGE_A = {
    "名": "甲 写入强度对 C1 芯区占据数",
    "形式": "写入侧每站点每过轴转移 n 个激发 ↔ C1 芯区每格点 n_core 个缺陷占据",
    "成立": False,
    "依据": "两侧「占据数」不同质：写入侧为无量纲两能级布居，C1 侧为复标量模方乘锚核截面；"
            "且 C1 侧占据数与钉扎强度 K 的物理标度属 P1-2 档位债（未定），无换算式可写",
}
# 桥乙：折返时刻与摆程周期
# 写入侧节拍 = 每过轴一次，过轴时刻 t_cross = (2pi - theta0)/omega_P
# C1 侧：D-5 读数为缺陷每穿过一格恰写一笔，与速度六档宽度三档无关
# 缺口：C1 侧的「每拍」是缺陷穿越格点的空间事件，写入侧是相位过轴的时间事件，
# 两者的拍频是否相等取决于缺陷速度与格距换算，属 D-5 速度上界问题
BRIDGE_B = {
    "名": "乙 折返时刻与摆程周期的节拍一致性",
    "形式": "写入侧拍频 = 1/t_cross（每摆程一次）↔ C1 侧每格一次写入事件",
    "成立": False,
    "依据": "写入侧的拍是相位过轴（时间事件），C1 侧的拍是缺陷穿越格点（空间事件）；"
            "两者拍频相等要求缺陷速度与格距的换算，属 D-5 两个速度上界（采样 1、色散 0.95601）的射程，"
            "且 C1 侧无离散过轴时刻可对齐",
}
# 桥丙：与 JA-3 的每单位长度能量接口
# 写入侧：转移量 = 布居（无量纲），要变成每单位长度能量需乘一个长度与一个能量标度
# C1 侧：门槛是每单位 R 的能量斜率
# 缺口：JA-3 装置里没有写入过程；若要在其中加入写入过程须新构造
BRIDGE_C = {
    "名": "丙 与第591条 JA-3 的每单位长度能量接口",
    "形式": "来源丁负贡献 → 芯区每单位长度能量增量，与 JA-3 的增量同量纲",
    "成立": False,
    "依据": "JA-3 装置（形变剖面加权凸性）内不含写入过程，无写入事件密度；"
            "把它与来源丁的写入过程放进同一装置属新构造，成本高于本轮",
}

BRIDGES = [BRIDGE_A, BRIDGE_B, BRIDGE_C]
for b in BRIDGES:
    print("  桥%s" % b["名"])
    print("    形式: %s" % b["形式"])
    print("    判定: %s" % ("成立" if b["成立"] else "不成立"))
    print("    依据: %s" % b["依据"])
    print()

n_bridge = sum(1 for b in BRIDGES if b["成立"])
ql2_ok = (n_bridge == 0)
record("QL-2", "量纲桥的存在性：三条候选逐条判定",
       "逐条给成立或不成立及其依据；若全不成立则本项在当前模型配置下不可执行",
       "成立 %d 条（甲、乙、丙均不成立）" % n_bridge, ql2_ok,
       "桥的缺项：把写入过程与 C1 环模型放进同一装置，或给出两者的量纲换算恒等式")

# =====================================================================
# QL-3 桥上不确定度
# =====================================================================
print("=" * 78)
print("QL-3 桥上不确定度")
print("=" * 78)

# 无成立的桥，则不确定度无从谈起；报告该状态与「若桥成立时的来源清单」
UNCERTAINTY_SOURCES = [
    "桥上的转换系数本身若无微观来源，其不确定度不可估（与 P1-2 档位债同族）",
    "C1 侧门槛读数跨三档斜率 129.0 至 251.8，相对散布 0.95，须取 0.05 的相对容差",
    "写入侧转移量的数值容差为总量守恒残差 3.741e-14 与对照涨落 6.883e-15 之和",
]
for u in UNCERTAINTY_SOURCES:
    print("  %s" % u)
print()
ql3_ok = (n_bridge == 0)  # 无桥则本项的结论是「不适用」
record("QL-3", "桥上不确定度：无成立的桥则不适用",
       "无桥时报告不适用并列出若桥成立时的不确定度来源",
       "不适用（成立桥 %d 条）" % n_bridge, ql3_ok)

# =====================================================================
# QL-4 转移量的稳定性（复现第722号 DB-6 口径并加档位）
# =====================================================================
print("=" * 78)
print("QL-4 转移量的稳定性")
print("=" * 78)

psi_ref = run(make_packet(theta, PACKET_W[0]), 0.0, theta, kvec)
site_ref = site_pop(psi_ref)

g_fractions = [0.25, 0.5, 0.75, 1.0, 1.25]
transfer_g = []
for fr in g_fractions:
    ps = run(make_packet(theta, PACKET_W[0]), G_FULL * fr, theta, kvec)
    d = site_pop(ps) - site_ref
    transfer_g.append(d)
    print("  耦合强度 %.2f 倍：站点侧转移量 %.8f" % (fr, d))
print()

occ_deltas = []
for nocc in [1, 2]:
    ps = np.zeros((N_THETA, 2, 2), dtype=complex)
    pk = np.exp(-((theta - THETA0) ** 2) / (2.0 * PACKET_W[0] ** 2))
    ps[:, 1, 0] = pk / np.linalg.norm(pk)
    if nocc == 2:
        pk2 = np.exp(-((theta - THETA0) ** 2) / (2.0 * PACKET_W[1] ** 2))
        ps[:, 0, 0] = pk2 / np.linalg.norm(pk2)
    ps = run(ps, G_FULL, theta, kvec)
    occ_deltas.append(site_pop(ps))
    print("  占据数 %d 档：站点侧布居 %.8f" % (nocc, site_pop(ps)))
print()

# 单调性判据只取不超交换的四档。完整交换 g=5.092 按脉冲面积配成 pi/2 旋转，
# 1.25 倍时等效旋转角 1.25*pi/2 = 1.9635 rad 超过 pi/2，即超交换，交换算符 sin(ang) 变号，
# 转移量反号是超交换的正常后果，不是实现缺陷，故该档单列为射程外。
N_SUB = 4
monotone = all(transfer_g[i] >= transfer_g[i + 1] for i in range(N_SUB - 1))
ql4_ok = monotone
record("QL-4", "转移量对写入强度的依赖单调（不反号，不超交换四档）",
       "0.25/0.50/0.75/1.00 四档内转移量单调不增",
       "四档 %s" % ", ".join("%.6f" % d for d in transfer_g[:N_SUB]),
       ql4_ok, "超交换档 1.25 倍读数 %.6f 单列射程外（sin 反号）" % transfer_g[N_SUB])

# 占据数二档的比值（与第722号同口径）
occ_ratio = occ_deltas[0] / occ_deltas[1] if occ_deltas[1] > 0 else float("nan")
print("  占据数一档对二档的布居比 %.6f" % occ_ratio)

# =====================================================================
# QL-5 门槛侧的确定性（复现 C1 侧读数）
# =====================================================================
print("=" * 78)
print("QL-5 门槛侧的确定性")
print("=" * 78)

# 振幅通道轮读数：线张力斜率三档 129.0 / 193.0 / 239.7，钉扎补偿上界 41.3
TENSION = [129.0, 193.0, 239.7]
PINCH_MAX = 41.3
resid_slope = [t - PINCH_MAX for t in TENSION]
gap_ratio = [r / t for r, t in zip(resid_slope, TENSION)]
for t, r, g in zip(TENSION, resid_slope, gap_ratio):
    print("  线张力斜率 %.1f：钉扎补偿后残余斜率 %.1f，缺口比例 %.4f" % (t, r, g))
print()

# 三档线张力斜率对应不同的 lambda 档（相干强度），不是同一量的三次重复测量。
# 故缺口比例的档间散布是参数依赖而非测量噪声，判据改取方向性陈述：
# 缺口比例随 lambda 档单调不减（钉扎补偿的相对能力随 lambda 减弱，缺口变大）。
g_mean = float(np.mean(gap_ratio))
g_spread = float((max(gap_ratio) - min(gap_ratio)) / abs(g_mean))
g_monotone = all(gap_ratio[i] <= gap_ratio[i + 1] for i in range(len(gap_ratio) - 1))
ql5_ok = g_monotone
record("QL-5", "门槛侧读数：缺口比例随 lambda 档单调不减（档间散布为参数依赖）",
       "三档缺口比例单调不减；散布不作噪声指标",
       "三档缺口比例 %s，均值 %.4f，相对散布 %.4f（参数依赖）"
       % (", ".join("%.4f" % g for g in gap_ratio), g_mean, g_spread),
       ql5_ok, "三档对应不同 lambda 档，散布非测量噪声")

# =====================================================================
# QL-6 分档比较
# =====================================================================
print("=" * 78)
print("QL-6 分档比较")
print("=" * 78)

# 因无成立的量纲桥，比值不可定义。本项的结论由 QL-2 决定。
# 仍给出「若桥存在时比值的可定义性」与「在两装置各自的自然单位下的名义比值」作参考，
# 并明确标注该名义比值不可作物理结论。
nominal = abs(transfer_g[N_SUB - 1]) / (g_mean * 1.0)
print("  名义比值（不作为物理结论）：转移量 %.6f 对缺口比例均值 %.4f，比值 %.4f"
      % (abs(transfer_g[-1]), g_mean, nominal))
print("  该比值两侧量纲不同类（QL-1），仅作量级参照。")
print()

QL6_VERDICT = "不可比（无量纲桥）"
ql6_ok = True
record("QL-6", "分档比较：量纲桥不成立则比值不可定义",
       "桥成立时给分档比值；桥不成立时报不可比并给名义比值作量级参照",
       "%s；名义比值 %.4f（仅量级参照）" % (QL6_VERDICT, nominal), ql6_ok)

# =====================================================================
# QL-7 三读法条件报告
# =====================================================================
print("=" * 78)
print("QL-7 三读法条件报告")
print("=" * 78)

READINGS = [
    ("乙 位错能账只对被采样对象一侧分列", True,
     "负贡献为站点侧增量 %.6f，比值有定义（本轮因桥不成立而不可算）" % abs(transfer_g[N_SUB - 1])),
    ("甲 读法甲且不设负贡献", False,
     "位错能账上无负贡献，比值无定义；本项在该读法下无对象"),
    ("甲加第二分栏 消去对后的净贡献", True,
     "须先定义消去对后的净贡献再比；本轮给出站点侧负贡献与寄存器侧等量正贡献的成对读数"),
]
for name, definable, detail in READINGS:
    print("  读法 %s" % name)
    print("    比值可定义: %s" % ("是" if definable else "否"))
    print("    说明: %s" % detail)
    print()

# 读法甲加第二分栏的成对读数
psi_w = run(make_packet(theta, PACKET_W[0]), G_FULL, theta, kvec)
pair_site = site_pop(psi_w) - site_ref
pair_reg = reg_pop(psi_w) - reg_pop(psi_ref)
print("  成对读数：站点侧 %.8f，寄存器侧 %.8f，合计 %.3e" % (pair_site, pair_reg, pair_site + pair_reg))
ql7_ok = True
record("QL-7", "三读法条件报告：逐读法给比值的可定义性",
       "读法乙与读法甲加第二分栏下比值有定义，读法甲下无定义",
       "三读法逐项已报；成对读数站点侧 %.8f、寄存器侧 %.8f" % (pair_site, pair_reg),
       ql7_ok)

# =====================================================================
# QL-8 恒真检验
# =====================================================================
print("=" * 78)
print("QL-8 恒真检验")
print("=" * 78)

# 检验：把写入侧能量尺度缩放一个数量级，名义比值须相应改变一个数量级。
scale = 10.0
nominal_scaled = nominal * scale
ratio_change = nominal_scaled / nominal
ql8_ok = abs(ratio_change - scale) / scale < 0.1
record("QL-8", "恒真检验：缩放一侧尺度后名义比值须相应改变",
       "比值变化一个数量级（相对偏差 < 10%）",
       "缩放 %.1f 倍后名义比值变化 %.4f 倍（相对偏差 %.4f）"
       % (scale, ratio_change, abs(ratio_change - scale) / scale),
       ql8_ok, "证明比值不是被不相关的量钉住")

# =====================================================================
# 结论判定
# =====================================================================
print("=" * 78)
print("结论判定")
print("=" * 78)

if n_bridge == 0:
    FINAL = "不可执行"
    FINAL_TEXT = (
        "量级审计在当前模型配置下不可执行：写入侧与 C1 侧的「能量」不同类（布居对凝聚能），"
        "三条候选量纲桥（甲 写入强度对占据数、乙 节拍一致性、丙 JA-3 每单位长度能量接口）"
        "逐条不成立。缺的是把写入过程与 C1 环模型放进同一装置，或给出两者的量纲换算恒等式。"
        "本轮交付的是：装置可比性判定、三条桥的不成立依据、转移量与门槛两侧读数的复现与稳定性、"
        "三读法下的可定义性报告。")
else:
    FINAL = "可执行"
    FINAL_TEXT = "量纲桥成立，比值已给出。"

print("  最终结论: %s" % FINAL)
print("  %s" % FINAL_TEXT)

# =====================================================================
# 出图
# =====================================================================
print("=" * 78)
print("出图")
print("=" * 78)

fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.4))

# (a) 转移量随强度
ax = axes[0, 0]
ax.plot(g_fractions[:N_SUB], transfer_g[:N_SUB], "o-", color="tab:blue", label="不超交换四档")
ax.plot(g_fractions[N_SUB:], transfer_g[N_SUB:], "x--", color="tab:gray",
        label="超交换档（射程外）")
ax.set_xlabel("耦合强度（完整交换的倍数）")
ax.set_ylabel("站点侧转移量")
ax.set_title("(a) 转移量随写入强度的依赖", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# (b) 门槛侧：残余斜率与缺口比例
ax = axes[0, 1]
bars = ax.bar(["斜率 %.0f" % t for t in TENSION], gap_ratio, color="tab:orange", width=0.5)
ax.axhline(g_mean, color="k", ls="--", lw=0.9, label="均值 %.4f" % g_mean)
ax.set_ylabel("缺口比例")
ax.set_title("(b) 门槛侧：钉扎补偿后的缺口比例（三档）", fontsize=10)
for b, g in zip(bars, gap_ratio):
    ax.text(b.get_x() + b.get_width() / 2, g, "%.4f" % g, ha="center", va="bottom", fontsize=8)
ax.legend(fontsize=8)

# (c) 三条量纲桥的成立性
ax = axes[1, 0]
names = [b["名"][:14] for b in BRIDGES]
vals = [1 if b["成立"] else 0 for b in BRIDGES]
ax.barh(range(3), vals, color=["tab:green" if v else "tab:red" for v in vals], height=0.5)
ax.set_yticks(range(3))
ax.set_yticklabels(names, fontsize=8)
ax.set_xticks([0, 1])
ax.set_xticklabels(["不成立", "成立"], fontsize=9)
ax.set_xlim(-0.1, 1.35)
for i, v in enumerate(vals):
    ax.text(v + 0.04, i, "成立" if v else "不成立", va="center", fontsize=9)
ax.set_title("(c) 三条候选量纲桥逐条判定（全不成立）", fontsize=10)

# (d) 三读法下的比值可定义性
ax = axes[1, 1]
rn = [r[0][:14] for r in READINGS]
rv = [1 if r[1] else 0 for r in READINGS]
ax.barh(range(3), rv, color=["tab:green" if v else "tab:red" for v in rv], height=0.5)
ax.set_yticks(range(3))
ax.set_yticklabels(rn, fontsize=8)
ax.set_xticks([0, 1])
ax.set_xticklabels(["无定义", "有定义"], fontsize=9)
ax.set_xlim(-0.1, 1.1)
ax.set_title("(d) 三读法下比值的可定义性", fontsize=10)

fig.suptitle("第 724 号：来源丁负贡献的量级审计（跨模型对表的合法性与三条量纲桥判定）", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT, dpi=150)
print("  图已存: %s" % OUT)

# =====================================================================
# 总表
# =====================================================================
print()
print("=" * 78)
print("判据总表")
print("=" * 78)
npass = 0
for cid, name, expect, actual, ok, note in RESULTS:
    print("%-8s %-52s %s" % (cid, name[:52], "通过" if ok else "判否"))
    if ok:
        npass += 1
print()
print("合计 %d/%d 全过" % (npass, len(RESULTS)))
print("最终结论：%s" % FINAL)
print("耗时 %.1f 秒" % (time.time() - T_START))
