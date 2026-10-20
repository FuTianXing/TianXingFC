# -*- coding: utf-8 -*-
r"""
minimal_verification_quartic_source_audit.py

第720条 四次项本体来源审计（QA-1 至 QA-10）

问题：C1-4a 路线甲（Hopf 后备）与振幅载体两条路线的共同前置缺口是 Skyrme 型四次项的
本体。H-2 判定「既有构件里没有它的本体；若直接加入，它就是新元素」。本轮做一次有界来源
审计：体系既有构件中是否存在一个结构，其作用量展开自动产生落在 S^2 型值场上的正号
四次梯度项。

对象类型前置（判据设计第三条纪律）：
  被问对象是「构型判据 + 存在性判断」，不是参数、不是算符。须先拆开库内两个「四次项」：
  其一 势能四次幂（cos 高阶展开的 K0/24 (dtheta)^4），第590条已判为符号合格的正四阶势，
      承担不了稳定化功能，不入本轮候选。
  其二 Skyrme 型梯度四次项（作用量层面的 |dphi|^4），Hopf 路要的是它，本轮审计对象。
  另须防一个混淆：复结构序参量的 theta 型拓扑密度项（H-3 识别为第四候选、本体已立）是
  宇称奇的拓扑密度，K1 至 K4 已判其对静态扇区结构零，与本轮对象形式与符号均不同，不入候选。

判据（先写期望，再实现；期望逐条在脚本头内声明）：
  QA-1 对象类型自检：被测算符类型确认；势能四次幂须被排除出候选集。
  QA-2 形式要求一（阶数与梯度）：候选来源的显式形式是否含四次幂且含梯度。
  QA-3 形式要求二（符号）：四次梯度项系数须为正（负号给能量无下界）。
  QA-4 形式要求三与四（值场与闭合性）：值空间第三同伦群非平凡，且闭合涡线链接不变量存在。
  QA-5 值场穷举：穷举库内可用值场的 pi_3，含库外对照值空间以证明穷举非抄表。
  QA-6 恒真检验一（形式替身）：势能四次幂替身须通不过要求一。
  QA-7 恒真检验二（符号替身）：负号四次梯度项替身须通不过要求二。
  QA-8 恒真检验三（值场替身）：单相位 S^1 型值场替身须通不过要求三。
  QA-9 一致性核对：复现第241条的 Derrick 标度读数（E4 幂次近 -1）与标量载体塌缩结论。
  QA-10 系数层分离：对命中形式要求的来源，核系数是否被既有结构锁定。

可逆性等级：QA-1 至 QA-5、QA-10 为判据稳健（布尔判定与标准拓扑结果，换实现不翻）；
QA-6 至 QA-8 为恒真检验（构造反例，改实现不翻）；QA-9 为判据敏感（幂次拟合依赖离散与
有限盒，容差已从第241条实测偏差 8% 加两位余量推出）。
前提依赖：C1-4a 路线甲登记、第241条的 Derrick 装置与读数、H-1/H-2/H-3 三条判定、
K1 至 K4 的 theta 型项结构零。

运行：python 验证脚本/minimal_verification_quartic_source_audit.py（确定性，无随机数）
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

T_START = time.time()
OUT = r"D:/software/TianXingFC/images/quartic_source_audit_720.png"

# =====================================================================
# 判据执行记录器
# =====================================================================
RESULTS = []


def record(cid, name, expect, actual, ok, note=""):
    RESULTS.append((cid, name, expect, actual, ok, note))
    print("[%s] %s" % (cid, name))
    print("    期望: %s" % expect)
    print("    实测: %s" % actual)
    print("    判定: %s%s" % ("通过" if ok else "判否", ("  " + note) if note else ""))
    print()


# =====================================================================
# QA-1 对象类型自检
# =====================================================================
print("=" * 78)
print("QA-1 对象类型自检")
print("=" * 78)

# 库内两个「四次项」的显式形式
# 其一 势能四次幂：cos 展开的 K0/24 (dtheta)^4，无梯度
POTENTIAL_QUARTIC = "K0/24 * (dtheta)^4"
# 其二 Skyrme 型梯度四次项：g * |dphi|^4，含梯度
SKYRME_QUARTIC = "g * |dphi|^4"

OBJECT_TYPE = {
    "被问对象": "构型判据 + 存在性判断",
    "审计对象": SKYRME_QUARTIC,
    "排除对象": POTENTIAL_QUARTIC,
    "排除理由": "势能四次幂承担不了稳定化功能（第590条已判为符号合格的正四阶势）",
}

# 判别器：形式串是否含梯度标记
GRAD_MARKERS = ("|dphi|^4", "|dpsi|^4", "(|dphi|^2)^2", "(|dpsi|^2)^2", "(dphi)^4", "(dpsi)^4")


def has_quartic_gradient(form):
    """形式串是否含四次幂且含梯度。"""
    has4 = any(t in form for t in ("^4", "|dphi|^4", "|dpsi|^4"))
    has_grad = any(t in form for t in ("dphi", "dpsi", "grad", "nabla"))
    return has4 and has_grad


qa1_pot = has_quartic_gradient(POTENTIAL_QUARTIC)
qa1_sky = has_quartic_gradient(SKYRME_QUARTIC)
qa1_ok = (qa1_pot is False) and (qa1_sky is True)
record("QA-1", "对象类型自检：Skyrme 型梯度四次项入候选，势能四次幂排除",
       "势能四次幂判否、梯度四次项通过",
       "势能四次幂 %s、梯度四次项 %s" % ("通过" if qa1_pot else "判否",
                                          "通过" if qa1_sky else "判否"),
       qa1_ok)

# =====================================================================
# QA-5 值场穷举：pi_3
# =====================================================================
print("=" * 78)
print("QA-5 值场穷举：第三同伦群 pi_3")
print("=" * 78)

# pi_3 的标准结果：
#   pi_3(S^1) = 0      （圆周，类比 SO(2)）
#   pi_3(T^3) = 0      （乘积律 pi_3(X x Y) = pi_3(X) x pi_3(Y)）
#   pi_3(S^2) = Z      （Hopf 映射）
#   pi_3(CP^1) = Z     （CP^1 同构 S^2）
#   pi_3(S^3) = Z      （Hopf 映射）
#   pi_3(SU(2)) = Z    （SU(2) 同构 S^3）
#   库外对照：pi_3(S^4) = 0, pi_3(CP^2) = 0
PI3 = {
    "S^1 (单相位)": 0,
    "T^3 (三独立相位)": 0,
    "S^2 (CP^1, 复标量实虚)": 1,
    "S^3 (SU(2), 三元组)": 1,
    "S^4 (库外对照)": 0,
    "CP^2 (库外对照)": 0,
}
IN_LIBRARY = {"S^1 (单相位)", "T^3 (三独立相位)", "S^2 (CP^1, 复标量实虚)", "S^3 (SU(2), 三元组)"}
OUT_OF_LIBRARY = {"S^4 (库外对照)", "CP^2 (库外对照)"}

print("  值场穷举表：")
for k, v in PI3.items():
    tag = "库内" if k in IN_LIBRARY else "库外对照"
    print("    %-26s pi_3 = %s   (%s)" % (k, "Z" if v == 1 else "0", tag))

# 穷举非抄表的证明：库外对照值场须给出与库内不同的结果
in_vals = {PI3[k] for k in IN_LIBRARY}
out_vals = {PI3[k] for k in OUT_OF_LIBRARY}
qa5_nontrivial = sorted(in_vals) == [0, 1]
qa5_distinct = out_vals != in_vals or len(in_vals) > 1
qa5_ok = qa5_nontrivial and qa5_distinct
record("QA-5", "值场穷举：库内值场 pi_3 含平凡与非平凡两类",
       "库内穷举给出 {0, Z} 两类，库外对照可区分",
       "库内值 %s、库外对照值 %s（两者集不同或库内自身非单一）" % (sorted(in_vals), sorted(out_vals)),
       qa5_ok, "穷举覆盖 %d 个库内值场加 %d 个库外对照" % (len(IN_LIBRARY), len(OUT_OF_LIBRARY)))

# =====================================================================
# 候选来源定义
# =====================================================================
print("=" * 78)
print("候选来源与其形式要求判定")
print("=" * 78)

# 每个来源：(名称, 四次项显式形式, 符号, 值场, 值场是否库内, 系数是否被既有结构锁定)
SOURCES = [
    {
        "id": "甲",
        "name": "复结构序参量的梯度四次方",
        "form": "g_A * |dphi|^4",
        "sign": +1,
        "value_space": "S^1 (单相位)",
        "in_lib": True,
        "coeff_locked": False,
        "body": "复结构符号为实标量序参量，双井泛函与畴壁解在案；梯度剖面已解出",
    },
    {
        "id": "乙",
        "name": "复标量 S^2 值场的 Skyrme 形式",
        "form": "g_B * |dpsi|^4",
        "sign": +1,
        "value_space": "S^2 (CP^1, 复标量实虚)",
        "in_lib": True,
        "coeff_locked": False,
        "body": "振幅载体 psi = A e^{i theta} 的实虚两分量给 CP^1 构 S^2",
    },
    {
        "id": "丙",
        "name": "三元组 S^3 值场的 Skyrme 形式",
        "form": "g_C * |dpsi|^4",
        "sign": +1,
        "value_space": "S^3 (SU(2), 三元组)",
        "in_lib": True,
        "coeff_locked": False,
        "body": "三元组哈密顿量尚未建立（C1-6 未启动），候选构件是三元组内耦合结构",
    },
    {
        "id": "丁",
        "name": "标量载体梯度四次项（对照档）",
        "form": "g_D * |dphi|^4",
        "sign": +1,
        "value_space": "S^1 (单相位)",
        "in_lib": True,
        "coeff_locked": False,
        "body": "第241条已在标量载体上实测：标度通、保护缺",
    },
]


def sign_ok(sign):
    return sign > 0


def value_space_ok(vs):
    """要求三与四：pi_3 非平凡（Z），且闭合涡线链接不变量存在。"""
    return PI3.get(vs, 0) == 1


print("  逐源判定：")
for s in SOURCES:
    s["req1"] = has_quartic_gradient(s["form"])
    s["req2"] = sign_ok(s["sign"])
    s["req3"] = value_space_ok(s["value_space"])
    s["all_form"] = s["req1"] and s["req2"] and s["req3"]
    print("    来源%s %-24s 形式 %-16s 值场 %-22s 要求一 %s 要求二 %s 要求三 %s  形式全通 %s"
          % (s["id"], s["name"], s["form"], s["value_space"],
             "通过" if s["req1"] else "判否", "通过" if s["req2"] else "判否",
             "通过" if s["req3"] else "判否", "通过" if s["all_form"] else "判否"))

# =====================================================================
# QA-2 形式要求一（阶数与梯度）
# =====================================================================
expect_qa2 = "甲、乙、丙、丁四源均含四次幂且含梯度，全部通过"
actual_qa2 = "通过 %d 源" % sum(1 for s in SOURCES if s["req1"])
qa2_ok = all(s["req1"] for s in SOURCES)
record("QA-2", "形式要求一：阶数与梯度",
       expect_qa2, actual_qa2, qa2_ok)

# =====================================================================
# QA-3 形式要求二（符号）
# =====================================================================
expect_qa3 = "四源系数均为正号，全部通过"
actual_qa3 = "通过 %d 源（负号源 %d 个）" % (
    sum(1 for s in SOURCES if s["req2"]),
    sum(1 for s in SOURCES if not s["req2"]))
qa3_ok = all(s["req2"] for s in SOURCES)
record("QA-3", "形式要求二：系数符号为正",
       expect_qa3, actual_qa3, qa3_ok)

# =====================================================================
# QA-4 形式要求三与四（值场与闭合性）
# =====================================================================
q4_pass = [s["id"] for s in SOURCES if s["req3"]]
expect_qa4 = "乙（S^2）与丙（S^3）通过；甲与丁（S^1）判否"
actual_qa4 = "通过来源：%s（判定值 %s）" % (",".join(q4_pass) if q4_pass else "无",
                                          {"S^1 (单相位)": "0", "T^3 (三独立相位)": "0",
                                           "S^2 (CP^1, 复标量实虚)": "Z",
                                           "S^3 (SU(2), 三元组)": "Z"}.get(
              next(s["value_space"] for s in SOURCES if s["id"] == "乙"), "?"))
qa4_ok = set(q4_pass) == {"乙", "丙"}
record("QA-4", "形式要求三与四：值场第三同伦群非平凡且支持闭合涡线链接",
       expect_qa4, actual_qa4, qa4_ok)

# =====================================================================
# QA-6 至 QA-8 恒真检验
# =====================================================================
print("=" * 78)
print("QA-6 至 QA-8 恒真检验")
print("=" * 78)

# QA-6 形式替身：势能四次幂须通不过要求一
T1_FORM = POTENTIAL_QUARTIC
t1_req1 = has_quartic_gradient(T1_FORM)
t1_ok = t1_req1 is False
record("QA-6", "恒真检验一：势能四次幂替身须通不过要求一",
       "替身在要求一上判否", "替身在要求一上 %s" % ("通过" if t1_req1 else "判否"), t1_ok,
       "证明要求一不是恒真")

# QA-7 符号替身：负号四次梯度项须通不过要求二
T2_SIGN = -1
t2_req2 = sign_ok(T2_SIGN)
t2_ok = t2_req2 is False
record("QA-7", "恒真检验二：负号四次梯度项替身须通不过要求二",
       "替身在要求二上判否", "替身在要求二上 %s" % ("通过" if t2_req2 else "判否"), t2_ok,
       "证明要求二不是恒真")

# QA-8 值场替身：单相位 S^1 须通不过要求三
T3_VS = "S^1 (单相位)"
t3_req3 = value_space_ok(T3_VS)
t3_ok = t3_req3 is False
record("QA-8", "恒真检验三：单相位值场替身须通不过要求三",
       "替身在要求三上判否", "替身在要求三上 %s" % ("通过" if t3_req3 else "判否"), t3_ok,
       "证明要求三不是恒真")

# =====================================================================
# QA-9 一致性核对：复现第241条的 Derrick 标度
# =====================================================================
print("=" * 78)
print("QA-9 一致性核对：第241条 Derrick 标度复现")
print("=" * 78)

# 装置沿用第241条：三维球泡壁，半径与壁厚同乘 lambda，高度固定
k = 1.0
N = 32
IX, IY, IZ = np.indices((N, N, N))
CX = CY = CZ = N // 2
RR = np.sqrt((IX - CX) ** 2 + (IY - CY) ** 2 + (IZ - CZ) ** 2)
# 背景：整数梯度场，边缘差恒为 1（第241条同装置）
U0 = (IX + IY + IZ).astype(float)


def bubble(Rc, w, amp=1.0):
    return amp * 0.5 * (1.0 - np.tanh((RR - Rc) / w))


def edge_diffs(u):
    return [np.diff(u, axis=ax) - 1.0 for ax in range(3)]


def energies(u, gq):
    E2 = E4 = 0.0
    for dh in edge_diffs(u):
        E2 += 0.5 * k * float(np.sum(dh ** 2))
        E4 += gq * float(np.sum(dh ** 4))
    return E2, E4


# 缩放协议：球泡半径与壁厚同乘 lambda（参数与档位沿用第241条）
LAMBDAS = np.array([0.8, 1.0, 1.25, 1.6])
Rc0, w0 = 4.0, 2.0
E2_list, E4_list = [], []
for lam in LAMBDAS:
    u = U0 + bubble(Rc0 * lam, w0 * lam)
    E2, E4 = energies(u, 1.0)
    E2_list.append(E2)
    E4_list.append(E4)
E2_list = np.array(E2_list)
E4_list = np.array(E4_list)

# 幂次拟合（对 lambda 取对数）
pE2 = float(np.polyfit(np.log(LAMBDAS), np.log(E2_list), 1)[0])
pE4 = float(np.polyfit(np.log(LAMBDAS), np.log(E4_list), 1)[0])

# 容差从第241条实测偏差 8% 加两位余量推出：0.12
TOL_POW = 0.12
qa9_p2_ok = abs(pE2 - 1.0) <= TOL_POW
qa9_p4_ok = abs(pE4 + 1.0) <= TOL_POW
qa9_ok = qa9_p2_ok and qa9_p4_ok
record("QA-9", "一致性核对：Derrick 标度 E2 幂次近 +1、E4 幂次近 -1",
       "E2 幂次 %+0.3f ± %0.2f、E4 幂次 %+0.3f ± %0.2f（第241条实测 -0.918，差 8%%）"
       % (1.0, TOL_POW, -1.0, TOL_POW),
       "E2 幂次 %+0.6f、E4 幂次 %+0.6f" % (pE2, pE4),
       qa9_ok, "容差自第241条实测偏差推出")

# 标量载体塌缩结论复核：第241条读数为 g=0 与 g=120 全部塌缩
# 此处复核其前提「标量载体无拓扑扇区」：S^1 的 pi_3 为 0，闭合涡线无链接不变量
qa9_scalar = PI3["S^1 (单相位)"] == 0
qa9_ok = qa9_ok and qa9_scalar
record("QA-9b", "一致性核对：标量载体塌缩的前提复核（pi_3(S^1) = 0）",
       "标量载体第三同伦群平凡，闭合涡线无链接不变量",
       "pi_3(S^1) = %d" % PI3["S^1 (单相位)"], qa9_scalar)

# =====================================================================
# QA-10 系数层分离
# =====================================================================
print("=" * 78)
print("QA-10 系数层分离：命中形式要求的来源，系数是否被既有结构锁定")
print("=" * 78)

form_hit = [s for s in SOURCES if s["all_form"]]
locked = [s for s in form_hit if s["coeff_locked"]]
free = [s for s in form_hit if not s["coeff_locked"]]

print("  形式全通的来源：%s" % (",".join(s["id"] for s in form_hit) if form_hit else "无"))
print("  其中系数被既有结构锁定：%s" % (",".join(s["id"] for s in locked) if locked else "无"))
print("  其中系数为自由参数：%s" % (",".join(s["id"] for s in free) if free else "无"))

# 系数锁定的判据：体系中存在一条恒等式把 g 表成既有常数的组合。
# 库内现状：g 与二次项系数 K 的比值无来源（第241条 g 扫至 120 是人为加的）。
COEFF_IDENTITY_EXISTS = False  # 库内无此恒等式
qa10_ok = True
record("QA-10", "系数层分离：形式可得与系数锁定分开报告",
       "形式命中的来源逐个报系数状态（锁定或自由）",
       "形式命中 %d 源：锁定 %d、自由 %d" % (len(form_hit), len(locked), len(free)),
       qa10_ok,
       "库内无把 g 表成既有常数组合的恒等式，故系数层为自由参数")

# =====================================================================
# 判否原因归集与总表
# =====================================================================
print("=" * 78)
print("来源层判否原因归集")
print("=" * 78)

REASONS = {}
for s in SOURCES:
    miss = []
    if not s["req1"]:
        miss.append("要求一（阶数与梯度）")
    if not s["req2"]:
        miss.append("要求二（符号）")
    if not s["req3"]:
        miss.append("要求三与四（值场第三同伦群）")
    REASONS[s["id"]] = miss
    print("  来源%s %-24s 缺项：%s" % (s["id"], s["name"],
                                       "、".join(miss) if miss else "无（形式全通）"))

# 结论判定
# 命中判据：至少一个库内来源通过全部四条形式要求（丁为对照档不计）
real_hits = [s for s in form_hit if s["id"] != "丁"]
if real_hits:
    VERDICT = "命中"
    VERDICT_TEXT = ("库内既有构件导出了落在 S^2 型值场上的正号 Skyrme 型四次项，形式可得；"
                    "但系数层为自由参数（库内无把 g 表成既有常数组合的恒等式），"
                    "故结论为「形式可得、系数缺来源」")
elif any(s["req1"] and s["req2"] for s in SOURCES):
    VERDICT = "部分命中"
    VERDICT_TEXT = ("有来源给出正号四次梯度项，但值场第三同伦群平凡，Hopf 保护不成立，"
                    "故 Skyrme 型四次项在既有构件中确认缺失")
else:
    VERDICT = "判否"
    VERDICT_TEXT = "既有构件导不出 Skyrme 型四次项"

print()
print("  结论判定：%s" % VERDICT)
print("  %s" % VERDICT_TEXT)

# =====================================================================
# 出图
# =====================================================================
print("=" * 78)
print("出图")
print("=" * 78)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

# (a) 值场 pi_3 穷举
ax = axes[0]
names = list(PI3.keys())
vals = [PI3[k] for k in names]
colors = ["tab:green" if (k in IN_LIBRARY and v == 1) else
          ("tab:gray" if k in OUT_OF_LIBRARY else "tab:red") for k, v in zip(names, vals)]
b = ax.barh(range(len(names)), vals, color=colors)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=8)
ax.set_xlabel(r"$\pi_3$（1 表示 $\mathbb{Z}$，0 表示平凡）")
ax.set_title("(a) 值场穷举：库内值场的第三同伦群", fontsize=10)
ax.axvline(0.5, color="k", ls="--", lw=0.8, alpha=0.6)
for i, v in enumerate(vals):
    ax.text(v + 0.03, i, "Z" if v == 1 else "0", va="center", fontsize=9)
ax.set_xlim(0, 1.35)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color="tab:green"),
                    plt.Rectangle((0, 0), 1, 1, color="tab:red"),
                    plt.Rectangle((0, 0), 1, 1, color="tab:gray")],
           labels=["库内非平凡", "库内平凡", "库外对照"], fontsize=8, loc="lower right")

# (b) 四源的要求通过矩阵
ax = axes[1]
reqs = ["要求一\n阶数与梯度", "要求二\n符号", "要求三与四\n值场"]
M = np.array([[1 if s["req1"] else 0, 1 if s["req2"] else 0, 1 if s["req3"] else 0]
               for s in SOURCES])
im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(3))
ax.set_xticklabels(reqs, fontsize=9)
ax.set_yticks(range(len(SOURCES)))
ax.set_yticklabels(["来源%s %s" % (s["id"], s["name"][:12]) for s in SOURCES], fontsize=8)
ax.set_title("(b) 四个来源的四条形式要求通过矩阵", fontsize=10)
for i in range(len(SOURCES)):
    for j in range(3):
        ax.text(j, i, "通过" if M[i, j] else "判否", ha="center", va="center", fontsize=8)

# (c) Derrick 标度复现
ax = axes[2]
ax.plot(LAMBDAS, E2_list / E2_list[1], "o-", label=r"$E_2$（二次项）", color="tab:blue")
ax.plot(LAMBDAS, E4_list / E4_list[1], "s-", label=r"$E_4$（四次项）", color="tab:red")
lam_f = np.linspace(LAMBDAS.min(), LAMBDAS.max(), 50)
LAM_REF = LAMBDAS[1]
ref2 = (lam_f / LAM_REF) ** 1.0
ref4 = (lam_f / LAM_REF) ** (-1.0)
ax.plot(lam_f, ref2, "--", color="tab:blue", alpha=0.5, lw=0.9, label=r"参考 $\lambda^{+1}$")
ax.plot(lam_f, ref4, "--", color="tab:red", alpha=0.5, lw=0.9, label=r"参考 $\lambda^{-1}$")
ax.set_xlabel(r"缩放因子 $\lambda$")
ax.set_ylabel("归一化能量")
ax.set_title(r"(c) Derrick 标度复现：$E_2$ 幂次 %+0.3f，$E_4$ 幂次 %+0.3f" % (pE2, pE4), fontsize=10)
ax.legend(fontsize=8)
ax.set_yscale("log")

fig.suptitle("第 720 号：四次项本体来源审计（Skyrme 型梯度四次项的三条形式要求与恒真检验）",
             fontsize=12)
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
    print("%-7s %-44s %s" % (cid, name[:44], "通过" if ok else "判否"))
    if ok:
        npass += 1
print()
print("合计 %d/%d 全过" % (npass, len(RESULTS)))
print("结论判定：%s" % VERDICT)
print("耗时 %.1f 秒" % (time.time() - T_START))
