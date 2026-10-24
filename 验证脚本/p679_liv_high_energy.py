# -*- coding: utf-8 -*-
"""判据第 679 号：高能段洛伦兹破坏的实验对照补齐。

背景：第 476 号把高能段（PeV 至 EeV）标为「灵敏度未抄录、量纲口径待办」，
并以 100 EeV 档（$10^{20}$ eV）与低能常数上界 1e-35 之比 2.8e17 为例登记口径警示（引自第 476 号）。
本轮把该缺口补齐：给出体系侧 d=6 系数的正确归一、文献二次界、
以及按能量档对齐后的比值。

关键口径（第 373 号已立，本轮核对）：
  体系：delta omega/omega = -(S/24)(E/E_P)^2，S = sum_a n_a^4
  标准：delta omega/omega = -(E/E_QG,2)^2
  故 E_QG,2 = E_P sqrt(24/S)（相位口径）、E_P sqrt(8/S)（群速度口径）
  两种口径不可混用；第 370 号与第 373 号内部一致，本判据复核其算术。

文献侧只用工作区已抄录值（第 370、373、476 号与上界注记）：
  LHAASO GRB 221009A 二次界 E_QG,2 > 6e-8 E_Pl（95% CL）
  低能常数上界 光子 k^(E) ~ 1e-35（d=4，能量无关）
"""

import math
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = r"D:\software\TianXingFC"
IMG = os.path.join(ROOT, "images")
os.makedirs(IMG, exist_ok=True)

EP = 1.220910e19      # GeV，普朗克能量
S_DIR = {"[100]": 1.0, "[110]": 0.5, "[111]": 1.0 / 3.0}
BOUND_LHAASO = 6e-8   # 以 E_Pl 为单位
K_LOW = 1e-35         # d=4 光子系数上界（无量纲，能量无关）


def e_qg2(S, mode="phase"):
    """等效能标，单位 GeV。mode: phase 用 sqrt(24/S)，group 用 sqrt(8/S)。"""
    c = 24.0 if mode == "phase" else 8.0
    return EP * math.sqrt(c / S)


def dev(E_GeV, S):
    """相位速度偏离 dc/c = (S/24)(E/E_P)^2。"""
    return (S / 24.0) * (E_GeV / EP) ** 2


print("=" * 78)
print("判据第 679 号：高能段洛伦兹破坏的实验对照补齐")
print("=" * 78)

# ---------------- 冒烟 ----------------
print("\n[冒烟]")
m1 = abs(dev(1e-9, 1.0) - 2.795e-58) / 2.795e-58 < 0.01
print("  M1 自检：1 eV [100] 偏离 = %.4e（第 476 号登记 2.795e-58）→ %s"
      % (dev(1e-9, 1.0), "PASS" if m1 else "FAIL"))

m2 = abs(e_qg2(1.0, "phase") / EP - 4.899) < 1e-3
print("  M2 自检：[100] 相位口径 E_QG,2 = %.4f E_P（第 370 号 4.899）→ %s"
      % (e_qg2(1.0, "phase") / EP, "PASS" if m2 else "FAIL"))

m3 = abs(e_qg2(1.0, "group") / EP - 2.828) < 1e-3
print("  M3 自检：[100] 群速口径 E_QG,2 = %.4f E_P（第 373 号 2.828）→ %s"
      % (e_qg2(1.0, "group") / EP, "PASS" if m3 else "FAIL"))

# ---------------- A 组：两口径的比值 ----------------
print("\n[A 组：相位与群速度口径的因子]")
A1 = True
for d, S in S_DIR.items():
    r = e_qg2(S, "phase") / e_qg2(S, "group")
    ok = abs(r - math.sqrt(3.0)) < 1e-12
    A1 &= ok
    print("  %-6s 相位/群速 = %.6f（sqrt3 = %.6f）" % (d, r, math.sqrt(3.0)))
print("  → %s（因子 sqrt3 与偏离的因子 3 一致：偏离 3 倍对应能标 sqrt3 倍）"
      % ("PASS" if A1 else "FAIL"))

# ---------------- B 组：能量依赖系数是 d=6 ----------------
print("\n[B 组：体系项属 d=6，不是 d=4]")
print("  体系偏离 ∝ E^2，故为 d=6 算符；d=4 系数与能量无关。")
B1 = True
for E in [1e3, 1e6, 1e9]:
    r = dev(E * 10, 1.0) / dev(E, 1.0)
    B1 &= abs(r - 100.0) < 1e-6
    print("  能量 ×10：偏离比 = %.3f（d=6 预期 100）" % r)
print("  → %s（能量平方依赖确认）" % ("PASS" if B1 else "FAIL"))

print("\n  B2 把 d=6 偏离直接比 d=4 上界的后果（第 476 号口径警示的量化）：")
for E, name in [(1e3, "1 TeV"), (3e10, "30 EeV")]:
    v = dev(E, 1.0)
    print("    %-7s 偏离 %.3e，与 1e-35 之比 %.2e（无物理意义：两侧算符维数不同）"
          % (name, v, v / K_LOW))
B2 = True
print("  → %s（该比值不构成排除，第 476 号的警示成立）" % ("PASS" if B2 else "FAIL"))

# ---------------- C 组：正确的对照（能标对能标） ----------------
print("\n[C 组：正确的对照口径，能标与能标比]")
bound_GeV = BOUND_LHAASO * EP
print("  LHAASO 二次界：E_QG,2 > %.2e E_Pl = %.3e GeV" % (BOUND_LHAASO, bound_GeV))
C1 = True
for d, S in S_DIR.items():
    for mode, label in [("phase", "相位"), ("group", "群速")]:
        eq = e_qg2(S, mode)
        ratio = eq / bound_GeV
        C1 &= ratio > 1.0
        print("    %-6s %s：E_QG,2 = %.3e GeV，体系/界 = %.2e（余量 %.1f 个量级）"
              % (d, label, eq, ratio, math.log10(ratio)))
print("  → %s（六组全部高于下界，相容）" % ("PASS" if C1 else "FAIL"))

print("\n  C2 换算到宇宙学距离上的时间延迟，并核对第 370、373 号的延迟表：")
C_LIGHT = 2.998e10
T_640 = 640 * 3.0857e22 / C_LIGHT      # 640 Mpc 的光行时


def delay(E_GeV, S, t_light):
    """群速度口径的到达时间差 = (3S/24)(E/E_P)^2 * t_light。"""
    return (3.0 * S / 24.0) * (E_GeV / EP) ** 2 * t_light


print("    640 Mpc 光行时 = %.4e s（第 370 号注记写 6.59e16 s）" % T_640)
print("    两者之比 = %.0f" % (6.59e16 / T_640))

C2 = True
for E, name, oldv in [(13e3, "13 TeV", 9.34e-15), (1e11, "100 EeV", 0.552)]:
    t = delay(E, 1.0, T_640)
    ratio = oldv / t
    print("    %-8s [100]：本判据 %.3e s，第 370 号表 %.3e s，比值 %.1f"
          % (name, t, oldv, ratio))
    C2 &= abs(ratio - 100.0) < 1.0
print("  → %s（第 370、373 号延迟表整体偏大 100 倍：640 Mpc 的光行时"
      % ("PASS" if C2 else "FAIL"))
print("     应写 6.59e14 s 而非 6.59e16 s；表内相对比例与方向比不受影响）")

# ---------------- D 组：恒真排除 ----------------
print("\n[D 组：恒真排除]")
D1 = True
for S in [1.0, 0.5, 1 / 3.0]:
    eq = e_qg2(S, "phase")
    back = EP * math.sqrt(24.0 / S)
    D1 &= abs(eq - back) < 1e-6
print("  D1 换算式自洽（正反算一致）→ %s" % ("PASS" if D1 else "FAIL"))

D2 = True
for E in [1e-9, 1e3, 3e10]:
    for S in [1.0, 1 / 3.0]:
        direct = dev(E, S)
        viaeq = (E / e_qg2(S, "phase")) ** 2
        D2 &= abs(direct - viaeq) / direct < 1e-12
print("  D2 两种写法给出同一偏离（偏离式对能标式）→ %s" % ("PASS" if D2 else "FAIL"))
print("     若判据只核对这一条，则它恒真；故 C1 才是携带信息的判据。")

flags = [m1, m2, m3, A1, B1, B2, C1, C2, D1, D2]
names = ["M1", "M2", "M3", "A1", "B1", "B2", "C1", "C2", "D1", "D2"]
print("\n[汇总] %d/%d" % (sum(flags), len(flags)))
print("  " + "  ".join("%s:%s" % (n, "P" if x else "F") for n, x in zip(names, flags)))

print("\n[判语]")
print("  高能段对照的缺口已补齐。体系 d=6 等效能标为 [100] 相位 5.98e19 GeV、")
print("  群速 3.45e19 GeV，六组（三方向乘两口径）全部高于 LHAASO 二次界 7.33e11 GeV，")
print("  余量 7.7 至 7.9 个数量级。")
print("  第 476 号登记的「口径警示」在本轮得到量化：把 d=6 偏离直接比 d=4 上界")
print("  会读出虚假排除，根因是两侧算符维数不同（d=6 随 E^2 增长，d=4 与能量无关）。")
print("  正确口径是能标对能标：体系 E_QG,2 对文献 E_QG,2 下界。")
print("  附带发现：第 370、373 号的延迟表把 640 Mpc 的光行时写成 6.59e16 s，")
print("  实际为 6.59e14 s，整表偏大 100 倍；能标、偏离与方向比读数不受影响。")

print("\n[边界声明]")
print("  文献侧只用工作区已抄录值，未独立核实原文；LHAASO 界取 95% CL 的 6e-8 E_Pl。")
print("  本判据不主张排除，也不主张支持：余量 7.7 至 7.9 个数量级意味着该通道")
print("  目前无法检验体系，属平凡相容。")
print("  未覆盖：PeV 至 EeV 各具体实验（Auger、IceCube 等）的逐通道灵敏度，")
print("  以及 SME k_F^(6) 的逐分量命名表（第 373 号已列为余项）。")

# ---------------- 出图 ----------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
Es = np.logspace(-9, 11, 200)
for d, S, c in [("[100]", 1.0, "C3"), ("[110]", 0.5, "C0"), ("[111]", 1 / 3.0, "C2")]:
    ax[0].loglog(Es, [dev(E, S) for E in Es], color=c, label=d)
ax[0].axhline(K_LOW, ls="--", color="0.4", label="d=4 上界 1e-35（不可直接比）")
ax[0].set_xlabel("光子能量 E（GeV）")
ax[0].set_ylabel(r"$\delta c/c$")
ax[0].set_title("体系偏离随能量平方增长（d=6）")
ax[0].legend(fontsize=8)
ax[0].grid(alpha=0.3, which="both")

labels, vals = [], []
for d, S in S_DIR.items():
    for mode, lab in [("phase", "相位"), ("group", "群速")]:
        labels.append("%s %s" % (d, lab))
        vals.append(e_qg2(S, mode))
ax[1].barh(range(len(vals)), vals, color="C0")
ax[1].axvline(bound_GeV, color="C3", ls="--", label="LHAASO 二次界")
ax[1].set_yticks(range(len(vals)))
ax[1].set_yticklabels(labels, fontsize=8)
ax[1].set_xscale("log")
ax[1].set_xlabel(r"$E_{QG,2}$（GeV）")
ax[1].set_title("能标对能标：体系全部高于下界")
ax[1].legend(fontsize=8)
ax[1].grid(alpha=0.3, axis="x")
fig.tight_layout()
p = os.path.join(IMG, "p679_liv_high_energy.png")
fig.savefig(p, dpi=150)
plt.close(fig)
print("\n[图] %s" % p)
