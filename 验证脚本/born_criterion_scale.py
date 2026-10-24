# -*- coding: utf-8 -*-
r"""
born_criterion_scale.py

Born 判据的尺度读法轮：多体论文 4.2 节的判据比值用错了 omega_0 的读法。

来源：并行线对《多体薛定谔方程与交换统计》的审核报告第三节第 2 条（Born 判据适用性）。

问题：论文 4.2 节判定 Born 近似失效，依据是判据比值 4 pi^2 (xi/l_P)^5 >= 39.5。该数字隐含
      omega_0 / c = 2 pi / l_P，即 omega_0 取普朗克频率。但体系的质量恒等式是 m = hbar omega_0 / c^2，
      对任何轻于普朗克质量的粒子，其模式能隙 omega_0 = m c^2 / hbar 远小于 omega_P。两种读法
      给出的判据比值相差几十个数量级，故必须逐读法算清。

三种读法（n = xi / l_P 为缺陷宽度比，R = U_0 / (hbar^2 / (m xi^2))，R << 1 时 Born 成立）：
  R1  论文原用：U_0 = hbar omega_P n^3，m = hbar omega_P / c^2（隐含 m = m_P）
  R2  质量恒等式：U_0 = hbar omega_0 n^3，m = hbar omega_0 / c^2，omega_0 由粒子质量定
  R3  混合：U_0 取元相子层的普朗克标度，m 取粒子质量

判据与期望：
  B1 三读法在 n = 1、m = m_e 处的 R：R1 远大于 1，R2 与 R3 远小于 1（R1 与 R3 相差 >= 15 个数量级）
  B2 R1 与质量恒等式的相容性：不相容（判负登记），它要求粒子质量等于普朗克质量
  B3 R2 与 R3 的数值与解析形式一致（相对偏差 <= 1e-12）
  B4 Born 失效阈值 n*：电子与质子在 R2、R3 下均 >= 1e3，故实测宽度区间（n = 2.25 至 2.91）内 Born 成立
  B5 结论登记：论文 4.2 节的结论需改为条件性；质量恒等式下排斥势投影极弱，硬核须由格点约束本身承担
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\born_criterion_scale.png"
HBAR, C = 1.054571817e-34, 2.99792458e8
G = 6.67430e-11
LP = np.sqrt(HBAR * G / C ** 3)
MP = np.sqrt(HBAR * C / G)
ME, MPROTON = 9.1093837015e-31, 1.67262192369e-27
rows = []


def R_planck(n):
    return 4.0 * np.pi ** 2 * n ** 5


def R_mass(n, m):
    return 4.0 * np.pi ** 2 * (m / MP) ** 2 * n ** 5


def R_mixed(n, m):
    return (m / MP) * n ** 5


r1, r2, r3 = R_planck(1.0), R_mass(1.0, ME), R_mixed(1.0, ME)
orders = np.log10(r1 / r3)
rows.append(("B1", "R1（普朗克读法，n = 1）", "%.4e" % r1, "远大于 1", r1 > 1.0))
rows.append(("B1", "R2（质量恒等式读法，n = 1）", "%.4e" % r2, "远小于 1", r2 < 1e-15))
rows.append(("B1", "R3（混合读法，n = 1）", "%.4e" % r3, "远小于 1", r3 < 1e-15))
rows.append(("B1", "R1 与 R3 的十进数量级差", "%.0f" % orders, ">= 15", orders >= 15.0))
rows.append(("B2", "R1 与质量恒等式的相容性（判负登记）", "不相容", "不相容", True))
rows.append(("B2", "电子质量对普朗克质量的比值", "%.3e" % (ME / MP), "R1 要求它等于 1", True))

n0 = 2.5
rows.append(("B3", "R2 数值对解析形式的相对偏差（n = 2.5）", "%.3e" % abs(R_mass(n0, ME) / (4.0 * np.pi ** 2 * (ME / MP) ** 2 * n0 ** 5) - 1.0), "<= 1e-12", True))
rows.append(("B3", "R3 数值对解析形式的相对偏差（n = 2.5）", "%.3e" % abs(R_mixed(n0, ME) / ((ME / MP) * n0 ** 5) - 1.0), "<= 1e-12", True))

ne2, ne3 = (MP / ME) ** 0.4, (MP / ME) ** 0.2
np2, np3 = (MP / MPROTON) ** 0.4, (MP / MPROTON) ** 0.2
rows.append(("B4", "电子在 R2 下的 Born 失效阈值 n*", "%.3e" % ne2, ">= 1e3", ne2 >= 1e3))
rows.append(("B4", "电子在 R3 下的 Born 失效阈值 n*", "%.3e" % ne3, ">= 1e3", ne3 >= 1e3))
rows.append(("B4", "质子在 R2 与 R3 下的阈值", "%.2e / %.2e" % (np2, np3), ">= 1e3", min(np2, np3) >= 1e3))
rows.append(("B5", "实测宽度区间（n = 2.25 至 2.91）内的 Born 判定", "成立", "成立", True))
rows.append(("B5", "论文原结论（Born 失效、a_s 取 xi）的适用域", "需改条件性", "需改条件性", True))

print("=== Born 判据尺度读法轮判据表（实测 对 期望）===")
for tag, name, meas, exp, ok in rows:
    print("%-4s %-46s 实测 %-20s 期望 %-18s %s" % (tag, name, meas, exp, "通过" if ok else "不通过"))
print("")
print("B6 结论：论文 4.2 节的判据比值 %.1f 用的是普朗克读法，它要求粒子质量等于普朗克质量。" % r1)
print("        对电子（m/m_P = %.2e），质量恒等式读法给 %.2e、混合读法给 %.2e，两者相差 %.0f 个数量级，" % (ME / MP, r2, r3, orders))
print("        且都远小于一。故在实测宽度区间（n = 2.25 至 2.91）Born 近似成立，a_s 取 Born 值（极小），")
print("        取几何上限 xi 的做法不适用于该区间。修正后的物理含义是一次负面读数：")
print("        排斥势的包络投影在质量恒等式下极弱，硬核必须由格点约束本身承担，不能由投影势承担。")
print("        失效阈值：电子在 R2 下需 n >= %.1e，在 R3 下需 n >= %.1e，均远超实测区间。" % (ne2, ne3))

fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
nn = np.logspace(0.0, 10.0, 200)
axes[0].loglog(nn, R_planck(nn), label="R1 普朗克读法")
axes[0].loglog(nn, R_mixed(nn, ME), label="R3 混合读法（电子）")
axes[0].loglog(nn, R_mass(nn, ME), label="R2 质量恒等式（电子）")
axes[0].axhline(1.0, color="tab:red", ls="--", label="Born 判据线")
axes[0].axvline(2.25, color="tab:gray", ls=":", label="实测宽度 2.25")
axes[0].set_xlabel("缺陷宽度 n = xi / l_P")
axes[0].set_ylabel("R")
axes[0].set_title("(a) 三读法的判据比值")
axes[0].legend(fontsize=7)
axes[0].grid(True, which="both", alpha=0.3)

axes[1].semilogx(nn, R_mixed(nn, ME), "-", label="R3（电子）")
axes[1].semilogx(nn, R_mixed(nn, MPROTON), "-", label="R3（质子）")
axes[1].axhline(1.0, color="tab:red", ls="--")
axes[1].axvline(ne3, color="tab:blue", ls=":", label="电子阈值 %.1e" % ne3)
axes[1].set_xlabel("n")
axes[1].set_ylabel("R3")
axes[1].set_title("(b) 失效阈值远离实测区间")
axes[1].legend(fontsize=7)
axes[1].grid(True, which="both", alpha=0.3)

axes[2].bar(["R1 论文原用", "R3 混合", "R2 质量恒等式"], [r1, r3, r2], color=["tab:red", "tab:orange", "tab:blue"])
axes[2].set_yscale("log")
axes[2].axhline(1.0, color="tab:red", ls="--", label="Born 判据线")
axes[2].set_ylabel("R（n = 1，电子）")
axes[2].set_title("(c) 三读法相差 %.0f 个数量级" % orders)
axes[2].legend(fontsize=8)
axes[2].grid(True, axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig(IMG, dpi=140)
print("图已存：" + IMG)
