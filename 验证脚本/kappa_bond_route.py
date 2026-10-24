# -*- coding: utf-8 -*-
r"""
kappa_bond_route.py

P4-16 第一步第六轮（路线丙）：键能曲率通道的绝对标度。
第 178 条的键能模型为 E = sum_e (1/2) kappa_e (Delta u_e - 1)^2 + sum_i K (1 - cos 2 pi u_i)，
其中 kappa_e 为键曲率，其可行域由稳定性给定：kappa_e >= 0.1 kappa（刚度下限）。
本轮把该曲率换算成无量纲的锁定率 kappa_bond，并在两种归一化口径下与路线乙的 kappa = 1/d = 0.125 对照。

两种口径：
  平方根口径（键频口径）：kappa_bond = sqrt(kappa_e / kappa_ref)，
    其中 kappa_ref 取"无反馈处键频等于普朗克频率"的参考点（kappa_e = kappa_ref 时 kappa_bond = 1）。
  线性口径（曲率比口径）：kappa_bond = kappa_e / kappa_ref。

判据与期望：
  C1 丙通道范围（平方根口径）：kappa_e/kappa_ref 在 [0.1, 1] 上给 kappa_bond 在 [0.3162, 1]
  C2 两口径的判别：路线乙的 0.125 落在线性口径的范围 [0.1, 1] 内，却不落在平方根口径的 [0.3162, 1] 内
  C3 所需曲率比：线性口径要求 kappa_e/kappa_ref = 0.125（在 178 的可行域内）；
     平方根口径要求 0.015625（低于 178 的刚度下限 0.1，不可达）
  C4 稳定性检查：线性口径下的要求点与 178 的下限之比为 1.25（余量 25%）
  C5 接口登记（口径裁决：线性还是平方根）
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\kappa_bond_route.png"
KAPPA_B = 0.125
KAPPA_E_MIN = 0.1
rows = []


def report(tag, ok, expect, actual):
    rows.append((tag, ok, expect, actual))
    print("[%s] %s" % ("通过" if ok else "未过", tag))
    print("      期望 " + expect)
    print("      实测 " + actual)


ratios = np.linspace(KAPPA_E_MIN, 1.0, 200)
sqrt_range = np.sqrt(ratios)
lin_range = ratios

report("C1 丙通道范围（平方根口径）",
       abs(sqrt_range.min() - np.sqrt(KAPPA_E_MIN)) <= 1e-12 and abs(sqrt_range.max() - 1.0) <= 1e-12,
       "kappa_e/kappa_ref 在 [0.1, 1] 上给 kappa_bond 在 [%.4f, 1]（解析）" % np.sqrt(KAPPA_E_MIN),
       "平方根口径 [%.4f, %.4f]；线性口径 [%.4f, %.4f]" % (sqrt_range.min(), sqrt_range.max(), lin_range.min(), lin_range.max()))

in_sqrt = bool(np.any(np.abs(sqrt_range - KAPPA_B) < 5e-3))
in_lin = bool(np.any(np.abs(lin_range - KAPPA_B) < 5e-3))
report("C2 两口径的判别", (not in_sqrt) and in_lin,
       "路线乙的 0.125 落在线性口径范围内、不落在平方根口径范围内",
       "平方根口径命中 0.125：%s；线性口径命中 0.125：%s" % (in_sqrt, in_lin))

need_lin = KAPPA_B
need_sqrt = KAPPA_B ** 2
report("C3 所需曲率比", abs(need_lin - 0.125) <= 1e-12 and need_sqrt < KAPPA_E_MIN,
       "线性口径所需 kappa_e/kappa_ref = 0.125（在可行域内）；平方根口径所需 0.015625（低于下限 0.1，不可达）",
       "线性所需 %.6f（可行：%s）；平方根所需 %.6f（可行：%s）" % (need_lin, need_lin >= KAPPA_E_MIN, need_sqrt, need_sqrt >= KAPPA_E_MIN))

report("C4 稳定性检查", abs(need_lin / KAPPA_E_MIN - 1.25) <= 1e-9,
       "线性口径的要求点与 178 的刚度下限之比为 1.25（余量 25%）",
       "比值 %.4f" % (need_lin / KAPPA_E_MIN))

print("[登记] C5：口径裁决（线性还是平方根），即 kappa 的绝对标度取键曲率比还是键频比；本轮只登记，不作判据")

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
ax[0].plot(ratios, sqrt_range, "-", label="平方根口径 sqrt(kappa_e/kappa_ref)")
ax[0].plot(ratios, lin_range, "--", label="线性口径 kappa_e/kappa_ref")
ax[0].axhline(KAPPA_B, ls=":", c="tab:red", lw=1, label="路线乙 kappa = 0.125")
ax[0].axvline(KAPPA_E_MIN, ls=":", c="k", lw=1, label="178 刚度下限 0.1")
ax[0].set_xlabel("kappa_e / kappa_ref")
ax[0].set_ylabel("kappa_bond")
ax[0].set_title("两口径下的丙通道读数")
ax[0].legend(fontsize=8)
labs = ["线性口径\n所需", "平方根口径\n所需", "178 下限"]
vals = [need_lin, need_sqrt, KAPPA_E_MIN]
ax[1].bar(range(3), vals, color=["tab:blue", "tab:orange", "tab:gray"])
ax[1].set_xticks(range(3))
ax[1].set_xticklabels(labs, fontsize=8)
ax[1].set_ylabel("kappa_e / kappa_ref")
ax[1].set_title("复现 0.125 所需的曲率比")
plt.tight_layout()
plt.savefig(IMG, dpi=150)
plt.close(fig)

bad = [r for r in rows if not r[1]]
print("---- 判据 %d 项，通过 %d 项；图存 %s ----" % (len(rows), len(rows) - len(bad), IMG))
sys.exit(1 if bad else 0)
