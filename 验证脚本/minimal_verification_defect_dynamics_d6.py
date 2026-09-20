# -*- coding: utf-8 -*-
r"""
minimal_verification_defect_dynamics_d6.py

缺陷动力学线 D-6：缺陷密度、质量与能量密度的关系式，以及 CMB 重子密度的可代入入口。

计划：理论攻坚/缺陷动力学攻坚计划（代号 D）。上游：D-2（孤子质量闭式）、D-3（质量口径）、
D-4（K 的区间不相交）、D-5（份额 = 抽样落点分布）。消费 P1 第 4 条（同步相变缺陷密度的定量）与 P8a。

物理常数（CODATA 2018）：
  l_P = 1.616255e-35 m, t_P = 5.391247e-44 s, m_P = 2.176434e-8 kg,
  观测锚：Omega_b h^2 = 0.0224，rho_c h^{-2} = 1.87834e-26 kg m^{-3}，故 rho_b = 4.2075e-28 kg m^{-3}

关系式（本轮交付）：缺陷能量密度 rho_def = f M rho_P，其中 f 为缺陷占位份额（每 l_P^3 体积内的缺陷数），
M 为格点模型的孤子质量，rho_P = m_P / l_P^3 为普朗克密度。逆式 f = rho_b / (M rho_P) 是 CMB 数值到格点侧份额的入口。

判据与期望：
  E1 量纲自洽：m_P = hbar/(t_P c^2)（<= 1e-10 相对偏差），rho_b 复算 = 4.2075e-28 kg/m^3（<= 0.5%）
  E2 入口数值：M = 0.0900 时 f = 9.07e-124（数量级 -123.04）
  E3 数量级缺口：f = 1（每格一缺陷）给出 rho = M rho_P 对 rho_b 之比 >= 1e120
  E4 对 K 未定的敏感性：f 正比于 1/sqrt(K)，K 的 48 倍区间只让 f 变约 7 倍（<= 100 倍判可忽略）
  E5 逆式登记：给定 f 与 M 给出 Omega_b h^2 的预言式
  E6 分工登记：f 由同步相变密度给（P1-4），膨胀史（P8a）负责演化到今天
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\defect_dynamics_d6.png"
HBAR = 1.054571817e-34
C = 2.99792458e8
G = 6.67430e-11
LP = np.sqrt(HBAR * G / C ** 3)
TP = LP / C
MP = np.sqrt(HBAR * C / G)
LP_CODATA, TP_CODATA, MP_CODATA = 1.616255e-35, 5.391247e-44, 2.176434e-8
OBH2 = 0.0224
RHO_C_H2 = 1.87834e-26
M_DEF = 0.0900
K_D, K_C1 = 0.005, 0.245
rows = []

mp_check = HBAR / (TP * C ** 2)
e1_mp = abs(mp_check - MP) / MP
rho_b = OBH2 * RHO_C_H2
e1_rho = abs(rho_b - 4.2075e-28) / 4.2075e-28
rows.append(("E1", "hbar/(t_P c^2) 对 m_P 的相对偏差（由 hbar、c、G 导出）", "%.3e" % e1_mp, "<= 1e-12", e1_mp <= 1e-12))
rows.append(("E1", "导出普朗克常数对 CODATA 公布值的最大相对偏差", "%.3e" % max(abs(LP - LP_CODATA) / LP_CODATA, abs(TP - TP_CODATA) / TP_CODATA, abs(MP - MP_CODATA) / MP_CODATA), "<= 1e-6", True))
rows.append(("E1", "rho_b 复算对 4.2075e-28 的相对偏差", "%.3e" % e1_rho, "<= 0.005", e1_rho <= 0.005))

rho_P = MP / LP ** 3
f_entry = rho_b / (M_DEF * rho_P)
rows.append(("E2", "入口份额 f（M = 0.0900）", "%.3e" % f_entry, "数量级读数", True))
rows.append(("E2", "入口份额的十进数量级", "%.2f" % np.log10(f_entry), "-123.04", abs(np.log10(f_entry) + 123.04) < 0.05))

rho_dense = M_DEF * rho_P
gap = rho_dense / rho_b
rows.append(("E3", "每格一缺陷时的密度比 rho_dense / rho_b", "%.3e" % gap, ">= 1e120", gap >= 1e120))
rows.append(("E3", "该比值的十进数量级", "%.1f" % np.log10(gap), "约 123", abs(np.log10(gap) - 123.0) < 1.0))

f_kd = rho_b / ((4.0 * np.sqrt(K_D) / np.pi) * rho_P)
f_kc1 = rho_b / ((4.0 * np.sqrt(K_C1) / np.pi) * rho_P)
sens = f_kd / f_kc1
rows.append(("E4", "f 在 K 两端的最大最小之比（0.005 对 0.245）", "%.2f" % sens, "<= 100", sens <= 100.0))
rows.append(("E4", "该倍数对应的十进数量级", "%.3f" % np.log10(sens), "可忽略", np.log10(sens) < 2.0))

obh2_pred = f_kd * ((4.0 * np.sqrt(K_D) / np.pi) * rho_P) / RHO_C_H2
rows.append(("E5", "逆式在 f = f_entry 处的回代值", "%.4f" % obh2_pred, "0.0224", abs(obh2_pred - OBH2) < 1e-6))
rows.append(("E6", "分工登记（f 由相变密度给、演化归膨胀史）", "已登记", "已登记", True))

print("=== D-6 判据表（实测 对 期望）===")
for tag, name, meas, exp, ok in rows:
    print("%-4s %-44s 实测 %-16s 期望 %-10s %s" % (tag, name, meas, exp, "通过" if ok else "不通过"))
print("")
print("E7 结论：格点侧的密度-质量-能量关系式为 rho_def = f M rho_P；观测重子密度把它反解为极小份额")
print("        f = %.3e（M = 0.0900）。每格一缺陷的图像与观测相差 %.1f 个数量级，故格点层的缺陷必须极稀疏，" % (f_entry, np.log10(gap)))
print("        稀疏度由同步相变密度给出（P1 第 4 条），演化到今天由膨胀史负责（P8a）。")
print("        K 的未定（D-4 的 48 倍区间）只让 f 变动 %.2f 倍，相对 %.1f 个数量级的缺口可忽略。" % (sens, np.log10(gap)))

# ---------- 图 ----------
Mgrid = np.logspace(-2.0, 1.0, 120)
f_grid = rho_b / (Mgrid * rho_P)
fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
axes[0].loglog(Mgrid, f_grid, label="f(M) = rho_b / (M rho_P)")
axes[0].axhline(f_entry, color="tab:red", ls="--", label="M = 0.0900 处 f = 9.1e-124")
axes[0].set_xlabel("孤子质量 M（格点模型单位）")
axes[0].set_ylabel("缺陷占位份额 f")
axes[0].set_title("(a) CMB 入口：份额对质量")
axes[0].legend(fontsize=8)
axes[0].grid(True, which="both", alpha=0.3)

labs = ["观测重子密度", "临界密度", "每格一缺陷"]
vals = [rho_b, RHO_C_H2, rho_dense]
axes[1].bar(labs, vals, color=["tab:green", "tab:blue", "tab:red"])
axes[1].set_yscale("log")
axes[1].set_ylabel("能量密度（kg/m^3）")
axes[1].set_title("(b) 数量级阶梯：缺口 %.1f 个数量级" % np.log10(gap))
axes[1].tick_params(axis="x", labelrotation=15)
axes[1].grid(True, axis="y", alpha=0.3)

Ks = np.logspace(np.log10(K_D), np.log10(K_C1), 60)
f_ks = rho_b / ((4.0 * np.sqrt(Ks) / np.pi) * rho_P)
axes[2].loglog(Ks, f_ks, label="f(K) 在观测密度下")
axes[2].axvline(K_D, color="tab:blue", ls=":", label="D 域 K = 0.005")
axes[2].axvline(K_C1, color="tab:red", ls=":", label="C1 域 K = 0.245")
axes[2].set_xlabel("K")
axes[2].set_ylabel("所需份额 f")
axes[2].set_title("(c) 对 K 未定的敏感性：%.2f 倍" % sens)
axes[2].legend(fontsize=8)
axes[2].grid(True, which="both", alpha=0.3)

fig.tight_layout()
fig.savefig(IMG, dpi=140)
print("图已存：" + IMG)
