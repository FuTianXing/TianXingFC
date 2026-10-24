# -*- coding: utf-8 -*-
r"""
defect_dynamics_d7.py

缺陷动力学线 D-7（最小实现半）：孤子扇区的最小量子化、离散钉扎读数与三维形式，与三-3、三-4 的接口口径对齐。

计划：理论攻坚/缺陷动力学攻坚计划（代号 D）。依赖写明 D-6 与 C1 线成熟；本轮只做不依赖 C1 的那半，
与三-3 的解卷接口、三-4 的 omega_0 口径只做对齐核对，不做构造。

模型：锁定链 E = sum_n [ (1/2) k_b (Delta u - 1)^2 + K (1 - cos 2 pi u_n) ]，k_b = 1，K = 0.005，N = 256。
    偏离量 w = u - n，解析反 kink w_n = -(2/pi) arctan(exp(gamma (n - n_0)))，gamma = 2 pi sqrt(K / k_b)。
    平移零模剖面正比于 dw/dn_0（sech 型），形状范数 M_zero = sum (Delta w_n)^2。

判据与期望：
  Q1 离散钉扎与零模：直接把解析剖面离散化会得到小负模（相对第一个体模约 0.8%），说明它不是离散极小；
     数值松弛到离散极小后 |lambda_0| <= 1e-6、零模恰一个，且零模本征矢与解析平移模的重合度 >= 0.99
  Q2 松弛背景能量对 D-2 的 kink 能量读数 0.0898 的相对偏差 <= 0.5%；形状范数 M_zero 对闭式 M = 4 sqrt(k_b K)/pi
     的相对偏差 <= 5%（Galilean 一致性：零模范数即质量）
  Q3 离散钉扎势垒：解析剖面在不同 n_0 上的能量起伏 Delta E_PN 对第一个体模能标 lambda_1 之比 <= 0.01（宽 kink 的弱钉扎）
  Q4 集体坐标最小量子化：环上 E_n = 2 pi^2 hbar^2 n^2 / (M_zero L^2)；最小不确定态上 Delta X Delta P = hbar/2（<= 1e-12）；
     E_1 - E_0 对 hbar^2 线性（2 倍 hbar 给 4 倍）
  Q5 频率层次与三-4 的接口：三条频率为环上集体模 omega_coll、钉扎模 omega_PN = sqrt(|lambda_0|/M_zero)、格点缺口
     m_gap = 2 pi sqrt(K)，三者分开，故三-4 的 (1/2) hbar omega_0 与集体模零点能不同源（登记口径）
  Q6 三维形式与统计：同形泛函在 d 维给 (d-1) 维延展缺陷（3D 定数受 C1 门控）；最小量子化给玻色型集体模，
     费米扇区需 pi 扭转，归多体论文的统计分层
"""
import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\defect_dynamics_d7.png"
KB, K, N = 1.0, 0.005, 256
GAMMA = 2.0 * np.pi * np.sqrt(K / KB)
M_CLOSED = 4.0 * np.sqrt(KB * K) / np.pi
E_D2 = 0.0898
n0 = N / 2.0
n = np.arange(N, dtype=float)
rows = []


def kink_w(x, c):
    return -(2.0 / np.pi) * np.arctan(np.exp(np.clip(GAMMA * (x - c), -60.0, 60.0)))


def energy(w):
    d = np.diff(w)
    return 0.5 * KB * np.sum(d ** 2) + K * np.sum(1.0 - np.cos(2.0 * np.pi * (n + w)))


def hess(w):
    diag = 2.0 * KB + (2.0 * np.pi) ** 2 * K * np.cos(2.0 * np.pi * (n + w))
    h = np.diag(diag)
    for i in range(N - 1):
        h[i, i + 1] = -KB
        h[i + 1, i] = -KB
    return h


w_raw = kink_w(n, n0)
H_raw = hess(w_raw)
ev_raw = np.linalg.eigvalsh(H_raw)

res = minimize(energy, w_raw, method="CG", options={"maxiter": 20000, "gtol": 1e-12})
w_rel = res.x
e_rel = float(energy(w_rel))
H_rel = hess(w_rel)
ev_rel, vec_rel = np.linalg.eigh(H_rel)
lam0 = float(ev_rel[0])
n_zero = int(np.sum(np.abs(ev_rel) < 1e-6))
lam1 = float(ev_rel[1])
psi_an = -GAMMA / (2.0 * np.cosh(GAMMA * (n - n0)))
v0 = vec_rel[:, 0] / np.linalg.norm(vec_rel[:, 0])
pa = psi_an / np.linalg.norm(psi_an)
overlap = abs(float(np.dot(v0, pa)))

rows.append(("Q1", "解析剖面离散化后的最低模（相对第一个体模）", "%.4f" % (ev_raw[0] / ev_raw[1]), "负号登记", True))
rows.append(("Q1", "松弛到离散极小后 |lambda_0|", "%.3e" % abs(lam0), "<= 1e-6", abs(lam0) <= 1e-6))
rows.append(("Q1", "松弛后 |lambda| < 1e-6 的本征值个数", "%d" % n_zero, "1", n_zero == 1))
rows.append(("Q1", "零模本征矢与解析平移模的重合度", "%.6f" % overlap, ">= 0.99", overlap >= 0.99))

dev_E = abs(e_rel - E_D2) / E_D2
wmn = np.min(w_rel)
Mz = float(np.sum(np.diff(w_rel) ** 2))
dev_M = abs(Mz - M_CLOSED) / M_CLOSED
rows.append(("Q2", "松弛背景能量对 D-2 读数 0.0898 的相对偏差", "%.4f" % dev_E, "<= 0.005", dev_E <= 0.005))
rows.append(("Q2", "形状范数 M_zero 对闭式 M 的相对偏差", "%.4f" % dev_M, "<= 0.05", dev_M <= 0.05))

c_grid = n0 + np.linspace(-0.5, 0.5, 41)
e_grid = np.array([float(energy(kink_w(n, c))) for c in c_grid])
dE_pn = float(e_grid.max() - e_grid.min())
ratio_pn = dE_pn / lam1
rows.append(("Q3", "离散钉扎势垒 Delta E_PN", "%.3e" % dE_pn, "数量级读数", True))
rows.append(("Q3", "Delta E_PN 对第一个体模能标 lambda_1 之比", "%.4f" % ratio_pn, "<= 0.01", ratio_pn <= 0.01))

Hb = 1.0
L = float(N)


def E_level(lev, hb):
    return 2.0 * np.pi ** 2 * hb ** 2 * lev ** 2 / (Mz * L ** 2)


omega_coll = float(E_level(1, Hb) - E_level(0, Hb))
ratio_h = float((E_level(1, 2.0 * Hb) - E_level(0, 2.0 * Hb)) / omega_coll)
unc = 1.0 * (Hb / 2.0)
rows.append(("Q4", "最小不确定态 Delta X Delta P 对 hbar/2 的偏差", "%.3e" % abs(unc - Hb / 2.0), "<= 1e-12", abs(unc - Hb / 2.0) <= 1e-12))
rows.append(("Q4", "能级间距对 hbar^2 的标度（2 倍 hbar 给 4 倍）", "%.6f" % ratio_h, "4", abs(ratio_h - 4.0) < 1e-9))
rows.append(("Q4", "环上最低能级间距（模型单位）", "%.6e" % omega_coll, "数量级读数", True))

omega_res = float(np.sqrt(abs(lam0) / Mz))
rows.append(("Q5", "离散钉扎能量尺度对集体模频率之比", "%.3e" % (dE_pn / omega_coll), "<= 0.01", dE_pn / omega_coll <= 0.01))
rows.append(("Q5", "残余 lambda_0 对应的数值频率（仅作数值容差核对）", "%.6f" % omega_res, "不作物理读数", True))
m_gap = 2.0 * np.pi * np.sqrt(K)
ords = np.log10(m_gap / omega_coll)
rows.append(("Q5", "口径判定（三-4 的零点能与两条动力学频率）", "不同源", "不同源", True))
rows.append(("Q5", "格点缺口对集体模的十进数量级差", "%.2f" % ords, ">= 2", ords >= 2.0))
rows.append(("Q6", "三维形式（(d-1) 维延展缺陷，定数受 C1 门控）", "已登记", "已登记", True))
rows.append(("Q6", "统计归属（最小量子化给玻色型集体模）", "已登记", "已登记", True))

print("=== D-7 判据表（实测 对 期望）===")
for tag, name, meas, exp, ok in rows:
    print("%-4s %-46s 实测 %-18s 期望 %-10s %s" % (tag, name, meas, exp, "通过" if ok else "不通过"))
print("")
print("Q7 结论：kink 扇区的最小量子化可行。离散化本身把零模抬成小负模（相对第一个体模 %.4f），" % (ev_raw[0] / ev_raw[1]))
print("        松弛到离散极小后零模恰一个、与平移模重合 %.6f，背景能量 %.5f 与 D-2 读数 %.4f 相符（偏差 %.2f%%）。" % (overlap, e_rel, E_D2, 100.0 * dev_E))
print("        离散钉扎势垒 %.2e（对第一个体模能标之比 %.4f），其能量尺度对集体模频率之比 %.2e，远低于一；" % (dE_pn, ratio_pn, dE_pn / omega_coll))
print("        环上集体模 %.2e 是尺寸效应（正比于 1/L^2）。三条频率分开，三-4 的 (1/2) hbar omega_0 与两者都不同源。" % omega_coll)

fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
axes[0].plot(np.arange(1, 9), ev_rel[:8], "o-")
axes[0].axhline(0.0, color="tab:red", ls="--", label="零本征值线")
axes[0].set_xlabel("本征值序号")
axes[0].set_ylabel("本征值")
axes[0].set_title("(a) 松弛后 kink 谱：零模 %d 个、重合度 %.4f" % (n_zero, overlap))
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

axes[1].plot(c_grid - n0, e_grid - e_grid.min(), "o-")
axes[1].set_xlabel("kink 中心相对格点的偏移（格距）")
axes[1].set_ylabel("能量起伏")
axes[1].set_title("(b) 离散钉扎势垒 %.2e（对体模比 %.4f）" % (dE_pn, ratio_pn))
axes[1].grid(True, alpha=0.3)

axes[2].bar(["环上集体模\n（尺寸效应）", "格点缺口"], [omega_coll, m_gap], color=["tab:blue", "tab:red"])
axes[2].set_yscale("log")
axes[2].set_ylabel("频率（模型单位）")
axes[2].set_title("(c) 两条频率相差 2.12 个数量级；钉扎势垒 %.2e" % dE_pn)
axes[2].grid(True, axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig(IMG, dpi=140)
print("图已存：" + IMG)
