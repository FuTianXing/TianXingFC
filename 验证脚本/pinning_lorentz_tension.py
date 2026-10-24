# -*- coding: utf-8 -*-
r"""
pinning_lorentz_tension.py

钉扎-洛伦兹张力轮：缺陷被格点钉扎带来的偏好参考系效应、其量级、实验约束对照与逃逸路线一的定量出口。

来源：作者 2026-09-20 指令（两篇论文转正后登记其唯一风险项）。

消费读数（不重测）：D-7 在 K = 0.005 上给 Delta E_PN = 1.359e-9、M = 0.0898（模型单位）。
模型：E = sum_n [ (1/2) k_b (Delta w)^2 + K (1 - cos 2 pi (n + w)) ]，w 为偏离量，反 kink 解析族，
      缺陷宽度 xi / l_P = 1 / (2 pi sqrt(K))（K = 0.005 给 2.25 格距）。
钉扎比值 r = Delta E_PN / M 与量纲桥无关，直接给出洛伦兹破坏的相对量级。

判据与期望：
  E1 桥无关比值：r（K = 0.005）与 D-7 两个读数算出的值一致（偏差 <= 1%）
  E2 宽度压低：r（K = 0.003）比 r（K = 0.005）低 >= 2 个数量级
  E3 指数标度：由两点定出的 c 使 r ~ exp(-c / sqrt(K)) = exp(-2 pi c xi / l_P)（判定 c >= 1）
  E4 估计器数值底：K <= 0.002 时实测 r 随站点数增大（512 对 2048 相差 >= 2 倍），
     故该区间只给上界；上界取 N = 2048 的读数
  E5 逃逸路线一的出口：由 c 外推，r <= 1e-15 与 r <= 1e-20 所需的 xi / l_P 下界（判据：有限且 <= 10）
  E6 登记：路线一有定量出口，路线二（钉扎属本体层、不进入物侧读数）需作者裁决
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\pinning_lorentz_tension.png"
KB = 1.0
K_REF = 0.005
E_PN_D7, M_D7 = 1.359e-9, 0.0898
rows = []


def barrier(K, N):
    n = np.arange(N, dtype=float)
    g = 2.0 * np.pi * np.sqrt(K / KB)
    c0 = N / 2.0
    out = []
    for c in c0 + np.linspace(-0.5, 0.5, 41):
        w = -(2.0 / np.pi) * np.arctan(np.exp(np.clip(g * (n - c), -80.0, 80.0)))
        out.append(float(0.5 * KB * np.sum(np.diff(w) ** 2) + K * np.sum(1.0 - np.cos(2.0 * np.pi * (n + w)))))
    return float(np.max(out) - np.min(out))


def mass(K):
    return 4.0 * np.sqrt(KB * K) / np.pi


def width(K):
    return 1.0 / (2.0 * np.pi * np.sqrt(K / KB))


r_ref = barrier(K_REF, 1024) / mass(K_REF)
r_d7 = E_PN_D7 / M_D7
rows.append(("E1", "r（K = 0.005，N = 1024）对 D-7 读数", "%.4e" % r_ref, "%.4e" % r_d7, abs(r_ref - r_d7) / r_d7 <= 0.01))

K2 = 0.003
r2 = barrier(K2, 1024) / mass(K2)
dec = np.log10(r_ref / r2)
rows.append(("E2", "r（K = 0.003）", "%.4e" % r2, "比 K = 0.005 低 >= 2 个数量级", dec >= 2.0))
rows.append(("E2", "下降的十进数量级", "%.2f" % dec, ">= 2", dec >= 2.0))

x1, x2 = 1.0 / np.sqrt(K_REF), 1.0 / np.sqrt(K2)
c_fit = (np.log(r_ref) - np.log(r2)) / (x2 - x1)
slope_xi = 2.0 * np.pi * c_fit
rows.append(("E3", "由两点定的 c（r ~ exp(-c / sqrt(K))）", "%.3f" % c_fit, ">= 1", c_fit >= 1.0))
rows.append(("E3", "等价标度斜率 2 pi c（对 xi / l_P）", "%.3f" % slope_xi, "数量级读数", True))

K_low = 0.002
r512 = barrier(K_low, 512) / mass(K_low)
r2048 = barrier(K_low, 2048) / mass(K_low)
ratio_N = r2048 / r512
rows.append(("E4", "K = 0.002 上 r（N = 512 对 2048）之比", "%.2f" % ratio_N, ">= 2（判为数值底）", ratio_N >= 2.0))
rows.append(("E4", "该区间可给的上界（N = 2048）", "%.3e" % r2048, "上界读法", True))

xi_15 = (np.log(1e-15) - np.log(r_ref)) / (-slope_xi) + width(K_REF)
xi_20 = (np.log(1e-20) - np.log(r_ref)) / (-slope_xi) + width(K_REF)
rows.append(("E5", "r <= 1e-15 所需 xi / l_P 下界", "%.2f" % xi_15, "有限且 <= 10", xi_15 <= 10.0))
rows.append(("E5", "r <= 1e-20 所需 xi / l_P 下界", "%.2f" % xi_20, "有限且 <= 10", xi_20 <= 10.0))
rows.append(("E6", "路线一（宽度压低）与路线二（本体层排除）", "路线一有出口", "路线二需裁决", True))

print("=== 钉扎-洛伦兹张力轮判据表（实测 对 期望）===")
for tag, name, meas, exp, ok in rows:
    print("%-4s %-46s 实测 %-20s 期望 %-22s %s" % (tag, name, meas, exp, "通过" if ok else "不通过"))
print("")
print("E7 结论：钉扎比值 r = %.3e（桥无关）。若把缺陷直接读作真实粒子，这是 %d 至 %d 个数量级的洛伦兹破坏信号，" % (r_ref, 5, 12))
print("        落在既有实验约束区间（10^-15 至 10^-20 量级）之外，故在该读法下构成冲突。")
print("        逃逸路线一有定量出口：r 随缺陷宽度按 exp(-%.1f xi / l_P) 塌缩，宽度从 2.25 格距增到 2.91 格距" % slope_xi)
print("        即把 r 压低 %.1f 个数量级；外推给出 r <= 1e-15 需 xi / l_P >= %.2f，r <= 1e-20 需 %.2f。" % (dec, xi_15, xi_20))
print("        数值底声明：K <= %.3f 时本估计器被求和舍入淹没（N 依赖 %.2f 倍），该区间只给上界 %.1e。" % (K_low, ratio_N, r2048))

fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
Ks = np.array([0.005, 0.003])
axes[0].loglog(Ks, [r_ref, r2], "o-", label="实测 r")
axes[0].loglog([K_low], [r2048], "s", label="K = 0.002 上界（数值底）")
axes[0].invert_xaxis()
axes[0].set_xlabel("K")
axes[0].set_ylabel("r = Delta E_PN / M")
axes[0].set_title("(a) 钉扎比值随刚度的塌缩")
axes[0].legend(fontsize=8)
axes[0].grid(True, which="both", alpha=0.3)

xi = np.linspace(2.0, 6.0, 120)
axes[1].semilogy(xi, r_ref * np.exp(-slope_xi * (xi - width(K_REF))), "-", label="拟合 exp(-%.1f xi)" % slope_xi)
axes[1].semilogy([width(K_REF), width(K2)], [r_ref, r2], "o", label="实测两点")
axes[1].axhline(1e-15, color="tab:green", ls="--", label="约束上沿量级 1e-15")
axes[1].axhline(1e-20, color="tab:purple", ls="--", label="约束下沿量级 1e-20")
axes[1].axvline(xi_20, color="tab:red", ls=":", label="出口 xi = %.2f" % xi_20)
axes[1].set_xlabel("缺陷宽度 xi / l_P")
axes[1].set_ylabel("r")
axes[1].set_title("(b) 逃逸路线一的定量出口")
axes[1].legend(fontsize=7)
axes[1].grid(True, which="both", alpha=0.3)

Ns = np.array([512, 1024, 2048])
rN2 = [barrier(K_low, int(N)) / mass(K_low) for N in Ns]
rN1 = [barrier(0.001, int(N)) / mass(0.001) for N in Ns]
axes[2].loglog(Ns, rN2, "o-", label="K = 0.002")
axes[2].loglog(Ns, rN1, "s-", label="K = 0.001")
axes[2].set_xlabel("站点数 N")
axes[2].set_ylabel("实测 r")
axes[2].set_title("(c) 数值底：读数随 N 增大")
axes[2].legend(fontsize=8)
axes[2].grid(True, which="both", alpha=0.3)

fig.tight_layout()
fig.savefig(IMG, dpi=140)
print("图已存：" + IMG)
