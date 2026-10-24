# -*- coding: utf-8 -*-
r"""
tail_amplitude_kappa.py

尾振幅的弹性系数标度轮：判定教科书形式与实测形式谁对。

来源：作者 2026-09-20 提问（教科书对还是目前方案对）。上一轮把教科书形式写成 8K exp(-gamma d)，
      但未把弹性系数 k_b 的字典做全，故对照不成立，本轮补做无字典歧义的判据。

字典：把 模型 E = sum [ (1/2) k_b (Delta w)^2 + K (1 - cos 2 pi w) ] 映到教科书形式
      L = (1/2) phi_t^2 - (1/2) phi_x^2 - (m^2/beta^2)(1 - cos beta phi)：
      取 phi = sqrt(k_b) w，则 beta = 2 pi / sqrt(k_b)，m = gamma = 2 pi sqrt(K / k_b)；
      字面结果 V = (8 m^2 / beta^2) exp(- m d) 在该字典下等于 2 k_b gamma^2 / pi^2 exp(-gamma d) = 8 K exp(-gamma d)，
      即教科书形式在固定 K 下与 k_b 无关；而上一轮实测形式 A = 4.263 sqrt(k_b K) 在固定 K 下正比于 sqrt(k_b)。

判据（先行）：
  J1 固定 K = 0.005、变 k_b（0.25、1、4、16）时，振幅 A 对 k_b 的拟合指数 q_k：
     若 q_k ≈ 0.5 则目前方案对；若 q_k ≈ 0 则教科书形式对
  J2 与教科书值 8K 的比值随 k_b 的散布（若稳定在 1 附近则教科书对）
  J3 登记：由实测决定，并为胜出一方给出后续口径
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\tail_amplitude_kappa.png"
N = 1024
n = np.arange(N, dtype=float)
rows = []


def energy(kb, K, w):
    return float(0.5 * kb * np.sum(np.diff(w) ** 2) + K * np.sum(1.0 - np.cos(2.0 * np.pi * (n + w))))


def kink(kb, K, c):
    g = 2.0 * np.pi * np.sqrt(K / kb)
    return -(2.0 / np.pi) * np.arctan(np.exp(np.clip(g * (n - c), -200.0, 200.0)))


K_FIX = 0.005
kbs = [0.25, 1.0, 4.0, 16.0]
Am = {}
for kb in kbs:
    g = 2.0 * np.pi * np.sqrt(K_FIX / kb)
    xi = 1.0 / g
    c0 = N / 2.0
    E1 = energy(kb, K_FIX, kink(kb, K_FIX, c0))
    A = []
    for m in [3.0, 4.0, 5.0]:
        d = m * xi
        w2 = kink(kb, K_FIX, c0 - 0.5 * d) + kink(kb, K_FIX, c0 + 0.5 * d)
        v = energy(kb, K_FIX, w2) - 2.0 * E1
        A.append(v * np.exp(m))
    Am[kb] = float(np.median(A))
    rows.append(("J2", "k_b = %.2f 的振幅 A（中位，m = 3 至 5）" % kb, "%.5e" % Am[kb], "教科书 8K = %.5e" % (8.0 * K_FIX), True))

kb_arr = np.array(kbs)
A_arr = np.array([Am[kb] for kb in kbs])
qk, lc = np.polyfit(np.log(kb_arr), np.log(A_arr), 1)
pred = np.exp(lc) * kb_arr ** qk
res = np.abs(np.log10(A_arr / pred))
rows.append(("J1", "A 对 k_b 的拟合指数 q_k", "%.4f" % qk, "0.5（目前方案）或 0（教科书）", abs(qk - 0.5) < 0.05 or abs(qk) < 0.05))
rows.append(("J1", "拟合最大十进残差", "%.4f" % float(res.max()), "<= 0.05", float(res.max()) <= 0.05))
ratio_th = A_arr / (8.0 * K_FIX)
spread = float((ratio_th.max() - ratio_th.min()) / ratio_th.mean())
rows.append(("J2", "A/(8K) 的四档值与散布", "%s（散布 %.3f）" % (", ".join("%.2f" % v for v in ratio_th), spread), "散布 <= 0.10 则教科书对", spread <= 0.10))

print("=== 尾振幅弹性系数标度轮判据表（实测 对 期望）===")
for tag, name, meas, exp, ok in rows:
    print("%-4s %-44s 实测 %-30s 期望 %-26s %s" % (tag, name, meas, exp, "通过" if ok else "不通过"))
print("")
print("J3 判定：拟合指数 q_k = %.4f，A/(8K) 的四档散布 %.3f。" % (qk, spread))
print("        若 q_k 接近 0.5，则振幅正比于 sqrt(k_b K)，教科书形式 V = 8K exp(-gamma d) 在弹性系数依赖上不成立；")
print("        若 q_k 接近 0，则振幅与 k_b 无关，教科书形式成立。")

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
axes[0].loglog(kb_arr, A_arr, "o-", label="实测 A")
axes[0].loglog(kb_arr, np.exp(lc) * kb_arr ** qk, "--", label="拟合 q_k = %.3f" % qk)
axes[0].loglog(kb_arr, kb_arr ** 0.5 * A_arr[0] / kb_arr[0] ** 0.5, ":", label="sqrt(k_b) 参考")
axes[0].axhline(8.0 * K_FIX, color="tab:red", ls="-.", label="教科书 8K")
axes[0].set_xlabel("k_b")
axes[0].set_ylabel("A")
axes[0].set_title("(a) 固定 K 下振幅对弹性系数的标度")
axes[0].legend(fontsize=7)
axes[0].grid(True, which="both", alpha=0.3)

axes[1].semilogx(kb_arr, ratio_th, "o-", label="A/(8K)")
axes[1].axhline(1.0, color="tab:red", ls="--", label="教科书形式")
axes[1].set_xlabel("k_b")
axes[1].set_ylabel("A/(8K)")
axes[1].set_title("(b) 与教科书值的比值（散布 %.3f）" % spread)
axes[1].legend(fontsize=7)
axes[1].grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig(IMG, dpi=140)
print("图已存：" + IMG)
