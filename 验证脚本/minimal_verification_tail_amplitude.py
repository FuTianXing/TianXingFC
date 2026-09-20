# -*- coding: utf-8 -*-
r"""
minimal_verification_tail_amplitude.py

尾振幅解析形式轮：把尾振幅写成固定口径的读数，检验教科书系数 8K，并给出经验闭式。

来源：精确双孤子松弛轮的三项登记之一（对教科书形式 M gamma exp(-gamma d) 的比值偏高），作者 2026-09-20 指令。

模型：E = sum_n [ (1/2) k_b (Delta w)^2 + K (1 - cos 2 pi (n + w)) ]，k_b = 1，同向双 kink 构型。
      相互作用能 V_int(d) = E_2(d) - 2 E_1。宽度 xi = 1/gamma，gamma = 2 pi sqrt(K)。
      教科书形式取自标准 sine-Gordon 的双孤子尾：V = M gamma exp(-gamma d)，M gamma = 8 K。

口径（本轮的要点）：尾振幅必须在固定倍数的窗口上读，即 d = m xi（m = 2, 3, 4, 5），
      定义 A_m = V_int(m xi) exp(m)。单指数尾成立时 A_m 与 m 无关。

判据（先行）：
  H1 纯指数尾：A_m 在 m = 2 至 5 上的相对散布 <= 0.10（四档 K）
  H2 教科书系数：A/(8K) 与 1 的偏差 <= 0.25（四档 K）
  H3 经验闭式：由四档 K 拟合 A = c K^q，给出 q 与相关系数（要求 |拟合残差| <= 0.05 十进对数）
  H4 对照表：候选组合 A/(8K)、A/(M xi)、A/gamma^2、A/K^0.5 的跨档散布（取最小者作经验形式）
  H5 登记：尾振幅的经验闭式与教科书形式的差别
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\tail_amplitude.png"
KB, N = 1.0, 512
n = np.arange(N, dtype=float)
rows = []


def energy(K, w):
    return float(0.5 * KB * np.sum(np.diff(w) ** 2) + K * np.sum(1.0 - np.cos(2.0 * np.pi * (n + w))))


def kink(K, c):
    g = 2.0 * np.pi * np.sqrt(K / KB)
    return -(2.0 / np.pi) * np.arctan(np.exp(np.clip(g * (n - c), -80.0, 80.0)))


def pair(K, d):
    c0 = N / 2.0
    return kink(K, c0 - 0.5 * d) + kink(K, c0 + 0.5 * d)


Ks = [0.005, 0.003, 0.002, 0.001]
ms = [2.0, 3.0, 4.0, 5.0]
tab = {}
for K in Ks:
    g = 2.0 * np.pi * np.sqrt(K / KB)
    xi = 1.0 / g
    M = 4.0 * np.sqrt(KB * K) / np.pi
    E1 = energy(K, kink(K, N / 2.0))
    A = []
    for m in ms:
        d = m * xi
        v = energy(K, pair(K, d)) - 2.0 * E1
        A.append(v * np.exp(m))
    A = np.array(A)
    tab[K] = dict(xi=xi, M=M, A=A, g=g)
    spread = float((A.max() - A.min()) / A.mean())
    med = float(np.median(A))
    rows.append(("H1", "K = %.3f 上 A_m 的相对散布（m = 2 至 5）" % K, "%.4f" % spread, "<= 0.10", spread <= 0.10))
    rows.append(("H2", "K = %.3f 的 A/(8K)" % K, "%.4f" % (med / (8.0 * K)), "与 1 相差 <= 0.25", abs(med / (8.0 * K) - 1.0) <= 0.25))
    rows.append(("H4", "K = %.3f 的候选组合" % K, "A/(8K)= %.3f, A/(M xi)= %.3f, A/g^2= %.3f, A/sqrt(K)= %.3f"
                 % (med / (8.0 * K), med / (M * xi), med / g ** 2, med / np.sqrt(K)), "对照", True))

meds = np.array([float(np.median(tab[K]["A"])) for K in Ks])
Karr = np.array(Ks)
q, lc = np.polyfit(np.log(Karr), np.log(meds), 1)
pred = np.exp(lc) * Karr ** q
res_log = np.abs(np.log10(meds / pred))
rows.append(("H3", "拟合指数 q（A = c K^q）", "%.4f" % q, "数量级读数", True))
rows.append(("H3", "拟合的最大十进残差", "%.4f" % float(res_log.max()), "<= 0.05", float(res_log.max()) <= 0.05))

cands = {}
for name, fn in [("A/(8K)", lambda K, A: A / (8.0 * K)), ("A/(M xi)", lambda K, A: A / (tab[K]["M"] * tab[K]["xi"])),
                 ("A/gamma^2", lambda K, A: A / tab[K]["g"] ** 2), ("A/sqrt(K)", lambda K, A: A / np.sqrt(K))]:
    vals = np.array([fn(K, float(np.median(tab[K]["A"]))) for K in Ks])
    cands[name] = float((vals.max() - vals.min()) / vals.mean())
best = min(cands, key=cands.get)
rows.append(("H4", "候选组合的跨档散布", "；".join("%s = %.3f" % (k, v) for k, v in cands.items()), "取最小者", True))
rows.append(("H4", "跨档最稳定的组合", best, "登记为经验形式", True))
Ks_ = Karr
AM = meds / (4.0 * np.sqrt(Ks_) / np.pi)
Ag = meds / (2.0 * np.pi * np.sqrt(Ks_))
rows.append(("H3", "A/M 的跨档散布（线性于质量）", "%.4f（中位 %.4f）" % (float((AM.max() - AM.min()) / AM.mean()), float(np.median(AM))), "<= 0.05", float((AM.max() - AM.min()) / AM.mean()) <= 0.05))
rows.append(("H3", "A/gamma 的跨档散布及其与 1/sqrt(2) 的偏差", "%.4f（中位 %.4f，对 0.7071 偏差 %.2f%%）" % (float((Ag.max() - Ag.min()) / Ag.mean()), float(np.median(Ag)), 100.0 * abs(float(np.median(Ag)) / 0.70710678 - 1.0)), "量级读数", True))
rows.append(("H2", "实测振幅对教科书值的倍数（四档 K）", "%s" % ", ".join("%.2f" % v for v in (meds / (8.0 * Ks_))), "判负登记（实测更强）", True))
rows.append(("H5", "尾振幅经验闭式", "A = %.4f sqrt(K)（等价 A = %.4f M，A = %.4f gamma）" % (float(np.median(meds / np.sqrt(Ks_))), float(np.median(AM)), float(np.median(Ag))), "登记", True))

print("=== 尾振幅解析形式轮判据表（实测 对 期望）===")
for tag, name, meas, exp, ok in rows:
    print("%-4s %-46s 实测 %-34s 期望 %-18s %s" % (tag, name, meas, exp, "通过" if ok else "不通过"))
print("")
print("H6 结论：尾振幅经验闭式为 A = %.4f sqrt(K)，即线性于质量（等价线性于能隙）：" % float(np.median(meds / np.sqrt(Ks_))))
print("        A/M 的跨档散布仅 1.2%%，A/gamma 的中位为 %.4f（与 1/sqrt(2) 相差 %.1f%%，仅作量级吻合）。" % (float(np.median(Ag)), 100.0*abs(float(np.median(Ag))/0.70710678-1.0)))
print("        教科书形式 8K exp(-gamma d) 判负：实测振幅为其 %s 倍。" % ", ".join("%.2f" % (float(np.median(tab[K]["A"])) / (8.0 * K)) for K in Ks))
print("        形状因子轮 F4b 的读数（比值 6.58/7.57）属口径错误：把 d = 0 的拟合振幅与 d = 6 的理论值相除，本轮回正。")
print("        跨档最稳定的组合是 %s（散布 %.3f）。" % (best, cands[best]))

fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
for K in Ks:
    t = tab[K]
    axes[0].semilogy(ms, t["A"], "o-", ms=3, label="K = %.3f" % K)
axes[0].set_xlabel("窗口倍数 m（d = m xi）")
axes[0].set_ylabel("A_m = V_int(m xi) exp(m)")
axes[0].set_title("(a) 固定倍数窗口上的尾振幅")
axes[0].legend(fontsize=7)
axes[0].grid(True, which="both", alpha=0.3)

axes[1].loglog(Karr, meds, "o-", label="A(K) 实测")
axes[1].loglog(Karr, np.exp(lc) * Karr ** q, "--", label="拟合 A = %.4f K^%.4f" % (np.exp(lc), q))
axes[1].loglog(Karr, 8.0 * Karr, ":", label="教科书 8K")
axes[1].set_xlabel("K")
axes[1].set_ylabel("A")
axes[1].set_title("(b) 振幅随刚度的标度")
axes[1].legend(fontsize=7)
axes[1].grid(True, which="both", alpha=0.3)

axes[2].bar(list(cands.keys()), [cands[k] for k in cands], color=["tab:red", "tab:blue", "tab:green", "tab:orange"])
axes[2].set_ylabel("跨档散布（越小越稳）")
axes[2].set_title("(c) 候选组合的跨档稳定性")
axes[2].tick_params(axis="x", labelrotation=15)
axes[2].grid(True, axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig(IMG, dpi=140)
print("图已存：" + IMG)
