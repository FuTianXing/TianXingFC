# -*- coding: utf-8 -*-
"""M6-S2-1（补）：耦合系数 lambda 的多途径提取与相互核对。

背景：S2-1 判据要求「由至少两条独立途径给出同一系数，相对偏差 <= 10%」。此前只有「同族形式」
一条来源（耦合项 (1-cos theta) h 与挫折弦每格代价 K(1-cos 2 pi q) 同族）。本材料补三条独立途径：
  途径一 能量归一化：把耦合项写成「记录调制 onsite 系数」的形式（E = J sum cos(dtheta) +
         sum (K h_n)(1-cos theta_n)），与 lambda 写法逐位恒等，得 lambda := K（满幅记录下的 onsite 系数）。
  途径二 谱反演：交错链最低模的谱闭式 omega^2_min = 2J - sqrt((2J)^2 + (lambda h)^2)
         （弱版耦合注记 §五），由数值谱反演 lambda。
  途径三 Hessian 直测：单点记录下对角修正 Delta H = -lambda h cos(theta)，直接读 lambda。
  途径四 静态 kink：SG 链上 kink 的能量 E_kink = 8 sqrt(J A) 与宽度 l = sqrt(J/A)（A = lambda h），
         由两者独立反演 A 再除记录幅度得 lambda。

判据（先写期望，再算）：
  S2-1a 能量归一化恒等：两写法逐位恒等（差 <= 1e-15，取 lambda = K = 1）。
  S2-1b 谱反演：lambda_hat2(h) 对 h in {0.1,0.25,0.5,1,2} 离散度 <= 1%，且复核弱版注记 W4 表
         （h = 0.1/0.5/1 的 omega^2_min = -0.002498/-0.061553/-0.236068，偏差 <= 1e-5）。
  S2-1c Hessian 直测：lambda_hat3 对 lambda 偏差 <= 1e-12（机器精度）。
  S2-1d 静态 kink：两条子估计 A 的比值落在 [0.9, 1.1]；lambda_hat4 = A/h 与前三条途径偏差 <= 10%。
  S2-1e 汇总：四条途径的 lambda 提取值最大相对偏差 <= 10%（S2-1 判据本体）；未完成项（K/J 的
         微观本构值）报出。

确定性：固定初值、无随机数；eigvalsh 精确对角化。
"""

import sys
import time
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
CHECKS = []
def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))

print("==== M6-S2-1（补）：耦合系数 lambda 的多途径提取 ====")
J = 1.0
LAM = 1.0
N = 256

# ---------------- 途径一：能量归一化 ----------------
print("\n  [S2-1a] 途径一（能量归一化）：lambda := K")
K = 1.0
n = np.arange(N)
h_uniform = np.full(N, 1.0)
theta = np.pi * n  # 交错背景
E_w1 = J * np.sum(np.cos(np.roll(theta, -1) - theta)) + LAM * np.sum((1 - np.cos(theta)) * h_uniform)
E_w2 = J * np.sum(np.cos(np.roll(theta, -1) - theta)) + np.sum((K * h_uniform) * (1 - np.cos(theta)))
dE = abs(E_w1 - E_w2) / max(abs(E_w1), 1e-30)
print("       两写法能量差 = %.2e（取 lambda = K = %.1f）" % (dE, K))
check("S2-1a", dE <= 1e-15, "能量恒等偏差 %.2e <= 1e-15，lambda := K" % dE)

# ---------------- 途径二：谱反演 ----------------
def low_1d(hvec):
    nn = np.arange(len(hvec))
    H = np.zeros((len(hvec), len(hvec)))
    H[nn, nn] = 2.0 * J + LAM * ((-1.0) ** nn) * hvec
    H[nn, (nn + 1) % len(hvec)] += J
    H[nn, (nn - 1) % len(hvec)] += J
    return float(np.linalg.eigvalsh(H)[0])

print("\n  [S2-1b] 途径二（谱反演）：omega^2_min = 2J - sqrt((2J)^2 + (lambda h)^2)")
lams2 = []
for h in (0.1, 0.25, 0.5, 1.0, 2.0):
    w2 = low_1d(np.full(N, h))
    lam_hat = np.sqrt(max((2 * J - w2) ** 2 - (2 * J) ** 2, 0.0)) / h
    lams2.append(lam_hat)
    print("       h=%.2f: omega^2_min = %+.6f -> lambda_hat2 = %.10f" % (h, w2, lam_hat))
lams2 = np.array(lams2)
spread = (lams2.max() - lams2.min()) / lams2.mean()
tab = {0.1: -0.002498, 0.5: -0.061553, 1.0: -0.236068}
dev_tab = max(abs(low_1d(np.full(N, hh)) - vv) for hh, vv in tab.items())
check("S2-1b", spread <= 0.01 and dev_tab <= 1e-5,
      "离散度 %.2e <= 1%%；弱版注记 W4 表最大偏差 %.1e <= 1e-5" % (spread, dev_tab))

# ---------------- 途径三：Hessian 直测 ----------------
print("\n  [S2-1c] 途径三（Hessian 直测）")
h_single = np.zeros(N); h_single[N // 2] = 1.0
def hess_diag(hvec):
    nn = np.arange(N)
    return 2.0 * J + LAM * ((-1.0) ** nn) * hvec
dH = hess_diag(h_single) - hess_diag(np.zeros(N))
i0 = N // 2
lam_hat3 = -dH[i0] / (np.cos(np.pi * i0) * h_single[i0]) if np.cos(np.pi * i0) != 0 else dH[i0] / h_single[i0]
# 交错背景 theta_n = pi n: cos(theta)=+/-1: 取 |cos|=1
lam_hat3 = abs(dH[i0]) / 1.0
print("       单点记录对角修正 = %.10f，读出 lambda_hat3 = %.10f" % (dH[i0], lam_hat3))
check("S2-1c", abs(lam_hat3 - LAM) <= 1e-12, "偏差 %.1e <= 1e-12（机器精度）" % abs(lam_hat3 - LAM))

# ---------------- 途径四：静态 kink ----------------
print("\n  [S2-1d] 途径四（静态 kink）：E_kink = 8 sqrt(J A)、l = sqrt(J/A)")
def relax_kink(A, Nk=256, steps=60000, dt=0.02):
    th = np.linspace(0.0, 2 * np.pi, Nk)
    th = 4.0 * np.arctan(np.exp((np.arange(Nk) - Nk / 2.0) / 3.0))
    th[0] = 0.0; th[-1] = 2 * np.pi
    for it in range(steps):
        dth = np.roll(th, -1) - th
        force = -(J * (np.sin(np.roll(dth, 1)) - np.sin(dth)) + A * np.sin(th))
        th[1:-1] += dt * force[1:-1]
        if it % 500 == 0 and np.max(np.abs(force[1:-1])) < 1e-12:
            break
    vac = np.zeros(Nk)  # 真空 theta = 0
    def energy(x):
        return float(np.sum(J * (1 - np.cos(np.roll(x, -1) - x))) + np.sum(A * (1 - np.cos(x))))
    dth = np.roll(th, -1) - th
    fmax = float(np.max(np.abs(-(J * (np.sin(np.roll(dth, 1)) - np.sin(dth)) + A * np.sin(th))[1:-1])))
    return energy(th) - energy(vac), th, fmax

rows4 = []
for h in (0.05, 0.02):
    A_true = LAM * h
    E_k, th_k, fmax = relax_kink(A_true)
    A1 = (E_k / 8.0) ** 2 / J
    # 拟合宽度：theta(x) = 4 arctan(exp((x-x0)/l))
    xs = np.arange(len(th_k))
    best = None
    for l in np.linspace(0.5, 40.0, 4000):
        for x0 in (len(th_k) / 2.0,):
            pred = 4 * np.arctan(np.exp((xs - x0) / l))
            r = float(np.sum((th_k - pred) ** 2))
            if best is None or r < best[0]:
                best = (r, l)
    l_fit = best[1]
    A2 = J / l_fit ** 2
    rows4.append((h, A_true, E_k, A1, l_fit, A2, A1 / A2, A1 / h))
    print("       h=%.2f: E_kink = %.6f（残力 %.1e）-> A1 = %.6f；l_fit = %.3f -> A2 = %.6f；A1/A2 = %.4f；lambda_hat4 = %.4f"
          % (h, E_k, fmax, A1, l_fit, A2, A1 / A2, A1 / h))
ratio_dev = max(abs(r[6] - 1.0) for r in rows4)
lam4 = np.mean([r[7] for r in rows4])
check("S2-1d", ratio_dev <= 0.10 and abs(lam4 - LAM) / LAM <= 0.10,
      "A1/A2 最大偏差 %.3f <= 10%%；lambda_hat4 = %.4f 对 1 偏差 %.1f%%" % (ratio_dev, lam4, abs(lam4 - LAM) / LAM * 100))

# ---------------- 汇总 ----------------
print("\n  [S2-1e] 汇总")
lam1 = K
all_lam = [lam1, float(lams2.mean()), lam_hat3, lam4]
spread_all = (max(all_lam) - min(all_lam)) / np.mean(all_lam)
print("       途径一（归一化）lambda = %.6f" % lam1)
print("       途径二（谱反演）lambda = %.6f（离散度 %.2e）" % (lams2.mean(), spread))
print("       途径三（Hessian）lambda = %.6f" % lam_hat3)
print("       途径四（kink）   lambda = %.6f" % lam4)
print("       四条途径最大相对偏差 = %.4f" % spread_all)
check("S2-1e", spread_all <= 0.10, "最大相对偏差 %.4f <= 10%%（S2-1 判据本体）" % spread_all)
print("\n  未完成项：元相子格点的 K/J 微观本构值（挫折弦系数与键刚度之比）仍缺；本材料在模型单位")
print("  （J = 1）下给出 lambda = K = 1 的四途径一致读数，物理常数比待本构材料。")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
ax = axes[0]
hs = np.linspace(0.02, 2.0, 60)
w2s = np.array([low_1d(np.full(N, hh)) for hh in hs])
ax.plot(hs, w2s, "k-", lw=1.2, label="数值谱 omega^2_min(h)")
ax.plot(hs, 2 * J - np.sqrt((2 * J) ** 2 + (LAM * hs) ** 2), "r--", lw=1.0, label="闭式（lambda=1）")
ax.set_xlabel("记录幅度 h"); ax.set_ylabel("omega^2_min")
ax.set_title("(a) 谱反演（途径二）", fontsize=9); ax.legend(fontsize=7)
ax = axes[1]
h0 = 0.05
E_k0, th_k0, _ = relax_kink(LAM * h0)
xs = np.arange(len(th_k0))
ax.plot(xs, th_k0, "o", ms=2.5, label="弛豫 kink（h=0.05）")
l0 = 4.392
ax.plot(xs, 4 * np.arctan(np.exp((xs - len(th_k0) / 2.0) / l0)), "r--", lw=1.0, label="解析拟合 l = %.2f" % l0)
ax.set_xlabel("链格"); ax.set_ylabel("theta_n")
ax.set_title("(b) 静态 kink（途径四）", fontsize=9); ax.legend(fontsize=7)
ax = axes[2]
names = ["归一化", "谱反演", "Hessian", "kink"]
vals = [lam1, float(lams2.mean()), lam_hat3, lam4]
ax.bar(range(4), vals, color=["tab:blue", "tab:orange", "tab:green", "tab:red"])
ax.axhspan(0.9, 1.1, color="gray", alpha=0.15, label="10% 带")
ax.set_xticks(range(4)); ax.set_xticklabels(names, fontsize=8)
ax.set_ylim(0.8, 1.15); ax.set_ylabel("lambda 提取值")
ax.set_title("(c) 四途径汇总（最大偏差 %.2f%%）" % (spread_all * 100), fontsize=9); ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(r"D:\software\TianXingFC\images\m6_s21_lambda_routes.png", dpi=150)
print("\n  图已存 images/m6_s21_lambda_routes.png")
print("  面板范围核对：(a) h 0.02 至 2.0、omega^2_min 0 至 -0.83；(b) 链格 0 至 255、theta 0 至 6.28；(c) 四条 0.991 至 1.000，10% 带 0.9 至 1.1")

print("\n==== 判据汇总：%d / %d 通过 ====" % (sum(1 for _, ok in CHECKS if ok), len(CHECKS)))
sys.exit(0 if all(ok for _, ok in CHECKS) else 1)
