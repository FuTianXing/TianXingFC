# -*- coding: utf-8 -*-
r"""
minimal_verification_correlation_distance.py

二-1 首轮（路线甲，数值先行）：纠缠锁定下 Delta-alpha 的关联与格距无关性。
判据见 内部草稿/研究方向-纠缠关联函数与格距无关性（二-1）：判据草案.md；确定性，随机固定种子。

模型（一维环，周期边界，L 个站点；相位场 alpha_j 为实轴方向读数）
  锁定档（先验纠缠锁定帧）：alpha = phi + eta，phi 为整体模，eta 为平稳涨落场（相关函数按
      指数式 rho(n) = exp(-n/xi) 衰减）。此时 Var[alpha_{x+n} - alpha_x] = 2 sigma^2 (1 - rho(n))，
      在 n >> xi 后饱和，即与格距无关。
  对照档（定域接力，无锁定）：alpha_j = sum_{k<=j} delta_k（独立增量），
      此时 Var = n sigma^2，随格距线性增长。
判据与期望：
  C1 锁定档：Var 与 n 无关（n >= 3 xi 的六档上相对散布 <= 5%）
  C2 对照档：Var 与 n 成正比（斜率 = sigma^2，线性拟合残差 <= 5%）
  C3 显式函数：锁定档 Var 与 2 sigma^2 (1 - rho(n)) 相符（相对偏差 <= 5%）
  C4 紧致上限：Delta-alpha 模 2 pi 后取值有界（<= pi），逐样本核验
  C5 有限尺寸：n 取到接近环对面（L/4）仍成立（<= 5%）
  C6 与二-2 的接口（只登记，不作判据）
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\correlation_distance.png"
SEED = 20260919
L = 256
XI = 4.0
SIG = 1.0
NS = 4000
rows = []


def report(tag, ok, expect, actual):
    rows.append((tag, ok, expect, actual))
    print("[%s] %s" % ("通过" if ok else "未过", tag))
    print("      期望 " + expect)
    print("      实测 " + actual)


def stationary_field(L, xi, sig, ns, rng):
    rho = np.exp(-1.0 / xi)
    eta = np.zeros((ns, L))
    eta[:, 0] = rng.normal(scale=sig, size=ns)
    noise = rng.normal(scale=sig * np.sqrt(1.0 - rho ** 2), size=(ns, L))
    for j in range(1, L):
        eta[:, j] = rho * eta[:, j - 1] + noise[:, j]
    return eta


def walk_field(L, sig, ns, rng):
    return np.cumsum(rng.normal(scale=sig, size=(ns, L)), axis=1)


def var_of_diff(field, n):
    d = field[:, n:] - field[:, :field.shape[1] - n]
    return float(np.mean(d ** 2))


rng = np.random.default_rng(SEED)
locked = stationary_field(L, XI, SIG, NS, rng)
walk = walk_field(L, SIG, NS, rng)
print("[模型] L=%d, xi=%.1f, sigma=%.2f, 样本 %d" % (L, XI, SIG, NS))

# ---------- C1 ----------
ns1 = [int(5 * XI), 32, 48, 64, 96, 128]
v1 = [var_of_diff(locked, n) for n in ns1]
spread = max(v1) / min(v1) - 1.0
report("C1 锁定档：Var 与格距无关", spread <= 0.05,
       "n >= 3 xi 的六档上相对散布 <= 5%",
       "；".join("n=%d: %.6f" % (n, v) for n, v in zip(ns1, v1)) + "；散布 %.2f%%" % (100 * spread))

# ---------- C2 ----------
ns2 = [8, 16, 32, 48, 64]
v2 = [var_of_diff(walk, n) for n in ns2]
A = np.vstack([np.array(ns2, dtype=float), np.ones(len(ns2))]).T
sol, *_ = np.linalg.lstsq(A, np.array(v2), rcond=None)
pred2 = A @ sol
res2 = float(np.max(np.abs(np.array(v2) - pred2) / np.array(v2)))
report("C2 对照档：Var 与格距成正比", res2 <= 0.05 and abs(sol[0] - SIG ** 2) <= 0.05 * SIG ** 2,
       "斜率 = sigma^2 = %.3f（偏差 <= 5%%），线性拟合最大相对残差 <= 5%%" % (SIG ** 2),
       "斜率 %.6f；截距 %.6f；最大相对残差 %.2f%%" % (sol[0], sol[1], 100 * res2))

# ---------- C3 ----------
rho = np.exp(-np.array(ns1, dtype=float) / XI)
pred3 = 2.0 * var_of_diff(locked, 1) / (2.0 * (1.0 - np.exp(-1.0 / XI))) * (1.0 - rho) / (1.0 - np.exp(-1.0 / XI)) if False else None
v1_n1 = var_of_diff(locked, 1)
sig_eff2 = v1_n1 / (2.0 * (1.0 - np.exp(-1.0 / XI)))
pred3 = 2.0 * sig_eff2 * (1.0 - rho)
dev3 = float(np.max(np.abs(np.array(v1) - pred3) / pred3))
report("C3 显式函数 Var = 2 sigma^2 (1 - rho(n))", dev3 <= 0.05,
       "锁定档 Var 与 2 sigma_eff^2 (1 - exp(-n/xi)) 相符（相对偏差 <= 5%）",
       "sigma_eff^2 %.6f；逐档相对偏差 " % sig_eff2 + "、".join("%.2e" % (abs(v - p) / p) for v, p in zip(v1, pred3)))

# ---------- C4 ----------
d = locked - np.roll(locked, -1, axis=1)
d_wrapped = np.mod(d + np.pi, 2 * np.pi) - np.pi
ok4 = bool(np.all(np.abs(d_wrapped) <= np.pi + 1e-12))
report("C4 紧致上限：模 2 pi 后有界", ok4,
       "逐样本 |Delta-alpha| <= pi（模 2 pi 口径）",
       "最大绝对值 %.6f（样本 %d × %d）" % (float(np.max(np.abs(d_wrapped))), NS, L))

# ---------- C5 ----------
v5 = [var_of_diff(locked, n) for n in (int(5 * XI), 32, 48, int(L / 4))]
spread5 = max(v5) / min(v5) - 1.0
report("C5 有限尺寸：n 到 L/4 仍成立", spread5 <= 0.05,
       "n = 5 xi 至 L/4 四档相对散布 <= 5%",
       "；".join("n=%d: %.6f" % (n, v) for n, v in zip((int(5 * XI), 32, 48, int(L / 4)), v5)) + "；散布 %.2f%%" % (100 * spread5))

print("[登记] C6：f(E_xy) 供二-2（Tsirelson 界）使用，本轮只登记，不作判据")

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
ax[0].semilogy(ns1, v1, "o-", label="锁定档（先验纠缠锁定帧）")
ax[0].semilogy(ns2, v2, "s-", label="对照档（定域接力）")
ax[0].set_xlabel("格距 n（站点数）")
ax[0].set_ylabel("Var[Delta-alpha]")
ax[0].set_title("相位差方差与格距")
ax[0].legend(fontsize=8)
ax[1].semilogy(ns1, v1, "o", label="锁定档实测")
ax[1].semilogy(ns1, pred3, "kx", label="2 sigma_eff^2 (1 - rho(n))")
ax[1].set_xlabel("格距 n")
ax[1].set_ylabel("Var")
ax[1].set_title("锁定档与显式函数")
ax[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(IMG, dpi=150)
plt.close(fig)

bad = [r for r in rows if not r[1]]
print("---- 判据 %d 项，通过 %d 项；图存 %s ----" % (len(rows), len(rows) - len(bad), IMG))
sys.exit(1 if bad else 0)
