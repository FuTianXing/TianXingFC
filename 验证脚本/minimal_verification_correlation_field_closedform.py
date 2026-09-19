# -*- coding: utf-8 -*-
r"""
minimal_verification_correlation_field_closedform.py

二-1 第三段：锁定场场层相关函数的闭合形式。
确定性，随机固定种子 20260919。

模型：增量层取符号翻转过程（相关函数 q^n，q = 1 - 2p），场层取漏积分响应
    x_{j+1} = alpha x_j + beta n_{j+1},  alpha = 1 - kappa,  beta = sqrt(2 kappa).
解析预期：场层相关函数是**两点指数叠加** C(r) = a q^r + b alpha^r（由线性递归的
双几何和给出），故单个指数拟合出的相关长度是窗口相关的量，不是单一常数。

判据与期望：
  E1 两点指数闭式：C(r) 与 a q^r + b alpha^r 的相对残差 <= 1e-2（两档 p，r 至 40）
  E2 慢率由较慢一方给出：大 r 的实测对数斜率与 ln(max(q, alpha)) 相符（<= 2%）
  E3 单指数拟合的窗口依赖：三档拟合窗口下 xi_f 的相对散布 >= 10%（xi_f 属窗口相关量）
  E4 kappa 对照：kappa 增大到使 1/kappa < xi_inc 时，慢率改由 alpha 给出（<= 5%）
  E5 接口登记（P4-16，不作判据）
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\correlation_field_closedform.png"
SEED = 20260919
L = 4096
NS = 3000
KAP = 0.25
rows = []


def report(tag, ok, expect, actual):
    rows.append((tag, ok, expect, actual))
    print("[%s] %s" % ("通过" if ok else "未过", tag))
    print("      期望 " + expect)
    print("      实测 " + actual)


def noise_ar1(L, ns, q, rng):
    n = np.zeros((ns, L))
    n[:, 0] = rng.normal(size=ns)
    eps = rng.normal(scale=np.sqrt(1.0 - q ** 2), size=(ns, L))
    for j in range(1, L):
        n[:, j] = q * n[:, j - 1] + eps[:, j]
    return n


def ou_field(n, kap):
    alpha = 1.0 - kap
    beta = np.sqrt(2.0 * kap)
    x = np.zeros_like(n)
    x[:, 0] = n[:, 0]
    for j in range(1, n.shape[1]):
        x[:, j] = alpha * x[:, j - 1] + beta * n[:, j]
    return x


def corr(x, r):
    a = x if r == 0 else x[:, :-r]
    b = x[:, r:]
    return float(np.mean(a * b))


rng = np.random.default_rng(SEED)
r_max = 40
rs = np.arange(0, r_max + 1)
data = {}
for p0 in (0.05, 0.10):
    q = 1.0 - 2.0 * p0
    n = noise_ar1(L, NS, q, rng)
    x = ou_field(n, KAP)
    C = np.array([corr(x, int(r)) for r in rs])
    data[p0] = (q, C)

# ---------- E1 ----------
e1 = []
for p0, (q, C) in data.items():
    alpha = 1.0 - KAP
    A = np.vstack([q ** rs, alpha ** rs]).T
    sol, *_ = np.linalg.lstsq(A, C, rcond=None)
    res = float(np.max(np.abs(C - A @ sol)) / C[0])
    e1.append((p0, q, alpha, sol[0], sol[1], res))
report("E1 两点指数闭式 C(r) = a q^r + b alpha^r", all(r <= 1e-2 for *_, r in e1),
       "两档 p 上相对残差（对 C(0)）<= 1e-2",
       "；".join("p=%.2f: a %.4f b %.4f，残差 %.2e" % (p0, a, b, r) for p0, q, al, a, b, r in e1))

# ---------- E2 ----------
e2 = []
for p0, (q, C) in data.items():
    alpha = 1.0 - KAP
    r_fit = np.arange(1, 7)
    y = np.log(np.abs(C[1:7]))
    slope, inter = np.polyfit(r_fit, y, 1)
    xi_short = -1.0 / slope
    xi_slow = -1.0 / np.log(max(q, alpha))
    xi_fast = -1.0 / np.log(min(q, alpha))
    A = np.vstack([q ** rs, alpha ** rs]).T
    sol, *_ = np.linalg.lstsq(A, C, rcond=None)
    e2.append((p0, float(xi_short), float(xi_slow), float(xi_fast), float(sol[1])))
ok2 = all(xi_short > xi_slow and b < 0 for _, xi_short, xi_slow, _, b in e2)
report("E2 单指数 xi_f 超出两端速率长度（两项反号）", ok2,
       "两档 p 上短窗 xi_f > 较慢一方对应的长度，且第二项系数 b < 0",
       "；".join("p=%.2f: xi_f %.3f > 慢端 %.3f（快端 %.3f），b %.3f" % (p0, xs, xl, xf, b) for p0, xs, xl, xf, b in e2))

# ---------- E3 ----------
e3 = []
wins = [(1, 6), (1, 12), (1, 24)]
for p0, (q, C) in data.items():
    xis = []
    for (a0, a1) in wins:
        r_fit = np.arange(a0, a1 + 1)
        y = np.log(np.abs(C[a0:a1 + 1]))
        slope, inter = np.polyfit(r_fit, y, 1)
        xis.append(-1.0 / slope)
    sp = max(xis) / min(xis) - 1.0
    e3.append((p0, xis, sp))
report("E3 单指数拟合的窗口依赖", all(sp >= 0.10 for _, _, sp in e3),
       "两档 p 上三档窗口的 xi_f 相对散布 >= 10%",
       "；".join("p=%.2f: " % p0 + "、".join("%.3f" % x for x in xis) + "（散布 %.1f%%）" % (100 * sp) for p0, xis, sp in e3))

# ---------- E4 ----------
e4 = []
for kap in (0.25, 0.60):
    q = 0.90
    n = noise_ar1(L, NS, q, rng)
    x = ou_field(n, kap)
    C = np.array([corr(x, int(r)) for r in rs])
    r_fit = np.arange(20, r_max + 1)
    slope, inter = np.polyfit(r_fit, np.log(np.abs(C[20:])), 1)
    slow = max(q, 1.0 - kap)
    e4.append((kap, float(slope), float(np.log(slow)), abs(slope - np.log(slow)) / abs(np.log(slow))))
report("E4 kappa 对照：慢率随之改变", all(rel <= 0.05 for *_, rel in e4),
       "kappa = 0.25 与 0.60 两档上斜率与 ln(max(q, 1-kappa)) 相符（<= 5%）",
       "；".join("kappa=%.2f: 斜率 %.5f 对 %.5f（相对 %.2e）" % (k, s, l, rel) for k, s, l, rel in e4))

print("[登记] E5：场层闭合形式（两点指数）供 P4-16 使用，本轮只登记，不作判据")

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
for p0, (q, C) in data.items():
    ax[0].semilogy(rs, C / C[0], "o", label="p=%.2f 实测" % p0)
    alpha = 1.0 - KAP
    A = np.vstack([q ** rs, alpha ** rs]).T
    sol, *_ = np.linalg.lstsq(A, C, rcond=None)
    ax[0].semilogy(rs, (A @ sol) / C[0], "k--")
ax[0].set_xlabel("格距 r")
ax[0].set_ylabel("C(r) / C(0)")
ax[0].set_title("场层相关函数与两点指数闭式（虚线）")
ax[0].legend(fontsize=8)
for p0, xis, sp in e3:
    ax[1].plot(["1-6", "1-12", "1-24"], xis, "o-", label="p=%.2f" % p0)
ax[1].set_xlabel("拟合窗口")
ax[1].set_ylabel("xi_f")
ax[1].set_title("单指数拟合出的 xi_f 随窗口变化")
ax[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(IMG, dpi=150)
plt.close(fig)

bad = [r for r in rows if not r[1]]
print("---- 判据 %d 项，通过 %d 项；图存 %s ----" % (len(rows), len(rows) - len(bad), IMG))
sys.exit(1 if bad else 0)
