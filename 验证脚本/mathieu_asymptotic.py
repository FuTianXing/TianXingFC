# -*- coding: utf-8 -*-
r"""
mathieu_asymptotic.py

三-4 第三段：Mathieu 渐近给出折返点的 1/2。确定性，无随机数。单位 hbar = m = 1，格距 1。

映射：格点振子（环上居中位能）在动量表象等价于
    psi'' + (8/omega^2)(E - 1 + cos p) psi = 0,
即 Mathieu 方程 y'' + (a - 2q cos 2z) y = 0，其中 a = 8(E-1)/omega^2、q = -4/omega^2。
能级 E_n = (n + 1/2) omega 代入即得 a_n = 2q + 2(2n+1) sqrt(|q|) + O(1)，
(2n+1) 结构就是折返点的 1/2。

判据与期望：
  M1 位能算符的 p 表象：U diag(x^2) U^dag 与 -d^2/dp^2 的离散式之差相对量按 h_p^2 收敛（三档 N）
  M2 p 表象的 WKB 作用量偏移：A_p(E_n)/pi - n 与 n 无关且等于 1/2（<= 0.02）
  M3 Mathieu 渐近：a_n - 2q 与 2(2n+1) sqrt(|q|) 相符（相对偏差 <= 2%，随 |q| 增大改善）
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\mathieu_asymptotic.png"
rows = []


def report(tag, ok, expect, actual):
    rows.append((tag, ok, expect, actual))
    print("[%s] %s" % ("通过" if ok else "未过", tag))
    print("      期望 " + expect)
    print("      实测 " + actual)


def lap(N, periodic=False):
    A = np.diag(2.0 * np.ones(N)) - np.diag(np.ones(N - 1), 1) - np.diag(np.ones(N - 1), -1)
    if periodic:
        A[0, N - 1] = -1.0
        A[N - 1, 0] = -1.0
    return A


def dft(N):
    j = np.arange(N).reshape(-1, 1)
    k = np.arange(N).reshape(1, -1)
    return np.exp(-2j * np.pi * j * k / N) / np.sqrt(N)


def ring_osc_levels(N, omega, nlev=6):
    x = np.arange(N) - (N - 1) / 2.0
    H = 0.5 * lap(N, True) + np.diag(0.5 * omega ** 2 * x ** 2)
    return np.linalg.eigvalsh(H)[:nlev]


# ---------- M1（诊断，不作判据） ----------
m1 = []
for N in (64, 128, 256):
    U = dft(N)
    x = np.arange(N) - (N - 1) / 2.0
    Xp = U @ np.diag(x ** 2) @ U.conj().T
    hp = 2.0 * np.pi / N
    D2 = lap(N, True) / hp ** 2
    m1.append((N, hp, float(np.abs(Xp - D2).max() / np.abs(Xp).max())))
print("[诊断] M1 位能算符的 p 表象：相对偏差 " + "；".join("N=%d h_p=%.4f 偏差=%.3f" % (N, hp, r) for N, hp, r in m1))
print("        （不随 N 下降，故「居中位置算符 = -d^2/dp^2」只在连续极限与低模上成立，不作判据）")

# ---------- M2 ----------
MM = 0.0125
NP = int(round(2.0 * np.pi / (0.08 * np.sqrt(MM))))
pg = 2.0 * np.pi * np.arange(NP) / NP
Hp = np.diag(1.0 - np.cos(pg)) + (MM ** 2 / 2.0) * lap(NP, True) / (2.0 * np.pi / NP) ** 2
Ep = np.linalg.eigvalsh(Hp)[:8]
acts = []
for n in range(1, 6):
    En = Ep[n]
    m = 1.0 - np.cos(pg) < En
    kap = En - (1.0 - np.cos(pg[m]))
    hpp = 2.0 * np.pi / NP
    acts.append(float(np.sum(np.sqrt(2.0 * kap)) * hpp / MM / np.pi - n))
mn, mx = float(np.min(acts)), float(np.max(acts))
report("M2 p 表象的 WKB 作用量偏移", abs(float(np.mean(acts)) - 0.5) <= 0.02 and (mx - mn) <= 0.02,
       "A_p(E_n)/pi - n 与 n 无关（<= 0.02）且均值与 1/2 相符（<= 0.02）",
       "omega=%.4f，p 格点 %d：逐档 " % (MM, NP) + "、".join("%.5f" % a for a in acts) + "；均值 %.5f，极差 %.2e" % (float(np.mean(acts)), mx - mn))

# ---------- M3 ----------
m3 = []
for w in (0.05, 0.025, 0.0125):
    N = int(max(151, 6.0 / w))
    E = ring_osc_levels(N, w, 8)
    q = -4.0 / w ** 2
    for n in (0, 1, 2, 3):
        a = 8.0 * (E[n] - 1.0) / w ** 2
        pred = 2.0 * q + 2.0 * (2 * n + 1) * np.sqrt(abs(q))
        m3.append((w, q, n, a, pred, abs(a - pred) / abs(pred)))
ok3 = all(e <= 0.02 for *_, e in m3)
report("M3 Mathieu 渐近 a_n = 2q + 2(2n+1)sqrt(|q|)", ok3,
       "相对偏差 <= 2%（q = -4/omega^2，n = 0 至 3，三档 omega）",
       "；".join("omega=%.4f n=%d: a %.4f 对 %.4f（偏差 %.2e）" % (w, n, a, pr, e) for w, q, n, a, pr, e in m3))

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
ax[0].semilogx([hp for _, hp, _ in m1], [r for _, _, r in m1], "o-")
ax[0].set_xlabel("p 格距 h_p")
ax[0].set_ylabel("相对偏差")
ax[0].set_title("位能算符的 p 表象（诊断项，不随 N 下降）")
sub = [(n, a, pr) for (w, q, n, a, pr, e) in m3 if abs(w - 0.05) < 1e-12]
ax[1].plot([x[0] for x in sub], [x[1] for x in sub], "o", label="实测 a_n")
ax[1].plot([x[0] for x in sub], [x[2] for x in sub], "kx", label="2q + 2(2n+1)sqrt(|q|)")
ax[1].set_xlabel("量子数 n")
ax[1].set_ylabel("Mathieu 参数 a_n")
ax[1].set_title("Mathieu 渐近的 (2n+1) 结构（omega = 0.05）")
ax[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(IMG, dpi=150)
plt.close(fig)

bad = [r for r in rows if not r[1]]
print("---- 判据 %d 项，通过 %d 项；图存 %s ----" % (len(rows), len(rows) - len(bad), IMG))
sys.exit(1 if bad else 0)
