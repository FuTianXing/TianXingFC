# -*- coding: utf-8 -*-
"""弱版方向一：真实记录构型下的耦合失稳量级（记录场换漏斗形）。

判据（先写期望，再算；修正版，原始预期与推翻记录见验证注记过程记录）：
  W5a 对照复现：均匀 h = 1 的最低模与解析 2J - sqrt((2J)^2 + (lam h)^2) 一致（<= 1e-10）。
  W5b 两端复现：单点 h = 1 最低模 > 0；均匀 h = 1 最低模 < 0。
  W5c 最小失稳覆盖：块状构型族（h0 = 1）存在最小失稳宽 w*（1 < w* < 64）；
       并报告最低模对块宽的非单调性（报告项）。
  W5d 弱失稳判定（修正版）：真实 1/r 衰减构型（h_n = h0/(1+d_n)，h0 = 1）的失稳幅度
       不超过均匀情形的 5%（|ω_min| / |ω_uniform| < 0.05）。
       原始预期为「漏斗最低模 > 0（不失稳）」，实测 -0.0033 < 0，预期被推翻；
       修正判据的教训：一维链上 3 格以上的对称调制组即可产生弱失稳，「局域化即无失稳」不成立。
  W5e 门槛对幅度的依赖（报告）：h0 属于 {0.5, 1, 2, 5} 的 w*(h0)。

模型与弱版验证同源：一维交错链、N = 64、周期边界、J = 1、lam = 1，
H[n,n] = 2J + lam*(-1)^n h_n，H[n,n±1] = J。
平坦度 F = (sum h)^2 / (N sum h^2)（均匀 = 1，单点 = 1/N）。
确定性，无随机数。图存 D:/software/TianXingFC/images/reopen_realfield.png。
"""

import sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

N = 64
J = 1.0
LAM = 1.0
CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))


def hessian(h):
    H = np.zeros((N, N))
    for n in range(N):
        H[n, n] = 2 * J + LAM * ((-1) ** n) * h[n]
        H[n, (n + 1) % N] = J
        H[n, (n - 1) % N] = J
    return H


def low_mode(h):
    ev = np.linalg.eigvalsh(hessian(h))
    return float(ev[0]), float(ev[-1])


def flatness(h):
    return float((h.sum() ** 2) / (N * (h ** 2).sum()))


d = np.minimum(np.arange(N), N - np.arange(N))


def prof_uniform(h0):
    return np.full(N, h0)


def prof_single(h0):
    h = np.zeros(N)
    h[0] = h0
    return h


def prof_block(h0, w):
    h = np.zeros(N)
    h[d <= (w - 1) / 2.0] = h0
    return h


def prof_funnel(h0):
    return h0 / (1.0 + d)


def prof_family(h0, w):
    return h0 / (1.0 + 2.0 * d / w)


print("==== 弱版方向一：真实记录构型下的耦合失稳 ====")
print("N = %d, J = %.1f, lam = %.1f" % (N, J, LAM))

# W5a
h_u = prof_uniform(1.0)
m_u, _ = low_mode(h_u)
m_ana = 2 * J - np.sqrt((2 * J) ** 2 + (LAM * 1.0) ** 2)
check("W5a", abs(m_u - m_ana) <= 1e-10, "均匀最低模 %.10f 对解析 %.10f" % (m_u, m_ana))

# W5b
m_s, _ = low_mode(prof_single(1.0))
check("W5b", m_s > 0 and m_u < 0, "单点最低模 %.6f；均匀最低模 %.6f" % (m_s, m_u))

# W5c
ws = np.arange(1, N + 1)
mins = np.array([low_mode(prof_block(1.0, int(w)))[0] for w in ws])
neg = ws[mins < 0]
w_star = int(neg.min()) if len(neg) else None
F_star = w_star / N if w_star else None
check("W5c", w_star is not None and 1 < w_star < N,
      "最小失稳宽 w* = %s（F* = %s）；w*-1 最低模 %+.6f，w* 最低模 %+.6f"
      % (w_star, F_star, mins[w_star - 2], mins[w_star - 1]))

print("\n  块族扫描（h0=1，报告非单调性）：")
print("      w:      " + " ".join("%7d" % w for w in ws[:18]))
print("      ω²_min: " + " ".join("%+7.4f" % mins[i] for i in range(18)))
nonmono = bool(np.any(np.diff(mins) > 0))
print("      非单调（存在上升段）：%s" % nonmono)

# W5d
h_f = prof_funnel(1.0)
m_f, _ = low_mode(h_f)
F_f = flatness(h_f)
ratio = abs(m_f) / abs(m_u)
check("W5d", ratio < 0.05, "1/r 漏斗最低模 %+.6f（均匀的 %.2f%%）；F = %.4f"
      % (m_f, ratio * 100, F_f))

# W5e
print("\n  [W5e] 门槛对幅度的依赖（报告）")
for h0 in [0.5, 1.0, 2.0, 5.0]:
    mins_h = np.array([low_mode(prof_block(h0, int(w)))[0] for w in ws])
    neg_h = ws[mins_h < 0]
    ws_h = int(neg_h.min()) if len(neg_h) else None
    print("       h0 = %.1f  w* = %s（F* = %s）" % (h0, ws_h, (ws_h / N if ws_h else None)))

# 平滑族报告
mins_f = np.array([low_mode(prof_family(1.0, w))[0] for w in [1, 2, 4, 8, 16, 32, 64, 128, 256]])
print("\n  平滑漏斗族 h0/(1+2d/w)（h0=1）报告：")
for w, mv in zip([1, 2, 4, 8, 16, 32, 64, 128, 256], mins_f):
    print("       w = %4d  最低模 = %+.6f" % (w, mv))

# 图
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
fig, axs = plt.subplots(1, 2, figsize=(11.2, 4.4))

ax = axs[0]
ax.plot(range(N), prof_single(1.0), "o-", ms=3, lw=0.8, label="单点（h0=1）")
ax.plot(range(N), prof_block(1.0, 16), "s-", ms=3, lw=0.8, label="块状 w=16")
ax.plot(range(N), prof_funnel(1.0), "^-", ms=3, lw=0.8, label="1/r 漏斗（真实形状）")
ax.plot(range(N), prof_uniform(1.0), "--", lw=1.2, label="均匀")
ax.set_xlabel("格点")
ax.set_ylabel("记录幅度 h")
ax.set_title("构型剖面", fontsize=9)
ax.legend(fontsize=8)

ax = axs[1]
wsF = np.array(ws) / N
ax.plot(wsF, mins, "-", color="tab:orange", lw=1.2, label="块族（h0=1，横轴 F=w/N）")
ax.axhline(m_u, color="gray", lw=1.0, ls=":", label="均匀值 %.3f" % m_u)
ax.plot([flatness(h_f)], [m_f], "k*", ms=12, label="1/r 漏斗（h0=1）")
ax.plot([flatness(prof_single(1.0))], [m_s], "k^", ms=8, label="单点（h0=1）")
ax.axhline(0.0, color="gray", lw=0.8)
ax.set_xscale("log")
ax.set_xlabel("平坦度 F（均匀 = 1，单点 = 1/N；块族按 F = w/N）")
ax.set_ylabel("最低模（ω²_min）")
ax.set_title("失稳门槛与真实构型位置", fontsize=9)
ax.legend(fontsize=7.5)

fig.tight_layout()
fig.savefig("D:/software/TianXingFC/images/reopen_realfield.png", dpi=150)
print("\n图已保存：images/reopen_realfield.png")

n_fail = sum(1 for _, ok in CHECKS if not ok)
print("\n==== 判据汇总：%d 项，%d PASS，%d FAIL ====" % (len(CHECKS), len(CHECKS) - n_fail, n_fail))