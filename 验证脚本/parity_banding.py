# -*- coding: utf-8 -*-
"""奇偶分档的精确形态：块族最低模对半宽奇偶的带边锁定（弱版方向一的遗留待分析项）。

判据（先写期望，再算）：
  PB1 参数化澄清（报告 + 判据）：旧材料的「块宽 w」中 w 与 w+1 生成同一构型
       （h[d <= (w-1)/2] 的整数截断），有效参数为调制格点数 m = 2k+1 的半宽 k；
       期望：w 为奇数时 prof(w) 与 prof(w+1) 逐位相等（w = 1..39 全过）。
  PB2 带边锁定（奇半宽）：k 为奇数（1 至 15）时，最低模精确等于均匀带边值 2 - sqrt(5)
       （偏差 <= 1e-12）。
  PB3 显式本征矢（奇半宽）：锁定时本征矢为「平台 + 几何尾」精确形态：
       平台区 |n| <= k+1（偶格点 P、奇格点 -phi*P；phi = (1+sqrt(5))/2），
       尾部 |n| >= k+1 每格乘 -1/phi；期望：H v = mu v（相对残差 <= 1e-10），
       且与数值本征矢重叠 |<v_num, v_ans>| >= 1 - 1e-12。
  PB4 非锁定（偶半宽）：k 为偶数（2 至 16）时最低模严格高于带边值（偏离 > 0），
       且偏离满足匹配条件 T_{K+1}(c)/T_K(c) = rho^2 - rho
       （K = k/2、c = cos q = (s^2-3)/2、s = 2 - mu、rho = (s - sqrt(s^2-4))/2 为外侧衰减根）；
       期望：条件残差 <= 1e-10。
  PB5 代数特例（报告 + 判据）：k = 2 时 c = 塑料常数/2，满足 8c^3 - 2c - 1 = 0（<= 1e-12）。

模型：与弱版方向一同源的一维交错链、周期边界、J = lam = 1、
H[n,n] = 2J + lam*(-1)^n h_n、H[n,n±1] = J；块构型 h_n = 1 于 |n| <= k。
格距 N = 256（两端尾部不重叠，见注记诚实边界）。确定性，无随机数。
图存 D:/software/TianXingFC/images/parity_banding.png。
"""

import sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

N = 256
J = 1.0
LAM = 1.0
PHI = (1.0 + np.sqrt(5.0)) / 2.0
MU_EDGE = 2.0 - np.sqrt(5.0)
CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))


d = np.minimum(np.arange(N), N - np.arange(N))


def prof_block_w(w):
    h = np.zeros(N)
    h[d <= (w - 1) / 2.0] = 1.0
    return h


def prof_block_k(k):
    h = np.zeros(N)
    h[d <= k] = 1.0
    return h


def hessian(h):
    H = np.zeros((N, N))
    for n in range(N):
        H[n, n] = 2 * J + LAM * ((-1) ** n) * h[n]
        H[n, (n + 1) % N] = J
        H[n, (n - 1) % N] = J
    return H


def low_mode(h):
    ev, evec = np.linalg.eigh(hessian(h))
    return float(ev[0]), evec[:, 0]


def ansatz_locked(k):
    v = np.zeros(N)
    for n in range(N):
        s = n if n <= N // 2 else n - N
        a = abs(s)
        if a <= k + 1:
            v[n] = 1.0 if s % 2 == 0 else -PHI
        else:
            v[n] = ((-1) ** (a - (k + 1))) * (PHI ** (-(a - (k + 1))))
    return v


print("==== 奇偶分档的精确形态：半宽奇偶与带边锁定 ====")
print("N = %d, J = %.1f, lam = %.1f, 带边值 2 - sqrt(5) = %.12f" % (N, J, LAM, MU_EDGE))

# PB1
same = all(np.array_equal(prof_block_w(w), prof_block_w(w + 1)) for w in range(1, 40, 2))
ms = [int(prof_block_w(w).sum()) for w in range(1, 13)]
check("PB1", same, "w 与 w+1 轮廓逐位相等（w = 1,3,...,39）；w = 1..12 的格点数 %s" % ms)

# PB2
k_odd = list(range(1, 16, 2))
devs_odd = []
for k in k_odd:
    mu, _ = low_mode(prof_block_k(k))
    devs_odd.append(mu - MU_EDGE)
worst_odd = max(abs(x) for x in devs_odd)
check("PB2", worst_odd <= 1e-12, "奇半宽 k = %s：最大偏差 %.3e（期望 <= 1e-12）"
      % (k_odd, worst_odd))

# PB3
worst_res = 0.0
worst_ovl = 1.0
for k in [1, 3, 5, 7, 9]:
    mu, vnum = low_mode(prof_block_k(k))
    v = ansatz_locked(k)
    v = v / np.linalg.norm(v)
    H = hessian(prof_block_k(k))
    res = float(np.linalg.norm(H @ v - MU_EDGE * v) / np.linalg.norm(v))
    ovl = float(abs(np.dot(vnum, v)))
    worst_res = max(worst_res, res)
    worst_ovl = min(worst_ovl, ovl)
check("PB3", worst_res <= 1e-10 and worst_ovl >= 1 - 1e-12,
      "解析矢相对残差最大 %.3e（期望 <= 1e-10）；与数值本征矢重叠最低 %.15f" % (worst_res, worst_ovl))

# 平台比值报告（k = 7）
k = 7
mu, vnum = low_mode(prof_block_k(k))
vnum = vnum / np.linalg.norm(vnum)
even_amp = abs(vnum[(0 + N) % N])
odd_amp = abs(vnum[(1 + N) % N])
check("PB3b", abs(odd_amp / even_amp - PHI) <= 1e-12,
      "k = 7 平台相邻比 |v_奇|/|v_偶| = %.15f（期望 phi = %.15f）" % (odd_amp / even_amp, PHI))

# PB4
k_even = list(range(2, 17, 2))
worst_match = 0.0
devs_even = []
for k in k_even:
    mu, _ = low_mode(prof_block_k(k))
    dev = mu - MU_EDGE
    devs_even.append(dev)
    s = 2.0 - mu
    c = (s * s - 3) / 2
    rho = (s - np.sqrt(s * s - 4)) / 2
    q = np.arccos(c)
    K = k // 2
    lhs = np.cos(q * (K + 1)) / np.cos(q * K)
    rhs = rho * rho - rho
    worst_match = max(worst_match, abs(lhs - rhs))
check("PB4", all(x > 0 for x in devs_even) and worst_match <= 1e-10,
      "偶半宽 k = %s：偏离全为正（最小 %.3e）；匹配条件最大残差 %.3e"
      % (k_even, min(devs_even), worst_match))

# PB5
mu, _ = low_mode(prof_block_k(2))
s = 2.0 - mu
c = (s * s - 3) / 2
res_plastic = abs(8 * c ** 3 - 2 * c - 1)
check("PB5", res_plastic <= 1e-12, "k = 2 的 c = %.15f，8c^3-2c-1 = %.3e（期望 <= 1e-12）" % (c, res_plastic))

# 报告：偶半宽偏离的衰减
print("\n  [PB6] 报告：偶半宽偏离（mu - 带边值）随 k 的衰减（k 至 40）")
for k in range(2, 41, 2):
    mu_k, _ = low_mode(prof_block_k(k))
    dev = mu_k - MU_EDGE
    print("       k = %2d  偏离 = %.6e  偏离*(k+1)^2 = %.4f" % (k, dev, dev * (k + 1) ** 2))

# 图
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
fig, axs = plt.subplots(1, 2, figsize=(11.2, 4.4))

ax = axs[0]
ks = list(range(1, 21))
mus = [low_mode(prof_block_k(k))[0] for k in ks]
ax.plot([k for k in ks if k % 2 == 1], [m for k, m in zip(ks, mus) if k % 2 == 1],
        "o", ms=6, color="tab:blue", label="奇半宽（锁定带边）")
ax.plot([k for k in ks if k % 2 == 0], [m for k, m in zip(ks, mus) if k % 2 == 0],
        "s", ms=6, color="tab:red", label="偶半宽（高于带边）")
ax.axhline(MU_EDGE, color="gray", lw=1.0, ls=":", label="带边值 %.4f" % MU_EDGE)
ax.axhline(0.0, color="gray", lw=0.8)
ax.set_xlabel("半宽 k（调制格点数 m = 2k+1）")
ax.set_ylabel("最低模（ω²_min）")
ax.set_title("半宽奇偶与带边锁定", fontsize=9)
ax.legend(fontsize=8)

ax = axs[1]
k = 5
mu, vnum = low_mode(prof_block_k(k))
sel = np.arange(-16, 17)
vals_num = np.array([abs(vnum[(s + N) % N]) for s in sel])
v = ansatz_locked(k)
v = v / np.linalg.norm(v)
vals_ana = np.array([abs(v[(s + N) % N]) for s in sel])
ax.semilogy(sel, vals_num, "o", ms=5, color="tab:blue", label="数值本征矢（k = 5）")
ax.semilogy(sel, vals_ana, "-", lw=1.0, color="tab:red", label="解析形态（平台 + 几何尾）")
ax.axvline(k + 1, color="gray", lw=0.8, ls="--")
ax.text(k + 1.3, 1.4e-3, "平台边界 |n| = k+1 = %d" % (k + 1), fontsize=8, color="gray")
ax.axvline(-(k + 1), color="gray", lw=0.8, ls="--")
ax.set_xlabel("格点 n（块中心为 0）")
ax.set_ylabel("|v_n|（对数轴）")
ax.set_title("锁定本征矢：平台与按 1/phi 衰减的尾部", fontsize=9)
ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig(r"D:/software/TianXingFC/images/parity_banding.png", dpi=150)
print("\n  图已保存：images/parity_banding.png")

n_pass = sum(1 for _, ok in CHECKS if ok)
print("\n==== 判据汇总：%d / %d 通过 ====" % (n_pass, len(CHECKS)))
sys.exit(0 if n_pass == len(CHECKS) else 1)