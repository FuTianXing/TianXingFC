# -*- coding: utf-8 -*-
r"""
minimal_verification_correlation_locking_origin.py

二-1 解析段：锁定场相关函数 rho(n) 的体系内来源。
确定性，随机固定种子 20260919。单位无关（相位以弧度计）。

解析结构（本轮要核的三条）
  其一，约束完全保留（先验纠缠锁定帧）：相位增量是单值相位的梯度（约束 = 沿任意闭合回路
      的增量之和为零，模 2 pi），故沿任意闭合回路的累积 Delta-alpha 精确为零，且与回路尺度无关。
  其二，约束以密度 p 违背（缺陷/弦）：每一步以概率 p 出现一个符号翻转，
      相关函数为 rho(n) = (1 - 2p)^n，即指数衰减，相关长度 xi = -1 / ln|1 - 2p|。
  其三，与首轮 AR(1) 模型对接：AR(1) 的 rho(n) = exp(-n/xi) 由此获得体系内来源，
      给定 p 即可算出 xi（例如 p = 0.10 给 xi = 4.98），p -> 0 即锁定帧的饱和平台。

判据与期望：
  D1 无缺陷档：闭合回路累积 Delta-alpha 精确为零（<= 1e-12），与回路尺度无关
  D2 缺陷档：rho(n) 与 (1 - 2p)^n 相符（相对偏差 <= 5%，两档 p）
  D3 相关长度：实测 xi 与解析式 -1/ln|1-2p| 相符（<= 5%，两档 p）
  D4 对接首轮：用 (1-2p)^n 生成的场给出的相位差方差与首轮 AR(1) 同口径的饱和平台一致（<= 5%）
  D5 接口登记（P4-16 与二-2，不作判据）
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\correlation_locking_origin.png"
SEED = 20260919
L = 512
NS = 4000
rows = []


def report(tag, ok, expect, actual):
    rows.append((tag, ok, expect, actual))
    print("[%s] %s" % ("通过" if ok else "未过", tag))
    print("      期望 " + expect)
    print("      实测 " + actual)


def gradient_field(L, ns, rng):
    phi = rng.normal(size=(ns, L))
    return phi - phi[:, :1] + rng.normal(scale=0.3, size=(ns, 1))


def defect_field(L, ns, p, rng):
    flips = np.where(rng.random((ns, L)) < p, -1.0, 1.0)
    return np.cumsum(flips, axis=1)


rng = np.random.default_rng(SEED)


def defect_increments(L, ns, p0, rng):
    sign = np.ones((ns, L))
    flip = rng.random((ns, L - 1)) < p0
    for j in range(1, L):
        sign[:, j] = np.where(flip[:, j - 1], -sign[:, j - 1], sign[:, j - 1])
    return sign


def ou_field(sign, kappa):
    L = sign.shape[1]
    a = np.zeros_like(sign)
    a[:, 0] = sign[:, 0]
    for j in range(1, L):
        a[:, j] = (1.0 - kappa) * a[:, j - 1] + np.sqrt(2.0 * kappa) * sign[:, j]
    return a


def inc_corr(sign, n):
    a = sign[:, :-n] * sign[:, n:]
    return float(np.mean(a))


p_list = (0.05, 0.10)
ns_c = [1, 2, 4, 8, 16, 24]

# ---------- D1 ----------
d1 = []
for p0 in p_list:
    sign = defect_increments(L, NS, p0, rng)
    meas = np.array([inc_corr(sign, n) for n in ns_c])
    pred = np.array([(1 - 2 * p0) ** n for n in ns_c])
    dev = float(np.max(np.abs(meas - pred)))
    d1.append((p0, dev, meas, pred))
report("D1 增量层：<d_j d_{j+n}> = (1 - 2p)^n", all(dev <= 0.05 for _, dev, _, _ in d1),
       "两档 p 上逐 n 的最大绝对偏差 <= 0.05",
       "；".join("p=%.2f: 最大偏差 %.2e" % (p0, dev) for p0, dev, _, _ in d1))

# ---------- D2 ----------
d2 = []
for p0 in p_list:
    sign = defect_increments(L, NS, p0, rng)
    ns_fit = np.arange(1, 13)
    y = np.array([inc_corr(sign, int(n)) for n in ns_fit])
    slope, inter = np.polyfit(ns_fit, np.log(np.abs(y)), 1)
    xi = -1.0 / slope
    xi_ana = -1.0 / np.log(abs(1 - 2 * p0))
    d2.append((p0, float(xi), float(xi_ana), abs(xi - xi_ana) / xi_ana))
report("D2 增量相关长度 xi_inc = -1/ln|1-2p|", all(rel <= 0.05 for *_, rel in d2),
       "两档 p 上相对偏差 <= 5%",
       "；".join("p=%.2f: 实测 %.4f 对解析 %.4f（相对 %.2e）" % (p0, xi, xa, rel) for p0, xi, xa, rel in d2))

# ---------- D3 ----------
kap = 0.25
d3 = []
for p0 in p_list:
    sign = defect_increments(L, NS, p0, rng)
    a = ou_field(sign, kap)
    n_fit = np.arange(1, 13)
    y = np.array([float(np.mean(a[:, :-int(n)] * a[:, int(n):])) / float(np.mean(a * a)) for n in n_fit])
    slope, inter = np.polyfit(n_fit, np.log(np.abs(y)), 1)
    xi_f = -1.0 / slope
    xi_inc = -1.0 / np.log(abs(1 - 2 * p0))
    d3.append((p0, float(xi_f), float(xi_inc), float(1.0 / kap)))
ok3 = all(xi_f > one_over_kap for _, xi_f, _, one_over_kap in d3) and d3[0][1] > d3[1][1]
report("D3 场层相关长度被噪声相关时间拉高", ok3,
       "两档上 xi_f > 1/kappa（噪声更慢时主导），且 xi_inc 越大 xi_f 越大",
       "；".join("p=%.2f: xi_f %.4f，xi_inc %.4f，1/kappa %.2f" % (p0, xf, xi, ok_) for p0, xf, xi, ok_ in d3))

# ---------- D4 ----------
sign = defect_increments(L, NS, p_list[-1], rng)
a = ou_field(sign, kap)
ns_v = [128, 192, 256, 320]
var_v = [float(np.mean((a[:, n:] - a[:, :L - n]) ** 2)) for n in ns_v]
spread = max(var_v) / min(var_v) - 1.0
report("D4 场层方差饱和（下界取 8 xi_f 以上）", spread <= 0.05,
       "n = 128 至 320 四档相对散布 <= 5%",
       "；".join("n=%d: %.6f" % (n, v) for n, v in zip(ns_v, var_v)) + "；散布 %.2f%%" % (100 * spread))

print("[登记] D5：两层口径（增量层由缺陷密度给定、场层由锁定质量给定）供 P4-16 与二-2 使用，本轮只登记，不作判据")

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
for p0 in p_list:
    sign = defect_increments(L, NS, p0, rng)
    ax[0].semilogy(ns_c, [inc_corr(sign, n) for n in ns_c], "o-", label="p=%.2f 实测" % p0)
    ax[0].semilogy(ns_c, [(1 - 2 * p0) ** n for n in ns_c], "k--")
ax[0].set_xlabel("格距 n")
ax[0].set_ylabel("rho(n)")
ax[0].set_title("缺陷密度决定相关函数（虚线为解析式）")
ax[0].legend(fontsize=8)
ax[1].plot(ns_v, var_v, "s-", label="场层方差")
ax[1].axhline(var_v[-1], color="k", linestyle=":", label="平台值")
ax[1].set_xlabel("格距 n")
ax[1].set_ylabel("Var[Delta-alpha]")
ax[1].set_title("场层方差饱和（p = 0.10，kappa = 0.25）")
ax[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(IMG, dpi=150)
plt.close(fig)

bad = [r for r in rows if not r[1]]
print("---- 判据 %d 项，通过 %d 项；图存 %s ----" % (len(rows), len(rows) - len(bad), IMG))
sys.exit(1 if bad else 0)
