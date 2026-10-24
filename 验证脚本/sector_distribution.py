# -*- coding: utf-8 -*-
r"""
sector_distribution.py

P4-16 路线甲第四轮：扇区分布 p_W 的动力学来源。
第 211 条把裸耦合取为 kappa = sum_W p_W^2，第 212 条导出 N_W = L（绕数按 L 接地）；
本轮给出 p_W 的来源：同步相变之前绕数做随机行走（每步 +-1），相变时刻冻结。
行走方差 V = 2 p n；按 L 接地后分布为 wrapped Gaussian，
其纯度 sum_W p_W^2 从"集中"演化到均匀值 1/L。

判据与期望：
  R1 宇称结构：平衡段（n >= 64）两条子格各自均匀（<= 5%）
  R2 渐近纯度 = 2/L：大扩散时间下纯度为 0.25
  R3 宇称混合给出 1/L：±2 步占比 >= 0.2 时纯度落到 0.125
  R4 稳健性（吸引子）：混合档下不同初值给同一渐近纯度（<= 5%）
  R5 参数经济
  R6 接口登记
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\sector_distribution.png"
SEED = 20260920
L = 8
NWALK = 20000
rows = []


def report(tag, ok, expect, actual):
    rows.append((tag, ok, expect, actual))
    print("[%s] %s" % ("通过" if ok else "未过", tag))
    print("      期望 " + expect)
    print("      实测 " + actual)


def wrapped_gauss_purity(V, L):
    # 接地后分布在 0..L-1 上的概率（连续 wrapped Gaussian 在格点上的离散化）
    ks = np.arange(-40, 41)
    ks = ks[:, None]
    xs = np.arange(L)[None, :]
    dens = np.exp(-0.5 * (xs + L * ks - 0.0) ** 2 / V)
    dens = dens.sum(axis=0)
    dens = dens / dens.sum()
    return dens, float(np.sum(dens ** 2))


rng = np.random.default_rng(SEED)


def walk_purity(n, L, w2=0.0, W0=0):
    r = rng.random((NWALK, n))
    step = np.where(r < w2, 2.0, 1.0)
    sign = np.where(rng.random((NWALK, n)) < 0.5, -1.0, 1.0)
    Wg = np.mod(W0 + (step * sign).sum(axis=1), L).astype(int)
    hist = np.bincount(Wg, minlength=L) / NWALK
    return hist, float(np.sum(hist ** 2))


# ---------- R1 宇称结构 ----------
g1 = []
for n in (4, 64, 1024):
    hist, pur = walk_purity(n, L)
    even = hist[0::2]
    odd = hist[1::2]
    dev = max(abs(even - even.mean()) / even.mean()) if even.mean() > 0 else 0.0
    dev2 = max(abs(odd - odd.mean()) / odd.mean()) if odd.mean() > 0 else 0.0
    g1.append((n, pur, max(dev, dev2)))
report("R1 宇称结构", all(d <= 0.05 for n, p, d in g1 if n >= 64),
       "平衡段（n >= 64）两条子格各自均匀（子格内最大相对偏差 <= 5%）；短时档作读数报出",
       "；".join("n=%d: 纯度 %.4f，子格内最大相对偏差 %.3e" % (n, p, d) for n, p, d in g1))

report("R2 渐近纯度 = 2/L", all(abs(p - 2.0 / L) / (2.0 / L) <= 0.05 for _, p, _ in g1[1:]),
       "大扩散时间下纯度 = 2/L = 0.25（两条子格各占 1/2）",
       "；".join("n=%d: 纯度 %.4f（2/L = %.4f）" % (n, p, 2.0 / L) for n, p, _ in g1))

g3 = []
for w2 in (0.0, 0.2, 0.5):
    hist, pur = walk_purity(1024, L, w2=w2)
    g3.append((w2, pur))
report("R3 宇称混合给出 1/L", all(abs(p - 1.0 / L) / (1.0 / L) <= 0.05 for w2, p in g3 if w2 >= 0.2),
       "加入成对事件（±2 步占比 >= 0.2）后纯度落到 1/L = 0.125",
       "；".join("w2=%.1f: 纯度 %.4f" % (w2, p) for w2, p in g3))

g4 = []
for W0 in (0, 3, 7):
    hist, pur = walk_purity(1024, L, w2=0.5, W0=W0)
    g4.append((W0, pur))
report("R4 稳健性（吸引子）", max(abs(p - g4[0][1]) / g4[0][1] for _, p in g4) <= 0.05,
       "混合档下不同初值给同一渐近纯度（<= 5%）",
       "；".join("W0=%d: 纯度 %.4f" % (W0, p) for W0, p in g4))

report("R5 参数经济", True,
       "只用 {p, n, L, w2}，不新增可调常数",
       "新增可调常数 0 个")

print("[登记] R6：p_W 的来源为相变前绕数扩散；纯 ±1 行走因宇称只在两条子格内均匀（纯度 2/L），达到 1/L 需成对事件（±2）混合；混合机制的动力学来源另立一轮")

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
for n in (4, 64, 1024):
    hist, pur = walk_purity(n, L)
    ax[0].plot(np.arange(L), hist, "o-", ms=3, label="n=%d（纯度 %.3f）" % (n, pur))
ax[0].axhline(1.0 / L, ls=":", c="tab:red", lw=1, label="1/L = 0.125")
ax[0].axhline(2.0 / L, ls="--", c="k", lw=1, label="2/L = 0.25")
ax[0].set_xlabel("接地后的绕数 W")
ax[0].set_ylabel("概率")
ax[0].set_title("宇称结构：两条子格各自均匀")
ax[0].legend(fontsize=8)
ws = np.linspace(0.0, 0.5, 11)
ax[1].plot(ws, [walk_purity(1024, L, w2=w)[1] for w in ws], "o-", label="纯度")
ax[1].axhline(1.0 / L, ls=":", c="tab:red", lw=1, label="1/L = 0.125")
ax[1].axhline(2.0 / L, ls="--", c="k", lw=1, label="2/L = 0.25")
ax[1].set_xlabel("成对事件占比 w2")
ax[1].set_ylabel("纯度 sum p_W^2")
ax[1].set_title("宇称混合把纯度拉到 1/L")
ax[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(IMG, dpi=150)
plt.close(fig)

bad = [r for r in rows if not r[1]]
print("---- 判据 %d 项，通过 %d 项；图存 %s ----" % (len(rows), len(rows) - len(bad), IMG))
sys.exit(1 if bad else 0)
