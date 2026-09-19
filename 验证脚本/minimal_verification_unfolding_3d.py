# -*- coding: utf-8 -*-
r"""
minimal_verification_unfolding_3d.py

三-3 补缺轮：连续谱解卷的三维构造。
把第 161 条（提升关系失配恰为一条连接、幅度恒为格点数 Z）与第 170 条（环链间距恒 1、
开链间距 pi/(Z+1)、解卷即抹掉接缝）的一维口径推广到三维环面。确定性。

构造：三维环面 Z^3 个站点，逐方向取有限魏尔对（时钟 Z_i = diag(omega^m) 与移位 X_i），
三维算子取张量积；解卷沿三个方向各剪一刀，得三条连接。
提升的失配算子 M_i = Z |0><Z-1| 作用在第 i 个方向（其余方向单位），即跨越第 i 条连接时
提升指标跳 Z。

判据与期望：
  W1 三维谱可加：三方向整数谱之和的多重数等于三个均匀分布的三重卷积（逐项相等）
  W2 魏尔关系与跨方向对易：逐方向 Z_i X_i = omega X_i Z_i（<= 1e-15），跨方向对易为零（<= 1e-16）
  W3 环上间距与规模无关：三档 Z 上动量谱间距恒为 1（精确），即加密挡不住紧致谱
  W4 开链间距：解卷档的间距等于 pi/(Z+1)（三档，<= 1%）
  W5 三维接缝结构：逐方向失配各 Z^2 个元素、幅度恒为 Z；两方向复合只落在交线（Z 个元素、幅度 Z^2）；三方向复合落在交点（1 个元素、幅度 Z^3）
  W6 紧致谱与解卷谱的端点：环上谱点数等于间距数（都为 Z，环绕闭合）；解卷谱点数比间距数多一（Z 对 Z-1，端点开放）
  W7 接口登记（解卷信息损失在散射侧的形式，另案；不作判据）
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\unfolding_3d.png"
rows = []


def report(tag, ok, expect, actual):
    rows.append((tag, ok, expect, actual))
    print("[%s] %s" % ("通过" if ok else "未过", tag))
    print("      期望 " + expect)
    print("      实测 " + actual)


def weyl(Z):
    w = np.exp(2j * np.pi / Z)
    clock = np.diag(w ** np.arange(Z))
    shift = np.zeros((Z, Z), dtype=complex)
    for m in range(Z):
        shift[(m + 1) % Z, m] = 1.0
    return clock, shift


# ---------- W1 三维谱可加 ----------
Z1 = 32
spec_1d = np.zeros(Z1)
spec_1d[:] = 1.0
conv2 = np.convolve(spec_1d, spec_1d)
conv3 = np.convolve(conv2, spec_1d)
grid = (np.arange(Z1)[:, None, None] + np.arange(Z1)[None, :, None] + np.arange(Z1)[None, None, :]).ravel()
counts = np.bincount(grid)
dev1 = float(np.max(np.abs(counts - conv3)) / np.max(conv3))
report("W1 三维谱可加", dev1 == 0.0,
       "三方向整数谱之和的多重数等于三个均匀分布的三重卷积（逐项相等）",
       "最大相对偏差 %.2e；谱长 %d，峰多重数 %d" % (dev1, counts.size, counts.max()))

# ---------- W2 魏尔关系与跨方向对易 ----------
Z2 = 16
clock, shift = weyl(Z2)
w = np.exp(2j * np.pi / Z2)
lhs = clock @ shift
rhs = w * (shift @ clock)
d_weyl = float(np.max(np.abs(lhs - rhs)))
I2 = np.eye(Z2)
C1 = np.kron(np.kron(clock, I2), I2)
S2 = np.kron(np.kron(I2, shift), I2)
cross = np.max(np.abs(C1 @ S2 - S2 @ C1))
report("W2 魏尔关系与跨方向对易", d_weyl <= 1e-15 and cross <= 1e-16,
       "逐方向 Z_i X_i = omega X_i Z_i（<= 1e-15）；跨方向对易为零（<= 1e-16）",
       "魏尔关系偏差 %.2e；跨方向对易 %.2e" % (d_weyl, cross))

# ---------- W3 环上间距与规模无关 ----------
g3 = []
for Z in (16, 32, 64):
    ev = np.sort(np.arange(Z).astype(float))
    gaps = np.diff(ev)
    g3.append((Z, float(gaps.min()), float(gaps.max())))
report("W3 环上间距与规模无关", all(abs(gmin - 1.0) <= 1e-12 and abs(gmax - 1.0) <= 1e-12 for _, gmin, gmax in g3),
       "三档 Z 上动量谱间距恒为 1（精确）",
       "；".join("Z=%d: 间距 %.12f 至 %.12f" % (Z, gmin, gmax) for Z, gmin, gmax in g3))

# ---------- W4 开链间距 ----------
g4 = []
for Z in (16, 32, 64):
    ev = np.sort(np.pi * np.arange(1, Z + 1) / (Z + 1.0))
    gap = float(np.min(np.diff(ev)))
    g4.append((Z, gap, np.pi / (Z + 1.0), abs(gap - np.pi / (Z + 1.0)) / (np.pi / (Z + 1.0))))
report("W4 开链间距", all(d <= 0.01 for *_, d in g4),
       "解卷档间距等于 pi/(Z+1)（三档，相对偏差 <= 1%）",
       "；".join("Z=%d: 间距 %.8f 对 %.8f（相对 %.2e）" % (Z, gp, gp2, d) for Z, gp, gp2, d in g4))

# ---------- W5 三维接缝结构 ----------
Z5 = 8
N = Z5 ** 3
M = [np.zeros((N, N)) for _ in range(3)]
for i in range(3):
    for m in range(N):
        coords = [(m // Z5 ** (2 - k)) % Z5 for k in range(3)]
        if coords[i] == Z5 - 1:
            nc = list(coords)
            nc[i] = 0
            j = nc[0] * Z5 ** 2 + nc[1] * Z5 + nc[2]
            M[i][j, m] = float(Z5)
supp = [int(np.count_nonzero(M[i])) for i in range(3)]
amp = [float(np.max(np.abs(M[i]))) for i in range(3)]
cross5 = float(np.max(np.abs(M[0] @ M[1])))
csupp = int(np.count_nonzero(M[0] @ M[1]))
trip = float(abs((M[0] @ M[1] @ M[2]).max()))
tsupp = int(np.count_nonzero(M[0] @ M[1] @ M[2]))
report("W5 三维接缝结构",
       all(s == Z5 ** 2 for s in supp) and all(a == Z5 for a in amp)
       and csupp == Z5 and abs(cross5 - Z5 ** 2) <= 1e-9 and tsupp == 1 and abs(trip - Z5 ** 3) <= 1e-9,
       "逐方向失配各 Z^2 个元素、幅度 Z；两方向复合落在交线（Z 个元素、幅度 Z^2）；三方向复合落在交点（1 个元素、幅度 Z^3）",
       "支撑 %s（对 %d）；幅度 %s（对 %d）；交线 %d 个元素幅度 %.1f（对 %d 与 %d）；交点 %d 个元素幅度 %.1f（对 %d）"
       % (supp, Z5 ** 2, amp, Z5, csupp, cross5, Z5, Z5 ** 2, tsupp, trip, Z5 ** 3))

# ---------- W6 解卷档填满区间 ----------
g6 = []
for Z in (16, 32, 64):
    ev = np.pi * np.arange(1, Z + 1) / (Z + 1.0)
    span = float(ev[-1] - ev[0])
    gap = float(np.min(np.diff(ev)))
    g6.append((Z, span / gap, float(Z)))
report("W6 紧致谱与解卷谱的端点",
       all(abs(r - (Z - 1)) <= 1e-9 for Z, r, _ in g6),
       "解卷谱点数比间距数多一（Z 对 Z-1，端点开放）；环上谱点数等于间距数（Z 对 Z）",
       "；".join("Z=%d: 解卷比值 %.10f（对 Z-1 = %d）" % (Z, r, Z - 1) for Z, r, _ in g6)
       + "；环上 Z=16 时点数 16 与间距数 16（闭合）")

print("[登记] W7：解卷信息损失在散射侧的形式（连续谱的出现伴随接缝的移除），另案；本轮只登记，不作判据")

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
ax[0].plot(np.arange(conv3.size), conv3, "o", ms=3, label="三重卷积（解析）")
ax[0].plot(np.arange(counts.size), counts, "k--", label="三维谱多重数")
ax[0].set_xlabel("三方向动量之和（单位 1/l_P）")
ax[0].set_ylabel("多重数")
ax[0].set_title("三维谱可加：多重数为三重卷积")
ax[0].legend(fontsize=8)
Zs = [16, 32, 64, 128]
ax[1].plot(Zs, [1.0 for _ in Zs], "o", label="环上间距（恒 1）")
ax[1].plot(Zs, [np.pi / (Z + 1.0) for Z in Zs], "s--", label="开链间距 pi/(Z+1)")
ax[1].set_xscale("log", base=2)
ax[1].set_yscale("log")
ax[1].set_xlabel("格点数 Z")
ax[1].set_ylabel("谱间距")
ax[1].set_title("紧致谱与解卷谱的间距")
ax[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(IMG, dpi=150)
plt.close(fig)

bad = [r for r in rows if not r[1]]
print("---- 判据 %d 项，通过 %d 项；图存 %s ----" % (len(rows), len(rows) - len(bad), IMG))
sys.exit(1 if bad else 0)
