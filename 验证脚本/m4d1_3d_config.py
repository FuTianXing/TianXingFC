# -*- coding: utf-8 -*-
"""M4-D1：三维构型层读数（耦合失稳的一维读数三维化）。

判据（先写期望，再算；设计见《黑洞线攻坚计划：坍缩机制重定位与强场终态》§7.1）：
  D1-1a 解析锚点：均匀记录（h 处处为 1）的三维最低模等于 6 - sqrt(37)（<= 1e-10）。
  D1-1b 约化锚点：薄板（记录只依赖一个坐标）的最低模等于两分量一维链
        (2-mu)phi + delta phi + (-1)^i h_i psi = 0、(10-mu)psi + delta psi + (-1)^i h_i phi = 0
        的最低模（<= 1e-10）。
  D1-2a 立方体符号门槛：边长 1 的立方体最低模为正、边长 2 为负；该符号在格距 12、16、24 上保持。
  D1-2b 柱体扫描（报告）：截面 2x2 与 3x3、长度 1 至 6 的读数。
  D1-3a 三维漏斗衰减：h_n = h_0/(1+r_n)（r 为最小镜像欧氏距离）的比值随格距单调下降（16 至 96）。
  D1-3b 平均密度主导：比值 mu_funnel / mu_uniform(平均幅度) 落在 1.0 至 1.5（全部格距）。
  D1-3c 一维对照（报告）：一维漏斗比值在 N >= 256 收敛（相邻档变化 < 1%）；三维无此收敛。
  D1-4 分档的三维命运：薄板族（半宽 1 至 12）无一档锁定带边值（偏离 > 1e-6）。
  D1-5 形状依赖（报告）：等调制体积 8 下薄板、柱体、立方体的读数排序与平移敏感性。
  D1-6 尺寸稳定性：薄板读数在格距 32 与 64 间的相对变化 < 1%；小形状读数呈 1/V 标度（报告）。

模型：简单立方格点、周期边界、交错背景 theta_n = pi(i+j+k)、J = lam = 1，
H_nn = 6 + (-1)^(i+j+k) h_n、近邻 -1（度数为 6）。一维对照模型与弱版方向一同源。
确定性：稀疏求解器固定初矢（按格距定种子），无随机结果；无时间演化。
图存 D:/software/TianXingFC/images/m4d1_3d_config.png。
"""

import sys
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MU_EDGE3 = 6.0 - np.sqrt(37.0)
CHECKS = []
T0 = time.time()


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))


def build_3d(L, h):
    N = L ** 3
    idx = np.arange(N)
    i = idx // (L * L); j = (idx // L) % L; k = idx % L
    sgn = (-1.0) ** (i + j + k)
    nxp = i * L * L + j * L + (k + 1) % L
    nyp = i * L * L + ((j + 1) % L) * L + k
    nzp = ((i + 1) % L) * L * L + j * L + k
    rows = np.concatenate([idx, idx, idx, nxp, nyp, nzp])
    cols = np.concatenate([nxp, nyp, nzp, idx, idx, idx])
    A = sp.coo_matrix((-np.ones(6 * N), (rows, cols)), shape=(N, N)).tocsr() + sp.diags(6.0 + sgn * h)
    return A


def low_3d(L, h):
    N = L ** 3
    v0 = np.random.RandomState(100000 + N).standard_normal(N)
    v0 /= np.linalg.norm(v0)
    return float(spla.eigsh(build_3d(L, h), k=1, which="SA", v0=v0, tol=0, return_eigenvectors=False)[0])


def coords(L):
    idx = np.arange(L ** 3)
    i = idx // (L * L); j = (idx // L) % L; k = idx % L
    return i, j, k, idx


def shape_mask(L, kind, size):
    i, j, k, idx = coords(L)
    if kind == "cube":
        return (i < size[0]) & (j < size[1]) & (k < size[2])
    raise ValueError(kind)


def low_2comp(h_plane):
    L = len(h_plane)
    A = np.zeros((2 * L, 2 * L))
    for a in range(L):
        A[a, a] = 2.0
        A[L + a, L + a] = 10.0
        c = ((-1.0) ** a) * h_plane[a]
        A[a, L + a] = c
        A[L + a, a] = c
        A[a, (a + 1) % L] += 1.0
        A[a, (a - 1) % L] += 1.0
        A[L + a, L + (a + 1) % L] += 1.0
        A[L + a, L + (a - 1) % L] += 1.0
    return float(np.linalg.eigvalsh(A)[0])


def low_1d(h):
    N = len(h)
    n = np.arange(N)
    H = np.zeros((N, N))
    H[n, n] = 2.0 + ((-1.0) ** n) * h
    H[n, (n + 1) % N] += 1.0
    H[n, (n - 1) % N] += 1.0
    return float(np.linalg.eigvalsh(H)[0])


print("==== M4-D1：三维构型层读数 ====")
print("三维带边值（均匀 h=1）6 - sqrt(37) = %.12f" % MU_EDGE3)

# D1-1a
L = 32
mu_u = low_3d(L, np.ones(L ** 3))
check("D1-1a", abs(mu_u - MU_EDGE3) <= 1e-10, "L=32 均匀最低模 %.12f（期望 %.12f）" % (mu_u, MU_EDGE3))

# D1-1b
L = 32
devs = []
for k in [1, 3, 5]:
    n = np.arange(L)
    dd = np.minimum(n, L - n)
    hp = np.zeros(L); hp[dd <= k] = 1.0
    i, j, kk, idx = coords(L)
    m3 = low_3d(L, hp[i])
    mr = low_2comp(hp)
    devs.append(abs(m3 - mr))
check("D1-1b", max(devs) <= 1e-10, "薄板 k=1,3,5 三维对两分量约化最大差 %.2e" % max(devs))

# D1-2a
sign_rows = []
ok_sign = True
for L in [12, 16, 24]:
    i, j, kk, idx = coords(L)
    m1 = low_3d(L, ((i < 1) & (j < 1) & (kk < 1)).astype(float))
    m2 = low_3d(L, ((i < 2) & (j < 2) & (kk < 2)).astype(float))
    sign_rows.append((L, m1, m2))
    ok_sign = ok_sign and (m1 > 0) and (m2 < 0)
check("D1-2a", ok_sign, "立方体符号：%s" % "; ".join("L=%d 1^3 %+.2e 2^3 %+.2e" % r for r in sign_rows))

# D1-2b（报告）
L = 16
i, j, kk, idx = coords(L)
print("\n  [D1-2b] 柱体扫描（L=16，角位放置，截面 cs x cs、长度 w）")
for cs in [2, 3]:
    vals = []
    for w in range(1, 7):
        sel = (i < cs) & (j < cs) & (kk < w)
        vals.append(low_3d(L, sel.astype(float)))
    print("       截面 %dx%d：w=1..6 → %s" % (cs, cs, " ".join("%+.6f" % v for v in vals)))

# D1-3a / D1-3b
Ls = [16, 24, 32, 48, 64, 96]
funnel = []
for L in Ls:
    i, j, kk, idx = coords(L)
    di = np.minimum(i, L - i); dj = np.minimum(j, L - j); dk = np.minimum(kk, L - kk)
    r = np.sqrt(di * di + dj * dj + dk * dk)
    h = 1.0 / (1.0 + r)
    hb = float(h.mean())
    mu = low_3d(L, h)
    mu_eq = 6.0 - np.sqrt(36.0 + hb * hb)
    funnel.append((L, mu, abs(mu) / abs(MU_EDGE3), mu / mu_eq))
mono = all(funnel[a + 1][2] < funnel[a][2] for a in range(len(funnel) - 1))
check("D1-3a", mono and funnel[0][1] < 0, "比值随格距单调下降：%s"
      % " ".join("L=%d %.4f%%" % (f[0], 100 * f[2]) for f in funnel))
in_band = all(1.0 <= f[3] <= 1.5 for f in funnel)
check("D1-3b", in_band, "mu/均匀等价：%s" % " ".join("L=%d %.3f" % (f[0], f[3]) for f in funnel))

# D1-3c（报告）
print("\n  [D1-3c] 一维对照（同类漏斗 h=1/(1+d)）")
one_d = []
for N in [64, 128, 256, 512]:
    d = np.minimum(np.arange(N), N - np.arange(N))
    m1 = low_1d(1.0 / (1.0 + d))
    one_d.append((N, m1, abs(m1) / abs(2 - np.sqrt(5))))
for N, m1, rr in one_d:
    print("       N=%3d 最低模 %+.8f 比值 %.4f%%" % (N, m1, 100 * rr))
conv_1d = abs(one_d[-1][2] - one_d[-2][2]) / one_d[-2][2] < 0.01
check("D1-3c", conv_1d, "一维 N=256 对 512 的相对变化 %.3f%%（期望 < 1%%）"
      % (100 * abs(one_d[-1][2] - one_d[-2][2]) / one_d[-2][2]))

# D1-4
L = 128
n = np.arange(L)
dd = np.minimum(n, L - n)
devs4 = []
for k in range(1, 13):
    hp = np.zeros(L); hp[dd <= k] = 1.0
    devs4.append(low_2comp(hp) - MU_EDGE3)
check("D1-4", min(devs4) > 1e-6, "薄板半宽 1..12 最小偏离 %.3e（无一档锁定带边值）" % min(devs4))

# D1-5（报告）
print("\n  [D1-5] 等调制体积 8 的形状比较（L=16）")
i, j, kk, idx = coords(16)
shapes = [("slab 1x1x8", (i == 0) & (j == 0) & (kk < 8)),
          ("rod 1x2x4", (i == 0) & (j < 2) & (kk < 4)),
          ("cube 2x2x2", (i < 2) & (j < 2) & (kk < 2))]
for name, sel in shapes:
    print("       %-12s vol=%2d mu=%+.8f" % (name, int(sel.sum()), low_3d(16, sel.astype(float))))

# D1-6
L = 16
slab16 = None
for L in [16, 32, 64]:
    n = np.arange(L)
    dd = np.minimum(n, L - n)
    hp = np.zeros(L); hp[dd <= 3] = 1.0
    i, j, kk, idx = coords(L)
    m = low_3d(L, hp[i])
    print("  [D1-6] 薄板 k=3：L=%2d mu=%+.8f" % (L, m))
    if L == 32:
        m32 = m
    if L == 64:
        m64 = m
rel = abs(m64 - m32) / abs(m32)
check("D1-6", rel < 0.01, "薄板读数 L=32 对 64 的相对变化 %.3f%%（期望 < 1%%）" % (100 * rel))
print("  [D1-6] 报告：立方体 2^3 的 1/V 标度")
for L in [12, 16, 24, 32]:
    i, j, kk, idx = coords(L)
    m = low_3d(L, ((i < 2) & (j < 2) & (kk < 2)).astype(float))
    print("       L=%2d V=%6d mu=%+.8e  mu*V=%+.4f" % (L, L ** 3, m, m * L ** 3))

# 图
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
fig, axs = plt.subplots(2, 2, figsize=(11.2, 8.0))

ax = axs[0, 0]
ax.semilogy([f[0] for f in funnel], [f[2] for f in funnel], "o-", color="tab:blue", label="三维漏斗（1/(1+r)）")
ax.semilogy([o[0] for o in one_d], [o[2] for o in one_d], "s--", color="tab:red", label="一维漏斗（1/(1+d)）")
ax.axhline(one_d[-1][2], color="gray", lw=0.8, ls=":")
ax.set_xlabel("格距 L（一维为 N）")
ax.set_ylabel("比值 |mu_min| / |mu_均匀|")
ax.set_title("漏斗构型的失稳比值：三维衰减、一维收敛", fontsize=9)
ax.legend(fontsize=8)

ax = axs[0, 1]
ax.plot([f[0] for f in funnel], [f[3] for f in funnel], "o-", color="tab:green")
ax.axhline(1.0, color="gray", lw=0.8, ls="--", label="均匀等价线（比值 1）")
ax.axhline(1.5, color="gray", lw=0.8, ls=":")
ax.set_xlabel("格距 L")
ax.set_ylabel("mu / 平均幅度均匀等价")
ax.set_title("三维漏斗逼近平均密度效应", fontsize=9)
ax.legend(fontsize=8)

ax = axs[1, 0]
ks = list(range(1, 13))
dev3 = []
L = 128
n = np.arange(L)
dd = np.minimum(n, L - n)
for k in ks:
    hp = np.zeros(L); hp[dd <= k] = 1.0
    dev3.append(low_2comp(hp) - MU_EDGE3)
dev3 = np.array(dev3)
ax.plot(ks, dev3, "o-", color="tab:blue", label="三维薄板（两分量约化）")
k1 = [1, 3, 5, 7, 9, 11]
ax.plot(k1, [0.0] * len(k1), "rx", ms=8, label="一维奇半宽（锁定 0）")
ax.set_xlabel("半宽 k")
ax.set_ylabel("最低模与带边值之差")
ax.set_title("分档的三维命运：无锁定档", fontsize=9)
ax.legend(fontsize=8)

ax = axs[1, 1]
names = [s[0] for s in shapes]
vals = [low_3d(16, s[1].astype(float)) for s in shapes]
ax.bar(names, vals, color=["tab:red", "tab:orange", "tab:blue"])
ax.axhline(0.0, color="gray", lw=0.8)
ax.set_ylabel("最低模（L=16，等体积 8）")
ax.set_title("形状依赖：越各向异性越深", fontsize=9)

fig.tight_layout()
fig.savefig(r"D:/software/TianXingFC/images/m4d1_3d_config.png", dpi=150)
print("\n  图已保存：images/m4d1_3d_config.png")

n_pass = sum(1 for _, ok in CHECKS if ok)
print("\n==== 判据汇总：%d / %d 通过（运行 %.0f 秒）====" % (n_pass, len(CHECKS), time.time() - T0))
sys.exit(0 if n_pass == len(CHECKS) else 1)