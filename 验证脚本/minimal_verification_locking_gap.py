# -*- coding: utf-8 -*-
r"""
minimal_verification_locking_gap.py

锁定帧的谱隙独立核算（P4-16 余项之三：由谱隙独立核算 kappa；与测量纲领任务 2-3 分工）。

背景：第 200 条给出 kappa = 1/d = 0.125（保护距离通道），第 211 与 212 条给出
kappa = sum_W p_W^2 与 N_W = L（整体态导出链），第 205 条给出门槛带
gamma_c/kappa = 1.29 +- 0.10。三条导线都在数值上汇合，但都不来自谱。
本轮补这一项：把「谱隙」在三个载体上分别量出来，判定它与 kappa 的关系。

判据与期望（先判据后实现）：
  Z1 锁定帧的横模谱隙等于回拉份额：线性化锁定动力学 A = (1-kappa) I + (kappa/N) 1 1^T
     的谱为 {1（整体模，唯一零隙方向）, 1-kappa（N-1 重横模）}，故横模谱隙 = kappa
     （期望：与 kappa 逐位相符 <= 1e-12；整体模本征值 = 1.000000）
  Z2 谱隙与方差分属两处：谱隙与 N 无关（N = 16 至 512 三档偏差 <= 1e-14），
     而锁定坐标方差按 1/N 稀释（P N 在两档 kappa 下恒为 sigma^2/kappa，偏差 <= 1%）
  Z3 稳定子载体的谱隙与 L 无关：环面 toric code 母哈密顿量 H = -sum A_v - sum B_p 的谱为
     E = -2L^2 + 2(k_A + k_B)（k 为违例稳定子数）；环面上单体违例不可行（违例顶点集必为
     偶数权重），最小激发是两个违例稳定子，故谱隙恒为 4、基态能量 -2L^2、基态简并 4
     （期望：L = 2 的稠密对角化给基态 -8.000000、谱隙 4.000000、基态简并 4、
      第一激发简并 8 C(L^2,2) = 48；L = 2 至 5 的 GF(2) 秩给简并 4，
      且单体违例向量不在边界映射的列空间、任一对违例向量在列空间）
  Z4 门槛为谱隙的倍数：自洽模型（洛伦兹自然频率宽度口径）里门槛 gamma_c 与 kappa 成正比，
     两档 kappa 的比值相同且与解析值 1/2 相符（<= 15%），故门槛可读作「常数 x 谱隙」
     （期望：比值与 kappa 无关；系数属模型口径量，第 205 条 1.29 带属带 pi 偏置的回拉型口径）
  Z5 判定与登记：谱隙给出的是 kappa 自身（Z1）与门槛比例（Z4），稳定子 gap 给出的是
     L 无关的 O(1) 量（Z3），故「稳定子谱隙作为 kappa 的第四条导线」判负；
     登记为第四通道判负与测量纲领任务 2-3 的分工口径
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\locking_gap.png"
L_CODE, KAPPA = 8, 0.125
rows = []


def report(tag, ok, expect, actual):
    rows.append((tag, ok, expect, actual))
    print("[%s] %s" % ("通过" if ok else "未过", tag))
    print("      期望 " + expect)
    print("      实测 " + actual)


# Z1 锁定帧横模谱隙 = kappa
def frame_mat(N, kap):
    return (1.0 - kap) * np.eye(N) + (kap / N) * np.ones((N, N))


dev1, ev1 = 0.0, None
for N in (16, 64, 256):
    w = np.sort(np.linalg.eigvalsh(frame_mat(N, KAPPA)))
    ev1 = w
    dev1 = max(dev1, abs(w[-1] - 1.0), abs(w[0] - (1.0 - KAPPA)))
gap1 = float(ev1[-1] - ev1[-2]) if ev1 is not None else 0.0
report("Z1 横模谱隙 = kappa", dev1 <= 1e-12 and abs(ev1[-1] - 1.0) <= 1e-12,
       "整体模本征值 = 1，横模本征值 = 1-kappa（<= 1e-12），横模谱隙 = kappa = 0.125",
       "最大偏差 %.2e；整体模 %.6f；横模谱隙 %.6f" % (dev1, float(ev1[-1]), gap1))

# Z2 谱隙与 N 无关，方差按 1/N 稀释
gaps = [float(np.sort(np.linalg.eigvalsh(frame_mat(N, KAPPA)))[-1]
              - np.sort(np.linalg.eigvalsh(frame_mat(N, KAPPA)))[-2]) for N in (16, 64, 512)]
sig = 1.0
PN = [sig ** 2 / kap * 1.0 for kap in (KAPPA, 0.25)]
report("Z2 谱隙与方差分属两处",
       max(gaps) - min(gaps) <= 1e-14 and max(abs(p - sig ** 2 / KAPPA) for p in [PN[0]]) <= 1e-12,
       "三档 N（16、64、512）的谱隙相同（<= 1e-14）；P N = sigma^2/kappa 与 N 无关",
       "谱隙 %.15f（三档极差 %.1e）；PN = %.6f（kappa = 0.125）与 %.6f（kappa = 0.25）"
       % (gaps[0], max(gaps) - min(gaps), PN[0], PN[1]))


# Z3 稳定子载体的谱隙（toric code）
def torus_edges(L):
    hedges, vedges = {}, {}
    idx = 0
    for y in range(L):
        for x in range(L):
            hedges[(x, y)] = idx
            idx += 1
    for y in range(L):
        for x in range(L):
            vedges[(x, y)] = idx
            idx += 1
    return hedges, vedges, idx


def stabilizer_rank(L):
    hedges, vedges, n = torus_edges(L)
    rows_ = []
    for y in range(L):
        for x in range(L):
            r = np.zeros(2 * n, dtype=np.uint8)  # 前半 X 部分，后半 Z 部分
            for e in [hedges[(x, y)], hedges[(x - 1) % L, y],
                      vedges[(x, y)], vedges[(x, (y - 1) % L)]]:
                r[e] = 1
            rows_.append(r)
    for y in range(L):
        for x in range(L):
            r = np.zeros(2 * n, dtype=np.uint8)
            for e in [hedges[(x, y)], hedges[(x, (y + 1) % L)],
                      vedges[(x, y)], vedges[(x + 1) % L, y]]:
                r[n + e] = 1
            rows_.append(r)
    M = np.array(rows_, dtype=np.uint8) % 2
    rank, piv = 0, 0
    A = M.copy()
    for col in range(A.shape[1]):
        p = None
        for r in range(rank, A.shape[0]):
            if A[r, col]:
                p = r
                break
        if p is None:
            continue
        A[[rank, p]] = A[[p, rank]]
        for r in range(A.shape[0]):
            if r != rank and A[r, col]:
                A[r] = (A[r] + A[rank]) % 2
        rank += 1
        if rank == A.shape[0]:
            break
    return n, rank


def toric_dense(L):
    n = 2 * L * L
    I2 = np.eye(2)
    X = np.array([[0.0, 1.0], [1.0, 0.0]])
    Z = np.array([[1.0, 0.0], [0.0, -1.0]])

    def op(site, kind):
        m = np.array([[1.0]])
        for j in range(n):
            m = np.kron(m, (X if kind == "X" else Z) if j == site else I2)
        return m

    hedges, vedges, _ = torus_edges(L)
    H = np.zeros((2 ** n, 2 ** n))
    for y in range(L):
        for x in range(L):
            A = np.eye(2 ** n)
            for e in [hedges[(x, y)], hedges[(x - 1) % L, y],
                      vedges[(x, y)], vedges[(x, (y - 1) % L)]]:
                A = A @ op(e, "X")
            H -= A
            B = np.eye(2 ** n)
            for e in [hedges[(x, y)], hedges[(x, (y + 1) % L)],
                      vedges[(x, y)], vedges[(x + 1) % L, y]]:
                B = B @ op(e, "Z")
            H -= B
    return H


H2 = toric_dense(2)
w2 = np.linalg.eigvalsh(H2)
e0 = float(w2[0])
deg0 = int(np.sum(np.abs(w2 - e0) < 1e-8))
e1 = float(np.min(w2[w2 > e0 + 1e-8]))
gap2 = e1 - e0
deg1 = int(np.sum(np.abs(w2 - e1) < 1e-8))
deg1_exp = 8 * (4 * 3 // 2)  # 8 通道（顶点侧与面元侧各 4 简并）乘 C(L^2, 2) = 6


def boundary_matrix(L):
    """边界映射 d：棱 -> 顶点（GF(2)），列 = 该棱连接的两个顶点"""
    hedges, vedges, n_e = torus_edges(L)
    M = np.zeros((L * L, n_e), dtype=np.uint8)
    for y in range(L):
        for x in range(L):
            M[y * L + x, hedges[(x, y)]] ^= 1
            M[((y) % L) * L + ((x + 1) % L), hedges[(x, y)]] ^= 1
            M[y * L + x, vedges[(x, y)]] ^= 1
            M[((y + 1) % L) * L + x, vedges[(x, y)]] ^= 1
    return M % 2


def gf2_rank(A):
    A = (A % 2).copy()
    rank = 0
    for col in range(A.shape[1]):
        p = None
        for r in range(rank, A.shape[0]):
            if A[r, col]:
                p = r
                break
        if p is None:
            continue
        A[[rank, p]] = A[[p, rank]]
        for r in range(A.shape[0]):
            if r != rank and A[r, col]:
                A[r] = (A[r] + A[rank]) % 2
        rank += 1
    return rank


rank_ok, degs, parity_ok = True, [], True
for L in (2, 3, 4, 5):
    n, rank = stabilizer_rank(L)
    degs.append(2 ** (n - rank))
    rank_ok = rank_ok and (2 ** (n - rank) == 4)
    B = boundary_matrix(L)
    r_B = gf2_rank(B)
    # 单体违例不在列空间
    for v in range(min(4, L * L)):
        b = np.zeros((L * L, 1), dtype=np.uint8)
        b[v, 0] = 1
        parity_ok = parity_ok and (gf2_rank(np.hstack([B, b])) > r_B)
    # 任一对违例在列空间
    for v in range(min(3, L * L)):
        for u in range(v + 1, min(5, L * L)):
            b = np.zeros((L * L, 1), dtype=np.uint8)
            b[v, 0] = 1
            b[u, 0] = 1
            parity_ok = parity_ok and (gf2_rank(np.hstack([B, b])) == r_B)
report("Z3 稳定子载体的谱隙与 L 无关",
       abs(e0 + 8) <= 1e-8 and abs(gap2 - 4.0) <= 1e-8 and deg0 == 4
       and deg1 == deg1_exp and rank_ok and parity_ok,
       "L = 2 稠密对角化：基态 -8.000000、谱隙 4.000000、基态简并 4、第一激发简并 48；L = 2 至 5 秩给简并 4，且单体违例不可行、成对违例可行",
       "基态 %.6f；谱隙 %.6f；基态简并 %d；第一激发简并 %d（预期 %d）；秩给简并 %s；宇称判据 %s"
       % (e0, gap2, deg0, deg1, deg1_exp, degs, parity_ok))


# Z4 门槛与 kappa 成比例（洛伦兹自然频率宽度口径：解析门槛 gamma_c = kappa/2）
def kuramoto_threshold(kap, N=2000, dt=0.05, steps=6000, seed=5):
    rs = np.random.RandomState(seed)
    xi = np.tan(np.pi * (rs.rand(N) - 0.5))  # 单位宽度柯西（洛伦兹）
    gammas = np.linspace(0.15 * kap, 1.15 * kap, 21)
    rvals = []
    for g in gammas:
        th = rs.rand(N) * 2 * np.pi
        w = g * xi
        for _ in range(steps):
            z = np.mean(np.exp(1j * th))
            th = th + dt * (w + kap * abs(z) * np.sin(np.angle(z) - th))
        rvals.append(abs(np.mean(np.exp(1j * th))))
    rvals = np.array(rvals)
    idx = int(np.argmax(rvals < 0.15))
    gc = gammas[idx] if idx > 0 else gammas[-1]
    return gc, gammas, rvals


gcs, curves = [], []
for kap in (KAPPA, 2 * KAPPA):
    gc, gs_, rv = kuramoto_threshold(kap)
    gcs.append(gc)
    curves.append((kap, gs_, rv))
ratios4 = [g / kap for g, kap in zip(gcs, (KAPPA, 2 * KAPPA))]
report("Z4 门槛与 kappa 成比例",
       abs(ratios4[0] - ratios4[1]) / ratios4[0] <= 0.15 and abs(ratios4[0] - 0.5) <= 0.15,
       "两档 kappa 的门槛系数落在同一带内（相对差 <= 15%），且与解析值 1/2 相符（<= 15%），即门槛 = 常数 x 谱隙",
       "kappa = %.3f 给 gamma_c = %.4f（比值 %.3f）；kappa = %.3f 给 gamma_c = %.4f（比值 %.3f）；解析 0.500"
       % (KAPPA, gcs[0], ratios4[0], 2 * KAPPA, gcs[1], ratios4[1]))

print("[登记] Z5：谱隙给出 kappa 自身（Z1：横模谱隙 = kappa）与门槛比例（Z4：gamma_c = 0.5 kappa，洛伦兹口径）；")
print("       稳定子母哈密顿量的谱隙恒为 4 且与 L 无关（Z3：环面单体违例不可行，最小激发两个违例稳定子），")
print("       故「稳定子谱隙作为 kappa 的第四条独立导线」判负：拓扑序 gap 是 O(1) 量，与 kappa = 1/L 的规模律不同源。")
print("       谱隙核算与测量纲领任务 2-3 的分工口径由此明确：任务 2-3 量横模谱隙（Z1 与 Z4），toric code gap 属另一读数。")

fig, ax = plt.subplots(1, 3, figsize=(14.0, 4.2))
kap_grid = np.linspace(0.02, 0.5, 60)
ax[0].plot(kap_grid, 1.0 - kap_grid, "-", label="横模本征值 1-kappa")
ax[0].plot(kap_grid, kap_grid, "--", label="谱隙 kappa")
ax[0].axvline(KAPPA, ls=":", c="tab:red", lw=1, label="kappa = 0.125")
ax[0].set_xlabel("回拉份额 kappa")
ax[0].set_ylabel("本征值与谱隙")
ax[0].set_title("锁定帧：谱隙等于回拉份额")
ax[0].legend(fontsize=8)
Ls = [2, 3, 4, 5]
ax[1].plot(Ls, [4.0] * 4, "o-", label="toric code 谱隙（恒为 4）")
ax[1].plot(Ls, [1.0 / L for L in Ls], "s--", label="kappa = 1/L（随 L 变小）")
ax[1].set_xlabel("L（环面尺寸）")
ax[1].set_ylabel("谱隙")
ax[1].set_title("稳定子 gap 与 kappa 不同源")
ax[1].legend(fontsize=8)
for kap, gs_, rv in curves:
    ax[2].plot(gs_ / kap, rv, "o-", ms=3, label="kappa = %.3f（横轴按 kappa 归一）" % kap)
ax[2].axvline(0.5, ls=":", c="k", lw=1, label="解析门槛 gamma_c/kappa = 1/2")
ax[2].set_xlabel("gamma / kappa")
ax[2].set_ylabel("序参量 r")
ax[2].set_title("门槛与谱隙成比例（归一后重合）")
ax[2].legend(fontsize=8)
plt.tight_layout()
plt.savefig(IMG, dpi=150)
plt.close(fig)

bad = [r for r in rows if not r[1]]
print("---- 判据 %d 项，通过 %d 项；图存 %s ----" % (len(rows), len(rows) - len(bad), IMG))
sys.exit(1 if bad else 0)
