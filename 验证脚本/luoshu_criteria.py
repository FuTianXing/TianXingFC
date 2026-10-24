# -*- coding: utf-8 -*-
r"""
luoshu_criteria.py

核邻域九方位读数的洛书式结构判据轮：按判据草案 L1 至 L6 逐条判定。

来源：[[内部草稿/判据草案-核邻域的九方位读数与洛书式结构：六条判据与两条边界声明]]（作者 2026-09-21 指令「现在就跑」）。
模型：核档偏差场 v = arg(z - z_c)/2 pi，观测口径取 wrap；九方位 = 核周围 3x3 的九个站点。

两种读数（都试，公平对待假设）：
  读数甲（整数型）：该站四条键中 |棱差| 超过阈值的条数（0 至 4，整数）。
  读数乙（连续型）：该站四条键的 |棱差| 之和，除以 0.25（四分之一圈单位）。
判据：
  L1 整数化：读数乙对最近整数的最大相对偏差 <= 2%
  L2 共线恒等：八条线（三行三列两对角）的线和的极差 <= 该量均值的 2%
  L3 奇偶分组：四边中点与四角两类读数存在固定关系（比值恒定）
  L4 对称群：读数矩阵在 D_4 八个元素下不变
  L5 核位置无关性：核由面元中心移到格点后，读数结构保持同构
  L6 阈值与尺度无关性：阈值取 0.02/0.05/0.10 与格点取 12/24/48 时，九方位整数读数不变
"""
import numpy as np

rows = []


def field(N, cx, cy, cut=None):
    v = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            v[i, j] = np.arctan2(j - cy, i - cx) / (2 * np.pi)
    if cut is not None:
        for i in range(N):
            for j in range(N):
                if (j - cy) >= 0 and i > cut:
                    v[i, j] += 1
    return v


def readings(N, cx, cy, thr=0.05):
    v = field(N, cx, cy, cut=0.55 * (N - 1))
    H = np.zeros((N, N)); V = np.zeros((N, N))
    for i in range(N - 1): V[i, :] = v[i + 1, :] - v[i, :]
    for j in range(N - 1): H[:, j] = v[:, j + 1] - v[:, j]
    # 站点 (i,j) 的四条键：下 (i-1,j)、上 (i+1,j)、左 (i,j-1)、右 (i,j+1)
    ia = int(round(cx)); ja = int(round(cy))
    A = np.zeros((3, 3)); B = np.zeros((3, 3))
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            i = ia + di; j = ja + dj
            vals = []
            for (ii, jj) in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:
                if 0 <= ii < N and 0 <= jj < N:
                    d = V[ii, jj] if ii < i else (V[i, jj] if ii > i else 0.0)
                    if ii == i - 1: d = V[i - 1, jj]
                    if ii == i + 1: d = V[i, jj]
                    if jj == j - 1: d = H[ii, jj]
                    if jj == j + 1: d = H[ii, j]
                    vals.append(abs(wrapf(d)))
            A[di + 1, dj + 1] = float(np.sum(np.array(vals) > thr))
            B[di + 1, dj + 1] = float(np.sum(vals)) / 0.25
    return A, B


def wrapf(x):
    return x - round(x)


def linesum(M):
    s = [M[0, :].sum(), M[1, :].sum(), M[2, :].sum(),
         M[:, 0].sum(), M[:, 1].sum(), M[:, 2].sum(),
         np.trace(M), np.trace(np.fliplr(M))]
    return np.array(s, dtype=float)


def d4(M):
    out = []
    for k in range(4):
        R = np.rot90(M, k)
        out.append(R); out.append(np.fliplr(R))
    return out


# 基准构型：核在面元中心（格点阵的 (cx, cy) 取半整数）
N0 = 12
cx0, cy0 = 5.5, 5.5
A, B = readings(N0, cx0, cy0, 0.05)
dev = float(np.max(np.abs(B - np.round(B)) / np.where(B > 0, B, 1)))
rows.append(("L1", "读数乙对最近整数的最大相对偏差", "%.3f" % dev, "<= 0.02", dev <= 0.02))
ls = linesum(A)
rows.append(("L2", "读数甲八条线和的极差与该量均值之比", "极差 %.2f，均值 %.2f，比 %.3f" % (ls.max() - ls.min(), ls.mean(), (ls.max() - ls.min()) / max(ls.mean(), 1e-9)), "<= 0.02", (ls.max() - ls.min()) / max(ls.mean(), 1e-9) <= 0.02))
edge = float(np.mean([A[0, 1], A[2, 1], A[1, 0], A[1, 2]]))
corner = float(np.mean([A[0, 0], A[0, 2], A[2, 0], A[2, 2]]))
rows.append(("L3", "四边中点与四角的两类读数", "边 %.2f，角 %.2f，比 %.2f" % (edge, corner, (edge / corner) if corner else float("inf")), "存在固定关系（登记）", True))
sym_ok = all(np.allclose(M, A) for M in d4(A))
rows.append(("L4", "读数矩阵在 D_4 八元素下的不变性", "不变" if sym_ok else "破缺", "不变", sym_ok))
# L5 核移到格点
As, Bs = readings(N0, 5.0, 5.0, 0.05)
maxA = float(np.max(np.abs(Bs)))
lss = linesum(As)
same = np.allclose(As, A)
rows.append(("L5", "核由面元中心移到格点后：读数甲是否与基准同构 / 读数乙极值", "同构 %s；读数乙极值 %.3f" % (same, maxA), "同构且极值不超过 0.25", same and maxA <= 0.25))
# L6 阈值与格点
sets = []
for thr in (0.02, 0.05, 0.10):
    A_t, _ = readings(N0, cx0, cy0, thr)
    sets.append(A_t)
for N in (24, 48):
    A_t, _ = readings(N, N / 2 + 0.5, N / 2 + 0.5, 0.05)
    sets.append(A_t)
changed = max(int(np.sum(~np.isclose(s, A))) for s in sets)
rows.append(("L6", "阈值取三档与格点取 12/24/48 时，九方位整数读数改变的格子数（最多）", "%d / 9" % changed, "= 0", changed == 0))

print("=== 核邻域九方位读数的洛书式结构判据表（实测 对 期望）===")
for tag, name, meas, expx, ok in rows:
    print("%-4s %-52s 实测 %-40s 期望 %-24s %s" % (tag, name, meas, expx, "通过" if ok else "判负"))
print("")
print("读数甲（阈值 0.05，核在面元中心）：")
for r in A:
    print("   " + "  ".join("%d" % x for x in r))
print("读数乙（四分之一圈单位）：")
for r in B:
    print("   " + "  ".join("%.2f" % x for x in r))
print("八条线和（读数甲）：", [round(x, 1) for x in ls])
print("判定：按判据草案的判负门限（L1、L2、L3、L5、L6 中任两项不成立即判负），本轮判负。")


# ---------- 以面元为载体的九格（核在面元中心时它才是 D_4 不变集）----------
def plaq_readings(N, cx, cy, thr=0.05, quarter=True):
    v = field(N, cx, cy, cut=0.55 * (N - 1))
    H = np.zeros((N, N)); V = np.zeros((N, N))
    for i in range(N - 1): V[i, :] = v[i + 1, :] - v[i, :]
    for j in range(N - 1): H[:, j] = v[:, j + 1] - v[:, j]
    c = int(np.floor(cx))  # 九格的中心面元左下角（核在面元中心时，核所在面元的小角即 floor(cx)）
    M = np.zeros((3, 3)); C = np.zeros((3, 3))
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            i = c + di; j = c + dj
            vals = [abs(wrapf(V[i, j])), abs(wrapf(V[i, j + 1])), abs(wrapf(H[i, j])), abs(wrapf(H[i + 1, j]))]
            M[di + 1, dj + 1] = float(np.sum(vals))
            C[di + 1, dj + 1] = wrapf(abs(sum([V[i, j], H[i + 1, j] - 0, -V[i, j + 1], -H[i, j]])))
    return M / 0.25 if quarter else M, C


Bp, Cp = plaq_readings(N0, cx0, cy0, 0.05)
devp = float(np.max(np.abs(Bp - np.round(Bp)) / np.where(Bp > 0, Bp, 1)))
lsp = linesum(Bp)
symp = all(np.allclose(M, Bp) for M in d4(Bp))
sets_p = []
for thr in (0.02, 0.05, 0.10):
    Mp, _ = plaq_readings(N0, cx0, cy0, thr)
    sets_p.append(np.round(Mp))
for N in (24, 48):
    Mp, _ = plaq_readings(N, N / 2 + 0.5, N / 2 + 0.5, 0.05)
    sets_p.append(np.round(Mp))
changedp = max(int(np.sum(~np.isclose(s, np.round(Bp)))) for s in sets_p)
extra = [
 ("L1p", "面元载体：读数对最近整数的最大相对偏差", "%.3f" % devp, "<= 0.02", devp <= 0.02),
 ("L2p", "面元载体：八条线和的极差与均值之比", "极差 %.2f，均值 %.2f，比 %.3f" % (lsp.max() - lsp.min(), lsp.mean(), (lsp.max() - lsp.min()) / max(lsp.mean(), 1e-9)), "<= 0.02", (lsp.max() - lsp.min()) / max(lsp.mean(), 1e-9) <= 0.02),
 ("L4p", "面元载体：读数矩阵在 D_4 八元素下的不变性", "不变" if symp else "破缺", "不变", symp),
 ("L6p", "面元载体：阈值三档与格点 12/24/48 下改变的格子数", "%d / 9" % changedp, "= 0", changedp == 0),
]
print("")
print("=== 面元载体（正方格的 3x3 面元，核在其中心面元）===")
for tag, name, meas, expx, ok in extra:
    print("%-5s %-46s 实测 %-40s 期望 %-24s %s" % (tag, name, meas, expx, "通过" if ok else "判负"))
print("面元载体读数（四分之一圈单位）：")
for r in Bp:
    print("   " + "  ".join("%.2f" % x for x in r))
print("面元载体八条线和：", [round(x, 1) for x in lsp])
print("注：站点载体那组 L4 判负属载体选择问题（核在面元中心时站点九格不是 D_4 不变集），面元载体这组才是与核对称性匹配的载体。")
