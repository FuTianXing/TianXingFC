# -*- coding: utf-8 -*-
"""判据第 632 号：粗格涌现构造——常值平方生成元的提取与号差读数。

构造：单分量交错算子在动量空间按奇偶对折（k_1 与 k_1+pi 配对），二分量化后
      M(k) = i sin k_1 * sigma_3 + i sin k_2 * sigma_1。

判据与先立的期望：
  T1 对折形式与格点算符逐位一致（期望机精度）。
  T2 M(k)^2 = -(sin^2 k_1 + sin^2 k_2) * I，即味道空间上常数乘单位阵（期望机精度）。
  T3 生成元平方为常值：gamma_1 = sigma_3、gamma_2 = sigma_1，平方为 +1、反对易（精确）。
  T4 号差读数：两平方同号则为欧氏型；洛伦兹型（一负一正）不成立（读数）。
  T5 反例：朴素算子在折叠基下平方与 c*I 的偏差 4|sin k_1 sin k_2|（期望非零）。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = r"D:\software\TianXingFC\images\coarse_grain_generators.png"
N = 16
H = N // 2


def shift1d(n, seam=1.0):
    S = np.zeros((n, n))
    for i in range(n):
        S[i, (i + 1) % n] = seam
    return S


I2 = np.eye(N)
Sx, Sy = shift1d(N), shift1d(N)
S1 = np.kron(Sx, I2)
S2 = np.kron(I2, Sy)
D1 = (S1 - S1.T) / 2.0
D2 = (S2 - S2.T) / 2.0
n1arr = np.repeat(np.arange(N), N)
P1 = np.diag([(-1.0) ** a for a in n1arr])

sx3 = np.array([[1.0, 0.0], [0.0, -1.0]])
sx1 = np.array([[0.0, 1.0], [1.0, 0.0]])

D_stag = D1 + P1 @ D2
D_naive = D1 + D2

# 对比基准：直接作用 + 二维 FFT
rng = np.random.default_rng(20261015)
psi = rng.standard_normal(N * N)
F = np.fft.fft2(psi.reshape(N, N))

j1 = np.arange(H)                      # k_1 取 [0, pi)：j_1 = 0..H-1
s1 = np.sin(2 * np.pi * j1 / N)
j2 = np.arange(N)
s2 = np.sin(2 * np.pi * j2 / N)

print("=== 判据第 632 号（N=%d）===" % N)
for name, D in [("交错", D_stag), ("朴素", D_naive)]:
    Fd = np.fft.fft2((D @ psi).reshape(N, N))
    dev = 0.0
    for a in range(H):
        for b in range(N):
            u = F[a, b]
            v = F[(a + H) % N, b]          # k_1 - pi 对应 j_1 - H
            if name == "交错":
                M = 1j * s1[a] * sx3 + 1j * s2[b] * sx1
            else:
                M = np.diag([1j * (s1[a] + s2[b]), 1j * (-s1[a] + s2[b])])
            pred = M @ np.array([u, v])[..., None]
            tgt = np.array([Fd[a, b], Fd[(a + H) % N, b]])[..., None]
            dev = max(dev, np.abs(pred - tgt).max())
    print("T1  %s：对折形式与格点算符最大偏差 = %.3e" % (name, dev))

# T2：交错算子的 M(k)^2 是否为常数乘单位阵
worst = 0.0
for a in range(H):
    for b in range(N):
        M = 1j * s1[a] * sx3 + 1j * s2[b] * sx1
        worst = max(worst, np.abs(M @ M + (s1[a] ** 2 + s2[b] ** 2) * np.eye(2)).max())
print("T2  交错：|M^2 + (s1^2+s2^2) I| 最大值 = %.3e（期望 0）" % worst)

# T3：生成元平方与反对易
for lbl, g in [("gamma_1 = sigma_3", sx3), ("gamma_2 = sigma_1", sx1)]:
    print("T3  %s：平方 - I 最大偏差 = %.3e" % (lbl, np.abs(g @ g - np.eye(2)).max()))
print("T3  反对易子 {gamma_1,gamma_2} 最大模 = %.3e（期望 0）"
      % np.abs(sx3 @ sx1 + sx1 @ sx3).max())

# T4：号差（读平方的本征值）
sq = [np.linalg.eigvalsh(g @ g).tolist() for g in (sx3, sx1)]
print("T4  两生成元平方的本征值 = %s 与 %s → 同号 (+,+)：欧氏型；洛伦兹型不成立"
      % (sq[0], sq[1]))

# T5：反例（朴素算子）
bad = 0.0
for a in range(H):
    for b in range(N):
        Mn = np.diag([1j * (s1[a] + s2[b]), 1j * (-s1[a] + s2[b])])
        Mn2 = Mn @ Mn
        c = -0.5 * (np.trace(Mn2).real)
        bad = max(bad, np.abs(Mn2 - c * np.eye(2)).max())
ana = 2 * (np.max(np.abs(s1)) ** 2 + np.max(np.abs(s1)) * np.max(np.abs(s2)) + np.max(np.abs(s2)) ** 2)
print("T5  朴素：与常数乘单位阵的最大偏差 = %.3e（期望非零）" % bad)
print("    解析对照：max 2(s1^2+s1 s2+s2^2) = %.6f" % ana)

# ---------- 图 ----------
fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.2))

grid = np.zeros((H, N))
for a in range(H):
    for b in range(N):
        M = 1j * s1[a] * sx3 + 1j * s2[b] * sx1
        grid[a, b] = np.abs(M @ M + (s1[a] ** 2 + s2[b] ** 2) * np.eye(2)).max()
im0 = ax[0].imshow(grid, cmap="viridis", aspect="auto", vmin=0.0, vmax=1e-14)
ax[0].set_title("(a) 交错：$|M^2+(s_1^2+s_2^2)I|$\n（固定标尺 $0$ 至 $10^{-14}$，全图近零）")
ax[0].set_xlabel("$j_2$（$k_2$ 扇区）"); ax[0].set_ylabel("$j_1$（$k_1$ 折半）")
fig.colorbar(im0, ax=ax[0], fraction=0.046)

ax[1].bar([0, 1], [np.linalg.eigvalsh(sx3 @ sx3)[0], np.linalg.eigvalsh(sx1 @ sx1)[0]],
          color=["tab:blue", "tab:green"], width=0.5)
ax[1].axhline(0, color="k", lw=0.8)
ax[1].set_xticks([0, 1]); ax[1].set_xticklabels([r"$\gamma_1^2$", r"$\gamma_2^2$"])
ax[1].set_ylim(-1.5, 1.5)
ax[1].set_ylabel("平方的本征值")
ax[1].set_title("(b) 号差读数：两者平方同为 $+1$\n即欧氏型，洛伦兹型不成立")

dev_map = np.zeros((H, N))
for a in range(H):
    for b in range(N):
        Mn = np.diag([1j * (s1[a] + s2[b]), 1j * (-s1[a] + s2[b])])
        Mn2 = Mn @ Mn
        c = -0.5 * (np.trace(Mn2).real)
        dev_map[a, b] = np.abs(Mn2 - c * np.eye(2)).max()
im = ax[2].imshow(dev_map, cmap="magma", aspect="auto")
ax[2].set_title("(c) 反例：朴素算子的偏差\n（零只在坐标轴上，非整片）")
ax[2].set_xlabel("$j_2$"); ax[2].set_ylabel("$j_1$")
fig.colorbar(im, ax=ax[2])

fig.tight_layout()
fig.savefig(OUT, dpi=170)
print("\n图已存：%s" % OUT)
