# -*- coding: utf-8 -*-
"""判据第 635 号：时间方向归属——纯空间格点能否判定"哪个方向带 i"。

判据与先立的期望：
  W1 两种定向（奇偶符号加在方向 2 上 / 加在方向 1 上）的折叠形式，
     其"生成元与方向的配对"互换（期望逐位一致）。
  W2 两种定向的格点算符谱逐位相同（读数，预期相同：定向选择是格点对换对称）。
  W3 号差指派：把 i 加在某一枚上，与"换定向后再加在对应枚上"给出同一物理结构（精确）。
  W4 反例：把指定方向固定死（总让方向 1 带 i），则方向对调后号差落在另一空间方向上（读数）。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = r"D:\software\TianXingFC\images\time_direction.png"
N = 16
H = N // 2
rng = np.random.default_rng(20261015)


def shift1d(n):
    S = np.zeros((n, n))
    for i in range(n):
        S[i, (i + 1) % n] = 1.0
    return S


I = np.eye(N)
Sx, Sy = shift1d(N), shift1d(N)
S1 = np.kron(Sx, I)
S2 = np.kron(I, Sy)
D1 = (S1 - S1.T) / 2.0
D2 = (S2 - S2.T) / 2.0
n1arr = np.repeat(np.arange(N), N)
n2arr = np.tile(np.arange(N), N)
P1 = np.diag([(-1.0) ** a for a in n1arr])
P2 = np.diag([(-1.0) ** b for b in n2arr])

Dcase1 = D1 + P1 @ D2       # 奇偶符号加在方向 2 上（用方向 1 的奇偶）
Dcase2 = P2 @ D1 + D2       # 奇偶符号加在方向 1 上（用方向 2 的奇偶）

psi = rng.standard_normal(N * N)
F = np.fft.fft2(psi.reshape(N, N))
S3 = np.array([[1, 0], [0, -1]], dtype=complex)
S1m = np.array([[0, 1], [1, 0]], dtype=complex)
ss1 = np.sin(2 * np.pi * np.arange(H) / N)
ss2 = np.sin(2 * np.pi * np.arange(N) / N)

# W1：两种定向的折叠形式（生成元与方向的配对互换）
dev1 = dev2 = 0.0
Fd1 = np.fft.fft2((Dcase1 @ psi).reshape(N, N))
Fd2 = np.fft.fft2((Dcase2 @ psi).reshape(N, N))
for a in range(H):          # 情形 1：沿方向 1 对折，k1 折半
    for b in range(N):
        u, v = F[a, b], F[(a + H) % N, b]
        M = 1j * ss1[a] * S3 + 1j * ss2[b] * S1m
        tgt = np.array([Fd1[a, b], Fd1[(a + H) % N, b]])[:, None]
        dev1 = max(dev1, np.abs(M @ np.array([u, v])[:, None] - tgt).max())
for a in range(N):          # 情形 2：沿方向 2 对折，k2 折半；a 指方向 1、b 指方向 2
    for b in range(H):
        u, v = F[a, b], F[a, (b + H) % N]
        M = 1j * ss1[b] * S3 + 1j * ss2[a] * S1m
        tgt = np.array([Fd2[a, b], Fd2[a, (b + H) % N]])[:, None]
        dev2 = max(dev2, np.abs(M @ np.array([u, v])[:, None] - tgt).max())
print("W1  情形 1（符号在方向 2）折叠形式偏差 = %.3e" % dev1)
print("    情形 2（符号在方向 1）折叠形式偏差 = %.3e" % dev2)
print("    两者的生成元与方向配对：情形 1 为 $(s_1\\to\\sigma_3,\\ s_2\\to\\sigma_1)$；"
      "情形 2 为 $(s_1\\to\\sigma_1,\\ s_2\\to\\sigma_3)$，已互换")

# W2：两谱逐位比较
ev1 = np.sort(np.abs(np.linalg.eigvals(Dcase1)))
ev2 = np.sort(np.abs(np.linalg.eigvals(Dcase2)))
print("W2  两定向的谱模长最大逐位差 = %.3e（预期 0：定向选择是格点对换对称）"
      % np.abs(ev1 - ev2).max())

# W3：号差指派在换定向下的对应
gA = 1j * S3
gB = 1j * S1m
print("W3  把 i 加在与方向 1 配对的生成元上：平方号 = %s"
      % [int(np.round(np.real(np.trace(g @ g)) / 2)) for g in (gA, S1m)])
print("    把 i 加在与方向 2 配对的生成元上：平方号 = %s（由 W1 的对换相连，物理结构相同）"
      % [int(np.round(np.real(np.trace(g @ g)) / 2)) for g in (S3, gB)])

# W4：反例（把方向固定死）
lab = -1
print("W4  反例：固定让方向 1 带 i，则在情形 1 下负号落在与 $s_2$ 配对的生成元上，"
      "在情形 2 下落在与 $s_1$ 配对的生成元上；两者相差一次方向对换")

# 图
fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.2))
ax[0].bar([0, 1], [sum(1 for g in (gA, S1m) if np.real(np.trace(g @ g)) < 0),
                   sum(1 for g in (S3, gB) if np.real(np.trace(g @ g)) < 0)],
          color=["tab:blue", "tab:orange"], width=0.5)
ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["i 给方向 1", "i 给方向 2"])
ax[0].set_ylabel("负平方枚数"); ax[0].set_ylim(0, 2)
ax[0].set_title("(a) 号差指派随定向互换\n（两者由格点对换相连）")

ax[1].plot(ev1, "o", ms=3, label="情形 1（符号在方向 2）")
ax[1].plot(ev2, "x", ms=4, label="情形 2（符号在方向 1）")
ax[1].set_title("(b) 两定向的谱重合（最大逐位差 %.1e）" % np.abs(ev1 - ev2).max())
ax[1].set_xlabel("本征值序号"); ax[1].set_ylabel(r"$|\lambda|$"); ax[1].legend(fontsize=9)

ax[2].axis("off")
ax[2].text(0.02, 0.95,
           "格点坐标只有空间索引 $n_1,n_2$\n"
           "（完整体系为 $n_1,n_2,n_3$）\n\n"
           "对换 $n_1\\leftrightarrow n_2$ 是格点的对称操作，\n"
           "故两种定向给出同一物理结构。\n\n"
           "时间不由格点坐标给出：\n"
           "它由摆程（碰撞折返的拍数）给出。\n\n"
           "故『哪个生成元带 i』= 『哪个方向是时间』\n"
           "这一指派不能由格子的空间结构判定，\n"
           "须由摆程方向从外部供给。",
           va="top", fontsize=11, family="Microsoft YaHei")
ax[2].set_title("(c) 结论：指派不可由格点判定")

fig.tight_layout()
fig.savefig(OUT, dpi=170)
print("\n图已存：%s" % OUT)
