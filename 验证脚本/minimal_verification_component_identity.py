# -*- coding: utf-8 -*-
"""判据第 634 号：旋量分量的几何身份——它是位置奇偶的本征值，还是"住在不同子格"。

判据与先立的期望：
  V1 折叠空间里位置奇偶 P 的作用是交换两个分量（期望逐位精确）。
  V2 P 在折叠空间的本征值为 ±1（期望精确）。
  V3 朴素实空间分块下，两个分量方向上的差分算符是否相同（期望：不相同，差一个两步平移）。
  V4 反例：把 P 换成常数 1，折叠空间不再有二分量标签（期望判据可区分）。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = r"D:\software\TianXingFC\images\component_identity.png"
N = 32
H = N // 2
rng = np.random.default_rng(20261015)


def shift1d(n):
    S = np.zeros((n, n))
    for i in range(n):
        S[i, (i + 1) % n] = 1.0
    return S


S = shift1d(N)
D = (S - S.T) / 2.0
P = np.diag([(-1.0) ** n for n in range(N)])

psi = rng.standard_normal(N)
F = np.fft.fft(psi)

# 折叠：k_1 取前半布里渊区，(u,v) = (psi~(k), psi~(k-pi))
u = F[:H]
v = F[H:]

# V1：P 的作用是交换分量
Fp = np.fft.fft(P @ psi)
up, vp = Fp[:H], Fp[H:]
dev_swap = max(np.abs(up - v).max(), np.abs(vp - u).max())
print("V1  折叠后 P 的交换偏差 = %.3e（期望 0）" % dev_swap)

# V2：P 在折叠空间的本征值
Pf = np.zeros((2 * H, 2 * H), dtype=complex)
for a in range(H):
    Pf[a, H + a] = 1.0
    Pf[H + a, a] = 1.0
ev = np.linalg.eigvalsh((Pf + Pf.conj().T) / 2)
print("V2  P 在折叠空间的本征值 = %s（期望 ±1）" % np.round(ev, 12).tolist()[:3] + " ...")

# V3：朴素实空间分块
perm = np.concatenate([np.arange(0, N, 2), np.arange(1, N, 2)])   # 偶数位在前
B = np.zeros((N, N))
for i, j in enumerate(perm):
    B[i, j] = 1.0
Db = B @ D @ B.T
b01 = Db[:H, H:]
b10 = Db[H:, :H]
S2 = shift1d(H)                    # 粗格一步 = 细格两步
print("V3  朴素分块：||block10 - block01|| = %.6f（期望非零）" % np.abs(b10 - b01).max())
print("    朴素分块：||block10 - S2 block01|| = %.3e（期望 0：差一个两步平移）"
      % np.abs(b10 - S2 @ b01).max())

# V4：反例
Pone = np.eye(N)
F1 = np.fft.fft(Pone @ psi)
dev_one = max(np.abs(F1[:H] - v).max(), np.abs(F1[H:] - u).max())
print("V4  反例（P 换成常数 1）：交换偏差 = %.6f（期望非零，判据可区分）" % dev_one)

# 图
fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.2))

ax[0].stem(np.arange(N), np.real(P[:N, 0]) * 0 + [(-1.0) ** n for n in range(N)],
           basefmt=" ", linefmt="C0-", markerfmt="C0o")
ax[0].set_title("(a) 位置奇偶 $P=(-1)^n$ 的取值")
ax[0].set_xlabel("格点 $n$"); ax[0].set_ylabel("$P$ 的取值"); ax[0].set_ylim(-1.4, 1.4)

ax[1].plot(np.arange(H), np.abs(u), "o", ms=3, label="$u=\\tilde\\psi(k)$")
ax[1].plot(np.arange(H), np.abs(v), "x", ms=3, label="$v=\\tilde\\psi(k-\\pi)$")
ax[1].set_title("(b) 折叠后的两个分量（$P$ 交换它们）")
ax[1].set_xlabel("折半动量序号"); ax[1].set_ylabel("模"); ax[1].legend(fontsize=9)

im = ax[2].imshow(np.abs(b10 - S2 @ b01), cmap="viridis", aspect="auto")
ax[2].set_title("(c) 朴素分块：$|block_{10}-S_2\\,block_{01}|$\n（全零即两块差分恰差一个两步平移）")
ax[2].set_xlabel("粗格列指标"); ax[2].set_ylabel("粗格行指标")
fig.colorbar(im, ax=ax[2], fraction=0.046)

for a in ax:
    a.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(OUT, dpi=170)
print("\n图已存：%s" % OUT)
