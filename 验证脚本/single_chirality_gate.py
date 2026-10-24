# -*- coding: utf-8 -*-
"""判据第 650 号：单手征障碍候选四的闸门——谷能否被守恒的扇区标签分开。

模型：二维周期格点 N=32，交错算符 D = D1 + P1 D2（第 632 号口径，P1 = diag((-1)^{n1})）。

先立的期望（写实现之前）：
  冒烟  i*D 厄米（偏差 0）。
  G1    D 的零模（谷）计数：期望 4（配第 631 号附读数「零模由 30 切为 4 个孤立点」）。
  G2    零模子空间上位置奇偶 P1 的谱：期望同时含 +1 与 -1，即各谷带不同的格点标签。
  G3    P1 与 D 的反对易子：期望非零，即该标签不守恒，不能作扇区约束。
  反例  把算符换成朴素加 Wilson 项：零模计数下降（读数），说明该读数对算符修改有区分力。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

N = 32
I2 = np.eye(N)


def shift(n):
    S = np.zeros((n, n))
    for i in range(n):
        S[i, (i + 1) % n] = 1.0
    return S


Sx, Sy = shift(N), shift(N)
S1 = np.kron(Sx, I2)
S2 = np.kron(I2, Sy)
D1 = (S1 - S1.T) / 2.0
D2 = (S2 - S2.T) / 2.0
n1arr = np.repeat(np.arange(N), N)
P1 = np.diag([(-1.0) ** a for a in n1arr])

D = D1 + P1 @ D2

def dim_null(M, tol=1e-10):
    sv = np.linalg.svd(M, compute_uv=False)
    return int(np.sum(sv < tol)), sv


def null_basis(M, tol=1e-10):
    u, sv, _ = np.linalg.svd(M)
    k = int(np.sum(sv < tol))
    return u[:, :k]


# ---- 冒烟 ----
print("冒烟  i*D 厄米偏差 = %.3e（期望 0）" % np.abs(1j * D - (1j * D).conj().T).max())

# ---- G1 零模计数 ----
k1, _ = dim_null(D)
print("G1    交错算符 D 的零模计数 = %d（期望 4）" % k1)

# ---- G2 零模子空间上 P1 的谱 ----
B = null_basis(D)
Ppi = B @ B.conj().T
Pr = Ppi @ P1 @ Ppi
ev = np.linalg.eigvalsh(Pr)
ev = ev[np.abs(ev) > 1e-6]
print("G2    零模子空间上 P1 的谱 = %s（期望同时含 +1 与 -1）"
      % np.round(np.sort(ev), 6).tolist())

# ---- G3 标签的守恒性 ----
ac = np.abs(P1 @ D + D @ P1).max()
print("G3    P1 与 D 的反对易子最大模 = %.6f（期望非零，即标签不守恒）" % ac)

# ---- 反例：朴素加 Wilson 项 ----
r = 1.0
Ibig = np.eye(N * N)
W1 = (2.0 * Ibig - S1 - S1.T) / 2.0
W2 = (2.0 * Ibig - S2 - S2.T) / 2.0
D_W = (D1 + D2) + r * (W1 + W2)
k2, _ = dim_null(D_W)
print("反例  朴素加 Wilson（r=%.1f）后的零模计数 = %d（读数，期望下降）" % (r, k2))

# ---- 图 ----
fig, ax = plt.subplots(1, 2, figsize=(10.6, 4.2))
ax[0].bar([0, 1], [k1, k2], color=["tab:blue", "tab:orange"], width=0.5)
ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["交错算符", "朴素+Wilson"], fontsize=9)
ax[0].set_ylabel("零模（谷）计数")
ax[0].set_title("(a) 谷计数对算符修改敏感", fontsize=10)
for i, v in enumerate([k1, k2]):
    ax[0].text(i, v + 0.1, "%d" % v, ha="center", fontsize=9)
ax[0].set_ylim(0, max(k1, k2) + 1.5)

ax[1].bar(range(len(ev)), np.sort(ev), color="tab:green", width=0.5)
ax[1].set_xticks(range(len(ev)))
ax[1].set_ylabel("零模子空间上 P1 的本征值")
ax[1].set_title("(b) 各谷带不同的位置奇偶标签", fontsize=10)
ax[1].set_ylim(-1.4, 1.4)
fig.tight_layout()
fig.savefig(r"D:\software\TianXingFC\images\single_chirality_gate.png", dpi=150)
print("图已存 images/single_chirality_gate.png")
