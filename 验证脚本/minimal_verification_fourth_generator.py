# -*- coding: utf-8 -*-
"""判据第 645 号：第四枚（时间）生成元能否由体系选出。

先立的期望（写实现之前）：
  冒烟  三枚空间生成元两两反对易子为零、平方偏差为零；omega = G1 G2 G3 与三枚对易。
  K1    与三枚全反对易的厄米算符构成实空间，期望维数 2（配第 637 号 Y7 读数）；
        该空间内任元素平方恒为标量倍 I（归一化即给出厄米对合）。
  K2    「候选须与 omega 对易」这一条件：期望为空集（全部候选与 omega 反对易），
        即该条件是恒空的，不能用来选第四枚。
  K3    族成员到局域张成空间 {I, I⊗s1, s1⊗I, s1⊗s1} 的相对距离：期望非零（读数）。
        对照为第 639 号口径：Gamma_3 = 0.000000，Gamma_1 = Gamma_2 = 1.000000。
  反例  把张成空间换成「包含族成员自身」的空间，K3 须翻为零（判别器有区分力）。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

s1 = np.array([[0, 1], [1, 0]], dtype=complex)
s3 = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)
I4 = np.eye(4)

G = [np.kron(I2, s3), np.kron(s3, s1), np.kron(s1, s1)]
w = G[0] @ G[1] @ G[2]

# ---- 冒烟 ----
ac = max(np.abs(G[i] @ G[j] + G[j] @ G[i]).max() for i in range(3) for j in range(i + 1, 3))
sq = max(np.abs(X @ X - I4).max() for X in G)
cw = max(np.abs(w @ X - X @ w).max() for X in G)
print("冒烟  三对反对易子最大 = %.3e（期望 0）" % ac)
print("冒烟  三枚平方偏差最大 = %.3e（期望 0）" % sq)
print("冒烟  omega 与三枚对易子最大 = %.3e（期望 0）" % cw)

# ---- 厄米基（16 维实） ----
B = []
for i in range(4):
    E = np.zeros((4, 4), complex); E[i, i] = 1; B.append(E)
for i in range(4):
    for j in range(i + 1, 4):
        E = np.zeros((4, 4), complex); E[i, j] = 1; E[j, i] = 1; B.append(E)
        E = np.zeros((4, 4), complex); E[i, j] = 1j; E[j, i] = -1j; B.append(E)
B = np.array(B)
assert len(B) == 16

# ---- K1 与三枚全反对易的厄米空间 ----
rows = [[(B[m] @ G[k] + G[k] @ B[m])[r, c] for m in range(16)]
        for k in range(3) for r in range(4) for c in range(4)]
A = np.array(rows)
Ar = np.vstack([A.real, A.imag])
_, sv, vh = np.linalg.svd(Ar)
null = vh[sv < 1e-9]
print("K1    与三枚反对易的厄米空间实维数 = %d（期望 2）" % len(null))
Xs = [(c @ B.reshape(16, 16)).reshape(4, 4) for c in null]
scal = [float(np.abs(X @ X - (np.trace(X @ X) / 4) * I4).max()) for X in Xs]
print("K1    空间内元素平方恒为标量倍 I（非标量偏差）=%s（期望 0）" % ["%.1e" % v for v in scal])

# ---- K2 「与 omega 对易」条件是否为空 ----
comm = [float(np.abs(X @ w - w @ X).max()) for X in Xs]
anti = [float(np.abs(X @ w + w @ X).max()) for X in Xs]
print("K2    候选与 omega 对易子最大 = %s（均非零，故「对易」条件空集）" % ["%.3f" % v for v in comm])
print("K2    候选与 omega 反对易子最大 = %s（期望 0）" % ["%.1e" % v for v in anti])

# ---- K3 局域张成空间与相对距离（第 639 号口径） ----
L = np.array([np.eye(4), np.kron(I2, s1), np.kron(s1, I2), np.kron(s1, s1)])
Q, _ = np.linalg.qr(L.reshape(4, 16).T)


def dist(X):
    v = X.reshape(16)
    v = v / np.linalg.norm(v)
    return float(np.linalg.norm(v - Q @ (Q.conj().T @ v)))


X0, X1 = [X / np.sqrt(np.trace(X @ X) / 4) for X in Xs]
ts = np.linspace(0.0, np.pi, 13)
fam = [np.cos(t) * X0 + np.sin(t) * X1 for t in ts]
fam_d = [dist(X) for X in fam]
print("K3    族成员到局域张成空间的相对距离 = %.6f 至 %.6f（期望非零；最小 %.6f）"
      % (min(fam_d), max(fam_d), min(fam_d)))
print("K3    对照（第 639 号口径）：Gamma_1 = %.6f，Gamma_2 = %.6f，Gamma_3 = %.6f"
      % (dist(G[0]), dist(G[1]), dist(G[2])))

# ---- 反例：张成空间含族成员本身 ----
Lc = np.array([np.eye(4), np.kron(I2, s1), np.kron(s1, I2), np.kron(s1, s1), X0.astype(complex)])
Qc, _ = np.linalg.qr(Lc.reshape(5, 16).T)


def dist_c(X):
    v = X.reshape(16); v = v / np.linalg.norm(v)
    return float(np.linalg.norm(v - Qc @ (Qc.conj().T @ v)))


print("反例  把 X0 并入张成空间后，X0 的距离 = %.1e（期望 0，判别器有区分力）" % dist_c(X0))

# ---- 图 ----
fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.3))
names = ["Gamma_1", "Gamma_2", "Gamma_3", "族成员 t=0", "族成员 t=pi/2"]
vals = [dist(G[0]), dist(G[1]), dist(G[2]), fam_d[0], fam_d[len(fam_d) // 2]]
cols = ["tab:red", "tab:red", "tab:green", "tab:purple", "tab:purple"]
ax[0].bar(range(5), vals, color=cols, width=0.5)
ax[0].set_xticks(range(5)); ax[0].set_xticklabels(names, rotation=20, fontsize=8)
ax[0].set_ylabel("到局域张成空间的相对距离")
ax[0].set_title("(a) 局域判别：只有 Gamma_3 在局域类内", fontsize=10)
for i, v in enumerate(vals):
    ax[0].text(i, v + 0.03, "%.3f" % v, ha="center", fontsize=8)
ax[0].set_ylim(0, 1.2)

ax[1].plot(ts / np.pi, fam_d, "o-", color="tab:purple")
ax[1].axhline(1.0, ls="--", color="gray", lw=0.8)
ax[1].set_ylim(0, 1.2); ax[1].set_xlabel(r"族参数 $t/\pi$")
ax[1].set_ylabel("相对距离")
ax[1].set_title("(b) 族成员全体非局域（距离恒为 1）", fontsize=10)
fig.tight_layout()
fig.savefig(r"D:\software\TianXingFC\images\fourth_generator_locality.png", dpi=150)
print("图已存 images/fourth_generator_locality.png")
