# -*- coding: utf-8 -*-
"""判据第 639 号：格点三枚生成元是否对应自旋空间的三枚（局域性穷举）。

问题：第 637 号的 $\Gamma_1=I\otimes\sigma_3$、$\Gamma_2=\sigma_3\otimes\sigma_1$、$\Gamma_5=-(\sigma_2\otimes\sigma_3)$
      若对应标准物理的 $\alpha_i$（内部自旋空间矩阵），则它们应当由格点上的内部（零位移、局域）算符实现。
      本轮把局域字穷举一遍，看哪些字的折叠形式是常值矩阵，再问三枚生成元是否在其中。

构造：三维周期格点 $8^3$，折叠基为（分支 $(d_a,d_b)$，方向 3 指标 $c$），维数 $4\times8=32$。
      基本局域字：移位 $S_a$、$S_a^\dagger$（$a=1,2,3$）与奇偶 $P_a=(-1)^{n_a}$（$a=1,2,3$）。
      折叠形式：$S_a$ 为对角相位 $\mathrm{e}^{\mp\mathrm{i}k_a}$；$P_1,P_2$ 为分支交换（常值）；$P_3$ 为 $c$ 指标平移。

先立的判据与期望：
  W1 零位移局域字（奇偶的乘积，八枚）的折叠形式在分支空间上只张出四维：$\{I,\;I\otimes\sigma_1,\;\sigma_1\otimes I,\;\sigma_1\otimes\sigma_1\}$。
  W2 字长 $\le 3$ 的全部局域字中，折叠形式与轨道无关者恰为上述零位移字（非零位移字的折叠形式随轨道变）。
  W3 $\Gamma_1$、$\Gamma_2$、$\Gamma_5$ 到该四维空间的距离非零（期望非零，给出数值）；$\Gamma_3=\sigma_1\otimes\sigma_1$ 的距离为零。
  W4 $\Gamma_3$ 恰为 $\eta_3=P_1P_2$ 的折叠形式。
  反例（恒真检验）：去掉局域限制（允许非局域算符）后 $\Gamma_1$ 当然可由格点算符实现，判据翻，
     说明"局域性"是判据的实质而非装饰。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = r"D:\software\TianXingFC\images\chirality_locality.png"
N = 8
H = N // 2
NB = 4
DIM = NB * N
ORB = [(0, 0), (1, 0), (0, 1), (1, 1)]

s1 = np.array([[0, 1], [1, 0]], dtype=complex)
s2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
s3 = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2)
I4 = np.eye(4)
G1 = np.kron(I2, s3)
G2 = np.kron(s3, s1)
G3 = np.kron(s1, s1)
G5 = 1j * (G1 @ G2 @ G3)


def branch_index(da, db):
    return 2 * db + da


def factor(name, a, b, c):
    """基本局域字在轨道 (a,b,c) 上的折叠形式（32×32）。"""
    M = np.zeros((DIM, DIM), dtype=complex)
    if name in ("S1", "S2", "S3", "S1d", "S2d", "S3d"):
        sgn = 1.0 if name.endswith("d") else -1.0
        ax_idx = int(name[1])
        for da, db in ORB:
            if ax_idx == 1:
                k = 2 * np.pi * (a + da * H) / N
            elif ax_idx == 2:
                k = 2 * np.pi * (b + db * H) / N
            else:
                k = 2 * np.pi * c / N
            for cc in range(N):
                M[branch_index(da, db) * N + cc, branch_index(da, db) * N + cc] = np.exp(1j * sgn * k)
    elif name in ("P1", "P2"):
        for da, db in ORB:
            dst = (1 - da, db) if name == "P1" else (da, 1 - db)
            for cc in range(N):
                M[branch_index(*dst) * N + cc, branch_index(da, db) * N + cc] = 1.0
    elif name == "P3":
        for da, db in ORB:
            for cc in range(N):
                M[branch_index(da, db) * N + (cc + H) % N, branch_index(da, db) * N + cc] = 1.0
    return M


def word_form(word, a, b, c):
    M = np.eye(DIM, dtype=complex)
    for w in word:
        M = M @ factor(w, a, b, c)
    return M


def branch_block(M, tol=1e-10):
    """若 M 是（4×4 分支）⊗（c 指标单位阵）型，返回 4×4 分支块；否则返回 None。"""
    for ci in range(N):
        for cj in range(N):
            sub = np.array([[M[b * N + ci, b2 * N + cj] for b2 in range(NB)] for b in range(NB)])
            ref = np.array([[M[b * N, b2 * N] for b2 in range(NB)] for b in range(NB)])
            if ci == cj:
                if np.abs(sub - ref).max() > tol:
                    return None
            elif np.abs(sub).max() > tol:
                return None
    return np.array([[M[b * N, b2 * N] for b2 in range(NB)] for b in range(NB)])


# ---------------- W1/W2 穷举局域字 ----------------
BASIC = ["S1", "S1d", "S2", "S2d", "S3", "S3d", "P1", "P2", "P3"]
words = [()]
for L in (1, 2, 3):
    for w in list(words):
        if len(w) == L - 1:
            for f in BASIC:
                words.append(w + (f,))
print("W2  局域字总数（字长 ≤ 3）= %d" % len(words))

orbits = [(1, 1, 1), (2, 3, 5), (0, 2, 7)]
const_words, blocks = [], []
for w in words:
    forms = [word_form(w, *o) for o in orbits]
    if max(np.abs(forms[0] - f).max() for f in forms[1:]) < 1e-10:
        B = branch_block(forms[0])
        const_words.append((w, B))
        if B is not None:
            blocks.append(B)
print("W2  折叠形式与轨道无关者 = %d 枚；其中在分支空间上为常值块者 = %d 枚"
      % (len(const_words), len(blocks)))

zero_disp = [w for w in words if all(w.count(x) == w.count(x + "d") for x in ("S1", "S2", "S3"))]
print("W1  零位移局域字（移位成对出现）= %d 枚" % len(zero_disp))

# 分支块的张成空间
if blocks:
    Mat = np.array([B.reshape(-1) for B in blocks]).T
    sv = np.linalg.svd(Mat, compute_uv=False)
    rank = int(np.sum(sv > 1e-9))
    basis_vecs = np.linalg.svd(Mat)[0][:, :rank]
    print("W1  分支块张成空间维数 = %d（期望 4）" % rank)
else:
    basis_vecs = np.zeros((16, 0), dtype=complex)
    rank = 0

# ---------------- W3 三枚生成元到该空间的距离 ----------------
def dist_to_span(G, basis_vecs):
    g = G.reshape(-1)
    if basis_vecs.shape[1] == 0:
        return float(np.linalg.norm(g))
    proj = basis_vecs @ (basis_vecs.conj().T @ g)
    return float(np.linalg.norm(g - proj) / np.linalg.norm(g))


print("W3  到局域张成空间的相对距离：")
targets = [("Gamma_1", G1), ("Gamma_2", G2), ("Gamma_3", G3), ("Gamma_5", G5)]
dists = []
for nm, G in targets:
    d = dist_to_span(G, basis_vecs)
    dists.append(d)
    print("    %-8s = %.6f" % (nm, d))

# ---------------- W4 Gamma_3 与 eta_3 ----------------
eta3 = factor("P1", 1, 1, 1) @ factor("P2", 1, 1, 1)
B3 = branch_block(eta3)
print("W4  eta_3 = P_1 P_2 的折叠形式与 Gamma_3 的偏差 = %.3e（期望 0）" % np.abs(B3 - G3).max())

# ---------------- 反例：去掉局域限制 ----------------
perm = []
for a in range(H):
    for b in range(H):
        for c in range(N):
            for da, db in ORB:
                k1, k2, k3 = (a + da * H) % N, (b + db * H) % N, c
                perm.append(((k1 * N + k2) * N + k3, ((a * H + b) * 4 + branch_index(da, db)) * N + c))
Pf = np.zeros((N ** 3, N ** 3))
for i, j in perm:
    Pf[i, j] = 1.0
target = np.kron(np.eye(H * H), np.kron(G1, np.eye(N)))
G1_full = Pf.T @ target @ Pf
print("反例：把分支块 Gamma_1 按非局域算符实现后，其折叠形式与 Gamma_1 的偏差 = %.3e（期望 0，判据翻）"
      % np.abs(Pf @ G1_full @ Pf.T - target).max())

# ---------------- 图 ----------------
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
ax[0].bar(range(4), dists, color=["tab:red", "tab:red", "tab:green", "tab:red"], width=0.5)
ax[0].axhline(0, color="k", lw=0.8)
ax[0].set_xticks(range(4))
ax[0].set_xticklabels([r"$\Gamma_1$", r"$\Gamma_2$", r"$\Gamma_3$", r"$\Gamma_5$"], fontsize=11)
ax[0].set_ylabel("到局域张成空间的相对距离")
ax[0].set_title("三枚生成元与局域算符的可实现性", fontsize=11)
for i, v in enumerate(dists):
    ax[0].text(i, v + 0.02, "%.3f" % v, ha="center", fontsize=9)

tile = np.zeros((8, 8))
for idx, B in enumerate([I4, np.kron(I2, s1), np.kron(s1, I2), np.kron(s1, s1)]):
    r, cc = idx // 2, idx % 2
    tile[r * 4:(r + 1) * 4, cc * 4:(cc + 1) * 4] = np.abs(B)
ax[1].imshow(tile, cmap="Blues", vmin=0, vmax=1)
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[1].set_title("局域字能给出的分支结构：\n$I$、$I\\otimes\\sigma_1$、$\\sigma_1\\otimes I$、$\\sigma_1\\otimes\\sigma_1$",
                fontsize=10)
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print("\n图已存 %s" % OUT)
