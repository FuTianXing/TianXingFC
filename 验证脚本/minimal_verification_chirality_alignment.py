# -*- coding: utf-8 -*-
"""判据第 638 号：手性口径对表的算符层核对（标准表示、格点折叠表示、代数盲区与局域性判别）。

构造：
  标准侧：3+1 维 Dirac 表示，gamma^0 = diag(I,-I)、gamma^i = [[0,sigma_i],[-sigma_i,0]]，
          alpha_i = gamma^0 gamma_i、beta = gamma^0、gamma_5 = i gamma^0 gamma^1 gamma^2 gamma^3。
  格点侧：636 的四分量折叠表示，Gamma_1 = I⊗sigma_3、Gamma_2 = sigma_3⊗sigma_1、
          Gamma_3 = sigma_1⊗sigma_1、Gamma_5 = i Gamma_1 Gamma_2 Gamma_3；
          位置奇偶 P_1 = I⊗sigma_1、P_2 = sigma_1⊗I；第四枚生成元 X 取 637 的核。

先立的判据与期望：
  Z1 标准侧复现库内第 500 条的两条关系（[gamma_5, alpha_i] = 0、{gamma_5, beta} = 0、
     [gamma_5, H] = 2m gamma_5 beta、{gamma_5, H} = 2 gamma_5 (alpha.p)，H = alpha.p + beta m），
     以及第 494 条口径的 {gamma_5, D} = 0（D = m gamma^0 + gamma.p）。期望全为零。
  Z2 格点侧的同一模式：{Gamma_5, Gamma_i} = 2 Gamma_5 Gamma_i（即对易）、{Gamma_5, X} = 0。
  Z3 位置奇偶与单枚 gamma 同型：P_1、P_2 对 {Gamma_1..Gamma_5} 的反对易模式，与 gamma^1、gamma^2
     对 {alpha_1..alpha_3, gamma_5} 的模式逐条相同。
  Z4 第四枚生成元在格点原生清单（七枚）里缺位；但该清单生成的代数为全矩阵代数，X 在其内。
  Z5 代数盲区：存在酉变换 U 使 U Gamma_i U^dag = alpha_i，故代数模式匹配是必要而非充分。
  Z6 局域性判别：Gamma_3 恰为格点原生局域算符 eta_3 = P_1 P_2 的折叠形式；Gamma_1、Gamma_2
     在格点原生自然清单里没有对应的局域单项（其折叠形式随动量变）。
  反例（恒真检验）：用单位阵冒充手性算符，与第四枚生成元的反对易子非零，Z2 的模式失配。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = r"D:\software\TianXingFC\images\chirality_alignment.png"

# ---------------- 标准侧 ----------------
I2 = np.eye(2)
s1 = np.array([[0, 1], [1, 0]], dtype=complex)
s2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
s3 = np.array([[1, 0], [0, -1]], dtype=complex)
Z2 = np.zeros((2, 2), dtype=complex)
blk = lambda a, b, c, d: np.block([[a, b], [c, d]])
g0 = blk(I2, Z2, Z2, -I2)
gi = [blk(Z2, s, -s, Z2) for s in (s1, s2, s3)]
g5 = 1j * (g0 @ gi[0] @ gi[1] @ gi[2])
al = [g0 @ g for g in gi]
be = g0


def mx(M):
    return float(np.abs(M).max())


p = np.array([0.37, -0.62, 0.85])
m = 0.51
H = sum(p[i] * al[i] for i in range(3)) + m * be
Dop = m * g0 + sum(p[i] * gi[i] for i in range(3))

print("Z1  标准侧（复现第 500、494 条口径）")
print("    [gamma_5, alpha_i] 最大模 = %.3e（期望 0）"
      % max(mx(g5 @ a - a @ g5) for a in al))
print("    {gamma_5, beta} = %.3e（期望 0）" % mx(g5 @ be + be @ g5))
print("    [gamma_5, H] 与 2m gamma_5 beta 之差 = %.3e（期望 0）"
      % mx((g5 @ H - H @ g5) - 2 * m * (g5 @ be)))
print("    {gamma_5, H} 与 2 gamma_5 (alpha.p) 之差 = %.3e（期望 0）"
      % mx((g5 @ H + H @ g5) - 2 * (g5 @ sum(p[i] * al[i] for i in range(3)))))
print("    {gamma_5, D} = %.3e（期望 0，第 494 条的 Dirac 算子形式）" % mx(g5 @ Dop + Dop @ g5))

# ---------------- 格点侧 ----------------
I4 = np.eye(4)
G1 = np.diag([1.0, -1.0, 1.0, -1.0])
G2 = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, -1], [0, 0, -1, 0]], dtype=float)
G3 = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]], dtype=float)
G5 = 1j * (G1 @ G2 @ G3)
Pf1 = np.kron(np.eye(2), s1.real)
Pf2 = np.kron(s1.real, np.eye(2))
gens = [G1, G2, G3]

# 第四枚生成元 X：与三枚全反对易的厄米对合（637 的核）
basis = []
for i in range(4):
    E = np.zeros((4, 4), dtype=complex); E[i, i] = 1.0; basis.append(E)
for i in range(4):
    for j in range(i + 1, 4):
        E = np.zeros((4, 4), dtype=complex); E[i, j] = E[j, i] = 1.0; basis.append(E)
        F = np.zeros((4, 4), dtype=complex); F[i, j] = 1j; F[j, i] = -1j; basis.append(F)
nrm2 = [np.trace(B.conj().T @ B).real for B in basis]
rows = []
for g in gens:
    for B in basis:
        C = B @ g + g @ B
        rows.append([np.trace(Bb.conj().T @ C).real / n for Bb, n in zip(basis, nrm2)])
ker = np.linalg.svd(np.array(rows), compute_uv=False)
ker = np.linalg.svd(np.array(rows))[2][np.sum(ker > 1e-10):]
X = sum(ker[0, i] * basis[i] for i in range(len(basis)))
X = X / np.linalg.norm(X) * 2.0
print("\nZ2  格点侧（对表的目标模式）")
print("    {Gamma_5, Gamma_i} 与 2 Gamma_5 Gamma_i 之差 = %.3e（期望 0，即对易）"
      % max(mx(G5 @ g + g @ G5 - 2 * (G5 @ g)) for g in gens))
print("    {Gamma_5, X} = %.3e（期望 0，即反对易；X 为第四枚生成元）" % mx(G5 @ X + X @ G5))
print("    X 自检：与三枚的反对易子最大模 = %.3e，X^2 与 I 之差 = %.3e"
      % (max(mx(X @ g + g @ X) for g in gens), mx(X @ X - I4)))

# ---------------- Z3 模式对应 ----------------
def pattern(ops, targets):
    out = []
    for A in ops:
        out.append([0 if mx(A @ T + T @ A) < 1e-9 else 1 for T in targets])
    return np.array(out)


pat_std = pattern([gi[0], gi[1]], al + [g5])
pat_lat = pattern([Pf1, Pf2], gens + [G5])
print("\nZ3  反对易模式（0 = 反对易，1 = 对易）")
print("    标准侧 gamma^1、gamma^2 对 (alpha_1, alpha_2, alpha_3, gamma_5)：")
print(pat_std)
print("    格点侧 P_1、P_2 对 (Gamma_1, Gamma_2, Gamma_3, Gamma_5)：")
print(pat_lat)
print("    两侧模式之差 = %d（期望 0）" % int(np.abs(pat_std - pat_lat).sum()))

# ---------------- Z4 第四枚在原生清单里的缺位 ----------------
native = [("Gamma_1", G1), ("Gamma_2", G2), ("Gamma_3", G3), ("Gamma_5", G5),
          ("P_1", Pf1), ("P_2", Pf2), ("I", I4)]
found = [nm for nm, M in native
         if mx(M @ M - I4) < 1e-12 and mx(M - M.conj().T) < 1e-12
         and all(mx(M @ g + g @ M) < 1e-12 for g in gens)]
print("\nZ4  格点原生七枚中，与三枚全反对易的厄米对合 = %s（期望空）" % (found or "无"))
prod = []
for a in range(len(native)):
    for b in range(a, len(native)):
        prod.append(native[a][1] @ native[b][1])
sv = np.linalg.svd(np.array([Pr.reshape(-1) for Pr in prod]), compute_uv=False)
print("    七枚两两乘积张成的空间维数 = %d（复数维，上界 16）" % int(np.sum(sv > 1e-9)))
Amat = np.array([Pr.reshape(-1) for Pr in prod]).T
coef, res, rank, _ = np.linalg.lstsq(Amat, X.reshape(-1), rcond=None)
print("    X 在乘积张成空间内的最小二乘残差 = %.3e（为零则 X 在该代数内）" % np.abs(Amat @ coef - X.reshape(-1)).max())

# ---------------- Z5 酉等价盲区 ----------------
# 线性映射 U -> (U Gamma_i - alpha_i U)_i，按基矩阵逐列装配（避开向量化约定）
cols = []
for pp in range(4):
    for qq in range(4):
        E = np.zeros((4, 4), dtype=complex)
        E[pp, qq] = 1.0
        cols.append(np.concatenate([(E @ g - a @ E).reshape(-1) for g, a in zip(gens, al)]))
Mbig = np.array(cols).T
sv = np.linalg.svd(Mbig, compute_uv=False)
kern = np.linalg.svd(Mbig)[2][np.sum(sv > 1e-9):]
print("\nZ5  酉等价（代数盲区）")
print("    交织空间维数（复数）= %d（非零即有解）" % kern.shape[0])
K = [kern[k].conj().reshape(4, 4) for k in range(kern.shape[0])]
best = (9e9, None, 0.0)
for chart in (0, 1):
    for x in np.linspace(-4, 4, 161):
        for y in np.linspace(-4, 4, 161):
            z = x + 1j * y
            W = (K[0] + z * K[1]) if chart == 0 else (z * K[0] + K[1])
            s = np.sqrt(np.trace(W.conj().T @ W).real / 4)
            if s < 1e-6:
                continue
            Wn = W / s
            r = mx(Wn.conj().T @ Wn - np.eye(4))
            if r < best[0]:
                best = (r, Wn, z)
r5, U, z0 = best
# 局部细化：以最佳 z 为中心三轮小步长搜索
for step in (1e-2, 1e-4, 1e-6):
    for chart in (0, 1):
        for dx in (-step, 0.0, step):
            for dy in (-step, 0.0, step):
                z = z0 + dx + 1j * dy
                W = (K[0] + z * K[1]) if chart == 0 else (z * K[0] + K[1])
                s = np.sqrt(np.trace(W.conj().T @ W).real / 4)
                if s < 1e-6:
                    continue
                Wn = W / s
                r = mx(Wn.conj().T @ Wn - np.eye(4))
                if r < r5:
                    r5, U, z0 = r, Wn, z
print("    交织空间内酉解：W^dag W 与单位阵之差 = %.3e（期望 0）" % r5)
print("    U Gamma_i U^dag 与 alpha_i 之差 = %.3e（期望 0，即两组三元组酉等价）"
      % max(mx(U @ gens[i] @ U.conj().T - al[i]) for i in range(3)))

# ---------------- Z6 局域性判别 ----------------
N = 8
H = N // 2


def shift1d(n):
    S = np.zeros((n, n))
    for i in range(n):
        S[i, (i + 1) % n] = 1.0
    return S


I3 = np.eye(N)
Sa = np.kron(np.kron(shift1d(N), I3), I3)
Sb = np.kron(np.kron(I3, shift1d(N)), I3)
Sc = np.kron(np.kron(I3, I3), shift1d(N))
n1 = np.repeat(np.arange(N), N * N)
n2 = np.tile(np.repeat(np.arange(N), N), N)
n3 = np.tile(np.arange(N), N * N)
P1r = np.diag([(-1.0) ** a for a in n1])
P2r = np.diag([(-1.0) ** b for b in n2])
P3r = np.diag([(-1.0) ** c for c in n3])
ORB = [(0, 0), (1, 0), (0, 1), (1, 1)]


def folded_matrix(op, a, b, c):
    cols = []
    for da, db in ORB:
        ind = np.zeros((N, N, N), dtype=complex)
        ind[(a + da * H) % N, (b + db * H) % N, c] = 1.0
        psi = np.fft.ifftn(ind).reshape(-1)
        F = np.fft.fftn((op @ psi).reshape(N, N, N))
        cols.append(np.array([F[(a + da2 * H) % N, (b + db2 * H) % N, c] for da2, db2 in ORB]))
    return np.array(cols).T


cands = [("eta_1 = P_1", P1r), ("eta_2 = P_2", P2r), ("eta_3 = P_1 P_2", P1r @ P2r),
         ("(-1)^{n_3}", P3r), ("S_1", Sa), ("S_2", Sb), ("S_3", Sc),
         ("D_1", (Sa - Sa.T) / 2), ("D_2", (Sb - Sb.T) / 2), ("D_3", (Sc - Sc.T) / 2)]
best = {0: (None, 9e9), 1: (None, 9e9), 2: (None, 9e9)}
a0, b0, c0 = 1, 1, 1
print("\nZ6  局域性判别（格点原生算符的折叠形式与三枚生成元的偏差）")
for nm, op in cands:
    dev = [mx(folded_matrix(op, a0, b0, c0) - g) for g in gens]
    for k in range(3):
        if dev[k] < best[k][1]:
            best[k] = (nm, dev[k])
    print("    %-16s 对 Gamma_1/2/3 偏差 = %.3e / %.3e / %.3e" % (nm, dev[0], dev[1], dev[2]))
print("    与三枚生成元最接近者：%s -> %s（%.3e）；%s -> %s（%.3e）；%s -> %s（%.3e）"
      % ("Gamma_1", best[0][0], best[0][1], "Gamma_2", best[1][0], best[1][1],
         "Gamma_3", best[2][0], best[2][1]))

# ---------------- 反例 ----------------
ce = mx(I4 @ X + X @ I4)
print("\n反例：以单位阵冒充手性算符，{I, X} = %.3e（期望非零，模式失配，判据可区分）" % ce)

# ---------------- 图 ----------------
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
ax[0].imshow(np.vstack([pat_std, np.full((1, 4), np.nan), pat_lat]),
             cmap="Blues", vmin=0, vmax=1)
ax[0].set_yticks([0, 1, 3, 4])
ax[0].set_yticklabels(["标准 $\\gamma^1$", "标准 $\\gamma^2$", "格点 $P_1$", "格点 $P_2$"], fontsize=9)
ax[0].set_xticks(range(4))
ax[0].set_xticklabels(["第 1 枚", "第 2 枚", "第 3 枚", "手性"], fontsize=9)
ax[0].set_title("反对易模式（深色 = 对易）\n两侧逐条相同", fontsize=11)
for i, row in enumerate(np.vstack([pat_std, pat_lat])):
    for j, v in enumerate(row):
        ax[0].text(j, i + (0.0 if i < 2 else 1.0), str(v), ha="center", va="center",
                   color="white" if v else "black", fontsize=10)

names = [nm for nm, _ in cands]
fig_labels = [r"$\eta_1=P_1$", r"$\eta_2=P_2$", r"$\eta_3=P_1P_2$", r"$(-1)^{n_3}$",
              r"$S_1$", r"$S_2$", r"$S_3$", r"$D_1$", r"$D_2$", r"$D_3$"]
devs = [min(mx(folded_matrix(op, a0, b0, c0) - g) for g in gens) for _, op in cands]
ax[1].barh(range(len(names)), devs, color=["tab:green" if d < 1e-9 else "tab:gray" for d in devs])
ax[1].set_yticks(range(len(names)))
ax[1].set_yticklabels(fig_labels, fontsize=10)
ax[1].set_xlabel("与三枚生成元的最小偏差")
ax[1].set_title("局域性：只有 $\\eta_3$ 的折叠形式等于一枚生成元", fontsize=11)
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print("\n图已存 %s" % OUT)
