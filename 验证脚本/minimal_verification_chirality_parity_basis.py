# -*- coding: utf-8 -*-
"""判据第 637 号：手性算符的格点原生表达式与两套标号的相容性（狄拉克线阶段 5 收束轮）。

构造：三维交错算符 D = D_1 + eta_2 D_2 + eta_3 D_3，沿方向 1、2 对折得四分量（与 636 同）。
      折叠基次序 (da, db) = (0,0), (1,0), (0,1), (1,1)，位置指标 p = 2 db + da。
      张量约定：A ⊗ B 中 A 作用在 db（方向 2 的折叠指标）上，B 作用在 da（方向 1）上。

先立的判据与期望（Y2 的期望在首轮写错，见文末过程记录）：
  Y1 手性算符 Gamma_5 = i Gamma_1 Gamma_2 Gamma_3 厄米、平方为 +1、谱为 ±1 各二重。
  Y2 Gamma_5 与三枚空间生成元**对易**（三枚之积是奇维体积元，与每枚对易；标准物理里
     gamma_5 = i alpha_1 alpha_2 alpha_3 与 alpha_i 对易是同一关系）；并穷举 Cl(3) 的十六个
     群元素，确认其中没有与三枚全反对易的厄米对合，故四枚不构成 Cl(4)。
  Y3 位置奇偶 P_1 = (-1)^{n_1}、P_2 = (-1)^{n_2} 的折叠形式为 da、db 的交换（与格点算符逐位一致），
     且 [P_1, P_2] = 0，给出位置奇偶网格。
  Y4 两套标号的对表：{P_1, Gamma_5} = {P_2, Gamma_5} = 0（反对易），故任一位置奇偶本征态上手性
     期望为零，且它在两枚手性支上的权重各为 1/2（行和判据，非逐格判据）；子空间内随机矢亦然。
  Y5 反例（恒真检验）：把 Gamma_5 换成与 P_1 对易的 P_2，重叠退化为 1 与 0，判据可区分。
  Y6 逐位对照：Gamma_5 与由格点数据装配的折叠矩阵对易（不用符号形式）。
  Y7 读数：四分量空间内与三枚生成元全反对易的厄米矩阵的维数，以及其中是否存在对合；
     若不存在，则第四枚生成元（质量/时间方向）须由外部供给，与 635 一致。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = r"D:\software\TianXingFC\images\chirality_parity_basis.png"
N = 8
H = N // 2
rng = np.random.default_rng(20261015)
ORB = [(0, 0), (1, 0), (0, 1), (1, 1)]


def shift1d(n):
    S = np.zeros((n, n))
    for i in range(n):
        S[i, (i + 1) % n] = 1.0
    return S


I3 = np.eye(N)
Sa = np.kron(np.kron(shift1d(N), I3), I3)
Sb = np.kron(np.kron(I3, shift1d(N)), I3)
Sc = np.kron(np.kron(I3, I3), shift1d(N))
Da, Db, Dc = (Sa - Sa.T) / 2, (Sb - Sb.T) / 2, (Sc - Sc.T) / 2
n1 = np.repeat(np.arange(N), N * N)
n2 = np.tile(np.repeat(np.arange(N), N), N)
P1 = np.diag([(-1.0) ** a for a in n1])
P2 = np.diag([(-1.0) ** b for b in n2])
P12 = np.diag([(-1.0) ** (a + b) for a, b in zip(n1, n2)])
D3d = Da + P1 @ Db + P12 @ Dc

I4 = np.eye(4)
G1 = np.diag([1.0, -1.0, 1.0, -1.0])
G2 = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, -1], [0, 0, -1, 0]], dtype=float)
G3 = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]], dtype=float)
G5 = 1j * (G1 @ G2 @ G3)

S1m = np.array([[0, 1], [1, 0]], dtype=float)
S2m = np.array([[0, -1j], [1j, 0]], dtype=complex)
S3m = np.array([[1, 0], [0, -1]], dtype=float)


def fold_vector(v):
    F = np.fft.fftn(np.asarray(v).reshape(N, N, N))
    out = {}
    for a in range(H):
        for b in range(H):
            for c in range(N):
                out[(a, b, c)] = np.array([F[(a + da * H) % N, (b + db * H) % N, c]
                                           for da, db in ORB])
    return out


def folded_matrix(op, a, b, c):
    """由格点数据装配算符在该轨道上的 4×4 折叠矩阵（四个分量探针各作一列）。"""
    cols = []
    for da, db in ORB:
        ind = np.zeros((N, N, N), dtype=complex)
        ind[(a + da * H) % N, (b + db * H) % N, c] = 1.0
        psi = np.fft.ifftn(ind).reshape(-1)
        g = fold_vector(op @ psi)
        cols.append(g[(a, b, c)])
    return np.array(cols).T


# ---------------- Y1 手性算符的原生表达式与谱 ----------------
print("Y1  Gamma_5 = i Gamma_1 Gamma_2 Gamma_3")
print("    厄米性偏差 = %.3e（期望 0）" % np.abs(G5 - G5.conj().T).max())
print("    平方偏差   = %.3e（期望 0）" % np.abs(G5 @ G5 - I4).max())
ev5 = np.sort(np.linalg.eigvalsh((G5 + G5.conj().T) / 2).real)
print("    谱 = %s（期望 -1 二重、+1 二重）" % np.round(ev5, 12).tolist())
print("    与 -(sigma_2 ⊗ sigma_3) 的偏差 = %.3e（期望 0，给出原生矩阵形式）"
      % np.abs(G5 + np.kron(S2m, S3m)).max())
print("    与 i·(三枚生成元之积) 的偏差 = %.3e（期望 0，定义式自检）"
      % np.abs(G5 - 1j * (G1 @ G2 @ G3)).max())

# ---------------- Y2 对易关系与 Cl(3) 内的穷举 ----------------
gens = [("Gamma_1", G1), ("Gamma_2", G2), ("Gamma_3", G3)]
print("\nY2  Gamma_5 与三枚空间生成元：")
ac = []
for nm, g in gens:
    v = np.abs(G5 @ g + g @ G5).max()
    cm = np.abs(G5 @ g - g @ G5).max()
    ac.append(v)
    print("    {%s, Gamma_5} = %.3e，[%s, Gamma_5] = %.3e" % (nm, v, nm, cm))
print("    三对反对易子最大模 = %.3e（期望 2，即对易：Gamma_5 是三枚之积乘复结构，属中心元）"
      % max(ac))

# Cl(3) 的十六个群元素：子集乘积乘 ±1
def cl_group():
    out = []
    for mask in range(8):
        M = np.eye(4, dtype=complex)
        for i in range(3):
            if mask >> i & 1:
                M = M @ gens[i][1]
        for s in (1.0, -1.0):
            out.append(s * M)
    return out


herm_inv_anti = []
for M in cl_group():
    if np.abs(M - M.conj().T).max() < 1e-12 and np.abs(M @ M - I4).max() < 1e-12:
        if all(np.abs(M @ g + g @ M).max() < 1e-12 for _, g in gens):
            herm_inv_anti.append(M)
print("    Cl(3) 十六个群元素中「厄米、平方为 I、与三枚全反对易」者：%d 枚（期望 0，故四枚不成 Cl(4)）"
      % len(herm_inv_anti))

# ---------------- Y3 位置奇偶的折叠形式 ----------------
Pf1 = np.kron(np.eye(2), S1m)          # da 的交换
Pf2 = np.kron(S1m, np.eye(2))          # db 的交换
dev_p1 = dev_p2 = 0.0
for a in range(H):
    for b in range(H):
        for c in range(N):
            dev_p1 = max(dev_p1, np.abs(folded_matrix(P1, a, b, c) - Pf1).max())
            dev_p2 = max(dev_p2, np.abs(folded_matrix(P2, a, b, c) - Pf2).max())
print("\nY3  位置奇偶折叠形式与格点算符的逐位偏差：P_1 = %.3e，P_2 = %.3e（期望 0）"
      % (dev_p1, dev_p2))
print("    [P_1, P_2] 最大模 = %.3e（期望 0，两枚奇偶可同时对角化，给出位置奇偶网格）"
      % np.abs(Pf1 @ Pf2 - Pf2 @ Pf1).max())

# ---------------- Y4 两套标号的对表 ----------------
# 位置奇偶的折叠形式是交换，故本征基为组合矢：u± = (1, ±1)/√2
up = np.array([1.0, 1.0]) / np.sqrt(2)
um = np.array([1.0, -1.0]) / np.sqrt(2)
gridvecs = np.array([np.kron(a, b) for a in (up, um) for b in (up, um)]).T
# 列序：(P_2, P_1) = (+,+), (+,-), (-,+), (-,-)
w, V = np.linalg.eigh((G5 + G5.conj().T) / 2)
Vp, Vm = V[:, w > 0], V[:, w < 0]
Bp, Bm = gridvecs[:, :2], gridvecs[:, 2:]


def overlap(A, B):
    return np.abs(A.conj().T @ B) ** 2


Ov = np.block([[overlap(Bp, Vp), overlap(Bp, Vm)],
               [overlap(Bm, Vp), overlap(Bm, Vm)]])
print("\nY4  位置奇偶网格基与手性本征基的重叠模方（4×4）：")
print(np.round(Ov.real, 12))
rowsum = np.array([[Ov[i, :2].sum(), Ov[i, 2:].sum()] for i in range(4)])
print("    每行在两枚手性支上的权重（期望全为 1/2）：")
print(np.round(rowsum.real, 12))
print("    与 1/2 的最大偏差 = %.3e（期望 0）" % np.abs(rowsum.real - 0.5).max())

# 手性在两枚位置奇偶的子空间上的期望（子空间内随机矢，简并子空间读数合法性）
worst = 0.0
for nm, op in (("P_1", Pf1), ("P_2", Pf2)):
    wo, Vo = np.linalg.eigh((op + op.conj().T) / 2)
    for cols in (Vo[:, wo > 0], Vo[:, wo < 0]):
        for _ in range(20):
            c = rng.standard_normal(2) + 1j * rng.standard_normal(2)
            v = cols @ c
            v = v / np.linalg.norm(v)
            worst = max(worst, abs(complex(v.conj() @ G5 @ v)))
print("    位置奇偶子空间内随机矢上手性期望的最大模 = %.3e（期望 0，两套标号互不相容）" % worst)
print("    {P_1, Gamma_5} = %.3e，{P_2, Gamma_5} = %.3e（期望 0，即反对易）"
      % (np.abs(Pf1 @ G5 + G5 @ Pf1).max(), np.abs(Pf2 @ G5 + G5 @ Pf2).max()))
print("    [P_1, P_2] = %.3e（对照：网格内部相容）" % np.abs(Pf1 @ Pf2 - Pf2 @ Pf1).max())
G5g = gridvecs.T @ G5 @ gridvecs
print("    网格基下 |Gamma_5| 的块结构（期望反对角）：")
print(np.round(np.abs(G5g), 12))

a0, b0, c0 = 1, 1, 1
Mf = folded_matrix(D3d, a0, b0, c0)
Hf = (1j * Mf + (1j * Mf).conj().T) / 2
wh, Vh = np.linalg.eigh(Hf)
OvE = np.block([[overlap(gridvecs[:, :2], Vh[:, wh > 0]), overlap(gridvecs[:, :2], Vh[:, wh < 0])],
                [overlap(gridvecs[:, 2:], Vh[:, wh > 0]), overlap(gridvecs[:, 2:], Vh[:, wh < 0])]])
print("\nY4b 读数（非判据）：采样轨道 (a,b,c)=(1,1,1) 的能量符号投影与位置奇偶网格的重叠模方：")
print(np.round(OvE.real, 6))

# ---------------- Y5 反例（恒真检验） ----------------
dev_genuine = np.abs(rowsum.real - 0.5).max()
wc, Vc = np.linalg.eigh((Pf2 + Pf2.conj().T) / 2)
Ovc = np.block([[overlap(gridvecs[:, :2], Vc[:, wc > 0]), overlap(gridvecs[:, :2], Vc[:, wc < 0])],
                [overlap(gridvecs[:, 2:], Vc[:, wc > 0]), overlap(gridvecs[:, 2:], Vc[:, wc < 0])]])
rowsumc = np.array([[Ovc[i, :2].sum(), Ovc[i, 2:].sum()] for i in range(4)])
ce_worst = np.abs(rowsumc.real - 0.5).max()
print("\nY5  反例：把 Gamma_5 换成与 P_1 对易的 P_2（其本征基即网格基）")
print("    行和与 1/2 的最大偏差 = %.3f（期望 0.5，判据可区分）" % ce_worst)

# ---------------- Y6 逐位对照：折叠矩阵与 Gamma_5 对易 ----------------
dev6 = 0.0
for a in range(H):
    for b in range(H):
        for c in range(N):
            Mf2 = folded_matrix(D3d, a, b, c)
            dev6 = max(dev6, np.abs(G5 @ Mf2 - Mf2 @ G5).max())
print("\nY6  [Gamma_5, 格点折叠矩阵] 的最大模 = %.3e（期望 0，逐位对照）" % dev6)

# ---------------- Y7 读数：第四枚反对易生成元 ----------------
basis = []
for i in range(4):
    E = np.zeros((4, 4), dtype=complex)
    E[i, i] = 1.0
    basis.append(E)
for i in range(4):
    for j in range(i + 1, 4):
        E = np.zeros((4, 4), dtype=complex)
        E[i, j] = E[j, i] = 1.0
        basis.append(E)
        F = np.zeros((4, 4), dtype=complex)
        F[i, j] = 1j
        F[j, i] = -1j
        basis.append(F)
nrm2 = [np.trace(B.conj().T @ B).real for B in basis]


def expand(H):
    return np.array([np.trace(B.conj().T @ H).real / n2 for B, n2 in zip(basis, nrm2)])


rows = []
for _, g in gens:
    for B in basis:
        rows.append(expand(B @ g + g @ B))
Amat = np.array(rows)
sv = np.linalg.svd(Amat, compute_uv=False)
ker = np.linalg.svd(Amat)[2][np.sum(sv > 1e-10):]
kermats = [sum(ker[k, i] * basis[i] for i in range(len(basis))) for k in range(ker.shape[0])]
print("\nY7  16 维厄米基中，与三枚生成元全反对易的实维数 = %d" % ker.shape[0])
found = None
for M in kermats:
    Mn = M / np.linalg.norm(M) * 2.0
    if np.abs(Mn @ Mn - I4).max() < 1e-8:
        found = Mn
        break
if found is not None:
    print("    取到一例厄米对合 X：与三枚的反对易子最大模 = %.3e，X^2 偏差 = %.3e，"
          "与 Gamma_5 的反对易子 = %.3e"
          % (max(np.abs(found @ g + g @ found).max() for _, g in gens),
             np.abs(found @ found - I4).max(), np.abs(found @ G5 + G5 @ found).max()))
    if len(kermats) == 2:
        ac = kermats[0] @ kermats[1] + kermats[1] @ kermats[0]
        print("    两枚核元素之间的反对易子最大模 = %.3e（为零则对合族为一维，格点不选出其中之一）"
              % np.abs(ac).max())
    print("    该 X 不在 Cl(3) 的十六个群元素之内（见 Y2 的穷举），故第四枚生成元在空间内存在而不由格点选出")
else:
    print("    其中无厄米对合：第四枚反对易生成元不在四分量空间内")

# ---------------- 图 ----------------
fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.3))
im = ax[0].imshow(np.abs(G5g), cmap="viridis", vmin=0, vmax=1)
ax[0].set_title("手性算符在位置奇偶网格基下\n（反对角块：同时翻转两个奇偶）")
ax[0].set_xticks(range(4))
ax[0].set_xticklabels(["(+,+)", "(+,-)", "(-,+)", "(-,-)"], fontsize=8)
ax[0].set_yticks(range(4))
ax[0].set_yticklabels(["(+,+)", "(+,-)", "(-,+)", "(-,-)"], fontsize=8)
fig.colorbar(im, ax=ax[0], fraction=0.046)

im2 = ax[1].imshow(Ov.real, cmap="magma", vmin=0, vmax=1)
ax[1].set_title("两套标号的重叠模方\n（每格 1/4，行和 1/2）")
ax[1].set_xticks(range(4))
ax[1].set_xticklabels(["手性 +", "手性 +", "手性 -", "手性 -"], fontsize=8)
ax[1].set_yticks(range(4))
ax[1].set_yticklabels(["(+,+)", "(+,-)", "(-,+)", "(-,-)"], fontsize=8)
fig.colorbar(im2, ax=ax[1], fraction=0.046)

ax[2].bar([0, 1], [dev_genuine, ce_worst], color=["tab:blue", "tab:red"], width=0.5)
ax[2].axhline(0, color="k", lw=0.8)
ax[2].set_xticks([0, 1])
ax[2].set_xticklabels(["原生手性算符", "反例（对易替身）"], fontsize=9)
ax[2].set_ylabel("行和与 1/2 的最大偏差")
ax[2].set_title("恒真检验：换掉算符即判据翻")
ax[2].set_ylim(0, 0.6)
for i, v in enumerate([dev_genuine, ce_worst]):
    ax[2].text(i, v + 0.02, "%.3f" % v, ha="center", fontsize=9)

fig.tight_layout()
fig.savefig(OUT, dpi=150)
print("\n图已存 %s" % OUT)
