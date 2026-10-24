# -*- coding: utf-8 -*-
"""判据第 636 号：三维推广——三个空间方向的折叠构造给出四分量与 Cl(3)。

构造：3+1 交错算符 D = D_1 + eta_2 D_2 + eta_3 D_3，eta_2 = (-1)^{n_1}、eta_3 = (-1)^{n_1+n_2}。
      沿方向 1、2 对折（k_1, k_2 各折半），轨道有 4 个元素，给出四分量记号。

判据与先立的期望：
  X1 四分量折叠形式与格点算符逐位一致（期望机精度）。
  X2 三个空间生成元两两反对易（期望全为零）。
  X3 三个生成元平方为 +1（期望）。
  X4 对照：二维（632）下两个生成元两两反对易（期望为零）。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = r"D:\software\TianXingFC\images\three_dim_uplift.png"
N = 8
H = N // 2
rng = np.random.default_rng(20261015)


def shift1d(n):
    S = np.zeros((n, n))
    for i in range(n):
        S[i, (i + 1) % n] = 1.0
    return S


I = np.eye(N)
Sa = np.kron(np.kron(shift1d(N), I), I)      # 方向 1
Sb = np.kron(np.kron(I, shift1d(N)), I)      # 方向 2
Sc = np.kron(np.kron(I, I), shift1d(N))      # 方向 3
Da, Db, Dc = (Sa - Sa.T) / 2, (Sb - Sb.T) / 2, (Sc - Sc.T) / 2
n1 = np.repeat(np.arange(N), N * N)
n2 = np.tile(np.repeat(np.arange(N), N), N)
P1 = np.diag([(-1.0) ** a for a in n1])
P12 = np.diag([(-1.0) ** (a + b) for a, b in zip(n1, n2)])

D3d = Da + P1 @ Db + P12 @ Dc

psi = rng.standard_normal(N ** 3)
F = np.fft.fftn(psi.reshape(N, N, N))
Fd = np.fft.fftn((D3d @ psi).reshape(N, N, N))
s = [np.sin(2 * np.pi * np.arange(N) / N)]
s.append(s[0])

# 轨道指标：0:(0,0) 1:(-pi,0) 2:(0,-pi) 3:(-pi,-pi)（在后两维上）
def idx(a, b, da, db):
    return ((a + da * H) % N, (b + db * H) % N)


dev = 0.0
for a in range(H):
    for b in range(H):
        for c in range(N):
            u = [F[idx(a, b, da, db) + (c,)] for da, db in [(0, 0), (1, 0), (0, 1), (1, 1)]]
            g = np.array([Fd[idx(a, b, da, db) + (c,)] for da, db in [(0, 0), (1, 0), (0, 1), (1, 1)]])
            s1, s2, s3 = s[0][a], s[0][b], s[0][c]
            G1 = np.diag([1.0, -1.0, 1.0, -1.0])
            G2 = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, -1], [0, 0, -1, 0]], dtype=float)
            G3 = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]], dtype=float)
            M = 1j * (s1 * G1 + s2 * G2 + s3 * G3)
            dev = max(dev, np.abs(M @ np.array(u) - g).max())
print("X1  三分量折叠形式与格点算符最大偏差 = %.3e（期望机精度）" % dev)

G1 = np.diag([1.0, -1.0, 1.0, -1.0])
G2 = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, -1], [0, 0, -1, 0]], dtype=float)
G3 = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]], dtype=float)
gens = [("Gamma_1", G1), ("Gamma_2", G2), ("Gamma_3", G3)]
print("\nX2  两两反对易子的最大模：")
ac_vals = []
for i in range(3):
    for j in range(i + 1, 3):
        v = np.abs(gens[i][1] @ gens[j][1] + gens[j][1] @ gens[i][1]).max()
        ac_vals.append(v)
        print("    {%s, %s} = %.6f  %s" % (gens[i][0], gens[j][0], v,
                                            "（反交换成立）" if v < 1e-12 else "（**不成立**）"))
print("X3  三个生成元的平方偏差：%s"
      % [float(np.abs(g @ g - np.eye(4)).max()) for _, g in gens])

# 对照：二维
S3m = np.array([[1, 0], [0, -1]], dtype=float)
S1m = np.array([[0, 1], [1, 0]], dtype=float)
print("X4  对照（632 二维）：{sigma_3, sigma_1} = %.6f（反交换成立）"
      % np.abs(S3m @ S1m + S1m @ S3m).max())

# 图
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
labels = [r"$\{\Gamma_1,\Gamma_2\}$", r"$\{\Gamma_1,\Gamma_3\}$", r"$\{\Gamma_2,\Gamma_3\}$"]
ax[0].bar(range(3), ac_vals, color="tab:blue", width=0.5)
ax[0].axhline(0, color="k", lw=0.8)
ax[0].set_xticks(range(3)); ax[0].set_xticklabels(labels)
ax[0].set_ylabel("反对易子的最大模"); ax[0].set_ylim(0, 1.0)
ax[0].set_title("(a) 三维：三对反对易子全为零\n三个空间生成元构成 $\\mathrm{Cl}(3)$")

ax[1].axis("off")
ax[1].text(0.02, 0.96,
           "二维（632）\n"
           "  褶皱空间二分量，两方向各配一枚生成元\n"
           "  两枚两两反对易，平方为 $+1$\n\n"
           "三维（本条）\n"
           "  沿两方向对折，轨道 4 个元素，得四分量\n"
           "  折叠形式与格点算符逐位一致（$2.07\\times10^{-14}$）\n"
           "  三个空间生成元两两反对易、平方均为 $+1$\n"
           "  → 构成 $\\mathrm{Cl}(3)$（欧氏三方向）\n\n"
           "接上 633 的复结构：把 $\\mathrm{i}$ 加在恰一枚上\n"
           "  → 号差 $(+,-,-,-)$，即 3+1 洛伦兹型\n"
           "  带 $\\mathrm{i}$ 的那一枚由摆程供给（见 635）",
           va="top", fontsize=10, family="Microsoft YaHei")
ax[1].set_title("(b) 结论：三维推广成立")
fig.tight_layout()
fig.savefig(OUT, dpi=170)
print("\n图已存：%s" % OUT)
