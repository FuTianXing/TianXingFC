# -*- coding: utf-8 -*-
"""判据第 631 号（施工中）前置读数：格点奇偶供给的反交换、谱关系与生成元资格。

判据与先立的期望（详见注记）：
  S1 一维 {P, D} = 0（P = diag((-1)^n)，D 为中心差分）。期望 0。
  S2 一维交错与朴素逐位相同（一维 η_1 = 1），故一维不存在"去掉奇偶符"的反例。期望差 0。
  S3 二维交错与朴素的谱是否逐位相同（若交错是朴素的对角酉共轭则相同）。读数，不预设。
  S4 方向差算符的平方是否常值（Clifford 生成元资格）。期望非常值：D^2 本征值有散布。
  S5 反例：把 P 换成常数 1，{1, D} 不为零。期望非零（判据有区分力）。
另作零模计数读数（非判据）：单分量差分和算符的零模是否成点。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = r"D:\software\TianXingFC\images\staggered_parity_supply.png"
N1, N2 = 32, 16


def shift1d(n, seam=1.0):
    S = np.zeros((n, n))
    for i in range(n):
        S[i, (i + 1) % n] = seam
    return S


def ac_max(A, B):
    return np.abs(A @ B + B @ A).max()


# ---------- 一维 ----------
S = shift1d(N1)
D1 = (S - S.T) / 2.0                      # 中心差分（反厄米）
P1 = np.diag([(-1.0) ** n for n in range(N1)])
stag1 = D1.copy()                          # 一维 η_1 = 1，交错即朴素

s1 = ac_max(P1, D1)
s2 = np.abs(stag1 - D1).max()
s5 = ac_max(np.eye(N1), D1)
print("=== 一维（N=%d）===" % N1)
print("S1  {P,D} 最大模      = %.3e   （期望 0）" % s1)
print("S2  交错与朴素最大差  = %.3e   （期望 0）" % s2)
print("S5  反例 {1,D} 最大模 = %.3e   （期望非零）" % s5)
print("    D 的零模个数      = %d" % int(np.sum(np.abs(np.linalg.eigvals(D1)) < 1e-12)))

# ---------- 二维 ----------
Sx, Sy = shift1d(N2), shift1d(N2)
I = np.eye(N2)
S1 = np.kron(Sx, I)                        # 方向 1 移位（先指标 n1）
S2 = np.kron(I, Sy)
Dx = (S1 - S1.T) / 2.0
Dy = (S2 - S2.T) / 2.0
n1arr = np.repeat(np.arange(N2), N2)
eta2 = np.diag([(-1.0) ** a for a in n1arr])   # η_2(n) = (-1)^{n_1}
D_naive = Dx + Dy
D_stag = Dx + eta2 @ Dy

Px = np.diag([(-1.0) ** a for a in n1arr])
Py = np.diag([(-1.0) ** b for b in np.tile(np.arange(N2), N2)])

print("\n=== 二维（%d×%d）===" % (N2, N2))
for name, D in [("朴素", D_naive), ("交错", D_stag)]:
    print("  %s：{Px,Dx}=%.3e  {Py,Dy}=%.3e  {Px,D}=%.3e  {Py,D}=%.3e"
          % (name, ac_max(Px, Dx), ac_max(Py, Dy), ac_max(Px, D), ac_max(Py, D)))

ev_n = np.sort(np.abs(np.linalg.eigvals(D_naive)))
ev_s = np.sort(np.abs(np.linalg.eigvals(D_stag)))
s3 = np.abs(ev_n - ev_s).max()
print("S3  两谱模长最大逐位差 = %.3e   （读数：0 表示酉共轭同一谱）" % s3)

ev2 = np.linalg.eigvalsh(1j * D_naive)      # iD 厄米，本征值即符号
sq = ev2 ** 2
s4 = sq.max() - sq.min()
print("S4  D^2 本征值范围    = [%.6f, %.6f]，散布 %.3e（期望非常值）"
      % (sq.min(), sq.max(), s4))

tol = 1e-10
print("    朴素零模个数      = %d（单分量差分和算符）"
      % int(np.sum(np.abs(np.linalg.eigvals(D_naive)) < tol)))
print("    交错零模个数      = %d" % int(np.sum(np.abs(np.linalg.eigvals(D_stag)) < tol)))
print("    零模个数占格点比  = %.3f" % (np.sum(np.abs(np.linalg.eigvals(D_naive)) < tol) / (N2 * N2)))

# ---------- 图 ----------
fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.2))

ax[0].imshow(np.abs(P1 @ D1 + D1 @ P1), cmap="viridis", aspect="auto")
ax[0].set_title("(a) 一维 {P,D} 的模（全零则为反交换）")
ax[0].set_xlabel("列指标"); ax[0].set_ylabel("行指标")

ax[1].plot(np.abs(np.linalg.eigvals(D_naive)), "o", ms=3, label="朴素", alpha=0.7)
ax[1].plot(np.abs(np.linalg.eigvals(D_stag)), "x", ms=4, label="交错", alpha=0.7)
ax[1].set_title("(b) 二维两谱模长对照（最大逐位差 %.1e）" % s3)
ax[1].set_xlabel("本征值序号"); ax[1].set_ylabel(r"$|\lambda|$"); ax[1].legend()

ax[2].hist(sq, bins=40, color="tab:orange")
ax[2].axhline(sq.mean(), color="tab:red", ls="--", lw=1.2,
              label=r"均值 $%.3f$" % sq.mean())
ax[2].set_title(r"(c) $D^2$ 本征值分布（常值则应为一根柱）")
ax[2].set_xlabel(r"$D^2$ 本征值"); ax[2].set_ylabel("计数"); ax[2].legend()

for a in ax:
    a.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(OUT, dpi=170)
print("\n图已存：%s" % OUT)
