# -*- coding: utf-8 -*-
"""判据第 653 号：质量通道首轮：复结构符号畴壁能否给出手征选择的费米零模。

模型：一维格点，两分量 Dirac 算符 H = -i sigma_2 (x) D + sigma_3 (x) diag(m)，
其中 D = (S - S^dag)/2 为格点导数（开链或周期环）。手征算符取 gamma_5 = sigma_1 (x) I。
质量项取符号翻转的畴壁剖面（体系的复结构符号场 sigma 的 Z_2 畴壁）：
  m(n) = Delta * prod_{i=1..w} tanh((n - c_i)/xi)，c_i = N i/(w+1)，共 w 个壁。
该剖面的端点符号为 m(-inf) = Delta (-1)^w、m(+inf) = Delta，故连续指数 = (1-(-1)^w)/2。

先立的期望（写实现之前）：
  冒烟1  H 厄米（偏差 0）。
  冒烟2  常数质量：零模计数 = 0，最低 |本征值| = |Delta|（谱开隙）。
  G1    单壁（w = 1）：零模计数 = 2（机零），手征谱为 [-1, +1]，密度峰落在壁位。
        连续层单壁只给 1 个零模，故格点读到的 2 是味加倍。
  G2    壁数扫描：计数跟踪指数的两倍，w = 1、3 给 2 个，w = 2 给 0 个
        （指数为零；成对的 kink-antikink 模式被劈裂，劈裂随 xi 增大而减小）。
  G3    手征分辨 n+ 与 n- 恒相等，净手征性为零。
  反例  常数质量与无符号翻转的非平凡剖面：零模计数均为 0。
  读数  壁宽扫描：零模存在、质心落在壁位、密度宽度随 xi 单调增。
  简并  w = 1 的零模子空间二重简并，读数前须过简并检测与四段检验（EP1 至 EP4）。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

N = 64
DELTA = 1.0
XI = 0.5
TOL = 1e-8

S2 = np.array([[0.0, -1.0j], [1.0j, 0.0]])
S3 = np.array([[1.0, 0.0], [0.0, -1.0]])
S1 = np.array([[0.0, 1.0], [1.0, 0.0]])
G5 = np.kron(S1, np.eye(N))


def deriv(periodic):
    S = np.zeros((N, N))
    for i in range(N - 1):
        S[i, i + 1] = 1.0
    if periodic:
        S[N - 1, 0] = 1.0
    return (S - S.T) / 2.0


def hamiltonian(m, periodic):
    D = deriv(periodic)
    return np.kron(-1j * S2, D) + np.kron(S3, np.diag(m))


def prod_walls(w, xi=XI, delta=DELTA):
    n = np.arange(N)
    m = np.ones(N)
    for i in range(1, w + 1):
        m = m * np.tanh((n - N * i / float(w + 1)) / xi)
    return delta * m


def analyse(H, tol=TOL):
    ev, V = np.linalg.eigh(H)
    aev = np.abs(ev)
    order = np.argsort(aev)
    k = int(np.sum(aev < tol))
    # 简并检测：在 |本征值| 上按 tol 聚类，报零簇重数与下一簇的首值
    nxt = aev[order][k] if k < len(aev) else float("nan")
    return k, V[:, order[:k]], aev[order][:8], nxt


def chir(B):
    return np.linalg.eigvalsh(B.conj().T @ G5 @ B) if B.shape[1] else np.array([])


def dens(B):
    d = (np.abs(B) ** 2).sum(axis=1)
    d = d[:N] + d[N:]
    return d / d.sum() if d.sum() > 0 else d


# ================= 冒烟 =================
Hc = hamiltonian(np.full(N, DELTA), True)
print("冒烟1  H 厄米偏差 = %.3e（期望 0）" % np.abs(Hc - Hc.conj().T).max())
kc0, _, low0, nxt0 = analyse(Hc)
Hc_chain = hamiltonian(np.full(N, DELTA), False)
print("冒烟2  常数质量（周期环）：零模计数 = %d（期望 0），最低 |本征值| = %.6f（期望 %.6f）；"
      "开链档因 k 量子化为 %.6f（略高于 |Delta|，同为开隙无零模）"
      % (kc0, low0[0], DELTA, analyse(Hc_chain)[2][0]))

# ================= 主读数 =================
print("\n%-26s %-8s %-14s %-20s %s"
      % ("构型", "零模数", "手征分辨", "gamma_5 子空间谱", "最低 4 个 |本征值|"))
rec = {}
for per, tag in ((False, "开链"), (True, "周期环")):
    for w in (1, 2, 3):
        m = prod_walls(w)
        k, B, low, nxt = analyse(hamiltonian(m, per))
        g = chir(B)
        npl = int(np.sum(g > 0.5))
        nmi = int(np.sum(g < -0.5))
        if w == 1:
            rec[tag] = (k, B, m)
        print("%-26s %-8d %-14s %-20s %s"
              % ("%s，%d 个壁" % (tag, w), k, "%d / %d" % (npl, nmi),
                 np.round(g, 6).tolist(), np.round(low[:4], 6).tolist()))

print("\n反例与恒真检验：")
print("    常数质量：零模计数 = %d（期望 0）" % kc0)
m_flat = DELTA * (1.0 + 0.5 * np.sin(2.0 * np.pi * np.arange(N) / float(N)))
print("    无符号翻转的剖面（范围 [%.2f, %.2f]）：零模计数 = %d（期望 0）"
      % (m_flat.min(), m_flat.max(), analyse(hamiltonian(m_flat, False))[0]))

print("\n壁数奇偶与劈裂（开链，读数层）：")
rec["split"] = []
for w in (2,):
    for xi in (0.5, 1.0, 2.0):
        k, _, low, nxt = analyse(hamiltonian(prod_walls(w, xi=xi), False))
        rec["split"].append((xi, low[0]))
        print("    w = %d，xi = %.1f：零模计数 = %d，最低 |本征值| = %.4f（劈裂随 xi 减小）"
              % (w, xi, k, low[0]))

print("\n壁宽扫描（单壁，开链）：")
rec["xi"] = []
n_ax = np.arange(N)
for xi in (0.5, 1.0, 2.0, 4.0):
    k, B, _, _ = analyse(hamiltonian(prod_walls(1, xi=xi), False))
    d = dens(B) if k else np.zeros(N)
    ctr = (d * n_ax).sum() / d.sum() if k else float("nan")
    wid = np.sqrt((d * (n_ax - ctr) ** 2).sum() / d.sum()) if k else float("nan")
    rec["xi"].append((xi, k, ctr, wid))
    print("    xi = %.1f：零模计数 = %d，质心 = %.3f（壁位 %d），密度半宽 = %.3f"
          % (xi, k, ctr, N // 2, wid))

# ================= 简并检测与四段检验 =================
k1, B1, m1 = rec["开链"]
print("\n简并检测与四段检验（单壁，二重零模子空间）：")
print("    简并检测：零簇重数 = %d（期望 2）" % k1)
H1 = hamiltonian(m1, False)
EP1 = np.abs(B1.conj().T @ B1 - np.eye(k1)).max()
print("    EP1 子空间正交性 |B^dag B - I| = %.3e（期望 0）" % EP1)
EP2 = np.abs(B1.conj().T @ H1 @ B1).max()
print("    EP2 提取正确性：子空间内算符残差 = %.3e（期望 0）" % EP2)
rng = np.random.default_rng(20261016)
vals = []
for _ in range(400):
    v = rng.normal(size=k1) + 1j * rng.normal(size=k1)
    v = v / np.linalg.norm(v)
    wv = B1 @ v
    vals.append(np.real(wv.conj() @ G5 @ wv))
vals = np.array(vals)
print("    EP3 污染量化：子空间内随机组合矢的 gamma_5 期望极差 = %.6f（半宽 %.6f）"
      % (vals.max() - vals.min(), (vals.max() - vals.min()) / 2.0))
anti = np.abs(G5 @ H1 + H1 @ G5).max()
com = np.abs(G5 @ H1 - H1 @ G5).max()
if anti < 1e-10:
    lab = "反对易：E != 0 态上期望恒零，E = 0 档不受此约束，故核上读数合法"
elif com < 1e-10:
    lab = "对易：守恒标签，读数合法"
else:
    lab = "两者皆非：无守恒性，读数须另议"
print("    EP4 守恒标签自检：{gamma_5, H} = %.3e，[gamma_5, H] = %.3e，判为 %s"
      % (anti, com, lab))
print("    子空间谱分类：gamma_5 在零模子空间上的谱 = %s（各取 +-1，即两模手征相反）"
      % np.round(chir(B1), 6).tolist())

# ================= 图 =================
fig, ax = plt.subplots(1, 3, figsize=(14.0, 4.2))
d1 = dens(B1)
ax[0].plot(n_ax, m1, "k-", lw=1.2, label="质量剖面 $m(n)$")
ax0b = ax[0].twinx()
ax0b.plot(n_ax, d1, "tab:blue", lw=1.8, label="零模密度")
ax0b.set_ylabel("零模密度（归一）")
ax[0].set_xlabel("格点 $n$")
ax[0].set_ylabel("质量 $m(n)$")
ax[0].set_title("(a) 单壁：零模落在壁位", fontsize=10)
ax[0].legend(fontsize=7, loc="upper left")
ax0b.legend(fontsize=7, loc="upper right")
ax0b.grid(alpha=0.3)

for w, col in ((1, "tab:blue"), (2, "tab:orange"), (3, "tab:green")):
    _, _, low, _ = analyse(hamiltonian(prod_walls(w), False))
    ax[1].semilogy(range(len(low)), np.maximum(low, 1e-17), "o-", color=col,
                   label="%d 个壁" % w)
ax[1].axhline(TOL, color="k", ls=":", lw=1.0, label="零模阈值 $10^{-8}$")
ax[1].set_xlabel("序号")
ax[1].set_ylabel("|本征值|")
ax[1].set_title("(b) 计数跟踪指数：奇壁给两个，偶壁给零个", fontsize=10)
ax[1].legend(fontsize=8)
ax[1].grid(alpha=0.3, which="both")

xis = [r[0] for r in rec["split"]]
splits = [r[1] for r in rec["split"]]
ax[2].semilogy(xis, splits, "o-", color="tab:red", label="偶壁档的劈裂（读数层）")
ax[2].axhline(TOL, color="k", ls=":", lw=1.0, label="零模阈值 $10^{-8}$")
ax[2].set_xlabel("壁宽 $\\xi$")
ax[2].set_ylabel("最低 |本征值|")
ax[2].set_title("(c) 偶壁档：劈裂随壁宽减小", fontsize=10)
ax[2].legend(fontsize=8)
ax[2].grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(r"D:\software\TianXingFC\images\mass_channel_kink.png", dpi=150)
print("\n图已存 images/mass_channel_kink.png")
