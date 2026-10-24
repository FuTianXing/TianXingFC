# -*- coding: utf-8 -*-
"""判据第 659 号：质量通道非紧致边界复核。

背景：第 653 号判质量通道给不出净手征性（单壁一对零模，净手征性为零），
其开链档读数与周期环档相同，但 N=64 的硬截断不等于严格非紧致，
且周期环上 Z_2 符号必须绕回原值这一拓扑约束可能被误当成体系必然。
层数通道（物理空间生成方向）非紧致，故本轮以开放边界、大 N 扫描加
有限尺寸外推复核：单壁零模的成对性是算符产物还是边界产物。

模型：一维开放链，两分量 Dirac 算符
  H = -i sigma_2 (x) D + sigma_3 (x) diag(m)，D = (S - S^dag)/2，S 为上移算符。
手征算符 gamma_5 = sigma_1 (x) I。质量剖面取单壁 m(n) = Delta tanh((n - N/2)/xi)。

先立的期望（写实现之前）：
  冒烟1  H 厄米（偏差 0）。
  冒烟2  常数质量：零模计数 = 0，最低 |lambda| 趋近 |Delta|（谱开隙，无隙内边缘态）。
  冒烟3  质量剖面端点：m(0) < 0 且 m(N-1) > 0（符号确实翻转），随 N 稳定。
  K1     单壁、N 扫描（64 至 512）：零模计数恒为 2，两个零模为机零。
   K1b   第三个 |lambda| 不随 N 缩小（排除有限尺寸把非零模压进阈值）。
  K2     零模密度峰落在壁位；链端各 10% 区域内的密度占比小于 5%（非边缘态）。
  K3     gamma_5 子空间谱恒为 [-1,+1]，手征分辨 1/1，净手征性 0。
  K4     壁宽扫描（xi = 0.5, 1, 2, 4）：零模计数恒为 2。
  反例1  常数质量：0 个零模。
  反例2  无符号翻转的非平凡剖面（m 恒正、有起伏）：0 个零模。
  反例3  半壁（m 从 0 到 +Delta，只抬不翻）：0 个零模（判据对完整翻转敏感）。
  对照   连续层单壁解析给 1 个零模（格点给的 2 是味加倍，k=0 与 k=pi 两个味）。
  简并   K1 的零模子空间二重简并，读数前过简并检测与四段检验（EP1 至 EP4）。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

DELTA = 1.0
TOL = 1e-8
NS = (64, 128, 256, 512)
S1 = np.array([[0.0, 1.0], [1.0, 0.0]])
S2 = np.array([[0.0, -1.0j], [1.0j, 0.0]])
S3 = np.array([[1.0, 0.0], [0.0, -1.0]])


def deriv_open(N):
    S = np.zeros((N, N))
    for i in range(N - 1):
        S[i, i + 1] = 1.0
    return (S - S.T) / 2.0


def ham(m, N):
    return np.kron(-1j * S2, deriv_open(N)) + np.kron(S3, np.diag(m))


def wall(N, xi=0.5, delta=DELTA):
    n = np.arange(N)
    return delta * np.tanh((n - N / 2.0) / xi)


def half_wall(N, xi=0.5, delta=DELTA):
    n = np.arange(N)
    return delta * 0.5 * (1.0 + np.tanh((n - N / 2.0) / xi))


def analyse(H, tol=TOL):
    ev, V = np.linalg.eigh(H)
    aev = np.abs(ev)
    order = np.argsort(aev)
    k = int(np.sum(aev < tol))
    return k, V[:, order[:k]], aev[order[:6]]


def dens(B, N):
    d = (np.abs(B) ** 2).sum(axis=1)
    d = d[:N] + d[N:]
    return d / d.sum() if d.sum() > 0 else d


def chir(B, N):
    G5 = np.kron(S1, np.eye(N))
    return np.linalg.eigvalsh(B.conj().T @ G5 @ B) if B.shape[1] else np.array([])


# ================= 冒烟 =================
print("冒烟1  H 厄米偏差 = %.3e（期望 0）"
      % np.abs(ham(np.full(64, DELTA), 64) - ham(np.full(64, DELTA), 64).conj().T).max())

print("\n冒烟2  常数质量开放链：谱开隙与边缘态检查")
for N in NS:
    k, _, low = analyse(ham(np.full(N, DELTA), N))
    print("    N = %4d：零模计数 = %d（期望 0），最低 |lambda| = %.6f（期望趋近 %.6f）"
          % (N, k, low[0], DELTA))

print("\n冒烟3  单壁剖面端点符号（须翻号）：")
for N in NS:
    m = wall(N)
    print("    N = %4d：m(0) = %+.6f，m(N-1) = %+.6f（期望一负一正）"
          % (N, m[0], m[-1]))

# ================= K1 / K1b / K2 / K3 =================
print("\nK1 至 K3  单壁、N 扫描（开放链）")
print("%-8s %-8s %-14s %-22s %-12s %s"
      % ("N", "零模数", "手征分辨", "gamma_5 子空间谱", "端区占比", "最低 4 个 |lambda|"))
rec = {}
for N in NS:
    m = wall(N)
    k, B, low = analyse(ham(m, N))
    g = chir(B, N)
    npl = int(np.sum(g > 0.5)); nmi = int(np.sum(g < -0.5))
    d = dens(B, N) if k else np.zeros(N)
    edge = 0.1
    e = int(edge * N)
    frac = (d[:e].sum() + d[-e:].sum()) if k else float("nan")
    rec[N] = (k, B, m, d, g, low)
    print("%-8d %-8d %-14s %-22s %-12s %s"
          % (N, k, "%d / %d" % (npl, nmi), np.round(g, 6).tolist(),
             "%.4f" % frac, np.round(low[:4], 6).tolist()))

print("\nK1b  第三个 |lambda| 随 N 的行为（排除有限尺寸压模）：")
for N in NS:
    low = rec[N][5]
    print("    N = %4d：第三个 |lambda| = %.6f" % (N, low[2]))

print("\nK4  壁宽扫描（单壁，开放链 N = 256）：")
for xi in (0.5, 1.0, 2.0, 4.0):
    N = 256
    k, B, low = analyse(ham(wall(N, xi=xi), N))
    g = chir(B, N)
    print("    xi = %.1f：零模计数 = %d（期望 2），gamma_5 谱 = %s"
          % (xi, k, np.round(g, 6).tolist()))

# ================= 反例 =================
print("\n反例：")
k0, _, _ = analyse(ham(np.full(256, DELTA), 256))
print("    反例1 常数质量：零模计数 = %d（期望 0）" % k0)
N = 256
n = np.arange(N)
m_flat = DELTA * (1.0 + 0.5 * np.sin(2.0 * np.pi * n / float(N)))
print("    反例2 无翻转的非平凡剖面（范围 [%.2f, %.2f]）：零模计数 = %d（期望 0）"
      % (m_flat.min(), m_flat.max(), analyse(ham(m_flat, N))[0]))
print("    反例3 半壁（m 从 0 抬到 +Delta，无翻号）：零模计数 = %d（期望 0）"
      % analyse(ham(half_wall(N), N))[0])

# ================= 简并检测与四段检验 =================
N = 256
k1, B1, m1, d1, g1, low1 = rec[N]
H1 = ham(m1, N)
G5 = np.kron(S1, np.eye(N))
print("\n简并检测与四段检验（单壁 N = 256，二重零模子空间）：")
print("    简并检测：零簇重数 = %d（期望 2）" % k1)
print("    EP1 子空间正交性 |B^dag B - I| = %.3e（期望 0）"
      % np.abs(B1.conj().T @ B1 - np.eye(k1)).max())
print("    EP2 提取正确性：子空间内算符残差 = %.3e（期望 0）"
      % np.abs(B1.conj().T @ H1 @ B1).max())
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
anti = np.abs(G5 @ H1 + H1 @ G5).max(); com = np.abs(G5 @ H1 - H1 @ G5).max()
lab = ("反对易：E != 0 态上期望恒零，E = 0 档不受此约束，故核上读数合法"
       if anti < 1e-10 else ("对易：守恒标签，读数合法" if com < 1e-10 else "两者皆非：无守恒性"))
print("    EP4 守恒标签自检：{gamma_5, H} = %.3e，[gamma_5, H] = %.3e，判为 %s"
      % (anti, com, lab))
print("    子空间谱分类：gamma_5 在零模子空间上的谱 = %s（各取 +-1，两模手征相反）"
      % np.round(chir(B1, N), 6).tolist())

# ================= 图 =================
fig, ax = plt.subplots(1, 3, figsize=(14.0, 4.2))
nn = np.arange(N)
ax[0].plot(nn, m1, "k-", lw=1.2, label="质量剖面 $m(n)$")
ax0b = ax[0].twinx()
ax0b.plot(nn, d1, "tab:blue", lw=1.8, label="零模密度")
ax0b.set_ylabel("零模密度（归一）")
ax[0].set_xlabel("格点 $n$")
ax[0].set_ylabel("质量 $m(n)$")
ax[0].set_title("(a) 开放链单壁：零模落在壁位", fontsize=10)
ax[0].legend(fontsize=7, loc="upper left")
ax0b.legend(fontsize=7, loc="upper right")
ax0b.grid(alpha=0.3)

for i, Nn in enumerate(NS):
    low = rec[Nn][5]
    ax[1].semilogy(range(len(low)), np.maximum(low, 1e-17), "o-", label="N = %d" % Nn)
ax[1].axhline(TOL, color="k", ls=":", lw=1.0, label="零模阈值 $10^{-8}$")
ax[1].set_xlabel("序号")
ax[1].set_ylabel("|本征值|")
ax[1].set_title("(b) N 扫描：两个机零零模不随尺寸变", fontsize=10)
ax[1].legend(fontsize=8)
ax[1].grid(alpha=0.3, which="both")

fracs = []
for Nn in NS:
    d = rec[Nn][3]
    e = int(0.1 * Nn)
    fracs.append(d[:e].sum() + d[-e:].sum())
ax[2].plot(NS, fracs, "o-", color="tab:red")
ax[2].axhline(0.05, color="k", ls=":", lw=1.0, label="5% 参考线")
ax[2].set_xlabel("链长 $N$")
ax[2].set_ylabel("链端 10% 区域的密度占比")
ax[2].set_title("(c) 零模不落在链端（非边缘态）", fontsize=10)
ax[2].legend(fontsize=8)
ax[2].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(r"D:\software\TianXingFC\images\mass_channel_open_boundary.png", dpi=150)
print("\n图已存 images/mass_channel_open_boundary.png")
