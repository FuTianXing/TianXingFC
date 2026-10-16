# -*- coding: utf-8 -*-
"""判据第 652 号：缺陷与费米耦合首轮：绕数（电荷）通道能否给出手征选择。

模型：二维周期格点 N×N，交错算符 D = D1^U + P1 D2^U（第 632、650 号口径）。
耦合通道取体系的电荷通道：绕数 = 电荷，背景构型即体系的相位场绕数构型，
以 Peierls 相位加在跳跃上（连线变量 U_a(n) = exp(i(theta(n+e_a) - theta(n)))）。

三组构型：
  绕数构型  theta(n) = w * arg(n - c)，c 为缺陷芯所在格点（体系的满周闭合环绕数）。
  纯规范    theta 用非平凡单值函数 Lambda = 2 pi x y / N（有相位、无绕数）。
  均匀通量  Landau 规范 U1 = 1、U2 = exp(i 2 pi w x / N^2)，总通量 2 pi w（手工放进的非平凡背景）。

先立的期望（写实现之前）：
  冒烟1  i*D 厄米（偏差 0）。
  冒烟2  w=0 档与无相位算符逐位一致（差 0），即耦合关掉时回到第 650 号的算符。
  冒烟3  纯规范构型：与自由算符的谱逐位一致（非平凡相位但零通量，不应改变物理）。
  冒烟4  绕数构型的连线变量是纯规范的：逐线恒等式 U_a(n) = g(n+e_a)/g(n)（g = e^{i theta}）
         成立，相位差最大模达 2 pi 量级（绕数住在未回卷的相位里，回卷后 e^{i 2 pi w} = 1）。
  G1    绕数构型下费米子谱与自由逐位一致（w = 1、2、3）：零模计数与手征分辨均不变，
         即守恒绕数在电荷通道上对费米子透明。
  G2    非平凡背景（均匀通量）确实改变零模计数，但净手征性恒为零（n+ = n-）。
  反例  两条合起来构成区分力检验：平凡背景（纯规范）读数不变，非平凡背景读数改变。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

N = 36
TOL = 1e-9
WMAX = 4

Sx = np.roll(np.eye(N), 1, axis=1)           # (S psi)_n = psi(n+e_a)，与第 650 号同向
Sy = np.roll(np.eye(N), 1, axis=1)
S1 = np.kron(Sx, np.eye(N))
S2 = np.kron(np.eye(N), Sy)

xarr = np.repeat(np.arange(N), N)            # 格点序 i = x*N + y
yarr = np.tile(np.arange(N), N)
P1 = np.diag((-1.0) ** xarr)
P2 = np.diag((-1.0) ** yarr)
EPS = P1 @ P2                                # 交错手征生成元 eps = (-1)^{x+y}
ONE = np.ones(N * N, dtype=complex)


def staggered(U1, U2):
    """把 U(1) 联络按 Peierls 相位加在跳跃上，返回交错算符。"""
    S1U = np.diag(U1) @ S1
    S2U = np.diag(U2) @ S2
    D1 = (S1U - S1U.conj().T) / 2.0
    D2 = (S2U - S2U.conj().T) / 2.0
    return D1 + P1 @ D2


def links_from_site_phase(theta):
    """由格点相位场造连线变量：U_a(n) = exp(i(theta(n+e_a) - theta(n)))。"""
    th = theta.reshape(N, N)
    d1 = np.roll(th, -1, axis=0) - th          # 沿 x 的相位差
    d2 = np.roll(th, -1, axis=1) - th          # 沿 y 的相位差
    return np.exp(1j * d1).reshape(-1), np.exp(1j * d2).reshape(-1), d1.reshape(-1), d2.reshape(-1)


def winding_theta(w, xc, yc):
    """体系原生绕数构型：theta(n) = w * arg(n - c)，割线沿 45 度对角线。"""
    c45, s45 = np.cos(np.pi / 4.0), np.sin(np.pi / 4.0)
    vx = (xarr - xc).astype(float)
    vy = (yarr - yc).astype(float)
    vx[np.abs(vx) < 1e-12] = 1e-12
    vp = c45 * vx + s45 * vy
    vq = -s45 * vx + c45 * vy
    return w * np.arctan2(vq, vp)


def uniform_phases(w):
    """均匀通量（Landau 规范）：总通量 2 pi w。"""
    return ONE.copy(), np.exp(1j * 2.0 * np.pi * w * xarr / float(N * N))


def plaquette_flux(a1, a2):
    """格面通量（原始值，不取模）：F(x,y) = a1(x,y) + a2(x+1,y) - a1(x,y+1) - a2(x,y)。"""
    A1 = a1.reshape(N, N)
    A2 = a2.reshape(N, N)
    return (A1 + np.roll(A2, -1, axis=0) - np.roll(A1, -1, axis=1) - A2).reshape(-1)



def count_flux(Ns, w, tol=TOL):
    """在 Ns 尺寸上算均匀通量的零模计数（读数层，用于格点尺寸依赖核对）。"""
    Sx = np.roll(np.eye(Ns), 1, axis=1)
    S1 = np.kron(Sx, np.eye(Ns))
    S2 = np.kron(np.eye(Ns), Sx)
    xa = np.repeat(np.arange(Ns), Ns)
    P1s = np.diag((-1.0) ** xa)
    A = np.diag(np.exp(1j * 2.0 * np.pi * w * xa / float(Ns * Ns))) @ S2
    D = (S1 - S1.T) / 2.0 + P1s @ ((A - A.conj().T) / 2.0)
    sv = np.linalg.svd(D, compute_uv=False)
    return int(np.sum(sv < tol))

def null_info(M, tol=TOL):
    u, sv, _ = np.linalg.svd(M)
    k = int(np.sum(sv < tol))
    if k == 0:
        return 0, np.array([]), sv
    ev = np.linalg.eigvalsh(u[:, :k].conj().T @ EPS @ u[:, :k])
    return k, ev, sv


def small(sv, n=6):
    return np.array2string(np.sort(sv)[:n], precision=3, floatmode="fixed")


Dfree = staggered(ONE, ONE)
sv_free = np.linalg.svd(Dfree, compute_uv=False)

# ================= 冒烟 =================
print("冒烟1  i*D 厄米偏差 = %.3e（期望 0）" % np.abs(1j * Dfree - (1j * Dfree).conj().T).max())
u1z, u2z = uniform_phases(0)
print("冒烟2  w=0 档与无相位算符逐位差 = %.3e（期望 0）" % np.abs(staggered(u1z, u2z) - Dfree).max())

xg = np.arange(N)[:, None] * np.ones((1, N))
yg = np.ones((N, 1)) * np.arange(N)[None, :]
Lam = (0.9 * np.sin(2.0 * np.pi * xg / float(N))
       * np.cos(2.0 * np.pi * yg / float(N))).reshape(-1)
U1g, U2g, _, _ = links_from_site_phase(Lam)
Dg = staggered(U1g, U2g)
Gd = np.diag(np.exp(1j * Lam))
print("冒烟3  纯规范构型：与自由算符谱最大差 = %.3e，规范协变恒等式逐位差 = %.3e（期望均 0）"
      % (np.abs(np.linalg.svd(Dg, compute_uv=False) - sv_free).max(),
         np.abs(Dg - Gd.conj().T @ Dfree @ Gd).max()))

th1 = winding_theta(1, N // 2, N // 2)
U1w, U2w, d1w, d2w = links_from_site_phase(th1)
g1 = np.exp(1j * th1)
i1 = np.arange(N * N)
j1 = (i1 + N) % (N * N)                       # n + e1
j2 = (i1 // N) * N + (i1 % N + 1) % N         # n + e2
print("冒烟4  绕数构型（w=1）：纯规范逐线恒等式 U_a(n) = g(n+e_a)/g(n) 的最大偏差 = %.3e / %.3e"
      "（期望 0），相位差最大模 = %.6f" 
      % (np.abs(U1w - g1[j1] / g1[i1]).max(), np.abs(U2w - g1[j2] / g1[i1]).max(),
         max(np.abs(d1w).max(), np.abs(d2w).max())))

# ================= G1 绕数构型 =================
print("\nG1  绕数构型（体系原生）：费米子谱与自由逐位比较")
for w in (1, 2, 3):
    th = winding_theta(w, N // 2, N // 2)
    u1, u2, a1, a2 = links_from_site_phase(th)
    M = staggered(u1, u2)
    k, ev, sv = null_info(M)
    npl = int(np.sum(ev > 0.5)) if k else 0
    nmi = int(np.sum(ev < -0.5)) if k else 0
    print("    w = %d：谱最大差 = %.3e（期望 0），n0 = %d，手征分辨 %d / %d，最小奇异值 %s"
          % (w, np.abs(sv - sv_free).max(), k, npl, nmi, small(sv)))

# ================= G2 非平凡背景 =================
print("\nG2  非平凡背景（均匀通量）：零模计数与手征分辨")
print("%-4s %-8s %-12s %-10s %s" % ("w", "n0", "n+/n-", "净手征性", "最小的 6 个奇异值"))
rec = []
for w in range(0, WMAX + 1):
    u1, u2 = uniform_phases(w)
    k, ev, sv = null_info(staggered(u1, u2))
    npl = int(np.sum(ev > 0.5)) if k else 0
    nmi = int(np.sum(ev < -0.5)) if k else 0
    rec.append((w, k, npl, nmi))
    print("%-4d %-8d %-12s %-10d %s" % (w, k, "%d / %d" % (npl, nmi), npl - nmi, small(sv)))

# ================= 读数层：计数律的格点尺寸依赖 =================
print("")
print("读数  均匀通量下零模计数的格点尺寸依赖（判定该形态是结构性质还是分辨率伪影）")
for Ns in (20, 24, 30):
    print("      N = %2d：" % Ns,
          "  ".join("w=%d n0=%d" % (w, count_flux(Ns, w)) for w in range(0, WMAX + 1)))
print("      N = %2d（主扫描）：" % N,
      "  ".join("w=%d n0=%d" % (r[0], r[1]) for r in rec))

# ================= 对照 =================
print("\n对照  连续朗道能级（解析）：总通量 2 pi w 下最低朗道能级简并 = w、手征由磁场符号定")
print("      即同一守恒荷在连续层给得出净手征性；格点层读数恒为零，差值即加倍抵消的部分")

# ================= 图 =================
fig, ax = plt.subplots(1, 3, figsize=(14.0, 4.2))
ws = list(range(0, WMAX + 1))

diffs = []
for w in (1, 2, 3):
    th = winding_theta(w, N // 2, N // 2)
    u1, u2, _, _ = links_from_site_phase(th)
    diffs.append(np.abs(np.linalg.svd(staggered(u1, u2), compute_uv=False) - sv_free).max())
ax[0].bar([1, 2, 3], diffs, color="tab:blue", width=0.5)
ax[0].axhline(1e-12, color="tab:red", ls="--", lw=1.0, label="机精度线 $10^{-12}$")
ax[0].set_xlabel("绕数 $w$")
ax[0].set_ylabel("与自由算符谱的最大差")
ax[0].set_title("(a) 绕数构型对费米子透明", fontsize=10)
ax[0].legend(fontsize=8)
ax[0].grid(alpha=0.3)

pos = [r[2] for r in rec]
neg = [r[3] for r in rec]
ax[1].bar([x - 0.16 for x in ws], pos, width=0.3, color="tab:blue", label="$n_+$")
ax[1].bar([x + 0.16 for x in ws], [-v for v in neg], width=0.3, color="tab:orange", label="$n_-$")
ax[1].axhline(0, color="k", lw=0.8)
ax[1].set_xlabel("绕数（通量量子数）$w$")
ax[1].set_ylabel("手征分辨计数（上 $n_+$，下 $n_-$）")
ax[1].set_title("(b) 非平凡背景下手征恒成对", fontsize=10)
ax[1].legend(fontsize=8)
ax[1].grid(alpha=0.3)

lows = []
for w in ws:
    u1, u2 = uniform_phases(w)
    sv = np.linalg.svd(staggered(u1, u2), compute_uv=False)
    lows.append(max(np.sort(sv)[0], 1e-17))
ax[2].semilogy(ws, lows, "o-", color="tab:green", label="均匀通量")
ax[2].semilogy(ws, [max(np.sort(sv_free)[0], 1e-17)] * len(ws), "s--", color="tab:gray",
               label="绕数构型（= 自由）")
ax[2].axhline(1e-9, color="k", ls=":", lw=1.0, label="零模判据阈值 $10^{-9}$")
ax[2].set_xlabel("绕数 $w$")
ax[2].set_ylabel("最小奇异值")
ax[2].set_title("(c) 零模是否存在：最低奇异值", fontsize=10)
ax[2].legend(fontsize=8)
ax[2].grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(r"D:\software\TianXingFC\images\winding_zero_mode.png", dpi=150)
print("\n图已存 images/winding_zero_mode.png")
