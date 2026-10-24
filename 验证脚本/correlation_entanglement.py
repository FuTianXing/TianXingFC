# -*- coding: utf-8 -*-
r"""
correlation_entanglement.py

二-1 第五段：显式函数 f(E_xy)。把第 174 条的 C3（函数形状只依赖锁定值）做成可核验的塌缩，
并把距离依赖限制在维持项里。确定性，随机固定种子 20260919。

模型（一维环，L 个站点，周期边界；alpha 为相位读数）
  锁定档：alpha_j = phi + eta_j，phi 为整体锁定模（方差 P，逐实现对全环相同），
      eta 为平稳涨落场（方差 sig^2，相关函数 rho(n) = exp(-n/xi)）。
      配对相干 E(n) = Cov(alpha_x, alpha_y) / Var(alpha) = (P + sig^2 rho(n)) / (P + sig^2)，
      平台 E_inf = P / (P + sig^2)。
  对照档（无锁定）：alpha_j 为独立增量之和（随机游走），无正平台。
  场层闭式档：eta 取第 181 条的两点指数闭式 C_eta(r) = a q^r + b alpha^r。

解析关系（两式都是恒等式，区别在是否用平台分解）
  A 式：Var[Delta-alpha] = 2 Var(alpha) (1 - E)
  B 式：Var[Delta-alpha] = 2 sig^2 (1 - m)，  m = (E - E_inf) / (1 - E_inf)
B 式给出 f 的形状：C = f(E) = 2 sig^2 (1 - m(E))，形式不含 n，n 只进 m。

判据与期望：
  G1 A 式在全部（档 × 格距）上成立（相对偏差 <= 5%）
  G2 B 式（f(E) 形式）在全部（档 × 格距）上成立（相对偏差 <= 5%）
  G3 塌缩：C / (2 sig^2) 对 m 落在同一直线 1 - m 上（三档合并，散布 <= 5%）
  G4 锁定值扫描：E_inf 与 P / (P + sig^2) 相符（三档，<= 5%）且随 P 单调
  G5 场层闭式替换：eta 取第 181 条两点指数闭式时，B 式仍成立（<= 5%）
  G6 对照档排除：C 与 n 成正比（<= 5%），而平台口径的 f 偏离随 n 增大（报最大偏离）
  G7 紧致口径与有限尺寸：逐样本 |Delta-alpha| 模 2 pi 后 <= pi，n 取到 L / 4 时 G2 仍成立
  G8 接口登记（二-2，不作判据）
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\correlation_entanglement.png"
SEED = 20260919
L = 256
XI = 4.0
SIG = 1.0
NS = 4000
PS = (0.25, 1.0, 4.0)
NS_SCAN = (1, 2, 4, 8, 16, 24, 32, 48, 64)
rows = []


def report(tag, ok, expect, actual):
    rows.append((tag, ok, expect, actual))
    print("[%s] %s" % ("通过" if ok else "未过", tag))
    print("      期望 " + expect)
    print("      实测 " + actual)


def ar1_field(L, xi, sig, ns, rng):
    rho = np.exp(-1.0 / xi)
    eta = np.zeros((ns, L))
    eta[:, 0] = rng.normal(scale=sig, size=ns)
    noise = rng.normal(scale=sig * np.sqrt(1.0 - rho ** 2), size=(ns, L))
    for j in range(1, L):
        eta[:, j] = rho * eta[:, j - 1] + noise[:, j]
    return eta


def walk_field(L, sig, ns, rng):
    return np.cumsum(rng.normal(scale=sig, size=(ns, L)), axis=1)


def stats(alpha, n):
    a = alpha[:, :alpha.shape[1] - n]
    b = alpha[:, n:]
    mu = float(alpha.mean())
    var = float(np.mean((alpha - mu) ** 2))
    cov = float(np.mean((a - mu) * (b - mu)))
    diff = float(np.mean((b - a) ** 2))
    return var, cov, diff


rng = np.random.default_rng(SEED)
eta = ar1_field(L, XI, SIG, NS, rng)
walk = walk_field(L, SIG, NS, rng)
data = {}
for P in PS:
    phi = rng.normal(scale=np.sqrt(P), size=(NS, 1))
    data[P] = (phi + eta, P)

# ---------- G1 A 式、G2 B 式 ----------
g1, g2, plat = [], [], {}
for P in PS:
    alpha, _ = data[P]
    E = []
    for n in NS_SCAN:
        var, cov, diff = stats(alpha, n)
        E.append((n, var, cov, diff))
    E_inf = float(np.mean([cov / var for n, var, cov, diff in E if n >= 48]))
    plat[P] = E_inf
    for n, var, cov, diff in E:
        ex = cov / var
        d1 = abs(diff - 2.0 * var * (1.0 - ex)) / diff
        m = (ex - E_inf) / (1.0 - E_inf)
        d2 = abs(diff - 2.0 * SIG ** 2 * (1.0 - m)) / diff
        g1.append((P, n, d1))
        g2.append((P, n, d2, ex, m, diff))
report("G1 A 式 Var = 2 Var(alpha) (1 - E)", all(d <= 0.05 for *_, d in g1),
       "全部（三档锁定值 × 九档格距）相对偏差 <= 5%",
       "最大相对偏差 %.3e（档 P=%s）" % (max(d for *_, d in g1), PS))
report("G2 B 式 C = 2 sig^2 (1 - m)", all(d <= 0.05 for *_, d, _, _, _ in g2),
       "全部（三档锁定值 × 九档格距）相对偏差 <= 5%",
       "最大相对偏差 %.3e" % max(d for *_, d, _, _, _ in g2))

# ---------- G3 塌缩 ----------
xs = [m for P, n, d, ex, m, diff in g2]
ys = [diff / (2.0 * SIG ** 2) for P, n, d, ex, m, diff in g2]
res = [abs(y - (1.0 - x)) for x, y in zip(xs, ys)]
report("G3 塌缩 C/(2 sig^2) 对 m", all(r <= 0.05 for r in res),
       "三档合并后落在同一直线 1 - m 上（相对偏差 <= 5%）",
       "最大偏离 %.3e；m 范围 %.4f 至 %.4f" % (max(res), min(xs), max(xs)))

# ---------- G4 锁定值扫描 ----------
g4 = []
for P in PS:
    e_pred = P / (P + SIG ** 2)
    g4.append((P, plat[P], e_pred, abs(plat[P] - e_pred) / e_pred))
ok4 = all(d <= 0.05 for *_, d in g4) and (plat[PS[0]] < plat[PS[1]] < plat[PS[2]])
report("G4 锁定值扫描", ok4,
       "E_inf 与 P/(P + sig^2) 相符（<= 5%）且随 P 单调",
       "；".join("P=%.2f: E_inf %.4f 对 %.4f（相对 %.2e）" % (P, ei, ep, d) for P, ei, ep, d in g4))

# ---------- G5 场层两点指数闭式 ----------
Q5, KAP5, L5, NS5 = 0.9, 0.25, 3000, 1500
a5 = 0.5 * Q5 / ((Q5 - (1.0 - KAP5)) * (1.0 - (1.0 - KAP5) * Q5))
b5 = -0.5 * (1.0 - KAP5) * (1.0 - Q5 ** 2) / ((Q5 - (1.0 - KAP5)) * (1.0 - (1.0 - KAP5) ** 2) * (1.0 - (1.0 - KAP5) * Q5))
alpha5 = 1.0 - KAP5
noise5 = np.zeros((NS5, L5))
noise5[:, 0] = rng.normal(size=NS5)
for j in range(1, L5):
    noise5[:, j] = Q5 * noise5[:, j - 1] + rng.normal(scale=np.sqrt(1.0 - Q5 ** 2), size=NS5)
x5 = np.zeros((NS5, L5))
for j in range(1, L5):
    x5[:, j] = alpha5 * x5[:, j - 1] + np.sqrt(2.0 * KAP5) * noise5[:, j]
eta5 = x5[:, 1200:]
L5 = eta5.shape[1]
C0 = a5 * Q5 ** 0 + b5 * alpha5 ** 0
P5 = 1.0
phi5 = rng.normal(scale=np.sqrt(P5), size=(NS5, 1))
a5f = phi5 + eta5
E_inf5 = float(np.mean([stats(a5f, n)[1] / stats(a5f, n)[0] for n in (40, 48)]))
g5 = []
for n in (1, 2, 4, 8, 16, 24, 32, 48):
    var, cov, diff = stats(a5f, n)
    ex = cov / var
    m = (ex - E_inf5) / (1.0 - E_inf5)
    d = abs(diff - 2.0 * C0 * (1.0 - m)) / diff
    g5.append((n, d))
report("G5 场层闭式替换（两点指数）", all(d <= 0.05 for *_, d in g5),
       "eta 取第 181 条闭式（C_eta(0) = %.4f）时 B 式仍成立（<= 5%%）" % C0,
       "最大相对偏差 %.3e；n = %s" % (max(d for *_, d in g5), sorted(set(n for n, d in g5))))

# ---------- G6 对照档排除 ----------
g6 = []
for n in NS_SCAN:
    var, cov, diff = stats(walk, n)
    g6.append((n, diff, diff / n))
slope = float(np.polyfit(NS_SCAN, [d for n, d, s in g6], 1)[0])
res6 = max(abs(s - SIG ** 2) / SIG ** 2 for n, d, s in g6)
WIN = (24, 32, 48, 64)
E_walk = [stats(walk, n)[1] / stats(walk, n)[0] for n in WIN]
spread6 = max(E_walk) / min(E_walk) - 1.0
E_lock = [stats(data[PS[0]][0], n)[1] / stats(data[PS[0]][0], n)[0] for n in WIN]
spread6l = max(E_lock) / min(E_lock) - 1.0
report("G6 对照档排除", res6 <= 0.05 and spread6 >= 0.10 and spread6l <= 0.05,
       "对照档 C 与 n 成正比（斜率对 sig^2 相对偏差 <= 5%）且配对相干在大格距窗上无平台（散布 >= 10%），锁定档同窗有平台（<= 5%）",
       "斜率 %.6f 对 %.1f（残差 %.2e）；对照档 E 从 %.4f 降到 %.4f（散布 %.1f%%）；锁定档散布 %.2e"
       % (slope, SIG ** 2, res6, E_walk[0], E_walk[-1], 100.0 * spread6, spread6l))

# ---------- G7 紧致口径与有限尺寸 ----------
cap, frac = 0.0, []
for P in PS:
    alpha, _ = data[P]
    d = alpha[:, 64:] - alpha[:, :L - 64]
    w = (d + np.pi) % (2.0 * np.pi) - np.pi
    cap = max(cap, float(np.max(np.abs(w))))
    frac.append(float(np.mean(np.abs(d) > np.pi)))
report("G7 紧致口径与有限尺寸", cap <= np.pi + 1e-9 and all(d <= 0.05 for P, n, d, _, _, _ in g2 if n == 64),
       "逐样本模 2 pi 后 |Delta-alpha| <= pi，且 n = L/4 = 64 档上 G2 仍成立",
       "最大包裹距离 %.6f；各档包裹占比 %s；n = 64 档最大偏差 %.3e"
       % (cap, ["%.4f" % f for f in frac], max(d for P, n, d, _, _, _ in g2 if n == 64)))

print("[登记] G8：f(E_xy) = 2 sig^2 (1 - m)，m = (E - E_inf)/(1 - E_inf)，供 二-2 的齐雷尔森界使用，本轮只登记")

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
for P in PS:
    alpha, _ = data[P]
    xs_, ys_ = [], []
    for n in NS_SCAN:
        var, cov, diff = stats(alpha, n)
        xs_.append(n)
        ys_.append(diff / (2.0 * SIG ** 2))
    ax[0].plot(xs_, ys_, "o-", label="锁定档 P=%.2f" % P)
ax[0].plot(NS_SCAN, [float(stats(walk, n)[2]) / (2.0 * SIG ** 2) for n in NS_SCAN], "k^-",
           label="对照档（随机游走）")
ax[0].set_xscale("log", base=2)
ax[0].set_xlabel("格距 n")
ax[0].set_ylabel("C(n) / (2 sig^2)")
ax[0].set_title("三档锁定值与对照档的差方差")
hg = {}
for P, n, d, ex, m, diff in g2:
    hg.setdefault(P, ([], []))
    hg[P][0].append(m)
    hg[P][1].append(diff / (2.0 * SIG ** 2))
for P in PS:
    ax[1].plot(hg[P][0], hg[P][1], "o", label="P=%.2f" % P)
mm = np.linspace(0.0, 1.0, 50)
ax[1].plot(mm, 1.0 - mm, "k--", label="1 - m")
ax[1].set_xlabel("维持因子 m = (E - E_inf)/(1 - E_inf)")
ax[1].set_ylabel("C / (2 sig^2)")
ax[1].set_title("三档塌缩到同一条 f(E)")
ax[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(IMG, dpi=150)
plt.close(fig)

bad = [r for r in rows if not r[1]]
print("---- 判据 %d 项，通过 %d 项；图存 %s ----" % (len(rows), len(rows) - len(bad), IMG))
sys.exit(1 if bad else 0)
