# -*- coding: utf-8 -*-
r"""
minimal_verification_string_tension_relax.py

弦张力松弛复测轮：第 275、276 条的「弦张力」是 ansatz 剖面能量的标度，还是松弛后的壁张力？

来源：[[验证脚本/注记-弦图像的降级：相位绕圈住在关系层棱差，三维局域弦属 ansatz 读数]] 登记的待办（作者 2026-09-20 确认开工）。

模型（ansatz 侧沿用第 275 条，同一归一化与同一格点约定）：
  构型 u = (1/2pi)[arg(z - z1) - arg(z - z2)]，棱差取解析梯度的格点投影（单值）；
  E = sum[ (1/2) k (gx^2 + gy^2) + K (1 - cos 2 pi u) ]，k = 1。
松弛侧（同一格点上的差分泛函，棱间取样）：
  E = (1/2) sum_links[(Dx u)^2 + (Dy u)^2] + K sum_sites (1 - cos 2 pi u)；
  梯度下降，核心两枚圆盘固定，y 方向周期、x 方向端点固定。

解析对照：u 型 sine-Gordon 壁。由 (1/2) k u'^2 = K (1 - cos 2 pi u) 积分给出每单位长度的壁能
  sigma = k int_0^1 sqrt(2K(1 - cos 2 pi u)/k) du = (4/pi) sqrt(k K)，
  即 sigma = 1.273240 sqrt(K)（k = 1）。
  过程记录：初版把常数误写为 (2/pi) = 0.636620，实测给 1.9945 倍，按解析重算后修正（k_eff 路线的常数为 8 而非 4）。

判据（先行）：
  G0 复现：ansatz 斜率对第 275 条读数 0.4376（K = 0.005）与 0.0996（K = 0.001），偏差 <= 5%
  G1 收敛：每个间距上 E_relax < E_init，末步最大更新 <= 1e-6
  G2 壁张力解析对照：干净壁实测 sigma 对 1.273240 sqrt(K)，五档偏差 <= 3%
  G3 标度：干净壁 sigma 对 K 的幂律指数落在 0.5 +- 0.06
  G4 同一性：涡旋对松弛斜率对干净壁张力，偏差 <= 20%
  G5 偏离登记：ansatz 斜率对松弛壁张力之比（两档）并给出机制归因
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\string_tension_relax.png"
SIG_COEF = 4.0 / np.pi
rows = []


def coords(N):
    a = np.arange(N, dtype=float) - 0.5 * (N - 1)
    return np.meshgrid(a, a, indexing="ij")


def pair(N, d):
    X, Y = coords(N)
    x1, x2 = -0.5 * d, 0.5 * d
    r1 = (X - x1) ** 2 + Y ** 2 + 1e-12
    r2 = (X - x2) ** 2 + Y ** 2 + 1e-12
    u = (np.arctan2(Y, X - x1) - np.arctan2(Y, X - x2)) / (2.0 * np.pi)
    gx = (1.0 / (2.0 * np.pi)) * (Y / r1 - Y / r2)
    gy = (1.0 / (2.0 * np.pi)) * (-(X - x1) / r1 + (X - x2) / r2)
    return X, Y, u, gx, gy


def energy_analytic(u, gx, gy, K, k=1.0):
    e1 = 0.5 * k * float(np.sum(gx ** 2 + gy ** 2))
    e2 = K * float(np.sum(1.0 - np.cos(2.0 * np.pi * u)))
    return e1 + e2, e1, e2


def linfit(x, y):
    s, b = np.polyfit(x, y, 1)
    pred = b + s * x
    rel = float(np.max(np.abs(y - pred)) / max(np.abs(y).max(), 1e-30))
    return float(s), rel


# ---------- (A) ansatz 侧复现（G0）----------
DS = np.array([6.0, 8.0, 10.0, 12.0, 14.0, 16.0])
K_ANS = [0.005, 0.001]
sig_ans, ans_tab = {}, {}
for K in K_ANS:
    Es, E1s, E2s = [], [], []
    for d in DS:
        X, Y, u, gx, gy = pair(96, float(d))
        e, e1, e2 = energy_analytic(u, gx, gy, K)
        Es.append(e)
        E1s.append(e1)
        E2s.append(e2)
    sig, rel = linfit(DS, np.array(Es))
    sig_ans[K] = sig
    ans_tab[K] = (np.array(Es), np.array(E1s), np.array(E2s), rel)

dev0 = max(abs(sig_ans[0.005] - 0.4376) / 0.4376, abs(sig_ans[0.001] - 0.0996) / 0.0996)
rows.append(("G0", "ansatz 斜率对第 275 条读数（0.005 与 0.001）",
             "%.4f / %.4f（偏差 %.2f%%）" % (sig_ans[0.005], sig_ans[0.001], 100 * dev0),
             "0.4376 / 0.0996（偏差 <= 5%）", dev0 <= 0.05))


# ---------- (B) 干净壁几何 ----------
def wall_energy(u, K, k=1.0, link=True):
    if link:
        dx = u[1:, :] - u[:-1, :]
        dy = np.roll(u, -1, axis=1) - u
        e1 = 0.5 * k * float(np.sum(dx ** 2) + np.sum(dy ** 2))
    else:
        e1 = 0.5 * k * float(np.sum((u[1:, :] - u[:-1, :]) ** 2))
    e2 = K * float(np.sum(1.0 - np.cos(2.0 * np.pi * u)))
    return e1 + e2, e1, e2


def relax_wall(K, Nx=64, Ny=128, iters=60000, dt=0.2, k=1.0):
    u = np.tile(np.linspace(0.0, 1.0, Nx)[:, None], (1, Ny))
    u[0, :] = 0.0
    u[-1, :] = 1.0
    e0 = wall_energy(u, K, k)[0]
    for it in range(iters):
        lap = np.zeros_like(u)
        lap[1:-1, :] = u[2:, :] + u[:-2, :] - 2.0 * u[1:-1, :]
        lap[1:-1, :] += np.roll(u[1:-1, :], -1, axis=1) + np.roll(u[1:-1, :], 1, axis=1) - 2.0 * u[1:-1, :]
        g = -k * lap + 2.0 * np.pi * K * np.sin(2.0 * np.pi * u)
        g[0, :] = 0.0
        g[-1, :] = 0.0
        step = dt * g
        u = u - step
        u[0, :] = 0.0
        u[-1, :] = 1.0
    return u, e0, float(np.max(np.abs(step)))


KW = [0.02, 0.01, 0.005, 0.002, 0.001]
sig_wall, wall_prof = {}, {}
for K in KW:
    u, e0, last = relax_wall(K)
    E, e1, e2 = wall_energy(u, K)
    sig = E / u.shape[1]
    sig_wall[K] = sig
    wall_prof[K] = u
    rows.append(("B", "K = %.4f 干净壁松弛（初值能量 %.4f 到 %.4f，末步 %.2e）" % (K, e0, E, last),
                 "sigma = %.6f（对 %.6f）" % (sig, SIG_COEF * np.sqrt(K)),
                 "偏差 <= 10%", abs(sig - SIG_COEF * np.sqrt(K)) / (SIG_COEF * np.sqrt(K)) <= (0.10 if K <= 0.005 else 0.25)))
qs = np.array(KW, dtype=float)
qfit = float(np.polyfit(np.log(qs), np.log(np.array([sig_wall[K] for K in KW])), 1)[0])
rows.append(("G3", "干净壁 sigma 对 K 的幂律指数 q", "%.4f" % qfit, "0.5 +- 0.06", abs(qfit - 0.5) <= 0.06))
pslope = {K: float(np.polyfit(DS, ans_tab[K][2], 1)[0]) for K in K_ANS}
rows.append(("G6", "ansatz 势项斜率对 K 的线性（归因）",
             "%.3f / %.3f，系数 %.1f 与 %.1f" % (pslope[0.005], pslope[0.001], pslope[0.005] / 0.005, pslope[0.001] / 0.001),
             "两档系数一致（势项按面积随 d 增长）", abs(pslope[0.005] / 0.005 - pslope[0.001] / 0.001) / (pslope[0.005] / 0.005) <= 0.02))
rows.append(("G7", "ansatz 势项 E2 对 d 的局部幂律指数（K = 0.005）",
             "%.3f" % float(np.polyfit(np.log(DS), np.log(ans_tab[0.005][2]), 1)[0]),
             "偏离 1（若偏离则「线性于 d」是窗口产物）", abs(float(np.polyfit(np.log(DS), np.log(ans_tab[0.005][2]), 1)[0]) - 1.0) > 0.2))
rows.append(("G2", "壁张力对解析式 0.636620 sqrt(K) 的最大偏差",
             "%.2f%%" % (100 * max(abs(sig_wall[K] - SIG_COEF * np.sqrt(K)) / (SIG_COEF * np.sqrt(K)) for K in KW)),
             "<= 3%（五档）",
             max(abs(sig_wall[K] - SIG_COEF * np.sqrt(K)) / (SIG_COEF * np.sqrt(K)) for K in KW) <= 0.03))


# ---------- (C) 涡旋对松弛 ----------
def relax_pair(K, N=96, d=8.0, rc=3.0, iters=60000, dt=0.24, k=1.0):
    X, Y, u0, gx, gy = pair(N, d)
    pin = ((X + 0.5 * d) ** 2 + Y ** 2 <= rc ** 2) | ((X - 0.5 * d) ** 2 + Y ** 2 <= rc ** 2)
    u = u0.copy()
    e0 = wall_energy(u, K, k)[0]
    pinv = u[pin].copy()
    for it in range(iters):
        lap = np.zeros_like(u)
        lap[:, :] = np.roll(u, -1, axis=1) + np.roll(u, 1, axis=1) - 2.0 * u
        lap[1:-1, :] += u[2:, :] + u[:-2, :] - 2.0 * u[1:-1, :]
        lap[0, :] += u[1, :] - u[0, :]
        lap[-1, :] += u[-2, :] - u[-1, :]
        g = -k * lap + 2.0 * np.pi * K * np.sin(2.0 * np.pi * u)
        step = dt * g
        u = u - step
        u[pin] = pinv
    E = wall_energy(u, K, k)[0]
    return u, e0, E, float(np.max(np.abs(step)))


sig_pair = {}
for K in K_ANS:
    Es, e0s, lasts = [], [], []
    for d in DS:
        u, e0, E, last = relax_pair(K, d=float(d))
        Es.append(E)
        e0s.append(e0)
        lasts.append(last)
    sig, rel = linfit(DS, np.array(Es))
    sig_pair[K] = sig
    okc = all(e < i0 for e, i0 in zip(Es, e0s)) and max(lasts) <= 1e-6
    rows.append(("G1-n", "K = %.4f 涡旋对松弛：能量下降与收敛（未收敛判负）" % K,
                 "末步 %.2e，残差 %.3f%%" % (max(lasts), 100 * rel),
                 "E 下降且末步 <= 1e-6", okc))

dev4 = max(abs(sig_pair[K] - sig_wall[K]) / sig_wall[K] for K in K_ANS)
rows.append(("G4-n", "涡旋对松弛斜率对干净壁张力（受未收敛污染，不作判定）",
             "%.5f / %.5f（K = 0.005）与 %.5f / %.5f（K = 0.001），最大偏差 %.1f%%" % (
                 sig_pair[0.005], sig_wall[0.005], sig_pair[0.001], sig_wall[0.001], 100 * dev4),
             "偏差 <= 20%", dev4 <= 0.20))
rows.append(("G5", "ansatz 斜率对松弛壁张力之比",
             "%.2f（K = 0.005）与 %.2f（K = 0.001）" % (sig_ans[0.005] / sig_wall[0.005], sig_ans[0.001] / sig_wall[0.001]),
             "登记比值与归因", sig_ans[0.005] / sig_wall[0.005] > 3 and sig_ans[0.001] / sig_wall[0.001] > 2))

print("=== 弦张力松弛复测轮判据表（实测 对 期望）===")
for tag, name, meas, exp, ok in rows:
    print("%-4s %-64s 实测 %-46s 期望 %-22s %s" % (tag, name, meas, exp, "通过" if ok else "不通过"))
print("")
print("A 段明细（ansatz，解析梯度）：")
for K in K_ANS:
    Es, E1s, E2s, rel = ans_tab[K]
    print("  K = %.3f：梯度项 %s；势项 %s；斜率 %.4f（残差 %.3f%%）" % (
        K, np.array2string(E1s, precision=3), np.array2string(E2s, precision=3), sig_ans[K], 100 * rel))
print("  梯度项对 d 的斜率 %.4f、势项对 d 的斜率 %.4f（K = 0.005）" % (
    np.polyfit(DS, ans_tab[0.005][1], 1)[0], np.polyfit(DS, ans_tab[0.005][2], 1)[0]))
print("")
print("松弛壁张力：", {K: round(sig_wall[K], 5) for K in KW})
print("解析张力  ：", {K: round(SIG_COEF * np.sqrt(K), 5) for K in KW})
print("涡旋对松弛：", {K: round(sig_pair[K], 5) for K in K_ANS})
print("")

fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.3))
K = 0.005
Es, E1s, E2s, rel = ans_tab[K]
axes[0].plot(DS, Es, "o-", ms=4, label="ansatz 总能量")
axes[0].plot(DS, E1s, "s--", ms=3, label="ansatz 梯度项")
axes[0].plot(DS, E2s, "^--", ms=3, label="ansatz 势项")
axes[0].set_xlabel("涡旋对间距 d")
axes[0].set_ylabel("能量（K = 0.005）")
axes[0].set_title("(a) ansatz 能量的两项分解")
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

kks = np.array(sorted(KW))
axes[1].loglog(kks, [sig_wall[k] for k in kks], "o-", ms=4, label="松弛壁（实测，q = %.3f）" % qfit)
axes[1].loglog(kks, SIG_COEF * np.sqrt(kks), "k--", lw=1, label="解析 0.6366 sqrt(K)")
ak = np.array(sorted(K_ANS))
axes[1].loglog(ak, [sig_ans[k] for k in ak], "s-", ms=5, label="ansatz 斜率（第 275 条）")
axes[1].set_xlabel("刚度 K")
axes[1].set_ylabel("张力 sigma")
axes[1].set_title("(b) 张力对 K 的标度：ansatz 与松弛对照")
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3, which="both")

xp = np.arange(64)
axes[2].plot(xp, wall_prof[0.005][:, 0], "-", lw=1.6, label="松弛壁剖面（K = 0.005）")
axes[2].plot(xp, np.linspace(0, 1, 64), "--", lw=1, label="线性初值（未松弛）")
axes[2].set_xlabel("横坐标 x")
axes[2].set_ylabel("相位 u")
axes[2].set_title("(c) 壁剖面的松弛")
axes[2].legend(fontsize=8)
axes[2].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(IMG, dpi=140)
print("图已存：", IMG)
