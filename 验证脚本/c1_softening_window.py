# -*- coding: utf-8 -*-
r"""
c1_softening_window.py

[C1 负线项来源判据，L 系列续判] 软化反馈的强度上限与所需驻留半径的定量对接
（判据 U-1 至 U-3）

背景：L-1 至 L-4（feedback_interface.py）给出软化分支
的界面优惠幅度（-2.6% 至 -8.2%）与刚度下限（0.1 kappa），并把「这份幅度能否
让环构型线项 a 越零、把稳定半径 R* 推进落窗线」留给 C1 侧的窗口算术。本判据
把三件事定量对接：软化的结构上限、越零所需折扣、落窗所需折扣。

模型（与 L 系列同口径）：一维链 N = 600，E = sum_e (1/2) kappa_e (du_e - 1)^2
+ sum_i K (1 - cos 2 pi u_i)，kappa = 1；站点记录 theta_i = u_i - x_i，
链接记录 s_e = (theta_i + theta_{i+1}) / 2，kappa_e = clip(kappa (1 + c s_e),
floor * kappa)。自洽求解：弛豫 4000 步与记录更新交替 6 轮。记录在位移侧的
平台值为 s = 1，故刚度下限首次触及的条件是 |c| = 1 - floor。

环构型算术（C1-4b 口径）：E(R) = 2 pi R a + pi R^2 delta，稳定半径
R* = |a| / delta（需 a < 0 < delta）。线项取最有利加法合成：
a = sigma_0 (eps - eta)，eps 为锚核钉扎后的残差比例（W1/W2 档 0.100，
C1-4c 构造档 0.256），eta 为软化折扣比例。落窗要求 R* >= R*_req，等价于
eta >= eps + R*_req * delta / sigma_0。

判据
U-1 上限标定：细扫 c 从 0 至 -1.2（含 -0.75、-0.85 加密点），给 eta(c) 上确
    界 eta_max、首次触及刚度下限的 c_cap 与结构预测 c_floor = 1 - floor 核对；
    另测下限敏感性（floor = 0.05、0.02 的前沿读数）。
U-2 越零对接：越零条件 eta > eps；输出实测上限对 eps 的裕量或缺口倍数，
    并给最有利总节省率（锚核残差 + 软化折扣）。
U-3 落窗对接：eta_req = eps + R*_req * delta / sigma_0，R*_req 取 1、2、3.61
    （落窗线 2 与洛伦兹窗口 3.61 取库内口径）；delta / sigma_0 取连续参照与
    库内离散读数两档；输出缺口倍数与结论。

GL 标度（构造轮模型 lambda = 0.25，V(phi) = (lambda / 2) (phi^2 - 1)^2）：
delta = V(0) - V(1) = lambda / 2；sigma_0 连续参照 (4 / 3) sqrt(lambda)，
另取库内离散读数 0.198（C1-4c S1），并用本脚本离散复算一列作对照。

运行：python 验证脚本/c1_softening_window.py
确定性、无随机数；终端输出 + 图 images/c1_softening_window.png；运行约 15 秒。
"""
import numpy as np

N, KAP, ETA_STEP, STEPS_OUT, STEPS_IN = 600, 1.0, 0.005, 6, 4000
CS = [0.0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.75, -0.8, -0.85,
      -0.9, -1.0, -1.1, -1.2]
FLOOR = 0.1
LAM = 0.25
DELTA_GL = LAM / 2.0
SIGMA_CONT = (4.0 / 3.0) * np.sqrt(LAM)
SIGMA_DISC_LIB = 0.198
EPS_LIST = [("W1/W2 档（残差 10%）", 0.100), ("C1-4c 构造档（残差 25.6%）", 0.256)]
R_REQ = [1.0, 2.0, 3.61]


def relax(u, kappa_e, K, steps=STEPS_IN, eta=ETA_STEP):
    for _ in range(steps):
        du = np.diff(u)
        g = np.zeros(N)
        g[:-1] -= kappa_e * (du - 1.0)
        g[1:] += kappa_e * (du - 1.0)
        g += K * 2.0 * np.pi * np.sin(2.0 * np.pi * u)
        g[0] = g[-1] = 0.0
        u = u - eta * g
        u[0], u[-1] = 0.0, float(N)
    return u


def energy(u, kappa_e, K):
    du = np.diff(u)
    return float(0.5 * (kappa_e * (du - 1.0) ** 2).sum()
                 + K * (1.0 - np.cos(2.0 * np.pi * u)).sum())


def width(u):
    du = np.diff(u)
    dens = 0.5 * (du - 1.0) ** 2
    idx = np.arange(N - 1) + 0.5
    c0 = (dens * idx).sum() / dens.sum()
    return 2.0 * np.sqrt((dens * (idx - c0) ** 2).sum() / dens.sum())


def wall(c, K, floor=FLOOR):
    x = np.arange(N, dtype=float)
    u = np.where(x < N // 2, x, x + 1.0)
    u[N // 2] = N / 2 + 0.5
    ke = np.ones(N - 1)
    hit, s_max = False, 0.0
    for _ in range(STEPS_OUT):
        u = relax(u, ke, K)
        s = 0.5 * ((u - x)[:-1] + (u - x)[1:])
        s_max = float(np.abs(s).max())
        ke_new = KAP * (1.0 + c * s)
        if (ke_new < floor * KAP).any():
            hit = True
        ke = np.clip(ke_new, floor * KAP, None)
    return u, ke, hit, s_max


def gl_sigma(Ng=400, lam=LAM, step=0.05, iters=40000):
    x = np.arange(Ng, dtype=float)
    phi = np.tanh(np.sqrt(lam / 2.0) * (x - Ng / 2.0))
    for _ in range(iters):
        lap = np.zeros(Ng)
        lap[1:-1] = phi[:-2] - 2.0 * phi[1:-1] + phi[2:]
        dV = lam * phi * (phi ** 2 - 1.0)
        phi = phi - step * (-lap + dV)
        phi[0], phi[-1] = -1.0, 1.0
    return float(0.5 * (np.diff(phi) ** 2).sum()
                 + (lam / 2.0 * (phi ** 2 - 1.0) ** 2).sum())


print("=" * 74)
print("C1 窗口算术判据：软化反馈强度上限与所需驻留半径的定量对接（U-1 至 U-3）")
print("=" * 74)

scan = {}
for K in (0.02, 0.005):
    u0, _, _, s0 = wall(0.0, K)
    E0 = energy(u0, np.ones(N - 1), K)
    w0 = width(u0)
    c_floor_pred = (1.0 - FLOOR) / s0
    print(f"\n[U-1] K = {K}：c = 0 基准 E_0 = {E0:.6f}，ell* = {w0:.3f}，"
          f"s_max = {s0:.3f}，结构预测 c_floor = -{c_floor_pred:.3f}")
    rows = []
    first_hit = None
    for c in CS[1:]:
        u, ke, hit, sm = wall(c, K)
        E = energy(u, ke, K)
        dE = (E - E0) / E0
        w = width(u)
        rows.append((c, dE, w, hit, sm))
        if hit and first_hit is None:
            first_hit = c
        print(f"   c = {c:5.2f}：Delta E / E_0 = {dE * 100:+7.3f}%，ell* = {w:.3f}，"
              f"kappa_e 最小值 = {ke.min():.3f}，下限触及 = {hit}，s_max = {sm:.3f}")
    c_best, eta_max = max(((c, -dE) for (c, dE, w, hit, sm) in rows), key=lambda t: t[1])
    print(f"   U-1 读数：eta_max = {eta_max * 100:.3f}%（c = {c_best}）；"
          f"首次触及下限 c_cap = {first_hit}，结构预测 -{c_floor_pred:.3f}")
    scan[K] = dict(E0=E0, w0=w0, s0=s0, c_floor_pred=c_floor_pred, rows=rows,
                   c_best=c_best, eta_max=eta_max, first_hit=first_hit)

print("\n[U-1 补充] 刚度下限取值对折扣前沿的影响（前沿点取 c = -(1 - floor) 与再进 0.05）")
for K in (0.02, 0.005):
    E0 = scan[K]["E0"]
    for fl in (0.05, 0.02):
        for c in (-(1.0 - fl) + 0.05, -(1.0 - fl)):
            u, ke, hit, sm = wall(c, K, floor=fl)
            dE = (energy(u, ke, K) - E0) / E0
            print(f"   K = {K:5.3f}，floor = {fl:.2f}，c = {c:5.2f}："
                  f"Delta E / E_0 = {dE * 100:+7.3f}%，ell* = {width(u):.3f}，"
                  f"kappa_e 最小值 = {ke.min():.3f}，下限触及 = {hit}")

eta_max_all = max(scan[K]["eta_max"] for K in scan)

print("\n[U-2] 越零对接（环线项 a = sigma_0 (eps - eta)，越零需 eta > eps）")
for name, eps in EPS_LIST:
    eta_tot = (1.0 - eps) + eta_max_all
    if eta_max_all > eps:
        verdict = f"越零成立（裕量 {eta_max_all / eps:.2f} 倍）"
    else:
        verdict = f"越零不成立（缺口 {eps / eta_max_all:.2f} 倍）"
    print(f"   {name}：所需 eta = {eps * 100:.1f}%，实测上限 = {eta_max_all * 100:.3f}%，"
          f"{verdict}；最有利总节省率 = {eta_tot * 100:.2f}%（越零需 > 100%）")

sigma_disc_own = gl_sigma()
print("\n[U-3] 落窗对接（R* = sigma_0 (eta - eps) / delta；落窗需 R* >= R*_req）")
print(f"   GL 标度：delta = {DELTA_GL:.4f}；sigma_0 连续参照 = {SIGMA_CONT:.4f}，"
      f"库内离散读数 = {SIGMA_DISC_LIB:.3f}，本脚本离散复算 = {sigma_disc_own:.4f}")
dsig = [("连续参照", DELTA_GL / SIGMA_CONT), ("库内离散读数", DELTA_GL / SIGMA_DISC_LIB)]
n_fail = 0
for eps_name, eps in EPS_LIST:
    print(f"   eps = {eps:.3f}（{eps_name}）：")
    for rreq in R_REQ:
        parts = []
        for nm, ds in dsig:
            eta_req = eps + rreq * ds
            if eta_req > eta_max_all:
                n_fail += 1
                parts.append(f"{nm} 需 eta = {eta_req * 100:.1f}%（缺口 {eta_req / eta_max_all:.1f} 倍）")
            else:
                parts.append(f"{nm} 需 eta = {eta_req * 100:.1f}%（可及）")
        print(f"      R*_req = {rreq:4.2f}：" + "；".join(parts))
print(f"   汇总：落窗对接共 {len(EPS_LIST) * len(R_REQ) * len(dsig)} 档，不达 {n_fail} 档。")
print("   读法：eta > 100% 的行表示即使裸线张力全数反转（eta = 1）仍不够，"
      "落窗需要负项量级超过裸线张力本身。")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
ax = axes[0]
for K, mk in ((0.02, "o"), (0.005, "s")):
    d = scan[K]
    xs = [abs(c) for (c, dE, w, hit, sm) in d["rows"]]
    ys = [-dE * 100 for (c, dE, w, hit, sm) in d["rows"]]
    ax.plot(xs, ys, mk + "-", ms=4, label=f"K = {K}")
    ax.plot([abs(d["c_best"])], [d["eta_max"] * 100], mk, ms=11, mfc="none", mew=2)
ax.axvline(0.9, color="gray", ls="-.", lw=1.0)
ax.text(0.905, 1.0, "下限触及 |c| = 0.9", fontsize=8, color="gray", rotation=90, va="bottom")
ax.axhline(10.0, color="tab:red", ls="--", lw=1.2, label="越零门槛 10%（W1/W2 档）")
ax.axhline(25.6, color="darkred", ls=":", lw=1.2, label="越零门槛 25.6%（构造档）")
ax.set_xlabel("软化强度 |c|")
ax.set_ylabel("界面优惠 |Delta E| / E_0 (%)")
ax.set_title("(a) 软化反馈的优惠幅度与上限")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1]
xpos = np.arange(len(R_REQ))
wb = 0.35
for i, (nm, ds) in enumerate(dsig):
    reqs = [EPS_LIST[0][1] + r * ds for r in R_REQ]
    ax.bar(xpos + (i - 0.5) * wb, [v * 100 for v in reqs], width=wb, label=nm)
    for xi, v in zip(xpos + (i - 0.5) * wb, reqs):
        ax.text(xi, v * 100 + 4, f"{v * 100:.0f}%", ha="center", fontsize=8)
ax.axhline(eta_max_all * 100, color="tab:green", ls="--", lw=1.2,
           label=f"实测上限 {eta_max_all * 100:.1f}%")
ax.axhline(100.0, color="black", ls="-", lw=1.0, label="全数反转（100%）")
ax.set_xticks(xpos)
ax.set_xticklabels([f"R* >= {r}" for r in R_REQ])
ax.set_ylabel("所需折扣 eta_req (%)")
ax.set_title("(b) 落窗所需折扣与实测上限")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")

fig.tight_layout()
out = "D:/software/TianXingFC/images/c1_softening_window.png"
fig.savefig(out, dpi=150)
print(f"\n[图] 已保存 {out}")
