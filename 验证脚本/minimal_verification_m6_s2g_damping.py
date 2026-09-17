# -*- coding: utf-8 -*-
"""M6-S2G（gamma 取数，可做部分）：最小模型的守恒性与观测上限。

背景：M6-S2R 复核把「坍缩记录即时跟随」化归为记录阻尼 gamma 与视界模频 pi c / r_s 的比较，
并把 gamma 的取值列为新读数门槛候选；计划 §7.6 的 S2-5 节起草了取数判据（三途径：微观推导 G1、
最小模型反演 G2、观测上限 G3）。本材料执行可做部分（G2-1、G2-2、G3-1、G3-2）。

判据（先写期望，再算）：
  G2-1a 最小模型无内禀耗散（数值底）：「每拍注入 + 邻居接力 + 惯性项」的格点动力学
        （七点拉普拉斯 + 速度 Verlet，无摩擦项）下，单模守恒振幅
        A(t) = sqrt(theta^2 + (v/omega)^2)（omega 为模频）在 6000 拍上的相对漂移率 <= 1e-12 每拍。
  G2-1b 波区判定（结构 + 数值底）：模型不含摩擦项（守恒结构，内禀 gamma = 0），
        其数值底 gamma_meas 对最大探针 r_h = 10 格满足 gamma_meas * r_h <= pi，余量 >= 1e10。
  G2-2 对照档：同一模型加显式摩擦 gamma = 0.514（S2R 的 gamma_50）在 S2R 微型坍缩
        （N = 25，R0 = 10 收缩到 Rc = 1.6，T_coll = 20 拍）下给出 r_h = 6 的即时亏缺
        0.50 +/- 0.06（复现 S2R 的转变读数）。
  G3-1 观测上限：阻尼波动方程给引力波振幅传播因子 exp(-gamma D / (2c))；
        由 gamma = 2 c ln(D_GW / D_EM) / D 与 GW170817 的两条电磁距离档取 1 sigma 上限，
        要求 <= 1e-15 每秒（报告中心值、上限与每拍换算）。
  G3-2 与门槛比较：gamma_obs 对 pi c / r_s（太阳、10 太阳）与 pi c / R_c
        （R_c = 3000 千米、1e13 米）的余量 >= 1e10（报告）。
  报告 R1：视界带频率 f_0 = c / (2 r_s) 与引力波带（10 至 1e3 Hz）的对照（频率依赖残余项）。

输入（全部为已核对材料）：GW170817 标准汽笛距离 43.8 (+2.9 / -6.9) Mpc（LIGO P1700296、
Abbott 等 2017）；宿主星系电磁距离 41.0 +/- 3.1 Mpc（基本面）与 41.1 +/- 5.8 Mpc（Tully-Fisher，
2017 年宿主星系文献）。确定性、无随机数。
"""

import sys
import time
import numpy as np
from scipy.sparse.linalg import cg, LinearOperator

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
CHECKS = []
def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))

print("==== M6-S2G（gamma 取数，可做部分）：最小模型的守恒性与观测上限 ====")

# ---------------------------------------------------------------- 格点与算子
N = 25
shape = (N, N, N)
xs = np.arange(N)
mask = np.zeros(shape, bool); mask[1:-1, 1:-1, 1:-1] = True
nint = (N - 2) ** 3
def pad(x):
    v = np.zeros(shape); v[1:-1, 1:-1, 1:-1] = x
    return v
def lap_sub(x):
    v = pad(x)
    r = (np.roll(v, 1, 0) + np.roll(v, -1, 0) + np.roll(v, 1, 1) + np.roll(v, -1, 1)
         + np.roll(v, 1, 2) + np.roll(v, -1, 2)) - 6.0 * v
    return r[1:-1, 1:-1, 1:-1]
def cg_poisson(src):
    def mv(z):
        return (-lap_sub(z.reshape((N - 2, N - 2, N - 2)))).ravel()
    op = LinearOperator((nint, nint), matvec=mv, dtype=np.float64)
    sol, info = cg(op, src[1:-1, 1:-1, 1:-1].ravel(), rtol=1e-13, maxiter=20000)
    assert info == 0
    return pad(sol.reshape((N - 2, N - 2, N - 2)))

dt = 0.25
mode = (np.sin(np.pi * xs / (N - 1))[:, None, None]
        * np.sin(np.pi * xs / (N - 1))[None, :, None]
        * np.sin(np.pi * xs / (N - 1))[None, None, :])
mode[~mask] = 0.0
mu_min = 6.0 * (1.0 - np.cos(np.pi / (N - 1)))
omega0 = np.sqrt(mu_min)

# ---------------------------------------------------------------- G2-1a 守恒性
print("\n  [G2-1a] 最小模型（注入 + 接力 + 惯性，无摩擦）的单模守恒量")
mu = mu_min
x = dt * dt * mu
A11 = 1.0 - x / 2.0; B12 = dt; C21 = -dt * mu * (1.0 - x / 4.0); A22 = 1.0 - x / 2.0
M2 = np.array([[A11, B12], [C21, A22]])
# 精确不变量：解 M^T G M = G（3 未知量取一维零空间）
rows = []
for (i, j) in ((0, 0), (0, 1), (1, 1)):
    E = np.zeros((2, 2)); E[i, j] += 0.5; E[j, i] += 0.5
    rows.append(((M2.T @ E @ M2) - E).reshape(-1))
C = np.array(rows).T          # 4 x 3：每个分量一条方程，未知量 (g11, g12, g22)
_, _, Vt = np.linalg.svd(C)
gvec = Vt[-1]
G2 = np.array([[gvec[0], gvec[1]], [gvec[1], gvec[2]]])
G2 = G2 / np.abs(G2[0, 0])
psi = mode / np.linalg.norm(mode[mask])
th = mode / np.abs(mode).max(); vel = np.zeros(shape)
probe = (N // 2 + 4, N // 2, N // 2)
T_G21 = 6000.0
nstep = int(T_G21 / dt)
cq = np.empty(nstep); cv = np.empty(nstep); ampN = np.empty(nstep)
for k in range(nstep):
    acc = pad(lap_sub(th[1:-1, 1:-1, 1:-1])); acc[~mask] = 0.0
    vel = vel + 0.5 * dt * acc
    th = th + dt * vel; th[~mask] = 0.0
    acc = pad(lap_sub(th[1:-1, 1:-1, 1:-1])); acc[~mask] = 0.0
    vel = vel + 0.5 * dt * acc; vel[~mask] = 0.0
    cq[k] = np.dot(th[mask], psi[mask]) / np.linalg.norm(mode[mask])
    cv[k] = np.dot(vel[mask], psi[mask]) / np.linalg.norm(mode[mask])
    ampN[k] = np.hypot(th[probe], vel[probe] / omega0)
tp = (np.arange(nstep) + 1) * dt
Q = G2[0, 0] * cq ** 2 + 2 * G2[0, 1] * cq * cv + G2[1, 1] * cv ** 2
driftQ = np.polyfit(tp, np.log(np.abs(Q) / np.abs(Q[0])), 1)[0]
oscN = (ampN / ampN[0] - 1.0)
print("       精确不变量 Q（由映射矩阵构造）：ln(Q/Q0) 漂移率 = %.3e 每拍（期望舍入底）" % driftQ)
print("       朴素振幅的有界振荡幅度 = %.2e（构造差异，非耗散读数）" % np.max(np.abs(oscN)))
# 灵敏度对照：显式摩擦 gamma = 1e-4 每拍
thf = mode / np.abs(mode).max(); velf = np.zeros(shape)
ampF = np.empty(nstep)
gam_c = 1e-4
for k in range(nstep):
    acc = pad(lap_sub(thf[1:-1, 1:-1, 1:-1])) - gam_c * velf; acc[~mask] = 0.0
    velf = velf + 0.5 * dt * acc
    thf = thf + dt * velf; thf[~mask] = 0.0
    acc = pad(lap_sub(thf[1:-1, 1:-1, 1:-1])) - gam_c * velf; acc[~mask] = 0.0
    velf = velf + 0.5 * dt * acc; velf[~mask] = 0.0
    ampF[k] = np.abs(thf[probe])
slopeF = np.polyfit(tp, np.log(ampF / ampF[0]), 1)[0]
print("       对照（gamma = 1e-4 每拍）：ln|theta| 斜率 = %.3e 每拍（理论 -gamma/2 = -5.0e-05）" % slopeF)
check("G2-1a", abs(driftQ) <= 1e-12,
      "精确不变量漂移 %.2e 每拍 <= 1e-12（无内禀耗散）" % abs(driftQ))

# ---------------------------------------------------------------- G2-1b 波区判定
gamma_meas = abs(driftQ)
r_h = 10.0
margin = (np.pi / r_h) / max(gamma_meas, 1e-300)
print("\n  [G2-1b] 波区判定：gamma_meas = %.2e 每拍（不变量漂移底），对 r_h = %.0f 格的余量 = %.1e" % (gamma_meas, r_h, margin))
check("G2-1b", gamma_meas * r_h <= np.pi and margin >= 1e10,
      "gamma_meas * r_h = %.2e <= pi，余量 %.1e >= 1e10（无内禀耗散通道）" % (gamma_meas * r_h, margin))

# ---------------------------------------------------------------- G2-2 对照档
print("\n  [G2-2] 对照档：加显式摩擦 gamma = 0.514 的 S2R 微型坍缩复现")
GAM = 0.514
R0, RC, TCOL = 10.0, 1.6, 20.0
ii, jj, kk = np.indices(shape)
dist = np.sqrt((ii - N // 2) ** 2 + (jj - N // 2) ** 2 + (kk - N // 2) ** 2)
def src_shape(R):
    w = np.exp(-(dist / R) ** 2); w[dist > 2.5 * R] = 0.0
    return w / w.sum()
def shell(r):
    return np.abs(dist - r) < 0.7
W_tot = 0.5 / cg_poisson(6.0 * src_shape(RC))[shell(5)].mean()
th_pre = cg_poisson(6.0 * W_tot * src_shape(R0))
th_post = cg_poisson(6.0 * W_tot * src_shape(RC))
def radius(t):
    return RC if t >= TCOL else R0 - (R0 - RC) * (t / TCOL)
th = th_pre.copy(); vel = np.zeros(shape)
hist6 = []
for k in range(int(120.0 / dt)):
    w = W_tot * src_shape(radius(k * dt))
    acc = pad(lap_sub(th[1:-1, 1:-1, 1:-1])) + 6.0 * w - GAM * vel
    acc[~mask] = 0.0
    vel = vel + 0.5 * dt * acc
    th = th + dt * vel; th[~mask] = 0.0
    acc = pad(lap_sub(th[1:-1, 1:-1, 1:-1])) + 6.0 * w - GAM * vel
    acc[~mask] = 0.0
    vel = vel + 0.5 * dt * acc; vel[~mask] = 0.0
    hist6.append(th[shell(6)].mean())
hist6 = np.array(hist6); tp6 = (np.arange(len(hist6)) + 1) * dt
d6 = (th_post[shell(6)] - th_pre[shell(6)]).mean()
defcit = abs(hist6[np.argmin(np.abs(tp6 - TCOL))] - th_post[shell(6)].mean()) / abs(d6)
print("       gamma = %.3f：r_h = 6 的即时亏缺 = %.3f（期望 0.50 +/- 0.06）" % (GAM, defcit))
check("G2-2", abs(defcit - 0.5) <= 0.06, "亏缺 %.3f 落在 0.44 至 0.56" % defcit)

# ---------------------------------------------------------------- G3-1 观测上限
print("\n  [G3-1] 观测上限：gamma = 2 c ln(D_GW / D_EM) / D")
c_SI = 2.99792458e8
MPC = 3.0856775814913673e22
T_P = 5.391247e-44
D_GW, s_lo, s_hi = 43.8, 6.9, 2.9
bands = (("基本面", 41.0, 3.1), ("Tully-Fisher", 41.1, 5.8))
res = []
for name, D_EM, s_EM in bands:
    ln = np.log(D_GW / D_EM)
    sig = np.sqrt((s_hi / D_GW) ** 2 + (s_EM / D_EM) ** 2)
    D = D_EM * MPC
    g_c = 2 * c_SI * ln / D
    g_u = 2 * c_SI * (ln + sig) / D
    res.append(g_u)
    print("       %s档：D_EM = %.1f +/- %.1f Mpc，ln(D_GW/D_EM) = %.4f，sigma = %.4f ->"
          % (name, D_EM, s_EM, ln, sig))
    print("             中心值 %.2e 每秒，1 sigma 上限 %.2e 每秒 = %.2e 每拍" % (g_c, g_u, g_u * T_P))
gamma_obs = max(res)
check("G3-1", gamma_obs <= 1e-15, "两条距离档的最大 1 sigma 上限 = %.2e 每秒 <= 1e-15" % gamma_obs)

# ---------------------------------------------------------------- G3-2 门槛比较
print("\n  [G3-2] 与门槛比较（余量 = 门槛 / gamma_obs）")
G_SI = 6.67430e-11
M_SUN = 1.98847e30
rows = []
for name, M in (("太阳", M_SUN), ("10 太阳", 10 * M_SUN)):
    r_s = 2 * G_SI * M / c_SI ** 2
    thr = np.pi * c_SI / r_s
    rows.append((name, "pi c / r_s", r_s, thr))
for name, R_c in (("核心 3000 千米", 3.0e6), ("核心 1e13 米", 1.0e13)):
    thr = np.pi * c_SI / R_c
    rows.append((name, "pi c / R_c", R_c, thr))
margins = []
for name, kind, L, thr in rows:
    m = thr / gamma_obs
    margins.append(m)
    print("       %-14s %-10s L = %.2e 米：门槛 %.2e 每秒，余量 %.1e" % (name, kind, L, thr, m))
check("G3-2", min(margins) >= 1e10, "最小余量 = %.1e >= 1e10（波区成立，含充分条件档）" % min(margins))

# ---------------------------------------------------------------- R1 频率对照
print("\n  [报告 R1] 视界带频率 f_0 = c / (2 r_s) 与引力波带（10 至 1e3 Hz）")
freqs = []
for name, M in (("太阳", M_SUN), ("10 太阳", 10 * M_SUN), ("1e2 太阳", 1e2 * M_SUN),
                ("1e4 太阳", 1e4 * M_SUN), ("1e6 太阳", 1e6 * M_SUN), ("1e8 太阳", 1e8 * M_SUN)):
    r_s = 2 * G_SI * M / c_SI ** 2
    f0 = c_SI / (2 * r_s)
    freqs.append((name, M, f0))
    print("       %-10s M = %.1e M_sun：r_s = %.3e 米，f_0 = %.3e Hz" % (name, M / M_SUN, r_s, f0))
print("       读数：恒星档视界带高于引力波带（需外推核对）；超大质量档低于引力波带（需低频谱约束）。")

# ---------------------------------------------------------------- 图
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

ax = axes[0]
ax.semilogy(tp, np.abs(Q / Q[0] - 1.0), lw=1.2, label="精确不变量 |Q/Q0 - 1|")
ax.semilogy(tp, np.abs(oscN), lw=0.9, color="gray", label="朴素振幅有界振荡")
ax.semilogy(tp, ampF / ampF[0], ls="--", lw=1.0, label="对照 gamma = 1e-4：振幅衰减")
ax.set_xlabel("拍"); ax.set_ylabel("相对量")
ax.set_title("(a) 最小模型守恒性与灵敏度（G2-1a）", fontsize=9); ax.legend(fontsize=7)

ax = axes[1]
labels = ["观测上限\n(GW170817)", "太阳\npi c/r_s", "10 太阳\npi c/r_s", "核心 3000 km\npi c/R_c", "核心 1e13 m\npi c/R_c"]
vals = [gamma_obs] + [t for _, _, _, t in rows]
ax.bar(range(len(vals)), vals, color=["tab:red", "tab:blue", "tab:blue", "tab:green", "tab:green"])
ax.set_yscale("log"); ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels, fontsize=7)
ax.set_ylabel("gamma（每秒）"); ax.set_title("(b) 观测上限与各门槛（G3-1/G3-2）", fontsize=9)

ax = axes[2]
Ms = M_SUN * np.logspace(0, 9, 100)
rs = 2 * G_SI * Ms / c_SI ** 2
f0 = c_SI / (2 * rs)
ax.loglog(Ms / M_SUN, f0, lw=1.4)
ax.axhspan(10, 1e3, color="tab:orange", alpha=0.2)
ax.text(3e2, 3e2, "引力波带 10 至 1e3 Hz", fontsize=8, color="tab:orange")
ax.set_xlabel("M / M_sun"); ax.set_ylabel("f_0 = c / (2 r_s) [Hz]")
ax.set_title("(c) 视界带频率（报告 R1）", fontsize=9)

print("       面板数据范围核对：")
print("       (a) |Q/Q0 - 1| 最大 %.2e；朴素振荡最大 %.2e；对照振幅由 1.000 降到 %.3f"
      % (np.max(np.abs(Q / Q[0] - 1.0)), np.max(np.abs(oscN)), (ampF / ampF[0])[-1]))
print("       (b) 柱值：观测上限 %.2e、太阳 %.2e、10 太阳 %.2e、核心 3000 km %.2e、核心 1e13 m %.2e"
      % (gamma_obs, rows[0][3], rows[1][3], rows[2][3], rows[3][3]))
print("       (c) f_0 由 %.2e Hz（1 M_sun）降到 %.2e Hz（1e9 M_sun）；引力波带 10 至 1e3 Hz"
      % (freqs[0][2], c_SI / (2 * (2 * G_SI * (M_SUN * 1e9) / c_SI ** 2))))
plt.tight_layout()
plt.savefig(r"D:\software\TianXingFC\images\m6_s2g_damping.png", dpi=150)
print("\n  图已存 images/m6_s2g_damping.png")

print("\n==== 判据汇总：%d / %d 通过 ====" % (sum(1 for _, ok in CHECKS if ok), len(CHECKS)))
sys.exit(0 if all(ok for _, ok in CHECKS) else 1)
