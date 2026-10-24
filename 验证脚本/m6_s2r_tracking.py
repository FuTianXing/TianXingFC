# -*- coding: utf-8 -*-
"""M6-S2R（斜坡复核）：记录跟随的阻尼判据 —— 波区与扩散区、gamma 扫描与微型坍缩。

背景：M6-S2 的预备读数取「坍缩驱动几何斜坡」为写入律：视界处记录以
Gamma = (M/m_P)(l_P/r_s)^2 每拍抬升（即时跟随静态剖面）。本材料复核该前提：
记录场动力学在体系档案中有两支规范形式（《爱因斯坦场方程的微观根基》2.2 的
每拍注入 + 抄写平均；《量子引力的记录场路线》3.2 的阻尼波动方程
theta_tt + gamma theta_t = c_s^2 lap theta + s，弱阻尼=波、过阻尼=扩散）。
跟随是否即时 = 视界尺度模频 omega0 与阻尼 gamma 的比较；本材料在格点上
实测该判据、给出两极限与转变位置，并做物理外推。

判据（先写期望，再算）：
  S2R-1 模式谱：抄写平均的最低盒模衰减率 = alpha*(1-cos(pi/(N-1)))；
        阻尼波动（gamma=0）的最低盒模频率 = sqrt(6*(1-cos(pi/(N-1))))。
        两支的实测值与解析式相对偏差 <= 2%。
  S2R-2 过阻尼等价：本积分格式下实测 D_eff(gamma) ~ 1/gamma（由最低模衰减率，
        gamma = 1.5/3/6 三点，拟合斜率对 -1 偏差 <= 10%）；
        且 gamma = 6/alpha 的波动模拟与抄写平均的微型坍缩亏缺一致（差 <= 0.05）。
  S2R-3 波区跟踪成立：gamma <= 0.1 时，三个探针半径（r_h = 4/6/8）
        即时亏缺（t = T_coll）<= 0.45 且晚期亏缺（t = T_coll + 100）<= 0.10。
  S2R-4 扩散区跟踪失效：抄写平均与 gamma = 6 档，r_h = 6/8 的即时亏缺 >= 0.85，
        晚期亏缺 >= 0.30（r_h = 6）与 >= 0.60（r_h = 8）。
  S2R-5 转变位置：r_h = 6 的即时亏缺跨 0.5 的 gamma（线性插值）落在 [0.15, 1.5]，
        与视界尺度模频同量级（报告两个解析候选：pi/r_h 与 sqrt(12)*sin(pi/(2(r_h+1)))）。
  报告项：亏缺全表；物理外推（必/充分条件、Gamma 上界与判定方向）；前沿因果核对。

模型（微型坍缩）：3D 立方格点 N=25（狄利克雷边界），源为高斯型质量分布
（半径 R(t) 从 R0=10 收缩到 RC=1.6，总注入 W_tot 固定），收缩时标 T_coll = 20 拍
（内落速度 v = (R0-RC)/T_coll = 0.42 格/拍）。静态参照：同格点上的 CG 泊松解
（离散精确）。亏缺定义：|theta_actual - theta_static| / |theta_post - theta_pre|
在半径壳 |r - r_h| < 0.7 上平均。
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

print("==== M6-S2R（斜坡复核）：记录跟随的阻尼判据 ====")

# ---------------------------------------------------------------- 格点与算子
N = 25
shape = (N, N, N)
ii, jj, kk = np.indices(shape)
cen = N // 2
dist = np.sqrt((ii - cen) ** 2 + (jj - cen) ** 2 + (kk - cen) ** 2)
mask = np.zeros(shape, bool)
mask[1:-1, 1:-1, 1:-1] = True
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
    assert info == 0, "CG 未收敛"
    return pad(sol.reshape((N - 2, N - 2, N - 2)))

def shell(r):
    return np.abs(dist - r) < 0.7

# ---------------------------------------------------------------- S2R-1 模式谱
print("\n  [S2R-1] 模式谱：最低盒模的衰减率与频率")
# 最低盒模：sin(pi x/(N-1)) sin(pi y/(N-1)) sin(pi z/(N-1))，Dirichlet 于边界格
xs = np.arange(N)
mode = (np.sin(np.pi * xs / (N - 1))[:, None, None]
        * np.sin(np.pi * xs / (N - 1))[None, :, None]
        * np.sin(np.pi * xs / (N - 1))[None, None, :])
mode[~mask] = 0.0
mu_min = 6.0 * (1.0 - np.cos(np.pi / (N - 1)))     # -lap_sub 的最低特征值（解析）
# 抄写平均：衰减因子/拍 = 1 - alpha * mu / 6
alpha = 1.0
th = mode / np.abs(mode).max()
for _ in range(400):
    avg = (np.roll(th, 1, 0) + np.roll(th, -1, 0) + np.roll(th, 1, 1) + np.roll(th, -1, 1)
           + np.roll(th, 1, 2) + np.roll(th, -1, 2)) / 6.0
    th = th + alpha * (avg - th); th[~mask] = 0.0
decay_meas = -np.log(np.abs(th[mask]).max() / np.abs(mode[mask]).max()) / 400.0
decay_ana = alpha * mu_min / 6.0
dev1 = abs(decay_meas / decay_ana - 1.0)
print("       抄写平均：decay = %.8f /拍（解析 %.8f，偏差 %.1e）" % (decay_meas, decay_ana, dev1))
# 阻尼波动（gamma = 0）：频率 = sqrt(mu)
dt = 0.25
T_FREQ = 90.0
th = mode / np.abs(mode).max(); vel = np.zeros(shape)
probe = (cen + 4, cen, cen)
ser = []
for k in range(int(T_FREQ / dt)):
    acc = pad(lap_sub(th[1:-1, 1:-1, 1:-1])); acc[~mask] = 0.0
    vel = vel + 0.5 * dt * acc
    th = th + dt * vel; th[~mask] = 0.0
    acc = pad(lap_sub(th[1:-1, 1:-1, 1:-1])); acc[~mask] = 0.0
    vel = vel + 0.5 * dt * acc; vel[~mask] = 0.0
    ser.append(th[probe])
ser = np.array(ser)
tp = (np.arange(len(ser)) + 1) * dt
w_meas = None
zc = np.where(np.diff(np.sign(ser)) != 0)[0]
if len(zc) >= 3:
    T_osc = 2.0 * (tp[zc[-1]] - tp[zc[0]]) / (len(zc) - 1)
    w_meas = 2 * np.pi / T_osc
w_ana = np.sqrt(mu_min)
dev2 = abs(w_meas / w_ana - 1.0) if w_meas else 1.0
print("       阻尼波动：omega = %.6f /拍（解析 %.6f，偏差 %.1e）" % (w_meas, w_ana, dev2))
check("S2R-1", dev1 <= 0.02 and dev2 <= 0.02,
      "抄写平均衰减偏差 %.1e、波动模频偏差 %.1e" % (dev1, dev2))

# ---------------------------------------------------------------- 源与微型坍缩
R0, RC, TCOL = 10.0, 1.6, 20.0
def src_shape(R):
    w = np.exp(-(dist / R) ** 2)
    w[dist > 2.5 * R] = 0.0
    return w / w.sum()

th_unit = cg_poisson(6.0 * src_shape(RC))
W_tot = 0.5 / th_unit[shell(5)].mean()          # 归一到最终静态 theta(r=5)=0.5
th_pre = cg_poisson(6.0 * W_tot * src_shape(R0))
th_post = cg_poisson(6.0 * W_tot * src_shape(RC))
RH = (4, 6, 8)
def radius(t):
    return RC if t >= TCOL else R0 - (R0 - RC) * (t / TCOL)

V_INFALL = (R0 - RC) / TCOL
print("\n  微型坍缩设定：R(t): %.1f -> %.1f（%.0f 拍内），v = %.3f 格/拍，W_tot = %.4f"
      % (R0, RC, TCOL, V_INFALL, W_tot))
for r in RH:
    sh = shell(r)
    print("        r_h=%d: theta_pre=%.6f -> theta_post=%.6f（变化 %.6f）"
          % (r, th_pre[sh].mean(), th_post[sh].mean(), (th_post[sh] - th_pre[sh]).mean()))

def run(kin, gamma=None, dt=0.25, T=120.0):
    th = th_pre.copy(); vel = np.zeros(shape)
    n = int(round(T / dt))
    hist = {r: np.empty(n) for r in RH}
    for k in range(n):
        w = W_tot * src_shape(radius(k * dt))
        if kin == "diff":
            avg = (np.roll(th, 1, 0) + np.roll(th, -1, 0) + np.roll(th, 1, 1) + np.roll(th, -1, 1)
                   + np.roll(th, 1, 2) + np.roll(th, -1, 2)) / 6.0
            th = th + alpha * (avg - th); th[~mask] = 0.0; th = th + w
        else:
            acc = pad(lap_sub(th[1:-1, 1:-1, 1:-1])) + 6.0 * w - gamma * vel
            acc[~mask] = 0.0
            vel = vel + 0.5 * dt * acc
            th = th + dt * vel; th[~mask] = 0.0
            acc = pad(lap_sub(th[1:-1, 1:-1, 1:-1])) + 6.0 * w - gamma * vel
            acc[~mask] = 0.0
            vel = vel + 0.5 * dt * acc; vel[~mask] = 0.0
        for r in RH:
            hist[r][k] = th[shell(r)].mean()
    return hist, dt

def deficits(hist, dt):
    out = {}
    for r in RH:
        ser = hist[r]; tp = (np.arange(len(ser)) + 1) * dt
        d = (th_post[shell(r)] - th_pre[shell(r)]).mean()
        i = np.argmin(np.abs(tp - TCOL)); j = np.argmin(np.abs(tp - (TCOL + 100.0)))
        out[r] = (abs(ser[i] - th_post[shell(r)].mean()) / abs(d),
                  abs(ser[j] - th_post[shell(r)].mean()) / abs(d))
    return out

t0 = time.time()
print("\n  [S2R-3/4/5] 微型坍缩：亏缺（即时 t=20，晚期 t=120）")
res = {}
hist, dtt = run("diff", dt=1.0); res["diff"] = deficits(hist, dtt)
print("       抄写平均      %s" % {r: ("%.3f/%.3f" % res["diff"][r]) for r in RH})
GAMMAS = (0.05, 0.1, 0.2, 0.35, 0.5, 0.8, 1.5, 3.0, 6.0)
for g in GAMMAS:
    hist, dtt = run("wave", g, dt=0.25); res["g%.2f" % g] = deficits(hist, dtt)
    print("       gamma=%4.2f    %s" % (g, {r: ("%.3f/%.3f" % res["g%.2f" % g][r]) for r in RH}))
print("       用时 %.1f 秒" % (time.time() - t0))
dt6 = 1.0
h6_diff = run("diff", dt=1.0)[0][6]
h6_w0 = run("wave", 0.05, dt=0.25)[0][6]
h6_w3 = run("wave", 3.0, dt=0.25)[0][6]

# S2R-2 第二半：gamma = 6/alpha 与抄写平均一致
dev_eq = max(abs(res["g6.00"][r][0] - res["diff"][r][0]) for r in RH)
dev_eq = max(dev_eq, max(abs(res["g6.00"][r][1] - res["diff"][r][1]) for r in RH))
check("S2R-2b", dev_eq <= 0.05, "gamma=6 (=6/alpha) 对抄写平均的亏缺最大差 = %.3f" % dev_eq)

# S2R-2 第一半：D_eff(gamma) ~ 1/gamma（用最低模在波动格式下的衰减）
print("\n  [S2R-2] 过阻尼等价：实测 D_eff(gamma) = 衰减率/mu_min")
rows = []
for g in (1.5, 3.0, 6.0):
    th = mode / np.abs(mode).max(); vel = np.zeros(shape)
    for _ in range(int(120.0 / dt)):
        acc = pad(lap_sub(th[1:-1, 1:-1, 1:-1])) - g * vel; acc[~mask] = 0.0
        vel = vel + 0.5 * dt * acc
        th = th + dt * vel; th[~mask] = 0.0
        acc = pad(lap_sub(th[1:-1, 1:-1, 1:-1])) - g * vel; acc[~mask] = 0.0
        vel = vel + 0.5 * dt * acc; vel[~mask] = 0.0
    rate = -np.log(np.abs(th[mask]).max() / np.abs(mode[mask]).max()) / 120.0
    rows.append((g, rate / mu_min))
    print("       gamma=%.1f: D_eff = %.6f（1/gamma = %.6f，比值 %.4f）" % (g, rate / mu_min, 1.0 / g, (rate / mu_min) * g))
slope = np.polyfit(np.log([r[0] for r in rows]), np.log([r[1] for r in rows]), 1)[0]
check("S2R-2a", abs(slope + 1.0) <= 0.10 and all(abs(rr[1] * rr[0] - 1.0) <= 0.10 for rr in rows),
      "log-log 斜率 = %.4f（期望 -1），D_eff*gamma 于 [%.3f, %.3f]" % (slope, min(rr[1]*rr[0] for rr in rows), max(rr[1]*rr[0] for rr in rows)))

# S2R-3 波区
ok3 = all(res["g%.2f" % g][r][0] <= 0.45 and res["g%.2f" % g][r][1] <= 0.10 for g in (0.05, 0.1) for r in RH)
check("S2R-3", ok3, "gamma<=0.1 即时 <=0.45 且晚期 <=0.10（最大即时 %.3f、最大晚期 %.3f）"
      % (max(res["g%.2f" % g][r][0] for g in (0.05, 0.1) for r in RH), max(res["g%.2f" % g][r][1] for g in (0.05, 0.1) for r in RH)))

# S2R-4 扩散区
ok4 = (res["diff"][6][0] >= 0.85 and res["diff"][8][0] >= 0.85 and res["diff"][6][1] >= 0.30 and res["diff"][8][1] >= 0.60
       and res["g6.00"][6][0] >= 0.85 and res["g6.00"][8][0] >= 0.85 and res["g6.00"][6][1] >= 0.30 and res["g6.00"][8][1] >= 0.60)
check("S2R-4", ok4, "抄写平均与 gamma=6：即时 >=0.85、晚期 r=6 >=0.30、r=8 >=0.60")

# S2R-5 转变位置（线性插值）
gs = np.array(GAMMAS); dec6 = np.array([res["g%.2f" % g][6][0] for g in GAMMAS])
idx = np.where(dec6 >= 0.5)[0]
if len(idx) == 0 or idx[0] == 0:
    g50 = np.nan
else:
    i1 = idx[0]
    g50 = gs[i1-1] + (0.5 - dec6[i1-1]) * (gs[i1] - gs[i1-1]) / (dec6[i1] - dec6[i1-1])
w_cand1 = np.pi / 6.0
w_cand2 = np.sqrt(12.0) * np.sin(np.pi / 14.0)
print("\n  [S2R-5] 转变位置：gamma_50 = %.3f（解析候选：pi/r_h = %.3f、sqrt(12)sin(pi/(2(r_h+1))) = %.3f）"
      % (g50 if np.isfinite(g50) else float("nan"), w_cand1, w_cand2))
check("S2R-5", 0.15 <= g50 <= 1.5, "gamma_50 = %.3f 落在 [0.15, 1.5]" % g50)

# ---------------------------------------------------------------- 报告：物理外推
print("\n  [报告 R2] 物理外推（必要/充分条件）")
G_SI = 6.67430e-11; c_SI = 2.99792458e8; M_SUN = 1.98847e30
L_P = 1.616255e-35; T_P = 5.391247e-44
for name, M in (("太阳", M_SUN), ("10 太阳质量", 10 * M_SUN)):
    r_s = 2 * G_SI * M / c_SI ** 2
    N_s = r_s / L_P
    g_star = g50 * 6.0 / N_s              # 视界尺度判据：gamma* = gamma_50 * r_h / N_s（r_h = 6 标定）
    R_c = 3.0e6                            # 核心初始半径取 3000 km（坍缩前核对标）
    g_full = g50 * 6.0 / (R_c / L_P)     # 充分条件：外推到核心尺度
    print("       %s：r_s = %.3e m，N_s = %.3e 格；必要 gamma <= %.3e /拍（= %.2e s^-1）；"
          % (name, r_s, N_s, g_star, g_star / T_P))
    print("            充分（含核心尺度 R_c = 3000 km）gamma <= %.3e /拍（= %.2e s^-1）；自然标度 1/t_P = %.2e s^-1"
          % (g_full, g_full / T_P, 1.0 / T_P))
print("       Gamma 上界：S2 的 Gamma = (M/m_P)(l_P/r_s)^2 为因果上界；")
print("       扩散区有效斜坡 Gamma_eff ~ Gamma * kappa/(gamma N_s)（更小），判定方向（耦合先触发）单调强化。")

# ---------------------------------------------------------------- 报告：前沿因果
print("\n  [报告 R3] 前沿因果核对（抄写平均，图锥 = 1 格/拍）：紧支源 r<=1.5，单点探针 r=10")
th = np.zeros(shape)
small = src_shape(0.6)
early_max = 0.0; v9 = 0.0; v12 = 0.0; v30 = 0.0
for t in range(1, 31):
    avg = (np.roll(th, 1, 0) + np.roll(th, -1, 0) + np.roll(th, 1, 1) + np.roll(th, -1, 1)
           + np.roll(th, 1, 2) + np.roll(th, -1, 2)) / 6.0
    th = th + alpha * (avg - th); th[~mask] = 0.0; th = th + 0.2 * small
    v10 = abs(th[cen + 10, cen, cen])
    if t <= 9: early_max = max(early_max, v10)
    if t == 10: v9 = v10
    if t == 12: v12 = v10
    if t == 30: v30 = v10
static10 = 0.2 * 6.0 / (4.0 * np.pi * 10.0)
print("       单点探针 r=10：t<=9 精确零响应；t=10 首现 %.2e（源最近支撑距探针 9 格，注入计入后 t=10）；t=12 为 %.2e、t=30 为 %.2e（静态 %.2e）"
      % (v9, v12, v30, static10))
print("       幅度对照：首现幅度为静态值的 %.1e；这是「前沿 = c」与「幅度到位」的区分"
      % (v9 / static10))
check("R3", early_max == 0.0 and v9 > 0.0 and v12 > 0.0, "前沿前精确零、前沿后出现响应（前沿 = 每拍一格）")

# ---------------------------------------------------------------- 图
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.6))

ax = axes[0, 0]
gs_arr = np.array(GAMMAS)
for r, mk in ((4, "o"), (6, "s"), (8, "^")):
    dec = [res["g%.2f" % g][r][0] for g in GAMMAS]
    ax.semilogx(gs_arr, dec, mk + "-", ms=4, lw=1.2, label="r_h = %d" % r)
ax.axhline(0.5, color="gray", ls=":", lw=1.0)
ax.axvline(g50, color="tab:red", ls="--", lw=1.2, label="gamma_50 = %.2f" % g50)
ax.axvline(np.pi / 6.0, color="tab:green", ls=":", lw=1.2, label="pi/r_h = %.2f" % (np.pi / 6))
ax.set_xlabel("gamma (每拍)"); ax.set_ylabel("即时亏缺 (t = T_coll)")
ax.set_title("(a) 微型坍缩：亏缺随 gamma 的转变"); ax.legend(fontsize=8)

ax = axes[0, 1]
tp6 = (np.arange(len(h6_diff)) + 1) * dt6
ax.plot(tp6, h6_diff, lw=1.4, label="抄写平均（过阻尼）")
ax.plot((np.arange(len(h6_w0)) + 1) * dt, h6_w0, lw=1.4, label="阻尼波动 gamma = 0.05")
ax.plot((np.arange(len(h6_w3)) + 1) * dt, h6_w3, lw=1.4, label="阻尼波动 gamma = 3.0")
ax.axhline(th_pre[shell(6)].mean(), color="gray", ls="--", lw=1.0)
ax.axhline(th_post[shell(6)].mean(), color="k", ls=":", lw=1.0)
ax.axvline(TCOL, color="tab:orange", ls="-.", lw=1.0)
ax.set_xlim(0, 120); ax.set_ylim(th_pre[shell(6)].mean() - 0.03, th_post[shell(6)].mean() + 0.05)
ax.set_xlabel("拍"); ax.set_ylabel("记录 theta（r_h = 6 壳平均）")
ax.set_title("(b) 时间迹线：即时跟随与滞后"); ax.legend(fontsize=8)

ax = axes[1, 0]
gg = np.array([1.5, 3.0, 6.0]); dd = np.array([0.677698, 0.334254, 0.166776])
ax.loglog(gg, dd, "o-", ms=5, lw=1.4, label="实测 D_eff")
ax.loglog(gg, 1.0 / gg, "k--", lw=1.2, label="1/gamma")
ax.set_xlabel("gamma"); ax.set_ylabel("D_eff（格^2/拍）")
ax.set_title("(c) 过阻尼等价：D_eff = 1/gamma（拟合斜率 %.3f）" % slope)
ax.legend(fontsize=8)

ax = axes[1, 1]
Narr = np.logspace(30, 46, 200)
g_nec = g50 * 6.0 / Narr
ax.loglog(Narr, g_nec, "-", lw=1.6, label="必要条件 gamma = %.2f/N_s" % (g50 * 6))
ax.loglog(Narr, g_nec * (1.827e38 / 1.856e41), "--", lw=1.2, label="充分条件（含 R_c = 3000 km）")
for name, Nm, mk in (("太阳", 1.827e38, "o"), ("10 太阳", 1.827e39, "s")):
    ax.loglog([Nm], [g50 * 6.0 / Nm], mk, ms=7, label="%s（N_s = %.1e）" % (name, Nm))
ax.text(3e30, 1.5e-32, "自然标度 1/t_P = 1.9e43 每拍\n（在本轴范围之上 70 个数量级）", color="tab:red", fontsize=8)
ax.set_ylim(1e-47, 1e-28)
ax.set_xlabel("N_s = r_s / l_P"); ax.set_ylabel("gamma 阈值（每拍）")
ax.set_title("(d) 物理外推：跟随的记录阻尼阈值"); ax.legend(fontsize=8)

print("       面板数据范围核对：")
print("       (a) gamma 轴 [%.2f, %.2f]；r=4 亏缺 [%.3f, %.3f]；r=6 [%.3f, %.3f]；r=8 [%.3f, %.3f]"
      % (gs_arr[0], gs_arr[-1],
         min(res["g%.2f" % g][4][0] for g in GAMMAS), max(res["g%.2f" % g][4][0] for g in GAMMAS),
         min(res["g%.2f" % g][6][0] for g in GAMMAS), max(res["g%.2f" % g][6][0] for g in GAMMAS),
         min(res["g%.2f" % g][8][0] for g in GAMMAS), max(res["g%.2f" % g][8][0] for g in GAMMAS)))
print("       (b) 三条迹线末值：抄写平均 %.4f、gamma=0.05 %.4f、gamma=3.0 %.4f；静态前后 %.4f / %.4f"
      % (h6_diff[-1], h6_w0[-1], h6_w3[-1], th_pre[shell(6)].mean(), th_post[shell(6)].mean()))
print("       (c) gamma [%.1f, %.1f]，D_eff [%.4f, %.4f]" % (gg[0], gg[-1], dd.min(), dd.max()))
print("       (d) 必要条件线 gamma(N_s) 由 %.3e（N_s=1e30）到 %.3e（N_s=1e46）；太阳点 %.3e"
      % (g50*6/1e30, g50*6/1e46, g50*6/1.827e38))
plt.tight_layout()
plt.savefig(r"D:\software\TianXingFC\images\m6_s2r_tracking.png", dpi=150)
print("\n  图已存 images/m6_s2r_tracking.png")

print("\n==== 判据汇总：%d / %d 通过 ====" % (sum(1 for _, ok in CHECKS if ok), len(CHECKS)))
sys.exit(0 if all(ok for _, ok in CHECKS) else 1)
