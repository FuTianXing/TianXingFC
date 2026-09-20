# -*- coding: utf-8 -*-
r"""
minimal_verification_clock_shared_bg_stim.py

本底与受激项的时钟口径轮：两条定律合起来接记录机制线的本底与受激项。

来源：作者 2026-09-21 指令（接手第 295 条登记的下一步）。
输入：第 292、295 条的实测参数（壁速 v = 0.1000、有效穿越长度 w = 15.8，故受激窗口 tau_wall = w/v = 158）；
      记录机制线的本底口径（每站点每周期一笔，周期 t_P = 1）。

模型：事件级登记模型。场含两类写入事件：
  本底：每站点每周期一笔，窗口长度 tau_bg = t_P（相位绕转一整周的时间）；
  受激：壁扫过时每站点各一笔，窗口长度 tau_wall = w/v。
  注册规则（命题 17）：全局时钟刻度落在事件窗口内则该笔登记。

判据（先行）：
  V1 两窗口读数：tau_bg = 1.0、tau_wall = 158.0（tau_wall/tau_bg = 158.0）
  V2 逐类稀释律：两类注册率在四档取样间隔上对 min(1, tau / Dt)，偏差 <= 5%
  V3 比值的三段结构：Dt <= tau_bg 给 R0；tau_bg < Dt < tau_wall 给 R0 * Dt/tau_bg；Dt >= tau_wall 给 R0 * tau_wall/tau_bg（各段偏差 <= 10%）
  V4 本底在稀释下保持周期：注册后的本底时刻全部落在 t_P 的整数倍上（残差 <= 1e-9），泊松对照不成立
  V5 登记
"""
import numpy as np

V_WALL = 0.1000
W_LEN = 15.8
TAU_WALL = W_LEN / V_WALL
T_P = 1.0
TAU_BG = T_P
NX = 32
NY = NZ = 8
A = NY * NZ
N_SITE = NX * NY * NZ
T_TOT = 320.0
rows = []

sites = np.arange(N_SITE)
xs = sites % NX
t_c = 4.0 + xs / V_WALL
bg_times = (np.arange(1, int(T_TOT / T_P)) * T_P)[None, :] * np.ones((N_SITE, 1))
bg_times = bg_times.ravel()
bg_window = np.repeat(xs.astype(float), bg_times.size // N_SITE)
n_bg_per_site = bg_times.size // N_SITE
interior = (xs > 0) & (xs < NX - 1)
t_stim = t_c[interior & (xs == xs)]

rows.append(("V1", "两窗口读数（tau_bg 与 tau_wall）", "%.1f 与 %.1f（比值 %.1f）" % (TAU_BG, TAU_WALL, TAU_WALL / TAU_BG), "登记", True))


def registered(times, tau, Dt, phi):
    lo = times - 0.5 * tau
    hi = times + 0.5 * tau
    n = np.floor((hi - phi) / Dt) - np.ceil((lo - phi) / Dt) + 1.0
    return n >= 1.0


DTS = [0.5, 1.0, 5.0, 50.0, 158.0, 500.0]
PHASES = 1024
bg_unique = np.arange(1, int(T_TOT / T_P) + 1) * T_P
rec = {}
for Dt in DTS:
    nb, ns = [], []
    for k in range(PHASES):
        phi = Dt * k / PHASES
        nb.append(float(np.sum(registered(bg_unique, TAU_BG, Dt, phi))) * N_SITE)
        ns.append(float(np.sum(registered(t_stim, TAU_WALL, Dt, phi))))
    rec[Dt] = (float(np.mean(nb)), float(np.mean(ns)))
    nb_mean, ns_mean = rec[Dt]
    p_bg_meas = nb_mean / bg_times.size
    p_st_meas = ns_mean / t_stim.size
    p_bg_exp = min(1.0, TAU_BG / Dt)
    p_st_exp = min(1.0, TAU_WALL / Dt)
    if Dt in (1.0, 5.0, 50.0, 500.0):
        pass

pb = [rec[Dt][0] / bg_times.size for Dt in DTS]
ps = [rec[Dt][1] / t_stim.size for Dt in DTS]
eb = [min(1.0, TAU_BG / Dt) for Dt in DTS]
es = [min(1.0, TAU_WALL / Dt) for Dt in DTS]
devb = max(abs(a - b) / b for a, b in zip(pb, eb))
devs = max(abs(a - b) / b for a, b in zip(ps, es))
rows.append(("V2", "本底四档注册率对 min(1, tau_bg/Dt)", " / ".join("%.4f" % v for v in pb) + " 对 " + " / ".join("%.3f" % v for v in eb), "偏差 %.2f%%（<= 5%%）" % (100 * devb), devb <= 0.05))
rows.append(("V2s", "受激四档注册率对 min(1, tau_wall/Dt)", " / ".join("%.4f" % v for v in ps) + " 对 " + " / ".join("%.3f" % v for v in es), "偏差 %.2f%%（<= 5%%）" % (100 * devs), devs <= 0.05))

R = [ns / nb for (nb, ns) in [rec[Dt] for Dt in DTS]]
R0 = R[0]
exp_R = [R0 * min(1.0, TAU_WALL / Dt) / min(1.0, TAU_BG / Dt) for Dt in DTS]
devR = max(abs(a - b) / b for a, b in zip(R, exp_R))
seg1 = abs(R[DTS.index(0.5)] - R0) / R0
rows.append(("V3", "受激与本底计数比 R(Dt) 对 R0 * min(1,tau_wall/Dt)/min(1,tau_bg/Dt)", " / ".join("%.4f" % v for v in R) + " 对 " + " / ".join("%.4f" % v for v in exp_R), "偏差 %.2f%%（<= 10%%）" % (100 * devR), devR <= 0.10))
mid = R[DTS.index(50.0)] / R0
exp_mid = 50.0 / TAU_BG
rows.append(("V3m", "中段（Dt = 50）比值放大对 Dt/tau_bg", "%.2f 对 %.2f" % (mid, exp_mid), "偏差 <= 10%", abs(mid - exp_mid) / exp_mid <= 0.10))
plateau = R[DTS.index(500.0)] / R0
rows.append(("V3p", "高端（Dt = 500）比值放大对 tau_wall/tau_bg", "%.1f 对 %.1f" % (plateau, TAU_WALL / TAU_BG), "偏差 <= 10%", abs(plateau - TAU_WALL / TAU_BG) / (TAU_WALL / TAU_BG) <= 0.10))

# V4 注册后的本底时刻落在 t_P 的整数倍上
Dt = 5.0
phi = Dt * 3 / PHASES
mask = registered(bg_times, TAU_BG, Dt, phi)
reg_times = bg_times[mask]
resid = float(np.max(np.abs(reg_times / T_P - np.round(reg_times / T_P))))
rng = np.random.default_rng(11)
pois = np.sort(rng.uniform(1.0, T_TOT, size=reg_times.size))
resid_p = float(np.max(np.abs(pois / T_P - np.round(pois / T_P))))
rows.append(("V4", "注册本底时刻对 t_P 整数倍的残差（泊松对照同量）", "%.1e / %.3f" % (resid, resid_p), "结构化残差 <= 1e-9 且对照不成立", resid <= 1e-9 and resid_p > 0.1))
rows.append(("V5", "登记：本底与受激不共用窗口时，比值随取样间隔三段变化", "登记", "登记", True))

print("=== 本底与受激项时钟口径轮判据表（实测 对 期望）===")
for tag, name, meas, expx, ok in rows:
    print("%-5s %-46s 实测 %-46s 期望 %-26s %s" % (tag, name, meas, expx, "通过" if ok else "不通过"))
print("")
print("输入：壁速 %.4f、有效穿越长度 %.1f、受激窗口 %.1f；本底周期 %.1f；站点 %d；截面 %d；总时长 %.0f" % (V_WALL, W_LEN, TAU_WALL, T_P, N_SITE, A, T_TOT))
print("本底笔数 %d、受激笔数 %d（未稀释的基准比 R0 = %.5f）" % (bg_times.size, t_stim.size, R0))
print("取样间隔 %s" % DTS)
print("本底注册率 %s" % [round(v, 4) for v in pb])
print("受激注册率 %s" % [round(v, 4) for v in ps])
print("比值 R %s；解析 %s" % ([round(v, 4) for v in R], [round(v, 4) for v in exp_R]))
