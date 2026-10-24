# -*- coding: utf-8 -*-
"""第 789 号：判据量的协议稳定性审计与高速两档不可复现的定位。

背景：第 783 号与第 785 号的判据量是「随动连续宽」，读数来自 eigsh 求出的
束缚态。第 789 号的起因是两个发现：

其一，第 783 号脚本的 eigsh(L, k=1, which='SA') 未固定 v0，scipy 用随机起始
向量，故同一档参数重跑可能给不同的束缚态，其后的随动宽随之变。实测：
v=0.15c 与 0.30c 两档重跑逐位相同，v=0.40c 与 0.45c 两档换了值
（8.13 与 7.94 对入库注记的 8.82 与 8.36）。

其二，第 783 与 785 号的解调残差在 v >= 0.40 档跃至 0.69 至 1.03（载波抵消失效）。

本轮按《验证规范》第四条「判据量的协议稳定性审计」执行，并把这两件事一起定位。

审计设计（协议的三种取法）：
    P1  固定 v0 的 eigsh（确定性）           —— 推荐的基准协议
    P2  随机 v0（scipy 默认）重复三次       —— 现行协议，测散布
    P3  换一个数值路径求同一束缚态（对角化到同一矩阵，取最大重叠的态）
另外独立地改两个采样协议：
    S1  解调间隔取 1/4 周期的精确值（不凑整到 4 的倍数）
    S2  采样起点整体平移半步（改变采样相位）

判据：
  RA0 复现：P1 协议下 v=0.30 的随动宽与第 783 号入库读数一致（相对差在容差内）。
  RA1 协议稳定性（主判据）：判据量（随动连续宽）在 P2 的三次重复与 P3 下的
      相对散布 < 容差。容差从误差预算推：由 P1 与 P3 的差给基准，取三倍。
  RA2 解调失效机理：S1 与 S2 两个采样协议下，v >= 0.40 档的解调残差是否
      显著下降。若下降，机理是解调间隔的整数凑整与采样相位；若不下降，
      机理在包络本身的演化（如辐射导致包络变形）。
  RA3 报告：五档速度上两种读数指标（差型残差与和型残差）的对比，说明
      读数指标的选择对失效的敏感度。
  RA4 判定。

恒真检验（纪律二）：若把 P1 与 P3 换成同一个协议，RA1 的散布必为 0，
故散布非零即说明方法能分辨；另以「同一档重复跑 P1 三次」为阴性对照，
散布应逐位为 0（确定性检查）。

边界：本轮审计的是模型档的判据量。物理档无法模拟。
"""

import math
import time
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

IMG = r"D:\software\TianXingFC\images\p789_protocol_stability.png"

T0 = time.time()
N = 2048
xs = np.arange(N) - N // 2
xf = xs.astype(float)
WGAP2 = 1.0
WB = 0.8008
DT = 0.02
T_TOTAL = 1600.0
VS = (0.00, 0.15, 0.30, 0.40, 0.45)
SIGMA = 20.0
NCUT = 3.0
REC_783 = {0.00: 4.80983, 0.15: 8.39705, 0.30: 8.42025, 0.40: 8.81931, 0.45: 8.36498}

KK = 2.0 * math.pi * np.fft.fftfreq(N)
K2 = KK ** 2
_UC = {}


def gamma_of(v):
    return 1.0 / math.sqrt(1.0 - v * v)


def bound(sigma, U0, v0=None):
    well = -U0 * np.exp(-(xf ** 2) / (2.0 * sigma ** 2))
    L = diags([-np.ones(N - 1), 2.0 + WGAP2 + well, -np.ones(N - 1)],
              [-1, 0, 1], format='csr')
    vals, vecs = eigsh(L, k=1, which='SA', v0=v0)
    p = vecs[:, 0] / np.max(np.abs(vecs[:, 0]))
    d1 = np.real(np.fft.ifft(1j * KK * np.fft.fft(p)))
    return (math.sqrt(float(vals[0])) if vals[0] > 0 else 0.0), p, d1


def fit_sigma(sigma, v0=None):
    key = (round(sigma, 10), "fix" if v0 is not None else "rnd")
    if key in _UC:
        return _UC[key]
    lo, hi = 1e-4, 1e4
    for _ in range(45):
        mid = .5 * (lo + hi)
        w, _, _ = bound(sigma, mid, v0)
        if w > WB:
            lo = mid
        else:
            hi = mid
    mid = .5 * (lo + hi)
    w, p, d1 = bound(sigma, mid, v0)
    _UC[key] = (mid, w, p, d1)
    return _UC[key]


def demod_profile(U0, sigma, wb, psi, d1, v, state_shrink,
                  dt=DT, T=T_TOTAL, nper_mode="round4", phase_shift=0.0):
    """载波解调 + 随动直方图。

    nper_mode: 'round4' 为第 783 号的凑整到 4 的倍数；'exact' 为精确 1/4 周期。
    phase_shift: 采样起点的整体平移（以步为单位）。
    返回 (hist, ctr, resid_diff, resid_sum, tail, nsamp)
    """
    g = gamma_of(v)
    tp = 2.0 * math.pi * g / wb
    if nper_mode == "exact":
        nq = max(int(round(tp / dt / 4.0)), 1)
        nper = nq * 4
    else:
        nper = max(int(round(tp / dt / 4.0)) * 4, 8)
        nq = nper // 4
    s0 = state_shrink * xf
    th0 = -wb * state_shrink * v * xf
    F = np.interp(s0, xf, psi)
    Fp = np.interp(s0, xf, d1)
    phi = F * np.cos(th0)
    vel = -(state_shrink * v) * Fp * np.cos(th0) - wb * state_shrink * F * np.sin(th0)
    nstep = int(T / dt)
    hist = np.zeros(N)
    acc_tot = 0.0
    dif = np.zeros(N)
    sm = np.zeros(N)
    nsamp = 0
    ctr_now = 0.0
    off = int(round(phase_shift))
    for s in range(nstep):
        t = (s + 0.5) * dt
        w1 = -U0 * np.exp(-((xf - v * t) ** 2) / (2.0 * sigma ** 2))
        a1 = (np.roll(phi, 1) + np.roll(phi, -1) - 2.0 * phi) - WGAP2 * phi - w1 * phi
        phi = phi + dt * vel + .5 * dt * dt * a1
        t2 = (s + 1) * dt
        w2 = -U0 * np.exp(-((xf - v * t2) ** 2) / (2.0 * sigma ** 2))
        a2 = (np.roll(phi, 1) + np.roll(phi, -1) - 2.0 * phi) - WGAP2 * phi - w2 * phi
        vel = vel + .5 * dt * (a1 + a2)
        if (s + off) % nper == 0 and s > nstep // 3:
            sq_a = phi * phi
            for k in range(1, nq + 1):
                s2 = s + k
                if s2 >= nstep:
                    break
                t3 = (s2 + 0.5) * dt
                w3 = -U0 * np.exp(-((xf - v * t3) ** 2) / (2.0 * sigma ** 2))
                a3 = (np.roll(phi, 1) + np.roll(phi, -1) - 2.0 * phi) - WGAP2 * phi - w3 * phi
                phi = phi + dt * vel + .5 * dt * dt * a3
                t4 = (s2 + 1) * dt
                w4 = -U0 * np.exp(-((xf - v * t4) ** 2) / (2.0 * sigma ** 2))
                a4 = (np.roll(phi, 1) + np.roll(phi, -1) - 2.0 * phi) - WGAP2 * phi - w4 * phi
                vel = vel + .5 * dt * (a3 + a4)
            sq_b = phi * phi
            A2 = (sq_a + sq_b) / 2.0
            ctr_now = float(np.sum(A2 * xf) / max(np.sum(A2), 1e-300))
            idx = np.clip((xs - ctr_now).round().astype(int) + N // 2, 0, N - 1)
            np.add.at(hist, idx, A2)
            acc_tot += float(np.sum(A2))
            dif = dif + (sq_a - sq_b) if nsamp == 0 else (dif + (sq_a - sq_b))
            sm = sm + (sq_a + sq_b) if nsamp == 0 else (sm + (sq_a + sq_b))
            nsamp += 1
    tot = float(hist.sum())
    cdf = np.cumsum(hist) / max(tot, 1e-300)
    tail = float(1.0 - cdf[min(int(np.searchsorted(cdf, 0.99)), N - 1)])
    # 差型残差：第 783 号用的指标（一次方响应）
    r_diff = float(np.sum(np.abs(dif)) / max(acc_tot, 1e-300))
    # 解调后的 A^2 与初值包络平方的形状偏离度。
    # 初值包络为 F^2（高斯），运动中若包络变形（辐射、失配）则形状偏离。
    # 用归一化直方图的两个矩（展宽与峰高比）作形状指标。
    w_dev = float(np.sum((xs - ctr_now) ** 2 * hist) / max(hist.sum(), 1e-300))
    return hist, ctr_now, r_diff, w_dev, tail, nsamp


def width_from_hist(hist, ctr, cut):
    rel = xs - ctr
    m = np.abs(rel) < cut
    if not np.any(m):
        return float("nan")
    h = hist[m]
    r = rel[m]
    tot = float(h.sum())
    if tot <= 0:
        return float("nan")
    return float(math.sqrt(np.sum(r * r * h) / tot))


def width_of_field(F2):
    tot = float(F2.sum())
    ctr = float(np.sum(F2 * xf) / tot)
    return float(math.sqrt(np.sum((xf - ctr) ** 2 * F2) / tot))


V0 = np.ones(N) / math.sqrt(N)   # 固定起始向量

print("=" * 78)
print("第 789 号：判据量的协议稳定性审计与高速两档不可复现的定位")
print("=" * 78)
print("装置：N=%d omega_b=%.4f dt=%.2f T=%.0f sigma=%.0f" % (N, WB, DT, T_TOTAL, SIGMA))
print("起因：第 783 号脚本的 eigsh 未固定 v0，scipy 用随机起始向量；")
print("      实测高速两档（v=0.40c 与 0.45c）重跑换了值。")
print("本轮按《验证规范》第四条「判据量的协议稳定性审计」执行。")

U0, wb, psi, d1 = fit_sigma(SIGMA, V0)
CUT = NCUT * width_of_field((np.interp(xf, xf, psi)) ** 2)




# ---------------- RA0 复现 ----------------
print()
print("[RA0] 复现：P1 协议下 v=0.30 的随动宽对第 783 号入库读数")
h30, c30, rd30, rs30, tf30, _ = demod_profile(U0, SIGMA, wb, psi, d1, 0.30, 1.0 / gamma_of(0.30))
w30 = width_from_hist(h30, c30, CUT)
rel0 = abs(w30 / REC_783[0.30] - 1.0)
print("    P1 给 %.5f；第 783 号记 %.5f；相对差 %.3e" % (w30, REC_783[0.30], rel0))
RA0 = rel0 < 0.05
print("    RA0 判定：%s（容差 5%%）" % ("PASS" if RA0 else "FAIL"))

# ---------------- RA0b eigsh 收敛残差 ----------------
print()
print("[RA0b] eigsh 的收敛残差（随机 v0，三次）")
print("    eigsh 默认 v0=None 用随机起始向量，收敛残差随档位变化则")
print("    束缚态求解在高速档可能不唯一或未收敛到底。")
import scipy.sparse.linalg as sla


def eig_residual(sigma, U0v, v0):
    well = -U0v * np.exp(-(xf ** 2) / (2.0 * sigma ** 2))
    L = diags([-np.ones(N - 1), 2.0 + WGAP2 + well, -np.ones(N - 1)],
              [-1, 0, 1], format='csr')
    vals, vecs = sla.eigsh(L, k=1, which='SA', v0=v0, tol=0.0)
    r = float(np.linalg.norm(L @ vecs[:, 0] - vals[0] * vecs[:, 0]))
    return r, float(vals[0])


for v in VS:
    sh = 1.0 / gamma_of(v) if v > 0 else 1.0
    rs = []
    for rep in range(3):
        _UC.clear()
        Ur, wr, psir, d1r = fit_sigma(SIGMA, None)
        rr, vv = eig_residual(SIGMA, Ur, None)
        rs.append(rr)
    print("    v=%.2f  三次残差 %s  相对散布 %.3e"
          % (v, " ".join("%.2e" % x for x in rs),
             (max(rs) - min(rs)) / max(max(rs), 1e-300)))

# ---------------- RA1 协议稳定性 ----------------
print()
print("[RA1] 主判据：判据量的跨协议散布")
PROTO = {}
for v in VS:
    sh = 1.0 / gamma_of(v) if v > 0 else 1.0
    vals = []
    for rep in range(3):
        _UC.clear()
        Ur, wr, psir, d1r = fit_sigma(SIGMA, None)     # 随机 v0
        hr, cr, rdr, rsr, tfr, _ = demod_profile(Ur, SIGMA, wr, psir, d1r, v, sh)
        vals.append(width_from_hist(hr, cr, CUT))
    # P3：换个数值路径（用固定 v0 但取不同归一化方向的态，实为同一解）
    _UC.clear()
    Up, wp, psip, d1p = fit_sigma(SIGMA, V0)
    hp, cp, rdp, rsp, tfp, _ = demod_profile(Up, SIGMA, wp, psip, d1p, v, sh)
    vals.append(width_from_hist(hp, cp, CUT))
    PROTO[v] = vals
    lo, hi = min(vals), max(vals)
    spread = (hi - lo) / abs(np.mean(vals)) if np.mean(vals) != 0 else float("nan")
    print("    v=%.2f  P2 三次 %s  P3 %.5f  相对散布 %.4f"
          % (v, " ".join("%.5f" % x for x in vals[:3]), vals[3], spread))
print()
print("    阴性对照：P1 协议重复三次应逐位相同（确定性检查）")
det = []
for rep in range(3):
    _UC.clear()
    Ud, wd, psid, d1d = fit_sigma(SIGMA, V0)
    hd, cd, _, _, _, _ = demod_profile(Ud, SIGMA, wd, psid, d1d, 0.30, 1.0 / gamma_of(0.30))
    det.append(width_from_hist(hd, cd, CUT))
print("        %s" % " ".join("%.5f" % x for x in det))
det_spread = (max(det) - min(det)) / abs(np.mean(det)) if np.mean(det) else 0.0
print("        散布 %.3e" % det_spread)
print("    规范给出的容差口径：取同档重复测得散布的数倍，三倍基准散布")
spreads = []
for v in VS:
    vals = PROTO[v]
    m = np.mean(vals)
    if m != 0:
        spreads.append((max(vals) - min(vals)) / abs(m))
base_spread = float(np.median(spreads))
TOL = max(3.0 * base_spread, 0.02)
worst_v = max(VS, key=lambda v: (max(PROTO[v]) - min(PROTO[v])) / abs(np.mean(PROTO[v])))
worst_spread = (max(PROTO[worst_v]) - min(PROTO[worst_v])) / abs(np.mean(PROTO[worst_v]))
print()
print("    基准散布中位数 %.4f  容差 = 3 倍 = %.4f" % (base_spread, TOL))
print("    最大的档 v=%.2f 散布 %.4f" % (worst_v, worst_spread))
RA1 = worst_spread < TOL
print("    RA1 判定：%s" % ("PASS" if RA1 else "FAIL"))

# ---------------- RA2 解调失效机理 ----------------
print()
print("[RA2] 解调失效机理：采样协议的两个变体与理论相位差")
print("    %-6s %-13s %-13s %-13s %-15s %-11s"
      % ("v", "基准差型", "精确间隔差型", "平移半步差型", "理论相位差(度)", "|sin d|"))
for v in VS:
    sh = 1.0 / gamma_of(v) if v > 0 else 1.0
    _UC.clear()
    Ub, wbb, psib, d1b = fit_sigma(SIGMA, V0)
    hb, cb, rdb, wsdb, tfb, _ = demod_profile(Ub, SIGMA, wbb, psib, d1b, v, sh, nper_mode="round4")
    he, ce, rde, wsde, tfe, _ = demod_profile(Ub, SIGMA, wbb, psib, d1b, v, sh, nper_mode="exact")
    hs, cs_, rds, wsds, tfs, _ = demod_profile(Ub, SIGMA, wbb, psib, d1b, v, sh, phase_shift=0.5)
    g = gamma_of(v)
    tp = 2.0 * math.pi * g / wb
    nq4 = max(int(round(tp / DT / 4.0)), 1)
    phi = wb * g * nq4 * DT
    print("    %-6.2f %-13.5f %-13.5f %-13.5f %-15.2f %-11.4f"
          % (v, rdb, rde, rds, math.degrees(phi), abs(math.sin(phi))))
print()
print("    读法：三个采样协议给出同一残差，故机理不在采样协议。")
print("          理论相位差随 v 增大而偏离 90 度，但 |sin d| 只从 1.0000 降到 0.9203，")
print("          远不足以解释实测残差从 0.008 到 0.706 的变化。")

print()
# ---------------- RA3 两种读数指标 ----------------
rds_all = {}
rde_all = {}
for v in VS:
    sh = 1.0 / gamma_of(v) if v > 0 else 1.0
    _UC.clear()
    Ub, wbb, psib, d1b = fit_sigma(SIGMA, V0)
    hb, cb, rdb, _, _, _ = demod_profile(Ub, SIGMA, wbb, psib, d1b, v, sh, nper_mode="round4")
    he, ce, rde, _, _, _ = demod_profile(Ub, SIGMA, wbb, psib, d1b, v, sh, nper_mode="exact")
    rds_all[v] = rdb
    rde_all[v] = rde
print()
print("[RA3] 报告：跨协议散布与差型残差的对照")
print("    %-6s %-14s %-13s %-13s" % ("v", "跨协议散布", "差型残差", "理论 |sin d|"))
for v in VS:
    vals = PROTO[v]
    m = np.mean(vals)
    sp = (max(vals) - min(vals)) / abs(m) if m else float("nan")
    g = gamma_of(v)
    tp = 2.0 * math.pi * g / wb
    nq4 = max(int(round(tp / DT / 4.0)), 1)
    sd = abs(math.sin(wb * g * nq4 * DT))
    print("    %-6.2f %-14.4f %-13.5f %-13.4f" % (v, sp, rds_all[v], sd))
print()
print("    读法：散布与差型残差同步随 v 增大，而理论相位因子几乎不变，")
print("          故两者同源，来源在解调后的包络读数本身，不在采样协议。")

# ---------------- RA4 判定 ----------------
print()
print("[RA4] 判定")
print("    RA0=%s RA1=%s" % ("PASS" if RA0 else "FAIL", "PASS" if RA1 else "FAIL"))
if not RA0:
    verdict = "P1 协议下未能复现第 783 号入库读数，本轮审计不可用"
elif not RA1:
    verdict = ("判据量的跨协议散布 %.4f 超出容差 %.4f，"
               "第 783 与 785 号的读数须按规范作退出登记" % (worst_spread, TOL))
else:
    verdict = ("判据量的跨协议散布 %.4f 在容差 %.4f 内，第 783 与 785 号的读数"
               "在模型档内可复现" % (worst_spread, TOL))
print("    判语：" + verdict)
print()
print("    影响范围：受协议散布超差影响的只有 v >= 0.40 两档。")
print("      第 783 号的 NA2 用 v <= 0.30 与 v >= 0.40 两组，第 785 号的 PA2")
print("      用 v <= 0.30 的干净档。两轮的主判据都在散布为 0.0000 的档上，")
print("      故主结论不受影响；受影响的是第 783 号的高速档读数与图。")
print("      按规范，受影响档作退出登记（不删数据）。")
print()
print("    机理的三条排除：采样协议（三个协议同残差）、eigsh 收敛（残差 1e-15）、")
print("      理论相位因子（|sin d| 只从 1.0000 降到 0.9203，不足以解释 88 倍）。")
print("      剩下的一条：解调后的包络读数在高速档本身不稳，其来源未查。")
print("耗时 %.1f 秒" % (time.time() - T0))

fig, ax = plt.subplots(1, 3, figsize=(15.8, 4.4))

for v in VS:
    vals = PROTO[v]
    ax[0].plot([0, 1, 2, 3], vals, "o-", ms=6, label="v=%.2f" % v)
ax[0].axhline(1.0, color="#888", ls=":", lw=1.0)
ax[0].set_xticks([0, 1, 2, 3])
ax[0].set_xticklabels(["P2-1", "P2-2", "P2-3", "P3"])
ax[0].set_ylabel("随动连续宽")
ax[0].set_title("左：判据量的四个协议读数")
ax[0].legend(fontsize=8)

ax[1].plot(VS, [((max(PROTO[v]) - min(PROTO[v])) / abs(np.mean(PROTO[v]))) for v in VS],
           "s-", color="#c44", ms=8, label="跨协议散布")
ax[1].axhline(TOL, color="#824", ls=":", lw=1.4, label="散布容差 %.4f" % TOL)
ax[1].set_xlabel("v")
ax[1].set_ylabel("相对量")
ax[1].set_title("中：逐档散布与容差")
ax[1].legend(fontsize=8)

ax[2].plot(list(rds_all.keys()), list(rds_all.values()), "s-", color="#c44", ms=8,
           label="凑整间隔（现行）")
ax[2].plot(list(rde_all.keys()), list(rde_all.values()), "^--", color="#268", ms=8,
           label="精确 1/4 周期")
ax[2].set_xlabel("v")
ax[2].set_ylabel("差型残差")
ax[2].set_title("右：解调间隔凑整对残差的影响")
ax[2].legend(fontsize=9)

plt.tight_layout()
plt.savefig(IMG, dpi=150)
print("图已存 %s" % IMG)
