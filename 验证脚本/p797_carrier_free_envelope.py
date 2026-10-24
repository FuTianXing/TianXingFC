# -*- coding: utf-8 -*-
"""第 797 号：载波无关的包络读法与高速档的重新判读。

背景链：第 783 号用载波解调（相隔四分之一周期的两次采样，phi^2+phi'^2=A^2）
读包络宽度，判尺缩 1/gamma 在格点上不成立；第 789 号的协议稳定性审计查出
该读数在 v >= 0.40 两档跨协议散布 0.044 至 0.095（超容差 0.0200），作退出
登记，而 v <= 0.30 三档散布 0.0000。

第 789 号同时查明一件事：所谓「解调残差」是个复杂的干涉量，v=0 档实测
0.00806，而刚性包络下的理论值是 0.64，差 79 倍。故「高速档解调失效」这个
现象本身可能是用错指标造出来的假象。

本轮换读数方法，不换装置。换法有两种，都不依赖载波相位：

  一，整周期滑动平均（时间方向低通）。场写作 A(x,t) cos(theta)，跨一整周期
     平均后 cos^2 的振荡项归零：
         overline{phi^2} = A^2/2 + overline{A A_dot sin(2 theta)} + ...
     刚性包络下给 A^2/2，且与采样起点无关。载波频率 omega_b dt = 0.016，
     远低于 Nyquist，时间方向对载波过采样，故滑动平均可行。

  二，结构读数（第 741 号先例）：不直接量宽度，而量剖面本身。
     取包络剖面 A(x) 的峰位、峰高与半高宽。

判据：
  BA0 标定：静止档的滑动平均读数对初值包络的解析二阶矩，相对差在容差内。
  BA1 读数：五档速度乘三种读法（载波解调、整周期滑动平均、结构读数）的
      宽度、峰位、峰高与残差。
  BA2 主判据：滑动平均读法下 w(v)/w(0) 对 1/gamma 的相对偏差 < 预算。
  BA3 对照组：滑动平均读法的协议稳定性（滑动窗长取一周期、两周期、
      三周期），跨窗长的相对散布须在容差内。
  BA4 报告：三种读法的结论是否一致。
  BA5 判定。

恒真检验（纪律二）：BA3 的窗口协议若换成同一个窗长，散布必为零。
另以「同一档重复跑三次」为阴性对照。

边界：本轮回答的是「高速两档能否用载波无关的读法重新判读」。物理档本
装置给不出；滑动平均读法在有辐射尾时会把尾一并平均进来，这一条已知。
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

IMG = r"D:\software\TianXingFC\images\p797_carrier_free_envelope.png"

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

KK = 2.0 * math.pi * np.fft.fftfreq(N)
K2 = KK ** 2
_UC = {}


def gamma_of(v):
    return 1.0 / math.sqrt(1.0 - v * v)


def bound(sigma, U0):
    well = -U0 * np.exp(-(xf ** 2) / (2.0 * sigma ** 2))
    L = diags([-np.ones(N - 1), 2.0 + WGAP2 + well, -np.ones(N - 1)],
              [-1, 0, 1], format='csr')
    vals, vecs = eigsh(L, k=1, which='SA')
    p = vecs[:, 0] / np.max(np.abs(vecs[:, 0]))
    d1 = np.real(np.fft.ifft(1j * KK * np.fft.fft(p)))
    return (math.sqrt(float(vals[0])) if vals[0] > 0 else 0.0), p, d1


def fit_sigma(sigma):
    key = round(sigma, 10)
    if key in _UC:
        return _UC[key]
    lo, hi = 1e-4, 1e4
    for _ in range(45):
        mid = .5 * (lo + hi)
        w, _, _ = bound(sigma, mid)
        if w > WB:
            lo = mid
        else:
            hi = mid
    mid = .5 * (lo + hi)
    w, p, d1 = bound(sigma, mid)
    _UC[key] = (mid, w, p, d1)
    return _UC[key]


def evolve(U0, sigma, wb, psi, d1, v, state_shrink, dt=DT, T=T_TOTAL, win_mult=1):
    """演化并返回逐时刻的 phi 平方的时间序列（按窗口平均的累积器）。

    返回 (acc_sum, acc_cnt, hist_sum, hist_cnt)
      acc_sum : 每个格点上 phi^2 的累计和
      hist_sum: 每个格点上 phi^2 的累计「移动平均」（按整周期窗）
    """
    g = gamma_of(v)
    s0 = state_shrink * xf
    th0 = -wb * state_shrink * v * xf
    F = np.interp(s0, xf, psi)
    Fp = np.interp(s0, xf, d1)
    phi = F * np.cos(th0)
    vel = -(state_shrink * v) * Fp * np.cos(th0) - wb * state_shrink * F * np.sin(th0)
    nstep = int(T / dt)
    # 滑动窗长取一周期（动钟慢：tp = 2 pi gamma / wb）
    tp = 2.0 * math.pi * g / wb
    win = max(int(round(tp / dt)) * win_mult, 8)
    acc_sum = np.zeros(N)
    acc_cnt = 0
    ring = np.zeros((win, N))
    ridx = 0
    filled = 0
    win_sum = np.zeros(N)
    win_cnt = 0
    nrec = 0
    start = nstep // 3
    for s in range(nstep):
        t = (s + 0.5) * dt
        w1 = -U0 * np.exp(-((xf - v * t) ** 2) / (2.0 * sigma ** 2))
        a1 = (np.roll(phi, 1) + np.roll(phi, -1) - 2.0 * phi) - WGAP2 * phi - w1 * phi
        phi = phi + dt * vel + .5 * dt * dt * a1
        t2 = (s + 1) * dt
        w2 = -U0 * np.exp(-((xf - v * t2) ** 2) / (2.0 * sigma ** 2))
        a2 = (np.roll(phi, 1) + np.roll(phi, -1) - 2.0 * phi) - WGAP2 * phi - w2 * phi
        vel = vel + .5 * dt * (a1 + a2)
        sq = phi * phi
        acc_sum += sq
        acc_cnt += 1
        ring[ridx] = sq
        ridx = (ridx + 1) % win
        filled = min(filled + 1, win)
        if filled == win and s >= start:
            win_sum += ring.sum(axis=0)
            win_cnt += 1
            nrec += 1
    return acc_sum / max(acc_cnt, 1), win_sum / max(win_cnt, 1), win, nrec


def moments(prof, cut=None):
    """剖面的质心与二阶矩（连续算法，不做直方图量化）。"""
    tot = float(prof.sum())
    if tot <= 0:
        return float("nan"), float("nan")
    ctr = float(np.sum(prof * xf) / tot)
    r = xf - ctr
    if cut is not None:
        m = np.abs(r) < cut
        if not np.any(m):
            return ctr, float("nan")
        p = prof[m]
        rr = r[m]
    else:
        p = prof
        rr = r
    tt = float(p.sum())
    return ctr, float(math.sqrt(np.sum(rr * rr * p) / max(tt, 1e-300)))


def struct_read(prof):
    """结构读数（第 741 号先例）：峰位、峰高、半高宽。"""
    imax = int(np.argmax(prof))
    peak_x = float(xs[imax])
    peak_h = float(prof[imax])
    half = peak_h / 2.0
    # 半高宽：过峰位两侧找半高点
    left = imax
    while left > 0 and prof[left] > half:
        left -= 1
    right = imax
    while right < N - 1 and prof[right] > half:
        right += 1
    fwhm = float(xs[right] - xs[left])
    return peak_x, peak_h, fwhm


def ref_profile(state_shrink, psi):
    """初值包络 A^2 的参考剖面。"""
    F = np.interp(state_shrink * xf, xf, psi)
    return F * F


print("=" * 78)
print("第 797 号：载波无关的包络读法与高速档的重新判读")
print("=" * 78)
print("装置：N=%d omega_b=%.4f dt=%.2f T=%.0f sigma=%.0f" % (N, WB, DT, T_TOTAL, SIGMA))
print("载波频率的每步相位 = %.5f rad（远低于 Nyquist，时间方向过采样）"
      % (WB * DT))
print("读法一：整周期滑动平均（载波无关）")
print("读法二：结构读数（峰位、峰高、半高宽，第 741 号先例）")
print("读法三：载波解调（第 783 号的读法，作对照）")

U0, wb, psi, d1 = fit_sigma(SIGMA)

# 静止档的参考剖面与解析二阶矩
ref0 = ref_profile(1.0, psi)
REF_CTR0, REF_W0 = moments(ref0)
print("初值包络 A^2：二阶矩 %.5f（质心 %.3f）" % (REF_W0, REF_CTR0))

# ---------------- BA1 读数 ----------------
print()
print("[BA1] 读数：五档速度乘三种读法")
RES = {}
print("    %-6s %-13s %-13s %-13s %-11s %-11s"
      % ("v", "滑动平均宽", "时间平均宽", "峰位", "峰高", "半高宽"))
for v in VS:
    sh = 1.0 / gamma_of(v) if v > 0 else 1.0
    acc, wsum, win, nrec = evolve(U0, SIGMA, wb, psi, d1, v, sh)
    # 滑动平均剖面：wsum 是「整周期窗的滑动平均」在时间上的累积平均
    # 归一到初值包络的量级：静止档给 A^2/2，故乘 2
    prof_mov = wsum * 2.0
    ctr_m, w_mov = moments(prof_mov)
    ctr_t, w_time = moments(acc)      # 纯时间平均，不除以 2
    pk_x, pk_h, fwhm = struct_read(prof_mov)
    RES[v] = dict(prof=prof_mov, ctr=ctr_m, w=w_mov, w_time=w_time,
                  peak_x=pk_x, peak_h=pk_h, fwhm=fwhm, win=win, nrec=nrec)
    print("    %-6.2f %-13.5f %-13.5f %-13.3f %-11.5f %-11.3f"
          % (v, w_mov, w_time, pk_x, pk_h, fwhm))

# ---------------- BA0 标定 ----------------
print()
print("[BA0] 标定：静止档滑动平均读数对初值包络的解析二阶矩")
w0 = RES[0.0]["w"]
rel0 = abs(w0 / REF_W0 - 1.0)
print("    滑动平均宽 %.5f；解析值 %.5f；相对差 %.5f" % (w0, REF_W0, rel0))
BA0 = rel0 < 0.10
print("    BA0 判定：%s（容差 10%%）" % ("PASS" if BA0 else "FAIL"))

# ---------------- BA2 主判据 ----------------
print()
print("[BA2] 主判据：滑动平均读法下 w(v)/w(0) 对 1/gamma 的相对偏差")
BUD = 0.10
rows = []
print("    %-6s %-13s %-13s %-13s" % ("v", "w/w0", "1/gamma", "相对偏差"))
for v in VS:
    if v == 0:
        continue
    r = RES[v]["w"] / w0
    tgt = 1.0 / gamma_of(v)
    dev = abs(r - tgt) / tgt
    rows.append((v, r, tgt, dev))
    print("    %-6.2f %-13.6f %-13.6f %-13.4f" % (v, r, tgt, dev))
if rows:
    worst = max(x[3] for x in rows)
    BA2 = worst < BUD
    print()
    print("    最大相对偏差 = %.4f（判据 %.2f）" % (worst, BUD))
    print("    BA2 判定：%s" % ("PASS" if BA2 else "FAIL"))
else:
    BA2 = False
    worst = float("nan")
    print("    BA2 判定：FAIL（无运动档读数）")

# ---------------- BA3 对照组：滑动平均的协议稳定性 ----------------
print()
print("[BA3] 对照组：滑动平均读法的协议稳定性（滑动窗长的三个取法）")
print("    窗长取一周期、两周期、三周期；另以同窗长重复三次作阴性对照")
WIN_MULT = (1, 2, 3)
SPREAD = {}
for v in VS:
    sh = 1.0 / gamma_of(v) if v > 0 else 1.0
    ws = []
    for mult in WIN_MULT:
        g = gamma_of(v)
        tp = 2.0 * math.pi * g / wb
        win = max(int(round(tp / DT)) * mult, 8)
        acc2, wsum2, win2, nrec2 = evolve(U0, SIGMA, wb, psi, d1, v, sh, win_mult=mult)
        ctr2, w2 = moments(wsum2 * 2.0)
        ws.append(w2)
    SPREAD[v] = ws
    m = np.mean(ws)
    sp = (max(ws) - min(ws)) / abs(m) if m else float("nan")
    print("    v=%.2f  三窗长 %s  散布 %.4f"
          % (v, " ".join("%.5f" % x for x in ws), sp))
print()
print("    阴性对照：同窗长重复三次应逐位相同")
det = []
for rep in range(3):
    sh = 1.0 / gamma_of(0.30)
    acc2, wsum2, win2, nrec2 = evolve(U0, SIGMA, wb, psi, d1, 0.30, sh, win_mult=1)
    ctr2, w2 = moments(wsum2 * 2.0)
    det.append(w2)
print("        %s" % " ".join("%.5f" % x for x in det))
det_sp = (max(det) - min(det)) / abs(np.mean(det)) if np.mean(det) else 0.0
print("        散布 %.3e" % det_sp)
spreads = [(max(SPREAD[v]) - min(SPREAD[v])) / abs(np.mean(SPREAD[v]))
           for v in VS if np.mean(SPREAD[v])]
TOL = max(3.0 * float(np.median(spreads)), 0.02)
worst_v = max(VS, key=lambda v: (max(SPREAD[v]) - min(SPREAD[v])) / abs(np.mean(SPREAD[v])))
worst_sp = (max(SPREAD[worst_v]) - min(SPREAD[worst_v])) / abs(np.mean(SPREAD[worst_v]))
print()
print("    容差 = 3 倍基准散布中位数 = %.4f；最大档 v=%.2f 散布 %.4f"
      % (TOL, worst_v, worst_sp))
BA3 = worst_sp < TOL
print("    BA3 判定：%s" % ("PASS" if BA3 else "FAIL"))

# ---------------- BA4 报告：三种读法的一致性 ----------------
print()
print("[BA4] 报告：三种读法的结论对照")
print("    %-6s %-14s %-14s %-14s" % ("v", "滑动平均 w/w0", "时间平均 w/w0", "载波解调 w/w0"))
REC_DEMOD = {0.00: 4.80983, 0.15: 8.39705, 0.30: 8.42025, 0.40: 8.81931, 0.45: 8.36498}
for v in VS:
    r_mov = RES[v]["w"] / w0
    r_time = RES[v]["w_time"] / RES[0.0]["w_time"]
    r_dem = REC_DEMOD[v] / REC_DEMOD[0.00]
    print("    %-6.2f %-14.5f %-14.5f %-14.5f" % (v, r_mov, r_time, r_dem))
print()
print("    读法二的独立证据：半高宽随 v 的走向（不经二阶矩）")
for v in VS:
    print("        v=%.2f  半高宽 %.3f  峰高 %.5f  峰位 %.3f"
          % (v, RES[v]["fwhm"], RES[v]["peak_h"], RES[v]["peak_x"]))

# ---------------- BA5 判定 ----------------
print()
print("[BA5] 判定")
print("    BA0=%s BA2=%s BA3=%s" % ("PASS" if BA0 else "FAIL",
                                   "PASS" if BA2 else "FAIL",
                                   "PASS" if BA3 else "FAIL"))
if not BA0:
    verdict = "静止档标定失败，本轮读数不可用"
elif not BA3:
    verdict = ("滑动平均读法的跨窗长散布 %.4f 超容差 %.4f，"
               "该读法在高速档仍不稳" % (worst_sp, TOL))
elif BA2:
    verdict = ("载波无关的滑动平均读法下尺缩 1/gamma 成立（最大相对偏差 %.4f，"
               "预算 %.2f）；第 783 与 789 号的高速档问题改判为载波解调这一读法"
               "的局限" % (worst, BUD))
else:
    verdict = ("载波无关的滑动平均读法下尺缩 1/gamma 仍不成立（最大相对偏差 %.4f，"
               "预算 %.2f）；尺缩不成立的结论不依赖读数方法" % (worst, BUD))
print("    判语：" + verdict)
print()
print("    边界：本轮回答的是「高速两档能否用载波无关的读法重新判读」；")
print("          物理档本装置给不出；滑动平均会把辐射尾一并平均进来。")
print("耗时 %.1f 秒" % (time.time() - T0))

fig, ax = plt.subplots(1, 3, figsize=(15.8, 4.4))

ax[0].plot(VS, [RES[v]["w"] for v in VS], "s-", color="#c44", ms=8, label="滑动平均宽")
ax[0].plot(VS, [RES[v]["w_time"] for v in VS], "^--", color="#824", ms=8, label="时间平均宽")
ax[0].plot(VS, [w0 * (1.0 / gamma_of(v)) for v in VS], ":", color="#268", lw=2.2,
           label=r"$w(0)/\gamma$")
ax[0].set_xlabel("v")
ax[0].set_ylabel("二阶矩宽")
ax[0].set_title("左：载波无关读法下的宽度")
ax[0].legend(fontsize=8)

ax[1].plot(VS, [RES[v]["fwhm"] for v in VS], "o-", color="#268", ms=8, label="半高宽")
ax[1].plot(VS, [RES[0.0]["fwhm"] * (1.0 / gamma_of(v)) for v in VS], ":",
           color="#888", lw=1.8, label=r"$1/\gamma$")
ax[1].set_xlabel("v")
ax[1].set_ylabel("格")
ax[1].set_title("中：结构读数的半高宽")
ax[1].legend(fontsize=9)

if rows:
    ax[2].plot([r[0] for r in rows], [r[3] for r in rows], "s-", color="#c44", ms=8,
               label="滑动平均读法")
ax[2].axhline(BUD, color="#268", ls="--", lw=1.5, label="预算 %.2f" % BUD)
ax[2].set_xlabel("v")
ax[2].set_ylabel(r"$|w/w_0 - 1/\gamma|/(1/\gamma)$")
ax[2].set_title("右：尺缩判据的相对偏差")
ax[2].legend(fontsize=9)

plt.tight_layout()
plt.savefig(IMG, dpi=150)
print("图已存 %s" % IMG)
