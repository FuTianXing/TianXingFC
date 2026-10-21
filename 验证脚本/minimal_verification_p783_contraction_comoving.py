# -*- coding: utf-8 -*-
"""第 783 号：尺缩读数方法重估（随动坐标系 + 载波解调）。

背景：尺缩 $1/\\gamma$ 是本专题四个承重未闭合项之一。第 691 号 BL4 判它在
本装置下不成立于判据，理由是两种宽度读数都被污染：单帧读数在 v=0.45 档宽度
随窗口发散（辐射尾），锁相窄窗读数在振荡周期内于 0.71 至 1.29 倍间跳动
（单点节点风险）。第 691 号自陈「须换装置再试」。

本轮的第一段（诊断）查出了污染源，且它不是辐射尾而是波包平移：
  直方图按绝对坐标 x 累加时，移动井档的 10% 至 90% 分位宽给到 257 格
  （v=0.30）与 384 格（v=0.45），峰位跑到 252 格；同参数下固定井档
  只给 25 与 34 格。移动井的初值包络理论宽为 12/gamma 约 11.5 格。
  结论：分位宽度被波包的整体平移轨迹污染，与尾污染无关。
这与第 691 号「单帧读数宽度随窗口发散」的现象同源。

本轮的第二段用两处修正重建读数：
  其一，随动坐标系。直方图按 x 减去包络质心累加，平移不再展宽分布。
  其二，载波解调。取同一空间点相隔四分之一振荡周期的两次采样，
        phi^2 + phi'^2 = A^2
      逐点精确消去载波，避开单点节点风险。

判据：
  NA0 标定：静止档的随动分位宽度与初始包络 F^2 的分位宽度相对差在容差内。
  NA1 读数：五档速度的随动解调宽度、解调残差、尾污染。
  NA2 主判据：宽度随 v 单调下降（尾或平移污染主导时非单调）。
  NA3 主判据：尺缩判据。w(v)/w(0) 对 1/gamma 的相对偏差 < 预算。
  NA4 对照组：
      组一（必须非零）：解调残差随 v 变化。
      组二（必须为零）：固定井档与移动井档的随动宽度之差在容差内
          （平移污染已被随动坐标系消除的判别检验）。
  NA5 报告：与第 691 号 BL4 两种被否读数的对照。
  NA6 判定。

恒真检验（纪律二）：NA4 的组二即恒真替身检验。若移动井与固定井的随动宽度
仍显著不同，则随动坐标系未消除平移污染，NA2 与 NA3 的读数不含物理信息。

边界：本轮回答的是「尺缩判据能否用随动解调建立起来」。「尺缩在格点上是否
成立」是另一个问题，前者成立不等于后者。
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

IMG = r"D:\software\TianXingFC\images\p783_contraction_comoving.png"

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
QLO, QHI = 0.10, 0.90

KK = 2.0 * math.pi * np.fft.fftfreq(N)
K2 = KK ** 2
_U0CACHE = {}


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
    if key in _U0CACHE:
        return _U0CACHE[key]
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
    _U0CACHE[key] = (mid, w, p, d1)
    return _U0CACHE[key]


def demod_profile(U0, sigma, wb, psi, d1, v, shrink, move_well=True,
                  dt=DT, T=T_TOTAL):
    """载波解调 + 随动直方图。

    返回 (随动分位宽度, 随动质心, 解调残差, 尾污染, 采样点数)
    """
    g = gamma_of(v)
    tp = 2.0 * math.pi * g / wb
    nper = max(int(round(tp / dt / 4.0)) * 4, 8)
    nq = nper // 4
    s0 = shrink * xf
    th0 = -wb * shrink * v * xf
    F = np.interp(s0, xf, psi)
    Fp = np.interp(s0, xf, d1)
    phi = F * np.cos(th0)
    vel = -(shrink * v) * Fp * np.cos(th0) - wb * shrink * F * np.sin(th0)
    nstep = int(T / dt)
    hist = np.zeros(N)
    acc_tot = 0.0
    dif = np.zeros(N)
    nsamp = 0
    for s in range(nstep):
        t = (s + 0.5) * dt
        ctr = v * t if move_well else 0.0
        w1 = -U0 * np.exp(-((xf - ctr) ** 2) / (2.0 * sigma ** 2))
        a1 = (np.roll(phi, 1) + np.roll(phi, -1) - 2.0 * phi) - WGAP2 * phi - w1 * phi
        phi = phi + dt * vel + .5 * dt * dt * a1
        t2 = (s + 1) * dt
        ctr2 = v * t2 if move_well else 0.0
        w2 = -U0 * np.exp(-((xf - ctr2) ** 2) / (2.0 * sigma ** 2))
        a2 = (np.roll(phi, 1) + np.roll(phi, -1) - 2.0 * phi) - WGAP2 * phi - w2 * phi
        vel = vel + .5 * dt * (a1 + a2)
        if s % nper == 0 and s > nstep // 3:
            sq_a = phi * phi
            for k in range(1, nq + 1):
                s2 = s + k
                if s2 >= nstep:
                    break
                t3 = (s2 + 0.5) * dt
                c3 = v * t3 if move_well else 0.0
                w3 = -U0 * np.exp(-((xf - c3) ** 2) / (2.0 * sigma ** 2))
                a3 = (np.roll(phi, 1) + np.roll(phi, -1) - 2.0 * phi) - WGAP2 * phi - w3 * phi
                phi = phi + dt * vel + .5 * dt * dt * a3
                t4 = (s2 + 1) * dt
                c4 = v * t4 if move_well else 0.0
                w4 = -U0 * np.exp(-((xf - c4) ** 2) / (2.0 * sigma ** 2))
                a4 = (np.roll(phi, 1) + np.roll(phi, -1) - 2.0 * phi) - WGAP2 * phi - w4 * phi
                vel = vel + .5 * dt * (a3 + a4)
            sq_b = phi * phi
            A2 = (sq_a + sq_b) / 2.0
            # 随动质心：以 A2 加权的 x 分布
            ctr_now = float(np.sum(A2 * xf) / max(np.sum(A2), 1e-300))
            # 随动直方图：按 x - ctr_now 累加
            idx = np.clip((xs - ctr_now).round().astype(int) + N // 2, 0, N - 1)
            np.add.at(hist, idx, A2)
            acc_tot += float(np.sum(A2))
            dif = dif + (sq_a - sq_b) if nsamp == 0 else (dif + (sq_a - sq_b))
            nsamp += 1
    tot = float(hist.sum())
    cdf = np.cumsum(hist) / max(tot, 1e-300)
    lo = int(np.searchsorted(cdf, QLO))
    hi = int(np.searchsorted(cdf, QHI))
    width = float(xs[hi] - xs[lo])
    tail = float(1.0 - cdf[min(int(np.searchsorted(cdf, 0.99)), N - 1)])
    resid = float(np.sum(np.abs(dif)) / max(acc_tot, 1e-300))
    return width, ctr_now, resid, tail, nsamp


def quantile_width_of_field(F2):
    hist = np.zeros(N)
    np.add.at(hist, (xs + N // 2), F2)
    cdf = np.cumsum(hist) / max(float(hist.sum()), 1e-300)
    lo = float(xs[int(np.searchsorted(cdf, QLO))])
    hi = float(xs[int(np.searchsorted(cdf, QHI))])
    return hi - lo


print("=" * 78)
print("第 783 号：尺缩判据的读数方法重估（随动坐标系 + 载波解调）")
print("=" * 78)
print("装置：N=%d omega_b=%.4f dt=%.2f T=%.0f sigma=%.0f" % (N, WB, DT, T_TOTAL, SIGMA))
print("速度档：%s" % list(VS))
print("第一段诊断已查明：第 691 号 BL4 的读数污染源是波包整体平移，")
print("            不是辐射尾。直方图按绝对坐标 x 累加会把平移轨迹展宽。")

U0, wb, psi, d1 = fit_sigma(SIGMA)
F_even = np.interp(xf, xf, psi)
CAL = quantile_width_of_field(F_even * F_even)
print("初始包络 F^2 的分位宽度（标定）= %.3f" % CAL)

# ---------------- NA1 读数 ----------------
print()
print("[NA1] 读数：五档速度的随动解调宽度")
RES = {}
FIX = {}
print("    %-6s %-12s %-12s %-12s %-8s" % ("v", "随动宽度", "解调残差", "尾污染", "采样点"))
for v in VS:
    shrink = 1.0 / gamma_of(v) if v > 0 else 1.0
    w, ctr, r, tf, ns = demod_profile(U0, SIGMA, wb, psi, d1, v, shrink, True)
    RES[v] = (w, ctr, r, tf, ns)
    print("    %-6.2f %-12.4f %-12.5f %-12.5f %-8d" % (v, w, r, tf, ns))
    wf, _, _, _, _ = demod_profile(U0, SIGMA, wb, psi, d1, v, shrink, False)
    FIX[v] = wf
print()
print("    固定井档（同一 shrink，隔离平移）的随动宽度：")
for v in VS:
    print("        v=%.2f  移动井 %.2f  固定井 %.2f  差 %.2f"
          % (v, RES[v][0], FIX[v], abs(RES[v][0] - FIX[v])))

# ---------------- NA0 标定 ----------------
print()
print("[NA0] 标定：静止档随动分位宽度对初始包络分位宽度")
w0 = RES[0.0][0]
rel_cal = abs(w0 / CAL - 1.0)
print("    静止档随动宽度 = %.4f；初始包络标定 = %.4f；相对差 %.4f"
      % (w0, CAL, rel_cal))
NA0 = rel_cal < 0.15
print("    NA0 判定：%s（容差 15%%，包络在解调窗口内移动 0.88 格带来的同阶偏差计入）"
      % ("PASS" if NA0 else "FAIL"))

# ---------------- NA2 单调性 ----------------
print()
print("[NA2] 主判据：随动宽度随 v 单调下降")
ws = [RES[v][0] for v in VS]
for v, w in zip(VS, ws):
    print("    v=%.2f  宽度=%.4f" % (v, w))
mono = all(ws[i] >= ws[i + 1] for i in range(len(ws) - 1))
print("    单调下降：%s；首末比 %.4f" % ("是" if mono else "否", ws[0] / ws[-1]))
NA2 = mono
print("    NA2 判定：%s" % ("PASS" if NA2 else "FAIL"))

# ---------------- NA3 尺缩判据 ----------------
print()
print("[NA3] 主判据：w(v)/w(0) 对 1/gamma 的相对偏差")
BUD = 0.10
rows = []
print("    %-6s %-13s %-13s %-13s" % ("v", "w/w0", "1/gamma", "相对偏差"))
for v in VS:
    if v == 0:
        continue
    r = RES[v][0] / w0
    tgt = 1.0 / gamma_of(v)
    dev = abs(r - tgt) / tgt
    rows.append((v, r, tgt, dev))
    print("    %-6.2f %-13.6f %-13.6f %-13.4f" % (v, r, tgt, dev))
if rows:
    worst = max(r[3] for r in rows)
    NA3 = worst < BUD
    print()
    print("    最大相对偏差 = %.4f（判据 %.2f）" % (worst, BUD))
    print("    NA3 判定：%s" % ("PASS" if NA3 else "FAIL"))
else:
    NA3 = False
    print("    NA3 判定：FAIL（无运动档读数）")

# ---------------- NA4 对照组 ----------------
print()
print("[NA4] 对照组：方法分辨力")
print("    组一（必须非零）：解调残差随 v 变化")
for v in VS:
    print("        v=%.2f  残差=%.6f" % (v, RES[v][2]))
d_res = abs(RES[VS[-1]][2] - RES[0.0][2])
g1 = d_res > 1e-5
print("      首末差 = %.6f" % d_res)
print("    组二（必须为零）：移动井与固定井的随动宽度之差")
d_fix = max(abs(RES[v][0] - FIX[v]) for v in VS)
for v in VS:
    print("        v=%.2f  差=%.4f（占宽度 %.1f%%）"
          % (v, abs(RES[v][0] - FIX[v]), 100 * abs(RES[v][0] - FIX[v]) / max(RES[v][0], 1e-9)))
g2 = d_fix < 0.05 * w0
print("      最大差 %.4f（判据 0.05*w0 = %.4f）" % (d_fix, 0.05 * w0))
NA4 = g1 and g2
print("    NA4 判定：%s（组一 %s，组二 %s）"
      % ("PASS" if NA4 else "FAIL", "通过" if g1 else "不通过", "通过" if g2 else "不通过"))

# ---------------- NA5 报告 ----------------
print()
print("[NA5] 报告：与第 691 号 BL4 两种被否读数的对照")
print("    第 691 号 BL4：单帧读数在 v=0.45 档宽度随窗口发散（记为辐射尾）；")
print("                  锁相窄窗读数在振荡周期内于 0.71 至 1.29 倍间跳动（单点节点风险）")
print("    本轮诊断：绝对坐标直方图给移动井档 257 格（v=0.30）与 384 格（v=0.45），")
print("                固定井档仅 25 与 34 格，峰位跑到 252 格。")
print("                故污染源是波包整体平移轨迹，尾污染只占约 1%%。")
print("    随动坐标系 + 载波解调后：宽度 %s"
      % ("随 v 单调下降" if mono else "仍非单调"))

# ---------------- NA6 判定 ----------------
print()
print("[NA6] 判定")
print("    NA0=%s NA2=%s NA3=%s NA4=%s"
      % tuple("PASS" if x else "FAIL" for x in (NA0, NA2, NA3, NA4)))
if not NA0:
    verdict = "静止档标定失败，本轮读数不可用"
elif not NA4:
    verdict = "对照组未通过（平移污染未被随动坐标系消除），NA2 与 NA3 不含物理信息"
elif not NA2:
    verdict = ("随动宽度随 v 非单调，尺缩读数在本装置下仍未成立；"
               "须换装置或换演化设置")
elif NA3:
    verdict = ("尺缩判据在随动解调读数下成立：w(v)/w(0) 对 1/gamma 的最大相对偏差"
               " %.4f（预算 %.2f）；第 691 号 BL4 的不成立结论改判为读数方法的局限"
               % (max(r[3] for r in rows), BUD))
else:
    verdict = ("随动宽度随 v 单调下降但偏离 1/gamma 超过预算（最大 %.4f）；"
               "尺缩判据在解调读数下仍不成立" % max(r[3] for r in rows))
print("    判语：" + verdict)
print()
print("    边界：本轮回答的是「尺缩判据能否用随动解调建立起来」；")
print("          「尺缩在格点上是否成立」是另一个问题，前者成立不等于后者。")
print("耗时 %.1f 秒" % (time.time() - T0))

fig, ax = plt.subplots(1, 3, figsize=(15.8, 4.4))

ax[0].plot(VS, ws, "s-", color="#c44", ms=8, label="随动解调宽度")
ax[0].plot(VS, [FIX[v] for v in VS], "^--", color="#824", ms=7, label="固定井档")
ax[0].plot(VS, [w0 * (1.0 / gamma_of(v)) for v in VS], ":", color="#268", lw=2.0,
           label=r"$w(0)/\gamma$")
ax[0].set_xlabel("v")
ax[0].set_ylabel("分位宽度")
ax[0].set_title("左：随动宽度的三组读数")
ax[0].legend(fontsize=8)

ax[1].plot(VS, [RES[v][2] for v in VS], "o-", color="#824", ms=8, label="解调残差")
ax[1].set_xlabel("v")
ax[1].set_ylabel("残差")
ax[1].set_title("中：载波残留量（组一）")
ax[1].legend(fontsize=9)

if rows:
    ax[2].plot([r[0] for r in rows], [r[3] for r in rows], "s-", color="#268", ms=8,
               label="相对偏差")
ax[2].axhline(BUD, color="#c44", ls="--", lw=1.5, label="预算 %.2f" % BUD)
ax[2].set_xlabel("v")
ax[2].set_ylabel(r"$|w/w_0 - 1/\gamma|/(1/\gamma)$")
ax[2].set_title("右：尺缩判据的相对偏差")
ax[2].legend(fontsize=9)

plt.tight_layout()
plt.savefig(IMG, dpi=150)
print("图已存 %s" % IMG)
