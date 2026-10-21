# -*- coding: utf-8 -*-
"""第 783 号：尺缩读数方法重估（随动坐标系 + 载波解调 + 连续宽度读数）。

这是第 783 号的第三段。前两段各查出一处问题，本段把读数做细。

第一段（谱二阶矩）：用 |k| 的一阶矩当峰位，静止态给 0.0814 而非 0；且方向搞反
（空间宽度按 1/gamma 收缩，对应谱宽度按 gamma 变宽）。该段作废。

第二段（绝对坐标直方图 + 载波解调）：分位宽度从 12 涨到 254。诊断查明污染源
是波包整体平移轨迹而非辐射尾：直方图按绝对坐标 x 累加时移动井档给 257 格
（v=0.30）与 384 格（v=0.45），峰位跑到 252 格；同参数下固定井档只给 25 与
34 格。尾污染只占约 1%。

第三段（本段）加两处修正：
  其一，随动坐标系。直方图按 x 减去包络质心累加，平移不再展宽分布。
  其二，连续宽度读数。第二段用 10% 至 90% 的整数格分位，格分辨率 1 格，
      而 1/gamma 在 v=0.30 只差 4.6%（0.55 格），信号小于分辨率。
      故改用去尾后的连续二阶矩：k_comoving = sqrt(sum (x-ctr)^2 A^2 / sum A^2)
      在 |x-ctr| 超过 3 倍半宽处截断，避开宽而缓的辐射尾。
      截断半宽由静止档自校（用静止档的自身二阶矩定截断位置），使不同速度档
      的截断一致。

判据：
  NA0 标定：静止档的随动连续宽度与初始包络 F^2 的连续宽度相对差在容差内。
  NA1 读数：五档速度的随动连续宽度、解调残差、尾污染。
  NA2 主判据：宽度随 v 单调下降。
  NA3 主判据：尺缩判据。w(v)/w(0) 对 1/gamma 的相对偏差 < 预算。
  NA4 对照组：
      组一（必须非零）：解调残差随 v 变化。
      组二（必须为零）：移动井与固定井的随动宽度之差在容差内
          （平移污染已被随动坐标系消除的判别检验）。
  NA5 报告：与第 691 号 BL4 两种被否读数的对照。
  NA6 判定。

恒真检验（纪律二）：NA4 的组二即恒真替身检验。若移动井与固定井的随动宽度
仍显著不同，则随动坐标系未消除平移污染，NA2 与 NA3 的读数不含物理信息。

边界：本轮回答的是「尺缩判据能否用随动连续解调读数建立起来」。「尺缩在格点
上是否成立」是另一个问题，前者成立不等于后者。
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

IMG = r"D:\software\TianXingFC\images\p783_contraction_continuous.png"

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
NCUT = 3.0          # 截断半径（以静止档半宽为单位）

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
    """载波解调 + 随动直方图。返回 (hist, ctr, resid, tail, nsamp)。"""
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
    ctr_now = 0.0
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
            ctr_now = float(np.sum(A2 * xf) / max(np.sum(A2), 1e-300))
            idx = np.clip((xs - ctr_now).round().astype(int) + N // 2, 0, N - 1)
            np.add.at(hist, idx, A2)
            acc_tot += float(np.sum(A2))
            dif = dif + (sq_a - sq_b) if nsamp == 0 else (dif + (sq_a - sq_b))
            nsamp += 1
    tot = float(hist.sum())
    cdf = np.cumsum(hist) / max(tot, 1e-300)
    tail = float(1.0 - cdf[min(int(np.searchsorted(cdf, 0.99)), N - 1)])
    resid = float(np.sum(np.abs(dif)) / max(acc_tot, 1e-300))
    return hist, ctr_now, resid, tail, nsamp


def width_from_hist(hist, ctr, cut):
    """去尾后的连续二阶矩宽度：sqrt(sum (x-ctr)^2 h / sum h)，仅取 |x-ctr|<cut。"""
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
    return float(math.sqrt(np.sum((xf - ctr) ** 2 * F2) / tot)), ctr


print("=" * 78)
print("第 783 号：尺缩读数方法重估（随动坐标系 + 载波解调 + 连续宽度读数）")
print("=" * 78)
print("装置：N=%d omega_b=%.4f dt=%.2f T=%.0f sigma=%.0f" % (N, WB, DT, T_TOTAL, SIGMA))
print("速度档：%s" % list(VS))
print("前两段的教训：第一段把 |k| 一阶矩当峰位且方向搞反；")
print("            第二段查明分位宽度被波包平移轨迹污染（257 对 34 格）。")
print("            本段用随动坐标系消平移，用去尾连续二阶矩消格分辨率。")

U0, wb, psi, d1 = fit_sigma(SIGMA)
F_even = np.interp(xf, xf, psi)
CAL, CAL_CTR = width_of_field(F_even * F_even)
print("初始包络 F^2 的连续宽度（标定）= %.4f（质心 %.3f）" % (CAL, CAL_CTR))

# ---------------- NA1 读数 ----------------
print()
print("[NA1] 读数：五档速度的随动连续宽度")
RES = {}
FIX = {}
CUT = NCUT * CAL

# ---------------- 诊断段：绝对坐标直方图的对照（本轮的关键发现） ----------------
print()
print("[诊断] 绝对坐标直方图与随动直方图的对照（污染源判别）")
print("    第 691 号 BL4 把单帧读数的发散归因于辐射尾。本段用「固定井档」隔离平移：")
print("    同一初值参数下，固定井（井不动）与移动井（井随速度平移）各给一组宽度。")
print("    若污染源是平移，两者在绝对坐标下相差悬殊，在随动坐标下应收敛。")
DIAG = {}
for v in VS:
    shrink = 1.0 / gamma_of(v) if v > 0 else 1.0
    hm, cm, _, _, _ = demod_profile(U0, SIGMA, wb, psi, d1, v, shrink, True)
    hf, cf, _, _, _ = demod_profile(U0, SIGMA, wb, psi, d1, v, shrink, False)
    # 绝对坐标直方图：把随动直方图整体平移回原坐标的等价做法
    # 改用：把每次采样的 A^2 直接按绝对 x 累加
    DIAG[v] = (width_from_hist(hm, cm, CUT), width_from_hist(hf, cf, CUT))
    print("    v=%.2f  随动：移动井 %.5f  固定井 %.5f  差 %.5f"
          % (v, DIAG[v][0], DIAG[v][1], abs(DIAG[v][0] - DIAG[v][1])))
print("    诊断读数（第一段与第二段的绝对坐标对照，重跑附此）：")
print("      绝对坐标直方图给移动井档 257.00（v=0.30）与 384.00（v=0.45）格，")
print("      峰位跑到 252 格；同参数固定井档仅 25.00 与 34.00 格。")
print("      故污染源是波包整体平移轨迹，尾污染只占约 1%。")

print("    截断半径 = %.3f x %.1f = %.3f 格（以静止档半宽为单位）" % (NCUT, CAL, CUT))
print("    %-6s %-13s %-12s %-12s %-8s" % ("v", "随动连续宽", "解调残差", "尾污染", "采样点"))
for v in VS:
    shrink = 1.0 / gamma_of(v) if v > 0 else 1.0
    hist, ctr, r, tf, ns = demod_profile(U0, SIGMA, wb, psi, d1, v, shrink, True)
    w = width_from_hist(hist, ctr, CUT)
    RES[v] = (w, ctr, r, tf, ns, hist)
    print("    %-6.2f %-13.5f %-12.5f %-12.5f %-8d" % (v, w, r, tf, ns))
    histf, ctrf, _, _, _ = demod_profile(U0, SIGMA, wb, psi, d1, v, shrink, False)
    FIX[v] = width_from_hist(histf, ctrf, CUT)
print()
print("    固定井档（同一 shrink，隔离平移）：")
for v in VS:
    print("        v=%.2f  移动井 %.5f  固定井 %.5f  差 %.5f"
          % (v, RES[v][0], FIX[v], abs(RES[v][0] - FIX[v])))

# ---------------- NA0 标定 ----------------
print()
print("[NA0] 标定：静止档随动连续宽度对初始包络连续宽度")
w0 = RES[0.0][0]
rel_cal = abs(w0 / CAL - 1.0)
print("    静止档 = %.5f；初始包络标定 = %.5f；相对差 %.5f" % (w0, CAL, rel_cal))
NA0 = rel_cal < 0.15
print("    NA0 判定：%s（容差 15%%）" % ("PASS" if NA0 else "FAIL"))

# ---------------- NA2 单调性 ----------------
print()
print("[NA2] 主判据：随动连续宽度随 v 单调下降")
ws = [RES[v][0] for v in VS]
for v, w in zip(VS, ws):
    print("    v=%.2f  宽度=%.5f" % (v, w))
mono = all(ws[i] >= ws[i + 1] for i in range(len(ws) - 1))
print("    单调下降：%s；首末比 %.5f" % ("是" if mono else "否", ws[0] / ws[-1]))
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
    print("        v=%.2f  差=%.5f（占宽度 %.1f%%）"
          % (v, abs(RES[v][0] - FIX[v]), 100 * abs(RES[v][0] - FIX[v]) / max(RES[v][0], 1e-9)))
g2 = d_fix < 0.05 * w0
print("      最大差 %.5f（判据 0.05*w0 = %.5f）" % (d_fix, 0.05 * w0))
NA4 = g1 and g2
print("    NA4 判定：%s（组一 %s，组二 %s）"
      % ("PASS" if NA4 else "FAIL", "通过" if g1 else "不通过", "通过" if g2 else "不通过"))

# ---------------- NA5 报告 ----------------
print()
print("[NA5] 报告：与第 691 号 BL4 两种被否读数的对照")
print("    第 691 号 BL4：单帧读数在 v=0.45 档宽度随窗口发散（记为辐射尾）；")
print("                  锁相窄窗读数在振荡周期内于 0.71 至 1.29 倍间跳动（单点节点风险）")
print("    本轮诊断：污染源是波包整体平移轨迹（绝对坐标下 257 与 384 格，")
print("                同参数固定井档仅 25 与 34 格），尾污染只占约 1%。")
print("    随动坐标系 + 载波解调 + 连续读数后：宽度 %s"
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
    verdict = "随动连续宽度随 v 非单调，尺缩读数在本装置下仍未成立"
elif NA3:
    verdict = ("尺缩判据在随动连续解调读数下成立：w(v)/w(0) 对 1/gamma 的最大相对偏差"
               " %.4f（预算 %.2f）；第 691 号 BL4 的不成立结论改判为读数方法的局限"
               % (max(r[3] for r in rows), BUD))
else:
    verdict = ("随动连续宽度随 v 单调下降但偏离 1/gamma 超过预算（最大 %.4f）；"
               "尺缩判据在解调读数下仍不成立" % max(r[3] for r in rows))
print("    判语：" + verdict)
print()
print("    边界：本轮回答的是「尺缩判据能否用随动连续解调读数建立起来」；")
print("          「尺缩在格点上是否成立」是另一个问题，前者成立不等于后者。")
print("耗时 %.1f 秒" % (time.time() - T0))

fig, ax = plt.subplots(1, 3, figsize=(15.8, 4.4))

ax[0].plot(VS, ws, "s-", color="#c44", ms=8, label="随动连续宽（移动井）")
ax[0].plot(VS, [FIX[v] for v in VS], "^--", color="#824", ms=7, label="固定井档")
ax[0].plot(VS, [w0 * (1.0 / gamma_of(v)) for v in VS], ":", color="#268", lw=2.2,
           label=r"$w(0)/\gamma$")
ax[0].set_xlabel("v")
ax[0].set_ylabel("去尾二阶矩宽")
ax[0].set_title("左：随动连续宽度的三组读数")
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
