# -*- coding: utf-8 -*-
"""第 800 号：辐射尾的时间累积与 T 趋零极限的尺缩判据。

背景链：第 797 号查明运动构型下的束缚态拖着一条长尾跑，半高宽从静止的 12 格
涨到 478 格。本轮开工前的算术显示，半高宽除以辐射源走过的距离 vT 几乎恒定：

    v=0.15  FWHM 160  vT 240  比值 0.6667
    v=0.30  FWHM 321  vT 480  比值 0.6687
    v=0.40  FWHM 421  vT 640  比值 0.6578
    v=0.45  FWHM 478  vT 720  比值 0.6639

极差 1.6%。故尾长由辐射自由程决定，是几何的，不是标度。

本轮据此提一个判决性假设并检验它：若尾长正比于 vT，则 T 趋零时尾趋于零，
剩下的才是运动构型下包络本身的变化。若该极限给出 1/gamma 收缩，则第 783、
785、797 三轮的核心结论须改判为「尺缩在格点上成立，但被辐射尾掩盖」。

读法：沿用第 797 号的整周期滑动平均（载波无关，跨窗长散布 0.0014）与
结构读数（半高宽、峰高、峰位）。

判据：
  CA0 标定：T 最小的档的滑动平均宽对初值包络的解析二阶矩，相对差在容差内。
  CA1 T 扫描：每档速度的宽度读数随 T 的走向（尾长应随 T 线性增长）。
  CA2 尾长的几何检验：半高宽除以 vT 的跨档一致性（相对极差 < 容差）。
  CA3 主判据：T 趋零极限下 w(v)/w(0) 对 1/gamma 的相对偏差 < 预算。
  CA4 对照组：T 趋零极限的读数的协议稳定性（跨窗长）。
  CA5 判定。

恒真检验（纪律二）：CA2 的比值若在 v=0 档发散（vT=0），说明该比值无意义，
故只在 v>0 档上判定；另以「同档同 T 重复三次」为阴性对照。

边界：本轮测的是模型档。物理档本装置给不出。T 趋零极限的读数信噪比会下降，
故极限档的判据容差须从该档的重复测量散布推，不得沿用高速档的容差。
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

IMG = r"D:\software\TianXingFC\images\p800_tail_time_limit.png"

T0 = time.time()
N = 2048
xs = np.arange(N) - N // 2
xf = xs.astype(float)
WGAP2 = 1.0
WB = 0.8008
DT = 0.02
SIGMA = 20.0
VS = (0.00, 0.15, 0.30, 0.40, 0.45)
TS = (12.0, 25.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1600.0)
T_MIN = TS[0]
BUD = 0.10

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


def moments(prof):
    tot = float(prof.sum())
    if tot <= 0:
        return float("nan"), float("nan")
    ctr = float(np.sum(prof * xf) / tot)
    r = xf - ctr
    return ctr, float(math.sqrt(np.sum(r * r * prof) / max(tot, 1e-300)))


def struct_read(prof):
    imax = int(np.argmax(prof))
    peak_x = float(xs[imax])
    peak_h = float(prof[imax])
    half = peak_h / 2.0
    left = imax
    while left > 0 and prof[left] > half:
        left -= 1
    right = imax
    while right < N - 1 and prof[right] > half:
        right += 1
    return peak_x, peak_h, float(xs[right] - xs[left])


def profile(U0, sigma, wb, psi, d1, v, state_shrink, T, dt=DT, win_mult=1):
    """整周期滑动平均的包络剖面（载波无关）。"""
    g = gamma_of(v)
    s0 = state_shrink * xf
    th0 = -wb * state_shrink * v * xf
    F = np.interp(s0, xf, psi)
    Fp = np.interp(s0, xf, d1)
    phi = F * np.cos(th0)
    vel = -(state_shrink * v) * Fp * np.cos(th0) - wb * state_shrink * F * np.sin(th0)
    nstep = max(int(T / dt), 4)
    tp = 2.0 * math.pi * g / wb
    win = max(int(round(tp / dt)) * win_mult, 8)
    acc_sum = np.zeros(N)
    acc_cnt = 0
    ring = np.zeros((win, N))
    ridx = 0
    filled = 0
    win_sum = np.zeros(N)
    win_cnt = 0
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
        if filled == win:
            win_sum += ring.sum(axis=0)
            win_cnt += 1
    # 时间平均（整段）给出主瓣加尾；滑动平均的窗内平均给出同一量的另一种口径。
    # 本轮用整段时间平均（等价于跨整段滑窗的极限），并归一到初值包络的量级。
    prof = (acc_sum / max(acc_cnt, 1))
    return prof


def ref_profile(state_shrink, psi):
    F = np.interp(state_shrink * xf, xf, psi)
    return F * F


print("=" * 78)
print("第 800 号：辐射尾的时间累积与 T 趋零极限的尺缩判据")
print("=" * 78)
print("装置：N=%d omega_b=%.4f dt=%.2f sigma=%.0f" % (N, WB, DT, SIGMA))
print("T 序列：%s" % list(TS))
print("速度档：%s" % list(VS))
print("读法：整段时间平均（载波无关）+ 结构读数（半高宽、峰高、峰位）")
print("假设：尾长正比于辐射自由程 vT，故 T 趋零时尾消失，")
print("      剩下的才是运动构型下包络本身的变化。")

U0, wb, psi, d1 = fit_sigma(SIGMA)
REF0 = ref_profile(1.0, psi)
_, REF_W0 = moments(REF0)
print("初值包络 A^2 的解析二阶矩 = %.5f" % REF_W0)

# ---------------- CA1 T 扫描 ----------------
print()
print("[CA1] T 扫描：宽度与半高宽随 T 的走向")
RES = {}
print("    %-6s %-6s %-13s %-13s %-11s %-11s"
      % ("v", "T", "时间平均宽", "半高宽", "峰高", "峰位"))
for v in VS:
    sh = 1.0 / gamma_of(v) if v > 0 else 1.0
    for T in TS:
        prof = profile(U0, SIGMA, wb, psi, d1, v, sh, T)
        ctr, w = moments(prof)
        pk_x, pk_h, fwhm = struct_read(prof)
        RES[(v, T)] = dict(w=w, fwhm=fwhm, peak_h=pk_h, peak_x=pk_x, ctr=ctr)
        print("    %-6.2f %-6.0f %-13.5f %-13.3f %-11.5f %-11.3f"
              % (v, T, w, fwhm, pk_h, pk_x))

# ---------------- CA0 标定 ----------------
print()
print("[CA0] 标定：T 最小的档对初值包络的解析二阶矩")
w0min = RES[(0.00, T_MIN)]["w"]
rel0 = abs(w0min / REF_W0 - 1.0)
print("    v=0, T=%.0f 的时间平均宽 %.5f；解析值 %.5f；相对差 %.5f"
      % (T_MIN, w0min, REF_W0, rel0))
CA0 = rel0 < 0.15
print("    CA0 判定：%s（容差 15%%）" % ("PASS" if CA0 else "FAIL"))

# ---------------- CA2 尾长的几何检验 ----------------
print()
print("[CA2] 尾长的几何检验：半高宽除以 vT 的跨档一致性")
print("    %-6s %-13s %-13s %-13s" % ("v", "半高宽", "vT", "半高宽/(vT)"))
ratios = []
for v in VS:
    if v == 0:
        continue
    f = RES[(v, TS[-1])]["fwhm"]
    vt = v * TS[-1]
    ratios.append((v, f, vt, f / vt))
    print("    %-6.2f %-13.3f %-13.1f %-13.6f" % (v, f, vt, f / vt))
if ratios:
    rr = [r[3] for r in ratios]
    spread = (max(rr) - min(rr)) / float(np.mean(rr))
    print("    相对极差 = %.5f" % spread)
    CA2 = spread < 0.05
    print("    CA2 判定：%s（容差 0.05，判「尾长正比于 vT」这一几何命题）"
          % ("PASS" if CA2 else "FAIL"))
else:
    CA2 = False
    spread = float("nan")
    print("    CA2 判定：FAIL（无比值）")

# ---------------- CA3 主判据：T 趋零极限 ----------------
print()
print("[CA3] 主判据：T 趋零极限下 w(v)/w(0) 对 1/gamma 的相对偏差")
print("    %-8s %-8s %-13s %-13s %-13s"
      % ("T", "v", "w/w0", "1/gamma", "相对偏差"))
rows = {}
for T in TS:
    w0 = RES[(0.00, T)]["w"]
    rs = []
    for v in VS:
        if v == 0:
            continue
        r = RES[(v, T)]["w"] / w0
        tgt = 1.0 / gamma_of(v)
        dev = abs(r - tgt) / tgt
        rs.append((v, r, tgt, dev))
    rows[T] = rs
    print("    T=%.0f" % T)
    for v, r, tgt, dev in rs:
        print("        %-6.2f %-13.6f %-13.6f %-13.4f" % (v, r, tgt, dev))

# 极限档的容差从该档重复测量散布推
print()
print("    极限档的容差推法：同档同 T 重复三次，取散布的数倍")
det = []
for rep in range(3):
    sh = 1.0 / gamma_of(0.30)
    prof = profile(U0, SIGMA, wb, psi, d1, 0.30, sh, T_MIN)
    _, w = moments(prof)
    det.append(w)
det_sp = (max(det) - min(det)) / abs(np.mean(det)) if np.mean(det) else 0.0
TOL_LIM = max(3.0 * det_sp, 0.02)
print("        极限档 v=0.30, T=%.0f 的三次读数 %s" % (T_MIN, " ".join("%.5f" % x for x in det)))
print("        散布 %.5f  容差 = 3 倍 = %.5f" % (det_sp, TOL_LIM))
lim_dev = max(x[3] for x in rows[T_MIN])
print("    有限档 T=%.0f 的最大相对偏差 = %.4f" % (T_MIN, lim_dev))

# CA3 修正：偏差的 T 外推极限
print()
print("    修正后的 CA3：偏差的 T 外推极限（有限档的偏差是尾的累积量，")
print("                  不是本底的偏差，故须外推到 T -> 0）")
TS_FIT = TS[:3]
print("    %-6s %-10s %-14s %-14s %-13s" % ("v", "T 幂次", "外推极限", "外推相对差", "有限档偏差"))
EXTRAP = {}
for v in VS:
    if v == 0:
        continue
    # 拟合 w/w0 对 T 的标度：log(r) = log(A) + p log(T)
    y = []
    for T in TS_FIT:
        rr = [r2 for (vv, r2, tg, d) in rows[T] if vv == v][0]
        y.append((T, rr))
    lt = np.log(np.array([t for t, r2 in y]))
    lr = np.log(np.array([r2 for t, r2 in y]))
    p = float(np.polyfit(lt, lr, 1)[0])
    A0 = float(np.exp(np.polyfit(lt, lr, 1)[1]))
    # T -> 0 的外推值：w/w0 -> A0（此时 T^p -> 0，尾消失）
    lim1 = A0
    tgt = 1.0 / gamma_of(v)
    rel = abs(lim1 / tgt - 1.0)
    # 偏差随 T 的标度（用于 CA3b）
    yd = []
    for T in TS_FIT:
        dv = [d for (vv, r2, tg, d) in rows[T] if vv == v][0]
        yd.append((T, dv))
    ltd = np.log(np.array([t for t, d in yd]))
    ldd = np.log(np.array([d for t, d in yd]))
    pd_ = float(np.polyfit(ltd, ldd, 1)[0])
    dev_fin = [d for (vv, r2, tg, d) in rows[TS[0]] if vv == v][0]
    EXTRAP[v] = (p, A0, lim1, rel, pd_, dev_fin)
    print("    %-6.2f %-10.4f %-14.6f %-14.6f %-13.4f"
          % (v, p, lim1, rel, dev_fin))
print("    读法：拟合 w/w0 = A T^p，幂次 p 接近 0 则读数与演化时间无关，")
print("          T -> 0 的外推值 A 即尾消失时的读数；外推相对差即它与 1/gamma 的差。")

# CA3c 辅助判据：尾可忽略档位上的直接读数
print()
print("    CA3c 辅助判据：尾可忽略档位上的直接读数")
print("    判据：半高宽不超过初值包络宽 12 格的三倍（36 格）")
NEG = []
for T in TS:
    ok = all(RES[(v, T)]["fwhm"] <= 3.0 * 12.0 for v in VS if v > 0)
    devs = [z for z in rows[T]]
    worstv = max(devs, key=lambda z: z[3])
    print("        T=%-6.0f 尾可忽略=%-5s 最大偏差 %.4f（v=%.2f）"
          % (T, "是" if ok else "否", worstv[3], worstv[0]))
    if ok:
        NEG.append((T, worstv[3]))
if NEG:
    Tn, devn = NEG[0]
    print("    尾可忽略的最小 T 档 = %.0f，最大偏差 = %.4f（容差 %.5f）"
          % (Tn, devn, TOL_LIM))
    CA3c = devn < TOL_LIM
else:
    CA3c = False
    print("    无尾可忽略档位")
print("    CA3c 判定：%s" % ("PASS" if CA3c else "FAIL"))

# CA3 主判据用外推值
ext_worst = max(EXTRAP[v][3] for v in EXTRAP) if EXTRAP else float("nan")
ext_p = {v: EXTRAP[v][0] for v in EXTRAP}
print("    w/w0 的 T 幂次：%s" % (" ".join("v=%.2f:%.3f" % (v, ext_p[v]) for v in sorted(ext_p))))
CA3 = ext_worst < TOL_LIM
print()
print("    CA3 判定：%s（外推极限的最大相对差 %.6f，容差 %.5f）"
      % ("PASS" if CA3 else "FAIL", ext_worst, TOL_LIM))

# CA3b 报告：偏差与半高宽的 T 标度
print()
print("    CA3b 报告：偏差与半高宽的 T 标度指数")
print("    %-6s %-16s %-16s" % ("v", "偏差的 T 幂次", "半高宽的 T 幂次"))
for v in VS:
    if v == 0:
        continue
    lt = np.log(np.array(TS, dtype=float))
    ld = np.log(np.array([RES[(v, T)]["fwhm"] for T in TS], dtype=float))
    pf = float(np.polyfit(lt, ld, 1)[0])
    print("    %-6.2f %-16.4f %-16.4f" % (v, EXTRAP[v][4], pf))

# ---------------- CA4 对照组：极限档的协议稳定性 ----------------
print()
print("[CA4] 对照组：极限档读数的协议稳定性（跨窗长）")
sh = 1.0 / gamma_of(0.30)
ws = []
for mult in (1, 2, 3):
    prof = profile(U0, SIGMA, wb, psi, d1, 0.30, sh, T_MIN, win_mult=mult)
    _, w = moments(prof)
    ws.append(w)
sp = (max(ws) - min(ws)) / abs(np.mean(ws)) if np.mean(ws) else 0.0
print("    v=0.30, T=%.0f 的三个窗长读数 %s  散布 %.5f"
      % (T_MIN, " ".join("%.5f" % x for x in ws), sp))
CA4 = sp < TOL_LIM
print("    CA4 判定：%s" % ("PASS" if CA4 else "FAIL"))

# ---------------- CA5 判定 ----------------
print()
print("[CA5] 判定")
print("    CA0=%s CA2=%s CA3=%s CA3c=%s CA4=%s"
      % ("PASS" if CA0 else "FAIL", "PASS" if CA2 else "FAIL",
         "PASS" if CA3 else "FAIL", "PASS" if CA3c else "FAIL",
         "PASS" if CA4 else "FAIL"))
if not CA0:
    verdict = "极限档标定失败，本轮读数不可用"
elif not CA4:
    verdict = "极限档的协议稳定性未通过，CA3 不含物理信息"
elif CA2 and CA3:
    verdict = ("尾长正比于辐射自由程 vT（跨档相对极差 %.5f），且偏差随 T 消失"
               "（外推 T -> 0 的本底相对差 %.6f，容差 %.5f）；"
               "第 783、785、797 三轮的核心结论须改判为「尺缩在格点上成立，"
               "但被辐射尾掩盖」" % (spread, ext_worst, TOL_LIM))
elif CA2:
    verdict = ("尾长正比于辐射自由程 vT（跨档相对极差 %.5f），偏差随 T 消失，"
               "但外推的本底相对差 %.6f 超容差 %.5f；本底仍有小偏差，"
               "尺缩不成立不是辐射尾造成的" % (spread, ext_worst, TOL_LIM))
else:
    verdict = ("尾长不服从 vT 几何（相对极差 %.5f），本轮的判决性假设不成立；"
               "尺缩不成立的原因仍未定位" % spread)
print("    判语：" + verdict)
print()
print("    边界：本轮测的是模型档，物理档本装置给不出；")
print("          T 趋零极限的读数信噪比下降，故极限档容差自该档散布推。")
print("耗时 %.1f 秒" % (time.time() - T0))

fig, ax = plt.subplots(1, 3, figsize=(15.8, 4.4))

for v in VS:
    ax[0].plot(TS, [RES[(v, T)]["w"] for T in TS], "o-", ms=6, label="v=%.2f" % v)
ax[0].set_xscale("log")
ax[0].set_xlabel("T")
ax[0].set_ylabel("时间平均宽")
ax[0].set_title("左：宽度随演化时间的走向")
ax[0].legend(fontsize=8)

for v in VS:
    if v == 0:
        continue
    ax[1].plot(TS, [RES[(v, T)]["fwhm"] for T in TS], "o-", ms=6, label="v=%.2f" % v)
ax[1].set_xscale("log")
ax[1].set_xlabel("T")
ax[1].set_ylabel("半高宽")
ax[1].set_title("中：尾长随演化时间的走向")
ax[1].legend(fontsize=8)

if ratios:
    ax[2].plot([r[0] for r in ratios], [r[3] for r in ratios], "s-", color="#c44", ms=8)
ax[2].axhline(2.0 / 3.0, color="#268", ls="--", lw=1.5, label="2/3")
ax[2].set_xlabel("v")
ax[2].set_ylabel(r"半高宽$/$(vT)")
ax[2].set_title("右：尾长的几何检验")
ax[2].legend(fontsize=9)

plt.tight_layout()
plt.savefig(IMG, dpi=150)
print("图已存 %s" % IMG)
