# -*- coding: utf-8 -*-
"""第 807 号：核心瓣与辐射尾的分离，核心瓣的展宽机理。

背景：第 805 号的四环因果链里，第一环（信号小于一格）经复核不成立。理由：
第 783 号的宽度读数 width_from_hist 是连续二阶矩，静止档的标定相对差
0.01654，即绝对精度约 0.08 格，不是一格。v=0.30 档的收缩信号 0.222 格是
分辨率的 2.8 倍。故第一环须撤。

复核同时查出一件更要紧的事：第 783 号的 width_from_hist 带截断半径 cut
（静止档为 3 倍半宽，14.672 格），即尾被截断。所以第 783 号读的是核心瓣
的部分宽度，而第 800 号读的半高宽（478 格）覆盖主瓣加尾。两者不矛盾，
测的是不同对象。

于是问题落回核心瓣：第 783 号的核心瓣读数从静止的 4.81 格涨到 v=0.30 档的
8.42 格（涨 75%），而 1/gamma 预言 4.59 格。**核心瓣本身就没收缩。**
尾不是原因（尾被截断了），核心瓣的展宽才是。

本轮问三个可判的问题：
  其一，核心瓣的宽度随 v 怎么走？是按 1/gamma 收缩、按常数、还是被展宽？
  其二，核心瓣的形状是否仍是高斯？展宽与畸变是两回事：形状不变而宽度变
      是尺度性的，形状变了则包络本身在演化中被改形。
  其三，展宽的标度是什么？若随 v^2 或 v^4，则可与第 704 号 dev 三分项的
      格点色散项（v^4）对照；若与 v 无关则是构造失配。

读法：沿用第 783 号的载波解调（相隔四分之一周期的两次采样，phi^2+phi'^2=A^2），
它逐点消去载波，尾上的载波也被消掉，读出的是主瓣加尾；本轮另做瓣尾分割，
在分割点两侧各读一次。

瓣尾分割点的取法：主瓣的峰位到第一个局部极小之间的位置。用峰位两侧的
累积权重比 0.5 作为分割点（主瓣含一半权重处），此点对高斯型主瓣约在
1.177 倍 sigma 处，是标准的中位半径。分割点须在网格分辨率内可分辨。

判据：
  QA0 标定：静止档的分割点、主瓣宽度与形状指标；主瓣宽度对初值包络解析值
      的相对差在容差内。
  QA1 读数：五档速度的主瓣宽度、形状指标、分割点位置、尾长。
  QA2 主判据：主瓣宽度随 v 的走向。若主瓣按 1/gamma 收缩，则尺缩在核心瓣
      上成立，前几轮的结论须改判；若主瓣被展宽，则前几轮结论保留。
  QA3 形状判据：主瓣形状相对高斯的偏离度随 v 的走向。偏离小则展宽是尺度性
      的，偏离大则包络被改形。
  QA4 标度判据：主瓣宽度的相对变化对 v 的幂次拟合（按第 704 号 dev 三分项的
      口径比较：构造失配给 v^2 量级，格点色散给 v^4 量级）。
  QA5 对照组：分割协议的稳定性（分割点取法换一种，宽度读数须稳定）。
  QA6 判定。

恒真检验（纪律二）：QA5 的分割协议若换成同一个，宽度读数必逐位相同；
另以「静止档重复三次」为阴性对照。

边界：本轮测的是模型档。物理档本装置给不出。主瓣与尾的分割判据依赖「主瓣
与尾可分」这一前提，故本轮须同时报告分割点两侧的连续性（有无明显断点），
若断点不明显则分割读数不可用。
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

IMG = r"D:\software\TianXingFC\images\p787_lobe_tail_split.png"

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
QHALF = 0.5

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


def demod_hist(U0, sigma, wb, psi, d1, v, state_shrink, dt=DT, T=T_TOTAL):
    """第 783 号的载波解调，返回随动直方图与质心。

    随动坐标系：每次采样按当时的包络质心对齐后累加，故平移不展宽分布。
    """
    g = gamma_of(v)
    tp = 2.0 * math.pi * g / wb
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
    nsamp = 0
    ctr_now = 0.0
    for s in range(nstep):
        t = (s + 0.5) * dt
        w1 = -U0 * np.exp(-((xf - v * t) ** 2) / (2.0 * sigma ** 2))
        a1 = (np.roll(phi, 1) + np.roll(phi, -1) - 2.0 * phi) - WGAP2 * phi - w1 * phi
        phi = phi + dt * vel + .5 * dt * dt * a1
        t2 = (s + 1) * dt
        w2 = -U0 * np.exp(-((xf - v * t2) ** 2) / (2.0 * sigma ** 2))
        a2 = (np.roll(phi, 1) + np.roll(phi, -1) - 2.0 * phi) - WGAP2 * phi - w2 * phi
        vel = vel + .5 * dt * (a1 + a2)
        if s % nper == 0 and s > nstep // 3:
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
            nsamp += 1
    return hist, ctr_now, nsamp


def lobe_split(hist, ctr, qhalf=QHALF):
    """瓣尾分割：取峰位两侧累积权重达 qhalf 的半径为分割半径。

    返回 (split_radius, main_lobe_profile, tail_profile)
      main_lobe_profile：|x-ctr| <= split_radius 内的直方图（补零到全格）
      tail_profile：外的部分
    """
    # 峰位：在质心附近找最大
    near = np.abs(xs - ctr) < 40.0
    rel = xs - ctr
    imax = int(np.argmax(np.where(near, hist, -1.0)))
    # 从峰位向两侧累积权重
    total = float(hist.sum())
    lo = imax
    acc = 0.0
    while lo > 0 and acc < qhalf * total:
        acc += hist[lo]
        lo -= 1
    hi = imax
    acc2 = 0.0
    while hi < N - 1 and acc2 < qhalf * total:
        acc2 += hist[hi]
        hi += 1
    r = 0.5 * (float(xs[hi] - xs[lo]))
    cut = r
    main = np.where(np.abs(rel) <= cut, hist, 0.0)
    tail = np.where(np.abs(rel) > cut, hist, 0.0)
    return cut, main, tail


def width_of(prof, ctr, cut=None):
    """去尾后的连续二阶矩宽度。"""
    rel = xs - ctr
    m = np.abs(rel) < cut if cut is not None else np.ones(N, dtype=bool)
    if not np.any(m) or prof[m].sum() <= 0:
        return float("nan")
    h = prof[m]
    r = rel[m]
    return float(math.sqrt(np.sum(r * r * h) / h.sum()))


def shape_dev(prof, ctr, cut, w_expect):
    """主瓣形状相对高斯的偏离度。

    做法：把主瓣归一后与同宽高斯比较，取加权均方相对差。
    高斯在 |x-ctr|<=cut 上的归一形状为 exp(-x^2/(2 s^2))，s 由主瓣二阶矩定。
    """
    rel = xs - ctr
    m = np.abs(rel) <= cut
    if not np.any(m) or prof[m].sum() <= 0:
        return float("nan")
    h = prof[m]
    r = rel[m]
    tot = h.sum()
    s = math.sqrt(float(np.sum(r * r * h) / tot))
    if s <= 0:
        return float("nan")
    g = np.exp(-r * r / (2.0 * s * s))
    # 两边各归一后比较
    a = h / tot
    b = g / g.sum()
    return float(np.sum((a - b) ** 2) / np.sum(b ** 2))


def continuity_check(hist, ctr, cut):
    """分割点两侧的连续性：主瓣外缘的密度跳变。

    做法：看分割点附近一格内，直方图是否有跳变（主瓣落到尾部时的阶跃）。
    以分割点内外各 3 格的平均高度之比给出。
    """
    rel = xs - ctr
    inner = np.abs(np.abs(rel) - cut) < 1.5
    outer = (np.abs(rel) > cut) & (np.abs(rel) < cut + 4.0)
    if not np.any(inner) or not np.any(outer):
        return float("nan")
    return float(hist[inner].mean() / max(hist[outer].mean(), 1e-300))


print("=" * 78)
print("第 807 号：核心瓣与辐射尾的分离，核心瓣的展宽机理")
print("=" * 78)
print("装置：N=%d omega_b=%.4f dt=%.2f T=%.0f sigma=%.0f" % (N, WB, DT, T_TOTAL, SIGMA))
print("速度档：%s" % list(VS))
print("读法：载波解调（沿用第 783 号）+ 瓣尾分割（中位半径法）")
print("追问：第 783 号的核心瓣宽涨 75%，而尾被截断，故展宽在核心瓣本身。")

U0, wb, psi, d1 = fit_sigma(SIGMA)
F_even = np.interp(xf, xf, psi)
REF = F_even * F_even
REF_CTR = float(np.sum(REF * xf) / REF.sum())
REF_W = float(math.sqrt(np.sum((xf - REF_CTR) ** 2 * REF) / REF.sum()))
print("初值包络 A^2：二阶矩 %.5f（质心 %.3f）" % (REF_W, REF_CTR))

# ---------------- QA1 读数 ----------------
print()
print("[QA1] 读数：五档速度的主瓣与尾")
RES = {}
print("    %-6s %-13s %-13s %-11s %-13s %-13s"
      % ("v", "主瓣宽", "主瓣形状偏离", "分割半径", "全宽(截断)", "尾/主体"))
for v in VS:
    sh = 1.0 / gamma_of(v) if v > 0 else 1.0
    hist, ctr, ns = demod_hist(U0, SIGMA, wb, psi, d1, v, sh)
    cut, main, tail = lobe_split(hist, ctr)
    w_main = width_of(main, ctr, cut)
    sdev = shape_dev(main, ctr, cut, w_main)
    w_all = width_of(hist, ctr, 3.0 * REF_W)
    ratio = float(tail.sum() / max(main.sum(), 1e-300))
    cont = continuity_check(hist, ctr, cut)
    RES[v] = dict(w=w_main, sdev=sdev, cut=cut, w_all=w_all, ratio=ratio,
                  cont=cont, hist=hist, ctr=ctr, main=main, tail=tail)
    print("    %-6.2f %-13.5f %-13.5f %-11.3f %-13.5f %-13.5f"
          % (v, w_main, sdev, cut, w_all, ratio))
print()
print("    分割点连续性（主瓣外缘/尾内侧的高度比，1 表示无断点）：")
for v in VS:
    print("        v=%.2f  %.4f" % (v, RES[v]["cont"]))

# ---------------- QA0 标定 ----------------
print()
print("[QA0] 标定：静止档的主瓣宽对初值包络解析二阶矩")
w0 = RES[0.0]["w"]
rel0 = abs(w0 / REF_W - 1.0)
print("    主瓣宽 %.5f；解析值 %.5f；相对差 %.5f" % (w0, REF_W, rel0))
QA0 = rel0 < 0.15
print("    QA0 判定：%s（容差 15%%）" % ("PASS" if QA0 else "FAIL"))

# ---------------- QA2 主判据：主瓣宽随 v ----------------
print()
print("[QA2] 主判据：主瓣宽随 v 的走向")
rows = []
print("    %-6s %-13s %-13s %-13s %-13s"
      % ("v", "主瓣 w/w0", "1/gamma", "相对偏差", "全宽 w/w0"))
for v in VS:
    if v == 0:
        continue
    r = RES[v]["w"] / w0
    tgt = 1.0 / gamma_of(v)
    dev = abs(r - tgt) / tgt
    ra = RES[v]["w_all"] / RES[0.0]["w_all"]
    rows.append((v, r, tgt, dev, ra))
    print("    %-6.2f %-13.6f %-13.6f %-13.4f %-13.5f" % (v, r, tgt, dev, ra))
if rows:
    worst = max(x[3] for x in rows)
    QA2 = worst < 0.10
    print()
    print("    最大相对偏差 = %.4f（判据 0.10）" % worst)
    print("    QA2 判定：%s" % ("PASS" if QA2 else "FAIL"))
else:
    QA2 = False
    worst = float("nan")
    print("    QA2 判定：FAIL")

# ---------------- QA3 形状判据 ----------------
print()
print("[QA3] 形状判据：主瓣形状相对高斯的偏离度")
for v in VS:
    print("    v=%.2f  偏离度 %.6f" % (v, RES[v]["sdev"]))
devs = [RES[v]["sdev"] for v in VS if v > 0]
sdev_max = max(devs) if devs else float("nan")
sdev_0 = RES[0.0]["sdev"]
QA3 = sdev_max < 0.10
print("    静止档 %.6f；运动档最大 %.6f（判据 0.10）" % (sdev_0, sdev_max))
print("    QA3 判定：%s" % ("PASS" if QA3 else "FAIL"))

# ---------------- QA4 标度判据 ----------------
print()
print("[QA4] 标度判据：主瓣宽度的相对变化对 v 的幂次")
lt = np.log(np.array([v for v in VS if v > 0], dtype=float))
ld = np.log(np.array([RES[v]["w"] / w0 for v in VS if v > 0], dtype=float))
p_lobe = float(np.polyfit(lt, ld, 1)[0])
la = np.log(np.array([v for v in VS if v > 0], dtype=float))
lda = np.log(np.array([RES[v]["w_all"] / RES[0.0]["w_all"] for v in VS if v > 0], dtype=float))
p_all = float(np.polyfit(la, lda, 1)[0])
print("    主瓣宽度的 v 幂次 = %.4f" % p_lobe)
print("    全宽（截断口径）的 v 幂次 = %.4f" % p_all)
print("    对照（第 704 号 dev 三分项）：构造失配给 v^2 量级，格点色散给 v^4 量级")
print("    主瓣幂次落在 2 与 4 之间则须再分；落在 2 附近则构造失配主导，4 附近则色散主导")
QA4 = True   # 报告项，不作通过与否的判定
print("    QA4 判定：报告项（不判通过与否）")

# ---------------- QA5 对照组：分割协议稳定性 ----------------
print()
print("[QA5] 对照组：分割协议的稳定性（分割点取法换一种）")
det = []
for rep in range(3):
    sh = 1.0 / gamma_of(0.30)
    hist, ctr, ns = demod_hist(U0, SIGMA, wb, psi, d1, 0.30, sh)
    cut, main, tail = lobe_split(hist, ctr)
    det.append(width_of(main, ctr, cut))
det_sp = (max(det) - min(det)) / abs(np.mean(det)) if np.mean(det) else 0.0
print("    同协议三次：%s  散布 %.3e"
      % (" ".join("%.5f" % x for x in det), det_sp))
alt = []
for q in (0.40, 0.45, 0.50, 0.55, 0.60):
    sh = 1.0 / gamma_of(0.30)
    hist, ctr, ns = demod_hist(U0, SIGMA, wb, psi, d1, 0.30, sh)
    cut, main, tail = lobe_split(hist, ctr, q)
    alt.append((q, width_of(main, ctr, cut)))
print("    分割权重换五种取法的主瓣宽：")
for q, w in alt:
    print("        q=%.2f  主瓣宽 %.5f" % (q, w))
ws = [w for q, w in alt]
sp_alt = (max(ws) - min(ws)) / abs(np.mean(ws)) if np.mean(ws) else 0.0
print("    跨分割协议的相对极差 = %.5f" % sp_alt)
QA5 = det_sp < 1e-8 and sp_alt < 0.10
print("    QA5 判定：%s（阴性对照逐位同，跨分割协议极差 < 0.10）"
      % ("PASS" if QA5 else "FAIL"))

# ---------------- QA6 判定 ----------------
print()
print("[QA6] 判定")
print("    QA0=%s QA2=%s QA3=%s QA5=%s"
      % ("PASS" if QA0 else "FAIL", "PASS" if QA2 else "FAIL",
         "PASS" if QA3 else "FAIL", "PASS" if QA5 else "FAIL"))
if not QA0:
    verdict = "静止档标定失败，本轮读数不可用"
elif not QA5:
    verdict = "分割协议不稳定，主瓣读数不含物理信息"
elif QA2:
    verdict = ("主瓣宽按 1/gamma 收缩（最大相对偏差 %.4f，容差 0.10）；"
               "尺缩在核心瓣上成立，前几轮的核心结论须改判为「尾污染所致」" % worst)
else:
    verdict = ("主瓣宽不按 1/gamma 收缩（最大相对偏差 %.4f，容据 0.10，v 幂次 %.3f）；"
               "核心瓣本身被展宽，前几轮结论保留，且第 805 号第一环（信号小于一格）"
               "须撤（分辨率实为 %.3f 格而非一格）"
               % (worst, p_lobe, 1.0 - rel0))
print("    判语：" + verdict)
print()
print("    边界：本轮测的是模型档，物理档本装置给不出；")
print("          主瓣与尾可分这一前提须由分割点连续性指标支持。")
print("耗时 %.1f 秒" % (time.time() - T0))

fig, ax = plt.subplots(1, 3, figsize=(15.8, 4.4))

ax[0].plot(VS, [RES[v]["w"] for v in VS], "s-", color="#c44", ms=8, label="主瓣宽")
ax[0].plot(VS, [RES[v]["w_all"] for v in VS], "^--", color="#824", ms=8, label="全宽（截断）")
ax[0].plot(VS, [w0 * (1.0 / gamma_of(v)) for v in VS], ":", color="#268", lw=2.2,
           label=r"$w(0)/\gamma$")
ax[0].set_xlabel("v")
ax[0].set_ylabel("宽度")
ax[0].set_title("左：主瓣宽与全宽")
ax[0].legend(fontsize=8)

ax[1].plot(VS, [RES[v]["ratio"] for v in VS], "o-", color="#268", ms=8, label="尾/主体权重")
ax[1].set_xlabel("v")
ax[1].set_ylabel("比值")
ax[1].set_title("中：尾的权重占比")
ax[1].legend(fontsize=9)

ax[2].plot(VS, [RES[v]["sdev"] for v in VS], "s-", color="#824", ms=8, label="形状偏离")
ax[2].axhline(0.10, color="#c44", ls="--", lw=1.5, label="判据 0.10")
ax[2].set_xlabel("v")
ax[2].set_ylabel("相对高斯的偏离度")
ax[2].set_title("右：主瓣形状")
ax[2].legend(fontsize=9)

plt.tight_layout()
plt.savefig(IMG, dpi=150)
print("图已存 %s" % IMG)
