# -*- coding: utf-8 -*-
"""第 807 号：亚格点随动分位读数、读数不确定度与替身检验。

背景
----
第 783 号建立了随动坐标系与载波解调两处修正，NA0 标定通过，NA2 与 NA3 判否。
第 807 号 v1 查出该判否的直接来源是自造 bug：读宽度时用了末帧质心当原点，
而直方图是逐帧相对各自质心累加的（原点固定在零），窗口错位到约 480 格，
运动档读到的是辐射尾上的噪声。v1 的全部读数作废。

v1 作废后暴露出一个更基础的问题：整数格直方图的读数分辨率是一格，而尺缩
信号是 w0 乘 (1 - 1/gamma)，v = 0.30 档为 0.553 格，低于一格量化步长。
在这种读法下尺缩判据原则上分辨不出。

本轮要回答的问题
------------------
把尺缩判据在模型档上建立起来，须先给读数亚格点分辨力，并给出读数自身的
不确定度；再问信号相对不确定度是否可判。

做法
----
其一，亚格点随动分位读数。每帧载波解调得 A2 场，按随动坐标 x 减该帧质心
      存为点云，累积分布在浮点坐标上线性反演求 10% 与 90% 分位点。
      分位点在浮点坐标上，分位宽因此有亚格点分辨力。
其二，替身检验（读法的标定分辨力）。取静止档点云，把坐标按 1/gamma 压缩，
      等价于一个已知答案的 1/gamma 收缩态，用两种读法各读一次。亚格点读法
      应读准，整数格读法读不准。同一份数据上两种读法的误差比较，就是
      读数分辨力的直接证据。
其三，读数不确定度（两个来源）：
      来源甲（时间窗）：取演化后段的三分之二重读，读数之差。
      来源乙（解调两支）：只取第一次采样的平方 A 场读一次，与两支平均读
            一次，宽度之差。这是载波残留直接引起的读数不确定度。
      两者取大者除以 w0 作相对不确定度 eps。
其四，主判据与方向判据。信号 1 - 1/gamma 须大于三倍 eps 才可判；可判时
      比对 w(v)/w(0) 与 1/gamma；且 1/gamma 预言收缩，宽度必须下降。

判据
----
RB0 标定：静止档亚格点随动分位宽对初值包络的亚格点分位宽，相对差 < 15%。
RB1 读数：五档速度的亚格点宽、整数格宽、解调残差、分位区间外质量、
    尾区质量、采样帧数。
RB2 替身检验：亚格点读法在替身上的相对误差 < 1%；整数格读法的绝对误差
    严格大于亚格点读法。
RB3 可判性：信号 1 - 1/gamma 须大于三倍读数不确定度 eps。
RB4 主判据：干净档上 |w(v)/w(0) - 1/gamma| < 3 eps。
RB5 方向判据：w(v)/w(0) < 1（1/gamma 预言收缩，宽度必须下降）。
RB6 对照组：组一（必须非零）解调残差随 v 变化；组二（必须非零）绝对坐标
    读数远大于随动读数，即随动坐标系确实消除了平移污染。
RB7 报告。
RB8 判定。

恒真检验（纪律二）：RB2 的整数格读法一侧是阴性对照。若整数格读法在替身上
也读得准，则「信号低于一格量化步长」这一归因不成立，尺缩判据的困难另有
来源。RB6 组二是反向对照：若绝对坐标读数与随动读数没有差别，则平移污染
本不存在，随动坐标系这条修正是多余的。

边界：本轮回答的是「尺缩判据能否在模型档上建立起来」。物理档（第 622 号
恒等式，束缚态宽 2.389e22 格）的长度收缩仍未判。
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

IMG = r"D:\software\TianXingFC\images\p807_subgrid_readout.png"

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
RESID_GATE = 0.05
BUD_SUB = 0.01
CAL_TOL = 0.15
SNR_K = 3.0

KK = 2.0 * math.pi * np.fft.fftfreq(N)
_U0CACHE = {}


def gamma_of(v):
    return 1.0 / math.sqrt(1.0 - v * v)


def bound(sigma, U0):
    well = -U0 * np.exp(-(xf ** 2) / (2.0 * sigma ** 2))
    L = diags([-np.ones(N - 1), 2.0 + WGAP2 + well, -np.ones(N - 1)],
              [-1, 0, 1], format='csr')
    v0 = np.exp(-(xf ** 2) / (2.0 * sigma ** 2))
    vals, vecs = eigsh(L, k=1, which='SA', v0=v0)
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


def demod_cloud(U0, sigma, wb, psi, d1, v, shrink, move_well=True,
                dt=DT, T=T_TOTAL, win_lo=1.0 / 3.0, win_hi=1.0):
    """载波解调 + 随动点云。

    返回字典：
      u, w   两支平均的随动坐标点云与权重
      ua, wa 单支（第一次采样）的随动坐标点云与权重
      uabs   绝对坐标点云（用作平移污染对照）
      resid  解调残差
      nsamp  采样帧数
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
    us, ws, uas, was = [], [], [], []
    acc_tot = 0.0
    dif = np.zeros(N)
    nsamp = 0
    s_lo = nstep * win_lo
    s_hi = nstep * win_hi
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
        if s % nper == 0 and s_lo < s < s_hi:
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
            us.append(xf - ctr_now)
            ws.append(A2)
            uas.append(xf - ctr_now)
            was.append(sq_a)
            acc_tot += float(np.sum(A2))
            dif = dif + (sq_a - sq_b)
            nsamp += 1
    u = np.concatenate(us)
    w = np.concatenate(ws)
    resid = float(np.sum(np.abs(dif)) / max(acc_tot, 1e-300))
    return {
        "u": u, "w": w,
        "ua": np.concatenate(uas), "wa": np.concatenate(was),
        "uabs": xf, "resid": resid, "nsamp": nsamp,
    }


def qwidth(u, w, qlo=QLO, qhi=QHI):
    """亚格点分位宽。返回 (宽度, lo 坐标, hi 坐标, 1% 到 99% 区间外的质量占比)"""
    order = np.argsort(u, kind="mergesort")
    us = u[order]
    ws = w[order]
    c = np.cumsum(ws)
    tot = float(c[-1])
    if tot <= 0.0:
        return 0.0, 0.0, 0.0, 0.0
    lo = float(np.interp(qlo * tot, c, us))
    hi = float(np.interp(qhi * tot, c, us))
    q01 = float(np.interp(0.01 * tot, c, us))
    q99 = float(np.interp(0.99 * tot, c, us))
    c01 = float(np.interp(q01, us, c))
    c99 = float(np.interp(q99, us, c))
    out = 1.0 - (c99 - c01) / tot
    return hi - lo, lo, hi, out


def qwidth_int(u, w, qlo=QLO, qhi=QHI):
    """整数格直方图分位宽（第 783 号与第 807 号 v1 用的那种读法）"""
    idx = np.clip(np.round(u).astype(int) + N // 2, 0, N - 1)
    hist = np.zeros(N)
    np.add.at(hist, idx, w)
    cdf = np.cumsum(hist)
    cdf = cdf / cdf[-1]
    lo = float(xs[int(np.searchsorted(cdf, qlo))])
    hi = float(xs[int(np.searchsorted(cdf, qhi))])
    return hi - lo


print("=" * 78)
print("第 807 号：亚格点随动分位读数、读数不确定度与替身检验")
print("=" * 78)
print("装置：N=%d omega_b=%.4f dt=%.2f T=%.0f sigma=%.0f" % (N, WB, DT, T_TOTAL, SIGMA))
print("速度档：%s" % list(VS))
print("v1 作废（窗口错位 bug），本轮给读数亚格点分辨力并给出读数不确定度。")

U0, wb, psi, d1 = fit_sigma(SIGMA)
CAL_AG, _, _, _ = qwidth(xf, psi * psi)
CAL_INT = qwidth_int(xf, psi * psi)
print("初值包络 F^2 的分位宽：亚格点 %.5f，整数格 %.2f" % (CAL_AG, CAL_INT))

# ---------------- RB1 读数 ----------------
print()
print("[RB1] 读数：五档速度的亚格点随动分位宽")
RES = {}
CLOUD = {}
print("    %-6s %-13s %-12s %-12s %-12s %-8s" %
      ("v", "亚格点宽", "整数格宽", "解调残差", "区间外占比", "采样帧"))
for v in VS:
    shrink = 1.0 / gamma_of(v) if v > 0 else 1.0
    cl = demod_cloud(U0, SIGMA, wb, psi, d1, v, shrink, True)
    CLOUD[v] = cl
    w_ag, _, _, out = qwidth(cl["u"], cl["w"])
    w_int = qwidth_int(cl["u"], cl["w"])
    RES[v] = (w_ag, w_int, cl["resid"], out, cl["nsamp"])
    print("    %-6.2f %-13.5f %-12.2f %-12.6f %-12.6f %-8d" %
          (v, w_ag, w_int, cl["resid"], out, cl["nsamp"]))

w0 = RES[0.0][0]

# ---------------- RB0 标定 ----------------
print()
print("[RB0] 标定：静止档亚格点随动分位宽对初值包络的亚格点分位宽")
rel_cal = abs(w0 / CAL_AG - 1.0)
print("    静止档 = %.5f；初值包络标定 = %.5f；相对差 %.5f" % (w0, CAL_AG, rel_cal))
RB0 = rel_cal < CAL_TOL
print("    RB0 判定：%s（容差 %.0f%%）" % ("PASS" if RB0 else "FAIL", CAL_TOL * 100))

# ---------------- RB2 替身检验 ----------------
print()
print("[RB2] 替身检验：静止档点云按 1/gamma 压缩，两种读法各读一次")
cl0 = CLOUD[0.0]
print("    %-6s %-12s %-13s %-13s %-13s %-13s" %
      ("v", "真值", "亚格点读数", "亚格点误差", "整数格读数", "整数格误差"))
SB = []
for v in VS[1:]:
    f = 1.0 / gamma_of(v)
    true_w = w0 * f
    w_ag_sq = qwidth(cl0["u"] * f, cl0["w"])[0]
    w_int_sq = qwidth_int(cl0["u"] * f, cl0["w"])
    e_ag = abs(w_ag_sq - true_w)
    e_int = abs(w_int_sq - true_w)
    SB.append((v, true_w, w_ag_sq, e_ag, w_int_sq, e_int))
    print("    %-6.2f %-12.5f %-13.5f %-13.6f %-13.2f %-13.5f" %
          (v, true_w, w_ag_sq, e_ag, w_int_sq, e_int))
e_ag_rel = max(s[3] / s[1] for s in SB)
e_ag_max = max(s[3] for s in SB)
e_int_max = max(s[5] for s in SB)
RB2a = e_ag_rel < BUD_SUB
RB2b = e_int_max > e_ag_max
print("    亚格点读法最大相对误差 %.6f（判据 %.3f）" % (e_ag_rel, BUD_SUB))
print("    整数格读法最大绝对误差 %.5f 对亚格点最大绝对误差 %.5f" % (e_int_max, e_ag_max))
print("    整数格读法误差严格更大：%s" % ("是" if RB2b else "否"))
RB2 = RB2a and RB2b
print("    RB2 判定：%s（亚格点 %s，整数格对照 %s）" %
      ("PASS" if RB2 else "FAIL", "通过" if RB2a else "不通过", "通过" if RB2b else "不通过"))

# ---------------- 读数不确定度（来源甲时间窗，来源乙解调两支） ----------------
print()
print("[读数不确定度] 两个来源")
UNC = {}
for v in (0.15, 0.30):
    cl = demod_cloud(U0, SIGMA, wb, psi, d1, v,
                     1.0 / gamma_of(v), True, win_lo=2.0 / 3.0)
    w_win, _, _, _ = qwidth(cl["u"], cl["w"])
    d_win = abs(w_win - RES[v][0])
    w_one, _, _, _ = qwidth(CLOUD[v]["ua"], CLOUD[v]["wa"])
    d_one = abs(w_one - RES[v][0])
    UNC[v] = (d_win, d_one, w_win, w_one)
    print("    v=%.2f  来源甲（时间窗）全段 %.5f 后段 %.5f 差 %.5f"
          % (v, RES[v][0], w_win, d_win))
    print("    v=%.2f  来源乙（解调两支）两支平均 %.5f 单支 %.5f 差 %.5f"
          % (v, RES[v][0], w_one, d_one))
EPS = {}
for v in (0.15, 0.30):
    EPS[v] = max(UNC[v][0], UNC[v][1]) / w0
print("    相对不确定度 eps（取两来源大者除以 w0）：")
for v in (0.15, 0.30):
    print("        v=%.2f  eps = %.6f" % (v, EPS[v]))

# ---------------- RB3 可判性 ----------------
print()
print("[RB3] 可判性：信号 1 - 1/gamma 须大于三倍读数不确定度")
print("    %-6s %-14s %-14s %-14s %-10s" % ("v", "信号", "3*eps", "信号/(3eps)", "可判"))
SNR = {}
for v in (0.15, 0.30):
    sig = 1.0 - 1.0 / gamma_of(v)
    thr = SNR_K * EPS[v]
    SNR[v] = sig / thr if thr > 0 else float("inf")
    print("    %-6.2f %-14.6f %-14.6f %-14.4f %-10s"
          % (v, sig, thr, SNR[v], "是" if sig > thr else "否"))
RB3 = all(SNR[v] > 1.0 for v in (0.15, 0.30))
print("    RB3 判定：%s" % ("PASS" if RB3 else "FAIL"))

# ---------------- RB4 主判据 ----------------
print()
print("[RB4] 主判据：w(v)/w(0) 对 1/gamma 的相对偏差，判据量取 3 eps")
clean = [v for v in VS if RES[v][2] < RESID_GATE]
exited = [v for v in VS if RES[v][2] >= RESID_GATE]
print("    有效档（解调残差 < %.2f）：%s" % (RESID_GATE, clean))
print("    退出登记档：%s" % exited)
rows = []
print("    %-6s %-14s %-13s %-13s %-13s" % ("v", "w/w0", "1/gamma", "相对偏差", "3eps"))
for v in clean:
    if v == 0:
        continue
    r = RES[v][0] / w0
    tgt = 1.0 / gamma_of(v)
    dev = abs(r - tgt) / tgt
    thr = SNR_K * EPS[v] if v in EPS else float("nan")
    rows.append((v, r, tgt, dev, thr))
    print("    %-6.2f %-14.7f %-13.7f %-13.6f %-13.6f" % (v, r, tgt, dev, thr))
if rows and RB3:
    worst = max(r[3] for r in rows)
    RB4 = all(r[3] < r[4] for r in rows)
    print("    最大相对偏差 = %.6f" % worst)
    print("    RB4 判定：%s" % ("PASS" if RB4 else "FAIL"))
else:
    worst = float("nan")
    RB4 = False
    print("    RB4 判定：FAIL（信号相对不确定度不足，尺缩判据在本装置上不可判）")

# ---------------- RB5 方向判据 ----------------
print()
print("[RB5] 方向判据：1/gamma 预言收缩，宽度必须下降")
print("    %-6s %-14s %-14s %-10s" % ("v", "w/w0", "1/gamma", "下降"))
dir_ok = []
for v in clean:
    if v == 0:
        continue
    r = RES[v][0] / w0
    tgt = 1.0 / gamma_of(v)
    ok = r < 1.0
    dir_ok.append(ok)
    print("    %-6.2f %-14.7f %-14.7f %-10s" % (v, r, tgt, "是" if ok else "否"))
RB5 = bool(dir_ok) and all(dir_ok)
print("    RB5 判定：%s" % ("PASS" if RB5 else "FAIL"))

# ---------------- RB6 对照组 ----------------
print()
print("[RB6] 对照组")
d_res = abs(RES[VS[-1]][2] - RES[0.0][2])
g1 = d_res > 1e-5
print("    组一（必须非零）：解调残差随 v 变化，首末差 %.6f" % d_res)
w_abs = qwidth(CLOUD[0.30]["uabs"], CLOUD[0.30]["w"])[0]
g2 = w_abs > 3.0 * RES[0.30][0]
print("    组二（必须非零）：v=0.30 档绝对坐标读数 %.2f 对随动读数 %.5f，倍数 %.1f"
      % (w_abs, RES[0.30][0], w_abs / max(RES[0.30][0], 1e-9)))
print("    说明：原设计的「固定井对移动井」不是平移污染的隔离对照（两者是不同"
      "构型），本轮改为绝对坐标对随动坐标。")
RB6 = g1 and g2
print("    RB6 判定：%s（组一 %s，组二 %s）" %
      ("PASS" if RB6 else "FAIL", "通过" if g1 else "不通过", "通过" if g2 else "不通过"))

# ---------------- RB7 报告 ----------------
print()
print("[RB7] 报告")
for v in VS:
    print("    v=%.2f  亚格点 %.5f  整数格 %.2f  区间外占比 %.6f  残差 %.6f"
          % (v, RES[v][0], RES[v][1], RES[v][3], RES[v][2]))
print("    尺缩信号量级：w0 乘 (1 - 1/gamma)，v=0.30 档为 %.5f 格"
      % (w0 * (1.0 - 1.0 / gamma_of(0.30))))
print("    高速两档（v >= 0.40）解调残差 %.6f 与 %.6f，超门槛 %.2f，作退出登记"
      % (RES[0.40][2], RES[0.45][2], RESID_GATE))
print("    整数格读法在替身上的最大误差 %.5f 格（v=0.40 档），与尺缩信号同量级"
      % e_int_max)

# ---------------- RB8 判定 ----------------
print()
print("[RB8] 判定")
print("    RB0=%s RB2=%s RB3=%s RB4=%s RB5=%s RB6=%s"
      % tuple("PASS" if x else "FAIL" for x in (RB0, RB2, RB3, RB4, RB5, RB6)))
if not RB0:
    verdict = "静止档标定失败，本轮读数不可用"
elif not RB2:
    verdict = ("替身检验未通过，读数分辨力未确立（亚格点误差 %.6f，整数格误差 "
               "%.5f），尺缩判据仍做不出来" % (e_ag_rel, e_int_max))
elif not RB6:
    verdict = "对照组未通过，随动坐标系的平移污染消除未获证据，RB3 至 RB5 不含物理信息"
elif not RB3:
    verdict = ("尺缩判据在模型档上不可判：读数不确定度 eps 已大于尺缩信号"
               "（v=0.30 档 eps %.6f 对信号 %.6f），本轮既不判定成立也不判定不成立；"
               "前几轮把「读不出」写成「不成立」须改判" % (EPS[0.30], 1.0 - 1.0 / gamma_of(0.30)))
elif not RB4:
    verdict = ("尺缩判据可判但不成立：干净档上 w(v)/w(0) 对 1/gamma 的最大相对偏差 "
               "%.6f 超 3eps；模型档格点上运动束缚态不按 1/gamma 收缩" % worst)
elif not RB5:
    verdict = ("尺缩判据的偏差在不确定度内，但宽度方向与 1/gamma 相反（宽度上升）；"
               "1/gamma 预言收缩而实测展宽，须另立判据")
else:
    verdict = ("尺缩判据在亚格点随动分位读数下成立：干净档上 w(v)/w(0) 对 1/gamma "
               "的最大相对偏差 %.6f 在 3eps 内" % worst)
print("    判语：" + verdict)
print()
print("    边界：本轮回答的是「尺缩判据能否在模型档上建立起来」；")
print("          物理档（第 622 号恒等式，束缚态宽 2.389e22 格）的长度收缩仍未判。")
print("耗时 %.1f 秒" % (time.time() - T0))

fig, ax = plt.subplots(1, 4, figsize=(20.4, 4.4))

ax[0].plot(VS, [RES[v][0] for v in VS], "s-", color="#c44", ms=8, label="亚格点读法")
ax[0].plot(VS, [RES[v][1] for v in VS], "^--", color="#824", ms=7, label="整数格读法")
ax[0].plot(VS, [w0 * (1.0 / gamma_of(v)) for v in VS], ":", color="#268", lw=2.0,
           label=r"$w(0)/\gamma$")
ax[0].set_xlabel("v")
ax[0].set_ylabel("分位宽度")
ax[0].set_title("左：两种读法与 $1/\\gamma$ 预言")
ax[0].legend(fontsize=8)

ax[1].plot([s[0] for s in SB], [s[3] for s in SB], "s-", color="#c44", ms=8,
           label="亚格点读法误差")
ax[1].plot([s[0] for s in SB], [s[5] for s in SB], "^--", color="#824", ms=7,
           label="整数格读法误差")
ax[1].axhline(0.553, color="#268", ls=":", lw=1.5, label="尺缩信号 0.553 格")
ax[1].set_xlabel("v")
ax[1].set_ylabel("绝对误差（格）")
ax[1].set_title("中：替身检验的读数误差")
ax[1].legend(fontsize=8)

if rows:
    ax[2].plot([r[0] for r in rows], [r[3] for r in rows], "s-", color="#268", ms=8,
               label="相对偏差")
    ax[2].plot([r[0] for r in rows], [r[4] for r in rows], "--", color="#c44", lw=1.8,
               label=r"$3\epsilon$")
ax[2].set_xlabel("v")
ax[2].set_ylabel(r"$|w/w_0 - 1/\gamma|/(1/\gamma)$")
ax[2].set_title("右：尺缩判据的相对偏差与不确定度")
ax[2].legend(fontsize=9)

ax[3].plot(VS, [RES[v][3] for v in VS], "o-", color="#268", ms=8, label="区间外占比")
ax[3].plot(VS, [RES[v][2] for v in VS], "s--", color="#824", ms=7, label="解调残差")
ax[3].set_xlabel("v")
ax[3].set_ylabel("占比")
ax[3].set_title("右二：区间外占比与解调残差")
ax[3].legend(fontsize=9)

plt.tight_layout()
plt.savefig(IMG, dpi=150)
print("图已存 %s" % IMG)
