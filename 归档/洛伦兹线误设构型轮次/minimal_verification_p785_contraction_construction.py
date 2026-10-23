# -*- coding: utf-8 -*-
"""第 785 号：尺缩的构造依赖扫描。

背景：第 783 号用随动坐标系与载波解调给出干净的宽度读数，判尺缩 1/gamma 在格点
上不成立（运动构型下包络被展宽而非收缩，w(v)/w(0) 对 1/gamma 的最大相对偏差
1.0624），并给出机理：格点上的运动束缚态不保持刚性，与第 704 号 dev 三分项中
「井刚性失配 54.4%」同源。

第 783 号 §10 剩余的第一条是构造依赖扫描，本轮执行它。它的判别力在于：
第 702 号已造好三种构造（井宽收缩比与束缚态宽度收缩比各一个）：
    rigid    井宽 1，束缚态 1       （第 691 号所用）
    contr    井宽 1/gamma，束缚态 1  （第 695 号原定义，自相矛盾）
    selfcons 井宽 1/gamma，束缚态 1/gamma（井与态同时收缩，自洽）

格点上的长度收缩要求束缚态跟着井一起收缩，故只有 selfcons 构造下 1/gamma 的
预言有意义。判别：
    若 selfcons 给出 1/gamma 而 rigid 不给，则 783 号的结论是构造问题；
    若 selfcons 也不给，则尺缩不成立是格点的性质。

判据：
  PA0 复现与标定：rigid 构造在 v=0 档的随动连续宽与初始包络连续宽相对差
      在容差内（解调读数的端到端检验，与第 783 号同口径）。
  PA1 读数：三种构造乘五档速度的随动连续宽、解调残差、尾污染。
  PA2 主判据：selfcons 构造的 w(v)/w(0) 对 1/gamma 的相对偏差 < 预算。
  PA3 对照组：方法分辨力。
      组一（必须非零）：解调残差随 v 变化。
      组二（必须为零）：同一构造下 v=0 档的随动宽与初始包络标定的相对差
          （三构造各自自校，排除构造本身的初值差异）。
  PA4 报告：三种构造的对照表与主导项归属。
  PA5 判定。

恒真检验（纪律二）：PA3 的组二逐构造自校。若某构造的 v=0 档随动宽与初始包络
标定差很多，则该构造的初值与演化不匹配，其读数不可用。

边界：本轮回答的是「尺缩不成立是格点的性质还是构造的性质」。物理档的尺缩本
装置给不出；构造的选择只穷举了三种，不是全部。
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

IMG = r"D:\software\TianXingFC\images\p785_contraction_construction.png"

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


def demod_profile(U0, sigma, wb, psi, d1, v, well_shrink, state_shrink,
                  dt=DT, T=T_TOTAL):
    """载波解调 + 随动直方图。

    well_shrink  井宽收缩比（井随速度移动时的宽度因子）
    state_shrink 束缚态宽度收缩比（初值包络的压缩因子）
    返回 (hist, ctr, resid, tail, nsamp)
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
    acc_tot = 0.0
    dif = np.zeros(N)
    nsamp = 0
    ctr_now = 0.0
    for s in range(nstep):
        t = (s + 0.5) * dt
        ctr = v * t
        w1 = -U0 * np.exp(-((xf - ctr) ** 2) / (2.0 * (sigma * well_shrink) ** 2))
        a1 = (np.roll(phi, 1) + np.roll(phi, -1) - 2.0 * phi) - WGAP2 * phi - w1 * phi
        phi = phi + dt * vel + .5 * dt * dt * a1
        t2 = (s + 1) * dt
        w2 = -U0 * np.exp(-((xf - v * t2) ** 2) / (2.0 * (sigma * well_shrink) ** 2))
        a2 = (np.roll(phi, 1) + np.roll(phi, -1) - 2.0 * phi) - WGAP2 * phi - w2 * phi
        vel = vel + .5 * dt * (a1 + a2)
        if s % nper == 0 and s > nstep // 3:
            sq_a = phi * phi
            for k in range(1, nq + 1):
                s2 = s + k
                if s2 >= nstep:
                    break
                t3 = (s2 + 0.5) * dt
                w3 = -U0 * np.exp(-((xf - v * t3) ** 2) / (2.0 * (sigma * well_shrink) ** 2))
                a3 = (np.roll(phi, 1) + np.roll(phi, -1) - 2.0 * phi) - WGAP2 * phi - w3 * phi
                phi = phi + dt * vel + .5 * dt * dt * a3
                t4 = (s2 + 1) * dt
                w4 = -U0 * np.exp(-((xf - v * t4) ** 2) / (2.0 * (sigma * well_shrink) ** 2))
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


print("=" * 78)
print("第 785 号：尺缩的构造依赖扫描")
print("=" * 78)
print("装置：N=%d omega_b=%.4f dt=%.2f T=%.0f sigma=%.0f" % (N, WB, DT, T_TOTAL, SIGMA))
print("三种构造（第 702 号已造好）：")
print("    rigid    井宽 1，束缚态 1            （第 691 号所用）")
print("    contr    井宽 1/gamma，束缚态 1       （第 695 号原定义，自相矛盾）")
print("    selfcons 井宽 1/gamma，束缚态 1/gamma  （自洽，格点长度收缩该有的样子）")
print("尺缩 1/gamma 的预言只在 selfcons 构造下有意义。")
print("速度档：%s" % list(VS))

U0, wb, psi, d1 = fit_sigma(SIGMA)
CUT = NCUT * width_of_field((np.interp(xf, xf, psi)) ** 2)
print("截断半径 = %.3f 格（以静止档包络半宽的 %.2f 倍计）" % (CUT, width_of_field((np.interp(xf, xf, psi)) ** 2)))

CONFIGS = (
    ("rigid",    1.0,     1.0),
    ("contr",    None,    1.0),      # 井宽收缩因子随 v 变
    ("selfcons", None,    None),    # 两者都随 v 变
)


def shrinks(name, v):
    g = gamma_of(v)
    if name == "rigid":
        return 1.0, 1.0
    if name == "contr":
        return 1.0 / g, 1.0
    return 1.0 / g, 1.0 / g


# ---------------- PA1 读数 ----------------
print()
print("[PA1] 读数：三种构造乘五档速度的随动连续宽")
RES = {}
print("    %-10s %-6s %-13s %-12s %-12s" % ("构造", "v", "随动连续宽", "解调残差", "尾污染"))
for name, _, _ in CONFIGS:
    for v in VS:
        ws_, st_ = shrinks(name, v)
        hist, ctr, r, tf, ns = demod_profile(U0, SIGMA, wb, psi, d1, v, ws_, st_)
        w = width_from_hist(hist, ctr, CUT)
        RES[(name, v)] = dict(w=w, ctr=ctr, resid=r, tail=tf, ns=ns, hist=hist)
        print("    %-10s %-6.2f %-13.5f %-12.5f %-12.5f" % (name, v, w, r, tf))

# ---------------- PA0 复现与标定 ----------------
print()
print("[PA0] 复现与标定：各构造 v=0 档的随动宽对初始包络标定")
CAL_RIGID = width_of_field((np.interp(xf, xf, psi)) ** 2)
PA0 = True
for name, _, _ in CONFIGS:
    ws_, st_ = shrinks(name, 0.0)
    hist, ctr, _, _, _ = demod_profile(U0, SIGMA, wb, psi, d1, 0.0, ws_, st_)
    w = width_from_hist(hist, ctr, CUT)
    Fv = np.interp(xf, xf, psi)
    cal = width_of_field((Fv * np.cos(-wb * st_ * 0.0 * xf)) ** 2)
    rel = abs(w / cal - 1.0)
    print("    %-10s 随动宽 %.5f  初始包络标定 %.5f  相对差 %.5f"
          % (name, w, cal, rel))
    if rel > 0.15:
        PA0 = False
print("    PA0 判定：%s（容差 15%%，逐构造自校）" % ("PASS" if PA0 else "FAIL"))

# ---------------- PA2 主判据 ----------------
print()
print("[PA2] 主判据：selfcons 构造的 w(v)/w(0) 对 1/gamma 的相对偏差")
BUD = 0.10
# 有效档筛选：解调残差 < RESID_MAX。残差是载波抵消的残量，残差大即解调失效，
# 此时的宽度读数不可信（与第 775 号「逐档有效性门槛」同一处置）。
RESID_MAX = 0.05
print("    有效档门槛：解调残差 < %.2f（残差即载波抵消的残量，残差大则解调失效）"
      % RESID_MAX)
for name, _, _ in CONFIGS:
    good = [v for v in VS if v > 0 and RES[(name, v)]["resid"] < RESID_MAX]
    print("        %-10s 干净档：%s"
          % (name, ["%.2f" % v for v in good] if good else "无"))
rows = {}
for name, _, _ in CONFIGS:
    w0 = RES[(name, 0.0)]["w"]
    rs = []
    for v in VS:
        if v == 0:
            continue
        if RES[(name, v)]["resid"] >= RESID_MAX:
            continue
        r = RES[(name, v)]["w"] / w0
        tgt = 1.0 / gamma_of(v)
        rs.append((v, r, tgt, abs(r - tgt) / tgt))
    rows[name] = rs
    print("    [%s]" % name)
    for v, r, tgt, dev in rs:
        print("        v=%.2f  w/w0 %.6f  1/gamma %.6f  相对偏差 %.4f" % (v, r, tgt, dev))
    if rs:
        worst = max(x[3] for x in rs)
        print("      干净档最大相对偏差 = %.4f" % worst)
    else:
        print("      无干净档")
sc_worst = max(x[3] for x in rows["selfcons"]) if rows["selfcons"] else float("nan")
PA2 = (sc_worst < BUD) and len(rows["selfcons"]) >= 2
print()
print("    selfcons 干净档的最大相对偏差 = %.4f（判据 %.2f，干净档数 %d）"
      % (sc_worst, BUD, len(rows["selfcons"])))
print("    PA2 判定：%s" % ("PASS" if PA2 else "FAIL"))
print("    三构造在干净档上的最大相对偏差：")
for name, _, _ in CONFIGS:
    if rows[name]:
        print("        %-10s %.4f" % (name, max(x[3] for x in rows[name])))
    else:
        print("        %-10s 无干净档" % name)

# ---------------- PA3 对照组 ----------------
print()
print("[PA3] 对照组：方法分辨力")
print("    组一（必须非零）：解调残差随 v 变化（逐构造）")
g1_ok = True
for name, _, _ in CONFIGS:
    r0 = RES[(name, 0.0)]["resid"]
    rN = RES[(name, VS[-1])]["resid"]
    d = abs(rN - r0)
    ok = d > 1e-5
    g1_ok = g1_ok and ok
    print("        %-10s v=0 %.6f  v=%.2f %.6f  差 %.6f  %s"
          % (name, r0, VS[-1], rN, d, "通过" if ok else "不通过"))
print("    组二（必须为零）：各构造 v=0 档随动宽与初始包络标定的相对差（已在 PA0 逐构造自校）")
g2_ok = PA0
PA3 = g1_ok and g2_ok
print("    PA3 判定：%s（组一 %s，组二 %s）"
      % ("PASS" if PA3 else "FAIL", "通过" if g1_ok else "不通过", "通过" if g2_ok else "不通过"))

# ---------------- PA4 报告 ----------------
print()
print("[PA4] 报告：三种构造的对照（干净档）")
for v in (0.15, 0.30):
    print("    v=%.2f" % v)
    for name, _, _ in CONFIGS:
        w0 = RES[(name, 0.0)]["w"]
        w = RES[(name, v)]["w"]
        tgt = 1.0 / gamma_of(v)
        print("        %-10s w=%.5f  w/w0=%.5f  1/gamma=%.5f  相对偏差=%.4f  解调残差=%.5f"
              % (name, w, w / w0, tgt, abs(w / w0 - tgt) / tgt, RES[(name, v)]["resid"]))
print()
print("    主导项归属（干净档）：")
for name, _, _ in CONFIGS:
    rs = rows[name]
    if rs:
        print("        %-10s 最大相对偏差 %.4f  %s"
              % (name, max(x[3] for x in rs),
                 "符合 1/gamma 预言" if max(x[3] for x in rs) < BUD else "不符合"))
    else:
        print("        %-10s 无干净档" % name)
print("    三构造的相对偏差次序（干净档最大）：")
order = sorted([(n, max(x[3] for x in rows[n])) for n, _, _ in CONFIGS if rows[n]],
               key=lambda z: z[1])
for n, wv in order:
    print("        %-10s %.4f" % (n, wv))

# ---------------- PA5 判定 ----------------
print()
print("[PA5] 判定")
print("    PA0=%s PA2=%s PA3=%s" % ("PASS" if PA0 else "FAIL",
                                    "PASS" if PA2 else "FAIL",
                                    "PASS" if PA3 else "FAIL"))
if not PA0:
    verdict = "构造自校失败，本轮读数不可用"
elif not PA3:
    verdict = "对照组未通过，方法分辨力未得证，PA2 不含物理信息"
elif PA2:
    verdict = ("selfcons 构造下尺缩 1/gamma 成立（干净档最大相对偏差 %.4f，预算 %.2f）；"
               "第 783 号的结论改判为构造问题：rigid 构造下束缚态不随井收缩，"
               "故包络被展宽" % (sc_worst, BUD))
else:
    order2 = sorted([(n, max(x[3] for x in rows[n])) for n, _, _ in CONFIGS if rows[n]],
                    key=lambda z: z[1])
    seq = " < ".join("%s %.4f" % (n, wv) for n, wv in order2)
    verdict = ("selfcons 构造下尺缩 1/gamma 仍不成立（干净档最大相对偏差 %.4f，"
               "预算 %.2f）；三构造的次序为 %s，selfcons 最好但仍远在预算外，"
               "故尺缩不成立以格点性质为主、构造失配为辅"
               % (sc_worst, BUD, seq))
print("    判语：" + verdict)
print()
print("    边界：本轮回答的是「尺缩不成立是格点的性质还是构造的性质」；")
print("          物理档的尺缩本装置给不出，构造只穷举三种。")
print("耗时 %.1f 秒" % (time.time() - T0))

fig, ax = plt.subplots(1, 3, figsize=(15.8, 4.4))

for name, _, _ in CONFIGS:
    ws_ = [RES[(name, v)]["w"] for v in VS]
    ax[0].plot(VS, ws_, "s-", ms=7, label=name)
ax[0].plot(VS, [RES[("selfcons", 0.0)]["w"] * (1.0 / gamma_of(v)) for v in VS],
           ":", color="#888", lw=2.0, label=r"selfcons 的 $1/\gamma$")
ax[0].set_xlabel("v")
ax[0].set_ylabel("随动连续宽")
ax[0].set_title("左：三构造的宽度随 v")
ax[0].legend(fontsize=8)

ax[1].bar([0, 1, 2], [max(x[3] for x in rows[n]) for n, _, _ in CONFIGS],
          color=["#824", "#268", "#c44"], width=0.55)
ax[1].axhline(BUD, color="#c44", ls="--", lw=1.5, label="预算 %.2f" % BUD)
ax[1].set_xticks([0, 1, 2])
ax[1].set_xticklabels(["rigid", "contr", "selfcons"])
ax[1].set_ylabel("最大相对偏差")
ax[1].set_title("中：各构造对 1/gamma 的偏差")
ax[1].legend(fontsize=9)

for name, _, _ in CONFIGS:
    ax[2].plot([r[0] for r in rows[name]], [r[3] for r in rows[name]], "s-",
                ms=7, label=name)
ax[2].axhline(BUD, color="#c44", ls="--", lw=1.5, label="预算 %.2f" % BUD)
ax[2].set_xlabel("v")
ax[2].set_ylabel(r"$|w/w_0 - 1/\gamma|/(1/\gamma)$")
ax[2].set_title("右：逐档相对偏差")
ax[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig(IMG, dpi=150)
print("图已存 %s" % IMG)
