# -*- coding: utf-8 -*-
"""第 714 号：把速度推到高速端，用两个 Lorentz 候选的判决性差异分开它们。

背景：第 713 号在 v 从 0.25 到 0.50 的五档上判否了纯 v^2 标度，但 gamma-1 与
1-1/gamma 两个候选都通过留一检验且彼此只差 0.35 个百分点。两者的预言比恰为

    [K1 (gamma-1)] / [K2 (1-1/gamma)] = (K1/K2) * gamma

随 gamma 线性增长，v=0.50 处只差 15%，v=0.70 处差 25%，v=0.80 处差 49%。
故把速度推高是唯一的分开手段。本轮先做冒烟，判明装置在高速端能撑到哪一档，
再在有可用档的前提下做判决。

装置约束（本轮要实测的）：c0 正比于 |f(k_mod)|/(2 omega^2)，随着 v 升高
k_mod = omega_b*gamma*v 快速变大而 omega = omega_b/gamma 变小，c0 不是缓慢
增长而是按十几倍的速度膨胀（v=0.70 处约为 v=0.50 的 12.5 倍）。波包同时被
压缩 gamma 倍，格点采样变粗。这两件事都可能把读数推出有效区，故冒烟先行。

判据：
  OA0 冒烟：v in {0.55, 0.60, 0.65, 0.70, 0.75} 乘四档 xi，读 dev 与残差比。
      可用档的判据沿用第 696 号 P1 的门槛：残差比 < 0.30。
  OA1 复现：v=0.50 的 c1 复现第 713 号的 6.2212e-2。
  OA2 主判据：取最高可用档 v*，按第 713 号的两个 K 外推两候选的预言，
      与实测 c1(v*) 比较。判据为「落在谁的三倍合成标准误内」，三向判决：
      只支持一个、两个都排除、两个都分不开。
  OA3 判定。

合成标准误：sigma_tot^2 = sigma_c1^2 + (K 的相对标准误 * 预言值)^2，两候选各算。

恒真检验（纪律二）：两候选的预言在低速区只差一个常数（gamma-1 与 1-1/gamma
在小 v 都是 v^2/2），故低速档必然分不开。本轮必须报出「两候选预言之差」与
「实测标准误」的比值，若该比值小于 3，则判决不成立，须判分不开。
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

IMG = r"D:\software\TianXingFC\images\p714_high_velocity.png"

T0 = time.time()
N = 2048
xs = np.arange(N) - N // 2
xf = xs.astype(float)
WGAP2 = 1.0
WB = 0.8008
DT = 0.02
SIGS4 = (14.0, 20.0, 28.0, 42.0)
VS_HI = (0.55, 0.60, 0.65, 0.70, 0.75)

KK = 2.0 * math.pi * np.fft.fftfreq(N)
K2 = KK ** 2

# 第 713 号五档拟合所得（来源：p713_c1_velocity.py 的 NA2 段）
C1_P713 = {0.25: 1.3893e-2, 0.30: 2.0529e-2, 0.35: 2.8449e-2,
           0.40: 3.8223e-2, 0.50: 6.2212e-2}
K_GAMMA = 4.0984e-1
K_INVG = 4.5904e-1
RELSE_K_GAMMA = 0.0088  # 第 713 号 NA2 段输出
RELSE_K_INVG = 0.0079


def gamma_of(v):
    return 1.0 / math.sqrt(1.0 - v * v)


def c0_theory(v):
    """理论常数项 |f(k_mod)|/(2 omega^2)，k_mod = omega_b*gamma*v。"""
    g = gamma_of(v)
    km = WB * g * v
    f = 4.0 * math.sin(km / 2.0) ** 2 - km ** 2
    return abs(f) / (2.0 * (WB / g) ** 2)


_U0CACHE = {}


def bound(sigma, U0):
    well = -U0 * np.exp(-(xf ** 2) / (2.0 * sigma ** 2))
    L = diags([-np.ones(N - 1), 2.0 + WGAP2 + well, -np.ones(N - 1)],
              [-1, 0, 1], format='csr')
    vals, vecs = eigsh(L, k=1, which='SA')
    p = vecs[:, 0] / np.max(np.abs(vecs[:, 0]))
    d1 = np.real(np.fft.ifft(1j * KK * np.fft.fft(p)))
    return (math.sqrt(float(vals[0])) if vals[0] > 0 else 0.0), p, d1


def fit_U0(sigma):
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


def make_ic(psi, d1, v, style):
    g = gamma_of(v)
    if style == "boost":
        s0 = g * xf
        th0 = WB * g * v * xf
        c = g * v
        wfac = WB * g
        sgn = +1.0
    else:
        sh = 1.0 / g
        s0 = sh * xf
        th0 = -WB * sh * v * xf
        c = sh * v
        wfac = WB * sh
        sgn = -1.0
    F = np.interp(s0, xf, psi)
    Fp = np.interp(s0, xf, d1)
    return F * np.cos(th0), -c * Fp * np.cos(th0) + sgn * wfac * F * np.sin(th0)


def run(phi, vel, U0, wsig, v, evo, well_mode, dt=DT, T=1200.0, every=4,
        sample="interp"):
    g = gamma_of(v)
    cs = []
    nst = int(T / dt)
    gain = 1.0 if well_mode == "rigid" else g * g
    for s in range(nst):
        t = (s + .5) * dt
        w1 = -U0 * np.exp(-gain * ((xf - v * t) ** 2) / (2.0 * wsig ** 2))
        if evo == "lattice":
            acc = (np.roll(phi, 1) + np.roll(phi, -1) - 2.0 * phi) - WGAP2 * phi - w1 * phi
        else:
            acc = -np.real(np.fft.ifft(K2 * np.fft.fft(phi))) - WGAP2 * phi - w1 * phi
        phi = phi + dt * vel + .5 * dt * dt * acc
        t2 = (s + 1) * dt
        w2 = -U0 * np.exp(-gain * ((xf - v * t2) ** 2) / (2.0 * wsig ** 2))
        if evo == "lattice":
            acc2 = (np.roll(phi, 1) + np.roll(phi, -1) - 2.0 * phi) - WGAP2 * phi - w2 * phi
        else:
            acc2 = -np.real(np.fft.ifft(K2 * np.fft.fft(phi))) - WGAP2 * phi - w2 * phi
        vel = vel + .5 * dt * (acc + acc2)
        if s % every == 0:
            if sample == "round":
                cs.append(float(phi[N // 2 + int(round(v * t2))]))
            else:
                cs.append(float(np.interp(v * t2, xf, phi)))
    return np.array(cs)


def freq_refined(cs, dt_s, center, frac=0.25, ncoarse=2001, nfine=201, nstage=2):
    tt = np.arange(len(cs)) * dt_s
    y = cs - np.mean(cs)
    if float(np.std(y)) <= 0:
        return center, 1.0
    lo, hi = center * (1 - frac), center * (1 + frac)

    def scan(lo, hi, n):
        best, bw = None, None
        for w in np.linspace(lo, hi, n):
            A = np.stack([np.cos(w * tt), np.sin(w * tt)], axis=1)
            c, *_ = np.linalg.lstsq(A, y, rcond=None)
            r = float(np.sum((A @ c - y) ** 2))
            if best is None or r < best:
                best, bw = r, w
        return bw, best

    bw, best = scan(lo, hi, ncoarse)
    step = (hi - lo) / (ncoarse - 1)
    for _ in range(nstage):
        lo, hi = bw - 2 * step, bw + 2 * step
        bw, best = scan(lo, hi, nfine)
        step = (hi - lo) / (nfine - 1)
    A = np.stack([np.cos(bw * tt), np.sin(bw * tt)], axis=1)
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return bw, float(np.linalg.norm(A @ c - y) / max(np.linalg.norm(y), 1e-300))


def measure(sg, v, evo, well_mode, style, sample="interp"):
    U0, _, psi, d1 = fit_U0(sg)
    phi, vel = make_ic(psi, d1, v, style)
    cs = run(phi, vel, U0, sg, v, evo, well_mode, sample=sample)
    tgt = WB / gamma_of(v)
    wm, res = freq_refined(cs, 0.08, tgt)
    return abs(wm - tgt) / tgt, res


def fit_poly(xis, ys, terms):
    xi = np.asarray(xis, float)
    y = np.asarray(ys, float)
    X = np.stack([xi ** (-t) for t in terms], axis=1)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    rss = float(np.sum((y - pred) ** 2))
    ss = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - rss / max(ss, 1e-300)
    maxrel = float(np.max(np.abs(pred / y - 1)))
    dof = max(len(y) - len(terms), 1)
    try:
        cov = (rss / dof) * np.linalg.pinv(X.T @ X)
        se = list(np.sqrt(np.maximum(np.diag(cov), 0.0)))
    except Exception:
        se = [float("nan")] * len(terms)
    return coef, r2, maxrel, se, pred


print("=" * 78)
print("第 714 号：高速档冒烟与两个 Lorentz 候选的判决（判据 OA0 至 OA3）")
print("=" * 78)
print("装置：N=%d omega_b=%.4f xi in %s  dt=%.2f" % (N, WB, list(SIGS4), DT))
print("高速档：v in %s" % list(VS_HI))
print()
print("两候选的预言比 (K1/K2)*gamma：v=0.70 处 %.3f，v=0.75 处 %.3f"
      % (K_GAMMA / K_INVG * gamma_of(0.70), K_GAMMA / K_INVG * gamma_of(0.75)))
print("理论 c0 随 v 的膨胀：v=0.50 给 %.3e，v=0.70 给 %.3e（涨 %.1f 倍）"
      % (c0_theory(0.50), c0_theory(0.70), c0_theory(0.70) / c0_theory(0.50)))

# ---------------- OA0 冒烟 ----------------
print()
print("[OA0] 冒烟：高速五档的 dev 与残差比（线性插值采样）")
print("    %-6s %-13s %-13s %-13s %-13s %-9s"
      % ("v", "dev(xi=14)", "dev(20)", "dev(28)", "dev(42)", "最大残差比"))
SMOKE = {}
for v in VS_HI:
    devs, ress = [], []
    for sg in SIGS4:
        dev, res = measure(sg, v, "lattice", "rigid", "g692")
        devs.append(dev)
        ress.append(res)
    SMOKE[v] = dict(devs=devs, ress=ress)
    print("    %-6.2f %-13.4e %-13.4e %-13.4e %-13.4e %-9.3f"
          % (v, devs[0], devs[1], devs[2], devs[3], max(ress)))

USABLE = [v for v in VS_HI if max(SMOKE[v]["ress"]) < 0.30]
print()
print("    可用档（残差比全部 < 0.30）：%s" % (["%.2f" % v for v in USABLE] or "无"))
print("    不可用档：%s"
      % (["%.2f(%.3f)" % (v, max(SMOKE[v]["ress"])) for v in VS_HI
          if max(SMOKE[v]["ress"]) >= 0.30] or "无"))

# ---------------- OA0b 采样协议对照 ----------------
print()
print("[OA0b] 采样协议对照：单点取整对线性插值（xi=42，三档）")
CMPV = (0.50, 0.55, 0.60)
print("    %-6s %-13s %-9s %-13s %-9s"
      % ("v", "取整 dev", "残差比", "插值 dev", "残差比"))
PROTO = {}
for v in CMPV:
    dr, rr = measure(42.0, v, "lattice", "rigid", "g692", sample="round")
    di, ri = measure(42.0, v, "lattice", "rigid", "g692", sample="interp")
    PROTO[v] = (dr, rr, di, ri)
    print("    %-6.2f %-13.4e %-9.3f %-13.4e %-9.3f" % (v, dr, rr, di, ri))
fix = [v for v in CMPV if PROTO[v][1] >= 0.30 and PROTO[v][3] < 0.30]
print("    协议修复有效的档：%s" % (["%.2f" % v for v in fix] or "无"))
OA0b = len(fix) > 0
print("    OA0b 判定：%s"
      % ("PASS（高速档残差比由采样取整造成，可修）" if OA0b
         else "FAIL（换采样不改善，属物理边界）"))

# ---------------- OA0c 第 713 号所用档的残差比矩阵 ----------------
print()
print("[OA0c] 第 713 号所用六档的残差比矩阵（门槛 0.30）")
print("    %-6s %-9s %-9s %-9s %-9s %-7s"
      % ("v", "xi=14", "xi=20", "xi=28", "xi=42", "越界数"))
LOWV = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
LOW = {}
nbad = 0
for v in LOWV:
    row_dev, row_res = [], []
    for sg in SIGS4:
        dev, res = measure(sg, v, "lattice", "rigid", "g692")
        row_dev.append(dev)
        row_res.append(res)
    LOW[v] = (row_dev, row_res)
    nb = sum(1 for r in row_res if r >= 0.30)
    nbad += nb
    print("    %-6.2f %-9.3f %-9.3f %-9.3f %-9.3f %-7d"
          % (v, row_res[0], row_res[1], row_res[2], row_res[3], nb))
print("    全部 24 个 (v, xi) 组合中越门槛的个数：%d" % nbad)
OA0c = nbad == 0
print("    OA0c 判定：%s"
      % ("PASS（第 713 号所用档全部有效）" if OA0c
         else "FAIL（第 713 号含超门槛读数，其标度结论须限定）"))

# ---------------- OA1 复现 ----------------
print()
print("[OA1] 复现：v=0.50 的 c1 应给第 713 号的 6.2212e-2")
cf50, r2_50, _, se50, _ = fit_poly(SIGS4, LOW[0.50][0], [0, 1])
rel50 = abs(cf50[1] / C1_P713[0.50] - 1.0)
print("    本轮 c1(0.50) = %.4e；第 713 号 %.4e；相对差 %.3e（该档最大残差比 %.3f）"
      % (cf50[1], C1_P713[0.50], rel50, max(LOW[0.50][1])))
OA1 = rel50 < 2.0e-3
print("    OA1 判定：%s（容差 2e-3；读数复现不等于读数有效，有效性见 OA0c）"
      % ("PASS" if OA1 else "FAIL"))

# ---------------- OA2 主判据 ----------------
print()
print("[OA2] 主判据：最高可用档上两候选的预言与实测")
if not USABLE:
    print("    无可用档，判决不成立。")
    OA2 = False
    VSTAR = None
    c1s = None
    sig_c1 = None
else:
    VSTAR = max(USABLE)
    coefH, r2H, mrfH, seH, _ = fit_poly(SIGS4, SMOKE[VSTAR]["devs"], [0, 1])
    c1s = coefH[1]
    sig_c1 = seH[1]
    gs = gamma_of(VSTAR)
    predA = K_GAMMA * (gs - 1.0)
    predB = K_INVG * (1.0 - 1.0 / gs)
    sigA = math.sqrt(sig_c1 ** 2 + (RELSE_K_GAMMA * predA) ** 2)
    sigB = math.sqrt(sig_c1 ** 2 + (RELSE_K_INVG * predB) ** 2)
    dA = abs(c1s - predA) / sigA
    dB = abs(c1s - predB) / sigB
    gap = abs(predA - predB) / max(sig_c1, 1e-30)
    print("    v* = %.2f，四档 xi 拟合 R^2 = %.6f，最大残差 %.2f%%"
          % (VSTAR, r2H, 100 * mrfH))
    print("    实测 c1 = %.4e (1 sigma = %.3e)" % (c1s, sig_c1))
    print("    候选 gamma-1    预言 %.4e (sigma = %.2e)，偏差 %.1f sigma"
          % (predA, sigA, dA))
    print("    候选 1-1/gamma  预言 %.4e (sigma = %.2e)，偏差 %.1f sigma"
          % (predB, sigB, dB))
    print("    两候选预言之差 = %.4e，为实测标准误的 %.1f 倍" % (abs(predA - predB), gap))
    if gap < 3.0:
        OA2 = False
        print("    判决不成立：两候选预言之差不足三倍标准误（恒真检验不过）。")
    else:
        okA = dA < 3.0
        okB = dB < 3.0
        if okA and not okB:
            OA2 = True
            print("    OA2 判定：PASS，支持 gamma-1（1-1/gamma 被 %.1f sigma 排除）" % dB)
        elif okB and not okA:
            OA2 = True
            print("    OA2 判定：PASS，支持 1-1/gamma（gamma-1 被 %.1f sigma 排除）" % dA)
        elif okA and okB:
            OA2 = False
            print("    OA2 判定：FAIL，两个候选都在三倍标准误内，仍分不开。")
        else:
            OA2 = False
            print("    OA2 判定：FAIL，两个候选都被排除，标度形式需重新考虑。")

# ---------------- OA3 判定 ----------------
print()
print("[OA3] 判定")
print("    OA0 可用档数 = %d；OA0b=%s；OA0c=%s；OA1=%s；OA2=%s"
      % (len(USABLE), "PASS" if OA0b else "FAIL", "PASS" if OA0c else "FAIL",
         "PASS" if OA1 else "FAIL", "PASS" if OA2 else "FAIL"))
if not USABLE:
    verdict = ("高速档在插值采样下仍全部不可用，装置边界止于 v=0.50 以下，"
               "两候选无法在本装置上分开")
if not OA0c:
    verdict += "；且第 713 号所用档含超门槛读数，其标度结论须限定（见 OA0d）"
elif not OA1:
    verdict = "复现失败，本轮读数不可用"
elif OA2:
    verdict = "两个 Lorentz 候选已在 v*=%.2f 上分开，判语见上" % VSTAR
else:
    verdict = ("高速档可用但判决不成立（两候选仍分不开或均被排除），"
               "装置能给出的最大区分度不足")
print("    判语：" + verdict)
# ---------------- OA0d 只用全效档重做标度检验 ----------------
print()
print("[OA0d] 只用四档 xi 全部有效的 v 档重做标度检验")
CLEANV = [v for v in LOWV if max(LOW[v][1]) < 0.30]
print("    全效档：%s" % (["%.2f" % v for v in CLEANV] or "无"))


def fit_scale_1(xs, ys, form):
    A = np.array([[form(x)] for x in xs])
    K, *_ = np.linalg.lstsq(A, np.asarray(ys), rcond=None)
    pred = (A @ K).ravel()
    y = np.asarray(ys)
    return float(K[0]), float(np.max(np.abs(pred / y - 1.0)))


if len(CLEANV) < 3:
    print("    全效档少于三档，无法重做标度检验，只在注记中报告。")
    OA0d = None
else:
    C1H = {}
    for v in CLEANV:
        cf, r2c, _, sec, _ = fit_poly(SIGS4, LOW[v][0], [0, 1])
        C1H[v] = (cf[1], sec[1])
    lv2 = [C1H[v][0] for v in CLEANV]
    CANDS3 = (
        ("gamma-1", lambda x: gamma_of(x) - 1.0),
        ("1-1/gamma", lambda x: 1.0 - 1.0 / gamma_of(x)),
        ("v^2", lambda x: x * x),
    )
    for nm, fn in CANDS3:
        K, rel = fit_scale_1(CLEANV, lv2, fn)
        loo = []
        for vout in CLEANV:
            tr = [v for v in CLEANV if v != vout]
            Kt, _ = fit_scale_1(tr, [C1H[v][0] for v in tr], fn)
            loo.append(abs(Kt * fn(vout) / C1H[vout][0] - 1.0))
        print("    %-11s K=%.4e  全效档最大残差 %.4f  留一最大误差 %.4f"
              % (nm, K, rel, max(loo)))
    print("    对照：第 713 号五档（含 v=0.50 坏档）的 gamma-1 留一最大误差为 0.0473。")
    OA0d = True

print()
print("    高速端实测 dev 与理论 c0 的比值（判读波包失真程度）：")
for v in VS_HI:
    print("        v=%.2f  dev(xi=42)=%.3e  理论 c0=%.3e  比值=%.2f"
          % (v, SMOKE[v]["devs"][3], c0_theory(v),
             SMOKE[v]["devs"][3] / max(c0_theory(v), 1e-300)))
print("耗时 %.1f 秒" % (time.time() - T0))

# ---------------- 图 ----------------
fig, ax = plt.subplots(1, 3, figsize=(15.8, 4.4))

vlo = sorted(C1_P713)
vhi = list(VS_HI)
allv = vlo + [v for v in vhi if v not in vlo]

ax[0].set_title("左：高速档 dev 随 v（四档 xi）")
for k, sg in enumerate(SIGS4):
    ys = [SMOKE[v]["devs"][k] for v in vhi]
    ax[0].plot(vhi, ys, "o-", ms=6, label=r"$\xi$=%g" % sg)
ax[0].set_yscale("log")
ax[0].set_xlabel("v")
ax[0].set_ylabel("dev")
ax[0].legend(fontsize=8)

ax[1].set_title("中：残差比与可用性门槛")
for k, sg in enumerate(SIGS4):
    ys = [SMOKE[v]["ress"][k] for v in vhi]
    ax[1].plot(vhi, ys, "s-", ms=6, label=r"$\xi$=%g 插值" % sg)
ax[1].plot(CMPV, [PROTO[v][1] for v in CMPV], "x", color="#824", ms=11,
           label="取整采样（旧协议）")
ax[1].axhline(0.30, color="#c44", ls="--", lw=1.4, label="门槛 0.30")
ax[1].set_xlabel("v")
ax[1].set_ylabel("残差比")
ax[1].legend(fontsize=7)

ax[2].set_title("右：$c_1(v)$ 两段与两个候选外推")
ax[2].plot(vlo, [C1_P713[v] for v in vlo], "o", color="#268", ms=8, label="第 713 号五档")
hi_vals = []
for v in USABLE:
    cf, _, _, _, _ = fit_poly(SIGS4, SMOKE[v]["devs"], [0, 1])
    hi_vals.append(cf[1])
if USABLE:
    ax[2].plot(USABLE, hi_vals, "D", color="#c94", ms=8, label="本轮可用档")
vx = np.linspace(0.20, 0.80, 200)
ax[2].plot(vx, K_GAMMA * np.array([gamma_of(x) - 1 for x in vx]),
           "-", color="#c44", lw=1.6, label=r"$K(\gamma-1)$")
ax[2].plot(vx, K_INVG * np.array([1 - 1 / gamma_of(x) for x in vx]),
           "--", color="#2a8", lw=1.6, label=r"$K(1-1/\gamma)$")
ax[2].set_xlabel("v")
ax[2].set_ylabel("$c_1$")
ax[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig(IMG, dpi=150)
print("图已存 %s" % IMG)
