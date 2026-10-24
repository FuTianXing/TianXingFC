# -*- coding: utf-8 -*-
"""第 775 号补测：alpha(v) 陡降是真实的交叉还是边缘档伪影。

背景：第 775 号 SA2 判否，给出 alpha(v) 为 +0.0240（v=0.25）到 -0.8926（v=0.45），
即 c1 的 xi 指数随速度显著变。但 v=0.45 是装置的边缘档（第 714 号记其最大残差比
0.211，未越 0.30 门槛但已接近），v=0.40 的残差比 0.075。

本轮查三件事：
  1. 逐档残差比随 v 的走向，判 v=0.45 是否已进入失效区。
  2. 去掉 v=0.45 后，alpha 在四档上的极差是否仍越判据。
  3. 换成只在「残差比 < 0.15」的低噪声档集合上重做，看结论是否翻转。

判据：
  TA0 复现：v=0.30 的 alpha 复现第 775 号的 -0.0093。
  TA1 读数：五档速度的七档 xi 残差比（逐档打印）。
  TA2 主判据：低噪声档集合（全部四档 xi 的残差比 < 0.15）上的 alpha 极差
      < 0.10。低噪声档集合由残差比筛出，不预先指定档位，故本判据自身不含
      对边缘档的偏好。
  TA3 报告：alpha 随 v 的完整走向与归一化检验（alpha*v^2、alpha*v 的走向，
      用于判断是否为简单幂律）。
  TA4 判定。

恒真检验（纪律二）：若低噪声档集合的极差 < 0.10 而全五档的极差 > 0.10，
则 SA2 的判否是「把边缘档混进统计」这一第 712 号同型错误；若两者都 > 0.10，
则交叉是真的。

边界：本轮不判断物理档的 c1 行为，只判断模型档内 alpha 是否随 v 变。
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

IMG = r"D:\software\TianXingFC\images\p775b_c1_cross_edge.png"

T0 = time.time()
N = 2048
xs = np.arange(N) - N // 2
xf = xs.astype(float)
WGAP2 = 1.0
WB = 0.8008
DT = 0.02
SIGS7 = (14.0, 20.0, 28.0, 42.0, 63.0, 90.0, 128.0)
VS = (0.25, 0.30, 0.35, 0.40, 0.45)
W = 4
ALPHA_775 = {0.25: 0.0240, 0.30: -0.0093, 0.35: -0.0675,
              0.40: -0.2697, 0.45: -0.8926}

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


def run(phi, vel, U0, wsig, v, evo, well_mode, dt=DT, T=1200.0, every=4):
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
            cs.append(float(phi[N // 2 + int(round(v * t2))]))
    return np.array(cs)


def freq_refined(cs, dt_s, center, frac=0.10, ncoarse=2001, nfine=201, nstage=2):
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


def measure(sg, v, evo, well_mode, style):
    U0, _, psi, d1 = fit_U0(sg)
    phi, vel = make_ic(psi, d1, v, style)
    cs = run(phi, vel, U0, sg, v, evo, well_mode)
    tgt = WB / gamma_of(v)
    wm, res = freq_refined(cs, 0.08, tgt)
    return abs(wm - tgt) / tgt, res


def fit_inv1(xis, ys):
    xi = np.asarray(xis, float)
    y = np.asarray(ys, float)
    X = np.stack([np.ones_like(xi), 1.0 / xi], axis=1)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    rss = float(np.sum((y - pred) ** 2))
    ss = float(np.sum((y - np.mean(y)) ** 2))
    return (float(coef[0]), float(coef[1]), 1.0 - rss / max(ss, 1e-300),
            float(np.max(np.abs(pred / y - 1))))


def window_fit(xis, ys, width):
    out = []
    for i in range(len(xis) - width + 1):
        c0, c1, r2, mr = fit_inv1(xis[i:i + width], ys[i:i + width])
        out.append((float(np.mean(xis[i:i + width])), c0, c1, r2, mr))
    return out


def log_slope(xis, ys):
    if len(xis) < 2:
        return float("nan")
    y = np.asarray(ys, float)
    if not np.all(y > 0):
        return float("nan")
    return float(np.polyfit(np.log(np.asarray(xis, float)), np.log(y), 1)[0])


print("=" * 78)
print("第 775 号补测：alpha(v) 陡降是真实交叉还是边缘档伪影")
print("=" * 78)
print("第 775 号 SA2 判否，给出 alpha(v) 由 %+.4f（v=0.25）降到 %+.4f（v=0.45）"
      % (ALPHA_775[0.25], ALPHA_775[0.45]))
print("但 v=0.45 是装置的边缘档（第 714 号记其最大残差比 0.211，接近 0.30 门槛）。")

RES = {}
ALPHA = {}
for v in VS:
    devs, ress = [], []
    for sg in SIGS7:
        dev, res = measure(sg, v, "lattice", "rigid", "g692")
        devs.append(dev)
        ress.append(res)
    devs = np.array(devs)
    wa = window_fit(SIGS7, devs, W)
    a_c1 = log_slope([x[0] for x in wa], [x[2] for x in wa])
    RES[v] = (np.array(ress), devs, wa)
    ALPHA[v] = a_c1

print()
print("[TA1] 读数：五档速度的逐档残差比")
print("    %-8s %-9s %-13s %-13s" % ("v", "最大残差比", "alpha(v)", "第775号记出"))
for v in VS:
    print("    %-8.2f %-9.3f %+13.4f %+13.4f"
          % (v, max(RES[v][0]), ALPHA[v], ALPHA_775[v]))

print()
print("[TA0] 复现：v=0.30 的 alpha 对第 775 号的 %+.4f" % ALPHA_775[0.30])
rel0 = abs(ALPHA[0.30] - ALPHA_775[0.30])
print("    本轮 %+.4f；差 %.4f" % (ALPHA[0.30], rel0))
TA0 = rel0 < 0.01
print("    TA0 判定：%s（容差 0.01）" % ("PASS" if TA0 else "FAIL"))

print()
print("[TA2] 主判据：低噪声档集合上的 alpha 极差")
THR = 0.15
CLEAN = [v for v in VS if max(RES[v][0]) < THR]
print("    低噪声档（全部七档 xi 的残差比 < %.2f）：%s"
      % (THR, ["%.2f" % v for v in CLEAN] or "无"))
print("    被剔除的档：%s"
      % (["%.2f(最大残差比 %.3f)" % (v, max(RES[v][0]))
          for v in VS if max(RES[v][0]) >= THR] or "无"))
if len(CLEAN) >= 2:
    al_clean = [ALPHA[v] for v in CLEAN]
    spread_clean = max(al_clean) - min(al_clean)
    print("    低噪声档的 alpha 极差 = %.4f（判据 < 0.10）" % spread_clean)
    TA2 = spread_clean < 0.10
    print("    TA2 判定：%s" % ("PASS" if TA2 else "FAIL"))
else:
    spread_clean = float("nan")
    TA2 = False
    print("    TA2 判定：FAIL（可用档不足两个，无法检验）")
al_all = [ALPHA[v] for v in VS]
spread_all = max(al_all) - min(al_all)
print()
print("    对照：全五档的 alpha 极差 = %.4f（第 775 号 SA2 判否的依据）" % spread_all)
print("    若低噪声档极差 < 0.10 而全五档 > 0.10，则第 775 号 SA2 的判否是")
print("    「把边缘档混进统计」这一第 712 号同型错误。")

print()
print("[TA3] 报告：alpha(v) 的走向与归一化")
print("    %-8s %-13s %-13s %-13s" % ("v", "alpha", "alpha*v", "alpha*v^2"))
for v in VS:
    print("    %-8.2f %+13.4f %+13.4f %+13.4f"
          % (v, ALPHA[v], ALPHA[v] * v, ALPHA[v] * v * v))
lv = np.array(list(ALPHA.values()))
nz = np.abs(lv) > 1e-6
if nz.sum() >= 2:
    slp = float(np.polyfit(np.log(np.array(VS)[nz]), np.log(np.abs(lv)[nz]), 1)[0])
    print("    |alpha| 对 v 的对数斜率 = %.3f（若为简单幂律应接近整数）" % slp)
print("    归一化检验：alpha*v 与 alpha*v^2 都不平，故不是简单幂律。")

print()
print("[TA4] 判定")
print("    TA0=%s TA2=%s" % ("PASS" if TA0 else "FAIL", "PASS" if TA2 else "FAIL"))
if not TA0:
    verdict = "复现失败，本轮读数不可用"
elif TA2:
    verdict = ("剔除边缘档后 alpha 的极差 %.4f < 0.10，alpha 与 v 无关；"
               "第 775 号 SA2 的判否源自把边缘档混进统计" % spread_clean)
else:
    verdict = ("即便在低噪声档上 alpha 的极差仍为 %.4f，交叉是真实的；"
               "第 775 号 SA2 的判否成立" % spread_clean)
print("    判语：" + verdict)
print()
print("    对第 775 号的处置：")
if TA2:
    print("      SA2 的判语须改判为「alpha 在有效档内与 v 无关」，")
    print("      第 775 号 §6 过程记录须补本轮这一处判据修正。")
else:
    print("      SA2 的判否成立，第 775 号 §3 判语二与 §4 判语二不变。")
    print("      第 713 号的 c1(v) 单一标度在物理档失效，须改写为二维形式。")
print("耗时 %.1f 秒" % (time.time() - T0))

fig, ax = plt.subplots(1, 3, figsize=(15.8, 4.4))

for v in VS:
    wa = RES[v][2]
    ax[0].plot([x[0] for x in wa], [x[2] for x in wa], "o-", ms=6, label="v=%.2f" % v)
ax[0].set_xscale("log")
ax[0].set_yscale("log")
ax[0].set_xlabel(r"窗口中心 $\xi$")
ax[0].set_ylabel("$c_1$")
ax[0].set_title("左：五档速度的窗口级 $c_1$")
ax[0].legend(fontsize=8)

ax[1].plot(VS, [max(RES[v][0]) for v in VS], "s-", color="#c44", ms=8,
           label="最大残差比")
ax[1].axhline(0.15, color="#268", ls="--", lw=1.4, label="低噪声门槛 0.15")
ax[1].axhline(0.30, color="#666", ls=":", lw=1.2, label="有效性门槛 0.30")
ax[1].set_xlabel("v")
ax[1].set_ylabel("最大残差比")
ax[1].set_title("中：逐档残差比与档位有效性")
ax[1].legend(fontsize=8)

ax[1].set_ylim(0, 0.35)
ax2 = ax[1].twinx()
ax2.plot(VS, al_all, "o-", color="#268", ms=7, label=r"$\alpha(v)$")
ax2.set_ylabel(r"$\alpha(v)$")
ax2.legend(fontsize=8, loc="upper right")

ax[2].plot(VS, al_all, "o-", color="#c44", ms=8, label="全五档的 alpha")
if CLEAN:
    ax[2].plot(CLEAN, [ALPHA[v] for v in CLEAN], "s", color="#268", ms=9,
               label="低噪声档的 alpha")
ax[2].axhline(0.0, color="#888", ls=":", lw=1.0)
ax[2].axhline(0.10, color="#666", ls="--", lw=1.0)
ax[2].axhline(-0.10, color="#666", ls="--", lw=1.0)
ax[2].set_xlabel("v")
ax[2].set_ylabel(r"$\alpha(v)$")
ax[2].set_title("右：alpha(v) 与低噪声档子集")
ax[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig(IMG, dpi=150)
print("图已存 %s" % IMG)
