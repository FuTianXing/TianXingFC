# -*- coding: utf-8 -*-
r"""
minimal_verification_p711_r_origin.py

[观测者响应线 / 洛伦兹线] 第 711 号：修正因子 r 的来源与 v=0 反例（判据 KA0 至 KA3）

背景：第 710 号证明 omega_lat^2 - omega_cont^2 = r <f>，r 在 0.87 至 0.90 之间，
并证明它与 c0 的实测比值自洽。但 r 为什么不是 1，没有解释。

假说：Rayleigh 商对本征态精确成立，此时恒等式两边相等（r=1）。运动束缚态是叠加态，
Rayleigh 商给能量平均，而 <f> 是对平面波分量的加权，两者权重不同，故 r != 1。

判据的关键在 v=0：此时初值退化为静止束缚态，它是格点哈密顿量的本征态。
若假说成立，r(v=0) 必须等于 1。
若 v=0 也不等于 1，则 Rayleigh 商在「格点本征态 + 连续演化」这个组合下本就不成立，假说作废。

判据：
  KA0 冒烟：v=0.30 四档 r 复现第 709/710 号
  KA1 v=0 反例（主判据）：r(v=0) 是否等于 1
  KA2 r 随 v 的走势（报告）：四档 xi 乘三档 v
  KA3 判定

预登记判据（先于读数写定）：
  KA0 PASS：四档 r 与第 710 号相对差均 < 2%
  KA1 PASS：|r(v=0) - 1| < 0.02（四档均值）
  KA2 报告量：r(v) 在 v=0、0.30、0.50 三档的取值
  KA3 综合：KA1 过判「r 的偏离确由叠加态引起，Rayleigh 商对单态成立」；
            KA1 不过判「Rayleigh 商在本装置的单态上也不成立，须另找 r 的来源」

运行：python 验证脚本/minimal_verification_p711_r_origin.py（确定性，约 5 分钟）
"""
import numpy as np
import math
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

IMG = r"D:\software\TianXingFC\images\p711_r_origin.png"

T0 = time.time()
N = 2048
xs = np.arange(N) - N // 2
xf = xs.astype(float)
WGAP2 = 1.0
WB = 0.8008
SIGS = (14.0, 20.0, 28.0, 42.0)
VS = (0.0, 0.30, 0.50)
DT = 0.02

KK = 2.0 * math.pi * np.fft.fftfreq(N)
K2 = KK ** 2
FK = 4.0 * np.sin(KK / 2.0) ** 2 - KK ** 2

REC_R = {14.0: 0.8695, 20.0: 0.8747, 28.0: 0.8880, 42.0: 0.8895}

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


def make_ic(psi, d1, v):
    g = gamma_of(v)
    s0 = g * xf
    th0 = WB * g * v * xf
    c = g * v
    wfac = WB * g
    F = np.interp(s0, xf, psi)
    Fp = np.interp(s0, xf, d1)
    return F * np.cos(th0), -c * Fp * np.cos(th0) + wfac * F * np.sin(th0)


def run(phi, vel, U0, wsig, v, evo, dt=DT, T=1200.0, spec_every=200):
    g = gamma_of(v)
    every = max(int(round(0.08 / dt)), 1)
    cs, specs = [], []
    nst = int(T / dt)
    gain = g * g
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
        if s % spec_every == 0 and s > nst // 4:
            p2 = np.abs(np.fft.fft(phi)) ** 2
            tot = float(p2.sum())
            if tot > 0:
                specs.append(p2 / tot)
    return np.array(cs), specs


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


def measure(sg, v):
    g = gamma_of(v)
    tgt = WB / g
    U0, _, psi, d1 = fit_U0(sg)
    out = {}
    for evo in ("lattice", "continuum"):
        phi, vel = make_ic(psi, d1, v)
        cs, specs = run(phi, vel, U0, sg, v, evo)
        wm, res = freq_refined(cs, 0.08, tgt)
        spec = np.mean(np.array(specs), axis=0)
        out[evo] = dict(w=wm, res=res, fbar=float(np.sum(spec * FK)))
    wl, wc = out["lattice"]["w"], out["continuum"]["w"]
    fb = out["lattice"]["fbar"]
    out["r"] = (wl * wl - wc * wc) / fb if abs(fb) > 1e-30 else float("nan")
    out["dev"] = abs(wl - tgt) / tgt
    return out


print("=" * 78)
print("第 711 号：修正因子 r 的来源与 v=0 反例（判据 KA0 至 KA3）")
print("=" * 78)
print("装置：N=%d omega_b=%.4f xi in %s  v in %s" % (N, WB, list(SIGS), list(VS)))
print("假说：Rayleigh 商对本征态精确成立，故 r(v=0) 应为 1。")

# ---------------- KA0 冒烟 ----------------
print()
print("[KA0] 冒烟：v=0.30 四档 r 复现第 709/710 号")
R030 = []
worst = 0.0
for sg in SIGS:
    m = measure(sg, 0.30)
    R030.append(m["r"])
    rel = abs(m["r"] / REC_R[sg] - 1)
    worst = max(worst, rel)
    print("    xi=%-6.0f r=%.4f  w_lat=%.6f  w_cont=%.6f  <f>=%.4e"
          % (sg, m["r"], m["lattice"]["w"], m["continuum"]["w"],
             m["lattice"]["fbar"]))
print("    最大相对差 = %.1f%%（判据 < 2%%）" % (100 * worst))
KA0 = worst < 0.02
print("    KA0 判定：%s" % ("PASS" if KA0 else "FAIL"))

# ---------------- KA1 v=0 反例 ----------------
print()
print("[KA1] v=0 反例（主判据）：r 是否等于 1")
print("    在 v=0，初值退化为静止束缚态，它是格点哈密顿量的本征态。")
print("    %-6s %-13s %-13s %-13s %-11s %-9s"
      % ("xi", "w_lat", "w_cont", "<f>", "r", "|r-1|"))
R0 = []
for sg in SIGS:
    m = measure(sg, 0.0)
    R0.append(m["r"])
    print("    %-6.0f %-13.6f %-13.6f %-13.4e %-11.4f %-9.4f"
          % (sg, m["lattice"]["w"], m["continuum"]["w"],
             m["lattice"]["fbar"], m["r"], abs(m["r"] - 1)))
r0_mean = float(np.mean(R0))
print("    r(v=0) 四档均值 = %.4f，偏离 1 为 %.4f" % (r0_mean, abs(r0_mean - 1)))
KA1 = abs(r0_mean - 1) < 0.02
print("    KA1 判定：%s（偏离 < 0.02 判 Rayleigh 商对单态成立）"
      % ("PASS" if KA1 else "FAIL"))

# ---------------- KA2 r 随 v 的走势 ----------------
print()
print("[KA2] r 随 v 的走势（报告）")
print("    %-6s %-13s %-13s %-13s" % ("v", "r 四档均值", "r 四档极差", "dev 均值"))
RV = {0.0: R0}
for v in (0.30, 0.50):
    if v == 0.30:
        rs = R030
        devs = [measure(sg, 0.30)["dev"] for sg in SIGS]
    else:
        rs, devs = [], []
        for sg in SIGS:
            m = measure(sg, v)
            rs.append(m["r"])
            devs.append(m["dev"])
    RV[v] = rs
    print("    %-6.2f %-13.4f %-13.4f %-13.4e"
          % (v, np.mean(rs), max(rs) - min(rs), np.mean(devs)))

# ---------------- KA3 判定 ----------------
print()
print("[KA3] 判定")
print("    KA0=%s KA1=%s" % ("PASS" if KA0 else "FAIL", "PASS" if KA1 else "FAIL"))
if not KA0:
    verdict = "复现失败，本轮读数不可用"
elif KA1:
    verdict = ("r 的偏离确由叠加态引起：单态（v=0）给 r=%.4f，运动态给 %.4f 至 %.4f；"
               "Rayleigh 商在单态上成立，r 度量的是叠加态与原子的权重差"
               % (r0_mean, np.mean(RV[0.30]), np.mean(RV[0.50])))
else:
    verdict = ("Rayleigh 商在本装置的单态上也不成立（r(v=0)=%.4f），"
               "r 的来源须另找，不能只归给叠加态" % r0_mean)
print("    判语：%s" % verdict)
print()
print("    r 的实测全貌：")
for v in VS:
    print("      v=%.2f  r 均值 %.4f（四档 %s）"
          % (v, np.mean(RV[v]), ", ".join("%.4f" % x for x in RV[v])))
print("    若 r 随 v 单调变化且 v=0 处为 1，则叠加态假说获得支持；")
print("    若 r(v=0) 已明显偏离 1，则该假说被否定。")
print("耗时 %.1f 秒" % (time.time() - T0))

# ---------------- 图 ----------------
fig, ax = plt.subplots(1, 3, figsize=(15.8, 4.4))
xi = np.array(SIGS)

ax[0].plot(xi, R0, "o-", color="#268", ms=7, label="v=0（单态）")
ax[0].plot(xi, R030, "s-", color="#c94", ms=7, label="v=0.30")
ax[0].plot(xi, RV[0.50], "^-", color="#824", ms=7, label="v=0.50")
ax[0].axhline(1.0, color="#c44", ls="--", lw=1.3, label="假说预期 r=1")
ax[0].set_xscale("log")
ax[0].set_xlabel(r"$\xi$")
ax[0].set_ylabel("r")
ax[0].set_title("左：三档速度下的 r（$v$=0 是否给 1）")
ax[0].legend(fontsize=8)

ax[1].plot(VS, [np.mean(RV[v]) for v in VS], "o-", color="#268", ms=8)
ax[1].axhline(1.0, color="#c44", ls="--", lw=1.2)
ax[1].set_xlabel("v")
ax[1].set_ylabel("r（四档均值）")
ax[1].set_title("中：r 随速度的走势")

for idx, (sg, c) in enumerate(zip(SIGS, plt.cm.viridis(np.linspace(0.2, 0.85, len(SIGS))))):
    ys = [RV[v][idx] for v in VS]
    ax[2].plot(VS, ys, "o-", color=c, ms=6, label=r"$\xi$=%.0f" % sg)
ax[2].axhline(1.0, color="#c44", ls="--", lw=1.0)
ax[2].set_xlabel("v")
ax[2].set_ylabel("r")
ax[2].set_title("右：四档宽度的 $r$ 随速度")
ax[2].legend(fontsize=7)
plt.tight_layout()
plt.savefig(IMG, dpi=130)
print("图已存：", IMG)
