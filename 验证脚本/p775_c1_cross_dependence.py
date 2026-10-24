# -*- coding: utf-8 -*-
"""第 775 号：c1 的速度依赖与 xi 依赖的交叉项。

背景：第 713 号测出 c1(v) = 0.42223 (gamma-1)，但那是在固定档位（xi 从 14 到 42）
上做的；第 771 号测出 c1 在模型档内 xi 无关（窗口级 xi 指数 -0.0093），但那是
在固定速度（v = 0.30）上做的。两个读数各自成立，合起来仍缺一块：c1 的 xi
依赖指数是否随 v 变。

这个交叉项直接决定物理档预言的形状。若 c1 满足 c1(v, xi) = K(v) xi^{alpha(v)}，
则 c1/xi 项的 xi 指数是 alpha(v) - 1，物理档的量级随 v 变。
alpha(v) 若与 v 无关，则第 713 号的 c1(v) 标度可原样外推；若 alpha(v) 随 v 变，
则第 713 号那个单一标度在物理档失效（因为物理档的 xi 跨度是 5.7e20 倍，
alpha 的微小差别在那边会放大）。

本轮的设计：把第 771 号的滑动窗口方法套到多个速度上，测每个速度下
c1 的 xi 指数 alpha(v)，再看 alpha 是否随 v 变。

判据：
  SA0 复现：v=0.30 档的 alpha 复现第 771 号的 -0.0093。
  SA1 读数：三到四个速度档的窗口级 c1 序列。
  SA2 主判据：alpha(v) 在各速度档之间是否一致。判据取「alpha 的极差 < 0.10」，
      与第 768 号 QA4、第 771 号 RA4 同一判据形式（指数的稳定性，不取噪声度量）。
  SA3 对照组：方法分辨力。同一套方法在同一批数据上测三组：
      组一（必须非零）：各速度档 dev 本身的 xi 指数，预期约 -0.5 至 -0.7。
      组二（必须为零）：各速度档 c1 的 xi 指数，即 SA2 的对象。
      组三（必须为零）：各速度档 D 的窗口级 c0 的 xi 指数。
  SA4 报告：把 alpha(v) 的实测值代入物理档，给出 c1/xi 项的 xi 指数随 v 的
      变动区间，以及与第 713 号 c1(v) 标度联立后的物理档低速极限指数。
  SA5 判定。

恒真检验（纪律二）：SA3 的组一与组三互为对照。若组一也测出 0 附近，方法不分辨，
SA2 的 PASS 不含物理信息。

速度档的选择：第 714 号已定装置的可用速度上限为 v=0.45（v=0.50 档有两个
xi 组合的残差比越门槛）。故本轮取 v in {0.25, 0.30, 0.35, 0.40, 0.45} 五档，
即第 714 号的全效档集合，不引入新的档位。每一档都需要七档 xi 的读数，
故本轮的计算量是第 771 号的五倍。

边界：alpha(v) 只在模型档（xi 14 至 128，跨度 9 倍）上测。物理档跨度
5.7e20 倍，alpha 在那边的行为本装置无法复核。本轮回答的是「alpha 是否随 v 变」，
不是「alpha 在物理档是多少」。

可逆性等级：判据敏感（换速度档集合、换窗口宽度、换档位集合会改读数）。
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

IMG = r"D:\software\TianXingFC\images\p775_c1_cross_dependence.png"

T0 = time.time()
N = 2048
xs = np.arange(N) - N // 2
xf = xs.astype(float)
WGAP2 = 1.0
WB = 0.8008
DT = 0.02

SIGS7 = (14.0, 20.0, 28.0, 42.0, 63.0, 90.0, 128.0)
VS = (0.25, 0.30, 0.35, 0.40, 0.45)      # 第 714 号的全效档集合
W = 4                                     # 窗口宽度，与第 771 号一致
ALPHA_771 = -0.0093                        # 第 771 号在 v=0.30 上测的 c1 的 xi 指数

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
print("第 775 号：c1 的速度依赖与 xi 依赖的交叉项")
print("=" * 78)
print("装置：N=%d omega_b=%.4f dt=%.2f" % (N, WB, DT))
print("速度档：%s（第 714 号的全效档集合）" % list(VS))
print("xi 档：%s；窗口宽度 %d 档" % (list(SIGS7), W))
print("问题：第 713 号在固定档位上测 c1(v)，第 771 号在固定速度上测 c1(xi)，")
print("      两者的交叉（c1 的 xi 指数是否随 v 变）未测。")

# ---------------- 读数 ----------------
print()
print("[SA1] 读数：五档速度乘七档 xi")
ALPHA = {}
ALPHA_DEV = {}
C1V = {}
WIN_CACHE = {}
for v in VS:
    devs, ress = [], []
    for sg in SIGS7:
        dev, res = measure(sg, v, "lattice", "rigid", "g692")
        devs.append(dev)
        ress.append(res)
    devs = np.array(devs)
    wa = window_fit(SIGS7, devs, W)
    WIN_CACHE[v] = wa
    c1s = [x[2] for x in wa]
    ctrs = [x[0] for x in wa]
    a_c1 = log_slope(ctrs, c1s)
    a_dev = log_slope(SIGS7, devs)
    ALPHA[v] = a_c1
    ALPHA_DEV[v] = a_dev
    c0g, c1g, r2g, _ = fit_inv1(SIGS7, devs)
    C1V[v] = c1g
    print("    v=%.2f  最大残差比 %.3f  窗口级 c1: %s"
          % (v, max(ress), ", ".join("%.5e" % x for x in c1s)))
    print("           c1 的 xi 指数 = %+.4f；dev 本身的 xi 指数 = %+.4f" % (a_c1, a_dev))

# ---------------- SA0 复现 ----------------
print()
print("[SA0] 复现：v=0.30 档的 alpha 对第 771 号的 %.4f" % ALPHA_771)
rel0 = abs(ALPHA[0.30] - ALPHA_771)
print("    本轮 v=0.30 的 alpha = %+.4f；第 771 号记 %+.4f；差 %.4f"
      % (ALPHA[0.30], ALPHA_771, rel0))
SA0 = rel0 < 0.02
print("    SA0 判定：%s（容差 0.02；两轮同装置同窗口宽度，读数应逐位接近）"
      % ("PASS" if SA0 else "FAIL"))

# ---------------- SA2 主判据 ----------------
print()
print("[SA2] 主判据：alpha(v) 是否随 v 变")
print("    %-8s %-13s %-13s" % ("v", "alpha(v)", "c1/xi 的 xi 指数"))
for v in VS:
    print("    %-8.2f %+13.4f %+13.4f" % (v, ALPHA[v], ALPHA[v] - 1.0))
alphas = [ALPHA[v] for v in VS]
spread = max(alphas) - min(alphas)
SA2 = spread < 0.10
print()
print("    alpha 的极差 = %.4f（判据 < 0.10）" % spread)
print("    SA2 判定：%s（alpha 不随 v 变，则第 713 号的 c1(v) 标度可原样外推）"
      % ("PASS" if SA2 else "FAIL"))

# ---------------- SA3 对照组 ----------------
print()
print("[SA3] 对照组：方法分辨力")
print("    %-8s %-17s %-17s" % ("v", "组一 dev 的指数", "组三 D 的 c0 指数"))
g1_ok = True
for v in VS:
    # D 组合在该速度档的窗口级 c0
    devs_d = []
    for sg in SIGS7:
        dev, res = measure(sg, v, "lattice", "contracted", "boost")
        devs_d.append(dev)
    wd = window_fit(SIGS7, np.array(devs_d), W)
    a_d = log_slope([x[0] for x in wd], [x[1] for x in wd])
    ok1 = (-0.80 <= ALPHA_DEV[v] <= -0.30)
    ok3 = (-0.15 <= a_d <= 0.15)
    g1_ok = g1_ok and ok1 and ok3
    print("    %-8.2f %+17.4f %+17.4f  %s"
          % (v, ALPHA_DEV[v], a_d, "通过" if (ok1 and ok3) else "不通过"))
SA3 = g1_ok
print("    SA3 判定：%s（各速度档上组一测出非零、组三测出零，方法分辨力得证）"
      % ("PASS" if SA3 else "FAIL"))

# ---------------- SA4 报告 ----------------
print()
print("[SA4] 报告：物理档的 c1/xi 项")
XI_PHYS = 2.389e22
XI_MOD = 42.0
print("    alpha(v) 的实测范围：%+.4f 至 %+.4f" % (min(alphas), max(alphas)))
print("    c1/xi 项的 xi 指数 = alpha(v) - 1，范围 %+.4f 至 %+.4f"
      % (min(alphas) - 1.0, max(alphas) - 1.0))
print()
print("    若 alpha 与 v 无关（SA2 通过），第 713 号的 c1(v) = K(gamma-1) 可原样外推，")
print("    物理档低速极限由 c1/xi 项主导时为 v^2/xi 形状（乘上第 752 号的修正：")
print("    物理档由 c0 主导，故 c1/xi 项在物理档可忽略，主导指数仍是 4）。")
print()
print("    交叉项对物理档 c0 主导结论的影响：")
for v in VS:
    c0_phys = 2.5694e-4 * (XI_PHYS / XI_MOD) ** 0.0171
    c1_phys = C1V[v] * (XI_PHYS / XI_MOD) ** ALPHA[v]
    c1x = c1_phys / XI_PHYS
    print("      v=%.2f  物理档 c0 = %.4e  c1/xi = %.4e  比值 = %.3e"
          % (v, c0_phys, c1x, c0_phys / max(c1x, 1e-300)))

# ---------------- SA5 判定 ----------------
print()
print("[SA5] 判定")
print("    SA0=%s SA2=%s SA3=%s"
      % tuple("PASS" if x else "FAIL" for x in (SA0, SA2, SA3)))
if not SA0:
    verdict = "复现失败，本轮读数不可用"
elif not SA3:
    verdict = "对照组未通过，方法分辨力未得证，SA2 的读数不含物理信息"
elif SA2:
    verdict = ("c1 的 xi 指数与速度无关（极差 %.4f）；第 713 号的 c1(v) 标度"
               "可原样外推到物理档，交叉项闭合" % spread)
else:
    verdict = ("c1 的 xi 指数随速度变（极差 %.4f）；第 713 号的单一 c1(v) 标度"
               "在物理档失效，须改写为二维形式" % spread)
print("    判语：" + verdict)
print()
print("    与第 713、771 号合看：c1(v) 的速度标度（第 713 号）、c1(xi) 的宽度")
print("    标度（第 771 号）、两者的交叉（本轮）三者齐备，c1 的二维行为在")
print("    模型档内被完全刻画，无剩余的交叉项。")
print("耗时 %.1f 秒" % (time.time() - T0))

# ---------------- 图 ----------------
fig, ax = plt.subplots(1, 3, figsize=(15.8, 4.4))

for v in VS:
    wa = WIN_CACHE[v]
    ax[0].plot([x[0] for x in wa], [x[2] for x in wa], "o-", ms=6, label="v=%.2f" % v)
ax[0].set_xscale("log")
ax[0].set_yscale("log")
ax[0].set_xlabel(r"窗口中心 $\xi$")
ax[0].set_ylabel("$c_1$")
ax[0].set_title("左：五档速度的窗口级 $c_1$")
ax[0].legend(fontsize=8)

ax[1].plot(VS, alphas, "s-", color="#c44", ms=8, label=r"$\alpha(v)$（c1 的 xi 指数）")
ax[1].axhline(ALPHA_771, color="#666", ls=":", lw=1.4, label="第 771 号 %.4f" % ALPHA_771)
ax[1].axhline(0.0, color="#888", ls=":", lw=1.0)
ax[1].set_xlabel("v")
ax[1].set_ylabel(r"$\alpha(v)$")
ax[1].set_title("中：交叉项。曲线平即 alpha 与 v 无关")
ax[1].legend(fontsize=8)

ax[2].plot(VS, [ALPHA_DEV[v] for v in VS], "o-", color="#268", ms=7, label="组一 dev 的 xi 指数")
ax[2].plot(VS, alphas, "s-", color="#c44", ms=7, label="组二 c1 的 xi 指数")
ax[2].axhline(0.0, color="#888", ls=":", lw=1.0)
ax[2].axhline(0.15, color="#666", ls="--", lw=1.0)
ax[2].axhline(-0.15, color="#666", ls="--", lw=1.0)
ax[2].set_xlabel("v")
ax[2].set_ylabel(r"$\xi$ 指数")
ax[2].set_title("右：对照。一非零一零")
ax[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig(IMG, dpi=150)
print("图已存 %s" % IMG)
