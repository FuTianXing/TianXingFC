# -*- coding: utf-8 -*-
r"""
p706_fit_form.py

[观测者响应线 / 洛伦兹线] 第 706 号：用正确形式重拟合全部读数（判据 GA0 至 GA5）

背景：第 696 至 704 号一直用 dev = A/xi^2 + C 拟合，并追那个 C 追了八轮。
第 705 号查出真实标度是「常数加 xi^-1 项」：格点算符差 f(k) = 4sin^2(k/2)-k^2
的载波值给出常数项，波包展宽给出 xi^-1 项。用 xi^-2 去拟 xi^-1 的数据，
必然把一部分 xi^-1 项吸收进常数，于是 C 无论怎么测都不稳。

本轮用正确形式重拟合，并核对两件事：
  其一，新形式的残差是否显著小于旧形式；
  其二，新形式的常数项是否等于 |f(k_mod)|/(2 omega^2)。

判据：
  GA0 冒烟：三组合（总 A、井刚性 B、纯格点 D）的四档读数复现第 704 号
  GA1 形式对比（主判据）：F1 = A/xi^2+C 与 F2 = c0+c1/xi 的 R^2 与最大残差
  GA2 常数项的物理核对：F2 的 c0 是否等于 |f(k_mod)|/(2 omega^2)
  GA3 井刚性项的标度：B 是否纯 xi^-1（即 F2 的 c0 是否近零）
  GA4 三参数对照：F3 = c0+c1/xi+c2/xi^2 的 c2 是否显著
  GA5 判定

预登记判据（先于读数写定）：
  GA0 PASS：三组合四档读数与第 704 号记录值的相对差均 < 5%
  GA1 PASS：F2 的 R^2 > F1 的 R^2，且 F2 的最大相对残差 < F1 的一半
  GA2 PASS：F2 的 c0 与 |f(k_mod)|/(2 omega^2) 的比值落在 [0.7, 1.4]
  GA3 PASS：B 的 F2 拟合中 |c0| < 0.2 * c1/xi_max（常数项可忽略）
  GA4 报告量：c2 的标准误与显著性
  GA5 综合：GA1 与 GA2 皆过判「形式已定、常数项有物理身份」；
            GA1 不过判「新形式未优于旧形式」；
            GA1 过而 GA2 不过判「形式对了但常数项身份待查」

运行：python 验证脚本/p706_fit_form.py（确定性，约 2.5 分钟）
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

IMG = r"D:\software\TianXingFC\images\p706_fit_form.png"

T0 = time.time()
N = 2048
xs = np.arange(N) - N // 2
xf = xs.astype(float)
WGAP2 = 1.0
WB = 0.8008
V = 0.30
G = 1.0 / math.sqrt(1.0 - V * V)
TARGET = WB / G
KMOD = WB * G * V
SIGS = (14.0, 20.0, 28.0, 42.0)
DT = 0.02

KK = 2.0 * math.pi * np.fft.fftfreq(N)
K2 = KK ** 2

# 第 704 号的记录值，用于 GA0 复现核对
REC = {
    "A": {14.0: 1.7215e-3, 20.0: 1.2868e-3, 28.0: 9.8952e-4, 42.0: 7.4460e-4},
    "B": {14.0: 1.0068e-3, 20.0: 7.1676e-4, 28.0: 5.1400e-4, 42.0: 3.4036e-4},
    "D": {14.0: 6.8024e-4, 20.0: 5.5116e-4, 28.0: 4.6608e-4, 42.0: 3.9468e-4},
}

# 理论常数项
F_KMOD = 4.0 * math.sin(KMOD / 2.0) ** 2 - KMOD ** 2
C0_THEORY = abs(F_KMOD) / (2.0 * TARGET ** 2)

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


def make_ic(psi, d1, style):
    """style='g692' 配刚性井；'boost' 配收缩井（第 704 号更正后的符号）。"""
    if style == "boost":
        s0 = G * xf
        th0 = WB * G * V * xf
        c = G * V
        wfac = WB * G
        sgn = +1.0
    else:
        sh = 1.0 / G
        s0 = sh * xf
        th0 = -WB * sh * V * xf
        c = sh * V
        wfac = WB * sh
        sgn = -1.0
    F = np.interp(s0, xf, psi)
    Fp = np.interp(s0, xf, d1)
    return F * np.cos(th0), -c * Fp * np.cos(th0) + sgn * wfac * F * np.sin(th0)


def run(phi, vel, U0, wsig, evo, well_mode, dt=DT, T=1200.0, every=4):
    cs = []
    nst = int(T / dt)
    gain = 1.0 if well_mode == "rigid" else G * G
    for s in range(nst):
        t = (s + .5) * dt
        w1 = -U0 * np.exp(-gain * ((xf - V * t) ** 2) / (2.0 * wsig ** 2))
        if evo == "lattice":
            acc = (np.roll(phi, 1) + np.roll(phi, -1) - 2.0 * phi) - WGAP2 * phi - w1 * phi
        else:
            acc = -np.real(np.fft.ifft(K2 * np.fft.fft(phi))) - WGAP2 * phi - w1 * phi
        phi = phi + dt * vel + .5 * dt * dt * acc
        t2 = (s + 1) * dt
        w2 = -U0 * np.exp(-gain * ((xf - V * t2) ** 2) / (2.0 * wsig ** 2))
        if evo == "lattice":
            acc2 = (np.roll(phi, 1) + np.roll(phi, -1) - 2.0 * phi) - WGAP2 * phi - w2 * phi
        else:
            acc2 = -np.real(np.fft.ifft(K2 * np.fft.fft(phi))) - WGAP2 * phi - w2 * phi
        vel = vel + .5 * dt * (acc + acc2)
        if s % every == 0:
            cs.append(float(phi[N // 2 + int(round(V * t2))]))
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


def measure(sg, evo, well_mode, style):
    U0, _, psi, d1 = fit_U0(sg)
    phi, vel = make_ic(psi, d1, style)
    cs = run(phi, vel, U0, sg, evo, well_mode)
    wm, res = freq_refined(cs, 0.08, TARGET)
    return abs(wm - TARGET) / TARGET, res


def fit_form(sgs, ys, kind):
    """kind: 'inv2' 为 A/xi^2+C；'inv1' 为 c0+c1/xi；'inv1_inv2' 为三参数。"""
    xi = np.asarray(sgs, float)
    y = np.asarray(ys, float)
    if kind == "inv2":
        X = np.stack([1.0 / xi ** 2, np.ones_like(xi)], axis=1)
    elif kind == "inv1":
        X = np.stack([np.ones_like(xi), 1.0 / xi], axis=1)
    else:
        X = np.stack([np.ones_like(xi), 1.0 / xi, 1.0 / xi ** 2], axis=1)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    rss = float(np.sum((y - pred) ** 2))
    ss = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - rss / max(ss, 1e-300)
    maxrel = float(np.max(np.abs(pred / y - 1)))
    se = []
    dof = max(len(y) - X.shape[1], 1)
    try:
        cov = (rss / dof) * np.linalg.pinv(X.T @ X)
        se = list(np.sqrt(np.maximum(np.diag(cov), 0.0)))
    except Exception:
        se = [float("nan")] * X.shape[1]
    return coef, r2, maxrel, se


print("=" * 78)
print("第 706 号：用正确形式重拟合全部读数（判据 GA0 至 GA5）")
print("=" * 78)
print("装置：N=%d omega_b=%.4f v=%.2f gamma=%.6f k_mod=%.6f"
      % (N, WB, V, G, KMOD))
print("理论常数项 |f(k_mod)|/(2 omega^2) = %.6e" % C0_THEORY)

# ---------------- GA0 冒烟 ----------------
print()
print("[GA0] 冒烟：三组合的四档读数，复现第 704 号")
COMBOS = (("A", "lattice", "rigid", "g692"),
          ("B", "continuum", "rigid", "g692"),
          ("D", "lattice", "contracted", "boost"))
DATA = {}
worst_rec = 0.0
for name, evo, wm_, st in COMBOS:
    vals = []
    for sg in SIGS:
        dev, res = measure(sg, evo, wm_, st)
        vals.append(dev)
        rel = abs(dev / REC[name][sg] - 1)
        worst_rec = max(worst_rec, rel)
        print("    %-3s xi=%-6.0f dev=%.4e 残差比=%.3f  与第704号相对差=%+.1f%%"
              % (name, sg, dev, res, 100 * (dev / REC[name][sg] - 1)))
    DATA[name] = np.array(vals)
print("    最大相对差 = %.1f%%（判据 < 5%%）" % (100 * worst_rec))
GA0 = worst_rec < 0.05
print("    GA0 判定：%s" % ("PASS" if GA0 else "FAIL"))

# ---------------- GA1 形式对比 ----------------
print()
print("[GA1] 形式对比（主判据）")
print("    F1 = A/xi^2 + C       F2 = c0 + c1/xi       F3 = c0 + c1/xi + c2/xi^2")
FITS = {}
for name in ("A", "B", "D"):
    print("    [%s]" % name)
    for kind, lab in (("inv2", "F1"), ("inv1", "F2"), ("inv1_inv2", "F3")):
        coef, r2, maxrel, se = fit_form(SIGS, DATA[name], kind)
        FITS[(name, kind)] = (coef, r2, maxrel, se)
        print("      %s  R^2=%.6f  最大相对残差=%.2f%%  参数=%s"
              % (lab, r2, 100 * maxrel,
                 ", ".join("%.4e" % c for c in coef)))
r2_old = FITS[("A", "inv2")][1]
r2_new = FITS[("A", "inv1")][1]
mr_old = FITS[("A", "inv2")][2]
mr_new = FITS[("A", "inv1")][2]
print("    总 dev（A）：F1 的 R^2=%.6f、最大残差 %.2f%%；F2 的 R^2=%.6f、最大残差 %.2f%%"
      % (r2_old, 100 * mr_old, r2_new, 100 * mr_new))
print("    残差之比 旧/新 = %.1f 倍" % (mr_old / max(mr_new, 1e-300)))
GA1 = (r2_new > r2_old) and (mr_new < 0.5 * mr_old)
print("    GA1 判定：%s（R^2 更高且残差降到一半以下）" % ("PASS" if GA1 else "FAIL"))

# ---------------- GA2 常数项的物理核对 ----------------
print()
print("[GA2] 常数项的物理核对")
c0_A = FITS[("A", "inv1")][0][0]
c0_D = FITS[("D", "inv1")][0][0]
print("    F2 对总 dev 的 c0 = %.6e" % c0_A)
print("    F2 对纯格点 D 的 c0 = %.6e" % c0_D)
print("    理论值 |f(k_mod)|/(2 omega^2) = %.6e" % C0_THEORY)
rA = c0_A / C0_THEORY
rD = c0_D / C0_THEORY
print("    总 dev 的 c0 / 理论 = %.3f" % rA)
print("    纯格点 D 的 c0 / 理论 = %.3f" % rD)
GA2 = 0.7 <= rD <= 1.4
print("    GA2 判定：%s（纯格点部分的 c0 与理论比值落在 [0.7,1.4]）"
      % ("PASS" if GA2 else "FAIL"))

# ---------------- GA3 井刚性项的标度 ----------------
print()
print("[GA3] 井刚性项 B 是否为纯 xi^-1")
c0_B, c1_B = FITS[("B", "inv2")][0]
print("    F1 拟合 B：A=%.4e  C=%.4e" % (c0_B, c1_B))
c0_B2, c1_B2 = FITS[("B", "inv1")][0]
thr = 0.2 * c1_B2 / SIGS[-1]
print("    F2 拟合 B：c0=%.4e  c1=%.4e" % (c0_B2, c1_B2))
print("    判据：|c0| < 0.2*c1/xi_max = %.4e" % thr)
GA3 = abs(c0_B2) < thr
print("    GA3 判定：%s（c0 可忽略，B 为纯 xi^-1）" % ("PASS" if GA3 else "FAIL"))

# ---------------- GA4 三参数对照 ----------------
print()
print("[GA4] 三参数对照：c2 是否显著")
for name in ("A", "B", "D"):
    coef, r2, maxrel, se = FITS[(name, "inv1_inv2")]
    print("    %-3s c2=%.4e  se=%.4e  显著性=%.2f sigma  R^2=%.6f"
          % (name, coef[2], se[2], abs(coef[2]) / max(se[2], 1e-300), r2))
c2_A = FITS[("A", "inv1_inv2")][0][2]
c2_se = FITS[("A", "inv1_inv2")][3][2]
sig = abs(c2_A) / max(c2_se, 1e-300)
print("    总 dev 的 c2 显著性 = %.1f sigma（< 2 判不显著，两参数已足够）" % sig)

# ---------------- GA5 判定 ----------------
print()
print("[GA5] 判定")
print("    GA0=%s GA1=%s GA2=%s GA3=%s"
      % tuple("PASS" if x else "FAIL" for x in (GA0, GA1, GA2, GA3)))
if not GA0:
    verdict = "复现失败（与第 704 号记录差超 5%），本轮读数不可用"
elif GA1 and GA2:
    verdict = "拟合形式已定：dev = c0 + c1/xi，c0 有物理身份（%.3f 倍理论值）" % rD
elif GA1:
    verdict = "形式已定（%s 优于 %s），但 c0 与理论值差 %.2f 倍，身份待查" % (
        "F2", "F1", rD)
else:
    verdict = "新形式未优于旧形式（最大残差只降到 %.1f 倍）" % (mr_new / max(mr_old, 1e-300))
print("    判语：%s" % verdict)
print()
print("    新旧形式的参数对照（总 dev）：")
print("      旧 F1: A=%.4e  C=%.4e  (R^2=%.6f)" % (FITS[("A", "inv2")][0][0],
                                                  FITS[("A", "inv2")][0][1], r2_old))
print("      新 F2: c0=%.4e  c1=%.4e  (R^2=%.6f)" % (c0_A, FITS[("A", "inv1")][0][1], r2_new))
print()
print("    物理分解（新形式）：")
print("      c0 应等于 |f(k_mod)|/(2 omega^2) = %.4e，实测 %.4e（比值 %.3f）"
      % (C0_THEORY, c0_A, rA))
print("      c1 是 xi^-1 项，来自波包展宽（格点部分）与井刚性失配（连续部分）之和")
print("      三组合的 c1：B=%.4e  D=%.4e  和=%.4e  总 A=%.4e"
      % (c1_B2, FITS[("D", "inv1")][0][1],
         c1_B2 + FITS[("D", "inv1")][0][1], FITS[("A", "inv1")][0][1]))
print()
print("    这解释了追了八轮的那个 C：它是 F1 把 xi^-1 项硬塞进常数后的产物，")
print("    本身没有独立身份，其数值取决于拟合区间与读数口径。")
print("耗时 %.1f 秒" % (time.time() - T0))

# ---------------- 图 ----------------
fig, ax = plt.subplots(1, 3, figsize=(15.8, 4.4))
xi = np.array(SIGS)
inv1 = np.linspace(0.020, 0.075, 100)

ax[0].plot(inv1, FITS[("A", "inv1")][0][0] + FITS[("A", "inv1")][0][1] * inv1,
           "-", color="#268", lw=1.3, label="F2 = $c_0+c_1/\\xi$")
ax[0].plot(inv1, FITS[("A", "inv2")][0][0] * inv1 ** 2 + FITS[("A", "inv2")][0][1],
           "--", color="#c44", lw=1.3, label="F1 = $A/\\xi^2+C$")
ax[0].plot(1.0 / xi, DATA["A"], "o", color="k", ms=6, label="实测")
ax[0].set_xlabel(r"$1/\xi$")
ax[0].set_ylabel("dev")
ax[0].set_title("左：两种拟合形式（总 dev）")
ax[0].legend(fontsize=8)

ax[1].plot(1.0 / xi, DATA["A"] - (FITS[("A", "inv1")][0][0] + FITS[("A", "inv1")][0][1] / xi),
           "o-", color="#268", ms=6, label="F2 残差")
ax[1].plot(1.0 / xi, DATA["A"] - (FITS[("A", "inv2")][0][0] / xi ** 2 + FITS[("A", "inv2")][0][1]),
           "s--", color="#c44", ms=6, label="F1 残差")
ax[1].axhline(0, color="k", lw=0.7, ls=":")
ax[1].set_xlabel(r"$1/\xi$")
ax[1].set_ylabel("残差")
ax[1].set_title("中：残差对比（F2 的为 %.1f 倍小）" % (mr_old / max(mr_new, 1e-300)))
ax[1].legend(fontsize=8)

names = ["总 A", "井刚性 B", "格点 D"]
c0s = [FITS[(k, "inv1")][0][0] for k in ("A", "B", "D")]
ax[2].bar(range(3), c0s, color=["#268", "#c94", "#824"], width=0.6)
ax[2].axhline(C0_THEORY, color="#c44", ls="--", lw=1.3, label="理论 $|f(k_{\\rm mod})|/(2\\omega^2)$")
for i, v in enumerate(c0s):
    ax[2].text(i, v * 1.05 if v > 0 else 0.05 * C0_THEORY, "%.2e" % v,
               ha="center", fontsize=8)
ax[2].set_xticks(range(3))
ax[2].set_xticklabels(names, fontsize=8)
ax[2].set_ylabel(r"$c_0$")
ax[2].set_title("右：新形式的常数项与理论值")
ax[2].legend(fontsize=7)
plt.tight_layout()
plt.savefig(IMG, dpi=130)
print("图已存：", IMG)
