# -*- coding: utf-8 -*-
r"""
p708_dt_convergence.py

[观测者响应线 / 洛伦兹线] 第 708 号：c0 的 dt 收敛与 0.88 缺口的归因（判据 HA0 至 HA4）

背景：第 706/707 号把形式定为 dev = c0 + c1/xi，c0 的实测与第 705 号理论值
|f(k_mod)|/(2 omega^2) 之比在五档速度下为 0.80 至 0.91（均值 0.88）。
这是测量部分最后一个纯公式层面的缺口，也是物理档外推唯一要用的系数。

本轮检验一个具体候选：Verlet 时间离散化。运动束缚态的 dev 为正，因为格点色散
让频率偏低（<f> < 0）。而 Verlet 会抬高频率，第 703 号已精确验证其相对量为
(2/dt)arcsin(omega dt/2)/omega - 1。抬高频率的效果是让实测 dev 变小，
故 c0 的实测值应随 dt 减小而上升。

预测（dt=0.02, omega=0.8008）：
  Verlet 相对修正 = 1.0688e-5
  它占 c0 的比例 = 1.0688e-5 / 2.5694e-4 = 4.2%
  dt 由 0.02 降到 0.01，修正减到 1/4，c0 应增加 0.75*1.0688e-5 = 8.0e-6
  即以 c0 计的相对增幅 3.1%

判据：
  HA0 冒烟：dt=0.02 的四档读数复现第 706/707 号
  HA1 dt 扫描（主判据）：三档 dt 乘四档 xi，提取 c0(dt) 与 c1(dt)
  HA2 Verlet 预测对照（主判据）：c0(dt) 的变化量与解析预测比
  HA3 dt->0 外推：c0(dt) = c0_inf - a*dt^2 拟合，给 c0_inf 与理论比
  HA4 判定

预登记判据（先于读数写定）：
  HA0 PASS：四档读数与第 707 号相对差均 < 2%
  HA1 报告量：c0(0.02)、c0(0.01)、c0(0.005)
  HA2 PASS：c0(0.01)-c0(0.02) 的实测值与 Verlet 预测 8.0e-6 之比落在 [0.5, 2.0]
  HA3 报告量：c0_inf 与理论值之比
  HA4 综合：HA2 PASS 判「0.88 缺口主要由 Verlet 时间离散化贡献」；
            HA2 FAIL 且 c0 随 dt 不变判「缺口不在积分器」；
            HA2 FAIL 且变化方向相反判「缺口来源与积分器相反，须另查」

运行：python 验证脚本/p708_dt_convergence.py（确定性，约 7 分钟）
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

IMG = r"D:\software\TianXingFC\images\p708_dt_convergence.png"

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
DTS = (0.02, 0.01, 0.005)

KK = 2.0 * math.pi * np.fft.fftfreq(N)
K2 = KK ** 2

REC_A = {14.0: 1.7215e-3, 20.0: 1.2868e-3, 28.0: 9.8952e-4, 42.0: 7.4460e-4}
C0_MEAS = 2.5694e-4          # 第 707 号四档拟合
C0_THEORY = abs(4.0 * math.sin(KMOD / 2.0) ** 2 - KMOD ** 2) / (2.0 * TARGET ** 2)


def verlet_rel(dt):
    """Verlet 对简谐振子的相对频率偏差 (2/dt)arcsin(w dt/2)/w - 1。"""
    return (2.0 / dt) * math.asin(WB * dt / 2.0) / WB - 1.0


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


def make_ic(psi, d1):
    sh = 1.0 / G
    s0 = sh * xf
    th0 = -WB * sh * V * xf
    c = sh * V
    wfac = WB * sh
    F = np.interp(s0, xf, psi)
    Fp = np.interp(s0, xf, d1)
    return F * np.cos(th0), -c * Fp * np.cos(th0) - wfac * F * np.sin(th0)


def run(phi, vel, U0, wsig, dt, T=1200.0, every_base=4):
    """every_base 表示每 0.08 个时间单位采一次，与 dt 解耦。"""
    every = max(int(round(0.08 / dt)), 1)
    cs = []
    nst = int(T / dt)
    for s in range(nst):
        t = (s + .5) * dt
        w1 = -U0 * np.exp(-((xf - V * t) ** 2) / (2.0 * wsig ** 2))
        acc = (np.roll(phi, 1) + np.roll(phi, -1) - 2.0 * phi) - WGAP2 * phi - w1 * phi
        phi = phi + dt * vel + .5 * dt * dt * acc
        t2 = (s + 1) * dt
        w2 = -U0 * np.exp(-((xf - V * t2) ** 2) / (2.0 * wsig ** 2))
        acc2 = (np.roll(phi, 1) + np.roll(phi, -1) - 2.0 * phi) - WGAP2 * phi - w2 * phi
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


def measure(sg, dt):
    U0, _, psi, d1 = fit_U0(sg)
    phi, vel = make_ic(psi, d1)
    cs = run(phi, vel, U0, sg, dt)
    wm, res = freq_refined(cs, 0.08, TARGET)
    return abs(wm - TARGET) / TARGET, res


def fit_poly(xis, ys, terms):
    xi = np.asarray(xis, float)
    y = np.asarray(ys, float)
    X = np.stack([xi ** (-t) for t in terms], axis=1)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    rss = float(np.sum((y - pred) ** 2))
    ss = float(np.sum((y - np.mean(y)) ** 2))
    return coef, 1.0 - rss / max(ss, 1e-300), float(np.max(np.abs(pred / y - 1)))


print("=" * 78)
print("第 708 号：c0 的 dt 收敛与 0.88 缺口的归因（判据 HA0 至 HA4）")
print("=" * 78)
print("装置：N=%d omega_b=%.4f v=%.2f gamma=%.6f k_mod=%.6f"
      % (N, WB, V, G, KMOD))
print("第 707 号：c0 实测 %.4e，理论 %.4e，比值 %.3f"
      % (C0_MEAS, C0_THEORY, C0_MEAS / C0_THEORY))
print("Verlet 相对修正：dt=0.02 给 %.4e（占 c0 的 %.1f%%）"
      % (verlet_rel(0.02), 100 * verlet_rel(0.02) / C0_MEAS))
for dt in DTS:
    print("  dt=%.3f  Verlet 相对修正 = %.4e" % (dt, verlet_rel(dt)))

# ---------------- HA0 冒烟 ----------------
print()
print("[HA0] 冒烟：dt=0.02 复现第 707 号四档读数")
A002 = []
worst = 0.0
for sg in SIGS:
    dev, res = measure(sg, 0.02)
    A002.append(dev)
    worst = max(worst, abs(dev / REC_A[sg] - 1))
    print("    xi=%-6.0f dev=%.4e 残差比=%.3f 与第707号相对差=%+.1f%%"
          % (sg, dev, res, 100 * (dev / REC_A[sg] - 1)))
print("    最大相对差 = %.1f%%（判据 < 2%%）" % (100 * worst))
HA0 = worst < 0.02
print("    HA0 判定：%s" % ("PASS" if HA0 else "FAIL"))

# ---------------- HA1 dt 扫描 ----------------
print()
print("[HA1] dt 扫描（三档 dt 乘四档 xi）")
COEF = {}
for dt in DTS:
    if dt == 0.02:
        devs = list(A002)
        print("    dt=0.020（复用 HA0 读数）")
    else:
        devs = []
        for sg in SIGS:
            dev, res = measure(sg, dt)
            devs.append(dev)
            print("    dt=%.3f xi=%-6.0f dev=%.4e 残差比=%.3f" % (dt, sg, dev, res))
    cf, r2f, mrf = fit_poly(SIGS, devs, [0, 1])
    COEF[dt] = (cf[0], cf[1], r2f, mrf, devs)
    print("    拟合：dt=%.3f  c0=%.6e  c1=%.6e  R^2=%.8f 最大残差=%.3f%%"
          % (dt, cf[0], cf[1], r2f, 100 * mrf))
print()
print("    三档 c0 汇总：")
for dt in DTS:
    c0 = COEF[dt][0]
    print("      dt=%.3f  c0=%.6e  c0/理论=%.4f" % (dt, c0, c0 / C0_THEORY))

# ---------------- HA2 Verlet 预测对照 ----------------
print()
print("[HA2] Verlet 预测对照（主判据）")
pred_0102 = verlet_rel(0.02) - verlet_rel(0.01)
pred_00501 = verlet_rel(0.01) - verlet_rel(0.005)
real_0102 = COEF[0.01][0] - COEF[0.02][0]
real_00501 = COEF[0.005][0] - COEF[0.01][0]
print("    Verlet 预测 c0 增幅：")
print("      dt 0.02 -> 0.01：%.4e" % pred_0102)
print("      dt 0.01 -> 0.005：%.4e" % pred_00501)
print("    实测 c0 增幅：")
print("      dt 0.02 -> 0.01：%.4e" % real_0102)
print("      dt 0.01 -> 0.005：%.4e" % real_00501)
r1 = real_0102 / pred_0102 if abs(pred_0102) > 1e-30 else float("nan")
r2 = real_00501 / pred_00501 if abs(pred_00501) > 1e-30 else float("nan")
print("    实测/预测：%.3f（第一段）、%.3f（第二段）" % (r1, r2))
HA2 = (0.5 <= r1 <= 2.0) if r1 == r1 else False
print("    HA2 判定（以第一段为准）：%s" % ("PASS" if HA2 else "FAIL"))

# ---------------- HA3 dt->0 外推 ----------------
print()
print("[HA3] dt -> 0 外推：c0(dt) = c0_inf - a*dt^2")
dts = np.array(DTS)
c0s = np.array([COEF[d][0] for d in DTS])
X = np.stack([np.ones_like(dts), dts ** 2], axis=1)
cf, *_ = np.linalg.lstsq(X, c0s, rcond=None)
c0_inf = float(cf[0])
print("    c0_inf = %.6e，外推系数 a = %.4e" % (c0_inf, cf[1]))
print("    c0_inf / 理论 = %.4f" % (c0_inf / C0_THEORY))
print("    三档实测对拟合：%s"
      % ", ".join("dt=%.3f 残差%+.2f%%" % (d, 100 * (c / c0_inf - cf[1] * d * d / c0_inf - 1))
                  for d, c in zip(DTS, c0s)))
print("    若 c0_inf/理论 接近 1，则 0.88 缺口完全由时间离散化贡献。")

# ---------------- HA4 判定 ----------------
print()
print("[HA4] 判定")
print("    HA0=%s HA2=%s" % tuple("PASS" if x else "FAIL" for x in (HA0, HA2)))
c0_trend = c0s[-1] - c0s[0]
print("    c0 随 dt 减小的总变化 = %+.4e（%.2f%%）"
      % (c0_trend, 100 * c0_trend / c0s[0]))
if not HA0:
    verdict = "复现失败，本轮读数不可用"
elif HA2:
    verdict = ("0.88 缺口主要由 Verlet 时间离散化贡献；dt->0 外推给 c0_inf/理论 = %.3f"
               % (c0_inf / C0_THEORY))
elif c0_trend <= 0:
    verdict = "缺口不在积分器（c0 不随 dt 减小而上升，实测变化 %+.2e）" % c0_trend
else:
    verdict = ("c0 随 dt 上升但幅度与 Verlet 预测不符（实测/预测 %.2f），缺口另需解释" % r1)
print("    判语：%s" % verdict)
print()
print("    缺口分解（以 c0/理论 为 1 计）：")
print("      实测 0.02 档：%.4f" % (c0s[0] / C0_THEORY))
print("      Verlet 贡献：%.4f" % (verlet_rel(0.02) / C0_THEORY))
print("      剩余：%.4f" % ((c0_inf - verlet_rel(0.02) * 0) / C0_THEORY))
print("耗时 %.1f 秒" % (time.time() - T0))

# ---------------- 图 ----------------
fig, ax = plt.subplots(1, 3, figsize=(15.8, 4.4))
xi = np.array(SIGS)
inv = np.linspace(0.020, 0.075, 100)
cols = plt.cm.viridis(np.linspace(0.2, 0.85, len(DTS)))
for i, dt in enumerate(DTS):
    ax[0].plot(1.0 / xi, COEF[dt][4], "o", ms=5, color=cols[i],
               label="dt=%.3f" % dt)
    ax[0].plot(inv, COEF[dt][0] + COEF[dt][1] * inv, "-", lw=1.0, color=cols[i])
ax[0].set_xlabel(r"$1/\xi$")
ax[0].set_ylabel("dev")
ax[0].set_title("左：三档 dt 的宽度标度")
ax[0].legend(fontsize=7)

ax[1].errorbar(DTS, c0s, yerr=None, fmt="o-", color="#268", ms=7, label="$c_0$ 实测")
ax[1].axhline(C0_THEORY, color="#c44", ls="--", lw=1.2, label="理论值")
ax[1].axhline(c0_inf, color="#888", ls=":", lw=1.1, label="$c_0^{\\infty}$ 外推")
ax[1].set_xscale("log")
ax[1].set_xlabel("dt")
ax[1].set_ylabel(r"$c_0$")
ax[1].set_title("中：$c_0$ 的 dt 收敛")
ax[1].legend(fontsize=8)

ax[2].plot(dts ** 2, c0s, "o", color="#268", ms=7, label="$c_0$ 实测")
xx = np.linspace(0, max(dts ** 2) * 1.1, 50)
ax[2].plot(xx, c0_inf + cf[1] * xx, "-", color="#c44", lw=1.1, label="二次外推")
ax[2].axhline(C0_THEORY, color="#888", ls="--", lw=1.1, label="理论值")
ax[2].set_xlabel(r"$dt^2$")
ax[2].set_ylabel(r"$c_0$")
ax[2].set_title("右：Richardson 外推（$c_0^{\\infty}$/理论 = %.3f）"
                % (c0_inf / C0_THEORY))
ax[2].legend(fontsize=8)
plt.tight_layout()
plt.savefig(IMG, dpi=130)
print("图已存：", IMG)
