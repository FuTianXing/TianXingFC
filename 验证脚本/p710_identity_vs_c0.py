# -*- coding: utf-8 -*-
r"""
p710_identity_vs_c0.py

[观测者响应线 / 洛伦兹线] 第 710 号：恒等式比值与 c0 比值的不自洽（判据 JA0 至 JA4）

背景：第 709 号量出 omega_lat^2 - omega_cont^2 = r(xi) * <f>，四档 r 为 0.8695 至 0.8895。
若该式成立，则 dev = r*<f>/(2 omega^2)，故 c0 应为 r_inf * |f(k_mod)|/(2 omega^2)，
其中 r_inf 是 r(xi) 在 xi->inf 的极限。而第 706/707 号实测 c0/理论 = 0.8965（dt=0.02）。

两者必须自洽，否则第 705 号那套公式在定量上不可靠，包括它给出的 2.8660e-4。
手算：r 的有限值 0.8695 至 0.8895 按 1/xi 外推给约 0.90，与 0.8965 接近。
但第 708 号的 dt->0 外推给 c0/理论 = 0.9416，若 r_inf 不随 dt 变，则出现 4.4% 的不自洽。

判据：
  JA0 冒烟：dt=0.02 四档读数复现第 706/707/709 号
  JA1 r_inf 的外推（dt=0.02，七档 xi）
  JA2 r_inf 的 dt 依赖（dt=0.01，四档 xi）
  JA3 自洽性（主判据）：r_inf(dt) 与 c0(dt)/理论 的一致性
  JA4 判定

预登记判据（先于读数写定）：
  JA0 PASS：四档读数与第 709 号相对差均 < 2%
  JA1/JA2 报告量：r(xi) 与 r_inf
  JA3 PASS：|r_inf(0.02) - c0(0.02)/理论| < 0.03
  JA4 综合：JA3 过判「恒等式外推解释了 dt=0.02 下的全部缺口」；
            JA3 不过判「两套读数不自洽，第 705 号公式的定量使用须加保留」

运行：python 验证脚本/p710_identity_vs_c0.py（确定性，约 6 分钟）
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

IMG = r"D:\software\TianXingFC\images\p710_identity_vs_c0.png"

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
SIGS4 = (14.0, 20.0, 28.0, 42.0)
SIGS7 = (14.0, 20.0, 28.0, 42.0, 63.0, 90.0, 128.0)

KK = 2.0 * math.pi * np.fft.fftfreq(N)
K2 = KK ** 2
FK = 4.0 * np.sin(KK / 2.0) ** 2 - KK ** 2

F_KMOD = abs(4.0 * math.sin(KMOD / 2.0) ** 2 - KMOD ** 2)
C0_THEORY = F_KMOD / (2.0 * TARGET ** 2)
C0_DT002 = 2.5694e-4      # 第 707 号（dt=0.02，四档）
C0_INF = 2.6987e-4        # 第 708 号（dt->0 外推）
REC_R = {14.0: 0.8695, 20.0: 0.8747, 28.0: 0.8880, 42.0: 0.8895}

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


def make_ic_boost(psi, d1):
    s0 = G * xf
    th0 = WB * G * V * xf
    c = G * V
    wfac = WB * G
    F = np.interp(s0, xf, psi)
    Fp = np.interp(s0, xf, d1)
    return F * np.cos(th0), -c * Fp * np.cos(th0) + wfac * F * np.sin(th0)


def run(phi, vel, U0, wsig, evo, dt, T=1200.0, spec_every=200):
    every = max(int(round(0.08 / dt)), 1)
    cs, specs = [], []
    nst = int(T / dt)
    for s in range(nst):
        t = (s + .5) * dt
        w1 = -U0 * np.exp(-G * G * ((xf - V * t) ** 2) / (2.0 * wsig ** 2))
        if evo == "lattice":
            acc = (np.roll(phi, 1) + np.roll(phi, -1) - 2.0 * phi) - WGAP2 * phi - w1 * phi
        else:
            acc = -np.real(np.fft.ifft(K2 * np.fft.fft(phi))) - WGAP2 * phi - w1 * phi
        phi = phi + dt * vel + .5 * dt * dt * acc
        t2 = (s + 1) * dt
        w2 = -U0 * np.exp(-G * G * ((xf - V * t2) ** 2) / (2.0 * wsig ** 2))
        if evo == "lattice":
            acc2 = (np.roll(phi, 1) + np.roll(phi, -1) - 2.0 * phi) - WGAP2 * phi - w2 * phi
        else:
            acc2 = -np.real(np.fft.ifft(K2 * np.fft.fft(phi))) - WGAP2 * phi - w2 * phi
        vel = vel + .5 * dt * (acc + acc2)
        if s % every == 0:
            cs.append(float(phi[N // 2 + int(round(V * t2))]))
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


def measure(sg, dt):
    U0, _, psi, d1 = fit_U0(sg)
    out = {}
    for evo in ("lattice", "continuum"):
        phi, vel = make_ic_boost(psi, d1)
        cs, specs = run(phi, vel, U0, sg, evo, dt)
        wm, res = freq_refined(cs, 0.08, TARGET)
        spec = np.mean(np.array(specs), axis=0)
        out[evo] = dict(w=wm, res=res, fbar=float(np.sum(spec * FK)))
    wl, wc = out["lattice"]["w"], out["continuum"]["w"]
    fb = out["lattice"]["fbar"]
    out["r"] = (wl * wl - wc * wc) / fb if abs(fb) > 1e-30 else float("nan")
    out["dev"] = abs(wl - TARGET) / TARGET
    return out


def fit_c0(xis, devs):
    xi = np.asarray(xis, float)
    y = np.asarray(devs, float)
    X = np.stack([np.ones_like(xi), 1.0 / xi], axis=1)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    r2 = 1.0 - float(np.sum((y - pred) ** 2)) / max(float(np.sum((y - y.mean()) ** 2)), 1e-300)
    return float(coef[0]), float(coef[1]), r2


print("=" * 78)
print("第 710 号：恒等式比值与 c0 比值的不自洽（判据 JA0 至 JA4）")
print("=" * 78)
print("装置：N=%d omega_b=%.4f v=%.2f gamma=%.6f k_mod=%.6f"
      % (N, WB, V, G, KMOD))
print("理论 |f(k_mod)|/(2omega^2) = %.6e" % C0_THEORY)
print("第 707 号 c0(dt=0.02) = %.4e，比值 %.4f" % (C0_DT002, C0_DT002 / C0_THEORY))
print("第 708 号 c0(dt->0) = %.4e，比值 %.4f" % (C0_INF, C0_INF / C0_THEORY))

# ---------------- JA0 冒烟 + JA1 七档 ----------------
print()
print("[JA0/JA1] 冒烟与 r_inf 外推（dt=0.02，七档 xi）")
R7, A7 = [], []
worst = 0.0
for sg in SIGS7:
    m = measure(sg, 0.02)
    R7.append(m["r"])
    A7.append(m["dev"])
    if sg in REC_R:
        rel = abs(m["r"] / REC_R[sg] - 1)
        worst = max(worst, rel)
        print("    xi=%-6.0f r=%.4f  dev=%.4e  r 与第709号相对差=%+.1f%%"
              % (sg, m["r"], m["dev"], 100 * (m["r"] / REC_R[sg] - 1)))
    else:
        print("    xi=%-6.0f r=%.4f  dev=%.4e" % (sg, m["r"], m["dev"]))
print("    JA0：四档 r 与第 709 号的最大相对差 = %.1f%%（判据 < 2%%）" % (100 * worst))
JA0 = worst < 0.02
print("    JA0 判定：%s" % ("PASS" if JA0 else "FAIL"))
inv7 = 1.0 / np.array(SIGS7)
X = np.stack([np.ones_like(inv7), inv7], axis=1)
cf, *_ = np.linalg.lstsq(X, np.array(R7), rcond=None)
r_inf_02 = float(cf[0])
print("    七档 r 对 1/xi 的线性外推：r_inf = %.4f（斜率 %.4f）" % (r_inf_02, cf[1]))
c0_02, c1_02, r2_02 = fit_c0(SIGS7, A7)
print("    七档 dev 的 c0+c1/xi 拟合：c0=%.4e c1=%.4e R^2=%.6f" % (c0_02, c1_02, r2_02))
print("    r_inf 与 c0/理论 = %.4f 的差 = %+.4f" % (c0_02 / C0_THEORY,
                                                  r_inf_02 - c0_02 / C0_THEORY))

# ---------------- JA2 r_inf 的 dt 依赖 ----------------
print()
print("[JA2] r_inf 的 dt 依赖（dt=0.01，四档 xi）")
R4_01, A4_01 = [], []
for sg in SIGS4:
    m = measure(sg, 0.01)
    R4_01.append(m["r"])
    A4_01.append(m["dev"])
    print("    xi=%-6.0f r=%.4f  dev=%.4e" % (sg, m["r"], m["dev"]))
inv4 = 1.0 / np.array(SIGS4)
cf4, *_ = np.linalg.lstsq(np.stack([np.ones_like(inv4), inv4], axis=1),
                          np.array(R4_01), rcond=None)
r_inf_01 = float(cf4[0])
c0_01, _, r2_01 = fit_c0(SIGS4, A4_01)
print("    四档 r 外推：r_inf(dt=0.01) = %.4f" % r_inf_01)
print("    四档 dev 拟合：c0(dt=0.01) = %.4e，c0/理论 = %.4f"
      % (c0_01, c0_01 / C0_THEORY))
print("    对照第 708 号：c0(dt=0.01) 实测 2.6665e-4，比值 %.4f"
      % (2.666545e-4 / C0_THEORY))
print("    r_inf 的 dt 变化：%.4f -> %.4f（%+.4f）"
      % (r_inf_02, r_inf_01, r_inf_01 - r_inf_02))

# ---------------- JA3 自洽性 ----------------
print()
print("[JA3] 自洽性（主判据）：r_inf 与 c0/理论 是否一致")
print("    %-10s %-13s %-13s %-11s"
      % ("dt", "r_inf", "c0/理论", "差"))
rows = [(0.02, r_inf_02, c0_02 / C0_THEORY), (0.01, r_inf_01, c0_01 / C0_THEORY)]
for dt, ri, ratio in rows:
    print("    %-10.3f %-13.4f %-13.4f %+.4f" % (dt, ri, ratio, ri - ratio))
gaps = [abs(ri - ratio) for _, ri, ratio in rows]
worst_gap = max(gaps)
print("    最大差 = %.4f（判据 < 0.03）" % worst_gap)
JA3 = worst_gap < 0.03
print("    JA3 判定：%s" % ("PASS" if JA3 else "FAIL"))
print()
print("    与第 708 号的 dt->0 外推对照：c0(dt->0)/理论 = %.4f"
      % (C0_INF / C0_THEORY))
print("    若 r_inf 本身也随 dt 上升，则两者可继续自洽；")
print("    r_inf 从 dt=0.02 到 0.01 的变化是 %+.4f，c0/理论 的变化是 %+.4f。"
      % (r_inf_01 - r_inf_02, c0_01 / C0_THEORY - c0_02 / C0_THEORY))

# ---------------- JA4 判定 ----------------
print()
print("[JA4] 判定")
print("    JA0=%s JA1/JA2=报告 JA3=%s" % ("PASS" if JA0 else "FAIL",
                                        "PASS" if JA3 else "FAIL"))
if not JA0:
    verdict = "复现失败，本轮读数不可用"
elif JA3:
    verdict = ("恒等式外推解释了 dt=0.02 下的缺口：r_inf=%.4f 与 c0/理论=%.4f 一致到 %.1f%%；"
               "第 705 号公式的定量使用成立，剩余缺口即恒等式在叠加态上的偏差"
               % (r_inf_02, c0_02 / C0_THEORY, 100 * worst_gap))
else:
    verdict = ("两套读数不自洽：r_inf=%.4f 对 c0/理论=%.4f，差 %.1f%%；"
               "第 705 号公式的定量使用须加保留" % (r_inf_02, c0_02 / C0_THEORY,
                                                    100 * worst_gap))
print("    判语：%s" % verdict)
print()
print("    这十几轮攒下的账，按本轮的结果重排：")
print("      理论值（未修正）              : %.4e" % C0_THEORY)
print("      恒等式修正后的理论（r_inf 倍）: %.4e" % (r_inf_02 * C0_THEORY))
print("      实测（dt=0.02, 七档）         : %.4e" % c0_02)
print("      实测（dt->0 外推）            : %.4e" % C0_INF)
print("耗时 %.1f 秒" % (time.time() - T0))

# ---------------- 图 ----------------
fig, ax = plt.subplots(1, 3, figsize=(15.8, 4.4))
xi = np.array(SIGS7)

ax[0].plot(1.0 / xi, R7, "o", color="#268", ms=6, label="dt=0.02")
ax[0].plot(1.0 / np.array(SIGS4), R4_01, "s", color="#c94", ms=6, label="dt=0.01")
inv = np.linspace(0.0, 0.075, 100)
ax[0].plot(inv, r_inf_02 + cf[1] * inv, "-", color="#268", lw=1.0)
ax[0].axhline(c0_02 / C0_THEORY, color="#c44", ls="--", lw=1.2,
              label="c0/理论（dt=0.02）")
ax[0].set_xlabel(r"$1/\xi$")
ax[0].set_ylabel("r")
ax[0].set_title("左：恒等式比值的外推（$r_\\infty$=%.4f）" % r_inf_02)
ax[0].legend(fontsize=7)

ax[1].bar([0, 1], [r_inf_02, r_inf_01], width=0.45, color=["#268", "#c94"])
ax[1].bar([0, 1], [c0_02 / C0_THEORY, c0_01 / C0_THEORY], width=0.20,
          color=["#7ab", "#f4b"], label="c0/理论")
ax[1].set_xticks([0, 1])
ax[1].set_xticklabels(["dt=0.02", "dt=0.01"])
ax[1].set_ylabel("比值")
ax[1].set_title("中：$r_\\infty$ 与 $c_0$/理论 的一致性")
ax[1].legend(fontsize=7)

labels = ["理论(未修正)", "理论 x $r_\\infty$", "实测 dt=0.02", "实测 dt$\\to$0"]
vals = [C0_THEORY, r_inf_02 * C0_THEORY, c0_02, C0_INF]
ax[2].bar(range(4), vals, color=["#888", "#268", "#c94", "#824"], width=0.6)
for i, v in enumerate(vals):
    ax[2].text(i, v * 1.004, "%.2e" % v, ha="center", fontsize=7)
ax[2].set_xticks(range(4))
ax[2].set_xticklabels(labels, fontsize=7, rotation=15, ha="right")
ax[2].set_ylabel(r"$c_0$")
ax[2].set_title("右：四档数值的重排")
plt.tight_layout()
plt.savefig(IMG, dpi=130)
print("图已存：", IMG)
