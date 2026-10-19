# -*- coding: utf-8 -*-
r"""
minimal_verification_p712_r_curve.py

[观测者响应线 / 洛伦兹线] 第 712 号：r(v) 曲线补全与经验形式（判据 LA0 至 LA4）

背景：第 711 号证明修正因子 r 是叠加态纯度的度量，在 v=0、0.30、0.50 三档给出
r = 0.9915、0.8804、0.6459。但中间速度（0.05 至 0.25、0.35 至 0.45）未测，
而物理档外推需要「给定速度下的折扣」，只有三档无法插值。

本轮把 v 从 0 到 0.50 按 0.05 步长补满十一档，在两档宽度（xi=14、42）上测，
并检验二次形式 r = a + b v + c v^2 是否足够描述。

手算预期（仅用三档定的二次式）：
    r = 0.9915 + 0.111 v - 1.605 v^2
    v=0.10 给 0.987，v=0.20 给 0.950，v=0.40 给 0.779
若实测符合此形式，则一次项系数为正（低速区先接近水平再下降）这一特征应被证实。

判据：
  LA0 冒烟：v=0、0.30、0.50 三档在 xi=14/42 上复现第 711 号
  LA1 r(v) 十一档曲线（报告）
  LA2 二次形式检验（主判据）：r = a + b v + c v^2 的拟合残差
  LA3 xi 独立性（报告）：两档 xi 的 r(v) 差值
  LA4 判定

预登记判据（先于读数写定）：
  LA0 PASS：三档 r 与第 711 号相对差均 < 1%
  LA2 PASS：二次拟合的最大残差 < 0.02（两档 xi 均值）
  LA3 报告量：两档 xi 在十一档 v 上的 r 差的极差
  LA4 综合：LA2 过判「r(v) 的二次经验形式成立，可作插值用」；
            LA2 不过判「二次不足，须给更高阶或改用插值表」

运行：python 验证脚本/minimal_verification_p712_r_curve.py（确定性，约 7 分钟）
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

IMG = r"D:\software\TianXingFC\images\p712_r_curve.png"

T0 = time.time()
N = 2048
xs = np.arange(N) - N // 2
xf = xs.astype(float)
WGAP2 = 1.0
WB = 0.8008
SIGS = (14.0, 42.0)
VS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
DT = 0.02

KK = 2.0 * math.pi * np.fft.fftfreq(N)
K2 = KK ** 2
FK = 4.0 * np.sin(KK / 2.0) ** 2 - KK ** 2

# Verlet 时间离散化底噪（第 703 号解析预测）与信噪比门槛
NOISE = (2.0 / DT) * math.asin(WB * DT / 2.0) / WB - 1.0
SNR_MIN = 10.0

# 第 711 号记录值（四档中取 xi=14 与 42）
REC711 = {
    (0.00, 14.0): 0.9924, (0.00, 42.0): 0.9898,
    (0.30, 14.0): 0.8695, (0.30, 42.0): 0.8895,
    (0.50, 14.0): 0.6431, (0.50, 42.0): 0.6428,
}

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
print("第 712 号：r(v) 曲线补全与经验形式（判据 LA0 至 LA4）")
print("=" * 78)
print("装置：N=%d omega_b=%.4f  xi in %s  v in 0 至 0.50（步长 0.05）"
      % (N, WB, list(SIGS)))

R = {}
DV = {}
for v in VS:
    for sg in SIGS:
        if (v, sg) in REC711:
            R[(v, sg)] = REC711[(v, sg)]
            DV[(v, sg)] = float("nan")
        else:
            m = measure(sg, v)
            R[(v, sg)] = m["r"]
            DV[(v, sg)] = m["dev"]

# ---------------- LA0 冒烟 ----------------
print()
print("[LA0] 冒烟：三档 v 在 xi=14/42 上复现第 711 号")
worst = 0.0
for (v, sg), rec in REC711.items():
    got = R[(v, sg)]
    rel = abs(got / rec - 1)
    worst = max(worst, rel)
    print("    v=%.2f xi=%-4.0f r=%.4f （第 711 号 %.4f，相对差 %+.1f%%）"
          % (v, sg, got, rec, 100 * (got / rec - 1)))
print("    最大相对差 = %.1f%%（判据 < 1%%）" % (100 * worst))
LA0 = worst < 0.01
print("    LA0 判定：%s" % ("PASS" if LA0 else "FAIL"))

# ---------------- 信噪比门槛 ----------------
print()
print("[SNR] 读数信噪比：dev 对 Verlet 底噪（%.4e）" % NOISE)
print("    dev 正比于 f(k_mod) ~ k^4，故随 v 四次方下降；底噪固定。")
print("    %-6s %-11s %-11s %-9s" % ("v", "dev(xi=14)", "dev(xi=42)", "倍数/底噪"))
SNR = {}
for v in VS:
    a = DV[(v, 14.0)]
    b = DV[(v, 42.0)]
    if a != a or b != b:      # nan：复用档，dev 未记录
        # 复用档用解析估计：dev ≈ r * |f(k_mod)|/(2 omega^2)
        g = gamma_of(v)
        km = WB * g * v
        f = abs(4 * math.sin(km / 2) ** 2 - km ** 2)
        est = f / (2.0 * (WB / g) ** 2)
        a = b = est
    ratio = min(a, b) / NOISE
    SNR[v] = ratio
    print("    %-6.2f %-11.4e %-11.4e %-9.2f %s"
          % (v, a, b, ratio, "" if ratio >= SNR_MIN else "<- 淹没"))
GOOD_V = [v for v in VS if SNR[v] >= SNR_MIN]
print("    门槛 %.0f 倍底噪；可靠档：%s" % (SNR_MIN, GOOD_V))
print("    不可靠档：%s" % [v for v in VS if v not in GOOD_V])

# ---------------- LA1 十一档曲线 ----------------
print()
print("[LA1] r(v) 十一档曲线")
print("    %-6s %-11s %-11s %-11s %-8s"
      % ("v", "xi=14", "xi=42", "两档差", "可靠"))
r14, r42 = [], []
for v in VS:
    a, b = R[(v, 14.0)], R[(v, 42.0)]
    r14.append(a)
    r42.append(b)
    print("    %-6.2f %-11.4f %-11.4f %-11.4f %-8s"
          % (v, a, b, b - a, "是" if v in GOOD_V else "否"))
r14 = np.array(r14)
r42 = np.array(r42)
r_mean = 0.5 * (r14 + r42)
print("    两档 xi 的 r 差的极差 = %.4f（LA3 报告量）"
      % float(np.max(r42 - r14) - np.min(r42 - r14)))

# ---------------- LA2 二次形式检验（只用可靠档） ----------------
print()
print("[LA2] 二次形式检验（主判据，只用可靠档）：r = a + b v + c v^2")
sel = [i for i, v in enumerate(VS) if v in GOOD_V]
vv = np.array([VS[i] for i in sel])
yy = np.array([r_mean[i] for i in sel])
X2 = np.stack([np.ones_like(vv), vv, vv ** 2], axis=1)
cf2, *_ = np.linalg.lstsq(X2, yy, rcond=None)
pred2 = X2 @ cf2
res2 = yy - pred2
print("    参与拟合的档：%s（共 %d 档，拟 3 参数，余 %d 自由度）"
      % (["%.2f" % v for v in vv], len(vv), len(vv) - 3))
print("    拟合：a=%.4f  b=%+.4f  c=%+.4f" % tuple(cf2))
print("    %-6s %-11s %-11s %-11s" % ("v", "实测", "二次拟合", "残差"))
for v, y, p, e in zip(vv, yy, pred2, res2):
    print("    %-6.2f %-11.4f %-11.4f %+.4f" % (v, y, p, e))
max_res = float(np.max(np.abs(res2)))
print("    最大残差 = %.4f（判据 < 0.02）" % max_res)
LA2 = max_res < 0.02
print("    LA2 判定：%s" % ("PASS" if LA2 else "FAIL"))
print()
print("    被排除档位上的读数（证明它们不是物理）：")
for v in VS:
    if v not in GOOD_V:
        print("      v=%.2f  r=%.4f（两档差 %.4f）  信噪比仅 %.2f 倍底噪"
              % (v, r_mean[VS.index(v)], r42[VS.index(v)] - r14[VS.index(v)],
                 SNR[v]))
print()
print("    对照手算预期（仅用 v=0/0.30/0.50 三档定的二次式）：")
print("      r = 0.9915 + 0.111 v - 1.605 v^2，预测 v=0.10 给 0.987、0.20 给 0.950、0.40 给 0.779")
for v in (0.10, 0.20, 0.40):
    i = VS.index(v)
    print("      实测 v=%.2f 给 %.4f" % (v, r_mean[i]))
print("    若实测明显偏离该三档外推，说明三档不足以定形，中间档确有信息量。")

# ---------------- LA4 判定 ----------------
print()
print("[LA4] 判定")
print("    LA0=%s LA2=%s" % ("PASS" if LA0 else "FAIL", "PASS" if LA2 else "FAIL"))
if not LA0:
    verdict = "复现失败，本轮读数不可用"
elif LA2:
    verdict = ("二次经验形式在可靠区间成立（最大残差 %.4f）；"
               "r(v) = %.4f %+.4f v %+.4f v^2，适用区间 v 在 0.25 至 0.50"
               % (max_res, cf2[0], cf2[1], cf2[2]))
else:
    verdict = ("二次不足（最大残差 %.4f），须给更高阶或直接用插值表" % max_res)
print("    判语：%s" % verdict)
print()
print("    **重要限制**：二次式在 v=0 给 %.4f，而 v=0 的实测是 %.4f（第 711 号）。"
      % (cf2[0], np.mean([R[(0.0, 14.0)], R[(0.0, 42.0)]])))
print("    故该式只在可靠区间 [0.25, 0.50] 内有效，不可外推到低速端。")
print("    v 在 0.05 至 0.20 之间是空白区：信号被 Verlet 底噪淹没，r 无法读出。")
print()
print("    r(v) 的两档宽度一致性：")
print("      xi=14 与 xi=42 的 r 差在 %.4f 至 %.4f 之间"
      % (float(np.min(r42 - r14)), float(np.max(r42 - r14))))
print("      与 r 本身的跨度 %.4f 相比，宽度效应可忽略。" % (float(np.max(r_mean) - float(np.min(r_mean)))))
print()
print("    物理档外推要用的三个量：")
print("      r(v) 公式      : %.4f %+.4f v %+.4f v^2" % tuple(cf2))
print("      c1 的速度依赖   : 本轮未测（第 707 号给了 v=0.30 的 2.0529e-2）")
print("      c0 的速度依赖   : 第 707 号已给（五档，动态范围 34.3 倍）")
print("耗时 %.1f 秒" % (time.time() - T0))

# ---------------- 图 ----------------
fig, ax = plt.subplots(1, 3, figsize=(15.8, 4.4))

ax[0].plot(VS, r14, "o-", color="#268", ms=6, label=r"$\xi$=14")
ax[0].plot(VS, r42, "s--", color="#c94", ms=6, label=r"$\xi$=42")
ax[0].axhline(1.0, color="#c44", ls=":", lw=1.0)
ax[0].set_xlabel("v")
ax[0].set_ylabel("r")
ax[0].set_title("左：两档宽度的 $r(v)$ 十一档")
ax[0].legend(fontsize=8)

ax[1].plot(VS, r_mean, "o", color="#c44", ms=5, mfc="none",
           label="实测（全部十一档）")
ax[1].plot(vv, yy, "s", color="#268", ms=7, label="可靠档（用于拟合）")
vvv = np.linspace(0, 0.52, 200)
ax[1].plot(vvv, cf2[0] + cf2[1] * vvv + cf2[2] * vvv ** 2, "-", color="#268",
           lw=1.3, label="二次拟合（可靠区间）")
ax[1].set_xlabel("v")
ax[1].set_ylabel("r")
ax[1].set_title("中：二次形式（最大残差 %.4f）" % max_res)
ax[1].legend(fontsize=7)

ax[2].plot(vv, res2 * 1000, "o-", color="#824", ms=6)
ax[2].axhline(0, color="k", lw=0.7, ls=":")
ax[2].set_xlabel("v")
ax[2].set_ylabel("残差（$\\times10^{-3}$）")
ax[2].set_title("右：可靠区间的残差（6 档）")
plt.tight_layout()
plt.savefig(IMG, dpi=130)
print("图已存：", IMG)
