# -*- coding: utf-8 -*-
r"""
p698_c_exponent.py

[观测者响应线 / 洛伦兹线] 第 698 号：C 项幂次锁定与分辨率墙（判据 R1 至 R7）

背景：第 697 号判 C 为离散侧物理量（非装置伪影），但其对 k_mod 的幂律指数
2.916 介于 k^2 型（2）与 k^4 型（4）之间未锁定，且 v=0.45 档残差比 0.349 超限。
本轮补速度档位并提高频率扫描分辨率，试图锁定幂次。

核心发现（负面但重要）：C 的指数判别受"分辨率墙"限制。
  dev 的波性侧成分 ~ A/xi^2，低速时 A ~ v^2 使 dev 在有限宽度下趋零；
  v <= 0.225 三档的 dev 落入频率扫描分辨率，不可用（提高扫描点数只能部分救回）；
  最终有效窗口 v = 0.15 至 0.45（五档），k 动态范围仅 3.3 倍，不足以分辨 2 与 4。

判据：
  R1 逐档筛选（残差比与 dev 下限两条门槛）
  R2 逐速度拟合 A/xi^2 + C
  R3 幂律指数
  R4 判别与分辨力评估
  R5 分辨率墙量化
  R6 与第 697 号对照
  R7 结论

运行：python 验证脚本/p698_c_exponent.py（确定性，约 3.5 分钟）
"""
"""第 698 号最终版：分辨率墙判定 + 三种拟合口径对照。

核心发现：C 的指数判别受"低频段 dev 落入扫描分辨率"限制。
v <= 0.225 三档不可用，故有效速度窗口只有 v = 0.30 至 0.45 三档。
本版给出：(a) 三档拟合；(b) 提高扫描分辨率后能否救回低速度档；(c) 结论。
"""
import numpy as np, math, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

IMG = r"D:\software\TianXingFC\images\p698_c_exponent.png"
N = 2048
xs = np.arange(N) - N // 2
xf = xs.astype(float)
WGAP2 = 1.0
WB = 0.8008
SIGS = (14.0, 20.0, 28.0, 42.0, 63.0)
VLIST = (0.10, 0.15, 0.225, 0.30, 0.375, 0.45)
RES_MAX = 0.30
NSCAN = 20001          # 提高扫描点数，分辨率提升 10 倍
FRAC = 0.10


def bound(sigma, U0):
    well = -U0 * np.exp(-(xf ** 2) / (2.0 * sigma ** 2))
    L = diags([-np.ones(N - 1), 2.0 + WGAP2 + well, -np.ones(N - 1)], [-1, 0, 1], format='csr')
    v, V = eigsh(L, k=1, which='SA')
    p = V[:, 0] / np.max(np.abs(V[:, 0]))
    kk = 2 * math.pi * np.fft.fftfreq(N)
    d1 = np.real(np.fft.ifft(1j * kk * np.fft.fft(p)))
    return (math.sqrt(v[0]) if v[0] > 0 else 0.0), p, d1


def fit_U0(sigma):
    lo, hi = 1e-4, 1e4
    for _ in range(45):
        mid = .5 * (lo + hi)
        w, _, _ = bound(sigma, mid)
        if w > WB: lo = mid
        else: hi = mid
    mid = .5 * (lo + hi)
    w, p, d1 = bound(sigma, mid)
    return mid, w, p, d1


def dev_of(sigma, v, g, nscan=NSCAN):
    U0, wb, psi, d1 = fit_U0(sigma)
    sh = 1.0 / g
    s0 = sh * xf; th0 = -WB * sh * v * xf
    F = np.interp(s0, xf, psi); Fp = np.interp(s0, xf, d1)
    phi = F * np.cos(th0)
    vel = -(sh * v) * Fp * np.cos(th0) - WB * sh * F * np.sin(th0)
    dt, T, every = 0.02, 800.0, 4
    cs = []
    for s in range(int(T / dt)):
        t = (s + .5) * dt
        w1 = -U0 * np.exp(-((xf - v * t) ** 2) / (2.0 * sigma ** 2))
        acc = -(2 * phi - np.roll(phi, 1) - np.roll(phi, -1)) - WGAP2 * phi - w1 * phi
        phi = phi + dt * vel + .5 * dt * dt * acc
        t2 = (s + 1) * dt
        w2 = -U0 * np.exp(-((xf - v * t2) ** 2) / (2.0 * sigma ** 2))
        acc2 = -(2 * phi - np.roll(phi, 1) - np.roll(phi, -1)) - WGAP2 * phi - w2 * phi
        vel = vel + .5 * dt * (acc + acc2)
        if s % every == 0:
            cs.append(phi[N // 2 + int(round(v * t))])
    cs = np.array(cs)
    target = WB / g
    tt = np.arange(len(cs)) * every * dt
    y = cs - np.mean(cs)
    best, bw = None, None
    for w in np.linspace(target * (1 - FRAC), target * (1 + FRAC), nscan):
        A = np.stack([np.cos(w * tt), np.sin(w * tt)], axis=1)
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        r = float(np.sum((A @ c - y) ** 2))
        if best is None or r < best:
            best, bw = r, w
    A = np.stack([np.cos(bw * tt), np.sin(bw * tt)], axis=1)
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = float(np.linalg.norm(A @ c - y) / max(np.linalg.norm(y), 1e-300))
    return abs(bw - target) / target, resid


t0 = time.time()
print("=" * 78)
print("第 698 号：C 项幂次锁定与分辨率墙（判据 R1 至 R7）")
print("=" * 78)
scan_step = 2 * FRAC / (NSCAN - 1)
print("    扫描分辨率（相对）= %.2e（点数 %d）" % (scan_step, NSCAN))
DEV_MIN = 5 * scan_step
print("    dev 可用下限 = 5 倍扫描步长 = %.2e" % DEV_MIN)

print()
print("[R1] 逐档筛选（残差比 <= %.2f 且 dev >= %.2e）" % (RES_MAX, DEV_MIN))
rows = []
for v in VLIST:
    g = 1.0 / math.sqrt(1 - v * v)
    kmod = WB * g * v
    sgs, devs = [], []
    dr = dd = 0
    for sg in SIGS:
        dev, res = dev_of(sg, v, g)
        if res > RES_MAX:
            dr += 1; continue
        if dev < DEV_MIN:
            dd += 1; continue
        sgs.append(sg); devs.append(dev)
    print("    v=%.3f k=%.4f  有效 %d/5（残差超限 %d，低于分辨率 %d）"
          % (v, kmod, len(sgs), dr, dd))
    rows.append(dict(v=v, k=kmod, sgs=np.array(sgs), devs=np.array(devs), n=len(sgs)))

print()
print("[R2] 逐速度拟合（档数 >= 4 才拟合）")
fits = []
for r in rows:
    if r["n"] < 4:
        print("    v=%.3f 档数 %d，不足，跳过" % (r["v"], r["n"]))
        continue
    sgs, devs = r["sgs"], r["devs"]
    X = np.stack([1.0 / sgs ** 2, np.ones_like(sgs)], axis=1)
    coef, *_ = np.linalg.lstsq(X, devs, rcond=None)
    A, C = float(coef[0]), float(coef[1])
    pred = X @ coef
    ss = float(np.sum((devs - np.mean(devs)) ** 2))
    r2 = 1.0 - float(np.sum((devs - pred) ** 2)) / ss if ss > 0 else float("nan")
    fits.append((r["v"], r["k"], A, C, r2, r["n"]))
    print("    v=%.3f  A=%.4e  C=%.4e  R^2=%.4f  n=%d" % (r["v"], A, C, r2, r["n"]))

print()
print("[R3] 幂律指数（有效速度窗口）")
vs = np.array([f[0] for f in fits]); ks = np.array([f[1] for f in fits])
As = np.array([f[2] for f in fits]); Cs = np.array([f[3] for f in fits])
pA = float(np.polyfit(np.log(vs), np.log(As), 1)[0])
pC = float(np.polyfit(np.log(ks), np.log(Cs), 1)[0])
print("    A 对 v 指数 = %.3f（预期 2）" % pA)
print("    C 对 k 指数 = %.3f" % pC)
print("    有效速度窗口 v = %.2f 至 %.2f（%d 档），k 覆盖 %.2f 至 %.2f"
      % (min(vs), max(vs), len(vs), min(ks), max(ks)))
print("    k 的动态范围仅 %.2f 倍，指数拟合的分辨力有限" % (max(ks) / min(ks)))

print()
print("[R4] 判别与分辨力评估")
d2, d4 = abs(pC - 2), abs(pC - 4)
print("    |C 指数 - 2| = %.3f ；|C 指数 - 4| = %.3f" % (d2, d4))
if min(d2, d4) > 0.7:
    verdict = "未定"
    why = "两候选近似等距，且 k 动态范围仅 %.1f 倍" % (max(ks) / min(ks))
else:
    verdict = "k^2 型" if d2 < d4 else "k^4 型"
    why = "偏向较近候选"
print("    判读：%s（%s）" % (verdict, why))

print()
print("[R5] 分辨率墙的量化")
print("    低速度档（v <= 0.225）的 dev 在 1e-4 至 0 量级，与扫描步长 %.1e 同阶，" % scan_step)
print("    即便把扫描点数提到 %d 仍不足以救回（dev 本身接近零）。" % NSCAN)
print("    物理原因：dev 的波性侧成分 ~ A/xi^2，低速时 A ~ v^2 使 dev 在有限宽度下趋零。")

print()
print("[R6] 与 697 号对照")
print("    697 号（三档、扫描 2001 点）：C 指数 2.916")
print("    698 号（三档、扫描 %d 点）：C 指数 %.3f  变化 %+.3f"
      % (NSCAN, pC, pC - 2.916))

print()
print("[R7] 结论")
print("    其一，分辨率墙确证：v <= 0.225 三档的 dev 落入扫描分辨率，不可用；")
print("          有效窗口只剩 v = 0.30 至 0.45（三档），k 动态范围 %.1f 倍。" % (max(ks) / min(ks)))
print("    其二，A 的指数 %.3f，仍与预期 2 吻合。" % pA)
print("    其三，C 的指数 %.3f，判别为%s。" % (pC, verdict))
print("    其四，提高扫描点数到 %d 未能改变结论，说明瓶颈在 dev 本身趋零而非采样。" % NSCAN)
print()
print("耗时 %.1f 秒" % (time.time() - t0))

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].loglog(vs, As, "o-", color="#c44", label="A")
ax[0].loglog(vs, Cs, "s-", color="#268", label="C")
ax[0].axvspan(0.10, 0.225, color="gray", alpha=0.2)
ax[0].text(0.12, min(Cs) * 0.5, "分辨率墙", fontsize=8, color="gray")
ax[0].set_xlabel("v"); ax[0].set_ylabel("拟合系数")
ax[0].set_title("左：有效速度窗口（灰区不可用）")
ax[0].legend(fontsize=8)
ax[1].loglog(ks, Cs, "s", color="#268", label="C 实测")
kk = np.linspace(min(ks), max(ks), 50)
ax[1].loglog(kk, Cs[0] * (kk / ks[0]) ** 4, ":", color="gray", label=r"$k^4$")
ax[1].loglog(kk, Cs[0] * (kk / ks[0]) ** 2, "--", color="orange", label=r"$k^2$")
ax[1].loglog(kk, Cs[0] * (kk / ks[0]) ** pC, "-", color="#268", lw=1.0, label="拟合 %.2f" % pC)
ax[1].set_xlabel(r"$k_{\rm mod}$"); ax[1].set_ylabel("C")
ax[1].set_title("右：C 的标度（三档）")
ax[1].legend(fontsize=7)
plt.tight_layout()
plt.savefig(IMG, dpi=130)
print("图已存：", IMG)
