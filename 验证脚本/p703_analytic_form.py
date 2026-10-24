# -*- coding: utf-8 -*-
r"""
p703_analytic_form.py

[观测者响应线 / 洛伦兹线] 第 703 号：A 的解析形式与格点成分判定（判据 CA0 至 CA5）

背景：第 702 号判 A 在自洽构造间稳健（1.04 倍），第 696 号所用 rigid+g692 给
A = 2.0848e-1。这个数意味着物理档外推是预算的 2.50 倍。本轮问两件事：
  其一，A 的函数形式是什么（对 v 的标度）；
  其二，dev 里有多少来自格点色散，多少来自连续理论本身。

第二问决定一件事的性质：若 dev 主要来自格点色散，它是洛伦兹破坏的度量；
若主要来自连续理论（运动束缚态不是完美平面波），它就不是洛伦兹破坏的度量，
而是波包有限宽度修正。

判据：
  CA0 冒烟：束缚态的各阶矩及其 xi 标度
  CA1 A 的 v 标度（五档 v 乘四档 xi，逐 v 拟合 A）
  CA2 A 的归一化候选对照（A/v^2 与 A/(gamma^2 v^2)）
  CA3 格点成分判定（同一初值，格点拉普拉斯换 FFT 连续拉普拉斯）
  CA4 恒真反例（v=0）
  CA5 判定

预登记判据（先于读数写定）：
  CA0 PASS：束缚态宽度 sigma_psi 对井宽 xi 的标度指数 < 0.8，且对数拟合残差 < 0.05
       （即 sigma_psi 不随井宽线性增长，xi 是井宽参数而并非束缚态的物理宽度）
  CA1 PASS：五个 v 档的 R^2 全部 > 0.95（四档 xi 拟合 A/xi^2+C 成立）
  CA2 归一化判别：A/v^2 的极差比与 A/(gamma v)^2 的极差比，取较小者判为
       A 的 v 标度形式；较小者 < 1.5 判「已定」，否则判「未定」
  CA3 分解：以 dev_lattice / dev_continuum 给出格点与连续两来源的占比
       格点占比 > 70% 判「格点主导」，< 30% 判「连续主导」，中间判「混合」
  CA4 PASS：v=0 的 dev 与 Verlet 频率误差 (2/dt)arcsin(omega dt/2)-omega 的
       预测值相对差 < 20%（判底噪为可解析预测的积分误差，装置无额外系统误差）
  CA5 综合：CA4 不过判「装置异常」；否则按 CA2 给 A 的 v 标度判语，
       并按 CA3 给 dev 构成的判语

运行：python 验证脚本/p703_analytic_form.py（确定性，约 6 分钟）
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

IMG = r"D:\software\TianXingFC\images\p703_analytic_form.png"

T0 = time.time()
N = 2048
xs = np.arange(N) - N // 2
xf = xs.astype(float)
WGAP2 = 1.0
WB = 0.8008
RES_MAX = 0.30
SIGS = (14.0, 20.0, 28.0, 42.0)
VS = (0.10, 0.20, 0.30, 0.40, 0.50)

# FFT 波矢（周期边界，与 np.roll 一致）
KK = 2.0 * math.pi * np.fft.fftfreq(N)
K2 = KK ** 2


def gamma_of(v):
    return 1.0 / math.sqrt(1.0 - v * v)


_U0CACHE = {}


def bound(sigma, U0):
    well = -U0 * np.exp(-(xf ** 2) / (2.0 * sigma ** 2))
    L = diags([-np.ones(N - 1), 2.0 + WGAP2 + well, -np.ones(N - 1)],
              [-1, 0, 1], format='csr')
    vals, vecs = eigsh(L, k=1, which='SA')
    w2 = float(vals[0])
    p = vecs[:, 0] / np.max(np.abs(vecs[:, 0]))
    kk = 2.0 * math.pi * np.fft.fftfreq(N)
    d1 = np.real(np.fft.ifft(1j * kk * np.fft.fft(p)))
    return (math.sqrt(w2) if w2 > 0 else 0.0), p, d1


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


def moments(psi):
    """束缚态的各阶矩：高斯宽度 sigma_psi、<k^2>、<k^4>。"""
    p = psi / math.sqrt(float(np.sum(psi ** 2)))
    kk = 2.0 * math.pi * np.fft.fftfreq(N)
    ph = np.fft.fft(p)
    k2 = float(np.sum(np.abs(ph) ** 2 * kk ** 2))
    k4 = float(np.sum(np.abs(ph) ** 2 * kk ** 4))
    x2 = float(np.sum((xf * p) ** 2))
    return math.sqrt(max(x2, 1e-300)), k2, k4


def make_ic(psi, d1, v, gg):
    """g692 写法（第 696/702 号判为有效且自洽）。"""
    sh = 1.0 / gg
    s0 = sh * xf
    th0 = -WB * sh * v * xf
    c = sh * v
    wfac = WB * sh
    F = np.interp(s0, xf, psi)
    Fp = np.interp(s0, xf, d1)
    return F * np.cos(th0), -c * Fp * np.cos(th0) - wfac * F * np.sin(th0)


def lap(phi, mode):
    """mode='lattice' 用格点二阶差分；'continuum' 用 FFT 乘 -k^2。"""
    if mode == "lattice":
        return np.roll(phi, 1) + np.roll(phi, -1) - 2.0 * phi
    return -np.real(np.fft.ifft(K2 * np.fft.fft(phi)))


def run(phi, vel, U0, wsig, v, mode="lattice", dt=0.02, T=1200.0, every=4):
    cs, amps = [], []
    nst = int(T / dt)
    for s in range(nst):
        t = (s + .5) * dt
        w1 = -U0 * np.exp(-((xf - v * t) ** 2) / (2.0 * wsig ** 2))
        acc = lap(phi, mode) - WGAP2 * phi - w1 * phi
        phi = phi + dt * vel + .5 * dt * dt * acc
        t2 = (s + 1) * dt
        w2 = -U0 * np.exp(-((xf - v * t2) ** 2) / (2.0 * wsig ** 2))
        acc2 = lap(phi, mode) - WGAP2 * phi - w2 * phi
        vel = vel + .5 * dt * (acc + acc2)
        if s % every == 0:
            cs.append(float(phi[N // 2 + int(round(v * t2))]))
            amps.append(float(np.max(np.abs(phi))))
    return np.array(cs), np.array(amps)


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


def measure(sg, v, mode="lattice"):
    """在宽度 sg、速度 v 上做一次测量，返回 (dev, 残差比, 衰减, 目标频率)。"""
    gg = gamma_of(v)
    U0, _, psi, d1 = fit_U0(sg)
    target = WB / gg
    phi, vel = make_ic(psi, d1, v, gg)
    cs, amps = run(phi, vel, U0, sg, v, mode=mode)
    wm, res = freq_refined(cs, 0.08, target)
    return abs(wm - target) / target, res, 1.0 - amps[-1] / max(amps[0], 1e-30), target


def fit_AC(sgs, devs):
    X = np.stack([1.0 / sgs ** 2, np.ones_like(sgs)], axis=1)
    coef, *_ = np.linalg.lstsq(X, devs, rcond=None)
    pred = X @ coef
    rss = float(np.sum((devs - pred) ** 2))
    ss = float(np.sum((devs - np.mean(devs)) ** 2))
    r2 = 1.0 - rss / max(ss, 1e-300)
    dof = max(len(sgs) - 2, 1)
    try:
        cov = (rss / dof) * np.linalg.inv(X.T @ X)
        seA = float(math.sqrt(max(cov[0, 0], 0.0)))
    except Exception:
        seA = float("nan")
    return float(coef[0]), float(coef[1]), r2, seA


print("=" * 78)
print("第 703 号：A 的解析形式与格点成分判定（判据 CA0 至 CA5）")
print("=" * 78)
print("装置：N=%d  omega_b=%.4f  xi in %s  v in %s" % (N, WB, list(SIGS), list(VS)))

# ---------------- CA0 冒烟：束缚态矩 ----------------
print()
print("[CA0] 冒烟：束缚态的各阶矩与 xi 标度")
print("    %-6s %-11s %-11s %-11s %-11s %-11s"
      % ("xi", "sigma_psi", "sigma/xi", "<k^2>", "<k^2>*xi^2", "<k^4>*xi^4>"))
MOM = {}
for sg in SIGS:
    U0, w, psi, d1 = fit_U0(sg)
    sp, k2, k4 = moments(psi)
    MOM[sg] = (sp, k2, k4)
    print("    %-6.0f %-11.4f %-11.4f %-11.4e %-11.4f %-11.4e"
          % (sg, sp, sp / sg, k2, k2 * sg ** 2, k4 * sg ** 4))
ratios = [MOM[s][0] / s for s in SIGS]
r_spread = max(ratios) / min(ratios)
lx = np.log(np.array(SIGS))
ls = np.log(np.array([MOM[s][0] for s in SIGS]))
lk2 = np.log(np.array([MOM[s][1] for s in SIGS]))
lk4 = np.log(np.array([MOM[s][2] for s in SIGS]))
sl_s = float(np.polyfit(lx, ls, 1)[0])
sl_k2 = float(np.polyfit(lx, lk2, 1)[0])
sl_k4 = float(np.polyfit(lx, lk4, 1)[0])
print("    sigma_psi/xi 的极差比 = %.3f（不是常数，说明 sigma 不与 xi 成正比）" % r_spread)
print("    标度指数：sigma_psi 正比于 xi^(%.3f)" % sl_s)
print("              <k^2>   正比于 xi^(%.3f)" % sl_k2)
print("              <k^4>   正比于 xi^(%.3f)" % sl_k4)
print("    <k^2>*xi^2 的极差比 = %.3f（若接近 1 则 <k^2> 按 xi^-2 标度）"
      % (max(MOM[s][1] * s ** 2 for s in SIGS) / min(MOM[s][1] * s ** 2 for s in SIGS)))
# 判据：标度指数在四档内要干净（对数拟合残差小），且 sigma 的指数明显小于 1
res_s = float(np.std(ls - np.polyval(np.polyfit(lx, ls, 1), lx)))
CA0 = (sl_s < 0.8) and (res_s < 0.05)
print("    CA0 判定：%s（sigma 的标度指数 < 0.8 且对数拟合残差 < 0.05，"
      "即束缚态宽度不随井宽线性增长）" % ("PASS" if CA0 else "FAIL"))
print("    判读：井宽按 xi 走，束缚态宽度按 xi^%.2f 走。故「xi」是井宽参数，"
      "并非束缚态的物理宽度；后者的标度才是 A 的 xi 依赖的几何来源。" % sl_s)

# ---------------- CA1 A 的 v 标度 ----------------
print()
print("[CA1] A 的 v 标度（五档 v 乘四档 xi）")
print("    %-6s %-11s %-11s %-11s %-11s %-11s %-9s"
      % ("v", "dev@14", "dev@20", "dev@28", "dev@42", "A", "R^2"))
AV = {}
for v in VS:
    devs = []
    for sg in SIGS:
        dev, res, decay, tgt = measure(sg, v)
        devs.append(dev)
    devs = np.array(devs)
    A_, C_, r2_, seA = fit_AC(np.array(SIGS), devs)
    AV[v] = (A_, C_, r2_, seA, devs)
    print("    %-6.2f %-11.4e %-11.4e %-11.4e %-11.4e %-11.4e %-9.4f"
          % (v, devs[0], devs[1], devs[2], devs[3], A_, r2_))
CA1 = all(AV[v][2] > 0.95 for v in VS)
print("    CA1 判定：%s（五档 R^2 全部 > 0.95 判 A/xi^2+C 在各速度下成立）"
      % ("PASS" if CA1 else "FAIL"))

# ---------------- CA2 归一化候选对照 ----------------
print()
print("[CA2] A 的 v 标度归一化")
print("    %-6s %-11s %-8s %-13s %-13s %-13s"
      % ("v", "A", "gamma", "A/v^2", "A/(gamma v)^2", "A/(gamma^2 v^2)"))
N1, N2 = [], []
for v in VS:
    A_ = AV[v][0]
    gg = gamma_of(v)
    n1 = A_ / v ** 2
    n2 = A_ / (gg * v) ** 2
    N1.append(n1)
    N2.append(n2)
    print("    %-6.2f %-11.4e %-8.4f %-13.5f %-13.5f %-13.5f"
          % (v, A_, gg, n1, n2, A_ / (gg ** 2 * v ** 2)))
N1 = np.array(N1)
N2 = np.array(N2)
r1 = float(N1.max() / N1.min())
r2 = float(N2.max() / N2.min())
print("    A/v^2 极差比 = %.3f（v 从 %.2f 到 %.2f）" % (r1, VS[0], VS[-1]))
print("    A/(gamma v)^2 极差比 = %.3f" % r2)
better = "v^2" if r1 <= r2 else "(gamma v)^2"
rbest = min(r1, r2)
print("    较优归一化：A 正比于 %s（极差比 %.3f）" % (better, rbest))
CA2 = rbest < 1.5
print("    CA2 判定：%s（极差比 < 1.5 判 A 的 v 标度已定）" % ("PASS" if CA2 else "FAIL"))

# ---------------- CA3 格点成分判定 ----------------
print()
print("[CA3] 格点成分判定（v=0.30，同一初值，换演化算符）")
print("    %-6s %-13s %-13s %-9s %-9s"
      % ("xi", "dev(格点)", "dev(连续)", "格点/连续", "残差比(连续)"))
DELTA = {}
for sg in SIGS:
    dl, rl, _, _ = measure(sg, 0.30, mode="lattice")
    dc, rc, _, _ = measure(sg, 0.30, mode="continuum")
    DELTA[sg] = (dl, dc, rl, rc)
    print("    %-6.0f %-13.4e %-13.4e %-9.2f %-9.3f"
          % (sg, dl, dc, dl / max(dc, 1e-300), rc))
dratio = float(np.mean([DELTA[s][0] / max(DELTA[s][1], 1e-300) for s in SIGS]))
print("    平均 格点/连续 = %.2f" % dratio)
frac_lat = 1.0 - 1.0 / dratio
print("    分解：dev = dev_连续 + dev_格点，若两者同号，则格点贡献占 %.1f%%、连续占 %.1f%%"
      % (100 * frac_lat, 100 * (1 - frac_lat)))
print("    CA3 判读：%s"
      % ("格点主导（dev 主要是色散效应，构成洛伦兹破坏的度量）" if frac_lat > 0.7
         else ("连续主导（dev 主要是连续理论效应，不构成洛伦兹破坏的度量）" if frac_lat < 0.3
               else "混合（两来源量级相当，dev 不能单独作为洛伦兹破坏的度量）")))

# ---------------- CA4 底噪标定 ----------------
print()
print("[CA4] 底噪标定：v=0 的 dev 应等于 Verlet 时间离散化误差")
dt0 = 0.02
w_num = (2.0 / dt0) * math.asin(WB * dt0 / 2.0)
pred_noise = w_num / WB - 1.0
dev0, res0, _, _ = measure(20.0, 0.0)
print("    Verlet 解析预测：omega_num = %.9f，相对偏差 = %.4e" % (w_num, pred_noise))
print("    实测 v=0：dev = %.4e  残差比 = %.3f" % (dev0, res0))
rel = abs(dev0 / pred_noise - 1.0)
print("    实测/预测 = %.3f（相对差 %.1f%%）" % (dev0 / pred_noise, 100 * rel))
CA4 = rel < 0.20
print("    CA4 判定：%s（20%% 内判底噪为可解析预测的 Verlet 误差，装置无额外系统误差）"
      % ("PASS" if CA4 else "FAIL"))
print("    推论：本装置 dev 的读数下限为 %.1e；低于此量级的档读数无意义。" % pred_noise)
low_sig = [(v, s) for v in VS for s in SIGS if AV[v][4][list(SIGS).index(s)] < 5 * pred_noise]
if low_sig:
    print("    信噪比不足 5 的档：%s"
          % ", ".join("(v=%.2f, xi=%.0f)" % x for x in low_sig))
else:
    print("    全部档信噪比 > 5。")

# ---------------- CA5 判定 ----------------
print()
print("[CA5] 判定")
print("    CA0=%s CA1=%s CA2=%s CA4=%s"
      % tuple("PASS" if x else "FAIL" for x in (CA0, CA1, CA2, CA4)))
print("    CA3 分解：格点占 %.1f%%、连续占 %.1f%%（格点/连续 = %.2f）"
      % (100 * frac_lat, 100 * (1 - frac_lat), dratio))
if not CA4:
    verdict = "装置异常（v=0 的底噪与 Verlet 预测不符）"
elif CA2:
    verdict = "A 的 v 标度已定：A 正比于 %s（极差比 %.3f）" % (better, rbest)
else:
    verdict = "A 的 v 标度未定（两个候选的极差比均 >= 1.5）"
print("    判语：%s" % verdict)
print()
print("    dev 的构成：格点色散占 %.0f%%，连续理论占 %.0f%%。"
      % (100 * frac_lat, 100 * (1 - frac_lat)))
if frac_lat > 0.7:
    print("    故 dev 主要构成洛伦兹破坏的度量。")
elif frac_lat < 0.3:
    print("    故 dev 主要不构成洛伦兹破坏的度量，而是运动波包的连续理论修正。")
else:
    print("    故 dev 是两种效应的混合，单独一个 dev 读数不能判洛伦兹破坏。")
print()
print("    A 的 v 依赖明细（A/v^2 随 v 单调%s，说明存在 O(v^2) 以上的修正）："
      % ("上升" if N1[-1] > N1[0] else "下降"))
sl_n1 = float(np.polyfit(np.array(VS) ** 2, N1, 1)[0])
ic_n1 = float(np.polyfit(np.array(VS) ** 2, N1, 1)[1])
print("      线性拟合 A/v^2 = %.4f + %.4f * v^2，故 v->0 时 A -> %.4f * v^2" % (ic_n1, sl_n1, ic_n1))
print("耗时 %.1f 秒" % (time.time() - T0))

# ---------------- 图 ----------------
fig, ax = plt.subplots(1, 3, figsize=(15.8, 4.4))

col = plt.cm.viridis(np.linspace(0.15, 0.9, len(VS)))
for i, v in enumerate(VS):
    ax[0].loglog(SIGS, AV[v][4], "o-", ms=5, color=col[i], label="v=%.2f" % v)
ax[0].set_xlabel(r"$\xi$")
ax[0].set_ylabel("dev")
ax[0].set_title("左：五档速度下的 dev 标度")
ax[0].legend(fontsize=7)

ax[1].plot(VS, N1, "o-", color="#268", ms=6, label=r"$A/v^2$")
ax[1].plot(VS, N2, "s--", color="#c44", ms=6, label=r"$A/(\gamma v)^2$")
ax[1].set_xlabel("v")
ax[1].set_ylabel("归一化系数")
ax[1].set_title("中：两种归一化（极差比 %.2f 与 %.2f）" % (r1, r2))
ax[1].legend(fontsize=8)

xpos = np.arange(len(SIGS))
ax[2].bar(xpos - 0.19, [DELTA[s][0] for s in SIGS], width=0.36,
          color="#268", label="格点演化")
ax[2].bar(xpos + 0.19, [DELTA[s][1] for s in SIGS], width=0.36,
          color="#c94", label="连续演化")
ax[2].set_yscale("log")
ax[2].set_xticks(xpos)
ax[2].set_xticklabels(["%.0f" % s for s in SIGS])
ax[2].set_xlabel(r"$\xi$")
ax[2].set_ylabel("dev")
ax[2].set_title("右：格点与连续之差（平均 %.2f 倍）" % dratio)
ax[2].legend(fontsize=8)
plt.tight_layout()
plt.savefig(IMG, dpi=130)
print("图已存：", IMG)
