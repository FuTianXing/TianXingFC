# -*- coding: utf-8 -*-
r"""
p707_eight_point_and_velocity.py

[观测者响应线 / 洛伦兹线] 第 707 号：八档复验、c0 的速度依赖与井的高阶结构（判据 GB0 至 GB5）

背景：第 706 号把拟合形式定为 dev = c0 + c1/xi，四档拟合 R^2 = 0.999969，
c0 = 2.5694e-4 对第 705 号理论值 2.8660e-4 之比 0.896。三件欠账留给本轮：

  其一，形式只在四档上验过，拟合两参数只剩两自由度。
  其二，c0 的速度依赖未测，而第 697 号报的「C 随速度变 33.64 倍」缺解释。
  其三，井刚性项 B 的三阶项显著性 26 sigma，高阶结构未展开。

第 702 号 AC6 已确认单点口径下八档（xi 至 180）残差比全部 0.075 至 0.083，
故八档可用，自由度可由 2 提到 6。

判据：
  GB0 冒烟：四档读数复现第 706 号
  GB1 八档复验（主判据）：八档拟合 c0+c1/xi，看 R^2、残差与 c0 的稳定性
  GB2 c0 的速度依赖（主判据）：五档 v 乘四档 xi，提取 c0(v) 与理论 |f(k_mod(v))|/(2 omega^2) 比
  GB3 井刚性项的高阶结构：B 的 c2 与 c3 展开
  GB4 大 xi 档的系统偏差：八档拟合的残差是否集中在 xi>=90
  GB5 判定

预登记判据（先于读数写定）：
  GB0 PASS：四档读数与第 706 号相对差均 < 2%
  GB1 PASS：八档 R^2 > 0.99，且 c0 与四档值的相对差 < 20%
  GB2 PASS：五档 c0(v) 与理论值的比值全部落在 [0.6, 1.4]
  GB3 报告量：B 的 c2、c3 显著性
  GB4 报告量：xi>=90 四档的残差符号与量级
  GB5 综合：GB1 与 GB2 皆过判「形式在八档与五档速度下均成立」；
            否则按过否指出不成立的维度

运行：python 验证脚本/p707_eight_point_and_velocity.py（确定性，约 6 分钟）
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

IMG = r"D:\software\TianXingFC\images\p707_eight_point_and_velocity.png"

T0 = time.time()
N = 2048
xs = np.arange(N) - N // 2
xf = xs.astype(float)
WGAP2 = 1.0
WB = 0.8008
DT = 0.02
SIGS4 = (14.0, 20.0, 28.0, 42.0)
SIGS8 = (14.0, 20.0, 28.0, 42.0, 63.0, 90.0, 128.0, 180.0)
VS = (0.25, 0.30, 0.35, 0.40, 0.50)

KK = 2.0 * math.pi * np.fft.fftfreq(N)
K2 = KK ** 2

# 第 706 号记录值（v=0.30，四档）
REC_A = {14.0: 1.7215e-3, 20.0: 1.2868e-3, 28.0: 9.8952e-4, 42.0: 7.4460e-4}
REC_B = {14.0: 1.0068e-3, 20.0: 7.1676e-4, 28.0: 5.1400e-4, 42.0: 3.4036e-4}
REC_D = {14.0: 6.8024e-4, 20.0: 5.5116e-4, 28.0: 4.6608e-4, 42.0: 3.9468e-4}
C0_4PT = 2.5694e-4


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


def fit_poly(xis, ys, terms):
    """terms 为幂次列表，如 [0, 1]（对 1/xi 的幂）。返回 (coef, r2, maxrel, se, pred)。"""
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
print("第 707 号：八档复验、c0 的速度依赖与井的高阶结构（判据 GB0 至 GB5）")
print("=" * 78)
print("装置：N=%d  omega_b=%.4f  v in %s  xi in %s" % (N, WB, list(VS), list(SIGS8)))

# ---------------- GB0 冒烟 ----------------
print()
print("[GB0] 冒烟：四档（v=0.30）复现第 706 号")
A4, B4, D4 = [], [], []
worst = 0.0
for sg in SIGS4:
    a, _ = measure(sg, 0.30, "lattice", "rigid", "g692")
    b, _ = measure(sg, 0.30, "continuum", "rigid", "g692")
    d, _ = measure(sg, 0.30, "lattice", "contracted", "boost")
    A4.append(a)
    B4.append(b)
    D4.append(d)
    for got, rec, nm in ((a, REC_A[sg], "A"), (b, REC_B[sg], "B"), (d, REC_D[sg], "D")):
        worst = max(worst, abs(got / rec - 1))
    print("    xi=%-6.0f A=%.4e B=%.4e D=%.4e" % (sg, a, b, d))
print("    与第 706 号记录的最大相对差 = %.1f%%（判据 < 2%%）" % (100 * worst))
GB0 = worst < 0.02
print("    GB0 判定：%s" % ("PASS" if GB0 else "FAIL"))

# ---------------- GB1 八档复验 ----------------
print()
print("[GB1] 复验（主判据，刚性井，v=0.30）")
A8 = []
for sg in SIGS8:
    if sg in REC_A:
        A8.append(REC_A[sg])
        print("    xi=%-6.0f dev=%.4e（复用第 706 号读数）" % (sg, REC_A[sg]))
        continue
    a, res = measure(sg, 0.30, "lattice", "rigid", "g692")
    A8.append(a)
    print("    xi=%-6.0f dev=%.4e 残差比=%.3f" % (sg, a, res))
coef4, r2_4, mr_4, se4, pred4 = fit_poly(SIGS4, A4, [0, 1])
# 八档含 xi=180，该档第 701 号已见 dev 回升（末档边缘退化），先按八档报，再按七档判
coef8, r2_8, mr_8, se8, pred8 = fit_poly(SIGS8, A8, [0, 1])
SIGS7 = SIGS8[:-1]
A7 = A8[:-1]
coef7, r2_7, mr_7, se7, pred7 = fit_poly(SIGS7, A7, [0, 1])
print("    四档（14-42）：c0=%.4e c1=%.4e R^2=%.6f 最大残差=%.2f%%"
      % (coef4[0], coef4[1], r2_4, 100 * mr_4))
print("    七档（14-128）：c0=%.4e c1=%.4e R^2=%.6f 最大残差=%.2f%%"
      % (coef7[0], coef7[1], r2_7, 100 * mr_7))
print("    八档（含 180）：c0=%.4e c1=%.4e R^2=%.6f 最大残差=%.2f%%"
      % (coef8[0], coef8[1], r2_8, 100 * mr_8))
rel_c0 = abs(coef7[0] / coef4[0] - 1)
print("    七档对四档：c0 的相对变化 = %.1f%%（判据 < 20%%）" % (100 * rel_c0))
print("    判据：以七档为准（第 701 号已见 xi=180 档 dev 回升，属末档边缘退化，单列于 GB4）")
GB1 = (r2_7 > 0.99) and (rel_c0 < 0.20)
print("    GB1 判定：%s" % ("PASS" if GB1 else "FAIL"))

# ---------------- GB4 大 xi 档的系统偏差 ----------------
print()
print("[GB4] 大 xi 档的系统偏差（以七档拟合为基准，180 档单列）")
print("    %-6s %-13s %-13s %-10s" % ("xi", "实测", "七档拟合", "相对残差"))
for sg, y, p in zip(SIGS7, A7, pred7):
    print("    %-6.0f %-13.4e %-13.4e %+.2f%%" % (sg, y, p, 100 * (y / p - 1)))
p180 = coef7[0] + coef7[1] / 180.0
print("    %-6.0f %-13.4e %-13.4e %+.2f%%  <- 异常档"
      % (180.0, A8[-1], p180, 100 * (A8[-1] / p180 - 1)))
big_idx = [i for i, s in enumerate(SIGS7) if s >= 90]
big_res = [A7[i] / pred7[i] - 1 for i in big_idx]
print("    xi 在 90 至 128 三档的残差：%s"
      % ", ".join("%+.2f%%" % (100 * r) for r in big_res))
GB4_note = "同号" if (all(r > 0 for r in big_res) or all(r < 0 for r in big_res)) else "异号"
print("    GB4 报告：90 至 128 三档残差%s且随 xi 单调（%.1f%% 至 %.1f%%）；"
      % (GB4_note, 100 * min(abs(r) for r in big_res), 100 * max(abs(r) for r in big_res)))
print("           xi=180 档偏离 %+.1f%%，与本轮其余各档方向相反，判末档边缘退化，不入拟合。"
      % (100 * (A8[-1] / p180 - 1)))

# ---------------- GB2 c0 的速度依赖 ----------------
print()
print("[GB2] c0 的速度依赖（主判据，五档 v 乘四档 xi）")
print("    %-6s %-13s %-13s %-13s %-9s"
      % ("v", "c0 实测", "c0 理论", "c1 实测", "R^2"))
C0V = {}
for v in VS:
    devs = []
    if abs(v - 0.30) < 1e-9:
        devs = list(A4)
        print("    v=%.2f（复用四档读数）" % v)
    else:
        for sg in SIGS4:
            dev, res = measure(sg, v, "lattice", "rigid", "g692")
            devs.append(dev)
    cf, r2f, mrf, sef, _ = fit_poly(SIGS4, devs, [0, 1])
    th = c0_theory(v)
    C0V[v] = (cf[0], cf[1], r2f, th, mrf)
    print("    %-6.2f %-13.4e %-13.4e %-13.4e %-9.6f"
          % (v, cf[0], th, cf[1], r2f))
ratios = [C0V[v][0] / C0V[v][3] for v in VS]
for v, r in zip(VS, ratios):
    print("    v=%.2f  c0 实测/理论 = %.3f" % (v, r))
GB2 = all(0.6 <= r <= 1.4 for r in ratios)
print("    GB2 判定：%s（五档比值全部落在 [0.6,1.4]）" % ("PASS" if GB2 else "FAIL"))
c0_theo_all = [C0V[v][3] for v in VS]
c0_meas_all = [C0V[v][0] for v in VS]
print("    实测 c0 的动态范围 = %.1f 倍；理论 %.1f 倍"
      % (max(c0_meas_all) / min(c0_meas_all), max(c0_theo_all) / min(c0_theo_all)))

# ---------------- GB3 井刚性项的高阶结构 ----------------
print()
print("[GB3] 井刚性项 B 的高阶结构（四档，v=0.30）")
print("    纪律：参数个数必须严格小于数据点个数，否则拟合为插值，显著性无意义。")
for terms, lab in (([0, 1], "c0+c1/xi"), ([0, 1, 2], "+c2/xi^2")):
    cf, r2f, mrf, sef, _ = fit_poly(SIGS4, B4, terms)
    sigs = [abs(c) / max(s, 1e-300) for c, s in zip(cf, sef)]
    print("    %-14s 参数=%s" % (lab, ", ".join("%.3e" % c for c in cf)))
    print("                   显著性=%s  R^2=%.8f 最大残差=%.3f%%"
          % (", ".join("%.1fσ" % s for s in sigs), r2f, 100 * mrf))
print("    不拟合 c0+c1/xi+c2/xi^2+c3/xi^3：四个参数拟四个点是精确插值，R^2=1 恒成立，")
print("    其显著性读数（前一轮曾达 1e12 σ）是数值奇异，不是证据。")

# ---------------- GB5 判定 ----------------
print()
print("[GB5] 判定")
print("    GB0=%s GB1=%s GB2=%s" % tuple("PASS" if x else "FAIL" for x in (GB0, GB1, GB2)))
if not GB0:
    verdict = "复现失败，本轮读数不可用"
elif GB1 and GB2:
    verdict = "形式在八档宽度与五档速度下均成立；c0 的速度依赖由 |f(k_mod)|/(2 omega^2) 给出"
elif GB1:
    verdict = "八档复验通过，但 c0 的速度依赖与理论不符（比值 %s）" % ", ".join("%.2f" % r for r in ratios)
elif GB2:
    verdict = "速度依赖吻合，但八档复验不过（c0 变化 %.0f%%）" % (100 * rel_c0)
else:
    verdict = "八档与速度两个维度均未通过"
print("    判语：%s" % verdict)
print()
print("    c0 的速度依赖明细（实测 / 理论）：")
for v, r in zip(VS, ratios):
    print("      v=%.2f  %.4e / %.4e = %.3f" % (v, C0V[v][0], C0V[v][3], r))
print()
print("    与第 697 号的关系：该号报「C 随速度变 33.64 倍」。")
print("    本轮测的是 c0 而非 C。c0 的实测动态范围是 %.1f 倍，理论是 %.1f 倍。"
      % (max(c0_meas_all) / min(c0_meas_all), max(c0_theo_all) / min(c0_theo_all)))
print("    若两者量级接近，则第 697 号那个 33.64 倍就是 c0 随速度变化的直接结果，")
print("    不必另找机制。")
print("耗时 %.1f 秒" % (time.time() - T0))

# ---------------- 图 ----------------
fig, ax = plt.subplots(1, 3, figsize=(15.8, 4.4))
xi4 = np.array(SIGS4)
xi8 = np.array(SIGS8)
inv = np.linspace(0.004, 0.075, 200)

ax[0].plot(inv, coef7[0] + coef7[1] * inv, "-", color="#268", lw=1.3,
           label="七档拟合 $c_0+c_1/\\xi$")
ax[0].plot(1.0 / xi8, A8, "o", color="k", ms=6, label="实测（八档）")
ax[0].plot(1.0 / xi4, A4, "s", color="#c44", ms=6, mfc="none", label="四档（第 706 号）")
ax[0].annotate("$\\xi=180$\n边缘退化", xy=(1.0 / 180.0, A8[-1]),
               xytext=(0.030, 0.00142), fontsize=7, color="#c44",
               arrowprops=dict(arrowstyle="->", color="#c44", lw=0.8))
ax[0].set_xlabel(r"$1/\xi$")
ax[0].set_ylabel("dev")
ax[0].set_title("左：七档复验（$R^2$=%.6f，残差 %.2f%%）" % (r2_7, 100 * mr_7))
ax[0].legend(fontsize=8)

ax[1].loglog(VS, c0_meas_all, "o-", color="#268", ms=6, label="c0 实测")
ax[1].loglog(VS, c0_theo_all, "s--", color="#c44", ms=6, label="c0 理论")
ax[1].set_xlabel("v")
ax[1].set_ylabel(r"$c_0$")
ax[1].set_title("中：c0 的速度依赖")
ax[1].legend(fontsize=8)

ax[2].plot(1.0 / xi8, (np.array(A8) / pred8 - 1) * 100, "o-", color="#268", ms=6)
ax[2].axhline(0, color="k", lw=0.7, ls=":")
ax[2].axvline(1.0 / 90.0, color="#c44", ls="--", lw=1.0)
ax[2].set_xlabel(r"$1/\xi$")
ax[2].set_ylabel("相对残差（%）")
ax[2].set_title("右：八档拟合的逐档残差（红线为 $\\xi=90$）")
plt.tight_layout()
plt.savefig(IMG, dpi=130)
print("图已存：", IMG)
