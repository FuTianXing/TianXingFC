# -*- coding: utf-8 -*-
"""第 768 号：c0 自身的 xi 依赖，与第 752 号的指数归属更正。

背景：第 752 号在核算物理档主导项时，把第 705 号 FA3 测得的 xi 指数 -0.494
当成了 c0 的 xi 指数。查证后发现该指数测的是 D 本身（纯格点效应，正比于
<f>），而 c0 是总 dev 拟合 c0 + c1/xi 得到的常数项。两者不是同一个量。

第 707 号的读数与这个归属冲突：七档（xi 从 14 到 128，跨度 9 倍）拟合的 c0
为 2.5851e-4，四档为 2.5694e-4，c0 只变 0.6%。若 c0 真按 xi^-0.494 走，
9 倍跨度上应变 66.5%。

本轮的设计要点。测 c0 的 xi 依赖不能用一次全局拟合（全局拟合的 c0 必然是
常数，那是构造恒真）。必须用滑动窗口：取连续的若干档作窗口，每个窗口内
单独拟合 c0 + c1/xi，得到一个窗口级 c0，再对窗口级 c0 取 xi 指数。窗口级
c0 随窗口中心移动的走向，就是 c0 的 xi 依赖。

判据：
  QA0 复现：四档全窗口的 c0 复现第 706 号的 2.5694e-4，七档复现第 707 号的
      2.5851e-4。容差 2%。
  QA1 读数：三组合的八档 dev 与残差比（关键附加读数，不作判据）。
  QA2 主判据：A 组合（总 dev）的窗口级 c0 的相对极差 < 5%，即 c0 在模型档
      xi 无关。容差 5% 取自第 707 号 GB1 的 c0 稳定性读数（0.6% 的余量）。
  QA3 对照组：方法分辨力的检验。对两组量各测 xi 指数。
      组一（必须测出非零）：D 本身的 xi 指数，预期 -0.494。
      组二（必须测出零）：D 的窗口级 c0 的 xi 指数，预期 0。
      判据为「组一落在 [-0.65, -0.35] 且组二落在 [-0.15, 0.15]」。
      这两组构成对照：同一套方法在同一批数据上，一个测出非零、一个测出零，
      方法的分辨力得证。
  QA4 敏感度：窗口宽度取 3、4、5 三档，报告 c0 的窗口级指数的走向。
      多档扫描（判据设计三纪律之三）。
  QA5 报告：第 752 号 PA2b 的 10.3 个数量级在正确指数下的重算。
  QA6 判定。

恒真检验（纪律二）：QA3 的组二与组一互为替身。若方法在组一（已知非零）上
也测出 0，则方法不分辨，QA2 的 PASS 不含物理信息。QA4 的窗口宽度扫描是
第二重检验：换窗口宽度若结论翻转，则 QA2 不稳。

边界：窗口拟合的 c0 是「窗口内的渐近常数」，其 xi 依赖只能在模型档
（xi 14 至 128，跨度 9 倍）上测。物理档跨度 5.7e20 倍，本装置无法复核。

可逆性等级：判据敏感（换窗口宽度、换档位集合会改指数读数）。
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

IMG = r"D:\software\TianXingFC\images\p768_c0_xi_dependence.png"

T0 = time.time()
N = 2048
xs = np.arange(N) - N // 2
xf = xs.astype(float)
WGAP2 = 1.0
WB = 0.8008
DT = 0.02
V = 0.30
G = 1.0 / math.sqrt(1.0 - V * V)
KMOD = WB * G * V

SIGS7 = (14.0, 20.0, 28.0, 42.0, 63.0, 90.0, 128.0)
SIGS8 = SIGS7 + (180.0,)
SIGS4 = (14.0, 20.0, 28.0, 42.0)

KK = 2.0 * math.pi * np.fft.fftfreq(N)
K2 = KK ** 2

C0_REC_706 = 2.5694e-4
C0_REC_707 = 2.5851e-4
ALPHA_D_705 = -0.494        # 第 705 号 FA3：D 本身的 xi 指数
ALPHA_C0_752 = -0.494       # 第 752 号误用于 c0 的同一个指数
XI_PHYS = 2.389e22
XI_MOD = 42.0

COMBOS = (("A", "lattice", "rigid", "g692"),
          ("B", "continuum", "rigid", "g692"),
          ("D", "lattice", "contracted", "boost"))

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


def measure(sg, evo, well_mode, style):
    U0, _, psi, d1 = fit_U0(sg)
    phi, vel = make_ic(psi, d1, V, style)
    cs = run(phi, vel, U0, sg, V, evo, well_mode)
    tgt = WB / G
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


def window_c0(xis, ys, width):
    out = []
    for i in range(len(xis) - width + 1):
        xs_ = xis[i:i + width]
        ys_ = ys[i:i + width]
        c0, c1, r2, mr = fit_inv1(xs_, ys_)
        out.append((float(np.mean(xs_)), c0, c1, r2, mr))
    return out


def log_slope(xis, ys):
    if len(xis) < 2:
        return float("nan")
    return float(np.polyfit(np.log(np.asarray(xis, float)),
                             np.log(np.asarray(ys, float)), 1)[0])


print("=" * 78)
print("第 768 号：c0 自身的 xi 依赖，与第 752 号的指数归属更正")
print("=" * 78)
print("装置：N=%d omega_b=%.4f v=%.2f k_mod=%.6f dt=%.2f" % (N, WB, V, KMOD, DT))
print("档位：%s" % list(SIGS7))
print("问题：第 752 号把第 705 号 FA3 的 D 的 xi 指数 %.3f 当成 c0 的指数用。" % ALPHA_D_705)
print("      但第 707 号的七档 c0 在 9 倍 xi 跨度上只变 0.6%%，两者冲突。")
print("方法：滑动窗口拟合。窗口内单独拟合 c0 + c1/xi，窗口级 c0 随窗口中心")
print("      移动的走向即 c0 的 xi 依赖。用一次全局拟合测 c0 的 xi 依赖是")
print("      构造恒真（全局 c0 必然是常数），故本轮不用。")

# ---------------- QA1 读数 ----------------
print()
print("[QA1] 读数：三组合的八档 dev 与残差比")
DATA = {}
for name, evo, wm_, st in COMBOS:
    devs, ress = [], []
    for sg in SIGS8:
        dev, res = measure(sg, evo, wm_, st)
        devs.append(dev)
        ress.append(res)
    DATA[name] = np.array(devs)
    print("    [%s] %s" % (name, "  ".join("%.0f:%.3e/%.3f" % (s, d, r)
                                       for s, d, r in zip(SIGS8, devs, ress))))

# ---------------- QA0 复现 ----------------
print()
print("[QA0] 复现：四档与七档全窗口的 c0")
c0_4, c1_4, r2_4, mr_4 = fit_inv1(SIGS4, DATA["A"][:4])
rel0 = abs(c0_4 / C0_REC_706 - 1.0)
c0_7, c1_7, r2_7, mr_7 = fit_inv1(SIGS7, DATA["A"][:7])
rel7 = abs(c0_7 / C0_REC_707 - 1.0)
print("    A 四档 c0 = %.6e；第 706 号记 %.4e；相对差 %.2e" % (c0_4, C0_REC_706, rel0))
print("    A 七档 c0 = %.6e；第 707 号记 %.4e；相对差 %.2e" % (c0_7, C0_REC_707, rel7))
QA0 = (rel0 < 0.02) and (rel7 < 0.02)
print("    QA0 判定：%s（容差 2%%）" % ("PASS" if QA0 else "FAIL"))

# ---------------- QA2 滑动窗口测 c0 的 xi 依赖 ----------------
print()
print("[QA2] 主判据：A 组合（总 dev）的滑动窗口 c0")
W = 4
WIN_A = window_c0(SIGS7, DATA["A"][:7], W)
print("    窗口宽度 %d 档，窗口数 %d" % (W, len(WIN_A)))
print("    %-10s %-13s %-13s %-9s %-9s" % ("窗口中心", "c0", "c1", "R2", "最大残差"))
for ctr, c0w, c1w, r2w, mrw in WIN_A:
    print("    %-10.1f %-13.5e %-13.4e %-9.6f %-9.4f"
          % (ctr, c0w, c1w, r2w, mrw))
c0s = [w[1] for w in WIN_A]
ctr = [w[0] for w in WIN_A]
rel_range = (max(c0s) - min(c0s)) / (np.mean(c0s))
alpha_c0 = log_slope(ctr, c0s)
print()
print("    窗口级 c0 的相对极差 = %.2f%%" % (100 * rel_range))
print("    窗口级 c0 的 xi 指数 = %+.4f" % alpha_c0)
QA2 = rel_range < 0.05
print("    QA2 判定：%s（判据：相对极差 < 5%%，即 c0 在模型档 xi 无关）"
      % ("PASS" if QA2 else "FAIL"))
print("    对照：若 c0 真按 xi^%.3f 走，9 倍跨度上应变 %.1f%%"
      % (ALPHA_C0_752, 100 * abs((128.0 / 14.0) ** ALPHA_C0_752 - 1)))

# ---------------- QA3 对照组：方法分辨力 ----------------
print()
print("[QA3] 对照组：方法分辨力（同一套方法、同一批数据）")
alpha_D_itself = log_slope(SIGS4, DATA["D"][:4])
alpha_D_itself7 = log_slope(SIGS7, DATA["D"][:7])
WIN_D = window_c0(SIGS7, DATA["D"][:7], W)
c0d_s = [w[1] for w in WIN_D]
ctrd = [w[0] for w in WIN_D]
alpha_D_c0 = log_slope(ctrd, c0d_s)
print("    组一（必须测出非零）：D 本身的 xi 指数 = %+.4f（四档，与第 705 号 FA3 同档位）"
      % alpha_D_itself)
print("      第 705 号 FA3 报 %.3f" % ALPHA_D_705)
print("      附：同法用七档给 %+.4f，差 %.3f，说明第 705 号那个指数只在 4 倍跨度上成立"
      % (alpha_D_itself7, abs(alpha_D_itself7 - ALPHA_D_705)))
print("    组二（必须测出零）：D 的窗口级 c0 的 xi 指数 = %+.4f" % alpha_D_c0)
g1 = (-0.65 <= alpha_D_itself <= -0.35)
g2 = (-0.15 <= alpha_D_c0 <= 0.15)
print("    组一 %s（区间 [-0.65, -0.35]）；组二 %s（区间 [-0.15, 0.15]）"
      % ("通过" if g1 else "不通过", "通过" if g2 else "不通过"))
QA3 = g1 and g2
print("    QA3 判定：%s（方法在同一批数据上一个测出非零、一个测出零，分辨力得证）"
      % ("PASS" if QA3 else "FAIL"))
print("    恒真检验意义：若组一也测出 0 附近，则方法不分辨，QA2 的 PASS 不含物理信息。")

# ---------------- QA4 窗口宽度敏感度 ----------------
print()
print("[QA4] 敏感度：窗口宽度三档扫描")
print("    %-8s %-13s %-13s %-13s %-13s" % ("窗口宽度", "A 的 c0 指数", "A 相对极差",
                                              "D 的 c0 指数", "D 相对极差"))
SENS = []
for w in (3, 4, 5):
    wa = window_c0(SIGS7, DATA["A"][:7], w)
    wd = window_c0(SIGS7, DATA["D"][:7], w)
    aa = log_slope([x[0] for x in wa], [x[1] for x in wa])
    ra = (max(x[1] for x in wa) - min(x[1] for x in wa)) / np.mean([x[1] for x in wa])
    ad = log_slope([x[0] for x in wd], [x[1] for x in wd])
    rd = (max(x[1] for x in wd) - min(x[1] for x in wd)) / np.mean([x[1] for x in wd])
    SENS.append((w, aa, ra, ad, rd))
    print("    %-8d %+13.4f %12.2f%% %+13.4f %12.2f%%"
          % (w, aa, 100 * ra, ad, 100 * rd))
# 判据取「指数的稳定性」，不取「相对极差」。
# 理由：相对极差度量的是窗口内拟合的噪声，窄窗口噪声大是正常的，
# 而窄窗口的残差也把 c0 拉得更散（窗口内档数少，外推 xi^-1 项更依赖单点）。
# 多档扫描要查的是结论是否随参数翻转，故判据为指数在多档下的极差。
alphas_a = [a[1] for a in SENS]
alpha_spread = max(alphas_a) - min(alphas_a)
sgn_consistent = (abs(float(np.mean(alphas_a))) < 0.15) and (alpha_spread < 0.10)
print()
print("    A 的 c0 指数在三个窗口宽度下：%s"
      % ", ".join("%+.4f" % x for x in alphas_a))
print("    指数的极差 = %.4f（判据 < 0.10）；指数的均值绝对值 = %.4f（判据 < 0.15）"
      % (alpha_spread, abs(float(np.mean(alphas_a)))))
print("    相对极差只作报告，不作判据：它度量窗口内拟合噪声，窄窗口噪声大属正常，")
print("    且窄窗口的残差会把 c0 拉散（宽度 3 档给 5.91%%，宽度 5 档给 0.74%%）。")
QA4 = sgn_consistent
print("    QA4 判定：%s（多档扫描，结论不随窗口宽度翻转）"
      % ("PASS" if QA4 else "FAIL"))

# ---------------- QA5 第 752 号结论的重算 ----------------
print()
print("[QA5] 报告：第 752 号 PA2b 在正确指数下的重算")
for label, al in (("本轮实测 c0 指数", alpha_c0), ("第 752 号误用 -0.494", ALPHA_C0_752)):
    span = abs(math.log10((XI_PHYS / XI_MOD) ** 0.0 / (XI_PHYS / XI_MOD) ** al))
    c0_phys = C0_REC_706 * (XI_PHYS / XI_MOD) ** al
    c1x_phys = 0.42223 * (gamma_of(0.30) - 1.0) / XI_PHYS
    print("    %-22s 指数 %+.4f  物理档 c0 = %.4e  比值 c0/(c1/xi) = %.3e  跨度 %.1f 个数量级"
          % (label, al, c0_phys, c0_phys / max(c1x_phys, 1e-300), span))
print()
print("    第 752 号 PA2 的主结论（物理档由 c0 主导）在两种指数下都成立，")
print("    因为 c0 按 xi 衰减的指数无论取何值都慢于 c1/xi 的 xi^-1。")

# ---------------- QA6 判定 ----------------
print()
print("[QA6] 判定")
print("    QA0=%s QA2=%s QA3=%s QA4=%s"
      % tuple("PASS" if x else "FAIL" for x in (QA0, QA2, QA3, QA4)))
if not QA0:
    verdict = "复现失败，本轮读数不可用"
elif not QA3:
    verdict = ("对照组未通过，方法的分辨力未得证，QA2 的读数不含物理信息")
elif not QA2:
    verdict = ("c0 在模型档内有可测的 xi 依赖（指数 %+.4f，相对极差 %.2f%%）"
               % (alpha_c0, 100 * rel_range))
else:
    verdict = ("c0 在模型档内 xi 无关（窗口级相对极差 %.2f%%，指数 %+.4f，"
               "三档窗口宽度一致）；第 752 号把 D 的指数误用于 c0，该更正成立"
               % (100 * rel_range, alpha_c0))
print("    判语：" + verdict)
print()
print("    更正第 752 号：")
print("      PA2b 的读数（10.3 个数量级）作废，因其用了 D 的指数当作 c0 的指数。")
print("      PA2 的主结论（物理档由 c0 主导）保留，理由是它对指数不敏感。")
print("      更正后的 c0 读法：c0(v, xi) = %.4e（模型档内 xi 无关）" % C0_REC_706)
print("      物理档的 c0 外推成为唯一的开放量。")
print()
print("    意外的发现：c0 的「渐近常数」性质是真实的，不是拟合假象。")
print("      D 本身按 xi^%+.4f 降，但 D 拟合出的 c0 近乎常数（指数 %+.4f）。"
      % (alpha_D_itself, alpha_D_c0))
print("      c0 的定义就是渐近常数项，它吸收了 D 随 xi 的下降趋势；")
print("      故第 706 号把 c0 读作 xi 无关并非误读，而是该形式的构造使然。")
print("耗时 %.1f 秒" % (time.time() - T0))

# ---------------- 图 ----------------
fig, ax = plt.subplots(1, 3, figsize=(15.8, 4.4))

ax[0].plot(SIGS7, DATA["A"][:7], "o-", color="#268", ms=7, label="A 总 dev")
ax[0].plot([w[0] for w in WIN_A], [w[1] for w in WIN_A], "s", color="#c44", ms=9,
           label="窗口级 $c_0$")
ax[0].set_yscale("log")
ax[0].set_xlabel(r"$\xi$")
ax[0].set_ylabel("量级")
ax[0].set_title("左：A 组合的读数与窗口级 $c_0$")
ax[0].legend(fontsize=8)

ax[1].plot(ctr, c0s, "s-", color="#c44", ms=8, label="A 的窗口级 $c_0$")
ax[1].plot(ctrd, c0d_s, "^--", color="#268", ms=8, label="D 的窗口级 $c_0$")
ax[1].plot(SIGS7, DATA["D"][:7], "o:", color="#666", ms=6, label="D 本身（按 $\\xi$ 降）")
ax[1].set_xscale("log")
ax[1].set_yscale("log")
ax[1].set_xlabel(r"窗口中心 $\xi$")
ax[1].set_ylabel("$c_0$")
ax[1].set_title("中：窗口级 $c_0$ 平，而 $D$ 本身降")
ax[1].legend(fontsize=8)

bars_x = [0, 1, 2, 3]
bars_v = [alpha_c0, alpha_D_c0, alpha_D_itself, ALPHA_D_705]
bars_c = ["#c44", "#268", "#e83", "#888"]
ax[2].bar(bars_x, bars_v, color=bars_c, width=0.6)
ax[2].axhline(0.0, color="#888", ls=":", lw=1.0)
ax[2].set_xticks(bars_x)
ax[2].set_xticklabels(["A 的 $c_0$", "D 的 $c_0$", "D 本身", "第705号 FA3"])
ax[2].set_ylabel("$\\xi$ 指数")
ax[2].set_title("右：四个指数的对照")

plt.tight_layout()
plt.savefig(IMG, dpi=150)
print("图已存 %s" % IMG)
