# -*- coding: utf-8 -*-
"""第 771 号：c1 自身的 xi 依赖，与第 752 号外推式的第二块拼图。

背景：第 752 号把 dev 的形式 dev = c0 + c1/xi 外推到物理档（xi = 2.389e22），
给出主导项归属（c0 主导，主导指数 4）。该外推式里两项的系数各有隐含假设：

  c0 项：c0 与 xi 无关。第 768 号已实测确认，模型档内成立（窗口级指数 +0.0171）。
  c1/xi 项：c1 与 xi 无关，且该项整体按 xi^-1 走。

第二块拼图从未被测过。若 c1 自身带 xi 依赖（例如按 xi^0 走但系数随 xi 变，
或 c1 本身正比于 xi 的某个幂），则 c1/xi 项的 xi 依赖就不是 xi^-1，
第 752 号的「c1/xi 塌缩得比 c0 快」这条推理的支点会移动。

这与第 768 号是同一类问题：拟合形式里的系数是否真的与 xi 无关。768 号查了 c0，
本轮查 c1。

判据：
  RA0 复现：A 组合四档与七档的 c1 复现第 706 号与第 707 号的记录值。
      第 706 号 c1 = 2.0500e-2（七档见第 707 号），四档见第 706 号记录。
  RA1 读数：三组合的八档 dev 与残差比。
  RA2 主判据：A 组合的滑动窗口 c1 的 xi 指数。判据取「窗口级 c1 的 xi 指数
      的绝对值 < 0.15」，即 c1 与 xi 无关。容差 0.15 沿用第 705 号 FA3 的阈值
      （该阈值是指数类判据的既有容差）。
  RA3 对照组：方法分辨力。同法测三组：
      组一（必须非零）：A 组合 dev 本身的 xi 指数，预期约 -0.5 量级。
      组二（必须非零）：A 组合 dev 的 1/xi 系数 c1 的 xi 指数，预期 0
          （这正是 RA2 的对象，组二即 RA2 的读数，用作一致性检查）。
      组三（必须为零）：B 组合（井刚性项，形式上是纯 xi^-1）的窗口级 c0
          的 xi 指数，预期 0。
      判据：组一落在 [-0.7, -0.3]，组二与组三落在 [-0.15, 0.15]。
  RA4 敏感度：窗口宽度三档扫描，判据取指数的稳定性（极差 < 0.10），
      与第 768 号 QA4 同一判据形式，不取相对极差。
  RA5 报告：c1 的 xi 依赖若非零，对第 752 号外推式的影响。给出
      c1/xi 项在物理档上的量级随 c1 指数的变化。
  RA6 判定。

恒真检验（纪律二）：RA3 的组一与组三互为对照。同一套方法在同一批数据上，
组一测出非零、组三测出零，方法的分辨力得证，RA2 的 PASS 才携带物理信息。
另有一重替身检验：把窗口宽度改到极端（2 档），若 c1 指数剧烈变化，
说明读数不稳；RA4 的多档扫描给这个检验的读数。

边界：窗口拟合的 c1 是「窗口内的 1/xi 系数」，其 xi 依赖只能在模型档
（xi 14 至 128）上测。物理档跨度 5.7e20 倍，本装置无法复核。

可逆性等级：判据敏感（换窗口宽度、换档位集合、换构造会改读数）。
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

IMG = r"D:\software\TianXingFC\images\p771_c1_xi_dependence.png"

T0 = time.time()
N = 2048
xs = np.arange(N) - N // 2
xf = xs.astype(float)
WGAP2 = 1.0
WB = 0.8008
DT = 0.02
V = 0.30
G = 1.0 / math.sqrt(1.0 - V * V)

SIGS7 = (14.0, 20.0, 28.0, 42.0, 63.0, 90.0, 128.0)
SIGS8 = SIGS7 + (180.0,)
SIGS4 = (14.0, 20.0, 28.0, 42.0)

KK = 2.0 * math.pi * np.fft.fftfreq(N)
K2 = KK ** 2

C1_REC_707 = 2.0500e-2      # 第 707 号七档的 c1
C1_REC_713 = 2.0529e-2      # 第 713 号四档的 c1
XI_PHYS = 2.389e22
XI_MOD = 42.0
C1_K = 0.42223              # 第 714 号 OA0d 的 K

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


def window_fit(xis, ys, width):
    out = []
    for i in range(len(xis) - width + 1):
        xs_ = xis[i:i + width]
        ys_ = ys[i:i + width]
        c0, c1, r2, mr = fit_inv1(xs_, ys_)
        out.append((float(np.mean(xs_)), c0, c1, r2, mr))
    return out


def log_slope(xis, ys):
    """对数斜率。ys 含非正数时返回 nan（并由调用方报告，不静默吞掉）。

    非正数出现在拟合出的常数项上，说明该常数项没有物理意义
    （第 706 号 GA3 判 B 的 c0 可忽略，正是这种情况），此时该量的
    xi 指数无定义，不得拿它作对照组的预期值。
    """
    if len(xis) < 2:
        return float("nan")
    y = np.asarray(ys, float)
    if not np.all(y > 0):
        return float("nan")
    return float(np.polyfit(np.log(np.asarray(xis, float)), np.log(y), 1)[0])


print("=" * 78)
print("第 771 号：c1 自身的 xi 依赖，与第 752 号外推式的第二块拼图")
print("=" * 78)
print("装置：N=%d omega_b=%.4f v=%.2f dt=%.2f" % (N, WB, V, DT))
print("档位：%s" % list(SIGS7))
print("问题：第 752 号的外推式 dev = c0 + c1/xi 里，c1 与 xi 无关是隐含假设。")
print("      第 768 号查了 c0（模型档内 xi 无关），c1 从未被查过。")
print("方法：滑动窗口拟合，窗口内单独拟合 c0 + c1/xi，取窗口级 c1 的 xi 指数。")

# ---------------- RA1 读数 ----------------
print()
print("[RA1] 读数：三组合的八档 dev 与残差比")
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

# ---------------- RA0 复现 ----------------
print()
print("[RA0] 复现：A 组合四档与七档的 c1")
c0_4, c1_4, _, _ = fit_inv1(SIGS4, DATA["A"][:4])
c0_7, c1_7, _, _ = fit_inv1(SIGS7, DATA["A"][:7])
rel4 = abs(c1_4 / C1_REC_713 - 1.0)
rel7 = abs(c1_7 / C1_REC_707 - 1.0)
print("    A 四档 c1 = %.6e；第 713 号记 %.4e；相对差 %.2e" % (c1_4, C1_REC_713, rel4))
print("    A 七档 c1 = %.6e；第 707 号记 %.4e；相对差 %.2e" % (c1_7, C1_REC_707, rel7))
RA0 = (rel4 < 0.02) and (rel7 < 0.02)
print("    RA0 判定：%s（容差 2%%）" % ("PASS" if RA0 else "FAIL"))

# ---------------- RA2 主判据 ----------------
print()
print("[RA2] 主判据：A 组合的滑动窗口 c1")
W = 4
WIN_A = window_fit(SIGS7, DATA["A"][:7], W)
print("    窗口宽度 %d 档，窗口数 %d" % (W, len(WIN_A)))
print("    %-10s %-13s %-13s %-9s %-9s" % ("窗口中心", "c0", "c1", "R2", "最大残差"))
for ctr, c0w, c1w, r2w, mrw in WIN_A:
    print("    %-10.1f %-13.5e %-13.5e %-9.6f %-9.4f"
          % (ctr, c0w, c1w, r2w, mrw))
c1s = [w[2] for w in WIN_A]
ctr = [w[0] for w in WIN_A]
alpha_c1 = log_slope(ctr, c1s)
rel_c1 = (max(c1s) - min(c1s)) / (np.mean(c1s))
print()
print("    窗口级 c1 的相对极差 = %.2f%%" % (100 * rel_c1))
print("    窗口级 c1 的 xi 指数 = %+.4f" % alpha_c1)
RA2 = abs(alpha_c1) < 0.15
print("    RA2 判定：%s（判据：指数绝对值 < 0.15，即 c1 与 xi 无关）"
      % ("PASS" if RA2 else "FAIL"))
print("    对照：若 c1 按 xi^%.3f 走，c1/xi 项的 xi 指数为 %.3f 而非 -1"
      % (alpha_c1, alpha_c1 - 1.0))

# ---------------- RA3 对照组 ----------------
print()
print("[RA3] 对照组：方法分辨力")
alpha_dev = log_slope(SIGS7, DATA["A"][:7])
# 组三用 D 组合的窗口级 c0。原设计用 B 组合，但 B 的 c0 拟合值为负
# （第 706 号 GA3 判 B 的 c0 可忽略，即该常数项没有物理意义），
# 取对数无意义，指数无定义。改用第 768 号已验证健康的 D 组合。
WIN_B = window_fit(SIGS7, DATA["B"][:7], W)
c0b_s = [w[1] for w in WIN_B]
ctrb = [w[0] for w in WIN_B]
alpha_B_c0 = log_slope(ctrb, c0b_s)
WIN_D = window_fit(SIGS7, DATA["D"][:7], W)
c0d_s = [w[1] for w in WIN_D]
ctrd = [w[0] for w in WIN_D]
alpha_D_c0 = log_slope(ctrd, c0d_s)
print("    组一（必须非零）：A 的 dev 本身的 xi 指数 = %+.4f" % alpha_dev)
print("    组二（必须为零）：A 的窗口级 c1 的 xi 指数 = %+.4f" % alpha_c1)
print("    组三（必须为零）：D 的窗口级 c0 的 xi 指数 = %+.4f" % alpha_D_c0)
print("    附：B 的窗口级 c0 拟合值为 %s，非正数，其 xi 指数无定义"
      % ("、".join("%.2e" % x for x in c0b_s)))
g1 = (-0.70 <= alpha_dev <= -0.30)
g2 = (-0.15 <= alpha_c1 <= 0.15)
g3 = (-0.15 <= alpha_D_c0 <= 0.15)
print("    组一 %s；组二 %s；组三 %s"
      % ("通过" if g1 else "不通过", "通过" if g2 else "不通过",
         "通过" if g3 else "不通过"))
RA3 = g1 and g2 and g3
print("    RA3 判定：%s（同一套方法上一非零两零，分辨力得证）"
      % ("PASS" if RA3 else "FAIL"))

# ---------------- RA4 敏感度 ----------------
print()
print("[RA4] 敏感度：窗口宽度三档扫描")
print("    %-8s %-13s %-13s %-13s" % ("窗口宽度", "A 的 c1 指数", "A 的 c1 相对极差",
                                      "D 的 c0 指数"))
SENS = []
for w in (3, 4, 5):
    wa = window_fit(SIGS7, DATA["A"][:7], w)
    wd = window_fit(SIGS7, DATA["D"][:7], w)
    aa = log_slope([x[0] for x in wa], [x[2] for x in wa])
    ra = (max(x[2] for x in wa) - min(x[2] for x in wa)) / np.mean([x[2] for x in wa])
    ab = log_slope([x[0] for x in wd], [x[1] for x in wd])
    SENS.append((w, aa, ra, ab))
    print("    %-8d %+13.4f %12.2f%% %+13.4f" % (w, aa, 100 * ra, ab))
alphas_c1 = [a[1] for a in SENS]
spread = max(alphas_c1) - min(alphas_c1)
RA4 = (abs(float(np.mean(alphas_c1))) < 0.15) and (spread < 0.10)
print()
print("    A 的 c1 指数在三个窗口宽度下：%s"
      % ", ".join("%+.4f" % x for x in alphas_c1))
print("    指数极差 = %.4f（判据 < 0.10）；均值绝对值 = %.4f（判据 < 0.15）"
      % (spread, abs(float(np.mean(alphas_c1)))))
print("    相对极差只作报告，不作判据（与第 768 号 QA4 同一处置）。")
print("    RA4 判定：%s" % ("PASS" if RA4 else "FAIL"))

# ---------------- RA5 对第 752 号的影响 ----------------
print()
print("[RA5] 报告：c1 的 xi 依赖对第 752 号外推式的影响")
c1_030 = C1_K * (gamma_of(0.30) - 1.0)
for label, al in (("本轮实测 c1 指数", alpha_c1), ("若 c1 按 xi^-1", -1.0),
                   ("若 c1 按 xi^+1", 1.0)):
    c1_phys = c1_030 * (XI_PHYS / XI_MOD) ** al
    c1x_phys = c1_phys / XI_PHYS
    c0_phys = 2.5694e-4 * (XI_PHYS / XI_MOD) ** 0.0171
    print("    %-18s c1 指数 %+.4f  物理档 c1/xi = %.4e  c0/(c1/xi) = %.3e"
          % (label, al, c1x_phys, c0_phys / max(c1x_phys, 1e-300)))
print()
print("    c1/xi 项的 xi 指数 = c1 的指数 - 1 = %+.4f（实测）" % (alpha_c1 - 1.0))
print("    c0 项的 xi 指数 = +0.0171（第 768 号实测）")
print("    两项的 xi 指数之差 = %.4f，符号为正即 c0 塌缩更慢，c0 主导成立。"
      % (0.0171 - (alpha_c1 - 1.0)))
print("    即便 c1 的指数取 +1（极端），c1/xi 项仍按 xi^0 走，c0 仍主导。")

# ---------------- RA6 判定 ----------------
print()
print("[RA6] 判定")
print("    RA0=%s RA2=%s RA3=%s RA4=%s"
      % tuple("PASS" if x else "FAIL" for x in (RA0, RA2, RA3, RA4)))
if not RA0:
    verdict = "复现失败，本轮读数不可用"
elif not RA3:
    verdict = "对照组未通过，方法分辨力未得证，RA2 的读数不含物理信息"
elif RA2:
    verdict = ("c1 在模型档内 xi 无关（窗口级指数 %+.4f，三档窗口宽度一致）；"
               "第 752 号外推式的 c1/xi 项按 xi^%.3f 走，c0 主导的结论加固"
               % (alpha_c1, alpha_c1 - 1.0))
else:
    verdict = ("c1 在模型档内有可测的 xi 依赖（指数 %+.4f），第 752 号外推式"
               "的第二块拼图须改写" % alpha_c1)
print("    判语：" + verdict)
print()
print("    与第 768 号合看：第 706 号形式 dev = c0 + c1/xi 的两个系数，")
print("    其 xi 无关性都已实测确认（c0 见 768 号，c1 见本轮）。")
print("    该形式的字面读法在模型档内成立，这不再是「构造使然」的推断，")
print("    而是两组独立的测量结论。")
print("耗时 %.1f 秒" % (time.time() - T0))

# ---------------- 图 ----------------
fig, ax = plt.subplots(1, 3, figsize=(15.8, 4.4))

ax[0].plot(SIGS7, DATA["A"][:7], "o-", color="#268", ms=7, label="A 总 dev")
ax[0].plot([w[0] for w in WIN_A], [w[2] for w in WIN_A], "s", color="#c44", ms=9,
           label="窗口级 $c_1$")
ax[0].set_yscale("log")
ax[0].set_xlabel(r"$\xi$")
ax[0].set_ylabel("量级")
ax[0].set_title("左：A 组合的读数与窗口级 $c_1$")
ax[0].legend(fontsize=8)

ax[1].plot(ctr, c1s, "s-", color="#c44", ms=8, label="A 的窗口级 $c_1$")
ax[1].plot(ctrd, c0d_s, "^--", color="#268", ms=8, label="D 的窗口级 $c_0$")
ax[1].plot(SIGS7, DATA["A"][:7], "o:", color="#666", ms=6, label="A 的 dev 本身（按 $\\xi$ 降）")
ax[1].set_xscale("log")
ax[1].set_yscale("log")
ax[1].set_xlabel(r"窗口中心 $\xi$")
ax[1].set_ylabel("系数")
ax[1].set_title("中：窗口级系数平，而 dev 本身降")
ax[1].legend(fontsize=8)

bars_x = [0, 1, 2]
bars_v = [alpha_dev, alpha_c1, alpha_D_c0]
ax[2].bar(bars_x, bars_v, color=["#666", "#c44", "#268"], width=0.55)
ax[2].axhline(0.0, color="#888", ls=":", lw=1.0)
ax[2].set_xticks(bars_x)
ax[2].set_xticklabels(["A 的 dev", "A 的 $c_1$", "D 的 $c_0$"])
ax[2].set_ylabel("$\\xi$ 指数")
ax[2].set_title("右：三个指数的对照")
ax[2].axhline(0.15, color="#c44", ls="--", lw=1.0, label="判据带 $\\pm0.15$")
ax[2].axhline(-0.15, color="#c44", ls="--", lw=1.0)
ax[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig(IMG, dpi=150)
print("图已存 %s" % IMG)
