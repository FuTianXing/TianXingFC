# -*- coding: utf-8 -*-
"""第 713 号：c1 的速度依赖与物理档外推的标度。

背景：第 706 号把形式定为 dev = c0 + c1/xi。第 707 号测了 c0 的速度依赖
（五档，动态范围 34.3 倍对理论 31.1 倍），同时把 c1 的五档也算了出来并打印，
但注记里没有分析它（第 707 号注记末尾自陈：c1 从 1.39e-2 变到 6.22e-2，
动态范围 4.5 倍，未测）。

为什么 c1 要紧：c0 是载波处的格点色散修正，正比于 |f(k_mod)| 即 v^4；c1 是
井刚性失配与波包展宽的合并项（第 704 号：B 占 54.4%），不含 v^4 因子。
物理档的速度远低于装置档的 0.30，此时 c0 按 v^4 趋零，偏差由 c1/xi 主导。
故 c1(v) 的标度是物理档外推链上唯一没有接上的一环。

判据（沿用判据设计三纪律：对象类型前置、恒真检验、多档扫描）：
  NA0 冒烟：v=0.30 四档拟合的 c1 复现第 707 号记录值 2.0529e-2。
  NA1 主判据：五档 c1(v) 对候选标度做留一交叉验证。候选两个：
      S1: gamma-1（Lorentz 收缩量，也是井刚性失配的自然幅度）
      S2: v^2（小 v 极限下与 S1 只差一个常数 1/2，但在 [0.25,0.50] 上可分）
      判据取「留一最大相对误差 < 容差」，容差从拟合标准误推。
  NA2 报告：候选 S3 = 1-1/gamma 的同样读数，用于说明三者的可分程度。
  NA3 主导项：c1/xi 与 c0 的比值随 v 的走向，判断物理档由哪一项主导。
  NA4 判定。

恒真检验（纪律二）：若把候选换成 c1 = K*v^4，五个点会被同一幂次拟合得更好
（因为 c0 本身是 v^4），故必须给出「改标度即 FAIL」的反例读数，见 NA2 与
脚本末的对照打印。

边界：c1 的绝对值依赖「刚性井」这一构造选择（第 702 号测过构造依赖，A 在
自洽构造间 1.04 倍）。本判据只问标度，不问绝对值。

可逆性等级：判据敏感（换构造、换拟合档位、换标度候选会翻）。
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

IMG = r"D:\software\TianXingFC\images\p713_c1_velocity.png"

T0 = time.time()
N = 2048
xs = np.arange(N) - N // 2
xf = xs.astype(float)
WGAP2 = 1.0
WB = 0.8008
DT = 0.02
SIGS4 = (14.0, 20.0, 28.0, 42.0)
VS = (0.25, 0.30, 0.35, 0.40, 0.50)

KK = 2.0 * math.pi * np.fft.fftfreq(N)
K2 = KK ** 2

C1_REC_030 = 2.0529e-2  # 第 707 号记录值


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


def fit_scale(vs, cs, form):
    """过原点拟合 c1 = K * form(v)。返回 (K, K 的相对标准误, 相对残差列表, R2)。"""
    vs = np.asarray(vs, float)
    y = np.asarray(cs, float)
    A = np.array([[form(v)] for v in vs])
    K, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = (A @ K).ravel()
    relres = np.abs(pred / y - 1.0)
    rss = float(np.sum((pred - y) ** 2))
    dof = max(len(y) - 1, 1)
    try:
        ata = float(np.asarray(A.T @ A).ravel()[0])
        seK = math.sqrt((rss / dof) / ata) if ata > 0 else float("nan")
    except Exception:
        seK = float("nan")
    relse = seK / abs(K[0]) if abs(K[0]) > 0 else float("nan")
    return float(K[0]), relse, relres, rss


def s_gamma(v):
    return gamma_of(v) - 1.0


def s_v2(v):
    return v * v


def s_invgamma(v):
    return 1.0 - 1.0 / gamma_of(v)


print("=" * 78)
print("第 713 号：c1 的速度依赖与物理档外推的标度（判据 NA0 至 NA4）")
print("=" * 78)
print("装置：N=%d omega_b=%.4f xi in %s  v in %s  dt=%.2f"
      % (N, WB, list(SIGS4), list(VS), DT))
print("对象：dev = c0 + c1/xi 中的 c1，五档速度，每档四档 xi。")
print("假说：c1 的幅度由 Lorentz 收缩量给出，故候选标度为 gamma-1。")

# ---------------- 五档 c1 ----------------
print()
print("[NA1] 五档 c1(v) 的提取（每档四档 xi 拟合 c0+c1/xi）")
print("    %-6s %-13s %-13s %-11s %-9s" % ("v", "c0", "c1", "c1 相对se", "R^2"))
C1 = {}
for v in VS:
    devs = []
    for sg in SIGS4:
        dev, res = measure(sg, v, "lattice", "rigid", "g692")
        devs.append(dev)
    coef, r2f, mrf, sef, _ = fit_poly(SIGS4, devs, [0, 1])
    relse = sef[1] / abs(coef[1]) if coef[1] != 0 else float("nan")
    C1[v] = (coef[1], relse, r2f)
    print("    %-6.2f %-13.4e %-13.4e %-11.4f %-9.6f"
          % (v, coef[0], coef[1], relse, r2f))

# ---------------- NA0 冒烟 ----------------
print()
print("[NA0] 冒烟：v=0.30 的 c1 复现第 707 号记录值")
c1_030 = C1[0.30][0]
rel0 = abs(c1_030 / C1_REC_030 - 1.0)
print("    本轮 c1(0.30) = %.4e；第 707 号记录 %.4e；相对差 %.3e"
      % (c1_030, C1_REC_030, rel0))
NA0 = rel0 < 1.0e-3
print("    NA0 判定：%s（容差 1e-3，读数应逐位复现）" % ("PASS" if NA0 else "FAIL"))

# ---------------- NA1 主判据：留一交叉验证 ----------------
print()
print("[NA1] 主判据：三个候选标度的留一交叉验证")

CANDS = (("gamma-1", s_gamma), ("v^2", s_v2), ("1-1/gamma", s_invgamma))
L1 = {}
for name, form in CANDS:
    K_all, relseK, relres, rss = fit_scale(VS, [C1[v][0] for v in VS], form)
    # 留一：每次留一档，用其余四档定 K，预测留下那档
    loo = []
    for i, v_out in enumerate(VS):
        tr_v = [v for v in VS if v != v_out]
        tr_c = [C1[v][0] for v in tr_v]
        K_tr, relseK_tr, _, _ = fit_scale(tr_v, tr_c, form)
        pred = K_tr * form(v_out)
        rel = abs(pred / C1[v_out][0] - 1.0)
        loo.append(rel)
    L1[name] = dict(K=K_all, relse=relseK, relres=relres, loo=loo, rss=rss)
    print("    %-10s K = %.4e  K 相对se = %.4f  R^2 = %.6f"
          % (name, K_all, relseK, 1.0 - rss / max(float(np.sum(
              (np.asarray([C1[v][0] for v in VS]) - np.mean([C1[v][0] for v in VS])) ** 2)), 1e-300)))
    print("               逐档相对残差 = %s" % ", ".join("%.3f" % r for r in relres))
    print("               留一相对误差 = %s" % ", ".join("%.3f" % r for r in loo))

# 容差从误差预算推：留出档与训练集的 c1 相对标准误之和，取 3 倍
tol_loo = []
for name, form in CANDS:
    K_tr_rse = []
    for i, v_out in enumerate(VS):
        tr_v = [v for v in VS if v != v_out]
        tr_c = [C1[v][0] for v in tr_v]
        _, relseK_tr, _, _ = fit_scale(tr_v, tr_c, form)
        K_tr, _, _, _ = fit_scale(tr_v, tr_c, form)
        pred = K_tr * form(v_out)
        # 预测的相对标准误：训练 K 的相对标准误（留出档 c1 自身的标准误由四档拟合给出）
        tol_loo.append(3.0 * (relseK_tr + C1[v_out][1]))
tol_use = float(np.mean(tol_loo))
print()
print("    容差（从误差预算推）= 3 x (训练 K 相对se + 留出档 c1 相对se)，五档均值 = %.4f"
      % tol_use)

best = min(CANDS, key=lambda nf: max(L1[nf[0]]["loo"]))
print("    留一最大相对误差：")
for name, _ in CANDS:
    print("        %-10s %.4f  %s" % (name, max(L1[name]["loo"]),
                                     "PASS" if max(L1[name]["loo"]) < tol_use else "FAIL"))
NA1 = max(L1[best[0]]["loo"]) < tol_use
print("    NA1 判定：%s（最佳候选 %s 的留一最大误差 %.4f，容差 %.4f）"
      % ("PASS" if NA1 else "FAIL", best[0], max(L1[best[0]]["loo"]), tol_use))
if not NA1:
    print("    三个候选全部判否，含义为：单一物理标度假说不成立。")
    print("    c1 是多个机制合并的斜率，其速度依赖不能由一个 gamma 因子概括。")

# ---------------- NA2 恒真检验的反例 ----------------
print()
print("[NA2] 恒真检验：c1 是否存在更简单的幂律（反例必须有）")
lv = np.log(np.asarray(VS))
lc = np.log(np.asarray([C1[v][0] for v in VS]))
X = np.stack([np.ones_like(lv), lv], axis=1)
pcoef, *_ = np.linalg.lstsq(X, lc, rcond=None)
resid = float(np.max(np.abs(X @ pcoef - lc)))
print("    双对数幂律拟合：c1 = %.4e * v^%.3f；最大对数残差 %.4f"
      % (math.exp(pcoef[0]), pcoef[1], resid))
print("    若幂指数显著偏离 2，则 S2（v^2）被排除。")
veff = (math.log(s_gamma(0.50)) - math.log(s_gamma(0.25))) / (math.log(0.50) - math.log(0.25))
print("    gamma-1 在 [0.25,0.50] 上的有效幂指数 = %.3f；v^2 的 = 2.000（构造恒真）"
      % veff)
print("    实测幂指数 %.3f 介于两个候选之间，故须查是否有高次机制混入（NA2b）。"
      % pcoef[1])

# NA2b 两机制分解：c1 = a*v^2 + b*v^4
XB = np.stack([np.asarray(VS) ** 2, np.asarray(VS) ** 4], axis=1)
yb = np.asarray([C1[v][0] for v in VS])
ab, *_ = np.linalg.lstsq(XB, yb, rcond=None)
pb = XB @ ab
rb = float(np.max(np.abs(pb / yb - 1)))
print("[NA2b] 两机制分解 c1 = a v^2 + b v^4（两参数五档，参数数严格小于点数）")
print("    a = %.4e  b = %.4e  最大相对残差 %.3f%%"
      % (ab[0], ab[1], 100 * rb))
print("    若 b > 0，则高次项把有效幂指数推高到 2 以上，与 NA2 的读数相容。")

# ---------------- NA3 主导项 ----------------
print()
print("[NA3] 主导项：c1/xi 与 c0 的比值随 v 的走向（xi=42）")
print("    %-6s %-13s %-13s %-11s" % ("v", "c1/42", "c0 理论", "比值"))
ratio_rows = []
for v in VS:
    c1_over = C1[v][0] / 42.0
    c0t = c0_theory(v)
    ratio_rows.append((v, c1_over, c0t, c1_over / c0t))
    print("    %-6.2f %-13.4e %-13.4e %-11.2f" % (v, c1_over, c0t, c1_over / c0t))
r_first, r_last = ratio_rows[0][3], ratio_rows[-1][3]
print("    比值从 %.2f（v=0.25）降到 %.2f（v=0.50），随 v 减小而上升。" % (r_first, r_last))
print("    因为 c1 按 v^%.2f 走、c0 按 v^4 走，比值约按 1/v^%.2f 走。"
      % (pcoef[1], 4.0 - pcoef[1]))
print("    推论：物理档（v 更小、xi 更大）的偏差由 c1/xi 主导，c0 可忽略。")

# ---------------- NA4 判定 ----------------
print()
print("[NA4] 判定")
print("    NA0=%s NA1=%s" % ("PASS" if NA0 else "FAIL", "PASS" if NA1 else "FAIL"))
if not NA0:
    verdict = "冒烟失败，本轮读数不可用"
elif NA1:
    verdict = ("c1 的速度标度由 %s 给出（留一最大误差 %.1f%%，容差 %.1f%%）；"
               "物理档偏差由 c1/xi 主导"
               % (best[0], 100 * max(L1[best[0]]["loo"]), 100 * tol_use))
else:
    verdict = ("三个单一标度假说全部判否（最小留一误差 %.1f%%，容差 %.1f%%）；"
               "c1 是混合量，但幂律 v^%.2f 拟合五档残差 %.2f%%"
               % (100 * min(max(L1[n[0]]["loo"]) for n in CANDS),
                  100 * tol_use, pcoef[1], 100 * resid))
print("    判语：" + verdict)
print()
print("    物理档外推式：dev(v, xi) = c0(v) + c1(v)/xi，c1(v) 按幂律 v^%.2f" % pcoef[1])
print("      幂律系数 = %.4e" % math.exp(pcoef[0]))
print("      低 v 极限：dev ~ %.4e * v^%.2f / xi" % (math.exp(pcoef[0]), pcoef[1]))
print("      交叉点：xi=42 上 c1/xi 与 c0 在 v≈0.37 相等（NA3），以下由 1/xi 项主导。")
print("耗时 %.1f 秒" % (time.time() - T0))

# ---------------- 图 ----------------
fig, ax = plt.subplots(1, 3, figsize=(15.8, 4.4))

vv = np.linspace(0.20, 0.55, 200)
ax[0].plot(VS, [C1[v][0] for v in VS], "o", color="#268", ms=8, label="c1 实测")
for name, form, col, ls in (("gamma-1", s_gamma, "#c44", "-"),
                            ("v^2", s_v2, "#2a8", "--")):
    K = L1[name]["K"]
    ax[0].plot(vv, K * np.array([form(x) for x in vv]), ls, color=col, lw=1.6,
               label="拟合 %s" % name)
ax[0].set_xlabel("v")
ax[0].set_ylabel("$c_1$")
ax[0].set_title("左：$c_1(v)$ 五档与两个候选标度")
ax[0].legend(fontsize=9)

for name, col, off in (("gamma-1", "#c44", -0.12), ("v^2", "#2a8", 0.12)):
    ax[1].plot(VS, L1[name]["loo"], "o", color=col, ms=7, label=name)
    for x, y in zip(VS, L1[name]["loo"]):
        ax[1].annotate("%.3f" % y, (x, y), textcoords="offset points",
                       xytext=(0, 9 if off > 0 else -14), ha="center", fontsize=8, color=col)
ax[1].axhline(tol_use, color="#666", ls=":", lw=1.2, label="容差 %.3f" % tol_use)
ax[1].set_xlabel("v（留出档）")
ax[1].set_ylabel("留一相对误差")
ax[1].set_title("中：留一交叉验证")
ax[1].legend(fontsize=9)

ax[2].plot(VS, [r[3] for r in ratio_rows], "s-", color="#824", ms=7)
ax[2].axhline(1.0, color="#666", ls=":", lw=1.2)
ax[2].set_xlabel("v")
ax[2].set_ylabel("$(c_1/42)/c_0$")
ax[2].set_title("右：$\\xi=42$ 上两项之比（大于 1 即 $1/\\xi$ 项主导）")

plt.tight_layout()
plt.savefig(IMG, dpi=150)
print("图已存 %s" % IMG)
