# -*- coding: utf-8 -*-
r"""
minimal_verification_correlation_field_analytic.py

二-1 第四段（收口）：场层相关函数系数的解析式与 xi_f 窗口漂移的定量预测。
确定性，随机固定种子 20260919。

模型：噪声取 AR(1)（rho(m) = q^|m|，q = 1 - 2p，单位方差），场层取漏积分响应
    x_{j+1} = alpha x_j + beta n_{j+1},  alpha = 1 - kappa,  beta = sqrt(2 kappa).

解析：把响应写成对噪声的双重几何和，逐项对几何级数求和得
    C(r) = a q^r + b alpha^r,
    a = -beta^2 q / ((alpha - q)(1 - alpha q)),
    b = beta^2 alpha (1 - q^2) / ((alpha - q)(1 - alpha^2)(1 - alpha q)).
两系数恒反号，且缓慢一方取正号（q 慢则 a 大于零，alpha 慢则 b 大于零）。
自洽校验：a + b = beta^2 (1 + alpha q) / ((1 - alpha^2)(1 - alpha q)) = Var(x)。

判据与期望：
  F1 系数解析式：拟合值与解析式的相对偏差 <= 2%（两档 p）
  F2 精确双重和：解析式与截断双重和的相对残差 <= 1e-6（r 至 40）
  F3 反号与慢者取正：四档 (p, kappa) 组合的拟合符号与解析符号逐一相同
  F4 窗口漂移的定量预测：三档窗口下预测 xi_f 与实测 xi_f 的相对偏差 <= 3%
  F5 渐近区在窗外：解析形式的局部斜率在 r = 100 至 120 上等于 ln(max(q, alpha))（<= 1%），
     而窗 25 至 40 上仍显著偏离（报出偏离）
  F6 实测斜率在统计容差内：窗 25 至 40 上实测与解析预测之差 <= 3 倍预估标准差
  F7 读数地板：报出 r_floor（解析值降到三倍统计尺度以下的距离），并核验窗 1 至 24 在可用区内
  F8 系数和与方差：a + b 对 C(0) 相对偏差 <= 1%
  F6 接口登记（对接第 174 条方差平台，不作判据）
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\correlation_field_analytic.png"
SEED = 20260919
L = 6000
BURN = 1500
NS = 3000
R_MAX = 40
rows = []


def report(tag, ok, expect, actual):
    rows.append((tag, ok, expect, actual))
    print("[%s] %s" % ("通过" if ok else "未过", tag))
    print("      期望 " + expect)
    print("      实测 " + actual)


def ar1(L, ns, q, rng):
    n = np.zeros((ns, L))
    n[:, 0] = rng.normal(size=ns)
    eps = rng.normal(scale=np.sqrt(1.0 - q ** 2), size=(ns, L))
    for j in range(1, L):
        n[:, j] = q * n[:, j - 1] + eps[:, j]
    return n


def ou_field(n, kap):
    alpha = 1.0 - kap
    beta = np.sqrt(2.0 * kap)
    x = np.zeros_like(n)
    for j in range(1, n.shape[1]):
        x[:, j] = alpha * x[:, j - 1] + beta * n[:, j]
    return x


def corr(x, r):
    a = x if r == 0 else x[:, :-r]
    b = x[:, r:]
    return float(np.mean(a * b))


def coef(kap, q):
    alpha = 1.0 - kap
    beta2 = 2.0 * kap
    den = (alpha - q) * (1.0 - alpha * q)
    a = -beta2 * q / den
    b = beta2 * alpha * (1.0 - q ** 2) / (den * (1.0 - alpha ** 2))
    return alpha, a, b


def var_closed(kap, q):
    alpha = 1.0 - kap
    beta2 = 2.0 * kap
    return beta2 * (1.0 + alpha * q) / ((1.0 - alpha ** 2) * (1.0 - alpha * q))


def exact_C(kap, q, rs, K=600):
    alpha = 1.0 - kap
    beta2 = 2.0 * kap
    k = np.arange(K)
    W = alpha ** (k[:, None] + k[None, :])
    out = []
    for r in rs:
        out.append(beta2 * float((W * q ** np.abs(r + k[None, :] - k[:, None])).sum()))
    return np.array(out)


rng = np.random.default_rng(SEED)
rs = np.arange(0, R_MAX + 1)
sims = {}
for p0 in (0.05, 0.10):
    q = 1.0 - 2.0 * p0
    n = ar1(L, NS, q, rng)
    x = ou_field(n, 0.25)[:, BURN:]
    sims[p0] = (q, np.array([corr(x, int(r)) for r in rs]))

# ---------- F1 系数解析式 ----------
f1 = []
for p0, (q, C) in sims.items():
    alpha, a_c, b_c = coef(0.25, q)
    A = np.vstack([q ** rs, alpha ** rs]).T
    sol, *_ = np.linalg.lstsq(A, C, rcond=None)
    d1 = abs(sol[0] - a_c) / abs(a_c)
    d2 = abs(sol[1] - b_c) / abs(b_c)
    f1.append((p0, q, a_c, sol[0], b_c, sol[1], max(d1, d2)))
report("F1 系数解析式 a、b", all(d <= 0.02 for *_, d in f1),
       "两档 p 上拟合系数与解析式的相对偏差 <= 2%",
       "；".join("p=%.2f: a 解析 %.4f 对拟合 %.4f，b 解析 %.4f 对拟合 %.4f（最大相对 %.2e）"
                % (p0, ac, af, bc, bf, d) for p0, q, ac, af, bc, bf, d in f1))

# ---------- F2 精确双重和 ----------
f2 = []
for p0, (q, C) in sims.items():
    alpha, a_c, b_c = coef(0.25, q)
    C_ex = exact_C(0.25, q, rs)
    C_cl = a_c * q ** rs + b_c * alpha ** rs
    rel = float(np.max(np.abs(C_ex - C_cl)) / C_ex[0])
    f2.append((p0, rel))
report("F2 解析式对精确双重和", all(r <= 1e-6 for _, r in f2),
       "解析式与截断双重和（K = 600）的相对残差 <= 1e-6",
       "；".join("p=%.2f: 相对残差 %.2e" % (p0, r) for p0, r in f2))

# ---------- F3 反号与慢者取正 ----------
f3 = []
for kap in (0.25, 0.60, 0.05):
    for p0 in ((0.05, 0.10) if kap == 0.25 else (0.05,)):
        q = 1.0 - 2.0 * p0
        alpha, a_c, b_c = coef(kap, q)
        A = np.vstack([q ** rs, alpha ** rs]).T
        if kap == 0.25:
            C = sims[p0][1]
        else:
            n = ar1(L, NS, q, rng)
            x = ou_field(n, kap)[:, BURN:]
            C = np.array([corr(x, int(r)) for r in rs])
        sol, *_ = np.linalg.lstsq(A, C, rcond=None)
        slow_is_q = q > alpha
        pos_is_q = a_c > 0
        fit_pos_is_q = sol[0] > 0
        f3.append((kap, p0, q, alpha, a_c, b_c, sol[0], sol[1],
                   (a_c * b_c < 0), (pos_is_q == slow_is_q), (fit_pos_is_q == pos_is_q)))
report("F3 反号与慢者取正", all(r[8] and r[9] and r[10] for r in f3),
       "四档组合上两系数反号、正号落在缓慢一方、拟合符号与解析一致",
       "；".join("kappa=%.2f p=%.2f: 解析 a %.3f b %.3f，拟合 a %.3f b %.3f，慢者 %s"
                % (k, p0, ac, bc, af, bf, "q" if q > al else "alpha")
                for k, p0, q, al, ac, bc, af, bf, _, _, _ in f3))

# ---------- F4 窗口漂移的定量预测 ----------
wins = [(1, 6), (1, 12), (1, 24)]
f4 = []
for p0, (q, C) in sims.items():
    alpha, a_c, b_c = coef(0.25, q)
    C_cl = a_c * q ** rs + b_c * alpha ** rs
    for (r0, r1) in wins:
        rf = np.arange(r0, r1 + 1)
        xi_meas = -1.0 / np.polyfit(rf, np.log(np.abs(C[r0:r1 + 1])), 1)[0]
        xi_pred = -1.0 / np.polyfit(rf, np.log(np.abs(C_cl[r0:r1 + 1])), 1)[0]
        f4.append((p0, (r0, r1), float(xi_meas), float(xi_pred),
                   abs(xi_meas - xi_pred) / xi_pred))
report("F4 窗口漂移的定量预测", all(d <= 0.03 for *_, d in f4),
       "三档窗口上预测 xi_f 与实测 xi_f 的相对偏差 <= 3%",
       "；".join("p=%.2f 窗 %d-%d: 实测 %.4f 对预测 %.4f（相对 %.2e）"
                % (p0, r0, r1, xm, xp, d) for p0, (r0, r1), xm, xp, d in f4))

# ---------- F5 渐近区在窗外 ----------
f5 = []
for p0, (q, C) in sims.items():
    alpha, a_c, b_c = coef(0.25, q)
    slow = max(q, alpha)
    rf_far = np.arange(100, 121)
    C_far = a_c * q ** rf_far + b_c * alpha ** rf_far
    sl_far = float(np.polyfit(rf_far, np.log(C_far), 1)[0])
    rel_far = abs(sl_far - np.log(slow)) / abs(np.log(slow))
    rf_win = np.arange(25, R_MAX + 1)
    C_win = a_c * q ** rf_win + b_c * alpha ** rf_win
    sl_win = float(np.polyfit(rf_win, np.log(C_win), 1)[0])
    dev_win = abs(sl_win - np.log(slow)) / abs(np.log(slow))
    f5.append((p0, sl_far, float(np.log(slow)), rel_far, dev_win))
report("F5 渐近区在窗外", all(rf <= 0.01 for *_, rf, dw in f5),
       "解析形式在 r = 100 至 120 上的局部斜率等于 ln(max(q, alpha))（<= 1%），而窗 25 至 40 上仍偏离",
       "；".join("p=%.2f: 远端斜率 %.5f 对 %.5f（相对 %.2e），窗内仍偏离 %.2f%%"
                % (p0, sf, l, rf, 100.0 * dw) for p0, sf, l, rf, dw in f5))


def c_vals(p0, lags):
    q, C = sims[p0]
    alpha, a_c, b_c = coef(0.25, q)
    lags = np.asarray(lags, dtype=float)
    return a_c * q ** lags + b_c * alpha ** lags


M_LAG = 600
_IDX = np.arange(-M_LAG, M_LAG + 1)


def _S(p0, d):
    cf = c_vals(p0, np.abs(_IDX))
    if abs(d) > 2 * M_LAG:
        return 0.0
    lo = max(0, -int(d))
    hi = min(len(cf), len(cf) - int(d))
    return float(np.dot(cf[lo:hi], cf[lo + int(d):hi + int(d)]))


def sigma_cov(p0, r, s):
    n_pairs = NS * (L - BURN - max(r, s))
    return (_S(p0, s - r) + _S(p0, r + s)) / n_pairs


def sigma_scale(p0, r):
    return float(np.sqrt(sigma_cov(p0, int(r), int(r))))


# ---------- F6 实测斜率在统计容差内 ----------
f6 = []
for p0, (q, C) in sims.items():
    alpha, a_c, b_c = coef(0.25, q)
    rf = np.arange(25, R_MAX + 1)
    C_cl = c_vals(p0, rf)
    sl_meas = float(np.polyfit(rf, np.log(np.abs(C[25:])), 1)[0])
    sl_pred = float(np.polyfit(rf, np.log(C_cl), 1)[0])
    x = rf - rf.mean()
    W = x / np.sum(x ** 2)
    cov = np.array([[sigma_cov(p0, int(a_), int(b_)) for b_ in rf] for a_ in rf])
    rel = 1.0 / C_cl
    err = float(np.sqrt(np.einsum("i,j,ij,i,j->", W, W, cov, rel, rel)))
    f6.append((p0, sl_meas, sl_pred, err, abs(sl_meas - sl_pred) / err))
report("F6 实测斜率在统计容差内", all(z <= 3.0 for *_, z in f6),
       "窗 25 至 40 上实测对数斜率与解析预测之差不超过三倍预估标准差（Bartlett 协方差）",
       "；".join("p=%.2f: 实测 %.5f 对预测 %.5f（标准差 %.5f，偏差 %.2f 倍）"
                % (p0, sm, sp, er, z) for p0, sm, sp, er, z in f6))

# ---------- F7 读数地板 ----------
f7 = []
for p0, (q, C) in sims.items():
    alpha, a_c, b_c = coef(0.25, q)
    floor = None
    for r in rs:
        if (a_c * q ** int(r) + b_c * alpha ** int(r)) > 3.0 * sigma_scale(p0, int(r)):
            floor = int(r)
    f7.append((p0, floor, 24))
report("F7 读数地板", all(fl is not None and fl >= 24 for fl, _, _ in
                          [(x[1], x[0], x[2]) for x in f7]),
       "解析读数降到三倍统计尺度以下的距离 r_floor 不小于 24（三档窗口处在可用区内）",
       "；".join("p=%.2f: r_floor = %d" % (p0, fl) for p0, fl, _ in f7))

# ---------- F8 系数和与方差 ----------
f8 = []
for p0, (q, C) in sims.items():
    alpha, a_c, b_c = coef(0.25, q)
    f8.append((p0, a_c + b_c, C[0], abs((a_c + b_c) - C[0]) / C[0]))
report("F8 系数和与方差", all(rv <= 0.01 for *_, rv in f8),
       "a + b 与实测 C(0) 一致（<= 1%）",
       "；".join("p=%.2f: a+b %.4f 对 C(0) %.4f（相对 %.2e）" % (p0, ab, c0, rv) for p0, ab, c0, rv in f8))

print("[登记] F9：系数解析式与窗口漂移公式供 P4-16 与 二-1 首轮方差平台对接，本轮只登记，不作判据")

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
for p0, (q, C) in sims.items():
    alpha, a_c, b_c = coef(0.25, q)
    ax[0].semilogy(rs, C / C[0], "o", label="p=%.2f 实测" % p0)
    ax[0].semilogy(rs, (a_c * q ** rs + b_c * alpha ** rs) / C[0], "k--")
    ax[0].semilogy(rs, [3.0 * sigma_scale(p0, int(r)) / C[0] for r in rs], ":", label="三倍统计尺度 p=%.2f" % p0)
ax[0].set_xlabel("格距 r")
ax[0].set_ylabel("C(r) / C(0)")
ax[0].set_title("场层相关函数与系数解析式（虚线）")
ax[0].legend(fontsize=8)
lab = ["1-6", "1-12", "1-24"]
for p0, (q, C) in sims.items():
    alpha, a_c, b_c = coef(0.25, q)
    C_cl = a_c * q ** rs + b_c * alpha ** rs
    xm, xp = [], []
    for (r0, r1) in wins:
        rf = np.arange(r0, r1 + 1)
        xm.append(-1.0 / np.polyfit(rf, np.log(np.abs(C[r0:r1 + 1])), 1)[0])
        xp.append(-1.0 / np.polyfit(rf, np.log(np.abs(C_cl[r0:r1 + 1])), 1)[0])
    ax[1].plot(lab, xm, "o-", label="p=%.2f 实测" % p0)
    ax[1].plot(lab, xp, "k--", label="p=%.2f 预测" % p0)
ax[1].set_xlabel("拟合窗口")
ax[1].set_ylabel("xi_f")
ax[1].set_title("xi_f 的窗口漂移：实测与解析预测")
ax[1].legend(fontsize=7)
plt.tight_layout()
plt.savefig(IMG, dpi=150)
plt.close(fig)

bad = [r for r in rows if not r[1]]
print("---- 判据 %d 项，通过 %d 项；图存 %s ----" % (len(rows), len(rows) - len(bad), IMG))
sys.exit(1 if bad else 0)
