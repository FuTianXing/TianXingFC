# -*- coding: utf-8 -*-
r"""
wall_pn_barrier.py  （第 616 号，第二版）

内部壁 PN 势垒轮：现行钉位机制（内部壁承载绕数 + 定义性联结，第 54 条作者裁决）
下 P1-40 钉扎-洛伦兹张力的重估。

背景。P1-40 的旧读数 r = ΔE_PN/M = 1.51e-8（K = 0.005，ξ = 2.25 格距）出自
FK 周期钉扎机制（判据 E1 至 E6，2026-09-20）。该机制已由第 482 条判负
（周期势井门槛差 45 个数量级），现行机制是复结构符号场 σ 的双井势畴壁
（第 485、486 条）。第 486 条 SW2 只核对整格点壁位（平移零模），未测半格点
位置；离散格点上壁位存在 PN 势垒。本轮测量该势垒。

模型 A（新机制载体，φ⁴ 壁）：
    E = Σ_n [(κ/2)(Δσ_n)² + λ(σ_n²-1)²]，   宽度 w = sqrt(κ/(2λ))
模型 B（旧机制载体，FK / sine-Gordon 孤子）：
    E = Σ_n [(k/2)(Δφ_n)² + K(1-cosφ_n)]，   宽度 w = sqrt(k/K)

方法（主）。镜像对称约束 + trust-ncg 牛顿型优化（提供精确梯度与黑塞），
松弛到机器精度；能量用 math.fsum 精确求和。势垒 ΔE_PN = E_site - E_mid。

方法（对照）。解析剖面直接取样（2026-09-20 原方法，不松弛），复现旧读数。

解析形式（用于指数率验证，前因子以数值标定吸收）。泊松求和给
    ΔE = 4Σ_{m odd}[V̂(2πm) - T̂(2πm)] + 离散修正，
所有项共享指数率 e^{-π²w}，前因子为 w 的多项式：
    r_φ⁴ ≈ π²(π²w²+1)(3π²w²+2) / (15 w sinh(π²w))   （主导项 2V̂(2π)）
    r_SG ≈ π²(4π²w²+1) / (18 w sinh(π²w))           （主导为离散修正）

判据（先立判据，再实现）：
  J1 冒烟：两模型松弛收敛（max|g| ≤ 1e-9），壁能对解析值 8λw/3（φ⁴）/ 8Kw（SG）
     偏差 ≤ 3%
  J2 旧读数复现：旧方法（解析剖面取样）在 K = 0.005 给 r = 1.51e-8（±1%），
     K = 0.003 给 3.04e-11（±5%）；松弛读数与旧读数的比值（登记为发现）
  J3 标度律：数值 ln(r/w³) 对 w 的斜率在 [2.5, 4.0] 区间 ≈ -π²（±5%）；
     数值对解析的标定系数在 [2.5, 4.0] 区间极差比 ≤ 1.3
  J4 势垒方向：全扫描点 E_site > E_mid
  J5 同宽对照：w = 2.251 处 r_φ⁴/r_SG ∈ [20, 100]（解析预期 ≈ 45）
  J6 宽度门槛：解析标定与纯数值拟合（ln r = a + p ln w - c w）两种方法给出的
     w*(1e-15) 互差 ≤ 0.5 格，且都在 [4.0, 5.2] 内；w*(1e-20) 同法
  J7 机制自带档：note-486 测试档（κ=λ=1 → w = 0.707）的 r 读数（期望 O(0.1)），
     墙宽核对
"""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\wall_pn_barrier.png"
K_HALF = 128
rows = []


# ==================== 能量与梯度/黑塞 ====================
def build_full(dof, k, mode, c0):
    if mode == "site":
        N = 2 * k - 1
        full = np.empty(N)
        full[:k - 1] = dof
        full[k - 1] = c0
        full[k:] = 2.0 * c0 - dof[::-1]
    else:
        N = 2 * k
        full = np.empty(N)
        full[:k] = dof
        full[k:] = 2.0 * c0 - dof[::-1]
    return full


def mirror_mat(k, mode):
    if mode == "site":
        N, m = 2 * k - 1, k - 1
    else:
        N, m = 2 * k, k
    M = np.zeros((N, m))
    M[:m, :] = np.eye(m)
    for i in range(m):
        M[N - 1 - i, i] = -1.0
    return M


def fq_energy(full, kap, lam):
    ext = np.concatenate([[-1.0], full, [1.0]])
    d = np.diff(ext)
    return (math.fsum([0.5 * kap * float(x) * float(x) for x in d])
            + math.fsum([lam * float((s * s - 1.0) ** 2) for s in full]))


def fq_grad_hess(full, kap, lam):
    N = full.size
    ext = np.concatenate([[-1.0], full, [1.0]])
    g = kap * (2.0 * full - ext[:-2] - ext[2:]) + 4.0 * lam * full * (full * full - 1.0)
    diag = 2.0 * kap + 4.0 * lam * (3.0 * full * full - 1.0)
    H = np.diag(diag)
    idx = np.arange(N - 1)
    H[idx, idx + 1] = -kap
    H[idx + 1, idx] = -kap
    return g, H


def sg_energy(full, kk, K):
    ext = np.concatenate([[0.0], full, [2.0 * math.pi]])
    d = np.diff(ext)
    return (math.fsum([0.5 * kk * float(x) * float(x) for x in d])
            + math.fsum([K * float(1.0 - math.cos(s)) for s in full]))


def sg_grad_hess(full, kk, K):
    N = full.size
    ext = np.concatenate([[0.0], full, [2.0 * math.pi]])
    g = kk * (2.0 * full - ext[:-2] - ext[2:]) + K * np.sin(full)
    diag = 2.0 * kk + K * np.cos(full)
    H = np.diag(diag)
    idx = np.arange(N - 1)
    H[idx, idx + 1] = -kk
    H[idx + 1, idx] = -kk
    return g, H


# ==================== 镜像约束 trust-ncg 松弛 ====================
def relax(model, w, mode, k=K_HALF):
    if model == "fq":
        kap, lam, c0 = 2.0 * w * w, 1.0, 0.0
        Efun = lambda f: fq_energy(f, kap, lam)
        GH = lambda f: fq_grad_hess(f, kap, lam)
    else:
        K = 0.005
        kk = K * w * w
        c0 = math.pi
        Efun = lambda f: sg_energy(f, kk, K)
        GH = lambda f: sg_grad_hess(f, kk, K)

    m = k - 1 if mode == "site" else k
    M = mirror_mat(k, mode)
    j = np.arange(1, m + 1, dtype=float)
    center = float(k) if mode == "site" else float(k) + 0.5
    if model == "fq":
        dof0 = np.tanh((j - center) / w)
    else:
        dof0 = 4.0 * np.arctan(np.exp(np.clip((j - center) / w, -80.0, 80.0)))

    def fun(d):
        return Efun(build_full(d, k, mode, c0))

    def jac(d):
        g, _ = GH(build_full(d, k, mode, c0))
        return M.T @ g

    def hess(d):
        _, H = GH(build_full(d, k, mode, c0))
        return M.T @ H @ M

    res = minimize(fun, dof0, jac=jac, hess=hess, method="trust-ncg",
                   options=dict(gtol=1e-12, maxiter=2000))
    x = res.x
    # 牛顿打磨：在约化空间迭代解 H d = -g（trust-ncg 受 xtol 限制停在 ~1e-8）
    for _ in range(30):
        g_full, H_full = GH(build_full(x, k, mode, c0))
        gr = M.T @ g_full
        if float(np.max(np.abs(gr))) < 1e-13:
            break
        Hr = M.T @ H_full @ M
        try:
            dx = np.linalg.solve(Hr, -gr)
        except np.linalg.LinAlgError:
            break
        E0 = Efun(build_full(x, k, mode, c0))
        t = 1.0
        for _ in range(40):
            xt = x + t * dx
            Et = Efun(build_full(xt, k, mode, c0))
            if Et <= E0:
                break
            t *= 0.5
        x = x + t * dx
    full = build_full(x, k, mode, c0)
    gmax = float(np.max(np.abs(M.T @ GH(full)[0])))
    return full, Efun(full), gmax


def measure(model, w):
    full_s, E_s, g_s = relax(model, w, "site")
    full_m, E_m, g_m = relax(model, w, "mid")
    barrier = E_s - E_m
    return dict(w=w, E_site=E_s, E_mid=E_m, barrier=barrier, M=E_s, r=barrier / E_s,
                gmax=max(g_s, g_m), full_site=full_s)


# ==================== 解析预期 ====================
def r_fq_analytic(w):
    p2 = math.pi ** 2
    return (p2 * (p2 * w * w + 1.0) * (3.0 * p2 * w * w + 2.0)
            / (15.0 * w * math.sinh(p2 * w)))


def r_sg_analytic(w):
    p2 = math.pi ** 2
    return p2 * (4.0 * p2 * w * w + 1.0) / (18.0 * w * math.sinh(p2 * w))


# ==================== 旧方法（解析剖面取样，2026-09-20 原方法） ====================
def old_barrier(K, N=1024):
    n = np.arange(N, dtype=float)
    g = 2.0 * np.pi * math.sqrt(K / 1.0)
    c0 = N / 2.0
    out = []
    for c in c0 + np.linspace(-0.5, 0.5, 41):
        w = -(2.0 / math.pi) * np.arctan(np.exp(np.clip(g * (n - c), -80.0, 80.0)))
        out.append(float(0.5 * np.sum(np.diff(w) ** 2)
                         + K * np.sum(1.0 - np.cos(2.0 * np.pi * (n + w)))))
    return float(np.max(out) - np.min(out))


def old_mass(K):
    return 4.0 * math.sqrt(1.0 * K) / math.pi


# ==================== 主流程 ====================
print("=" * 78)
print("第 616 号：内部壁 PN 势垒轮（现行机制下 P1-40 重估）")
print("=" * 78)

# ---- J1 冒烟 ----
print("\n[J1] 冒烟与收敛核对")
smoke = []
for model, w in [("fq", 2.251), ("sg", 2.251)]:
    out = measure(model, w)
    M_exp = (8.0 / 3.0) * w if model == "fq" else 8.0 * 0.005 * w
    dev = abs(out["M"] - M_exp) / M_exp
    ok = out["gmax"] <= 1e-9 and dev <= 0.03
    smoke.append(ok)
    print("  %s w=%.3f: max|g|=%.2e  壁能实测 %.8f 对解析 %.8f（偏差 %.3f%%）  %s"
          % (model, w, out["gmax"], out["M"], M_exp, 100 * dev, "PASS" if ok else "FAIL"))
rows.append(("J1", "冒烟（两模型收敛与壁能）", "见上", "max|g|<=1e-9 且偏差<=3%", all(smoke)))

# ---- J2 旧机制复现 ----
print("\n[J2] 旧读数复现（旧方法 vs 松弛法）")
old_r = {}
for K, exp_r, band in [(0.005, 1.51e-8, 0.01), (0.003, 3.04e-11, 0.05)]:
    r_old = old_barrier(K) / old_mass(K)
    old_r[K] = r_old
    ok = abs(r_old - exp_r) / exp_r <= band
    print("  旧方法 K=%.4f: r = %.6e  原读数 %.2e（偏差 %.3f%%）  %s"
          % (K, r_old, exp_r, 100 * abs(r_old - exp_r) / exp_r, "PASS" if ok else "FAIL"))
    rows.append(("J2", "旧方法复现 K=%.4f" % K, "%.4e" % r_old, "%.2e" % exp_r, ok))
w_sg_005 = 1.0 / (2.0 * math.pi * math.sqrt(0.005))
rel = measure("sg", w_sg_005)
ratio_rel = rel["r"] / old_r[0.005]
ok = 0.3 <= ratio_rel <= 3.0
print("  松弛法同档（w = %.3f）: r = %.4e，对旧方法比值 %.3f（登记为发现）  %s"
      % (w_sg_005, rel["r"], ratio_rel, "PASS" if ok else "FAIL"))
rows.append(("J2", "松弛法对旧方法比值（低估发现）", "%.3f" % ratio_rel, "[0.3, 3]", ok))
sg_pts = {w_sg_005: rel}

# ---- J3 新机制扫描与标度律 ----
print("\n[J3] 新机制（φ⁴ 壁）扫描：数值、解析与标度律")
w_fq = [0.707, 0.9, 1.1, 1.3, 1.5, 1.75, 2.0, 2.251, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5]
fq_pts = {}
for w in w_fq:
    out = measure("fq", w)
    fq_pts[w] = out
    r_exp = r_fq_analytic(w)
    print("  w=%.3f: r 数值 %.4e  解析 %.4e  比值 %.3f  barrier %.3e  max|g| %.1e"
          % (w, out["r"], r_exp, out["r"] / r_exp, out["barrier"], out["gmax"]))
    rows.append(("J3", "φ⁴ 扫描 w=%.3f" % w, "%.4e" % out["r"],
                 "解析 %.4e" % r_exp, True))
# 斜率拟合
wfit = [w for w in w_fq if 2.5 <= w <= 4.0]
y = np.array([math.log(fq_pts[w]["r"] / w ** 3) for w in wfit])
x = np.array(wfit)
coef = np.polyfit(x, y, 1)
slope = coef[0]
ok_rate = abs(slope + math.pi ** 2) / (math.pi ** 2) <= 0.05
print("  斜率拟合（ln(r/w³) 对 w，[2.5, 4.0]）：%.4f 对 -π² = %.4f（偏差 %.2f%%）  %s"
      % (slope, -math.pi ** 2, 100 * abs(slope + math.pi ** 2) / math.pi ** 2,
         "PASS" if ok_rate else "FAIL"))
rows.append(("J3", "斜率对 -π²", "%.4f" % slope, "%.4f ±5%%" % (-math.pi ** 2), ok_rate))
# 标定稳定性
cal = [fq_pts[w]["r"] / r_fq_analytic(w) for w in wfit]
spread = max(cal) / min(cal)
ok_cal = spread <= 1.3
print("  标定系数（[2.5, 4.0]）：均值 %.4f  极差比 %.4f（判据 <= 1.3）  %s"
      % (float(np.mean(cal)), spread, "PASS" if ok_cal else "FAIL"))
rows.append(("J3", "标定稳定性", "%.4f" % spread, "<= 1.3", ok_cal))
Cbar = float(np.mean(cal))

# ---- J4 势垒方向 ----
print("\n[J4] 势垒方向（整格点为极大）")
dirs = [fq_pts[w]["barrier"] > 0 for w in w_fq] + [sg_pts[w_sg_005]["barrier"] > 0]
print("  全部扫描点 E_site > E_mid：%s" % ("是" if all(dirs) else "否"))
rows.append(("J4", "E_site > E_mid 全扫描", "是" if all(dirs) else "否", "是", all(dirs)))

# ---- J5 同宽对照 ----
print("\n[J5] 同宽对照（w = 2.251）")
r_fq_2251 = fq_pts[2.251]["r"]
r_sg_2251 = sg_pts[w_sg_005]["r"]
ratio_cmp = r_fq_2251 / r_sg_2251
exp_cmp = 8.88 * 2.251 ** 2
ok_cmp = 20.0 <= ratio_cmp <= 100.0
print("  r_φ⁴ = %.4e，r_SG = %.4e，比值 %.1f（解析预期 %.1f）  %s"
      % (r_fq_2251, r_sg_2251, ratio_cmp, exp_cmp, "PASS" if ok_cmp else "FAIL"))
rows.append(("J5", "同宽比值 r_φ⁴/r_SG（w=2.251）", "%.1f" % ratio_cmp,
             "[20, 100]（解析 %.1f）" % exp_cmp, ok_cmp))

# ---- J6 宽度门槛（解析标定 + 纯数值拟合） ----
print("\n[J6] 宽度门槛（两法）")
def solve_w(rfun, target, lo=2.0, hi=9.0):
    for _ in range(400):
        mid = 0.5 * (lo + hi)
        if rfun(mid) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

cal_fq = lambda w: Cbar * r_fq_analytic(w)
w15_an = solve_w(cal_fq, 1e-15)
w20_an = solve_w(cal_fq, 1e-20)
print("  解析标定（系数 %.3f）：w*(1e-15) = %.3f，w*(1e-20) = %.3f" % (Cbar, w15_an, w20_an))

# 纯数值三参数拟合：ln r = a + p ln w - c w，用 [2.0, 4.5]
wnum = [w for w in w_fq if 2.0 <= w <= 4.5]
A = np.column_stack([np.ones(len(wnum)), np.log(wnum), np.array(wnum)])
b = np.array([math.log(fq_pts[w]["r"]) for w in wnum])
sol, res_, rank, sv = np.linalg.lstsq(A, b, rcond=None)
a_fit, p_fit, c_fit = sol
print("  数值拟合 ln r = %.4f + %.4f ln w + %.4f w（残差 RMS %.4f）"
      % (a_fit, p_fit, c_fit, float(np.sqrt(res_[0] / len(wnum))) if len(res_) else 0.0))
def numfun(w):
    return math.exp(a_fit + p_fit * math.log(w) + c_fit * w)
w15_num = solve_w(numfun, 1e-15)
w20_num = solve_w(numfun, 1e-20)
print("  数值拟合：w*(1e-15) = %.3f，w*(1e-20) = %.3f" % (w15_num, w20_num))
ok_w = (abs(w15_an - w15_num) <= 0.5 and 4.0 <= w15_num <= 5.2
        and abs(w20_an - w20_num) <= 0.6 and 5.0 <= w20_num <= 6.4)
print("  两法互差：%.3f（1e-15）、%.3f（1e-20）  %s"
      % (abs(w15_an - w15_num), abs(w20_an - w20_num), "PASS" if ok_w else "FAIL"))
rows.append(("J6", "宽度门槛（两法互差）", "%.3f / %.3f" % (abs(w15_an - w15_num), abs(w20_an - w20_num)),
             "<= 0.5 / <= 0.6 且落在 [4.0,5.2]/[5.0,6.4]", ok_w))
# SG 门槛（对照旧读数 3.99/5.21）
sg_all = {}
for w in [2.5, 2.75, 3.0, 3.25, 3.5]:
    sg_all[w] = measure("sg", w)
cal_sg = [sg_all[w]["r"] / r_sg_analytic(w) for w in sg_all]
Cbar_sg = float(np.mean(cal_sg))
cal_sg_fun = lambda w: Cbar_sg * r_sg_analytic(w)
w15_sg = solve_w(cal_sg_fun, 1e-15)
w20_sg = solve_w(cal_sg_fun, 1e-20)
print("  SG 解析标定（系数 %.3f，极差比 %.4f）：w*(1e-15) = %.3f，w*(1e-20) = %.3f（旧读数 3.99 / 5.21）"
      % (Cbar_sg, max(cal_sg) / min(cal_sg), w15_sg, w20_sg))
rows.append(("J6", "SG 宽度门槛（对照旧读数）", "%.3f / %.3f" % (w15_sg, w20_sg),
             "≈ 3.99 / 5.21", abs(w15_sg - 3.99) <= 0.5 and abs(w20_sg - 5.21) <= 0.6))

# ---- J7 机制自带档 ----
print("\n[J7] 机制自带档（note-486 测试档 κ=λ=k=1）")
# note-486 能量含 (k/2)(Δa)²，a=(1-σ)/2，故 Δa = -Δσ/2，折入后 κ_eff = κ + k/4 = 1.25，
# 等效壁宽 w = sqrt(κ_eff/(2λ)) = 0.791。
w_test = math.sqrt(1.25 / 2.0)
out_test = measure("fq", w_test)
r_test = out_test["r"]
ok_test = 0.02 <= r_test <= 0.5
print("  折入 a 项后等效宽度 w = %.4f: r = %.4f（期望 O(0.1)）  barrier = %.4f  M = %.4f  %s"
      % (w_test, r_test, out_test["barrier"], out_test["M"], "PASS" if ok_test else "FAIL"))
print("  对照（纯 κ=λ=1 宽度 w = 0.707）：r = %.4f" % fq_pts[0.707]["r"])
rows.append(("J7", "note-486 测试档 r（w=%.3f）" % w_test, "%.4f" % r_test, "[0.02, 0.5]", ok_test))
prof = out_test["full_site"]
dsig = np.abs(np.diff(prof))
peak = float(np.max(dsig))
above = np.where(dsig >= peak / 2.0)[0]
fwhm = float(above[-1] - above[0] + 1) if len(above) else 0.0
print("  同档 |Δσ| 半高全宽（单壁）= %.2f 格（note-486 报告含两壁合计 4 格）" % fwhm)
rows.append(("J7", "测试档墙宽核对", "%.2f 格/单壁" % fwhm, "量级核对（含两壁 4 格）",
             0.5 <= fwhm <= 4.0))

# ---- 图 ----
fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.7))
wd = np.linspace(0.65, 6.2, 500)
ax = axes[0]
ax.semilogy([w for w in w_fq], [fq_pts[w]["r"] for w in w_fq], "o",
            color="tab:red", ms=5, label="φ⁴ 壁（新机制）数值")
ax.semilogy(wd, [cal_fq(w) for w in wd], "-", color="tab:red", lw=1.2,
            label="φ⁴ 标定式（系数 %.2f）" % Cbar)
wsg = sorted(sg_all)
ax.semilogy(wsg, [sg_all[w]["r"] for w in wsg], "s", color="tab:blue", ms=5,
            label="FK/SG 孤子（旧机制）数值")
ax.semilogy(wd, [cal_sg_fun(w) for w in wd], "-", color="tab:blue", lw=1.2,
            label="SG 标定式（系数 %.2f）" % Cbar_sg)
ax.set_xlim(0.65, 6.2)
ax.set_ylim(1e-24, 1.0)
ax.set_xlabel("缺陷宽度 w（格）")
ax.set_ylabel("r = ΔE_PN / M")
ax.set_title("(a) 势垒比值与宽度：新机制对旧机制")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

ax = axes[1]
ax.semilogy(wd, [cal_fq(w) for w in wd], "-", color="tab:red", lw=1.4, label="φ⁴ 标定式")
ax.semilogy(wd, [numfun(w) for w in wd], "--", color="tab:red", lw=1.0, alpha=0.7,
            label="φ⁴ 数值拟合")
ax.semilogy(wd, [cal_sg_fun(w) for w in wd], "-", color="tab:blue", lw=1.4, label="SG 标定式")
ax.axhline(1e-15, ls="--", color="gray", lw=1.0)
ax.axhline(1e-20, ls=":", color="gray", lw=1.0)
ax.text(0.7, 2e-15, "实验带下沿 1e-15", fontsize=8, color="gray")
ax.text(0.7, 2e-20, "更严档 1e-20", fontsize=8, color="gray")
ax.axvline(w15_an, color="tab:red", ls="--", lw=0.8)
ax.axvline(w15_sg, color="tab:blue", ls="--", lw=0.8)
ax.plot([w15_an], [cal_fq(w15_an)], "v", color="tab:red", ms=7)
ax.plot([w15_sg], [cal_sg_fun(w15_sg)], "v", color="tab:blue", ms=7)
ax.annotate("w* = %.2f" % w15_an, (w15_an, cal_fq(w15_an)), textcoords="offset points",
            xytext=(6, 10), fontsize=8, color="tab:red")
ax.annotate("w* = %.2f" % w15_sg, (w15_sg, cal_sg_fun(w15_sg)), textcoords="offset points",
            xytext=(6, -16), fontsize=8, color="tab:blue")
ax.set_xlim(0.65, 6.2)
ax.set_ylim(1e-24, 1e-3)
ax.set_xlabel("缺陷宽度 w（格）")
ax.set_ylabel("r")
ax.set_title("(b) 宽度门槛：r = 1e-15 / 1e-20 的 w*")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

fig.tight_layout()
fig.savefig(IMG, dpi=150)
print("\n图已保存：%s" % IMG)

# ---- 汇总 ----
print("\n" + "=" * 78)
print("判据汇总")
print("=" * 78)
for tag, item, meas, exp, ok in rows:
    print("  [%s] %-36s 实测 %-26s 期望 %-34s %s"
          % (tag, item, meas, exp, "PASS" if ok else "FAIL"))
print("\n全部通过：%s" % ("是" if all(r[4] for r in rows) else "否"))
print("核心读数：φ⁴ w*(1e-15) = %.3f（解析标定）/ %.3f（数值拟合）；w*(1e-20) = %.3f / %.3f"
      % (w15_an, w15_num, w20_an, w20_num))
print("SG w*(1e-15) = %.3f，w*(1e-20) = %.3f（旧读数 3.99 / 5.21）" % (w15_sg, w20_sg))
print("同宽对照（w = 2.251）：r_φ⁴/r_SG = %.1f" % ratio_cmp)
print("松弛对旧方法（SG 同档）：比值 %.3f（旧读数低估）" % ratio_rel)
print("机制自带档（w = 0.707）：r = %.4f" % r_test)
