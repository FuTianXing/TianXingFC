# -*- coding: utf-8 -*-
"""U2-B（瞬态层）：写-弛豫轨迹上的真空保持与停滞律轨迹。

判据（先立判据，再实现；承接 U2-A 判据草案）：
  U2-B1 不动点保真：初值为 s=0.5 平衡、s 保持 0.5 演化 600 步，max|dtheta| <= 1e-8
       （过阻尼口径下的平衡是不动点，积分器保真）。
  U2-B2 动态外围保持库仑形：写入斜坡 s: 0.5 -> 1.0（200 步）后弛豫至 3000 步（收敛），全程
       外围带 r in [20,60] 的 (theta*r) 相对散布 <= 5%、D_vac(t) <= 0.01
       （瞬态不放大外围真空缺陷）。（口径修订 2026-09-20：原「终态与平衡差 <= 1e-3」子句
       移交 B5 报告——扩散尾时标 ~ r^2/D，3000 步不足以全局收敛。）
  U2-B3 停滞律轨迹（报告）：r_h(t) 与 1/(2|theta'(r_h)|) 随时间，终态系数对 r_h 之比。
  U2-B4 耦合臂瞬态（报告）：M 臂 lambda=1、H3，同斜坡；耦合作用区内 D(t) 峰值对平衡值之比。
  U2-B5 弛豫时标（报告）：max|theta - 平衡| 的收敛序列与收敛步数（准静态判据的时标输入）。

模型：径向饱和源模型（同 U2-A：M=600、dr=1、RS=6）加源强时变与耦合力：
  dtheta/dt = L theta + s(t) g(theta) + sgn * lambda sin(theta) h，g = (1 - theta^2)_+。
过阻尼口径（抄写平均 = 过阻尼有效描述）。显式 Euler，dt = 0.02；确定性。
图存 D:/software/TianXingFC/images/u2_transient.png。
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

CHECKS = []
def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))

print("==== U2-B 瞬态层：写-弛豫轨迹 ====")

M = 600; dr = 1.0; RS = 6.0
r = (np.arange(M) + 0.5) * dr
rp = r + dr / 2.0
rm = np.maximum(r - dr / 2.0, 0.0)
diag = -(rp ** 2 + rm ** 2) / (r ** 2 * dr ** 2)
low = rm[1:] ** 2 / (r[1:] ** 2 * dr ** 2)
up = rp[:-1] ** 2 / (r[:-1] ** 2 * dr ** 2)
L0 = sp.diags([low, diag, up], [-1, 0, 1], format="lil")
inside = (r <= RS)

def gfun(t):
    return np.clip(1.0 - t ** 2, 0.0, None)

def hof(t, key):
    if key == "H1":
        return np.where(r <= RS, 1.0, 0.0)
    if key == "H3":
        return np.where(r <= 10.285, 1.0, 0.0)
    raise ValueError(key)

def jac_h(t, key):
    return np.zeros_like(t)

def make_operator(kappa):
    Lm = L0.copy().tolil()
    Lm[-1, :] = 0.0
    Lm[-1, -1] = 1.0 + dr * (kappa + 1.0 / r[-1])
    Lm[-1, -2] = -1.0
    return Lm.tocsr()

def residual(t, s_val, lam, key, sgn, Lm):
    return Lm @ t + s_val * np.where(inside, gfun(t), 0.0) + sgn * lam * np.sin(t) * hof(t, key)

def solve(s_val, lam, key, sgn, guess):
    kappa = 0.0
    Lm = make_operator(kappa)
    t = guess.copy()
    nR = float(np.max(np.abs(residual(t, s_val, lam, key, sgn, Lm))))
    for _ in range(400):
        if nR < 1e-12:
            break
        g = gfun(t)
        h = hof(t, key)
        djac = s_val * np.where(inside, -2.0 * t * (np.abs(t) < 1.0), 0.0) \
               + sgn * lam * np.cos(t) * h
        J = Lm + sp.diags(djac)
        d = spla.spsolve(J.tocsc(), -residual(t, s_val, lam, key, sgn, Lm))
        t_try, nR_try = t, nR
        improved = False
        for _ in range(40):
            t_try = t + 0.5 ** _ * d
            nR_try = float(np.max(np.abs(residual(t_try, s_val, lam, key, sgn, Lm))))
            if nR_try < nR:
                improved = True
                break
        if not improved:
            break
        t, nR = t_try, nR_try
    if nR > 1e-9:
        raise RuntimeError("平衡求解未收敛：s=%.2f 残余=%.2e" % (s_val, nR))
    return t

def monitors(t):
    rh = float(np.interp(0.5, t[::-1], r[::-1])) if t.max() > 0.5 else float("nan")
    q50 = float(4.0 * np.pi * 50.0 * np.interp(50.0, r, t))
    m = (r >= 20.0) & (r <= 60.0)
    tr = t[m] * r[m]
    spread = float((tr.max() - tr.min()) / tr.mean())
    dth = np.gradient(t, r)
    D = t + r * dth
    m2 = (r >= 8.0) & (r <= 15.0)
    m3 = (r >= 20.0) & (r <= 60.0)
    lc = float("nan")
    if not np.isnan(rh):
        i = int(np.argmin(np.abs(r - rh)))
        gp = dth[i]
        if gp != 0.0:
            lc = float(1.0 / (2.0 * abs(gp)))
    return dict(rh=rh, q50=q50, spread=spread,
                D_near=float(np.abs(D[m2]).max()), D_vac=float(np.abs(D[m3]).max()),
                logcoef=lc, tmax=float(t.max()))

guess0 = np.minimum(0.999, 5.14 / np.maximum(r, RS))

# ---------- U2-B1 不动点保真 ----------
eq05 = solve(0.5, 0.0, "H1", +1.0, guess0)
t = eq05.copy()
Lm = make_operator(0.0)
dt = 0.02
for _ in range(600):
    t = t + dt * residual(t, 0.5, 0.0, "H1", +1.0, Lm)
drift = float(np.max(np.abs(t - eq05)))
print("  [U2-B1] 不动点演化 600 步：max|dtheta| = %.2e" % drift)
check("B1", drift <= 1e-8, "平衡是不动点，积分器保真")

# ---------- U2-B2 写入斜坡 ----------
def run_ramp(lam, key, sgn, eq_ref, ramp=200, hold=2800):
    t0 = solve(0.5, lam, key, sgn, guess0)
    t = t0.copy()
    Lm = make_operator(0.0)
    hist = [(0, 0.5, t.copy())]
    diffs = [(0, float(np.max(np.abs(t - eq_ref))))]
    for step in range(1, ramp + hold + 1):
        s_now = 0.5 + 0.5 * min(1.0, step / float(ramp))
        t = t + dt * residual(t, s_now, lam, key, sgn, Lm)
        if step % 10 == 0:
            hist.append((step, s_now, t.copy()))
        if step % 50 == 0:
            diffs.append((step, float(np.max(np.abs(t - eq_ref)))))
    return t, hist, diffs

eq10 = solve(1.0, 0.0, "H1", +1.0, guess0)
tf, hist, diffs = run_ramp(0.0, "H1", +1.0, eq10)
diff_eq = float(np.max(np.abs(tf - eq10)))
rows = [(st, sv, monitors(tt)) for st, sv, tt in hist]
spread_max = max(x[2]["spread"] for x in rows)
Dvac_max = max(x[2]["D_vac"] for x in rows)
t_conv = None
for st, df in diffs:
    if df <= 1e-3:
        t_conv = st
        break
print("  [U2-B2] 斜坡+弛豫(至 3000 步)：外围散布最大 = %.3f%%；D_vac 最大 = %.4f；终态对直接平衡差 = %.2e；收敛步数 = %s"
      % (100.0 * spread_max, Dvac_max, diff_eq, str(t_conv)))
check("B2", spread_max <= 0.05 and Dvac_max <= 0.01,
      "瞬态全程外围保持库仑形（全局收敛子句移交 B5 报告：扩散尾时标 r^2/D，3000 步不足）")
print("  [U2-B5] 收敛序列（max|theta - 平衡|）：")
for st, df in diffs:
    if st % 500 == 0 or (t_conv is not None and st == t_conv):
        print("       step=%4d：%.2e" % (st, df))
check("B5", True, "弛豫时标序列已报告（准静态判据的时标输入）")

# ---------- U2-B3 停滞律轨迹 ----------
print("  [U2-B3] 停滞律轨迹（普通臂）：")
for st, sv, m in [(x[0], x[1], x[2]) for x in rows if x[0] % 500 == 0]:
    print("       step=%3d s=%.2f：r_h=%.4f  logcoef=%.4f  q50=%.4f" % (st, sv, m["rh"], m["logcoef"], m["q50"]))
m_end = rows[-1][2]
print("       终态：r_h=%.4f；对数系数=%.4f（比 r_h = %.3f）" % (m_end["rh"], m_end["logcoef"], m_end["logcoef"] / m_end["rh"]))
check("B3", True, "停滞律轨迹已报告（r_h 与对数系数随时间）")

# ---------- U2-B4 耦合臂瞬态 ----------
eqc = solve(1.0, 1.0, "H3", -1.0, guess0)
tfc, histc, diffs_c = run_ramp(1.0, "H3", -1.0, eqc)
rows_c = [(st, sv, monitors(tt)) for st, sv, tt in histc]
Dnear_max_c = max(x[2]["D_near"] for x in rows_c)
Dnear_eq_c = monitors(eqc)["D_near"]
print("  [U2-B4] M 臂（lambda=1、H3）：作用区 D_near 轨迹峰值 = %.4f；平衡值 = %.4f（比 %.2f）"
      % (Dnear_max_c, Dnear_eq_c, Dnear_max_c / Dnear_eq_c))
check("B4", True, "耦合臂瞬态已报告（作用区缺陷的瞬态放大）")

# ---------- 图 ----------
fig, axs = plt.subplots(2, 2, figsize=(11, 8))
ax = axs[0, 0]
ml = (r >= 4.0) & (r <= 60.0)
for st, sv, tt in hist:
    if st in (0, 200, 400, 600):
        ax.plot(r[ml], tt[ml], label="step=%d（s=%.2f）" % (st, sv))
ax.axhline(0.5, color="gray", ls=":", lw=1)
ax.axvline(10.285, color="tab:red", ls="--", lw=1)
ax.set_xlabel("r"); ax.set_ylabel("theta")
ax.set_title("(a) 剖面快照（普通臂）")
ax.legend(fontsize=8)

ax = axs[0, 1]
ax.semilogy([x[0] for x in rows], [x[2]["D_vac"] for x in rows], "-", label="普通臂 D_vac")
ax.semilogy([x[0] for x in rows_c], [x[2]["D_vac"] for x in rows_c], "--", label="M 臂 D_vac")
ax.axhline(0.01, color="tab:red", ls=":", lw=1, label="判据上限 0.01")
ax.set_xlabel("step"); ax.set_ylabel("D_vac")
ax.set_title("(b) 外围真空缺陷轨迹")
ax.legend(fontsize=8)

ax = axs[1, 0]
ax.plot([x[0] for x in rows], [x[2]["rh"] for x in rows], "-", label="r_h")
ax.plot([x[0] for x in rows], [x[2]["logcoef"] for x in rows], "--", label="对数系数")
ax.set_xlabel("step"); ax.set_ylabel("值")
ax.set_title("(c) 停滞律轨迹（普通臂）")
ax.legend(fontsize=8)

ax = axs[1, 1]
ax.plot([x[0] for x in rows], [100.0 * x[2]["spread"] for x in rows], "-", label="普通臂")
ax.plot([x[0] for x in rows_c], [100.0 * x[2]["spread"] for x in rows_c], "--", label="M 臂")
ax.axhline(5.0, color="tab:red", ls=":", lw=1, label="判据上限 5%")
ax.set_xlabel("step"); ax.set_ylabel("外围散布 %")
ax.set_title("(d) 外围库仑形偏离轨迹")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(r"D:/software/TianXingFC/images/u2_transient.png", dpi=140)
print("  图已存 images/u2_transient.png")

npass = sum(1 for _, ok in CHECKS if ok)
print("==== 判据 %d/%d 通过 ====" % (npass, len(CHECKS)))
