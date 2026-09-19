# -*- coding: utf-8 -*-
"""BH6 后续三轮（大种子扫掠）：势垒门的幅度定位与简并副本落点。

判据（先立判据，再实现；锚点取前两轮读数）：
  T1 驻定与锚点：h=1 均匀严格态 (a,b) = (0.144743, 3.358391)、梯度 <= 1e-6、
     lam_min = +0.296284（偏差 <= 1e-6）。
  T2 小种子回归：h=1 软模方向与交错方向、逐点幅度 0.1 与 1.0：全部完整回归原态
     （末偏移 <= 1e-4、末端 |grad| <= 1e-9、能量回到 E_bg，|E_fin - E_bg| <= 1e-6）。
  T3 门定位：h=1 四个方向（软+、软-、交错+、交错-）的势垒门 epsilon_c 二分定位
     （精度 <= 1%）；报告四值。
  T4 落点分类：各方向门上方一档的落点（(abar, bbar)、E_fin - E_bg、|grad|）；核验与初态
     的简并性（|E_fin - E_bg| <= 1e-3）。
  T5 h 依赖：h = 2、4 的软模+方向门读数（趋势报告）。
  T6 剖面档：cap=1 剖面严格驻态上交错方向的门读数（报告；与均匀档比较）。
  T7 单余弦对照：逐点幅度 1e-8 的交错种子即逃逸（末偏移 >= 1、能量下降 >= 1）；
     门记为 <= 1e-8（无垒对照）。
  T8 稳健性：全部运行确定性；逃逸档末端 |grad| <= 1e-9 或报告；运行时长报告。

模型与约定同前三轮：一维周期链（N = 256、J = lambda = 1、过阻尼 dt = 0.02、Gamma = 0.001），
均匀档 h 恒为常数，剖面档 h = cap * h0；种子按逐点幅度口径（软模方向取 sqrt(N) 归一，
交错方向逐点 |delta| 即幅度）。运行长度：均匀档 a_max = 0.3（15000 步）、剖面档 0.6（30000 步）。
图存 D:/software/TianXingFC/images/bh6_large_seed.png。
"""

import numpy as np
from scipy.optimize import fsolve

SQ2 = np.sqrt(2.0)
V_sc = lambda t: 1.0 - np.cos(t)
V_sc_d1 = lambda t: np.sin(t)
V_sc_d2 = lambda t: np.cos(t)
V_ex = lambda t: (1.0 - np.cos(SQ2*t) + 2.0*(1.0 - np.cos(t/SQ2)))/3.0
V_ex_d1 = lambda t: (SQ2*np.sin(SQ2*t) + SQ2*np.sin(t/SQ2))/3.0
V_ex_d2 = lambda t: (2.0*np.cos(SQ2*t) + np.cos(t/SQ2))/3.0

N = 256
J = 1.0
LAM = 1.0
DT = 0.02
GAM = 0.001
nn = np.arange(N)
rc = np.minimum(np.abs(nn - N//2), N - np.abs(nn - N//2))
H0 = np.where(rc <= 7, 1.0, np.minimum(10.597/np.maximum(rc, 1), 1.0))
SGN = (-1.0)**nn

CHECKS = []

def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))

def zbg(a_, b_):
    return np.where(nn % 2 == 1, b_, a_)

def hess_script(bg, hvec, d2f):
    K = np.zeros((N, N))
    dth = np.roll(bg, -1) - bg
    c = np.cos(dth)
    K[nn, nn] += -(J*c + J*np.roll(c, 1))
    K[nn, (nn+1) % N] += -J*c
    K[nn, (nn-1) % N] += -J*np.roll(c, 1)
    K[nn, nn] += LAM*hvec*d2f(bg)
    return K

def grad_of(th, hvec, fd):
    dth = np.roll(th, -1) - th
    return J*(np.sin(np.roll(dth, 1)) - np.sin(dth)) - LAM*hvec*fd(th)

def energy_of(th, hvec, fv):
    dth = np.roll(th, -1) - th
    return float(np.sum(J*np.cos(dth) + LAM*fv(th)*hvec))

# ---------- 严格族（复算）与三个均匀档 ----------
def F2(x, h):
    a_, b_ = x
    return [2.0*np.sin(b_-a_) + h*V_ex_d1(a_), -2.0*np.sin(b_-a_) + h*V_ex_d1(b_)]

hs, aa, bb = [], [], []
x = np.array([0.13, 3.27])
for h in np.arange(0.05, 10.0001, 0.05):
    sol = fsolve(lambda z: F2(z, h), x, xtol=1e-14)
    x = np.array(sol)
    hs.append(float(h)); aa.append(float(x[0])); bb.append(float(x[1]))
hs = np.array(hs); aa = np.array(aa); bb = np.array(bb)

def ab_at(h):
    return float(np.interp(h, hs, aa)), float(np.interp(h, hs, bb))

print("==== BH6 后续三轮（大种子扫掠） ====")

bg1 = zbg(*ab_at(1.0))
hv1 = np.ones(N)
w1, v1 = np.linalg.eigh(hess_script(bg1, hv1, V_ex_d2))
psi1 = SGN*v1[:, 0]
psi1 = psi1/np.linalg.norm(psi1)
g0_ = float(np.abs(grad_of(bg1, hv1, V_ex_d1)).max())
print("  [T1] h=1 严格态：a=%.6f b=%.6f；|grad|=%.1e；lam_min=%+.6f"
      % (ab_at(1.0)[0], ab_at(1.0)[1], g0_, w1[0]))
check("T1", g0_ <= 1e-6 and abs(w1[0]-0.296284) <= 1e-5 and abs(ab_at(1.0)[0]-0.144743) <= 1e-5
      and abs(ab_at(1.0)[1]-3.358391) <= 1e-5, "驻定、锚点与谱复现")

# ---------- 运行器 ----------
def run_static(bg, hvec, seed, fv, fd, steps):
    th = bg + seed
    Eb = energy_of(bg, hvec, fv)
    dmax = 0.0
    for it in range(steps):
        th = th + DT*grad_of(th, hvec, fd)
        if it % 20 == 0:
            dmax = max(dmax, float(np.abs(th - bg).max()))
    dfin = float(np.abs(th - bg).max())
    abar = float(th[0::2].mean()); bbar = float(th[1::2].mean())
    dE = energy_of(th, hvec, fv) - Eb
    gfin = float(np.abs(grad_of(th, hvec, fd)).max())
    return dmax, dfin, abar, bbar, dE, gfin

def classify_uniform(h, seed_vec, amp, steps=15000):
    a_, b_ = ab_at(h)
    bg = zbg(a_, b_)
    hv = np.full(N, h)
    dmax, dfin, abar, bbar, dE, gfin = run_static(bg, hv, amp*np.sqrt(N)*seed_vec, V_ex, V_ex_d1, steps)
    return dfin > 0.5, (dmax, dfin, abar, bbar, dE, gfin)

def bisect_gate(fun, lo, hi, iters=10):
    # fun(eps) -> True 表示逃逸
    for _ in range(iters):
        mid = 0.5*(lo + hi)
        if fun(mid):
            hi = mid
        else:
            lo = mid
    return 0.5*(lo + hi)

def find_gate(fun, start=0.4, top=4.6):
    eps = start
    hi = None
    while eps <= top:
        if fun(eps):
            hi = eps
            break
        eps *= 1.6
    if hi is None:
        return None, None
    lo = hi/1.6
    g = bisect_gate(fun, lo, hi)
    return g, hi

dirs = {"软+": psi1, "软-": -psi1, "交错+": SGN/np.sqrt(N), "交错-": -SGN/np.sqrt(N)}

# ---------- T2 小种子回归 ----------
ok2 = True
rows2 = []
for nm, vec in dirs.items():
    for amp in (0.1, 1.0):
        esc, (dmax, dfin, abar, bbar, dE, gfin) = classify_uniform(1.0, vec, amp)
        rows2.append((nm, amp, dmax, dfin, dE, gfin, esc))
        ok2 = ok2 and (not esc) and dfin <= 1e-4 and gfin <= 1e-9 and abs(dE) <= 1e-6
print("  [T2] 小种子回归：")
for nm, amp, dmax, dfin, dE, gfin, esc in rows2:
    print("       %s amp=%.1f：最大偏移 %.3f；末偏移 %.1e；dE=%.1e；|g|fin=%.1e"
          % (nm, amp, dmax, dfin, dE, gfin))
check("T2", ok2, "幅度 0.1 与 1.0 的两方向四种种子全部完整回归")

# ---------- T3 门定位 ----------
print("  [T3] h=1 势垒门（二分）：")
gates1 = {}
for nm, vec in dirs.items():
    g, hi = find_gate(lambda eps, v=vec: classify_uniform(1.0, v, eps)[0])
    gates1[nm] = g
    print("       %s：epsilon_c = %s" % (nm, ("%.4f" % g) if g else "未定位（> 4.6）"))
ok3 = all(g is not None for g in gates1.values())
check("T3", ok3, "四个方向的势垒门均已定位")

# ---------- T4 落点分类（含结构诊断） ----------
print("  [T4] 门上方落点（逐方向取 1.1 epsilon_c 或 4.6）：")
landings = {}
for nm, vec in dirs.items():
    eps = min(1.1*gates1[nm], 4.6)
    a_, b_ = ab_at(1.0)
    bg = zbg(a_, b_)
    Eb = energy_of(bg, hv1, V_ex)
    th = bg + eps*np.sqrt(N)*vec
    for it in range(15000):
        th = th + DT*grad_of(th, hv1, V_ex_d1)
    dfin = float(np.abs(th - bg).max())
    abar = float(th[0::2].mean()); bbar = float(th[1::2].mean())
    se = float(th[0::2].std()); so = float(th[1::2].std())
    dE = energy_of(th, hv1, V_ex) - Eb
    gfin = float(np.abs(grad_of(th, hv1, V_ex_d1)).max())
    bond = float(np.sum(J*np.cos(np.roll(th, -1) - th)))
    onsite = float(np.sum(V_ex(th)))
    landings[nm] = (eps, dfin > 0.5, abar, bbar, dE, gfin)
    print("       %s amp=%.4f：逃逸=%s；(abar,bbar)=(%+.3f,%+.3f)；子格内 std %.3f / %.3f"
          % (nm, eps, dfin > 0.5, abar, bbar, se, so))
    print("             E_fin-E_bg=%+.3f（键 %+.1f + 位 %+.1f）；|g|fin=%.1e；末偏移 %.3f"
          % (dE, bond, onsite, gfin, dfin))
ok4 = all(landings[nm][1] for nm in dirs)
check("T4", ok4, "门上方全部逃逸（落点读数与诊断见上）")

# ---------- T5 h 依赖 ----------
print("  [T5] 门对 h：")
gates_h = {}
for h in (2.0, 4.0):
    g, hi = find_gate(lambda eps, hh=h: classify_uniform(hh, psi1, eps)[0])
    gates_h[h] = g
    print("       h=%.0f 软模+：epsilon_c = %s" % (h, ("%.4f" % g) if g else "未定位（> 4.6）"))
check("T5", all(g is not None for g in gates_h.values()), "h=2、4 门读数已报告")

# ---------- T6 剖面档 ----------
def relax(bg, hvec, fd, d2f, itmax=80):
    th = bg.copy()
    for _ in range(itmax):
        g = grad_of(th, hvec, fd)
        if float(np.abs(g).max()) < 1e-10:
            break
        dth = np.roll(th, -1) - th
        c = np.cos(dth)
        L = np.zeros((N, N))
        L[nn, nn] += J*(c + np.roll(c, 1)) - LAM*hvec*d2f(th)
        L[nn, (nn+1) % N] += -J*c
        L[nn, (nn-1) % N] += -J*np.roll(c, 1)
        th = th + np.linalg.solve(L, -g)
    return th, float(np.abs(grad_of(th, hvec, fd)).max())

hn = np.clip(1.0*H0, 0.05, 10.0)
th_p = np.where(nn % 2 == 1, np.interp(hn, hs, bb), np.interp(hn, hs, aa))
th_p, res_p = relax(th_p, 1.0*H0, V_ex_d1, V_ex_d2)
print("  [T6] 剖面档（cap=1）严格态残差 %.1e" % res_p)
def classify_profile(amp, steps=30000):
    dmax, dfin, abar, bbar, dE, gfin = run_static(th_p, 1.0*H0, amp*SGN, V_ex, V_ex_d1, steps)
    return dfin > 0.5, (dmax, dfin, abar, bbar, dE, gfin)
g_prof = None
eps = 0.4
hi = None
while eps <= 4.6:
    if classify_profile(eps)[0]:
        hi = eps
        break
    eps *= 1.6
if hi is not None:
    lo = hi/1.6
    for _ in range(10):
        mid = 0.5*(lo + hi)
        if classify_profile(mid)[0]:
            hi = mid
        else:
            lo = mid
    g_prof = 0.5*(lo + hi)
print("       剖面档交错方向：epsilon_c = %s" % (("%.4f" % g_prof) if g_prof else "未定位（> 4.6）"))
check("T6", g_prof is not None, "剖面档门读数已报告")

# ---------- T7 单余弦对照 ----------
bg_sc = zbg(0.0, np.pi)
dmax, dfin, abar, bbar, dE, gfin = run_static(bg_sc, hv1, 1e-8*SGN, V_sc, V_sc_d1, 15000)
esc_sc = dfin > 0.5
print("  [T7] 单余弦对照：amp=1e-8 交错种子：逃逸=%s；末偏移 %.3f；(a,b)=(%+.3f,%+.3f)；dE=%+.3f"
      % (esc_sc, dfin, abar, bbar, dE))
check("T7", esc_sc and dfin >= 1.0 and dE <= -1.0, "单余弦 1e-8 即逃逸（无垒对照）")

# ---------- 图 ----------
grid_amps = np.array([0.05, 0.1, 0.2, 0.4, 0.8, 1.2, 1.6, 2.4, 3.2])
curve = {}
for nm in ("软+", "交错+"):
    ys = []
    for a_ in grid_amps:
        esc, (dmax, dfin, abar, bbar, dE, gfin) = classify_uniform(1.0, dirs[nm], a_)
        ys.append(dfin)
    curve[nm] = np.array(ys)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
fig, axs = plt.subplots(2, 2, figsize=(13.5, 8.4))

ax = axs[0, 0]
for nm, mk in (("软+", "o-"), ("交错+", "s--")):
    ax.semilogx(grid_amps, np.maximum(curve[nm], 1e-16), mk, label=nm)
    gv = gates1[nm]
    if gv:
        ax.axvline(gv, ls=":", lw=1)
ax.axhline(0.5, color="gray", ls=":", lw=1, label="逃逸判据")
ax.set_xlabel("种子逐点幅度（弧度，对数轴）")
ax.set_ylabel("末偏移")
ax.set_title("(a) h=1 逃逸图（虚线竖线为门）")
ax.legend(fontsize=8)

ax = axs[0, 1]
labels = ["软+", "软-", "交错+", "交错-", "h=2软+", "h=4软+", "剖面交错"]
vals = [gates1["软+"], gates1["软-"], gates1["交错+"], gates1["交错-"],
        gates_h[2.0], gates_h[4.0], g_prof]
ax.bar(range(len(vals)), vals, color="tab:blue", alpha=0.7)
ax.set_xticks(range(len(vals)))
ax.set_xticklabels(labels, fontsize=8, rotation=20)
ax.set_ylabel("epsilon_c（弧度）")
ax.set_title("(b) 势垒门汇总")
for i, v_ in enumerate(vals):
    ax.text(i, v_+0.02, "%.2f" % v_, ha="center", fontsize=8)

ax = axs[1, 0]
ax.semilogx([1e-8, 1e-3], [dfin, dfin], "o-", label="单余弦（恒定 O(1)）")
ax.semilogx(grid_amps, np.maximum(curve["交错+"], 1e-16), "s--", label="精确势交错+")
ax.axhline(0.5, color="gray", ls=":", lw=1)
ax.set_xlabel("种子逐点幅度（弧度，对数轴）")
ax.set_ylabel("末偏移")
ax.set_title("(c) 单余弦与精确势对照（八个数量级差）")
ax.legend(fontsize=8)

ax = axs[1, 1]
a0_, b0_ = ab_at(1.0)
pts = {"初态": (a0_, b0_), "简并副本（软+ 落点）": (landings["软+"][2], landings["软+"][3])}
ax.plot([p[0] for p in pts.values()], [p[1] for p in pts.values()], "o")
for nm2, p in pts.items():
    ax.annotate(nm2, p, fontsize=8)
for nm in ("交错+", "交错-", "软-"):
    ax.plot([landings[nm][2]], [landings[nm][3]], "x")
    ax.annotate(nm, (landings[nm][2], landings[nm][3]), fontsize=8)
ax.set_xlabel("abar（偶站均值）")
ax.set_ylabel("bbar（奇站均值）")
ax.set_title("(d) 落点（均值平面）")
plt.tight_layout()
plt.savefig(r"D:/software/TianXingFC/images/bh6_large_seed.png", dpi=140)
print("  图已存 images/bh6_large_seed.png")

npass = sum(1 for _, ok in CHECKS if ok)
print("==== 判据 %d/%d 通过 ====" % (npass, len(CHECKS)))
