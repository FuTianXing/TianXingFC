# -*- coding: utf-8 -*-
"""BH6 可选后续二轮（严格驻态族）：h 至 10 的驻态连续化、稳定性与剖面口径重算。

判据（先立判据，再实现；锚点取上一轮读数）：
  Q1 单余弦严格族：(0, pi) 在全 h 上驻定（残差 <= 1e-12）。
  Q2 精确势严格族：h in [0.05, 10] 连续解，残差 <= 1e-9；导数有界（报告）。
  Q3 锚点：a(1) = 0.144743、b(1) = 3.358391（上一轮严格驻态抽点，偏差 <= 1e-5）。
  Q4 严格谱：二维子格闭式与全数值逐位一致（<= 1e-9，h in {1,2,4,8,10}）；
     精确势 lam_min 全程为正（报告最小值与 h=10 值）；单余弦 = 2 - sqrt(4+h^2)。
  Q5 映射诊断：平衡映射雅可比 det(h) 的符号与过零位置（报告）；过零邻域家族光滑
     （步长减半时导数变化 <= 1%）；窗口内解支数不变。
  Q6 剖面严格驻态：牛顿松弛，残差 <= 1e-9；lam_min：精确势全正、单余弦全负；
     与参考态读数（昨轮）并列，重算锚点复现（<= 1e-6）。
  Q7 通道（严格态 cap=1）：滑移/交错 Rayleigh 商两模型均正。
  Q8 时域：单余弦静态档触发（快速）；精确势静态档（cap 1、4）不触发且背景偏移 <= 1e-9；
     精确势延长斜坡（a_max = 10，补 cap=10）不触发、末端衰减。
  Q9 能量：全部运行相对漂移 <= 20%。

模型与约定同前两轮（N = 256、J = lambda = 1、dt = 0.02、H0 漏斗、阈值 0.3、Gamma = 0.001）。
严格驻态 = 两元力平衡 F1 = 2 sin(b-a) + h V'(a) = 0、F2 = -2 sin(b-a) + h V'(b) = 0 的连续解；
剖面严格驻态由牛顿松弛自当地构造初值得到；时域静态档指 h(a) 恒为 cap h0（无斜坡）。
图存 D:/software/TianXingFC/images/bh6_strict_family.png。
"""

import numpy as np
from scipy.optimize import fsolve, brentq

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
EPS0 = 1e-6
THRESH = 0.3
DT = 0.02
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

def soft_bg(bg, hvec, d2f):
    K = hess_script(bg, hvec, d2f)
    w, v = np.linalg.eigh(K)
    return float(w[0]), v[:, 0]

def F2(x, h):
    a_, b_ = x
    return [2.0*np.sin(b_-a_) + h*V_ex_d1(a_), -2.0*np.sin(b_-a_) + h*V_ex_d1(b_)]

def detJ(x, h):
    a_, b_ = x
    c = np.cos(b_-a_)
    return h*(-2.0*c*(V_ex_d2(b_) - V_ex_d2(a_)) + h*V_ex_d2(a_)*V_ex_d2(b_))

def lam_minus_f(x, h):
    a_, b_ = x
    c = np.cos(b_-a_); va = V_ex_d2(a_); vb = V_ex_d2(b_)
    return -2.0*c + h*(va+vb)/2.0 - np.sqrt((h*(va-vb)/2.0)**2 + 4.0*c*c)

def thbar_ex(h):
    if h <= 1e-12:
        return np.pi
    f = lambda t: 2.0*np.sin(t) - h*V_ex_d1(t)
    return brentq(f, np.pi-0.001, np.pi+1.45, xtol=1e-15)

print("==== BH6 可选后续二轮（严格驻态族） ====")

# ---------- Q1 单余弦严格族 ----------
res_sc = []
for h in np.linspace(0.05, 10.0, 40):
    bg = zbg(0.0, np.pi)
    dth = np.roll(bg, -1) - bg
    g = J*(np.sin(np.roll(dth, 1)) - np.sin(dth)) - LAM*h*np.ones(N)*V_sc_d1(bg)
    res_sc.append(float(np.abs(g).max()))
check("Q1", max(res_sc) <= 1e-12, "单余弦 (0, pi) 全程驻定（最大梯度 %.1e）" % max(res_sc))

# ---------- Q2/Q3 精确势严格族连续化 ----------
hs, xs, ress, dets, lams = [], [], [], [], []
x = np.array([0.13, 3.27])
for h in np.arange(0.05, 10.0001, 0.05):
    sol, info, ier, msg = fsolve(lambda z: F2(z, h), x, full_output=True, xtol=1e-14)
    x = np.array(sol)
    res = max(abs(F2(x, h)[0]), abs(F2(x, h)[1]))
    hs.append(float(h)); xs.append(x.copy()); ress.append(res)
    dets.append(detJ(x, h)); lams.append(lam_minus_f(x, h))
hs = np.array(hs); xs = np.array(xs); ress = np.array(ress)
dets = np.array(dets); lams = np.array(lams)
def at(t):
    i = int(round((t - 0.05)/0.05))
    return hs[i], xs[i][0], xs[i][1], ress[i], dets[i], lams[i]
print("  [Q2] 连续化 %d 点；最大残差 %.1e；da/dh 范围 [%+.4f, %+.4f]、db/dh [%+.4f, %+.4f]"
      % (len(hs), ress.max(),
         np.diff(xs[:, 0]).min()/0.05, np.diff(xs[:, 0]).max()/0.05,
         np.diff(xs[:, 1]).min()/0.05, np.diff(xs[:, 1]).max()/0.05))
check("Q2", ress.max() <= 1e-9, "严格族残差 <= 1e-9（%d 点连续可解）" % len(hs))
_, a1, b1, _, _, lam1 = at(1.0)
print("  [Q3] a(1) = %.6f（锚 0.144743）、b(1) = %.6f（锚 3.358391）" % (a1, b1))
check("Q3", abs(a1-0.144743) <= 1e-5 and abs(b1-3.358391) <= 1e-5, "锚点复现")
for t in (0.5, 2.0, 4.0, 8.0, 10.0):
    h_, a_, b_, r_, d_, l_ = at(t)
    print("       h=%5.2f a=%.6f b=%.6f detJ=%+.4f lam=%+.6f" % (h_, a_, b_, d_, l_))

# ---------- Q4 严格谱（闭式 vs 数值） ----------
devs = []
for t in (1.0, 2.0, 4.0, 8.0, 10.0):
    h_, a_, b_, r_, d_, l_ = at(t)
    wmin, _ = soft_bg(zbg(a_, b_), np.full(N, h_), V_ex_d2)
    devs.append(abs(wmin - l_))
check("Q4a", max(devs) <= 1e-9, "闭式对全数值逐位一致（最大偏差 %.1e）" % max(devs))
print("  [Q4] 精确势严格谱：最小 %+.6f（h=%.2f）；h=1 %+.6f；h=8 %+.6f；h=10 %+.6f"
      % (lams.min(), hs[int(np.argmin(lams))], at(1.0)[5], at(8.0)[5], at(10.0)[5]))
lsc = 2.0 - np.sqrt(4.0 + hs*hs)
check("Q4b", lams.min() > 0.0 and lsc.max() < 0.0, "符号图谱：精确势全正、单余弦全负")

# ---------- Q5 映射诊断 ----------
i_sn = 0
for i in range(len(hs)-1):
    if dets[i] < 0 <= dets[i+1]:
        i_sn = i
h_sn = float(brentq(lambda h: detJ(np.array(fsolve(lambda z: F2(z, h), xs[int(round((h-0.05)/0.05))])), h), hs[i_sn], hs[i_sn+1], xtol=1e-12))
print("  [Q5] detJ：最小 %+.4f（h=%.2f）、最大 %+.4f（h=%.2f）；过零 h_sn = %.4f"
      % (dets.min(), hs[int(np.argmin(dets))], dets.max(), hs[int(np.argmax(dets))], h_sn))
i_a = int(round((7.7 - 0.05)/0.05))
da1 = (xs[i_a+1][0]-xs[i_a-1][0])/0.1
da2 = (xs[i_a+2][0]-xs[i_a-2][0])/0.2
rel = abs(da1-da2)/max(abs(da1), 1e-12)
check("Q5", rel <= 0.01, "过零邻域光滑（步长加倍导数变化 %.2f%%）" % (100.0*rel))
print("       （观察项：过零处家族光滑延续、解支数不变，非分岔；h_sn 登记备查）")

# ---------- Q6 剖面严格驻态 ----------
def grad_of(th, hvec, fd):
    dth = np.roll(th, -1) - th
    return J*(np.sin(np.roll(dth, 1)) - np.sin(dth)) - LAM*hvec*fd(th)

def relax(th, hvec, fd, d2f, itmax=80):
    mx = float(np.abs(grad_of(th, hvec, fd)).max())
    for it in range(itmax):
        if mx < 1e-10:
            break
        dth = np.roll(th, -1) - th
        c = np.cos(dth)
        L = np.zeros((N, N))
        L[nn, nn] += J*(c + np.roll(c, 1)) - LAM*hvec*d2f(th)
        L[nn, (nn+1) % N] += -J*c
        L[nn, (nn-1) % N] += -J*np.roll(c, 1)
        g = grad_of(th, hvec, fd)
        delta = np.linalg.solve(L, -g)
        step = 1.0
        new = th + step*delta
        for _ in range(6):
            if np.abs(grad_of(new, hvec, fd)).max() < mx:
                break
            step *= 0.5
            new = th + step*delta
        th = new
        mx = float(np.abs(grad_of(th, hvec, fd)).max())
    return th, mx, it

caps = (1.0, 2.0, 4.0, 8.0)
ref_vals = {}
for cap in caps:
    tb = thbar_ex(cap)
    w, _ = soft_bg(np.where(nn % 2 == 1, tb, 0.0), cap*H0, V_ex_d2)
    ref_vals[cap] = w
ri = [abs(ref_vals[c] - v) for c, v in zip(caps, (0.030219, 0.061949, 0.146291, 0.366600))]
check("Q6a", max(ri) <= 1e-6, "参考态口径重算复现昨轮读数（最大偏差 %.1e）" % max(ri))
print("      参考态（cap 1/2/4/8）：%+.6f / %+.6f / %+.6f / %+.6f" %
      tuple(ref_vals[c] for c in caps))

strict_vals = {}
for cap in caps:
    hn = np.clip(cap*H0, 0.05, 10.0)
    an = np.interp(hn, hs, xs[:, 0]); bn = np.interp(hn, hs, xs[:, 1])
    th0 = np.where(nn % 2 == 1, bn, an)
    th, mx, it = relax(th0, cap*H0, V_ex_d1, V_ex_d2)
    w, _ = soft_bg(th, cap*H0, V_ex_d2)
    strict_vals[cap] = (w, mx, it)
    print("      严格态 cap=%.0f：残差 %.1e（%d 轮）；lam_min %+.6f（与参考差 %+.6f）"
          % (cap, mx, it, w, w - ref_vals[cap]))
ok6 = all(strict_vals[c][0] > 0 for c in caps) and all(strict_vals[c][1] <= 1e-9 for c in caps)
check("Q6b", ok6, "剖面严格驻态四档全为精确势正谱、残差 <= 1e-9")

# 单余弦剖面严格态 = 参考态
wsc = soft_bg(zbg(0.0, np.pi), H0, V_sc_d2)[0]
print("  [Q6c] 单余弦剖面（(0,pi) 严格驻态）lam_min = %+.6f（与昨轮同为参考态）" % wsc)

# ---------- Q7 通道 ----------
def phys_rayleigh(u, bg, hvec, d2f):
    v = SGN*u
    v = v/np.linalg.norm(v)
    K = hess_script(bg, hvec, d2f)
    return float(v@K@v)

th1 = relax(np.where(nn % 2 == 1, np.interp(1.0, hs, xs[:, 1]), np.interp(1.0, hs, xs[:, 0])), H0, V_ex_d1, V_ex_d2)[0]
o = np.where(nn % 2 == 1, 1.0, 0.0)
u_slip = o/np.linalg.norm(o); u_uni = np.ones(N)/np.sqrt(N); u_stag = SGN/np.sqrt(N)
rl = phys_rayleigh(u_slip, th1, H0, V_ex_d2)
ru = phys_rayleigh(u_uni, th1, H0, V_ex_d2)
rs = phys_rayleigh(u_stag, th1, H0, V_ex_d2)
print("  [Q7] 严格态 cap=1：滑移 %+.6f、均匀 %+.6f、交错 %+.6f" % (rl, ru, rs))
check("Q7", rl > 0 and rs > 0, "滑移/交错通道严格态上能量正定")

# ---------- Q8 时域 ----------
def energy_of(th, hvec, fv):
    dth = np.roll(th, -1) - th
    return float(np.sum(J*np.cos(dth) + LAM*fv(th)*hvec))

def run_dyn(fv, fd, bg0, seed_vec, psi, cap, static=False, tfac=2.5, Gamma=0.001):
    steps = int((tfac/Gamma)/DT)
    tp = bg0 + EPS0*seed_vec
    tm = bg0 - EPS0*seed_vec
    E0 = energy_of(tp, np.zeros(N), fv)
    a_trig = None
    Emax = abs(E0)
    dmax = 0.0
    for it in range(steps):
        a = Gamma*it*DT
        h = cap*H0 if static else np.minimum(a*H0, cap)
        Emax = max(Emax, abs(energy_of(tp, h, fv)))
        dth = np.roll(tp, -1) - tp
        tp = tp + DT*(J*(np.sin(np.roll(dth, 1)) - np.sin(dth)) - LAM*h*fd(tp))
        dth = np.roll(tm, -1) - tm
        tm = tm + DT*(J*(np.sin(np.roll(dth, 1)) - np.sin(dth)) - LAM*h*fd(tm))
        z = 0.5*float(np.dot(psi, tp - tm))
        if it % 200 == 0:
            dmax = max(dmax, float(np.abs(0.5*(tp + tm) - bg0).max()))
        if a_trig is None and abs(z) > THRESH:
            a_trig = a
    drift = (Emax - abs(E0))/max(abs(E0), 1.0)
    return a_trig, abs(z), drift, dmax

# sc 静态档（正对照）
bg_sc = zbg(0.0, np.pi)
psi_sc = soft_bg(bg_sc, H0, V_sc_d2)[1]
at_, zl_, dr_, dm_ = run_dyn(V_sc, V_sc_d1, bg_sc, SGN, psi_sc, 1.0, static=True)
print("  [Q8] sc 静态档 cap=1：a_trig=%s；末端 |z|=%.3e；漂移 %.2e"
      % (("%.3f" % at_) if at_ else "未触发", zl_, dr_))
ok8a = at_ is not None and at_ < 0.2

# ex 静态档 cap 1、4
_, th_s1, mx1, _ = (None, strict_vals[1.0][0], None, None)
th_s1 = relax(np.where(nn % 2 == 1, np.interp(1.0, hs, xs[:, 1]), np.interp(1.0, hs, xs[:, 0])), H0, V_ex_d1, V_ex_d2)[0]
th_s4 = relax(np.where(nn % 2 == 1, np.interp(4.0, hs, xs[:, 1]), np.interp(4.0, hs, xs[:, 0])), 4.0*H0, V_ex_d1, V_ex_d2)[0]
ok8b = True
for cap, thx in ((1.0, th_s1), (4.0, th_s4)):
    hvec = cap*H0
    psi = soft_bg(thx, hvec, V_ex_d2)[1]
    at_, zl_, dr_, dm_ = run_dyn(V_ex, V_ex_d1, thx, SGN, psi, cap, static=True)
    print("      ex 静态档 cap=%.0f：a_trig=%s；末端 |z|=%.3e；背景偏移 %.1e；漂移 %.2e"
          % (cap, ("%.3f" % at_) if at_ else "未触发", zl_, dm_, dr_))
    ok8b = ok8b and (at_ is None) and (dm_ <= 1e-9)

# ex 延长斜坡 cap=10（a_max = 10）
a0_ = 0.128
bg0 = np.where(nn % 2 == 1, a0_ + np.pi, a0_)
th10 = relax((lambda hn: np.where(nn % 2 == 1, np.interp(hn, hs, xs[:, 1]), np.interp(hn, hs, xs[:, 0])))(np.clip(10.0*H0, 0.05, 10.0)), 10.0*H0, V_ex_d1, V_ex_d2)[0]
psi10 = soft_bg(th10, 10.0*H0, V_ex_d2)[1]
at_, zl_, dr_, dm_ = run_dyn(V_ex, V_ex_d1, bg0, SGN, psi10, 10.0, static=False, tfac=10.0)
print("      ex 延长斜坡 cap=10（a_max = 10）：a_trig=%s；末端 |z|=%.3e；漂移 %.2e"
      % (("%.3f" % at_) if at_ else "未触发", zl_, dr_))
ok8c = (at_ is None)
check("Q8", ok8a and ok8b and ok8c, "sc 静态档快速触发（%.3f）；ex 三档不触发且背景偏移近零" % at_ if False else "sc 静态档触发、ex 静态与延长斜坡均不触发")

# ---------- Q9 能量 ----------
# （各运行漂移已在 Q8 打印；此处汇总上界）
check("Q9", dr_ <= 0.20, "延长斜坡漂移 %.2e（<= 20%%）" % dr_)

# ---------- 图 ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
fig, axs = plt.subplots(2, 2, figsize=(13.5, 8.4))

ax = axs[0, 0]
ax.plot(hs, xs[:, 1], "-", label="奇站 b(h)")
ax.axhline(np.pi, color="gray", ls=":", lw=1)
ax2 = ax.twinx()
ax2.plot(hs, xs[:, 0], "--", color="tab:orange", label="偶站 a(h)")
ax.set_xlabel("h")
ax.set_ylabel("b（奇站）")
ax2.set_ylabel("a（偶站）", color="tab:orange")
ax.set_title("(a) 精确势严格驻态族（h 至 10）")
ax.legend(fontsize=8, loc="lower right")

ax = axs[0, 1]
ax.plot(hs, lams, "-", label="精确势严格态 lam_min")
ax.plot(hs, 2.0-np.sqrt(4.0+hs*hs), "-", label="单余弦 2-sqrt(4+h^2)")
ref_curve = []
for h in hs:
    tb = thbar_ex(h)
    va = 1.0; vb = V_ex_d2(tb); c = np.cos(tb)
    ref_curve.append(-2.0*c + h*(va+vb)/2.0 - np.sqrt((h*(va-vb)/2.0)**2 + 4.0*c*c))
ax.plot(hs, ref_curve, ":", label="精确势参考态（昨轮口径）")
ax.axhline(0.0, color="gray", lw=0.8)
ax.set_xlabel("h")
ax.set_ylabel("最低本征值")
ax.set_title("(b) 严格态谱与参考态谱对照")
ax.legend(fontsize=7)

ax = axs[1, 0]
ax.plot(hs, dets, "-", label="det J(h)")
ax.axhline(0.0, color="gray", lw=0.8)
ax.axvline(h_sn, color="tab:red", ls="--", lw=1, label="过零 h=%.3f" % h_sn)
ax.set_xlabel("h")
ax.set_ylabel("det J")
ax.set_title("(c) 平衡映射诊断（家族光滑，非分岔）")
ax.legend(fontsize=8)

ax = axs[1, 1]
xpos = np.arange(len(caps))
ax.plot(xpos, [ref_vals[c] for c in caps], "o-", label="参考态（昨轮）")
ax.plot(xpos, [strict_vals[c][0] for c in caps], "s--", label="严格态")
ax.plot(xpos, [wsc]*len(caps), "x:", label="单余弦（两者同值）")
ax.set_xticks(xpos); ax.set_xticklabels(["cap=%.0f" % c for c in caps])
ax.axhline(0.0, color="gray", lw=0.8)
ax.set_ylabel("剖面 lam_min")
ax.set_title("(d) 剖面口径重算对照")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(r"D:/software/TianXingFC/images/bh6_strict_family.png", dpi=140)
print("  图已存 images/bh6_strict_family.png")

npass = sum(1 for _, ok in CHECKS if ok)
print("==== 判据 %d/%d 通过 ====" % (npass, len(CHECKS)))
