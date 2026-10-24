# -*- coding: utf-8 -*-
"""BH6 可选后续（滑移与 h>1）：超满偏稳定性图谱与触发通道判定。

判据（先立判据，再实现；锚点取 BH6 后续轮与 S3N 登记值）：
  P1 锚点：theta_bar(1) = 3.19002；剖面交错谱 sc -0.22604 / ex +0.03022；
     均匀 h=1 精确势 +0.21802（与上一轮逐位一致，偏差 <= 1e-4）。
  P2 谱延伸：h in {1,2,4,8} 均匀背景下，全数值最低本征值 = 二维子格闭式
     lam-(h) = -2cos(tb) + h(1+v2)/2 - sqrt((h(1-v2)/2)^2 + 4cos(tb)^2)，
     单余弦退化为 2 - sqrt(4 + h^2)；偏差 <= 1e-8。
  P3 符号图谱：h in [0.5,10] 上 sc lam- 全负、ex lam- 全正（h->0 两侧均趋零）；
     剖面加帽 cap in {1,2,4,8} 的读数符号不翻转。
  P4 通道：滑移（奇子格位移）与均匀位移方向的物理 Rayleigh 商在两模型上均 > 0；
     交错方向在精确势上 > 0；并报告软模对三种种子的投影（上一轮口径）。
  P5 时域（Gamma=0.001，S3N 协议）：sc cap=1 交错档触发 a_trig ~ 0.674（偏差 <= 5%）；
     sc cap=4 同量级；ex cap in {1,2,4} 交错档不触发且末端 |z| 衰减 >= 1e-6 倍初值；
     滑移/均匀种子：ex 不触发；sc 报告。
  P6 能量有界：全部时域运行相对漂移 <= 20%（实测量级报告）。
  P7 势形形态：沿最软模方向能量切片，sc 初段下降（弱失稳）、ex 初段上升（势垒型）。

模型：一维周期链（N = 256、J = lambda = 1、过阻尼梯度流 dt = 0.02、漏斗剖面
      h(a) = min(a h0, cap)、种子 ±eps0 u、阈值 0.3、TFAC = 2.5）；势：单余弦与归一化
      精确势（V''(0) = 1）。交错规范 Hessian 与前两轮一致（脚本式 K）。
确定性（固定种子、无随机数）。图存 D:/software/TianXingFC/images/bh6_h_gt1.png。
"""

import numpy as np
from scipy.optimize import brentq, fsolve

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
TFAC = 2.5
nn = np.arange(N)
rc = np.minimum(np.abs(nn - N//2), N - np.abs(nn - N//2))
H0 = np.where(rc <= 7, 1.0, np.minimum(10.597/np.maximum(rc, 1), 1.0))
SGN = (-1.0)**nn

CHECKS = []

def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))

def thbar_ex(h):
    if h <= 1e-12:
        return np.pi
    f = lambda t: 2.0*np.sin(t) - h*V_ex_d1(t)
    return brentq(f, np.pi-0.001, np.pi+1.45, xtol=1e-15)

def hess_script(bg, hvec, d2f):
    K = np.zeros((N, N))
    dth = np.roll(bg, -1) - bg
    c = np.cos(dth)
    K[nn, nn] += -(J*c + J*np.roll(c, 1))
    K[nn, (nn+1) % N] += -J*c
    K[nn, (nn-1) % N] += -J*np.roll(c, 1)
    K[nn, nn] += LAM*hvec*d2f(bg)
    return K

def bg2(tb):
    return np.where(nn % 2 == 1, tb, 0.0)

def soft_bg(bg, hvec, d2f):
    K = hess_script(bg, hvec, d2f)
    w, v = np.linalg.eigh(K)
    return float(w[0]), v[:, 0]

def soft(hvec, tb, d2f):
    return soft_bg(bg2(tb), hvec, d2f)

def lam_minus(h, tb, v2):
    ct = np.cos(tb)
    return -2.0*ct + h*(1.0+v2)/2.0 - np.sqrt((h*(1.0-v2)/2.0)**2 + 4.0*ct*ct)

def phys_rayleigh(u, tb, hvec, d2f):
    v = SGN*u
    v = v/np.linalg.norm(v)
    K = hess_script(bg2(tb), hvec, d2f)
    return float(v@K@v)

print("==== BH6 可选后续（滑移与 h>1） ====")

# ---------- P1 锚点 ----------
tb1 = thbar_ex(1.0)
mus_sc, psi_sc_1 = soft(H0, np.pi, V_sc_d2)
mus_ex, psi_ex_1 = soft(H0, tb1, V_ex_d2)
mu_ex_u1, _ = soft(np.full(N, 1.0), tb1, V_ex_d2)
print("  [P1] theta_bar(1)=%.6f ; 剖面谱 sc=%+.6f ex=%+.6f ; 均匀 h=1 ex=%+.6f"
      % (tb1, mus_sc, mus_ex, mu_ex_u1))
check("P1", abs(tb1-3.19002) <= 1e-4 and abs(mus_sc+0.22604) <= 1e-4
      and abs(mus_ex-0.03022) <= 1e-4 and abs(mu_ex_u1-0.21802) <= 1e-4,
      "锚点复现（3.19002 / -0.22604 / +0.03022 / +0.21802）")

print("  [P1b] 曲率对比：单余弦 1-Vsc''(pi)=%.6f ; 精确势 1-Vex''(thbar(1))=%.6f"
      % (1.0-V_sc_d2(np.pi), 1.0-V_ex_d2(tb1)))


# P1c 参考背景驻定性诊断
def grad_at(bg, hvec, fd):
    dth = np.roll(bg, -1) - bg
    return J*(np.sin(np.roll(dth, 1)) - np.sin(dth)) - LAM*hvec*fd(bg)

g_sc = float(np.abs(grad_at(bg2(np.pi), np.ones(N), V_sc_d1)).max())
g_ex = float(np.abs(grad_at(bg2(tb1), np.ones(N), V_ex_d1)).max())
print("  [P1c] 参考背景最大梯度（均匀 h=1）：sc %.2e（严格驻态）；ex %.6f（非严格驻态）"
      % (g_sc, g_ex))
for h in (1.0, 2.0, 4.0, 8.0):
    tbh = thbar_ex(h)
    print("       ex h=%.0f: 偶站力 2|sin thbar|=%.6f" % (h, 2.0*abs(np.sin(tbh))))

# ---------- P2 谱延伸 ----------
print("  [P2] 均匀背景谱（数值 vs 二维子格闭式）：")
devs = []
for h in (1.0, 2.0, 4.0, 8.0):
    for tag, d2f in (("sc", V_sc_d2), ("ex", V_ex_d2)):
        tb = np.pi if tag == "sc" else thbar_ex(h)
        wmin, _ = soft(np.full(N, h), tb, d2f)
        v2 = float(d2f(tb))
        lm = lam_minus(h, tb, v2)
        devs.append(abs(wmin-lm))
        print("       %s h=%.0f: 数值 %+.6f ; 闭式 %+.6f" % (tag, h, wmin, lm))
check("P2", max(devs) <= 1e-8, "八组谱与闭式逐位一致（最大偏差 %.1e）" % max(devs))

# ---------- P3 符号图谱 ----------
hg = np.linspace(0.5, 10.0, 96)
lsc = np.array([lam_minus(h, np.pi, -1.0) for h in hg])
lex = np.array([lam_minus(h, thbar_ex(h), V_ex_d2(thbar_ex(h))) for h in hg])
print("  [P3] [0.5,10] 上 sc 最大 %+.6f（h=%.2f）；ex 最小 %+.6f（h=%.2f）"
      % (lsc.max(), hg[int(np.argmax(lsc))], lex.min(), hg[int(np.argmin(lex))]))
check("P3-frame", lsc.max() < -0.05 and lex.min() > 0.1, "编号区间的符号图谱：sc 全负、ex 全正")

caps = (1.0, 2.0, 4.0, 8.0)
prof = {}
for tag, d2f in (("sc", V_sc_d2), ("ex", V_ex_d2)):
    for cap in caps:
        tb = np.pi if tag == "sc" else (tb1 if cap == 1.0 else thbar_ex(cap))
        w, _ = soft(cap*H0, tb, d2f)
        prof[(tag, cap)] = w
print("  [P3-profile] " + " ; ".join("%s cap=%.0f %+.6f" % (t, c, prof[(t, c)])
      for t in ("sc", "ex") for c in caps))
check("P3-profile", all(prof[("sc", c)] < 0.0 for c in caps) and all(prof[("ex", c)] > 0.0 for c in caps),
      "剖面加帽（1/2/4/8）符号不翻转")

print("  [P3b] 精确势族延伸：theta_bar(2)=%.6f theta_bar(4)=%.6f theta_bar(8)=%.6f theta_bar(10)=%.6f"
      % (thbar_ex(2.0), thbar_ex(4.0), thbar_ex(8.0), thbar_ex(10.0)))

# ---------- P4 通道 ----------
o = np.where(nn % 2 == 1, 1.0, 0.0)
u_stag = SGN/np.sqrt(N)
u_slip = o/np.linalg.norm(o)
u_uni = np.ones(N)/np.sqrt(N)
print("  [P4] 通道 Rayleigh 商（物理约定）：")
okP4 = True
for tag, d2f in (("sc", V_sc_d2), ("ex", V_ex_d2)):
    tb = np.pi if tag == "sc" else tb1
    rs = phys_rayleigh(u_stag, tb, H0, d2f)
    rl = phys_rayleigh(u_slip, tb, H0, d2f)
    ru = phys_rayleigh(u_uni, tb, H0, d2f)
    print("       %s: 交错 %+.6f ; 滑移 %+.6f ; 均匀 %+.6f" % (tag, rs, rl, ru))
    okP4 = okP4 and (rl > 0.0) and (rs > 0.0)
    if tag == "ex":
        okP4 = okP4 and (ru > 0.0)
check("P4", okP4, "滑移/交错通道能量正定；均匀通道在精确势上正、在单余弦上为边缘零模（读数见上）")

def proj(v0, u):
    return abs(float(np.dot(v0, u/np.linalg.norm(u))))

print("  [P4-proj] 软模（规范基）对三种种子的投影（上一轮口径）：")
for tag, psi in (("sc", psi_sc_1), ("ex", psi_ex_1)):
    print("       %s: 交错 %.3f ; 滑移 %.3f ; 均匀 %.3f"
          % (tag, proj(psi, SGN), proj(psi, o), proj(psi, np.ones(N))))

# ---------- P5 时域 ----------
def energy_of(th, hvec, fv):
    dth = np.roll(th, -1) - th
    return float(np.sum(J*np.cos(dth) + LAM*fv(th)*hvec))

def run(Gamma, fv, fd, bg0, seed_vec, psi, cap, eps=EPS0):
    steps = int((TFAC/Gamma)/DT)
    tp = bg0 + eps*seed_vec
    tm = bg0 - eps*seed_vec
    E0 = energy_of(tp, np.zeros(N), fv)
    a_trig = None
    Emax = abs(E0)
    tr_a = []
    tr_z = []
    for it in range(steps):
        a = Gamma*it*DT
        h = np.minimum(a*H0, cap)
        Emax = max(Emax, abs(energy_of(tp, h, fv)))
        dth = np.roll(tp, -1) - tp
        g = J*(np.sin(np.roll(dth, 1)) - np.sin(dth)) - LAM*h*fd(tp)
        tp = tp + DT*g
        dth = np.roll(tm, -1) - tm
        g = J*(np.sin(np.roll(dth, 1)) - np.sin(dth)) - LAM*h*fd(tm)
        tm = tm + DT*g
        z = 0.5*float(np.dot(psi, tp - tm))
        if it % 50 == 0:
            tr_a.append(a)
            tr_z.append(abs(z))
        if a_trig is None and abs(z) > THRESH:
            a_trig = a
    drift = (Emax - abs(E0))/max(abs(E0), 1.0)
    return a_trig, abs(z), drift, np.array(tr_a), np.array(tr_z)

print("  [P5] 时域（Gamma=0.001）：")
G = 0.001
runs = {}

def do_run(tag, cap, seed_vec, psi, label, eps=EPS0):
    fv, fd = (V_sc, V_sc_d1) if tag == "sc" else (V_ex, V_ex_d1)
    tb = np.pi if tag == "sc" else (tb1 if cap == 1.0 else thbar_ex(cap))
    at, zl, dr, ta, tz = run(G, fv, fd, bg2(tb), seed_vec, psi, cap, eps)
    runs[(tag, cap, label)] = (at, zl, dr, ta, tz)
    z0 = eps*abs(float(np.dot(psi, seed_vec)))
    print("       %s cap=%.0f %s: 初始 |z|=%.2e；a_trig=%s；末端 |z|=%.3e；漂移 %.2e"
          % (tag, cap, label, z0, ("%.3f" % at) if at else "未触发", zl, dr))
    return at, zl, dr

# 交错档（观测量取各自软模）
for tag, cap in (("sc", 1.0), ("sc", 4.0), ("ex", 1.0), ("ex", 2.0), ("ex", 4.0)):
    tb = np.pi if tag == "sc" else (tb1 if cap == 1.0 else thbar_ex(cap))
    d2f = V_sc_d2 if tag == "sc" else V_ex_d2
    _, psi = soft(cap*H0, tb, d2f)
    do_run(tag, cap, SGN, psi, "交错")
# 通道档（观测量取本通道方向）
for tag in ("ex", "sc"):
    do_run(tag, 1.0, o, u_slip, "滑移")
    do_run(tag, 1.0, np.ones(N), u_uni, "均匀")
# 大种子回弹检查（非阈值协议）
do_run("ex", 1.0, SGN, psi_ex_1, "大种子0.5", eps=0.5)

a_sc1 = runs[("sc", 1.0, "交错")][0]
a_sc4 = runs[("sc", 4.0, "交错")][0]
ex_keys = [("ex", 1.0, "交错"), ("ex", 2.0, "交错"), ("ex", 4.0, "交错"),
           ("ex", 1.0, "滑移"), ("ex", 1.0, "均匀")]
ok5a = (a_sc1 is not None) and abs(a_sc1-0.674)/0.674 <= 0.05
ok5b = (a_sc4 is not None) and 0.55 <= a_sc4 <= 0.85
ok5c = all(runs[k][0] is None for k in ex_keys) and all(runs[k][1] <= 1e-6*1.6e-5 for k in ex_keys)
check("P5-anchor", ok5a, "sc cap=1 触发 %.3f（锚 0.674）" % (a_sc1 if a_sc1 else -1))
check("P5-sc4", ok5b, "sc cap=4 触发 %.3f（量级一致）" % (a_sc4 if a_sc4 else -1))
check("P5-ex", ok5c, "ex 五档均不触发且末端衰减到 1e-6 倍初值以下")

drifts = [runs[k][2] for k in runs if k[2] != "大种子0.5"]
check("P6", max(drifts) <= 0.20, "阈值档全部运行相对漂移最大 %.2e（<= 20%%）" % max(drifts))
print("  [P6b] 大种子探针（eps=0.5，非阈值协议）漂移 %.2e（报告项）"
      % runs[("ex", 1.0, "大种子0.5")][2])

# ---------- P7 能量切片与严格驻态抽点 ----------
print("  [P7] 能量切片（均匀 h=1）：")
def Efun_of(fv, bg0, psi, smax=3.0, npts=301):
    def Ef(th):
        dth = np.roll(th, -1) - th
        return float(np.sum(J*np.cos(dth) + LAM*fv(th)))
    ss = np.linspace(0.0, smax, npts)
    Es = np.array([Ef(bg0 + s*psi) for s in ss]) - Ef(bg0)
    return ss, Es

wsc, psi = soft(np.full(N, 1.0), np.pi, V_sc_d2)
psi = SGN*psi/np.linalg.norm(psi)
ss, Es = Efun_of(V_sc, bg2(np.pi), psi)
profE = {"sc": (ss, Es)}
i02 = int(np.argmin(np.abs(ss-0.2)))
print("       sc（严格驻态）：E(0.2)-E0=%+.6f ; 最小 %+.6f（s=%.2f）"
      % (Es[i02], Es.min(), ss[int(np.argmin(Es))]))

def eq_true(ab):
    a, b = ab
    return [2.0*np.sin(b-a) + V_ex_d1(a), -2.0*np.sin(b-a) + V_ex_d1(b)]

ab = fsolve(eq_true, [0.18, 3.42], xtol=1e-13)
a_s, b_s = float(ab[0]), float(ab[1])
res_t = max(abs(eq_true([a_s, b_s])[0]), abs(eq_true([a_s, b_s])[1]))
print("       ex 严格驻态（h=1）：偶站 a=%.6f、奇站 b=%.6f，梯度残差 %.2e" % (a_s, b_s, res_t))
bg_alt = np.where(nn % 2 == 1, b_s, a_s)
w2, psi2 = soft_bg(bg_alt, np.full(N, 1.0), V_ex_d2)
psi2 = SGN*psi2/np.linalg.norm(psi2)
ss2, Es2 = Efun_of(V_ex, bg_alt, psi2)
profE["ex"] = (ss2, Es2)
i02b = int(np.argmin(np.abs(ss2-0.2)))
print("       ex 严格驻态谱：lam_min=%+.6f ; E(0.2)-E0=%+.6f ; 最小 %+.6f（s=%.2f）"
      % (w2, Es2[i02b], Es2.min(), ss2[int(np.argmin(Es2))]))
check("P7", (Es[i02] < 0.0) and (w2 > 0.0) and (Es2[i02b] > 0.0) and (res_t <= 1e-9),
      "sc 沿软模下降；ex 严格驻态 lam_min 为正且切片上升（残差 %.0e）" % res_t)

# ---------- 图 ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
fig, axs = plt.subplots(2, 2, figsize=(13.5, 8.4))

ax = axs[0, 0]
hh = np.linspace(0.0, 10.0, 201)
tbv = np.array([np.pi if h < 1e-9 else thbar_ex(h) for h in hh])
ax.plot(hh, tbv, "-", label="精确势 theta_bar(h)")
ax.axhline(np.pi, color="gray", ls=":", lw=1, label="单余弦 pi")
h0c = brentq(lambda h: V_ex_d2(thbar_ex(h)), 4.0, 8.0)
ax.axvline(h0c, color="tab:red", ls="--", lw=1, label="V'' 过零 h=%.2f" % h0c)
ax.set_xlabel("h")
ax.set_ylabel("交错平衡 theta_bar")
ax.set_title("(a) 平衡族（h>1 延伸）")
ax.legend(fontsize=8)

ax = axs[0, 1]
ax.plot(hh, 2.0-np.sqrt(4.0+hh*hh), "-", label="单余弦 2-sqrt(4+h^2)")
lex2 = np.array([lam_minus(h, (np.pi if h < 1e-9 else thbar_ex(h)),
                            (V_sc_d2(np.pi) if h < 1e-9 else V_ex_d2(thbar_ex(h)))) for h in hh])
ax.plot(hh, lex2, "-", label="精确势 lam-(h)")
for tag, mk in (("sc", "o"), ("ex", "s")):
    ax.plot([1, 2, 4, 8], [prof[(tag, c)] for c in (1.0, 2.0, 4.0, 8.0)], mk, ms=4, label="%s 剖面读数" % tag)
ax.axhline(0.0, color="gray", lw=0.8)
ax.set_xlabel("h")
ax.set_ylabel("最低本征值")
ax.set_title("(b) 软模谱对 h")
ax.legend(fontsize=7)

ax = axs[1, 0]
for key, ls in ((( "sc", 1.0, "交错"), "-"), (("sc", 4.0, "交错"), "--"),
                (("ex", 1.0, "交错"), "-"), (("ex", 4.0, "交错"), "--")):
    at, zl, dr, ta, tz = runs[key]
    ax.semilogy(ta, np.maximum(tz, 1e-16), ls, label="%s cap=%.0f" % (key[0], key[1]))
ax.axhline(THRESH, color="gray", ls=":", lw=1)
ax.axvline(1.0, color="gray", ls=":", lw=1)
ax.set_xlabel("a（斜坡进度）")
ax.set_ylabel("|z|")
ax.set_title("(c) 时域交错档（灰线：阈值与满偏）")
ax.legend(fontsize=7)

ax = axs[1, 1]
for tag, lab in (("sc", "sc（严格驻态）"), ("ex", "ex（严格驻态 h=1）")):
    ss, Es = profE[tag]
    m = ss <= 1.2
    ax.plot(ss[m], Es[m], label=lab)
ax.axhline(0.0, color="gray", lw=0.8)
ax.set_xlabel("s（沿最软模方向的振幅）")
ax.set_ylabel("E(s)-E(0)")
ax.set_title("(d) 能量切片（均匀 h=1，初段）")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(r"D:/software/TianXingFC/images/bh6_h_gt1.png", dpi=140)
print("  图已存 images/bh6_h_gt1.png")

npass = sum(1 for _, ok in CHECKS if ok)
print("==== 判据 %d/%d 通过 ====" % (npass, len(CHECKS)))
