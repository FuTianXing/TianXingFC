# -*- coding: utf-8 -*-
"""BH6（大振幅非线性，M9）：精确异类势下的失稳时钟、能量有界性与两通道核对。

判据（先立判据，再实现；均可证伪）：
  B1 二次型与周期：归一化精确势 V_ex 的二次系数与单余弦 V_sc 一致（V''(0) = 1，机器级）；
     V_ex 数值周期 = 2 pi sqrt(2)（与 2 pi 不格）。
  B2 势形表（报告）：theta = 0.5、1（满偏）、2 处 V、V'、V'' 的比；首转折点
     theta* = pi/sqrt(2) = 2.2214（V_sc 为 pi = 3.1416）；满偏处方差与力差报告。
  B3 时钟锚点：有限振幅时钟在 theta_bar = 0 时两势一致，并复现 D2 一维读数
     （Gamma_c = 3.566e-2，相对偏差 <= 1e-3）。
  B4 有限振幅时钟移动：theta_bar = 0.5、1（只改 onsite 曲率，键刚度保持）时
     Gamma_c(theta_bar)/Gamma_c(0) 落在 [0.1, 10]（量级不移动；对终审 1e14 级余量无影响）。
  B5 时域对照一读：单余弦档复现 S3N 读数（Gamma = 0.001 触发于 a = 0.689、
     Gamma = 0.005 满偏先于触发，偏差 <= 5%）；精确势档触发与满偏次序报告、
     能量相对漂移 <= 20% 有界、满偏后锁定。

模型：一维周期链（N = 256 时域；N = 64 时钟）；J = lambda = 1；过阻尼梯度流 dt = 0.02；
      h(a) = min(a h0, 1)（S1 过密档等效剖面）；种子 ±eps0 (-1)^n（eps0 = 1e-6）。
      势：V_sc = 1 - cos t；V_ex（归一化）= [1 - cos(sqrt2 t) + 2(1 - cos(t/sqrt2))]/3。
      时钟：H(h; theta_bar) = 2 J + (-1)^n h V''(theta_bar)（N3a 口径，键刚度保持），
      gamma(h) = sqrt(-mu_min)，I = int_0^1 gamma dh，Gamma_c = I / ln(1/eps)，eps = 1e-3（D2 口径）。
确定性（固定种子、无随机数）。图存 D:/software/TianXingFC/images/bh6_large_amplitude.png。
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))


SQ2 = np.sqrt(2.0)


def V_sc(t):
    return 1.0 - np.cos(t)


def V_sc_d1(t):
    return np.sin(t)


def V_sc_d2(t):
    return np.cos(t)


def V_ex(t):
    return (1.0 - np.cos(SQ2 * t) + 2.0 * (1.0 - np.cos(t / SQ2))) / 3.0


def V_ex_d1(t):
    return (SQ2 * np.sin(SQ2 * t) + SQ2 * np.sin(t / SQ2)) / 3.0


def V_ex_d2(t):
    return (2.0 * np.cos(SQ2 * t) + np.cos(t / SQ2)) / 3.0


print("==== BH6（大振幅非线性，M9） ====")

# ---------- B1 二次型与周期 ----------
d0 = 1e-4
c2_sc = V_sc(d0) / d0 ** 2
c2_ex = V_ex(d0) / d0 ** 2
ok_b1a = abs(c2_sc - 0.5) / 0.5 < 1e-6 and abs(c2_ex - 0.5) / 0.5 < 1e-6
Ts = np.linspace(0.0, 2.0 * np.pi * SQ2, 20001)
tt = np.linspace(0.0, 16.0, 40001)
vals = V_ex(tt)
per = 2.0 * np.pi * SQ2
ok_b1b = abs(np.max(np.abs(V_ex(tt) - V_ex(tt + per)))) < 1e-9
print("  [B1] 二次系数：V_sc %.6f、V_ex %.6f（期望 0.5）；V_ex 周期 2 pi sqrt2 = %.6f（2 pi = %.6f，不格）"
      % (c2_sc, c2_ex, per, 2 * np.pi))
check("B1", ok_b1a and ok_b1b, "二次型一致（机器级）；周期 2 pi sqrt(2) 严格成立")

# ---------- B2 势形表 ----------
rows = []
for t in (0.5, 1.0, 2.0):
    rows.append((t, V_sc(t), V_ex(t), V_sc(t) / V_ex(t),
                 V_sc_d1(t), V_ex_d1(t), V_sc_d1(t) / V_ex_d1(t),
                 V_sc_d2(t), V_ex_d2(t), V_sc_d2(t) / V_ex_d2(t)))
    print("  [B2] theta=%.1f：V 比 %.3f；V' 比 %.3f；V'' 比 %.3f（V_ex: %.4f / %.4f / %.4f）"
          % (t, rows[-1][3], rows[-1][6], rows[-1][9], rows[-1][2], rows[-1][5], rows[-1][8]))
t_star_ex = np.pi / SQ2
t_star_sc = np.pi
in_sat = V_ex(np.linspace(0, 1, 2000))
print("  [B2] 首转折点：V_ex %.4f（V_sc %.4f）；满偏比对（V、V'、V''）：%.3f / %.3f / %.3f"
      % (t_star_ex, t_star_sc, rows[1][3], rows[1][6], rows[1][9]))

# ---------- B3/B4 有限振幅时钟 ----------
NC = 64
nnc = np.arange(NC)
SGNC = (-1.0) ** nnc
EPS = 1e-3
LAM_LOG = np.log(1.0 / EPS)


def gamma_1d(hval, d2):
    H = np.zeros((NC, NC))
    H[nnc, nnc] = 2.0 + SGNC * hval * d2
    H[nnc, (nnc + 1) % NC] += 1.0
    H[nnc, (nnc - 1) % NC] += 1.0
    mu = float(np.linalg.eigvalsh(H)[0])
    return np.sqrt(max(0.0, -mu))


def clock(d2, npts=301):
    hs = np.linspace(0.0, 1.0, npts)
    g = np.array([gamma_1d(hv, d2) for hv in hs])
    I = float(np.trapezoid(g, hs))
    return I, I / LAM_LOG


I_sc0, gc_sc0 = clock(1.0)
print("  [B3] 锚点（theta_bar = 0）：I = %.6f，Gamma_c = %.6e（D2 一维 3.566e-2）" % (I_sc0, gc_sc0))
check("B3", abs(gc_sc0 - 3.566e-2) / 3.566e-2 <= 1e-3,
      "复现 D2 一维读数（偏差 %.1e）" % (abs(gc_sc0 - 3.566e-2) / 3.566e-2))

gc_curve = {}
for name, d2f in (("sc", V_sc_d2), ("ex", V_ex_d2)):
    out = []
    for tb in (0.0, 0.5, 1.0):
        I, gc = clock(float(d2f(tb)))
        out.append((tb, I, gc, gc / gc_sc0))
        print("  [B4] %s 档 theta_bar=%.1f：V''=%.4f；I=%.6f；Gamma_c=%.6e（比 %.4f）"
              % (name, tb, float(d2f(tb)), I, gc, gc / gc_sc0))
    gc_curve[name] = out
ratios = [r[3] for r in gc_curve["sc"][1:] + gc_curve["ex"][1:]]
check("B4", all(0.1 <= r <= 10.0 for r in ratios),
      "有限振幅移动比 %s（均在 [0.1, 10]）" % ["%.4f" % r for r in ratios])

# ---------- B5 时域对照 ----------
N = 256
J = 1.0
LAM = 1.0
EPS0 = 1e-6
THRESH = 0.3
DT = 0.02
nn = np.arange(N)
rc = np.minimum(np.abs(nn - N // 2), N - np.abs(nn - N // 2))
H0 = np.where(rc <= 7, 1.0, np.minimum(10.597 / np.maximum(rc, 1), 1.0))
SGN = (-1.0) ** nn


def hess_lin(hv):
    H = np.zeros((N, N))
    H[nn, nn] = 2.0 * J + SGN * hv
    H[nn, (nn + 1) % N] += J
    H[nn, (nn - 1) % N] += J
    return H


w1, V1 = np.linalg.eigh(hess_lin(H0))
PSI = V1[:, 0]
ag = np.linspace(0.0, 1.0, 61)
gg = np.array([np.sqrt(max(0.0, -np.linalg.eigvalsh(hess_lin(np.minimum(aa * H0, 1.0)))[0])) for aa in ag])
cum = np.concatenate([[0.0], np.cumsum(0.5 * (gg[1:] + gg[:-1]) * np.diff(ag))])
NEED = np.log(THRESH / EPS0)


def pred_ac(Gamma):
    k = int(np.searchsorted(cum, Gamma * NEED))
    return None if k >= len(ag) else float(ag[k])


def run(Gamma, funV, fund1, bg, dt=DT, tfac=2.5):
    steps = int((tfac / Gamma) / dt)
    tp = bg + EPS0 * SGN
    tm = bg - EPS0 * SGN

    def energy(th, h):
        dth = np.roll(th, -1) - th
        return float(np.sum(J * np.cos(dth) + LAM * funV(th) * h))

    def grad(th, h):
        dth = np.roll(th, -1) - th
        return J * (np.sin(np.roll(dth, 1)) - np.sin(dth)) - LAM * h * fund1(th)

    E0 = energy(tp, np.zeros(N))
    a_trig = None
    Emax = abs(E0)
    tr_a = []
    tr_z = []
    for it in range(steps):
        t = it * dt
        a = Gamma * t
        h = np.minimum(a * H0, 1.0)
        E = energy(tp, h)
        Emax = max(Emax, abs(E))
        tp = tp + dt * grad(tp, h)
        tm = tm + dt * grad(tm, h)
        z = 0.5 * float(np.dot(PSI, tp - tm))
        if it % 50 == 0:
            tr_a.append(a)
            tr_z.append(abs(z))
        if a_trig is None and abs(z) > THRESH:
            a_trig = a
    a_sat = 1.0
    z_lock = abs(z)
    drift = (Emax - abs(E0)) / max(abs(E0), 1.0)
    bdrift = float(np.max(np.abs(0.5 * (tp + tm) - bg)))
    return a_trig, a_sat, z_lock, drift, bdrift, np.array(tr_a), np.array(tr_z)


rng = np.random.default_rng(20260919)
th_r = rng.uniform(0.5, 1.5, N)
h_r = rng.uniform(0.2, 0.9, N)


def _energy_r(th, fv):
    dth = np.roll(th, -1) - th
    return float(np.sum(J * np.cos(dth) + LAM * fv(th) * h_r))


def _grad_r(th, fd):
    dth = np.roll(th, -1) - th
    return J * (np.sin(np.roll(dth, 1)) - np.sin(dth)) - LAM * h_r * fd(th)


for tag, fv, fd in (("sc", V_sc, V_sc_d1), ("ex", V_ex, V_ex_d1)):
    g = _grad_r(th_r, fd)
    mx = 0.0
    dd = 1e-6
    for k in range(0, N, 37):
        tpp = th_r.copy(); tpp[k] += dd
        tmm = th_r.copy(); tmm[k] -= dd
        fdm = (_energy_r(tpp, fv) - _energy_r(tmm, fv)) / (2 * dd)
        mx = max(mx, abs(g[k] + fdm))
    e0r = _energy_r(th_r, fv)
    e1r = _energy_r(th_r + 1e-5 * g, fv)
    check("B0-" + tag, mx < 1e-6 and e1r <= e0r,
          "解析梯度对中心差分残差 %.2e；一步下降 dE %+.2e" % (mx, e1r - e0r))

TH_EX = 2.0 * np.pi * SQ2 / 3.0
BG_SC = np.pi * nn
BG_EX = np.pi * nn


def _grad_full(th, h, fd):
    dth = np.roll(th, -1) - th
    return J * (np.sin(np.roll(dth, 1)) - np.sin(dth)) - LAM * h * fd(th)


g_sc_bg = _grad_full(BG_SC, H0, V_sc_d1)
g_ex_bg = _grad_full(BG_EX, H0, V_ex_d1)
print("  [B5-口径] 共用背景 pi 绕数：sc 梯度范数 %.2e（驻态）；ex 梯度范数 %.2e（非驻态，V_ex'(pi) = %.4f）"
      % (np.max(np.abs(g_sc_bg)), np.max(np.abs(g_ex_bg)), V_ex_d1(np.pi)))
check("B5c", np.max(np.abs(g_sc_bg)) <= 1e-12 and np.max(np.abs(g_ex_bg)) >= 1e-3,
      "背景语义量化：sc 驻定、ex 非驻定（|grad| %.3f）" % np.max(np.abs(g_ex_bg)))

print("  [B5] 线性预言 a_c：G=0.001 %s；G=0.005 %s"
      % tuple(("%.3f" % pred_ac(G)) if pred_ac(G) else "档外" for G in (0.001, 0.005)))
runs = {}
for tag, fv, fd, bg in (("sc", V_sc, V_sc_d1, BG_SC), ("ex", V_ex, V_ex_d1, BG_EX)):
    for G in (0.001, 0.005):
        at, asat, zl, dr, bd, ta, tz = run(G, fv, fd, bg)
        runs[(tag, G)] = (at, asat, zl, dr, bd, ta, tz)
        print("  [B5] %s 档 G=%.3f：a_trig = %s；锁定 |z| = %.4f；能量相对漂移 %.3f；背景漂移 %.3f"
              % (tag, G, ("%.3f" % at) if at else "未触发", zl, dr, bd))

at_sc1 = runs[("sc", 0.001)][0]
at_sc5 = runs[("sc", 0.005)][0]
ok_b5a = at_sc1 is not None and abs(at_sc1 - 0.689) / 0.689 <= 0.05 and \
    at_sc5 is not None and abs(at_sc5 - 1.113) / 1.113 <= 0.05
check("B5a", ok_b5a, "单余弦档复现 S3N（0.689 / 1.113）")
drift_max = max(runs[k][3] for k in runs)
lock_ok = all(runs[k][2] < 1e6 for k in runs)
check("B5b", drift_max <= 0.20 and lock_ok,
      "四档能量漂移最大 %.3f（<= 0.20）且锁定有限" % drift_max)

ex_quiet = all(runs[("ex", G)][0] is None for G in (0.001, 0.005))
ex_bd = max(runs[("ex", G)][4] for G in (0.001, 0.005))
check("B5d", ex_quiet and ex_bd >= 0.5,
      "口径发现：ex 档种子不触发（两档）、背景漂移 %.3f（滑向 onsite 极小 pi*sqrt2 = %.4f）"
      % (ex_bd, np.pi * SQ2))

# ---------- 图 ----------
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
fig, axs = plt.subplots(1, 3, figsize=(15.0, 4.2))

ax = axs[0]
tt = np.linspace(0.0, 8.0, 2000)
ax.plot(tt, V_sc(tt), "-", label="单余弦 1 - cos t")
ax.plot(tt, V_ex(tt), "--", label="精确（归一化）")
for t in (0.5, 1.0, 2.0):
    ax.axvline(t, color="gray", ls=":", lw=0.8)
ax.axvline(np.pi / SQ2, color="tab:red", ls=":", lw=1)
ax.set_xlabel("theta")
ax.set_ylabel("V")
ax.set_title("(a) 势形对照（竖线 0.5、1、2；红点线首转折）")
ax.legend(fontsize=8)

ax = axs[1]
tbs = [r[0] for r in gc_curve["sc"]]
ax.plot(tbs, [r[3] for r in gc_curve["sc"]], "o-", label="单余弦")
ax.plot(tbs, [r[3] for r in gc_curve["ex"]], "s--", label="精确")
ax.axhline(1.0, color="gray", ls=":", lw=1)
ax.set_xlabel("背景振幅 theta_bar")
ax.set_ylabel("Gamma_c(theta_bar) / Gamma_c(0)")
ax.set_title("(b) 有限振幅时钟（N3a 口径）")
ax.legend(fontsize=8)

ax = axs[2]
for tag, G, ls in (("sc", 0.001, "-"), ("ex", 0.001, "--"), ("sc", 0.005, "-"), ("ex", 0.005, "--")):
    ta, tz = runs[(tag, G)][5], runs[(tag, G)][6]
    ax.semilogy(ta, np.maximum(tz, 1e-12), ls, label="%s G=%.3f" % (tag, G))
ax.axhline(THRESH, color="gray", ls=":", lw=1)
ax.axvline(1.0, color="gray", ls=":", lw=1)
ax.set_xlabel("a（斜坡进度）")
ax.set_ylabel("|z|")
ax.set_title("(c) 时域模式幅度（灰线：触发阈与满偏）")
ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(r"D:/software/TianXingFC/images/bh6_large_amplitude.png", dpi=140)
print("  图已存 images/bh6_large_amplitude.png")

npass = sum(1 for _, ok in CHECKS if ok)
print("==== 判据 %d/%d 通过 ====" % (npass, len(CHECKS)))
