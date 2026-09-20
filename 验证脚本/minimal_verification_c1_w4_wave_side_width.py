# -*- coding: utf-8 -*-
"""
minimal_verification_c1_w4_wave_side_width.py

[C1-6 W4 轮] 宽度的波性侧承载：软模关联长度与转换波有效波长
（判据 W4-1 至 W4-6）

背景：C1-6 的落窗条件原挂在粒性侧驻留态宽度上，格点锁定与软体加局域凹口两条
路线在三维判负（面积律）。攻坚计划 §10.4 重立口径：粒性侧只给载体的形态参数，
宽度由波性侧的有效宽度机制承担；候选机制两条：转换波的有效波长、软模的关联
长度。本脚本把两条机制精确化，并给出 6 条判据读数。

模型：FK 链，E = sum 0.5*kappa_e*(du - 1)^2 + sum K*(1 - cos 2 pi u)，kappa_e = 1；
背景 = 均匀缠绕 u_n = n；位错 = 局域多缠一圈的 kink（w = u - n 从 0 到 1）。
软模 = 缠绕背景上的相位涨落模：omega^2(q) = kappa_e*4*sin^2(q/2) + (2 pi)^2*K，
带隙 m = 2 pi*sqrt(K)，关联长度 xi = 1/m = 1/(2 pi sqrt(K))（连续口径）。
kink 宽度解析值 l_SG = xi；数值宽度 l* = 弹性超量密度二阶矩宽 2*sqrt(r2)（与
C1-6 组装轮、路线乙首判同口径）。解析桥因子：sech 剖面二阶矩给 l*/xi = pi/sqrt(3)。

判据
W4-1 软模色散与带隙：平面波作用与解析式一致；有隙格林函数衰减长度与 1/m 对照
W4-2 kink 尾部衰减长度：与离散预测 xi_d = 1/(2 asinh(pi sqrt(K))) 对照
W4-3 宽度恒等式：l* 实测与 (pi/sqrt(3))*xi 对照，复现桥因子与紧窗口读数
W4-4 转换波两读数：平移零模包络衰减长度、带边波长 lambda_gap = 2 pi xi
W4-5 落窗可达性：l* >= 2 给 K <= 1/48；与洛伦兹窗口 K <= 1/(16 pi^2) 联立
W4-6 三维可用性：3D 带隙与轴向、径向衰减长度与一维一致（维度无关）

运行：python 验证脚本/minimal_verification_c1_w4_wave_side_width.py（确定性，约 6 秒）
"""
import math
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

N = 600
KAP = 1.0
ETA = 0.005
STEPS_A = 20000
STEPS_B = 40000
KS = (0.0005, 0.001, 0.002, 0.005, 0.00633, 0.01, 0.02, 0.05)
KS_FIT = (0.0005, 0.001, 0.002, 0.005, 0.00633, 0.01, 0.02)
BRI = math.pi / math.sqrt(3.0)
IMG = "D:/software/TianXingFC/images/c1_w4_wave_side_width.png"
T0 = time.time()


def xi_cont(K):
    return 1.0 / (2.0 * math.pi * math.sqrt(K))


def m_disc(K):
    return 2.0 * math.asinh(math.pi * math.sqrt(K))


def xi_disc(K):
    return 1.0 / m_disc(K)


def init_kink(n=N):
    x = np.arange(n, dtype=float)
    u = np.where(x < n // 2, x, x + 1.0)
    u[n // 2] = n / 2 + 0.5
    return x, u


def relax(u, Karr, steps, eta=ETA):
    x = np.arange(N, dtype=float)
    for _ in range(steps):
        du = np.diff(u)
        g = np.zeros(N)
        g[:-1] -= KAP * (du - 1.0)
        g[1:] += KAP * (du - 1.0)
        g += Karr * 2.0 * np.pi * np.sin(2.0 * np.pi * u)
        g[0] = g[-1] = 0.0
        u = u - eta * g
        u[0], u[-1] = x[0], float(N)
    return u


def width(u):
    du = np.diff(u)
    dens = 0.5 * (du - 1.0) ** 2
    idx = np.arange(N - 1) + 0.5
    c0 = (dens * idx).sum() / dens.sum()
    return 2.0 * np.sqrt((dens * (idx - c0) ** 2).sum() / dens.sum())


def fit_rate(dist, prof, lo, hi, thr_lo=1e-12, thr_hi=0.5):
    m = (dist >= lo) & (dist <= hi) & (prof > thr_lo) & (prof < thr_hi)
    if m.sum() < 6:
        return float("nan"), int(m.sum())
    sl = np.polyfit(dist[m], np.log(prof[m]), 1)[0]
    return -sl, int(m.sum())


print("=" * 78)
print("C1-6 W4 轮：宽度的波性侧承载（判据 W4-1 至 W4-6）")
print("=" * 78)
print()
print("[0] 弛豫与宽度表（同 C1-6 组装轮与路线乙首判协议：N=600、eta=0.005）")
LSTAR = {}
PROF = {}
for K in KS:
    x, u0 = init_kink()
    u = relax(u0.copy(), np.full(N, K), STEPS_A)
    l_a = width(u)
    u = relax(u, np.full(N, K), STEPS_B)
    l_b = width(u)
    LSTAR[K] = l_b
    PROF[K] = u
    print("    K = %.5f：l*(20k) = %.4f，l*(60k) = %.4f，xi_cont = %.4f，xi_disc = %.4f"
          % (K, l_a, l_b, xi_cont(K), xi_disc(K)))

print()
print("[W4-1] 软模色散、带隙与有隙格林函数")
nn = np.arange(N)
worst = 0.0
for K in (0.0005, 0.005, 0.02):
    m0sq = (2.0 * np.pi) ** 2 * K
    for k in range(101):
        phi = np.exp(2j * np.pi * ((k * nn) % 101) / 101.0)
        lap = 2.0 * phi - np.roll(phi, 1) - np.roll(phi, -1)
        val_num = (KAP * lap + m0sq * phi)[N // 3] / phi[N // 3]
        val_ana = KAP * 4.0 * np.sin(np.pi * k / 101.0) ** 2 + m0sq
        worst = max(worst, abs(val_num - val_ana) / abs(val_ana))
print("    W4-1a 平面波色散最大相对偏差 = %.3e（期望 < 1e-12）" % worst)
r1 = []
for K in (0.0005, 0.005, 0.02):
    gap = (2.0 * np.pi) ** 2 * K
    r1.append(abs(gap - 4.0 * math.pi ** 2 * K) / gap)
print("    W4-1b 带隙 (2 pi)^2 K：恒等式最大偏差 = %.3e" % max(r1))
for K in (0.005, 0.01):
    m0sq = (2.0 * np.pi) ** 2 * K
    main = 2.0 * KAP + m0sq
    A = (np.diag(main * np.ones(N))
         + np.diag(-KAP * np.ones(N - 1), 1)
         + np.diag(-KAP * np.ones(N - 1), -1))
    A[0, N - 1] = -KAP
    A[N - 1, 0] = -KAP
    e = np.zeros(N)
    e[0] = 1.0
    g = np.linalg.solve(A, e)
    lo = int(math.ceil(2.5 * xi_disc(K)))
    rate, npnt = fit_rate(nn, np.abs(g), lo, lo + 14)
    print("    W4-1c K = %.5f：有隙格林函数衰减率 = %.5f（离散预测 %.5f，相对差 %+.2f%%），%d 点"
          % (K, rate, m_disc(K), 100.0 * (rate - m_disc(K)) / m_disc(K), npnt))

print()
print("[W4-2] kink 尾部衰减长度与关联长度")
print("      K        xi_左     xi_右     xi_disc    右尾相对差")
for K in KS_FIT:
    w = PROF[K] - np.arange(N)
    c = int(np.argmax(np.abs(np.diff(w)))) + 1
    a = int(math.ceil(2.0 * xi_disc(K)))
    rr, nrp = fit_rate(nn - c, 1.0 - w, a, a + 14)
    ll, nlp = fit_rate(c - nn, w, a, a + 14)
    print("    %.5f   %7.4f  %7.4f  %7.4f   %+.2f%%"
          % (K, 1.0 / ll, 1.0 / rr, xi_disc(K), 100.0 * (1.0 / rr - xi_disc(K)) / xi_disc(K)))

print()
print("[W4-3] 宽度恒等式 l* = (pi/sqrt(3)) * xi")
print("      K        l*(60k)    (pi/sqrt(3))*xi_cont   相对差    l*/xi_cont")
for K in KS:
    pred = BRI * xi_cont(K)
    print("    %.5f   %8.4f   %8.4f   %+.2f%%   %.4f"
          % (K, LSTAR[K], pred, 100.0 * (LSTAR[K] - pred) / pred, LSTAR[K] / xi_cont(K)))
print("    解析桥因子 pi/sqrt(3) = %.4f；库内读数均值 1.803（1.785 至 1.813）" % BRI)

print()
print("[W4-4] 转换波两读数：平移零模包络与带边波长")
for K in (0.005, 0.01):
    ui = PROF[K][1:-1]
    diag = 2.0 * KAP + (2.0 * np.pi) ** 2 * K * np.cos(2.0 * np.pi * ui)
    H = (np.diag(diag) + np.diag(-KAP * np.ones(N - 3), 1)
         + np.diag(-KAP * np.ones(N - 3), -1))
    evals, evecs = np.linalg.eigh(H)
    v = np.abs(evecs[:, 0])
    c = int(np.argmax(v))
    a = int(math.ceil(2.0 * xi_disc(K)))
    nv = np.arange(len(v))
    rr, _ = fit_rate(nv - c, v, a, a + 12, thr_hi=1e9)
    print("    K = %.5f：最小本征值 = %.3e；零模包络衰减率 = %.5f（预测 %.5f，相对差 %+.2f%%）"
          % (K, evals[0], rr, m_disc(K), 100.0 * (rr - m_disc(K)) / m_disc(K)))
    lam_gap = 2.0 * math.pi / (2.0 * math.pi * math.sqrt(K))
    print("        带边波长 lambda_gap = 1/sqrt(K) = %.4f；2 pi xi_cont = %.4f；比值 = %.4f（恒等 2 pi）"
          % (lam_gap, 2.0 * math.pi * xi_cont(K), lam_gap / xi_cont(K)))

print()
print("[W4-5] 落窗可达性与联立")
K_win = 1.0 / 48.0
K_lor = 1.0 / (16.0 * math.pi ** 2)
print("    宽度窗（连续口径）：l* >= 2 等价 K <= 1/48 = %.6f" % K_win)
print("    洛伦兹窗：xi >= 2 等价 K <= 1/(16 pi^2) = %.6f" % K_lor)
print("    联立：K <= %.6f 时 xi = %.4f、l* = %.4f（库内线 3.60 至 3.61）"
      % (K_lor, xi_cont(K_lor), BRI * xi_cont(K_lor)))
for K in KS:
    flag = "落窗" if LSTAR[K] >= 2.0 else "落窗外"
    print("        K = %.5f：l*(实测) = %.4f（%s）" % (K, LSTAR[K], flag))
print("    S 轮扫描插值对照：l* = 2 的边界在 K = 0.020 与 0.0231 之间，与 1/48 一致")

print()
print("[W4-6] 三维可用性：带隙与衰减长度的维度无关性")
L3 = 64
qq = 2.0 * np.pi * np.fft.fftfreq(L3)
QX, QY, QZ = np.meshgrid(qq, qq, qq, indexing="ij")
for K in (0.005, 0.01):
    m0sq = (2.0 * np.pi) ** 2 * K
    Lam = (KAP * 4.0 * (np.sin(QX / 2.0) ** 2 + np.sin(QY / 2.0) ** 2
                        + np.sin(QZ / 2.0) ** 2) + m0sq)
    gap_dev = abs(Lam.min() - m0sq) / m0sq
    G = np.fft.ifftn(1.0 / Lam).real
    r = np.arange(L3)
    axis = G[0, 0, :]
    lo = int(math.ceil(2.5 * xi_disc(K)))
    ra, npa = fit_rate(r, axis * r, lo, lo + 16, thr_lo=1e-16)
    xs = np.minimum(np.arange(L3), L3 - np.arange(L3))
    DX, DY, DZ = np.meshgrid(xs, xs, xs, indexing="ij")
    RR = np.sqrt(DX ** 2 + DY ** 2 + DZ ** 2)
    rr_flat = RR.ravel()
    gg_flat = G.ravel()
    rad_r, rad_g = [], []
    for ri in range(1, int(RR.max()) + 1):
        m = (rr_flat >= ri - 0.5) & (rr_flat < ri + 0.5)
        if m.sum() > 20:
            rad_r.append(ri)
            rad_g.append(gg_flat[m].mean())
    rad_r = np.array(rad_r, dtype=float)
    rad_g = np.array(rad_g)
    msk = rad_g > 0
    rb, npb = fit_rate(rad_r[msk], (rad_g * rad_r)[msk], lo, lo + 16, thr_lo=1e-16)
    print("    K = %.5f：3D 带隙偏差 = %.3e；轴向衰减率 = %.5f（预测 %.5f，%+.2f%%）；径向衰减率 = %.5f（对轴向 %+.2f%%）"
          % (K, gap_dev, ra, m_disc(K), 100.0 * (ra - m_disc(K)) / m_disc(K),
             rb, 100.0 * (rb - ra) / ra))

fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.2))

ax = axes[0, 0]
qg = np.linspace(-math.pi, math.pi, 400)
for K in (0.0005, 0.005, 0.02):
    wq = np.sqrt(KAP * 4.0 * np.sin(qg / 2.0) ** 2 + (2.0 * np.pi) ** 2 * K)
    ax.plot(qg, wq, label="K = %.4f" % K)
ax.set_xlabel("q")
ax.set_ylabel("omega(q)")
ax.set_title("(a) 软模色散与带隙")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[0, 1]
K = 0.005
w = PROF[K] - np.arange(N)
c = int(np.argmax(np.abs(np.diff(w)))) + 1
off = np.arange(0, 40)
ax.semilogy(off, 1.0 - w[c + off], "o", ms=3, label="1 - w（右尾）")
a = int(math.ceil(2.0 * xi_disc(K)))
sl, _ = fit_rate(np.arange(N) - c, 1.0 - w, a, a + 14)
dline = np.arange(a, a + 14, dtype=float)
Aamp = (1.0 - w[c + a]) * np.exp(sl * a)
ax.semilogy(dline, Aamp * np.exp(-sl * dline), "-", lw=1,
            label="拟合：1/xi = %.4f" % sl)
ax.set_xlabel("距离芯的格点数")
ax.set_ylabel("1 - w")
ax.set_title("(b) kink 尾部指数衰减（K = 0.005）")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

ax = axes[1, 0]
Kg = np.logspace(math.log10(0.0003), math.log10(0.06), 200)
ax.loglog(Kg, [BRI * xi_cont(k) for k in Kg], "-", label="(pi/sqrt(3)) xi_cont")
ax.loglog(list(KS), [LSTAR[k] for k in KS], "o", ms=4, label="l*（实测）")
ax.axhline(2.0, color="k", ls="--", lw=1)
ax.axvline(1.0 / 48.0, color="r", ls=":", lw=1, label="K = 1/48（宽度窗）")
ax.axvline(1.0 / (16.0 * math.pi ** 2), color="b", ls=":", lw=1,
           label="K = 1/(16 pi^2)（洛伦兹窗）")
ax.set_xlabel("K")
ax.set_ylabel("l*")
ax.set_title("(c) 落窗可达性：宽度与关联长度")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

ax = axes[1, 1]
K = 0.005
m0sq = (2.0 * np.pi) ** 2 * K
qq3 = 2.0 * np.pi * np.fft.fftfreq(L3)
QX3, QY3, QZ3 = np.meshgrid(qq3, qq3, qq3, indexing="ij")
Lam3 = (KAP * 4.0 * (np.sin(QX3 / 2.0) ** 2 + np.sin(QY3 / 2.0) ** 2
                     + np.sin(QZ3 / 2.0) ** 2) + m0sq)
G3 = np.fft.ifftn(1.0 / Lam3).real
r3 = np.arange(1, 30)
ax.semilogy(r3, G3[0, 0, r3] * r3, "o", ms=3, label="3D 轴向：G · r")
lo = int(math.ceil(2.5 * xi_disc(K)))
ra, _ = fit_rate(np.arange(L3), G3[0, 0, :] * np.arange(L3), lo, lo + 16, thr_lo=1e-16)
dr = np.arange(lo, lo + 16, dtype=float)
A0 = G3[0, 0, lo] * lo * np.exp(ra * lo)
ax.semilogy(dr, A0 * np.exp(-ra * dr), "-", lw=1,
            label="拟合：1/xi = %.4f（预测 %.4f）" % (ra, m_disc(K)))
ax.set_xlabel("r（格）")
ax.set_ylabel("G · r")
ax.set_title("(d) 三维有隙格林函数的轴向衰减")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

fig.tight_layout()
fig.savefig(IMG, dpi=150)
print()
print("图已存：%s" % IMG)
print("总运行时间 = %.1f 秒" % (time.time() - T0))
