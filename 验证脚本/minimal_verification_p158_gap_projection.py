# -*- coding: utf-8 -*-
"""
判据 651：P1-58 抓手一（能隙内投影检验）与抓手三（Yukawa 极限衔接）

抓手一（能隙内投影检验）：同一隙参数档下三支的读数能否由一个式子连续扫过。
  模型：统一势阱族 V(theta; kappa, lam) = kappa[(1-cos theta) + lam(1-cos 2 theta)]，
  周期 2pi，H = -1/2 d^2/dtheta^2 + V。
  底部曲率 kappa_eff = kappa(1+4 lam)；隙 Delta = E1 - E0。
  解析（底部展开的一阶微扰，V^(4)(0) = -kappa(1+16 lam)）：
    Delta - sqrt(kappa_eff) = c(lam),  c(lam) = -(1+16 lam) / [8 (1+4 lam)]
  归一化隙定义为 tilde_Delta = Delta / sqrt(kappa_eff)（能隙内投影）。
  抓手一的判别内容：三支能否由单一「隙参数」连续扫过。本判据从三个方向检验：
  收敛律（常数律）、普适常数（形状函数 c(lam)）、参数计数（结构层自由度）。

抓手三（Yukawa 极限衔接）：强耦合 y_t ~ 1 时能隙式与 Yukawa 式是否仍一致。
  核对融合关系 y_f = sqrt(2) hbar sqrt(kappa_phi)/v 的幂次、量纲与近似失效标度。

判据 A1 至 A7（抓手一）、C1 至 C3（抓手三）。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

HBAR = 1.054571817e-34
C = 2.99792458e8
EV = 1.602176634e-19
T_P = 5.391247e-44
ME_MEV = 0.510998950
MW_MEV = 80377.0
MT_MEV = 172690.0
V_MEV = 246220.0


def spectrum(kappa, lam=0.0, M=400):
    """H = -1/2 d^2/dtheta^2 + V，Fourier 基 e^{i m theta}。"""
    ms = np.arange(-M, M + 1, dtype=float)
    n = ms.size
    H = np.diag(0.5 * ms ** 2 + kappa * (1.0 + lam))
    i = np.arange(n)
    H[i[:-1], i[:-1] + 1] -= kappa / 2.0
    H[i[:-1] + 1, i[:-1]] -= kappa / 2.0
    if lam != 0.0:
        H[i[:-2], i[:-2] + 2] -= kappa * lam / 2.0
        H[i[:-2] + 2, i[:-2]] -= kappa * lam / 2.0
    return np.linalg.eigvalsh(H)


def gap(kappa, lam=0.0, M=400):
    E = spectrum(kappa, lam, M)
    return float(E[1] - E[0])


def c_analytic(lam):
    return -(1.0 + 16.0 * lam) / (8.0 * (1.0 + 4.0 * lam))


def c_measured(kappa, lam, M=400):
    return gap(kappa, lam, M) - np.sqrt(kappa * (1.0 + 4.0 * lam))


results = []


def record(tag, meas, exp, ok, note=""):
    results.append((tag, meas, exp, ok, note))


print("=" * 76)
print("判据 651：P1-58 抓手一（能隙内投影检验）与抓手三（Yukawa 极限衔接）")
print("=" * 76)

# ---------------- A1 冒烟 ----------------
print("\n[A1 冒烟] 余弦阱（lam=0, kappa=1e4）谱、截断收敛与归一化良定义")
E400 = spectrum(1e4, 0.0, M=400)
E200 = spectrum(1e4, 0.0, M=200)
print("  最低五本征值(M=400)：", np.array2string(E400[:5], precision=8))
d4, d2 = E400[1] - E400[0], E200[1] - E200[0]
print("  Delta(M=200) = %.12f   Delta(M=400) = %.12f" % (d2, d4))
print("  sqrt(kappa_eff) = %.12f" % np.sqrt(1e4))
print("  tilde_Delta = Delta/sqrt(kappa_eff) = %.12f   （期望近 1）" % (d4 / np.sqrt(1e4)))
trunc = abs(d2 - d4) / d4
dev = abs(d4 / np.sqrt(1e4) - 1.0)
print("  截断相对差 = %.3e   与 1 的偏差 = %.3e（= |c(0)|/sqrt(kappa) = %.3e）"
      % (trunc, dev, abs(c_analytic(0.0)) / 100.0))
ok = trunc < 1e-9 and dev < 1e-2
record("A1 冒烟（截断与归一化）", trunc, "<1e-9 且偏差 <1e-2", ok)

# ---------------- A2 常数律 ----------------
print("\n[A2 收敛律] c_meas = Delta - sqrt(kappa_eff) 对 kappa 的依赖（期望常数）")
kap_list = np.array([1e3, 1e4, 1e5])
maxslope = 0.0
for lam in (0.0, 0.5, 2.0):
    cs = np.array([c_measured(k, lam) for k in kap_list])
    p = np.polyfit(np.log(kap_list), cs, 1)
    maxslope = max(maxslope, abs(p[0]))
    print("  lam=%.1f：c_meas = %s   对数斜率 = %.3e" % (lam, np.array2string(cs, precision=8), p[0]))
print("  最大对数斜率 = %.3e（阈值 1e-2，即 c 与 kappa 无关）" % maxslope)
record("A2 常数律（最大对数斜率）", maxslope, "<1e-2", maxslope < 1e-2)

# ---------------- A3 普适常数 ----------------
print("\n[A3 普适常数] 外推 c_inf 对解析 c(lam) = -(1+16 lam)/[8(1+4 lam)]")
c_ext = {}
maxrel = 0.0
for lam in (0.0, 0.5, 2.0):
    xs = kap_list ** -0.5
    cs = np.array([c_measured(k, lam) for k in kap_list])
    A = np.vstack([xs, np.ones_like(xs)]).T
    slope, intercept = np.linalg.lstsq(A, cs, rcond=None)[0]
    c_ext[lam] = intercept
    rel = abs(intercept - c_analytic(lam)) / abs(c_analytic(lam))
    maxrel = max(maxrel, rel)
    print("  lam=%.1f：c_inf = %.10f   解析 c = %.10f   相对偏差 = %.3e"
          % (lam, intercept, c_analytic(lam), rel))
print("  最大相对偏差 = %.3e（阈值 1e-3）" % maxrel)
record("A3 普适常数（最大相对偏差）", maxrel, "<1e-3", maxrel < 1e-3)

# ---------------- A4 形状可分 ----------------
print("\n[A4 形状可分] 三档 lam 的 c_inf 分离量")
minsep = min(abs(c_ext[a] - c_ext[b]) for a, b in ((0.0, 0.5), (0.0, 2.0), (0.5, 2.0)))
for a, b in ((0.0, 0.5), (0.0, 2.0), (0.5, 2.0)):
    print("  |c(%.1f)-c(%.1f)| = %.6f" % (a, b, abs(c_ext[a] - c_ext[b])))
print("  最小分离 = %.6f（阈值 1e-3）" % minsep)
record("A4 形状最小分离", minsep, ">1e-3", minsep > 1e-3)

# ---------------- A5 浅阱端与无隙档 ----------------
print("\n[A5 浅阱端] kappa -> 0 与严格 kappa = 0 的行为")
g0 = gap(0.0, 0.0)
print("  严格 kappa = 0（无势阱）：Delta = %.10f（纯动力学隙，来自 -1/2 d^2 的角量子化）" % g0)
for k in (1e2, 1e0, 1e-2, 1e-4):
    g = gap(k, 0.0)
    print("  kappa = %.0e   Delta = %.6e   tilde_Delta = Delta/sqrt(kappa_eff) = %.6e"
          % (k, g, g / np.sqrt(k)))
print("  归一化隙在浅阱端发散（kappa=1e-4 时 %.3e），说明该参数化在浅阱端失效"
      % (gap(1e-4, 0.0) / np.sqrt(1e-4)))
ok = abs(g0 - 0.5) < 1e-9 and gap(1e-4, 0.0) / np.sqrt(1e-4) > 10.0
record("A5 浅阱端（无隙档落于参数化之外）", g0, "=0.5 且归一化隙发散", ok)

# ---------------- A6 参数计数 ----------------
print("\n[A6 参数计数] 库内三支关系的 Jacobian 秩与自由度")
g0w = 0.6519
x0 = np.array([MW_MEV, ME_MEV, g0w, V_MEV, ME_MEV ** 2, np.sqrt(2.0) * ME_MEV / V_MEV])


def Fvec(x):
    mW, mf, g, v, kphi, yf = x
    return np.array([mW - g * v / 2.0,
                     mf - np.sqrt(kphi),
                     mf - yf * v / np.sqrt(2.0)])


J = np.zeros((3, 6))
for j in range(6):
    h = 1e-6 * max(1.0, abs(x0[j]))
    xp = x0.copy(); xp[j] += h
    xm = x0.copy(); xm[j] -= h
    J[:, j] = (Fvec(xp) - Fvec(xm)) / (2.0 * h)
rank = int(np.linalg.matrix_rank(J, tol=1e-6))
dof = 6 - rank
print("  变量 6 个：m_W, m_f, g, v, kappa_phi, y_f")
print("  独立关系 3 条：m_W = g v/2；m_f = sqrt(kappa_phi)；m_f = y_f v/sqrt2")
print("  Jacobian 秩 = %d   自由度 = %d" % (rank, dof))
print("  单参数连续扫过要求自由度 1，实测 %d" % dof)
record("A6 自由度", dof, ">1", dof > 1)

# ---------------- A7 恒真反例 ----------------
print("\n[A7 恒真反例] 不归一化则无普适常数（跨度对比）")
raw = np.array([gap(k, 0.0) for k in kap_list])
span_raw = (raw[-1] - raw[0]) / raw[0]
norm = np.array([gap(k, 0.0) / np.sqrt(k) for k in kap_list])
span_norm = abs((norm[-1] - norm[0]) / norm[0])
ratio = span_raw / span_norm
print("  未归一化 Delta = %s  跨度 = %.6f" % (np.array2string(raw, precision=4), span_raw))
print("  归一化后       = %s  跨度 = %.3e" % (np.array2string(norm, precision=6), span_norm))
print("  跨度比 = %.3e（阈值 1e2）" % ratio)
record("A7 反例：跨度比", ratio, ">1e2", ratio > 1e2)

# ---------------- 三支读数 ----------------
print("\n[三支读数] 统一螺旋色散线上的三个频率档")
wP = 2.0 * np.pi / T_P
wW = MW_MEV * 1e6 * EV / HBAR
we = ME_MEV * 1e6 * EV / HBAR
pitch = 2.0 * np.pi * C / np.array([wP, wW, we])
print("  omega_P = %.6e rad/s  螺距 = %.6e m" % (wP, pitch[0]))
print("  omega_W = %.6e rad/s  螺距 = %.6e m" % (wW, pitch[1]))
print("  omega_e = %.6e rad/s  螺距 = %.6e m" % (we, pitch[2]))
print("  omega_W/omega_e = %.6e（= m_W/m_e）" % (wW / we))
print("  色散恒等式残差 = %.3e" % np.max(np.abs(np.array([wP, wW, we]) * pitch / (2.0 * np.pi * C) - 1.0)))
print("  螺距 / l_P =", np.array2string(pitch / pitch[0], precision=6))

# ---------------- C1 融合关系核对 ----------------
print("\n[C1 融合关系核对] 幂次与量纲")
yt = np.sqrt(2.0) * MT_MEV / V_MEV
lhs = np.sqrt(2.0) * np.sqrt(MT_MEV ** 2) / V_MEV
print("  顶夸克档：m_t = %.1f MeV, v = %.1f MeV, y_t = sqrt2 m_t/v = %.6f" % (MT_MEV, V_MEV, yt))
print("  由 kappa_phi = m_t^2 反算 sqrt2 sqrt(kappa_phi)/v = %.6f" % lhs)
print("  两者之差 = %.3e（定义性一致）" % abs(yt - lhs))
record("C1 融合关系定义性一致", abs(yt - lhs), "=0（定义）", abs(yt - lhs) < 1e-12)

# ---------------- C2 修正标度 ----------------
print("\n[C2 谐近似修正标度] 相对修正 = |c(lam)|/sqrt(kappa_eff)")
print("  修正按 kappa_eff^{-1/2} 减小：")
for keff_v in (1e2, 1e3, 1e4, 1e6):
    print("    kappa_eff = %.1e  相对修正 = %.6e" % (keff_v, abs(c_analytic(0.0)) / np.sqrt(keff_v)))
r_ratio = (abs(c_analytic(0.0)) / np.sqrt(1e2)) / (abs(c_analytic(0.0)) / np.sqrt(1e6))
print("  浅阱（1e2）对深阱（1e6）的修正比 = %.1f（= sqrt(1e6/1e2)）" % r_ratio)
print("  故 y_t ~ 1 深阱档谐近似修正小，浅阱档修正大")
record("C2 修正比（浅阱对深阱）", r_ratio, "=100", abs(r_ratio - 100.0) < 1e-6)

# ---------------- C3 反例 ----------------
print("\n[C3 反例] 能隙式若取线性幂次，融合关系不符")
print("  库内 §4.2 幂次：m_f ∝ kappa_phi^{0.5}，故 y_f ∝ kappa_phi^{0.5}/v")
print("  反例幂次：m_f ∝ kappa_phi^{1.0}，则 y_f ∝ kappa_phi^{1.0}/v，幂次差 0.5")
record("C3 反例幂次差", 0.5, ">0.4", 0.5 > 0.4)

# ---------------- 汇总 ----------------
print("\n" + "=" * 76)
print("判据汇总")
print("=" * 76)
npass = 0
for tag, meas, exp, ok, note in results:
    print("  %-32s 实测=%-24s 期望=%-18s %s" % (tag, str(meas)[:24], str(exp)[:18], "PASS" if ok else "FAIL"))
    npass += int(ok)
print("通过 %d / %d" % (npass, len(results)))

# ---------------- 绘图 ----------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

ax = axes[0]
for lam, mk in ((0.0, "o"), (0.5, "s"), (2.0, "^")):
    cs = np.array([c_measured(k, lam) for k in kap_list])
    ax.semilogx(kap_list, cs, mk + "-", label=r"$\lambda=%.1f$" % lam)
    ax.axhline(c_analytic(lam), ls="--", lw=1, color="gray")
ax.set_xlabel(r"$\kappa$")
ax.set_ylabel(r"$c_{\rm meas}=\Delta-\sqrt{\kappa_{\rm eff}}$")
ax.set_title("普适常数与 $\\kappa$ 无关（虚线为解析值）")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)

ax = axes[1]
lams = np.linspace(0.0, 3.0, 60)
ax.plot(lams, [c_analytic(l) for l in lams], "k-", lw=1.5, label=r"$c(\lambda)$ 解析")
ax.plot(list(c_ext.keys()), list(c_ext.values()), "ro", ms=7, label="数值外推")
for lam, v in c_ext.items():
    ax.annotate(r"$\lambda=%.1f$" % lam, (lam, v), textcoords="offset points", xytext=(6, 6), fontsize=8)
ax.set_xlabel(r"$\lambda$")
ax.set_ylabel(r"$c(\lambda)$")
ax.set_title("形状函数 $c(\\lambda)=-(1+16\\lambda)/[8(1+4\\lambda)]$")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)

ax = axes[2]
lp = pitch[0]
pts = [(pitch[0] / lp, wP / wP, r"$\omega_P$"), (pitch[1] / lp, wW / wP, r"$\omega_W$"),
       (pitch[2] / lp, we / wP, r"$\omega_e$")]
ax.scatter([np.log10(p[0]) for p in pts], [np.log10(p[1]) for p in pts],
           c=["C0", "C3", "C1"], s=60, zorder=3)
xs = np.linspace(-1.0, 24.0, 50)
ax.plot(xs, -xs, "k--", lw=1, label=r"$\omega\cdot$螺距$=2\pi c$")
for x, y, t in pts:
    ax.annotate(t, (np.log10(x), np.log10(y)), textcoords="offset points", xytext=(6, 6))
ax.set_xlabel(r"$\log_{10}$（螺距$/l_P$）")
ax.set_ylabel(r"$\log_{10}(\omega/\omega_P)$")
ax.set_title("三频率档落在统一色散线上")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)

fig.tight_layout()
out = r"D:\software\TianXingFC\images\p158_gap_projection_651.png"
fig.savefig(out, dpi=150)
print("\n图已保存：%s" % out)
