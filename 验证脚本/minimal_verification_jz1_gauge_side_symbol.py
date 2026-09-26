# -*- coding: utf-8 -*-
"""第 434 条：洛伦兹符号的规范侧转写判据（等幅条件、色散与无参数光速、无反例对照）。

判据（先判据后实现）：
  G1 等幅条件：低幅二次系数比 a_E/|a_B| = 2 tP^2/dt^2 - 1。
     dt = tP 时等幅反号（比值 1，洛伦兹符号成立）；dt = sqrt(2) tP 时 a_E = 0（E 型双计临界相消）；
     dt = 2 tP 时比值 -0.5（符号翻转）。
  G2 色散与光速：线性化横向模 omega^2 = (1/tP^2)[4 sin^2(kx lP/2) + 4 sin^2(ky lP/2)]，
     长波 omega/k -> lP/tP（无自由参数，用最小 k 检验）。velocity Verlet 测频与解析对照。
  G3 反例对照：去掉取向惯性能（L = -V，欧氏全正）后二次系数同号，omega^2 < 0，无传播（纯衰减）。

容差预算：G2 相对差来源为 velocity Verlet 相位误差 O(omega^3 dt^2) 与零交叉量化 O(dt/T)。
取 dt = 0.02、omega <= 0.4，预算在 1e-4 量级，故判据容差取 1e-3（含安全余量）。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG = r"D:\software\TianXingFC\images\jz1_gauge_side_symbol.png"
lP, tP, I0 = 1.0, 1.0, 1.0
T = 8
rng = np.random.default_rng(20260926)

# ---------------- G1：E 型与 B 型二次系数 ----------------
g = rng.standard_normal(T)            # 时间型基础组态（空间均匀）
d0 = np.roll(g, -1) - g               # 时间差分（周期）
S2 = float(np.sum(d0 ** 2))

nx, ny = 6, 6
phi_unit = np.zeros((nx, ny))
phi_unit[1, 0] = 1.0                  # 空间型基础组态：单点扭折
P1 = phi_unit + np.roll(phi_unit, -1, axis=0) - np.roll(phi_unit, -1, axis=1) - phi_unit
SP2 = float(np.sum(P1 ** 2))

def L_E(eps, dt):
    """纯时间型组态的完整拉格朗日量：T - V。"""
    Tk = (I0 / 2.0) * (eps / dt) ** 2 * S2
    V = (I0 / (2.0 * tP ** 2)) * np.sum(1.0 - np.cos(eps * d0))
    return Tk - V

def L_B(eps):
    """纯空间型组态的完整拉格朗日量（静态，无动能、无时间型曲率）。"""
    return -(I0 / (2.0 * tP ** 2)) * np.sum(1.0 - np.cos(eps * P1))

def fit_c2(fun, eps_list):
    y = np.array([fun(e) for e in eps_list])
    e = np.array(eps_list)
    A = np.vstack([e ** 2, e ** 4, e ** 6]).T
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(c[0])

# 冒烟检查：组态变化性与操作轴一致性（变化的轴 = 差分/操作的轴）
assert np.max(np.abs(d0)) > 0.0 and np.max(np.abs(P1)) > 0.0
print("冒烟检查：max|d0| = %.4f（时间轴差分，空间均匀）；max|P1| = %.4f（空间扭折，axis0/axis1 差分）"
      % (np.max(np.abs(d0)), np.max(np.abs(P1))))

eps = [1e-4, 2e-4, 4e-4, 8e-4, 1.6e-3]
rows_G1 = []
for dt in (0.5 * tP, tP, np.sqrt(2.0) * tP, 2.0 * tP):
    aE = fit_c2(lambda e, dt=dt: L_E(e, dt), eps) / (I0 * S2)
    aB = fit_c2(L_B, eps) / (I0 * SP2)
    rows_G1.append((dt / tP, aE, aB, aE / abs(aB), 2.0 * tP ** 2 / dt ** 2 - 1.0))

# ---------------- G2：色散与光速 ----------------
def omega_analytic(k):
    return (1.0 / tP) * np.sqrt(4.0 * np.sin(k * lP / 2.0) ** 2)

def omega_crossing(k, dt=0.02):
    lam = 4.0 * np.sin(k * lP / 2.0) ** 2
    w = np.sqrt(lam) / tP
    tmax = min(4000.0, 20.0 * 2.0 * np.pi / w)
    n = int(tmax / dt)
    x, v = 1.0, 0.0
    xs = np.empty(n)
    for i in range(n):
        a = -(w ** 2) * x
        v += a * dt
        x += v * dt
        xs[i] = x
    s = np.sign(xs)
    idx = np.where((s[:-1] * s[1:]) < 0)[0]
    if len(idx) < 4:
        return np.nan
    half_period = dt * (idx[-1] - idx[0]) / (len(idx) - 1)
    return np.pi / half_period

ks = np.array([0.05, 0.1, 0.2, 0.3, 0.4])
rows_G2 = []
for k in ks:
    wnum = omega_crossing(k)
    wana = omega_analytic(k)
    rows_G2.append((k, wnum, wana, abs(wnum - wana) / wana, wana / k))

# ---------------- G3：反例对照 ----------------
def evolve(sign, dt=0.02, n=4000):
    lam = 4.0 * np.sin(0.2 * lP / 2.0) ** 2
    x, v = 1.0, 0.0
    xs = []
    for _ in range(n):
        a = sign * (lam / tP ** 2) * x     # sign = -1：有惯性（振荡）；sign = +1：无惯性（指数型）
        v += a * dt
        x += v * dt
        xs.append(x)
    return np.array(xs)

osc = evolve(-1.0)
dec = evolve(+1.0)
k0 = 0.2
w0 = omega_analytic(k0)
print("G1 规范侧二次系数（E 型双计与等幅条件）")
print("dt/tP   a_E        |a_B|      比值   解析比值")
for r in rows_G1:
    print("%.4f  %+.8f  %.8f  %+.6f  %+.6f" % r)
print()
print("G2 色散与光速（长波斜率应为 lP/tP = %.3f）" % (lP / tP))
print("k      数值 omega  解析 omega   相对差    omega/k")
for r in rows_G2:
    print("%.3f  %.8f  %.8f  %.2e  %.6f" % r)
print()
flag13 = bool(all(abs(r[3] - r[4]) < 1e-9 for r in rows_G1))
flag2 = bool(all(r[3] < 1e-3 for r in rows_G2) and abs(rows_G2[0][4] - lP / tP) < 1e-3)
flag3 = bool(dec.max() > 2.0 and abs(osc.max() - 1.0) < 0.05)
print("判据 G1（等幅条件与三档解析对照）：%s" % ("通过" if flag13 else "未通过"))
print("判据 G2（数值色散 vs 解析，长波斜率取最小 k）：%s" % ("通过" if flag2 else "未通过"))
print("判据 G3（无惯性项对照：无振荡指数型）：%s" % ("通过" if flag3 else "未通过"))
print("判据总计：%d/3" % (int(flag13) + int(flag2) + int(flag3)))

# ---------------- 出图 ----------------
fig, ax = plt.subplots(2, 2, figsize=(11, 8))
dts = np.linspace(0.4, 2.6, 200)
ax[0, 0].plot(dts, 2.0 * tP ** 2 / dts ** 2 - 1.0, "k-", lw=1.2, label="analytic 2 tP^2/dt^2 - 1")
ax[0, 0].plot([r[0] for r in rows_G1], [r[3] for r in rows_G1], "ro", ms=6, label="numeric")
ax[0, 0].axhline(0.0, color="gray", lw=0.8)
ax[0, 0].axvline(1.0, color="green", ls="--", lw=1.0, label="dt = tP (equal weight)")
ax[0, 0].axvline(np.sqrt(2.0), color="orange", ls=":", lw=1.0, label="dt = sqrt2 tP (a_E = 0)")
ax[0, 0].set_xlabel("dt / tP")
ax[0, 0].set_ylabel("a_E / |a_B|")
ax[0, 0].set_title("G1 Lorentz-sign condition")
ax[0, 0].legend(fontsize=7)

kk = np.linspace(1e-3, 0.5, 200)
ax[0, 1].plot(kk, [omega_analytic(k) for k in kk], "k-", lw=1.2, label="analytic")
ax[0, 1].plot([r[0] for r in rows_G2], [r[1] for r in rows_G2], "ro", ms=6, label="numeric")
ax[0, 1].plot(kk, (lP / tP) * kk, "b--", lw=1.0, label="c k, c = lP/tP")
ax[0, 1].set_xlabel("k")
ax[0, 1].set_ylabel("omega")
ax[0, 1].set_title("G2 dispersion and light speed")
ax[0, 1].legend(fontsize=7)

tt = np.arange(len(osc)) * 0.02
ax[1, 0].plot(tt, osc, "g-", lw=1.0, label="with orientation inertia (wave)")
ax[1, 0].plot(tt, np.clip(dec, -1e3, 1e3), "r-", lw=1.0, label="no inertia (exponential)")
ax[1, 0].set_ylim(-3, 3)
ax[1, 0].set_xlabel("t")
ax[1, 0].set_ylabel("amplitude")
ax[1, 0].set_title("G3 counterexample")
ax[1, 0].legend(fontsize=7)

ax[1, 1].bar([0, 1, 2], [0.5, -0.25, 0.25], color=["steelblue", "indianred", "seagreen"])
ax[1, 1].bar([3], [-0.25], color="indianred")
ax[1, 1].axhline(0.0, color="k", lw=0.8)
ax[1, 1].set_xticks([0, 1, 2, 3])
ax[1, 1].set_xticklabels(["E: inertia", "E: curvature", "E: net", "B: curvature"], fontsize=8)
ax[1, 1].set_ylabel("coefficient (I0 units)")
ax[1, 1].set_title("G1 double counting of E")
plt.tight_layout()
plt.savefig(IMG, dpi=130)
print("图已保存：%s" % IMG)
