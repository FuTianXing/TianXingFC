# -*- coding: utf-8 -*-
"""
mass_energy_relation.py
551 号：质能关系的晶格推导配套验证（四环合并的数值层面）

自然单位 c = m = hbar = 1，omega_0 = 1（由 m = hbar*omega_0/c^2）。

判据（先立理论期望再实现，打印"实测 vs 期望"对照）：
  J1 主线恒等与区分力（EP5）：长波相对论色散 omega^2 = omega_0^2 + c^2 k^2
     经结算（E = hbar*omega, p = hbar*k）后 E^2 - p^2 c^2 - m^2 c^4 残差为
     机器零；牛顿色散 omega_N = omega_0 + hbar k^2/(2m)（同 omega_0 同结算）
     残差随 p 增大（p = mc 档解析值 0.25）。改掉色散形状判据即 FAIL，
     判据不含物理信息的恒真风险由该反例排除。
  J2 完整晶格色散的低能收敛：一维链 omega^2 = omega_0^2 + (2c/l_P)^2
     sin^2(k l_P/2) 的同口径残差按 -1/12 (k l_P)^4 标度（首阶洛伦兹破坏
     家族），三档拟合幂实测 4 +/- 0.1。
  J3 低速展开 p^4 系数：E(p) = sqrt(m^2 c^4 + p^2 c^2) 的数值四阶系数
     对 -1/(8 m^3 c^2)，窗口 p <= 0.3 mc，p^6 泄漏预算内偏差 < 5%。
  J4 能量-速度关系：数值差分群速度 v_g = domega/dk 与 c^2 p/E 相等
     （相对论色散下恒等式级）；牛顿色散下不等（v_g = hbar k/m，
     反例）。
  J5 速度上限：完整一维含能隙色散的群速全域 max < c（严格不等式）。

容差预算：
  J1 解析恒等浮点舍入 1e-16 量级，阈值 1e-12（富余四个量级）；
  J2 三档 k l_P = 0.05/0.1/0.2，残差约 (kl_P)^4/12（1e-6 至 1e-4），
     双精度充分，幂拟合容差 ±0.1；
  J3 窗口 p_max = 0.3 mc，p^6 泄漏相对量 ~ p_max^2 < 0.09，系数偏差
     阈值 5%；
  J4 中心差分 O(h^2)，h = 1e-3，误差 ~1e-6，阈值 1e-4；
  J5 严格不等式无容差（网格最大 < c*(1-1e-12)）。

冒烟检查：先打印各曲线残差的量级与非零性，核对"变化的轴 = 色散形状
的轴"，再进判据。
图：images/mass_energy_relation_551.png（三面板，见注记 §四 图 1）。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG = r"D:\software\TianXingFC\images\mass_energy_relation_551.png"

c = m = hbar = 1.0
w0 = 1.0
lP = 1.0
h_diff = 1e-3

# ── 色散形状定义 ─────────────────────────────────────────────
def omega_rel(k):
    """长波相对论色散 omega^2 = omega_0^2 + c^2 k^2"""
    return np.sqrt(w0**2 + c**2 * k**2)

def omega_newton(k):
    """牛顿色散（EP5 反例）：同 omega_0，动能按 hbar^2 k^2/(2m) 叠加"""
    return w0 + hbar * k**2 / (2.0 * m)

def omega_lattice(k):
    """完整一维含能隙晶格色散 omega^2 = omega_0^2 + (2c/lP)^2 sin^2(k lP/2)"""
    return np.sqrt(w0**2 + (2.0 * c / lP) ** 2 * np.sin(k * lP / 2.0) ** 2)

def residual(omega_of_k, k):
    """结算（E=hbar*omega, p=hbar*k）后的主线残差 E^2 - p^2c^2 - m^2c^4"""
    E = hbar * omega_of_k(k)
    p = hbar * k
    return E**2 - p**2 * c**2 - m**2 * c**4

# ── 冒烟检查 ─────────────────────────────────────────────────
k_smoke = np.array([0.1, 0.3, 0.5, 1.0, 2.0])
r_rel = residual(omega_rel, k_smoke)
r_newton = residual(omega_newton, k_smoke)
r_latt = residual(omega_lattice, k_smoke)
print("== 冒烟检查 ==")
print("k 档               :", k_smoke)
print("rel 残差（应机器零）:", r_rel)
print("牛顿残差（应随k增） :", r_newton)
print("晶格残差（应-1/12 k^4 量级）:", r_latt)
assert np.max(np.abs(r_rel)) < 1e-12, "冒烟失败：长波相对论色散残差非机器零"
assert np.max(np.abs(r_newton)) > 0.1 and np.all(np.diff(r_newton) > 0), "冒烟失败：牛顿反例残差未随 k 单调展开"
assert np.all(r_latt < 0), "冒烟失败：晶格残差应为负（-k^4/12 族）"
print("冒烟通过：变化的轴 = 色散形状的轴，量级合理\n")

# ── J1 主线恒等与区分力 ──────────────────────────────────────
k1 = np.linspace(0.03, 3.0, 400)
r1_rel = np.abs(residual(omega_rel, k1))
r1_newton = np.abs(residual(omega_newton, k1))
j1a = np.max(r1_rel) < 1e-12
newton_at_1 = np.abs(residual(omega_newton, np.array([1.0])))[0]
j1b = newton_at_1 > 0.1
print("== J1 主线恒等与区分力 ==")
print("实测 vs 期望：rel 残差 max = %.3e（期望 < 1e-12）" % np.max(r1_rel))
print("实测 vs 期望：牛顿残差 @p=mc = %.4f（期望 > 0.1，解析 0.25）" % newton_at_1)
print("J1:", "PASS" if (j1a and j1b) else "FAIL")

# ── J2 完整晶格低能收敛标度 ──────────────────────────────────
k2 = np.array([0.05, 0.1, 0.2])
r2 = np.abs(residual(omega_lattice, k2))
theo2 = k2**4 / 12.0
pow_fit = np.polyfit(np.log(k2), np.log(r2), 1)[0]
j2 = abs(pow_fit - 4.0) < 0.1
print("== J2 完整晶格低能收敛 ==")
for kk, rr, tt in zip(k2, r2, theo2):
    print("  k=%.2f: 残差实测 %.4e vs 期望 k^4/12=%.4e（比值 %.4f，期望 1）" % (kk, rr, tt, rr / tt))
print("实测 vs 期望：拟合幂 = %.3f（期望 4±0.1）" % pow_fit)
print("J2:", "PASS" if j2 else "FAIL")

# ── J3 低速展开 p^4 系数 ─────────────────────────────────────
# 提取方法：含 p^6 项的四参数拟合（纯 p^4 基的 p^6 泄漏经设计矩阵
# 条件数放大，实测偏差 ~6.5% 超预算；p^6 入基后残余为 p^8 泄漏，
# 预算 <1%，阈值 5% 富余）。
p3 = np.linspace(0.05, 0.3, 6)
E3 = omega_rel(p3)  # E = hbar*omega，自然单位同值
A = np.vstack([np.ones_like(p3), p3**2, p3**4, p3**6]).T
coef, *_ = np.linalg.lstsq(A, E3, rcond=None)
a4_num = coef[2]
a4_theo = -1.0 / (8.0 * m**3 * c**2)
dev = abs(a4_num - a4_theo) / abs(a4_theo)
j3 = dev < 0.05
print("== J3 低速展开 p^4 系数 ==")
print("实测 vs 期望：a4 = %.5f（期望 %.5f，偏差 %.2f%%，阈值 5%%）" % (a4_num, a4_theo, dev * 100))
print("J3:", "PASS" if j3 else "FAIL")

# ── J4 能量-速度关系（群速度 = c^2 p / E）────────────────────
k4 = np.linspace(0.05, 2.0, 50)
def vg_num(omega_of_k, k):
    return (omega_of_k(k + h_diff) - omega_of_k(k - h_diff)) / (2.0 * h_diff)
vg_rel = vg_num(omega_rel, k4)
vg_theo_rel = c**2 * (hbar * k4) / (hbar * omega_rel(k4))
dev4 = np.max(np.abs(vg_rel - vg_theo_rel))
j4a = dev4 < 1e-4
vg_newton = vg_num(omega_newton, np.array([1.0]))[0]
vg_newton_theo = c**2 * 1.0 / omega_newton(1.0)
j4b = abs(vg_newton - vg_newton_theo) > 0.1
print("== J4 能量-速度关系 ==")
print("实测 vs 期望：rel 最大偏差 = %.3e（期望 < 1e-4）" % dev4)
print("实测 vs 期望：牛顿反例 @p=mc：v_g=%.4f vs c^2p/E=%.4f（差 %.4f，期望 > 0.1）"
      % (vg_newton, vg_newton_theo, abs(vg_newton - vg_newton_theo)))
print("J4:", "PASS" if (j4a and j4b) else "FAIL")

# ── J5 速度上限 ──────────────────────────────────────────────
k5 = np.linspace(0.02, np.pi, 4000)
v5 = np.abs(vg_num(omega_lattice, k5))
max5 = np.max(v5)
j5 = max5 < c * (1 - 1e-12)
print("== J5 速度上限 ==")
print("实测 vs 期望：晶格群速全域 max = %.6f（期望 < c=1，严格）" % max5)
print("J5:", "PASS" if j5 else "FAIL")

# ── 汇总 ─────────────────────────────────────────────────────
verdicts = [j1a and j1b, j2, j3, j4a and j4b, j5]
names = ["J1 主线恒等与区分力", "J2 晶格低能收敛标度", "J3 低速展开系数", "J4 能量-速度关系", "J5 速度上限"]
print("\n== 判据汇总 ==")
for nm, v in zip(names, verdicts):
    print("%s: %s" % (nm, "PASS" if v else "FAIL"))
print("总判据: %d/5" % sum(verdicts))

# ── 三面板图 ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
ax = axes[0]
ax.loglog(k1, np.maximum(r1_rel, 1e-18), "b-", lw=2, label="relativistic dispersion (machine zero)")
ax.loglog(k1, r1_newton, "r--", lw=2, label="Newtonian dispersion (counterexample)")
ax.axhline(1e-12, color="gray", ls=":", lw=1)
ax.axvline(1.0, color="k", lw=0.8, ls=":")
ax.scatter([1.0], [newton_at_1], color="r", zorder=5, s=28)
ax.annotate("0.25 at p=mc", (1.0, newton_at_1), textcoords="offset points", xytext=(8, 8), fontsize=9)
ax.set_xlabel(r"$p/(mc)$")
ax.set_ylabel(r"$|E^2-p^2c^2-m^2c^4|/(m^2c^4)$ scale")
ax.set_title("(1) identity vs counterexample dispersion")
ax.legend(fontsize=8)

ax = axes[1]
ax.loglog(k2, r2, "o-", lw=2, label="lattice residual")
ref = r2[0] * (k2 / k2[0]) ** 4
ax.loglog(k2, ref, "k:", lw=1.5, label=r"$\propto(k l_P)^4$")
ax.set_xlabel(r"$k\, l_P$")
ax.set_ylabel(r"$|E^2-p^2c^2-m^2c^4|$")
ax.set_title("(2) low-energy convergence, fitted power = %.2f" % pow_fit)
ax.legend(fontsize=8)

ax = axes[2]
vv = np.linspace(0.0, 0.95, 200)
gg = 1.0 / np.sqrt(1 - vv**2)
ax.plot(vv, gg, "k-", lw=2, label=r"$\gamma$")
ax.plot(vv, 1.0 + vv**2 / 2.0, "g--", lw=2, label=r"Newtonian $1+p^2/2$")
kk = np.linspace(0.05, 2.0, 60)
vv_pts = c**2 * kk / omega_rel(kk)  # 能量-速度关系的自洽读数 v = c^2 p/E
ax.plot(vv_pts, omega_rel(kk), "bo", ms=4, label="lattice-derived $E(v)$")
ax.set_xlabel(r"$v/c$")
ax.set_ylabel(r"$E/(mc^2)$")
ax.set_title("(3) energy-velocity: Lorentz factor emerges")
ax.legend(fontsize=8, loc="upper left")

fig.tight_layout()
fig.savefig(IMG, dpi=150)
print("图已存: %s" % IMG)
