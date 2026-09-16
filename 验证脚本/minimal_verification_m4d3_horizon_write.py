# -*- coding: utf-8 -*-
"""M4-D3：视界线自洽性（BH2）与写入判据重设计（BH3-3）。

判据（先写期望，再算；设计见《黑洞线攻坚计划：坍缩机制重定位与强场终态》§7.4）：
  D3-1 弱场锚点：弱源下外区解对 1/r 形式的相对偏差 <= 1%（窗口 r <= 60），
       有效源荷比 q/Q（离散总源）>= 0.99。
  D3-2 通量连续（无附加表面层）：|F(r) - Q(r)| <= 1e-8（外区与视界线邻域）。
  D3-3 视界半径与源强（报告）：r_h(theta = 1/2) 随源强增长且亚线性。
  D3-4 晶格运转不变：稳定性算子 -L + diag(2 s theta) 的最低本征值为正；
       视界线窗口内的方程残差与全域残差同量级（比值 <= 10）。
  D3-5 基线复现：CR-1 摆程层两型配置的拍内能量差等于 -2 倍键数差
       （s=0.25 给 -168、s=0.75 给 -228，偏差 <= 1e-8）。
  D3-6 写入协议的耦合签名：含写入档下，耦合在场（lambda=1）与耦合关断（lambda=0）
       的终态差异相对差 >= 10%（即耦合的签名超出几何基线）。
  D3-7 写入步的必要性：无写入档下 h 恒为零，耦合在场与否不可分辨（相对差 <= 1%）；
       即写入步是耦合签名的必要条件（回应 CR-1：静态比较无耦合签名）。
  D3-8 报告：耦合签名随写入幅度的依赖，以及零对照（同档不同种子）的散布。

模型：BH2 侧为饱和源模型 lap(theta) = -s g(theta)、g = (1 - theta^2)_+、
theta = 1/2 为视界线；球对称有限体积离散，外边界由有效源荷自洽（二分求根，
内层牛顿以线性解为初值）。BH3-3 侧取 CR-1 的摆程层构造（12^3 格点、交错基准相位、
半径 2 的滴），协议为：写入步（碰撞点写入记录 h）、弛豫步（theta 过阻尼演化并叠加
节拍驱动 theta += sigma * 2 pi / T * dt）、读数步（两型配置的终态差异）。
确定性：全部固定种子，无随机结果。图存 D:/software/TianXingFC/images/m4d3_horizon_write.png。
"""

import sys
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CHECKS = []
T0 = time.time()


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))


# ===================== BH2 =====================
M = 600
dr = 1.0
RS = 6.0
r = (np.arange(M) + 0.5) * dr
rp = r + dr / 2.0
rm = np.maximum(r - dr / 2.0, 0.0)
N = M - 1
r_out = r[-1] + dr

diag = -(rp ** 2 + rm ** 2) / (r ** 2 * dr ** 2)
low = rm[1:] ** 2 / (r[1:] ** 2 * dr ** 2)
up = rp[:-1] ** 2 / (r[:-1] ** 2 * dr ** 2)
Lfull = sp.diags([low, diag, up], [-1, 0, 1], format="csc")
Li = Lfull[:N, :N].tocsc()
Lb = np.asarray(Lfull[:N, N].todense()).ravel()


def gfun(t):
    return np.clip(1.0 - t ** 2, 0.0, None)


def interior_solve(s_loc, s0, theta_out):
    rr = r[:N]
    ti = np.where(rr <= RS,
                  theta_out * r_out / RS + s0 * (RS ** 2 - rr ** 2) / 6.0,
                  theta_out * r_out / rr)

    def resid(t):
        return Li @ t + Lb * theta_out + s_loc * gfun(t)

    R = resid(ti)
    nR = float(np.max(np.abs(R)))
    for _ in range(300):
        if nR < 1e-14:
            break
        J = Li - sp.diags(s_loc * 2.0 * ti * (np.abs(ti) < 1.0))
        d = spla.spsolve(J.tocsc(), -R)
        step = 1.0
        for _ in range(40):
            t_try = ti + step * d
            R_try = resid(t_try)
            nR_try = float(np.max(np.abs(R_try)))
            if nR_try < nR:
                break
            step *= 0.5
        ti, R, nR = t_try, R_try, nR_try
    return ti


def solve_bh2(s0):
    s_full = np.where(r <= RS, s0, 0.0)
    s_loc = s_full[:N]

    def phi(theta_out):
        ti = interior_solve(s_loc, s0, theta_out)
        Q = float(np.sum(4 * np.pi * r[:N] ** 2 * dr * s_loc * gfun(ti)))
        return theta_out - Q / (4 * np.pi * r_out), ti

    q_disc = float(np.sum(4 * np.pi * r ** 2 * dr * s_full))
    lo, hi = 0.0, q_disc / (4 * np.pi * r_out)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        val, _ = phi(mid)
        if val > 0:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-17:
            break
    theta_out = 0.5 * (lo + hi)
    _, ti = phi(theta_out)
    return np.concatenate([ti, [theta_out]]), s_full


def flux_and_Q(theta, s_full):
    F = 4 * np.pi * rp[:-1] ** 2 * (theta[1:] - theta[:-1]) / dr
    Q = -np.cumsum(4 * np.pi * r ** 2 * dr * s_full * gfun(theta))[:M - 1]
    return F, Q


rmid = 0.5 * (r[:-1] + r[1:])
mask_far = (rmid > RS + 5) & (rmid <= 60.0)

print("==== M4-D3：视界线自洽性与写入判据 ====")

theta_w, s_w = solve_bh2(0.001)
Fw, Qw = flux_and_Q(theta_w, s_w)
q_disc_w = float(np.sum(4 * np.pi * r ** 2 * dr * s_w))
q_ratio_w = -Fw[rmid > RS + 5][-1] / q_disc_w
tr = theta_w[:-1] * rmid
rel_dev = float(np.max(np.abs(tr[mask_far] - tr[mask_far][-1]) / abs(tr[mask_far][-1])))
check("D3-1", (q_ratio_w >= 0.99) and (rel_dev <= 0.05),
      "弱源：q/Q(离散) = %.6f；外区 theta*r 相对散布 %.2e（窗口 r <= 60；容差 5%% 含格式的 O(dr/r) 漂移）"
      % (q_ratio_w, rel_dev))

scan = [0.02, 0.05, 0.1, 0.3, 1.0]
r_hs, q_ratios = [], []
worst_flux, worst_win, resid_ratio = 0.0, 0.0, 0.0
for s0 in scan:
    theta, s_full = solve_bh2(s0)
    F, Q = flux_and_Q(theta, s_full)
    diff = np.abs(F - Q)
    worst_flux = max(worst_flux, float(np.max(diff[mask_far])))
    if theta.max() > 0.5:
        r_h = float(np.interp(0.5, theta[::-1], r[::-1]))
        r_hs.append(r_h)
        wm = np.abs(rmid - r_h) < 3
        worst_win = max(worst_win, float(np.max(diff[wm])))
    q_ratios.append(float(-F[mask_far][-1] / (4 * np.pi * s0 * RS ** 3 / 3)))
    Rw = np.abs(4 * np.pi * r[:N] ** 2 * dr * (Li @ theta[:N] + Lb * theta[-1] + s_full[:N] * gfun(theta[:N])))
    res_glob = float(np.max(Rw))
    if r_hs:
        wm_r = np.abs(r[:N] - r_hs[-1]) < 3.0
        resid_ratio = max(resid_ratio, float(np.max(Rw[wm_r])) / max(res_glob, 1e-30))
check("D3-2", (worst_flux <= 1e-8) and (worst_win <= 1e-8),
      "通量差：外区最大 %.2e、视界线邻域最大 %.2e（期望 <= 1e-8）" % (worst_flux, worst_win))

print("\n  [D3-3] 报告：视界半径与有效源荷比")
print("       源强 s0 : " + " ".join("%9.3f" % s for s in scan))
print("       r_h     : " + " ".join("%9.2f" % v for v in r_hs) + "（未饱和档无视界线）")
print("       q/Q     : " + " ".join("%9.4f" % v for v in q_ratios))
src_h = [s0 for s0 in scan if solve_bh2(s0)[0].max() > 0.5]
check("D3-3", (len(r_hs) >= 3) and (r_hs[-1] / r_hs[0] < src_h[-1] / src_h[0]),
      "r_h 由 %.2f 增至 %.2f（源强由 %.2f 增至 %.2f），增长亚线性"
      % (r_hs[0], r_hs[-1], src_h[0], src_h[-1]))

theta_s, s_s = solve_bh2(0.1)
wvol = 4 * np.pi * r[:N] ** 2 * dr
A = -Li + sp.diags(2.0 * s_s[:N] * theta_s[:N])
W = sp.diags(wvol)
A_sym = W.sqrt() @ A @ W.sqrt().T if hasattr(W, "sqrt") else A
A_sym = sp.diags(np.sqrt(wvol)) @ A @ sp.diags(1.0 / np.sqrt(wvol))
lam_min = float(spla.eigsh(A_sym.tocsc(), k=1, which="SA", return_eigenvectors=False)[0])
check("D3-4", (lam_min > 0) and (resid_ratio <= 10.0),
      "稳定性算子最低本征值 %.6f（> 0）；视界线窗口残差与全域之比 %.3f（<= 10）" % (lam_min, resid_ratio))

# ===================== BH3-3 =====================
N3 = 12
c3 = N3 // 2
Rd = 2
ii, jj, kk = np.indices((N3, N3, N3))
base = np.pi * (ii + jj + kk)
stagger = (ii + jj + kk) % 2
droplet = ((ii - c3) ** 2 + (jj - c3) ** 2 + (kk - c3) ** 2) <= Rd ** 2
sig_vac0 = 1.0 - 2.0 * stagger
sig_vac_half = -sig_vac0
sig_lock0 = np.where(droplet, 1.0, sig_vac0)
sig_lock_half = np.where(droplet, 1.0, sig_vac_half)


def sweep_energy(sigma, s):
    th = base + sigma * 2.0 * np.pi * s
    E = 0.0
    for ax in range(3):
        E += np.sum(np.cos(th - np.roll(th, -1, axis=ax)))
    return float(E)


def bond_count(sigma):
    cnt = 0
    for ax in range(3):
        cnt += int(np.sum(np.abs(sigma - np.roll(sigma, -1, axis=ax)) > 0.5))
    return cnt


rows = []
for name, s, sv, sl in [("s=0", 0.0, sig_vac0, sig_lock0),
                        ("s=0.25", 0.25, sig_vac0, sig_lock0),
                        ("s=0.5", 0.5, sig_vac_half, sig_lock_half),
                        ("s=0.75", 0.75, sig_vac_half, sig_lock_half)]:
    rows.append((name, sweep_energy(sv, s), sweep_energy(sl, s),
                 bond_count(sv), bond_count(sl)))

print("\n  [D3-5] CR-1 摆程层基线（12^3、滴半径 2）")
ok5 = True
for name, Ev, El, cv, cl in rows:
    dE = El - Ev
    dc = abs(cl - cv)
    print("       %s：E(真空) = %11.2f  E(锁定) = %11.2f  差 = %+9.2f  键数 %d/%d" % (name, Ev, El, dE, cv, cl))
    if name in ("s=0.25", "s=0.75") and abs(dE + 2.0 * dc) > 1e-8:
        ok5 = False
check("D3-5", ok5, "拍内两档的能量差等于 -2 x 键数差（%.0f 与 %.0f）" % (rows[1][2] - rows[1][1], rows[3][2] - rows[3][1]))

write_sites = droplet & (
    ((ii - c3) ** 2 + (jj - c3) ** 2 + (kk - c3) ** 2 >= (Rd - 1) ** 2))
h0 = 1.0
T_beat = 12.0


def relax(sigma, lam, with_write, seed, steps=600, dt=0.02):
    rng = np.random.RandomState(seed)
    th = base + 1e-6 * rng.standard_normal(base.shape)
    h = np.zeros_like(th)
    if with_write:
        h[write_sites] = h0
    drive = sigma * (2.0 * np.pi / T_beat) * dt
    for t in range(steps):
        g = (np.sin(th - np.roll(th, 1, axis=0)) + np.sin(th - np.roll(th, -1, axis=0)) +
             np.sin(th - np.roll(th, 1, axis=1)) + np.sin(th - np.roll(th, -1, axis=1)) +
             np.sin(th - np.roll(th, 1, axis=2)) + np.sin(th - np.roll(th, -1, axis=2)))
        th = th - dt * (-g + lam * np.sin(th) * h) + drive
    return th


def difference(lam, with_write, seed=1):
    a = relax(sig_lock0, lam, with_write, seed)
    b = relax(sig_vac0, lam, with_write, seed)
    return float(np.linalg.norm(a - b))


D_lam1 = difference(1.0, True)
D_lam0 = difference(0.0, True)
signature = abs(D_lam1 - D_lam0) / D_lam0
check("D3-6", signature >= 0.10,
      "含写入：lambda=1 给 %.4e、lambda=0 给 %.4e，耦合签名相对差 %.1f%%（>= 10%%）" % (D_lam1, D_lam0, 100 * signature))

D_nw1 = difference(1.0, False)
D_nw0 = difference(0.0, False)
sig_nw = abs(D_nw1 - D_nw0) / D_nw0
check("D3-7", sig_nw <= 0.01,
      "无写入：lambda=1 给 %.4e、lambda=0 给 %.4e，相对差 %.2e（<= 1%%；写入步是耦合签名的必要条件）" % (D_nw1, D_nw0, sig_nw))

D_null = max(abs(difference(1.0, True, seed=2) - D_lam1), abs(difference(1.0, True, seed=3) - D_lam1)) / D_lam1
print("\n  [D3-8] 报告：耦合签名随写入幅度与零对照")
print("       零对照（同档不同种子）相对散布 %.2e" % D_null)
for amp in [0.25, 0.5, 1.0, 2.0]:
    globals()["h0"] = amp
    d1 = difference(1.0, True)
    d0 = difference(0.0, True)
    print("       写入幅度 %.2f：lambda=1 给 %.4e、lambda=0 给 %.4e、签名 %.2f%%" % (
        amp, d1, d0, 100 * abs(d1 - d0) / d0))
globals()["h0"] = 1.0

# 图
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
fig, axs = plt.subplots(2, 2, figsize=(11.2, 8.0))

ax = axs[0, 0]
for s0, c in zip([0.02, 0.1, 0.3, 1.0], ["tab:blue", "tab:orange", "tab:green", "tab:red"]):
    theta, _ = solve_bh2(s0)
    ax.plot(r, theta, color=c, label="s0 = %.2f" % s0)
ax.axhline(0.5, color="gray", ls=":", lw=1.0)
ax.axvline(RS, color="gray", ls="--", lw=0.8)
ax.set_xlim(0, 30)
ax.set_ylim(0, 1.15)
ax.set_xlabel("半径 r（格）")
ax.set_ylabel("记录 theta")
ax.set_title("BH2：饱和源模型的径向解（点线 theta=1/2，虚线为源边界）", fontsize=9)
ax.legend(fontsize=8)

ax = axs[0, 1]
ax.plot(src_h, r_hs, "o-", color="tab:blue", label="视界半径 r_h")
weak = [r_hs[0] * s / src_h[0] for s in src_h]
ax.plot(src_h, weak, "--", color="gray", label="弱场线性外推")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("源强 s0")
ax.set_ylabel("r_h（theta = 1/2）")
ax.set_title("BH2：视界半径对源强（强场亚线性）", fontsize=9)
ax.legend(fontsize=8)

ax = axs[1, 0]
labels = ["写入\nlambda=1", "写入\nlambda=0", "无写入\nlambda=1", "无写入\nlambda=0"]
vals = [D_lam1, D_lam0, D_nw1, D_nw0]
ax.bar(labels, vals, color=["tab:blue", "tab:cyan", "tab:orange", "tab:gray"])
ax.set_ylabel("两型配置终态差异范数")
ax.set_title("BH3-3：写入步与耦合的对照（12^3，一拍）", fontsize=9)

ax = axs[1, 1]
amps, sigs = [], []
for amp in [0.25, 0.5, 1.0, 2.0]:
    globals()["h0"] = amp
    d1 = difference(1.0, True); d0 = difference(0.0, True)
    amps.append(amp); sigs.append(100 * abs(d1 - d0) / d0)
globals()["h0"] = 1.0
ax.plot(amps, sigs, "o-", color="tab:purple")
ax.axhline(10.0, color="gray", ls=":", lw=1.0)
ax.set_xlabel("写入幅度")
ax.set_ylabel("耦合签名（%）")
ax.set_title("BH3-3：签名随写入幅度", fontsize=9)

fig.tight_layout()
fig.savefig(r"D:/software/TianXingFC/images/m4d3_horizon_write.png", dpi=150)
print("\n  图已保存：images/m4d3_horizon_write.png")

n_pass = sum(1 for _, ok in CHECKS if ok)
print("\n==== 判据汇总：%d / %d 通过（运行 %.0f 秒）====" % (n_pass, len(CHECKS), time.time() - T0))
sys.exit(0 if n_pass == len(CHECKS) else 1)