# -*- coding: utf-8 -*-
"""BH3-3 后续（时标侧）：格点退出的完成时标与重标瞬时性。

判据（先立判据，再实现；均可证伪）：
  K0 静态锚点：复现 BH3-5 一维先导的静态读数（结构留痕 ≈ 2.13e-2、
     结构/质量比 ≈ 0.202、线性分解残差 ≈ 0.018；相对偏差 ≤ 5%）。
  K1 静止极限：每个 γ 档时间演化末态与静态解一致（末态偏差 ≤ 1e-6 × e(0)）。
  K2a 小 γ 极限：完成时标 τ_tail = 2/λ_E 随 1/γ 标度（ln τ 对 ln γ 斜率 ≈ −1，容差 ±0.1）。
  K2b 大 γ 极限：τ_tail 随 γ 标度（斜率 ≈ +1，容差 ±0.1）。
  K3 慢模对照：能量衰减率 λ_E 与解析对照（小 γ 对 γ；大 γ 对 2Ω1²/γ；偏差 ≤ 10%）。
  K4 视界语境实例化：链格数 = r_h、γ = π/r_h（S2R 校准口径）时，
     τ_tail / r_h ∈ [0.5, 0.8]（解析 2/π ≈ 0.637）。

模型：一维链、格距 1、Dirichlet 0 边界（两端恒为 0）；静态方程 Lφ = −s；
      基本动力学取阻尼波动 φ_tt + γ φ_t = Lφ + s（裁决十六）。
      退出构型与 BH3-5 一致：101 格、源 46..55（各 1）、退出点 50；
      新链 100 格、源 {46..55} 去 50；场按纯标记重标（k<50 → k，k≥50 → k+1）。
      重标本身为零时标（瞬时、无传播）；可测完成时标取记录场弛豫到新稳态的时间：
      主口径为能量尾段率（τ_tail = 2/λ_E，振幅降 1/e 的时间），旁证口径为 e(t) 首次到 1/e。
确定性（无随机数）。图存 D:/software/TianXingFC/images/bh33_exit_timescale.png。
"""

import sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))


def solve_poisson(n, src):
    A = np.zeros((n, n))
    b = -np.asarray(src, dtype=float)
    for i in range(n):
        A[i, i] = -2.0
        if i > 0:
            A[i, i - 1] = 1.0
        if i < n - 1:
            A[i, i + 1] = 1.0
    A[0, :] = 0.0
    A[0, 0] = 1.0
    b[0] = 0.0
    A[-1, :] = 0.0
    A[-1, -1] = 1.0
    b[-1] = 0.0
    return np.linalg.solve(A, b)


def lap(phi):
    out = -2.0 * phi
    out[1:] += phi[:-1]
    out[:-1] += phi[1:]
    return out


def evolve(n, src, phi_init, phi_ref, gamma, dt, t_end):
    steps = int(round(t_end / dt))
    phi = phi_init.copy()
    phi_prev = phi_init.copy()
    s = np.asarray(src, dtype=float)
    c1 = 1.0 + gamma * dt / 2.0
    c2 = 1.0 - gamma * dt / 2.0
    hist = np.empty(steps + 1)
    energy = np.empty(steps + 1)
    dev = phi - phi_ref
    hist[0] = float(np.max(np.abs(dev)))
    energy[0] = float(np.sum(dev * dev))
    for k in range(1, steps + 1):
        phi_new = (2.0 * phi - c2 * phi_prev + dt * dt * (lap(phi) + s)) / c1
        phi_new[0] = 0.0
        phi_new[-1] = 0.0
        phi_prev = phi
        phi = phi_new
        dev = phi - phi_ref
        hist[k] = float(np.max(np.abs(dev)))
        energy[k] = float(np.sum(dev * dev))
    return phi, hist, energy


print("==== BH3-3 后续（时标侧）：格点退出完成时标 ====")

# ---------- K0 静态锚点（复现 BH3-5 先导） ----------
base_src = list(range(46, 56))
drop = 50


def src_profile(n, keep):
    s = np.zeros(n)
    for i in keep:
        if 0 <= i < n:
            s[i] = 1.0
    return s


phi0 = solve_poisson(101, src_profile(101, base_src))
phi_struct = solve_poisson(100, src_profile(100, base_src))
phi_mass = solve_poisson(101, src_profile(101, [i for i in base_src if i != drop]))
phi_true = solve_poisson(100, src_profile(100, [i for i in base_src if i != drop]))

amp0 = float(np.max(np.abs(phi0)))
d_struct = phi_struct - phi0[:100]
d_mass = phi_mass[:100] - phi0[:100]
d_true = phi_true - phi0[:100]
r_struct = float(np.max(np.abs(d_struct)) / amp0)
r_mass = float(np.max(np.abs(d_mass)) / amp0)
ratio = r_struct / r_mass
resid = float(np.max(np.abs(d_true - (d_struct + d_mass))) / np.max(np.abs(d_true)))
print("  静态读数：结构 %.5f；质量 %.5f；比 %.4f；残差 %.4f" % (r_struct, r_mass, ratio, resid))
check("K0", abs(r_struct - 0.0213) / 0.0213 <= 0.05
      and abs(r_mass - 0.105) / 0.105 <= 0.05
      and abs(ratio - 0.202) / 0.202 <= 0.05
      and abs(resid - 0.018) / 0.018 <= 0.05,
      "与 BH3-5 先导一致（2.13e-2 / 0.105 / 0.202 / 0.018）")

phi_relabel = np.empty(100)
phi_relabel[:50] = phi0[:50]
phi_relabel[50:] = phi0[51:]
new_src = src_profile(100, [i for i in base_src if i != drop])
e0 = float(np.max(np.abs(phi_relabel - phi_true)))
E0 = float(np.sum((phi_relabel - phi_true) ** 2))
print("  重标初值到新稳态：e(0) = %.5f（基准幅度 %.5f）" % (e0, amp0))

n = 100
Omega1 = 2.0 * np.sin(np.pi / (2.0 * (n + 1)))
lam1 = Omega1 * Omega1
print("  模式谱：Omega1 = %.5f；交叉 γ* = 2 Omega1 = %.5f" % (Omega1, 2 * Omega1))


def tau_pred(gamma):
    if gamma < 2.0 * Omega1:
        return 2.0 / gamma
    return 2.0 / (gamma - np.sqrt(gamma * gamma - 4.0 * lam1))


gamma_scan = [0.02, 0.05, 0.1, 0.3, 1.0, 2.0]
dt = 0.05
rows = []
curves = {}
for gamma in gamma_scan:
    tp = tau_pred(gamma)
    t_end = 16.0 * tp
    phi_final, hist, energy = evolve(n, new_src, phi_relabel, phi_true, gamma, dt, t_end)
    curves[gamma] = (hist, energy)
    tgrid = np.arange(len(hist)) * dt
    idx = int(np.argmax(hist <= e0 / np.e)) if np.any(hist <= e0 / np.e) else -1
    tau_1e = tgrid[idx] if idx >= 0 else float("nan")
    win = energy <= 1e-6 * E0
    tt = tgrid[win]
    ee = energy[win]
    slope = np.polyfit(tt, np.log(ee), 1)[0]
    lam_E = -slope
    tau_tail = 2.0 / lam_E
    lam_ref = gamma if gamma < 2.0 * Omega1 else (gamma - np.sqrt(gamma * gamma - 4.0 * lam1))
    rows.append((gamma, tau_1e, tau_tail, tau_tail / tp, lam_E, lam_ref, lam_E / lam_ref, hist[-1] / e0))
    print("  γ=%.2f：τ(1/e)=%.1f；τ_tail=%.1f（预测 %.1f，比 %.3f）；λ_E=%.4e 对解析 %.4e（比 %.3f）；末态 %.1e"
          % (gamma, tau_1e, tau_tail, tp, tau_tail / tp, lam_E, lam_ref, lam_E / lam_ref, hist[-1] / e0))

k1_ok = all(r[7] <= 1e-6 for r in rows)
check("K1", k1_ok, "末态偏差最大 %.1e（阈值 1e-6 × e(0)）" % max(r[7] for r in rows))

small = [r for r in rows if r[0] < 2 * Omega1]
large = [r for r in rows if r[0] > 2 * Omega1]
sl_small = np.polyfit(np.log([r[0] for r in small]), np.log([r[2] for r in small]), 1)[0]
sl_large = np.polyfit(np.log([r[0] for r in large]), np.log([r[2] for r in large]), 1)[0]
check("K2a", abs(sl_small + 1.0) <= 0.10, "小 γ 斜率 %.3f（期望 −1）" % sl_small)
check("K2b", abs(sl_large - 1.0) <= 0.10, "大 γ 斜率 %.3f（期望 +1）" % sl_large)

k3_dev = max(abs(r[6] - 1.0) for r in rows)
check("K3", k3_dev <= 0.10, "λ_E 对解析最大偏差 %.1f%%" % (100 * k3_dev))

# K4 视界语境实例化
rh = 1000
gamma_h = np.pi / rh
src_h = np.zeros(rh)
src_h[rh // 2] = 1.0
phi_h = solve_poisson(rh, src_h)
phi_rel = np.concatenate([phi_h[:rh // 2], phi_h[rh // 2 + 1:]])
phi_ref_h = np.zeros(rh - 1)
src_h_new = np.zeros(rh - 1)
dt_h = 0.2
phi_fin_h, hist_h, energy_h = evolve(rh - 1, src_h_new, phi_rel, phi_ref_h, gamma_h, dt_h, 16.0 * (2.0 / gamma_h))
tgrid_h = np.arange(len(hist_h)) * dt_h
E0h = energy_h[0]
win = energy_h <= 1e-6 * E0h
lam_h = -np.polyfit(tgrid_h[win], np.log(energy_h[win]), 1)[0]
tau_tail_h = 2.0 / lam_h
ratio_h = tau_tail_h / rh
idxh = int(np.argmax(hist_h <= hist_h[0] / np.e))
tau_1e_h = idxh * dt_h
print("  K4：r_h=%d、γ=π/r_h=%.6f；τ_tail=%.1f 拍（τ(1/e)=%.1f）；τ_tail/r_h=%.3f（解析 2/π=0.637）"
      % (rh, gamma_h, tau_tail_h, tau_1e_h, ratio_h))
check("K4", 0.5 <= ratio_h <= 0.8, "τ_tail/r_h = %.3f" % ratio_h)

# ---------- 图 ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
fig, axs = plt.subplots(1, 2, figsize=(11.8, 4.5))

ax = axs[0]
for gamma in gamma_scan:
    hist, energy = curves[gamma]
    tgrid = np.arange(len(hist)) * dt
    ax.semilogy(tgrid, np.maximum(hist, 1e-14 * e0) / e0, label="γ = %.2f" % gamma)
ax.axhline(1.0 / np.e, color="gray", ls=":", lw=1)
ax.set_xlabel("时间（拍）")
ax.set_ylabel("e(t) / e(0)")
ax.set_title("(a) 退出现场弛豫（不同阻尼）")
ax.legend(fontsize=8)
ax.set_ylim(1e-9, 1.5)

ax = axs[1]
gg = np.array([r[0] for r in rows])
tt_ = np.array([r[2] for r in rows])
ax.loglog(gg, tt_, "o-", label="实测 τ_tail")
gline = np.logspace(np.log10(0.015), np.log10(2.2), 100)
tline = np.where(gline < 2 * Omega1, 2.0 / gline,
                 2.0 / (gline - np.sqrt(np.maximum(gline * gline - 4.0 * lam1, 1e-30))))
ax.loglog(gline, tline, "k--", label="解析：欠阻尼 2/γ、过阻尼 γ/Ω^2")
ax.set_xlabel("阻尼 γ（每拍）")
ax.set_ylabel("完成时标 τ（拍）")
ax.set_title("(b) τ 对 γ（两极限斜率 -1 与 +1）")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(r"D:/software/TianXingFC/images/bh33_exit_timescale.png", dpi=140)
print("  图已存 images/bh33_exit_timescale.png")

npass = sum(1 for _, ok in CHECKS if ok)
print("==== 判据 %d/%d 通过 ====" % (npass, len(CHECKS)))
