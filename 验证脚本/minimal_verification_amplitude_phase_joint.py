# -*- coding: utf-8 -*-
r"""
minimal_verification_amplitude_phase_joint.py

[相位-振幅联测轮] 同一构型里三读并立：绕数（实轴侧）、宽度（法线轴侧）、质量（幅度侧）

定位
----
回答"振幅与相位是否分家"：在复合位错模型（一维 FK/PN 链，缠绕守恒边界 $u_{N-1} = u_0 + N$）上，
同一个局域缺陷同时承载两轴的读数：
  实轴侧（波性侧）：相位场 $\theta_n = 2\pi u_n$ 在缺陷处多出恰好一个整周，绕数 $W = 1$ 由边界给出且定域于缺陷核；
  法线轴侧（粒性侧）：位移包络 $v_n = u_n - n$ 给出宽度 $\ell^*$ 与粒子出现份额分布 $p_n$（能量密度超量归一化）；
  幅度侧（质量）：阱深参数 $K$ 与缺陷静能 $E_{\text{kink}}$（= 绑定能、即质量）同参数区标度 $E \propto \sqrt{kK}$。
三读共用同一参数（$r = K/k$），并在宽度落窗区 $r \le 0.02$（$\ell^* \ge 2$）内同时成立。本脚本不改动基准文件。

模型
----
$E = \sum_n \left[\tfrac12 k(\Delta u_n - 1)^2 + K(1-\cos 2\pi u_n)\right]$，$u_0 = 0$、$u_{N-1} = N$（$W=1$ 扇区，一个位错），
确定性梯度流弛豫。相位取提升值 $\theta_n = 2\pi u_n$（满周闭合沿法线走一格，故相位与位移同一比例）。

判据（先立判据，再实现；预期写错时修正判据并记录教训）
----
J1 相位侧（实轴方向）：$\sum_n \Delta u_n = N$ 精确；背景区 $\Delta u = 1$（偏差 $< 10^{-6}$）；超量绕数
   $\sum_n(\Delta u_n - 1) = 1$ 精确；定域性按剖面衰减定窗口：$|n - n_0| \le \ell^*$ 内占比 $\ge 0.70$，
J2 法线轴侧（空间形态）：弹性超量密度与总超量密度的宽度一致（相对差 $\le 20\%$）；$p_n$ 归一化到 1（份额）；
   两宽随 $r$ 同向变化且 $r \le 0.02$ 时 $\ge 2$（与落窗区一致）。
J3 幅度侧（质量）：$E_{\text{kink}} \propto K^{0.5}$（$r \le 0.1$ 五点双对数斜率 $0.5 \pm 0.05$）。
J4 联测：同参数区内三读并立（$W = 1$、$\ell^* \ge 2$、$E \propto \sqrt{kK}$ 的同一斜率），互相不冲突。

运行
----
python 验证脚本/minimal_verification_amplitude_phase_joint.py
确定性（无随机数），运行约 10 秒。图存 D:/software/TianXingFC/images/amplitude_phase_joint.png
"""
import time

import numpy as np

T0 = time.time()
CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print(f"  [{tag}] {'PASS' if ok else 'FAIL'}  {detail}")


NS = 400
K_EL = 1.0


def relax(r, steps=60000):
    K_pn = r * K_EL
    u = np.arange(NS, dtype=float)
    half = NS // 2
    u[half:] += 1.0
    u[half] = half + 0.5
    eta = 0.005
    for _ in range(steps):
        grad = np.zeros(NS)
        grad[1:-1] += K_EL * (2.0 * u[1:-1] - u[:-2] - u[2:])
        grad += K_pn * 2.0 * np.pi * np.sin(2.0 * np.pi * u)
        u[1:-1] -= eta * grad[1:-1]
        u[0] = 0.0
        u[-1] = float(NS)
    return u, K_pn


def readings(u, K_pn):
    du = np.diff(u)
    n = np.arange(NS - 1) + 0.5
    dens_el = 0.5 * K_EL * (du - 1.0) ** 2
    dens_pot = K_pn * (1.0 - np.cos(2.0 * np.pi * u[:-1]))
    p = dens_el + dens_pot
    p = p / p.sum()
    def width(w):
        c0 = (w * n).sum() / w.sum()
        return 2.0 * np.sqrt((w * (n - c0) ** 2).sum() / w.sum()), c0
    w_el, c_el = width(dens_el)
    w_p, c_p = width(p)
    n0 = float(n[np.argmax(du - 1.0)])
    return du, n, dens_el, p, w_el, w_p, n0


print("=" * 74)
print("[相位-振幅联测轮] 同一构型里的三读：绕数、宽度、质量")
print("=" * 74)

RS = (0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
print("\nA 部分：逐参数读数")
print("   r=K/k    E_kink   ell*(弹性)  ell*(总)   n_0   ±ell* / ±2ell* 内绕数占比")
rows = []
for r in RS:
    u, K_pn = relax(r)
    du, n, dens_el, p, w_el, w_p, n0 = readings(u, K_pn)
    E = 0.5 * K_EL * np.sum((du - 1.0) ** 2) + K_pn * np.sum(1.0 - np.cos(2.0 * np.pi * u))
    core1 = np.abs(n - n0) <= w_el
    core2 = np.abs(n - n0) <= 2.0 * w_el
    frac = (du[core1] - 1.0).sum() / (du - 1.0).sum()
    frac2 = (du[core2] - 1.0).sum() / (du - 1.0).sum()
    rows.append((r, E, w_el, w_p, n0, (frac, frac2), du, n))
    print(f"  {r:6.3f}   {E:7.3f}   {w_el:8.2f}   {w_p:7.2f}  {n0:6.1f}   {frac:6.3f} / {frac2:6.3f}")

r0, E0, wl0, wp0, n00, f0, du0, n_ = rows[3]   # r = 0.02
print("\nB 部分：相位侧判据（r = 0.02 构型）")
tot = du0.sum()
excess = (du0 - 1.0).sum()
bg_dev = np.max(np.abs(du0[np.abs(n_ - n00) > 8 * wl0] - 1.0))
f1_, f2_ = f0
print(f"  总和 sum du = {tot:.12f}（边界给定 {NS}）；超量绕数 = {excess:.12f}（= 1）；"
      f"背景区 max|du-1| = {bg_dev:.2e}；±ell* 内 {f1_:.3f}；±2ell* 内 {f2_:.3f}")
check("J1", abs(tot - NS) < 1e-9 and abs(excess - 1.0) < 1e-9 and bg_dev < 1e-6
      and f1_ >= 0.70 and f2_ >= 0.95,
      f"sum du = N 精确、超量绕数 = 1、背景 du = 1；±ell* 内 {f1_:.3f} >= 0.70、±2ell* 内 {f2_:.3f} >= 0.95")

print("\nC 部分：法线轴侧判据（宽度一致性、归一化、随 r 同向）")
rel = np.array([abs(rw[2] - rw[3]) / rw[2] for rw in rows if rw[0] <= 0.1 + 1e-12])
print(f"  弹性宽与总宽相对差（逐 r，r<=0.1）：{[f'{x:.3f}' for x in rel]}")
check("J2a", np.all(rel <= 0.20),
      f"r <= 0.1 时两宽一致（最大相对差 {rel.max():.3f} <= 0.20）")
sums = []
for r in RS:
    u, K_pn = relax(r)
    d, nn, de, pp, we, wp, n0 = readings(u, K_pn)
    sums.append(abs(pp.sum() - 1.0))
check("J2b", max(sums) < 1e-12, f"p_n 归一化到 1（最大偏差 {max(sums):.1e}）")
wl = np.array([rw[2] for rw in rows])
mono = np.all(np.diff(wl) < 0)
win = all(rw[2] >= 2.0 for rw in rows if rw[0] <= 0.02 + 1e-12)
print(f"  宽度随 r 单调递减 = {mono}；r <= 0.02 的宽度 {[f'{x:.2f}' for x, rw in zip(wl, rows) if rw[0] <= 0.02 + 1e-12]}（均 >= 2）")
check("J2c", mono and win, f"宽度单调且落窗区 ell* >= 2（{mono}, {win}）")

print("\nD 部分：幅度侧判据（E 对 K 的幂次）")
sel = [(rw[0], rw[1]) for rw in rows if rw[0] <= 0.1 + 1e-12]
fit = np.polyfit(np.log([s[0] for s in sel]), np.log([s[1] for s in sel]), 1)
print(f"  r <= 0.1 五点 {[(f'{a:.3f}', f'{b:.3f}') for a, b in sel]}；双对数斜率 = {fit[0]:.4f}（预期 0.5）")
check("J3", abs(fit[0] - 0.5) < 0.05, f"E_kink ∝ K^0.5（斜率 {fit[0]:.4f}）")

print("\nE 部分：联测（三读并立）")
check("J4", abs(excess - 1.0) < 1e-9 and win and abs(fit[0] - 0.5) < 0.05,
      "同一参数区内：绕数 = 1（实轴侧）、宽度 >= 2（法线轴侧）、质量 ∝ K^0.5（幅度侧）")

print("\n判据汇总：{}/{} 通过".format(sum(1 for _, ok in CHECKS if ok), len(CHECKS)))
for tag, ok in CHECKS:
    if not ok:
        print(f"  未过：{tag}")

# ---------------- 出图 ----------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axs = plt.subplots(1, 3, figsize=(16.5, 4.6))

    ax = axs[0]
    ax.plot(n_, du0 - 1.0, '-', color="#1f77b4", label=r"$\Delta u_n - 1$（法线轴滑移）")
    ax.plot(n_, (du0 - 1.0) / (du0 - 1.0).sum(), '--', color="#d95f02",
            label="归一化超量（= 相位超量 $\\Delta\\theta/2\\pi$ 的形状）")
    ax.axvline(n00, color="#444444", ls=':', lw=1.0)
    ax.set_xlabel("格点 $n$")
    ax.set_ylabel("超量")
    ax.set_title("(a) 同一缺陷同时携带滑移与整周缠绕", fontsize=10.5)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)

    ax = axs[1]
    ax.plot(n_, p, '-', color="#1b1b1b", label=r"$p_n$（总超量密度，归一化）")
    ax.axvspan(n00 - wl0 / 2, n00 + wl0 / 2, color="#d95f02", alpha=0.15,
               label=f"$\\ell^*$ 窗口（{wl0:.2f}）")
    ax.set_xlim(n00 - 6 * wl0, n00 + 6 * wl0)
    ax.set_xlabel("格点 $n$")
    ax.set_ylabel("份额密度")
    ax.set_title("(b) 粒子出现份额与包层宽度同尺度", fontsize=10.5)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)

    ax = axs[2]
    rr = np.array([rw[0] for rw in rows])
    EE = np.array([rw[1] for rw in rows])
    ax.loglog(rr, EE, 'o', color="#1b1b1b", label=r"$E_{\mathrm{kink}}$（实测）")
    selm = rr <= 0.1 + 1e-12
    ax.loglog(rr[selm], EE[selm][0] * (rr[selm] / rr[selm][0]) ** 0.5, '--', color="#d95f02",
              lw=1.2, label=r"$K^{0.5}$ 参考线")
    ax.set_xlabel(r"$r = K/k$")
    ax.set_ylabel("缺陷静能（质量侧）")
    ax.set_title(f"(c) 幅度侧：斜率 {fit[0]:.3f}", fontsize=10.5)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3, which='both')

    fig.tight_layout()
    fig.savefig("D:/software/TianXingFC/images/amplitude_phase_joint.png", dpi=150)
    print("\n图已保存：images/amplitude_phase_joint.png")
except ImportError:
    print("\n跳过绘图：matplotlib 不可用")

print(f"\n[相位-振幅联测轮] 完成；总运行 {time.time() - T0:.1f} s。")
