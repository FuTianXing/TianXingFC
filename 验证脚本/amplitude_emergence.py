# -*- coding: utf-8 -*-
r"""
amplitude_emergence.py

[振幅涌现轮] 从固定模单体集合与相位分布导出局部振幅场

定位
----
`内部草稿/注记-振幅的集体起源：单体模固定与涌现振幅的分工.md` 待做工作 2：写出涌现推导，从"单体模固定 + 相位分布"导出局部振幅场，
落点为 [[序参量份额定理：Born规则、归一化与幺正性的系综推导|序参量份额定理]] §3.2 的"单粒子描述是集体相位模式在公共参考系上的约化"。
本脚本把该推导做成可复算读数：分格粗粒化 → 局部集体相位与局部相干度 → 振幅分配 $A_a=\sqrt{N_a/N}\,\eta_a$ 与统计幅度 $c_a=\sqrt{N_a/N}e^{i\Phi_a}$
→ 公共参考系约化（规范冗余）→ 采样充填（格间涨落按 $N_a^{-1/2}$）→ 动力学（匀速旋转给出 $i\partial_tA=\omega A$；相邻耦合使相干上升）。
本脚本不改动基准文件。

模型
----
一维元相子链（$L$ 个元相子，单体一份单位振幅、模固定）：相位 $\theta_i = \Phi(x_i) + \delta_i$，
$\Phi(x) = qx$ 为光滑斜坡（整条链一个全周，$q = 2\pi/L$），$\delta_i$ 为两值噪声 $\pm\delta_0$（等概率，单体本征相干因子 $r = \cos\delta_0$）。
分格后格内量：$\eta_a = \frac{1}{N_a}\sum_{i\in a}e^{i\theta_i}$、$\Phi_a = \arg\sum_{i\in a}e^{i\theta_i}$、
$c_a = \sqrt{N_a/N}e^{i\Phi_a}$（$|c_a|^2$ 为份额）、$A_a = \sqrt{N_a/N}\,\eta_a$。

判据（先立判据，再实现；预期写错时修正判据并记录教训）
----
EMG-1 振幅分配：$|A_a| = \sqrt{N_a/N}\,|\eta_a|$（恒等）；无噪声时 $\eta_a$ 精确等于格内斜坡平均；
   有噪声时多次实现的 $\eta_a/\text{斜坡}$ 平均值趋于 $r = \cos\delta_0$（±0.03）。
EMG-2 相位跟随：无噪声时 $\Phi_a$ 精确等于格内斜坡相位；$\delta_0 = 0.6$ 时逐格偏差平均 $\le 0.25$ rad。
EMG-3 相位梯度：相邻格相位步 $\equiv qN_a\ (\mathrm{mod}\ 2\pi)$；无噪声精确，有噪声时圆形均值偏差 $\le 0.05$ rad。
EMG-4 公共参考系约化：全体相位重标使 $A_a \to A_a e^{i\chi}$，$|A_a|$、格间相位差、份额全不变（规范冗余）。
EMG-5 采样充填：$\mathrm{Im}\,\eta_a$ 的格间标准差 $= \sin\delta_0/\sqrt{N_a}$（逐点 ±25%），双对数斜率 $\approx -0.5$（±0.15）。
EMG-6 动力学：(a) 匀速旋转给 $A_a(t)/A_a(0) = e^{i\omega t}$（精确）；(b) 相邻耦合下平均 $|\eta_a|$ 上升（相位锁定）。

运行
----
python 验证脚本/amplitude_emergence.py
确定性（固定种子序列），运行 < 60 秒。图存 D:/software/TianXingFC/images/amplitude_emergence.png
"""
import time

import numpy as np

T0 = time.time()
CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print(f"  [{tag}] {'PASS' if ok else 'FAIL'}  {detail}")


def build_chain(L, q, delta0, seed=20260917):
    rng = np.random.default_rng(seed)
    x = np.arange(L, dtype=float)
    signs = np.where(rng.random(L) < 0.5, -1.0, +1.0)
    return q * x + signs * delta0, q * x


def cells(L, sizes):
    out, k = [], 0
    for s in sizes:
        if k + s > L:
            break
        out.append(np.arange(k, k + s))
        k += s
    return out


def cell_quantities(theta, cell_list):
    N = sum(len(c) for c in cell_list)
    eta, phi, share = [], [], []
    for c in cell_list:
        z = np.exp(1j * theta[c]).sum() / len(c)
        eta.append(z)
        phi.append(np.angle(z))
        share.append(len(c) / N)
    eta, phi, share = np.array(eta), np.array(phi), np.array(share)
    return eta, phi, share, np.sqrt(share) * eta, N


def circ_mean(angles):
    return np.angle(np.exp(1j * np.array(angles)).mean())


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


print("=" * 74)
print("[振幅涌现轮] 从固定模单体集合到局部振幅场")
print("=" * 74)

L = 512
Q = 2.0 * np.pi / L
SIZES = [L // 8] * 8
QS = [float(s) / L for s in SIZES]
print(f"\nA 部分：振幅分配与相位跟随（L = {L}，q = 2pi/L，等分 8 格，格内斜坡跨度 {Q * SIZES[0]:.4f}）")

# 无噪声：精确
theta, base = build_chain(L, Q, 0.0)
cl = cells(L, SIZES)
eta, phi, share, A, N = cell_quantities(theta, cl)
ramp = np.array([np.exp(1j * base[c]).sum() / len(c) for c in cl])
ident = np.max(np.abs(np.abs(A) - np.sqrt(share) * np.abs(eta)))
d_eta = np.max(np.abs(eta - ramp))
d_ph = np.max(np.abs(wrap(phi - np.angle(ramp))))
print(f"  无噪声：|A| = sqrt(share)|eta| 恒等偏差 {ident:.2e}；eta 对斜坡平均偏差 {d_eta:.2e}；相位偏差 {d_ph:.2e}")
check("EMG-1a", ident < 1e-12 and d_eta < 1e-12,
      f"振幅分配恒等且 eta = 格内斜坡平均（偏差 {ident:.2e} 与 {d_eta:.2e}）")
check("EMG-2", d_ph < 1e-12, f"无噪声时格相位 = 斜坡相位（偏差 {d_ph:.2e}）")

# 有噪声：系综平均
K = 64
for delta0 in (0.2, 0.6, 1.0):
    acc = 0j
    devs = []
    for k in range(K):
        th, bs = build_chain(L, Q, delta0, seed=1000 + k)
        et, ph, sh, _, _ = cell_quantities(th, cells(L, SIZES))
        rp = np.array([np.exp(1j * bs[c]).sum() / len(c) for c in cells(L, SIZES)])
        acc += np.mean(et / rp)
        devs.append(np.mean(np.abs(wrap(ph - np.angle(rp)))))
    print(f"  delta0 = {delta0:4.2f}：<eta/斜坡> = {abs(acc / K):.4f}（理论 r = {np.cos(delta0):.4f}）；"
          f"逐格相位偏差平均 {np.mean(devs):.3f} rad")
    if delta0 == 0.6:
        check("EMG-1b", abs(abs(acc / K) - np.cos(delta0)) < 0.03,
              f"系综平均 eta/斜坡 = {abs(acc / K):.4f} 对 r = {np.cos(delta0):.4f}")
        check("EMG-2b", np.mean(devs) <= 0.25, f"逐格相位偏差平均 {np.mean(devs):.3f} rad <= 0.25")

print("\nB 部分：相位梯度（相邻格相位步 对 q*N_a，模 2pi）")
step_tgt = Q * SIZES[0]
steps = np.diff(phi)
print(f"  无噪声：相位步余数（模 2pi 居中）取值集 = {sorted(set(np.round(wrap(steps - step_tgt), 12)))}；目标 {step_tgt:.6f}")
check("EMG-3a", np.max(np.abs(wrap(steps - step_tgt))) < 1e-12,
      f"无噪声时相位步精确等于 q*N_a（模 2pi，偏差 {np.max(np.abs(wrap(steps - step_tgt))):.2e}）")
dev_circ = []
for delta0 in (0.2, 0.6, 1.0):
    th, _ = build_chain(L, Q, delta0)
    _, ph, _, _, _ = cell_quantities(th, cells(L, SIZES))
    dev_circ.append(abs(wrap(circ_mean(np.diff(ph)) - step_tgt)))
print(f"  有噪声时圆形均值偏差（delta0 = 0.2/0.6/1.0）：{[f'{d:.2e}' for d in dev_circ]}")
check("EMG-3b", max(dev_circ) <= 0.05, f"圆形均值仍等于 q*N_a（最大偏差 {max(dev_circ):.2e}）")

print("\nC 部分：公共参考系约化（全局相位重标）")
theta, base = build_chain(L, Q, 0.6)
cl = cells(L, SIZES)
eta, phi, share, A, N = cell_quantities(theta, cl)
CHI = 0.7
eta2, phi2, share2, A2, _ = cell_quantities(theta + CHI, cl)
same_mod = np.allclose(np.abs(A2), np.abs(A))
same_rel = np.allclose(wrap(phi2 - phi2[0]), wrap(phi - phi[0]))
same_share = np.allclose(share2, share)
same_ph = np.allclose(wrap(np.angle(A2) - np.angle(A) - CHI), 0.0, atol=1e-9)
print(f"  重标 chi = {CHI}：|A| 不变 {same_mod}；格间相位差不变 {same_rel}；份额不变 {same_share}；A 整体乘 e^{{i chi}} {same_ph}")
check("EMG-4", same_mod and same_rel and same_share and same_ph,
      "全局相位重标只改 A 的整体相位（规范冗余），物理读数不变")

print("\nD 部分：采样充填（相对斜坡的横向分量 Im(eta/斜坡) 的格间标准差对格内人数）")
mults = [4, 8, 16, 32, 64]
DELTA_D = 0.3
stds = []
for m in mults:
    szs = [m] * (L // m)
    vals = []
    for k in range(64):
        th, bs2 = build_chain(L, Q, DELTA_D, seed=500 + k)
        eth, _, _, _, _ = cell_quantities(th, cells(L, szs))
        cl2 = cells(L, szs)
        rp2 = np.array([np.exp(1j * bs2[c]).sum() / len(c) for c in cl2])
        vals.append(np.std(np.imag(eth / rp2)))
    stds.append(np.mean(vals))
stds = np.array(stds)
pred = np.sin(DELTA_D) / np.sqrt(np.array(mults, dtype=float))
fit = np.polyfit(np.log(mults), np.log(stds), 1)
print(f"  格内人数 {mults}：标准差实测 {[f'{s:.5f}' for s in stds]}")
print(f"                  理论 sin(delta)/sqrt(N_a) {[f'{s:.5f}' for s in pred]}；相对偏差 "
      f"{[f'{abs(a-b)/b:.3f}' for a, b in zip(stds, pred)]}")
print(f"  双对数拟合斜率 = {fit[0]:.4f}（预期 -0.5）")
check("EMG-5", np.all(np.abs(stds - pred) / pred < 0.25) and abs(fit[0] + 0.5) < 0.15,
      f"标准差逐点符合 sin(delta0)/sqrt(N_a)（最大相对偏差 "
      f"{np.max(np.abs(stds - pred) / pred):.3f}），斜率 {fit[0]:.4f}")

print("\nE 部分：动力学")
Ld = 256
omega, Jc, dt = 0.3, 0.5, 0.01
theta0, _ = build_chain(Ld, 0.0, 0.8, seed=11)
clD = cells(Ld, [Ld // 8] * 8)
_, _, shD, A0, _ = cell_quantities(theta0, clD)
t1 = 2.0
_, _, _, A1, _ = cell_quantities(theta0 + omega * t1, clD)
ratio_dev = np.max(np.abs(A1 / A0 - np.exp(1j * omega * t1)))
print(f"  匀速旋转：A(t)/A(0) 对 e^{{i omega t}} 的最大偏差 {ratio_dev:.2e}")
check("EMG-6a", ratio_dev < 1e-12, f"A(t)/A(0) = e^(i omega t)（偏差 {ratio_dev:.2e}）")
th = theta0.copy()
for _ in range(int(3.0 / dt)):
    th = th + dt * (omega + Jc * (np.sin(np.roll(th, 1) - th) + np.sin(np.roll(th, -1) - th)))
_, _, _, A_end, _ = cell_quantities(th, clD)
c0 = float(np.mean(np.abs(A0) / np.sqrt(shD)))
c1 = float(np.mean(np.abs(A_end) / np.sqrt(shD)))
print(f"  相邻耦合锁定：平均 |eta| 由 {c0:.4f} 升到 {c1:.4f}（局部振幅随之增大）")
check("EMG-6b", c1 > c0 + 0.1, f"耦合使平均相干上升（{c0:.4f} 到 {c1:.4f}）")

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

    th, bs = build_chain(128, 2 * np.pi / 128, 0.6, seed=5)
    clg = cells(128, [16] * 8)
    et, ph, shg, Ag, _ = cell_quantities(th, clg)
    rp = np.array([np.exp(1j * bs[c]).sum() / len(c) for c in clg])

    ax = axs[0]
    ax.plot(np.arange(128), th, '.', ms=3, color="#1f77b4", label=r"元相子相位 $\theta_i$")
    ax.plot(np.arange(128), bs, '-', color="#d95f02", lw=1.2, label=r"光滑斜坡 $\Phi(x)=qx$")
    for c in clg:
        ax.axvspan(c[0] - 0.5, c[-1] + 0.5, color="#eeeeee", zorder=0)
    ax.set_xlabel("格点 $x$")
    ax.set_ylabel("相位")
    ax.set_title("(a) 相位分布与分格（灰带为格）", fontsize=10.5)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)

    ax = axs[1]
    ax.plot(shg, np.abs(et), 'o', color="#1f77b4", label=r"局部相干度 $|\eta_a|$")
    ax.plot(shg, np.abs(Ag) / np.sqrt(shg), 's', mfc='none', color="#d95f02",
            label=r"局部振幅 $|A_a|/\sqrt{s_a}$")
    ax.plot(shg, np.abs(rp) * np.cos(0.6), ':', color="#444444", lw=1.2,
            label=r"理论 $r\,|$斜坡$|$")
    ax.set_xlabel(r"份额 $s_a = N_a/N$")
    ax.set_ylabel("相干度")
    ax.set_title("(b) 局部振幅 = 份额开方乘局部相干度", fontsize=10.5)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)

    ax = axs[2]
    ax.loglog(mults, stds, 'o-', color="#1b1b1b", label="实测格间标准差")
    ax.loglog(mults, pred, 's--', color="#1f77b4", lw=1.0, label=r"理论 $\sin\delta_0/\sqrt{N_a}$")
    ax.loglog(mults, stds[0] * (np.array(mults) / mults[0]) ** -0.5, '--', color="#d95f02",
              lw=1.2, label=r"$N_a^{-1/2}$ 参考线")
    ax.set_xlabel(r"格内人数 $N_a$")
    ax.set_ylabel(r"$\mathrm{Im}(\eta_a/\mathrm{ramp})$ 的格间标准差")
    ax.set_title(f"(c) 采样充填：斜率 {fit[0]:.3f}", fontsize=10.5)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3, which='both')

    fig.tight_layout()
    fig.savefig("D:/software/TianXingFC/images/amplitude_emergence.png", dpi=150)
    print("\n图已保存：images/amplitude_emergence.png")
except ImportError:
    print("\n跳过绘图：matplotlib 不可用")

print(f"\n[振幅涌现轮] 完成；总运行 {time.time() - T0:.1f} s。")
