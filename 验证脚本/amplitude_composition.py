# -*- coding: utf-8 -*-
r"""
amplitude_composition.py

[振幅合成轮] 宏观振幅 = 份额开方乘块内相干度：相干/漫射分解与三条竞争写法的对照

定位
----
`内部草稿/注记-振幅的集体起源：单体模固定与涌现振幅的分工.md` 留了第一项待做工作：宏观振幅的集体载体取份额、取相干度，还是取合成。
本脚本用一个可精确枚举的系综模型判定合成形式，并与三种竞争写法逐点对照：
  本脚本主张：宏观相干振幅 $A_a = \sqrt{N_a/N}\,\eta_a$，相干强度 $= N_a^2|\eta_a|^2$，漫射强度 $= N_a(1-|\eta_a|^2)$；
  竞争写法 S1：相干强度 $\propto |\eta_a|$（即 $|c_a|^2 = (N_a/N)|\eta_a|$ 的写法）：相干 $= N_a^2|\eta_a|$、漫射 $= N_a(1-|\eta_a|)$；
  竞争写法 S2：无相干因子（相干 $= N_a^2$、漫射 $= 0$）；
  竞争写法 S3：振幅无相干因子（相干 $= N_a^2|\eta_a|^2$，但漫射不被压制，$= N_a$）。S3 只作对照行。
本脚本不改动基准文件。

模型
----
$N = N_1 + N_2$ 个元相子分入两块；第 $a$ 块内每个元相子的内相位取 $+\delta_a$ 或 $-\delta_a$，
概率各为 $(1+q_a)/2$ 与 $(1-q_a)/2$，$\delta_a = \pi/2$，于是块内相干因子 $\eta_a = i\,q_a$ 精确可算（$|\eta_a| = q_a$）。
块的共同相位 $\Phi_a$ 可调，干涉仪相位 $\varphi = \Phi_1 - \Phi_2$。物理场取每粒子一份单位振幅并按构型直接求和：
$E(\varphi) = \sum_i e^{i\delta_i}\,e^{i\Phi_{a(i)}}$；全部 $2^N$ 个构型按其概率精确求和（不作近似）。

判据（先立判据，再实现；预期写错时修正判据并记录教训）
----
AMP-1 相干强度：$|\langle E_a\rangle|^2 = N_a^2 q_a^2$（扫描 $q_a = 0$ 到 $1$，逐点）。
AMP-2 漫射强度：$\langle|E_a|^2\rangle - |\langle E_a\rangle|^2 = N_a(1-q_a^2)$；
   对照 S1 的 $N_a(1-q_a)$ 应给系统偏差。
AMP-3 和式守恒：相干强度加漫射强度 $= \langle|E_a|^2\rangle$（精确）。
AMP-4 总强度的幂次：$\langle|E_a|^2\rangle = N_a + N_a(N_a-1)q_a^2$（二次律）；对照 S1 的线性律应不符。
AMP-5 干涉项幅值：$\varphi$ 的 $\cos$ 分量幅值 $= 2N_1N_2 q_1q_2$（每块一个因子）；对照 $\sqrt{q_1q_2}$ 写法应不符。
AMP-6 完全退相干极限：$q_a = 0$ 时 $\langle|E_a|^2\rangle = N_a$，份额 $N_a/N$ 不变（人数不动，齐整度归零）。

运行
----
python 验证脚本/amplitude_composition.py
确定性（穷举系综，无随机数），运行 < 60 秒。图存 D:/software/TianXingFC/images/amplitude_composition.png
"""
import time
from itertools import product

import numpy as np

T0 = time.time()
CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print(f"  [{tag}] {'PASS' if ok else 'FAIL'}  {detail}")


DELTA = np.pi / 2.0        # 内相位步长（±delta）
PHI_GRID = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)


def mu_of(q):
    """两值模型的块内相干因子：eta = cos(delta) + i q sin(delta)。"""
    return np.cos(DELTA) + 1j * q * np.sin(DELTA)


def ensemble(N1, N2, q1, q2, phi_grid):
    """穷举全部构型，返回 <E(phi)>、<|E(phi)|^2>、<|E_a|^2> 与归一化权重。"""
    N = N1 + N2
    p1 = (1.0 + q1) / 2.0
    p2 = (1.0 + q2) / 2.0
    E_sum = np.zeros(len(phi_grid), dtype=complex)
    E2_sum = np.zeros(len(phi_grid))
    Ea2_1 = 0.0
    Ea2_2 = 0.0
    for bits in product((0, 1), repeat=N):
        w = 1.0
        ph = []
        for idx, b in enumerate(bits):
            if idx < N1:
                w *= p1 if b == 0 else (1.0 - p1)
                ph.append(+DELTA if b == 0 else -DELTA)
            else:
                w *= p2 if b == 0 else (1.0 - p2)
                ph.append(+DELTA if b == 0 else -DELTA)
        ph = np.array(ph)
        e1 = np.exp(1j * ph[:N1]).sum()
        e2 = np.exp(1j * ph[N1:]).sum()
        E = e1 + e2 * np.exp(1j * phi_grid)
        E_sum += w * E
        E2_sum += w * np.abs(E) ** 2
        Ea2_1 += w * abs(e1) ** 2
        Ea2_2 += w * abs(e2) ** 2
    return E_sum, E2_sum, Ea2_1, Ea2_2


def cos_amp(vals, grid):
    """提取对 cos(phi) 的幅值（网格含整数个周期，离散正交）。"""
    c = 2.0 * np.sum(vals * np.cos(grid)) / len(grid)
    s = 2.0 * np.sum(vals * np.sin(grid)) / len(grid)
    return float(np.hypot(c, s))


print("=" * 74)
print("[振幅合成轮] 宏观振幅 = 份额开方乘块内相干度")
print("=" * 74)

N1, N2 = 6, 6
QS = [round(0.1 * k, 2) for k in range(11)]
print(f"\nA 部分：单块分解扫描（N_1 = {N1}，delta = pi/2，q = 0 到 1 共 {len(QS)} 档）")
print("   q      相干(实测)   相干(主张)   漫射(实测)   漫射(q^2 主张)  漫射(S1 线性)   总强度(实测)")
rows = []
for q in QS:
    # 单块情形：N2 = 0，用 q2 = 0 且把块 2 关掉（N_2 = 0 时 e2 = 0）
    Es, E2s, Ea2_1, _ = ensemble(N1, 0, q, 0.0, PHI_GRID)
    coh = abs(Es[0]) ** 2
    diff = Ea2_1 - coh
    pred_coh = N1 ** 2 * q ** 2
    pred_diff = N1 * (1.0 - q ** 2)
    pred_diff_S1 = N1 * (1.0 - q)
    rows.append((q, coh, pred_coh, diff, pred_diff, pred_diff_S1, Ea2_1))
    print(f"  {q:4.2f}   {coh:9.3f}   {pred_coh:9.3f}   {diff:9.3f}   {pred_diff:11.3f}   "
          f"{pred_diff_S1:11.3f}   {Ea2_1:9.3f}")

qA = np.array([r[0] for r in rows])
cohA = np.array([r[1] for r in rows])
diffA = np.array([r[3] for r in rows])
totA = np.array([r[6] for r in rows])
check("AMP-1", np.allclose(cohA, N1 ** 2 * qA ** 2, atol=1e-9),
      f"相干强度 = N^2 q^2（最大偏差 {np.max(np.abs(cohA - N1 ** 2 * qA ** 2)):.2e}）")
d2 = np.max(np.abs(diffA - N1 * (1.0 - qA ** 2)))
d1 = np.max(np.abs(diffA - N1 * (1.0 - qA)))
print(f"  漫射强度与两种写法的最大偏差：二次律 {d2:.2e}；S1 线性律 {d1:.3f}")
check("AMP-2", d2 < 1e-9 and d1 > 0.3, f"漫射 = N(1-q^2)（偏差 {d2:.2e}）；S1 写法偏差 {d1:.3f} 被否")
check("AMP-3", np.allclose(cohA + diffA, totA, atol=1e-9),
      f"相干加漫射 = 总强度（最大偏差 {np.max(np.abs(cohA + diffA - totA)):.2e}）")
check("AMP-4", np.allclose(totA, N1 + N1 * (N1 - 1) * qA ** 2, atol=1e-9),
      f"总强度 = N + N(N-1)q^2（二次律，最大偏差 {np.max(np.abs(totA - (N1 + N1 * (N1 - 1) * qA ** 2))):.2e}）")

print(f"\nB 部分：两块的干涉项（N_1 = N_2 = {N1}，扫描 q_1 = q_2 = q）")
print("   q     干涉幅值(实测)   2N1N2 q^2(主张)   2N1N2 q(S1 写法)")
inter_meas = []
for q in QS:
    Es, E2s, _, _ = ensemble(N1, N2, q, q, PHI_GRID)
    amp = cos_amp(E2s, PHI_GRID)
    inter_meas.append(amp)
    print(f"  {q:4.2f}   {amp:12.3f}   {2 * N1 * N2 * q ** 2:13.3f}   {2 * N1 * N2 * q:13.3f}")
inter_meas = np.array(inter_meas)
e2_ = np.max(np.abs(inter_meas - 2 * N1 * N2 * qA ** 2))
e1_ = np.max(np.abs(inter_meas - 2 * N1 * N2 * qA))
check("AMP-5", e2_ < 1e-8 and e1_ > 1.0,
      f"干涉幅值 = 2N1N2 q^2（偏差 {e2_:.2e}）；sqrt(q) 类写法偏差 {e1_:.3f} 被否")

print("\nC 部分：完全退相干极限与份额不变")
Es, E2s, Ea2_1, Ea2_2 = ensemble(N1, N2, 0.0, 0.0, PHI_GRID)
share1 = N1 / (N1 + N2)
check("AMP-6", abs(Ea2_1 - N1) < 1e-9 and abs(Ea2_2 - N2) < 1e-9 and abs(share1 - 0.5) < 1e-12,
      f"q = 0 时每块强度 = 人数（{Ea2_1:.3f} 与 {Ea2_2:.3f}），份额仍为 {share1:.3f}")

print("\nD 部分：单块 vs 相干参考块的可见度（振幅级每块一个因子）")
print("   q1    可见度(实测)   q1 主张   sqrt(q1) 写法")
vis = []
for q1 in QS:
    Es, E2s, _, _ = ensemble(N1, N2, q1, 1.0, PHI_GRID)
    amp = cos_amp(E2s, PHI_GRID)
    vis.append(amp)
    print(f"  {q1:4.2f}   {amp / (2 * N1 * N2):12.4f}   {q1:8.4f}   {np.sqrt(q1):13.4f}")
vis = np.array(vis) / (2 * N1 * N2)
check("AMP-5b", np.allclose(vis, qA, atol=1e-9),
      f"一块退相干一块相干时干涉幅值 = 2N1N2 q（每块一个因子，最大偏差 "
      f"{np.max(np.abs(vis - qA)):.2e}）")

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
    ax.plot(qA, cohA, 'o', color="#1f77b4", label="相干强度（实测）")
    ax.plot(qA, N1 ** 2 * qA ** 2, '-', color="#1f77b4", lw=1.0, label=r"$N^2q^2$")
    ax.plot(qA, diffA, 's', color="#d95f02", label="漫射强度（实测）")
    ax.plot(qA, N1 * (1 - qA ** 2), '-', color="#d95f02", lw=1.0, label=r"$N(1-q^2)$")
    ax.plot(qA, N1 * (1 - qA), ':', color="#7570b3", lw=1.4, label=r"竞争写法 $N(1-q)$")
    ax.set_xlabel("块内相干度 $|\\eta| = q$")
    ax.set_ylabel("强度")
    ax.set_title(f"(a) 相干与漫射分解（N = {N1}）", fontsize=10.5)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axs[1]
    ax.plot(qA, inter_meas, 'o', color="#1b1b1b", label="干涉项幅值（实测）")
    ax.plot(qA, 2 * N1 * N2 * qA ** 2, '-', color="#1f77b4", lw=1.0, label=r"$2N_1N_2q^2$")
    ax.plot(qA, 2 * N1 * N2 * qA, '--', color="#7570b3", lw=1.0, label=r"$2N_1N_2q$")
    ax.plot(qA, 2 * N1 * N2 * np.sqrt(qA), ':', color="#d95f02", lw=1.2, label=r"$2N_1N_2\sqrt{q}$")
    ax.set_xlabel("块内相干度 $|\\eta| = q$")
    ax.set_ylabel("$\\cos\\varphi$ 分量幅值")
    ax.set_title("(b) 干涉项：每块一个因子（两块各一次）", fontsize=10.5)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axs[2]
    for q, col in [(1.0, "#1b1b1b"), (0.5, "#1f77b4"), (0.0, "#d95f02")]:
        Es, E2s, _, _ = ensemble(N1, N2, q, q, PHI_GRID)
        ax.plot(PHI_GRID, E2s, '-', color=col, lw=1.4, label=f"q = {q}")
    ax.set_xlabel(r"干涉仪相位 $\varphi$")
    ax.set_ylabel(r"系综平均强度 $\langle|E|^2\rangle$")
    ax.set_title("(c) 条纹：退相干压低幅度、抬高支底", fontsize=10.5)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("D:/software/TianXingFC/images/amplitude_composition.png", dpi=150)
    print("\n图已保存：images/amplitude_composition.png")
except ImportError:
    print("\n跳过绘图：matplotlib 不可用")

print(f"\n[振幅合成轮] 完成；总运行 {time.time() - T0:.1f} s。")
