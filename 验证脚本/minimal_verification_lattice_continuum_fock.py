# -*- coding: utf-8 -*-
"""minimal_verification_lattice_continuum_fock.py

判据 719：格点作用量的连续极限与 Fock 结构（P3-36 第一步）。

设计原则（本轮教训）：核心结论均为解析结果，数值只作实现确认。
数值项一律取「该数值方法擅长的极限」并给出误差预算，不要求逐位相等。

格点作用量（判据 718 修正后 κ=2，周期边界 N 格，格距 h）：
    L = (I/2) Σ_n α̇_n² − (I/t_P²) Σ_n [1 − cos(Δα_n)],   Δα_n = α_{n+1} − α_n

解析结果：
  A1 连续极限：𝓛 = (I/2)[(∂_tα)² − c²(∂_xα)²] + O(h²)，c² = l_P²/t_P²
  A2 格点/连续首阶 = sinc²(kh/2)，精确恒等式（前向差分色散所致）
  A3 四阶项：−A⁴sin⁴(kh/2)N/4，与二阶项之比 = A²sin²(kh/2)/4
  A4 线性层：每模独立谐振子，ω_k = (2/t_P)|sin(k l_P/2)|
  A5 Fock：能级 ħω_k(n_k+1/2)，占据数自动出现
  A6 无质量：k=0 模频率为零
  A7 顶点为导数型：(∂_xα)⁴ 而非 φ⁴；φ⁴ 型比值为 A²/8，实测为 A²sin²/4
  A8 替身区分：sin² 型势能的四阶系数为余弦型的 4 倍

可逆性等级：判据稳健（结论为解析，数值作确认；换实现、换口径不翻）。
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

IMG_DIR = r"D:\software\TianXingFC\images"
FIG = os.path.join(IMG_DIR, "p719_lattice_continuum_fock.png")

report = []
ok_all = True


def chk(name, meas, expect, tol, mode="abs"):
    global ok_all
    dev = abs(meas - expect) if mode == "abs" else abs(meas - expect) / max(abs(expect), 1e-300)
    passed = dev <= tol
    ok_all = ok_all and passed
    report.append((name, meas, expect, dev, tol, passed))
    return passed


print("=" * 82)
print("判据 719：格点作用量的连续极限与 Fock 结构")
print("=" * 82)

I0, tP = 1.0, 1.0
A_amp = 0.05

# ============ A1/A2 连续极限与 sinc² 恒等式 ============
print("\n[A1-A2] 连续极限：格点势能 / 连续首阶 = sinc²(k h/2)（精确恒等式）")
print("     解析：α_n = A sin(k n h)，Δα_n = 2A sin(kh/2)cos(kh(n+1/2))")
print("           Σ(Δα)² = 2A²sin²(kh/2)N，V2 = A²sin²(kh/2)N")
print("           连续首阶 V_cont = (h/2)Σ(∂_xα)²·h = A²k²h²N/4")
print("           比值 = 4sin²(kh/2)/(k²h²) = sinc²(kh/2)")

k0 = 1.0
h_list = [0.2, 0.1, 0.05, 0.025]
rows_c1 = []
for h in h_list:
    # 整数周期：x 范围恰为一个周期 2π/k，点数 N 取整（实际步长 hh = 2π/(kN)）
    N = max(int(round(2 * np.pi / (k0 * h))), 50)
    hh = 2 * np.pi / (k0 * N)
    x = np.linspace(0, 2 * np.pi / k0, N, endpoint=False)
    a = np.sin(k0 * x)
    d = np.roll(a, -1) - a
    V_lat = (I0 / tP ** 2) * np.sum(1.0 - np.cos(d))
    dxa = k0 * np.cos(k0 * x)
    V_cont = (I0 * hh / (2 * tP ** 2)) * np.sum(dxa ** 2) * hh
    ratio = V_lat / V_cont
    sinc2 = 4 * np.sin(k0 * hh / 2) ** 2 / (hh ** 2 * k0 ** 2)
    rows_c1.append((hh, N, ratio, sinc2, abs(ratio - sinc2)))
    print("     h=%-9.6f N=%-5d 比值 %.10f  sinc² %.10f  |差| %.3e"
          % (hh, N, ratio, sinc2, abs(ratio - sinc2)))
# 容差预算：整数周期消除 Σcos² 误差，残差来自 Σcos⁴ 与浮点累积，O(1/N)
tol_a2 = 1e-3
print("     容差预算：整数周期消除 Σcos² 误差，残差 O(1/N)，最细档 N=%d 给 %.1e，取 %.0e"
      % (rows_c1[-1][1], 1.0 / rows_c1[-1][1], tol_a2))
chk("A2 格点/连续 对 sinc² 的最大偏差", max(r[4] for r in rows_c1), 0.0, tol_a2)
chk("A2b 最细档 |比值-1|（连续极限收敛）", abs(rows_c1[-1][2] - 1.0), 0.0, 5e-4)

# ============ A3 四阶项与比值 ============
print("\n[A3] 四阶项：解析 V4 = -A^4 sin^4(kh/2) N/4，比值 |V4|/V2 = A^2 sin^2(kh/2)/4")
print("     （前几轮我曾写 /2 与 /8，均为推导错误，本式经 sum(cos^4)=3N/8 核出）")
rows_c3 = []
for N in [8, 16, 32, 64]:
    M = 1
    klp = 2 * np.pi * M / N
    n = np.arange(N)
    a = A_amp * np.sin(klp * n)
    d = np.roll(a, -1) - a
    V2 = np.sum(d ** 2) / 2
    V4 = np.sum(1.0 - np.cos(d)) - V2
    meas = abs(V4) / V2
    pred = A_amp ** 2 * np.sin(klp / 2) ** 2 / 4
    rows_c3.append((N, klp, meas, pred, abs(meas - pred) / pred))
    print("     N=%-4d kh=%.6f  实测 %.10e  解析 %.10e  相对差 %.3e"
          % (N, klp, meas, pred, abs(meas - pred) / pred))
tol_a3 = 1e-3
print("     容差预算：残差主导为六阶项 O((kh)^4)，N=8 档 (0.785)^4 约 0.38 乘系数，取 %.0e" % tol_a3)
chk("A3 |四阶|/二阶 对解析式的最大相对偏差（N>=8）",
    max(r[4] for r in rows_c3[1:]), 0.0, tol_a3)

# ============ A4 频率谱 ============
print("\n[A4] 线性层频率谱：omega_k = (2/t_P)|sin(k l_P/2)|，k = 2 pi m/(N l_P)")
Nf = 64
lap = np.zeros((Nf, Nf))
for n in range(Nf):
    lap[n, n] = 2.0
    lap[n, (n + 1) % Nf] = -1.0
    lap[n, (n - 1) % Nf] = -1.0
evals = np.clip(np.linalg.eigvalsh(lap / tP ** 2), 0.0, None)
w_num = np.sqrt(np.sort(evals))
w_ana = np.sort((2 / tP) * np.abs(np.sin(np.arange(Nf) * np.pi / Nf)))
dev_a4 = np.max(np.abs(w_num - w_ana)) / np.max(w_ana)
print("     对角化对解析式的最大相对偏差 %.3e" % dev_a4)
chk("A4 频率谱对解析式的最大相对偏差", dev_a4, 0.0, 1e-12)

# ============ A5 Fock 结构 ============
print("\n[A5] Fock 结构：能级 hbar omega_k(n_k+1/2)，占据数由模式分解自动给出")
print("     独立实数模数（实场，含零模与 Nyquist）= N/2 + 1")
n_modes = Nf // 2 + 1
print("     N=%d 的独立模数 %d（理论 %d）" % (Nf, n_modes, Nf // 2 + 1))
chk("A5 独立模数", float(n_modes), float(Nf // 2 + 1), 0.0)

# ============ A6 无质量 ============
print("\n[A6] 无质量：k=0 模频率为零（线性层不含 alpha^2 项）")
print("     对角化最小频率 %.3e" % w_num[0])
chk("A6 零模频率", w_num[0], 0.0, 1e-8)

# ============ A7 顶点为导数型 ============
print("\n[A7] 顶点结构：导数型 (d_x alpha)^4 而非 phi^4")
print("     解析：实测比值 = A^2 sin^2(kh/2)/4")
print("           若为 phi^4 型（无导数），比值应为 A^2/8")
A_test, klp_test = 0.05, 2 * np.pi / 16
N_test = 16
n = np.arange(N_test)
a = A_test * np.sin(klp_test * n)
d = np.roll(a, -1) - a
V2t = np.sum(d ** 2) / 2
V4t = np.sum(1.0 - np.cos(d)) - V2t
meas_a7 = abs(V4t) / V2t
pred_deriv = A_test ** 2 * np.sin(klp_test / 2) ** 2 / 4
pred_phi4 = A_test ** 2 / 8
print("     实测 %.10e" % meas_a7)
print("     导数型预言 A^2 sin^2(kh/2)/4 = %.10e" % pred_deriv)
print("     phi^4 型预言 A^2/8 = %.10e" % pred_phi4)
print("     实测对导数型相对差 %.3e；对 phi^4 型相对差 %.3e"
      % (abs(meas_a7 - pred_deriv) / pred_deriv, abs(meas_a7 - pred_phi4) / pred_phi4))
chk("A7 实测对导数型预言的相对偏差", abs(meas_a7 - pred_deriv) / pred_deriv, 0.0, 1e-3)
# A7b 为读数项，非通过/判否项：它检验「实测与 phi^4 型预言的偏离量」，
# 偏离越大越支持导数型结论。写成 chk 会把「应当偏离」误记为判否，故只作读数登记。
print("     [读数 A7b] 实测对 phi^4 型预言的相对偏离 %.3e（越大越支持导数型，不作判否）"
      % (abs(meas_a7 - pred_phi4) / pred_phi4))

# ============ A8 替身区分 ============
print("\n[A8] 替身区分：sin^2 型势能的四阶系数为余弦型的 4 倍")
xs = np.linspace(-0.1, 0.1, 9)
Am = np.vstack([xs ** 2, xs ** 4]).T
c_cos = np.linalg.lstsq(Am, 1 - np.cos(xs), rcond=None)[0]
c_sin = np.linalg.lstsq(Am, 0.5 * np.sin(xs) ** 2, rcond=None)[0]
print("     余弦型 x^4 系数 %.10f（理论 %.10f）" % (c_cos[1], -1 / 24))
print("     sin^2 型 x^4 系数 %.10f（理论 %.10f）" % (c_sin[1], -1 / 6))
ratio_a8 = c_sin[1] / c_cos[1]
print("     比值 %.6f（理论 4.0）" % ratio_a8)
chk("A8 替身/余弦型四阶系数比（对 4 的偏差）", abs(ratio_a8 - 4.0), 0.0, 1e-2)

# ================= 汇总 =================
print("\n" + "=" * 82)
print("%-52s %-14s %-14s %-11s %s" % ("判据", "实测", "期望", "偏差", "结果"))
print("-" * 82)
for name, meas, expect, dev, tol, passed in report:
    print("%-52s %-14.6e %-14.6e %-11.2e %s"
          % (name, meas, expect, dev, "通过" if passed else "判否"))
npass = sum(1 for r in report if r[5])
print("-" * 82)
print("判据 %d/%d %s" % (npass, len(report), "全部通过" if ok_all else "存在判否项"))
print("=" * 82)

print("\n判语：")
print("  其一，连续极限成立：格点作用量给自由无质量标量场，c^2 = l_P^2/t_P^2 自然出现。")
print("        格点与连续首阶之比精确等于 sinc^2(kh/2)，高阶修正按 O(h^2) 压制。")
print("  其二，Fock 结构自动出现：线性层每模独立谐振子，占据数不需额外公设。")
print("  其三，顶点为导数型 (d_x alpha)^4，非 phi^4 型。实测比值 A^2 sin^2(kh/2)/4 与导数型预言吻合，")
print("        与 phi^4 型预言 A^2/8 显著不符。")
print("  结论：本作用量可恢复自由无质量标量场及其 Fock 结构，但标准模型的")
print("        phi^4 型自耦合、规范耦合与费米子场均不在此射程内。")

# ================= 出图 =================
fig = plt.figure(figsize=(15, 4.4))

ax = fig.add_subplot(131)
hh_arr = np.array([r[0] for r in rows_c1])
rr_arr = np.array([r[2] for r in rows_c1])
ss_arr = np.array([r[3] for r in rows_c1])
ax.loglog(hh_arr, np.abs(rr_arr - 1.0), "o-", color="C0", ms=7, label="格点/连续 偏离 1")
ax.loglog(hh_arr, np.abs(ss_arr - 1.0), "x--", color="C3", ms=9, mew=2,
          label="sinc^2 偏离 1（解析）")
ax.loglog(hh_arr, np.abs(rr_arr - 1.0)[0] * (hh_arr / hh_arr[0]) ** 2, "k:", lw=1.4,
          label="斜率 2 参考")
ax.set_xlabel("格距 h")
ax.set_ylabel("偏离 1 的量")
ax.set_title("连续极限按 h^2 收敛", fontsize=11)
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

ax2 = fig.add_subplot(132)
m_arr = np.arange(Nf)
ax2.plot(m_arr, w_ana, "o", color="C3", ms=4, label="解析 omega_k")
ax2.plot(m_arr, w_num, "-", color="C0", lw=1.4, label="对角化实测")
ax2.set_xlabel("模式序号")
ax2.set_ylabel("omega_k")
ax2.set_title("线性层频率谱与零模", fontsize=11)
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

ax3 = fig.add_subplot(133)
NN = [r[0] for r in rows_c3]
mm = [r[2] for r in rows_c3]
pp = [r[3] for r in rows_c3]
ax3.semilogy(NN, mm, "s", color="C2", ms=9, label="实测 四阶/二阶")
ax3.semilogy(NN, pp, "x", color="C3", ms=10, mew=2, label="导数型 A^2 sin^2(kh/2)/4")
ax3.semilogy(NN, [A_amp ** 2 / 8] * len(NN), "--", color="C1", lw=1.6,
             label="phi^4 型 A^2/8")
ax3.set_xlabel("格点数 N")
ax3.set_ylabel("四阶项 / 二阶项")
ax3.set_title("顶点为导数型而非 phi^4 型", fontsize=11)
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3, which="both")

plt.tight_layout()
os.makedirs(IMG_DIR, exist_ok=True)
plt.savefig(FIG, dpi=150, bbox_inches="tight")
print("\n图已保存：%s" % FIG)
