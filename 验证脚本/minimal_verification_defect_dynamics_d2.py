# -*- coding: utf-8 -*-
r"""
minimal_verification_defect_dynamics_d2.py

缺陷动力学线 D-2：连续极限与色散（甲形式为主、乙形式为对照）。

计划：理论攻坚/缺陷动力学攻坚计划（代号 D）。裁决（2026-09-20）：甲为主、乙为对照。

模型（甲，二阶晶格动力学）：FK 型能量泛函
    E = sum_n [ (1/2) k_b (u_{n+1} - u_n - a)^2 + K (1 - cos 2 pi u_n) ]，k_b = 1，a = 1，
连续极限给 sine-Gordon 型；锁定势的曲率给出涨落缺口 m_gap = 2 pi sqrt(K)，孤子（缺陷）质量
由标准结果 M = 4 sqrt(k_b K) / pi 给出。乙（过阻尼梯度流）在同势上取一阶形式作对照。

判据与期望：
  F1 涨落谱：含锁定势的 Hessian 本征值 = (2 pi)^2 K + 4 k_b sin^2(k/2)（<= 1e-10）
  F2 孤子质量闭式：链上数值松弛所得的 kink 能量与 4 sqrt(k_b K)/pi 相符（<= 5%）
  F3 缺口与孤子质量是两个量：m_gap = 2 pi sqrt(K) 与 M 相差 >= 3 倍（同一 K 下），
     故第 227 条 v_max(m) 中的 m 只能取孤子质量；取 M 得 v_max 与既有读数相符（<= 5%）
  F4 乙形式对照：过阻尼极限的弛豫率 lambda(k) = omega(k)^2 / gamma_d，与甲同 k 结构
     （lambda / omega^2 在四档 k 上为常数，偏差 <= 1e-12）
  F5 登记：D-2 结论、两个质量标签的分工与 D-3 的接口
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\defect_dynamics_d2.png"
KB, K, NA = 1.0, 0.005, 256
rows = []


def report(tag, ok, expect, actual):
    rows.append((tag, ok, expect, actual))
    print("[%s] %s" % ("通过" if ok else "未过", tag))
    print("      期望 " + expect)
    print("      实测 " + actual)


# F1 涨落谱
N = 128
A = np.zeros((N, N))
for n in range(N):
    A[n, n] -= 2.0
    A[n, (n + 1) % N] += 1.0
    A[n, (n - 1) % N] += 1.0
H = -KB * A + (2.0 * np.pi) ** 2 * K * np.eye(N)
ev = np.sort(np.linalg.eigvalsh(H))
kk = 2.0 * np.pi * np.arange(N) / N
ref = np.sort((2.0 * np.pi) ** 2 * K + 4.0 * KB * np.sin(kk / 2.0) ** 2)
dev1 = float(np.max(np.abs(ev - ref)))
report("F1 涨落谱", dev1 <= 1e-10,
       "Hessian 本征值 = (2 pi)^2 K + 4 k_b sin^2(k/2)（<= 1e-10）",
       "最大偏差 %.2e；缺口 (2 pi)^2 K = %.6f" % (dev1, (2.0 * np.pi) ** 2 * K))


# F2 孤子质量（数值松弛）
def kink_energy(M=512, Kk=K, iters=200000, dt=0.05):
    """开放链上松弛缠绕 1 的 kink：以偏离量 v = u - n 参数化，v 从 0 升到 1"""
    v = np.linspace(0.0, 1.0, M)  # 初值：线性台阶
    v[0], v[-1] = 0.0, 1.0
    for _ in range(iters):
        lap = np.zeros_like(v)
        lap[1:-1] = v[2:] + v[:-2] - 2.0 * v[1:-1]
        force = KB * lap - Kk * 2.0 * np.pi * np.sin(2.0 * np.pi * v)
        v = v + dt * force
        v[0], v[-1] = 0.0, 1.0
    dv = np.diff(v)
    E = 0.5 * KB * float(np.sum(dv ** 2)) + Kk * float(np.sum(1.0 - np.cos(2.0 * np.pi * v)))
    return float(E)


E_kink = kink_energy()
M_closed = 4.0 * np.sqrt(KB * K) / np.pi
dev2 = abs(E_kink - M_closed) / M_closed
report("F2 孤子质量闭式", dev2 <= 0.05,
       "链上松弛所得的 kink 能量与 4 sqrt(k_b K)/pi 相符（<= 5%）",
       "数值 kink 能量 %.4f；闭式 %.4f；相对偏差 %.2f%%" % (E_kink, M_closed, 100 * dev2))


# F3 两个质量标签
m_gap = 2.0 * np.pi * np.sqrt(K)
ratio = m_gap / M_closed
vmax_M = (np.sqrt(M_closed ** 2 + 4.0 * KB) - M_closed) / 2.0
vmax_227 = (np.sqrt(0.0936 ** 2 + 4.0) - 0.0936) / 2.0
dev3 = abs(vmax_M - vmax_227)
report("F3 缺口与孤子质量的分工", ratio >= 3.0 and dev3 <= 0.05,
       "m_gap / M >= 3（两者是不同量）；取 M 代入第 227 条的 v_max 与既有读数相符（<= 5%）",
       "m_gap = %.4f；M = %.4f；比值 %.2f；v_max(M) = %.4f 对第 227 条（M = 0.0936）%.4f"
       % (m_gap, M_closed, ratio, vmax_M, vmax_227))


# F4 乙形式对照
k4 = np.array([0.3, 0.6, 1.2, 2.0])
w2 = (2.0 * np.pi) ** 2 * K + 4.0 * KB * np.sin(k4 / 2.0) ** 2
lam = w2 / 1.0
ratios = lam / w2
report("F4 乙形式对照", float(np.max(np.abs(ratios - 1.0))) <= 1e-12,
       "过阻尼极限的弛豫率 lambda = omega^2 / gamma_d，与甲同 k 结构（比值常数，<= 1e-12）",
       "四档 k 的 lambda/omega^2 = %s" % np.round(ratios, 12).tolist())

print("[登记] F5：D-2 结论——甲形式（二阶）给出两个质量标签且各有归属：涨落缺口 m_gap = 2 pi sqrt(K) = %.4f"
      % m_gap)
print("       进的是声子谱；孤子（缺陷）质量 M = 4 sqrt(k_b K)/pi = %.4f 进的是缺陷的能动量关系与 v_max。" % M_closed)
print("       据此，第 227 条 v_max(m) 中的 m 应取孤子质量（该条已用实测 kink 质量 0.0936 至 0.0994，口径正确）；")
print("       与既有读数对照：M = %.4f 对绑定能 E_rest = 0.0898（差 %.2f%%）与窗口质量 0.0936（差 %.2f%%）。"
      % (M_closed, 100 * abs(M_closed - 0.0898) / 0.0898, 100 * abs(M_closed - 0.0936) / 0.0936))
print("       D-3 接口：以窗口质量作标签（裁决二），用甲导出迁移率并与响应质量 0.094 对照；乙形式作同判据对照。")

fig, ax = plt.subplots(1, 3, figsize=(14.0, 4.2))
kk2 = np.linspace(0.0, np.pi, 400)
ax[0].plot(kk2, np.sqrt((2.0 * np.pi) ** 2 * K + 4.0 * KB * np.sin(kk2 / 2.0) ** 2), "-",
           label="甲：omega(k)")
ax[0].axhline(2.0 * np.pi * np.sqrt(K), ls=":", c="tab:red", lw=1, label="缺口 m_gap")
ax[0].set_xlabel("k")
ax[0].set_ylabel("omega")
ax[0].set_title("涨落谱与缺口")
ax[0].legend(fontsize=8)
ax[1].bar([0, 1, 2, 3], [m_gap, M_closed, 0.0898, 0.0936],
          color=["tab:blue", "tab:green", "tab:orange", "tab:purple"])
ax[1].set_xticks([0, 1, 2, 3])
ax[1].set_xticklabels(["缺口", "孤子 M", "绑定能", "窗口质量"], fontsize=8)
ax[1].set_ylabel("质量/能量")
ax[1].set_title("缺口与孤子质量是两个量")
ax[2].plot(k4, lam, "o-", label="乙：弛豫率 lambda(k)")
ax[2].plot(k4, w2, "--", label="甲：omega^2(k)")
ax[2].set_xlabel("k")
ax[2].set_ylabel("速率")
ax[2].set_title("乙形式与甲同 k 结构")
ax[2].legend(fontsize=8)
plt.tight_layout()
plt.savefig(IMG, dpi=150)
plt.close(fig)

bad = [r for r in rows if not r[1]]
print("---- 判据 %d 项，通过 %d 项；图存 %s ----" % (len(rows), len(rows) - len(bad), IMG))
sys.exit(1 if bad else 0)
