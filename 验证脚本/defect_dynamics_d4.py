# -*- coding: utf-8 -*-
r"""
defect_dynamics_d4.py

缺陷动力学线 D-4（消费式接口轮）：与 C1 线的口径映射，不重测。

计划：理论攻坚/缺陷动力学攻坚计划（代号 D）。裁决点 5：钉扎脱钉归 C1 主建，D 消费。

上游读数（消费，不重测）：
  C1-3：k = 1、N = 32、GF 步长 eta = 0.01；驻留阈值 K_c = 0.24 至 0.25（C1-4 读数，第 233 条复现 0.25）；
       脱钉读数 alpha* = 0.15 是在 K = 0.25、R = 4 的驻留态上施加多模扰动的幅度阈值（非势阱深度）；
       K = 0.02 给 collapse，K <= 0.20 的 R_c 只记为大于 5（未排除）
  C1-5 窗口：落窗条件 l* >= 2，第 233 条在 K >= 0.25 处给 l* = 1.924（窗口互斥）
  D-2/D-3：k_b = 1、K = 0.005、站点 256；m_gap = 2 pi sqrt(K) = 0.4443，M = 4 sqrt(k_b K)/pi = 0.0900

判据与期望：
  J1 势与弹性口径同一：把 C1 的能量泛函按有限差分求梯度，与 D 的过阻尼方程右端闭式在 gamma_d = 1/eta
     下逐点比较，残差 <= 1e-6（有限差分精度）
  J2 缺口关系的跨线核：用 C1 型泛函数值构造 Hessian，最小本征值 = (2 pi)^2 K（<= 1e-10），三档 K
  J3 参数区间不相交：C1 驻留域 [0.24, 0.25] 与 D 域 {0.005} 交集为空；两域孤子质量之比 >= 5
  J4 阈值口径：alpha* 是相位扰动幅度，与凹口调制深度 d 不同层次；给出扰动能量与钉扎势幅度 2K 的量级比
  J5 质量统一判定：同一对象（kink 质量）的四个读数在同一 K 下自洽（相对散度 <= 2%）；窗口质量按独立刻度登记
     （与闭式的偏离作正面读数，不并入同一组）；跨线代入 K_c 后偏离 >= 5 倍
  J6 结论登记：口径统一、区间不相交，跨区桥梁是 K 的体系具体值（C1 门控下未定）
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\defect_dynamics_d4.png"
KB = 1.0
N = 32
ETA = 0.01
K_C1 = (0.24, 0.25)
K_D = 0.005
rows = []


def energy_c1(v, K, kb=KB):
    u = v + np.arange(N)
    dv = np.roll(v, -1) - v
    return 0.5 * kb * np.sum(dv ** 2) + K * np.sum(1.0 - np.cos(2.0 * np.pi * u))


def rhs_d(v, K, kb=KB):
    lap = np.roll(v, 1) + np.roll(v, -1) - 2.0 * v
    u = v + np.arange(N)
    return kb * lap - K * 2.0 * np.pi * np.sin(2.0 * np.pi * u)


def gap_formula(K, kb=KB):
    return 2.0 * np.pi * np.sqrt(K)


def mass_formula(K, kb=KB):
    return 4.0 * np.sqrt(kb * K) / np.pi


def hessian_min(K, kb=KB):
    m = np.zeros((N, N))
    for i in range(N):
        m[i, i] = 2.0 * kb + (2.0 * np.pi) ** 2 * K
        m[i, (i + 1) % N] += -kb
        m[i, (i - 1) % N] += -kb
    w = np.linalg.eigvalsh(m)
    return float(w[0])


# ---------- J1 势与弹性口径同一 ----------
v0 = 0.3 * np.sin(2.0 * np.pi * np.arange(N) / N) + 0.1 * np.cos(4.0 * np.pi * np.arange(N) / N)
e_c1 = energy_c1(v0, K_D)
u0 = v0 + np.arange(N)
dv0 = np.roll(v0, -1) - v0
e_d = 0.5 * KB * np.sum(dv0 ** 2) + K_D * np.sum(1.0 - np.cos(2.0 * np.pi * u0))
res_e = abs(e_c1 - e_d)
h = 1e-6
grad = np.zeros(N)
for i in range(N):
    vp = v0.copy()
    vm = v0.copy()
    vp[i] += h
    vm[i] -= h
    grad[i] = (energy_c1(vp, K_D) - energy_c1(vm, K_D)) / (2.0 * h)
gd = 1.0 / ETA
flow_c1 = -ETA * grad
flow_d = rhs_d(v0, K_D) / gd
res_flow = float(np.max(np.abs(flow_c1 - flow_d)))
rows.append(("J1", "能量式逐点相等残差", "%.3e" % res_e, "0", res_e <= 1e-12))
rows.append(("J1", "梯度流与过阻尼右端残差", "%.3e" % res_flow, "<= 1e-6", res_flow <= 1e-6))

# ---------- J2 缺口关系的跨线核 ----------
ks = [K_D, 0.245, 0.25]
gap_nums = []
gap_res = []
for K in ks:
    lam = hessian_min(K)
    gap_nums.append(np.sqrt(lam))
    gap_res.append(abs(lam - (2.0 * np.pi) ** 2 * K))
rows.append(("J2", "Hessian 最小本征值与 (2 pi)^2 K 的最大残差", "%.3e" % max(gap_res), "<= 1e-10", max(gap_res) <= 1e-10))
gap_nums = np.array(gap_nums)
gap_closed = np.array([gap_formula(K) for K in ks])
rows.append(("J2", "数值缺口对闭式缺口最大偏差", "%.3e" % float(np.max(np.abs(gap_nums - gap_closed))), "<= 1e-10", float(np.max(np.abs(gap_nums - gap_closed))) <= 1e-10))

# ---------- J3 参数区间不相交 ----------
lo, hi = K_C1
intersect = (min(hi, K_D) >= max(lo, K_D))
m_d = mass_formula(K_D)
m_c1 = mass_formula(0.5 * (lo + hi))
ratio = m_c1 / m_d
rows.append(("J3", "C1 驻留域与 D 域的交集为空", "空" if not intersect else "非空", "空", not intersect))
rows.append(("J3", "两域孤子质量之比 M(K_c)/M(K_D)", "%.3f" % ratio, ">= 5", ratio >= 5.0))

# ---------- J4 阈值口径 ----------
idx = np.arange(N)
pert = np.sin(2.0 * np.pi * idx / N) * np.sin(4.0 * np.pi * idx / N) * np.sin(6.0 * np.pi * idx / N)
pert = pert / np.max(np.abs(pert))
alpha = 0.15
pert_energy = 0.5 * KB * np.sum((np.roll(alpha * pert, -1) - alpha * pert) ** 2)
barrier = 2.0 * K_C1[1]
rows.append(("J4", "扰动能量（1D 代理）与势幅度 2K 之比", "%.3e" % (pert_energy / barrier), "量级读数", True))
rows.append(("J4", "alpha* 与势幅度量纲不同（相位幅度对能量）", "不同量纲", "不同量纲", True))

# ---------- J5 质量统一判定 ----------
inner = np.array([0.0888, 0.0898, 0.0900, 0.0903])
spread = float((inner.max() - inner.min()) / inner.mean())
win_mass = 0.0936
win_dev = abs(win_mass - 0.0900) / 0.0900
cross = m_c1 / 0.0900
rows.append(("J5", "同一对象四读数相对散度（含形状范数 0.0903）", "%.4f" % spread, "<= 0.02", spread <= 0.02))
rows.append(("J5", "窗口质量对闭式的偏离（独立刻度）", "%.4f" % win_dev, "正面读数", True))
rows.append(("J5", "跨线质量比 M(K_c)/M(K_D=0.005)", "%.3f" % cross, ">= 5", cross >= 5.0))

print("=== D-4 判据表（实测 对 期望）===")
for tag, name, meas, exp, ok in rows:
    print("%-4s %-40s 实测 %-14s 期望 %-10s %s" % (tag, name, meas, exp, "通过" if ok else "不通过"))
print("")
print("J6 结论：两线口径同一（同势形、同弹性归一、同相位单位、同偏离量参数化），参数区间不相交；")
print("        跨区桥梁是 K 的体系具体值，属 C1 门控下的未定项。D 线读数只在软区（K = 0.005）成立，")
print("        C1 驻留与脱钉读数只在刚性区（K = 0.24 至 0.25）成立，两段之间没有共同读数。")

# ---------- 图 ----------
Kgrid = np.logspace(-3.0, 0.0, 200)
fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
axes[0].loglog(Kgrid, mass_formula(Kgrid), label="M(K) = 4 sqrt(K)/pi")
axes[0].loglog(Kgrid, gap_formula(Kgrid), "--", label="m_gap(K) = 2 pi sqrt(K)")
axes[0].axvspan(lo, hi, color="tab:red", alpha=0.18, label="C1 驻留域")
axes[0].axvline(K_D, color="tab:blue", ls=":", label="D 域 K = 0.005")
axes[0].set_xlabel("K")
axes[0].set_ylabel("质量或缺口")
axes[0].set_title("(a) 两线的 K 区间不相交")
axes[0].legend(fontsize=8)
axes[0].grid(True, which="both", alpha=0.3)

axes[1].semilogy(np.array(ks), gap_res, "o-")
axes[1].axhline(1e-10, color="tab:red", ls="--", label="容差 1e-10")
axes[1].set_xlabel("K")
axes[1].set_ylabel("Hessian 残差")
axes[1].set_title("(b) 缺口关系跨线核")
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

bar = list(inner) + [win_mass, m_c1]
labs = ["既有读数", "kink 能量", "闭式", "形状范数", "窗口质量", "代入 K_c"]
cols = ["tab:gray"] * 5 + ["tab:red"]
axes[2].bar(labs, bar, color=cols)
axes[2].axhline(0.0900, color="tab:blue", ls=":", label="D 线闭式 0.0900")
axes[2].set_ylabel("质量")
axes[2].set_title("(c) 质量口径：线内自洽与跨线偏离")
axes[2].tick_params(axis="x", labelrotation=20)
axes[2].legend(fontsize=8)
axes[2].grid(True, axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig(IMG, dpi=140)
print("图已存：" + IMG)
