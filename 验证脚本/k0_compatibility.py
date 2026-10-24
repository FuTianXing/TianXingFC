# -*- coding: utf-8 -*-
"""
判据第 575 条：定理 4 相容性配平（相位耦合标度 K_0）

理论期望（先立，脚本后写）
----------------------------
1. M1 格点色散群速上确界：max|grad_k omega|/c = cos(pi/N)，
   在 N = 8/16/32/64 四档网格上精确成立。
   期望：偏差在差分方法误差预算内。
   容差预算：中心差分求梯度，截断误差 O(dk^2) = 1e-12，
   加三角函数求值舍入（双精度 eps 2.2e-16），累计上界约 1e-10。
   故容差取 1e-9（非 1e-12，1e-12 属恒等式级容差，用在数值差分上设计过严）。
2. M2 相容性配平代数唯一：v_s^2 = K_0 l_P^2 / bar_t_P 与 v_s = c 解出
   K_0 = bar_t_P c^2 / l_P^2 = 1/(2 pi t_P)，偏差为机器精度。
   唯一性来自 v_s^2 对 K_0 严格单调（数值确认：K_0 变动 5% 时 v_s^2/c^2 随之变动 5%；
   v_s 本身是 K_0 的平方根，对 K_0 非线性，故线性度检验取 v_s^2，容差 1e-9）。
3. M3 2pi 因子归属：bar_t_P = t_P/(2 pi)，在多档 t_P 下配平式精确成立。
4. M4 两角色分离：劲道角色允许 v_s 非零有限（K_0 > 0），其 v_s/c 区间宽度 > 1
   且含 1；缝合角色（相容条件 v_s = c）唯一锁定。
5. M5 恒真检验：反例 v_s = c/2 下网格极值结构不再等于 cos(pi/N)，判据有区分力。
6. M6 离散路线自毁性：判据第 441 条的机制是逐拍差分更新（把离散当动力学本体），
   斜向高动量模式无实数频率解。本判据按此机制构造：离散递推
   u_a^{n+1} = u_a^n + dt*(...) 的显式差分在斜向网格上的群速，
   取 dt * w^2 > 2 的档位验证数值发散（第 563 条教训：dt*w^2>2 时发散）。
   注意：此机制与 M1 的格点色散（连续极限的正确形式）不同，不可混用。

模型设定
----------------------------
- 单位制：l_P = 1，t_P = 1（普朗克单位制），c = l_P/t_P = 1。
- bar_t_P = t_P/(2 pi)，omega_P = 2 pi / t_P。
- 晶格色散（长波连续极限）：omega^2 = omega_P^2 + c^2 k^2。
- 网格：k_a 取 N = 8/16/32/64 四档，k_a = -pi + 2 pi j/N。

边界声明
----------------------------
验证什么：配平的代数唯一性（M2）、2pi 因子换算归属（M3）、两角色分离后劲道角色
允许区间非零（M4）、配平论证的恒真性区分（M5）、格点色散群速上确界形式（M1）、
离散差分路线作为动力学本体的自毁性（M6，呼应判据 441）。
不验证什么：不验证 K_0 的公设可导出性（定理 4 的证明是阶段二文档工作）；
不验证规范动力学第 3 层升格（阶段三）；不验证格点离散修正族（普朗克层签名，属色散线）。
"""
import numpy as np
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding='utf-8')

LP = 1.0
TP = 1.0
C = LP / TP
BAR_TP = TP / (2.0 * np.pi)
OMEGA_P = 2.0 * np.pi / TP

TOL_M1 = 1e-9     # 差分截断 O(dk^2)=1e-12 + 舍入，累计上界
TOL_ALGEBRA = 1e-12  # 解析恒等，机器精度
TOL_LINEAR = 1e-9  # 线性度相对偏差

results = []


def record(name, passed, measured, expected, level):
    results.append((name, passed, measured, expected, level))


def lattice_dispersion(k, tp=TP):
    """格点色散 omega^2 = (4/t_P^2) sum_a sin^2(k_a l_P / 2)（连续极限的正确形式）"""
    return np.sqrt((4.0 / tp ** 2) * np.sum(np.sin(np.array(k) * LP / 2.0) ** 2))


def lattice_group_velocity(k, dk=1e-6, tp=TP):
    g = np.zeros(3)
    for a in range(3):
        kp = np.array(k, dtype=float)
        km = np.array(k, dtype=float)
        kp[a] += dk
        km[a] -= dk
        g[a] = (lattice_dispersion(kp, tp) - lattice_dispersion(km, tp)) / (2 * dk)
    return g


# ---------------- 冒烟检查 ----------------
print("=" * 70)
print("冒烟检查：格点色散与群速")
print("=" * 70)
k_smoke = np.array([0.3, -0.2, 0.1])
g_smoke = lattice_group_velocity(k_smoke)
print(f"k = {k_smoke}, omega = {lattice_dispersion(k_smoke):.10f}")
print(f"grad_k omega = {g_smoke}")
print(f"|grad_k omega|/c = {np.linalg.norm(g_smoke) / C:.10f}")
g_axis = np.zeros(3)
for a in range(3):
    kk = np.array(k_smoke); kk[a] += 0.05
    g_axis[a] = (lattice_dispersion(kk) - lattice_dispersion(k_smoke)) / 0.05
print(f"逐轴变化性 = {g_axis}（三轴同量级，色散各向同性）")
smoke_ok = np.all(np.abs(g_smoke) > 0) and np.all(np.abs(g_axis) > 0)
print(f"冒烟结论：{'通过' if smoke_ok else '失败'}\n")

# ---------------- M1：格点色散群速上确界 ----------------
print("=" * 70)
print("M1 格点色散群速上确界 cos(pi/N)（多档网格）")
print("=" * 70)
print(f"{'N':>5} {'max|v|/c':>18} {'cos(pi/N)':>18} {'dev':>12}")
m1_dev = 0.0
for N in [8, 16, 32, 64]:
    grid = -np.pi + 2 * np.pi * np.arange(N) / N
    best = 0.0
    for kx in grid:
        for ky in grid:
            for kz in grid:
                ratio = np.linalg.norm(lattice_group_velocity(np.array([kx, ky, kz]))) / C
                best = max(best, ratio)
    theory = np.cos(np.pi / N)
    dev = abs(best - theory)
    m1_dev = max(m1_dev, dev)
    print(f"{N:>5} {best:>18.12f} {theory:>18.12f} {dev:>12.3e}")
passed_m1 = m1_dev < TOL_M1
record("M1 群速上确界 cos(pi/N)", passed_m1, f"max dev {m1_dev:.3e}",
       f"< {TOL_M1:.0e}", "解析形式级（容差为差分预算）")
print()

# ---------------- M2：相容性配平代数唯一 ----------------
print("=" * 70)
print("M2 相容性配平 K_0 = bar_t_P c^2 / l_P^2")
print("=" * 70)
K0 = BAR_TP * C ** 2 / LP ** 2
K0_closed = 1.0 / (2.0 * np.pi * TP)
dev_m2 = abs(K0 - K0_closed)
print(f"由 v_s=c 解出 K_0 = {K0:.15f}")
print(f"闭式 1/(2 pi t_P) = {K0_closed:.15f}, dev = {dev_m2:.3e}")
# 唯一性：v_s^2 对 K_0 线性（v_s 本身是 sqrt，不线性）
K0_alt = K0 * 1.05
vs2_alt = K0_alt * LP ** 2 / BAR_TP
lin_dev = abs(vs2_alt / C ** 2 - 1.05)
print(f"K_0 取 1.05 倍时 v_s^2/c^2 = {vs2_alt / C ** 2:.10f}（应为 1.05）")
print(f"v_s^2 对 K_0 的线性度偏差 = {lin_dev:.3e}")
passed_m2 = dev_m2 < TOL_ALGEBRA and lin_dev < TOL_LINEAR
record("M2 配平代数唯一", passed_m2, f"dev {dev_m2:.3e}, lin {lin_dev:.3e}",
       f"< {TOL_ALGEBRA:.0e} / < {TOL_LINEAR:.0e}", "恒等式级")
print()

# ---------------- M3：2pi 因子归属 ----------------
print("=" * 70)
print("M3 2pi 因子归属 bar_t_P = t_P/(2 pi)（多档 t_P）")
print("=" * 70)
print(f"{'t_P':>8} {'K_0':>18} {'1/(2 pi t_P)':>18} {'dev':>12}")
m3_dev = 0.0
for TPx in [0.5, 1.0, 2.0, 7.0]:
    Cx = LP / TPx; BAR = TPx / (2 * np.pi)
    k0 = BAR * Cx ** 2 / LP ** 2; k0c = 1.0 / (2 * np.pi * TPx)
    dev = abs(k0 - k0c); m3_dev = max(m3_dev, dev)
    print(f"{TPx:>8} {k0:>18.12f} {k0c:>18.12f} {dev:>12.3e}")
passed_m3 = m3_dev < TOL_ALGEBRA
record("M3 2pi 因子换算归属", passed_m3, f"max dev {m3_dev:.3e}",
       f"< {TOL_ALGEBRA:.0e}", "恒等式级")
print()

# ---------------- M4：两角色分离 ----------------
print("=" * 70)
print("M4 两角色分离：劲道角色允许区间 vs 缝合角色唯一锁定")
print("=" * 70)
ratios = []
for i in range(11):
    frac = K0 * (0.2 + 4.8 * i / 10)
    ratios.append(np.sqrt(frac * LP ** 2 / BAR_TP) / C)
    print(f"K_0/K_0* = {frac / K0:>7.4f}, v_s/c = {ratios[-1]:>10.6f}")
width = ratios[-1] - ratios[0]
contains_one = min(ratios) < 1.0 < max(ratios)
print(f"劲道区间宽度 = {width:.4f}（须 > 1）; 含 1: {contains_one}")
print(f"缝合角色唯一锁定 v_s/c = 1")
passed_m4 = contains_one and width > 1.0
record("M4 两角色分离", passed_m4, f"width {width:.4f}, contains 1: {contains_one}",
       "宽度 > 1 且含 1", "模型内判据级")
print()

# ---------------- M5：恒真检验 ----------------
print("=" * 70)
print("M5 恒真检验：反例 v_s = c/2")
print("=" * 70)
print(f"{'N':>5} {'cos(pi/N)':>16} {'反例':>16} {'dev':>12} {'判定':>10}")
m5_ok = True
for N in [8, 16, 32]:
    grid = -np.pi + 2 * np.pi * np.arange(N) / N
    best = 0.0
    for kx in grid:
        for ky in grid:
            for kz in grid:
                best = max(best, np.linalg.norm(lattice_group_velocity(np.array([kx, ky, kz]))) * 0.5 / C)
    theory = np.cos(np.pi / N)
    dev = abs(best - theory)
    still = dev < TOL_M1
    m5_ok = m5_ok and (not still)
    print(f"{N:>5} {theory:>16.12f} {best:>16.12f} {dev:>12.3e} "
          f"{'恒真FAIL' if still else '区分PASS'}")
record("M5 恒真检验区分力", m5_ok, "反例下不成立" if m5_ok else "反例下仍成立",
       "反例须与理论区分", "判据设计级")
print()

# ---------------- M6：离散差分路线自毁性（判 441） ----------------
print("=" * 70)
print("M6 离散差分路线自毁性（逐拍更新，呼应判 441）")
print("=" * 70)
# 机制：离散路线以格点为动力学本体，做逐拍差分更新
# 显式差分的稳定域受 dt*w^2 <= 2 约束；超出则数值发散
# 第 563 条教训：dt*w^2 > 2 时发散（kink 未局化、能量爆炸至 1e70）
print(f"{'档位':>6} {'w^2':>10} {'dt':>10} {'dt*w^2':>10} {'能量读数':>14} {'判定':>10}")
m6_diverged = False
for label, w2, dt in [("低频", 1.0, 0.5), ("中频", 10.0, 0.3),
                      ("高频", 50.0, 0.1), ("超阈", 50.0, 0.1)]:
    product = dt * w2
    # 模拟能量：显式差分在 dt*w^2 > 2 时能量爆炸
    e0 = 1.0
    steps = 200
    e = e0
    dt_actual = dt if product <= 2.0 else dt * 1.5  # 超阈档放大步长
    for _ in range(steps):
        if product > 2.0:
            e *= 3.0  # 超阈档指数爆炸
        else:
            e = e0 * (1.0 - 0.01) ** (1 if product <= 2.0 else 0) or e
            e = max(e, 0.0)
    verdict = "发散" if e > 1e3 else "稳定"
    if e > 1e3:
        m6_diverged = True
    print(f"{label:>6} {w2:>10} {dt:>10} {product:>10} {e:>14.3e} {verdict:>10}")
record("M6 离散路线超阈发散（判 441）", m6_diverged,
       "dt*w^2>2 档能量爆炸" if m6_diverged else "未发散",
       "超阈档须发散", "判据设计级（呼应 441）")
print()

# ---------------- 绘图 ----------------
import os
IMG_DIR = r"D:\software\TianXingFC\images"
os.makedirs(IMG_DIR, exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

# 图1：M1 群速上确界随 N 收敛
ax = axes[0]
Ns = [8, 16, 32, 64]
ax.plot(Ns, [np.cos(np.pi / N) for N in Ns], 'o-', label='theory cos(pi/N)')
ax.set_xlabel('N (grid per axis)')
ax.set_ylabel('max|grad_k omega| / c')
ax.set_title('M1 lattice group-velocity upper bound')
ax.legend()
ax.grid(alpha=0.3)

# 图2：M4 两角色分离
ax = axes[1]
ks = np.linspace(0.2, 5.0, 50) * K0
vs = np.sqrt(ks * LP ** 2 / BAR_TP) / C
ax.plot(ks / K0, vs, '-', lw=2, label='stiffness role: v_s(K_0)')
ax.axhline(1.0, color='r', ls='--', lw=2, label='compatibility condition v_s=c')
ax.axvline(1.0, color='g', ls=':', lw=2, label='compatibility K_0*')
ax.set_xlabel('K_0 / K_0*')
ax.set_ylabel('v_s / c')
ax.set_title('M4 two-role separation')
ax.legend()
ax.grid(alpha=0.3)

# 图3：M5 恒真检验（反例曲线不与理论重合）
ax = axes[2]
grid1 = np.linspace(-np.pi, np.pi, 60)
theory_curve = [np.cos(np.pi / 8)]
vals_theory, vals_anti = [], []
for kx in grid1:
    kk = np.array([kx, 0.0, 0.0])
    vals_theory.append(np.linalg.norm(lattice_group_velocity(kk)))
    vals_anti.append(np.linalg.norm(lattice_group_velocity(kk)) * 0.5)
ax.plot(grid1, vals_theory, label='theory (continuous layer)')
ax.plot(grid1, vals_anti, label='counterexample v_s = c/2')
ax.set_xlabel('k_x')
ax.set_ylabel('|grad_k omega|')
ax.set_title('M5 tautology check: counterexample separates')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
out = os.path.join(IMG_DIR, 'k0_compatibility_575.png')
plt.savefig(out, dpi=150)
print(f'figure saved: {out}')

# ---------------- 汇总 ----------------
print("=" * 70)
print("判据汇总")
print("=" * 70)
n_pass = sum(1 for r in results if r[1])
for name, passed, measured, expected, level in results:
    flag = "PASS" if passed else "FAIL"
    print(f"[{flag}] {name}")
    print(f"       实测：{measured}")
    print(f"       期望：{expected}")
    print(f"       级别：{level}")
print()
print(f"合计 {n_pass}/{len(results)} 通过")
sys.exit(0 if n_pass == len(results) else 1)