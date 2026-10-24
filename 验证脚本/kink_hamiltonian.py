# -*- coding: utf-8 -*-
"""
体系完整哈密顿量的 kink 构造（临时，未固化）

σ 锁定 kink 的三维实现：三元组相位场（每格点三个正交元相子 φ^x,φ^y,φ^z）
上，φ^x 携带环状拓扑缠绕（绕数 n，电荷 = 绕数定案），叠加在交错真空
（φ^a = π(i+j+k)）上。哈密顿量 = 体系既有两项：

  H = κ_φ Σ_{a<b}(1 - cos δ^{ab})   格点内异类耦合（相对相位阱）
    + κ_∥ Σ_{a,c}(1 - cos Δ_cφ^a)   格点间同类耦合（相位梯度）

真空：δ = 0（阱底）、Δφ = π（交错序）。缺陷 = 相对真空的拓扑偏离，
绑定能 = E(构型) - E(真空)。

变分：扫环半径 R 与管半径 ρ0，找能量极小；检验 E(R) 是否在物理允许区
（R > ρ0）有内部极小（aR + b/R 型），及 ρ0* 与 κ_φ/κ_∥ 的关系
（缺陷紧致度 = κ_φ 族的经济性主张的数值检验）。
"""
import numpy as np

KAPPAPAR = 1.0  # κ_∥ 固定


def minimage(d):
    return (d + np.pi) % (2.0 * np.pi) - np.pi


def energy(phi, kappa_phi):
    """phi: (3,N,N,N)，返回相对真空能量。"""
    N = phi.shape[1]
    # 格点内：δ^{ab}（最小像）
    E_in = 0.0
    for a, b in [(0, 1), (1, 2), (2, 0)]:
        E_in += np.sum(1.0 - np.cos(minimage(phi[a] - phi[b])))
    # 格点间：Δ_cφ^a（最小像），减真空常数（Δφ=π -> 1-cos=2）
    E_out = 0.0
    for a in range(3):
        for ax in range(3):
            d = np.diff(phi[a], axis=ax)
            E_out += np.sum(1.0 - np.cos(minimage(d))) - 2.0 * (N - 1) * N * N
    return kappa_phi * E_in + KAPPAPAR * E_out


def make_config(N, R, rho0, w, n_topo):
    """真空交错序 + φ^x 上环状缠绕（环在 xy 平面，中心原点）。"""
    i = np.arange(N) - N // 2
    x, y, z = np.meshgrid(i, i, i, indexing="ij")
    vac = np.pi * (x + y + z)                       # 交错真空（对所有分量）
    rho = np.sqrt(x**2 + y**2)
    d = np.sqrt((rho - R) ** 2 + z**2)              # 到环最短距离
    g = 0.5 * (1.0 - np.tanh((d - rho0) / w))       # 管权重：管内 1，管外 0
    wrap = n_topo * np.arctan2(y, x) * g            # φ^x 上的环向缠绕
    phi = np.zeros((3, N, N, N))
    phi[0] = vac + wrap
    phi[1] = vac
    phi[2] = vac
    return phi


def vac_energy(N, kappa_phi):
    phi = np.zeros((3, N, N, N))
    i = np.arange(N) - N // 2
    x, y, z = np.meshgrid(i, i, i, indexing="ij")
    vac = np.pi * (x + y + z)
    phi[0] = phi[1] = phi[2] = vac
    return energy(phi, kappa_phi)


print("=" * 72)
print("体系完整哈密顿量的 kink 构造（σ 锁定涡旋环）")
print("=" * 72)
print(f"κ_∥ = {KAPPAPAR}；真空 = 交错序（δ=0, Δφ=π）")

N = 61
n_topo = 1
w = 1.0
R_grid = np.arange(3.0, 19.0, 1.0)
rho_grid = np.array([1.5, 2.0, 2.5, 3.0, 3.5, 4.0])

for kappa_phi in (0.5, 1.0, 2.0):
    print(f"\n--- κ_φ = {kappa_phi}（κ_φ/κ_∥ = {kappa_phi}）---")
    Ev = vac_energy(N, kappa_phi)
    # 对每个 R，变分最优 ρ0
    E_opt = []
    rho_opt = []
    for R in R_grid:
        best = None
        for r0 in rho_grid:
            phi = make_config(N, R, r0, w, n_topo)
            E = energy(phi, kappa_phi) - Ev
            if best is None or E < best[0]:
                best = (E, r0)
        E_opt.append(best[0])
        rho_opt.append(best[1])
    E_opt = np.array(E_opt)
    rho_opt = np.array(rho_opt)
    imin = int(np.argmin(E_opt))
    print("  R    E_rel       ρ0*    dE/dR")
    for i, R in enumerate(R_grid):
        tag = "" if i == 0 else ("  (增)" if E_opt[i] - E_opt[i - 1] > 0 else "  (降)")
        print(f"  {R:4.0f}  {E_opt[i]:10.3f}   {rho_opt[i]:4.1f}{tag}")
    in_region = R_grid[imin] > rho_opt[imin]
    print(f"  极小：R* = {R_grid[imin]:.0f}，E_B = {E_opt[imin]:.3f}，ρ0* = {rho_opt[imin]:.1f}")
    print(f"  极小在物理允许区（R* > ρ0*） = {in_region}")
    # 拟合 E = aR + b/R + c（用允许区 R >= 6 的数据，避免小 R 自交区）
    mask = R_grid >= 6.0
    A = np.stack([R_grid[mask], 1.0 / R_grid[mask], np.ones(mask.sum())], axis=1)
    coef, *_ = np.linalg.lstsq(A, E_opt[mask], rcond=None)
    E_fit = A @ coef
    res = np.max(np.abs(E_fit - E_opt[mask])) / np.max(np.abs(E_opt[mask]))
    print(f"  拟合（R≥6）E = aR + b/R + c：a={coef[0]:.4f}, b={coef[1]:.4f}, c={coef[2]:.4f}，残差 {res:.2e}")
    if coef[0] > 0 and coef[1] > 0:
        R_opt = np.sqrt(coef[1] / coef[0])
        print(f"  解析极小 R* = sqrt(b/a) = {R_opt:.2f}（a>0,b>0：碗形稳定极小）")
    # ρ0* 随 κ_φ 趋势
    print(f"  最优管半径 ρ0* = {rho_opt[imin]:.1f}（缺陷紧致度，κ_φ 族）")

print("\n构造完成。")
