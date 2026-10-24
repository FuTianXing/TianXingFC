# -*- coding: utf-8 -*-
"""
Derrick 型检查：三维晶格里 σ 锁定 kink 能否稳定存在

两层分开检验（关键建模决定）：

第一层（摆程层，注释 03 的交错真空）：σ 锁定 = 摆程方向 σ 恒定（相位每拍 +2π 单调推进），
真空 = 交错锯齿（相邻反相、σ 交替）。检验：层内能量学（相邻格点耦合 J·cos(Δθ)）
对锁定是否敏感——若 2π 相对扫掠不可见，则摆程层对锁定既无稳定化也无塌缩力。

第二层（绕数层，κ_φ 层的拓扑代理）：三维格点 XY 场 φ ∈ S¹，铁磁耦合 -J·cos(Δφ)，
缠绕缺陷的四种嵌入：
  2a 无限线（周期盒，绕环面一圈）——拓扑/势垒稳定？
  2b 有限段（开放盒，自由端）——Derrick 塌缩？
  2c 环（开放盒，环形缠绕）——线张力收缩？
  并测绕数 W 在弛豫中的守恒/流失。

第三（A3 钉扎玩具检验）：晶格缺陷柱（局部弱耦合）对涡旋线的钉扎能。

输出：每项判据的实测 vs 期望，全部打印，不写注记。
"""
import numpy as np

print("=" * 72)
print("Derrick 型检查：三维晶格里 σ 锁定 kink 能否稳定存在")
print("=" * 72)

# ----------------------------------------------------------------------
# 第一层：摆程层对 σ 锁定的可见性
# ----------------------------------------------------------------------
print("\n[1] 摆程层：σ 锁定区的能量学可见性（2π 相对扫掠不可见性检验）")
N1 = 12
c1 = N1 // 2
Rd = 2
ii, jj, kk = np.indices((N1, N1, N1))
base = np.pi * (ii + jj + kk)                       # 交错真空的基准相位（mod 2π 锯齿）
stagger = (ii + jj + kk) % 2                        # 交错符号：相邻相反
droplet = ((ii - c1) ** 2 + (jj - c1) ** 2 + (kk - c1) ** 2) <= Rd ** 2

def sweep_energy(sigma, s):
    """s ∈ [0,1) 为拍内相位：theta = base + sigma*2*pi*s；E = J*sum cos(Δθ)，J=1，相邻键"""
    th = base + sigma * 2.0 * np.pi * s
    E = 0.0
    for ax in range(3):
        E += np.sum(np.cos(th - np.roll(th, -1, axis=ax)))
    return E

# 真空 σ：交错符号 × 拍的交替（σ_{n}(t) = stagger * (-1)^t）
sig_vac0 = 1.0 - 2.0 * stagger                       # 边界拍：相邻 σ 相反
sig_vac_half = -sig_vac0                             # 拍中反向扫掠的交错亦相反
# 锁定液滴：滴内 σ = +1 恒定，滴外真空模式
sig_lock0 = np.where(droplet, 1.0, sig_vac0)
sig_lock_half = np.where(droplet, 1.0, sig_vac_half)

for name, s, sv, sl in [("边界拍 s=0", 0.0, sig_vac0, sig_lock0),
                        ("拍中 s=0.25", 0.25, sig_vac0, sig_lock0),
                        ("拍中 s=0.5", 0.5, sig_vac_half, sig_lock_half),
                        ("拍中 s=0.75", 0.75, sig_vac_half, sig_lock_half)]:
    Ev = sweep_energy(sv, s)
    El = sweep_energy(sl, s)
    print(f"    {name}：E(真空) = {Ev:12.6f}  E(锁定滴) = {El:12.6f}  "
          f"差 = {El - Ev:+.3e}")
print("    结论：碰撞点（s=0, 0.5）锁定与真空严格简并（差 = 0）；"
      "拍内出现模型依赖的小差异（本扫掠参数化下为粗略代理，"
      "强 Derrick 力的判定在绕数层 [2]）")

# ----------------------------------------------------------------------
# 第二层：绕数层（三维格点 XY，铁磁耦合），四种缠绕嵌入
# ----------------------------------------------------------------------
def laplacian_sum(phi, periodic):
    """六邻居求和（sin(φ_i - φ_j) 之和用于梯度；能量另算）"""
    if periodic:
        s = (np.roll(phi, 1, 0) + np.roll(phi, -1, 0) +
             np.roll(phi, 1, 1) + np.roll(phi, -1, 1) +
             np.roll(phi, 1, 2) + np.roll(phi, -1, 2))
    else:
        p = np.pad(phi, 1, mode="constant", constant_values=0.0)
        s = (p[2:, 1:-1, 1:-1] + p[:-2, 1:-1, 1:-1] +
             p[1:-1, 2:, 1:-1] + p[1:-1, :-2, 1:-1] +
             p[1:-1, 1:-1, 2:] + p[1:-1, 1:-1, :-2])
    return s

def energy_xy(phi, periodic, Jmask=None):
    """E = -J*sum cos(Δφ)"""
    if periodic:
        pairs = [(np.roll(phi, -1, ax), phi, None) for ax in range(3)]
        E = 0.0
        for a, b, _ in pairs:
            E -= np.sum(np.cos(a - b))
        return E
    else:
        p = np.pad(phi, 1, mode="constant", constant_values=0.0)
        E = 0.0
        for ax in range(3):
            a = np.roll(p, -1, axis=ax)
            sl = [slice(1, -1)] * 3; sl[ax] = slice(1, -1)
            sl2 = [slice(1, -1)] * 3; sl2[ax] = slice(0, -2)
            E -= np.sum(np.cos(a[tuple(sl)] - p[tuple(sl2)]))
        return E

def relax(phi, periodic, iters=4000, eta=0.05):
    phi = phi.copy()
    for t in range(iters):
        # 精确梯度：dE/dφ_i = Σ sin(φ_i - φ_j)
        if periodic:
            s = (np.sin(phi - np.roll(phi, 1, 0)) + np.sin(phi - np.roll(phi, -1, 0)) +
                 np.sin(phi - np.roll(phi, 1, 1)) + np.sin(phi - np.roll(phi, -1, 1)) +
                 np.sin(phi - np.roll(phi, 1, 2)) + np.sin(phi - np.roll(phi, -1, 2)))
        else:
            p = np.pad(phi, 1, mode="constant", constant_values=0.0)
            s = np.zeros_like(phi)
            c = (slice(1, -1),) * 3
            s += (np.sin(p[c] - p[2:, 1:-1, 1:-1]) + np.sin(p[c] - p[:-2, 1:-1, 1:-1]) +
                  np.sin(p[c] - p[1:-1, 2:, 1:-1]) + np.sin(p[c] - p[1:-1, :-2, 1:-1]) +
                  np.sin(p[c] - p[1:-1, 1:-1, 2:]) + np.sin(p[c] - p[1:-1, 1:-1, :-2]))
        phi = phi - eta * s
        if not periodic:
            pass  # 边界由 pad 隐式固定为 0（仅内区更新）
    return phi

def winding(phi, loop_pts):
    """沿闭合路径的提升差分绕数"""
    W = 0.0
    for a, b in zip(loop_pts[:-1], loop_pts[1:]):
        d = phi[b] - phi[a]
        d = (d + np.pi) % (2 * np.pi) - np.pi
        W += d
    d = phi[loop_pts[0]] - phi[loop_pts[-1]]
    d = (d + np.pi) % (2 * np.pi) - np.pi
    W += d
    return W / (2 * np.pi)

print("\n[2a] 无限线（周期盒 N=32，沿 z 缠绕 1）：稳定性与绕数守恒")
N = 32
c = N // 2
shape = (N, N, N)
Y2, X2 = np.meshgrid(np.arange(N) - c, np.arange(N) - c, indexing="ij")
phi_line = np.broadcast_to(np.arctan2(Y2, X2)[:, :, None], shape).copy()
phi_line += 2 * np.pi * (X2[:, :, None] < -c)    # 展开分支（数值整洁，能量不受影响）
E0 = energy_xy(phi_line, periodic=True)
# 弛豫
phi_r = relax(phi_line, periodic=True, iters=3000, eta=0.05)
E1 = energy_xy(phi_r, periodic=True)
# 绕数：xy 平面方环（z=c 平面，绕线一圈）
r_w = 8
loop = [(c + r_w, c + t, c) for t in range(-r_w, r_w + 1)] + \
       [(c + t, c + r_w, c) for t in range(r_w, -r_w - 1, -1)] + \
       [(c - r_w, c + t, c) for t in range(r_w, -r_w - 1, -1)] + \
       [(c + t, c - r_w, c) for t in range(-r_w, r_w + 1)]
W0 = winding(phi_line, loop)
W1 = winding(phi_r, loop)
print(f"    E(线)/真空 = {E0 - (-3.0 * N ** 3):.3f}；弛豫后 = {E1 - (-3.0 * N ** 3):.3f}"
      f"（变化 {E1 - E0:+.3f}）")
print(f"    绕数 W：弛豫前 {W0:+.0f} → 弛豫后 {W1:+.0f}（线稳定则守恒）")

print("\n[2b] 有限段（开放盒 N=40，自由端）：Derrick 塌缩检验")
N2 = 40
c2 = N2 // 2
shape2 = (N2, N2, N2)
Y2, X2, Z2 = np.meshgrid(np.arange(N2) - c2, np.arange(N2) - c2,
                         np.arange(N2) - c2, indexing="ij")
Lseg = 8.0
taper = np.clip((Lseg / 2 - np.abs(Z2)) / 4.0, 0.0, 1.0)   # 端部 4 格平滑过渡
Sx = 1.0 + taper * (np.cos(np.arctan2(Y2, X2)) - 1.0)
Sy = taper * np.sin(np.arctan2(Y2, X2))
phi_seg = np.arctan2(Sy, Sx)
E0s = energy_xy(phi_seg, periodic=False)
phi_r2 = relax(phi_seg, periodic=False, iters=6000, eta=0.05)
E1s = energy_xy(phi_r2, periodic=False)
# 绕数：绕段中部横截面的方环
loop2 = [(c2 + 6, c2 + t, c2) for t in range(-6, 7)] + \
        [(c2 + t, c2 + 6, c2) for t in range(6, -7, -1)] + \
        [(c2 - 6, c2 + t, c2) for t in range(6, -7, -1)] + \
        [(c2 + t, c2 - 6, c2) for t in range(-6, 7)]
W0s = winding(phi_seg, loop2)
W1s = winding(phi_r2, loop2)
print(f"    E(段)/真空 = {E0s - (-3.0 * N2 ** 3):.3f}；弛豫 6000 步后 = {E1s - (-3.0 * N2 ** 3):.3f}"
      f"（变化 {E1s - E0s:+.3f}，期望下降至 ≈0：塌缩）")
print(f"    绕数 W：{W0s:+.0f} → {W1s:+.0f}（期望流失至 0）")

print("\n[2c] 环（开放盒 N=48，环形缠绕，线张力收缩检验）")
N3 = 48
c3 = N3 // 2
R3 = 10.0
shape3 = (N3, N3, N3)
x3 = np.arange(N3) - c3
X3, Y3, Z3 = np.meshgrid(x3, x3, x3, indexing="ij")
rho = np.sqrt(X3 ** 2 + Y3 ** 2)
u = rho - R3
v = Z3
psi = np.arctan2(v, u)
m_ring = np.clip((8.0 - np.sqrt(u ** 2 + v ** 2)) / 4.0, 0.0, 1.0)  # 管域平滑掩模
Sx = 1.0 + m_ring * (np.cos(psi) - 1.0)
Sy = m_ring * np.sin(psi)
phi_ring = np.arctan2(Sy, Sx)
E0r = energy_xy(phi_ring, periodic=False)
phi_r3 = relax(phi_ring, periodic=False, iters=6000, eta=0.05)
E1r = energy_xy(phi_r3, periodic=False)
# 绕管一圈的测量回路：在 α=0 处绕管截面（xz 平面内环绕管芯）
al = np.linspace(0, 2 * np.pi, 72, endpoint=False)
loop3 = [(int(round(c3 + R3 + 5 * np.cos(a))), c3, int(round(c3 + 5 * np.sin(a)))) for a in al]
W0r = winding(phi_ring, loop3)
W1r = winding(phi_r3, loop3)
print(f"    E(环)/真空 = {E0r - (-3.0 * N3 ** 3):.3f}；弛豫 6000 步后 = {E1r - (-3.0 * N3 ** 3):.3f}"
      f"（变化 {E1r - E0r:+.3f}，期望下降：线张力收缩）")
print(f"    绕管绕数 W：{W0r:+.0f} → {W1r:+.0f}（期望流失至 0）")

print("\n[3] A3 钉扎玩具检验：线固定于盒中心，移动缺陷弱耦合柱的位置（精确平移 mask）")
Jw = 0.3
def energy_column(phi, col):
    E = 0.0
    for ax in range(3):
        a = np.roll(phi, -1, axis=ax)
        Jb = np.where(col | np.roll(col, -1, axis=ax), Jw, 1.0)
        E -= np.sum(Jb * np.cos(a - phi))
    return E
# 线固定：phi_line0 = 沿 z 过盒中心的缠绕 1 涡旋线（周期盒 N=32）
N4 = 32
c4 = N4 // 2
shape4 = (N4, N4, N4)
Y4, X4 = np.meshgrid(np.arange(N4) - c4, np.arange(N4) - c4, indexing="ij")
phi_l = np.broadcast_to(np.arctan2(Y4, X4)[:, :, None], shape4).copy()
i4, j4, k4 = np.indices(shape4)
E_far = energy_column(phi_l, np.zeros(shape4, dtype=bool))     # 无柱基准
print(f"    线能量基准（无柱）= {E_far - (-3.0 * N4 ** 3):.3f}（每长度 {E_far - (-3.0 * N4 ** 3):.2f}）")
print(f"    {'柱-线距离':>10} {'E(线)-E(真空)':>16}")
for D in (0, 2, 4, 6, 9, 12, 16, 20):
    colD = ((i4 - D) ** 2 + j4 ** 2) <= 9.0     # 柱轴沿 z，位于距线 D 处（沿 x）
    Ee = energy_column(phi_l, colD) - E_far
    print(f"    {D:10d} {Ee:16.3f}")
print("    期望：D = 0 处（涡旋芯在弱耦合柱内）能量最低 → 钉扎阱；否则无钉扎")

print("\n检查完成。")
