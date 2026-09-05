# -*- coding: utf-8 -*-
"""
最小验证（SO(3) 链接变量版）：路径 B《缺陷标架与格面曲率》规则 1-7 的数值检验

路径 B 构造：
  规则 1-3  每格点三维标架 F(n) ∈ SO(3)，来自三元组慢相位梯度（真空下为常数）
  规则 4    链接变量 U_a(n) = F(n+a) F(n)^{-1}
  规则 5    曲率 = 链接变量绕格面的净转角亏损 ε_h（Regge）
  规则 6    缺陷 = 局域标架扭转（Volterra），绑定能作源（ε_h ∝ m）
  规则 7    真空平坦性：线性构型 → F 常数 → U = I → ε_h = 0

验证内容：
  [1] SO(3) 机制层（机器精度）
      1a 真空平坦性（含非单位常数标架）
      1b 纯规范陷阱：光滑单值标架的格面 holonomy 恒为 I（任意转角幅度，曲率是拓扑性的）
      1c 非可积链接才产生曲率：随机链接的格面角 ~ κ² 标度
  [2] Volterra 缺陷精确几何
      2a 分支切割：亏损恰为 Frank 矢量，绕环 holonomy 半径无关（锥面签名）
      2b 可加性与局域性：双缺陷同轴亏损相加，小环只读各自缺陷
      2c ε_h ∝ m 线性（√m 陷阱在机制中不存在）
  [3] 线性化远场（§四 步骤 2-4 的可验证部分）
      3a 分量离散泊松：ω_i(r) = Ω_i/(4πr)，格面通量 = Ω_i，(l_P/r)² 修正标度
      3b 非阿贝尔修正压低：精确链接转角 vs 线性化 Δω（非共轴源），偏差 ~ 1/r³
      3c 平滑/奇异分解：远场响应零曲率，缺陷承载全部 holonomy，无重复计数
"""
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.sparse.linalg import cg, LinearOperator

def axl2R(w):
    """轴向向量 -> SO(3) 矩阵（支持批量 (...,3)）"""
    return Rot.from_rotvec(w).as_matrix()

def R2axl(R):
    """SO(3) 矩阵 -> 轴向量（转角 ∈ [0, π]）"""
    return Rot.from_matrix(R).as_rotvec()

def ploop_angle(U_seq):
    """闭路径链接变量（按路径顺序）的净转角"""
    H = np.eye(3)
    for M in U_seq:
        H = M @ H
    return np.linalg.norm(R2axl(H))

print("=" * 72)
print("最小验证（SO(3) 链接变量版）：路径 B 规则 1-7")
print("=" * 72)

# ----------------------------------------------------------------------
# [1] SO(3) 机制层
# ----------------------------------------------------------------------
print("\n[1a] 真空平坦性（规则 7）：线性构型 → U = I → 格面角 = 0")
N = 10
R0 = axl2R(np.array([0.7, -0.4, 1.1]))   # 非单位常数标架（整体旋转的真空）
F = np.broadcast_to(R0, (N, N, N, 3, 3)).copy()

def links_from_F(F):
    """三个方向的链接变量 U_a(n) = F(n+a) F(n)^T（前向约定）"""
    return [np.roll(F, -1, axis=a) @ np.swapaxes(F, -1, -2) for a in range(3)]

def plaquette_angles(U):
    """全部 xy/yz/xz 格面的净转角（路径序：U_y^-1 U_x^-1 U_y U_x 伸缩抵消）"""
    angs = []
    for a in range(3):
        for b in range(a + 1, 3):
            Ux, Uy = U[a], U[b]
            H = (np.swapaxes(Uy, -1, -2)
                 @ np.roll(np.swapaxes(Ux, -1, -2), -1, axis=b)
                 @ np.roll(Uy, -1, axis=a)
                 @ Ux)
            angs.append(np.linalg.norm(R2axl(H), axis=-1))
    return angs

max_ang = max(A.max() for A in plaquette_angles(links_from_F(F)))
print(f"    全部格面最大转角 = {max_ang:.3e}（平坦）")

print("\n[1b] 纯规范陷阱：光滑单值标架的格面 holonomy 恒为 I（任意转角幅度）")
ijk = np.indices((N, N, N)).transpose(1, 2, 3, 0)
for eps in (0.05, 0.5, 0.95):
    w = eps * np.sin(2.0 * np.pi * ijk / N)
    F = axl2R(w)
    max_ang = max(A.max() for A in plaquette_angles(links_from_F(F)))
    print(f"    ε = {eps:4.2f} rad：全部格面最大转角 = {max_ang:.3e}"
          f"（纯规范精确抵消：曲率是拓扑性而非幅度性）")

print("\n[1c] 非可积链接才产生曲率：格面角随链接强度线性增长，可积性是判别式")
rng = np.random.default_rng(7)
kappas = [0.05, 0.1, 0.2, 0.4]
means = []
for k in kappas:
    axes = rng.normal(size=(N, N, N, 3))
    axes /= np.linalg.norm(axes, axis=-1, keepdims=True)
    U3 = [axl2R(k * axes) for _ in range(3)]   # 每方向同一随机场（非可积：不来自单值 F）
    angs = plaquette_angles(U3)
    means.append(np.mean([A.mean() for A in angs]))
slope = np.polyfit(np.log(kappas), np.log(means), 1)[0]
print(f"    κ = {kappas} → 平均格面角 = {['%.3e' % m for m in means]}")
print(f"    标度 ~ κ^{slope:.2f}（期望 ≈ 1：非可积链接的一阶项不抵消；"
      f"结合 [1b]：可积 = 严格零，非可积 = 线性增长，判别式是可积性而非幅度）")

# ----------------------------------------------------------------------
# [2] Volterra 缺陷精确几何（规则 5-6）
# ----------------------------------------------------------------------
print("\n[2] Volterra 线缺陷（沿 z，Frank 矢量关于 z 轴）")
N2 = 24
c2 = N2 // 2

def make_link(Omega1, Omega2, use2):
    """切割链接函数：缺陷 1 于 (c2,c2) 切沿 +x（y=c2, x>c2）；缺陷 2 于 (c2,c2-8) 切沿 +x（y=c2-8, x>c2）"""
    R1 = axl2R(np.array([0.0, 0.0, Omega1]))
    R2 = axl2R(np.array([0.0, 0.0, Omega2]))
    def lk(pos, a):
        x, y, z = pos
        if a == 1 and y == c2 and x > c2:
            return R1
        if use2 and a == 1 and y == c2 - 8 and x > c2:
            return R2
        return np.eye(3)
    return lk

def loop_angle(lk, center, radius):
    """xy 平面方形环（逆时针）绕 center 的净转角"""
    x0, y0 = center
    U_seq = []
    for i in range(-radius, radius):
        U_seq.append(lk((x0 + i, y0 - radius, c2), 0))
    for j in range(-radius, radius):
        U_seq.append(lk((x0 + radius, y0 + j, c2), 1))
    for i in range(radius, -radius, -1):
        U_seq.append(np.linalg.inv(lk((x0 + i, y0 + radius, c2), 0)))
    for j in range(radius, -radius, -1):
        U_seq.append(np.linalg.inv(lk((x0 - radius, y0 + j, c2), 1)))
    return ploop_angle(U_seq)

lk1 = make_link(0.8, 0.0, use2=False)
angs = [loop_angle(lk1, (c2, c2), R) for R in (3, 5, 8, 10)]
print("\n[2a] 分支切割：亏损 = Frank 矢量，半径无关（锥面签名）")
print(f"    绕缺陷环（半径 3/5/8/10）转角 = {['%.12f' % a for a in angs]}，期望 Ω = 0.8")

lk2 = make_link(0.5, 0.3, use2=True)
print("\n[2b] 可加性与局域性：双缺陷（Ω₁=0.5 于 (c2,c2)，Ω₂=0.3 于 (c2,c2-8)）")
print(f"    大环（心 (c2,c2-4)，半径 8，含双缺陷） = {loop_angle(lk2, (c2, c2 - 4), 8):.12f}，期望 0.8")
print(f"    小环 1（心缺陷 1，半径 3）             = {loop_angle(lk2, (c2, c2), 3):.12f}，期望 0.5")
print(f"    小环 2（心缺陷 2，半径 3）             = {loop_angle(lk2, (c2, c2 - 8), 3):.12f}，期望 0.3")

print("\n[2c] ε_h ∝ m 线性（√m 陷阱不存在于机制中）")
k_line = 0.4
for m in (1.0, 2.0, 3.0):
    lkm = make_link(k_line * m, 0.0, use2=False)
    a_m = loop_angle(lkm, (c2, c2), 3)
    print(f"    m = {m:.0f}：亏损 = {a_m:.12f}（期望 {k_line * m}），残差 {abs(a_m - k_line * m):.2e}")

# ----------------------------------------------------------------------
# [3] 线性化远场（§四 步骤 2-4 可验证部分）
# ----------------------------------------------------------------------
print("\n[3] 线性化远场（N=81 接地盒，向量源 Ω = (1.0, 0.6, 0.3)）")
N3 = 81
ctr = N3 // 2
shape = (N3, N3, N3)
src = np.zeros(shape)
src[ctr, ctr, ctr] = 1.0

def lap(u):
    s = 6.0 * u
    s[1:, :, :] -= u[:-1, :, :]
    s[:-1, :, :] -= u[1:, :, :]
    s[:, 1:, :] -= u[:, :-1, :]
    s[:, :-1, :] -= u[:, 1:, :]
    s[:, :, 1:] -= u[:, :, :-1]
    s[:, :, :-1] -= u[:, :, 1:]
    return s

def solve3d(s_val, rtol=1e-12, maxiter=3000):
    def mv(x):
        u = x.reshape(shape).copy()
        u[0, :, :] = 0; u[-1, :, :] = 0; u[:, 0, :] = 0
        u[:, -1, :] = 0; u[:, :, 0] = 0; u[:, :, -1] = 0
        r = np.zeros(shape)
        r[1:-1, 1:-1, 1:-1] = lap(u)[1:-1, 1:-1, 1:-1]
        return r.ravel()
    op = LinearOperator((N3**3, N3**3), matvec=mv, dtype=np.float64)
    x, info = cg(op, src.ravel() * s_val, rtol=rtol, maxiter=maxiter)
    u = x.reshape(shape)
    u[0, :, :] = 0; u[-1, :, :] = 0; u[:, 0, :] = 0
    u[:, -1, :] = 0; u[:, :, 0] = 0; u[:, :, -1] = 0
    return u, info

Omega_vec = np.array([1.0, 0.6, 0.3])
w = np.zeros(shape + (3,))
A_th = 1.0 / (4.0 * np.pi)
ii, jj, kk = np.indices(shape)
r = np.sqrt((ii - ctr) ** 2.0 + (jj - ctr) ** 2.0 + (kk - ctr) ** 2.0)
mask = np.zeros(shape, dtype=bool); mask[1:-1, 1:-1, 1:-1] = True
fit = mask & (r >= 4.0) & (r <= 25.0)
vol10 = ((ii - ctr) ** 2 + (jj - ctr) ** 2 + (kk - ctr) ** 2) <= 100
A_fits = []
for i in range(3):
    ui, info = solve3d(Omega_vec[i])
    w[..., i] = ui
    M = np.stack([1.0 / r[fit], np.ones(fit.sum())], axis=1)
    (Ai, Ci), *_ = np.linalg.lstsq(M, ui[fit], rcond=None)
    A_fits.append(Ai)
    flux = np.sum(lap(ui)[vol10])
    print(f"    分量 {i}：拟合 A = {Ai:.8f}，理论 Ω_i/(4π) = {Omega_vec[i] * A_th:.8f}，"
          f"相对偏差 {abs(Ai - Omega_vec[i] * A_th) / (Omega_vec[i] * A_th):.3e}，"
          f"格面通量（R=10）= {flux:.10f}（期望 {Omega_vec[i]}）")
print("    注：单源场 w = g(n)·Ω 严格同轴，转动可交换（非阿贝尔偏差为零），"
      "非阿贝尔检验需非平行双源（见 [3b]）")

print("\n[3b] 非阿贝尔修正压低（非平行双源）：精确链接转角 vs 线性化 Δω")
srcB = np.zeros(shape); srcB[ctr + 8, ctr, ctr] = 1.0
def solve3d_src(rhs, rtol=1e-12, maxiter=3000):
    def mv(x):
        u = x.reshape(shape).copy()
        u[0, :, :] = 0; u[-1, :, :] = 0; u[:, 0, :] = 0
        u[:, -1, :] = 0; u[:, :, 0] = 0; u[:, :, -1] = 0
        r = np.zeros(shape)
        r[1:-1, 1:-1, 1:-1] = lap(u)[1:-1, 1:-1, 1:-1]
        return r.ravel()
    op = LinearOperator((N3**3, N3**3), matvec=mv, dtype=np.float64)
    x, info = cg(op, rhs.ravel(), rtol=rtol, maxiter=maxiter)
    u = x.reshape(shape)
    u[0, :, :] = 0; u[-1, :, :] = 0; u[:, 0, :] = 0
    u[:, -1, :] = 0; u[:, :, 0] = 0; u[:, :, -1] = 0
    return u
w2 = np.zeros(shape + (3,))
w2[..., 2] = solve3d_src(src)          # 源 A：Ω_A = ẑ 于中心
w2[..., 0] = solve3d_src(0.7 * srcB)   # 源 B：Ω_B = 0.7 x̂ 于 (ctr+8, ctr, ctr)
rs, devs, lins = [], [], []
for x in range(ctr + 14, N3 - 2):
    n = (x, ctr, ctr); n1 = (x + 1, ctr, ctr)
    Fn = axl2R(w2[n]); Fn1 = axl2R(w2[n1])
    Ux = Fn1 @ Fn.T
    rot_exact = R2axl(Ux)
    rot_lin = w2[n1] - w2[n]
    rs.append(x - ctr)
    devs.append(np.linalg.norm(rot_exact - rot_lin))
    lins.append(np.linalg.norm(rot_lin))
rs, devs, lins = np.array(rs), np.array(devs), np.array(lins)
zone = devs > 0
slope_dev = np.polyfit(np.log(rs[zone]), np.log(devs[zone]), 1)[0]
slope_lin = np.polyfit(np.log(rs), np.log(lins), 1)[0]
slope_rel = np.polyfit(np.log(rs[zone]), np.log(devs[zone] / lins[zone]), 1)[0]
print(f"    绝对偏差 ~ r^{slope_dev:.2f}（理论 ≈ -4：BCH 交换子 |Δω×ω|，共线双源下 1/r³ 项相消）")
print(f"    线性项 |Δω| ~ r^{slope_lin:.2f}；相对压低 (偏差/线性项) ~ r^{slope_rel:.2f}"
      f"（非阿贝尔修正在远场强烈压低，线性化泊松充分）")

print("\n[3c] 平滑/奇异分解：远场响应零曲率（纯规范），缺陷承载全部 holonomy")
Fs = axl2R(w)                                   # 平滑响应场的标架（单值）
Us = links_from_F(Fs)
max_ang = max(A.max() for A in plaquette_angles(Us))
print(f"    平滑响应场全部格面最大转角 = {max_ang:.3e}（= 0：远场不复制源，无重复计数）")
print(f"    缺陷 holonomy 由分支切割承载（[2] 已证）；总连接 = 平滑（零曲率）+ 奇异（亏损 Ω）")

# ----------------------------------------------------------------------
# 图
# ----------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    angmap = np.zeros((N2 - 1, N2 - 1))
    for x in range(N2 - 1):
        for y in range(N2 - 1):
            U_seq = [lk1((x, y, c2), 0),
                     lk1((x + 1, y, c2), 1),
                     np.linalg.inv(lk1((x, y + 1, c2), 0)),
                     np.linalg.inv(lk1((x, y, c2), 1))]
            angmap[x, y] = ploop_angle(U_seq)
    im = ax[0].imshow(angmap.T, origin="lower", cmap="viridis")
    fig.colorbar(im, ax=ax[0], fraction=0.046)
    ax[0].plot(c2, c2, "r*", ms=10)
    ax[0].set_title("plaquette deficit angle (Volterra cut, z=mid)")
    ax[0].set_xlabel("x"); ax[0].set_ylabel("y")
    rb = np.round(r * 2) / 2
    rsv = np.unique(rb[(rb >= 4) & (rb <= 30) & mask])
    tsv = np.array([w[..., 0][(np.abs(rb - v) < 1e-9) & mask].mean() for v in rsv])
    ax[1].loglog(rsv, tsv, "o", ms=4, label=r"$\omega_1(r)$ shell avg")
    ax[1].loglog(rsv, Omega_vec[0] * A_th / rsv, "-", lw=1, label=r"$\Omega_1/(4\pi r)$")
    ax[1].set_xlabel(r"r / $l_P$"); ax[1].set_title("far field of frame rotation (component 1)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    out = r"D:\software\TianXingFC\images\minimal_verification_so3.png"
    fig.savefig(out, dpi=150)
    print(f"\n图已保存：{out}")
except Exception as e:
    print(f"\n（图跳过：{e}）")

print("\n验证完成。")
