# -*- coding: utf-8 -*-
"""
最小验证：碰撞写入 + 标架联络（P1-32 方向 6 的落地第一步）

机制翻译为方程：
  场变量：格点标架倾斜 theta（连续标量，住波性侧取向场，注释 09"方向归波"）
  机制过程：每拍 (i) 邻居抄写平均（碰撞接力）theta <- theta + alpha*(邻居平均 - theta)
           (ii) 缺陷区每拍注入 w（写入幅度 ∝ delta_theta = m/m_P）
  稳态：-nabla^2 theta = s * delta（离散泊松）

验证内容：
  A 一维：稳态剖面 = 离散格林函数（分段线性，1D 牛顿势）；
          "每拍写入 + 邻居抄写"动力学收敛到同一稳态，幅度比 s_eff = 2w/alpha 精确
  B 三维：远场 1/r，系数 A = s*l_P^2/(4*pi)；
          离散高斯定律：格面通量 = 源强（半径无关，机器精度）；
          4*pi 作为"格点通量=源"与"连续立体角"的换算自动出现；
          对 1/r 的绝对偏差 ~ 1/r^3（相对偏差 ~ (l_P/r)^2，弱场展开成立）；
          场对注入幅度线性（幅度 ∝ m/m_P）
"""
import numpy as np
from scipy.sparse.linalg import cg, LinearOperator

R_REPORT = []
print("=" * 72)
print("最小验证：碰撞写入 + 标架联络")
print("=" * 72)

# ----------------------------------------------------------------------
# A. 一维链：精确解 + 写入动力学收敛
# ----------------------------------------------------------------------
print("\n[A] 一维链（N=201，狄利克雷边界，源 s=1 于中心）")
N = 201
c = N // 2
A1 = 2.0 * np.eye(N) - np.eye(N, k=1) - np.eye(N, k=-1)
A1[0, :] = 0.0; A1[0, 0] = 1.0
A1[-1, :] = 0.0; A1[-1, -1] = 1.0
b1 = np.zeros(N); b1[c] = 1.0
theta1 = np.linalg.solve(A1, b1)

n = np.arange(N)
exact1 = np.where(n <= c,
                  n * (N - 1 - c) / (N - 1),
                  (N - 1 - n) * c / (N - 1))
err_exact = np.max(np.abs(theta1 - exact1))
print(f"  线性求解 vs 分段线性精确解：最大绝对误差 = {err_exact:.3e}")

# "每拍写入 + 邻居抄写"动力学：th <- th + alpha*(avg - th)；每拍向缺陷注入 w
alpha, w = 1.0, 1.0e-3
th = np.zeros(N)
ticks = 600000
for t in range(ticks):
    avg = 0.5 * (th[:-2] + th[2:])
    th[1:-1] += alpha * (avg - th[1:-1])
    th[c] += w
s_eff = 2.0 * w / alpha  # 稳态方程 -lap theta = s_eff * delta
err_dyn = np.max(np.abs(th - s_eff * exact1)) / s_eff
print(f"  写入动力学（alpha={alpha}, w={w}, {ticks} 拍）：稳态 = 源强 {s_eff} x s=1 格林函数")
print(f"  动力学稳态相对误差 = {err_dyn:.3e}")
print(f"  一维结论：剖面精确线性（1D 牛顿势），写入机制收敛到离散泊松，幅度比精确")

# ----------------------------------------------------------------------
# B. 三维格点：远场 1/r、格面通量、4pi、修正标度、线性
# ----------------------------------------------------------------------
print("\n[B] 三维立方格点（N=81，狄利克雷边界，源 s=1 于中心，l_P=1）")
N3 = 81
ctr = N3 // 2
shape = (N3, N3, N3)
src = np.zeros(shape)
src[ctr, ctr, ctr] = 1.0

def lap(u):
    """6*theta - 六邻居和（边界外为零填充），u 为全场（边界应为 0）"""
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
        u = x.reshape(shape).copy()  # 必须拷贝：原地修改会污染 CG 内部向量
        u[0, :, :] = 0; u[-1, :, :] = 0; u[:, 0, :] = 0
        u[:, -1, :] = 0; u[:, :, 0] = 0; u[:, :, -1] = 0
        r = np.zeros(shape)
        r[1:-1, 1:-1, 1:-1] = lap(u)[1:-1, 1:-1, 1:-1]  # 算子 = 6θ−Σnb = −∇²θ
        return r.ravel()
    op = LinearOperator((N3**3, N3**3), matvec=mv, dtype=np.float64)
    b = src.ravel() * s_val  # 方程 −∇²θ = s·δ，解 θ = s/(4πr) > 0
    x, info = cg(op, b, rtol=rtol, maxiter=maxiter)
    u = x.reshape(shape)
    u[0, :, :] = 0; u[-1, :, :] = 0; u[:, 0, :] = 0
    u[:, -1, :] = 0; u[:, :, 0] = 0; u[:, :, -1] = 0
    return u, info

u1, info1 = solve3d(1.0)
resid_int = np.max(np.abs((lap(u1) - src)[1:-1, 1:-1, 1:-1]))
print(f"  CG 收敛 info={info1}，内部残差 max|−∇²θ − s·δ| = {resid_int:.3e}")

# 远场 1/r 拟合（r in [4, 25]；接地盒引入常数项 C，双参数拟合）
ii, jj, kk = np.indices((N3, N3, N3))
r = np.sqrt((ii - ctr) ** 2.0 + (jj - ctr) ** 2.0 + (kk - ctr) ** 2.0)
mask_int = np.zeros(shape, dtype=bool)
mask_int[1:-1, 1:-1, 1:-1] = True
fit_zone = mask_int & (r >= 4.0) & (r <= 25.0) & (r > 0)
rf, tf = r[fit_zone], u1[fit_zone]
M = np.stack([1.0 / rf, np.ones_like(rf)], axis=1)
(A_fit, C_fit), *_ = np.linalg.lstsq(M, tf, rcond=None)
A_theory = 1.0 / (4.0 * np.pi)
print(f"  远场拟合 A = {A_fit:.8f}（常数项 C = {C_fit:.3e}），理论 A* = s*l_P^2/(4*pi) = {A_theory:.8f}")
print(f"  相对偏差 = {abs(A_fit - A_theory) / A_theory:.3e}")

# 对 1/r 的偏差标度（分壳平均，扣除拟合常数）
rb = np.round(r * 2) / 2.0
sel = mask_int & (rb >= 3.0) & (rb <= 30.0)
r_sel, t_sel, rb_sel = r[sel], u1[sel], rb[sel]
keys = np.unique(rb_sel)
r_sh = keys
t_sh = np.array([t_sel[rb_sel == k].mean() for k in keys])
dev = t_sh - (A_fit / r_sh + C_fit)
zone = (r_sh >= 6) & (r_sh <= 25) & (np.abs(dev) > 0)
slope = np.polyfit(np.log(r_sh[zone]), np.log(np.abs(dev[zone])), 1)[0]
print(f"  扣除常数后绝对偏差 ~ r^{slope:.2f}（期望约 -3，对应相对偏差 ~ (l_P/r)^2）")

# 格面通量 = 源强（离散高斯定律，半径无关）
print("  离散高斯定律：闭合格面通量（应恒等于 s=1）")
for R in (5, 10, 20, 30):
    vol = (ii - ctr) ** 2 + (jj - ctr) ** 2 + (kk - ctr) ** 2 <= R * R
    flux = np.sum(lap(u1)[vol])
    print(f"    R = {R:2d}: 通量 = {flux:.12f}（偏差 {abs(flux - 1.0):.2e}）")

# 4pi 涌现：连续立体角换算 4*pi*r^2*E(r) ≈ s
r_t = 20.0
i_sh = np.argmin(np.abs(r_sh - r_t))
E = -(t_sh[i_sh + 1] - t_sh[i_sh - 1]) / (r_sh[i_sh + 1] - r_sh[i_sh - 1])
flux_cont = 4.0 * np.pi * r_t**2 * E
print(f"  连续球面通量 4*pi*r^2*E(r={r_t:.0f}) = {flux_cont:.6f}（源强 1，4pi 由立体角换算自动出现）")

# 线性：s=3 的解 = 3 倍 s=1 的解
u3, info3 = solve3d(3.0)
err_lin = np.max(np.abs(u3 - 3.0 * u1))
print(f"  线性检验：max|theta(s=3) - 3*theta(s=1)| = {err_lin:.3e}（幅度 ∝ m/m_P）")

# ----------------------------------------------------------------------
# 图（可选）
# ----------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(n - c, theta1, ".", ms=3, label="线性求解稳态")
    ax[0].plot(n - c, exact1, "-", lw=1, label="1D 牛顿势（分段线性精确解）")
    ax[0].plot(n - c, s_eff * exact1, "--", lw=1, label=f"写入动力学稳态（×{s_eff}）")
    ax[0].set_xlabel("格点 n - c"); ax[0].set_ylabel("标架倾斜 theta")
    ax[0].set_title("一维链：写入动力学收敛到离散泊松"); ax[0].legend(fontsize=8)
    rlo = (r_sh >= 4) & (r_sh <= 30)
    ax[1].loglog(r_sh[rlo], t_sh[rlo], "o", ms=4, label="三维壳平均 theta(r)")
    ax[1].loglog(r_sh[rlo], A_theory / r_sh[rlo], "-", lw=1,
                 label=r"$s\,l_P^2/(4\pi r)$")
    ax[1].set_xlabel("r / l_P"); ax[1].set_ylabel("theta")
    ax[1].set_title("三维：远场 1/r 与 4pi 系数"); ax[1].legend(fontsize=8)
    fig.tight_layout()
    out = r"D:\software\TianXingFC\images\frame_tilt.png"
    fig.savefig(out, dpi=150)
    print(f"\n图已保存：{out}")
except Exception as e:
    print(f"\n（图跳过：{e}）")

print("\n验证完成。")
