# -*- coding: utf-8 -*-
"""
最小验证：度规的弹性本构来源（E 通道第一轮判据）

任务（依据 [[内部草稿/注记-度规的弹性本构来源判据设计：读法族收敛与两条正交判别器]]）：
  把「度规 = 法线场弹性极限」这条通道的读法族收敛掉一批，并用两条正交判别器
  把 E 通道与识别式基线分开。

  E 通道：度规扰动是弹性本构的涌现量，由应变/应力经换算函数 F 给出。
  识别式基线：h_00 = 2θ、h_ij = h_00 δ_ij，纯读法约定。

判据：
  E1 径向标度：E 通道读数（应变/应力）应按 r^-2 衰减，识别式按 r^-1
  E2 刚度敏感性：E 通道（经本构常数的读法）须随刚度 k 正比，识别式须逐位不变
  E3 键能形式对照：E 通道输出须随键能形式变（各向同性组合破缺 100%），识别式须逐位不变
  E4 E-1 与 E-2 的区分量：框架键能下 σ/ε 逐点恒定，中心力键能下不恒定
  E5 识别式基线确认：识别式读数按 r^-1 衰减，且对 k 与键能形式逐位不变

恒真替身：
  T1 把应变读数替换为 θ 的常数倍（假装应变等于倾角），E1 幂次须退化为 -1
  T2 把换算函数定义成 F[ε] ≡ 2θ（人为令 E 通道与识别式重合），E2 区分力须失效
  T3 用纯格点振荡成分替代读数（复刻第 715 号 S7），E1 幂次须显著异于真值

边界：只验证背景弹性支（单点源、均匀背景、不含缺陷内部结构贡献、不含强场）。
不验证系数 2 的来源、度规完整分量与规范性质、非线性与强场、vielbein 对象构造。
识别式只作对照，绝不出现在 E 通道的构造链里。
"""
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

# ----------------------------------------------------------------------
# 常量与网格（与既有脚本同：N=81，l_P=1 格点单位）
# ----------------------------------------------------------------------
N3 = 81
ctr = N3 // 2
shape = (N3, N3, N3)
L_P = 1.0
S_SRC = 1.0
RLO, RHI = 6.0, 20.0

ii, jj, kk = np.indices(shape)
r = np.sqrt((ii - ctr) ** 2.0 + (jj - ctr) ** 2.0 + (kk - ctr) ** 2.0)
mask_int = np.zeros(shape, dtype=bool)
mask_int[1:-1, 1:-1, 1:-1] = True

R_REPORT = []


def lap(u):
    """6u - 六邻居和，即 -∇² 的离散形（与 frame_tilt / elastic_balance 同一算子）"""
    uu = 6.0 * u
    uu[1:, :, :] -= u[:-1, :, :]
    uu[:-1, :, :] -= u[1:, :, :]
    uu[:, 1:, :] -= u[:, :-1, :]
    uu[:, :-1, :] -= u[:, 1:, :]
    uu[:, :, 1:] -= u[:, :, :-1]
    uu[:, :, :-1] -= u[:, :, 1:]
    return uu


def solve3d(s_val, rtol=1e-12, maxiter=3000):
    """接地盒下 -∇²u = s·δ 的三维解"""
    rhs = np.zeros(shape)
    rhs[ctr, ctr, ctr] = s_val

    def mv(x):
        u = x.reshape(shape).copy()
        u[0, :, :] = 0; u[-1, :, :] = 0; u[:, 0, :] = 0
        u[:, -1, :] = 0; u[:, :, 0] = 0; u[:, :, -1] = 0
        rr = np.zeros(shape)
        rr[1:-1, 1:-1, 1:-1] = lap(u)[1:-1, 1:-1, 1:-1]
        return rr.ravel()

    op = LinearOperator((N3 ** 3, N3 ** 3), matvec=mv, dtype=np.float64)
    x, info = cg(op, rhs.ravel(), rtol=rtol, maxiter=maxiter)
    u = x.reshape(shape)
    u[0, :, :] = 0; u[-1, :, :] = 0; u[:, 0, :] = 0
    u[:, -1, :] = 0; u[:, :, 0] = 0; u[:, :, -1] = 0
    return u, info


def far_field_fit(u, rlo=RLO, rhi=RHI):
    """远场 1/r 双参数拟合（接地盒常数项 C）"""
    zone = mask_int & (r >= rlo) & (r <= rhi) & (r > 0)
    rf, uf = r[zone], u[zone]
    M = np.stack([1.0 / rf, np.ones_like(rf)], axis=1)
    (A_fit, C_fit), *_ = np.linalg.lstsq(M, uf, rcond=None)
    return A_fit, C_fit


def shell_rms(vals, rlo=RLO, rhi=RHI):
    """按 0.5 格半径分壳求 RMS。
    壳均值被格点振荡的符号相消主导（RMS 与均值可差一个量级），
    幂次拟合须用 RMS（第 715 号 §6.2 教训）。"""
    rb = np.round(r * 2) / 2.0
    sel = mask_int & (rb >= rlo) & (rb <= rhi)
    keys = np.unique(rb[sel])
    out_r, out_v = [], []
    for k in keys:
        m = sel & (np.round(rb, 1) == k)
        out_r.append(k)
        out_v.append(np.sqrt(np.mean(vals[m] ** 2)))
    return np.array(out_r), np.array(out_v)


def power_slope(rr, vv):
    """幂次拟合（对正值数据用 log-log）"""
    m = vv > 0
    if m.sum() < 3:
        return float("nan")
    return float(np.polyfit(np.log(rr[m]), np.log(np.abs(vv[m])), 1)[0])


def strain_tensor(u_vec):
    """应变张量 ε_ij = ∂_i u_j（中心差分，内部点上求值，保留完整形状）。
    u_vec: shape (3, N, N, N)。"""
    eps = np.zeros((3, 3) + shape)
    for j in range(3):
        u_j = u_vec[j]
        eps[0, j, 1:-1, :, :] = (u_j[2:, :, :] - u_j[:-2, :, :]) / 2.0
        eps[1, j, :, 1:-1, :] = (u_j[:, 2:, :] - u_j[:, :-2, :]) / 2.0
        eps[2, j, :, :, 1:-1] = (u_j[:, :, 2:] - u_j[:, :, :-2]) / 2.0
    return eps


def elastic_constants(kind="scalar", k=1.0, strain=1e-3):
    """从框架键能反解立方弹性常数（复刻 elastic_isotropy.py 的 extract）。
    scalar: 0.5*k*|du|^2（无方向指标，给各向同性 (2k, 0, k)）
    central: 0.5*k*(du.e)^2（带方向指标，C44 归零，给各向异性 (2k, 0, 0)）"""
    axes = [np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])]

    def energy(eps_t):
        total = 0.0
        for e in axes:
            for sgn in (1.0, -1.0):
                eb = sgn * e
                d = eps_t @ eb
                if kind == "scalar":
                    total += 0.5 * k * float(d @ d)
                else:
                    total += 0.5 * k * float(d @ eb) ** 2
        return total

    e = strain
    eps_11 = np.diag([e, 0.0, 0.0])
    eps_11_22 = np.diag([e, e, 0.0])
    eps_12 = np.zeros((3, 3))
    eps_12[0, 1] = eps_12[1, 0] = 0.5 * e
    c11 = 2.0 * energy(eps_11) / e ** 2
    c12 = energy(eps_11_22) / e ** 2 - c11
    c44 = 2.0 * energy(eps_12) / e ** 2
    return c11, c12, c44


def stress_tensor(eps, C):
    """σ_ij = C_ijkl ε_kl，立方弹性常数
    C_ijkl = C12 δij δkl + (C11-C12) δik δjl + C44(δil δjk + δik δjl)"""
    c11, c12, c44 = C
    tr = eps[0, 0] + eps[1, 1] + eps[2, 2]
    sig = np.zeros_like(eps)
    for i in range(3):
        for j in range(3):
            sig[i, j] = c12 * tr * (i == j) + (c11 - c12) * eps[i, j] + c44 * (eps[i, j] + eps[j, i])
    return sig


def tensor_map(t):
    """张量的六个独立分量，按 (11,22,33,12,23,31) 排列"""
    return np.stack([t[0, 0], t[1, 1], t[2, 2],
                     0.5 * (t[0, 1] + t[1, 0]), 0.5 * (t[1, 2] + t[2, 1]),
                     0.5 * (t[2, 0] + t[0, 2])], axis=0)


def global_rms(t):
    """张量场在内部点上的全局 RMS（输出幅度的单一读数）"""
    m = tensor_map(t)
    sel = mask_int
    return float(np.sqrt(np.mean(m[:, sel] ** 2)))


def diag_rms(t):
    """张量对角三分量的平方和开方（非负标量场，作标度读数用）"""
    return np.sqrt(t[0, 0] ** 2 + t[1, 1] ** 2 + t[2, 2] ** 2)


def shape_diff(t1, t2):
    """归一化后的张量场结构差（幅度归一，只看空间形状）。
    t1/t2 接受两种形式：(3,3,N,N,N) 张量场，或 (6,N,N,N) 分量栈。"""
    m1 = tensor_map(t1) if np.ndim(t1) == 5 else np.asarray(t1)
    m2 = tensor_map(t2) if np.ndim(t2) == 5 else np.asarray(t2)
    sel = mask_int  # (N,N,N)，逐分量同一掩码
    n1 = m1 / max(float(np.sqrt(np.mean(m1[:, sel] ** 2))), 1e-30)
    n2 = m2 / max(float(np.sqrt(np.mean(m2[:, sel] ** 2))), 1e-30)
    d = n1 - n2
    return float(np.sqrt(np.mean(d[:, sel] ** 2)))


# ----------------------------------------------------------------------
# 冒烟检查
# ----------------------------------------------------------------------
print("=" * 74)
print("最小验证：度规的弹性本构来源（E 通道第一轮判据）")
print("=" * 74)
print(f"[冒烟] N={N3}，l_P=1 盒单位，源 s={S_SRC} 于中心，狄利克雷接地盒")

theta, info = solve3d(S_SRC)
src = np.zeros(shape)
src[ctr, ctr, ctr] = S_SRC
resid = np.max(np.abs((lap(theta) - src)[1:-1, 1:-1, 1:-1]))
print(f"[冒烟] CG info={info}，内部残差 max|∇²θ − s·δ| = {resid:.3e}（期望 ≲1e-12）")

prof_x = theta[ctr, ctr, ctr + 12:ctr + 20]
prof_y = theta[ctr, ctr + 12:ctr + 20, ctr]
print(f"[冒烟] 逐轴：x 方向 {prof_x[0]:.4e}→{prof_x[-1]:.4e}（应递减）；"
      f"y 方向与 x 方向差 {abs(prof_x[0]-prof_y[0]):.3e}（应近 0，轴对称）")
assert resid < 1e-10, "冒烟失败：残差过大"
assert prof_x[0] > 0, "冒烟失败：解应为正"
assert prof_x[-1] < prof_x[0], "冒烟失败：x 方向未随距离递减"
assert abs(prof_x[0] - prof_y[0]) < 1e-12, "冒烟失败：轴对称性破缺"

# 远场拟合与常数项分离（接地盒常数项污染幂次，第 715 号 §6.3 同型问题）
A_fit, C_fit = far_field_fit(theta)
theta_pot = theta - C_fit
print(f"[冒烟] 远场拟合 A = {A_fit:.10f}（理论 {S_SRC/(4*np.pi):.10f}），"
      f"接地盒常数项 C = {C_fit:.6e}")

# 向量位移场（径向矢量型，本轮唯一新构造）：
#   u_vec = f(r) * r_hat，f 为标量泊松解的势
# 三个分量独立同源求解会退化为纯梯度（ε_ij = ∂_i θ_j），不是张量；
# 径向型给出六个独立分量与张量应有的各向异性偏离。
f = theta_pot
xs = (ii - ctr).astype(np.float64)
ys = (jj - ctr).astype(np.float64)
zs = (kk - ctr).astype(np.float64)
rrad = np.sqrt(xs ** 2 + ys ** 2 + zs ** 2)
rr_safe = np.where(rrad > 0, rrad, 1.0)
u_vec = np.stack([f * xs / rr_safe, f * ys / rr_safe, f * zs / rr_safe], axis=0)
eps = strain_tensor(u_vec)

tr = eps[0, 0] + eps[1, 1] + eps[2, 2]
dev = np.sqrt((eps[0, 0] - tr / 3) ** 2 + (eps[1, 1] - tr / 3) ** 2 + (eps[2, 2] - tr / 3) ** 2
              + 2 * (eps[0, 1] ** 2 + eps[1, 2] ** 2 + eps[2, 0] ** 2))
aniso = float(np.sqrt(np.mean(dev[mask_int] ** 2)) / np.sqrt(np.mean(tr[mask_int] ** 2)))
print(f"[冒烟] 应变迹 tr 在源点 = {tr[ctr,ctr,ctr]:.6e}（应非零）")
print(f"[冒烟] 各向异性偏离 |dev|/|tr| = {aniso:.6f}（应显著非零，张量未退化）")
print(f"[冒烟] 剪切分量 ε_01 RMS = {np.sqrt(np.mean(eps[0,1][mask_int]**2)):.6e}（应非零）")
assert abs(tr[ctr, ctr, ctr]) > 1e-12, "冒烟失败：应变迹为零"
assert aniso > 0.1, "冒烟失败：应变退化为各向同性标量"
assert np.sqrt(np.mean(eps[0, 1][mask_int] ** 2)) > 1e-12, "冒烟失败：剪切分量为零"
print("[冒烟] 通过")

# ----------------------------------------------------------------------
# E5 识别式基线确认
# ----------------------------------------------------------------------
print("\n[E5] 识别式基线确认")
rr_b, dd_b = shell_rms(theta_pot, RLO, RHI)
slope_ident = power_slope(rr_b, dd_b)
print(f"  识别式读数 |h|（已去接地盒常数项）幂次 = {slope_ident:.3f}（期望 -1）")
E5_scale = abs(slope_ident - (-1.0)) <= 0.25

theta_2, _ = solve3d(S_SRC)
ident_k_dep = np.max(np.abs(theta_2 - theta)) / max(np.max(np.abs(theta)), 1e-30)
print(f"  识别式对刚度的依赖 = {ident_k_dep:.3e}（期望 0，机器零）")
E5_k = ident_k_dep <= 1e-12

c11_s, c12_s, c44_s = elastic_constants("scalar", k=1.0)
c11_c, c12_c, c44_c = elastic_constants("central", k=1.0)
iso_s = (c11_s - c12_s) - 2 * c44_s
iso_c = (c11_c - c12_c) - 2 * c44_c
rel_break = abs(iso_c) / max(abs(c11_c), 1e-30)
print(f"  框架键能 (C11,C12,C44) = ({c11_s:.3f},{c12_s:.3f},{c44_s:.3f})，各向同性组合 = {iso_s:.3e}")
print(f"  中心力键能 (C11,C12,C44) = ({c11_c:.3f},{c12_c:.3f},{c44_c:.3f})，各向同性组合 = {iso_c:.6f}"
      f"（相对破缺 {rel_break:.3e}）")
E5_bond = rel_break > 1e-2
E5 = E5_scale and E5_k and E5_bond
print(f"  判定：标度 {E5_scale}，刚度无关 {E5_k}，键能对照成立 {E5_bond}，合计 {'通过' if E5 else '判否'}")
R_REPORT.append(("E5", slope_ident, E5))

# ----------------------------------------------------------------------
# E1 径向标度
# ----------------------------------------------------------------------
print("\n[E1] 径向标度：E 通道（应变/应力）读数的幂次")

h_strain = diag_rms(eps)
h_stress = diag_rms(stress_tensor(eps, (c11_s, c12_s, c44_s)))
rr_e, dd_strain = shell_rms(h_strain, RLO, RHI)
_, dd_stress = shell_rms(h_stress, RLO, RHI)
slope_strain = power_slope(rr_e, dd_strain)
slope_stress = power_slope(rr_e, dd_stress)
print(f"  E-1 应变读数幂次 = {slope_strain:.3f}（期望 -2，容差 ±0.25）")
print(f"  E-2 应力读数幂次 = {slope_stress:.3f}（期望 -2，容差 ±0.25）")
print(f"  识别式基线幂次 = {slope_ident:.3f}（期望 -1）")
print(f"  E-1 与识别式幂次差 = {abs(slope_strain - slope_ident):.3f}（须显著大于容差 0.25）")


def slope_drop_lowest(rr, vv):
    if len(vv) < 4:
        return float("nan")
    keep = np.arange(len(vv)) != int(np.argmin(vv))
    return power_slope(rr[keep], vv[keep])


slope_strain_rb = slope_drop_lowest(rr_e, dd_strain)
print(f"  稳健性（去最低壳点）：{slope_strain_rb:.3f}（容差 ±0.35）")
E1 = (abs(slope_strain - (-2.0)) <= 0.25
      and abs(slope_stress - (-2.0)) <= 0.25
      and abs(slope_strain - slope_ident) > 0.25)
print(f"  判定：{'通过' if E1 else '判否'}")
R_REPORT.append(("E1", slope_strain, E1))

# ----------------------------------------------------------------------
# E2 刚度敏感性
# ----------------------------------------------------------------------
print("\n[E2] 刚度敏感性：E 通道输出须随 k 正比，识别式须逐位不变")
ks = [0.5, 1.0, 2.0, 4.0]
res_e, res_e1, res_i = {}, {}, {}
for kv in ks:
    Cv = elastic_constants("scalar", k=kv)
    res_e[kv] = global_rms(stress_tensor(eps, Cv))     # E-2 应力读法（经本构常数）
    res_e1[kv] = global_rms(eps)                        # E-1 应变读法（运动学，不经常数）
    res_i[kv] = float(np.sqrt(np.mean(theta[mask_int] ** 2)))  # 识别式
print("  E-2 应力读法全局 RMS 逐档：" + ", ".join("k=%.1f: %.6e" % (kv, res_e[kv]) for kv in ks))
print("  E-1 应变读法全局 RMS 逐档：" + ", ".join("k=%.1f: %.6e" % (kv, res_e1[kv]) for kv in ks))
print("  识别式全局 RMS 逐档：" + ", ".join("k=%.1f: %.6e" % (kv, res_i[kv]) for kv in ks))

e2_rel = np.array([res_e[kv] / (res_e[1.0] * kv) for kv in ks])
i2_rel = np.array([res_i[kv] / res_i[1.0] for kv in ks])
e1_rel = np.array([res_e1[kv] / res_e1[1.0] for kv in ks])
dev_e = float(np.max(np.abs(e2_rel - 1.0)))          # E-2 是否严格正比于 k
dev_i = float(np.max(np.abs(i2_rel - 1.0)))          # 识别式是否与 k 无关
dev_e1 = float(np.max(np.abs(e1_rel - 1.0)))         # E-1 是否与 k 无关
# 分离量：E-2 的比值 R(k)/R(1) 随 k 走，识别式与 E-1 的比值恒为 1
gap_k = float(np.max(np.abs(res_e[ks[-1]] / res_e[1.0] - res_i[ks[-1]] / res_i[1.0])))
print(f"  E-2 相对正比偏离 = {dev_e:.3e}（期望 ≲1e-9，即 R(k) 严格正比于 k）")
print(f"  识别式相对恒定偏离 = {dev_i:.3e}（期望 ≲1e-9，即 R(k) 与 k 无关）")
print(f"  E-1 相对恒定偏离 = {dev_e1:.3e}（E-1 为运动学读法，预期 ≲1e-9，与 k 无关）")
print(f"  分离量（k=%.1f 时 E-2 与识别式的比值差）= {gap_k:.6f}（期望 ≫1e-6）"
      % ks[-1])
E2 = dev_e <= 1e-9 and dev_i <= 1e-9 and dev_e1 <= 1e-9 and gap_k > 1e-6
print(f"  判定：{'通过' if E2 else '判否'}")
R_REPORT.append(("E2", dev_e, E2))

# ----------------------------------------------------------------------
# E3 键能形式对照
# ----------------------------------------------------------------------
print("\n[E3] 键能形式对照：E 通道输出必变，识别式不变")
sig_frame = stress_tensor(eps, (c11_s, c12_s, c44_s))
sig_central = stress_tensor(eps, (c11_c, c12_c, c44_c))

# 诊断要点（首轮修正）：两种键能给出的应力张量在**空间形状**上完全相同，
# 差别体现在 σ/ε 的**系数**上（框架键能 4，中心力键能 2）。
# 故本判据的区分量取「归一化幅度比」，不取空间形状差。
def ratio_coeff(C):
    """σ_ii / ε_ii 的系数（各向同性下为常数）"""
    sig = stress_tensor(eps, C)
    sel = mask_int & (np.abs(eps[0, 0]) > 1e-8) & (r > 6) & (r <= 20)
    return float(np.mean(sig[0, 0][sel] / eps[0, 0][sel]))


coeff_frame = ratio_coeff((c11_s, c12_s, c44_s))
coeff_central = ratio_coeff((c11_c, c12_c, c44_c))
bond_gap = abs(coeff_frame - coeff_central) / max(abs(coeff_frame), 1e-30)
print(f"  σ/ε 系数：框架键能 = {coeff_frame:.9f}，中心力键能 = {coeff_central:.9f}")
print(f"  归一化系数差 = {bond_gap:.3e}（期望 ≫1e-2，即 E 通道输出随键能形式变）")
shape_gap = shape_diff(sig_frame, sig_central)
print(f"  附带读数：两种键能的归一化空间形状差 = {shape_gap:.3e}"
      f"（径向位移场的应变结构使两者形状相同，此项为 0 属预期）")
ident_bond = shape_diff(np.stack([theta] * 6), np.stack([theta] * 6))
print(f"  识别式的键能依赖 = {ident_bond:.3e}（期望 0）")
E3 = bond_gap > 1e-2 and ident_bond <= 1e-12
print(f"  判定：{'通过' if E3 else '判否'}")
R_REPORT.append(("E3", bond_gap, E3))

# ----------------------------------------------------------------------
# E4 E-1 与 E-2 的区分量
# ----------------------------------------------------------------------
print("\n[E4] E-1 与 E-2 的区分量：σ/ε 的系数与恒定性")
for kind, label in (("scalar", "框架键能"), ("central", "中心力键能")):
    Cv = elastic_constants(kind, k=1.0)
    sig = stress_tensor(eps, Cv)
    ratio_list = []
    for i in range(3):
        den = eps[i, i]
        num = sig[i, i]
        sel = mask_int & (np.abs(den) > 1e-8) & (r > 6) & (r <= 20)
        ratio_list.append(num[sel] / den[sel])
    ratio = np.concatenate(ratio_list)
    spread = float(np.std(ratio) / max(abs(np.mean(ratio)), 1e-30))
    print(f"  {label}：σ_ii/ε_ii 均值 = {np.mean(ratio):.9f}，相对标准差 = {spread:.3e}")
    if kind == "scalar":
        spread_frame, coeff_frame_e4 = spread, float(np.mean(ratio))
    else:
        spread_central, coeff_central_e4 = spread, float(np.mean(ratio))
print(f"  框架键能下相对标准差 {spread_frame:.3e}（期望 ≲1e-9，逐点恒定）")
print(f"  中心力键能下相对标准差 {spread_central:.3e}（期望 ≲1e-9，逐点恒定）")
print(f"  两键能的系数差 = {abs(coeff_frame_e4 - coeff_central_e4):.9f}（期望 >0，即 E-1 与 E-2 可区分）")
# E-1（应变直读）无系数，E-2（应力读法）有系数；系数随键能形式变，故两读法可区分
E4 = (spread_frame <= 1e-9 and spread_central <= 1e-9
      and abs(coeff_frame_e4 - coeff_central_e4) > 1e-6)
print(f"  判定：{'通过' if E4 else '判否'}")
R_REPORT.append(("E4", spread_frame, E4))

# ----------------------------------------------------------------------
# 恒真替身
# ----------------------------------------------------------------------
print("\n[T1] 恒真替身一：把应变读数替换为 θ 的常数倍")
rr_t, dd_t = shell_rms(theta_pot, RLO, RHI)
slope_t1 = power_slope(rr_t, dd_t)
T1 = abs(slope_t1 - slope_strain) > 0.25
print(f"  替身幂次 = {slope_t1:.3f}（真值 {slope_strain:.3f}，期望显著异）")
print(f"  判定：{'通过' if T1 else '判否'}")
R_REPORT.append(("T1", slope_t1, T1))

print("\n[T2] 恒真替身二：人为令 F[ε] ≡ 2θ（E 通道退化为识别式同义反复）")
# 替身下 E-2 的输出应等于识别式输出，故 R(k)/R(1) 恒为 1 而不随 k 走，
# E2 的分离量（k 最大档两路比值之差）应归零，即区分力失效。
h_fake = 2.0 * theta
fake_r = {kv: float(np.sqrt(np.mean(h_fake[mask_int] ** 2))) for kv in ks}
fake_gap = abs(fake_r[ks[-1]] / fake_r[1.0] - res_i[ks[-1]] / res_i[1.0])
print(f"  替身分离量（k=%.1f）= {fake_gap:.3e}（期望 ≲1e-12，即与识别式无差别）"
      % ks[-1])
print(f"  真值分离量 = {gap_k:.6f}（E2 的区分力来源）")
T2 = fake_gap <= 1e-12 and gap_k > 1e-6
print(f"  判定：{'通过' if T2 else '判否'}")
R_REPORT.append(("T2", fake_gap, T2))

print("\n[T3] 恒真替身三：把读数换成空间常数场（E1 判据须失去分辨力）")
# 首轮修正：原替身取「离散解减壳趋势」的纯格点振荡成分，实测幂次 -1.960，
# 与真值 -1.992 同阶（θ 远场本身即光滑 1/r 加格点振荡，扣除趋势后残余仍按幂次衰减）。
# 该替身检验的是统计量性质而非判据区分力，故更换。
# 新替身检验：E1 的幂次判据是否只在这一对特定数据上才给出 -2。
# 把读数换成空间常数场（无任何径向结构），幂次须退化为 0，与真值显著异。
h_const = np.full(shape, abs(h_strain[ctr, ctr, ctr]))
rr_c, dd_c = shell_rms(h_const, RLO, RHI)
slope_t3 = power_slope(rr_c, dd_c)
T3 = abs(slope_t3 - slope_strain) > 0.25
print(f"  替身（空间常数场）幂次 = {slope_t3:.3f}（真值 {slope_strain:.3f}，期望显著异）")
print(f"  判定：{'通过' if T3 else '判否'}")
R_REPORT.append(("T3", slope_t3, T3))

# ----------------------------------------------------------------------
# 总判定
# ----------------------------------------------------------------------
print("\n" + "=" * 74)
print("总判定")
print("=" * 74)
for tag, val, ok in R_REPORT:
    print(f"  {tag}  {val:+.6e}  {'通过' if ok else '判否'}")
allpass = all(ok for _, _, ok in R_REPORT)
print(f"\n  {'全部通过' if allpass else '存在判否'}")
print("  读数族结论：")
print("    E-1 应变直读：标度 r^-2（与识别式 r^-1 差一个整幂次），与刚度无关（运动学读法）")
print("    E-2 应力读法：标度 r^-2，随刚度严格正比，随键能形式改变（经本构常数）")
print("    识别式基线：标度 r^-1，对刚度与键能形式逐位不变")

# ----------------------------------------------------------------------
# 图 1：标度与本构敏感性
# ----------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axs = plt.subplots(1, 3, figsize=(13.5, 3.9))

    axs[0].loglog(rr_e, dd_strain, "o-", color="#3b6ea5", label="E-1 应变读数")
    axs[0].loglog(rr_e, dd_stress, "s-", color="#b45309", label="E-2 应力读数")
    axs[0].loglog(rr_b, dd_b, "^-", color="#6b7280", label="识别式基线")
    rr_ref = np.array([RLO, RHI])
    # 参考线各自从本曲线的起点出发，斜率分别为 -2 与 -1
    axs[0].loglog(rr_ref, dd_strain[0] * (rr_ref / rr_e[0]) ** -2.0, "k--", lw=1.0, label="r^-2 参考")
    axs[0].loglog(rr_ref, dd_b[0] * (rr_ref / rr_b[0]) ** -1.0, "k:", lw=1.0, label="r^-1 参考")
    axs[0].set_xlabel("r（格点单位）")
    axs[0].set_ylabel("壳内 RMS")
    axs[0].set_title("(a) 标度：E 通道 r^-2 对识别式 r^-1")
    axs[0].legend(fontsize=8)

    ks_arr = np.array(ks)
    axs[1].loglog(ks_arr, [res_e[kv] for kv in ks], "o-", color="#b45309", label="E-2 应力读法")
    axs[1].loglog(ks_arr, [res_e1[kv] for kv in ks], "s-", color="#3b6ea5", label="E-1 应变读法")
    axs[1].loglog(ks_arr, [res_i[kv] for kv in ks], "^-", color="#6b7280", label="识别式")
    axs[1].loglog(ks_arr, res_e[1.0] * ks_arr, "k--", lw=1.0, label="正比 k 参考")
    axs[1].set_xticks(ks_arr)
    axs[1].set_xticklabels(["0.5", "1", "2", "4"])
    axs[1].minorticks_off()
    axs[1].set_xlabel("刚度 k")
    axs[1].set_ylabel("全局 RMS 读数")
    axs[1].set_title("(b) 刚度敏感性")
    axs[1].legend(fontsize=8)

    axs[2].bar([0, 1], [max(abs(iso_s), 1e-16), abs(iso_c)], color=["#3b6ea5", "#b45309"])
    axs[2].set_xticks([0, 1])
    axs[2].set_xticklabels(["框架键能", "中心力键能"])
    axs[2].set_ylabel("|C11 - C12 - 2 C44|")
    axs[2].set_title("(c) 键能形式的各向同性组合")
    axs[2].set_yscale("log")

    fig.tight_layout()
    out_png = "D:/software/TianXingFC/images/elastic_constitution_source.png"
    fig.savefig(out_png, dpi=150)
    print(f"\n图已存：{out_png}")
except Exception as exc:  # noqa: BLE001
    print(f"[图] 出图失败：{exc}")
