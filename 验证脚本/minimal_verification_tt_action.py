# -*- coding: utf-8 -*-
"""G4 第二半的第一块（L4）：TT 扇区二次作用量与无质量自旋 2 标准形式的匹配。

判据（先立判据，再实现；每条都要求可证伪）：
  L4-1 作用量到运动方程：对离散作用量做数值泛函变分，变分梯度须等于解析运动方程算子
       （相对偏差 <= 1e-8）；该算子作用在平面波上为零给出离散色散
       sin(omega dt/2) = c dt sin(k/2)；把连续色散 omega = c k 代入，Courant 数
       c dt = 1 时残差恰为零（步长选取的巧合，不是普适结论），c dt = 0.8 时残差非零，
       连续极限下偏差的首阶系数须为 (c^2 dt^2 - 1)/24 且随 k 收缩。
  L4-2 投影结构与自由度计数：TT 投影幂等、正交、自伴，像为横向无迹；六维对称张量空间上秩为 2；
       余核维数为 4，且余核恰由迹方向、沿 k 的纵向-纵向无迹方向、纵向-横向方向张成。
  L4-3 单参数双极限：静态腿与波动腿各自独立测同一个 c_s。把 c_s 从 1.0 换成 0.8 重跑，
       静态读数须按 c_s^{-2} 变化，波动腿实测波速须随 c_s 变化；任一条腿自带自己的常数
       都会让该检验失败。静态腿用离散高斯定律核对源强归一化（与边界数据无关）；
       波动腿取零初速高斯波包（前后分裂），并在脉冲完全进入观测窗后取样。
  L4-4 归一化与单位链：A * kappa = 1/4 是线性化标准作用量与标准耦合写法之间的代数恒等式，
       标注为恒等式，不作读数；单位链恒等式 l_P m_P c = hbar、l_P / m_P = G / c^2、
       m_P / l_P = c^2 / G 的数值偏差；锚定重建：由静态腿的数值读数经单位换算得 SI 势幅，
       与用 SI 常数独立写出的闭式对照，偏差应停在格点离散量级。

单位：格距 l_P = 1、时间以 dt = 1 计。确定性，无随机数。
"""

import numpy as np

CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))


print("=" * 78)
print("L4-1 作用量到运动方程：数值泛函变分、离散色散与连续极限")
print("=" * 78)
A_COEF = 1.0
C_UNIT = 1.0
DT = 1.0
NX, NT = 8, 8


def action(h):
    kin = np.sum(((np.roll(h, -1, axis=1) - h) / DT) ** 2)
    pot = np.sum((np.roll(h, -1, axis=0) - h) ** 2)
    return A_COEF * DT * (kin / C_UNIT ** 2 - pot)


def grad_analytic(h):
    return ((2.0 * A_COEF / (C_UNIT ** 2 * DT))
            * (2.0 * h - np.roll(h, 1, axis=1) - np.roll(h, -1, axis=1))
            + 2.0 * A_COEF * DT
            * (np.roll(h, -1, axis=0) - 2.0 * h + np.roll(h, 1, axis=0)))


jx = np.arange(NX)[:, None]
nt = np.arange(NT)[None, :]
h_field = (np.cos(2.0 * np.pi * jx / 3.0 + 0.7) * np.sin(2.0 * np.pi * nt / 5.0 + 1.3)
           + 0.4 * np.cos(2.0 * np.pi * nt / 4.0))
EPS = 1e-5
g_num = np.zeros_like(h_field)
for a in range(NX):
    for b in range(NT):
        hp = h_field.copy()
        hp[a, b] += EPS
        hm = h_field.copy()
        hm[a, b] -= EPS
        g_num[a, b] = (action(hp) - action(hm)) / (2.0 * EPS)
g_ana = grad_analytic(h_field)
rel_grad = float(np.max(np.abs(g_num - g_ana)) / np.max(np.abs(g_ana)))
print("  数值泛函变分 vs 解析运动方程算子：最大相对偏差 = %.3e（期望 <= 1e-8）" % rel_grad)
check("L4-1a", rel_grad <= 1e-8, "变分梯度与解析算子在非平面波构型上一致")


def eig(k, w, cc):
    return ((4.0 * A_COEF / (cc ** 2 * DT)) * np.sin(w * DT / 2.0) ** 2
            - 4.0 * A_COEF * DT * np.sin(k / 2.0) ** 2)


worst_exact = 0.0
for cdt in (1.0, 0.8):
    for k in (0.1, 0.3, 0.5, 1.0):
        w = (2.0 / DT) * np.arcsin(cdt * DT * np.sin(k / 2.0))
        scale = 4.0 * A_COEF * DT * np.sin(k / 2.0) ** 2
        worst_exact = max(worst_exact, abs(eig(k, w, cdt)) / scale)
print("  离散色散 sin(omega dt/2) = c dt sin(k/2) 处算子最大相对残差 = %.3e（期望机器精度）"
      % worst_exact)
check("L4-1b", worst_exact <= 1e-14, "离散色散零点上算子为零（两种 Courant 数各四档 k）")

cdt1_worst = 0.0
for k in (0.1, 0.5, 1.0):
    scale = 4.0 * A_COEF * DT * np.sin(k / 2.0) ** 2
    cdt1_worst = max(cdt1_worst, abs(eig(k, C_UNIT * k, C_UNIT)) / scale)
res_cdt = abs(eig(1.0, 0.8, 0.8)) / (4.0 * A_COEF * DT * np.sin(0.5) ** 2)
print("  连续色散 omega = c k 代入：Courant 数 1 时最大相对残差 = %.3e；Courant 数 0.8、"
      "k = 1.0 时 = %.3e" % (cdt1_worst, res_cdt))
print("    说明：c dt = 1 时 sin(omega dt/2) = sin(k/2) 恒成立，两条色散在此单位选择下重合；")
print("          这不是普适恒等，Courant 数一离开 1 残差即出现。")
check("L4-1d", res_cdt > 1e-3, "连续色散在 Courant 数不为 1 时被离散色散否决")

cdt = 0.8
coef_limit = (cdt ** 2 - 1.0) / 24.0
print("  连续极限首阶系数（解析）(c^2 dt^2 - 1)/24 = %.6e" % coef_limit)
dev_ratio = []
for k in (0.02, 0.04, 0.06):
    w = (2.0 / DT) * np.arcsin(cdt * DT * np.sin(k / 2.0))
    delta = w / (cdt * k) - 1.0
    dev_ratio.append(abs(delta / k ** 2 / coef_limit - 1.0))
    print("    k = %.2f：omega/(c k) - 1 = %+.6e，除以 k^2 = %+.6e（对照 %+.6e）"
          % (k, delta, delta / k ** 2, coef_limit))
check("L4-1c", max(dev_ratio) <= 5e-3 and dev_ratio[0] <= dev_ratio[2],
      "首阶系数被数值复现（最大相对偏差 %.1e，且 k 越小越贴近）" % max(dev_ratio))

print()
print("=" * 78)
print("L4-2 投影结构、自由度计数与余核分解")
print("=" * 78)
k_hat = np.array([0.267, 0.535, 0.802])
k_hat = k_hat / np.linalg.norm(k_hat)
d_perp = np.eye(3) - np.outer(k_hat, k_hat)


def tt(S):
    return d_perp @ S @ d_perp - 0.5 * d_perp * np.trace(d_perp @ S)


IDX = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]


def sym_basis(a, b):
    M = np.zeros((3, 3))
    M[a, b] = 1.0
    M[b, a] = 1.0
    return M


def coeffs(M):
    return np.array([M[0, 0], M[1, 1], M[2, 2], M[0, 1], M[0, 2], M[1, 2]])


def mat_from_coeffs(cv):
    M = np.zeros((3, 3))
    for val, (a, b) in zip(cv, IDX):
        M[a, b] = val
        M[b, a] = val
    return M


Pmat = np.array([coeffs(tt(sym_basis(a, b))) for (a, b) in IDX]).T
gram = np.diag([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
err_idem = float(np.max(np.abs(Pmat @ Pmat - Pmat)))
S_rand = mat_from_coeffs(np.array([0.31, -0.72, 0.44, 0.58, -0.19, 0.83]))
err_orth = float(np.linalg.norm(tt(S_rand - tt(S_rand))))
err_self = float(np.max(np.abs(gram @ Pmat - (gram @ Pmat).T)))
err_traceless = abs(float(np.trace(tt(S_rand))))
err_transverse = float(np.linalg.norm(k_hat @ tt(S_rand)))
rank_tt = int(np.linalg.matrix_rank(Pmat, tol=1e-10))
print("  幂等 |P^2 - P| = %.3e；正交补 |tt(S - tt(S))| = %.3e；自伴 |G P - (G P)^T| = %.3e"
      % (err_idem, err_orth, err_self))
print("  像为横向无迹：|tr tt(S)| = %.3e，|k^i tt(S)_ij| = %.3e" % (err_traceless, err_transverse))
print("  TT 投影的秩 = %d（期望 2）；余核维数 = %d（期望 4）" % (rank_tt, 6 - rank_tt))
check("L4-2a", err_idem <= 1e-14 and err_orth <= 1e-14 and err_self <= 1e-14
      and err_traceless <= 1e-14 and err_transverse <= 1e-14,
      "投影幂等、自伴、像为横向无迹")

v1 = np.array([0.0, -k_hat[2], k_hat[1]])
v1 = v1 / np.linalg.norm(v1)
v2 = np.cross(k_hat, v1)
ker_vecs = [np.eye(3), np.outer(k_hat, k_hat) - np.eye(3) / 3.0,
            np.outer(k_hat, v1) + np.outer(v1, k_hat),
            np.outer(k_hat, v2) + np.outer(v2, k_hat)]
Kmat = np.array([coeffs(M) for M in ker_vecs])
res_named = float(np.max(np.abs(Kmat @ Pmat.T)))
rank_named = int(np.linalg.matrix_rank(Kmat, tol=1e-10))
basis_ker = np.linalg.svd(Pmat)[2][rank_tt:]
Q_ker = basis_ker.T @ basis_ker
Q_named = np.linalg.qr(Kmat.T)[0]
Q_named = Q_named @ Q_named.T
res_span = float(np.max(np.abs(Q_ker - Q_named)))
print("  四个具名方向（迹、k 方向无迹纵向-纵向、两条纵向-横向）：秩 %d，被投影零化至 %.3e，"
      "与余核投影矩阵之差 %.3e" % (rank_named, res_named, res_span))
check("L4-2b", rank_tt == 2 and rank_named == 4 and res_named <= 1e-14 and res_span <= 1e-12,
      "传播子空间维数 2，约束子空间 4 维且由具名方向张成")
print("  计数说明：独立自由度 2 取自 TT 投影的秩；正定动能型限制到任一二维子空间都给秩 2，")
print("            该读数不含信息，故不用作计数依据（判据 U 的约束结构承担其余）。")

print()
print("=" * 78)
print("L4-3 单参数双极限：静态腿与波动腿同取一个 c_s")
print("=" * 78)
MBOX = 41
CEN = MBOX // 2
gx = np.arange(MBOX)
X, Y, Z = np.meshgrid(gx, gx, gx, indexing="ij")
RAD = np.sqrt((X - CEN) ** 2 + (Y - CEN) ** 2 + (Z - CEN) ** 2)
MASK = np.zeros_like(RAD, dtype=bool)
MASK[1:-1, 1:-1, 1:-1] = True
SRC = np.zeros_like(RAD)
SRC[CEN, CEN, CEN] = 1.0
COORDS = (X - CEN, Y - CEN, Z - CEN)


def laplacian(u):
    out = -6.0 * u
    out[1:, :, :] += u[:-1, :, :]
    out[:-1, :, :] += u[1:, :, :]
    out[:, 1:, :] += u[:, :-1, :]
    out[:, :-1, :] += u[:, 1:, :]
    out[:, :, 1:] += u[:, :, :-1]
    out[:, :, :-1] += u[:, :, 1:]
    return out


def solve_static(cs, tol=1e-12, itmax=6000):
    rad_safe = np.where(RAD == 0.0, 1.0, RAD)
    th = np.where(MASK, 0.0, 1.0 / (4.0 * np.pi * cs ** 2 * rad_safe))
    res = -(SRC + cs ** 2 * laplacian(th))
    res[~MASK] = 0.0
    pk = res.copy()
    rs = float(np.sum(res * res))
    iters = 0
    for it in range(itmax):
        iters = it + 1
        ap = cs ** 2 * laplacian(pk)
        ap[~MASK] = 0.0
        alpha = rs / float(np.sum(pk * ap))
        th = th + np.where(MASK, alpha * pk, 0.0)
        res = res - alpha * ap
        res[~MASK] = 0.0
        rs_new = float(np.sum(res * res))
        if np.sqrt(rs_new) < tol:
            break
        pk = res + (rs_new / rs) * pk
        rs = rs_new
    return th, iters, np.sqrt(rs)


def gauss_flux(th, half):
    flux = 0.0
    for axis in range(3):
        others = [COORDS[b] for b in range(3) if b != axis]
        inner = (np.abs(others[0]) <= half) & (np.abs(others[1]) <= half)
        for sgn in (1, -1):
            plate = inner & (COORDS[axis] == sgn * half)
            flux += float(np.sum((np.roll(th, -sgn, axis=axis) - th)[plate]))
    return flux


falloffs = {}
for cs in (1.0, 0.8):
    th, iters, res = solve_static(cs)
    flux = gauss_flux(th, 8)
    print("  c_s = %.1f：CG %d 步，残差 %.2e；离散高斯定律通量 = %.9f（对照 -s/c_s^2 = %.9f）"
          % (cs, iters, res, flux, -1.0 / cs ** 2))
    check("L4-3a(c_s=%.1f)" % cs, abs(flux * cs ** 2 + 1.0) <= 1e-8,
          "源强归一化由通量核对到 %.1e" % abs(flux * cs ** 2 + 1.0))
    val = {}
    for r0 in (8, 12, 16, 24):
        shell = (RAD > r0 - 0.5) & (RAD < r0 + 0.5)
        val[r0] = float(np.mean((th * RAD)[shell]))
    falloffs[cs] = val
    print("    theta*r：r=8 %.6f，r=12 %.6f，r=16 %.6f，r=24 %.6f；解析 s/(4 pi c_s^2) = %.6f"
          % (val[8], val[12], val[16], val[24], 1.0 / (4.0 * np.pi * cs ** 2)))

target = 1.0 / (4.0 * np.pi)
dev16 = abs(falloffs[1.0][16] - target) / target
dev24 = abs(falloffs[1.0][24] - target) / target
check("L4-3b", dev24 <= dev16 and dev16 <= 5e-3,
      "r=16 相对解析值偏差 %.2e，r=24 偏差 %.2e（随半径下降）" % (dev16, dev24))
ratio = falloffs[0.8][16] / falloffs[1.0][16]
print("  双参数：theta*r(0.8)/theta*r(1.0) = %.6f（对照 c_s^{-2} 之比 %.6f）"
      % (ratio, 1.0 / 0.8 ** 2))
check("L4-3c", abs(ratio - 1.0 / 0.8 ** 2) <= 1e-3, "静态读数按 c_s^{-2} 标度")


def wave_speed(cs, n=3001, t_end=400, half_width=12.0, x0=800.0, t_start=160):
    xs = np.arange(n, dtype=float)
    u = np.exp(-((xs - x0) ** 2) / (2.0 * half_width ** 2))
    up = u.copy()
    ts, peaks = [], []
    for t in range(1, t_end + 1):
        lap_u = np.roll(u, 1) + np.roll(u, -1) - 2.0 * u
        un = 2.0 * u - up + (cs * DT) ** 2 * lap_u
        up, u = u, un
        if t % 20 == 0 and t >= t_start:
            ts.append(t * DT)
            peaks.append(float(np.argmax(np.where(xs > x0 + 120.0, u, -np.inf))))
    return float(np.polyfit(np.array(ts), np.array(peaks), 1)[0])


speeds = {}
for cs in (1.0, 0.8):
    sp = wave_speed(cs)
    speeds[cs] = sp
    print("  波动腿（零初速高斯波包分裂，脉冲进入观测窗后取样）c_s = %.1f：实测波速 = %.6f，"
          "相对偏差 = %.2e" % (cs, sp, abs(sp / cs - 1.0)))
print("  两档波速之比 = %.6f（对照 c_s 之比 0.800000）" % (speeds[0.8] / speeds[1.0]))
check("L4-3d", all(abs(speeds[cs] / cs - 1.0) <= 1e-3 for cs in (1.0, 0.8))
      and abs(speeds[0.8] / speeds[1.0] - 0.8) <= 1e-3,
      "波动腿实测波速跟随 c_s 取值（%.6f、%.6f）" % (speeds[1.0], speeds[0.8]))
print("  单参数结论：两条腿在同一 c_s 上同时被锁定，且都随该参数变化。")

print()
print("=" * 78)
print("L4-4 归一化关系与单位链")
print("=" * 78)
G_N = 6.67430e-11
C_LIGHT = 299792458.0
HBAR = 1.054571817e-34
A_GR = C_LIGHT ** 4 / (64.0 * np.pi * G_N)
KAPPA_GR = 16.0 * np.pi * G_N / C_LIGHT ** 4
print("  线性化标准的对应量（代数恒等式，不作读数）：")
print("    A = c^4/(64 pi G) = %.6e；kappa = 16 pi G/c^4 = %.6e；A * kappa = %.15f"
      % (A_GR, KAPPA_GR, A_GR * KAPPA_GR))

L_PLANCK = np.sqrt(HBAR * G_N / C_LIGHT ** 3)
M_PLANCK = np.sqrt(HBAR * C_LIGHT / G_N)
T_PLANCK = L_PLANCK / C_LIGHT
ident = [("l_P m_P c / hbar", L_PLANCK * M_PLANCK * C_LIGHT / HBAR),
         ("(l_P/m_P)/(G/c^2)", (L_PLANCK / M_PLANCK) / (G_N / C_LIGHT ** 2)),
         ("(m_P/l_P)/(c^2/G)", (M_PLANCK / L_PLANCK) / (C_LIGHT ** 2 / G_N)),
         ("(l_P/t_P)/c", (L_PLANCK / T_PLANCK) / C_LIGHT)]
worst_ident = 0.0
for name, val in ident:
    worst_ident = max(worst_ident, abs(val - 1.0))
    print("    单位链恒等式 %s = %.15f（偏差 %.2e）" % (name, val, abs(val - 1.0)))
check("L4-4a", worst_ident <= 1e-14, "单位链四条恒等式到浮点精度")

R_LAT = 16
theta_r_meas = falloffs[1.0][R_LAT]
h_measured = L_PLANCK * theta_r_meas / float(R_LAT)
m_unit = M_PLANCK / (4.0 * np.pi)
r_si = R_LAT * L_PLANCK
h_closed = L_PLANCK ** 2 * m_unit / (M_PLANCK * r_si)
h_schwarz = L_PLANCK * (2.0 * G_N * m_unit / C_LIGHT ** 2) / (2.0 * r_si)
print("  锚定重建（两条路径的输入不共享）：")
print("    路径甲（格点静态解）：theta*r = %.6f，取 r = %d 格，h = l_P theta = %.6e m"
      % (theta_r_meas, R_LAT, h_measured))
print("    路径乙（SI 闭式）：s = 1 对应 m = m_P/(4 pi) = %.6e kg，r = %.6e m，"
      "h = l_P^2 m/(m_P r) = %.6e m" % (m_unit, r_si, h_closed))
print("            相对偏差 = %.2e；史瓦西形式 h = l_P r_s/(2r) = %.6e m（恒等式，与闭式同值）"
      % (abs(h_measured / h_closed - 1.0), h_schwarz))
check("L4-4b", abs(h_measured / h_closed - 1.0) <= 5e-3,
      "数值读数经单位链换算与 SI 闭式一致（偏差 %.1e，停在离散量级）"
      % abs(h_measured / h_closed - 1.0))
print("  结论：静态腿的源强与波动腿的系数出自同一条方程与同一组格点系数；")
print("        波扇区不需要第二套归一化入口，锚定关系由单位链与离散 Green 函数共同承担。")

print()
n_pass = sum(1 for _, ok in CHECKS if ok)
print("=" * 78)
print("判据汇总：%d/%d 通过" % (n_pass, len(CHECKS)))
for tag, ok in CHECKS:
    print("  %-16s %s" % (tag, "PASS" if ok else "FAIL"))
print("=" * 78)
