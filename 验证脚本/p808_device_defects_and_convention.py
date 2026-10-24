# -*- coding: utf-8 -*-
"""第 808 号：三处装置级缺陷的定位与 3+1 关系式的口径归属。

背景
----
第 801 至 804 号在 G_ij 侧反复读到一处不为零的差额，并给出两条定量结论：
差额的主体落在 N 随位置变的项上（801 号），全部落在 gamma 的径向依赖上
（803 号），且与 gamma 径向曲率严格正比、比例是跨档近似的无量纲常数（804 号）。

引入与两侧实现无关的坐标解析闭式作第三方裁判后，定位到三处装置级缺陷：
  (a) 3D 侧 Christoffel 求导下标错位。装置里存的是 E[(i,j,k)] = d_k gamma_ij，
      却按下标在首位取用，相当于把 d_k gamma_ij 当成 d_i gamma_jk。正确形式是
      Gamma^lam_{mu nu} = 0.5 gamma^{lam sig}(E[(sig,nu,mu)] + E[(sig,mu,nu)]
      - E[(mu,nu,sig)])。
  (b) 4D 侧 Ricci 的 d_nu Gamma^rho_{rho mu} 项符号错。装置里该分项已带负号，
      返回式又写成 a1 - a2，净效果成了加号。
  (c) 4D 侧 Ricci 标量未按逆度规收缩。装置里写成 R = Ric00 + sum Ricii，
      标准形式是 R = g^{mu nu} R_{mu nu}。

判据
----
P808-1a 构型判据：三处缺陷之一（3D Christoffel 下标错位）可测。
P808-1b 构型判据：三处缺陷之二（4D Ricci 符号）与之三（标量收缩）可测。
P808-2  构型判据：订正后 4D 侧与 3D 侧在 Christoffel 与 Ricci 上相容。
P808-2b 构型判据：订正后相容性不随步长细化而漂移（收敛检验）。
P808-3  构型判据：平凡档（N 与 G 都为常数）两侧机器零。
P808-4  构型判据：801 至 804 号的定量对象在订正后消失。
P808-5  构型判据：库内登记式 G_00 = (1/2) N^2 R^(3) 的口径归属。
P808-5c 构型判据：严格口径与直接构造之差是系统性的 N^2 因子。
P808-5d 构型判据：库口径在四档 N^2 上逐位相容。
P808-5e 构型判据：常数 N 档对两个候选无区分力（区分力边界登记）。
P808-6  构型判据：N 随位置变时两侧空间 Christoffel 之差等于 N 导数修正。
P808-7  恒真检验：把三处订正改回原实现，相关判据必须判否。
P808-8  状态描述：登记。

误差预算
--------
方案误差的独立估计取自坐标解析闭式裁判：3D 侧 Christoffel 对解析闭式的差
（同一档同一步长）。凡两侧残差与该尺度同阶即判为截断误差量级，不作物理
归因；凡残差超过该尺度十倍才进入判否。判据中出现的相对差一律与对应的
绝对读数同时给出。

边界
----
不判 3+1 分解式在完整动源构型上的成立性；不判 G_ij 侧的完整 ADM 形式
（lapse 项的写法留作下一轮）；不判强场；不判动态波方程的 3+1 结构；
不判源侧 O(v^2) 缺口；不判注入-扩散平衡那一侧。
"""
import numpy as np

HALF = 8
EPS_LIST = (0.0125, 0.00625)
M_WIDTH = 0.05
R0_MAIN = 1.4
R0_SECOND = 1.05

RESULT = []


def record(tag, ok, detail):
    RESULT.append((tag, bool(ok), detail))
    print("[%s] %s  %s" % ("PASS" if ok else "FAIL", tag, detail))


def make(m_width):
    def th_c(r):
        r = np.asarray(r, dtype=float)
        t = 1.0 / (1.0 + np.exp((r - 1.0) / m_width))
        return t - t * t

    def F_of(r):
        return 2.0 * th_c(r)

    def G_of(r):
        t = th_c(r)
        return 2.0 * t + 3.5 * t * t
    return th_c, F_of, G_of


def dd(a, axis, eps):
    b = np.moveaxis(a, axis, -1)
    out = np.full_like(b, np.nan)
    out[..., 2:-2] = (b[..., :-4] - 8 * b[..., 1:-3] + 8 * b[..., 3:-1] - b[..., 4:]) / (12 * eps)
    return np.moveaxis(out, -1, axis)


def build_slice(x0, eps, half, F_of, G_of, N_const, G_const, n2_value=0.5):
    k = np.arange(-half, half + 1) * eps
    X = x0[0] + k[:, None, None]
    Y = x0[1] + k[None, :, None]
    Z = x0[2] + k[None, None, :]
    r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    Fv, Gv = F_of(r), G_of(r)
    if G_const:
        Gv = np.ones_like(Gv) * float(Gv[half, half, half])
    N2 = 1.0 - Fv
    if N_const:
        N2 = np.ones_like(N2) * n2_value
    g = np.zeros((4, 4, 2 * half + 1, 2 * half + 1, 2 * half + 1))
    g[0, 0] = -N2
    g[1, 1] = g[2, 2] = g[3, 3] = N2 * (1.0 + Gv)
    return g, X, Y, Z, r, Gv


def christoffel_4d(g, eps):
    dg = [np.zeros_like(g), dd(g, 2, eps), dd(g, 3, eps), dd(g, 4, eps)]
    gi = np.zeros_like(g)
    for i in range(4):
        gi[i, i] = 1.0 / g[i, i]
    sh = g.shape[2:]
    Gam = np.zeros((4, 4, 4) + sh)
    for lam in range(4):
        for mu in range(4):
            for nu in range(4):
                acc = np.zeros(sh)
                for sig in range(4):
                    acc = acc + 0.5 * gi[lam, sig] * (
                        dg[mu][sig, nu] + dg[nu][sig, mu] - dg[sig][mu, nu])
                Gam[lam, mu, nu] = acc
    return Gam, dg, gi, sh


def ricci_4d(Gam, gi, sh, eps, mode="std"):
    """mode = std : a2 直接相加，标量按逆度规收缩（订正版）。
    mode = buggy: a2 再取负号，标量按对角元和相加（第 803 号原实现）。"""
    Z = np.zeros(sh)

    def ricci(mu, nu):
        a1 = Z.copy()
        for rho in (1, 2, 3):
            a1 = a1 + dd(Gam[rho, nu, mu], rho - 1, eps)
        a2 = Z.copy()
        if nu != 0:
            for rho in range(4):
                a2 = a2 - dd(Gam[rho, rho, mu], nu - 1, eps)
        a3 = Z.copy()
        for rho in range(4):
            for sig in range(4):
                a3 = a3 + Gam[rho, rho, sig] * Gam[sig, nu, mu]
        a4 = Z.copy()
        for rho in range(4):
            for sig in range(4):
                a4 = a4 - Gam[rho, nu, sig] * Gam[sig, rho, mu]
        if mode == "std":
            return a1 + a2 + a3 + a4
        return a1 - a2 + a3 - a4

    Ric00 = ricci(0, 0)
    Ricij = np.zeros((3, 3) + sh)
    for i in range(1, 4):
        for j in range(1, 4):
            Ricij[i - 1, j - 1] = ricci(i, j)
    if mode == "std":
        R = gi[0, 0] * Ric00
        for i in range(1, 4):
            R = R + gi[i, i] * Ricij[i - 1, i - 1]
    else:
        R = Ric00.copy()
        for i in range(1, 4):
            R = R + Ricij[i - 1, i - 1]
    return Ric00, Ricij, R


def inv3(metric):
    sh = metric.shape[2:]
    ginv = np.zeros((3, 3) + sh)
    for i in range(3):
        for j in range(3):
            co = (metric[(i + 1) % 3][(j + 1) % 3] * metric[(i + 2) % 3][(j + 2) % 3]
                  - metric[(i + 1) % 3][(j + 2) % 3] * metric[(i + 2) % 3][(j + 1) % 3])
            ginv[i, j] = co
    det = (metric[0, 0] * (metric[1, 1] * metric[2, 2] - metric[1, 2] * metric[2, 1])
           - metric[0, 1] * (metric[1, 0] * metric[2, 2] - metric[1, 2] * metric[2, 0])
           + metric[0, 2] * (metric[1, 0] * metric[2, 1] - metric[1, 1] * metric[2, 0]))
    return ginv / det


def gam_christoffel(gam, ginv, eps, mode):
    """mode = fixed : E[(i,j,k)] = d_k gamma_ij 按标准式配对（订正版）。
    mode = buggy: 按第 803 号原实现的配对（下标错位）。"""
    sh = gam.shape[2:]
    E = {}
    for k in range(3):
        for i in range(3):
            for j in range(3):
                E[(i, j, k)] = dd(gam[i, j], k, eps)
    Gam = np.zeros((3, 3, 3) + sh)
    for lam in range(3):
        for mu in range(3):
            for nu in range(3):
                acc = np.zeros(sh)
                for sig in range(3):
                    if mode == "fixed":
                        acc = acc + 0.5 * ginv[lam, sig] * (
                            E[(sig, nu, mu)] + E[(sig, mu, nu)] - E[(mu, nu, sig)])
                    else:
                        acc = acc + 0.5 * ginv[lam, sig] * (
                            E[(mu, nu, sig)] + E[(nu, mu, sig)] - E[(sig, mu, nu)])
                Gam[lam, mu, nu] = acc
    return Gam


def gam_ricci(gam, ginv, Gam, eps):
    sh = gam.shape[2:]
    Ric = np.zeros((3, 3) + sh)
    for i in range(3):
        for j in range(3):
            acc = np.zeros(sh)
            for k in range(3):
                acc = acc + dd(Gam[k, i, j], k, eps)
                acc = acc - dd(Gam[k, i, k], j, eps)
            for k in range(3):
                tr = np.zeros(sh)
                for l in range(3):
                    tr = tr + Gam[l, k, l]
                acc = acc + Gam[k, i, j] * tr
                for l in range(3):
                    acc = acc - Gam[l, i, k] * Gam[k, j, l]
            Ric[i, j] = acc
    R3 = np.zeros(sh)
    for i in range(3):
        for j in range(3):
            R3 = R3 + ginv[i, j] * Ric[i, j]
    return Ric, R3


def analytic_gam_christoffel(rr, G_eff, coord):
    h = 1e-6
    fp = (G_eff(rr + h) - G_eff(rr - h)) / (2 * h)
    fk = fp * coord / rr
    f = 1.0 + G_eff(rr)
    Gam = np.zeros((3, 3, 3) + rr.shape)
    for lam in range(3):
        for mu in range(3):
            for nu in range(3):
                num = np.zeros(rr.shape)
                if mu == lam:
                    num = num + fk[nu]
                if nu == lam:
                    num = num + fk[mu]
                if mu == nu:
                    num = num - fk[lam]
                Gam[lam, mu, nu] = num / (2.0 * f)
    return Gam


def lapse_correction(N_arr, eps, gam, ginv_gam):
    """g_ij = N^2 gamma_ij 时两侧空间 Christoffel 之差。

    共形变换的标准形式：令 sigma = ln N（g_ij = e^{2 sigma} gamma_ij），则
    Gamma^k_{ij}(g) = Gamma^k_{ij}(gamma)
                    + delta^k_i d_j sigma + delta^k_j d_i sigma
                    - gamma_{ij} gamma^{kl} d_l sigma。
    这里 d 是坐标偏导（两者取同一坐标图），故 d sigma = dN / N。
    """
    sh = N_arr.shape
    dN = [dd(N_arr, ax, eps) for ax in (0, 1, 2)]
    dlnN = [dN[ax] / N_arr for ax in range(3)]
    up_dlnN = np.zeros((3,) + sh)
    for k in range(3):
        acc = np.zeros(sh)
        for l in range(3):
            acc = acc + ginv_gam[k, l] * dlnN[l]
        up_dlnN[k] = acc
    corr = np.zeros((3, 3, 3) + sh)
    for k in range(3):
        for i in range(3):
            for j in range(3):
                val = 0.0
                if k == j:
                    val = val + dlnN[i]
                if k == i:
                    val = val + dlnN[j]
                val = val - gam[i, j] * up_dlnN[k]
                corr[k, i, j] = val
    return corr


def ctr(arr, half=HALF):
    return arr[half, half, half]


def run(N_const, G_const, eps, r0, half=HALF, m_width=M_WIDTH, bug3d=False, bug4d=False,
        n2_value=0.5):
    th_c, F_of, G_of = make(m_width)
    g, X, Y, Z, r, Gv = build_slice((r0, 0.0, 0.0), eps, half, F_of, G_of, N_const,
                                   G_const, n2_value)
    Gam4, dg, gi, sh = christoffel_4d(g, eps)

    Ric00_std, Ricij_std, R4d_std = ricci_4d(Gam4, gi, sh, eps, mode="std")
    Ric00_bug, Ricijb_bug, R4d_bug = ricci_4d(Gam4, gi, sh, eps, mode="buggy")

    N2 = -g[0, 0]
    gam = np.zeros((3, 3) + sh)
    gsp = np.zeros((3, 3) + sh)
    for i in range(3):
        for j in range(3):
            gam[i, j] = g[i + 1, j + 1] / N2
            gsp[i, j] = g[i + 1, j + 1]
    ginv_gam = inv3(gam)
    ginv_gsp = inv3(gsp)

    G3f = gam_christoffel(gam, ginv_gam, eps, "fixed")
    G3b = gam_christoffel(gam, ginv_gam, eps, "buggy")
    Ric3f, R3f = gam_ricci(gam, ginv_gam, G3f, eps)
    Ric3b, R3b = gam_ricci(gam, ginv_gam, G3b, eps)

    if bug3d:
        Gam3_sel, Ric3_sel, R3_sel = G3b, Ric3b, R3b
    else:
        Gam3_sel, Ric3_sel, R3_sel = G3f, Ric3f, R3f
    if bug4d:
        Ric00_sel, Ricij_sel, R4d_sel = Ric00_bug, Ricijb_bug, R4d_bug
    else:
        Ric00_sel, Ricij_sel, R4d_sel = Ric00_std, Ricij_std, R4d_std

    gconst = float(Gv[half, half, half])
    if G_const:
        G_eff = lambda rr: np.ones_like(np.asarray(rr, dtype=float)) * gconst
    else:
        G_eff = G_of
    coord = np.stack(np.broadcast_arrays(X, Y, Z))
    Gan = analytic_gam_christoffel(r, G_eff, coord)

    def d3(A, B):
        return max(abs(ctr(A[l, m, n]) - ctr(B[l, m, n]))
                   for l in range(3) for m in range(3) for n in range(3))

    def amp3(A):
        return max(abs(ctr(A[l, m, n])) for l in range(3) for m in range(3) for n in range(3))

    def amp_m(A):
        return max(abs(ctr(A[i, j])) for i in range(3) for j in range(3))

    sp = slice(1, 4)
    g4sp = Gam4[sp, sp, sp]
    corr = lapse_correction(np.sqrt(N2), eps, gam, ginv_gam)

    o = {}
    o["gam_4d_vs_3"] = d3(g4sp, Gam3_sel)
    o["ric_vs_3"] = max(abs(ctr(Ricij_sel[i, j]) - ctr(Ric3_sel[i, j]))
                        for i in range(3) for j in range(3))
    o["gam_4d_vs_3bug"] = d3(g4sp, G3b)
    o["gam_3fix_vs_an"] = d3(G3f, Gan)
    o["gam_3bug_vs_an"] = d3(G3b, Gan)
    o["gam_4d_vs_an"] = d3(g4sp, Gan)
    o["ric_bug_vs_3"] = max(abs(ctr(Ricijb_bug[i, j]) - ctr(Ric3f[i, j]))
                            for i in range(3) for j in range(3))
    o["R4d"] = abs(ctr(R4d_sel))
    o["R4d_bug"] = abs(ctr(R4d_bug))
    o["R4d_std"] = abs(ctr(R4d_std))
    o["R3_gamma"] = abs(ctr(R3_sel))
    Gsp = gam_christoffel(gsp, ginv_gsp, eps, "fixed")
    Ricsp, Rsp = gam_ricci(gsp, ginv_gsp, Gsp, eps)
    o["R3_gsp"] = abs(ctr(Rsp))
    o["R3_gamma_std"] = abs(ctr(R3f))

    o["corr_amp"] = amp3(corr)
    o["gam_lapse_resid"] = d3(g4sp, Gam3_sel + corr)

    G4 = np.zeros((3, 3) + sh)
    for i in range(3):
        for j in range(3):
            G4[i, j] = Ricij_sel[i, j] - 0.5 * g[i + 1, j + 1] * R4d_sel

    def dG(cand):
        return max(abs(ctr(cand[i, j]) - ctr(G4[i, j])) for i in range(3) for j in range(3))

    c_n2free = np.zeros((3, 3) + sh)
    c_n2 = np.zeros((3, 3) + sh)
    c_lib = np.zeros((3, 3) + sh)
    for i in range(3):
        for j in range(3):
            c_n2free[i, j] = Ric3_sel[i, j] - 0.5 * gam[i, j] * R3_sel
            c_n2[i, j] = N2 * (Ric3_sel[i, j] - 0.5 * gam[i, j] * R3_sel)
            c_lib[i, j] = Ric3_sel[i, j] - 0.5 * gam[i, j] * Rsp
    o["gij_resid_n2free"] = dG(c_n2free)
    o["gij_resid_n2"] = dG(c_n2)
    o["gij_resid_lib"] = dG(c_lib)
    o["gij_scale"] = amp_m(G4)

    g00_direct = ctr(Ric00_sel - 0.5 * g[0, 0] * R4d_sel)
    o["g00_direct"] = abs(g00_direct)
    o["g00_c1_lib"] = abs(ctr(0.5 * N2 * Rsp))
    o["g00_c2_strict"] = abs(ctr(0.5 * N2 * R3_sel))
    o["g00_c3_n2free"] = abs(ctr(0.5 * R3_sel))
    scale = max(o["g00_direct"], o["g00_c1_lib"], o["g00_c2_strict"], o["g00_c3_n2free"])
    o["g00_scale"] = scale

    def rel(a, b):
        return abs(a - b) / scale if scale > 0 else float("inf")

    o["g00_rel_lib"] = rel(o["g00_c1_lib"], o["g00_direct"])
    o["g00_rel_strict"] = rel(o["g00_c2_strict"], o["g00_direct"])
    o["g00_rel_n2free"] = rel(o["g00_c3_n2free"], o["g00_direct"])

    o["g00_sd_direct"] = g00_direct
    o["g00_sd_lib"] = ctr(0.5 * N2 * Rsp)
    o["g00_sd_strict"] = ctr(0.5 * N2 * R3_sel)
    o["g00_sd_n2free"] = ctr(0.5 * R3_sel)
    o["g00_ratio_lib"] = (o["g00_sd_lib"] / o["g00_sd_direct"]
                          if o["g00_sd_direct"] != 0 else float("nan"))
    o["g00_ratio_strict"] = (o["g00_sd_strict"] / o["g00_sd_direct"]
                             if o["g00_sd_direct"] != 0 else float("nan"))
    o["g00_ratio_n2free"] = (o["g00_sd_n2free"] / o["g00_sd_direct"]
                             if o["g00_sd_direct"] != 0 else float("nan"))
    o["n2_value"] = n2_value

    o["gij_rel_n2free"] = o["gij_resid_n2free"] / o["gij_scale"] if o["gij_scale"] > 0 else float("inf")
    o["gij_rel_n2"] = o["gij_resid_n2"] / o["gij_scale"] if o["gij_scale"] > 0 else float("inf")
    o["gij_rel_lib"] = o["gij_resid_lib"] / o["gij_scale"] if o["gij_scale"] > 0 else float("inf")

    o["gamma_radial_curv"] = o["R3_gamma_std"]
    o["resid_over_curv"] = (o["gij_resid_n2free"] / o["gamma_radial_curv"]
                            if o["gamma_radial_curv"] > 0 else float("inf"))
    return o


CASES = (("甲 N 常数 G(r)", True, False),
         ("乙 N(r) G(r)", False, False),
         ("丙 N、G 都常数", True, True))


def show(title, o):
    print()
    print("  %s" % title)
    print("    Gamma 4D vs 3D 订正   %.3e ｜ 4D vs 3D 原实现   %.3e"
          % (o["gam_4d_vs_3"], o["gam_4d_vs_3bug"]))
    print("    Gamma 3D 订正 vs 解析 %.3e ｜ 3D 原实现 vs 解析 %.3e"
          % (o["gam_3fix_vs_an"], o["gam_3bug_vs_an"]))
    print("    Ric 4D vs 3D 订正 %.3e ｜ 4D 原实现 vs 3D %.3e" % (o["ric_vs_3"], o["ric_bug_vs_3"]))
    print("    R^(4D) 订正 %.3e ｜ 原实现 %.3e ｜ R^(3) gamma %.3e ｜ R^(3) 空间度规 %.3e"
          % (o["R4d"], o["R4d_bug"], o["R3_gamma"], o["R3_gsp"]))
    print("    N 导数修正幅度 %.3e ｜ 扣修正后残余 %.3e" % (o["corr_amp"], o["gam_lapse_resid"]))
    print("    G_ij 残余: 无 N^2 因子 %.3e ｜ 带 N^2 因子 %.3e ｜ 库口径 %.3e"
          % (o["gij_resid_n2free"], o["gij_resid_n2"], o["gij_resid_lib"]))
    print("    G_00: 直接 %.3e ｜ 库口径 0.5 N^2 R^(3)[g] %.3e ｜ 严格口径 0.5 N^2 R^(3)[gamma] %.3e"
          % (o["g00_direct"], o["g00_c1_lib"], o["g00_c2_strict"]))
    print("    G_00 相对差: 库口径 %.3e ｜ 严格口径 %.3e ｜ 无 N^2 因子 %.3e"
          % (o["g00_rel_lib"], o["g00_rel_strict"], o["g00_rel_n2free"]))


print("=" * 78)
print("第 808 号：三处装置级缺陷的定位与 3+1 关系式的口径归属")
print("=" * 78)

reads = {}
for r0 in (R0_MAIN, R0_SECOND):
    for eps in EPS_LIST:
        print()
        print(">>> r0 = %.2f，步长 eps = %.4f（HALF = %d，m = %.2f）" % (r0, eps, HALF, M_WIDTH))
        for name, nc, gc in CASES:
            o = run(nc, gc, eps, r0)
            reads[(name, r0, eps)] = o
            show("构型 %s" % name, o)

oA = reads[("甲 N 常数 G(r)", R0_MAIN, EPS_LIST[0])]
oA2 = reads[("甲 N 常数 G(r)", R0_MAIN, EPS_LIST[1])]
oB = reads[("乙 N(r) G(r)", R0_MAIN, EPS_LIST[0])]
oB2 = reads[("乙 N(r) G(r)", R0_MAIN, EPS_LIST[1])]
oB5 = reads[("乙 N(r) G(r)", R0_SECOND, EPS_LIST[0])]
oC = reads[("丙 N、G 都常数", R0_MAIN, EPS_LIST[0])]

# 方案误差的独立尺度：3D 侧对解析闭式的差（同一档同一步长）
schA = oA["gam_3fix_vs_an"]
schB = oB["gam_3fix_vs_an"]
tolA = max(10.0 * schA, 1e-12)
tolB = max(10.0 * schB, 1e-12)
print()
print("  方案误差尺度（3D 侧对解析闭式）: 甲档 %.3e ｜ 乙档 %.3e" % (schA, schB))
print("  由此得到的残差容差: 甲档 %.3e ｜ 乙档 %.3e" % (tolA, tolB))

# P808-1a 3D Christoffel 下标错位
record("P808-1a 3D Christoffel 下标错位",
       oA["gam_3bug_vs_an"] >= 1e-2 and oA["gam_3fix_vs_an"] <= 1e-5,
       "原实现对解析 %.3e（应 >= 1e-2）｜ 订正后 %.3e（应 <= 1e-5）"
       % (oA["gam_3bug_vs_an"], oA["gam_3fix_vs_an"]))

# P808-1b 4D Ricci 符号与标量收缩
record("P808-1b 4D Ricci 符号与标量收缩",
       oA["ric_bug_vs_3"] >= 1e-2 and oA["ric_vs_3"] <= 1e-12,
       "原实现对 3D %.3e（应 >= 1e-2）｜ 订正后 %.3e（应 <= 1e-12）"
       % (oA["ric_bug_vs_3"], oA["ric_vs_3"]))

# P808-2 订正后 4D 与 3D 相容
record("P808-2 甲档 4D 与 3D 相容",
       oA["gam_4d_vs_3"] <= 1e-12 and oA["ric_vs_3"] <= 1e-12,
       "Gamma 差 %.3e ｜ Ricci 差 %.3e（均应 <= 1e-12）"
       % (oA["gam_4d_vs_3"], oA["ric_vs_3"]))

record("P808-2b 收敛检验（步长减半）",
       oA2["gam_4d_vs_3"] <= 1e-12 and oA2["ric_vs_3"] <= 1e-12,
       "eps = %.4f 时 Gamma 差 %.3e ｜ Ricci 差 %.3e"
       % (EPS_LIST[1], oA2["gam_4d_vs_3"], oA2["ric_vs_3"]))

# P808-3 平凡档机器零
record("P808-3 平凡档机器零",
       oC["gam_4d_vs_3"] <= 1e-12 and oC["ric_vs_3"] <= 1e-12
       and oC["gam_3fix_vs_an"] <= 1e-12,
       "Gamma 差 %.3e ｜ Ricci 差 %.3e ｜ 3D 对解析 %.3e"
       % (oC["gam_4d_vs_3"], oC["ric_vs_3"], oC["gam_3fix_vs_an"]))

# P808-4 801 至 804 号的定量对象消失
record("P808-4 801 至 804 号的定量对象消失",
       oA["gij_resid_n2free"] <= 1e-12 and oA["gamma_radial_curv"] >= 1e-3,
       "订正后 G_ij 残余 %.3e（应 <= 1e-12）｜ gamma 径向曲率 %.3e（应非零）"
       % (oA["gij_resid_n2free"], oA["gamma_radial_curv"]))

record("P808-4b 残余与曲率之比不再有意义",
       oA["resid_over_curv"] <= 1e-12,
       "残余除以 gamma 径向曲率 %.3e（应 <= 1e-12）" % oA["resid_over_curv"])

# P808-5 库内登记式的口径归属
record("P808-5 库口径登记式在甲档相容",
       oA["g00_rel_lib"] <= 1e-10,
       "0.5 N^2 R^(3)[空间度规] 对直接构造相对差 %.3e（应 <= 1e-10）" % oA["g00_rel_lib"])

record("P808-5b 严格口径代入同一式不相容",
       oA["g00_rel_strict"] >= 1e-1,
       "0.5 N^2 R^(3)[gamma] 对直接构造相对差 %.3e（应 >= 1e-1）" % oA["g00_rel_strict"])

# P808-6 N 导数修正
record("P808-6 乙档两侧 Christoffel 之差等于 N 导数修正",
       oB["corr_amp"] >= 1e-4 and oB["gam_lapse_resid"] <= tolB,
       "修正幅度 %.3e（应 >= 1e-4）｜ 扣修正后残余 %.3e（容差 %.3e）"
       % (oB["corr_amp"], oB["gam_lapse_resid"], tolB))

record("P808-6b 甲档 N 为常数时修正项消失且两侧相容",
       oA["corr_amp"] <= 1e-12 and oA["gam_lapse_resid"] <= 1e-12,
       "修正幅度 %.3e ｜ 残余 %.3e（均应 <= 1e-12）"
       % (oA["corr_amp"], oA["gam_lapse_resid"]))

record("P808-6c 乙档残差随步长细化而收缩",
       oB2["gam_lapse_resid"] <= oB["gam_lapse_resid"],
       "eps = %.4f 残余 %.3e ｜ eps = %.4f 残余 %.3e（应不增）"
       % (EPS_LIST[0], oB["gam_lapse_resid"], EPS_LIST[1], oB2["gam_lapse_resid"]))

# 适用域读数（登记）
print()
print("  适用域读数（乙档，N 随位置变）")
for tag, o in (("r0 = 1.40, eps = 0.0125", oB), ("r0 = 1.40, eps = 0.0063", oB2),
                ("r0 = 1.05, eps = 0.0125", oB5)):
    print("    %s：G_00 直接 %.3e ｜ 库口径相对差 %.3e ｜ 严格口径相对差 %.3e ｜ 无 N^2 因子相对差 %.3e"
          % (tag, o["g00_direct"], o["g00_rel_lib"], o["g00_rel_strict"],
             o["g00_rel_n2free"]))
    print("      G_ij 残余 无 N^2 因子 %.3e ｜ 带 N^2 因子 %.3e ｜ 库口径 %.3e"
          % (o["gij_resid_n2free"], o["gij_resid_n2"], o["gij_resid_lib"]))

# N^2 扫描：严格口径与库口径之差是系统性的 N^2 因子
print()
print("  N^2 扫描（构型甲，N 为常数）：三个候选对直接构造的有符号比值")
scan = []
for n2v in (0.25, 0.5, 0.75, 0.9):
    o_s = run(True, False, EPS_LIST[0], R0_MAIN, n2_value=n2v)
    scan.append((n2v, o_s))
    print("    N^2 = %.2f ｜ 库口径比值 %+.10f ｜ 严格口径比值 %+.10f ｜ 无 N^2 因子比值 %+.10f ｜ 预测 N^2 = %.2f"
          % (n2v, o_s["g00_ratio_lib"], o_s["g00_ratio_strict"],
             o_s["g00_ratio_n2free"], n2v))


def ratio_err(o, key, target):
    return abs(o[key] - target)


strict_err = max(ratio_err(o, "g00_ratio_strict", n2v) for n2v, o in scan)
record("P808-5c 严格口径与直接构造之差恰是 N^2 因子",
       strict_err <= 1e-9,
       "四档 |严格口径比值 - N^2| 最大 %.3e（应 <= 1e-9）" % strict_err)

lib_err = max(ratio_err(o, "g00_ratio_lib", 1.0) for n2v, o in scan)
record("P808-5d 库口径在四档 N^2 上都逐位相容",
       lib_err <= 1e-9,
       "四档 |库口径比值 - 1| 最大 %.3e（应 <= 1e-9）" % lib_err)

n2free_err = max(ratio_err(o, "g00_ratio_n2free", 1.0) for n2v, o in scan)
record("P808-5e 去掉 N^2 因子的候选在常数 N 档不可区分",
       n2free_err <= 1e-9,
       "四档 |无 N^2 因子比值 - 1| 最大 %.3e（应 <= 1e-9）；该档对两个候选无区分力"
       % n2free_err)

# P808-7 恒真检验
print()
print("  恒真检验：把三处订正改回原实现")
oBug = run(True, False, EPS_LIST[0], R0_MAIN, bug3d=True, bug4d=True)
show("构型 甲（N 常数 G(r)），原实现", oBug)
print("    原实现下 G_ij 残余 %.3e ｜ 除以 gamma 径向曲率 %.3e（803 与 804 号报告的量级）"
      % (oBug["gij_resid_n2free"], oBug["resid_over_curv"]))
flip = (oBug["gam_4d_vs_3"] >= 1e-2) or (oBug["ric_vs_3"] >= 1e-2)
record("P808-7 恒真检验", flip,
       "原实现下 Gamma 差 %.3e ｜ Ricci 差 %.3e（相关判据应判否）"
       % (oBug["gam_4d_vs_3"], oBug["ric_vs_3"]))

# P808-8 状态登记
record("P808-8 状态登记", True,
       "本轮判据共 %d 条；撤回事项与登记读数见注记" % (len(RESULT) + 1))

print()
print("=" * 78)
print("判据汇总：%d/%d 通过" % (sum(1 for _, ok, _ in RESULT if ok), len(RESULT)))
for tag, ok, detail in RESULT:
    print("  [%s] %s  %s" % ("PASS" if ok else "FAIL", tag, detail))
print("=" * 78)

# 图
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    labels = ["Gamma: 3D 原实现\n对解析闭式", "Gamma: 3D 订正后\n对解析闭式",
              "Ricci: 4D 原实现\n对 3D 订正", "Ricci: 4D 订正后\n对 3D 订正"]
    vals = [oA["gam_3bug_vs_an"], oA["gam_3fix_vs_an"], oA["ric_bug_vs_3"], oA["ric_vs_3"]]
    floorv = 1e-17
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    ax.bar(range(4), [max(v, floorv) for v in vals],
           color=["#b03030", "#3070b0", "#b03030", "#3070b0"])
    ax.set_yscale("log")
    ax.set_ylim(floorv, 10.0)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("最大绝对差")
    ax.set_title("三处装置级缺陷：原实现与订正后（构型甲）")
    ax.axhline(1e-12, ls="--", lw=0.8, color="#666666")
    ax.text(0.02, 1.4e-12, "机器零阈值 1e-12", fontsize=7, color="#666666")
    for i, v in enumerate(vals):
        ax.text(i, max(v, floorv) * 1.6, "%.2e" % v, ha="center", fontsize=7)

    ax = axes[1]
    names = ["库口径\n(1/2)N^2 R^(3)[g_ij]", "严格口径\n(1/2)N^2 R^(3)[gamma]",
             "无 N^2 因子\n(1/2) R^(3)[gamma]"]
    a1 = [oA["g00_rel_lib"], oA["g00_rel_strict"], oA["g00_rel_n2free"]]
    a2 = [oB["g00_rel_lib"], oB["g00_rel_strict"], oB["g00_rel_n2free"]]
    x = range(3)
    ax.bar([i - 0.2 for i in x], a1, width=0.38, label="构型甲 N 常数", color="#3070b0")
    ax.bar([i + 0.2 for i in x], a2, width=0.38, label="构型乙 N(r)", color="#d08030")
    ax.set_yscale("log")
    ax.set_ylim(1e-10, 10.0)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("对四维直接构造的相对差")
    ax.set_title("G_00 各口径的相对差")
    ax.legend(fontsize=8)

    ax = axes[2]
    tags = ["r0=1.40\neps=0.0125", "r0=1.40\neps=0.0063", "r0=1.05\neps=0.0125"]
    res = [oB["gam_lapse_resid"], oB2["gam_lapse_resid"], oB5["gam_lapse_resid"]]
    amp = [oB["corr_amp"], oB2["corr_amp"], oB5["corr_amp"]]
    ax.bar([i - 0.2 for i in range(3)], res, width=0.38, label="扣修正后残余", color="#3070b0")
    ax.bar([i + 0.2 for i in range(3)], amp, width=0.38, label="修正幅度", color="#d08030")
    ax.set_yscale("log")
    ax.set_ylim(1e-18, 10.0)
    ax.set_xticks(range(3))
    ax.set_xticklabels(tags, fontsize=8)
    ax.set_ylabel("量级")
    ax.set_title("乙档：N 导数修正的幅度与残余")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = "images/p808_device_defects.png"
    fig.savefig(out, dpi=140)
    print()
    print("图已存：%s" % out)
except Exception as exc:
    print()
    print("图未生成：%s" % exc)
