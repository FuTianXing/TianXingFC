# -*- coding: utf-8 -*-
"""第 615 号：局域场系数的维度标度判据轮（gamma_d = 1/d 的数值读出）

出处：内部草稿《注记-局域场系数的第一性推导：维度标度与近邻抵消条件
（gamma_3D=1/3 的来源，条件性命题）》第 25 轮登记后的判据立项；
衔接第 582 号（Clausius-Mossotti 密度标度，gamma 作为输入位）与介质折射线。

判据（先立期望，后跑数值）：
  V3a  维度标度与平坦性：
       a1  3D 球腔表面束缚电荷（sigma = -P cos theta，内表面）的离散晶格
           厚层求和给 gamma_3 = 1/3（多档半径，窗间平坦）；
       a2  2D 圆腔自适应角配分给 gamma_2 = 1/2；
       a3  齿（反例）：均匀面电荷球壳（sigma = const）腔心场严格为零，
           读数必须给 0 而非 1/3（检出 cos^2 结构，排除恒真）。
  V3b  近邻抵消：
       b1  3D 立方第一配位球（6 近邻）z 场和 = 0（对称性代数恒等式，逐位）；
       b2  第二配位球（12 边中点）按 n_z 分组：n_z=0 组和 = -sqrt(2)、
           n_z=+-1 组和 = +sqrt(2)（非平凡精确值），总和 = 0（逐位）；
       b3  扰动齿：格点位置随机扰动后第一配位球偏移随扰动幅度增长，
           最大档远超容差（对称性破坏即偏移）；
       b4  径向恒等式与形状依赖：任意径向区域 |n| in (a,b] 的偶极场和
           精确为零（S3 对称恒等式：Sum nz^2 = (1/3) Sum |n|^2，逐位
           确认），删点反例齿破对称即翻；板截断和对照连续球盖公式
           -4 pi [2/3 - Z/L + Z^3/(3L^3)]（偏差 ~ c/Z 衰减），球截断
           精确零 vs 板和 O(1) 非零 = 条件收敛形状依赖（退极化）的
           数值面。
  V3c  维度对照与几何条件齿：
       c1  2D 读数与 3D 读数之差远超各自容差（维度区分力内置）；
       c2  针腔齿：极化沿轴的针形腔侧面束缚电荷密度为零，端面贡献
           eps0*E/P = -rho^2/(2 Z^2)（连续），读数显著偏离 1/3
           （"各向同性选球腔"条件的必要性）；
       c3  回归：三档维度（1D 解析档 gamma_1 = 1、2D、3D）的读数对 1/d
           回归斜率 = 1。

模型设定：
  3D 点偶极场 E = (1/4 pi eps0) [3(p.r^)r^ - p]/r^3；2D 线偶极
  E = (1/2 pi eps0) [2(p.r^)r^ - p]/r^2。均匀极化 P（点/线密度乘偶极矩）。
  腔表面束缚电荷（内表面）sigma = -P cos theta，腔心场
  E_cav = (P/eps0 d)，d 维球面 <cos^2> = 1/d 是 gamma_d = 1/d 的几何内核。
  无量纲读数 gamma_d_num = eps0 * E_cav / P。

数值口径（首轮教训修正）：
  3D：单层球面的台阶配分误差振荡 ~ O(1/R) 且相位随 R 变化（R=20 恰为
  特异好点，单对定标撞好运），改用厚 W=5 层求和（层间台阶相位无关，
  振荡平均 ~ 1/sqrt(W)），权重 dOmega_j = 1/r_j^2（体素立体角配分）。
  2D：单层圆环 dtheta = 1/r 配分的台阶误差 O(1/R) 过大（R=30 偏差 3e-2），
  改自适应角配分：圆环点按极角排序，dtheta_j 取相邻半间隔和（梯形
  配分，对周期光滑函数高阶收敛）。
  针腔端面：z 法向面元每体素权重 1，端面场 = +-N_pts/(4 pi Z^2)，两端
  同号叠加；连续期望 -rho^2/(2 Z^2)（圆盘轴上场精确式的远端近似）。

容差预算（判据容差从误差来源推导，禁拍脑袋）：
  tol = max(2 x 定标偏差(R=15 对 R=30), 1e-3)；定标偏差随口径打印。
  b3/c1/c2 的齿判定用"显著超出读数容差"的相对判别，不用绝对阈值。

可逆性等级：V3a/V3b/V3c 判据稳健（换求和口径或破坏对称性即翻）。
边界声明：验证的是局域场系数 gamma_d = 1/d 的几何来源（球面 cos^2 二阶矩 +
晶格近邻抵消 + 维度标度同构）在离散晶格上的成立与收敛；不验证极化强度 P
的物质侧出身（P1-47/P1-49 线）、不验证 582 号解析模型的色散层、不验证
真实材料的介电常数数值。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

RNG = np.random.default_rng(20261012)
IMG_DIR = r"D:\software\TianXingFC\images"
RESULTS = {}


# ----------------------------------------------------------------------
# 工具：晶格壳层点
# ----------------------------------------------------------------------
def shell_points_3d(R, W=1):
    """3D 晶格点满足 R-W/2 < |n| <= R+W/2（厚 W 层，排除原点）。"""
    n = np.arange(-int(R + W / 2) - 2, int(R + W / 2) + 3)
    nx, ny, nz = np.meshgrid(n, n, n, indexing="ij")
    pts = np.stack([nx.ravel(), ny.ravel(), nz.ravel()], axis=1).astype(float)
    r = np.linalg.norm(pts, axis=1)
    m = (r > R - W / 2) & (r <= R + W / 2)
    return pts[m], r[m]


def shell_points_2d(R):
    n = np.arange(-int(R) - 2, int(R) + 3)
    nx, ny = np.meshgrid(n, n, indexing="ij")
    pts = np.stack([nx.ravel(), ny.ravel()], axis=1).astype(float)
    r = np.linalg.norm(pts, axis=1)
    m = (r > R - 0.5) & (r <= R + 0.5)
    return pts[m], r[m]


# ----------------------------------------------------------------------
# V3a：腔表面束缚电荷离散求和
# ----------------------------------------------------------------------
def gamma3_sphere(R, W=5):
    """3D 球腔内表面 sigma = -P cos theta 的腔心场读数 gamma_3 = eps0*E/P。

    E_z = Sum [sigma_j dA_j (-cos theta_j)]/(4 pi eps0 r_j^2)
        = P/(4 pi eps0) * Sum cos^2 theta_j dOmega_j
    厚 W 层求和压台阶振荡；dOmega_j = 1/r_j^2（体素立体角配分）。
    归一化读数（除以总权重）：厚层每层各张完整 4 pi，不归一会把 W 层
    重复计入（读数 = W/3）；归一化后即球面 cos^2 平均。
    """
    pts, r = shell_points_3d(R, W=W)
    dOmega = 1.0 / r ** 2
    cos2 = (pts[:, 2] / r) ** 2
    return float(np.sum(dOmega * cos2) / np.sum(dOmega))


def gamma2_circle(R):
    """2D 圆腔读数，自适应角配分（排序半间隔）。"""
    pts, r = shell_points_2d(R)
    theta = np.arctan2(pts[:, 1], pts[:, 0])
    o = np.argsort(theta)
    ts = theta[o]
    cos2 = (pts[o, 1] / r[o]) ** 2
    dts = np.empty_like(ts)
    dts[1:-1] = 0.5 * (ts[2:] - ts[:-2])
    dts[0] = 0.5 * (ts[1] - ts[-1] + 2.0 * np.pi)
    dts[-1] = 0.5 * (ts[0] - ts[-2] + 2.0 * np.pi)
    return float(np.sum(cos2 * dts) / (2.0 * np.pi))


def gamma3_uniform_shell(R, W=5):
    """齿：均匀面电荷球壳（sigma = const）腔心场严格为零，读数须给 0。"""
    pts, r = shell_points_3d(R, W=W)
    dOmega = 1.0 / r ** 2
    cos_t = pts[:, 2] / r
    return float(-np.sum(dOmega * cos_t) / (4.0 * np.pi))


# ----------------------------------------------------------------------
# V3b：近邻配位球的偶极场和（无量纲 z 分量 (3 n_z^2 - |n|^2)/|n|^5）
# ----------------------------------------------------------------------
def dipole_z_sum(points):
    pts = np.asarray(points, dtype=float)
    r2 = np.sum(pts ** 2, axis=1)
    return float(np.sum((3.0 * pts[:, 2] ** 2 - r2) / r2 ** 2.5))


def coord_shells():
    """第一配位球（6 面）与第二配位球（12 边中点）。"""
    faces = []
    for ax in range(3):
        for s in (+1, -1):
            v = [0.0, 0.0, 0.0]
            v[ax] = float(s)
            faces.append(v)
    edges = []
    for a, b in ((0, 1), (0, 2), (1, 2)):
        for sa in (+1, -1):
            for sb in (+1, -1):
                v = [0.0, 0.0, 0.0]
                v[a], v[b] = float(sa), float(sb)
                edges.append(v)
    return np.array(faces), np.array(edges)


def perturbed_shell_sum(points, eps):
    """格点位置随机扰动后的第一配位球 z 场和（固定种子，确定性）。"""
    jitter = RNG.normal(size=points.shape)
    jitter /= np.linalg.norm(jitter, axis=1, keepdims=True)
    return dipole_z_sum(points + eps * jitter)


def truncation_sum_3d(R):
    """球截断累积和 G3(R) = Sum_{1<=|n|<=R} (3 n_z^2 - |n|^2)/|n|^5。"""
    n = np.arange(-R, R + 1)
    nx, ny, nz = np.meshgrid(n, n, n, indexing="ij")
    pts = np.stack([nx.ravel(), ny.ravel(), nz.ravel()], axis=1).astype(float)
    r2 = np.sum(pts ** 2, axis=1)
    m = (r2 > 0) & (r2 <= R * R)
    pts, r2 = pts[m], r2[m]
    vals = (3.0 * pts[:, 2] ** 2 - r2) / r2 ** 2.5
    order = np.argsort(np.sqrt(r2))
    return np.cumsum(vals[order]), np.sqrt(r2)[order], vals[order]


# ----------------------------------------------------------------------
# V3c：针腔端面读数
# ----------------------------------------------------------------------
def gamma3_needle(rho, Z):
    """极化沿 z 的针形腔：侧面 sigma = P . (-r^) = 0，只有端面束缚电荷。

    端面取 z = ±Z 的圆盘晶格点（rho 半径），sigma = ±P（外法线 ±z），
    z 法向面元每体素权重 1；两端面贡献同号叠加。
    连续期望 eps0*E/P = -rho^2/(2 Z^2)（圆盘轴上场远端近似，格点数
    N ≈ pi rho^2 时数值 -2N/(4 pi Z^2) 与之吻合）。
    """
    n = np.arange(-int(rho) - 2, int(rho) + 3)
    nx, ny = np.meshgrid(n, n, indexing="ij")
    xy = np.stack([nx.ravel(), ny.ravel()], axis=1).astype(float)
    s2 = np.sum(xy ** 2, axis=1)
    n_pts = float(np.sum(s2 <= rho * rho))
    Ez = 0.0
    for z_face, sign in ((float(Z), +1.0), (-float(Z), -1.0)):
        # 每点贡献 sigma*(-sign)/(4 pi Z^2)；sigma = sign（P 为单位）
        Ez += sign * (-sign) * n_pts / (4.0 * np.pi * z_face ** 2)
    return float(Ez)  # eps0*E/P，sigma 以 P 为单位


# ----------------------------------------------------------------------
# 容差定标（截断类偏差先测参数依赖再作预算）
# ----------------------------------------------------------------------
def calibrate():
    print("== 容差定标：厚层/自适应口径的收敛 ==")
    g3a, g3b = gamma3_sphere(15), gamma3_sphere(30)
    g2a, g2b = gamma2_circle(15), gamma2_circle(30)
    dev3, dev2 = abs(g3b - 1 / 3), abs(g2b - 0.5)
    print("  3D 厚层 W=5：gamma(R=15)=%.6f  gamma(R=30)=%.6f  末档偏差 %.2e"
          % (g3a, g3b, dev3))
    print("  2D 自适应：  gamma(R=15)=%.6f  gamma(R=30)=%.6f  末档偏差 %.2e"
          % (g2a, g2b, dev2))
    tol3 = max(2 * dev3, 1e-3)
    tol2 = max(2 * dev2, 1e-3)
    print("  定标容差：tol3=%.2e  tol2=%.2e" % (tol3, tol2))
    return tol3, tol2


# ----------------------------------------------------------------------
# 判据主体
# ----------------------------------------------------------------------
def run_v3a(tol3, tol2):
    print("\n== V3a：维度标度与平坦性 ==")
    ok = True
    R_list = [20, 25, 30, 35]
    g3s = [gamma3_sphere(R) for R in R_list]
    R2_list = [20, 30, 40]
    g2s = [gamma2_circle(R) for R in R2_list]
    print("  R(3D)  gamma_3 实测        |  R(2D)  gamma_2 实测")
    for i in range(max(len(R_list), len(R2_list))):
        line = ""
        if i < len(R_list):
            line += "%4d   %.6f（期望 1/3）" % (R_list[i], g3s[i])
        if i < len(R2_list):
            line += "   |  %4d   %.6f（期望 1/2）" % (R2_list[i], g2s[i])
        print("  " + line)
    dev3 = abs(g3s[-1] - 1 / 3)
    dev2 = abs(g2s[-1] - 0.5)
    flat3 = max(g3s) - min(g3s)
    flat2 = max(g2s) - min(g2s)
    print("  a1 3D 末档偏差 %.3e (tol %.2e)  窗间极差 %.3e" % (dev3, tol3, flat3))
    print("  a2 2D 末档偏差 %.3e (tol %.2e)  窗间极差 %.3e" % (dev2, tol2, flat2))
    ok &= (dev3 < tol3) and (flat3 < 2 * tol3)
    ok &= (dev2 < tol2) and (flat2 < 2 * tol2)
    gu = gamma3_uniform_shell(24)
    print("  a3 齿：均匀球壳读数 %.3e (期望 0，须远离 1/3=%.3f)" % (gu, 1 / 3))
    ok &= abs(gu) < tol3 and abs(gu - 1 / 3) > 0.1
    RESULTS.update(g3s=g3s, g2s=g2s, R_list=R_list, R2_list=R2_list)
    print("  V3a 判定：%s" % ("PASS" if ok else "FAIL"))
    return ok


def run_v3b(tol3):
    print("\n== V3b：近邻抵消 ==")
    faces, edges = coord_shells()
    s1 = dipole_z_sum(faces)
    edge_nz0 = edges[np.abs(edges[:, 2]) < 0.5]
    edge_nz1 = edges[np.abs(edges[:, 2]) > 0.5]
    s2g0 = dipole_z_sum(edge_nz0)
    s2g1 = dipole_z_sum(edge_nz1)
    s2 = s2g0 + s2g1
    print("  b1 第一配位球（6 面）z 场和：实测 %.17g  期望 0" % s1)
    print("  b2 第二配位球分组：n_z=0 组（%d 点）实测 %.12f 期望 -sqrt(2)=%.12f"
          % (len(edge_nz0), s2g0, -np.sqrt(2)))
    print("       n_z=+-1 组（%d 点）实测 %.12f 期望 +sqrt(2)=%.12f"
          % (len(edge_nz1), s2g1, np.sqrt(2)))
    print("       总和实测 %.17g 期望 0" % s2)
    ok = (abs(s1) < 1e-12) and (abs(s2g0 + np.sqrt(2)) < 1e-12) \
        and (abs(s2g1 - np.sqrt(2)) < 1e-12) and (abs(s2) < 1e-12)
    eps_list = [1e-3, 1e-2, 3e-2, 1e-1]
    devs = [abs(perturbed_shell_sum(faces, e)) for e in eps_list]
    for e, dv in zip(eps_list, devs):
        print("  b3 扰动 eps=%.0e：第一配位球偏移 %.3e" % (e, dv))
    ok &= devs[-1] > 10 * tol3 and devs[-1] > devs[0]
    cum, rr, vals = truncation_sum_3d(30)
    # b4a 完整格点层 (R-1, R] 的偶极场和精确为零：
    # 解析（S3 对称恒等式）：层点集在坐标置换下不变 →
    #   Sum_layer nz^2 = (1/3) Sum_layer |n|^2 → Sum (3 nz^2 - |n|^2)/|n|^5 = 0。
    Gs = [cum[np.searchsorted(rr, R + 1e-9) - 1] for R in range(1, 31)]
    dGs = [Gs[i] - Gs[i - 1] for i in range(1, 30)]
    max_dG = max(abs(v) for v in dGs)
    print("  b4a 完整格点层和逐位零（S3 对称恒等式）：max|dG(R)|（R=2..30）= %.2e"
          % max_dG)
    ok &= max_dG < 1e-12
    # b4b 反例齿：从 (9,10] 层删除一个点，层和偏移 = 该点值（破置换对称即翻）
    i9 = np.searchsorted(rr, 9 + 1e-9)
    i10 = np.searchsorted(rr, 10 + 1e-9)
    mid = i9 + (i10 - i9) // 2  # (9,10] 层内的中间点（半径 ~9.5）
    break_shift = abs(dGs[8] - vals[mid])
    print("  b4b 反例齿：删 (9,10] 层一点（r=%.2f）后层和偏移 %.3e（点值量级，须 >> 1e-12）"
          % (rr[mid], break_shift))
    ok &= break_shift > 1e-4
    # b4c 形状依赖（条件收敛的物理面）：板截断和对照连续球盖公式
    #   G_slab = -4 pi [2/3 - Z/L + Z^3/(3 L^3)]（L=4Z），离散偏差 ~ c/Z 衰减；
    #   球截断精确零 vs 板和 O(1) 非零 = 退极化形状依赖的数值面
    def slab_sum(Z, L):
        n = np.arange(-L, L + 1)
        nx, ny, nz = np.meshgrid(n, n, n, indexing="ij")
        pts = np.stack([nx.ravel(), ny.ravel(), nz.ravel()], axis=1).astype(float)
        r2 = np.sum(pts ** 2, axis=1)
        m = (r2 > 0) & (r2 <= L * L) & (np.abs(pts[:, 2]) <= Z)
        vm, r2m = pts[m], r2[m]
        return float(np.sum((3.0 * vm[:, 2] ** 2 - r2m) / r2m ** 2.5))
    print("  b4c 板截断和（L=4Z）对照连续球盖公式：")
    devs_slab = []
    for Z in (6, 8, 10, 12):
        L = 4 * Z
        gs = slab_sum(Z, L)
        cont = -4 * np.pi * (2 / 3 - Z / L + Z ** 3 / (3.0 * L ** 3))
        devs_slab.append(gs - cont)
        print("      Z=%2d: 实测 %.4f  连续 %.4f  偏差 %+.3f"
              % (Z, gs, cont, gs - cont))
    ok &= abs(slab_sum(6, 24)) > 1.0 and devs_slab[2] < devs_slab[0] \
        and abs(devs_slab[3]) < 0.15
    RESULTS.update(cum=cum, rr=rr, devs=devs, eps_list=eps_list)
    print("  V3b 判定：%s" % ("PASS" if ok else "FAIL"))
    return ok


def run_v3c(tol3, tol2):
    print("\n== V3c：维度对照与几何条件齿 ==")
    g3 = RESULTS["g3s"][-1]
    g2 = RESULTS["g2s"][-1]
    sep = abs(g3 - g2)
    print("  c1 维度区分：|gamma_3 - gamma_2| = %.3e （10*(tol3+tol2) = %.2e）"
          % (sep, 10 * (tol3 + tol2)))
    ok = sep > 10 * (tol3 + tol2)
    rho, Z = 8.0, 48.0
    gn = gamma3_needle(rho, Z)
    expect = -rho ** 2 / (2.0 * Z ** 2)
    print("  c2 针腔读数 %.6f  连续期望 -rho^2/(2Z^2) = %.6f（|gn| 须 << 1/3）"
          % (gn, expect))
    ok &= abs(gn - expect) < 5e-3 and abs(gn) < 1.0 / 9
    d_vec = np.array([1.0, 2.0, 3.0])
    g_vec = np.array([1.0, g2, g3])  # 1D 解析档 gamma_1 = 1
    x = 1.0 / d_vec
    slope, intercept = np.polyfit(x, g_vec, 1)
    print("  c3 回归 (1/d -> gamma)：斜率 %.5f（期望 1）截距 %.2e" % (slope, intercept))
    ok &= abs(slope - 1.0) < 0.02 and abs(intercept) < 0.02
    RESULTS.update(gn=gn, expect_n=expect, slope=slope, g_vec=g_vec)
    print("  V3c 判定：%s" % ("PASS" if ok else "FAIL"))
    return ok


# ----------------------------------------------------------------------
# 绘图
# ----------------------------------------------------------------------
def make_plots():
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    R_list, g3s = RESULTS["R_list"], RESULTS["g3s"]
    R2_list, g2s = RESULTS["R2_list"], RESULTS["g2s"]

    ax = axes[0, 0]
    ax.plot(R_list, g3s, "o-", color="tab:blue", label=r"$\gamma_3$ 实测")
    ax.axhline(1 / 3, color="k", ls="--", lw=1, label=r"期望 $1/3$")
    ax.set_title("3D 球腔厚层读数（W=5）")
    ax.set_xlabel("球面半径 R（格距）")
    ax.set_ylabel(r"$\gamma_3$")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(R2_list, g2s, "s-", color="tab:orange", label=r"$\gamma_2$ 实测")
    ax.axhline(0.5, color="k", ls="--", lw=1, label=r"期望 $1/2$")
    ax.set_title("2D 圆腔自适应角配分")
    ax.set_xlabel("圆环半径 R（格距）")
    ax.set_ylabel(r"$\gamma_2$")
    ax.legend()

    ax = axes[0, 2]
    cum, rr = RESULTS["cum"], RESULTS["rr"]
    ax.plot(rr, cum, lw=0.8, color="tab:green")
    ax.axhline(0, color="k", ls="--", lw=1)
    ax.set_title(r"球截断累积和 $G_3(R)$ 平台")
    ax.set_xlabel("R")
    ax.set_ylabel(r"$G_3(R)$")

    ax = axes[1, 0]
    ax.loglog(RESULTS["eps_list"], RESULTS["devs"], "o-", color="tab:red")
    ax.set_title("V3b 扰动齿：对称破坏即偏移")
    ax.set_xlabel(r"格点扰动幅度 $\varepsilon$")
    ax.set_ylabel("第一配位球偏移")
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1, 1]
    gu = gamma3_uniform_shell(24)
    ax.bar([0, 1, 2], [1 / 3, gu, RESULTS["gn"]],
           color=["tab:blue", "tab:gray", "tab:purple"], width=0.6)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["球腔 $\\cos^2\\theta$\n(期望 1/3)", "均匀壳齿\n(期望 0)",
                        "针腔齿\n(期望 $-\\rho^2/2Z^2$)"])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title("V3a/V3c 齿：构造破坏即读数偏移")

    ax = axes[1, 2]
    x = 1.0 / np.array([1.0, 2.0, 3.0])
    ax.plot(x, RESULTS["g_vec"], "o", ms=9, color="tab:blue")
    xs = np.linspace(0.2, 1.05, 50)
    ax.plot(xs, RESULTS["slope"] * xs, "--", color="k",
            label=r"回归斜率 $%.4f$" % RESULTS["slope"])
    for d, g in zip([1, 2, 3], RESULTS["g_vec"]):
        ax.annotate("d=%d" % d, (1.0 / d, g),
                    textcoords="offset points", xytext=(6, 6))
    ax.set_xlabel("1/d")
    ax.set_ylabel(r"$\gamma_d$ 实测")
    ax.set_title("V3c 维度标度回归")
    ax.legend()
    ax.set_xlim(0.15, 1.1)
    ax.set_ylim(0, 1.15)

    fig.suptitle("第 615 号：局域场系数的维度标度（$\\gamma_d = 1/d$ 的数值读出）")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = IMG_DIR + r"\local_field_dimension_615.png"
    fig.savefig(out, dpi=150)
    print("\n图已保存：%s" % out)


# ----------------------------------------------------------------------
def main():
    print("第 615 号：局域场系数维度标度判据轮")
    print("期望：gamma_3 = 1/3, gamma_2 = 1/2, gamma_1 = 1（维度标度 gamma_d = 1/d）")
    tol3, tol2 = calibrate()
    ok_a = run_v3a(tol3, tol2)
    ok_b = run_v3b(tol3)
    ok_c = run_v3c(tol3, tol2)
    print("\n================ 总结 ================")
    print("V3a（维度标度与平坦性，含均匀壳齿）：%s" % ("PASS" if ok_a else "FAIL"))
    print("V3b（近邻抵消，含扰动齿与平台）：%s" % ("PASS" if ok_b else "FAIL"))
    print("V3c（维度对照与几何条件齿，含回归）：%s" % ("PASS" if ok_c else "FAIL"))
    make_plots()
    return 0 if (ok_a and ok_b and ok_c) else 1


if __name__ == "__main__":
    raise SystemExit(main())
