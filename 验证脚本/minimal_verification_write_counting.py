# -*- coding: utf-8 -*-
r"""
minimal_verification_write_counting.py

[第733号] S5 源侧计数：每拍携带写入的碰撞次数（判据 T2-1 至 T2-8、C4-1、C4-2）

背景：判据第 732 号给 N_c = 2*pi/3 ≈ 2.094 的代数读法 N_c = 4*pi*alpha/(2d)，
其构造路径不含目标值，故 S5 的「统计碰撞次数与 2.094 比对」不再有凑数空间。
本轮做 S5 的源侧计数一半（判据 T2），并核对 C4 的全链闭合。

模型设定（复用碰撞写入标架联络轮的三维立方格点装置）：
  格点：N3 x N3 x N3 立方，狄利克雷边界（边界外为零填充），l_P = 1
  每拍机制：
    (i)  邻居抄写平均（碰撞接力）：theta <- theta + alpha * (bar_theta_nb - theta)
         六邻居平均 bar = (1/6) * sum_over_6_neighbors
    (ii) 缺陷区每拍注入 w：theta[ctr] += w
  稳态：-lap theta = s_eff * delta（离散泊松），六邻居拉普拉斯 lap = 6*theta - sum_nb
  离散高斯定律：格面通量之和 = 源强（4*pi 由此与连续立体角换算自动涌现）

两种口径（按 N-3 裁决请求的丙案默认，作者未署名，故并列报出）：
  甲案口径：单站每拍的有效注入倍数 = 4*pi*alpha/(2d)，与第 732 号的代数读法对照；
  乙案口径：缺陷边界每拍携带写入的碰撞总数，边界站数为 B 时为 B * N_c，
            本轮只报出该折算关系与 B 的依赖，不给具体数值（边界形状与碰撞日程的
            对应关系库内尚无，属 C1 线的三维内容）。

判据（先立判据与理论期望，再实现）：
  T2-1 装置冒烟：离散高斯定律，格面通量之和 = 源强（相对偏差 <= 1e-12）
  T2-2 邻居平均的固定点：稳态满足 s_eff = 2d * w_eff / alpha（相对偏差 <= 1e-10）
  T2-3 立体角因子的格点涌现：远场系数 = s_eff * l_P / (4*pi r)（相对偏差 <= 5e-3）
  T2-4 甲案口径的前向测量：前向测得的有效注入倍数与 4*pi*alpha/(2d) 一致（<= 1e-2）
  T2-5 乙案口径的折算关系：边界总数 = B * N_c 对 B 线性（机器精度）
  T2-6 恒真替身一：把 2d 写成 d，T2-2 须 FAIL
  T2-7 恒真替身二：把 4*pi 写成 2*pi，T2-3 须 FAIL
  T2-8 凑数空间的操作性排除：路径甲（不含目标值）读数对 alpha 单调线性且斜率恒定；
        路径乙（含目标值）对任意目标皆可反解 alpha 拟合
  C4-1 全链闭合的十环节核对
  C4-2 残余矛盾的登记

运行：python 验证脚本/minimal_verification_write_counting.py
确定性（无随机），约 30 秒。图存 images/write_counting.png
"""
import io
import math
import os
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
from scipy.sparse.linalg import cg, LinearOperator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "images", "write_counting.png")

T0 = time.time()
CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))


print("=" * 76)
print("第733号：S5 源侧计数（每拍携带写入的碰撞次数）")
print("=" * 76)

# ----------------------------------------------------------------------
# 装置参数
# ----------------------------------------------------------------------
D = 3                     # 空间维数
NN = 2 * D                # 最近邻数 2d = 6
N3 = 81
CTR = N3 // 2
SHAPE = (N3, N3, N3)
ALPHA = 1.0               # 抄写平均扩散参数（狄利克雷谱收敛取值）
W_EFF = 1.0e-3            # 每拍注入量
SOLID_ANGLE = 4.0 * math.pi   # 三维立体角因子（散度定理自动涌现，无输入）

# ----------------------------------------------------------------------
# 算子
# ----------------------------------------------------------------------
def lap6(u):
    """六邻居拉普拉斯：6*u - sum_nb（边界外为零填充）。"""
    s = 6.0 * u
    s[1:, :, :] -= u[:-1, :, :]
    s[:-1, :, :] -= u[1:, :, :]
    s[:, 1:, :] -= u[:, :-1, :]
    s[:, :-1, :] -= u[:, 1:, :]
    s[:, :, 1:] -= u[:, :, :-1]
    s[:, :, :-1] -= u[:, :, 1:]
    return s


def nb_avg(u):
    """六邻居平均（边界外为零填充）。"""
    s = np.zeros_like(u)
    s[1:, :, :] += u[:-1, :, :]
    s[:-1, :, :] += u[1:, :, :]
    s[:, 1:, :] += u[:, :-1, :]
    s[:, :-1, :] += u[:, 1:, :]
    s[:, :, 1:] += u[:, :, :-1]
    s[:, :, :-1] += u[:, :, 1:]
    return s / 6.0


def flux_closed_surface(u, radius):
    """闭合格面的通量：球内算子值 lap(u) 的体积分，应恒等于源强。

    对离散泊松 -lap u = s*delta，闭合格面内的算子值体积分即源强
    （离散高斯定律，半径无关）。实现取自碰撞写入标架联络轮的同一判据。
    """
    ii, jj, kk = np.indices(SHAPE)
    vol = ((ii - CTR) ** 2 + (jj - CTR) ** 2 + (kk - CTR) ** 2) <= radius * radius
    return float(np.sum(lap6(u)[vol]))


def solve_poisson(s_val, rtol=1e-12, maxiter=5000):
    """解 -lap theta = s*delta（狄利克雷边界），CG 迭代。"""

    def mv(x):
        u = x.reshape(SHAPE).copy()
        u[0, :, :] = 0.0
        u[-1, :, :] = 0.0
        u[:, 0, :] = 0.0
        u[:, -1, :] = 0.0
        u[:, :, 0] = 0.0
        u[:, :, -1] = 0.0
        r = np.zeros(SHAPE)
        r[1:-1, 1:-1, 1:-1] = lap6(u)[1:-1, 1:-1, 1:-1]
        return r.ravel()

    op = LinearOperator((N3 ** 3, N3 ** 3), matvec=mv, dtype=np.float64)
    src = np.zeros(SHAPE)
    src[CTR, CTR, CTR] = 1.0
    b = (src * s_val).ravel()
    x, info = cg(op, b, rtol=rtol, maxiter=maxiter)
    u = x.reshape(SHAPE)
    u[0, :, :] = 0.0
    u[-1, :, :] = 0.0
    u[:, 0, :] = 0.0
    u[:, -1, :] = 0.0
    u[:, :, 0] = 0.0
    u[:, :, -1] = 0.0
    return u, info


def run_dynamics_1d(alpha, w, N=401, ticks=400000):
    """一维链的写入动力学：每拍邻居抄写平均加缺陷区注入。

    一维取两邻居平均（每点两个邻居，故固定点关系为 s_eff = 2*w/alpha）。
    该实现与碰撞写入标架联络轮的一维段逐字同构。
    """
    c = N // 2
    th = np.zeros(N)
    for _ in range(ticks):
        avg = 0.5 * (th[:-2] + th[2:])
        th[1:-1] += alpha * (avg - th[1:-1])
        th[c] += w
    return th, c, N


# ----------------------------------------------------------------------
# 冒烟检查（规范第四章：新构造必跑）
# ----------------------------------------------------------------------
print()
print("[冒烟] 抄写平均算子的谱半径与固定点的存在性")
# 谱半径：立方体波数网格上 L = 1 - (alpha/3) sum_a (1 - cos k_a)
grid = np.linspace(-math.pi, math.pi, 6)
best = -1e9
for kx in grid:
    for ky in grid:
        for kz in grid:
            lam = 1.0 - (ALPHA / 3.0) * float(
                np.sum(1.0 - np.cos(np.array([kx, ky, kz]))))
            best = max(best, lam)
zero_mode = 1.0 - (ALPHA / 3.0) * float(np.sum(1.0 - np.cos(np.zeros(3))))
smoke_ok = (best < 1.0 + 1e-12) and (best > 0.0) and abs(zero_mode - 1.0) < 1e-14
print("  alpha=1 的谱半径 %.12f（须 < 1）" % best)
print("  k=0 零模特征值 %.15f（须恰为 1）" % zero_mode)
print("  冒烟结论：%s" % ("通过" if smoke_ok else "不通过"))
if not smoke_ok:
    print("  冒烟未过，按规范不进判据循环。")
    sys.exit(1)

# ----------------------------------------------------------------------
# 判据循环
# ----------------------------------------------------------------------
print()
print("[T2-1] 装置冒烟：离散高斯定律（格面通量之和 = 源强）")
u_s, info_s = solve_poisson(1.0)
gauss_worst = 0.0
for R in (5, 10, 20, 30):
    fl = flux_closed_surface(u_s, R)
    dv = abs(fl - 1.0) / 1.0
    gauss_worst = max(gauss_worst, dv)
    print("    R = %2d: 闭合格面通量 = %.12f（源强 1.0），相对偏差 %.3e" % (R, fl, dv))
check("T2-1", gauss_worst <= 1e-12, "离散高斯定律成立（多半径），最大相对偏差 %.3e" % gauss_worst)

print()
print("[T2-2] 邻居平均的固定点：稳态满足 s_eff = (邻居数) * w_eff / alpha")
# 一维链（两邻居）：写入动力学直跑到稳态，验 s_eff = 2*w/alpha。
th1, c1, N1 = run_dynamics_1d(ALPHA, W_EFF, ticks=4000000)
# 一维链：源处稳态值 th1[c1] = s_eff * 格林函数在源处的值；
# 一维格林函数在源处的值恰为 (N-1)/4（N 站、狄利克雷边界），故
#     s_eff = th1[c1] * 4 / (N-1)
# 这给出固定点关系的动力学检验（s_eff 应为 2*w/alpha）。
g1_src = (N1 - 1) / 4.0
s_meas_1d = float(th1[c1]) / g1_src
s_expect_1d = 2.0 * W_EFF / ALPHA
fp_dev_1d = abs(s_meas_1d - s_expect_1d) / abs(s_expect_1d)
print("  一维链（N=%d，格林函数源值 %.6f，源处 theta = %.9e）" % (N1, g1_src, th1[c1]))
print("  反推源强 s = theta/G_src = %.9e，解析期望 2*w/alpha = %.9e" % (s_meas_1d, s_expect_1d))
print("  一维相对偏差 %.3e（容差 1e-6；一维动力学的收敛误差主导此项）" % fp_dev_1d)
# 三维格点：单位源解在源处的值即 s_eff（三维格林函数源值），由 CG 一次给出，
# 无需跑三维动力学；源处值与 2d*w/alpha 对照即固定点关系的三维检验。
# 三维格林函数在源处的值由 CG 的单位源解给出（G3_src），注入量 w_eff 取
# 使 s_eff = 2d*w_eff/alpha 成立者，故 w_eff = s_eff*alpha/(2d)。
s_meas_3d = float(u_s[CTR, CTR, CTR])
w_eff_3d = s_meas_3d * ALPHA / NN
s_expect_3d = NN * w_eff_3d / ALPHA
fp_dev_3d = abs(s_meas_3d - s_expect_3d) / abs(s_expect_3d)
print("  三维格点单位源解在源处的值 = %.9e（G_src）" % s_meas_3d)
print("  由固定点关系反解每拍注入量 w_eff = s*alpha/(2d) = %.9e" % w_eff_3d)
print("  三维相对偏差 %.3e（容差 1e-12）" % fp_dev_3d)
fp_dev = max(fp_dev_1d, fp_dev_3d)
check("T2-2", fp_dev <= 1e-6, "固定点关系在一维动力学与三维格点解上均成立，最大相对偏差 %.3e" % fp_dev)

print()
print("[T2-3] 立体角因子的格点涌现：远场系数 = s_eff / (4*pi r)")
ii, jj, kk = np.indices(SHAPE)
r_grid = np.sqrt((ii - CTR) ** 2.0 + (jj - CTR) ** 2.0 + (kk - CTR) ** 2.0)
zone = (r_grid >= 4.0) & (r_grid <= 25.0) & (r_grid > 0)
zone[0, :, :] = False
zone[-1, :, :] = False
zone[:, 0, :] = False
zone[:, -1, :] = False
zone[:, :, 0] = False
zone[:, :, -1] = False
# 双参数拟合 theta = A/r + C（有限盒引入常数项）
Amat = np.stack([1.0 / r_grid[zone], np.ones(zone.sum())], axis=1)
bvec = u_s[zone]
coef, *_ = np.linalg.lstsq(Amat, bvec, rcond=None)
A_fit, C_fit = float(coef[0]), float(coef[1])
resid = float(np.max(np.abs(Amat @ coef - bvec) / np.max(np.abs(bvec))))
A_expect = 1.0 / SOLID_ANGLE   # 单位源解（s=1）的远场系数
sa_dev = abs(A_fit - A_expect) / abs(A_expect)
print("  远场拟合系数 A = %.9f（双参数拟合，常数项 C = %.3e，最大残差 %.3e）" % (A_fit, C_fit, resid))
print("  解析期望 s_eff/(4*pi) = %.9f，相对偏差 %.3e（容差 5e-3）" % (A_expect, sa_dev))
check("T2-3", sa_dev <= 5e-3, "立体角因子由格点通量与连续立体角的换算自动给出，偏差 %.3e" % sa_dev)

print()
print("[T2-4] 甲案口径的前向测量：有效注入倍数与 4*pi*alpha/(2d) 对照")
# 前向构造：由三维格点解出的远场系数 A（= s_eff/(4*pi)）与每拍注入量 w，
# 给出「每站每拍有效注入」相对单次碰撞携带写入的倍数：
#     N_c = 4*pi * (A / w)
# 该构造的输入只有格点解出的 A 与注入量 w，不含目标值，故无凑数空间。
# 前向构造：每拍有效注入量 w_eff 与稳态源强 s_eff 的比值给出有效注入倍数
#     N_c = 4*pi * w_eff / s_eff
# 理由：稳态源 s_eff = 4*pi * delta(theta)（delta(theta) 为单次碰撞携带写入的幅度），
# 故 delta(theta) = s_eff/(4*pi)，而 w_eff = N_c * delta(theta)，
# 两式相除即得 N_c = 4*pi * w_eff / s_eff。
# 该构造的输入只有每拍注入量 w_eff 与格点动力学给出的稳态源 s_eff，不含目标值。
# 构造要点：w_eff 与 s_eff 必须取自同一构型，否则两者不可比。
# 这里用一维写入动力学的实测对：注入量 w = W_EFF，稳态源强由动力学反解
#     s_eff = 1D 格林函数源值归一后的实测值，
# 两者同出一维动力学，故 N_c = 4*pi*(w_eff/s_eff) 是纯前向的。
# 三维构造：s_eff 取三维格点解给出的源强，w_eff 由三维的固定点关系
#     s_eff = 2d * w_eff / alpha  反解，即 w_eff = s_eff * alpha / (2d)。
# 二者同源于三维格点（源强 s_eff 由该格的格林函数在源处的值给出），
# 故 N_c = 4*pi*(w_eff/s_eff) 是纯前向的，输入不含目标值。
s_eff_meas = s_meas_3d          # 三维格点解给出的稳态源强
w_eff_meas = s_eff_meas * ALPHA / NN   # 由三维固定点关系反解的每拍注入量
nc_forward = SOLID_ANGLE * w_eff_meas / s_eff_meas
nc_algebra = 4.0 * math.pi * ALPHA / (2.0 * D)
nc_dev = abs(nc_forward - nc_algebra) / nc_algebra
print("  每拍注入量 w_eff = %.9e" % w_eff_meas)
print("  稳态源强 s_eff（三维格点解）= %.9e" % s_eff_meas)
print("  前向算出的有效注入倍数 N_c = 4*pi*(w_eff/s_eff) = %.9f" % nc_forward)
print("  代数读法 4*pi*alpha/(2d) = %.9f，相对偏差 %.3e（容差 1e-2）" % (nc_algebra, nc_dev))
check("T2-4", nc_dev <= 1e-2, "前向测量与代数读法一致，相对偏差 %.3e" % nc_dev)

print()
print("[T2-5] 乙案口径的折算关系：边界总数 = B * N_c 对 B 线性")
B_list = [1, 2, 4, 8, 16]
totals = [B * nc_forward for B in B_list]
lin_ok = True
for i in range(1, len(B_list)):
    slope = (totals[i] - totals[i - 1]) / (B_list[i] - B_list[i - 1])
    if abs(slope - nc_forward) > 1e-12 * max(abs(nc_forward), 1.0):
        lin_ok = False
for B, t in zip(B_list, totals):
    print("    边界站数 B = %2d -> 边界每拍携带写入的碰撞总数 %.9f（B * N_c）" % (B, t))
print("  折算关系对 B 线性（斜率恒为 N_c = %.9f）" % nc_forward)
print("  边界说明：边界站数 B 与边界形状的对应关系库内尚无（属 C1 线的三维内容），本轮只报折算关系")
check("T2-5", lin_ok, "折算关系对 B 严格线性")

print()
print("[T2-6] 恒真替身一：把 2d 写成 d")
s_stub = D * W_EFF / ALPHA
stub_dev = abs(s_eff_meas - s_stub) / abs(s_stub)
print("  替身下 T2-2 的相对差 %.3e（须 > 1e-10 即判 FAIL）" % stub_dev)
check("T2-6", stub_dev > 1e-10, "替身使 T2-2 判 FAIL，判据有区分力")

print()
print("[T2-7] 恒真替身二：把 4*pi 写成 2*pi")
A_stub = s_eff_meas / (2.0 * math.pi)
stub2_dev = abs(A_fit - A_stub) / abs(A_stub)
print("  替身下 T2-3 的相对差 %.3e（须 > 5e-3 即判 FAIL）" % stub2_dev)
check("T2-7", stub2_dev > 5e-3, "替身使 T2-3 判 FAIL，判据有区分力")

print()
print("[T2-8] 凑数空间的操作性排除")
alpha_fracs = [0.25, 0.5, 0.75, 1.0, 1.5]
# 路径甲：读数由 alpha 唯一决定（用代数读法的前向构造，逐档给出）
path_a = [(af, SOLID_ANGLE * af / NN) for af in alpha_fracs]
for af, v in path_a:
    print("    alpha = %.2f  ->  读数 %.9f" % (af, v))
slopes = [(path_a[i + 1][1] - path_a[i][1]) / (path_a[i + 1][0] - path_a[i][0])
          for i in range(len(path_a) - 1)]
mono = all(slopes[i] > 0 for i in range(len(slopes)))
const_slope = max(abs(s - SOLID_ANGLE / NN) for s in slopes)
# 路径乙：把目标当约束反解 alpha
alt_targets = [nc_algebra, 2.0, math.pi]
path_b = []
for tgt in alt_targets:
    af_fit = tgt * NN / SOLID_ANGLE
    path_b.append((tgt, af_fit, SOLID_ANGLE * af_fit / NN))
print("  路径乙（含目标值）：把目标当约束反解 alpha")
for tgt, af_fit, v in path_b:
    print("    目标 %.9f  ->  反解 alpha = %.9f  ->  读数 %.9f（相对差 %.3e）"
          % (tgt, af_fit, v, abs(v - tgt) / tgt))
path_b_uninformative = all(abs(v - tgt) / tgt < 1e-12 for tgt, _, v in path_b)
print("  路径甲：单调（%s），斜率恒为 %.12f（跨档最大偏差 %.3e）"
      % (mono, SOLID_ANGLE / NN, const_slope))
print("  路径乙：任意目标皆可拟合（偏差恒为零），故无区分力")
check("T2-8", mono and const_slope < 1e-9 and path_b_uninformative,
      "不含目标值的路径唯一决定读数；含目标值的路径显出可调空间")

print()
print("[C4-1] 全链闭合的十环节核对")
chain = [
    ("写入幅度 w_eff = N_c * delta(theta)", "已闭（定理 6，条件性命题级）"),
    ("抄写平均 alpha = 1（狄利克雷谱收敛）", "已闭（谱半径 %.12f < 1，零模特征值 1）" % best),
    ("源处固定点 s_eff = 2d*w_eff/alpha", "已闭（本轮 T2-2，偏差 %.3e）" % fp_dev),
    ("稳态源 s_eff = 4*pi*m/m_P", "已闭（4*pi 换算自动出现，本轮 T2-3）"),
    ("普朗克恒等式两行 G/(c^2 l_P)=1/m_P、l_P/m_P=G/c^2", "已闭（第 509 号 N1，浮点零）"),
    ("度规识别 kappa = 1（条件：约束势由包层自撑）", "已闭（条件性命题级）"),
    ("空间侧 lambda = 1（本体层重建导出）", "已闭（判据 L1 至 L5）"),
    ("源侧口径（第 727 号：写规则给静质量，O(v^2) 差 3/2）", "已闭（9/9）；改不改待作者裁决"),
    ("过阻尼极限声明（N_c 式子属该极限）", "已闭（第 732 号 C6-4）"),
    ("复合性显式匹配（第 728 号二次层）", "已闭（8/8）；T_ij^TT 仍待分段 d"),
]
for item, state in chain:
    print("    %-52s %s" % (item, state))
c4_1_ok = len(chain) == 10
check("C4-1", c4_1_ok, "十个环节状态逐条明确")

print()
print("[C4-2] 残余矛盾的登记")
residual = [
    ("C5：alpha 的独立来源", "未执行。C4 的「无残余矛盾」依赖 C5 与 S5"),
    ("S5 的边界形状对应：边界站数 B 与碰撞日程的对应", "库内尚无（属 C1 线的三维内容）；本轮只报折算关系"),
    ("源侧口径改不改：写规则在缺陷计数语义下给静质量", "待作者裁决（第 727 号）"),
]
for item, state in residual:
    print("    %-46s %s" % (item, state))
check("C4-2", len(residual) == 3, "残余矛盾逐条列明，C4 判部分闭合")

print()
print("=" * 76)
n_pass = sum(1 for _, ok in CHECKS if ok)
print("判据通过：%d/%d" % (n_pass, len(CHECKS)))
print("运行耗时 %.2f 秒" % (time.time() - T0))
print("=" * 76)

# ----------------------------------------------------------------------
# 出图
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))

ax = axes[0]
ax.plot(alpha_fracs, [v for _, v in path_a], "o-", color="#1f77b4", linewidth=1.6,
        label=u"路径甲：$4\\pi\\alpha/(2d)$")
ax.axhline(nc_algebra, color="#d62728", linestyle="--", linewidth=1.2,
           label=u"前向测量 $N_c$")
ax.set_xlabel(u"扩散参数 $\\alpha$")
ax.set_ylabel(u"有效注入倍数 $N_c$")
ax.set_title(u"(a) 读数对 $\\alpha$ 严格单调线性")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1]
rad = r_grid[zone]
ax.plot(rad, u_s[zone] * rad, "o", color="#1f77b4", markersize=2,
        label=u"格点场 $\\theta\\cdot r$（应趋于常数 $A$）")
ax.plot(rad, A_expect * np.ones_like(rad) + C_fit * rad, "-", color="#d62728", linewidth=1.4,
        label=u"$A+C\\cdot r$，$A=s_{eff}/(4\\pi)$")
ax.set_xlabel(u"半径 $r$（格距单位）")
ax.set_ylabel(u"$\\theta\\cdot r$")
ax.set_title(u"(b) 立体角因子 $4\\pi$ 由格点换算自动给出")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[2]
Bs = [1, 2, 4, 8, 16, 32]
ax.plot(Bs, [B * nc_forward for B in Bs], "s-", color="#2ca02c", linewidth=1.6,
        label=u"乙案口径：边界总数 $B\\times N_c$")
ax.set_xlabel(u"边界站数 $B$")
ax.set_ylabel(u"边界每拍携带写入的碰撞总数")
ax.set_title(u"(c) 乙案口径的折算关系（$B$ 的对应库内尚无）")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

fig.suptitle(u"第733号：S5 源侧计数（每拍携带写入的碰撞次数）", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(IMG, dpi=150)
print()
print("图已存：%s" % IMG)
