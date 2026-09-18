# -*- coding: utf-8 -*-
r"""
minimal_verification_tidal_ricci_lambda.py

[GV1] 分段 (c)：潮汐项与里奇对应，以及真空条件对 lambda 的约束

设定
----
静态源取严格 1/r 场 Phi = -A/r（只在源外取值，最小半径 3），度量取
h00 = -2 Phi、h_ij = -2 lambda Phi delta_ij。潮汐张量取二阶差分；里奇按线性化公式
用显式指标和装配。源外 Phi 为调和函数（拉普拉斯恒为零），故所有读数都在真空区取。

判据
----
T1 离散误差的阶：离散拉普拉斯在固定物理半径处的相对残差随格距平方收敛（比值约 4）。
T2 分量比：轴上点的 E_xx、E_yy 与解析二阶导相符（相对偏差 < 5e-3），比值 E_xx/E_yy = -2（偏差 < 1%）。
T3 对应式：数值里奇与解析式 R_ij = -lambda delta_ij lap Phi + (1-lambda) d_i d_j Phi 相符（偏差 < 1e-5）。
T4 真空条件：源外 lambda = 1 时 |R_ij| 落到离散误差量级（比 lambda = 0 降三个数量级以上）。

运行
----
python 验证脚本/minimal_verification_tidal_ricci_lambda.py
确定性（无随机数），运行约 15 秒。
"""
import time

import numpy as np

T0 = time.time()
CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print(f"  [{tag}] {'PASS' if ok else 'FAIL'}  {detail}")


A = 1.0


def phi_field(N, h, rmin=2.5):
    x = (np.arange(N) - N / 2) * h
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    R = np.where(R < rmin, rmin, R)   # 源区用常数截断，取样只取源外
    return -A / R


def d2_axis(coord, axis):
    """严格 1/r 场在点 (coord,0,0) 处沿 axis 的二阶导"""
    r = abs(coord)
    xa = coord if axis == 0 else 0.0
    return A * (r ** 2 - 3 * xa ** 2) / r ** 5


def d1(f, axis, h):
    return (np.roll(f, -1, axis=axis) - np.roll(f, 1, axis=axis)) / (2 * h)


def d2(f, axis, h):
    return (np.roll(f, -1, axis=axis) - 2 * f + np.roll(f, 1, axis=axis)) / h ** 2


def ricci(phi, lam, h):
    hij = [[(-2.0 * lam * phi if i == j else np.zeros_like(phi)) for j in range(3)] for i in range(3)]
    hh = -(-2.0 * phi) + sum(hij[i][i] for i in range(3))
    R = np.zeros((3, 3) + phi.shape)
    for i in range(3):
        for j in range(3):
            lap_hij = sum(d1(d1(hij[i][j], a, h), a, h) for a in range(3))
            t1 = sum(d1(d1(hij[j][k], k, h), i, h) for k in range(3))
            t2 = sum(d1(d1(hij[i][k], k, h), j, h) for k in range(3))
            dij_h = d1(d1(hh, i, h), i, h) if i == j else d1(d1(hh, j, h), i, h)
            R[i][j] = 0.5 * (lap_hij - t1 - t2 + dij_h)
    return R


print("=" * 72)
print("最小验证 GV1：潮汐项与里奇对应，真空条件对 lambda 的约束")
print("=" * 72)

# ---------------------------------------------------------------- T1
print("\n[T1] 离散误差的阶（源外 1/r 场为调和场）")
res = {}
for N, h in ((128, 0.5), (256, 0.25)):
    phi = phi_field(N, h)
    lap_d = sum(d2(phi, a, h) for a in range(3))
    c = N // 2
    idx = [c + int(round(r / h)) for r in (6.0, 8.0, 10.0)]
    res[N] = max(abs(lap_d[i, c, c]) for i in idx)
r = res[128] / res[256]
print(f"      h = 0.5 残差 {res[128]:.3e}；h = 0.25 残差 {res[256]:.3e}；比值 {r:.3f}（期望约 4）")
check("T1", 3.0 < r < 5.0, f"残差随格距平方收敛，比值 {r:.3f}（判据 4 ± 25%）")

# ---------------------------------------------------------------- T2
print("\n[T2] 分量比（轴上点，源外）")
N, h = 256, 0.25
phi = phi_field(N, h)
c = N // 2
worst2, rat = 0.0, None
for rp in (6.0, 8.0, 10.0):
    i = c + int(round(rp / h))
    Exx = d2(phi, 0, h)[i, c, c]
    Eyy = d2(phi, 1, h)[i, c, c]
    ax, ay = d2_axis(rp, 0), d2_axis(rp, 1)
    worst2 = max(worst2, abs(Exx - ax) / abs(ax), abs(Eyy - ay) / abs(ay))
    rat = Exx / Eyy
print(f"      数值与解析二阶导的最大相对偏差 {worst2:.2e}；r = 10 处比值 {rat:.4f}（解析 -2）")
check("T2", worst2 < 5e-3 and abs(rat + 2.0) < 0.02,
      f"偏差 {worst2:.1e}（判据 < 5e-3），比值 {rat:.4f}（判据 -2 ± 1%）")

# ---------------------------------------------------------------- T3
print("\n[T3] 对应式：数值里奇对解析式")
lam = 0.7
phi = phi_field(N, h)
R = ricci(phi, lam, h)
lapPhi = sum(d1(d1(phi, a, h), a, h) for a in range(3))
worst3, amp = 0.0, 0.0
for (i, j) in [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2)]:
    dij_phi = d1(d1(phi, i, h), i, h) if i == j else d1(d1(phi, j, h), i, h)
    R_ana = -lam * (1.0 if i == j else 0.0) * lapPhi + (1.0 - lam) * dij_phi
    sl = slice(c + 24, c + 44)
    if i == j:
        num, ana = R[i][j][sl, c, c], R_ana[sl, c, c]
    else:
        num, ana = R[i][j][sl, c + 24, c], R_ana[sl, c + 24, c]
    den = max(float(np.max(np.abs(ana))), 1e-30)
    worst3 = max(worst3, float(np.max(np.abs(num - ana))) / den)
    amp = max(amp, float(np.max(np.abs(ana))))
print(f"      数值里奇与解析式的最大相对偏差 {worst3:.3e}（解析量级 {amp:.2e}）")
check("T3", worst3 < 1e-5, f"偏差 {worst3:.1e}（判据 < 1e-5）")

# ---------------------------------------------------------------- T4
print("\n[T4] 真空条件：lambda = 1 时源外里奇为零")
norms = {}
for lam_t in (0.0, 1.0):
    Rt = ricci(phi, lam_t, h)
    vals = [float(np.max(np.abs(Rt[i][j][c + 24:c + 44, c, c])))
            for (i, j) in [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2)]]
    norms[lam_t] = max(vals)
floor = float(np.max(np.abs(lapPhi[c + 24:c + 44, c, c])))
print(f"      源外 |R_ij| 最大：lambda = 0 给 {norms[0.0]:.3e}，lambda = 1 给 {norms[1.0]:.3e}")
print(f"      离散底噪（同区间的复合拉普拉斯）{floor:.3e}；lambda = 1 的残差与底噪之比 {norms[1.0] / floor:.2f}")
print(f"      下降倍数 {norms[0.0] / norms[1.0]:.1e}")
check("T4", norms[0.0] / norms[1.0] > 50 and norms[1.0] <= 3 * floor,
      f"lambda = 1 时残差 {norms[1.0]:.1e} 落到底噪 {floor:.1e} 的 3 倍以内（降 {norms[0.0] / norms[1.0]:.0e} 倍）")

print("\n" + "=" * 72)
n_pass = sum(1 for _, ok in CHECKS if ok)
print(f"判据汇总：{n_pass}/{len(CHECKS)} 通过")
for tag, ok in CHECKS:
    print(f"  {tag}: {'PASS' if ok else 'FAIL'}")
print(f"耗时 {time.time() - T0:.1f} s")