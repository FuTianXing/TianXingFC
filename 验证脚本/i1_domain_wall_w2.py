# -*- coding: utf-8 -*-
r"""
i1_domain_wall_w2.py

[接口判定 I-1] 复结构畴壁面与标架侧 Z_2 的对偶面是否重合：二维最小模型

定位
----
接口注记 I-1（内部草稿/注记-费米子与标架联络通道：双覆盖Z2的接口判定 §三、§四）提出：
把复结构符号写成 Z_2 值场 sigma(x)，问它的畴壁面与标架侧双覆盖标签（旋转结构的
第二 Stiefel-Whitney 类的对偶面）是否重合。本脚本在二维格点上做最小模型。

模型
----
标架侧：二维补丁，中心放一个奇绕数向错（强度 q = 1），链接角取极角增量的主值乘 q
（沿用 T1b-2 的构造）。回路读数 A = 提升积的符号：沿回路把 SU(2) 提升相乘，
A = 实数部分（+1 或 -1）。

畴壁侧：一条从端点出发、走到补丁边界的壁（射线或折线）。回路读数 B = 回路与该壁
相交次数的奇偶（相交为 -1，不相交为 +1）。

判据（读数取配对形式，消掉提升代表的整体符号约定）
----
J1 端点重合时两读数一致（壁方向无关）：壁端点取在向错核心，5 个方向（含折线），
   40 条回路两两配对的相对符号一致（偏差为 0）。
J2 负对照：壁端点平移 3 格离开核心，存在回路使两读数不一致（列出计数与一例）。
J3 壁的形状无关：端点同在核心时，折线壁与直射线壁给出同一组读数（逐回路一致）。
J4 回路连续形变：半径 1 至 4 的回路两读数各自不变。

运行
----
python 验证脚本/i1_domain_wall_w2.py
确定性（随机数种子固定 20260918），运行约 2 秒。
"""
import time

import numpy as np

T0 = time.time()
CHECKS = []
RNG = np.random.default_rng(20260918)

L = 25
CORE = (12, 12)


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print(f"  [{tag}] {'PASS' if ok else 'FAIL'}  {detail}")


def wrap_pi(x):
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def polar_links(L, core, q):
    """极角增量的主值乘 q，作为 x、y 链接角（平面补丁）"""
    g = np.meshgrid(np.arange(L), np.arange(L), indexing='ij')
    ang = np.arctan2(g[1] - core[1], g[0] - core[0])
    tx = np.zeros_like(ang)
    ty = np.zeros_like(ang)
    tx[:-1, :] = wrap_pi(ang[1:, :] - ang[:-1, :])
    ty[:, :-1] = wrap_pi(ang[:, 1:] - ang[:, :-1])
    return q * tx, q * ty


def su2_z(ang):
    """绕 z 轴的 SU(2) 提升，输入为标量角"""
    h = float(ang) / 2.0
    c, s = np.cos(h), np.sin(h)
    return np.array([[c + 1j * s, 0.0 + 0.0j], [0.0 + 0.0j, c - 1j * s]])


def rect_steps(ci, cj, r):
    s = []
    s += [(ci - r + k, cj - r, 'x', +1) for k in range(2 * r)]
    s += [(ci + r, cj - r + k, 'y', +1) for k in range(2 * r)]
    s += [(ci + r - 1 - k, cj + r, 'x', -1) for k in range(2 * r)]
    s += [(ci - r, cj + r - 1 - k, 'y', -1) for k in range(2 * r)]
    return s


def reading_A(tx, ty, steps):
    """提升积的符号（配对使用时与代表约定无关）"""
    U = np.eye(2, dtype=complex)
    for (i, j, d, o) in steps:
        th = tx[i, j] if d == 'x' else ty[i, j]
        U = U @ su2_z(o * th)
    return float(np.real(U[0, 0]))


def cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


def seg_cross(p1, p2, q1, q2):
    r = (p2[0] - p1[0], p2[1] - p1[1])
    s = (q2[0] - q1[0], q2[1] - q1[1])
    den = cross2(r, s)
    if abs(den) < 1e-14:
        return False
    w = (q1[0] - p1[0], q1[1] - p1[1])
    t = cross2(w, s) / den
    u = cross2(w, r) / den
    return (0.0 <= t <= 1.0) and (0.0 <= u <= 1.0)


def reading_B(steps, wall):
    """回路与壁相交次数的奇偶"""
    n = 0
    for (i, j, d, o) in steps:
        p1 = (float(i), float(j))
        p2 = (float(i + 1), float(j)) if d == 'x' else (float(i), float(j + 1))
        for k in range(len(wall) - 1):
            if seg_cross(p1, p2, wall[k], wall[k + 1]):
                n += 1
    return float((-1) ** n)


def ray_wall(start, ang_deg, length=3 * L):
    a = np.deg2rad(ang_deg)
    return [start, (start[0] + length * np.cos(a), start[1] + length * np.sin(a))]


print("=" * 72)
print("最小验证 I-1：复结构畴壁与标架侧 Z_2 的对偶面（二维最小模型）")
print("=" * 72)

tx, ty = polar_links(L, CORE, 1.0)
loops = [((12, 12, r), rect_steps(12, 12, r)) for r in (1, 2, 3, 4)]
for (ci, cj) in [(8, 12), (16, 12), (12, 8), (12, 16)]:
    for r in (1, 2):
        loops.append(((ci, cj, r), rect_steps(ci, cj, r)))
for (ci, cj) in [(8, 8), (16, 16), (8, 16), (16, 8)]:
    for r in (1, 2, 3):
        loops.append(((ci, cj, r), rect_steps(ci, cj, r)))
A = {key: reading_A(tx, ty, s) for key, s in loops}

# ---------------------------------------------------------------- J1
print("\n[J1] 壁端点重合于核心：两读数配对一致，且与壁方向无关")
worst = 0
for ang in (20.0, 55.0, 110.0, 200.0, 300.0):
    wall = ray_wall((float(CORE[0]), float(CORE[1])), ang)
    B = {key: reading_B(s, wall) for key, s in loops}
    keys = list(A)
    for k1 in keys:
        for k2 in keys:
            worst = max(worst, abs(A[k1] * A[k2] - B[k1] * B[k2]))
print(f"      5 个方向 x {len(loops)} 条回路的配对相对符号最大偏差 {worst:.1e}")
check("J1", worst < 1e-12, f"两读数在所有回路上同真同假（偏差 {worst:.1e}，判据 < 1e-12）")

# ---------------------------------------------------------------- J2
print("\n[J2] 负对照：壁端点平移 3 格离开核心")
mismatch, example = 0, None
wall_bad = ray_wall((float(CORE[0]), float(CORE[1]) + 3.7), 55.0)
B_bad = {key: reading_B(s, wall_bad) for key, s in loops}
for key, s in loops:
    if A[key] != B_bad[key]:
        mismatch += 1
        if example is None:
            example = (key, A[key], B_bad[key])
print(f"      不一致回路 {mismatch}/{len(loops)} 条，一例：回路 {example[0]} 给 A = {example[1]:+.0f}、B = {example[2]:+.0f}")
check("J2", mismatch > 0, f"端点离开核心后两读数解耦（不一致 {mismatch} 条）")

# ---------------------------------------------------------------- J3
print("\n[J3] 壁的形状无关：折线壁与直射线壁给同一组读数")
bent = [(float(CORE[0]), float(CORE[1])), (float(CORE[0]) + 0.4, float(CORE[1]) + 0.6),
        (float(CORE[0]) + 6.2, float(CORE[1]) + 0.9), (float(CORE[0]) + 6.6, float(CORE[1]) + 6.4),
        (float(CORE[0]) + 30.3, float(CORE[1]) + 6.9)]
B_bent = {key: reading_B(s, bent) for key, s in loops}
B_ray = {key: reading_B(s, ray_wall((float(CORE[0]), float(CORE[1])), 20.0)) for key, s in loops}
dev = max(abs(B_bent[k] - B_ray[k]) for k in B_bent)
print(f"      折线壁与直射线壁的逐回路最大差 {dev:.1e}")
check("J3", dev < 1e-12, f"形状无关（最大差 {dev:.1e}，判据 < 1e-12）")

# ---------------------------------------------------------------- J4
print("\n[J4] 回路连续形变：半径 1 至 4 的两读数各自不变")
dA = dB = 0.0
wall = ray_wall((float(CORE[0]), float(CORE[1])), 110.0)
for (ci, cj, rs) in [(12, 12, (1, 2, 3, 4)), (12, 5, (1, 2, 3))]:
    valsA = [reading_A(tx, ty, rect_steps(ci, cj, r)) for r in rs]
    valsB = [reading_B(rect_steps(ci, cj, r), wall) for r in rs]
    dA = max(dA, max(valsA) - min(valsA))
    dB = max(dB, max(valsB) - min(valsB))
print(f"      半径扫描下 A 的极差 {dA:.1e}，B 的极差 {dB:.1e}")
check("J4", dA < 1e-12 and dB < 1e-12, f"两读数对形变不变（极差 {dA:.1e} 与 {dB:.1e}，判据 < 1e-12）")

print("\n" + "=" * 72)
n_pass = sum(1 for _, ok in CHECKS if ok)
print(f"判据汇总：{n_pass}/{len(CHECKS)} 通过")
for tag, ok in CHECKS:
    print(f"  {tag}: {'PASS' if ok else 'FAIL'}")
print(f"耗时 {time.time() - T0:.1f} s")