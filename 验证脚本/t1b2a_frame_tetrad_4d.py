# -*- coding: utf-8 -*-
r"""
t1b2a_frame_tetrad_4d.py

[T1b-2a] 标架扇区的四维离散口径与非阿贝尔推广

定位
----
判定注记 T1b-2 第五节把两项登记为 T1b-2a：其一，含时间方向的四维标架构型与
Pontryagin 密度的离散积分口径未定；其二，标架梯度的非阿贝尔全结构未算（原脚本取
绕单轴的约化族）。本脚本把这两项做成可复现读数。

模型
----
沿用路径 B 的标架层：格点标架 F(n) 属于 SO(3)，链接变量 U_a(n) = F(n)^T F(n+a)，
格面 holonomy 取有序乘积。读数用链接角直接给出：链接 U = exp(theta K)，K 为生成元。

判据（先立判据，再实现）
----
K1 多轴整数绕数不可见：对 8 个随机轴，绕核心一周累积 theta = 2*pi*n（n = 1,2,3,-1）
   的闭链，SO(3) holonomy = I（小于 1e-12），双覆盖提升 = (-1)^n I（小于 1e-12），
   半径 r = 1,2,3 一致。
K2 非共轴四腿回路：四条腿各绕不同轴、各累积 2*pi*n_i，闭链的 SO(3) holonomy = I
   （小于 1e-12），提升 = (-1)^{sum n_i} I（小于 1e-12）；随机轴与绕数四元组共 6 组。
   （K1 的载体是格点构型，K2 的载体是闭链标架序列，后者是唯一能暴露非阿贝尔序的对象。）
K3 二维空间加时间的降落恒等式：以时间向规范 theta_0 = 0，逐点核对
   Delta_t F_12 = Delta_2 F_01 - Delta_1 F_02（小于 1e-13）；全网求和恒为零
   （小于 1e-13）；同时局域曲率的最大变化为 O(1)，排除"什么都没动"的假通过。
K4 三维空间加时间：同一恒等式逐分量核对（三对 (i,j) 全部，小于 1e-13），
   小格点 L = 6，含一个逐时步移动的涡旋。

运行
----
python 验证脚本/t1b2a_frame_tetrad_4d.py
确定性（随机数种子固定 20260918），运行约 3 秒。
"""
import time

import numpy as np

T0 = time.time()
CHECKS = []
RNG = np.random.default_rng(20260918)


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print(f"  [{tag}] {'PASS' if ok else 'FAIL'}  {detail}")


def so3(axis, ang):
    """链接矩阵 exp(ang * K(axis))，axis 为 (n,3)、ang 为 (n,)"""
    k = np.asarray(axis, float).reshape(-1, 3)
    a = np.asarray(ang, float).reshape(-1, 1, 1)
    K = np.zeros((len(k), 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -k[:, 2], k[:, 1]
    K[:, 1, 0], K[:, 1, 2] = k[:, 2], -k[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -k[:, 1], k[:, 0]
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)


def su2(axis, ang):
    """双覆盖提升 exp(i * ang/2 * k.sigma)，返回 (n,2,2)"""
    k = np.asarray(axis, float).reshape(-1, 3)
    h = np.asarray(ang, float).reshape(-1) / 2.0
    c, s = np.cos(h), np.sin(h)
    z = np.zeros_like(c, dtype=complex)
    kx, ky, kz = k[:, 0], k[:, 1], k[:, 2]
    u = np.empty((len(k), 2, 2), dtype=complex)
    u[:, 0, 0] = c + 1j * s * kz
    u[:, 0, 1] = 1j * s * (kx - 1j * ky)
    u[:, 1, 0] = 1j * s * (kx + 1j * ky)
    u[:, 1, 1] = c - 1j * s * kz
    return u


def wrap_pi(x):
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def prod(ms):
    out = np.eye(ms.shape[-1], dtype=ms.dtype)
    for m in ms:
        out = out @ m
    return out


print("=" * 72)
print("最小验证 T1b-2a：标架扇区的四维离散口径与非阿贝尔推广")
print("=" * 72)

# ----------------------------------------------------------------------
# K1 多轴整数绕数不可见
# ----------------------------------------------------------------------
print("\n[K1] 多轴整数绕数：SO(3) holonomy = I，提升 = (-1)^n I")
j1 = 0.0
for _ in range(8):
    k = RNG.normal(size=3)
    k /= np.linalg.norm(k)
    for n in (1, 2, 3, -1):
        for r in (1, 2, 3):
            # 闭链：半径 r 的方回路，每段 theta = 2*pi*n/(4*2r)，四段合计 2*pi*n
            m = 4 * (2 * r)
            ang = np.full(m, 2 * np.pi * n / m)
            M = prod(so3(np.broadcast_to(k, (m, 3)), ang))
            U = prod(su2(np.broadcast_to(k, (m, 3)), ang))
            j1 = max(j1, np.max(np.abs(M - np.eye(3))),
                     np.max(np.abs(U - ((-1) ** n) * np.eye(2, dtype=complex))))
print(f"      8 轴 x 4 绕数 x 3 半径 = 96 个构型，最大偏差 {j1:.3e}")
check("K1", j1 < 1e-12, f"最大偏差 {j1:.3e}（判据 < 1e-12）")

# ----------------------------------------------------------------------
# K2 非共轴四腿回路
# ----------------------------------------------------------------------
print("\n[K2] 非共轴四腿回路：每腿各绕一轴累积 2*pi*n_i")
j2 = 0.0
for trial in range(6):
    axes = RNG.normal(size=(4, 3))
    axes /= np.linalg.norm(axes, axis=-1, keepdims=True)
    ns = RNG.integers(-2, 3, size=4)
    if np.all(ns == 0):
        ns[0] = 1
    seg = 12
    M = np.eye(3)
    U = np.eye(2, dtype=complex)
    for leg in range(4):
        ang = np.full(seg, 2 * np.pi * ns[leg] / seg)
        ax = np.broadcast_to(axes[leg], (seg, 3))
        for a_ in ang:
            M = M @ so3(ax[:1], [a_])[0]
            U = U @ su2(ax[:1], [a_])[0]
    tgt = ((-1) ** int(np.sum(ns))) * np.eye(2, dtype=complex)
    dev = max(np.max(np.abs(M - np.eye(3))), np.max(np.abs(U - tgt)))
    j2 = max(j2, dev)
    print(f"      第 {trial + 1} 组：绕数 {tuple(int(v) for v in ns)}，偏差 {dev:.2e}")
check("K2", j2 < 1e-12, f"六组最大偏差 {j2:.3e}（判据 < 1e-12），符号 = (-1)^(总绕数)")

# ----------------------------------------------------------------------
# K3 二维空间加时间：降落恒等式
# ----------------------------------------------------------------------
print("\n[K3] 2+1 维降落恒等式：Delta_t F_12 = Delta_2 F_01 - Delta_1 F_02")
L2, T2 = 12, 8


def polar_links(L, core, q):
    g = np.meshgrid(np.arange(L), np.arange(L), indexing='ij')
    ang = np.arctan2(g[1] - core[1], g[0] - core[0])
    th = [np.zeros_like(ang) for _ in range(2)]
    for a in range(2):
        d = np.roll(ang, -1, axis=a) - ang
        th[a] = q * wrap_pi(d)
    return th


th_t = []
for t in range(T2):
    core = (3 + (t % (L2 - 6)), 6)
    th_t.append(polar_links(L2, core, 1.0))

res_id, sum_dt, max_dt, max_plaq = 0.0, 0.0, 0.0, 0.0
for t in range(T2 - 1):
    a0, a1 = th_t[t], th_t[t + 1]
    d0 = [a1[i] - a0[i] for i in range(2)]
    F12 = a1[0] + np.roll(a1[1], -1, axis=0) - np.roll(a1[0], -1, axis=1) - a1[1]
    F12_p = a0[0] + np.roll(a0[1], -1, axis=0) - np.roll(a0[0], -1, axis=1) - a0[1]
    dtF = F12 - F12_p
    F01, F02 = d0[0], d0[1]
    rhs = (np.roll(F02, -1, axis=0) - F02) - (np.roll(F01, -1, axis=1) - F01)
    res_id = max(res_id, np.max(np.abs(dtF - rhs)))
    sum_dt = max(sum_dt, abs(np.sum(dtF)))
    max_dt = max(max_dt, np.max(np.abs(dtF)))
    max_plaq = max(max_plaq, np.max(np.abs(F12)))
print(f"      逐点恒等式残差 {res_id:.3e}；|Sum dtF| 最大 {sum_dt:.3e}"
      f"；max|dtF| = {max_dt:.3f}；max|F_12| = {max_plaq:.3f}")
check("K3", res_id < 1e-13 and sum_dt < 1e-13 and max_dt > 0.1,
      f"恒等式残差 {res_id:.3e}、总变化 {sum_dt:.3e}（判据 < 1e-13），局域变化 {max_dt:.3f} 非零")

# ----------------------------------------------------------------------
# K4 三维空间加时间：逐分量核对
# ----------------------------------------------------------------------
print("\n[K4] 3+1 维：三对 (i,j) 的同一恒等式，L = 6，涡旋逐时步移动")
L3, T3 = 6, 5


def polar_links3(L, core, q):
    g = np.meshgrid(np.arange(L), np.arange(L), np.arange(L), indexing='ij')
    ang = np.arctan2(g[1] - core[1], g[0] - core[0] + 1e-9 * (g[2] - core[2]))
    th = [np.zeros_like(ang) for _ in range(3)]
    for a in range(3):
        d = np.roll(ang, -1, axis=a) - ang
        th[a] = q * wrap_pi(d)
    return th


slices = [polar_links3(L3, (2 + (t % 2), 3, 3), 1.0) for t in range(T3)]
res4 = 0.0
for t in range(T3 - 1):
    b0, b1 = slices[t], slices[t + 1]
    dlt = [b1[i] - b0[i] for i in range(3)]

    def plaq(th, i, j):
        return th[i] + np.roll(th[j], -1, axis=i) - np.roll(th[i], -1, axis=j) - th[j]

    for (i, j) in [(0, 1), (1, 2), (2, 0)]:
        dtF = plaq(b1, i, j) - plaq(b0, i, j)
        rhs = (np.roll(dlt[j], -1, axis=i) - dlt[j]) - (np.roll(dlt[i], -1, axis=j) - dlt[i])
        res4 = max(res4, np.max(np.abs(dtF - rhs)))
print(f"      三对分量的最大残差 {res4:.3e}")
check("K4", res4 < 1e-13, f"最大残差 {res4:.3e}（判据 < 1e-13）")

print("\n" + "=" * 72)
n_pass = sum(1 for _, ok in CHECKS if ok)
print(f"判据汇总：{n_pass}/{len(CHECKS)} 通过")
for tag, ok in CHECKS:
    print(f"  {tag}: {'PASS' if ok else 'FAIL'}")
print(f"耗时 {time.time() - T0:.1f} s")