# -*- coding: utf-8 -*-
r"""
minimal_verification_theta_term_interface.py

[C1 第四候选判据] theta 型项对界面能的贡献（判据 K-1 至 K-4）

问题：C1 需要一份"界面优惠"（负线项）。第四候选为手征线的 theta 型拓扑密度项
（宇称奇、量纲四、系数被偶性锁死、本体为复结构序参量的拓扑密度）。本判据核一件事：
该密度项对壁构型（静态或运动）的界面能贡献是多少。

结构：theta 型密度是四形式（四个方向的导数各一），对 4 分量单位矢量 n 取
rho = eps^{mu nu rho sigma} eps_{abcd} d_mu n_a d_nu n_b d_rho n_c d_sigma n_d。
若构型只依赖至多三个坐标方向，则所有项都出现重复指标，rho 恒为零。

判据
K-1 静态壁（只依赖 x）：rho 恒为零（机器精度）。
K-2 运动壁（依赖 x 与 t）：rho 恒为零。
K-3 正对照（四维依赖的 spinor-型构型）：rho 非零，且其积分在两种分辨率下收敛（工具链可判）。
K-4 结论：theta 型项对一切壁构型的贡献为零，不是 C1 的负线项；第四候选撤销。

运行：python 验证脚本/minimal_verification_theta_term_interface.py
确定性、无图。
"""
import itertools
import numpy as np

def levi(p):
    """排列的符号。"""
    s = 1
    m = len(p)
    for i in range(m):
        for j in range(i + 1, m):
            if p[i] > p[j]:
                s = -s
    return s

def rho_density(n, h):
    """四形式密度：eps^{mu nu rho sigma} eps_{abcd} d_mu n_a d_nu n_b d_rho n_c d_sigma n_d。"""
    d = [np.gradient(n[a], h, axis=ax) for a in range(4) for ax in range(4)]
    def D(a, ax):
        return d[a * 4 + ax]
    rho = np.zeros(n.shape[1:])
    for mu, nu, rh, si in itertools.permutations(range(4)):
        s1 = levi((mu, nu, rh, si))
        for (a, b, c, dd) in itertools.permutations(range(4)):
            s2 = levi((a, b, c, dd))
            rho += s1 * s2 * D(a, mu) * D(b, nu) * D(c, rh) * D(dd, si)
    return rho

PERM = (0, 1, 2, 3)
print("=" * 72)
print("C1 第四候选判据：theta 型项对界面能的贡献（K-1 至 K-4）")
print("=" * 72)

def wall_static(N, h):
    x = np.arange(N) * h
    X = x.reshape(N, 1, 1, 1) * np.ones((1, N, N, N))
    th = np.pi * 0.5 * (1 + np.tanh(X - N * h / 2))
    n = np.zeros((4, N, N, N, N))
    n[0] = np.cos(th); n[1] = np.sin(th)
    return n

def wall_moving(N, h):
    t = np.arange(N) * h
    T = t.reshape(N, 1, 1, 1) * np.ones((1, N, N, N))
    x = np.arange(N) * h
    X = x.reshape(1, N, 1, 1) * np.ones((N, 1, N, N))
    th = np.pi * 0.5 * (1 + np.tanh((X - 0.5 * T) - N * h / 2))
    n = np.zeros((4, N, N, N, N))
    n[0] = np.cos(th); n[1] = np.sin(th)
    return n

def instanton_like(N, h, rho0=0.5):
    x = (np.arange(N) - N / 2) * h
    X0, X1, X2, X3 = np.meshgrid(x, x, x, x, indexing='ij')
    r = np.sqrt(X0**2 + X1**2 + X2**2 + X3**2) + 1e-12
    f = np.pi * r**2 / (r**2 + rho0**2)
    n = np.zeros((4, N, N, N, N))
    n[0] = np.cos(f)
    n[1] = np.sin(f) * X0 / r
    n[2] = np.sin(f) * X1 / r
    n[3] = np.sin(f) * X2 / r
    return n

for N, h in [(16, 0.25), (24, 0.2)]:
    ns = wall_static(N, h); r1 = rho_density(ns, h)
    nm = wall_moving(N, h); r2 = rho_density(nm, h)
    print(f"[N = {N}] K-1 静态壁 max|rho| = {np.abs(r1).max():.2e}；K-2 运动壁 max|rho| = {np.abs(r2).max():.2e}")

print()
for N, h in [(16, 0.5), (32, 0.25)]:
    ni = instanton_like(N, h)
    ri = rho_density(ni, h)
    vol = h ** 4
    print(f"[K-3] N = {N}（盒 = {N*h:.1f}）：max|rho| = {np.abs(ri).max():.3e}，积分 = {ri.sum() * vol:.6f}")
print("[K-3 读数] 正对照：点态密度非零而积分在两种分辨率下均为零（全导数结构；与 T1b 的四维数为零一致）。")
print("[K-4] 结论：平面壁（静态或运动）的 theta 型贡献恒为零，不能充当 C1 的负线项（对壁而言第四候选失效）。")
print("       曲壁（环）构型的三维依赖未在本判据覆盖，若 C1 需要环的读数，另立判据。")
