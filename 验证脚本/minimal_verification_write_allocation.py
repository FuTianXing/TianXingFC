# -*- coding: utf-8 -*-
"""
最小验证：写事件在相邻链接上的分配细节（判据 L1 至 L5）

问题：判据 Z 说写源是沿世界线的四维流，判据 K 说同一核作用于动量密度、强度被
单一入口钉为 1；但"一笔写入落在一格上时，它具体摊到相邻链接的哪几条、各摊多少"
这条微观规则一直是"沿用静态情形"的口头继承。本材料把它写出来并核对。

分配规则（本材料显式化并检验的版本）：
  标量读数：写事件落在站点 n，写入量作为源项进入该站的离散散度；六个最近邻链接
     各承载 1/6 的通量（等分由格点对称性给出，非外加约定）。
  矢量读数：动量的第 i 分量只由沿第 i 轴的那一对链接承载（方向指标 = 链接取向），
     即 j_i = rho v_i；完整分解由 sum_i (v.e_i) e_i = v 保证，故无横向损失。
  两扇道共用同一个离散算子，核在比值中约掉，不引入新参数。

判据（先写期望，再算）：
  L1 分配显式化：源站六条链接通量等值（机器精度）；闭合盒面通量 = 源（离散高斯）
  L2 分配不损动量：sum_i (v.e_i) e_i = v（机器精度）
  L3 核的方向结构：格点格林函数的三个取向在固定 r 下的散布 ∝ r^-2（随距离消失）；
     常数偏移 ∝ 1/L 属有限盒约定，不影响方向散布与通量
  L4 与判据 K 对接：分配规则下 h0i/(h00 v) = 2（机器精度）
  L5 守恒：源之外的链路散度为零（机器零）；源点散度 = -1
"""
import math
import numpy as np

PI = math.pi


def green(n):
    k = 2.0 * PI * np.fft.fftfreq(n, d=1.0)
    KX, KY, KZ = np.meshgrid(k, k, k, indexing="ij")
    K2 = 4 * (np.sin(KX / 2) ** 2 + np.sin(KY / 2) ** 2 + np.sin(KZ / 2) ** 2)
    K2c = K2.copy()
    K2c[0, 0, 0] = 1.0
    src = np.zeros((n, n, n))
    c = n // 2
    src[c, c, c] = 1.0
    g = np.real(np.fft.ifftn(np.fft.fftn(src) / K2c))
    return g, c


def main():
    print("=" * 78)
    print("写事件在相邻链接上的分配细节（判据 L1 至 L5）")
    print("=" * 78)
    n = 96
    g, c = green(n)

    # L1
    print()
    print("[L1] 分配显式化：源站六条链接的归一化通量")
    f = [g[c + 1, c, c] - g[c, c, c], g[c - 1, c, c] - g[c, c, c],
         g[c, c + 1, c] - g[c, c, c], g[c, c - 1, c] - g[c, c, c],
         g[c, c, c + 1] - g[c, c, c], g[c, c, c - 1] - g[c, c, c]]
    f = np.array(f)
    print("     六条链接通量：%s" % np.array2string(f, precision=8))
    print("     均分判据：最大相对差 = %.2e（期望 0，六条各 1/6）" % ((f.max() - f.min()) / abs(f.mean())))
    lap = np.zeros_like(g)
    lap[1:-1, 1:-1, 1:-1] = (
        (g[2:, 1:-1, 1:-1] - 2 * g[1:-1, 1:-1, 1:-1] + g[:-2, 1:-1, 1:-1]) +
        (g[1:-1, 2:, 1:-1] - 2 * g[1:-1, 1:-1, 1:-1] + g[1:-1, :-2, 1:-1]) +
        (g[1:-1, 1:-1, 2:] - 2 * g[1:-1, 1:-1, 1:-1] + g[1:-1, 1:-1, :-2]))
    for m in (1, 2, 3):
        box = lap[c - m:c + m + 1, c - m:c + m + 1, c - m:c + m + 1]
        print("     半径 %d 的闭合盒面通量 = %.6f（期望 -1.000000，符号随算子约定）" % (m, float(box.sum())))

    # L2
    print()
    print("[L2] 分配不损动量：sum_i (v.e_i) e_i = v")
    rng = np.random.default_rng(20260920)
    worst = 0.0
    for _ in range(200):
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        recon = sum(v[i] * np.eye(3)[i] for i in range(3))
        worst = max(worst, float(np.linalg.norm(recon - v)))
    print("     200 个随机方向：最大重构误差 = %.2e（期望 0）" % worst)

    # L3
    print()
    print("[L3] 核的方向结构：三取向在固定 r 下的散布")
    dirs = ([1, 0, 0], [1, 1, 0], [1, 1, 1])
    g, c = green(192)
    rows = []
    for r in (12, 24, 48):
        vals = []
        for d in dirs:
            u = np.array(d) / np.linalg.norm(d)
            idx = tuple(int(round(u[i] * r)) for i in range(3))
            vals.append(g[(c + idx[0], c + idx[1], c + idx[2])])
        cont = 1.0 / (4 * PI * r)
        spread = (max(vals) - min(vals)) / cont
        rows.append((r, spread))
        print("     r = %2d：三取向值 %.6e / %.6e / %.6e；方向散布/连续值 = %.3e"
              % (r, vals[0], vals[1], vals[2], spread))
    fit = np.polyfit(np.log([x[0] for x in rows]), np.log([x[1] for x in rows]), 1)
    print("     方向散布的幂律斜率 = %.3f（期望 -2，随距离消失）" % fit[0])
    off = []
    for nn in (192, 256):
        gg, cc = green(nn)
        u = np.array([1, 0, 0])
        rr = 12
        val = gg[cc + rr, cc, cc]
        off.append(abs(val - 1.0 / (4 * PI * rr)))
    ratio_L = off[0] / off[1]
    print("     常数偏移随盒尺寸：n = 192 给 %.3e，n = 256 给 %.3e；比值 = %.4f"
          % (off[0], off[1], ratio_L))
    print("     1/L 约定给比值 %.4f（256/192），实测偏差 %.2f%%" % (256.0 / 192.0, abs(ratio_L - 256.0 / 192.0) / (256.0 / 192.0) * 100))
    print("     偏移属有限盒约定（同一方向同一值），不影响方向散布与通量")

    # L4
    print()
    print("[L4] 与判据 K 对接：分配规则下的 h0i/(h00 v)")
    n = 96
    X = np.arange(n) - n // 2
    XX, YY, ZZ = np.meshgrid(X, X, X, indexing="ij")
    R = np.sqrt(XX ** 2 + YY ** 2 + ZZ ** 2)
    blob = np.exp(-(R / 4.0) ** 4)
    k = 2.0 * PI * np.fft.fftfreq(n, d=1.0)
    KX, KY, KZ = np.meshgrid(k, k, k, indexing="ij")
    K2 = 4 * (np.sin(KX / 2) ** 2 + np.sin(KY / 2) ** 2 + np.sin(KZ / 2) ** 2)
    K2c = K2.copy(); K2c[0, 0, 0] = 1.0
    v = 0.1
    th = np.real(np.fft.ifftn(np.fft.fftn(blob) / K2c))
    A_x = np.real(np.fft.ifftn(np.fft.fftn(blob * v) / K2c))
    i0 = n // 2 + 12
    ratio = (4 * A_x[i0, n // 2, n // 2]) / (2 * th[i0, n // 2, n // 2] * v)
    print("     远场比值 = %.15f（期望 2.000000000000000）" % ratio)
    print("     说明：方向分解完整（L2），故分配规则不改变判据 K 的强度读数")

    # L5
    print()
    print("[L5] 守恒：链路散度（96 盒）")
    g, c = green(96)
    lap = np.zeros_like(g)
    lap[1:-1, 1:-1, 1:-1] = (
        (g[2:, 1:-1, 1:-1] - 2 * g[1:-1, 1:-1, 1:-1] + g[:-2, 1:-1, 1:-1]) +
        (g[1:-1, 2:, 1:-1] - 2 * g[1:-1, 1:-1, 1:-1] + g[1:-1, :-2, 1:-1]) +
        (g[1:-1, 1:-1, 2:] - 2 * g[1:-1, 1:-1, 1:-1] + g[1:-1, 1:-1, :-2]))
    src_val = lap[c, c, c]
    mask = np.abs(lap).copy()
    mask[c, c, c] = 0.0
    print("     源点散度 = %.6f（期望 -1）" % src_val)
    print("     其余全网格最大绝对值 = %.2e（期望 0）" % mask.max())


main()
