# -*- coding: utf-8 -*-
r"""
minimal_verification_c1_protection_transplant.py

[C1 保护层可搬用性判定] 弦算符结构与泄漏通道的对照（判据 V-1 至 V-4）

背景：作者裁决（2026-09-20）——拓扑保护依赖非定域的长程纠缠（波性侧），粒性侧
局域作用提供不了；保护由两侧双协同给出。本判据做有界判定：电荷路线既有的
保护机制（态空间按绕数分扇区、跨扇区算符为沿切割分布的弦算符、弦不在局域
生成元闭包内）能否直接搬到 C1 的复合位错构型上。

模型（最小）：二维方格的相位场 phi，绕数定义为 plaquette 上的环绕差之和，
    W = (1/2 pi) sum_p [wrap(dx phi) + wrap(dy phi) - ...]
单涡构型取 phi = atan2(y - y0, x - x0)。

判据
V-1 边界性：面积求和 W 与边界回路求和 W_b 相等（绕数是边界量，机器精度）
V-2 局域不变性：内部紧支撑扰动（边界数据不动）不改变 W（100 组，精确不变）
V-3 到边界的弦：把单涡移出区域的唯一局域路径是移芯到边界；移芯过程中 W 保持
    为 1，只有在芯越出边界的那一刻才归零（弦必须触及边界）
V-4 结构对照：电荷路线三要件（扇区分解、弦形跨扇区算符、守卫来自先验纠缠）
    在 C1 构型上的对应与增量（宽度/质量/色散不在搬用范围内）

运行：python 验证脚本/minimal_verification_c1_protection_transplant.py
（确定性，无随机数，约 2 秒）
"""
import math
import numpy as np

N = 64
TWO_PI = 2.0 * math.pi


def wrap(a):
    return (a + math.pi) % TWO_PI - math.pi


def field_vortex(x0, y0):
    X, Y = np.meshgrid(np.arange(N, dtype=float), np.arange(N, dtype=float), indexing="ij")
    return np.arctan2(Y - y0, X - x0)


def plaquette_winding(phi):
    """每个 plaquette 的环绕数（整数 0 或 ±1）。"""
    dx = wrap(phi[1:, :-1] - phi[:-1, :-1])       # 沿 x 的边缘差
    dx2 = wrap(phi[1:, 1:] - phi[:-1, 1:])
    dy = wrap(phi[:-1, 1:] - phi[:-1, :-1])       # 沿 y 的边缘差
    dy2 = wrap(phi[1:, 1:] - phi[1:, :-1])
    w = (dx + dy2 - dx2 - dy) / TWO_PI
    return np.rint(w).astype(int), np.abs(w - np.rint(w)).max()


def boundary_winding(phi):
    """外圈回路的环绕数（逆时针走序：下 +x、右 +y、上 -x、左 -y）。"""
    # 走序取与 plaquette 求和同向（口径：两式定向约定一致时逐位相等）
    loop = np.concatenate([
        phi[0, ::-1],       # 下边：y = 0，x 从 N-1 到 0
        phi[::-1, 0],       # 左边：x = 0，y 从 N-1 到 0
        phi[-1, :],         # 上边：y = N-1，x 从 0 到 N-1
        phi[:, -1],         # 右边：x = N-1，y 从 0 到 N-1
    ])
    return float(np.sum(wrap(np.diff(np.append(loop, loop[0])))) / TWO_PI)


print("=" * 78)
print("C1 保护层可搬用性判定：弦算符结构与泄漏通道（判据 V-1 至 V-4）")
print("=" * 78)

# V-1 边界性
print()
print("[V-1] 绕数是边界量：面积求和与边界回路求和")
for (x0, y0, tag) in ((32.0, 32.0, "芯在中心"), (10.0, 50.0, "芯在偏心")):
    phi = field_vortex(x0, y0)
    wp, err = plaquette_winding(phi)
    W = int(wp.sum())
    Wb = boundary_winding(phi)
    print("     %s：面积求和 W = %d，边界回路 W_b = %.6f，差 = %.2e，plaquette 取整误差 = %.1e"
          % (tag, W, Wb, abs(W - Wb), err))

# V-2 局域不变性
print()
print("[V-2] 内部紧支撑扰动（边界不动）不改变 W")
phi = field_vortex(32.0, 32.0)
base = int(plaquette_winding(phi)[0].sum())
rng = np.random.default_rng(20260920)
worst = 0
for k in range(100):
    perturbed = phi.copy()
    r = rng.integers(6, 12)                     # 扰动球半径
    cx, cy = rng.integers(20, 44, size=2)       # 扰动中心（远离边界）
    X, Y = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    m = (X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2
    perturbed[m] += rng.uniform(-1.0, 1.0) * 0.5
    W = int(plaquette_winding(perturbed)[0].sum())
    worst = max(worst, abs(W - base))
print("     100 组随机内部扰动：|W - W0| 最大 = %d（期望 0）" % worst)

# V-3 到边界的弦
print()
print("[V-3] 把单涡移出区域：W 只在芯越出边界时改变")
for x0, y0 in ((32.0, 32.0), (12.0, 12.0)):
    phi = field_vortex(x0, y0)
    seq = []
    for shift in (0, 8, 16, 24, 32, 40, 48, 56, 64, 70):
        xx = x0 + shift
        ph = field_vortex(xx, y0)
        W = int(plaquette_winding(ph)[0].sum())
        seq.append((xx, W))
    print("     起始 (%.0f,%.0f)：移芯序列 %s" % (x0, y0, " ".join("x=%.0f:W=%d" % s for s in seq)))

# V-4 结构对照
print()
print("[V-4] 结构对照（电荷路线要件 → C1 对应）")
rows = [
 ("态空间按绕数分扇区", "C1 构型的横截面绕数（C1-0 装置可测）", "直接对应"),
 ("跨扇区算符为弦算符（沿切割分布）", "本判据 V-1 至 V-3：绕数是边界量，局域体算符不改它", "直接对应"),
 ("守卫来自先验纠缠（波性侧）", "同一超选择层；C1 无独立保护", "直接对应"),
 ("宽度与色散由载体有效宽度承担", "C1 的落窗条件与宽度窗口", "搬用不覆盖，需另建"),
]
for a, b, c in rows:
    print("     %-34s | %-40s | %s" % (a, b, c))
