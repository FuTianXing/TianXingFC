# -*- coding: utf-8 -*-
r"""
tension_reeval.py

张力常数重估轮：第 278 条的环判据是否依赖 sigma，第 279 条的能量量级落在哪里。

来源：[[验证脚本/注记-弦张力松弛复测轮：ansatz 斜率属势项面积产物、松弛壁张力随根号 K（判据G0至G7）]] 的结论（sigma_wall = (4/pi) sqrt(kK)）落地后的两次重估。

模型：一维链 E = sum[(1/2) k (du)^2 + K (1 - cos 2 pi v)]，u = i + v，故 du - 1 = dv（与第 279 条同一泛函）。
  解析：kink 质量 M = (4/pi) sqrt(kK)（k = 1）。

判据（先行）：
  T1 恒等式：sigma_wall = (4/pi) sqrt(kK) 与第 279 条的 M = 4 sqrt(K)/pi 逐点相同（相对差 0）
  T2 数值：一维链松弛的 kink 能量对 (4/pi) sqrt(K)，两档（0.005、0.001）偏差 <= 2%
  T3 实测对齐：第 279 条实测 0.0402 对 (4/pi) sqrt(0.001)，残差 <= 1%
  T4 旧值对照：旧读数 90.1 K^0.95 对新值，在 C1 驻留区与 W3 上界的比值（登记）
  T5 跨窗张力表（K = 0.25 至 0.0005）
"""
import numpy as np

rows = []


def chain_kink(K, N=400, iters=200000, dt=0.24, k=1.0):
    v = np.zeros(N)
    v[:] = np.linspace(0.0, -1.0, N)
    v[0], v[-1] = 0.0, -1.0
    e0 = 0.5 * k * float(np.sum(np.diff(v) ** 2)) + K * float(np.sum(1.0 - np.cos(2.0 * np.pi * v)))
    for it in range(iters):
        lap = np.zeros_like(v)
        lap[1:-1] = v[2:] + v[:-2] - 2.0 * v[1:-1]
        g = -k * lap + 2.0 * np.pi * K * np.sin(2.0 * np.pi * v)
        g[0] = 0.0
        g[-1] = 0.0
        v = v - dt * g
        v[0], v[-1] = 0.0, -1.0
    E = 0.5 * k * float(np.sum(np.diff(v) ** 2)) + K * float(np.sum(1.0 - np.cos(2.0 * np.pi * v)))
    return E, e0


# T1 恒等式
Ks = [0.25, 0.05, 0.02, 0.005, 0.001583, 0.001, 0.0005]
sig = {K: (4.0 / np.pi) * np.sqrt(K) for K in Ks}
M = {K: 4.0 * np.sqrt(K) / np.pi for K in Ks}
d1 = max(abs(sig[K] - M[K]) / sig[K] for K in Ks)
rows.append(("T1", "sigma_wall 与第 279 条 M = 4 sqrt(K)/pi 的恒等性", "最大相对差 %.1e" % d1, "== 0", d1 < 1e-15))

# T2 一维链数值
e1, e0a = chain_kink(0.005)
e2, e0b = chain_kink(0.001)
r1 = abs(e1 - sig[0.005]) / sig[0.005]
r2 = abs(e2 - sig[0.001]) / sig[0.001]
rows.append(("T2", "一维链 kink 能量对 (4/pi) sqrt(K)（0.005 与 0.001）",
             "%.6f 与 %.6f（偏差 %.2f%% 与 %.2f%%）" % (e1, e2, 100 * r1, 100 * r2),
             "偏差 <= 2%", max(r1, r2) <= 0.02))

# T3 与第 279 条实测对齐
r3 = abs(0.0402 - sig[0.001]) / sig[0.001]
rows.append(("T3", "第 279 条实测 0.0402 对 (4/pi) sqrt(0.001) = %.6f" % sig[0.001], "残差 %.2f%%" % (100 * r3), "<= 1%", r3 <= 0.01))

# T4 旧值对照
old = {K: 90.1 * K ** 0.95 for K in Ks}
rows.append(("T4", "旧读数 90.1 K^0.95 对新值的比值（登记）",
             "K = 0.25 给 %.1f；K = 0.001583 给 %.1f；K = 0.001 给 %.1f" % (old[0.25] / sig[0.25], old[0.001583] / sig[0.001583], old[0.001] / sig[0.001]),
             "登记（旧值在高 K 端放大近 40 倍）", True))

print("=== 张力常数重估轮判据表（实测 对 期望）===")
for tag, name, meas, exp, ok in rows:
    print("%-4s %-52s 实测 %-56s 期望 %-24s %s" % (tag, name, meas, exp, "通过" if ok else "不通过"))
print("")
print("T5 跨窗张力表（sigma_wall = (4/pi) sqrt(K)，与第 279 条 M 同式）：")
for K in Ks:
    print("   K = %.6f：sigma_wall = %.6f；旧读数 90.1 K^0.95 = %.6f（比值 %.2f）" % (K, sig[K], old[K], old[K] / sig[K]))
print("")
print("T2 过程：一维链初值能量 %.4f 到 %.4f（K = 0.005）、%.4f 到 %.4f（K = 0.001）" % (e0a, e1, e0b, e2))
print("T4 结论：旧读数在新值上的放大倍数随 K 下降（C1 驻留区 K = 0.25 放大约 38 倍，W3 上界放大约 3.9 倍），")
print("         故任何以旧 sigma 估量级的判据都须换用 sigma_wall；第 278 条的定版依据（C1-3 的 R_c 扫描与 C1-4 的 K >= 0.24 阈值）来自直接数值，不含 sigma，故判定不变。")
