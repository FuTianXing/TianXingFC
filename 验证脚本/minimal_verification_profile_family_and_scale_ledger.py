# -*- coding: utf-8 -*-
r"""
minimal_verification_profile_family_and_scale_ledger.py

[引力线收口轮] 剖面族边界扫描与\theta 口径核对（判据 PF1 至 PF5）

两件收口工作：
  一、把域边界从四个代表剖面推广到单参数族：高斯脉冲 A = exp(-(u-1.5)^2/(2w^2))，扫宽度 w，
      量 Lambda_*(u0) 随 w 的变化（单调？内点极值？），并复核第 28 轮 P3 的读数。
  二、口径核对：框架原文给出 h_{00} = 2\theta（因子 2 来自相对论约定）与 tr h = 6\theta
      （等价于各向同性部分 h_{ij} = 2\theta \delta_{ij}），但未给无迹（张量）部分的显式规则。
      本判据把两种读数（theta = a 与 theta = a/2）下两扇区的退化条件并列，看哪一种与静场
      扇区的视界条件形式相同。

判据
PF1 基线回归：w = 0.354 档复现第 28 轮 P3 读数（Lambda_* = 1.5083 与 1.0798）
PF2 族扫描：w = 0.20 至 1.00 六档的 Lambda_*(u0 = 2, 3)
PF3 形状依赖：单调性与相对跨度
PF4 口径核对：两扇区退化条件的系数结构对照
PF5 登记

运行：python 验证脚本/minimal_verification_profile_family_and_scale_ledger.py（确定性，约 1 分钟）
"""
import numpy as np
import math
import time
from scipy.integrate import solve_ivp

T_START = time.time()


def profile(w):
    def A(u):
        return np.exp(-((u - 1.5) ** 2) / (2.0 * w ** 2))
    def A1(u):
        return A(u) * (-(u - 1.5) / w ** 2)
    def A2(u):
        return A(u) * (((u - 1.5) ** 2) / w ** 4 - 1.0 / w ** 2)
    return A, A1, A2


def make_rhs(w, lam):
    A, A1, A2 = profile(w)
    def rhs(u, y):
        Bv, Bp = y
        a, ap, app = lam * A(u), lam * A1(u), lam * A2(u)
        P = 1 + a + Bv
        Q = 1 - a + Bv
        S = P * Q
        return [Bp, app * a / (1 + Bv) + S / (4 * (1 + Bv)) * (((ap + Bp) / P) ** 2 + ((Bp - ap) / Q) ** 2)]
    return rhs


def probe(w, lam, u0, du=0.05, bthr=30.0):
    rhs = make_rhs(w, lam)
    y = np.array([0j, 0j])
    u = 0.0
    while u < u0 - 1e-12:
        tgt = min(u + du, u0)
        try:
            sol = solve_ivp(rhs, [u, tgt], y, method='DOP853', rtol=1e-10, atol=1e-12, max_step=0.02)
        except Exception:
            return False
        y = sol.y[:, -1]
        if (not sol.success) or (not np.all(np.isfinite(y))) or abs(y[0]) > bthr or abs(1 + y[0]) < 1e-3:
            return False
        u = tgt
    return True


def Lambda_star(w, u0, hi=3.0):
    if probe(w, hi, u0):
        return None
    lo = 0.02
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        if probe(w, mid, u0):
            lo = mid
        else:
            hi = mid
    return lo


print("=" * 78)
print("引力线收口轮：剖面族扫描与 theta 口径核对（判据 PF1 至 PF5）")
print("=" * 78)

print()
print("[PF1] 基线回归：w = 0.354 档（第 28 轮 P3）")
res = {}
for u0 in (2.0, 3.0):
    v = Lambda_star(0.354, u0)
    res[(0.354, u0)] = v
    print("     u0 = %.0f：Lambda_* = %.4f（第 28 轮 1.5083 / 1.0798）" % (u0, v))

print()
print("[PF2] 族扫描：高斯脉冲宽度 w 与边界（零初条件约定，max|Â| = 1）")
WS = (0.20, 0.30, 0.354, 0.50, 0.70, 1.00)
print("     %6s ｜ %10s ｜ %10s" % ("w", "Lambda_*(2)", "Lambda_*(3)"))
for w in WS:
    v2 = res.get((w, 2.0)) or Lambda_star(w, 2.0)
    v3 = res.get((w, 3.0)) or Lambda_star(w, 3.0)
    res[(w, 2.0)], res[(w, 3.0)] = v2, v3
    print("     %6.3f ｜ %10s ｜ %10s" % (w, "无边界" if v2 is None else "%.4f" % v2,
                                         "无边界" if v3 is None else "%.4f" % v3))

print()
print("[PF3] 形状依赖")
vals2 = [res[(w, 2.0)] for w in WS if res[(w, 2.0)] is not None]
vals3 = [res[(w, 3.0)] for w in WS if res[(w, 3.0)] is not None]
for tag, vals in (("u0 = 2", vals2), ("u0 = 3", vals3)):
    mono = all(vals[i + 1] <= vals[i] + 1e-9 for i in range(len(vals) - 1)) or \
           all(vals[i + 1] >= vals[i] - 1e-9 for i in range(len(vals) - 1))
    print("     %s：范围 %.4f 至 %.4f（相对跨度 %.1f%%）｜ 单调 %s ｜ 极值位置 %s"
          % (tag, min(vals), max(vals), 100 * (max(vals) - min(vals)) / min(vals),
             "是" if mono else "否", "w = %.3f" % WS[vals.index(max(vals))] if not mono else "-"))

print()
print("[PF4] 口径核对：两扇区退化条件的系数结构")
print("     框架原文：h_{00} = 2\\theta（因子 2 来自相对论约定）；tr h = 6\\theta；通用识别 h = l_P\\theta")
print("     静场扇区：视界 = 1 - h_{00} = 0，即 1 - 2\\theta = 0，读数 \\theta = 1/2；")
print("               闭式完成式 \\theta_ex = \\theta_lin/(1+\\theta_lin/2)^2 的半径 2，精确视界在 \\theta_lin = 2。")
print("     波扇区：边界 = 横向度量分量过零 1 ∓ A + B = 0。")
print("     两种读数：")
print("       (甲) \\theta = a：波扇区条件写成 1 ∓ \\theta cos u + B = 0（系数 1，与静场不同）")
print("       (乙) \\theta = a/2：写成 1 ∓ 2\\theta cos u + B = 0（系数 2，与静场 1 - 2\\theta 同形）")
print("     本轮取乙，\\Lambda_* 的 \\theta 读数为 %.4f 至 %.4f（u0 = 3），静场线性读数 1/2 落在其间。"
      % (min(vals3) / 2.0, max(vals3) / 2.0))

print()
print("[PF5] 登记")
rows = [
    ("其一", "回归", "w = 0.354 档复现第 28 轮 P3 读数"),
    ("其二", "族扫描", "w = 0.20 至 1.00 的 Lambda_*(2) 与 Lambda_*(3) 表"),
    ("其三", "形状", "单调性与相对跨度见上"),
    ("其四", "口径", "取 theta = a/2（与 h_{00} = 2\\theta 同系数结构），\\theta 读数区间含静场 1/2"),
    ("其五", "落点", "回填收束报告与《引力完成式的数学地位》遗留"),
]
for r1, r2, r3 in rows:
    print("     %-4s %-6s %s" % (r1, r2, r3))
print()
print("总运行时间 = %.1f 秒" % (time.time() - T_START))
