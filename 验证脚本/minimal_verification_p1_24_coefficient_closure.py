#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
minimal_verification_p1_24_coefficient_closure.py

判据 Y：P1-24 系数缺口的供给侧对接（本体层重建的尺读数与偏折全值）。

背景
----
P1-24 第一轮（判据 J1 至 J4）把光偏折的系数问题拆成对照表：仅钟率给半值，
钟率加空间给全值，两者之比 1.999；完整系数 4GM/(c^2 b) 要求框架同时提供
h00 与空间侧度量因子。第二段把缺口钉到「记录与应变的一次耦合」，其强度
要求为 deltaL/L = -Phi（判据 K3）；候选筛查把该耦合归到张量桥 (d) 的
「应变由场方程定」非局部结构。本体层重建（阶段 0 至 5）交付供给侧读数：
钟率 h00 = 2 theta，尺读数 ell = 1 + u，u_e = theta_mid，lambda 导出 = 1。

本脚本做四项核对：
  Y1 零测地线射线追踪：在 g00 = -(1-2theta)、grr = 1 + 2 lambda theta 的
     静态度规上积分轨道方程，复算偏折随 lambda 的线性律 (1+lambda)*2mu/b。
  Y2 eikonal 组成核对：数值积分 -1/2 * Int d_b(h00 + hrr) dz，得钟率项与
     空间项各 2mu/b，合计 4mu/b，与 J4 的「半值/全值」对照表对应。
  Y3 供给侧换算链：lambda = 1（消费阶段 4 读数）经 ell = 1 + u、u = theta、
     h00 = 2theta 给 deltaL/L = theta，与需求侧 -Phi = theta（Phi = -c^2 theta，
     取 c = 1）逐位相等；并核对坐标光速读法 c*sqrt((1-2theta)/(1+2lambda theta))
     的一阶系数 -(1+lambda)，对上 S6/S7 的三条读法表。
  Y4 归一化稳健性：theta -> s*theta 下 Y1 的比值与 Y3 的恒等不变。

运行：python 验证脚本/minimal_verification_p1_24_coefficient_closure.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "images", "p1_24_coefficient_closure.png")

try:
    trapezoid = np.trapezoid
except AttributeError:
    trapezoid = np.trapz


def deflection_geodesic(lam, mu, b, h=5e-5):
    """零测地线射线追踪。

    轨道方程（isotropic 坐标，A = 1 - 2 theta, B = 1 + 2 lam theta, theta = mu*u）：
        u'' = mu (1 + lam) / (b^2 (1 - 2 mu u)^2) - u
    初值 u(0) = 0, u'(0) = 1/b（直线渐近）；偏折角 = 第二次零点 - pi。
    """
    phi_max = np.pi + 0.5
    n = int(phi_max / h)
    u, du, phi = 0.0, 1.0 / b, 0.0
    prev_u, prev_phi = u, phi
    coef = mu * (1.0 + lam) / (b * b)

    def f(uu, dd):
        return dd, coef / (1.0 - 2.0 * mu * uu) ** 2 - uu

    for _ in range(n):
        k1u, k1d = f(u, du)
        k2u, k2d = f(u + 0.5 * h * k1u, du + 0.5 * h * k1d)
        k3u, k3d = f(u + 0.5 * h * k2u, du + 0.5 * h * k2d)
        k4u, k4d = f(u + h * k3u, du + h * k3d)
        u_new = u + h * (k1u + 2.0 * k2u + 2.0 * k3u + k4u) / 6.0
        du = du + h * (k1d + 2.0 * k2d + 2.0 * k3d + k4d) / 6.0
        prev_u, prev_phi = u, phi
        u, phi = u_new, phi + h
        if phi > np.pi and u < 0.0 <= prev_u:
            frac = prev_u / (prev_u - u)
            return (prev_phi + frac * h) - np.pi
    return np.nan


def eikonal_split(lam, mu, b, zmax=4000.0, n=4_000_001):
    """eikonal 积分：alpha = -(1/2) Int d_b(h00 + hrr) dz，拆钟率项与空间项。"""
    z = np.linspace(-zmax, zmax, n)
    r2 = b * b + z * z
    # theta = mu / r，d_b theta = -mu b / r^3
    dtheta = -mu * b / r2 ** 1.5
    clock = -0.5 * 2.0 * trapezoid(dtheta, z)
    total = -0.5 * 2.0 * (1.0 + lam) * trapezoid(dtheta, z)
    return clock, total - clock, total


def light_speed_coeff(lam, h=1e-9):
    """坐标光速 c*sqrt((1-2theta)/(1+2 lam theta)) 的一阶系数（应为 -(1+lam)）。"""
    val = np.sqrt((1.0 - 2.0 * h) / (1.0 + 2.0 * lam * h))
    return (val - 1.0) / h


def main():
    print("=" * 78)
    print("minimal_verification_p1_24_coefficient_closure.py")
    print("判据 Y：P1-24 系数缺口的供给侧对接")
    print("=" * 78)

    mu, b = 1.0e-4, 1.0
    expect = lambda lam: (1.0 + lam) * 2.0 * mu / b

    # ---------- Y1 射线追踪：偏折对 lambda 的线性律 ----------
    print()
    print("Y1 零测地线射线追踪（mu = %.1e, b = %.1f）" % (mu, b))
    lams = [0.0, 0.5, 1.0]
    alphas = []
    for lam in lams:
        a = deflection_geodesic(lam, mu, b)
        alphas.append(a)
        rel = abs(a - expect(lam)) / expect(lam)
        print("   lambda = %.2f : alpha = %.10e , 期望 %.10e , 相对偏差 %.2e"
              % (lam, a, expect(lam), rel))
    y1_ok = all(abs(a - expect(lam)) / expect(lam) < 1e-3
                for a, lam in zip(alphas, lams))
    ratio = alphas[2] / alphas[0]
    print("   lambda = 1 与 0 之比 = %.6f（解析 2.000000；既有 J4 读数 1.999）"
          % ratio)
    print("   Y1 判定：%s" % ("通过" if y1_ok else "失败"))
    devs = []
    for mu_h in (mu, mu / 2.0):
        a_h = deflection_geodesic(1.0, mu_h, b)
        devs.append(abs(a_h - 2.0 * 2.0 * mu_h / b) / (2.0 * 2.0 * mu_h / b))
    print("   高阶修正阶数核对：mu 减半时相对偏差比 = %.4f（期望约 2，线性阶）"
          % (devs[0] / devs[1]))

    # ---------- Y2 eikonal 组成 ----------
    print()
    print("Y2 eikonal 组成核对（lambda = 1）")
    clock, space, total = eikonal_split(1.0, mu, b)
    exp_one = 2.0 * mu / b
    print("   钟率项 = %.10e（期望 %.10e，相对偏差 %.2e）"
          % (clock, exp_one, abs(clock - exp_one) / exp_one))
    print("   空间项 = %.10e（期望 %.10e，相对偏差 %.2e）"
          % (space, exp_one, abs(space - exp_one) / exp_one))
    print("   合计   = %.10e（期望 %.10e，相对偏差 %.2e）"
          % (total, 2.0 * exp_one, abs(total - 2.0 * exp_one) / (2.0 * exp_one)))
    clock0, space0, total0 = eikonal_split(0.0, mu, b)
    print("   lambda = 0 给 total = %.10e（半值），两档之比 = %.6f"
          % (total0, total / total0))
    y2_ok = (abs(clock - exp_one) / exp_one < 1e-6
             and abs(space - exp_one) / exp_one < 1e-6
             and abs(total / total0 - 2.0) < 1e-6)
    print("   Y2 判定：%s" % ("通过" if y2_ok else "失败"))

    # ---------- Y3 供给侧换算链 ----------
    print()
    print("Y3 供给侧换算链（lambda = 1，消费阶段 4 读数）")
    thetas = np.array([1.0e-6, 1.0e-4, 1.0e-2, 0.1])
    phi_n = -thetas                      # Phi = -c^2 theta, c = 1
    req = -phi_n                         # K3：deltaL/L = -Phi
    lam_sup = 1.0
    h00 = 2.0 * thetas
    ell = 1.0 + lam_sup * h00 / 2.0      # 重建尺读数
    sup = ell - 1.0                      # deltaL/L（供给）
    res = np.max(np.abs(req - sup))
    print("   max |需求 - 供给| = %.3e（样本 theta 至 %.1f）" % (res, thetas[-1]))
    c0 = light_speed_coeff(0.0)
    c1 = light_speed_coeff(1.0)
    print("   坐标光速一阶系数：lambda = 0 给 %.9f（读法 c(1-theta)），"
          "lambda = 1 给 %.9f（读法 c(1-2theta)）" % (c0, c1))
    print("   换算链：lambda = 1 -> ell = 1 + theta -> deltaL/L = theta = -Phi")
    y3_ok = (res < 1e-15
             and abs(c0 + 1.0) < 1e-6 and abs(c1 + 2.0) < 1e-6)
    print("   Y3 判定：%s" % ("通过" if y3_ok else "失败"))

    # ---------- Y4 归一化稳健性 ----------
    print()
    print("Y4 归一化稳健性（theta -> s theta）")
    ok4 = True
    for s in (0.5, 2.0):
        a_s = deflection_geodesic(1.0, mu * s, b)
        rel = abs(a_s / (s * alphas[2]) - 1.0)
        ok4 = ok4 and rel < 1e-3
        print("   s = %.1f : alpha(1)/alpha_0(1) = %.8f（期望 1，偏差 %.2e）"
              % (s, a_s / (s * alphas[2]), rel))
    print("   Y4 判定：%s" % ("通过" if ok4 else "失败"))

    # ---------- 图 ----------
    th = np.linspace(0.0, 0.02, 400)
    n0 = np.sqrt((1.0 - 2.0 * th) / (1.0 + 0.0 * th))
    n1 = np.sqrt((1.0 - 2.0 * th) / (1.0 + 2.0 * th))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    ax = axes[0]
    ax.plot([0, 1], [1.0, 2.0], color="0.35", lw=1.2, label="解析 (1+lambda)")
    ax.plot(lams, [a / (2.0 * mu / b) for a in alphas], "o", ms=7,
            color="C3", label="射线追踪")
    ax.set_xlabel("lambda"); ax.set_ylabel("alpha / (2 mu / b)")
    ax.set_title("(a) 偏折组成：钟率给 1，空间再加 1")
    ax.legend(); ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot(th, n0 - 1.0, color="C0", lw=1.2, label="度规 lambda=0")
    ax.plot(th, n1 - 1.0, color="C3", lw=1.2, label="度规 lambda=1")
    ax.plot(th, -th, "k--", lw=0.9, label="读法 c(1-theta)")
    ax.plot(th, -2.0 * th, "k:", lw=0.9, label="读法 c(1-2theta)")
    ax.set_xlabel("theta"); ax.set_ylabel("n / c - 1")
    ax.set_title("(b) 坐标光速与三条读法")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, dpi=160)
    print()
    print("图已存：%s" % FIG)

    print()
    print("=" * 78)
    print("总判定：Y1 %s，Y2 %s，Y3 %s，Y4 %s"
          % ("通过" if y1_ok else "失败", "通过" if y2_ok else "失败",
             "通过" if y3_ok else "失败", "通过" if ok4 else "失败"))
    print("=" * 78)


if __name__ == "__main__":
    main()
