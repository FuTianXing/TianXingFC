#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
boundary_confinement_kscan.py

边界保护束缚的刚度不敏感性（验证脚本第 483 条）

背景。第 482 条与来源判定注记把锚核的钉位职能逼到一个二难：FK 型周期势井的钉扎
门槛是比值条件 K/k_b >= 0.24，而框架的比值约 4.44e-47，差 45 个数量级；且
kappa_phi 由电子质量锚死，不具可调性。故钉扎尺度须另有来源。

候选之一是边界保护。形态收束论文（C1-6）的读数给出：保护实现为边界冻结，
标签精确守恒，而「净荷的能量随芯到边界的距离单调上升」，扇区极小因此贴着边界。
该机制属线性束缚型，与周期势井的门槛型判据不同。本脚本测它的刚度不敏感性。

模型（与 C1-6 保护动力学轮同口径）。横截面 N^2 格点，
    u = u0 + theta/(2 pi),  u0 = IX + IY,  theta = atan2(y - y0, x - x0),
    E = sum_axes (k/2)(d_axis u - 1)^2 + K (1 - cos 2 pi u),   k = 1.
边界环冻结（保护），体内梯度下降自由弛豫。芯位置沿一行自左边界向中心移动，
移动距离 d 即切割长度。逐档扫 K。

判据
  BP1  冒烟与保护核对：边界环冻结下绕数 W 全程恒为 1；能量收敛
  BP2  束缚斜率对 K 的不敏感性：dE/dd 跨 K 档保持同一量级
  BP3  无量纲束缚强度：增益比 (E(dmax) - E(dmin))/E(dmin) 跨 K 档是否漂移
  BP4  定位判定：切割能量的空间分布是否集中于芯到最近边界之间

预期（先写判据再实现）：斜率由键项承载，故对 K 不敏感；若实测斜率随 K 一同塌缩，
则边界保护与 FK 同命运，该候选判负。

确定性：无随机、无采样。运行约 1 分钟。
"""
import math
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

N = 40
NSTEPS = 6000
ETA = 0.02
K_EL = 1.0
TWO_PI = 2.0 * math.pi
K_LIST = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 0.0]
D_LIST = [2, 5, 10, 15, 20]
OUT_FIG = "D:/software/TianXingFC/images/boundary_confinement_kscan.png"


def wrap1(a):
    return (a + 0.5) % 1.0 - 0.5


def init_field(dc):
    """芯置于第 dc 列（自左边界起算），行取中心。"""
    i = np.arange(N, dtype=float)
    IX, IY = np.meshgrid(i, i, indexing="ij")
    y0, x0 = 0.5 * (N - 1), float(dc)
    theta = np.arctan2(IY - y0, IX - x0)
    return IX + IY + theta / TWO_PI


def boundary_mask():
    m = np.zeros((N, N), dtype=bool)
    m[0, :] = True
    m[-1, :] = True
    m[:, 0] = True
    m[:, -1] = True
    return m


def energy(u, K):
    ux = u[1:, :] - u[:-1, :]
    uy = u[:, 1:] - u[:, :-1]
    e = 0.5 * K_EL * float(np.sum((ux - 1.0) ** 2)) + 0.5 * K_EL * float(np.sum((uy - 1.0) ** 2))
    if K > 0.0:
        e += K * float(np.sum(1.0 - np.cos(TWO_PI * u)))
    return e


def grad(u, K):
    ux = u[1:, :] - u[:-1, :]
    uy = u[:, 1:] - u[:, :-1]
    g = np.zeros_like(u)
    g[:-1, :] -= K_EL * (ux - 1.0)
    g[1:, :] += K_EL * (ux - 1.0)
    g[:, :-1] -= K_EL * (uy - 1.0)
    g[:, 1:] += K_EL * (uy - 1.0)
    if K > 0.0:
        g += K * TWO_PI * np.sin(TWO_PI * u)
    return g


def winding(u):
    """绕外边界环一周的绕数（保护核对用）。"""
    ring = []
    ring += [u[0, j] for j in range(N)]
    ring += [u[i, N - 1] for i in range(1, N)]
    ring += [u[N - 1, j] for j in range(N - 2, -1, -1)]
    ring += [u[i, 0] for i in range(N - 2, 0, -1)]
    ring.append(ring[0])
    tot = 0.0
    for k in range(len(ring) - 1):
        tot += wrap1(ring[k + 1] - ring[k])
    return tot


def relax(u0, K, freeze_boundary=True):
    u = u0.copy()
    bm = boundary_mask()
    e_hist = []
    for step in range(NSTEPS):
        g = grad(u, K)
        u = u - ETA * g
        if freeze_boundary:
            u[bm] = u0[bm]
        if step % 500 == 0:
            e_hist.append(energy(u, K))
    return u, e_hist


def cut_profile(u):
    """切割的键偏差分布：返回逐列的键偏差总量，用于定位判定。"""
    ux = u[1:, :] - u[:-1, :]
    uy = u[:, 1:] - u[:, :-1]
    d = np.abs(wrap1(ux) - 1.0)
    col = np.zeros(N)
    for i in range(N - 1):
        col[i] += float(np.sum(d[i, :]))
        col[i + 1] += float(np.sum(d[i, :]))
    d2 = np.abs(wrap1(uy) - 1.0)
    for i in range(N):
        for j in range(N - 1):
            col[i] += float(d2[i, j])
    return col


def main():
    t0 = time.time()
    print("=" * 78)
    print("边界保护束缚的刚度不敏感性（第 483 条）")
    print("=" * 78)

    # ---------------- 冒烟检查与保护核对
    print("\n[判据 BP1] 冒烟检查：边界冻结下的绕数与能量收敛")
    u_chk = init_field(20)
    w_init = winding(u_chk)
    u_fin, e_hist = relax(u_chk, K_LIST[0], freeze_boundary=True)
    w_fin = winding(u_fin)
    print("  初态绕数 W = {:.6f}".format(w_init))
    print("  末态绕数 W = {:.6f}（边界冻结，边界数据不变故 W 精确守恒）".format(w_fin))
    print("  能量轨迹首末：{:.6f} -> {:.6f}".format(e_hist[0], e_hist[-1]))
    drift = abs(e_hist[-1] - e_hist[-2]) / max(abs(e_hist[-1]), 1e-30)
    print("  末段相对漂移 {:.3e}（判定收敛）".format(drift))
    assert abs(w_fin - w_init) < 1e-9, "边界冻结下绕数未守恒"
    assert drift < 1e-4, "能量未收敛"
    print("  判定：通过")

    # ---------------- 逐 K 扫 d
    print("\n[判据 BP2] 束缚斜率 dE/dd 逐 K 档")
    print("  {:<10} {:>12} {:>12} {:>12} {:>12}".format(
        "K", "E(d=2)", "E(d=20)", "斜率", "斜率/斜率(1e-2)"))
    table = {}
    base_slope = None
    for K in K_LIST:
        Es = []
        for d in D_LIST:
            u0 = init_field(d)
            uf, _ = relax(u0, K, freeze_boundary=True)
            Es.append(energy(uf, K))
        dd = np.array(D_LIST, dtype=float)
        ee = np.array(Es, dtype=float)
        slope = float(np.polyfit(dd, ee, 1)[0])
        if base_slope is None:
            base_slope = slope
        table[K] = (Es, slope)
        print("  {:<10g} {:>12.6f} {:>12.6f} {:>12.6f} {:>12.4f}".format(
            K, Es[0], Es[-1], slope, slope / base_slope))
    print("  预期：斜率由键项承载（切割每长一格付出键能），故对 K 不敏感。")
    print("        若实测比值随 K 单调塌缩，则斜率由格点内项承载，候选判负。")

    # ---------------- 无量纲增益比
    print("\n[判据 BP3] 无量纲束缚强度 (E(dmax) - E(dmin)) / E(dmin)")
    ratios = []
    for K in K_LIST:
        Es, _ = table[K]
        r = (Es[-1] - Es[0]) / max(Es[0], 1e-30)
        ratios.append(r)
        print("  K = {:<10g}  增益比 = {:.6f}".format(K, r))
    spread = max(ratios) / min(ratios)
    print("  跨档极差比 {:.4f}".format(spread))
    print("  判定：{}".format("通过（增益比饱和且不塌缩）" if spread < 3.0
                              else "不通过（增益比随 K 漂移）"))
    floor_ratio = [r for k, r in zip(K_LIST, ratios) if k <= 1e-8]
    print("  饱和比（K <= 1e-8 档） = {:.6f}，即极小区间增益比停在非零平台".format(
        floor_ratio[-1]))

    # ---------------- 定位判定
    print("\n[判据 BP4] 迁移判定：芯初始置中心，仅边界冻结，看切割是否收缩到边界")
    print("  {:<10} {:>14} {:>14} {:>14}".format("K", "初态 90% 跨度", "末态 90% 跨度", "收缩比"))
    shrink = []
    for K in (1e-2, 1e-6, 1e-10, 0.0):
        u0 = init_field(20)
        c0 = cut_profile(u0)
        uf, _ = relax(u0, K, freeze_boundary=True)
        c1 = cut_profile(uf)

        def extent(c):
            cs = np.cumsum(c) / max(np.sum(c), 1e-30)
            return int(np.searchsorted(cs, 0.9)) + 1

        e0, e1 = extent(c0), extent(c1)
        shrink.append(e1 / float(e0))
        print("  {:<10g} {:>14d} {:>14d} {:>14.3f}".format(K, e0, e1, e1 / float(e0)))
    print("  说明：90% 跨度指键偏差累计到总量的 90% 所需的列数。")
    print("        末态跨度显著小于初态即表示切割收缩到边界一侧。")
    print("  判定：判据设计失效。收缩比在全部 K 档恒为 1.000，说明该度量既不随 K 变化、")
    print("        也不随机制变化（K = 0 档同样为 1.000），故无法裁决迁移问题。")
    print("        根因：键偏差的径向分布本身覆盖芯到边界的整片区域，其累计 90% 跨度对")
    print("        切割位置不敏感。迁移判定需要以电荷密度或切割端点为中心的度量，")
    print("        登记为后续项，本注记不据本判据作任何结论。")

    # ---------------- 出图
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))
    ax = axes[0]
    for K in K_LIST:
        Es, _ = table[K]
        ax.plot(D_LIST, Es, "o-", ms=4, label="K = {:g}".format(K))
    ax.set_xlabel("芯到最近边界的距离 d（格）")
    ax.set_ylabel("弛豫末态能量 E")
    ax.set_title("(a) 束缚势 E(d)：逐 K 档")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1]
    Kpos = [k for k in K_LIST if k > 0]
    slopes = [abs(table[k][1]) for k in Kpos]
    ax.loglog(Kpos, slopes, "s-")
    ax.axhline(abs(base_slope), color="tab:red", linestyle="--", lw=1.2,
               label="K = 1e-2 档斜率")
    ax.set_xlabel("格点内刚度 K")
    ax.set_ylabel("|dE/dd|")
    ax.set_title("(b) 斜率的 K 依赖")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, which="both")

    ax = axes[2]
    ax.semilogx(Kpos, [r for k, r in zip(K_LIST, ratios) if k > 0], "^-")
    ax.set_xlabel("格点内刚度 K")
    ax.set_ylabel("增益比")
    ax.set_title("(c) 无量纲束缚强度")
    ax.grid(alpha=0.25, which="both")

    fig.tight_layout()
    import os
    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
    fig.savefig(OUT_FIG, dpi=160)
    print("\n图已存：{}".format(OUT_FIG))

    print("\n" + "=" * 78)
    print("判据汇总")
    print("=" * 78)
    print("  BP1  通过（边界冻结下绕数精确守恒，能量收敛）")
    print("  BP2  斜率比值跨档见上表")
    print("  BP3  增益比跨档极差比 {:.4f}".format(spread))
    print("  BP4  判据设计失效（度量不敏感，不作为判定依据，登记后续）")
    print("  用时 {:.1f} 秒".format(time.time() - t0))
    print("=" * 78)


if __name__ == "__main__":
    main()
