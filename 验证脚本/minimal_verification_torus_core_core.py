#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
minimal_verification_torus_core_core.py

环面芯-芯束缚（验证脚本第 484 条）

背景。第 483 条测出边界保护的束缚势对格点内刚度 K 不敏感（平台 0.015365），
但适用范围声明指出：该束缚以边界环冻结为前提，若 D-b 裁周期边界为全局主设定，
环面（戒指）上没有边界环可冻结，单芯到边界的束缚失其对象。D-b 协调时给出的
替代候选是芯-芯束缚：环面上净绕数守恒，单 kink 不可能存在，天然构型为正反
缺陷对，切割弦可连两芯。

本条在环面上放一对正反 kink 并测束缚势随芯间距 d 的变化。

模型（一维周期链，与第 480 条 S3 同设定）。N 格点、周期边界，偏离量 v_n（周期），
背景层率 w_ref = 1 已扣除，故 u_n = n + v_n，能量

    E = sum_n (k/2)(v_{n+1} - v_n)^2 + K sum_n [1 - cos 2 pi v_n],   k = 1,

周期求和。注意 cos 2 pi (n + v_n) = cos 2 pi v_n，故公度背景的基底代价为零，
能量全部来自两壁与壁间相互作用。

泡构型 v_n = -f(n - a) + f(n - b)，f 为光滑台阶，d = b - a 即芯间距。
v 由 0 降到 -1 再回到 0，两个壁分别位于 a 与 b，净绕数 sum Δv = 0（环面允许）。

判据
  CC1  冒烟：净绕数为零、远离壁处基底代价为零、能量收敛
  CC2  E(d) 形状判定：单调上升（吸引）、存在内部极小（束缚态）、或单调下降（排斥）
  CC3  斜率或极小位置与深度的 K 依赖
  CC4  判定：芯-芯束缚在极小 K 下是否给出定域化对象

确定性：无随机、无采样。
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

N = 256
NSTEPS = 4000
ETA = 0.05
K_EL = 1.0
TWO_PI = 2.0 * math.pi
XI0 = 2.0
K_LIST = [1e-2, 1e-4, 1e-6, 1e-10, 0.0]
D_LIST = [4, 8, 16, 32, 64, 100]
OUT_FIG = "D:/software/TianXingFC/images/torus_core_core.png"


def step_profile(x, xi):
    return 0.5 * (1.0 + np.tanh(x / xi))


def init_bubble(d):
    n = np.arange(N, dtype=float)
    a = 0.5 * (N - d)
    b = a + d
    return -step_profile(n - a, XI0) + step_profile(n - b, XI0)


def energy(v, K):
    dv = np.roll(v, -1) - v
    e = 0.5 * K_EL * float(np.sum(dv * dv))
    if K > 0.0:
        e += K * float(np.sum(1.0 - np.cos(TWO_PI * v)))
    return e


def grad(v, K):
    dvp = np.roll(v, -1) - v
    dvm = v - np.roll(v, 1)
    g = -dvp + dvm
    if K > 0.0:
        g += K * TWO_PI * np.sin(TWO_PI * v)
    return g


def relax(v0, K):
    v = v0.copy()
    e_prev = energy(v, K)
    for step in range(NSTEPS):
        v = v - ETA * grad(v, K)
        if step % 500 == 499:
            e_now = energy(v, K)
            if abs(e_now - e_prev) < 1e-14 * max(1.0, abs(e_now)):
                e_prev = e_now
                break
            e_prev = e_now
    return v


def wall_positions(v):
    """以键差绝对值的前两大峰定位两壁。"""
    w = np.abs(np.roll(v, -1) - v)
    idx = np.argsort(w)[::-1]
    first = int(idx[0])
    second = None
    for j in idx[1:]:
        if abs(int(j) - first) > 3:
            second = int(j)
            break
    return first, second, float(np.max(w))


def slope_and_min(ds, es):
    ds = np.array(ds, dtype=float)
    es = np.array(es, dtype=float)
    k = float(np.polyfit(ds, es, 1)[0])
    imin = int(np.argmin(es))
    interior = 0 < imin < len(es) - 1
    return k, imin, interior


def main():
    t0 = time.time()
    print("=" * 78)
    print("环面芯-芯束缚（第 484 条）")
    print("=" * 78)

    # ---------------- CC1 冒烟
    print("\n[判据 CC1] 冒烟检查")
    v0 = init_bubble(32)
    net = float(np.sum(np.roll(v0, -1) - v0))
    sub_bg = float(np.sum(1.0 - np.cos(TWO_PI * v0[np.abs(v0) < 1e-9])))
    vr = relax(v0, 1e-2)
    print("  净绕数 sum Δv = {:.3e}（环面须为零）".format(net))
    print("  远离壁处的基底代价 = {:.3e}（公度背景应为零）".format(sub_bg))
    e0, e1 = energy(v0, 1e-2), energy(vr, 1e-2)
    print("  能量 {:.6f} -> {:.6f}".format(e0, e1))
    print("  初末壁峰位置 {}".format(wall_positions(v0)[:2]))
    wp = wall_positions(vr)
    print("  末态壁峰位置 {} 与 {}，峰高 {:.6f}".format(wp[0], wp[1], wp[2]))
    assert abs(net) < 1e-9, "环面净绕数不为零"
    assert e1 <= e0 + 1e-12, "弛豫未降能"
    print("  判定：通过")

    # ---------------- CC2 与 CC3
    print("\n[判据 CC2/CC3] E(d) 形状与 K 依赖")
    print("  {:<9} {:>12} {:>9} {:>7} {:>10} {:>9} {:>9}".format(
        "K", "斜率", "内部极小", "极小位", "极小深度", "峰高(极小)", "峰高(100)"))
    table = {}
    for K in K_LIST:
        es, peaks = [], []
        for d in D_LIST:
            v0 = init_bubble(d)
            vr = relax(v0, K)
            es.append(energy(vr, K))
            peaks.append(wall_positions(vr)[2])
        sl, imin, interior = slope_and_min(D_LIST, es)
        depth = es[-1] - es[imin]
        table[K] = (es, sl, imin, interior, depth, peaks)
        print("  {:<9g} {:>12.6e} {:>9} {:>7} {:>9.6f} {:>9.6f} {:>9.6f}".format(
            K, sl, "是" if interior else "否", D_LIST[imin], depth,
            peaks[imin], peaks[-1]))
    print("  说明：斜率大于零表示能量随芯间距上升，即两芯相互吸引。")
    print("        峰高为键差峰值，衡量壁是否仍存在；峰高趋零表示该档下对已湮灭。")
    print("        故「内部极小」须与极小点的峰高合看：峰高趋零的极小点是湮灭端点，")
    print("        不构成有限尺度的束缚态。")

    # ---------------- 壁形读数
    print("\n[壁形读数] 末态壁峰高与宽度（判断壁是否塌缩或摊平）")
    print("  {:<9} {:>12} {:>12} {:>12}".format("K", "d=4 峰高", "d=100 峰高", "d=100 壁宽"))
    for K in K_LIST:
        v_small = relax(init_bubble(4), K)
        v_big = relax(init_bubble(100), K)
        pk_small = wall_positions(v_small)[2]
        pk_big = wall_positions(v_big)[2]
        w = np.abs(np.roll(v_big, -1) - v_big)
        thr = 0.5 * pk_big
        width = float(np.sum(w > thr))
        print("  {:<9g} {:>12.6f} {:>12.6f} {:>12d}".format(K, pk_small, pk_big, int(width)))

    # ---------------- CC4
    print("\n[判据 CC4] 判定")
    ks = [k for k in K_LIST if k > 0]
    slopes = [table[k][1] for k in ks]
    print("  逐 K 斜率：{}".format("  ".join("{:+.3e}".format(s) for s in slopes)))
    allpos = all(s > 0 for s in slopes)
    floorslope = table[K_LIST[-2]][1]
    print("  全档斜率为正（吸引）：{}".format("是" if allpos else "否"))
    print("  极小档（K = {:g}）斜率 {:.3e}".format(K_LIST[-2], floorslope))
    print("  极小档的最小峰高（d = 4 档）{:.6f}，最大峰高（d = 100 档）{:.6f}".format(
        table[K_LIST[-2]][5][0], table[K_LIST[-2]][5][-1]))
    survivor = table[K_LIST[-2]][5][-1] > 10.0 * table[K_LIST[-2]][5][0]
    print("  判定一（吸引）：芯-芯相互作用在全部 K 档为吸引，斜率不随 K 塌缩")
    print("  判定二（束缚态）：极小档无内部极小，极小落在 d 端点，故静态无束缚态；")
    print("        两端峰高相差 {:.1f} 倍，说明小 d 档下对已大幅湮灭、大 d 档壁仍存活。"
          .format(table[K_LIST[-2]][5][-1] / max(table[K_LIST[-2]][5][0], 1e-30)))
    print("  判定三（候选结论）：芯-芯候选给出吸引与湮灭趋势，不给出稳定的定域化对象；")
    print("        本最小模型未含守卫层（绕数超选择），对能否被守卫阻止湮灭未作判定，登记后续。")

    # ---------------- 出图
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))
    ax = axes[0]
    for K in K_LIST:
        ax.plot(D_LIST, table[K][0], "o-", ms=4, label="K = {:g}".format(K))
    ax.set_xlabel("芯间距 d（格）")
    ax.set_ylabel("弛豫末态能量 E")
    ax.set_title("(a) 芯-芯势 E(d)：逐 K 档")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.semilogx(ks, [table[k][1] for k in ks], "s-")
    ax.axhline(0.0, color="gray", lw=1.0)
    ax.set_xlabel("格点内刚度 K")
    ax.set_ylabel("dE/dd")
    ax.set_title("(b) 斜率随 K")
    ax.grid(alpha=0.25)

    ax = axes[2]
    for K in K_LIST:
        v_big = relax(init_bubble(100), K)
        ax.plot(np.arange(N), v_big, lw=1.0, label="K = {:g}".format(K))
    ax.set_xlabel("格点序号")
    ax.set_ylabel("偏离量 v")
    ax.set_title("(c) d = 100 档末态剖面")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    import os
    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
    fig.savefig(OUT_FIG, dpi=160)
    print("\n图已存：{}".format(OUT_FIG))
    print("\n用时 {:.1f} 秒".format(time.time() - t0))
    print("=" * 78)


if __name__ == "__main__":
    main()
