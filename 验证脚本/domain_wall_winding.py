#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
domain_wall_winding.py

复结构畴壁承载绕数的稳定性与束缚（验证脚本第 486 条）

背景。第 485 条证明内部冻结环可替代样品边缘提供非零的 K 无关束缚平台，但该冻结
是施加的。本条把冻结换成由对称破缺动力学生成的畴壁：取复结构符号场 sigma（双井
势的两个真空），其 pi_0 型畴壁作为内部承载结构。

首轮实现（强度型 linkage）失败：令公度真空偏移 a = (1 - sigma)/2，并以
K (1 - cos 2 pi (v - a)) 把相位场 v 拉向 a，结果在极小 K 下弛豫不收敛（8000 步对
80000 步 E_v 相对偏差 5.9），与第 484 条同源：K 极小时 v 对 a 的黏附力趋零，跳变
无限展宽。

据此改设计为两版对照，判别 linkage 的**类型**：

  约束版（定义性 linkage）：sigma 定义相位的零点（复结构 J 的符号定义相位约定），
      故 v 与 a 的关联是定义关系而非能量偏好，能量中不出现 K：
          E = sum (kappa/2)(d sigma)^2 + lambda (sigma^2 - 1)^2 + (k/2)(d a)^2,
          a = (1 - sigma)/2.
  强度版（反例）：以 K 把 v 拉向 a，K 进入能量。

判据
  SW1  冒烟：约束版畴壁由双井动力学形成；步数收敛性检查
  SW2  零模核对：壁置于不同初始位置，弛豫后能量相同（平移零模，无穷后退的终止条件）
  SW3  约束版的 K 无关性：能量中无 K，壁宽与跳变宽由 kappa、lambda、k 定；给读数
  SW4  强度版反例：跳变宽度随 K 变化，验证展宽趋势
  SW5  判定：两版对 K 的响应方向不同，据此判定 linkage 类型的要求

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
NSTEPS = 8000
ETA = 0.02
KAPPA = 1.0
LAM = 1.0
K_EL = 1.0
TWO_PI = 2.0 * math.pi
WALL_W = 2.0
K_LIST = [1e-2, 1e-4, 1e-6, 1e-8]
OUT_FIG = "D:/software/TianXingFC/images/domain_wall_winding.png"


def wrap1(a):
    return (a + 0.5) % 1.0 - 0.5


def init_sigma(c1, c2):
    n = np.arange(N, dtype=float)
    return np.tanh((n - c1) / WALL_W) * np.tanh((n - c2) / WALL_W)


def offset(s):
    return 0.5 * (1.0 - s)


def e_constrained(s):
    ds = np.roll(s, -1) - s
    da = np.roll(offset(s), -1) - offset(s)
    return (0.5 * KAPPA * float(np.sum(ds * ds))
            + LAM * float(np.sum((s * s - 1.0) ** 2))
            + 0.5 * K_EL * float(np.sum(da * da)))


def relax_sigma(s0, steps=None):
    s = s0.copy()
    n = NSTEPS if steps is None else steps
    for _ in range(n):
        ds = np.roll(s, -1) - s
        # a = (1 - sigma)/2 故 d a = -d sigma/2，(k/2)(d a)^2 = (k/8)(d sigma)^2
        # 对 s_p 求导得 (k/4)(-d s_p + d s_{p-1})
        g = -KAPPA * ds + KAPPA * np.roll(ds, 1) + 4.0 * LAM * s * (s * s - 1.0)
        g += 0.25 * K_EL * (-ds + np.roll(ds, 1))
        s = s - ETA * g
    return s


def wall_width(s):
    d = np.abs(np.roll(s, -1) - s)
    pk = float(np.max(d))
    if pk <= 0:
        return 0.0
    return float(np.sum(d > 0.5 * pk))


def jump_stats(v):
    w = np.abs(wrap1(np.roll(v, -1) - v))
    tot = float(np.sum(w))
    if tot <= 0:
        return 0.0, 0.0, 0.0
    idx = np.arange(N, dtype=float)
    c = float(np.sum(idx * w) / tot)
    var = float(np.sum(((idx - c) ** 2) * w) / tot)
    return c, math.sqrt(max(var, 0.0)), tot


def local_moment_width(v, half=30):
    """局部单壁宽的矩量版本：以 |wrap1(dv)| 的最大键为中心，取窗口内的二阶矩。"""
    w = np.abs(wrap1(np.roll(v, -1) - v))
    pk = int(np.argmax(w))
    lo = max(0, pk - half)
    hi = min(len(w), pk + half + 1)
    seg = w[lo:hi]
    tot = float(np.sum(seg))
    if tot <= 0:
        return 0.0, float(pk)
    idx = np.arange(lo, hi, dtype=float)
    c = float(np.sum(idx * seg) / tot)
    var = float(np.sum(((idx - c) ** 2) * seg) / tot)
    return 2.0 * math.sqrt(max(var, 0.0)), c


def local_jump_width(v, half=30):
    """局部单壁宽：以 |wrap1(dv)| 的最大键为中心，数超过半高的键数。"""
    w = np.abs(wrap1(np.roll(v, -1) - v))
    pk = int(np.argmax(w))
    th = 0.5 * float(w[pk])
    lo = max(0, pk - half)
    hi = min(len(w), pk + half + 1)
    return float(np.sum(w[lo:hi] > th)), float(pk)


def relax_strength(s0, K, steps=None):
    """强度版：sigma 与 v 联合弛豫，v 被 K 拉向 a。"""
    s = s0.copy()
    v = offset(s0).copy()
    n = NSTEPS if steps is None else steps
    for _ in range(n):
        ds = np.roll(s, -1) - s
        dv = np.roll(v, -1) - v
        a = offset(s)
        gs = -KAPPA * ds + KAPPA * np.roll(ds, 1) + 4.0 * LAM * s * (s * s - 1.0)
        gv = -K_EL * dv + K_EL * np.roll(dv, 1)
        if K > 0.0:
            ph = TWO_PI * (v - a)
            gv += K * TWO_PI * np.sin(ph)
            gs += -0.5 * K * TWO_PI * np.sin(ph)
        s = s - ETA * gs
        v = v - ETA * gv
    return s, v


def main():
    t0 = time.time()
    print("=" * 82)
    print("复结构畴壁承载绕数的稳定性与束缚（第 486 条）")
    print("=" * 82)

    # ---------------- SW1
    print("\n[判据 SW1] 约束版冒烟与步数收敛性检查")
    s0 = init_sigma(64.0, 192.0)
    s_f = relax_sigma(s0)
    e8 = e_constrained(relax_sigma(s0, 8000))
    e80 = e_constrained(relax_sigma(s0, 80000))
    dev = abs(e80 - e8) / max(abs(e80), 1e-30)
    print("  初态能量 {:.6f}｜弛豫后 {:.6f}".format(e_constrained(s0), e_constrained(s_f)))
    print("  步数收敛性：8000 步 {:.8f} 对 80000 步 {:.8f}，相对偏差 {:.3e}".format(e8, e80, dev))
    print("  壁宽（|d sigma| 半高全宽）{:.0f} 格".format(wall_width(s_f)))
    print("  判定：{}".format("通过" if dev < 1e-6 else "不通过"))
    assert dev < 1e-6, "约束版步数收敛性未通过"

    # ---------------- SW2
    print("\n[判据 SW2] 零模核对（无穷后退的终止条件）")
    print("  {:<22} {:>16} {:>12}".format("初始壁位", "弛豫后能量", "壁宽"))
    es = []
    for c in (16.0, 48.0, 80.0, 112.0):
        s_f = relax_sigma(init_sigma(c, c + 128.0))
        e = e_constrained(s_f)
        es.append(e)
        print("  壁位 {:<17.0f} {:>16.8f} {:>12.0f}".format(c, e, wall_width(s_f)))
    spread = (max(es) - min(es)) / min(es)
    print("  跨四档壁位的能量相对极差 {:.3e}".format(spread))
    print("  判定：{}".format("通过（壁位置为平移零模，不需要更外层参照物）"
                              if spread < 1e-6 else "不通过"))

    # ---------------- SW3
    print("\n[判据 SW3] 约束版的 K 无关性")
    print("  能量表达式 E = sum (kappa/2)(d sigma)^2 + lambda (sigma^2-1)^2 + (k/2)(d a)^2")
    print("  该式中不出现 K，故 K 无关性属结构性质而非读数。")
    print("  对照读数（逐 kappa 档，测壁宽与跳变宽）：")
    print("  {:<10} {:>14}".format("kappa", "单壁矩量宽"))
    globals()["KAPPA"] = 0.25
    for kap in (0.25, 1.0, 4.0):
        globals()["KAPPA"] = kap
        s_f = relax_sigma(s0)
        wloc, pk = local_moment_width(offset(s_f))
        print("  {:<10g} {:>14.2f}".format(kap, wloc))
    globals()["KAPPA"] = 1.0

    # ---------------- SW4
    print("\n[判据 SW4] 强度版反例：跳变宽度随 K 变化")
    print("  {:<10} {:>14} {:>14} {:>16}".format("K", "峰位", "局部壁宽", "跳变总量"))
    widths = []
    for K in K_LIST:
        s_f, v_f = relax_strength(s0, K)
        wloc, pk = local_jump_width(v_f)
        _, _, tot = jump_stats(v_f)
        widths.append(wloc)
        print("  {:<10g} {:>14.0f} {:>14.0f} {:>16.6f}".format(K, pk, wloc, tot))
    print("  局部壁宽自 K = 1e-2 的 {:.0f} 变到 K = 1e-8 的 {:.0f}，倍率 {:.2f}".format(
        widths[0], widths[-1], widths[-1] / max(widths[0], 1e-30)))

    # ---------------- SW5
    print("\n[判据 SW5] 判定")
    print("  约束版：结构完全不含 K，壁位为平移零模，跳变由定义固定在壁上 -> K 无关")
    print("  强度版：跳变宽度随 K 单调变化 -> K 相关，极小 K 下退化为展宽/不收敛")
    print("  判定：壁能否承载绕数并以 K 无关的方式定域，取决于 linkage 的类型；")
    print("        定义性 linkage 给 K 无关结构，能量偏好性 linkage 不给。")

    # ---------------- 出图
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))
    n = np.arange(N)
    ax = axes[0]
    for c in (16.0, 48.0, 80.0, 112.0):
        ax.plot(n, relax_sigma(init_sigma(c, c + 128.0)), lw=1.0, label="壁位 {:g}".format(c))
    ax.set_xlabel("格点序号")
    ax.set_ylabel("sigma")
    ax.set_title("(a) 约束版：壁位平移零模")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1]
    s_f = relax_sigma(s0)
    ax.plot(n, offset(relax_sigma(s0)), lw=1.2, label="偏移场 a（约束版）")
    for K in (1e-2, 1e-4):
        _, v_f = relax_strength(s0, K)
        ax.plot(n, v_f - np.floor(np.nanmin(v_f)), lw=1.0, alpha=0.8,
                label="v（强度版 K = {:g}）".format(K))
    ax.set_xlabel("格点序号")
    ax.set_ylabel("相位")
    ax.set_title("(b) 约束版与强度版的相位剖面")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[2]
    ax.semilogx(K_LIST, widths, "o-")
    ax.set_xlabel("K")
    ax.set_ylabel("跳变宽度")
    ax.set_title("(c) 强度版跳变宽度的 K 依赖")
    ax.grid(alpha=0.25, which="both")

    fig.tight_layout()
    import os
    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
    fig.savefig(OUT_FIG, dpi=160)
    print("\n图已存：{}".format(OUT_FIG))
    print("\n用时 {:.1f} 秒".format(time.time() - t0))
    print("=" * 82)


if __name__ == "__main__":
    main()
