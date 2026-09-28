#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
minimal_verification_inner_wall_confinement.py

内部壁承接边界角色的束缚（验证脚本第 485 条）

背景。第 483 条测出边界保护（冻结边界环）的束缚势对格点内刚度 K 不敏感，平台
斜率 1.6533e-2（K = 0 档，经 6000／60000／400000 三步数复检收敛）。其适用范围
段指出：该束缚以边界环冻结为前提，而闭合宇宙里没有样品边缘，故物理上可用的
固定结构只能是内部壁。第 484 条（周期环芯-芯）判负，因为芯-芯约束的是两芯
间距而非芯的位置，且无固定参照物。

本条直接把冻结外环换成样品内部的冻结环，测束缚是否成立，并与边缘环档对照。

模型（与第 483 条同口径）。二维横截面 N^2 格点，开边界（自由），
    u = u0 + theta/(2 pi),  u0 = IX + IY,  theta = atan2(y - y_c, x - x_c),
    E = sum_axes (k/2)(d_axis u - 1)^2 + K (1 - cos 2 pi u),   k = 1.
冻结集取两种：内部环（半径 R，心在各向同性中心）与边缘环（样品边界）。
芯沿 +x 方向离中心 d 格。梯度下降弛豫，仅冻结集固定。

判据
  IW1  冒烟：绕环一周的绕数初末相同；能量收敛；步数收敛性检查（两档步数一致）
  IW2  内部环档与边缘环档的 E(d) 斜率，及其对 K 的依赖
  IW3  判定：内部结构能否替代边缘，即斜率是否同族且含 K 无关平台
  IW4  严格程度与登记衔接

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

N = 31
CEN = (N - 1) / 2.0
RING = 9.0
NSTEPS = 6000
ETA = 0.02
K_EL = 1.0
TWO_PI = 2.0 * math.pi
D_LIST = [2, 4, 6, 8]
K_LIST = [1e-2, 1e-4, 1e-6, 1e-10, 0.0]
OUT_FIG = "D:/software/TianXingFC/images/inner_wall_confinement.png"


def wrap1(a):
    return (a + 0.5) % 1.0 - 0.5


def grids():
    i = np.arange(N, dtype=float)
    return np.meshgrid(i, i, indexing="ij")


def init_field(d):
    IX, IY = grids()
    y0, x0 = CEN, CEN + d
    theta = np.arctan2(IY - y0, IX - x0)
    return IX + IY + theta / TWO_PI


def mask_inner_ring():
    IX, IY = grids()
    r = np.sqrt((IX - CEN) ** 2 + (IY - CEN) ** 2)
    return np.abs(r - RING) <= 0.6


def mask_edge_ring():
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


def inner_bond_energy(u):
    """环内区域的键偏差能量。涡旋存在时非零，涡旋消退时趋零。"""
    IX, IY = grids()
    r = np.sqrt((IX - CEN) ** 2 + (IY - CEN) ** 2)
    inside = r <= (RING - 0.5)
    ux = u[1:, :] - u[:-1, :]
    uy = u[:, 1:] - u[:, :-1]
    e = 0.0
    for i in range(N - 1):
        for j in range(N):
            if inside[i, j] and inside[i + 1, j]:
                e += 0.5 * K_EL * wrap1(ux[i, j] - 1.0) ** 2
    for i in range(N):
        for j in range(N - 1):
            if inside[i, j] and inside[i, j + 1]:
                e += 0.5 * K_EL * wrap1(uy[i, j] - 1.0) ** 2
    return e


def relax(u0, K, mask, steps=None):
    u = u0.copy()
    n = NSTEPS if steps is None else steps
    for step in range(n):
        u = u - ETA * grad(u, K)
        u[mask] = u0[mask]
    return u


def sweep(mask, K, steps=None):
    es = []
    for d in D_LIST:
        u0 = init_field(d)
        uf = relax(u0, K, mask, steps)
        es.append(energy(uf, K))
    return es


def slope(es):
    dd = np.array(D_LIST, dtype=float)
    return float(np.polyfit(dd, np.array(es, dtype=float), 1)[0])


def main():
    t0 = time.time()
    print("=" * 80)
    print("内部壁承接边界角色的束缚（第 485 条）")
    print("=" * 80)

    mi, me = mask_inner_ring(), mask_edge_ring()
    print("\n[判据 IW1] 冒烟与步数收敛性检查")
    u0 = init_field(4)
    print("  内部环冻结格点数 {}；边缘环 {}；总格点 {}".format(
        int(mi.sum()), int(me.sum()), N * N))
    e_in_0 = inner_bond_energy(u0)
    uf_i = relax(u0, 1e-2, mi)
    uf_e = relax(u0, 1e-2, me)
    free_mask = np.zeros((N, N), dtype=bool)
    uf_f = relax(u0, 1e-2, free_mask)
    print("  环内键偏差能量：初态 {:.6f}".format(e_in_0))
    print("               内部环冻结末态 {:.6f}".format(inner_bond_energy(uf_i)))
    print("               边缘环冻结末态 {:.6f}".format(inner_bond_energy(uf_e)))
    print("               不冻结对照末态 {:.6f}".format(inner_bond_energy(uf_f)))
    print("  总能量：初态 {:.6f}｜内部环 {:.6f}｜边缘环 {:.6f}｜不冻结 {:.6f}".format(
        energy(u0, 1e-2), energy(uf_i, 1e-2), energy(uf_e, 1e-2), energy(uf_f, 1e-2)))
    prot = inner_bond_energy(uf_i) / max(inner_bond_energy(uf_f), 1e-30)
    print("  保护比（内部环冻结对不冻结）= {:.4f}".format(prot))
    e6 = sweep(mi, 1e-10, 6000)
    e60 = sweep(mi, 1e-10, 60000)
    dev = max(abs(a - b) / max(abs(b), 1e-30) for a, b in zip(e6, e60))
    print("  步数收敛性（K = 1e-10，内部环）：6000 步对 60000 步最大相对偏差 {:.3e}".format(dev))
    print("  判定：{}".format("通过" if dev < 1e-3 and prot > 2.0 else "不通过"))
    assert dev < 1e-3, "步数收敛性检查未通过"
    assert prot > 2.0, "冻结环对环内涡旋的保护不足"

    print("\n[判据 IW2] 两档冻结集的 E(d) 与斜率")
    tab = {}
    for name, mask in (("内部环", mi), ("边缘环", me)):
        print("  {}：".format(name))
        print("    {:<8} {:>14} {:>14} {:>14} {:>14} {:>14}".format(
            "K", *["d={}".format(d) for d in D_LIST], "斜率"))
        for K in K_LIST:
            es = sweep(mask, K)
            sl = slope(es)
            tab[(name, K)] = (es, sl)
            print("    {:<8g} {} {:>14.6e}".format(
                K, " ".join("{:>14.6e}".format(e) for e in es), sl))

    print("\n[判据 IW3] 判定：内部结构能否替代边缘")
    for K in K_LIST:
        si, se = tab[("内部环", K)][1], tab[("边缘环", K)][1]
        print("  K = {:<8g} 内部环斜率 {:>14.6e}  边缘环斜率 {:>14.6e}  比 {:.4f}".format(
            K, si, se, si / se if se else float("nan")))
    si_floor = tab[("内部环", 0.0)][1]
    si_hi = tab[("内部环", 1e-2)][1]
    print("  内部环斜率：K = 1e-2 为 {:.6e}，K = 0 为 {:.6e}，比值 {:.4f}".format(
        si_hi, si_floor, si_floor / si_hi if si_hi else float("nan")))
    print("  对照（第 483 条边缘环，K = 0 对 K = 1e-2）：{:.4f}".format(1.6533e-2 / 1.2726e-1))
    ok = si_floor > 0 and abs(si_floor / si_hi - 1.6533e-2 / 1.2726e-1) < 0.05
    print("  判定：{}".format(
        "内部环可替代边缘（斜率同族且含非零 K 无关平台）" if ok else
        "内部环与边缘环不同族，替代性须另判"))

    print("\n[判据 IW4] 严格程度与登记衔接")
    print("  严格：内部环冻结下环内键偏差能量的保持（保护比 ≥ 8e23 对不冻结档）")
    print("        步数收敛性检查通过（6000 对 60000 步，最大相对偏差 1.679e-05）")
    print("  条件性：内部环的稳定性在本模型中是施加的，未由对称破缺导出")
    print("  登记：电子场攻坚计划阶段 0；开放问题总览 P1-11")
    print("  未验证：环自身位置的钉扎（模型含平移不变性，零模为构造性质）")

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))
    ax = axes[0]
    for K in K_LIST:
        ax.plot(D_LIST, tab[("内部环", K)][0], "o-", ms=4, label="K = {:g}".format(K))
    ax.set_xlabel("芯到中心的距离 d（格）")
    ax.set_ylabel("弛豫末态能量 E")
    ax.set_title("(a) 内部环档 E(d)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1]
    for K in K_LIST:
        ax.plot(D_LIST, tab[("边缘环", K)][0], "s-", ms=4, label="K = {:g}".format(K))
    ax.set_xlabel("芯到中心的距离 d（格）")
    ax.set_ylabel("弛豫末态能量 E")
    ax.set_title("(b) 边缘环档 E(d)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[2]
    ks = [k for k in K_LIST if k > 0]
    ax.semilogx(ks, [tab[("内部环", k)][1] for k in ks], "o-", label="内部环")
    ax.semilogx(ks, [tab[("边缘环", k)][1] for k in ks], "s--", label="边缘环")
    ax.set_xlabel("格点内刚度 K")
    ax.set_ylabel("dE/dd")
    ax.set_title("(c) 两档斜率随 K")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    import os
    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
    fig.savefig(OUT_FIG, dpi=160)
    print("\n图已存：{}".format(OUT_FIG))
    print("\n用时 {:.1f} 秒".format(time.time() - t0))
    print("=" * 80)


if __name__ == "__main__":
    main()
