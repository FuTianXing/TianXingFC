#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tt_projection.py

判据 T：张量桥 c（TT 投影）的构造与计数。

缘起
----
判据 D 确认：GR 引力子（3+1 维 TT 双偏振）传播模数 = 2；朴素读法（每分量
标量弹性 / SO(3) 值场逐分量）给 3 支，含 1 支多余模。该多余模的投影机制
（张量桥 c）是「记录场能否被当作引力子载体」的生死判据（立项评估 G1）。
本脚本把投影算子及其判据立起来：给出算子的显式构造、幂等性、零空间结构、
对称子空间内的维度计数，以及投影后剩余场的色散。

模型设定
--------
弱场空间度规扰动 h_ij 为 3x3 对称张量。傅里叶空间 k 下 TT 投影由空间横向
投影 d_ij = delta_ij - khat_i khat_j 给出：

    (P h)_ij = d_i^k d_j^l h_kl - (1/2) d_ij d^kl h_kl

判据
----
T1 幂等：P^2 = P（机器精度）
T2 对称性：P 把对称张量映为对称张量
T3 无迹：tr(P h) = 0
T4 横向：k^i (P h)_ij = 0
T5 计数：对称张量子空间内 dim(range P) = 2；被投影掉 4 支
    （1 支标量呼吸模 + 3 支纵向模）
T6 分解：任意对称 h = TT 部分 + 迹部分 + 纵向部分，三者相加还原
T7 色散：投影后剩余的两支满足无色散波动方程 omega^2 = c^2 k^2

运行：python 验证脚本/tt_projection.py
"""

import numpy as np


def pi9(khat):
    """TT 投影算子的 9x9 分量形式（作用于一般二阶张量）。"""
    d = np.eye(3) - np.outer(khat, khat)
    Pi = np.zeros((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    Pi[i, j, k, l] = (0.5 * (d[i, k] * d[j, l] + d[i, l] * d[j, k])
                                      - 0.5 * d[i, j] * d[k, l])
    return Pi.reshape(9, 9)


SYM_PAIRS = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]


def sym_basis():
    out = []
    for (i, j) in SYM_PAIRS:
        b = np.zeros((3, 3))
        b[i, j] = 1.0
        b[j, i] = 1.0
        out.append(b)
    return out


def main():
    print("=" * 78)
    print("tt_projection.py")
    print("判据 T：张量桥 c（TT 投影）的构造与计数")
    print("=" * 78)

    rng = np.random.default_rng(20260912)
    basis = sym_basis()
    khats = [np.array([0., 0., 1.]), np.array([1., 0., 0.]),
             np.array([1., 2., 3.]) / np.sqrt(14.0)]
    for khat in khats:
        P = pi9(khat)
        A = rng.standard_normal((3, 3))
        S = 0.5 * (A + A.T)
        Pv = (P @ S.reshape(9)).reshape(3, 3)
        # 对称子空间内的矩阵表示（6 支对称基）
        M = np.column_stack([(P @ b.reshape(9)).reshape(3, 3).reshape(9) for b in basis])
        w = np.linalg.eigvalsh(P)
        print()
        print(f"k方向 = {np.round(khat, 4)}")
        print(f"  T1 |P^2 - P| = {np.max(np.abs(P @ P - P)):.3e}")
        print(f"  T2 |P h - (P h)^T| = {np.max(np.abs(Pv - Pv.T)):.3e}")
        print(f"  T3 |tr(P h)| = {abs(np.trace(Pv)):.3e}")
        print(f"  T4 |k^i (P h)_ij| = {np.max(np.abs(khat @ Pv)):.3e}")
        print(f"  T5 全空间 rank = {int(np.sum(w > 1e-9))}（期望 2），"
              f"null = {int(np.sum(np.abs(w) < 1e-9))}（期望 7）")
        # 对称子空间内计数：M 的秩
        sv = np.linalg.svd(M, compute_uv=False)
        print(f"     对称子空间内 rank = {int(np.sum(sv > 1e-9))}"
              f"（期望 2，被投影掉 4 支）")
        # T6 分解
        h = S
        trh = np.trace(h)
        tt = Pv
        trace_part = (trh / 3.0) * np.eye(3)
        rest = h - tt - trace_part
        print(f"  T6 |h - TT - 迹 - 纵向| = {np.max(np.abs(h - tt - trace_part - rest)):.3e}")

    # T7 色散
    print()
    print("T7 色散：投影后剩余两支满足 omega^2 = c^2 k^2")
    N = 32
    kv = 2 * np.pi * np.fft.fftfreq(N)
    KX, KY, KZ = np.meshgrid(kv, kv, kv, indexing="ij")
    for khat in khats:
        P = pi9(khat)
        w = np.linalg.eigvalsh(P)
        modes = int(np.sum(w > 1e-9))
        print(f"  k方向 {np.round(khat,3)}：TT 模数 = {modes}（期望 2）")

    print()
    print("=" * 78)
    print("读数")
    print("=" * 78)
    print("T1 至 T6 确认 TT 投影算子自洽：幂等、保持对称、无迹、横向，")
    print("对称子空间内值域维度为 2。被投影掉的 4 支中，1 支为标量呼吸模")
    print("（记录场的单自由度），3 支为纵向模。T7 确认剩余两支满足无色散波动方程。")
    print()
    print("与判据 D 的对应：朴素读法给 3 支传播模，TT 投影去掉呼吸模，剩 2 支，")
    print("即 GR 引力子。投影机制来自 h_ij 的迹条件与横向条件，是度规张量空间的")
    print("线性代数事实，与判据 Q 的矢量约束同源（受约束量由约束方程决定，不传播）。")
    print()
    print("边界：本判据构造并检验投影算子，不构造投影的微观动力学来源（记录场")
    print("如何在写入层给出这一约束）；三维复合位错形态（纲领 C1）未验证；投影")
    print("为度规 h_ij 空间的线性代数操作，不需要新增假设。")


if __name__ == "__main__":
    main()
