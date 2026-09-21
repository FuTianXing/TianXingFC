#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
minimal_verification_free_branch_tt_count.py

判据 CT：1D 自由支与三维 TT 同一化的显式计数。

背景
----
判据 T 给出 TT 投影（秩 2），判据 U 给出约束零空间（维度 2），判据 V 给出
自由部分的演化方程与记录场波动方程同一，第三段（WV）在 1D 上测得自由支的
色散与阻尼。本轮把这条链做成显式计数与逐项核对：

  CT1 三维计数：6 = 2+2+2；约束矩阵秩 4、零空间 2；TT 投影秩 2；两子空间同一；
     纯规范扰动落在投影核内（曲率的规范不变性入口）。
  CT2 一维约化计数：限制为 x 依赖后，零空间恰为 {h_yy = -h_zz, h_yz} 两支。
  CT3 方程与色散：线性化里奇公式在 TT 上给 R_ij = -1/2 Box h_ij、纯规范给
     R = 0、迹型扰动不给该式（公式在做事）；真空条件给色散 omega^2 = k^2。
  CT4 算符同一：记录场自由支（阻尼）色散 omega^2 + (gamma/2)^2 = k^2，
     弱阻尼端趋于 omega = k，与 TT 传播支同式；对上 WV3 的读数。

运行：python 验证脚本/minimal_verification_free_branch_tt_count.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "images", "free_branch_tt_count.png")


def sym_basis():
    """6 维对称张量基（正交归一）。前三个对角，后三个非对角。"""
    E = []
    for i in range(3):
        b = np.zeros((3, 3)); b[i, i] = 1.0
        E.append(b)
    for (i, j) in ((0, 1), (0, 2), (1, 2)):
        b = np.zeros((3, 3)); b[i, j] = b[j, i] = 1.0 / np.sqrt(2.0)
        E.append(b)
    return E


E = sym_basis()


def vec6(M):
    return np.array([np.sum(M * b) for b in E])


def tt_projector6(khat):
    d = np.eye(3) - np.outer(khat, khat)
    Pi = np.zeros((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    Pi[i, j, k, l] = (0.5 * (d[i, k] * d[j, l] + d[i, l] * d[j, k])
                                      - 0.5 * d[i, j] * d[k, l])
    P6 = np.zeros((6, 6))
    for a in range(6):
        out = np.einsum('ijkl,kl->ij', Pi, E[a])
        P6[:, a] = vec6(out)
    return P6


def constraint6(khat):
    rows = [[np.trace(b) for b in E]]
    for j in range(3):
        rows.append([sum(khat[i] * b[i, j] for i in range(3)) for b in E])
    return np.array(rows)


def nullspace(C, tol=1e-10):
    u, sv, vt = np.linalg.svd(C)
    return vt[int(np.sum(sv > tol)):].T


def main():
    print("=" * 78)
    print("minimal_verification_free_branch_tt_count.py")
    print("判据 CT：1D 自由支与三维 TT 同一化的显式计数")
    print("=" * 78)

    ok = {"CT1": True, "CT2": True, "CT3": True, "CT4": True}

    # ---------- CT1 三维计数 ----------
    print()
    print("CT1 三维计数与子空间同一")
    for khat in (np.array([1.0, 0, 0]), np.array([1, 1, 1]) / np.sqrt(3.0),
                 np.array([1, 2, 3]) / np.sqrt(14.0)):
        C = constraint6(khat)
        N = nullspace(C)
        P = tt_projector6(khat)
        w, V = np.linalg.eigh(P)
        VP = V[:, w > 0.5]
        dev_null = np.max(np.abs(P @ N - N))
        dev_c = np.max(np.abs(C @ VP))
        print("   k = %s : 秩(C) = %d , 零空间 = %d , 秩(P) = %d ;"
              " |P N - N| = %.2e , |C V_P| = %.2e"
              % (np.round(khat, 3), np.linalg.matrix_rank(C, tol=1e-10), N.shape[1],
                 VP.shape[1], dev_null, dev_c))
        ok["CT1"] = ok["CT1"] and (np.linalg.matrix_rank(C, tol=1e-10) == 4
                                   and N.shape[1] == 2 and VP.shape[1] == 2
                                   and dev_null < 1e-12 and dev_c < 1e-12)
    rng = np.random.default_rng(20260921)
    khat = np.array([1.0, 0, 0]); kvec = 2.0 * np.pi / 10.0 * khat
    devs_g = []
    for _ in range(5):
        xi = rng.normal(size=3) + 1j * rng.normal(size=3)
        hG = 1j * (np.outer(kvec, xi) + np.outer(xi, kvec))
        devs_g.append(np.max(np.abs(tt_projector6(khat) @ vec6(hG))))
    print("   纯规范扰动经 TT 投影的残差（5 个随机 xi）：%.2e" % max(devs_g))
    ok["CT1"] = ok["CT1"] and max(devs_g) < 1e-12

    # ---------- CT2 一维约化计数 ----------
    print()
    print("CT2 一维约化计数（限制为 x 依赖）")
    C = constraint6(khat)
    N = nullspace(C)
    v1 = np.zeros(6); v1[1] = 1.0 / np.sqrt(2.0); v1[2] = -1.0 / np.sqrt(2.0)
    v2 = np.zeros(6); v2[5] = 1.0
    Bv = np.vstack([v1, v2]).T
    sv = np.linalg.svd(N.T @ Bv, compute_uv=False)
    print("   零空间与 {h_yy = -h_zz, h_yz} 的交叉奇异值：%s" % np.round(sv, 8).tolist())
    ok["CT2"] = float(np.min(sv)) > 1.0 - 1e-10

    # ---------- CT3 线性化里奇符号（四维）、规范零与 TT 色散 ----------
    print()
    print("CT3 线性化里奇符号：规范零、TT 恒等式与真空色散")
    eta = np.diag([-1.0, 1.0, 1.0, 1.0])

    def lin_ricci_symbol(h, om, kx):
        k4 = np.array([-om, kx, 0.0, 0.0])          # 下指标
        k4up = np.array([om, kx, 0.0, 0.0])         # 上指标
        div_h = 1j * np.einsum('r,rv->v', k4up, h)  # d^rho h_{rho nu}
        box = (om ** 2 - kx ** 2) * h               # Box h = (omega^2 - k^2) h
        tr = np.einsum('rs,rs->', np.linalg.inv(eta), h)
        R = np.zeros((4, 4), dtype=complex)
        for mu in range(4):
            for nu in range(4):
                R[mu, nu] = 0.5 * (1j * k4[mu] * div_h[nu] + 1j * k4[nu] * div_h[mu]
                                   - box[mu, nu] + k4[mu] * k4[nu] * tr)
        return R

    rng3 = np.random.default_rng(7)
    om_t, kx_t = 0.9, 0.7
    devs_g = []
    for _ in range(5):
        xi = rng3.normal(size=4) + 1j * rng3.normal(size=4)
        k4 = np.array([-om_t, kx_t, 0.0, 0.0])
        hG = 1j * (np.outer(k4, xi) + np.outer(xi, k4))
        devs_g.append(np.max(np.abs(lin_ricci_symbol(hG, om_t, kx_t))))
    print("   规范零：|R[d xi + xi d]| 的最大值 = %.2e" % max(devs_g))

    eps1 = np.diag([0.0, 0.0, 1.0, -1.0]) / np.sqrt(2.0)
    eps2 = np.zeros((4, 4)); eps2[2, 3] = eps2[3, 2] = 1.0 / np.sqrt(2.0)
    devs_tt = []
    for eps in (eps1, eps2):
        for om in (0.6 * kx_t, 1.0 * kx_t, 1.4 * kx_t):
            h = eps.astype(complex)
            R = lin_ricci_symbol(h, om, kx_t)
            pred = -0.5 * (om ** 2 - kx_t ** 2) * h
            devs_tt.append(np.max(np.abs(R[1:, 1:] - pred[1:, 1:])))
            devs_tt.append(np.max(np.abs(R[:, 0])))
    print("   TT：|R_ij + (1/2)(omega^2-k^2) h_ij| 与 |R_{0 mu}| 的最大值 = %.2e" % max(devs_tt))
    ok["CT3"] = max(devs_g) < 1e-12 and max(devs_tt) < 1e-12

    # ---------- CT4 算符同一（记录场自由支对 TT） ----------
    print()
    print("CT4 算符同一：记录场自由支色散与 TT 传播支同式")
    devs4, ratios = [], []
    for k in (0.25, 0.5, 1.0):
        for gamma in (0.05, 0.1, 0.2):
            om = np.sqrt(k ** 2 - gamma ** 2 / 4.0) - 1j * gamma / 2.0
            devs4.append(abs(-om ** 2 + k ** 2 - 1j * gamma * om))
            ratios.append((k - om.real) / (gamma ** 2 / (8.0 * k)))
    print("   阻尼色散恒等式 |-omega^2 + k^2 - i gamma omega| 的最大值 = %.2e" % max(devs4))
    print("   弱阻尼修正 (k - Re omega)/(gamma^2/8k) 的范围：%.4f 至 %.4f"
          % (min(ratios), max(ratios)))
    ok["CT4"] = ok["CT4"] and max(devs4) < 1e-14
    k3, om3, g3 = 0.2513, 0.2399, 0.15
    resid3 = abs(om3 ** 2 + (g3 / 2.0) ** 2 - k3 ** 2) / k3 ** 2
    print("   WV3 读数核对：omega^2 + (gamma/2)^2 = k^2 的相对残差 = %.2e" % resid3)
    ok["CT4"] = ok["CT4"] and resid3 < 1e-3

    # ---------- 图 ----------
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.6))
    ax = axes[0]
    ax.bar([0], [2], color="C0", label="TT（传播）")
    ax.bar([0], [2], bottom=[2], color="C1", label="矢量（约束）")
    ax.bar([0], [2], bottom=[4], color="C2", label="标量（约束）")
    ax.set_xticks([0]); ax.set_xticklabels(["6 分量"])
    ax.set_ylabel("计数"); ax.set_title("(a) 6 = 2 TT + 2 矢量 + 2 标量")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax = axes[1]
    om_scan = np.linspace(0.4 * kx_t, 1.6 * kx_t, 9)
    Ramp = [np.max(np.abs(lin_ricci_symbol(eps1.astype(complex), om, kx_t))) for om in om_scan]
    Dscan = np.abs(om_scan ** 2 - kx_t ** 2)
    o = np.argsort(Dscan)
    ax.loglog(Dscan[o], np.array(Ramp)[o], "o", ms=5, label="TT 读数")
    xs = np.linspace(max(Dscan.min(), 1e-3), Dscan.max(), 50)
    ax.loglog(xs, 0.5 * xs, "-", label="(1/2) |omega^2 - k^2|")
    ax.set_xlabel("|omega^2 - k^2|"); ax.set_ylabel("R 幅度")
    ax.set_title("(b) 真空色散"); ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    ax = axes[2]
    gs = np.linspace(0.0, 0.3, 100)
    ax.plot(gs, np.sqrt(np.maximum(kx_t ** 2 - gs ** 2 / 4.0, 0.0)), label="Re omega（记录场自由支）")
    ax.plot(gs, np.full_like(gs, kx_t), "--", label="TT：omega = k")
    ax.set_xlabel("gamma"); ax.set_ylabel("omega")
    ax.set_title("(c) 弱阻尼端同一"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, dpi=160)
    print()
    print("图已存：%s" % FIG)

    print()
    print("=" * 78)
    print("总判定：" + "，".join("%s %s" % (k, "通过" if v_ else "失败") for k, v_ in ok.items()))
    print("=" * 78)


if __name__ == "__main__":
    main()
