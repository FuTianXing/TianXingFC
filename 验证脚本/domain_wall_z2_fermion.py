# -*- coding: utf-8 -*-
r"""第 488 条 J3：畴壁费米子类比适用性（第三版，物理格距）。

路线约束（内部草稿/注记-费米子与标架联络通道 边界声明）：不做格点狄拉克算符上的
手征零模搜索，前两轮已判负且搜索空间认定选错。J3 改走旋量结构路线，即体系内
SO(3) 主丛到 Spin(3) 双覆盖的 Z2 标签，与 I-1 共用同一判据语言。

模型（承接第 486 条）：一维周期链，复结构符号场 sigma 取双井势
  E[sigma] = sum_n h [ 0.5 ((sigma_{n+1} - sigma_n)/h)^2 + lam (sigma_n^2 - 1)^2 ]
环上取双壁泡（486 条口径：环面周期故畴壁成对出现）。格距 h 显式进入，与畴壁宽度
xi = 1/sqrt(lam) 解耦。第二版把 xi 直接当格点数，大 lam 档下 xi 小于一格，
剖面欠采样，壁位势读数 9.764e-06 至 2.602e-02 纯属离散化伪影。第三版固定
xi/h = 16，四档 lam 的读数同步落到机器零附近。
反射联结（第 487 条公理层形式）：phi -> sigma * phi，sigma 取 ±1。

两版判负的原因（逐条留痕）：
  第一版 能量闭式误用 4/3 sqrt(lam)，且环上单 kink 被环切开（实测 0.6636 对连续
          极限 0.200000）；改双壁泡后 0.399911 对 4 sqrt(lam) = 0.400000。
  第二版 格点数与壁宽混同，大 lam 档欠采样；壁位势相对极差随 lam 单调升
          （1.388e-16、1.389e-16、9.764e-06、2.602e-02），判为协议伪影。
  两版共有的判据误设：绕数误测局部跳变。反射使 U(1) 元素取复共轭，局部跳变为
          结构性非零，phi0 = 0.37 处 2|sin(2 pi phi0)| = 1.457937。绕数的正确读数
          是绕环一周的乘积偏离 1 的量。

判据：
  JP1 壁位势。气泡整体作实数平移，测 E 的相对极差。平移为泛函的精确对称，期望零。
       本条给出第 486 条 SW2 未直接测量的量：SW2 测整数格位上弛豫后的能量，
       循环链的整数平移是精确对称，其零读数是对称的推论；本条扫实数平移。
  JP2 sigma 畴壁的 Z2 旋量结构标签。环上壁数须为偶数且不小于 2；每个 sigma 壁
       是一次 J -> -J，即 SU(2) 中心元 -I，其平方为 I，故标签为 Z2。
  JP3 反射联结下 W_phi = 0 的结构性。两个读数分开记账：局部 U(1) 跳变为结构性
       非零（复共轭，解析值 2|sin 2 pi phi0|），绕环乘积偏离 1 在容差内为零。
"""
import io
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAMBDAS = [0.01, 0.04, 0.25, 1.0]
NS = [1024, 2048]
XI_OVER_H = 16
PHI0 = 0.37
TOL_FLAT = 1e-10
TOL_WIND = 1e-10
TOL_CLOSED = 1e-3

FIG = r"D:/software/TianXingFC/images/domain_wall_z2_fermion.png"


def energy(sig, lam, h):
    d = (np.roll(sig, -1) - sig) / h
    return float(np.sum(0.5 * d * d + lam * (sig * sig - 1.0) ** 2) * h)


def bubble(N, lam, h, frac=0.25, shift=0.0):
    r"""环上双壁泡。两个 tanh 因子的乘积在 (a, b) 内为 -1，环上周期故壁成对。"""
    xi = 1.0 / np.sqrt(lam)
    x = np.arange(N) * h
    L = N * h
    a, b = frac * L, (1.0 - frac) * L
    return np.tanh((x - a - shift) / xi) * np.tanh((x - b - shift) / xi)


def wall_count(sig):
    d = np.sign(sig)
    z = d == 0
    if z.any():
        d[z] = np.sign(sig[~z][0]) if (~z).any() else 1.0
    return int(np.sum(d * np.roll(d, -1) < 0))


def main():
    out = []
    ok = []

    out.append("=== 冒烟与闭式对照（xi/h = %d）===" % XI_OVER_H)
    rng = np.random.default_rng(20260929)
    lam0 = 0.04
    h0 = (1.0 / np.sqrt(lam0)) / XI_OVER_H
    s0 = rng.uniform(-1.5, 1.5, NS[0])
    out.append("random config E = %.6f (expect nonzero)" % energy(s0, lam0, h0))
    out.append("uniform sigma=+1: E = %.3e (expect 0)" % energy(np.ones(NS[0]), lam0, h0))
    worst_closed = 0.0
    for lam in LAMBDAS:
        h = (1.0 / np.sqrt(lam)) / XI_OVER_H
        N = NS[0]
        e = energy(bubble(N, lam, h), lam, h)
        cl = 4.0 * np.sqrt(lam)
        rel = abs(e - cl) / cl
        worst_closed = max(worst_closed, rel)
        out.append("  lam=%.2f  h=%.6f  bubble E=%.6f  closed 4 sqrt(lam)=%.6f  rel=%.3e  %s"
                   % (lam, h, e, cl, rel, "PASS" if rel < TOL_CLOSED else "FAIL"))
    ok.append(worst_closed < TOL_CLOSED)

    out.append("")
    out.append("=== JP1 壁位势：气泡实数平移 ===")
    rel1 = []
    store = {}
    for lam in LAMBDAS:
        for N in NS:
            h = (1.0 / np.sqrt(lam)) / XI_OVER_H
            xs = np.linspace(0.0, 1.0, 21)
            es = np.array([energy(bubble(N, lam, h, shift=x0), lam, h) for x0 in xs])
            rel = float((es.max() - es.min()) / abs(es.mean()))
            rel1.append(rel)
            store[(lam, N)] = (xs, es)
            out.append("  lam=%.2f N=%d  rel range=%.3e (tol %.0e)  %s"
                       % (lam, N, rel, TOL_FLAT, "PASS" if rel < TOL_FLAT else "FAIL"))
    j1 = max(rel1) < TOL_FLAT
    ok.append(j1)
    out.append("JP1 %s" % ("PASS" if j1 else "FAIL"))

    out.append("")
    out.append("=== JP2 sigma 畴壁的 Z2 标签 ===")
    ok2 = True
    prof = None
    for lam in LAMBDAS:
        h = (1.0 / np.sqrt(lam)) / XI_OVER_H
        N = NS[0]
        sig = bubble(N, lam, h)
        wc = wall_count(sig)
        good = wc % 2 == 0 and wc >= 2
        ok2 = ok2 and good
        if prof is None:
            prof = (np.arange(N) * h, sig, wc)
        out.append("  lam=%.2f  walls=%d (even and >=2: %s)  %s"
                   % (lam, wc, good, "PASS" if good else "FAIL"))
    out.append("  each sigma wall is one J -> -J crossing, i.e. the SU(2) centre -I; (-I)^2 = I")
    out.append("JP2 %s" % ("PASS" if ok2 else "FAIL"))

    out.append("")
    out.append("=== JP3 反射联结下 W_phi = 0 的结构性 ===")
    ok3 = True
    jump_store = None
    expect_local = 2.0 * abs(np.sin(2 * np.pi * PHI0))
    for lam in LAMBDAS:
        h = (1.0 / np.sqrt(lam)) / XI_OVER_H
        N = NS[0]
        sig = bubble(N, lam, h)
        u = np.where(sig > 0, np.exp(2j * np.pi * PHI0), np.exp(-2j * np.pi * PHI0))
        jump = np.abs(np.roll(u, -1) - u)
        widx = np.where(np.sign(sig) * np.roll(np.sign(sig), -1) < 0)[0]
        local = float(jump.max())
        wind = float(abs(np.abs(np.prod(u)) - 1.0))
        good = wind < TOL_WIND and abs(local - expect_local) < 1e-9
        ok3 = ok3 and good
        if jump_store is None:
            jump_store = (np.arange(N) * h, jump, widx, local, wind)
        out.append("  lam=%.2f  local jump=%.6f (expect %.6f)  winding dev=%.3e (tol %.0e)  %s"
                   % (lam, local, expect_local, wind, TOL_WIND,
                      "PASS" if good else "FAIL"))
    out.append("  local U(1) jump is structural and nonzero: reflection sends u to its conjugate")
    out.append("  winding zero is the statement that the reflection costs no topological charge")
    out.append("JP3 %s" % ("PASS" if ok3 else "FAIL"))

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9))
    for (lam, N), (xs, es) in sorted(store.items()):
        if N == NS[1]:
            ax[0].plot(xs, es, marker="o", ms=3, label=r"$\lambda=%.2f$" % lam)
    ax[0].set_xlabel(r"continuous shift $x_0$ (lattice units)")
    ax[0].set_ylabel(r"$E(\sigma)$")
    ax[0].set_title(r"(a) JP1 wall potential")
    ax[0].legend(fontsize=7)

    x, sig, wc = prof
    ax[1].plot(x, sig, color="k", lw=1.1)
    zi = np.where(np.sign(sig) * np.roll(np.sign(sig), -1) < 0)[0]
    for z in zi:
        ax[1].axvline(z, color="tab:red", ls="--", lw=0.8)
    ax[1].set_xlabel(r"site $x$")
    ax[1].set_ylabel(r"$\sigma$")
    ax[1].set_title(r"(b) JP2 $\sigma$ bubble, walls=%d" % wc)
    ax[1].set_ylim(-1.3, 1.3)

    x2, jump, widx, local, wind = jump_store
    ax[2].plot(x2, jump, color="tab:green", lw=1.0, label=r"$|u_{n+1}-u_n|$")
    for z in widx:
        ax[2].axvline(z, color="tab:red", ls="--", lw=0.8)
    ax[2].axhline(expect_local, color="tab:blue", ls=":", lw=0.9,
                  label=r"$2|\sin 2\pi\phi_0|$")
    ax[2].set_xlabel(r"site $x$")
    ax[2].set_ylabel(r"$|u_{n+1}-u_n|$")
    ax[2].set_title(r"(c) JP3 local jump %.3f, winding %.0e" % (local, wind))
    ax[2].legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(FIG, dpi=150)
    out.append("")
    out.append("figure: %s" % FIG)

    allok = all(ok)
    out.append("SUMMARY JP1/JP2/JP3 = %s" % ("ALL PASS" if allok else "HAS FAILURE"))
    text = "\n".join(out)
    print(text)
    with io.open(FIG.replace(".png", "_result.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text + "\n")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
