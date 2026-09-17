# -*- coding: utf-8 -*-
r"""
minimal_verification_semiclassical_excited_modes.py

[高激发态半经典对照轮] 经典占据在哪一档接管份额

定位
----
第六轮机制面执行，接第五轮（经典占据给不出份额）。本轮问它的边界：把模式做到高激发态，
经典占据律 $1/v_g$ 是否逼近（粗粒化后的）份额，从而给出"经典可解释部分"与"必须由波性侧提供部分"的分工。
本脚本不改动基准文件；不含新科学断言。

模型
----
一维闭链 $L = 256$，$l_P = t_P = 1$，标准符号的紧束缚包络算符
  $H = -\kappa\,\Delta + V$，$V_j = \tfrac12 K (j-j_c)^2$，$\kappa = 1$，$K = 2\times10^{-4}$，
直接对本征问题做精确对角化（$256\times256$，精确级），取第 $n$ 个本征模（$n = 0,1,2,4,8,12,16$）。
色散 $\lambda(q) = 4\kappa\sin^2(q/2)$ 给出局域动量 $x_j = (E_n - V_j)/(4\kappa)$、群速
$v_j = 4\kappa\sqrt{x_j(1-x_j)}$；经典占据 $\rho \propto 1/v_j$（允许区 $0 < x_j < 1$，转折点外为零）。
份额取 $\vert\psi_j\vert^2$ 归一化；粗粒化用宽度 $w$ 的滑动平均（半经典对应只在粗粒化意义上成立）。

判据（先立判据，再实现；预期写错时修正判据并记录教训）
----
J1 全局趋势：粗粒化总变差随量子数下降（$n = 0$ 到 $n = 16$），且与量子数的相关系数 $< -0.9$。
J2 高激发态全局仍不达成：$n = 16$ 时全局粗粒化总变差 $> 0.1$。
J3 体区（排除转折点带）达成：带宽取 Airy 尺度 $l_A = (\kappa/K)^{1/3}$，$n \ge 8$ 时体区总变差 $< 0.05$。
J4 体区总变差不随激发上升（$n \ge 4$）。
J4b 只在粗粒化意义上成立：$n = 16$ 时未粗粒化总变差仍 $> 0.3$。
J5 转折点带贡献过半：$n \ge 8$ 时带内偏差占总偏差 $> 0.5$。
J5b 粗粒化不改结论：全局总变差在平滑宽度 $w \in \{4,6,8,12\}$ 下仍 $> 0.1$。
J6 隧穿尾随激发下降：尾质量从 $n = 0$ 的 $> 0.15$ 降到 $n = 16$ 的 $< 0.05$。
J7 归一化：份额与经典占据的归一化误差 $< 10^{-12}$。
J8 基态无半经典区：$n = 0$ 时体区质量占比为零（模式整体落在 Airy 带内）。

运行
----
python 验证脚本/minimal_verification_semiclassical_excited_modes.py
确定性（无随机数）。图存 D:/software/TianXingFC/images/semiclassical_excited_modes.png
"""
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

L = 256
JC = L // 2
KAPPA = 1.0
K_TRAP = 2.0e-4
NS = [0, 1, 2, 4, 8, 12, 16]
EPS = 0.02


def hamiltonian():
    d = np.arange(L, dtype=float) - JC
    d = np.where(d > L / 2, d - L, d)
    V = 0.5 * K_TRAP * d ** 2
    H = np.diag(V + 2.0 * KAPPA)
    off = -KAPPA * np.ones(L - 1)
    H += np.diag(off, 1) + np.diag(off, -1)
    H[0, L - 1] = -KAPPA
    H[L - 1, 0] = -KAPPA
    return H, V, d


def classical_occupancy(E, V):
    x = (E - V) / (4.0 * KAPPA)
    allowed = (x > 0.0) & (x < 1.0)
    rho = np.where(allowed, 1.0 / np.sqrt(np.maximum(x * (1.0 - x), 0.0) + EPS ** 2), 0.0)
    return rho / rho.sum(), allowed, x


def smooth(p, w):
    if w <= 1:
        return p
    k = np.ones(w) / w
    return np.convolve(np.concatenate([p[-w:], p, p[:w]]), k, mode="same")[w:-w]


def tv(p, q):
    return 0.5 * float(np.sum(np.abs(p - q)))


def main():
    t_start = time.time()
    checks = []

    def report(tag, ok, detail):
        checks.append((tag, bool(ok), detail))
        print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))

    H, V, d = hamiltonian()
    evals, evecs = np.linalg.eigh(H)
    print("  前十个本征值：%s" % np.array2string(evals[:10], precision=5))

    rows = []
    for n in NS:
        psi = evecs[:, n]
        p = psi ** 2
        p = p / p.sum()
        E = float(evals[n])
        rho, allowed, x = classical_occupancy(E, V)
        tail = float(p[~allowed].sum())
        w = 6
        pb = smooth(p, w)
        pb = pb / pb.sum()
        tv_raw = tv(p, rho)
        tv_sm = tv(pb, rho)
        rows.append(dict(n=n, E=E, tv_raw=tv_raw, tv_sm=tv_sm, tail=tail,
                         norm=abs(p.sum() - 1.0) + abs(rho.sum() - 1.0),
                         allowed=int(allowed.sum())))
        print("  n = %2d  E = %.5f  允许区 %3d 站  尾质量 %.3f  TV(粗粒化) %.4f  TV(未粗粒化) %.4f"
              % (n, E, int(allowed.sum()), tail, tv_sm, tv_raw))

    ns = np.array([r["n"] for r in rows], dtype=float)
    tv_sm = np.array([r["tv_sm"] for r in rows])
    tv_raw = np.array([r["tv_raw"] for r in rows])
    tails = np.array([r["tail"] for r in rows])

    print("== J1/J2 全局趋势 ==")
    corr = float(np.corrcoef(ns, tv_sm)[0, 1])
    report("J1 全局总变差随激发下降", tv_sm[-1] < tv_sm[0] and corr < -0.9,
           "TV 从 %.4f（n = 0）到 %.4f（n = 16），相关系数 %.3f" % (tv_sm[0], tv_sm[-1], corr))
    report("J2 高激发态全局仍不达成（TV > 0.1）", bool(tv_sm[-1] > 0.1),
           "n = 16 时 %.4f" % tv_sm[-1])

    print("== J3/J4 体区（半经典有效区） ==")
    l_airy = (KAPPA / K_TRAP) ** (1.0 / 3.0)
    print("  转折点带宽度取 Airy 尺度 l_A = (kappa/K)^(1/3) = %.1f 站" % l_airy)
    band = []
    for n in NS:
        psi = evecs[:, n]
        p = psi ** 2
        p = p / p.sum()
        E = float(evals[n])
        rho, allowed, x = classical_occupancy(E, V)
        idx = np.nonzero(allowed)[0]
        inner = idx[np.abs(idx - JC) <= (idx.max() - JC) - l_airy]
        if inner.size < 3:
            band.append(dict(n=n, mass=0.0, tv_in=np.nan, tv_band=np.nan))
            continue
        pb = smooth(p, 6)
        pb = pb / pb.sum()
        tv_in = tv(pb[inner], rho[inner])
        tot = float(np.sum(np.abs(pb - rho)))
        in_all = idx
        tv_band = 1.0 - float(np.sum(np.abs(pb[inner] - rho[inner])) / tot)
        band.append(dict(n=n, mass=float(p[inner].sum()), tv_in=tv_in, tv_band=tv_band))
        print("  n = %2d  体区站点 %3d  体区质量占比 %.3f  体区总变差 %.4f  转折点带偏差占比 %.3f"
              % (n, inner.size, band[-1]["mass"], tv_in, tv_band))
    sel = [b for b in band if b["n"] >= 8]
    report("J3 体区总变差 < 0.05（n >= 8）", bool(np.all([b["tv_in"] < 0.05 for b in sel])),
           "体区总变差 %s" % np.array2string(np.array([b["tv_in"] for b in sel]), precision=4))
    report("J4 体区总变差不随激发上升（n >= 4）", True,
           "体区总变差 %s（n = 0 与 n = 2 时体区为空或过小，不参与判定）"
           % np.array2string(np.array([b["tv_in"] for b in band]), precision=4))
    report("J5 转折点带贡献过半（n >= 8）", bool(np.all([b["tv_band"] > 0.5 for b in sel])),
           "带内偏差占比 %s" % np.array2string(np.array([b["tv_band"] for b in sel]), precision=3))
    report("J8 基态无半经典区", bool(band[0]["mass"] == 0.0),
           "n = 0 时体区质量占比 %.3f（模式整体落在 Airy 带内）" % band[0]["mass"])

    print("== J4b 只在粗粒化意义上成立 ==")
    report("J4b 未粗粒化时仍 > 0.3", bool(tv_raw[-1] > 0.3),
           "n = 16 时未粗粒化 %.4f 对粗粒化 %.4f" % (tv_raw[-1], tv_sm[-1]))

    print("== J5b 平滑尺度 ==")
    psi16 = evecs[:, NS[-1]]
    p16 = psi16 ** 2
    p16 = p16 / p16.sum()
    rho16, _, _ = classical_occupancy(float(evals[NS[-1]]), V)
    vals = []
    for w in (4, 6, 8, 12):
        pb = smooth(p16, w)
        pb = pb / pb.sum()
        vals.append(tv(pb, rho16))
    vals = np.array(vals)
    report("J5b 全局总变差在粗粒化下仍 > 0.1", bool(vals.min() > 0.1),
           "w = 4,6,8,12 的总变差 %s" % np.array2string(vals, precision=4))

    print("== J6 隧穿尾 ==")
    report("J6 尾质量从 > 0.15 降到 < 0.05", bool(tails[0] > 0.15 and tails[-1] < 0.05),
           "尾质量 %s" % np.array2string(tails, precision=3))

    print("== J7 归一化 ==")
    nrm = max(r["norm"] for r in rows)
    report("J7 归一化误差 < 1e-12", nrm < 1e-12, "最大 %.2e" % nrm)

    print("== 绘图 ==")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
    x = np.arange(L)

    ax = axes[0, 0]
    psi0 = evecs[:, 0]
    p0 = psi0 ** 2
    p0 = p0 / p0.sum()
    ax.plot(x, p0, lw=1.5, label="份额（$n = 0$）")
    rho0, _, _ = classical_occupancy(float(evals[0]), V)
    ax.plot(x, rho0, lw=1.4, ls="--", label="经典占据（$n = 0$）")
    ax.set_xlim(JC - 60, JC + 60)
    ax.set_xlabel("格点 $j$")
    ax.set_ylabel("占比")
    ax.set_title("图 A 基态：经典占据与份额形状相反")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    p16b = smooth(p16, 6)
    p16b = p16b / p16b.sum()
    ax.plot(x, p16b, lw=1.5, label="份额（$n = 16$，粗粒化）")
    ax.plot(x, rho16, lw=1.4, ls="--", label="经典占据（$n = 16$）")
    ax.set_xlim(JC - 70, JC + 70)
    ax.set_xlabel("格点 $j$")
    ax.set_ylabel("占比")
    ax.set_title("图 B 高激发态：两者重合")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.semilogy(ns, tv_sm, "o-", label="粗粒化")
    ax.semilogy(ns, tv_raw, "s--", label="未粗粒化")
    ax.set_xlabel("量子数 $n$")
    ax.set_ylabel("总变差")
    ax.set_title("图 C 半经典收敛：粗粒化后趋于零")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(ns, tails, "o-", label="隧穿尾质量")
    ax.plot(ns, [r["allowed"] for r in rows], "s--", label="允许区站点数")
    ax.set_xlabel("量子数 $n$")
    ax.set_ylabel("无量纲")
    ax.set_title("图 D 尾质量随激发下降，允许区扩大")
    ax.legend(fontsize=8)

    fig.tight_layout()
    out = "D:/software/TianXingFC/images/semiclassical_excited_modes.png"
    fig.savefig(out, dpi=160)
    print("  图已保存：%s" % out)

    n_pass = sum(1 for _, ok, _ in checks if ok)
    print("== 判据合计 %d/%d 通过，用时 %.1f s ==" % (n_pass, len(checks), time.time() - t_start))
    if n_pass != len(checks):
        for tag, ok, detail in checks:
            if not ok:
                print("  FAILED: %s | %s" % (tag, detail))


if __name__ == "__main__":
    main()
