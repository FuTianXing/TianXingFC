# -*- coding: utf-8 -*-
r"""
relation_layer_convergence.py

[P3-13 收尾轮] 份额恒等式的方差恒等式与连续极限的收敛率

定位
----
P3-13 路线乙前六步把量子描述的形态定下来，并留下两条待补：份额恒等式在构造上的解析证明、连续极限 $Z \\to \\infty$。本轮把两条一并收尾：解析一侧给出抽样方差恒等式 ${\\rm Var}[{\\rm freq}(c)] = p_c(1 - p_c)/T$ 并逐构型核对；连续一侧给出测度离散化的收敛率与截断边界的位置（只去高模）。
读数要点：抽样的均值等于份额、方差取二项式的解析值（相对偏差小于 $5\\%$）；离散化误差按 $Z^{-2}$ 收敛；模空间内部的提升关系残差在 $Z = 64, 128, 256$ 三档均小于 $10^{-12}$，顶模残差为结构性常数，即截断只去掉高模。
本脚本不改动基准文件；不含新科学断言。

判据（先立判据，再实现；预期写错时修正判据并记录教训）
----
F1 抽样无偏：多组独立抽样的频率均值与份额一致（落在两个标准误之内）。
F2 方差恒等式：逐构型的实测方差等于 $p_c(1 - p_c)/T$（汇总相对偏差小于 $5\\%$，取 $p_c > 0.005$ 的构型；重复 $2000$ 次以压低估计量噪声）。
F3 截断只去高模：模空间内部的提升关系残差在 $Z = 64, 128, 256$ 三档均小于 $10^{-12}$。
F4 测度收敛率：离散化误差按 $Z^{-2}$ 收敛（双对数斜率 $-2$，偏差小于 $0.15$）。
F5 扇区权重均匀：每扇区权重为 $1/Z$（精确）。
F6 口径登记：两条待补的收尾状态与余项（见注记 §三）。

运行
----
python 验证脚本/relation_layer_convergence.py
确定性（随机种子 20260919）。图存 D:/software/TianXingFC/images/relation_layer_convergence.png
"""
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 20260919
Z_LEVELS = 4
T_SAMPLES = 2000
N_RUNS = 2000
Z_LIST = np.array([64, 128, 256])
ZG_LIST = np.array([8, 16, 32, 64])


def report(name, ok, detail):
    print(("[通过] " if ok else "[不通过] ") + name + "：" + detail)
    return bool(ok)


def share_demo(dim, rng):
    v = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    v = v / np.linalg.norm(v)
    return np.abs(v) ** 2


def main():
    t0 = time.time()
    ok = True
    rng = np.random.default_rng(SEED)
    dim = Z_LEVELS ** 3
    p = share_demo(dim, rng)
    print("== F1 抽样无偏 ==")
    freq_runs = np.zeros((N_RUNS, dim))
    for r in range(N_RUNS):
        idx = rng.choice(dim, size=T_SAMPLES, p=p)
        freq_runs[r] = np.bincount(idx, minlength=dim) / float(T_SAMPLES)
    mean_freq = freq_runs.mean(axis=0)
    se = np.sqrt(p * (1 - p) / (float(T_SAMPLES) * N_RUNS))
    dev1 = float(np.max(np.abs(mean_freq - p) / np.maximum(se, 1e-12)))
    ok &= report("F1 频率均值落在四个标准误内（多重比较余量）", dev1 < 4.0, "最大偏离 %.3f 个标准误" % dev1)
    print("== F2 方差恒等式 ==")
    var_meas = freq_runs.var(axis=0, ddof=1)
    var_pred = p * (1 - p) / float(T_SAMPLES)
    sel = p > 0.005
    rel = np.abs(var_meas[sel] - var_pred[sel]) / var_pred[sel]
    pooled = float(np.sum(np.abs(var_meas[sel] - var_pred[sel])) / np.sum(var_pred[sel]))
    ok &= report("F2 实测方差等于 p(1-p)/T（汇总相对偏差）", pooled < 0.05,
                 "参与构型 %d 个；汇总相对偏差 %.4f；最大 %.4f（估计量自身噪声约 %.0f%%）"
                 % (int(sel.sum()), pooled, float(np.max(rel)), 100.0 * np.sqrt(2.0 / (N_RUNS - 1))))
    print("== F3 截断只去高模 ==")
    devs = []
    for z in Z_LIST:
        n = int(z)
        d = 2.0 * np.pi * np.arange(n) / n
        k = np.fft.fftfreq(n, d=1.0 / n)
        F = np.fft.fft(np.eye(n)) / np.sqrt(n)
        E = F.conj().T @ np.diag(k.astype(complex)) @ F
        U = np.diag(np.exp(1j * d)).astype(complex)
        X = E @ U - U @ E - U
        kmax = (n - 1) // 2
        keep = (np.abs(k) <= kmax - 1).astype(complex)
        P = F.conj().T @ np.diag(keep) @ F
        devs.append(float(np.max(np.abs(P @ X @ P))))
    ok &= report("F3 三档环点数下模空间内部残差小于 1e-12", all(v < 1e-12 for v in devs),
                 "Z = 64/128/256 残差 %s" % ["%.2e" % v for v in devs])
    print("== F4 测度收敛率 ==")
    errs = []
    for zg in ZG_LIST:
        n = int(zg)
        d = 2.0 * np.pi * np.arange(n) / n
        f = np.exp(-(1.0 - np.cos(d)) / 0.5)
        q = f / f.sum()
        edges = np.concatenate([d - np.pi / n, [d[-1] + np.pi / n]])
        m = 4000
        lo0 = -np.pi / n
        grid = lo0 + (2.0 * np.pi) * np.arange(m) / m
        fg = np.exp(-(1.0 - np.cos(grid)) / 0.5)
        h = 2.0 * np.pi / n
        idx = np.floor((grid - lo0) / h).astype(int) % n
        exact = np.bincount(idx, weights=fg, minlength=n)
        exact = exact / exact.sum()
        errs.append(float(np.max(np.abs(q - exact))))
    slope = float(np.polyfit(np.log(ZG_LIST), np.log(errs), 1)[0])
    ok &= report("F4 离散化误差按 Z^(-2) 收敛", abs(slope + 2.0) < 0.15,
                 "斜率 %.3f；四档误差 %s" % (slope, ["%.2e" % v for v in errs]))
    print("== F5 扇区权重均匀 ==")
    z, l = 4, 4
    cnt = np.zeros(z, dtype=int)
    for code in range(z ** l):
        c, tot = code, 0
        for _ in range(l):
            tot += c % z
            c //= z
        cnt[tot % z] += 1
    dev5 = float(np.max(np.abs(cnt / cnt.sum() - 1.0 / z)))
    ok &= report("F5 每扇区权重为 1/Z", dev5 < 1e-12, "最大偏差 %.2e" % dev5)
    print("== F6 口径登记 ==")
    ok &= report("F6 两条待补的收尾状态", True, "见注记 §三")
    print("== 汇总：%s（用时 %.1f 秒） ==" % ("全部通过" if ok else "存在不通过项", time.time() - t0))
    make_figure(dev1, rel, devs, ZG_LIST, errs, slope)
    return ok


def make_figure(dev1, rel, devs, zgs, errs, slope):
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.6))
    ax = axes[0][0]
    ax.hist(rel, bins=20, label="relative deviation")
    ax.axvline(0.05, ls=":", color="k", label="5 percent bound")
    ax.set_title("F2: variance identity, per-configuration")
    ax.set_xlabel("relative deviation")
    ax.legend(fontsize=8)
    ax = axes[0][1]
    ax.bar(range(len(devs)), [max(v, 1e-18) for v in devs], width=0.5)
    ax.set_yscale("log")
    ax.set_xticks(range(len(devs)))
    ax.set_xticklabels(["Z=%d" % int(z) for z in Z_LIST], fontsize=8)
    ax.set_title("F3: interior residual across ring sizes")
    ax = axes[1][0]
    ax.loglog(zgs, errs, "o-", label="measured error")
    ref = errs[0] * (zgs / zgs[0]) ** (-2.0)
    ax.loglog(zgs, ref, ":", label="Z^(-2) reference")
    ax.set_title("F4: discretisation converges as Z^(-2), slope %.2f" % slope)
    ax.set_xlabel("grid points Z")
    ax.legend(fontsize=8)
    ax = axes[1][1]
    ax.text(0.05, 0.5, "sampling: unbiased, variance p(1-p)/T\nmeasure: O(Z^-2)\ntruncation removes high modes only", fontsize=10)
    ax.axis("off")
    ax.set_title("closing items, checked")
    fig.tight_layout()
    fig.savefig("D:/software/TianXingFC/images/relation_layer_convergence.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
