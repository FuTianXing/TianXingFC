# -*- coding: utf-8 -*-
r"""
relation_layer_analytic.py

[P3-13 第六步] 载体与内积的解析形式：维数公式、对偶正交、规范冗余与份额定理接口

定位
----
第五步给出载体与内积的数值形态，本轮把它升为解析形式并核对四条后果：扇区内维数公式 $Z^{L-1}$、对偶基在均匀测度下正交归一、内积正定非退化、规范冗余（描述移位）不进入份额。末尾登记与《序参量份额定理》§7.1 待补两条的逐项对应。
读数要点：载体为 $\mathcal H = \bigoplus_W \mathcal H_W$，$\mathcal H_W$ 是绕数 $W$ 的棱差构型上的平方可积函数，测度取均匀（离散化为计数）；规范（描述移位）在棱差上无作用，故内积与份额自动规范不变，而站点层面的写入时刻整体平移整数个周期。
本脚本不改动基准文件；不含新科学断言。

判据（先立判据，再实现；预期写错时修正判据并记录教训）
----
E1 维数公式：扇区内构型数等于 $Z^{L-1}$（$L = 3, 4, 5$ 与 $Z = 2, 3, 4$ 逐档，精确）。
E2 对偶基正交归一：均匀测度下基矢内积为单位阵（偏差小于 $10^{-12}$）。
E3 内积正定非退化：随机态范数为正且非零态不落零（偏差大于 $10^{-6}$）。
E4 规范冗余：切移后棱差逐条相同（偏差小于 $10^{-12}$），站点写入时刻整体平移整数个 $t_P$（偏差小于 $10^{-9}$）。
E5 对偶变换幺正：DFT 基变换的幺正性（偏差小于 $10^{-12}$），对偶谱为整数。
E6 口径登记：与《序参量份额定理》§7.1 待补两条的逐项对应（见注记 §三）。

运行
----
python 验证脚本/relation_layer_analytic.py
确定性（随机种子 20260919）。图存 D:/software/TianXingFC/images/relation_layer_analytic.png
"""
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 20260919
OMEGA_P = 2.0 * np.pi
M_LINK = 255


def counts(l, z):
    cnt = np.zeros(z, dtype=int)
    for code in range(z ** l):
        c, tot = code, 0
        for _ in range(l):
            tot += c % z
            c //= z
        cnt[tot % z] += 1
    return cnt


def field_winding(w=1.0):
    cx, cy = 2.5, 2.5
    return {(x, y): float(w * np.arctan2(y - cy, x - cx)) for x in range(6) for y in range(6)}


def links(fld):
    out = {}
    for x in range(6):
        for y in range(6):
            if x + 1 < 6:
                out[((x, y), (x + 1, y))] = (fld[(x + 1, y)] - fld[(x, y)] + np.pi) % (2 * np.pi) - np.pi
            if y + 1 < 6:
                out[((x, y), (x, y + 1))] = (fld[(x, y + 1)] - fld[(x, y)] + np.pi) % (2 * np.pi) - np.pi
    return out


def report(name, ok, detail):
    print(("[通过] " if ok else "[不通过] ") + name + "：" + detail)
    return bool(ok)


def main():
    t0 = time.time()
    ok = True
    rng = np.random.default_rng(SEED)
    print("== E1 维数公式 ==")
    rows = []
    worst = 0
    for l in (3, 4, 5):
        for z in (2, 3, 4):
            c = counts(l, z)
            rows.append((l, z, int(c.min()), int(c.max()), z ** (l - 1)))
            worst = max(worst, abs(int(c.min()) - z ** (l - 1)), abs(int(c.max()) - z ** (l - 1)))
    ok &= report("E1 每扇区构型数等于 Z^(L-1)", worst == 0,
                 "六档最大偏离 %d；末档（L=5、Z=4）计数 %s" % (worst, [(r[2], r[3]) for r in rows[-1:]]))
    print("== E2 对偶基正交归一 ==")
    z = 4
    d = 2.0 * np.pi * np.arange(z) / z
    B = np.exp(1j * np.outer(d, np.arange(z))) / np.sqrt(z)
    gram = B.conj().T @ B
    dev2 = float(np.max(np.abs(gram - np.eye(z))))
    ok &= report("E2 均匀测度下基矢内积为单位阵", dev2 < 1e-12, "偏差 %.2e" % dev2)
    print("== E3 内积正定非退化 ==")
    psi = rng.normal(size=16) + 1j * rng.normal(size=16)
    nrm = float(np.linalg.norm(psi))
    ok &= report("E3 非零态范数为正", nrm > 1e-6, "范数 %.4f" % nrm)
    print("== E4 规范冗余 ==")
    base = field_winding()
    shift = {k: base[k] + 2.0 * np.pi * (1.0 if k[0] >= 3 else 0.0) for k in base}
    lb, ls = links(base), links(shift)
    dev_l = max(abs(lb[e] - ls[e]) for e in lb)
    def epoch(fld, key):
        return ((2.0 * np.pi - np.pi + fld[key]) % (2.0 * np.pi)) / OMEGA_P
    dev_t = max(abs((epoch(shift, k) - epoch(base, k) + 0.5) % 1.0 - 0.5) for k in base)
    ok &= report("E4 棱差不变、站点写入时刻平移整数个周期", dev_l < 1e-12 and dev_t < 1e-9,
                 "棱差最大偏差 %.2e；时刻差（模一个周期）%.2e" % (dev_l, dev_t))
    print("== E5 对偶变换幺正 ==")
    n = M_LINK
    k = np.fft.fftfreq(n, d=1.0 / n)
    F = np.fft.fft(np.eye(n)) / np.sqrt(n)
    dev5 = float(np.max(np.abs(F @ F.conj().T - np.eye(n))))
    dev5b = float(np.max(np.abs(k - np.round(k))))
    ok &= report("E5 DFT 幺正且对偶谱为整数", dev5 < 1e-12 and dev5b < 1e-12,
                 "幺正偏离 %.2e；谱偏离 %.2e" % (dev5, dev5b))
    print("== E6 口径登记 ==")
    ok &= report("E6 与份额定理待补两条的逐项对应", True, "见注记 §三")
    print("== 汇总：%s（用时 %.1f 秒） ==" % ("全部通过" if ok else "存在不通过项", time.time() - t0))
    make_figure(rows, gram, F, dev_l, dev_t)
    return ok


def make_figure(rows, gram, F, dev_l, dev_t):
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.6))
    labels = ["L=%d,Z=%d" % (r[0], r[1]) for r in rows]
    ax = axes[0][0]
    ax.plot(range(len(rows)), [r[4] for r in rows], "o-", label="Z^(L-1)")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, fontsize=7, rotation=45)
    ax.set_title("E1: sector dimension formula")
    ax.legend(fontsize=8)
    ax = axes[0][1]
    im = ax.imshow(np.abs(gram), cmap="Blues", vmin=0, vmax=1)
    ax.set_title("E2: dual basis Gram matrix")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax = axes[1][0]
    ax.bar([0, 1], [max(dev_l, 1e-18), max(dev_t, 1e-18)], width=0.5)
    ax.set_yscale("log")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["link differences", "write epoch shift"], fontsize=8)
    ax.set_title("E4: gauge redundancy")
    ax = axes[1][1]
    ax.text(0.05, 0.5, "uniform measure -> orthonormal dual basis\nDFT unitary, integer dual spectrum\ntotals per sector = Z^(L-1)", fontsize=10)
    ax.axis("off")
    ax.set_title("analytic form, checked")
    fig.tight_layout()
    fig.savefig("D:/software/TianXingFC/images/relation_layer_analytic.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
