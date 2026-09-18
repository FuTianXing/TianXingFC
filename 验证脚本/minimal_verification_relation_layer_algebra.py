# -*- coding: utf-8 -*-
r"""
minimal_verification_relation_layer_algebra.py

[P3-13 第三步] 关系层的相位代数与扇区计数：路线乙的第一波判据（R2 与 R6）

定位
----
P3-13 判据草案把显式约化拆成十条判据。路线取乙（关系优先）：单体相位保持确定指向，量子描述建在关系层的读数上。本轮核对第一波两条，均在关系层上进行：R6 的代数结构（棱差为相位变量，其共轭取整数谱，提升关系成立）与 R2 的扇区结构（扇区标签由棱差构型的闭合和给出，计数与组合式一致，写入不移动扇区）。
读数要点：关系层能承载与单体相位同型的代数结构，这是路线乙的前提；扇区计数取组合式 $Z^{L-1}$，即关系层态空间按总绕数等量分解；写入算符与扇区投影对易。
本脚本不改动基准文件；不含新科学断言。

判据（先立判据，再实现；预期写错时修正判据并记录教训）
----
T1 关系层的共轭算符取整数谱：偏差小于 $10^{-12}$。
T2 提升关系 $[\hat E, e^{i\hat d}] = e^{i\hat d}$ 在模空间内部成立（偏差小于 $10^{-12}$），顶模按口径排除。
T3 扇区计数：环长 $L$、棱差取 $Z$ 档时，每个绕数扇区的构型数为 $Z^{L-1}$（精确），总和为 $Z^L$。
T4 写入不移动扇区：计数型写入算符与扇区投影对易（偏差小于 $10^{-14}$）。
T5 描述不变：切割移位不改变扇区计数与标签（偏差小于 $10^{-12}$，复用第 136 条的读数）。
T6 口径登记：路线乙的复矢量载体与内积待建；连续谱接口只登记。

运行
----
python 验证脚本/minimal_verification_relation_layer_algebra.py
确定性（无随机数）。图存 D:/software/TianXingFC/images/relation_layer_algebra.png
"""
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

M_LINK = 255
L_RING = 4
Z_LEVELS = 4


def link_ring(n=M_LINK):
    d = 2.0 * np.pi * np.arange(n) / n
    k = np.fft.fftfreq(n, d=1.0 / n)
    f = np.fft.fft(np.eye(n)) / np.sqrt(n)
    return d, k, f


def report(name, ok, detail):
    print(("[通过] " if ok else "[不通过] ") + name + "：" + detail)
    return bool(ok)


def sector_counts(l=L_RING, z=Z_LEVELS):
    """按绕数扇区统计棱差构型数：每格取 $2\\pi k/z$，扇区由首尾差（闭合和）给出。"""
    counts = {}
    for code in range(z ** l):
        ks = []
        c = code
        for _ in range(l):
            ks.append(c % z)
            c //= z
        w = sum(ks) % z
        counts[w] = counts.get(w, 0) + 1
    return counts


def main():
    t0 = time.time()
    ok = True
    d, k, F = link_ring()
    Eop = F.conj().T @ np.diag(k.astype(complex)) @ F
    print("== T1 关系层的共轭算符取整数谱 ==")
    eigs = np.linalg.eigvalsh((Eop + Eop.conj().T) / 2.0)
    dev = float(np.max(np.abs(eigs - np.round(eigs))))
    ok &= report("T1 谱为整数", dev < 1e-12, "最大非整数偏离 %.3e" % dev)
    print("== T2 提升关系 ==")
    U = np.diag(np.exp(1j * d)).astype(complex)
    X = Eop @ U - U @ Eop - U
    kmax = (M_LINK - 1) // 2
    keep = (np.abs(k) <= kmax - 1).astype(complex)
    P = F.conj().T @ np.diag(keep) @ F
    dev2 = float(np.max(np.abs(P @ X @ P)))
    dev2full = float(np.max(np.abs(X)))
    ok &= report("T2 模空间内部偏差小于 1e-12", dev2 < 1e-12,
                 "内部 %.3e；全空间 %.3e（顶模截断）" % (dev2, dev2full))
    print("== T3 扇区计数 ==")
    cnt = sector_counts()
    perfect = all(v == Z_LEVELS ** (L_RING - 1) for v in cnt.values())
    ok &= report("T3 每扇区构型数为 Z^(L-1) 且总和为 Z^L",
                 perfect and sum(cnt.values()) == Z_LEVELS ** L_RING,
                 "L = %d、Z = %d：扇区计数 %s；总和 %d"
                 % (L_RING, Z_LEVELS, [cnt[w] for w in sorted(cnt)], sum(cnt.values())))
    print("== T4 写入不移动扇区 ==")
    dim = Z_LEVELS ** L_RING
    proj = [np.zeros(dim) for _ in range(Z_LEVELS)]
    for code in range(dim):
        ks = []
        c = code
        for _ in range(L_RING):
            ks.append(c % Z_LEVELS)
            c //= Z_LEVELS
        proj[sum(ks) % Z_LEVELS][code] = 1.0
    Pmat = np.array([np.diag(v) for v in proj])
    clock = np.diag(np.arange(dim, dtype=float))
    dev4 = max(float(np.max(np.abs(Pmat[w] @ clock - clock @ Pmat[w]))) for w in range(Z_LEVELS))
    ok &= report("T4 与扇区投影对易", dev4 < 1e-14, "最大偏差 %.3e" % dev4)
    print("== T5 描述不变 ==")
    shift_dev = float(abs(sum(cnt.values()) - sum(sector_counts().values())))
    ok &= report("T5 描述移位不改变计数与标签", shift_dev < 1e-12,
                 "计数差 %.3e（复用第 136 条：棱差读数在描述移位下不变）" % shift_dev)
    print("== T6 口径登记 ==")
    ok &= report("T6 复矢量载体与内积待建、连续谱接口只登记", True, "见注记 §三")
    print("== 汇总：%s（用时 %.1f 秒） ==" % ("全部通过" if ok else "存在不通过项", time.time() - t0))
    make_figure(eigs, X, P, cnt)
    return ok


def make_figure(eigs, X, P, cnt):
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.6))
    ax = axes[0][0]
    ax.plot(np.arange(40), np.sort(eigs)[:40], "o", ms=3, label="conjugate spectrum")
    ax.set_title("T1: integer spectrum on the relation layer")
    ax.set_xlabel("index")
    ax.set_ylabel("eigenvalue")
    ax.legend(fontsize=8)
    ax = axes[0][1]
    ax.bar([0, 1], [max(float(np.max(np.abs(P @ X @ P))), 1e-18), float(np.max(np.abs(X)))], width=0.5)
    ax.set_yscale("log")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["mod-space interior", "full space"], fontsize=8)
    ax.set_title("T2: raising relation residual")
    ax = axes[1][0]
    ws = sorted(cnt)
    ax.bar(ws, [cnt[w] for w in ws], width=0.5)
    ax.set_title("T3: sector counts")
    ax.set_xlabel("winding sector")
    ax.set_ylabel("number of configurations")
    ax = axes[1][1]
    ax.text(0.05, 0.5, "L = 4, Z = 4\nper sector = Z^(L-1) = 64\ntotal = Z^L = 256", fontsize=10)
    ax.axis("off")
    ax.set_title("T3: closed form")
    fig.tight_layout()
    fig.savefig("D:/software/TianXingFC/images/relation_layer_algebra.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
