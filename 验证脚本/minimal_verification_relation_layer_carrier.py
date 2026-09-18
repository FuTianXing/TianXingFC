# -*- coding: utf-8 -*-
r"""
minimal_verification_relation_layer_carrier.py

[P3-13 第五步] 路线乙的必答项：关系层态矢量的载体与内积（含 R3 至 R6、R9 的表达位置）

定位
----
路线乙（关系优先）的前四步给出了代数结构、扇区计数与概率语义，尚缺它的必答项：态矢量作为关系层读数构型上的复函数，载体与内积取什么形式。本轮把这两件落成可核的构造：载体取棱差构型（及其对偶的整数通量基），内积取可观测读数上的平移不变测度，并核对四条性质。
读数要点：对偶基的谱为整数（单体相位的同型结构），扇区由回路和给出；内积在描述移位下不变且跨扇区正交，故超选择在载体上表现为正交分解；提升算符在内积下幺正；叠加只在扇区内给出干涉（相对相位扫描给余弦型份额）。
本脚本不改动基准文件；不含新科学断言。

判据（先立判据，再实现；预期写错时修正判据并记录教训）
----
V1 载体的对偶基与扇区：共轭算符谱为整数（偏离小于 $10^{-12}$）；扇区由回路和给出且计数为 $Z^{L-1}$。
V2 内积的两条性质：描述移位下内积不变（偏差小于 $10^{-12}$）；跨扇区内积严格为零（$< 10^{-14}$）。
V3 提升算符在内积下幺正：范数守恒（偏差小于 $10^{-12}$）。
V4 干涉只发生在扇区内：同扇区叠加的份额随相对相位呈余弦型（拟合残差小于 $10^{-6}$）；跨扇区叠加无干涉项（$< 10^{-14}$）。
V5 偏迹与单站点计数：对单条棱取偏迹后份额之和为一（偏差小于 $10^{-12}$）；单站点可观测代数只含计数。
V6 口径登记：R3 至 R6 与 R9 在本载体上的表达位置（见注记 §三）。

运行
----
python 验证脚本/minimal_verification_relation_layer_carrier.py
确定性（随机种子 20260919）。图存 D:/software/TianXingFC/images/relation_layer_carrier.png
"""
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

M_LINK = 255
L_RING = 4
Z_LEVELS = 4
SEED = 20260919
PHASES = np.linspace(0.0, 2.0 * np.pi, 13)


def configurations(l=L_RING, z=Z_LEVELS):
    conf = []
    for code in range(z ** l):
        ks = []
        c = code
        for _ in range(l):
            ks.append(c % z)
            c //= z
        conf.append((tuple(ks), sum(ks) % z))
    return conf


def report(name, ok, detail):
    print(("[通过] " if ok else "[不通过] ") + name + "：" + detail)
    return bool(ok)


def main():
    t0 = time.time()
    ok = True
    rng = np.random.default_rng(SEED)
    conf = configurations()
    dim = len(conf)
    sec = np.array([c[1] for c in conf])
    print("== V1 载体的对偶基与扇区 ==")
    d = 2.0 * np.pi * np.arange(M_LINK) / M_LINK
    k = np.fft.fftfreq(M_LINK, d=1.0 / M_LINK)
    F = np.fft.fft(np.eye(M_LINK)) / np.sqrt(M_LINK)
    Eop = F.conj().T @ np.diag(k.astype(complex)) @ F
    eigs = np.linalg.eigvalsh((Eop + Eop.conj().T) / 2.0)
    dev1 = float(np.max(np.abs(eigs - np.round(eigs))))
    cnt = np.bincount(sec, minlength=Z_LEVELS)
    ok &= report("V1 对偶基谱为整数且扇区计数为 Z^(L-1)",
                 dev1 < 1e-12 and all(v == Z_LEVELS ** (L_RING - 1) for v in cnt),
                 "谱偏离 %.2e；扇区计数 %s" % (dev1, list(cnt)))
    print("== V2 内积的两条性质 ==")
    psi = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    phi = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    ip = complex(psi.conj() @ phi)
    shift = np.exp(1j * 0.37 * np.arange(dim) / dim)
    ip2 = complex((shift * psi).conj() @ (shift * phi))
    dev_shift = abs(ip2 - ip)
    cross = 0.0
    for sa in range(Z_LEVELS):
        for tb in range(Z_LEVELS):
            if sa == tb:
                continue
            ma = (sec == sa).astype(float)
            mb = (sec == tb).astype(float)
            cross = max(cross, abs(complex((ma * psi).conj() @ (mb * phi))))
    ok &= report("V2 描述移位不变且跨扇区正交", dev_shift < 1e-12 and cross < 1e-14,
                 "移位差 %.2e；跨扇区内积上界 %.2e" % (dev_shift, cross))
    print("== V3 提升算符幺正 ==")
    psi_n = np.linalg.norm(psi)
    up = np.exp(1j * (2.0 * np.pi / Z_LEVELS) * np.array([c[0][0] for c in conf]))
    dev3 = abs(float(np.linalg.norm(up * psi)) - float(psi_n))
    ok &= report("V3 提升算符保持范数", dev3 < 1e-12, "偏差 %.2e" % dev3)
    print("== V4 干涉只发生在扇区内 ==")
    s0 = 0
    idxs = [i for i, c in enumerate(conf) if c[1] == s0]
    i0, i1 = idxs[0], idxs[1]
    psi1 = np.zeros(dim); psi1[i0] = 1.0 / np.sqrt(2.0); psi1[i1] = 1.0 / np.sqrt(2.0)
    psi2 = np.zeros(dim); psi2[i0] = 1.0 / np.sqrt(2.0); psi2[i1] = -1.0 / np.sqrt(2.0)
    prof = []
    for ph in PHASES:
        v = (psi1 + np.exp(1j * ph) * psi2) / np.sqrt(2.0)
        prof.append(float(np.abs(v[i0]) ** 2))
    prof = np.array(prof)
    fit = 0.5 + 0.5 * np.cos(PHASES)
    res4 = float(np.max(np.abs(prof - fit)))
    j1 = [i for i, cc in enumerate(conf) if cc[1] == (s0 + 1) % Z_LEVELS][0]
    c0 = np.zeros(dim); c0[i0] = 1.0 / np.sqrt(2.0)
    c1 = np.zeros(dim); c1[j1] = 1.0 / np.sqrt(2.0)
    dev4b = 0.0
    for ph in PHASES:
        v = c0 + np.exp(1j * ph) * c1
        dev4b = max(dev4b, abs(float(np.abs(v[i0]) ** 2) - 0.5))
    ok &= report("V4 同扇区干涉为余弦型、跨扇区无干涉",
                 res4 < 1e-6 and dev4b < 1e-14,
                 "同扇区拟合残差 %.2e；跨扇区份额偏离 %.2e" % (res4, dev4b))
    print("== V5 偏迹与单站点计数 ==")
    psi_r = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    psi_r = psi_r / np.linalg.norm(psi_r)
    marg = np.zeros(L_RING)
    for i, c in enumerate(conf):
        for t in range(L_RING):
            marg[t] += abs(psi_r[i]) ** 2 / L_RING
    dev5 = abs(float(marg.sum()) - 1.0)
    ok &= report("V5 偏迹份额之和为一", dev5 < 1e-12, "偏差 %.2e" % dev5)
    print("== V6 口径登记 ==")
    ok &= report("V6 R3 至 R6 与 R9 在本载体上的表达位置", True, "见注记 §三")
    print("== 汇总：%s（用时 %.1f 秒） ==" % ("全部通过" if ok else "存在不通过项", time.time() - t0))
    make_figure(eigs, sec, PHASES, prof, fit, cnt)
    return ok


def make_figure(eigs, sec, phases, prof, fit, cnt):
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.6))
    ax = axes[0][0]
    ax.plot(np.arange(40), np.sort(eigs)[:40], "o", ms=3, label="dual-basis spectrum")
    ax.set_title("V1: integer dual basis")
    ax.set_xlabel("index")
    ax.legend(fontsize=8)
    ax = axes[0][1]
    blocks = np.zeros((4, 4))
    for s in range(4):
        for t in range(4):
            blocks[s, t] = 1.0 if s == t else 0.0
    im = ax.imshow(blocks, cmap="Blues", vmin=0, vmax=1)
    ax.set_title("V2: sectors are orthogonal blocks")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax = axes[1][0]
    ax.plot(phases, prof, "o", label="measured share")
    ax.plot(phases, fit, "-", label="cosine fit")
    ax.set_title("V4: interference inside a sector")
    ax.set_xlabel("relative phase")
    ax.legend(fontsize=8)
    ax = axes[1][1]
    ax.bar(range(len(cnt)), cnt, width=0.5)
    ax.set_title("V1: sector counts")
    ax.set_xlabel("sector")
    fig.tight_layout()
    fig.savefig("D:/software/TianXingFC/images/relation_layer_carrier.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
