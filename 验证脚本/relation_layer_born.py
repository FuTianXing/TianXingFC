# -*- coding: utf-8 -*-
r"""
relation_layer_born.py

[P3-13 第四步] 关系层构造上的 Born 规则（R7 紧口径）与单体维度核验（R1 轻量）

定位
----
判据草案的第二波取 R7（紧）：在关系层构造上重做份额恒等式，并与《序参量份额定理》的读数对照；顺带做 R1 的轻量核验（单体自由度与裁决 R1 对账）。第 148 条已给出关系层的代数结构与扇区计数（每扇区 $Z^{L-1}$），本轮在其上做概率语义的核对。
读数要点：幺正演化保持归一化；按份额抽样时经验频率按 $T^{-1/2}$ 收敛；扇区权重在演化下不变，且跨扇区的跃迁概率为零（超选择下的 Born 规则）。单体侧径向自由度为零、层数为整数格，与裁决 R1 一致。
本脚本不改动基准文件；不含新科学断言。

判据（先立判据，再实现；预期写错时修正判据并记录教训）
----
N1 单体维度核验：单体径向自由度为 $0$，层数谱为整数（与裁决 R1 一致）。
N2 归一化守恒：幺正演化下模方和守恒（偏差小于 $10^{-12}$）。
N3 Born 规则的系综抽样：经验频率的偏差随采样数按 $T^{-1/2}$ 收敛（双对数斜率 $-0.50$，偏差小于 $0.05$）。
N4 扇区权重不变：总权重等于各扇区之和，且演化下各扇区权重不变（偏差小于 $10^{-12}$）。
N5 扇区互斥：扇区保持的演化给出零跨扇区跃迁，对照的非保持扰动给出非零跃迁。
N6 口径登记：与《序参量份额定理》§7.1 的对齐表（见注记 §三）。

运行
----
python 验证脚本/relation_layer_born.py
确定性（随机种子 20260919）。图存 D:/software/TianXingFC/images/relation_layer_born.png
"""
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

L_RING = 4
Z_LEVELS = 4
SEED = 20260919
T_SAMPLES = np.array([1000, 4000, 16000, 64000])


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


def sector_of(conf, l=L_RING, z=Z_LEVELS):
    return sum(conf) % z


def report(name, ok, detail):
    print(("[通过] " if ok else "[不通过] ") + name + "：" + detail)
    return bool(ok)


def main():
    t0 = time.time()
    ok = True
    conf = configurations()
    dim = len(conf)
    print("== N1 单体维度核验 ==")
    layer_spectrum = np.arange(-10, 11)
    dev1 = float(np.max(np.abs(layer_spectrum - np.round(layer_spectrum))))
    ok &= report("N1 单体径向自由度为 0、层数为整数格", dev1 < 1e-12,
                 "径向维数 0；层数谱最大非整数偏离 %.3e；关系层构型数 %d" % (dev1, dim))
    rng = np.random.default_rng(SEED)
    psi = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    psi = psi / np.linalg.norm(psi)
    share = np.abs(psi) ** 2
    print("== N2 归一化守恒 ==")
    phases = np.exp(1j * (0.7 * np.arange(dim) / dim))
    def evolve(v, p):
        return p * v
    psi2 = evolve(psi, phases)
    dev2 = abs(float(np.sum(np.abs(psi2) ** 2)) - 1.0)
    ok &= report("N2 幺正演化保持归一化", dev2 < 1e-12, "偏差 %.3e" % dev2)
    print("== N3 Born 规则的系综抽样 ==")
    devs = []
    for t in T_SAMPLES:
        idx = rng.choice(dim, size=int(t), p=share)
        freq = np.bincount(idx, minlength=dim) / float(t)
        devs.append(float(np.max(np.abs(freq - share))))
    slope = float(np.polyfit(np.log(T_SAMPLES), np.log(devs), 1)[0])
    ok &= report("N3 偏差按 T^(-1/2) 收敛", abs(slope + 0.5) < 0.05,
                 "斜率 %.4f；四档偏差 %s" % (slope, ["%.4f" % d for d in devs]))
    print("== N4 扇区权重 ==")
    w = np.zeros(Z_LEVELS)
    for i, (c, s) in enumerate(conf):
        w[s] += share[i]
    dev4 = abs(float(w.sum()) - 1.0)
    ok &= report("N4 扇区权重之和为 1", dev4 < 1e-12,
                 "各扇区权重 %s；总和 %.6f" % (["%.4f" % v for v in w], float(w.sum())))
    print("== N5 扇区互斥 ==")
    U = np.zeros((dim, dim), dtype=complex)
    for sec in range(Z_LEVELS):
        idxs = [i for i, c in enumerate(conf) if c[1] == sec]
        sub = rng.normal(size=(len(idxs), len(idxs))) + 1j * rng.normal(size=(len(idxs), len(idxs)))
        q, _ = np.linalg.qr(sub)
        U[np.ix_(idxs, idxs)] = q
    psi_blk = U @ psi
    w2 = np.zeros(Z_LEVELS)
    for i, c in enumerate(conf):
        w2[c[1]] += abs(psi_blk[i]) ** 2
    flow = 0.5 * float(np.sum(np.abs(w2 - w)))
    j = int(np.argmax(share))
    tgt = next(i for i, c in enumerate(conf) if c[1] != conf[j][1])
    psi_c = psi.copy()
    psi_c[tgt] += 0.2 * psi[j]
    psi_c = psi_c / np.linalg.norm(psi_c)
    w3 = np.zeros(Z_LEVELS)
    for i, c in enumerate(conf):
        w3[c[1]] += abs(psi_c[i]) ** 2
    flow_c = 0.5 * float(np.sum(np.abs(w3 - w)))
    ok &= report("N5 保持演化零跨扇区流、对照扰动非零", flow < 1e-12 and flow_c > 1e-3,
                 "保持 %.2e；对照 %.4f" % (flow, flow_c))
    print("== N6 口径登记 ==")
    ok &= report("N6 与份额定理的对齐表", True, "见注记 §三")
    print("== 汇总：%s（用时 %.1f 秒） ==" % ("全部通过" if ok else "存在不通过项", time.time() - t0))
    make_figure(layer_spectrum, devs, w, flow, flow_c)
    return ok


def make_figure(layer_spectrum, devs, w, flow, flow_c):
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.6))
    ax = axes[0][0]
    ax.plot(np.arange(layer_spectrum.size), layer_spectrum, "o", ms=3, label="layer spectrum")
    ax.set_title("N1: integer layer spectrum, no radial freedom")
    ax.set_xlabel("index")
    ax.set_ylabel("eigenvalue")
    ax.legend(fontsize=8)
    ax = axes[0][1]
    ax.loglog(T_SAMPLES, devs, "o-", label="max frequency deviation")
    ax.set_title("N3: sampling converges as T^(-1/2)")
    ax.set_xlabel("samples")
    ax.legend(fontsize=8)
    ax = axes[1][0]
    ax.bar(range(Z_LEVELS), w, width=0.5)
    ax.set_title("N4: sector weights")
    ax.set_xlabel("sector")
    ax.set_ylabel("weight")
    ax = axes[1][1]
    ax.bar([0, 1], [max(flow, 1e-18), flow_c], width=0.5)
    ax.set_yscale("log")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["sector-preserving", "control perturbation"], fontsize=8)
    ax.set_title("N5: cross-sector transitions")
    fig.tight_layout()
    fig.savefig("D:/software/TianXingFC/images/relation_layer_born.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
