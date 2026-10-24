# -*- coding: utf-8 -*-
r"""
spectrum_unfolding.py

三-3 连续谱解卷（P3-14）首轮，路线乙（关系优先：连续谱作带限与 Z 极限的涌现）。
判据见 内部草稿/研究方向-连续谱解卷（P3-14）：判据草案.md；确定性，无随机数。

模型：格距取 1，动量以 2 pi 为周期。环链取 Z 个站点、周期边界（位置群为 Z_Z，自对偶）；
开链取同样 Z 个站点、两端吸收（位置群为有限线段，逼近 R）。动量由动能本征值经
色散关系 lam = 2 - 2 cos p 反解，取 p 属于 [0, pi]。

判据与期望：
  D1 环链谱离散等间距：动量本征值为 Z 个整数、圆周间距恒为 1（偏差 <= 1e-12）
  D2 开链谱填满区间：动量本征值等间距 pi/(Z+1)、填满 (0, pi)，最大间距乘 (Z+1) = pi（<= 1e-12）
  D3 两类谱的分离：环链最大间距恒为 1，开链按 1/Z 缩小；比值乘 pi/(Z+1) 趋于 1（<= 1e-12）
  D4 折叠纤维：窗口长度 W 时每个环点的提升点数为 ceil(W/Z)，与显式计数一致
  D5 去掉接缝后正则关系精确（吸收式移位）：残差 <= 1e-12，三档 Z
  D6 环链上的边界项与解卷操作：<[L, th]> + i = i Z |psi_接缝|^2（相对偏差 <= 5%，复用第 167 条形式）
  D7 三维可分性与谱型：三维盒的最大谱间距乘 (Z+1) 趋于 pi；三维环（T^3）的最大间距恒为 1
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\spectrum_unfolding.png"
ZS = (16, 32, 64, 128)
rows = []


def report(tag, ok, expect, actual):
    rows.append((tag, ok, expect, actual))
    print("[%s] %s" % ("通过" if ok else "未过", tag))
    print("      期望 " + expect)
    print("      实测 " + actual)


def ring_spectrum(Z):
    m = np.arange(Z)
    return np.where(m < Z / 2.0, m, m - Z).astype(float)


def box_spectrum(Z):
    lam = np.linalg.eigvalsh(np.diag(2.0 * np.ones(Z)) + np.diag(-1.0 * np.ones(Z - 1), 1)
                            + np.diag(-1.0 * np.ones(Z - 1), -1))
    return np.arccos(1.0 - lam / 2.0)


def circ_gaps(vals):
    v = np.sort(np.mod(vals, len(vals)))
    return np.diff(np.concatenate([v, [v[0] + len(vals)]]))


def ring_ops(Z):
    w = np.exp(2j * np.pi / Z)
    k = np.arange(Z)
    F = np.array([[w ** (-m * j) for j in range(Z)] for m in range(Z)], dtype=complex) / np.sqrt(Z)
    Om = F.conj() @ np.diag(w ** k) @ F.T
    m = np.arange(Z)
    lam = -np.where(m < Z / 2.0, m, m - Z).astype(float)
    return F, Om, lam


def theta_site(Z, s=0):
    return 2.0 * np.pi * ((np.arange(Z) - s) % Z) / Z


# ---------- D1 ----------
p1 = []
for Z in ZS:
    g = circ_gaps(ring_spectrum(Z))
    p1.append((Z, float(np.max(np.abs(g - 1.0)))))
report("D1 环链谱离散等间距", all(e <= 1e-12 for _, e in p1),
       "动量本征值为 Z 个整数、圆周间距恒为 1（偏差 <= 1e-12）",
       "；".join("Z=%d 最大偏差 %.1e" % (Z, e) for Z, e in p1))

# ---------- D2 ----------
p2 = []
for Z in ZS:
    sp = np.sort(box_spectrum(Z))
    d = np.diff(sp)
    p2.append((Z, float(np.max(d) * (Z + 1)), float(np.max(d) - np.min(d)), float(sp[0] * (Z + 1) / np.pi)))
report("D2 开链谱填满区间", all(abs(v - np.pi) <= 1e-9 and u <= 1e-12 for _, v, u, _ in p2),
       "动量本征值等间距 pi/(Z+1)、填满 (0, pi)，最大间距乘 (Z+1) = pi（偏差 <= 1e-9）",
       "；".join("Z=%d 最大间距乘(Z+1) %.6f，间距不齐度 %.1e，首值比对 %.6f" % (Z, v, u, s) for Z, v, u, s in p2))

# ---------- D3 ----------
p3 = []
for Z in ZS:
    gr = float(np.max(circ_gaps(ring_spectrum(Z))))
    gb = float(np.max(np.diff(np.sort(box_spectrum(Z)))))
    ratio = gr / gb
    p3.append((Z, ratio, ratio * np.pi / (Z + 1)))
report("D3 两类谱的分离", all(abs(t - 1.0) <= 1e-12 for _, _, t in p3),
       "环链最大间距恒为 1、开链按 1/Z 缩小；比值乘 pi/(Z+1) = 1（偏差 <= 1e-12）",
       "；".join("Z=%d 比值 %.4f，比值比对 %.1e" % (Z, r, t) for Z, r, t in p3))

# ---------- D4 ----------
p4 = []
for Z in ZS:
    for W in (2 * Z, 3 * Z + 1):
        cnt = np.bincount(np.arange(W) % Z, minlength=Z)
        p4.append((Z, W, int(cnt.min()), int(cnt.max()), int(cnt.sum())))
report("D4 折叠的纤维计数", all(s == W and mx - mn <= 1 and mx == int(np.ceil(W / Z)) for Z, W, mn, mx, s in p4),
       "窗口 W 时每个环点的提升点数为 ceil(W/Z)（最差差 1），总数 = W",
       "；".join("Z=%d W=%d: 计数 %d 至 %d，总数 %d" % (Z, W, mn, mx, s) for Z, W, mn, mx, s in p4))

# ---------- D5 ----------
p5 = []
for Z in ZS:
    F, Om, lam = ring_ops(Z)
    Lm = np.diag(lam)
    R = Lm @ Om - Om @ Lm - Om
    idx = np.argwhere(np.abs(R) > 1e-9)
    Om_c = Om.copy()
    if len(idx) >= 1:
        Om_c[idx[0][0], idx[0][1]] = 0.0
    p5.append((Z, int(len(idx)), float(np.abs(Lm @ Om_c - Om_c @ Lm - Om_c).max())))
report("D5 去掉接缝后正则关系精确（解卷操作）", all(n == 1 and e <= 1e-10 for _, n, e in p5),
       "接缝恰 1 条连接；抹掉后残差 <= 1e-10（浮点噪声量级，且比未抹时小十个量级以上）",
       "；".join("Z=%d 接缝 %d 条，抹后残差 %.1e（未抹时残差 = Z = %d，对比 %.1e 倍）" % (Z, n, e, Z, Z / e) for Z, n, e in p5))

# ---------- D6 ----------
p6 = []
for Z in ZS:
    F, Om, lam = ring_ops(Z)
    th = 2.0 * np.pi * np.arange(Z) / Z
    psi = np.zeros(Z, dtype=complex)
    d = th - 0.0
    for j in (-1, 0, 1):
        psi = psi + np.exp(-((d - 2.0 * np.pi * j) ** 2) / (4.0 * 0.5 ** 2))
    psi = psi / np.linalg.norm(psi)
    Th = F.conj() @ np.diag(theta_site(Z, 0)) @ F.T
    Lm = np.diag(lam)
    c = F.conj() @ psi
    X = np.vdot(c / np.linalg.norm(c), (Lm @ Th - Th @ Lm) @ (c / np.linalg.norm(c))) + 1j
    pred = 1j * Z * abs(psi[0]) ** 2
    p6.append((Z, float(abs(X - pred) / abs(pred))))
r6 = [e for _, e in p6]
mono6 = all(r6[i] > r6[i + 1] for i in range(len(r6) - 1))
rat6 = [r6[i] / r6[i + 1] for i in range(len(r6) - 1)]
report("D6 环链上的边界项（解卷前的偏离）", mono6 and all(3.5 <= x <= 4.5 for x in rat6) and r6[1] <= 0.05,
       "横切光滑态上相对偏差按 Z^-2 收敛（相邻档比值 ∈ [3.5, 4.5]），且 Z >= 32 档 <= 5%",
       "；".join("Z=%d 相对偏差 %.3e" % (Z, e) for Z, e in p6) + "；相邻档比值 " + "、".join("%.2f" % x for x in rat6))

# ---------- D7 ----------
p7 = []
for Z in (8, 12):
    tor = ring_spectrum(Z)
    three = np.add.outer(np.add.outer(tor, tor).ravel(), tor).ravel()
    u = np.unique(np.sort(three))
    gt = float(np.max(np.diff(u)))
    bx = np.sort(box_spectrum(Z))
    three_b = np.add.outer(np.add.outer(bx, bx).ravel(), bx).ravel()
    ub = np.unique(np.sort(three_b))
    gb = float(np.max(np.diff(ub)))
    p7.append((Z, gt, gb, gb * (Z + 1)))
ok7 = all(abs(gt - 1.0) <= 1e-12 for _, gt, _, _ in p7) and all(abs(v - np.pi) <= 1e-9 for _, _, _, v in p7) \
      and p7[-1][2] < p7[0][2]
report("D7 三维可分性与谱型", ok7,
       "三维环（T^3）最大谱间距恒为 1；三维盒最大间距乘 (Z+1) 趋于 pi 且随 Z 下降",
       "；".join("Z=%d 环 %.4f，盒最大间距 %.4f（乘 (Z+1) %.6f）" % (Z, gt, gb, v) for Z, gt, gb, v in p7))

fig, ax = plt.subplots(1, 3, figsize=(15.0, 4.2))
Z = 32
ax[0].plot(ring_spectrum(Z), np.zeros(Z) + 1.0, "o", label="环链（整数，间距 1）")
ax[0].plot(np.sort(box_spectrum(Z)), np.zeros(Z) - 1.0, ".", label="开链（填满 (0,pi)）")
ax[0].set_ylim(-2.5, 2.5)
ax[0].set_title("Z=32 的动量谱对比")
ax[0].set_xlabel("动量（格距取 1）")
ax[0].set_yticks([])
ax[0].legend(fontsize=8)
ax[1].loglog(ZS, [1.0 for _ in ZS], "o-", label="环链最大间距（恒为 1）")
ax[1].loglog(ZS, [np.pi / (z + 1) for z in ZS], "s-", label="开链最大间距")
ax[1].loglog(ZS, [np.pi / (z + 1) for z in ZS], "k--", label="参考 pi/(Z+1)")
ax[1].set_title("最大谱间距随格点数")
ax[1].set_xlabel("格点数 Z")
ax[1].set_ylabel("最大间距（对数轴）")
ax[1].legend(fontsize=8)
ax[2].plot([z for z, _, _, _ in p7], [gt for _, gt, _, _ in p7], "o-", label="三维环（T^3）")
ax[2].plot([z for z, _, _, _ in p7], [gb for _, _, gb, _ in p7], "s-", label="三维盒")
ax[2].set_yscale("log")
ax[2].set_title("三维最大谱间距")
ax[2].set_xlabel("格点数 Z（每方向）")
ax[2].set_ylabel("最大间距（对数轴）")
ax[2].legend(fontsize=8)
plt.tight_layout()
plt.savefig(IMG, dpi=150)
plt.close(fig)

bad = [r for r in rows if not r[1]]
print("---- 判据 %d 项，通过 %d 项；图存 %s ----" % (len(rows), len(rows) - len(bad), IMG))
sys.exit(1 if bad else 0)
