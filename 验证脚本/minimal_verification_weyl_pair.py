# -*- coding: utf-8 -*-
r"""
minimal_verification_weyl_pair.py

量子力学攻坚计划 任务 三-1（P3-10/11）：对易关系与海森堡不等式的衔接。
判据先立后测；确定性（随机固定种子）。对应注记：
验证脚本/注记-魏尔对与海森堡衔接轮的最小验证：时钟-移位魏尔关系、提升关系的接缝分解与不确定性界的衔接条件.md

约定：站点基 |k>，相位 th_k = 2 pi k / Z；时钟 Om = e^{i th}；移位 S|k> = |k+1>；
模基为傅里叶基，Om 在其中取升移位（系数下标 m -> m+1，回绕落在 m=Z-1 与 0 之间）。
角动量 L 的本征值取 lam = -m（线性口径）或 lam = -n_s(m)（带符号口径），
使提升关系在接缝以外取 [L, Om] = +Om（与既有论文的 [J, e^{i th}] = hbar e^{i th} 同号）。

判据与理论期望：
  W1 魏尔关系（时钟-移位）：Om S = w S Om，残差 <= 1e-14（Z = 8, 16, 32）
  W2 提升关系的接缝结构（线性口径）：[L, Om] - Om 非零元恰 1 个，位于 (Z-1, 0)，幅度 = Z
  W3 接缝结构（带符号口径）：非零元恰 1 个，位于 (Z/2-1, Z/2)（奈奎斯特连接），幅度 = Z
  W4 覆盖空间对照：抹掉那条接缝连接后 [L, Om] - Om <= 1e-12（浮点噪声量级；未抹时残差为 Z，相差十余个量级）
  W5 罗伯逊关系全形式：200 个固定种子态上 dL * dOm >= |<[L, Om]>| / 2 - 1e-12
  W6 无接缝权带限态：<[L, Om]> = <Om>（偏差 <= 1e-12，浮点噪声量级），界退化为 |<Om>| / 2
  W7 准经典衔接：相位高斯族的比值 dL * dOm / (|<Om>| / 2) 随宽度收窄单调趋近 1，末档 <= 1.1
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\weyl_pair.png"
SEED = 20260919
ZS = (8, 16, 32)
ZW = 64
TOL = 1e-14
rows = []


def report(tag, ok, expect, actual):
    rows.append((tag, ok, expect, actual))
    print("[%s] %s" % ("通过" if ok else "未过", tag))
    print("      期望 " + expect)
    print("      实测 " + actual)


def ring(Z):
    w = np.exp(2j * np.pi / Z)
    Om = np.diag(w ** np.arange(Z))
    S = np.zeros((Z, Z), dtype=complex)
    for j in range(Z):
        S[j, (j - 1) % Z] = 1.0
    F = np.array([[w ** (-m * j) for j in range(Z)] for m in range(Z)]) / np.sqrt(Z)
    return w, Om, S, F


def to_modes(Om, F):
    return F.conj() @ Om @ F.T


def seam_of(R):
    idx = np.argwhere(np.abs(R) > 1e-9)
    return idx, np.array([R[i, j] for i, j in idx])


def signed(Z):
    m = np.arange(Z)
    return np.where(m < Z / 2.0, m, m - Z)


def lam_lin(Z):
    return -np.arange(Z, dtype=float)


def lam_sgn(Z):
    return -signed(Z).astype(float)


def phase_gauss(Z, sig, theta0):
    th = 2.0 * np.pi * np.arange(Z) / Z
    d = th - theta0
    psi = np.zeros(Z, dtype=complex)
    for j in (-1, 0, 1):
        psi = psi + np.exp(-((d - 2.0 * np.pi * j) ** 2) / (4.0 * sig ** 2))
    return psi / np.linalg.norm(psi)


def resid(Lam, Um):
    Lm = np.diag(Lam)
    return Lm @ Um - Um @ Lm - Um


w1, lin, sgn, cov = [], [], [], []
Rshow = None
for Z in ZS:
    w, Om, S, F = ring(Z)
    w1.append(float(np.abs(Om @ S - w * (S @ Om)).max()))
    Um = to_modes(Om, F)
    il, vl = seam_of(resid(lam_lin(Z), Um))
    isg, vs = seam_of(resid(lam_sgn(Z), Um))
    a = Um.copy()
    a[Z - 1, 0] = 0.0
    rca = float(np.abs(resid(lam_lin(Z), a)).max())
    b = Um.copy()
    b[Z // 2 - 1, Z // 2] = 0.0
    rcb = float(np.abs(resid(lam_sgn(Z), b)).max())
    lin.append((len(il), tuple(int(x) for x in il[0]) if len(il) == 1 else None,
                float(abs(vl[0])) if len(vl) else 0.0))
    sgn.append((len(isg), tuple(int(x) for x in isg[0]) if len(isg) == 1 else None,
                float(abs(vs[0])) if len(vs) else 0.0))
    cov.append((rca, rcb))
    if Z == 16:
        Rshow = np.abs(resid(lam_lin(Z), Um)).copy()

report("W1 魏尔关系（时钟-移位）", max(w1) <= TOL, "三档 Z 上 |Om S - w S Om| <= 1e-14",
       "三档最大残差 %.2e" % max(w1))

ok2 = all(c == 1 and pos == (Z - 1, 0) and abs(amp - Z) < 1e-9
          for (c, pos, amp), Z in zip(lin, ZS))
report("W2 提升关系接缝（线性口径）", ok2,
       "非零元恰 1 个、位于 (Z-1, 0)、幅度 = Z",
       "；".join("Z=%d: 非零 %d 个, 位置 %s, 幅度 %.6f" % (Z, c, str(pp), a)
                 for (c, pp, a), Z in zip(lin, ZS)))

ok3 = all(c == 1 and pos == (Z // 2 - 1, Z // 2) and abs(amp - Z) < 1e-9
          for (c, pos, amp), Z in zip(sgn, ZS))
report("W3 接缝结构（带符号口径）", ok3,
       "非零元恰 1 个、位于 (Z/2-1, Z/2)、幅度 = Z",
       "；".join("Z=%d: 非零 %d 个, 位置 %s, 幅度 %.6f" % (Z, c, str(pp), a)
                 for (c, pp, a), Z in zip(sgn, ZS)))

report("W4 覆盖空间对照（抹掉接缝连接）", max(max(x) for x in cov) <= 1e-12,
       "两种口径下残差 <= 1e-12（浮点噪声量级），未抹时残差为 Z",
       "；".join("Z=%d: 抹后线性 %.1e / 带符号 %.1e，未抹 %.0f（对比 %.1e 倍）" % (Z, x[0], x[1], Z, Z / max(x))
                 for Z, x in zip(ZS, cov)))

rng = np.random.default_rng(SEED)
w, Omw, Sw, Fw = ring(ZW)
Umw = to_modes(Omw, Fw)
Lam = lam_sgn(ZW)
Lm_op = np.diag(Lam)
com_op = Lm_op @ Umw - Umw @ Lm_op


def stats(psi):
    c = Fw.conj() @ psi
    c = c / np.linalg.norm(c)
    p = np.abs(c) ** 2
    m1 = float(np.sum(p * Lam))
    dL = float(np.sqrt(max(np.sum(p * Lam ** 2) - m1 * m1, 0.0)))
    om = np.vdot(c, Umw @ c)
    dOm = float(np.sqrt(max(1.0 - abs(om) ** 2, 0.0)))
    com = np.vdot(c, com_op @ c)
    return dL, dOm, om, com


worst = 1e9
for _ in range(200):
    psi = rng.normal(size=ZW) + 1j * rng.normal(size=ZW)
    psi = psi / np.linalg.norm(psi)
    dL, dOm, om, com = stats(psi)
    worst = min(worst, dL * dOm - abs(com) / 2.0)
report("W5 罗伯逊关系（全形式）", worst >= -1e-12,
       "200 个固定种子态上 dL*dOm - |<[L, Om]>|/2 >= -1e-12", "最小余量 %.3e" % worst)

band = np.arange(ZW // 8, 3 * ZW // 8)
gap = 0.0
for _ in range(50):
    c = np.zeros(ZW, dtype=complex)
    c[band] = rng.normal(size=band.size) + 1j * rng.normal(size=band.size)
    c = c / np.linalg.norm(c)
    psi = Fw.T @ c
    dL, dOm, om, com = stats(psi)
    gap = max(gap, abs(com - om))
report("W6 无接缝权带限态的退化形式", gap <= 1e-12,
       "|<[L, Om]> - <Om>| <= 1e-12，浮点噪声量级（带限带 %d 到 %d）" % (band[0], band[-1]),
       "最大偏差 %.2e" % gap)

sigmas = (0.60, 0.45, 0.35, 0.25)
ratios, omegas = [], []
for sg in sigmas:
    psi = phase_gauss(ZW, sg, np.pi)
    dL, dOm, om, com = stats(psi)
    ratios.append(float(dL * dOm / (abs(om) / 2.0)))
    omegas.append(float(abs(om)))
mono = all(ratios[i] >= ratios[i + 1] - 1e-9 for i in range(len(ratios) - 1))
report("W7 准经典衔接（相位高斯族）", mono and ratios[-1] <= 1.1,
       "比值随宽度收窄单调趋近 1，末档 <= 1.1",
       "；".join("sig=%.2f: 比值 %.4f, |<Om>| %.4f" % (s, r, o)
                 for s, r, o in zip(sigmas, ratios, omegas)))

fig, ax = plt.subplots(1, 3, figsize=(15.0, 4.2))
im = ax[0].imshow(Rshow, cmap="viridis", vmin=0.0, vmax=float(Rshow.max()))
ax[0].set_title("Z=16 线性口径的失配矩阵（唯一非零元在接缝）")
ax[0].set_xlabel("模指标 m'")
ax[0].set_ylabel("模指标 m")
fig.colorbar(im, ax=ax[0], fraction=0.046)
ax[1].plot(ZS, [a for (_, _, a) in lin], "o-", label="线性口径")
ax[1].plot(ZS, [a for (_, _, a) in sgn], "s--", label="带符号口径")
ax[1].plot(ZS, list(ZS), "k:", label="y = Z")
ax[1].set_title("失配幅度随格点数的标度")
ax[1].set_xlabel("格点数 Z")
ax[1].set_ylabel("失配幅度 |[L, Om] - Om|")
ax[1].legend()
ax[2].plot(sigmas, ratios, "o-", label="比值")
ax[2].axhline(1.0, color="k", linestyle=":")
ax[2].axhline(1.1, color="r", linestyle="--", label="判据上界 1.1")
ax[2].set_title("相位高斯族的罗伯逊比值（Z=64）")
ax[2].set_xlabel("相位高斯宽度 sigma_theta（弧度）")
ax[2].set_ylabel("dL * dOm / (|<Om>| / 2)")
ax[2].legend()
plt.tight_layout()
plt.savefig(IMG, dpi=150)
plt.close(fig)

bad = [r for r in rows if not r[1]]
print("---- 判据 %d 项，通过 %d 项；图存 %s ----" % (len(rows), len(rows) - len(bad), IMG))
sys.exit(1 if bad else 0)
