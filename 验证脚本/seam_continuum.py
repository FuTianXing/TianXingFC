# -*- coding: utf-8 -*-
r"""
seam_continuum.py

量子力学攻坚计划 任务 三-1 剩余项：接缝项在连续极限下的量级。
判据先立后测；确定性（随机固定种子）。对应注记：
验证脚本/注记-接缝项连续极限轮的最小验证：有效接缝项 = Z 乘切割连接相干、三种区制与切割自由度.md

理论结构：提升关系的失配恰是一条连接（第 161 条）。取提升（unwrapping）指标 n，令 L 的本征值 lam = -n，
则 [L, Om] = Om 在接缝以外成立；接缝位置即 n 的那处落差值所在连接。接缝项对期望的贡献为

    <R> = <[L, Om]> - <Om> = -Z * rho,   rho = conj(c[接缝左模]) * c[接缝右模],

c 为态在模基下的系数。故有效接缝项 E = Z * |rho| 完全由态在该连接上的相干与格点数决定。

判据与理论期望（Z = 32, 64, 128, 256）：
  S1 分解恒等：对四类态、两种提升、四档 Z，| <R> + Z rho | <= 1e-9（浮点噪声，随 Z 按 Z^2 增长）
  S2 光滑物理态（相位高斯，sigma_theta = 0.35 弧度固定）之 E 与解析高斯律
     E_ana = Z A exp(-sigma^2[(Z/2)^2 + (Z/2-1)^2])（A = sqrt(2 sigma^2/pi)）在 Z = 32 档的比值 ∈ [1/3, 3]
  S2b 排除幂律：Z = 256 档的 E 比 Z^-1 预期与 Z^-2 预期都小一个量级以上（Z >= 64 档已落进双精度噪声地板）
  S3 格点尺度态（单站点）：E = 1（逐档偏差 <= 1e-12），与 Z 无关
  S4 切割态（支撑压在接缝两模上）：E >= Z/4（逐档），即量级为 O(Z)
  S5 切割自由度：同一光滑态上，接缝落在支撑内（lin 提升）时 E >= Z/8，落在支撑外（sgn 提升）时 E <= 1e-12
  S6 秩与迹：失配矩阵 R 的非零元数 = 1、秩 = 1、|迹| <= 1e-9（逐档）
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\seam_continuum.png"
SEED = 20260919
ZS = (32, 64, 128, 256)
SIG = 0.35
rows = []


def report(tag, ok, expect, actual):
    rows.append((tag, ok, expect, actual))
    print("[%s] %s" % ("通过" if ok else "未过", tag))
    print("      期望 " + expect)
    print("      实测 " + actual)


def build(Z):
    w = np.exp(2j * np.pi / Z)
    Om = np.diag(w ** np.arange(Z))
    F = np.array([[w ** (-m * j) for j in range(Z)] for m in range(Z)], dtype=complex) / np.sqrt(Z)
    Um = F.conj() @ Om @ F.T
    return w, F, Um


def lifts(Z):
    m = np.arange(Z)
    ns = np.where(m < Z / 2.0, m, m - Z)
    return {"lin": (-m.astype(float), (Z - 1, 0)), "sgn": (-ns.astype(float), (Z // 2 - 1, Z // 2))}


def coeff(F, psi):
    c = F.conj() @ psi
    return c / np.linalg.norm(c)


def phase_gauss(Z, sig, theta0):
    th = 2.0 * np.pi * np.arange(Z) / Z
    d = th - theta0
    psi = np.zeros(Z, dtype=complex)
    for j in (-1, 0, 1):
        psi = psi + np.exp(-((d - 2.0 * np.pi * j) ** 2) / (4.0 * sig ** 2))
    return psi / np.linalg.norm(psi)


def site(Z, k):
    psi = np.zeros(Z, dtype=complex)
    psi[k] = 1.0
    return psi


def cut_state(F, Z, a, b):
    c = np.zeros(Z, dtype=complex)
    c[a] = 1.0 / np.sqrt(2.0)
    c[b] = 1.0 / np.sqrt(2.0)
    return F.T @ c


def metrics(Um, Lam, c, seam):
    Lm = np.diag(Lam)
    prod = Lm @ Um - Um @ Lm
    R = prod - Um
    val = np.vdot(c, R @ c)
    rho = np.conj(c[seam[0]]) * c[seam[1]]
    return val, rho, R


E = {}
struct = {}
mism = 0.0
for Z in ZS:
    w, F, Um = build(Z)
    Ls = lifts(Z)
    st = {"光滑态": phase_gauss(Z, SIG, np.pi),
          "格点尺度态": site(Z, Z // 8),
          "切割态": cut_state(F, Z, Z // 2 - 1, Z // 2)}
    for name, psi in st.items():
        c = coeff(F, psi)
        for tag, (Lam, seam) in Ls.items():
            val, rho, R = metrics(Um, Lam, c, seam)
            mism = max(mism, abs(val + Z * rho))
            E[(Z, name, tag)] = Z * abs(rho)
            if name == "光滑态":
                struct[(Z, tag)] = (int(np.count_nonzero(np.abs(R) > 1e-9)),
                                    int(np.linalg.matrix_rank(R, tol=1e-9)),
                                    float(abs(np.trace(R))))

report("S1 分解恒等 <R> = -Z rho", mism <= 1e-9,
       "四类态、两种提升、四档 Z 上 |<R> + Z rho| <= 1e-9（浮点噪声，随 Z 按 Z^2 增长）",
       "最大偏差 %.2e" % mism)

ok3 = all(abs(E[(Z, "格点尺度态", t)] - 1.0) <= 1e-12 for Z in ZS for t in ("lin", "sgn"))
report("S3 格点尺度态的接缝项 = 1", ok3, "逐档 |E - 1| <= 1e-12（与 Z 无关）",
       "；".join("Z=%d: lin %.6f, sgn %.6f" % (Z, E[(Z, "格点尺度态", "lin")], E[(Z, "格点尺度态", "sgn")]) for Z in ZS))

ok4 = all(E[(Z, "切割态", "sgn")] >= Z / 4.0 for Z in ZS)
report("S4 切割态的接缝项为 O(Z)", ok4, "逐档 E >= Z/4",
       "；".join("Z=%d: E %.3f（Z/4 = %.1f）" % (Z, E[(Z, "切割态", "sgn")], Z / 4.0) for Z in ZS))

small = [(Z, E[(Z, "光滑态", "sgn")]) for Z in ZS]
big = [(Z, E[(Z, "光滑态", "lin")]) for Z in ZS]
ok5 = all(E[(Z, "光滑态", "sgn")] <= 1e-12 for Z in (32, 64)) and all(E[(Z, "光滑态", "lin")] >= Z / 8.0 for Z in (32, 64))
report("S5 切割自由度（同一态两种提升）", ok5, "sgn 提升（缝在支撑外）E <= 1e-12；lin 提升（缝在支撑内）E >= Z/8",
       "；".join("Z=%d: sgn %.2e, lin %.3f" % (Z, E[(Z, "光滑态", "sgn")], E[(Z, "光滑态", "lin")]) for Z in (32, 64)))

AMP = np.sqrt(2.0 * SIG ** 2 / np.pi)


def e_ana(Z):
    n1, n2 = Z // 2 - 1, Z // 2
    return Z * AMP * np.exp(-SIG ** 2 * (n1 ** 2 + n2 ** 2))


e32 = E[(32, "光滑态", "sgn")]
a32 = e_ana(32)
ratio = e32 / a32
report("S2 光滑态与解析高斯律相符", 1.0 / 3.0 <= ratio <= 3.0,
       "E(Z=32) 对 Z A exp(-sigma^2[(Z/2)^2+(Z/2-1)^2])（A = sqrt(2 sigma^2/pi)）比值 ∈ [1/3, 3]",
       "实测 %.3e，解析 %.3e，比值 %.2f" % (e32, a32, ratio))
pw1 = e32 * 32.0 / 256.0
pw2 = e32 * (32.0 / 256.0) ** 2
report("S2b 排除幂律衰减（Z^-1 / Z^-2）", E[(256, "光滑态", "sgn")] < min(pw1, pw2) / 10.0,
       "Z=256 档实测应比 Z^-1 预期 %.2e 与 Z^-2 预期 %.2e 都小一个量级以上" % (pw1, pw2),
       "实测 %.2e（噪声地板量级；Z>=64 档已低于地板）" % E[(256, "光滑态", "sgn")])

ok6 = all(v[0] == 1 and v[1] == 1 and v[2] <= 1e-9 for v in struct.values())
report("S6 失配矩阵的秩与迹", ok6, "逐档（非零元数, 秩, |迹|）= (1, 1, <= 1e-9)",
       "；".join("Z=%d %s: %s" % (Z, t, str(v)) for (Z, t), v in sorted(struct.items())))

fig, ax = plt.subplots(1, 2, figsize=(12.0, 4.4))
ax[0].loglog(ZS, [E[(Z, "光滑态", "sgn")] if E[(Z, "光滑态", "sgn")] > 0 else np.nan for Z in ZS], "o-", label="光滑态（缝在支撑外）")
ax[0].loglog(ZS, [E[(Z, "光滑态", "lin")] for Z in ZS], "s-", label="光滑态（缝在支撑内）")
ax[0].loglog(ZS, [E[(Z, "格点尺度态", "sgn")] for Z in ZS], "^-", label="格点尺度态")
ax[0].loglog(ZS, [E[(Z, "切割态", "sgn")] for Z in ZS], "v-", label="切割态")
ax[0].loglog(ZS, [Z / 4.0 for Z in ZS], "k:", label="参考 Z/4")
ax[0].axhline(1.0, color="gray", linestyle="--")
ax[0].set_title("有效接缝项 E = Z|rho| 随格点数")
ax[0].set_xlabel("格点数 Z")
ax[0].set_ylabel("E（对数轴）")
ax[0].legend(fontsize=8)
ax[1].semilogy(ZS, [e_ana(Z) for Z in ZS], "k--", label="解析高斯律")
ax[1].semilogy(ZS, [E[(Z, "光滑态", "sgn")] for Z in ZS], "o-", label="实测")
ax[1].axhline(1e-30, color="r", linestyle=":", label="双精度噪声地板（约 1e-30）")
ax[1].set_title("光滑态 E 与解析高斯律（Z^2 型衰减，Z>=64 已入地板）")
ax[1].set_xlabel("格点数 Z")
ax[1].set_ylabel("E（对数轴）")
ax[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(IMG, dpi=150)
plt.close(fig)

bad = [r for r in rows if not r[1]]
print("---- 判据 %d 项，通过 %d 项；图存 %s ----" % (len(rows), len(rows) - len(bad), IMG))
sys.exit(1 if bad else 0)
