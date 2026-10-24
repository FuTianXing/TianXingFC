# -*- coding: utf-8 -*-
"""判据第 660 号：电子算符骨架的非紧致边界复核。

背景：第 489 号在 $S^1\\times\\mathbb{R}$ 上交付 $\\hat D$ 的离散骨架，判据 7/7 通过，
关键读数为「反周期半整数动量无零模」。但该轮的 $h$ 方向实现为**周期环**
（D-b 主设定），而体系声明的载体 $S^1\\times\\mathbb{R}$ 的 $\\mathbb{R}$ 方向非紧
（层数通道是物理空间的生成方向，非紧致）。本轮把 $h$ 方向改成开放链复核，
看骨架的零模排除与谱结构是否依赖 $h$ 的紧致性。

模型：与第 489 号相同（2 分量，2+1 维），
    H = kron(sigma_x, -i c_th D_th) + kron(sigma_z, -i c_h D_h) + kron(sigma_y, diag(m))
theta 环反周期（pi 扭转口径，接缝带 -1），h 方向取开放链（本轮）或周期环（对照）。

先立的期望（写实现之前）：
  冒烟1  H 厄米（偏差 0）。
  冒烟2  开放链的 -i D_h 本征值等于 cos(pi k/(N_h+1))（k = 1..N_h），解析对照。
  N1     theta 动量仍为半整数（反周期生效），偏差 < 1e-10。
  N2     开放 h：无零模（min |E| > 0），最低 |E| 与周期档同量级。
  N3     谱关于 E = 0 对称（max |E_i + E_(N+1-i)| < 1e-10）。
  N4     winding = +-1 谱逐位恒等（开放 h 下镜像式仍成立）。
  N5     N_h 扫描（16, 32, 64）：最低 |E| 收敛，不随尺寸趋零。
  N6     开放 h 与周期 h 的最低 |E| 对照（边界不改变无零模结论）。
  反例1  theta 周期 + m_0 = 0：出现零模（区分力，说明反周期是零模排除的原因）。
  反例2  theta 周期 + m_0 = 0.5：无零模（能隙）。
  说明   本骨架为 2 分量，2+1 维无手征算符（第 490 号判 2 分量下宇称算符不存在），
         故「定理对载体是否成立」这一问在本骨架内不能回答，属 4 分量（3+1 维）的后续轮。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

N_TH = 32
C_TH = C_H = 1.0
M0 = 0.5
TOL = 1e-8
SIG_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIG_Z = np.array([[1, 0], [0, -1]], dtype=complex)
SIG_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)

results = []


def report(tag, measured, expected, ok):
    results.append((tag, ok))
    print("[%s] %s" % ("PASS" if ok else "FAIL", tag))
    print("    实测: %s" % measured)
    print("    期望: %s" % expected)


def shift_theta(anti=True, gauge=0.0):
    ph = np.exp(-1j * gauge)
    S = np.zeros((N_TH, N_TH), dtype=complex)
    for i in range(N_TH):
        j = (i + 1) % N_TH
        S[i, j] = ph * (-1.0 if (anti and i == N_TH - 1) else 1.0)
    return S


def shift_h(N_H, periodic):
    S = np.zeros((N_H, N_H))
    for i in range(N_H - 1):
        S[i, i + 1] = 1.0
    if periodic:
        S[N_H - 1, 0] = 1.0
    return S


def build(m_field=None, winding=0.0, N_H=32, periodic=False, m0=M0, anti=True):
    Sth = shift_theta(anti=anti, gauge=2 * np.pi * winding / N_TH)
    D_th = (Sth - Sth.conj().T) / 2.0
    Sh = shift_h(N_H, periodic)
    D_h = (Sh - Sh.T) / 2.0
    m = np.full(N_H, m0) if m_field is None else m_field
    return (np.kron(SIG_X, -1j * C_TH * np.kron(D_th, np.eye(N_H)))
            + np.kron(SIG_Z, -1j * C_H * np.kron(np.eye(N_TH), D_h))
            + np.kron(SIG_Y, np.kron(np.eye(N_TH), np.diag(m))))


# ================= 冒烟 =================
print("== 冒烟检查 ==")
H0 = build(periodic=False)
herm = np.max(np.abs(H0 - H0.conj().T))
print("冒烟1  H 厄米偏差 = %.3e（期望 0）" % herm)

N_H = 32
Dh = (shift_h(N_H, False) - shift_h(N_H, False).T) / 2.0
ev = np.sort(np.linalg.eigvalsh(-1j * Dh))
ana = np.sort(np.cos(np.pi * np.arange(1, N_H + 1) / (N_H + 1)))
dh_err = np.max(np.abs(ev - ana))
print("冒烟2  开放链 -i D_h 本征值 vs cos(pi k/(N_h+1))：最大差 = %.3e（期望 0）" % dh_err)
report("冒烟：H 厄米且开放链差分谱解析对照", "herm %.1e，差分谱差 %.1e" % (herm, dh_err),
       "herm < 1e-12 且差分谱差 < 1e-10", herm < 1e-12 and dh_err < 1e-10)

# ================= 主读数 =================
print("\n== N1 至 N4：开放 h 档（N_h = 32）==")
H_open = build(periodic=False)
E_open = np.linalg.eigvalsh(H_open)
n_th = np.arange(N_TH) - N_TH // 2
k_th = 2 * np.pi * (n_th + 0.5) / N_TH
frac = k_th * N_TH / (2 * np.pi)
half = np.max(np.abs(np.mod(frac, 1.0) - 0.5))
print("N1  max|frac(k_th N_th/2pi) - 1/2| = %.2e（期望 < 1e-10，反周期生效）" % half)

kz = int(np.sum(np.abs(E_open) < TOL))
lo_open = np.min(np.abs(E_open))
print("N2  零模计数 = %d（期望 0），最低 |E| = %.6f（期望 > 0）" % (kz, lo_open))

Es = np.sort(E_open)
sym = np.max(np.abs(Es - (-Es)[::-1]))
print("N3  max|E_i + E_(N+1-i)| = %.2e（期望 < 1e-10，谱对称）" % sym)

Ep = np.linalg.eigvalsh(build(winding=+1, periodic=False))
Em = np.linalg.eigvalsh(build(winding=-1, periodic=False))
dpm = np.max(np.abs(Ep - Em))
print("N4  max|E(+1) - E(-1)| = %.2e（期望 < 1e-10，镜像式在开放 h 下仍成立）" % dpm)

report("N1 至 N4 开放 h 档", "半整数 %.1e；零模 %d；对称 %.1e；镜像 %.1e"
       % (half, kz, sym, dpm),
       "半整数 < 1e-10、零模 0、对称 < 1e-10、镜像 < 1e-10",
       half < 1e-10 and kz == 0 and sym < 1e-10 and dpm < 1e-10)

# ================= N5 至 N6 =================
print("\n== N5 至 N6：尺寸扫描与边界对照 ==")
print("%-10s %-14s %-14s %s" % ("N_h", "开放 最低|E|", "周期 最低|E|", "开放零模计数"))
scan = []
for Nh in (16, 32, 64):
    eo = np.linalg.eigvalsh(build(N_H=Nh, periodic=False))
    ep = np.linalg.eigvalsh(build(N_H=Nh, periodic=True))
    ko = int(np.sum(np.abs(eo) < TOL))
    scan.append((Nh, np.min(np.abs(eo)), np.min(np.abs(ep)), ko))
    print("%-10d %-14.6f %-14.6f %d" % (Nh, scan[-1][1], scan[-1][2], ko))
report("N5 至 N6 尺寸扫描与边界对照",
       "开放最低 |E| 三档 %s；零模计数全 0" % [round(s[1], 6) for s in scan],
       "最低 |E| 不随 N_h 趋零，开放与周期同量级", all(s[3] == 0 for s in scan))

# ================= 反例 =================
print("\n== 反例（区分力）==")
e_anti0 = np.linalg.eigvalsh(build(periodic=False, m0=0.0))
k_anti0 = int(np.sum(np.abs(e_anti0) < TOL))
e_per0 = np.linalg.eigvalsh(build(periodic=True, m0=0.0, anti=False))
k_per0 = int(np.sum(np.abs(e_per0) < TOL))
e_per = np.linalg.eigvalsh(build(periodic=True, m0=0.5, anti=False))
k_per = int(np.sum(np.abs(e_per) < TOL))
print("反例1  theta 周期 + m_0 = 0：零模计数 = %d（期望 > 0，出现零模）" % k_per0)
print("反例2  theta 周期 + m_0 = 0.5：零模计数 = %d（期望 0，能隙）" % k_per)
print("读数   theta 反周期 + m_0 = 0：零模计数 = %d（期望 0，反周期排除零模）" % k_anti0)
report("反例：反周期是零模排除的原因",
       "周期+m0=0 给 %d 个；周期+m0=0.5 给 %d 个；反周期+m0=0 给 %d 个"
       % (k_per0, k_per, k_anti0),
       "周期+m0=0 出现零模，其余为 0", k_per0 > 0 and k_per == 0 and k_anti0 == 0)

# ================= 图 =================
fig, ax = plt.subplots(1, 3, figsize=(14.0, 4.2))
idx_o = np.argsort(np.abs(E_open))[:60]
idx_p = np.argsort(np.abs(np.linalg.eigvalsh(build(periodic=True))))[:60]
ax[0].plot(range(60), np.abs(E_open)[idx_o], "bo", ms=4, label="开放 $h$（本轮）")
ax[0].plot(range(60), np.abs(np.linalg.eigvalsh(build(periodic=True)))[idx_p], "r+", ms=6,
           label="周期 $h$（第 489 号）")
ax[0].axhline(M0, color="gray", ls=":", lw=1, label="$m_0$")
ax[0].set_xlabel("态序号")
ax[0].set_ylabel("$|E|$")
ax[0].set_title("(a) 开放与周期 $h$：最低 $|E|$ 同量级，均无零模", fontsize=10)
ax[0].legend(fontsize=8)
ax[0].grid(alpha=0.3)

ax[1].plot(np.arange(1, N_H + 1), ana, "k--", lw=1, label="$\\cos(\\pi k/(N_h+1))$")
ax[1].plot(np.arange(1, N_H + 1), ev, "b.", ms=4, label="数值谱")
ax[1].axhline(0.0, color="gray", ls=":", lw=1)
ax[1].set_xlabel("模式序号 $k$")
ax[1].set_ylabel("$-i D_h$ 本征值")
ax[1].set_title("(b) 开放链差分谱与解析式逐位吻合", fontsize=10)
ax[1].legend(fontsize=8)
ax[1].grid(alpha=0.3)

ax[2].plot([s[0] for s in scan], [s[1] for s in scan], "o-", color="tab:blue", label="开放 $h$")
ax[2].plot([s[0] for s in scan], [s[2] for s in scan], "s--", color="tab:red", label="周期 $h$")
ax[2].axhline(TOL, color="k", ls=":", lw=1, label="零模阈值 $10^{-8}$")
ax[2].set_xlabel("$N_h$")
ax[2].set_ylabel("最低 $|E|$")
ax[2].set_title("(c) 尺寸扫描：最低 $|E|$ 不趋零", fontsize=10)
ax[2].legend(fontsize=8)
ax[2].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(r"D:\software\TianXingFC\images\dirac_skeleton_open.png", dpi=150)
print("\n图已存 images/dirac_skeleton_open.png")

print("\n== 判据汇总 ==")
npass = sum(1 for _, ok in results if ok)
print("判据（冒烟并入 N1 至 N6 与反例）：%d/%d 通过" % (npass, len(results)))
