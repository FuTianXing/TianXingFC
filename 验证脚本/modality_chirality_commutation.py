# -*- coding: utf-8 -*-
"""判据第 661 号：模态-时空接口的对易性审计（第 494 号其五「反对易」结论复核）。

问题：第 494 号其五写「$\\Gamma_5$ 为 $I_2\\otimes\\Gamma_5^{sp}$，与模态因子 $\\kappa\\sigma_z$
反对易，全空间反对易子实测为 $2\\kappa=0.600$」。该结论被用作「模态手性与时空手性不共享
本征态」的依据，并传播进正稿摘要、§3.1 与多份注记。

但该轮构造是 $H = \\kappa\\sigma_z\\otimes I + I\\otimes H_{sp}$，模态因子与时空因子分属
张量积的**两个不同因子**，而 $\\Gamma_5 = I_2\\otimes\\Gamma_5^{sp}$ 在模态因子上取单位阵。
不同因子上的算符对易，故 $[\\Gamma_5, \\sigma_z\\otimes I] = 0$。此时反对易子
$\\{\\Gamma_5, \\sigma_z\\otimes I\\} = 2\\Gamma_5(\\sigma_z\\otimes I)$ 非零，其谱范数恰为 $2$；
乘 $\\kappa$ 后为 $2\\kappa$，与第 494 号实测值吻合。也就是说，第 494 号测到的 $2\\kappa$
正是**对易**的贡献，却被读成了反对易。

本轮以 8 分量构造逐条复核。

先立的期望（写实现之前）：
  冒烟    gamma 基签名 (+,-,-,-)，Gamma_5 厄米、平方为 I、与四个 gamma 反对易。
  C1     [Gamma_5, M] = 0（M = sigma_z (x) I_4 为模态因子）——对易。
  C2     {Gamma_5, M} 的谱范数 = 2（非零，但这是对易的推论，不是反对易）。
  C3     {Gamma_5, H_sp} = 0 —— 时空部分确实反对易。
  C4     [Gamma_5, H_sp] 的谱范数 = 2 m（非零）。
  C5     M 与 Gamma_5 可同时对角化（联合本征基存在），四扇区各二重。
  C6     {H, Gamma_5} 的谱范数 = 2 kappa（复现第 494 号实测值）。
  C7     [H, Gamma_5] 的谱范数 = 2 m（非零）。
  区分力 以 gamma^0 为对照：{Gamma_5, gamma^0} = 0、[Gamma_5, gamma^0] 范数 = 2，
         即真正的反对易对，用来确认判据能区分两种关系。
  结论   第 494 号其五的「反对易」系对易子与反对易子混淆；实际关系为「与模态因子对易、
         与时空 Dirac 反对易」，故「模态手性与时空手性不共享本征态」须更正为共享。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
KAPPA, M0 = 0.3, 0.5

G0 = np.kron(SZ, I2)
G1 = np.kron(1j * SY, SX)
G2 = np.kron(1j * SY, SY)
G3 = np.kron(1j * SY, SZ)
G5_SP = np.kron(SX, I2)

G5 = np.kron(I2, G5_SP)              # I_2 (x) Gamma_5^sp
M_MOD = np.kron(SZ, np.eye(4))       # 模态因子
H_SP = M0 * G0                       # 时空部分（最小：质量项，k = 0）
H = KAPPA * M_MOD + np.kron(I2, H_SP)

results = []


def nrm(A):
    return float(np.linalg.norm(A, 2))


def report(tag, measured, expected, ok):
    results.append((tag, ok))
    print("[%s] %s" % ("PASS" if ok else "FAIL", tag))
    print("    实测: %s" % measured)
    print("    期望: %s" % expected)


# ================= 冒烟 =================
print("== 冒烟：gamma 基与 Gamma_5 ==")
sig = [round(float(np.real(np.trace(G @ G))) / 4.0, 6) for G in (G0, G1, G2, G3)]
anti_g = max(nrm(G5_SP @ G + G @ G5_SP) for G in (G0, G1, G2, G3))
herm5 = nrm(G5_SP - G5_SP.conj().T)
sq5 = nrm(G5_SP @ G5_SP - np.eye(4))
print("    gamma 签名 = %s（期望 [+1, -1, -1, -1]）" % sig)
print("    Gamma_5^sp 厄米偏差 = %.3e，平方偏差 = %.3e，与四个 gamma 反对易子最大范数 = %.3e"
      % (herm5, sq5, anti_g))
report("冒烟：gamma 基与 Gamma_5 自洽",
       "签名 %s；厄米 %.1e；平方 %.1e；反对易子 %.1e" % (sig, herm5, sq5, anti_g),
       "签名 [+1,-1,-1,-1] 且三者均 < 1e-12",
       sig == [1.0, -1.0, -1.0, -1.0] and sq5 < 1e-12 and anti_g < 1e-12)

# ================= C1 至 C4 =================
print("\n== C1 至 C4：模态因子与时空部分的对易/反对易 ==")
com_mod = nrm(G5 @ M_MOD - M_MOD @ G5)
anti_mod = nrm(G5 @ M_MOD + M_MOD @ G5)
H_SP_FULL = np.kron(I2, H_SP)
com_sp = nrm(G5 @ H_SP_FULL - H_SP_FULL @ G5)
anti_sp = nrm(G5 @ H_SP_FULL + H_SP_FULL @ G5)
print("C1  [Gamma_5, M] 谱范数 = %.3e（期望 0，对易）" % com_mod)
print("C2  {Gamma_5, M} 谱范数 = %.6f（期望 2.000000，非零但属对易的推论）" % anti_mod)
print("C3  {Gamma_5, H_sp} 谱范数 = %.3e（期望 0，时空部分反对易）" % anti_sp)
print("C4  [Gamma_5, H_sp] 谱范数 = %.6f（期望 2m = %.6f，非零）" % (com_sp, 2 * M0))
report("C1 至 C4 模态对易、时空反对易",
       "模态对易子 %.1e、模态反对易子 %.3f；时空反对易子 %.1e、时空对易子 %.3f"
       % (com_mod, anti_mod, anti_sp, com_sp),
       "模态对易子 < 1e-12 且反对易子 = 2；时空反对易子 < 1e-12 且对易子 = 2m",
       com_mod < 1e-12 and abs(anti_mod - 2.0) < 1e-10 and anti_sp < 1e-12
       and abs(com_sp - 2 * M0) < 1e-10)

# ================= C5：联合本征基 =================
print("\n== C5：M 与 Gamma_5 的联合本征基 ==")
ev_m, V_m = np.linalg.eigh(M_MOD)
SECT = {}
for mv in (1.0, -1.0):
    idx = np.where(np.abs(ev_m - mv) < 1e-9)[0]
    B = V_m[:, idx]
    P = B.conj().T @ G5 @ B
    evg = np.linalg.eigvalsh(P)
    for gv in (1.0, -1.0):
        SECT[(mv, gv)] = int(np.sum(np.abs(evg - gv) < 1e-9))
com_joint = nrm(M_MOD @ G5 - G5 @ M_MOD)
KEYS = ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0))
print("    四个联合扇区维数（M, Gamma_5）：(+1,+1)=%d, (+1,-1)=%d, (-1,+1)=%d, (-1,-1)=%d"
      % tuple(SECT[k] for k in KEYS))
report("C5 M 与 Gamma_5 可同时对角化（联合本征基存在）",
       "四扇区维数 %s（总 %d）" % ([SECT[k] for k in KEYS], sum(SECT.values())),
       "四扇区均非空、维数各 2（对易的直接后果，模态与手性可同时标记）",
       com_joint < 1e-12 and all(SECT[k] == 2 for k in KEYS))

# ================= C6 至 C7 =================
print("\n== C6 至 C7：全哈密顿量的反对易子与对易子 ==")
anti_H = nrm(H @ G5 + G5 @ H)
com_H = nrm(H @ G5 - G5 @ H)
print("C6  {H, Gamma_5} 谱范数 = %.6f（期望 2kappa = %.6f，复现第 494 号实测）"
      % (anti_H, 2 * KAPPA))
print("C7  [H, Gamma_5] 谱范数 = %.6f（期望 2m = %.6f，非零）" % (com_H, 2 * M0))
report("C6 至 C7 第 494 号实测值复现",
       "反对易子 %.6f = 2kappa；对易子 %.6f = 2m" % (anti_H, com_H),
       "反对易子 = 2kappa 且对易子 = 2m，两者均非零（既不对易也不反对易）",
       abs(anti_H - 2 * KAPPA) < 1e-10 and abs(com_H - 2 * M0) < 1e-10)

# ================= 区分力对照 =================
print("\n== 区分力对照：真正的反对易对（Gamma_5 与 gamma^0）==")
com_ctrl = nrm(G5_SP @ G0 - G0 @ G5_SP)
anti_ctrl = nrm(G5_SP @ G0 + G0 @ G5_SP)
print("    [Gamma_5, gamma^0] 谱范数 = %.3e（期望 2，非零）" % com_ctrl)
print("    {Gamma_5, gamma^0} 谱范数 = %.3e（期望 0，真反对易）" % anti_ctrl)
report("区分力：判据能分辨对易与反对易",
       "对照组对易子 %.3f、反对易子 %.1e" % (com_ctrl, anti_ctrl),
       "对照组对易子 = 2、反对易子 = 0（真反对易）",
       abs(com_ctrl - 2.0) < 1e-10 and anti_ctrl < 1e-12)

# ================= 图 =================
fig, ax = plt.subplots(1, 3, figsize=(14.0, 4.2))
labels = ["[G5, M]", "{G5, M}", "[G5, Hsp]", "{G5, Hsp}", "[G5, g0]", "{G5, g0}"]
vals = [com_mod, anti_mod, com_sp, anti_sp, com_ctrl, anti_ctrl]
colors = ["tab:blue", "tab:red", "tab:blue", "tab:red", "tab:blue", "tab:red"]
ax[0].bar(range(6), [max(v, 1e-16) for v in vals], color=colors)
ax[0].set_yscale("log")
ax[0].set_xticks(range(6))
ax[0].set_xticklabels(labels, fontsize=8, rotation=30)
ax[0].set_ylabel("谱范数（对数）")
ax[0].set_title("(a) 蓝为对易子、红为反对易子", fontsize=10)
ax[0].axhline(1e-12, color="k", ls=":", lw=1)
ax[0].grid(alpha=0.3, axis="y")

# (b) 四扇区维数
sectors = np.zeros((2, 2))
ev_m2, V_m2 = np.linalg.eigh(M_MOD)
for i, mv in enumerate((1.0, -1.0)):
    idx = np.where(np.abs(ev_m2 - mv) < 1e-9)[0]
    B = V_m2[:, idx]
    P = B.conj().T @ G5 @ B
    evg, Vg = np.linalg.eigh(P)
    for j, gv in enumerate((1.0, -1.0)):
        cnt = 0
        for k in range(len(evg)):
            if abs(evg[k] - gv) < 1e-9:
                v = B @ Vg[:, k]
                cnt += 1
        sectors[i, j] = cnt
im = ax[1].imshow(sectors, cmap="Blues", vmin=0)
for i in range(2):
    for j in range(2):
        ax[1].text(j, i, "%d" % sectors[i, j], ha="center", va="center", fontsize=14)
ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["$\\Gamma_5=+1$", "$\\Gamma_5=-1$"])
ax[1].set_yticks([0, 1]); ax[1].set_yticklabels(["$M=+1$", "$M=-1$"])
ax[1].set_title("(b) 模态与手性的联合扇区维数（各 2）", fontsize=10)

# (c) 能量本征态上的 Gamma_5 期望
ev_h, V_h = np.linalg.eigh(H)
exp_g = np.array([float(np.real(V_h[:, k].conj() @ G5 @ V_h[:, k])) for k in range(len(ev_h))])
ax[2].plot(ev_h, exp_g, "o", ms=4, color="tab:green")
ax[2].axhline(0.0, color="k", ls=":", lw=1)
ax[2].set_xlabel("能量 $E$")
ax[2].set_ylabel("$\\langle\\Gamma_5\\rangle$")
ax[2].set_title("(c) 能量本征态上 $\\langle\\Gamma_5\\rangle$ 全零（非守恒）", fontsize=10)
ax[2].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(r"D:\software\TianXingFC\images\modality_chirality_commutation.png", dpi=150)
print("\n图已存 images/modality_chirality_commutation.png")

print("\n== 判据汇总 ==")
npass = sum(1 for _, ok in results if ok)
print("判据（冒烟、C1 至 C7 与区分力对照）：%d/%d 通过" % (npass, len(results)))
