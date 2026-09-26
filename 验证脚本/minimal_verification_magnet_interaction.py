# -*- coding: utf-8 -*-
"""第 449 条：同斥异吸的相互作用能判据（磁铁机制余项）。

模型沿用第 448 条：自旋序参量 s(n) 锚定链路错配 A_a lP = kappa s.a，磁场 B = kappa rot s，
磁场能量取线性区 E = (1/2) sum |B|^2。两块磁畴（自旋沿 z 有序的方块）间距 d，
相互作用能 E_int(d) = E_total(d) - 2 E_single。

判据：
  F1 平行排列（两块磁化同向）：E_int > 0，且随 d 递减（排斥）。
  F2 反平行排列（磁化反向）：E_int < 0，且绝对值随 d 递减（吸引）。
  F3 叠加分解：E_int 等于两块磁场点积的积分（int B1.B2），验证相互作用能就是场的叠加交叉项。

容差预算：F1 与 F2 的符号判据为不等号（量级 1e-8 以上，远离机器精度）；
F3 为叠加恒等式，偏差仅舍入，容差 1e-12。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG = r"D:\software\TianXingFC\images\jz1_magnet_interaction.png"
kappa = 1.0
L = 32
rng = np.random.default_rng(20260927)

def F_ab(A, a, b):
    return (np.roll(A[..., b], -1, axis=a) - A[..., b]) - (np.roll(A[..., a], -1, axis=b) - A[..., a])

def bound_current(s):
    """束缚电流密度 j = curl(kappa s)：第 448 条的结果在此读作源。"""
    A = kappa * s
    return np.stack([F_ab(A, 1, 2), F_ab(A, 2, 0), F_ab(A, 0, 1)], axis=-1)

def B_of(s):
    """静磁求解：curl B = j 且 div B = 0（库仑规范下解 A_hat = j_hat / lambda）。"""
    j = bound_current(s)
    J = np.fft.fftn(j, axes=(0, 1, 2))
    k1 = 2.0 * np.pi * np.fft.fftfreq(L)
    KX, KY, KZ = np.meshgrid(k1, k1, k1, indexing="ij")
    lam = 4.0 * np.sin(KX / 2.0) ** 2 + 4.0 * np.sin(KY / 2.0) ** 2 + 4.0 * np.sin(KZ / 2.0) ** 2
    lam[0, 0, 0] = 1.0
    A_hat = J / lam[..., None]
    ikx, iky, ikz = 1j * np.sin(KX), 1j * np.sin(KY), 1j * np.sin(KZ)
    Bx = iky * A_hat[..., 2] - ikz * A_hat[..., 1]
    By = ikz * A_hat[..., 0] - ikx * A_hat[..., 2]
    Bz = ikx * A_hat[..., 1] - iky * A_hat[..., 0]
    B = np.stack([Bx, By, Bz], axis=-1)
    return np.real(np.fft.ifftn(B, axes=(0, 1, 2)))

def energy(B):
    return 0.5 * float(np.sum(B ** 2))

def block_mask(com, half):
    ix = np.arange(L)
    X, Y, Z = np.meshgrid(ix, ix, ix, indexing="ij")
    return (np.abs(X - com[0]) <= half) & (np.abs(Y - com[1]) <= half) & (np.abs(Z - com[2]) <= half)

def spin_config(com1, com2, half, sign2):
    s = np.zeros((L, L, L, 3))
    m1 = block_mask(com1, half)
    m2 = block_mask(com2, half)
    s[m1, 2] = 1.0
    s[m2, 2] = sign2 * 1.0
    return s, m1, m2

half = 4
d_list = [10, 11, 12, 13, 14]   # 中心距须大于 2 x half，且不超过 L/2 以免周期卷绕
rows = []
E1_single = None
for d in d_list:
    c1 = (L // 2 - d // 2, L // 2, L // 2)
    c2 = (L // 2 + d - d // 2, L // 2, L // 2)
    res = {}
    for tag, sign2 in (("par", 1.0), ("anti", -1.0)):
        s, m1, m2 = spin_config(c1, c2, half, sign2)
        B = B_of(s)
        E_tot = energy(B)
        if E1_single is None:
            s1, _, _ = spin_config(c1, (-100, -100, -100), half, 1.0)
            E1_single = energy(B_of(s1))
        E_int = E_tot - 2.0 * E1_single
        # 叠加交叉项
        sA, _, _ = spin_config(c1, (-100, -100, -100), half, 1.0)
        sB, _, _ = spin_config((-100, -100, -100), c2, half, 1.0)
        if sign2 < 0:
            sB = -sB
        cross = float(np.sum(B_of(sA) * B_of(sB)))
        res[tag] = (E_int, cross)
    rows.append((d, res["par"][0], res["anti"][0], res["par"][1], res["anti"][1]))

print("F1/F2 相互作用能（E_int = E_total - 2 E_single）")
print("d   平行 E_int      反平行 E_int    平行 cross     反平行 cross")
for r_ in rows:
    print("%2d  %+.6e  %+.6e  %+.6e  %+.6e" % r_)
print()
f1 = bool(all(r[1] > 0 for r in rows) and rows[0][1] > rows[-1][1])
f2 = bool(all(r[2] < 0 for r in rows) and abs(rows[0][2]) > abs(rows[-1][2]))
f3 = bool(all(abs(r[1] - r[3]) / max(abs(r[3]), 1e-30) < 1e-12 and abs(r[2] - r[4]) / max(abs(r[4]), 1e-30) < 1e-12 for r in rows))
print("判据 F1（平行排斥：E_int > 0 且随 d 递减）：%s" % ("通过" if f1 else "未通过"))
print("判据 F2（反平行吸引：E_int < 0 且绝对值随 d 递减）：%s" % ("通过" if f2 else "未通过"))
print("判据 F3（相互作用能等于叠加交叉项）：%s" % ("通过" if f3 else "未通过"))
print("判据总计：%d/3" % (int(f1) + int(f2) + int(f3)))

fig, axp = plt.subplots(1, 2, figsize=(11, 4))
dd = [r[0] for r in rows]
axp[0].plot(dd, [r[1] for r in rows], "ro-", label="parallel (repulsive)")
axp[0].plot(dd, [r[2] for r in rows], "bs-", label="antiparallel (attractive)")
axp[0].axhline(0.0, color="k", lw=0.8)
axp[0].set_xlabel("separation d"); axp[0].set_ylabel("E_int")
axp[0].set_title("F1/F2 interaction energy vs separation")
axp[0].legend(fontsize=7)
axp[1].semilogy(dd, [abs(r[1]) for r in rows], "ro-", label="|parallel|")
axp[1].semilogy(dd, [abs(r[2]) for r in rows], "bs-", label="|antiparallel|")
axp[1].set_xlabel("d"); axp[1].set_ylabel("|E_int| (log)")
axp[1].set_title("decay with separation")
axp[1].legend(fontsize=7)
plt.tight_layout()
plt.savefig(IMG, dpi=130)
print("图已保存：%s" % IMG)
