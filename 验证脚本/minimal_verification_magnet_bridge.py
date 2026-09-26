# -*- coding: utf-8 -*-
"""第 448 条：磁铁机制的定量桥梁（均匀区无源、变化处有源、无磁单极）。

模型：自旋序参量场 s(n)（单位矢量）锚定基准取向的链路错配 A_a lP = kappa * s.a，
离散曲率 F_ab = d_a(A_b lP) - d_b(A_a lP)，磁场取对偶 B = kappa * curl s。

判据：
  E1 均匀自旋序：B 处处为零（机器零）。
  E2 畴壁：B 非零且定位于自旋方向变化处，均匀区两侧仍为零。
  E3 无磁单极：任意随机自旋配置下 div B 处处为零（机器零）。

容差预算：三条判据均为离散恒等式或零结果，偏差仅双精度舍入（1e-16 量级），
容差取 1e-13；E2 的非零判定阈值按场幅量级取 1e-6（kappa = 1、L = 16 时 |B| 量级 1e-1）。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG = r"D:\software\TianXingFC\images\jz1_magnet_bridge.png"
kappa = 1.0
L = 16
rng = np.random.default_rng(20260927)

def A_of(s):
    return kappa * s

def F_ab(A, a, b):
    return (np.roll(A[..., b], -1, axis=a) - A[..., b]) - (np.roll(A[..., a], -1, axis=b) - A[..., a])

def B_of(A):
    B = np.zeros_like(A)
    B[..., 0] = F_ab(A, 1, 2)
    B[..., 1] = F_ab(A, 2, 0)
    B[..., 2] = F_ab(A, 0, 1)
    return B

def div_B(B):
    d = np.zeros(B.shape[:-1])
    for i in range(3):
        d = d + (np.roll(B[..., i], -1, axis=i) - B[..., i])
    return d

# ---------------- E1：均匀自旋序 ----------------
s_uniform = np.zeros((L, L, L, 3)); s_uniform[..., 0] = 1.0
B_uniform = B_of(A_of(s_uniform))
e1 = float(np.max(np.abs(B_uniform)))

# ---------------- E2：畴壁 ----------------
x = np.arange(L)
s_wall = np.zeros((L, L, L, 3))
prof = np.clip((7.5 - x) / 3.0, -1.0, 1.0)   # 分段线性：两侧均匀区严格为 ±1
s_wall[..., 0] = prof[:, None, None]
s_wall[..., 2] = np.sqrt(np.maximum(0.0, 1.0 - prof[:, None, None] ** 2))
B_wall = B_of(A_of(s_wall))
Bmag_wall = np.linalg.norm(B_wall, axis=-1)
e2_max = float(np.max(Bmag_wall))
# 均匀区（距畴壁中心 4 格以上）的场应仍为零
uniform_slice = np.r_[0:4, L - 4:L]   # 分段剖面下这两段严格均匀
e2_uniform = float(np.max(Bmag_wall[uniform_slice, :, :]))

# ---------------- E3：无磁单极 ----------------
s_rand = rng.standard_normal((L, L, L, 3))
s_rand = s_rand / np.linalg.norm(s_rand, axis=-1, keepdims=True)
B_rand = B_of(A_of(s_rand))
e3 = float(np.max(np.abs(div_B(B_rand))))

print("E1 均匀自旋序（期望零）")
print("max|B| = %.3e" % e1)
print()
print("E2 畴壁（源定位于变化处）")
print("max|B| = %.6f" % e2_max)
print("均匀区 max|B| = %.3e（期望零）" % e2_uniform)
print()
print("E3 无磁单极（随机自旋配置，期望零）")
print("max|div B| = %.3e" % e3)
print()
flag1 = bool(e1 < 1e-13)
flag2 = bool(e2_max > 1e-6 and e2_uniform < 1e-13)
flag3 = bool(e3 < 1e-13)
print("判据 E1（均匀区无源）：%s" % ("通过" if flag1 else "未通过"))
print("判据 E2（变化处有源、均匀区无源）：%s" % ("通过" if flag2 else "未通过"))
print("判据 E3（无磁单极）：%s" % ("通过" if flag3 else "未通过"))
print("判据总计：%d/3" % (int(flag1) + int(flag2) + int(flag3)))

# ---------------- 出图 ----------------
fig, axp = plt.subplots(2, 2, figsize=(11, 8))
axp[0, 0].imshow(Bmag_wall[:, :, L // 2].T, origin="lower", cmap="inferno", aspect="auto")
axp[0, 0].set_title("E2 |B| in domain wall (localized)")
axp[0, 0].set_xlabel("x (wall normal)"); axp[0, 0].set_ylabel("y")
axp[0, 1].imshow(prof[:, None] * np.ones((L, L))[:, :], origin="lower", cmap="RdBu", aspect="auto")
axp[0, 1].set_title("E2 spin profile s_x(x)")
axp[0, 1].set_xlabel("x"); axp[0, 1].set_ylabel("y")
axp[1, 0].semilogy(np.abs(Bmag_wall).ravel() + 1e-18, "b.", ms=2)
axp[1, 0].axhline(1e-13, color="r", ls="--", lw=1.0, label="zero threshold")
axp[1, 0].set_title("E1/E2 |B| values across sites (log)")
axp[1, 0].set_xlabel("site index"); axp[1, 0].set_ylabel("|B|")
axp[1, 0].legend(fontsize=7)
axp[1, 1].semilogy(np.abs(div_B(B_rand)).ravel() + 1e-18, "g.", ms=2)
axp[1, 1].set_title("E3 |div B| for random spins (log)")
axp[1, 1].set_xlabel("site index"); axp[1, 1].set_ylabel("|div B|")
plt.tight_layout()
plt.savefig(IMG, dpi=130)
print("图已保存：%s" % IMG)
