# -*- coding: utf-8 -*-
"""第 442 条：紫外截断的层属归属（存在性上限与可分辨性上限分属两层）。

判据：
  V1 两个上限的数值关系：演化层的谱支撑上限 omega_BZ = 2 sqrt3 / tP（布里渊体对角角点），
     读数层的可分辨性上限 omega_N = omegaP/2 = pi / tP；前者大于后者，比值 2 sqrt3 / pi。
  V2 两上限的机制来源不同：omega_BZ 由色散谱的支撑集给出（S 的最大值 3 出现在角点），
     omega_N 由采样率给出（每摆程一次事件），两者分属演化层与读数层。
  V3 区间模式的地位：omega 落在 (omega_N, omega_BZ] 的模式存在但不可分辨，
     其时间采样序列与混叠频率 omegaP - omega 的序列逐点相同（机器零）。

容差预算：omega_BZ 由 61^3 网格扫描给出，角点被网格包含故为精确值，容差取 1e-12；
V3 的序列比对为三角恒等式，偏差仅舍入，容差取 1e-10。区间上下界的比较为精确不等式。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG = r"D:\software\TianXingFC\images\jz1_uv_cutoff_layers.png"
lP, tP = 1.0, 1.0
omegaP = 2.0 * np.pi / tP
omegaN = omegaP / 2.0

def S_of(kvec):
    return float(np.sum(np.sin(np.asarray(kvec) * lP / 2.0) ** 2))

def omega_cont(kvec):
    return (2.0 / tP) * np.sqrt(S_of(kvec))

# ---------------- V1：谱支撑上限 ----------------
m = 61
ax = np.linspace(-np.pi / lP, np.pi / lP, m)
KX, KY, KZ = np.meshgrid(ax, ax, ax, indexing="ij")
Sg = np.sin(KX * lP / 2.0) ** 2 + np.sin(KY * lP / 2.0) ** 2 + np.sin(KZ * lP / 2.0) ** 2
wmax_num = float((2.0 / tP) * np.sqrt(Sg.max()))
wmax_ana = 2.0 * np.sqrt(3.0) / tP
ratio = wmax_ana / omegaN

# 沿体对角方向的色散
ks_body = np.linspace(0.0, np.pi * np.sqrt(3.0), 400)
w_body = [(2.0 / tP) * np.sqrt(3.0 * np.sin(k / (2.0 * np.sqrt(3.0))) ** 2) for k in ks_body]

# ---------------- V3：区间模式的混叠 ----------------
w_bz = wmax_ana
w_ali = omegaP - w_bz
s = np.arange(0, 200)
seq_bz = np.cos(w_bz * s * tP)
seq_ali = np.cos(w_ali * s * tP)
seq_diff = float(np.max(np.abs(seq_bz - seq_ali)))

# 采样点重建检验：用两频率各自重建 s 与 s+1 之间的值，比较采样点一致性
grid_s = np.arange(0, 12)
chk = float(np.max(np.abs(np.cos(w_bz * grid_s * tP) - np.cos(w_ali * grid_s * tP))))

print("V1 两个上限（单位 1/tP）")
print("演化层谱支撑上限 omega_BZ = %.9f（解析 2 sqrt3 = %.9f）" % (wmax_num, wmax_ana))
print("读数层可分辨上限 omega_N  = %.9f（pi = %.9f）" % (omegaN, np.pi))
print("比值 omega_BZ / omega_N   = %.9f（解析 2 sqrt3 / pi = %.9f）" % (ratio, 2.0 * np.sqrt(3.0) / np.pi))
print("区间宽度 omega_BZ - omega_N = %.9f / tP" % (wmax_ana - omegaN))
print()
print("V2 两上限的来源")
print("omega_BZ 来源：色散谱支撑集，S 的最大值 %.6f 出现在布里渊角点（分量各 pi/lP）" % Sg.max())
print("omega_N  来源：采样率 1/tP 给出的尼奎斯特界限（每摆程一次事件），与色散谱无关")
print()
print("V3 区间模式 (omega_N, omega_BZ] 的地位")
print("取 omega = omega_BZ = %.6f / tP，混叠频率 omegaP - omega = %.6f / tP" % (w_bz, w_ali))
print("混叠频率落在可分辨区（< omega_N = %.6f）：%s" % (omegaN, w_ali < omegaN))
print("两模式在采样点上的序列最大差 = %.3e（期望机器零）" % seq_diff)
print()
flag1 = bool(abs(wmax_num - wmax_ana) < 1e-12 and wmax_ana > omegaN
             and abs(ratio - 2.0 * np.sqrt(3.0) / np.pi) < 1e-12)
flag2 = bool(abs(Sg.max() - 3.0) < 1e-12)
flag3 = bool(seq_diff < 1e-10 and w_ali < omegaN)
print("判据 V1（两上限数值关系：谱支撑大于可分辨界限）：%s" % ("通过" if flag1 else "未通过"))
print("判据 V2（谱支撑上限来自角点 S = 3，与采样率无关）：%s" % ("通过" if flag2 else "未通过"))
print("判据 V3（区间模式存在但不可分辨：混叠序列一致）：%s" % ("通过" if flag3 else "未通过"))
print("判据总计：%d/3" % (int(flag1) + int(flag2) + int(flag3)))

# ---------------- 出图 ----------------
fig, axp = plt.subplots(2, 2, figsize=(11, 8))
axp[0, 0].plot(ks_body, w_body, "b-", lw=1.2, label="spectral support (body diagonal)")
axp[0, 0].axhline(omegaN, color="r", ls="--", lw=1.1, label="Nyquist (readout layer)")
axp[0, 0].axhline(wmax_ana, color="g", ls=":", lw=1.1, label="Brillouin max (evolution layer)")
axp[0, 0].fill_between([0, ks_body[-1]], omegaN, wmax_ana, color="orange", alpha=0.18,
                       label="exists but unresolvable")
axp[0, 0].set_xlabel("k (body diagonal)"); axp[0, 0].set_ylabel("omega")
axp[0, 0].set_title("V1 two cutoffs"); axp[0, 0].legend(fontsize=7)

axp[0, 1].hist(((2.0 / tP) * np.sqrt(Sg)).ravel(), bins=60, color="steelblue")
axp[0, 1].axvline(omegaN, color="r", ls="--", lw=1.2, label="Nyquist")
axp[0, 1].axvline(wmax_ana, color="g", ls=":", lw=1.2, label="Brillouin max")
axp[0, 1].set_xlabel("omega"); axp[0, 1].set_ylabel("count (61^3 grid)")
axp[0, 1].set_title("V2 spectrum support"); axp[0, 1].legend(fontsize=7)

tt = np.arange(0, 12) * tP
axp[1, 0].plot(tt, np.cos(w_bz * tt), "go", ms=5, label="mode at Brillouin max")
axp[1, 0].plot(tt, np.cos(w_ali * tt), "r+", ms=9, label="aliased frequency")
axp[1, 0].set_xlabel("t (sample points)"); axp[1, 0].set_ylabel("amplitude")
axp[1, 0].set_title("V3 identical on sample points"); axp[1, 0].legend(fontsize=7)

axp[1, 1].barh([0, 1], [omegaN, wmax_ana - omegaN], color=["steelblue", "orange"])
axp[1, 1].set_yticks([0, 1]); axp[1, 1].set_yticklabels(["resolvable", "exists, unresolvable"], fontsize=8)
axp[1, 1].set_xlabel("omega (1/tP)")
axp[1, 1].set_title("layer assignment of the two bounds")
plt.tight_layout()
plt.savefig(IMG, dpi=130)
print("图已保存：%s" % IMG)
