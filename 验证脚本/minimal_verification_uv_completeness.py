# -*- coding: utf-8 -*-
"""第 441 条：紫外完备性判据（时间层属的硬分界）。

判据：
  U1 递推路线的无解区域：色散 sin^2(omega tP/2) = S(k) 要求 sqrt(S) <= 1；
     三维布里渊区中 S > 1 的区域存在且占比可观，斜向临界 k* 与解析值一致。
  U2 递推路线的群速度发散：斜向 v_g/c 在 S -> 1^- 时发散（arcsin 导数奇性）；
     轴向 v_g/c = 1 恒为常数（一维 arcsin o sin 为恒等）。
  U3 连续时间路线全域良性：omega = 2 sqrt(S)/tP 对一切 k 有实数解，群速度有上界（<= c）。

容差预算：临界 k* 的解析值为 k*_face = pi/sqrt2、k*_body = 2 sqrt3 arcsin(1/sqrt3)，
数值由二分法求 S(k) - 1 = 0 得到，二分终止容差 1e-12；群速度由中心差分给出，
步长 1e-6 时截断误差 O(h^2) = 1e-12，故 U2/U3 的绝对值比较容差取 1e-6。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG = r"D:\software\TianXingFC\images\jz1_uv_completeness.png"
lP, tP = 1.0, 1.0
c = lP / tP

def S_of(kvec):
    k = np.asarray(kvec, dtype=float)
    return float(np.sum(np.sin(k * lP / 2.0) ** 2))

def omega_rec(kvec):
    S = S_of(kvec)
    return (2.0 / tP) * np.arcsin(np.sqrt(S)) if S <= 1.0 else np.nan

def omega_cont(kvec):
    return (2.0 / tP) * np.sqrt(S_of(kvec))

# ---------------- U1：无解区域 ----------------
n = 96
ax = np.linspace(-np.pi / lP, np.pi / lP, n)
KX, KY, KZ = np.meshgrid(ax, ax, np.arange(0.0, 0.001, 1.0), indexing="ij")
Sgrid = np.sin(KX * lP / 2) ** 2 + np.sin(KY * lP / 2) ** 2 + np.sin(KZ * lP / 2) ** 2
frac_nosol = float(np.mean(Sgrid > 1.0))

def kcrit(u):
    """沿单位方向 u 求 S(k u) = 1 的临界 k（二分）。"""
    lo, hi = 1e-9, np.pi
    f = lambda k: S_of(np.asarray(u) * k) - 1.0
    if f(hi) < 0:
        return np.nan
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

u_face = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
u_body = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
k_face, k_body = kcrit(u_face), kcrit(u_body)
k_face_ana = np.pi / np.sqrt(2.0)
k_body_ana = 2.0 * np.sqrt(3.0) * np.arcsin(1.0 / np.sqrt(3.0))

# ---------------- U2/U3：群速度 ----------------
def vg(fun, u, k, h=1e-6):
    return (fun(np.asarray(u) * (k + h)) - fun(np.asarray(u) * (k - h))) / (2.0 * h)

ks_face = np.array([0.3, 0.6, 1.0, 1.4, 1.8, 2.0, 2.2, k_face - 1e-3])
rows_v = []
for k in ks_face:
    rows_v.append((k, vg(omega_rec, u_face, k) / c, vg(omega_cont, u_face, k) / c))

ks_axis = np.array([1e-3, 0.5, 1.0, 2.0, np.pi - 1e-3])   # 留出差分步长余量，避免越过布里渊边界
rows_axis = [(k, vg(omega_rec, np.array([1.0, 0.0, 0.0]), k) / c,
                 vg(omega_cont, np.array([1.0, 0.0, 0.0]), k) / c) for k in ks_axis]

# 连续路线的群速度上界（球面方向扫描）
rng = np.random.default_rng(20260926)
vg_max = 0.0
for _ in range(400):
    u = rng.standard_normal(3); u = u / np.linalg.norm(u)
    k = rng.uniform(0.05, np.pi)
    if S_of(u * k) <= 0.0:
        continue
    v = abs(vg(omega_cont, u, k)) / c
    if np.isfinite(v):
        vg_max = max(vg_max, v)

print("U1 递推路线的无解区域（三维布里渊区，k_z = 0 切片统计 %d x %d）" % (n, n))
print("S > 1 的网格占比 = %.4f" % frac_nosol)
print("面对角临界 k* = %.9f（解析 %.9f）" % (k_face, k_face_ana))
print("体对角临界 k* = %.9f（解析 %.9f）" % (k_body, k_body_ana))
print()
print("U2/U3 斜向（面对角）群速度对比")
print("k       递推 v_g/c     连续 v_g/c")
for r in rows_v:
    print("%.4f  %12.4f  %12.6f" % r)
print()
print("轴向群速度（递推应为常数 1，连续应在边界趋于 0）")
print("k       递推 v_g/c     连续 v_g/c")
for r in rows_axis:
    print("%.6f  %12.6f  %12.6f" % r)
print()
print("连续路线群速度上界（400 个随机方向与 k）：%.6f c" % vg_max)
print()
flag1 = bool(frac_nosol > 0.10 and abs(k_face - k_face_ana) < 1e-9 and abs(k_body - k_body_ana) < 1e-9)
flag2 = bool(rows_v[-1][1] > 10.0 and all(abs(r[1] - 1.0) < 1e-6 for r in rows_axis))
flag3 = bool(np.isfinite(vg_max) and vg_max <= 1.0 + 1e-6 and abs(rows_axis[-1][2]) < 1e-3)
print("判据 U1（递推路线无解区存在且临界值与解析一致）：%s" % ("通过" if flag1 else "未通过"))
print("判据 U2（递推路线斜向群速度发散、轴向恒为 1）：%s" % ("通过" if flag2 else "未通过"))
print("判据 U3（连续路线全域有解、群速度有上界、边界趋于零）：%s" % ("通过" if flag3 else "未通过"))
print("判据总计：%d/3" % (int(flag1) + int(flag2) + int(flag3)))

# ---------------- 出图 ----------------
fig, axp = plt.subplots(2, 2, figsize=(11, 8))
kx = np.linspace(-np.pi, np.pi, 400)
KX2, KY2 = np.meshgrid(kx, kx, indexing="ij")
S2 = np.sin(KX2 / 2) ** 2 + np.sin(KY2 / 2) ** 2
cs = axp[0, 0].contour(KX2, KY2, S2, levels=[1.0], colors="k", linewidths=1.2)
axp[0, 0].contourf(KX2, KY2, S2, levels=[1.0, S2.max() + 0.01], colors=["indianred"], alpha=0.35)
axp[0, 0].plot([0, 0], [-np.pi, np.pi], "b--", lw=0.9)
axp[0, 0].plot([-np.pi, np.pi], [0, 0], "b--", lw=0.9)
axp[0, 0].set_xlabel("k_x"); axp[0, 0].set_ylabel("k_y")
axp[0, 0].set_title("U1 no-real-solution region (red, S>1), k_z=0")

kk = np.linspace(1e-3, k_face - 1e-3, 400)
vg_r = [abs(vg(omega_rec, u_face, k)) / c for k in kk]
vg_c = [abs(vg(omega_cont, u_face, k)) / c for k in kk]
axp[0, 1].semilogy(kk, vg_r, "r-", lw=1.2, label="recursive (diverges)")
axp[0, 1].semilogy(kk, vg_c, "b-", lw=1.2, label="continuous (bounded)")
axp[0, 1].axvline(k_face, color="k", ls="--", lw=1.0, label="k* (S=1)")
axp[0, 1].set_xlabel("k (face diagonal)"); axp[0, 1].set_ylabel("|v_g|/c (log)")
axp[0, 1].set_title("U2/U3 group velocity")
axp[0, 1].legend(fontsize=7)

ka = np.linspace(1e-3, np.pi - 1e-3, 400)
axp[1, 0].plot(ka, [(2.0 / tP) * np.arcsin(np.sin(k / 2)) for k in ka], "r-", lw=1.2, label="recursive: k (exact linear)")
axp[1, 0].plot(ka, [(2.0 / tP) * np.sin(k / 2) for k in ka], "b-", lw=1.2, label="continuous: 2 sin(k/2)")
axp[1, 0].plot(ka, ka, "k--", lw=0.8, label="ck")
axp[1, 0].set_xlabel("k (axis)"); axp[1, 0].set_ylabel("omega")
axp[1, 0].set_title("axial dispersion (both benign)")
axp[1, 0].legend(fontsize=7)

axp[1, 1].hist((np.sin(KX / 2) ** 2 + np.sin(KY / 2) ** 2).ravel(), bins=60, color="steelblue")
axp[1, 1].axvline(1.0, color="r", ls="--", lw=1.2, label="S=1 (solvability bound)")
axp[1, 1].set_xlabel("S on k_z=0 plane"); axp[1, 1].set_ylabel("count")
axp[1, 1].set_title("U1 S distribution"); axp[1, 1].legend(fontsize=7)
plt.tight_layout()
plt.savefig(IMG, dpi=130)
print("图已保存：%s" % IMG)
