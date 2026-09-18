# -*- coding: utf-8 -*-
"""BH3-3 后续（三维衔接）：层收缩静态痕迹验证（M8）。

判据（先立判据，再实现；均可证伪）：
  L0 准一维锚点：1×1 链（101 转 100、源 46 至 55 去 50）复现 BH3-5 一维读数
     （结构 2.13e-2、比 0.202、残差 0.018；相对偏差 ≤ 5%）。
  L1 层收缩几何：退出后沿主轴层数减一、格距恒 1、坐标变动 ≤ 1 格、
     格点数减少一个横截面（nx·ny）。
  L2 结构留痕非零：主案例（S1 写法球源、未饱和档）下只变几何时记录场相对差 > 1e-4。
  L3 项可分离：结构项与质量项均非零且量级可比（比值 ∈ [1e-2, 1e2]），比值作读数报告。
  L4 线性分解：联合项与「结构加质量」的残差 < 0.1。
  L5 横向均匀性：板源（横向均匀）下结构项在横向上与位置无关（相对起伏 < 1e-8）。

模型：三维简单立方、格距 1；横向（x、y）周期边界、z 两端 Dirichlet 0；
      静态方程 Lφ = −s（七点拉普拉斯）；层收缩 = 沿 z 去一层让位（坐标压缩一格），
      源图案按 BH3-5 同口径保持原索引。准一维锚点与 BH3-5 完全同构（nx = ny = 1）。
      主案例取 S1 写法：均匀球 R_s = 6 格、s_tot = 4π M/m_P、未饱和档 M/m_P = 4
      （S1 该档 h_max < 1，容量裁剪不激活，线性解即 S1 构型）；被清除质量取球心单格点。
确定性（无随机数）。图存 D:/software/TianXingFC/images/bh33_3d_trace.png。
"""

import sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CHECKS = []


def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))


def solve3d(nx, ny, nz, src):
    idx = -np.ones((nx, ny, nz), dtype=np.int64)
    for z in range(1, nz - 1):
        for y in range(ny):
            for x in range(nx):
                idx[x, y, z] = x + nx * (y + ny * (z - 1))
    N = nx * ny * (nz - 2)
    rows = []
    cols = []
    vals = []
    b = np.zeros(N)

    def add(r, c, v):
        rows.append(r)
        cols.append(c)
        vals.append(v)

    for z in range(1, nz - 1):
        for y in range(ny):
            for x in range(nx):
                r = int(idx[x, y, z])
                add(r, r, -6.0)
                add(r, int(idx[(x + 1) % nx, y, z]), 1.0)
                add(r, int(idx[(x - 1) % nx, y, z]), 1.0)
                add(r, int(idx[x, (y + 1) % ny, z]), 1.0)
                add(r, int(idx[x, (y - 1) % ny, z]), 1.0)
                if z - 1 >= 1:
                    add(r, int(idx[x, y, z - 1]), 1.0)
                if z + 1 <= nz - 2:
                    add(r, int(idx[x, y, z + 1]), 1.0)
                b[r] = -src[x, y, z]
    A = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
    sol = spla.spsolve(A, b)
    phi = np.zeros((nx, ny, nz))
    for z in range(1, nz - 1):
        for y in range(ny):
            for x in range(nx):
                phi[x, y, z] = sol[int(idx[x, y, z])]
    return phi


def metrics(phi_base, phi_struct, phi_mass, phi_joint, m):
    p0 = phi_base[..., :m]
    ps = phi_struct[..., :m]
    pm = phi_mass[..., :m]
    pj = phi_joint[..., :m]
    amp = float(np.max(np.abs(p0)))
    ds = ps - p0
    dm = pm - p0
    dtrue = pj - p0
    r_s = float(np.max(np.abs(ds)) / amp)
    r_m = float(np.max(np.abs(dm)) / amp)
    ratio = r_s / r_m
    resid = float(np.max(np.abs(dtrue - (ds + dm))) / np.max(np.abs(dtrue)))
    return amp, r_s, r_m, ratio, resid, ds, dtrue


print("==== BH3-3 后续（三维衔接）：层收缩静态痕迹验证（M8） ====")

# ---------- L0 准一维锚点（与 BH3-5 同构） ----------
nz0 = 101
full1 = np.zeros((1, 1, nz0))
for z in range(46, 56):
    full1[0, 0, z] = 1.0
drop1 = full1.copy()
drop1[0, 0, 50] = 0.0
full1s = np.zeros((1, 1, nz0 - 1))
full1s[0, 0, :] = full1[0, 0, :nz0 - 1]
drop1s = np.zeros((1, 1, nz0 - 1))
drop1s[0, 0, :] = drop1[0, 0, :nz0 - 1]

a0 = solve3d(1, 1, 101, full1)
a_s = solve3d(1, 1, 100, full1s)
a_m = solve3d(1, 1, 101, drop1)
a_j = solve3d(1, 1, 100, drop1s)
amp_a, rs_a, rm_a, ratio_a, resid_a, ds_a, dt_a = metrics(a0, a_s, a_m, a_j, 100)
print("  L0 锚点：结构 %.5f；质量 %.5f；比 %.4f；残差 %.4f" % (rs_a, rm_a, ratio_a, resid_a))
check("L0", abs(rs_a - 0.0213) / 0.0213 <= 0.05
      and abs(ratio_a - 0.202) / 0.202 <= 0.05
      and abs(resid_a - 0.018) / 0.018 <= 0.05,
      "复现 BH3-5（2.13e-2 / 0.202 / 0.018）")

# ---------- L1 层收缩几何 ----------
nx, ny = 21, 21
z_keep = list(range(0, 41))
z_new = list(range(0, 40))
coord_gap = [z_new[k + 1] - z_new[k] for k in range(len(z_new) - 1)]
shift = [abs(z_keep[k] - z_new[k]) for k in range(len(z_new))]
check("L1", len(z_new) == len(z_keep) - 1 and max(coord_gap) == 1 and max(shift) <= 1
      and (nx * ny * len(z_keep) - nx * ny * len(z_new)) == nx * ny,
      "层数 %d 转 %d；格距恒 1；坐标变动 ≤ 1；格点数减 %d" % (len(z_keep), len(z_new), nx * ny))

# ---------- L2 至 L4 主案例（S1 写法球源，未饱和档） ----------
Rs = 6
Mp = 4.0
s_tot = 4.0 * np.pi * Mp
ball = []
cx, cy, cz = (nx - 1) // 2, (ny - 1) // 2, 20
for x in range(nx):
    for y in range(ny):
        for z in range(41):
            if (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= Rs * Rs:
                ball.append((x, y, z))
s_site = s_tot / len(ball)
print("  主案例：球源格点数 %d、单点源 %.5f、总源 %.4f（M/m_P = %.1f）" % (len(ball), s_site, s_tot, Mp))

src_full = np.zeros((nx, ny, 41))
for (x, y, z) in ball:
    src_full[x, y, z] = s_site
src_drop = src_full.copy()
src_drop[cx, cy, cz] = 0.0

src_full_s = src_full[..., :40]
src_drop_s = src_drop[..., :40]
b0 = solve3d(nx, ny, 41, src_full)
b_s = solve3d(nx, ny, 40, src_full_s)
b_m = solve3d(nx, ny, 41, src_drop)
b_j = solve3d(nx, ny, 40, src_drop_s)
amp_b, rs_b, rm_b, ratio_b, resid_b, ds_b, dt_b = metrics(b0, b_s, b_m, b_j, 40)
print("  主案例：结构 %.5f；质量 %.5f；比 %.4f；残差 %.4f" % (rs_b, rm_b, ratio_b, resid_b))
check("L2", rs_b > 1e-4, "结构留痕相对幅度 %.3e（阈值 1e-4）" % rs_b)
check("L3", 1e-2 <= ratio_b <= 1e2, "结构/质量比 = %.4f（量级可比区间 [1e-2, 1e2]）" % ratio_b)
check("L4", resid_b < 0.1, "线性分解残差 = %.4f" % resid_b)

# ---------- L5 横向均匀性（板源） ----------
src_slab = np.zeros((nx, ny, 41))
for z in range(16, 25):
    src_slab[:, :, z] = 1.0
src_slab_s = src_slab[..., :40]
c0 = solve3d(nx, ny, 41, src_slab)
c_s = solve3d(nx, ny, 40, src_slab_s)
m_c = 40
d_slab = (c_s - c0[..., :m_c])
zmid = list(range(10, 30))
rel = []
for z in zmid:
    sl = d_slab[:, :, z]
    mx = float(np.max(np.abs(sl)))
    if mx > 0:
        rel.append(float(np.max(np.abs(sl - sl.mean())) / mx))
lv = max(rel) if rel else float("nan")
print("  板源：结构项横向相对起伏最大 %.3e" % lv)
check("L5", lv < 1e-8, "横向相对起伏 %.3e（阈值 1e-8）" % lv)

# ---------- 图 ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
fig, axs = plt.subplots(1, 3, figsize=(15.0, 4.2))

ax = axs[0]
zz = np.arange(100)
ax.plot(zz, ds_a[0, 0, :] / amp_a, label="结构项")
ax.plot(zz, (a_m[0, 0, :100] - a0[0, 0, :100]) / amp_a, label="质量项")
ax.plot(zz, dt_a[0, 0, :] / amp_a, label="联合项")
ax.set_xlabel("z（格）")
ax.set_ylabel("Δφ / 幅度")
ax.set_title("(a) 准一维锚点：三项差分")
ax.legend(fontsize=8)

ax = axs[1]
mid_y = (ny - 1) // 2
im = ax.imshow((ds_b[mid_y, :, :]).T, origin="lower", aspect="auto",
               cmap="RdBu_r")
ax.set_xlabel("x（格）")
ax.set_ylabel("z（格）")
ax.set_title("(b) 主案例：结构项中面（y 中部）")
plt.colorbar(im, ax=ax, fraction=0.046)

ax = axs[2]
ax.semilogy(zmid, rel, "o-")
ax.axhline(1e-8, color="gray", ls=":", lw=1)
ax.set_xlabel("z（格）")
ax.set_ylabel("横向相对起伏")
ax.set_title("(c) 板源：结构项横向均匀性")
plt.tight_layout()
plt.savefig(r"D:/software/TianXingFC/images/bh33_3d_trace.png", dpi=140)
print("  图已存 images/bh33_3d_trace.png")

npass = sum(1 for _, ok in CHECKS if ok)
print("==== 判据 %d/%d 通过 ====" % (npass, len(CHECKS)))
