# -*- coding: utf-8 -*-
"""BH3-3 后续（饱和档三维痕迹）：容量裁剪在场下三维层收缩痕迹的读数与档位订正。

判据（先立判据，再实现；均可证伪）：
  S0 复现与容量状态复核：线性解在 M8 主案例（R_s = 6、M/m_P = 4）复现 M8 读数
     （结构 4.185e-2、质量 1.017e-2、比 4.1151、残差 0.0011；相对偏差 <= 5%）；
     线性 h_max：M/m_P = 2 档 < 1（未饱和锚点）、4 档 > 1（超阈档，标定订正依据）。
  S1 裁剪激活：写入迭代稳态在 4、16、64 档 h_max = 1（>= 1 - 1e-12）；
     裁剪格点数 > 0 且随档位单调不减。
  S2 锚点退化：2 档非线性解 = 线性解（场最大偏差 <= 1e-9 乘幅度）。
  S3 结构留痕非零：饱和档（4、16、64）结构项 > 1e-4。
  S4 质量项吞没：饱和档质量项低于 1e-12（中心格点邻域全在裁剪集内时精确为零；
     首轮曾以「随档位下降」表述以噪声级数值通过，订正为阈值型判据，见过程记录）。
  S5 分解残差：饱和档残差 < 0.1。
  S6 横向均匀性：饱和板源结构项横向相对起伏 < 1e-8。
  S7 平台边界对照：清除格点移至球面外缘（邻域不全在裁剪集内）时质量项恢复非零
     （> 1e-6；4 与 64 档）。

模型：与 M8 同几何（21 x 21 横向周期、z 两端 Dirichlet；41 转 40 层收缩，源图案保持
      原索引）；写入迭代与 S1 同源：th <- min(邻域均值 + w, 1)，w = s/6，未饱和区固定点
      满足 L th = -s（与 M8 线性方程同口径）；球源均匀球 R_s = 6、s_tot = 4 pi M/m_P；
      被清除质量取球心单格点（S7 对照取球面格点）；板源 z 向 16 至 24、与 M8 同口径。
确定性、无随机数。图存 D:/software/TianXingFC/images/bh33_3d_sat_trace.png。
"""

import sys
import time
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

def relax_sat(nz, w, warm=None, tol=1e-12, itmax=80000):
    nx, ny = w.shape[0], w.shape[1]
    th = np.zeros((nx, ny, nz)) if warm is None else warm.copy()
    th[:, :, 0] = 0.0
    th[:, :, -1] = 0.0
    zup = np.zeros_like(th)
    zdn = np.zeros_like(th)
    dm = np.inf
    it = 0
    for it in range(1, itmax + 1):
        zup[:, :, 1:] = th[:, :, :-1]
        zdn[:, :, :-1] = th[:, :, 1:]
        avg = (np.roll(th, 1, 0) + np.roll(th, -1, 0) + np.roll(th, 1, 1)
               + np.roll(th, -1, 1) + zup + zdn) / 6.0
        th_new = np.minimum(avg + w, 1.0)
        th_new[:, :, 0] = 0.0
        th_new[:, :, -1] = 0.0
        dm = float(np.max(np.abs(th_new - th)))
        th = th_new
        if dm < tol:
            break
    return th, it, dm

def metrics(p0f, psf, pmf, pjf, m):
    p0 = p0f[..., :m]
    ps = psf[..., :m]
    pm = pmf[..., :m]
    pj = pjf[..., :m]
    amp = float(np.max(np.abs(p0)))
    ds = ps - p0
    dm_ = pm - p0
    dtrue = pj - p0
    r_s = float(np.max(np.abs(ds)) / amp)
    r_m = float(np.max(np.abs(dm_)) / amp)
    ratio = r_s / r_m if r_m > 0 else float("inf")
    resid = float(np.max(np.abs(dtrue - (ds + dm_))) / np.max(np.abs(dtrue)))
    return amp, r_s, r_m, ratio, resid

print("==== BH3-3 后续（饱和档三维痕迹）：容量裁剪在场下三维层收缩痕迹读数 ====")
t_start = time.time()
nx, ny = 21, 21
cx, cy, cz = 10, 10, 20
Rs = 6
ball_cells = [(x, y, z) for x in range(nx) for y in range(ny) for z in range(41)
              if (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= Rs * Rs]
n_ball = len(ball_cells)
print("球源格点数 %d" % n_ball)

def sphere_src(Mp):
    src = np.zeros((nx, ny, 41))
    s_tot = 4.0 * np.pi * Mp
    for (x, y, z) in ball_cells:
        src[x, y, z] = s_tot / n_ball
    return src

print("-- S0 线性复现与容量状态复核 --")
scan_mps = [1.0, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0, 16.0, 32.0, 64.0]
scan_hmax = {}
for Mp in scan_mps:
    phi = solve3d(nx, ny, 41, sphere_src(Mp))
    scan_hmax[Mp] = float(np.max(phi))
    print("  扫描 M/m_P = %5.1f  线性 h_max = %.6f" % (Mp, scan_hmax[Mp]))
cross = None
for i in range(len(scan_mps) - 1):
    a, b = scan_mps[i], scan_mps[i + 1]
    ha, hb = scan_hmax[a], scan_hmax[b]
    if (ha - 1.0) * (hb - 1.0) < 0.0:
        cross = a + (b - a) * (1.0 - ha) / (hb - ha)
print("  线性过阈点（h_max 跨 1）：M/m_P ≈ %.3f" % cross)

src4 = sphere_src(4.0)
drop4 = src4.copy()
drop4[cx, cy, cz] = 0.0
m8_lin = (solve3d(nx, ny, 41, src4), solve3d(nx, ny, 40, src4[..., :40]),
          solve3d(nx, ny, 41, drop4), solve3d(nx, ny, 40, drop4[..., :40]))
amp8, rs8, rm8, ratio8, resid8 = metrics(*m8_lin, 40)
print("  M8 主案例线性复现：结构 %.5f；质量 %.5f；比 %.4f；残差 %.4f" % (rs8, rm8, ratio8, resid8))
ok_rep = (abs(rs8 - 0.04185) / 0.04185 <= 0.05 and abs(rm8 - 0.01017) / 0.01017 <= 0.05
          and abs(ratio8 - 4.1151) / 4.1151 <= 0.05 and abs(resid8 - 0.0011) / 0.0011 <= 0.05)
check("S0a", ok_rep, "复现 M8 读数（4.185e-2 / 1.017e-2 / 4.1151 / 0.0011，偏差 <= 5%）")
check("S0b", scan_hmax[2.0] < 1.0 and scan_hmax[4.0] > 1.0,
      "线性 h_max：2 档 %.4f < 1；4 档 %.4f > 1（超阈档，订正依据）" % (scan_hmax[2.0], scan_hmax[4.0]))

print("-- S1 至 S5 四档读数（线性与饱和） --")
MP_LIST = [2.0, 4.0, 16.0, 64.0]
RES = {}
for Mp in MP_LIST:
    src = sphere_src(Mp)
    w = src / 6.0
    drop = src.copy()
    drop[cx, cy, cz] = 0.0
    wdrop = w.copy()
    wdrop[cx, cy, cz] = 0.0
    l0 = solve3d(nx, ny, 41, src)
    ls = solve3d(nx, ny, 40, src[..., :40])
    lm = solve3d(nx, ny, 41, drop)
    lj = solve3d(nx, ny, 40, drop[..., :40])
    ampL, rsL, rmL, ratioL, residL = metrics(l0, ls, lm, lj, 40)
    s0, it0, dm0 = relax_sat(41, w, warm=np.minimum(l0, 1.0))
    ss, it1, dm1 = relax_sat(40, w[..., :40], warm=np.minimum(ls, 1.0))
    sm, it2, dm2 = relax_sat(41, wdrop, warm=np.minimum(lm, 1.0))
    sj, it3, dm3 = relax_sat(40, wdrop[..., :40], warm=np.minimum(lj, 1.0))
    ampS, rsS, rmS, ratioS, residS = metrics(s0, ss, sm, sj, 40)
    hmax = float(np.max(s0))
    nclip = int(np.sum(s0 >= 1.0 - 1e-12))
    diff_anchor = float(np.max(np.abs(s0 - l0)))
    RES[Mp] = dict(lin=(rsL, rmL, ratioL, residL), sat=(rsS, rmS, ratioS, residS),
                   hmax=hmax, nclip=nclip, its=(it0, it1, it2, it3),
                   dms=(dm0, dm1, dm2, dm3), anchor=diff_anchor, amp=ampS,
                   prof_lin=l0[cx, cy, :].copy(), prof_sat=s0[cx, cy, :].copy(),
                   arr0=s0, arrs=ss)
    print("  档 M/m_P = %5.1f" % Mp)
    print("    线性：结构 %.5f；质量 %.5f；比 %.4f；残差 %.4f" % (rsL, rmL, ratioL, residL))
    print("    饱和：结构 %.5f；质量 %.2e；比 %.2e；残差 %.2e" % (rsS, rmS, ratioS, residS))
    print("    h_max = %.6f；裁剪格点 %d；迭代 (基/构/质/联) = %d/%d/%d/%d；末拍 dmax = %.2e"
          % (hmax, nclip, it0, it1, it2, it3, max(dm0, dm1, dm2, dm3)))

sat_mps = [4.0, 16.0, 64.0]
check("S1", all(abs(RES[Mp]["hmax"] - 1.0) <= 1e-12 for Mp in sat_mps)
      and all(RES[Mp]["nclip"] > 0 for Mp in sat_mps)
      and RES[4.0]["nclip"] <= RES[16.0]["nclip"] <= RES[64.0]["nclip"],
      "裁剪激活：h_max = 1；格点数 %d / %d / %d（单调不减）"
      % (RES[4.0]["nclip"], RES[16.0]["nclip"], RES[64.0]["nclip"]))
check("S2", RES[2.0]["anchor"] <= 1e-9 * RES[2.0]["amp"],
      "2 档非线性 = 线性（最大场偏差 %.2e，阈值 %.2e）"
      % (RES[2.0]["anchor"], 1e-9 * RES[2.0]["amp"]))
check("S3", all(RES[Mp]["sat"][0] > 1e-4 for Mp in sat_mps),
      "饱和档结构项 %.5f / %.5f / %.5f（阈值 1e-4）"
      % (RES[4.0]["sat"][0], RES[16.0]["sat"][0], RES[64.0]["sat"][0]))
check("S4", all(RES[Mp]["sat"][1] < 1e-12 for Mp in sat_mps),
      "质量项吞没：%.2e / %.2e / %.2e（阈值 1e-12，噪声级残留）"
      % (RES[4.0]["sat"][1], RES[16.0]["sat"][1], RES[64.0]["sat"][1]))
check("S5", all(RES[Mp]["sat"][3] < 0.1 for Mp in sat_mps),
      "分解残差 %.2e / %.2e / %.2e（阈值 0.1）"
      % (RES[4.0]["sat"][3], RES[16.0]["sat"][3], RES[64.0]["sat"][3]))

print("-- S6 板源横向均匀性（饱和） --")
src_slab = np.zeros((nx, ny, 41))
for z in range(16, 25):
    src_slab[:, :, z] = 1.0
w_slab = src_slab / 6.0
lb = solve3d(nx, ny, 41, src_slab)
ls2 = solve3d(nx, ny, 40, src_slab[..., :40])
tb, itb, _ = relax_sat(41, w_slab, warm=np.minimum(lb, 1.0))
ts, its_, _ = relax_sat(40, w_slab[..., :40], warm=np.minimum(ls2, 1.0))
hmax_slab = float(np.max(tb))
d_slab = ts - tb[..., :40]
zmid = list(range(10, 30))
rel = []
for z in zmid:
    sl = d_slab[:, :, z]
    mx = float(np.max(np.abs(sl)))
    if mx > 0:
        rel.append(float(np.max(np.abs(sl - sl.mean())) / mx))
lv = max(rel) if rel else float("nan")
print("  板源：h_max = %.6f；结构项横向相对起伏最大 %.3e（迭代 %d/%d）" % (hmax_slab, lv, itb, its_))
check("S6", lv < 1e-8, "横向相对起伏 %.3e（阈值 1e-8）" % lv)

print("-- S7 平台边界对照（清除格点移至球面外缘） --")
surf = (cx, cy, cz + Rs)
SURF = {}
for Mp in [4.0, 64.0]:
    src = sphere_src(Mp)
    w = src / 6.0
    dropS = src.copy()
    dropS[surf] = 0.0
    wS = w.copy()
    wS[surf] = 0.0
    lms = solve3d(nx, ny, 41, dropS)
    sms, its1, _ = relax_sat(41, wS, warm=np.minimum(lms, 1.0))
    rm_surf = float(np.max(np.abs(sms - RES[Mp]["arr0"])) / RES[Mp]["amp"])
    SURF[Mp] = rm_surf
    print("  档 M/m_P = %5.1f：球面清除质量项 %.3e（迭代 %d）" % (Mp, rm_surf, its1))
check("S7", SURF[4.0] > 1e-6 and SURF[64.0] > 1e-6,
      "球面清除恢复非零：%.2e / %.2e（阈值 1e-6）" % (SURF[4.0], SURF[64.0]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
fig, axs = plt.subplots(1, 3, figsize=(15.0, 4.2))

ax = axs[0]
xs = MP_LIST
ax.semilogy(xs, [RES[Mp]["lin"][0] for Mp in xs], "s--", color="tab:red", label="结构项（线性）")
ax.semilogy(xs, [RES[Mp]["sat"][0] for Mp in xs], "o-", color="tab:red", label="结构项（饱和）")
ax.semilogy(xs, [max(RES[Mp]["lin"][1], 1e-16) for Mp in xs], "s--", color="tab:blue", label="质量项（线性）")
ax.semilogy(xs, [max(RES[Mp]["sat"][1], 1e-16) for Mp in xs], "o-", color="tab:blue", label="质量项（饱和）")
ax.set_xscale("log", base=2)
ax.set_xticks(xs)
ax.set_xticklabels([str(int(v)) for v in xs])
ax.set_xlabel("M / m_P")
ax.set_ylabel("相对幅度")
ax.set_title("(a) 读数对档位（线性与饱和）")
ax.legend(fontsize=7)

ax = axs[1]
zc = np.arange(41)
ax.plot(zc, RES[4.0]["prof_lin"], "--", color="gray", label="M/m_P = 4 线性")
ax.plot(zc, RES[4.0]["prof_sat"], "-", color="tab:blue", label="M/m_P = 4 饱和")
ax.plot(zc, RES[16.0]["prof_sat"], "-", color="tab:red", label="M/m_P = 16 饱和")
ax.axhline(1.0, color="black", ls=":", lw=1, label="容量上界")
ax.set_xlabel("z（格）")
ax.set_ylabel("θ（中心线）")
ax.set_title("(b) 中心线剖面")
ax.legend(fontsize=7)

ax = axs[2]
xs2 = scan_mps
ax.semilogy(xs2, [scan_hmax[v] for v in xs2], "o-", color="tab:green")
ax.axhline(1.0, color="black", ls=":", lw=1)
if cross is not None:
    ax.axvline(cross, color="gray", ls="--", lw=1)
    ax.annotate("过阈 ≈ %.2f" % cross, xy=(cross, 1.0), xytext=(cross * 1.15, 1.5), fontsize=8)
ax.set_xscale("log", base=2)
ax.set_xlabel("M / m_P")
ax.set_ylabel("线性 h_max")
ax.set_title("(c) 线性 h_max 扫描（容量上界 = 1）")
plt.tight_layout()
plt.savefig(r"D:/software/TianXingFC/images/bh33_3d_sat_trace.png", dpi=140)
print("  图已存 images/bh33_3d_sat_trace.png")

print("运行时 %.1f 秒" % (time.time() - t_start))
npass = sum(1 for _, ok in CHECKS if ok)
print("==== 判据 %d/%d 通过 ====" % (npass, len(CHECKS)))
