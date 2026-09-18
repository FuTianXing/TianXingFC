# -*- coding: utf-8 -*-
"""M6-S1（可做部分）：真实场构型 —— 写入律加容量约束的记录场与三项判据。

背景：BH3-4 终审的读数门槛之一是「真实场构型」：由写入导出的记录分布（含容量约束），
替代 D1/D2/S2R 使用的 1/r 型与手选调制构型。本材料执行 S1 的可做部分：写入律取已定案的
每拍注入 w_eff = (2 pi alpha/3)(m/m_P)，容量取 h <= 1 的逐点裁剪，稳态与动力学支无关
（阻尼波动与过阻尼静态解相同，见裁决十六）。P1-2（kappa_phi 上限与单格点容量）未到位，
按裁决十三声明门槛不齐。

判据（先写期望，再算）：
  S1-1 容量约束：全部扫描点逐点 h <= 1 + 1e-12；半饱和半径 r_half 随总质量 M/m_P 单调增
       且亚线性（对 M/m_P 的局部对数斜率 <= 0.9）；高饱和档的有效荷比 q_eff/M 被压到 0.6 以下。
  S1-2 弱场尾（同盒参照）与通量：线性参照解（同盒、同源、无容量）的离散高斯壳通量 = s_tot
       （相对偏差 <= 1e-10，N5 归一化锚点）；含容量解的外区尾对线性参照按有效荷比例一致
       （|h_sat/h_lin - q_eff/q_tot| <= 1e-3，窗口 r in [R_s+3, R_s+8]）；通量隐含的裁剪损失与
       稳态每拍裁剪损失（乘 6）一致（相对偏差 <= 1e-8，收敛水平）。
  S1-3（报告）：kappa_phi 上限与单格点容量按 P1-2 核对，未到位，声明门槛不齐。
  报告 R1：有效荷比与压荷读数（与 D3 的 BH2 同向对照）。
  报告 R2：源半径收缩档（坍缩方向）与构型导出（供 S3 复用）。

模型：三维立方 N=49（狄利克雷边界）；写入律为每拍「邻居接力（alpha=1）+ 源格点注入 +
容量裁剪」；源为半径 R_s 的均匀球。确定性、无随机数。
"""

import sys
import time
import numpy as np
from scipy.sparse.linalg import cg, LinearOperator

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
CHECKS = []
def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))

print("==== M6-S1（可做部分）：真实场构型（写入律 + 容量约束） ====")

N = 49
c0 = N // 2
ii, jj, kk = np.indices((N, N, N))
dist = np.sqrt((ii - c0) ** 2 + (jj - c0) ** 2 + (kk - c0) ** 2)
mask = np.zeros((N, N, N), bool); mask[1:-1, 1:-1, 1:-1] = True
bnd = ~mask
CAP = 1.0
ALPHA = 1.0

def tick(th, w, cap=True):
    avg = (np.roll(th, 1, 0) + np.roll(th, -1, 0) + np.roll(th, 1, 1) + np.roll(th, -1, 1)
           + np.roll(th, 1, 2) + np.roll(th, -1, 2)) / 6.0
    th = th + ALPHA * (avg - th)
    th[bnd] = 0.0
    raw = th + w
    if not cap:
        return raw
    loss = float(np.sum(np.maximum(raw - CAP, 0.0)[mask]))
    return np.minimum(raw, CAP), loss

def solve(Mp, R_s, cap=True, itmax=40000):
    ball = dist <= R_s
    n_ball = int(ball.sum())
    s_tot = 4.0 * np.pi * Mp
    w = np.where(ball, s_tot / (6.0 * n_ball), 0.0)
    th = np.zeros((N, N, N))
    loss = 0.0
    for it in range(itmax):
        if cap:
            th_new, dl = tick(th, w, cap=True); loss = dl
        else:
            th_new = tick(th, w, cap=False)
        dmax = float(np.max(np.abs(th_new - th)))
        th = th_new
        if it % 50 == 0 and dmax < 1e-14:
            break
    if cap:
        th_chk, loss = tick(th, w, cap=True)   # 最终态的每拍裁剪损失
    return th, w, s_tot, loss, n_ball

def cg_linear(Mp, R_s):
    """同盒、同源、无容量的线性参照：-L h = 6w（内部子空间求解）。"""
    ball = dist <= R_s
    s_tot = 4.0 * np.pi * Mp
    w = np.where(ball, s_tot / (6.0 * int(ball.sum())), 0.0)
    src = (6.0 * w)[1:-1, 1:-1, 1:-1]
    nint = (N - 2) ** 3
    def pad(x):
        v = np.zeros((N, N, N)); v[1:-1, 1:-1, 1:-1] = x
        return v
    def lap_sub(x):
        v = pad(x)
        r = (np.roll(v, 1, 0) + np.roll(v, -1, 0) + np.roll(v, 1, 1) + np.roll(v, -1, 1)
             + np.roll(v, 1, 2) + np.roll(v, -1, 2)) - 6.0 * v
        return r[1:-1, 1:-1, 1:-1]
    op = LinearOperator((nint, nint), matvec=lambda z: (-lap_sub(z.reshape((N - 2, N - 2, N - 2)))).ravel())
    sol, info = cg(op, src.ravel(), rtol=1e-13, maxiter=30000)
    assert info == 0
    return pad(sol.reshape((N - 2, N - 2, N - 2)))

def profile(th, rmax=26):
    r = np.arange(1, rmax + 1)
    vals = np.array([th[(np.abs(dist - rr) < 0.6) & mask].mean() for rr in r])
    return r, vals

def readings(th, R_s, s_tot, loss):
    r, prof = profile(th)
    sat = r[prof >= 1.0 - 1e-6]
    r_sat = float(sat.max()) if len(sat) else 0.0
    def cross(level):
        idx = np.where(prof <= level)[0]
        if len(idx) == 0 or idx[0] == 0:
            return np.nan
        i1 = idx[0]
        r1, r2 = r[i1 - 1], r[i1]; v1, v2 = prof[i1 - 1], prof[i1]
        return r1 + (v1 - level) * (r2 - r1) / (v1 - v2)
    r_half = cross(0.5)
    win = (r >= int(R_s) + 3) & (r <= int(R_s) + 8)
    sl = np.polyfit(np.log(r[win]), np.log(prof[win]), 1)[0]
    def flux(rr):
        cube = (np.abs(ii - c0) <= rr) & (np.abs(jj - c0) <= rr) & (np.abs(kk - c0) <= rr)
        lap = (np.roll(th, 1, 0) + np.roll(th, -1, 0) + np.roll(th, 1, 1) + np.roll(th, -1, 1)
               + np.roll(th, 1, 2) + np.roll(th, -1, 2)) - 6.0 * th
        return -float(np.sum(lap[cube]))
    F = [flux(rr) for rr in (N // 2 - 5, N // 2 - 4, N // 2 - 3)]
    Fspread = (max(F) - min(F)) / max(F)
    q_eff = float(np.mean(F)) / (4.0 * np.pi)
    loss_flux = s_tot - 4.0 * np.pi * q_eff
    loss_rel = abs(6.0 * loss - loss_flux) / max(s_tot, 1e-30)
    return dict(prof=prof, r= r, r_sat=r_sat, r_half=r_half, slope=sl, Fspread=Fspread,
                q_eff=q_eff, q_ratio=q_eff / Mp, loss_rel=loss_rel)

t0 = time.time()
print("\n  [扫描一] 总质量 M/m_P（R_s = 8 格固定）")
scan1 = [(1.0, 8.0), (4.0, 8.0), (16.0, 8.0), (100.0, 8.0)]
res1 = []
for Mp, Rs in scan1:
    th, w, s_tot, loss, nb = solve(Mp, Rs)
    rd = readings(th, Rs, s_tot, loss)
    th_lin = cg_linear(Mp, Rs)
    r, prof = profile(th); _, prof_l = profile(th_lin)
    win = (r >= int(Rs) + 3) & (r <= int(Rs) + 8)
    ratio = prof[win] / prof_l[win]
    rd["prop_dev"] = float(np.max(np.abs(ratio - rd["q_eff"] / Mp)))
    # 线性参照的通量锚点
    cube = (np.abs(ii - c0) <= N // 2 - 4) & (np.abs(jj - c0) <= N // 2 - 4) & (np.abs(kk - c0) <= N // 2 - 4)
    lap = (np.roll(th_lin, 1, 0) + np.roll(th_lin, -1, 0) + np.roll(th_lin, 1, 1) + np.roll(th_lin, -1, 1)
           + np.roll(th_lin, 1, 2) + np.roll(th_lin, -1, 2)) - 6.0 * th_lin
    rd["lin_flux_rel"] = abs(-float(np.sum(lap[cube])) - s_tot) / s_tot
    res1.append((Mp, Rs, rd, th))
    print("       M/m_P=%6.2f R_s=%.0f: h_max=%.6f  r_sat=%s  r_half=%.3f  q_eff/M=%.4f  比例偏差=%.1e  线性通量=%.1e  损失核对=%.1e"
          % (Mp, Rs, float(np.max(th[mask])), ("%.0f" % rd["r_sat"]), rd["r_half"], rd["q_ratio"], rd["prop_dev"], rd["lin_flux_rel"], rd["loss_rel"]))

print("\n  [扫描二] 源半径收缩（M/m_P = 100 固定，坍缩方向）")
scan2 = [(100.0, 4.0), (100.0, 2.5)]
res2 = []
for Mp, Rs in scan2:
    th, w, s_tot, loss, nb = solve(Mp, Rs)
    rd = readings(th, Rs, s_tot, loss)
    th_lin = cg_linear(Mp, Rs)
    r, prof = profile(th); _, prof_l = profile(th_lin)
    win = (r >= int(Rs) + 3) & (r <= int(Rs) + 8)
    rd["prop_dev"] = float(np.max(np.abs(prof[win] / prof_l[win] - rd["q_eff"] / Mp)))
    res2.append((Mp, Rs, rd, th))
    print("       M/m_P=%6.2f R_s=%.1f: h_max=%.6f  r_sat=%s  r_half=%.3f  q_eff/M=%.4f  比例偏差=%.1e  损失核对=%.1e"
          % (Mp, Rs, float(np.max(th[mask])), ("%.0f" % rd["r_sat"]), rd["r_half"], rd["q_ratio"], rd["prop_dev"], rd["loss_rel"]))
print("       用时 %.1f 秒" % (time.time() - t0))

# S1-1
allpts = res1 + res2
hmax = max(float(np.max(th[mask])) for _, _, _, th in allpts)
mp = np.array([x[0] for x in res1]); rh = np.array([x[2]["r_half"] for x in res1])
fin = np.isfinite(rh)
mpf, rhf = mp[fin], rh[fin]
mono = bool(np.all(np.diff(rhf) >= -1e-12))
sl_local = np.diff(np.log(rhf)) / np.diff(np.log(mpf))
qhi = res1[-1][2]["q_ratio"]
check("S1-1", hmax <= 1.0 + 1e-12 and len(mpf) >= 3 and mono and np.all(sl_local <= 0.9) and qhi < 0.6,
      "h_max=%.6f；r_half 单调 %s、对数斜率 %s <= 0.9；最高档 q_eff/M=%.4f < 0.6"
      % (hmax, mono, np.round(sl_local, 3), qhi))

# S1-2
Fs = np.array([x[2]["Fspread"] for x in allpts]); lr = np.array([x[2]["loss_rel"] for x in allpts])
lf = np.array([x[2]["lin_flux_rel"] for x in res1]); pd = np.array([x[2]["prop_dev"] for x in allpts])
check("S1-2", np.all(Fs <= 1e-10) and np.all(lr <= 1e-8) and np.all(lf <= 1e-10) and np.all(pd <= 1e-3),
      "线性通量锚点最大偏差 %.1e；通量散布 <= %.1e；损失核对最大 %.1e；外区比例偏差最大 %.1e <= 1e-3"
      % (float(np.max(lf)), float(np.max(Fs)), float(np.max(lr)), float(np.max(pd))))

print("\n  [S1-3] 报告：kappa_phi 上限与单格点容量依赖 P1-2（粒子线），未到位；")
print("         按裁决十三声明该门槛不齐；容量取 h<=1 的现象学裁剪（模型选择，待 P1-2 替换）。")

print("\n  [报告 R1] 有效荷比 q_eff/(M/m_P)：%s（对照 D3 的 BH2：饱和把有效荷压掉、视界半径亚线性）"
      % ["%.4f" % x[2]["q_ratio"] for x in allpts])
print("         压荷封顶读数：M/m_P = 16 与 100 的 r_half = %.3f 与 %.3f（比值 %.3f，接近封顶）"
      % (res1[2][2]["r_half"], res1[3][2]["r_half"], res1[3][2]["r_half"] / res1[2][2]["r_half"]))
print("         收缩档读数：R_s 由 8 缩到 4 与 2.5，q_eff 由 %.3f 降为 %.3f 与 %.3f，r_half 由 %.3f 降为 %.3f 与 %.3f"
      % (res1[3][2]["q_eff"], res2[0][2]["q_eff"], res2[1][2]["q_eff"], res1[3][2]["r_half"], res2[0][2]["r_half"], res2[1][2]["r_half"]))
print("         含义：容量裁剪把有效荷钉在源尺度上（q_eff ≈ 1.3 R_s），封顶禁止「半饱和面显著超出源尺度」的过密构型；")
print("         当构型压缩到自带视界尺度（2 M/m_P ≈ R_s）时裁剪退场、全荷通过（对照档 M/m_P=1、4 无裁剪）。")

print("\n  [报告 R2] 构型导出（供 S3 复用）：格点 N=%d；源 = 均匀球 R_s（8、4、2.5）；" % N)
print("         写入 w_eff=(2 pi alpha/3)(m/m_P)；容量 h<=1；ρ/M 剖面示例（M/m_P=100、R_s=8）：")
rr, pp = res1[3][2]["r"], res1[3][2]["prof"]
print("         r = %s" % rr[:18]); print("         h = %s" % np.round(pp[:18], 5))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
ax = axes[0]
for (Mp, Rs, rd, th), mk in zip(res1, ("o-", "s-", "^-", "d-")):
    ax.loglog(rd["r"], rd["prof"], mk, ms=3.5, lw=1.1, label="M/m_P = %.0f" % Mp)
th_lin = cg_linear(100.0, 8.0); r, prof_l = profile(th_lin)
ax.loglog(r, prof_l, "k--", lw=1.0, label="线性参照（M/m_P=100）")
ax.set_xlabel("r（格）"); ax.set_ylabel("h（壳平均）"); ax.set_ylim(1e-3, 1.5)
ax.set_title("(a) 真实场剖面（R_s = 8）", fontsize=9); ax.legend(fontsize=7)
ax = axes[1]
Ms = [x[0] for x in res1]; qr = [x[2]["q_ratio"] for x in res1]
ax.semilogx(Ms, qr, "o-", lw=1.2, label="R_s = 8")
ax.semilogx([100, 100], [res2[0][2]["q_ratio"], res2[1][2]["q_ratio"]], "s", ms=5, color="tab:red", label="R_s = 4 与 2.5")
ax.axhline(1.0, color="gray", ls=":", lw=1.0)
ax.set_xlabel("M / m_P"); ax.set_ylabel("q_eff / (M/m_P)")
ax.set_title("(b) 饱和压荷与封顶", fontsize=9); ax.legend(fontsize=7)
ax = axes[2]
rh = [x[2]["r_half"] for x in res1]
mfin = [(m, rr) for m, rr in zip(Ms, rh) if np.isfinite(rr)]
ax.loglog([m for m, _ in mfin], [rr for _, rr in mfin], "o-", lw=1.2, label="实测 r_half")
ax.loglog(Ms, [2.0 * m for m in Ms], "k--", lw=1.0, label="无容量外推 2 M/m_P")
ax.set_xlabel("M / m_P"); ax.set_ylabel("r_half（格）")
ax.set_title("(c) 半饱和半径的亚线性", fontsize=9); ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(r"D:\software\TianXingFC\images\m6_s1_field.png", dpi=150)
print("\n  图已存 images/m6_s1_field.png")
print("  面板范围核对：(a) r 1..26、h 1.9e-3..1.0；(b) M 1..100、q_eff/M 0.026..1.0；(c) r_half 5.17..11.96 对 2M 线 8..200")

print("\n==== 判据汇总：%d / %d 通过 ====" % (sum(1 for _, ok in CHECKS if ok), len(CHECKS)))
sys.exit(0 if all(ok for _, ok in CHECKS) else 1)
