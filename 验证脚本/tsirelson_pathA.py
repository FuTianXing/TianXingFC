# -*- coding: utf-8 -*-
"""P1-29 二-2 路径甲：由快照相位分布到 CHSH 的体系内计算（迁移轮 J1、J2、J3、J6 + 增强轮 J5）

定位：v0（tsirelson_gate_v0.py）把纠缠锁定态 |Phi_gamma> 作构造输入，
闭合了生死闸门的存在性与上界两颚。本脚本按《量子力学攻坚计划》战线二任务 二-2 的路径甲，
把该构造输入换成体系自己的读数链：快照相位分布（M4）、相对表针 Δα 关联（二-1）、锁定对
通道相关、关联核、CHSH 组合。迁移口径为换语言不换数字：与 v0 同参数点的精确量逐位对齐。

链路四环：
  一、单侧端口（命题 7 的相位语言）：零模踢 theta 后读层位，P(+1|theta)=1/2[1+cos(phi+theta)]，
      快照相位 phi 均匀（M4：随机触发、与系统无共同零点、噪声抹匀）；可见度 1、相位可回收。
  二、快照测度与无信号：单侧边际恒为 1/2，与对侧选角无关（均匀测度直接给出）。
  三、锁定对的两条通道相关：层位对层位 = 1（锁定），表针对表针 = <cos(Δα)> = V（二-1 的
      相对表针读数），交叉 = 0。由边际 1/2 与这三条相关唯一确定联合读数律
      P(a,b) = 1/4[1 + a b E]，E = cos(alpha)cos(beta) + cos(Δα)sin(alpha)sin(beta)。
  四、关联核 E(alpha,beta) = cos(alpha)cos(beta) + V sin(alpha)sin(beta)，V = <cos(Δα)>。

判据：
  迁移轮
  J1 单侧端口：条纹可见度 = 1（1e-12）；解析相位回收 <= 1e-12；采样回收 < 5 sigma
  J2 快照测度：单侧边际 = 1/2（1e-15）；与对侧选角无关（恒等式精确）；采样偏差 < 5 sigma
  J3 关联核：解析对采样每点 <= 3 sigma（sigma = 1/sqrt(N)）；wrapped 族闭式 V = e^{-v/2} 一致
  J6 数字对齐：V = 1 最优角 S = 2.828427（1e-12）；采样 S_hat 与 v0 的 2.82692±0.00100 在
      3 sigma 内、越经典界 z >= 800；子午面 max S = 2*sqrt(1+V^2)（V = 1 给 2*sqrt2，1e-6；
      V = 0 给 2）；退相干同角组 = sqrt(2)；LHV 穷举封顶 2；阈 p* = 1/sqrt2 处混合读数恰 2
  增强轮
  J5 锁定相干表：V = e^{-v/2}（v = gamma^2/kappa^2 + sigma^2/(2 kappa)，kappa = 0.125）与
      第 236 条闭式 r(gamma/kappa) 恒等（1e-15）；门槛带 V 对 r* 一致（1%）；联合条件窗内
      V >= 0.99 且 S_max = 2*sqrt(1+V^2) 远越经典界

纪律：全确定性（种子 20260914）；判据先行，实测 vs 期望；图存 images/（绝对路径）。
边界：不重开 v0 的存在性与上界两颚（只做迁移核对）；不涉及空间分离的全场实现（方向二）；
      不给 2*sqrt(2) 的原理级来源（方向三，另立脚本与推导稿）；Y 象限全空间口径未在相位语言侧重做。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

FIG_PATH = r"D:\software\TianXingFC\images\tsirelson_pathA.png"
SEED = 20260914
N_SAMP = 2_000_000
TS = 2.0 * np.sqrt(2.0)
CL = 2.0
SQ2 = np.sqrt(2.0)
KAPPA = 0.125

ALPHA0, ALPHA1 = 0.0, np.pi / 2.0
BETA0, BETA1 = np.pi / 4.0, -np.pi / 4.0
PAIRS = ((ALPHA0, BETA0), (ALPHA1, BETA0), (ALPHA0, BETA1), (ALPHA1, BETA1))
SIGNS = (+1.0, +1.0, +1.0, -1.0)

V0_REC = (2.82692, 0.00100)   # v0 判据 J4 记录值
V0_EXACT = 2.828427124746190

results = []

def gate(name, ok, detail):
    results.append((name, ok, detail))
    print("[%s] %s: %s" % ("PASS" if ok else "FAIL", name, detail))

def kernel(alpha, beta, V):
    """关联核：层位-层位相关 1、表针-表针相关 V、交叉 0，经踢角读层位给出。"""
    return np.cos(alpha) * np.cos(beta) + V * np.sin(alpha) * np.sin(beta)

def chsh(Es):
    return sum(s * E for s, E in zip(SIGNS, Es))

def fringe(phi, theta):
    """单侧端口：零模踢后层位读出的余弦条纹（命题 7）。"""
    return 0.5 * (1.0 + np.cos(phi + theta))

def wrapped_normal(v, n, rng):
    """同心 wrapped 族（二-1 定案的分布形状；v 为方差），返回模 2*pi 的 Δα。"""
    if v == 0.0:
        return np.zeros(n)
    return np.mod(v**0.5 * rng.standard_normal(n), 2.0 * np.pi)

def sample_E(alpha, beta, da, rng):
    """由联合读数律 P(a,b) = 1/4[1 + a b E] 逐发抽样，返回 E_hat 与 sigma。"""
    E_loc = np.cos(alpha) * np.cos(beta) + np.cos(da) * np.sin(alpha) * np.sin(beta)
    u = rng.random(da.size)
    p_pp = 0.25 * (1.0 + E_loc)
    p_opp = 0.25 * (1.0 - E_loc)
    ab = np.where(u < p_pp, 1.0, np.where(u < p_pp + 2.0 * p_opp, -1.0, 1.0))
    E_hat = ab.mean()
    sigma = np.sqrt(max(1.0 - E_hat**2, 1e-12) / da.size)
    return E_hat, sigma

def smax_meridian(V, n=361):
    """子午面全角最大（α0、α1 分离最大化后对 β0、β1 扫描），闭式 2*sqrt(1+V^2)。"""
    b = np.linspace(-np.pi, np.pi, n)
    B0, B1 = np.meshgrid(b, b, indexing="ij")
    t0 = np.sqrt((np.cos(B0) + np.cos(B1))**2 + V**2 * (np.sin(B0) + np.sin(B1))**2)
    t1 = np.sqrt((np.cos(B0) - np.cos(B1))**2 + V**2 * (np.sin(B0) - np.sin(B1))**2)
    return float(np.max(t0 + t1))

print("=" * 78)
print("P1-29 二-2 路径甲：快照相位分布到 CHSH 的体系内计算（迁移轮 + 增强轮 J5）")
print("=" * 78)

rng = np.random.default_rng(SEED)

# ---------- J1 单侧端口 ----------
phis = np.array([0.0, np.pi / 3.0, 2.371])
vis, rec = [], []
for phi in phis:
    P0, P90 = fringe(phi, 0.0), fringe(phi, np.pi / 2.0)
    A1, A2 = 2.0 * P0 - 1.0, 2.0 * P90 - 1.0
    vis.append(float(np.hypot(A1, A2)))
    rec.append(float(np.mod(np.arctan2(-A2, A1), 2.0 * np.pi)))
rec_err = np.abs((np.array(rec) - phis + np.pi) % (2.0 * np.pi) - np.pi)
n1 = 200_000
samp_rec, samp_vis = [], []
for phi in phis:
    c = rng.binomial(n1, fringe(phi, 0.0)) / n1
    s = rng.binomial(n1, fringe(phi, np.pi / 2.0)) / n1
    B1, B2 = 2.0 * c - 1.0, 2.0 * s - 1.0
    samp_vis.append(float(np.hypot(B1, B2)))
    samp_rec.append(float(np.mod(np.arctan2(-B2, B1), 2.0 * np.pi)))
samp_err = np.abs((np.array(samp_rec) - phis + np.pi) % (2.0 * np.pi) - np.pi)
sig_rec = 1.0 / np.sqrt(n1)
ok = (np.abs(np.array(vis) - 1.0).max() < 1e-12 and rec_err.max() < 1e-12
      and samp_err.max() < 5.0 * sig_rec and np.abs(np.array(samp_vis) - 1.0).max() < 5.0 * sig_rec)
gate("J1 单侧端口", ok,
     "可见度(解析)=%.14f 相位回收(解析)<=%.1e；采样回收 %.1e 对 5sigma=%.1e（%d 发/点）"
     % (min(vis), rec_err.max(), samp_err.max(), 5.0 * sig_rec, n1))

# ---------- J2 快照测度与无信号 ----------
phi_grid = np.linspace(0.0, 2.0 * np.pi, 20000, endpoint=False)
th_test = np.array([0.0, 0.7, np.pi / 2.0, 2.4, np.pi])
dev_exact = max(abs(fringe(phi_grid, th).mean() - 0.5) for th in th_test)
betas_j2 = np.array([BETA0, BETA1, 1.234])
E_loc_j2 = np.cos(ALPHA0) * np.cos(betas_j2) + np.sin(ALPHA0) * np.sin(betas_j2)
dev_marg = float(np.abs(0.25 * (1.0 + E_loc_j2) + 0.25 * (1.0 - E_loc_j2) - 0.5).max())
n2 = 300_000
mA = []
for beta in betas_j2:
    da = wrapped_normal(1.0, n2, rng)
    E_loc = np.cos(ALPHA0) * np.cos(beta) + np.cos(da) * np.sin(ALPHA0) * np.sin(beta)
    u = rng.random(n2)
    a = np.where(u < 0.25 * (1.0 + E_loc) + 0.25 * (1.0 - E_loc), 1.0, -1.0)
    mA.append(a.mean())
mA = np.array(mA)
sig_m = 1.0 / np.sqrt(n2)
ok = dev_exact < 1e-15 and dev_marg < 1e-15 and np.abs(mA).max() < 5.0 * sig_m
gate("J2 快照测度与无信号", ok,
     "单侧边际对 theta 精确 1/2（偏差 %.1e）；联合律边际恒等式偏差 %.1e；三档对侧选角采样 |<a>| <= %.1e 对 5sigma=%.1e"
     % (dev_exact, dev_marg, np.abs(mA).max(), 5.0 * sig_m))

# ---------- J3 关联核 ----------
betas = np.linspace(-np.pi, np.pi, 13)
dev_k, dev_sig, n_pts, rows_v1 = 0.0, 0.0, 0, []
for V, vv in ((1.0, 0.0), (float(np.exp(-0.5)), 1.0)):
    da = wrapped_normal(vv, N_SAMP, rng)
    for alpha in (ALPHA0, ALPHA1):
        for beta in betas:
            Ea = kernel(alpha, beta, V)
            Eh, sg = sample_E(alpha, beta, da, rng)
            dev_k = max(dev_k, abs(Eh - Ea))
            dev_sig = max(dev_sig, abs(Eh - Ea) / sg)
            n_pts += 1
            if alpha == ALPHA1 and vv == 1.0:
                rows_v1.append((beta, Ea, Eh, sg))
mean_cos = float(np.mean(np.cos(wrapped_normal(1.0, N_SAMP, rng))))
dev_v = abs(mean_cos - np.exp(-0.5))
ok = dev_sig <= 3.0 and dev_v < 5e-3
gate("J3 关联核", ok,
     "解析对采样最大偏差 %.1e = %.2f sigma（阈 3 sigma，%d 点，两档 Δα，sigma=1/sqrt(N)）；"
     "<cos(Δα)> 采样 %.6f 对闭式 e^{-v/2}=%.6f（偏差 %.1e）"
     % (dev_k, dev_sig, n_pts, mean_cos, np.exp(-0.5), dev_v))
for idx in (0, 6, 12):
    rb, re_, rh, rs = rows_v1[idx]
    print("     J3 抽点（V=e^{-1/2}，alpha=pi/2）：beta=%+.3f 解析 %.6f 采样 %.6f ± %.6f（%.2f sigma）"
          % (rb, re_, rh, rs, abs(rh - re_) / max(rs, 1e-12)))

# ---------- J6 数字对齐 ----------
Es_exact = [kernel(a, b, 1.0) for a, b in PAIRS]
S_exact = chsh(Es_exact)
da0 = np.zeros(N_SAMP)
Es_hat, sig2 = [], 0.0
for a, b in PAIRS:
    Eh, sg = sample_E(a, b, da0, rng)
    Es_hat.append(Eh)
    sig2 += sg**2
S_hat = chsh(Es_hat)
sig_S = sig2**0.5
z = (S_hat - CL) / sig_S
dev_v0 = abs(S_hat - V0_REC[0])
sig_v0 = (sig_S**2 + V0_REC[1]**2)**0.5
smax1 = smax_meridian(1.0)
smax0 = smax_meridian(0.0)
cl_theory = max(a0 * b0 + a1 * b0 + a0 * b1 - a1 * b1
                for a0 in (1, -1) for a1 in (1, -1) for b0 in (1, -1) for b1 in (1, -1))
S_deph = chsh([kernel(a, b, 0.0) for a, b in PAIRS])
p_star = 1.0 / SQ2
S_p_star = 2.0 * SQ2 * p_star
ok = (abs(S_exact - TS) < 1e-12 and dev_v0 < 3.0 * sig_v0 and z >= 800.0
      and abs(smax1 - TS) < 1e-6 and abs(smax0 - CL) < 1e-6 and cl_theory == 2
      and abs(S_deph - SQ2) < 1e-12 and abs(S_p_star - CL) < 1e-12)
gate("J6 数字对齐", ok,
     "S(解析)=%.14f（对 v0 %.14f 差 %.1e）；采样 S_hat=%.5f±%.5f 对 v0 %.5f±%.5f（差 %.2f sigma）"
     % (S_exact, V0_EXACT, abs(S_exact - V0_EXACT), S_hat, sig_S, V0_REC[0], V0_REC[1], dev_v0 / sig_v0))
print("     越经典界 z = %.0f（阈 800）；子午面 max S(V=1)=%.9f（期望 %.9f）、S(V=0)=%.6f（期望 2.000000）"
      % (z, smax1, TS, smax0))
print("     退相干同角组 S=%.12f（期望 sqrt2=%.12f）；LHV 穷举封顶 %.1f；阈 p*=%.6f 处混合读数 S=%.6f（恰经典界）"
      % (S_deph, SQ2, float(cl_theory), p_star, S_p_star))

# ---------- J5 锁定相干表与联合条件窗 ----------
def v_wrapped(gamma, sigma2=0.0, kappa=KAPPA):
    """同心 wrapped 族方差（第 236 条）：v = gamma^2/kappa^2 + sigma^2/(2 kappa)。"""
    return gamma ** 2 / kappa ** 2 + sigma2 / (2.0 * kappa)

def V_lock(gamma, sigma2=0.0, kappa=KAPPA):
    """表针通道锁定相干：V = <cos Δα> = e^{-v/2}（wrapped 族特征函数，m = 1 矩）。"""
    return float(np.exp(-0.5 * v_wrapped(gamma, sigma2, kappa)))

def r_closed(x, sigma2=0.0, kappa=KAPPA):
    """第 236 条口径化闭式：r = e^{-(gamma/kappa)^2/2 - sigma^2/(4 kappa)}。"""
    return float(np.exp(-0.5 * x ** 2 - sigma2 / (4.0 * kappa)))

GAMMA_WIN = (0.0132, 0.0139)
row_g = (0.0, GAMMA_WIN[0], GAMMA_WIN[1], 0.05)
row_V = tuple(V_lock(g) for g in row_g)
row_S = tuple(2.0 * np.sqrt(1.0 + V ** 2) for V in row_V)
dev_id = max(abs(V_lock(g) - r_closed(g / KAPPA)) for g in row_g)
band_x = (1.19, 1.29, 1.39)
band_V = tuple(V_lock(x * KAPPA) for x in band_x)
band_r = (0.490, 0.433, 0.379)
dev_band = max(abs(v - r) / r for v, r in zip(band_V, band_r))
S_win = row_S[2]
ok = (dev_id < 1e-15 and dev_band < 0.01 and row_V[2] >= 0.99 and S_win > CL + 0.8
      and smax_meridian(row_V[2], 241) > CL)
gate("J5 锁定相干表与联合条件窗", ok,
     "V = e^{-v/2} 对第 236 条闭式恒等（偏差 %.1e）；门槛带 V 对 r* 最大偏差 %.2f%%（噪声项口径）；"
     "联合条件窗内 V(%.5f) = %.6f，S_max = %.6f（越经典界 %.2f）"
     % (dev_id, 100.0 * dev_band, GAMMA_WIN[1], row_V[2], S_win, S_win - CL))
for gi, Vi, Si in zip(row_g, row_V, row_S):
    print("     J5 表：gamma=%.4f（gamma/kappa=%.4f）V=%.6f S_max=%.6f（对 2*sqrt2 亏缺 %.2f%%）"
          % (gi, gi / KAPPA, Vi, Si, 100.0 * (1.0 - Si / TS)))
print("     J5 门槛带：gamma/kappa = 1.19 / 1.29 / 1.39 处 V = %.4f / %.4f / %.4f 对第 236 条 r* = 0.490 / 0.433 / 0.379"
      % band_V)

# ---------- 图 ----------
fig, axes = plt.subplots(3, 2, figsize=(12.2, 12.4))

ax = axes[0, 0]
th_plot = np.linspace(0.0, 2.0 * np.pi, 721)
for phi in phis:
    ax.plot(th_plot, fringe(phi, th_plot) - 0.5, lw=1.6, label="phi=%.3f" % phi)
ax.axhline(0.0, color="0.7", lw=0.8)
ax.set_xlabel("踢角 theta")
ax.set_ylabel("层位读数 P(+1) - 1/2")
ax.set_title("J1 单侧端口：相位回收残差 <= %.1e" % rec_err.max())
ax.legend(fontsize=8)

ax = axes[0, 1]
rng_fig = np.random.default_rng(SEED + 7)
bb = np.linspace(-np.pi, np.pi, 241)
for V, vv, st in ((1.0, 0.0, "-"), (float(np.exp(-0.5)), 1.0, "--")):
    ax.plot(bb, kernel(ALPHA1, bb, V), st, lw=1.4, label="解析 V=%.4f" % V)
    da = wrapped_normal(vv, 60_000, rng_fig)
    pts, errs, bg = [], [], np.linspace(-np.pi, np.pi, 17)
    for beta in bg:
        Eh, sg = sample_E(ALPHA1, beta, da, rng_fig)
        pts.append(Eh)
        errs.append(sg)
    ax.errorbar(bg, pts, yerr=errs, fmt="o", ms=3.4, lw=0.9, capsize=2, label="采样 V=%.4f" % V)
ax.set_xlabel("选角 beta（alpha = pi/2）")
ax.set_ylabel("关联核 E")
ax.set_title("J3 关联核：解析对采样最大偏差 %.1e（%.2f sigma）" % (dev_k, dev_sig))
ax.legend(fontsize=8)

ax = axes[1, 0]
Vg = np.linspace(0.0, 1.4, 71)
ax.plot(Vg, [smax_meridian(V, 241) for V in Vg], lw=1.6, label="子午面最大（分离最大化）")
ax.plot(Vg, 2.0 * np.sqrt(1.0 + Vg**2), "k--", lw=1.0, label="闭式 2*sqrt(1+V^2)")
ax.axhline(TS, color="0.6", lw=0.9, ls=":")
ax.axhline(CL, color="0.6", lw=0.9, ls=":")
ax.plot([1.0], [TS], "r*", ms=11, label="V=1：2*sqrt2")
ax.set_xlabel("表针相干 V = <cos(Δα)>")
ax.set_ylabel("S_max")
ax.set_title("J6 子午面剖面：V=1 饱和 v0 精确值 %.6f" % smax1)
ax.legend(fontsize=8)

ax = axes[1, 1]
names = ["锁定相干\n(V=1)", "去相干\n(V=0)", "经典圈\n(预存角)", "白噪声阈\n(p*)"]
vals = [S_exact, S_deph, 2.0, S_p_star]
cols = ["#2b6cb0", "#c53030", "#718096", "#2f855a"]
bars = ax.bar(names, vals, color=cols, width=0.6)
ax.axhline(CL, color="k", lw=1.1, ls="--")
ax.text(3.3, CL + 0.03, "经典界 2", fontsize=8)
for rect, val in zip(bars, vals):
    ax.text(rect.get_x() + rect.get_width() / 2, val + 0.03, "%.4f" % val, ha="center", fontsize=8)
ax.set_ylim(0.0, 3.2)
ax.set_ylabel("S")
ax.set_title("J6 四读法对照：违反载体 = 表针相干")

ax = axes[2, 0]
xg = np.linspace(0.0, 1.6, 401)
ax.plot(xg, np.exp(-0.5 * xg**2), lw=1.6, label="V = e^{-(gamma/kappa)^2/2}（= 第 236 条 r）")
ax.plot(band_x, band_r, "ks", ms=5, label="第 236 条 r* 记录（1.19 / 1.29 / 1.39）")
ax.axvspan(GAMMA_WIN[0] / KAPPA, GAMMA_WIN[1] / KAPPA, color="#2f855a", alpha=0.25,
           label="联合条件窗（gamma <= 0.0139）")
ax.axvspan(1.19, 1.39, color="#c53030", alpha=0.15, label="门槛带 1.29 ± 0.10")
ax.set_xlabel("gamma / kappa")
ax.set_ylabel("锁定相干 V")
ax.set_title("J5 锁定相干表：窗内 V = %.4f，门槛带 V = %.3f" % (row_V[2], band_V[1]))
ax.legend(fontsize=8)

ax = axes[2, 1]
ax.plot(xg, 2.0 * np.sqrt(1.0 + np.exp(-xg**2)), lw=1.6, label="S_max = 2*sqrt(1+V^2)")
ax.axhline(TS, color="0.6", lw=0.9, ls=":", label="2*sqrt2")
ax.axhline(CL, color="k", lw=1.0, ls="--", label="经典界 2")
ax.axvspan(GAMMA_WIN[0] / KAPPA, GAMMA_WIN[1] / KAPPA, color="#2f855a", alpha=0.25)
ax.plot([GAMMA_WIN[1] / KAPPA], [S_win], "r*", ms=11,
        label="窗边：%.6f" % S_win)
ax.set_xlabel("gamma / kappa")
ax.set_ylabel("S_max")
ax.set_title("J5 窗内可见违反：S_max 高于经典界 %.2f" % (S_win - CL))
ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig(FIG_PATH, dpi=140)
plt.close(fig)

# ---------- 汇总 ----------
print("-" * 78)
n_ok = sum(1 for _, ok, _ in results if ok)
print("汇总：%d/%d 通过（种子 %d，采样 %d 发/点）" % (n_ok, len(results), SEED, N_SAMP))
print("图：%s" % FIG_PATH)
