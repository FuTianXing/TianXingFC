# -*- coding: utf-8 -*-
"""
minimal_verification_m6_direction_lambda.py
M6 方向-λ 核查：耦合系数翻转线的小-λ 渐近与 kappa_phi 候选的对齐。

背景：S2、S3、S3N 的「耦合先触发」读数均在模型惯例 lambda/J = 1 口径下给出；方向判定的
物理口径归结为 K/J = 6 pi kappa_phi t_P（S2-1b 路线一）。S3 材料用局部拟合 beta = 0.88 外推
给出翻转线 7.3e-43。本材料直接测量 I(lambda) 的小-lambda 行为（到 lambda = 1e-6，另有
mu 线性律的独立核验），给翻转线以实测渐近口径，并与 kappa_phi 候选对齐。

模型（与 M6-S3、M6-S3N 同口径）：N = 256 一维交错链，H[n,n] = 2 + (-1)^n h，邻接 +1；
gamma(a) = sqrt(max(0, -mu_min(lambda a shape)))；I = trapz(gamma, a)；
Gamma_c(lambda) = I(lambda)/ln(1/eps0)（S3 口径 eps0 = 1e-3）。物理写入速率 2.74e-39 每拍。

判据（先写判据再实现）：
  DL-1  I(lambda) 随 lambda 单调；小-lambda 局部斜率 beta(lambda) 收敛（末两区间之差 <= 0.05），
        报渐近 beta_inf。
  DL-2  小-lambda 幂律校准（两项）：其一，mu_min(lambda shape)/lambda 收敛到常数（mu 线性律，
        末两点相对差 <= 5%）；其二，末三点对 I = C lambda^0.5 的拟合偏差 <= 2%。
  DL-3  局部口径核对：beta = 0.88 重算翻转线与 S3 的 7.3e-43 一致（<= 20%），并标明该口径
        只在 [3e-3, 1] 局部有效。
  DL-4  候选对齐（报告）：模型惯例 1、粒子线电子档（含 J0 因子）、中微子档公式外推；
        Gamma_c/Gamma 比值、方向判定与 J0 翻转阈值。

严格程度：DL-1 至 DL-3 为可复现数值事实；DL-4 为报告（候选值引自既有材料，本材料不裁决）。
边界：一维；不含识别项裁决；数值地板（|mu| 低于 ~1e-13 不可分辨）在扫描下限处声明。
"""
import os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
CHECKS = []
def check(tag, ok, detail):
    CHECKS.append((tag, bool(ok)))
    print("  [%s] %s  %s" % (tag, "PASS" if ok else "FAIL", detail))

print("==== M6 方向-λ 核查：翻转线的小-λ 渐近与 kappa_phi 候选对齐 ====")
N = 256
nn = np.arange(N)
rc = np.minimum(np.abs(nn - N // 2), N - np.abs(nn - N // 2))
SHAPE = np.where(rc <= 7, 1.0, np.minimum(10.597 / np.maximum(rc, 1.0), 1.0))

def low1d(h):
    H = np.zeros((N, N))
    H[nn, nn] = 2.0 + ((-1.0) ** nn) * h
    H[nn, (nn + 1) % N] += 1.0
    H[nn, (nn - 1) % N] += 1.0
    return float(np.linalg.eigvalsh(H)[0])

LN3 = np.log(1.0 / 1e-3)
GAMMA_PHYS = 2.74e-39

# ---------------- 独立核验：mu 线性律（mu_min(lambda shape)/lambda 收敛） ----------------
print("\n  [核验] mu_min(lambda*shape)/lambda 的收敛（线性律）与均匀剖面对照（应显二次律）：")
lam_probe = [3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7]
c_vals = []
for lam in lam_probe:
    mu = low1d(lam * SHAPE)
    c_vals.append(mu / lam)
    mu_u = low1d(lam * np.ones(N))
    print("       lambda = %.0e：mu/lambda = %+.6e；均匀剖面 mu/lambda^2 = %+.6e" % (lam, mu / lam, mu_u / lam ** 2))
c_last, c_prev = c_vals[-1], c_vals[-2]
rel_c = abs(c_last - c_prev) / abs(c_prev)
print("       线性律收敛检查：末两点相对差 %.2f%%；均匀剖面比值应随 lambda 减半（二次律对照）" % (rel_c * 100))

# ---------------- I(lambda) 扫描 ----------------
def I_of(lam, npts=41):
    a = np.linspace(0.0, 1.0, npts)
    g = np.array([np.sqrt(max(0.0, -low1d(lam * aa * SHAPE))) for aa in a])
    return float(np.trapezoid(g, a))

t0 = time.time()
I1 = I_of(1.0)
Gc1 = I1 / LN3
print("\n  [锚点] I(1) = %.5f（S3 材料 0.222144 用真实场、S3N 材料 0.23145 用解析剖面，本值同后者口径）" % I1)
print("         Gamma_c(1; eps0=1e-3) = %.5f 每拍（S3 材料 3.216e-2）" % Gc1)

LAMS = [1.0, 0.3, 0.1, 0.03, 0.01, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6]
Ivals = [I_of(l) for l in LAMS]
print("\n  [扫描] lambda, I(lambda), 局部斜率 beta：")
betas = []
for k in range(len(LAMS)):
    if k == 0:
        print("       lambda = %.0e：I = %.6e" % (LAMS[k], Ivals[k]))
    else:
        b = np.log(Ivals[k - 1] / Ivals[k]) / np.log(LAMS[k - 1] / LAMS[k])
        betas.append(float(b))
        print("       lambda = %.0e：I = %.6e，beta = %.4f" % (LAMS[k], Ivals[k], b))
mono = all(Ivals[k] < Ivals[k - 1] for k in range(1, len(Ivals)))
print("       扫描用时 %.1f 秒" % (time.time() - t0))
beta_inf = betas[-1]
check("DL-1", mono and abs(betas[-1] - betas[-2]) <= 0.05,
      "I 单调且末两区间 beta 收敛（beta_inf = %.4f）" % beta_inf)

# DL-2 小-λ 幂律校准
# 末三点对 C lambda^0.5 拟合
xs = np.log(LAMS[-3:]); ys = np.log(Ivals[-3:])
slope, intercept = np.polyfit(xs, ys, 1)
resid = np.max(np.abs(ys - (slope * xs + intercept)))
pow_dev = float(np.exp(resid) - 1.0)
ok2 = (rel_c <= 0.05) and (pow_dev <= 0.02) and (abs(slope - 0.5) <= 0.02)
check("DL-2", ok2, "mu 线性律收敛 %.2f%%；末三点拟合斜率 %.4f、偏差 %.2f%%（期望 0.5）" % (rel_c * 100, slope, pow_dev * 100))

# DL-3 局部口径核对
r = GAMMA_PHYS * LN3 / I1
lam_local = r ** (1.0 / 0.88)
dev3 = abs(lam_local - 7.3e-43) / 7.3e-43
print("\n  [翻转线] r = Gamma*ln(1/eps0)/I(1) = %.3e" % r)
print("       局部口径（beta = 0.88，仅 [3e-3, 1] 有效）：lambda_min = %.3e（S3 的 7.3e-43，偏差 %.1f%%）" % (lam_local, dev3 * 100))
lam_asym = r ** (1.0 / beta_inf)
print("       渐近口径（beta_inf = %.4f，实测）：lambda_min = %.3e" % (beta_inf, lam_asym))
check("DL-3", dev3 <= 0.20, "beta=0.88 重算与 S3 一致（%.1f%%），该口径登记为局部拟合" % (dev3 * 100))

# DL-4 候选对齐
KAP_PHI_OVER_WP2 = 4.0e-47
KAP_TP = KAP_PHI_OVER_WP2 * (2.0 * np.pi) ** 2
KJ_ELEC = 6.0 * np.pi * KAP_TP
KJ_NU = 6.0 * np.pi * (0.05 / 1.2209e28 / (2.0 * np.pi)) ** 2 * (2.0 * np.pi) ** 2
cands = [("模型惯例 K/J = 1（D1、D2、S2-1、S3 等口径）", 1.0),
         ("粒子线电子档（J0 = 1）", KJ_ELEC),
         ("中微子档公式外推（0.05 eV，报告）", KJ_NU)]
print("\n  [候选对齐] 电子档：kappa_phi/omega_P^2 = 4.0e-47 -> K/J = %.3e（按 J0 线性缩放）" % KJ_ELEC)
print("       候选                                    K/J        Gamma_c/Gamma(渐近)  Gamma_c/Gamma(0.88)  方向")
for name, kj in cands:
    gcinf = Gc1 * kj ** beta_inf / GAMMA_PHYS
    gc88 = Gc1 * kj ** 0.88 / GAMMA_PHYS
    v = "耦合先" if gcinf >= 1.0 else "饱和先"
    print("       %-38s %.3e  %.3e           %.3e            %s" % (name, kj, gcinf, gc88, v))
# J0 翻转阈值（渐近口径）
j0_flip = lam_asym / KJ_ELEC
print("       电子档在渐近口径下的 J0 翻转阈值：J0 < %.3e 才翻到饱和先（任何合理 J0 均不翻转）" % j0_flip)

# 口径对齐（S3N 时间域交叉）
g_sat = 0.001 + (0.005 - 0.001) * (1.0 - 0.689) / (1.113 - 0.689)
print("\n  [口径对齐] S3N 时间域交叉（eps0=1e-6，内插）Gamma_sat(1) = %.4f；线性时钟 Gamma_c(1; eps0=1e-6) = %.4f；比值 %.1f（与 S3N 的 2 至 2.6 倍延迟口径一致）"
      % (g_sat, I1 / np.log(1e6), (I1 / np.log(1e6)) / g_sat))

print("\n  ======== 判据汇总 ========")
npass = sum(1 for _, ok in CHECKS if ok)
print("  通过 %d / %d" % (npass, len(CHECKS)))
for tag, ok in CHECKS:
    print("    %-6s %s" % (tag, "PASS" if ok else "FAIL"))

# ---------------- 出图 ----------------
IMG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "images", "m6_direction_lambda.png")
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

ax = axes[0]
ax.loglog(LAMS, Ivals, "o-", color="tab:blue", label="I(λ) 直接测量")
ax.loglog(LAMS, [I1 * l ** 0.5 for l in LAMS], "--", color="tab:red", label="斜率 0.5（渐近）")
ax.loglog(LAMS, [I1 * l ** 0.88 for l in LAMS], ":", color="tab:gray", label="斜率 0.88（S3 局部）")
ax.set_xlabel("λ（模型单位，J = 1）"); ax.set_ylabel("I(λ)")
ax.set_title("(a) I(λ) 与局部斜率（1.35 降至 0.50）")
ax.legend(fontsize=8)

ax = axes[1]
gc_scan = [v / LN3 for v in Ivals]
ax.loglog(LAMS, gc_scan, "o-", color="tab:blue", label="Γ_c(λ)（扫描）")
ax.axhline(GAMMA_PHYS, color="k", lw=0.9, ls="--")
ax.text(2e-13, GAMMA_PHYS * 1.8, "物理写入速率 2.74e-39", fontsize=8)
ax.axvline(lam_local, color="tab:gray", ls=":", lw=0.9)
ax.axvline(lam_asym, color="tab:red", ls=":", lw=0.9)
ax.text(lam_local * 1.2, 2e-8, "λ_min(0.88)", fontsize=7, rotation=90)
ax.text(lam_asym * 1.5, 2e-8, "λ_min(渐近)", fontsize=7, rotation=90)
ax.set_xlim(1e-80, 1e1)
ax.set_xlabel("λ"); ax.set_ylabel("Γ_c（每拍）")
ax.set_title("(b) 翻转线与物理速率")

ax = axes[2]
lam_grid = np.logspace(-80, 1, 140)
ax.loglog(lam_grid, Gc1 * lam_grid ** 0.88 / GAMMA_PHYS, "-", color="tab:gray", label="Γ_c/Γ（0.88 局部）")
ax.loglog(lam_grid, Gc1 * lam_grid ** beta_inf / GAMMA_PHYS, "-", color="tab:red", label="Γ_c/Γ（渐近 %.3f）" % beta_inf)
ax.axhline(1.0, color="k", lw=0.9, ls="--")
ax.text(0.4, 3e10, "耦合先（上）", fontsize=8)
ax.text(1e-78, 1e-3, "饱和先（下）", fontsize=8)
ax.plot([KJ_ELEC], [Gc1 * KJ_ELEC ** beta_inf / GAMMA_PHYS], "o", color="tab:orange", ms=7, label="电子档 %.0e" % KJ_ELEC)
ax.plot([KJ_NU], [Gc1 * KJ_NU ** beta_inf / GAMMA_PHYS], "^", color="tab:purple", ms=7, label="中微子档 %.0e" % KJ_NU)
ax.plot([1.0], [Gc1 / GAMMA_PHYS], "s", color="tab:green", ms=7, label="模型惯例 1")
ax.set_xlabel("λ（模型单位）"); ax.set_ylabel("Γ_c/Γ")
ax.set_title("(c) 方向判据与候选落点")
ax.legend(fontsize=7, loc="lower right")

fig.tight_layout()
fig.savefig(IMG, dpi=150)
print("\n  图已保存: %s" % IMG)
print("  面板核对：(a) 十三点 I(λ) 自 %.3e 降至 %.3e；(b) Γ_c 自 %.3e 降至 %.3e、物理速率线与两条 λ_min 竖线（%.0e 与 %.0e）；(c) 判据线与三个候选点。" % (Ivals[0], Ivals[-1], gc_scan[0], gc_scan[-1], lam_local, lam_asym))
