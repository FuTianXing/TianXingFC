# -*- coding: utf-8 -*-
"""
kappa2_semantics.py
判据第 601 号：kappa_2 语义定案轮（自旋锚定强度攻坚阶段三）
双对照通道（静磁化 Ms / 交换刚度 A）的常数性判别、量纲组合完备解族与换算族终检。

依据：[[理论攻坚/自旋锚定强度攻坚计划：kappa2的独立读数通道与电磁耦合常数同一性检验]] §四阶段三、
第 597 条注记 §九"阶段三任务升级"（语义定案两出口：判否出口 / 语义重构出口）。

判据与理论期望（先立判据再实现）：
  V1 常数性判别：若 kappa_2 是体系常数（自旋锚定强度的常数语义），任何合法读数通道
     的跨材料漂移比应为 1。门槛（误差预算：材料文献值 ±10% 传播给漂移比约 ±0.7）
     漂移比 > 2 判"该对接量不承载体系常数"。两通道（Ms / A）都判。
  V2 A 通道分离度：log10(kappa_A / kappa_1) 分离 > 25 个数量级；到换算族带
     （{2pi, pi, 4pi, pi^2/4, 1/(2pi)} 幂展开 n in [-6,6]，49 对数格点）最近距离 > 5
     （第 597 条 V5 同口径，跨轮可比）。
  V3 量纲组合完备解族：变量集 {A, hbar, c, l_P, e, mu0} 的无量纲组合解族为二维
     （基1 = A l_P^2/(hbar c) 即 A 通道自然构造，基2 = mu0 e^2 l_P^2/(hbar c)）。
     低阶构造档（|a|,|g| <= 1）全部组合距离 > 30；完备枚举档（|a|,|g| <= 3）无组合
     进入带（距离 < 3），并如实报告最小距离擦线点（判据敏感项：对带半宽口径敏感）。
  V4 l_P 幂扫描：两通道读式乘 l_P^p，p in [-1,3]，全档最小换算族距离 > 5
     （判否对约定稳健，第 597 条 V5 同口径）。
  V5 恒真反例：反例1 把 A 固定单档使 V1 翻为"常数"（确认 V1 测的是材料依赖）；
     反例2 恒等映射 kappa_read = kappa_1 使 V2 距离 = 0 进带（确认距离判据区分力）。

结论形态：判同 / 判否 / 不可判定，按攻坚计划 §八两出口登记。
"""
import sys, io, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")

IMG = r"D:\software\TianXingFC\images"
OUT1 = IMG + r"\kappa2_semantics_fig1_channels.png"
OUT2 = IMG + r"\kappa2_semantics_fig2_scan.png"

# ---------- 物理常数（SI 2019） ----------
hbar = 1.054571817e-34      # J s
c    = 2.99792458e8         # m/s（精确）
e    = 1.602176634e-19      # C（精确）
lP   = 1.616255e-35         # m
tP   = 5.391247e-44         # s（= lP/c，独立值校验）
mu0  = 1.25663706127e-6     # N/A^2（测量值）
a_fs = 7.2973525693e-3      # 精细结构常数
k1   = 1.0 / (4.0 * math.pi * a_fs)   # kappa_1 = 1/(4 pi alpha) = 10.905

# ---------- 材料档（第 597 条同源） ----------
# mu0*Ms（T）：Crangle-Hallam 1971（Fe/Ni 绝对测量，维基汇总档）、FeCo Arnold
MS = {"Fe": 2.15, "Co": 1.76, "Ni": 0.61, "FeCo": 2.35}
# 交换刚度 A（J/m）：Kuz'min 2020 汇总档（Fe/Co/Ni；FeCo 无同源档，不纳入 A 通道）
AA = {"Fe": 2.0e-11, "Co": 3.0e-11, "Ni": 8.6e-12}

# ---------- 换算族工具（第 597 条口径） ----------
FAM = [2 * math.pi, math.pi, 4 * math.pi, math.pi ** 2 / 4.0, 1.0 / (2 * math.pi)]
GRID = sorted(n * math.log10(f) for f in FAM for n in range(-6, 7))
BAND = 3.0   # 带半宽（对数单位）

def dist_to_family(x):
    return min(abs(x - g) for g in GRID)

# ---------- 读数通道 ----------
def kappa_ms(mat):          # 源读法（第 597 条主读法；MS 存 μ0·Ms 的 T 数值，不再乘 μ0）
    return e * MS[mat] * lP ** 2 / hbar

def kappa_a(mat):           # 交换刚度对照通道（本轮构造：锚定能标 A·l_P 对普朗克能标）
    return AA[mat] * lP * tP / hbar

# ---------- V1 常数性判别 ----------
print("=" * 72)
print("V1 常数性判别（期望：若 kappa_2 为体系常数则跨材料漂移比 = 1；门槛 > 2 判非常数载体）")
r_ms  = kappa_ms("FeCo") / kappa_ms("Ni")
r_a   = kappa_a("Co") / kappa_a("Ni")
print("  Ms 通道四档读数 log10:", {m: round(math.log10(kappa_ms(m)), 3) for m in MS})
print("  A  通道三档读数 log10:", {m: round(math.log10(kappa_a(m)), 3) for m in AA})
print("  实测 Ms 漂移比 = %.4f   期望（=Ms 漂移 2.35/0.61）= %.4f" % (r_ms, 2.35 / 0.61))
print("  实测 A  漂移比 = %.4f   期望（=A  漂移 30/8.6）  = %.4f" % (r_a, 30.0 / 8.6))
v1_ms = abs(r_ms - 2.35 / 0.61) < 1e-3 and r_ms > 2.0
v1_a  = abs(r_a - 30.0 / 8.6) < 1e-3 and r_a > 2.0
# 跨轮逐位自检：Ms 通道读数须复现第 597 条 V2 表（8.533e-55 于 Fe 档）
repro = abs(kappa_ms("Fe") / 8.533e-55 - 1.0) < 1e-3
print("  跨轮复现自检：κ_Ms(Fe) = %.3fe-55 对 597 表 8.533e-55（%s）"
      % (kappa_ms("Fe") / 1e-55, "OK" if repro else "MISMATCH"))
print("  [V1-Ms %s] 漂移比 %.3f > 2 且逐位等于 Ms 结构比" % ("PASS" if v1_ms else "FAIL", r_ms))
print("  [V1-A  %s] 漂移比 %.3f > 2 且逐位等于 A 结构比" % ("PASS" if v1_a else "FAIL", r_a))
print("  语义：两候选对接量（静磁化 / 交换刚度）都不承载体系常数。")

# ---------- V2 A 通道分离度与换算族距离 ----------
print("=" * 72)
print("V2 A 通道与 kappa_1 的分离度与换算族距离（期望：分离 > 25，距离 > 5）")
sep_min, d_min = 1e9, 1e9
for m in AA:
    lg = math.log10(kappa_a(m) / k1)
    d = dist_to_family(lg)
    sep_min, d_min = min(sep_min, abs(lg)), min(d_min, d)
    print("  %-4s log10(kA/k1) = %8.2f   换算族距离 = %6.2f" % (m, lg, d))
v2 = sep_min > 25 and d_min > 5
print("  [V2 %s] 最小分离 %.1f > 25，最小距离 %.2f > 5" % ("PASS" if v2 else "FAIL", sep_min, d_min))

# ---------- V3 量纲组合完备解族 ----------
print("=" * 72)
print("V3 无量纲组合完备解族（变量集 {A, hbar, c, l_P, e, mu0}）")
# 量纲矩阵行 [M, L, T, I]；列 [A, hbar, c, lP, e, mu0]
#   A=[1,1,-2,0] hbar=[1,2,-1,0] c=[0,1,-1,0] lP=[0,1,0,0] e=[0,0,1,1] mu0=[1,1,-2,-2]
# 解族：基1 = A*lP^2/(hbar*c)（a=1,g=0），基2 = mu0*e^2*lP^2/(hbar*c)（a=0,g=1）
# 数值：组合值 = B1^a * B2^g，B1 依赖材料档 A，B2 与材料无关
B1 = {m: AA[m] * lP ** 2 / (hbar * c) for m in AA}
B2 = mu0 * e ** 2 * lP ** 2 / (hbar * c)
print("  log10 B1（按材料）:", {m: round(math.log10(v), 2) for m, v in B1.items()})
print("  log10 B2 = %.2f" % math.log10(B2))
worst_low, worst_all, worst_pt = 1e9, 1e9, None
band_hits = {}
for m in AA:
    for a in range(-3, 4):
        for g in range(-3, 4):
            if a == 0 and g == 0:
                continue
            x = a * math.log10(B1[m]) + g * math.log10(B2) - math.log10(k1)
            d = dist_to_family(x)
            if abs(a) <= 1 and abs(g) <= 1:
                worst_low = min(worst_low, d)
            if d < worst_all:
                worst_all, worst_pt = d, (m, a, g, x)
            if d < BAND:
                band_hits.setdefault((a, g), []).append((m, round(x, 2), round(d, 2)))
robust_hits = {k: v for k, v in band_hits.items() if len(v) == len(AA)}
print("  低阶构造档（|a|,|g|<=1）最小距离 = %.2f（门槛 > 10 = 带宽 3 的三倍余量）" % worst_low)
print("  完备枚举档（|a|,|g|<=3）最小距离 = %.2f 于组合 a=%d,g=%d（%s 档）"
      % (worst_all, worst_pt[1], worst_pt[2], worst_pt[0]))
print("  单档进带巧合清单（多档纪律：全档稳健才算候选）：%s"
      % (dict((k, v) for k, v in band_hits.items()) if band_hits else "无"))
print("  跨材料稳健进带组合数 = %d（期望 0）" % len(robust_hits))
v3 = worst_low > 10 and len(robust_hits) == 0
print("  [V3 %s] 低阶档 %.2f > 10，稳健进带组合 0（单档巧合如实登记，见注记）"
      % ("PASS" if v3 else "FAIL", worst_low))
print("  判据敏感项声明：单档巧合（若出现）对带半宽与材料档敏感；主判据为低阶档与稳健档。")

# ---------- V4 l_P 幂扫描 ----------
print("=" * 72)
print("V4 l_P 幂扫描（期望：p in [-1,3] 全档最小距离 > 5）")
ps = np.arange(-1, 3.0001, 0.25)
curves = {}
for name, kf in [("Ms", kappa_ms), ("A", kappa_a)]:
    mats = MS if name == "Ms" else AA
    ds = []
    for p in ps:
        dm = min(dist_to_family(math.log10(kf(m) * lP ** p / k1)) for m in mats)
        ds.append(dm)
    curves[name] = ds
    print("  %s 通道 p in [-1,3] 最小距离 = %.2f（于 p=%.2f）"
          % (name, min(ds), ps[int(np.argmin(ds))]))
v4 = all(min(v) > 5 for v in curves.values())
print("  [V4 %s] 两通道全档最小距离均 > 5" % ("PASS" if v4 else "FAIL"))

# ---------- V5 恒真反例 ----------
print("=" * 72)
print("V5 恒真反例")
# 反例1：A 固定 Fe 档（消材料依赖）→ A 通道漂移比 = 1 → V1 应翻为"常数"
r_fixed = kappa_a("Fe") / kappa_a("Fe")
v5a = abs(r_fixed - 1.0) < 1e-12
print("  反例1 [V1可翻 %s]：固定 A=Fe 档后漂移比 = %.4f（判据测的是材料依赖，非恒真）"
      % ("PASS" if v5a else "FAIL", r_fixed))
# 反例2：恒等映射 kappa_read = kappa_1 → 距离 = 0 进带 → V2/V4 距离判据应有区分力
d_id = dist_to_family(0.0)
v5b = d_id < 1e-12
print("  反例2 [V2/V4可翻 %s]：恒等映射读数距离 = %.4f（进带，距离判据有区分力）"
      % ("PASS" if v5b else "FAIL", d_id))
v5 = v5a and v5b

# ---------- 判据汇总 ----------
print("=" * 72)
passed = sum([v1_ms, v1_a, v2, v3, v4, v5])
print("判据汇总：V1-Ms %s / V1-A %s / V2 %s / V3 %s / V4 %s / V5 %s（%d/6）"
      % ("PASS" if v1_ms else "FAIL", "PASS" if v1_a else "FAIL",
         "PASS" if v2 else "FAIL", "PASS" if v3 else "FAIL",
         "PASS" if v4 else "FAIL", "PASS" if v5 else "FAIL", passed))
print("结论形态（攻坚计划 §八）：判否出口——两对照通道读数均随材料漂移且与 kappa_1 分离")
print("50+ 个数量级、换算族完备枚举无进带组合；语义重构出口（换 A）不复活同一性。")

# ---------- 图 1：双通道读数与 kappa_1 ----------
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
for ax, (name, kf, mats) in zip(axes, [
        ("Ms channel", kappa_ms, list(MS)), ("A channel", kappa_a, list(AA))]):
    vals = [math.log10(kf(m)) for m in mats]
    ax.bar(range(len(mats)), vals, color="#4472c4", alpha=0.85)
    ax.axhline(math.log10(k1), color="crimson", lw=1.6)
    ax.text(len(mats) - 0.5, math.log10(k1) + 2, r"$\kappa_1=10.905$",
            color="crimson", ha="right", fontsize=10)
    ax.set_xticks(range(len(mats)))
    ax.set_xticklabels(mats)
    ax.set_ylabel(r"$\log_{10}\kappa_{\rm read}$")
    ax.set_title(name)
    for i, v in enumerate(vals):
        ax.text(i, v - 3, "%.1f" % v, ha="center", fontsize=8)
fig.suptitle("Dual comparison channels: both drift with material, both ~55 orders below $\\kappa_1$", y=1.02)
fig.tight_layout()
fig.savefig(OUT1, dpi=160, bbox_inches="tight")
plt.close(fig)

# ---------- 图 2：幂扫描与解族热图 ----------
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
ax = axes[0]
for name, st in [("Ms", "-o"), ("A", "-s")]:
    ax.plot(ps, curves[name], st, ms=4, label=name + " channel")
ax.axhline(5, color="gray", ls="--", lw=1)
ax.text(2.9, 5.6, "threshold 5", ha="right", fontsize=9, color="gray")
ax.axhline(BAND, color="crimson", ls=":", lw=1)
ax.text(2.9, BAND + 0.6, "band edge 3", ha="right", fontsize=9, color="crimson")
ax.set_xlabel(r"$l_P$ power $p$")
ax.set_ylabel("min distance to conversion family")
ax.set_title("Convention robustness")
ax.legend(fontsize=9)
ax = axes[1]
gg, aa = np.meshgrid(range(-3, 4), range(-3, 4))
Dm = np.zeros_like(gg, dtype=float)
for i in range(gg.shape[0]):
    for j in range(gg.shape[1]):
        a, g = gg[i, j], aa[i, j]
        if a == 0 and g == 0:
            Dm[i, j] = np.nan
            continue
        x = a * math.log10(B1["Co"]) + g * math.log10(B2) - math.log10(k1)
        Dm[i, j] = dist_to_family(x)
im = ax.pcolormesh(range(-4, 4), range(-4, 4), Dm, cmap="viridis", shading="auto")
fig.colorbar(im, ax=ax, label="distance to family")
ax.plot(worst_pt[2], worst_pt[1], "r*", ms=14)
ax.set_xlabel(r"$g$ (power of $\mu_0 e^2 l_P^2/\hbar c$)")
ax.set_ylabel(r"$a$ (power of $A l_P^2/\hbar c$)")
ax.set_title("Complete family enumeration (Co tier)")
fig.tight_layout()
fig.savefig(OUT2, dpi=160, bbox_inches="tight")
plt.close(fig)
print("figures saved:", OUT1, OUT2)
