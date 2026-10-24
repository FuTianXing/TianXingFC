# -*- coding: utf-8 -*-
"""
最小验证：Born 判据物理档重估轮（第 624 号，判据 PR1 至 PR5）
出处：内部草稿/排查报告-依赖缺陷只有几格宽前提的旧结论清点（2026-10-12）甲节
「多体论文 Born 判据」建议处置（物理档重估轮加硬核执行者命题重判）；
宽度输入由康普顿波长词条恒等式供给（第 622 号 K1：定稿档宽度 = m_P/m_e，格）。

模型设定（明示假设，口径逐字移植第 259 条注记与多体论文 §4.1 至 §4.2）：
  有效势 u_kl(r) = U_0 theta(xi - |r|)，U_0 ~ hbar*omega_0*(xi/l_P)^3（权重体积图像：
  重叠区 (xi/l_P)^3 个格点，每格点禁入能量取能隙 hbar*omega_0）；
  质量恒等式 omega_0 = m*c^2/hbar；Born 判据比值 R = U_0 / [hbar^2/(m*xi^2)]。
  读法二（质量恒等式）：R_2 = 4*pi^2 * n^5 * (m/m_P)^2；
  读法三（混合，禁入视为元相子层占位事件）：R_3 = n^5 * (m/m_P)；n = xi/l_P。
  物理档宽度输入（第 622 号恒等式，CODATA 核验相对差 0）：n = m_P/m（对每档粒子
  各自成立，即 n_i = m_P/m_i）；模型档宽度为演示参数 n = 2.25 至 2.91（K = 0.005 档）。

判据与理论期望（先立后测）：
  P0  移植自检：复现第 259 条注记模型档读数（R_1 = 39.5、R_2(n=1) = 6.916e-44、
      R_3(n=1) = 4.186e-23、阈值 n*_3 = 2.990e4）。
  PR1 物理档比值：四档粒子（e、mu、tau、p）代入 n = m_P/m 后 R_2 = 4*pi^2*(m_P/m)^3、
      R_3 = (m_P/m)^4，全部远大于 1（电子 R_2 = 5.384e68、R_3 = 3.258e89）。
      期望：Born 判据物理档翻转（失效），与排查报告甲节预判同向。
  PR2 代数等价与宽度双路：n = m_P/m 代入的化简式与代入前原式逐位一致（< 1e-12）；
      宽度定稿值（登记 K）与 CODATA 直算 m_P/m_e 相对差 ~1e-7 量级（第 622 号 K2 已登记）。
  PR3 阈值超越：物理档 n 超失效阈值 n*（R_2 口径 4.29e8 至 8.94e8、R_3 口径 2.99e4）
      13 至 18 个数量级。
  PR4 硬核执行者重判：Born 支 a_s^Born/xi = (2/3)*R_2 >> 1（Born 公式自破：给出的
      a_s 超硬核几何上限 xi 数十个数量级），深势极限 a_s -> xi（硬球饱和），故物理档
      a_s 取几何支 ~ xi；u_kl 从描述项升为承重项（与模型档结论相反，档位标注义务）。
  PR5 稳健性与有牙：k_b 口径变体（xi 乘除 2*pi）不翻方向；恒真反例（n 设回模型档
      2.25）判定翻回 Born 成立（判据有牙）。

可逆性等级：P0、PR2、PR5 恒真反例半边为判据稳健（逐位/代数）；PR1、PR3、PR4 为
  读数（十进数量级，无阈值调参）；PR5 口径变体为判据敏感（口径依赖，外推须重跑）。
前提依赖：U_0 的权重体积图像（多体论文 §4.1，O(1) 因子与势型待定；结论对 O(1)
  因子不敏感，R 超阈值 13 个数量级以上）；物理档宽度恒等式（第 622 号，判据级）；
  质量恒等式（M-E 裁决）。边界：本脚本只重估 Born 判据与硬核执行者命题的档位判定，
  不重算 U_0 精确系数、不做二粒子散射数值（多体论文遗留 (c)）。
"""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

print("=" * 76)
print("Born 判据物理档重估轮（第 624 号，PR1 至 PR5）")
print("=" * 76)

IMG_PATH = r"D:\software\TianXingFC\images\born_physical_tier.png"
VERDICT = {}

# ----------------------------------------------------------------------
# 常量（CODATA 常用值；第 622 号 K2 已做 CODATA 核验）
# ----------------------------------------------------------------------
MP_ME = 2.389222e22          # 第 622 号 K1 定稿值（格）= m_P/m_e（登记 K 给出）
MP_ME_CODATA = 2.3892221e22  # CODATA 直算（用于 PR2 双路核对）
MASS_RATIOS = {              # m/m_e
    "电子": 1.0,
    "μ子": 206.7682830,
    "τ子": 3477.23,
    "质子": 1836.15267343,
}

def R2(n, mu):
    """质量恒等式读法：mu = m/m_P"""
    return 4.0 * math.pi ** 2 * n ** 5 * mu ** 2

def R3(n, mu):
    """混合读法"""
    return n ** 5 * mu

# ----------------------------------------------------------------------
# P0. 移植自检：复现第 259 条注记模型档读数
# ----------------------------------------------------------------------
print("\n[P0] 移植自检：复现第 259 条注记读数（模型档口径）")
mu_e = 1.0 / MP_ME
R1 = 4.0 * math.pi ** 2 * 1.0 ** 5
R2_n1 = R2(1.0, mu_e)
R3_n1 = R3(1.0, mu_e)
n_star_2 = (MP_ME ** 2) ** 0.2            # (m_P/m)^{2/5}，注记口径（忽略 4pi^2）
n_star_2x = (4.0 * math.pi ** 2 * mu_e ** 2) ** (-0.2)   # 精确 R_2 = 1
n_star_3 = MP_ME ** 0.2                    # (m_P/m)^{1/5}
t = {"R1": (R1, 39.5), "R2n1": (R2_n1, 6.916e-44), "R3n1": (R3_n1, 4.186e-23),
     "n*2": (n_star_2, 8.939e8), "n*3": (n_star_3, 2.990e4)}
p0_ok = True
for k, (meas, exp) in t.items():
    rel = abs(meas - exp) / abs(exp)
    ok = rel < 5e-3
    p0_ok = p0_ok and ok
    print(f"  {k}: 实测 {meas:.6e}，注记 {exp:.6e}，相对差 {rel:.2e}")
print(f"  精确 R_2 = 1 阈值（含 4pi^2 因子）: {n_star_2x:.6e}（注记口径 {n_star_2:.6e}，差 4pi^2 的 1/5 次方）")
VERDICT["P0"] = p0_ok
print(f"  [P0] 判定：{'通过' if p0_ok else '未通过'}")

# ----------------------------------------------------------------------
# PR1. 物理档比值（四档粒子）
# ----------------------------------------------------------------------
print("\n[PR1] 物理档比值：n_i = m_P/m_i（第 622 号恒等式）")
print(f"  {'粒子':<4} {'n = m_P/m':>12} {'R_2 = 4pi^2(m_P/m)^3':>22} {'R_3 = (m_P/m)^4':>18}")
pr1_ok = True
res_pr1 = {}
for name, ratio in MASS_RATIOS.items():
    n_i = MP_ME / ratio
    r2i = 4.0 * math.pi ** 2 * n_i ** 3
    r3i = n_i ** 4
    res_pr1[name] = (n_i, r2i, r3i)
    ok = r2i > 1e50 and r3i > 1e50
    pr1_ok = pr1_ok and ok
    print(f"  {name:<4} {n_i:>12.5e} {r2i:>22.5e} {r3i:>18.5e}")
print("  期望：全部 >> 1（Born 失效，物理档翻转）")
VERDICT["PR1"] = pr1_ok
print(f"  [PR1] 判定：{'通过' if pr1_ok else '未通过'}")

# ----------------------------------------------------------------------
# PR2. 代数等价与宽度双路
# ----------------------------------------------------------------------
print("\n[PR2] 代数等价：代入前原式 vs 化简式（n = m_P/m）")
pr2_ok = True
for name, ratio in MASS_RATIOS.items():
    mu = ratio / MP_ME
    n_i = MP_ME / ratio
    r2_before = R2(n_i, mu)           # 原式：4pi^2 n^5 (m/m_P)^2
    r2_after = 4.0 * math.pi ** 2 * n_i ** 3   # 化简：4pi^2 (m_P/m)^3
    rel = abs(r2_before - r2_after) / r2_after
    pr2_ok = pr2_ok and rel < 1e-12
    print(f"  {name}: 原式 {r2_before:.6e} vs 化简 {r2_after:.6e}，相对差 {rel:.2e}")
rel_w = abs(MP_ME - MP_ME_CODATA) / MP_ME_CODATA
print(f"  宽度双路：定稿值 {MP_ME:.6e} vs CODATA {MP_ME_CODATA:.6e}，相对差 {rel_w:.2e}（第 622 号 K2 登记）")
pr2_ok = pr2_ok and rel_w < 1e-5
VERDICT["PR2"] = pr2_ok
print(f"  [PR2] 判定：{'通过' if pr2_ok else '未通过'}")

# ----------------------------------------------------------------------
# PR3. 阈值超越
# ----------------------------------------------------------------------
print("\n[PR3] 阈值超越：物理档 n_e 对失效阈值 n*")
dec_2 = math.log10(MP_ME) - math.log10(n_star_2)
dec_2x = math.log10(MP_ME) - math.log10(n_star_2x)
dec_3 = math.log10(MP_ME) - math.log10(n_star_3)
print(f"  n_e = {MP_ME:.6e}；R_2 口径（注记）超阈 {dec_2:.2f} 个数量级，"
      f"精确口径超阈 {dec_2x:.2f}；R_3 口径超阈 {dec_3:.2f}")
pr3_ok = 12.0 < dec_2 < 15.0 and 17.0 < dec_3 < 19.0
print("  期望：13 至 18 个数量级（与排查报告甲节一致）")
VERDICT["PR3"] = pr3_ok
print(f"  [PR3] 判定：{'通过' if pr3_ok else '未通过'}")

# ----------------------------------------------------------------------
# PR4. 硬核执行者重判：Born 支自破
# ----------------------------------------------------------------------
print("\n[PR4] 硬核执行者重判：Born 支 vs 几何支")
print("  直接由第一原理算 a_s^Born/xi（不经 R_2 归一，避免判据 4pi^2 因子混入）：")
print("  a_s^Born = 2 m U_0 xi^3 / (3 hbar^2)，U_0 = m c^2 n^3，xi = n l_P")
print("  => a_s^Born/xi = (2/3) n^5 (m/m_P)^2 = (2/3) (m_P/m)^3（用 n = m_P/m 恒等式）")
pr4_ok = True
for name, ratio in MASS_RATIOS.items():
    mu = ratio / MP_ME
    n_i = MP_ME / ratio
    as_over_xi_direct = (2.0 / 3.0) * n_i ** 5 * mu ** 2
    ok = as_over_xi_direct > 1e50
    pr4_ok = pr4_ok and ok
    print(f"  {name}: a_s^Born/xi = {as_over_xi_direct:.3e}（远超 1：Born 公式自破，深势极限 a_s -> xi）")
print("  物理档结论：a_s 取几何支 ~ xi（硬球饱和）；u_kl 从描述项升为承重项")
VERDICT["PR4"] = pr4_ok
print(f"  [PR4] 判定：{'通过' if pr4_ok else '未通过'}")

# ----------------------------------------------------------------------
# PR5. 稳健性与有牙
# ----------------------------------------------------------------------
print("\n[PR5] 稳健性（口径变体）与恒真反例（判据有牙）")
pr5_ok = True
for fac, tag in ((2.0 * math.pi, "xi x 2pi"), (1.0 / (2.0 * math.pi), "xi / 2pi")):
    n_v = MP_ME * fac
    r2_v = R2(n_v, mu_e)   # 完整原式（n 变、m 不变）
    ok = r2_v > 1e50
    pr5_ok = pr5_ok and ok
    print(f"  口径变体 {tag}: R_2 = {r2_v:.3e}（仍 >> 1，方向不翻）")
# 恒真反例：模型档宽度
n_model = 2.25
r2_model = R2(n_model, mu_e)
r3_model = R3(n_model, mu_e)
counter_ok = r2_model < 1.0 and r3_model < 1.0
pr5_ok = pr5_ok and counter_ok
print(f"  恒真反例（n = 2.25 模型档）：R_2 = {r2_model:.3e}、R_3 = {r3_model:.3e}（< 1，"
      "判定翻回 Born 成立：判据有牙，改掉就 FAIL）")
VERDICT["PR5"] = pr5_ok
print(f"  [PR5] 判定：{'通过' if pr5_ok else '未通过'}")

# ----------------------------------------------------------------------
# 出图
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2))
ax = axes[0]
n_arr = np.logspace(0, 23, 400)
r2_arr = 4.0 * math.pi ** 2 * n_arr ** 5 * mu_e ** 2
r3_arr = n_arr ** 5 * mu_e
ax.loglog(n_arr, r2_arr, "-", lw=1.6, label=r"$R_2 = 4\pi^2 n^5 (m/m_P)^2$")
ax.loglog(n_arr, r3_arr, "--", lw=1.6, label=r"$R_3 = n^5 (m/m_P)$")
ax.axhline(1.0, color="k", lw=1.0, ls=":", label="判据线 R = 1")
ax.axvspan(2.25, 2.91, color="gray", alpha=0.25, label="模型档宽度 2.25 至 2.91")
ax.axvline(MP_ME, color="C3", lw=1.4, label=f"物理档 $n = m_P/m_e$ = {MP_ME:.3e}")
ax.axvline(n_star_3, color="C2", lw=0.9, ls="-.", label=f"$n^*_{{R_3}}$ = {n_star_3:.2e}")
ax.set_xlabel(r"缺陷宽度 $n = \xi/l_P$（格）")
ax.set_ylabel("Born 判据比值 R")
ax.set_title("Born 判据随宽度的翻转（电子档）：模型档成立，物理档失效")
ax.legend(fontsize=7.5, loc="upper left")
ax.grid(True, which="both", alpha=0.25)

ax = axes[1]
names = list(MASS_RATIOS.keys())
xs = np.arange(len(names))
r2_vals = [res_pr1[nm][1] for nm in names]
r3_vals = [res_pr1[nm][2] for nm in names]
ax.semilogy(xs - 0.18, r2_vals, "o", ms=7, label=r"$R_2 = 4\pi^2(m_P/m)^3$")
ax.semilogy(xs + 0.18, r3_vals, "s", ms=7, label=r"$R_3 = (m_P/m)^4$")
ax.axhline(1.0, color="k", lw=1.0, ls=":")
ax.set_xticks(xs)
ax.set_xticklabels(names)
ax.set_ylabel("Born 判据比值 R")
ax.set_title("物理档四档粒子：R 全部远超 1（Born 失效）")
ax.legend(fontsize=9)
ax.grid(True, which="both", alpha=0.25)

fig.suptitle("Born 判据物理档重估轮（第 624 号）：档位翻转与硬核执行者重判", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(IMG_PATH, dpi=150)
print(f"\n图已存：{IMG_PATH}")

# ----------------------------------------------------------------------
# 总判定
# ----------------------------------------------------------------------
print("\n" + "=" * 76)
print("总判定")
print("=" * 76)
npass = 0
for k in ("P0", "PR1", "PR2", "PR3", "PR4", "PR5"):
    ok = VERDICT[k]
    npass += int(ok)
    print(f"  {k}：{'通过' if ok else '未通过'}")
print(f"  合计：{npass}/6")
print("\n结论要点：物理档宽度（第 622 号恒等式 n = m_P/m）代入后 Born 判据翻转：")
print(f"  电子 R_2 = {res_pr1['电子'][1]:.3e}、R_3 = {res_pr1['电子'][2]:.3e}；")
print("  a_s 取几何支 ~ xi（硬球饱和），硬核执行者为强投影势 u_kl（档位标注：模型档相反）。")
