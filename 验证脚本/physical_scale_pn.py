# -*- coding: utf-8 -*-
r"""
physical_scale_pn.py  （第 622 号，第二版）

P1-40 物理档评估轮：定稿档缺陷宽度恒等式核验与 PN 势垒的物理档压制。

背景。P1-40（钉扎-洛伦兹张力）的读数历来在模型档（K ~ 0.005）。第 616 号
给出 PN 势垒标定律 r ~ e^{-pi^2 w}（w 为第 616 号约定下的缺陷宽度参数，
格），安全门槛为 w >= 4.69（r <= 1e-15）。本轮的出发点是第 479 条注记登记的
一个附读数：定稿档软模关联长度 xi = 2.3892e22 格（该注记未识别其解析身份）。

核心恒等式（解析）。代入三条已裁决的体系关系：
  M6 字典链：  kappa_phi t_P = (m_delta/m_P)^2 / (6 pi)
  M-E 锚：     m_delta = m_e（缺口模式 = 电子）
  公理刚度：   k_b = 2 pi（u 场）
  势系数：     K = 3 kappa_phi t_P（MB-8/MB-9）
  关联长度：   xi = 1/(2 pi) sqrt(k_b/K)
化简：
  xi = 1/(2 pi) * sqrt(2 pi / (3 kappa_phi t_P))
     = 1/(2 pi) * sqrt(2 pi * 6 pi / (3 (m_e/m_P)^2))
     = 1/(2 pi) * 2 pi * (m_P/m_e)
     = m_P / m_e
即：定稿档的缺陷宽度（第 616 号约定，以格距为单位）精确等于普朗克质量与
电子质量之比。物理含义：缺陷的物理宽度精确等于电子的约化康普顿波长
  xi * l_P = l_P * m_P/m_e = hbar/(m_e c)。

判据（先立判据，再实现）：
  K1 恒等式核验（CODATA）：xi 与 m_P/m_e 相对差 <= 1e-14
  K2 登记值对比：xi(登记 K) 与 m_P/m_e 相对差 <= 1e-4（差来源为 K 舍入）
  K3 物理宽度：xi * l_P 对电子约化康普顿波长之比 = 1（1e-5 内）
  K4 物理档势垒压制（对数域）：SG 口径 ln r < -1e23；phi^4 口径同量级
  K5 安全裕度：物理档 r 对实验约束 1e-15 与 1e-20 的对数裕度 > 1e22
  K6 载体对照（防恒真）：sigma 壁测试档 r = 0.1143（第 616 号 J7 实测），
     与物理档结论相差 > 1e20 量级
  K7 锚替代稳健性：锚取孤子口径（m_delta -> (pi^2/2) m_e）时 xi' =
     2 m_P/(pi^2 m_e)，压制依然极端

数值处理。w ~ 1e22 下 sinh(pi^2 w) 与 e^{-pi^2 w} 均溢出双精度，全部
在对数域计算：ln r = ln C + ln(前因子多项式) - ln(w) - pi^2 w + ln 2。

口径说明（重要）。第 616 号的宽度参数 w 即 SG 宽度参数 xi_phi = sqrt(k_phi/K)
（等价于 u 场关联长度 xi = (1/(2 pi)) sqrt(k_b/K)，两约定同值）。本轮物理档
取 w = xi = m_P/m_e = 2.389e22 格。phi^4 口径的物理宽度按同一 w 评估（两条
公式共享指数率 -pi^2 w，前因子差 w^2），结论量级不因口径而异。
"""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

IMG = r"D:\software\TianXingFC\images\physical_scale_pn.png"
rows = []

# ==================== 常数与体系定稿值 ====================
MP = 2.176434e-8          # 普朗克质量（kg，CODATA）
ME = 9.1093837015e-31     # 电子质量（kg，CODATA）
KB_KB = 2.0 * math.pi     # 公理刚度 k_b（u 场）
K_REG = 2.788142e-46      # 体系定稿势系数（登记值）
KP_REG = 9.293805e-47     # kappa_phi t_P（登记值）
C_FQ = 3.0599             # 第 616 号 phi^4 标定系数
C_SG = 1.7930             # 第 616 号 SG 标定系数

ratio_codata = MP / ME

print("=" * 78)
print("第 622 号：P1-40 物理档评估轮")
print("=" * 78)

# ==================== K1 恒等式核验 ====================
print("\n[K1] 恒等式核验（CODATA）")
kp_codata = (ME / MP) ** 2 / (6.0 * math.pi)
K_codata = 3.0 * kp_codata
xi_codata = 1.0 / (2.0 * math.pi) * math.sqrt(KB_KB / K_codata)
rel = abs(xi_codata - ratio_codata) / ratio_codata
ok1 = rel <= 1e-14
print("  kappa_phi t_P（CODATA 式）= %.12e" % kp_codata)
print("  K（CODATA 式）            = %.12e" % K_codata)
print("  xi（CODATA 式）           = %.12e" % xi_codata)
print("  m_P/m_e                   = %.12e" % ratio_codata)
print("  相对差                    = %.3e  %s" % (rel, "PASS" if ok1 else "FAIL"))
rows.append(("K1", "恒等式 xi = m_P/m_e（CODATA）", "%.3e" % rel, "<= 1e-14", ok1))

# ==================== K2 登记值对比 ====================
print("\n[K2] 登记值对比")
xi_reg_calc = 1.0 / (2.0 * math.pi) * math.sqrt(KB_KB / K_REG)
rel_reg = abs(xi_reg_calc - ratio_codata) / ratio_codata
ok2 = rel_reg <= 1e-4
print("  xi（登记 K 重算） = %.10e" % xi_reg_calc)
print("  与 m_P/m_e 相对差 = %.3e  %s" % (rel_reg, "PASS" if ok2 else "FAIL"))
rel_k = abs(KP_REG - kp_codata) / kp_codata
print("  （差异来源：kappa_phi t_P 登记值对 CODATA 相对差 %.2e）" % rel_k)
rows.append(("K2", "登记值与 m_P/m_e 对比", "%.3e" % rel_reg, "<= 1e-4", ok2))

# ==================== K3 物理宽度 ====================
print("\n[K3] 物理宽度")
xi_phys = ratio_codata          # 物理档宽度（第 616 号约定，格）
lP = 1.616255e-35               # 普朗克长度（m）
width_m = xi_phys * lP
lam_c = 3.8615926796e-13        # 电子约化康普顿波长（m）
print("  xi（物理档宽度）    = %.6e 格" % xi_phys)
print("  物理宽度 xi * l_P   = %.6e m" % width_m)
print("  电子约化康普顿波长   = %.6e m" % lam_c)
print("  两者之比            = %.10f" % (width_m / lam_c))
ok3 = abs(width_m / lam_c - 1.0) <= 1e-5
rows.append(("K3", "物理宽度 = 约化康普顿波长", "%.10f" % (width_m / lam_c), "= 1（1e-5 内）", ok3))

# ==================== K4 物理档势垒压制（对数域） ====================
print("\n[K4] 物理档势垒压制（对数域，w = %.4e 格）" % xi_phys)

def ln_r_fq(w):
    p2 = math.pi ** 2
    return (math.log(C_FQ) + math.log(p2) + math.log(p2 * w * w + 1.0)
            + math.log(3.0 * p2 * w * w + 2.0) - math.log(15.0)
            - math.log(w) - p2 * w + math.log(2.0))

def ln_r_sg(w):
    p2 = math.pi ** 2
    return (math.log(C_SG) + math.log(p2) + math.log(4.0 * p2 * w * w + 1.0)
            - math.log(18.0) - math.log(w) - p2 * w + math.log(2.0))

ln_sg = ln_r_sg(xi_phys)
ln_fq = ln_r_fq(xi_phys)
log10_sg = ln_sg / math.log(10.0)
log10_fq = ln_fq / math.log(10.0)
print("  SG 口径  ：ln r = %.6e，log10 r = %.6e" % (ln_sg, log10_sg))
print("  phi^4 口径：ln r = %.6e，log10 r = %.6e" % (ln_fq, log10_fq))
print("  即 r ~ 10^{%.4e}（SG 口径）" % log10_sg)
ok4 = ln_sg < -1e23
rows.append(("K4", "物理档 ln r（SG 口径）", "%.6e" % ln_sg, "< -1e23", ok4))
rows.append(("K4", "物理档 ln r（phi^4 口径）", "%.6e" % ln_fq, "< -1e23", ln_fq < -1e23))

# ==================== K5 安全裕度 ====================
print("\n[K5] 安全裕度（对实验约束）")
for thr, tag in [(1e-15, "1e-15"), (1e-20, "1e-20")]:
    ln_margin = math.log(thr) - ln_sg
    log10_margin = ln_margin / math.log(10.0)
    print("  对 %s 的裕度：log10 = %.6e（r 比约束小 10^{%.4e} 倍）"
          % (tag, log10_margin, log10_margin))
    rows.append(("K5", "裕度 log10（对 %s）" % tag, "%.6e" % log10_margin, "> 1e22", log10_margin > 1e22))

# ==================== K6 载体对照（防恒真） ====================
print("\n[K6] 载体对照（防恒真）")
r_test = 0.1143   # 第 616 号 J7 实测（sigma 壁测试档 w = 0.791）
gap_log10 = (math.log(r_test) - ln_sg) / math.log(10.0)
print("  sigma 壁测试档（w = 0.791）：r = %.4f（第 616 号 J7 实测）" % r_test)
print("  物理档结论与该对照相差 log10 = %.4e（物理档更小）" % gap_log10)
ok6 = r_test > 0.01 and gap_log10 > 1e20
print("  判据：对照档 r > 0.01 且与物理档结论相差 > 1e20 量级  %s" % ("PASS" if ok6 else "FAIL"))
rows.append(("K6", "载体对照 r（sigma 壁测试档）", "%.4f" % r_test, "> 0.01 且差 > 1e20", ok6))

# ==================== K7 锚替代稳健性（极端带） ====================
print("\n[K7] 锚替代稳健性（极端带）")
# MB 线两锚口径差为 pi^2 量级（MB-4/MB-8），故取质量因子的极端带 pi^{+2} 与 pi^{-2}
xi_hi = (math.pi ** 2) * ratio_codata
xi_lo = ratio_codata / (math.pi ** 2)
ln_hi = ln_r_sg(xi_hi)
ln_lo = ln_r_sg(xi_lo)
print("  锚质量 × pi^2（宽度 × 1/pi^2）：xi = %.4e 格，ln r = %.4e" % (xi_hi, ln_hi))
print("  锚质量 / pi^2（宽度 × pi^2）  ：xi = %.4e 格，ln r = %.4e" % (xi_lo, ln_lo))
ok7 = ln_hi < -1e21 and ln_lo < -1e21
print("  判据：极端带两端 ln r 均 < -1e21  %s" % ("PASS" if ok7 else "FAIL"))
rows.append(("K7", "锚替代极端带 ln r（两端）", "%.2e / %.2e" % (ln_hi, ln_lo), "均 < -1e21", ok7))

# ==================== K8 口径稳健性 ====================
print("\n[K8] 口径稳健性（k_b 归一与宽度约定）")
# 口径甲：k_b = 1（479 注记口径甲），xi_jia = (1/(2 pi)) sqrt(1/K_sys)
xi_jia = 1.0 / (2.0 * math.pi) * math.sqrt(1.0 / K_REG)
ln_jia = ln_r_sg(xi_jia)
# 宽度约定变体：w = 2 pi xi（若物理宽度取 2 pi 倍关联长度）
w_2pi = 2.0 * math.pi * ratio_codata
ln_2pi = ln_r_sg(w_2pi)
print("  口径甲（k_b = 1）      ：xi = %.4e 格，ln r = %.4e" % (xi_jia, ln_jia))
print("  宽度约定变体（w = 2 pi xi）：w = %.4e 格，ln r = %.4e" % (w_2pi, ln_2pi))
ok8 = ln_jia < -1e21 and ln_2pi < -1e21
print("  判据：两变体 ln r 均 < -1e21（结论不依赖口径）  %s" % ("PASS" if ok8 else "FAIL"))
rows.append(("K8", "口径变体 ln r（两端）", "%.2e / %.2e" % (ln_jia, ln_2pi), "均 < -1e21", ok8))

# ==================== 图 ====================
fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.7))

wd = np.linspace(0.5, 6.0, 400)
ax = axes[0]
ax.semilogy(wd, [math.exp(ln_r_sg(w)) for w in wd], "-", color="tab:blue", lw=1.4,
            label="SG 标定式")
ax.semilogy(wd, [math.exp(ln_r_fq(w)) for w in wd], "-", color="tab:red", lw=1.4,
            label="phi^4 标定式")
ax.axhline(1e-15, ls="--", color="gray", lw=1.0)
ax.text(0.55, 2e-15, "实验约束 1e-15", fontsize=8, color="gray")
ax.axvline(4.69, color="tab:red", ls=":", lw=1.0)
ax.text(4.75, 1e-6, "phi^4 门槛 4.69", fontsize=8, color="tab:red", rotation=90)
ax.axvline(4.08, color="tab:blue", ls=":", lw=1.0)
ax.text(3.4, 1e-6, "SG 门槛 4.08", fontsize=8, color="tab:blue", rotation=90)
ax.set_xlim(0.5, 6.0)
ax.set_ylim(1e-16, 1.0)
ax.set_xlabel("缺陷宽度 w（格）")
ax.set_ylabel("r = ΔE_PN / M")
ax.set_title("(a) 模型档区域：势垒与安全门槛")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

ax = axes[1]
wl = np.logspace(0, 23.0, 400)
lnln = [math.log(-ln_r_sg(w)) / math.log(10.0) for w in wl]
ax.plot(np.log10(wl), lnln, "-", color="tab:blue", lw=1.6, label="SG 标定式（对数域）")
ax.plot([math.log10(xi_phys)], [math.log10(-ln_sg)], "o", color="tab:red", ms=8,
        label="物理档 w = %.3e 格" % xi_phys)
ax.annotate("物理档\nln r = %.2e" % ln_sg, (math.log10(xi_phys), math.log10(-ln_sg)),
            textcoords="offset points", xytext=(-120, 4), fontsize=8, color="tab:red")
ax.axvline(math.log10(4.08), color="gray", ls=":", lw=1.0)
ax.text(math.log10(4.08) + 0.15, 2.0, "安全门槛 4.08", fontsize=8, color="gray", rotation=90)
ax.set_xlabel("log10(w)（格）")
ax.set_ylabel("log10(-ln r)")
ax.set_title("(b) 全档：势垒对数压制的线性律")
ax.legend(fontsize=8, loc="upper left")
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(IMG, dpi=150)
print("\n图已保存：%s" % IMG)

# ==================== 汇总 ====================
print("\n" + "=" * 78)
print("判据汇总")
print("=" * 78)
for tag, item, meas, exp, ok in rows:
    print("  [%s] %-38s 实测 %-26s 期望 %-24s %s"
          % (tag, item, meas, exp, "PASS" if ok else "FAIL"))
print("\n全部通过：%s" % ("是" if all(r[4] for r in rows) else "否"))
print("\n核心读数：")
print("  xi = m_P/m_e = %.6e 格（恒等式，机器精度）" % xi_phys)
print("  物理宽度 = 电子约化康普顿波长（比值 %.8f）" % (width_m / lam_c))
print("  物理档 ln r = %.6e（SG 口径），即 r ~ 10^{%.4e}" % (ln_sg, log10_sg))
print("  对实验约束 1e-15 的裕度 log10 = %.6e"
      % ((math.log(1e-15) - ln_sg) / math.log(10.0)))
print("  载体对照（sigma 壁测试档）r = %.4f，与物理档相差 10^{%.4e} 量级"
      % (r_test, gap_log10))
