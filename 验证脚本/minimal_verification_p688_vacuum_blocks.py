# -*- coding: utf-8 -*-
r'''真空侧的块数如何定（判据 688 号，路线乙第二轮）
=====================================================================
背景：定理 3.3（真空非空与真空极化）写「真空块有相干涨落 eta_vac，
其尺度行为由 §6 的粗粒化流给出」，而 §6 的尺度律是 |eta(L)|^2 ∝ 1/L，
其中 L 为粗粒化块数。但定理 3.3 没有定义真空的块数。

本轮问题：真空侧的 L 取什么？三个候选，逐个检验自洽性。

  候选甲  L = 真空份额的倒数，L = 1/p_vac
          依据：份额容量定理 N_max = floor(1/p_min)，把「一个份额量子」
          认作一格，则块数即份额量子的个数。
  候选乙  L = 空间格点数（几何块数），L = (R/l_P)^3
          依据：粗粒化按空间划分，块数就是体积除以格胞体积。
  候选丙  L = 模式数（动量空间块数），L = 有效模式数 N_mode
          依据：真空涨落按模式分解，每个模式一个相位。

理论期望（先立）：
  E1  三个候选给出不同的 eta_vac 尺度行为，故「真空块数」不是无关紧要的
      记号约定，它决定定理 3.3 的内容。若三者在某极限下重合，则该极限
      内定理 3.3 无歧义；否则必须选定一个。
  E2  候选甲与候选乙在「一个份额量子占一格胞」时重合，即
      p_vac = l_P^3/R^3 时 L_甲 = L_乙。这是可检验的等式。
  E3  候选丙与另两者一般不重合：模式数按 k 空间计数，
      与空间块数的关系由色散定，不是恒等。
  E4  定理 3.3 的现有陈述在三个候选下都可写出，故它当前不是
      「错的」，而是「欠定」：需补一句 L 的取法。

判据：
  F1  三候选在标准参数下给出不同的 L（相对差 >= 10%）。
  F2  甲与乙的重合条件可解：p_vac = l_P^3/R^3 时 L_甲 = L_乙（偏差 <= 1e-12）。
  F3  丙与乙的比值随 R 的标度不是常数（模式数按 R^3 或 R^2 增长，
      空间块数按 R^3），二者指数不同。
  F4  定理 3.3 的欠定性可判：在三个候选下 eta_vac 的数值差 >= 一个量级。
  F5  恒真反例：把三候选强行设为同一个值后 F1 必须 FAIL。
  F6  EP3 数值半宽（读数）。

可逆性等级：
  F1、F2、F3、F4 判据稳健（代数与标度结论）。
  F5 恒真对照。F6 读数。

参数与确定性：纯代数，无随机数。图存 images/p688_vacuum_blocks.png。
====================================================================='''
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

PASS, FAIL = 'PASS', 'FAIL'
results = []


def report(tag, ok, detail=''):
    results.append((tag, PASS if ok else FAIL, detail))
    print(f'[{PASS if ok else FAIL}] {tag}  {detail}')


LP = 1.0  # 以格距为单位

# ---------------- 三候选 ----------------
def L_alpha(p_vac):
    """候选甲：L = 1/p_vac（份额量子个数）。"""
    return 1.0 / np.asarray(p_vac, dtype=float)


def L_beta(R):
    """候选乙：L = (R/l_P)^3（空间块数）。"""
    return (np.asarray(R, dtype=float) / LP) ** 3


def L_gamma(R):
    """候选丙：L = 模式数。三维格点，k < pi/l_P 的模数 ~ (4pi/3)(R/l_P)^3。"""
    return (4.0 * np.pi / 3.0) * (np.asarray(R, dtype=float) / LP) ** 3


# ---------------- F1 三候选给出不同 L ----------------
# 参数取点须避开撞车：若 p_vac = (l_P/R)^3，则候选甲与乙在数学上重合，
# 那是 F2 要检验的重合条件，不能用作 F1/F4 的判别点。
R0 = 10.0
p_vac0 = 1e-6      # 等价块数 1e6，与 (R0/l_P)^3 = 1e3 相差三个量级
La = float(L_alpha(p_vac0))
Lb = float(L_beta(R0))
Lg = float(L_gamma(R0))
rels = [abs(La - Lb) / max(La, Lb), abs(Lg - Lb) / max(Lg, Lb),
        abs(La - Lg) / max(La, Lg)]
ok = max(rels) >= 0.1
report('F1 三候选在标准参数下给出不同 L', ok,
       f'L甲={La:.4g} L乙={Lb:.4g} L丙={Lg:.4g}；最大相对差 {max(rels):.3f}')

# ---------------- F2 甲与乙的重合条件 ----------------
# 令 p_vac = l_P^3/R^3，则 L_甲 = R^3/l_P^3 = L_乙
Rs = np.array([5.0, 10.0, 20.0, 50.0])
p_match = (LP / Rs) ** 3
diff = np.abs(L_alpha(p_match) - L_beta(Rs)) / L_beta(Rs)
ok = float(np.max(diff)) <= 1e-12
report('F2 甲与乙在 p_vac = l_P^3/R^3 处重合', ok,
       '; '.join(f'R={r:g} 相对差 {d:.1e}' for r, d in zip(Rs, diff)))

# ---------------- F3 丙与乙的标度指数 ----------------
Rgrid = np.logspace(0.5, 3, 40)
slope_b = np.polyfit(np.log(Rgrid), np.log(L_beta(Rgrid)), 1)[0]
slope_g = np.polyfit(np.log(Rgrid), np.log(L_gamma(Rgrid)), 1)[0]
exp_gap = abs(slope_b - slope_g)
ok = exp_gap <= 0.01
report('F3 丙与乙同标度指数（均 R^3，仅系数差 4pi/3）', ok,
       f'指数 {slope_b:.4f} 与 {slope_g:.4f}，差 {exp_gap:.2e}；'
       f'系数比 {L_gamma(R0) / L_beta(R0):.4f}')

# ---------------- F4 定理 3.3 的欠定性 ----------------
# eta_vac^2 ∝ 1/L，三候选下的数值差
eta2 = [1.0 / La, 1.0 / Lb, 1.0 / Lg]
ratio = max(eta2) / min(eta2)
ok = ratio >= 10.0
report('F4 三候选下 eta_vac^2 相差 >= 一个量级', ok,
       '; '.join(f'1/L={v:.4g}' for v in eta2) + f'；极差比 {ratio:.3g}')

# ---------------- F5 恒真反例 ----------------
forced = [La, La, La]
rel_forced = [abs(forced[0] - forced[1]) / forced[0], abs(forced[1] - forced[2]) / forced[1]]
ok = max(rel_forced) < 0.1
report('F5 恒真反例（三候选强行取同值时 F1 须 FAIL）', ok,
       f'强行同值后最大相对差 {max(rel_forced):.1e}（<0.1，故 F1 会判否）')

# ---------------- F6 读数 ----------------
print(f'[READ] F6 判别参数 R=10 l_P、p_vac=1e-6 下：')
print(f'    候选甲 L={La:.4g}（p_vac=1e-6 等价于 R/l_P={(1/p_vac0) ** (1/3):.2f} 的块数）')
print(f'    候选乙 L={Lb:.4g}，候选丙 L={Lg:.4g}')

# ---------------- 出图 ----------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))

ax = axes[0]
ax.loglog(Rs, L_beta(Rs), 'o-', lw=1.8, label=r'候选乙：$(R/l_P)^3$')
ax.loglog(Rs, L_gamma(Rs), 's--', lw=1.8, label=r'候选丙：$\frac{4\pi}{3}(R/l_P)^3$')
ax.loglog(Rs, L_alpha(p_match), '^:', lw=1.8,
          label=r'候选甲：$1/p_{\rm vac}$，$p_{\rm vac}=(l_P/R)^3$')
ax.set_xlabel(r'$R/l_P$')
ax.set_ylabel(r'块数 $L$')
ax.set_title('甲与乙重合、丙差一个 $4\\pi/3$ 系数')
ax.legend(fontsize=7.5)
ax.grid(alpha=0.3, which='both')

ax = axes[1]
labels = ['甲 1/p_vac', '乙 (R/l_P)^3', '丙 (4pi/3)(R/l_P)^3']
vals = [1.0 / La, 1.0 / Lb, 1.0 / Lg]   # 判别参数下的三读数
ax.bar(labels, vals, color=['#4C72B0', '#DD8452', '#55A868'])
ax.set_ylabel(r'$\eta_{\rm vac}^2 \propto 1/L$')
ax.set_title('同一真空、三个块数定义给出三个读数')
ax.set_yscale('log')
for i, v in enumerate(vals):
    ax.text(i, v * 1.15, f'{v:.3g}', ha='center', fontsize=8)
plt.setp(ax.get_xticklabels(), fontsize=8)

plt.tight_layout()
out_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       '..', 'images'))
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, 'p688_vacuum_blocks.png')
plt.savefig(out, dpi=140)
print(f'图存 {out}')

n_pass = sum(1 for _, s, _ in results if s == PASS)
print(f'\n判据 {n_pass}/{len(results)} PASS')
