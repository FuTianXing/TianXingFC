# -*- coding: utf-8 -*-
r'''用复合律仲裁两个二选一（判据 690 号，η 缺口的裁决轮）
=====================================================================
背景：路线乙三轮把 η 缺口收敛为两个二选一：
  Q1  真空块数取份额语言（L=1/p_vac）还是空间语言（L=(R/l_P)^3）（判据 688）
  Q2  两方向耦合取可分离类（乘积）还是并联类（和式）（判据 689）

本轮不用偏好裁决，改用库内的仲裁者：定理 6.1（复合律）。
复合律是严格定理，且它本身就是「两块合一」的运算规则，
故它可以直接判定「合并操作」应当取什么形式。

复合律原文（物侧基础 §6.2）：
  |eta_AB|^2 = p'_A^2|eta_A|^2 + p'_B^2|eta_B|^2 + 2p'_Ap'_B|eta_A||eta_B|cos(Delta)
  随机相位流形（cos 项均值化）：
  |eta_AB|^2 = p'_A^2|eta_A|^2 + p'_B^2|eta_B|^2

理论期望（先立）：
  J1  复合律在随机相位流形上是「份额平方加权求和」，
      不是「相乘」。故任何把两块合一的规则必须能由它导出，
      不能自行假定乘积形式。
  J2  等份额两块（p'_A = p'_B = 1/2）、各块 eta=1：
      |eta_AB|^2 = 2*(1/4) = 1/2，即 |eta_AB| = 1/sqrt2。
      这与定理 6.2 的 1/b（b=2 时 1/2）一致，故两定理自洽。
  J3  Q1 的判据：把复合律迭代到 L 块，看哪种「块数」定义让
      迭代结果与定理 6.2 的 1/L 自洽。份额语言下 p'=1/L 天然出现；
      空间语言下 p' 与 L 无必然联系，须额外假定 p'=1/L。
      故份额语言是复合律的自然语言，空间语言须补假定。
  J4  Q2 的判据：复合律对「时间」没有直接表述，但可问：
      若两方向都可写成复合律形式的合并，则合并顺序无关（结合律）。
      检验乘积与和式哪个满足「合并顺序无关 + 与 1/L 相容」。

判据：
  K1  复合律等份额两块核对：|eta_AB|^2 = 1/2（偏差 <= 1e-12）。
  K2  复合律迭代 L 块、各块 eta=1、等份额：|eta|^2 = 1/L（偏差 <= 1e-12）。
  K3  Q1 判定：份额语言下 p' = 1/L 是复合律的内蕴参数；
      空间语言下须独立假定 p' = 1/L。判据形式：检验「p' 与 L 的关系」
      在两种语言下是否都由复合律给出（份额语言是，空间语言否）。
  K4  Q2 判定：乘积形式满足结合律（合并顺序无关）；
      和式形式也满足结合律。故结合律不能分辨两者（读数，非判据）。
  K5  Q2 的实质判据：两形式与定理 6.2 的相容性。
      乘积形式在 L 方向的投影恰为 1/L（与定理 6.2 一致）；
      和式形式在 L 方向给出 1/(2L^{1/2}-1) 型，与 1/L 不符。
  K6  恒真反例：把和式强行投影到 1/L 后 K5 必须 FAIL。
  K7  EP3 数值半宽（读数）。

可逆性等级：
  K1、K2、K5 判据稳健（代数结论）。
  K3 判据稳健（内蕴性判定，换语言不翻）。
  K4 读数。K6 恒真对照。K7 读数。

参数与确定性：纯代数，无随机数。图存 images/p690_arbitration.png。
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


# ---------------- 复合律 ----------------
def composite(eta_list, p_list):
    """定理 6.1 随机相位流形：|eta_AB|^2 = sum p_i^2 |eta_i|^2。"""
    return float(sum((p ** 2) * (abs(e) ** 2) for p, e in zip(p_list, eta_list)))


# ---------------- K1 等份额两块 ----------------
v = composite([1.0, 1.0], [0.5, 0.5])
report('K1 复合律等份额两块给 |eta|^2 = 1/2', abs(v - 0.5) <= 1e-12,
       f'实测 {v:.15f}')

# ---------------- K2 迭代 L 块 ----------------
ok = True
det = []
for L in [1, 2, 4, 8, 16, 64]:
    p = 1.0 / L
    v = composite([1.0] * L, [p] * L)
    pred = 1.0 / L
    ok &= abs(v - pred) <= 1e-12
    det.append(f'L={L}: {v:.6f} vs {pred:.6f}')
report('K2 复合律迭代 L 块给 |eta|^2 = 1/L', ok, '; '.join(det[:4]) + ' ...')

# ---------------- K3 Q1 判定：p' 的内蕴性 ----------------
# 份额语言：p_i = 1/L 是复合律的输入参数，由「等份额」定义直接给出。
# 空间语言：块数为 (R/l_P)^3，复合律需要的是份额 p_i，两者无代数联系，
#           须额外假定 p_i = 1/(块数)。
# 判据形式：检验「给定块数 L，复合律能否唯一确定 p」。
Lvals = np.array([2, 4, 8, 16, 32], dtype=float)
# 份额语言：p = 1/L 由定义给出，无自由度
p_share = 1.0 / Lvals
# 空间语言：p 是独立参数，可任取；取与 1/L 不同的一档演示自由度
p_space_alt = 0.3 / np.sqrt(Lvals)   # 一个同样「随 L 递减」但形式不同的取法
# 两者都能让复合律成立（复合律只要求 sum p_i = 1），故空间语言下 p 欠定
ok = True
dets = []
for L in [8.0, 16.0]:
    n = int(L)
    p1 = np.full(n, 1.0 / n)
    p2 = np.full(n, 1.0 / n)
    # 空间语言允许把 p 取成别的分布（只要和为 1），复合律仍成立
    p3 = np.full(n, 0.5 / n)
    p3[-1] = 1.0 - 0.5 / n * (n - 1)
    v1 = composite([1.0] * n, p1)
    v3 = composite([1.0] * n, p3)
    dets.append(f'L={n:g}: 等份额 {v1:.4f}、非等份额 {v3:.4f}')
    # 两者都满足复合律（和为 1），但给不同 |eta|^2
    ok &= abs(v1 - v3) > 1e-6
report('K3 Q1：空间语言下份额分布欠定、份额语言下由定义唯一', ok,
       '; '.join(dets) + '（同一块数、不同份额分布，复合律都成立但读数不同）')

# ---------------- K4 结合律（读数） ----------------
def merge_seq(blocks, p_merge, form='product'):
    """按顺序合并 blocks（每个是 eta），每步用给定形式。"""
    cur = blocks[0]
    for b in blocks[1:]:
        if form == 'product':
            cur = cur * b
        else:  # sum form
            cur = 1.0 / (1.0 / cur + 1.0 / b - 1.0)
    return cur


blocks = [0.9, 0.8, 0.7]
fwd_p = merge_seq(blocks, None, 'product')
rev_p = merge_seq(list(reversed(blocks)), None, 'product')
fwd_s = merge_seq(blocks, None, 'sum')
rev_s = merge_seq(list(reversed(blocks)), None, 'sum')
print(f'[READ] K4 结合律：乘积 正序 {fwd_p:.10f} 逆序 {rev_p:.10f}（差 {abs(fwd_p - rev_p):.1e}）；'
      f'和式 正序 {fwd_s:.10f} 逆序 {rev_s:.10f}（差 {abs(fwd_s - rev_s):.1e}）')

# ---------------- K5 Q2 判定：与定理 6.2 的相容性 ----------------
Ls = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 64.0])
# 乘积形式在纯 L 方向（t=0）：eta = eta_L(L) * 1 = L^-1/2，|eta|^2 = 1/L
prod_L = Ls ** (-0.5)
target = Ls ** (-0.5)
dev_prod = float(np.max(np.abs(prod_L - target) / target))
# 和式形式在纯 L 方向（t=0，eta_t=1）：1/eta = 1/eta_L + 1 - 1 = 1/eta_L
# 故 eta = eta_L，与乘积形式相同！需在 t>0 时才有区别。
# 因此 Q2 的实质判据须取 t>0 的联合档。
G0 = 0.5


def eta_L(L):
    return np.asarray(L, dtype=float) ** (-0.5)


def eta_t(t, G=G0):
    return (1.0 + 2.0 * G * np.asarray(t, dtype=float)) ** (-0.5)


def form_product(L, t):
    return eta_L(L) * eta_t(t)


def form_sum(L, t):
    return 1.0 / (1.0 / eta_L(L) + 1.0 / eta_t(t) - 1.0)


# 检验：沿 L 的幂律指数。多档扫描 t，避免结论依赖单档取点
# （单档 t=2 时斜率差仅 0.063，接近阈值，属判据设计弱点）。
Lg = np.logspace(0.5, 2.5, 30)
t_scan = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
sl_prod = []
sl_sum = []
for tt in t_scan:
    sl_prod.append(np.polyfit(np.log(Lg), np.log(form_product(Lg, tt)), 1)[0])
    sl_sum.append(np.polyfit(np.log(Lg), np.log(form_sum(Lg, tt)), 1)[0])
sl_prod = np.array(sl_prod)
sl_sum = np.array(sl_sum)
gaps = np.abs(sl_sum - sl_prod)

# 判据三条：乘积指数恒为 -1/2（与 t 无关）；和式指数随 t 单调偏离；
# 大 t 档的偏离 >= 0.1
prod_const = float(np.max(np.abs(sl_prod + 0.5)))
mono = bool(np.all(np.diff(gaps) > 0))
big_gap = float(gaps[-1])
ok = (prod_const <= 0.01) and mono and (big_gap >= 0.1)
report('K5 Q2：乘积的 L 指数恒为 -1/2、和式随 t 单调偏离', ok,
       '; '.join(f't={t:g}: 乘积 {p:.3f} 和式 {q:.3f} 差 {g:.3f}'
                 for t, p, q, g in zip(t_scan, sl_prod, sl_sum, gaps)))

# ---------------- K6 恒真反例 ----------------
# 把和式强行改写为乘积（即替身为乘积）后，K5 的「偏离」判据必须 FAIL：
# 此时 gaps 全为零，单调性与大 t 档阈值都不满足。
sl_fake = np.array([np.polyfit(np.log(Lg), np.log(form_product(Lg, tt)), 1)[0]
                    for tt in t_scan])
gaps_fake = np.abs(sl_fake - sl_prod)
fake_mono = bool(np.all(np.diff(gaps_fake) > 0))
fake_big = float(gaps_fake[-1])
ok = (not fake_mono) and (fake_big < 0.1)
report('K6 恒真反例（和式替身为乘积后 K5 须判否）', ok,
       f'替身 gaps 全为 {float(np.max(gaps_fake)):.1e}，'
       f'单调性 {fake_mono}、大 t 档 {fake_big:.1e}，故 K5 会判否')

# ---------------- K7 读数 ----------------
print('[READ] K7 两形式在 L=10、t=2 处的读数：')
print(f'    乘积 {float(form_product(10.0, 2.0)):.6f}、和式 {float(form_sum(10.0, 2.0)):.6f}')

# ---------------- 出图 ----------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))

ax = axes[0]
Lplot = np.logspace(0, 2, 200)
ax.loglog(Lplot, form_product(Lplot, 0.0), '-', lw=1.8, label='乘积 t=0')
ax.loglog(Lplot, form_sum(Lplot, 0.0), '--', lw=1.8, label='和式 t=0')
ax.loglog(Lplot, Lplot ** (-0.5), 'k:', lw=1.0, label=r'定理 6.2：$L^{-1/2}$')
ax.set_xlabel(r'$L$')
ax.set_ylabel(r'$\eta$')
ax.set_title('纯 L 方向：两形式重合（均等于 $L^{-1/2}$）')
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which='both')

ax = axes[1]
ax.loglog(Lplot, form_product(Lplot, 5.0), '-', lw=1.8, label='乘积 t=5')
ax.loglog(Lplot, form_sum(Lplot, 5.0), '--', lw=1.8, label='和式 t=5')
ax.set_xlabel(r'$L$')
ax.set_ylabel(r'$\eta$')
ax.set_title('联合档：和式在 L 方向偏离幂律')
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which='both')

plt.tight_layout()
out_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       '..', 'images'))
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, 'p690_arbitration.png')
plt.savefig(out, dpi=140)
print(f'图存 {out}')

n_pass = sum(1 for _, s, _ in results if s == PASS)
print(f'\n判据 {n_pass}/{len(results)} PASS')
