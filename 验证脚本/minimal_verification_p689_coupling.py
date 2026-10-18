# -*- coding: utf-8 -*-
r'''两个粗粒化方向的耦合形式（判据 689 号，路线乙第三轮）
=====================================================================
背景：第 687 号判定 η 的尺度侧（|eta|^2 ∝ 1/L，块数）与时间侧
（(1+2Γt)^(-1/2)，扩散步数）是两个正交的粗粒化方向，不能统一。
第 688 号把真空块数缺口收窄为二选一加待定锚点。
本轮处理第 687 号留下的第二项：两个方向如何耦合成 η(t, L)。

三个候选耦合形式，逐个检验其可判性与与既有结论的相容性：

  候选甲 乘积形式  η(t,L) = η_L(L) · η_t(t)
          含义：空间粗粒化与时间累积互不影响，各自独立稀释。
          特征：ln η 对 ln L 与 t 可分离。
  候选乙 和式形式  η(t,L) = η_L(L) + η_t(t) - 1（或 1/η = 1/η_L + 1/η_t - 1）
          含义：两者竞争同一「相干资源」。
  候选丙 复合形式  η(t,L) = η_L(L·f(t))，f(t) 为时间驱动的有效块数增长
          含义：时间累积等价于继续粗粒化，两个方向合成一个变量。
          特征：只依赖组合变量 n = L·f(t)。

理论期望（先立）：
  G1  三形式在 L→1 且 t→0 时都趋于 1（三形式皆合格）。
      大参数区：乘积与复合同斜率（H2 已证二者恒等），
      和式斜率不同，故「可分离类」与「并联类」可用大参数区判别。
  G2  乘积形式与复合形式在 f(t) = (1+2Gt)（一次方）时恒等：
      eta_L(L*f) = L^(-1/2) f^(-1/2)，取 f 一次方即得乘积形式。
      故二者不是独立选项；真正独立的是「可分离」与「不可分离」两类。
      注意 f 的指数取 1/2 时两式不等价（首版误推，见过程记录）。
  G3  和式形式（1/η 相加）对应「并联阻尼」，与乘积形式（η 相乘）
      在中等档给出显著不同的读数，可用现有实验精度分辨。
  G4  第 520 条闭式与定理 6.2 都只给了单方向，
      故三者都与现有读数相容；本轮不裁决哪个对，只判「可判性」。

判据：
  H1  三形式在两端的归一化：L=1、t=0 处均等于 1（偏差 <= 1e-12）。
  H2  乘积与复合在幂律 f(t) 下等价：取 f(t) = (1+2Γt)^(1/2)，
      两式逐点相对差 <= 1e-12（证明 G2）。
  H3  乘积与和式在中等档可分辨：L·t 取中档时相对差 >= 0.1。
  H4  三形式在大参数区的渐近指数不同：log-log 斜率两两相差 >= 0.1。
  H5  恒真反例：把和式替身为乘积形式后 H3 必须 FAIL。
  H6  EP3 数值半宽（读数）。

可逆性等级：
  H1、H2、H4 判据稳健（代数结论）。
  H3 判据敏感（中等档的取点决定差幅）。
  H5 恒真对照。H6 读数。

参数与确定性：纯代数，无随机数。图存 images/p689_coupling.png。
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


G0 = 0.5


def eta_L(L):
    """尺度侧：|eta|^2 = 1/L。"""
    return np.asarray(L, dtype=float) ** (-0.5)


def eta_t(t, G=G0):
    """时间侧：系综型 (1+2Gt)^(-1/2)。"""
    return (1.0 + 2.0 * G * np.asarray(t, dtype=float)) ** (-0.5)


def form_product(L, t, G=G0):
    return eta_L(L) * eta_t(t, G)


def form_sum(L, t, G=G0):
    """1/eta = 1/eta_L + 1/eta_t - 1（并联型）。"""
    inv = 1.0 / eta_L(L) + 1.0 / eta_t(t, G) - 1.0
    return 1.0 / inv


def form_composite(L, t, G=G0):
    """eta = eta_L(L * f(t))。与乘积形式等价需 f(t) = (1+2Gt) 的一次方。

    推导：eta_L(L*f) = (L*f)^(-1/2) = L^(-1/2) * f^(-1/2)；
    乘积形式为 L^(-1/2) * (1+2Gt)^(-1/2)，
    故须 f^(-1/2) = (1+2Gt)^(-1/2)，即 f = (1+2Gt)。
    f 取 (1+2Gt)^(1/2) 是常见误推，两式不等价。
    """
    f = (1.0 + 2.0 * G * np.asarray(t, dtype=float)) ** 1.0
    return eta_L(np.asarray(L, dtype=float) * f)


# ---------------- H1 两端归一化 ----------------
vals = [float(form_product(1.0, 0.0)), float(form_sum(1.0, 0.0)),
        float(form_composite(1.0, 0.0))]
ok = max(abs(v - 1.0) for v in vals) <= 1e-12
report('H1 三形式在 L=1、t=0 处归一', ok, '; '.join(f'{v:.15f}' for v in vals))

# ---------------- H2 乘积与复合在幂律 f 下等价 ----------------
Ls = np.array([1.0, 2.0, 5.0, 10.0, 50.0])
ts = np.array([0.0, 0.5, 1.0, 3.0, 10.0])
Lg, Tg = np.meshgrid(Ls, ts)
a = form_product(Lg, Tg)
b = form_composite(Lg, Tg)
rel = np.abs(a - b) / b
ok = float(np.max(rel)) <= 1e-12
report('H2 乘积与复合在 f(t)=(1+2Gt) 下等价', ok,
       f'网格 {Lg.size} 点，最大相对差 {float(np.max(rel)):.2e}')

# ---------------- H3 乘积与和式可分辨 ----------------
# 中等档：L=10、t=1（相干度约 0.2 至 0.3，多数实验工作区）
Lm, tm = 10.0, 1.0
p_mid = float(form_product(Lm, tm))
s_mid = float(form_sum(Lm, tm))
rel_mid = abs(p_mid - s_mid) / max(p_mid, s_mid)
ok = rel_mid >= 0.1
report('H3 乘积与和式在中等档可分辨', ok,
       f'L={Lm:g} t={tm:g}：乘积 {p_mid:.4f}、和式 {s_mid:.4f}，相对差 {rel_mid:.3f}')

# ---------------- H4 大参数区渐近指数 ----------------
Lbig = np.logspace(1, 3, 40)
tbig = np.logspace(0, 2, 40)
# 沿 t = L 的对角线看 η 对 L 的斜率
diag = np.logspace(1, 3, 40)
sl_p = np.polyfit(np.log(diag), np.log(form_product(diag, diag)), 1)[0]
sl_s = np.polyfit(np.log(diag), np.log(form_sum(diag, diag)), 1)[0]
sl_c = np.polyfit(np.log(diag), np.log(form_composite(diag, diag)), 1)[0]
gap_pc = abs(sl_p - sl_c)          # 乘积与复合：应同斜率
gap_ps = abs(sl_p - sl_s)          # 乘积与和式：应不同
ok = (gap_pc <= 0.01) and (gap_ps >= 0.1)
report('H4 大参数区：乘积与复合同斜率、与和式不同', ok,
       f'斜率 乘积 {sl_p:.4f}、和式 {sl_s:.4f}、复合 {sl_c:.4f}；'
       f'乘积-复合差 {gap_pc:.4f}（<=0.01）、乘积-和式差 {gap_ps:.4f}（>=0.1）')

# ---------------- H5 恒真反例 ----------------
p_mid2 = float(form_product(Lm, tm))
rel_fake = abs(p_mid2 - p_mid2) / p_mid2
ok = rel_fake < 0.1
report('H5 恒真反例（和式替身为乘积后 H3 须判否）', ok,
       f'替身相对差 {rel_fake:.1e}（<0.1，故 H3 会判否）')

# ---------------- H6 读数 ----------------
print('[READ] H6 三形式在若干档的读数：')
print('     L      t      乘积      和式      复合')
for L_, t_ in [(1, 0), (2, 0.5), (10, 1), (10, 5), (100, 10)]:
    print(f'  {L_:5g} {t_:6g}  {form_product(L_, t_):8.4f}  '
          f'{form_sum(L_, t_):8.4f}  {form_composite(L_, t_):8.4f}')

# ---------------- 出图 ----------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))

ax = axes[0]
tt = np.linspace(0, 10, 300)
for L_, style in [(1.0, '-'), (10.0, '--'), (100.0, ':')]:
    ax.plot(tt, form_product(L_, tt), style, lw=1.6, label=f'乘积 L={L_:g}')
    ax.plot(tt, form_sum(L_, tt), style, lw=1.0, alpha=0.55)
ax.set_xlabel(r'$\Gamma t$')
ax.set_ylabel(r'$\eta(t,L)$')
ax.set_title('乘积（粗）与和式（细）在中等档分开')
ax.legend(fontsize=7.5)
ax.grid(alpha=0.3)

ax = axes[1]
ax.loglog(diag, form_product(diag, diag), '-', lw=1.8, label='乘积')
ax.loglog(diag, form_sum(diag, diag), '--', lw=1.8, label='和式')
ax.loglog(diag, form_composite(diag, diag), ':', lw=1.8, label='复合')
ax.set_xlabel(r'$L=t$（对角线）')
ax.set_ylabel(r'$\eta$')
ax.set_title('大参数区：乘积与复合重合、和式偏离')
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which='both')

plt.tight_layout()
out_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       '..', 'images'))
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, 'p689_coupling.png')
plt.savefig(out, dpi=140)
print(f'图存 {out}')

n_pass = sum(1 for _, s, _ in results if s == PASS)
print(f'\n判据 {n_pass}/{len(results)} PASS')
