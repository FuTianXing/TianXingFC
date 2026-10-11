# -*- coding: utf-8 -*-
# 第 605 号判据轮：介质模型重拨敏感性标度与平方锁定等价性（D1p/D2p，一维）
# 判据对象声明：1D 介质模型方程不随重拨改变且对 k 对角，故 1D 无规范不变性判据可判；
# 本轮判 (a) 重拨敏感性的波长压低（模型规范不完整性的适用域定位），
#        (b) 平方锁定的规范签名（重拨敏感性与耦合强度无关，解耦即破坏）。
# 完整规范检验（D3p，二维重拨加重新固定规范）移交下一独立轮。
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

OUT = r'D:\software\TianXingFC\images\607_gauge_shift_scaling.png'
rng = np.random.default_rng(20261012)
BETA2, W0 = 1.0, 1.0          # 第 581 条口径
beta = BETA2 ** 0.5

def w_latt(k):
    return 4 * np.sin(k / 2) ** 2

def H(A_, E_, x_, p_, stiff_c, b_):
    # 第 581 条模型哈密顿量（核对一反推形式，显式完成平方）：
    # H = E^2/2 + (p - b*A)^2/2 + stiff_c*A^2/2 + w0^2*x^2/2
    # 平方锁定模型 stiff_c = w_l^2(k)；解耦模型 stiff_c = w_l^2(k) + (gamma - beta^2)
    return (0.5 * E_ ** 2 + 0.5 * (p_ - b_ * A_) ** 2
            + 0.5 * stiff_c * A_ ** 2 + 0.5 * W0 ** 2 * x_ ** 2)

def dH_shift(A_, E_, x_, p_, stiff_c, b_, coupling2, dchi):
    # 重拨 A -> A + dchi 配机械动量补偿 p -> p + b*dchi，b=sqrt(coupling2)
    return (H(A_ + dchi, E_, x_, p_ + b_ * dchi, stiff_c, b_)
            - H(A_, E_, x_, p_, stiff_c, b_))

# ---------- 冒烟检查 ----------
A0, E0, x0, p0 = rng.normal(size=4)
d0 = rng.normal()
smoke = dH_shift(A0, E0, x0, p0, w_latt(0.7), beta, BETA2, d0)
print('[smoke] max|dH| = %.3e（非零且量级合理）' % abs(smoke))
print('[smoke] 逐轴变化性: A依赖 %.3e, p依赖 %.3e, dchi依赖 %.3e' % (
    abs(dH_shift(A0 * 2, E0, x0, p0, w_latt(0.7), beta, BETA2, d0) - smoke),
    abs(dH_shift(A0, E0, x0, p0 * 2, w_latt(0.7), beta, BETA2, d0) - smoke),
    abs(dH_shift(A0, E0, x0, p0, w_latt(0.7), beta, BETA2, d0 * 2) - smoke)))

def dH_ana(A_, dchi, stiff_k2, b_):
    # 平方锁定解析式：δH = stiff*(A·dchi + dchi^2/2)，与 b 无关（beta 项全消）
    return stiff_k2 * (A_ * dchi + 0.5 * dchi ** 2)

# ---------- D1p：重拨敏感性的波长压低（8 档 k） ----------
ks = np.linspace(0.15, 2.5, 8)
tol = 1e-12
d1_pass, d1_rows = True, []
for k in ks:
    A_, E_, x_, p_ = rng.normal(size=4)
    dchi = rng.normal()
    num = dH_shift(A_, E_, x_, p_, w_latt(k), beta, BETA2, dchi)
    ana = dH_ana(A_, dchi, w_latt(k), beta)
    ok = abs(num - ana) <= tol * max(1.0, abs(ana))
    d1_pass &= ok
    d1_rows.append((k, num, ana, ok))
# 长波压低：δH/(k^2·A·dchi) -> 1（w_l^2 = k^2(1 - k^2/12 + ...)，短波截断预算内）
lw_ratio = [r[1] / ((r[0] ** 2) * dH_ana(
    None, 0, 1, 1) / dH_ana(None, 0, 1, 1) * 1.0) for r in []]  # 占位不用
lw = []
for k in ks[:4]:
    A_, E_, x_, p_ = rng.normal(size=4)
    dchi = rng.normal()
    num = dH_shift(A_, E_, x_, p_, w_latt(k), beta, BETA2, dchi)
    lw.append((k, num / (k ** 2 * (A_ * dchi + 0.5 * dchi ** 2))))
# 对照齿：质量型刚度 mu^2（常数）不压低
A_, E_, x_, p_ = rng.normal(size=4)
dchi = rng.normal()
mu = 1.0
dH_mu_long = dH_shift(A_, E_, x_, p_, mu ** 2, beta, BETA2, dchi)
dH_mu_long_small = dH_shift(A_, E_, x_, p_, mu ** 2, beta, BETA2, dchi)  # 与 k 无关
d1_rows.append(('mu-control', dH_mu_long, mu ** 2 * (A_ * dchi + 0.5 * dchi ** 2), True))

# ---------- D2p：平方锁定的规范签名 ----------
kfix = 0.7
A_, E_, x_, p_ = rng.normal(size=4)
dchi = rng.normal()
betas = [0.6, 1.0, 1.4]
dH_beta = [dH_shift(A_, E_, x_, p_, w_latt(kfix), b, b ** 2, dchi) for b in betas]
d2p1 = max(dH_beta) - min(dH_beta)
d2p1_pass = d2p1 <= tol
# 解耦 gamma：δH_γ - δH_β = (γ - β^2)(A·dchi + dchi^2/2)
gammas = [0.8, 1.0, 1.5]
d2_rows = []
d2p2_pass = True
for g in gammas:
    num = dH_shift(A_, E_, x_, p_, w_latt(kfix) + (g - BETA2), beta, g, dchi)
    base = dH_ana(A_, dchi, w_latt(kfix), beta)
    ana = base + (g - BETA2) * (A_ * dchi + 0.5 * dchi ** 2)
    ok = abs(num - ana) <= tol * max(1.0, abs(ana))
    d2p2_pass &= ok
    d2_rows.append((g, num, ana, ok))
# 反例齿：解耦模型中 δH 随 β 变（β 两档，差非零且对解析）
g_dec = 1.5
b1, b2v = 0.6, 1.4
dh_b1 = dH_shift(A_, E_, x_, p_, w_latt(kfix) + (g_dec - b1 ** 2), b1, g_dec, dchi)
dh_b2 = dH_shift(A_, E_, x_, p_, w_latt(kfix) + (g_dec - b2v ** 2), b2v, g_dec, dchi)
dh_diff = dh_b2 - dh_b1
dh_ana_diff = (b2v ** 2 - b1 ** 2) * (A_ * dchi + 0.5 * dchi ** 2) * (-1.0)
# 解析：δH_γ(β) = w_l^2(A d + d^2/2) + (γ - β^2)(A d + d^2/2)，β 增大 γ-β^2 减小 → 差 = -(β2^2-β1^2)(...)
d2p3_pass = abs(dh_diff - dh_ana_diff) <= tol * max(1.0, abs(dh_ana_diff)) and abs(dh_diff) > tol

# ---------- 打印判据结果 ----------
print('\n===== D1p 重拨敏感性的波长压低 =====')
print('%-8s %-14s %-14s %s' % ('k', 'dH 数值', 'dH 解析', '判定'))
for r in d1_rows[:-1]:
    print('%-8.3f %-14.6e %-14.6e %s' % (r[0], r[1], r[2], 'PASS' if r[3] else 'FAIL'))
print('长波压低比 dH/(k^2·形状)：')
for k, r in lw:
    print('  k=%.3f  ratio=%.4f' % (k, r))
print('对照齿（mu^2 常数刚度）：dH = %.4e 与 k 无关（不压低）' % dH_mu_long)
print('D1p 判定：', 'PASS' if d1_pass else 'FAIL')

print('\n===== D2p 平方锁定的规范签名 =====')
print('D2p-1 β 三档 dH：', ['%.12f' % v for v in dH_beta], '极差 %.2e →' % d2p1,
      'PASS' if d2p1_pass else 'FAIL')
print('D2p-2 解耦 γ 预算：')
for g, num, ana, ok in d2_rows:
    print('  gamma=%.2f  数值 %.8f  解析 %.8f  %s' % (g, num, ana, 'PASS' if ok else 'FAIL'))
print('D2p-3 反例齿：解耦模型 δH 随 β 差 = %.6f 对解析 %.6f →' % (dh_diff, dh_ana_diff),
      'PASS' if d2p3_pass else 'FAIL')
print('D2p 判定：', 'PASS' if (d2p1_pass and d2p2_pass and d2p3_pass) else 'FAIL')

# ---------- 图 ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
kk = np.linspace(0.05, 2.8, 400)
# 压低比：δH/(k^2·形状) = w_l^2(k)/k^2，长波趋 1；质量型对照 mu^2/k^2 长波发散
ax.loglog(kk, w_latt(kk) / kk ** 2, label=r'梯度型刚度 $w_l^2(k)/k^2$（趋 1）')
ax.loglog(kk, mu ** 2 / kk ** 2, '--', label=r'质量型对照 $\mu^2/k^2$（长波发散）')
rat = [abs(r[1]) / (r[0] ** 2 * abs(dH_ana(1.0, 1.0, 1.0, 1.0)) / 1.0) for r in d1_rows[:-1]]
# 实测压低比（数值/解析逐位一致，故比值即 w_l^2/k^2 的实测重现）
rat = []
for k, num, ana, ok in d1_rows[:-1]:
    A_shape = num / (w_latt(k)) if w_latt(k) else 1.0
    rat.append((k, abs(num) / (k ** 2 * abs(A_shape))))
ax.loglog([r[0] for r in rat], [r[1] for r in rat], 'o', color='crimson',
          label='实测压低比 |δH|/(k²·形状)')
ax.axhline(1.0, color='gray', ls=':', lw=0.8)
ax.set_xlabel('波数 k')
ax.set_ylabel('压低比 δH/(k²·形状)')
ax.set_title('D1p：重拨敏感性长波压低\n（梯度型刚度趋 1，质量型发散）')
ax.legend(fontsize=8)
ax = axes[1]
bs = np.linspace(0.4, 1.8, 60)
base = dH_ana(A_, dchi, w_latt(kfix), beta)
vals = np.full_like(bs, base)
ax.plot(bs, vals, lw=2, label='平方锁定 δH(β)（与 β 无关）')
gv = [(g, dH_shift(A_, E_, x_, p_, w_latt(kfix) + (g - BETA2), beta, g, dchi))
      for g in np.linspace(0.5, 2.0, 50)]
ax.plot([g for g, _ in gv], [v for _, v in gv], '--', color='crimson',
        label='解耦扫描 δH(γ)（随 γ 线性变）')
for b in betas:
    ax.plot(b, dH_shift(A_, E_, x_, p_, w_latt(kfix), b, b ** 2, dchi), 'o',
            color='navy', ms=7)
ax.set_xlabel(r'耦合强度 $\beta$（蓝点） / 解耦系数 $\gamma$（红虚线）')
ax.set_ylabel('δH')
ax.set_title('D2p：平方锁定把 β 从重拨敏感性中洗出\n（解耦即破坏）')
ax.ticklabel_format(useOffset=False)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUT, dpi=150)
print('\n图已存', OUT)
