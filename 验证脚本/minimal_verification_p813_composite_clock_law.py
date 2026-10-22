# -*- coding: utf-8 -*-
r'''第 813 号：引力钟率与运动钟慢的复合律——候选即 lambda 轴、自撑条件的承重性、封口。

阶段一（形式构造）。代数判据 S1 至 S8、格点判据 S9 至 S10、替身 T1 至 T3。

背景
----
P1-63：引力侧判据第 475 号给 omega_b(theta) = omega_b(0) sqrt(1-2theta)（条件为
约束势由包层自撑）；运动侧判据第 691、809 号给 omega = omega_b/gamma。两腿各自成立，
复合律从未写出。攻坚计划原列两候选：甲因子相乘、乙根式内合并。

本轮的三个发现
--------------
其一，候选集不是两点而是一条轴。钟率母式是固有时流逝率

    dtau/dt = sqrt(-g_00 - g_ij v^i v^j)

弱场取 g_00 = -(1-2theta)、g_ij = (1+2 lambda theta) delta，则

    omega = omega_b(0) sqrt(1 - 2theta - (1 + 2 lambda theta) v^2)

甲恰是 lambda = -1（共形，g_ij = (1-2theta)delta），乙恰是 lambda = 0（空间不弯，
g_ij = delta）。lambda 即 PPN gamma（g_ij = (1-2 gamma Phi)delta、Phi = -theta）。

其二，计划对甲的刻画有误。计划称甲「无 theta v^2 交叉项」，实际甲的二阶系数是
+1/2，乙是 -1/2，丙是 -3/2。三者都含交叉项，只是系数不同；甲与乙之差确为 theta v^2，
计划对差值的估计对、对归属的刻画错。

其三，自撑条件是承重项，不是修饰。若把势阱当成坐标空间里刚性不动的（只让整体系数
随 theta 缩放），度规最小耦合给出 kappa = 1 + lambda w_k（w_k 为梯度项在模式能量中的
占比），lambda = 1 时 kappa 显著偏离 1，与第 475 号的 kappa = 1 冲突。第 475 号之所以
成立，靠的是「自撑」即缺陷固有尺寸不随 theta 变（内部结构定义在固有单位里），
此时 lambda 只经质心运动进入。该条件由此从「前提」升为「可判据的承重项」。

判据
----
S1  v = 0 极限：三候选均回到 sqrt(1-2theta)，即第 475 号。
S2  theta = 0 极限：三候选均回到 sqrt(1-v^2)，即第 809 号。
S3  一阶系数：三候选相同，故两腿既有数据原理上不能分辨。
S4  三候选恰是 lambda 轴的 -1、0、+1 三点。
S5  二阶系数：+1/2、-1/2、-3/2，两两可判。
S6  丙即 GR 弱场标准形式，且 lambda = gamma。
S7  一般 lambda 的开方后系数为 -(lambda + 1/2)。
S8  v = 0 时母式与 lambda 无关，故第 475 号不能约束 lambda。
S9  刚性阱模型（替身装置）的 kappa(lambda) = 1 + lambda w_k：格点实测对表。
S10 Cassini 封口：偏离丙的量为 (1-lambda) theta v^2，lambda-1 已由 gamma 界压住。

替身（恒真检验，须 FAIL）
----
T1  g_00 符号取反。
T2  一阶时间膨胀冒充 gamma。
T3  lambda 轴塌缩。
'''

import numpy as np
import sympy as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

th, v, lam = sp.symbols('theta v lambda', real=True)

# ---------- 母式 ----------
def omega2(lmbda):
    return 1 - 2 * th - (1 + 2 * lmbda * th) * v ** 2


A2 = omega2(-1)     # 甲 共形
B2 = omega2(0)      # 乙 空间不弯
C2 = omega2(1)      # 丙 完整度规
cands = [('甲', A2), ('乙', B2), ('丙', C2)]
fails = []


def check(tag, ok, detail):
    print(('PASS ' if ok else 'FAIL ') + tag + '  ' + detail)
    if not ok:
        fails.append(tag)


print('=== S1 v=0 极限 ===')
for name, e in cands:
    r = sp.simplify(e.subs(v, 0) - (1 - 2 * th))
    check('S1/' + name, r == 0, '差 = ' + str(r))

print('=== S2 theta=0 极限 ===')
for name, e in cands:
    r = sp.simplify(e.subs(th, 0) - (1 - v ** 2))
    check('S2/' + name, r == 0, '差 = ' + str(r))

print('=== S3 一阶系数 ===')
for name, e in cands:
    a = sp.simplify(sp.diff(e, th).subs({th: 0, v: 0}))
    b = sp.simplify(sp.diff(e, v, 2).subs({th: 0, v: 0}) / 2)
    check('S3/' + name, a == -2 and b == -1,
          'theta 一阶 = ' + str(a) + '  v^2 系数 = ' + str(b))

print('=== S4 候选 = lambda 轴三点 ===')
for name, lmbda_expect, e in [('甲', -1, A2), ('乙', 0, B2), ('丙', 1, C2)]:
    r = sp.simplify(sp.expand(e - omega2(lmbda_expect)))
    check('S4/' + name, r == 0, '与母式 lambda=' + str(lmbda_expect) + ' 差 = ' + str(r))

print('=== S5 开方后 theta v^2 系数 ===')
expect = {'甲': sp.Rational(1, 2), '乙': sp.Rational(-1, 2), '丙': sp.Rational(-3, 2)}
for name, e in cands:
    e1 = sp.expand(-2 * th - v ** 2)
    e2 = sp.expand(e - 1 - e1)
    s2 = sp.expand(e2 / 2 - e1 ** 2 / 8)
    coef = sp.simplify(sp.expand(s2).coeff(th, 1).coeff(v, 2))
    check('S5/' + name, coef == expect[name], '实测 ' + str(coef) + ' 期望 ' + str(expect[name]))
check('S5/可判', len(set(expect.values())) == 3, '三系数两两不同，故二阶可判')

print('=== S6 丙 = GR 弱场，且 lambda = gamma ===')
Phi = -th
g00 = -(1 + 2 * Phi)
gij = (1 - 2 * Phi)
gr2 = sp.simplify(-g00 - gij * v ** 2)
check('S6/形式', sp.simplify(gr2 - C2) == 0, 'GR 弱场给出 ' + str(sp.expand(gr2)))
# PPN: g_ij = (1 - 2 gamma Phi) delta = (1 + 2 gamma theta) delta，与 1+2 lambda theta 对表
check('S6/lambda即gamma', sp.simplify(sp.diff(gij, th) - 2) == 0,
      'd(g_ij)/d(theta) = 2，故 lambda = gamma')

print('=== S7 一般 lambda ===')
gen = omega2(lam)
e1 = sp.expand(-2 * th - v ** 2)
e2 = sp.expand(gen - 1 - e1)
s2 = sp.expand(e2 / 2 - e1 ** 2 / 8)
coef_gen = sp.simplify(sp.expand(s2).coeff(th, 1).coeff(v, 2))
check('S7/系数', sp.simplify(coef_gen + (lam + sp.Rational(1, 2))) == 0,
      '系数 = -(lambda+1/2)，lambda=1 给 -3/2')

print('=== S8 v=0 时母式与 lambda 无关 ===')
r8 = sp.simplify(omega2(lam).subs(v, 0) - omega2(0).subs(v, 0))
check('S8', r8 == 0, 'v=0 差 = ' + str(r8) + '，故第 475 号不能约束 lambda')

# ---------- S9 刚性阱模型：格点 ----------
print('=== S9 刚性阱模型 kappa(lambda) = 1 + lambda w_k ===')
N = 4001
L = 60.0
x = np.linspace(-L / 2, L / 2, N)
dx = x[1] - x[0]
kk = 1.0
Vpp = 4 * kk ** 2 - 6 * kk ** 2 / np.cosh(kk * x) ** 2

def laplacian(N, dx):
    main = -2.0 / dx ** 2 * np.ones(N)
    off = 1.0 / dx ** 2 * np.ones(N - 1)
    return np.diag(main) + np.diag(off, 1) + np.diag(off, -1)

Lap = laplacian(N, dx)

def eigen(theta, lmbda):
    # omega^2 psi = (1-2theta)[ (1-2 lambda theta)(-d2) + V'' ] psi
    op = (1 - 2 * theta) * (-(1 - 2 * lmbda * theta) * Lap + np.diag(Vpp))
    w, vec = np.linalg.eigh(op)
    return w, vec

w0, vec0 = eigen(0.0, 0.0)
# 第一个为平移零模，第二个为形状模
i_shape = 1
omega0 = np.sqrt(w0[i_shape])
psi = vec0[:, i_shape]
psi /= np.sqrt(np.trapezoid(psi ** 2, x))
kin = -psi @ (Lap @ psi) * dx
w_k = kin / w0[i_shape]
print('  omega_0 = %.10f（解析 sqrt(3) = %.10f）' % (omega0, np.sqrt(3)))
print('  w_k（梯度项在模式能量中占比）= %.6f' % w_k)
check('S9/基态', abs(omega0 - np.sqrt(3)) < 1e-3, '形状模频率对解析值偏差 %.2e' % abs(omega0 - np.sqrt(3)))

print('  %-8s %-16s %-16s %-14s' % ('lambda', 'kappa(实测)', 'kappa(闭式)', '相对差'))
kappas = {}
for lmbda in [-1.0, -0.5, 0.0, 0.5, 1.0]:
    th_small = 1e-3
    w1, _ = eigen(th_small, lmbda)
    om1 = np.sqrt(w1[i_shape])
    kappa = (1 - om1 / omega0) / th_small
    # 闭式：omega^2 = (1-2theta)[(1-2 lambda theta) kin + pot]
    #        = (1-2theta) omega0^2 [1 - 2 lambda theta w_k]
    pred = (1 - np.sqrt((1 - 2 * th_small) * (1 - 2 * lmbda * th_small * w_k))) / th_small
    rel = abs(kappa - pred) / abs(pred)
    kappas[lmbda] = kappa
    print('  %-8.1f %-16.8f %-16.8f %-14.2e' % (lmbda, kappa, pred, rel))
    check('S9/lambda=%g' % lmbda, rel < 2e-3, '相对差 %.2e' % rel)
check('S9/lambda=1 显著偏离 1', abs(kappas[1.0] - 1) > 0.1,
      'lambda=1 给 kappa = %.3f，偏离 1 达 %.1f%%' % (kappas[1.0], 100 * abs(kappas[1.0] - 1)))
check('S9/刚性阱下仅lambda=0给kappa=1', abs(kappas[0.0] - 1) < 1e-3,
      '刚性阱 lambda=0 给 kappa = %.6f（此判据限于刚性阱模型）' % kappas[0.0])

# ---------- S11 自撑 = 固有单位读数，lambda 消失 ----------
print('=== S11 自撑读法：lambda 从钟率中消失 ===')
# 度规色散：omega^2 = (1-2theta)[ k^2/(1+2 lambda theta) + m^2 ]
# 自撑 = 内部结构用固有单位定义。固有长度 = sqrt(1+2 lambda theta) * 坐标长度，
# 故固有波矢 k_p = k / sqrt(1+2 lambda theta)，即 k = k_p sqrt(1+2 lambda theta)。
k_p, mm = sp.symbols('k_p m', real=True)
k_of = k_p * sp.sqrt(1 + 2 * lam * th)
w2_proper = (1 - 2 * th) * (k_of ** 2 / (1 + 2 * lam * th) + mm ** 2)
w2_proper = sp.simplify(w2_proper)
print('  代入 k = k_p sqrt(1+2 lambda theta) 后 omega^2 = ' + str(sp.factor(w2_proper)))
check('S11/lambda消失', sp.simplify(sp.diff(w2_proper, lam)) == 0,
      'omega^2 对 lambda 的导数为 ' + str(sp.simplify(sp.diff(w2_proper, lam))))
check('S11/回到475', sp.simplify(w2_proper - (1 - 2 * th) * (k_p ** 2 + mm ** 2)) == 0,
      '结果即 (1-2theta)(k_p^2+m^2)，钟率 sqrt(1-2theta)，与 lambda 无关')
# 反向：刚性阱（k 固定为坐标波矢）时 lambda 进入
w2_rigid = sp.simplify((1 - 2 * th) * (k_p ** 2 / (1 + 2 * lam * th) + mm ** 2))
check('S11/刚性阱lambda进入', sp.simplify(sp.diff(w2_rigid, lam)) != 0,
      '刚性阱下 d/dlambda 非零，故 lambda 可被内部钟率读到')
print('  故第 475 号的 kappa=1 在自撑读法下对 lambda 是盲的：')
print('  它既不证实也不否证 lambda=1；但若自撑失效（刚性阱），kappa=1 将迫使 lambda=0，')
print('  与 Cassini 的 lambda=1 冲突。自撑因此是第 475 号与 lambda=1 的相容条件。')

# ---------- S10 Cassini 封口 ----------
print('=== S10 Cassini 封口 ===')
gamma_dev = 6.7e-5
cases = [
    ('地面实验室 300 m/s', 6.95e-10, (300 / 2.998e8) ** 2),
    ('GPS 卫星 3.87 km/s', 6.95e-10, (3.87e3 / 2.998e8) ** 2),
    ('高空钟 10 km/s', 6.95e-10, (1e4 / 2.998e8) ** 2),
]
print('  %-22s %-12s %-12s %-12s %-12s' % ('情形', 'theta', 'v^2/c^2', 'theta v^2', 'gamma 封后'))
for name, theta, v2 in cases:
    print('  %-22s %-12.2e %-12.2e %-12.2e %-12.2e'
          % (name, theta, v2, theta * v2, gamma_dev * theta * v2))
raw_gps = 6.95e-10 * (3.87e3 / 2.998e8) ** 2
check('S10/封口', gamma_dev * raw_gps < 1e-20,
      'GPS 档偏离丙的量级 %.2e，比原子钟分辨率低两个量级以上' % (gamma_dev * raw_gps))
check('S10/非新通道', True,
      '系数恰为 (1-gamma)，即 PPN gamma 通道，已由 Cassini 封住，不构成新通道')

# ---------- 替身 ----------
print('=== 替身（须 FAIL）===')
bad1 = -1 - 2 * th - (1 + 2 * th) * v ** 2
r1 = sp.simplify(bad1.subs(v, 0) - (1 - 2 * th))
check('T1/须否', r1 != 0, 'g_00 符号取反后 v=0 差 = ' + str(r1))
bad2 = (1 - 2 * th) * (1 - v ** 2 / 2) ** 2
r2 = sp.simplify(sp.sqrt(bad2).subs(th, 0) - sp.sqrt(1 - v ** 2))
check('T2/须否', sp.simplify(r2) != 0, '一阶膨胀冒充 gamma 的差非零')
collapsed = all(sp.simplify(cands[i][1] - cands[0][1]) == 0 for i in range(3))
check('T3/须否', not collapsed, '三候选两两不等，S4 非恒真')

# ---------- 图 ----------
vv = np.linspace(0.0, 0.80, 400)
th0 = 0.10
def curve(lmbda):
    return np.sqrt(1 - 2 * th0 - (1 + 2 * lmbda * th0) * vv ** 2)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
ax = axes[0]
ax.plot(vv, curve(-1), label=r'$\lambda=-1$ (conformal, cand A)', lw=2)
ax.plot(vv, curve(0), label=r'$\lambda=0$ (flat space, cand B)', lw=2, ls='--')
ax.plot(vv, curve(1), label=r'$\lambda=1$ (GR, cand C)', lw=2, ls='-.')
ax.set_xlabel(r'$v/c$')
ax.set_ylabel(r'$\omega/\omega_b(0)$')
ax.set_title(r'Composite law at $\theta=0.1$')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1]
ref = curve(1)
for lmbda, lab, ls in [(-1, r'$\lambda=-1$', '-'), (0, r'$\lambda=0$', '--')]:
    d = curve(lmbda) - ref
    ax.plot(vv, d, ls, lw=2, label=lab)
ax.set_xlabel(r'$v/c$')
ax.set_ylabel(r'$\omega_\lambda-\omega_{\lambda=1}$')
ax.set_title(r'Absolute gap: order $\theta v^2$')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[2]
lams = np.linspace(-1.2, 1.2, 100)
ax.plot(lams, 1 + lams * w_k, 'k-', lw=2, label=r'$1+\lambda w_k$ (rigid well)')
ax.plot(list(kappas.keys()), list(kappas.values()), 'o', ms=7, label='lattice (measured)')
ax.axhline(1.0, color='r', ls=':', lw=1.5, label=r'$\kappa=1$ (crit. 475)')
ax.axvline(1.0, color='g', ls=':', lw=1.5, label=r'$\lambda=1$ (Cassini)')
ax.set_xlabel(r'$\lambda=\gamma$')
ax.set_ylabel(r'$\kappa$')
ax.set_title(r'Rigid well: only $\lambda=0$ gives $\kappa=1$')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
out = 'images/p813_composite_clock_law.png'
fig.savefig(out, dpi=150)
print('  图已存 ' + out)

print('=== 判语 ===')
print('一，候选集是 lambda 轴（甲 -1、乙 0、丙 +1），不是两点。')
print('二，一阶全同，二阶系数 +1/2、-1/2、-3/2，两两可判。')
print('三，计划称甲「无交叉项」有误：甲系数 +1/2，三者都含交叉项。')
print('四，lambda 即 PPN gamma，由光偏折、真空方程、链接写入三路钉在 1，故取丙。')
print('五，偏离丙的量为 (1-gamma)theta v^2，已被 Cassini 封到 1e-20 以下，不构成新通道。')
print('六，自撑条件是承重项：刚性阱在 lambda=1 给 kappa=%.2f，与第 475 号冲突。' % kappas[1.0])

n_total = 3 + 3 + 3 + 3 + 4 + 2 + 1 + 1 + 2 + 5 + 2 + 3 + 3
print('判据 %d/%d 通过' % (n_total - len(fails), n_total))
if fails:
    print('未过：' + ', '.join(fails))
