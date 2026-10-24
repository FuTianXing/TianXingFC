# -*- coding: utf-8 -*-
r"""
c1_anchor_feedback_unified.py

[C1 越零最后一问] 锚核与反馈的统一构型：合成规则判定（判据 AF-1 至 AF-4）

背景：窗口算术注记（判据 U-1 至 U-3）把越零问题留给一条合成规则：锚核残差与
软化折扣按加法合成（最有利）时 W1/W2 档越零成立，按乘法合成不成立。本判据在
同一个正定泛函里同时开锚核（偏好凹口：K 局域压低）与反馈（kappa_e 记录软化），
判死合成规则，并给出该构型类可达到的最小线项残差。

模型：一维链 N = 600，
E = sum_e (1/2) kappa_e (du_e - 1)^2 + sum_i K_i (1 - cos 2 pi u_i)，
K_i = K (1 - d exp[-(x_i - x_0)^2 / (2 w^2)])，x_0 = N/2；
kappa_e = clip(1 + c s_e, 0.1)，s_e = (theta_i + theta_{i+1})/2，theta_i = u_i - x_i。
自洽求解：弛豫 4000 步与记录更新交替 6 轮（口径同 L 系列）。

结构性事实（精确）：K_i >= 0（d <= 1）与 kappa_e >= 0.1 使 E 逐点非负，真空
E = 0，故任何壁的线项 a = E(构型) >= 0：越零不可能；加法合成的越零读法要求
总节省率超过 1，被正定性排除。

判据
AF-1 锚核标定：扫描 (d, w)，给锚核节省 S_a 的可达范围，并拾取三档配置
     （S_a 接近 0.40、0.70 与最大值）。
AF-2 反馈基准：d = 0 时复验 eta(c)，口径同 L 系列。
AF-3 统一网格：拾取配置 x c，实测总节省 S_tot，与加法预测 S_a + eta、
     乘法预测 1 - (1 - S_a)(1 - eta) 对照，给比值
     r = [E(d,0) - E(d,c)] / [E(0,0) - E(0,c)]（加法 r = 1，乘法 r = 1 - S_a）。
AF-4 越零判死：全网格最小残差 1 - max S_tot；列出加法预测超过 100% 的档位。

运行：python 验证脚本/c1_anchor_feedback_unified.py
确定性、无随机数；终端输出 + 图 images/c1_unified_composition.png。
"""
import numpy as np

N, KAP, ETA_STEP, STEPS_OUT, STEPS_IN = 600, 1.0, 0.005, 6, 4000
FLOOR = 0.1
K_MAIN = 0.005
W_LIST = [1.5, 3.0, 6.0, 12.0, 24.0, 48.0]
D_LIST = [0.3, 0.5, 0.7, 0.85, 0.95]
CS_FB = [-0.2, -0.4, -0.6, -0.8, -0.85]
U_C = [-0.4, -0.8, -0.85]


def relax(u, ke, Ks, steps=STEPS_IN, eta=ETA_STEP):
    for _ in range(steps):
        du = np.diff(u)
        g = np.zeros(N)
        g[:-1] -= ke * (du - 1.0)
        g[1:] += ke * (du - 1.0)
        g += Ks * 2.0 * np.pi * np.sin(2.0 * np.pi * u)
        g[0] = g[-1] = 0.0
        u = u - eta * g
        u[0], u[-1] = 0.0, float(N)
    return u


def energy(u, ke, Ks):
    du = np.diff(u)
    return float(0.5 * (ke * (du - 1.0) ** 2).sum()
                 + (Ks * (1.0 - np.cos(2.0 * np.pi * u))).sum())


def width(u):
    du = np.diff(u)
    dens = 0.5 * (du - 1.0) ** 2
    idx = np.arange(N - 1) + 0.5
    c0 = (dens * idx).sum() / dens.sum()
    return 2.0 * np.sqrt((dens * (idx - c0) ** 2).sum() / dens.sum())


def profile(K, d=0.0, w=3.0):
    x = np.arange(N, dtype=float)
    return K * (1.0 - d * np.exp(-0.5 * ((x - N // 2) / w) ** 2))


def wall(c, K, d=0.0, w=3.0):
    x = np.arange(N, dtype=float)
    Ks = profile(K, d, w)
    u = np.where(x < N // 2, x, x + 1.0)
    u[N // 2] = N / 2 + 0.5
    ke = np.ones(N - 1)
    for _ in range(STEPS_OUT):
        u = relax(u, ke, Ks)
        s = 0.5 * ((u - x)[:-1] + (u - x)[1:])
        ke = np.clip(KAP * (1.0 + c * s), FLOOR * KAP, None)
    return u, ke, Ks


print('=' * 74)
print('C1 统一构型判据：锚核与反馈的合成规则（AF-1 至 AF-4）')
print('=' * 74)
print('结构性事实：K_i >= 0 与 kappa_e >= 0.1 逐点成立，真空 E = 0，')
print('故任何构型 E >= 0，线项 a = E >= 0；越零不可能（精确）。')

u0, ke0, Ks0 = wall(0.0, K_MAIN)
E0 = energy(u0, ke0, Ks0)
print('基准（K = 0.005，d = 0，c = 0）：E_0 = %.6f，ell* = %.3f' % (E0, width(u0)))

print('')
print('[AF-1] 锚核扫描（c = 0，S_a = 1 - E / E_0）')
scans = []
for w in W_LIST:
    vals = []
    for d in D_LIST:
        u, ke, Ks = wall(0.0, K_MAIN, d, w)
        E = energy(u, ke, Ks)
        Sa = 1.0 - E / E0
        scans.append((w, d, Sa, E))
        vals.append('%.1f%%' % (Sa * 100))
    print('   w = %5.1f：d 0.3/0.5/0.7/0.85/0.95 -> ' % w + ' / '.join(vals))
u, ke, Ks = wall(0.0, K_MAIN, 0.95, 96.0)
E = energy(u, ke, Ks)
Sa96 = 1.0 - E / E0
print('   加测 w = 96.0，d = 0.95：S_a = %.1f%%，ell* = %.2f' % (Sa96 * 100, width(u)))


def pick(target):
    return min(scans, key=lambda t: abs(t[2] - target))


A1 = pick(0.40)
A2 = pick(0.70)
A3 = max(scans, key=lambda t: t[2])
for tag, (w, d, Sa, E) in (('A1', A1), ('A2', A2), ('A3', A3)):
    u, ke, Ks = wall(0.0, K_MAIN, d, w)
    print('   拾取 %s：d = %.2f，w = %.1f，S_a = %.3f（残差 %.3f），ell* = %.2f'
          % (tag, d, w, Sa, 1.0 - Sa, width(u)))

print('')
print('[AF-2] 反馈基准（d = 0）')
ETA = {}
EFB = {}
for c in CS_FB:
    u, ke, Ks = wall(c, K_MAIN)
    E = energy(u, ke, Ks)
    ETA[c] = 1.0 - E / E0
    EFB[c] = E
    print('   c = %5.2f：eta = %.3f%%（E = %.6f，ell* = %.3f）' % (c, ETA[c] * 100, E, width(u)))

print('')
print('[AF-3] 统一网格（S_add = S_a + eta；S_mult = 1 - (1 - S_a)(1 - eta)；')
print('        r = [E(d,0) - E(d,c)] / [E(0,0) - E(0,c)]，加法 r = 1，乘法 r = 1 - S_a）')
uni = []
for tag, (w, d, Sa, Ea) in (('A1', A1), ('A2', A2), ('A3', A3)):
    for c in U_C:
        u, ke, Ks = wall(c, K_MAIN, d, w)
        E = energy(u, ke, Ks)
        St = 1.0 - E / E0
        Sad = Sa + ETA[c]
        Smu = 1.0 - (1.0 - Sa) * (1.0 - ETA[c])
        r = (Ea - E) / (E0 - EFB[c])
        uni.append((tag, w, d, Sa, c, St, Sad, Smu, r))
        print('   %s（d = %.2f，w = %.1f，c = %5.2f）：S_tot = %.1f%%，S_add = %.1f%%，'
              'S_mult = %.1f%%，r = %.3f（加法 1，乘法 %.3f）'
              % (tag, d, w, c, St * 100, Sad * 100, Smu * 100, r, 1.0 - Sa))

print('')
print('[AF-4] 汇总')
St_max = max(t[5] for t in uni)
print('   统一网格 max S_tot = %.2f%%，最小残差 a / sigma_0 = %.2f%%（> 0，越零判死）'
      % (St_max * 100, (1.0 - St_max) * 100))
over = [(t[0], t[4], t[6]) for t in uni if t[6] > 1.0]
print('   加法预测超过 100%% 的档位：%d 个' % len(over))
for tag, c, Sad in over:
    print('      %s，c = %5.2f：S_add = %.1f%%（被正定性上限 100%% 排除）' % (tag, c, Sad * 100))

print('')
print('[AF-1b] K = 0.02 对照（c = 0 与统一点）')
u0b, ke0b, Ks0b = wall(0.0, 0.02)
E0b = energy(u0b, ke0b, Ks0b)
for (d, w) in ((0.85, 3.0), (0.95, 12.0)):
    u, ke, Ks = wall(0.0, 0.02, d, w)
    Sab = 1.0 - energy(u, ke, Ks) / E0b
    uu, kk, KK = wall(-0.8, 0.02, d, w)
    Stb = 1.0 - energy(uu, kk, KK) / E0b
    print('   d = %.2f，w = %.1f：S_a = %.1f%%，c = -0.8 时 S_tot = %.1f%%'
          % (d, w, Sab * 100, Stb * 100))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))
ax = axes[0]
for w in W_LIST:
    ds = [t[1] for t in scans if t[0] == w]
    ss = [t[2] * 100 for t in scans if t[0] == w]
    ax.plot(ds, ss, 'o-', ms=4, label='w = %.1f' % w)
for (tag, (w, d, Sa, E)) in (('A1', A1), ('A2', A2), ('A3', A3)):
    ax.plot([d], [Sa * 100], 'k*', ms=13)
ax.set_xlabel('锚核深度 d')
ax.set_ylabel('锚核节省 S_a (%)')
ax.set_title('(a) 锚核标定：S_a 对 (d, w)')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1]
tags = ['A1', 'A2', 'A3']
xb = np.arange(3)
wb = 0.2
s_tot = [t[5] * 100 for t in uni if abs(t[4] + 0.8) < 1e-9]
s_add = [t[6] * 100 for t in uni if abs(t[4] + 0.8) < 1e-9]
s_mul = [t[7] * 100 for t in uni if abs(t[4] + 0.8) < 1e-9]
s_anc = [t[3] * 100 for t in uni if abs(t[4] + 0.8) < 1e-9]
ax.bar(xb - 1.5 * wb, s_anc, wb, label='S_a（锚核）')
ax.bar(xb - 0.5 * wb, s_tot, wb, label='S_tot（实测）')
ax.bar(xb + 0.5 * wb, s_add, wb, label='S_add（加法预测）')
ax.bar(xb + 1.5 * wb, s_mul, wb, label='S_mult（乘法预测）')
ax.axhline(100.0, color='black', ls='-', lw=1.2)
ax.text(2.35, 101.5, '正定性上限 100%', fontsize=8)
ax.set_xticks(xb)
ax.set_xticklabels(['A1', 'A2', 'A3'])
ax.set_ylabel('百分比 (%)')
ax.set_title('(c = -0.8) 两机制合成的三种读法')
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis='y')

ax = axes[2]
r_meas = [t[8] for t in uni if abs(t[4] + 0.8) < 1e-9]
r_mult = [1.0 - t[3] for t in uni if abs(t[4] + 0.8) < 1e-9]
ax.bar(xb, r_meas, 0.5, label='r（实测）')
ax.plot(xb, [1.0, 1.0, 1.0], 'k--', label='加法参考 r = 1')
ax.plot(xb, r_mult, 'k:', label='乘法参考 r = 1 - S_a')
ax.set_xticks(xb)
ax.set_xticklabels(['A1', 'A2', 'A3'])
ax.set_ylabel('增量反馈节省比 r')
ax.set_title('(c = -0.8) 合成规则实测')
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis='y')

fig.tight_layout()
out = 'D:/software/TianXingFC/images/c1_unified_composition.png'
fig.savefig(out, dpi=150)
print('')
print('[图] 已保存 ' + out)
