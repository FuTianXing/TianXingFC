# -*- coding: utf-8 -*-
"""
minimal_verification_composition_universality.py

组成普适判据的最小验证（第 459 条）：多形态同总能量缺陷的等效原理账目验证。

背景（等效原理推导注记）：引力半边（作用量转录，第 448 条 TA1 增益 1）与
惯性半边（变率继承，第 438 条折算系数 1）读出同一份缺陷能量 E，相除得
WEP 零参数形式。本轮验证其组成普适性：同一总能量、不同内部结构（单质点、
两质点互连、三质点链、四质点链）的系统，在均匀重力场中自由下落的质心
加速度彼此相同（eta = 0），且总能量守恒（账目与力学层）。

判据：
  CU1（能量守恒，弛豫段）：四种结构的总能量（动能 + 内弹簧势能）恒等于
      初值（Verlet 有界误差 <= 1e-6，预算 (omega_max*DT)^2）。
  CU2（自由下落质心加速度结构无关）：重力段四结构的质心加速度彼此相同
      （内力成对抵消，恒外力 Verlet 精确，<= 1e-6；判据取 500/1000/2000/4000
      四档窗宽的最大值，阈值依据见 CU2 代码段注释）。
  CU3（加速度值 = 场强）：质心加速度 / G_FIELD = 1（<= 1e-6）。
  CU4（Eotvos 参数）：eta = 2|a_A - a_C|/(a_A + a_C) = 0（<= 1e-6）。

模型设定与建模决策（分层声明）：
  - 质点网络：质点由互连弹簧（内力成对抵消）构成"缺陷内部结构"，整体置于
    均匀重力场（加速度 G_FIELD 施加于每个质点）自由下落。等效原理的检验
    对象是自由下落，模型不得含连墙弹簧（墙力是外力，会污染质心）——初版
    模型的概念错误，已修正（教训入注记）。
  - 牛顿动力学 + velocity Verlet；模型单位（c = 1），构造上 mu_i = E_i
    （引力荷与惯性同载体，两种重力编码简并；惯性能量等价的机制归变率
    继承第 438 条）。
  - 绝对标度（转录读数的绝对值）已由第 446、447 条钉死，本轮只验结构
    无关性。

边界声明：验证结构无关性在力学层的离散实现；不验证三维复合位错的完整
动力学（C1 线）、不验证真实重力场中场对缺陷的力定律（强场工具域）、
不验证 EEP 的局域性两半合成。

可逆性等级：判据稳健（结构无关性不依赖力编码细节与结构参数量值）。
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

results = []
E_TOT = 1.0
G_FIELD = 0.01
DT = 2.5e-4
N_RELAX = 20000
N_FALL = 20000
NT = N_RELAX + N_FALL

def make_structure(kind):
    """返回 (mus, springs [(i, j, k)], x0, p0, name)。E_kin(0) + E_pot(0) = E_TOT。"""
    if kind == 'A':      # 单质点，无内力
        mus = [1.0]
        springs = []
        p0 = [np.sqrt(2 * E_TOT / mus[0])]
        x0 = [0.0]                                   # E_kin = p0^2/(2 mu) = 1
        name = 'single'
    elif kind == 'B':    # 两质点互连
        mus = [0.5, 0.5]
        springs = [(0, 1, 2.0)]
        x0 = [0.0, 1.0]; p0 = [0.0, 0.0]             # E_pot = 2*1^2/2 = 1
        name = 'two-linked'
    elif kind == 'C':    # 三质点链
        mus = [1/3, 1/3, 1/3]
        springs = [(0, 1, 2.0), (1, 2, 2.0)]
        x0 = [-0.5, 0.0, 0.5]
        p0 = [np.sqrt(1/6), 0.0, -np.sqrt(1/6)]      # E_kin = 3 p0^2 = 0.5
        # E_pot = k*(0.5^2)/2 * 2 springs = 0.5；总 1
        name = 'three-chain'
    else:                # 四质点链
        mus = [0.25, 0.25, 0.25, 0.25]
        springs = [(0, 1, 2.0), (1, 2, 2.0), (2, 3, 2.0)]
        x0 = [-0.6, -0.2, 0.2, 0.6]
        amp = np.sqrt(0.52 * 2 * 0.25 / 2.0)         # 两外侧质点各承 E_kin/2
        p0 = [amp, 0.0, 0.0, -amp]
        # E_pot = 3 * 2*(0.4^2)/2 = 0.48；E_kin = 2*amp^2/(2*0.25) = 0.52
        name = 'four-chain'
    return np.array(mus), springs, np.array(x0), np.array(p0), name

def run(kind):
    mus, springs, x, p, name = make_structure(kind)
    n = len(mus)

    def accel(x, p, t):
        aa = np.zeros(n)
        for (i, j, k) in springs:
            f = -k * (x[i] - x[j])
            aa[i] += f / mus[i]
            aa[j] -= f / mus[j]
        if t >= N_RELAX:
            aa = aa + G_FIELD
        return aa

    def E_total(x, p):
        E = np.sum(p**2 / (2 * mus))
        for (i, j, k) in springs:
            E += 0.5 * k * (x[i] - x[j])**2
        return E

    S_list = np.empty(NT)
    X_list = np.empty(NT)
    for t in range(NT):
        a_now = accel(x, p, t)
        p = p + a_now * mus * DT / 2
        x = x + p / mus * DT
        a_new = accel(x, p, t)
        p = p + a_new * mus * DT / 2
        S_list[t] = E_total(x, p) / E_TOT if t < N_RELAX else np.nan
        X_list[t] = np.sum(mus * x) / np.sum(mus)
    return S_list, X_list, name

structs = ['A', 'B', 'C', 'D']
S_all, X_all, names = [], [], []
for k in structs:
    S, X, nm = run(k)
    S_all.append(S); X_all.append(X); names.append(nm)

# ---------- CU1：能量守恒（弛豫段） ----------
for nm, S in zip(names, S_all):
    dev = np.nanmax(np.abs(S[:N_RELAX] - 1.0))
    results.append(('CU1-' + nm[:5], '弛豫段总能量 |S-1| 最大', dev,
                    '<= 1e-6（Verlet 预算 (omega_max*DT)^2）', dev <= 1e-6))

# ---------- CU2：自由下落质心加速度结构无关 ----------
W_WIN = 2000      # 窗平均半宽（步）

def accel_est(X, W):
    """窗平均速度的中心差分估计质心加速度（窗宽 2W，中心间隔 1 步）。"""
    fall = X[N_RELAX:]
    v = (fall[W:] - fall[:-W]) / (W * DT)
    return np.median(np.diff(v) / DT)

# 阈值依据（2026-10-10 修订）：判据要排除的是等效原理破坏带来的结构依赖，
# 该效应的量级为 O(1) 至 1e-3（Eotvos 参数的物质依赖信号即在此量级）；
# 数值噪声的实测上界由多档窗宽扫描给出（见下，随窗宽变化、不随结构变化）。
# 阈值 1e-6 取两者之间：高于噪声上界约三个量级，低于物理信号约三个量级。
# 说明：朴素的浮点舍入预算（随机游走模型）经试算给出 1.6e-7，高估实测约 450 倍，
# 且多档扫描不支持 1/sqrt(W) 标度，故不采用该模型作阈值来源。
THR_CU = 1e-6

g_est = np.array([accel_est(X, W_WIN) for X in X_all])
spread_g = (g_est.max() - g_est.min()) / abs(g_est.mean())

# 多档窗宽扫描：散布随窗宽变化，判据取所有档位的最大值（不依赖单一窗宽选择）
scale_rows = []
for W in (500, 1000, 2000, 4000):
    ge = np.array([accel_est(X, W) for X in X_all])
    sp = (ge.max() - ge.min()) / abs(ge.mean())
    scale_rows.append((W, sp))
spread_max = max(sp for _, sp in scale_rows)

results.append(('CU2', '自由下落质心加速度四结构相对散布', spread_g,
                '<= %.0e（多档窗宽最大值 %.1e）' % (THR_CU, spread_max),
                spread_g <= THR_CU and spread_max <= THR_CU))

# ---------- CU3：加速度值 = 场强 ----------
results.append(('CU3', '质心加速度 / G_FIELD', abs(g_est.mean()) / G_FIELD,
                '≈ 1（|a/G-1| <= %.0e）' % THR_CU,
                abs(abs(g_est.mean()) / G_FIELD - 1.0) <= THR_CU))

# ---------- CU4：Eotvos 参数 ----------
eta = 2 * abs(g_est[0] - g_est[2]) / (g_est[0] + g_est[2])
results.append(('CU4', 'Eotvos 参数（单质点对三质点链）', eta,
                '<= %.0e' % THR_CU, eta <= THR_CU))

# ---------- 打印 ----------
print('=' * 72)
print('组成普适判据的最小验证（第 459 条）：CU1 至 CU4')
print('=' * 72)
n_pass = 0
for tag, name, val, expect, ok in results:
    n_pass += ok
    print('%s %-36s 实测 %.12g  期望 %s  %s' % (
        tag, name, val, expect, 'PASS' if ok else 'FAIL'))
print('-' * 72)
print('自由下落质心加速度四结构：', np.array2string(g_est, precision=12))
print('-' * 72)
print('多档窗宽扫描（判据取所有档位最大值；散布随窗宽变化、不随结构变化）')
for W, sp in scale_rows:
    print('  W = %5d : 散布 = %.3e' % (W, sp))
print('-' * 72)
print('判据 %d/%d 通过' % (n_pass, len(results)))
assert n_pass == len(results), 'FAIL: 判据未全过'

# ---------- 图 ----------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
ax = axes[0]
for nm, S in zip(names, S_all):
    tt = np.arange(NT)[np.isfinite(S)] * DT
    ax.plot(tt, S[np.isfinite(S)] - 1.0, lw=0.8, label=nm)
ax.set_xlabel('time (model)')
ax.set_ylabel('E(t)/E_tot - 1')
ax.set_ylim(-2e-6, 2e-6)
ax.set_title('(a) CU1: energy conserved (4 structures)')
ax.legend(fontsize=8)
ax = axes[1]
for nm, X in zip(names, X_all):
    tt = np.arange(NT) * DT
    ax.plot(tt, X, lw=0.8, label=nm)
ax.axvline(N_RELAX * DT, color='k', ls=':', lw=1, label='gravity on')
ax.set_xlabel('time (model)')
ax.set_ylabel('center position X')
ax.set_title('(b) free fall: center acceleration')
ax.legend(fontsize=8)
ax = axes[2]
ax.bar(range(4), g_est / G_FIELD, width=0.5)
ax.axhline(1.0, color='k', ls='--', lw=1)
ax.set_xticks(range(4)); ax.set_xticklabels(names)
ax.set_ylabel('a_cm / G_FIELD')
ax.set_ylim(0.99, 1.01)
ax.set_title('(c) CU4: eta = 0 (universal free fall)')
plt.tight_layout()
plt.savefig(r'D:\software\TianXingFC\images\composition_universality.png', dpi=150)
print('figure saved: images/composition_universality.png')
