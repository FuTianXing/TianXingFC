# -*- coding: utf-8 -*-
r"""
tension_reconciliation.py

[张力对表交叉核对轮] 第 598 条台阶位错能量对第 341 条台阶面壁张力闭式的
逐项对表（判据 R1 至 R4，第 599 条）

第 598 条发现 FK 口径位错能量 +0.3270（J=1），第 341 条台阶面壁张力线的
大刚度档 sigma_1D 外推 K=1 约 0.477（表止于 0.3）。两条线同族（一维链、键
刚度 1、异类势强度同位），装配不同：
  甲装配（第 341 条同构）：单通道 v，E = (1/2)Sum(dv)^2 + 2K Sum(1-cos 2πv)，
    wrap 扣 Q=1，sigma_1D = 台阶极小能量（真空 = 全平 v 整数，能量 0）。
  乙装配（第 598 条同构）：三分量 FK 扣 Q = (1,0,0)（逐股整化），
    E = (1/2)Sum_a (dv^a)^2 + K Sum_{a<b} (1-cos 2π(v^a-v^b))。
乙的自由度包含甲的嵌入（v^2 = v^3 = 0、Q = (1,0,0) 时与甲逐位同构），同 K
下 E_乙 <= sigma_1D 为结构不等式。

首版两处协议错（教训入库）：其一，甲的对称 tanh 初值把中心格点钉死在
v = 0.5（势极大点，对称键力平衡，梯度机器零的鞍点），E = 4.2469，破对称
微扰后收敛真极小 0.4878（两格快跳形态，与第 341 条两格族一致）；其二，
乙的 wrap 键误用统一 Q=1，平股 v^2 v^3 被强加一圈伪扇区（终态能量虚高
0.333），修复为逐股 Q = (1,0,0)。首版 R3 的"v^1 喂甲"分解设计错误（乙的
能量结构靠 v^2 跟随把 delta 阶跃化，v^1 单独不是甲装配的合法竞争剖面，
喂甲给 165 的无意义读数），改为键/势构成分解。

判据（先立理论期望再实现）

R1（甲装配复现）：破对称初值下 K = 0.1/0.2/0.3 的 sigma_1D 对照第 341 条
  读数总表（0.345133 / 0.405285 / 0.431444），偏差 < 0.1%。
R2（结构不等式与趋势）：同 K 档 E_乙 <= sigma_1D（实现校验，违反即装配
  有错），读差值 Delta(K) 及其 K 依赖。
R3（K=1 构成分解，核心）：甲极小与乙极小各分解为键项与势项：
  势_乙（delta 整数化程度）对 势_甲（中间值格点的势罚），
  键_乙（跟随分摊）对 键_甲。先立预期：势_乙 << 势_甲。
R4（恒真防护，两层）：其一纯函数同构：同一 v 上 energy_A(v,K) 与
  energy_B([v,0,0],K) 逐位一致（< 1e-12）；其二弛豫一致：破对称同扰动下
  甲弛豫与乙冻结弛豫终态能量一致（< 1e-9）。

可逆性：判据敏感（两装配、K 网格、破对称扰动、弛豫协议绑定）。
运行：python 验证脚本/tension_reconciliation.py
"""
import math
import time

import numpy as np

T_START = time.time()
N = 256
TWO_PI = 2.0 * math.pi
NSTEPS = 220000
PAIRS = ((0, 1), (0, 2), (1, 2))
Q_BETA = (1, 0, 0)

print('=' * 76)
print('张力对表交叉核对轮：第 598 条台阶位错对第 341 条壁张力闭式（第 599 条）')
print('=' * 76)
print()
print('甲装配（第 341 条同构）：单通道 E = (1/2)Sum(dv)^2 + 2K Sum(1-cos 2πv)，Q=1')
print('乙装配（第 598 条同构）：三分量 FK 扣 Q=(1,0,0)，E = (1/2)Sum(dv^a)^2')
print('  + K Sum_{a<b}(1-cos 2π(v^a-v^b))；乙包含甲的嵌入（v^2=v^3=0）。')
print()


def chain_dvQ(v_a, Q_a):
    dv = np.diff(np.concatenate([v_a, v_a[:1]]))
    dv[-1] += Q_a
    return dv


def energy_A(v, K, Q=1):
    dv = chain_dvQ(v, Q)
    return 0.5 * float(np.sum(dv ** 2)) + 2.0 * K * float(np.sum(1.0 - np.cos(TWO_PI * v)))


def grad_A(v, K, Q=1):
    dv = chain_dvQ(v, Q)
    return (np.roll(dv, 1) - dv) + 4.0 * math.pi * K * np.sin(TWO_PI * v)


def energy_B(v3, K, Q=Q_BETA):
    dv = [chain_dvQ(v3[a], Q[a]) for a in range(3)]
    e_el = 0.5 * sum(float(np.sum(d ** 2)) for d in dv)
    e_het = K * sum(float(np.sum(1.0 - np.cos(TWO_PI * (v3[a] - v3[b]))))
                    for (a, b) in PAIRS)
    return e_el + e_het


def grad_B(v3, K, Q=Q_BETA, freeze=False):
    dv = [chain_dvQ(v3[a], Q[a]) for a in range(3)]
    g = np.empty_like(v3)
    for a in range(3):
        g[a] = np.roll(dv[a], 1) - dv[a]
    for (a, b) in PAIRS:
        d = TWO_PI * (v3[a] - v3[b])
        s = K * TWO_PI * np.sin(d)
        g[a] += s
        g[b] -= s
    if freeze:
        g[1] = 0.0
        g[2] = 0.0
    return g


def decompose_B(v3, K, Q=Q_BETA):
    dv = [chain_dvQ(v3[a], Q[a]) for a in range(3)]
    e_el = 0.5 * sum(float(np.sum(d ** 2)) for d in dv)
    e_het = K * sum(float(np.sum(1.0 - np.cos(TWO_PI * (v3[a] - v3[b]))))
                    for (a, b) in PAIRS)
    return e_el, e_het


def relax_A(v0, K, nsteps=NSTEPS):
    v = v0.copy()
    eta = min(0.01, 0.3 / (4.0 + 8.0 * math.pi ** 2 * K))
    for step in range(nsteps):
        v = v - eta * grad_A(v, K)
        if step % 10000 == 9999:
            gv = grad_A(v, K)
            if float(np.max(np.abs(gv))) < 1e-10:
                return v, energy_A(v, K), True
    return v, energy_A(v, K), False


def relax_B(v0, K, freeze=False, nsteps=NSTEPS, Q=Q_BETA):
    v = v0.copy()
    eta = min(0.01, 0.3 / (4.0 + 8.0 * math.pi ** 2 * K))
    for step in range(nsteps):
        v = v - eta * grad_B(v, K, Q, freeze)
        if freeze:
            v[1] = 0.0
            v[2] = 0.0
        if step % 10000 == 9999:
            gv = grad_B(v, K, Q, freeze)
            if float(np.max(np.abs(gv))) < 1e-10:
                return v, energy_B(v, K, Q), True
    return v, energy_B(v, K, Q), False


def init_wall(N, w=2.0):
    n = np.arange(N)
    return 0.5 * (1.0 + np.tanh((n - N / 2) / w))


def break_sym(v):
    """破对称扰动：对称初值会把台阶中心格点钉死在 v=0.5 的鞍点。"""
    n = np.arange(len(v))
    return v + 1e-3 * np.sin(TWO_PI * 3.0 * n / len(v) + 0.7)


def two_seed(N, eps):
    """两格快跳种子：整数平台加两格非整分配（eps, 1-eps）。"""
    v = np.zeros(N)
    v[N // 2] = eps
    v[N // 2 + 1] = 1.0 - eps
    return v


def eps_star(K):
    """第 341 条尾闭式内点平衡（超越方程逐位刻画快跳极小）：
    eps*(3-exp(-kappa)) + 2πK sin(2π eps) = 1，cosh kappa = 1 + 2π^2 K。
    粗网格两格种子会坍缩出一格跳/光滑支（K=1 的 eps* ≈ 0.024 贴近
    整数），必须按解析平衡点构造种子。"""
    kap = math.acosh(1.0 + 2.0 * math.pi ** 2 * K)
    c = 3.0 - math.exp(-kap)
    lo, hi = 1e-6, 0.5
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        f = mid * c + 2.0 * math.pi * K * math.sin(TWO_PI * mid) - 1.0
        if f > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def sigma_global(K, nsteps=NSTEPS):
    """甲装配全局极小：尾闭式平衡种子（快跳支）与光滑支种子取最低。
    K < K* ≈ 0.05 时快跳支不存在，eps_star 种子弛豫滑回光滑支，取
    min 自动选对。"""
    best = None
    seeds = [two_seed(N, eps_star(K)),
             break_sym(init_wall(N, 2.0)), break_sym(init_wall(N, 8.0))]
    for sd in seeds:
        _, e, _ = relax_A(sd, K, nsteps)
        if best is None or e < best[1]:
            best = (sd, e)
    return best[1]


def init_beta(N, w=2.0):
    return np.stack([init_wall(N, w), np.zeros(N), np.zeros(N)])


rng = np.random.default_rng(20261011)

# ---------- R0：纯函数同构（R4 之一，先验门） ----------
print('--- R4a 纯函数同构（energy_A(v) 对 energy_B([v,0,0])，随机场 20 例）---')
dev_max = 0.0
for _ in range(20):
    v_t = rng.normal(0.3, 0.25, N)
    v_t = v_t - np.floor(v_t)
    e_a = energy_A(v_t, 1.0)
    e_b = energy_B(np.stack([v_t, np.zeros(N), np.zeros(N)]), 1.0)
    dev_max = max(dev_max, abs(e_a - e_b))
print('   max |E_甲 - E_乙冻结| = %.2e（判据 < 1e-12）' % dev_max)
r4a_ok = dev_max < 1e-12
print('   [%s] R4a' % ('PASS' if r4a_ok else 'FAIL'))
print()

# ---------- R1：甲装配复现第 341 条 ----------
print('--- R1 甲装配复现（破对称初值，第 341 条读数总表）---')
REF = {0.1: 0.345133, 0.2: 0.405285, 0.3: 0.431444}
r1_ok = True
sigA = {}
for K, ref in REF.items():
    e = sigma_global(K)
    dev = abs(e - ref) / ref
    sigA[K] = e
    print('   K = %-4g：sigma_1D = %.6f（第 341 条 %.6f，偏差 %.4f%%）'
          % (K, e, ref, 100 * dev))
    if dev > 1e-3:
        r1_ok = False
print('   R1b 诊断（装配复原检验）：eps_star 种子终态能量 =')
for K in REF:
    es = eps_star(K)
    _, e_seed, conv = relax_A(two_seed(N, es), K)
    _, e_sm, _ = relax_A(break_sym(init_wall(N, 2.0)), K)
    print('   K = %-4g：eps* = %.4f，快跳种子终态 = %.6f，光滑支 = %.6f'
          '（快跳种子滑向光滑支：反推装配无更深的快跳极小）'
          % (K, es, e_seed, e_sm))
print('   [%s] R1（FAIL = 反推装配未复原第 341 条模型：其快跳支在反推'
      '装配下不是极小，须回源第 337/341 条注记核对完整装配；本轮对表'
      '读数降级为条件性）' % ('PASS' if r1_ok else 'FAIL'))
print()

# ---------- R2：结构不等式与趋势 ----------
print('--- R2 同档对照（乙位错能量 <= sigma_1D）---')
r2_ok = True
eB = {}
for K in [0.1, 0.2, 0.3, 1.0]:
    v3_end, e, conv = relax_B(init_beta(N), K)
    eB[K] = e
    if K != 1.0:
        delta = sigA[K] - e
        tag = '（乙 <= 甲）' if delta >= -1e-9 else '【违反结构不等式】'
        print('   K = %-4g：E_乙 = %.6f（收敛 %s），sigma_1D = %.6f，Delta = %+.6f %s'
              % (K, e, conv, sigA[K], delta, tag))
        if delta < -1e-9:
            r2_ok = False
    else:
        print('   K = %-4g：E_乙 = %.6f（收敛 %s）（甲档见 R3）' % (K, e, conv))
print('   [%s] R2' % ('PASS' if r2_ok else 'FAIL'))
print()

# ---------- R3：K=1 构成分解（核心） ----------
print('--- R3 K=1 构成分解（键/势两列，甲极小对乙极小）---')
sig1 = sigma_global(1.0)
v1_end, _, conv_a = relax_A(two_seed(N, eps_star(1.0)), 1.0)
el_a = sig1 - 2.0 * 1.0 * float(np.sum(1.0 - np.cos(TWO_PI * v1_end)))
pot_a = sig1 - el_a
v3_end, e_b1, conv_b = relax_B(init_beta(N), 1.0)
el_b, pot_b = decompose_B(v3_end, 1.0)
asym = 0.5 - 1.0 / (4.0 * math.pi ** 2) + 3.0 / (16.0 * math.pi ** 4)
print('   甲极小：E = %.6f = 键 %.6f + 势 %.6f（收敛 %s；渐近式 %.6f）'
      % (sig1, el_a, pot_a, conv_a, asym))
print('   乙极小：E = %.6f = 键 %.6f + 势 %.6f（收敛 %s）'
      % (e_b1, el_b, pot_b, conv_b))
print('   势比 势_乙/势_甲 = %.4f（先立 << 1：乙的 delta 整数化压势）'
      % (pot_b / pot_a))
print('   键比 键_乙/键_甲 = %.4f（跟随分摊读数）' % (el_b / el_a))
print('   差值 sigma_1D(1) - E_乙(1) = %+.6f' % (sig1 - e_b1))
r3_ok = pot_b < pot_a and e_b1 < sig1
print('   [%s] R3' % ('PASS' if r3_ok else 'FAIL'))
print()

# ---------- R4b：弛豫一致（恒真防护第二层） ----------
print('--- R4b 弛豫一致（破对称同扰动：甲弛豫 对 乙冻结弛豫）---')
v_pert = break_sym(init_wall(N))
_, eA_f, _ = relax_A(v_pert, 1.0)
v0_f = np.stack([v_pert, np.zeros(N), np.zeros(N)])
_, eB_f, _ = relax_B(v0_f, 1.0, freeze=True, Q=(1, 0, 0))
dev4 = abs(eA_f - eB_f)
print('   E_甲 = %.10f，E_乙冻结 = %.10f，差 = %.2e（判据 < 1e-9）'
      % (eA_f, eB_f, dev4))
r4_ok = r4a_ok and dev4 < 1e-9
print('   [%s] R4（含 R4a 纯函数层）' % ('PASS' if r4_ok else 'FAIL'))
print()

# ---------- 图 ----------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2))
Ks = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
sig_curve = []
for K in Ks:
    if K in sigA:
        sig_curve.append(sigA[K])
    elif K == 1.0:
        sig_curve.append(sig1)
    elif K >= 0.1:
        sig_curve.append(sigma_global(K))
    else:
        _, e, _ = relax_A(break_sym(init_wall(N)), K)
        sig_curve.append(e)
ax1.semilogx(Ks, sig_curve, 'o-', color='#1f77b4',
             label='甲 sigma_1D(K)（第 341 条同构，本轮）')
ax1.semilogx(list(REF.keys()), list(REF.values()), 's', color='#9467bd',
             label='第 341 条表值')
ax1.semilogx([K for K in [0.1, 0.2, 0.3] if K in eB],
             [eB[K] for K in [0.1, 0.2, 0.3]], 'd-', color='#d62728',
             label='乙 E_位错(K)（第 598 条同构）')
ax1.axhline(asym, color='gray', ls=':', lw=1,
            label='大 K 渐近 0.5 - 1/(4π²)')
ax1.set_xlabel('K（异类势强度，对数轴）')
ax1.set_ylabel('位错能量 / 张力')
ax1.set_title('两装配同档对照（第 599 条）', fontsize=10)
ax1.legend(fontsize=8)
bars = [pot_a, pot_b, el_a, el_b]
ax2.bar(['势_甲', '势_乙', '键_甲', '键_乙'], bars,
        color=['#ff7f0e', '#2ca02c', '#ff7f0e', '#2ca02c'])
for i, b in enumerate(bars):
    ax2.text(i, b + 0.006, '%.4f' % b, ha='center', fontsize=9)
ax2.set_ylabel('K=1 能量构成')
ax2.set_title('K=1 构成分解：势比 %.3f，键比 %.3f'
              % (pot_b / pot_a, el_b / el_a), fontsize=10)
fig.tight_layout()
fig.savefig(r'D:\software\TianXingFC\images\recon599_tension.png', dpi=150)
plt.close(fig)
print('图已存 images/recon599_tension.png')
print()

print('=' * 76)
print('汇总（第 599 条）：R1 %s，R2 %s，R3 %s，R4 %s'
      % ('PASS' if r1_ok else 'FAIL', 'PASS' if r2_ok else 'FAIL',
         'PASS' if r3_ok else 'FAIL', 'PASS' if r4_ok else 'FAIL'))
print('K=1（条件性对表，甲 = 反推装配全局极小）：sigma_1D = %.6f，'
      'E_乙 = %.6f，差 %+.6f（势比 %.4f，键比 %.4f）'
      % (sig1, e_b1, sig1 - e_b1, pot_b / pot_a, el_b / el_a))
print('判读与登记见验证注记。')
print()
print('运行时间 %.1f 秒' % (time.time() - T_START))
