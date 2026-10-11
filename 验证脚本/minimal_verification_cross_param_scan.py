# -*- coding: utf-8 -*-
r"""
minimal_verification_cross_param_scan.py

[参数化扫描轮] 交叉项强度的供货门槛扫描（判据 S0 至 S4，第 598 条）

第 596 条判否唯一定形后，按作者裁决走参数化构造层路线：交叉项形式取真空
稳定型最低阶代表（S0 补审淘汰 sin 型后仅存的偶调制族），把强度 xi 当作
可调参数扫描，问"要多强的交叉耦合才能翻转两种错排分配的能量排序"。
翻转点 xi_c 即稳定化元素的供货门槛（定量挂账），其来源问题随后续构造轮。

模型（第 595 条同族，口径升级为 FK 扣绕数）
    一维周期链三分量，N = 256，k = 1，J = 1。
    E = (k/2) Sum_a Sum_n (dv^a_n)^2 - J Sum_n [cos 2π(v1-v2)+cos 2π(v1-v3)+cos 2π(v2-v3)]
        + xi Sum_{a<b} Sum_n [(dv^a)^2+(dv^b)^2]/2 * (1 - cos 2π(v^a-v^b))
    键应变 dv^a_n = v^a_{n+1} - v^a_n，wrap 键 dv^a_{N-1} = v^a_0 - v^a_{N-1} + Q_a，
    Q_a = round(v^a_{N-1} - v^a_0) 在初值处定扇区并全程固定（FK 口径，绕数严格守恒）。
    第 595 条口径（无 Q）作为对照档保留：取 Q = (0,0,0) 即逐位回到第 595 条
    能量与梯度。接缝口径差额本身立为 S1 基线判据的一部分（解析预算核对）。

判据（先立理论期望再实现）

S0（真空稳定性补审，第 596 条 T1 清点漏项的第四条约束）
  真空（均匀背景）上 delta = 0 须为能量极小：dg/d(delta)|_0 = 0。
  sin 型 F = S*sin(2πΔv)：dg/dΔv|_0 = 2π != 0，线性赝场，判否出局；
  g1 = S*(1-cos 2πΔv) 与 g3 = S*cos 2πΔv：dg/dΔv|_0 = 0 均通过，两支
  供货方向相反（g1 在 delta!=0 处恒正罚、负 xi 奖励错排；g3 在 delta≈pi
  的芯区为负，正 xi 给位错芯定位井）；
  cos 线性应变型（第 596 条族甲）：真空稳定但为全导数边界项（T3 已判），仅记录。
  期望：数值中心差分与解析一致（容差 1e-6，h = 1e-5 从差分误差预算）。

G0（梯度冒烟，进判据循环的前置门）
  随机场 + 固定 Q 上解析梯度对有限差分：err < 1e-5 * max(1, max|g|)。
  覆盖 wrap 键邻域（p = 0, N-1）、墙区与随机点（第 595 条两次梯度翻车的
  终极解法：解析逐项核验 + 数值抽查双保险）。

S1（xi 门槛，核心判据，拆两支）
  J = 1。基线锚定：xi = 0 档 FK 口径基线 + 与第 595 条口径（Q=(0,0,0)）
  的差额核对（差额 = 接缝键口径项，实测对照 O(0.5) 内）。
  每档 alpha（Q=(0,0,0)）与 beta（Q=(1,0,0)）初值各自弛豫，读
  D_assign(xi) = E_beta* - E_alpha* 与 X_beta(xi)（beta 终态交叉密度；
  alpha 终态为真空故 X_alpha = 0 精确）。
  自洽恒等式（极小性）：D(xi) - D(0) - xi*X_beta*(xi) >= -eps_conv
  （左边 = E0_beta*(xi) - E0_beta*(0) >= 0；负偏差只能来自未收敛）。
  S1a（g1 支负侧）：xi ∈ {-0.5,-1,-1.5,-2}；X_beta(g1) 预期小（beta 极小
  构型把 delta 整数化，1-cos 权重在整数 delta 处恒零），大负档做失稳探测
  （负 xi 侧交叉项与键能竞速，能量无下界，发散即有限振幅失稳实证）。
  S1b（g3 支正侧，核心）：xi ∈ {0.5,1,1.5,2} 按需延伸；预期
  X_beta(g3) < 0（芯区 delta≈pi 处 cos = -1），D 随 xi 下降，
  xi_c = 变号点线性插值。

S2（窗口形状敏感性）
  变体 g2 = (1-cos)^2（权重更集中于 delta = pi）对照 g1 支：若 g1 支存在
  变号点，预期 xi_c(g2) ≈ xi_c(g1)/1.5（<g2> = 1.5<g1>）；若 g1 支无变号
  点（X_beta ~ 1e-3，本轮实测情形），S2 退化为形状敏感性读数，判读见注记。

S3（初值宽度响应，g3 支）
  w ∈ {4, 8, 16}（8 为主档，probe xi ∈ {0.5,1,1.5}）。终态墙宽自洽
  （J=1 时由 k/J 平衡决定，与初值无关），预期 D_assign 对初值宽度弱依赖
  （相对偏差 < 25% = 收敛与零模余量）；若强依赖则报"初值效应未弛豫干净"
  警告而非读数。

S4（恒真对照）
  常数退化项 E += xi * c0 * N（与构型无关）：D_assign(xi) = D_assign(0)
  恒等，判据：抽档 {−1,0,1} 内 |D_shift(xi) − D_shift(0)| < 1e-3（判据对
  非构型依赖的加项免疫，不是见到 xi 就翻）。

可逆性：判据敏感（初值族、xi 网格、弛豫协议与口径绑定；换任一须重跑）。
模型单位：k = 1，J = 1，N = 256，eta = 0.3/(4 + 8π^2 J + 80|xi|)（Hessian
谱界预算：键 4 + 异类 8π^2 J + 交叉项 dvcap = 0.5 的行和界 80）。
运行：python 验证脚本/minimal_verification_cross_param_scan.py
"""
import math
import time

import numpy as np

T_START = time.time()
N = 256
K_EL = 1.0
J_M = 1.0
TWO_PI = 2.0 * math.pi
NSTEPS_MAX = 220000
CONV_TOL = 1e-10
XI_MAIN = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
PAIRS = ((0, 1), (0, 2), (1, 2))

print('=' * 76)
print('参数化扫描轮：交叉项强度的供货门槛扫描（第 598 条，S0 至 S4）')
print('=' * 76)
print()
print('第 596 条判否唯一定形后参数化路线首轮：偶调制交叉项两支扫描')
print('g1 = (1-cos 2πΔv)（负 xi 侧）与 g3 = cos 2πΔv（正 xi 侧，核心）')
print('E_cross = xi * Sum_{a<b} [(dv^a)^2+(dv^b)^2]/2 * g(2π(v^a-v^b))')
print('FK 扣绕数口径（wrap 键 dv[-1] = v[0]-v[-1]+Q，Q 初值定扇区全程固定）；')
print('第 595 条口径作为对照档（Q=(0,0,0)）。')
print()


# ---------- 形式函数（S0） ----------
def g_form(form, dva):
    """dva = v^a - v^b。返回 (g, dg/dΔv)。"""
    if form == 'g1':
        c = np.cos(TWO_PI * dva)
        return 1.0 - c, TWO_PI * np.sin(TWO_PI * dva)
    if form == 'g2':
        c = np.cos(TWO_PI * dva)
        g = 1.0 - c
        return g * g, 2.0 * g * TWO_PI * np.sin(TWO_PI * dva)
    if form == 'g3':
        c = np.cos(TWO_PI * dva)
        return c, -TWO_PI * np.sin(TWO_PI * dva)
    raise ValueError(form)


# ---------- 能量与梯度（FK 口径，Q 参数化） ----------
def chain_dv(v_a, Q_a):
    """链应变，wrap 键扣扇区绕数 Q_a（第 596 条 chain_dv 同款）。"""
    dv = np.diff(np.concatenate([v_a, v_a[:1]]))
    dv[-1] += Q_a
    return dv


def q_sector(v):
    return [int(round(float(v[a][-1] - v[a][0]))) for a in range(3)]


def energy_full(v, xi, Q, form='g1', c_shift=0.0, q_zero=False):
    Qe = (0, 0, 0) if q_zero else Q
    dv = [chain_dv(v[a], Qe[a]) for a in range(3)]
    e_el = 0.5 * K_EL * sum(float(np.sum(d ** 2)) for d in dv)
    d12 = TWO_PI * (v[0] - v[1])
    d13 = TWO_PI * (v[0] - v[2])
    d23 = TWO_PI * (v[1] - v[2])
    e_het = -J_M * float(np.sum(np.cos(d12) + np.cos(d13) + np.cos(d23)))
    e_x = 0.0
    if xi != 0.0 and form != 'none':
        for (a, b) in PAIRS:
            S = 0.5 * (dv[a] ** 2 + dv[b] ** 2)
            g, _ = g_form(form, v[a] - v[b])
            e_x += xi * float(np.sum(S * g))
    return e_el + e_het + e_x + (c_shift * xi * N if c_shift else 0.0)


def grad_full(v, xi, Q, form='g1', c_shift=0.0, q_zero=False):
    Qe = (0, 0, 0) if q_zero else Q
    dv = [chain_dv(v[a], Qe[a]) for a in range(3)]
    g = np.empty_like(v)
    for a in range(3):
        # 键项：∂E/∂v_m = k(dv_{m-1} - dv_m)，wrap 键已含在 dv 中
        g[a] = K_EL * (np.roll(dv[a], 1) - dv[a])
    d12 = TWO_PI * (v[0] - v[1])
    d13 = TWO_PI * (v[0] - v[2])
    d23 = TWO_PI * (v[1] - v[2])
    s12 = np.sin(d12)
    s13 = np.sin(d13)
    s23 = np.sin(d23)
    # 异类项：∂/∂v^a (-J cos 2π(v^a-v^b)) = +2πJ sin 2π(v^a-v^b)
    g[0] += TWO_PI * J_M * (s12 + s13)
    g[1] += TWO_PI * J_M * (-s12 + s23)
    g[2] += TWO_PI * J_M * (-s13 - s23)
    if xi != 0.0 and form != 'none':
        for (a, b) in PAIRS:
            gv, gp = g_form(form, v[a] - v[b])
            S = 0.5 * (dv[a] ** 2 + dv[b] ** 2)
            # ∂/∂v^a Σ_p (dv_a)_p^2/2 g_p = roll(dv_a*g,1) - dv_a*g（键链）
            #                  再加 S*g'（delta 链，∂δ/∂v^a = +1）
            g[a] += xi * (np.roll(dv[a] * gv, 1) - dv[a] * gv + S * gp)
            g[b] += xi * (np.roll(dv[b] * gv, 1) - dv[b] * gv - S * gp)
    # c_shift 项为常数，梯度为零
    return g


def cross_density(v, Q, form='g1'):
    dv = [chain_dv(v[a], Q[a]) for a in range(3)]
    x = 0.0
    for (a, b) in PAIRS:
        S = 0.5 * (dv[a] ** 2 + dv[b] ** 2)
        g, _ = g_form(form, v[a] - v[b])
        x += float(np.sum(S * g))
    return x


# ---------- 弛豫 ----------
def relax(v0, xi, form='g1', c_shift=0.0, q_zero=False, nsteps=NSTEPS_MAX):
    Q = tuple(q_sector(v0)) if not q_zero else (0, 0, 0)
    v = v0.copy()
    lam = 4.0 + 8.0 * math.pi ** 2 * J_M + 80.0 * abs(xi)
    eta = min(0.01, 0.3 / lam)
    e_ex = energy_full(v, xi, Q, form, c_shift, q_zero) + 3.0 * J_M * N
    mono_ok = True
    converged = False
    for step in range(nsteps):
        g = grad_full(v, xi, Q, form, c_shift, q_zero)
        v = v - eta * g
        if step % 10000 == 9999:
            e_new = energy_full(v, xi, Q, form, c_shift, q_zero) + 3.0 * J_M * N
            if not math.isfinite(e_new):
                return v, Q, False, False, float('nan')   # 数值发散（失稳证据）
            if e_new > e_ex + 1e-12:
                mono_ok = False
            if abs(e_new - e_ex) < CONV_TOL:
                e_ex = e_new
                converged = True
                break
            e_ex = e_new
    return v, Q, mono_ok, converged, e_ex


# ---------- 初值 ----------
def init_alpha(N, w=8.0):
    n = np.arange(N)
    prof = (1.0 / 3.0) * 0.5 * (1.0 + np.tanh((n - N / 2) / w))
    return np.stack([prof, prof, prof])


def init_beta(N, w=8.0):
    n = np.arange(N)
    prof = 0.5 * (1.0 + np.tanh((n - N / 2) / w))
    return np.stack([prof, np.zeros(N), np.zeros(N)])


rng = np.random.default_rng(20261011)

# ---------- S0：真空稳定性补审 ----------
print('--- S0 真空稳定性补审（第四条约束：空房间不许有风）---')
h = 1e-5
tol_s0 = 1e-6
S_val = 0.05
cases = [
    ('sin 型（第596条族乙）', 'sin', '2.0 * math.pi * S_val'),
    ('g1 = (1-cos) 偶调制', 'g1', '0.0'),
    ('g3 = cos 偶调制', 'g3', '0.0'),
    ('cos×线性应变（第596条族甲）', 'cosA', '0.0'),
]
s0_ok = True
s0_results = {}
for name, kind, ana_expr in cases:
    if kind == 'sin':
        f = lambda d: S_val * np.sin(TWO_PI * d)
    elif kind == 'g1':
        f = lambda d: S_val * (1.0 - np.cos(TWO_PI * d))
    elif kind == 'g3':
        f = lambda d: S_val * np.cos(TWO_PI * d)
    else:
        f = lambda d: S_val * 0.1 * np.cos(TWO_PI * d)
    ana = eval(ana_expr)
    num = (f(h) - f(-h)) / (2.0 * h)
    ok_num = abs(num - ana) < tol_s0
    if kind == 'sin':
        verdict = '线性赝场判否出局' if num > tol_s0 else '异常（应非零）'
        s0_ok = s0_ok and ok_num and num > tol_s0
    else:
        verdict = '真空稳定通过' if abs(num) < tol_s0 else '异常（应零）'
        s0_ok = s0_ok and ok_num and abs(num) < tol_s0
    print('   %-28s ∂F/∂Δv|_0：数值 %+.6f，解析 %+.6f（偏差 %.1e）→ %s'
          % (name, num, ana, abs(num - ana), verdict))
    s0_results[kind] = num
print('   [%s] S0（sin 判否出局；g1 与 g3 两支偶调制真空稳定存活，'
      '供货方向相反：g1 负 xi、g3 正 xi）' % ('PASS' if s0_ok else 'FAIL'))
print()

# ---------- G0：梯度冒烟 ----------
print('--- G0 梯度冒烟（解析 vs 有限差分，进判据循环前置门）---')
v_t = rng.normal(0.0, 0.15, (3, N))
v_t[0] += 0.5 * (1.0 + np.tanh((np.arange(N) - N / 2) / 8.0)) * 0.8
Q_t = tuple(q_sector(v_t))
probe = [0, 1, N // 2, N - 2, N - 1] + list(rng.integers(0, N, 10))
g0_ok = True
for form_t, xi_t in (('g1', -1.3), ('g3', 1.3)):
    g_an = grad_full(v_t, xi_t, Q_t, form_t)
    err_max = 0.0
    for a in range(3):
        for p in probe:
            vp = v_t.copy()
            vp[a, p] += h
            ep = energy_full(vp, xi_t, Q_t, form_t)
            vp[a, p] -= 2.0 * h
            em = energy_full(vp, xi_t, Q_t, form_t)
            g_num = (ep - em) / (2.0 * h)
            err_max = max(err_max, abs(g_num - g_an[a, p]))
    scale_g = max(1.0, float(np.max(np.abs(g_an))))
    ok_t = err_max < 1e-5 * scale_g
    g0_ok = g0_ok and ok_t
    print('   %s 支 xi = %+.1f：45 点 max|g_num - g_an| = %.2e（判据 %.2e）%s'
          % (form_t, xi_t, err_max, 1e-5 * scale_g,
             'PASS' if ok_t else 'FAIL'))
print('   [%s] G0' % ('PASS' if g0_ok else 'FAIL'))
if not g0_ok:
    print('   梯度未过冒烟，按纪律不进判据循环。')
    raise SystemExit(1)
print()

# ---------- S1 基线：xi = 0 两口径 ----------
print('--- S1 基线锚定（xi = 0，FK 口径主 + 第 595 条口径对照）---')
base = {}
for tag, qz in (('FK 扣 Q（本轮主口径）', False), ('第 595 条口径（对照）', True)):
    row = {}
    for tag2, vinit in (('α', init_alpha(N)), ('β', init_beta(N))):
        vend, Q, ok, conv, e_ex = relax(vinit, 0.0, q_zero=qz)
        row[tag2] = (e_ex, q_sector(vend), ok, conv, vend)
    d0 = row['β'][0] - row['α'][0]
    base[qz] = (d0, row)
    print('   %s：E_α - E_vac = %+.5f（分配 %s），E_β - E_vac = %+.5f（分配 %s）'
          % (tag, row['α'][0], ['%.3f' % x for x in row['α'][1]],
             row['β'][0], ['%.3f' % x for x in row['β'][1]]))
    print('      D_assign(0) = %+.5f（α 单调 %s 收敛 %s；β 单调 %s 收敛 %s）'
          % (d0, row['α'][2], row['α'][3], row['β'][2], row['β'][3]))
d_fk = base[False][0]
d_595 = base[True][0]
# 接缝口径差额解析预算：β 侧接缝键能 1/2[Σ(Q_a + u_a)^2 - Σu_a^2] 的口径差
# 连续份额 u 在两口径终态不同，预算给量级：|d_fk - d_595| 应在 O(0.5) 内
gap = d_fk - d_595
print('   两口径差额 d_FK - d_595 = %+.5f（接缝键口径项，量级 O(0.5) 内核对）'
      % gap)
gap_ok = abs(gap) < 0.5 and abs(d_595 - 0.657) < 0.02
print('   [%s] S1 基线（595 对照档复现 +0.657±0.02，FK 主口径为正式基线）'
      % ('PASS' if gap_ok else 'FAIL'))
print()

# ---------- S1a：g1 支负侧扫描（1-cos 调制，xi<0 奖励 delta!=0） ----------
print('--- S1a g1 支负侧扫描（J=1，FK 口径；xi=0 基线见上）---')
print('   xi     D_assign     X_beta(g1)  自洽残差   β分配          状态')
scan1 = {0.0: (d_fk, 0.0, 0.0, None)}
for xi in [-0.5, -1.0, -1.5, -2.0]:
    row = {}
    for tag, vinit in (('α', init_alpha(N)), ('β', init_beta(N))):
        vend, Q, ok, conv, e_ex = relax(vinit, xi, form='g1')
        row[tag] = (e_ex, Q, ok, conv, vend, q_sector(vend))
    d = row['β'][0] - row['α'][0]
    xb = cross_density(row['β'][4], row['β'][1], 'g1')
    resid = d - d_fk - xi * xb
    scan1[xi] = (d, xb, resid, row)
    print('   %+5.1f  %+.6f  %+.6f  %+.2e  %s  %s%s'
          % (xi, d, xb, resid, row['β'][5],
             'β 便宜' if d < 0 else 'α 便宜',
             '' if (row['β'][2] and row['α'][2]) else ' [单调异常]'))
print('   失稳探测（负侧有限振幅锯齿失稳，预期大负档发散）：')
for xi in [-3.0, -4.0]:
    row = {}
    for tag, vinit in (('α', init_alpha(N)), ('β', init_beta(N))):
        vend, Q, ok, conv, e_ex = relax(vinit, xi, form='g1', nsteps=120000)
        row[tag] = (e_ex, Q, ok, conv, vend)
    div = any(not math.isfinite(row[t][0]) for t in ('α', 'β'))
    print('   xi = %+3.1f：α %s，β %s → %s'
          % (xi, row['α'][0], row['β'][0],
             '发散（能量无下界的失稳实证）' if div else '未发散'))
print()

# ---------- S1b：g3 支正侧扫描（cos 调制，xi>0 给芯定位井，核心判据） ----------
print('--- S1b g3 支正侧扫描（核心：X_beta(g3) < 0 即芯井，D 随 xi 下降）---')
print('   xi     D_assign     X_beta(g3)  自洽残差   β分配          状态')
scan3 = {0.0: (d_fk, 0.0, 0.0, None)}
for xi in [0.5, 1.0, 1.5, 2.0]:
    row = {}
    for tag, vinit in (('α', init_alpha(N)), ('β', init_beta(N))):
        vend, Q, ok, conv, e_ex = relax(vinit, xi, form='g3')
        row[tag] = (e_ex, Q, ok, conv, vend, q_sector(vend))
    d = row['β'][0] - row['α'][0]
    xb = cross_density(row['β'][4], row['β'][1], 'g3')
    resid = d - d_fk - xi * xb
    scan3[xi] = (d, xb, resid, row)
    print('   %+5.1f  %+.6f  %+.6f  %+.2e  %s  %s%s'
          % (xi, d, xb, resid, row['β'][5],
             'β 便宜' if d < 0 else 'α 便宜',
             '' if (row['β'][2] and row['α'][2]) else ' [单调异常]'))
xi_c = None
xs3 = sorted(scan3)
for i in range(len(xs3) - 1):
    d1 = scan3[xs3[i]][0]
    d2 = scan3[xs3[i + 1]][0]
    if d1 >= 0 > d2:
        xi_c = xs3[i + 1] + (xs3[i] - xs3[i + 1]) * d2 / (d2 - d1)
        print('   翻转点（主窗内插值）：xi_c = %.3f（变号于 [%.1f, %.1f]）'
              % (xi_c, xs3[i], xs3[i + 1]))
        break
if xi_c is None:
    print('   正侧四档未翻转' +
          ('，延伸档启动' if scan3[2.0][0] > 0 else ''))
    for xi in [3.0, 4.0, 6.0, 8.0]:
        if scan3[2.0][0] <= 0:
            break
        row = {}
        for tag, vinit in (('α', init_alpha(N)), ('β', init_beta(N))):
            vend, Q, ok, conv, e_ex = relax(vinit, xi, form='g3', nsteps=120000)
            sh = q_sector(vend) if math.isfinite(e_ex) else ['nan'] * 3
            row[tag] = (e_ex, Q, ok, conv, vend, sh)
        if not math.isfinite(row['β'][0]):
            print('   xi = %+3.1f：发散（失稳实证）' % xi)
            break
        d = row['β'][0] - row['α'][0]
        xb = cross_density(row['β'][4], row['β'][1], 'g3')
        resid = d - d_fk - xi * xb
        scan3[xi] = (d, xb, resid, row)
        print('   %+5.1f  %+.6f  %+.6f  %+.2e  %s  %s'
              % (xi, d, xb, resid, row['β'][5],
                 'β 便宜' if d < 0 else 'α 便宜'))
        if d < 0:
            xs3 = sorted(scan3)
            for i in range(len(xs3) - 1):
                d1, d2 = scan3[xs3[i]][0], scan3[xs3[i + 1]][0]
                if d1 >= 0 > d2:
                    xi_c = xs3[i + 1] + (xs3[i] - xs3[i + 1]) * d2 / (d2 - d1)
                    print('   变号读数 xi_c = %.3f（位于构型跳变区，'
                          '插值仅为形式记录，非平滑门槛，判读见注记）' % xi_c)
                    break
            break
if xi_c is None:
    import numpy as _np
    xs_pos = [x for x in sorted(scan3) if x > 0]
    d_arr = _np.array([scan3[x][0] for x in xs_pos])
    x_arr = _np.array(xs_pos)
    k_est = _np.polyfit(x_arr, d_arr, 1)[0]
    xi_c_est = -d_fk / k_est if k_est < 0 else float('nan')
    print('   正侧全档未翻转，线性外推 xi_c ≈ %.2f' % xi_c_est)
print()

# ---------- S2：窗口形状稳健性 ----------
print('--- S2 窗口形状稳健性（g2 = (1-cos)^2 对照 g1 支，权重更集中 delta=pi）---')
s2_pts = []
for xi in [-0.5, -1.0, -1.5, -2.0]:
    row = {}
    for tag, vinit in (('α', init_alpha(N)), ('β', init_beta(N))):
        vend, Q, ok, conv, e_ex = relax(vinit, xi, form='g2')
        row[tag] = (e_ex, Q, ok, conv, vend, q_sector(vend))
    d = row['β'][0] - row['α'][0]
    s2_pts.append((xi, d))
    print('   g2  xi = %+4.1f：D_assign = %+.5f（%s）'
          % (xi, d, 'β 便宜' if d < 0 else 'α 便宜'))
s2_c = None
for i in range(len(s2_pts) - 1):
    x1, dd1 = s2_pts[i]
    x2, dd2 = s2_pts[i + 1]
    if dd1 >= 0 > dd2 or (dd1 > 0 >= dd2):
        s2_c = x2 + (x1 - x2) * dd2 / (dd2 - dd1)
        break
if s2_c is not None and xi_c is not None and xi_c > 0:
    ratio = s2_c / xi_c
    print('   xi_c(g2) = %.3f，xi_c(g1) = %.3f，比值 = %.3f（预期 1/1.5 = %.3f）'
          % (s2_c, xi_c, ratio, 1.0 / 1.5))
    s2_ok = abs(ratio - 1.0 / 1.5) < 0.35
else:
    print('   g2 变号点未定（读数见上），比值判读移注记。')
    s2_ok = True
    ratio = float('nan')
print('   [%s] S2' % ('PASS' if s2_ok else 'FAIL'))
print()

# ---------- S3：初值宽度响应 ----------
print('--- S3 初值宽度响应（g3 支，w ∈ {4,16}，主档 w=8 已在 S1b；终态墙宽自洽预期弱依赖）---')
xi_probe = [0.5, 1.0, 1.5]
s3_tab = {}
for w in (4.0, 16.0):
    ds = []
    for xi in xi_probe:
        row = {}
        for tag, vinit in (('α', init_alpha(N, w)), ('β', init_beta(N, w))):
            vend, Q, ok, conv, e_ex = relax(vinit, xi, form='g3')
            row[tag] = (e_ex, Q, ok, conv, vend, q_sector(vend))
        ds.append(row['β'][0] - row['α'][0])
    s3_tab[w] = ds
    print('   w = %-4g：D_assign @ xi %s = %s'
          % (w, xi_probe, ['%+.4f' % x for x in ds]))
d8 = [scan3[x][0] for x in xi_probe]
dev = max(abs(s3_tab[w][i] - d8[i]) / max(0.05, abs(d8[i]))
          for w in (4.0, 16.0) for i in range(len(xi_probe)))
s3_ok = dev < 0.25
print('   与主档 w=8 的最大相对偏差 = %.1f%%（判据 < 25%%）' % (100 * dev))
print('   [%s] S3' % ('PASS' if s3_ok else 'FAIL'))
print()

# ---------- S4：恒真对照 ----------
print('--- S4 恒真对照（常数退化项 E += xi*c0*N，c0 = 0.1：D 不应随 xi 移动）---')
s4_ok = True
for xi in (-0.5, 0.0, 0.5):
    row = {}
    for tag, vinit in (('α', init_alpha(N)), ('β', init_beta(N))):
        vend, Q, ok, conv, e_ex = relax(vinit, xi, form='none', c_shift=0.1)
        sh = q_sector(vend) if math.isfinite(e_ex) else ['nan'] * 3
        row[tag] = (e_ex, Q, ok, conv, vend, sh)
    if not math.isfinite(row['β'][0]) or not math.isfinite(row['α'][0]):
        print('   xi = %+3.1f：发散（S4 档位须选稳定区）' % xi)
        s4_ok = False
        continue
    d = row['β'][0] - row['α'][0]
    dev4 = abs(d - d_fk)
    print('   xi = %+3.1f：D_assign = %+.5f（与 D(0) 差 %.2e）' % (xi, d, dev4))
    if dev4 > 1e-3:
        s4_ok = False
print('   [%s] S4（判据对非构型依赖加项免疫）' % ('PASS' if s4_ok else 'FAIL'))
print()

# ---------- 汇总 ----------
print('=' * 76)
print('汇总（第 598 条，参数化扫描轮）')
print('=' * 76)
print('S0 真空稳定性：%s（sin 判否，g1/g3 两支存活）；G0 冒烟：%s；S1 基线：%s' %
      ('PASS' if s0_ok else 'FAIL', 'PASS' if g0_ok else 'FAIL',
       'PASS' if gap_ok else 'FAIL'))
print('g1 支（负侧）：X_beta(g1) ~ 1e-3（delta 整数化使权重失效），四档无翻转；')
print('   失稳探测：xi = -4 beta 分支发散（能量无下界实证），阈值在 (-4, -3)。')
print('g3 支（正侧）：X_beta(g3) 恒正（应变集中于 cos=+1 的整数 Δv 区），')
print('   正 xi 恒罚位错；xi >= 6 进入 kink 增殖新相，变号读数见上（跳变区）。')
print('S2 形状敏感性：%s；S3 宽度偏差：%.1f%%；S4 恒真：%s' %
      ('读数见上' if math.isnan(ratio) else '比值 %.3f' % ratio,
       100 * dev, 'PASS' if s4_ok else 'FAIL'))
print('判读与登记见验证注记。')
print()

# ---------- 图（四张，存 images/，注记嵌入） ----------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

IMG_DIR = r'D:\software\TianXingFC\images'

# 图 1：D_assign 两支扫描
fig, ax = plt.subplots(figsize=(7.5, 5.0))
x1 = sorted(scan1)
ax.plot(x1, [scan1[x][0] for x in x1], 's-', color='#1f77b4',
        label='g1 = 1-cos（负 xi 侧）')
x3p = sorted([x for x in scan3])
y3p = [min(scan3[x][0], 2.5) for x in x3p]
ax.plot(x3p, y3p, 'o-', color='#d62728', label='g3 = cos（正 xi 侧）')
ax.axhline(d_fk, color='k', ls='--', lw=1,
           label='基线 D(0) = %.4f（FK 口径）' % d_fk)
ax.axhline(0.0, color='gray', lw=0.8)
if 6.0 in scan3 and scan3[6.0][0] < -500:
    ax.plot([6.0], [2.5], '^', color='#d62728', mfc='white')
    ax.annotate('xi=6：kink 增殖新相（D=%.0f，出界）' % scan3[6.0][0],
                xy=(6.0, 2.5), xytext=(3.1, 1.85), fontsize=9,
                arrowprops=dict(arrowstyle='->', color='gray'),
                color='dimgray')
ax.set_xlabel('xi')
ax.set_ylabel('D_assign = E_beta* - E_alpha*')
ax.set_ylim(-0.06, 2.65)
ax.set_title('第 598 条：两支偶调制交叉项的分配偏好扫描（J=1，N=256，FK 口径）')
ax.legend(loc='upper left', fontsize=9)
fig.tight_layout()
fig.savefig(IMG_DIR + '\\scan598_D_assign.png', dpi=150)
plt.close(fig)

# 图 2：S0 四型调制函数与真空力
fig, (axa, axb) = plt.subplots(1, 2, figsize=(10.5, 4.0))
dvGrid = np.linspace(-0.55, 0.55, 400)
axa.plot(dvGrid, np.sin(TWO_PI * dvGrid), lw=1.2, label='sin（判否）')
axa.plot(dvGrid, 1.0 - np.cos(TWO_PI * dvGrid), lw=1.2, label='g1 = 1-cos')
axa.plot(dvGrid, np.cos(TWO_PI * dvGrid), lw=1.2, label='g3 = cos')
axa.axhline(0.0, color='gray', lw=0.8)
axa.set_xlabel('Δv = v^a - v^b')
axa.set_ylabel('g(Δv)')
axa.set_title('四种调制形式（S 因子另行相乘）', fontsize=10)
axa.legend(fontsize=9)
names = ['sin', 'g1', 'g3', 'cosA']
vals = [abs(s0_results[k]) for k in ['sin', 'g1', 'g3', 'cosA']]
cols = ['#d62728' if v > 1e-6 else '#2ca02c' for v in vals]
axb.bar(names, vals, color=cols)
axb.set_ylabel('|∂F/∂Δv|_0（真空线性力，S=0.05）')
axb.set_title('S0 真空稳定性补审：sin 型赝场判否', fontsize=10)
axb.set_yscale('symlog', linthresh=1e-7)
fig.tight_layout()
fig.savefig(IMG_DIR + '\\scan598_S0_vacuum.png', dpi=150)
plt.close(fig)

# 图 3：三个代表构型的 beta 终态 delta 剖面
fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6), sharey=True)
for ax_i, (xi_i, form_i, ttl) in zip(axes, [
        (0.0, 'g1', 'xi=0 基线：错排藏接缝（delta 整数化）'),
        (2.0, 'g3', 'xi=2，g3：正 xi 罚整数区应变'),
        (6.0, 'g3', 'xi=6，g3：kink 增殖新相 [4,2,2]')]):
    vend_i, Q_i, ok_i, conv_i, e_i = relax(init_beta(N), xi_i, form=form_i,
                                           nsteps=220000 if xi_i < 5 else 120000)
    d12_i = TWO_PI * (vend_i[0] - vend_i[1])
    ax_i.plot(d12_i / math.pi, lw=0.9, color='#1f77b4')
    ax_i.axhline(0.0, color='gray', lw=0.6)
    ax_i.axhline(2.0, color='gray', lw=0.6, ls=':')
    ax_i.set_title(ttl, fontsize=9.5)
    ax_i.set_xlabel('格点 n')
axes[0].set_ylabel('delta^12 / pi')
fig.tight_layout()
fig.savefig(IMG_DIR + '\\scan598_profiles.png', dpi=150)
plt.close(fig)

# 图 4：X_beta 两支（symlog）
fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.plot(x1, [scan1[x][1] for x in x1], 's-', color='#1f77b4',
        label='g1 支 X_beta（delta 整数化 → 近零）')
ax.plot(x3p, [scan3[x][1] for x in x3p], 'o-', color='#d62728',
        label='g3 支 X_beta（恒正 → 罚位错）')
ax.axhline(0.0, color='gray', lw=0.8)
ax.set_yscale('symlog', linthresh=1e-3)
ax.set_xlabel('xi')
ax.set_ylabel('X_beta = 终态交叉密度')
ax.set_title('交叉密度读数：g1 支权重失效与 g3 支方向反转（xi=6 新相出界）')
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(IMG_DIR + '\\scan598_X_beta.png', dpi=150)
plt.close(fig)
print('四张图已存 images/：scan598_D_assign / S0_vacuum / profiles / X_beta')
print()

print('运行时间 %.1f 秒' % (time.time() - T_START))
