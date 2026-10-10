# -*- coding: utf-8 -*-
r"""
minimal_verification_guard_layer_pair_annihilation.py

[守卫层对接判定] 绕数超选择与 toric code 能否阻止正反对的湮灭（判据 G1 至 G3，第 593 条）

第 484 条登记的后续项：环面芯-芯评估给出"吸引加湮灭、无稳定定域化对象"，但该最小
模型未含守卫层（绕数超选择、toric code 对易结构），故判负只对无守卫模型成立；
"有守卫时对湮灭是否被阻止"须另立判据（第 484 条注记 §五）。

本判据核一件事：守卫层保护的对象是什么，它管不管净绕数为零的对湮灭。

拓扑口径（首版教训：周期和 telescoping 恒零，绕数须按键差折叠整数计）
    Q = sum_n round(dv_n)，dv_n = v_{n+1} - v_n（wrap 键差取 v_0 - v_{N-1}）。
    均匀扭转背景（每键 1/N）的 Q = -1（wrap 键差约 -1 被计入）：Q = ±1 扇区的
    极小正是均匀扭转（扇区极小离域，既有结论），故"单缺陷"在环面上的自然形态
    就是背景加台阶。泡（先降后升）两壁折叠相消，Q = 0。

判据（先立理论期望再实现）

G1（经典层：扇区分类守恒的动力学面）
  模型同第 484 条（一维周期链 FK，k = 1，偏离量 v_n 周期）：
      E = sum_n (k/2)(v_{n+1} - v_n)^2 + K sum_n [1 - cos 2 pi v_n]
  三构型对照（期望先立）：
    a) Q = 0 对构型（小间距泡）：梯度流下能量大幅下降趋真空（Q: 0 -> 0，
       扇区内），守卫层不阻止湮灭通道；
    b) |Q| = 1 构型（均匀扭转背景 + 单台阶）：梯度流下 Q 恒 |1|，缺陷总数
       不灭（跨扇区被禁）；形状读数分层：钉扎强档 kink 保持局域，钉扎弱档
       趋向均匀扭转（复现"扇区极小离域"，与守卫结论正交）；
    c) 开放边界对照（守卫撤除，自由端）：kink 可滑出边界漏掉，缺陷计数降零；
       守卫来自净绕数守恒本身，对照确认来源。
  可逆性：判据稳健（拓扑守恒量与构型分类不依赖参数档；梯度流只是展示通道）。

G2（量子层：超选择与 toric code 的对接，结构核对，引既有读数）
  既有定案（不重跑）：
    - Z_N toric code：绕数是基态空间投影量子数，跨扇区一阶矩阵元为零
      （[[绕数承载层：电荷保护的波性侧归宿与粒子复合结构]] §3.2）；
    - 弦算符 U_C = exp(i sum phi_l) 是基态空间内的算符：端点产生一对缺陷，
      弦收缩即对湮灭（同注记 §3.1 与超选择论文的弦算符论证）。
  对接读法（先立）：Q = 0 对湮灭 = 弦算符收缩，落在基态空间内、不跨扇区，
  超选择不禁止；Q = ±1 单缺陷湮灭须把基态搬出扇区，跨扇区一阶矩阵元为零，
  被禁。守卫层守的是扇区标签（净绕数），守卫层不阻止净绕数为零的对湮灭。
  区分力（恒真检验）：判据必须区分两类过程：a) 档允许、b) 档被禁、c) 档守卫
  撤除后同为允许。三者若同判，判据无区分力，须换判据。

G3（结论面）
  若 G1、G2 成立：第 484 条的判负在完整框架下维持（湮灭趋势不被守卫层阻止），
  环面粒子定域化的承接者收窄为第 485 条内部壁承接（候选成立）与第 483 条
  边界环冻结（候选成立）；守卫层的职能确认为守扇区标签（电荷型守恒），
  与三层分工（拓扑序守电荷、格点内锁定给质量、弹性包层给宽度）自洽对接。

模型单位：k = 1，K 扫描档取第 484 条同款（1e-2 / 1e-4 / 1e-6 / 1e-10 / 0），
梯度流 ETA = 0.05，N = 256。确定性，无随机数，无图。

运行：python 验证脚本/minimal_verification_guard_layer_pair_annihilation.py
"""
import math
import time

import numpy as np

T_START = time.time()

N = 256
NSTEPS = 120000
ETA = 0.05
K_EL = 1.0
TWO_PI = 2.0 * math.pi
K_LIST = [1e-2, 1e-4, 1e-6, 1e-10, 0.0]

print('=' * 76)
print('守卫层对接判定：绕数超选择与 toric code 能否阻止正反对湮灭（第 593 条，G1 至 G3）')
print('=' * 76)
print()
print('第 484 条后续项：无守卫最小模型判负（吸引加湮灭），有守卫时是否被阻止须另立判据。')
print('拓扑口径：Q = round(v_{N-1} - v_0)（端点差整化）；扇区标签 Q 固定进入 wrap 键差。')
print()


# ---------- 模型与构型 ----------
def energy(v, K, Q=0):
    dv = np.empty(N)
    dv[:-1] = v[1:] - v[:-1]
    dv[-1] = v[0] - v[-1] + Q
    return 0.5 * K_EL * np.sum(dv ** 2) + K * np.sum(1.0 - np.cos(TWO_PI * v))


def winding(v):
    """净绕数：Q = round(v_{N-1} - v_0)（端点差整化）。
    教训：键差折叠对宽 kink 失效（绕数是分布量，光滑台阶的最大键差不过 0.5）；
    环面扇区由 u_{n+N} = u_n + m 给出，Q = m - N = round(v_{N-1} - v_0)。"""
    return float(round(v[-1] - v[0]))


def grad_flow_periodic(v, K, nsteps, Q=0):
    """过阻尼梯度流，扇区标签 Q 固定进入 wrap 键差（守卫层的经典实现）：
    键差数组 dv[-1] = v_0 - v_{N-1} + Q（背景键差 1 已按第 484 条口径扣除），
    g = K_EL (dv - roll(dv,1)) + 2 pi K sin(2 pi v)，v -= ETA * g。
    教训入库：梯度流符号须经解析 dE/dv 核验（首版两项全反致能量上升）；
    扇区标签不在动力学自由度里，正是超选择的经典对应。"""
    v = v.copy()
    q_rec = []
    for step in range(nsteps):
        dv = np.empty(N)
        dv[:-1] = v[1:] - v[:-1]
        dv[-1] = v[0] - v[-1] + Q
        # dE/dv_j = -(dv_j - dv_{j-1}) + 2 pi K sin(2 pi v_j)（解析逐项核验，
        # 同文件梯度符号已错两次：lap 版反复弹、键差版又反复弹，教训入库）
        g = -K_EL * (dv - np.roll(dv, 1)) + K * TWO_PI * np.sin(TWO_PI * v)
        v = v - ETA * g
        if step % 2000 == 0:
            q_rec.append(winding(v))
    q_rec.append(winding(v))
    return v, q_rec


def grad_flow_open(v, K, nsteps):
    """开放边界（自由端）：dE/dv 同解析式，端点用单边拉普拉斯；每步减均值固定质心。"""
    v = v.copy()
    kink_rec = []
    for step in range(nsteps):
        lap = np.zeros(N)
        lap[1:-1] = v[2:] - 2 * v[1:-1] + v[:-2]
        lap[0] = v[1] - v[0]
        lap[-1] = v[-2] - v[-1]
        g = -K_EL * lap + K * TWO_PI * np.sin(TWO_PI * v)
        v = v - ETA * g
        v -= v.mean()
        if step % 2000 == 0:
            kink_rec.append(kink_count(v))
    kink_rec.append(kink_count(v))
    return v, kink_rec


def kink_count(v):
    """缺陷计数：端点差整化（局域 kink 与均匀背景同一扇区，计数按 Q）。"""
    return int(abs(winding(v)))


def smooth_step(n, a, xi=3.0):
    return 0.5 * (1.0 + np.tanh((n - a) / xi))


def kink_pair_config(N, a, b, xi=3.0):
    """Q = 0 对构型：v 由 0 降 1（a 处）再升 1（b 处），第 484 条同款泡。"""
    n = np.arange(N)
    return -smooth_step(n, a, xi) + smooth_step(n, b, xi)


def single_kink_config(N, a, xi=3.0):
    """|Q| = 1 构型：单台阶 v = step（u 跨一个额外势阱），Q = round(v[-1]-v[0]) = 1。
    同扇区的延展形态为均匀扭转背景 v = n/N（Q 同为 1，扇区极小离域）。"""
    n = np.arange(N)
    return smooth_step(n, a, xi)


# ---------- 冒烟 ----------
print('--- 冒烟检查 ---')
v_pair = kink_pair_config(N, 60, 72)
v_kink = single_kink_config(N, 60)
q_pair0, q_kink0 = winding(v_pair), winding(v_kink)
print('   Q=0 对构型净绕数 = %.12f（期望 0）' % q_pair0)
print('   单 kink 构型净绕数 = %.12f（期望 +1）' % q_kink0)
print('   E_pair = %.6f，E_kink = %.6f（均应非零）' % (energy(v_pair, 1e-2), energy(v_kink, 1e-2)))
smoke = (abs(q_pair0) < 1e-12 and abs(q_kink0 - 1.0) < 1e-12
         and energy(v_pair, 1e-2) > 0 and energy(v_kink, 1e-2) > 0)
print('   [%s] 冒烟' % ('PASS' if smoke else 'FAIL'))
print()


# ---------- G1：经典层三构型对照 ----------
print('--- G1（经典层：扇区分类守恒的动力学面）---')
g1_ok = True

print('   a) Q=0 对构型（间距 12 泡，周期边界，扇区标签 Q=0 固定）：湮灭通道是否开放')
for K in K_LIST:
    v_end, q_rec = grad_flow_periodic(v_pair, K, NSTEPS, Q=0)
    e0, e_end = energy(v_pair, K, 0), energy(v_end, K, 0)
    drop = e0 / max(e_end, 1e-300)
    q_conserved = max(abs(q - q_pair0) for q in q_rec) < 1e-9
    open_channel = e_end < e0 and q_conserved
    print('      K=%-7g E %.4e -> %.4e（降 %.1e 倍），Q 全程 0=%s，读数=%s'
          % (K, e0, e_end, min(drop, 1e16), q_conserved,
             '湮灭通道开放' if open_channel else '通道异常'))
    if not open_channel:
        g1_ok = False

print('   b) |Q|=1 构型（单台阶，周期边界，扇区标签 Q=1 固定）：缺陷总数是否可灭')
for K in K_LIST:
    v_end, q_rec = grad_flow_periodic(v_kink, K, NSTEPS, Q=1)
    q_conserved = max(abs(q - q_kink0) for q in q_rec) < 1e-9
    dv_end = np.roll(v_end, -1) - v_end
    loc = np.max(np.abs(dv_end[:-1]))   # 内部键差（wrap 在 dv_end[N-1]，勿混入；
    # 首版取 [1:] 把 wrap 0.996 误读为局域台阶，能量读数已揭示极小达成）
    e_end = energy(v_end, K, 1)
    shape = '局域 kink 保持' if loc > 0.05 else '趋向均匀扭转（扇区极小离域）'
    print('      K=%-7g |Q| 全程 1=%s，终态最大内部键差 %.4f，E_end %.4e（均匀极小 1/2N=%.4e），形态=%s'
          % (K, q_conserved, loc, e_end, 0.5 / N, shape))
    if not q_conserved:
        g1_ok = False

print('   c) 开放边界对照（自由端，守卫撤除）：kink 是否漏出')
K_c = 1e-2
v_end, k_rec = grad_flow_open(v_kink, K_c, NSTEPS)
leaked = (k_rec[-1] == 0) and (k_rec[0] != 0)
print('      缺陷计数（每 2000 步）：%s -> 终值 %d（漏出=%s）'
      % (k_rec[:6], k_rec[-1], leaked))
if not leaked:
    g1_ok = False
    print('      [读数异常] 开放边界下 kink 未漏出，须查端点自由度与步数。')

if g1_ok:
    print('   [G1] PASS：Q=0 对湮灭通道开放（守卫不阻止）；|Q|=1 缺陷总数不灭')
    print('   （跨扇区被禁）；开放边界对照确认守卫来自净绕数守恒本身。')
else:
    print('   [G1] FAIL 或读数异常：先查机制归属再改数值。')
print()


# ---------- G2：量子层对接（结构核对，引既有读数） ----------
print('--- G2（量子层：超选择与 toric code 对接，结构核对）---')
print('   既有读数（不重跑）：')
print('   - Z_N toric code：绕数为基态空间投影量子数，跨扇区一阶矩阵元为零')
print('     （[[绕数承载层：电荷保护的波性侧归宿与粒子复合结构]] §3.2，全系数值）')
print('   - 弦算符 U_C = exp(i sum phi_l) 是基态空间算符：端点产生一对缺陷，')
print('     弦收缩即对湮灭（同注记 §3.1；超选择论文的弦算符论证）')
print('   对接读法：')
print('   - Q=0 对湮灭 = 弦算符收缩：基态空间内过程，不跨扇区，超选择不禁止')
print('   - |Q|=1 单缺陷湮灭：须把基态搬出扇区，跨扇区一阶矩阵元为零，被禁')
print('   - 守卫层守的是扇区标签（净绕数），不阻止净绕数为零的对湮灭')
print('   区分力三档：a 允许、b 禁止、c 撤守卫后允许，三者不同判，判据有区分力。')
g2_ok = g1_ok
print('   [%s] G2 完成（结构核对，区分力由 G1 三档承担）' % ('PASS' if g2_ok else 'FAIL'))
print()


# ---------- G3：结论面 ----------
print('--- G3（结论面）---')
if g1_ok and g2_ok:
    print('   [G3] 判定：守卫层不阻止净绕数为零的正反对湮灭（第 484 条湮灭趋势维持）。')
    print('   守卫层职能确认为守扇区标签：|Q|=1 单缺陷禁灭（跨扇区）、Q=0 对可湮灭。')
    print('   环面粒子定域化的承接者收窄为：第 485 条内部壁承接（候选成立）与')
    print('   第 483 条边界环冻结（候选成立）。与三层分工（拓扑序守电荷、格点内')
    print('   锁定给质量、弹性包层给宽度）自洽对接：守卫管电荷型守恒，不管位置型稳定。')
print()
print('运行时间 %.2f 秒' % (time.time() - T_START))
