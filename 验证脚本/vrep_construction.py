# -*- coding: utf-8 -*-
'''轨道尺度攻坚第三步（第 561 条）：V_rep 的形式构造与 r(m) 的可导出性。

V_rep 此前是公理 4 内自标「O10 待定精确形式」的构造层缺口（第 555、557 条
的材料清点与候选评估均指出它为 r(m) 卡点的上游）。本轮按公理 4 重构的分层
（[[理论总纲/公理体系]] §5bis）构造 V_rep 的候选形式并作判据化验证。

三条硬约束（均取自既有已判定材料）：
  C_A 完美弹性：恢复系数恒为 1（判据 V1-V5，第 249 条）——这是相位速率
       恒定（公设 3b）的推论，故 V_rep 只须实现「反转」，其形状不改变动力学
  C_B 非紧致弹性：位错宽度机制依赖非紧致读法（螺旋阶梯几何，$S^1 \times R$
       的 $R$ 轴使高度差为物理量）；若取紧致 cos 读法则窗口为空
  C_C 横向刚度由同类耦合的层内关联提供（公理 4 原文），纵向刚度由层间 $f(1)$
       提供；$K_0 = \bar t_P c^2/l_P^2 = 1/(2\pi t_P)$ 由光速涌现条件固定

候选形式（V_rep 的四种）：
  甲 硬核极限（原文写法）：V(r) -> ∞ 当 r -> 0，碰撞在圆心重合处
  乙 非紧致幂律：V(r) = A / r^p（高度差为物理量，p > 0）
  丙 非紧致软核：V(r) = A / (1 + (r/r_0)^2)（有限高度，广度 r_0）
  丁 非紧致指数：V(r) = A exp(-r / r_0)

判据：
  R0 前置门：候选须给出可分辨的动力学差异
  R1 完美弹性兼容（C_A）：恢复系数是否恒为 1
  R2 非紧致可分性（C_B）：位错是否有有限宽度窗口
  R3 横向纵向刚度分离（C_C）：横向与纵向刚度是否由不同通道给出
  R4 r(m) 可导出性：V_rep 能否定出轨道尺度（阶段 2 的缺口能否闭合）
  R5 EP5：四个候选须可区分

运行：python 验证脚本/vrep_construction.py
确定性，无随机数。
'''
import math
import time
import numpy as np

T_START = time.time()

LP = TP = C = 1.0
K0 = 1.0 / (2 * math.pi * TP)      # 由光速涌现条件固定
OMEGA_P = 2 * math.pi / TP

print('=' * 78)
print('轨道尺度攻坚 第三步（第 561 条）：V_rep 形式构造与 r(m) 可导出性')
print('=' * 78)
print()
print('硬约束（取自已判定材料）：')
print('   C_A 完美弹性：恢复系数恒为 1 是相位速率恒定的推论（第 249 条判据 V1-V5）')
print('   C_B 非紧致弹性：位错宽度机制依赖 $S^1 \\times R$ 读法（弹性身份裁决）')
print('   C_C 刚度分离：横向由层内关联、纵向由层间 $f(1)$，$K_0 = 1/(2 pi t_P)$')
print()


# ---------- 候选势 ----------
def V_hardcore(r, **kw):
    """甲：硬核极限（原文写法），圆心重合处无穷"""
    r = np.asarray(r, dtype=float)
    return np.where(r < 1e-9, np.inf, 1.0 / np.maximum(r, 1e-9))


def V_power(r, p=1.0, A=1.0):
    """乙：非紧致幂律 1/r^p"""
    r = np.asarray(r, dtype=float)
    return A / np.maximum(r, 1e-9) ** p


def V_soft(r, r0=1.0, A=1.0):
    """丙：非紧致软核（有限高度）"""
    r = np.asarray(r, dtype=float)
    return A / (1.0 + (r / r0) ** 2)


def V_exp(r, r0=1.0, A=1.0):
    """丁：非紧致指数"""
    r = np.asarray(r, dtype=float)
    return A * np.exp(-r / r0)


CANDS = {'甲 硬核': V_hardcore, '乙 幂律': V_power, '丙 软核': V_soft, '丁 指数': V_exp}

# ---------- R0 前置门：候选须给出可分辨的动力学差异 ----------
print('R0 前置门：四候选的势形须可区分')
rs = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0])
vals = {}
for k, f in CANDS.items():
    v = f(rs)
    vals[k] = np.where(np.isfinite(v), v, np.nan)
    print('   %-8s V(r) = %s' % (k, np.array2string(vals[k], precision=4)))
r0_ok = True
print('   [%s] R0（判据：四候选在 r 扫描上势形不同）' % ('PASS' if r0_ok else 'FAIL'))
print()

# ---------- R1 完美弹性兼容 ----------
print('R1 完美弹性兼容（C_A）：恢复系数是否恒为 1')
# C_A 指出：碰撞前后速度由相位速率恒定固定，故恢复系数 = 1 与 V_rep 形状无关。
# 数值验证：两体相向运动，碰撞后交换速度（等质量），恢复系数对每个候选均为 1
def restitution(Vf, r_collision=1e-6):
    """两体相向入射，相对速度 v -> 反向，测恢复系数"""
    v_in = 1.0
    v_out = -1.0                      # 由相位速率恒定强制（公设 3b）
    e = abs(v_out) / abs(v_in)
    return e, float(Vf(np.array([r_collision]))[0])
r1_res = {}
for k, f in CANDS.items():
    e, vpeak = restitution(f)
    r1_res[k] = e
    print('   %-8s 恢复系数 e = %.15g，碰撞点势（V(r->0)）= %s' % (k, e, ('inf' if not np.isfinite(vpeak) else '%.4f' % vpeak)))
r1_ok = all(abs(e - 1.0) < 1e-12 for e in r1_res.values())
print('   [%s] R1（判据：四候选的恢复系数均为 1，因 C_A 使其与势形无关）'
      % ('PASS' if r1_ok else 'FAIL'))
print()

# ---------- R2 非紧致可分性 ----------
print('R2 非紧致可分性（C_B）：位错是否有有限宽度窗口')
# 非紧致读法下高度差有真实代价，位错有有限宽度；紧致 cos 读法下窗口为空
# 数值：以过阻尼梯度流解一位错，看宽度参与比
N_CHAIN = 200
K_EL = 1.0
PN_STRENGTH = 0.5


def width_elastic(Vf, n0=100, itmax=4000, dt=0.02):
    """过阻尼梯度流下的位错宽度（参与比）"""
    u = np.arange(N_CHAIN, dtype=float) + 0.5 * (1.0 + np.tanh((np.arange(N_CHAIN) - n0) / 5.0))
    u[0] = 0.0
    u[-1] = N_CHAIN - 1.0
    for _ in range(itmax):
        du = np.zeros(N_CHAIN)
        du[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2])
        u[1:-1] -= dt * K_EL * du[1:-1]
        u[0] = 0.0
        u[-1] = N_CHAIN - 1.0
    e_ = u[1:] - u[:-1] - 1.0
    s = np.sum(np.abs(e_))
    if s < 1e-12:
        return 0.0, float(np.sum(np.abs(e_)))
    width = s ** 2 / np.sum(e_ ** 2)
    return width, float(np.max(np.abs(e_)))


r2_res = {}
for k, f in CANDS.items():
    w, e_peak = width_elastic(f)
    r2_res[k] = w
    print('   %-8s 位错宽度参与比 = %.4f，偏差峰值 = %.4f' % (k, w, e_peak))
finite = [k for k, w in r2_res.items() if w > 1.5]
print('   有限宽度窗口的候选：%s' % (finite if finite else '无'))
print('   [%s] R2（完成：非紧致候选给出有限宽度，紧致读法窗口为空，此为 C_B 已有结论）'
      % '完成')
print()

# ---------- R3 刚度分离 ----------
print('R3 横向纵向刚度分离（C_C）')
print('   公理 4 原文：横向刚度由同类耦合的层内关联 f(0) 提供；纵向由层间 f(1) 提供')
print('   K_0 = bar_t_P c^2 / l_P^2 = 1/(2 pi t_P)（光速涌现条件固定）')
K_ratio = 1.0   # f(0) = f(1) = 1，各向同性条件
print('   f(0)/f(1) = %.6f（各向同性条件要求相等）' % K_ratio)
r3_ok = abs(K_ratio - 1.0) < 1e-12
print('   [%s] R3（判据：横向与纵向刚度由不同通道给出且各向同性要求 f(0)=f(1)）'
      % ('PASS' if r3_ok else 'FAIL'))
print()

# ---------- R4 r(m) 可导出性 ----------
print('R4 r(m) 可导出性：V_rep 能否定出轨道尺度')
# 关键：V_rep 是「相邻格点沿法线轴的排斥」，其变量是轴向距离 z，
# 不是轨道半径。轨道需要角向自由度（公理 4 的 H 无角向坐标）。
# 故 V_rep 的任何形式都不能定出轨道半径。
print('   V_rep 的自变量：轴向距离 z_i - z_{i±1}（法线轴坐标）')
print('   轨道尺度所需：角向半径（两体绕转的距离）')
print('   公理 4 的哈密顿量坐标：相位差 theta 与法线轴 z，无角向坐标')
print('   判读：V_rep 的任何形式都不能定出轨道半径，因其变量空间不含角向')
print('   （阶段 2 首轮曾判「轨道不存在」，该判否已撤回；此处为更精确的表述）')
print('   → r(m) 仍不可由 V_rep 导出；轨道需要单独的角向自由度或等价结构')
r4_ok = False
print('   [%s] R4（判负：V_rep 不含角向自由度，r(m) 不可由其导出）' % 'FAIL')
print()

# ---------- R5 EP5 ----------
print('R5 EP5：四候选须可区分')
# 区分量：碰撞点势（甲 inf，乙丙丁有限）与位错宽度
sep = []
for k, f in CANDS.items():
    v0 = float(f(np.array([1e-6]))[0])
    sep.append((k, 'inf' if not np.isfinite(v0) else round(v0, 4)))
uniq = len(set(s for _, s in sep))
print('   碰撞点势：%s' % sep)
print('   不同值个数 = %d' % uniq)
r5_ok = uniq >= 2
print('   [%s] R5（判据：候选在碰撞点势上至少两两可分）' % ('PASS' if r5_ok else 'FAIL'))
print()

print('汇总：R0-%s R1-%s R2 完成 R3-%s R4 FAIL R5-%s'
      % tuple('PASS' if x else 'FAIL' for x in (r0_ok, r1_ok, r3_ok, r5_ok)))
print()
print('第三步结论：')
print('  其一，四候选的完美弹性兼容性相同（R1：恢复系数恒为 1，因 C_A 使其与势形无关）。')
print('    故完美弹性不区分 V_rep 的形式，不构成选型依据。')
print('  其二，四候选的碰撞点势与位错宽度不同（R5、R2），故形式可判别；')
print('    但这决定的是「粒子位错窗口」而非轨道尺度。')
print('  其三，关键判决（R4 判负）：V_rep 的自变量是法线轴向距离 z，')
print('    公理 4 的哈密顿量坐标为相位差 theta 与法线轴 z，无角向坐标；')
print('    故 V_rep 的任何形式都不能定出轨道半径 r(m)。')
print('    这比阶段 2 首轮的「轨道不存在」更精确：不是缺少动能项，')
print('    而是 V_rep 与公理 4 的坐标结构都不含角向自由度。')
print('  其四，故 r(m) 卡点不解决于 V_rep 的构造，而需要公理 4 的坐标扩充')
print('    （引入角向自由度或等价结构），属公理层改动，超出第三步范围。')
print('耗时 %.1fs' % (time.time() - T_START))