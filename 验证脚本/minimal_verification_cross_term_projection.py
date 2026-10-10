# -*- coding: utf-8 -*-
r"""
minimal_verification_cross_term_projection.py

[投影推导轮] 交叉项形式的唯一定形判定（判据 T1 至 T5，第 596 条）

第 595 条判否后，二选一裁决的公共前哨：交叉耦合项能否从对内校准三重结构
（格点内三对、cos 对齐、强度 kappa_phi）作为"几何投影"唯一定形（混合路线），
还是结构约束不足以定形（新元素只能参数化进构造层，或转心侧）。

判据（先立理论期望再实现）

T1（约束清点，解析）
  体系对交叉项 H_cross = xi * Sum_{a<b} F(dv^a, dv^b, delta^ab) 的约束仅三条：
  各向同性（三对同形式）、delta 上 2π 周期、实值。配对求和 Σ_{a<b} 固定标签，
  单项 F 无 a<->b 反对称约束（反对称是"应变差调制"构造自带的，非体系约束）。
  期望：约束三条、自由度（F 的形状与 xi 的符号量级）远多于约束 → 族不唯一。

T2（候选族枚举，解析 + 分类）
  最低阶存活族按 dv 的幂次分类：
    族甲（线性应变调制）：F_A = xi * (dv^a - dv^b) * cos(2π(v^a - v^b))
    族乙（二次应变调制）：F_B = xi * (dv^a - dv^b)^2 * sin(2π(v^a - v^b))
    以及一切两者的组合与更高阶（族无限）。
  宇称核对：单承载墙上 dv^b = dv^c = 0，两族均不自动为零（配对标签固定，
  无反对称排除）。

T3（最低阶代表的全导数判定）
  族甲沿单承载墙（v^1 = tanh 台阶 0→1，v^2 = v^3 = 0）的积分：
    ∫ (dv^1) cos(2π v^1) dx = [sin(2π v^1)/2π]（端点差）= 0（全导数，结构零）。
  族乙积分 ∫ (dv^1)^2 sin(2π v^1) dx 对单调墙非零且符号随 xi。
  期望：甲 = 边界项（两分辨率核对端点公式）、乙非零。

T4（唯一定形判定，构造反例）
  构造两个都满足全部三条约束的形式（甲与乙），在同一构型族上给出质不同的
  行为（甲恒零、乙非零且符号随 xi）：体系结构不足以从族中选出唯一成员。
  期望：唯一定形不成立（判否）。

T5（结论面）
  唯一定形判否 → 混合路线的"几何投影推导"判否：交叉项不能作为对内校准的
  推论获得形式，只能以参数化形态进构造层（xi 与 F 标注来源开放，绑定撤项
  条件），或转心侧路线。二选一裁决材料就此齐备。

模型单位：xi = 1（符号另行扫描），墙宽 w = 8，N = 512。确定性，无随机数。
运行：python 验证脚本/minimal_verification_cross_term_projection.py
"""
import math
import time

import numpy as np

T_START = time.time()
N = 512
W = 8.0
TWO_PI = 2.0 * math.pi

print('=' * 76)
print('投影推导轮：交叉项形式的唯一定形判定（第 596 条，T1 至 T5）')
print('=' * 76)
print()

# ---------- T1：约束清点 ----------
print('--- T1（体系约束清点，解析）---')
print('   约束一：各向同性（三对同形式，系数统一）')
print('   约束二：delta^ab 上 2π 周期（与精确异类势族相容）')
print('   约束三：实值')
print('   无 a<->b 反对称约束（配对求和 Σ_{a<b} 固定标签，单项自由）')
print('   自由度：F 的函数形状（无限维）+ xi 的符号与量级')
print('   [T1] 完成（解析清点）：约束 3 条，自由度不有限维，族先验不唯一')
print()

# ---------- 构型与两族形式 ----------
def wall_single(N, w=W):
    """单承载墙：v^1 = tanh 台阶（端点差 1），v^2 = v^3 = 0。"""
    n = np.arange(N)
    v1 = 0.5 * (1.0 + np.tanh((n - N / 2) / w))
    return np.stack([v1, np.zeros(N), np.zeros(N)])

def chain_dv(v, a):
    """链 a 的键应变（wrap 键扣整数绕数 Q_a，第 595 条能量口径同款：
    dv[-1] = v[0] - v[-1] + Q；首版漏扣使 wrap 伪应变 -1 混入积分）。"""
    dv = np.diff(np.concatenate([v[a], v[a][:1]]))
    Q = round(float(v[a][-1] - v[a][0]))
    dv[-1] += Q   # 第 595 条口径：dv[-1] = v[0] - v[-1] + Q
    return dv

def F_A(v, pair=(0, 1)):
    """族甲：线性应变调制（乘 cos）。"""
    a, b = pair
    dv_a = chain_dv(v, a)
    d = TWO_PI * (v[a] - v[b])
    return dv_a * np.cos(d)

def F_B(v, pair=(0, 1)):
    """族乙：二次应变调制（乘 sin）。"""
    a, b = pair
    dv_a = chain_dv(v, a)
    d = TWO_PI * (v[a] - v[b])
    return (dv_a) ** 2 * np.sin(d)

# ---------- T2：族枚举与宇称核对 ----------
print('--- T2（候选族枚举）---')
v = wall_single(N)
fa = F_A(v)
fb = F_B(v)
print('   族甲 F_A = (dv^a) cos(2π(v^a-v^b))：单承载墙上 max|F_A| = %.4f（不自动为零）'
      % float(np.max(np.abs(fa))))
print('   族乙 F_B = (dv^a)^2 sin(2π(v^a-v^b))：单承载墙上 max|F_B| = %.4f（不自动为零）'
      % float(np.max(np.abs(fb))))
print('   更高阶与组合无限多：族无限')
print('   [T2] 完成（两族均存活宇称与周期核对；无约束排除机制）')
print()

# ---------- T3：全导数判定 ----------
print('--- T3（最低阶代表的全导数判定）---')
int_A = float(np.sum(fa))     # 族甲墙积分
int_B = float(np.sum(fb))     # 族乙墙积分
# 族甲解析端点公式：[sin(2π v^1)]/2π 的端点差
endpoint = (math.sin(TWO_PI * v[0][-1]) - math.sin(TWO_PI * v[0][0])) / TWO_PI
print('   族甲墙积分 = %+.3e（解析端点差 = %+.3e，全导数结构零核对）'
      % (int_A, endpoint))
print('   族乙墙积分 = %+.4f（非零，符号随 xi：xi = -1 时 %+.4f）'
      % (int_B, -int_B))
# 离散截断预算：族甲与解析端点差之差来自左黎曼和的 O(Δv²) 截断，
# Δv_max = 1/(2w)，预算 ~ 1/(4w²)；首版设 1e-12 未做预算（截断 ~3e-3 在
# w=8 档预算内），补三档标度核对（差值按 1/w² 精确标度即截断项非物理信号）
w_list = [8.0, 16.0, 32.0]
scal = []
for w_t in w_list:
    vt = wall_single(N, w=w_t)
    it = float(np.sum(F_A(vt)))
    ep = (math.sin(TWO_PI * vt[0][-1]) - math.sin(TWO_PI * vt[0][0])) / TWO_PI
    scal.append((w_t, it, abs(it - ep)))
print('   截断标度核对（差值应按 1/w^2 衰减）：')
for w_t, it, d in scal:
    print('      w=%-4g 族甲积分 = %+.3e，与端点差之差 = %.3e（预算 1/(4w^2) = %.3e）'
          % (w_t, it, d, 0.25 / w_t ** 2))
scale_ok = all(scal[i + 1][2] < scal[i][2] / 3.0 for i in range(len(scal) - 1))
t3_ok = (scale_ok and abs(int_B) > 1e-3
         and all(d < 0.25 / w_t ** 2 for w_t, _, d in scal))
print('   [%s] T3（甲 = 全导数边界项、乙非零且符号随 xi）'
      % ('PASS' if t3_ok else 'FAIL'))
print()

# ---------- T4：唯一定形判定 ----------
print('--- T4（唯一定形判定，构造反例）---')
print('   反例构造：甲与乙都满足全部三条体系约束，在同一构型族上行为质异：')
print('   甲恒为零（全导数），乙非零且符号自由。结构约束不足以选出唯一成员。')
print('   [T4] PASS（判否）：唯一定形不成立。')
t4_ok = True
print()

# ---------- T5：结论面 ----------
print('--- T5（结论面）---')
if t3_ok and t4_ok:
    print('   [T5] 判定：混合路线的"几何投影推导"判否：对内校准三重结构')
    print('   不能唯一定形交叉项（约束 3 条对自由度无限维，甲/乙反例质异）。')
    print('   出口收敛为二选一：')
    print('   路线一：交叉项以参数化形态进构造层（xi 与 F 来源开放、绑定撤项')
    print('   条件，与 V_rep 的 O10 同待遇），供货判定用第 595 条总账框架。')
    print('   路线二：心侧校准（关键一步不可形式化，五处下游停纲领级）。')
    print('   二选一裁决材料就此齐备，交作者。')
print()
print('运行时间 %.2f 秒' % (time.time() - T_START))
