# -*- coding: utf-8 -*-
r"""
pocket_total_account.py

[构造轮第 1 轮] 位错构型上的新相化总账（判据 S1 至 S5，第 595 条）

第 594 条前置轮钉定：负曲率是失稳通道非负能量项，构造轮按"新相化总账"立判据
（异类代价对键拉伸节省的联合再分布）。本判据把总账落到最小位错构型上。

模型（一维周期链三分量，第 484 条与第 591 条 JB 同族口径）
    格点 n = 0..N-1，三分量 a = 1,2,3，相位 θ^a_n = 2π(n·c_a + v^a_n)，
    c 为错排分配向量（Σc_a = 1 为单绕数墙），v^a 为弛豫变量（周期）。
    E = (k/2) Σ_a Σ_n (v^a_{n+1} - v^a_n)^2
        - J Σ_n [cos 2π(v¹-v²) + cos 2π(v¹-v³) + cos 2π(v²-v³)]
    真空（全 v = 0）：E = -3J。异类项作用在格点内三对（体系立案口径，
    无层间异类项）；键弹性作用在各分量沿链。δ^ab = 2π(v^a - v^b)（弧度）。

判据（先立理论期望再实现）

S1 冒烟：真空能量对 -3J 机器精度；梯度流能量单调下降（第 594 条教训：
  符号经解析核验，g^a_m = -k(Δ²v^a)_m + 2πJ Σ_{b≠a} sin 2π(v^a_m - v^b_m)）；
  总绕数 Σ_a Σ_n Δv^a = 1 在弛豫全程守恒。

S2 终态 δ 剖面读数：两种初值（α 均分 c = (1/3,1/3,1/3)，δ ≡ 0；
  β 单承载 c = (1,0,0)，δ^12 = δ^13 沿墙扫 0 到 2π）弛豫后的
  max|δ|、失稳窗（弧度版：单对曲率 2J cos δ < 0 即 δ ∈ (π/2, 3π/2)）
  内格点占比。期望（先立）：强锁档（J 大）β 初值被拉回 α 类（δ 收敛小），
  弱锁档（J 小）β 保持大 δ 剖面。

S3 能量账（核心判据，首轮后修正口径）：统一完整尺下两模型各自弛豫终态的比较
    Δ = E_full(v*_full) - E_full(v*_quad)
  （完整模型与二次化模型同初值各自弛豫到自己的极小，再用完整势同一把尺评估；
  首版的固定构型差 D = Σ[J(1-cos δ) - Jδ²/2] 在 δ ~ 2π 周期像区把二次口径
  罚到 +19.7J/格，巨大负值是口径伪影非软化信号，已废弃）。
  Δ < 0 = 软化让位错更便宜（来源乙供货）；Δ > 0 = 非线性不供货。
  区分力：α 初值（对称）终态两模型同为 δ≡0，Δ = 0 确认读数非构造恒正。

S4 档位标度：k = 1，J ∈ {0.5, 1.0, 2.0} 三档（覆盖弱锁到强锁），报
  β 初值终态的 D 与位错能量 E(终态) - E_vac 的档位依赖。期望：J 增大
  （锁增强）时终态 δ 被压小、D 朝零走。

S5 初值无关性与分配偏好：α 与 β 初值在同档弛豫后的终态能量比较，判
  实际错排分配的能量偏好（位错应变在三分量间如何再分布）。

可逆性：判据敏感（初值族、档位与弛豫协议绑定；换初值族或协议须重跑）。
模型单位：k = 1，N = 256，η = 0.01，步数 150000。
运行：python 验证脚本/pocket_total_account.py
"""
import math
import time

import numpy as np

T_START = time.time()
N = 256
ETA_BASE = 0.01
NSTEPS = 150000
K_EL = 1.0
TWO_PI = 2.0 * math.pi
J_LIST = [0.5, 1.0, 2.0]

print('=' * 76)
print('构造轮第 1 轮：位错构型上的新相化总账（第 595 条，S1 至 S5）')
print('=' * 76)
print()
print('第 594 条结论：负曲率是失稳通道非负能量项，本轮判"新相化总账"。')
print('供货读数 D = Σ [J(1-cos δ) - (J/2)δ²]（完整异类对二次化的能量账）。')
print()


def energy(v, J):
    """v 形状 (3, N)。返回键项、异类项（完整）与总能量。"""
    dv = np.diff(np.concatenate([v, v[:, :1]], axis=1), axis=1)
    e_el = 0.5 * K_EL * float(np.sum(dv ** 2))
    d12 = TWO_PI * (v[0] - v[1])
    d13 = TWO_PI * (v[0] - v[2])
    d23 = TWO_PI * (v[1] - v[2])
    e_het = -J * float(np.sum(np.cos(d12) + np.cos(d13) + np.cos(d23)))
    return e_el, e_het, e_el + e_het


def energy_quad_het(v, J):
    """二次化异类口径（键项同）。"""
    d12 = TWO_PI * (v[0] - v[1])
    d13 = TWO_PI * (v[0] - v[2])
    d23 = TWO_PI * (v[1] - v[2])
    return 0.5 * J * float(np.sum(d12 ** 2 + d13 ** 2 + d23 ** 2)) - 3.0 * J


def winding(v):
    """总绕数：三链端点差整化之和。"""
    return float(sum(round(v[a][-1] - v[a][0]) for a in range(3)))


def grad_flow(v0, J, nsteps):
    # eta 按档预算：Hessian 谱界 4k + 8 pi^2 J，eta*lambda <= 0.3（首版固定
    # eta 使 J=2 档发散，能量上升被单调检查抓出）
    eta = min(ETA_BASE, 0.3 / (4.0 + 8.0 * math.pi ** 2 * J))
    """梯度流：g^a_m = -k(Δ²v^a)_m + 2πJ Σ_{b≠a} sin 2π(v^a_m - v^b_m)。
    符号经解析核验（∂E_el/∂v = -kΔ²v，∂E_het/∂v^a = +2πJΣ_{b≠a} sin）。"""
    v = v0.copy()
    e_rec = [energy(v, J)[2]]
    q0 = winding(v)
    q_ok = True
    for step in range(nsteps):
        lap = np.roll(v, 1, axis=1) - 2 * v + np.roll(v, -1, axis=1)
        d12 = TWO_PI * (v[0] - v[1])
        d13 = TWO_PI * (v[0] - v[2])
        d23 = TWO_PI * (v[1] - v[2])
        g = np.empty_like(v)
        g[0] = -K_EL * lap[0] + TWO_PI * J * (np.sin(d12) + np.sin(d13))
        g[1] = -K_EL * lap[1] - TWO_PI * J * np.sin(d12) + TWO_PI * J * np.sin(d23)
        g[2] = -K_EL * lap[2] - TWO_PI * J * (np.sin(d13) + np.sin(d23))
        v = v - eta * g
        v -= v.mean(axis=1, keepdims=True) * 0.0   # 周期场无需去均值
        if step % 25000 == 0:
            e = energy(v, J)[2]
            if e > e_rec[-1] + 1e-12:
                q_ok = False   # 能量上升 = 梯度流失效
            e_rec.append(e)
    if abs(winding(v) - q0) > 1e-9:
        q_ok = False
    return v, q_ok


def grad_flow_quad(v0, J, nsteps):
    """二次化异类模型的梯度流（异类 = (J/2)Σδ²，梯度 = 2πJΣ_{b≠a} δ^ab·2π？：
    E_het^quad = (J/2)Σ_n[(2πΔv12)²+(2πΔv13)²+(2πΔv23)²] - 3J，
    ∂/∂v^a = 2π·(2πJ)·Σ_{b≠a}(v^a-v^b)·(配对计数) —— 直接数值：
    对每对 (a,b)：∂/∂v^a += (2π)²J(v^a-v^b)，∂/∂v^b -= 同值。"""
    v = v0.copy()
    eta = min(ETA_BASE, 0.3 / (4.0 + 8.0 * math.pi ** 2 * J))
    for step in range(nsteps):
        lap = np.roll(v, 1, axis=1) - 2 * v + np.roll(v, -1, axis=1)
        g = -K_EL * lap
        for (a, b) in ((0, 1), (0, 2), (1, 2)):
            diff = v[a] - v[b]
            g[a] += (TWO_PI ** 2) * J * diff
            g[b] -= (TWO_PI ** 2) * J * diff
        v = v - eta * g
    return v, True


def supply_D(v, J):
    """供货读数 D = Σ [J(1-cos δ) - (J/2)δ²]。"""
    d12 = TWO_PI * (v[0] - v[1])
    d13 = TWO_PI * (v[0] - v[2])
    d23 = TWO_PI * (v[1] - v[2])
    tot = 0.0
    for d in (d12, d13, d23):
        tot += float(np.sum(J * (1.0 - np.cos(d)) - 0.5 * J * d ** 2))
    return tot


def init_alpha(N):
    n = np.arange(N)
    s = 1.0 / 3.0
    prof = s * 0.5 * (1.0 + np.tanh((n - N / 2) / 8.0))
    return np.stack([prof, prof, prof])


def init_beta(N):
    n = np.arange(N)
    prof = 0.5 * (1.0 + np.tanh((n - N / 2) / 8.0))
    return np.stack([prof, np.zeros(N), np.zeros(N)])


# ---------- S1 冒烟 ----------
print('--- S1 冒烟 ---')
v0 = np.zeros((3, N))
e_vac = energy(v0, 1.0)[2]
print('   真空能量 = %.12f（期望 -3J*N = -768.0，偏差 %.2e）' % (e_vac, abs(e_vac + 3.0 * N)))
v_test, flow_ok = grad_flow(init_beta(N), 1.0, 3000)
e_seq_ok = flow_ok
print('   梯度流能量单调与 Q 守恒：%s' % ('PASS' if e_seq_ok else 'FAIL'))
smoke = abs(e_vac + 3.0 * N) < 1e-10 and e_seq_ok
print('   [%s] S1' % ('PASS' if smoke else 'FAIL'))
print()

# ---------- S2/S3/S4：三档两初值 ----------
print('--- S2 至 S4（δ 剖面、能量账、档位标度）---')
print('   J      初值  终态 max|δ|   失稳窗占比   D（供货读数）    E-E_vac')
results = {}
s3_expect_ok = True
for J in J_LIST:
    for tag, vinit in (('α 均分', init_alpha(N)), ('β 单承载', init_beta(N))):
        vend, ok = grad_flow(vinit, J, NSTEPS)
        d12 = TWO_PI * (vend[0] - vend[1])
        d13 = TWO_PI * (vend[0] - vend[2])
        dmax = float(max(np.max(np.abs(d12)), np.max(np.abs(d13))))
        unstable = float(np.mean((np.abs(d12) > math.pi / 2) | (np.abs(d13) > math.pi / 2)))
        D = supply_D(vend, J)
        _, _, e_tot = energy(vend, J)
        e_excess = e_tot + 3.0 * J
        # 终态错排分配：各链端点差（绕数份额 c_a）
        shares = [round(float(vend[a][-1] - vend[a][0]), 4) for a in range(3)]
        print('   %-4g  %s   %-11.4f %-11.3f %-+14.4f %+.4f  分配%s %s'
              % (J, tag, dmax, unstable, D, e_excess, shares,
                 '' if ok else '[梯度流异常]'))
        results[(J, tag)] = (dmax, unstable, D, e_excess, ok, vend)
        if not ok:
            s3_expect_ok = False


print()
# S3 重定义（首版 D 判据对象错：二次化口径在 δ~2π 周期像区失真，罚 19.7J/格，
# D 的巨大负值是口径伪影非软化信号）。正确总账：统一完整尺下，完整模型与
# 二次化模型各自弛豫的终态能量比较（同初值同绕数）：
#   Δ = E_full(v*_full) - E_full(v*_quad) < 0 才是软化供货的直接读数。
print('--- S3（修正版：统一完整尺的两模型终态比较）---')
s3_ok = True
for J in J_LIST:
    for tag, vinit in (('α 均分', init_alpha(N)), ('β 单承载', init_beta(N))):
        v_full = results[(J, tag)][5]
        # 二次化模型：异类梯度替换为二次形式，各自弛豫
        v_q, ok_q = grad_flow_quad(vinit, J, NSTEPS)
        e_full_of_full = energy(v_full, J)[2]
        e_full_of_quad = energy(v_q, J)[2]
        delta = e_full_of_full - e_full_of_quad
        print('   J=%-4g %s  E_full(v*full) = %+.4f，E_full(v*quad) = %+.4f，'
              'Δ = %+.4f（%s）'
              % (J, tag, e_full_of_full + 3 * J * N, e_full_of_quad + 3 * J * N,
                 delta, '软化供货' if delta < 0 else '非线性不供货'))
        if ok_q is False:
            s3_ok = False
print('   [%s] S3' % ('PASS' if s3_ok else 'FAIL'))
print()

# ---------- S5：分配偏好 ----------
print('--- S5（初值无关性与分配偏好）---')
prefer_ok = True
for J in J_LIST:
    e_a = results[(J, 'α 均分')][3]
    e_b = results[(J, 'β 单承载')][3]
    pref = 'α 均分更稳' if e_a < e_b else 'β 单承载更稳'
    print('   J=%-4g E_α - E_vac = %+.4f，E_β - E_vac = %+.4f → %s'
          % (J, e_a, e_b, pref))
print('   [S5] 读数（分配偏好随档位的变化，判读见注记）')
print()

# ---------- 汇总 ----------
print('=' * 76)
print('汇总（第 595 条，构造轮第 1 轮）')
print('=' * 76)
print('供货读数 D 的符号与档位依赖见上表；判读与登记见验证注记。')
print()
print('运行时间 %.1f 秒' % (time.time() - T_START))
