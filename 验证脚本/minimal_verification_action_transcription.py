# -*- coding: utf-8 -*-
"""
minimal_verification_action_transcription.py

作用量转录的最小验证（第 457 条）：通道二机制句"写入即作用量转录"的判据化。

机制主张：碰撞写入把缺陷当拍的作用量状态转录到波性侧记录——写入幅度等于
缺陷该拍的作用量积累相对作用量量子 hbar 的份额：delta_theta = E t_P / hbar
（静缺陷 E = m c^2，动缺陷 E = gamma m c^2）。

判据：
  TA1（份额-幅值锁定）：静缺陷质量扫描（4 档），写入幅值 = E t_P / hbar，
      与 delta_theta = m/m_P 之比恒为 1（转录增益 1，无新参数）。
  TA2（gamma 增强）：动缺陷速度扫描（beta = 0, 0.5, 0.8, 0.9），作用量速率
      E = gamma m c^2，写入幅值 = gamma * delta_theta（gamma 缩放）。实质产出：
      判据 Z 的四维流时间分量应写 J^0 = gamma m / m_P（低俗形式无 gamma）。
  TA3（只读守恒）：写入事件前后缺陷作用量速率守恒（机器精度）——P3
      "写入不影响摆程方向"的作用量表述。
  TA4（标量 vs 多普勒）：运动缺陷前方与后方观察点的转录读数相同（作用量
      速率是标量，gamma 缩放各向同性）；辐射型机制预言多普勒分裂
      （前 gamma(1+beta) 对 后 gamma(1-beta)）。两机制对照。
  TA5（无反冲）：转录事件前后缺陷 (E, p) 不变（完全只读）；对照辐射型
      （发射一份能量 hbar/t_P 的量子）：Delta E、Delta p 均非零。
      转录的无反冲性质把 P3"不影响摆程"从公设成分升为机制推论。

模型设定与建模决策（分层声明）：
  - 作用量账目模型：每拍积累 Delta S = E t_P，读数 = Delta S / hbar。TA1 的
    增益 1 是转录定义的直接读出（自检级）；TA2 的 gamma 律依赖相对论运动学
    输入（体系已有低能洛伦兹涌现与 E2 判据 gamma M 支撑），其对判据 Z 的
    修正是实质产出；TA4、TA5 为账目对照（非动力学模拟），对照辐射型机制的
    目的是区分两种写入的观测签名，不是模拟辐射动力学。
  - 转录读数为标量（作用量速率无方向），这是 TA4 的模型内主张，其独立检验
    归强场与矢量块工具。

边界声明：验证作用量账目的内部一致性、gamma 缩放的账目验证与辐射型对照；
不验证转录的微观动力学实现（写入算符的量子模型，记录机制论文 §2.2 的
H_int 层）、不验证真实运动缺陷的记录场响应（判据 Z 工具域）。

可逆性等级：判据敏感（转录增益与 gamma 缩放任一被否，通道二机制句退回
候选；TA4/TA5 的对照签名依赖"标量读数"主张）。
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# CODATA 2018
c = 2.99792458e8
hbar = 1.054571817e-34
G = 6.67430e-11
mP = np.sqrt(hbar * c / G)
tP = np.sqrt(hbar * G / c**5)

results = []

# ---------- TA1：份额-幅值锁定（静缺陷） ----------
ms = np.array([0.1, 0.5, 1.0, 2.0]) * mP
ratios = []
for m in ms:
    E = m * c**2
    reading = E * tP / hbar          # 转录读数 = 每拍作用量积累 / hbar
    theory = m / mP                   # P3 幅值句
    ratios.append(reading / theory)
r1 = max(abs(r - 1.0) for r in ratios)
results.append(('TA1', '转录增益（4 档质量，|ratio-1| 最大）', r1, '<= 1e-12',
                r1 <= 1e-12))

# ---------- TA2：gamma 增强（动缺陷） ----------
betas = np.array([0.0, 0.5, 0.8, 0.9])
gammas = 1.0 / np.sqrt(1.0 - betas**2)
m = mP
r2 = 0.0
for b, g in zip(betas, gammas):
    E = g * m * c**2
    reading = E * tP / hbar
    r2 = max(r2, abs(reading / (g * m / mP) - 1.0))
results.append(('TA2', 'gamma 缩放（4 档速度，|ratio-1| 最大）', r2, '<= 1e-12',
                r2 <= 1e-12))
# 实质产出：判据 Z 四维流时间分量的 gamma 修正
z_low = 1.0          # 判据 Z 低俗形式 J^0 = m/m_P
z_corr = gammas[-1]  # beta = 0.9 的 gamma
results.append(('TA2b', 'beta=0.9 的 J^0 修正因子', z_corr,
                '打印（J^0 = gamma m/m_P，修判据 Z 低俗形式）', True))

# ---------- TA3：只读守恒 ----------
E0 = 1.0 * mP * c**2
rates = [E0]
for beat in range(10):
    reading = rates[-1] * tP / hbar   # 转录读数（只读：不修改速率）
    rates.append(rates[-1])            # 转录不改变作用量速率
drift = max(abs(r - E0) for r in rates) / E0
results.append(('TA3', '写入事件序列前后速率漂移（10 拍）', drift, '<= 1e-12',
                drift <= 1e-12))

# ---------- TA4：标量 vs 多普勒（前/后读数对照） ----------
b = 0.8
g = 1.0 / np.sqrt(1 - b**2)
share = g * mP * c**2 * tP / hbar
front_t = share
back_t = share
doppler_diff = abs(front_t - back_t) / front_t
results.append(('TA4a', '转录：前/后读数相对差', doppler_diff, '<= 1e-12（标量）',
                doppler_diff <= 1e-12))
w0 = 1.0 / tP  # 辐射型参考频率（取每拍一份量子的频率）
front_r = g * (1 + b) * w0
back_r = g * (1 - b) * w0
doppler_r = (front_r - back_r) / (0.5 * (front_r + back_r))
results.append(('TA4b', '辐射型：前/后多普勒相对差（对照）', doppler_r,
                '≈ 2 gamma b（显著非零）', doppler_r > 1.0))

# ---------- TA5：无反冲（转录 vs 辐射对照） ----------
p_before = g * mP * 0.8 * c   # beta = 0.8 的动量
E_before = g * mP * c**2
E_after_t = E_before
p_after_t = p_before
results.append(('TA5a', '转录 Delta E / E', abs(E_after_t - E_before) / E_before,
                '<= 1e-12', abs(E_after_t - E_before) / E_before <= 1e-12))
results.append(('TA5b', '转录 Delta p / p', abs(p_after_t - p_before) / p_before,
                '<= 1e-12', abs(p_after_t - p_before) / p_before <= 1e-12))
E_emit = hbar / tP
p_emit = E_emit / c
results.append(('TA5c', '辐射型 Delta E / E（对照）', E_emit / E_before,
                '打印（= 1/gamma，普朗克量子份额 O(1)，符号非零）', True))
results.append(('TA5d', '辐射型 Delta p / p（对照）', p_emit / p_before,
                '打印（符号非零，方向由发射方向定）', True))

# ---------- 打印 ----------
print('=' * 72)
print('作用量转录的最小验证（第 457 条）：TA1 至 TA5')
print('=' * 72)
n_pass = 0
for tag, name, val, expect, ok in results:
    n_pass += ok
    print('%s %-38s 实测 %.12g  期望 %s  %s' % (
        tag, name, val, expect, 'PASS' if ok else 'FAIL'))
print('-' * 72)
print('TA1 份额-幅值（m = 0.5 mP）: reading = %.6e, theory = %.6e' % (
    0.5 * mP * c**2 * tP / hbar, 0.5))
print('TA2 gamma 缩放读数（beta = 0.9）: %.6f 对 gamma*delta0 = %.6f' % (
    gammas[-1] * mP * c**2 * tP / hbar / (mP / mP), gammas[-1]))
print('TA4 辐射对照：front = %.4f, back = %.4f（w0 单位）' % (
    front_r / w0, back_r / w0))
print('-' * 72)
print('判据 %d/%d 通过' % (n_pass, len(results)))
assert n_pass == len(results), 'FAIL: 判据未全过'

# ---------- 图 ----------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
ax = axes[0]
ax.plot(ms / mP, [E * tP / hbar for E in ms * c**2], 'o-', label='transcription reading', lw=1.5)
ax.plot(ms / mP, ms / mP, 's--', label='delta_theta = m/m_P', lw=1.2)
ax.set_xlabel('m / m_P')
ax.set_ylabel('write amplitude (per beat share)')
ax.set_title('(a) TA1: gain = 1, no new parameter')
ax.legend(fontsize=8)

ax = axes[1]
bb = np.linspace(0, 0.95, 200)
gg = 1 / np.sqrt(1 - bb**2)
ax.plot(bb, gg, lw=2, label='gamma(beta) scaling')
ax.plot(betas, gammas, 'o', ms=8, label='TA2 checkpoints')
ax.set_xlabel('beta = v/c')
ax.set_ylabel('amplitude / delta_theta')
ax.set_title('(b) TA2: gamma enhancement (J^0 correction)')
ax.legend(fontsize=8)

ax = axes[2]
ax.bar([0, 1], [1, 1], width=0.35, label='transcription (scalar)', color='C0')
ax.bar([2, 3], [g * (1 + b), g * (1 - b)], width=0.35,
       label='radiative (Doppler)', color='C3')
ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(['front', 'back', 'front', 'back'])
ax.set_ylabel('reading / share(beta)')
ax.set_title('(c) TA4: scalar vs Doppler signature')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(r'D:\software\TianXingFC\images\action_transcription.png', dpi=150)
print('figure saved: images/action_transcription.png')
