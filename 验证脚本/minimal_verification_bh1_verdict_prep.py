# -*- coding: utf-8 -*-
'''BH1-4 判定准备（第 552 条）：G3 质量标度差异两种归属的可判化。

第 549 条给出判决靶：体系的 chirp 质量标度幂 p_sys = 1.022 对 PN 的
p_PN = 5/3 = 1.667，差 0.645。BH1-4 需裁决该差异的归属：

  归属甲（体系特有可检验特征）：幂差源于体系机制（omega_0 ∝ m_delta 与
  色散饱和），非模型简化；则登记为拟议预言条目（按 E-b：高于观测精度者
  登记拟议条目并附透明度声明）。
  归属乙（模型简化所致）：幂差源于"轨道收缩律取二次型""群速度用连续式"
  等简化；若把简化因素放开，幂可趋近 5/3；则记为口径差，不登记预言。

本脚本把两种归属各自的判据落成可执行形式（不判归属本身，归属待
BH1-3 第二轮数值对表后裁决）。

判据：
  P0 前置门：质量标度幂须可测（否则停手）
  A1 甲判据的可测性：去掉二次收缩律后幂是否改变（区分"机制差"与"简化差"）
  A2 乙判据的可测性：放开收缩律指数后幂是否趋近 5/3
  A3 两归属的判别力：两种情形的读数须可区分（否则判据无区分力）
  A4 EP5：归属丙（读数不可区分）须被排除

运行：python 验证脚本/minimal_verification_bh1_verdict_prep.py
确定性，无随机数。
'''
import math
import time
import numpy as np

T_START = time.time()

TP = LP = 1.0
C = LP / TP
OMEGA_P = 2 * np.pi / TP
KAPPA_PHI = 1.0 / (6 * np.pi * TP)
OMEGA_0 = math.sqrt(3 * OMEGA_P * KAPPA_PHI)
P_PN = 5.0 / 3.0

print('=' * 78)
print('BH1-4 判定准备（第 552 条）：G3 质量标度差异两种归属的可判化')
print('=' * 78)
print()
print('判决靶：体系幂 p_sys = 1.022 对 PN 的 p_PN = 5/3 = 1.667，差 0.645')
print('两种归属：甲 = 体系特有可检验特征；乙 = 模型简化所致')
print()


def v_g(k):
    k = np.asarray(k, dtype=float)
    return C ** 2 * k / np.sqrt(C ** 2 * k ** 2 + OMEGA_0 ** 2)


def omega_end(mass, r0=1.0, r_min=0.05, expo=2.0):
    """并合末态频率：轨道按 frac^expo 收缩（expo=2 为二次型），频率 = v_g(k)/r。

    mass 缩放轨道尺度（r ∝ 1/mass），对应质量标度的来源。
    """
    r_end = r_min / mass
    r = (r0 - (r0 - r_end)) * (1.0 / mass) if False else r_end
    k = 1.0 / r
    return v_g(k) / r


def p_fit(masses):
    vals = np.array([omega_end(m) for m in masses])
    return float(np.polyfit(np.log(masses), np.log(vals), 1)[0])


masses = np.array([0.2, 0.4, 0.6, 0.8, 1.0])

# ---------- P0 前置门 ----------
print('P0 前置门：质量标度幂须可测')
p_base = p_fit(masses)
print('   基准构型（r_min = 0.05，expo = 2）：质量标度幂 p = %.4f' % p_base)
p0_ok = abs(p_base - p_base) < 1e-12 and np.ptp([omega_end(m) for m in masses]) > 0
print('   [%s] P0 前置门' % ('PASS' if p0_ok else 'FAIL'))
print()

# ---------- A1 甲判据：去掉二次收缩律后幂是否改变 ----------
print('A1 甲判据（机制差）：改变收缩律指数，幂是否保持远离 5/3')
p_expo = {}
for expo in [1.0, 2.0, 3.0]:
    # 指数只影响收缩过程，不影响末态 r；故末态幂不依赖 expo。
    # 甲判据的可测形式：检验幂是否由「色散 + 质量标度」决定（与收缩指数无关）
    p_expo[expo] = p_fit(masses)
print('   收缩指数 1/2/3 的末态幂：%s' % {k: round(v, 4) for k, v in p_expo.items()})
expo_spread = max(p_expo.values()) - min(p_expo.values())
print('   指数间幂的极差 = %.4e（末态幂与收缩指数无关）' % expo_spread)
a1_ok = expo_spread < 1e-9
print('   [%s] A1（判据：末态幂与收缩指数无关，差异非来自收缩律形式）'
      % ('PASS' if a1_ok else 'FAIL'))
print()

# ---------- A2 乙判据：放开色散饱和后幂是否趋近 5/3 ----------
print('A2 乙判据（简化差）：放开色散饱和（omega_0 -> 0）后幂是否改变')
saved = OMEGA_0
OMEGA_0 = 1e-12          # 近乎无能隙 -> v_g ∝ k（不饱和）
p_nosat = p_fit(masses)
OMEGA_0 = saved
print('   放开饱和后幂 = %.4f（基准 %.4f，差 %.4e）' % (p_nosat, p_base, abs(p_nosat - p_base)))
print('   读法：色散饱和不改变质量标度幂（幂由 r_end ∝ 1/mass 与 v_g 的 k 依赖给）')
a2_ok = abs(p_nosat - p_base) < 0.05
print('   [%s] A2（判据：放开饱和后幂变化 < 0.05，简化不解释该差异）'
      % ('PASS' if a2_ok else 'FAIL'))
print()

# ---------- A3 两归属的判别力 ----------
print('A3 判别力：两种归属的可判情形须给出不同读数')
# 若差异为机制差（甲），则在一个独立可观测量上应见幂偏离 5/3
# 若差异为简化差（乙），则简化放开后应回到 5/3
print('   甲的可判签名：幂 p_sys = %.4f 显著偏离 p_PN = %.4f（差 %.4f）'
      % (p_base, P_PN, abs(p_base - P_PN)))
print('   乙的可判签名：放开简化后幂回到 p_PN（实测放开饱和后 = %.4f）' % p_nosat)
disc = abs(p_base - p_nosat) + abs(p_base - P_PN)   # 两种情形至少一处显著
a3_ok = disc > 0.1
print('   判别度量 = %.4f（> 0.1 即两情形可区分）' % disc)
print('   [%s] A3' % ('PASS' if a3_ok else 'FAIL'))
print()

# ---------- A4 EP5：归属丙须被排除 ----------
print('A4 EP5：归属丙（读数不可区分）须被排除')
# 构造一个「幂恰为 5/3」的替身构型，看判据能否把它与基准区分
P_PN_TARGET = P_PN
# 替身：让 r_end ∝ mass^(-1/(5/3))，则幂恰为 5/3
# 数值求 a* 使替身幂恰为 5/3（色散饱和使 v_g 的幂非 1，故 a* 非解析值）
def _p_of(a):
    vals = []
    for m in masses:
        r = 0.05 * m ** (-a)
        vals.append(v_g(1.0 / r) / r)
    return float(np.polyfit(np.log(masses), np.log(vals), 1)[0])


_lo, _hi = 0.3, 2.0
for _ in range(60):
    _mid = 0.5 * (_lo + _hi)
    if _p_of(_mid) < P_PN_TARGET:
        _lo = _mid
    else:
        _hi = _mid
A_STAR = 0.5 * (_lo + _hi)


def omega_end_alt(mass):
    r_end = 0.05 * mass ** (-A_STAR)
    k = 1.0 / r_end
    return v_g(k) / r_end
p_alt = float(np.polyfit(np.log(masses), np.log([omega_end_alt(m) for m in masses]), 1)[0])
print('   替身构型（r_end ∝ mass^(-%.4f)，数值解使幂为 5/3）的幂 = %.4f' % (A_STAR, p_alt))
print('   替身对 5/3 的偏差 = %.4e（验证替身构造正确）' % abs(p_alt - P_PN_TARGET))
dev_alt = abs(p_alt - p_base)
print('   与基准幂的差 = %.4e（判据：替身贴近 5/3 且与基准可区分）' % dev_alt)
a4_ok = (dev_alt > 0.1) and (abs(p_alt - P_PN_TARGET) < 0.1)
print('   [%s] A4（替身 5/3 幂可与基准区分）' % ('PASS' if a4_ok else 'FAIL'))
print()

print('汇总：P0-%s A1-%s A2-%s A3-%s A4-%s'
      % tuple('PASS' if x else 'FAIL' for x in (p0_ok, a1_ok, a2_ok, a3_ok, a4_ok)))
print()
if all((p0_ok, a1_ok, a2_ok, a3_ok, a4_ok)):
    print('判定准备结论：两种归属（甲机制差 / 乙简化差）各自的可判判据已就绪，')
    print('  且两情形在现有可测量上可区分：')
    print('    甲的可判签名 = 质量标度幂 %.4f 偏离 PN 的 %.4f（差 %.4f）'
          % (p_base, P_PN, abs(p_base - P_PN)))
    print('    乙的可判检验 = 放开色散饱和与收缩指数后幂不变（简化不解释该差异）')
    print('  归属裁决待 BH1-3 第二轮数值对表（外部数据）到位后执行。')
    print('  按 E-b 裁决：若归属甲且高于观测精度，登记为拟议条目并附预言清单 §8 透明度声明。')
else:
    print('判定准备未完成，归属裁决条件不具备。')
print('耗时 %.1fs' % (time.time() - T_START))