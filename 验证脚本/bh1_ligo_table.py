# -*- coding: utf-8 -*-
'''BH1-3 第二轮（第 553 条）：体系相位族与 GW150914 拟合族的数值对表。

【撤回提示 2026-10-03】本脚本的归属裁决已撤回：质量标度幂由构造 r ∝ 1/m 决定，
非体系物理。N2 判负测的是构造选择。待质量标度前置判据完成后重做。

按 E-a 裁决：以已公开的并合波形拟合族为观测允许族代表，区间口径处理拟合
参数不确定性，不做逐事件拟合。官方数据见外部理论参考条目「LIGO GW150914
拟合参数」：源系 M = 65.0 +4.5/-4.0 M_sun，M_c = 28.0 +2.0/-1.7 M_sun，
q = 0.82 +0.17/-0.20（下界 > 0.66），峰值频率约 150 Hz（90% 区间）。

判据：
  N0 前置门：观测区间须非退（否则停手）
  N1 质量标度幂的观测侧：观测族的 M_c 与总质量区间给出的幂约束
  N2 体系预言与观测允许族的相容性（区间口径）
  N3 eta 依赖的相容性（q 区间 vs 体系 mu）
  N4 EP5：替身（不可区分族）须被排除

运行：python 验证脚本/bh1_ligo_table.py
确定性，无随机数（区间端点为输入）。
'''
import math
import time
import numpy as np

T_START = time.time()

# ---------- 观测输入（GWTC-1 源系，90% 区间；见外部理论参考条目） ----------
M_src = (65.0, 60.5, 69.5)          # 总质量（下限、上限）
Mc_src = (28.0, 26.3, 30.0)         # 约化质量
q_src = (0.82, 0.66, 0.99)           # 质量比（下界为 90% 下界）
f_peak = 150.0                        # 峰值频率 Hz
M_SUN_MP = 1.0e-6                     # M☉ / M_Pl（近似，仅作标度换算的锚）

TP = LP = 1.0
C = LP / TP
OMEGA_P = 2 * np.pi / TP
KAPPA_PHI = 1.0 / (6 * np.pi * TP)
OMEGA_0 = math.sqrt(3 * OMEGA_P * KAPPA_PHI)
P_PN = 5.0 / 3.0

print('=' * 78)
print('BH1-3 第二轮（第 553 条）：体系相位族与 GW150914 拟合族的数值对表')
print('=' * 78)
print()
print('观测输入（GWTC-1 源系，90% 区间）：')
print('   M = %.1f (+%.1f/-%.1f) M_sun；M_c = %.1f (+%.1f/-%.1f) M_sun'
      % (M_src[0], M_src[0] - M_src[1], M_src[2] - M_src[0], Mc_src[0],
         Mc_src[0] - Mc_src[1], Mc_src[2] - Mc_src[0]))
print('   q = %.2f（下界 %.2f，90%%）；峰值频率 %.0f Hz' % (q_src[0], q_src[1], f_peak))
print('   口径（E-a）：以拟合族为观测允许族代表，区间处理，不做逐事件拟合')
print()


def v_g(k):
    k = np.asarray(k, dtype=float)
    return C ** 2 * k / np.sqrt(C ** 2 * k ** 2 + OMEGA_0 ** 2)


def omega_end(mass_norm, eta, r_min=0.05):
    mu = (1.0 + eta) / (1.0 + 3.0 * eta)
    r = r_min * (0.5 + mu) / mass_norm
    return v_g(1.0 / r) / r


# ---------- N0 前置门 ----------
print('N0 前置门：观测区间须非退')
width_M = (M_src[2] - M_src[1]) / M_src[0]
width_Mc = (Mc_src[2] - Mc_src[1]) / Mc_src[0]
n0_ok = width_M > 0.05 and width_Mc > 0.05
print('   M 的相对宽度 = %.4f，M_c 的相对宽度 = %.4f' % (width_M, width_Mc))
print('   [%s] N0 前置门（判据：相对宽度 > 5%%，区间非退）' % ('PASS' if n0_ok else 'FAIL'))
print()

# ---------- N1 观测侧质量标度幂约束 ----------
print('N1 观测侧的质量标度幂约束')
# PN 侧：omega ∝ M_c^(5/3)；观测的 M_c 区间给出 omega 的区间约束
# 幂宽容度的正确量纲：观测的 M_c 区间（对数宽度）能吸收多大的幂差
# 若体系的幂差为 dp，则在 M_c 区间上表现为 (M_c^hi/M_c^lo)^dp 的频率变化；
# 观测允许族的区间恰好覆盖该变化，则相容。
tol = math.log(Mc_src[2] / Mc_src[1])
span_freq = (Mc_src[2] / Mc_src[1]) ** P_PN
print('   M_c 区间（%.1f 至 %.1f）的对数宽度 = %.4f（幂宽容度的正确量纲）'
      % (Mc_src[1], Mc_src[2], tol))
print('   该区间在 PN 幂 5/3 下给出的频率比 = %.4f' % span_freq)
print('   读法：观测区间能吸收的幂差上限 = %.4f（区间宽度所致）' % tol)
n1_ok = True
print('   [%s] N1（观测侧的幂宽容度已量化，供 N2 判定）' % '完成')
print()

# ---------- N2 体系预言与观测允许族的相容性 ----------
print('N2 相容性：体系的 chirp 幂 vs 观测允许族')
masses = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
eta_ref = 0.5
vals = np.array([omega_end(m, eta_ref) for m in masses])
p_sys = float(np.polyfit(np.log(masses), np.log(vals), 1)[0])
print('   体系幂 p_sys = %.4f；PN 幂 p_PN = %.4f；差 %.4f' % (p_sys, P_PN, abs(p_sys - P_PN)))
# 相容判据：若体系预言的幂差落在观测区间宽容度内，则相容（零结果一致确认）
gap = abs(p_sys - P_PN)
compat = gap <= tol
print('   观测区间宽容度 = %.4f；幂差 = %.4f' % (tol, gap))
print('   [%s] N2（判据：幂差 <= 区间宽容度则相容，一致确认）'
      % ('PASS（相容，一致确认）' if compat else 'FAIL（超出区间）'))
print()

# ---------- N3 eta 依赖相容性 ----------
print('N3 eta 依赖的相容性（q 区间）')
etas = np.linspace(q_src[1], min(q_src[2], 1.0), 9)
om_eta = np.array([omega_end(1.0, e) for e in etas])
om_span = float(np.max(om_eta) / np.min(om_eta))
print('   q 区间（%.2f 至 %.2f）对应的体系末态频率跨度 = %.4f'
      % (etas[0], etas[-1], om_span))
# 观测的 q 不确定度大到接近先验，故对表结论只能是「不可判定」或「宽容」
q_wide = (q_src[2] - q_src[1]) / q_src[0]
print('   q 的相对宽度 = %.4f（论文称 q 只被弱约束）' % q_wide)
n3_ok = True   # q 弱约束，结构相容但数值不可判定
print('   [%s] N3（q 弱约束，结构相容；数值层面不可判定，记为区间口径）' % '完成')
print()

# ---------- N4 EP5 ----------
print('N4 EP5：不可区分的替身须被排除')
# 替身：把体系幂改设为 5/3（数值解 r_min 指数使幂恰为 5/3）
def p_of(a):
    vv = []
    for m in masses:
        mu = (1.0 + eta_ref) / (1.0 + 3.0 * eta_ref)
        r = 0.05 * m ** (-a) * (0.5 + mu)
        vv.append(v_g(1.0 / r) / r)
    return float(np.polyfit(np.log(masses), np.log(vv), 1)[0])
lo, hi = 0.3, 2.0
for _ in range(60):
    mid = 0.5 * (lo + hi)
    if p_of(mid) < P_PN:
        lo = mid
    else:
        hi = mid
A_STAR = 0.5 * (lo + hi)
p_alt = p_of(A_STAR)
dev = abs(p_alt - p_sys)
print('   替身（数值解 a* = %.4f 使幂为 5/3）的幂 = %.4f；与体系幂差 = %.4f'
      % (A_STAR, p_alt, dev))
n4_ok = dev > 0.1
print('   [%s] N4（判据：替身与体系幂可区分，> 0.1）' % ('PASS' if n4_ok else 'FAIL'))
print()

print('汇总：N0-%s N1 完成 N2 %s N3 完成 N4-%s'
      % ('PASS' if n0_ok else 'FAIL', '相容' if compat else '不相容',
         'PASS' if n4_ok else 'FAIL'))
print()
if n0_ok and compat and n4_ok:
    print('数值对表结论：体系 chirp 幂 %.4f 与 PN 的 %.4f 的差 %.4f 落在观测区间'
          % (p_sys, P_PN, gap))
    print('宽容度 %.4f 内 -> 按 E-b 裁决记为一致确认（不登记拟议条目）。' % tol)
    print('q 依赖因官方弱约束（相对宽度 %.2f）记为区间口径，数值层面不可判定。' % q_wide)
    print('BH1-3 第二轮数值对表完成；归属裁决随之落定为「口径差」而非「体系特有特征」。')
else:
    print('数值对表结论（N2 判负）：')
print('  体系 chirp 幂 %.4f 与 PN 的 %.4f 的差 %.4f，为观测区间宽容度 %.4f 的 %.1f 倍，'
      % (p_sys, P_PN, gap, tol, gap / tol))
print('  故体系的质量标度幂落在观测允许族之外。')
print('  按 E-b 裁决：若归属甲（体系特有特征）且该差异高于观测精度，')
print('  则登记为拟议预言条目并附预言清单 §8 透明度声明。')
print('  归属甲的可判签名已在第 552 条建立（幂与收缩指数无关、简化不解释该差异），')
print('  故本条判负支持归属甲而非乙：若差异源于模型简化，放开简化后幂应回到 5/3，')
print('  但第 552 条 A1/A2 显示放开简化后幂不变（1.018 不变）。')
print('  BH1-3 第二轮数值对表完成。')
print()
print('【2026-10-03 撤回】上述归属裁决作废：幂差源于构造选择 r ∝ 1/m，')
print('  四个构造（r∝1/m 给 +1.03、r∝m 给 -1.00、r∝m^(1/3) 给 -0.33、PN 给 +1.67）')
print('  都是构造，N2 判负测的是构造而非体系物理。改为「待质量标度前置判据」。')
print('耗时 %.1fs' % (time.time() - T_START))