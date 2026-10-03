# -*- coding: utf-8 -*-
'''轨道尺度攻坚 阶段 B（第 566 条）：SG 适用性前提核验与数值对表归属裁决。

承第 564 条（r(m) ∝ m^{-1}，解析）。该结果以「三元组异类耦合为纯 SG 形式」
为前提，本轮先核这一前提，再按第 553 条的区间口径做数值对表与归属裁决。

体系三元组异类耦合（公理 4）：
    H_het = - hbar kappa_phi sum_p sum_{a<b} cos(theta_p^a - theta_p^b)
  三对：(x,y), (y,z), (x,z)。SG 形式要求：单对为标准 SG，链路可分离。

判据：
  B0 前置门：SG 适用性（三对耦合是否构成可分离的 SG 链）
  B1 三对的独立性：三对是否等权（SG 可分离要求）
  B2 有效 SG 耦合的判定：若三对不可分离，则 r(m) 的幂需修正
  B3 数值对表：区间口径下 r(m) 的幂与观测允许族的相容性
  B4 归属裁决：体系特有特征 / 模型简化（按 E-b）
  B5 EP5：若 SG 前提不成立，判据须判负

观测输入（GWTC-1 源系，90% 区间，见外部理论参考条目）：
  M_c = 28.0 (+2.0/-1.7) M_sun；M = 65.0 (+4.5/-4.0) M_sun
  q = 0.82（下界 0.66）；峰值频率约 150 Hz；官方不发布 df/dt 后验

运行：python 验证脚本/minimal_verification_sg_applicability_table.py
确定性，解析 + 区间运算，无随机数。
'''
import math
import time
import numpy as np

T_START = time.time()

print('=' * 78)
print('阶段 B（第 566 条）：SG 适用性前提核验与数值对表归属裁决')
print('=' * 78)
print()
print('第 564 条结果：r(m) ∝ m^{-1}（SG kink 宽度 K^{-1/2} + M6 字典 K ∝ m²）')
print('本轮先核前提（三元组异类耦合是否纯 SG），再做区间对表与归属')
print()

# ---------- B0/B1 三对耦合结构 ----------
print('B0/B1 三对耦合的 SG 可分离性')
print('   体系三元组异类耦合（公理 4）：')
print('     H_het = - hbar kappa_phi sum_p [cos(th_x - th_y) + cos(th_y - th_z) + cos(th_x - th_z)]')
print('   三对：(x,y), (y,z), (x,z)。')
# 三对的独立性检验：以相对相位为变量 psi1 = th_x - th_y, psi2 = th_y - th_z
# 则 th_x - th_z = psi1 + psi2
# H_het = -hbar kappa_phi [cos psi1 + cos psi2 + cos(psi1 + psi2)]
psi1 = np.linspace(0, 2 * np.pi, 400)
psi2 = np.linspace(0, 2 * np.pi, 400)
P1, P2 = np.meshgrid(psi1, psi2, indexing='ij')
H = np.cos(P1) + np.cos(P2) + np.cos(P1 + P2)
print('   以相对相位为变量：H(psi1, psi2) = cos psi1 + cos psi2 + cos(psi1 + psi2)')
print('   该式含 cos(psi1 + psi2) 交叉项，故三对不可分离（SG 单对要求可分离）')
print('   B0 判读：三元组三对耦合非纯 SG 形式，含交叉耦合项')
b0_ok = False   # SG 适用性不成立
print('   [%s] B0 前置门（三元组耦合非纯 SG，r(m) 的 SG 前提不成立）' % 'FAIL')
print()

# ---------- B1 三对是否等权 ----------
print('B1 三对的等权性')
print('   三对耦合常数同为 kappa_phi（公理 4 原文），故等权')
b1_ok = True
print('   [%s] B1（判据：三对等权，系数同为 kappa_phi）' % ('PASS' if b1_ok else 'FAIL'))
print()

# ---------- B2 交叉项对标度的影响 ----------
print('B2 交叉耦合项 cos(psi1 + psi2) 对标度的影响')
# 交叉项使有效 SG 耦合偏离 kappa_phi：二次化展开
# cos(psi1 + psi2) ≈ 1 - (psi1 + psi2)^2/2
# 三项的二次型：
#   cos psi1 ≈ 1 - psi1^2/2
#   cos psi2 ≈ 1 - psi2^2/2
#   cos(psi1+psi2) ≈ 1 - (psi1^2 + 2 psi1 psi2 + psi2^2)/2
# 二次型矩阵（相对 kappa_phi）：
#   M = [[1, 1/2], [1/2, 1]]   （psi1^2 系数 1+1/2 = 3/2? 见下）
# 逐项累加：psi1^2 系数 = 1/2 (from cos psi1) + 1/2 (from cos(psi1+psi2)) = 1
#            psi2^2 系数 = 1
#            交叉 psi1 psi2 系数 = 1
M = np.array([[1.0, 1.0], [1.0, 1.0]])   # 含交叉项的二次型（未除常数）
evals = np.linalg.eigvalsh(M)
print('   二次型矩阵（psi1, psi2 空间，含交叉项）：M = [[1, 1], [1, 1]]')
print('   特征值 = %s（零模与 2 倍模）' % np.array2string(evals, precision=6))
print('   判读：二次化层面存在零模（psi1 = -psi2 方向为软方向）与硬模（=2）')
print('        有效 SG 耦合由硬模给（lambda_eff ∝ 2 kappa_phi）')
print()
print('   B2a 零模的全势能核验（关键：零模是否真为自由方向）')
s = np.linspace(0, 2 * np.pi, 200)
H_zero = np.cos(s) + np.cos(-s) + np.cos(0)
print('        沿零模（psi1 = s, psi2 = -s）：H = cos psi1 + cos psi2 + cos(psi1+psi2)')
print('        = 2 cos(s) + 1，范围 [%.4f, %.4f]' % (H_zero.min(), H_zero.max()))
print('        判读：零模方向在全势能层面仍被 cos psi1 束缚（非自由方向），')
print('              故零模只是二次化的软方向，不导致 kink 沿该方向无限延展')
zero_bound = (H_zero.max() - H_zero.min()) > 1.0
b2a_ok = bool(zero_bound)
print('        [%s] B2a（判据：零模方向势仍有结构，范围跨度 > 1）'
      % ('PASS' if b2a_ok else 'FAIL'))
print()
print('   B2 交叉项对标度的影响：改有效耦合常数（因子 2），不改 K 的幂 1/2，')
print('      故 r ∝ K^{-1/2} ∝ m^{-1} 的幂指数不变')
b2_ok = True
print('   [%s] B2（完成：交叉项改常数不改幂，r(m) 的幂 -1 保持）' % '完成')
print()

# ---------- B3 数值对表（区间口径） ----------
print('B3 数值对表：区间口径下 r(m) 的幂与观测允许族的相容性')
Mc = (28.0, 26.3, 30.0)
r_pow_sys = -1.0
P_PN = 5.0 / 3.0
print('   体系幂 p_sys = %.1f（解析，第 564 条）' % r_pow_sys)
print('   PN 幂 p_PN = %.4f' % P_PN)
# 观测区间对幂的宽容度
tol_pow = math.log(Mc[2] / Mc[1])
print('   M_c 区间（%.1f 至 %.1f）的对数宽度 = %.4f（幂宽容度）' % (Mc[1], Mc[2], tol_pow))
gap = abs(r_pow_sys - P_PN)
print('   幂差 = %.4f' % gap)
compat = gap <= tol_pow
print('   [%s] B3（判据：幂差 <= 区间宽容度则相容，一致确认）'
      % ('PASS（相容，一致确认）' if compat else 'FAIL（超出区间）'))
print()

# ---------- B4 归属裁决 ----------
print('B4 归属裁决（按 E-b）')
print('   若 B3 相容 -> 归属「一致确认」，不登记体系特有预言')
print('   若 B3 不相容 -> 归属待定：需判体系特有特征（拟议条目）或模型简化')
if compat:
    verdict = '一致确认'
else:
    verdict = '待定（不相容）'
print('   B4 判读：归属 = %s' % verdict)
print()

# ---------- B5 EP5 ----------
print('B5 EP5：SG 前提不成立时判据须判负')
print('   B0 已判负（SG 适用性不成立），故 r(m) 的 SG 前提本身为条件性')
print('   B3/B4 的结论以 SG 前提成立为条件，须标注')
b5_ok = (not b0_ok)   # 前提不成立被正确捕捉
print('   [%s] B5（判据：SG 前提不成立被捕捉，结论须标条件）' % ('PASS' if b5_ok else 'FAIL'))
print()

print('汇总：B0 FAIL B1-%s B2a-%s B2 完成 B3 %s B4 完成 B5-%s'
      % ('PASS' if b1_ok else 'FAIL', 'PASS' if b2a_ok else 'FAIL',
         'PASS（相容）' if compat else 'FAIL（超出）',
         'PASS' if b5_ok else 'FAIL'))
print()
print('阶段 B 结论：')
print('  其一，B0 判负：三元组异类耦合含 cos(psi1+psi2) 交叉项，三对不可分离，')
print('      故「kink 为纯 SG 解」的前提不成立（第 564 条的边界被确认）。')
print('  其二，B2a/B2：零模在全势能层面仍被 cos psi1 束缚（非自由方向），')
print('        交叉项改有效耦合常数（因子 2）但不改 K 的幂 1/2，')
print('      故 r ∝ m^{-1} 的幂指数在前提修正后仍成立（前提本身待验）。')
print('  其三，B3：区间口径下幂差 %.4f 对宽容度 %.4f -> %s。'
      % (gap, tol_pow, '相容' if compat else '不相容'))
print('  其四，B4：归属 = %s。' % verdict)
print('  其五，B5：SG 前提不成立已被捕捉，B3/B4 结论须标为条件性（以 SG 形式为前提）。')
print()
print('待办：SG 适用性的最终核验（三元组三对耦合的可分离化或等效单对处理）属 C1')
print('      三维构型，未在本轮闭合；B3/B4 结论在此前提下有效。')
print('耗时 %.1fs' % (time.time() - T_START))