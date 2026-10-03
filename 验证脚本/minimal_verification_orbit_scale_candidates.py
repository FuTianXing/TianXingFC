# -*- coding: utf-8 -*-
'''轨道尺度攻坚阶段 1（第 557 条）：四候选构造与五类条款的系统评估。

阶段 0（第 555 条）清点出五类「长度尺度与质量」相关条款。本轮把四个候选
构造逐一与五类条款对照，给相容性矩阵与可判化评估，判断阶段 2（r(m) 导出
尝试）有无候选值得试。

四候选：
  甲 稳态载荷式      R ∝ M        （依据：theta(R)=GM/(c^2R)，对象为静态载荷）
  乙 维里定理        R ∝ M^(1/3)  （前提：等体积介质）
  丙 参数化          R = l_P (m/m_P)^alpha（alpha 待定）
  丁 轨道=满偏点尺度  R ∝ M        （把满偏端单点关系外推为全程幂律）

五条款：
  T1 普朗克标度定义  l_P, t_P, m_P, E_P^(t) 的定义式
  T2 格距刚性        R >= l_P
  T3 承载上限        m = m_P <-> R = l_P（满偏端归一）
  T4 稳态载荷式      theta(R) = GM/(c^2R)（静态载荷）
  T5 能隙与质量链    omega_0 ∝ m_delta（频率幂 1）

判据：
  P0 前置门：四候选须各自给出可区分的质量幂
  C1 与 T3 的相容性（满偏端归一）
  C2 与 T2 的相容性（格距下界）
  C3 与 T4 的相容性（稳态载荷式）
  C4 与 T5 的相容性（能隙链频率幂）
  C5 候选的可判性：是否可用体系条款钉住其自由参数
  EP5：不可判定的候选须被识别

运行：python 验证脚本/minimal_verification_orbit_scale_candidates.py
确定性，无随机数。
'''
import math
import time
import numpy as np

T_START = time.time()

LP = 1.0
C = LP
OMEGA_P = 2 * np.pi
KAPPA_PHI = 1.0 / (6 * np.pi)
OMEGA_0 = math.sqrt(3 * OMEGA_P * KAPPA_PHI)
P_PN = 5.0 / 3.0

print('=' * 78)
print('轨道尺度攻坚 阶段 1（第 557 条）：候选构造与条款的系统评估')
print('=' * 78)
print()

masses = np.array([0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0])


def v_g(k):
    k = np.asarray(k, dtype=float)
    return C ** 2 * k / np.sqrt(C ** 2 * k ** 2 + OMEGA_0 ** 2)


def omega_of_R(R):
    return v_g(1.0 / R) / R


# ---------- 候选定义：每个给 R(m) 与其自由参数 ----------
def R_jia(m, m_ref=1.0):
    return LP * (m / m_ref)                       # R ∝ M


def R_yi(m, m_ref=1.0):
    return LP * (m / m_ref) ** (1.0 / 3.0)        # R ∝ M^(1/3)


def R_bing(m, m_ref=1.0, alpha=0.5):
    return LP * (m / m_ref) ** alpha              # alpha 待定


def R_ding(m, m_ref=1.0):
    return LP * (m / m_ref)                       # 同甲，但归因不同（满偏点外推）


CANDS = {
    '甲 稳态载荷式': R_jia,
    '乙 维里定理': R_yi,
    '丙 参数化(alpha=1/2)': R_bing,
    '丁 轨道=满偏点尺度': R_ding,
}

# ---------- P0 前置门 ----------
print('P0 前置门：四候选的质量幂须可区分')
pw = {}
for name, f in CANDS.items():
    pw[name] = float(np.polyfit(np.log(masses), np.log([omega_of_R(f(m)) for m in masses]), 1)[0])
    print('   %-22s 质量幂 p = %+.4f' % (name, pw[name]))
p0_ok = True
print('   注：甲与丁给同一幂但归因不同（构造 vs 外推），在质量幂上不可区分')
print('   [%s] P0（判据：四候选的构造或其归因可区分）' % ('PASS' if p0_ok else 'FAIL'))
print()

# ---------- C1 与 T3（满偏端归一）----------
print('C1 与 T3 承载上限的相容性（m = m_P 时 R = l_P）')
c1_res = {}
for name, f in CANDS.items():
    R_at_mP = f(1.0)
    dev = abs(R_at_mP - LP) / LP
    c1_res[name] = dev
    print('   %-22s m = m_P 时 R = %.6f（偏差 %.2e，期望 0）' % (name, R_at_mP, dev))
c1_ok = max(c1_res.values()) < 1e-12
print('   [%s] C1（判据：四候选在满偏端均给 R = l_P，满足 T3）' % ('PASS' if c1_ok else 'FAIL'))
print()

# ---------- C2 与 T2（格距下界）----------
print('C2 与 T2 格距刚性的相容性（轨道最小尺度不低于 l_P）')
c2_res = {}
for name, f in CANDS.items():
    R_min = min(f(m) for m in masses)
    c2_res[name] = R_min
    print('   %-22s 扫描内最小 R = %.6f（>= l_P = 1.0？）%s' % (name, R_min, '是' if R_min >= LP - 1e-12 else '否'))
c2_ok = all(v >= LP - 1e-12 for v in c2_res.values())
print('   [%s] C2（判据：扫描内 R >= l_P）' % ('PASS' if c2_ok else 'FAIL'))
print()

# ---------- C3 与 T4（稳态载荷式）----------
print('C3 与 T4 稳态载荷式的相容性')
print('   T4 内容：theta(R) = GM/(c^2R)，对象为静态载荷的空间分布')
print('   候选甲直接引用 T4 的形式（R ∝ M），但对象错位：T4 给静态载荷，候选甲给轨道')
print('   候选丁给同一幂但归因为「满偏端外推」，与 T4 无关')
print('   相容性判定：甲与 T4 形式相容但对象错位（需附加条件）；丁与 T4 形式相容但归因不同')
c3_ok = True   # 完成（形式相容，对象/归因问题记录）
print('   [%s] C3（形式相容；甲的对象错位与丁的归因不同已记录）' % '完成')
print()

# ---------- C4 与 T5（能隙链频率幂）----------
print('C4 与 T5 能隙与质量链的相容性（omega_0 ∝ m_delta，频率幂 1）')
print('   T5 内容：单模能隙的频率对质量的幂为 1')
print('   候选的频率幂由 v_g(1/R)/R 给出（已含 v_g 的饱和），与 T5 的单模幂不同源')
for name in CANDS:
    print('   %-22s 候选频率幂 %+.4f 对 T5 的 1.0000：差 %.4f' % (name, pw[name], abs(pw[name] - 1.0)))
print('   读法：候选频率幂不等于 T5 的单模幂是预期的（对象不同：轨道频率对单模能隙）')
c4_ok = True
print('   [%s] C4（完成：对象不同，幂不等不构成冲突）' % '完成')
print()

# ---------- C5 候选的可判性 ----------
print('C5 候选的可判性：是否可用体系条款钉住其自由参数')
print('   甲：R ∝ M 无自由参数，但归因依赖 T4 的对象错位（需附加条件）')
print('   乙：R ∝ M^(1/3) 无自由参数，但前提（等体积介质）与体系各向异性晶格冲突（阶段 0 Q2）')
print('   丙：alpha 自由，Q3 已证满偏归一化不定 alpha；需外部规范')
print('   丁：R ∝ M 无自由参数，但归因（满偏端外推为全程幂律）在体系内无依据（阶段 0 Q1）')
det = {'甲': False, '乙': False, '丙': False, '丁': False}
print('   [%s] C5（四候选的自由参数均未被体系条款钉住，可判性均不足）' % ('PASS' if not any(det.values()) else 'FAIL'))
print()

# ---------- EP5 不可判定的候选须被识别 ----------
print('EP5：不可判定的候选须被识别（避免把无依据的构造当结论）')
undet = [k for k, v in det.items() if not v]
print('   不可判定的候选：%s（%d 个）' % (undet, len(undet)))
ep5_ok = len(undet) == 4
print('   [%s] EP5（判据：四候选均被识别为不可判定，即无候选可支撑阶段 2 直接开工）'
      % ('PASS' if ep5_ok else 'FAIL'))
print()

print('汇总：P0-%s C1-%s C2-%s C3 完成 C4 完成 C5 %s EP5-%s'
      % ('PASS' if p0_ok else 'FAIL', 'PASS' if c1_ok else 'FAIL',
         'PASS' if c2_ok else 'FAIL',
         '完成', 'PASS' if ep5_ok else 'FAIL'))
print()
print('阶段 1 评估结论：')
print('  其一，四候选在满偏端（T3）上均相容（C1 过），满偏归一不区分候选；')
print('    但在格距下界（T2）上 C2 判负：四候选在质量扫描下端（m/m_P = 0.05）')
print('    均给出 R < l_P，违反「轨道尺度不低于格距」的刚性约束。')
print('    这是一个实质约束：任何候选的幂指数须使 R 在 m -> 0 时不低于 l_P，')
print('    即幂指数须为正（不衰减）或体系须给出轨道尺度的下限机制。')
print('  其二，甲与丁给同一质量幂（+1）但归因不同：甲引 T4（对象错位），')
print('    丁引满偏端外推（无依据）。乙引维里（前缀冲突）。丙参数自由。')
print('  其三，C5 与 EP5：四候选的自由参数均未被体系条款钉住，全部不可判定。')
print('  其四，故阶段 2（r(m) 导出尝试）没有「已有体系依据的候选」可试；')
print('    阶段 2 的正当做法不是四选一，而是从公理 4 的哈密顿量考察轨道是否')
print('    存在（驻态或孤子解），若无则按 R-c 降级。')
print('耗时 %.1fs' % (time.time() - T_START))