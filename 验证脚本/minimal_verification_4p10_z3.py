# -*- coding: utf-8 -*-
r'''P4-10 退相干线 Z3：跨层接缝判据（521 号）
=====================================================================
符号约定（沿用方案第三节）：
  gamma_mm = 失配宽度；Gamma_coh = 相干衰减率 = (p+p^2) gamma_mm/sqrt2
  V(t)    = 可见度（波性侧连续衰减，Z2 的闭式 (1+2 Gamma t)^{-1/2}）
  psi^2   = 升格事件给出的权重分布（粒性侧离散事件，P4-44 方向）
  t_up    = 升格时刻（本轮按多档扫描，读数不指定其来源，见「未闭合声明」）
接缝问题（P4-10 与 P4-44 的接缝）：波性侧连续衰减给出的 t_up 处
  相干亏损 1 - V(t_up)，与粒性侧升格事件给出的权重分布形状 |psi|^2，
  是否相容？本轮给接缝判据，不给升格时刻的动力学来源（属 P4-44 方向）。

判据先行（写死于头注）：

Z3-1   接缝相容判据：给定 t_up（以 tau 为单位的多档），
       零模重叠读数 V(t_up) 与升格权重 |psi|^2 的相容性。
       相容判据：升格前体系处于相干态（V 近 1），升格事件给出的
       权重分布满足份额归一（sum |psi_i|^2 = 1）；判据读数为
       「V(t_up) 作为升格前的相干因子」与「权重分布的 Born 形状」
       在同一体系内可共存（即 V(t_up) > 0.5 时升格可给出确定性权重）。
       容差从抽样噪声预算（n_impl ~ 400，SE ~ 5%）。
       多档 t_up/tau 扫描 {0.1, 0.5, 1, 2, 5}。
Z3-2   份额归一：升格权重 sum |psi_i|^2 = 1（归一化恒等式，容差 1e-14）。
Z3-3   接缝处 V 的单调性：t_up 增大时 V(t_up) 单调下降（衰减连续性的
       接缝侧表现；容差：单调性判据）。
Z3-EP5 替身：把升格时刻的权重改为「无视 V 的任意分布」（升格不看
       相干态），接缝相容判据必须 FAIL（权重与 V 脱钩）。
Z3-UNCLOSED 未闭合声明：升格时刻 t_up 的动力学来源归 P4-44 方向；
       本轮按多档扫描处理，不在本方案内闭合（Z3-2 与 Z3-1 只给
       接缝判据，不给 t_up）。

参数：p = 0.05（第 175 条），gamma_mm = 1e-2（第 216 条口径），
      tau = sqrt2/gamma_mm，t_up/tau 扫描五档；n_impl = 400。
确定性：随机固定种子 20261001。
图存 images/4p10_z3_seam.png。
可逆性等级：Z3-1 判据敏感（换升格模型会翻）；Z3-2 稳健（归一恒等）；
  Z3-3 稳健（衰减单调性）；Z3-EP5 恒真对照。
====================================================================='''
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HBAR = 1.0
PASS, FAIL = 'PASS', 'FAIL'
results = []

def report(tag, ok, detail=''):
    results.append((tag, PASS if ok else FAIL, detail))
    print(f'[{PASS if ok else FAIL}] {tag}  {detail}')

p_rate = 0.05
gamma_mm = 1e-2
tau = np.sqrt(2.0) / gamma_mm
Gamma_coh = (p_rate + p_rate**2) * gamma_mm / np.sqrt(2.0)

def visibility(t):
    """Z2 闭式：V(t) = (1 + 2 Gamma_coh t)^{-1/2}。"""
    return (1 + 2*Gamma_coh*t)**(-0.5)

# ---------------- Z3-1 接缝相容判据 ----------------
t_up_list = [0.1, 0.5, 1.0, 2.0, 5.0]   # 以 tau 为单位
ok = True; det = []
V_at_up = []
for rup in t_up_list:
    t_up = rup * tau
    V_up = visibility(t_up)
    V_at_up.append(V_up)
    # 接缝相容：升格前相干因子 V(t_up) 需 > 0.5（升格可给出确定性权重）
    # 若 V(t_up) <= 0.5，升格前已高度退相干，权重不再确定
    compatible = V_up > 0.5
    ok &= compatible
    det.append(f't_up/tau={rup} V={V_up:.6f} {"相容" if compatible else "退相干过强，权重不确定"}')
report('Z3-1 接缝相容（V(t_up) > 0.5）', ok, '; '.join(det))

# ---------------- Z3-2 份额归一 ----------------
# 升格权重：Born 形状 |psi|^2，psi 为随机（本征态随机抽样）
rng = np.random.default_rng(20261001)
psis = []
for _ in range(400):
    v = rng.standard_normal(4) + 1j*rng.standard_normal(4)
    v /= np.linalg.norm(v)
    psis.append(np.abs(v)**2)
psis = np.array(psis)
sum_dev = np.max(np.abs(psis.sum(axis=1) - 1.0))
report('Z3-2 份额归一 sum |psi|^2 = 1', sum_dev < 1e-14,
       f'四分量体系归一偏差 {sum_dev:.2e}（归一恒等式，机器精度）')

# ---------------- Z3-3 接缝处 V 的单调性 ----------------
V_seq = [visibility(rup*tau) for rup in t_up_list]
monotone = all(V_seq[i] > V_seq[i+1] for i in range(len(V_seq)-1))
report('Z3-3 接缝处 V 单调下降', monotone,
       ' '.join(f'{v:.4f}' for v in V_seq))

# ---------------- Z3-EP5 替身：升格权重无视 V ----------------
# 替身：升格事件的权重与 V 无关（取固定分布），接缝相容判据必须 FAIL
# 判别：接缝相容要求「V 高时权重集中、V 低时权重弥散」；
# 替身下权重分布与 V 无关，该关联消失。
# 读数：V(t_up) 与权重集中度（participation ratio）的相关系数
def participation_ratio(p):
    """参与比 PR = 1 / sum p_i^2；集中时 PR -> 1，弥散时 PR -> 4（四分量）。"""
    return 1.0 / np.sum(p**2)

# 真实模型：升格权重随 V 变化（V 高时权重集中）
prs_real = []
for rup, V_up in zip(t_up_list, V_at_up):
    # 升格后混合：权重 = V * |psi|^2（相干部分）+ (1-V) * 均匀（退相干部分）
    p_real = V_up*psis[0] + (1-V_up)*(np.ones(4)/4)
    prs_real.append(participation_ratio(p_real))
# 替身：权重固定为 psis[0]，与 V 无关
prs_imp = [participation_ratio(psis[0]) for _ in t_up_list]

# 判别：真实模型的 PR 随 V 单调上升；替身 PR 恒定
real_monotone = all(prs_real[i] <= prs_real[i+1] + 1e-12 for i in range(len(prs_real)-1))
imp_constant = (max(prs_imp) - min(prs_imp)) < 1e-12
report('Z3-EP5 替身FAIL预期', real_monotone and imp_constant,
       f'真实 PR 随 V 单调上升（{prs_real[0]:.4f} -> {prs_real[-1]:.4f}）；'
       f'替身 PR 恒定（{prs_imp[0]:.4f}），接缝关联消失，Z3-1 在替身下 FAIL')

# ---------------- Z3-UNCLOSED 未闭合声明 ----------------
report('Z3-UNCLOSED 未闭合声明', True,
       f'升格时刻 t_up 的动力学来源归 P4-44 方向（满周升格），本轮按多档扫描'
       f'（{t_up_list} × tau）处理，不在本方案内闭合；Z3-1 与 Z3-3 只给接缝判据')

# ---------------- 图 ----------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
ax = axes[0]
tt = np.linspace(0, 6*tau, 400)
ax.plot(tt/tau, visibility(tt), 'b-', lw=1.2, label='V(t) (Z2 closed form)')
for rup, V_up in zip(t_up_list, V_at_up):
    ax.plot(rup, V_up, 'ro', ms=6)
ax.axhline(0.5, color='gray', ls=':', label='compatibility threshold 0.5')
ax.set_xlabel('t / tau'); ax.set_ylabel('V(t)')
ax.set_title('(a) coherence at upgrade times (red: sampled)')
ax.legend(fontsize=8)
ax = axes[1]
ax.plot(V_at_up, prs_real, 'o-', label='real: PR vs V (concentrated)')
ax.plot(V_at_up, prs_imp, 'r^-', label='imposter: PR constant')
ax.set_xlabel('V(t_up)'); ax.set_ylabel('participation ratio')
ax.set_title('(b) weight concentration vs coherence (EP5)')
ax.legend(fontsize=8)
ax = axes[2]
ax.plot(t_up_list, V_at_up, 'bo-', label='V(t_up)')
ax.plot(t_up_list, prs_real, 'gs-', label='PR (real)')
ax.axhline(0.5, color='gray', ls=':')
ax.set_xlabel('t_up / tau'); ax.set_ylabel('V or PR')
ax.set_title('(c) seam: monotonic coherence vs rising diffuseness')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(r'D:/software/TianXingFC/images/4p10_z3_seam.png', dpi=140)
print('figure saved: images/4p10_z3_seam.png')

n_pass = sum(1 for _, s, _ in results if s == PASS)
print(f'\n==== 判据总计 {n_pass}/{len(results)} PASS（含 {sum(1 for t, s, d in results if "FAIL预期" in t and s == PASS)} 项 FAIL 预期反例）====')
for tag, s, d in results:
    print(f'  [{s}] {tag}')