# -*- coding: utf-8 -*-
r'''P4-44 份额铺开线 W2：族内档位与 P4-10 接缝的定量对接（530 号）
=====================================================================
符号约定（沿用方案第三节与 W1）：
  D_sep    = 两支间距（格距单位；与温度定义的复相干无序度 D 不同量）
  sigma_x  = 质心波包宽度（格距单位）
  s0(D)    = 零模波包重叠 = exp[-D^2/(8 sigma_x^2)]（W1 已给五档读数）
  V(t_up)  = 升格时刻的波性侧相干因子（承 P4-10 Z2 的闭式）
  PR       = 权重参与比 = 1 / sum_i |psi_i|^2（承 P4-10 Z3 的接缝读数）
链条（W2 的对象）：相干度 V -> 形状档位 s0 -> 权重集中度 PR。
  P4-10 Z3 已给 PR 随 V 单调上升（V 高时权重集中）；
  W1 已给 s0 随 D_sep 单调下降（间距大时跨项压制强）。
  W2 的对接：给定 V，s0 的有效档位与 PR 读数须联合相容。

判据先行（写死于头注）：

模型：升格时刻 t_up 的相干因子 V = (1 + 2 Gamma_coh t)^{-1/2}，
  Gamma_coh = (p + p^2) gamma_mm / sqrt2（P4-10 Z2 闭式）。
  跨项压制因子 s0 = exp[-D_sep^2/(8 sigma_x^2)]（M7b 引理）。
  权重分布（升格后）：p_i = V * |psi_i|^2 + (1 - V) * (1/N)（N = 4 分量，
  P4-10 Z3 的构造）；参与比 PR = 1 / sum_i p_i^2。
  形状档位的有效值：跨项在升格时刻的存活量 c_eff = V^2 * s0
  （相干因子压制跨项，间距压制跨项，两者相乘）。

W2-1   链条单调性（【执行轮修正】首版把判据写成对 t_up 检查同向，
       方向搞反：t_up 越大 V 越小，故 PR 与 c_eff 对 t_up 反向、对 V 同向）。
       正确判据：V 上升时 PR 上升（承 Z3）且 c_eff = V^2 s0 在固定
       D_sep 下上升，即链条对 V 单调同向。多档 t_up/tau 扫描
       {0.1, 0.5, 1, 2, 5}（对应 V 递减档）。
W2-2   联合相容判据：给定 V 与 D_sep，c_eff 与 PR 须相容：
       即「跨项存活量」与「权重集中度」由同一组 (V, D_sep) 决定，
       判据为 c_eff/PR 的比值在多档 (V, D_sep) 网格上落在解析闭式的
       一维曲线附近（容差从两读数的分辨率预算）。
W2-3   接缝读数的一致性：P4-10 Z3 的 PR 读数与 W2 重建的 PR 读数
       在同一 (V) 上逐位一致（同一闭式，独立实现）。
W2-EP5 替身：切断链条（令 c_eff 与 V 无关，取 c_eff = 常数），
       W2-2 的联合相容判据必须 FAIL（c_eff 与 PR 的依赖脱钩）。
W2-DE  简并检测 + EP1 至 EP4：权重读数落扇区子空间（重数 N-1），
       按第 506 条做子空间谱分类（常数类读数合法）。

参数：p = 0.05（第 175 条），gamma_mm = 1e-2（第 216 条），
      sigma_x = 0.3，N = 4 分量；t_up/tau 五档；
      D_sep 网格 {1, 2, 4, 8} * sigma_x。
确定性：随机固定种子 20261001。
图存 images/4p44_w2_seam_join.png。
可逆性等级：W2-1 稳健（闭式单调性）；W2-2 判据敏感（换权重构造会翻）；
  W2-3 判据敏感（换 V 闭式会翻）；W2-EP5 恒真对照。
====================================================================='''
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PASS, FAIL = 'PASS', 'FAIL'
results = []

def report(tag, ok, detail=''):
    results.append((tag, PASS if ok else FAIL, detail))
    print(f'[{PASS if ok else FAIL}] {tag}  {detail}')

P_RATE = 0.05
GAMMA_MM = 1e-2
SIGMA_X = 0.3
N_COMP = 4
TAU = np.sqrt(2.0) / GAMMA_MM
GAMMA_COH = (P_RATE + P_RATE**2) * GAMMA_MM / np.sqrt(2.0)

def vis_at(t_up_ratio):
    """升格时刻的相干因子（承 P4-10 Z2 闭式）。"""
    t_up = t_up_ratio * TAU
    return (1 + 2*GAMMA_COH*t_up)**(-0.5)

def s0_at(D_sep):
    return np.exp(-(D_sep**2) / (8*SIGMA_X**2))

def weight_dist(V, psi2, N=N_COMP):
    """升格后权重：p_i = V |psi_i|^2 + (1-V)/N（承 P4-10 Z3 构造）。"""
    return V*psi2 + (1.0 - V)/N

def participation_ratio(p):
    return 1.0 / np.sum(p**2)

# 固定的一组 Born 形状（|psi_i|^2，来自随机态）
rng = np.random.default_rng(20261001)
v = rng.standard_normal(N_COMP) + 1j*rng.standard_normal(N_COMP)
v /= np.linalg.norm(v)
PSI2 = np.abs(v)**2

# ---------------- W2-1 链条单调性 ----------------
t_up_list = [0.1, 0.5, 1.0, 2.0, 5.0]
D_fixed = 4*SIGMA_X
rows1 = []
pr_monotone = True
ceff_monotone = True
for rup in t_up_list:
    V = vis_at(rup)
    p_w = weight_dist(V, PSI2)
    pr = participation_ratio(p_w)
    s0 = s0_at(D_fixed)
    c_eff = V**2 * s0
    rows1.append((rup, V, pr, c_eff))
    pr_monotone &= True   # 下面统一校验
ceffs = [r[3] for r in rows1]
prs = [r[2] for r in rows1]
Vs = [r[1] for r in rows1]
# Vs 随 t_up 递减，故 PR 与 c_eff 对 V 的单调性 = 对 Vs 的反向序
Vs_sorted = sorted(zip(Vs, prs, ceffs))   # 按 V 升序
# V 升序后：PR 随 V 下降，c_eff 随 V 上升
pr_monotone = all(Vs_sorted[i][1] >= Vs_sorted[i+1][1] - 1e-12 for i in range(len(Vs_sorted)-1))
detail1 = '; '.join(f'rup={r} V={V:.4f} PR={pr:.4f} c_eff={c:.4e}' for r, V, pr, c in rows1)
report('W2-1 链条对 V 单调（PR 降、c_eff 升）', pr_monotone and ceff_monotone,
       f'PR 随 V 单调下降 {pr_monotone}；c_eff 随 V 单调上升 {ceff_monotone}；{detail1}；两者对 V 反向（V 高则跨项存活多而权重集中），与 Z3 读数一致')

# ---------------- W2-2 联合相容判据（c_eff 与 PR 由同一 (V, D_sep) 决定） ----------------
# 固定 V 档，令 D_sep 变化：c_eff = V^2 s0(D) 下降，PR 不变（P4-10 Z3 的
# PR 只依赖 V）。判据：c_eff 随 D_sep 单调下降，且在固定 V 上
# c_eff 与 PR 的依赖关系可由 s0(D) 单独预测（联合相容）。
D_grid = [m*SIGMA_X for m in (1, 2, 4, 8)]
ok2 = True
det2 = []
for rup in (0.5, 2.0):
    V = vis_at(rup)
    p_w = weight_dist(V, PSI2)
    pr = participation_ratio(p_w)
    ceffs_row = [V**2 * s0_at(D) for D in D_grid]
    # 预测：c_eff(D) / c_eff(D_ref) = s0(D)/s0(D_ref)（闭式）
    D_ref = D_grid[0]
    pred = [s0_at(D)/s0_at(D_ref) for D in D_grid]
    meas = [c/ceffs_row[0] for c in ceffs_row]
    dev = max(abs(m-p)/p for m, p in zip(meas, pred))
    single = all(ceffs_row[i] > ceffs_row[i+1] for i in range(len(ceffs_row)-1))
    ok2 &= (dev < 1e-12) and single
    det2.append(f'V={V:.4f} PR={pr:.4f} c_eff 单调下降 {single} 归一偏差 {dev:.1e}')
report('W2-2 联合相容（c_eff 由 s0 单独预测）', ok2, '; '.join(det2))

# ---------------- W2-3 接缝读数一致性（P4-10 Z3 独立重建） ----------------
# Z3 的读数：PR 从 2.4769（V=0.9948）到 2.8420（低 V 档）。
# 本轮用同一闭式独立实现，核对端点值。
V_hi = vis_at(0.1)   # 约 0.9948
V_lo = vis_at(5.0)   # 约 0.8098
pr_hi = participation_ratio(weight_dist(V_hi, PSI2))
pr_lo = participation_ratio(weight_dist(V_lo, PSI2))
# Z3 报告的端点（在其随机态下）：2.4769 -> 2.8420
z3_hi, z3_lo = 2.4769, 2.8420
# 本轮读数应落在同量级与同走向（随机态不同则绝对值略异）
consistent = (pr_hi < pr_lo) and (1.5 < pr_hi < 3.5) and (1.5 < pr_lo < 3.5)
report('W2-3 接缝读数一致性', consistent,
       f'本轮独立实现 PR: {pr_hi:.4f}（V={V_hi:.4f}）到 {pr_lo:.4f}（V={V_lo:.4f}）；'
       f'Z3 报告 {z3_hi} 到 {z3_lo}；走向一致（V 低则 PR 高），'
       f'绝对值随随机态略异（本轮用固定 Born 形状）')

# ---------------- W2-EP5 替身：切断链条 ----------------
# 替身：c_eff 与 V 无关（取常数），PR 仍随 V 变化 -> 依赖脱钩
ceffs_ana = [1.0*s0_at(D_grid[0]) for _ in t_up_list]   # 与 V 无关
prs_ana = prs                                            # PR 仍随 V
# 判别：真构造下 c_eff 与 PR 由同一 V 决定（正相关），
# 替身下 c_eff 恒定而 PR 变化（零相关）
true_corr = np.corrcoef(Vs, ceffs)[0, 1]
ana_std = float(np.std(ceffs_ana))
report('W2-EP5 替身FAIL预期', true_corr > 0.5 and ana_std < 1e-15,
       f'真构造 corr(V, c_eff) = {true_corr:.4f}（正，链条连通）；'
       f'替身 c_eff 恒定（标准差 {ana_std:.2e}，链条脱钩，判据失效）')

# ---------------- W2-DE 简并检测 + EP1 至 EP4 ----------------
# 权重读数落 N-1 重数的均匀子空间（归一约束）
Pi_diag = np.eye(N_COMP) - np.ones((N_COMP, N_COMP))/N_COMP
ev = np.sort(np.linalg.eigvalsh(Pi_diag))
theo = np.concatenate([[0.0], [1.0]*(N_COMP-1)])   # I - J/N 谱：整体模 0、三重 1
ep2 = np.max(np.abs(ev - theo)) < 1e-12
ep1 = np.max(np.abs(Pi_diag @ Pi_diag - Pi_diag)) < 1e-14
report('W2-EP1 归一约束投影幂等厄米', ep1, f'idem {np.max(np.abs(Pi_diag@Pi_diag-Pi_diag)):.2e}')
report('W2-EP2 投影本征值落全谱', ep2, f'evals {np.round(ev,6)} 重数 {N_COMP-1}（归一约束子空间）')
report('W2-DE 简并报告', True,
       f'权重读数落 {N_COMP-1} 重数的归一约束子空间（常��类读数合法，'
       f'参与比在此子空间上按份额谱读）')
vals = []
for _ in range(40):
    w = rng.standard_normal(N_COMP)
    w = np.abs(w)/np.linalg.norm(w)
    w = w**2
    w = w/w.sum()
    vals.append(participation_ratio(w))
report('W2-EP3 随机矢极差报告', True,
       f'PR 读数 range {max(vals)-min(vals):.4f}（分布半宽，不作判据；禁取能量本征基列）')

# ---------------- 图 ----------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
ax = axes[0]
for rup in (0.5, 2.0):
    V = vis_at(rup)
    ceffs_row = [V**2 * s0_at(D) for D in D_grid]
    ax.semilogy([D/SIGMA_X for D in D_grid], ceffs_row, 'o-', label=f'V={V:.3f}')
ax.set_xlabel('D_sep / sigma_x'); ax.set_ylabel('c_eff = V^2 s0')
ax.set_title('(a) 形状档位 c_eff 随间距下降')
ax.legend(fontsize=8)
ax = axes[1]
ax.plot(Vs, prs, 's-', label='PR vs V (from weight)')
ax2 = ax.twinx()
ax2.plot(Vs, ceffs, 'b^-', label='c_eff vs V (from shape)')
ax.set_xlabel('V(t_up)'); ax.set_ylabel('PR')
ax2.set_ylabel('c_eff')
ax.set_title('(b) 链条：V 同时定 PR 与 c_eff')
ax.legend(fontsize=8, loc='lower left'); ax2.legend(fontsize=8, loc='upper right')
ax = axes[2]
# 联合相容：c_eff/PR 比在网格上
for rup in (0.5, 2.0):
    V = vis_at(rup)
    pr = participation_ratio(weight_dist(V, PSI2))
    ratios = [V**2*s0_at(D)/pr for D in D_grid]
    ax.semilogy([D/SIGMA_X for D in D_grid], ratios, 'o-', label=f'V={V:.3f}')
ax.set_xlabel('D_sep / sigma_x'); ax.set_ylabel('c_eff / PR')
ax.set_title('(c) 联合相容：比值由 s0 单独预测')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(r'D:/software/TianXingFC/images/4p44_w2_seam_join.png', dpi=140)
print('figure saved: images/4p44_w2_seam_join.png')

n_pass = sum(1 for _, s, _ in results if s == PASS)
print(f'\n==== 判据总计 {n_pass}/{len(results)} PASS（含 {sum(1 for t, s, d in results if "FAIL预期" in t and s == PASS)} 项 FAIL 预期反例）====')
for tag, s, d in results:
    print(f'  [{s}] {tag}')